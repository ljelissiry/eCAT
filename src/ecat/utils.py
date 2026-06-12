"""Shared numerical, unit, and display helpers for eCAT internals."""

import numpy as np
import pandas as pd
import glob
import os
import re
import warnings
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
import scipy
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter
from scipy.signal import find_peaks, peak_prominences
from sklearn.metrics import r2_score
from contextlib import contextmanager
from datetime import datetime
from dataclasses import fields, replace
from copy import deepcopy
from numbers import Real

from ._version import __version__
from ._plot_style import _active_plot_style_value, plotting_style
from .metadata import (
    concentration_to_float,
    format_chemical_formulas,
    get_file_times,
    parse_concentration_value_and_unit,
    parse_concentration_value_and_unit as _parse_concentration_value_and_unit,
)
from .options import (
    FitPeakPotentialOptions,
    FilterOptions,
    FOWAOptions,
    FitPeakCurrentOptions,
    FitRateOptions,
    GroupSummaryOptions,
    ImportOptions,
    MultiMultiplotOptions,
    MultiScatterplotOptions,
    MultiplotOptions,
    NicholsonOptions,
    NormalizeOptions,
    NormalizationOptions,
    OptionError,
    PeakCurrentOptions,
    PeakPotentialOptions,
    PlateauCurrentOptions,
    PlotOptions,
    SevcikAnalysisOptions,
    ScaleCurrentOptions,
    SortGroupOptions,
    TafelAnalysisOptions,
    TrimOptions,
    TrumpetAnalysisOptions,
    describe_options as _describe_options,
    get_defaults,
    load_defaults,
    normalize_key,
    reset_defaults,
    reset_defaults_option,
    reset_defaults_section,
    set_defaults,
)


def _format_path_for_display(path, relative_to=None, mode="auto"):
    """
    Return a compact path string for user-facing messages.

    This is intentionally display-only: callers should continue storing and
    passing the original absolute/resolved paths for loading and exporting.
    """
    if path in (None, ""):
        return "" if path is None else str(path)

    if mode not in {"auto", "absolute", "name"}:
        raise ValueError("mode must be 'auto', 'absolute', or 'name'.")

    path_text = os.fspath(path)
    if mode == "name":
        return os.path.basename(path_text)

    expanded = os.path.expanduser(path_text)
    path_abs = os.path.abspath(expanded)
    if mode == "absolute":
        return path_abs

    base = _default_path_display_base() if relative_to is None else os.fspath(relative_to)
    base_abs = os.path.abspath(os.path.expanduser(base))
    try:
        common = os.path.commonpath([path_abs, base_abs])
    except ValueError:
        return path_abs

    if common == base_abs:
        rel = os.path.relpath(path_abs, base_abs)
        return "." if rel == "." else rel
    return path_abs


def _default_path_display_base():
    cwd = os.path.abspath(os.getcwd())
    current = cwd
    while True:
        if (
            os.path.isfile(os.path.join(current, "pyproject.toml"))
            or os.path.isdir(os.path.join(current, ".git"))
            or os.path.isfile(os.path.join(current, ".git"))
        ):
            return current
        parent = os.path.dirname(current)
        if parent == current:
            return cwd
        current = parent
from .parsers import (
    exp_type_short as _exp_type_short,
    parse_ch_timestamp as _parse_ch_timestamp,
    parse_duration_seconds as _parse_duration_seconds,
    parse_quiet_time_from_lines as _parse_quiet_time_from_lines,
)

try:
    from IPython.display import display, Math
except ImportError:
    display = None
    Math = None


def describe_options(option_model_or_section=None, options=None):
    """Describe available eCAT options for a public workflow or options section.
    
    Parameters
    ----------
    option_model_or_section : str or option model, optional
        Public function name, options section name, or options dataclass to describe.
    options : dict, optional
        Display options for the options table. See ``e.describe_options("describe_options")``.
    
    Returns
    -------
    pandas.DataFrame or None
        Option menu/table when dataframe return is requested; otherwise
        displays the table unless suppressed and returns None.
    
    Examples
    --------
    >>> e.describe_options("multiplot")
    """
    display_options = {} if options is None else dict(options)
    display_options.setdefault("display function", display)
    return _describe_options(
        option_model_or_section,
        options=display_options,
    )


def _plot_legend_option_enabled(value):
    if isinstance(value, str):
        return value.strip().lower() not in {"", "false", "none", "off", "0"}
    return bool(value)


def resolve_electrode_area_option(options):
    """Return the effective electrode area from import options."""
    area = options.get("electrode area", 0)
    diameter = options.get("electrode diameter", 0)
    area_provided = bool(options.get("_electrode area provided", False))

    if area is None:
        area = 0
        area_provided = False
    if diameter is None:
        diameter = 0

    if diameter != 0 and area == 0 and not area_provided:
        return np.pi * (diameter / 2) ** 2
    return area


def _plot_legend_requested(options, ax):
    value = options.get("legend", False)
    if isinstance(value, str) and value.strip().lower() == "auto":
        _handles, labels = ax.get_legend_handles_labels()
        visible = [
            label for label in labels
            if label not in (None, "") and not str(label).startswith("_")
        ]
        return len(visible) > 1
    return _plot_legend_option_enabled(value)

###===================###
### Global Variables  ###
###===================###

# Constants
F = 96485.33212331 # Faraday's constant in C/mol
R = 8.31446261815324 # Gas constant in V C/(mol K)

plotting_style(True)


###===================###
###      Classes      ###
###===================###

def _best_datetime_for_sort(*values):
    for value in values:
        if isinstance(value, datetime):
            return value
    return None


def _object_time_for_sort(echem_object):
    return _best_datetime_for_sort(
        getattr(echem_object, "timestamp", None),
        getattr(echem_object, "creation_time", None),
        getattr(echem_object, "modification_time", None),
    )



# Moved to focused module; imported near the end of this file.
@contextmanager
def _temporary_figure_dpi(fig, dpi):
    old_dpi = fig.get_dpi()
    try:
        fig.set_dpi(dpi)
        yield
    finally:
        fig.set_dpi(old_dpi)

BASE_UNITS = {
    's','min','h','day',       # time
    'V','A','C','F','Hz','Ω','ohm',  # electrical
    'M', '%', 'equiv', 'x', # concentration / dimensionless composition
    # add more as needed...
}
SI_PREFIX_EXPONENTS = {
    'Y': 24,  'Z': 21,  'E': 18,  'P': 15,  'T': 12,
    'G': 9,   'M': 6,   'k': 3,   'h': 2,   'da': 1,
    'd': -1,  'c': -2,  'm': -3,  'μ': -6,  'u': -6,
    'n': -9,  'p': -12, 'f': -15, 'a': -18, 'z': -21,
    'y': -24,
}
SI_PREFIXES = SI_PREFIX_EXPONENTS.keys()

def get_column_name_case_insensitive(requested_name, columns):
    """
    Return the actual column name from a list of columns, case-insensitively.

    Parameters:
        requested_name (str): The user-requested column name.
        columns (Iterable[str]): The list or index of available column names.

    Returns:
        str: The actual column name matching the request (with correct case).

    Raises:
        ValueError: If no match is found.
    """
    column_map = {col.lower(): col for col in columns}
    lookup = requested_name.lower()

    if lookup not in column_map:
        raise ValueError(f"Column '{requested_name}' not found (case-insensitive).")

    return column_map[lookup]

def _resolve_savgol_params(npts, options, deriv=0):
    """
    Resolve Savitzky-Golay window and polyorder from options.

    Accepts:
      - noise window: int or "auto"
      - noise polyorder: int or "auto"
    """
    if npts < max(deriv + 3, 5):
        return None, None

    window = options.get("noise window", "auto")
    polyorder = options.get("noise polyorder", "auto")

    # ---- auto window ----
    if window in (None, "auto"):
        # ~5% of region length, clamped, odd
        window = int(round(0.05 * npts))
        window = max(7, min(window, 51))

        if window % 2 == 0:
            window += 1

        max_valid = npts if npts % 2 == 1 else npts - 1
        window = min(window, max_valid)

        min_valid = max(deriv + 3, 5)
        if min_valid % 2 == 0:
            min_valid += 1
        window = max(window, min_valid)

        if window > max_valid:
            window = max_valid

    else:
        window = int(window)
        if window % 2 == 0:
            window += 1

        max_valid = npts if npts % 2 == 1 else npts - 1
        window = min(window, max_valid)

    # ---- auto polyorder ----
    if polyorder in (None, "auto"):
        polyorder = 3 if window >= 9 else 2
    else:
        polyorder = int(polyorder)

    polyorder = max(deriv, min(polyorder, window - 1))

    if window < 5 or polyorder >= window:
        return None, None

    return window, polyorder


def _savgol_apply(y, options, deriv=0, delta=1.0):
    """
    Safe SG wrapper. Falls back to raw data if the region is too short.
    """
    y = np.asarray(y, dtype=float)
    window, polyorder = _resolve_savgol_params(len(y), options, deriv=deriv)

    if window is None:
        return y.copy(), {"window": None, "polyorder": None}

    out = savgol_filter(
        y,
        window_length=window,
        polyorder=polyorder,
        deriv=deriv,
        delta=delta,
    )
    return out, {"window": window, "polyorder": polyorder}


def _savgol_bundle(y, options, delta=1.0):
    """
    Return smoothed signal + first and second derivatives,
    all using the same resolved SG parameters.
    """
    y = np.asarray(y, dtype=float)

    smooth, meta = _savgol_apply(y, options, deriv=0, delta=delta)

    # use same resolved params for all derivatives
    window = meta["window"]
    polyorder = meta["polyorder"]

    if window is None:
        return smooth, y.copy(), np.zeros_like(y), meta

    d1 = savgol_filter(
        y,
        window_length=window,
        polyorder=polyorder,
        deriv=1,
        delta=delta,
    )
    d2 = savgol_filter(
        y,
        window_length=window,
        polyorder=polyorder,
        deriv=2,
        delta=delta,
    )

    return smooth, d1, d2, meta

def get_conversion_factor(unit_str, to_unit_str=None):
    """
    Returns the conversion factor to convert a value from `unit_str` to the base unit.
    If `to_unit_str` is provided, returns the conversion factor from `unit_str` to `to_unit_str`.

    Examples:
        get_conversion_factor("mA") -> 1e-3
        get_conversion_factor("mA", "uA") -> 1e3
    """
    from_exp = SI_PREFIX_EXPONENTS.get(extract_prefix_and_base(unit_str)[0], 0)
    if to_unit_str is None:
        return 10 ** from_exp

    to_exp = SI_PREFIX_EXPONENTS.get(extract_prefix_and_base(to_unit_str)[0], 0)
    return 10 ** (from_exp - to_exp)

def extract_prefix_and_base(unit_str: str) -> tuple[str, str]:
    """
    Split an SI-prefixed unit string into (prefix, base_unit).

    Examples
    --------
    >>> extract_prefix_and_base("mA")
    ('m', 'A')
    >>> extract_prefix_and_base("μs")
    ('μ', 's')
    >>> extract_prefix_and_base("min")
    ('', 'min')
    >>> extract_prefix_and_base("V vs Fc")
    ('', 'V vs Fc')
    >>> extract_prefix_and_base("A")          # single character
    ('', 'A')

    Rules
    -----
    1.  If the text contains whitespace, “/”, or “vs”, we treat it as a
        compound expression and return it unchanged.
    2.  Any ASCII “u” used for micro is normalised to “μ”.
    3.  If the whole string (after normalisation) is in `BASE_UNITS`,
        it has no prefix.
    4.  If the string is a **single printable character**, we assume
        it is itself the base unit.
    5.  Otherwise we try each prefix (longest → shortest) and split on
        the first that leaves a recognised base unit.
    """
    unit_str = unit_str.strip()

    # 1.  Single-character input → treat as base
    if len(unit_str) == 1:
        return "", unit_str

    # 2.  Compound units → leave intact
    lowered = unit_str.lower()
    if " " in unit_str or "/" in unit_str or "vs" in lowered:
        return "", unit_str

    # 3.  Normalise ASCII 'u' to Greek 'μ'
    if "u" in unit_str:
        unit_str = unit_str.replace("u", "μ")

    # 4.  Explicit base unit
    if unit_str in BASE_UNITS:
        return "", unit_str

    # 5.  Try SI prefixes (check longer ones first: 'da' before 'd')
    for prefix in sorted(SI_PREFIXES, key=len, reverse=True):
        if unit_str.startswith(prefix):
            base = unit_str[len(prefix):]
            if base in BASE_UNITS:
                return prefix, base

    # Fall-back: nothing matched
    return "", unit_str

def _estimate_peak_prominence(signal, options):
    """
    Shared auto prominence estimate.
    Uses user override if provided, otherwise estimates from point-to-point noise.
    """
    prominence = options.get("peak prominence")
    if prominence is not None:
        return prominence

    signal = np.asarray(signal, dtype=float)
    if len(signal) < 3:
        return 0.0

    diffs = np.diff(signal)
    noise_std = np.std(diffs) / np.sqrt(2) if len(diffs) > 0 else 0.0
    return 5 * noise_std

def _filter_extrema_by_curvature(extrema, a, options):
    """
    Keep extrema that have enough local curvature to plausibly represent
    a real feature boundary.

    This is sign-agnostic:
    - maxima and minima are both allowed
    - filtering uses |acceleration| at the extremum
    """
    extrema = np.asarray(extrema, dtype=int)
    if len(extrema) == 0:
        return extrema, {
            "curvature cutoff": None,
            "num extrema": 0,
            "num relevant": 0,
        }

    abs_a = np.abs(np.asarray(a, dtype=float))
    a_ext = abs_a[extrema]

    # Use a high-percentile reference from the local acceleration trace.
    # A raw 5th percentile is usually too close to zero because most baseline
    # regions have very low acceleration.
    ref_pct = options.get("tangent curvature reference percentile", 95)
    curv_ref = np.nanpercentile(abs_a, ref_pct)

    if not np.isfinite(curv_ref) or curv_ref == 0:
        return extrema, {
            "curvature cutoff": 0,
            "num extrema": int(len(extrema)),
            "num relevant": int(len(extrema)),
        }

    # Require extrema to have at least a small fraction of the local high curvature.
    min_frac = options.get("tangent extremum min curvature fraction", 0.05)
    cutoff = min_frac * curv_ref

    relevant = extrema[a_ext >= cutoff]

    return relevant, {
        "curvature cutoff": cutoff,
        "curvature reference": curv_ref,
        "curvature reference percentile": ref_pct,
        "min curvature fraction": min_frac,
        "num extrema": int(len(extrema)),
        "num relevant": int(len(relevant)),
    }

def _classify_extremum_kind(y_smooth, idx, window=3):
    """
    Classify a point as a local maximum or minimum using a smoothed signal.

    Returns
    -------
    str or None
        "max", "min", or None if the point is not clearly an extremum.
    """
    y_smooth = np.asarray(y_smooth, dtype=float)
    idx = int(idx)

    if idx < 0 or idx >= len(y_smooth):
        return None

    lo = max(0, idx - window)
    hi = min(len(y_smooth), idx + window + 1)

    local = y_smooth[lo:hi]
    center = y_smooth[idx]

    if len(local) < 3 or not np.isfinite(center):
        return None

    if center >= np.nanmax(local):
        return "max"

    if center <= np.nanmin(local):
        return "min"

    return None

def _find_extrema_indices(y, options):
    """
    Return all local extrema (maxima + minima) in the smoothed signal,
    sorted by index, along with smoothed_y and a prominence map.
    """
    y = np.asarray(y, dtype=float)
    smoothed_y, meta = _savgol_apply(y, options, deriv=0)

    prominence = _estimate_peak_prominence(smoothed_y, options)

    maxima, max_props = find_peaks(smoothed_y, prominence=prominence)
    minima, min_props = find_peaks(-smoothed_y, prominence=prominence)

    extrema = np.sort(np.concatenate([maxima, minima]))

    prom_map = {}
    for idx, prom in zip(maxima, max_props.get("prominences", [])):
        prom_map[int(idx)] = float(prom)
    for idx, prom in zip(minima, min_props.get("prominences", [])):
        prom_map[int(idx)] = float(prom)

    extrema_kind_map = {}

    for idx in maxima:
        extrema_kind_map[int(idx)] = "max"

    for idx in minima:
        extrema_kind_map[int(idx)] = "min"

    return extrema, smoothed_y, prom_map, {
        "prominence": prominence,
        "sg window": meta["window"],
        "sg polyorder": meta["polyorder"],
        "maxima": maxima,
        "minima": minima,
        "extrema kind map": extrema_kind_map,
    }

def _scale_trace_to_match_current(trace, current_display):
    trace = np.asarray(trace, dtype=float)
    current_display = np.asarray(current_display, dtype=float)

    trace_max = np.nanmax(np.abs(trace))
    current_max = np.nanmax(np.abs(current_display))

    if trace_max == 0 or not np.isfinite(trace_max):
        return np.zeros_like(trace)

    if current_max == 0 or not np.isfinite(current_max):
        return np.zeros_like(trace)

    return trace * (current_max / trace_max)

def scale_value(val, unit, selected_unit='auto', candidates=('k', 'm', 'μ', 'n', 'p')):
    """
    Scale a single numeric value `val` whose unit is something like
    'V/s', 'mA', 's', etc.  If `selected_unit=='auto'` it picks the largest
    SI prefix so that |scaled_val| >= 1.  Otherwise, it honors your
    explicit prefix.  Returns (scaled_val, full_unit_str).
    """
    # 1) split numerator/denominator (e.g. 'V/s' → 'V', 's')
    if '/' in unit:
        num_unit, denom_unit = unit.split('/', 1)
    else:
        num_unit, denom_unit = unit, ''

    # 2) peel off any existing prefix from numerator
    prefix0, base = extract_prefix_and_base(num_unit)

    # 3) convert val into the true base‐unit quantity
    factor0 = get_conversion_factor(prefix0 + base)
    base_val = val * factor0

    # 4) choose new prefix
    if selected_unit == 'auto':
        m = abs(base_val)
        for p in candidates:
            f = get_conversion_factor(p + base)
            if m / f >= 1 and m / f < 10**3:
                new_val = base_val / f
                new_unit = p + base
                break
        else:
            new_val = base_val
            new_unit = base
    elif selected_unit is not None:
        selected_prefix, selected_base = extract_prefix_and_base(selected_unit)
        if selected_base != base:
            raise ValueError(f"Cannot convert {base} to {selected_base} (incompatible units)")
        # explicit prefix
        f = get_conversion_factor(selected_prefix + base)
        new_val = base_val / f
        new_unit = selected_prefix + base
    else:
        new_val = val
        new_unit = unit

    # 5) reattach denominator
    if denom_unit:
        new_unit = f"{new_unit}/{denom_unit}"

    return new_val, new_unit


def replace_keyword(name, keyword):
    # Use regular expression to find the text after "- " and before the underscore
    match = re.search(r'- (.*?)_', name)

    if match:
        replacement_text = match.group(1)
        # Replace the specified keyword with the found text
        new_name = name.replace(keyword, replacement_text)
        return new_name
    else:
        # Keyword not found, return the original name
        return name

def apply_text_alterations(text, alterations):
    """
    Apply standardized text alterations.

    Accepted forms
    --------------
    None
    dict[str, str]
    tuple[str, str]
    list[tuple[str, str]]
    callable
    """
    if text is None:
        return text
    text = str(text)

    if alterations is None:
        return text

    if callable(alterations):
        return alterations(text)

    if isinstance(alterations, dict):
        alterations = list(alterations.items())
    elif isinstance(alterations, tuple) and len(alterations) == 2:
        alterations = [alterations]

    for old, new in alterations:
        text = text.replace(old, new)

    return text

def animate(plot,rate=1,fps=5,minrate=0,repeat=False):
    fig = plt.gcf()
    ax = plt.gca()
    x = plot.get_xdata()
    y = plot.get_ydata()
    
    delta = np.abs(np.diff(x)[0])
    length = len(x) * delta
    time = length / rate
    frames = int(fps * time)
    interval = int(np.ceil(1000 / fps))
    
    def update(frame):
        # for each frame, update the data stored on each artist.
        index = int(np.ceil(frame / frames * (len(x)-1)))
        a = x[:index]
        b = y[:index]
        # update the line plot:
        plot.set_xdata(a)
        plot.set_ydata(b)
        return (plot)

    frame_extention = 0
    if minrate != 0:
        time_minrate = length / minrate
        frames_minrate = int(fps * time_minrate)
        frame_extention = frames_minrate - frames

    ani = animation.FuncAnimation(
        fig=fig,
        func=update,
        frames=frames+frame_extention,
        interval=interval,
        repeat=repeat
    )
    return ani

def round_sigfigs(number, sigfigs):
    if number == 0:
        return 0
    return round(number, sigfigs - 1 - int(np.floor(np.log10(abs(number)))))

def count_segments(x_values):
    if len(x_values) <= 1:
        return 1  # If there's only one or zero data points, return 1 segment

    # Calculate the when the sign of the differences between consecutive x-values changes
    differences_sign = np.diff(np.sign(np.diff(x_values)))

    # Count the number of times the sign changes (i.e., change in direction)
    num_segments = np.sum(differences_sign != 0) + 1

    return num_segments

_AREA_UNIT_TO_M = {
    "m": 1.0,
    "cm": 1e-2,
    "mm": 1e-3,
    "um": 1e-6,
    "μm": 1e-6,
    "nm": 1e-9,
}


def _normalize_current_unit_text(unit):
    unit = str(unit).strip()
    return unit.replace("u", "μ") if unit.startswith("u") else unit


def _normalize_area_unit_text(unit):
    unit = str(unit).strip()
    unit = unit.replace("μ", "u")
    unit = unit.replace("$", "")
    unit = unit.replace("{", "").replace("}", "")
    unit = unit.replace("²", "2")
    unit = unit.replace("^2", "")
    if unit.endswith("2"):
        unit = unit[:-1]
    return unit


def _format_area_squared_unit(area_unit):
    display = "μm" if area_unit in {"um", "μm"} else area_unit
    return f"{display}$^2$"


def _parse_current_density_unit(unit):
    if unit in (None, ""):
        return None
    text = str(unit).strip()
    if "/" not in text:
        return None

    numerator, denominator = text.split("/", 1)
    numerator = _normalize_current_unit_text(numerator)
    area_unit = _normalize_area_unit_text(denominator)
    prefix, base = extract_prefix_and_base(numerator)

    if base != "A" or area_unit not in _AREA_UNIT_TO_M:
        return None
    return prefix + base, area_unit


def _parse_current_density_selected_unit(selected_unit, default_area_unit):
    if selected_unit in (None, "auto"):
        return None, default_area_unit, True

    text = str(selected_unit).strip()
    if text == "":
        return None, default_area_unit, True

    if "/" in text:
        numerator, denominator = text.split("/", 1)
        if numerator.strip() == "":
            area_unit = _normalize_area_unit_text(denominator)
            if area_unit in _AREA_UNIT_TO_M:
                return None, area_unit, True
        numerator = _normalize_current_unit_text(numerator)
        area_unit = _normalize_area_unit_text(denominator)
        prefix, base = extract_prefix_and_base(numerator)
        if base != "A" or area_unit not in _AREA_UNIT_TO_M:
            raise ValueError(f"Cannot convert current density to {selected_unit} (incompatible units)")
        return prefix + base, area_unit, False

    area_unit = _normalize_area_unit_text(text)
    if area_unit in _AREA_UNIT_TO_M:
        return None, area_unit, True

    current_unit = _normalize_current_unit_text(text)
    prefix, base = extract_prefix_and_base(current_unit)
    if base == "A":
        return prefix + base, default_area_unit, False

    raise ValueError(f"Cannot convert current density to {selected_unit} (incompatible units)")


def _area_density_conversion_factor(from_area_unit, to_area_unit):
    return (_AREA_UNIT_TO_M[to_area_unit] / _AREA_UNIT_TO_M[from_area_unit]) ** 2


def _scale_current_density_axis(y_base, current_unit, selected_unit='auto', candidates=('m', 'μ', 'n', 'p')):
    parsed = _parse_current_density_unit(current_unit)
    if parsed is None:
        return None

    source_current_unit, source_area_unit = parsed
    target_current_unit, target_area_unit, auto_current = _parse_current_density_selected_unit(
        selected_unit,
        source_area_unit,
    )

    source_prefix, current_base = extract_prefix_and_base(source_current_unit)
    source_current_factor = get_conversion_factor(source_prefix + current_base)
    area_factor = _area_density_conversion_factor(source_area_unit, target_area_unit)
    y_in_base_target_area = y_base * source_current_factor * area_factor

    if auto_current:
        abs_max = max(abs(np.nanmin(y_in_base_target_area)), abs(np.nanmax(y_in_base_target_area)))
        target_prefix = ""
        for prefix in (candidates or SI_PREFIXES):
            factor = get_conversion_factor(prefix + current_base)
            if abs_max / factor >= 1:
                target_prefix = prefix
                break
        target_current_unit = target_prefix + current_base

    target_current_unit = _normalize_current_unit_text(target_current_unit)
    target_factor = get_conversion_factor(target_current_unit)
    display_unit = f"{target_current_unit}/{_format_area_squared_unit(target_area_unit)}"
    return source_current_factor * area_factor / target_factor, display_unit


def scale_axis(y_base, current_unit, selected_unit='auto', candidates=('m', 'μ', 'n', 'p')):
    """
    Given a base-unit y-array, determine the best scaling factor and unit prefix.
    Returns (scale_factor, unit_label), where y_scaled = y_base * scale_factor.
    """
    density_scaled = _scale_current_density_axis(
        y_base,
        current_unit,
        selected_unit=selected_unit,
        candidates=candidates,
    )
    if density_scaled is not None:
        return density_scaled

    if selected_unit not in (None, "auto") and "/" in str(selected_unit):
        raise ValueError(
            f"Cannot convert {current_unit} to {selected_unit} (incompatible units)"
        )

    # peel off whatever prefix was on the raw unit:
    prefix, base = extract_prefix_and_base(current_unit)

    # convert y_base into the true base-unit values:
    factor_base = get_conversion_factor(prefix + base)
    y_in_base = y_base * factor_base

    if selected_unit == 'auto':
        abs_max = max(abs(np.nanmin(y_in_base)), abs(np.nanmax(y_in_base)))
        for p in (candidates or SI_PREFIXES):
            f = get_conversion_factor(p + base)
            if abs_max / f >= 1:
                return 1 / f, p + base
        return 1.0, base
    else:
        # explicit
        f = get_conversion_factor(selected_unit)
        return 1 / f, selected_unit

def scale_time_axis(x, base_unit, selected_unit='auto'):
    """
    Given x in base_unit (e.g. 's', 'min', 'h', 'day'), choose the best
    scale (seconds→minutes→hours→days) or honor an explicit unit.
    Returns (scale_factor, unit_label), where x_scaled = x * scale_factor.
    """
    # map each unit to how many seconds it represents
    conversions = {
        's': 1,
        'min': 60,
        'h': 3600,
        'day': 86400,
    }
    # allowed targets in order of “readability”
    candidates = ['day', 'h', 'min', 's']

    # normalize everything to seconds
    factor_base = conversions.get(base_unit, 1)
    x_in_sec = x * factor_base

    if selected_unit == 'auto':
        abs_max = np.nanmax(np.abs(x_in_sec))
        for u in candidates:
            factor = conversions[u]
            if abs_max / factor >= 10:
                # to go FROM seconds INTO u-units, multiply by 1/factor
                return 1 / factor, u
        return 1.0, base_unit

    elif selected_unit in conversions:
        return 1 / conversions[selected_unit], selected_unit

    else:
        # unknown explicit → no scaling
        return 1.0, base_unit

def _legacy_normalize_option_keys(options):
    return {
        normalize_key(key).replace("_", " "): value
        for key, value in (options or {}).items()
    }

def _cv_trim_window_mode(options):
    text = str(options.get("mode", "expand")).strip().lower().replace("_", " ").replace("-", " ")
    if text not in {"expand", "pointwise", "strict"}:
        raise ValueError("'mode' must be 'expand', 'pointwise', or 'strict'.")
    return text

def _cv_trim_window_info(E, potential_window, options):
    E = np.asarray(E, dtype=float)
    requested = [float(potential_window[0]), float(potential_window[1])]
    lo, hi = sorted(requested)
    mask = (E >= lo) & (E <= hi)
    selected = np.flatnonzero(mask)
    mode = _cv_trim_window_mode(options)
    if len(selected) == 0:
        return {
            "mask": mask,
            "requested": requested,
            "effective": None,
            "mode": mode,
            "expanded": False,
            "break_count": 0,
        }

    breaks = np.where(np.diff(selected) > 1)[0]
    break_count = int(len(breaks))
    expanded = False
    if break_count:
        if mode == "strict":
            raise ValueError(
                "The requested potential window would disconnect the CV scan history. "
                "Use mode='expand' to preserve a connected waveform, or "
                "mode='pointwise' to keep the pointwise trim."
            )
        if mode == "expand":
            connected_mask = np.zeros(len(E), dtype=bool)
            connected_mask[selected[0] : selected[-1] + 1] = True
            mask = connected_mask
            expanded = True

    effective = [float(np.nanmin(E[mask])), float(np.nanmax(E[mask]))] if np.any(mask) else None
    return {
        "mask": mask,
        "requested": requested,
        "effective": effective,
        "mode": mode,
        "expanded": expanded,
        "break_count": break_count,
    }


def _select_fit_indices(*args, **kwargs):
    from .analysis_batch import _select_fit_indices as impl
    return impl(*args, **kwargs)


def _scatterfit_legend_fontsize(*args, **kwargs):
    from .plotting import _scatterfit_legend_fontsize as impl
    return impl(*args, **kwargs)


def _fit_color_value(options, index=0, fallback="tab:red"):
    value = (options or {}).get("fit colors")
    if value is None:
        value = (options or {}).get("fit color")
    if value is None:
        return fallback
    if not isinstance(value, str):
        try:
            if mpl.colors.is_color_like(value):
                return value
        except Exception:
            pass
        if isinstance(value, (list, tuple, np.ndarray, pd.Series)):
            values = list(value)
            if not values:
                return fallback
            if index < len(values):
                return values[index]
            return values[-1]
    return value


def fit(x, y, label="", degree=1, plot_fit=True, options=None):
    if options is None:
        options = {}
    else:
        options = _legacy_normalize_option_keys(options)

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    fit_indices = options.get("fit indices")
    x_fit, y_fit = _select_fit_indices(x, y, fit_indices)

    if len(x_fit) <= degree:
        raise ValueError("At least degree + 1 selected points are required for the fit.")

    coeffs = np.polyfit(x_fit, y_fit, degree)
    y_hat = np.poly1d(coeffs)(x_fit)
    r_squared = r2_score(y_fit, y_hat)
    rmse = float(np.sqrt(np.mean((y_fit - y_hat) ** 2)))

    stats = {
        "r2": float(r_squared),
        "rmse": rmse,
        "n": int(len(x_fit)),
        "degree": int(degree),
        "coefficients": coeffs,
        "y fit": y_hat,
        "x fit": x_fit,
    }
    if degree == 1 and len(coeffs) >= 2:
        stats["slope"] = float(coeffs[0])
        stats["intercept"] = float(coeffs[1])

    if degree == 1 and len(coeffs) >= 2:
        equation_text = f"y={coeffs[0]:0.3g}x{coeffs[1]:+0.3g}"
    else:
        terms = []
        for i, coef in enumerate(coeffs):
            power = degree - i
            if power == 0:
                terms.append(f"{coef:+0.3g}")
            elif power == 1:
                terms.append(f"{coef:+0.3g}x")
            else:
                terms.append(f"{coef:+0.3g}x^{power}")
        equation_text = "y=" + "".join(terms).lstrip("+")

    if options.get("print", False):
        if label:
            print(f"{label}:")
        print(f"Equation: {equation_text}")
        print(f"R2: {r_squared:0.6g}")
        print(f"RMSE: {rmse:0.6g}")
        print(f"n: {len(x_fit)}")
        print("Coefficients:", coeffs)

    if plot_fit:
        fit_label_opt = options.get("fit label", False)
        plot_kwargs = {
            "color": _fit_color_value(options, index=0, fallback="tab:red"),
            "linestyle": options.get("fit linestyle", "--"),
            "lw": options.get("fit linewidth", 1),
            "alpha": options.get("fit alpha", 1),
        }

        if isinstance(fit_label_opt, str):
            plot_kwargs["label"] = fit_label_opt
        elif fit_label_opt is True:
            text = f"${equation_text}$\n$R^2 = {r_squared:0.3f}$"
            if label != "":
                text = label + "\n" + text
            plot_kwargs["label"] = text

        plt.plot(x_fit, y_hat, **plot_kwargs)
        if "label" in plot_kwargs and options.get("legend", True):
            plt.legend(fontsize=_scatterfit_legend_fontsize(options))

    if options.get("return stats", False):
        return coeffs, stats
    return coeffs, r_squared



# Moved to focused module; imported near the end of this file.
def save_animation(plots, filename, **kwargs):
    print(
        "Saving animation as ",
        _format_path_for_display(filename),
        "\nMessage will display when save is complete.",
    )
    if type(plots) == list:
        #frames_list = [plot.get_frame_count() for plot in plots]
        #most_frames = np.argmax(frames_list)
        plot = plots.pop(0)
        ani = plot.save(
            filename=filename,
            writer = "pillow",
            extra_anim=plots,
            progress_callback=lambda i, n: print('■',end=''),
            **kwargs
        )
    else:
        ani = plots.save(
            filename=filename,
            writer = "pillow",
            progress_callback=lambda i, n: print('■',end=''),
            **kwargs
        )
    print("  Save complete!")
    return ani


# Moved to focused module; imported near the end of this file.
    options = normalize_legacy_reference_options(options)
    reference_config = resolve_reference_options(options)
    reference_label = reference_config.get("label", None)

    user_supplied_reference_label = (
            "reference label" in options or "shift label" in options
    )

    folder_path = os.path.expanduser(options.get("folder path", "."))
    root_abs = os.path.abspath(folder_path)
    glob_root = glob.escape(root_abs)

    recursive = options.get("recursive search", True)
    recursive_search = "recursively" if recursive else "exclusively"

    # Validate folder
    if not os.path.exists(root_abs):
        print(f"Folder does not exist:\n{root_abs}")
        return None

    if not os.path.isdir(root_abs):
        print(f"Path exists but is not a folder:\n {root_abs}")
        return None

    print(f"Searching {recursive_search} through:\n {root_abs}")

    # ---------------------------
    # 2. Find candidate files
    # ---------------------------
    try:
        if recursive:
            search_pattern = os.path.join(glob_root, "**", "*")
        else:
            search_pattern = os.path.join(glob_root, "*")

        candidates = glob.glob(search_pattern, recursive=recursive)
        all_files = [os.path.abspath(p) for p in candidates if os.path.isfile(p)]
        txt_files = [p for p in all_files if os.path.splitext(p)[1].lower() == ".txt"]

    except Exception as exc:
        print(f"Error while searching folder:\n {root_abs}\n{exc}")
        return None

    file_paths = sorted(
        txt_files,
        key=lambda p: os.path.normcase(os.path.relpath(p, root_abs)),
    )

    if len(file_paths) == 0:
        if len(all_files) == 0:
            print(f"No files were found in the folder:\n {root_abs}")
        else:
            suffix_counts = {}
            for p in all_files:
                suffix = os.path.splitext(p)[1]
                if suffix == "":
                    suffix = "[no extension]"
                suffix_counts[suffix] = suffix_counts.get(suffix, 0) + 1

            suffix_summary = ", ".join(
                f"{ext}: {count}" for ext, count in sorted(suffix_counts.items())
            )

            print(
                f"Found {len(all_files)} file(s) in the folder, but none were .txt files.\n"
                f"Folder:\n {root_abs}\n"
                f"File types found: {suffix_summary}"
            )
        return None
    else:
        s = ""
        if len(file_paths) > 1:
            s = "s"
        print(f"{len(file_paths)} .txt file{s} found.\n")

    # ---------------------------
    # 3. Build raw objects and sort
    # ---------------------------
    raw_file_options = options.copy()
    raw_file_options["shift label"] = reference_label
    raw_file_options["shift guess"] = None
    raw_file_options["shift potential"] = False

    object_list = []
    for filepath in file_paths:
        if options.get("troubleshoot", False):
            print("Getting data from", filepath)

        try:
            echem_object = echem.from_file(filepath, raw_file_options.copy())
        except Exception as exc:
            display_name = _format_reference_display(filepath, root_abs)
            print(f"Warning: could not convert {display_name}: {exc}")
            continue

        file_folder = os.path.dirname(filepath)
        echem_object.folderpath = os.path.relpath(file_folder, folder_path)
        object_list.append(echem_object)

    if not object_list:
        print("No .txt files could be converted into eCAT objects.")
        return []

    sort_keys = options.get("sort keys", ["timestamp"])
    if isinstance(sort_keys, str):
        sort_keys = [sort_keys]
    if sort_keys:
        sort_options = {"print": False}
        object_list = sort(object_list.copy(), sort_keys, options=sort_options)

    file_paths = [os.path.abspath(obj.filepath) for obj in object_list]

    # ---------------------------
    # 4. Reference mode summary
    # ---------------------------
    reference_mode = reference_config.get("mode", "none")
    reference_label = reference_config.get("label", None)
    reference_guess = reference_config.get("guess", "auto")
    reference_active = reference_mode != "none" or bool(options.get("reference map"))
    manual_shift = reference_mode == "manual"

    if reference_mode == "manual":
        print(
            "Using manual reference shift of",
            round_sigfigs(reference_config["offset"], 4),
            "V"
        )

    elif reference_mode == "keyword":
        if str(reference_guess).lower() == "auto":
            print(
                f'Finding reference couple automatically using keyword "{reference_config.get("keyword")}".'
            )
        else:
            print(
                f'Finding reference couple using keyword "{reference_config.get("keyword")}" near',
                round_sigfigs(reference_guess, 4),
                "V"
            )

    elif reference_mode == "auto":
        keywords = reference_config.get("keywords") or []
        if str(reference_guess).lower() == "auto":
            print(
                "Automatically trying reference keywords:",
                ", ".join(keywords)
            )
        else:
            print(
                "Trying reference keywords near",
                round_sigfigs(reference_guess, 4),
                "V:",
                ", ".join(keywords)
            )

    elif reference_mode == "file":
        print("Using explicit reference file:", reference_config.get("file"))

    # ---------------------------
    # 5. Resolve reference assignments
    # ---------------------------
    # Keep current _resolve_reference_shifts() machinery for now by feeding it
    # a legacy-compatible shim.
    reference_info = {
        "use_reference_files": False,
        "ref_name": None,
        "ref_mapping": {},
        "ref_shift_guess": {},
        "self_ref_shift_guess": {},
        "self_ref_failures": {},
        "chosen_keyword": None,
    }
    explicit_reference_file = None
    explicit_reference_shift = None

    if reference_mode == "keyword":
        legacy_ref_options = options.copy()
        legacy_ref_options["shift potential"] = reference_config.get("keyword")
        legacy_ref_options["shift guess"] = reference_guess
        legacy_ref_options["shift label"] = reference_label
        legacy_ref_options["allow self reference"] = reference_config.get("allow self reference", True)

        reference_info = _resolve_reference_shifts(
            file_paths=file_paths,
            root_abs=root_abs,
            options=legacy_ref_options,
        )
        reference_info["chosen_keyword"] = reference_config.get("keyword")

    elif reference_mode == "auto":
        keywords = reference_config.get("keywords") or []
        last_error = None

        for keyword in keywords:
            legacy_ref_options = options.copy()
            legacy_ref_options["shift potential"] = keyword
            legacy_ref_options["shift guess"] = reference_guess
            legacy_ref_options["shift label"] = reference_label
            legacy_ref_options["allow self reference"] = reference_config.get("allow self reference", True)

            try:
                candidate_info = _resolve_reference_shifts(
                    file_paths=file_paths,
                    root_abs=root_abs,
                    options=legacy_ref_options,
                )

                # Accept first successful keyword that actually produced a mapping
                if candidate_info.get("use_reference_files", False):
                    reference_info = candidate_info
                    reference_info["chosen_keyword"] = keyword

                    if not user_supplied_reference_label:
                        reference_label = canonical_reference_label(
                            keyword,
                            default=reference_label,
                        )
                    break

            except Exception as exc:
                last_error = exc

        # Optional: print a soft warning if auto found nothing
        if not reference_info.get("use_reference_files", False):
            if last_error is not None and options.get("troubleshoot", False):
                print("Automatic reference search did not resolve any keyword.")
                print(last_error)

    elif reference_mode == "file":
        explicit_reference_file, explicit_reference_shift = (
            _compute_explicit_reference_file_shift(
                reference_config=reference_config,
                root_abs=root_abs,
                options=options,
            )
        )
        print(
            "Explicit reference file midpoint:",
            round_sigfigs(explicit_reference_shift, 4),
            "V"
        )

    use_reference_files = reference_info["use_reference_files"]
    ref_name = reference_info["ref_name"]
    ref_mapping = reference_info["ref_mapping"]
    ref_shift_guess = reference_info["ref_shift_guess"]
    self_ref_shift_guess = reference_info["self_ref_shift_guess"]
    self_ref_failures = reference_info["self_ref_failures"]

    # ---------------------------
    # ---------------------------
    # 6. Attach reference metadata
    # ---------------------------
    reference_records = []

    for i, echem_object in enumerate(object_list):
        filepath_abs = os.path.abspath(echem_object.filepath)

        record = {
            "index": i,
            "filepath": filepath_abs,
            "display_name": _format_reference_display(filepath_abs, root_abs),
            "mode": "none",
            "ref_file": None,
            "shift": None,
            "failure_message": None,
        }

        if manual_shift:
            record["mode"] = "manual"
            record["shift"] = float(reference_config["offset"])

        elif reference_mode == "file":
            record["mode"] = "file"
            record["ref_file"] = explicit_reference_file
            record["shift"] = explicit_reference_shift

        if use_reference_files:
            folder_abs = os.path.abspath(os.path.dirname(filepath_abs))
            basename = os.path.basename(filepath_abs)

            assigned_ref_file = ref_mapping.get(folder_abs)
            is_reference_like = _contains_string_case_insensitive(basename, ref_name)
            is_folder_reference = (
                    assigned_ref_file is not None
                    and os.path.abspath(filepath_abs) == os.path.abspath(assigned_ref_file)
            )
            is_self_reference = (
                    reference_config.get("allow self reference", True)
                    and is_reference_like
                    and not is_folder_reference
            )

            if is_self_reference:
                # Try self-reference first
                if filepath_abs in self_ref_shift_guess:
                    shift_guess = self_ref_shift_guess[filepath_abs]

                    record["mode"] = "self"
                    record["ref_file"] = filepath_abs
                    record["shift"] = shift_guess
                    
                else:
                    # Self-reference failed -> fall back to the designated nearest-ancestor reference
                    if assigned_ref_file is None:
                        raise ValueError(
                            f"No designated folder/ancestor reference file was available for fallback: {filepath_abs}"
                        )

                    fallback_shift = ref_shift_guess[assigned_ref_file]
                    failure_message = self_ref_failures.get(
                        filepath_abs,
                        "Self-reference failed for an unknown reason."
                    )

                    record["mode"] = "fallback"
                    record["ref_file"] = assigned_ref_file
                    record["shift"] = fallback_shift
                    record["failure_message"] = failure_message

            else:
                # Standard designated folder/ancestor reference path
                if assigned_ref_file is not None:
                    try:
                        shift_guess = ref_shift_guess[assigned_ref_file]

                        record["mode"] = "folder"
                        record["ref_file"] = assigned_ref_file
                        record["shift"] = shift_guess
                    except KeyError as exc:
                        display_name = _format_reference_display(assigned_ref_file, root_abs)
                        raise ValueError(
                            f"Reference shift assignment failed for: {display_name}\n"
                            "The designated folder/ancestor reference file was selected, but no "
                            "reference voltage was successfully computed for it."
                        ) from exc
        record["is_cv"] = getattr(echem_object, "type", None) == "Cyclic Voltammetry"

        # Store stable reference metadata on the object
        echem_object.reference_shift = record["shift"]
        echem_object.reference_label = reference_label
        echem_object.reference_mode = record["mode"]
        echem_object.reference_source_file = record["ref_file"]
        echem_object.reference_failure_message = record["failure_message"]

        reference_records.append(record)

    # ---------------------------
    # 6. Print summaries
    # ---------------------------
    _apply_reference_map(
        object_list=object_list,
        reference_records=reference_records,
        reference_map=options.get("reference map"),
        reference_config=reference_config,
        reference_label=reference_label,
        options=options,
    )

    _print_reference_usage_summary(reference_records)

    if options.get("troubleshoot", False):
        _print_reference_usage_troubleshoot(reference_records, root_abs)

    elif options.get("print", False):
        print_options = options.copy()

        if reference_active:
            extra_cols = [
                "reference shift",
                "reference label",
                "reference mode",
                "reference source",
            ]

            existing_cols = _coerce_display_columns(print_options.get("columns", []))

            for col in extra_cols:
                if col not in existing_cols:
                    existing_cols.append(col)

            print_options["columns"] = existing_cols

        show_objects(object_list, print_options)

    return object_list


# Moved to focused module; imported near the end of this file.

__all__ = [name for name in globals() if not name.startswith("__")]
