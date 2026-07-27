"""Validated option models for eCAT notebook-facing APIs."""

from __future__ import annotations

from dataclasses import MISSING, dataclass, field, fields
from pathlib import Path
from importlib import resources
import difflib
import re
import types
import typing
from copy import deepcopy

try:
    from IPython.display import display
except Exception:  # pragma: no cover - only used outside notebook/IPython
    display = None

try:
    import tomllib
    from tomllib import TOMLDecodeError
except ModuleNotFoundError:  # pragma: no cover - exercised on Python 3.10
    tomllib = None
    TOMLDecodeError = ValueError


_MISSING = object()
_PACKAGE_DEFAULTS = None
_USER_DEFAULTS = {}
_SESSION_DEFAULTS = {}
_GLOBAL_OPTION_DEFAULTS = {}


class OptionError(ValueError):
    """Raised when user-facing options are unknown, invalid, or conflicting."""


def normalize_key(key):
    return re.sub(r"[\s\-/]+", "_", str(key).strip().lower())


_OPTION_KEY_ALIASES = {
    "colorbar_height": "colorbar_height_scale",
    "exact_potentials": "exact_potential",
    "guess_potentials": "guess_potential",
    "in_place": "inplace",
    "invert_y": "invert_y_axis",
    "minimum_gradient_entries": "min_gradient_entries",
    "n_cat": "catalyst_electrons",
    "ncat": "catalyst_electrons",
    "n_turn": "turnover_electrons",
    "nturn": "turnover_electrons",
    "non_catalytic_guess_potentials": "non_catalytic_guess_potential",
    "peak_potentials": "peak_potential",
    "redox_potentials": "redox_potential",
    "sig_fig": "sig_figs",
    "sigfig": "sig_figs",
    "sigfigs": "sig_figs",
    "significant_figure": "sig_figs",
    "significant_figures": "sig_figs",
    "tangent_potentials": "tangent_potential",
    "wave_ranges": "wave_range",
}

_SECTION_ALIASES = {
    "cv_data": "simulation.cv_data",
    "fit_cv": "simulation.fit_cv",
    "import_data": "get_data",
    "get_data_from_excel": "get_data",
    "nicholson_analysis": "nicholson",
    "peak_potential": "cv_analysis",
    "sevcik": "sevcik_analysis",
    "simulate_cv": "simulation.simulate_cv",
    "trumpet": "trumpet_analysis",
    "tafel": "tafel_analysis",
}

_METHOD_SECTION_ALIASES = {
    "animate": "plot",
    "echem.from_file": "get_data",
    "get_data_from_excel": "get_data",
    "echem.x": "plot",
    "echem.y": "plot",
    "echem.xy": "plot",
    "echem.plot": "plot",
    "cv.x": "plot",
    "cv.y": "plot",
    "cv.xy": "plot",
    "cv.plot": "plot",
    "cv.plot_program": "plot",
    "cv.trim": "trim",
    "cv.current_at_potential": "cv_analysis",
    "cv.peak_potential": "cv_analysis",
    "cv.peak_current": "peak_current",
    "cv.peak_info": "peak_current",
    "cv.plateau_current": "plateau_current",
    "cv.half_peak_potential": "cv_analysis",
    "cv.half_wave_potential": "peak_current",
    "cv.wave_info": "peak_current",
    "cv.normalize": "normalize",
    "cv.normalize_current": "normalize_current",
    "cv.scale_current": "scale_current",
    "dpv.x": "plot",
    "dpv.y": "plot",
    "dpv.xy": "plot",
    "dpv.plot": "plot",
    "dpv.peak_potential": "cv_analysis",
    "cp.plot": "plot",
    "cp.get_cycles": "plot",
    "cp.plot_cycles": "plot",
    "cp.cycling_plot": "plot",
    "cp.cycle_info": "plot",
    "ca.plot": "plot",
    "ca.charge": "plot",
    "ca.time_at_charge": "plot",
    "ca.current_at_time": "plot",
    "ca.average_current": "plot",
    "ca.rate_at_time": "plot",
    "ca.average_rate": "plot",
}


def _canonical_option_key(key):
    norm = normalize_key(key)
    return _OPTION_KEY_ALIASES.get(norm, norm)


def _canonical_section_key(key):
    norm = normalize_key(key)
    if norm in _METHOD_SECTION_ALIASES:
        return _METHOD_SECTION_ALIASES[norm]
    return _SECTION_ALIASES.get(norm, norm)


def _option_values_equivalent(first, second):
    if first is second:
        return True
    try:
        return bool(first == second)
    except (TypeError, ValueError):
        return False


def _friendly_key(key):
    return str(key).replace("_", " ")


_DISPLAY_KEY_OVERRIDES = {
    "c": "C",
    "d": "D",
    "e0": "E0",
    "ehalf": "Ehalf",
    "follow_e1_2": "follow e1/2",
    "plot_cv": "plot CV",
}

_SECTION_OPTION_DISPLAY_OVERRIDES = {
    "fowa": {
        "exact_potential": "exact potential(s)",
        "guess_potential": "guess potential(s)",
        "non_catalytic_cv": "non-catalytic cv(s)",
        "non_catalytic_cvs": "non-catalytic cv(s)",
        "non_catalytic_guess_potential": "non-catalytic guess potential(s)",
        "peak_potential": "peak potential(s)",
        "redox_potential": "redox potential(s)",
        "tangent_potential": "tangent potential(s)",
        "wave_range": "wave range(s)",
    },
    "fit_peak_current": {
        "exact_potential": "exact potential(s)",
        "guess_potential": "guess potential(s)",
        "tangent_potential": "tangent potential(s)",
    },
    "fit_peak_potential": {
        "exact_potential": "exact potential(s)",
        "guess_potential": "guess potential(s)",
    },
    "fit_rate": {
    },
    "multi_scatterplot": {
    },
    "nicholson": {
        "exact_potential": "exact potential(s)",
        "guess_potential": "guess potential(s)",
        "tangent_potential": "tangent potential(s)",
    },
    "plateau_current": {
        "exact_potential": "exact potential(s)",
        "guess_potential": "guess potential(s)",
        "non_catalytic_cv": "non-catalytic cv(s)",
        "non_catalytic_cvs": "non-catalytic cv(s)",
        "non_catalytic_guess_potential": "non-catalytic guess potential(s)",
        "peak_potential": "peak potential(s)",
        "redox_potential": "redox potential(s)",
        "tangent_potential": "tangent potential(s)",
    },
    "sevcik_analysis": {
        "exact_potential": "exact potential(s)",
        "guess_potential": "guess potential(s)",
        "tangent_potential": "tangent potential(s)",
    },
    "trumpet_analysis": {
        "exact_potential": "exact potential(s)",
        "guess_potential": "guess potential(s)",
        "tangent_potential": "tangent potential(s)",
    },
}

_ANIMATION_OPTION_DEFAULTS = {
    "trace_mode": "auto",
    "schedule": "auto",
    "timing_mode": "auto",
    "normalized_duration": 2.0,
    "speedup": 1.0,
    "fps": 20,
    "stride": 1,
    "stagger_time": 0.5,
    "end_hold": 2,
    "loop": True,
    "include_quiet_time": False,
    "progress": True,
}

_ANIMATE_OPTION_KEYS = (
    "title",
    "subtitle",
    "labels",
    "label_alterations",
    "plot_convention",
    "x_axis",
    "y_axis",
    "x_unit",
    "y_unit",
    "color",
    "color_mode",
    "gradient_by",
    "gradient_species",
    "gradient_scale",
    "gradient_colormap",
    "gradient_colormaps",
    "gradient_colors",
    "gradient_gamma",
    "gradient_reverse",
    "min_gradient_entries",
    "legend",
    "legend_mode",
    "legend_loc",
    "legend_outside",
    "legend_fontsize",
    "colorbar_style",
    "colorbar_height_scale",
    "colorbar_reverse",
    "colorbar_tick_labels",
    "colorbar_trace_ticks",
    "plot_segments",
    "segment_color_mode",
    "segment_color_groups",
    "trace_mode",
    "schedule",
    "timing_mode",
    "normalized_duration",
    "speedup",
    "fps",
    "stride",
    "stagger_time",
    "end_hold",
    "loop",
    "include_quiet_time",
    "progress",
    "sig_figs",
    "print",
    "pretty_print",
)

_SIMULATION_OPTION_SCHEMAS = {
    "simulation.cv_data": {
        "potential window": {
            "category": "Selection/filtering",
            "default": None,
            "type": "list[float] or None",
            "description": "Potential window used to select measured CV data before simulation or fitting. With trim mode='expand', eCAT keeps connected scan data needed to preserve segment continuity.",
        },
        "trim mode": {
            "category": "Selection/filtering",
            "default": "expand",
            "type": "str",
            "choices": ["expand", "pointwise", "strict"],
            "description": "How potential-window trimming is handled: expand preserves connected CV segments, pointwise keeps only points inside the requested window, and strict raises if the requested window would disconnect the CV.",
        },
        "segments": {
            "category": "Selection/filtering",
            "default": None,
            "type": "int or list[int] or None",
            "description": "CV segment or segments to extract. If omitted, eCAT uses the selected CV data according to the window and trim mode.",
        },
        "points": {
            "category": "Selection/filtering",
            "default": None,
            "type": "int or None",
            "description": "Target number of extracted points. When supplied without stride, eCAT chooses an automatic stride from the selected potential span.",
        },
        "stride": {
            "category": "Selection/filtering",
            "default": "auto",
            "type": "int or str",
            "description": "'auto' chooses a downsampling stride from points, points per volt, and min/max point targets; an integer keeps every nth selected point.",
        },
        "points per volt": {
            "category": "Selection/filtering",
            "default": "auto",
            "type": "float or str",
            "description": "Point-density target used by automatic stride selection when points is omitted.",
        },
        "background correction": {
            "category": "Reference/correction",
            "default": None,
            "type": "str or bool or None",
            "choices": [None, "start current", "tangent"],
            "description": "Measured-current background subtraction applied after potential-window/segment selection and before stride. 'start current' subtracts the first selected current point; 'tangent' subtracts a fitted tangent baseline.",
        },
        "tangent range": {
            "category": "Reference/correction",
            "default": "auto",
            "type": "str or list[float] or tuple[float, float]",
            "description": "Potential range used only when background correction is 'tangent' and eCAT anchors the tangent from a peak/exact/guess potential.",
        },
        "tangent potential": {
            "category": "Reference/correction",
            "default": None,
            "type": "float or None",
            "description": "Manual tangent anchor potential used by cv_data background correction when 'background correction' is 'tangent'.",
        },
        "percent threshold": {
            "category": "Reference/correction",
            "default": None,
            "type": "float or None",
            "description": "Percentile threshold passed to tangent-point selection when 'background correction' is 'tangent'.",
        },
        "estimate Cdl": {
            "category": "Fitting/analysis",
            "default": "auto",
            "type": "bool or str",
            "description": "'auto' estimates total Cdl in F from measured current separation near the start potential when measured current and scan-rate metadata are available; False skips the estimate.",
        },
        "Cdl window": {
            "category": "Fitting/analysis",
            "default": "auto",
            "type": "float or str or None",
            "description": "Potential width around the start potential used for Cdl estimation; auto derives a small local window from the CV span.",
        },
        "Cdl method": {
            "category": "Fitting/analysis",
            "default": "median",
            "type": "str",
            "choices": ["median", "mean"],
            "description": "Aggregation method used for automatic Cdl estimation.",
        },
        "incubation time": {
            "category": "Data/input",
            "default": 0.0,
            "type": "float",
            "description": "Bulk homogeneous chemical incubation time in seconds applied after thermodynamic pre-equilibrium and before the electrochemical quiet-time hold; surface and mixed-phase steps remain backend-only.",
        },
    },
    "simulation.simulate_cv": {
        "plot": {
            "category": "Plotting",
            "default": True,
            "type": "bool",
            "description": "Plot the simulated CV immediately after the backend simulation completes.",
        },
        "plot all": {
            "category": "Plotting",
            "default": False,
            "type": "bool",
            "description": "Also plot backend/debug current columns when present.",
        },
        "current sign": {
            "category": "Fitting/analysis",
            "default": "auto",
            "type": "str or int",
            "choices": ["auto", "backend", "native", "flip", 1, -1],
            "description": "'auto' matches simulated-current sign to measured current when measured data are present; otherwise the backend current sign is preserved.",
        },
        "use quiet time": {
            "category": "Data/input",
            "default": True,
            "type": "bool",
            "description": "Materialize quiet-time metadata only for the backend simulation input. Stored SimulationInput arrays remain unchanged.",
        },
        "print setup": {
            "category": "Output/display",
            "default": False,
            "type": "bool or str",
            "description": "Print simulation input/mechanism setup before or after simulation; use 'raw' for debugging-style output.",
        },
        "print params": {
            "category": "Output/display",
            "default": False,
            "type": "bool or str",
            "description": "Print prepared simulation parameters. True uses the default pretty table; 'compact' groups related parameters; 'raw' prints raw dictionaries.",
        },
        "check params": {
            "category": "Output/display",
            "default": False,
            "type": "bool",
            "description": "Print diagnostic checks for likely simulation-parameter interpretation issues such as missing diffusion values or mechanism/species mismatches.",
        },
        "print states": {
            "category": "Output/display",
            "default": False,
            "type": "bool",
            "description": "Display entered, equilibrated, and post-incubation concentrations for species whose prepared amount changed.",
        },
        "plot options": {
            "category": "Plotting",
            "default": None,
            "type": "dict or None",
            "description": "Nested plotting options passed to the simulated-CV plot; omitted values reuse the top-level simulation options.",
        },
    },
    "simulation.fit_cv": {
        "residual": {
            "category": "Fitting/analysis",
            "default": "direct",
            "type": "str",
            "choices": ["direct", "scale", "scale linear baseline"],
            "description": "Residual model used during optimization. Post corrections are final-only unless the residual mode itself includes scale or baseline terms.",
        },
        "post correction": {
            "category": "Fitting/analysis",
            "default": None,
            "type": "str or None",
            "choices": [None, "scale", "vertical shift", "scale linear baseline"],
            "description": "Final-only nuisance correction applied after optimization for reporting/plotting; it is not written back into mechanism parameters.",
        },
        "residual normalization": {
            "category": "Fitting/analysis",
            "default": "max_abs_measured",
            "type": "str or None",
            "choices": [None, "max_abs_measured"],
            "description": "Residual normalization used to make optimizer cost less dependent on current magnitude; it helps compare fits within a workflow but is not a universal goodness-of-fit statistic.",
        },
        "max nfev": {
            "category": "Fitting/analysis",
            "default": None,
            "type": "int or None",
            "description": "Maximum optimizer function evaluations. Structured method dictionaries can override this budget.",
        },
        "progress": {
            "category": "Output/display",
            "default": "notebook",
            "type": "bool or str",
            "description": "Show fit progress. Notebook mode uses eCAT's progress display; False disables it.",
        },
        "progress label": {
            "category": "Output/display",
            "default": "Fitting CV",
            "type": "str",
            "description": "Label shown next to the fit progress indicator.",
        },
        "print setup": {
            "category": "Output/display",
            "default": True,
            "type": "bool or str",
            "description": "Print fitting setup, including method, residual, correction, budget, and varied/fixed parameter targets.",
        },
        "print progress": {
            "category": "Output/display",
            "default": False,
            "type": "bool or str",
            "choices": [False, True, "summary", "all"],
            "description": "Print the fitting progression table. Use 'all' to show every recorded evaluation; otherwise eCAT shows a compact summary.",
        },
        "print stats": {
            "category": "Output/display",
            "default": False,
            "type": "bool",
            "description": "Print fit statistics such as cost, residual norm, point count, and evaluation count.",
        },
        "print corrections": {
            "category": "Output/display",
            "default": False,
            "type": "bool",
            "description": "Print final-only residual/post-correction terms separately from mechanism parameters.",
        },
        "print params": {
            "category": "Output/display",
            "default": True,
            "type": "bool or str",
            "description": "Print initial/final fit parameter tables with fit status and parameter paths.",
        },
        "plot": {
            "category": "Plotting",
            "default": True,
            "type": "bool",
            "description": "Plot measured data and fitted simulated current after fitting.",
        },
        "plot all": {
            "category": "Plotting",
            "default": False,
            "type": "bool",
            "description": "Also plot raw backend current alongside the fitted/corrected current.",
        },
        "cv data": {
            "category": "Data/input",
            "default": None,
            "type": "dict or None",
            "description": "Nested options passed to simulation.cv_data when fit_cv receives a real eCAT CV object.",
        },
    },
}


OPTION_CATEGORY_ORDER = (
    "Data/input",
    "Selection/filtering",
    "Reference/correction",
    "Units/normalization",
    "Axes",
    "Labels/titles",
    "Color mapping",
    "Colorbar",
    "Legend",
    "Plotting",
    "Animation",
    "Fitting/analysis",
    "Output/display",
    "Advanced",
)

OPTION_CATEGORIES = set(OPTION_CATEGORY_ORDER)


_OPTION_CATEGORY_BY_KEY = {
    # Data and metadata inputs
    "folder_path": "Data/input",
    "file_name": "Data/input",
    "format": "Data/input",
    "delimiter": "Data/input",
    "decimal": "Data/input",
    "columns": "Data/input",
    "software": "Data/input",
    "experiment_type": "Data/input",
    "custom_reader": "Data/input",
    "custom_parser": "Data/input",
    "custom_parser_mode": "Data/input",
    "parser_settings": "Data/input",
    "recursive_search": "Data/input",
    "name_alterations": "Data/input",
    "sort_keys": "Data/input",
    "compounds": "Data/input",
    "gas": "Data/input",
    "solvent": "Data/input",
    "temperature": "Data/input",
    "metric": "Data/input",
    "x_column": "Data/input",
    "y_column": "Data/input",
    "y_columns": "Data/input",
    "group_keys": "Data/input",
    "group_by": "Data/input",
    "group_mode": "Data/input",
    "scan_rate": "Data/input",
    "species": "Data/input",
    "metadata_columns": "Selection/filtering",
    "data_columns": "Selection/filtering",
    "share_x_axes": "Output/display",

    # Selection and filtering
    "segment": "Selection/filtering",
    "segments": "Selection/filtering",
    "cycles": "Selection/filtering",
    "plot_segment": "Selection/filtering",
    "plot_segments": "Selection/filtering",
    "guess_potential": "Selection/filtering",
    "exact_potential": "Selection/filtering",
    "wave_range": "Selection/filtering",
    "fit_indices": "Selection/filtering",
    "log_fit_indices": "Selection/filtering",
    "mode": "Selection/filtering",
    "logic": "Selection/filtering",
    "overpotential_range": "Selection/filtering",
    "potential_window": "Selection/filtering",

    # Reference/correction
    "reference_mode": "Reference/correction",
    "reference_keywords": "Reference/correction",
    "reference_keyword": "Reference/correction",
    "reference_file": "Reference/correction",
    "reference_map": "Reference/correction",
    "reference_offset": "Reference/correction",
    "reference_guess": "Reference/correction",
    "reference_label": "Reference/correction",
    "allow_self_reference": "Reference/correction",
    "reference_window": "Reference/correction",
    "reference_smooth": "Reference/correction",
    "reference_max_delta_ep": "Reference/correction",
    "reference_target_delta_ep": "Reference/correction",
    "background_correction": "Fitting/analysis",
    "ecat_shift_warning_threshold": "Reference/correction",
    "ip0": "Reference/correction",
    "reference_index": "Reference/correction",
    "reference_cv": "Reference/correction",
    "reference_cvs": "Reference/correction",
    "reference_guess_potential": "Reference/correction",
    "scale": "Reference/correction",

    # Units and normalization
    "invert_current": "Units/normalization",
    "electrode_diameter": "Units/normalization",
    "electrode_area": "Units/normalization",
    "area": "Units/normalization",
    "x_unit": "Units/normalization",
    "y_unit": "Units/normalization",
    "c": "Units/normalization",
    "c_unit": "Units/normalization",
    "d": "Units/normalization",
    "e0": "Units/normalization",
    "s": "Units/normalization",
    "v": "Units/normalization",
    "n": "Units/normalization",
    "num_electrons": "Units/normalization",
    "k_homo": "Units/normalization",
    "k0": "Units/normalization",
    "formula_mode": "Units/normalization",

    # Plotting
    "plot": "Plotting",
    "plot_all": "Plotting",
    "plot_data": "Plotting",
    "plot_fit": "Plotting",
    "plot_log_log": "Plotting",
    "plot_local_slopes": "Plotting",
    "plot_diagnostic": "Plotting",
    "new_plot": "Plotting",
    "label": "Plotting",
    "labels": "Plotting",
    "deduplicate_labels": "Plotting",
    "plot_convention": "Plotting",
    "offset": "Plotting",
    "legend": "Plotting",
    "legend_mode": "Plotting",
    "legend_loc": "Plotting",
    "legend_outside": "Plotting",
    "legend_pad": "Plotting",
    "legend_bbox_to_anchor": "Plotting",
    "legend_sample_length": "Plotting",
    "legend_fontsize": "Plotting",
    "grid": "Plotting",
    "title": "Plotting",
    "subtitle": "Plotting",
    "titles": "Plotting",
    "subtitles": "Plotting",
    "title_fontsize": "Plotting",
    "subtitle_fontsize": "Plotting",
    "color": "Plotting",
    "colors": "Plotting",
    "default_discrete_colormap": "Plotting",
    "default_gradient_colormap": "Plotting",
    "color_mode": "Plotting",
    "colorbar_height_scale": "Plotting",
    "colorbar_reverse": "Plotting",
    "colorbar_style": "Plotting",
    "colorbar_tick_length": "Plotting",
    "colorbar_tick_pad": "Plotting",
    "colorbar_tick_labels": "Plotting",
    "colorbar_trace_ticks": "Plotting",
    "gradient_by": "Plotting",
    "gradient_species": "Plotting",
    "gradient_scale": "Plotting",
    "gradient_colormap": "Plotting",
    "gradient_colormaps": "Plotting",
    "gradient_colors": "Plotting",
    "gradient_gamma": "Plotting",
    "gradient_reverse": "Plotting",
    "directional_arrows": "Plotting",
    "linestyle": "Plotting",
    "simulation_linestyle": "Plotting",
    "min_gradient_entries": "Plotting",
    "plot_style": "Plotting",
    "fit_band": "Plotting",
    "fit_band_level": "Plotting",
    "fit_color": "Plotting",
    "fit_linestyle": "Plotting",
    "fit_linewidth": "Plotting",
    "fit_line_range": "Plotting",
    "fit_alpha": "Plotting",
    "fit_label": "Plotting",
    "y_col": "Plotting",
    "invert_y_axis": "Plotting",
    "invert_current_axis": "Axes",
    "invert_charge_axis": "Axes",
    "stacking": "Plotting",
    "label_alterations": "Plotting",
    "xlabel": "Axes",
    "ylabel": "Axes",
    "x_axis": "Axes",
    "y_axis": "Axes",
    "one_column": "Plotting",
    "segment_color_mode": "Plotting",
    "segment_color_groups": "Plotting",
    "trace_mode": "Animation",
    "schedule": "Animation",
    "timing_mode": "Animation",
    "normalized_duration": "Animation",
    "speedup": "Animation",
    "fps": "Animation",
    "stride": "Animation",
    "stagger_time": "Animation",
    "end_hold": "Animation",
    "loop": "Animation",
    "include_quiet_time": "Animation",
    "progress": "Output/display",
    "plot_options": "Plotting",
    "minor_ticks": "Plotting",
    "symbol_labels": "Axes",
    "scale_bar": "Plotting",
    "xscale": "Plotting",
    "yscale": "Plotting",
    "plot_scale": "Plotting",

    # Analysis and fitting behavior
    "noise_window": "Fitting/analysis",
    "noise_polyorder": "Fitting/analysis",
    "peak_prominence": "Fitting/analysis",
    "tangent_range": "Fitting/analysis",
    "tangent_min_points": "Fitting/analysis",
    "tangent_potential": "Fitting/analysis",
    "percent_threshold": "Fitting/analysis",
    "fit": "Fitting/analysis",
    "fit_basis": "Fitting/analysis",
    "min_r2": "Fitting/analysis",
    "min_fit_points": "Fitting/analysis",
    "redox_mode": "Fitting/analysis",
    "diagnostic_y_axis": "Fitting/analysis",
    "catalyst_electrons": "Fitting/analysis",
    "turnover_electrons": "Fitting/analysis",
    "x_transform": "Fitting/analysis",
    "y_transform": "Fitting/analysis",
    "transform_mode": "Fitting/analysis",
    "floor": "Fitting/analysis",
    "x_floor": "Fitting/analysis",
    "y_floor": "Fitting/analysis",
    "y_mode": "Fitting/analysis",
    "y0": "Fitting/analysis",
    "follow_e1_2": "Fitting/analysis",
    "scan_dependence": "Fitting/analysis",
    "exclude_warnings": "Fitting/analysis",
    "exclude_low_r2": "Fitting/analysis",
    "local_slope_mode": "Fitting/analysis",
    "exclude_invalid_delta_ep": "Fitting/analysis",
    "fit_through_origin": "Fitting/analysis",
    "num_electrons": "Fitting/analysis",
    "ilim": "Fitting/analysis",
    "ic": "Fitting/analysis",
    "ip0_scan_rate": "Fitting/analysis",
    "ip0_sqrt_scan_rate_slope": "Fitting/analysis",
    "plateau_slope_tolerance": "Fitting/analysis",
    "plateau_min_cvs": "Fitting/analysis",
    "plateau_average_method": "Fitting/analysis",
    "plateau_selection_mode": "Fitting/analysis",
    "validate_plateau": "Fitting/analysis",
    "require_plateau": "Fitting/analysis",
    "integrate": "Fitting/analysis",

    # Output and diagnostics
    "print": "Output/display",
    "pretty_print": "Output/display",
    "print_conditions": "Output/display",
    "print_all": "Output/display",
    "return": "Output/display",
    "return_stats": "Output/display",
    "sig_figs": "Output/display",
    "troubleshoot": "Output/display",
    "warnings": "Output/display",
    "print_local_slopes": "Output/display",
    "analysis": "Output/display",

    # Advanced/scientific-model details
    "mechanism": "Advanced",
    "sigma": "Advanced",
    "gaussian_weight": "Advanced",
    "gaussian_skew": "Advanced",
    "psi_source": "Advanced",
    "nicholson_delta_ep_min_mv": "Advanced",
    "nicholson_delta_ep_max_mv": "Advanced",
    "empirical_psi_equation": "Advanced",
    "warn_ir_drop": "Advanced",
}


_OPTION_CATEGORY_BY_SECTION = {
    "get_data": {
        "peak_prominence": "Reference/correction",
        "temperature": "Data/input",
    },
    "filter": {
        "mode": "Selection/filtering",
    },
    "group_summary": {
        "columns": "Selection/filtering",
    },
    "plot": {
        "scan_rate": "Data/input",
        "derivative": "Fitting/analysis",
        "directional_arrows": "Plotting",
    },
    "multiplot": {
        "x_axis": "Axes",
        "y_axis": "Axes",
        "x_unit": "Axes",
        "y_unit": "Axes",
        "xlabel": "Axes",
        "ylabel": "Axes",
        "labels": "Labels/titles",
        "deduplicate_labels": "Labels/titles",
        "label_alterations": "Labels/titles",
        "title": "Labels/titles",
        "subtitle": "Labels/titles",
        "titles": "Labels/titles",
        "subtitles": "Labels/titles",
        "title_fontsize": "Labels/titles",
        "subtitle_fontsize": "Labels/titles",
        "legend": "Legend",
        "legend_mode": "Legend",
        "legend_loc": "Legend",
        "legend_outside": "Legend",
        "legend_pad": "Legend",
        "legend_bbox_to_anchor": "Legend",
        "legend_sample_length": "Legend",
        "legend_fontsize": "Legend",
        "color_mode": "Color mapping",
        "colors": "Color mapping",
        "default_discrete_colormap": "Color mapping",
        "default_gradient_colormap": "Color mapping",
        "gradient_by": "Color mapping",
        "gradient_species": "Color mapping",
        "gradient_scale": "Color mapping",
        "gradient_colormap": "Color mapping",
        "gradient_colormaps": "Color mapping",
        "gradient_colors": "Color mapping",
        "gradient_gamma": "Color mapping",
        "gradient_reverse": "Color mapping",
        "min_gradient_entries": "Color mapping",
        "colorbar_height_scale": "Colorbar",
        "colorbar_reverse": "Colorbar",
        "colorbar_style": "Colorbar",
        "colorbar_tick_length": "Colorbar",
        "colorbar_tick_pad": "Colorbar",
        "colorbar_tick_labels": "Colorbar",
        "colorbar_trace_ticks": "Colorbar",
        "directional_arrows": "Plotting",
    },
    "animate": {
        "x_axis": "Axes",
        "y_axis": "Axes",
        "x_unit": "Axes",
        "y_unit": "Axes",
        "label_alterations": "Labels/titles",
        "title": "Labels/titles",
        "subtitle": "Labels/titles",
        "title_fontsize": "Labels/titles",
        "subtitle_fontsize": "Labels/titles",
        "legend": "Legend",
        "legend_mode": "Legend",
        "legend_loc": "Legend",
        "legend_outside": "Legend",
        "legend_fontsize": "Legend",
        "color_mode": "Color mapping",
        "colors": "Color mapping",
        "gradient_by": "Color mapping",
        "gradient_species": "Color mapping",
        "gradient_scale": "Color mapping",
        "gradient_colormap": "Color mapping",
        "gradient_colormaps": "Color mapping",
        "gradient_colors": "Color mapping",
        "gradient_gamma": "Color mapping",
        "gradient_reverse": "Color mapping",
        "min_gradient_entries": "Color mapping",
        "colorbar_height_scale": "Colorbar",
        "colorbar_reverse": "Colorbar",
        "colorbar_style": "Colorbar",
        "colorbar_tick_labels": "Colorbar",
        "colorbar_trace_ticks": "Colorbar",
    },
    "tafel_analysis": {
        "overpotential_range": "Fitting/analysis",
    },
}


OPTION_DESCRIPTIONS = {
    "allow_self_reference": "Allow a CV to be considered as its own reference candidate during automatic reference shifting.",
    "analysis": "Enable additional analysis output for multi-panel plotting workflows.",
    "trace_mode": "How each animated trace appears after its scheduled start time.",
    "schedule": "How multiple traces are offset relative to one another during animation playback.",
    "timing_mode": "Whether animation timing uses experiment-derived timing, normalized timing, or auto selection.",
    "normalized_duration": "Per-trace display duration in seconds when animation timing resolves to normalized mode.",
    "speedup": "Playback speed factor applied when animation timing resolves to physical mode.",
    "fps": "Animation playback frame rate in frames per second.",
    "stride": "Use every nth plotted point when building animation traces. This reduces render size without mutating source data.",
    "stagger_time": "Delay in seconds between staggered animation trace starts.",
    "end_hold": "Seconds to hold the final fully rendered frame before looping or ending.",
    "loop": "Whether animation playback loops after the final frame.",
    "include_quiet_time": "Whether animation timing includes quiet time holds when timing metadata supports them.",
    "baseline_correction": "Baseline-current correction for CA charge integration: False, True, tail, or threshold.",
    "baseline_tail_fraction": "Final fraction of a CA trace used for tail baseline correction.",
    "baseline_threshold": "Current threshold in A used for threshold CA baseline correction.",
    "corrected_current": "Use the baseline-corrected CA current trace for CA plotting or current extraction when baseline correction is enabled.",
    "method": "Filtering method: savgol, gaussian, median, butterworth, or moving average.",
    "column": "Data column to filter; matching is case-insensitive.",
    "window": "Point window for Savitzky-Golay or moving-average filtering; 'auto' resolves to a valid odd window.",
    "polyorder": "Polynomial order for Savitzky-Golay filtering.",
    "size": "Kernel size in points for median filtering.",
    "cutoff": "Butterworth low-pass cutoff as a fraction of the Nyquist frequency.",
    "order": "Butterworth filter order.",
    "background_correction": "Background correction method used before kinetic analysis.",
    "c": "Analyte or catalyst concentration; strings are parsed as concentration units when supported. For normalize, explicit C overrides species-based lookup.",
    "c_unit": "Unit for numeric concentration values. Required for numeric C; not needed when C is a concentration string or normalize resolves C from species.",
    "catalyst_electrons": "Catalyst redox-wave electron count n_cat used in kinetic equations. Aliases: n_cat, ncat.",
    "charge_color": "Color used for cumulative charge overlays and target markers.",
    "color": "Primary plot color.",
    "colors": "Explicit discrete colors for multi-trace or multi-segment plots. If omitted, eCAT uses its internal default palette.",
    "color_mode": "Color assignment mode for multi-trace plots.",
    "cycles": "Cycle selection for CP cycle plots: int, list of ints, (start, end), or (start, end, step).",
    "colorbar_height_scale": "Scale factor for colorbar height in gradient legends.",
    "colorbar_reverse": "Reverse colorbar order.",
    "colorbar_style": "Colorbar rendering style, such as auto, continuous, or discrete swatches.",
    "colorbar_tick_length": "Length of colorbar ticks.",
    "colorbar_tick_labels": "Which colorbar ticks receive text labels.",
    "colorbar_tick_pad": "Padding between colorbar ticks and labels.",
    "colorbar_trace_ticks": "Show tick marks for each trace value on gradient colorbars.",
    "columns": "Number of columns expected in imported data files.",
    "compounds": "Compound names associated with the electrochemical object.",
    "custom_formula": "Callable custom kinetic formula used instead of the built-in formula.",
    "custom_reader": "User-provided file reader for custom import formats.",
    "custom_parser": "User-provided filename metadata parser.",
    "custom_parser_mode": "How a custom filename metadata parser combines with the built-in filename metadata parser.",
    "data_columns": "Data columns included in Excel exports: 'all', 'x', 'y', or an explicit column/list of columns.",
    "d": "Diffusion coefficient in cm^2/s.",
    "decimal": "Decimal separator used in imported text files.",
    "default_discrete_colormap": "Default colormap used for discrete color legends.",
    "default_gradient_colormap": "Default colormap used for gradient legends.",
    "deduplicate_labels": "Append distinguishing metadata to duplicate multiplot labels; True uses scan window and segments.",
    "delimiter": "Column delimiter used in imported text files.",
    "diagnostic_y_axis": "Y axis used for FOWA diagnostic multiplot output.",
    "directional_arrows": "Draw scan-direction arrowhead markers at selected potentials. Use one dict or a list of dicts; each dict requires potential and may include segment, color, alpha, arrowstyle, and size. If segment is omitted, arrows are added to every segment containing that potential. color defaults to the trace color. arrowstyle is passed to Matplotlib, e.g. ->, -|>, fancy, simple, or wedge. size controls arrowhead scale.",
    "e0": "Formal potential used for physical dimensionless CV normalization.",
    "ecat_shift_warning_threshold": "Potential-shift threshold for warning that catalytic and reference waves may not align.",
    "electrode_area": "Electrode area in cm^2.",
    "electrode_diameter": "Electrode diameter in cm.",
    "ehalf": "Half-wave potential used for display or multi-panel analysis.",
    "empirical_psi_equation": "Empirical Nicholson psi equation used when not using the table lookup.",
    "exact_potential": "Exact potential to use for peak or current extraction. In complex CV analyses, the plural alias 'exact potentials' accepts per-CV values.",
    "file_name": "Base output filename without extension.",
    "format": "Export file format, such as 'csv' or 'xlsx'.",
    "exclude_invalid_delta_ep": "Exclude Nicholson points outside the valid nDelta Ep range from fitting.",
    "exclude_low_r2": "Exclude fits whose R2 is below the requested threshold.",
    "exclude_warnings": "Exclude rows or fits that emitted analysis warnings.",
    "experiment_type": "Experiment type to assume or require during import.",
    "fit": "Whether to fit the analysis result.",
    "fit_alpha": "Alpha transparency for plotted fit lines.",
    "fit_band": "Shaded uncertainty band around plotted fitted model lines: none, confidence, prediction, or both. Confidence bands show uncertainty in the fitted mean curve; prediction bands include residual scatter for a new observation.",
    "fit_band_level": "Confidence level for plotted fit bands, such as 0.95 for a 95% band.",
    "fit_basis": "Axis or quantity used to select the fit region.",
    "fit_color": "Color of plotted fit lines. Use a single color or a list for multiple fits.",
    "fit_indices": "Row/position-based fit selection. Use [start, stop] with Python-style exclusive stop, None for an open-ended side, a boolean mask, explicit indices, multiple windows for one disconnected fit, or a dict for named fits.",
    "fit_label": "Whether to label the fit line, or custom label text.",
    "fit_line_range": "Plot-only x-value range for drawing fitted model lines. A dict or list may configure multiple fit lines. This does not change fitted points, parameters, residuals, or fit statistics.",
    "fit_model": "Fit model name, callable, or formula string.",
    "fit_params": "Parameter names for a custom callable or formula fit model.",
    "fit_init": "Initial parameter guesses for a fit model.",
    "fit_bounds": "Lower and upper bounds for a fit model; use None for an unbounded side.",
    "fit_residual": "Residual mode used for model fitting.",
    "fit_max_evals": "Maximum function evaluations for model fitting.",
    "fit_method": "SciPy curve_fit method: auto, lm, trf, or dogbox.",
    "fit_sigma": "Sigma/uncertainty weights passed to scipy.optimize.curve_fit.",
    "fit_absolute_sigma": "Whether fit sigma is absolute for covariance/error estimates.",
    "fit_check_finite": "check_finite argument passed to scipy.optimize.curve_fit.",
    "fit_nan_policy": "nan_policy argument passed to scipy.optimize.curve_fit.",
    "fit_jac": "Jacobian callable or finite-difference scheme passed to scipy.optimize.curve_fit.",
    "curve_fit_options": "Advanced scipy.optimize.curve_fit keyword passthrough; eCAT still owns fit init and bounds.",
    "print_fit": "Fit print style: auto, summary, or details.",
    "print_fit_details": "If True, force detailed two-table fit printing.",
    "fit_linestyle": "Line style for plotted fit lines.",
    "fit_linewidth": "Line width for plotted fit lines.",
    "fit_range": "Single x-value fit window [x_min, x_max] on the resolved/transformed x axis. Bounds are inclusive; use None for an open-ended side where supported.",
    "fit_through_origin": "Fit the model through the origin rather than fitting an intercept.",
    "folder_path": "Folder path searched for electrochemical data files.",
    "follow_e1_2": "Use or follow E1/2 values when fitting peak-potential trends.",
    "formula_label": "Display label for a custom kinetic formula.",
    "formula_mode": "Formula-selection mode for plateau-current analysis.",
    "gas": "Gas condition metadata for the electrochemical object.",
    "gaussian_skew": "Skew parameter for Gaussian-style smoothing or weighting.",
    "gaussian_weight": "Weight parameter for Gaussian-style smoothing or weighting.",
    "gradient_by": "Metadata field used to assign gradient colors.",
    "gradient_colormap": "Colormap used for a gradient legend.",
    "gradient_colormaps": "List of colormaps used for multiple gradient legends.",
    "gradient_colors": "Explicit colors used for gradient mapping.",
    "gradient_gamma": "Gamma adjustment applied to gradient color interpolation.",
    "gradient_reverse": "Reverse gradient color order.",
    "gradient_scale": "Scale used for gradient color mapping.",
    "gradient_species": "Species used to resolve concentration-based gradient coloring.",
    "grid": "Whether to show major grid lines on plots.",
    "group_keys": "Metadata field or fields used to group objects before summarizing.",
    "group_by": "Metadata field or fields used by plateau_current when auto-grouping a flat CV list into conditions.",
    "group_mode": "Plateau-current grouping behavior: auto groups flat lists by metadata, as given treats a flat list as one validation group, and each analyzes every CV independently.",
    "guess_potential": "Initial potential guess for peak or wave selection. In complex CV analyses, the plural alias 'guess potentials' accepts per-CV values; scalar guesses keep running-guess behavior where supported.",
    "ic": "Manual catalytic plateau current.",
    "ilim": "Manual limiting or plateau current.",
    "integrate": "Integrate the selected signal when supported.",
    "internal_call": "Mark an internal helper call to suppress user-facing side effects.",
    "invert_current": "Multiply current by -1.",
    "inplace": "Mutate the existing object instead of returning a copied result.",
    "invert_y_axis": "Invert the plotted y-axis.",
    "invert_current_axis": "Override shared y-axis inversion for a CA current axis; None inherits invert y axis.",
    "invert_charge_axis": "Override shared y-axis inversion for a CA charge axis; None inherits invert y axis.",
    "ip0": "Non-catalytic reference peak current.",
    "ip0_scan_rate": "Scan rate of the CV used to measure ip0.",
    "ip0_sqrt_scan_rate_slope": "Forced-origin slope for ip0 versus sqrt(scan rate).",
    "label": "Label for a plotted trace or analysis output.",
    "label_alterations": "Text replacements applied to generated labels.",
    "labels": "Explicit labels for plotted objects.",
    "legend": "Whether to show a plot legend.",
    "legend_bbox_to_anchor": "Matplotlib legend anchor box.",
    "legend_fontsize": "Font size for plot legends.",
    "legend_loc": "Matplotlib legend location.",
    "legend_mode": "Legend mode for discrete or gradient color encodings.",
    "legend_outside": "Place the legend outside the axes.",
    "legend_pad": "Padding used when placing legends outside the axes.",
    "legend_sample_length": "Visual sample length used for legend handles.",
    "linestyle": "Line style for plotted traces.",
    "simulation_linestyle": "Line style for simulated traces in multiplot overlays.",
    "local_slope_mode": "Method for calculating local slopes.",
    "log_fit_indices": "Fit indices used for log-transformed fits.",
    "logic": "Logical rule used to combine top-level filter criteria. For membership keys such as compounds, concentrations, and species, lists require all values by default; use {'any': [...]} or {'all': [...]} inside the filter key for per-key logic.",
    "mechanism": "Electrocatalytic mechanism label or model choice.",
    "metadata_columns": "Object metadata columns included in exported manifests: 'used', 'all', or an explicit column/list of columns.",
    "method": "Filtering algorithm or model method selected by the operation.",
    "metric": "Metric column or quantity to analyze.",
    "min_fit_points": "Minimum recommended number of fit points.",
    "min_gradient_entries": "Minimum number of entries before using gradient color mapping.",
    "minor_ticks": "Minor tick behavior for plots; use True for eCAT defaults, False to disable, or an integer locator subdivision count.",
    "symbol_labels": "Use compact electrochemistry symbols for axis labels, such as E, i, j, t, and Q; 'auto' follows the active plotting style.",
    "min_r2": "Minimum recommended R2 threshold.",
    "mode": "Mode selector for the current operation.",
    "name_alterations": "Filename text replacements applied during import.",
    "new_plot": "Create a new figure before plotting.",
    "nicholson_delta_ep_max_mv": "Maximum valid nDelta Ep in mV for Nicholson analysis.",
    "nicholson_delta_ep_min_mv": "Minimum valid nDelta Ep in mV for Nicholson analysis.",
    "noise_polyorder": "Savitzky-Golay smoothing polynomial order.",
    "noise_window": "Savitzky-Golay smoothing window.",
    "non_catalytic_current": "Manual non-catalytic reference current.",
    "non_catalytic_cv": "Single non-catalytic reference CV.",
    "non_catalytic_cvs": "List of non-catalytic reference CVs.",
    "non_catalytic_guess_potential": "Potential guess used for non-catalytic reference extraction. The plural alias 'non-catalytic guess potentials' accepts per-CV values.",
    "normalize": "Whether to normalize current or axes during processing.",
    "n": "Number of electrons used in dimensionless or kinetic equations.",
    "num_electrons": "Number of electrons in the redox event.",
    "offset": "Vertical offset applied to plotted traces.",
    "one_column": "Use a one-column plot or document layout when supported.",
    "area": "Electrode area used for physical dimensionless CV normalization.",
    "overpotential_range": "Potential range used for Tafel or overpotential analysis.",
    "peak_potential": "Manual peak potential. In complex CV analyses, the plural alias 'peak potentials' accepts per-CV values.",
    "peak_fallback": "Fallback used by peak_current when peak_potential cannot find a local extremum.",
    "peak_kind": "Extremum kind used for peak-potential selection: both, infer, max, or min. The default 'both' considers maxima and minima because the current sign convention determines which one is cathodic or anodic. 'infer' maps increasing selected current to maxima and decreasing selected current to minima.",
    "peak_prominence": "Minimum peak prominence for automatic peak detection. None uses the automatic noise estimate.",
    "parser_settings": "Advanced settings for filename metadata parsing such as prefer file metadata, compound stopwords, and recognized gases/solvents.",
    "percent_threshold": "Percent threshold used in peak or tangent selection.",
    "plateau_average_method": "Average method used to combine accepted plateau currents.",
    "plateau_min_cvs": "Minimum number of CVs required in a plateau-validation subset.",
    "plateau_selection_mode": "Strategy used to select scan-rate-independent plateau subsets.",
    "plateau_slope_tolerance": "Maximum fractional slope metric accepted as scan-rate independent.",
    "plot": "Whether to plot the result.",
    "plot_all": "Whether to show diagnostic or child plots.",
    "plot_ca": "Whether CA charge helpers include the current-vs-time trace before adding diagnostics.",
    "plot_charge": "Overlay cumulative charge on a secondary axis when plotting chronoamperometry data.",
    "plot_convention": "Electrochemical plotting convention for axis orientation and signs.",
    "plot_cv": "Whether CV analysis helpers redraw the underlying CV trace before adding diagnostics.",
    "plot_diagnostic": "Whether to show diagnostic plots for the analysis.",
    "plot_data": "Whether fit_model should draw the original data points when plotting.",
    "plot_fit": "Whether to overlay fitted curves or lines.",
    "plot_local_slopes": "Whether to plot local slope diagnostics.",
    "plot_log_log": "Whether to include a log-log plot.",
    "plot_quiet_time": "When plotting a CV potential program, prepend quiet time as a negative-time hold at the starting potential.",
    "plot_scale": "Convenience axis-scale preset for scatter plots, such as log-log, semilogx, semilogy, symlog, or linear. Uses Matplotlib axis scaling and does not transform fit values.",
    "plot_options": "Nested plotting options passed through to plot helpers.",
    "scale_bar": "Draw a vertical scale bar on plot axes. Use False to hide, True to auto-pick a nice round length near 20-25% of the displayed y-axis range, a numeric value as the displayed y-axis length, or a dict with length, loc, label, unit, color, linewidth, cap width, label pad, fontsize/font size, ha, va, and remove y ticks. loc may be lower right, lower left, upper right, upper left, or an explicit (x, y) data-coordinate pair.",
    "potential_window": "Two-value potential window used to select or trim CV data.",
    "_provided_options": "Internal record of explicitly provided options.",
    "plot_peak_potential": "Whether peak-potential diagnostics are plotted during peak-current extraction.",
    "plot_reference_diagnostic": "Whether normalize_current plot-all output includes the reference CV peak-current diagnostic before the normalized overlay. Defaults to False.",
    "plot_segment": "Segment to emphasize or plot.",
    "plot_segments": "Segments to emphasize or plot.",
    "plot_style": "Plot style such as scatter or line.",
    "plot_target": "Mark the requested charge target on charge or chronoamperometry plots.",
    "pretty_print": "Use rich table display when printing object lists or summaries. False uses plain-text output when print is True.",
    "print": "Whether to emit output. False suppresses output; it is independent of pretty print.",
    "progress": "Whether to show a rendering/export progress bar when animations are displayed or saved.",
    "print_all": "Whether child helper calls should print their own summaries.",
    "print_conditions": "Whether to include condition columns in printed object tables.",
    "print_local_slopes": "Whether to print local slope diagnostics.",
    "psi_source": "Source used to resolve Nicholson psi values.",
    "recursive_search": "Recursively search subfolders during import.",
    "redox_mode": "Method used to resolve the reference redox potential.",
    "redox_potential": "Manual redox reference potential. In FOWA, the plural alias 'redox potentials' accepts per-CV values.",
    "reference_file": "Explicit file used as a reference-shift source.",
    "reference_guess": "Potential guess used to locate a reference wave.",
    "reference_cv": "Single reference CV used for current normalization or scaling.",
    "reference_cvs": "List of reference CVs used for current normalization or scaling.",
    "reference_guess_potential": "Potential guess used when extracting current from reference CVs.",
    "reference_index": "Index of the reference CV used for current normalization or scaling.",
    "reference_keyword": "Single filename keyword used to identify reference files.",
    "reference_keywords": "Filename keywords used to identify reference files.",
    "reference_label": "Axis label used after reference shifting.",
    "reference_map": "Explicit imported-object index mapping, e.g. {45: 54}, where the key is the object to shift and the value is the object used as its reference source.",
    "reference_mode": "Reference mode for the current operation.",
    "reference_offset": "Manual reference potential offset.",
    "require_plateau": "Raise an error if no scan-rate-independent plateau subset is found.",
    "return_stats": "Return a statistics dictionary instead of only a compact fit result.",
    "return": "Return the options table or menu as a pandas DataFrame.",
    "scan_dependence": "Exponent applied to scan rate.",
    "scan_rate": "Scan rate in V/s. A scalar applies to all CVs; a sequence supplies one value per CV.",
    "s": "Electrode area alias used for physical dimensionless CV normalization.",
    "scale": "Multiplier applied to raw current columns by scale_current.",
    "segment": "CV segment to analyze.",
    "segment_color_groups": "Segment grouping used for cv.plot segment coloring; integer minimum size or explicit segment groups.",
    "segment_color_mode": "Segment color mode for cv.plot, such as off, discrete, discrete gradient, or continuous gradient.",
    "segments": "One or more CV segments to analyze.",
    "reference_max_delta_ep": "Maximum reference peak separation allowed during reference matching.",
    "reference_smooth": "Smooth data before locating reference peaks.",
    "reference_target_delta_ep": "Target reference peak separation used during reference matching.",
    "reference_window": "Potential window used to locate reference peaks.",
    "sig_figs": "Significant figures used for reported values.",
    "share_x_axes": "In Excel export, combine objects with equivalent x axes into shared x-axis blocks.",
    "sigma": "Stoichiometric or pathway exponent used in FOWA kinetics.",
    "software": "Instrument software or parser to use during import.",
    "solvent": "Solvent metadata for the electrochemical object.",
    "sort_keys": "Sort keys used to order imported objects after parsing and before reference assignment.",
    "species": "Chemical species used to resolve concentration or metadata. For normalize, exact-matches cv.compounds and pulls the paired cv.concentrations value when C/C unit are omitted.",
    "stacking": "Vertically stack plotted traces.",
    "subtitle": "Plot subtitle text or subtitle mode.",
    "subtitle_fontsize": "Font size for subtitles.",
    "subtitles": "Subtitles for multiple plots or panels.",
    "tangent_min_points": "Minimum number of points used for tangent baseline fitting.",
    "tangent_potential": "Potential at which to anchor tangent baseline fitting. In complex CV analyses, the plural alias 'tangent potentials' accepts per-CV values.",
    "tangent_range": "Potential range for tangent baseline fitting.",
    "temperature": "Temperature in K.",
    "target_charge": "Target cumulative charge in Coulombs.",
    "target_electrons": "Number of electrons per molecule used to convert target moles to Coulombs.",
    "target_label": "Custom label for a plotted charge target marker.",
    "target_moles": "Target amount in moles used with target electrons to calculate charge.",
    "title": "Plot title text or title mode.",
    "title_fontsize": "Font size for titles.",
    "titles": "Titles for multiple plots or panels.",
    "transform_mode": "Transform mode applied to x and/or y data.",
    "troubleshoot": "Print additional troubleshooting output.",
    "turnover_electrons": "Turnover electron count n_turn used in kinetic equations. Aliases: n_turn, nturn.",
    "units": "Column-specific unit overrides, as a dict mapping column names to target units.",
    "validate_plateau": "Validate plateau-current scan-rate independence.",
    "v": "Scan-rate alias used for physical dimensionless CV normalization.",
    "k_homo": "Homogeneous rate constant used for EC' dimensionless CV normalization.",
    "k0": "Heterogeneous rate constant used for electrochemical dimensionless CV normalization.",
    "warn_ir_drop": "Warn when iR compensation or background subtraction cannot be verified.",
    "warnings": "Whether to emit Python warnings for non-fatal analysis quality issues.",
    "wave_range": "Manual FOWA catalytic-wave potential range; may be one range or one range per CV.",
    "x_axis": "Column or axis used as x data.",
    "x_column": "DataFrame column used as x data.",
    "x_column_index": "Column index used as x data.",
    "x_mode": "X-series adjustment mode for multi-scatter plots before x transforms.",
    "x_scale": "Scale factor applied to x values for plotting.",
    "xscale": "Matplotlib x-axis scale such as linear, log, symlog, or logit. This changes axis spacing only; use x transform to change fit values.",
    "x_transform": "Transform applied to x values.",
    "x_unit": "Requested x-axis unit.",
    "xlabel": "Custom x-axis label.",
    "y_axis": "Column or axis used as y data.",
    "y_col": "Column index used as y data.",
    "y_column": "DataFrame column used as y data.",
    "y_column_index": "Column index used as y data.",
    "y_columns": "Multiple DataFrame columns used as y data.",
    "y_mode": "Y-series adjustment mode before y transforms, such as raw, delta, ratio, or enhancement.",
    "y0": "Baseline y value for y-series adjustment; may be a scalar or keyed mapping.",
    "y_scale": "Scale factor applied to y values for plotting.",
    "yscale": "Matplotlib y-axis scale such as linear, log, symlog, or logit. This changes axis spacing only; use y transform to change fit values.",
    "y_transform": "Transform applied to y values.",
    "y_unit": "Requested y-axis unit.",
    "ylabel": "Custom y-axis label.",
    "derivative": "Derivative order or mode used for plotted data.",
    "smooth": "Smooth plotted data before display.",
    "floor": "Floor threshold for positive-only transforms. True uses 0.1x the smallest positive axis value, a number is absolute, and strings like '0.1x' are relative. Values below the threshold are replaced before transforming.",
    "x_floor": "X-axis floor threshold overriding floor for positive-only x transforms.",
    "y_floor": "Y-axis floor threshold overriding floor for positive-only y transforms.",
}


OPTION_CHOICES = {
    "background_correction": ["tangent", "start current"],
    "color_mode": ["auto", "gradient", "discrete"],
    "diagnostic_y_axis": ["i/ip0", "current"],
    "fit_basis": ["x", "y"],
    "formula_mode": ["auto", "normalized", "slope normalized", "direct"],
    "gradient_by": ["auto", "scan rate", "concentration"],
    "gradient_scale": ["auto", "linear", "sqrt", "log", "index"],
    "colorbar_tick_labels": ["endpoints", "all", "none"],
    "legend_mode": ["auto", "colorbar", "discrete"],
    "colorbar_style": ["auto", "continuous", "discrete"],
    "trace_mode": ["auto", "draw", "instant"],
    "schedule": ["auto", "simultaneous", "staggered", "sequential"],
    "timing_mode": ["auto", "physical", "normalized"],
    "local_slope_mode": ["adjacent", "gradient"],
    "logic": ["AND", "OR"],
    "mode": ["include", "exclude"],
    "plateau_average_method": ["mean", "median"],
    "plateau_selection_mode": ["high scan suffix"],
    "plot_convention": ["US", "IUPAC"],
    "plot_style": ["scatter", "line", "line+markers"],
    "plot_scale": ["linear", "log-log", "semilogx", "semilogy", "symlog"],
    "fit_band": ["none", "confidence", "prediction", "both"],
    "psi_source": ["agarwal table", "empirical"],
    "redox_mode": ["half wave", "half peak", "manual"],
    "reference_mode": ["auto", "manual", "keyword", "file", "none"],
    "segment_color_mode": ["auto", "off", "discrete", "discrete gradient", "continuous gradient"],
    "transform_mode": ["log-log", "lineweaver-burk"],
    "y_mode": ["raw", "delta", "negative delta", "ratio", "enhancement"],
}

OPTION_CHOICES_BY_SECTION = {
    "trim": {
        "mode": ["expand", "pointwise", "strict"],
    },
    "scale_current": {
        "reference_mode": ["single", "both"],
    },
    "nicholson": {
        "fit_model": ["origin", "linear"],
    },
    "save_data": {
        "format": ["csv", "xlsx", "excel"],
        "metadata_columns": ["used", "all"],
        "data_columns": ["all", "x", "y"],
    },
}


def _build_option_metadata():
    keys = set(_OPTION_CATEGORY_BY_KEY) | set(OPTION_DESCRIPTIONS) | set(OPTION_CHOICES)
    metadata = {
        "*": {
            key: {
                field: value
                for field, value in {
                    "category": _OPTION_CATEGORY_BY_KEY.get(key),
                    "choices": OPTION_CHOICES.get(key),
                    "description": OPTION_DESCRIPTIONS.get(key),
                }.items()
                if value not in (None, "")
            }
            for key in sorted(keys)
        }
    }

    for section, section_categories in _OPTION_CATEGORY_BY_SECTION.items():
        section_key = _canonical_section_key(section)
        metadata.setdefault(section_key, {})
        for key, category in section_categories.items():
            metadata[section_key].setdefault(normalize_key(key), {})["category"] = category

    for section, section_choices in OPTION_CHOICES_BY_SECTION.items():
        section_key = _canonical_section_key(section)
        metadata.setdefault(section_key, {})
        for key, choices in section_choices.items():
            metadata[section_key].setdefault(normalize_key(key), {})["choices"] = list(choices)

    def update(section, entries):
        section_key = _canonical_section_key(section)
        section_metadata = metadata.setdefault(section_key, {})
        for key, values in entries.items():
            section_metadata.setdefault(normalize_key(key), {}).update(values)

    update("get_data", {
        "temperature": {
            "description": "Temperature in K assigned to imported objects when the file does not provide one.",
        },
        "electrode_area": {
            "description": "Electrode area in cm^2 assigned to imported objects for later current-density plotting, normalization, and kinetic analysis.",
        },
        "electrode_diameter": {
            "description": "Electrode diameter in cm used to compute electrode area when electrode area is not provided.",
        },
        "scan_rate": {
            "description": "Scan rate in V/s assigned to imported objects when the parser or file metadata does not provide one.",
        },
        "reference_mode": {
            "description": "'auto' searches imported files with reference keywords; 'keyword', 'file', and 'manual' use the corresponding explicit reference source.",
        },
        "reference_guess": {
            "description": "'auto' locates the reference wave automatically; a numeric value searches near that potential.",
        },
        "reference_keywords": {
            "description": "Reference keywords tried by reference mode 'auto'; the first successful keyword defines the shift source.",
        },
        "reference_keyword": {
            "description": "Single keyword used by reference mode 'keyword' to choose reference files.",
        },
        "reference_file": {
            "description": "Explicit reference file used by reference mode 'file'; no file matching is inferred.",
        },
        "reference_map": {
            "description": "Explicit target-to-reference object index mapping; overrides automatic reference assignment for those objects.",
        },
        "peak_prominence": {
            "description": "Minimum peak prominence for automatic reference-wave detection; None lets peak detection use its internal default.",
        },
        "software": {
            "description": "Instrument software/parser. If omitted, import tries to infer the parser from file contents and extension.",
        },
        "experiment_type": {
            "description": "Experiment type to require or assign. If omitted, eCAT promotes objects from parser/file metadata when possible.",
        },
        "custom_parser": {
            "description": "Callable filename metadata parser. It can return gas, solvent, compounds, concentrations, or scan rate metadata from the object name and optional path/options context.",
        },
        "custom_parser_mode": {
            "choices": ["merge", "override"],
            "description": "How the custom filename metadata parser combines with the built-in filename parser: 'merge' fills only missing filename-derived metadata, while 'override' replaces built-in filename parser values without overriding file-derived metadata unless parser settings disable that preference.",
        },
        "parser_settings": {
            "description": "Filename parser settings dictionary. Supported keys include 'prefer file metadata', 'compound stopwords', 'solvents', and 'gases'.",
        },
    })

    update("save_data", {
        "format": {
            "description": "Output format. 'csv' writes one wide table; 'xlsx'/'excel' writes a workbook-style manifest and class-specific data sheets.",
        },
        "folder_path": {
            "description": "Folder where the exported file should be written.",
        },
        "file_name": {
            "description": "Base output filename without extension; eCAT appends .csv or .xlsx from the selected format.",
        },
        "x_unit": {
            "description": "Optional x-axis unit override applied to exported x columns when conversion is available.",
        },
        "y_unit": {
            "description": "Optional y-axis unit override applied to exported y columns when conversion is available.",
        },
        "units": {
            "description": "Column-specific unit overrides, as a dict mapping column names to target units.",
        },
        "metadata_columns": {
            "description": "Metadata columns included in the export manifest: 'used', 'all', or an explicit column/list of columns.",
        },
        "data_columns": {
            "description": "Data columns included in Excel data sheets: 'all', 'x', 'y', or an explicit column/list of columns.",
        },
        "share_x_axes": {
            "description": "When True, Excel export combines objects with equivalent x axes into shared x-axis blocks.",
        },
        "sig_figs": {
            "description": "Significant figures used in exported metadata/manifest values.",
        },
    })

    update("filter", {
        "mode": {
            "choices": ["include", "exclude"],
            "description": "Mode selector: include or exclude objects matching the filter criteria.",
        },
    })

    update("trim", {
        "potential_window": {
            "description": "Two-value potential window used to trim CV data. The shorthand e.trim(cvs, [start, stop]) fills this option.",
        },
        "mode": {
            "choices": ["expand", "pointwise", "strict"],
            "description": "Trim mode: expand preserves connected CV segments, pointwise keeps only points inside the requested window, and strict raises if the window would disconnect the CV.",
        },
        "inplace": {
            "description": "When True, mutate the source CV instead of returning a trimmed copy.",
        },
        "x_axis": {
            "description": "Optional x-axis used to evaluate the trim window; by default eCAT uses the CV potential axis, including reference shift when active.",
        },
    })

    update("plot", {
        "x_unit": {
            "description": "'auto' auto scales from the displayed data and object units; explicit units force the requested x-axis unit.",
        },
        "y_unit": {
            "description": "'auto' auto scales from the displayed data and object units; explicit units force the requested y-axis unit.",
        },
        "legend": {
            "description": "'auto' suppresses single-entry legends and shows legends when multiple plotted entries need labels.",
        },
        "legend_loc": {
            "description": "'auto' uses Matplotlib's best location for in-axes legends.",
        },
        "legend_mode": {
            "description": "'auto' auto chooses colorbar-style legends when gradient coloring is active; otherwise discrete legend behavior is used.",
        },
        "legend_sample_length": {
            "description": "'auto' sizes custom legend line samples from the legend font and tick geometry.",
        },
        "colorbar_style": {
            "description": "'auto' uses discrete swatches for small segment groups and continuous bars for continuous gradients.",
        },
        "segment_color_mode": {
            "description": "'auto' auto colors multi-segment CVs with discrete gradients when enough segments are present and keeps simple line styling for short CVs.",
        },
        "noise_window": {
            "description": "'auto' chooses an odd Savitzky-Golay window from the selected data length when smoothing is requested; None disables Savitzky-Golay smoothing.",
        },
        "noise_polyorder": {
            "description": "'auto' chooses a Savitzky-Golay polynomial order compatible with the resolved smoothing window.",
        },
        "title": {
            "description": "Plot title text or title mode. 'auto' derives a concise title from object metadata.",
        },
        "subtitle": {
            "description": "Plot subtitle text or subtitle mode. 'auto' uses object metadata that is useful context but not already in the title.",
        },
        "scan_rate": {
            "description": "Scan rate in V/s used for animation or normalization helpers; plotting otherwise uses the object's stored scan_rate when needed.",
        },
        "trace_mode": {
            "description": "'auto' resolves to 'draw' for single traces and for multi-trace animations with mixed scan rates; otherwise it resolves to 'instant'. 'draw' progressively reveals each trace, while 'instant' shows each trace fully at its scheduled start.",
        },
        "schedule": {
            "description": "'auto' resolves to simultaneous for multi-trace animations with mixed scan rates and to staggered otherwise; single-object animations do not use a schedule. Simultaneous starts all traces together, staggered offsets each start by stagger time, and sequential waits for one trace to finish before the next begins.",
        },
        "timing_mode": {
            "description": "'auto' resolves to physical when every animated object has usable timing metadata such as scan rate or time columns; otherwise it resolves to normalized. Physical uses experiment-derived timing, while normalized scales traces to a shared display duration.",
        },
        "normalized_duration": {
            "description": "Per-trace display duration in seconds when timing mode resolves to normalized.",
        },
        "speedup": {
            "description": "Playback speed factor for physical timing; values above 1 shorten the rendered animation relative to experiment time.",
        },
        "fps": {
            "description": "Animation playback frame rate in frames per second.",
        },
        "stride": {
            "description": "Use every nth plotted point when building animation traces. This reduces render size without mutating source data.",
        },
        "stagger_time": {
            "description": "Delay in seconds between staggered animation trace starts.",
        },
        "end_hold": {
            "description": "Seconds to hold the final fully rendered frame before looping or ending.",
        },
        "loop": {
            "description": "Whether animation playback loops after the final frame.",
        },
        "include_quiet_time": {
            "description": "Whether animation timing includes quiet time holds when timing metadata supports them.",
        },
    })

    update("multiplot", {
        "title": {
            "description": "'auto' builds a shared title from metadata common to the plotted objects.",
        },
        "subtitle": {
            "description": "'auto' builds shared context from metadata common to the plotted objects.",
        },
        "titles": {
            "description": "'auto' builds panel titles from grouped-object metadata.",
        },
        "subtitles": {
            "description": "'auto' builds panel subtitles from grouped-object metadata.",
        },
        "labels": {
            "description": "Explicit labels for plotted objects. If omitted, labels are generated from object metadata.",
        },
        "color_mode": {
            "description": "'auto' detects scan-rate or concentration gradients and colors those groups by gradient; remaining traces use discrete colors.",
        },
        "gradient_by": {
            "description": "'auto' first looks for scan-rate gradients, then concentration gradients.",
        },
        "gradient_species": {
            "description": "'auto' uses the species whose concentration varies within the detected concentration-gradient group.",
        },
        "gradient_scale": {
            "description": "'auto' uses log scale for scan rate and concentration gradients, and linear scale otherwise.",
        },
        "gradient_reverse": {
            "description": "Reverse trace color assignment within the gradient; unlike colorbar reverse, this changes which colors are applied to traces.",
        },
        "legend_mode": {
            "description": "'auto' uses colorbar legends for detected gradients and discrete labels for non-gradient traces.",
        },
        "legend_loc": {
            "description": "'auto' uses Matplotlib's best location for in-axes legends.",
        },
        "legend_sample_length": {
            "description": "'auto' sizes custom legend line samples from the legend font and tick geometry.",
        },
        "colorbar_reverse": {
            "description": "Reverse the displayed colorbar direction without changing trace color assignment.",
        },
        "colorbar_tick_labels": {
            "description": "'endpoints' labels only min/max colorbar values, 'all' labels every trace-value tick, and 'none' hides tick text.",
        },
        "colorbar_trace_ticks": {
            "description": "If True, draw tick marks at each trace's gradient value; if False, draw only endpoint ticks.",
        },
    })

    update("multimultiplot", {
        "titles": {
            "description": "'auto' derives one title per panel/group from grouped-object metadata.",
        },
        "subtitles": {
            "description": "'auto' derives one subtitle per panel/group from grouped-object metadata.",
        },
    })

    update("multi_scatterplot", {
        "x_column": {
            "description": "'auto' auto prefers transformed/raw x columns and falls back to the first sensible x column. Explicit columns control plotted points.",
        },
        "y_column": {
            "description": "'auto' auto prefers transformed, metric, kobs, TOFmax, ip, then Ep result columns. Explicit columns control plotted points.",
        },
        "y_columns": {
            "description": "Explicit y columns to plot; when omitted, y-column auto-resolution is used.",
        },
        "metric": {
            "description": "Preferred metric column used during y-column auto-resolution.",
        },
    })

    update("ca.plot", {
        "invert_y_axis": {
            "category": "Axes",
            "description": "Invert both current and charge y axes. A specific current/charge-axis option overrides this shared setting.",
        },
        "invert_current_axis": {
            "category": "Axes",
            "description": "Invert only the current axis. None inherits invert y axis.",
        },
        "invert_charge_axis": {
            "category": "Axes",
            "description": "Invert only the charge axis. None inherits invert y axis.",
        },
        "y_unit": {
            "category": "Units/normalization",
            "description": "Current display unit, or [current unit, charge unit] for a charge overlay. A scalar controls current only and leaves charge on auto scaling.",
        },
        "plot_charge": {
            "category": "Plotting",
            "description": "Plot cumulative charge on a styled secondary y axis.",
        },
        "charge_color": {
            "category": "Color mapping",
            "description": "Color for the charge trace, secondary-axis label, ticks, and right spine.",
        },
    })

    cv_auto = {
        "x_unit": {
            "description": "'auto' auto scales from the selected CV data and stored units.",
        },
        "y_unit": {
            "description": "'auto' auto scales from the selected CV data and stored units.",
        },
        "noise_window": {
            "description": "'auto' chooses an odd Savitzky-Golay window from the selected data length; None disables Savitzky-Golay smoothing.",
        },
        "noise_polyorder": {
            "description": "'auto' chooses a Savitzky-Golay polynomial order compatible with the resolved smoothing window.",
        },
        "peak_prominence": {
            "description": "Minimum peak prominence for automatic peak detection; None uses an automatic noise estimate, local to the guess potential when one is provided.",
        },
        "peak_kind": {
            "description": "Extremum kind to consider for peak-potential selection. 'both' considers maxima and minima; 'infer' maps increasing selected current to maxima and decreasing selected current to minima; 'max' and 'min' force one kind.",
        },
        "peak_fallback": {
            "description": "Fallback used by peak_current when no local peak is detected. 'highest current' uses the largest absolute current in the selected segment; 'guess potential' treats the guess as an exact potential; None/'none' keeps the strict error.",
        },
        "guess_potential": {
            "description": "Initial potential guess for automatic peak or wave selection. In complex CV analyses, the plural alias 'guess potentials' accepts per-CV values; scalar guesses keep running-guess behavior where supported.",
        },
        "exact_potential": {
            "description": "Exact potential for current extraction; when provided it bypasses peak-location auto-selection. In complex CV analyses, the plural alias 'exact potentials' accepts per-CV values.",
        },
        "plot_cv": {
            "description": "When plotting CV analysis diagnostics, draw the underlying CV trace before adding markers and lines.",
        },
    }
    update("cv_analysis", cv_auto)
    update("peak_current", {
        **cv_auto,
        "tangent_range": {
            "description": "'auto' chooses a pre-peak baseline region from derivative/curvature behavior before the peak.",
        },
        "tangent_min_points": {
            "description": "Minimum points for tangent fitting; when omitted, eCAT derives a minimum from the pre-peak data length.",
        },
        "tangent_potential": {
            "description": "Manual tangent anchor potential; when omitted, eCAT anchors from the resolved tangent region. In complex CV analyses, the plural alias 'tangent potentials' accepts per-CV values.",
        },
    })

    update("normalize", {
        "mode": {
            "category": "Units/normalization",
            "choices": ["homogeneous", "heterogeneous"],
            "description": "Normalization family: homogeneous uses Φ for current; heterogeneous uses χ.",
        },
        "temperature": {
            "category": "Units/normalization",
            "description": "Temperature in K. If omitted, uses the CV's temperature.",
        },
        "electrode_area": {
            "category": "Units/normalization",
            "description": "Electrode area in cm^2. If omitted for Dimensionless Current, uses the CV's electrode_area.",
        },
        "area": {
            "category": "Units/normalization",
            "description": "Electrode area alias in cm^2. If omitted for Dimensionless Current, uses the CV's electrode_area.",
        },
        "s": {
            "category": "Units/normalization",
            "description": "Electrode area alias in cm^2. If omitted for Dimensionless Current, uses the CV's electrode_area.",
        },
        "scan_rate": {
            "category": "Units/normalization",
            "description": "Scan rate in V/s. If omitted for Dimensionless Current, uses the CV's scan_rate.",
        },
        "v": {
            "category": "Units/normalization",
            "description": "Scan-rate alias in V/s. If omitted for Dimensionless Current, uses the CV's scan_rate.",
        },
        "species": {
            "category": "Units/normalization",
            "description": "If C/C unit are omitted, exact-matches cv.compounds and uses the paired cv.concentrations value.",
        },
        "c": {
            "category": "Units/normalization",
            "description": "Analyte or catalyst concentration. Explicit C overrides species-based lookup.",
        },
        "c_unit": {
            "category": "Units/normalization",
            "description": "Unit for numeric C. Required for numeric C; not needed when C is a concentration string or species resolves it.",
        },
        "d": {
            "category": "Units/normalization",
            "description": "Diffusion coefficient in cm^2/s. Required for Dimensionless Current; not inferred.",
        },
        "e0": {
            "category": "Units/normalization",
            "description": "Formal potential in V. Required for Dimensionless Potential; not inferred.",
        },
    })

    update("normalize_current", {
        "ip0": {
            "description": "Manual non-catalytic reference peak current; if omitted, eCAT extracts ip0 from the selected reference CV.",
        },
        "reference_index": {
            "description": "Reference CV index used when ip0/reference CVs are omitted; default 0 uses the first input CV.",
        },
        "reference_cv": {
            "description": "Single reference CV used to extract ip0 automatically with peak_current.",
        },
        "reference_cvs": {
            "description": "Reference CVs paired with input CVs to extract ip0 automatically with peak_current.",
        },
        "reference_guess_potential": {
            "description": "Guess potential used when extracting ip0 from reference CVs; omitted values use the peak-current guess potential.",
        },
        "plot_reference_diagnostic": {
            "category": "Plotting",
            "description": "When True, plot all is True, and ip0 is extracted from reference CVs, plot the reference CV peak-current diagnostic before the normalized overlay. Defaults to False.",
        },
        "tangent_range": {
            "description": "'auto' is passed to peak_current so ip0 extraction uses automatic tangent-baseline selection.",
        },
    })

    update("scale_current", {
        "scale": {
            "description": "Manual current multiplier. If omitted, eCAT computes a scale factor from reference CV peak currents.",
        },
        "reference_mode": {
            "choices": ["single", "both"],
            "description": "'single' scales by one reference peak current; 'both' extracts paired segment currents and computes a best scale factor.",
        },
        "reference_index": {
            "description": "Reference CV index used when scale/reference CVs are omitted; default 0 uses the first input CV.",
        },
        "reference_cv": {
            "description": "Single reference CV used to compute scale automatically with peak_current.",
        },
        "reference_cvs": {
            "description": "Reference CVs paired with input CVs to compute scale automatically with peak_current.",
        },
        "tangent_range": {
            "description": "'auto' is passed to peak_current so reference-current extraction uses automatic tangent-baseline selection.",
        },
    })

    update("fowa", {
        "fit": {
            "description": "Whether to fit each transformed FOWA trace. True fits each CV independently and warns/skips only traces with unusable fit regions; False returns and plots transformed traces without regression or kinetic values.",
        },
        "redox_mode": {
            "description": "Method used to resolve redox potential: half-wave, half-peak, or manual redox potential.",
        },
        "redox_potential": {
            "description": "Manual redox potential; required only when redox mode is manual. The plural alias 'redox potentials' accepts per-CV values.",
        },
        "ip0": {
            "description": "Manual non-catalytic peak current. If omitted, FOWA can extract ip0 from non-catalytic CV inputs.",
        },
        "non_catalytic_current": {
            "description": "Manual non-catalytic current alternative to ip0; no CV reference extraction is used when provided.",
        },
        "non_catalytic_cv": {
            "description": "Single non-catalytic CV used to extract the reference current automatically.",
        },
        "non_catalytic_cvs": {
            "description": "Non-catalytic CVs paired with catalytic CVs to extract reference currents automatically.",
        },
        "non_catalytic_guess_potential": {
            "description": "Guess potential used only for non-catalytic reference-current extraction; omitted values fall back to guess potential. The plural alias 'non-catalytic guess potentials' accepts per-CV values.",
        },
        "wave_range": {
            "description": "Manual catalytic-wave potential window. Use 'wave range' for one shared range or 'wave ranges' for one [min, max] range per catalytic CV; if omitted, FOWA selects each wave independently.",
        },
        "fit_range": {
            "description": "Single FOWA fit window on the selected fit basis. Use [min, max] for one shared window, or one [min, max] window per CV where supported.",
        },
        "tangent_range": {
            "description": "'auto' is passed to peak_current so background/current extraction uses automatic tangent-baseline selection.",
        },
        "y_axis": {
            "description": "Convenience alias for diagnostic y axis in plot-all CV overlays. Use 'Current' for raw current or 'i/ip0' for normalized current.",
        },
        "y_unit": {
            "description": "Display unit for raw-current plot-all diagnostic overlays; ignored for dimensionless i/ip0 overlays.",
        },
        "background_correction": {
            "description": "Background correction method used before FOWA; tangent uses the auto/manual tangent-baseline controls.",
        },
    })

    update("plateau_current", {
        "temperature": {
            "description": "Temperature in K. If omitted, uses CV or reference-CV temperature metadata before falling back to the option default.",
        },
        "electrode_area": {
            "description": "Electrode area in cm^2. If omitted for direct mode, uses the catalytic CV's electrode_area metadata.",
        },
        "scan_rate": {
            "description": "Manual scan rate in V/s used for normalized kobs when ip0 scan rate is omitted; otherwise catalytic/reference CV scan_rate metadata are used.",
        },
        "species": {
            "description": "Species used to resolve concentration from the catalytic CV's compounds/concentrations metadata when C is omitted.",
        },
        "c": {
            "description": "Catalyst concentration. If omitted, plateau_current can resolve it from species metadata; otherwise not inferred.",
        },
        "c_unit": {
            "description": "Unit for numeric C. If omitted for numeric C, plateau_current assumes M and converts to mol/cm^3.",
        },
        "d": {
            "description": "Diffusion coefficient in cm^2/s for direct plateau-current mode; not inferred.",
        },
        "ilim": {
            "description": "Manual catalytic limiting current. If omitted, plateau_current extracts it from catalytic CVs with peak_current.",
        },
        "ic": {
            "description": "Manual catalytic plateau current alias for ilim. If omitted, plateau_current extracts it from catalytic CVs with peak_current.",
        },
        "ip0": {
            "description": "Manual non-catalytic peak current. If omitted, plateau_current can extract it from non-catalytic CVs.",
        },
        "non_catalytic_current": {
            "description": "Manual non-catalytic reference current alternative to ip0.",
        },
        "non_catalytic_cv": {
            "description": "Single non-catalytic CV used to extract ip0 automatically.",
        },
        "non_catalytic_cvs": {
            "description": "Non-catalytic CVs used to extract ip0 or an ip0-versus-sqrt(scan rate) slope automatically.",
        },
        "ip0_scan_rate": {
            "description": "Scan rate associated with manual ip0; if omitted, scan rate may supply it.",
        },
        "ip0_sqrt_scan_rate_slope": {
            "description": "Manual forced-origin slope of ip0 versus sqrt(scan rate); if omitted, it can be fitted from multiple non-catalytic CVs.",
        },
        "formula_mode": {
            "description": "'auto' chooses direct, slope-normalized, or normalized kobs formula based on which inputs are available: D/C/electrode area, ip0 sqrt scan rate slope or multiple non-catalytic CVs, or ip0 plus scan rate from manual inputs or one non-catalytic CV.",
        },
        "diagnostic_y_axis": {
            "description": "Y-axis for plot-all CV overlays. The default 'i/ip0' makes one combined normalized overlay for catalytic CVs and available non-catalytic CVs when ip0 can be resolved from manual inputs, non-catalytic CVs, or an ip0 sqrt scan-rate slope; otherwise plateau_current falls back to current and records a warning.",
        },
        "y_axis": {
            "description": "Convenience alias for diagnostic y axis in plot-all CV overlays. Use 'Current' for raw current or 'i/ip0' for normalized current.",
        },
        "y_unit": {
            "description": "Display unit for raw-current plot-all diagnostic overlays; ignored for dimensionless i/ip0 overlays.",
        },
        "group_mode": {
            "description": "How plateau_current interprets multiple catalytic CVs: auto groups a flat list by group by, as given treats the flat list as one plateau-validation condition, and each analyzes every CV independently. Nested lists are always explicit validation groups.",
        },
        "group_by": {
            "description": "Metadata key(s) passed to e.group when group mode is auto. The default species groups by concentration-qualified compound identity.",
        },
        "tangent_range": {
            "description": "'auto' is passed to peak_current so catalytic/reference current extraction uses automatic tangent-baseline selection.",
        },
    })

    scatter_fit_auto = {
        "x_unit": {
            "description": "'auto' auto scales the resolved x data using available units.",
        },
        "y_unit": {
            "description": "'auto' auto scales the resolved y data using available units.",
        },
        "species": {
            "description": "Species used when auto-resolving concentration-based x data from CV compounds/concentrations metadata.",
        },
        "fit_indices": {
            "description": "Row/position-based fit selection on the resolved points. Use [start, stop] with Python-style exclusive stop; when omitted, all resolved points are included.",
        },
        "fit_model": {
            "description": "Model fit on the resolved x/y values. Supported models are linear, power, power offset, exponential, michaelis menten, logistic, callables, and restricted formulas such as k0 + k1*x + k2*x^2.",
        },
        "fit_params": {
            "description": "Parameter names for a custom callable or formula model; formula strings infer names when omitted.",
        },
        "fit_init": {
            "description": "'auto' chooses model-specific initial guesses from the resolved data; a list or dict overrides one or more parameter guesses.",
        },
        "fit_bounds": {
            "description": "'auto' chooses broad model-specific bounds; a [lower, upper] pair or dict by parameter name overrides them.",
        },
        "fit_residual": {
            "description": "Residual used for direct model fitting: direct, relative, log, or log10.",
        },
        "fit_max_evals": {
            "description": "Maximum function evaluations for direct model fitting.",
        },
        "fit_method": {
            "description": "SciPy curve_fit method. 'auto' lets SciPy choose; 'lm' is unconstrained Levenberg-Marquardt, while 'trf' and 'dogbox' support bounds and least_squares keyword options such as robust losses.",
            "choices": ["auto", "lm", "trf", "dogbox"],
        },
        "fit_sigma": {
            "description": "Sigma/uncertainty weights passed to scipy.optimize.curve_fit. If omitted, fit residual='relative' supplies sigma from |y| automatically; direct/log residuals leave sigma unset unless provided.",
        },
        "fit_absolute_sigma": {
            "description": "Passed as absolute_sigma to scipy.optimize.curve_fit; controls whether parameter covariance uses absolute sigma values.",
        },
        "fit_check_finite": {
            "description": "Passed as check_finite to scipy.optimize.curve_fit; leave None for SciPy's default behavior.",
        },
        "fit_nan_policy": {
            "description": "Passed as nan_policy to scipy.optimize.curve_fit, such as 'omit' or 'raise'.",
        },
        "fit_jac": {
            "description": "Jacobian callable or finite-difference scheme passed to scipy.optimize.curve_fit.",
        },
        "curve_fit_options": {
            "description": "Advanced scipy.optimize.curve_fit keyword passthrough for optimizer details such as loss, f_scale, x_scale, or maxfev. eCAT owns p0 and bounds through fit init and fit bounds; full_output is managed internally.",
        },
        "print_fit": {
            "description": "Fit print style: auto, summary, or details. Auto uses details for explicit init/bounds, custom or constrained models, 3+ parameters, or bound-hitting fits.",
        },
        "print_fit_details": {
            "description": "If True, force detailed two-table fit printing.",
        },
    }
    update("sevcik_analysis", {
        **cv_auto,
        **scatter_fit_auto,
        "c": {
            "description": "Concentration used for diffusion coefficient reporting; not inferred unless species-based concentration metadata are available to the analysis.",
        },
        "tangent_range": {
            "description": "'auto' is passed to peak_current so peak-current extraction uses automatic tangent-baseline selection.",
        },
    })
    update("fit_peak_current", {
        **cv_auto,
        **scatter_fit_auto,
        "tangent_range": {
            "description": "'auto' is passed to peak_current so peak-current extraction uses automatic tangent-baseline selection.",
        },
    })
    update("fit_peak_potential", {
        **cv_auto,
        **scatter_fit_auto,
        "follow_e1_2": {
            "description": "When True, follows paired peak-potential behavior using resolved E1/2-style values.",
        },
    })
    update("trumpet_analysis", {
        **cv_auto,
        "temperature": {
            "description": "Temperature in K. If omitted, uses each CV's temperature metadata before falling back to the option default.",
        },
        "d": {
            "description": "Diffusion coefficient in cm^2/s for ks reporting; not inferred.",
        },
        "tangent_range": {
            "description": "'auto' is passed to peak_current so peak-potential/current extraction uses automatic tangent-baseline selection.",
        },
    })
    update("nicholson", {
        **cv_auto,
        "fit_model": {
            "description": "Nicholson fit model. 'origin' uses the theoretical ψ = k0 x relation; 'linear' adds a fitted intercept as a diagnostic backup.",
        },
        "scan_rate": {
            "description": "Manual scan rate(s) in V/s. Use a scalar for all CVs or a list matching the CV list; if omitted, nicholson_analysis uses each CV's scan_rate metadata.",
        },
        "temperature": {
            "description": "Temperature in K. If omitted, nicholson_analysis uses each CV's temperature metadata before the option default.",
        },
        "d": {
            "description": "Diffusion coefficient in cm^2/s for heterogeneous rate reporting; not inferred.",
        },
        "fit_through_origin": {
            "description": "Legacy Nicholson fit toggle. When both are provided, 'fit model' wins: 'origin' constrains through the origin and 'linear' fits an intercept.",
        },
        "psi_source": {
            "description": "Source for Nicholson psi values: Agarwal table or empirical Lavagnini equation.",
        },
        "tangent_range": {
            "description": "'auto' is passed to peak_current so half-wave extraction uses automatic tangent-baseline selection.",
        },
    })
    update("fit_rate", {
        "x_column": {
            "description": "'auto' auto chooses a sensible x column: varying scan rate first, then concentration columns/compound metadata.",
        },
        "species": {
            "description": "Species used to filter concentration-column auto-resolution when x column is auto.",
        },
        "metric": {
            "description": "Metric column to fit; if omitted from plotting helpers, common rate/output metric columns are preferred.",
        },
        "fit_indices": {
            "description": "Row/position-based fit selection on the resolved rate table. Use [start, stop], disconnected windows, or a dict of named fits; stops follow Python's exclusive convention.",
        },
        "fit_model": {
            "description": "Model fit on resolved x/y values. Supported models are linear, power, power offset, exponential, michaelis menten, logistic, callables, and restricted formulas such as k0 + k1*x + k2*x^2.",
        },
        "fit_params": {
            "description": "Parameter names for a custom callable or formula model; formula strings infer names when omitted.",
        },
        "fit_init": {
            "description": "'auto' chooses model-specific initial guesses from the resolved data; a list or dict overrides one or more parameter guesses.",
        },
        "fit_bounds": {
            "description": "'auto' chooses broad model-specific bounds; a [lower, upper] pair or dict by parameter name overrides them.",
        },
        "fit_residual": {
            "description": "Residual used for direct model fitting: direct, relative, log, or log10.",
        },
        "fit_max_evals": {
            "description": "Maximum function evaluations for direct model fitting.",
        },
        "print_fit": {
            "description": "Fit print style: auto, summary, or details. Auto uses details for explicit init/bounds, custom or constrained models, 3+ parameters, or bound-hitting fits.",
        },
        "print_fit_details": {
            "description": "If True, force detailed two-table fit printing.",
        },
    })

    update("describe_options", {
        "print": {
            "description": "Whether to emit the options table or menu. False suppresses output but still allows dataframe return.",
        },
        "pretty_print": {
            "description": "Whether emitted output uses rich notebook display. False prints a plain text table; it does not suppress output.",
        },
        "return": {
            "description": "Whether to return the options table/menu as a pandas DataFrame in addition to any printed output.",
        },
    })

    return metadata


OPTION_METADATA = _build_option_metadata()


def _simple_toml_loads(text):
    data = {}
    section = None

    def parse_value(raw):
        raw = raw.strip()
        if raw == "null":
            return None
        if raw == "true":
            return True
        if raw == "false":
            return False
        if raw.startswith('"') and raw.endswith('"'):
            return raw[1:-1]
        if raw.startswith("[") and raw.endswith("]"):
            inner = raw[1:-1].strip()
            if not inner:
                return []
            return [parse_value(part.strip()) for part in inner.split(",")]
        try:
            if any(ch in raw for ch in ".eE"):
                return float(raw)
            return int(raw)
        except ValueError:
            return raw

    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("[") and line.endswith("]"):
            section = line[1:-1].strip()
            data[section] = {}
            continue
        if "=" in line and section is not None:
            key, value = line.split("=", 1)
            data[section][key.strip()] = parse_value(value)
    return data


def _toml_loads(text):
    if tomllib is None:
        return _simple_toml_loads(text)
    try:
        return tomllib.loads(text)
    except TOMLDecodeError:
        if re.search(r"=\s*null\b", text):
            return _simple_toml_loads(text)
        raise


def _load_toml_path(path):
    path = Path(path)
    return _toml_loads(path.read_text(encoding="utf-8"))


def _load_package_defaults():
    global _PACKAGE_DEFAULTS
    if _PACKAGE_DEFAULTS is None:
        default_file = resources.files("ecat").joinpath("defaults.toml")
        _PACKAGE_DEFAULTS = _toml_loads(default_file.read_text(encoding="utf-8"))
    return deepcopy(_PACKAGE_DEFAULTS)


def _normalize_mapping(mapping):
    normalized = {}
    for section, values in (mapping or {}).items():
        section_key = _canonical_section_key(section)
        normalized[section_key] = {
            _canonical_option_key(key): value
            for key, value in (values or {}).items()
        }
    return normalized


def _deep_merge(base, override):
    merged = deepcopy(base)
    for section, values in (override or {}).items():
        merged.setdefault(section, {})
        merged[section].update(values or {})
    return merged


def _option_default_registry():
    return [
        (ImportOptions, ["get_data"]),
        (TrimOptions, ["trim"]),
        (PlotOptions, ["plot", "normalize_current"]),
        (MultiplotOptions, ["plot", "multiplot", "normalize_current"]),
        (MultiMultiplotOptions, ["plot", "multiplot", "multimultiplot", "normalize_current"]),
        (MultiScatterplotOptions, ["plot", "multiplot", "multi_scatterplot"]),
        (PeakPotentialOptions, ["plot", "cv_selection", "cv_analysis"]),
        (PeakCurrentOptions, ["plot", "cv_selection", "cv_analysis", "peak_current"]),
        (NormalizeOptions, ["normalize"]),
        (NormalizationOptions, ["cv_selection", "cv_analysis", "peak_current", "normalize_current"]),
        (ScaleCurrentOptions, ["cv_analysis", "peak_current", "scale_current"]),
        (FOWAOptions, ["plot", "multiplot", "cv_selection", "cv_analysis", "peak_current", "fowa"]),
        (FitModelOptions, ["fit_model"]),
        (FitRateOptions, ["fit_rate"]),
        (FitPeakPotentialOptions, ["plot", "cv_selection", "cv_analysis", "fit_peak_potential"]),
        (SevcikAnalysisOptions, ["plot", "cv_selection", "cv_analysis", "peak_current", "sevcik_analysis"]),
        (FitPeakCurrentOptions, ["plot", "cv_selection", "cv_analysis", "peak_current", "fit_peak_current"]),
        (TrumpetAnalysisOptions, ["plot", "cv_selection", "cv_analysis", "peak_current", "trumpet_analysis"]),
        (NicholsonOptions, ["plot", "cv_selection", "cv_analysis", "peak_current", "nicholson"]),
        (PlateauCurrentOptions, ["plot", "cv_selection", "cv_analysis", "peak_current", "fowa", "plateau_current"]),
        (SortGroupOptions, ["sort_group"]),
        (GroupSummaryOptions, ["group_summary"]),
        (TafelAnalysisOptions, ["tafel_analysis"]),
        (FilterOptions, ["filter"]),
    ]


def _known_global_option_names():
    names = set()
    for cls, _sections in _option_default_registry():
        names.update(_field_names(cls))
    return names


def _default_section_names():
    sections = set(_normalize_mapping(_load_package_defaults()).keys())
    sections.update(_USER_DEFAULTS.keys())
    sections.update(_SESSION_DEFAULTS.keys())
    return sections


def _unknown_global_option_error(key):
    norm = _canonical_option_key(key)
    suggestions = difflib.get_close_matches(norm, sorted(_known_global_option_names()), n=3)
    if len(suggestions) == 1:
        hint = f" Did you mean '{_friendly_key(suggestions[0])}'?"
    elif suggestions:
        friendly = "', '".join(_friendly_key(suggestion) for suggestion in suggestions)
        hint = f" Did you mean one of '{friendly}'?"
    else:
        hint = ""
    return OptionError(f"Unknown global option '{key}'.{hint}")


def _sections_for_global_option(option_key):
    option_key = _canonical_option_key(option_key)
    sections = []
    for cls, model_sections in _option_default_registry():
        if option_key not in _field_names(cls):
            continue
        for section in model_sections:
            if section not in sections:
                sections.append(section)
    if not sections:
        raise _unknown_global_option_error(option_key)
    return sections


def _expanded_global_option_defaults():
    expanded = {}
    for option_key, value in _GLOBAL_OPTION_DEFAULTS.items():
        for section in _sections_for_global_option(option_key):
            expanded.setdefault(section, {})[option_key] = value
    return expanded


def load_defaults(path):
    """Load user defaults from a TOML file for this Python session.

    Parameters
    ----------
    path : str or path-like
        TOML file containing eCAT defaults sections.

    Returns
    -------
    dict
        Merged defaults after loading the file.

    Examples
    --------
    >>> e.load_defaults("lab_ecat_defaults.toml")
    >>> e.describe_options("get_data")
    """
    global _USER_DEFAULTS
    _USER_DEFAULTS = _normalize_mapping(_load_toml_path(path))
    return get_defaults()


def set_defaults(section_or_mapping, updates=None):
    """Set runtime defaults for an eCAT options section or global option name.

    Parameters
    ----------
    section_or_mapping : str or mapping
        Section name, global option name, or mapping of sections to updates.
    updates : object, optional
        Updates for the section or value for a global option shorthand.

    Returns
    -------
    dict
        Current merged defaults after applying the update.

    Examples
    --------
    >>> e.set_defaults("plot", {"legend mode": "colorbar"})
    >>> e.describe_options("multiplot")
    """
    global _SESSION_DEFAULTS, _GLOBAL_OPTION_DEFAULTS
    if updates is None:
        mapping = _normalize_mapping(section_or_mapping)
    elif isinstance(updates, dict):
        mapping = _normalize_mapping({section_or_mapping: updates})
    else:
        option_key = _canonical_option_key(section_or_mapping)
        _sections_for_global_option(option_key)
        _GLOBAL_OPTION_DEFAULTS[option_key] = updates
        return get_defaults()
    _SESSION_DEFAULTS = _deep_merge(_SESSION_DEFAULTS, mapping)
    return get_defaults()


def get_defaults(section=None):
    """Return merged eCAT defaults for all sections or one section.

    Parameters
    ----------
    section : str, optional
        Public defaults section name, such as ``"get_data"`` or ``"multiplot"``.

    Returns
    -------
    dict
        Defaults after package, user, session, and global overrides are merged.

    Examples
    --------
    >>> e.get_defaults("fowa")
    >>> e.describe_options("fowa")
    """
    defaults = _load_package_defaults()
    defaults = _deep_merge(defaults, _USER_DEFAULTS)
    defaults = _deep_merge(defaults, _expanded_global_option_defaults())
    defaults = _deep_merge(defaults, _SESSION_DEFAULTS)
    defaults = _normalize_mapping(defaults)
    if section is None:
        return defaults
    return deepcopy(defaults.get(_canonical_section_key(section), {}))


def reset_defaults(section=None):
    """Reset user-loaded and session defaults for all sections or one target.

    Parameters
    ----------
    section : str, optional
        Section or global option shorthand to reset.

    Returns
    -------
    dict
        Current defaults after reset.

    Examples
    --------
    >>> e.reset_defaults()
    >>> e.describe_options("plot")
    """
    global _USER_DEFAULTS, _SESSION_DEFAULTS, _GLOBAL_OPTION_DEFAULTS
    if section is None:
        _USER_DEFAULTS = {}
        _SESSION_DEFAULTS = {}
        _GLOBAL_OPTION_DEFAULTS = {}
        return get_defaults()
    else:
        section_key = _canonical_section_key(section)
        if section_key in _default_section_names():
            _USER_DEFAULTS.pop(section_key, None)
            _SESSION_DEFAULTS.pop(section_key, None)
        else:
            reset_defaults_option(section_key)
        return get_defaults(section_key)


def reset_defaults_option(option):
    """Reset a runtime shorthand default from every section that supports an option."""
    global _GLOBAL_OPTION_DEFAULTS, _SESSION_DEFAULTS
    option_key = _canonical_option_key(option)
    _sections_for_global_option(option_key)
    _GLOBAL_OPTION_DEFAULTS.pop(option_key, None)
    for section_values in _SESSION_DEFAULTS.values():
        section_values.pop(option_key, None)
    return get_defaults()


def reset_defaults_section(section):
    """Explicitly reset one defaults section."""
    global _USER_DEFAULTS, _SESSION_DEFAULTS
    section_key = _canonical_section_key(section)
    _USER_DEFAULTS.pop(section_key, None)
    _SESSION_DEFAULTS.pop(section_key, None)
    return get_defaults(section_key)


def _field_names(cls):
    return {field.name for field in fields(cls)}


def _choice_token(value):
    return str(value).strip().lower().replace("_", " ").replace("-", " ")


def _choice_lookup_token(value):
    return _choice_token(value).replace(" ", "").replace("-", "")


def _choices_for_option(option_key, sections):
    option_key = _canonical_option_key(option_key)
    choices = OPTION_CHOICES.get(option_key)
    for section in sections:
        section_choices = OPTION_CHOICES_BY_SECTION.get(_canonical_section_key(section), {})
        for key, section_values in section_choices.items():
            if _canonical_option_key(key) == option_key:
                choices = section_values
    return list(choices) if choices else None


def _canonicalize_choice_value(option_key, value, sections):
    choices = _choices_for_option(option_key, sections)
    if not choices:
        return value

    canonical_by_token = {}
    for choice in choices:
        canonical_by_token[_choice_token(choice)] = choice
        canonical_by_token[_choice_lookup_token(choice)] = choice

    none_choice = canonical_by_token.get("none")
    if none_choice is not None:
        if value is None or value is False:
            return none_choice
        if isinstance(value, str) and _choice_token(value) in {"none", "null", "off", "false", "no", "0"}:
            return none_choice

    if isinstance(value, str):
        token = _choice_token(value)
        return canonical_by_token.get(token, canonical_by_token.get(_choice_lookup_token(value), value))

    if isinstance(value, list):
        return [_canonicalize_choice_value(option_key, item, sections) for item in value]
    if isinstance(value, tuple):
        return tuple(_canonicalize_choice_value(option_key, item, sections) for item in value)

    return value


def _coerce_options(cls, raw, sections):
    if isinstance(raw, cls):
        return raw
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise TypeError(f"{cls.__name__} accepts a dict or {cls.__name__}, not {type(raw).__name__}.")

    valid = _field_names(cls)
    normalized = {}
    original_keys = {}
    for key, value in raw.items():
        if str(key).startswith("_") and str(key) != "_provided_options":
            continue
        norm = _canonical_option_key(key)
        if cls.__name__ == "ImportOptions" and norm == "shift_potential":
            norm = "reference_mode"
            value = "none" if value is False or str(value).strip().lower() in {"false", "off", "none", "0"} else "auto"
        if cls.__name__ in {"PlotOptions", "MultiplotOptions"} and norm == "y_flip":
            raise OptionError("'y flip' was removed; use 'invert y axis' to invert the plotted y-axis.")
        if norm not in valid:
            if cls.__name__ == "ImportOptions" and norm == "convert_current":
                raise OptionError(
                    "'convert current' is not an import option. eCAT stores imported "
                    "current in SI units (A); use plot/display options such as "
                    "'y unit': 'uA' when you want microamp axes."
                )
            if cls.__name__ == "ImportOptions" and norm == "current_density":
                raise OptionError(
                    "'current density' is not an import option. Pass 'electrode area' "
                    "during import, then use plot/display options such as "
                    "'y axis': 'current density'."
                )
            suggestions = difflib.get_close_matches(norm, sorted(valid), n=3)
            if len(suggestions) == 1:
                hint = f" Did you mean '{_friendly_key(suggestions[0])}'?"
            elif suggestions:
                friendly = "', '".join(_friendly_key(suggestion) for suggestion in suggestions)
                hint = f" Did you mean one of '{friendly}'?"
            else:
                hint = ""
            raise OptionError(f"Unknown option '{key}' for {cls.__name__}.{hint}")
        if norm in normalized:
            first_key = original_keys[norm]
            alias_key = None
            if norm == "labels":
                for candidate in (first_key, key):
                    if _canonical_option_key(candidate) == norm and normalize_key(candidate) != norm:
                        alias_key = candidate
                        break
            if alias_key is not None:
                raise OptionError(
                    f"Unknown option '{alias_key}' for {cls.__name__}. "
                    "Did you mean 'labels'?"
                )
            raise OptionError(
                f"Options '{first_key}' and '{key}' both resolve to "
                f"'{_friendly_key(norm)}'. Use only one spelling."
            )
        normalized[norm] = _canonicalize_choice_value(norm, value, sections)
        original_keys[norm] = key

    defaults = {}
    for section in sections:
        defaults.update(get_defaults(section))

    kwargs = {
        name: defaults.get(name, field.default)
        for name, field in ((field.name, field) for field in fields(cls))
        if field.default is not MISSING
    }
    kwargs.update(normalized)
    if "_provided_options" in valid:
        kwargs["_provided_options"] = tuple(normalized)
    if (
        "segments" in normalized
        and normalized["segments"] is not None
        and "segment" not in normalized
        and "segment" in valid
    ):
        kwargs["segment"] = None
    if (
        "segment" in normalized
        and normalized["segment"] is not None
        and "segments" not in normalized
        and "segments" in valid
    ):
        kwargs["segments"] = None
    opts = cls(**kwargs)
    opts.validate()
    return opts


def _validate_fit_band_options(opts):
    fit_band = getattr(opts, "fit_band", None)
    if fit_band not in (None, False):
        token = str(fit_band).strip().lower().replace("_", " ").replace("-", " ")
        if token not in {"none", "confidence", "prediction", "both"}:
            raise OptionError("'fit band' must be none, confidence, prediction, or both.")
    level = float(getattr(opts, "fit_band_level", 0.95))
    if not 0 < level < 1:
        raise OptionError("'fit band level' must be between 0 and 1.")


def _validate_common_cv(opts):
    if opts.segment is not None and opts.segments is not None:
        raise OptionError("Use either 'segment' or 'segments', not both.")
    if opts.exact_potential is not None and opts.guess_potential is not None:
        raise OptionError("Use either 'exact potential' or 'guess potential', not both.")
    if opts.noise_window not in (None, "auto"):
        if not isinstance(opts.noise_window, int) or opts.noise_window < 3 or opts.noise_window % 2 == 0:
            raise OptionError("'noise window' must be None, 'auto', or an odd integer >= 3.")
    if (
        opts.noise_polyorder != "auto"
        and opts.noise_window not in (None, "auto")
        and int(opts.noise_polyorder) >= int(opts.noise_window)
    ):
        raise OptionError("'noise polyorder' must be less than 'noise window'.")
    peak_kind = getattr(opts, "peak_kind", "both")
    if peak_kind is not None:
        token = _choice_token(peak_kind)
        if token not in {
            "both",
            "any",
            "all",
            "none",
            "infer",
            "inferred",
            "max",
            "maximum",
            "min",
            "minimum",
        }:
            raise OptionError("'peak kind' must be 'both', 'infer', 'max', or 'min'.")


@dataclass(frozen=True, slots=True)
class ImportOptions:
    folder_path: str = "."
    delimiter: str = ","
    decimal: str = "."
    columns: int = 3
    software: str | None = None
    experiment_type: str | None = None
    custom_reader: object | None = None
    custom_parser: object | None = None
    custom_parser_mode: str = "merge"
    parser_settings: dict | None = None
    print: bool = False
    troubleshoot: bool = False
    recursive_search: bool = True
    name_alterations: object | None = None
    pretty_print: bool = True
    sort_keys: object = field(default_factory=lambda: ["subfolder", "timestamp"])
    reference_mode: str = "auto"
    reference_keywords: list[str] | None = None
    reference_keyword: str | None = None
    reference_file: str | None = None
    reference_map: dict | None = None
    reference_offset: float | None = None
    reference_guess: float | str | None = "auto"
    reference_label: str = "Fc/Fc+"
    allow_self_reference: bool = True
    reference_window: float = 0.3
    reference_smooth: bool = True
    reference_max_delta_ep: float = 0.20
    reference_target_delta_ep: float = 0.08
    peak_prominence: float | None = None
    compounds: object | None = None
    gas: str | None = None
    solvent: str | None = None
    temperature: float = 298
    electrode_diameter: float = 0
    electrode_area: float = 0
    invert_current: bool = False
    scan_rate: float | None = None
    _provided_options: tuple[str, ...] = field(default=(), repr=False, compare=False)

    @classmethod
    def from_options(cls, options=None):
        return _coerce_options(cls, options, ["get_data"])

    def validate(self):
        if not isinstance(self.recursive_search, bool):
            raise OptionError("'recursive search' must be True or False.")
        if self.sort_keys is not None and not isinstance(self.sort_keys, (str, list, tuple)):
            raise OptionError("'sort keys' must be a string, list, tuple, or None.")
        if self.custom_parser is not None and not callable(self.custom_parser):
            raise OptionError("'custom parser' must be callable or None.")
        custom_parser_mode = str(self.custom_parser_mode).strip().lower().replace("_", " ").replace("-", " ")
        if custom_parser_mode not in {"merge", "override"}:
            raise OptionError("'custom parser mode' must be 'merge' or 'override'.")
        if self.parser_settings is not None and not isinstance(self.parser_settings, dict):
            raise OptionError("'parser settings' must be a dictionary or None.")
        allowed_modes = {"none", "off", "false", "manual", "keyword", "auto", "file"}
        if str(self.reference_mode).strip().lower() not in allowed_modes:
            raise OptionError("'reference mode' must be 'auto', 'manual', 'keyword', 'file', or 'none'.")
        if self.reference_map is not None:
            if not isinstance(self.reference_map, dict):
                raise OptionError("'reference map' must be a dictionary such as {45: 54}.")
            for target_idx, reference_idx in self.reference_map.items():
                if not isinstance(target_idx, int) or not isinstance(reference_idx, int):
                    raise OptionError("'reference map' keys and values must be integer object indices.")
    def to_options_dict(self):
        data = {
            field.name.replace("_", " "): getattr(self, field.name)
            for field in fields(self)
            if not field.name.startswith("_")
        }
        data["folder path"] = self.folder_path
        data["experiment type"] = self.experiment_type
        data["custom reader"] = self.custom_reader
        data["custom parser"] = self.custom_parser
        data["custom parser mode"] = self.custom_parser_mode
        data["parser settings"] = self.parser_settings
        data["recursive search"] = self.recursive_search
        data["name alterations"] = self.name_alterations
        data["pretty print"] = self.pretty_print
        data["sort keys"] = self.sort_keys
        data["gas"] = self.gas
        data["solvent"] = self.solvent
        data["reference mode"] = self.reference_mode
        data["reference keywords"] = self.reference_keywords
        data["reference keyword"] = self.reference_keyword
        data["reference file"] = self.reference_file
        data["reference map"] = self.reference_map
        data["reference offset"] = self.reference_offset
        data["reference guess"] = self.reference_guess
        data["reference label"] = self.reference_label
        data["allow self reference"] = self.allow_self_reference
        data["reference window"] = self.reference_window
        data["reference smooth"] = self.reference_smooth
        data["reference max delta ep"] = self.reference_max_delta_ep
        data["reference target delta ep"] = self.reference_target_delta_ep
        data["peak prominence"] = self.peak_prominence
        data["electrode diameter"] = self.electrode_diameter
        data["electrode area"] = self.electrode_area
        data["_electrode area provided"] = "electrode_area" in self._provided_options
        data["invert current"] = self.invert_current
        data["scan rate"] = self.scan_rate
        return data


def resolve_import_options(options=None):
    if isinstance(options, dict) and "_electrode area provided" in options:
        return dict(options)
    return ImportOptions.from_options(options).to_options_dict()


@dataclass(frozen=True, slots=True)
class TrimOptions:
    potential_window: object | None = None
    mode: str | None = None
    inplace: bool = False
    x_axis: object | None = None

    @classmethod
    def from_options(cls, options=None):
        return _coerce_options(cls, options, [])

    @staticmethod
    def _canonical_mode(value):
        return str(value).strip().lower().replace("_", " ").replace("-", " ")

    def validate(self):
        if self.potential_window is None:
            raise OptionError("cv.trim requires 'potential window'.")
        try:
            values = list(self.potential_window)
        except TypeError as exc:
            raise OptionError("'potential window' must be a two-value sequence like [start, stop].") from exc
        if len(values) != 2:
            raise OptionError("'potential window' must contain exactly two values.")
        try:
            float(values[0])
            float(values[1])
        except (TypeError, ValueError) as exc:
            raise OptionError("'potential window' values must be numeric.") from exc

        mode = self._canonical_mode("expand" if self.mode is None else self.mode)
        if mode not in {"expand", "pointwise", "strict"}:
            raise OptionError("'mode' must be 'expand', 'pointwise', or 'strict'.")

    def to_options_dict(self):
        mode = self._canonical_mode("expand" if self.mode is None else self.mode)
        return {
            "potential window": self.potential_window,
            "mode": mode,
            "inplace": self.inplace,
            "x axis": self.x_axis,
        }


@dataclass(frozen=True, slots=True)
class PlotOptions:
    plot: bool = True
    print: bool = True
    pretty_print: bool = True
    plot_all: bool = False
    print_all: bool = False
    new_plot: bool = False
    label: str | None = None
    linestyle: str | None = None
    legend: bool | str = "auto"
    title: bool | str = True
    subtitle: bool | str | None = "auto"
    color: str | None = "black"
    colors: list[str] = field(default_factory=list)
    default_gradient_colormap: str = "viridis"
    color_mode: str = "auto"
    gradient_by: str = "auto"
    gradient_species: str = "auto"
    gradient_scale: str = "auto"
    gradient_colormap: str | None = None
    gradient_colormaps: list[str] = field(default_factory=list)
    gradient_colors: list[str] = field(default_factory=list)
    gradient_gamma: float = 1.0
    gradient_reverse: bool = False
    min_gradient_entries: int = 3
    colorbar_height_scale: float = 1.0
    colorbar_reverse: bool = True
    colorbar_style: str = "auto"
    colorbar_tick_length: int = 5
    colorbar_tick_pad: int = 8
    colorbar_tick_labels: str = "endpoints"
    colorbar_trace_ticks: bool = True
    legend_mode: str = "auto"
    y_col: int = -1
    invert_y_axis: bool = False
    invert_current_axis: bool | None = None
    invert_charge_axis: bool | None = None
    plot_convention: str = "IUPAC"
    sig_figs: int = 4
    offset: float = 0
    stacking: bool = False
    label_alterations: object | None = None
    legend_sample_length: float | str = "auto"
    legend_fontsize: int | None = None
    legend_loc: str = "auto"
    legend_outside: bool = False
    legend_pad: float = 0.02
    grid: bool = False
    title_fontsize: int | str | None = None
    subtitle_fontsize: int | str | None = None
    x_unit: str | None = "auto"
    y_unit: str | list[str | None] | tuple[str | None, str | None] | None = "auto"
    x_scale: float | int = 1
    y_scale: float | int = 1
    xlabel: str | None = None
    ylabel: str | None = None
    x_axis: str | None = None
    y_axis: str | None = None
    x_column_index: int = -1
    y_column_index: int = -1
    derivative: int | float | None = 0
    smooth: bool = False
    noise_window: int | str | None = "auto"
    noise_polyorder: int | str = "auto"
    one_column: bool = True
    cycles: object = None
    plot_segment: int | None = None
    plot_segments: list[int] | int | None = None
    segment_color_mode: str = "auto"
    segment_color_groups: object = 2
    integrate: bool = False
    plot_charge: bool = False
    plot_ca: bool = True
    plot_target: bool = True
    plot_quiet_time: bool = True
    target_charge: float | None = None
    target_moles: float | None = None
    target_electrons: float | None = None
    target_label: str | None = None
    charge_color: str = "tab:red"
    baseline_correction: bool | str = False
    baseline_threshold: float | None = None
    baseline_tail_fraction: float = 0.05
    corrected_current: bool = False
    trace_mode: str = "auto"
    schedule: str = "auto"
    timing_mode: str = "auto"
    normalized_duration: int | float = 2.0
    speedup: int | float = 1.0
    fps: int | float = 20
    stride: int = 1
    stagger_time: int | float = 0.5
    end_hold: int | float = 2
    loop: bool = True
    include_quiet_time: bool = False
    progress: bool | str = True
    scan_rate: float | None = None
    ip0: float | list[float] | None = None
    reference_index: int = 0
    reference_cv: object | None = None
    reference_cvs: object | None = None
    plot_options: dict | None = None
    scale_bar: object = False
    directional_arrows: object = False
    minor_ticks: bool | int = 2
    symbol_labels: bool | str = "auto"

    @classmethod
    def from_options(cls, options=None):
        return _coerce_options(cls, options, ["plot", "normalize_current"])

    def validate(self):
        def _normalized_choice(value):
            return str(value).strip().lower().replace("_", " ").replace("-", " ")

        def _require_non_negative_number(label, value):
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                raise OptionError(f"'{label}' must be a non-negative number.")
            if numeric < 0:
                raise OptionError(f"'{label}' must be a non-negative number.")

        def _require_bool(label, value):
            if value is not True and value is not False:
                raise OptionError(f"'{label}' must be True or False.")

        sources = [
            self.ip0 is not None,
            self.reference_cv is not None,
            self.reference_cvs is not None,
            self.reference_index != 0,
        ]
        if sum(bool(source) for source in sources) > 1:
            raise OptionError("Use only one ip0/reference source for i/ip0 plotting.")
        mode = _normalized_choice(self.segment_color_mode)
        if mode in {"none", "false", "0"}:
            mode = "off"
        if mode not in {"auto", "off", "discrete", "discrete gradient", "continuous gradient"}:
            raise OptionError(
                "'segment color mode' must be 'auto', 'off', 'discrete', "
                "'discrete gradient', or 'continuous gradient'."
            )
        legend_mode = _normalized_choice(self.legend_mode)
        if legend_mode not in {"auto", "colorbar", "discrete"}:
            raise OptionError("'legend mode' must be 'auto', 'colorbar', or 'discrete'.")
        color_mode = _normalized_choice(self.color_mode)
        if color_mode not in {"auto", "gradient", "discrete"}:
            raise OptionError("'color mode' must be 'auto', 'gradient', or 'discrete'.")
        gradient_scale = _normalized_choice(self.gradient_scale)
        if gradient_scale not in {"auto", "linear", "sqrt", "log", "index"}:
            raise OptionError("'gradient scale' must be 'auto', 'linear', 'sqrt', 'log', or 'index'.")
        colorbar_style = _normalized_choice(self.colorbar_style)
        if colorbar_style not in {"auto", "continuous", "discrete", "swatch", "swatches"}:
            raise OptionError("'colorbar style' must be 'auto', 'continuous', or 'discrete'.")
        trace_mode = _normalized_choice(self.trace_mode)
        if trace_mode not in {"auto", "draw", "instant"}:
            raise OptionError("'trace mode' must be 'auto', 'draw', or 'instant'.")
        schedule = _normalized_choice(self.schedule)
        if schedule not in {"auto", "simultaneous", "staggered", "sequential"}:
            raise OptionError(
                "'schedule' must be 'auto', 'simultaneous', 'staggered', or 'sequential'."
            )
        timing_mode = _normalized_choice(self.timing_mode)
        if timing_mode not in {"auto", "physical", "normalized"}:
            raise OptionError("'timing mode' must be 'auto', 'physical', or 'normalized'.")
        try:
            if float(self.normalized_duration) <= 0:
                raise ValueError
        except (TypeError, ValueError):
            raise OptionError("'normalized duration' must be a positive number.")
        try:
            if float(self.speedup) <= 0:
                raise ValueError
        except (TypeError, ValueError):
            raise OptionError("'speedup' must be a positive number.")
        _require_non_negative_number("fps", self.fps)
        try:
            if int(self.stride) <= 0:
                raise ValueError
        except (TypeError, ValueError):
            raise OptionError("'stride' must be a positive integer.")
        _require_non_negative_number("stagger time", self.stagger_time)
        _require_non_negative_number("end hold", self.end_hold)
        _require_bool("loop", self.loop)
        _require_bool("include quiet time", self.include_quiet_time)
        if self.invert_current_axis is not None:
            _require_bool("invert current axis", self.invert_current_axis)
        if self.invert_charge_axis is not None:
            _require_bool("invert charge axis", self.invert_charge_axis)
        baseline_correction = self.baseline_correction
        if isinstance(baseline_correction, str):
            baseline_correction = _normalized_choice(baseline_correction)
            if baseline_correction not in {"tail", "threshold", "true", "false", "on", "off"}:
                raise OptionError(
                    "'baseline correction' must be True, False, 'tail', or 'threshold'."
                )
        try:
            if float(self.baseline_tail_fraction) <= 0 or float(self.baseline_tail_fraction) > 1:
                raise ValueError
        except (TypeError, ValueError):
            raise OptionError("'baseline tail fraction' must be a number between 0 and 1.")
        if self.minor_ticks is not True and self.minor_ticks is not False:
            try:
                if int(self.minor_ticks) < 0:
                    raise ValueError
            except (TypeError, ValueError):
                raise OptionError("'minor ticks' must be True, False, or a non-negative integer.")
        symbol_labels = _normalized_choice(self.symbol_labels)
        if self.symbol_labels is not True and self.symbol_labels is not False and symbol_labels != "auto":
            raise OptionError("'symbol labels' must be True, False, or 'auto'.")

        # Import locally to avoid importing plotting helpers at module import time.
        from ._plot_helpers import _normalize_directional_arrows_options

        try:
            _normalize_directional_arrows_options({"directional arrows": self.directional_arrows})
        except ValueError as exc:
            raise OptionError(str(exc)) from exc

    def to_options_dict(self):
        data = {
            item.name.replace("_", " "): getattr(self, item.name)
            for item in fields(self)
            if not item.name.startswith("_")
        }
        data["plot all"] = self.plot_all
        data["print all"] = self.print_all
        data["new plot"] = self.new_plot
        data["y col"] = self.y_col
        data["invert y axis"] = self.invert_y_axis
        data["invert current axis"] = self.invert_current_axis
        data["invert charge axis"] = self.invert_charge_axis
        data["plot convention"] = self.plot_convention
        data["sig figs"] = self.sig_figs
        data["label alterations"] = self.label_alterations
        data["legend sample length"] = self.legend_sample_length
        data["legend fontsize"] = self.legend_fontsize
        data["legend loc"] = self.legend_loc
        data["legend mode"] = self.legend_mode
        data["legend outside"] = self.legend_outside
        data["legend pad"] = self.legend_pad
        data["default gradient colormap"] = self.default_gradient_colormap
        data["color mode"] = self.color_mode
        data["gradient by"] = self.gradient_by
        data["gradient species"] = self.gradient_species
        data["gradient scale"] = self.gradient_scale
        data["gradient colormap"] = self.gradient_colormap
        data["gradient colormaps"] = list(self.gradient_colormaps)
        data["gradient colors"] = list(self.gradient_colors)
        data["gradient gamma"] = self.gradient_gamma
        data["gradient reverse"] = self.gradient_reverse
        data["min gradient entries"] = self.min_gradient_entries
        data["colorbar height scale"] = self.colorbar_height_scale
        data["colorbar reverse"] = self.colorbar_reverse
        data["colorbar style"] = self.colorbar_style
        data["colorbar tick length"] = self.colorbar_tick_length
        data["colorbar tick pad"] = self.colorbar_tick_pad
        data["colorbar tick labels"] = self.colorbar_tick_labels
        data["colorbar trace ticks"] = self.colorbar_trace_ticks
        data["title fontsize"] = self.title_fontsize
        data["subtitle fontsize"] = self.subtitle_fontsize
        data["x unit"] = self.x_unit
        data["y unit"] = self.y_unit
        data["x scale"] = self.x_scale
        data["y scale"] = self.y_scale
        data["x axis"] = self.x_axis
        data["y axis"] = self.y_axis
        data["x column index"] = self.x_column_index
        data["y column index"] = self.y_column_index
        data["noise window"] = self.noise_window
        data["noise polyorder"] = self.noise_polyorder
        data["one column"] = self.one_column
        data["cycles"] = self.cycles
        data["plot segment"] = self.plot_segment
        data["plot segments"] = self.plot_segments
        data["segment color mode"] = self.segment_color_mode
        data["segment color groups"] = self.segment_color_groups
        data["plot charge"] = self.plot_charge
        data["plot ca"] = self.plot_ca
        data["plot target"] = self.plot_target
        data["plot quiet time"] = self.plot_quiet_time
        data["target charge"] = self.target_charge
        data["target moles"] = self.target_moles
        data["target electrons"] = self.target_electrons
        data["target label"] = self.target_label
        data["charge color"] = self.charge_color
        data["baseline correction"] = self.baseline_correction
        data["baseline threshold"] = self.baseline_threshold
        data["baseline tail fraction"] = self.baseline_tail_fraction
        data["corrected current"] = self.corrected_current
        data["trace mode"] = self.trace_mode
        data["schedule"] = self.schedule
        data["timing mode"] = self.timing_mode
        data["normalized duration"] = self.normalized_duration
        data["speedup"] = self.speedup
        data["fps"] = self.fps
        data["stagger time"] = self.stagger_time
        data["end hold"] = self.end_hold
        data["loop"] = self.loop
        data["include quiet time"] = self.include_quiet_time
        data["progress"] = self.progress
        data["scan rate"] = self.scan_rate
        data["reference index"] = self.reference_index
        data["reference cv"] = self.reference_cv
        data["reference cvs"] = self.reference_cvs
        data["directional arrows"] = self.directional_arrows
        data["plot options"] = {} if self.plot_options is None else dict(self.plot_options)
        return data


@dataclass(frozen=True, slots=True)
class MultiplotOptions(PlotOptions):
    _provided_options: tuple[str, ...] = field(default=(), repr=False, compare=False)
    print: bool = False
    legend: bool = True
    title: bool | str = "auto"
    subtitle: bool | str | None = "auto"
    labels: list[str] | None = None
    deduplicate_labels: object = False
    legend_bbox_to_anchor: object | None = None
    titles: list[str] | str | None = "auto"
    subtitles: list[str] | str | None = "auto"
    default_discrete_colormap: str = "tab20"
    default_gradient_colormap: str = "viridis"
    color_mode: str = "auto"
    legend_mode: str = "auto"
    min_gradient_entries: int = 3
    colorbar_height_scale: float = 1.0
    colorbar_reverse: bool = True
    colorbar_tick_length: int = 5
    colorbar_tick_pad: int = 8
    gradient_by: str = "auto"
    gradient_species: str = "auto"
    gradient_scale: str = "auto"
    gradient_colormap: str | None = None
    gradient_colormaps: list[str] = field(default_factory=list)
    gradient_colors: list[str] = field(default_factory=list)
    gradient_gamma: float = 1.0
    gradient_reverse: bool = False
    simulation_linestyle: str | None = None
    colorbar_tick_labels: str = "endpoints"
    colorbar_trace_ticks: bool = True

    @classmethod
    def from_options(cls, options=None):
        return _coerce_options(cls, options, ["plot", "multiplot", "normalize_current"])

    def to_options_dict(self):
        data = PlotOptions.to_options_dict(self)
        data["_deduplicate labels explicit"] = "deduplicate_labels" in self._provided_options
        data["labels"] = self.labels
        data["deduplicate labels"] = self.deduplicate_labels
        data["legend loc"] = self.legend_loc
        data["legend outside"] = self.legend_outside
        data["legend pad"] = self.legend_pad
        data["legend bbox to anchor"] = self.legend_bbox_to_anchor
        data["titles"] = self.titles
        data["subtitles"] = self.subtitles
        data["default discrete colormap"] = self.default_discrete_colormap
        data["default gradient colormap"] = self.default_gradient_colormap
        data["color mode"] = self.color_mode
        data["legend mode"] = self.legend_mode
        data["min gradient entries"] = self.min_gradient_entries
        data["colorbar height scale"] = self.colorbar_height_scale
        data["colorbar reverse"] = self.colorbar_reverse
        data["colorbar tick length"] = self.colorbar_tick_length
        data["colorbar tick pad"] = self.colorbar_tick_pad
        data["gradient by"] = self.gradient_by
        data["gradient species"] = self.gradient_species
        data["gradient scale"] = self.gradient_scale
        data["gradient colormap"] = self.gradient_colormap
        data["gradient colormaps"] = list(self.gradient_colormaps)
        data["gradient colors"] = list(self.gradient_colors)
        data["gradient gamma"] = self.gradient_gamma
        data["gradient reverse"] = self.gradient_reverse
        data["simulation linestyle"] = self.simulation_linestyle
        data["colorbar tick labels"] = self.colorbar_tick_labels
        data["colorbar trace ticks"] = self.colorbar_trace_ticks
        return data


@dataclass(frozen=True, slots=True)
class MultiMultiplotOptions(MultiplotOptions):
    titles: list[str] | str | None = "auto"
    subtitles: list[str] | str | None = "auto"
    analysis: bool = False
    ehalf: float | None = None

    @classmethod
    def from_options(cls, options=None):
        return _coerce_options(cls, options, ["plot", "multiplot", "multimultiplot", "normalize_current"])

    def to_options_dict(self):
        data = MultiplotOptions.to_options_dict(self)
        data["titles"] = self.titles
        data["subtitles"] = self.subtitles
        data["analysis"] = self.analysis
        data["Ehalf"] = self.ehalf
        return data


@dataclass(frozen=True, slots=True)
class CVFilterOptions(MultiplotOptions):
    method: str = "savgol"
    column: str = "Current"
    window: int | str = "auto"
    polyorder: int = 3
    sigma: float = 1.0
    size: int = 3
    cutoff: float = 0.1
    order: int = 3
    inplace: bool = False
    plot: bool = False
    print: bool = True

    @classmethod
    def from_options(cls, options=None):
        return _coerce_options(cls, options, ["plot", "multiplot", "cv_filter"])

    @staticmethod
    def _method(value):
        token = str(value).strip().lower().replace("_", " ").replace("-", " ")
        aliases = {
            "savitzky golay": "savgol",
            "savitzkygolay": "savgol",
            "butter": "butterworth",
            "movingaverage": "moving average",
            "mean": "moving average",
        }
        return aliases.get(token, token)

    def validate(self):
        MultiplotOptions.validate(self)
        method = self._method(self.method)
        choices = {"savgol", "gaussian", "median", "butterworth", "moving average"}
        if method not in choices:
            raise OptionError(
                "'method' must be 'savgol', 'gaussian', 'median', "
                "'butterworth', or 'moving average'."
            )
        if not isinstance(self.column, str) or not self.column.strip():
            raise OptionError("'column' must name a data column.")
        if self.window != "auto":
            try:
                window = int(self.window)
            except (TypeError, ValueError) as exc:
                raise OptionError("'window' must be 'auto' or a positive integer.") from exc
            if window <= 0:
                raise OptionError("'window' must be 'auto' or a positive integer.")
        if int(self.polyorder) < 0:
            raise OptionError("'polyorder' must be a non-negative integer.")
        if float(self.sigma) <= 0:
            raise OptionError("'sigma' must be positive.")
        if int(self.size) <= 0:
            raise OptionError("'size' must be a positive integer.")
        if not 0 < float(self.cutoff) < 1:
            raise OptionError("'cutoff' must be between 0 and 1 as a fraction of Nyquist.")
        if int(self.order) <= 0:
            raise OptionError("'order' must be a positive integer.")

    def to_options_dict(self):
        data = MultiplotOptions.to_options_dict(self)
        data.update(
            {
                "method": self._method(self.method),
                "column": self.column,
                "window": self.window,
                "polyorder": int(self.polyorder),
                "sigma": float(self.sigma),
                "size": int(self.size),
                "cutoff": float(self.cutoff),
                "order": int(self.order),
                "inplace": self.inplace,
                "plot": self.plot,
                "print": self.print,
            }
        )
        return data


@dataclass(frozen=True, slots=True)
class MultiScatterplotOptions(MultiplotOptions):
    print: bool = True
    x_column: str | int | None = "auto"
    y_column: str | int | None = "auto"
    y_columns: object | None = None
    metric: str | None = None
    x_mode: str | None = None
    y_mode: str | None = None
    transform_mode: str | None = None
    x_transform: str | float | int | None = None
    y_transform: str | float | int | None = None
    floor: bool | str | float | int | None = None
    x_floor: bool | str | float | int | None = None
    y_floor: bool | str | float | int | None = None
    y0: object | None = None
    fit: bool | str | None = True
    fit_model: object | None = "linear"
    fit_params: object | None = None
    fit_init: object | None = "auto"
    fit_bounds: object | None = "auto"
    fit_residual: str = "direct"
    fit_max_evals: int = 10000
    fit_method: str | None = "auto"
    fit_sigma: object | None = None
    fit_absolute_sigma: bool = False
    fit_check_finite: bool | None = None
    fit_nan_policy: str | None = None
    fit_jac: object | None = None
    curve_fit_options: dict | None = None
    sig_figs: int = 4
    plot_style: str = "scatter"
    xscale: str | None = None
    yscale: str | None = None
    plot_scale: str | None = None
    plot_fit: bool = True
    fit_band: str | None = None
    fit_band_level: float = 0.95
    fit_line_range: object | None = None
    fit_linestyle: str = "--"
    fit_linewidth: int | float = 1
    fit_alpha: int | float = 1

    @classmethod
    def from_options(cls, options=None):
        return _coerce_options(cls, options, ["plot", "multiplot", "multi_scatterplot"])

    def validate(self):
        style = str(self.plot_style).strip().lower()
        if style not in {"scatter", "line", "line+markers", "line markers", "line-and-markers"}:
            raise OptionError("'plot style' must be 'scatter', 'line', or 'line+markers'.")
        if self.plot_scale is not None:
            scale = str(self.plot_scale).strip().lower().replace("_", "-").replace(" ", "-")
            if scale not in {"linear", "log-log", "loglog", "semilogx", "semi-log-x", "semilogy", "semi-log-y", "symlog"}:
                raise OptionError("'plot scale' must be 'linear', 'log-log', 'semilogx', 'semilogy', or 'symlog'.")
        if not 0 <= float(self.fit_alpha) <= 1:
            raise OptionError("'fit alpha' must be between 0 and 1.")
        if float(self.fit_linewidth) < 0:
            raise OptionError("'fit linewidth' must be non-negative.")
        _validate_fit_band_options(self)

    def to_options_dict(self):
        data = MultiplotOptions.to_options_dict(self)
        data["x column"] = self.x_column
        data["y column"] = self.y_column
        data["y columns"] = self.y_columns
        data["metric"] = self.metric
        data["x mode"] = self.x_mode
        data["y mode"] = self.y_mode
        data["transform mode"] = self.transform_mode
        data["x transform"] = self.x_transform
        data["y transform"] = self.y_transform
        data["floor"] = self.floor
        data["x floor"] = self.x_floor
        data["y floor"] = self.y_floor
        data["y0"] = self.y0
        data["fit"] = self.fit
        data["fit model"] = self.fit_model
        data["fit params"] = self.fit_params
        data["fit init"] = self.fit_init
        data["fit bounds"] = self.fit_bounds
        data["fit residual"] = self.fit_residual
        data["fit max evals"] = self.fit_max_evals
        data["fit method"] = self.fit_method
        data["fit sigma"] = self.fit_sigma
        data["fit absolute sigma"] = self.fit_absolute_sigma
        data["fit check finite"] = self.fit_check_finite
        data["fit nan policy"] = self.fit_nan_policy
        data["fit jac"] = self.fit_jac
        data["curve fit options"] = self.curve_fit_options
        data["sig figs"] = self.sig_figs
        data["plot style"] = self.plot_style
        data["xscale"] = self.xscale
        data["yscale"] = self.yscale
        data["plot scale"] = self.plot_scale
        data["plot fit"] = self.plot_fit
        data["fit band"] = self.fit_band
        data["fit band level"] = self.fit_band_level
        data["fit line range"] = self.fit_line_range
        data["fit linestyle"] = self.fit_linestyle
        data["fit linewidth"] = self.fit_linewidth
        data["fit alpha"] = self.fit_alpha
        return data


@dataclass(frozen=True, slots=True)
class PeakPotentialOptions:
    plot: bool = True
    print: bool = True
    pretty_print: bool = True
    plot_all: bool = False
    print_all: bool = False
    x_axis: str | None = None
    y_axis: str | None = None
    x_unit: str | None = "auto"
    y_unit: str | None = "auto"
    xlabel: str | None = None
    ylabel: str | None = None
    new_plot: bool = False
    plot_cv: bool = True
    derivative: int | float | None = 0
    plot_segment: int | None = None
    plot_segments: list[int] | int | None = None
    segment: int | None = None
    segments: list[int] | None = None
    noise_window: int | str | None = "auto"
    noise_polyorder: int | str = "auto"
    sig_figs: int = 4
    peak_prominence: float | None = None
    peak_kind: str | None = "both"
    guess_potential: float | list[float] | list[list[float]] | None = None
    exact_potential: float | list[float] | None = None
    troubleshoot: bool = False
    internal_call: bool = False
    offset: float = 0

    @classmethod
    def from_options(cls, options=None):
        return _coerce_options(cls, options, ["plot", "cv_selection", "cv_analysis"])

    def validate(self):
        _validate_common_cv(self)

    def to_options_dict(self):
        data = {field.name.replace("_", " "): getattr(self, field.name) for field in fields(self)}
        data["sig figs"] = self.sig_figs
        data["peak prominence"] = self.peak_prominence
        data["peak kind"] = self.peak_kind
        data["guess potential"] = self.guess_potential
        data["exact potential"] = self.exact_potential
        data["noise window"] = self.noise_window
        data["noise polyorder"] = self.noise_polyorder
        data["x axis"] = self.x_axis
        data["y axis"] = self.y_axis
        data["x unit"] = self.x_unit
        data["y unit"] = self.y_unit
        data["new plot"] = self.new_plot
        data["plot segment"] = self.plot_segment
        data["plot segments"] = self.plot_segments
        data["internal call"] = self.internal_call
        data["pretty print"] = self.pretty_print
        data["plot all"] = self.plot_all
        data["print all"] = self.print_all
        return data


@dataclass(frozen=True, slots=True)
class PeakCurrentOptions(PeakPotentialOptions):
    tangent_range: str | float | list[float] | tuple[float, float] = "auto"
    tangent_min_points: int | None = None
    tangent_potential: float | list[float] | None = None
    percent_threshold: float | None = None
    plot_peak_potential: bool = True
    peak_fallback: str | None = "highest current"

    @classmethod
    def from_options(cls, options=None):
        return _coerce_options(cls, options, ["plot", "cv_selection", "cv_analysis", "peak_current"])

    def validate(self):
        _validate_common_cv(self)
        if self.peak_fallback is None:
            return
        fallback = str(self.peak_fallback).strip().lower().replace("_", " ").replace("-", " ")
        valid = {
            "highest current",
            "highest absolute current",
            "max current",
            "max absolute current",
            "abs current",
            "guess potential",
            "exact potential",
            "none",
            "error",
            "raise",
        }
        if fallback not in valid:
            raise OptionError(
                "'peak fallback' must be 'highest current', 'guess potential', or None/'none'."
            )

    def for_peak_potential(self):
        return PeakPotentialOptions(
            **{
                field.name: getattr(self, field.name)
                for field in fields(PeakPotentialOptions)
            }
        )

    def to_options_dict(self):
        data = PeakPotentialOptions.to_options_dict(self)
        data["tangent range"] = self.tangent_range
        data["tangent min points"] = self.tangent_min_points
        data["tangent potential"] = self.tangent_potential
        data["percent threshold"] = self.percent_threshold
        data["plot peak potential"] = self.plot_peak_potential
        data["peak fallback"] = self.peak_fallback
        return data


@dataclass(frozen=True, slots=True)
class FitModelOptions:
    plot: bool = False
    print: bool = True
    pretty_print: bool = True
    new_plot: bool = True
    x_column: str | int | None = "auto"
    y_column: str | int | None = "auto"
    fit: bool = True
    fit_model: object | None = "linear"
    fit_params: object | None = None
    fit_init: object | None = "auto"
    fit_bounds: object | None = "auto"
    fit_residual: str = "direct"
    fit_max_evals: int = 10000
    fit_method: str | None = "auto"
    fit_sigma: object | None = None
    fit_absolute_sigma: bool = False
    fit_check_finite: bool | None = None
    fit_nan_policy: str | None = None
    fit_jac: object | None = None
    curve_fit_options: dict | None = None
    fit_indices: object | None = None
    print_fit: str = "auto"
    print_fit_details: bool = False
    sig_figs: int = 4
    plot_fit: bool = True
    plot_data: bool = True
    fit_band: str | None = None
    fit_band_level: float = 0.95
    fit_line_range: object | None = None
    legend: bool = False
    legend_fontsize: int | float | None = None
    fit_label: bool | str = False
    fit_color: object = "tab:red"
    fit_linestyle: str = "--"
    fit_linewidth: int | float = 1
    fit_alpha: int | float = 1

    @classmethod
    def from_options(cls, options=None):
        return _coerce_options(cls, options, ["fit_model"])

    def validate(self):
        if not 0 <= float(self.fit_alpha) <= 1:
            raise OptionError("'fit alpha' must be between 0 and 1.")
        if float(self.fit_linewidth) < 0:
            raise OptionError("'fit linewidth' must be non-negative.")
        _validate_fit_band_options(self)

    def to_options_dict(self):
        data = {field.name.replace("_", " "): getattr(self, field.name) for field in fields(self)}
        data["pretty print"] = self.pretty_print
        data["new plot"] = self.new_plot
        data["x column"] = self.x_column
        data["y column"] = self.y_column
        data["fit model"] = self.fit_model
        data["fit params"] = self.fit_params
        data["fit init"] = self.fit_init
        data["fit bounds"] = self.fit_bounds
        data["fit residual"] = self.fit_residual
        data["fit max evals"] = self.fit_max_evals
        data["fit method"] = self.fit_method
        data["fit sigma"] = self.fit_sigma
        data["fit absolute sigma"] = self.fit_absolute_sigma
        data["fit check finite"] = self.fit_check_finite
        data["fit nan policy"] = self.fit_nan_policy
        data["fit jac"] = self.fit_jac
        data["curve fit options"] = self.curve_fit_options
        data["fit indices"] = self.fit_indices
        data["print fit"] = self.print_fit
        data["print fit details"] = self.print_fit_details
        data["plot fit"] = self.plot_fit
        data["plot data"] = self.plot_data
        data["fit band"] = self.fit_band
        data["fit band level"] = self.fit_band_level
        data["fit line range"] = self.fit_line_range
        data["legend fontsize"] = self.legend_fontsize
        data["fit label"] = self.fit_label
        data["fit color"] = self.fit_color
        data["fit linestyle"] = self.fit_linestyle
        data["fit linewidth"] = self.fit_linewidth
        data["fit alpha"] = self.fit_alpha
        return data


@dataclass(frozen=True, slots=True)
class NormalizeOptions:
    print: bool = False
    pretty_print: bool = True
    mode: str = "homogeneous"
    e0: float | None = None
    n: float = 1
    num_electrons: float | None = None
    temperature: float | None = None
    d: float | None = None
    c: float | str | None = None
    c_unit: str | None = None
    species: str | None = None
    area: float | None = None
    s: float | None = None
    electrode_area: float | None = None
    scan_rate: float | None = None
    v: float | None = None
    k_homo: float | None = None
    k0: float | None = None

    @classmethod
    def from_options(cls, options=None):
        return _coerce_options(cls, options, ["normalize"])

    def validate(self):
        mode = str(self.mode).strip().lower()
        if mode not in {"homogeneous", "heterogeneous"}:
            raise OptionError("'mode' must be 'homogeneous' or 'heterogeneous'.")
        n_value = self.num_electrons if self.num_electrons is not None else self.n
        if n_value is None or float(n_value) <= 0:
            raise OptionError("'n' must be positive.")

    def to_options_dict(self):
        data = {
            field.name.replace("_", " "): getattr(self, field.name)
            for field in fields(self)
            if not field.name.startswith("_")
        }
        data["E0"] = self.e0
        data["D"] = self.d
        data["C"] = self.c
        data["C unit"] = self.c_unit
        data["electrode area"] = self.electrode_area
        data["scan rate"] = self.scan_rate
        data["num electrons"] = self.num_electrons
        data["k homo"] = self.k_homo
        return data


@dataclass(frozen=True, slots=True)
class NormalizationOptions:
    print: bool = True
    plot_all: bool = False
    plot_reference_diagnostic: bool = False
    pretty_print: bool = True
    print_conditions: bool = True
    legend: bool = True
    title: bool | str = "auto"
    subtitle: bool | str | None = "auto"
    labels: list[str] | None = None
    segment: int | None = None
    segments: list[int] | int | None = None
    noise_window: int | str | None = "auto"
    noise_polyorder: int | str = "auto"
    sig_figs: int = 4
    peak_prominence: float | None = None
    guess_potential: float | None = None
    exact_potential: float | None = None
    tangent_range: str | float | list[float] | tuple[float, float] = "auto"
    tangent_min_points: int | None = None
    tangent_potential: float | None = None
    percent_threshold: float | None = None
    ip0: float | list[float] | None = None
    reference_index: int = 0
    reference_cv: object | None = None
    reference_cvs: object | None = None
    reference_guess_potential: float | None = None

    @classmethod
    def from_options(cls, options=None):
        return _coerce_options(cls, options, ["cv_selection", "cv_analysis", "peak_current", "normalize_current"])

    def validate(self):
        _validate_common_cv(self)
        if not isinstance(self.reference_index, int):
            raise OptionError("'reference index' must be an integer.")
        sources = [
            self.ip0 is not None,
            self.reference_cv is not None,
            self.reference_cvs is not None,
            self.reference_index != 0,
        ]
        if sum(bool(source) for source in sources) > 1:
            raise OptionError("Use only one ip0/reference source for normalize_current.")

    def for_peak_current(self):
        return PeakCurrentOptions.from_options(
            {
                "plot": False,
                "print": False,
                "segment": self.segment,
                "segments": self.segments,
                "noise window": self.noise_window,
                "noise polyorder": self.noise_polyorder,
                "sig figs": self.sig_figs,
                "peak prominence": self.peak_prominence,
                "guess potential": (
                    self.reference_guess_potential
                    if self.reference_guess_potential is not None
                    else self.guess_potential
                ),
                "exact potential": self.exact_potential,
                "tangent range": self.tangent_range,
                "tangent min points": self.tangent_min_points,
                "tangent potential": self.tangent_potential,
                "percent threshold": self.percent_threshold,
                "internal call": True,
            }
        )

    def to_options_dict(self):
        data = {field.name.replace("_", " "): getattr(self, field.name) for field in fields(self)}
        data["print"] = self.print
        data["plot all"] = self.plot_all
        data["plot reference diagnostic"] = self.plot_reference_diagnostic
        data["pretty print"] = self.pretty_print
        data["print conditions"] = self.print_conditions
        data["legend"] = self.legend
        data["title"] = self.title
        data["subtitle"] = self.subtitle
        data["labels"] = self.labels
        data["noise window"] = self.noise_window
        data["noise polyorder"] = self.noise_polyorder
        data["sig figs"] = self.sig_figs
        data["peak prominence"] = self.peak_prominence
        data["guess potential"] = self.guess_potential
        data["exact potential"] = self.exact_potential
        data["tangent range"] = self.tangent_range
        data["tangent min points"] = self.tangent_min_points
        data["tangent potential"] = self.tangent_potential
        data["percent threshold"] = self.percent_threshold
        data["reference index"] = self.reference_index
        data["reference cv"] = self.reference_cv
        data["reference cvs"] = self.reference_cvs
        data["reference guess potential"] = self.reference_guess_potential
        return data


@dataclass(frozen=True, slots=True)
class ScaleCurrentOptions:
    print: bool = False
    plot_all: bool = False
    pretty_print: bool = True
    print_conditions: bool = True
    scale: float | list[float] | None = None
    reference_mode: str = "single"
    reference_index: int = 0
    reference_cv: object | None = None
    reference_cvs: object | None = None
    segment: int | None = 1
    segments: list[int] | int | None = None
    noise_window: int | str | None = "auto"
    noise_polyorder: int | str = "auto"
    sig_figs: int = 4
    peak_prominence: float | None = None
    guess_potential: float | None = None
    exact_potential: float | None = None
    tangent_range: str | float | list[float] | tuple[float, float] = "auto"
    tangent_min_points: int | None = None
    tangent_potential: float | None = None
    percent_threshold: float | None = None

    @classmethod
    def from_options(cls, options=None):
        return _coerce_options(cls, options, ["cv_analysis", "peak_current", "scale_current"])

    def validate(self):
        _validate_common_cv(self)
        if str(self.reference_mode).strip().lower() not in {"single", "both"}:
            raise OptionError("'reference mode' must be 'single' or 'both' for scale_current.")
        if not isinstance(self.reference_index, int):
            raise OptionError("'reference index' must be an integer.")
        sources = [
            self.scale is not None,
            self.reference_cv is not None,
            self.reference_cvs is not None,
            self.reference_index != 0,
        ]
        if sum(bool(source) for source in sources) > 1:
            raise OptionError("Use only one scale/reference source for scale_current.")

    def for_peak_current(self, segment=None):
        return PeakCurrentOptions.from_options(
            {
                "plot": self.plot_all,
                "plot all": self.plot_all,
                "print": False,
                "print all": False,
                "segment": self.segment if segment is None else segment,
                "noise window": self.noise_window,
                "noise polyorder": self.noise_polyorder,
                "sig figs": self.sig_figs,
                "peak prominence": self.peak_prominence,
                "guess potential": self.guess_potential,
                "exact potential": self.exact_potential,
                "tangent range": self.tangent_range,
                "tangent min points": self.tangent_min_points,
                "tangent potential": self.tangent_potential,
                "percent threshold": self.percent_threshold,
            }
        )

    def to_options_dict(self):
        data = {field.name.replace("_", " "): getattr(self, field.name) for field in fields(self)}
        data["pretty print"] = self.pretty_print
        data["plot all"] = self.plot_all
        data["print conditions"] = self.print_conditions
        data["scale"] = self.scale
        data["reference mode"] = str(self.reference_mode).strip().lower()
        data["reference index"] = self.reference_index
        data["reference cv"] = self.reference_cv
        data["reference cvs"] = self.reference_cvs
        data["noise window"] = self.noise_window
        data["noise polyorder"] = self.noise_polyorder
        data["sig figs"] = self.sig_figs
        data["peak prominence"] = self.peak_prominence
        data["guess potential"] = self.guess_potential
        data["exact potential"] = self.exact_potential
        data["tangent range"] = self.tangent_range
        data["tangent min points"] = self.tangent_min_points
        data["tangent potential"] = self.tangent_potential
        data["percent threshold"] = self.percent_threshold
        return data


@dataclass(frozen=True, slots=True)
class FOWAOptions:
    plot: bool = True
    print: bool = True
    pretty_print: bool = True
    plot_all: bool = False
    print_all: bool = False
    legend: bool = False
    legend_fontsize: int | float | None = None
    legend_loc: str = "best"
    legend_outside: bool = False
    legend_pad: float = 0.02
    legend_bbox_to_anchor: object | None = None
    legend_mode: str = "auto"
    min_gradient_entries: int = 3
    colorbar_height_scale: float = 1.0
    title: bool | str = True
    offset: float = 0
    color: str = "black"
    labels: list[str] | None = None
    x_axis: str | None = None
    y_axis: str | None = None
    x_unit: str | None = "auto"
    y_unit: str | None = "auto"
    new_plot: bool = False
    segment: int | None = None
    segments: list[int] | int | None = None
    plot_segments: list[int] | int | None = None
    noise_window: int | str | None = "auto"
    noise_polyorder: int | str = "auto"
    sig_figs: int = 4
    peak_prominence: float | None = None
    guess_potential: float | list[float] | list[list[float]] | None = None
    exact_potential: float | list[float] | None = None
    tangent_range: str | float | list[float] | tuple[float, float] = "auto"
    tangent_min_points: int | None = None
    tangent_potential: float | list[float] | None = None
    percent_threshold: float | None = None
    peak_fallback: str | None = "highest current"
    peak_potential: float | list[float] | None = None
    troubleshoot: bool = False
    fit: bool = True
    fit_basis: str = "x"
    fit_range: list[float] | tuple[float, float] = (0.0, 0.2)
    wave_range: list[float] | tuple[float, float] | None = None
    plot_fit: bool = True
    fit_label: bool | str = False
    fit_color: object | None = None
    fit_linestyle: str = "--"
    fit_linewidth: int | float = 1.5
    fit_alpha: int | float = 1
    min_r2: float = 0.98
    redox_mode: str = "half wave"
    min_fit_points: int = 20
    background_correction: str | None = "tangent"
    diagnostic_y_axis: str = "i/ip0"
    ecat_shift_warning_threshold: float | bool | None = 0.05
    warnings: bool = True
    mechanism: str = "EC'"
    custom_formula: object | None = None
    formula_label: str | None = None
    catalyst_electrons: float = 1
    turnover_electrons: float = 1
    sigma: float = 1.0
    gaussian_weight: float = 0.4
    gaussian_skew: float = -3
    redox_potential: float | list[float] | None = None
    ip0: float | list[float] | None = None
    non_catalytic_current: float | None = None
    non_catalytic_cv: object | None = None
    non_catalytic_cvs: object | None = None
    non_catalytic_guess_potential: float | list[float] | None = None

    @classmethod
    def from_options(cls, options=None):
        return _coerce_options(
            cls,
            options,
            ["plot", "multiplot", "cv_selection", "cv_analysis", "peak_current", "fowa"],
        )

    def validate(self):
        if isinstance(self.fit_basis, str):
            fit_basis_values = [self.fit_basis]
        elif isinstance(self.fit_basis, (list, tuple)):
            fit_basis_values = list(self.fit_basis)
        else:
            raise OptionError("'fit basis' must be 'x', 'y', or a list of those values.")

        bad_fit_basis = [value for value in fit_basis_values if str(value).strip().lower() not in {"x", "y"}]
        if bad_fit_basis:
            raise OptionError("'fit basis' must be 'x' or 'y'.")
        redox_mode = str(self.redox_mode).strip().lower()
        if redox_mode not in {"half wave", "half peak", "manual"}:
            raise OptionError("'redox mode' must be 'half wave', 'half peak', or 'manual'.")
        if self.redox_potential is not None and redox_mode != "manual":
            raise OptionError("Use 'redox mode': 'manual' when providing 'redox potential'.")
        diagnostic_y_axis = str(self.diagnostic_y_axis).strip().lower()
        if diagnostic_y_axis not in {"i/ip0", "current"}:
            raise OptionError("'diagnostic y axis' must be 'i/ip0' or 'current'.")
        if self.peak_fallback is not None:
            fallback = str(self.peak_fallback).strip().lower().replace("_", " ").replace("-", " ")
            valid = {
                "highest current",
                "highest absolute current",
                "max current",
                "max absolute current",
                "abs current",
                "guess potential",
                "exact potential",
                "none",
                "error",
                "raise",
            }
            if fallback not in valid:
                raise OptionError(
                    "'peak fallback' must be 'highest current', 'guess potential', or None/'none'."
                )
        if self.ip0 is not None and self.non_catalytic_current is not None:
            raise OptionError("Use either 'ip0' or 'non-catalytic current', not both.")
        if self.non_catalytic_cv is not None and self.non_catalytic_cvs is not None:
            raise OptionError("Use either 'non-catalytic cv' or 'non-catalytic cvs', not both.")
        _validate_common_cv(self)

    def for_peak_current(self):
        return PeakCurrentOptions.from_options({
            "plot": False,
            "print": False,
            "plot all": self.plot_all,
            "print all": self.print_all,
        })

    def to_options_dict(self):
        data = {field.name.replace("_", " "): getattr(self, field.name) for field in fields(self)}
        data["plot all"] = self.plot_all
        data["print all"] = self.print_all
        data["pretty print"] = self.pretty_print
        data["labels"] = self.labels
        data["colorbar height scale"] = self.colorbar_height_scale
        data["min gradient entries"] = self.min_gradient_entries
        data["new plot"] = self.new_plot
        data["noise window"] = self.noise_window
        data["noise polyorder"] = self.noise_polyorder
        data["sig figs"] = self.sig_figs
        data["peak prominence"] = self.peak_prominence
        data["guess potential"] = self.guess_potential
        data["exact potential"] = self.exact_potential
        data["tangent range"] = self.tangent_range
        data["tangent min points"] = self.tangent_min_points
        data["tangent potential"] = self.tangent_potential
        data["percent threshold"] = self.percent_threshold
        data["peak fallback"] = self.peak_fallback
        data["peak potential"] = self.peak_potential
        data["fit"] = self.fit
        data["fit basis"] = self.fit_basis
        data["fit range"] = self.fit_range
        data["wave range"] = self.wave_range
        data["min fit points"] = self.min_fit_points
        data["min r2"] = self.min_r2
        data["redox mode"] = str(self.redox_mode).strip().lower()
        data["background correction"] = self.background_correction
        data["diagnostic y axis"] = str(self.diagnostic_y_axis).strip().lower()
        data["ecat shift warning threshold"] = self.ecat_shift_warning_threshold
        data["warnings"] = self.warnings
        data["custom formula"] = self.custom_formula
        data["formula label"] = self.formula_label
        data["catalyst electrons"] = self.catalyst_electrons
        data["turnover electrons"] = self.turnover_electrons
        data["gaussian weight"] = self.gaussian_weight
        data["gaussian skew"] = self.gaussian_skew
        data["redox potential"] = self.redox_potential
        data["non-catalytic current"] = self.non_catalytic_current
        data["non-catalytic cv"] = self.non_catalytic_cv
        data["non-catalytic cvs"] = self.non_catalytic_cvs
        data["non-catalytic guess potential"] = self.non_catalytic_guess_potential
        return data


@dataclass(frozen=True, slots=True)
class PlateauCurrentOptions(FOWAOptions):
    ilim: float | list[float] | None = None
    ic: float | list[float] | None = None
    ip0_scan_rate: float | None = None
    ip0_sqrt_scan_rate_slope: float | None = None
    plateau_slope_tolerance: float = 0.10
    plateau_min_cvs: int = 2
    plateau_average_method: str = "mean"
    plateau_selection_mode: str = "high scan suffix"
    validate_plateau: bool = True
    require_plateau: bool = True
    formula_mode: str = "auto"
    c: float | str | None = None
    c_unit: str | None = None
    species: str | None = None
    d: float | None = None
    electrode_area: float | None = None
    scan_rate: float | None = None
    temperature: float = 298
    warn_ir_drop: bool = True
    group_mode: str = "auto"
    group_by: object = "species"

    @classmethod
    def from_options(cls, options=None):
        return _coerce_options(
            cls,
            options,
            ["plot", "cv_selection", "cv_analysis", "peak_current", "fowa", "plateau_current"],
        )

    def validate(self):
        FOWAOptions.validate(self)
        if self.ilim is not None and self.ic is not None:
            raise OptionError("Use either 'ilim' or 'ic', not both.")
        if float(self.plateau_slope_tolerance) < 0:
            raise OptionError("'plateau slope tolerance' must be non-negative.")
        if int(self.plateau_min_cvs) < 1:
            raise OptionError("'plateau min cvs' must be at least 1.")
        if str(self.plateau_average_method).strip().lower() not in {"mean", "median"}:
            raise OptionError("'plateau average method' must be 'mean' or 'median'.")
        if str(self.plateau_selection_mode).strip().lower() != "high scan suffix":
            raise OptionError("Only 'high scan suffix' plateau selection mode is currently supported.")
        group_mode = str(self.group_mode).strip().lower().replace("_", " ").replace("-", " ")
        if group_mode not in {"auto", "as given", "each"}:
            raise OptionError("'group mode' must be 'auto', 'as given', or 'each'.")
        mode = str(self.formula_mode).strip().lower()
        if mode not in {"auto", "normalized", "slope normalized", "slope-normalized", "direct"}:
            raise OptionError("'formula mode' must be 'auto', 'normalized', 'slope normalized', or 'direct'.")
        if self.d is not None and float(self.d) <= 0:
            raise OptionError("'D' must be positive when provided.")
        if self.electrode_area is not None and float(self.electrode_area) <= 0:
            raise OptionError("'electrode area' must be positive when provided.")

    def to_options_dict(self):
        data = FOWAOptions.to_options_dict(self)
        data["ilim"] = self.ilim
        data["ic"] = self.ic
        data["ip0 scan rate"] = self.ip0_scan_rate
        data["ip0 sqrt scan rate slope"] = self.ip0_sqrt_scan_rate_slope
        data["plateau slope tolerance"] = self.plateau_slope_tolerance
        data["plateau min cvs"] = self.plateau_min_cvs
        data["plateau average method"] = self.plateau_average_method
        data["plateau selection mode"] = self.plateau_selection_mode
        data["validate plateau"] = self.validate_plateau
        data["require plateau"] = self.require_plateau
        data["formula mode"] = self.formula_mode
        data["C"] = self.c
        data["C unit"] = self.c_unit
        data["species"] = self.species
        data["D"] = self.d
        data["electrode area"] = self.electrode_area
        data["scan rate"] = self.scan_rate
        data["temperature"] = self.temperature
        data["warn ir drop"] = self.warn_ir_drop
        data["group mode"] = str(self.group_mode).strip().lower().replace("_", " ").replace("-", " ")
        data["group by"] = self.group_by
        return data


@dataclass(frozen=True, slots=True)
class FitRateOptions:
    plot: bool = True
    print: bool = True
    print_all: bool = False
    fit: bool = True
    metric: str = "kobs"
    x_column: str | None = "auto"
    species: str | None = None
    x_transform: str | float | int | None = None
    y_transform: str | float | int | None = None
    transform_mode: str | None = None
    floor: bool | str | float | int | None = None
    x_floor: bool | str | float | int | None = None
    y_floor: bool | str | float | int | None = None
    y_mode: str | None = None
    y0: object | None = None
    xscale: str | None = None
    yscale: str | None = None
    plot_scale: str | None = None
    plot_log_log: bool = False
    fit_indices: object | None = None
    log_fit_indices: object | None = None
    fit_model: object | None = "linear"
    fit_params: object | None = None
    fit_init: object | None = "auto"
    fit_bounds: object | None = "auto"
    fit_residual: str = "direct"
    fit_max_evals: int = 10000
    fit_method: str | None = "auto"
    fit_sigma: object | None = None
    fit_absolute_sigma: bool = False
    fit_check_finite: bool | None = None
    fit_nan_policy: str | None = None
    fit_jac: object | None = None
    curve_fit_options: dict | None = None
    print_fit: str = "auto"
    print_fit_details: bool = False
    sig_figs: int = 4
    plot_fit: bool = True
    fit_band: str | None = None
    fit_band_level: float = 0.95
    fit_line_range: object | None = None
    exclude_warnings: bool = False
    exclude_low_r2: bool = False
    min_r2: float = 0.95
    local_slope_mode: str = "adjacent"
    print_local_slopes: bool = False
    plot_local_slopes: bool = False
    legend: bool = False
    legend_fontsize: int | float | None = None
    fit_label: bool | str = False
    fit_color: object = "tab:red"
    fit_linestyle: str = "--"
    fit_linewidth: int | float = 1
    fit_alpha: int | float = 1
    return_stats: bool = False

    @classmethod
    def from_options(cls, options=None):
        return _coerce_options(cls, options, ["fit_rate"])

    def validate(self):
        mode = str(self.local_slope_mode).strip().lower()
        if mode not in {"adjacent", "gradient"}:
            raise OptionError("'local slope mode' must be 'adjacent' or 'gradient'.")
        if self.plot_scale is not None:
            scale = str(self.plot_scale).strip().lower().replace("_", "-").replace(" ", "-")
            if scale not in {"linear", "log-log", "loglog", "semilogx", "semi-log-x", "semilogy", "semi-log-y", "symlog"}:
                raise OptionError("'plot scale' must be 'linear', 'log-log', 'semilogx', 'semilogy', or 'symlog'.")
        if not 0 <= float(self.min_r2) <= 1:
            raise OptionError("'min r2' must be between 0 and 1.")
        if not 0 <= float(self.fit_alpha) <= 1:
            raise OptionError("'fit alpha' must be between 0 and 1.")
        if float(self.fit_linewidth) < 0:
            raise OptionError("'fit linewidth' must be non-negative.")
        _validate_fit_band_options(self)

    def to_options_dict(self):
        data = {field.name.replace("_", " "): getattr(self, field.name) for field in fields(self)}
        data["x column"] = self.x_column
        data["x transform"] = self.x_transform
        data["y transform"] = self.y_transform
        data["transform mode"] = self.transform_mode
        data["floor"] = self.floor
        data["x floor"] = self.x_floor
        data["y floor"] = self.y_floor
        data["y mode"] = self.y_mode
        data["y0"] = self.y0
        data["xscale"] = self.xscale
        data["yscale"] = self.yscale
        data["plot scale"] = self.plot_scale
        data["plot log-log"] = self.plot_log_log
        data["fit indices"] = self.fit_indices
        data["log fit indices"] = self.log_fit_indices
        data["fit model"] = self.fit_model
        data["fit params"] = self.fit_params
        data["fit init"] = self.fit_init
        data["fit bounds"] = self.fit_bounds
        data["fit residual"] = self.fit_residual
        data["fit max evals"] = self.fit_max_evals
        data["fit method"] = self.fit_method
        data["fit sigma"] = self.fit_sigma
        data["fit absolute sigma"] = self.fit_absolute_sigma
        data["fit check finite"] = self.fit_check_finite
        data["fit nan policy"] = self.fit_nan_policy
        data["fit jac"] = self.fit_jac
        data["curve fit options"] = self.curve_fit_options
        data["print fit"] = self.print_fit
        data["print fit details"] = self.print_fit_details
        data["sig figs"] = self.sig_figs
        data["plot fit"] = self.plot_fit
        data["fit band"] = self.fit_band
        data["fit band level"] = self.fit_band_level
        data["fit line range"] = self.fit_line_range
        data["exclude warnings"] = self.exclude_warnings
        data["exclude low r2"] = self.exclude_low_r2
        data["min r2"] = self.min_r2
        data["local slope mode"] = self.local_slope_mode
        data["print local slopes"] = self.print_local_slopes
        data["plot local slopes"] = self.plot_local_slopes
        data["legend"] = self.legend
        data["legend fontsize"] = self.legend_fontsize
        data["fit label"] = self.fit_label
        data["fit color"] = self.fit_color
        data["fit linestyle"] = self.fit_linestyle
        data["fit linewidth"] = self.fit_linewidth
        data["fit alpha"] = self.fit_alpha
        data["return stats"] = self.return_stats
        return data


@dataclass(frozen=True, slots=True)
class FitPeakPotentialOptions(PeakPotentialOptions):
    follow_e1_2: bool = False
    fit: bool = True
    x_unit: str | None = "auto"
    y_unit: str | None = "auto"
    species: str | None = None
    y_mode: str | None = None
    y0: object | None = None
    fit_indices: object | None = None
    fit_model: object | None = "linear"
    fit_params: object | None = None
    fit_init: object | None = "auto"
    fit_bounds: object | None = "auto"
    fit_residual: str = "direct"
    fit_max_evals: int = 10000
    fit_method: str | None = "auto"
    fit_sigma: object | None = None
    fit_absolute_sigma: bool = False
    fit_check_finite: bool | None = None
    fit_nan_policy: str | None = None
    fit_jac: object | None = None
    curve_fit_options: dict | None = None
    print_fit: str = "auto"
    print_fit_details: bool = False
    sig_figs: int = 6
    plot_fit: bool = True
    fit_band: str | None = None
    fit_band_level: float = 0.95
    fit_line_range: object | None = None
    legend: bool = False
    legend_fontsize: int | float | None = None
    fit_label: bool | str = False
    fit_color: object = "tab:red"
    fit_linestyle: str = "--"
    fit_linewidth: int | float = 1
    fit_alpha: int | float = 1
    return_stats: bool = False

    @classmethod
    def from_options(cls, options=None):
        return _coerce_options(cls, options, ["plot", "cv_selection", "cv_analysis", "fit_peak_potential"])

    def validate(self):
        _validate_common_cv(self)
        if not 0 <= float(self.fit_alpha) <= 1:
            raise OptionError("'fit alpha' must be between 0 and 1.")
        _validate_fit_band_options(self)

    def to_options_dict(self):
        data = PeakPotentialOptions.to_options_dict(self)
        data["follow e1/2"] = self.follow_e1_2
        data["fit"] = self.fit
        data["x unit"] = self.x_unit
        data["y unit"] = self.y_unit
        data["species"] = self.species
        data["y mode"] = self.y_mode
        data["y0"] = self.y0
        data["fit indices"] = self.fit_indices
        data["fit model"] = self.fit_model
        data["fit params"] = self.fit_params
        data["fit init"] = self.fit_init
        data["fit bounds"] = self.fit_bounds
        data["fit residual"] = self.fit_residual
        data["fit max evals"] = self.fit_max_evals
        data["fit method"] = self.fit_method
        data["fit sigma"] = self.fit_sigma
        data["fit absolute sigma"] = self.fit_absolute_sigma
        data["fit check finite"] = self.fit_check_finite
        data["fit nan policy"] = self.fit_nan_policy
        data["fit jac"] = self.fit_jac
        data["curve fit options"] = self.curve_fit_options
        data["print fit"] = self.print_fit
        data["print fit details"] = self.print_fit_details
        data["sig figs"] = self.sig_figs
        data["plot fit"] = self.plot_fit
        data["fit band"] = self.fit_band
        data["fit band level"] = self.fit_band_level
        data["fit line range"] = self.fit_line_range
        data["legend"] = self.legend
        data["legend fontsize"] = self.legend_fontsize
        data["fit label"] = self.fit_label
        data["fit color"] = self.fit_color
        data["fit linestyle"] = self.fit_linestyle
        data["fit linewidth"] = self.fit_linewidth
        data["fit alpha"] = self.fit_alpha
        data["return stats"] = self.return_stats
        return data


@dataclass(frozen=True, slots=True)
class SevcikAnalysisOptions(PeakCurrentOptions):
    x_unit: str | None = "auto"
    y_unit: str | None = "auto"
    species: str | None = None
    fit_indices: object | None = None
    plot_fit: bool = True
    fit_band: str | None = None
    fit_band_level: float = 0.95
    fit_line_range: object | None = None
    legend: bool = False
    legend_fontsize: int | float | None = None
    fit_label: bool | str = False
    fit_color: object = "tab:red"
    fit_linestyle: str = "--"
    fit_linewidth: int | float = 1
    fit_alpha: int | float = 1
    return_stats: bool = False
    c: float | None = None
    num_electrons: int | float = 1
    scan_dependence: int | float = 0.5

    @classmethod
    def from_options(cls, options=None):
        return _coerce_options(cls, options, ["plot", "cv_selection", "cv_analysis", "peak_current", "sevcik_analysis"])

    def validate(self):
        _validate_common_cv(self)
        if float(self.num_electrons) <= 0:
            raise OptionError("'num electrons' must be positive.")
        if float(self.scan_dependence) <= 0:
            raise OptionError("'scan dependence' must be positive.")
        if not 0 <= float(self.fit_alpha) <= 1:
            raise OptionError("'fit alpha' must be between 0 and 1.")
        _validate_fit_band_options(self)

    def to_options_dict(self):
        data = PeakCurrentOptions.to_options_dict(self)
        data["x unit"] = self.x_unit
        data["y unit"] = self.y_unit
        data["species"] = self.species
        data["fit indices"] = self.fit_indices
        data["plot fit"] = self.plot_fit
        data["fit band"] = self.fit_band
        data["fit band level"] = self.fit_band_level
        data["fit line range"] = self.fit_line_range
        data["legend"] = self.legend
        data["legend fontsize"] = self.legend_fontsize
        data["fit label"] = self.fit_label
        data["fit color"] = self.fit_color
        data["fit linestyle"] = self.fit_linestyle
        data["fit linewidth"] = self.fit_linewidth
        data["fit alpha"] = self.fit_alpha
        data["return stats"] = self.return_stats
        data["C"] = self.c
        data["num electrons"] = self.num_electrons
        data["scan dependence"] = self.scan_dependence
        return data

    def for_peak_current(self, segment=None):
        return PeakCurrentOptions.from_options(
            {
                "plot": self.plot,
                "print": self.print,
                "plot all": self.plot_all,
                "print all": self.print_all,
                "segment": self.segment if segment is None else segment,
                "x axis": self.x_axis,
                "y axis": self.y_axis,
                "noise window": self.noise_window,
                "noise polyorder": self.noise_polyorder,
                "sig figs": self.sig_figs,
                "x unit": self.x_unit,
                "y unit": self.y_unit,
                "peak prominence": self.peak_prominence,
                "guess potential": self.guess_potential,
                "exact potential": self.exact_potential,
                "tangent range": self.tangent_range,
                "tangent min points": self.tangent_min_points,
                "tangent potential": self.tangent_potential,
                "percent threshold": self.percent_threshold,
            }
        )


@dataclass(frozen=True, slots=True)
class FitPeakCurrentOptions(PeakCurrentOptions):
    fit: bool = True
    x_unit: str | None = "auto"
    y_unit: str | None = "auto"
    species: str | None = None
    x_transform: str | float | int | None = None
    y_transform: str | float | int | None = None
    transform_mode: str | None = None
    floor: bool | str | float | int | None = None
    x_floor: bool | str | float | int | None = None
    y_floor: bool | str | float | int | None = None
    y_mode: str | None = None
    y0: object | None = None
    xscale: str | None = None
    yscale: str | None = None
    plot_scale: str | None = None
    fit_indices: object | None = None
    fit_model: object | None = "linear"
    fit_params: object | None = None
    fit_init: object | None = "auto"
    fit_bounds: object | None = "auto"
    fit_residual: str = "direct"
    fit_max_evals: int = 10000
    fit_method: str | None = "auto"
    fit_sigma: object | None = None
    fit_absolute_sigma: bool = False
    fit_check_finite: bool | None = None
    fit_nan_policy: str | None = None
    fit_jac: object | None = None
    curve_fit_options: dict | None = None
    print_fit: str = "auto"
    print_fit_details: bool = False
    sig_figs: int = 6
    plot_fit: bool = True
    fit_band: str | None = None
    fit_band_level: float = 0.95
    fit_line_range: object | None = None
    legend: bool = False
    legend_fontsize: int | float | None = None
    fit_label: bool | str = False
    fit_color: object = "tab:red"
    fit_linestyle: str = "--"
    fit_linewidth: int | float = 1
    fit_alpha: int | float = 1
    return_stats: bool = False

    @classmethod
    def from_options(cls, options=None):
        return _coerce_options(cls, options, ["plot", "cv_selection", "cv_analysis", "peak_current", "fit_peak_current"])

    def validate(self):
        _validate_common_cv(self)
        if self.plot_scale is not None:
            scale = str(self.plot_scale).strip().lower().replace("_", "-").replace(" ", "-")
            if scale not in {"linear", "log-log", "loglog", "semilogx", "semi-log-x", "semilogy", "semi-log-y", "symlog"}:
                raise OptionError("'plot scale' must be 'linear', 'log-log', 'semilogx', 'semilogy', or 'symlog'.")
        if not 0 <= float(self.fit_alpha) <= 1:
            raise OptionError("'fit alpha' must be between 0 and 1.")
        if float(self.fit_linewidth) < 0:
            raise OptionError("'fit linewidth' must be non-negative.")
        _validate_fit_band_options(self)

    def to_options_dict(self):
        data = PeakCurrentOptions.to_options_dict(self)
        data["fit"] = self.fit
        data["x unit"] = self.x_unit
        data["y unit"] = self.y_unit
        data["species"] = self.species
        data["x transform"] = self.x_transform
        data["y transform"] = self.y_transform
        data["transform mode"] = self.transform_mode
        data["floor"] = self.floor
        data["x floor"] = self.x_floor
        data["y floor"] = self.y_floor
        data["y mode"] = self.y_mode
        data["y0"] = self.y0
        data["xscale"] = self.xscale
        data["yscale"] = self.yscale
        data["plot scale"] = self.plot_scale
        data["fit indices"] = self.fit_indices
        data["fit model"] = self.fit_model
        data["fit params"] = self.fit_params
        data["fit init"] = self.fit_init
        data["fit bounds"] = self.fit_bounds
        data["fit residual"] = self.fit_residual
        data["fit max evals"] = self.fit_max_evals
        data["fit method"] = self.fit_method
        data["fit sigma"] = self.fit_sigma
        data["fit absolute sigma"] = self.fit_absolute_sigma
        data["fit check finite"] = self.fit_check_finite
        data["fit nan policy"] = self.fit_nan_policy
        data["fit jac"] = self.fit_jac
        data["curve fit options"] = self.curve_fit_options
        data["print fit"] = self.print_fit
        data["print fit details"] = self.print_fit_details
        data["sig figs"] = self.sig_figs
        data["plot fit"] = self.plot_fit
        data["fit band"] = self.fit_band
        data["fit band level"] = self.fit_band_level
        data["fit line range"] = self.fit_line_range
        data["legend"] = self.legend
        data["legend fontsize"] = self.legend_fontsize
        data["fit label"] = self.fit_label
        data["fit color"] = self.fit_color
        data["fit linestyle"] = self.fit_linestyle
        data["fit linewidth"] = self.fit_linewidth
        data["fit alpha"] = self.fit_alpha
        data["return stats"] = self.return_stats
        return data

    def for_peak_current(self, segment=None):
        return PeakCurrentOptions.from_options(
            {
                "plot": self.plot,
                "print": self.print,
                "plot all": self.plot_all,
                "print all": self.print_all,
                "segment": self.segment if segment is None else segment,
                "x axis": self.x_axis,
                "y axis": self.y_axis,
                "x unit": self.x_unit,
                "y unit": self.y_unit,
                "noise window": self.noise_window,
                "noise polyorder": self.noise_polyorder,
                "sig figs": self.sig_figs,
                "peak prominence": self.peak_prominence,
                "guess potential": self.guess_potential,
                "exact potential": self.exact_potential,
                "tangent range": self.tangent_range,
                "tangent min points": self.tangent_min_points,
                "tangent potential": self.tangent_potential,
                "percent threshold": self.percent_threshold,
            }
        )


@dataclass(frozen=True, slots=True)
class TrumpetAnalysisOptions(PeakCurrentOptions):
    segment: int | None = 1
    fit_indices: object | None = None
    plot_fit: bool = True
    fit_band: str | None = None
    fit_band_level: float = 0.95
    fit_line_range: object | None = None
    legend: bool = False
    legend_fontsize: int | float | None = None
    fit_label: bool | str = False
    fit_color: object = "tab:red"
    fit_linestyle: str = "--"
    fit_linewidth: int | float = 1
    fit_alpha: int | float = 1
    return_stats: bool = False
    d: float | None = None
    temperature: float = 298

    @classmethod
    def from_options(cls, options=None):
        return _coerce_options(cls, options, ["plot", "cv_selection", "cv_analysis", "peak_current", "trumpet_analysis"])

    def validate(self):
        _validate_common_cv(self)
        if _resolve_trumpet_base_segment(self.segment, self.segments) is None:
            raise OptionError("'segment' or consecutive 'segments' is required for trumpet_analysis.")
        if self.d is not None and float(self.d) <= 0:
            raise OptionError("'D' must be positive when provided.")
        if float(self.temperature) <= 0:
            raise OptionError("'temperature' must be positive.")
        if not 0 <= float(self.fit_alpha) <= 1:
            raise OptionError("'fit alpha' must be between 0 and 1.")
        _validate_fit_band_options(self)

    def to_options_dict(self):
        data = PeakCurrentOptions.to_options_dict(self)
        if self.segment is None and self.segments is not None:
            data["segment"] = _resolve_trumpet_base_segment(self.segment, self.segments)
            data["segments"] = None
        data["fit indices"] = self.fit_indices
        data["plot fit"] = self.plot_fit
        data["fit band"] = self.fit_band
        data["fit band level"] = self.fit_band_level
        data["fit line range"] = self.fit_line_range
        data["legend"] = self.legend
        data["legend fontsize"] = self.legend_fontsize
        data["fit label"] = self.fit_label
        data["fit color"] = self.fit_color
        data["fit linestyle"] = self.fit_linestyle
        data["fit linewidth"] = self.fit_linewidth
        data["fit alpha"] = self.fit_alpha
        data["return stats"] = self.return_stats
        data["D"] = self.d
        data["temperature"] = self.temperature
        return data


def _resolve_trumpet_base_segment(segment, segments):
    if segment is not None:
        return segment
    if segments is None:
        return None
    if isinstance(segments, int):
        return segments
    try:
        values = list(segments)
    except TypeError as exc:
        raise OptionError("'segments' for trumpet_analysis must be an int or a consecutive 2-element sequence.") from exc
    if len(values) != 2:
        raise OptionError("'segments' for trumpet_analysis must be an int or a consecutive 2-element sequence.")
    first, second = values
    if not isinstance(first, int) or not isinstance(second, int) or second != first + 1:
        raise OptionError("'segments' for trumpet_analysis must be an int or a consecutive 2-element sequence.")
    return first


@dataclass(frozen=True, slots=True)
class NicholsonOptions(PeakCurrentOptions):
    segment: int | None = 1
    plot_fit: bool = True
    fit_line_range: object | None = None
    fit_model: object | None = "origin"
    num_electrons: int | float = 1
    d: float | None = None
    psi_source: str = "agarwal table"
    nicholson_delta_ep_min_mv: int | float = 61
    nicholson_delta_ep_max_mv: int | float = 212
    fit_through_origin: bool = True
    exclude_invalid_delta_ep: bool = True
    plot_diagnostic: bool = True
    empirical_psi_equation: str = "lavagnini"
    warn_ir_drop: bool = True
    scan_rate: object | None = None

    @classmethod
    def from_options(cls, options=None):
        return _coerce_options(cls, options, ["plot", "cv_selection", "cv_analysis", "peak_current", "nicholson"])

    def validate(self):
        _validate_common_cv(self)
        if self.segment is None and self.segments is None:
            raise OptionError("'segment' or 'segments' is required for Nicholson.")
        if self.segments is not None:
            try:
                values = [self.segments] if isinstance(self.segments, int) else list(self.segments)
            except TypeError as exc:
                raise OptionError("'segments' for Nicholson must be an int or a 1- or 2-element sequence.") from exc
            if len(values) not in {1, 2}:
                raise OptionError("'segments' for Nicholson must contain one base segment or a 2-segment pair.")
            if not all(isinstance(value, int) for value in values):
                raise OptionError("'segments' for Nicholson must contain integer segment numbers.")
        model = str(self.fit_model).strip().lower().replace("_", " ").replace("-", " ")
        if model not in {"origin", "linear"}:
            raise OptionError("'fit model' for Nicholson must be 'origin' or 'linear'.")
        if float(self.num_electrons) <= 0:
            raise OptionError("'num electrons' must be positive.")
        if self.d is not None and float(self.d) <= 0:
            raise OptionError("'D' must be positive when provided.")
        if float(self.nicholson_delta_ep_min_mv) <= 0:
            raise OptionError("'nicholson delta ep min mv' must be positive.")
        if float(self.nicholson_delta_ep_max_mv) <= float(self.nicholson_delta_ep_min_mv):
            raise OptionError("'nicholson delta ep max mv' must be greater than the minimum.")

    def to_options_dict(self):
        data = PeakCurrentOptions.to_options_dict(self)
        data["plot fit"] = self.plot_fit
        data["fit line range"] = self.fit_line_range
        data["fit model"] = self.fit_model
        data["num electrons"] = self.num_electrons
        data["D"] = self.d
        data["psi source"] = self.psi_source
        data["nicholson delta ep min mv"] = self.nicholson_delta_ep_min_mv
        data["nicholson delta ep max mv"] = self.nicholson_delta_ep_max_mv
        data["fit through origin"] = self.fit_through_origin
        data["exclude invalid delta ep"] = self.exclude_invalid_delta_ep
        data["plot diagnostic"] = self.plot_diagnostic
        data["empirical psi equation"] = self.empirical_psi_equation
        data["warn ir drop"] = self.warn_ir_drop
        data["scan rate"] = self.scan_rate
        return data


@dataclass(frozen=True, slots=True)
class TafelAnalysisOptions:
    overpotential_range: list[float] | tuple[float, float] = (0, 1)
    color: str = "black"

    @classmethod
    def from_options(cls, options=None):
        return _coerce_options(cls, options, ["tafel_analysis"])

    def validate(self):
        if len(self.overpotential_range) != 2:
            raise OptionError("'overpotential range' must contain [start, end].")

    def to_options_dict(self):
        return {
            "overpotential range": list(self.overpotential_range),
            "color": self.color,
        }


@dataclass(frozen=True, slots=True)
class FilterOptions:
    print: bool = True
    pretty_print: bool = True
    print_conditions: bool = True
    mode: str = "include"
    logic: str | None = None

    @classmethod
    def from_options(cls, options=None):
        return _coerce_options(cls, options, ["filter"])

    def validate(self):
        mode = str(self.mode).strip().lower()
        if mode not in {"include", "exclude"}:
            raise OptionError("'mode' must be 'include' or 'exclude'.")
        if self.logic is not None and str(self.logic).strip().upper() not in {"AND", "OR"}:
            raise OptionError("'logic' must be 'AND', 'OR', or None.")

    def to_options_dict(self):
        return {
            "print": self.print,
            "pretty print": self.pretty_print,
            "print conditions": self.print_conditions,
            "mode": str(self.mode).strip().lower(),
            "logic": None if self.logic is None else str(self.logic).strip().upper(),
        }


@dataclass(frozen=True, slots=True)
class SortGroupOptions:
    print: bool = True
    pretty_print: bool = True
    print_conditions: bool = True

    @classmethod
    def from_options(cls, options=None):
        return _coerce_options(cls, options, ["sort_group"])

    def validate(self):
        return None

    def to_options_dict(self):
        return {
            "print": self.print,
            "pretty print": self.pretty_print,
            "print conditions": self.print_conditions,
        }


@dataclass(frozen=True, slots=True)
class GroupSummaryOptions:
    print: bool = True
    pretty_print: bool = True
    sig_figs: int = 3
    group_keys: object = None
    columns: object = field(default_factory=list)

    @classmethod
    def from_options(cls, options=None):
        return _coerce_options(cls, options, ["group_summary"])

    def validate(self):
        return None

    def to_options_dict(self):
        return {
            "print": self.print,
            "pretty print": self.pretty_print,
            "sig figs": self.sig_figs,
            "group keys": self.group_keys,
            "columns": self.columns,
        }


def _display_option_key(key, section=None):
    key = _canonical_option_key(key)
    if section is not None:
        section_key = _canonical_section_key(section)
        section_overrides = _SECTION_OPTION_DISPLAY_OVERRIDES.get(section_key, {})
        if key in section_overrides:
            return section_overrides[key]
    return _DISPLAY_KEY_OVERRIDES.get(key, _friendly_key(key))


def _display_section_key(key):
    return _canonical_section_key(key)


_DESCRIBE_WORKFLOW_ORDER = (
    "Overview",
    "Import and metadata",
    "Object plotting",
    "Overlay plotting",
    "CV preprocessing",
    "Single CV analysis",
    "DPV analysis",
    "CA/CP analysis",
    "Batch analysis",
    "Scatter and fit analysis",
    "Simulation and fitting",
    "Collection utilities",
    "Export",
    "General options",
)


_DESCRIBE_FUNCTION_WORKFLOWS = {
    "all": "Overview",
    "get_data": "Import and metadata",
    "get_data_from_excel": "Import and metadata",
    "echem.from_file": "Import and metadata",
    "plot": "Object plotting",
    "echem.x": "Object plotting",
    "echem.y": "Object plotting",
    "echem.xy": "Object plotting",
    "echem.plot": "Object plotting",
    "cv.x": "Object plotting",
    "cv.y": "Object plotting",
    "cv.xy": "Object plotting",
    "cv.plot": "Object plotting",
    "dpv.x": "Object plotting",
    "dpv.y": "Object plotting",
    "dpv.xy": "Object plotting",
    "dpv.plot": "Object plotting",
    "ca.plot": "Object plotting",
    "cp.plot": "Object plotting",
    "multiplot": "Overlay plotting",
    "multimultiplot": "Overlay plotting",
    "multi_scatterplot": "Overlay plotting",
    "save_data": "Export",
    "animate": "Overlay plotting",
    "cv.normalize": "CV preprocessing",
    "cv.normalize_current": "CV preprocessing",
    "cv.scale_current": "CV preprocessing",
    "cv.filter": "CV preprocessing",
    "cv.trim": "CV preprocessing",
    "normalize": "CV preprocessing",
    "normalize_current": "CV preprocessing",
    "scale_current": "CV preprocessing",
    "trim": "CV preprocessing",
    "cv.current_at_potential": "Single CV analysis",
    "cv.peak_potential": "Single CV analysis",
    "cv.peak_current": "Single CV analysis",
    "cv.peak_info": "Single CV analysis",
    "cv.plateau_current": "Single CV analysis",
    "cv.half_peak_potential": "Single CV analysis",
    "cv.half_wave_potential": "Single CV analysis",
    "cv.wave_info": "Single CV analysis",
    "cv_analysis": "Single CV analysis",
    "peak_current": "Single CV analysis",
    "dpv.peak_potential": "DPV analysis",
    "ca.charge": "CA/CP analysis",
    "ca.time_at_charge": "CA/CP analysis",
    "cp.get_cycles": "CA/CP analysis",
    "cp.plot_cycles": "CA/CP analysis",
    "cp.cycling_plot": "CA/CP analysis",
    "cp.cycle_info": "CA/CP analysis",
    "fowa": "Batch analysis",
    "sevcik_analysis": "Batch analysis",
    "trumpet_analysis": "Batch analysis",
    "nicholson": "Batch analysis",
    "nicholson_analysis": "Batch analysis",
    "tafel_analysis": "Batch analysis",
    "plateau_current": "Batch analysis",
    "fit_model": "Scatter and fit analysis",
    "fit_rate": "Scatter and fit analysis",
    "fit_peak_current": "Scatter and fit analysis",
    "fit_peak_potential": "Scatter and fit analysis",
    "simulation.cv_data": "Simulation and fitting",
    "simulation.simulate_cv": "Simulation and fitting",
    "simulation.fit_cv": "Simulation and fitting",
    "filter": "Collection utilities",
    "sort_group": "Collection utilities",
    "group_summary": "Collection utilities",
}


_DESCRIBE_FUNCTION_DESCRIPTIONS = {
    "all": "Show every registered option table, grouped by workflow and function.",
    "get_data": "Import supported electrochemistry files, parse filename/file metadata, apply reference handling, and return eCAT objects.",
    "echem.from_file": "Load one supported data file and promote it to the detected eCAT object type when possible.",
    "get_data_from_excel": "Load worksheet-based exported eCAT data back into eCAT objects.",
    "plot": "General object plot options shared by echem, CV, DPV, CA, and CP plots. For class-specific controls, use cv.plot, ca.plot, cp.plot, or dpv.plot.",
    "echem.plot": "Generic echem plot options for axis selection, units, titles, legends, scale bars, and shared plot styling.",
    "cv.plot": "CV trace plotting options, including segment selection, derivative/smoothing display, current normalization, segment coloring, scale bars, and directional arrows.",
    "dpv.plot": "DPV trace plotting options for potential-current display plus shared axis, label, legend, and styling controls.",
    "ca.plot": "Chronoamperometry plotting options, including current traces, charge overlays, target charge markers, and baseline-correction display.",
    "cp.plot": "Chronopotentiometry plotting options for potential-time traces and cycle-oriented displays.",
    "echem.x": "Options affecting x-axis extraction and display from generic echem objects.",
    "echem.y": "Options affecting y-axis extraction and display from generic echem objects.",
    "echem.xy": "Options affecting paired x/y extraction and display from generic echem objects.",
    "cv.x": "Options affecting CV x-axis extraction and display.",
    "cv.y": "Options affecting CV y-axis extraction and display.",
    "cv.xy": "Options affecting paired CV x/y extraction and display.",
    "dpv.x": "Options affecting DPV x-axis extraction and display.",
    "dpv.y": "Options affecting DPV y-axis extraction and display.",
    "dpv.xy": "Options affecting paired DPV x/y extraction and display.",
    "multiplot": "Overlay multiple eCAT objects with shared axes, metadata-derived labels, legends, gradients, colorbars, scale bars, and directional arrows.",
    "multimultiplot": "Create grouped overlay panels from multiple object groups with shared styling, titles, subtitles, and legends.",
    "multi_scatterplot": "Plot one or more result-table metrics with explicit x/y column selection, grouping, labels, and fit display.",
    "save_data": "Export eCAT objects to CSV or Excel-style processed tables while preserving units and useful metadata.",
    "animate": "Animate one object or an object list using eCAT plot styling plus trace timing, scheduling, frame rate, looping, and export controls.",
    "cv.normalize": "Normalize one CV with physical dimensionless CV variables and metadata-derived parameters when available.",
    "normalize": "Normalize one or more CVs with physical dimensionless CV variables and metadata-derived parameters when available.",
    "cv.normalize_current": "Normalize CV current by a reference peak current resolved from ip0, a reference CV, or a reference CV list.",
    "normalize_current": "Normalize current for one or more CVs by a reference peak current resolved from ip0 or reference CVs.",
    "cv.scale_current": "Scale one CV current against manual or reference-wave current values.",
    "cv.filter": "Filter one CV current trace with a recorded SciPy-backed preprocessing operation.",
    "scale_current": "Scale CV currents against manual or reference-wave current values.",
    "trim": "Trim one or more CVs to a potential window while preserving connected scan data by default.",
    "cv.trim": "Trim one CV to a potential window while preserving connected scan data by default.",
    "cv.current_at_potential": "Extract current from a CV at a requested potential with segment and interpolation controls.",
    "cv.peak_potential": "Locate a CV peak potential using segment selection, smoothing, prominence, and fallback controls.",
    "cv.peak_current": "Measure CV peak current with peak selection, tangent/background handling, fallback behavior, plotting, and printing controls.",
    "cv.peak_info": "Return peak-potential and peak-current details using the peak-current option surface.",
    "cv.plateau_current": "Analyze plateau current for one CV with optional normalization and plotting controls.",
    "cv.half_peak_potential": "Estimate a CV half-peak potential from selected peak-current diagnostics.",
    "cv.half_wave_potential": "Estimate a CV half-wave potential from paired peak diagnostics.",
    "cv.wave_info": "Return paired-wave information using peak-current and half-wave controls.",
    "cv_analysis": "Shared CV analysis controls for segment selection, smoothing, peak guesses, exact potentials, diagnostics, and significant figures.",
    "peak_current": "Shared peak-current controls for tangent baselines, percent thresholds, and peak fallback behavior.",
    "dpv.peak_potential": "Locate a DPV peak potential near a guess with smoothing, prominence, and diagnostic controls.",
    "ca.charge": "Integrate chronoamperometry current to cumulative charge and optionally plot current, charge, and target diagnostics.",
    "ca.time_at_charge": "Find when a chronoamperometry trace reaches a requested target charge.",
    "cp.get_cycles": "Split chronopotentiometry data into charge/discharge cycles.",
    "cp.plot_cycles": "Plot selected chronopotentiometry cycles with shared object-plot styling.",
    "cp.cycling_plot": "Plot chronopotentiometry cycling metrics such as capacity and efficiency versus cycle number.",
    "cp.cycle_info": "Summarize chronopotentiometry cycle capacity, efficiency, and potential metrics.",
    "fowa": "Foot-of-the-wave analysis for catalytic CVs, including reference-wave handling, tangent backgrounds, fit windows, and diagnostics.",
    "sevcik_analysis": "Sevcik-style peak-current trend analysis across scan rates.",
    "trumpet_analysis": "Trumpet analysis from paired peak potentials across scan rates.",
    "nicholson": "Nicholson-style heterogeneous electron-transfer analysis from peak separation and scan-rate trends.",
    "nicholson_analysis": "Nicholson-style heterogeneous electron-transfer analysis from peak separation and scan-rate trends.",
    "tafel_analysis": "Tafel-style turnover-frequency analysis for one CV or a CV series.",
    "plateau_current": "Batch plateau-current workflow for identifying scan-rate-independent catalytic limiting currents.",
    "fit_model": "Fit a generic model to x/y data, a result table, or an existing scatter-fit result.",
    "fit_rate": "Fit rate or transformed FOWA/result-table data with shared scatter-fit model controls.",
    "fit_peak_current": "Fit peak-current trends across a CV series using shared scatter-fit model controls.",
    "fit_peak_potential": "Fit peak-potential trends across a CV series using shared scatter-fit model controls.",
    "simulation.cv_data": "Convert an imported eCAT CV into simulation/fitting input while preserving measured current and metadata.",
    "simulation.simulate_cv": "Run an ElectroKitty-backed CV simulation and overlay simulated/measured current.",
    "simulation.fit_cv": "Fit one measured CV with eCAT simulation least-squares or strategy methods.",
    "filter": "Include or exclude eCAT objects by exact metadata criteria.",
    "sort_group": "Sort and group eCAT objects by metadata fields for plotting, summaries, or batch analysis.",
    "group_summary": "Summarize grouped object metadata in notebook-friendly tables.",
}


def _workflow_for_function(function):
    return _DESCRIBE_FUNCTION_WORKFLOWS.get(str(function), "General options")


def _workflow_sort_key(function):
    workflow = _workflow_for_function(function)
    try:
        workflow_index = _DESCRIBE_WORKFLOW_ORDER.index(workflow)
    except ValueError:
        workflow_index = len(_DESCRIBE_WORKFLOW_ORDER)
    return workflow_index, str(function)


def _description_for_function(function):
    function = str(function)
    if function in _DESCRIBE_FUNCTION_DESCRIPTIONS:
        return _DESCRIBE_FUNCTION_DESCRIPTIONS[function]
    friendly = function.replace("_", " ")
    return f"Options for the {friendly} workflow."


def _type_to_string(annotation):
    if isinstance(annotation, str):
        return annotation.replace("NoneType", "None").replace("typing.", "")

    if annotation is None or annotation is type(None):
        return "None"
    if annotation is object:
        return "object"

    origin = typing.get_origin(annotation)
    args = typing.get_args(annotation)
    if origin in {typing.Union, types.UnionType}:
        return " or ".join(_type_to_string(arg) for arg in args)
    if origin in {list, tuple, dict, set}:
        name = origin.__name__
        if not args:
            return name
        return f"{name}[{', '.join(_type_to_string(arg) for arg in args)}]"
    if hasattr(annotation, "__name__"):
        return annotation.__name__
    return str(annotation).replace("typing.", "")


def _type_hints_for_model(cls):
    hints = {}
    for base in reversed(cls.__mro__):
        try:
            hints.update(typing.get_type_hints(base))
        except Exception:  # pragma: no cover - defensive fallback for unusual annotations
            hints.update(getattr(base, "__annotations__", {}))
    return hints


def _option_type_lookup():
    lookup = {}
    for cls, _sections in _option_default_registry():
        hints = _type_hints_for_model(cls)
        for field in fields(cls):
            lookup.setdefault(field.name, hints.get(field.name, field.type))
    return lookup


def _default_for_field(field):
    if field.default is not MISSING:
        return field.default
    if field.default_factory is not MISSING:  # type: ignore[attr-defined]
        return field.default_factory()  # type: ignore[misc]
    return None


def _sections_for_option_model(cls):
    for registered_cls, sections in _option_default_registry():
        if registered_cls is cls:
            return sections
    return []


def _metadata_for_option(key, section=None):
    norm = normalize_key(key)
    metadata = dict(OPTION_METADATA.get("*", {}).get(norm, {}))
    if section is not None:
        section_norm = normalize_key(section)
        metadata.update(OPTION_METADATA.get(section_norm, {}).get(norm, {}))
        canonical_section = _canonical_section_key(section)
        if canonical_section != section_norm:
            metadata.update(OPTION_METADATA.get(canonical_section, {}).get(norm, {}))
    return metadata


def _description_for_option(key, section=None):
    return _metadata_for_option(key, section=section).get("description", "")


def _choices_for_option(key, section=None):
    return list(_metadata_for_option(key, section=section).get("choices", []))


def _category_for_option(key, section=None):
    return _metadata_for_option(key, section=section).get("category", "Advanced")


def _schema_entry(key, default, annotation, section=None):
    return {
        "category": _category_for_option(key, section=section),
        "default": deepcopy(default),
        "type": _type_to_string(annotation),
        "choices": _choices_for_option(key, section=section),
        "description": _description_for_option(key, section=section),
    }


def _format_choices_for_display(choices):
    if not choices:
        return ""
    return ", ".join(str(choice) for choice in choices)


def _options_schema_to_dataframe(schema):
    import pandas as pd

    rows = []

    def add_schema_rows(section, section_schema, include_workflow=False):
        for option, entry in section_schema.items():
            row = {
                "Function": section,
                "Category": entry.get("category", "Advanced"),
                "Option": option,
                "Default": entry.get("default"),
                "Type": entry.get("type", ""),
                "Choices": _format_choices_for_display(entry.get("choices", [])),
                "Description": entry.get("description", ""),
            }
            if include_workflow:
                row = {"Workflow": _workflow_for_function(section), **row}
            rows.append(row)

    def sort_option_rows(df):
        if df.empty or "Option" not in df.columns:
            return df
        sort_columns = []
        if "Function" in df.columns:
            if "Workflow" in df.columns:
                workflow_order = {
                    workflow: index
                    for index, workflow in enumerate(_DESCRIBE_WORKFLOW_ORDER)
                }
                df = df.assign(
                    _workflow_order=df["Workflow"].map(workflow_order).fillna(len(workflow_order))
                )
                sort_columns.append("_workflow_order")
            sort_columns.append("Function")
        if "Category" in df.columns:
            category_order = {
                category: index
                for index, category in enumerate(OPTION_CATEGORY_ORDER)
            }
            df = df.assign(
                _category_order=df["Category"].map(category_order).fillna(len(category_order))
            )
            sort_columns.append("_category_order")
        sort_columns.append("Option")
        sorted_df = df.sort_values(sort_columns, kind="stable").reset_index(drop=True)
        for helper_column in ("_workflow_order", "_category_order"):
            if helper_column in sorted_df.columns:
                sorted_df = sorted_df.drop(columns=[helper_column])
        return sorted_df

    if schema and all(
        isinstance(value, dict)
        and "default" not in value
        and "type" not in value
        and "choices" not in value
        and "description" not in value
        and "category" not in value
        for value in schema.values()
    ):
        for section, section_schema in schema.items():
            add_schema_rows(section, section_schema, include_workflow=True)
        return _drop_empty_display_columns(sort_option_rows(pd.DataFrame(rows)))

    add_schema_rows("", schema)
    df = pd.DataFrame(rows)
    return _drop_empty_display_columns(sort_option_rows(df.drop(columns=["Function"])))


def _drop_empty_display_columns(df):
    protected_columns = {"Workflow", "Function", "Category", "Option"}
    empty_columns = []
    for column in df.columns:
        if column in protected_columns:
            continue
        values = df[column]
        if values.map(lambda value: value is None or value == "").all():
            empty_columns.append(column)
    if empty_columns:
        return df.drop(columns=empty_columns)
    return df


def _display_options_schema(schema, display_function=None):
    df = _options_schema_to_dataframe(schema)
    _display_options_dataframe(df, display_function=display_function)


def _display_options_dataframe(df, display_function=None):
    display_func = display_function if display_function is not None else display
    if display_func is not None:
        try:
            styled = (
                df.style
                .set_properties(**{
                    "text-align": "left",
                    "white-space": "pre-wrap",
                    "vertical-align": "top",
                })
                .set_table_styles([
                    {"selector": "th", "props": [("text-align", "left")]},
                    {"selector": "td", "props": [("text-align", "left")]},
                ])
            )
            display_func(styled)
        except Exception:
            display_func(df)
    else:
        print(df.to_string(index=False))


def _describe_options_config(options=None):
    config = {} if options is None else dict(options)
    config.setdefault("print", True)
    config.setdefault("pretty print", True)
    config.setdefault("return", False)
    return config


def _describe_options_should_return_table(config):
    return bool(config.get("return", False))


def _describe_options_section_names():
    return [_display_section_key(section) for section in sorted(_default_section_names())]


def _describe_options_method_names():
    return sorted(_METHOD_SECTION_ALIASES)


def _describe_options_function_names():
    return sorted(
        set(_describe_options_section_names())
        | set(_describe_options_method_names())
        | set(_SIMULATION_OPTION_SCHEMAS)
    )


def _describe_options_menu_dataframe():
    import pandas as pd

    functions = ["all", *_describe_options_function_names()]
    functions = sorted(functions, key=_workflow_sort_key)
    rows = [
        {
            "Workflow": _workflow_for_function(function),
            "Function": function,
            "Description": _description_for_function(function),
        }
        for function in functions
    ]
    return pd.DataFrame(rows)


def _describe_options_schema_from_defaults(defaults, type_lookup, section):
    schema = {}
    for key, value in defaults.items():
        display_key = _display_option_key(key, section=section)
        if display_key in schema:
            continue
        schema[display_key] = _schema_entry(
            key,
            value,
            type_lookup.get(key, type(value)),
            section=section,
        )
    return schema


def _describe_options_schema_for_animate(type_lookup):
    defaults = get_defaults("plot")
    defaults.update(get_defaults("multiplot"))
    for key, value in _ANIMATION_OPTION_DEFAULTS.items():
        defaults.setdefault(key, value)
    return {
        _display_option_key(key, section="animate"): _schema_entry(
            key,
            defaults[key],
            type_lookup.get(key, type(defaults[key])),
            section="animate",
        )
        for key in _ANIMATE_OPTION_KEYS
        if key in defaults
    }


def _describe_options_schema_for_section(section, type_lookup):
    return _describe_options_schema_from_defaults(
        get_defaults(section),
        type_lookup,
        section=section,
    )


def _option_model_for_function(function):
    function_key = normalize_key(function)
    section_key = _canonical_section_key(function)
    model_by_function = {
        "get_data": ImportOptions,
        "get_data_from_excel": ImportOptions,
        "echem.from_file": ImportOptions,
        "trim": TrimOptions,
        "cv.trim": TrimOptions,
        "plot": PlotOptions,
        "echem.x": PlotOptions,
        "echem.y": PlotOptions,
        "echem.xy": PlotOptions,
        "echem.plot": PlotOptions,
        "cv.x": PlotOptions,
        "cv.y": PlotOptions,
        "cv.xy": PlotOptions,
        "cv.plot": PlotOptions,
        "cv.plot_program": PlotOptions,
        "dpv.x": PlotOptions,
        "dpv.y": PlotOptions,
        "dpv.xy": PlotOptions,
        "dpv.plot": PlotOptions,
        "ca.plot": PlotOptions,
        "ca.charge": PlotOptions,
        "ca.time_at_charge": PlotOptions,
        "ca.current_at_time": PlotOptions,
        "ca.average_current": PlotOptions,
        "ca.rate_at_time": PlotOptions,
        "ca.average_rate": PlotOptions,
        "cp.plot": PlotOptions,
        "cp.get_cycles": PlotOptions,
        "cp.plot_cycles": PlotOptions,
        "cp.cycling_plot": PlotOptions,
        "cp.cycle_info": PlotOptions,
        "multiplot": MultiplotOptions,
        "multimultiplot": MultiMultiplotOptions,
        "multi_scatterplot": MultiScatterplotOptions,
        "cv_analysis": PeakPotentialOptions,
        "cv.current_at_potential": PeakPotentialOptions,
        "cv.peak_potential": PeakPotentialOptions,
        "cv.half_peak_potential": PeakPotentialOptions,
        "dpv.peak_potential": PeakPotentialOptions,
        "peak_current": PeakCurrentOptions,
        "cv.peak_current": PeakCurrentOptions,
        "cv.peak_info": PeakCurrentOptions,
        "cv.half_wave_potential": PeakCurrentOptions,
        "cv.wave_info": PeakCurrentOptions,
        "normalize": NormalizeOptions,
        "cv.normalize": NormalizeOptions,
        "normalize_current": NormalizationOptions,
        "cv.normalize_current": NormalizationOptions,
        "scale_current": ScaleCurrentOptions,
        "cv.scale_current": ScaleCurrentOptions,
        "cv.filter": CVFilterOptions,
        "fowa": FOWAOptions,
        "plateau_current": PlateauCurrentOptions,
        "cv.plateau_current": PlateauCurrentOptions,
        "fit_model": FitModelOptions,
        "fit_rate": FitRateOptions,
        "fit_peak_potential": FitPeakPotentialOptions,
        "fit_peak_current": FitPeakCurrentOptions,
        "sevcik_analysis": SevcikAnalysisOptions,
        "trumpet_analysis": TrumpetAnalysisOptions,
        "nicholson": NicholsonOptions,
        "nicholson_analysis": NicholsonOptions,
        "tafel_analysis": TafelAnalysisOptions,
        "filter": FilterOptions,
        "sort_group": SortGroupOptions,
        "group_summary": GroupSummaryOptions,
    }
    return model_by_function.get(function_key) or model_by_function.get(section_key)


def _describe_options_schema_for_model(cls, type_lookup, metadata_section=None):
    sections = _sections_for_option_model(cls)
    if metadata_section is None:
        metadata_section = sections[0] if len(sections) == 1 else None
    defaults = {}
    for section in sections:
        defaults.update(get_defaults(section))

    hints = _type_hints_for_model(cls)
    schema = {}
    hidden_fields = set()
    for field in fields(cls):
        if field.name.startswith("_") or field.name in hidden_fields:
            continue
        default = defaults.get(field.name, _default_for_field(field))
        display_key = _display_option_key(field.name, section=metadata_section)
        if display_key in schema:
            continue
        schema[display_key] = _schema_entry(
            field.name,
            default,
            hints.get(field.name, field.type),
            section=metadata_section,
        )
    return schema


def _describe_options_all_schema(type_lookup):
    schema = {
        _display_section_key(section): _describe_options_schema_for_section(section, type_lookup)
        for section in sorted(_default_section_names())
    }
    schema["animate"] = _describe_options_schema_for_animate(type_lookup)
    schema.update(deepcopy(_SIMULATION_OPTION_SCHEMAS))
    return schema


def _describe_options_invalid_section_message(section):
    valid_inputs = ["all", *_describe_options_function_names()]
    section_text = str(section)
    matches = difflib.get_close_matches(
        normalize_key(section_text),
        valid_inputs,
        n=1,
        cutoff=0.6,
    )
    if matches:
        return f"Unknown options function '{section_text}'. Did you mean '{matches[0]}'?"
    return (
        f"Unknown options function '{section_text}'. "
        "Possible functions are listed below."
    )


def describe_options(option_model_or_section=None, options=None):
    """Display eCAT option functions or option tables.

    With no argument, display valid option function names. Pass ``"all"`` for
    every option function, a function name such as ``"multiplot"``, a public
    method name such as ``"cv.peak_current"`` or ``"ca.charge"``, or an option
    dataclass such as ``PeakCurrentOptions``.

    Output is enabled by default. ``options={"pretty print": False}`` emits a
    plain text table instead of rich notebook display. Pass
    ``options={"print": False}`` to suppress output, and
    ``options={"return": True}`` to receive the dataframe.
    """
    config = _describe_options_config(options)
    type_lookup = _option_type_lookup()
    print_output = bool(config.get("print", True))
    pretty_print = bool(config.get("pretty print", True))
    return_table = _describe_options_should_return_table(config)

    def display_and_return(df):
        if print_output:
            if pretty_print:
                _display_options_dataframe(df, config.get("display function"))
            else:
                print(df.to_string(index=False))
        if return_table:
            return df
        return None

    if option_model_or_section is None:
        return display_and_return(_describe_options_menu_dataframe())

    if isinstance(option_model_or_section, str):
        section = option_model_or_section
        section_input_key = normalize_key(section)
        section_key = _canonical_section_key(section)

        if section_input_key == "animate":
            return display_and_return(
                _options_schema_to_dataframe(_describe_options_schema_for_animate(type_lookup))
            )

        if section_input_key in _SIMULATION_OPTION_SCHEMAS:
            return display_and_return(
                _options_schema_to_dataframe(deepcopy(_SIMULATION_OPTION_SCHEMAS[section_input_key]))
            )
        if section_key in _SIMULATION_OPTION_SCHEMAS:
            return display_and_return(
                _options_schema_to_dataframe(deepcopy(_SIMULATION_OPTION_SCHEMAS[section_key]))
            )

        if section_key == "all":
            return display_and_return(
                _options_schema_to_dataframe(_describe_options_all_schema(type_lookup))
            )

        option_model = _option_model_for_function(section)
        if option_model is not None:
            return display_and_return(
                _options_schema_to_dataframe(
                    _describe_options_schema_for_model(
                        option_model,
                        type_lookup,
                        metadata_section=section_key,
                    )
                )
            )

        if section_key not in _default_section_names():
            if print_output:
                print(_describe_options_invalid_section_message(section))
            return display_and_return(_describe_options_menu_dataframe())

        schema = _describe_options_schema_for_section(section_key, type_lookup)
        return display_and_return(_options_schema_to_dataframe(schema))

    cls = option_model_or_section
    if not hasattr(cls, "__dataclass_fields__"):
        cls = type(option_model_or_section)
    if not hasattr(cls, "__dataclass_fields__"):
        raise TypeError("describe_options() accepts a defaults section name or option dataclass.")

    return display_and_return(
        _options_schema_to_dataframe(_describe_options_schema_for_model(cls, type_lookup))
    )

__all__ = [
    'MISSING',
    'dataclass',
    'field',
    'fields',
    'Path',
    'resources',
    'difflib',
    're',
    'types',
    'typing',
    'deepcopy',
    'display',
    'tomllib',
    'TOMLDecodeError',
    'OptionError',
    'normalize_key',
    'OPTION_CATEGORY_ORDER',
    'OPTION_CATEGORIES',
    'OPTION_DESCRIPTIONS',
    'OPTION_CHOICES',
    'OPTION_CHOICES_BY_SECTION',
    'OPTION_METADATA',
    'load_defaults',
    'set_defaults',
    'get_defaults',
    'reset_defaults',
    'reset_defaults_option',
    'reset_defaults_section',
    'ImportOptions',
    'resolve_import_options',
    'TrimOptions',
    'PlotOptions',
    'MultiplotOptions',
    'MultiMultiplotOptions',
    'CVFilterOptions',
    'MultiScatterplotOptions',
    'FitModelOptions',
    'PeakPotentialOptions',
    'PeakCurrentOptions',
    'NormalizeOptions',
    'NormalizationOptions',
    'ScaleCurrentOptions',
    'FOWAOptions',
    'PlateauCurrentOptions',
    'FitRateOptions',
    'FitPeakPotentialOptions',
    'SevcikAnalysisOptions',
    'FitPeakCurrentOptions',
    'TrumpetAnalysisOptions',
    'NicholsonOptions',
    'TafelAnalysisOptions',
    'FilterOptions',
    'SortGroupOptions',
    'GroupSummaryOptions',
    'describe_options',
]
