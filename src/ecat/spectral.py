"""Physical signal and time resolution for spectrum analysis."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import dataclass
import math
from numbers import Integral, Real
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_numeric_dtype
from scipy import signal

from ._cv_direction import split_cv_potential_segments
from ._plot_helpers import _apply_ecat_axis_style, _apply_plot_titles
from .options import SpectrumOptions
from .plotting import (
    _apply_matplotlib_axis_scales,
    _can_rich_table_display,
    _display_table,
    _normalize_legend_loc,
    _resolve_outside_matplotlib_legend_anchor,
    _resolve_matplotlib_axis_scales,
    _resolve_subtitle_fontsize,
    _resolve_title_fontsize,
)
from .results import AnalysisResult
from .utils import extract_prefix_and_base, format_sigfigs, get_conversion_factor


_MAX_RESAMPLED_POINTS = 2_000_000
_MAX_TRANSFORM_POINTS = 2_000_000
_TRACE_ARRAY_FIELDS = (
    "time",
    "signal",
    "source_indices",
    "original_time",
    "original_signal",
    "original_intervals",
    "resolved_intervals",
)


class _CompactDiagnosticSequence(tuple):
    """Immutable diagnostics sequence with a bounded export descriptor."""

    __slots__ = ()

    _PREVIEW_COUNT = 3
    _MAX_DESCRIPTOR_LENGTH = 320

    def __new__(cls, values=()):
        return super().__new__(cls, values)

    @classmethod
    def _item_descriptor(cls, item):
        descriptor = repr(item)
        if len(descriptor) > cls._MAX_DESCRIPTOR_LENGTH:
            return descriptor[: cls._MAX_DESCRIPTOR_LENGTH - 3] + "..."
        return descriptor

    def __repr__(self):
        descriptors = [
            self._item_descriptor(item) for item in self[: self._PREVIEW_COUNT]
        ]
        if len(self) > self._PREVIEW_COUNT:
            descriptors.append("...")
        preview = ", ".join(descriptors)
        return (
            f"{type(self).__name__}(count={len(self)}, "
            f"preview=[{preview}])"
        )

    __str__ = __repr__


@dataclass(frozen=True, repr=False)
class _SpectrumTrace:
    segment: int | None
    label: str
    time: np.ndarray
    signal: np.ndarray
    signal_column: object
    signal_unit: str
    time_source: str
    source_indices: np.ndarray
    original_time: np.ndarray
    original_signal: np.ndarray
    original_intervals: np.ndarray
    resolved_intervals: np.ndarray
    median_interval: float
    relative_interval_deviation: float
    sample_rate: float
    time_unit: str = "s"
    resampled: bool = False

    def __post_init__(self):
        for name in _TRACE_ARRAY_FIELDS:
            values = np.array(getattr(self, name), copy=True)
            values.setflags(write=False)
            object.__setattr__(self, name, values)

    def __repr__(self):
        return (
            "_SpectrumTrace("
            f"segment={self.segment!r}, label={self.label!r}, "
            f"samples={len(self.time)}, signal_column={self.signal_column!r}, "
            f"signal_unit={self.signal_unit!r}, time_source={self.time_source!r}, "
            f"sample_rate={self.sample_rate!r}, resampled={self.resampled!r})"
        )


def _freeze_window_metadata(value, *, named_parameters=False):
    if value is None or isinstance(value, (str, bytes)) or np.isscalar(value):
        return value

    named_specification = (
        not named_parameters
        and isinstance(value, (tuple, list))
        and bool(value)
        and isinstance(value[0], str)
    )
    if (named_specification or named_parameters) and isinstance(value, (tuple, list)):
        return tuple(
            _freeze_window_metadata(item, named_parameters=True) for item in value
        )

    values = np.array(value, dtype=float, copy=True)
    values.setflags(write=False)
    return values


@dataclass(frozen=True, repr=False)
class _SpectrumComputation:
    segment: int | None
    label: str
    signal_unit: str
    method: str
    frequency: np.ndarray
    sample_count: int
    sample_rate: float
    nyquist: float
    frequency_bin_spacing: float
    nfft: int
    window: object
    detrend: str | None
    psd: np.ndarray | None = None
    amplitude: np.ndarray | None = None
    phase: np.ndarray | None = None
    nperseg: int | None = None
    noverlap: int | None = None
    window_count: int | None = None
    coherent_gain: float | None = None

    def __post_init__(self):
        for name in ("frequency", "psd", "amplitude", "phase"):
            source = getattr(self, name)
            if source is None:
                continue
            values = np.array(source, copy=True)
            values.setflags(write=False)
            object.__setattr__(self, name, values)

        object.__setattr__(self, "window", _freeze_window_metadata(self.window))

    def __repr__(self):
        return (
            "_SpectrumComputation("
            f"segment={self.segment!r}, label={self.label!r}, "
            f"method={self.method!r}, bins={len(self.frequency)}, "
            f"sample_count={self.sample_count}, sample_rate={self.sample_rate!r}, "
            f"nfft={self.nfft}, window={_summarize_spectrum_window(self.window)!r}, "
            f"detrend={self.detrend!r})"
        )


def _column_tokens(column):
    values = column if isinstance(column, tuple) else (column,)
    return {
        str(value).strip().casefold()
        for value in values
        if isinstance(value, str)
    }


def _casefold_column_key(column):
    if isinstance(column, tuple):
        return tuple(
            value.strip().casefold() if isinstance(value, str) else value
            for value in column
        )
    return column.strip().casefold() if isinstance(column, str) else column


def _find_column(frame, name):
    token = str(name).strip().casefold()
    return next((column for column in frame.columns if token in _column_tokens(column)), None)


def _available_columns(frame):
    return ", ".join(repr(column) for column in frame.columns)


def _column_series(frame, column):
    values = frame[column]
    if isinstance(values, pd.DataFrame):
        raise ValueError(
            f"Column {column!r} is not unique. Available columns: {_available_columns(frame)}."
        )
    return values


def _is_finite_numeric_column(frame, column):
    values = _column_series(frame, column)
    if is_bool_dtype(values.dtype) or not is_numeric_dtype(values.dtype):
        return False
    numeric = values.to_numpy(dtype=float, copy=False)
    return bool(np.isfinite(numeric).any())


def _technique_name(obj):
    supported = {"cv", "ca", "cp", "dpv", "echem"}
    for cls in type(obj).__mro__:
        name = cls.__name__.casefold()
        if name in supported:
            return name
    return "echem"


def _resolve_signal_column(obj, frame, requested):
    if isinstance(requested, Integral) and not isinstance(requested, bool):
        position = int(requested)
        if not -len(frame.columns) <= position < len(frame.columns):
            raise IndexError(
                f"Signal column position {position} is out of range. "
                f"Available columns: {_available_columns(frame)}."
            )
        column = frame.columns[position]
    elif isinstance(requested, str) and requested.strip().casefold() != "auto":
        column = _find_column(frame, requested)
        if column is None:
            raise ValueError(
                f"Signal column {requested!r} was not found. "
                f"Available columns: {_available_columns(frame)}."
            )
    elif isinstance(requested, str):
        technique = _technique_name(obj)
        preferred = "Potential" if technique == "cp" else "Current"
        if technique != "echem":
            column = _find_column(frame, preferred)
            if column is None:
                raise ValueError(
                    f"Automatic signal selection for {technique.upper()} requires a "
                    f"{preferred!r} column. Available columns: {_available_columns(frame)}."
                )
        else:
            column = None
            for name in ("Current", "Potential"):
                candidate = _find_column(frame, name)
                if candidate is not None and _is_finite_numeric_column(frame, candidate):
                    column = candidate
                    break
            if column is None:
                for candidate in frame.columns:
                    if _column_tokens(candidate) & {"time", "segment"}:
                        continue
                    if _is_finite_numeric_column(frame, candidate):
                        column = candidate
                        break
            if column is None:
                raise ValueError(
                    "Automatic signal selection could not find a finite numeric column "
                    "other than Time or Segment. "
                    f"Available columns: {_available_columns(frame)}."
                )
    else:
        raise ValueError(
            "Signal column must be 'auto', a column name, or an integer DataFrame "
            f"position. Available columns: {_available_columns(frame)}."
        )

    values = _column_series(frame, column)
    if is_bool_dtype(values.dtype) or not is_numeric_dtype(values.dtype):
        raise ValueError(
            f"Signal column {column!r} must be numeric. "
            f"Available columns: {_available_columns(frame)}."
        )
    return column


def _unit_for_column(obj, column, default=""):
    units = getattr(obj, "units", None) or {}
    try:
        if column in units:
            return _normalize_unit_metadata(units[column], default)
    except TypeError:
        pass
    selected_key = _casefold_column_key(column)
    for key, unit in units.items():
        if _casefold_column_key(key) == selected_key:
            return _normalize_unit_metadata(unit, default)
    column_tokens = _column_tokens(column)
    for canonical_name in ("Current", "Potential"):
        if canonical_name.casefold() in column_tokens and canonical_name in units:
            return _normalize_unit_metadata(units[canonical_name], default)
    return default


def _normalize_unit_metadata(value, default=""):
    if value is None:
        return default
    try:
        missing = pd.isna(value)
    except (TypeError, ValueError):
        missing = False
    if isinstance(missing, (bool, np.bool_)) and missing:
        return default
    return str(value)


def _recorded_time_unit(obj, time_column):
    units = getattr(obj, "units", None) or {}
    try:
        if time_column in units:
            return _normalize_unit_metadata(units[time_column], "s")
    except TypeError:
        pass
    if "Time" in units:
        return _normalize_unit_metadata(units["Time"], "s")

    selected_key = _casefold_column_key(time_column)
    matches = [
        (key, _normalize_unit_metadata(unit, "s"))
        for key, unit in units.items()
        if _casefold_column_key(key) == selected_key
    ]
    if matches:
        matched_units = list(dict.fromkeys(unit for _, unit in matches))
        if len(matched_units) == 1:
            return matched_units[0]
        details = ", ".join(f"{key!r}={unit!r}" for key, unit in matches)
        raise ValueError(
            f"Ambiguous recorded Time-unit metadata for column {time_column!r}: "
            f"case-insensitive matches {details} conflict. Add one exact units key "
            "for the recorded Time column."
        )
    return "s"


def _recorded_time_seconds(obj, frame, time_column):
    values = _column_series(frame, time_column)
    if is_bool_dtype(values.dtype) or not is_numeric_dtype(values.dtype):
        raise ValueError(f"Recorded Time column {time_column!r} must be numeric.")
    source_unit = _recorded_time_unit(obj, time_column)
    try:
        _prefix, source_base = extract_prefix_and_base(source_unit)
        if source_base != "s":
            raise ValueError(f"{source_unit!r} is not a seconds unit")
        factor = get_conversion_factor(source_unit, "s")
    except (TypeError, ValueError, KeyError) as exc:
        raise ValueError(
            f"Could not convert recorded Time column {time_column!r} from unit "
            f"{source_unit!r} to seconds."
        ) from exc
    return values.to_numpy(dtype=float, copy=True) * float(factor)


def _potential_values_volts(obj, potential_column, values):
    source_unit = _unit_for_column(obj, potential_column, "V")
    try:
        _prefix, source_base = extract_prefix_and_base(source_unit)
        if source_base != "V":
            raise ValueError(f"{source_unit!r} is not a potential unit")
        factor = float(get_conversion_factor(source_unit, "V"))
        if not np.isfinite(factor) or factor <= 0:
            raise ValueError(f"{source_unit!r} has an invalid conversion factor")
    except (TypeError, ValueError, KeyError) as exc:
        raise ValueError(
            "CV spectrum could not reconstruct time because Potential column "
            f"{potential_column!r} uses unit {source_unit!r}, which cannot be "
            "converted to V. Set compatible unit metadata such as 'V' or 'mV', "
            "or provide a recorded Time column."
        ) from exc
    return np.asarray(values, dtype=float) * factor


def _selected_segments(options):
    selected = options.get("segments", "all")
    if selected == "all":
        return None
    if isinstance(selected, Integral) and not isinstance(selected, bool):
        return [int(selected)]
    return list(dict.fromkeys(int(segment) for segment in selected))


def _positive_timing_value(obj, attribute, technique):
    value = getattr(obj, attribute, None)
    try:
        resolved = float(value)
    except (TypeError, ValueError) as exc:
        resolved = np.nan
        error = exc
    else:
        error = None
    if isinstance(value, bool) or not np.isfinite(resolved) or resolved <= 0:
        message = (
            f"{technique} spectrum requires a finite positive {attribute} in seconds."
            if attribute == "pulse_period"
            else f"{technique} spectrum requires a finite positive {attribute} in V/s."
        )
        raise ValueError(message) from error
    return resolved


def _trace_label(obj, segment):
    name = str(getattr(obj, "name", "Spectrum"))
    return name if segment is None else f"{name}, segment {segment}"


def _finalize_trace(
    obj,
    options,
    *,
    segment,
    raw_time,
    raw_signal,
    source_indices,
    signal_column,
    signal_unit,
    time_source,
):
    time = np.asarray(raw_time, dtype=float)
    values = np.asarray(raw_signal, dtype=float)
    indices = np.asarray(source_indices)

    finite = np.isfinite(time) & np.isfinite(values)
    time = time[finite]
    values = values[finite]
    indices = indices[finite]

    selected_range = options.get("time range")
    if selected_range is not None:
        lower, upper = (float(bound) for bound in selected_range)
        selected = (time >= lower) & (time <= upper)
        time = time[selected]
        values = values[selected]
        indices = indices[selected]

    label = _trace_label(obj, segment)
    if len(time) < 8:
        raise ValueError(
            f"Spectrum trace {label!r} requires at least 8 finite time/signal pairs; "
            f"found {len(time)}."
        )

    original_time = time.copy()
    original_signal = values.copy()
    with np.errstate(over="ignore", invalid="ignore"):
        original_intervals = np.diff(original_time)
    if not np.all(np.isfinite(original_intervals)):
        raise ValueError(
            f"Sampling intervals for spectrum trace {label!r} must be finite and "
            "strictly positive; recorded Time subtraction overflowed or produced "
            "invalid values. Rescale or repair the Time data."
        )
    if np.any(original_intervals <= 0):
        raise ValueError(
            f"Time for spectrum trace {label!r} must be strictly increasing after "
            "finite filtering; duplicate or decreasing values were found."
        )

    median_interval = float(np.median(original_intervals))
    if not np.isfinite(median_interval) or median_interval <= 0:
        raise ValueError(
            f"Median sampling interval for spectrum trace {label!r} must be finite "
            "and positive. Rescale or repair the recorded Time data."
        )
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        deviation = float(
            np.max(np.abs(original_intervals - median_interval)) / median_interval
        )
        sample_rate = float(1.0 / median_interval)
    if not np.isfinite(deviation) or deviation < 0:
        raise ValueError(
            f"Relative sampling-interval deviation for spectrum trace {label!r} "
            "must be finite and nonnegative. Rescale or repair the recorded Time data."
        )
    if not np.isfinite(sample_rate) or sample_rate <= 0:
        raise ValueError(
            f"Resolved sample rate for spectrum trace {label!r} must be finite and "
            "positive. Use a coarser recorded Time unit or repair the Time data."
        )
    tolerance = float(options.get("sampling tolerance", 0.01))
    resample_requested = bool(options.get("resample", False))
    needs_resampling = False
    if deviation > tolerance:
        with np.errstate(over="ignore", invalid="ignore"):
            excess_deviation = float(deviation - tolerance)
        tolerance_guard = _floating_tolerance(deviation, tolerance)
        needs_resampling = bool(
            not np.isfinite(excess_deviation)
            or excess_deviation > tolerance_guard
        )
    if needs_resampling and not resample_requested:
        raise ValueError(
            f"Sampling for spectrum trace {label!r} has relative interval deviation "
            f"{deviation:.6g}, above the tolerance {tolerance:.6g}. Set "
            "resample=True to linearly interpolate onto the median-interval grid."
        )

    resampled = needs_resampling and resample_requested
    if resampled:
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            ratio = (original_time[-1] - original_time[0]) / median_interval
        if np.isfinite(ratio):
            nearest_interval_count = round(float(ratio))
            if math.isclose(
                float(ratio),
                nearest_interval_count,
                rel_tol=1e-12,
                abs_tol=1e-12,
            ):
                interval_count = nearest_interval_count
            else:
                interval_count = math.floor(float(ratio))
            resolved_count = interval_count + 1
        else:
            resolved_count = None
        if resolved_count is None or resolved_count > _MAX_RESAMPLED_POINTS:
            estimate = "more than" if resolved_count is None else f"{resolved_count:,}"
            raise ValueError(
                f"Resampling spectrum trace {label!r} would create {estimate} points, "
                f"exceeding the {_MAX_RESAMPLED_POINTS:,}-point limit. Narrow 'time "
                "range' or repair acquisition timestamps before retrying."
            )
        resolved_time = (
            original_time[0]
            + np.arange(resolved_count, dtype=float) * median_interval
        )
        endpoint_scale = max(
            abs(float(original_time[0])),
            abs(float(original_time[-1])),
            abs(median_interval),
            1.0,
        )
        endpoint_tolerance = endpoint_scale * 1e-12
        resolved_time = resolved_time[
            resolved_time <= original_time[-1] + endpoint_tolerance
        ]
        if len(resolved_time) and math.isclose(
            float(resolved_time[-1]),
            float(original_time[-1]),
            rel_tol=1e-12,
            abs_tol=endpoint_tolerance,
        ):
            resolved_time[-1] = original_time[-1]
        resolved_signal = np.interp(resolved_time, original_time, original_signal)
        if len(resolved_time) < 8:
            raise ValueError(
                f"Resampling spectrum trace {label!r} produced fewer than 8 points."
            )
    else:
        resolved_time = original_time.copy()
        resolved_signal = original_signal.copy()

    return _SpectrumTrace(
        segment=segment,
        label=label,
        time=resolved_time,
        signal=resolved_signal,
        signal_column=signal_column,
        signal_unit=signal_unit,
        time_source=time_source,
        source_indices=indices.copy(),
        original_time=original_time,
        original_signal=original_signal,
        original_intervals=original_intervals,
        resolved_intervals=np.diff(resolved_time),
        median_interval=median_interval,
        relative_interval_deviation=deviation,
        sample_rate=sample_rate,
        resampled=resampled,
    )


def _resolve_cv_traces(obj, frame, options, signal_column, signal_unit, time_column):
    potential_column = _find_column(frame, "Potential")
    if potential_column is None:
        raise ValueError(
            "CV spectrum requires a stored Potential column for physical segment "
            f"resolution. Available columns: {_available_columns(frame)}."
        )
    potential_series = _column_series(frame, potential_column)
    if is_bool_dtype(potential_series.dtype) or not is_numeric_dtype(potential_series.dtype):
        raise ValueError(f"CV Potential column {potential_column!r} must be numeric.")
    potential = potential_series.to_numpy(dtype=float, copy=True)
    try:
        segment_definitions = split_cv_potential_segments(potential)
    except ValueError as exc:
        raise ValueError(f"CV spectrum could not split the stored Potential trace: {exc}.") from exc

    available = [definition["segment"] for definition in segment_definitions]
    selected = _selected_segments(options)
    if selected is None:
        selected = available
    missing = [segment for segment in selected if segment not in available]
    if missing:
        raise ValueError(
            f"Requested CV segment(s) {missing} are unavailable; available segments: "
            f"{available}."
        )

    if time_column is not None:
        all_time = _recorded_time_seconds(obj, frame, time_column)
        potential_volts = None
        time_source = "recorded"
        warning_messages = []
        scan_rate = None
    else:
        all_time = None
        potential_volts = _potential_values_volts(
            obj,
            potential_column,
            potential,
        )
        time_source = "reconstructed from CV potential and scan rate"
        warning_messages = [
            "CV time was reconstructed from raw Potential and scan rate; verify the "
            "scan rate and acquisition program before interpreting frequencies."
        ]
        scan_rate = _positive_timing_value(obj, "scan_rate", "CV")

    traces = []
    for definition in segment_definitions:
        segment = definition["segment"]
        if segment not in selected:
            continue
        start, stop = definition["start"], definition["stop"]
        segment_frame = frame.iloc[start:stop]
        if all_time is None:
            segment_potential = potential_volts[start:stop]
            delta_time = np.abs(np.diff(segment_potential)) / scan_rate
            raw_time = np.concatenate(([0.0], np.cumsum(delta_time)))
        else:
            raw_time = all_time[start:stop].copy()
            finite_time = np.flatnonzero(np.isfinite(raw_time))
            if len(finite_time):
                raw_time -= raw_time[finite_time[0]]
        raw_signal = _column_series(segment_frame, signal_column).to_numpy(
            dtype=float,
            copy=True,
        )
        traces.append(
            _finalize_trace(
                obj,
                options,
                segment=segment,
                raw_time=raw_time,
                raw_signal=raw_signal,
                source_indices=segment_frame.index.to_numpy(copy=True),
                signal_column=signal_column,
                signal_unit=signal_unit,
                time_source=time_source,
            )
        )
    return traces, warning_messages


def _non_cv_groups(frame, options):
    selected = _selected_segments(options)
    segment_column = _find_column(frame, "Segment")
    if segment_column is None:
        if selected is None:
            return [(None, np.arange(len(frame), dtype=int))]
        raise ValueError(
            "Numeric segment selection for a non-CV spectrum requires a stored "
            f"Segment column. Available columns: {_available_columns(frame)}."
        )
    segment_series = _column_series(frame, segment_column)
    original_segment_values = segment_series.to_numpy(dtype=object, copy=False)
    boolean_rows = np.fromiter(
        (
            isinstance(value, (bool, np.bool_))
            for value in original_segment_values
        ),
        dtype=bool,
        count=len(original_segment_values),
    )
    segment_values = pd.to_numeric(
        segment_series,
        errors="coerce",
    ).to_numpy(dtype=float)
    valid = (
        np.isfinite(segment_values)
        & (segment_values > 0)
        & (segment_values == np.floor(segment_values))
        & ~boolean_rows
    )
    if not np.any(valid):
        raise ValueError(
            f"Stored Segment column {segment_column!r} has no valid segment values; "
            "expected positive integers in every row."
        )
    if not np.all(valid):
        invalid_positions = np.flatnonzero(~valid)
        invalid_indices = frame.index.to_numpy(copy=False)[invalid_positions]
        preview = invalid_indices[:5].tolist()
        suffix = "..." if len(invalid_indices) > len(preview) else ""
        raise ValueError(
            f"Stored Segment column {segment_column!r} contains invalid or missing "
            "values; every row must contain a positive integer. Invalid row indices: "
            f"{preview}{suffix}."
        )

    available = list(dict.fromkeys(int(value) for value in segment_values))
    if selected is None:
        selected = available
    missing = [segment for segment in selected if not np.any(segment_values == segment)]
    if missing:
        raise ValueError(
            f"Requested segment(s) {missing} are unavailable; available segments: "
            f"{available}."
        )
    return [
        (segment, np.flatnonzero(segment_values == segment))
        for segment in selected
    ]


def _resolve_non_cv_traces(
    obj,
    frame,
    options,
    signal_column,
    signal_unit,
    time_column,
    technique,
):
    if time_column is not None:
        all_time = _recorded_time_seconds(obj, frame, time_column)
        time_source = "recorded"
        warning_messages = []
    elif technique == "dpv":
        pulse_period = _positive_timing_value(obj, "pulse_period", "DPV")
        all_time = np.arange(len(frame), dtype=float) * pulse_period
        time_source = "reconstructed from DPV pulse period"
        warning_messages = [
            "DPV time was reconstructed from pulse period; verify the pulse timing "
            "before interpreting frequencies."
        ]
    else:
        raise ValueError(
            f"{technique.upper()} spectrum requires a recorded Time column; no "
            "physical or reconstructable time axis is available."
        )

    all_signal = _column_series(frame, signal_column).to_numpy(dtype=float, copy=True)
    source_indices = frame.index.to_numpy(copy=True)
    traces = []
    for segment, positions in _non_cv_groups(frame, options):
        traces.append(
            _finalize_trace(
                obj,
                options,
                segment=segment,
                raw_time=all_time[positions],
                raw_signal=all_signal[positions],
                source_indices=source_indices[positions],
                signal_column=signal_column,
                signal_unit=signal_unit,
                time_source=time_source,
            )
        )
    return traces, warning_messages


def _resolve_spectrum_traces(obj, options):
    """Resolve finite, physically timed traces without computing a spectrum."""
    if isinstance(options, SpectrumOptions):
        options = options.to_options_dict()
    else:
        options = dict(options)

    frame = getattr(obj, "data", None)
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise ValueError("Spectrum analysis requires a non-empty pandas DataFrame in obj.data.")

    signal_column = _resolve_signal_column(obj, frame, options.get("column", "auto"))
    signal_unit = _unit_for_column(obj, signal_column, "")
    time_column = _find_column(frame, "Time")
    technique = _technique_name(obj)

    if technique == "cv":
        return _resolve_cv_traces(
            obj,
            frame,
            options,
            signal_column,
            signal_unit,
            time_column,
        )
    return _resolve_non_cv_traces(
        obj,
        frame,
        options,
        signal_column,
        signal_unit,
        time_column,
        technique,
    )


def _window_values(window, length):
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            values = signal.get_window(window, length, fftbins=True)
    except (RuntimeWarning, FloatingPointError):
        raise ValueError(
            f"Invalid 'window' option {window!r}; the SciPy window specification "
            "produced invalid or nonfinite numerical values."
        ) from None
    except (TypeError, ValueError):
        try:
            values = np.asarray(window)
        except (TypeError, ValueError):
            raise ValueError(
                f"Invalid 'window' option {window!r}; choose a SciPy-compatible "
                "window name or specification."
            ) from None
        if np.iscomplexobj(values) or any(
            isinstance(value, (complex, np.complexfloating))
            for value in values.flat
        ):
            raise ValueError(
                f"Invalid 'window' option {window!r}; window coefficients must be "
                "real finite coefficients."
            ) from None
        try:
            values = np.asarray(values, dtype=float)
        except (TypeError, ValueError):
            raise ValueError(
                f"Invalid 'window' option {window!r}; choose a SciPy-compatible "
                "window name or specification."
            ) from None
        if values.ndim != 1:
            raise ValueError(
                f"Invalid 'window' option {window!r}; a custom window must contain "
                f"exactly {length} finite values in one dimension; received shape "
                f"{values.shape}."
            ) from None
        if len(values) != length:
            raise ValueError(
                f"Invalid 'window' option; a custom window must contain exactly "
                f"{length} finite values; received {len(values)}."
            ) from None

    values = np.asarray(values)
    if np.iscomplexobj(values) or any(
        isinstance(value, (complex, np.complexfloating))
        for value in values.flat
    ):
        raise ValueError(
            f"Invalid 'window' option {window!r}; window coefficients must be real "
            "finite coefficients."
        ) from None
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or len(values) != length or not np.all(np.isfinite(values)):
        raise ValueError(
            f"Invalid 'window' option {window!r}; the resolved window must contain "
            f"exactly {length} finite values."
        )
    return values


def _floating_tolerance(*values):
    finite_values = []
    for value in values:
        if not np.isscalar(value):
            raise ValueError("Floating tolerance requires finite scalar values.")
        try:
            numeric = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError("Floating tolerance requires finite scalar values.") from exc
        if not np.isfinite(numeric):
            raise ValueError("Floating tolerance requires finite scalar values.")
        finite_values.append(numeric)
    if not finite_values:
        raise ValueError("Floating tolerance requires at least one finite scalar value.")

    magnitude = max(abs(value) for value in finite_values)
    epsilon_floor = np.finfo(float).eps * magnitude
    epsilon_floor = max(epsilon_floor, np.finfo(float).eps)
    with np.errstate(over="ignore", invalid="ignore"):
        spacing = abs(float(np.spacing(magnitude)))
    if not np.isfinite(spacing):
        spacing = epsilon_floor
    return 16 * max(spacing, epsilon_floor)


def _frequency_selection(frequency, options, nyquist, label):
    selected = np.ones(len(frequency), dtype=bool)
    if not options.get("include dc", False):
        selected &= frequency != 0

    frequency_range = options.get("frequency range")
    if frequency_range is not None:
        lower, upper = (float(bound) for bound in frequency_range)
        nyquist_tolerance = _floating_tolerance(nyquist)
        if upper > nyquist + nyquist_tolerance:
            raise ValueError(
                f"Requested maximum frequency {upper:g} Hz exceeds the Nyquist "
                f"frequency {nyquist:g} Hz available for spectrum trace {label!r}."
            )
        selection_tolerance = _floating_tolerance(lower, upper, nyquist)
        selected &= (frequency >= lower - selection_tolerance) & (
            frequency <= upper + selection_tolerance
        )

    if not np.any(selected):
        raise ValueError(
            f"Spectrum filtering left no frequency bins for trace {label!r}; "
            "enable DC or choose a frequency range containing resolved bins."
        )
    return selected


def _validate_transform_points(nfft, *, method, requested_nfft, sample_count):
    if nfft <= _MAX_TRANSFORM_POINTS:
        return
    if requested_nfft is None:
        source = (
            f"Implicit {method} nfft {nfft:,} from the selected sample count "
            f"{sample_count:,}"
        )
    else:
        source = f"Requested nfft {nfft:,}"
    raise ValueError(
        f"{source} exceeds the {_MAX_TRANSFORM_POINTS:,}-point limit. Reduce nfft "
        "or narrow the selected trace before retrying."
    )


def _spectrum_numerical_error(method, trace, window, detail):
    window_description = _summarize_spectrum_window(window)
    return ValueError(
        f"{method} spectrum for trace {trace.label!r} with window "
        f"{window_description!r} produced nonfinite numerical output ({detail}). "
        "Reduce the signal or window magnitude and retry."
    )


def _compute_welch(trace, options, warning_messages):
    sample_count = len(trace.signal)
    requested_nperseg = options.get("points per window", "auto")
    if requested_nperseg == "auto":
        nperseg = min(256, sample_count)
    else:
        nperseg = int(requested_nperseg)
        if nperseg > sample_count:
            warning_messages.append(
                f"Requested points per window ({nperseg}) exceeds the {sample_count} "
                f"samples in spectrum trace {trace.label!r}; reduced to {sample_count}. "
                "Use a longer trace to retain the requested Welch window length."
            )
            nperseg = sample_count

    noverlap = math.floor(float(options.get("overlap", 0.5)) * nperseg)
    if noverlap >= nperseg:
        raise ValueError(
            f"Welch overlap resolves to {noverlap} points, but it must be less than "
            f"nperseg={nperseg}."
        )

    requested_nfft = options.get("nfft")
    nfft = nperseg if requested_nfft is None else int(requested_nfft)
    if nfft < nperseg:
        raise ValueError(f"Welch nfft must be at least nperseg={nperseg}; received {nfft}.")
    _validate_transform_points(
        nfft,
        method="Welch",
        requested_nfft=requested_nfft,
        sample_count=sample_count,
    )

    window = options.get("window", "hann")
    window_values = _window_values(window, nperseg)
    with np.errstate(over="ignore", invalid="ignore"):
        window_power = float(np.sum(window_values**2))
    if not np.isfinite(window_power) or window_power <= 0:
        raise ValueError(
            "Welch window must have finite positive power sum(window**2). "
            "Choose a finite, nonzero window."
        )
    if not np.all(np.isfinite(trace.signal)):
        raise _spectrum_numerical_error(
            "Welch",
            trace,
            window,
            "input signal",
        )
    sample_rate = 1.0 / trace.median_interval
    detrend = options.get("detrend", "constant")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            frequency, psd = signal.welch(
                trace.signal,
                fs=sample_rate,
                window=window_values,
                nperseg=nperseg,
                noverlap=noverlap,
                nfft=nfft,
                detrend=False if detrend is None else detrend,
                return_onesided=True,
                scaling="density",
            )
    except (RuntimeWarning, FloatingPointError):
        raise _spectrum_numerical_error(
            "Welch",
            trace,
            window,
            "backend numerical warning",
        ) from None

    frequency = np.asarray(frequency)
    psd = np.asarray(psd)
    if not np.all(np.isfinite(frequency)) or not np.all(np.isfinite(psd)):
        raise _spectrum_numerical_error(
            "Welch",
            trace,
            window,
            "frequency or PSD",
        )

    step = nperseg - noverlap
    window_count = 1 + (sample_count - nperseg) // step
    if window_count < 2:
        warning_messages.append(
            f"Welch spectrum for trace {trace.label!r} has fewer than two complete "
            "contributing windows; use a longer trace or fewer points per window "
            "for meaningful Welch averaging."
        )

    nyquist = sample_rate / 2.0
    selected = _frequency_selection(frequency, options, nyquist, trace.label)
    return _SpectrumComputation(
        segment=trace.segment,
        label=trace.label,
        signal_unit=trace.signal_unit,
        method="welch",
        frequency=frequency[selected],
        psd=psd[selected],
        sample_count=sample_count,
        sample_rate=sample_rate,
        nyquist=nyquist,
        frequency_bin_spacing=sample_rate / nfft,
        nfft=nfft,
        window=window,
        detrend=detrend,
        nperseg=nperseg,
        noverlap=noverlap,
        window_count=window_count,
    )


def _compute_fft(trace, options):
    sample_count = len(trace.signal)
    requested_nfft = options.get("nfft")
    nfft = sample_count if requested_nfft is None else int(requested_nfft)
    if nfft < sample_count:
        raise ValueError(
            f"FFT nfft must be at least the selected signal length {sample_count}; "
            f"received {nfft}."
        )
    _validate_transform_points(
        nfft,
        method="FFT",
        requested_nfft=requested_nfft,
        sample_count=sample_count,
    )

    detrend = options.get("detrend", "constant")
    values = np.array(trace.signal, dtype=float, copy=True)
    if not np.all(np.isfinite(values)):
        raise _spectrum_numerical_error(
            "FFT",
            trace,
            options.get("window", "hann"),
            "input signal",
        )
    if detrend is not None:
        with np.errstate(over="ignore", invalid="ignore"):
            values = signal.detrend(values, type=detrend)
    if not np.all(np.isfinite(values)):
        raise _spectrum_numerical_error(
            "FFT",
            trace,
            options.get("window", "hann"),
            "detrended signal",
        )

    window = options.get("window", "hann")
    window_values = _window_values(window, sample_count)
    with np.errstate(over="ignore", invalid="ignore"):
        coherent_gain = float(np.mean(window_values))
    if not np.isfinite(coherent_gain) or coherent_gain <= 0:
        raise ValueError(
            f"FFT window {window!r} has nonfinite or nonpositive coherent gain and "
            "cannot produce calibrated amplitudes. Choose a window with positive mean."
        )

    with np.errstate(over="ignore", invalid="ignore"):
        windowed_values = values * window_values
    if not np.all(np.isfinite(windowed_values)):
        raise _spectrum_numerical_error(
            "FFT",
            trace,
            window,
            "windowed signal",
        )
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            transformed = np.fft.rfft(windowed_values, n=nfft)
    except (RuntimeWarning, FloatingPointError):
        raise _spectrum_numerical_error(
            "FFT",
            trace,
            window,
            "rfft numerical warning",
        ) from None
    transformed = np.asarray(transformed)
    if not np.all(np.isfinite(transformed.real)) or not np.all(
        np.isfinite(transformed.imag)
    ):
        raise _spectrum_numerical_error(
            "FFT",
            trace,
            window,
            "complex transform",
        )
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        amplitude = np.abs(transformed) / (sample_count * coherent_gain)
        if nfft % 2 == 0:
            amplitude[1:-1] *= 2
        else:
            amplitude[1:] *= 2
        phase = np.angle(transformed)
    if not np.all(np.isfinite(amplitude)) or not np.all(np.isfinite(phase)):
        raise _spectrum_numerical_error(
            "FFT",
            trace,
            window,
            "amplitude or phase",
        )
    sample_rate = 1.0 / trace.median_interval
    frequency = np.fft.rfftfreq(nfft, d=trace.median_interval)
    if not np.all(np.isfinite(frequency)):
        raise _spectrum_numerical_error(
            "FFT",
            trace,
            window,
            "frequency",
        )
    nyquist = sample_rate / 2.0
    selected = _frequency_selection(frequency, options, nyquist, trace.label)
    return _SpectrumComputation(
        segment=trace.segment,
        label=trace.label,
        signal_unit=trace.signal_unit,
        method="fft",
        frequency=frequency[selected],
        amplitude=amplitude[selected],
        phase=phase[selected],
        sample_count=sample_count,
        sample_rate=sample_rate,
        nyquist=nyquist,
        frequency_bin_spacing=sample_rate / nfft,
        nfft=nfft,
        window=window,
        detrend=detrend,
        coherent_gain=coherent_gain,
    )


def _compute_trace(trace, options, warning_messages):
    """Compute one immutable Welch or FFT spectrum without emitting warnings."""
    if isinstance(options, SpectrumOptions):
        options = options.to_options_dict()
    else:
        options = dict(options)

    method = options.get("method", "welch")
    if method == "welch":
        return _compute_welch(trace, options, warning_messages)
    if method == "fft":
        return _compute_fft(trace, options)
    raise ValueError(f"Unknown spectrum method {method!r}; expected 'welch' or 'fft'.")


def _squared_signal_unit(signal_unit):
    unit = _normalize_unit_metadata(signal_unit, "").strip()
    if not unit:
        return "signal^2"
    if unit.isalnum():
        return f"{unit}^2"
    return f"({unit})^2"


def _spectrum_value_column(signal_unit, method):
    # Unitless/unknown channels stay explicitly generic instead of implying A or V.
    if method == "welch":
        return f"PSD / {_squared_signal_unit(signal_unit)} Hz^-1"
    if method == "fft":
        return f"Amplitude / {signal_unit or 'signal unit'}"
    raise ValueError(f"Unknown spectrum method {method!r}; expected 'welch' or 'fft'.")


def _spectrum_units(signal_unit, method):
    value_column = _spectrum_value_column(signal_unit, method)
    units = {"Frequency / Hz": "Hz"}
    if method == "welch":
        units[value_column] = f"{_squared_signal_unit(signal_unit)} Hz^-1"
    else:
        units[value_column] = signal_unit or "signal unit"
        units["Phase / rad"] = "rad"
    return units


def _spectrum_table(computations, method):
    computations = list(computations)
    if not computations:
        raise ValueError("At least one spectrum computation is required to build a table.")
    if any(computation.method != method for computation in computations):
        raise ValueError(f"All spectrum computations must use method {method!r}.")

    signal_unit = computations[0].signal_unit
    if any(computation.signal_unit != signal_unit for computation in computations):
        raise ValueError("All spectrum computations in one table must use the same signal unit.")
    value_column = _spectrum_value_column(signal_unit, method)
    include_segment = not (
        len(computations) == 1 and computations[0].segment is None
    )

    frames = []
    for computation in computations:
        data = {}
        if include_segment:
            data["Segment"] = np.repeat(
                computation.segment,
                len(computation.frequency),
            )
        data["Frequency / Hz"] = computation.frequency
        if method == "welch":
            data[value_column] = computation.psd
        else:
            data[value_column] = computation.amplitude
            data["Phase / rad"] = computation.phase
        frames.append(pd.DataFrame(data))

    table = pd.concat(frames, ignore_index=True)
    table.attrs["units"] = _spectrum_units(signal_unit, method)
    return table


def _spectrum_trace_keys(traces):
    keys = []
    issued = set()
    next_suffix = {}
    for index, trace in enumerate(traces, start=1):
        if trace.segment is not None:
            base = f"Segment {trace.segment}"
        else:
            base = str(trace.label).strip() or f"Trace {index}"
        candidate = base
        suffix = next_suffix.get(base, 2)
        while candidate in issued:
            candidate = f"{base} ({suffix})"
            suffix += 1
        next_suffix[base] = suffix
        issued.add(candidate)
        keys.append(candidate)
    return keys


def _spectrum_values_equal(left, right):
    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        try:
            return bool(np.array_equal(left, right, equal_nan=True))
        except TypeError:
            return bool(np.array_equal(left, right))
    if (
        isinstance(left, Real)
        and not isinstance(left, (bool, np.bool_))
        and isinstance(right, Real)
        and not isinstance(right, (bool, np.bool_))
    ):
        left_value = float(left)
        right_value = float(right)
        if math.isnan(left_value) and math.isnan(right_value):
            return True
        return left_value == right_value
    try:
        equal = left == right
    except (TypeError, ValueError):
        return False
    return bool(equal) if np.isscalar(equal) else bool(np.all(equal))


def _spectrum_audit_value(values, trace_keys):
    values = list(values)
    if not values:
        return None
    first = values[0]
    if all(_spectrum_values_equal(first, value) for value in values[1:]):
        return first
    return OrderedDict(zip(trace_keys, values))


def _spectrum_peak_frequency(computation):
    ordinate = computation.psd if computation.method == "welch" else computation.amplitude
    positive = np.isfinite(ordinate) & (ordinate > 0)
    if not np.any(positive):
        return float("nan")
    maximum = np.max(ordinate[positive])
    tied = positive & (ordinate == maximum)
    # Retained frequencies are sorted ascending, so the first tie is the lowest.
    return float(computation.frequency[np.flatnonzero(tied)[0]])


def _spectrum_selected_segments(traces, resolved):
    selected = list(
        dict.fromkeys(trace.segment for trace in traces if trace.segment is not None)
    )
    return selected if selected else resolved.get("segments", "all")


def _spectrum_summary(computations, traces, resolved):
    computations = tuple(computations)
    traces = tuple(traces)
    if not computations or len(computations) != len(traces):
        raise ValueError(
            "Spectrum summaries require one computation per resolved trace."
        )
    trace_keys = _spectrum_trace_keys(traces)
    summary = {
        "analysis": "spectrum",
        "method": resolved["method"],
        "signal column": _spectrum_audit_value(
            (trace.signal_column for trace in traces), trace_keys
        ),
        "signal unit": _spectrum_audit_value(
            (trace.signal_unit for trace in traces), trace_keys
        ),
        "selected segments": _spectrum_selected_segments(traces, resolved),
        "time source": _spectrum_audit_value(
            (trace.time_source for trace in traces), trace_keys
        ),
        "time unit": _spectrum_audit_value(
            (trace.time_unit for trace in traces), trace_keys
        ),
        "original sample count": _spectrum_audit_value(
            (len(trace.original_signal) for trace in traces), trace_keys
        ),
        "sample count": _spectrum_audit_value(
            (computation.sample_count for computation in computations), trace_keys
        ),
        "sample interval / s": _spectrum_audit_value(
            (trace.median_interval for trace in traces), trace_keys
        ),
        "relative interval deviation": _spectrum_audit_value(
            (trace.relative_interval_deviation for trace in traces), trace_keys
        ),
        "sample rate / Hz": _spectrum_audit_value(
            (computation.sample_rate for computation in computations), trace_keys
        ),
        "Nyquist frequency / Hz": _spectrum_audit_value(
            (computation.nyquist for computation in computations), trace_keys
        ),
        "frequency bin spacing / Hz": _spectrum_audit_value(
            (computation.frequency_bin_spacing for computation in computations),
            trace_keys,
        ),
        "resampled": _spectrum_audit_value(
            (trace.resampled for trace in traces), trace_keys
        ),
        "peak frequency / Hz": _spectrum_audit_value(
            (_spectrum_peak_frequency(computation) for computation in computations),
            trace_keys,
        ),
    }
    if resolved["method"] == "welch":
        summary["points per window"] = _spectrum_audit_value(
            (computation.nperseg for computation in computations), trace_keys
        )
    return summary


def _spectrum_diagnostics(computations, traces, resolved):
    computations = _CompactDiagnosticSequence(computations)
    traces = _CompactDiagnosticSequence(traces)
    if not computations or len(computations) != len(traces):
        raise ValueError(
            "Spectrum diagnostics require one computation per resolved trace."
        )
    trace_keys = _spectrum_trace_keys(traces)
    diagnostics = {
        "traces": traces,
        "computations": computations,
        "method": resolved["method"],
        "window": _summarize_spectrum_window(
            _spectrum_audit_value(
                (computation.window for computation in computations), trace_keys
            )
        ),
        "detrend": _spectrum_audit_value(
            (computation.detrend for computation in computations), trace_keys
        ),
        "nfft": _spectrum_audit_value(
            (computation.nfft for computation in computations), trace_keys
        ),
    }
    if resolved["method"] == "welch":
        diagnostics.update(
            {
                "nperseg": _spectrum_audit_value(
                    (computation.nperseg for computation in computations), trace_keys
                ),
                "noverlap": _spectrum_audit_value(
                    (computation.noverlap for computation in computations), trace_keys
                ),
                "window count": _spectrum_audit_value(
                    (computation.window_count for computation in computations),
                    trace_keys,
                ),
            }
        )
    else:
        diagnostics["coherent gain"] = _spectrum_audit_value(
            (computation.coherent_gain for computation in computations), trace_keys
        )
    selected_segments = _spectrum_selected_segments(traces, resolved)
    if isinstance(selected_segments, list):
        selected_segments = tuple(selected_segments)
    time_range = resolved.get("time range")
    frequency_range = resolved.get("frequency range")
    parameters = OrderedDict(
        [
            ("method", resolved["method"]),
            (
                "signal column",
                _spectrum_audit_value(
                    (trace.signal_column for trace in traces), trace_keys
                ),
            ),
            (
                "signal unit",
                _spectrum_audit_value(
                    (trace.signal_unit for trace in traces), trace_keys
                ),
            ),
            ("detrend", diagnostics["detrend"]),
            ("window", _summarize_spectrum_window(diagnostics["window"])),
            ("resample", bool(resolved.get("resample", False))),
            (
                "resampled",
                _spectrum_audit_value(
                    (trace.resampled for trace in traces), trace_keys
                ),
            ),
            ("segments", selected_segments),
            ("time range", None if time_range is None else tuple(time_range)),
            (
                "frequency range",
                None if frequency_range is None else tuple(frequency_range),
            ),
            ("include dc", bool(resolved.get("include dc", False))),
            ("sampling tolerance", float(resolved["sampling tolerance"])),
            ("nfft", diagnostics["nfft"]),
        ]
    )
    if resolved["method"] == "welch":
        parameters.update(
            {
                "points per window": diagnostics["nperseg"],
                "overlap": float(resolved["overlap"]),
                "overlap points": diagnostics["noverlap"],
            }
        )
    diagnostics["parameters"] = parameters
    return diagnostics


def _format_spectrum_display_value(value, sig_figs):
    if isinstance(value, Mapping):
        return "; ".join(
            f"{key}: {_format_spectrum_display_value(item, sig_figs)}"
            for key, item in value.items()
        )
    if isinstance(value, np.ndarray):
        return f"array ({value.size} values)"
    if isinstance(value, (list, tuple)):
        return ", ".join(
            _format_spectrum_display_value(item, sig_figs) for item in value
        )
    if isinstance(value, (bool, np.bool_)):
        return "Yes" if value else "No"
    if isinstance(value, Integral):
        return str(int(value))
    if isinstance(value, Real):
        return format_sigfigs(float(value), sig_figs)
    if value is None:
        return "None"
    return str(value)


def _summarize_spectrum_window(window):
    if isinstance(window, Mapping):
        return "; ".join(
            f"{key}: {_summarize_spectrum_window(value)}"
            for key, value in window.items()
        )
    if isinstance(window, str):
        return window
    if (
        isinstance(window, tuple)
        and window
        and isinstance(window[0], str)
    ):
        parameters = []
        for value in window[1:]:
            if value is None or isinstance(value, (str, bytes)) or np.isscalar(value):
                parameters.append(str(value))
                continue
            try:
                size = len(value)
            except TypeError:
                parameters.append(type(value).__name__)
            else:
                parameters.append(f"{size} values")
        details = ", ".join(parameters)
        return f"{window[0]} ({details})" if details else window[0]
    try:
        size = len(window)
    except TypeError:
        return str(window)
    return f"custom window ({size} points)"


def _format_spectrum_range(value, sig_figs):
    if value is None:
        return "all"
    return ", ".join(
        _format_spectrum_display_value(item, sig_figs) for item in value
    )


def _format_spectrum_nfft(resolved, diagnostics, sig_figs):
    used = _format_spectrum_display_value(diagnostics["nfft"], sig_figs)
    if resolved.get("nfft") is None:
        return f"auto ({used} used)"
    return used


def _spectrum_parameter_display_table(result, resolved):
    sig_figs = resolved.get("sig figs", 4)
    diagnostics = result.diagnostics
    resampled = result.summary["resampled"]
    resampling_used = (
        any(bool(value) for value in resampled.values())
        if isinstance(resampled, Mapping)
        else bool(resampled)
    )
    if resolved.get("resample", False):
        resampling = "Enabled; used" if resampling_used else "Enabled; not used"
    else:
        resampling = "Disabled"

    rows = [
        ("Method", "Welch" if resolved["method"] == "welch" else "FFT"),
        (
            "Signal Column",
            _format_spectrum_display_value(
                result.summary["signal column"], sig_figs
            ),
        ),
        (
            "Signal Unit",
            _format_spectrum_display_value(result.summary["signal unit"], sig_figs)
            or "signal unit",
        ),
        (
            "Detrend",
            _format_spectrum_display_value(diagnostics["detrend"], sig_figs),
        ),
        ("Window", _summarize_spectrum_window(diagnostics["window"])),
        ("Resampling", resampling),
        (
            "Segments",
            _format_spectrum_display_value(
                result.summary["selected segments"], sig_figs
            ),
        ),
        ("Time Range", _format_spectrum_range(resolved.get("time range"), sig_figs)),
        (
            "Frequency Range",
            _format_spectrum_range(resolved.get("frequency range"), sig_figs),
        ),
        (
            "Include DC",
            _format_spectrum_display_value(resolved.get("include dc", False), sig_figs),
        ),
        (
            "Sampling Tolerance",
            _format_spectrum_display_value(
                resolved.get("sampling tolerance"), sig_figs
            ),
        ),
        ("NFFT", _format_spectrum_nfft(resolved, diagnostics, sig_figs)),
    ]
    if resolved["method"] == "welch":
        points = diagnostics["nperseg"]
        requested = resolved.get("points per window", "auto")
        used = _format_spectrum_display_value(points, sig_figs)
        if requested == "auto":
            points_text = f"auto ({used} used)"
        elif _spectrum_values_equal(requested, points):
            points_text = used
        else:
            points_text = f"{requested} requested; {used} used"
        overlap = _format_spectrum_display_value(resolved.get("overlap"), sig_figs)
        overlap_points = _format_spectrum_display_value(
            diagnostics["noverlap"], sig_figs
        )
        rows.extend(
            [
                ("Points per Window", points_text),
                ("Overlap", f"{overlap} ({overlap_points} points)"),
            ]
        )
    return pd.DataFrame(rows, columns=["Setting", "Value"])


def _spectrum_summary_display_table(result, resolved):
    sig_figs = resolved.get("sig figs", 4)
    traces = result.diagnostics["traces"]
    computations = result.diagnostics["computations"]
    keys = _spectrum_trace_keys(traces)
    rows = []
    for key, trace, computation in zip(keys, traces, computations):
        rows.append(
            {
                "Segment/Trace": key,
                "Time Source": trace.time_source,
                "Samples": computation.sample_count,
                "Sample Rate / Hz": computation.sample_rate,
                "Nyquist Frequency / Hz": computation.nyquist,
                "Frequency Bin Spacing / Hz": computation.frequency_bin_spacing,
                "Resampled": trace.resampled,
                "Peak Frequency / Hz": _spectrum_peak_frequency(computation),
            }
        )
    display = pd.DataFrame(rows)
    for column in display.columns:
        if column in {"Segment/Trace", "Time Source"}:
            continue
        if column == "Peak Frequency / Hz":
            display[column] = display[column].map(
                lambda value: (
                    "not detected"
                    if isinstance(value, Real) and not np.isfinite(float(value))
                    else _format_spectrum_display_value(value, sig_figs)
                )
            )
        else:
            display[column] = display[column].map(
                lambda value: _format_spectrum_display_value(value, sig_figs)
            )
    if len(display) != 1:
        return display
    row = display.iloc[0]
    return pd.DataFrame(
        {
            "Setting": display.columns,
            "Value": [row[column] for column in display.columns],
        }
    )


def _spectrum_data_display_table(table, resolved):
    sig_figs = resolved.get("sig figs", 4)
    display = table.copy(deep=True)
    for column in display.columns:
        display[column] = display[column].map(
            lambda value: _format_spectrum_display_value(value, sig_figs)
        )
    return display


def _spectrum_equation_bundle(method):
    if method == "welch":
        return {
            "symbolic latex": (
                r"S_{xx}(f_k)=\frac{c_k}{K}\sum_{j=1}^{K}"
                r"\frac{\left|\sum_{n=0}^{N-1}w_nx_{j,n}"
                r"e^{-i2\pi kn/M}\right|^2}{f_s\sum_{n=0}^{N-1}w_n^2}"
            ),
            "definitions latex": (
                r"N=\mathrm{samples/window};\ M=\mathrm{nfft};\ "
                r"K=\mathrm{Welch\ window\ count};\ c_k=1\ \mathrm{at\ DC\ and\ "
                r"Nyquist},\ 2\ \mathrm{for\ interior\ one-sided\ bins};\ "
                r"f_s\ \mathrm{in\ Hz};\ S_{xx}\ \mathrm{in\ signal^2\,Hz^{-1}}"
            ),
            "symbolic": (
                "Sxx(f_k) = c_k/K sum_(j=1)^K |sum_(n=0)^(N-1) w_n "
                "x_(j,n) exp(-i 2 pi k n / M)|^2 / "
                "(fs sum_(n=0)^(N-1) w_n^2)"
            ),
            "definitions": (
                "N is samples per window; M is nfft; K is the number of Welch "
                "windows; c_k is 1 at DC and Nyquist and 2 for interior one-sided "
                "bins; fs is the sample rate in Hz; Sxx has signal^2/Hz units"
            ),
        }
    if method == "fft":
        return {
            "symbolic latex": (
                r"X_k=\sum_{n=0}^{N-1}x_nw_n e^{-i2\pi kn/M},\qquad "
                r"A_k=\frac{c_k|X_k|}{N\,G_w}"
            ),
            "definitions latex": (
                r"N=\mathrm{sample\ count};\ M=\mathrm{nfft};\ "
                r"G_w=N^{-1}\sum_n w_n;\ c_k=1\ \mathrm{at\ DC\ and\ Nyquist},\ "
                r"2\ \mathrm{for\ interior\ one-sided\ bins}"
            ),
            "symbolic": (
                "X_k = sum_(n=0)^(N-1) x_n w_n exp(-i 2 pi k n / M); "
                "A_k = c_k |X_k| / (N Gw)"
            ),
            "definitions": (
                "N is the sample count; M is nfft; Gw = N^-1 sum_n w_n; c_k is "
                "1 at DC and Nyquist and 2 for interior one-sided bins"
            ),
        }
    raise ValueError(f"Unknown spectrum method {method!r}; expected 'welch' or 'fft'.")


def _spectrum_notebook_rich_display_available():
    if not _can_rich_table_display():
        return False
    try:
        from IPython import get_ipython

        shell = get_ipython()
    except Exception:
        return False
    if shell is None:
        return False

    shell_type_names = {cls.__name__ for cls in type(shell).__mro__}
    if any("terminal" in name.casefold() for name in shell_type_names):
        return False
    if "ZMQInteractiveShell" in shell_type_names:
        return True
    return getattr(shell, "kernel", None) is not None


def _print_spectrum_result(result, resolved):
    rich_display = bool(
        resolved.get("pretty print", True)
        and _spectrum_notebook_rich_display_available()
    )
    display_options = dict(resolved)
    if not rich_display:
        display_options["pretty print"] = False

    _display_table(
        _spectrum_parameter_display_table(result, resolved),
        display_options,
        title="Spectrum Analysis Parameters",
        index=False,
    )
    equation = _spectrum_equation_bundle(resolved["method"])
    if rich_display:
        from .analysis_batch import _display_analysis_equation

        _display_analysis_equation(
            r"\text{Spectrum analysis equations:}",
            "Spectrum Analysis Equations",
            equation,
            resolved=False,
            compact=False,
            include_definitions=True,
        )
    else:
        print("\nSpectrum Analysis Equations:")
        print("  " + equation["symbolic"])
        print("  " + equation["definitions"])
    _display_table(
        _spectrum_summary_display_table(result, resolved),
        display_options,
        title="Spectrum Analysis Summary",
        index=False,
    )
    if resolved.get("print all", False):
        _display_table(
            _spectrum_data_display_table(result.table, resolved),
            display_options,
            title="Spectrum Analysis Data",
            index=False,
        )


def _mathtext_roman(text):
    escaped = str(text).strip()
    for character in ("_", "%", "#", "&", "{", "}"):
        escaped = escaped.replace(character, rf"\{character}")
    escaped = escaped.replace(" ", r"\ ")
    return rf"\mathrm{{{escaped}}}"


def _signal_unit_text(unit):
    normalized = _normalize_unit_metadata(unit, "")
    return str(normalized or "").strip()


def _spectrum_axis_ylabel(computation):
    unit = _signal_unit_text(computation.signal_unit)
    if computation.method == "welch":
        signal_unit = _mathtext_roman(unit or "signal units")
        return rf"PSD (${signal_unit}^{{2}}\,\mathrm{{Hz}}^{{-1}}$)"
    if computation.method == "fft":
        if not unit:
            return "Amplitude (signal units)"
        return rf"Amplitude (${_mathtext_roman(unit)}$)"
    raise ValueError(
        f"Unknown spectrum method {computation.method!r}; expected 'welch' or 'fft'."
    )


def _time_domain_ylabel(trace):
    signal_name = str(trace.signal_column)
    unit = _signal_unit_text(trace.signal_unit)
    if not unit:
        return f"{signal_name} (signal units)"
    return rf"{signal_name} (${_mathtext_roman(unit)}$)"


def _spectrum_trace_label(item):
    if item.segment is not None:
        return f"Segment {item.segment}"
    return item.label


def _spectrum_plot_titles(method, options):
    title_option = options.get("title", True)
    if title_option in (False, None, ""):
        return None, None
    if title_option is True or title_option == "auto":
        title = (
            "Welch Power Spectral Density"
            if method == "welch"
            else "FFT Amplitude Spectrum"
        )
    else:
        title = str(title_option)

    subtitle_option = options.get("subtitle", "auto")
    if subtitle_option in (False, None, "", True, "auto"):
        subtitle = None
    else:
        subtitle = str(subtitle_option)
    return title, subtitle


def _draw_spectrum_legend(ax, options):
    if not options.get("legend", True):
        return None

    handles, labels = ax.get_legend_handles_labels()
    visible = [
        (handle, label)
        for handle, label in zip(handles, labels)
        if label and not label.startswith("_")
    ]
    if not visible:
        return None

    handles, labels = zip(*visible)
    legend_loc = _normalize_legend_loc(options.get("legend loc", "auto"))
    legend_kwargs = {"loc": legend_loc}
    if options.get("legend outside", False):
        legend_loc, anchor = _resolve_outside_matplotlib_legend_anchor(legend_loc)
        legend_kwargs.update(loc=legend_loc, bbox_to_anchor=anchor)
    return ax.legend(handles, labels, **legend_kwargs)


def _spectrum_line_color(options, index):
    color = options.get("color")
    if color not in (None, ""):
        return color
    colors = list(options.get("colors", []) or [])
    if colors:
        return colors[index % len(colors)]
    return None


def _spectrum_ordinate_values(computation):
    return computation.psd if computation.method == "welch" else computation.amplitude


def _resolve_spectrum_plot_scales(computations, options):
    scale_options = dict(options)
    plot_scale = scale_options.get("plot scale")
    automatic_preset = plot_scale in (None, "", "auto")
    if automatic_preset:
        scale_options["plot scale"] = (
            "semilogy" if computations[0].method == "welch" else "linear"
        )

    xscale, yscale = _resolve_matplotlib_axis_scales(scale_options)
    xscale = "linear" if xscale in (None, "", "auto") else xscale
    yscale = "linear" if yscale in (None, "", "auto") else yscale

    retained_dc = any(
        np.any(np.asarray(computation.frequency) == 0)
        for computation in computations
    )
    if xscale == "log" and options.get("include dc", False) and retained_dc:
        raise ValueError(
            "A logarithmic x-axis cannot display the retained zero-frequency DC "
            "bin. Set include dc=False or use a non-log x scale."
        )

    has_positive_ordinate = any(
        np.any(np.isfinite(values) & (values > 0))
        for values in map(_spectrum_ordinate_values, computations)
    )
    if yscale == "log" and not has_positive_ordinate:
        automatic_y = automatic_preset and options.get("yscale") in (None, "", "auto")
        if automatic_y:
            yscale = "linear"
        else:
            raise ValueError(
                "A logarithmic y-axis requires at least one finite positive spectrum "
                "ordinate. Use a linear or symlog y scale for nonpositive data."
            )

    return xscale, yscale


def _resolve_spectrum_diagnostic_traces(result, computations, options):
    if not options.get("plot all", False):
        return None

    diagnostics = getattr(result, "diagnostics", {})
    traces = diagnostics.get("traces") if hasattr(diagnostics, "get") else None
    if traces is None:
        raise ValueError(
            "Spectrum time-domain diagnostics require resolved traces in "
            "result.diagnostics['traces']."
        )
    traces = list(traces)
    if len(traces) != len(computations):
        raise ValueError(
            "Spectrum time-domain diagnostics require one resolved trace per "
            "spectrum computation."
        )
    return traces


def _attach_spectrum_figures(result, computations, options):
    """Attach spectrum and optional time-domain figures to an AnalysisResult."""
    if isinstance(options, SpectrumOptions):
        options = options.to_options_dict()
    else:
        options = dict(options)

    if not options.get("plot", True):
        result.figure = None
        result.axes = None
        result.figures = []
        return result

    computations = list(computations)
    if not computations:
        raise ValueError("At least one spectrum computation is required to plot a spectrum.")
    method = computations[0].method
    if any(computation.method != method for computation in computations):
        raise ValueError("All spectrum computations in one figure must use the same method.")
    diagnostic_traces = _resolve_spectrum_diagnostic_traces(
        result,
        computations,
        options,
    )
    xscale, yscale = _resolve_spectrum_plot_scales(computations, options)

    existing_figure_numbers = set(plt.get_fignums())
    try:
        figure, axis = plt.subplots(constrained_layout=True)
        line_colors = []
        for index, computation in enumerate(computations):
            values = _spectrum_ordinate_values(computation)
            plot_kwargs = {"label": _spectrum_trace_label(computation)}
            color = _spectrum_line_color(options, index)
            if color is not None:
                plot_kwargs["color"] = color
            line, = axis.plot(computation.frequency, values, **plot_kwargs)
            line_colors.append(line.get_color())

        axis.set_xlabel(options.get("xlabel") or "Frequency (Hz)")
        axis.set_ylabel(options.get("ylabel") or _spectrum_axis_ylabel(computations[0]))
        title, subtitle = _spectrum_plot_titles(method, options)
        _apply_plot_titles(
            figure,
            axis,
            title,
            subtitle,
            _resolve_title_fontsize(title),
            _resolve_subtitle_fontsize(subtitle),
        )
        _apply_ecat_axis_style(axis, options)

        _apply_matplotlib_axis_scales(axis, {"xscale": xscale, "yscale": yscale})
        _draw_spectrum_legend(axis, options)

        figures = [figure]
        if diagnostic_traces is not None:
            diagnostic_figure, diagnostic_axis = plt.subplots(constrained_layout=True)
            for index, trace in enumerate(diagnostic_traces):
                color = line_colors[index]
                label = _spectrum_trace_label(trace)
                if trace.resampled:
                    diagnostic_axis.plot(
                        trace.original_time,
                        trace.original_signal,
                        linestyle="None",
                        marker="o",
                        markersize=4,
                        alpha=0.3,
                        color=color,
                        label=f"{label} (original samples)",
                    )
                    label = f"{label} (resolved)"
                diagnostic_axis.plot(
                    trace.time,
                    trace.signal,
                    color=color,
                    label=label,
                )

            diagnostic_axis.set_xlabel("Time (s)")
            diagnostic_axis.set_ylabel(_time_domain_ylabel(diagnostic_traces[0]))
            diagnostic_title = "Resolved Time-Domain Signal"
            _apply_plot_titles(
                diagnostic_figure,
                diagnostic_axis,
                diagnostic_title,
                None,
                _resolve_title_fontsize(diagnostic_title),
                _resolve_subtitle_fontsize(None),
            )
            _apply_ecat_axis_style(diagnostic_axis, options)
            _draw_spectrum_legend(diagnostic_axis, options)
            figures.append(diagnostic_figure)

        result.figure = figure
        result.axes = axis
        result.figures = figures
    except Exception:
        new_figure_numbers = set(plt.get_fignums()) - existing_figure_numbers
        for figure_number in new_figure_numbers:
            plt.close(figure_number)
        raise

    return result


def spectrum_analysis(obj, options=None):
    """Assemble a standard Welch or FFT spectrum analysis result."""
    typed_options = SpectrumOptions.from_options(options)
    typed_options.validate()
    resolved = typed_options.to_options_dict()
    traces, warning_messages = _resolve_spectrum_traces(obj, resolved)
    computations = [
        _compute_trace(trace, resolved, warning_messages) for trace in traces
    ]
    table = _spectrum_table(computations, resolved["method"])
    summary = _spectrum_summary(computations, traces, resolved)
    diagnostics = _spectrum_diagnostics(computations, traces, resolved)
    unique_warnings = list(dict.fromkeys(warning_messages))
    result = AnalysisResult(
        {"data": table, "summary": summary},
        table=table,
        raw_table=table,
        summary=summary,
        diagnostics=diagnostics,
        warnings=unique_warnings,
        analysis="spectrum",
    )
    for message in unique_warnings:
        warnings.warn(message, UserWarning, stacklevel=3)
    if resolved["plot"]:
        _attach_spectrum_figures(result, computations, resolved)
    if resolved["print"]:
        _print_spectrum_result(result, resolved)
    return result


__all__ = [
    "spectrum_analysis",
    "_SpectrumTrace",
    "_SpectrumComputation",
    "_resolve_spectrum_traces",
    "_compute_trace",
    "_spectrum_table",
    "_spectrum_units",
    "_attach_spectrum_figures",
]
