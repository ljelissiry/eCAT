"""Preprocessing helpers for eCAT electrochemistry objects."""

from __future__ import annotations

from copy import deepcopy

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d, median_filter
from scipy.signal import butter, savgol_filter, sosfiltfilt

from .options import CVFilterOptions


_FILTER_OPTION_KEYS = {
    "method",
    "column",
    "window",
    "polyorder",
    "sigma",
    "size",
    "cutoff",
    "order",
    "inplace",
}


def _resolve_column(data, requested):
    if requested in data.columns:
        return requested
    lookup = {str(column).strip().lower(): column for column in data.columns}
    resolved = lookup.get(str(requested).strip().lower())
    if resolved is None:
        available = ", ".join(str(column) for column in data.columns)
        raise ValueError(
            f"cv.filter could not find column '{requested}'. Available columns: {available}"
        )
    return resolved


def _odd_window(value, length, *, polyorder=None):
    if length < 3:
        raise ValueError("cv.filter requires at least 3 data points.")
    if value == "auto":
        window = min(11, length if length % 2 else length - 1)
        if polyorder is not None:
            window = max(window, int(polyorder) + 2)
    else:
        window = int(value)
    if window % 2 == 0:
        window += 1
    if window > length:
        window = length if length % 2 else length - 1
    if polyorder is not None and window <= int(polyorder):
        raise ValueError("'window' must be greater than 'polyorder' for Savitzky-Golay filtering.")
    if window < 3:
        raise ValueError("The resolved filter window must contain at least 3 points.")
    return window


def _moving_average(values, window):
    window = int(window)
    if window <= 0:
        raise ValueError("'window' must be a positive integer.")
    if window > len(values):
        raise ValueError("'window' cannot exceed the number of data points.")
    left = (window - 1) // 2
    right = window // 2
    padded = np.pad(values, (left, right), mode="edge")
    return np.convolve(padded, np.ones(window) / window, mode="valid")


def _filtered_values(values, options):
    method = options["method"]
    metadata = {"method": method, "column": options["column"]}

    if method == "savgol":
        window = _odd_window(options["window"], len(values), polyorder=options["polyorder"])
        polyorder = int(options["polyorder"])
        filtered = savgol_filter(values, window_length=window, polyorder=polyorder)
        metadata.update({"window": window, "polyorder": polyorder})
    elif method == "gaussian":
        sigma = float(options["sigma"])
        filtered = gaussian_filter1d(values, sigma=sigma)
        metadata["sigma"] = sigma
    elif method == "median":
        size = int(options["size"])
        filtered = median_filter(values, size=size, mode="nearest")
        metadata["size"] = size
    elif method == "butterworth":
        cutoff = float(options["cutoff"])
        order = int(options["order"])
        sos = butter(order, cutoff, btype="lowpass", output="sos")
        try:
            filtered = sosfiltfilt(sos, values)
        except ValueError as exc:
            raise ValueError(
                "The trace is too short for the requested Butterworth filter. "
                "Reduce 'order' or use a longer trace."
            ) from exc
        metadata.update({"cutoff": cutoff, "order": order})
    else:
        window = _odd_window(options["window"], len(values))
        filtered = _moving_average(values, window)
        metadata["window"] = window
    return np.asarray(filtered, dtype=float), metadata


def _filter_summary(metadata):
    method = metadata.get("method", "filter")
    details = [
        f"{key}={value}"
        for key, value in metadata.items()
        if key not in {"method", "column"}
    ]
    return f"{method} ({', '.join(details)})" if details else str(method)


def _print_filter(metadata, options):
    from .plotting import display_object_table

    detail_keys = [key for key in metadata if key not in {"method", "column"}]
    rows = pd.DataFrame(
        {
            "Setting": ["Method", "Column", *[key.title() for key in detail_keys]],
            "Value": [metadata["method"], metadata["column"], *[metadata[key] for key in detail_keys]],
        }
    )
    if options.get("pretty print", True):
        display_object_table(rows, options, title="CV Filter", plain_title=False)
    else:
        print(rows.to_string(index=False))


def _plot_filter(raw, filtered, options):
    from .plotting import multiplot

    plot_options = {
        key: value
        for key, value in options.items()
        if key not in _FILTER_OPTION_KEYS
    }
    plot_options.update({"print": False, "labels": ["Raw", "Filtered"], "legend": True})
    return multiplot([raw, filtered], plot_options)


def filter_cv(cv_obj, options=None):
    """Filter one CV data column with a reproducible SciPy-backed method."""
    resolved = CVFilterOptions.from_options(options).to_options_dict()
    column = _resolve_column(cv_obj.data, resolved["column"])
    values = pd.to_numeric(cv_obj.data[column], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError(f"cv.filter requires finite numeric values in column '{column}'.")
    resolved["column"] = str(column)
    filtered_values, metadata = _filtered_values(values, resolved)

    raw = deepcopy(cv_obj)
    target = cv_obj if resolved["inplace"] else deepcopy(cv_obj)
    target.data[column] = filtered_values
    target.units = dict(getattr(cv_obj, "units", {}) or {})
    target.filter_metadata = metadata
    history = list(getattr(target, "processing_history", []) or [])
    history.append({"operation": "filter", **metadata})
    target.processing_history = history
    target._sync_data_unit_attrs()
    target._refresh_parse_result()

    if resolved.get("print", True):
        _print_filter(metadata, resolved)
    if resolved.get("plot", False):
        _plot_filter(raw, target, resolved)
    return target


__all__ = ["filter_cv"]
