"""Matplotlib rendering helpers for browser display."""

from __future__ import annotations

import base64
import math
from io import BytesIO
from pathlib import Path
import tempfile

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt

import ecat as e


def figure_to_data_uri(fig, format: str = "png", dpi: int = 150) -> str:
    buffer = BytesIO()
    fig.savefig(buffer, format=format, bbox_inches="tight", dpi=dpi)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    mime = "application/pdf" if format == "pdf" else "image/svg+xml" if format == "svg" else f"image/{format}"
    return f"data:{mime};base64,{encoded}"


def bytes_to_data_uri(data: bytes, mime: str) -> str:
    encoded = base64.b64encode(data).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def _pop_render_options(options):
    image_format = str(options.pop("_format", "png") or "png").lower()
    dpi = int(options.pop("_dpi", 150) or 150)
    return image_format, dpi


def _pop_axis_label_options(options):
    return options.pop("x label", None), options.pop("y label", None)


def _apply_axis_labels(ax, x_label, y_label):
    if x_label is False:
        ax.set_xlabel("")
    elif x_label not in (None, ""):
        ax.set_xlabel(str(x_label))
    if y_label is False:
        ax.set_ylabel("")
    elif y_label not in (None, ""):
        ax.set_ylabel(str(y_label))


def _apply_plot_style_option(options):
    plot_style = options.pop("plot style", None)
    if plot_style in (None, ""):
        return None
    style = str(plot_style).strip().lower()
    if style in {"line", "scatter", "line+markers", "line markers", "line-and-markers"}:
        return None
    e.plotting_style(style)
    return style


def _numeric_potential_values(obj) -> list[float]:
    x = getattr(obj, "x", None)
    if not callable(x):
        return []
    try:
        values = x()
    except TypeError:
        values = x({})

    numeric = []
    for value in values:
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if not math.isnan(number):
            numeric.append(number)
    return numeric


def _fill_potential_window(objects, potential_window):
    if not isinstance(potential_window, (list, tuple)) or len(potential_window) != 2:
        return potential_window

    lower, upper = potential_window
    if lower is not None and upper is not None:
        return [lower, upper]

    values = []
    for obj in objects or []:
        values.extend(_numeric_potential_values(obj))
    if not values:
        return potential_window

    return [
        min(values) if lower is None else lower,
        max(values) if upper is None else upper,
    ]


def prepare_plot_objects_and_options(objects, options=None):
    options = dict(options or {})
    _apply_plot_style_option(options)
    objects = list(objects or [])
    potential_window = options.pop("potential window", None)
    trim_mode = options.pop("trim mode", "expand")
    if potential_window is None:
        return objects, options

    potential_window = _fill_potential_window(objects, potential_window)
    if any(value is None for value in potential_window):
        return objects, options
    prepared = []
    trim_options = {"potential window": potential_window, "mode": trim_mode or "expand"}
    for obj in objects:
        trim = getattr(obj, "trim", None)
        if callable(trim):
            prepared.append(trim(trim_options))
        else:
            prepared.append(obj)
    return prepared, options


def render_object_plot(obj, options=None) -> str:
    options = dict(options or {"legend": False, "title": True})
    image_format, dpi = _pop_render_options(options)
    x_label, y_label = _pop_axis_label_options(options)
    objects, options = prepare_plot_objects_and_options([obj], options)
    obj = objects[0] if objects else obj
    grid = bool(options.pop("grid", False))
    ax = obj.plot(options)
    try:
        ax.grid(grid)
        _apply_axis_labels(ax, x_label, y_label)
        return figure_to_data_uri(ax.figure, format=image_format, dpi=dpi)
    finally:
        plt.close(ax.figure)


def render_multiplot(objects, options=None) -> str:
    options = dict(options or {"legend": "auto", "title": False})
    image_format, dpi = _pop_render_options(options)
    x_label, y_label = _pop_axis_label_options(options)
    objects, options = prepare_plot_objects_and_options(objects, options)
    grid = bool(options.pop("grid", False))
    offset = options.get("offset", 0)
    if offset not in (None, "", 0) and not options.get("scale bar"):
        options["scale bar"] = {
            "length": abs(float(offset)),
            "loc": "upper left",
            "remove y ticks": True,
        }
    ax = e.multiplot(objects, options)
    try:
        ax.grid(grid)
        _apply_axis_labels(ax, x_label, y_label)
        scale_bar = options.get("scale bar") or {}
        if offset not in (None, "", 0) and (not isinstance(scale_bar, dict) or scale_bar.get("remove y ticks", True)):
            ax.tick_params(axis="y", which="both", labelleft=False)
        return figure_to_data_uri(ax.figure, format=image_format, dpi=dpi)
    finally:
        plt.close(ax.figure)


def render_animation(objects, options=None) -> str:
    options = dict(options or {})
    export_format = str(options.pop("_format", "html") or "html").lower()
    options.pop("_dpi", None)
    options.pop("_animate", None)
    objects, options = prepare_plot_objects_and_options(objects, options)
    animation = e.animate(objects, options)
    try:
        if export_format == "html":
            html_text = animation.to_html({"progress": True})
            return bytes_to_data_uri(html_text.encode("utf-8"), "text/html")
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / f"ecat_animation.{export_format}"
            animation.save(path, format=export_format, options={"progress": True})
            mime = "image/gif" if export_format == "gif" else "video/mp4"
            return bytes_to_data_uri(path.read_bytes(), mime)
    finally:
        plt.close(animation.figure)


def _model_parameter_value(rows, name, default=None):
    for row in rows or []:
        if row.get("name") != name:
            continue
        value = row.get("initial")
        if value in (None, ""):
            return default
        try:
            return float(value)
        except (TypeError, ValueError):
            return value
    return default


def _setting_value(settings, name, default=None):
    value = dict(settings or {}).get(name, default)
    if value in (None, ""):
        return default
    return value


def render_model_program_plot(parameter_rows=None, options=None, program_settings=None) -> str:
    options = dict(options or {})
    image_format, dpi = _pop_render_options(options)
    input_obj = e.simulation.cv_program(
        _setting_value(program_settings, "Ei", _model_parameter_value(parameter_rows, "E_initial", 0.0)),
        E_low=_setting_value(program_settings, "E_low", _model_parameter_value(parameter_rows, "E_vertex_1", -1.0)),
        E_high=_setting_value(program_settings, "E_high", _model_parameter_value(parameter_rows, "E_vertex_2", 1.0)),
        Ef=_setting_value(program_settings, "Ef", None),
        scan_rate=_setting_value(program_settings, "scan_rate", _model_parameter_value(parameter_rows, "scan_rate", 0.1)),
        segments=int(_setting_value(program_settings, "segments", _model_parameter_value(parameter_rows, "segments", 2)) or 2),
        points_per_segment=int(_setting_value(program_settings, "points_per_segment", 300) or 300),
        quiet_time=float(_setting_value(program_settings, "quiet_time", 0) or 0),
    )
    ax = input_obj.plot(
        {
            "title": "Simulation Input",
            "plot": False,
            "plot quiet time": bool(_setting_value(program_settings, "plot quiet time", False)),
        }
    )
    try:
        return figure_to_data_uri(ax.figure, format=image_format, dpi=dpi)
    finally:
        plt.close(ax.figure)
