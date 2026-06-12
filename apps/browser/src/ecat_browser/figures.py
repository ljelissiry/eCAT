"""Matplotlib rendering helpers for browser display."""

from __future__ import annotations

import base64
import math
from io import BytesIO

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt

import ecat as e


def figure_to_data_uri(fig, format: str = "png", dpi: int = 150) -> str:
    buffer = BytesIO()
    fig.savefig(buffer, format=format, bbox_inches="tight", dpi=dpi)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    mime = "application/pdf" if format == "pdf" else f"image/{format}"
    return f"data:{mime};base64,{encoded}"


def _pop_render_options(options):
    image_format = str(options.pop("_format", "png") or "png").lower()
    dpi = int(options.pop("_dpi", 150) or 150)
    return image_format, dpi


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
    objects = list(objects or [])
    potential_window = options.pop("potential window", None)
    if potential_window is None:
        return objects, options

    potential_window = _fill_potential_window(objects, potential_window)
    if any(value is None for value in potential_window):
        return objects, options
    prepared = []
    trim_options = {"potential window": potential_window, "mode": "expand"}
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
    objects, options = prepare_plot_objects_and_options([obj], options)
    obj = objects[0] if objects else obj
    grid = bool(options.pop("grid", False))
    ax = obj.plot(options)
    try:
        ax.grid(grid)
        return figure_to_data_uri(ax.figure, format=image_format, dpi=dpi)
    finally:
        plt.close(ax.figure)


def render_multiplot(objects, options=None) -> str:
    options = dict(options or {"legend": "auto", "title": False})
    image_format, dpi = _pop_render_options(options)
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
        scale_bar = options.get("scale bar") or {}
        if offset not in (None, "", 0) and (not isinstance(scale_bar, dict) or scale_bar.get("remove y ticks", True)):
            ax.tick_params(axis="y", which="both", labelleft=False)
        return figure_to_data_uri(ax.figure, format=image_format, dpi=dpi)
    finally:
        plt.close(ax.figure)
