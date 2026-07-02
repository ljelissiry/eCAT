"""Public animation entry points and result wrapper."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import html
import math
from pathlib import Path
import tempfile

from matplotlib import animation as mpl_animation
from matplotlib import pyplot as plt
from matplotlib.transforms import Bbox
import numpy as np

from ._progress import NotebookProgressDisplay, progress_enabled
from .options import MultiplotOptions, PlotOptions, _canonical_option_key, _drop_legacy_alias_mirrors

_PLOT_OPTION_KEYS = {field_name.replace("_", " ") for field_name in MultiplotOptions.__dataclass_fields__}
_ANIMATION_SAVE_OPTION_KEYS = {
    "fps": "fps",
    "dpi": "dpi",
    "codec": "codec",
    "bitrate": "bitrate",
    "extra args": "extra_args",
    "metadata": "metadata",
    "savefig kwargs": "savefig_kwargs",
    "extra anim": "extra_anim",
}


@dataclass
class AnimationResult:
    """Notebook-facing wrapper around an eCAT animation."""

    animation: object
    figure: object
    axes: object
    summary: dict

    def show(self, options=None):
        self.display(options=options)
        return None

    def display(self, options=None):
        try:
            from IPython.display import HTML, display
        except Exception:
            plt.close(self.figure)
            return None
        plt.close(self.figure)
        display(HTML(_autoplay_html(self.to_html(options=options))))
        return None

    def to_html(self, options=None):
        render_options = _merged_render_options(self.summary.get("resolved options", {}), options)
        fps = max(float(render_options.get("fps", self.summary.get("fps", 20))), 1.0)
        default_mode = "loop" if bool(render_options.get("loop", self.summary.get("loop", True))) else "once"
        progress = _AnimationRenderProgress(
            total=self.summary.get("frame count"),
            label="Rendering animation",
            progress=render_options.get("progress", True),
        )
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                target = Path(temp_dir) / "animation.html"
                writer = mpl_animation.HTMLWriter(
                    fps=fps,
                    embed_frames=True,
                    default_mode=default_mode,
                )
                self.animation.save(
                    str(target),
                    writer=writer,
                    progress_callback=progress.callback,
                    **_save_kwargs(render_options, include_writer_args=False),
                )
                return target.read_text(encoding="utf-8")
        finally:
            progress.close()

    def save(self, path, format=None, options=None):
        path = Path(path)
        render_options = _merged_render_options(self.summary.get("resolved options", {}), options)
        export_format = str(format or path.suffix.lstrip(".")).strip().lower()
        if export_format == "html":
            path.write_text(self.to_html(options=options), encoding="utf-8")
            return path
        writer = _writer_for_path(path, format=export_format)
        progress = _AnimationRenderProgress(
            total=self.summary.get("frame count"),
            label="Saving animation",
            progress=render_options.get("progress", True),
        )
        try:
            self.animation.save(
                str(path),
                writer=writer,
                progress_callback=progress.callback,
                **_save_kwargs(render_options),
            )
        finally:
            progress.close()
        return path


def _normalize_animation_options(options=None):
    if options is None:
        raw_options = {}
    elif hasattr(options, "to_legacy_dict"):
        raw_options = options.to_legacy_dict()
    elif isinstance(options, Mapping):
        raw_options = dict(options)
    else:
        raw_options = dict(options)
    if any(str(key).startswith("_") for key in raw_options):
        raw_options = _drop_legacy_alias_mirrors(raw_options)
    legacy_options = MultiplotOptions.from_options(raw_options).to_legacy_dict()
    if hasattr(options, "to_legacy_dict"):
        return legacy_options
    return {
        _canonical_option_key(key).replace("_", " "): legacy_options[_canonical_option_key(key).replace("_", " ")]
        for key in raw_options
    }


def _resolve_animation_options(options):
    return MultiplotOptions.from_options(options).to_legacy_dict()


def _normalize_animation_input(obj_or_list):
    if isinstance(obj_or_list, list):
        if not obj_or_list:
            raise ValueError("animate() requires at least one object.")
        return list(obj_or_list), True
    return [obj_or_list], False


def _scan_rates(objects):
    values = [getattr(obj, "scan_rate", None) for obj in objects]
    return [value for value in values if value is not None]


def _column_label(column):
    if isinstance(column, tuple):
        for part in reversed(column):
            if part not in (None, ""):
                return str(part)
        return ""
    return str(column)


def _has_physical_timing(obj):
    scan_rate = getattr(obj, "scan_rate", None)
    try:
        if scan_rate is not None and math.isfinite(float(scan_rate)) and float(scan_rate) > 0:
            return True
    except (TypeError, ValueError):
        pass

    data = getattr(obj, "data", None)
    columns = getattr(data, "columns", ())
    for column in columns:
        label = _column_label(column).strip().lower()
        if label in {"t", "time", "duration"} or label.startswith("time "):
            return True
        if label.startswith("time(") or label.startswith("time/"):
            return True
    return False


def _resolve_trace_mode_and_schedule(objects, options, is_list):
    trace_mode = options.get("trace mode", "auto")
    schedule = options.get("schedule", "auto")

    if not is_list:
        if trace_mode == "auto":
            trace_mode = "draw"
        return trace_mode, None

    rates = _scan_rates(objects)
    mixed_rates = len({float(rate) for rate in rates}) > 1 if rates else False

    if trace_mode == "auto":
        trace_mode = "draw" if mixed_rates else "instant"
    if schedule == "auto":
        schedule = "simultaneous" if mixed_rates else "staggered"
    return trace_mode, schedule


def _resolve_timing_mode(objects, options):
    timing_mode = options.get("timing mode", "auto")
    if timing_mode != "auto":
        return timing_mode
    if all(_has_physical_timing(obj) for obj in objects):
        return "physical"
    return "normalized"


def _static_animation_plot_options(options, is_list, provided_options=None):
    option_class = MultiplotOptions if is_list else PlotOptions
    valid_keys = {field_name.replace("_", " ") for field_name in option_class.__dataclass_fields__}
    plot_options = {
        key: value
        for key, value in dict(options).items()
        if key in valid_keys or str(key).startswith("_")
    }
    provided_keys = set(dict(provided_options or {}).keys())
    if not is_list and "legend" not in provided_keys:
        plot_options["legend"] = "auto"
    plot_options["animate"] = False
    plot_options["new plot"] = False if is_list else True
    return plot_options


def _build_static_figure_axes(objects, options, is_list, provided_options=None):
    plot_options = _static_animation_plot_options(options, is_list, provided_options=provided_options)
    if is_list:
        from .plotting import multiplot

        axes = multiplot(objects, plot_options)
    else:
        axes = objects[0].plot(plot_options)
    figure = axes.figure
    _adjust_figure_layout_for_animation(figure)
    return figure, axes


def _time_column_values(obj):
    data = getattr(obj, "data", None)
    columns = getattr(data, "columns", ())
    for column in columns:
        label = _column_label(column).strip().lower()
        if label in {"t", "time", "duration"} or label.startswith("time "):
            return np.asarray(data[column], dtype=float)
        if label.startswith("time(") or label.startswith("time/"):
            return np.asarray(data[column], dtype=float)
    return None


def _physical_object_duration(obj, include_quiet_time):
    duration = 1.0
    time_values = _time_column_values(obj)
    if time_values is not None and len(time_values) > 1:
        duration = float(np.nanmax(time_values) - np.nanmin(time_values))
    else:
        scan_rate = getattr(obj, "scan_rate", None)
        try:
            scan_rate = float(scan_rate)
        except (TypeError, ValueError):
            scan_rate = None
        if scan_rate is not None and math.isfinite(scan_rate) and scan_rate > 0:
            x_values = np.asarray(obj.x({"one column": True}), dtype=float)
            if len(x_values) > 1:
                duration = float(np.nansum(np.abs(np.diff(x_values))) / scan_rate)
    if include_quiet_time:
        quiet_time = getattr(obj, "quiet_time", None)
        try:
            quiet_time = float(quiet_time)
        except (TypeError, ValueError):
            quiet_time = 0.0
        if math.isfinite(quiet_time) and quiet_time > 0:
            duration += quiet_time
    return max(duration, 1e-9)


def _trace_durations(objects, options, timing_mode):
    include_quiet_time = options.get("include quiet time", False)
    if timing_mode == "normalized":
        duration = max(float(options.get("normalized duration", 2.0)), 1e-9)
        return [duration] * len(objects)

    speedup = max(float(options.get("speedup", 1.0)), 1e-9)
    return [
        max(_physical_object_duration(obj, include_quiet_time) / speedup, 1e-9)
        for obj in objects
    ]


def _line_groups(axes, objects, is_list):
    lines = list(axes.lines)
    if not lines:
        raise ValueError("animate() requires plotted line artists.")
    if not is_list:
        return [lines]
    return [[line] for line in lines[: len(objects)]]


def _animation_stride(options):
    try:
        stride = int(dict(options or {}).get("stride", 1))
    except (TypeError, ValueError):
        stride = 1
    return max(stride, 1)


def _stride_values(values, stride):
    if stride <= 1 or len(values) <= 2:
        return values
    thinned = values[::stride]
    if len(thinned) == 0 or thinned[-1] != values[-1]:
        thinned = np.concatenate([thinned, values[-1:]])
    return thinned


def _line_payloads(groups, options=None):
    stride = _animation_stride(options)
    payloads = []
    for group in groups:
        line_payload = []
        for line in group:
            x = np.asarray(line.get_xdata(), dtype=float)
            y = np.asarray(line.get_ydata(), dtype=float)
            x = _stride_values(x, stride)
            y = _stride_values(y, stride)
            line.set_data(x, y)
            line_payload.append((line, x, y))
        payloads.append(line_payload)
    return payloads


def _timeline(objects, options, durations, schedule, is_list):
    if not is_list:
        starts = [0.0]
    elif schedule == "simultaneous":
        starts = [0.0] * len(objects)
    elif schedule == "sequential":
        starts = []
        current = 0.0
        for duration in durations:
            starts.append(current)
            current += duration
    else:
        stagger_time = float(options.get("stagger time", 0.5))
        starts = [i * stagger_time for i in range(len(objects))]
    end_hold = float(options.get("end hold", 2))
    end_time = max(start + duration for start, duration in zip(starts, durations)) + max(end_hold, 0.0)
    return starts, end_time


def _frame_times(total_duration, fps):
    fps = max(float(fps), 1.0)
    frame_count = max(2, int(math.ceil(total_duration * fps)) + 1)
    return np.linspace(0.0, total_duration, frame_count)


def _writer_for_path(path, format=None):
    suffix = str(format or Path(path).suffix.lstrip(".")).strip().lower()
    if suffix == "html":
        return None
    if suffix == "gif":
        return "pillow"
    if suffix == "mp4":
        return "ffmpeg"
    raise ValueError("Animation save format must be HTML, GIF, or MP4.")


def _save_kwargs(options, *, include_writer_args=True):
    savefig_kwargs = dict(dict(options or {}).get("savefig kwargs") or {})
    savefig_kwargs.pop("bbox_inches", None)
    writer_arg_names = {"fps", "codec", "bitrate", "extra_args", "metadata"}
    kwargs = {
        argument_name: dict(options or {})[option_name]
        for option_name, argument_name in _ANIMATION_SAVE_OPTION_KEYS.items()
        if option_name in dict(options or {}) and (include_writer_args or argument_name not in writer_arg_names)
    }
    if savefig_kwargs:
        kwargs["savefig_kwargs"] = savefig_kwargs
    return kwargs


def _figure_content_bbox_in_figure_coords(fig):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    bboxes = []

    for ax in fig.axes:
        try:
            bbox = ax.get_tightbbox(renderer)
        except Exception:
            bbox = None
        if bbox is not None and np.isfinite(bbox.bounds).all():
            bboxes.append(bbox)

    for text in getattr(fig, "texts", []):
        try:
            bbox = text.get_window_extent(renderer)
        except Exception:
            bbox = None
        if bbox is not None and np.isfinite(bbox.bounds).all():
            bboxes.append(bbox)

    if not bboxes:
        return None

    union = Bbox.union(bboxes)
    return union.transformed(fig.transFigure.inverted())


def _adjust_figure_layout_for_animation(fig, *, pad=0.01, max_passes=3):
    if not getattr(fig, "axes", None):
        return

    min_span = 0.2
    for _ in range(max_passes):
        bbox = _figure_content_bbox_in_figure_coords(fig)
        if bbox is None:
            return

        left_over = max(0.0, pad - float(bbox.x0))
        right_over = max(0.0, float(bbox.x1) - (1.0 - pad))
        bottom_over = max(0.0, pad - float(bbox.y0))
        top_over = max(0.0, float(bbox.y1) - (1.0 - pad))

        if max(left_over, right_over, bottom_over, top_over) <= 1e-3:
            break

        subplotpars = fig.subplotpars
        left = min(max(float(subplotpars.left) + left_over, 0.0), 1.0 - min_span)
        right = max(min(float(subplotpars.right) - right_over, 1.0), min(1.0, left + min_span))
        bottom = min(max(float(subplotpars.bottom) + bottom_over, 0.0), 1.0 - min_span)
        top = max(min(float(subplotpars.top) - top_over, 1.0), min(1.0, bottom + min_span))

        previous = (float(subplotpars.left), float(subplotpars.right), float(subplotpars.bottom), float(subplotpars.top))
        updated = (left, right, bottom, top)
        if all(abs(a - b) <= 1e-4 for a, b in zip(previous, updated)):
            break

        fig.subplots_adjust(left=left, right=right, bottom=bottom, top=top)


def _set_line_progress(group_payload, progress, trace_mode):
    for line, x_values, y_values in group_payload:
        if len(x_values) == 0:
            line.set_data([], [])
            continue
        if trace_mode == "instant":
            if progress <= 0:
                line.set_data([], [])
            else:
                line.set_data(x_values, y_values)
            continue
        if progress <= 0:
            line.set_data([], [])
            continue
        if progress >= 1:
            line.set_data(x_values, y_values)
            continue
        points = max(1, min(len(x_values), int(math.ceil(progress * len(x_values)))))
        line.set_data(x_values[:points], y_values[:points])


def _build_animation(figure, axes, objects, options, trace_mode, schedule, durations, frame_times, is_list):
    groups = _line_groups(axes, objects, is_list)
    payloads = _line_payloads(groups, options)
    starts, _total_duration = _timeline(objects, options, durations, schedule, is_list)

    def update(frame_time):
        artists = []
        for payload, start, duration in zip(payloads, starts, durations):
            progress = (float(frame_time) - start) / duration
            _set_line_progress(payload, progress, trace_mode)
            artists.extend(line for line, _x, _y in payload)
        return artists

    animation = mpl_animation.FuncAnimation(
        figure,
        update,
        frames=frame_times,
        interval=1000.0 / max(float(options.get("fps", 20)), 1.0),
        repeat=bool(options.get("loop", True)),
        blit=False,
        cache_frame_data=False,
    )
    animation._draw_was_started = True
    return animation


def _timeline_summary(objects, options, is_list, trace_mode, schedule, timing_mode, durations, frame_times):
    starts, total_duration = _timeline(objects, options, durations, schedule, is_list)
    summary = {
        "count": len(objects),
        "trace mode": trace_mode,
        "schedule": schedule,
        "timing mode": timing_mode,
        "normalized duration": options.get("normalized duration", 2.0),
        "speedup": options.get("speedup", 1.0),
        "fps": options.get("fps", 20),
        "stride": options.get("stride", 1),
        "loop": options.get("loop", True),
        "end hold": options.get("end hold", 2),
        "stagger time": options.get("stagger time", 0.5) if is_list else None,
        "include quiet time": options.get("include quiet time", False),
        "estimated animation time": float(total_duration),
        "frame count": int(len(frame_times)),
        "trace durations": list(durations),
        "trace starts": list(starts),
    }
    return summary


def _format_duration(seconds):
    seconds = float(seconds)
    if seconds < 60:
        return f"{seconds:.3g} s"
    minutes = int(seconds // 60)
    remainder = seconds - 60 * minutes
    if remainder <= 0:
        return f"{minutes} min"
    return f"{minutes} min {remainder:.0f} s"


def _setup_table(summary):
    import pandas as pd

    requested = dict(summary.get("options") or {})

    def _resolved_value(option_name, resolved_value, *, empty="single"):
        if option_name not in requested:
            return resolved_value if resolved_value is not None else empty
        requested_value = requested.get(option_name)
        if requested_value == "auto" and resolved_value is not None:
            return f"auto -> {resolved_value}"
        return requested_value if requested_value is not None else empty

    rows = [
        ("Timing Mode", _resolved_value("timing mode", summary.get("timing mode"))),
        ("Trace Mode", _resolved_value("trace mode", summary.get("trace mode"))),
        ("Schedule", _resolved_value("schedule", summary.get("schedule"), empty="single")),
        ("Traces", summary.get("count")),
    ]
    if summary.get("timing mode") == "normalized":
        rows.append(("Normalized Duration", f"{float(summary.get('normalized duration', 2.0)):.3g} s/trace"))
    else:
        rows.append(("Speedup", f"{float(summary.get('speedup', 1.0)):.3g}x"))
    rows.extend(
        [
            ("Animation Time", _format_duration(summary.get("estimated animation time", 0.0))),
            ("FPS", summary.get("fps")),
            ("Frames", summary.get("frame count")),
            ("Loop", "Yes" if bool(summary.get("loop", True)) else "No"),
            ("End Hold", f"{float(summary.get('end hold', 0.0)):.3g} s"),
            ("Include Quiet Time", "Yes" if bool(summary.get("include quiet time", False)) else "No"),
        ]
    )
    if summary.get("schedule") == "staggered":
        rows.append(("Stagger Time", f"{float(summary.get('stagger time', 0.0)):.3g} s"))
    return pd.DataFrame(rows, columns=["Setting", "Value"])


def _display_animation_setup(summary, options):
    frame = _setup_table(summary)
    if options.get("pretty print", True):
        try:
            from IPython.display import Markdown, display

            styled = (
                frame.style
                .hide(axis="index")
                .set_properties(**{"text-align": "left", "white-space": "pre-wrap"})
                .set_table_styles(
                    [
                        {"selector": "th", "props": [("text-align", "left")]},
                        {"selector": "td", "props": [("text-align", "left")]},
                    ]
                )
            )
            display(Markdown("**Animation Setup:**"))
            display(styled)
            return
        except Exception:
            pass
    print("Animation Setup:")
    print(frame.to_string(index=False))


def _progress_enabled(progress):
    return progress_enabled(progress)


class _AnimationRenderProgress:
    def __init__(self, *, total=None, label="Rendering animation", progress=True):
        self.total = None if total is None else max(1, int(total))
        self.label = label
        self.progress = progress
        self.enabled = _progress_enabled(progress)
        self._bar = None
        self._bar_kind = None
        if not self.enabled:
            return
        self._create_bar()

    def _create_bar(self):
        preference = str(self.progress).strip().lower() if isinstance(self.progress, str) else "auto"
        tqdm_imports = [("tqdm.auto", "tqdm"), ("tqdm.notebook", "tqdm"), ("tqdm", "tqdm")]
        if preference in {"terminal", "cli"}:
            tqdm_imports = [("tqdm", "tqdm")]
        for module_name, attr in tqdm_imports:
            try:
                module = __import__(module_name, fromlist=[attr])
                tqdm = getattr(module, attr)
                self._bar = tqdm(total=self.total, desc=self.label, unit="frame", leave=True)
                self._bar_kind = "tqdm"
                return
            except Exception:
                continue
        try:
            self._bar = _NotebookAnimationProgressDisplay(total=self.total, label=self.label)
            self._bar_kind = "notebook"
        except Exception:
            self._bar = None
            self._bar_kind = None

    def callback(self, current_frame, total_frames):
        if self._bar is None:
            return
        count = int(current_frame) + 1
        total = self.total if self.total is not None else total_frames
        if self._bar_kind == "tqdm":
            if total is not None and self._bar.total != total:
                self._bar.total = total
            delta = max(0, count - int(self._bar.n))
            if delta:
                self._bar.update(delta)
        else:
            self._bar.update(count, total=total)

    def close(self):
        if self._bar is None:
            return
        self._bar.close()


class _NotebookAnimationProgressDisplay:
    def __init__(self, *, total=None, label="Rendering animation"):
        self._impl = NotebookProgressDisplay(
            total=total,
            label=label,
            leave=True,
            unit="frames",
            approx_total=False,
        )
        self.total = self._impl.total
        self.label = label

    def update(self, count, *, total=None):
        if total is not None and self.total is not None:
            self._impl.total = max(1, int(total))
            self.total = self._impl.total
        self._impl.update(count, metric=None)

    def close(self):
        self._impl.close()

    @staticmethod
    def _html(count, first, second=None, third=None, *, elapsed=0.0, remaining=None):
        if third is None:
            metric = None
            total = first
            label = second
        else:
            metric = first
            total = second
            label = third
        return NotebookProgressDisplay._html(
            count,
            metric,
            total,
            label,
            elapsed=elapsed,
            remaining=remaining,
            unit="frames",
            approx_total=False,
            metric_label=None,
            indeterminate=(total is None),
        )

    @staticmethod
    def _done_html(count, first, second=None, third=None, *, elapsed=0.0):
        if third is None:
            total = first
            label = second
        else:
            total = second
            label = third
        return NotebookProgressDisplay._done_html(
            count,
            None,
            total,
            label,
            elapsed=elapsed,
            unit="frames",
            approx_total=False,
            metric_label=None,
            indeterminate=(total is None),
        )


def _merged_render_options(base_options, override_options=None):
    merged = dict(base_options or {})
    if override_options is None:
        return merged
    plot_options, save_options = _split_animation_override_options(override_options)
    merged.update(plot_options)
    merged.update(save_options)
    return merged


def _split_animation_override_options(override_options):
    if hasattr(override_options, "to_legacy_dict"):
        return override_options.to_legacy_dict(), {}

    raw_options = dict(override_options or {})
    is_legacy_option_dict = any(str(key).startswith("_") for key in raw_options)
    plot_options = {}
    save_options = {}
    unknown = []
    for key, value in raw_options.items():
        if str(key).startswith("_"):
            continue
        canonical = _canonical_option_key(key).replace("_", " ")
        if canonical in _PLOT_OPTION_KEYS:
            plot_options[key] = value
        elif canonical in _ANIMATION_SAVE_OPTION_KEYS:
            save_options[canonical] = value
        else:
            unknown.append(key)
    if unknown:
        unknown_key = unknown[0]
        raise ValueError(f"Unknown animation render option '{unknown_key}'.")
    if is_legacy_option_dict:
        plot_options = _drop_legacy_alias_mirrors(plot_options)
    normalized_plot_options = _normalize_animation_options(plot_options) if plot_options else {}
    return normalized_plot_options, save_options


def _autoplay_html(html_text):
    import re

    match = re.search(r"([A-Za-z_][A-Za-z0-9_]*) = new Animation\(frames, img_id, slider_id,", html_text)
    if match is None:
        return html_text
    animation_var = match.group(1)
    autoplay_script = (
        "\n<script language=\"javascript\">\n"
        f"setTimeout(function() {{ if (typeof {animation_var} !== 'undefined' && {animation_var}) {{ {animation_var}.play_animation(); }} }}, 0);\n"
        "</script>\n"
    )
    if autoplay_script.strip() in html_text:
        return html_text
    return html_text + autoplay_script


def animate(obj_or_list, options=None):
    """Animate one electrochemistry object or a list of them.

    Parameters
    ----------
    obj_or_list : echem or list[echem]
        One electrochemistry object or a list of objects to animate.
    options : dict or PlotOptions, optional
        Animation timing, rendering, and plotting options. See
        ``e.describe_options("animate")``.

    Returns
    -------
    AnimationResult
        Wrapper containing the Matplotlib animation, figure, axes, and
        resolved timing/render summary.

    Examples
    --------
    >>> result = e.animate(cv_obj, {"trace mode": "draw"})
    >>> result.save("cv.gif")
    """

    objects, is_list = _normalize_animation_input(obj_or_list)
    options = _normalize_animation_options(options)
    resolved_options = _resolve_animation_options(options)
    trace_mode, schedule = _resolve_trace_mode_and_schedule(objects, resolved_options, is_list)
    timing_mode = _resolve_timing_mode(objects, resolved_options)
    durations = _trace_durations(objects, resolved_options, timing_mode)
    starts, total_duration = _timeline(objects, resolved_options, durations, schedule, is_list)
    frame_times = _frame_times(total_duration, resolved_options.get("fps", 20))
    figure, axes = _build_static_figure_axes(
        objects,
        resolved_options,
        is_list,
        provided_options=options,
    )
    animation = _build_animation(
        figure,
        axes,
        objects,
        resolved_options,
        trace_mode,
        schedule,
        durations,
        frame_times,
        is_list,
    )
    summary = {
        "object_type": type(obj_or_list).__name__,
        "options": options,
        "resolved options": dict(resolved_options),
        **_timeline_summary(objects, resolved_options, is_list, trace_mode, schedule, timing_mode, durations, frame_times),
    }
    summary["trace starts"] = starts
    result = AnimationResult(
        animation=animation,
        figure=figure,
        axes=axes,
        summary=summary,
    )
    if resolved_options.get("print", True):
        _display_animation_setup(summary, resolved_options)
    if resolved_options.get("plot", True):
        plt.close(figure)
        result.display()
    return result


__all__ = ["AnimationResult", "animate"]
