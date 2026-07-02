"""Shared plotting helper functions used by eCAT plot routines."""

import re

import matplotlib as mpl
import numpy as np
from matplotlib.font_manager import FontProperties
from matplotlib.ticker import AutoMinorLocator, FixedLocator

from ._plot_style import _active_plot_style_value


def _normalize_scale_bar_options(options):
    spec = options.get("scale bar", False)
    if spec in (False, None):
        return None
    if spec is True:
        spec = {"length": "auto"}
    elif isinstance(spec, (int, float, np.integer, np.floating)):
        spec = {"length": float(spec)}
    elif isinstance(spec, dict):
        spec = dict(spec)
    else:
        raise ValueError("'scale bar' must be False, True, a numeric length, or an options dictionary.")

    if "length" not in spec or spec.get("length") is None:
        raise ValueError("'scale bar' requires a numeric 'length' in displayed y-axis units.")
    if str(spec.get("length")).strip().lower() != "auto":
        try:
            spec["length"] = float(spec["length"])
        except (TypeError, ValueError) as exc:
            raise ValueError("'scale bar' length must be numeric or 'auto'.") from exc
        if spec["length"] <= 0:
            raise ValueError("'scale bar' length must be positive.")

    spec.setdefault("loc", "lower right")
    spec.setdefault("remove y ticks", True)
    return spec


def _nice_scale_bar_length(y_span):
    y_span = abs(float(y_span))
    if not np.isfinite(y_span) or y_span <= 0:
        return 1.0

    lower = 0.20 * y_span
    upper = 0.25 * y_span
    target = 0.225 * y_span
    exponent_min = int(np.floor(np.log10(lower))) - 1
    exponent_max = int(np.ceil(np.log10(upper))) + 1
    candidates = []
    for exponent in range(exponent_min, exponent_max + 1):
        scale = 10.0 ** exponent
        candidates.extend(base * scale for base in (1, 2, 3, 5, 10))

    in_range = [value for value in candidates if lower <= value <= upper]
    if in_range:
        return min(in_range, key=lambda value: (abs(value - target), -value))
    return min(candidates, key=lambda value: abs(value - target))


def _rc_fontsize_points(value, fallback):
    try:
        return float(value)
    except (TypeError, ValueError):
        pass
    try:
        return float(FontProperties(size=value).get_size_in_points())
    except (TypeError, ValueError):
        return fallback


def _default_scale_bar_fontsize():
    style_size = _active_plot_style_value("legend fontsize")
    if style_size is not None:
        return style_size
    font_size = _rc_fontsize_points(mpl.rcParams.get("font.size"), 10)
    return _rc_fontsize_points(mpl.rcParams.get("legend.fontsize"), font_size)


def _scale_bar_fontsize(spec):
    return spec.get("fontsize", spec.get("font size", _default_scale_bar_fontsize()))


def _scale_bar_position(ax, loc, length):
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    xmin, xmax = sorted([x0, x1])
    ymin, ymax = sorted([y0, y1])
    x_span = xmax - xmin or 1.0
    y_span = ymax - ymin or 1.0

    if isinstance(loc, (list, tuple)) and len(loc) == 2:
        return float(loc[0]), float(loc[1])

    loc_key = str(loc).strip().lower().replace("_", " ").replace("-", " ")
    if loc_key not in {"lower right", "lower left", "upper right", "upper left"}:
        raise ValueError(
            "'scale bar' loc must be a two-value (x, y) tuple or one of "
            "'lower right', 'lower left', 'upper right', 'upper left'."
        )

    visual_right = xmin if ax.xaxis_inverted() else xmax
    visual_left = xmax if ax.xaxis_inverted() else xmin
    visual_upper = ymin if ax.yaxis_inverted() else ymax
    visual_lower = ymax if ax.yaxis_inverted() else ymin

    x_pad = 0.08 * x_span
    y_pad = 0.10 * y_span
    if (
        _active_plot_style_value("axis labels") == "inside"
        and loc_key == "lower right"
    ):
        y_pad = 0.22 * y_span
    if "right" in loc_key:
        x = visual_right + x_pad if ax.xaxis_inverted() else visual_right - x_pad
    else:
        x = visual_left - x_pad if ax.xaxis_inverted() else visual_left + x_pad

    half = length / 2.0
    if "upper" in loc_key:
        y = visual_upper + y_pad + half if ax.yaxis_inverted() else visual_upper - y_pad - half
    else:
        y = visual_lower - y_pad - half if ax.yaxis_inverted() else visual_lower + y_pad + half
    return x, y


def _scale_bar_loc_key(loc):
    if isinstance(loc, (list, tuple)) and len(loc) == 2:
        return None
    return str(loc).strip().lower().replace("_", " ").replace("-", " ")


def _data_dx_to_points(ax, x, y, dx):
    p0 = ax.transData.transform((x, y))[0]
    p1 = ax.transData.transform((x + dx, y))[0]
    dpi = ax.figure.dpi or 72.0
    return abs(p1 - p0) * 72.0 / dpi


def _scale_bar_label_layout(ax, loc, x, y, cap_width, label_pad):
    loc_key = _scale_bar_loc_key(loc)
    label_side = "right"
    if loc_key is not None and "right" in loc_key:
        label_side = "left"

    cap_half_points = _data_dx_to_points(ax, x, y, cap_width / 2.0)
    pad_points = _data_dx_to_points(ax, x, y, label_pad)
    offset = max(4.0, cap_half_points + pad_points)

    if label_side == "left":
        return -offset, "right"
    return offset, "left"


def _set_line_collection_segment(collection, segment):
    collection.set_segments([np.asarray(segment, dtype=float)])


def _add_scale_bar(ax, options, unit=None):
    spec = _normalize_scale_bar_options(options)
    if spec is None:
        return None

    y0, y1 = ax.get_ylim()
    length = (
        _nice_scale_bar_length(y1 - y0)
        if str(spec["length"]).strip().lower() == "auto"
        else spec["length"]
    )
    spec["length"] = length
    x, y = _scale_bar_position(ax, spec.get("loc", "lower right"), length)
    cap_width_is_auto = "cap width" not in spec
    label_pad_is_auto = "label pad" not in spec
    x0, x1 = ax.get_xlim()
    cap_width = float(spec.get("cap width", 0.04 * abs(x1 - x0 or 1.0)))
    color = spec.get("color", "k")
    linewidth = spec.get("linewidth", 1.5)
    label_pad = float(spec.get("label pad", 0.02 * abs(x1 - x0 or 1.0)))
    fontsize = _scale_bar_fontsize(spec)

    vertical = ax.vlines(x=x, ymin=y - length / 2, ymax=y + length / 2, color=color, linewidth=linewidth)
    top_cap = ax.hlines(y=y + length / 2, xmin=x - cap_width / 2, xmax=x + cap_width / 2, color=color, linewidth=linewidth)
    bottom_cap = ax.hlines(y=y - length / 2, xmin=x - cap_width / 2, xmax=x + cap_width / 2, color=color, linewidth=linewidth)

    unit = spec.get("unit", unit)
    label = spec.get("label")
    if label is None:
        length_text = f"{length:g}"
        label = f"{length_text} {unit}".strip()
    x_offset_points, default_ha = _scale_bar_label_layout(
        ax,
        spec.get("loc", "lower right"),
        x,
        y,
        cap_width,
        label_pad,
    )
    annotation = ax.annotate(
        label,
        xy=(x, y),
        xycoords="data",
        xytext=(x_offset_points, 0),
        textcoords="offset points",
        color=color,
        fontsize=fontsize,
        ha=spec.get("ha", default_ha),
        va=spec.get("va", "center"),
    )

    if spec.get("remove y ticks", True):
        ax.set_yticks([])

    updating = {"active": False}

    def _update_scale_bar(_event_ax=None):
        if updating["active"]:
            return
        updating["active"] = True
        try:
            new_x, new_y = _scale_bar_position(ax, spec.get("loc", "lower right"), length)
            new_x0, new_x1 = ax.get_xlim()
            new_cap_width = (
                0.04 * abs(new_x1 - new_x0 or 1.0)
                if cap_width_is_auto
                else cap_width
            )
            new_label_pad = (
                0.02 * abs(new_x1 - new_x0 or 1.0)
                if label_pad_is_auto
                else label_pad
            )
            _set_line_collection_segment(
                vertical,
                [[new_x, new_y - length / 2], [new_x, new_y + length / 2]],
            )
            _set_line_collection_segment(
                top_cap,
                [
                    [new_x - new_cap_width / 2, new_y + length / 2],
                    [new_x + new_cap_width / 2, new_y + length / 2],
                ],
            )
            _set_line_collection_segment(
                bottom_cap,
                [
                    [new_x - new_cap_width / 2, new_y - length / 2],
                    [new_x + new_cap_width / 2, new_y - length / 2],
                ],
            )
            new_x_offset_points, new_default_ha = _scale_bar_label_layout(
                ax,
                spec.get("loc", "lower right"),
                new_x,
                new_y,
                new_cap_width,
                new_label_pad,
            )
            annotation.xy = (new_x, new_y)
            annotation.set_position((new_x_offset_points, 0))
            if "ha" not in spec:
                annotation.set_ha(new_default_ha)
        finally:
            updating["active"] = False

    callback_ids = (
        ax.callbacks.connect("xlim_changed", _update_scale_bar),
        ax.callbacks.connect("ylim_changed", _update_scale_bar),
    )
    callbacks = getattr(ax, "_ecat_scale_bar_callbacks", [])
    callbacks.append((callback_ids, _update_scale_bar))
    ax._ecat_scale_bar_callbacks = callbacks
    return ax


def _normalize_directional_arrows_options(options):
    spec = options.get("directional arrows", False)
    if spec in (False, None, []):
        return []
    if spec is True:
        raise ValueError(
            "'directional arrows' must be False or one/more arrow specs with at least "
            "a 'potential' value."
        )

    if isinstance(spec, dict):
        spec = [spec]
    elif isinstance(spec, (list, tuple)):
        spec = list(spec)
    else:
        raise ValueError(
            "'directional arrows' must be False, a dictionary spec, or a list of dictionary specs."
        )

    normalized = []
    for entry in spec:
        if not isinstance(entry, dict):
            raise ValueError(
                "'directional arrows' entries must each be a dictionary with a 'potential' key."
            )
        if "potential" not in entry:
            raise ValueError(
                "'directional arrows' entries must each define 'potential'."
            )

        allowed_keys = {
            "potential",
            "segment",
            "color",
            "alpha",
            "arrowstyle",
            "size",
        }
        extra_keys = set(entry) - allowed_keys
        if extra_keys:
            unknown = ", ".join(sorted(extra_keys))
            raise ValueError(f"'directional arrows' has unsupported keys: {unknown}")

        try:
            entry["potential"] = float(entry["potential"])
        except (TypeError, ValueError) as exc:
            raise ValueError("'directional arrows' potential must be numeric.") from exc

        if not np.isfinite(entry["potential"]):
            raise ValueError("'directional arrows' potential must be finite.")

        segment = entry.get("segment", None)
        if segment is not None:
            if isinstance(segment, int):
                segment = [int(segment)]
            elif isinstance(segment, (list, tuple, np.ndarray)):
                segment = [int(item) for item in segment]
            else:
                raise ValueError(
                    "'directional arrows' segment must be an integer segment index or a list of segment indices."
                )
            if len(segment) == 0:
                raise ValueError(
                    "'directional arrows' segment list must contain at least one segment index."
                )
            entry["segment"] = sorted(set(int(item) for item in segment if int(item) > 0))
            if not entry["segment"]:
                raise ValueError(
                    "'directional arrows' segment values must be positive integers."
                )

        color = entry.get("color", None)
        if color is not None:
            entry["color"] = color

        arrowstyle = entry.get("arrowstyle", "-|>")
        if not isinstance(arrowstyle, str):
            raise ValueError("'directional arrows' arrowstyle must be a string.")
        arrowstyle = arrowstyle.strip()
        if not arrowstyle:
            raise ValueError("'directional arrows' arrowstyle must be non-empty.")
        entry["arrowstyle"] = arrowstyle

        if "size" in entry:
            try:
                mutation_scale = float(entry["size"])
            except (TypeError, ValueError) as exc:
                raise ValueError("'directional arrows' size must be numeric.") from exc
            if mutation_scale <= 0:
                raise ValueError("'directional arrows' size must be positive.")
            entry["size"] = mutation_scale
        else:
            entry["size"] = 10.0

        if "alpha" in entry:
            try:
                alpha = float(entry["alpha"])
            except (TypeError, ValueError) as exc:
                raise ValueError("'directional arrows' alpha must be numeric.") from exc
            if not (0 <= alpha <= 1):
                raise ValueError("'directional arrows' alpha must be between 0 and 1.")
            entry["alpha"] = alpha
        else:
            entry["alpha"] = 1.0

        normalized.append(entry)
    return normalized


def _split_segment_arrays_for_arrows(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 2:
        return []

    dx = np.diff(x)
    change = np.where(np.diff(np.sign(dx)))[0] + 1
    split_x = np.split(x, change)
    split_y = np.split(y, change)
    return [
        {"number": idx + 1, "x": sx, "y": sy}
        for idx, (sx, sy) in enumerate(zip(split_x, split_y))
    ]


def _segment_contains_potential(segment_x, potential):
    if segment_x.size == 0:
        return False
    x_min = np.nanmin(segment_x)
    x_max = np.nanmax(segment_x)
    low = min(x_min, x_max)
    high = max(x_min, x_max)
    return low <= potential <= high


def _segment_direction_vector(xs, ys, idx):
    if len(xs) < 2 or idx < 0 or idx >= len(xs):
        return 1.0, 0.0

    if idx <= 0:
        dx = xs[1] - xs[0]
        dy = ys[1] - ys[0]
    elif idx >= len(xs) - 1:
        dx = xs[-1] - xs[-2]
        dy = ys[-1] - ys[-2]
    else:
        dx = xs[idx + 1] - xs[idx - 1]
        dy = ys[idx + 1] - ys[idx - 1]
        if np.isclose(dx, 0.0) and np.isclose(dy, 0.0):
            dx = xs[idx + 1] - xs[idx]
            dy = ys[idx + 1] - ys[idx]

    if np.isclose(dx, 0.0) and np.isclose(dy, 0.0):
        dx = 1.0
        dy = 0.0

    norm = np.hypot(dx, dy)
    if norm == 0 or not np.isfinite(norm):
        return 1.0, 0.0
    return dx / norm, dy / norm


def _add_directional_arrows(ax, options, x, y, line_color=None):
    specs = _normalize_directional_arrows_options(options)
    if not specs:
        return []

    default_color = line_color
    if default_color is None:
        default_color = "k"

    segments = _split_segment_arrays_for_arrows(x, y)
    if not segments:
        return []

    segment_lookup = {segment["number"]: segment for segment in segments}
    artists = []
    point_label = options.get("label", None)
    for spec in specs:
        potential = float(spec["potential"])
        explicit_segments = spec.get("segment")

        if explicit_segments is None:
            candidate_segments = [
                number for number, seg in segment_lookup.items()
                if _segment_contains_potential(seg["x"], potential)
            ]
            if not candidate_segments:
                if point_label is not None:
                    point_label = str(point_label)
                continue
        elif isinstance(explicit_segments, int):
            candidate_segments = [explicit_segments]
        else:
            candidate_segments = [int(item) for item in explicit_segments]

        for seg_number in candidate_segments:
            segment = segment_lookup.get(int(seg_number))
            if segment is None:
                continue

            xs = np.asarray(segment["x"], dtype=float)
            ys = np.asarray(segment["y"], dtype=float)
            if xs.size == 0 or ys.size == 0:
                continue

            if not _segment_contains_potential(xs, potential):
                continue

            nearest_idx = int(np.argmin(np.abs(xs - potential)))
            anchor_x = float(xs[nearest_idx])
            anchor_y = float(ys[nearest_idx])
            vx, vy = _segment_direction_vector(xs, ys, nearest_idx)
            if not np.isfinite(vx) or not np.isfinite(vy):
                vx, vy = 1.0, 0.0

            anchor_disp = ax.transData.transform((anchor_x, anchor_y))
            tail_disp = ax.transData.transform((anchor_x + vx, anchor_y + vy))
            tangent_disp = tail_disp - anchor_disp
            if not np.all(np.isfinite(tangent_disp)):
                continue

            tangent_length = np.hypot(tangent_disp[0], tangent_disp[1])
            if tangent_length <= 0:
                continue

            # Draw arrows in display coordinates with a fixed visual length.
            target_points = 16
            scale = target_points / tangent_length
            tip_disp = anchor_disp + tangent_disp * scale
            tip_x, tip_y = ax.transData.inverted().transform(tip_disp)
            color = spec.get("color", default_color)

            arrow = ax.annotate(
                "",
                xy=(tip_x, tip_y),
                xytext=(anchor_x, anchor_y),
                arrowprops={
                    "arrowstyle": spec["arrowstyle"],
                    "color": color,
                    "linewidth": 0.0,
                    "alpha": spec["alpha"],
                    "mutation_scale": spec["size"],
                },
                annotation_clip=True,
            )
            artists.append(arrow)

    return artists


def _place_axis_labels_inside(ax):
    existing_gids = {"ecat-inside-xlabel", "ecat-inside-ylabel"}
    for text in list(ax.texts):
        if text.get_gid() in existing_gids:
            text.remove()

    xlabel = ax.get_xlabel()
    ylabel = ax.get_ylabel()
    if _active_plot_style_value("compact axis labels"):
        xlabel = _compact_echem_axis_label(xlabel)
        ylabel = _compact_echem_axis_label(ylabel)

    if xlabel:
        ax.set_xlabel("")
        ax.text(
            0.97,
            0.055,
            xlabel,
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=mpl.rcParams.get("axes.labelsize"),
            color=mpl.rcParams.get("axes.labelcolor"),
            gid="ecat-inside-xlabel",
        )
    if ylabel:
        ax.set_ylabel("")
        ax.text(
            0.055,
            0.95,
            ylabel,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=mpl.rcParams.get("axes.labelsize"),
            color=mpl.rcParams.get("axes.labelcolor"),
            gid="ecat-inside-ylabel",
        )


def _compact_echem_axis_label(label):
    label = str(label)
    stripped = label.strip()
    if stripped.startswith("Current Density"):
        return _compact_symbol_label("j", stripped[len("Current Density"):])
    if stripped.startswith("Potential"):
        return _compact_symbol_label("E", stripped[len("Potential"):])
    if stripped.startswith("Current"):
        return _compact_symbol_label("i", stripped[len("Current"):])
    if stripped.startswith("Time"):
        return _compact_symbol_label("t", stripped[len("Time"):])
    if stripped.startswith("Charge"):
        return _compact_symbol_label("Q", stripped[len("Charge"):])
    return label


def _compact_symbol_label(symbol, suffix):
    suffix = str(suffix).strip()
    if suffix:
        return rf"${symbol}$ {suffix}"
    return rf"${symbol}$"


def _nice_major_tick_steps(span):
    if not np.isfinite(span) or span <= 0:
        return []
    exponent = int(np.floor(np.log10(span)))
    multipliers = (1.0, 2.0, 2.5, 5.0)
    steps = []
    for power in range(exponent - 3, exponent + 4):
        scale = 10.0 ** power
        steps.extend(multiplier * scale for multiplier in multipliers)
    return sorted({step for step in steps if step > 0})


def _ticks_for_step(lower, upper, step):
    snapped_lower = np.floor(lower / step) * step
    snapped_upper = np.ceil(upper / step) * step
    intervals = int(np.rint((snapped_upper - snapped_lower) / step))
    if intervals <= 0:
        return None
    ticks = snapped_lower + step * np.arange(intervals + 1)
    decimals = max(0, int(np.ceil(-np.log10(step))) + 6) if step < 1 else 6
    ticks = np.round(ticks, decimals)
    ticks[np.isclose(ticks, 0)] = 0
    return ticks


def _snapped_ticks_and_limits(limits, target_intervals=6, min_intervals=5, max_intervals=7):
    lo, hi = float(limits[0]), float(limits[1])
    lower, upper = sorted((lo, hi))
    if not np.isfinite(lower) or not np.isfinite(upper) or lower == upper:
        return None, None

    span = upper - lower
    candidates = []
    for step in _nice_major_tick_steps(span):
        ticks = _ticks_for_step(lower, upper, step)
        if ticks is None:
            continue
        intervals = len(ticks) - 1
        overhang = (lower - ticks[0]) + (ticks[-1] - upper)
        in_range = min_intervals <= intervals <= max_intervals
        candidates.append((
            0 if in_range else 1,
            abs(intervals - target_intervals),
            overhang / span if span else 0,
            step,
            ticks,
        ))

    if not candidates:
        return None, None

    _in_range_penalty, _interval_penalty, _overhang_penalty, _step, ticks = min(candidates)
    if len(ticks) < 2:
        return None, None
    snapped_limits = (float(ticks[-1]), float(ticks[0])) if lo > hi else (float(ticks[0]), float(ticks[-1]))
    return ticks, snapped_limits


def _snap_axis_bounds_to_major_ticks(ax):
    updating = {"active": False}

    def _apply(_ax):
        if updating["active"]:
            return
        updating["active"] = True
        try:
            autoscale_x = _ax.get_autoscalex_on()
            autoscale_y = _ax.get_autoscaley_on()
            xticks, xlim = _snapped_ticks_and_limits(_ax.get_xlim())
            yticks, ylim = _snapped_ticks_and_limits(_ax.get_ylim())
            if xticks is not None:
                _ax.xaxis.set_major_locator(FixedLocator(xticks))
            if xlim is not None:
                _ax.set_xlim(xlim)
            if yticks is not None:
                _ax.yaxis.set_major_locator(FixedLocator(yticks))
            if ylim is not None:
                _ax.set_ylim(ylim)
            _ax.set_autoscalex_on(autoscale_x)
            _ax.set_autoscaley_on(autoscale_y)
        finally:
            updating["active"] = False

    _apply(ax)
    if not getattr(ax, "_ecat_snap_bounds_callbacks", None):
        callback_ids = (
            ax.callbacks.connect("xlim_changed", _apply),
            ax.callbacks.connect("ylim_changed", _apply),
        )
        ax._ecat_snap_bounds_callbacks = callback_ids


def _apply_ecat_axis_style(ax, options=None):
    options = {} if options is None else options
    ax.grid(bool(options.get("grid", False)))
    style_minor_ticks = _active_plot_style_value("minor ticks")
    minor_ticks = options.get("minor ticks", style_minor_ticks or 2)
    if style_minor_ticks is not None and minor_ticks == 2:
        minor_ticks = style_minor_ticks
    if minor_ticks is True:
        minor_ticks = 2
    if minor_ticks is False or minor_ticks in (None, 0):
        ax.minorticks_off()
        if _active_plot_style_value("axis labels") == "inside":
            _place_axis_labels_inside(ax)
        if _active_plot_style_value("snap bounds to ticks"):
            _snap_axis_bounds_to_major_ticks(ax)
        return ax

    minor_ticks = int(minor_ticks)
    ax.xaxis.set_minor_locator(AutoMinorLocator(minor_ticks))
    ax.yaxis.set_minor_locator(AutoMinorLocator(minor_ticks))
    ax.tick_params(which="both", top=mpl.rcParams.get("xtick.top"), right=mpl.rcParams.get("ytick.right"))
    if _active_plot_style_value("axis labels") == "inside":
        _place_axis_labels_inside(ax)
    if _active_plot_style_value("snap bounds to ticks"):
        _snap_axis_bounds_to_major_ticks(ax)
    return ax


def _format_reference_label_mathtext(ref):
    ref = str(ref).strip()
    if ref in {"Fc/Fc+", "DmFc/DmFc+", "CoCp2/CoCp2+"}:
        base = ref[:-1]
        return rf"$\mathrm{{{base}^{{+}}}}$"
    ref = re.sub(r"\+", r"^{+}", ref)
    return rf"$\mathrm{{{ref}}}$"


def _apply_plot_titles(fig, ax, title, subtitle, title_fs, subtitle_fs):
    if title is not None and subtitle is not None:
        axes_top = _active_plot_style_value("title axes top")
        if axes_top is not None:
            fig.subplots_adjust(top=float(axes_top))
        suptitle_y = _active_plot_style_value("suptitle y") or 0.98
        subtitle_pad = _active_plot_style_value("subtitle pad")
        fig.suptitle(title, fontsize=title_fs, y=float(suptitle_y))
        if subtitle_pad is None:
            ax.set_title(subtitle, fontsize=subtitle_fs)
        else:
            ax.set_title(subtitle, fontsize=subtitle_fs, pad=float(subtitle_pad))
    elif title is not None:
        ax.set_title(title, fontsize=title_fs)
    return ax
