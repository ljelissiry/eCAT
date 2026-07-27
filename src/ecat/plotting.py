"""Plotting and object-list display helpers."""

import difflib
import re

from matplotlib.font_manager import FontProperties

from .utils import *  # noqa: F401,F403
from .options import *  # noqa: F401,F403
from .options import _canonical_option_key
from .results import AnalysisResult
from ._plot_helpers import (
    _add_directional_arrows,
    _add_scale_bar,
    _apply_ecat_axis_style,
    _apply_plot_titles,
    _format_reference_label_mathtext,
    _normalize_scale_bar_options,
    _scale_bar_position,
)
from ._plot_style import _active_plot_style_value, plotting_style

def _fit_options_from_analysis_options(options):
    if options is None:
        return {}
    fit_keys = [
        "fit label",
        "fit color",
        "fit linestyle",
        "fit linewidth",
        "fit alpha",
        "fit line range",
        "fit band",
        "fit band level",
        "legend",
        "return stats",
    ]
    return {key: options[key] for key in fit_keys if key in options}


def _scatter_fit(
    x,
    y,
    *,
    label="",
    degree=1,
    plot_fit=True,
    options=None,
):
    fit_options = _fit_options_from_analysis_options(options)
    fit_options["return stats"] = True
    fit_options["print"] = False
    return fit(
        x,
        y,
        label=label,
        degree=degree,
        plot_fit=plot_fit,
        options=fit_options,
    )


def _scatter_fit_row(series, coeffs, stats, fit_x=None):
    coeffs = np.asarray(coeffs, dtype=float)
    row = {"series": series}

    if len(coeffs) >= 2:
        slope = float(coeffs[-2])
        intercept = float(coeffs[-1])
        row["Fit"] = f"y = {slope:.4g}x {intercept:+.4g}"
        row["slope"] = slope
        row["intercept"] = intercept

    if stats is not None:
        if "r2" in stats:
            row["R2"] = float(stats["r2"])
        if "rmse" in stats:
            row["RMSE"] = float(stats["rmse"])
        if "n" in stats:
            row["Fit Points"] = int(stats["n"])

    if fit_x is not None:
        fit_x = np.asarray(fit_x, dtype=float)
        fit_x = fit_x[np.isfinite(fit_x)]
        if len(fit_x) > 0:
            row["fit x min"] = float(np.nanmin(fit_x))
            row["fit x max"] = float(np.nanmax(fit_x))

    return row


def _scatter_fit_table(rows):
    if not rows:
        return pd.DataFrame()
    preferred = ["series", "Fit", "slope", "intercept", "R2", "RMSE", "Fit Points"]
    table = pd.DataFrame(rows)
    ordered = [col for col in preferred if col in table.columns]
    ordered += [col for col in table.columns if col not in ordered]
    return table[ordered]


def _attach_scatter_fit_table(data, rows):
    if isinstance(data, pd.DataFrame):
        data.attrs["fit table"] = _scatter_fit_table(rows)


def _display_table_title(title):
    if title is None:
        return None
    text = str(title).strip()
    return text or None


def _print_table_title(title):
    text = _display_table_title(title)
    if text is None:
        return
    if text.endswith(":"):
        print(text)
    else:
        print(f"{text}:")


def _hide_table_index(styler):
    try:
        return styler.hide(axis="index")
    except TypeError:
        return styler.hide_index()


def _captioned_table_styles(table_styles=None):
    styles = [
        {
            "selector": "caption",
            "props": [
                ("caption-side", "top"),
                ("text-align", "left"),
                ("font-weight", "600"),
                ("color", "inherit"),
                ("margin-bottom", "0.35em"),
            ],
        }
    ]
    if table_styles is None:
        styles.extend(
            [
                {"selector": "th", "props": [("text-align", "left")]},
                {"selector": "td", "props": [("text-align", "left")]},
            ]
        )
    else:
        styles.extend(table_styles)
    return styles


def _can_rich_table_display():
    if display is None:
        return False
    try:
        from IPython import get_ipython

        if get_ipython() is not None:
            return True
    except Exception:
        pass
    display_module = getattr(display, "__module__", "")
    return not str(display_module).startswith("IPython.")


def _plain_table_text(table, *, index=True, justify="left"):
    if hasattr(table, "to_string"):
        return table.to_string(index=index, justify=justify)
    return str(table)


def _display_table(
    table,
    options=None,
    *,
    title=None,
    rich_table=None,
    plain_table=None,
    formatters=None,
    escape=None,
    index=True,
    justify="left",
    table_styles=None,
    plain_title=True,
    format_index=False,
):
    """Display a DataFrame as a captioned rich table, with plain-text fallback."""
    options = {} if options is None else dict(options)
    rich_table = table if rich_table is None else rich_table
    plain_table = table if plain_table is None else plain_table
    caption = _display_table_title(title)

    if options.get("pretty print", True) and _can_rich_table_display():
        try:
            styled = rich_table.style.format(formatters, escape=escape)
            if format_index:
                styled = styled.format_index(escape=escape, axis=0)
            styled = styled.set_properties(
                **{
                    "text-align": "left",
                    "white-space": "pre-wrap",
                    "vertical-align": "top",
                }
            )
            styled = styled.set_table_styles(_captioned_table_styles(table_styles))
            if not index:
                styled = _hide_table_index(styled)
            if caption:
                styled = styled.set_caption(caption)
            with pd.option_context(
                "display.max_colwidth", None,
                "display.max_columns", None,
                "display.max_rows", None,
                "display.width", None,
                "display.expand_frame_repr", False,
            ):
                display(styled)
            return table
        except Exception:
            pass

    if plain_title:
        _print_table_title(caption)
    with pd.option_context(
        "display.max_colwidth", None,
        "display.max_columns", None,
        "display.max_rows", None,
        "display.width", None,
        "display.expand_frame_repr", False,
    ):
        print(_plain_table_text(plain_table, index=index, justify=justify))
    return table


def _print_scatter_fit_statistics(title, fit_table):
    if fit_table is None or len(fit_table) == 0:
        return
    return _display_table(
        fit_table,
        title=f"{title} Fit Statistics",
        index=False,
    )


def _scatterfit_legend_requested(options):
    return bool(options.get("legend", False))


def _scatterfit_legend_fontsize(options):
    fontsize = options.get("legend fontsize", None)
    if fontsize in (None, "auto"):
        return _default_legend_fontsize()
    return fontsize


def _fit_line_option_value(options):
    options = {} if options is None else dict(options)
    return options.get("fit line range", None)


def _is_fit_line_range_pair(value):
    if not isinstance(value, (list, tuple, np.ndarray, pd.Series)) or len(value) != 2:
        return False
    lower, upper = list(value)
    return all(
        item is None
        or (
            isinstance(item, (int, float, np.integer, np.floating))
            and not isinstance(item, (bool, np.bool_))
        )
        for item in (lower, upper)
    )


def _fit_line_label_candidates(label=None, options=None, index=0):
    options = {} if options is None else dict(options)
    candidates = []
    for value in (label, options.get("model label"), options.get("_fit selection label")):
        if value in (None, ""):
            continue
        text = str(value)
        candidates.append(text)
        if text.endswith(" Fit"):
            candidates.append(text[:-4])
        if text.endswith(" fit"):
            candidates.append(text[:-4])
    candidates.extend([str(index), index, "default", "all"])

    deduped = []
    for candidate in candidates:
        if candidate not in deduped:
            deduped.append(candidate)
    return deduped


def _fit_line_range_for_trace(options, *, label=None, index=0):
    value = _fit_line_option_value(options)
    if value is None:
        return None

    if isinstance(value, dict):
        for candidate in _fit_line_label_candidates(label=label, options=options, index=index):
            if candidate in value:
                return value[candidate]
        return None

    if _is_fit_line_range_pair(value):
        return value

    if isinstance(value, (list, tuple)) and value and all(_is_fit_line_range_pair(item) for item in value):
        if index < len(value):
            return value[index]
        return value[-1]

    raise ValueError(
        "'fit line range' must be [x_min, x_max], a dict keyed by fit label, "
        "or a list of [x_min, x_max] ranges."
    )


def _fit_line_x_values(default_x, options=None, *, label=None, index=0, points=300):
    default_x = np.asarray(default_x, dtype=float)
    finite_x = default_x[np.isfinite(default_x)]
    if len(finite_x) == 0:
        return default_x

    default_min = float(np.nanmin(finite_x))
    default_max = float(np.nanmax(finite_x))
    range_spec = _fit_line_range_for_trace(options, label=label, index=index)
    if range_spec is None:
        lower, upper = default_min, default_max
    else:
        lower, upper = list(range_spec)
        lower = default_min if lower is None else float(lower)
        upper = default_max if upper is None else float(upper)

    if not np.isfinite(lower) or not np.isfinite(upper):
        raise ValueError("'fit line range' must resolve to finite x bounds.")
    if upper < lower or (range_spec is not None and upper == lower):
        raise ValueError("'fit line range' upper bound must be greater than the lower bound.")
    if upper == lower:
        return np.full(int(points), lower, dtype=float)

    xscale, _yscale = _resolve_matplotlib_axis_scales(options or {})
    if str(xscale).lower() == "log" and (lower <= 0 or upper <= 0):
        if range_spec is None:
            positive_x = finite_x[finite_x > 0]
            if len(positive_x) > 0:
                lower = float(np.nanmin(positive_x))
                upper = float(np.nanmax(positive_x))
            else:
                raise ValueError(
                    "Cannot draw a default fit line on a logarithmic x-axis because "
                    "the fitted x values contain no positive points."
                )
        else:
            raise ValueError("'fit line range' bounds must be positive when the plotted x-axis is logarithmic.")
        if upper <= lower:
            return np.full(int(points), lower, dtype=float)

    return np.linspace(lower, upper, int(points))


def _normalize_fit_band(value):
    if value in (None, False, ""):
        return None
    token = str(value).strip().lower().replace("_", " ").replace("-", " ")
    if token in {"none", "off", "false", "no", "0"}:
        return None
    if token in {"confidence", "ci", "mean", "mean confidence"}:
        return "confidence"
    if token in {"prediction", "pi", "predictive"}:
        return "prediction"
    if token in {"both", "all"}:
        return "both"
    raise ValueError("'fit band' must be None, 'confidence', 'prediction', or 'both'.")


def _fit_band_level(options):
    level = float((options or {}).get("fit band level", 0.95))
    if not 0 < level < 1:
        raise ValueError("'fit band level' must be between 0 and 1.")
    return level


def _model_prediction_jacobian(function, x_values, params):
    x_values = np.asarray(x_values, dtype=float)
    params = np.asarray(params, dtype=float)
    jacobian = np.empty((len(x_values), len(params)), dtype=float)

    for idx, value in enumerate(params):
        step = np.sqrt(np.finfo(float).eps) * max(abs(float(value)), 1.0)
        hi = params.copy()
        lo = params.copy()
        hi[idx] += step
        lo[idx] -= step
        with np.errstate(all="ignore"):
            y_hi = np.asarray(function(x_values, *hi), dtype=float)
            y_lo = np.asarray(function(x_values, *lo), dtype=float)
        derivative = (y_hi - y_lo) / (2 * step)
        if derivative.shape != x_values.shape:
            derivative = np.broadcast_to(derivative, x_values.shape)
        jacobian[:, idx] = derivative

    return jacobian


def _fit_band_arrays(model_result, x_line, y_line, options=None):
    band = _normalize_fit_band((options or {}).get("fit band"))
    if band is None:
        return []

    model_result = model_result or {}
    pcov = model_result.get("pcov")
    function = model_result.get("function")
    popt = model_result.get("popt")
    dof = int(model_result.get("dof", 0) or 0)
    if pcov is None or function is None or popt is None or dof <= 0:
        return []

    pcov = np.asarray(pcov, dtype=float)
    popt = np.asarray(popt, dtype=float)
    if pcov.shape != (len(popt), len(popt)) or not np.all(np.isfinite(pcov)):
        return []

    x_line = np.asarray(x_line, dtype=float)
    y_line = np.asarray(y_line, dtype=float)
    jacobian = _model_prediction_jacobian(function, x_line, popt)
    mean_variance = np.einsum("ij,jk,ik->i", jacobian, pcov, jacobian)
    mean_se = np.sqrt(np.maximum(mean_variance, 0))
    tcrit = float(scipy.stats.t.ppf((1 + _fit_band_level(options)) / 2, dof))
    if not np.isfinite(tcrit):
        return []

    bands = []
    if band in {"prediction", "both"}:
        residual_variance = float(model_result.get("residual variance", np.nan))
        if np.isfinite(residual_variance) and residual_variance >= 0:
            prediction_se = np.sqrt(np.maximum(mean_variance + residual_variance, 0))
            bands.append({
                "kind": "prediction",
                "lower": y_line - tcrit * prediction_se,
                "upper": y_line + tcrit * prediction_se,
            })
    if band in {"confidence", "both"}:
        bands.append({
            "kind": "confidence",
            "lower": y_line - tcrit * mean_se,
            "upper": y_line + tcrit * mean_se,
        })
    return bands


def _plot_fit_bands(ax, x_line, y_line, model_result, options=None, *, color=None):
    bands = _fit_band_arrays(model_result, x_line, y_line, options)
    if not bands:
        return []

    artists = []
    for band in bands:
        alpha = 0.12 if band["kind"] == "prediction" else 0.20
        artist = ax.fill_between(
            x_line,
            band["lower"],
            band["upper"],
            color=color,
            alpha=alpha,
            linewidth=0,
            label=None,
        )
        artists.append(artist)
    return artists


def _align_multiline_legend_handles_to_first_line(legend):
    """Move multiline legend handles up to the first text line center."""
    if legend is None:
        return legend
    handles = getattr(legend, "legend_handles", None)
    if handles is None:
        handles = getattr(legend, "legendHandles", [])
    texts = legend.get_texts()
    if not handles or not texts:
        return legend

    figure = legend.figure
    if figure is None or figure.canvas is None:
        return legend
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()

    for handle, text in zip(handles, texts):
        label = text.get_text()
        if "\n" not in label or not hasattr(handle, "get_ydata"):
            continue
        ydata = np.asarray(handle.get_ydata(), dtype=float)
        xdata = np.asarray(handle.get_xdata(), dtype=float)
        if ydata.size == 0 or xdata.size == 0:
            continue
        text_bbox = text.get_window_extent(renderer)
        line_count = max(1, label.count("\n") + 1)
        first_line_center_display_y = text_bbox.y1 - text_bbox.height / line_count / 2
        transform = handle.get_transform()
        current_display = transform.transform((xdata[0], ydata[0]))
        target_data_y = transform.inverted().transform(
            (current_display[0], first_line_center_display_y)
        )[1]
        handle.set_ydata(ydata + (target_data_y - ydata[0]))
    return legend


class ScatterFitResult(AnalysisResult):
    """
    Novice-friendly container returned by the standardized scatter-fit names.
    """
    def __init__(
        self,
        table=None,
        fits=None,
        fit_table=None,
        fit_model_results=None,
        raw_table=None,
        transformed_table=None,
        figure=None,
        axes=None,
        summary=None,
    ):
        super().__init__(
            table=table,
            fits=fits,
            fit_table=fit_table,
            fit_model_results=fit_model_results,
            raw_table=raw_table,
            transformed_table=transformed_table,
            figure=figure,
            axes=axes,
            summary=summary,
        )

    def __iter__(self):
        raise TypeError(
            "ScatterFitResult is not tuple-iterable; use .table, .fits, or .fit_table."
        )

    def show(self, options=None):
        options = {} if options is None else dict(options)
        if self.fit_model_results:
            from .analysis_batch import _print_fit_model_results

            _print_fit_model_results(self.fit_model_results, options)
        else:
            if self.summary:
                for key, value in self.summary.items():
                    print(f"{key}: {value}")
            if self.table is not None:
                _display_table(self.table, options, title="Result Table")
            if self.fit_table is not None:
                _display_table(self.fit_table, options, title="Fit Table")
        if options.get("return", False):
            return self.table
        return None


def _fit_table_from_fits(fits):
    if fits is None:
        return pd.DataFrame()

    if isinstance(fits, dict):
        rows = []
        for label, coeffs in fits.items():
            coeffs = np.asarray(coeffs, dtype=float)
            row = {"series": label}
            if len(coeffs) >= 2:
                row["slope"] = coeffs[-2]
                row["intercept"] = coeffs[-1]
            rows.append(row)
        return pd.DataFrame(rows)

    if isinstance(fits, (list, tuple)) and len(fits) > 0 and not np.isscalar(fits[0]):
        rows = []
        for i, coeffs in enumerate(fits):
            coeffs = np.asarray(coeffs, dtype=float)
            row = {"series": i}
            if len(coeffs) >= 2:
                row["slope"] = coeffs[-2]
                row["intercept"] = coeffs[-1]
            rows.append(row)
        return pd.DataFrame(rows)

    coeffs = np.asarray(fits, dtype=float)
    row = {}
    if len(coeffs) >= 2:
        row["slope"] = coeffs[-2]
        row["intercept"] = coeffs[-1]
    return pd.DataFrame([row])


def _scatter_result_from_payload(payload, summary=None):
    if not isinstance(payload, tuple):
        return ScatterFitResult(table=payload, summary=summary)

    table = payload[0] if len(payload) > 0 else None
    fits = payload[1] if len(payload) > 1 else None

    if len(payload) == 3 and isinstance(payload[1], pd.DataFrame):
        table = payload[1]
        fits = payload[2]

    fit_table = None
    fit_model_results = {}
    if isinstance(table, pd.DataFrame):
        fit_table = table.attrs.get("fit table")
        fit_model_results = table.attrs.get("fit model results", {}) or {}
    if fit_table is None:
        fit_table = _fit_table_from_fits(fits)

    return ScatterFitResult(
        table=table,
        fits=fits,
        fit_table=fit_table,
        fit_model_results=fit_model_results,
        summary=summary,
    )


class _ScatterTrace:
    def __init__(self, name):
        self.name = str(name)

    def txt_stats(self, options=None):
        return {"name": self.name}


def _multi_scatter_column_lookup(df):
    return {
        str(col).strip().lower().replace("_", " ").replace("\n", " "): col
        for col in df.columns
    }


def _multi_scatter_exact_column(df, requested, required=True):
    if requested in df.columns:
        return requested

    lookup = _multi_scatter_column_lookup(df)
    key = str(requested).strip().lower().replace("_", " ").replace("\n", " ")
    if key in lookup:
        return lookup[key]

    if required:
        raise ValueError(
            f"Could not resolve column '{requested}'. Available columns: {list(df.columns)}"
        )
    return None


def _multi_scatter_find_preferred_column(df, preferred, kind):
    lookup = _multi_scatter_column_lookup(df)

    for candidate in preferred:
        key = str(candidate).strip().lower().replace("_", " ").replace("\n", " ")
        if key in lookup:
            return lookup[key]

    matches = []
    if kind == "x":
        for col in df.columns:
            text = str(col).strip().lower().replace("_", " ")
            if "scan rate" in text or "concentration" in text or text in {"x", "potential"}:
                matches.append(col)
    else:
        for col in df.columns:
            text = str(col).strip().lower().replace("_", " ")
            compact = text.replace(" ", "")
            if (
                compact in {"kobs", "tofmax", "ip", "ep"}
                or "kobs" in compact
                or "tofmax" in compact
                or text.endswith(" ip")
                or " ip " in text
                or text.endswith(" ep")
                or " ep " in text
            ):
                matches.append(col)

    unique_matches = []
    for match in matches:
        if match not in unique_matches:
            unique_matches.append(match)

    if len(unique_matches) == 1:
        return unique_matches[0]
    if len(unique_matches) > 1:
        raise ValueError(
            f"Could not auto-resolve {kind} column because multiple candidates were found: "
            f"{unique_matches}. Available columns: {list(df.columns)}"
        )

    raise ValueError(
        f"Could not auto-resolve {kind} column. Available columns: {list(df.columns)}"
    )


def _resolve_multi_scatter_x_column(df, options):
    requested = options.get("x column", "auto")
    if requested not in (None, "auto"):
        return _multi_scatter_exact_column(df, requested)
    return _multi_scatter_find_preferred_column(
        df,
        ["x transformed", "x raw"],
        "x",
    )


def _resolve_multi_scatter_y_columns(df, options):
    requested_columns = options.get("y columns", None)
    if requested_columns is None:
        requested_column = options.get("y column", "auto")
        if requested_column in (None, "auto"):
            metric = options.get("metric")
            preferred = ["y transformed"]
            if metric not in (None, "auto", ""):
                preferred.append(metric)
            preferred.extend(["kobs", "TOFmax", "ip", "Ep"])
            return [_multi_scatter_find_preferred_column(df, preferred, "y")]
        requested_columns = [requested_column]
    elif isinstance(requested_columns, str):
        requested_columns = [requested_columns]

    return [_multi_scatter_exact_column(df, col) for col in requested_columns]


def _coerce_multi_scatter_dataset(value, options=None):
    if isinstance(value, AnalysisResult):
        if isinstance(value.raw_table, pd.DataFrame) and not value.raw_table.empty:
            return value.raw_table.copy(), value
        if isinstance(value.table, pd.DataFrame) and not value.table.empty:
            return value.table.copy(), value
        raise ValueError("AnalysisResult inputs must have plottable point data.")
    if isinstance(value, pd.DataFrame):
        return value.copy(), None
    raise TypeError(
        "multi_scatterplot datasets values must be pandas DataFrames or AnalysisResult objects."
    )


def _multi_scatter_column_key(column):
    return str(column).strip().lower().replace("_", " ").replace("\n", " ")


def _multi_scatter_constant_value(df, column, default=None):
    resolved = _multi_scatter_exact_column(df, column, required=False)
    if resolved is None:
        return default

    values = df[resolved].dropna().unique()
    if len(values) == 1:
        return values[0]
    return default


def _infer_multi_scatter_metric_column(df, y_col, options):
    metric = options.get("metric")
    if metric not in (None, "auto", ""):
        resolved = _multi_scatter_exact_column(df, metric, required=False)
        if resolved is not None:
            return resolved

    metadata_keys = {
        "x raw",
        "x transformed",
        "x transform",
        "x label",
        "x unit",
        "x kind",
        "y transformed",
        "y label",
        "y unit",
        "y transform",
        "y transform note",
    }
    preferred = ["kobs", "TOFmax", "ip", "Ep"]
    matches = [
        _multi_scatter_exact_column(df, candidate, required=False)
        for candidate in preferred
    ]
    matches = [match for match in matches if match is not None]
    if len(matches) == 1:
        return matches[0]

    numeric_candidates = []
    for column in df.columns:
        if column == y_col:
            continue
        if _multi_scatter_column_key(column) in metadata_keys:
            continue
        values = pd.to_numeric(df[column], errors="coerce")
        if values.notna().any():
            numeric_candidates.append(column)

    if len(numeric_candidates) == 1:
        return numeric_candidates[0]
    return y_col


def _multi_scatter_option_is_active(value):
    return value not in (None, "", False, "none", "None", "auto")


def _multi_scatter_requested_transforms(options):
    if not any(
        _multi_scatter_option_is_active(options.get(key))
        for key in ("transform mode", "x transform", "y transform")
    ):
        return None, None
    from .analysis_batch import _resolve_xy_transforms

    x_transform, y_transform, _mode_label = _resolve_xy_transforms(
        options,
        default_x="identity",
        default_y="identity",
    )
    return x_transform, y_transform


def _multi_scatter_x_metadata(df, column):
    key = _multi_scatter_column_key(column)
    if key in {"x raw", "x transformed"}:
        base_label = _multi_scatter_constant_value(df, "x label", column)
        unit = _multi_scatter_constant_value(df, "x unit", "")
        x_kind = _multi_scatter_constant_value(df, "x kind", "custom")
        return base_label, unit, x_kind
    return str(column), "", "custom"


def _multi_scatter_metric_metadata(df, column, options):
    key = _multi_scatter_column_key(column)
    if key not in {"y raw", "y adjusted", "y transformed"}:
        units = getattr(df, "attrs", {}).get("units", {}) or {}
        unit = units.get(column) or units.get(str(column).lower())
        return str(column), unit

    metric = _multi_scatter_constant_value(df, "y label", None)
    if metric is None:
        metric = _infer_multi_scatter_metric_column(df, column, options)
    unit = _multi_scatter_constant_value(df, "y unit", None)
    if unit is None:
        units = getattr(df, "attrs", {}).get("units", {}) or {}
        unit = units.get(metric) or units.get(str(metric).lower()) or units.get(column)
    return str(metric), unit


def _multi_scatter_effective_y_mode(df, column, options):
    mode = options.get("y mode")
    if not _multi_scatter_option_is_active(mode):
        key = _multi_scatter_column_key(column)
        if key in {"y adjusted", "y transformed"}:
            mode = _multi_scatter_constant_value(df, "y mode", "raw")
        else:
            mode = "raw"
    return _normalize_y_mode(mode)


def _multi_scatter_axis_label(df, column, axis, options):
    key = _multi_scatter_column_key(column)
    requested_x_transform, requested_y_transform = _multi_scatter_requested_transforms(options)

    if axis == "x":
        base_label, unit, x_kind = _multi_scatter_x_metadata(df, column)
        if requested_x_transform is not None:
            return _format_fit_rate_x_label(
                base_label,
                unit=unit,
                x_kind=x_kind,
                transform=requested_x_transform,
                log=str(requested_x_transform).strip().lower() == "log10",
            )
        if key == "x transformed":
            transform = _multi_scatter_constant_value(df, "x transform", "identity")
            return _format_fit_rate_x_label(
                base_label,
                unit=unit,
                x_kind=x_kind,
                transform=transform,
                log=str(transform).strip().lower() == "log10",
            )
        if key == "x raw":
            return _format_fit_rate_x_label(base_label, unit=unit, x_kind=x_kind)

    if axis == "y":
        transform = requested_y_transform
        if transform is None and key == "y transformed":
            transform = _multi_scatter_constant_value(df, "y transform", "identity")
        if transform is None:
            transform = "identity"
        transform_is_log10 = str(transform).strip().lower() == "log10"
        metric, unit = _multi_scatter_metric_metadata(df, column, options)
        mode = _multi_scatter_effective_y_mode(df, column, options)

        if mode == "raw":
            return _format_fit_rate_metric_label(
                metric,
                log=transform_is_log10,
                unit=unit,
            )

        y_mode_label = _format_y_mode_axis_label(
            _format_fit_rate_metric_label(metric),
            mode,
        )
        return _format_y_transform_axis_label(y_mode_label, transform)

    if axis == "x":
        return format_chemical_formulas(str(column), mode="mathtext")

    return str(column)


def _plot_multi_scatter_trace(ax, x_values, y_values, plot_style, color, label):
    if plot_style == "scatter":
        return ax.scatter(x_values, y_values, color=color, label=label, zorder=3)
    if plot_style == "line":
        line, = ax.plot(x_values, y_values, color=color, label=label)
        return line
    if plot_style in ("line+markers", "line markers", "line-and-markers"):
        line, = ax.plot(x_values, y_values, marker="o", color=color, label=label)
        return line
    raise ValueError("plot style must be 'scatter', 'line', or 'line+markers'.")


def _resolve_matplotlib_axis_scales(options):
    options = {} if options is None else options
    xscale = options.get("xscale")
    yscale = options.get("yscale")
    plot_scale = options.get("plot scale")

    if plot_scale not in (None, "", "auto"):
        normalized = str(plot_scale).strip().lower().replace("_", "-").replace(" ", "-")
        if normalized in {"linear", "none"}:
            preset_xscale, preset_yscale = "linear", "linear"
        elif normalized in {"log-log", "loglog"}:
            preset_xscale, preset_yscale = "log", "log"
        elif normalized in {"semilogx", "semi-log-x"}:
            preset_xscale, preset_yscale = "log", "linear"
        elif normalized in {"semilogy", "semi-log-y"}:
            preset_xscale, preset_yscale = "linear", "log"
        elif normalized == "symlog":
            preset_xscale, preset_yscale = "symlog", "symlog"
        else:
            raise ValueError(
                "'plot scale' must be 'linear', 'log-log', 'semilogx', 'semilogy', or 'symlog'."
            )
        if xscale in (None, "", "auto"):
            xscale = preset_xscale
        if yscale in (None, "", "auto"):
            yscale = preset_yscale

    return xscale, yscale


def _apply_matplotlib_axis_scales(ax, options):
    xscale, yscale = _resolve_matplotlib_axis_scales(options)
    if xscale not in (None, "", "auto"):
        ax.set_xscale(xscale)
    if yscale not in (None, "", "auto"):
        ax.set_yscale(yscale)
    return ax


def _multi_scatter_fit_curve(x_values, model_result, options=None, *, label=None, index=0):
    x_values = np.asarray(x_values, dtype=float)
    if len(x_values) < 2:
        raise ValueError("At least two x values are required for a fit overlay.")

    x_curve = _fit_line_x_values(
        x_values,
        options,
        label=label,
        index=index,
        points=100,
    )

    model_result = model_result or {}
    function = model_result.get("function")
    popt = model_result.get("popt")
    if function is None or popt is None:
        raise ValueError("Fit overlay is unavailable for this dataset.")

    y_curve = function(np.asarray(x_curve, dtype=float), *np.asarray(popt, dtype=float))
    finite = np.isfinite(x_curve) & np.isfinite(y_curve)
    if not np.any(finite):
        raise ValueError("Fit overlay could not be evaluated on finite points.")
    return x_curve[finite], y_curve[finite]


def _multi_scatter_fit_mode(options):
    fit_value = options.get("fit", True)
    if fit_value in (False, None, "none", "None", "off", "Off"):
        return "none"
    if str(fit_value).strip().lower() == "stored":
        return "stored"
    return "refit"


def _multi_scatter_transform_xy(x_values, y_values, options, series_keys=None):
    if not any(
        options.get(key) not in (None, "", False, "none", "None")
        for key in ("y mode", "transform mode", "x transform", "y transform", "floor", "x floor", "y floor", "y0")
    ):
        return np.asarray(x_values, dtype=float), np.asarray(y_values, dtype=float)

    try:
        from .analysis_batch import _apply_scatter_transforms, _apply_y_mode, _resolve_xy_transforms
    except Exception as exc:  # pragma: no cover - import path safety
        raise RuntimeError(
            "multi_scatterplot scatterfit modes require internal analysis helpers "
            "for the current install."
        ) from exc

    y_adjustment = _apply_y_mode(y_values, options, series_keys=series_keys)
    x_transform, y_transform, _mode_label = _resolve_xy_transforms(
        options,
        default_x="identity",
        default_y="identity",
    )
    transformed = _apply_scatter_transforms(
        x_values,
        y_adjustment["adjusted"],
        x_transform,
        y_transform,
        options,
    )
    return transformed["x"], transformed["y"]


def multi_scatterplot(datasets, options=None):
    """Overlay multiple scatter-fit result tables or dataframes on one plot.
    
    Parameters
    ----------
    datasets : dict
        Mapping of legend labels to dataframes or ScatterFitResult objects.
    options : dict or MultiScatterplotOptions, optional
        Column, style, legend, and fit-display options. See ``e.describe_options("multi_scatterplot")``.
    
    Returns
    -------
    ScatterFitResult
        Result object with figure, axes, plotted table, and fit table metadata.
    
    Examples
    --------
    >>> e.multi_scatterplot({"Fe only": result_a, "Mg": result_b})
    """
    options = MultiScatterplotOptions.from_options(options).to_options_dict()
    options.setdefault("legend", True)
    plot_fit = bool(options.get("plot fit", True))
    fit_mode = _multi_scatter_fit_mode(options)

    if not isinstance(datasets, dict) or len(datasets) == 0:
        raise ValueError("multi_scatterplot requires a non-empty dict of labeled datasets.")

    plot_style = str(options.get("plot style", "scatter")).strip().lower()
    dataset_items = list(datasets.items())
    trace_objects = [_ScatterTrace(label) for label, _ in dataset_items]
    style_options = options.copy()
    style_options["labels"] = list(datasets.keys())
    style = _prepare_multiplot_style(trace_objects, style_options)
    ax = style["ax"]
    color_spec = style["color spec"]

    plotted_rows = []
    fit_rows = []
    first_x_col = None
    first_y_col = None
    first_x_label = None
    first_y_label = None
    _multi_scatter_fit_series = None
    fit_line_index = 0

    for i, (label, value) in enumerate(dataset_items):
        df, result = _coerce_multi_scatter_dataset(value, options)
        x_col = _resolve_multi_scatter_x_column(df, options)
        y_cols = _resolve_multi_scatter_y_columns(df, options)

        if first_x_col is None:
            first_x_col = x_col
            first_x_label = _multi_scatter_axis_label(df, x_col, "x", options)
        if first_y_col is None and y_cols:
            first_y_col = y_cols[0]
            first_y_label = _multi_scatter_axis_label(df, y_cols[0], "y", options)

        trace_label_base = color_spec["labels"][i]
        color = color_spec["line colors"][i]

        for y_col_i, y_col in enumerate(y_cols):
            plot_df = df[[x_col, y_col]].copy()
            plot_df = plot_df.apply(pd.to_numeric, errors="coerce").dropna()
            if plot_df.empty:
                continue

            trace_label = trace_label_base
            if len(y_cols) > 1:
                trace_label = f"{trace_label_base} {y_col}"

            x_values = plot_df[x_col].to_numpy(dtype=float)
            y_values = plot_df[y_col].to_numpy(dtype=float)
            order = np.argsort(x_values)
            x_values = x_values[order]
            y_values = y_values[order]
            raw_x_values = np.asarray(x_values, dtype=float)
            raw_y_values = np.asarray(y_values, dtype=float)
            x_values, y_values = _multi_scatter_transform_xy(
                x_values,
                y_values,
                options,
                series_keys=[label, y_col, "y", "default"],
            )

            _plot_multi_scatter_trace(
                ax,
                x_values,
                y_values,
                plot_style,
                color,
                trace_label,
            )

            for raw_x, raw_y, x_value, y_value in zip(raw_x_values, raw_y_values, x_values, y_values):
                plotted_rows.append({
                    "series": label,
                    "x column": x_col,
                    "y column": y_col,
                    "x raw": raw_x,
                    "y raw": raw_y,
                    "x plotted": x_value,
                    "y plotted": y_value,
                    "x transformed": x_value,
                    "y transformed": y_value,
                    "x": x_value,
                    "y": y_value,
                })

            if fit_mode == "refit":
                if _multi_scatter_fit_series is None:
                    try:
                        from .analysis_batch import _fit_series_xy
                    except Exception as exc:  # pragma: no cover - import path safety
                        raise RuntimeError(
                            "multi_scatterplot fit overlays currently require internal analysis helpers "
                            "for the current install."
                        ) from exc
                    _multi_scatter_fit_series = _fit_series_xy

                try:
                    series_fit = _multi_scatter_fit_series(
                        x_values,
                        y_values,
                        options={**options, "print": False},
                        label=trace_label,
                    )
                except Exception as exc:
                    warnings.warn(f"Could not fit curve for {trace_label} / {y_col}: {exc}")
                    continue

                model_result = series_fit.get("model_result", {})
                if model_result:
                    if plot_fit:
                        try:
                            fit_x, fit_y = _multi_scatter_fit_curve(
                                x_values,
                                model_result,
                                options,
                                label=trace_label,
                                index=fit_line_index,
                            )
                        except ValueError as exc:
                            warnings.warn(str(exc))
                        else:
                            _plot_fit_bands(
                                ax,
                                fit_x,
                                fit_y,
                                model_result,
                                options,
                                color=color,
                            )
                            ax.plot(
                                fit_x,
                                fit_y,
                                color=color,
                                linestyle=options.get("fit linestyle", "--"),
                                linewidth=options.get("fit linewidth", 1),
                                alpha=options.get("fit alpha", 1),
                                label=f"{trace_label} fit",
                            )
                            fit_line_index += 1

                    for fit_row in series_fit.get("fit_rows", []):
                        fit_row = dict(fit_row)
                        fit_row["series"] = label
                        fit_row["y column"] = y_col
                        fit_rows.append(fit_row)

    if first_x_col is not None:
        ax.set_xlabel(options.get("xlabel") or first_x_label or str(first_x_col))
    if first_y_col is not None:
        ax.set_ylabel(options.get("ylabel") or first_y_label or str(first_y_col))

    _apply_matplotlib_axis_scales(ax, options)
    _finish_multiplot_style(trace_objects, style_options, style)

    fit_table = pd.DataFrame(fit_rows)
    if options.get("print", False):
        _print_scatter_fit_statistics("Multi Scatterplot", fit_table)

    plotted_table = pd.DataFrame(plotted_rows)
    return ScatterFitResult(
        table=plotted_table,
        fit_table=fit_table,
        raw_table=plotted_table,
        transformed_table=plotted_table,
        figure=ax.figure,
        axes=ax,
        summary={
            "analysis": "multi scatterplot",
            "n datasets": len(datasets),
            "plot style": plot_style,
        },
    )

def show_groups(groups, options=None):
    """
    Print a list of groups of echem objects.

    Parameters
    ----------
    groups : list[list]
        A list of grouped echem objects.
    options : dict or None
        Options passed through to show_objects().
    """
    if options is None:
        options = {}
    if options.get("group keys") is not None:
        raise ValueError(
            "show_groups() accepts already-grouped objects and does not accept "
            "'group keys'. Use group() or sort_and_group() to create groups first."
        )

    if not groups:
        print("No groups to print.")
        return [] if options.get("return", False) else None

    return_tables = bool(options.get("return", False))
    tables = []

    for n, grp in enumerate(groups):
        print(f"### Group {n} ###")

        if len(grp) == 0:
            print("(empty group)\n")
            if return_tables:
                tables.append(None)
            continue

        group_options = options.copy()
        group_options["group number"] = n
        if return_tables:
            group_options["return"] = True

        table = show_objects(grp, options=group_options)
        if return_tables:
            tables.append(table)
        print()

    return tables if return_tables else None


def _is_echem_object(value):
    from .objects import echem

    return isinstance(value, echem)


def _is_echem_list(value):
    return isinstance(value, list) and (
        len(value) == 0 or all(_is_echem_object(item) for item in value)
    )


def _is_grouped_echem_list(value):
    return isinstance(value, list) and len(value) > 0 and all(
        isinstance(item, list) and all(_is_echem_object(obj) for obj in item)
        for item in value
    )


def _single_object_reference_root(echem_object):
    filepath = getattr(echem_object, "filepath", None)
    if filepath in (None, ""):
        return None

    obj_dir = os.path.dirname(os.path.abspath(filepath))
    folderpath = getattr(echem_object, "folderpath", "")
    if folderpath in (None, "", "."):
        return obj_dir

    root = obj_dir
    parts = [
        part for part in str(folderpath).replace("\\", os.sep).split(os.sep)
        if part not in ("", ".")
    ]
    for _part in parts:
        root = os.path.dirname(root)
    return root


def _single_object_reference_source_display(echem_object, value):
    if value in (None, ""):
        return value
    root = _single_object_reference_root(echem_object)
    if root is None:
        return os.path.basename(str(value))
    return _format_reference_display(str(value), root)


def _display_folder_path_value(value):
    if value in (None, "", "."):
        return ""
    return _format_path_for_display(value)


def _object_info_table(echem_object, options=None):
    if options is None:
        options = {}
    info = echem_object.info()
    sig_figs = options.get("sig figs", 3)

    def split_metric_unit(key):
        match = re.match(r"^(.*?)\s*\(([^()]+)\)\s*$", str(key))
        if match is None:
            return key, None
        return match.group(1).strip(), match.group(2).strip()

    def is_empty_info_value(key, value):
        if key == "folder path" and value in (".", "", None):
            return True
        if key == "reference mode" and value in (None, "", "none"):
            return True
        if key.startswith("reference ") and value in (None, ""):
            return True
        if value is None:
            return True
        if isinstance(value, float) and pd.isna(value):
            return True
        if isinstance(value, str) and value.strip() == "":
            return True
        if isinstance(value, (list, tuple, set, dict)) and len(value) == 0:
            return True
        return False

    def combined_compounds_display():
        compounds = info.get("compounds")
        if is_empty_info_value("compounds", compounds):
            return None

        concentrations = info.get("concentrations", [])
        if isinstance(compounds, str):
            compounds = [compounds]
        else:
            compounds = list(compounds)
        if isinstance(concentrations, str):
            concentrations = [concentrations]
        else:
            concentrations = list(concentrations or [])

        combine = getattr(echem_object, "combine_concs_chems", None)
        if callable(combine):
            combined = combine(concentrations, compounds, options)
        else:
            padded = concentrations + [""] * len(compounds)
            combined = [
                f"{conc + ' ' if conc else ''}{compound}"
                for conc, compound in zip(padded, compounds)
            ]
        if isinstance(combined, str):
            return combined
        return ", ".join(str(entry) for entry in combined if entry not in (None, ""))

    def format_info_value(key, value):
        _metric_key, unit = split_metric_unit(key)
        if key == "reference source file":
            return _single_object_reference_source_display(echem_object, value)
        if getattr(echem_object, "type", None) == "Differential Pulse Voltammetry":
            if key in {"amplitude", "pulse width", "sample width", "pulse period"}:
                formatted = _format_dpv_display_value(key, value, sig_figs)
                if formatted != "":
                    return formatted
        if key in {"ir comp resistance", "ir uncomp resistance"}:
            if isinstance(value, (np.integer, int, np.floating, float)):
                if pd.isna(value):
                    return ""
                return f"{round_sigfigs(float(value), sig_figs):g} ohm"
        if key == "ir comp percent":
            if isinstance(value, (np.integer, int, np.floating, float)):
                if pd.isna(value):
                    return ""
                return f"{round_sigfigs(float(value), sig_figs):g} %"
        if isinstance(value, bool):
            return value
        if isinstance(value, (np.integer, int, np.floating, float)):
            if pd.isna(value):
                return ""
            if unit:
                scaled, scaled_unit = scale_value(float(value), unit, selected_unit="auto")
                scaled = round_sigfigs(float(scaled), sig_figs)
                return f"{scaled:g} {scaled_unit}"
            if isinstance(value, (np.integer, int)):
                return int(value)
            return f"{round_sigfigs(float(value), sig_figs):g}"
        return value

    combined_compounds = combined_compounds_display()
    if combined_compounds is not None:
        info["compounds"] = combined_compounds
        info.pop("concentrations", None)

    rows = [
        {
            "Metric": pretty_table_column_label(split_metric_unit(key)[0]),
            "Value": format_info_value(key, value),
        }
        for key, value in info.items()
        if not is_empty_info_value(key, value)
    ]
    return pd.DataFrame(rows)


def _show_single_object(echem_object, options=None):
    if options is None:
        options = {}
    options = dict(options)
    table = _object_info_table(echem_object, options)
    table = table.fillna("")

    if options.get("pretty print", True):
        display_object_table(table)
    else:
        print(table.to_string(index=False))

    return table if options.get("return", False) else None


def show(value, options=None):
    """Display one eCAT object, one group, grouped objects, or showable results."""
    if options is None:
        options = {}

    if _is_echem_object(value):
        return _show_single_object(value, options)
    if _is_grouped_echem_list(value):
        return show_groups(value, options)
    if _is_echem_list(value):
        return show_objects(value, options)

    show_method = getattr(value, "show", None)
    if callable(show_method):
        return show_method(options)

    raise TypeError(
        "show() expected an eCAT object, a list of eCAT objects, grouped eCAT "
        "objects, or an object with a show() method."
    )


def _requested_reference_columns(options):
    requested = _coerce_display_columns(options.get("columns", []))

    ordered = [
        "reference shift",
        "reference label",
        "reference mode",
        "reference source",
    ]
    return [col for col in ordered if col in requested]


def _reference_mode_value(echem_object):
    mode = getattr(echem_object, "reference_mode", None)
    if mode in (None, "none", ""):
        return ""
    return str(mode)


def _object_list_root(object_list):
    paths = [
        os.path.abspath(obj.filepath)
        for obj in object_list
        if getattr(obj, "filepath", None)
    ]
    if not paths:
        return None
    try:
        return os.path.commonpath(paths)
    except ValueError:
        return None


def _find_reference_index(reference_file, object_list):
    if reference_file is None:
        return None

    ref_abs = os.path.abspath(reference_file)

    for i, obj in enumerate(object_list):
        obj_path = getattr(obj, "filepath", None)
        if obj_path is None:
            continue
        if os.path.abspath(obj_path) == ref_abs:
            return i

    return None


def _reference_source_display(echem_object, object_list):
    ref_file = getattr(echem_object, "reference_source_file", None)
    if ref_file is None:
        return ""

    ref_idx = _find_reference_index(ref_file, object_list)
    if ref_idx is not None:
        return f"[{ref_idx}]"

    root_abs = _object_list_root(object_list)
    if root_abs is not None:
        return _format_reference_display(ref_file, root_abs)

    return os.path.basename(ref_file)


def _reference_shift_display(echem_object):
    shift = getattr(echem_object, "reference_shift", None)
    if shift is None:
        return ""
    return round_sigfigs(shift, 4)


def _reference_label_display(echem_object):
    label = getattr(echem_object, "reference_label", None)
    if label in (None, ""):
        return ""
    return str(label)


def _reference_inline_text(echem_object, object_list, inline_reference_columns):
    """
    Build plain-print inline reference text, e.g.
        -> fallback to [4] (0.413 V)
        -> self (0.412 V)
        -> folder (0.413 V)
        -> manual (0.400 V)
    Only uses columns that are requested and not already printed as shared
    reference conditions.
    """
    cols = set(inline_reference_columns)
    if not cols:
        return ""

    mode = _reference_mode_value(echem_object)
    source = _reference_source_display(echem_object, object_list) if "reference source" in cols else ""
    shift = _reference_shift_display(echem_object) if "reference shift" in cols else ""
    label = _reference_label_display(echem_object) if "reference label" in cols else ""

    left = ""

    if "reference mode" in cols and mode:
        if mode == "fallback":
            left = "fallback"
            if source:
                left += f" to {source}"
        elif mode == "manual":
            left = "manual"
        elif mode == "self":
            left = "self"
        elif mode == "folder":
            left = "folder"
        else:
            left = mode
    else:
        if source:
            left = source

    extras = []
    if label:
        extras.append(label)

    text = ""
    if left or extras or shift != "":
        text = "  -> "
        if left:
            text += left
        if extras:
            if left:
                text += " | "
            text += ", ".join(extras)
        if shift != "":
            text += f" ({shift} V)"

    return text

def pretty_table_column_label(key):
    """
    Display/returned dataframe column labels.

    Most columns are title-cased, but selected scientific terms preserve
    their conventional capitalization.
    """
    text = str(key).strip().replace("_", " ")
    lower = text.lower()

    phrase_map = {
        "reference cv": "Reference CV",
        "reference ep": "Reference Ep",
        "redox cv": "Redox CV",
        "redox delta e": "Redox Delta E",
        "exp type": "Exp Type",
        "r2": "R2",
        "ip": "ip",
        "ip0": "ip0",
        "ip0 source": "ip0 Source",
        "ip0 scan rate": "ip0 Scan Rate",
        "ip0 sqrt scan rate slope": "ip0 sqrt scan rate slope",
        "ip0 tangent": "ip0 Tangent",
        "ilim": "ilim",
        "ilim/ip0": "ilim/ip0",
        "ilim source": "ilim Source",
        "ilim tangent": "ilim Tangent",
        "ir comp resistance": "IR Comp Resistance",
        "ir uncomp resistance": "IR Uncomp Resistance",
        "ir comp percent": "IR Comp Percent",
        "k obs": "kobs",
        "kobs": "kobs",
        "tof max": "TOFmax",
        "tofmax": "TOFmax",
        "tof_max": "TOFmax",
        "init potential (v)": "Init Potential",
        "initial potential (v)": "Initial Potential",
        "final potential (v)": "Final Potential",
        "min potential (v)": "Min Potential",
        "max potential (v)": "Max Potential",
        "sample interval (s)": "Sample Interval",
        "run time (s)": "Run Time",
        "quiet time (s)": "Quiet Time",
        "sensitivity (a/v)": "Sensitivity",
        "cathodic current (a)": "Cathodic Current",
        "anodic current (a)": "Anodic Current",
        "cathodic time (s)": "Cathodic Time",
        "anodic time (s)": "Anodic Time",
        "high e limit (v)": "High E Limit",
        "low e limit (v)": "Low E Limit",
        "sample interval (sec)": "Sample Interval",
        "total duration (s)": "Total Duration",
        "catalytic ecat/2": "Catalytic Ecat/2",
        "fowa fit": "FOWA Fit",
        "n fit points": "Fit Points",
        "ecat/2 shift": "Ecat/2 - E1/2",
    }

    if lower in phrase_map:
        return phrase_map[lower]

    word_map = {
        "cv": "CV",
        "r2": "R2",
        "ip": "ip",
        "ip0": "ip0",
        "ep": "Ep",
        "e1/2": "E1/2",
    }

    words = text.split()
    out = [word_map.get(w.lower(), w.capitalize()) for w in words]
    return " ".join(out)


def _pretty_table_header_html_label(column):
    """
    Display-only HTML header formatting for notebook tables.

    Returned DataFrames keep plain-text column names; this helper only gives the
    pretty printer the usual electrochemical subscripts/superscripts.
    """
    text = str(column).strip()
    if " / " in text:
        base, unit = text.rsplit(" / ", 1)
        return f"{_pretty_table_header_html_label(base)} / {_pretty_unit_html_label(unit)}"
    key = text.lower()
    header_html_map = {
        "kobs": "k<sub>obs</sub>",
        "tofmax": "TOF<sub>max</sub>",
        "tof max": "TOF<sub>max</sub>",
        "ip": "i<sub>p</sub>",
        "ip0": "i<sub>p</sub><sup>0</sup>",
        "ip0 source": "i<sub>p</sub><sup>0</sup> Source",
        "ip0 scan rate": "i<sub>p</sub><sup>0</sup> Scan Rate",
        "ip0 sqrt scan rate slope": "i<sub>p</sub><sup>0</sup> sqrt(scan rate) slope",
        "ip0 tangent": "i<sub>p</sub><sup>0</sup> Tangent",
        "ilim": "i<sub>lim</sub>",
        "ilim/ip0": "i<sub>lim</sub>/i<sub>p</sub><sup>0</sup>",
        "ilim source": "i<sub>lim</sub> Source",
        "ilim tangent": "i<sub>lim</sub> Tangent",
        "reference ep": "Reference E<sub>p</sub>",
        "redox potential": "Redox Potential",
        "redox delta e": "Redox Delta E",
        "catalytic ecat/2": "Catalytic E<sub>cat/2</sub>",
        "ecat/2 - e1/2": "E<sub>cat/2</sub> - E<sub>1/2</sub>",
        "r2": "R<sup>2</sup>",
        "fowa fit": "FOWA Fit",
    }
    return header_html_map.get(key, column)


def _pretty_unit_html_label(unit):
    text = str(unit).strip()
    replacements = {
        "⁻¹": "<sup>-1</sup>",
        "¹ᐟ²": "<sup>1/2</sup>",
        "²": "<sup>2</sup>",
        "³": "<sup>3</sup>",
        "^-1": "<sup>-1</sup>",
        "^1/2": "<sup>1/2</sup>",
        "^2": "<sup>2</sup>",
        "^3": "<sup>3</sup>",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def _coerce_display_columns(value):
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple, set, pd.Index, np.ndarray)):
        return list(value)
    return []


def _object_table_column_key(value):
    return re.sub(r"[^a-z0-9]+", "", str(value).strip().lower())


def _object_table_column_mode(value):
    if not isinstance(value, str):
        return None
    mode = value.strip().lower().replace("_", " ").replace("-", " ")
    return " ".join(mode.split())


def _object_table_column_aliases(available_columns):
    aliases = {}
    for column in available_columns:
        for alias in (
            column,
            column.replace(" ", "_"),
            pretty_table_column_label(column),
            pretty_table_column_label(column).replace(" ", "_"),
        ):
            key = _object_table_column_key(alias)
            if key:
                aliases.setdefault(key, column)
    chrono_aliases = {
        "init potential v": "applied potential",
        "init potential": "applied potential",
        "run time s": "run time",
        "sample interval s": "sample interval",
    }
    for alias, column in chrono_aliases.items():
        if column in available_columns:
            aliases.setdefault(_object_table_column_key(alias), column)
    return aliases


def _suggest_object_table_column(requested, available_columns):
    requested_key = _object_table_column_key(requested)
    if not requested_key:
        return None

    best_column = None
    best_score = 0.0
    for column in available_columns:
        candidates = {column, pretty_table_column_label(column)}
        for candidate in candidates:
            candidate_key = _object_table_column_key(candidate)
            if not candidate_key:
                continue
            score = difflib.SequenceMatcher(None, requested_key, candidate_key).ratio()
            if candidate_key in requested_key or requested_key in candidate_key:
                score = max(score, 0.86)
            if score > best_score:
                best_score = score
                best_column = column

    if best_score >= 0.55:
        return best_column
    return None


def _resolve_object_table_columns(requested_columns, available_columns):
    aliases = _object_table_column_aliases(available_columns)
    resolved = []
    invalid = []

    for requested in requested_columns:
        column = aliases.get(_object_table_column_key(requested))
        if column is None:
            invalid.append(str(requested))
            continue
        resolved.append(column)

    if invalid:
        messages = []
        for requested in invalid:
            suggestion = _suggest_object_table_column(requested, available_columns)
            message = f'Unknown show_objects column: "{requested}"'
            if suggestion is not None:
                message += f'\nDid you mean: "{suggestion}"?'
            messages.append(message)
        messages.append("Available columns: " + ", ".join(available_columns))
        raise ValueError("\n".join(messages))

    return resolved


def _object_table_optional_metadata(obj):
    electrode_parts = []
    for label, attr in [
        ("WE", "working_electrode"),
        ("CE", "counter_electrode"),
        ("RE", "reference_electrode"),
    ]:
        value = getattr(obj, attr, None)
        if value not in (None, ""):
            electrode_parts.append(f"{label}: {value}")
    return {
        "type": getattr(obj, "type", type(obj).__name__),
        "temperature": getattr(obj, "temperature", None),
        "electrode area": getattr(obj, "electrode_area", None),
        "electrode": "; ".join(electrode_parts) if electrode_parts else None,
        "ir comp resistance": getattr(obj, "ir_comp_resistance", None),
        "ir uncomp resistance": getattr(obj, "ir_uncomp_resistance", None),
        "ir comp percent": getattr(obj, "ir_comp_percent", None),
        "creation time": getattr(obj, "creation_time", None),
        "timestamp": getattr(obj, "timestamp", None),
    }


def _object_has_reference_metadata(obj):
    return (
        getattr(obj, "reference_shift", None) is not None
        or getattr(obj, "reference_source_file", None) is not None
    )


def build_object_table(object_list, options=None):
    if options is None:
        options = {}

    group_number = options.get("group number")
    plot_labels = options.get("labels")
    print_conditions = options.get("print conditions", True)
    condition_keys = options.get("condition keys")
    if isinstance(condition_keys, str):
        condition_keys = [condition_keys]
    extra_conditions = options.get("extra conditions", {}) or {}

    def _index_label(i):
        if group_number is None:
            return f"[{i}]"
        return f"[{group_number}][{i}]"

    def is_blank(value):
        if value is None:
            return True
        if isinstance(value, float) and pd.isna(value):
            return True
        if isinstance(value, str) and value.strip() == "":
            return True
        return False

    def format_display_value(value):
        sig_figs = options.get("sig figs", 3)

        if isinstance(value, (np.integer, int)) and not isinstance(value, bool):
            return str(value)

        if isinstance(value, (np.floating, float)) and not isinstance(value, bool):
            if pd.isna(value):
                return ""
            return f"{round_sigfigs(float(value), sig_figs):g}"

        if isinstance(value, (list, tuple, set)):
            if len(value) == 0:
                return ""
            return "[" + ", ".join(format_display_value(v) for v in value) + "]"

        if isinstance(value, dict):
            if len(value) == 0:
                return ""
            return ", ".join(
                f"{k}: {format_display_value(v)}" for k, v in value.items()
            )

        return str(value)

    def normalize_value(key, value):
        if value is None:
            return ""
        if isinstance(value, float) and pd.isna(value):
            return ""

        if key.lower() == "reference shift":
            rounded = round_sigfigs(float(value), options.get("sig figs", 3))
            return f"{rounded:g}"

        if isinstance(value, (np.integer, int, np.floating, float)) and not isinstance(value, bool):
            return format_display_value(value)

        if isinstance(value, (list, tuple, set)):
            if key.lower() == "compounds":
                return ", ".join(
                    str(v) for v in value
                    if not is_blank(v)
                )
            return format_display_value(value)

        if isinstance(value, dict):
            return format_display_value(value)

        return value

    def pretty_key_label(key):
        return pretty_table_column_label(key)

    def reference_key_label(key):
        key = str(key).strip()
        if key.lower().startswith("reference "):
            key = key[len("reference "):]
        return pretty_table_column_label(key)

    def best_time(obj):
        for attr in ("timestamp", "creation_time", "modification_time"):
            value = getattr(obj, attr, None)
            if isinstance(value, datetime):
                return value
        return None

    def format_condition_value(key, value):
        if is_blank(value):
            return None

        sig_figs = options.get("sig figs", 3)

        if key in {"ir comp resistance", "ir uncomp resistance"}:
            if isinstance(value, (np.integer, int, np.floating, float)):
                if pd.isna(value):
                    return None
                return f"{round_sigfigs(float(value), sig_figs):g} ohm"
            if isinstance(value, str) and value.strip():
                return value if any(token in value.lower() for token in ("ohm", "ω", "Ω".lower())) else f"{value} ohm"

        if key == "ir comp percent":
            if isinstance(value, (np.integer, int, np.floating, float)):
                if pd.isna(value):
                    return None
                return f"{round_sigfigs(float(value), sig_figs):g} %"
            if isinstance(value, str) and value.strip():
                return value if "%" in value else f"{value} %"

        if isinstance(value, (list, tuple, set)):
            if len(value) == 0:
                return None
            return ", ".join(
                format_chemical_formulas(v, mode="unicode")
                if isinstance(v, str)
                else format_display_value(v)
                for v in value
            )

        if isinstance(value, str) and key.lower() in {"gas", "solvent", "compounds"}:
            return format_chemical_formulas(value, mode="unicode")

        return str(value)

    def as_list(value):
        if value is None:
            return []
        if isinstance(value, list):
            return value.copy()
        if isinstance(value, tuple):
            return list(value)
        if isinstance(value, set):
            return list(value)
        if isinstance(value, str):
            return [value] if value.strip() != "" else []
        return [value]

    common_values, different_values, stats_rows = echem_similar_different(
        object_list,
        options=options,
        return_values=True,
    )
    txt_stats_columns = []
    for stats in stats_rows:
        for key in stats.keys():
            if key not in txt_stats_columns:
                txt_stats_columns.append(key)

    shared_compounds = as_list(common_values.get("compounds", []))
    include_reference_columns = any(
        _object_has_reference_metadata(obj) for obj in object_list
    )

    rows = []
    for i, (echem_object, stats) in enumerate(zip(object_list, stats_rows)):
        stats = stats.copy()

        # Remove shared compounds from the per-row compounds display
        row_compounds = as_list(stats.get("compounds", []))
        if shared_compounds and row_compounds:
            row_compounds = [c for c in row_compounds if c not in shared_compounds]
            stats["compounds"] = row_compounds

        row = {
            "__row_index__": i,
            "__sort_time__": best_time(echem_object),
            "name": echem_object.name,
            "subfolder": _display_folder_path_value(getattr(echem_object, "folderpath", ".")),
        }

        if plot_labels is not None:
            row["plot label"] = plot_labels[i]

        if include_reference_columns:
            row["reference shift"] = getattr(echem_object, "reference_shift", None)
            row["reference label"] = getattr(echem_object, "reference_label", None)
            row["reference mode"] = getattr(echem_object, "reference_mode", None)
            row["reference source"] = _reference_source_display(echem_object, object_list)
        row.update(_object_table_optional_metadata(echem_object))
        if getattr(echem_object, "type", None) == "Chronoamperometry":
            stats = {
                k: v for k, v in stats.items()
                if k not in {"Min Current (A)", "Max Current (A)", "Avg Current (A)"}
            }
        row.update(stats)
        row = {k: normalize_value(k, v) for k, v in row.items()}
        rows.append(row)

    df = pd.DataFrame(rows)

    if "subfolder" in df.columns:
        df["subfolder"] = df["subfolder"].replace(".", "").fillna("")

    # Available display columns
    available_columns = [c for c in df.columns if not c.startswith("__")]

    # Parse requested additive columns
    columns_option = options.get("columns", [])
    columns_mode = _object_table_column_mode(columns_option)
    if columns_mode == "available":
        return available_columns
    columns_all = columns_mode == "all"
    if columns_all:
        requested_columns = available_columns.copy()
    else:
        requested_columns = _resolve_object_table_columns(
            _coerce_display_columns(columns_option),
            available_columns,
        )

    requested_reference_columns = _requested_reference_columns(options)

    shared_reference_values = {}
    for col in requested_reference_columns:
        if col not in df.columns:
            continue

        nonblank = [v for v in df[col] if not is_blank(v)]
        if len(nonblank) == 0:
            continue

        first = nonblank[0]
        if all(v == first for v in nonblank) and len(nonblank) == len(df):
            shared_reference_values[col] = first

    inline_reference_columns = [
        col for col in requested_reference_columns
        if col not in shared_reference_values
    ]

    if print_conditions:
        condition_order = [
            "exp type",
            "solvent",
            "gas",
            "compounds",
            "scan window",
            "scan rate",
            "segments",
        ]
        ordered_common_keys = [k for k in condition_order if k in common_values] + [
            k for k in common_values if k not in condition_order
        ]

        requested_condition_columns = set(requested_columns if not columns_all else available_columns)
        ir_condition_keys = {"ir comp resistance", "ir uncomp resistance"}
        ordered_common_keys = [
            key for key in ordered_common_keys
            if key not in ir_condition_keys or key in requested_condition_columns
        ]

        if condition_keys is not None:
            ordered_common_keys = [k for k in ordered_common_keys if k in condition_keys]

        folder_values = [getattr(obj, "folderpath", ".") for obj in object_list]
        unique_folders = list(dict.fromkeys(folder_values))

        if len(unique_folders) == 1 and unique_folders[0] not in (".", "", None):
            folder_display = _format_path_for_display(unique_folders[0])
            print(f"[Folder] `{folder_display}`")

        common_parts = []
        for key in ordered_common_keys:
            value = common_values[key]
            formatted = format_condition_value(key, value)
            if formatted is not None:
                common_parts.append(f"{pretty_key_label(key)}: {formatted}")

        for key, value in extra_conditions.items():
            if key in common_values:
                continue
            formatted = format_condition_value(key, value)
            if formatted is not None:
                common_parts.append(f"{pretty_key_label(key)}: {formatted}")

        shared_optional_condition_keys = ["ir comp percent"]
        for key in ("ir comp resistance", "ir uncomp resistance"):
            if key in requested_condition_columns:
                shared_optional_condition_keys.append(key)

        for key in shared_optional_condition_keys:
            if key in common_values or key in extra_conditions or key not in df.columns:
                continue
            nonblank = [v for v in df[key] if not is_blank(v)]
            if len(nonblank) != len(df) or len(nonblank) == 0:
                continue
            first = nonblank[0]
            if not all(v == first for v in nonblank):
                continue
            formatted = format_condition_value(key, first)
            if formatted is not None:
                common_parts.append(f"{pretty_key_label(key)}: {formatted}")

        if common_parts:
            print("[Conditions] " + ", ".join(common_parts))

        shared_reference_values = {}
        for col in requested_reference_columns:
            if col not in df.columns:
                continue

            nonblank = [v for v in df[col] if not is_blank(v)]
            if len(nonblank) == 0:
                continue

            first = nonblank[0]
            if all(v == first for v in nonblank) and len(nonblank) == len(df):
                shared_reference_values[col] = first

        inline_reference_columns = [
            col for col in requested_reference_columns
            if col not in shared_reference_values
        ]

        if requested_reference_columns and shared_reference_values:
            reference_parts = []
            for key in [
                "reference shift",
                "reference label",
                "reference mode",
                "reference source",
            ]:
                if key in shared_reference_values:
                    reference_parts.append(
                        f"{reference_key_label(key)}: {shared_reference_values[key]}"
                    )

            if reference_parts:
                print("[Reference] " + ", ".join(reference_parts))

    # Automatic columns: only those that differ
    auto_columns = []

    # subfolder only if it differs
    if "subfolder" in df.columns and df["subfolder"].nunique(dropna=False) > 1:
        auto_columns.append("subfolder")

    # txt_stats columns that differ
    auto_excluded_columns = {
        "ir comp resistance",
        "ir uncomp resistance",
        "ir comp percent",
        "electrode",
    }
    for key in different_values.keys():
        if key in auto_excluded_columns:
            continue
        if key in df.columns and key not in auto_columns:
            auto_columns.append(key)

    # User columns are additive, except shared requested reference columns,
    # which are printed above as a single "Reference:" line.
    requested_columns_for_table = [
        col for col in requested_columns
        if col not in shared_reference_values
    ]

    if columns_all:
        visible_columns = available_columns.copy()
    else:
        visible_columns = auto_columns.copy()
        for col in requested_columns_for_table:
            if col not in visible_columns:
                visible_columns.append(col)

    if plot_labels is not None and "plot label" in df.columns:
        if "plot label" not in visible_columns:
            visible_columns.insert(0, "plot label")

    # Drop columns that are still blank across all rows
    if columns_all:
        visible_columns = [col for col in visible_columns if col in df.columns]
    else:
        requested_blank_ok = set(requested_columns_for_table)
        visible_columns = [
            col for col in visible_columns
            if col in df.columns
            and (
                col in requested_blank_ok
                or not all(is_blank(v) for v in df[col])
            )
        ]

    # Add Rep if visible rows are not unique. Reference metadata is intentionally
    # excluded so small reference-shift/source differences do not hide replicates.
    reference_columns = {
        "reference shift",
        "reference label",
        "reference mode",
        "reference source",
    }
    compare_cols = [
        c for c in visible_columns
        if c in df.columns and c != "name" and c not in reference_columns
    ]

    if len(compare_cols) == 0:
        duplicate_mask = pd.Series([len(df) > 1] * len(df), index=df.index)
    else:
        duplicate_mask = df[compare_cols].duplicated(keep=False)

    if duplicate_mask.any():
        df["replicate"] = ""

        if len(compare_cols) == 0:
            grouped_indices = [list(df.index)]
        else:
            groupby_key = compare_cols[0] if len(compare_cols) == 1 else compare_cols
            grouped_indices = [
                list(idx)
                for _, idx in df.loc[duplicate_mask].groupby(groupby_key, dropna=False, sort=False).groups.items()
            ]

        for idx_group in grouped_indices:
            idx_sorted = sorted(
                idx_group,
                key=lambda j: (
                    pd.Timestamp.max if df.at[j, "__sort_time__"] is None else pd.Timestamp(df.at[j, "__sort_time__"]),
                    df.at[j, "__row_index__"],
                )
            )
            for rep_num, j in enumerate(idx_sorted, start=1):
                df.at[j, "replicate"] = str(rep_num)

        if "replicate" not in visible_columns:
            visible_columns.insert(0, "replicate")

    # Add name only if requested explicitly or still needed after Rep
    if "name" in requested_columns and "name" in df.columns:
        if "name" not in visible_columns:
            visible_columns.insert(0, "name")
    else:
        compare_without_name = [c for c in visible_columns if c != "name"]
        if len(compare_without_name) == 0:
            keep_name = True
        else:
            keep_name = df[compare_without_name].duplicated(keep=False).any()

        if keep_name and "name" in df.columns and "name" not in visible_columns:
            visible_columns.insert(0, "name")

    # If nothing remains, just return empty display df after Conditions
    if len(visible_columns) == 0:
        empty_df = pd.DataFrame(index=[_index_label(i) for i in range(len(object_list))])
        meta = {"inline_reference_columns": inline_reference_columns}
        return empty_df, meta

    preferred_stats_order = [
        "exp type",
        "solvent",
        "gas",
        "compounds",
        "scan window",
        "scan rate",
        "segments",
    ]
    ordered_txt_stats_columns = [
        col for col in preferred_stats_order
        if col in txt_stats_columns
    ] + [
        col for col in txt_stats_columns
        if col not in preferred_stats_order
    ]

    preferred_order = [
        "plot label",
        "subfolder",
        "name",
        *ordered_txt_stats_columns,
        "replicate",
        'reference shift',
        'reference label',
        'reference mode',
        'reference source',

    ]
    visible_columns = [c for c in preferred_order if c in visible_columns] + [
        c for c in visible_columns if c not in preferred_order
    ]

    display_df = df[visible_columns].copy()
    display_df = display_df.fillna("")
    display_df.index = [_index_label(i) for i in range(len(display_df))]
    display_df = display_df.rename(columns={c: pretty_key_label(c) for c in display_df.columns})

    meta = {"inline_reference_columns": inline_reference_columns}
    return display_df, meta

def display_object_table(display_df, options=None, *, title=None, plain_title=True):
    if options is None:
        options = {}

    col_lookup = {str(c).lower(): c for c in display_df.columns}
    html_formula_columns = [
        col_lookup[name]
        for name in ("plot label", "gas", "solvent", "compounds", "reference label", "electrode")
        if name in col_lookup
    ]

    def _html_formula(value):
        if value is None:
            return ""
        if isinstance(value, float) and pd.isna(value):
            return ""
        if isinstance(value, str) and value.strip() == "":
            return ""
        if isinstance(value, (list, tuple, set)):
            return ", ".join(
                format_chemical_formulas(str(v), mode="html")
                for v in value
                if v not in (None, "")
            )
        return format_chemical_formulas(str(value), mode="html")

    def _html_value(value):
        if value is None:
            return ""
        if isinstance(value, float) and pd.isna(value):
            return ""
        if isinstance(value, str) and value.strip() == "":
            return ""
        if isinstance(value, (list, tuple, set)):
            return "[" + ", ".join(str(v) for v in value) + "]"
        return str(value)

    display_df_html = display_df.copy()
    metric_col = None
    value_col = None
    metric_lookup = {str(c).lower(): c for c in display_df_html.columns}
    if "metric" in metric_lookup and "value" in metric_lookup:
        metric_col = metric_lookup["metric"]
        value_col = metric_lookup["value"]
        for idx, metric in display_df_html[metric_col].items():
            metric_key = str(metric).strip().lower()
            if metric_key in {"plot label", "gas", "solvent", "compounds", "reference label", "electrode"}:
                display_df_html.at[idx, value_col] = _html_formula(
                    display_df_html.at[idx, value_col]
                )

    display_df_html = display_df_html.rename(
        columns=_pretty_table_header_html_label
    )

    # update html-formatted value columns after renaming
    renamed_formula_columns = [
        _pretty_table_header_html_label(col)
        for col in html_formula_columns
    ]
    if value_col is not None:
        renamed_value_col = _pretty_table_header_html_label(value_col)
    else:
        renamed_value_col = None

    formatters = {col: _html_formula for col in renamed_formula_columns}
    if renamed_value_col is not None:
        formatters[renamed_value_col] = _html_value

    return _display_table(
        display_df,
        options,
        title=title,
        rich_table=display_df_html,
        formatters=formatters,
        escape=None,
        plain_title=plain_title,
    )
    
def show_objects(object_list, options=None):
    """
    Show a list of echem objects.

    Behavior
    --------
    - Plain print by default.
    - Pretty table if options['pretty print'] is True, or if pretty_print=True
      is passed explicitly.
    - User-specified `columns` are additive: they can add columns, but cannot
      remove columns needed to distinguish entries.
    """
    if options is None:
        options = {}

    group_number = options.get("group number")
    plot_labels = options.get("labels")
    pretty_print = options.get("pretty print", True)

    def _index_label(i):
        if group_number is None:
            return f"[{i}]"
        return f"[{group_number}][{i}]"

    def _show_objects_plain(objects, inline_reference_columns=None):
        if inline_reference_columns is None:
            inline_reference_columns = []

        for i, echem_object in enumerate(objects):
            subfolder = _display_folder_path_value(getattr(echem_object, "folderpath", "."))
            prefix = f"`{subfolder}` / " if subfolder else ""
            ref_text = _reference_inline_text(
                echem_object,
                objects,
                inline_reference_columns=inline_reference_columns,
            )
            plot_label = plot_labels[i] if plot_labels is not None else None
            if plot_label not in (None, "", echem_object.name):
                plot_label = format_chemical_formulas(plot_label, mode="unicode")
                print(f"{_index_label(i)}\t{plot_label}: {prefix}{echem_object.name}{ref_text}")
            else:
                print(f"{_index_label(i)}\t{prefix}{echem_object.name}{ref_text}")

    result = build_object_table(
        object_list,
        options=options
    )
    if isinstance(result, list):
        return result

    display_df, meta = result
    return_table = bool(options.get("return", False))

    if not pretty_print:
        _show_objects_plain(
            object_list,
            inline_reference_columns=meta.get("inline_reference_columns", []),
        )
        return display_df if return_table else None

    display_object_table(display_df)
    return display_df if return_table else None



def echem_similar_different(echem_list, options=None, ignore=None, return_values=False):
    if options is None:
        options = {}
    if ignore is None:
        ignore = []

    if not echem_list:
        return ({}, {}, []) if return_values else ([], [])

    def is_blank(value):
        if value is None:
            return True
        if isinstance(value, float) and pd.isna(value):
            return True
        if isinstance(value, str) and value.strip() == "":
            return True
        if isinstance(value, (list, tuple, set, dict)) and len(value) == 0:
            return True
        return False

    def normalize_compare_value(value):
        if is_blank(value):
            return ""
        if isinstance(value, dict):
            return tuple((k, normalize_compare_value(v)) for k, v in sorted(value.items()))
        if isinstance(value, (list, tuple, set)):
            return tuple(normalize_compare_value(v) for v in value)
        return value

    def as_list(value):
        if is_blank(value):
            return []
        if isinstance(value, list):
            return value.copy()
        if isinstance(value, tuple):
            return list(value)
        if isinstance(value, set):
            return list(value)
        return [value]

    stats_rows = []
    all_keys = []

    for obj in echem_list:
        stats = obj.txt_stats(options).copy()
        stats_rows.append(stats)
        for key in stats:
            if key not in all_keys:
                all_keys.append(key)

    common_values = {}
    different_values = {}

    for key in all_keys:
        if key in ignore:
            continue

        values = [row.get(key, "") for row in stats_rows]

        # skip keys that are blank everywhere
        if all(is_blank(v) for v in values):
            continue

        if key == "compounds":
            compound_lists = [as_list(v) for v in values]

            # shared compounds = ordered intersection, preserving first row order
            shared_compounds = []
            if compound_lists:
                for compound in compound_lists[0]:
                    if all(compound in lst for lst in compound_lists[1:]):
                        shared_compounds.append(compound)

            if len(shared_compounds) > 0:
                common_values[key] = shared_compounds

            # compounds still count as different if the full lists are not all identical
            normalized = [normalize_compare_value(v) for v in compound_lists]
            first = normalized[0]
            if not all(v == first for v in normalized):
                different_values[key] = values

            continue

        normalized = [normalize_compare_value(v) for v in values]
        first = normalized[0]

        if all(v == first for v in normalized):
            common_values[key] = values[0]
        else:
            different_values[key] = values

    if return_values:
        return common_values, different_values, stats_rows

    return list(common_values.keys()), list(different_values.keys())

### Package Functions ###
###===================###

def multimultiplot(echem_groups,options=None):
    # configure options
    """Plot multiple groups of electrochemistry objects as coordinated multiplots.
    
    Parameters
    ----------
    echem_groups : sequence of sequences of echem
        Groups of objects to plot.
    options : dict or MultiMultiplotOptions, optional
        Shared multiplot and subplot options. See ``e.describe_options("multimultiplot")``.
    
    Returns
    -------
    matplotlib.figure.Figure
        Figure containing the grouped plots.
    
    Examples
    --------
    >>> e.multimultiplot(groups, {"legend mode": "discrete"})
    """
    options = MultiMultiplotOptions.from_options(options).to_options_dict()

    groups = []
    two_dim = False
    if type(echem_groups[0]) != list:
        for item in echem_groups:
            groups.append([item])
    else:
        groups = echem_groups
        two_dim = True
    for n in range(len(groups)):
        titles = options.get('titles',"auto")
        if titles == "auto":
            if two_dim:
                title = "Group " + str(n) + ": " + "GENERATE"
            else:
                title = groups[n][0].name
        else:
            title = titles[n]
        plot_options = _multiplot_options_from_mapping(options)
        plot_options['title'] = title
        plot_options['subtitle'] = options.get('subtitles',"auto")
        multiplot(groups[n],options=plot_options)
        if options.get('analysis',False) and options.get('Ehalf',None) != None:
            for cv in groups[n]:
                cv.peak_current(E_half,options=options)

def _axis_common_unit(echem_list, array_name_getter, selected_opt='auto'):
    """
    Determine a common unit for a set of arrays and their axis names.

    array_name_getter: fn(e) -> (data_array, axis_name_str)
    selected_opt: 'auto', explicit unit string, or None to skip scaling.

    Returns a unit string (e.g. 'mA', 'min', 'V') or None to skip auto-scaling.
    """
    from .utils import (
        extract_prefix_and_base,
        get_conversion_factor,
        scale_axis,
        scale_time_axis,
    )

    # extract arrays and names
    pairs = [array_name_getter(e) for e in echem_list]
    arrays = [p[0] for p in pairs]
    names  = [p[1] for p in pairs]

    # explicit override: do nothing
    if selected_opt is not None and selected_opt != 'auto' and selected_opt != 'auto':
        return selected_opt

    # must share the same axis name to auto-scale
    if not all(n == names[0] for n in names):
        print(f"\033[91mWarning: Axis names differ ({set(names)}); skipping auto-scaling.\033[0m")
        return None

    name = names[0]
    # get raw unit string
    raw_unit = echem_list[0].units.get(name, '')
    if raw_unit in (None, ""):
        return ""
    prefix0, base = extract_prefix_and_base(raw_unit)

    # convert all to base units
    base_arrays = []
    for (arr, _name), e in zip(pairs, echem_list):
        pref, b = extract_prefix_and_base(e.units.get(_name, ''))
        factor = get_conversion_factor(pref + b)
        base_arrays.append(arr * factor)
    all_base = np.concatenate(base_arrays)

    # Potential: no scaling
    if 'potential' in name.lower():
        return base

    # Time: use time scaling
    if name.lower() in ('time', 't', 'duration'):
        _, unit = scale_time_axis(all_base, base, selected_unit='auto')
        return unit

    # Generic numerical: use SI scaling
    _, unit = scale_axis(all_base, base, selected_unit='auto')
    return unit

def _to_hashable(value):
    if isinstance(value, dict):
        return tuple((k, _to_hashable(v)) for k, v in sorted(value.items()))
    if isinstance(value, (list, tuple)):
        return tuple(_to_hashable(v) for v in value)
    if isinstance(value, set):
        return tuple(sorted(_to_hashable(v) for v in value))
    return value


def _as_list(value):
    if value is None:
        return []
    if isinstance(value, list):
        return value.copy()
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, set):
        return list(value)
    if isinstance(value, str):
        return [value] if value.strip() else []
    return [value]


def _coerce_gradient_color_sets(value):
    """
    Accept either:
        []
        ['navy', 'gold']
        [['navy', 'skyblue'], ['darkred', 'orange']]
    and normalize to list[list[str]].
    """
    if not value:
        return []

    if isinstance(value, (list, tuple)):
        if len(value) == 0:
            return []
        first = value[0]
        if isinstance(first, (list, tuple)):
            return [list(v) for v in value]
        return [list(value)]

    raise ValueError("'gradient colors' must be a list or list of lists.")


def _sample_hex_colors_from_cmap(cmap, n):
    if n <= 0:
        return []
    xs = np.linspace(0, 1, n)
    return [mpl.colors.to_hex(cmap(x)) for x in xs]


_DEFAULT_TRACE_COLORS = [
    "black",
    "tab:blue",
    "tab:red",
    "tab:green",
    "tab:orange",
    "tab:purple",
]


def _get_discrete_colors(n, options):
    colors = list(options.get("colors", []) or [])
    fallback_cmap_name = options.get("default discrete colormap", "tab20")

    if len(colors) == 0:
        colors = _DEFAULT_TRACE_COLORS.copy()

    if n <= len(colors):
        return colors[:n]

    needed = n - len(colors)
    fallback_cmap = mpl.colormaps.get_cmap(fallback_cmap_name)
    colors.extend(_sample_hex_colors_from_cmap(fallback_cmap, needed))
    return colors[:n]


def _truncate_cmap(cmap, start=0.0, stop=1.0, n=256):
    xs = np.linspace(start, stop, n)
    colors = cmap(xs)
    return mpl.colors.LinearSegmentedColormap.from_list(
        f"{getattr(cmap, 'name', 'cmap')}_{start:.3f}_{stop:.3f}",
        colors,
        N=n,
    )

def _coerce_gradient_cmap_names(value):
    if not value:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple)):
        return list(value)
    raise ValueError("'gradient colormaps' must be a string or list of strings.")

def _get_group_cmap(group_idx, n_groups, options):
    """
    Priority:
    1. per-group custom palette from 'gradient colors'
    2. one shared custom palette reused for every group
    3. split the default gradient colormap into equal subranges
    """
    gamma = options.get("gradient gamma", 1.0)
    reverse = options.get("gradient reverse", False)

    color_sets = _coerce_gradient_color_sets(options.get("gradient colors", []))
    cmap_names = _coerce_gradient_cmap_names(options.get("gradient colormaps", []))
    cmap_name = options.get("gradient colormap") or options.get("default gradient colormap", "viridis")

    if len(color_sets) >= n_groups and n_groups > 0:
        cmap = mpl.colors.LinearSegmentedColormap.from_list(
            f"gradient_group_{group_idx}",
            color_sets[group_idx],
            N=256,
            gamma=gamma,
        )
    elif len(color_sets) == 1:
        cmap = mpl.colors.LinearSegmentedColormap.from_list(
            "gradient_shared",
            color_sets[0],
            N=256,
            gamma=gamma,
        )
    elif len(cmap_names) >= n_groups and n_groups > 0:
        cmap = mpl.colormaps.get_cmap(cmap_names[group_idx])
    elif len(cmap_names) == 1:
        cmap = mpl.colormaps.get_cmap(cmap_names[0])
    else:
        base = mpl.colormaps.get_cmap(cmap_name)
        start = group_idx / max(n_groups, 1)
        stop = (group_idx + 1) / max(n_groups, 1)
        cmap = _truncate_cmap(base, start=start, stop=stop)

    if reverse:
        cmap = cmap.reversed()

    return cmap

def _resolve_gradient_scale(gradient_by, options):
    valid_scales = {"auto", "linear", "sqrt", "log", "index"}
    scale = options.get("gradient scale", "auto")

    if scale not in valid_scales:
        print(
            f"\033[91mWarning: Invalid 'gradient scale' value '{scale}'. "
            f"Valid options are: {sorted(valid_scales)}. "
            "Falling back to 'auto'.\033[0m"
        )
        scale = "auto"

    if scale != "auto":
        return scale

    if gradient_by == "scan rate":
        return "log"
    if gradient_by == "concentration":
        return "log"
    return "linear"

def _build_gradient_norm(values, gradient_by, options):
    values = np.asarray(values, dtype=float)
    scale = _resolve_gradient_scale(gradient_by, options)

    if len(values) == 0:
        return mpl.colors.Normalize(vmin=0, vmax=1), scale

    if scale == "index":
        vmax = max(len(values) - 1, 1)
        return mpl.colors.Normalize(vmin=0, vmax=vmax), scale

    vmin = float(np.nanmin(values))
    vmax = float(np.nanmax(values))

    if np.isclose(vmin, vmax):
        vmax = vmin + 1e-12

    if scale == "log":
        positive = values[values > 0]
        if len(positive) == 0 or len(positive) != len(values):
            return mpl.colors.Normalize(vmin=vmin, vmax=vmax), "linear"
        return mpl.colors.LogNorm(
            vmin=float(np.nanmin(positive)),
            vmax=float(np.nanmax(positive)),
        ), scale

    if scale == "sqrt":
        return mpl.colors.PowerNorm(gamma=0.5, vmin=vmin, vmax=vmax), scale

    return mpl.colors.Normalize(vmin=vmin, vmax=vmax), "linear"


def _format_gradient_tick_value(value, unit, raw_value=None):
    if raw_value not in (None, "") and np.isclose(float(value), 0.0):
        return str(raw_value).strip()
    scaled, scaled_unit = scale_value(float(value), unit, selected_unit="auto")
    if isinstance(scaled_unit, str) and scaled_unit.startswith("u"):
        scaled_unit = "μ" + scaled_unit[1:]
    return f"{scaled:g} {scaled_unit}".strip()


def _build_gradient_ticks(values, unit, options, tick_positions=None, raw_values=None):
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return [], [], [], []

    if tick_positions is None:
        tick_positions = list(values)
    else:
        tick_positions = list(np.asarray(tick_positions, dtype=float))
    raw_values = [None] * len(values) if raw_values is None else list(raw_values)

    tick_mode = options.get("colorbar tick labels", "endpoints")

    all_ticklabels = [""] * len(values)

    if tick_mode == "none":
        pass
    elif tick_mode == "all":
        all_ticklabels = [
            _format_gradient_tick_value(v, unit, raw_value=raw)
            for v, raw in zip(values, raw_values)
        ]
    else:
        all_ticklabels[0] = _format_gradient_tick_value(
            values[0],
            unit,
            raw_value=raw_values[0],
        )
        all_ticklabels[-1] = _format_gradient_tick_value(
            values[-1],
            unit,
            raw_value=raw_values[-1],
        )

    if len(tick_positions) == 1:
        endpoint_ticks = [tick_positions[0]]
        endpoint_ticklabels = [all_ticklabels[0]]
    else:
        endpoint_ticks = [tick_positions[0], tick_positions[-1]]
        endpoint_ticklabels = [all_ticklabels[0], all_ticklabels[-1]]

    return tick_positions, all_ticklabels, endpoint_ticks, endpoint_ticklabels


def _gradient_entry_count(group):
    values = np.asarray(group.get("values", []), dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return 0
    return len(np.unique(values))


def _detect_scan_rate_gradient_groups(echem_list, options):
    buckets = {}

    for idx, obj in enumerate(echem_list):
        if not isinstance(obj, cv):
            continue
        scan_rate = getattr(obj, "scan_rate", None)
        if scan_rate in (None, ""):
            continue

        stats = obj.txt_stats(options).copy()
        for key in ["scan rate", "scan window", "exp type"]:
            stats.pop(key, None)

        signature = _to_hashable(stats)
        buckets.setdefault(signature, []).append((idx, float(scan_rate)))

    groups = []
    for items in buckets.values():
        if len(items) < 2:
            continue

        indices = [i for i, _ in items]
        values = np.array([v for _, v in items], dtype=float)

        if len(np.unique(values)) < 2:
            continue

        order = np.argsort(values)
        indices = [indices[i] for i in order]
        values = values[order]

        common, _, _ = echem_similar_different(
            [echem_list[i] for i in indices],
            options=options,
            ignore=["scan rate", "scan window", "exp type"],
            return_values=True,
        )
        compounds = common.get("compounds", [])
        if isinstance(compounds, list) and len(compounds) > 0:
            title = ", ".join(compounds) + " scan rate (V/s)"
        else:
            title = "Scan rate (V/s)"

        groups.append({
            "indices": indices,
            "values": values,
            "gradient by": "scan rate",
            "gradient species": None,
            "legend title": "Scan rate",
            "legend unit": "V/s",
        })

    groups.sort(key=lambda g: min(g["indices"]))
    return groups

def _infer_concentration_legend_unit(concentration_strings):
    """
    Infer the base display unit for a concentration gradient.

    Uses the existing unit helpers:
    - molarity variants (nM, uM, μM, mM, M) -> "M"
    - percent -> "%"
    - equivalents -> "equiv"
    - mole fraction -> "x"

    Returns
    -------
    str
        Base unit to store in group["legend unit"].
    """
    units = []

    for conc in concentration_strings:
        if not isinstance(conc, str):
            continue

        conc = conc.strip()
        if not conc:
            continue

        # Get the trailing unit token only, e.g. "mM" from "10 mM"
        parts = conc.split()
        unit_token = parts[-1] if len(parts) > 0 else ""

        # Fallback for strings like "10mM" with no space
        if unit_token == conc:
            match = re.search(r'([A-Za-z%μu]+)$', conc)
            if match:
                unit_token = match.group(1)

        if not unit_token:
            continue

        prefix, base = extract_prefix_and_base(unit_token)

        if base == "M":
            units.append("M")
        elif unit_token in {"%", "equiv", "x"}:
            units.append(unit_token)
        elif base in {"%", "equiv", "x"}:
            units.append(base)

    unique_units = list(dict.fromkeys(units))

    if len(unique_units) == 1:
        return unique_units[0]

    # Mixed or unrecognized concentration styles: default to molarity base
    return "M"

def _concentration_entry_unit(concentration):
    try:
        _value, unit = parse_concentration_value_and_unit(str(concentration))
        return unit
    except Exception:
        return ""


def _object_concentration_gradient_entries(obj):
    entries = []
    occurrence_counts = {}

    normal_pairs = [
        (compound, concentration, False)
        for compound, concentration in zip(
            list(getattr(obj, "compounds", []) or []),
            list(getattr(obj, "concentrations", []) or []),
        )
    ]
    zero_pairs = [
        (compound, concentration, True)
        for compound, concentration in zip(
            list(getattr(obj, "zero_concentration_compounds", []) or []),
            list(getattr(obj, "zero_concentrations", []) or []),
        )
    ]

    for compound, concentration, is_zero_absent in normal_pairs + zero_pairs:
        try:
            value = float(concentration_to_float(concentration))
        except Exception:
            continue

        unit = _concentration_entry_unit(concentration)
        occurrence_key = (str(compound), unit)
        occurrence = occurrence_counts.get(occurrence_key, 0)
        occurrence_counts[occurrence_key] = occurrence + 1

        entries.append({
            "compound": compound,
            "concentration": concentration,
            "value": value,
            "unit": unit,
            "occurrence": occurrence,
            "target": (str(compound), unit, occurrence),
            "is_zero_absent": is_zero_absent,
        })

    return entries


def _detect_concentration_gradient_groups(echem_list, options):
    candidates = []
    entries_by_object = [
        _object_concentration_gradient_entries(obj)
        for obj in echem_list
    ]
    target_keys = []
    for entries in entries_by_object:
        for entry in entries:
            if entry.get("is_zero_absent") or entry.get("value", 0) <= 0:
                continue
            if entry["target"] not in target_keys:
                target_keys.append(entry["target"])

    for target in target_keys:
        buckets = {}

        for idx, (obj, entries) in enumerate(zip(echem_list, entries_by_object)):
            target_entries = [entry for entry in entries if entry["target"] == target]
            if not target_entries:
                continue
            entry = target_entries[0]
            if entry.get("is_zero_absent") or entry.get("value", 0) <= 0:
                continue

            stats = obj.txt_stats(options).copy()
            for key in ["compounds", "scan window", "exp type"]:
                stats.pop(key, None)

            remaining_pairs = []
            for other in entries:
                if other["target"] == target or other.get("is_zero_absent"):
                    continue
                remaining_pairs.append((other["compound"], other["concentration"]))

            signature = (
                target,
                entry["compound"],
                _to_hashable(stats),
                _to_hashable(remaining_pairs),
            )
            buckets.setdefault(signature, []).append((idx, entry["value"], entry["concentration"]))

        for signature, items in buckets.items():
            if len(items) < 2:
                continue

            indices = [i for i, _value, _concentration in items]
            values = np.array([value for _i, value, _concentration in items], dtype=float)
            raw_concentration_strings = [
                concentration for _i, _value, concentration in items
            ]

            if len(np.unique(values)) < 2:
                continue

            order = np.argsort(values)
            indices = [indices[i] for i in order]
            values = values[order]
            raw_concentration_strings = [
                raw_concentration_strings[i] for i in order
            ]

            species = signature[1]
            legend_unit = _infer_concentration_legend_unit(raw_concentration_strings)

            candidates.append({
                "indices": indices,
                "values": values,
                "raw concentration strings": raw_concentration_strings,
                "gradient by": "concentration",
                "gradient species": species,
                "legend title": species,
                "legend unit": legend_unit,
            })

    # Greedy non-overlapping selection, largest groups first
    candidates.sort(key=lambda g: (-len(g["indices"]), min(g["indices"])))
    selected = []
    used = set()

    for group in candidates:
        idx_set = set(group["indices"])
        if idx_set & used:
            continue
        selected.append(group)
        used |= idx_set

    selected.sort(key=lambda g: min(g["indices"]))
    return selected


def _detect_gradient_groups(echem_list, options):
    gradient_by = options.get("gradient by", "auto")

    if gradient_by == "scan rate":
        return _detect_scan_rate_gradient_groups(echem_list, options)

    if gradient_by == "concentration":
        return _detect_concentration_gradient_groups(echem_list, options)

    # auto: match peak-potential fit grouping logic
    scan_groups = _detect_scan_rate_gradient_groups(echem_list, options)
    concentration_groups = _detect_concentration_gradient_groups(echem_list, options)
    if len(scan_groups) > 0:
        used_indices = {
            idx
            for group in scan_groups
            for idx in group.get("indices", [])
        }
        combined = list(scan_groups)
        combined.extend(
            group
            for group in concentration_groups
            if not set(group.get("indices", [])) & used_indices
        )
        combined.sort(key=lambda g: min(g["indices"]))
        return combined

    return concentration_groups


def _resolve_multiplot_color_spec(echem_list, labels, options):
    n = len(echem_list)
    color_mode = options.get("color mode", "auto")
    legend_mode = _normalize_multiplot_legend_mode(options.get("legend mode", "auto"))
    min_gradient_entries = int(options.get("min gradient entries", 3))

    line_colors = [None] * n
    plot_labels = list(labels)

    if color_mode == "discrete":
        palette = _get_discrete_colors(n, options)
        return {
            "line colors": palette,
            "labels": plot_labels,
            "gradient groups": [],
            "discrete indices": list(range(n)),
        }

    detected_groups = _detect_gradient_groups(echem_list, options)

    if color_mode == "auto":
        gradient_groups = detected_groups
    elif color_mode == "gradient":
        gradient_groups = [{
            "indices": list(range(n)),
            "values": np.arange(n, dtype=float),
            "gradient by": "index",
            "gradient species": None,
            "legend title": "Series order",
        }]
    else:
        gradient_groups = []

    # Only groups with enough distinct gradient values get gradient treatment at all.
    eligible_gradient_groups = [
        g for g in gradient_groups
        if _gradient_entry_count(g) >= min_gradient_entries
    ]

    n_eligible = len(eligible_gradient_groups)

    assigned = set()
    legend_gradient_groups = []

    for g_idx, group in enumerate(eligible_gradient_groups):
        cmap = _get_group_cmap(g_idx, n_eligible, options)

        norm, resolved_scale = _build_gradient_norm(
            group["values"],
            group["gradient by"],
            options,
        )

        values = np.asarray(group["values"], dtype=float)
        if resolved_scale == "index":
            mapped_values = np.arange(len(values), dtype=float)
        else:
            mapped_values = values

        ticks, ticklabels, endpoint_ticks, endpoint_ticklabels = _build_gradient_ticks(
            group["values"],
            group.get("legend unit", ""),
            options,
            tick_positions=mapped_values,
            raw_values=group.get("raw concentration strings"),
        )

        group["cmap"] = cmap
        group["norm"] = norm
        group["resolved scale"] = resolved_scale
        group["ticks"] = ticks
        group["ticklabels"] = ticklabels
        group["endpoint ticks"] = endpoint_ticks
        group["endpoint ticklabels"] = endpoint_ticklabels

        for local_i, obj_idx in enumerate(group["indices"]):
            line_colors[obj_idx] = mpl.colors.to_hex(cmap(norm(mapped_values[local_i])))
            assigned.add(obj_idx)

        if legend_mode == "colorbar":
            legend_gradient_groups.append(group)
            for obj_idx in group["indices"]:
                plot_labels[obj_idx] = "_nolegend_"

    # Everything not assigned by an eligible gradient becomes discrete
    discrete_indices = [i for i in range(n) if i not in assigned]

    if len(discrete_indices) > 0:
        discrete_colors = _get_discrete_colors(len(discrete_indices), options)
        for color, idx in zip(discrete_colors, discrete_indices):
            line_colors[idx] = color

    for i in range(n):
        if line_colors[i] is None:
            line_colors[i] = "black"

    return {
        "line colors": line_colors,
        "labels": plot_labels,
        "gradient groups": legend_gradient_groups if legend_mode == "colorbar" else [],
        "discrete indices": discrete_indices,
    }

from matplotlib.transforms import Bbox, TransformedBbox

def _normalize_multiplot_legend_mode(legend_mode):
    """
    Treat 'auto' as the gradient-aware colorbar legend mode.
    """
    mode = str(legend_mode or "auto").strip().lower()
    if mode == "auto":
        return "colorbar"
    return mode

def _normalize_legend_loc(legend_loc):
    loc = str(legend_loc or "best").strip().lower()
    return "best" if loc == "auto" else loc

def _build_layout_cache(ax):
    fig = ax.figure
    fig.draw_without_rendering()
    renderer = fig.canvas.get_renderer()
    ax_bbox = ax.get_window_extent(renderer=renderer)

    return {
        "renderer": renderer,
        "ax_bbox": ax_bbox,
        "dpi": fig.dpi,
    }

def _legend_candidate_corners():
    """
    Order to try for 'best' inside placement.
    """
    return ["upper left", "upper right", "lower left", "lower right"]


def _line_overlap_score(ax, bbox_display, max_points_per_line=400):
    """
    Return a simple overlap score between plotted line data and a candidate legend bbox.
    Lower is better; 0 means no sampled points fell inside the box.
    """
    score = 0

    for line in ax.lines:
        xy = line.get_xydata()
        if xy is None or len(xy) == 0:
            continue

        step = max(1, len(xy) // max_points_per_line)
        pts = ax.transData.transform(xy[::step])

        inside = (
            (pts[:, 0] >= bbox_display.x0) &
            (pts[:, 0] <= bbox_display.x1) &
            (pts[:, 1] >= bbox_display.y0) &
            (pts[:, 1] <= bbox_display.y1)
        )
        score += int(np.count_nonzero(inside))

    return score


def _panel_bbox_display(ax, loc, panel_width, panel_height, outside=False, pad=0.02):
    """
    Build the display-coordinate bbox for a custom legend panel.
    """
    x0, y0 = _resolve_legend_panel_position(
        loc,
        panel_width,
        panel_height,
        outside=outside,
        pad=pad,
    )
    bbox_axes = Bbox.from_bounds(x0, y0, panel_width, panel_height)
    return TransformedBbox(bbox_axes, ax.transAxes)


def _choose_best_custom_panel_loc(ax, panel_width, panel_height, pad=0.02):
    """
    For custom colorbar legends:
    - try inside corners
    - if one has zero overlap, use it
    - otherwise fall back to outside upper right
    """
    best_loc = None
    best_score = None

    for loc in _legend_candidate_corners():
        bbox_display = _panel_bbox_display(
            ax,
            loc,
            panel_width,
            panel_height,
            outside=False,
            pad=pad,
        )
        score = _line_overlap_score(ax, bbox_display)

        if score == 0:
            return loc, False

        if best_score is None or score < best_score:
            best_score = score
            best_loc = loc

    return "upper right", True

def _resolve_legend_panel_position(loc, panel_width, panel_height, outside=False, pad=0.02):
    """
    Resolve custom legend-panel position in axes coordinates.

    Parameters
    ----------
    loc : str
        Matplotlib-like legend location string.
    panel_width : float
    panel_height : float
    outside : bool
        If True, place just outside the axes. If False, place inside.
    pad : float
        Padding in axes coordinates.
    """
    loc = (loc or "upper right").lower().strip()

    if outside:
        mapping = {
            "upper right":  (1 + pad, 1 - pad - panel_height),
            "center right": (1 + pad, 0.5 - panel_height / 2),
            "lower right":  (1 + pad, pad),

            "upper left":   (-panel_width - pad, 1 - pad - panel_height),
            "center left":  (-panel_width - pad, 0.5 - panel_height / 2),
            "lower left":   (-panel_width - pad, pad),

            "upper center": (0.5 - panel_width / 2, 1 - pad - panel_height),
            "lower center": (0.5 - panel_width / 2, pad),
            "center":       (0.5 - panel_width / 2, 0.5 - panel_height / 2),
        }
        return mapping.get(loc, mapping["upper right"])

    upper_left_y = 1 - pad - panel_height
    if (
        _active_plot_style_value("axis labels") == "inside"
        and loc == "upper left"
    ):
        upper_left_y = min(upper_left_y, 0.86 - panel_height)

    mapping = {
        "upper right":  (1 - pad - panel_width, 1 - pad - panel_height),
        "center right": (1 - pad - panel_width, 0.5 - panel_height / 2),
        "lower right":  (1 - pad - panel_width, pad),

        "upper left":   (pad, upper_left_y),
        "center left":  (pad, 0.5 - panel_height / 2),
        "lower left":   (pad, pad),

        "upper center": (0.5 - panel_width / 2, 1 - pad - panel_height),
        "lower center": (0.5 - panel_width / 2, pad),
        "center":       (0.5 - panel_width / 2, 0.5 - panel_height / 2),
    }

    return mapping.get(loc, mapping["upper right"])


def _resolve_outside_matplotlib_legend_anchor(loc, pad=0.02):
    """
    Translate eCAT's outside legend locations into matplotlib loc/anchor pairs.
    """
    loc = (loc or "upper right").lower().strip()
    mapping = {
        "upper right": ("upper left", (1 + pad, 1)),
        "center right": ("center left", (1 + pad, 0.5)),
        "lower right": ("lower left", (1 + pad, 0)),

        "upper left": ("upper right", (-pad, 1)),
        "center left": ("center right", (-pad, 0.5)),
        "lower left": ("lower right", (-pad, 0)),

        "upper center": ("lower center", (0.5, 1 + pad)),
        "lower center": ("upper center", (0.5, -pad)),
        "center": ("center left", (1 + pad, 0.5)),
        "best": ("upper left", (1 + pad, 1)),
    }
    return mapping.get(loc, mapping["upper right"])


def _inside_label_legend_anchor(loc, pad=0.02):
    if _active_plot_style_value("axis labels") != "inside":
        return None
    loc = _normalize_legend_loc(loc)
    if loc == "upper left":
        return (pad, 0.86)
    return None

def _measure_text_width_axes(ax, text, fontsize):
    """
    Measure text width in axes-coordinate units.
    """
    if text in (None, ""):
        return 0.0

    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    # For multi-line text, width is the max line width
    lines = str(text).splitlines()
    max_px = 0.0

    for line in lines:
        temp = fig.text(0, 0, line, fontsize=fontsize, alpha=0.0)
        try:
            bbox = temp.get_window_extent(renderer=renderer)
            max_px = max(max_px, bbox.width)
        finally:
            temp.remove()

    ax_bbox = ax.get_window_extent(renderer=renderer)
    if ax_bbox.width == 0:
        return 0.0

    return max_px / ax_bbox.width


def _legend_sample_length_axes(ax, options, legend_fs, layout_cache=None):
    sample_length = options.get("legend sample length", "auto")

    if sample_length is None or str(sample_length).strip().lower() == "auto":
        return _points_to_axes_x(
            ax,
            legend_fs * float(mpl.rcParams.get("legend.handlelength", 2.0)),
            layout_cache=layout_cache,
        )

    return float(sample_length)


def _show_gradient_context_line(
    context_line,
    prev_line_label=None,
    previous_entry_type=None,
    *,
    full_context_line=None,
    gradient_species=None,
):
    if context_line in ("", None):
        return False
    if previous_entry_type != "line" or prev_line_label in ("", None):
        return True
    return not _gradient_context_redundant_with_previous_line(
        context_line,
        prev_line_label,
        full_context_line=full_context_line,
        gradient_species=gradient_species,
    )


def _plain_label_parts_for_context_compare(label):
    parts = [
        _plain_formula_text(part).strip().lower()
        for part in _split_label_parts(label)
    ]
    return [part for part in parts if part]


def _plain_disambiguator_parts_for_context_compare(label):
    parts = [
        _plain_formula_text(part).strip().lower()
        for part in _split_label_disambiguator_parts(label)
    ]
    return [part for part in parts if part]


def _plain_slash_parts_for_context_compare(label):
    return [
        _plain_formula_text(part).strip().lower()
        for part in str(label).split("/")
        if _plain_formula_text(part).strip()
    ]


def _previous_extra_context_part_covered_by_full_context(extra_part, full_context_parts, gradient_species):
    gradient_plain = _plain_formula_text(gradient_species or "").strip().lower()
    if not gradient_plain:
        return False

    for full_part in full_context_parts:
        if extra_part == full_part:
            return True

        slash_parts = _plain_slash_parts_for_context_compare(full_part)
        if len(slash_parts) < 2:
            continue
        if not any(gradient_plain in part for part in slash_parts):
            continue

        non_gradient_parts = [
            part for part in slash_parts
            if gradient_plain not in part
        ]
        if extra_part in non_gradient_parts:
            return True

    return False


def _gradient_context_redundant_with_previous_line(
    context_line,
    prev_line_label,
    *,
    full_context_line=None,
    gradient_species=None,
):
    context_plain = _plain_formula_text(context_line).strip().lower()
    prev_plain = _plain_formula_text(prev_line_label).strip().lower()
    if context_plain == prev_plain:
        return True

    context_parts = _plain_label_parts_for_context_compare(context_line)
    prev_parts = set(_plain_label_parts_for_context_compare(prev_line_label))
    if not context_parts or not all(part in prev_parts for part in context_parts):
        return False

    context_disambig = _plain_disambiguator_parts_for_context_compare(context_line)
    prev_disambig = set(_plain_disambiguator_parts_for_context_compare(prev_line_label))
    if context_disambig and not all(part in prev_disambig for part in context_disambig):
        return False

    extra_prev_parts = [part for part in prev_parts if part not in set(context_parts)]
    if not extra_prev_parts:
        return True

    if full_context_line in ("", None):
        return False

    full_context_parts = _plain_label_parts_for_context_compare(full_context_line)
    if not full_context_parts:
        return False

    return all(
        _previous_extra_context_part_covered_by_full_context(
            part,
            full_context_parts,
            gradient_species,
        )
        for part in extra_prev_parts
    )


def _estimate_custom_legend_panel_width(ax, entries, options, legend_fs, layout_cache=None):
    """
    Estimate panel width from the widest visible legend text.
    Returns width in axes-coordinate units.
    """
    max_text_w = 0.0
    has_any_text = False

    for entry in entries:
        if entry["type"] == "line":
            has_any_text = True
            max_text_w = max(
                max_text_w,
                _measure_text_width_axes(ax, entry.get("label", ""), legend_fs),
            )
            continue

        group = entry["group"]

        context_line = group.get("legend context line", "")
        prev_line_label = entry.get("previous line label", "")
        show_context = _show_gradient_context_line(
            context_line,
            prev_line_label,
            entry.get("previous entry type"),
            full_context_line=group.get("legend full context line"),
            gradient_species=group.get("gradient species"),
        )

        if show_context:
            has_any_text = True
            max_text_w = max(
                max_text_w,
                _measure_text_width_axes(ax, format_chemical_formulas(context_line), legend_fs),
            )

        endpoint_labels = [lbl for lbl in group.get("endpoint ticklabels", []) if lbl not in (None, "")]
        if endpoint_labels:
            has_any_text = True
            for lbl in endpoint_labels:
                max_text_w = max(
                    max_text_w,
                    _measure_text_width_axes(ax, format_chemical_formulas(lbl), legend_fs),
                )

    left_pad_axes = 0.008
    right_pad_axes = 0.008
    sample_len_axes = _legend_sample_length_axes(
        ax,
        options,
        legend_fs,
        layout_cache=layout_cache,
    )
    text_gap_axes = _points_to_axes_x(
        ax,
        legend_fs * float(mpl.rcParams.get("legend.handletextpad", 0.8)),
        layout_cache=layout_cache,
    )

    if has_any_text:
        width = left_pad_axes + sample_len_axes + text_gap_axes + max_text_w + right_pad_axes
        legend_pad = float(options.get("legend pad", 0.02))
        max_width = max(0.12, 1.0 - 2.0 * legend_pad)
        return float(np.clip(width, 0.09, max_width))

    # bar-only legend
    width = left_pad_axes + sample_len_axes + right_pad_axes
    return float(np.clip(width, 0.05, 0.12))

def _points_to_axes_x(ax, points, layout_cache=None):
    if layout_cache is None:
        layout_cache = _build_layout_cache(ax)

    ax_bbox = layout_cache["ax_bbox"]
    dpi = layout_cache["dpi"]

    if ax_bbox.width == 0:
        return 0.0

    pixels = points * dpi / 72.0
    return pixels / ax_bbox.width


def _points_to_axes_y(ax, points, layout_cache=None):
    if layout_cache is None:
        layout_cache = _build_layout_cache(ax)

    ax_bbox = layout_cache["ax_bbox"]
    dpi = layout_cache["dpi"]

    if ax_bbox.height == 0:
        return 0.0

    pixels = points * dpi / 72.0
    return pixels / ax_bbox.height

def _legend_font_candidates(min_fs=6, max_fs=None, step=2):
    if max_fs is None:
        max_fs = _default_legend_fontsize()
    return list(range(int(max_fs), int(min_fs) - 1, -int(step)))

def _estimate_custom_panel_size(ax, color_spec, options, legend_fs, layout_cache=None, cap_height=True):
    """
    Return (panel_width, panel_height) in axes coordinates for a given fontsize.
    """
    gradient_groups = color_spec.get("gradient groups", [])
    entries = []

    for i, line in enumerate(ax.lines):
        label = line.get_label()
        if label not in (None, "") and not str(label).startswith("_"):
            entries.append({
                "type": "line",
                "order": i,
                "handle": line,
                "label": label,
            })

    for j, group in enumerate(gradient_groups):
        indices = group.get("indices", [])
        order = min(indices) if len(indices) > 0 else (len(ax.lines) + j)
        entries.append({
            "type": "colorbar",
            "order": order,
            "group": group,
        })

    if len(entries) == 0:
        return 0.0, 0.0

    entries.sort(key=lambda d: d["order"])

    for k, entry in enumerate(entries):
        prev_line_label = ""
        previous_entry_type = entries[k - 1]["type"] if k > 0 else None
        for j in range(k - 1, -1, -1):
            if entries[j]["type"] == "line":
                prev_line_label = entries[j].get("label", "")
                break
        entry["previous line label"] = prev_line_label
        entry["previous entry type"] = previous_entry_type

    panel_width = _estimate_custom_legend_panel_width(
        ax, entries, options, legend_fs, layout_cache=layout_cache
    )

    row_h, gap = _custom_legend_row_metrics(
        ax,
        legend_fs,
        layout_cache=layout_cache,
    )

    weights = []
    for entry in entries:
        if entry["type"] == "line":
            weights.append(1.0)
        else:
            group = entry["group"]
            n_items = len(group.get("indices", []))

            context_line = group.get("legend context line", "")
            prev_line_label = entry.get("previous line label", "")
            show_context = _show_gradient_context_line(
                context_line,
                prev_line_label,
                entry.get("previous entry type"),
                full_context_line=group.get("legend full context line"),
                gradient_species=group.get("gradient species"),
            )

            has_endpoint_text = any(
                lbl not in (None, "") for lbl in group.get("endpoint ticklabels", [])
            )

            bar_weight = max(0.8, np.sqrt(max(1.0, float(n_items))) * 1.5 * float(options.get("colorbar height scale", 1.0)))
            context_weight = 1.0 if show_context else 0.0
            endpoint_weight = 0.5 if has_endpoint_text else 0.0
            context_top_gap_weight = (
                float(mpl.rcParams.get("legend.labelspacing", 0.5))
                if show_context and entry.get("previous entry type") == "line"
                else 0.0
            )
            weights.append(bar_weight + context_weight + endpoint_weight + context_top_gap_weight)

    panel_height = max(row_h, row_h * sum(weights) + gap * max(len(entries) - 1, 0))
    if cap_height:
        panel_height = min(0.82, panel_height)

    return panel_width, panel_height

def _discrete_legend_overlap_score(ax, legend_fs, loc):
    fig = ax.figure
    handles, labels = ax.get_legend_handles_labels()
    visible = [
        (handle, label)
        for handle, label in zip(handles, labels)
        if label not in (None, "") and not str(label).startswith("_")
    ]
    if len(visible) == 0:
        return 0

    handles, labels = zip(*visible)
    temp = ax.legend(handles, labels, fontsize=legend_fs, loc=loc)
    fig.draw_without_rendering()
    renderer = fig.canvas.get_renderer()
    bbox_display = temp.get_window_extent(renderer=renderer)
    score = _line_overlap_score(ax, bbox_display)
    temp.remove()
    return score

def _custom_panel_overlap_score(ax, color_spec, options, legend_fs, loc, layout_cache=None):
    panel_width, panel_height = _estimate_custom_panel_size(
        ax, color_spec, options, legend_fs, layout_cache=layout_cache
    )
    bbox_display = _panel_bbox_display(
        ax,
        loc,
        panel_width,
        panel_height,
        outside=False,
        pad=float(options.get("legend pad", 0.02)),
    )
    score = _line_overlap_score(ax, bbox_display)
    return score, panel_width, panel_height

def _outside_discrete_legend_size(ax, legend_fs, pad=0.02):
    """
    Measure a normal matplotlib legend placed outside upper-right.

    Returns
    -------
    width_axes, height_axes : float, float
        Legend size in axes-coordinate units.
    """
    fig = ax.figure
    temp = ax.legend(
        fontsize=legend_fs,
        loc="upper left",
        bbox_to_anchor=(1 + pad, 1),
    )
    fig.draw_without_rendering()
    renderer = fig.canvas.get_renderer()

    bbox_display = temp.get_window_extent(renderer=renderer)
    ax_bbox = ax.get_window_extent(renderer=renderer)

    temp.remove()

    if ax_bbox.width == 0 or ax_bbox.height == 0:
        return 0.0, 0.0

    width_axes = bbox_display.width / ax_bbox.width
    height_axes = bbox_display.height / ax_bbox.height
    return width_axes, height_axes


def _resolve_outside_upper_right_fontsize(ax, color_spec, options, min_fs=6, max_fs=None):
    """
    Pick the largest fontsize that lets an outside upper-right legend/panel
    fit within the full axes height (respecting pad), staying within bounds.
    """
    if max_fs is None:
        max_fs = _default_legend_fontsize()
    pad = float(options.get("legend pad", 0.02))
    available_h = max(0.0, 1 - 2 * pad)

    use_custom_panel = (
        _normalize_multiplot_legend_mode(options.get("legend mode", "auto")) == "colorbar"
        and len(color_spec.get("gradient groups", [])) > 0
    )

    for fs in _legend_font_candidates(min_fs=min_fs, max_fs=max_fs):
        if use_custom_panel:
            panel_width, panel_height = _estimate_custom_panel_size(
                ax,
                color_spec,
                options,
                fs,
                cap_height=False,
            )
            if panel_height <= available_h:
                return fs
        else:
            _, legend_height = _outside_discrete_legend_size(ax, fs, pad=pad)
            if legend_height <= available_h:
                return fs

    return min_fs

def _legend_max_fontsize():
    axis_size = mpl.rcParams.get("axes.labelsize", _default_legend_fontsize())
    try:
        axis_size = float(axis_size)
    except (TypeError, ValueError):
        axis_size = float(_default_legend_fontsize())
    return max(6.0, min(float(_default_legend_fontsize()), axis_size))

def _resolve_adaptive_legend_layout(ax, color_spec, options):
    """
    Resolve legend fontsize + inside location together.

    Returns
    -------
    legend_fs, legend_loc, legend_outside
    """
    legend_outside = options.get("legend outside", False)
    legend_loc = _normalize_legend_loc(options.get("legend loc", "best"))
    explicit_fs = options.get("legend fontsize", None)

    # Outside legends: keep current behavior
    if legend_outside:
        if explicit_fs not in (None, "auto"):
            return explicit_fs, legend_loc, True
        return _legend_max_fontsize(), legend_loc, True

    # Explicit fontsize: just keep it
    if explicit_fs not in (None, "auto"):
        return explicit_fs, legend_loc, False

    min_fs = 6
    max_fs = _legend_max_fontsize()
    font_candidates = _legend_font_candidates(min_fs=min_fs, max_fs=max_fs)

    use_custom_panel = (
        _normalize_multiplot_legend_mode(options.get("legend mode", "auto")) == "colorbar"
        and len(color_spec.get("gradient groups", [])) > 0
    )
    max_inside_panel_height = min(0.82, max(0.0, 1 - 2 * float(options.get("legend pad", 0.02))))

    corners = _legend_candidate_corners()

    # --------------------------
    # BEST: search all corners
    # --------------------------
    layout_cache = _build_layout_cache(ax) if use_custom_panel else None
    
    with _temporary_figure_dpi(ax.figure, 100):
        if str(legend_loc).lower() == "best":
            for fs in font_candidates:
                layout_cache = _build_layout_cache(ax)
                for loc in corners:
                    if use_custom_panel:
                        _, required_panel_height = _estimate_custom_panel_size(
                            ax,
                            color_spec,
                            options,
                            fs,
                            layout_cache=layout_cache,
                            cap_height=False,
                        )
                        if required_panel_height > max_inside_panel_height:
                            continue
                        score, _, _ = _custom_panel_overlap_score(
                            ax, color_spec, options, fs, loc, layout_cache=layout_cache
                        )
                    else:
                        score = _discrete_legend_overlap_score(ax, fs, loc)

                    if score == 0:
                        return fs, loc, False

            # fallback: if no inside corner works even at min font,
            # go outside upper-right at the largest fontsize that fits the axes height
            outside_fs = _resolve_outside_upper_right_fontsize(
                ax,
                color_spec,
                options,
                min_fs=min_fs,
                max_fs=max_fs,
            )
            return outside_fs, "upper right", True

    # --------------------------
    # Fixed inside corner
    # --------------------------
    for fs in font_candidates:
        if use_custom_panel:
            _, required_panel_height = _estimate_custom_panel_size(
                ax,
                color_spec,
                options,
                fs,
                layout_cache=layout_cache,
                cap_height=False,
            )
            if required_panel_height > max_inside_panel_height:
                continue
            score, _, _ = _custom_panel_overlap_score(
                ax, color_spec, options, fs, legend_loc, layout_cache=layout_cache)
        else:
            score = _discrete_legend_overlap_score(ax, fs, legend_loc)

        if score == 0:
            return fs, legend_loc, False

    return min_fs, legend_loc, False

def _legend_fontsize_from_color_spec(color_spec, default_size=None, min_size=6, max_size=None):
    if default_size is None:
        default_size = _default_legend_fontsize()
    if max_size is None:
        max_size = default_size
    visible_labels = [
        lbl for lbl in color_spec.get("labels", [])
        if lbl not in (None, "", "_nolegend_")
    ]

    gradient_groups = color_spec.get("gradient groups", [])

    text_candidates = list(visible_labels)

    for group in gradient_groups:
        context_line = group.get("legend context line", "")
        if context_line not in (None, ""):
            text_candidates.append(context_line)

        for lbl in group.get("endpoint ticklabels", []):
            if lbl not in (None, ""):
                text_candidates.append(lbl)

    if not text_candidates:
        return default_size

    longest = max(len(str(lbl)) for lbl in text_candidates)
    n_entries = len(visible_labels) + len(gradient_groups)

    penalty = max(longest / 28, n_entries / 6, 1)
    size = default_size / penalty**0.35
    return float(np.clip(size, min_size, max_size))

def _custom_legend_layout(
    ax,
    panel_width,
    has_text,
    options,
    legend_fs=None,
    layout_cache=None,
):
    """
    Build panel-relative geometry so that:
    - discrete line length == legend sample length
    - colorbar width + endpoint tick length == legend sample length
    """
    if legend_fs is None:
        legend_fs = _default_legend_fontsize()
    left_pad_axes = 0.008
    right_pad_axes = 0.008
    text_gap_axes = _points_to_axes_x(
        ax,
        legend_fs * float(mpl.rcParams.get("legend.handletextpad", 0.8)),
        layout_cache=layout_cache,
    )

    sample_len_axes = _legend_sample_length_axes(
        ax,
        options,
        legend_fs,
        layout_cache=layout_cache,
    )
    bar_w_axes = sample_len_axes * (2.0 / 3.0)
    endpoint_tick_len_axes = max(0.0, sample_len_axes - bar_w_axes)

    # if no text, right-align the sample block inside the panel
    if has_text:
        sample_x0_axes = left_pad_axes
        text_x_axes = sample_x0_axes + sample_len_axes + text_gap_axes
    else:
        sample_x0_axes = max(left_pad_axes, panel_width - right_pad_axes - sample_len_axes)
        text_x_axes = None

    return {
        "line_x0": sample_x0_axes / panel_width,
        "line_x1": (sample_x0_axes + sample_len_axes) / panel_width,
        "bar_x": sample_x0_axes / panel_width,
        "bar_w": bar_w_axes / panel_width,
        "endpoint_tick_len": endpoint_tick_len_axes,
        "text_x": None if text_x_axes is None else text_x_axes / panel_width,
    }

def _custom_legend_row_metrics(ax, legend_fs, layout_cache=None):
    """
    Return Matplotlib-like custom legend row height and label gap in axes units.
    """
    row_points = legend_fs * max(
        1.0,
        float(mpl.rcParams.get("legend.handleheight", 0.7)),
    )
    gap_points = legend_fs * float(mpl.rcParams.get("legend.labelspacing", 0.5))

    return (
        _points_to_axes_y(ax, row_points, layout_cache=layout_cache),
        _points_to_axes_y(ax, gap_points, layout_cache=layout_cache),
    )

def _set_colorbar_endpoint_tick_lengths(cb, endpoint_ticks, endpoint_length, intermediate_length):
    """
    Keep colorbar endpoint ticks visually aligned with line swatches while
    making intermediate trace ticks quieter.
    """
    endpoint_values = [
        float(tick) for tick in endpoint_ticks
        if tick not in (None, "")
    ]

    for tick in cb.ax.yaxis.get_major_ticks():
        loc = tick.get_loc()
        is_endpoint = any(np.isclose(float(loc), endpoint) for endpoint in endpoint_values)
        length = endpoint_length if is_endpoint else intermediate_length
        tick.tick1line.set_markersize(length)
        tick.tick2line.set_markersize(length)

def _draw_multiplot_legend_and_colorbars(ax, color_spec, options, legend_fs):
    """
    Draw a combined legend panel containing:
    - discrete entries as line swatches + labels
    - gradient entries as mini colorbars + labels

    This makes the colorbars feel like part of one continuous legend list,
    instead of a separate block next to the legend.
    """
    legend_mode = _normalize_multiplot_legend_mode(options.get("legend mode", "auto"))
    gradient_groups = color_spec.get("gradient groups", [])

    if len(ax.lines) <= 1 and len(gradient_groups) == 0:
        return None

    # Fall back to normal legend behavior
    if legend_mode != "colorbar" or len(gradient_groups) == 0:
        legend_loc = _normalize_legend_loc(options.get("legend loc", "best"))
        bbox_to_anchor = options.get("legend bbox to anchor", None)
        if options.get("legend outside", False) and bbox_to_anchor is None:
            legend_loc, bbox_to_anchor = _resolve_outside_matplotlib_legend_anchor(
                legend_loc,
                pad=float(options.get("legend pad", 0.02)),
            )
        elif bbox_to_anchor is None:
            bbox_to_anchor = _inside_label_legend_anchor(
                legend_loc,
                pad=float(options.get("legend pad", 0.02)),
            )
        return ax.legend(
            fontsize=legend_fs,
            loc=legend_loc,
            bbox_to_anchor=bbox_to_anchor,
        )

    colorbar_height_scale = 1.5 * float(options.get("colorbar height scale", 1.0))
    colorbar_reverse = options.get("colorbar reverse", True)
    colorbar_tick_length = options.get("colorbar tick length", 5)
    colorbar_tick_pad = options.get("colorbar tick pad", 8)

    # ----------------------------
    # Build ordered legend entries
    # ----------------------------
    entries = []

    # Discrete entries: keep their plotting order
    for i, line in enumerate(ax.lines):
        label = line.get_label()
        if label not in (None, "") and not str(label).startswith("_"):
            entries.append({
                "type": "line",
                "order": i,
                "handle": line,
                "label": label,
            })

    # Gradient entries: place them where their first plotted trace appears
    for j, group in enumerate(gradient_groups):
        indices = group.get("indices", [])
        order = min(indices) if len(indices) > 0 else (len(ax.lines) + j)
        entries.append({
            "type": "colorbar",
            "order": order,
            "group": group,
        })

    if len(entries) == 0:
        return None

    entries.sort(key=lambda d: d["order"])

    for k, entry in enumerate(entries):
        prev_line_label = ""
        previous_entry_type = entries[k - 1]["type"] if k > 0 else None

        for j in range(k - 1, -1, -1):
            if entries[j]["type"] == "line":
                prev_line_label = entries[j].get("label", "")
                break

        entry["previous line label"] = prev_line_label
        entry["previous entry type"] = previous_entry_type

    # ---------------------------------------
    # Allocate variable height per legend row
    # ---------------------------------------
    weights = []
    for entry in entries:
        if entry["type"] == "line":
            weights.append(1.0)
            continue

        group = entry["group"]
        n_items = len(group.get("indices", []))

        context_line = group.get("legend context line", "")
        prev_line_label = entry.get("previous line label", "")
        show_context = _show_gradient_context_line(
            context_line,
            prev_line_label,
            entry.get("previous entry type"),
            full_context_line=group.get("legend full context line"),
            gradient_species=group.get("gradient species"),
        )

        # Gentler scaling than linear in n_items
        bar_weight = max(0.8, np.sqrt(max(1.0, float(n_items))) * colorbar_height_scale)
        has_endpoint_text = any(lbl not in (None, "") for lbl in group.get("endpoint ticklabels", []))
        context_weight = 1.0 if show_context else 0.0
        endpoint_weight = 0.5 if has_endpoint_text else 0.0
        context_top_gap_weight = (
            float(mpl.rcParams.get("legend.labelspacing", 0.5))
            if show_context and entry.get("previous entry type") == "line"
            else 0.0
        )
        weights.append(bar_weight + context_weight + endpoint_weight + context_top_gap_weight)

    n_entries = len(entries)
    layout_cache = _build_layout_cache(ax)
    row_h, gap_axes = _custom_legend_row_metrics(
        ax,
        legend_fs,
        layout_cache=layout_cache,
    )

    panel_width = _estimate_custom_legend_panel_width(
        ax,
        entries,
        options,
        legend_fs,
        layout_cache=layout_cache,
    )
    panel_height = min(
        0.82,
        max(row_h, row_h * sum(weights) + gap_axes * max(n_entries - 1, 0))
    )

    # ----------------------------
    # Resolve panel position HERE
    # ----------------------------
    legend_loc = _normalize_legend_loc(options.get("legend loc", "best"))
    legend_outside = options.get("legend outside", False)
    legend_pad = float(options.get("legend pad", 0.02))

    bbox = options.get("legend bbox to anchor", None)
    if bbox is not None and len(bbox) >= 2:
        x0, y0 = bbox[:2]
    else:
        if legend_loc == "best" and not legend_outside:
            resolved_loc, resolved_outside = _choose_best_custom_panel_loc(
                ax,
                panel_width,
                panel_height,
                pad=legend_pad,
            )
            x0, y0 = _resolve_legend_panel_position(
                resolved_loc,
                panel_width,
                panel_height,
                outside=resolved_outside,
                pad=legend_pad,
            )
        else:
            x0, y0 = _resolve_legend_panel_position(
                legend_loc if legend_loc != "best" else "upper right",
                panel_width,
                panel_height,
                outside=legend_outside,
                pad=legend_pad,
            )

    # Panel dimensions in axes coordinates
    has_line_entries = any(entry["type"] == "line" for entry in entries)

    has_context_entries = False
    for entry in entries:
        if entry["type"] != "colorbar":
            continue

        context_line = entry["group"].get("legend context line", "")
        prev_line_label = entry.get("previous line label", "")
        show_context = _show_gradient_context_line(
            context_line,
            prev_line_label,
            entry.get("previous entry type"),
            full_context_line=entry["group"].get("legend full context line"),
            gradient_species=entry["group"].get("gradient species"),
        )

        if show_context:
            has_context_entries = True
            break

    

    # Create one combined legend panel
    panel_ax = ax.inset_axes([x0, y0, panel_width, panel_height], transform=ax.transAxes)
    panel_ax.set_axis_off()

    gap = gap_axes / panel_height if panel_height > 0 else 0.0
    usable_h = 1.0 - gap * max(n_entries - 1, 0)
    unit_h = usable_h / sum(weights)

    y_top = 1.0

    for entry, weight in zip(entries, weights):
        entry_h = weight * unit_h
        y_bottom = y_top - entry_h
        y_mid = 0.5 * (y_top + y_bottom)

        # ----------------
        # Discrete entries
        # ----------------
        if entry["type"] == "line":
            line = entry["handle"]
            label = entry["label"]

            layout = _custom_legend_layout(
                ax,
                panel_width,
                has_text=True,
                options=options,
                legend_fs=legend_fs,
                layout_cache=layout_cache,
            )
            x_line0 = layout["line_x0"]
            x_line1 = layout["line_x1"]
            x_text = layout["text_x"]

            panel_ax.plot(
                [x_line0, x_line1],
                [y_mid, y_mid],
                transform=panel_ax.transAxes,
                color=line.get_color(),
                linestyle=line.get_linestyle(),
                linewidth=line.get_linewidth(),
                alpha=line.get_alpha() if line.get_alpha() is not None else 1.0,
                clip_on=False,
            )

            marker = line.get_marker()
            if marker not in (None, "", "None", " "):
                panel_ax.plot(
                    [0.105],
                    [y_mid],
                    transform=panel_ax.transAxes,
                    marker=marker,
                    markersize=line.get_markersize(),
                    markerfacecolor=line.get_markerfacecolor(),
                    markeredgecolor=line.get_markeredgecolor(),
                    linestyle="None",
                    color=line.get_color(),
                    clip_on=False,
                )

            panel_ax.text(
                x_text,
                y_mid,
                label,
                transform=panel_ax.transAxes,
                va="center",
                ha="left",
                fontsize=legend_fs,
            )

        # ----------------
        # Gradient entries
        # ----------------
        else:
            group = entry["group"]

            # Colorbar geometry inside the combined panel
            context_line = group.get("legend context line", "")
            prev_line_label = entry.get("previous line label", "")
            show_context = _show_gradient_context_line(
                context_line,
                prev_line_label,
                entry.get("previous entry type"),
                full_context_line=group.get("legend full context line"),
                gradient_species=group.get("gradient species"),
            )

            if not show_context:
                context_line = ""

            has_context = context_line not in ("", None)
            has_endpoint_text = any(lbl not in (None, "") for lbl in group.get("endpoint ticklabels", []))
            endpoint_pad = 0.45 * unit_h if has_endpoint_text else 0.0
            
            layout = _custom_legend_layout(
                ax,
                panel_width,
                has_text=(has_context or has_endpoint_text),
                options=options,
                legend_fs=legend_fs,
                layout_cache=layout_cache,
            )

            bar_x = layout["bar_x"]
            bar_w = layout["bar_w"]
            endpoint_tick_len_axes = layout["endpoint_tick_len"]
            text_x = layout["text_x"]

            context_line = group.get("legend context line", "")
            prev_line_label = entry.get("previous line label", "")
            show_context = _show_gradient_context_line(
                context_line,
                prev_line_label,
                entry.get("previous entry type"),
                full_context_line=group.get("legend full context line"),
                gradient_species=group.get("gradient species"),
            )

            if not show_context:
                context_line = ""
            has_context = context_line not in ("", None)
            context_top_gap = (
                gap if has_context and entry.get("previous entry type") == "line"
                else 0.0
            )

            # Reserve one text line above the colorbar when context exists
            context_h = .7 * unit_h if has_context else 0.0
            context_gap = gap if has_context else 0.0

            default_bar_pad_y = min(0.03, 0.15 * entry_h)
            default_bar_h = max(entry_h - 2 * default_bar_pad_y, 0.03)

            available_bar_h = max(
                0.03,
                entry_h - context_top_gap - context_h - context_gap - 2 * (default_bar_pad_y + endpoint_pad)
            )

            # user-scaled height
            bar_h = max(
                0.03,
                min(available_bar_h, default_bar_h * colorbar_height_scale)
            )

            # Place context at the top of the row
            if has_context and text_x is not None:
                context_y = y_top - context_top_gap - context_h / 2
                panel_ax.text(
                    text_x,
                    context_y,
                    format_chemical_formulas(context_line),
                    transform=panel_ax.transAxes,
                    va="center",
                    ha="left",
                    fontsize=legend_fs,
                )

            # Anchor colorbar below the context block
            bar_top = y_top - context_top_gap - context_h - context_gap - default_bar_pad_y - endpoint_pad
            bar_y = bar_top - bar_h

            cax = panel_ax.inset_axes(
                [bar_x, bar_y, bar_w, bar_h],
                transform=panel_ax.transAxes
            )

            sm = mpl.cm.ScalarMappable(norm=group["norm"], cmap=group["cmap"])
            sm.set_array([])

            cb = panel_ax.figure.colorbar(sm, cax=cax, orientation="vertical")

            show_trace_ticks = options.get("colorbar trace ticks", True)
            tick_label_mode = options.get("colorbar tick labels", "endpoints")

            if show_trace_ticks:
                cb.set_ticks(group.get("ticks", []))
                if tick_label_mode == "all":
                    cb.set_ticklabels(group.get("ticklabels", []))
                elif text_x is None:
                    cb.set_ticklabels(group.get("ticklabels", []))
                else:
                    cb.set_ticklabels([""] * len(group.get("ticks", [])))
            else:
                cb.set_ticks(group.get("endpoint ticks", []))
                if text_x is None:
                    cb.set_ticklabels(group.get("endpoint ticklabels", []))
                else:
                    cb.set_ticklabels([""] * len(group.get("endpoint ticks", [])))

            cb.ax.minorticks_off()
            cb.ax.tick_params(
                which='minor',
                length=0,
                width=0,
            )

            cb.ax.tick_params(
                which='major',
                labelsize=legend_fs,
                length=colorbar_tick_length,
                pad=colorbar_tick_pad,
            )

            if colorbar_reverse:
                cb.ax.invert_yaxis()

            # Draw endpoint labels manually so they line up with the line-label text
            endpoint_ticks = group.get("endpoint ticks", [])
            endpoint_ticklabels = group.get("endpoint ticklabels", [])
            endpoint_tick_length = (
                endpoint_tick_len_axes
                * layout_cache["ax_bbox"].width
                * 72.0
                / layout_cache["dpi"]
            )
            intermediate_tick_length = min(
                float(colorbar_tick_length),
                endpoint_tick_length * 0.55,
            )
            _set_colorbar_endpoint_tick_lengths(
                cb,
                endpoint_ticks,
                endpoint_tick_length,
                intermediate_tick_length,
            )

            if text_x is not None and tick_label_mode != "all":
                for tick_pos, tick_lbl in zip(endpoint_ticks, endpoint_ticklabels):
                    if tick_lbl in (None, ""):
                        continue

                    display_y = cax.transData.transform((0.0, float(tick_pos)))[1]
                    label_y = panel_ax.transAxes.inverted().transform((0.0, display_y))[1]

                    panel_ax.text(
                        text_x,
                        label_y,
                        _format_already_or_chemical(tick_lbl),
                        transform=panel_ax.transAxes,
                        va="center",
                        ha="left",
                        fontsize=legend_fs,
                    )

        y_top = y_bottom - gap

    return panel_ax

def _plain_formula_text(text):
    """
    Convert simple eCAT formula formatting back to plain text for comparisons.
    Example: H$_2$O -> H2O
    """
    if text is None:
        return ""
    text = str(text)
    text = re.sub(r"\$_(\d+)\$", r"\1", text)
    text = text.replace("$", "")
    return text

def _format_already_or_chemical(label):
    """
    Format plain labels as chemistry, but leave mathtext labels intact.
    """
    if label is None:
        return ""
    label = str(label)
    if "$_" in label or "$^" in label or "^{" in label:
        return label
    if re.match(r"^[+-]\d", label):
        return label[0] + format_chemical_formulas(label[1:])
    return format_chemical_formulas(label)

def _split_top_level_commas(text):
    parts = []
    current = []
    paren_depth = 0
    bracket_depth = 0

    for char in str(text):
        if char == "," and paren_depth == 0 and bracket_depth == 0:
            part = "".join(current).strip()
            if part:
                parts.append(part)
            current = []
            continue

        current.append(char)
        if char == "(":
            paren_depth += 1
        elif char == ")" and paren_depth > 0:
            paren_depth -= 1
        elif char == "[":
            bracket_depth += 1
        elif char == "]" and bracket_depth > 0:
            bracket_depth -= 1

    part = "".join(current).strip()
    if part:
        parts.append(part)
    return parts


def _split_trailing_disambiguator(label):
    text = str(label).strip()
    if not text.endswith(")"):
        return text, None

    depth = 0
    for idx in range(len(text) - 1, -1, -1):
        char = text[idx]
        if char == ")":
            depth += 1
        elif char == "(":
            depth -= 1
            if depth == 0:
                if idx > 0 and text[idx - 1].isspace():
                    return text[:idx].rstrip(), text[idx + 1:-1].strip()
                return text, None
    return text, None


def _split_label_parts(label):
    """
    Split an auto-generated multiplot label into comma-separated parts.
    Ignores empty labels and placeholders like 'Background'.
    """
    if label in (None, "", "_nolegend_", "Background"):
        return []
    main_text, _disambiguator = _split_trailing_disambiguator(label)
    return _split_top_level_commas(main_text)


def _split_label_disambiguator_parts(label):
    if label in (None, "", "_nolegend_", "Background"):
        return []
    _main_text, disambiguator = _split_trailing_disambiguator(label)
    if not disambiguator:
        return []
    return _split_top_level_commas(disambiguator)


def _ordered_common_label_parts(group_labels):
    """
    Find the ordered intersection of comma-separated label parts
    across the labels being consolidated into one colorbar.
    Order is preserved from the first label.
    """
    split_labels = [_split_label_parts(lbl) for lbl in group_labels]
    split_labels = [parts for parts in split_labels if len(parts) > 0]

    if len(split_labels) == 0:
        return []

    first_parts = split_labels[0]
    other_sets = [
        {_plain_formula_text(part).lower() for part in parts}
        for parts in split_labels[1:]
    ]

    common_parts = []
    for part in first_parts:
        part_plain = _plain_formula_text(part).lower()
        if all(part_plain in other_set for other_set in other_sets):
            common_parts.append(part)

    return common_parts


def _ordered_common_disambiguator_parts(group_labels):
    split_labels = [_split_label_disambiguator_parts(lbl) for lbl in group_labels]
    if len(split_labels) == 0 or any(len(parts) == 0 for parts in split_labels):
        return []

    first_parts = split_labels[0]
    other_sets = [
        {_plain_formula_text(part).lower() for part in parts}
        for parts in split_labels[1:]
    ]

    common_parts = []
    for part in first_parts:
        part_plain = _plain_formula_text(part).lower()
        if all(part_plain in other_set for other_set in other_sets):
            common_parts.append(part)

    return common_parts


def _attach_gradient_legend_text(color_spec, labels, label_alterations=None, user_labels_explicit=False):
    """
    Use the already-generated multiplot labels for the entries that are being
    consolidated into each colorbar.

    Example result:
        legend context line: CO2, 3 mM Fc, 1 mM Fe-tpyPY2Me
        ticklabels: +10 mM H2O, +2.8 M H2O
    """
    gradient_groups = color_spec.get("gradient groups", [])
    if len(gradient_groups) == 0:
        return color_spec

    for group in gradient_groups:
        indices = group.get("indices", [])
        group_labels = [
            labels[i] for i in indices
            if isinstance(i, int) and 0 <= i < len(labels)
        ]

        if user_labels_explicit:
            explicit_labels = [
                _format_already_or_chemical(apply_text_alterations(lbl, label_alterations))
                for lbl in group_labels
            ]
            group["legend context line"] = ""
            group["ticklabels"] = explicit_labels
            if len(explicit_labels) == 1:
                group["endpoint ticklabels"] = [explicit_labels[0]]
            elif len(explicit_labels) >= 2:
                group["endpoint ticklabels"] = [explicit_labels[0], explicit_labels[-1]]
            else:
                group["endpoint ticklabels"] = []
            continue

        common_parts = _ordered_common_label_parts(group_labels)

        group_title_raw = group.get("legend title", "")
        group_title_raw = apply_text_alterations(group_title_raw, label_alterations)
        group_title_fmt = format_chemical_formulas(group_title_raw) if group_title_raw else ""
        group_title_plain = _plain_formula_text(group_title_fmt).lower().strip()

        common_disambiguator_parts = _ordered_common_disambiguator_parts(group_labels)

        full_context_line = ", ".join(common_parts)
        if common_disambiguator_parts:
            disambiguator = ", ".join(common_disambiguator_parts)
            if full_context_line:
                full_context_line = f"{full_context_line} ({disambiguator})"
            else:
                full_context_line = f"({disambiguator})"
        group["legend full context line"] = full_context_line

        # Remove the varying delta species if it appears in the common context
        filtered_parts = []
        for part in common_parts:
            part_plain = _plain_formula_text(part).lower()
            if group_title_plain and group_title_plain in part_plain:
                continue
            filtered_parts.append(part)

        context_line = ", ".join(filtered_parts)
        if common_disambiguator_parts:
            disambiguator = ", ".join(common_disambiguator_parts)
            if context_line:
                context_line = f"{context_line} ({disambiguator})"
            else:
                context_line = f"({disambiguator})"
        group["legend context line"] = context_line

        # Rewrite visible endpoint labels to include the varying delta
        ticklabels = list(group.get("ticklabels", []))
        new_ticklabels = []

        for lbl in ticklabels:
            if lbl in (None, ""):
                new_ticklabels.append("")
                continue

            if group.get("gradient by") == "concentration" and group_title_raw:
                if group.get("legend unit") == "x":
                    mole_fraction = str(lbl).replace(" x", "").strip()
                    endpoint_raw = (
                        f"χ({format_chemical_formulas(group_title_raw)}) = "
                        f"{mole_fraction}"
                    )
                else:
                    endpoint_raw = f"+{format_chemical_formulas(f'{lbl} {group_title_raw}')}"
            else:
                endpoint_raw = f"{lbl}"

            endpoint_raw = apply_text_alterations(endpoint_raw, label_alterations)
            new_ticklabels.append(endpoint_raw)

        group["ticklabels"] = new_ticklabels

        if len(group.get("ticklabels", [])) == 1:
            group["endpoint ticklabels"] = [group["ticklabels"][0]]
        elif len(group.get("ticklabels", [])) >= 2:
            group["endpoint ticklabels"] = [
                group["ticklabels"][0],
                group["ticklabels"][-1],
            ]

    return color_spec

DPV_PULSE_SUBTITLE_FIELDS = [
    "amplitude",
    "pulse width",
    "sample width",
    "pulse period",
]

def _format_dpv_display_value(key, value, sig_figs=4):
    if value in ("", None):
        return ""
    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        return value

    if not np.isfinite(numeric_value):
        return ""

    if key == "amplitude":
        scaled, unit = scale_value(
            numeric_value,
            "V",
            selected_unit="auto",
            candidates=("m", "μ", "n", "p"),
        )
    else:
        scaled, unit = scale_value(
            numeric_value,
            "s",
            selected_unit="auto",
            candidates=("m", "μ", "n", "p"),
        )

    scaled = round_sigfigs(scaled, sig_figs)
    return f"{scaled:g} {unit}"

def _format_dpv_pulse_title_stat(key, value):
    symbol_by_key = {
        "amplitude": r"$\Delta E_p$",
        "pulse width": r"$t_p$",
        "sample width": r"$t_s$",
        "pulse period": "T",
    }
    symbol = symbol_by_key.get(key)
    if symbol is None:
        return value

    formatted_value = _format_dpv_display_value(key, value)
    if formatted_value == "":
        return ""

    return f"{symbol}={formatted_value}"

def _format_plot_title_stat(key, value):
    formatted = _format_dpv_pulse_title_stat(key, value)
    if formatted != value:
        return formatted

    if key in {"applied potential", "potential limits"}:
        return str(value)
    if key in {"run time", "current", "cycles", "technique"}:
        return str(value)

    return value

def _resolve_multiplot_labels_title_subtitle(echem_list, options):
    common_values, different_values, _ = echem_similar_different(
        echem_list,
        options=options,
        ignore=["scan window"],
        return_values=True,
    )

    similarities = list(common_values.keys())
    differences = list(different_values.keys())
    shared_compounds = list(common_values.get("compounds", []) or [])

    # prevent double-printing later
    similarities = [
        k for k in similarities
        if k not in {
            "compounds",
            "exp type",
            "ir comp resistance",
            "ir uncomp resistance",
            "ir comp percent",
        }
    ]

    # ---------- labels ----------
    user_labels = options.get("labels")

    if (
        user_labels is None
        and len(echem_list) > 1
        and all(callable(getattr(obj, "txt_stats", None)) for obj in echem_list)
    ):
        labels = []

        for echem_object in echem_list:
            stats = echem_object.txt_stats(options)
            parts = []

            for difference in differences:
                txt_difference = stats.get(difference, "")

                if txt_difference in ("", None):
                    continue

                if difference == "compounds":
                    if isinstance(txt_difference, list):
                        txt_difference = [
                            c for c in txt_difference if c not in shared_compounds
                        ]
                    if isinstance(txt_difference, list):
                        txt_difference = ", ".join(txt_difference)

                if difference == "segments":
                    if len(differences) > 1:
                        continue
                    txt_difference = f"{txt_difference} seg."

                if txt_difference not in ("", None):
                    parts.append(str(txt_difference))

            name = ", ".join(parts)

            if name == "":
                name = "Background"

            labels.append(name)

    else:
        if user_labels is None:
            labels = [echem_object.name for echem_object in echem_list]
        else:
            labels = []
            fallback_used = False

            for i, echem_object in enumerate(echem_list):
                try:
                    label = user_labels[i]
                    if label in (None, ""):
                        raise ValueError
                except (IndexError, ValueError):
                    label = echem_object.name
                    fallback_used = True

                labels.append(label)

            if fallback_used:
                print(
                    "\033[91mWarning: Some labels were missing or invalid. "
                    "Defaulted to echem.name where needed.\033[0m"
                )

    # ---------- subtitle ----------
    subtitle = options.get("subtitle")
    if subtitle is True:
        subtitle = "auto"
    elif subtitle is False:
        subtitle = None
    subtitle_is_explicit = (
        isinstance(subtitle, str)
        and subtitle not in ("auto", "")
    )
    subtitle_stats = [
        "solvent",
        "gas",
        "technique",
        "scan rate",
        "segments",
        "applied potential",
        "run time",
        "cycles",
        "current",
        "potential limits",
    ]
    if any(isinstance(echem_object, dpv) for echem_object in echem_list):
        for key in DPV_PULSE_SUBTITLE_FIELDS:
            if key not in subtitle_stats:
                subtitle_stats.append(key)

    if subtitle is None:
        subtitle = None
    elif subtitle_is_explicit:
        subtitle = str(subtitle)
    else:
        if subtitle != "auto":
            subtitle_stats = subtitle
        subtitle = ""

    # ---------- title ----------
    title = options.get("title", "auto")
    if title is True:
        title = "auto"
    elif title is False:
        title = None

    if title == "auto" or (isinstance(title, str) and "GENERATE" in title):
        if title == "auto":
            title = ""
        else:
            title = title[:-8]

        group_stats = echem_list[0].txt_stats(options)
        group_stats.pop('scan window', None)
        group_stats.pop('exp type', None)

        if shared_compounds:
            txt_shared_compounds = ", ".join(shared_compounds)
            if "compounds" in subtitle_stats and subtitle is not None:
                subtitle += txt_shared_compounds + ", "
            else:
                title += txt_shared_compounds + ", "

        for similarity in similarities:
            txt_similarity = group_stats.get(similarity, "")

            if txt_similarity in ("", None):
                continue

            if isinstance(txt_similarity, list):
                txt_similarity = ", ".join(txt_similarity)

            txt_similarity = _format_plot_title_stat(similarity, txt_similarity)

            if (
                similarity in subtitle_stats
                and subtitle is not None
                and not subtitle_is_explicit
            ):
                if similarity == "segments":
                    txt_similarity = f"{txt_similarity} seg."
                subtitle += str(txt_similarity) + ", "
            else:
                title += str(txt_similarity) + ", "

        label_alterations = options.get("label alterations")

        title = title[:-2] if title else None
        if subtitle is not None and not subtitle_is_explicit:
            subtitle = subtitle[:-2] if subtitle else None

        title = apply_text_alterations(title, label_alterations)
        subtitle = apply_text_alterations(subtitle, label_alterations)

        title = format_chemical_formulas(title) if title else None
        if subtitle is not None:
            subtitle = format_chemical_formulas(subtitle) if subtitle else None

    if subtitle == "":
        subtitle = None
    if title == "":
        title = None

    return labels, title, subtitle, shared_compounds, similarities


def _resolve_single_plot_title_subtitle(echem_object, options):
    """
    Resolve the title for one-object plots using the same auto-title logic as multiplot.
    """
    title_opt = options.get("title", True)

    if title_opt in (False, None, ""):
        return None, None

    if isinstance(title_opt, str) and title_opt != "auto":
        subtitle_opt = options.get("subtitle")
        if subtitle_opt is True:
            return title_opt, None
        if isinstance(subtitle_opt, str) and subtitle_opt.strip().lower() == "auto":
            return title_opt, None
        if subtitle_opt in (False, None, ""):
            return title_opt, None
        return title_opt, str(subtitle_opt)

    mp_options = _multiplot_options_from_mapping(options)
    mp_options["title"] = "auto"
    _labels, title, subtitle, _shared_compounds, _similarities = (
        _resolve_multiplot_labels_title_subtitle([echem_object], mp_options)
    )
    return title, subtitle


def _normalize_deduplicate_label_fields(option):
    if option in (False, None):
        return None
    if isinstance(option, str):
        text = option.strip()
        if text.lower() in {"", "false", "off", "none", "0"}:
            return None
        return [text]
    if option is True:
        return ["scan window", "segments"]
    if isinstance(option, (list, tuple, set)):
        return [str(item) for item in option if str(item).strip()]
    return ["scan window", "segments"]


def _format_deduplicate_label_value(field, value):
    if value in (None, ""):
        return ""
    field_key = str(field).strip().lower().replace("_", " ")
    if field_key == "segments":
        return f"{value} seg."
    if isinstance(value, list):
        return "[" + ", ".join(f"{item:g}" if isinstance(item, (int, float)) else str(item) for item in value) + "]"
    if isinstance(value, tuple):
        return "(" + ", ".join(str(item) for item in value) + ")"
    return str(value)


def _deduplicate_multiplot_labels(echem_list, labels, options):
    fields = _normalize_deduplicate_label_fields(options.get("deduplicate labels", False))
    if fields is None or len(labels) <= 1:
        return labels

    resolved = list(labels)
    from .collection import get_sort_group_dict, validate_keys

    opt_dict = get_sort_group_dict()
    normalized_fields = [str(field).strip().lower().replace("_", " ") for field in fields]
    if not validate_keys(normalized_fields, opt_dict, "deduplicate label"):
        return resolved
    stats_by_index = [obj.txt_stats(options) for obj in echem_list]

    label_to_indices = {}
    for idx, label in enumerate(resolved):
        if label in (None, "", "_nolegend_"):
            continue
        label_to_indices.setdefault(label, []).append(idx)

    for duplicate_indices in label_to_indices.values():
        if len(duplicate_indices) <= 1:
            continue

        suffix_parts_by_index = {idx: [] for idx in duplicate_indices}
        for field_key in normalized_fields:
            values = [
                stats_by_index[idx].get(field_key)
                for idx in duplicate_indices
            ]
            hashable_values = [_to_hashable(value) for value in values]
            if len(set(hashable_values)) <= 1:
                continue
            for idx, value in zip(duplicate_indices, values):
                formatted = _format_deduplicate_label_value(field_key, value)
                if formatted:
                    suffix_parts_by_index[idx].append(formatted)

        candidates = []
        for idx in duplicate_indices:
            suffix = ", ".join(suffix_parts_by_index[idx])
            candidates.append(f"{resolved[idx]} ({suffix})" if suffix else resolved[idx])

        if len(set(candidates)) < len(candidates):
            for replicate_number, idx in enumerate(duplicate_indices, start=1):
                suffix_parts = suffix_parts_by_index[idx] + [f"rep {replicate_number}"]
                resolved[idx] = f"{resolved[idx]} ({', '.join(suffix_parts)})"
        else:
            for idx, candidate in zip(duplicate_indices, candidates):
                resolved[idx] = candidate

    return resolved


def _warn_duplicate_multiplot_labels(labels, options):
    option = options.get("deduplicate labels", False)
    if option not in (False, None):
        return
    if option is False and options.get("_deduplicate labels explicit", False):
        return

    counts = {}
    for label in labels:
        if label in (None, "", "_nolegend_"):
            continue
        counts[label] = counts.get(label, 0) + 1

    duplicates = [label for label, count in counts.items() if count > 1]
    if not duplicates:
        return

    print(
        "\033[93mWarning: Duplicate multiplot labels detected: "
        + ", ".join(f'"{label}"' for label in duplicates)
        + ". Use {'deduplicate labels': True} to append distinguishing metadata, "
        + "or set {'deduplicate labels': False} to suppress this warning.\033[0m"
    )


def _bounded_fontsize(text, default_size, min_size, max_size, ref_len):
    if text is None:
        return default_size
    n = max(len(str(text)), 1)
    scale = min(1.0, ref_len / n) ** 0.5
    size = default_size * scale
    return float(np.clip(size, min_size, max_size))

def _default_title_fontsize():
    return _active_plot_style_value("title fontsize") or 18

def _default_subtitle_fontsize():
    return _active_plot_style_value("subtitle fontsize") or 14

def _rc_fontsize_points(value, fallback):
    try:
        return float(value)
    except (TypeError, ValueError):
        pass
    try:
        return float(FontProperties(size=value).get_size_in_points())
    except (TypeError, ValueError):
        return fallback

def _default_legend_fontsize():
    style_size = _active_plot_style_value("legend fontsize")
    if style_size is not None:
        return style_size
    label_size = mpl.rcParams.get("axes.labelsize", None)
    legend_size = mpl.rcParams.get("legend.fontsize", label_size)
    return _rc_fontsize_points(label_size, _rc_fontsize_points(legend_size, 10))

def _resolve_title_fontsize(text):
    default = _default_title_fontsize()
    return _bounded_fontsize(text, default, max(8, default - 4), default + 2, ref_len=45)

def _resolve_subtitle_fontsize(text):
    default = _default_subtitle_fontsize()
    return _bounded_fontsize(text, default, max(7, default - 4), default + 2, ref_len=60)

def _legend_fontsize(labels, default_size=None, min_size=6, max_size=None):
    if default_size is None:
        default_size = _default_legend_fontsize()
    if max_size is None:
        max_size = default_size
    if not labels:
        return default_size
    longest = max(len(str(lbl)) for lbl in labels)
    n = len(labels)
    penalty = max(longest / 28, n / 6, 1)
    size = default_size / penalty**0.35
    return float(np.clip(size, min_size, max_size))

def _prepare_multiplot_style(echem_list, options):
    labels, title, subtitle, shared_compounds, similarities = (
        _resolve_multiplot_labels_title_subtitle(echem_list, options)
    )

    label_alterations = options.get("label alterations")

    display_labels = [
        apply_text_alterations(lbl, label_alterations)
        for lbl in labels
    ]
    display_labels = _deduplicate_multiplot_labels(echem_list, display_labels, options)
    _warn_duplicate_multiplot_labels(display_labels, options)

    plt.figure()
    ax = plt.gca()

    num_curves = len(echem_list)
    if num_curves == 1:
        options['legend'] = False

    title_fs = options.get("title fontsize")
    if title_fs in (None, "auto"):
        title_fs = _resolve_title_fontsize(title)

    subtitle_fs = options.get("subtitle fontsize")
    if subtitle_fs in (None, "auto"):
        subtitle_fs = _resolve_subtitle_fontsize(subtitle)

    color_spec = _resolve_multiplot_color_spec(echem_list, display_labels, options)
    user_labels_explicit = options.get("labels") is not None

    color_spec = _attach_gradient_legend_text(
        color_spec,
        display_labels,
        label_alterations=label_alterations,
        user_labels_explicit=user_labels_explicit,
    )

    return {
        "ax": ax,
        "labels": labels,
        "display labels": display_labels,
        "title": title,
        "subtitle": subtitle,
        "title fontsize": title_fs,
        "subtitle fontsize": subtitle_fs,
        "color spec": color_spec,
    }


def _finish_multiplot_style(echem_list, options, style):
    ax = style["ax"]
    title = style["title"]
    subtitle = style["subtitle"]
    color_spec = style["color spec"]
    num_curves = len(echem_list)

    plots = []
    _apply_plot_titles(
        ax.figure,
        ax,
        title,
        subtitle,
        style["title fontsize"],
        style["subtitle fontsize"],
    )

    resolved_legend_fs = options.get("legend fontsize")
    resolved_legend_loc = _normalize_legend_loc(options.get("legend loc", "best"))
    resolved_legend_outside = options.get("legend outside", False)

    if resolved_legend_fs in (None, "auto"):
        resolved_legend_fs, resolved_legend_loc, resolved_legend_outside = _resolve_adaptive_legend_layout(
            ax,
            color_spec,
            options,
        )

    if options.get("legend", True) and num_curves > 1:
        legend_options = options.copy()
        legend_options["legend fontsize"] = resolved_legend_fs
        legend_options["legend loc"] = resolved_legend_loc
        legend_options["legend outside"] = resolved_legend_outside

        _draw_multiplot_legend_and_colorbars(ax, color_spec, legend_options, resolved_legend_fs)

    _add_scale_bar(ax, options, unit=options.get("y unit"))
    return plots


def _plot_multiplot_series(echem_list, options, series_getter):
    options = _coerce_multiplot_options(options)
    style = _prepare_multiplot_style(echem_list, options)
    ax = style["ax"]
    color_spec = style["color spec"]
    plots = []

    for i, echem_object in enumerate(echem_list):
        x, y = series_getter(echem_object)
        line, = ax.plot(
            x,
            y + options.get("offset", 0) * i,
            color=color_spec["line colors"][i],
            label=color_spec["labels"][i],
        )
        _add_directional_arrows(ax, options, x, y, line_color=line.get_color())
        plots.append((line.figure, ax))

    _finish_multiplot_style(echem_list, options, style)
    return plots, style


def _plot_options_from_mapping(options):
    routed = {}
    for field in fields(PlotOptions):
        option_key = field.name.replace("_", " ")
        if option_key in options:
            routed[option_key] = options[option_key]
        elif field.name in options:
            routed[field.name] = options[field.name]
    return PlotOptions.from_options(routed).to_options_dict()


def _multiplot_options_from_mapping(options):
    routed = {}
    for field in fields(MultiplotOptions):
        option_key = field.name.replace("_", " ")
        if option_key in options:
            routed[option_key] = options[option_key]
        elif field.name in options:
            routed[field.name] = options[field.name]
    return MultiplotOptions.from_options(routed).to_options_dict()


def _is_simulated_cv_object(echem_object):
    return echem_object.__class__.__name__ == "SimulatedCV" and hasattr(echem_object, "backend_result")


def _coerce_multiplot_options(options):
    internal_flags = {}
    public_options = options
    if isinstance(options, dict):
        internal_flags = {
            key: value
            for key, value in options.items()
            if str(key).startswith("_")
        }
        public_options = {
            key: value
            for key, value in options.items()
            if not str(key).startswith("_")
        }
    coerced = MultiplotOptions.from_options(public_options).to_options_dict()
    coerced.update(internal_flags)
    return coerced


def multiplot(echem_list, options=None):
    """Plot a list of electrochemistry objects on one shared axes.
    
    Parameters
    ----------
    echem_list : sequence of echem
        Objects to plot together.
    options : dict or MultiplotOptions, optional
        Plotting, labeling, axis, legend, and normalization options. See ``e.describe_options("multiplot")``.
    
    Returns
    -------
    matplotlib.axes.Axes
        Axes containing the overlaid traces.
    
    Examples
    --------
    >>> e.multiplot(cvs, {"legend mode": "colorbar"})
    """
    options = _coerce_multiplot_options(options)

    if not isinstance(echem_list, list):
        print("Must provide a list of echem objects!")
        return []

    ip0_values = None
    ip0_axis = _is_ip0_y_axis(options.get("y axis", ""))
    has_existing_ip0_columns = ip0_axis and all(_has_ip0_column(obj) for obj in echem_list)
    if ip0_axis:
        if not has_existing_ip0_columns:
            ip0_values = _resolve_ip0_values(echem_list, options)
        options["y unit"] = None
        options["ylabel"] = options.get("ylabel") or "$i / i_p^0$"

    options['x unit'] = _axis_common_unit(
        echem_list,
        lambda e: (e.x(options).values, e.x(options).name),
        options.get('x unit', 'auto')
    )
    if not ip0_axis:
        options['y unit'] = _axis_common_unit(
            echem_list,
            lambda e: (e.y(options), e.y(options).name),
            options.get('y unit', 'auto')
        )

    style = _prepare_multiplot_style(echem_list, options)
    color_spec = style["color spec"]

    for i, echem_object in enumerate(echem_list):
        plot_options = options.copy()
        plot_options['legend'] = False
        plot_options["segment color mode"] = "off"
        plot_options["scale bar"] = False
        plot_options["offset"] *= i
        plot_options["color"] = color_spec["line colors"][i]
        plot_options["label"] = color_spec["labels"][i]
        plot_options["label alterations"] = None
        if ip0_axis:
            if ip0_values is not None:
                plot_options["ip0"] = ip0_values[i]
            else:
                plot_options.pop("ip0", None)
            plot_options.pop("non-catalytic current", None)
            plot_options.pop("non catalytic current", None)
            plot_options.pop("non-catalytic cv", None)
            plot_options.pop("non catalytic cv", None)
            plot_options.pop("non-catalytic cvs", None)
            plot_options.pop("non catalytic cvs", None)
            plot_options["y unit"] = None
            plot_options["ylabel"] = "$i / i_p^0$"

        trace_options = _plot_options_from_mapping(plot_options)
        simulation_linestyle = options.get("simulation linestyle")
        if simulation_linestyle is not None and _is_simulated_cv_object(echem_object):
            trace_options["linestyle"] = simulation_linestyle
        echem_object.plot(trace_options)

    _finish_multiplot_style(echem_list, options, style)

    if options.get("print"):
        print_options = options.copy()
        print_options["labels"] = style["display labels"]
        show_objects(echem_list, print_options)

    return style["ax"]




def _has_ip0_column(*args, **kwargs):
    from .analysis_cv import _has_ip0_column as impl
    return impl(*args, **kwargs)


def _is_ip0_y_axis(*args, **kwargs):
    from .analysis_cv import _is_ip0_y_axis as impl
    return impl(*args, **kwargs)


def _resolve_ip0_values(*args, **kwargs):
    from .analysis_cv import _resolve_ip0_values as impl
    return impl(*args, **kwargs)


def _format_fit_rate_metric_label(*args, **kwargs):
    from .analysis_batch import _format_fit_rate_metric_label as impl
    return impl(*args, **kwargs)


def _format_fit_rate_x_label(*args, **kwargs):
    from .analysis_batch import _format_fit_rate_x_label as impl
    return impl(*args, **kwargs)


def _format_y_mode_axis_label(*args, **kwargs):
    from .analysis_batch import _format_y_mode_axis_label as impl
    return impl(*args, **kwargs)


def _format_y_transform_axis_label(*args, **kwargs):
    from .analysis_batch import _format_y_transform_axis_label as impl
    return impl(*args, **kwargs)


def _inverse_transform_values(*args, **kwargs):
    from .analysis_batch import _inverse_transform_values as impl
    return impl(*args, **kwargs)


def _inverse_y_mode_values(*args, **kwargs):
    from .analysis_batch import _inverse_y_mode_values as impl
    return impl(*args, **kwargs)


def _normalize_transform_token(*args, **kwargs):
    from .analysis_batch import _normalize_transform_token as impl
    return impl(*args, **kwargs)


def _normalize_y_mode(*args, **kwargs):
    from .analysis_batch import _normalize_y_mode as impl
    return impl(*args, **kwargs)


def _transform_valid_mask(*args, **kwargs):
    from .analysis_batch import _transform_valid_mask as impl
    return impl(*args, **kwargs)


def _transform_values(*args, **kwargs):
    from .analysis_batch import _transform_values as impl
    return impl(*args, **kwargs)


from .objects import cv, dpv
from .reference import _format_reference_display

__all__ = [
    "ScatterFitResult",
    "multiplot",
    "multimultiplot",
    "multi_scatterplot",
    "plotting_style",
    "show",
    "show_groups",
    "show_objects",
]
