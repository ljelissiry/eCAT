"""Batch and advanced electrochemical analysis helpers."""

import ast
import inspect
from pprint import pformat

from .utils import *  # noqa: F401,F403
from .options import *  # noqa: F401,F403
from .analysis_cv import (
    _resolve_manual_ip0_values,
    _coerce_cv_list,
    _resolve_non_catalytic_cvs,
    _is_ip0_y_axis,
    _default_normalized_axis,
    _resolve_reference_cvs,
    _resolve_reference_ip0,
    _resolve_ip0_values,
    _apply_normalized_current_axis,
    _copy_cv_with_normalized_current_axis,
    _raw_current_columns,
)
from .plotting import (
    ScatterFitResult,
    _axis_common_unit,
    _align_multiline_legend_handles_to_first_line,
    _apply_matplotlib_axis_scales,
    _attach_scatter_fit_table,
    _finish_multiplot_style,
    _fit_options_from_analysis_options,
    _fit_table_from_fits,
    _multiplot_options_from_mapping,
    _plot_multi_scatter_trace,
    _prepare_multiplot_style,
    _pretty_table_header_html_label,
    _print_scatter_fit_statistics,
    _resolve_multiplot_labels_title_subtitle,
    _scatter_fit,
    _scatter_fit_row,
    _scatter_fit_table,
    _scatter_result_from_legacy,
    _scatterfit_legend_fontsize,
    _scatterfit_legend_requested,
    build_object_table,
    display_object_table,
    echem_similar_different,
    multiplot,
    pretty_table_column_label,
)
from .reference import midpoint_potential


_FIT_MODEL_ALIASES = {
    None: None,
    False: None,
    "": None,
    "none": None,
    "false": None,
    "linear": "linear",
    "line": "linear",
    "power": "power",
    "power_law": "power",
    "power law": "power",
    "generic_power": "power_offset",
    "generic power": "power_offset",
    "power_offset": "power_offset",
    "power offset": "power_offset",
    "offset_power": "power_offset",
    "offset power": "power_offset",
    "exponential": "exponential",
    "exp": "exponential",
    "michaelis_menten": "michaelis_menten",
    "michaelis menten": "michaelis_menten",
    "michaelis-menten": "michaelis_menten",
    "mm": "michaelis_menten",
    "logistic": "logistic",
    "sigmoid": "logistic",
}

_FIT_MODEL_PARAMETERS = {
    "linear": ("m", "b"),
    "power": ("A", "n"),
    "power_offset": ("b", "A", "n"),
    "exponential": ("b", "A", "k"),
    "michaelis_menten": ("Vmax", "Km"),
    "logistic": ("b", "L", "k", "x0"),
}

_FIT_MODEL_ALLOWED_FORMULA_FUNCTIONS = {
    "abs": np.abs,
    "exp": np.exp,
    "log": np.log,
    "log10": np.log10,
    "sqrt": np.sqrt,
    "sin": np.sin,
    "cos": np.cos,
    "tan": np.tan,
    "tanh": np.tanh,
    "maximum": np.maximum,
    "minimum": np.minimum,
}

_FIT_MODEL_ALLOWED_AST_NODES = (
    ast.Expression,
    ast.BinOp,
    ast.UnaryOp,
    ast.Call,
    ast.Name,
    ast.Load,
    ast.Constant,
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.Pow,
    ast.USub,
    ast.UAdd,
)


_RETIRED_MODEL_OPTION_KEYS = {
    "model params",
    "model equation",
    "model init",
    "model bounds",
    "model residual",
    "model maxfev",
}

_FIT_MODEL_BARE_OPTION_ALIASES = {
    "model": "fit model",
    "params": "fit params",
    "equation": "fit equation",
    "init": "fit init",
    "bounds": "fit bounds",
    "residual": "fit residual",
    "max evals": "fit max evals",
    "max_evals": "fit max evals",
    "maxfev": "fit max evals",
    "range": "fit range",
    "ranges": "fit ranges",
    "indices": "fit indices",
}


def _normalize_shared_fit_options(options=None, *, allow_bare_aliases=False):
    normalized = _legacy_normalize_option_keys(options or {})
    for key, value in (options or {}).items():
        if isinstance(key, str) and key.startswith("_"):
            normalized[key] = value
    retired = sorted(key for key in normalized if key in _RETIRED_MODEL_OPTION_KEYS)
    if retired:
        pretty = "', '".join(retired)
        raise OptionError(
            f"Retired fit option '{pretty}'. Use the matching 'fit ...' option name."
        )
    if allow_bare_aliases:
        for source, target in _FIT_MODEL_BARE_OPTION_ALIASES.items():
            source_key = source.replace("_", " ")
            if source_key in normalized and target not in normalized:
                normalized[target] = normalized[source_key]
    return normalized


def _looks_like_formula_model(model):
    if not isinstance(model, str):
        return False
    text = model.strip()
    if "=" in text:
        lhs, rhs = text.split("=", 1)
        text = rhs if lhs.strip().lower() == "y" else text
    return "x" in text and any(token in text for token in ["+", "-", "*", "/", "^", "(", ")"])


def _normalize_fit_model(model):
    if model is None or model is False:
        return None
    if callable(model):
        return model
    key = str(model).strip().lower().replace("_", " ")
    key = re.sub(r"\s+", " ", key)
    normalized = _FIT_MODEL_ALIASES.get(key)
    if normalized is None and _looks_like_formula_model(model):
        return str(model).strip()
    if normalized is None and key not in {"", "none", "false"}:
        supported = "linear, power, power offset, exponential, michaelis-menten, logistic, a callable, or a formula string"
        raise ValueError(f"Unknown fit model '{model}'. Supported models: {supported}.")
    return normalized


def _strip_formula_lhs(equation):
    equation = str(equation).strip()
    if "=" in equation:
        lhs, rhs = equation.split("=", 1)
        if lhs.strip().lower() == "y":
            return rhs.strip()
    return equation


def _normalize_formula_expression(equation):
    return _strip_formula_lhs(equation).replace("^", "**")


def _infer_formula_params(tree):
    names = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            name = node.id
            if name == "x" or name in _FIT_MODEL_ALLOWED_FORMULA_FUNCTIONS:
                continue
            if name not in names:
                names.append(name)
    return names


def _validate_formula_ast(tree):
    for node in ast.walk(tree):
        if not isinstance(node, _FIT_MODEL_ALLOWED_AST_NODES):
            raise ValueError(f"Unsupported expression element in custom fit model: {type(node).__name__}.")
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.func.id not in _FIT_MODEL_ALLOWED_FORMULA_FUNCTIONS:
                raise ValueError("Custom fit model formulas may only call supported math functions.")


def _compile_formula_model(equation, names):
    expression = _normalize_formula_expression(equation)
    tree = ast.parse(expression, mode="eval")
    _validate_formula_ast(tree)
    code = compile(tree, "<ecat fit_model formula>", "eval")

    def model_function(x, *params):
        namespace = {"x": x}
        namespace.update(_FIT_MODEL_ALLOWED_FORMULA_FUNCTIONS)
        namespace.update({name: value for name, value in zip(names, params)})
        return eval(code, {"__builtins__": {}}, namespace)

    return model_function


def _callable_model_param_names(model, options):
    override = options.get("fit params")
    if override is not None:
        return list(override)
    try:
        signature = inspect.signature(model)
    except (TypeError, ValueError) as exc:
        raise ValueError("Custom callable fit models require 'fit params' when their signature cannot be inspected.") from exc
    parameters = list(signature.parameters.values())
    if not parameters:
        raise ValueError("Custom callable fit models must accept x as their first argument.")
    names = []
    for parameter in parameters[1:]:
        if parameter.kind in {inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD}:
            raise ValueError("Custom callable fit models with *args/**kwargs require explicit 'fit params'.")
        names.append(parameter.name)
    if not names:
        raise ValueError("Custom callable fit models need at least one fitted parameter.")
    return names


def _formula_model_param_names(equation, options):
    override = options.get("fit params")
    if override is not None:
        return list(override)
    tree = ast.parse(_normalize_formula_expression(equation), mode="eval")
    _validate_formula_ast(tree)
    names = _infer_formula_params(tree)
    if not names:
        raise ValueError("Custom formula fit models need at least one fitted parameter.")
    return names


def _resolve_fit_model_spec(model, options):
    model = _normalize_fit_model(model)
    if model is None:
        return None
    if callable(model):
        names = _callable_model_param_names(model, options)
        return {
            "model": "custom",
            "names": tuple(names),
            "function": model,
            "equation": options.get("fit equation") or getattr(model, "__name__", "custom callable"),
        }
    if model in _FIT_MODEL_PARAMETERS:
        return {
            "model": model,
            "names": _FIT_MODEL_PARAMETERS[model],
            "function": _fit_model_function(model),
            "equation": _fit_model_equation(model),
        }
    if _looks_like_formula_model(model):
        names = _formula_model_param_names(model, options)
        return {
            "model": "custom",
            "names": tuple(names),
            "function": _compile_formula_model(model, names),
            "equation": options.get("fit equation") or str(model).strip(),
        }
    raise ValueError(f"Unknown fit model '{model}'.")


def _fit_model_function(model):
    if model == "linear":
        return lambda x, m, b: m * x + b
    if model == "power":
        return lambda x, A, n: A * np.power(x, n)
    if model == "power_offset":
        return lambda x, b, A, n: b + A * np.power(x, n)
    if model == "exponential":
        return lambda x, b, A, k: b + A * np.exp(k * x)
    if model == "michaelis_menten":
        return lambda x, Vmax, Km: Vmax * x / (Km + x)
    if model == "logistic":
        return lambda x, b, L, k, x0: b + L / (1 + np.exp(-k * (x - x0)))
    raise ValueError(f"Unknown fit model '{model}'.")


def _fit_model_equation(model):
    equations = {
        "linear": "y = m x + b",
        "power": "y = A x^n",
        "power_offset": "y = b + A x^n",
        "exponential": "y = b + A exp(k x)",
        "michaelis_menten": "y = Vmax x / (Km + x)",
        "logistic": "y = b + L / (1 + exp(-k (x - x0)))",
    }
    return equations[model]


def _fit_model_default_sig_figs():
    try:
        defaults = get_defaults("fit_rate")
        default = defaults.get("sig figs", defaults.get("sig_figs", 6))
    except Exception:
        default = 6
    return int(default or 6)


def _fit_model_sig_figs(value=None):
    if value in (None, "", "auto"):
        return _fit_model_default_sig_figs()
    return int(value)


def _fit_model_equation_value(value, sig_figs=None):
    sig_figs = _fit_model_sig_figs(sig_figs)
    text = _format_fit_model_display_value(value, sig_figs=sig_figs)
    if text == "-0":
        return "0"
    return text


def _fit_model_signed_term(value, term="", sig_figs=None):
    sig_figs = _fit_model_sig_figs(sig_figs)
    value = float(value)
    sign = "+" if value >= 0 else "-"
    magnitude = _fit_model_equation_value(abs(value), sig_figs=sig_figs)
    return f" {sign} {magnitude}{term}"


def _substitute_fit_model_parameters(expression, parameters):
    substituted = str(expression)
    for name in sorted(parameters, key=len, reverse=True):
        value = _fit_model_equation_value(parameters[name])
        substituted = re.sub(rf"\b{re.escape(str(name))}\b", value, substituted)
    return substituted


def _format_fit_model_fitted_equation(model_result):
    params = model_result.get("parameters", {})
    model = model_result.get("model")
    sig_figs = _fit_model_sig_figs(model_result.get("sig figs"))
    if model == "linear" and {"m", "b"}.issubset(params):
        return f"y = {_fit_model_equation_value(params['m'], sig_figs)}x{_fit_model_signed_term(params['b'], sig_figs=sig_figs)}"
    if model == "power" and {"A", "n"}.issubset(params):
        return f"y = {_fit_model_equation_value(params['A'], sig_figs)}x^{{{_fit_model_equation_value(params['n'], sig_figs)}}}"
    if model == "power_offset" and {"b", "A", "n"}.issubset(params):
        exponent = _fit_model_equation_value(params["n"], sig_figs)
        return (
            f"y = {_fit_model_equation_value(params['b'], sig_figs)}"
            f"{_fit_model_signed_term(params['A'], f'x^{{{exponent}}}', sig_figs=sig_figs)}"
        )
    if model == "exponential" and {"b", "A", "k"}.issubset(params):
        k_text = _fit_model_equation_value(params["k"], sig_figs)
        return (
            f"y = {_fit_model_equation_value(params['b'], sig_figs)}"
            f"{_fit_model_signed_term(params['A'], f'exp({k_text}x)', sig_figs=sig_figs)}"
        )
    if model == "michaelis_menten" and {"Vmax", "Km"}.issubset(params):
        return (
            f"y = {_fit_model_equation_value(params['Vmax'], sig_figs)}x / "
            f"({_fit_model_equation_value(params['Km'], sig_figs)} + x)"
        )
    if model == "logistic" and {"b", "L", "k", "x0"}.issubset(params):
        k_text = _fit_model_equation_value(params["k"], sig_figs)
        x0_text = _fit_model_equation_value(params["x0"], sig_figs)
        return (
            f"y = {_fit_model_equation_value(params['b'], sig_figs)}"
            f"{_fit_model_signed_term(params['L'], f'/(1 + exp(-{k_text}(x - {x0_text})))', sig_figs=sig_figs)}"
        )
    equation = model_result.get("equation", "")
    if params and isinstance(equation, str):
        rhs = _substitute_fit_model_parameters(_strip_formula_lhs(equation), params)
        if "x" in rhs:
            return f"y = {rhs}"
    if params:
        assignments = ", ".join(
            f"{name}={_fit_model_equation_value(value)}"
            for name, value in params.items()
        )
        return f"{equation}({assignments})" if equation else assignments
    return equation


def _fit_model_auto_init(model, x, y, names=None):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    finite_x = x[np.isfinite(x)]
    finite_y = y[np.isfinite(y)]
    x_span = float(np.nanmax(finite_x) - np.nanmin(finite_x)) if len(finite_x) else 1.0
    if not np.isfinite(x_span) or x_span == 0:
        x_span = 1.0
    y_min = float(np.nanmin(finite_y)) if len(finite_y) else 0.0
    y_max = float(np.nanmax(finite_y)) if len(finite_y) else 1.0
    y_span = y_max - y_min
    if not np.isfinite(y_span) or y_span == 0:
        y_span = max(abs(y_max), 1.0)

    if model == "linear":
        if len(x) >= 2:
            m, b = np.polyfit(x, y, 1)
            return [float(m), float(b)]
        return [0.0, y_min]

    if model in {"power", "power_offset"}:
        offset = 0.0
        y_for_power = y
        if model == "power_offset":
            offset = y_min
            y_for_power = y - offset
        mask = (x > 0) & (y_for_power > 0) & np.isfinite(x) & np.isfinite(y_for_power)
        if np.count_nonzero(mask) >= 2:
            n, log_A = np.polyfit(np.log(x[mask]), np.log(y_for_power[mask]), 1)
            A = float(np.exp(log_A))
        else:
            n = 1.0
            x_ref = float(np.nanmax(np.abs(x))) if len(x) else 1.0
            A = y_span / max(x_ref, 1e-30)
        if model == "power":
            return [float(A), float(n)]
        return [float(offset), float(A), float(n)]

    if model == "exponential":
        b = y_min
        A = y_span
        shifted = y - b
        mask = (shifted > 0) & np.isfinite(x) & np.isfinite(shifted)
        if np.count_nonzero(mask) >= 2:
            k, log_A = np.polyfit(x[mask], np.log(shifted[mask]), 1)
            A = float(np.exp(log_A))
        else:
            k = 1.0 / x_span
        return [float(b), float(A), float(k)]

    if model == "michaelis_menten":
        vmax = y_max
        half = vmax / 2
        idx = int(np.nanargmin(np.abs(y - half))) if len(y) else 0
        km = float(abs(x[idx])) if len(x) else 1.0
        if not np.isfinite(km) or km <= 0:
            km = float(np.nanmedian(np.abs(x[x != 0]))) if np.any(x != 0) else 1.0
        return [float(vmax), float(km)]

    if model == "logistic":
        b = y_min
        L = y_span
        half = b + L / 2
        idx = int(np.nanargmin(np.abs(y - half))) if len(y) else 0
        x0 = float(x[idx]) if len(x) else 0.0
        k = 4.0 / x_span
        return [float(b), float(L), float(k), float(x0)]

    if model == "custom":
        return [1.0] * len(names or [])

    raise ValueError(f"Unknown fit model '{model}'.")


def _fit_model_auto_bounds(model, x, y, names=None):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    finite_x = x[np.isfinite(x)]
    finite_y = y[np.isfinite(y)]
    x_min = float(np.nanmin(finite_x)) if len(finite_x) else -np.inf
    x_max = float(np.nanmax(finite_x)) if len(finite_x) else np.inf
    x_span = x_max - x_min
    if not np.isfinite(x_span) or x_span == 0:
        x_span = 1.0
    y_abs = float(np.nanmax(np.abs(finite_y))) if len(finite_y) else 1.0
    if not np.isfinite(y_abs) or y_abs == 0:
        y_abs = 1.0
    broad_y = 100.0 * y_abs

    if model == "linear":
        return [-np.inf, -np.inf], [np.inf, np.inf]
    if model == "power":
        return [-np.inf, -10.0], [np.inf, 10.0]
    if model == "power_offset":
        return [-broad_y, -np.inf, -10.0], [broad_y, np.inf, 10.0]
    if model == "exponential":
        return [-broad_y, -np.inf, -np.inf], [broad_y, np.inf, np.inf]
    if model == "michaelis_menten":
        return [0.0, 0.0], [np.inf, np.inf]
    if model == "logistic":
        return [-broad_y, -np.inf, -np.inf, x_min - x_span], [broad_y, np.inf, np.inf, x_max + x_span]
    if model == "custom":
        count = len(names or [])
        return [-np.inf] * count, [np.inf] * count
    raise ValueError(f"Unknown fit model '{model}'.")


def _normalize_model_vector_override(spec, names, default, *, option_name):
    if spec is None or (isinstance(spec, str) and spec == "auto"):
        return list(default)
    if isinstance(spec, dict):
        values = list(default)
        for i, name in enumerate(names):
            if name in spec:
                values[i] = spec[name]
        return values
    values = list(spec)
    if len(values) != len(names):
        raise ValueError(f"'{option_name}' for this model must have {len(names)} values.")
    return values


def _normalize_model_bounds(spec, names, default_bounds):
    lower_default, upper_default = default_bounds
    if spec is None or (isinstance(spec, str) and spec == "auto"):
        return list(lower_default), list(upper_default)
    lower = list(lower_default)
    upper = list(upper_default)
    if isinstance(spec, dict):
        for i, name in enumerate(names):
            if name in spec:
                lo, hi = spec[name]
                lower[i] = lo
                upper[i] = hi
        return lower, upper
    if len(spec) != 2:
        raise ValueError("'fit bounds' must be [lower, upper] or a dict by parameter name.")
    lo_spec, hi_spec = spec
    if np.isscalar(lo_spec):
        lower = [lo_spec] * len(names)
    else:
        lower = list(lo_spec)
    if np.isscalar(hi_spec):
        upper = [hi_spec] * len(names)
    else:
        upper = list(hi_spec)
    if len(lower) != len(names) or len(upper) != len(names):
        raise ValueError(f"'fit bounds' for this model must have {len(names)} lower and upper values.")
    return lower, upper


def _model_fit_row(
    label,
    model,
    parameter_values,
    parameter_errors,
    stats,
    fit_x=None,
    equation=None,
    fit_equation=None,
):
    rows = []
    for name, value in parameter_values.items():
        row = {
            "series": label,
            "model": model,
            "parameter": name,
            "value": float(value),
            "equation": equation or _fit_model_equation(model),
        }
        if fit_equation is not None:
            row["fit equation"] = fit_equation
        if parameter_errors and name in parameter_errors:
            row["stderr"] = parameter_errors[name]
        row.update(stats)
        if model == "linear" and {"m", "b"}.issubset(parameter_values):
            row["Fit"] = (
                f"y = {float(parameter_values['m']):.4g}x "
                f"{float(parameter_values['b']):+.4g}"
            )
            row["slope"] = float(parameter_values["m"])
            row["intercept"] = float(parameter_values["b"])
        if fit_x is not None:
            fit_x = np.asarray(fit_x, dtype=float)
            fit_x = fit_x[np.isfinite(fit_x)]
            if len(fit_x) > 0:
                row["fit x min"] = float(np.nanmin(fit_x))
                row["fit x max"] = float(np.nanmax(fit_x))
        rows.append(row)
    return rows


def _fit_model_indices_mask(length, fit_indices):
    mask = np.zeros(length, dtype=bool)
    if fit_indices is None:
        mask[:] = True
        return mask
    if isinstance(fit_indices, slice):
        mask[np.arange(length)[fit_indices]] = True
        return mask

    fit_indices_array = np.asarray(fit_indices, dtype=object)

    if fit_indices_array.dtype == bool or (
        fit_indices_array.ndim == 1
        and len(fit_indices_array) == length
        and all(isinstance(value, (bool, np.bool_)) for value in fit_indices_array)
    ):
        bool_mask = np.asarray(fit_indices, dtype=bool)
        if len(bool_mask) != length:
            raise ValueError("'fit indices' boolean masks must match the data length.")
        return bool_mask

    if fit_indices_array.ndim == 1:
        if len(fit_indices_array) == 2:
            start, stop = int(fit_indices_array[0]), int(fit_indices_array[1])
            mask[np.arange(length)[start:stop]] = True
            return mask
        mask[np.asarray(fit_indices_array, dtype=int)] = True
        return mask

    if fit_indices_array.ndim == 2 and fit_indices_array.shape[1] == 2:
        base_indices = np.arange(length)
        for start, stop in fit_indices_array:
            mask[base_indices[int(start):int(stop)]] = True
        return mask

    raise ValueError(
        "'fit indices' should be [start, stop], [[start, stop], ...], "
        "a boolean mask, or explicit integer indices."
    )


def _fit_model_range_mask(x, fit_range):
    x = np.asarray(x, dtype=float)
    mask = np.zeros(len(x), dtype=bool)
    if fit_range is None:
        mask[:] = True
        return mask
    if not isinstance(fit_range, (list, tuple)) or len(fit_range) != 2:
        raise ValueError("'fit range' for fit_model should be [x_min, x_max].")
    lower, upper = fit_range
    mask[:] = np.isfinite(x)
    if lower is not None:
        mask &= x >= float(lower)
    if upper is not None:
        mask &= x <= float(upper)
    return mask


def _fit_model_selection_mask(x, options):
    options = {} if options is None else dict(options)
    mask = np.ones(len(x), dtype=bool)
    if options.get("fit range") is not None:
        mask &= _fit_model_range_mask(x, options.get("fit range"))
    if options.get("fit indices") is not None:
        mask &= _fit_model_indices_mask(len(x), options.get("fit indices"))
    return mask


def _fit_model_xy(x, y, model="power", options=None, label=None):
    options = _normalize_shared_fit_options(options)
    model_spec = _resolve_fit_model_spec(model if model is not None else options.get("fit model", "power"), options)
    if model_spec is None:
        raise ValueError("A fit model is required.")
    model = model_spec["model"]

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    keep = np.isfinite(x) & np.isfinite(y)
    private_fit_mask = options.get("_fit model mask")
    if private_fit_mask is not None:
        private_fit_mask = np.asarray(private_fit_mask, dtype=bool)
        if len(private_fit_mask) != len(x):
            raise ValueError("Internal fit_model mask length must match x/y data.")
        keep &= private_fit_mask
    x_fit = x[keep]
    y_fit = y[keep]
    if len(x_fit) < 2:
        raise ValueError("fit_model requires at least two finite x/y points.")

    names = model_spec["names"]
    p0 = _normalize_model_vector_override(
        options.get("fit init", "auto"),
        names,
        _fit_model_auto_init(model, x_fit, y_fit, names=names),
        option_name="fit init",
    )
    lower, upper = _normalize_model_bounds(
        options.get("fit bounds", "auto"),
        names,
        _fit_model_auto_bounds(model, x_fit, y_fit, names=names),
    )

    residual = str(options.get("fit residual", "direct")).strip().lower().replace("_", " ")
    func = model_spec["function"]
    sigma = None
    fit_target = y_fit
    fit_func = func
    if residual in {"direct", "ordinary"}:
        pass
    elif residual in {"relative", "fractional"}:
        sigma = np.maximum(np.abs(y_fit), np.finfo(float).eps)
    elif residual in {"log", "log y", "log10"}:
        if np.any(y_fit <= 0):
            raise ValueError("Log model residual requires positive y values.")

        def fit_func(x_values, *params):
            predicted = func(x_values, *params)
            if np.any(predicted <= 0):
                return np.full_like(np.asarray(x_values, dtype=float), np.inf, dtype=float)
            if residual == "log10":
                return np.log10(predicted)
            return np.log(predicted)

        fit_target = np.log10(y_fit) if residual == "log10" else np.log(y_fit)
    else:
        raise ValueError("'fit residual' must be direct, relative, log, or log10.")

    popt, pcov = curve_fit(
        fit_func,
        x_fit,
        fit_target,
        p0=np.asarray(p0, dtype=float),
        bounds=(np.asarray(lower, dtype=float), np.asarray(upper, dtype=float)),
        sigma=sigma,
        maxfev=int(options.get("fit max evals", 10000)),
    )

    predicted = func(x, *popt)
    predicted_fit = func(x_fit, *popt)
    residuals = y - predicted
    fit_residuals = y_fit - predicted_fit
    ss_res = float(np.nansum(fit_residuals ** 2))
    ss_tot = float(np.nansum((y_fit - np.nanmean(y_fit)) ** 2))
    r2 = np.nan if ss_tot == 0 else 1 - ss_res / ss_tot
    rmse = float(np.sqrt(np.nanmean(fit_residuals ** 2)))
    stats = {"R2": r2, "RMSE": rmse, "Fit Points": int(len(x_fit))}

    parameters = {name: float(value) for name, value in zip(names, popt)}
    sig_figs = _fit_model_sig_figs(options.get("sig figs"))
    errors = {}
    if pcov is not None and np.ndim(pcov) == 2:
        diag = np.diag(pcov)
        if len(diag) == len(names):
            errors = {
                name: float(np.sqrt(value)) if np.isfinite(value) and value >= 0 else np.nan
                for name, value in zip(names, diag)
            }

    fit_equation = _format_fit_model_fitted_equation(
        {
            "model": model,
            "parameters": parameters,
            "equation": model_spec["equation"],
            "sig figs": sig_figs,
        }
    )
    fit_rows = _model_fit_row(
        label or f"{model} fit",
        model,
        parameters,
        errors,
        stats,
        fit_x=x_fit,
        equation=model_spec["equation"],
        fit_equation=fit_equation,
    )
    return {
        "model": model,
        "names": names,
        "initial": {name: float(value) for name, value in zip(names, p0)},
        "bounds": {
            name: (float(lo), float(hi))
            for name, lo, hi in zip(names, lower, upper)
        },
        "residual": residual,
        "parameters": parameters,
        "errors": errors,
        "stats": stats,
        "equation": model_spec["equation"],
        "fit equation": fit_equation,
        "function": func,
        "sig figs": sig_figs,
        "x": x,
        "y": y,
        "keep": keep,
        "predicted": predicted,
        "residuals": residuals,
        "fit_rows": fit_rows,
        "popt": popt,
    }


def _fit_model_summary(model_result):
    return {
        "model": model_result["model"],
        "parameters": model_result["parameters"],
        "errors": model_result["errors"],
        "stats": model_result["stats"],
    }


def _fit_series_xy(x, y, *, options=None, label="Model", model=None):
    options = _normalize_shared_fit_options(options)
    fit_model = model if model is not None else options.get("fit model", "linear")
    model_result = _fit_model_xy(
        x,
        y,
        model=fit_model,
        options=options,
        label=label,
    )
    return {
        "model_result": model_result,
        "fits": _fit_model_summary(model_result),
        "fit_rows": list(model_result["fit_rows"]),
    }


def _print_fit_rate_model_results(fit_model_results, options):
    _print_fit_model_results(fit_model_results, options)


def _empty_scatter_fit_result(table, *, summary=None):
    fit_table = _scatter_fit_table([])
    if isinstance(table, pd.DataFrame):
        table.attrs["fit table"] = fit_table
    return ScatterFitResult(
        table=table,
        fits={},
        fit_table=fit_table,
        fit_model_results={},
        summary=summary or {},
        legacy_return=(table, {}),
    )


def _fit_model_input_xy(x_or_result, y=None, options=None):
    options = {} if options is None else dict(options)
    if y is not None:
        return np.asarray(x_or_result, dtype=float), np.asarray(y, dtype=float), "x", "y"

    table = x_or_result.table if isinstance(x_or_result, ScatterFitResult) else x_or_result
    if not isinstance(table, pd.DataFrame):
        raise TypeError("fit_model accepts x/y arrays, a pandas DataFrame, or a ScatterFitResult.")

    x_col = options.get("x column", "auto")
    y_col = options.get("y column", "auto")
    numeric_cols = [col for col in table.columns if pd.api.types.is_numeric_dtype(table[col])]
    if x_col in (None, "auto"):
        for candidate in ["x raw", "x", "Scan Rate", "Substrate Concentration (M)"]:
            if candidate in table.columns:
                x_col = candidate
                break
        else:
            if len(numeric_cols) < 2:
                raise ValueError("Could not auto-resolve an x column for fit_model.")
            x_col = numeric_cols[0]
    if y_col in (None, "auto"):
        for candidate in ["y adjusted", "y raw", "y transformed", "kobs", "ip", "Ep"]:
            if candidate in table.columns:
                y_col = candidate
                break
        else:
            choices = [col for col in numeric_cols if col != x_col]
            if not choices:
                raise ValueError("Could not auto-resolve a y column for fit_model.")
            y_col = choices[0]

    return (
        np.asarray(table[x_col], dtype=float),
        np.asarray(table[y_col], dtype=float),
        str(x_col),
        str(y_col),
    )


def _plot_fit_model_result(model_result, options=None):
    options = {} if options is None else dict(options)
    ax = plt.gca()
    new_plot = options.get("new plot", True)
    if new_plot:
        plt.figure()
        ax = plt.gca()
    x = model_result["x"]
    y = model_result["y"]
    plot_data = options.get("plot data")
    if plot_data is None:
        plot_data = bool(new_plot)
    data_artist = None
    if plot_data:
        data_artist = ax.scatter(x, y, label=options.get("data label", "Data"))
    finite_x = x[np.isfinite(x)]
    if len(finite_x) > 0:
        x_line = np.linspace(float(np.nanmin(finite_x)), float(np.nanmax(finite_x)), 300)
    else:
        x_line = x
    y_line = model_result["function"](x_line, *model_result["popt"])
    fit_label_opt = options.get("fit label", False)
    label = None
    if isinstance(fit_label_opt, str):
        label = fit_label_opt
    elif fit_label_opt is True:
        label = options.get("model label")
        if label is None:
            label = f"{model_result['model'].replace('_', ' ').title()} Fit"
        r2 = model_result.get("stats", {}).get("R2")
        if r2 is not None and np.isfinite(r2):
            equation = options.get("fit equation label")
            if equation is None:
                equation = model_result.get("fit equation") or model_result["equation"]
            r2_text = _format_fit_model_display_value(
                r2,
                sig_figs=model_result.get("sig figs"),
            )
            label = f"{label}\n${equation}$\n$R^2 = {r2_text}$"
    ax.plot(
        x_line,
        y_line,
        label=label,
        color=_fit_color_from_options(options, index=0, fallback=_artist_color(data_artist) or "tab:red"),
        linestyle=options.get("fit linestyle", "--"),
        linewidth=options.get("fit linewidth", 1),
        alpha=options.get("fit alpha", 1),
    )
    if options.get("x label"):
        ax.set_xlabel(options["x label"])
    if options.get("y label"):
        ax.set_ylabel(options["y label"])
    legend_requested = _scatterfit_legend_requested(options)
    if fit_label_opt and options.get("legend", None) is not False:
        legend_requested = True
    if legend_requested:
        legend = ax.legend(fontsize=_scatterfit_legend_fontsize(options))
        _align_multiline_legend_handles_to_first_line(legend)
    return ax


def _format_fit_model_display_value(value, sig_figs=None):
    sig_figs = _fit_model_sig_figs(sig_figs)
    if value is None:
        return ""
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        if np.isnan(value):
            return "nan"
        if np.isposinf(value):
            return "inf"
        if np.isneginf(value):
            return "-inf"
        return f"{float(value):.{int(sig_figs)}g}"
    return str(value)


def _format_fit_model_value_with_error(value, error=None, sig_figs=None):
    sig_figs = _fit_model_sig_figs(sig_figs)
    value_text = _format_fit_model_display_value(value, sig_figs=sig_figs)
    if error is None:
        return value_text
    try:
        error = float(error)
    except (TypeError, ValueError):
        return value_text
    if not np.isfinite(error):
        return value_text
    return f"{value_text} ± {_format_fit_model_display_value(error, sig_figs=sig_figs)}"


def _display_or_print_fit_model_table(table):
    if display is not None:
        with pd.option_context("display.max_columns", None):
            display(table)
    else:
        print(table.to_string(index=False, justify="left"))


def _fit_model_details_display_table(model_result):
    stats = model_result["stats"]
    sig_figs = _fit_model_sig_figs(model_result.get("sig figs"))
    x = np.asarray(model_result["x"], dtype=float)
    finite_x = x[np.isfinite(x)]
    if len(finite_x) > 0:
        x_range = (
            f"{_format_fit_model_display_value(float(np.nanmin(finite_x)), sig_figs=sig_figs)} to "
            f"{_format_fit_model_display_value(float(np.nanmax(finite_x)), sig_figs=sig_figs)}"
        )
    else:
        x_range = ""
    rows = [
        ("Model", model_result["model"]),
        ("Equation", model_result["equation"]),
        ("Residual", model_result.get("residual", "")),
        ("X Range", x_range),
        ("Fit Points", stats.get("Fit Points")),
        ("R2", stats.get("R2")),
        ("RMSE", stats.get("RMSE")),
    ]
    return pd.DataFrame(
        [
            {"Setting": setting, "Value": _format_fit_model_display_value(value, sig_figs=sig_figs)}
            for setting, value in rows
        ]
    )


def _fit_model_display_table(model_result):
    stats = model_result["stats"]
    sig_figs = _fit_model_sig_figs(model_result.get("sig figs"))
    x = np.asarray(model_result["x"], dtype=float)
    finite_x = x[np.isfinite(x)]
    if len(finite_x) > 0:
        x_range = (
            f"{_format_fit_model_display_value(float(np.nanmin(finite_x)), sig_figs=sig_figs)} to "
            f"{_format_fit_model_display_value(float(np.nanmax(finite_x)), sig_figs=sig_figs)}"
        )
    else:
        x_range = ""

    rows = [
        ("Model", model_result["model"]),
        ("Equation", model_result["equation"]),
        ("Residual", model_result.get("residual", "")),
        ("X Range", x_range),
        ("Fit Points", stats.get("Fit Points")),
        ("R²", stats.get("R2")),
        ("RMSE", stats.get("RMSE")),
    ]
    for name in model_result["names"]:
        rows.append(
            (
                name,
                _format_fit_model_value_with_error(
                    model_result["parameters"].get(name),
                    model_result.get("errors", {}).get(name),
                    sig_figs=sig_figs,
                ),
            )
        )
    return pd.DataFrame(
        [
            {"Field": field, "Value": _format_fit_model_display_value(value, sig_figs=sig_figs)}
            for field, value in rows
        ]
    )


def _fit_model_parameters_display_table(model_result):
    rows = []
    sig_figs = _fit_model_sig_figs(model_result.get("sig figs"))
    for name in model_result["names"]:
        lower, upper = model_result["bounds"].get(name, (None, None))
        rows.append(
            {
                "Parameter": name,
                "Initial": _format_fit_model_display_value(model_result["initial"].get(name), sig_figs=sig_figs),
                "Lower Bound": _format_fit_model_display_value(lower, sig_figs=sig_figs),
                "Upper Bound": _format_fit_model_display_value(upper, sig_figs=sig_figs),
                "Fit Value": _format_fit_model_display_value(model_result["parameters"].get(name), sig_figs=sig_figs),
                "Std. Error": _format_fit_model_display_value(model_result["errors"].get(name), sig_figs=sig_figs),
            }
        )
    return pd.DataFrame(rows)


def _fit_model_summary_dict(model_result):
    return {
        "model": model_result["model"],
        "equation": model_result["equation"],
        "fit_equation": model_result.get("fit equation"),
        "residual": model_result.get("residual"),
        "x_range": [
            float(np.nanmin(model_result["x"])),
            float(np.nanmax(model_result["x"])),
        ],
        "fit_points": model_result["stats"].get("Fit Points"),
        "parameters": model_result["parameters"],
        "stderr": model_result["errors"],
        "stats": {
            "r2": model_result["stats"].get("R2"),
            "rmse": model_result["stats"].get("RMSE"),
        },
    }


def _is_auto_fit_value(value):
    return value is None or (isinstance(value, str) and value.strip().lower() == "auto")


def _fit_model_param_near_bound(model_result, rtol=1e-6, atol=1e-12):
    for name, value in model_result.get("parameters", {}).items():
        bounds = model_result.get("bounds", {}).get(name)
        if bounds is None:
            continue
        lower, upper = bounds
        try:
            value = float(value)
            lower = float(lower)
            upper = float(upper)
        except (TypeError, ValueError):
            continue
        if np.isfinite(lower) and np.isclose(value, lower, rtol=rtol, atol=atol):
            return True
        if np.isfinite(upper) and np.isclose(value, upper, rtol=rtol, atol=atol):
            return True
    return False


def _fit_model_print_mode(model_result, options):
    mode = options.get("print fit", options.get("fit print", "auto"))
    if options.get("print fit details", False):
        mode = "details"
    if isinstance(mode, bool):
        mode = "details" if mode else "summary"
    mode = str(mode).strip().lower().replace("_", " ").replace("-", " ")
    if mode in {"detail", "detailed", "full", "two table", "two tables"}:
        return "details"
    if mode in {"summary", "summarize", "compact", "one table", "one"}:
        return "summary"
    if mode not in {"auto", ""}:
        raise ValueError("'print fit' must be 'auto', 'summary', or 'details'.")

    fit_init = options.get("fit init", "auto")
    fit_bounds = options.get("fit bounds", "auto")
    model = model_result.get("model")
    if not _is_auto_fit_value(fit_init):
        return "details"
    if not _is_auto_fit_value(fit_bounds):
        return "details"
    if model in {"custom", "power_offset", "logistic", "michaelis_menten"}:
        return "details"
    if len(model_result.get("names", ())) >= 3:
        return "details"
    if _fit_model_param_near_bound(model_result):
        return "details"
    return "summary"


def _print_fit_model_result(model_result, options=None):
    options = {} if options is None else dict(options)
    if options.get("pretty print", True):
        mode = _fit_model_print_mode(model_result, options)
        if mode == "details":
            print("Fit Model Details:")
            _display_or_print_fit_model_table(_fit_model_details_display_table(model_result))
            print("\nFit Model Parameters:")
            _display_or_print_fit_model_table(_fit_model_parameters_display_table(model_result))
        else:
            print("Fit Model:")
            _display_or_print_fit_model_table(_fit_model_display_table(model_result))
    else:
        print("Fit Model Summary:")
        print(pformat(_fit_model_summary_dict(model_result), sort_dicts=False))


def _combine_fit_model_tables(fit_model_results, table_builder, label_column):
    ordered_labels = []
    per_series = {}
    for label, model_result in fit_model_results.items():
        table = table_builder(model_result)
        key_col = table.columns[0]
        value_col = table.columns[1]
        mapping = {}
        for _, row in table.iterrows():
            key = row[key_col]
            if key not in ordered_labels:
                ordered_labels.append(key)
            mapping[key] = row[value_col]
        per_series[str(label)] = mapping

    rows = []
    for key in ordered_labels:
        row = {label_column: key}
        for label, mapping in per_series.items():
            row[label] = mapping.get(key, "")
        rows.append(row)
    return pd.DataFrame(rows)


def _fit_model_multi_parameters_display_table(fit_model_results):
    rows = []
    for label, model_result in fit_model_results.items():
        table = _fit_model_parameters_display_table(model_result).copy()
        table.insert(0, "Series", str(label))
        rows.append(table)
    if not rows:
        return pd.DataFrame(
            columns=["Series", "Parameter", "Initial", "Lower Bound", "Upper Bound", "Fit Value", "Std. Error"]
        )
    return pd.concat(rows, ignore_index=True)


def _print_fit_model_results(fit_model_results, options=None):
    options = {} if options is None else dict(options)
    if not fit_model_results:
        return

    if len(fit_model_results) == 1:
        _print_fit_model_result(next(iter(fit_model_results.values())), options)
        return

    if not options.get("pretty print", True):
        print("Fit Model Summary:")
        print(
            pformat(
                {
                    str(label): _fit_model_summary_dict(model_result)
                    for label, model_result in fit_model_results.items()
                },
                sort_dicts=False,
            )
        )
        return

    needs_details = any(
        _fit_model_print_mode(model_result, options) == "details"
        for model_result in fit_model_results.values()
    )

    if needs_details:
        print("Fit Model Details:")
        _display_or_print_fit_model_table(
            _combine_fit_model_tables(
                fit_model_results,
                _fit_model_details_display_table,
                "Setting",
            )
        )
        print("\nFit Model Parameters:")
        _display_or_print_fit_model_table(
            _fit_model_multi_parameters_display_table(fit_model_results)
        )
        return

    print("Fit Model:")
    _display_or_print_fit_model_table(
        _combine_fit_model_tables(
            fit_model_results,
            _fit_model_display_table,
            "Field",
        )
    )


def fit_model(x_or_result, y=None, model=None, options=None):
    """Fit a named nonlinear or linear model to scatter data.

    Parameters
    ----------
    x_or_result : array-like, pandas.DataFrame, or ScatterFitResult
        x data, a table containing x/y columns, or a previous scatter-fit result.
    y : array-like, optional
        y data when ``x_or_result`` is an x array.
    model : str or callable, optional
        Model name: linear, power, power offset, exponential, michaelis-menten, or logistic.
    options : dict, optional
        Supports ``plot``, ``print``, ``x column``, ``y column``, ``fit model``,
        ``fit init``, ``fit bounds``, ``fit residual``, and standard fit style
        options. Because this function is already a fit context, bare aliases
        such as ``init`` and ``bounds`` are also accepted.

    Returns
    -------
    ScatterFitResult
        Table with predictions/residuals, fitted parameter table, and summary metadata.
    """
    options = _normalize_shared_fit_options(options, allow_bare_aliases=True)
    x, y_values, x_label, y_label = _fit_model_input_xy(x_or_result, y, options)
    fit_model_name = model if model is not None else options.get("fit model", "linear")
    if not options.get("fit", True):
        table = pd.DataFrame({x_label: x, y_label: y_values})
        return _empty_scatter_fit_result(
            table,
            summary={"analysis": "model fit", "model": None, "fit": False},
        )
    fit_options = dict(options)
    fit_options["_fit model mask"] = _fit_model_selection_mask(x, options)
    model_result = _fit_model_xy(
        x,
        y_values,
        model=fit_model_name,
        options=fit_options,
        label=options.get("series", options.get("model label", "Model")),
    )
    table = pd.DataFrame(
        {
            x_label: x,
            y_label: y_values,
            "Predicted": model_result["predicted"],
            "Residual": model_result["residuals"],
        }
    )
    fit_table = _scatter_fit_table(model_result["fit_rows"])
    table.attrs["fit table"] = fit_table

    if options.get("plot", False):
        plot_options = dict(options)
        plot_options.setdefault("x label", x_label)
        plot_options.setdefault("y label", y_label)
        _plot_fit_model_result(model_result, plot_options)

    if options.get("print", False):
        _print_fit_model_result(model_result, options)

    fits = {
        "model": model_result["model"],
        "parameters": model_result["parameters"],
        "errors": model_result["errors"],
        "stats": model_result["stats"],
    }
    return ScatterFitResult(
        table=table,
        fits=fits,
        fit_table=fit_table,
        fit_model_results={options.get("series", options.get("model label", "Model")): model_result},
        summary={
            "analysis": "model fit",
            "model": model_result["model"],
            "equation": model_result["equation"],
            "fit equation": model_result.get("fit equation"),
            "parameters": model_result["parameters"],
            "stats": model_result["stats"],
        },
        legacy_return=(table, fits),
    )

def _trumpet_analysis_legacy_return(cvs, options={}):
    raw_options = options
    typed_options = TrumpetAnalysisOptions.from_options(options)
    options = typed_options.to_legacy_dict()
    if not options["plot"]:
        options["plot fit"] = False

    base_segment = int(options["segment"])
    paired_segment = base_segment + 1

    if options.get("plot all", False):
        multiplot(cvs, _multiplot_options_from_mapping(options))

    half_wave_options = options.copy()
    half_wave_options["plot"] = False
    half_wave_options["print"] = False
    half_wave_options["plot all"] = False
    half_wave_options["print all"] = False
    half_wave_options["internal call"] = True
    half_wave_options["new plot"] = False

    deltas, scan_rates, ep1_values, ep2_values = [], [], [], []
    for cv in cvs:
        half_wave = cv.half_wave_potential(half_wave_options)
        scan_rates.append(float(cv.scan_rate))
        deltas.append(half_wave["ΔE"])
        ep1_values.append(half_wave["peak 1"]["Ep"])
        ep2_values.append(half_wave["peak 2"]["Ep"])

    log_scan_rates = np.log10(np.asarray(scan_rates, dtype=float))
    ep1_values = np.asarray(ep1_values, dtype=float)
    ep2_values = np.asarray(ep2_values, dtype=float)

    data = pd.DataFrame(
        {
            "Scan Rates (V/s)": scan_rates,
            "Log(Scan Rates (V/s))": log_scan_rates,
            f"Seg {base_segment} Peak Potential (V)": ep1_values,
            f"Seg {paired_segment} Peak Potential (V)": ep2_values,
            "ΔE (V)": deltas,
        }
    )

    point_colors = [None, None]
    if options["plot"]:
        plt.figure()
        plt.xlabel("log(Scan Rate) (log(V/s))")
        plt.ylabel("Peak Potential (V)")
        point_colors[0] = _artist_color(
            plt.scatter(log_scan_rates, ep1_values, label=f"Seg {base_segment} Ep")
        )
        point_colors[1] = _artist_color(
            plt.scatter(log_scan_rates, ep2_values, label=f"Seg {paired_segment} Ep")
        )

    fit_indices = _trumpet_fit_indices_with_inclusive_stop(
        len(log_scan_rates),
        options.get("fit indices"),
    )
    fit_x, fit_y1 = _select_fit_indices(log_scan_rates, ep1_values, fit_indices)
    _, fit_y2 = _select_fit_indices(log_scan_rates, ep2_values, fit_indices)

    fit_model_results = {}
    fits = []

    seg1_label = f"Seg {base_segment}"
    seg2_label = f"Seg {paired_segment}"
    seg1_fit = _fit_series_xy(fit_x, fit_y1, options=options, label=seg1_label, model="linear")
    seg2_fit = _fit_series_xy(fit_x, fit_y2, options=options, label=seg2_label, model="linear")
    fit_model_results[seg1_label] = seg1_fit["model_result"]
    fit_model_results[seg2_label] = seg2_fit["model_result"]
    fits.append(np.asarray(seg1_fit["model_result"]["popt"], dtype=float))
    fits.append(np.asarray(seg2_fit["model_result"]["popt"], dtype=float))

    if options["plot"] and options["plot fit"]:
        plot_options_1 = _options_with_default_fit_color(options, raw_options, point_colors[0], index=0)
        plot_options_1.update({"new plot": False, "plot data": False, "model label": f"{seg1_label} Fit"})
        _plot_fit_model_result(seg1_fit["model_result"], plot_options_1)

        plot_options_2 = _options_with_default_fit_color(options, raw_options, point_colors[1], index=1)
        plot_options_2.update({"new plot": False, "plot data": False, "model label": f"{seg2_label} Fit"})
        _plot_fit_model_result(seg2_fit["model_result"], plot_options_2)

    if options["plot"] and _scatterfit_legend_requested(options):
        plt.legend(fontsize=_scatterfit_legend_fontsize(options))

    m1 = float(fits[0][0])
    m2 = float(fits[1][0])
    T = getattr(cvs[0], "temperature", None)
    if T is None:
        T = options.get("temperature", 298)
    T = float(T)
    if not np.isfinite(T) or T <= 0:
        raise ValueError("trumpet_analysis requires a positive temperature in K.")

    α = -R * T * np.log(10) / (2 * m1 * F)
    β = R * T * np.log(10) / (2 * m2 * F)

    line1 = np.poly1d(fits[0])
    line2 = np.poly1d(fits[1])
    roots = np.roots(line1 - line2)
    real_roots = roots[np.isclose(np.imag(roots), 0)]
    intercept_x = float(np.real(real_roots[0])) if len(real_roots) else np.nan

    D = options["D"]
    if D is not None and np.isfinite(α) and α > 0 and np.isfinite(intercept_x):
        ks = float(10 ** (0.78 + intercept_x / 2) / np.sqrt(R * T / (α * F * D)))
    else:
        ks = 0

    warning = _trumpet_reliability_warning(
        alpha=α,
        beta=β,
        slope_forward=m1,
        slope_reverse=m2,
        intercept_x=intercept_x,
        fit_x=fit_x,
    )

    trumpet_results = _trumpet_results_table(α, β, ks, D, warning, options)

    if options["print"]:
        print("Trumpet Analysis Summary:")
        _display_trumpet_equations(resolved=True, compact=False, include_definitions=False)
        _display_trumpet_parameter_table(
            _trumpet_parameter_table(
                base_segment=base_segment,
                paired_segment=paired_segment,
                temperature=T,
                diffusion_coefficient=D,
                fit_indices=options.get("fit indices"),
                options=options,
            ),
            options,
        )
        print("Trumpet Results:")
        _display_trumpet_results_table(trumpet_results, options)
        _print_fit_model_results(fit_model_results, options)

    if options["print all"]:
        print(data)
        print(fits)

    data.attrs["fit model results"] = fit_model_results
    data.attrs["fit table"] = trumpet_results
    return data, fits, ks


def trumpet_analysis(cvs, options={}):
    """Run trumpet analysis from paired anodic and cathodic peak potentials.
    
    Parameters
    ----------
    cvs : sequence of cv
        CV objects measured at different scan rates.
    options : dict or TrumpetAnalysisOptions, optional
        Segment, peak-picking, print, and plot options. See ``e.describe_options("trumpet_analysis")``.
    
    Returns
    -------
    ScatterFitResult
        Fit result with trumpet-analysis data, fit coefficients, and statistics.
    
    Examples
    --------
    >>> result = e.trumpet_analysis(cvs, {"segments": [1, 2]})
    """
    return _scatter_result_from_legacy(
        _trumpet_analysis_legacy_return(cvs, options),
        summary={"analysis": "trumpet"},
    )

_NICHOLSON_AGARWAL_DELTA_EP_MV = np.arange(61, 215, dtype=float)
_NICHOLSON_AGARWAL_PSI = np.array([
    9.5, 6.5, 5.3, 4.4, 3.7, 3.3, 2.9, 2.6, 2.35, 2.15,
    1.95, 1.83, 1.69, 1.58, 1.48, 1.40, 1.32, 1.25, 1.19, 1.13,
    1.08, 1.028, 0.984, 0.943, 0.906, 0.871, 0.839, 0.809, 0.781, 0.755,
    0.730, 0.706, 0.684, 0.663, 0.643, 0.625, 0.607, 0.589, 0.573, 0.558,
    0.544, 0.530, 0.517, 0.504, 0.492, 0.480, 0.469, 0.458, 0.448, 0.438,
    0.428, 0.419, 0.410, 0.402, 0.394, 0.386, 0.378, 0.370, 0.363, 0.356,
    0.349, 0.343, 0.337, 0.331, 0.325, 0.319, 0.313, 0.308, 0.302, 0.297,
    0.292, 0.287, 0.283, 0.278, 0.274, 0.269, 0.265, 0.261, 0.257, 0.253,
    0.249, 0.245, 0.241, 0.238, 0.234, 0.231, 0.227, 0.224, 0.221, 0.218,
    0.214, 0.211, 0.208, 0.205, 0.203, 0.200, 0.197, 0.195, 0.192, 0.190,
    0.187, 0.185, 0.182, 0.180, 0.177, 0.175, 0.173, 0.171, 0.169, 0.167,
    0.164, 0.162, 0.160, 0.158, 0.156, 0.154, 0.153, 0.151, 0.149, 0.147,
    0.145, 0.144, 0.142, 0.140, 0.138, 0.137, 0.135, 0.134, 0.132, 0.131,
    0.129, 0.127, 0.126, 0.125, 0.123, 0.122, 0.120, 0.119, 0.118, 0.116,
    0.115, 0.114, 0.112, 0.111, 0.110, 0.109, 0.107, 0.106, 0.105, 0.104,
    0.103, 0.102, 0.101, 0.100,
], dtype=float)


def _nicholson_psi_agarwal(delta_ep_mv):
    """Interpolate Agarwal 2025 Table 4 psi values from n*Delta Ep in mV."""
    values = np.asarray(delta_ep_mv, dtype=float)
    psi = np.interp(
        values,
        _NICHOLSON_AGARWAL_DELTA_EP_MV,
        _NICHOLSON_AGARWAL_PSI,
        left=np.nan,
        right=np.nan,
    )
    if np.isscalar(delta_ep_mv):
        return float(psi)
    return psi


def _nicholson_psi_lavagnini(delta_ep_mv):
    """Lavagnini-style empirical Nicholson psi backup; input must be in mV."""
    values = np.asarray(delta_ep_mv, dtype=float)
    psi = (-0.6288 + 0.0021 * values) / (1 - 0.017 * values)
    if np.isscalar(delta_ep_mv):
        return float(psi)
    return psi


def _nicholson_fit(x, y, through_origin=True):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x_fit = x[mask]
    y_fit = y[mask]
    if len(x_fit) < 2:
        raise ValueError("At least 2 included Nicholson points are required for fitting.")

    if through_origin:
        denom = float(np.sum(x_fit ** 2))
        if denom == 0:
            raise ValueError("Nicholson x values cannot all be zero.")
        slope = float(np.sum(x_fit * y_fit) / denom)
        intercept = 0.0
    else:
        slope, intercept = np.polyfit(x_fit, y_fit, 1)
        slope = float(slope)
        intercept = float(intercept)

    y_pred = slope * x_fit + intercept
    ss_res = float(np.sum((y_fit - y_pred) ** 2))
    ss_tot = float(np.sum((y_fit - np.mean(y_fit)) ** 2))
    r2 = np.nan if ss_tot == 0 else 1 - ss_res / ss_tot
    return {
        "slope": slope,
        "intercept": intercept,
        "r2": r2,
        "x fit": x_fit,
        "y fit": y_fit,
        "y predicted": y_pred,
    }


def _coerce_nicholson_cv_list(cvs):
    if isinstance(cvs, (list, tuple)):
        cv_list = list(cvs)
    else:
        cv_list = [cvs]
    if not cv_list:
        raise ValueError("nicholson_analysis requires at least one CV.")
    for item in cv_list:
        if not hasattr(item, "half_wave_potential"):
            raise TypeError("nicholson_analysis expects CV-like objects with half_wave_potential().")
    return cv_list


def _resolve_nicholson_scan_rates(cvs, options):
    manual = options.get("scan rates", None)
    if manual is None:
        manual = options.get("scan rate", None)
    if manual is None:
        rates = [getattr(item, "scan_rate", None) for item in cvs]
    elif isinstance(manual, (list, tuple, np.ndarray, pd.Series)):
        rates = list(manual)
    else:
        rates = [manual] * len(cvs)
    if len(rates) != len(cvs):
        raise ValueError("Manual scan rates must match the number of CVs.")
    rates = [float(rate) if rate is not None else None for rate in rates]
    for rate in rates:
        if rate is None or not np.isfinite(rate) or rate <= 0:
            raise ValueError("Each CV must have a positive scan_rate, or provide 'scan rates'.")
    return rates


def _resolve_nicholson_temperature(cv_obj, options):
    temperature = getattr(cv_obj, "temperature", None)
    if temperature is None:
        temperature = options.get("temperature", 298)
    temperature = float(temperature)
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("Temperature must be a positive value in K.")
    return temperature


def _nicholson_ir_drop_obvious(cv_obj):
    names = (
        "ir_compensated", "iR_compensated", "ru_compensated",
        "background_subtracted", "background_corrected",
    )
    for name in names:
        if bool(getattr(cv_obj, name, False)):
            return True
    options = getattr(cv_obj, "options", {}) or {}
    for key in ("ir compensated", "iR compensated", "background subtracted", "background corrected"):
        if bool(options.get(key, False)):
            return True
    return False


def _nicholson_psi_from_source(delta_ep_mv, options):
    source = str(options.get("psi source", "agarwal table")).strip().lower()
    if source in {"agarwal", "agarwal table", "table", "table 4"}:
        return _nicholson_psi_agarwal(delta_ep_mv), "agarwal table"
    if source in {"lavagnini", "empirical", "empirical equation"}:
        equation = str(options.get("empirical psi equation", "lavagnini")).strip().lower()
        if equation != "lavagnini":
            raise ValueError("Only the Lavagnini empirical psi equation is currently supported.")
        return _nicholson_psi_lavagnini(delta_ep_mv), "lavagnini empirical"
    raise ValueError("'psi source' must be 'agarwal table' or 'lavagnini'.")


def _normalize_nicholson_fit_model(options):
    raw_model = options.get("fit model", None)
    if raw_model in (None, ""):
        return "origin" if bool(options.get("fit through origin", True)) else "linear"
    model = str(raw_model).strip().lower().replace("_", " ").replace("-", " ")
    if model == "origin":
        return "origin"
    if model == "linear":
        return "linear"
    raise ValueError("Nicholson 'fit model' must be 'origin' or 'linear'.")


def _nicholson_display_unit_text(unit):
    if not unit:
        return ""
    unit = str(unit)
    if unit.startswith("u"):
        unit = "μ" + unit[1:]
    return unit


def _nicholson_scaled_series(values, unit, sig_figs=4):
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    display_unit = unit
    if len(finite):
        reference = float(np.nanmax(np.abs(finite)))
        if reference == 0:
            reference = 1.0
        try:
            _, display_unit = scale_value(reference, unit, selected_unit="auto")
        except Exception:
            display_unit = unit
    scaled = []
    for value in values:
        if not np.isfinite(value):
            scaled.append("")
            continue
        try:
            display_value = _nicholson_scale_value_to_unit(float(value), unit, display_unit)
        except Exception:
            display_value = float(value)
        scaled.append(_format_fit_model_display_value(display_value, sig_figs=sig_figs))
    return scaled, _nicholson_display_unit_text(display_unit)


def _nicholson_display_unit(values, unit):
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    display_unit = unit
    if len(finite):
        reference = float(np.nanmax(np.abs(finite)))
        if reference == 0:
            reference = 1.0
        try:
            _, display_unit = scale_value(reference, unit, selected_unit="auto")
        except Exception:
            display_unit = unit
    return _nicholson_display_unit_text(display_unit)


def _nicholson_scale_value_to_unit(value, source_unit, display_unit):
    if display_unit in (None, "", source_unit):
        return float(value)
    selected_prefix_unit = str(display_unit).split("/", 1)[0]
    scaled_value, _ = scale_value(float(value), source_unit, selected_unit=selected_prefix_unit)
    return float(scaled_value)


def _nicholson_summary_scalar_text(value, unit="", sig_figs=4, scientific=False):
    if unit == "cm^2/s":
        return _format_sevcik_value(value, sig_figs=sig_figs, unit=unit, scientific=True)
    if unit:
        scaled_values, display_unit = _nicholson_scaled_series([value], unit, sig_figs=sig_figs)
        text = scaled_values[0]
        return f"{text} {display_unit}".strip() if display_unit else text
    return _format_fit_model_display_value(value, sig_figs=sig_figs)


def _nicholson_equation_bundle(summary):
    n = float(summary.get("num electrons", 1))
    temperature = float(summary.get("temperature / K", 298))
    D = float(summary.get("D / cm^2 s^-1", 0))
    through_origin = bool(summary.get("fit through origin", True))
    fit_latex = r"\psi = k^0 x" if through_origin else r"\psi = k^0 x + b"
    symbolic_latex = fit_latex + r",\quad x=\left(\frac{RT}{\pi D n F \nu}\right)^{1/2},\quad k^0_{\mathrm{point}}=\frac{\psi}{x}"
    resolved_latex = (
        fit_latex
        + rf",\quad x=\left(\frac{{({R:.6g})({temperature:g})}}{{\pi({D:g})({n:g})({F:.6g})\nu}}\right)^{{1/2}}"
    )
    definitions_latex = (
        rf"R={R:.6g},\quad F={F:.6g},\quad T={temperature:g}\ \mathrm{{K}},\quad "
        rf"D={D:g}\ \mathrm{{cm^2/s}},\quad n={n:g},\quad \nu\ \mathrm{{in\ V/s}}"
    )
    return {
        "symbolic latex": symbolic_latex,
        "resolved latex": resolved_latex,
        "compact latex": resolved_latex,
        "definitions latex": definitions_latex,
        "symbolic": summary.get("equation", ""),
        "resolved": summary.get("x definition", ""),
        "compact": summary.get("x definition", ""),
        "definitions": (
            f"R = {R:.6g}, F = {F:.6g}, T = {temperature:g} K, "
            f"D = {D:g} cm^2/s, n = {n:g}, v in V/s"
        ),
    }


def _nicholson_parameter_display_table(summary, options=None):
    options = options or {}
    sig_figs = options.get("sig figs", 4)
    rows = [
        ("Fit Model", summary.get("fit model", "origin")),
        ("D", _nicholson_summary_scalar_text(summary.get("D / cm^2 s^-1"), "cm^2/s", sig_figs=sig_figs)),
        ("n", _format_fit_model_display_value(summary.get("num electrons"), sig_figs=sig_figs)),
        ("T", _nicholson_summary_scalar_text(summary.get("temperature / K"), "K", sig_figs=sig_figs)),
        ("ψ Source", summary.get("psi source")),
        (
            "Valid nΔEp Range",
            f"{_format_fit_model_display_value(summary.get('nicholson delta ep min mv'), sig_figs=sig_figs)} to "
            f"{_format_fit_model_display_value(summary.get('nicholson delta ep max mv'), sig_figs=sig_figs)} mV",
        ),
    ]
    return pd.DataFrame([{"Parameter": key, "Value": value} for key, value in rows])


def _nicholson_summary_display_table(summary, options=None):
    options = options or {}
    sig_figs = options.get("sig figs", 4)
    rows = [
        ("Included Points", f"{summary.get('num included')} / {summary.get('num points')}"),
        ("Excluded Points", _format_fit_model_display_value(summary.get("num excluded"), sig_figs=sig_figs)),
        ("k0", _nicholson_summary_scalar_text(summary.get("k0 / cm s^-1"), "cm/s", sig_figs=sig_figs)),
        ("Intercept", _format_fit_model_display_value(summary.get("intercept"), sig_figs=sig_figs)),
        ("R²", _format_fit_model_display_value(summary.get("r2"), sig_figs=sig_figs)),
        ("k0 Point Mean", _nicholson_summary_scalar_text(summary.get("k0 point mean / cm s^-1"), "cm/s", sig_figs=sig_figs)),
        ("k0 Point Median", _nicholson_summary_scalar_text(summary.get("k0 point median / cm s^-1"), "cm/s", sig_figs=sig_figs)),
        ("k0 Point Std", _nicholson_summary_scalar_text(summary.get("k0 point std / cm s^-1"), "cm/s", sig_figs=sig_figs)),
    ]
    return pd.DataFrame([{"Setting": key, "Value": value} for key, value in rows])


def _nicholson_display_data_table(data, options=None):
    options = options or {}
    sig_figs = options.get("sig figs", 4)
    display_df = data.copy()
    scaled_columns = [
        ("scan rate / V s^-1", "scan rate / mV/s", "V/s"),
        ("temperature / K", "temperature / K", "K"),
        ("Ep1 / V", "Ep1 / V", "V"),
        ("Ep2 / V", "Ep2 / V", "V"),
        ("E1/2 / V", "E1/2 / V", "V"),
        ("ΔEp / V", "ΔEp / mV", "V"),
        ("nΔEp / mV", "nΔEp / mV", "mV"),
        ("Nicholson x / s cm^-1", "Nicholson x / s/cm", "s/cm"),
        ("k0 point / cm s^-1", "k0 point / cm/s", "cm/s"),
    ]

    for column, default_label, unit in scaled_columns:
        if column not in display_df.columns:
            continue
        values = display_df[column].to_numpy(dtype=float)
        scaled_values, display_unit = _nicholson_scaled_series(values, unit, sig_figs=sig_figs)
        if column == "temperature / K":
            display_label = "temperature / K"
        elif column == "Ep1 / V":
            display_label = "Ep1 / V"
        elif column == "Ep2 / V":
            display_label = "Ep2 / V"
        elif column == "E1/2 / V":
            display_label = "E1/2 / V"
        else:
            base = default_label.split(" / ", 1)[0]
            display_label = f"{base} / {display_unit}" if display_unit else base
        display_df[column] = scaled_values
        display_df = display_df.rename(columns={column: display_label})

    if "ψ" in display_df.columns:
        display_df["ψ"] = [
            _format_fit_model_display_value(value, sig_figs=sig_figs) if np.isfinite(value) else ""
            for value in display_df["ψ"].to_numpy(dtype=float)
        ]
    if "included" in display_df.columns:
        display_df["included"] = display_df["included"].map(lambda value: "Yes" if bool(value) else "No")
    return display_df


def _nicholson_fit_failure_message(data, summary):
    num_included = int(summary.get("num included", 0))
    num_points = int(summary.get("num points", len(data)))
    min_mv = _format_fit_model_display_value(summary.get("nicholson delta ep min mv"), sig_figs=4)
    max_mv = _format_fit_model_display_value(summary.get("nicholson delta ep max mv"), sig_figs=4)
    included_names = [str(name) for name in data.loc[data["included"].astype(bool), "name"].tolist()]
    included_text = ", ".join(included_names) if included_names else "none"
    excluded_rows = data.loc[~data["included"].astype(bool), ["name", "nΔEp / mV", "exclusion reason"]]
    lines = [
        "Nicholson analysis requires at least 2 included points for fitting.",
        f"Found {num_included} included out of {num_points} total.",
        f"Included: {included_text}.",
        f"Nicholson includes only scans with finite ψ and nΔEp between {min_mv} and {max_mv} mV by default.",
    ]
    if len(excluded_rows) > 0:
        lines.append("Excluded:")
        for _, row in excluded_rows.iterrows():
            ndelta = _format_fit_model_display_value(row["nΔEp / mV"], sig_figs=6)
            lines.append(f"  - {row['name']}: nΔEp = {ndelta} mV; {row['exclusion reason']}")
    lines.append(
        "Try a different 'segment' or 'guess potential', fit a more quasireversible subset, "
        "or set 'exclude invalid delta ep': False for a diagnostic fit."
    )
    return "\n".join(lines)


def _plot_nicholson_analysis(data, fit_result, options):
    if options.get("new plot", True):
        plt.figure()
    included = data["included"].astype(bool)
    x_col = "Nicholson x / s cm^-1"
    y_col = "ψ"
    x_display_unit = _nicholson_display_unit(data[x_col].to_numpy(dtype=float), "s/cm")
    x_plot = np.asarray(
        [_nicholson_scale_value_to_unit(float(value), "s/cm", x_display_unit) for value in data[x_col].to_numpy(dtype=float)],
        dtype=float,
    )

    if included.any():
        plt.scatter(x_plot[included.to_numpy()], data.loc[included, y_col], color="black", label="included")
    if (~included).any():
        excluded = data.loc[~included]
        finite = np.isfinite(excluded[x_col]) & np.isfinite(excluded[y_col])
        if finite.any():
            plt.scatter(
                np.asarray(
                    [_nicholson_scale_value_to_unit(float(value), "s/cm", x_display_unit) for value in excluded.loc[finite, x_col].to_numpy(dtype=float)],
                    dtype=float,
                ),
                excluded.loc[finite, y_col],
                facecolors="none",
                edgecolors="tab:red",
                marker="o",
                label="excluded",
            )

    if fit_result is not None and len(fit_result["x fit"]) >= 2:
        x_line = np.linspace(np.min(fit_result["x fit"]), np.max(fit_result["x fit"]), 100)
        y_line = fit_result["slope"] * x_line + fit_result["intercept"]
        x_line_scaled = np.asarray(
            [_nicholson_scale_value_to_unit(float(value), "s/cm", x_display_unit) for value in x_line],
            dtype=float,
        )
        plt.plot(
            x_line_scaled,
            y_line,
            color=_fit_color_from_options(options, index=0, fallback="tab:red"),
            linestyle=options.get("fit linestyle", "--"),
            linewidth=options.get("fit linewidth", 1),
            alpha=options.get("fit alpha", 1),
            label="Nicholson fit" if options.get("fit label", False) else None,
        )

    plt.xlabel(rf"$(RT / \pi D n F \nu)^{{1/2}}$ / {x_display_unit}")
    plt.ylabel(r"$\psi$")
    plt.title("Nicholson Analysis")
    if options.get("legend", True):
        handles, labels = plt.gca().get_legend_handles_labels()
        if handles:
            plt.legend()


def _nicholson_equation_summary(through_origin=True):
    x_definition = "x = (R T / (pi D n F v))^1/2"
    if through_origin:
        equation = "psi = k0 x"
    else:
        equation = "psi = k0 x + b"
    return equation, x_definition


def _print_nicholson_summary(data, summary, options):
    display_data = _nicholson_display_data_table(data, options)
    if options.get("pretty print", True):
        print("Nicholson Analysis Equation:")
        _display_analysis_equation(
            r"\text{Nicholson analysis equation:}",
            "Nicholson analysis equation",
            _nicholson_equation_bundle(summary),
            resolved=False,
            compact=False,
            include_definitions=True,
        )
        print("\nNicholson Parameters:")
        display_object_table(_nicholson_parameter_display_table(summary, options), options)
        print("\nNicholson Analysis Summary:")
        display_object_table(_nicholson_summary_display_table(summary, options), options)
        print("\nNicholson Analysis Data:")
        display_object_table(display_data, options)
    else:
        print("Nicholson Analysis Equation:")
        equation = _nicholson_equation_bundle(summary)
        print("  " + equation["symbolic"])
        print("  " + equation["definitions"])
        print("\nNicholson Parameters:")
        print(_nicholson_parameter_display_table(summary, options).to_string(index=False, justify="left"))
        print("\nNicholson Analysis Summary:")
        print(_nicholson_summary_display_table(summary, options).to_string(index=False, justify="left"))
        print("\nNicholson Analysis Data:")
        print(display_data.to_string(index=False, justify="left"))
    excluded = data.loc[~data["included"].astype(bool), ["name", "exclusion reason"]]
    if not excluded.empty:
        print("Excluded points:")
        for _, row in excluded.iterrows():
            print(f"  {row['name']}: {row['exclusion reason']}")


def nicholson_analysis(cvs, options={}):
    """Estimate heterogeneous electron-transfer kinetics using Nicholson analysis.
    
    Parameters
    ----------
    cvs : sequence of cv
        CV objects containing a reversible or quasireversible redox couple.
    options : dict or NicholsonOptions, optional
        Peak-current, Nicholson-source, fit, print, and plot options. See ``e.describe_options("nicholson")``.
    
    Returns
    -------
    dict
        Dictionary with ``"data"`` for the Nicholson input/result table and
        ``"summary"`` for fitted kinetic values and equation metadata.
    
    Examples
    --------
    >>> result = e.nicholson_analysis(cvs, {"guess potential": 0.4})
    >>> result["summary"]["k0 / cm s^-1"]
    """
    raw_options = {} if options is None else dict(options)
    options = NicholsonOptions.from_options(raw_options).to_legacy_dict()
    cv_list = _coerce_nicholson_cv_list(cvs)

    D = options.get("D", None)
    if D is None:
        raise ValueError("Nicholson analysis requires 'D' in cm^2/s.")
    D = float(D)
    if not np.isfinite(D) or D <= 0:
        raise ValueError("Nicholson analysis requires a positive 'D' in cm^2/s.")

    n = float(options.get("num electrons", 1))
    min_mv = float(options.get("nicholson delta ep min mv", 61))
    max_mv = float(options.get("nicholson delta ep max mv", 212))
    exclude_invalid = bool(options.get("exclude invalid delta ep", True))
    fit_model = _normalize_nicholson_fit_model(options)
    through_origin = fit_model == "origin"
    scan_rates = _resolve_nicholson_scan_rates(cv_list, options)

    psi_source_name = str(options.get("psi source", "agarwal table")).strip().lower()
    if psi_source_name in {"lavagnini", "empirical", "empirical equation"}:
        warnings.warn(
            "Lavagnini psi is an empirical backup. Agarwal 2025 Table 4 is preferred, "
            "and empirical equations should not be extrapolated beyond their valid range.",
            UserWarning,
            stacklevel=2,
        )

    if options.get("warn ir drop", True) and not all(_nicholson_ir_drop_obvious(item) for item in cv_list):
        warnings.warn(
            "Nicholson analysis assumes appropriate iR compensation and background subtraction where needed; "
            "eCAT could not verify that these were performed.",
            UserWarning,
            stacklevel=2,
        )

    diagnostic_ax = None
    if options.get("plot all", False) and options.get("plot diagnostic", True):
        try:
            diagnostic_options = _multiplot_options_from_mapping(options)
            diagnostic_options.update({
                "plot": True,
                "print": False,
                "legend": True,
                "title": options.get("diagnostic title", "Nicholson Half-Wave Diagnostics"),
            })
            diagnostic_ax = multiplot(cv_list, diagnostic_options)
        except Exception:
            plt.figure()
            diagnostic_ax = plt.gca()

    rows = []
    for cv_obj, scan_rate in zip(cv_list, scan_rates):
        peak_options = dict(options)
        peak_options["plot"] = diagnostic_ax is not None
        peak_options["print"] = bool(options.get("print all", False))
        peak_options["internal call"] = True
        peak_options["new plot"] = False
        peak_options["plot cv"] = False
        if diagnostic_ax is not None:
            plt.sca(diagnostic_ax)
        half_wave = cv_obj.half_wave_potential(peak_options)
        e_half = half_wave["E(1/2)"]
        peak_info1 = half_wave["peak 1"]
        peak_info2 = half_wave["peak 2"]
        if "Ep" not in peak_info1 or "Ep" not in peak_info2:
            raise ValueError("half_wave_potential() must return peak_info dictionaries with 'Ep' values.")

        ep1 = float(peak_info1["Ep"])
        ep2 = float(peak_info2["Ep"])
        delta_ep_v = abs(ep1 - ep2)
        n_delta_ep_mv = n * delta_ep_v * 1000
        temperature = _resolve_nicholson_temperature(cv_obj, options)
        nicholson_x = np.sqrt(R * temperature / (np.pi * D * n * F * scan_rate))
        psi, resolved_source = _nicholson_psi_from_source(n_delta_ep_mv, options)

        exclusion_reason = ""
        if n_delta_ep_mv < min_mv:
            exclusion_reason = "too reversible / lower-bound estimate only"
        elif n_delta_ep_mv > max_mv:
            exclusion_reason = "outside Nicholson range; use Klingler-Kochi or digital simulation"
        elif not np.isfinite(psi):
            exclusion_reason = "psi unavailable from selected source"

        included = not exclusion_reason or not exclude_invalid
        if not np.isfinite(psi):
            included = False
        k0_point = psi / nicholson_x if np.isfinite(psi) and nicholson_x > 0 else np.nan

        rows.append({
            "name": getattr(cv_obj, "name", f"CV {len(rows) + 1}"),
            "scan rate / V s^-1": scan_rate,
            "temperature / K": temperature,
            "Ep1 / V": ep1,
            "Ep2 / V": ep2,
            "E1/2 / V": float(e_half),
            "ΔEp / V": delta_ep_v,
            "nΔEp / mV": n_delta_ep_mv,
            "ψ": psi,
            "Nicholson x / s cm^-1": nicholson_x,
            "k0 point / cm s^-1": k0_point,
            "included": bool(included),
            "exclusion reason": exclusion_reason,
            "psi source": resolved_source,
        })

    data = pd.DataFrame(rows)
    included_mask = data["included"].astype(bool)
    finite_fit_mask = included_mask & np.isfinite(data["Nicholson x / s cm^-1"]) & np.isfinite(data["ψ"])
    num_included = int(finite_fit_mask.sum())
    failure_summary = {
        "num points": int(len(data)),
        "num included": num_included,
        "nicholson delta ep min mv": min_mv,
        "nicholson delta ep max mv": max_mv,
    }
    if num_included < 2:
        message = _nicholson_fit_failure_message(data, failure_summary)
        warnings.warn(message, UserWarning, stacklevel=2)
        raise ValueError(message)

    fit_result = _nicholson_fit(
        data.loc[finite_fit_mask, "Nicholson x / s cm^-1"].to_numpy(),
        data.loc[finite_fit_mask, "ψ"].to_numpy(),
        through_origin=through_origin,
    )
    pointwise = data.loc[finite_fit_mask, "k0 point / cm s^-1"].astype(float)
    summary = {
        "k0 / cm s^-1": fit_result["slope"],
        "intercept": fit_result["intercept"],
        "r2": fit_result["r2"],
        "num points": int(len(data)),
        "num included": num_included,
        "num excluded": int(len(data) - num_included),
        "D / cm^2 s^-1": D,
        "num electrons": n,
        "temperature / K": float(data["temperature / K"].iloc[0]) if len(data) else options.get("temperature", 298),
        "psi source": data["psi source"].iloc[0] if len(data) else options.get("psi source"),
        "fit model": fit_model,
        "fit through origin": through_origin,
        "nicholson delta ep min mv": min_mv,
        "nicholson delta ep max mv": max_mv,
        "k0 point mean / cm s^-1": float(pointwise.mean()),
        "k0 point std / cm s^-1": float(pointwise.std(ddof=1)) if len(pointwise) > 1 else 0.0,
        "k0 point median / cm s^-1": float(pointwise.median()),
    }
    equation, x_definition = _nicholson_equation_summary(through_origin=through_origin)
    summary["equation"] = equation
    summary["x definition"] = x_definition

    if options.get("plot", True) or options.get("plot all", False):
        plot_options = dict(options)
        plot_options["new plot"] = True
        _plot_nicholson_analysis(data, fit_result, plot_options)
    if options.get("print", True):
        _print_nicholson_summary(data, summary, options)

    return {"data": data, "summary": summary}


def _sevcik_legacy_return(cvs, options={}):
    raw_options = options
    typed_options = SevcikAnalysisOptions.from_options(options)
    options = typed_options.to_legacy_dict()
    do_plot = options.get("plot", True)
    do_print = options.get("print", True)

    segments = _normalize_segment_option(options, default=[1, 2])

    num_electrons = options.get("num electrons", 1)
    scan_dependence = options.get("scan dependence", 0.5)

    peaks, x_values = [], []
    peak_unit = _axis_common_unit(
        cvs,
        lambda cv: (cv.y(options), cv.y(options).name),
        options.get('y unit', 'auto')
    )
    diffusion_coefficients, fits = [], []
    fit_rows = []
    data = pd.DataFrame()
    C = None
    v = None

    # Determine type of analysis
    similar, different = echem_similar_different(cvs)
    if 'scan rate' in different:
        if options.get("C") is not None:
            C = options["C"]
        elif len(cvs[0].concentrations) > 0:
            C = concentration_to_float(cvs[0].concentrations[-1]) / 1e3  # mol/cm^3
        else:
            C = 0
        for cv in cvs:
            x_values.append(cv.scan_rate)
        x_values = np.array(x_values) ** scan_dependence
        data[f'Scan Rate ^ {scan_dependence}'] = x_values
    else:
        v = cvs[0].scan_rate
        diff_conc_idx = next((i for i in range(len(cvs[0].concentrations))
                              if cvs[0].concentrations[i] != cvs[1].concentrations[i]), -1)
        for cv in cvs:
            x_values.append(concentration_to_float(cv.concentrations[diff_conc_idx]))
        data['Concentration (M)'] = x_values

    internal_options = typed_options.for_peak_current().to_legacy_dict()
    internal_options["internal call"] = True
    internal_options["new plot"] = False
    internal_options["plot"] = options.get("plot all", False)
    internal_options["print"] = options.get("print all", False)

    if options["plot all"]:
        multiplot(cvs, _multiplot_options_from_mapping(options))

    for seg in segments:
        internal_options["segment"] = seg
        segment_peaks = []
        for cv in cvs:
            y_name = cv.y(options).name
            y_unit = cv.units.get(y_name, '')
            if options.get('print all'):
                print(cv.name)
            peak_current = cv.peak_current(internal_options)["ip"]
            scaled_peak_current, _ = scale_value(peak_current, y_unit, selected_unit=peak_unit)
            segment_peaks.append(scaled_peak_current)
        peaks.append(segment_peaks)

    if do_plot:
        plt.figure()
        plt.xlabel('(Scan Rate)$^{' + str(scan_dependence) + '}$ (V/s)' if 'scan rate' in different else 'Concentration (M)')
        plt.ylabel(f'Peak {y_name} ({peak_unit})')

    for i, seg in enumerate(segments):
        data[f"Seg {seg} Peaks"] = peaks[i]
        point_color = None
        if do_plot:
            point_color = _artist_color(plt.scatter(x_values, peaks[i]))
        idxs = options.get('fit indices')
        if idxs is None:
            idxs = [0, len(x_values)]
        fit_x, fit_y = _select_fit_indices(x_values, peaks[i], idxs)
        scatter_options = _options_with_default_fit_color(options, raw_options, point_color, index=i)
        coeffs, stats = _scatter_fit(
            fit_x,
            fit_y,
            label=f"Seg {seg} Fit",
            plot_fit=options['plot fit'],
            options=scatter_options,
        )
        fits.append(coeffs)
        fit_rows.append(_scatter_fit_row(f"Seg {seg}", coeffs, stats, fit_x=fit_x))

    if do_plot and _scatterfit_legend_requested(options):
        plt.legend(fontsize=_scatterfit_legend_fontsize(options))

    # Compute diffusion coefficients
    current_adjustment = get_conversion_factor(peak_unit)
    T = cvs[0].temperature
    S = cvs[0].electrode_area
    try:
        S = float(S)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "sevcik_analysis requires a numeric electrode area. "
            "Set 'electrode area' or 'electrode diameter' when loading the CVs."
        ) from exc
    if "cm^2" in peak_unit:
        current_adjustment *= S

    if S != 0 and (C is not None and C != 0):
        for i, seg in enumerate(segments):
            if 'scan rate' not in different:
                current_adjustment *= 1e3  # mol/L to mol/cm^3
            m = fits[i][0] * current_adjustment
            if 'scan rate' in different:
                D = (R * T / (F * num_electrons) ** 3) * (m / (0.4463 * S * C)) ** 2
            else:
                D = (R * T / (F ** 2 * num_electrons ** 3 * v * S ** 2)) * (m / 0.4463) ** 2
            D = round_sigfigs(D, 3)
            diffusion_coefficients.append(D)

    sevcik_fit_table = _sevcik_fit_results_table(
        _scatter_fit_table(fit_rows),
        diffusion_coefficients,
        options=options,
    )

    if do_print:
        print("Sevcik Analysis Summary:")
        _display_sevcik_diffusion_equation(
            mode="scan rate" if "scan rate" in different else "concentration",
            num_electrons=num_electrons,
            temperature=T,
            electrode_area=S,
            concentration=C,
            scan_rate=v,
            scan_dependence=scan_dependence,
            resolved=False,
            compact=False,
            include_definitions=False,
        )
        _display_sevcik_parameter_table(
            _sevcik_parameter_table(
                mode="scan rate" if "scan rate" in different else "concentration",
                num_electrons=num_electrons,
                temperature=T,
                electrode_area=S,
                concentration=C,
                scan_rate=v,
                scan_dependence=scan_dependence,
                options=options,
            ),
            options,
        )
        _print_sevcik_fit_results(sevcik_fit_table)

    if options.get('print all'):
        print(data)
        print(fits)

    if isinstance(data, pd.DataFrame):
        data.attrs["fit table"] = sevcik_fit_table
    return diffusion_coefficients, data, fits


def sevcik_analysis(cvs, options={}):
    """Run Sevcik analysis on peak-current data across scan rates.
    
    Parameters
    ----------
    cvs : sequence of cv
        CV objects measured at different scan rates.
    options : dict or SevcikAnalysisOptions, optional
        Peak-current, fit, print, and plot options. See ``e.describe_options("sevcik_analysis")``.
    
    Returns
    -------
    ScatterFitResult
        Fit result with Sevcik-style data, fit coefficients, and statistics.
    
    Examples
    --------
    >>> result = e.sevcik_analysis(cvs, {"guess potential": -1.5})
    """
    return _scatter_result_from_legacy(
        _sevcik_legacy_return(cvs, options),
        summary={"analysis": "Sevcik"},
    )


def _trumpet_fit_indices_with_inclusive_stop(length, fit_indices):
    if fit_indices is None:
        return None
    fit_indices_array = np.asarray(fit_indices, dtype=object)
    if fit_indices_array.ndim == 1 and len(fit_indices_array) == 2:
        start = int(fit_indices_array[0])
        stop = int(fit_indices_array[1])
        if stop < 0:
            stop = length + stop
        stop = min(stop + 1, length)
        return [start, stop]
    if fit_indices_array.ndim == 2 and fit_indices_array.shape[1] == 2:
        windows = []
        for start, stop in fit_indices_array:
            start = int(start)
            stop = int(stop)
            if stop < 0:
                stop = length + stop
            stop = min(stop + 1, length)
            windows.append([start, stop])
        return windows
    return fit_indices


def _format_trumpet_equation():
    return {
        "symbolic latex": (
            r"E_{p,\mathrm{f}}=m_{\mathrm{f}}\log_{10}(\nu)+b_{\mathrm{f}},\quad "
            r"E_{p,\mathrm{r}}=m_{\mathrm{r}}\log_{10}(\nu)+b_{\mathrm{r}}"
        ),
        "resolved latex": (
            r"\alpha=-\frac{RT\ln(10)}{2Fm_{\mathrm{f}}},\quad "
            r"\beta=\frac{RT\ln(10)}{2Fm_{\mathrm{r}}},\quad "
            r"k^0=\frac{10^{0.78+x_{\mathrm{int}}/2}}{\sqrt{RT/(\alpha F D)}}"
        ),
        "compact latex": "",
        "definitions latex": "",
        "symbolic": (
            "Ep,f = mf * log10(v) + bf; "
            "Ep,r = mr * log10(v) + br"
        ),
        "resolved": (
            "alpha = -(R * T * ln(10)) / (2 * F * mf); "
            "beta = (R * T * ln(10)) / (2 * F * mr); "
            "k0 = 10^(0.78 + xint / 2) / sqrt(R * T / (alpha * F * D))"
        ),
        "compact": "",
        "definitions": "",
    }


def _display_trumpet_equations(resolved=False, compact=False, include_definitions=False):
    return _display_analysis_equation(
        r"\text{Trumpet analysis equations:}",
        "Trumpet analysis equations",
        _format_trumpet_equation(),
        resolved=resolved,
        compact=compact,
        include_definitions=include_definitions,
    )


def _trumpet_parameter_table(base_segment, paired_segment, temperature, diffusion_coefficient, fit_indices, options=None):
    options = options or {}
    sig_figs = options.get("sig figs", 4)
    rows = [
        {"Parameter": "Segments", "Value": f"{base_segment}, {paired_segment}"},
        {"Parameter": "T", "Value": _format_sevcik_value(temperature, sig_figs=sig_figs, unit="K")},
    ]
    if diffusion_coefficient is not None:
        rows.append(
            {
                "Parameter": "D",
                "Value": _format_sevcik_value(diffusion_coefficient, sig_figs=sig_figs, unit="cm^2/s", scientific=True),
            }
        )
    if fit_indices is not None:
        rows.append({"Parameter": "Fit Indices", "Value": f"{fit_indices} (inclusive stop)"})
    return pd.DataFrame(rows)


def _display_trumpet_parameter_table(table, options):
    if not options.get("pretty print", True) or display is None:
        print(table.to_string(index=False, justify="left"))
        return table

    display_table = table.copy()
    display_table["Parameter"] = [
        {
            "T": "<i>T</i>",
            "D": "<i>D</i>",
        }.get(str(value), value)
        for value in display_table["Parameter"]
    ]
    styled = (
        display_table.style
        .format(escape=None)
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
    display(styled)
    return table


def _trumpet_reliability_warning(alpha, beta, slope_forward, slope_reverse, intercept_x, fit_x):
    reasons = []
    if not (np.isfinite(slope_forward) and slope_forward < 0):
        reasons.append("forward branch slope does not have the expected negative sign")
    if not (np.isfinite(slope_reverse) and slope_reverse > 0):
        reasons.append("reverse branch slope does not have the expected positive sign")
    if not (np.isfinite(alpha) and 0 < alpha < 1):
        reasons.append("alpha is outside the physical 0-1 range")
    if not (np.isfinite(beta) and 0 < beta < 1):
        reasons.append("beta is outside the physical 0-1 range")
    fit_x = np.asarray(fit_x, dtype=float)
    fit_x = fit_x[np.isfinite(fit_x)]
    if len(fit_x) > 0 and np.isfinite(intercept_x):
        if intercept_x < float(np.nanmin(fit_x)) or intercept_x > float(np.nanmax(fit_x)):
            reasons.append("branch intersection falls outside the fitted log(scan rate) window")
    if not reasons:
        return ""
    return "α/β may not be reliable: " + "; ".join(reasons) + "."


def _trumpet_results_table(alpha, beta, ks, diffusion_coefficient, warning, options=None):
    options = options or {}
    sig_figs = options.get("sig figs", 4)
    rows = [
        {"Parameter": "α", "Value": _format_fit_model_display_value(alpha, sig_figs=sig_figs)},
        {"Parameter": "β", "Value": _format_fit_model_display_value(beta, sig_figs=sig_figs)},
    ]
    if diffusion_coefficient is not None:
        rows.append(
            {
                "Parameter": "k0",
                "Value": _format_sevcik_value(ks, sig_figs=sig_figs, unit="cm/s", scientific=True),
            }
        )
    else:
        rows.append({"Parameter": "k0", "Value": "not computed (D not provided)"})
    if warning:
        rows.append({"Parameter": "Warning", "Value": warning})
    return pd.DataFrame(rows)


def _display_trumpet_results_table(table, options):
    if not options.get("pretty print", True) or display is None:
        print(table.to_string(index=False, justify="left"))
        return table

    display_table = table.copy()
    display_table["Parameter"] = [
        {
            "α": "&alpha;",
            "β": "&beta;",
            "k0": "<i>k</i><sup>0</sup>",
        }.get(str(value), value)
        for value in display_table["Parameter"]
    ]
    styled = (
        display_table.style
        .format(escape=None)
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
    display(styled)
    return table

def _normalize_segment_option(options, default=None):
    """
    Normalize 'segment' / 'segments' to a list.

    Returns
    -------
    list
        [None] means: do not force a segment, keep the downstream default behavior.
    """
    segments = options.get("segments")
    if segments is None:
        segments = options.get("segment", default)

    if segments is None:
        return [None]
    if isinstance(segments, int):
        return [segments]
    if isinstance(segments, (list, tuple, np.ndarray)):
        return list(segments)

    raise TypeError("'segment' must be an int and 'segments' must be a list/tuple of ints.")


def _fit_indices_for_segment(fit_indices, segment, npts):
    """
    Support:
    - None
    - [start, end]
    - {segment_number: [start, end], 'default': [start, end]}
    """
    if fit_indices is None:
        return [0, npts]

    if isinstance(fit_indices, dict):
        idxs = fit_indices.get(segment, fit_indices.get("default", [0, npts]))
    else:
        idxs = fit_indices

    if idxs is None:
        return [0, npts]

    return idxs


def _species_concentration_entries(cv_obj):
    """
    Return concentration entries keyed by species, unit, and same-unit occurrence.

    The occurrence is needed for filenames containing the same species in two
    chemically distinct roles, for example ``2.8MD2O_0.8xD2O``.
    """
    entries = []
    counts = {}
    for species, concentration in zip(
        getattr(cv_obj, "compounds", []) or [],
        getattr(cv_obj, "concentrations", []) or [],
    ):
        try:
            value, unit = _parse_concentration_value_and_unit(concentration)
        except ValueError:
            continue
        key_base = (species, unit)
        occurrence = counts.get(key_base, 0)
        counts[key_base] = occurrence + 1
        entries.append(
            {
                "species": species,
                "value": value,
                "unit": unit,
                "occurrence": occurrence,
                "concentration": concentration,
            }
        )
    return entries


def _entry_key(entry):
    return entry["species"], entry["unit"], entry["occurrence"]


def _get_species_concentration(cv_obj, species, default=0.0, unit=None, occurrence=None):
    """
    Return the concentration of `species` in M.
    If the species is absent, return `default`.
    """
    for entry in _species_concentration_entries(cv_obj):
        if entry["species"] != species:
            continue
        if unit is not None and entry["unit"] != unit:
            continue
        if occurrence is not None and entry["occurrence"] != occurrence:
            continue
        return entry["value"]
    return float(default)


def _infer_varying_concentration_entries(cvs, atol=0.0):
    ordered = []
    metadata = {}

    for cv_obj in cvs:
        for entry in _species_concentration_entries(cv_obj):
            key = _entry_key(entry)
            if key not in metadata:
                ordered.append(key)
                metadata[key] = {
                    "species": entry["species"],
                    "unit": entry["unit"],
                    "occurrence": entry["occurrence"],
                }

    varying = []
    for key in ordered:
        species, unit, occurrence = key
        values = np.asarray(
            [
                _get_species_concentration(
                    cv_obj,
                    species,
                    default=0.0,
                    unit=unit,
                    occurrence=occurrence,
                )
                for cv_obj in cvs
            ],
            dtype=float,
        )
        if not np.allclose(values, values[0], atol=atol, rtol=0):
            varying.append(metadata[key])

    return varying


def _infer_varying_species(cvs, atol=0.0):
    """
    Infer which species concentration varies across the CV list, treating
    absence from the filename as 0 concentration.

    Returns
    -------
    varying_species : list[str]
        Species whose concentrations are not all identical across cvs.
    """
    return [entry["species"] for entry in _infer_varying_concentration_entries(cvs, atol=atol)]

def _resolve_varying_x(cvs, options=None, do_print=True):
    """
    Determine what varies across a CV list and return raw x values.

    Returns
    -------
    x_raw : np.ndarray
    x_label : str
    x_kind : str
        'scan rate' or 'concentration'
    extra : dict
        e.g. {'species': species}
    """
    if options is None:
        options = {}

    similar, different = echem_similar_different(cvs)

    x_raw = []
    extra = {}

    if "scan rate" in different:
        x_label = "Scan Rate (V/s)"
        x_kind = "scan rate"
        x_raw = [cv.scan_rate for cv in cvs]
        extra["unit"] = "V/s"

    elif "compounds" in different or "concentrations" in different:
        species = options.get("species", None)
        concentration_entry = None

        if species is None:
            varying_entries = _infer_varying_concentration_entries(cvs)
            if len(varying_entries) == 1:
                concentration_entry = varying_entries[0]
                species = concentration_entry["species"]
            else:
                if do_print:
                    print(
                        "Could not uniquely determine which species concentration varies.\n"
                        "Please provide options['species']."
                    )
                    if len(varying_entries) > 0:
                        candidates = [
                            (
                                f"{entry['species']} ({entry['unit']})"
                                if entry.get("unit") else entry["species"]
                            )
                            for entry in varying_entries
                        ]
                        print("Candidate varying species:", ", ".join(candidates))
                return None, None, None, None
        else:
            matching_entries = [
                entry for entry in _infer_varying_concentration_entries(cvs)
                if entry["species"] == species
            ]
            if len(matching_entries) == 1:
                concentration_entry = matching_entries[0]

        unit = concentration_entry["unit"] if concentration_entry else "M"
        occurrence = concentration_entry["occurrence"] if concentration_entry else None

        if unit == "x":
            x_label = species
            x_kind = "mole fraction"
        else:
            x_label = f"{species} Concentration (M)"
            x_kind = "concentration"

        x_raw = [
            _get_species_concentration(
                cv,
                species,
                default=0.0,
                unit=unit if concentration_entry else None,
                occurrence=occurrence,
            )
            for cv in cvs
        ]
        extra["species"] = species
        extra["unit"] = unit

    else:
        if do_print:
            print("All CVs in list must differ in concentration OR scan rate.")
        return None, None, None, None

    return np.asarray(x_raw, dtype=float), x_label, x_kind, extra


def _option_was_provided(options, option_name):
    if isinstance(options, dict):
        wanted = normalize_key(option_name)
        return any(normalize_key(key) == wanted for key in options)
    return hasattr(options, normalize_key(option_name))


def _artist_color(artist):
    if artist is None:
        return None
    if hasattr(artist, "get_facecolors"):
        colors = artist.get_facecolors()
        if colors is not None and len(colors) > 0:
            return mpl.colors.to_hex(colors[0])
    if hasattr(artist, "get_color"):
        color = artist.get_color()
        if color is not None:
            return mpl.colors.to_hex(color)
    return None


def _is_color_sequence(value):
    if value is None or isinstance(value, str):
        return False
    try:
        if mpl.colors.is_color_like(value):
            return False
    except Exception:
        pass
    return isinstance(value, (list, tuple, np.ndarray, pd.Series))


def _fit_color_from_options(options, index=0, fallback=None):
    options = options or {}
    value = options.get("fit colors")
    if value is None:
        value = options.get("fit color")
    if value is None:
        return fallback
    if _is_color_sequence(value):
        values = list(value)
        if not values:
            return fallback
        if index < len(values):
            return values[index]
        return values[-1]
    return value


def _options_with_default_fit_color(options, raw_options, color, index=0):
    resolved = dict(options or {})
    if _option_was_provided(raw_options, "fit colors") or _option_was_provided(raw_options, "fit color"):
        selected = _fit_color_from_options(resolved, index=index, fallback=color)
        if selected is not None:
            resolved["fit color"] = selected
        resolved.pop("fit colors", None)
    elif color is not None:
        resolved["fit color"] = color
    return resolved


def _plot_all_multiplot_options(options, raw_options):
    routed = _multiplot_options_from_mapping(options)
    if not _option_was_provided(raw_options, "legend"):
        routed["legend"] = True
    return routed


def _axis_label_unit_parts(label, default_unit=""):
    text = str(label)
    match = re.match(r"^(.*?)\s*\(([^()]*)\)\s*$", text)
    if match:
        return match.group(1).strip(), match.group(2).strip()
    return text, default_unit


def _format_power_text(power):
    value = float(power)
    if value == 0.5:
        return "1/2"
    if value.is_integer():
        return str(int(value))
    return f"{value:g}"


def _format_symbolic_axis_label(label, unit="", x_kind="custom", transform="", log=False):
    base_label, parsed_unit = _axis_label_unit_parts(label, default_unit=unit)
    unit = unit or parsed_unit
    formatted_unit = _format_axis_unit(unit, wrap=False)
    if x_kind == "concentration" and str(base_label).endswith(" Concentration"):
        base_label = str(base_label)[:-len(" Concentration")]
    formatted_label = format_chemical_formulas(str(base_label), mode="mathtext")
    if x_kind == "mole fraction":
        base = rf"$\chi$({formatted_label})"
        unit = ""
        formatted_unit = ""
    else:
        base = f"[{formatted_label}]" if x_kind == "concentration" else formatted_label

    if log:
        if formatted_unit:
            return rf"$\log_{{10}}$({base} / {formatted_unit})"
        return rf"$\log_{{10}}$({base})"

    operation, value = _normalize_transform_token(transform)

    if operation == "identity":
        return f"{base} ({formatted_unit})" if formatted_unit else base
    if operation == "power":
        power = _format_power_text(value)
        label_text = f"{base}$^{{{power}}}$"
        return label_text + (f" ({formatted_unit}$^{{{power}}}$)" if formatted_unit else "")
    if operation == "reciprocal":
        return f"1/{base}" + (f" ({formatted_unit}$^{{-1}}$)" if formatted_unit else "")
    if operation == "log10":
        if formatted_unit:
            return rf"$\log_{{10}}$({base} / {formatted_unit})"
        return rf"$\log_{{10}}$({base})"
    if operation == "ln":
        if formatted_unit:
            return rf"$\ln$({base} / {formatted_unit})"
        return rf"$\ln$({base})"
    if operation == "abs":
        return f"|{base}|" + (f" ({formatted_unit})" if formatted_unit else "")

    transform_label = _format_transform_label(operation, value)
    return f"{transform_label}({base})" + (f" ({formatted_unit})" if formatted_unit else "")


def _transform_x_values(x_raw, options=None, default="identity"):
    """
    Transform raw x values for fitting / plotting.
    """
    if options is None:
        options = {}

    transform = options.get("x transform", default)

    if callable(transform):
        return np.asarray(transform(x_raw), dtype=float), str(getattr(transform, "__name__", "custom"))

    if transform in (None, "identity"):
        return np.asarray(x_raw, dtype=float), ""

    if transform == "sqrt":
        if np.any(x_raw < 0):
            raise ValueError("sqrt transform requires all x values to be >= 0.")
        return np.sqrt(x_raw), "sqrt"

    if transform == "log10":
        if np.any(x_raw <= 0):
            raise ValueError("log10 transform requires all x values to be > 0.")
        return np.log10(x_raw), "log10"

    if transform == "ln":
        if np.any(x_raw <= 0):
            raise ValueError("ln transform requires all x values to be > 0.")
        return np.log(x_raw), "ln"

    raise ValueError(f"Unknown x transform: {transform}")


def _normalize_transform_token(transform):
    """
    Normalize a user-facing transform option into a canonical operation.

    Numeric transform values mean "raise to this power", so options can use
    2, 3, 0.5, "^2", "^0.5", "square", or "square root".
    """
    if callable(transform):
        return "callable", transform

    if transform in (None, "", "identity", "none", False):
        return "identity", None

    if isinstance(transform, Real) and not isinstance(transform, bool):
        return "power", float(transform)

    text = str(transform).strip().lower()
    text = text.replace("_", " ").replace("-", " ")
    compact = text.replace(" ", "")

    aliases = {
        "identity": ("identity", None),
        "none": ("identity", None),
        "sqrt": ("power", 0.5),
        "squareroot": ("power", 0.5),
        "root": ("power", 0.5),
        "square": ("power", 2.0),
        "squared": ("power", 2.0),
        "cube": ("power", 3.0),
        "cubed": ("power", 3.0),
        "reciprocal": ("reciprocal", None),
        "inverse": ("reciprocal", None),
        "1/x": ("reciprocal", None),
        "log10": ("log10", None),
        "log": ("log10", None),
        "ln": ("ln", None),
        "naturallog": ("ln", None),
        "abs": ("abs", None),
        "absolute": ("abs", None),
    }
    if compact in aliases:
        return aliases[compact]

    if compact.startswith("^"):
        return "power", float(compact[1:])

    if compact.startswith("power"):
        power_text = compact.replace("power", "", 1).lstrip(":=")
        return "power", float(power_text)

    try:
        return "power", float(compact)
    except ValueError as exc:
        raise ValueError(f"Unknown transform: {transform}") from exc


def _format_transform_label(operation, value=None):
    if operation == "identity":
        return ""
    if operation == "power":
        if value == 0.5:
            return "sqrt"
        if float(value).is_integer():
            return f"^{int(value)}"
        return f"^{value:g}"
    if operation == "reciprocal":
        return "1/"
    if operation == "callable":
        return str(getattr(value, "__name__", "custom"))
    return operation


def _power_allows_negative_base(power):
    return float(power).is_integer()


def _transform_valid_mask(values, operation, value=None):
    values = np.asarray(values, dtype=float)
    keep = np.isfinite(values)

    if operation in ("log10", "ln"):
        keep &= values > 0
    elif operation == "reciprocal":
        keep &= values != 0
    elif operation == "power":
        if not _power_allows_negative_base(value):
            keep &= values >= 0

    return keep


def _transform_requires_positive_values(operation, value=None):
    return operation in ("log10", "ln", "reciprocal") or (
        operation == "power" and not _power_allows_negative_base(value)
    )


def _floor_options_present(options):
    options = {} if options is None else options
    return any(options.get(key) not in (None, False, "", "none") for key in ("floor", "x floor", "y floor"))


def _parse_floor_option(value, values, axis):
    if value in (None, False, "", "none"):
        return None
    if value is True:
        value = "0.1x"

    if isinstance(value, str):
        text = value.strip().lower()
        if text.endswith("x"):
            relative = float(text[:-1].strip())
            if relative <= 0:
                raise ValueError(f"'{axis} floor' relative values must be positive.")
            values = np.asarray(values, dtype=float)
            positive = values[np.isfinite(values) & (values > 0)]
            if len(positive) == 0:
                raise ValueError(
                    f"Cannot apply {axis}-axis relative floor because there are no positive finite values."
                )
            return float(np.nanmin(positive) * relative)
        return float(text)

    return float(value)


def _resolve_axis_floor(values, axis, options):
    options = {} if options is None else options
    axis_floor = _parse_floor_option(options.get(f"{axis} floor"), values, axis)
    if axis_floor is not None:
        return axis_floor
    return _parse_floor_option(options.get("floor"), values, axis)


def _apply_floor_for_transform(values, operation, value, axis, options):
    values = np.asarray(values, dtype=float)
    source = values.copy()
    notes = np.full(len(values), "", dtype=object)

    if not _floor_options_present(options):
        return source, notes
    if not _transform_requires_positive_values(operation, value):
        return source, notes

    floor = _resolve_axis_floor(source, axis, options)
    if floor is None:
        return source, notes

    floor_mask = np.isfinite(source) & (source < floor)
    if not np.any(floor_mask):
        return source, notes

    source[floor_mask] = floor
    notes[floor_mask] = f"floor: {floor:g}"
    return source, notes


def _combine_transform_notes(primary, secondary):
    primary = np.asarray(primary, dtype=object)
    secondary = np.asarray(secondary, dtype=object)
    combined = np.full(len(primary), "", dtype=object)
    for idx, (left, right) in enumerate(zip(primary, secondary)):
        left = str(left) if left not in (None, "") else ""
        right = str(right) if right not in (None, "") else ""
        if left and right and left != right:
            combined[idx] = f"{left}; {right}"
        else:
            combined[idx] = left or right
    return combined


def _transform_values(values, transform="identity"):
    """
    Transform numeric values and return (transformed, label, metadata).

    This helper intentionally accepts both scientific names and lightweight
    notebook-friendly shorthand:
    - 2, 3, 0.5
    - "^2", "^3", "^0.5"
    - "square", "cube", "sqrt", "square root"
    """
    values = np.asarray(values, dtype=float)
    operation, value = _normalize_transform_token(transform)
    label = _format_transform_label(operation, value)
    meta = {"operation": operation, "label": label}

    if operation == "identity":
        return values.copy(), label, meta

    if operation == "callable":
        return np.asarray(value(values), dtype=float), label, meta

    keep = _transform_valid_mask(values, operation, value)
    if not np.all(keep):
        raise ValueError(f"{label or operation} transform received invalid values.")

    if operation == "abs":
        return np.abs(values), label, meta

    if operation == "power":
        meta["power"] = float(value)
        return values ** float(value), label, meta

    if operation == "reciprocal":
        return 1 / values, label, meta

    if operation == "log10":
        return np.log10(values), label, meta

    if operation == "ln":
        return np.log(values), label, meta

    raise ValueError(f"Unknown transform operation: {operation}")


def _resolve_xy_transforms(options, default_x="identity", default_y="identity"):
    mode = options.get("transform mode", "identity")
    mode_text = str(mode).strip().lower().replace("_", "-").replace(" ", "-")

    x_transform = default_x
    y_transform = default_y

    if options.get("plot log-log", False) or options.get("log log plot", False):
        mode_text = "log-log"

    if mode_text in ("identity", "none", ""):
        pass
    elif mode_text in ("sqrt-x", "square-root-x"):
        x_transform = "sqrt"
    elif mode_text in ("log-x", "log10-x"):
        x_transform = "log10"
    elif mode_text in ("log-log", "loglog"):
        x_transform = "log10"
        y_transform = "log10"
    elif mode_text in ("lineweaver-burk", "lineweaver", "burk", "reciprocal"):
        x_transform = "reciprocal"
        y_transform = "reciprocal"
    else:
        raise ValueError(f"Unknown transform mode: {mode}")

    explicit_x = options.get("x transform")
    explicit_y = options.get("y transform")

    if explicit_x not in (None, "auto", "identity"):
        x_transform = options.get("x transform")
    if explicit_y not in (None, "auto", "identity"):
        y_transform = options.get("y transform")

    return x_transform, y_transform, mode_text


def _apply_scatter_transforms(x_raw, y_raw, x_transform, y_transform, options=None):
    x_raw = np.asarray(x_raw, dtype=float)
    y_raw = np.asarray(y_raw, dtype=float)
    options = {} if options is None else options

    x_op, x_value = _normalize_transform_token(x_transform)
    y_op, y_value = _normalize_transform_token(y_transform)

    x_source, x_notes = _apply_floor_for_transform(
        x_raw,
        x_op,
        x_value,
        "x",
        options,
    )
    y_source = y_raw.copy()
    y_notes = np.full(len(y_raw), "", dtype=object)
    y_positive_transform = _transform_requires_positive_values(y_op, y_value)
    finite_y = y_source[np.isfinite(y_source)]
    if y_positive_transform and len(finite_y) > 0 and np.all(finite_y < 0):
        y_source = np.abs(y_source)
        y_notes[:] = "abs fallback"
    y_source, y_floor_notes = _apply_floor_for_transform(
        y_source,
        y_op,
        y_value,
        "y",
        options,
    )
    y_notes = _combine_transform_notes(y_notes, y_floor_notes)

    keep = _transform_valid_mask(x_source, x_op, x_value)
    keep &= _transform_valid_mask(y_source, y_op, y_value)

    transformed_x, x_label, x_meta = _transform_values(x_source[keep], x_transform)
    transformed_y, y_label, y_meta = _transform_values(y_source[keep], y_transform)

    return {
        "keep": keep,
        "x": transformed_x,
        "y": transformed_y,
        "x input": x_source[keep],
        "y input": y_source[keep],
        "x label": x_label,
        "y label": y_label,
        "x meta": x_meta,
        "y meta": y_meta,
        "x note": x_notes[keep],
        "y note": y_notes[keep],
        "x input changed": bool(np.any(x_source[keep] != x_raw[keep])) if np.any(keep) else False,
        "y input changed": bool(np.any(y_source[keep] != y_raw[keep])) if np.any(keep) else False,
        "dropped": int(np.count_nonzero(~keep)),
    }


def _normalize_y_mode(y_mode):
    if y_mode in (None, "", False, "none", "raw", "identity"):
        return "raw"

    text = str(y_mode).strip().lower().replace("_", " ").replace("-", " ")
    compact = text.replace(" ", "")

    aliases = {
        "raw": "raw",
        "identity": "raw",
        "delta": "delta",
        "yy0": "delta",
        "y-y0": "delta",
        "k-k0": "delta",
        "negative delta": "negative delta",
        "negativedelta": "negative delta",
        "y0y": "negative delta",
        "y0-y": "negative delta",
        "k0-k": "negative delta",
        "ratio": "ratio",
        "yy0ratio": "ratio",
        "y/y0": "ratio",
        "k/k0": "ratio",
        "enhancement": "enhancement",
        "relative change": "enhancement",
        "relativechange": "enhancement",
        "fractional": "enhancement",
        "fractional change": "enhancement",
        "fractionalchange": "enhancement",
        "yy01": "enhancement",
        "yy0-1": "enhancement",
        "y/y01": "enhancement",
        "y/y0-1": "enhancement",
        "kk01": "enhancement",
        "kk0-1": "enhancement",
        "k/k01": "enhancement",
        "k/k0-1": "enhancement",
    }

    if text in aliases:
        return aliases[text]
    if compact in aliases:
        return aliases[compact]

    raise ValueError("'y mode' must be 'raw', 'delta', 'negative delta', 'ratio', or 'enhancement'.")


def _resolve_y0_override(y0_option, series_keys):
    if y0_option is None:
        return None

    if isinstance(y0_option, dict):
        candidate_keys = []
        for key in series_keys:
            if key is None:
                continue
            candidate_keys.extend([key, str(key), str(key).strip().lower()])
        candidate_keys.extend(["default", "all"])

        normalized = {}
        for key, value in y0_option.items():
            normalized[key] = value
            normalized[str(key)] = value
            normalized[str(key).strip().lower()] = value

        for key in candidate_keys:
            if key in normalized:
                return normalized[key]
        return None

    return y0_option


def _first_finite_y0(values):
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    if len(finite) == 0:
        raise ValueError("Could not resolve y0 because the y series has no finite values.")
    return float(finite[0])


def _apply_y_mode(y_raw, options, series_keys=None):
    y_raw = np.asarray(y_raw, dtype=float)
    mode = _normalize_y_mode(options.get("y mode"))
    series_keys = [] if series_keys is None else list(series_keys)

    y0_value = _resolve_y0_override(options.get("y0"), series_keys)
    y0 = _first_finite_y0(y_raw) if y0_value is None else float(y0_value)

    if mode == "raw":
        adjusted = y_raw.copy()
    elif mode == "delta":
        adjusted = y_raw - y0
    elif mode == "negative delta":
        adjusted = y0 - y_raw
    elif mode == "ratio":
        if y0 == 0:
            raise ValueError("'ratio' y mode requires nonzero y0.")
        adjusted = y_raw / y0
    elif mode == "enhancement":
        if y0 == 0:
            raise ValueError("'enhancement' y mode requires nonzero y0.")
        adjusted = y_raw / y0 - 1
    else:  # pragma: no cover - guarded by _normalize_y_mode
        raise ValueError(f"Unknown y mode: {mode}")

    return {
        "raw": y_raw,
        "adjusted": adjusted,
        "mode": mode,
        "y0": y0,
    }


def _format_baseline_axis_label(label):
    label = str(label)
    if len(label) >= 2 and label.startswith("$") and label.endswith("$"):
        return f"{label[:-1]}^{{0}}$"
    return f"{label}$^{{0}}$"


def _format_y_mode_axis_label(label, y_mode):
    mode = _normalize_y_mode(y_mode)
    label = str(label)
    baseline_label = _format_baseline_axis_label(label)

    if mode == "raw":
        return label
    if mode == "delta":
        return f"{label} - {baseline_label}"
    if mode == "negative delta":
        return f"{baseline_label} - {label}"
    if mode == "ratio":
        return f"{label}/{baseline_label}"
    if mode == "enhancement":
        return f"{label}/{baseline_label} - 1"

    raise ValueError(f"Unknown y mode: {mode}")


def _format_y_transform_axis_label(label, transform):
    operation, value = _normalize_transform_token(transform)

    if operation == "identity":
        return str(label)
    if operation == "log10":
        return rf"$\log_{{10}}$({label})"
    if operation == "ln":
        return rf"$\ln$({label})"
    if operation == "reciprocal":
        return f"1/({label})"
    if operation == "power":
        transform_label = _format_transform_label(operation, value)
        return f"({label}){transform_label}"

    transform_label = _format_transform_label(operation, value)
    return f"{transform_label}({label})" if transform_label else str(label)


def _format_fit_peak_current_y_label(y_name, unit, cv_obj):
    axis_label = cv_obj.format_axis_label(y_name, unit)
    if str(y_name).strip().lower().replace(" ", "") == "i/ip0":
        return r"$i_p / i_p^0$"
    if axis_label == r"$i / i_p^0$":
        return r"$i_p / i_p^0$"
    return f"Peak {axis_label}"


def _inverse_transform_values(values, transform):
    values = np.asarray(values, dtype=float)
    operation, value = _normalize_transform_token(transform)

    if operation == "identity":
        return values.copy()
    if operation == "log10":
        return 10 ** values
    if operation == "ln":
        return np.exp(values)
    if operation == "reciprocal":
        return 1 / values
    if operation == "power":
        power = float(value)
        if power == 0:
            raise ValueError("Cannot invert a zero-power transform.")
        return values ** (1 / power)

    raise ValueError(f"Cannot invert transform: {transform}")


def _inverse_y_mode_values(adjusted, y_mode, y0):
    adjusted = np.asarray(adjusted, dtype=float)
    mode = _normalize_y_mode(y_mode)

    if mode == "raw":
        return adjusted.copy()

    y0 = float(y0)
    if mode == "delta":
        return adjusted + y0
    if mode == "negative delta":
        return y0 - adjusted
    if mode == "ratio":
        return adjusted * y0
    if mode == "enhancement":
        return (adjusted + 1) * y0

    raise ValueError(f"Unknown y mode: {mode}")


def _fit_peak_potential_legacy_return(cvs, options={}):
    """
    Fit peak potential vs log(scan rate) or log(concentration).

    New behavior
    ------------
    - Supports either 'segment' or 'segments'
    - Supports multiple segments on one plot
    - Optional 'follow E1/2' behavior for sequential segments
    - Reuses existing fit() and scaling helpers

    Suggested options
    -----------------
    'segment' : int or None
    'segments' : list[int] or None
    'follow E1/2' : bool, default False
    'x unit' : str or 'auto', optional
    'y unit' : str or 'auto', optional
    'fit indices' : [i0, i1] or {i0, i1], ...}
    """
    raw_options = options
    typed_options = FitPeakPotentialOptions.from_options(options)
    options = typed_options.to_legacy_dict()
    do_plot = options.get("plot", True)
    do_print = options.get("print", True)
    do_fit = options.get("fit", True)
    follow_e_half = options.get("follow e1/2", False)
    fit_color_index = 0

    if not do_plot:
        options["plot fit"] = False

    if not isinstance(cvs, list) or len(cvs) == 0:
        if do_print:
            print("Must provide a non-empty list of CVs.")
        return None, None

    segments = _normalize_segment_option(options)

    x_raw, x_label, x_kind, extra = _resolve_varying_x(cvs, options, do_print=do_print)
    if x_raw is None:
        return None, None

    x_raw = np.asarray(x_raw, dtype=float)

    positive_x_mask = np.isfinite(x_raw) & (x_raw > 0)
    if not np.any(positive_x_mask):
        raise ValueError(
            "fit_peak_potential requires at least one positive scan rate or concentration "
            "for log10(x) analysis."
        )
    if np.count_nonzero(~positive_x_mask) > 0 and do_print:
        print(
            f"fit_peak_potential: excluded {np.count_nonzero(~positive_x_mask)} "
            "non-positive or non-finite x value(s) before log10 transform."
        )

    x_for_fit = np.full_like(x_raw, np.nan, dtype=float)
    x_for_fit[positive_x_mask] = np.log10(x_raw[positive_x_mask])
    x_plot_label = _format_symbolic_axis_label(
        x_label,
        unit=extra.get("unit", ""),
        x_kind=x_kind,
        log=True,
    )

    # Use the CV x-axis unit helper path for Ep / E1/2 display
    ep_unit = _axis_common_unit(
        cvs,
        lambda cv: (cv.x(options), cv.x(options).name),
        options.get("y unit", "auto"),
    )

    # --------------------------------------------------
    # Collect Ep values
    # --------------------------------------------------
    internal_options = _analysis_options_for(PeakPotentialOptions, options)
    internal_options["internal call"] = True
    internal_options["new plot"] = False
    internal_options["plot"] = options.get("plot all", False)
    internal_options["print"] = options.get("print all", False)

    # Important: do not let peak_potential see a multi-segment request
    internal_options.pop("segments", None)
    internal_options.pop("plot segment", None)
    internal_options.pop("plot segments", None)

    # If plotting all intermediate work, first show the CVs together
    if options.get("plot all", False):
        multiplot(cvs, options=_plot_all_multiplot_options(options, raw_options))

    rows = []
    x_name = None

    # Organize by CV first so follow E1/2 is actually tracked within a CV
    for i, cv_obj in enumerate(cvs):
        row = {
            "name": cv_obj.name,
            x_label: x_raw[i],
            "x transformed": x_for_fit[i],
            x_plot_label: x_for_fit[i],
        }

        x_arr = cv_obj.x(options)
        x_name = x_arr.name
        x_unit = cv_obj.units.get(x_name, "V")

        running_guess = options.get("guess potential", None)
        ep_by_segment = {}

        # Collect Ep for each requested segment
        for seg in segments:
            seg_options = internal_options.copy()

            if seg is not None:
                seg_options["segment"] = seg
            else:
                seg_options.pop("segment", None)

            if seg_options.get("exact potential") is None and running_guess is not None:
                seg_options["guess potential"] = running_guess
            else:
                seg_options.pop("guess potential", None)

            # One segment at a time so peak_potential does not search across combined segments
            Ep = cv_obj.peak_potential(seg_options)["Ep"]
            scaled_Ep, _ = scale_value(Ep, x_unit, selected_unit=ep_unit)

            if seg is None:
                row["Ep (V)"] = Ep
                row[f"Ep ({ep_unit})"] = scaled_Ep
            else:
                row[f"Seg {seg} Ep (V)"] = Ep
                row[f"Seg {seg} Ep ({ep_unit})"] = scaled_Ep
                ep_by_segment[seg] = Ep

            # Default behavior: use the most recent Ep as the next guess
            running_guess = Ep

        # Interpret "follow E1/2" as:
        # for sequential segments, compute and store the half-wave potential
        # between segment n and n+1 using the already collected Ep values.
        if follow_e_half:
            numeric_segments = sorted(ep_by_segment.keys())

            for seg in numeric_segments:
                if seg + 1 not in ep_by_segment:
                    continue

                E_half = midpoint_potential(
                    ep_by_segment[seg],
                    ep_by_segment[seg + 1],
                    options.get("sig figs",4)
                )
                delta_E = round_sigfigs(
                    abs(ep_by_segment[seg] - ep_by_segment[seg + 1]),
                    options.get("sig figs",4)
                )

                scaled_E_half, _ = scale_value(E_half, x_unit, selected_unit=ep_unit)
                scaled_delta_E, _ = scale_value(delta_E, x_unit, selected_unit=ep_unit)

                row[f"Seg {seg}-{seg+1} E1/2 (V)"] = E_half
                row[f"Seg {seg}-{seg+1} E1/2 ({ep_unit})"] = scaled_E_half
                row[f"Seg {seg}-{seg+1} ΔE (V)"] = delta_E
                row[f"Seg {seg}-{seg+1} ΔE ({ep_unit})"] = scaled_delta_E

        rows.append(row)

    data = pd.DataFrame(rows)

    x_col = x_plot_label

    # --------------------------------------------------
    # Plot
    # --------------------------------------------------
    fits = {}
    fit_rows = []
    fit_model_results = {}
    fit_model_results = {}

    if do_plot:
        plt.figure()
        plt.xlabel(x_col)
        y_axis_label = _format_y_mode_axis_label(
            f"Peak {cvs[0].format_axis_label(x_name, ep_unit)}",
            options.get("y mode"),
        )
        plt.ylabel(y_axis_label)

    # Plot / fit Ep series
    for seg in segments:
        if seg is None:
            y_col = f"Ep ({ep_unit})"
            fit_key = "Ep"
            label = "Ep"
        else:
            y_col = f"Seg {seg} Ep ({ep_unit})"
            fit_key = f"Seg {seg} Ep"
            label = f"Seg {seg} Ep"

        if y_col not in data.columns:
            continue

        y_adjustment = _apply_y_mode(
            data[y_col].to_numpy(dtype=float),
            options,
            series_keys=[fit_key, label, y_col, "Ep", "default"],
        )
        if len(segments) == 1 and not follow_e_half:
            adjusted_col = "y adjusted"
            transformed_y_col = "y transformed"
            data["y raw"] = y_adjustment["raw"]
            data["y adjusted"] = y_adjustment["adjusted"]
            data["y transformed"] = y_adjustment["adjusted"]
            data["y0"] = y_adjustment["y0"]
            data["y mode"] = y_adjustment["mode"]
        else:
            adjusted_col = f"{fit_key} y adjusted"
            transformed_y_col = f"{fit_key} y transformed"
            data[f"{fit_key} y raw"] = y_adjustment["raw"]
            data[adjusted_col] = y_adjustment["adjusted"]
            data[transformed_y_col] = y_adjustment["adjusted"]
            data[f"{fit_key} y0"] = y_adjustment["y0"]
            data[f"{fit_key} y mode"] = y_adjustment["mode"]

        seg_data = data[[x_col, adjusted_col]].dropna().sort_values(by=x_col).reset_index(drop=True)

        point_color = None
        if do_plot:
            point_color = _artist_color(plt.scatter(seg_data[x_col], seg_data[adjusted_col], label=label))

        idxs = options.get("fit indices")
        if isinstance(idxs, dict):
            idxs = idxs.get(fit_key, idxs.get(seg, idxs.get("default", None)))
        if idxs is None:
            idxs = [0, len(seg_data)]

        if do_fit:
            fit_specs = _fit_rate_range_specs(options, idxs, fit_key)
            for range_label, fit_spec, is_fit_range in fit_specs:
                fit_x, fit_y = _fit_rate_selected_points(
                    seg_data[x_col].to_numpy(),
                    seg_data[adjusted_col].to_numpy(),
                    fit_spec,
                    is_fit_range,
                )
                output_key = fit_key if not is_fit_range else f"{fit_key} {range_label}"
                display_label = label if not is_fit_range else f"{label} {range_label}"
                fit_label = f"{display_label} Fit"
                series_fit = _fit_series_xy(
                    fit_x,
                    fit_y,
                    options=options,
                    label=output_key,
                )
                if do_plot and options["plot fit"]:
                    plot_options = _fit_rate_fit_options_for_range(options, range_label, is_fit_range)
                    plot_options = _options_with_default_fit_color(
                        plot_options,
                        raw_options,
                        point_color,
                        index=fit_color_index,
                    )
                    fit_color_index += 1
                    plot_options.update({"new plot": False, "plot data": False, "model label": fit_label})
                    _plot_fit_model_result(series_fit["model_result"], plot_options)
                fits[output_key] = series_fit["fits"]
                fit_model_results[display_label] = series_fit["model_result"]
                fit_rows.extend(series_fit["fit_rows"])

    # Plot / fit E1/2 series for adjacent sequential segment pairs
    if follow_e_half:
        numeric_segments = sorted([seg for seg in segments if isinstance(seg, int)])

        for seg in numeric_segments:
            if seg + 1 not in numeric_segments:
                continue

            y_col = f"Seg {seg}-{seg+1} E1/2 ({ep_unit})"
            fit_key = f"Seg {seg}-{seg+1} E1/2"
            label = f"Seg {seg}-{seg+1} E1/2"

            if y_col not in data.columns:
                continue

            y_adjustment = _apply_y_mode(
                data[y_col].to_numpy(dtype=float),
                options,
                series_keys=[fit_key, label, y_col, "E1/2", "default"],
            )
            adjusted_col = f"{fit_key} y adjusted"
            data[f"{fit_key} y raw"] = y_adjustment["raw"]
            data[adjusted_col] = y_adjustment["adjusted"]
            data[f"{fit_key} y transformed"] = y_adjustment["adjusted"]
            data[f"{fit_key} y0"] = y_adjustment["y0"]
            data[f"{fit_key} y mode"] = y_adjustment["mode"]

            ehalf_data = data[[x_col, adjusted_col]].dropna().sort_values(by=x_col).reset_index(drop=True)

            if do_plot:
                plt.scatter(
                    ehalf_data[x_col],
                    ehalf_data[adjusted_col],
                    label=label,
                )

            idxs = options.get("fit indices")
            if isinstance(idxs, dict):
                idxs = idxs.get(fit_key, idxs.get("default", None))
            if idxs is None:
                idxs = [0, len(ehalf_data)]

            if do_fit:
                fit_specs = _fit_rate_range_specs(options, idxs, fit_key)
                for range_label, fit_spec, is_fit_range in fit_specs:
                    fit_x, fit_y = _fit_rate_selected_points(
                        ehalf_data[x_col].to_numpy(),
                        ehalf_data[adjusted_col].to_numpy(),
                        fit_spec,
                        is_fit_range,
                    )
                    output_key = fit_key if not is_fit_range else f"{fit_key} {range_label}"
                    display_label = label if not is_fit_range else f"{label} {range_label}"
                    fit_label = f"{display_label} Fit"
                    series_fit = _fit_series_xy(
                        fit_x,
                        fit_y,
                        options=options,
                        label=output_key,
                    )
                    if do_plot and options["plot fit"]:
                        plot_options = _fit_rate_fit_options_for_range(options, range_label, is_fit_range)
                        plot_options.update({"new plot": False, "plot data": False, "model label": fit_label})
                        _plot_fit_model_result(series_fit["model_result"], plot_options)
                    fits[output_key] = series_fit["fits"]
                    fit_model_results[output_key] = series_fit["model_result"]
                    fit_rows.extend(series_fit["fit_rows"])

    if do_plot and _scatterfit_legend_requested(options):
        plt.legend(fontsize=_scatterfit_legend_fontsize(options))

    _attach_scatter_fit_table(data, fit_rows)
    data.attrs["fit model results"] = fit_model_results
    if do_print:
        _print_fit_model_results(fit_model_results, options)

    if len(fits) == 1:
        return data, next(iter(fits.values()))
    return data, fits


def fit_peak_potential(cvs, options={}):
    """Fit peak potential values across a CV series.
    
    Parameters
    ----------
    cvs : sequence of cv
        CV objects whose peak potentials are measured and fitted.
    options : dict or FitPeakPotentialOptions, optional
        Peak-potential, transform, fit-window, print, and plot options. See ``e.describe_options("fit_peak_potential")``.
    
    Returns
    -------
    ScatterFitResult
        Fit result with plotted data, fit coefficients, and fit statistics.
    
    Examples
    --------
    >>> result = e.fit_peak_potential(cvs, {"guess potential": -1.5, "segments": [1, 2]})
    """
    return _scatter_result_from_legacy(
        _fit_peak_potential_legacy_return(cvs, options),
        summary={"analysis": "peak potential fit"},
    )


def _fit_peak_current_legacy_return(cvs, options={}):
    """
    Fit peak current (ip) vs scan rate or concentration, using x^x_power scaling.

    Behavior
    --------
    - Uses _resolve_varying_x(...) to determine whether scan rate or concentration varies.
    - Uses x**x_power for the x-axis transform.
    - Supports either 'segment' or 'segments'.
    - If multiple segments are requested, analyzes one segment at a time and returns
      one fit per segment.
    """
    raw_options = options
    typed_options = FitPeakCurrentOptions.from_options(options)
    options = typed_options.to_legacy_dict()
    do_plot = options.get("plot", True)
    do_print = options.get("print", True)
    do_fit = options.get("fit", True)
    x_power = options.get("x power", 0.5)
    fit_color_index = 0

    if not do_plot:
        options["plot fit"] = False

    x_raw, x_label, x_kind, extra = _resolve_varying_x(cvs, options, do_print=do_print)
    if x_raw is None:
        return None, None

    x_raw = np.asarray(x_raw, dtype=float)

    default_x_transform = "identity" if x_power is None else x_power
    x_transform, y_transform, mode_label = _resolve_xy_transforms(
        options,
        default_x=default_x_transform,
        default_y="identity",
    )

    data = pd.DataFrame()
    data[x_label] = x_raw
    data["x raw"] = x_raw
    data["x label"] = x_label
    data["x unit"] = extra.get("unit", "")
    data["x kind"] = x_kind

    segments = _normalize_segment_option(options)

    # Determine a common current unit, like Sevcik / fit_peak_current.
    peak_unit = _axis_common_unit(
        cvs,
        lambda cv: (cv.y(options), cv.y(options).name),
        options.get("y unit", "auto"),
    )

    # If plotting all intermediate work, first show the CVs together
    if options.get("plot all", False):
        multiplot(cvs, options=_plot_all_multiplot_options(options, raw_options))

    internal_options = typed_options.for_peak_current().to_legacy_dict()
    internal_options["internal call"] = True
    internal_options["new plot"] = False
    internal_options["plot"] = options.get("plot all", False)
    internal_options["print"] = options.get("print all", False)

    # Important: only pass one segment at a time downstream
    internal_options.pop("segments", None)
    internal_options.pop("plot segment", None)
    internal_options.pop("plot segments", None)

    fits = {}
    fit_rows = []
    fit_model_results = {}
    y_name = None
    y_axis_label = None

    for seg in segments:
        seg_options = internal_options.copy()
        if seg is not None and len(segments) > 1:
            seg_options["segment"] = seg
            series_label = f"Seg {seg}"
            col_name = f"Seg {seg} ip"
            fit_label = f"Seg {seg} Fit"
        else:
            series_label = "ip"
            col_name = "ip"
            fit_label = "Fit"

        ips = []
        for cv in cvs:
            y_arr = cv.y(options)
            y_name = y_arr.name
            y_unit = cv.units.get(y_name, "")

            peak_current = cv.peak_current(seg_options)["ip"]
            scaled_peak_current, _ = scale_value(
                peak_current,
                y_unit,
                selected_unit=peak_unit,
            )
            ips.append(scaled_peak_current)

        ips = np.asarray(ips, dtype=float)
        data[col_name] = ips
        y_axis_label = _format_fit_peak_current_y_label(y_name, peak_unit, cvs[0])
        data["y label"] = y_axis_label

        y_adjustment = _apply_y_mode(
            ips,
            options,
            series_keys=[series_label, col_name, seg, "ip", "default"],
        )
        adjusted_ips = y_adjustment["adjusted"]

        if len(segments) == 1:
            data["y raw"] = y_adjustment["raw"]
            data["y adjusted"] = adjusted_ips
            data["y0"] = y_adjustment["y0"]
            data["y mode"] = y_adjustment["mode"]
        else:
            data[f"{series_label} y raw"] = y_adjustment["raw"]
            data[f"{series_label} y adjusted"] = adjusted_ips
            data[f"{series_label} y0"] = y_adjustment["y0"]
            data[f"{series_label} y mode"] = y_adjustment["mode"]

        transformed = _apply_scatter_transforms(
            x_raw,
            adjusted_ips,
            x_transform,
            y_transform,
            options,
        )
        if transformed["dropped"] > 0 and do_print:
            print(
                f"fit_peak_current {series_label}: excluded {transformed['dropped']} "
                "non-finite or non-transformable point(s)."
            )

        fit_x_all = transformed["x"]
        fit_y_all = transformed["y"]

        if len(segments) == 1:
            data["x transformed"] = np.nan
            data["y transformed"] = np.nan
            data.loc[transformed["keep"], "x transformed"] = fit_x_all
            data.loc[transformed["keep"], "y transformed"] = fit_y_all
            if transformed["x input changed"]:
                data["x transform input"] = np.nan
                data.loc[transformed["keep"], "x transform input"] = transformed["x input"]
            if transformed["y input changed"]:
                data["y transform input"] = np.nan
                data.loc[transformed["keep"], "y transform input"] = transformed["y input"]
            data["x transform"] = transformed["x label"] or "identity"
            data["y transform"] = transformed["y label"] or "identity"
            if np.any(np.asarray(transformed["x note"], dtype=object) != ""):
                data["x transform note"] = ""
                data.loc[transformed["keep"], "x transform note"] = transformed["x note"]
            data["y transform note"] = ""
            data.loc[transformed["keep"], "y transform note"] = transformed["y note"]
        else:
            data[f"{series_label} y transformed"] = np.nan
            data.loc[transformed["keep"], f"{series_label} y transformed"] = fit_y_all

        if do_plot:
            plt.figure()
            point_color = _artist_color(
                plt.scatter(fit_x_all, fit_y_all, label=series_label if len(segments) > 1 else None)
            )
            _apply_matplotlib_axis_scales(plt.gca(), options)
        else:
            point_color = None

        idxs = options.get("fit indices")
        if isinstance(idxs, dict):
            idxs = idxs.get(seg, idxs.get("default", None))
        if idxs is None:
            idxs = [0, len(fit_x_all)]

        if do_fit:
            fit_specs = _fit_rate_range_specs(options, idxs, series_label)
            for range_label, fit_spec, is_fit_range in fit_specs:
                fit_x, fit_y = _fit_rate_selected_points(fit_x_all, fit_y_all, fit_spec, is_fit_range)

                if len(fit_x) < 2:
                    raise ValueError("fit_peak_current requires at least two transformable points.")

                output_key = seg if not is_fit_range else f"{series_label} {range_label}"
                display_label = series_label if not is_fit_range else f"{series_label} {range_label}"
                series_fit = _fit_series_xy(
                    fit_x,
                    fit_y,
                    options=options,
                    label=display_label,
                )
                if do_plot and options["plot fit"]:
                    plot_options = _fit_rate_fit_options_for_range(options, range_label, is_fit_range)
                    plot_options = _options_with_default_fit_color(
                        plot_options,
                        raw_options,
                        point_color,
                        index=fit_color_index,
                    )
                    fit_color_index += 1
                    plot_options.update({"new plot": False, "plot data": False, "model label": f"{display_label} Fit"})
                    _plot_fit_model_result(series_fit["model_result"], plot_options)
                fits[output_key] = series_fit["fits"]
                fit_model_results[display_label] = series_fit["model_result"]
                fit_rows.extend(series_fit["fit_rows"])

    if do_plot:
        y_transform_label = _format_transform_label(*_normalize_transform_token(y_transform))
        plt.xlabel(
            _format_symbolic_axis_label(
                x_label,
                unit=extra.get("unit", ""),
                x_kind=x_kind,
                transform=x_transform,
            )
        )
        y_label = y_axis_label or f"Peak {cvs[0].format_axis_label(y_name, peak_unit)}"
        y_label = _format_y_mode_axis_label(y_label, options.get("y mode"))
        if _normalize_y_mode(options.get("y mode")) == "raw":
            plt.ylabel(f"{y_transform_label}({y_label})" if y_transform_label else y_label)
        else:
            plt.ylabel(_format_y_transform_axis_label(y_label, y_transform))
        if _scatterfit_legend_requested(options):
            plt.legend(fontsize=_scatterfit_legend_fontsize(options))

    _attach_scatter_fit_table(data, fit_rows)
    data.attrs["fit model results"] = fit_model_results
    if do_print:
        _print_fit_model_results(fit_model_results, options)

    if len(segments) == 1:
        return data, fits[segments[0]]
    return data, fits


def fit_peak_current(cvs, options={}):
    """Fit peak current values across a CV series.
    
    Parameters
    ----------
    cvs : sequence of cv
        CV objects whose peak currents are measured and fitted.
    options : dict or FitPeakCurrentOptions, optional
        Peak-current, fit-window, print, and plot options. See ``e.describe_options("fit_peak_current")``.
    
    Returns
    -------
    ScatterFitResult
        Fit result with plotted data, fit coefficients, and fit statistics.
    
    Examples
    --------
    >>> result = e.fit_peak_current(cvs, {"guess potential": -1.5, "segment": 1})
    """
    return _scatter_result_from_legacy(
        _fit_peak_current_legacy_return(cvs, options),
        summary={"analysis": "peak current fit"},
    )

import warnings


def _fowa_summary_table(cvs, results, plot_data, ref_cvs, options=None):
    """
    Build the final FOWA table:
    - base object summary from build_object_table()
    - analysis columns renamed with pretty_table_column_label()
    - shared columns split out into attrs['shared_summary']
    """
    if options is None:
        options = {}

    summary_df, _meta = build_object_table(cvs, options)

    analysis_df = pd.DataFrame(results)
    if not analysis_df.empty:
        analysis_df["background tangent"] = [
            _format_fowa_line(slope, intercept)
            for slope, intercept in zip(
                analysis_df.get("background slope", pd.Series([None] * len(analysis_df))),
                analysis_df.get("background intercept", pd.Series([None] * len(analysis_df))),
            )
        ]
        analysis_df["wave range"] = [
            _format_fowa_pair(start, end)
            for start, end in zip(
                analysis_df.get("wave start", pd.Series([None] * len(analysis_df))),
                analysis_df.get("wave end", pd.Series([None] * len(analysis_df))),
            )
        ]
        analysis_df["fowa fit"] = [
            _format_fowa_line(slope, intercept)
            for slope, intercept in zip(
                analysis_df.get("slope", pd.Series([None] * len(analysis_df))),
                analysis_df.get("intercept", pd.Series([None] * len(analysis_df))),
            )
        ]

    analysis_df = analysis_df.rename(
        columns={c: pretty_table_column_label(c) for c in analysis_df.columns}
    )

    results_df = pd.concat(
        [summary_df.reset_index(drop=True), analysis_df.reset_index(drop=True)],
        axis=1,
    )

    keep_in_table = {
        "Name",
        "Plot Label",
        "Status",
    }

    display_source_df = results_df.drop(
        columns=[
            "Background Slope",
            "Background Intercept",
            "Wave Start",
            "Wave End",
            "Slope",
            "Intercept",
            "Background Tangent Potential",
            "TOFmax",
            "Warning Details",
        ],
        errors="ignore",
    )
    display_source_df = _order_fowa_display_columns(display_source_df, summary_df.columns)

    display_df, shared_summary = _split_shared_columns(
        display_source_df,
        keep_in_table=keep_in_table,
    )

    display_df.attrs["shared_summary"] = shared_summary
    display_df.attrs["plot_data"] = plot_data
    display_df.attrs["full_results_df"] = results_df
    display_df.attrs["ref_cvs"] = ref_cvs
    unit_map = _fowa_units_map(results_df)
    display_df.attrs["units"] = unit_map
    results_df.attrs["units"] = unit_map
    if "Warning Details" in results_df.columns:
        warning_details = {
            str(name): details
            for name, details in zip(
                results_df.get("Name", pd.Series(range(len(results_df)))),
                results_df["Warning Details"].fillna("").astype(str),
            )
            if str(details).strip()
        }
    else:
        warning_details = {}
    display_df.attrs["warnings"] = warning_details
    results_df.attrs["warnings"] = warning_details

    return display_df


def _fowa_summary_display_table(summary):
    return pd.DataFrame(
        [
            {"Field": str(key), "Value": "" if value is None else str(value)}
            for key, value in summary.items()
        ]
    )


def _fowa_summary_field_html_label(field):
    return _pretty_table_header_html_label(field)


def _display_fowa_summary_table(summary, options=None):
    options = {} if options is None else options
    table = _fowa_summary_display_table(summary)
    if table.empty:
        return table

    if not options.get("pretty print", True) or display is None:
        print(table.to_string(index=False, justify="left"))
        return table

    display_table = table.copy()
    display_table["Field"] = [
        _fowa_summary_field_html_label(field)
        for field in display_table["Field"]
    ]
    styled = (
        display_table.style
        .format(escape=None)
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
    display(styled)
    return table


def _auto_fowa_wave_bounds(x, y, redox_potential, options):
    """
    Sign-agnostic foot-of-wave windowing based on slope magnitude.
    """
    if len(x) < 7:
        raise ValueError("Not enough points to determine a FOWA wave window.")

    dx = np.diff(x)
    dx = dx[np.isfinite(dx)]
    delta_x = float(np.nanmedian(np.abs(dx))) if len(dx) > 0 else 1.0
    if not np.isfinite(delta_x) or delta_x == 0:
        delta_x = 1.0

    smoothed_y, slopes, _, sg_meta = _savgol_bundle(
        y,
        options,
        delta=delta_x,
    )

    weights = np.exp(
        -((x - redox_potential) / options["gaussian weight"]) ** 2
        - options["gaussian skew"] * (x - redox_potential)
    )

    slope_mag = np.abs(slopes)
    weighted_slopes = slope_mag * weights
    idx_max_slope = int(np.argmax(weighted_slopes))

    diff_slopes = np.abs(slope_mag - slope_mag[idx_max_slope])
    weighted_diff_slopes = diff_slopes * weights
    idx_delta = int(np.argmax(weighted_diff_slopes))
    delta = abs(idx_max_slope - idx_delta)

    n = max(0, idx_max_slope - delta)
    m = min(len(x), idx_max_slope + delta + 1)

    if (m - n) < 8:
        pad = max(4, delta)
        n = max(0, idx_max_slope - pad)
        m = min(len(x), idx_max_slope + pad + 1)

    meta = {
        "sg window": sg_meta["window"],
        "sg polyorder": sg_meta["polyorder"],
        "smoothed y": smoothed_y,
        "slopes": slopes,
        "weights": weights,
        "weighted slopes": weighted_slopes,
        "weighted diff slopes": weighted_diff_slopes,
        "idx max slope": idx_max_slope,
        "wave start index": n,
        "wave end index": m,
    }
    return n, m, meta


def _manual_fowa_wave_mask(x, wave_range, cv_name):
    lo, hi = sorted(float(v) for v in wave_range)
    x = np.asarray(x, dtype=float)
    mask = np.isfinite(x) & (x >= lo) & (x <= hi)
    idx = np.flatnonzero(mask)

    if len(idx) < 8:
        raise ValueError(
            f"Manual FOWA wave range [{lo}, {hi}] for '{cv_name}' selects "
            f"only {len(idx)} point(s); at least 8 are required."
        )

    return mask, {
        "manual": True,
        "wave range": [lo, hi],
        "wave start index": int(idx[0]),
        "wave end index": int(idx[-1]) + 1,
    }


def _display_analysis_equation(
    title_latex,
    title_text,
    equation,
    resolved=True,
    compact=False,
    include_definitions=True,
):
    """
    Display an analysis equation in notebook Math when available.
    Falls back to plain text in non-notebook contexts.
    """
    use_math_display = display is not None and Math is not None

    if use_math_display:
        display(Math(title_latex))
        display(Math(equation["symbolic latex"]))

        if resolved:
            display(Math(equation["resolved latex"]))

        if compact:
            display(Math(equation["compact latex"]))

        if include_definitions:
            display(Math(equation["definitions latex"]))

    else:
        print(f"[{title_text}]")
        print("  " + equation["symbolic"])
        if resolved:
            print("  " + equation["resolved"])
        if compact:
            print("  " + equation["compact"])
        if include_definitions:
            print("  " + equation["definitions"])

    return equation


def _equation_value_text(value, units=""):
    if value is None:
        return "not specified"
    suffix = f" {units}" if units else ""
    try:
        return f"{float(value):g}{suffix}"
    except (TypeError, ValueError):
        return f"{value}{suffix}"


def _format_sevcik_diffusion_equation(
    mode,
    num_electrons,
    temperature,
    electrode_area,
    concentration=None,
    scan_rate=None,
    scan_dependence=0.5,
):
    """
    Return display-ready text/LaTeX for Sevcik diffusion coefficient equations.
    """
    mode_text = str(mode).strip().lower()
    is_scan_rate_mode = "scan" in mode_text

    n = float(num_electrons)
    T = float(temperature)
    S = float(electrode_area)
    C = None if concentration is None else float(concentration)
    v = None if scan_rate is None else float(scan_rate)
    scan_dependence = float(scan_dependence)

    if is_scan_rate_mode:
        fit_latex = rf"i_p=m\nu^{{{scan_dependence:g}}}+b"
        formula_latex = (
            r"D=\frac{RT}{(Fn)^3}"
            r"\left(\frac{m}{0.4463SC}\right)^2"
        )
        resolved_latex = (
            rf"D=\frac{{({R:.6g})({T:g})}}{{(({F:.6g})({n:g}))^3}}"
            rf"\left(\frac{{m}}{{0.4463({S:g})({0 if C is None else C:g})}}\right)^2"
        )
        compact_latex = (
            rf"D={R * T / (F * n) ** 3:.6g}"
            rf"\left(\frac{{m}}{{{0.4463 * S * (0 if C is None else C):.6g}}}\right)^2"
        )
        symbolic_text = (
            f"i_p = m * v^{scan_dependence:g} + b; "
            "D = (R * T / (F * n)^3) * (m / (0.4463 * S * C))^2"
        )
        resolved_text = (
            f"D = ({R:.6g} * {T:g} / ({F:.6g} * {n:g})^3) "
            f"* (m / (0.4463 * {S:g} * {_equation_value_text(C)}))^2"
        )
        compact_text = (
            f"D = {R * T / (F * n) ** 3:.6g} "
            f"* (m / ({0.4463 * S * (0 if C is None else C):.6g}))^2"
        )
        variable_text = f"C = {_equation_value_text(C, 'mol/cm^3')}"
        variable_latex = rf"C={0 if C is None else C:g}\ \mathrm{{mol/cm^3}}"
    else:
        fit_latex = r"i_p=mC+b"
        formula_latex = (
            r"D=\frac{RT}{F^2n^3\nu S^2}"
            r"\left(\frac{m}{0.4463}\right)^2"
        )
        resolved_latex = (
            rf"D=\frac{{({R:.6g})({T:g})}}"
            rf"{{({F:.6g})^2({n:g})^3({0 if v is None else v:g})({S:g})^2}}"
            r"\left(\frac{m}{0.4463}\right)^2"
        )
        denominator = F ** 2 * n ** 3 * (0 if v is None else v) * S ** 2
        compact_latex = (
            rf"D={R * T / denominator if denominator != 0 else 0:.6g}"
            r"\left(\frac{m}{0.4463}\right)^2"
        )
        symbolic_text = (
            "i_p = m * C + b; "
            "D = (R * T / (F^2 * n^3 * v * S^2)) * (m / 0.4463)^2"
        )
        resolved_text = (
            f"D = ({R:.6g} * {T:g} / ({F:.6g}^2 * {n:g}^3 "
            f"* {_equation_value_text(v)} * {S:g}^2)) * (m / 0.4463)^2"
        )
        compact_text = (
            f"D = {R * T / denominator if denominator != 0 else 0:.6g} "
            "* (m / 0.4463)^2"
        )
        variable_text = f"v = {_equation_value_text(v, 'V/s')}"
        variable_latex = rf"\nu={0 if v is None else v:g}\ \mathrm{{V/s}}"

    definitions_latex = (
        rf"{fit_latex},\quad "
        rf"R={R:.6g},\quad F={F:.6g},\quad "
        rf"T={T:g}\ \mathrm{{K}},\quad n={n:g},\quad "
        rf"S={S:g}\ \mathrm{{cm^2}},\quad {variable_latex}"
    )
    definitions_text = (
        f"m = converted fit slope, R = {R:.6g}, F = {F:.6g}, "
        f"T = {T:g} K, n = {n:g}, S = {S:g} cm^2, {variable_text}"
    )

    return {
        "symbolic latex": rf"{fit_latex},\quad {formula_latex}",
        "resolved latex": resolved_latex,
        "compact latex": compact_latex,
        "definitions latex": definitions_latex,
        "symbolic": symbolic_text,
        "resolved": resolved_text,
        "compact": compact_text,
        "definitions": definitions_text,
    }


def _display_sevcik_diffusion_equation(
    mode,
    num_electrons,
    temperature,
    electrode_area,
    concentration=None,
    scan_rate=None,
    scan_dependence=0.5,
    resolved=True,
    compact=False,
    include_definitions=True,
):
    equation = _format_sevcik_diffusion_equation(
        mode=mode,
        num_electrons=num_electrons,
        temperature=temperature,
        electrode_area=electrode_area,
        concentration=concentration,
        scan_rate=scan_rate,
        scan_dependence=scan_dependence,
    )
    return _display_analysis_equation(
        r"\text{Sevcik diffusion coefficient equation:}",
        "Sevcik diffusion coefficient equation",
        equation,
        resolved=resolved,
        compact=compact,
        include_definitions=include_definitions,
    )


def _sevcik_parameter_label(key, html=False):
    labels = {
        "n": "n",
        "T": "T",
        "S": "S",
        "C": "C",
        "v": "v",
        "scan dependence": "scan dependence",
    }
    html_labels = {
        "n": "<i>n</i>",
        "T": "<i>T</i>",
        "S": "<i>S</i>",
        "C": "<i>C</i><sup>*</sup>",
        "v": "<i>&nu;</i>",
        "scan dependence": "Scan dependence",
    }
    if html:
        return html_labels.get(key, labels.get(key, key))
    return labels.get(key, key)


def _format_sevcik_value(value, sig_figs=4, unit="", scientific=False):
    if value is None:
        return ""
    if isinstance(value, (int, float, np.integer, np.floating)):
        if not np.isfinite(float(value)):
            return str(value)
        if scientific:
            digits = max(int(sig_figs) - 1, 0)
            text = f"{float(value):.{digits}e}"
        else:
            text = f"{round_sigfigs(float(value), sig_figs):g}"
    else:
        text = str(value)
    return f"{text} {unit}".strip()


def _sevcik_parameter_table(
    mode,
    num_electrons,
    temperature,
    electrode_area,
    concentration=None,
    scan_rate=None,
    scan_dependence=0.5,
    options=None,
):
    options = options or {}
    keys = ["n", "T", "S"]
    values = {
        "n": num_electrons,
        "T": temperature,
        "S": electrode_area,
    }
    if "scan" in str(mode).strip().lower():
        keys.extend(["C", "scan dependence"])
        values["C"] = concentration
        values["scan dependence"] = scan_dependence
    else:
        keys.append("v")
        values["v"] = scan_rate

    sig_figs = options.get("sig figs", 4)
    units = {
        "T": "K",
        "S": "cm^2",
        "C": "mol/cm^3",
        "v": "V/s",
    }
    table = pd.DataFrame(
        [
            {
                "Parameter": _sevcik_parameter_label(key),
                "Value": _format_sevcik_value(values.get(key), sig_figs=sig_figs, unit=units.get(key, "")),
            }
            for key in keys
        ]
    )
    table.attrs["parameter_keys"] = keys
    return table


def _display_sevcik_parameter_table(table, options):
    if not options.get("pretty print", True) or display is None:
        print(table.to_string(index=False, justify="left"))
        return table

    display_table = table.copy()
    keys = table.attrs.get("parameter_keys", list(display_table.index))
    display_table["Parameter"] = [
        _sevcik_parameter_label(key, html=True)
        for key in keys
    ]
    styled = (
        display_table.style
        .format(escape=None)
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
    display(styled)
    return table


def _sevcik_fit_results_table(fit_table, diffusion_coefficients, options=None):
    if fit_table is None or len(fit_table) == 0:
        return pd.DataFrame()
    options = options or {}
    sig_figs = options.get("sig figs", 4)
    table = fit_table.copy()
    for column in ("slope", "intercept"):
        if column in table.columns:
            table = table.drop(columns=column)
    diffusion_values = list(diffusion_coefficients)
    if len(diffusion_values) < len(table):
        diffusion_values.extend([None] * (len(table) - len(diffusion_values)))
    elif len(diffusion_values) > len(table):
        diffusion_values = diffusion_values[:len(table)]
    diffusion_text = [
        _format_sevcik_value(value, sig_figs=sig_figs, unit="cm^2/s", scientific=True)
        if value is not None and not (isinstance(value, (float, np.floating)) and np.isnan(value))
        else ""
        for value in diffusion_values
    ]
    table["Diffusion Coefficient"] = diffusion_text
    return table


def _print_sevcik_fit_results(fit_table):
    if fit_table is None or len(fit_table) == 0:
        return
    print("Sevcik Fit Results:")
    if display is not None:
        with pd.option_context("display.max_columns", None):
            display(fit_table)
    else:
        print(fit_table.to_string(index=False))


def _format_fowa_kobs_equation(options):
    """
    Return display-ready text/LaTeX for the EC'-type FOWA kobs equation.

    Equation used:
        kobs = (m * 0.4463 * n_ref / n_cat^sigma)^2
               * (n_ref * F * v) / (R * T)
    """
    n_ref = float(options.get("catalyst electrons", options.get("num electrons", 1)))
    n_cat = float(options.get("turnover electrons", 1))
    sigma = float(options.get("sigma", 1.0))
    turnover_factor = n_cat ** sigma

    symbolic_latex = (
        r"k_{\mathrm{obs}}"
        r"=\left("
        r"\frac{m\,0.4463\,n_{\mathrm{ref}}}"
        r"{n_{\mathrm{cat}}^{\sigma}}"
        r"\right)^2"
        r"\frac{n_{\mathrm{ref}}F\nu}{RT}"
    )

    resolved_latex = (
        r"k_{\mathrm{obs}}"
        r"=\left("
        rf"\frac{{m\,0.4463\,({n_ref:g})}}"
        rf"{{({n_cat:g})^{{{sigma:g}}}}}"
        r"\right)^2"
        rf"\frac{{({n_ref:g})F\nu}}{{RT}}"
    )

    compact_latex = (
        r"k_{\mathrm{obs}}"
        r"=\left("
        rf"{0.4463 * n_ref / turnover_factor:.6g}\,m"
        r"\right)^2"
        rf"\frac{{({n_ref:g})F\nu}}{{RT}}"
    )

    definitions_latex = (
        rf"n_{{\mathrm{{ref}}}}={n_ref:g},\quad "
        rf"n_{{\mathrm{{cat}}}}={n_cat:g},\quad "
        rf"\sigma={sigma:g},\quad "
        rf"n_{{\mathrm{{cat}}}}^{{\sigma}}={turnover_factor:.6g}"
    )

    symbolic_text = (
        "k_obs = (m * 0.4463 * n_ref / n_cat^sigma)^2 "
        "* (n_ref * F * v) / (R * T)"
    )

    resolved_text = (
        f"k_obs = (m * 0.4463 * {n_ref:g} / "
        f"{n_cat:g}^{sigma:g})^2 * "
        f"({n_ref:g} * F * v) / (R * T)"
    )

    compact_text = (
        f"k_obs = ({0.4463 * n_ref / turnover_factor:.6g} * m)^2 "
        f"* ({n_ref:g} * F * v) / (R * T)"
    )

    definitions_text = (
        f"n_ref = {n_ref:g} ('catalyst electrons'), "
        f"n_cat = {n_cat:g} ('turnover electrons'), "
        f"sigma = {sigma:g}, "
        f"n_cat^sigma = {turnover_factor:.6g}"
    )

    return {
        "symbolic latex": symbolic_latex,
        "resolved latex": resolved_latex,
        "compact latex": compact_latex,
        "definitions latex": definitions_latex,
        "symbolic": symbolic_text,
        "resolved": resolved_text,
        "compact": compact_text,
        "definitions": definitions_text,
    }

def _display_fowa_kobs_equation(options, resolved=True, compact=False):
    """
    Display the EC'-type FOWA kobs equation in a notebook when possible.
    Falls back to plain text in non-notebook contexts.
    """
    eq = _format_fowa_kobs_equation(options)
    return _display_analysis_equation(
        r"\text{FOWA } k_{\mathrm{obs}}\text{ equation:}",
        "FOWA kobs equation",
        eq,
        resolved=resolved,
        compact=compact,
    )


def _coerce_plateau_cv_list(cvs, allow_empty=False):
    if cvs is None:
        cv_list = []
    elif isinstance(cvs, (list, tuple)):
        cv_list = list(cvs)
    else:
        cv_list = [cvs]
    if not cv_list and not allow_empty:
        raise ValueError("plateau_current requires at least one catalytic CV or manual ilim/ic.")
    for item in cv_list:
        if not hasattr(item, "peak_current"):
            raise TypeError("plateau_current expects CV-like objects with peak_current().")
    return cv_list


def _resolve_manual_ilim(options):
    value = options.get("ilim")
    if value is None:
        value = options.get("ic")
    if value is None:
        return None
    if isinstance(value, (list, tuple, np.ndarray, pd.Series)):
        values = [float(item) for item in value]
        return float(np.mean(values)) if values else None
    return float(value)


def _resolve_temperature(cvs, ref_cvs, options):
    for item in list(cvs or []) + list(ref_cvs or []):
        temperature = getattr(item, "temperature", None)
        if temperature is not None and np.isfinite(float(temperature)) and float(temperature) > 0:
            return float(temperature)
    temperature = options.get("temperature", 298)
    if temperature is None or not np.isfinite(float(temperature)) or float(temperature) <= 0:
        raise ValueError("plateau_current requires a positive temperature in K.")
    return float(temperature)


def _resolve_plateau_area(cv_obj, options):
    area = options.get("electrode area")
    explicit_area = area is not None
    if area is None and cv_obj is not None:
        area = getattr(cv_obj, "electrode_area", None)
    if area is None:
        return None
    area = float(area)
    if not np.isfinite(area) or area <= 0:
        if not explicit_area:
            return None
        raise ValueError("'electrode area' must be positive.")
    return area


def _resolve_concentration_cm3(cv_obj, options):
    value = options.get("C")
    if value is not None:
        if isinstance(value, str):
            return concentration_to_float(value) / 1000
        numeric = float(value)
        unit = options.get("C unit")
        if unit is None:
            warnings.warn(
                "Numeric 'C' was assumed to be in M and converted to mol/cm^3. "
                "Pass 'C unit' = 'mol/cm^3' to use numeric concentration directly.",
                UserWarning,
                stacklevel=2,
            )
            return numeric / 1000
        unit_key = str(unit).strip().replace("μ", "u").lower()
        if unit_key in {"m", "mol/l", "mol/liter"}:
            return numeric / 1000
        if unit_key in {"mol/cm3", "mol/cm^3"}:
            return numeric
        raise ValueError("'C unit' must be 'M', 'mol/L', 'mol/cm3', or 'mol/cm^3'.")

    species = options.get("species")
    normalize_params = options.get("normalize params", {}) or {}
    if species is None:
        species = normalize_params.get("species")
    concentrations = list(getattr(cv_obj, "concentrations", []) or []) if cv_obj is not None else []
    compounds = list(getattr(cv_obj, "compounds", []) or []) if cv_obj is not None else []
    if species is not None and concentrations:
        if compounds and len(compounds) == len(concentrations):
            for compound, concentration in zip(compounds, concentrations):
                if str(species).lower() in str(compound).lower():
                    return concentration_to_float(str(concentration)) / 1000
        return concentration_to_float(str(concentrations[-1])) / 1000
    raise ValueError("Direct plateau-current mode requires 'C' or a concentration resolvable from 'species'.")


def _extract_current_with_peak_current(cv_obj, options, role, fallback_potential=None):
    guess = options.get("guess potential")
    if role == "non-catalytic" and options.get("non-catalytic guess potential") is not None:
        guess = options.get("non-catalytic guess potential")

    exact = options.get("exact potential")
    exact_potential = None
    if exact is not None:
        exact_potential = exact
    elif fallback_potential is not None:
        exact_potential = fallback_potential

    internal = {
        "plot": bool(options.get("plot all", False)),
        "plot all": bool(options.get("plot all", False)),
        "print": bool(options.get("print all", False)),
        "print all": bool(options.get("print all", False)),
        "internal call": True,
        "new plot": False,
        "segment": options.get("segment"),
        "segments": options.get("segments"),
        "noise window": options.get("noise window", "auto"),
        "noise polyorder": options.get("noise polyorder", "auto"),
        "sig figs": options.get("sig figs", 4),
        "peak prominence": options.get("peak prominence"),
        "guess potential": guess,
        "exact potential": exact_potential,
        "peak fallback": options.get("peak fallback", "highest current"),
        "tangent range": options.get("tangent range", "auto"),
        "tangent min points": options.get("tangent min points"),
        "tangent potential": options.get("tangent potential"),
        "percent threshold": options.get("percent threshold"),
    }
    internal = {key: value for key, value in internal.items() if value is not None}

    try:
        current_result = cv_obj.peak_current(internal)
        current = current_result["ip"]
        tanline = current_result["tangent line"]
        return {
            "current": float(current),
            "potential": current_result.get("Ep", exact_potential),
            "source": current_result.get("peak source", "peak_current"),
            "tanline": tanline,
            "cv": getattr(cv_obj, "name", "CV"),
            "scan rate": float(getattr(cv_obj, "scan_rate", np.nan)),
        }
    except Exception as exc:
        raise ValueError(
            f"Could not extract {role} plateau current from '{getattr(cv_obj, 'name', 'CV')}'. "
            "Adjust 'peak fallback', provide 'exact potential' or 'guess potential', "
            "or verify that the selected segment contains usable current data."
        ) from exc


def _extract_catalytic_currents(cat_cvs, options):
    rows = []
    for cv_obj in cat_cvs:
        extracted = _extract_current_with_peak_current(cv_obj, options, role="catalytic")
        scan_rate = extracted["scan rate"]
        rows.append({
            "cv": extracted["cv"],
            "scan rate": scan_rate,
            "sqrt scan rate": np.sqrt(scan_rate) if np.isfinite(scan_rate) and scan_rate >= 0 else np.nan,
            "ic": extracted["current"],
            "abs ic": abs(extracted["current"]),
            "current source": extracted["source"],
            "extraction potential": extracted["potential"],
            "valid extraction": True,
        })
    return pd.DataFrame(rows)


def _extract_ip0_currents(ref_cvs, options):
    rows = []
    successes = []
    for ref_cv in ref_cvs:
        scan_rate = float(getattr(ref_cv, "scan_rate", np.nan))
        fallback = None
        if successes and np.isfinite(scan_rate):
            fallback = min(successes, key=lambda item: abs(item["scan rate"] - scan_rate)).get("potential")
        extracted = _extract_current_with_peak_current(
            ref_cv,
            options,
            role="non-catalytic",
            fallback_potential=fallback,
        )
        successes.append(extracted)
        rows.append({
            "reference cv": extracted["cv"],
            "scan rate": extracted["scan rate"],
            "sqrt scan rate": np.sqrt(extracted["scan rate"]),
            "ip0": extracted["current"],
            "abs ip0": abs(extracted["current"]),
            "current source": extracted["source"],
            "extraction potential": extracted["potential"],
            "valid extraction": True,
        })
    return pd.DataFrame(rows)


def _plateau_fit_forced_origin(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if len(x) < 1:
        raise ValueError("Forced-origin fit requires at least one finite point.")
    denom = float(np.sum(x ** 2))
    if denom == 0:
        raise ValueError("Forced-origin fit x values cannot all be zero.")
    slope = float(np.sum(x * y) / denom)
    predicted = slope * x
    residuals = y - predicted
    ss_res = float(np.sum(residuals ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = np.nan if ss_tot == 0 else 1 - ss_res / ss_tot
    return slope, predicted, r2, residuals


def _plateau_linear_fit_metric(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) == 1:
        return 0.0, float(y[0]), 0.0
    slope, intercept = np.polyfit(x, y, 1)
    mean_y = float(np.mean(np.abs(y)))
    metric = np.inf if mean_y == 0 else abs(float(slope)) * (float(np.max(x)) - float(np.min(x))) / mean_y
    return float(slope), float(intercept), float(metric)


def _select_plateau_subset(ic_df, options):
    if ic_df.empty:
        raise ValueError("No catalytic plateau currents are available.")
    df = ic_df.sort_values("scan rate").reset_index(drop=True)
    validate = bool(options.get("validate plateau", True))
    require = bool(options.get("require plateau", True))
    min_cvs = int(options.get("plateau min cvs", 2))
    tolerance = float(options.get("plateau slope tolerance", 0.10))
    method = str(options.get("plateau average method", "mean")).strip().lower()
    warnings_list = []

    if len(df) == 1:
        if validate:
            msg = "Only one catalytic CV was supplied; scan-rate independence cannot be tested."
            warnings.warn(msg, UserWarning, stacklevel=2)
            warnings_list.append("scan-rate independence cannot be tested")
        return {
            "ilim": float(df["ic"].iloc[0]),
            "accepted indices": [0],
            "accepted cvs": [df["cv"].iloc[0]],
            "accepted scan rates": [float(df["scan rate"].iloc[0])],
            "slope": 0.0,
            "intercept": float(df["abs ic"].iloc[0]),
            "slope metric": 0.0,
            "valid plateau": True,
            "warnings": warnings_list,
        }

    x = df["sqrt scan rate"].astype(float).to_numpy()
    y_abs = df["abs ic"].astype(float).to_numpy()
    all_slope, all_intercept, all_metric = _plateau_linear_fit_metric(x, y_abs)
    selected = None
    if not validate or all_metric <= tolerance:
        selected = (list(range(len(df))), all_slope, all_intercept, all_metric, True)
    else:
        candidates = []
        for start in range(0, len(df) - min_cvs + 1):
            indices = list(range(start, len(df)))
            slope, intercept, metric = _plateau_linear_fit_metric(x[indices], y_abs[indices])
            if metric <= tolerance:
                candidates.append((indices, slope, intercept, metric, True))
        if candidates:
            candidates.sort(key=lambda item: (-len(item[0]), item[3]))
            selected = candidates[0]

    if selected is None:
        msg = "No scan-rate-independent plateau current was detected. Use FOWA or provide manual ilim."
        if require:
            raise ValueError(msg)
        warnings.warn(msg, UserWarning, stacklevel=2)
        warnings_list.append(msg)
        selected = (list(range(len(df))), all_slope, all_intercept, all_metric, False)

    indices, slope, intercept, metric, valid = selected
    signed = df.loc[indices, "ic"].astype(float)
    ilim = float(signed.median() if method == "median" else signed.mean())
    return {
        "ilim": ilim,
        "accepted indices": [int(i) for i in indices],
        "accepted cvs": df.loc[indices, "cv"].tolist(),
        "accepted scan rates": [float(v) for v in df.loc[indices, "scan rate"].tolist()],
        "slope": slope,
        "intercept": intercept,
        "slope metric": metric,
        "valid plateau": bool(valid),
        "warnings": warnings_list,
    }


def _calculate_plateau_kobs(
    ilim,
    options,
    temperature,
    formula_mode="auto",
    ip0=None,
    ip0_scan_rate=None,
    ip0_sqrt_scan_rate_slope=None,
    D=None,
    C=None,
    electrode_area=None,
):
    n = float(options.get("catalyst electrons", 1))
    n_prime = float(options.get("turnover electrons", 1))
    if n <= 0 or n_prime <= 0:
        raise ValueError("'catalyst electrons' and 'turnover electrons' must be positive.")
    mode = str(formula_mode or "auto").strip().lower().replace("-", " ")
    direct_ready = D is not None and C is not None and electrode_area is not None
    slope_ready = ip0_sqrt_scan_rate_slope is not None
    normalized_ready = ip0 is not None and ip0_scan_rate is not None
    if mode == "auto":
        if direct_ready:
            mode = "direct"
        elif slope_ready:
            mode = "slope normalized"
        elif normalized_ready:
            mode = "normalized"
        else:
            raise ValueError(
                "plateau_current could not resolve formula mode. Provide D/C/electrode area, "
                "ip0 sqrt scan rate slope, or ip0 with ip0 scan rate."
            )
    if mode == "direct":
        if not direct_ready:
            raise ValueError("Direct plateau-current mode requires D, C, and electrode area.")
        kobs = (abs(ilim) / (n * F * float(electrode_area) * float(C))) ** 2 / (float(D) * n_prime)
        return "direct", "kobs = (|ilim| / (n F A C_cat))^2 / (D n')", float(kobs)
    if mode == "slope normalized":
        if not slope_ready:
            raise ValueError("Slope-normalized plateau-current mode requires 'ip0 sqrt scan rate slope'.")
        kobs = (
            0.446 * abs(ilim) / abs(float(ip0_sqrt_scan_rate_slope))
            * np.sqrt(n * F / (R * temperature))
        ) ** 2 / n_prime
        return "slope normalized", "kobs = (0.446 |ilim|/|s_ip0| sqrt(nF/RT))^2 / n'", float(kobs)
    if mode == "normalized":
        if not normalized_ready:
            raise ValueError("Normalized plateau-current mode requires ip0 and ip0 scan rate.")
        kobs = (
            0.446 * abs(ilim / float(ip0))
            * np.sqrt(n * F * float(ip0_scan_rate) / (R * temperature))
        ) ** 2 / n_prime
        return "normalized", "kobs = (0.446 |ilim/ip0| sqrt(nFv_ip0/RT))^2 / n'", float(kobs)
    raise ValueError("'formula mode' must be 'auto', 'normalized', 'slope normalized', or 'direct'.")


def _plot_plateau_validation(ic_df, selection, options):
    if ic_df is None or ic_df.empty or len(ic_df) < 2:
        return
    plt.figure()
    x = ic_df["sqrt scan rate"].astype(float).to_numpy()
    y = ic_df["abs ic"].astype(float).to_numpy()
    plt.scatter(x, y, color="0.55", label="all")
    accepted = selection.get("accepted indices", [])
    if accepted:
        plt.scatter(x[accepted], y[accepted], color=options.get("color", "black"), label="accepted")
    slope, intercept, _metric = _plateau_linear_fit_metric(x, y)
    line_x = np.linspace(np.min(x), np.max(x), 100)
    plt.plot(line_x, slope * line_x + intercept, color="tab:red", linestyle="--", label="all fit")
    if len(accepted) >= 2:
        sx = x[accepted]
        sy = y[accepted]
        sslope, sintercept, _ = _plateau_linear_fit_metric(sx, sy)
        plt.plot(line_x, sslope * line_x + sintercept, color="black", linestyle="-", label="accepted fit")
    plt.xlabel("sqrt(scan rate) / (V/s)^1/2")
    plt.ylabel("|i_c| / A")
    plt.title("Plateau-current validation")
    plt.legend()


def _plot_ip0_sqrt_fit(ip0_df, slope, options):
    if ip0_df is None or ip0_df.empty or len(ip0_df) < 2:
        return
    plt.figure()
    x = ip0_df["sqrt scan rate"].astype(float).to_numpy()
    y = ip0_df["abs ip0"].astype(float).to_numpy()
    plt.scatter(x, y, color=options.get("color", "black"))
    line_x = np.linspace(0, np.max(x), 100)
    plt.plot(line_x, abs(slope) * line_x, color="tab:red", linestyle="--")
    plt.xlabel("sqrt(scan rate) / (V/s)^1/2")
    plt.ylabel("|i_p0| / A")
    plt.title("ip0 vs sqrt(scan rate)")


def plateau_current(cvs, options={}):
    """Analyze catalytic plateau current from one CV or a CV group.
    
    Parameters
    ----------
    cvs : cv or sequence of cv
        Catalytic CV object or objects used for plateau analysis.
    options : dict or PlateauCurrentOptions, optional
        Plateau, normalization, Sevcik, print, and plot options. See ``e.describe_options("plateau_current")``.
    
    Returns
    -------
    pandas.DataFrame
        Plateau-current summary table.
    
    Examples
    --------
    >>> result = e.plateau_current(cvs, {"non-catalytic cvs": blanks, "plot all": True})
    """
    typed_options = PlateauCurrentOptions.from_options(options)
    options = typed_options.to_legacy_dict()
    manual_ilim = _resolve_manual_ilim(options)
    cat_cvs = _coerce_plateau_cv_list(cvs, allow_empty=manual_ilim is not None)

    shared_ref = options.get("non-catalytic cv")
    ref_list = options.get("non-catalytic cvs")
    if shared_ref is not None and ref_list is not None:
        raise ValueError("Use either 'non-catalytic cv' or 'non-catalytic cvs', not both.")
    if ref_list is not None and not isinstance(ref_list, (list, tuple)):
        ref_cvs = [ref_list]
    elif ref_list is not None:
        ref_cvs = list(ref_list)
    elif shared_ref is not None:
        ref_cvs = [shared_ref]
    else:
        ref_cvs = []

    ip0 = options.get("ip0")
    if ip0 is None:
        ip0 = options.get("non-catalytic current")
    ip0 = float(ip0) if ip0 is not None else None
    ip0_scan_rate = options.get("ip0 scan rate")
    if ip0_scan_rate is None:
        ip0_scan_rate = options.get("scan rate")
    ip0_scan_rate = float(ip0_scan_rate) if ip0_scan_rate is not None else None
    ip0_slope = options.get("ip0 sqrt scan rate slope")
    ip0_slope = float(ip0_slope) if ip0_slope is not None else None
    ip0_source = "manual" if ip0 is not None else None
    ip0_fit_r2 = np.nan
    ip0_df = pd.DataFrame()

    if ip0 is None and ip0_slope is None and ref_cvs:
        if options.get("plot all", False):
            try:
                ref_plot_options = _multiplot_options_from_mapping(options)
                ref_plot_options.update({"plot": True, "print": False, "legend": True})
                multiplot(ref_cvs, ref_plot_options)
            except Exception:
                pass
        ip0_df = _extract_ip0_currents(ref_cvs, options)
        if len(ip0_df) > 1:
            x = ip0_df["sqrt scan rate"].astype(float).to_numpy()
            y = ip0_df["abs ip0"].astype(float).to_numpy()
            ip0_slope, _pred, ip0_fit_r2, _resid = _plateau_fit_forced_origin(x, y)
            ip0_source = "non-catalytic cvs"
            if options.get("plot all", False):
                _plot_ip0_sqrt_fit(ip0_df, ip0_slope, options)
        else:
            ip0 = float(ip0_df["ip0"].iloc[0])
            ip0_scan_rate = float(ip0_df["scan rate"].iloc[0])
            ip0_source = "non-catalytic cv"

    if options.get("plot all", False) and cat_cvs:
        try:
            cat_plot_options = _multiplot_options_from_mapping(options)
            cat_plot_options.update({"plot": True, "print": False, "legend": True})
            multiplot(cat_cvs, cat_plot_options)
        except Exception:
            pass

    ic_df = pd.DataFrame()
    if manual_ilim is None:
        ic_df = _extract_catalytic_currents(cat_cvs, options)
        selection = _select_plateau_subset(ic_df, options)
        ilim = selection["ilim"]
        ilim_source = "peak_current"
    else:
        ilim = manual_ilim
        selection = {
            "accepted cvs": [getattr(item, "name", "CV") for item in cat_cvs],
            "accepted scan rates": [float(getattr(item, "scan_rate", np.nan)) for item in cat_cvs],
            "slope": np.nan,
            "intercept": np.nan,
            "slope metric": np.nan,
            "valid plateau": True,
            "warnings": [],
        }
        ilim_source = "manual"

    temperature = _resolve_temperature(cat_cvs, ref_cvs, options)
    D = options.get("D")
    D = float(D) if D is not None else None
    C = None
    if options.get("C") is not None or options.get("species") is not None:
        C = _resolve_concentration_cm3(cat_cvs[0] if cat_cvs else None, options)
    electrode_area = _resolve_plateau_area(cat_cvs[0] if cat_cvs else None, options)

    mode, formula, kobs = _calculate_plateau_kobs(
        ilim=ilim,
        options=options,
        temperature=temperature,
        formula_mode=options.get("formula mode", "auto"),
        ip0=ip0,
        ip0_scan_rate=ip0_scan_rate,
        ip0_sqrt_scan_rate_slope=ip0_slope,
        D=D,
        C=C,
        electrode_area=electrode_area,
    )

    row = {
        "cv / cvs used": ", ".join([getattr(item, "name", "CV") for item in cat_cvs]),
        "ilim": ilim,
        "ilim source": ilim_source,
        "ilim average method": options.get("plateau average method"),
        "valid plateau": selection["valid plateau"],
        "plateau warning": " | ".join(selection.get("warnings", [])),
        "catalytic scan rates": ic_df["scan rate"].tolist() if not ic_df.empty else [float(getattr(item, "scan_rate", np.nan)) for item in cat_cvs],
        "catalytic sqrt scan rates": ic_df["sqrt scan rate"].tolist() if not ic_df.empty else [],
        "plateau subset cvs": selection.get("accepted cvs", []),
        "plateau subset scan rates": selection.get("accepted scan rates", []),
        "plateau slope": selection.get("slope"),
        "plateau intercept": selection.get("intercept"),
        "plateau slope metric": selection.get("slope metric"),
        "ip0": ip0,
        "ip0 source": ip0_source,
        "ip0 scan rate": ip0_scan_rate,
        "ip0 sqrt scan rate slope": ip0_slope,
        "ip0 fit r2": ip0_fit_r2,
        "formula mode": mode,
        "formula": formula,
        "D": D,
        "C": C,
        "electrode area": electrode_area,
        "catalyst electrons": float(options.get("catalyst electrons", 1)),
        "turnover electrons": float(options.get("turnover electrons", 1)),
        "kobs": kobs,
    }
    display_df = pd.DataFrame([row])
    display_df.attrs["catalytic currents"] = ic_df
    display_df.attrs["ip0 currents"] = ip0_df

    if options.get("print", True):
        print("### Plateau Current Summary ###")
        print(f"formula mode: {mode}")
        print(f"valid plateau: {selection['valid plateau']}")
        print(f"ilim: {ilim:.6g} A ({ilim_source})")
        if ip0_slope is not None:
            print(f"ip0 sqrt scan rate slope: {ip0_slope:.6g}")
        elif ip0 is not None:
            print(f"ip0: {ip0:.6g} A")
        display_object_table(display_df)

    if options.get("plot all", False) or (options.get("plot", True) and len(ic_df) > 1):
        _plot_plateau_validation(ic_df, selection, options)
    if not options.get("plot all", False) and (options.get("plot", True) and len(ip0_df) > 1):
        _plot_ip0_sqrt_fit(ip0_df, ip0_slope, options)

    return display_df


def _resolve_fowa_formula(cv_obj, slope, options):
    """
    Resolve the FOWA slope into kinetic quantities.

    Default EC'-type expression:

        x_FOWA = [1 + exp(n_ref*F/(RT) * (E - E_ref))]^-1

        i/ip0 = slope*x_FOWA + intercept

        slope = n_cat^sigma / (0.4463*n_ref)
                * sqrt(RT*kobs / (n_ref*F*v))

    Therefore:

        kobs = (slope * 0.4463*n_ref / n_cat^sigma)^2
               * (n_ref*F*v)/(RT)

    where:
        n_ref = catalyst electrons
            Electron count for the non-catalytic catalyst wave used for ip0
            and for the FOWA x-axis exponent.

        n_cat = turnover electrons
            Total electrons required per catalytic turnover.

        sigma
            Stoichiometric ET-pathway exponent.
            sigma = 1.0 when all turnover electrons occur at the electrode.
            sigma approaches 0.5 when homogeneous ET contributes.
    """
    custom_formula = options.get("custom formula")
    if callable(custom_formula):
        result = custom_formula(
            slope=slope,
            cv=cv_obj,
            options=options,
            F=F,
            R=R,
        )
        if not isinstance(result, dict):
            raise TypeError("'custom formula' must return a dict.")
        label = options.get("formula label") or getattr(custom_formula, "__name__", "custom")
        return label, result

    mechanism = str(options.get("mechanism", "EC'")).strip().lower()
    scan_rate = getattr(cv_obj, "scan_rate", None)
    if scan_rate is None:
        raise ValueError(
            "Scan rate is required for the default FOWA formula but was not found on the CV."
        )

    catalyst_electrons = float(
        options.get("catalyst electrons", options.get("num electrons", 1))
    )
    turnover_electrons = float(options.get("turnover electrons", 1))
    sigma = float(options.get("sigma", 1.0))

    if catalyst_electrons <= 0:
        raise ValueError("'catalyst electrons' must be positive.")
    if turnover_electrons <= 0:
        raise ValueError("'turnover electrons' must be positive.")
    if sigma <= 0:
        raise ValueError("'sigma' must be positive.")

    if mechanism in {"ecprime", "ec'", "ec'-like", "default"}:
        turnover_factor = turnover_electrons ** sigma

        k_obs = (
            slope
            * 0.4463
            * catalyst_electrons
            / turnover_factor
        ) ** 2 * (
            catalyst_electrons
            * F
            * scan_rate
            / (R * cv_obj.temperature)
        )

        return "EC' apparent", {
            "kobs": float(k_obs),
            "TOFmax": float(k_obs),
            "catalyst electrons": catalyst_electrons,
            "turnover electrons": turnover_electrons,
            "sigma": sigma,
            "turnover factor": turnover_factor,
        }

    raise ValueError(
        f"Mechanism '{options.get('mechanism')}' is not implemented. "
        "Set 'custom formula' for a custom mechanism."
    )

def _extract_line_colors_from_ax(ax, echem_list):
    """
    Extract line colors from an already-drawn multiplot axis.
    Assumes the last len(echem_list) Line2D objects correspond to the
    raw CV traces.
    """
    lines = list(ax.get_lines())[-len(echem_list):]
    return {
        id(obj): line.get_color()
        for obj, line in zip(echem_list, lines)
    }


def _copy_cv_with_fowa_current_axis(cv_obj, ip0, options):
    """
    Return a CV copy with a temporary i/ip0 column for FOWA plot-all multiplot.
    """
    return _copy_cv_with_normalized_current_axis(cv_obj, ip0, options)


def _plot_fowa_transformed(plot_data, results_df, options):
    echem_list = [pdata["cat cv"] for pdata in plot_data]
    plot_options = _multiplot_options_from_mapping(options)
    plot_options["labels"] = [
        pdata.get("plot label", pdata.get("name", f"Trace {i + 1}"))
        for i, pdata in enumerate(plot_data)
    ]
    style = _prepare_multiplot_style(echem_list, plot_options)
    ax = style["ax"]
    color_spec = style["color spec"]

    for i, pdata in enumerate(plot_data):
        color = color_spec["line colors"][i]

        ax.plot(
            pdata["x fowa"],
            pdata["y fowa"],
            color=color,
            label=format_chemical_formulas(color_spec["plot labels"][i]),
        )

        ax.scatter(
            pdata["x fit"],
            pdata["y fit"],
            color=color,
            s=10,
            zorder=3,
            label="_nolegend_",
        )

        slope = (
            results_df.loc[i, "Slope"]
            if "Slope" in results_df.columns
            else results_df.loc[i, "slope"]
        )
        intercept = (
            results_df.loc[i, "Intercept"]
            if "Intercept" in results_df.columns
            else results_df.loc[i, "intercept"]
        )

        if slope > 0:
            y_end = float(np.nanmax(pdata["y fowa"]))
            x_end = (y_end - intercept) / slope
            x_end = max(0.0, min(1.0, x_end))
        elif slope < 0:
            y_end = float(np.nanmin(pdata["y fowa"]))
            x_end = (y_end - intercept) / slope
            x_end = max(0.0, min(1.0, x_end))
        else:
            x_end = min(1.0, float(np.nanmax(pdata["x fit"])))

        x_fit_line = np.array([0.0, x_end])
        y_fit_line = slope * x_fit_line + intercept

        if options.get("plot fit", True):
            fit_color = _fit_color_from_options(options, index=i, fallback=color)
            ax.plot(
                x_fit_line,
                y_fit_line,
                linestyle=options.get("fit linestyle", "--"),
                color=fit_color,
                linewidth=options.get("fit linewidth", 1.5),
                alpha=options.get("fit alpha", 1),
                label="_nolegend_",
            )

    _finish_multiplot_style(echem_list, plot_options, style)

    ax.set_xlim(0, 1)
    ax.set_ylabel("$i / i_p^0$")
    ax.set_xlabel(r"$[1+\exp(\frac{F}{RT}(E-E^0))]^{-1}$")

def _split_shared_columns(df, keep_in_table=None):
    if keep_in_table is None:
        keep_in_table = set()

    shared = {}
    table_df = df.copy()

    def is_blank(v):
        if v is None:
            return True
        if isinstance(v, float) and pd.isna(v):
            return True
        if isinstance(v, str) and v.strip() == "":
            return True
        return False

    for col in list(table_df.columns):
        if col in keep_in_table:
            continue

        vals = table_df[col].tolist()
        nonblank = [v for v in vals if not is_blank(v)]
        if len(nonblank) == 0:
            table_df.drop(columns=[col], inplace=True)
            continue

        if len(nonblank) == len(vals) and all(v == nonblank[0] for v in nonblank):
            shared[col] = nonblank[0]
            table_df.drop(columns=[col], inplace=True)

    return table_df, shared


def _format_fowa_pair(first, second):
    def is_blank(value):
        if value is None:
            return True
        try:
            return bool(pd.isna(value))
        except (TypeError, ValueError):
            return False

    if is_blank(first) or is_blank(second):
        return ""
    return f"[{first:.6g}, {second:.6g}]"


def _format_fowa_line(slope, intercept):
    def is_blank(value):
        if value is None:
            return True
        try:
            return bool(pd.isna(value))
        except (TypeError, ValueError):
            return False

    if is_blank(slope) or is_blank(intercept):
        return ""

    sign = "+" if float(intercept) >= 0 else "-"
    return f"y = {float(slope):.6g}x {sign} {abs(float(intercept)):.6g}"


def _fowa_units_map(df=None):
    units = {
        "Reference Ep": "V",
        "Redox Delta E": "V",
        "Redox Potential": "V",
        "Catalytic Ecat/2": "V",
        "Ecat/2 - E1/2": "V",
        "ip0": "A",
        "Background Slope": "A/V",
        "Background Intercept": "A",
        "Wave Start": "V",
        "Wave End": "V",
        "Slope": "dimensionless",
        "Intercept": "dimensionless",
        "R2": "dimensionless",
        "kobs": "s^-1",
        "TOFmax": "s^-1",
        "catalyst electrons": "dimensionless",
        "turnover electrons": "dimensionless",
        "sigma": "dimensionless",
        "turnover factor": "dimensionless",
    }
    if df is None:
        return units
    return {key: value for key, value in units.items() if key in df.columns}


def _order_fowa_display_columns(df, summary_columns):
    metadata_columns = [col for col in summary_columns if col in df.columns]
    operation_columns = [
        "Reference CV",
        "ip0 Source",
        "ip0",
        "Redox Mode",
        "Redox Source",
        "Reference Ep",
        "Redox Potential",
        "Redox Delta E",
        "Catalytic Ecat/2",
        "Ecat/2 - E1/2",
        "Background Correction",
        "Background Tangent",
        "Wave Range",
        "Fit Basis",
        "Fit Range",
        "Fit Points",
        "FOWA Fit",
        "R2",
        "Status",
        "Formula",
        "kobs",
    ]

    ordered = []
    for col in metadata_columns + operation_columns:
        if col in df.columns and col not in ordered:
            ordered.append(col)

    ordered.extend([col for col in df.columns if col not in ordered])
    return df.loc[:, ordered]


def _resolve_catalytic_half_peak_for_shift_check(cat_cv, ref_cv, options, internal_options):
    """
    Resolve catalytic Ecat/2 for the Ecat/2-shift diagnostic.

    This is intentionally softer than _resolve_fowa_redox_potential():
    it is only a diagnostic, so callers can catch exceptions and continue.

    Strategy
    --------
    1. Use any user-provided 'exact potential'.
    2. If missing and a non-catalytic reference CV is available, use the
       reference peak potential as the exact-potential anchor.
    3. If that fails, fall back to the catalytic peak potential.
    4. Then call cat_cv.half_peak_potential(...).
    """
    ecat_options = _analysis_options_for(PeakCurrentOptions, internal_options)
    ecat_options["plot"] = False
    ecat_options["plot all"] = False
    ecat_options["print"] = False
    ecat_options["print all"] = False
    ecat_options["internal call"] = True
    ecat_options["new plot"] = False

    if ecat_options.get("exact potential") is None:
        # Preferred fallback: use the non-catalytic peak as the anchor.
        if ref_cv is not None:
            try:
                ref_peak_options = _analysis_options_for(PeakPotentialOptions, ecat_options)
                ref_peak_options["plot"] = False
                ref_peak_options["print"] = False
                ref_peak_options["new plot"] = False
                ref_Ep = ref_cv.peak_potential(ref_peak_options)["Ep"]
                ecat_options["exact potential"] = float(ref_Ep)
            except Exception:
                pass

    if ecat_options.get("exact potential") is None:
        # Last-resort fallback: use catalytic peak if it exists.
        try:
            cat_peak_options = _analysis_options_for(PeakPotentialOptions, ecat_options)
            cat_peak_options["plot"] = False
            cat_peak_options["print"] = False
            cat_peak_options["new plot"] = False
            cat_Ep = cat_cv.peak_potential(cat_peak_options)["Ep"]
            ecat_options["exact potential"] = float(cat_Ep)
        except Exception as exc:
            raise ValueError(
                "Could not determine an anchor potential for catalytic Ecat/2. "
                "Try setting 'exact potential', 'guess potential', or disabling "
                "'ecat shift warning threshold'."
            ) from exc

    return cat_cv.half_peak_potential(ecat_options)


def _analysis_options_for(option_cls, source):
    routed = {}
    valid = {field.name for field in fields(option_cls)}
    plot_only = {"plot_segment", "plot_segments"}
    for key, value in (source or {}).items():
        norm = normalize_key(key)
        if norm in plot_only:
            continue
        if norm in valid:
            routed[norm] = value
    return option_cls.from_options(routed).to_legacy_dict()


def _resolve_fowa_redox_potential(cat_cv, ref_cv, options, internal_options, manual_redox=None):
    """
    Resolve the redox potential used in FOWA.

    Modes
    -----
    - manual: use a manually supplied redox potential
    - half wave: use ref_cv.half_wave_potential(...)
    - half peak: use cat_cv.half_peak_potential(...)
    """
    redox_mode = str(options.get("redox mode", "half wave")).strip().lower()

    if manual_redox is not None or redox_mode == "manual":
        if manual_redox is None:
            raise ValueError(
                "'redox mode' was set to 'manual' but no numeric "
                "'redox potential' was provided."
            )
        return float(manual_redox), "manual", None, "manual"

    # keep these silent so FOWA does not clutter the raw plot
    redox_options = _analysis_options_for(PeakCurrentOptions, internal_options)
    redox_options["plot"] = False
    redox_options["plot all"] = False
    redox_options["print"] = False
    redox_options["print all"] = False
    redox_options["internal call"] = True
    redox_options["new plot"] = False

    if redox_mode in {"half wave", "e1/2", "eredox"}:
        if ref_cv is None:
            raise ValueError(
                "Could not determine redox potential automatically. "
                "Please enter 'redox potential' or provide a "
                "'non-catalytic cv'."
            )
        try:
            redox_result = ref_cv.half_wave_potential(redox_options)
            redox_potential = redox_result["E(1/2)"]
            redox_delta_E = redox_result["ΔE"]
        except Exception as exc:
            raise ValueError(
                f"Could not determine redox potential from half_wave_potential() "
                f"for reference CV '{ref_cv.name}'. Please enter "
                f"'redox potential'."
            ) from exc

        return redox_potential, ref_cv.name, redox_delta_E, "half wave"

    if redox_mode in {"half peak", "ep/2", "epeak/2"}:
        try:
            half_peak_options = redox_options.copy()

            if half_peak_options.get("exact potential") is None and ref_cv is not None:
                ref_peak_options = _analysis_options_for(PeakPotentialOptions, half_peak_options)
                ref_Ep = ref_cv.peak_potential(ref_peak_options)["Ep"]
                half_peak_options["exact potential"] = float(ref_Ep)
                half_peak_options["guess potential"] = None

            redox_result = cat_cv.half_peak_potential(half_peak_options)
            redox_potential = redox_result["Ep/2"]
            redox_delta_E = redox_result["Δ(Ep - Ep/2)"]

        except Exception as exc:
            raise ValueError(
                f"Could not determine redox potential from half_peak_potential() "
                f"for catalytic CV '{cat_cv.name}'. Try setting "
                f"'exact potential', 'guess potential', 'tangent potential', or enter a manual "
                f"'redox potential'."
            ) from exc

        return redox_potential, cat_cv.name, redox_delta_E, "half peak"

    raise ValueError(
        "'redox mode' must be 'half wave', 'half peak', or 'manual'."
    )

def _resolve_fowa_range_or_sequence(value, n_items, option_name="fit range", default=None):
    """
    Accept either:
    - None -> broadcast default range
    - a single 2-number range -> broadcast to all CVs
        e.g. [0.1, 0.3]
    - a list of 2-number ranges with length n_items -> one range per CV
        e.g. [[0.1, 0.3], [0.1, 0.3], [0.01, 0.1]]
    """
    if default is None:
        default = [0.1, 0.3]

    if value is None:
        value = default

    if not isinstance(value, (list, tuple, np.ndarray)):
        raise TypeError(
            f"'{option_name}' must be a 2-number range or a list of 2-number ranges."
        )

    values = list(value)

    # Single shared range: [lo, hi]
    if len(values) == 2 and all(isinstance(v, Real) for v in values):
        lo, hi = sorted(float(v) for v in values)
        return [[lo, hi] for _ in range(n_items)]

    # Per-CV ranges: [[lo, hi], [lo, hi], ...]
    if len(values) != n_items:
        raise ValueError(
            f"'{option_name}' must be a single 2-number range or have one range "
            f"per catalytic CV ({n_items}). Got {len(values)} entries."
        )

    resolved = []
    for i, range_i in enumerate(values, start=1):
        if not isinstance(range_i, (list, tuple, np.ndarray)) or len(range_i) != 2:
            raise ValueError(
                f"Entry {i} of '{option_name}' must be a 2-number range. "
                f"Got {range_i!r}."
            )

        lo, hi = sorted(float(v) for v in range_i)
        resolved.append([lo, hi])

    return resolved

def _resolve_fowa_scalar_or_sequence(value, n_items, option_name, allow_none=True):
    """
    Accept either:
    - a single scalar -> broadcast to all CVs
    - a sequence of length n_items -> one value per CV
    - None -> returns [None] * n_items if allow_none is True
    """
    if value is None:
        if allow_none:
            return [None] * n_items
        raise ValueError(f"'{option_name}' cannot be None.")

    if isinstance(value, (list, tuple, np.ndarray, pd.Series)):
        values = list(value)
        if len(values) != n_items:
            raise ValueError(
                f"'{option_name}' must be a scalar or have the same length as the "
                f"number of catalytic CVs ({n_items}). Got {len(values)}."
            )
        out = []
        for i, v in enumerate(values, start=1):
            if v is None:
                if allow_none:
                    out.append(None)
                else:
                    raise ValueError(f"Entry {i} of '{option_name}' cannot be None.")
            else:
                out.append(float(v))
        return out

    return [float(value)] * n_items

def _contiguous_true_runs(mask):
    """
    Return contiguous index runs where mask is True.
    """
    mask = np.asarray(mask, dtype=bool)
    idx = np.flatnonzero(mask)

    if len(idx) == 0:
        return []

    breaks = np.where(np.diff(idx) > 1)[0] + 1
    return [run for run in np.split(idx, breaks) if len(run) > 0]


def _resolve_fowa_fit_mask(x_fowa, y_fowa, fit_basis, fit_lo, fit_hi, options):
    """
    Resolve the FOWA fit mask.

    For x-basis:
        select points where fit_lo <= x_fowa <= fit_hi.

    For y-basis:
        select points where fit_lo <= y_fowa <= fit_hi,
        then choose a continuous net-rising region.
    """
    x = np.asarray(x_fowa, dtype=float)
    y = np.asarray(y_fowa, dtype=float)

    finite = np.isfinite(x) & np.isfinite(y)
    fit_basis = str(fit_basis).strip().lower()

    if fit_basis in {"x", "x fowa", "xfowa", "transformed x"}:
        demand_mask = finite & (x >= fit_lo) & (x <= fit_hi)
        fit_basis_label = "x_fowa"

    elif fit_basis in {"y", "y fowa", "yfowa", "i/ip0", "current"}:
        demand_mask = finite & (y >= fit_lo) & (y <= fit_hi)
        fit_basis_label = "y_fowa"

    else:
        raise ValueError(
            "'fit basis' must be 'x' or 'y'. "
            "Use 'x' to select by x_fowa or 'y' to select by i/ip0."
        )

    runs = _contiguous_true_runs(demand_mask)

    meta = {
        "fit basis label": fit_basis_label,
        "num candidate regions": len(runs),
        "candidate region sizes": [int(len(run)) for run in runs],
    }

    if len(runs) == 0:
        return demand_mask, fit_basis_label, meta

    minimum_fit_points = int(options.get("minimum fit points", 8))
    region_mode = str(options.get("fit region", "largest continuous")).strip().lower()
    trend_mode = str(options.get("fit trend", "auto")).strip().lower()

    if trend_mode == "auto":
        trend_mode = "rising" if fit_basis_label == "y_fowa" else "any"

    trend_min_delta = float(options.get("fit trend min delta", 0.0))

    candidate_runs = []

    for run in runs:
        candidate = np.asarray(run, dtype=int)

        # For y-basis, remove the early downward hook by starting
        # at the minimum y-value inside the continuous demand region.
        if fit_basis_label == "y_fowa" and trend_mode in {"rising", "increasing"}:
            min_rel_idx = int(np.nanargmin(y[candidate]))
            candidate = candidate[min_rel_idx:]

        if len(candidate) < minimum_fit_points:
            continue

        if trend_mode in {"rising", "increasing"}:
            if y[candidate[-1]] <= y[candidate[0]] + trend_min_delta:
                continue

        candidate_runs.append(candidate)

    # Fallback: if no net-rising region survives, use the original runs
    # so the user gets a fit-points error/warning rather than a mysterious empty mask.
    if len(candidate_runs) == 0:
        candidate_runs = runs

    if region_mode in {"all", "none", "full"}:
        chosen = np.concatenate(candidate_runs)

    elif region_mode in {"largest", "largest continuous", "continuous"}:
        chosen = max(candidate_runs, key=len)

    elif region_mode in {"first", "first continuous"}:
        chosen = candidate_runs[0]

    elif region_mode in {"last", "last continuous"}:
        chosen = candidate_runs[-1]

    else:
        raise ValueError(
            "'fit region' must be 'largest continuous', 'first continuous', "
            "'last continuous', or 'all'."
        )

    fit_mask = np.zeros_like(demand_mask, dtype=bool)
    fit_mask[chosen] = True

    meta.update({
        "fit region mode": region_mode,
        "fit trend mode": trend_mode,
        "fit trend min delta": trend_min_delta,
        "fit start index": int(chosen[0]) if len(chosen) else None,
        "fit end index": int(chosen[-1]) if len(chosen) else None,
        "fit start y": float(y[chosen[0]]) if len(chosen) else None,
        "fit end y": float(y[chosen[-1]]) if len(chosen) else None,
        "selected region size": int(len(chosen)),
    })

    return fit_mask, fit_basis_label, meta


def _plot_fowa_normalized_diagnostics(
    ax,
    diagnostic_calls,
    copy_by_original_id,
    object_offsets,
    options,
):
    """
    Redraw FOWA plot-all diagnostics on the temporary i/ip0 CV copies.
    """
    seen = set()

    for call in diagnostic_calls:
        original = call["obj"]
        obj_copy = copy_by_original_id.get(id(original))
        if obj_copy is None:
            continue

        key = (call["kind"], id(original))
        if key in seen:
            continue
        seen.add(key)

        option_model = (
            PeakPotentialOptions
            if call["kind"] == "peak_potential"
            else PeakCurrentOptions
        )
        allowed_options = {normalize_key(field.name) for field in fields(option_model)}
        diag_options = {
            key: value
            for key, value in call["options"].items()
            if normalize_key(key) in allowed_options
        }
        diag_options["y axis"] = "i/ip0"
        diag_options["y unit"] = None
        diag_options["ylabel"] = "$i / i_p^0$"
        diag_options["plot"] = True
        diag_options["plot all"] = True
        diag_options["print"] = False
        diag_options["print all"] = False
        diag_options["internal call"] = True
        diag_options["new plot"] = False
        diag_options["offset"] = object_offsets.get(id(original), 0)
        if call["kind"] == "peak_current":
            diag_options["plot peak potential"] = False

        plt.sca(ax)
        try:
            if call["kind"] == "half_wave":
                obj_copy.half_wave_potential(diag_options)
            elif call["kind"] == "peak_current":
                obj_copy.peak_current(diag_options)
            elif call["kind"] == "peak_potential":
                obj_copy.peak_potential(diag_options)
        except Exception as exc:
            if options.get("troubleshoot", False):
                warnings.warn(
                    f"Could not redraw FOWA plot-all diagnostic for "
                    f"'{getattr(original, 'name', 'CV')}': {exc}"
                )


def _record_fowa_issue(row_warnings, row_status, status, message, options):
    row_status.append(status)
    row_warnings.append(message)
    if options.get("warnings", True):
        warnings.warn(message, UserWarning, stacklevel=2)


def _format_fowa_status(statuses):
    if not statuses:
        return "ok"
    unique = []
    for status in statuses:
        if status not in unique:
            unique.append(status)
    return " | ".join(unique)


def _format_fowa_tangent_background_error(cv_name, options):
    settings = {
        "background correction": options.get("background correction"),
        "tangent range": options.get("tangent range", "auto"),
        "tangent potential": options.get("tangent potential"),
        "tangent min points": options.get("tangent min points"),
        "percent threshold": options.get("percent threshold"),
    }
    settings_text = "\n".join(
        f"  {key}: {value}"
        for key, value in settings.items()
    )
    return (
        f"Could not fit FOWA tangent background for '{cv_name}'.\n\n"
        "Current tangent settings:\n"
        f"{settings_text}\n\n"
        "Try one of:\n"
        "  - set 'tangent range' manually around a flat pre-wave region\n"
        "  - set 'tangent potential' manually\n"
        "  - increase 'percent threshold'\n"
        "  - use {'background correction': 'start current'} if tangent correction is not appropriate\n"
        "  - set {'troubleshoot': True} to inspect tangent selection"
    )


def fowa(cvs, options={}):
    """Run foot-of-the-wave analysis on one CV or a list of CVs.
    
    Parameters
    ----------
    cvs : cv or sequence of cv
        Catalytic CV or CVs to analyze.
    options : dict or FOWAOptions, optional
        Redox, normalization, fit, print, and diagnostic-plot options. See ``e.describe_options("fowa")``.
    
    Returns
    -------
    pandas.DataFrame
        FOWA results table with fitted rate and diagnostic quantities.
    
    Examples
    --------
    >>> df = e.fowa(cvs, {"redox mode": "half wave", "non-catalytic cv": blank_cv})
    """
    typed_options = FOWAOptions.from_options(options)
    options = typed_options.to_legacy_dict()


    cvs = _coerce_cv_list(cvs)
    ref_cvs = _resolve_non_catalytic_cvs(cvs, options)

    manual_ip0_values = _resolve_manual_ip0_values(options, len(cvs))

    manual_redox_values = _resolve_fowa_scalar_or_sequence(
        options.get("redox potential"),
        len(cvs),
        "redox potential",
        allow_none=True,
    )

    fit_ranges = _resolve_fowa_range_or_sequence(
        options.get("fit range"),
        len(cvs),
        option_name="fit range",
        default=[0.1, 0.3],
    )
    wave_ranges = None
    if options.get("wave range") is not None:
        wave_ranges = _resolve_fowa_range_or_sequence(
            options.get("wave range"),
            len(cvs),
            option_name="wave range",
            default=None,
        )

    fit_basis_values = options.get("fit basis", "x")
    if isinstance(fit_basis_values, str):
        fit_basis_values = [fit_basis_values] * len(cvs)
    elif isinstance(fit_basis_values, (list, tuple, np.ndarray)):
        fit_basis_values = list(fit_basis_values)
        if len(fit_basis_values) != len(cvs):
            raise ValueError(
                "'fit basis' must be a string or have one entry per catalytic CV."
            )
    else:
        raise TypeError("'fit basis' must be a string or a list of strings.")

    min_fit_points = int(options.get("min fit points", 20))
    
    min_r2 = float(options.get("min r2", 0.98))
    ecat_shift_warning_threshold = options.get("ecat shift warning threshold", 0.05)

    # FOWA should analyze one segment at a time.
    analysis_segments = options.get("segments", None)
    analysis_segment = options.get("segment", None)

    if analysis_segments is not None:
        if isinstance(analysis_segments, int):
            analysis_segments = [analysis_segments]
        elif not isinstance(analysis_segments, (list, tuple, np.ndarray)):
            raise TypeError("'segments' must be an int or a list/tuple of ints.")
    else:
        # default to segment 1 if nothing is provided
        analysis_segments = [1 if analysis_segment is None else analysis_segment]

    # for now, FOWA analyzes one segment at a time; use the first requested segment
    fowa_segment = analysis_segments[0]

    diagnostic_y_axis = str(options.get("diagnostic y axis", "i/ip0")).strip().lower()
    if diagnostic_y_axis not in {"i/ip0", "current"}:
        raise ValueError("'diagnostic y axis' must be 'i/ip0' or 'current'.")

    raw_ax = None
    resolved_color_map = {}
    plot_all_normalized = options.get("plot all", False) and diagnostic_y_axis == "i/ip0"
    raw_plot_copies = {}
    raw_plot_overlays = []
    normalized_diagnostic_calls = []

    # Build the shared CV order once
    all_cvs = []
    seen = set()

    for ref_cv in ref_cvs:
        if ref_cv is not None and id(ref_cv) not in seen:
            all_cvs.append(ref_cv)
            seen.add(id(ref_cv))

    for cat_cv in cvs:
        if id(cat_cv) not in seen:
            all_cvs.append(cat_cv)
            seen.add(id(cat_cv))

    # Only minimally adjust options before calling multiplot
    raw_plot_options = options.copy()
    raw_plot_options["print"] = False
    raw_plot_options["print all"] = False
    raw_plot_options["new plot"] = False
    raw_plot_options["plot all"] = False
    raw_plot_options["y axis"] = diagnostic_y_axis

    fowa_plot_labels = options.get("plot labels")
    label_map = {}

    # Reference CV labels
    for ref_cv in ref_cvs:
        if ref_cv is not None and id(ref_cv) not in label_map:
            label_map[id(ref_cv)] = ref_cv.name

    # Catalytic CV labels
    if fowa_plot_labels is not None:
        if len(fowa_plot_labels) != len(cvs):
            raise ValueError("'plot labels' must match the number of catalytic CVs passed to FOWA.")
        for cat_cv, lbl in zip(cvs, fowa_plot_labels):
            label_map[id(cat_cv)] = lbl
    else:
        for cat_cv in cvs:
            label_map[id(cat_cv)] = cat_cv.name

    if fowa_plot_labels is not None:
        ordered_plot_labels = [
            label_map.get(id(obj), getattr(obj, "name", f"Trace {i + 1}"))
            for i, obj in enumerate(all_cvs)
        ]
        raw_plot_options["labels"] = ordered_plot_labels

    # Separate label-resolution options from the actual multiplot call
    mp_options = _multiplot_options_from_mapping(raw_plot_options)

    resolved_labels, _title, _subtitle, _shared_compounds, _similarities = (
        _resolve_multiplot_labels_title_subtitle(all_cvs, mp_options)
    )

    resolved_label_map = {
        id(obj): lbl
        for obj, lbl in zip(all_cvs, resolved_labels)
    }
    resolved_plot_labels = [
        resolved_label_map.get(id(obj), getattr(obj, "name", f"Trace {i + 1}"))
        for i, obj in enumerate(all_cvs)
    ]
    raw_plot_options["labels"] = resolved_plot_labels

    transformed_label_options = raw_plot_options.copy()
    if fowa_plot_labels is not None:
        transformed_label_options["labels"] = list(fowa_plot_labels)
    else:
        transformed_label_options["labels"] = None
    transformed_labels, _tx_title, _tx_subtitle, _tx_shared, _tx_similarities = (
        _resolve_multiplot_labels_title_subtitle(
            cvs,
            _multiplot_options_from_mapping(transformed_label_options),
        )
    )
    transformed_label_map = {
        id(obj): lbl
        for obj, lbl in zip(cvs, transformed_labels)
    }

    if options.get("plot all", False) and diagnostic_y_axis == "current":
        multiplot(all_cvs, _multiplot_options_from_mapping(raw_plot_options))
        raw_ax = plt.gca()
        resolved_color_map = _extract_line_colors_from_ax(raw_ax, all_cvs)

    results = []
    plot_data = []

    drawn_redox = set()

    for i, (cat_cv, ref_cv) in enumerate(zip(cvs, ref_cvs)):
        row_warnings = []
        row_status = []

        loop_options = options.copy()
        loop_options.pop("segments", None)
        loop_options["segment"] = fowa_segment
        loop_options["y axis"] = "Current"

        internal_options = loop_options.copy()
        internal_options["plot"] = False
        internal_options["plot all"] = False
        internal_options["print"] = options.get("print all", False)
        internal_options["internal call"] = True
        internal_options["new plot"] = False

        nc_guess = options.get("non-catalytic guess potential")
        if nc_guess is None:
            nc_guess = options.get("guess potential")
        if nc_guess is None:
            nc_guess = options.get("redox potential")
        if nc_guess is not None:
            internal_options["guess potential"] = nc_guess

        fit_lo, fit_hi = fit_ranges[i]
        wave_range = None if wave_ranges is None else wave_ranges[i]
        fit_basis = str(fit_basis_values[i]).strip().lower()

        # --- Reference current (ip0) ---
        manual_ip0 = manual_ip0_values[i]

        if manual_ip0 is not None:
            ip0 = float(manual_ip0)
            ip0_source = "manual"
            ref_Ep = None
            ref_tanline = None

        else:
            if ref_cv is None:
                raise ValueError(
                    "FOWA requires either 'non-catalytic current' or 'non-catalytic cv(s)'."
                )
            try:
                if raw_ax is not None and options.get("plot all", False):
                    plt.sca(raw_ax)
                ip0, ip0_source, ref_Ep, ref_tanline = _resolve_reference_ip0(
                    ref_cv,
                    internal_options,
                )
            except Exception as exc:
                raise ValueError(
                    f"Could not determine non-catalytic current from reference CV '{ref_cv.name}'."
                ) from exc
            if plot_all_normalized:
                normalized_diagnostic_calls.append({
                    "kind": "peak_current",
                    "obj": ref_cv,
                    "options": internal_options.copy(),
                })

        if ip0 == 0:
            raise ValueError(f"Resolved ip0 is zero for '{cat_cv.name}', so FOWA cannot proceed.")

        # --- Redox potential ---
        manual_redox = manual_redox_values[i]

        redox_potential, redox_source, redox_delta_E, redox_mode_used = _resolve_fowa_redox_potential(
            cat_cv=cat_cv,
            ref_cv=ref_cv,
            options=options,
            internal_options=internal_options,
            manual_redox=manual_redox,
        )
        if plot_all_normalized and redox_mode_used == "half wave" and ref_cv is not None:
            normalized_diagnostic_calls.append({
                "kind": "half_wave",
                "obj": ref_cv,
                "options": internal_options.copy(),
            })

        catalytic_ecat_half = None
        ecat_shift = None

        if redox_potential is not None:
            try:
                catalytic_ecat_half, _ecat_delta_E = _resolve_catalytic_half_peak_for_shift_check(
                    cat_cv=cat_cv,
                    ref_cv=ref_cv,
                    options=options,
                    internal_options=internal_options,
                )

                ecat_shift = float(catalytic_ecat_half) - float(redox_potential)

                if (
                    ecat_shift_warning_threshold not in (None, False)
                    and abs(ecat_shift) > float(ecat_shift_warning_threshold)
                ):
                    msg = (
                        f"FOWA Ecat/2 check for '{cat_cv.name}': catalytic Ecat/2 "
                        f"differs from the FOWA reference potential by {ecat_shift:.3g} V. "
                        "Interpret kobs as apparent."
                    )
                    _record_fowa_issue(
                        row_warnings,
                        row_status,
                        "Ecat/2 shift > threshold",
                        msg,
                        options,
                    )

            except Exception as exc:
                if options.get("troubleshoot", False):
                    msg = (
                        f"Could not calculate catalytic Ecat/2 shift for '{cat_cv.name}': {exc}"
                    )
                    _record_fowa_issue(
                        row_warnings,
                        row_status,
                        "Ecat/2 shift unavailable",
                        msg,
                        options,
                    )

        # --- Catalytic trace and wave window ---
        x, y = cat_cv.analysis_segment_data(loop_options)
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)

        if plot_all_normalized:
            raw_plot_copies[id(cat_cv)] = _copy_cv_with_fowa_current_axis(
                cat_cv,
                ip0,
                loop_options,
            )
            if ref_cv is not None and id(ref_cv) not in raw_plot_copies:
                raw_plot_copies[id(ref_cv)] = _copy_cv_with_fowa_current_axis(
                    ref_cv,
                    ip0,
                    loop_options,
                )

        if wave_range is None:
            n, m, wave_meta = _auto_fowa_wave_bounds(x, y, redox_potential, options)
            x_wave = x[n:m]
            y_wave = y[n:m]
        else:
            wave_mask, wave_meta = _manual_fowa_wave_mask(x, wave_range, cat_cv.name)
            x_wave = x[wave_mask]
            y_wave = y[wave_mask]

        if raw_ax is not None and diagnostic_y_axis == "current":
            plt.sca(raw_ax)

            color = resolved_color_map.get(id(cat_cv), options.get("color", "black"))
            x_scale, y_scale = cat_cv.xy_scale(options)
            
            plt.plot(
                x_wave * x_scale,
                y_wave * y_scale + options.get("offset", 0),
                color=color,
                linewidth=3,
                solid_capstyle="round",
                label="_nolegend_",
            )
        elif plot_all_normalized:
            raw_plot_overlays.append({
                "kind": "wave",
                "obj": cat_cv,
                "x": x_wave,
                "y": y_wave / ip0,
            })
            
        redox_key = round(float(redox_potential), 6)
        if raw_ax is not None and diagnostic_y_axis == "current" and redox_key not in drawn_redox:
            plt.sca(raw_ax)
            plt.axvline(
                redox_potential * x_scale,
                color=resolved_color_map.get(id(ref_cv), "0.4"),
                linestyle="--",
                alpha=0.5,
                label="_nolegend_",
            )
            drawn_redox.add(redox_key)
        elif plot_all_normalized and redox_key not in drawn_redox:
            raw_plot_overlays.append({
                "kind": "redox",
                "obj": ref_cv if ref_cv is not None else cat_cv,
                "x": redox_potential,
            })
            drawn_redox.add(redox_key)

        if len(x_wave) < 8:
            raise ValueError(
                f"Selected wave window for '{cat_cv.name}' is too small for FOWA."
            )

        # --- Background correction on catalytic CV ---
        background_mode = options.get("background correction", 'tangent')
        background_tangent_potential = None
        background_slope = None
        background_intercept = None

        if background_mode is None:
            y_corr = y_wave.copy()

        elif background_mode == "start current":
            y_corr = y_wave - y_wave[0]

        elif background_mode == "tangent":
            background_tangent_potential = loop_options.get('tangent potential')
            manual_peak_potential = loop_options.get("peak potential")

            tangent_options = loop_options.copy()
            tangent_options["print"] = False
            tangent_options["internal call"] = True
            tangent_options["new plot"] = False
            tangent_options["plot all"] = False

            try:
                # Case 1: user explicitly gives the tangent potential
                if background_tangent_potential is not None:
                    tangent_meta = cat_cv._fit_tangent_line(
                        x,
                        y,
                        idx_target=None,
                        tangent_potential=background_tangent_potential,
                        options=tangent_options,
                    )

                # Case 2: user gives a manual peak potential, even if no true local peak exists
                elif manual_peak_potential is not None:
                    idx_E_peak = int(np.argmin(np.abs(x - float(manual_peak_potential))))
                    tangent_meta = cat_cv._fit_tangent_line(
                        x,
                        y,
                        idx_target=idx_E_peak,
                        tangent_potential=None,
                        options=tangent_options,
                    )

                # Case 3: automatic anchor selection
                else:
                    anchor_potential = loop_options.get("peak potential")
                    if anchor_potential is None:
                        anchor_potential = loop_options.get("exact potential")
                    if anchor_potential is None:
                        anchor_potential = redox_potential  # fallback to resolved E1/2 or Ep/2

                    idx_E_peak = int(np.argmin(np.abs(x - float(anchor_potential))))
                    background_tangent_potential = float(anchor_potential)

                    tangent_meta = cat_cv._fit_tangent_line(
                        x,
                        y,
                        idx_target=idx_E_peak,
                        tangent_potential=None,
                        options=tangent_options,
                    )
            except Exception as exc:
                raise ValueError(
                    _format_fowa_tangent_background_error(cat_cv.name, tangent_options)
                ) from exc

            background_slope = tangent_meta["slope"]
            background_intercept = tangent_meta["intercept"]
            y_background = background_slope * x_wave + background_intercept
            y_corr = y_wave - y_background

            if raw_ax is not None and diagnostic_y_axis == "current" and options.get("plot all", False):
                plt.sca(raw_ax)
                x_scale, y_scale = cat_cv.xy_scale(options)
                color = resolved_color_map.get(id(cat_cv), options.get("color", "black"))

                plt.plot(
                    x_wave * x_scale,
                    y_background * y_scale + options.get("offset", 0),
                    linestyle="--",
                    color=color,
                    alpha=0.8,
                    label="_nolegend_",
                )

                fit_idx = np.asarray(tangent_meta.get("fit_indices", []), dtype=int)
                if fit_idx.size > 0:
                    plt.scatter(
                        x[fit_idx] * x_scale,
                        y[fit_idx] * y_scale + options.get("offset", 0),
                        s=10,
                        color=color,
                        zorder=3,
                        label="_nolegend_",
                    )
            elif plot_all_normalized:
                raw_plot_overlays.append({
                    "kind": "background",
                    "obj": cat_cv,
                    "x": x_wave,
                    "y": y_background / ip0,
                })

                fit_idx = np.asarray(tangent_meta.get("fit_indices", []), dtype=int)
                if fit_idx.size > 0:
                    raw_plot_overlays.append({
                        "kind": "fit points",
                        "obj": cat_cv,
                        "x": x[fit_idx],
                        "y": y[fit_idx] / ip0,
                    })

        else:
            raise ValueError(
                "'background correction' must be None, 'start current', or 'tangent'."
            )

        # --- Transform ---
        f = F / (R * cat_cv.temperature)
        catalyst_electrons = float(
            options.get("catalyst electrons", options.get("num electrons", 1))
        )

        if catalyst_electrons <= 0:
            raise ValueError("'catalyst electrons' must be positive.")

        x_fowa = 1.0 / (
                1.0 + np.exp(catalyst_electrons * f * (x_wave - redox_potential))
        )
        y_fowa = y_corr / ip0

        fit_mask, fit_basis_label, fit_region_meta = _resolve_fowa_fit_mask(
            x_fowa=x_fowa,
            y_fowa=y_fowa,
            fit_basis=fit_basis,
            fit_lo=fit_lo,
            fit_hi=fit_hi,
            options=options,
        )

        n_fit = int(np.count_nonzero(fit_mask))

        if n_fit < 5:
            raise ValueError(
                f"Only {n_fit} points fall in the requested FOWA fit range "
                f"[{fit_lo}, {fit_hi}] for '{cat_cv.name}'."
            )
        if n_fit < min_fit_points:
            msg = (
                f"FOWA fit for '{cat_cv.name}' uses only {n_fit} points "
                f"(< recommended {min_fit_points})."
            )
            _record_fowa_issue(
                row_warnings,
                row_status,
                "fit points < threshold",
                msg,
                options,
            )

        x_fit = x_fowa[fit_mask]
        y_fit = y_fowa[fit_mask]

        order = np.argsort(x_fit)
        x_fit = x_fit[order]
        y_fit = y_fit[order]

        slope, intercept = np.polyfit(x_fit, y_fit, 1)
        y_pred = slope * x_fit + intercept
        r2 = float(r2_score(y_fit, y_pred))

        if r2 < min_r2:
            msg = (
                f"FOWA fit for '{cat_cv.name}' has R^2 = {r2:.3f}, "
                f"below the recommended threshold of {min_r2:.2f}."
            )
            _record_fowa_issue(
                row_warnings,
                row_status,
                "fit R2 < threshold",
                msg,
                options,
            )

        if slope <= 0:
            msg = f"FOWA fit for '{cat_cv.name}' has a non-positive slope."
            _record_fowa_issue(
                row_warnings,
                row_status,
                "nonpositive slope",
                msg,
                options,
            )

        formula_label, kinetics = _resolve_fowa_formula(cat_cv, slope, options)

        results.append({
            "reference cv": None if ref_cv is None else ref_cv.name,
            "ip0 source": ip0_source,
            "redox source": redox_source,
            "reference Ep": ref_Ep,
            "redox mode": redox_mode_used,
            "redox delta E": redox_delta_E,
            "redox potential": redox_potential,
            "catalytic Ecat/2": catalytic_ecat_half,
            "Ecat/2 shift": ecat_shift,
            "ip0": ip0,
            "background correction": background_mode if background_mode is not None else "none",
            "background tangent potential": background_tangent_potential if background_tangent_potential is not None else "none",
            "background slope": background_slope,
            "background intercept": background_intercept,
            "fit basis": fit_basis,
            "fit range": f"[{fit_lo}, {fit_hi}]",
            "wave start": x_wave[0],
            "wave end": x_wave[-1],
            "n fit points": n_fit,
            "slope": slope,
            "intercept": intercept,
            "r2": r2,
            "formula": formula_label,
            "status": _format_fowa_status(row_status),
            "warning details": " | ".join(row_warnings),
            **kinetics,
        })

        plot_data.append({
            "name": cat_cv.name,
            "cat cv": cat_cv,
            "ref cv": ref_cv,
            "x raw": x,
            "y raw": y,
            "x wave": x_wave,
            "y wave": y_wave,
            "x fowa": x_fowa,
            "y fowa": y_fowa,
            "x fit": x_fit,
            "y fit": y_fit,
            "y pred": y_pred,
            "wave meta": wave_meta,
            "redox potential": redox_potential,
            "redox mode": redox_mode_used,
            "catalyst electrons": catalyst_electrons,
            "color": resolved_color_map.get(id(cat_cv), options.get("color", "black")),
            "plot label": transformed_label_map.get(id(cat_cv), cat_cv.name),
            "ref color": resolved_color_map.get(id(ref_cv), "0.4") if ref_cv is not None else "0.4",
        })

        if options.get("print all", False):
            print(f"\n### FOWA {i}: {cat_cv.name} ###")

            if options.get("non-catalytic current") is not None:
                print(f"Using manual non-catalytic current (ip0): {ip0:.6g}")
            else:
                print(f"Using non-catalytic CV for ip0: {ref_cv.name}")
                if ref_Ep is not None:
                    print(f"  Reference peak potential (Ep): {ref_Ep:.6g}")
                print(f"  Tangent-corrected ip0: {ip0:.6g}")

            if redox_mode_used == "manual":
                print(f"Using manual redox potential: {redox_potential:.6g}")
            elif redox_mode_used == "half wave":
                print(f"Using non-catalytic CV E1/2 for redox potential: {redox_source}")
                print(f"  Resolved E1/2: {redox_potential:.6g}")
            elif redox_mode_used == "half peak":
                print(f"Using catalytic CV Ep/2 for redox potential: {redox_source}")
                print(f"  Resolved Ep/2: {redox_potential:.6g}")

            print(f"Background correction: {background_mode if background_mode is not None else 'none'}")
            if background_mode == "tangent" and background_tangent_potential is not None:
                print(f"  Automatic tangent potential: {background_tangent_potential:.6g}")

            print(f"Wave window: {x_wave[0]:.6g} to {x_wave[-1]:.6g}")
            print(f"Fit range (transformed x): [{fit_lo}, {fit_hi}]")
            print(f"Fit points: {n_fit}")
            print(f"Formula: {formula_label}")

            if formula_label == "EC' apparent":
                print(
                    "k_obs = (slope * 0.4463 * n_ref / n_cat^sigma)^2 "
                    "* (n_ref*F*v)/(RT)\n"
                    "  TOF_max = k_obs"
                )
                print(
                    "where n_ref = 'catalyst electrons', n_cat = 'turnover electrons', "
                    "and sigma is the ET-pathway exponent."
                )
                print(
                    "  Key options to adjust if needed: "
                    "'catalyst electrons', 'turnover electrons', 'sigma', "
                    "'fit range', 'redox potential', 'non-catalytic current', "
                    "'background correction'."
                )
            else:
                print(
                    "  Custom formula used. Adjust 'custom formula' and 'formula label' "
                    "if you want a different mechanism."
                )

            print(f"Slope: {slope:.6g}")
            print(f"Intercept: {intercept:.6g}")
            print(f"R^2: {r2:.4f}")
            for key, value in kinetics.items():
                if isinstance(value, (int, float, np.floating)):
                    print(f"{key}: {value:.6g}")
                else:
                    print(f"{key}: {value}")

            if row_warnings:
                print("Warnings:")
                for msg in row_warnings:
                    print(f"  - {msg}")

    if plot_all_normalized:
        normalized_cvs = [
            raw_plot_copies.get(id(obj))
            for obj in all_cvs
        ]
        if any(obj is None for obj in normalized_cvs):
            missing = [
                getattr(obj, "name", f"Trace {i + 1}")
                for i, obj in enumerate(all_cvs)
                if raw_plot_copies.get(id(obj)) is None
            ]
            raise ValueError(
                "Could not create FOWA-normalized plot copies for: "
                + ", ".join(missing)
            )

        normalized_plot_options = raw_plot_options.copy()
        normalized_plot_options["y axis"] = "i/ip0"
        normalized_plot_options["y unit"] = None
        normalized_plot_options["ylabel"] = "$i / i_p^0$"
        normalized_plot_options["plot all"] = False

        multiplot(normalized_cvs, _multiplot_options_from_mapping(normalized_plot_options))
        raw_ax = plt.gca()
        copy_color_map = _extract_line_colors_from_ax(raw_ax, normalized_cvs)
        resolved_color_map = {
            id(orig): copy_color_map.get(id(copy), "0.4")
            for orig, copy in zip(all_cvs, normalized_cvs)
        }
        object_offsets = {
            id(obj): raw_plot_options.get("offset", 0) * i
            for i, obj in enumerate(all_cvs)
        }

        main_xlim = raw_ax.get_xlim()
        main_ylim = raw_ax.get_ylim()

        for overlay in raw_plot_overlays:
            obj = overlay.get("obj")
            color = resolved_color_map.get(id(obj), "0.4")
            offset = object_offsets.get(id(obj), 0)

            if overlay["kind"] == "wave":
                raw_ax.plot(
                    overlay["x"],
                    overlay["y"] + offset,
                    color=color,
                    linewidth=3,
                    solid_capstyle="round",
                    label="_nolegend_",
                )
            elif overlay["kind"] == "redox":
                raw_ax.axvline(
                    overlay["x"],
                    color=color,
                    linestyle="--",
                    alpha=0.5,
                    label="_nolegend_",
                )
            elif overlay["kind"] == "background":
                raw_ax.plot(
                    overlay["x"],
                    overlay["y"] + offset,
                    linestyle="--",
                    color=color,
                    alpha=0.8,
                    label="_nolegend_",
                )
            elif overlay["kind"] == "fit points":
                raw_ax.scatter(
                    overlay["x"],
                    overlay["y"] + offset,
                    s=10,
                    color=color,
                    zorder=3,
                    label="_nolegend_",
                )

        copy_by_original_id = {
            id(orig): copy
            for orig, copy in zip(all_cvs, normalized_cvs)
        }
        _plot_fowa_normalized_diagnostics(
            raw_ax,
            normalized_diagnostic_calls,
            copy_by_original_id,
            object_offsets,
            options,
        )

        raw_ax.set_ylabel("$i / i_p^0$")
        raw_ax.set_xlabel(all_cvs[0].format_axis_label(all_cvs[0].x(raw_plot_options).name, all_cvs[0].units.get(all_cvs[0].x(raw_plot_options).name, "")))
        raw_ax.set_xlim(main_xlim)
        raw_ax.set_ylim(main_ylim)

    display_df = _fowa_summary_table(cvs, results, plot_data, ref_cvs, options)
    shared_summary = display_df.attrs.get("shared_summary", {})

    if options.get("print", True):
        print("\n### FOWA Summary ###")

        combined_summary = {}

        # Keep the manually-added structural info you liked
        combined_summary["Segment"] = fowa_segment
        combined_summary["Fit Range"] = f"[{fit_lo}, {fit_hi}]"
        combined_summary["Background Correction"] = background_mode if background_mode is not None else "none"
        combined_summary["Mechanism"] = options.get("mechanism", "EC'")

        # If the source was manual, include the explicit manual value
        if options.get("non-catalytic current") is not None:
            combined_summary["ip0 Source"] = f"manual ({options['non-catalytic current']:.6g})"

        if options.get("redox potential") is not None:
            combined_summary["Redox Source"] = f"manual ({float(options['redox potential']):.6g} V)"

        # If a single shared ref CV was used and it got split out, keep it in summary
        unique_refs = []
        for ref_cv in ref_cvs:
            if ref_cv is not None and ref_cv.name not in unique_refs:
                unique_refs.append(ref_cv.name)

        if options.get("non-catalytic cvs") is None:
            if len(unique_refs) == 1:
                combined_summary.setdefault("Reference CV", unique_refs[0])
            elif len(unique_refs) > 1:
                combined_summary.setdefault("Reference CVs", ", ".join(unique_refs))

        # Add everything auto-detected as shared
        for key, value in shared_summary.items():
            if key not in combined_summary:
                combined_summary[key] = value

        preferred_order = [
            "Reference CV",
            "Reference CVs",
            "ip0 Source",
            "Redox Source",
            "Redox Potential",
            "Reference Ep",
            "Reference Delta E",
            "Segment",
            "Background Correction",
            "Background Tangent Potential",
            "Fit Range",
            "Mechanism",
        ]

        ordered_summary = {}
        for key in preferred_order:
            if key in combined_summary:
                ordered_summary[key] = combined_summary[key]
        for key, value in combined_summary.items():
            if key not in preferred_order:
                ordered_summary[key] = value

        _display_fowa_summary_table(ordered_summary, options)

        _display_fowa_kobs_equation(
            options,
            resolved=False,
            compact=options.get("print all", False),
        )

        display_object_table(display_df)


    if options.get("plot", True):
        _plot_fowa_transformed(
            plot_data,
            display_df.attrs.get("full_results_df", display_df),
            options,
        )

    if len(cvs) == 1:
        single_df = display_df.iloc[[0]].reset_index(drop=True)
        single_df.attrs.update(display_df.attrs)
        return single_df

    return display_df

def _resolve_df_column(df, requested, aliases=None, required=True):
    """
    Resolve a dataframe column name case-insensitively.

    Parameters
    ----------
    df : pd.DataFrame
        Source dataframe.
    requested : str
        Requested column name.
    aliases : list[str] or None
        Additional accepted names.
    required : bool
        If True, raise an error when no column is found.

    Returns
    -------
    str or None
        Actual dataframe column name.
    """
    aliases = [] if aliases is None else aliases
    candidates = [requested] + aliases

    col_lookup = {str(col).strip().lower(): col for col in df.columns}

    for candidate in candidates:
        if candidate is None:
            continue
        key = str(candidate).strip().lower()
        if key in col_lookup:
            return col_lookup[key]

    if required:
        raise ValueError(
            f"Could not find column '{requested}'. "
            f"Available columns: {list(df.columns)}"
        )

    return None


def _parse_scan_rate_value(value):
    """
    Parse scan-rate values into V/s.

    Accepts values like:
    - 0.1
    - '0.1 V/s'
    - '100 mV/s'
    - '100 mVs'
    """
    if isinstance(value, (int, float, np.number)) and not isinstance(value, bool):
        return float(value)

    text = str(value).strip().replace("μ", "u")
    match = re.search(
        r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*([numu]?V\s*/?\s*s?)?",
        text,
    )

    if match is None:
        return np.nan

    number = float(match.group(1))
    unit = match.group(2)

    if not unit:
        return number

    unit = unit.replace(" ", "").replace("u", "μ")

    if unit.endswith("s") and "/" not in unit:
        # handles mVs, uVs, etc. as mV/s
        unit = unit[:-1] + "/s"

    numerator = unit.split("/", 1)[0]
    return number * get_conversion_factor(numerator, "V")


def _parse_concentration_token(value):
    """
    Parse a concentration token.

    Returns
    -------
    tuple
        (numeric_value, display_unit, species)

    Notes
    -----
    Molar units are converted to M.
    Percent, equiv, and x are kept as their displayed numeric values.
    """
    text = str(value).strip().replace("μ", "u")

    match = re.search(
        r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*([nmu]?M|%|equiv|x)\s*"
        r"([A-Za-z\[\]\(\)\+\-][A-Za-z0-9\[\]\(\)\+\-]*)?",
        text,
    )

    if match is None:
        return np.nan, None, None

    number = float(match.group(1))
    unit = match.group(2).replace("u", "μ")
    species = match.group(3)

    if unit.endswith("M"):
        return number * get_conversion_factor(unit, "M"), "M", species

    return number, unit, species


def _numeric_series_from_column(series, parser=None):
    """
    Convert a dataframe column to numeric values, optionally using a parser.
    """
    direct = pd.to_numeric(series, errors="coerce")

    if direct.notna().all() or parser is None:
        return direct.to_numpy(dtype=float)

    return np.asarray([parser(value) for value in series], dtype=float)


def _resolve_fit_rate_x_from_df(df, options):
    """
    Infer the x-axis for fit_rate from a dataframe.

    Priority:
    1. options['x column'] if provided
    2. Scan Rate column
    3. Explicit concentration columns, e.g. 'CO2 Concentration (M)'
    4. Compounds column containing entries like '10 % CO2'
    """
    x_column = options.get("x column", "auto")
    species_requested = options.get("species")

    # ----- explicit x column -----
    if x_column not in (None, "auto"):
        col = _resolve_df_column(df, x_column)

        if "scan rate" in str(col).lower():
            x_raw = _numeric_series_from_column(df[col], _parse_scan_rate_value)
            return x_raw, "Scan Rate", "V/s", "scan rate", {}

        # If the column name has a unit, keep it.
        col_text = str(col)
        unit_match = re.search(r"\((.*?)\)", col_text)
        unit = unit_match.group(1) if unit_match else ""

        x_raw = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
        label = re.sub(r"\s*\(.*?\)\s*$", "", col_text).strip()
        return x_raw, label, unit, "custom", {}

    # ----- scan rate -----
    scan_col = _resolve_df_column(
        df,
        "Scan Rate",
        aliases=["scan rate", "Scan rate"],
        required=False,
    )

    if scan_col is not None:
        x_raw = _numeric_series_from_column(df[scan_col], _parse_scan_rate_value)
        finite_unique = pd.Series(x_raw[np.isfinite(x_raw)]).nunique()

        if finite_unique > 1:
            return x_raw, "Scan Rate", "V/s", "scan rate", {}

    # ----- explicit concentration columns -----
    concentration_cols = [
        col for col in df.columns
        if "concentration" in str(col).lower()
    ]

    if species_requested is not None:
        concentration_cols = [
            col for col in concentration_cols
            if species_requested.lower() in str(col).lower()
        ]

    for col in concentration_cols:
        col_text = str(col)

        unit_match = re.search(r"\((.*?)\)", col_text)
        unit = unit_match.group(1) if unit_match else "M"

        label = re.sub(r"\s*\(.*?\)\s*$", "", col_text).strip()
        label = re.sub(r"\s*concentration\s*$", "", label, flags=re.IGNORECASE).strip()
        species = species_requested or label

        def parser(value):
            parsed_value, parsed_unit, _parsed_species = _parse_concentration_token(value)
            return parsed_value

        x_raw = _numeric_series_from_column(df[col], parser)
        finite_unique = pd.Series(x_raw[np.isfinite(x_raw)]).nunique()

        if finite_unique > 1:
            x_kind = "mole fraction" if unit == "x" else "concentration"
            return x_raw, species, unit, x_kind, {"species": species}

    # ----- compounds column, e.g. '10 % CO2' -----
    compounds_col = _resolve_df_column(
        df,
        "Compounds",
        aliases=["compounds", "Plot Label", "plot label"],
        required=False,
    )

    if compounds_col is not None:
        parsed_rows = []
        species_order = []

        for value in df[compounds_col]:
            row = {}
            text = str(value).strip()
            if text.startswith("[") and text.endswith("]"):
                text = text[1:-1].strip()

            for token in re.finditer(
                r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*([nmuμ]?M|%|equiv|x)\s*"
                r"([A-Za-z\[\]\(\)\+\-][A-Za-z0-9\[\]\(\)\+\-]*)",
                text,
            ):
                number, unit, species = token.groups()
                parsed_value, display_unit, _ = _parse_concentration_token(
                    f"{number} {unit} {species}"
                )

                key = (species, display_unit)
                row[key] = (parsed_value, display_unit)
                if key not in species_order:
                    species_order.append(key)

            parsed_rows.append(row)

        if species_requested is not None:
            matching_keys = [key for key in species_order if key[0] == species_requested]
            if not matching_keys:
                raise ValueError(
                    f"Species '{species_requested}' was not found in the "
                    f"'{compounds_col}' column. Found: {[key[0] for key in species_order]}"
                )
            varying_keys = []
            for key in matching_keys:
                values = np.asarray(
                    [row.get(key, (0.0, None))[0] for row in parsed_rows],
                    dtype=float,
                )
                finite_unique = pd.Series(values[np.isfinite(values)]).nunique()
                if finite_unique > 1:
                    varying_keys.append(key)
            chosen_key = varying_keys[0] if len(varying_keys) == 1 else matching_keys[0]
        else:
            varying_keys = []
            for key in species_order:
                values = np.asarray(
                    [row.get(key, (0.0, None))[0] for row in parsed_rows],
                    dtype=float,
                )
                finite_unique = pd.Series(values[np.isfinite(values)]).nunique()
                if finite_unique > 1:
                    varying_keys.append(key)

            if len(varying_keys) != 1:
                candidates = [
                    f"{species} ({unit})" if unit else species
                    for species, unit in varying_keys
                ]
                raise ValueError(
                    "Could not uniquely infer the varying species from the "
                    f"'{compounds_col}' column. Provide options['species']. "
                    f"Candidate species: {candidates}"
                )

            chosen_key = varying_keys[0]

        chosen_species, chosen_unit = chosen_key

        x_raw = np.asarray(
            [row.get(chosen_key, (0.0, None))[0] for row in parsed_rows],
            dtype=float,
        )

        units = [
            row.get(chosen_key, (None, None))[1]
            for row in parsed_rows
            if row.get(chosen_key, (None, None))[1] is not None
        ]
        unit = units[0] if units else chosen_unit or "M"
        x_kind = "mole fraction" if unit == "x" else "concentration"

        return (
            x_raw,
            chosen_species,
            unit,
            x_kind,
            {"species": chosen_species},
        )

    raise ValueError(
        "Could not infer fit_rate x-values. The dataframe needs a varying "
        "'Scan Rate' column, a concentration column, or a 'Compounds' column "
        "containing entries like '10 % CO2'."
    )


def _format_fit_rate_x_label(label, unit="", x_kind="custom", transform="", log=False):
    """
    Build a matplotlib-ready x-axis label for fit_rate.
    """
    return _format_symbolic_axis_label(
        label,
        unit=unit,
        x_kind=x_kind,
        transform=transform,
        log=log,
    )


def _format_axis_unit(unit, wrap=False):
    if unit in (None, ""):
        return ""
    text = str(unit)
    text = text.replace("μ", r"\mu ")
    text = text.replace("$", "")
    text = re.sub(r"([A-Za-z\\]+)\^(-?\d+)", r"\1^{\2}", text)
    text = text.replace("*", r"\cdot ")
    return f"$\\mathrm{{{text}}}$" if wrap else text


def _format_fit_rate_metric_label(metric, log=False, unit=None):
    """
    Build a matplotlib-ready y-axis label for rate metrics.
    """
    metric_key = str(metric).strip().lower().replace(" ", "")
    unit_suffix = f" ({_format_axis_unit(unit, wrap=True)})" if unit else ""

    metric_math = {
        "ip": r"i_p",
        "ep": r"E_p",
        "kobs": r"k_{\mathrm{obs}}",
        "tofmax": r"\mathrm{TOF}_{\max}",
        "r2": r"R^2",
    }

    if metric_key in metric_math:
        base = metric_math[metric_key]
        return rf"$\log_{{10}}({base})${unit_suffix}" if log else rf"${base}${unit_suffix}"

    formatted = format_chemical_formulas(str(metric), mode="mathtext")
    return (rf"$\log_{{10}}$({formatted})" if log else formatted) + unit_suffix


def _select_fit_indices(x, y, fit_indices):
    """
    Select fit points using the eCAT convention:
    fit_indices = [start, stop] or [[start, stop], ...]

    If fit_indices is None, all points are used.
    """
    x = np.asarray(x)
    y = np.asarray(y)

    if fit_indices is None:
        return x, y

    if isinstance(fit_indices, slice):
        return x[fit_indices], y[fit_indices]

    fit_indices_array = np.asarray(fit_indices, dtype=object)

    if fit_indices_array.dtype == bool or (
        fit_indices_array.ndim == 1
        and len(fit_indices_array) == len(x)
        and all(isinstance(value, (bool, np.bool_)) for value in fit_indices_array)
    ):
        mask = np.asarray(fit_indices, dtype=bool)
        if len(mask) != len(x):
            raise ValueError("'fit indices' boolean masks must match the data length.")
        return x[mask], y[mask]

    if fit_indices_array.ndim == 1:
        if len(fit_indices_array) == 2:
            start, stop = int(fit_indices_array[0]), int(fit_indices_array[1])
            return x[start:stop], y[start:stop]

        positions = np.asarray(fit_indices_array, dtype=int)
        return x[positions], y[positions]

    if fit_indices_array.ndim == 2 and fit_indices_array.shape[1] == 2:
        positions = []
        base_indices = np.arange(len(x))
        for start, stop in fit_indices_array:
            positions.extend(base_indices[int(start):int(stop)])
        positions = np.asarray(positions, dtype=int)
        return x[positions], y[positions]

    raise ValueError(
        "'fit indices' should be [start, stop], [[start, stop], ...], "
        "a boolean mask, or explicit integer indices."
    )


def _is_fit_range_pair(spec):
    if not isinstance(spec, (list, tuple)) or len(spec) != 2:
        return False
    return all(
        value is None or isinstance(value, (int, float, np.integer, np.floating))
        for value in spec
    )


def _normalize_fit_ranges(fit_ranges):
    """
    Normalize fit_rate's multiple-fit range option.

    Forms
    -----
    {"early": [0, 5], "tail": [[6, 9], [9, None]]}
        Named fits.

    [[0, 5], [6, None]]
        Unnamed fits labeled Fit 1, Fit 2.

    [[[0, 3], [5, 8]], [[8, None]]]
        Unnamed fits where a single fit can use multiple windows.
    """
    if fit_ranges in (None, {}):
        return []

    if isinstance(fit_ranges, dict):
        return [(str(label), spec) for label, spec in fit_ranges.items()]

    if _is_fit_range_pair(fit_ranges):
        return [("Fit 1", fit_ranges)]

    if isinstance(fit_ranges, (list, tuple)):
        return [(f"Fit {i + 1}", spec) for i, spec in enumerate(fit_ranges)]

    raise ValueError(
        "'fit ranges' must be a dict of named ranges or a list of ranges."
    )


def _select_fit_range(x, y, range_spec):
    x = np.asarray(x)
    y = np.asarray(y)

    if range_spec is None:
        return x, y

    if _is_fit_range_pair(range_spec):
        windows = [range_spec]
    elif isinstance(range_spec, (list, tuple)) and all(_is_fit_range_pair(item) for item in range_spec):
        windows = list(range_spec)
    else:
        raise ValueError(
            "Each 'fit ranges' entry must be [start, stop] or a list of [start, stop] windows."
        )

    masks = []
    for window in windows:
        start, stop = window
        mask = np.isfinite(x)
        if start is not None:
            mask &= x >= float(start)
        if stop is not None:
            mask &= x <= float(stop)
        masks.append(mask)

    if not masks:
        return x, y
    combined = np.logical_or.reduce(masks)
    return x[combined], y[combined]


def _fit_rate_range_specs(options, fallback_fit_indices, default_label):
    fit_ranges = options.get("fit ranges")
    if fit_ranges is None and options.get("fit range") is not None:
        fit_ranges = options.get("fit range")
    if fit_ranges is None:
        return [(default_label, fallback_fit_indices, False)]
    return [
        (label, range_spec, True)
        for label, range_spec in _normalize_fit_ranges(fit_ranges)
    ]


def _fit_rate_selected_points(plot_x, plot_y, spec, is_fit_range):
    if is_fit_range:
        return _select_fit_range(plot_x, plot_y, spec)
    return _select_fit_indices(plot_x, plot_y, spec)


def _fit_rate_fit_options_for_range(options, label, is_fit_range):
    fit_options = options.copy()
    if is_fit_range and fit_options.get("fit label", False) is False:
        fit_options["fit label"] = label
    return fit_options

def _local_log_slopes(x, y, x_label="x", y_label="y", mode="adjacent"):
    """
    Estimate local apparent reaction order from log-log slopes.

    local order = d log(y) / d log(x)

    Parameters
    ----------
    x, y : array-like
        Raw positive x and y values.
    x_label, y_label : str
        Labels used in the output dataframe.
    mode : {'adjacent', 'gradient'}
        'adjacent' reports slopes between neighboring points.
        'gradient' reports a pointwise numerical derivative.

    Returns
    -------
    pd.DataFrame
        Local slope/order table.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    keep = (
        np.isfinite(x)
        & np.isfinite(y)
        & (x > 0)
        & (y > 0)
    )

    x = x[keep]
    y = y[keep]

    order = np.argsort(x)
    x = x[order]
    y = y[order]

    if len(x) < 2:
        return pd.DataFrame()

    log_x = np.log10(x)
    log_y = np.log10(y)

    if mode == "gradient":
        if len(x) < 3:
            mode = "adjacent"
        else:
            slopes = np.gradient(log_y, log_x)
            return pd.DataFrame({
                x_label: x,
                y_label: y,
                f"log10({x_label})": log_x,
                f"log10({y_label})": log_y,
                "local order": slopes,
            })

    slopes = np.diff(log_y) / np.diff(log_x)
    x_mid = np.sqrt(x[:-1] * x[1:])
    y_mid = np.sqrt(y[:-1] * y[1:])

    return pd.DataFrame({
        f"{x_label} midpoint": x_mid,
        f"{y_label} midpoint": y_mid,
        f"{x_label} low": x[:-1],
        f"{x_label} high": x[1:],
        "local order": slopes,
    })

def _fit_rate_source_df(df):
    """Return the full hidden rate table aligned to the visible input rows."""
    full_results = getattr(df, "attrs", {}).get("full_results_df")
    if full_results is None:
        return df.copy()

    try:
        row_positions = [int(idx) for idx in df.index]
    except Exception:
        row_positions = None

    if row_positions is None:
        return full_results.copy()

    if any(pos < 0 or pos >= len(full_results) for pos in row_positions):
        return full_results.copy()

    source_df = full_results.iloc[row_positions].copy()
    source_df.index = df.index
    return source_df


def _fit_rate_legacy_return(df, options={}):
    """Fit rate-style tabular data and return the internal legacy tuple.
    
    Parameters
    ----------
    df : pandas.DataFrame
        Table containing rate, transformed x, or metric columns.
    options : dict or FitRateOptions, optional
        Transform, fit-window, print, and plot options. See ``e.describe_options("fit_rate")``.
    
    Returns
    -------
    tuple
        Legacy fit data and fit coefficient return used by notebook workflows.
    
    Examples
    --------
    >>> data, fits = e.fit_rate(fowa_df, {"fit range": [0.1, 0.4]})
    """
    raw_options = options
    typed_options = FitRateOptions.from_options(options)
    options = typed_options.to_legacy_dict()
    if options.get("fit label", False) and not _option_was_provided(raw_options, "legend"):
        options["legend"] = True
    do_plot = options.get("plot", True)
    do_print = options.get("print", True)
    do_fit = options.get("fit", True)
    fit_color_index = 0

    # Use hidden FOWA results if fit_rate is passed a display table, but keep
    # user slicing such as fowa_df.iloc[-5:] scoped to those visible rows.
    source_df = _fit_rate_source_df(df)

    metric_requested = options.get("metric", "kobs")
    metric_col = _resolve_df_column(source_df, metric_requested)
    source_units = {}
    source_units.update(getattr(source_df, "attrs", {}).get("units", {}) or {})
    source_units.update(getattr(df, "attrs", {}).get("units", {}) or {})
    metric_unit = source_units.get(metric_col) or source_units.get(str(metric_col).lower())

    x_raw, x_label, x_unit, x_kind, x_extra = _resolve_fit_rate_x_from_df(
        source_df,
        options,
    )

    metric_values = pd.to_numeric(
        source_df[metric_col],
        errors="coerce",
    ).to_numpy(dtype=float)

    data = source_df.copy()
    data["x raw"] = np.asarray(x_raw, dtype=float)
    data["x label"] = x_label
    data["x unit"] = x_unit
    data["x kind"] = x_kind
    data["y label"] = metric_col
    data[metric_col] = metric_values
    if source_units:
        data.attrs["units"] = dict(source_units)

    # ------------------------------------------------------------
    # Optional filtering
    # ------------------------------------------------------------
    keep = np.isfinite(data["x raw"].to_numpy(dtype=float))
    keep &= np.isfinite(data[metric_col].to_numpy(dtype=float))

    warnings_col = _resolve_df_column(
        data,
        "Warning Details",
        aliases=["Warnings", "warnings", "warning details"],
        required=False,
    )
    if options.get("exclude warnings", False) and warnings_col is not None:
        keep &= (
            data[warnings_col]
            .fillna("")
            .astype(str)
            .str.strip()
            .eq("")
            .to_numpy()
        )

    r2_col = _resolve_df_column(
        data,
        "R2",
        aliases=["r2", "R^2"],
        required=False,
    )
    if options.get("exclude low r2", False) and r2_col is not None:
        keep &= (
            pd.to_numeric(data[r2_col], errors="coerce")
            >= options.get("min r2", 0.95)
        ).to_numpy()

    data = data.loc[keep].reset_index(drop=True)

    if len(data) < 2:
        raise ValueError("fit_rate requires at least two valid points after filtering.")

    y_adjustment = _apply_y_mode(
        data[metric_col].to_numpy(dtype=float),
        options,
        series_keys=[metric_col, "y", "default"],
    )
    data["y raw"] = y_adjustment["raw"]
    data["y adjusted"] = y_adjustment["adjusted"]
    data["y0"] = y_adjustment["y0"]
    data["y mode"] = y_adjustment["mode"]

    fitline = None
    fit_rows = []
    fit_model_results = {}

    plot_loglog = (
        options.get("plot log-log", False)
        or options.get("log log plot", False)  # backward-compatible alias
        or str(options.get("transform mode", "")).strip().lower().replace("_", "-").replace(" ", "-")
        in ("log-log", "loglog")
    )

    # ------------------------------------------------------------
    # Local slopes / apparent local orders
    # ------------------------------------------------------------
    local_slopes = _local_log_slopes(
        data["x raw"].to_numpy(dtype=float),
        data["y adjusted"].to_numpy(dtype=float),
        x_label=x_label,
        y_label=metric_col,
        mode=options.get("local slope mode", "adjacent"),
    )
    data.attrs["local slopes"] = local_slopes

    if options.get("print local slopes", False) and len(local_slopes) > 0:
        print("\nLocal log-log slopes / apparent orders:")
        print(local_slopes.to_string(index=False))

    if options.get("plot local slopes", False) and len(local_slopes) > 0:
        plt.figure()

        x_slope_col = (
            f"{x_label} midpoint"
            if f"{x_label} midpoint" in local_slopes.columns
            else x_label
        )

        plt.scatter(local_slopes[x_slope_col], local_slopes["local order"])
        plt.xscale("log")
        plt.axhline(0, linestyle="--", linewidth=1, label="Zero-order")
        plt.axhline(1, linestyle="--", linewidth=1, label="First-order")
        plt.axhline(2, linestyle="--", linewidth=1, label="Second-order")

        plt.xlabel(
            _format_fit_rate_x_label(
                x_label,
                unit=x_unit,
                x_kind=x_kind,
            )
        )
        plt.ylabel("Local apparent order")
        if _scatterfit_legend_requested(options):
            plt.legend(fontsize=_scatterfit_legend_fontsize(options))

    # ------------------------------------------------------------
    # Log-log mode: only make log-log plot
    # ------------------------------------------------------------
    if plot_loglog:
        x_transform, y_transform, mode_label = _resolve_xy_transforms(
            options,
            default_x="log10",
            default_y="log10",
        )

        transformed = _apply_scatter_transforms(
            data["x raw"].to_numpy(dtype=float),
            data["y adjusted"].to_numpy(dtype=float),
            x_transform,
            y_transform,
            options,
        )

        dropped = transformed["dropped"]

        if dropped > 0 and do_print:
            print(
                f"plot log-log: excluded {dropped} non-positive or non-finite "
                f"point(s) before transforming."
            )

        data = data.loc[transformed["keep"]].reset_index(drop=True)

        if len(data) < 2:
            raise ValueError(
                "plot log-log requires at least two finite transformable x/y points "
                "after filtering."
            )

        plot_x = transformed["x"]
        plot_y = transformed["y"]

        data["x transformed"] = plot_x
        data["y transformed"] = plot_y
        if transformed["x input changed"]:
            data["x transform input"] = transformed["x input"]
        if transformed["y input changed"]:
            data["y transform input"] = transformed["y input"]
        data["x transform"] = transformed["x label"] or "identity"
        data["y transform"] = transformed["y label"] or "identity"
        if np.any(np.asarray(transformed["x note"], dtype=object) != ""):
            data["x transform note"] = transformed["x note"]
        data["y transform note"] = transformed["y note"]

        fit_indices = options.get("log fit indices")
        if fit_indices is None:
            fit_indices = options.get("fit indices")

        fitline = {}
        fit_specs = _fit_rate_range_specs(
            options,
            fallback_fit_indices=fit_indices,
            default_label=metric_col,
        )
        if do_fit:
            for fit_label, fit_spec, is_fit_range in fit_specs:
                fit_x_sel, fit_y_sel = _fit_rate_selected_points(
                    plot_x,
                    plot_y,
                    fit_spec,
                    is_fit_range,
                )

                if len(fit_x_sel) < 2:
                    raise ValueError("At least two points are required for the log-log fit.")

                series_fit = _fit_series_xy(
                    fit_x_sel,
                    fit_y_sel,
                    options=options,
                    label=fit_label,
                )
                fitline[fit_label] = series_fit["fits"]
                fit_model_results[fit_label] = series_fit["model_result"]
                fit_rows.extend(series_fit["fit_rows"])

            if len(fitline) == 1 and options.get("fit ranges") is None:
                fitline = next(iter(fitline.values()))

        if do_plot:
            plt.figure()
            point_color = _artist_color(plt.scatter(
                plot_x,
                plot_y,
                label=_format_fit_rate_metric_label(
                    metric_col,
                    log=transformed["y label"] == "log10",
                    unit=metric_unit,
                ),
            ))

            if do_fit:
                for fit_label, fit_spec, is_fit_range in fit_specs:
                    fit_x_sel, fit_y_sel = _fit_rate_selected_points(
                        plot_x,
                        plot_y,
                        fit_spec,
                        is_fit_range,
                    )
                    series_fit = _fit_series_xy(
                        fit_x_sel,
                        fit_y_sel,
                        options=options,
                        label=fit_label,
                    )
                    plot_options = _fit_rate_fit_options_for_range(options, fit_label, is_fit_range)
                    plot_options = _options_with_default_fit_color(
                        plot_options,
                        raw_options,
                        point_color,
                        index=fit_color_index,
                    )
                    fit_color_index += 1
                    plot_options.update({"new plot": False, "plot data": False, "model label": f"{fit_label} Fit"})
                    _plot_fit_model_result(series_fit["model_result"], plot_options)

            plt.xlabel(
                _format_fit_rate_x_label(
                    x_label,
                    unit=x_unit,
                    x_kind=x_kind,
                    transform=transformed["x label"],
                    log=transformed["x label"] == "log10",
                )
            )
            if _normalize_y_mode(options.get("y mode")) == "raw":
                y_axis_label = _format_fit_rate_metric_label(
                    metric_col,
                    log=transformed["y label"] == "log10",
                    unit=metric_unit,
                )
            else:
                y_axis_label = _format_y_mode_axis_label(
                    _format_fit_rate_metric_label(metric_col),
                    options.get("y mode"),
                )
                y_axis_label = _format_y_transform_axis_label(y_axis_label, y_transform)
            plt.ylabel(y_axis_label)
            if _scatterfit_legend_requested(options):
                plt.legend(fontsize=_scatterfit_legend_fontsize(options))
            _apply_matplotlib_axis_scales(plt.gca(), options)

        if do_print:
            y_notes = np.asarray(transformed["y note"], dtype=object)
            if dropped > 0 or np.any(y_notes != ""):
                print("Rate Fit Summary:")
                print(f"  Mode: {mode_label}")
                print(f"  Metric: {metric_col}")
                print(f"  X: {x_label}")
                print(f"  Excluded from transform: {dropped}")
            if np.any(y_notes != ""):
                print(f"  Y transform note: {'; '.join(str(note) for note in np.unique(y_notes[y_notes != '']))}")

            if do_fit:
                _print_fit_rate_model_results(fit_model_results, options)

        _attach_scatter_fit_table(data, fit_rows)
        data.attrs["fit model results"] = fit_model_results
        return data, fitline

    # ------------------------------------------------------------
    # Normal mode: transformed x vs metric
    # ------------------------------------------------------------
    x_transform, y_transform, mode_label = _resolve_xy_transforms(
        options,
        default_x="identity",
        default_y="identity",
    )

    transformed = _apply_scatter_transforms(
        data["x raw"].to_numpy(dtype=float),
        data["y adjusted"].to_numpy(dtype=float),
        x_transform,
        y_transform,
        options,
    )

    if transformed["dropped"] > 0 and do_print:
        print(
            f"fit_rate: excluded {transformed['dropped']} non-finite or "
            "non-transformable point(s)."
        )

    data = data.loc[transformed["keep"]].reset_index(drop=True)

    if len(data) < 2:
        raise ValueError("fit_rate requires at least two transformable points.")

    plot_x = transformed["x"]
    plot_y = transformed["y"]
    data["x transformed"] = plot_x
    data["y transformed"] = plot_y
    if transformed["x input changed"]:
        data["x transform input"] = transformed["x input"]
    if transformed["y input changed"]:
        data["y transform input"] = transformed["y input"]
    data["x transform"] = transformed["x label"] or "identity"
    data["y transform"] = transformed["y label"] or "identity"
    if np.any(np.asarray(transformed["x note"], dtype=object) != ""):
        data["x transform note"] = transformed["x note"]
    data["y transform note"] = transformed["y note"]

    fitline = {}
    fit_specs = _fit_rate_range_specs(
        options,
        fallback_fit_indices=options.get("fit indices"),
        default_label=metric_col,
    )
    if do_fit:
        for fit_label, fit_spec, is_fit_range in fit_specs:
            fit_x_sel, fit_y_sel = _fit_rate_selected_points(
                plot_x,
                plot_y,
                fit_spec,
                is_fit_range,
            )

            if len(fit_x_sel) < 2:
                raise ValueError("At least two points are required for the fit.")

            series_fit = _fit_series_xy(
                fit_x_sel,
                fit_y_sel,
                options=options,
                label=fit_label,
            )
            fitline[fit_label] = series_fit["fits"]
            fit_model_results[fit_label] = series_fit["model_result"]
            fit_rows.extend(series_fit["fit_rows"])

        if len(fitline) == 1 and options.get("fit ranges") is None:
            fitline = next(iter(fitline.values()))

    if do_plot:
        plt.figure()
        point_color = _artist_color(plt.scatter(
            plot_x,
            plot_y,
            label=_format_fit_rate_metric_label(metric_col, unit=metric_unit),
        ))

        if do_fit:
            for fit_label, fit_spec, is_fit_range in fit_specs:
                fit_x_sel, fit_y_sel = _fit_rate_selected_points(
                    plot_x,
                    plot_y,
                    fit_spec,
                    is_fit_range,
                )
                series_fit = _fit_series_xy(
                    fit_x_sel,
                    fit_y_sel,
                    options=options,
                    label=fit_label,
                )
                plot_options = _fit_rate_fit_options_for_range(options, fit_label, is_fit_range)
                plot_options = _options_with_default_fit_color(
                    plot_options,
                    raw_options,
                    point_color,
                    index=fit_color_index,
                )
                fit_color_index += 1
                plot_options.update({"new plot": False, "plot data": False, "model label": f"{fit_label} Fit"})
                _plot_fit_model_result(series_fit["model_result"], plot_options)

        plt.xlabel(
            _format_fit_rate_x_label(
                x_label,
                unit=x_unit,
                x_kind=x_kind,
                transform=transformed["x label"],
                log=transformed["x label"] == "log10",
            )
        )
        if _normalize_y_mode(options.get("y mode")) == "raw":
            y_axis_label = _format_fit_rate_metric_label(
                metric_col,
                log=transformed["y label"] == "log10",
                unit=metric_unit,
            )
        else:
            y_axis_label = _format_y_mode_axis_label(
                _format_fit_rate_metric_label(metric_col),
                options.get("y mode"),
            )
            y_axis_label = _format_y_transform_axis_label(y_axis_label, y_transform)
        plt.ylabel(y_axis_label)
        if _scatterfit_legend_requested(options):
            plt.legend(fontsize=_scatterfit_legend_fontsize(options))
        _apply_matplotlib_axis_scales(plt.gca(), options)

    if do_print:
        y_notes = np.asarray(transformed["y note"], dtype=object)
        if np.any(y_notes != ""):
            print("Rate Fit Summary:")
            print(f"  Mode: {mode_label or 'identity'}")
            print(f"  Metric: {metric_col}")
            print(f"  X: {x_label}")
            print(f"  Y transform note: {'; '.join(str(note) for note in np.unique(y_notes[y_notes != '']))}")

        if do_fit:
            _print_fit_rate_model_results(fit_model_results, options)

    _attach_scatter_fit_table(data, fit_rows)
    data.attrs["fit model results"] = fit_model_results
    return data, fitline


def fit_rate(df, options={}):
    """Fit a rate or FOWA result table and return a scatter-fit result object.
    
    Parameters
    ----------
    df : pandas.DataFrame
        Table containing rate, transformed x, or metric columns.
    options : dict or FitRateOptions, optional
        Transform, fit-window, print, and plot options. See ``e.describe_options("fit_rate")``.
    
    Returns
    -------
    ScatterFitResult
        Fit result with table, fit table, figure, and axes metadata.
    
    Examples
    --------
    >>> result = e.fit_rate(fowa_df, {"transform mode": "log-log"})
    """
    return _scatter_result_from_legacy(
        _fit_rate_legacy_return(df, options),
        summary={"analysis": "rate fit"},
    )


def _tafel_options_from_mapping(options):
    if isinstance(options, TafelAnalysisOptions):
        return options.to_legacy_dict()
    if options is None:
        options = {}
    if not isinstance(options, dict):
        raise TypeError(
            "tafel_analysis options must be a dict or TafelAnalysisOptions, "
            f"not {type(options).__name__}."
        )
    tafel_keys = {"overpotential range", "overpotential_range", "color"}
    return TafelAnalysisOptions.from_options(
        {key: value for key, value in options.items() if key in tafel_keys}
    ).to_legacy_dict()


def _coerce_tafel_cv_list(cv_or_list):
    if isinstance(cv_or_list, (list, tuple)):
        if len(cv_or_list) == 0:
            raise ValueError("tafel_analysis requires at least one CV-like object.")
        return list(cv_or_list)
    return [cv_or_list]


def _coerce_tafel_tof_values(TOF_max, count):
    if isinstance(TOF_max, (list, tuple, np.ndarray, pd.Series)):
        values = list(TOF_max)
        if len(values) != count:
            raise ValueError(
                "'TOF_max' must be a scalar or contain one value per CV "
                f"({count} expected, {len(values)} received)."
            )
        return [float(value) for value in values]
    return [float(TOF_max)] * count


def _tafel_curve(overpotential, tof_max, thermodynamic_potential, redox_potential, temperature):
    f = F / R / float(temperature)
    exponent = f * (float(thermodynamic_potential) - float(redox_potential) - overpotential)
    tof = 2 * float(tof_max) / (1 + np.exp(exponent))
    return tof, np.log10(tof)


def tafel_analysis(cv, TOF_max, thermodynamic_potential, redox_potential, options={}):
    """Compute and plot Tafel-style turnover-frequency data for one or more CVs.
    
    Parameters
    ----------
    cv : cv or sequence of cv
        Catalytic CV object, or CV objects, used for temperature and plot labeling.
    TOF_max : float or sequence of float
        Maximum turnover frequency value, or one value per CV.
    thermodynamic_potential : float
        Thermodynamic potential for overpotential calculation.
    redox_potential : float
        Catalyst redox potential.
    options : dict or TafelAnalysisOptions, optional
        Print, plot, and display options. See ``e.describe_options("tafel_analysis")``.
    
    Returns
    -------
    dict
        Dictionary containing ``data``, ``summary``, and ``axes``.
    
    Examples
    --------
    >>> result = e.tafel_analysis(cvs, TOF_values, E_thermo, E_redox)
    """
    raw_options = {} if options is None else options
    raw_mapping = raw_options.to_legacy_dict() if isinstance(raw_options, TafelAnalysisOptions) else raw_options
    tafel_options = _tafel_options_from_mapping(raw_options)
    cvs = _coerce_tafel_cv_list(cv)
    tof_values = _coerce_tafel_tof_values(TOF_max, len(cvs))

    start, end = tafel_options["overpotential range"]
    overpotential = np.linspace(float(start), float(end), 1000)

    if len(cvs) > 1:
        plot_options = _multiplot_options_from_mapping(raw_mapping)
        style = _prepare_multiplot_style(cvs, plot_options)
        ax = style["ax"]
        color_spec = style["color spec"]
        plot_labels = color_spec["plot labels"]
        line_colors = color_spec["line colors"]
        display_labels = style["display labels"]
    else:
        fig, ax = plt.subplots()
        plot_labels = [None]
        line_colors = [tafel_options["color"]]
        display_labels = [getattr(cvs[0], "name", "CV")]
        plot_options = _multiplot_options_from_mapping(raw_mapping)
        style = None

    data_rows = []
    summary_rows = []

    for i, (cv_obj, tof_max) in enumerate(zip(cvs, tof_values)):
        temperature = getattr(cv_obj, "temperature", 298)
        tof, log_tof = _tafel_curve(
            overpotential,
            tof_max,
            thermodynamic_potential,
            redox_potential,
            temperature,
        )
        label = display_labels[i]
        ax.plot(
            overpotential,
            log_tof,
            color=line_colors[i],
            label=plot_labels[i],
        )

        summary_rows.append({
            "Index": i,
            "Label": label,
            "TOFmax": tof_max,
            "Temperature": temperature,
            "Thermodynamic Potential": thermodynamic_potential,
            "Redox Potential": redox_potential,
        })
        data_rows.extend(
            {
                "Index": i,
                "Label": label,
                "Overpotential": eta,
                "TOF": tof_value,
                "log10 TOF": log_value,
                "TOFmax": tof_max,
                "Temperature": temperature,
                "Thermodynamic Potential": thermodynamic_potential,
                "Redox Potential": redox_potential,
            }
            for eta, tof_value, log_value in zip(overpotential, tof, log_tof)
        )

    ax.set_xlabel(r"$\eta$ (V)")
    ax.set_ylabel(r"$\log_{10}(\mathrm{TOF}\ (s^{-1}))$")

    if style is not None:
        _finish_multiplot_style(cvs, plot_options, style)

    data = pd.DataFrame(data_rows)
    summary = pd.DataFrame(summary_rows)
    data.attrs["summary"] = summary
    return {"data": data, "summary": summary, "axes": ax}




__all__ = [
    "fowa",
    "sevcik_analysis",
    "trumpet_analysis",
    "nicholson_analysis",
    "tafel_analysis",
    "fit_model",
    "fit_rate",
    "plateau_current",
    "fit_peak_potential",
    "fit_peak_current",
]
