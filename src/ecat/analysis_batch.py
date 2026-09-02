"""Batch and advanced electrochemical analysis helpers."""

import ast
import inspect
from pprint import pformat
import warnings

from scipy.optimize import OptimizeWarning

from .utils import *  # noqa: F401,F403
from .options import *  # noqa: F401,F403
from .options import _project_options
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
    _fit_line_x_values,
    _fit_table_from_fits,
    _multiplot_options_from_mapping,
    _plot_fit_bands,
    _plot_multi_scatter_trace,
    _prepare_multiplot_style,
    _can_rich_table_display,
    _conditional_analysis_name_column,
    _display_table,
    _pretty_table_header_html_label,
    _print_scatter_fit_statistics,
    _resolve_multiplot_labels_title_subtitle,
    _scatter_fit,
    _scatter_fit_row,
    _scatter_fit_table,
    _scatter_result_from_payload,
    _scatterfit_legend_fontsize,
    _scatterfit_legend_requested,
    build_object_table,
    display_object_table,
    echem_similar_different,
    multiplot,
    pretty_table_column_label,
)
from .reference import midpoint_potential
from ._cv_direction import resolve_cv_segment_pair_branches
from .results import AnalysisResult, analysis_result_from_table


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


def _rich_table_output_enabled(options):
    options = {} if options is None else options
    return bool(options.get("pretty print", True)) and display is not None and _can_rich_table_display()


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
    "equation",
    "fit equation",
    "fit equation label",
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
    "init": "fit init",
    "bounds": "fit bounds",
    "residual": "fit residual",
    "band": "fit band",
    "band level": "fit band level",
    "band_level": "fit band level",
    "max evals": "fit max evals",
    "max_evals": "fit max evals",
    "maxfev": "fit max evals",
    "indices": "fit indices",
    "method": "fit method",
    "sigma": "fit sigma",
    "absolute sigma": "fit absolute sigma",
    "absolute_sigma": "fit absolute sigma",
    "check finite": "fit check finite",
    "check_finite": "fit check finite",
    "nan policy": "fit nan policy",
    "nan_policy": "fit nan policy",
    "jac": "fit jac",
}


def _normalize_shared_fit_options(options=None, *, allow_bare_aliases=False):
    normalized = _normalize_option_mapping(options or {})
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


def _normalize_curve_fit_method(value):
    if value in (None, False):
        return None
    text = str(value).strip().lower().replace("_", " ").replace("-", " ")
    text = re.sub(r"\s+", " ", text)
    if text in {"", "auto", "none", "default"}:
        return None
    aliases = {
        "levenberg marquardt": "lm",
        "levenberg-marquardt": "lm",
        "lm": "lm",
        "trust region reflective": "trf",
        "trust-region reflective": "trf",
        "trf": "trf",
        "dog box": "dogbox",
        "dogbox": "dogbox",
    }
    method = aliases.get(text)
    if method is None:
        raise ValueError("'fit method' must be 'auto', 'lm', 'trf', or 'dogbox'.")
    return method


def _normalize_curve_fit_options(options, default_sigma=None):
    options = {} if options is None else dict(options)
    raw_passthrough = options.get("curve fit options", None)
    if raw_passthrough in (None, False):
        curve_options = {}
    elif isinstance(raw_passthrough, dict):
        curve_options = dict(raw_passthrough)
    else:
        raise ValueError("'curve fit options' must be a dictionary of scipy.optimize.curve_fit keyword arguments.")

    reserved = {"p0", "bounds", "full_output"}
    reserved_present = sorted(reserved & set(curve_options))
    if reserved_present:
        pretty = "', '".join(reserved_present)
        raise ValueError(
            f"'curve fit options' cannot set '{pretty}'. Use eCAT 'fit init'/'fit bounds' "
            "for p0 and bounds; full_output is managed internally."
        )

    if "fit method" in options:
        method = _normalize_curve_fit_method(options.get("fit method"))
        if method is not None:
            curve_options["method"] = method
    elif "method" in curve_options:
        curve_options["method"] = _normalize_curve_fit_method(curve_options["method"])

    sigma = options.get("fit sigma", None)
    if sigma is not None:
        curve_options["sigma"] = sigma
    elif default_sigma is not None and "sigma" not in curve_options:
        curve_options["sigma"] = default_sigma

    if "fit absolute sigma" in options:
        curve_options["absolute_sigma"] = bool(options.get("fit absolute sigma"))
    if options.get("fit check finite", None) is not None:
        curve_options["check_finite"] = bool(options.get("fit check finite"))
    if options.get("fit nan policy", None) is not None:
        nan_policy = options.get("fit nan policy")
        if nan_policy in (False, "none", "None"):
            nan_policy = None
        if nan_policy is not None:
            curve_options["nan_policy"] = str(nan_policy).strip().lower().replace("_", " ")
    if options.get("fit jac", None) is not None:
        curve_options["jac"] = options.get("fit jac")

    if "maxfev" not in curve_options and "max_nfev" not in curve_options:
        curve_options["maxfev"] = int(options.get("fit max evals", 10000))

    return curve_options


def _curve_fit_method_display(curve_options):
    method = (curve_options or {}).get("method")
    if method in (None, "", "auto"):
        return "curve_fit / auto"
    return f"curve_fit / {method}"


def _curve_fit_options_are_nondefault(options, curve_options):
    options = {} if options is None else dict(options)
    if options.get("curve fit options") not in (None, False, {}):
        return True
    for key in (
        "fit method",
        "fit sigma",
        "fit absolute sigma",
        "fit check finite",
        "fit nan policy",
        "fit jac",
    ):
        value = options.get(key, None)
        if key == "fit method":
            if _normalize_curve_fit_method(value) is not None:
                return True
        elif value is not None and value is not False:
            return True
    # fit max evals is intentionally not enough to display a method row; it is
    # a budget, not a different fitting method.
    return False


def _warn_if_fit_model_is_underdetermined(fit_points, parameter_names):
    n_points = int(fit_points)
    n_params = len(parameter_names)
    if n_points > n_params:
        return
    if n_points < n_params:
        message = (
            f"Fit has {n_points} points but {n_params} fitted parameters "
            f"({', '.join(parameter_names)}); the model is underdetermined."
        )
    else:
        message = (
            f"Fit has {n_points} points and {n_params} fitted parameters "
            f"({', '.join(parameter_names)}); fit statistics and parameter "
            "uncertainty are underdetermined."
        )
    warnings.warn(message, UserWarning, stacklevel=3)


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
            "equation": getattr(model, "__name__", "custom callable"),
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
            "equation": str(model).strip(),
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


def _format_fit_equation_for_mathtext(equation):
    def scientific_repl(match):
        mantissa = match.group("mantissa")
        exponent = int(match.group("exponent"))
        return rf"{mantissa}\times 10^{{{exponent}}}"

    text = re.sub(
        r"(?P<mantissa>[+-]?(?:\d+(?:\.\d*)?|\.\d+))[eE](?P<exponent>[+-]?\d+)",
        scientific_repl,
        str(equation),
    )
    return re.sub(
        r"\^(?!\{)(?P<exponent>[+-]?(?:\d+(?:\.\d*)?|\.\d+))",
        lambda match: "^{" + match.group("exponent") + "}",
        text,
    )


def _fit_residual_r2_label(residual, *, mathtext=False):
    residual = str(residual or "direct").strip().lower().replace("_", " ")
    if residual == "log":
        return r"\ln R^2" if mathtext else "ln R²"
    if residual == "log10":
        return r"\log R^2" if mathtext else "log R²"
    return "R^2" if mathtext else "R²"


def _coefficient_of_determination(observed, predicted):
    observed = np.asarray(observed, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    keep = np.isfinite(observed) & np.isfinite(predicted)
    observed = observed[keep]
    predicted = predicted[keep]
    if len(observed) == 0:
        return np.nan
    ss_res = float(np.nansum((observed - predicted) ** 2))
    ss_tot = float(np.nansum((observed - np.nanmean(observed)) ** 2))
    return np.nan if ss_tot == 0 else 1 - ss_res / ss_tot


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


def _normalize_bound_value(value, *, upper):
    if value is None:
        return np.inf if upper else -np.inf
    return value


def _normalize_bound_values(values, *, upper):
    return [_normalize_bound_value(value, upper=upper) for value in values]


def _is_bound_scalar(value):
    return value is None or np.isscalar(value)


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
                lower[i] = _normalize_bound_value(lo, upper=False)
                upper[i] = _normalize_bound_value(hi, upper=True)
        return lower, upper
    if len(spec) != 2:
        raise ValueError("'fit bounds' must be [lower, upper] or a dict by parameter name.")
    lo_spec, hi_spec = spec
    if _is_bound_scalar(lo_spec):
        lower = [_normalize_bound_value(lo_spec, upper=False)] * len(names)
    else:
        lower = _normalize_bound_values(lo_spec, upper=False)
    if _is_bound_scalar(hi_spec):
        upper = [_normalize_bound_value(hi_spec, upper=True)] * len(names)
    else:
        upper = _normalize_bound_values(hi_spec, upper=True)
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
            start, stop = fit_indices_array
            start_i = _coerce_fit_index(start)
            stop_i = _coerce_fit_index(stop)
            mask[np.arange(length)[start_i:stop_i]] = True
            return mask
        mask[np.asarray([_coerce_fit_index(value) for value in fit_indices_array], dtype=int)] = True
        return mask

    if fit_indices_array.ndim == 2 and fit_indices_array.shape[1] == 2:
        base_indices = np.arange(length)
        for start, stop in fit_indices_array:
            start_i = _coerce_fit_index(start)
            stop_i = _coerce_fit_index(stop)
            mask[base_indices[start_i:stop_i]] = True
        return mask

    raise ValueError(
        "'fit indices' should be [start, stop], [[start, stop], ...], "
        "a boolean mask, or explicit integer indices."
    )


def _coerce_fit_index(value):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError("'fit indices' entries must be integers or None.")
    return int(value)


def _is_fit_index_range_pair(spec):
    if not isinstance(spec, (list, tuple)) or len(spec) != 2:
        return False
    start, stop = spec
    return (
        (start is None or (isinstance(start, (int, np.integer)) and not isinstance(start, (bool, np.bool_))))
        and (stop is None or (isinstance(stop, (int, np.integer)) and not isinstance(stop, (bool, np.bool_))))
    )


def _fit_model_selection_mask(x, options):
    options = {} if options is None else dict(options)
    mask = np.ones(len(x), dtype=bool)
    if options.get("fit indices") is not None:
        mask &= _fit_model_indices_mask(len(x), options.get("fit indices"))
    return mask


def _normalize_fit_index_ranges(fit_index_ranges):
    """Normalize the index-window-driven multi-fit option.

    Forms
    -----
    {"early": [0, 3], "tail": [[6, 9], [9, None]]}
        Named fits; each entry may contain a single window or multiple windows.

    [[0, 2], [4, 7]]
        Unnamed fits labeled "Fit 1", "Fit 2", ...

    [[[0, 2], [4, None]]]
        A single unnamed fit with disconnected windows, represented as a nested list.

    [0, 3]
        One unnamed fit using an index slice.
    """
    if fit_index_ranges in (None, {}):
        return []
    if isinstance(fit_index_ranges, slice):
        return [("Fit 1", fit_index_ranges)]

    if isinstance(fit_index_ranges, dict):
        return [(str(label), spec) for label, spec in fit_index_ranges.items()]

    if _is_fit_index_range_pair(fit_index_ranges):
        return [("Fit 1", fit_index_ranges)]

    if isinstance(fit_index_ranges, (list, tuple)):
        if len(fit_index_ranges) == 0:
            return [("Fit 1", None)]
        if all(isinstance(value, (bool, np.bool_)) for value in fit_index_ranges):
            return [("Fit 1", fit_index_ranges)]
        if (
            len(fit_index_ranges) == 1
            and isinstance(fit_index_ranges[0], (list, tuple))
            and all(_is_fit_index_range_pair(item) for item in fit_index_ranges[0])
        ):
            return [("Fit 1", fit_index_ranges[0])]
        if all(_is_fit_index_range_pair(item) for item in fit_index_ranges):
            return [(f"Fit {i + 1}", item) for i, item in enumerate(fit_index_ranges)]
        if all(isinstance(value, (int, np.integer)) for value in fit_index_ranges):
            return [("Fit 1", fit_index_ranges)]

    raise ValueError(
        "'fit indices' must be a dict of named index specs or an unnamed index spec."
    )


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
    _warn_if_fit_model_is_underdetermined(len(x_fit), names)
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

    p0_array = np.asarray(p0, dtype=float)
    curve_options = _normalize_curve_fit_options(options, default_sigma=sigma)
    if model == "custom":
        try:
            np.asarray(fit_func(x_fit, *p0_array), dtype=float)
        except Exception as exc:
            equation = model_spec.get("equation", options.get("fit model", "custom"))
            allowed = ", ".join(sorted(_FIT_MODEL_ALLOWED_FORMULA_FUNCTIONS))
            raise ValueError(
                "Could not evaluate custom fit model "
                f"{equation!r}. Use x, fitted parameter names, arithmetic "
                f"operators, and supported functions: {allowed}."
            ) from exc

    covariance_warning = False
    covariance_message = ""
    try:
        with warnings.catch_warnings(record=True) as fit_warnings:
            warnings.simplefilter("always", OptimizeWarning)
            popt, pcov = curve_fit(
                fit_func,
                x_fit,
                fit_target,
                p0=p0_array,
                bounds=(np.asarray(lower, dtype=float), np.asarray(upper, dtype=float)),
                **curve_options,
            )
        covariance_warning = any(
            issubclass(item.category, OptimizeWarning) for item in fit_warnings
        )
        if covariance_warning:
            covariance_message = (
                "Fit parameter covariance could not be estimated; standard errors "
                "are unavailable."
            )
    except KeyError as exc:
        if model == "custom":
            equation = model_spec.get("equation", options.get("fit model", "custom"))
            raise ValueError(
                f"Could not evaluate custom fit model {equation!r}."
            ) from exc
        raise

    predicted = func(x, *popt)
    predicted_fit = func(x_fit, *popt)
    residuals = y - predicted
    fit_residuals = y_fit - predicted_fit
    dof = int(len(x_fit) - len(popt))
    residual_variance = (
        float(np.nansum(fit_residuals ** 2) / dof)
        if dof > 0
        else np.nan
    )
    rmse = float(np.sqrt(np.nanmean(fit_residuals ** 2)))
    raw_r2 = _coefficient_of_determination(y_fit, predicted_fit)
    if residual == "log10":
        r2 = _coefficient_of_determination(np.log10(y_fit), np.log10(predicted_fit))
    elif residual == "log":
        r2 = _coefficient_of_determination(np.log(y_fit), np.log(predicted_fit))
    else:
        r2 = raw_r2
    stats = {
        "R2": r2,
        "R2 raw": raw_r2,
        "RMSE": rmse,
        "Fit Points": int(len(x_fit)),
        "residual space": residual,
    }

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
        "x fit": x_fit,
        "y fit": y_fit,
        "label": label,
        "keep": keep,
        "predicted": predicted,
        "residuals": residuals,
        "fit_rows": fit_rows,
        "popt": popt,
        "pcov": pcov,
        "dof": dof,
        "residual variance": residual_variance,
        "fit method": _curve_fit_method_display(curve_options),
        "show fit method": _curve_fit_options_are_nondefault(options, curve_options),
        "curve fit options": dict(curve_options),
        "covariance warning": covariance_warning,
        "covariance message": covariance_message,
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
    )


def _fit_model_input_xy(x_or_result, y=None, options=None):
    options = {} if options is None else dict(options)
    if y is not None:
        return np.asarray(x_or_result, dtype=float), np.asarray(y, dtype=float), "x", "y"

    table = x_or_result.table if isinstance(x_or_result, AnalysisResult) else x_or_result
    if not isinstance(table, pd.DataFrame):
        raise TypeError("fit_model accepts x/y arrays, a pandas DataFrame, or an AnalysisResult.")

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
    x_line = _fit_line_x_values(
        model_result.get("x fit", x),
        options,
        label=model_result.get("label"),
        index=int(options.get("_fit line index", 0) or 0),
        points=300,
    )
    y_line = model_result["function"](x_line, *model_result["popt"])
    fit_color = _fit_color_from_options(
        options,
        index=0,
        fallback=_artist_color(data_artist) or "tab:red",
    )
    _plot_fit_bands(ax, x_line, y_line, model_result, options, color=fit_color)
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
            equation = model_result.get("fit equation") or model_result["equation"]
            show_equation = True
            if (
                model_result.get("model") == "custom"
                and "x" not in str(model_result.get("equation", ""))
            ):
                show_equation = False
            if show_equation:
                equation = _format_fit_equation_for_mathtext(equation)
            r2_label = _fit_residual_r2_label(
                model_result.get("residual", "direct"),
                mathtext=True,
            )
            r2_text = _format_fit_model_display_value(
                r2,
                sig_figs=model_result.get("sig figs"),
            )
            if show_equation:
                label = f"{label}\n${equation}$\n${r2_label} = {r2_text}$"
            else:
                label = f"{label}\n${r2_label} = {r2_text}$"
    ax.plot(
        x_line,
        y_line,
        label=label,
        color=fit_color,
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
        return format_sigfigs(float(value), sig_figs)
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


def _display_or_print_fit_model_table(table, title=None, options=None):
    return _display_table(table, options, title=title, index=False)


def _fit_model_parameter_summary(model_result):
    names = list(model_result.get("names", ()))
    return f"{', '.join(str(name) for name in names)} ({len(names)})"


def _fit_model_details_display_table(model_result):
    stats = model_result["stats"]
    sig_figs = _fit_model_sig_figs(model_result.get("sig figs"))
    r2_label = _fit_residual_r2_label(model_result.get("residual", "direct"))
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
        ("Fit Parameters", _fit_model_parameter_summary(model_result)),
        ("Residual", model_result.get("residual", "")),
        ("X Range", x_range),
        ("Fit Points", stats.get("Fit Points")),
        (r2_label, stats.get("R2")),
        ("RMSE", stats.get("RMSE")),
    ]
    if model_result.get("show fit method", False):
        rows.insert(3, ("Fit Method", model_result.get("fit method", "curve_fit / auto")))
    if model_result.get("covariance warning", False):
        rows.append(("Fit Warning", model_result.get("covariance message", "")))
    return pd.DataFrame(
        [
            {"Setting": setting, "Value": _format_fit_model_display_value(value, sig_figs=sig_figs)}
            for setting, value in rows
        ]
    )


def _fit_model_display_table(model_result):
    stats = model_result["stats"]
    sig_figs = _fit_model_sig_figs(model_result.get("sig figs"))
    r2_label = _fit_residual_r2_label(model_result.get("residual", "direct"))
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
        ("Fit Parameters", _fit_model_parameter_summary(model_result)),
        ("Residual", model_result.get("residual", "")),
        ("X Range", x_range),
        ("Fit Points", stats.get("Fit Points")),
        (r2_label, stats.get("R2")),
        ("RMSE", stats.get("RMSE")),
    ]
    if model_result.get("show fit method", False):
        rows.insert(3, ("Fit Method", model_result.get("fit method", "curve_fit / auto")))
    if model_result.get("covariance warning", False):
        rows.append(("Fit Warning", model_result.get("covariance message", "")))
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
        "fit_parameters": list(model_result.get("names", ())),
        "fit_parameter_count": len(model_result.get("names", ())),
        "fit_method": model_result.get("fit method"),
        "covariance_warning": model_result.get("covariance warning", False),
        "covariance_message": model_result.get("covariance message", ""),
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
    if mode in {"detail", "details", "detailed", "full", "two table", "two tables"}:
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
            _display_or_print_fit_model_table(
                _fit_model_details_display_table(model_result),
                title="Fit Model Details",
                options=options,
            )
            _display_or_print_fit_model_table(
                _fit_model_parameters_display_table(model_result),
                title="Fit Model Parameters",
                options=options,
            )
        else:
            _display_or_print_fit_model_table(
                _fit_model_display_table(model_result),
                title="Fit Model",
                options=options,
            )
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
        _display_or_print_fit_model_table(
            _combine_fit_model_tables(
                fit_model_results,
                _fit_model_details_display_table,
                "Setting",
            ),
            title="Fit Model Details",
            options=options,
        )
        _display_or_print_fit_model_table(
            _fit_model_multi_parameters_display_table(fit_model_results),
            title="Fit Model Parameters",
            options=options,
        )
        return

    _display_or_print_fit_model_table(
        _combine_fit_model_tables(
            fit_model_results,
            _fit_model_display_table,
            "Field",
        ),
        title="Fit Model",
        options=options,
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
    index_range_specs = options.get("fit indices")
    use_index_range_specs = index_range_specs is not None

    if use_index_range_specs:
        normalized_index_ranges = _normalize_fit_index_ranges(index_range_specs)
        base_label = options.get("series", options.get("model label", "Model"))
        default_series_label = base_label if len(normalized_index_ranges) == 1 else None
        base_new_plot = options.get("new plot", True)
        base_plot = dict(options)
        table = pd.DataFrame({x_label: x, y_label: y_values})
        model_results = {}
        fit_rows = []

        for fit_index, (series_label, fit_spec) in enumerate(normalized_index_ranges):
            fit_label = default_series_label if default_series_label else series_label
            fit_options = dict(fit_options)
            fit_options["_fit model mask"] = _fit_model_indices_mask(len(x), fit_spec)
            model_result = _fit_model_xy(
                x,
                y_values,
                model=fit_model_name,
                options=fit_options,
                label=fit_label,
            )
            model_results[fit_label] = model_result
            fit_rows.extend(model_result["fit_rows"])

            if fit_index == 0:
                table["Predicted"] = model_result["predicted"]
                table["Residual"] = model_result["residuals"]

            if options.get("plot", False):
                current_axes = plt.gca()
                current_color = (
                    _artist_color(current_axes.collections[0]) if current_axes.collections else None
                )
                plot_options = _options_with_default_fit_color(
                    dict(base_plot),
                    options,
                    current_color,
                    index=fit_index,
                )
                plot_options.setdefault("x label", x_label)
                plot_options.setdefault("y label", y_label)
                if "model label" not in plot_options:
                    plot_options["model label"] = str(fit_label)
                if options.get("fit label") is None:
                    plot_options["fit label"] = False
                if fit_index == 0:
                    plot_options["new plot"] = base_new_plot
                    plot_options["plot data"] = True
                else:
                    plot_options["new plot"] = False
                    plot_options["plot data"] = False
                plot_options["model label"] = str(fit_label)
                plot_options["_fit line index"] = fit_index
                _plot_fit_model_result(model_result, plot_options)

        fit_table = _scatter_fit_table(fit_rows)
        table.attrs["fit table"] = fit_table

        if options.get("print", False):
            _print_fit_model_results(model_results, options)

        fits = {
            label: {
                "model": result["model"],
                "parameters": result["parameters"],
                "errors": result["errors"],
                "stats": result["stats"],
            }
            for label, result in model_results.items()
        }

        return ScatterFitResult(
            table=table,
            fits=fits,
            fit_table=fit_table,
            fit_model_results=model_results,
            summary={
                "analysis": "model fit",
                "model": fit_model_name,
                "equation": next(iter(model_results.values())).get("equation"),
                "fit equation": next(iter(model_results.values())).get("fit equation"),
                "models": {label: result["model"] for label, result in model_results.items()},
                "parameters": {
                    label: result["parameters"] for label, result in model_results.items()
                },
                "stats": {
                    label: result["stats"] for label, result in model_results.items()
                },
            },
        )

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
        plot_options.setdefault("_fit line index", 0)
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
    )

def _trumpet_analysis_payload(cvs, options=None):
    raw_options = options
    typed_options = TrumpetAnalysisOptions.from_options(options)
    options = typed_options.to_options_dict()
    if not options["plot"]:
        options["plot fit"] = False

    base_segment, paired_segment, segment_selection = _resolve_analysis_segment_pair(
        cvs,
        raw_options,
        options,
        analysis_name="trumpet_analysis",
    )
    options["segment"] = base_segment
    options["segments"] = [base_segment, paired_segment]

    if options.get("plot all", False):
        multiplot(cvs, _multiplot_options_from_mapping(options))

    half_wave_options = options.copy()
    half_wave_options["plot"] = False
    half_wave_options["print"] = False
    half_wave_options["plot all"] = False
    half_wave_options["print all"] = False
    half_wave_options["internal call"] = True
    half_wave_options["new plot"] = False
    half_wave_options["segments"] = [base_segment, paired_segment]
    potential_series = _resolve_complex_potential_series_map(
        raw_options,
        options,
        n_cvs=len(cvs),
        analysis_name="trumpet_analysis",
        option_names=["guess potential", "exact potential"],
        paired=True,
    )
    for key in potential_series:
        half_wave_options.pop(key, None)

    deltas, scan_rates, ep1_values, ep2_values = [], [], [], []
    for cv_index, cv in enumerate(cvs):
        cv_half_wave_options = _apply_resolved_potential_options(
            half_wave_options.copy(),
            potential_series,
            cv_index,
        )
        half_wave = cv.half_wave_potential(cv_half_wave_options)
        scan_rates.append(float(cv.scan_rate))
        deltas.append(half_wave["ΔE"])
        ep1_values.append(half_wave["peak 1"]["Ep"])
        ep2_values.append(half_wave["peak 2"]["Ep"])

    log_scan_rates = np.log10(np.asarray(scan_rates, dtype=float))
    ep1_values = np.asarray(ep1_values, dtype=float)
    ep2_values = np.asarray(ep2_values, dtype=float)

    fit_indices = options.get("fit indices")
    fit_x, fit_y1 = _select_fit_indices(log_scan_rates, ep1_values, fit_indices)
    _, fit_y2 = _select_fit_indices(log_scan_rates, ep2_values, fit_indices)

    branch_assignment, branch_diagnostics = resolve_cv_segment_pair_branches(
        cvs,
        base_segment,
        paired_segment,
        analysis_name="trumpet_analysis",
    )
    if branch_assignment is None:
        slope1 = float(np.polyfit(fit_x, fit_y1, 1)[0])
        slope2 = float(np.polyfit(fit_x, fit_y2, 1)[0])
        if np.isfinite(slope1) and slope1 < 0 and np.isfinite(slope2) and slope2 > 0:
            cathodic_segment, anodic_segment = base_segment, paired_segment
        elif np.isfinite(slope2) and slope2 < 0 and np.isfinite(slope1) and slope1 > 0:
            cathodic_segment, anodic_segment = paired_segment, base_segment
        else:
            raise ValueError(
                "trumpet_analysis could not assign cathodic and anodic branches from "
                "potential scan direction, and the fitted branch slopes were not one "
                f"negative and one positive (segment {base_segment}: {slope1:g} V/dec; "
                f"segment {paired_segment}: {slope2:g} V/dec). Pass a valid opposing "
                "segment pair and inspect the selected peak potentials."
            )
        branch_assignment = {
            "cathodic segment": int(cathodic_segment),
            "anodic segment": int(anodic_segment),
            "cathodic segments": [int(cathodic_segment)] * len(cvs),
            "anodic segments": [int(anodic_segment)] * len(cvs),
            "branch assignment source": "peak-potential fit slopes",
        }
        branch_diagnostics = {
            **branch_diagnostics,
            "source": "peak-potential fit slopes",
            "segment slopes / V per decade": {
                int(base_segment): slope1,
                int(paired_segment): slope2,
            },
        }

    cathodic_segment = branch_assignment["cathodic segment"]
    anodic_segment = branch_assignment["anodic segment"]
    cathodic_segments = branch_assignment["cathodic segments"]
    anodic_segments = branch_assignment["anodic segments"]
    cathodic_values = np.asarray([
        ep1_values[index] if segment == base_segment else ep2_values[index]
        for index, segment in enumerate(cathodic_segments)
    ])
    anodic_values = np.asarray([
        ep1_values[index] if segment == base_segment else ep2_values[index]
        for index, segment in enumerate(anodic_segments)
    ])
    fit_x, cathodic_fit_y = _select_fit_indices(log_scan_rates, cathodic_values, fit_indices)
    _, anodic_fit_y = _select_fit_indices(log_scan_rates, anodic_values, fit_indices)

    selection = dict(segment_selection or {})
    selection.update(branch_assignment)
    selection["branch assignment diagnostics"] = branch_diagnostics
    segment_selection = selection

    data = pd.DataFrame(
        {
            "Name": [getattr(cv_obj, "name", f"CV {index + 1}") for index, cv_obj in enumerate(cvs)],
            "Scan Rates (V/s)": scan_rates,
            "Log(Scan Rates (V/s))": log_scan_rates,
            "Cathodic Peak Potential (V)": cathodic_values,
            "Anodic Peak Potential (V)": anodic_values,
            "ΔE (V)": deltas,
        }
    )

    point_colors = [None, None]
    cathodic_segment_label = (
        f" (Seg {cathodic_segment})" if cathodic_segment is not None else ""
    )
    anodic_segment_label = (
        f" (Seg {anodic_segment})" if anodic_segment is not None else ""
    )
    if options["plot"]:
        plt.figure()
        plt.xlabel("log(Scan Rate) (log(V/s))")
        plt.ylabel("Peak Potential (V)")
        point_colors[0] = _artist_color(
            plt.scatter(
                log_scan_rates,
                cathodic_values,
                label=f"Cathodic{cathodic_segment_label} Ep",
            )
        )
        point_colors[1] = _artist_color(
            plt.scatter(
                log_scan_rates,
                anodic_values,
                label=f"Anodic{anodic_segment_label} Ep",
            )
        )

    fit_model_results = {}
    fits = []

    cathodic_label = f"Cathodic{cathodic_segment_label}"
    anodic_label = f"Anodic{anodic_segment_label}"
    cathodic_fit = _fit_series_xy(
        fit_x,
        cathodic_fit_y,
        options=options,
        label=cathodic_label,
        model="linear",
    )
    anodic_fit = _fit_series_xy(
        fit_x,
        anodic_fit_y,
        options=options,
        label=anodic_label,
        model="linear",
    )
    fit_model_results[cathodic_label] = cathodic_fit["model_result"]
    fit_model_results[anodic_label] = anodic_fit["model_result"]
    fits.append(np.asarray(cathodic_fit["model_result"]["popt"], dtype=float))
    fits.append(np.asarray(anodic_fit["model_result"]["popt"], dtype=float))

    if options["plot"] and options["plot fit"]:
        plot_options_1 = _options_with_default_fit_color(options, raw_options, point_colors[0], index=0)
        plot_options_1.update({"new plot": False, "plot data": False, "model label": f"{cathodic_label} Fit", "_fit line index": 0})
        _plot_fit_model_result(cathodic_fit["model_result"], plot_options_1)

        plot_options_2 = _options_with_default_fit_color(options, raw_options, point_colors[1], index=1)
        plot_options_2.update({"new plot": False, "plot data": False, "model label": f"{anodic_label} Fit", "_fit line index": 1})
        _plot_fit_model_result(anodic_fit["model_result"], plot_options_2)

    if options["plot"] and _scatterfit_legend_requested(options):
        plt.legend(fontsize=_scatterfit_legend_fontsize(options))

    cathodic_slope = float(fits[0][0])
    anodic_slope = float(fits[1][0])
    T = getattr(cvs[0], "temperature", None)
    if T is None:
        T = options.get("temperature", 298)
    T = float(T)
    if not np.isfinite(T) or T <= 0:
        raise ValueError("trumpet_analysis requires a positive temperature in K.")

    α = -R * T * np.log(10) / (2 * cathodic_slope * F)
    β = R * T * np.log(10) / (2 * anodic_slope * F)

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
        cathodic_slope=cathodic_slope,
        anodic_slope=anodic_slope,
        intercept_x=intercept_x,
        fit_x=fit_x,
    )

    trumpet_results = _trumpet_results_table(α, β, ks, D, warning, options)

    if options["print"]:
        _display_trumpet_parameter_table(
            _trumpet_parameter_table(
                cathodic_segments=cathodic_segments,
                anodic_segments=anodic_segments,
                branch_assignment_source=branch_assignment["branch assignment source"],
                temperature=T,
                diffusion_coefficient=D,
                fit_indices=options.get("fit indices"),
                options=options,
            ),
            options,
        )
        _display_trumpet_equations(resolved=True, compact=False, include_definitions=False)
        _display_trumpet_results_table(trumpet_results, options)
        _print_fit_model_results(fit_model_results, options)

    if options["print"] and options["print all"]:
        display_data = _conditional_analysis_name_column(
            data,
            ["Scan Rates (V/s)"],
            options,
        )
        _display_table(
            display_data,
            options,
            title="Trumpet Analysis Data",
            index=False,
        )

    data.attrs["fit model results"] = fit_model_results
    data.attrs["fit table"] = trumpet_results
    _attach_segment_selection_to_table(data, segment_selection)
    return data, fits, ks


def trumpet_analysis(cvs, options=None):
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
    payload = _trumpet_analysis_payload(cvs, options)
    return _scatter_result_from_payload(
        payload,
        summary=_summary_with_segment_selection({"analysis": "trumpet"}, payload),
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
    manual = options.get("scan rate", None)
    if manual is None:
        rates = [getattr(item, "scan_rate", None) for item in cvs]
    elif isinstance(manual, (list, tuple, np.ndarray, pd.Series)):
        rates = list(manual)
    else:
        rates = [manual] * len(cvs)
    if len(rates) != len(cvs):
        raise ValueError(
            f"'scan rate' for nicholson_analysis expected 1 scalar value or "
            f"{len(cvs)} scalar values (one per CV), but received {len(rates)} entries."
        )
    rates = [float(rate) if rate is not None else None for rate in rates]
    for rate in rates:
        if rate is None or not np.isfinite(rate) or rate <= 0:
            raise ValueError("Each CV must have a positive scan_rate, or provide 'scan rate'.")
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
        ("Fit Model", "", summary.get("fit model", "origin")),
        ("Diffusion Coefficient", "D", _nicholson_summary_scalar_text(summary.get("D / cm^2 s^-1"), "cm^2/s", sig_figs=sig_figs)),
        ("Electron Count", "n", _format_fit_model_display_value(summary.get("num electrons"), sig_figs=sig_figs)),
        ("Temperature", "T", _nicholson_summary_scalar_text(summary.get("temperature / K"), "K", sig_figs=sig_figs)),
        ("Psi Source", "ψ", summary.get("psi source")),
        (
            "Valid nΔEp Range",
            "nΔEp",
            f"{_format_fit_model_display_value(summary.get('nicholson delta ep min mv'), sig_figs=sig_figs)} to "
            f"{_format_fit_model_display_value(summary.get('nicholson delta ep max mv'), sig_figs=sig_figs)} mV",
        ),
    ]
    return _analysis_parameter_table(rows)


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
    return pd.DataFrame([{"Metric": key, "Value": value} for key, value in rows])


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
    scan_rate_columns = [
        column for column in display_df.columns
        if str(column).strip().lower().startswith("scan rate /")
    ]
    return _conditional_analysis_name_column(
        display_df,
        scan_rate_columns,
        options,
    )


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
        x_line = _fit_line_x_values(
            fit_result["x fit"],
            options,
            label="Nicholson",
            index=0,
            points=100,
        )
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
    parameter_table = _nicholson_parameter_display_table(summary, options)
    parameter_rich_table = _analysis_parameter_rich_table(parameter_table)
    if options.get("pretty print", True):
        _display_table(
            parameter_table,
            options,
            title="Nicholson Analysis Parameters",
            rich_table=parameter_rich_table,
            escape=None,
            index=False,
        )
        if not _rich_table_output_enabled(options):
            print("Nicholson Analysis Equations:")
        _display_analysis_equation(
            r"\text{Nicholson analysis equations:}",
            "Nicholson Analysis Equations",
            _nicholson_equation_bundle(summary),
            resolved=False,
            compact=False,
            include_definitions=False,
        )
        _display_table(
            _nicholson_summary_display_table(summary, options),
            options,
            title="Nicholson Analysis Summary",
            index=False,
        )
        if options.get("print all", False):
            display_object_table(display_data, options, title="Nicholson Analysis Data")
    else:
        print("Nicholson Analysis Parameters:")
        print(parameter_table.to_string(index=False, justify="left"))
        print("\nNicholson Analysis Equations:")
        equation = _nicholson_equation_bundle(summary)
        print("  " + equation["symbolic"])
        print("\nNicholson Analysis Summary:")
        print(_nicholson_summary_display_table(summary, options).to_string(index=False, justify="left"))
        if options.get("print all", False):
            print("\nNicholson Analysis Data:")
            print(display_data.to_string(index=False, justify="left"))
    excluded = data.loc[~data["included"].astype(bool), ["name", "exclusion reason"]]
    if not excluded.empty:
        print("Excluded points:")
        for _, row in excluded.iterrows():
            print(f"  {row['name']}: {row['exclusion reason']}")


def nicholson_analysis(cvs, options=None):
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
    options = NicholsonOptions.from_options(raw_options).to_options_dict()
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
    base_segment, paired_segment, segment_selection = _resolve_analysis_segment_pair(
        cv_list,
        raw_options,
        options,
        analysis_name="nicholson_analysis",
    )
    options["segment"] = base_segment
    options["segments"] = [base_segment, paired_segment]
    potential_series = _resolve_complex_potential_series_map(
        raw_options,
        options,
        n_cvs=len(cv_list),
        analysis_name="nicholson_analysis",
        option_names=["guess potential", "exact potential"],
        paired=True,
    )

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
    diagnostic_axis_options = _common_cv_plot_axis_options(cv_list, options)
    for cv_index, (cv_obj, scan_rate) in enumerate(zip(cv_list, scan_rates)):
        peak_options = dict(options)
        peak_options.update(diagnostic_axis_options)
        peak_options["plot"] = diagnostic_ax is not None
        peak_options["print"] = False
        peak_options["print all"] = False
        peak_options["internal call"] = True
        peak_options["new plot"] = False
        peak_options["plot cv"] = False
        peak_options["segments"] = [base_segment, paired_segment]
        _apply_resolved_potential_options(peak_options, potential_series, cv_index)
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
    if segment_selection:
        summary["segment selection"] = segment_selection

    if options.get("plot", True) or options.get("plot all", False):
        plot_options = dict(options)
        plot_options["new plot"] = True
        _plot_nicholson_analysis(data, fit_result, plot_options)
    if options.get("print", True):
        _print_nicholson_summary(data, summary, options)

    return AnalysisResult(
        {"data": data, "summary": summary},
        table=data,
        summary=summary,
        fits=fit_result,
        fit_table=pd.DataFrame([summary]),
        axes=plt.gca() if plt.get_fignums() else None,
        analysis="nicholson",
    )


def _sevcik_analysis_payload(cvs, options=None):
    raw_options = options
    typed_options = SevcikAnalysisOptions.from_options(options)
    options = typed_options.to_options_dict()
    do_plot = options.get("plot", True)
    do_print = options.get("print", True)

    segments, segment_selection = _resolve_auto_single_analysis_segment(
        cvs,
        raw_options,
        options,
        method_name="peak_potential",
        analysis_name="sevcik_analysis",
        default=1,
    )

    num_electrons = options.get("num electrons", 1)
    peaks, x_values = [], []
    peak_unit = _axis_common_unit(
        cvs,
        lambda cv: (cv.y(options), cv.y(options).name),
        options.get('y unit', 'auto')
    )
    diffusion_coefficients, fits = [], []
    fit_rows = []
    data = pd.DataFrame(
        {"Name": [getattr(cv_obj, "name", f"CV {index + 1}") for index, cv_obj in enumerate(cvs)]}
    )
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
        x_values = np.sqrt(np.asarray(x_values, dtype=float))
        data['Scan Rate^1/2 / (V s^-1)^1/2'] = x_values
    else:
        v = cvs[0].scan_rate
        diff_conc_idx = next((i for i in range(len(cvs[0].concentrations))
                              if cvs[0].concentrations[i] != cvs[1].concentrations[i]), -1)
        for cv in cvs:
            x_values.append(concentration_to_float(cv.concentrations[diff_conc_idx]))
        data['Concentration (M)'] = x_values

    internal_options = typed_options.for_peak_current().to_options_dict()
    internal_options["internal call"] = True
    internal_options["new plot"] = False
    internal_options["plot"] = options.get("plot all", False)
    internal_options["print"] = False
    internal_options["print all"] = False
    internal_options.update(_common_cv_plot_axis_options(cvs, options))
    internal_options.pop("segments", None)
    internal_options.pop("plot segment", None)
    internal_options.pop("plot segments", None)
    potential_series = _resolve_complex_potential_series_map(
        raw_options,
        options,
        n_cvs=len(cvs),
        analysis_name="sevcik_analysis",
        option_names=["guess potential", "exact potential", "tangent potential"],
    )
    for key in potential_series:
        internal_options.pop(key, None)

    if options["plot all"]:
        multiplot(cvs, _multiplot_options_from_mapping(options))

    for seg in segments:
        internal_options["segment"] = seg
        segment_peaks = []
        for cv_index, cv in enumerate(cvs):
            cv_peak_options = _apply_resolved_potential_options(
                internal_options.copy(),
                potential_series,
                cv_index,
            )
            y_name = cv.y(options).name
            y_unit = cv.units.get(y_name, '')
            peak_current = cv.peak_current(cv_peak_options)["ip"]
            scaled_peak_current, _ = scale_value(peak_current, y_unit, selected_unit=peak_unit)
            segment_peaks.append(scaled_peak_current)
        peaks.append(segment_peaks)

    if do_plot:
        plt.figure()
        plt.xlabel(r'(Scan Rate)$^{1/2}$ ((V/s)$^{1/2}$)' if 'scan rate' in different else 'Concentration (M)')
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
            diffusion_coefficients.append(float(D))

    sevcik_fit_table = _sevcik_fit_results_table(
        _scatter_fit_table(fit_rows),
        diffusion_coefficients,
        options=options,
    )

    if do_print:
        _display_sevcik_parameter_table(
            _sevcik_parameter_table(
                mode="scan rate" if "scan rate" in different else "concentration",
                num_electrons=num_electrons,
                temperature=T,
                electrode_area=S,
                concentration=C,
                scan_rate=v,
                options=options,
            ),
            options,
        )
        _display_sevcik_diffusion_equation(
            mode="scan rate" if "scan rate" in different else "concentration",
            num_electrons=num_electrons,
            temperature=T,
            electrode_area=S,
            concentration=C,
            scan_rate=v,
            resolved=False,
            compact=False,
            include_definitions=False,
        )
        _print_sevcik_fit_results(sevcik_fit_table, options)

    if do_print and options.get("print all"):
        identity_columns = [
            column for column in data.columns
            if column.startswith("Scan Rate") or column.startswith("Concentration")
        ]
        display_data = _conditional_analysis_name_column(
            data,
            identity_columns,
            options,
        )
        _display_table(
            display_data,
            options,
            title="Sevcik Analysis Data",
            index=False,
        )

    if isinstance(data, pd.DataFrame):
        data.attrs["fit table"] = sevcik_fit_table
        _attach_segment_selection_to_table(data, segment_selection)
    return diffusion_coefficients, data, fits


def sevcik_analysis(cvs, options=None):
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
    payload = _sevcik_analysis_payload(cvs, options)
    return _scatter_result_from_payload(
        payload,
        summary=_summary_with_segment_selection({"analysis": "Sevcik"}, payload),
    )


def _format_trumpet_equation():
    return {
        "symbolic latex": (
            r"E_{p,\mathrm{c}}=m_{\mathrm{c}}\log_{10}(\nu)+b_{\mathrm{c}},\quad "
            r"E_{p,\mathrm{a}}=m_{\mathrm{a}}\log_{10}(\nu)+b_{\mathrm{a}}"
        ),
        "resolved latex": (
            r"\alpha=-\frac{RT\ln(10)}{2Fm_{\mathrm{c}}},\quad "
            r"\beta=\frac{RT\ln(10)}{2Fm_{\mathrm{a}}},\quad "
            r"k^0=\frac{10^{0.78+x_{\mathrm{int}}/2}}{\sqrt{RT/(\alpha F D)}}"
        ),
        "compact latex": "",
        "definitions latex": "",
        "symbolic": (
            "Ep,c = mc * log10(v) + bc; "
            "Ep,a = ma * log10(v) + ba"
        ),
        "resolved": (
            "alpha = -(R * T * ln(10)) / (2 * F * mc); "
            "beta = (R * T * ln(10)) / (2 * F * ma); "
            "k0 = 10^(0.78 + xint / 2) / sqrt(R * T / (alpha * F * D))"
        ),
        "compact": "",
        "definitions": "",
    }


def _display_trumpet_equations(resolved=False, compact=False, include_definitions=False):
    return _display_analysis_equation(
        r"\text{Trumpet Analysis Equations:}",
        "Trumpet Analysis Equations",
        _format_trumpet_equation(),
        resolved=resolved,
        compact=compact,
        include_definitions=include_definitions,
    )


def _trumpet_parameter_table(
    cathodic_segments,
    anodic_segments,
    branch_assignment_source,
    temperature,
    diffusion_coefficient,
    fit_indices,
    options=None,
):
    options = options or {}
    sig_figs = options.get("sig figs", 4)

    def segment_value(values):
        values = [int(value) for value in values]
        if len(set(values)) == 1:
            return values[0]
        return "varies: " + ", ".join(str(value) for value in values)

    rows = [
        ("Cathodic Segment", "", segment_value(cathodic_segments)),
        ("Anodic Segment", "", segment_value(anodic_segments)),
        ("Branch Assignment", "", branch_assignment_source),
        ("Temperature", "T", _format_sevcik_value(temperature, sig_figs=sig_figs, unit="K")),
    ]
    if diffusion_coefficient is not None:
        rows.append(
            (
                "Diffusion Coefficient",
                "D",
                _format_sevcik_value(diffusion_coefficient, sig_figs=sig_figs, unit="cm^2/s", scientific=True),
            )
        )
    if fit_indices is not None:
        rows.append(("Fit Indices", "", f"{fit_indices} (Python-style exclusive stop)"))
    return _analysis_parameter_table(rows)


def _display_trumpet_parameter_table(table, options):
    display_table = _analysis_parameter_rich_table(table)
    return _display_table(
        table,
        options,
        title="Trumpet Analysis Parameters",
        rich_table=display_table,
        escape=None,
        index=False,
    )


def _trumpet_reliability_warning(alpha, beta, cathodic_slope, anodic_slope, intercept_x, fit_x):
    reasons = []
    if not (np.isfinite(cathodic_slope) and cathodic_slope < 0):
        reasons.append("cathodic branch slope does not have the expected negative sign")
    if not (np.isfinite(anodic_slope) and anodic_slope > 0):
        reasons.append("anodic branch slope does not have the expected positive sign")
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
    plain_table = table.rename(columns={"Parameter": "Metric"})
    display_table = plain_table.copy()
    display_table["Metric"] = [
        {
            "α": "&alpha;",
            "β": "&beta;",
            "k0": "<i>k</i><sup>0</sup>",
        }.get(str(value), value)
        for value in display_table["Metric"]
    ]
    return _display_table(
        table,
        options,
        title="Trumpet Analysis Summary",
        rich_table=display_table,
        plain_table=plain_table,
        escape=None,
        index=False,
    )

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


def _option_key_text(key):
    return str(key).strip().lower().replace("_", " ").replace("-", " ")


def _canonical_peak_kind_option(value):
    if value is None:
        return None
    token = _option_key_text(value)
    if token in {"both", "any", "all", "none"}:
        return None
    if token in {"infer", "inferred"}:
        return "infer"
    if token in {"max", "maximum"}:
        return "max"
    if token in {"min", "minimum"}:
        return "min"
    return value


def _raw_options_has_segment_selection(raw_options):
    if raw_options is None:
        return False
    if isinstance(raw_options, dict):
        normalized = {
            _option_key_text(key): value
            for key, value in raw_options.items()
        }
        return normalized.get("segment") is not None or normalized.get("segments") is not None
    return (
        getattr(raw_options, "segment", None) is not None
        or getattr(raw_options, "segments", None) is not None
    )


def _raw_options_segments(raw_options):
    if raw_options is None:
        return None
    if isinstance(raw_options, dict):
        normalized = {
            _option_key_text(key): value
            for key, value in raw_options.items()
        }
        if normalized.get("segments") is not None:
            return normalized["segments"]
        return normalized.get("segment")
    if getattr(raw_options, "segments", None) is not None:
        return getattr(raw_options, "segments")
    return getattr(raw_options, "segment", None)


def _segment_anchor_from_options(options):
    anchor = options.get("exact potential", None)
    if anchor is None:
        anchor = options.get("guess potential", None)
    if anchor is None:
        anchor = options.get("peak potential", None)
    if isinstance(anchor, (list, tuple, np.ndarray)):
        if len(anchor) == 0:
            return None
        anchor = anchor[0]
    try:
        return float(anchor)
    except (TypeError, ValueError):
        return None


def _shared_segment_anchor_options(options, n_cvs, *, paired=False):
    """
    Return an options view suitable for selecting one shared analysis segment.

    Per-CV potential lists should guide each downstream CV extraction, not be
    mistaken for a single shared segment-selection anchor.
    """
    segment_options = dict(options)
    for key in ("exact potential", "guess potential", "peak potential"):
        value = segment_options.get(key)
        if not _is_option_sequence(value):
            continue

        values = _as_option_list(value)
        if len(values) == 0:
            segment_options[key] = None
        elif len(values) == 1 and not _is_option_sequence(values[0]):
            segment_options[key] = values[0]
        elif (
            paired
            and key == "guess potential"
            and len(values) == 1
            and _is_pair_sequence(values[0])
        ):
            segment_options[key] = _as_option_list(values[0])[0]
        elif (
            paired
            and key == "guess potential"
            and len(values) == 2
            and int(n_cvs) != 2
            and not any(_is_option_sequence(item) for item in values)
        ):
            segment_options[key] = values[0]
        else:
            segment_options[key] = None
    return segment_options


def _cv_segment_numbers(cv_obj):
    total = getattr(cv_obj, "segments", None)
    try:
        total = int(total)
    except (TypeError, ValueError):
        total = None
    if total is None or total < 1:
        try:
            total = int(count_segments(cv_obj.x()))
        except Exception:
            total = 1
    return list(range(1, max(1, total) + 1))


def _analysis_segment_contains_anchor(cv_obj, segment, anchor):
    if anchor is None:
        return True
    try:
        x, _ = cv_obj.analysis_segment_data({"segment": int(segment)})
        x = np.asarray(x, dtype=float)
    except Exception:
        return False
    if len(x) == 0:
        return False
    lo = float(np.nanmin(x))
    hi = float(np.nanmax(x))
    target = float(anchor)
    return min(lo, hi) <= target <= max(lo, hi)


def _segment_global_index(cv_obj, segment, local_index):
    start = 0
    for previous in range(1, int(segment)):
        try:
            x_prev, _ = cv_obj.analysis_segment_data({"segment": previous})
            start += len(x_prev)
        except Exception:
            pass
    return start + int(local_index)


def _call_cv_segment_analysis(cv_obj, method_name, segment, options, *, guess=None):
    if method_name == "peak_potential":
        probe_options = _analysis_options_for(PeakPotentialOptions, options)
    elif method_name == "peak_current":
        probe_options = _analysis_options_for(PeakCurrentOptions, options)
    else:
        probe_options = dict(options)
    probe_options["plot"] = False
    probe_options["print"] = False
    probe_options["plot all"] = False
    probe_options["print all"] = False
    probe_options["internal call"] = True
    probe_options["new plot"] = False
    probe_options["plot cv"] = False
    probe_options["segment"] = int(segment)
    probe_options.pop("segments", None)
    probe_options.pop("plot segment", None)
    probe_options.pop("plot segments", None)
    if guess is not None and probe_options.get("exact potential") is None:
        probe_options["guess potential"] = guess
    method = getattr(cv_obj, method_name)
    return method(probe_options)


def _resolve_auto_single_analysis_segment(
    cvs,
    raw_options,
    options,
    *,
    method_name,
    analysis_name,
    default=1,
    paired_potential_guess=False,
):
    if _raw_options_has_segment_selection(raw_options):
        return _normalize_segment_option(options), None

    segment_options = _shared_segment_anchor_options(
        options,
        len(cvs),
        paired=paired_potential_guess,
    )
    anchor = _segment_anchor_from_options(segment_options)
    if anchor is None:
        selection = {
            "mode": "default",
            "analysis": analysis_name,
            "selected segment": int(default),
            "reason": "no anchor potential provided",
        }
        return [int(default)], selection

    support = {}
    details = []
    exact_anchor = segment_options.get("exact potential") is not None
    for cv_obj in cvs:
        candidates = [
            segment
            for segment in _cv_segment_numbers(cv_obj)
            if _analysis_segment_contains_anchor(cv_obj, segment, anchor)
        ]
        if exact_anchor and candidates:
            selected = int(default) if int(default) in candidates else int(candidates[0])
            support[selected] = support.get(selected, 0) + 1
            details.append({
                "name": getattr(cv_obj, "name", f"CV {len(details) + 1}"),
                "selected segment": selected,
                "Ep": float(anchor),
                "distance": 0.0,
                "source": "exact potential containment",
            })
            continue
        successes = []
        failures = []
        for segment in candidates:
            try:
                result = _call_cv_segment_analysis(cv_obj, method_name, segment, segment_options)
                ep = result.get("Ep", anchor) if hasattr(result, "get") else anchor
                distance = abs(float(ep) - float(anchor))
                successes.append((segment, distance, ep))
            except Exception as exc:
                failures.append((segment, str(exc)))
        if successes:
            successes.sort(key=lambda item: (item[1], item[0]))
            selected, distance, ep = successes[0]
            support[selected] = support.get(selected, 0) + 1
            details.append({
                "name": getattr(cv_obj, "name", f"CV {len(details) + 1}"),
                "selected segment": int(selected),
                "Ep": float(ep),
                "distance": float(distance),
            })
        elif len(candidates) == 1:
            selected = int(candidates[0])
            support[selected] = support.get(selected, 0) + 1
            details.append({
                "name": getattr(cv_obj, "name", f"CV {len(details) + 1}"),
                "selected segment": selected,
                "Ep": float(anchor),
                "distance": 0.0,
                "source": "single containing segment fallback",
                "failures": failures,
            })
        else:
            details.append({
                "name": getattr(cv_obj, "name", f"CV {len(details) + 1}"),
                "selected segment": None,
                "failures": failures,
            })

    if not support:
        raise ValueError(
            f"{analysis_name} could not auto-select a segment near {anchor:g} V. "
            "Pass 'segment' explicitly."
        )

    max_support = max(support.values())
    tied = sorted(segment for segment, count in support.items() if count == max_support)
    if len(tied) > 1:
        support_text = ", ".join(f"{segment}: {support[segment]}" for segment in tied)
        raise ValueError(
            f"{analysis_name} could not choose one segment near {anchor:g} V "
            f"because support was tied ({support_text}). Pass 'segment' explicitly."
        )

    selected = tied[0]
    selection = {
        "mode": "auto",
        "analysis": analysis_name,
        "selected segment": int(selected),
        "anchor potential": float(anchor),
        "method": method_name,
        "support": {int(k): int(v) for k, v in sorted(support.items())},
        "num cvs": int(len(cvs)),
        "details": details,
    }
    return [int(selected)], selection


def _resolve_closest_adjacent_segment_pair(cvs, base_segment, options):
    base_segment = int(base_segment)
    support = {}
    distances = {}
    details = []
    anchor = _segment_anchor_from_options(options)

    for cv_obj in cvs:
        candidate_segments = [
            segment
            for segment in (base_segment - 1, base_segment + 1)
            if segment in _cv_segment_numbers(cv_obj)
        ]
        if not candidate_segments:
            continue
        try:
            base_result = _call_cv_segment_analysis(
                cv_obj,
                "peak_potential",
                base_segment,
                options,
                guess=anchor,
            )
            base_ep = float(base_result["Ep"])
            base_index = _segment_global_index(cv_obj, base_segment, base_result["index"])
        except Exception:
            continue

        candidate_results = []
        for candidate in candidate_segments:
            try:
                result = _call_cv_segment_analysis(
                    cv_obj,
                    "peak_potential",
                    candidate,
                    options,
                    guess=base_ep,
                )
                global_index = _segment_global_index(cv_obj, candidate, result["index"])
                index_distance = abs(global_index - base_index)
                candidate_results.append((candidate, index_distance, float(result["Ep"])))
            except Exception:
                continue
        if not candidate_results:
            continue
        candidate_results.sort(
            key=lambda item: (
                item[1],
                0 if item[0] == base_segment + 1 else 1,
                item[0],
            )
        )
        selected, distance, ep = candidate_results[0]
        support[selected] = support.get(selected, 0) + 1
        distances.setdefault(selected, []).append(float(distance))
        details.append({
            "name": getattr(cv_obj, "name", f"CV {len(details) + 1}"),
            "base segment": base_segment,
            "paired segment": int(selected),
            "index distance": float(distance),
            "paired Ep": float(ep),
        })

    if support:
        max_support = max(support.values())
        tied = [segment for segment, count in support.items() if count == max_support]
        if len(tied) > 1:
            tied.sort(
                key=lambda segment: (
                    float(np.mean(distances.get(segment, [np.inf]))),
                    0 if segment == base_segment + 1 else 1,
                    segment,
                )
            )
        selected = int(tied[0])
    else:
        available = set()
        for cv_obj in cvs:
            available.update(_cv_segment_numbers(cv_obj))
        if base_segment + 1 in available:
            selected = base_segment + 1
        elif base_segment - 1 in available:
            selected = base_segment - 1
        else:
            selected = base_segment + 1

    selection = {
        "paired segment": int(selected),
        "paired support": {int(k): int(v) for k, v in sorted(support.items())},
        "paired details": details,
    }
    return int(selected), selection


def _resolve_analysis_segment_pair(cvs, raw_options, options, *, analysis_name):
    segment_options = _shared_segment_anchor_options(options, len(cvs), paired=True)
    requested = _raw_options_segments(raw_options)
    if _raw_options_has_segment_selection(raw_options) and requested is not None:
        if isinstance(requested, int):
            base = int(requested)
        elif isinstance(requested, (list, tuple, np.ndarray)):
            values = list(requested)
            if len(values) == 2:
                return int(values[0]), int(values[1]), {
                    "mode": "explicit",
                    "analysis": analysis_name,
                    "selected segment": int(values[0]),
                    "paired segment": int(values[1]),
                }
            if len(values) == 1:
                base = int(values[0])
            else:
                raise ValueError(
                    f"'segments' for {analysis_name} must contain one base segment or a 2-segment pair."
                )
        else:
            base = int(requested)
        paired, paired_selection = _resolve_closest_adjacent_segment_pair(cvs, base, segment_options)
        selection = {
            "mode": "explicit",
            "analysis": analysis_name,
            "selected segment": int(base),
            **paired_selection,
        }
        return int(base), int(paired), selection

    segments, selection = _resolve_auto_single_analysis_segment(
        cvs,
        raw_options,
        options,
        method_name="peak_potential",
        analysis_name=analysis_name,
        default=1,
        paired_potential_guess=True,
    )
    base = int(segments[0])
    paired, paired_selection = _resolve_closest_adjacent_segment_pair(cvs, base, segment_options)
    selection = dict(selection or {})
    selection.update(paired_selection)
    return int(base), int(paired), selection


def _format_segment_selection(selection):
    if not selection:
        return None
    base = selection.get("selected segment")
    paired = selection.get("paired segment")
    mode = selection.get("mode", "auto")
    anchor = selection.get("anchor potential")
    support = selection.get("support")
    parts = [f"{mode}: segment {base}"]
    if paired is not None:
        parts.append(f"paired with segment {paired}")
    if anchor is not None:
        parts.append(f"anchor {anchor:g} V")
    if support:
        support_text = ", ".join(f"{k}={v}" for k, v in support.items())
        parts.append(f"support {support_text}")
    return "; ".join(parts)


def _attach_segment_selection_to_table(data, selection):
    if isinstance(data, pd.DataFrame) and selection:
        data.attrs["segment selection"] = selection


def _summary_with_segment_selection(summary, payload):
    merged = dict(summary or {})
    table = None
    if isinstance(payload, tuple):
        if len(payload) == 3 and isinstance(payload[1], pd.DataFrame):
            table = payload[1]
        elif len(payload) > 0 and isinstance(payload[0], pd.DataFrame):
            table = payload[0]
    elif isinstance(payload, pd.DataFrame):
        table = payload
    if table is not None:
        selection = table.attrs.get("segment selection")
        if selection:
            merged["segment selection"] = selection
    return merged


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


def _diagnostic_y_axis_from_y_axis(value):
    key = str(value).strip().lower().replace(" ", "")
    if key in {"current", "rawcurrent"}:
        return "current"
    if key == "i/ip0":
        return "i/ip0"
    return None


def _apply_diagnostic_y_axis_alias(raw_options, options, analysis_name):
    if not isinstance(raw_options, dict):
        return
    if not _option_was_provided(raw_options, "y axis"):
        return
    if _option_was_provided(raw_options, "diagnostic y axis"):
        return
    mapped_axis = _diagnostic_y_axis_from_y_axis(options.get("y axis"))
    if mapped_axis is None:
        raise OptionError(
            f"For {analysis_name} diagnostic CV overlays, 'y axis' accepts "
            "'Current' or 'i/ip0'. Use 'diagnostic y axis' for the explicit "
            "analysis option."
        )
    options["diagnostic y axis"] = mapped_axis


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
    if _option_was_provided(raw_options, "fit color"):
        selected = _fit_color_from_options(resolved, index=index, fallback=color)
        if selected is not None:
            resolved["fit color"] = selected
    elif color is not None:
        resolved["fit color"] = color
    return resolved


def _plot_all_multiplot_options(options, raw_options):
    routed = _multiplot_options_from_mapping(options)
    if not _option_was_provided(raw_options, "legend"):
        routed["legend"] = True
    return routed


def _common_cv_plot_axis_options(cvs, options):
    """Resolve the axis choices that shared CV diagnostic overlays must use."""
    if not cvs:
        return {}

    common = {}
    if options.get("x axis") is not None:
        common["x axis"] = options.get("x axis")
    if options.get("y axis") is not None:
        common["y axis"] = options.get("y axis")

    try:
        common["x unit"] = _axis_common_unit(
            cvs,
            lambda cv_obj: (cv_obj.x(options).values, cv_obj.x(options).name),
            options.get("x unit", "auto"),
        )
    except Exception:
        if options.get("x unit") not in (None, "auto"):
            common["x unit"] = options.get("x unit")

    if _is_ip0_y_axis(options.get("y axis", "")):
        common["y unit"] = None
    else:
        try:
            common["y unit"] = _axis_common_unit(
                cvs,
                lambda cv_obj: (cv_obj.y(options), cv_obj.y(options).name),
                options.get("y unit", "auto"),
            )
        except Exception:
            if options.get("y unit") not in (None, "auto"):
                common["y unit"] = options.get("y unit")

    return common


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
        "1/": ("reciprocal", None),
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

    if options.get("plot log-log", False):
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


def _fit_peak_potential_payload(cvs, options=None):
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
    options = typed_options.to_options_dict()
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

    segments, segment_selection = _resolve_auto_single_analysis_segment(
        cvs,
        raw_options,
        options,
        method_name="peak_potential",
        analysis_name="fit_peak_potential",
        default=1,
    )

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
    internal_options.update(_common_cv_plot_axis_options(cvs, options))

    # Important: do not let peak_potential see a multi-segment request
    internal_options.pop("segments", None)
    internal_options.pop("plot segment", None)
    internal_options.pop("plot segments", None)
    potential_series = _resolve_complex_potential_series_map(
        raw_options,
        options,
        n_cvs=len(cvs),
        analysis_name="fit_peak_potential",
        option_names=["guess potential", "exact potential"],
    )
    for key in potential_series:
        internal_options.pop(key, None)
    user_peak_kind = _canonical_peak_kind_option(options.get("peak kind"))
    if user_peak_kind is None:
        internal_options.pop("peak kind", None)
    else:
        internal_options["peak kind"] = user_peak_kind

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

        running_guess = potential_series["guess potential"][i]
        exact_potential = potential_series["exact potential"][i]
        ep_by_segment = {}

        # Collect Ep for each requested segment
        for seg in segments:
            seg_options = internal_options.copy()

            if seg is not None:
                seg_options["segment"] = seg
            else:
                seg_options.pop("segment", None)

            guess_for_call = running_guess
            peak_kind_for_call = user_peak_kind

            if exact_potential is not None:
                seg_options["exact potential"] = exact_potential
                seg_options.pop("guess potential", None)
                seg_options.pop("peak kind", None)
            elif guess_for_call is not None:
                seg_options["guess potential"] = guess_for_call
                if peak_kind_for_call is not None:
                    seg_options["peak kind"] = peak_kind_for_call
                else:
                    seg_options.pop("peak kind", None)
            else:
                seg_options.pop("guess potential", None)
                if peak_kind_for_call is None:
                    seg_options.pop("peak kind", None)

            # One segment at a time so peak_potential does not search across combined segments
            peak_result = cv_obj.peak_potential(seg_options)
            Ep = peak_result["Ep"]
            if options.get("plot all", False) and exact_potential is not None:
                x_scale, y_scale = cv_obj.xy_scale(seg_options)
                plt.scatter(
                    peak_result["Ep"] * x_scale,
                    peak_result["current"] * y_scale + seg_options.get("offset", 0),
                    color="tab:blue",
                    zorder=3,
                )
            scaled_Ep, _ = scale_value(Ep, x_unit, selected_unit=ep_unit)

            if seg is None:
                row["Ep (V)"] = Ep
                row[f"Ep ({ep_unit})"] = scaled_Ep
            else:
                row[f"Seg {seg} Ep (V)"] = Ep
                row[f"Seg {seg} Ep ({ep_unit})"] = scaled_Ep
                ep_by_segment[seg] = Ep

            # Within each CV, use the most recent segment's Ep as the next guess.
            running_guess = Ep
        # Interpret "follow E1/2" as:
        # for sequential segments, compute and store the half-wave potential
        # between segment n and n+1 using the already collected Ep values.
        if follow_e_half:
            numeric_segments = sorted(ep_by_segment.keys())

            for seg in numeric_segments:
                if seg + 1 not in ep_by_segment:
                    continue

                E_half = float((ep_by_segment[seg] + ep_by_segment[seg + 1]) / 2)
                delta_E = float(abs(ep_by_segment[seg] - ep_by_segment[seg + 1]))

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
            fit_specs = _fit_rate_selection_specs(options, idxs, fit_key)
            for range_label, fit_spec, is_named_selection in fit_specs:
                fit_x, fit_y = _fit_rate_selected_points(
                    seg_data[x_col].to_numpy(),
                    seg_data[adjusted_col].to_numpy(),
                    fit_spec,
                )
                output_key = fit_key if not is_named_selection else f"{fit_key} {range_label}"
                display_label = label if not is_named_selection else f"{label} {range_label}"
                fit_label = f"{display_label} Fit"
                series_fit = _fit_series_xy(
                    fit_x,
                    fit_y,
                    options=options,
                    label=output_key,
                )
                if do_plot and options["plot fit"]:
                    plot_options = _fit_rate_fit_options(options, range_label)
                    fit_line_index = fit_color_index
                    plot_options = _options_with_default_fit_color(
                        plot_options,
                        raw_options,
                        point_color,
                        index=fit_color_index,
                    )
                    fit_color_index += 1
                    plot_options.update({"new plot": False, "plot data": False, "model label": fit_label, "_fit line index": fit_line_index})
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

            point_color = None
            if do_plot:
                point_color = _artist_color(plt.scatter(
                    ehalf_data[x_col],
                    ehalf_data[adjusted_col],
                    label=label,
                ))

            idxs = options.get("fit indices")
            if isinstance(idxs, dict):
                idxs = idxs.get(fit_key, idxs.get("default", None))
            if idxs is None:
                idxs = [0, len(ehalf_data)]

            if do_fit:
                fit_specs = _fit_rate_selection_specs(options, idxs, fit_key)
                for range_label, fit_spec, is_named_selection in fit_specs:
                    fit_x, fit_y = _fit_rate_selected_points(
                        ehalf_data[x_col].to_numpy(),
                        ehalf_data[adjusted_col].to_numpy(),
                        fit_spec,
                    )
                    output_key = fit_key if not is_named_selection else f"{fit_key} {range_label}"
                    display_label = label if not is_named_selection else f"{label} {range_label}"
                    fit_label = f"{display_label} Fit"
                    series_fit = _fit_series_xy(
                        fit_x,
                        fit_y,
                        options=options,
                        label=output_key,
                    )
                    if do_plot and options["plot fit"]:
                        plot_options = _fit_rate_fit_options(options, range_label)
                        fit_line_index = fit_color_index
                        plot_options = _options_with_default_fit_color(
                            plot_options,
                            raw_options,
                            point_color,
                            index=fit_color_index,
                        )
                        fit_color_index += 1
                        plot_options.update({"new plot": False, "plot data": False, "model label": fit_label, "_fit line index": fit_line_index})
                        _plot_fit_model_result(series_fit["model_result"], plot_options)
                    fits[output_key] = series_fit["fits"]
                    fit_model_results[output_key] = series_fit["model_result"]
                    fit_rows.extend(series_fit["fit_rows"])

    if do_plot and _scatterfit_legend_requested(options):
        plt.legend(fontsize=_scatterfit_legend_fontsize(options))

    _attach_scatter_fit_table(data, fit_rows)
    data.attrs["fit model results"] = fit_model_results
    _attach_segment_selection_to_table(data, segment_selection)
    if do_print:
        _print_fit_model_results(fit_model_results, options)

    if len(fits) == 1:
        return data, next(iter(fits.values()))
    return data, fits


def fit_peak_potential(cvs, options=None):
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
    payload = _fit_peak_potential_payload(cvs, options)
    return _scatter_result_from_payload(
        payload,
        summary=_summary_with_segment_selection({"analysis": "peak potential fit"}, payload),
    )


def _fit_peak_current_payload(cvs, options=None):
    """
    Fit peak current (ip) vs scan rate or concentration.

    Behavior
    --------
    - Uses _resolve_varying_x(...) to determine whether scan rate or concentration varies.
    - Supports either 'segment' or 'segments'.
    - If multiple segments are requested, analyzes one segment at a time and returns
      one fit per segment.
    """
    raw_options = options
    typed_options = FitPeakCurrentOptions.from_options(options)
    options = typed_options.to_options_dict()
    do_plot = options.get("plot", True)
    do_print = options.get("print", True)
    do_fit = options.get("fit", True)
    x_transform = options.get("x transform", "^0.5")
    fit_color_index = 0

    if not do_plot:
        options["plot fit"] = False

    x_raw, x_label, x_kind, extra = _resolve_varying_x(cvs, options, do_print=do_print)
    if x_raw is None:
        return None, None

    x_raw = np.asarray(x_raw, dtype=float)

    default_x_transform = "identity" if x_transform is None else x_transform
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

    segments, segment_selection = _resolve_auto_single_analysis_segment(
        cvs,
        raw_options,
        options,
        method_name="peak_potential",
        analysis_name="fit_peak_current",
        default=1,
    )

    # Determine a common current unit, like Sevcik / fit_peak_current.
    peak_unit = _axis_common_unit(
        cvs,
        lambda cv: (cv.y(options), cv.y(options).name),
        options.get("y unit", "auto"),
    )

    # If plotting all intermediate work, first show the CVs together
    if options.get("plot all", False):
        multiplot(cvs, options=_plot_all_multiplot_options(options, raw_options))

    internal_options = typed_options.for_peak_current().to_options_dict()
    internal_options["internal call"] = True
    internal_options["new plot"] = False
    internal_options["plot"] = options.get("plot all", False)
    internal_options["print"] = options.get("print all", False)
    internal_options.update(_common_cv_plot_axis_options(cvs, options))

    # Important: only pass one segment at a time downstream
    internal_options.pop("segments", None)
    internal_options.pop("plot segment", None)
    internal_options.pop("plot segments", None)
    potential_series = _resolve_complex_potential_series_map(
        raw_options,
        options,
        n_cvs=len(cvs),
        analysis_name="fit_peak_current",
        option_names=["guess potential", "exact potential", "tangent potential"],
    )
    for key in potential_series:
        internal_options.pop(key, None)

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
        for cv_index, cv in enumerate(cvs):
            cv_peak_options = _apply_resolved_potential_options(
                seg_options.copy(),
                potential_series,
                cv_index,
            )
            y_arr = cv.y(options)
            y_name = y_arr.name
            y_unit = cv.units.get(y_name, "")

            peak_current = cv.peak_current(cv_peak_options)["ip"]
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
            fit_specs = _fit_rate_selection_specs(options, idxs, series_label)
            for range_label, fit_spec, is_named_selection in fit_specs:
                fit_x, fit_y = _fit_rate_selected_points(fit_x_all, fit_y_all, fit_spec)

                if len(fit_x) < 2:
                    raise ValueError("fit_peak_current requires at least two transformable points.")

                output_key = seg if not is_named_selection else f"{series_label} {range_label}"
                display_label = series_label if not is_named_selection else f"{series_label} {range_label}"
                series_fit = _fit_series_xy(
                    fit_x,
                    fit_y,
                    options=options,
                    label=display_label,
                )
                if do_plot and options["plot fit"]:
                    plot_options = _fit_rate_fit_options(options, range_label)
                    fit_line_index = fit_color_index
                    plot_options = _options_with_default_fit_color(
                        plot_options,
                        raw_options,
                        point_color,
                        index=fit_color_index,
                    )
                    fit_color_index += 1
                    plot_options.update({"new plot": False, "plot data": False, "model label": f"{display_label} Fit", "_fit line index": fit_line_index})
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
    _attach_segment_selection_to_table(data, segment_selection)
    if do_print:
        _print_fit_model_results(fit_model_results, options)

    if len(segments) == 1:
        return data, fits[segments[0]]
    return data, fits


def fit_peak_current(cvs, options=None):
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
    payload = _fit_peak_current_payload(cvs, options)
    return _scatter_result_from_payload(
        payload,
        summary=_summary_with_segment_selection({"analysis": "peak current fit"}, payload),
    )

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
            _format_fowa_line(slope, intercept, options.get("sig figs", 4))
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
            _format_fowa_line(slope, intercept, options.get("sig figs", 4))
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
    identity_columns = [
        column for column in summary_df.columns
        if column in display_df.columns and column not in {"Name", "Plot Label"}
    ]
    display_df = _conditional_analysis_name_column(
        display_df,
        identity_columns,
        options,
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
    symbol_map = {
        "Catalyst Electrons": "n",
        "Turnover Electrons": "n'",
        "Sigma": "σ",
    }
    return _analysis_parameter_table(
        [
            (str(key), symbol_map.get(str(key), ""), "" if value is None else str(value))
            for key, value in summary.items()
        ]
    )


def _fowa_summary_field_html_label(field):
    return _pretty_table_header_html_label(field)


def _display_fowa_summary_table(summary, options=None, *, title="FOWA Summary", plain_title=True):
    options = {} if options is None else options
    table = _fowa_summary_display_table(summary)
    if table.empty:
        return table

    display_table = _analysis_parameter_rich_table(table)
    display_table["Parameter"] = [
        _fowa_summary_field_html_label(field)
        for field in display_table["Parameter"]
    ]
    return _display_table(
        table,
        options,
        title=title,
        rich_table=display_table,
        escape=None,
        index=False,
        plain_title=plain_title,
    )


def _format_fowa_display_value(column, value, options=None):
    options = {} if options is None else options
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass

    sig_figs = options.get("sig figs", 4)
    column_key = str(column).strip().lower()
    if isinstance(value, (int, float, np.integer, np.floating)):
        if column_key == "kobs":
            return _format_sevcik_value(value, sig_figs=sig_figs, scientific=True)
        return _format_fit_model_display_value(value, sig_figs=sig_figs)
    return value


def _table_column_unit(table, column):
    units = getattr(table, "attrs", {}).get("units", {}) or {}
    if not units:
        return None
    column_text = str(column).strip()
    if column_text in units:
        return units[column_text]
    column_key = column_text.lower()
    for key, unit in units.items():
        if str(key).strip().lower() == column_key:
            return unit
        if str(pretty_table_column_label(key)).strip().lower() == column_key:
            return unit
    return None


def _is_table_blank(value):
    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


_ANALYSIS_SYMBOL_HTML = {
    "n": "<i>n</i>",
    "n'": "<i>n</i><sup>&prime;</sup>",
    "σ": "<i>&sigma;</i>",
    "T": "<i>T</i>",
    "D": "<i>D</i>",
    "S": "<i>S</i>",
    "C": "<i>C</i>",
    "C*": "<i>C</i><sup>*</sup>",
    "ν": "<i>&nu;</i>",
    "ν_ip0": "<i>&nu;</i><sub>ip0</sub>",
    "s_ip0": "<i>s</i><sub>ip0</sub>",
    "ψ": "<i>&psi;</i>",
    "nΔEp": "<i>n</i>&Delta;<i>E</i><sub>p</sub>",
    "Ethermo": "<i>E</i><sub>thermo</sub>",
    "Eredox": "<i>E</i><sub>redox</sub>",
    "TOFmax": "TOF<sub>max</sub>",
    "η": "<i>&eta;</i>",
}


def _analysis_symbol_html(symbol):
    symbol_text = "" if symbol is None else str(symbol)
    return _ANALYSIS_SYMBOL_HTML.get(symbol_text, symbol_text)


def _analysis_parameter_table(rows):
    records = []
    for row in rows:
        if len(row) == 2:
            parameter, value = row
            symbol = ""
        else:
            parameter, symbol, value = row
        records.append(
            {
                "Parameter": str(parameter),
                "Symbol": "" if symbol is None else str(symbol),
                "Value": "" if _is_table_blank(value) else str(value),
            }
        )
    return pd.DataFrame(records, columns=["Parameter", "Symbol", "Value"])


def _analysis_parameter_rich_table(table):
    rich_table = table.copy()
    if "Symbol" in rich_table.columns:
        rich_table["Symbol"] = [_analysis_symbol_html(value) for value in rich_table["Symbol"]]
    return rich_table


def _display_unit_for_numeric_column(values, unit):
    clean_values = [
        float(value)
        for value in values
        if isinstance(value, (int, float, np.integer, np.floating)) and np.isfinite(float(value))
    ]
    if not clean_values or not unit or str(unit).strip().lower() in {"dimensionless", "none"}:
        return None
    unit = str(unit)
    if unit in {"A", "V/s"}:
        try:
            _scaled, display_unit = scale_value(max(abs(value) for value in clean_values), unit, selected_unit="auto")
            return display_unit
        except Exception:
            return unit
    return unit


def _format_table_numeric_without_unit(value, unit, display_unit, options=None):
    options = {} if options is None else options
    sig_figs = options.get("sig figs", 4)
    numeric_value = float(value)
    if display_unit and display_unit != unit:
        try:
            numeric_value, _display_unit = scale_value(numeric_value, unit, selected_unit=display_unit)
        except Exception:
            pass
    abs_value = abs(numeric_value)
    scientific = unit in {"cm^2/s", "mol/cm^3", "A/(V/s)^1/2"}
    if abs_value != 0 and (abs_value >= 1e4 or abs_value < 1e-3):
        scientific = True
    return _format_sevcik_value(numeric_value, sig_figs=sig_figs, scientific=scientific)


def _format_plain_unit_label(unit):
    text = str(unit)
    replacements = {
        "^-1": "⁻¹",
        "^1/2": "¹ᐟ²",
        "^2": "²",
        "^3": "³",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def _results_display_table_with_unit_headers(table, options=None, *, value_formatter):
    options = {} if options is None else options
    display_table = table.copy()
    rename_columns = {}
    for column in list(display_table.columns):
        unit = _table_column_unit(table, column)
        display_unit = _display_unit_for_numeric_column(display_table[column].tolist(), unit)
        if display_unit:
            rename_columns[column] = f"{column} / {_format_plain_unit_label(display_unit)}"
            formatted = []
            for value in display_table[column]:
                if _is_table_blank(value):
                    formatted.append("")
                elif isinstance(value, (int, float, np.integer, np.floating)):
                    formatted.append(
                        _format_table_numeric_without_unit(value, unit, display_unit, options)
                    )
                else:
                    formatted.append(str(value))
            display_table[column] = formatted
        else:
            display_table[column] = [
                value_formatter(column, value, options)
                for value in display_table[column]
            ]
    if rename_columns:
        display_table = display_table.rename(columns=rename_columns)
    return display_table


def _fowa_results_display_table(table, options=None):
    return _results_display_table_with_unit_headers(
        table,
        options,
        value_formatter=_format_fowa_display_value,
    )


def _display_fowa_results_table(table, options=None, *, title="FOWA Results", plain_title=True):
    options = {} if options is None else options
    if table.empty:
        return table

    display_table = _fowa_results_display_table(table, options)
    return _display_table(
        table,
        options,
        title=title,
        rich_table=display_table.rename(columns=_pretty_table_header_html_label),
        plain_table=display_table,
        escape=None,
        index=False,
        plain_title=plain_title,
    )


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

        if compact and equation.get("compact latex"):
            display(Math(equation["compact latex"]))

        if include_definitions:
            display(Math(equation["definitions latex"]))

    else:
        print(f"[{title_text}]")
        print("  " + equation["symbolic"])
        if resolved:
            print("  " + equation["resolved"])
        if compact and equation.get("compact"):
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
    if is_scan_rate_mode:
        fit_latex = r"i_p=m\nu^{1/2}+b"
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
            "i_p = m * v^0.5 + b; "
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
    )
    return _display_analysis_equation(
        r"\text{Sevcik analysis equations:}",
        "Sevcik Analysis Equations",
        equation,
        resolved=resolved,
        compact=compact,
        include_definitions=include_definitions,
    )


def _sevcik_parameter_label(key, html=False):
    labels = {
        "n": "Electron Count",
        "T": "Temperature",
        "S": "Electrode Area",
        "C": "Concentration",
        "v": "Scan Rate",
    }
    return labels.get(key, key)


def _sevcik_parameter_symbol(key):
    return {
        "n": "n",
        "T": "T",
        "S": "S",
        "C": "C*",
        "v": "ν",
    }.get(key, "")


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
            text = format_sigfigs(float(value), sig_figs)
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
        keys.append("C")
        values["C"] = concentration
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
                "Symbol": _sevcik_parameter_symbol(key),
                "Value": _format_sevcik_value(values.get(key), sig_figs=sig_figs, unit=units.get(key, "")),
            }
            for key in keys
        ]
        ,
        columns=["Parameter", "Symbol", "Value"],
    )
    table.attrs["parameter_keys"] = keys
    return table


def _display_sevcik_parameter_table(table, options):
    display_table = _analysis_parameter_rich_table(table)
    return _display_table(
        table,
        options,
        title="Sevcik Analysis Parameters",
        rich_table=display_table,
        escape=None,
        index=False,
    )


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


def _print_sevcik_fit_results(fit_table, options=None):
    if fit_table is None or len(fit_table) == 0:
        return
    return _display_table(
        fit_table,
        options,
        title="Sevcik Analysis Summary",
        index=False,
    )


def _format_fowa_kobs_equation(options):
    """
    Return display-ready text/LaTeX for the EC'-type FOWA kobs equation.

    Equation used:
        kobs = (m * 0.4463 * n / n'^sigma)^2
               * (n * F * v) / (R * T)

    Display notation follows the FOWA literature:
        n = n_cat = catalyst redox-wave electron count
        n' = n_turn = turnover electron count
    """
    n_cat = float(options.get("catalyst electrons", options.get("num electrons", 1)))
    n_turn = float(options.get("turnover electrons", 1))
    sigma = float(options.get("sigma", 1.0))
    turnover_factor = n_turn ** sigma

    symbolic_latex = (
        r"k_{\mathrm{obs}}"
        r"=\left("
        r"\frac{m\,0.4463\,n}"
        r"{(n^{\prime})^{\sigma}}"
        r"\right)^2"
        r"\frac{nF\nu}{RT}"
    )

    resolved_latex = (
        r"k_{\mathrm{obs}}"
        r"=\left("
        rf"\frac{{m\,0.4463\,({n_cat:g})}}"
        rf"{{({n_turn:g})^{{{sigma:g}}}}}"
        r"\right)^2"
        rf"\frac{{({n_cat:g})F\nu}}{{RT}}"
    )

    compact_latex = (
        r"k_{\mathrm{obs}}"
        r"=\left("
        rf"{0.4463 * n_cat / turnover_factor:.6g}\,m"
        r"\right)^2"
        rf"\frac{{({n_cat:g})F\nu}}{{RT}}"
    )

    definitions_latex = (
        rf"n={n_cat:g}\ (n_{{\mathrm{{cat}}}},\ \mathrm{{catalyst\ redox-wave\ electron\ count}}),\quad "
        rf"n^{{\prime}}={n_turn:g}\ (n_{{\mathrm{{turn}}}},\ \mathrm{{turnover\ electron\ count}}),\quad "
        rf"\sigma={sigma:g},\quad "
        rf"(n^{{\prime}})^{{\sigma}}={turnover_factor:.6g}"
    )

    symbolic_text = (
        "k_obs = (m * 0.4463 * n / n'^sigma)^2 "
        "* (n * F * v) / (R * T)"
    )

    resolved_text = (
        f"k_obs = (m * 0.4463 * {n_cat:g} / "
        f"{n_turn:g}^{sigma:g})^2 * "
        f"({n_cat:g} * F * v) / (R * T)"
    )

    compact_text = (
        f"k_obs = ({0.4463 * n_cat / turnover_factor:.6g} * m)^2 "
        f"* ({n_cat:g} * F * v) / (R * T)"
    )

    definitions_text = (
        f"n = {n_cat:g} (n_cat, catalyst redox-wave electron count), "
        f"n' = {n_turn:g} (n_turn, turnover electron count), "
        f"sigma = {sigma:g}, "
        f"n'^sigma = {turnover_factor:.6g}"
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
        r"\text{FOWA equations:}",
        "FOWA Equations",
        eq,
        resolved=resolved,
        compact=compact,
        include_definitions=False,
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
    non_catalytic_guess_override = False
    if role == "non-catalytic" and options.get("non-catalytic guess potential") is not None:
        guess = options.get("non-catalytic guess potential")
        non_catalytic_guess_override = True

    exact = None if non_catalytic_guess_override else options.get("exact potential")
    exact_potential = None
    if exact is not None:
        exact_potential = exact
    elif fallback_potential is not None:
        exact_potential = fallback_potential
    if exact_potential is not None:
        guess = None

    plot_peak_diagnostics = bool(options.get("_plot_peak_diagnostics", options.get("plot all", False)))
    internal = {
        "plot": plot_peak_diagnostics,
        "plot all": plot_peak_diagnostics,
        "print": bool(options.get("print all", False)),
        "print all": bool(options.get("print all", False)),
        "internal call": True,
        "new plot": False,
        "x axis": options.get("x axis"),
        "y axis": "Current",
        "y unit": "A",
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
        extracted = {
            "current": float(current),
            "potential": current_result.get("Ep", exact_potential),
            "peak index": current_result.get("Ep index"),
            "baseline current": current_result.get("baseline current"),
            "source": current_result.get("peak source", "peak_current"),
            "tanline": tanline,
            "tangent slope": current_result.get("tangent slope"),
            "tangent intercept": current_result.get("tangent intercept"),
            "tangent start": current_result.get("tangent start"),
            "fit indices": current_result.get("fit indices"),
            "cv": getattr(cv_obj, "name", "CV"),
            "scan rate": float(getattr(cv_obj, "scan_rate", np.nan)),
        }
        diagnostic_calls = options.get("_diagnostic_calls")
        if isinstance(diagnostic_calls, list):
            diagnostic_calls.append({
                "kind": "plateau_extraction",
                "obj": cv_obj,
                "options": internal.copy(),
                "role": role,
                **extracted,
            })
        return extracted
    except Exception as exc:
        raise ValueError(
            f"Could not extract {role} plateau current from '{getattr(cv_obj, 'name', 'CV')}'. "
            "Adjust 'peak fallback', provide 'exact potential' or 'guess potential', "
            "or verify that the selected segment contains usable current data."
        ) from exc


def _extract_catalytic_currents(cat_cvs, options, potential_series=None):
    rows = []
    for cv_index, cv_obj in enumerate(cat_cvs):
        cv_options = (
            _apply_resolved_potential_options(dict(options), potential_series, cv_index)
            if potential_series is not None
            else options
        )
        extracted = _extract_current_with_peak_current(cv_obj, cv_options, role="catalytic")
        scan_rate = extracted["scan rate"]
        rows.append({
            "cv": extracted["cv"],
            "scan rate": scan_rate,
            "sqrt scan rate": np.sqrt(scan_rate) if np.isfinite(scan_rate) and scan_rate >= 0 else np.nan,
            "ic": extracted["current"],
            "abs ic": abs(extracted["current"]),
            "current source": extracted["source"],
            "extraction potential": extracted["potential"],
            "peak index": extracted.get("peak index"),
            "baseline current": extracted.get("baseline current"),
            "tangent slope": extracted.get("tangent slope"),
            "tangent intercept": extracted.get("tangent intercept"),
            "tangent start": extracted.get("tangent start"),
            "fit indices": extracted.get("fit indices"),
            "ilim tangent": _format_plateau_tangent(extracted.get("tanline"), options),
            "valid extraction": True,
        })
    return pd.DataFrame(rows)


def _extract_ip0_currents(ref_cvs, options, potential_series=None):
    rows = []
    successes = []
    for cv_index, ref_cv in enumerate(ref_cvs):
        cv_options = (
            _apply_resolved_potential_options(dict(options), potential_series, cv_index)
            if potential_series is not None
            else options
        )
        if potential_series is not None:
            nc_guess = cv_options.pop("non-catalytic guess potential", None)
            if nc_guess is not None:
                cv_options.pop("exact potential", None)
                cv_options["guess potential"] = nc_guess
        scan_rate = float(getattr(ref_cv, "scan_rate", np.nan))
        fallback = None
        if successes and np.isfinite(scan_rate):
            fallback = min(successes, key=lambda item: abs(item["scan rate"] - scan_rate)).get("potential")
        extracted = _extract_current_with_peak_current(
            ref_cv,
            cv_options,
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
            "peak index": extracted.get("peak index"),
            "baseline current": extracted.get("baseline current"),
            "tangent slope": extracted.get("tangent slope"),
            "tangent intercept": extracted.get("tangent intercept"),
            "tangent start": extracted.get("tangent start"),
            "fit indices": extracted.get("fit indices"),
            "ip0 tangent": _format_plateau_tangent(extracted.get("tanline"), options),
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
            warnings_list.append("scan-rate independence cannot be tested")
        return {
            "ilim": float(df["ic"].iloc[0]),
            "accepted indices": [0],
            "accepted cvs": [df["cv"].iloc[0]],
            "accepted scan rates": [float(df["scan rate"].iloc[0])],
            "slope": np.nan,
            "intercept": np.nan,
            "slope metric": np.nan,
            "valid plateau": None,
            "validation status": "not tested",
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
        "validation status": "not requested" if not validate else ("passed" if bool(valid) else "failed"),
        "warnings": warnings_list,
    }


def _is_provided(value):
    if value is None:
        return False
    if isinstance(value, float) and np.isnan(value):
        return False
    return True


def _provided_label(value):
    return "provided" if _is_provided(value) else "missing"


def _format_plateau_formula_mode_error(
    options,
    *,
    D=None,
    C=None,
    electrode_area=None,
    ip0=None,
    ip0_scan_rate=None,
    ip0_sqrt_scan_rate_slope=None,
):
    direct_missing = []
    if D is None:
        direct_missing.append("D")
    if C is None:
        direct_missing.append("C (or species-resolved catalyst concentration)")
    if electrode_area is None:
        direct_missing.append("electrode area")

    slope_missing = []
    if ip0_sqrt_scan_rate_slope is None:
        slope_missing.append("ip0 sqrt scan rate slope (or multiple non-catalytic cvs)")

    normalized_missing = []
    if ip0 is None:
        normalized_missing.append("ip0/non-catalytic current (or one non-catalytic cv)")
    if ip0_scan_rate is None:
        normalized_missing.append("ip0 scan rate/scan rate (or the non-catalytic cv scan_rate)")

    noncat_status = (
        "provided"
        if options.get("non-catalytic cv") is not None or options.get("non-catalytic cvs") is not None
        else "missing"
    )
    received = [
        f"ilim/ic={_provided_label(options.get('ilim') if options.get('ilim') is not None else options.get('ic'))}",
        f"D={_provided_label(D)}",
        f"C={_provided_label(C)}",
        f"species={_provided_label(options.get('species'))}",
        f"electrode area={_provided_label(electrode_area)}",
        f"ip0/non-catalytic current={_provided_label(ip0)}",
        f"ip0 scan rate={_provided_label(ip0_scan_rate)}",
        f"ip0 sqrt scan rate slope={_provided_label(ip0_sqrt_scan_rate_slope)}",
        f"non-catalytic cv(s)={noncat_status}",
    ]

    return (
        "plateau_current could not resolve formula mode automatically. "
        "Auto mode needs one complete input path:\n"
        f"- direct: D + C/species + electrode area. Missing: {', '.join(direct_missing) or 'none'}.\n"
        "- slope-normalized: ip0 sqrt scan rate slope, or multiple non-catalytic cvs "
        f"to fit ip0 vs sqrt(scan rate). Missing: {', '.join(slope_missing) or 'none'}.\n"
        "- normalized: ip0/non-catalytic current + ip0 scan rate, or one non-catalytic cv "
        f"with scan_rate metadata. Missing: {', '.join(normalized_missing) or 'none'}.\n"
        f"Received: {'; '.join(received)}."
    )


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
    n_cat = float(options.get("catalyst electrons", 1))
    n_turn = float(options.get("turnover electrons", 1))
    if n_cat <= 0 or n_turn <= 0:
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
                _format_plateau_formula_mode_error(
                    options,
                    D=D,
                    C=C,
                    electrode_area=electrode_area,
                    ip0=ip0,
                    ip0_scan_rate=ip0_scan_rate,
                    ip0_sqrt_scan_rate_slope=ip0_sqrt_scan_rate_slope,
                )
            )
    if mode == "direct":
        if not direct_ready:
            raise ValueError("Direct plateau-current mode requires D, C, and electrode area.")
        kobs = (abs(ilim) / (n_cat * F * float(electrode_area) * float(C))) ** 2 / (float(D) * n_turn)
        return "direct", "kobs = (|ilim| / (n F A C_cat))^2 / (D n')", float(kobs)
    if mode == "slope normalized":
        if not slope_ready:
            raise ValueError("Slope-normalized plateau-current mode requires 'ip0 sqrt scan rate slope'.")
        kobs = (
            0.446 * abs(ilim) / abs(float(ip0_sqrt_scan_rate_slope))
            * np.sqrt(n_cat * F / (R * temperature))
        ) ** 2 / n_turn
        return "slope normalized", "kobs = (0.446 |ilim|/|s_ip0| sqrt(n F/RT))^2 / n'", float(kobs)
    if mode == "normalized":
        if not normalized_ready:
            raise ValueError("Normalized plateau-current mode requires ip0 and ip0 scan rate.")
        kobs = (
            0.446 * abs(ilim / float(ip0))
            * np.sqrt(n_cat * F * float(ip0_scan_rate) / (R * temperature))
        ) ** 2 / n_turn
        return "normalized", "kobs = (0.446 |ilim/ip0| sqrt(n F v_ip0/RT))^2 / n'", float(kobs)
    raise ValueError("'formula mode' must be 'auto', 'normalized', 'slope normalized', or 'direct'.")


def _format_plateau_kobs_equation(mode, values):
    mode = str(mode or "").strip().lower().replace("-", " ")
    n_cat = float(values.get("catalyst electrons", 1))
    n_turn = float(values.get("turnover electrons", 1))
    temperature = float(values.get("temperature", 298))
    ilim = values.get("ilim")
    ip0 = values.get("ip0")
    ip0_scan_rate = values.get("ip0 scan rate")
    ip0_slope = values.get("ip0 sqrt scan rate slope")
    D = values.get("D")
    C = values.get("C")
    area = values.get("electrode area")

    definitions_latex = ""
    definitions_text = ""

    if mode == "direct":
        symbolic_latex = (
            r"k_{\mathrm{obs}}="
            r"\frac{1}{D n^{\prime}}"
            r"\left(\frac{|i_{\lim}|}{nFSC}\right)^2"
        )
        resolved_latex = (
            r"k_{\mathrm{obs}}="
            rf"\frac{{1}}{{({0 if D is None else D:g})({n_turn:g})}}"
            rf"\left(\frac{{|{0 if ilim is None else ilim:g}|}}"
            rf"{{({n_cat:g})F({0 if area is None else area:g})({0 if C is None else C:g})}}\right)^2"
        )
        symbolic = "kobs = (|ilim| / (n F S C))^2 / (D n')"
        resolved = (
            f"kobs = (|{_equation_value_text(ilim, 'A')}| / "
            f"({n_cat:g} * F * {_equation_value_text(area, 'cm^2')} * "
            f"{_equation_value_text(C, 'mol/cm^3')}))^2 / "
            f"({_equation_value_text(D, 'cm^2/s')} * {n_turn:g})"
        )
    elif mode == "slope normalized":
        symbolic_latex = (
            r"k_{\mathrm{obs}}="
            r"\frac{1}{n^{\prime}}"
            r"\left(0.446\frac{|i_{\lim}|}{|s_{i_{p,0}}|}"
            r"\sqrt{\frac{nF}{RT}}\right)^2"
        )
        resolved_latex = (
            r"k_{\mathrm{obs}}="
            rf"\frac{{1}}{{{n_turn:g}}}"
            rf"\left(0.446\frac{{|{0 if ilim is None else ilim:g}|}}"
            rf"{{|{0 if ip0_slope is None else ip0_slope:g}|}}"
            rf"\sqrt{{\frac{{({n_cat:g})F}}{{R({temperature:g})}}}}\right)^2"
        )
        symbolic = "kobs = (0.446 |ilim|/|s_ip0| sqrt(n F/RT))^2 / n'"
        resolved = (
            f"kobs = (0.446 * |{_equation_value_text(ilim, 'A')}| / "
            f"|{_equation_value_text(ip0_slope, 'A/(V/s)^1/2')}| "
            f"* sqrt({n_cat:g} * F / (R * {temperature:g} K)))^2 / {n_turn:g}"
        )
    else:
        symbolic_latex = (
            r"k_{\mathrm{obs}}="
            r"\frac{1}{n^{\prime}}"
            r"\left(0.446\left|\frac{i_{\lim}}{i_{p,0}}\right|"
            r"\sqrt{\frac{nF\nu_{i_{p,0}}}{RT}}\right)^2"
        )
        resolved_latex = (
            r"k_{\mathrm{obs}}="
            rf"\frac{{1}}{{{n_turn:g}}}"
            rf"\left(0.446\left|\frac{{{0 if ilim is None else ilim:g}}}"
            rf"{{{0 if ip0 is None else ip0:g}}}\right|"
            rf"\sqrt{{\frac{{({n_cat:g})F({0 if ip0_scan_rate is None else ip0_scan_rate:g})}}"
            rf"{{R({temperature:g})}}}}\right)^2"
        )
        symbolic = "kobs = (0.446 |ilim/ip0| sqrt(n F v_ip0/RT))^2 / n'"
        resolved = (
            f"kobs = (0.446 * |{_equation_value_text(ilim, 'A')} / "
            f"{_equation_value_text(ip0, 'A')}| "
            f"* sqrt({n_cat:g} * F * {_equation_value_text(ip0_scan_rate, 'V/s')} "
            f"/ (R * {temperature:g} K)))^2 / {n_turn:g}"
        )

    return {
        "symbolic latex": symbolic_latex,
        "resolved latex": resolved_latex,
        "compact latex": "",
        "definitions latex": definitions_latex,
        "symbolic": symbolic,
        "resolved": resolved,
        "compact": "",
        "definitions": definitions_text,
    }


def _display_plateau_kobs_equation(mode, values, resolved=False, compact=False):
    equation = _format_plateau_kobs_equation(mode, values)
    return _display_analysis_equation(
        r"\text{Plateau current equations:}",
        "Plateau Current Equations",
        equation,
        resolved=resolved,
        compact=compact,
        include_definitions=False,
    )


def _plateau_scaled_value(value, unit, options=None, *, scientific=False):
    options = {} if options is None else options
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    sig_figs = options.get("sig figs", 4)
    if scientific:
        return _format_sevcik_value(value, sig_figs=sig_figs, unit=unit, scientific=True)
    try:
        scaled, display_unit = scale_value(float(value), unit, selected_unit="auto")
        return _format_sevcik_value(scaled, sig_figs=sig_figs, unit=display_unit)
    except Exception:
        return _format_sevcik_value(value, sig_figs=sig_figs, unit=unit)


def _format_plateau_display_value(key, value, options=None):
    key = str(key).strip().lower()
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass

    if key in {"ilim", "ip0"}:
        return _plateau_scaled_value(value, "A", options)
    if key in {"ip0 scan rate", "scan rate"}:
        return _plateau_scaled_value(value, "V/s", options)
    if key == "kobs":
        return _format_sevcik_value(value, sig_figs=(options or {}).get("sig figs", 4), unit="s^-1")
    if key == "temperature":
        return _format_sevcik_value(value, sig_figs=(options or {}).get("sig figs", 4), unit="K")
    if key == "d":
        return _format_sevcik_value(value, sig_figs=(options or {}).get("sig figs", 4), unit="cm^2/s", scientific=True)
    if key == "c":
        return _format_sevcik_value(value, sig_figs=(options or {}).get("sig figs", 4), unit="mol/cm^3", scientific=True)
    if key == "electrode area":
        return _format_sevcik_value(value, sig_figs=(options or {}).get("sig figs", 4), unit="cm^2")
    if key == "ip0 sqrt scan rate slope":
        return _format_sevcik_value(
            value,
            sig_figs=(options or {}).get("sig figs", 4),
            unit="A/(V/s)^1/2",
            scientific=True,
        )
    if key in {"ilim/ip0", "plateau slope metric", "ip0 fit r2"}:
        return _format_fit_model_display_value(value, sig_figs=(options or {}).get("sig figs", 4))
    if isinstance(value, (int, float, np.integer, np.floating)):
        return _format_fit_model_display_value(value, sig_figs=(options or {}).get("sig figs", 4))
    if isinstance(value, (list, tuple)):
        return ", ".join(str(item) for item in value)
    return str(value)


def _format_plateau_tangent(tanline, options=None):
    if tanline is None:
        return ""
    try:
        if isinstance(tanline, dict):
            slope = tanline.get("slope")
            intercept = tanline.get("intercept")
        else:
            slope, intercept = list(tanline)[:2]
    except Exception:
        return ""
    return _format_fowa_line(slope, intercept, (options or {}).get("sig figs", 4))


def _plateau_common_table_value(df, column, indices=None):
    if df is None or df.empty or column not in df:
        return ""
    source = df
    if indices is not None:
        valid_indices = [index for index in indices if index in df.index]
        if valid_indices:
            source = df.loc[valid_indices]
    values = [
        str(value)
        for value in source[column].tolist()
        if value is not None and str(value).strip() != ""
    ]
    if not values:
        return ""
    unique = list(dict.fromkeys(values))
    return unique[0] if len(unique) == 1 else "per CV"


def _plateau_validation_status(selection):
    status = selection.get("validation status")
    if status:
        return status
    valid = selection.get("valid plateau")
    if valid is None:
        return "not tested"
    return "passed" if bool(valid) else "failed"


_PLATEAU_PARAMETER_LABELS = {
    "catalyst electrons": "Catalyst Electrons",
    "turnover electrons": "Turnover Electrons",
    "temperature": "Temperature",
    "d": "Diffusion Coefficient",
    "c": "Catalyst Concentration",
    "electrode area": "Electrode Area",
    "ip0 scan rate": "ip0 Scan Rate",
    "ip0 sqrt scan rate slope": "ip0 sqrt(scan rate) slope",
}


_PLATEAU_PARAMETER_SYMBOLS = {
    "catalyst electrons": "n",
    "turnover electrons": "n'",
    "temperature": "T",
    "d": "D",
    "c": "C",
    "electrode area": "S",
    "ip0 scan rate": "ν_ip0",
    "ip0 sqrt scan rate slope": "s_ip0",
}


def _plateau_symbol_display_label(label):
    text = str(label)
    if " / " in text:
        base, unit = text.rsplit(" / ", 1)
        return f"{_plateau_symbol_display_label(base)} / {unit}"
    key = text.strip().lower()
    symbol = _PLATEAU_PARAMETER_SYMBOLS.get(key)
    base = _PLATEAU_PARAMETER_LABELS.get(key, text)
    return f"{base} ({symbol})" if symbol else base


def _plateau_parameter_table(rows):
    return _analysis_parameter_table([
        (
            _PLATEAU_PARAMETER_LABELS.get(str(key).strip().lower(), str(key)),
            _PLATEAU_PARAMETER_SYMBOLS.get(str(key).strip().lower(), ""),
            value,
        )
        for key, value in rows
    ])


def _plateau_summary_display_table(row, selection, warnings_list, options=None):
    rows = [
        ("Formula Mode", row.get("formula mode")),
        ("Plateau Validation", _plateau_validation_status(selection)),
        ("ilim", f"{_format_plateau_display_value('ilim', row.get('ilim'), options)} ({row.get('ilim source')})"),
    ]
    if row.get("ip0") is not None:
        rows.append(
            (
                "ip0",
                f"{_format_plateau_display_value('ip0', row.get('ip0'), options)} ({row.get('ip0 source')})",
            )
        )
    if row.get("ip0 scan rate") is not None:
        rows.append(("ip0 Scan Rate", _format_plateau_display_value("ip0 scan rate", row.get("ip0 scan rate"), options)))
    if row.get("ip0 sqrt scan rate slope") is not None:
        rows.append(
            (
                "ip0 sqrt(scan rate) slope",
                _format_plateau_display_value("ip0 sqrt scan rate slope", row.get("ip0 sqrt scan rate slope"), options),
            )
        )
    if row.get("D") is not None:
        rows.append(("D", _format_plateau_display_value("D", row.get("D"), options)))
    if row.get("C") is not None:
        rows.append(("C", _format_plateau_display_value("C", row.get("C"), options)))
    if row.get("electrode area") is not None:
        rows.append(("Electrode Area", _format_plateau_display_value("electrode area", row.get("electrode area"), options)))
    rows.extend([
        ("Catalyst Electrons", _format_plateau_display_value("catalyst electrons", row.get("catalyst electrons"), options)),
        ("Turnover Electrons", _format_plateau_display_value("turnover electrons", row.get("turnover electrons"), options)),
        ("Temperature", _format_plateau_display_value("temperature", row.get("temperature"), options)),
    ])
    if warnings_list:
        rows.append(("Warnings", " | ".join(str(item) for item in warnings_list)))
    return _plateau_parameter_table(rows)


def _plateau_compact_result_row(row, options=None):
    result = {
        "kobs": _format_plateau_display_value("kobs", row.get("kobs"), options),
        "ilim": _format_plateau_display_value("ilim", row.get("ilim"), options),
    }
    if row.get("ilim tangent"):
        result["ilim tangent"] = row.get("ilim tangent")
    mode = row.get("formula mode")
    if mode == "normalized":
        result["ip0"] = _format_plateau_display_value("ip0", row.get("ip0"), options)
        if row.get("ip0") not in (None, 0):
            result["ilim/ip0"] = _format_plateau_display_value("ilim/ip0", row.get("ilim") / row.get("ip0"), options)
        if row.get("ip0 tangent"):
            result["ip0 tangent"] = row.get("ip0 tangent")
        result["ip0 scan rate"] = _format_plateau_display_value("ip0 scan rate", row.get("ip0 scan rate"), options)
    elif mode == "slope normalized":
        result["ip0 sqrt scan rate slope"] = _format_plateau_display_value(
            "ip0 sqrt scan rate slope",
            row.get("ip0 sqrt scan rate slope"),
            options,
        )
        result["ip0 fit r2"] = _format_plateau_display_value("ip0 fit r2", row.get("ip0 fit r2"), options)
    elif mode == "direct":
        result["D"] = _format_plateau_display_value("D", row.get("D"), options)
        result["C"] = _format_plateau_display_value("C", row.get("C"), options)
        result["electrode area"] = _format_plateau_display_value("electrode area", row.get("electrode area"), options)
    return result


def _plateau_numeric_result_row(row):
    result = {
        "plateau validation": row.get("plateau validation"),
        "formula mode": row.get("formula mode"),
        "catalyst electrons": row.get("catalyst electrons"),
        "turnover electrons": row.get("turnover electrons"),
        "temperature": row.get("temperature"),
        "kobs": row.get("kobs"),
        "ilim": row.get("ilim"),
        "ilim source": row.get("ilim source"),
    }
    if row.get("ilim tangent"):
        result["ilim tangent"] = row.get("ilim tangent")
    mode = row.get("formula mode")
    if mode == "normalized":
        result["ip0"] = row.get("ip0")
        result["ip0 source"] = row.get("ip0 source")
        if row.get("ip0") not in (None, 0):
            result["ilim/ip0"] = row.get("ilim") / row.get("ip0")
        if row.get("ip0 tangent"):
            result["ip0 tangent"] = row.get("ip0 tangent")
        result["ip0 scan rate"] = row.get("ip0 scan rate")
    elif mode == "slope normalized":
        result["ip0 sqrt scan rate slope"] = row.get("ip0 sqrt scan rate slope")
        result["ip0 fit r2"] = row.get("ip0 fit r2")
    elif mode == "direct":
        result["D"] = row.get("D")
        result["C"] = row.get("C")
        result["electrode area"] = row.get("electrode area")
    return result


def _plateau_units_map():
    return {
        "kobs": "s^-1",
        "ilim": "A",
        "ilim tangent": "A/V, A",
        "ip0": "A",
        "ip0 tangent": "A/V, A",
        "ilim/ip0": "",
        "ip0 scan rate": "V/s",
        "ip0 sqrt scan rate slope": "A/(V/s)^1/2",
        "ip0 fit r2": "",
        "D": "cm^2/s",
        "C": "mol/cm^3",
        "electrode area": "cm^2",
        "catalyst electrons": "dimensionless",
        "turnover electrons": "dimensionless",
        "temperature": "K",
    }


def _plateau_result_table(row, options=None, *, transpose=False):
    compact = _plateau_compact_result_row(row, options)
    if transpose:
        return pd.DataFrame(
            [
                {"Metric": _plateau_symbol_display_label(key), "Value": value}
                for key, value in compact.items()
            ]
        )
    table = pd.DataFrame([_plateau_numeric_result_row(row)])
    table.attrs["units"] = _plateau_units_map()
    return table


def _plateau_results_display_table(table, options=None):
    options = {} if options is None else options
    if set(table.columns) == {"Metric", "Value"}:
        return table.copy()
    display_table = _results_display_table_with_unit_headers(
        table,
        options,
        value_formatter=_format_plateau_display_value,
    )
    return display_table.rename(
        columns={
            column: _plateau_symbol_display_label(column)
            for column in display_table.columns
        }
    )


def _display_plateau_table(title, table, options=None):
    options = options or {}
    rich_table = _analysis_parameter_rich_table(table) if "Symbol" in table.columns else None
    return _display_table(
        table,
        options,
        title=title,
        rich_table=rich_table,
        escape=None,
        index=False,
    )


def _display_plateau_results_table(title, table, options=None):
    options = {} if options is None else options
    display_table = _plateau_results_display_table(table, options)
    return _display_table(
        table,
        options,
        title=title,
        rich_table=display_table.rename(columns=_pretty_table_header_html_label),
        plain_table=display_table,
        escape=None,
        index=False,
        plain_title=True,
    )


def _is_plateau_nested_input(cvs):
    if not isinstance(cvs, (list, tuple)) or not cvs:
        return False
    return all(isinstance(item, (list, tuple)) for item in cvs)


def _normalize_plateau_group_mode(value):
    return str(value or "auto").strip().lower().replace("_", " ").replace("-", " ")


def _resolve_plateau_input_groups(cvs, options, *, allow_empty=False):
    """Return grouped catalytic CVs, or None when the input should be one group."""
    if allow_empty or cvs is None:
        return None

    if _is_plateau_nested_input(cvs):
        groups = [list(group_items) for group_items in cvs]
        for group_items in groups:
            _coerce_plateau_cv_list(group_items, allow_empty=False)
        return groups if len(groups) > 1 else None

    cv_list = _coerce_plateau_cv_list(cvs, allow_empty=False)
    if len(cv_list) <= 1:
        return None

    group_mode = _normalize_plateau_group_mode(options.get("group mode", "auto"))
    if group_mode == "as given":
        return None
    if group_mode == "each":
        return [[cv_obj] for cv_obj in cv_list]

    group_by = options.get("group by", "species")
    if group_by in (None, False, "", "none"):
        return None
    from .collection import group as group_objects

    groups = group_objects(cv_list, group_by, {"print": False})
    return groups if len(groups) > 1 else None


def _plateau_condition_label(group_items, index):
    names = [getattr(item, "name", f"CV {i + 1}") for i, item in enumerate(group_items)]
    if len(names) == 1:
        return names[0]
    return f"Condition {index + 1}"


def _plateau_group_compounds_label(group_items):
    if not group_items:
        return ""
    cv_obj = group_items[0]
    compounds = list(getattr(cv_obj, "compounds", []) or [])
    concentrations = list(getattr(cv_obj, "concentrations", []) or [])
    entries = []
    for concentration, compound in zip(concentrations, compounds):
        if concentration in (None, ""):
            entries.append(str(compound))
        else:
            entries.append(f"{concentration} {compound}")
    if not entries:
        entries = [str(compound) for compound in compounds]
    return ", ".join(entries)


def _plateau_group_concentration_columns(groups):
    first_items = [group_items[0] for group_items in groups if group_items]
    if len(first_items) < 2:
        return {}
    varying_entries = _infer_varying_concentration_entries(first_items)
    columns_by_index = {index: {} for index in range(len(groups))}
    for entry in varying_entries:
        species = entry["species"]
        unit = entry["unit"] or "M"
        occurrence = entry["occurrence"]
        x_kind = "Mole Fraction" if unit == "x" else "Concentration"
        column = f"{species} {x_kind} ({unit})"
        for index, group_items in enumerate(groups):
            if not group_items:
                continue
            columns_by_index[index][column] = _get_species_concentration(
                group_items[0],
                species,
                default=0.0,
                unit=unit,
                occurrence=occurrence,
            )
    return columns_by_index


def _plateau_result_row_from_child(child_result, condition, index, group_items=None, concentration_columns=None):
    details = child_result.diagnostics.get("plateau details")
    if isinstance(details, pd.DataFrame) and not details.empty:
        detail_row = details.iloc[0].to_dict()
        row = _plateau_numeric_result_row(detail_row)
        row["cv / cvs used"] = detail_row.get("cv / cvs used", "")
    else:
        table = child_result.table
        if isinstance(table, pd.DataFrame) and set(table.columns) == {"Metric", "Value"}:
            row = dict(zip(table["Metric"], table["Value"]))
        elif isinstance(table, pd.DataFrame) and not table.empty:
            row = table.iloc[0].to_dict()
        else:
            row = {}
        row.setdefault("cv / cvs used", "")
    if group_items:
        row["Compounds"] = _plateau_group_compounds_label(group_items)
    if concentration_columns:
        row.update(concentration_columns)
    return {"condition": condition, **row}


def _plateau_visible_context_column(column):
    key = str(column).strip().lower()
    if key in {"condition", "co2 %"}:
        return True
    if key.startswith("co2 ") and "%" in key:
        return True
    return False


def _order_plateau_display_columns(df, metadata_columns):
    metadata = [col for col in metadata_columns if col in df.columns]
    operation_columns = [
        "Plateau Validation",
        "Formula Mode",
        "Catalyst Electrons",
        "Turnover Electrons",
        "Temperature",
        "ilim",
        "ilim Source",
        "ilim Tangent",
        "ip0",
        "ip0 Source",
        "ip0 Tangent",
        "ilim/ip0",
        "ip0 Scan Rate",
        "ip0 sqrt scan rate slope",
        "ip0 Fit R2",
        "D",
        "C",
        "Electrode Area",
        "kobs",
    ]
    ordered = []
    for column in metadata + operation_columns:
        if column in df.columns and column not in ordered:
            ordered.append(column)
    ordered.extend([column for column in df.columns if column not in ordered])
    return df.loc[:, ordered]


def _plateau_grouped_result_tables(groups, result_rows, options):
    full_result_table = pd.DataFrame(result_rows)
    full_result_table.attrs["units"] = _plateau_units_map()

    representatives = [group_items[0] for group_items in groups if group_items]
    try:
        summary_df, _meta = build_object_table(representatives, options)
    except AttributeError:
        summary_df = pd.DataFrame({
            "Name": [getattr(obj, "name", f"CV {index + 1}") for index, obj in enumerate(representatives)],
            "Plot Label": [getattr(obj, "name", f"CV {index + 1}") for index, obj in enumerate(representatives)],
        })
    analysis_df = pd.DataFrame([
        _plateau_numeric_result_row(row)
        for row in result_rows
    ])
    analysis_df = analysis_df.rename(
        columns={column: pretty_table_column_label(column) for column in analysis_df.columns}
    )

    concentration_columns = [
        column
        for column in full_result_table.columns
        if (
            ("Concentration" in str(column) or "Mole Fraction" in str(column))
            and not _plateau_visible_context_column(column)
        )
    ]
    context_columns = [
        column
        for column in ["Compounds"]
        if column in full_result_table.columns and column not in summary_df.columns
    ]
    context_df = full_result_table.loc[:, context_columns] if context_columns else pd.DataFrame()
    concentration_df = full_result_table.loc[:, concentration_columns] if concentration_columns else pd.DataFrame()

    display_source_df = pd.concat(
        [
            summary_df.reset_index(drop=True),
            context_df.reset_index(drop=True),
            analysis_df.reset_index(drop=True),
            concentration_df.reset_index(drop=True),
        ],
        axis=1,
    )
    display_source_df = display_source_df.drop(
        columns=[
            column
            for column in display_source_df.columns
            if _plateau_visible_context_column(column)
        ],
        errors="ignore",
    )
    display_source_df = _order_plateau_display_columns(display_source_df, summary_df.columns)

    keep_in_table = {"Name", "Plot Label", "Status"}
    display_df, shared_summary = _split_shared_columns(
        display_source_df,
        keep_in_table=keep_in_table,
    )
    identity_columns = [
        column
        for column in [*summary_df.columns, *context_df.columns, *concentration_df.columns]
        if column in display_df.columns and column not in {"Name", "Plot Label"}
    ]
    identity_check = _conditional_analysis_name_column(
        display_df,
        identity_columns,
        options,
    )
    needs_condition = "Name" in identity_check.columns
    display_df = display_df.drop(columns=["Name", "Plot Label"], errors="ignore")
    if needs_condition and "condition" in full_result_table.columns:
        display_df.insert(0, "Condition", full_result_table["condition"].to_numpy())
    display_df.attrs["shared_summary"] = shared_summary
    display_df.attrs["full_results_df"] = full_result_table
    display_df.attrs["units"] = _plateau_units_map()
    full_result_table.attrs["units"] = _plateau_units_map()
    return display_df, full_result_table


def _common_table_value(df, column):
    if not isinstance(df, pd.DataFrame) or df.empty or column not in df:
        return None
    values = [
        value
        for value in df[column].tolist()
        if value is not None and not (isinstance(value, float) and pd.isna(value)) and str(value).strip() != ""
    ]
    if not values:
        return None
    unique = list(dict.fromkeys(str(value) for value in values))
    return unique[0] if len(unique) == 1 else "mixed"


def _format_plateau_summary_value(setting, value, options=None):
    if _is_table_blank(value):
        return ""
    key = str(setting).strip().lower()
    format_key_map = {
        "ip0": "ip0",
        "ip0 scan rate": "ip0 scan rate",
        "ip0 sqrt scan rate slope": "ip0 sqrt scan rate slope",
        "d": "D",
        "c": "C",
        "electrode area": "electrode area",
        "catalyst electrons": "catalyst electrons",
        "turnover electrons": "turnover electrons",
        "temperature": "temperature",
    }
    if key in format_key_map:
        return _format_plateau_display_value(format_key_map[key], value, options)
    if key.endswith("source") or key.endswith("mode") or key == "group by":
        return str(value).replace("_", " ")
    if isinstance(value, (int, float, np.integer, np.floating)):
        return _format_fit_model_display_value(value, sig_figs=(options or {}).get("sig figs", 4))
    if isinstance(value, (list, tuple)):
        return ", ".join(str(item) for item in value)
    return str(value)


def _plateau_grouped_summary_display_table(display_df, full_result_table, group_mode, options, warnings_list):
    shared_summary = dict(getattr(display_df, "attrs", {}).get("shared_summary", {}) or {})
    rows = {
        "Groups": len(full_result_table),
        "Group Mode": group_mode,
        "Group By": options.get("group by", "species"),
        "Formula Mode": _common_table_value(full_result_table, "formula mode"),
        "Plateau Validation": _common_table_value(full_result_table, "plateau validation"),
        "ilim Source": _common_table_value(full_result_table, "ilim source"),
        "ip0 Source": _common_table_value(full_result_table, "ip0 source"),
    }
    for key, value in shared_summary.items():
        rows.setdefault(key, value)
    rows = {
        key: value
        for key, value in rows.items()
        if value is not None and str(value).strip() != ""
    }
    if warnings_list:
        rows["Warnings"] = " | ".join(str(item) for item in warnings_list)
    return _plateau_parameter_table([
        (key, _format_plateau_summary_value(key, value, options))
        for key, value in rows.items()
    ])


def _add_condition_column(table, condition):
    if not isinstance(table, pd.DataFrame) or table.empty:
        return table
    copy = table.copy()
    copy.attrs = {}
    copy.insert(0, "condition", condition)
    return copy


def _combine_plateau_diagnostic_tables(child_results, key, conditions):
    tables = []
    for result, condition in zip(child_results, conditions):
        table = result.diagnostics.get(key)
        if isinstance(table, pd.DataFrame) and not table.empty:
            tables.append(_add_condition_column(table, condition))
    if not tables:
        return pd.DataFrame()
    return pd.concat(tables, ignore_index=True)


def _plateau_child_options(options):
    child_options = dict(options)
    child_options["group mode"] = "as given"
    child_options["print"] = False
    child_options["plot all"] = False
    child_options["plot"] = False
    for alias_key in (
        "non catalytic current",
        "non catalytic cv",
        "non catalytic cvs",
        "non catalytic guess potential",
        "c",
        "c unit",
        "d",
    ):
        child_options.pop(alias_key, None)
    return child_options


def _plateau_ref_cvs_from_options(options):
    shared_ref = options.get("non-catalytic cv")
    ref_list = options.get("non-catalytic cvs")
    if ref_list is not None and not isinstance(ref_list, (list, tuple)):
        return [ref_list]
    if ref_list is not None:
        return list(ref_list)
    if shared_ref is not None:
        return [shared_ref]
    return []


def _unique_cvs_in_order(*collections):
    result = []
    seen = set()
    for collection in collections:
        for item in collection or []:
            if id(item) in seen:
                continue
            result.append(item)
            seen.add(id(item))
    return result


def _plateau_ip0_from_detail(detail_row, cv_obj):
    ip0 = detail_row.get("ip0")
    if ip0 is not None:
        try:
            if np.isfinite(float(ip0)) and float(ip0) != 0:
                return float(ip0)
        except (TypeError, ValueError):
            pass
    slope = detail_row.get("ip0 sqrt scan rate slope")
    scan_rate = getattr(cv_obj, "scan_rate", None)
    try:
        if (
            slope is not None
            and np.isfinite(float(slope))
            and scan_rate is not None
            and np.isfinite(float(scan_rate))
            and float(scan_rate) > 0
        ):
            return abs(float(slope)) * np.sqrt(float(scan_rate))
    except (TypeError, ValueError):
        pass
    return None


def _plateau_diagnostic_call_for_cv(cv_obj, options, row=None, *, role=None, potential=None):
    call_options = dict(options)
    call_options["print"] = False
    call_options["print all"] = False
    call_options["plot"] = True
    call_options["plot all"] = True
    if row is None:
        row = {}
    if potential is not None:
        try:
            if np.isfinite(float(potential)):
                call_options.pop("guess potential", None)
                call_options["exact potential"] = float(potential)
        except (TypeError, ValueError):
            pass
    return {
        "kind": "plateau_extraction",
        "obj": cv_obj,
        "options": call_options,
        "role": role,
        "current": row.get("ic", row.get("ip0", row.get("current"))),
        "potential": row.get("extraction potential", potential),
        "peak index": row.get("peak index"),
        "baseline current": row.get("baseline current"),
        "tangent slope": row.get("tangent slope"),
        "tangent intercept": row.get("tangent intercept"),
        "tangent start": row.get("tangent start"),
        "fit indices": row.get("fit indices"),
    }


def _plateau_grouped_cv_diagnostic(groups, child_results, options):
    ref_cvs = _plateau_ref_cvs_from_options(options)
    cat_cvs = _unique_cvs_in_order(*groups)
    diagnostic_cvs = _unique_cvs_in_order(ref_cvs, cat_cvs)
    if not diagnostic_cvs:
        return []

    manual_ip0 = options.get("ip0")
    if manual_ip0 is None:
        manual_ip0 = options.get("non-catalytic current")

    ip0_by_id = {}
    diagnostic_calls = []
    object_by_name = {getattr(obj, "name", ""): obj for obj in diagnostic_cvs}

    for group_items, child_result in zip(groups, child_results):
        details = child_result.diagnostics.get("plateau details")
        if not isinstance(details, pd.DataFrame) or details.empty:
            continue
        detail_row = details.iloc[0].to_dict()
        for cv_obj in group_items:
            if manual_ip0 is not None:
                ip0_by_id[id(cv_obj)] = float(manual_ip0)
            else:
                ip0_value = _plateau_ip0_from_detail(detail_row, cv_obj)
                if ip0_value is not None:
                    ip0_by_id[id(cv_obj)] = ip0_value

    if ref_cvs:
        for child_result in child_results:
            details = child_result.diagnostics.get("plateau details")
            if not isinstance(details, pd.DataFrame) or details.empty:
                continue
            detail_row = details.iloc[0].to_dict()
            for ref_cv in ref_cvs:
                if manual_ip0 is not None:
                    ip0_by_id[id(ref_cv)] = float(manual_ip0)
                else:
                    ip0_value = _plateau_ip0_from_detail(detail_row, ref_cv)
                    if ip0_value is not None:
                        ip0_by_id[id(ref_cv)] = ip0_value
            break

    for child_result in child_results:
        catalytic = child_result.diagnostics.get("catalytic currents")
        if isinstance(catalytic, pd.DataFrame):
            for _, row in catalytic.iterrows():
                cv_obj = object_by_name.get(row.get("cv"))
                if cv_obj is not None:
                    diagnostic_calls.append(
                        _plateau_diagnostic_call_for_cv(
                            cv_obj,
                            options,
                            row.to_dict(),
                            role="catalytic",
                            potential=row.get("extraction potential"),
                        )
                    )
        ip0_currents = child_result.diagnostics.get("ip0 currents")
        if isinstance(ip0_currents, pd.DataFrame):
            for _, row in ip0_currents.iterrows():
                cv_obj = object_by_name.get(row.get("reference cv"))
                if cv_obj is not None:
                    diagnostic_calls.append(
                        _plateau_diagnostic_call_for_cv(
                            cv_obj,
                            options,
                            row.to_dict(),
                            role="non-catalytic",
                            potential=row.get("extraction potential"),
                        )
                    )

    ip0_values = [ip0_by_id.get(id(cv_obj)) for cv_obj in diagnostic_cvs]
    if any(value is None for value in ip0_values):
        return _plot_plateau_cv_diagnostic(diagnostic_cvs, options, diagnostic_calls=diagnostic_calls)
    return _plot_plateau_cv_diagnostic(
        diagnostic_cvs,
        options,
        ip0_values=ip0_values,
        diagnostic_calls=diagnostic_calls,
    )


def _plateau_ip0_values_for_diagnostic(cvs, ip0=None, ip0_slope=None, ip0_values=None):
    if ip0_values is not None:
        values = [float(value) for value in ip0_values]
        if len(values) != len(cvs):
            return None
        return values
    if ip0 is not None:
        return [float(ip0)] * len(cvs)
    if ip0_slope is None:
        return None
    values = []
    for cv_obj in cvs:
        scan_rate = getattr(cv_obj, "scan_rate", None)
        if scan_rate is None or not np.isfinite(float(scan_rate)) or float(scan_rate) <= 0:
            return None
        values.append(abs(float(ip0_slope)) * np.sqrt(float(scan_rate)))
    return values


def _as_finite_float(value):
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if np.isfinite(numeric) else None


def _as_index_array(value):
    if value is None:
        return np.asarray([], dtype=int)
    try:
        if pd.isna(value):
            return np.asarray([], dtype=int)
    except (TypeError, ValueError):
        pass
    try:
        array = np.asarray(value, dtype=int)
    except (TypeError, ValueError):
        return np.asarray([], dtype=int)
    return array[np.isfinite(array)] if np.issubdtype(array.dtype, np.floating) else array


def _plot_plateau_normalized_extraction_diagnostics(
    ax,
    diagnostic_calls,
    copy_by_original_id,
    object_offsets,
    ip0_by_original_id,
    options,
):
    seen = set()

    for call in diagnostic_calls or []:
        if call.get("kind") != "plateau_extraction":
            continue
        original = call.get("obj")
        obj_copy = copy_by_original_id.get(id(original))
        ip0 = _as_finite_float(ip0_by_original_id.get(id(original)))
        if original is None or obj_copy is None or ip0 in (None, 0):
            continue

        key = (id(original), call.get("role"), call.get("potential"))
        if key in seen:
            continue
        seen.add(key)

        diag_options = dict(call.get("options") or {})
        diag_options["plot"] = False
        diag_options["print"] = False
        normalized_axis_options = dict(diag_options)
        normalized_axis_options["y axis"] = "i/ip0"
        normalized_axis_options["y unit"] = None
        raw_current_options = dict(diag_options)
        raw_current_options["y axis"] = "Current"
        raw_current_options["y unit"] = "A"
        offset = object_offsets.get(id(original), 0)

        try:
            x, y = original.analysis_segment_data(raw_current_options)
        except Exception:
            continue
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        if len(x) == 0 or len(y) == 0:
            continue

        try:
            x_scale, _y_scale = obj_copy.xy_scale(normalized_axis_options)
        except Exception:
            x_scale = 1
        y_scale = 1

        slope = _as_finite_float(call.get("tangent slope"))
        intercept = _as_finite_float(call.get("tangent intercept"))
        peak_index = _as_finite_float(call.get("peak index"))
        tangent_start = _as_finite_float(call.get("tangent start"))
        current = _as_finite_float(call.get("current"))
        baseline = _as_finite_float(call.get("baseline current"))
        potential = _as_finite_float(call.get("potential"))
        source = str(call.get("source", "")).strip().lower().replace("_", " ").replace("-", " ")
        is_exact_anchor = (
            "exact potential" in source
            or (
                source in {"", "peak current", "peak_current"}
                and (call.get("options") or {}).get("exact potential") is not None
            )
        )

        if slope is not None and intercept is not None:
            if peak_index is not None and tangent_start is not None:
                i0, i1 = sorted([int(tangent_start), int(peak_index)])
                i0 = max(i0, 0)
                i1 = min(i1, len(x) - 1)
                x_tangent = x[i0:i1 + 1]
            else:
                fit_idx = _as_index_array(call.get("fit indices"))
                fit_idx = fit_idx[(fit_idx >= 0) & (fit_idx < len(x))]
                x_tangent = x[fit_idx] if len(fit_idx) else np.asarray([])
            if len(x_tangent):
                ax.plot(
                    x_tangent * x_scale,
                    ((slope * x_tangent + intercept) / ip0) * y_scale + offset,
                    linestyle="--",
                    color="tab:red",
                    label="_nolegend_",
                )

        if potential is not None and current is not None and baseline is not None:
            ax.vlines(
                potential * x_scale,
                (baseline / ip0) * y_scale + offset,
                ((baseline + current) / ip0) * y_scale + offset,
                color="tab:red",
                linestyle="--",
                label="_nolegend_",
            )
            if not is_exact_anchor:
                ax.scatter(
                    potential * x_scale,
                    ((baseline + current) / ip0) * y_scale + offset,
                    color="tab:blue",
                    zorder=3,
                    label="_nolegend_",
                )
        elif peak_index is not None and not is_exact_anchor:
            idx = int(peak_index)
            if 0 <= idx < len(x):
                ax.scatter(
                    x[idx] * x_scale,
                    (y[idx] / ip0) * y_scale + offset,
                    color="tab:blue",
                    zorder=3,
                    label="_nolegend_",
                )

        fit_idx = _as_index_array(call.get("fit indices"))
        fit_idx = fit_idx[(fit_idx >= 0) & (fit_idx < len(x))]
        if len(fit_idx):
            ax.scatter(
                x[fit_idx] * x_scale,
                (y[fit_idx] / ip0) * y_scale + offset,
                s=10,
                color="tab:red",
                zorder=3,
                label="_nolegend_",
            )


def _plot_plateau_cv_diagnostic(
    cvs,
    options,
    *,
    ip0=None,
    ip0_slope=None,
    ip0_values=None,
    diagnostic_calls=None,
):
    if not cvs:
        return []
    plot_options = _multiplot_options_from_mapping(options)
    plot_options.update(_common_cv_plot_axis_options(cvs, plot_options))
    plot_options.update({"plot": True, "print": False, "legend": True})
    requested_axis = str(options.get("diagnostic y axis", "i/ip0")).strip().lower().replace(" ", "")
    warnings_list = []

    if requested_axis == "i/ip0":
        resolved_ip0_values = _plateau_ip0_values_for_diagnostic(
            cvs,
            ip0=ip0,
            ip0_slope=ip0_slope,
            ip0_values=ip0_values,
        )
        if resolved_ip0_values is not None:
            try:
                normalized = [
                    _copy_cv_with_normalized_current_axis(cv_obj, ip0_value, options)
                    for cv_obj, ip0_value in zip(cvs, resolved_ip0_values)
                ]
                plot_options["y axis"] = "i/ip0"
                plot_options["y unit"] = None
                plot_options["ylabel"] = "$i / i_p^0$"
                plot_options["plot all"] = False
                multiplot(normalized, plot_options)
                ax = plt.gca()
                copy_by_original_id = {
                    id(original): copy
                    for original, copy in zip(cvs, normalized)
                }
                object_offsets = {
                    id(obj): plot_options.get("offset", 0) * index
                    for index, obj in enumerate(cvs)
                }
                ip0_by_original_id = {
                    id(obj): ip0_value
                    for obj, ip0_value in zip(cvs, resolved_ip0_values)
                }
                if diagnostic_calls:
                    _plot_plateau_normalized_extraction_diagnostics(
                        ax,
                        diagnostic_calls,
                        copy_by_original_id,
                        object_offsets,
                        ip0_by_original_id,
                        options,
                    )
                    _plot_fowa_normalized_diagnostics(
                        ax,
                        diagnostic_calls,
                        copy_by_original_id,
                        object_offsets,
                        options,
                    )
                _disable_scientific_offset(ax, axis="y")
                return warnings_list
            except Exception:
                warnings_list.append("i/ip0 diagnostic requested but current normalization failed; plotted current.")
        else:
            warnings_list.append("i/ip0 diagnostic requested but ip0 could not be resolved")

    plot_options["y axis"] = "Current"
    plot_options.pop("ylabel", None)
    if plot_options.get("y unit") is None:
        plot_options["y unit"] = "auto"
    plot_options["plot all"] = False
    multiplot(cvs, plot_options)
    _disable_scientific_offset(plt.gca(), axis="y")
    return warnings_list


def _plateau_current_grouped(groups, raw_options, options, *, group_mode=None):
    group_mode = group_mode or _normalize_plateau_group_mode(options.get("group mode", "auto"))
    child_options = _plateau_child_options(options)
    child_results = []
    conditions = []
    for index, group_items in enumerate(groups):
        condition = _plateau_condition_label(group_items, index)
        conditions.append(condition)
        child_results.append(plateau_current(group_items, child_options))

    concentration_columns = _plateau_group_concentration_columns(groups)
    result_rows = [
        _plateau_result_row_from_child(
            child_result,
            condition,
            index,
            group_items=group_items,
            concentration_columns=concentration_columns.get(index, {}),
        )
        for index, (child_result, condition, group_items) in enumerate(zip(child_results, conditions, groups))
    ]
    result_table, full_result_table = _plateau_grouped_result_tables(groups, result_rows, options)
    details = _combine_plateau_diagnostic_tables(child_results, "plateau details", conditions)
    catalytic = _combine_plateau_diagnostic_tables(child_results, "catalytic currents", conditions)
    ip0_currents = _combine_plateau_diagnostic_tables(child_results, "ip0 currents", conditions)
    warnings_list = list(dict.fromkeys(
        warning
        for child_result in child_results
        for warning in child_result.warnings
    ))

    if options.get("plot all", False):
        warnings_list.extend(
            warning
            for warning in _plateau_grouped_cv_diagnostic(groups, child_results, options)
            if warning not in warnings_list
        )
        for child_result in child_results:
            catalytic_table = child_result.diagnostics.get("catalytic currents")
            selection = child_result.diagnostics.get("plateau selection")
            if isinstance(catalytic_table, pd.DataFrame) and isinstance(selection, dict) and len(catalytic_table) > 1:
                _plot_plateau_validation(catalytic_table, selection, options)

    if options.get("print", True):
        _display_plateau_table(
            "Plateau Current Parameters",
            _plateau_grouped_summary_display_table(
                result_table,
                full_result_table,
                group_mode,
                options,
                warnings_list,
            ),
            options,
        )
        formula_mode = _common_table_value(full_result_table, "formula mode")
        if formula_mode and formula_mode != "mixed":
            _display_plateau_kobs_equation(
                formula_mode,
                full_result_table.iloc[0].to_dict(),
                resolved=False,
                compact=options.get("print all", False),
            )
        _display_plateau_results_table("Plateau Current Summary", result_table, options)
        if options.get("print all", False):
            if not details.empty:
                _display_plateau_table("Plateau Current Data", details, options)

    return analysis_result_from_table(
        result_table,
        analysis="plateau_current",
        summary={
            "groups": len(groups),
            "group mode": group_mode,
            "group by": options.get("group by", "species"),
        },
        diagnostics={
            "groups": child_results,
            "full results": full_result_table,
            "plateau details": details,
            "catalytic currents": catalytic,
            "ip0 currents": ip0_currents,
        },
        warnings=warnings_list,
        display_table_formatter=_plateau_results_display_table,
    )


def _disable_scientific_offset(ax, axis="y"):
    try:
        ax.ticklabel_format(axis=axis, style="plain", useOffset=False)
    except Exception:
        return


def _scale_array_for_display_unit(values, unit):
    array = np.asarray(values, dtype=float)
    finite = array[np.isfinite(array)]
    if finite.size == 0 or not unit:
        return array, unit
    reference = float(np.nanmax(np.abs(finite)))
    if reference == 0:
        return array, unit
    try:
        scaled_reference, display_unit = scale_value(reference, unit, selected_unit="auto")
    except Exception:
        return array, unit
    factor = float(scaled_reference) / reference
    return array * factor, display_unit


def _scale_value_to_display_unit(value, source_unit, display_unit):
    try:
        scaled, _unit = scale_value(float(value), source_unit, selected_unit=display_unit)
        return float(scaled)
    except Exception:
        return float(value)


def _plot_plateau_validation(ic_df, selection, options):
    if ic_df is None or ic_df.empty or len(ic_df) < 2:
        return
    plt.figure()
    ax = plt.gca()
    x = ic_df["sqrt scan rate"].astype(float).to_numpy()
    y = ic_df["abs ic"].astype(float).to_numpy()
    y, y_unit = _scale_array_for_display_unit(y, "A")
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
    plt.ylabel(f"|i_c| / {_format_plain_unit_label(y_unit)}")
    plt.title("Plateau-current validation")
    plt.legend()
    _disable_scientific_offset(ax, axis="y")


def _plot_ip0_sqrt_fit(ip0_df, slope, options):
    if ip0_df is None or ip0_df.empty or len(ip0_df) < 2:
        return
    plt.figure()
    ax = plt.gca()
    x = ip0_df["sqrt scan rate"].astype(float).to_numpy()
    y = ip0_df["abs ip0"].astype(float).to_numpy()
    y, y_unit = _scale_array_for_display_unit(y, "A")
    plt.scatter(x, y, color=options.get("color", "black"))
    line_x = np.linspace(0, np.max(x), 100)
    slope_y = _scale_value_to_display_unit(abs(slope), "A", y_unit)
    plt.plot(line_x, slope_y * line_x, color="tab:red", linestyle="--")
    plt.xlabel("sqrt(scan rate) / (V/s)^1/2")
    plt.ylabel(f"|i_p0| / {_format_plain_unit_label(y_unit)}")
    plt.title("ip0 vs sqrt(scan rate)")
    _disable_scientific_offset(ax, axis="y")


def plateau_current(cvs, options=None):
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
    raw_options = options
    typed_options = PlateauCurrentOptions.from_options(options)
    options = typed_options.to_options_dict()
    _apply_diagnostic_y_axis_alias(raw_options, options, "plateau_current")
    manual_ilim = _resolve_manual_ilim(options)
    explicit_nested = _is_plateau_nested_input(cvs)
    groups = _resolve_plateau_input_groups(cvs, options, allow_empty=manual_ilim is not None)
    if groups is not None:
        return _plateau_current_grouped(
            groups,
            raw_options,
            options,
            group_mode="nested" if explicit_nested else _normalize_plateau_group_mode(options.get("group mode", "auto")),
        )
    cat_cvs = _coerce_plateau_cv_list(cvs, allow_empty=manual_ilim is not None)
    cat_potential_series = _resolve_complex_potential_series_map(
        raw_options,
        options,
        n_cvs=len(cat_cvs),
        analysis_name="plateau_current",
        option_names=["guess potential", "exact potential", "tangent potential"],
    ) if cat_cvs else None

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
    ref_potential_series = _resolve_complex_potential_series_map(
        raw_options,
        options,
        n_cvs=len(ref_cvs),
        analysis_name="plateau_current",
        option_names=[
            "guess potential",
            "exact potential",
            "tangent potential",
            "non-catalytic guess potential",
        ],
    ) if ref_cvs else None

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
    ref_extraction_options = options
    cat_extraction_options = options
    single_overlay = bool(options.get("plot all", False) and len(cat_cvs) == 1 and len(ref_cvs) == 1)
    diagnostic_warnings = []
    peak_diagnostic_calls = []
    normalized_diagnostic_requested = (
        bool(options.get("plot all", False))
        and str(options.get("diagnostic y axis", "i/ip0")).strip().lower().replace(" ", "") == "i/ip0"
    )

    if single_overlay:
        try:
            combined_cvs = ref_cvs + cat_cvs
            combined_plot_options = _multiplot_options_from_mapping(options)
            combined_axis_options = _common_cv_plot_axis_options(combined_cvs, combined_plot_options)
            ref_extraction_options = dict(options)
            ref_extraction_options.update(combined_axis_options)
            cat_extraction_options = dict(options)
            cat_extraction_options.update(combined_axis_options)
        except Exception:
            ref_extraction_options = options
            cat_extraction_options = options

    if normalized_diagnostic_requested:
        ref_extraction_options = dict(ref_extraction_options)
        ref_extraction_options["_plot_peak_diagnostics"] = False
        ref_extraction_options["_diagnostic_calls"] = peak_diagnostic_calls
        cat_extraction_options = dict(cat_extraction_options)
        cat_extraction_options["_plot_peak_diagnostics"] = False
        cat_extraction_options["_diagnostic_calls"] = peak_diagnostic_calls

    if ip0 is None and ip0_slope is None and ref_cvs:
        if options.get("plot all", False) and not single_overlay and not normalized_diagnostic_requested:
            try:
                ref_plot_options = _multiplot_options_from_mapping(options)
                ref_plot_options.update(_common_cv_plot_axis_options(ref_cvs, ref_plot_options))
                ref_extraction_options = dict(options)
                ref_extraction_options.update(
                    _common_cv_plot_axis_options(ref_cvs, ref_plot_options)
                )
                ref_plot_options.update({"plot": True, "print": False, "legend": True})
                multiplot(ref_cvs, ref_plot_options)
            except Exception:
                pass
        ip0_df = _extract_ip0_currents(ref_cvs, ref_extraction_options, ref_potential_series)
        if len(ip0_df) > 1:
            x = ip0_df["sqrt scan rate"].astype(float).to_numpy()
            y = ip0_df["abs ip0"].astype(float).to_numpy()
            ip0_slope, _pred, ip0_fit_r2, _resid = _plateau_fit_forced_origin(x, y)
            ip0_source = "non-catalytic cvs"
            if options.get("plot all", False) and not normalized_diagnostic_requested:
                _plot_ip0_sqrt_fit(ip0_df, ip0_slope, options)
        else:
            ip0 = float(ip0_df["ip0"].iloc[0])
            ip0_scan_rate = float(ip0_df["scan rate"].iloc[0])
            ip0_source = "non-catalytic cv"

    if options.get("plot all", False) and cat_cvs and not single_overlay:
        try:
            cat_plot_options = _multiplot_options_from_mapping(options)
            cat_axis_options = _common_cv_plot_axis_options(cat_cvs, cat_plot_options)
            cat_extraction_options = dict(cat_extraction_options)
            cat_extraction_options.update(cat_axis_options)
        except Exception:
            pass

    ic_df = pd.DataFrame()
    if manual_ilim is None:
        ic_df = _extract_catalytic_currents(cat_cvs, cat_extraction_options, cat_potential_series)
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
            "validation status": "manual",
            "warnings": [],
        }
        ilim_source = "manual"

    if options.get("plot all", False) and cat_cvs:
        diagnostic_cvs = _unique_cvs_in_order(ref_cvs, cat_cvs) if normalized_diagnostic_requested else cat_cvs
        diagnostic_warnings.extend(
            _plot_plateau_cv_diagnostic(
                diagnostic_cvs,
                options,
                ip0=ip0,
                ip0_slope=ip0_slope,
                diagnostic_calls=peak_diagnostic_calls,
            )
        )

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
    ilim_tangent = _plateau_common_table_value(
        ic_df,
        "ilim tangent",
        selection.get("accepted indices"),
    )
    ip0_tangent = _plateau_common_table_value(ip0_df, "ip0 tangent")

    row = {
        "cv / cvs used": ", ".join([getattr(item, "name", "CV") for item in cat_cvs]),
        "ilim": ilim,
        "ilim source": ilim_source,
        "ilim tangent": ilim_tangent,
        "ilim average method": options.get("plateau average method"),
        "valid plateau": selection["valid plateau"],
        "plateau validation": _plateau_validation_status(selection),
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
        "ip0 tangent": ip0_tangent,
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
        "temperature": temperature,
        "kobs": kobs,
    }
    warnings_list = list(dict.fromkeys(
        str(item)
        for item in [*diagnostic_warnings, *selection.get("warnings", [])]
        if item
    ))
    if ilim_source == "manual":
        warnings_list.append("ilim was provided manually; catalytic current extraction was not performed.")
    if ip0_source == "manual":
        warnings_list.append("ip0 was provided manually; non-catalytic CV extraction was not performed.")
    if ip0_source == "non-catalytic cv" and ip0_scan_rate is None:
        warnings_list.append("ip0 scan rate is unavailable; normalized formula mode cannot be used.")
    warnings_list = list(dict.fromkeys(warnings_list))

    details_df = pd.DataFrame([row])
    details_df.attrs["catalytic currents"] = ic_df
    details_df.attrs["ip0 currents"] = ip0_df
    result_table = _plateau_result_table(
        row,
        options,
        transpose=len(cat_cvs) <= 1,
    )

    if options.get("print", True):
        _display_plateau_table(
            "Plateau Current Parameters",
            _plateau_summary_display_table(row, selection, warnings_list, options),
            options,
        )
        _display_plateau_kobs_equation(
            mode,
            row,
            resolved=False,
            compact=options.get("print all", False),
        )
        _display_plateau_results_table("Plateau Current Summary", result_table, options)
        if options.get("print all", False):
            _display_plateau_table("Plateau Current Data", details_df, options)

    if (options.get("plot all", False) or options.get("plot", True)) and len(ic_df) > 1:
        _plot_plateau_validation(ic_df, selection, options)
    if not options.get("plot all", False) and options.get("plot", True) and len(ip0_df) > 1:
        _plot_ip0_sqrt_fit(ip0_df, ip0_slope, options)

    return analysis_result_from_table(
        result_table,
        analysis="plateau_current",
        summary={
            "formula mode": mode,
            "formula": formula,
            "plateau validation": _plateau_validation_status(selection),
            "valid plateau": selection["valid plateau"],
            "ilim": ilim,
            "ilim source": ilim_source,
            "ip0": ip0,
            "ip0 source": ip0_source,
            "ip0 scan rate": ip0_scan_rate,
            "ip0 sqrt scan rate slope": ip0_slope,
            "ip0 fit r2": ip0_fit_r2,
            "D": D,
            "C": C,
            "electrode area": electrode_area,
            "catalyst electrons": float(options.get("catalyst electrons", 1)),
            "turnover electrons": float(options.get("turnover electrons", 1)),
            "temperature": temperature,
            "kobs": kobs,
        },
        diagnostics={
            "plateau details": details_df,
            "catalytic currents": ic_df,
            "ip0 currents": ip0_df,
            "plateau selection": selection,
        },
        warnings=warnings_list,
        display_table_formatter=_plateau_results_display_table,
    )


def _resolve_fowa_formula(cv_obj, slope, options):
    """
    Resolve the FOWA slope into kinetic quantities.

    Default EC'-type expression:

        x_FOWA = [1 + exp(n*F/(RT) * (E - E_ref))]^-1

        i/ip0 = slope*x_FOWA + intercept

        slope = n'^sigma / (0.4463*n)
                * sqrt(RT*kobs / (n*F*v))

    Therefore:

        kobs = (slope * 0.4463*n / n'^sigma)^2
               * (n*F*v)/(RT)

    where:
        n = n_cat = catalyst electrons
            Electron count for the non-catalytic catalyst wave used for ip0
            and for the FOWA x-axis exponent.

        n' = n_turn = turnover electrons
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


def _normalize_fowa_redox_mode_label(mode):
    if mode is None:
        return None
    text = str(mode).strip().lower().replace("_", " ").replace("-", " ")
    if text in {"half wave", "e1/2", "e 1/2"}:
        return "half wave"
    if text in {"half peak", "ep/2", "e p/2"}:
        return "half peak"
    if text == "manual":
        return "manual"
    return None


def _fowa_reference_potential_symbol(redox_modes):
    modes = {
        mode
        for mode in (
            _normalize_fowa_redox_mode_label(mode)
            for mode in redox_modes
        )
        if mode is not None
    }

    if modes == {"half wave"}:
        return r"E_{1/2}"
    if modes == {"half peak"}:
        return r"E_{p/2}"
    if modes == {"manual"}:
        return r"E_{\mathrm{redox}}"
    return r"E_{\mathrm{ref}}"


def _format_fowa_x_axis_label(plot_data=None, options=None):
    plot_data = [] if plot_data is None else list(plot_data)
    options = {} if options is None else options
    redox_modes = [pdata.get("redox mode") for pdata in plot_data if isinstance(pdata, dict)]
    if not redox_modes:
        redox_modes = [options.get("redox mode")]
    reference_symbol = _fowa_reference_potential_symbol(redox_modes)
    return (
        r"$\left[1+\exp\left("
        r"\frac{nF}{RT}"
        rf"(E-{reference_symbol})"
        r"\right)\right]^{-1}$"
    )


def _plot_fowa_transformed(plot_data, results_df, options):
    echem_list = [pdata["cat cv"] for pdata in plot_data]
    plot_options = _multiplot_options_from_mapping(options)
    style = _prepare_multiplot_style(echem_list, plot_options)
    ax = style["ax"]
    color_spec = style["color spec"]

    for i, pdata in enumerate(plot_data):
        color = color_spec["line colors"][i]

        ax.plot(
            pdata["x fowa"],
            pdata["y fowa"],
            color=color,
            label=format_chemical_formulas(color_spec["labels"][i]),
        )

        if len(pdata["x fit"]):
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

        has_fit = np.isfinite(slope) and np.isfinite(intercept) and len(pdata["x fit"])
        if not has_fit:
            continue
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
    ax.set_xlabel(_format_fowa_x_axis_label(plot_data, options))

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


def _format_fowa_line(slope, intercept, sig_figs=4):
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
    return (
        f"y = {_format_sevcik_value(float(slope), sig_figs=sig_figs)}x "
        f"{sign} {_format_sevcik_value(abs(float(intercept)), sig_figs=sig_figs)}"
    )


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
                ecat_options["guess potential"] = None
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
            ecat_options["guess potential"] = None
            ecat_options["exact potential"] = float(cat_Ep)
        except Exception as exc:
            raise ValueError(
                "Could not determine an anchor potential for catalytic Ecat/2. "
                "Try setting 'exact potential', 'guess potential', or disabling "
                "'ecat shift warning threshold'."
            ) from exc

    return cat_cv.half_peak_potential(ecat_options)


def _analysis_options_for(option_cls, source):
    routed = _project_options(option_cls, source).to_options_dict()
    routed.pop("plot segment", None)
    routed.pop("plot segments", None)
    return routed


_COMPLEX_POTENTIAL_PLURAL_KEYS = {
    "guess potential": "guess potentials",
    "exact potential": "exact potentials",
    "tangent potential": "tangent potentials",
    "peak potential": "peak potentials",
    "non-catalytic guess potential": "non-catalytic guess potentials",
    "redox potential": "redox potentials",
}


def _mapping_from_options(options):
    if options is None:
        return {}
    if isinstance(options, dict):
        return options
    if hasattr(options, "to_options_dict"):
        return options.to_options_dict()
    return {}


def _raw_option_value(raw_options, option_name):
    option_key = normalize_key(option_name)
    for key, value in _mapping_from_options(raw_options).items():
        if normalize_key(key) == option_key:
            return True, value, key
    return False, None, None


def _is_option_sequence(value):
    return isinstance(value, (list, tuple, np.ndarray, pd.Series))


def _as_option_list(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, pd.Series):
        return value.tolist()
    return list(value)


def _is_pair_sequence(value):
    return _is_option_sequence(value) and len(_as_option_list(value)) == 2


def _format_option_name_for_error(option_name):
    return str(option_name).replace("_", " ")


def _resolve_complex_potential_series(
    raw_options,
    options,
    *,
    n_cvs,
    option_name,
    analysis_name,
    paired=False,
    allow_none=True,
):
    """Resolve scalar/list potential options to one value per CV."""
    canonical = _format_option_name_for_error(option_name)
    plural = _COMPLEX_POTENTIAL_PLURAL_KEYS.get(canonical, f"{canonical}s")

    has_singular, singular_value, singular_key = _raw_option_value(raw_options, canonical)
    has_plural, plural_value, plural_key = _raw_option_value(raw_options, plural)

    if has_singular and has_plural:
        raise OptionError(
            f"Use either '{singular_key}' or '{plural_key}' for {analysis_name}, not both."
        )

    if has_plural:
        value = plural_value
    elif has_singular:
        value = singular_value
    else:
        value = options.get(canonical)

    if value is None:
        if allow_none:
            return [None] * int(n_cvs)
        raise ValueError(f"'{canonical}' cannot be None for {analysis_name}.")

    if not _is_option_sequence(value):
        return [value] * int(n_cvs)

    values = _as_option_list(value)

    if paired:
        if (
            len(values) == 1
            and _is_pair_sequence(values[0])
            and not any(_is_option_sequence(item) for item in _as_option_list(values[0]))
        ):
            return [_as_option_list(values[0]) for _ in range(int(n_cvs))]

        if (
            canonical == "guess potential"
            and len(values) == 2
            and int(n_cvs) != 2
            and not any(_is_option_sequence(item) for item in values)
        ):
            return [list(values) for _ in range(int(n_cvs))]

    if len(values) != int(n_cvs):
        raise ValueError(
            f"'{canonical}' for {analysis_name} expected 1 scalar value or "
            f"{n_cvs} scalar values (one per CV), "
            f"but received {len(values)} entries."
        )

    return values


def _resolve_complex_potential_series_map(
    raw_options,
    options,
    *,
    n_cvs,
    analysis_name,
    option_names,
    paired=False,
):
    return {
        option_name: _resolve_complex_potential_series(
            raw_options,
            options,
            n_cvs=n_cvs,
            option_name=option_name,
            analysis_name=analysis_name,
            paired=paired,
        )
        for option_name in option_names
    }


def _apply_resolved_potential_options(target_options, potential_series, cv_index):
    for option_name, values in potential_series.items():
        value = values[cv_index]
        if value is None:
            target_options.pop(option_name, None)
        else:
            target_options[option_name] = value
    return target_options


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
            f"'{option_name}' expected 1 range or {n_items} ranges "
            f"(one per catalytic CV), but received {len(values)} entries."
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
                f"'{option_name}' expected 1 scalar value or {n_items} scalar values "
                f"(one per catalytic CV), but received {len(values)} entries."
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
    objects_with_direct_peak_marker = {
        id(call["obj"])
        for call in diagnostic_calls
        if call.get("kind") in {"half_wave", "peak_potential"}
    }

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
        diag_options = _analysis_options_for(option_model, call["options"])
        diag_options["y axis"] = "i/ip0"
        diag_options["y unit"] = None
        diag_options["ylabel"] = "$i / i_p^0$"
        diag_options["plot"] = True
        diag_options["plot all"] = True
        diag_options["print"] = False
        diag_options["print all"] = False
        diag_options["internal call"] = True
        diag_options["new plot"] = False
        diag_options["plot cv"] = False
        diag_options["offset"] = object_offsets.get(id(original), 0)
        if (
            call["kind"] == "peak_current"
            and id(original) in objects_with_direct_peak_marker
        ):
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


def _format_fowa_manual_current_summary(value):
    if _is_option_sequence(value):
        values = [item for item in _as_option_list(value) if item is not None]
        numeric_values = []
        for item in values:
            try:
                numeric_values.append(float(item))
            except (TypeError, ValueError):
                return "manual (per CV)"
        if numeric_values and len({round(item, 18) for item in numeric_values}) == 1:
            return f"manual ({numeric_values[0]:.6g})"
        return "manual (per CV)"
    return f"manual ({float(value):.6g})"


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


def fowa(cvs, options=None):
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
    raw_options = options
    typed_options = FOWAOptions.from_options(options)
    options = typed_options.to_options_dict()
    _apply_diagnostic_y_axis_alias(raw_options, options, "fowa")


    cvs = _coerce_cv_list(cvs)
    ref_cvs = _resolve_non_catalytic_cvs(cvs, options)

    manual_ip0_values = _resolve_manual_ip0_values(options, len(cvs))
    potential_series = _resolve_complex_potential_series_map(
        raw_options,
        options,
        n_cvs=len(cvs),
        analysis_name="fowa",
        option_names=[
            "guess potential",
            "exact potential",
            "tangent potential",
            "peak potential",
            "non-catalytic guess potential",
            "redox potential",
        ],
    )
    manual_redox_values = potential_series["redox potential"]

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
    analysis_segments, segment_selection = _resolve_auto_single_analysis_segment(
        cvs,
        raw_options,
        options,
        method_name="peak_potential",
        analysis_name="fowa",
        default=1,
    )

    # for now, FOWA analyzes one segment at a time; use the first requested segment
    fowa_segment = analysis_segments[0]
    options["segment"] = fowa_segment
    options["segments"] = None

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

    fowa_plot_labels = options.get("labels")
    if fowa_plot_labels is not None:
        if len(fowa_plot_labels) != len(cvs):
            raise ValueError("'labels' must match the number of catalytic CVs passed to FOWA.")
        explicit_diagnostic_labels = {
            id(ref_cv): getattr(ref_cv, "name", f"Trace {i + 1}")
            for i, ref_cv in enumerate(ref_cvs)
            if ref_cv is not None
        }
        explicit_diagnostic_labels.update(
            {id(cat_cv): lbl for cat_cv, lbl in zip(cvs, fowa_plot_labels)}
        )
        raw_plot_options["labels"] = [
            explicit_diagnostic_labels.get(id(obj), getattr(obj, "name", f"Trace {i + 1}"))
            for i, obj in enumerate(all_cvs)
        ]

    raw_plot_options.update(_common_cv_plot_axis_options(all_cvs, raw_plot_options))

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
        _apply_resolved_potential_options(
            loop_options,
            {
                key: values
                for key, values in potential_series.items()
                if key != "non-catalytic guess potential"
            },
            i,
        )

        internal_options = loop_options.copy()
        internal_options["plot"] = False
        internal_options["plot all"] = False
        internal_options["print"] = options.get("print all", False)
        internal_options["internal call"] = True
        internal_options["new plot"] = False

        nc_guess = potential_series["non-catalytic guess potential"][i]
        if nc_guess is None:
            nc_guess = potential_series["guess potential"][i]
        if nc_guess is None:
            nc_guess = manual_redox_values[i]
        if nc_guess is not None and internal_options.get("exact potential") is None:
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
            x_scale, y_scale = cat_cv.xy_scale(raw_plot_options)
            
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
            x_scale, _ = cat_cv.xy_scale(raw_plot_options)
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
                x_scale, y_scale = cat_cv.xy_scale(raw_plot_options)
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

        fit_enabled = bool(options.get("fit", True))
        fit_mask = np.zeros_like(x_fowa, dtype=bool)
        fit_basis_label = f"{fit_basis}_fowa"
        fit_region_meta = {}
        n_fit = 0
        x_fit = np.asarray([], dtype=float)
        y_fit = np.asarray([], dtype=float)
        y_pred = np.asarray([], dtype=float)
        slope = np.nan
        intercept = np.nan
        r2 = np.nan
        formula_label = "not fit"
        kinetics = {"kobs": np.nan, "TOFmax": np.nan}
        fit_succeeded = False

        if fit_enabled:
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
                msg = (
                    f"Only {n_fit} points fall in the requested FOWA fit range "
                    f"[{fit_lo}, {fit_hi}] for '{cat_cv.name}'; fit skipped for this CV."
                )
                _record_fowa_issue(
                    row_warnings,
                    row_status,
                    "fit skipped",
                    msg,
                    options,
                )
            else:
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

                try:
                    slope, intercept = np.polyfit(x_fit, y_fit, 1)
                    y_pred = slope * x_fit + intercept
                    r2 = float(r2_score(y_fit, y_pred))
                    if not all(np.isfinite(value) for value in (slope, intercept, r2)):
                        raise ValueError("the regression returned non-finite fit statistics")
                except (FloatingPointError, ValueError, np.linalg.LinAlgError) as exc:
                    slope = np.nan
                    intercept = np.nan
                    r2 = np.nan
                    x_fit = np.asarray([], dtype=float)
                    y_fit = np.asarray([], dtype=float)
                    y_pred = np.asarray([], dtype=float)
                    msg = f"FOWA fit failed for '{cat_cv.name}' ({exc}); fit skipped for this CV."
                    _record_fowa_issue(
                        row_warnings,
                        row_status,
                        "fit skipped",
                        msg,
                        options,
                    )
                else:
                    fit_succeeded = True

            if fit_succeeded:
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
        else:
            row_status.append("not fit")

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
    _attach_segment_selection_to_table(display_df, segment_selection)
    shared_summary = display_df.attrs.get("shared_summary", {})

    if options.get("print", True):
        combined_summary = {}

        # Keep the manually-added structural info you liked
        combined_summary["Segment"] = fowa_segment
        formatted_selection = _format_segment_selection(segment_selection)
        if formatted_selection:
            combined_summary["Segment Selection"] = formatted_selection
        combined_summary["Fit"] = "enabled" if options.get("fit", True) else "disabled"
        if options.get("fit", True):
            combined_summary["Fit Range"] = f"[{fit_lo}, {fit_hi}]"
        combined_summary["Background Correction"] = background_mode if background_mode is not None else "none"
        combined_summary["Mechanism"] = options.get("mechanism", "EC'")
        combined_summary["Catalyst Electrons"] = _format_fit_model_display_value(
            options.get("catalyst electrons", options.get("num electrons", 1)),
            sig_figs=options.get("sig figs", 4),
        )
        combined_summary["Turnover Electrons"] = _format_fit_model_display_value(
            options.get("turnover electrons", 1),
            sig_figs=options.get("sig figs", 4),
        )
        combined_summary["Sigma"] = _format_fit_model_display_value(
            options.get("sigma", 1.0),
            sig_figs=options.get("sig figs", 4),
        )

        # If the source was manual, include the explicit manual value
        if options.get("non-catalytic current") is not None:
            combined_summary["ip0 Source"] = _format_fowa_manual_current_summary(
                options["non-catalytic current"]
            )

        if any(value is not None for value in manual_redox_values):
            manual_redox_floats = [
                float(value)
                for value in manual_redox_values
                if value is not None
            ]
            if (
                manual_redox_floats
                and len({round(value, 12) for value in manual_redox_floats}) == 1
            ):
                combined_summary["Redox Source"] = f"manual ({manual_redox_floats[0]:.6g} V)"
            else:
                combined_summary["Redox Source"] = "manual (per CV)"

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
            "Segment Selection",
            "Background Correction",
            "Background Tangent Potential",
            "Fit",
            "Fit Range",
            "Mechanism",
            "Catalyst Electrons",
            "Turnover Electrons",
            "Sigma",
        ]

        ordered_summary = {}
        for key in preferred_order:
            if key in combined_summary:
                ordered_summary[key] = combined_summary[key]
        for key, value in combined_summary.items():
            if key not in preferred_order:
                ordered_summary[key] = value

        _display_fowa_summary_table(
            ordered_summary,
            options,
            title="FOWA Parameters",
            plain_title=True,
        )

        if any(np.isfinite(float(row.get("kobs", np.nan))) for row in results):
            _display_fowa_kobs_equation(
                options,
                resolved=False,
                compact=False,
            )

        _display_fowa_results_table(
            display_df,
            options,
            title="FOWA Summary",
            plain_title=True,
        )
        if options.get("print all", False):
            full_results = display_df.attrs.get("full_results_df")
            if isinstance(full_results, pd.DataFrame) and not full_results.empty:
                _display_fowa_results_table(
                    full_results,
                    options,
                    title="FOWA Data",
                    plain_title=True,
                )


    if options.get("plot", True):
        _plot_fowa_transformed(
            plot_data,
            display_df.attrs.get("full_results_df", display_df),
            options,
        )

    if len(cvs) == 1:
        single_df = display_df.iloc[[0]].reset_index(drop=True)
        single_df.attrs.update(display_df.attrs)
        return analysis_result_from_table(
            single_df,
            analysis="fowa",
            summary=single_df.attrs.get("shared_summary", {}),
            diagnostics={
                "plot data": plot_data,
                "full results": single_df.attrs.get("full_results_df"),
            },
            warnings=single_df.attrs.get("warnings", {}),
            display_table_formatter=_fowa_results_display_table,
        )

    return analysis_result_from_table(
        display_df,
        analysis="fowa",
        summary=display_df.attrs.get("shared_summary", {}),
        diagnostics={
            "plot data": plot_data,
            "full results": display_df.attrs.get("full_results_df"),
        },
        warnings=display_df.attrs.get("warnings", {}),
        display_table_formatter=_fowa_results_display_table,
    )

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
            start, stop = fit_indices_array
            start = _coerce_fit_index(start)
            stop = _coerce_fit_index(stop)
            return x[start:stop], y[start:stop]

        positions = np.asarray([_coerce_fit_index(value) for value in fit_indices_array], dtype=int)
        return x[positions], y[positions]

    if fit_indices_array.ndim == 2 and fit_indices_array.shape[1] == 2:
        positions = []
        base_indices = np.arange(len(x))
        for start, stop in fit_indices_array:
            start = _coerce_fit_index(start)
            stop = _coerce_fit_index(stop)
            positions.extend(base_indices[start:stop])
        positions = np.asarray(positions, dtype=int)
        return x[positions], y[positions]

    raise ValueError(
        "'fit indices' should be [start, stop], [[start, stop], ...], "
        "a boolean mask, or explicit integer indices."
    )


def _fit_selection_specs(options, fallback_fit_indices=None, default_label="Fit 1"):
    """Return normalized fit selection specs for shared scatter-fit helpers."""
    fit_indices = fallback_fit_indices
    if fit_indices is None:
        fit_indices = options.get("fit indices")

    if fit_indices is None:
        return [(default_label, None, False)]

    normalized = _normalize_fit_index_ranges(fit_indices)
    if len(normalized) == 1 and not isinstance(fit_indices, dict):
        _label, spec = normalized[0]
        return [(default_label, spec, False)]
    return [(label, spec, True) for label, spec in normalized]


def _fit_rate_selection_specs(options, fallback_fit_indices, default_label):
    return _fit_selection_specs(
        options,
        fallback_fit_indices=fallback_fit_indices,
        default_label=default_label,
    )


def _fit_rate_selected_points(plot_x, plot_y, spec):
    return _select_fit_indices(plot_x, plot_y, spec)


def _fit_rate_fit_options(options, label):
    fit_options = options.copy()
    fit_options["_fit selection label"] = label
    return fit_options


def _without_matplotlib_axis_scales(options):
    plot_options = dict(options or {})
    plot_options["plot scale"] = None
    plot_options["xscale"] = None
    plot_options["yscale"] = None
    return plot_options


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


def _fit_rate_payload(df, options=None):
    """Fit rate-style tabular data and return internal table/fit payloads.
    
    Parameters
    ----------
    df : pandas.DataFrame
        Table containing rate, transformed x, or metric columns.
    options : dict or FitRateOptions, optional
        Transform, fit-window, print, and plot options. See ``e.describe_options("fit_rate")``.
    
    Returns
    -------
    tuple
        Internal fit data and fit coefficient payload used to build ScatterFitResult.

    Examples
    --------
    >>> result = e.fit_rate(fowa_df, {"fit indices": [1, 5]})
    >>> result.table
    """
    if isinstance(df, AnalysisResult):
        df = df.table
    if not isinstance(df, pd.DataFrame):
        raise TypeError("fit_rate accepts a pandas DataFrame or an AnalysisResult with a table.")

    raw_options = options
    typed_options = FitRateOptions.from_options(options)
    options = typed_options.to_options_dict()
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
        fit_specs = _fit_rate_selection_specs(
            options,
            fallback_fit_indices=fit_indices,
            default_label=metric_col,
        )
        if do_fit:
            for fit_label, fit_spec, is_named_selection in fit_specs:
                fit_x_sel, fit_y_sel = _fit_rate_selected_points(
                    plot_x,
                    plot_y,
                    fit_spec,
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

            if len(fitline) == 1 and not isinstance(options.get("fit indices"), dict):
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
                for fit_label, fit_spec, is_named_selection in fit_specs:
                    fit_x_sel, fit_y_sel = _fit_rate_selected_points(
                        plot_x,
                        plot_y,
                        fit_spec,
                    )
                    series_fit = _fit_series_xy(
                        fit_x_sel,
                        fit_y_sel,
                        options=options,
                        label=fit_label,
                    )
                    plot_options = _fit_rate_fit_options(options, fit_label)
                    fit_line_index = fit_color_index
                    plot_options = _options_with_default_fit_color(
                        plot_options,
                        raw_options,
                        point_color,
                        index=fit_color_index,
                    )
                    plot_options = _without_matplotlib_axis_scales(plot_options)
                    fit_color_index += 1
                    plot_options.update({"new plot": False, "plot data": False, "model label": f"{fit_label} Fit", "_fit line index": fit_line_index})
                    if len(fit_specs) > 1 and not plot_options.get("fit label", False):
                        plot_options["fit label"] = str(fit_label)
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
            _apply_matplotlib_axis_scales(plt.gca(), _without_matplotlib_axis_scales(options))

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
    fit_specs = _fit_rate_selection_specs(
        options,
        fallback_fit_indices=options.get("fit indices"),
        default_label=metric_col,
    )
    if do_fit:
        for fit_label, fit_spec, is_named_selection in fit_specs:
            fit_x_sel, fit_y_sel = _fit_rate_selected_points(
                plot_x,
                plot_y,
                fit_spec,
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

        if len(fitline) == 1 and not isinstance(options.get("fit indices"), dict):
            fitline = next(iter(fitline.values()))

    if do_plot:
        plt.figure()
        point_color = _artist_color(plt.scatter(
            plot_x,
            plot_y,
            label=_format_fit_rate_metric_label(metric_col, unit=metric_unit),
        ))

        if do_fit:
            for fit_label, fit_spec, is_named_selection in fit_specs:
                fit_x_sel, fit_y_sel = _fit_rate_selected_points(
                    plot_x,
                    plot_y,
                    fit_spec,
                )
                series_fit = _fit_series_xy(
                    fit_x_sel,
                    fit_y_sel,
                    options=options,
                    label=fit_label,
                )
                plot_options = _fit_rate_fit_options(options, fit_label)
                fit_line_index = fit_color_index
                plot_options = _options_with_default_fit_color(
                    plot_options,
                    raw_options,
                    point_color,
                    index=fit_color_index,
                )
                fit_color_index += 1
                plot_options.update({"new plot": False, "plot data": False, "model label": f"{fit_label} Fit", "_fit line index": fit_line_index})
                if len(fit_specs) > 1 and not plot_options.get("fit label", False):
                    plot_options["fit label"] = str(fit_label)
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
            if transformed["y label"] == "log10":
                y_axis_label = _format_fit_rate_metric_label(
                    metric_col,
                    log=True,
                    unit=metric_unit,
                )
            else:
                y_axis_label = _format_y_transform_axis_label(
                    _format_fit_rate_metric_label(metric_col, unit=metric_unit),
                    y_transform,
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


def fit_rate(df, options=None):
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
    return _scatter_result_from_payload(
        _fit_rate_payload(df, options),
        summary={"analysis": "rate fit"},
    )


def _tafel_options_from_mapping(options):
    if isinstance(options, TafelAnalysisOptions):
        return options.to_options_dict()
    if options is None:
        options = {}
    if not isinstance(options, dict):
        raise TypeError(
            "tafel_analysis options must be a dict or TafelAnalysisOptions, "
            f"not {type(options).__name__}."
        )
    return _project_options(TafelAnalysisOptions, options).to_options_dict()


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


def _tafel_equation_bundle():
    return {
        "symbolic latex": (
            r"\mathrm{TOF}=\frac{2\mathrm{TOF}_{\max}}"
            r"{1+\exp\left[\frac{F}{RT}(E_{\mathrm{thermo}}-E_{\mathrm{redox}}-\eta)\right]},\quad "
            r"y=\log_{10}(\mathrm{TOF})"
        ),
        "resolved latex": "",
        "compact latex": "",
        "definitions latex": "",
        "symbolic": "TOF = 2 TOFmax / (1 + exp((F / (R T)) * (Ethermo - Eredox - eta))); y = log10(TOF)",
        "resolved": "",
        "compact": "",
        "definitions": "",
    }


def _tafel_parameter_table(
    tof_values,
    thermodynamic_potential,
    redox_potential,
    overpotential_range,
    temperatures,
    options=None,
):
    options = {} if options is None else options
    sig_figs = options.get("sig figs", 4)
    unique_tof = list(dict.fromkeys(float(value) for value in tof_values))
    unique_temperatures = list(dict.fromkeys(float(value) for value in temperatures))
    tof_value = (
        _format_sevcik_value(unique_tof[0], sig_figs=sig_figs, unit="s^-1")
        if len(unique_tof) == 1
        else "per CV"
    )
    temperature_value = (
        _format_sevcik_value(unique_temperatures[0], sig_figs=sig_figs, unit="K")
        if len(unique_temperatures) == 1
        else "per CV"
    )
    eta_start, eta_end = overpotential_range
    return _analysis_parameter_table(
        [
            ("Maximum Turnover Frequency", "TOFmax", tof_value),
            ("Thermodynamic Potential", "Ethermo", _format_sevcik_value(thermodynamic_potential, sig_figs=sig_figs, unit="V")),
            ("Redox Potential", "Eredox", _format_sevcik_value(redox_potential, sig_figs=sig_figs, unit="V")),
            ("Temperature", "T", temperature_value),
            (
                "Overpotential Range",
                "η",
                (
                    f"{_format_fit_model_display_value(eta_start, sig_figs=sig_figs)} to "
                    f"{_format_fit_model_display_value(eta_end, sig_figs=sig_figs)} V"
                ),
            ),
        ]
    )


def _tafel_summary_display_table(summary, options=None):
    options = {} if options is None else options
    sig_figs = options.get("sig figs", 4)
    display = summary.copy()
    for column, unit in [
        ("TOFmax", "s^-1"),
        ("Temperature", "K"),
        ("Thermodynamic Potential", "V"),
        ("Redox Potential", "V"),
    ]:
        if column in display:
            display[column] = [
                _format_sevcik_value(value, sig_figs=sig_figs, unit=unit)
                for value in display[column]
            ]
    if len(display) == 1:
        row = display.iloc[0].to_dict()
        return pd.DataFrame(
            [
                {"Metric": "Label", "Value": row.get("Label", "")},
                {"Metric": "TOFmax", "Value": row.get("TOFmax", "")},
                {"Metric": "Temperature", "Value": row.get("Temperature", "")},
                {"Metric": "Thermodynamic Potential", "Value": row.get("Thermodynamic Potential", "")},
                {"Metric": "Redox Potential", "Value": row.get("Redox Potential", "")},
            ]
        )
    return display


def _display_tafel_report(summary, options, *, tof_values, thermodynamic_potential, redox_potential, overpotential_range):
    if not options.get("print", True):
        return
    temperatures = summary["Temperature"].tolist() if "Temperature" in summary else [298]
    parameter_table = _tafel_parameter_table(
        tof_values,
        thermodynamic_potential,
        redox_potential,
        overpotential_range,
        temperatures,
        options,
    )
    _display_table(
        parameter_table,
        options,
        title="Tafel Analysis Parameters",
        rich_table=_analysis_parameter_rich_table(parameter_table),
        escape=None,
        index=False,
    )
    equation = _tafel_equation_bundle()
    if _rich_table_output_enabled(options):
        _display_analysis_equation(
            r"\text{Tafel analysis equations:}",
            "Tafel Analysis Equations",
            equation,
            resolved=False,
            compact=False,
            include_definitions=False,
        )
    else:
        print("Tafel Analysis Equations:")
        print("  " + equation["symbolic"])
    _display_table(
        _tafel_summary_display_table(summary, options),
        options,
        title="Tafel Analysis Summary",
        index=False,
    )


def tafel_analysis(cv, TOF_max, thermodynamic_potential, redox_potential, options=None):
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
    raw_mapping = raw_options.to_options_dict() if isinstance(raw_options, TafelAnalysisOptions) else raw_options
    tafel_options = _tafel_options_from_mapping(raw_options)
    display_options = {} if raw_mapping is None else dict(raw_mapping)
    display_options.setdefault("print", True)
    display_options.setdefault("pretty print", True)
    display_options.setdefault("sig figs", 4)
    do_plot = bool(display_options.get("plot", True))
    cvs = _coerce_tafel_cv_list(cv)
    tof_values = _coerce_tafel_tof_values(TOF_max, len(cvs))

    start, end = tafel_options["overpotential range"]
    overpotential = np.linspace(float(start), float(end), 1000)

    if do_plot and len(cvs) > 1:
        plot_options = _multiplot_options_from_mapping(raw_mapping)
        style = _prepare_multiplot_style(cvs, plot_options)
        ax = style["ax"]
        color_spec = style["color spec"]
        plot_labels = color_spec["labels"]
        line_colors = color_spec["line colors"]
        display_labels = style["display labels"]
    elif do_plot:
        fig, ax = plt.subplots()
        plot_labels = [None]
        line_colors = [tafel_options["color"]]
        display_labels = [getattr(cvs[0], "name", "CV")]
        plot_options = _multiplot_options_from_mapping(raw_mapping)
        style = None
    else:
        ax = None
        plot_labels = [None] * len(cvs)
        line_colors = [tafel_options["color"]] * len(cvs)
        display_labels = [getattr(cv_obj, "name", f"CV {i + 1}") for i, cv_obj in enumerate(cvs)]
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
        if ax is not None:
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

    if ax is not None:
        ax.set_xlabel(r"$\eta$ (V)")
        ax.set_ylabel(r"$\log_{10}(\mathrm{TOF}\ (s^{-1}))$")

    if style is not None:
        _finish_multiplot_style(cvs, plot_options, style)

    data = pd.DataFrame(data_rows)
    summary = pd.DataFrame(summary_rows)
    data.attrs["summary"] = summary
    _display_tafel_report(
        summary,
        display_options,
        tof_values=tof_values,
        thermodynamic_potential=thermodynamic_potential,
        redox_potential=redox_potential,
        overpotential_range=(start, end),
    )
    return analysis_result_from_table(
        data,
        analysis="tafel",
        summary={"table": summary},
        values={"summary": summary, "axes": ax},
        axes=ax,
        figure=ax.figure if ax is not None else None,
    )




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
