"""CV-specific analysis helpers."""

from .utils import *  # noqa: F401,F403
from .options import *  # noqa: F401,F403
from .objects import cv

def _coerce_cv_list(cvs):
    if isinstance(cvs, cv):
        return [cvs]
    if isinstance(cvs, (list, tuple)):
        if len(cvs) == 0:
            raise ValueError("FOWA requires at least one CV.")
        if not all(isinstance(item, cv) for item in cvs):
            raise TypeError("All entries passed to FOWA must be cv objects.")
        return list(cvs)
    raise TypeError("FOWA expects a cv object or a list/tuple of cv objects.")


def _resolve_non_catalytic_cvs(cvs, options):
    shared_ref = options.get("non-catalytic cv")
    ref_list = options.get("non-catalytic cvs")

    if shared_ref is not None and ref_list is not None:
        raise ValueError(
            "Use either 'non-catalytic cv' or 'non-catalytic cvs', not both."
        )

    if shared_ref is not None:
        if not isinstance(shared_ref, cv):
            raise TypeError("'non-catalytic cv' must be a cv object.")
        return [shared_ref] * len(cvs)

    if ref_list is not None:
        if not isinstance(ref_list, (list, tuple)):
            raise TypeError("'non-catalytic cvs' must be a list or tuple of cv objects.")
        if len(ref_list) != len(cvs):
            raise ValueError(
                "'non-catalytic cvs' must have the same length as the catalytic CV list."
            )
        if not all(isinstance(item, cv) for item in ref_list):
            raise TypeError("All entries in 'non-catalytic cvs' must be cv objects.")
        return list(ref_list)

    return [None] * len(cvs)


def _is_ip0_y_axis(axis_name):
    return str(axis_name).strip().lower().replace(" ", "") == "i/ip0"


DIMENSIONLESS_POTENTIAL_AXIS = "Dimensionless Potential"
DIMENSIONLESS_CURRENT_AXIS = "Dimensionless Current"


def _default_normalized_axis(cv_obj, axis):
    if getattr(cv_obj, "plot_mode", None) != "normalized":
        return None
    axes = getattr(cv_obj, "normalization_axes", {}) or {}
    column_name = axes.get(axis)
    if column_name in getattr(cv_obj, "data", pd.DataFrame()).columns:
        return column_name
    return None


def _resolve_normalization_temperature(cv_obj, options):
    value = options.get("temperature", options.get("t"))
    if value is None:
        value = getattr(cv_obj, "temperature", None)
    if value is None or not np.isfinite(float(value)) or float(value) == 0:
        return None
    return float(value)


def _resolve_normalization_area(cv_obj, options):
    value = options.get("area", options.get("s", options.get("electrode area")))
    if value is None:
        value = getattr(cv_obj, "electrode_area", None)
    if value is None or not np.isfinite(float(value)) or float(value) == 0:
        return None
    return float(value)


def _resolve_normalization_scan_rate(cv_obj, options):
    value = options.get("scan rate", options.get("v"))
    if value is None:
        value = getattr(cv_obj, "scan_rate", None)
    if value is None or not np.isfinite(float(value)) or float(value) == 0:
        return None
    return float(value)


def _resolve_normalization_species_concentration(cv_obj, species):
    requested = str(species)
    compounds = list(getattr(cv_obj, "compounds", []) or [])
    concentrations = list(getattr(cv_obj, "concentrations", []) or [])
    matches = [
        idx
        for idx, compound in enumerate(compounds)
        if str(compound) == requested
    ]
    available = ", ".join(str(compound) for compound in compounds) or "none"

    if not matches:
        raise ValueError(
            f"Species '{requested}' was not found in CV compounds. "
            f"Available species: {available}."
        )
    if len(matches) > 1:
        raise ValueError(
            f"Species '{requested}' matched multiple CV compounds. "
            "Provide explicit C and C unit instead."
        )

    idx = matches[0]
    if idx >= len(concentrations) or concentrations[idx] in (None, ""):
        raise ValueError(
            f"Species '{requested}' does not have an available concentration. "
            "Provide explicit C and C unit instead."
        )

    concentration_text = str(concentrations[idx])
    try:
        return concentration_to_float(concentration_text) / 1000, concentration_text, requested
    except Exception as exc:
        raise ValueError(
            f"Concentration for species '{requested}' could not be parsed: "
            f"{concentration_text!r}. Provide explicit C and C unit instead."
        ) from exc


def _resolve_normalization_concentration(cv_obj, options):
    if "c" not in options or options.get("c") is None:
        species = options.get("species")
        if species in (None, ""):
            return None, None, None
        return _resolve_normalization_species_concentration(cv_obj, species)

    value = options.get("c")
    unit = options.get("c unit")

    if isinstance(value, str):
        return concentration_to_float(value) / 1000, value, None

    if unit is None:
        raise ValueError("Numeric concentration requires 'C unit'.")

    value = float(value)
    unit_key = str(unit).strip().replace("μ", "u").lower()

    conversions = {
        "mol/cm^3": 1.0,
        "mol/cm3": 1.0,
        "m": 1 / 1000,
        "mol/l": 1 / 1000,
        "mol/liter": 1 / 1000,
        "mm": 1e-3 / 1000,
        "um": 1e-6 / 1000,
        "nm": 1e-9 / 1000,
    }
    if unit_key not in conversions:
        raise ValueError(
            "'C unit' must be one of 'mol/cm^3', 'M', 'mM', 'uM', 'μM', or 'nM'."
        )
    return value * conversions[unit_key], f"{value:g} {unit}", None


def _normalization_axis_label(mode, axis):
    mode = str(mode).strip().lower()
    if axis == "x":
        return r"$\theta$"
    if mode.startswith("hetero"):
        return r"$\chi$"
    return r"$\Phi$"


def _format_cv_normalization_equation(mode, axes, values):
    mode_text = str(mode).strip().lower()
    current_symbol = r"\chi" if mode_text.startswith("hetero") else r"\Phi"
    current_name = "chi" if mode_text.startswith("hetero") else "Phi"

    symbolic_parts_latex = []
    resolved_parts_latex = []
    compact_parts_latex = []
    symbolic_parts_text = []
    resolved_parts_text = []
    compact_parts_text = []

    if axes.get("x"):
        symbolic_parts_latex.append(r"\theta=\frac{nF(E-E^0)}{RT}")
        resolved_parts_latex.append(
            rf"\theta=\frac{{({values['n']:g})F(E-({values['E0']:g}))}}"
            rf"{{R({values['T']:g})}}"
        )
        compact_parts_latex.append(
            rf"\theta={values['n'] * F / (R * values['T']):.6g}(E-({values['E0']:g}))"
        )
        symbolic_parts_text.append("theta = n * F * (E - E0) / (R * T)")
        resolved_parts_text.append(
            f"theta = {values['n']:g} * F * (E - {values['E0']:g}) / "
            f"(R * {values['T']:g})"
        )
        compact_parts_text.append(
            f"theta = {values['n'] * F / (R * values['T']):.6g} * "
            f"(E - {values['E0']:g})"
        )

    if axes.get("y"):
        symbolic_parts_latex.append(
            rf"{current_symbol}=\frac{{I}}{{nFSC^*\sqrt{{D nF\nu/(RT)}}}}"
        )
        resolved_parts_latex.append(
            rf"{current_symbol}=\frac{{I}}{{({values['n']:g})F({values['S']:g})"
            rf"({values['C']:g})\sqrt{{({values['D']:g})({values['n']:g})F"
            rf"({values['v']:g})/(R({values['T']:g}))}}}}"
        )
        denominator = values["denominator"]
        compact_parts_latex.append(rf"{current_symbol}=I/({denominator:.6g})")
        symbolic_parts_text.append(
            f"{current_name} = I / (n * F * S * C* * sqrt(D * n * F * v / (R * T)))"
        )
        resolved_parts_text.append(
            f"{current_name} = I / ({values['n']:g} * F * {values['S']:g} "
            f"* {values['C']:g} * sqrt({values['D']:g} * {values['n']:g} "
            f"* F * {values['v']:g} / (R * {values['T']:g})))"
        )
        compact_parts_text.append(f"{current_name} = I / {denominator:.6g}")

    definition_items = [
        f"mode = {mode}",
        f"n = {values['n']:g}",
        f"R = {R:.6g}",
        f"F = {F:.6g}",
        f"T = {values['T']:g} K",
    ]
    definition_latex_items = [
        rf"\mathrm{{mode}}={mode}",
        rf"n={values['n']:g}",
        rf"R={R:.6g}",
        rf"F={F:.6g}",
        rf"T={values['T']:g}\ \mathrm{{K}}",
    ]

    if axes.get("x"):
        definition_items.append(f"E0 = {values['E0']:g} V")
        definition_latex_items.append(rf"E^0={values['E0']:g}\ \mathrm{{V}}")
    if axes.get("y"):
        definition_items.extend(
            [
                f"D = {values['D']:g} cm^2/s",
                f"C = {values['C text']} ({values['C']:g} mol/cm^3)",
                f"S = {values['S']:g} cm^2",
                f"v = {values['v']:g} V/s",
            ]
        )
        definition_latex_items.extend(
            [
                rf"D={values['D']:g}\ \mathrm{{cm^2/s}}",
                rf"C^*={values['C']:g}\ \mathrm{{mol/cm^3}}",
                rf"S={values['S']:g}\ \mathrm{{cm^2}}",
                rf"\nu={values['v']:g}\ \mathrm{{V/s}}",
            ]
        )
        if "species" in values:
            definition_items.append(f"species = {values['species']}")
            definition_latex_items.append(rf"\mathrm{{species}}={values['species']}")
        if "lambda" in values:
            definition_items.append(f"lambda = {values['lambda']:.6g}")
            definition_latex_items.append(rf"\lambda={values['lambda']:.6g}")
        if "psi" in values:
            definition_items.append(f"psi = {values['psi']:.6g}")
            definition_latex_items.append(rf"\psi={values['psi']:.6g}")

    return {
        "symbolic latex": r",\quad ".join(symbolic_parts_latex),
        "resolved latex": r",\quad ".join(resolved_parts_latex),
        "compact latex": r",\quad ".join(compact_parts_latex),
        "definitions latex": r",\quad ".join(definition_latex_items),
        "symbolic": "; ".join(symbolic_parts_text),
        "resolved": "; ".join(resolved_parts_text),
        "compact": "; ".join(compact_parts_text),
        "definitions": ", ".join(definition_items),
    }


def _display_cv_normalization_equation(mode, axes, values):
    equation = _format_cv_normalization_equation(mode, axes, values)
    return _display_analysis_equation(
        r"\text{CV normalization equation:}",
        "CV normalization equation",
        equation,
        resolved=True,
        compact=False,
    )


def _format_cv_normalization_symbolic_equation(mode, axes):
    mode_text = str(mode).strip().lower()
    current_symbol = r"\chi" if mode_text.startswith("hetero") else r"\Phi"
    current_name = "chi" if mode_text.startswith("hetero") else "Phi"

    symbolic_latex = []
    symbolic_text = []
    if axes.get("x"):
        symbolic_latex.append(r"\theta=\frac{nF(E-E^0)}{RT}")
        symbolic_text.append("theta = n * F * (E - E0) / (R * T)")
    if axes.get("y"):
        symbolic_latex.append(
            rf"{current_symbol}=\frac{{I}}{{nFSC^*\sqrt{{D nF\nu/(RT)}}}}"
        )
        symbolic_text.append(
            f"{current_name} = I / (n * F * S * C* * sqrt(D * n * F * v / (R * T)))"
        )
    return {
        "latex": r",\quad ".join(symbolic_latex),
        "text": "; ".join(symbolic_text),
    }


def _display_cv_normalization_symbolic_equation(mode, axes):
    equation = _format_cv_normalization_symbolic_equation(mode, axes)
    if display is not None and Math is not None:
        display(Math(r"\text{CV normalization equation:}"))
        display(Math(equation["latex"]))
    else:
        print("[CV normalization equation]")
        print("  " + equation["text"])
    return equation


def _normalization_parameter_label(key, html=False):
    labels = {
        "n": "n",
        "T": "T (K)",
        "E0": "E0 (V)",
        "D": "D (cm^2/s)",
        "C": "C (mol/cm^3)",
        "C text": "C input",
        "species": "species",
        "S": "S (cm^2)",
        "v": "v (V/s)",
        "denominator": "denominator",
        "lambda": "lambda",
        "psi": "psi",
    }
    html_labels = {
        "n": "<i>n</i>",
        "T": "<i>T</i> / K",
        "E0": "<i>E</i><sup>0</sup> / V",
        "D": "<i>D</i> / cm<sup>2</sup> s<sup>-1</sup>",
        "C": "<i>C</i><sup>*</sup> / mol cm<sup>-3</sup>",
        "C text": "<i>C</i><sup>*</sup> input",
        "species": "Species",
        "S": "<i>S</i> / cm<sup>2</sup>",
        "v": "<i>&nu;</i> / V s<sup>-1</sup>",
        "denominator": "Denominator",
        "lambda": "<i>&lambda;</i>",
        "psi": "<i>&psi;</i>",
    }
    if html:
        return html_labels.get(key, labels.get(key, key))
    return labels.get(key, key)


def _normalization_parameter_display_value(value, sig_figs):
    if value is None:
        return ""
    if isinstance(value, (int, float, np.integer, np.floating)):
        if not np.isfinite(float(value)):
            return str(value)
        return f"{round_sigfigs(float(value), sig_figs):g}"
    return str(value)


def _cv_normalization_parameter_table(normalized_cvs, options):
    preferred_order = [
        "n",
        "T",
        "E0",
        "D",
        "C",
        "C text",
        "species",
        "S",
        "v",
        "lambda",
        "psi",
    ]
    available = []
    for cv_obj in normalized_cvs:
        available.extend(
            key for key in (getattr(cv_obj, "normalization_parameters", {}) or {}).keys()
            if key not in {"mode", "denominator"}
        )
    ordered_keys = [key for key in preferred_order if key in available]
    ordered_keys.extend(key for key in dict.fromkeys(available) if key not in ordered_keys)

    sig_figs = options.get("sig figs", 4)
    columns = []
    data = {}
    single = len(normalized_cvs) == 1
    for index, cv_obj in enumerate(normalized_cvs):
        column = "Value" if single else index
        columns.append(column)
        params = getattr(cv_obj, "normalization_parameters", {}) or {}
        data[column] = [
            _normalization_parameter_display_value(params.get(key), sig_figs)
            for key in ordered_keys
        ]

    table = pd.DataFrame(
        data,
        index=[_normalization_parameter_label(key) for key in ordered_keys],
        columns=columns,
    )
    table.index.name = "Parameter"
    table.attrs["parameter_keys"] = ordered_keys
    return table


def _display_cv_normalization_parameter_table(table, options):
    if not options.get("pretty print", True) or display is None:
        print(table.to_string())
        return table

    display_table = table.copy()
    keys = table.attrs.get("parameter_keys", list(display_table.index))
    display_table.index = [
        _normalization_parameter_label(key, html=True)
        for key in keys
    ]
    display_table.index.name = "Parameter"
    styled = (
        display_table.style
        .format(escape=None)
        .format_index(escape=None, axis=0)
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


def _print_cv_normalization_summary(normalized_cvs, options):
    print("CV normalization summary:")
    axes = {
        "x": any((getattr(cv_obj, "normalization_axes", {}) or {}).get("x") for cv_obj in normalized_cvs),
        "y": any((getattr(cv_obj, "normalization_axes", {}) or {}).get("y") for cv_obj in normalized_cvs),
    }
    mode = getattr(normalized_cvs[0], "normalization_mode", options.get("mode", "homogeneous"))
    print(f"Mode: {mode}")
    _display_cv_normalization_symbolic_equation(mode, axes)

    table = _cv_normalization_parameter_table(normalized_cvs, options)
    return _display_cv_normalization_parameter_table(table, options)


def _normalize_single_cv(cv_obj, options):
    mode = str(options.get("normalization mode", options.get("mode", "homogeneous"))).strip().lower()
    if mode not in {"homogeneous", "heterogeneous"}:
        raise ValueError("'mode' must be 'homogeneous' or 'heterogeneous'.")

    n = float(options.get("n", options.get("num electrons", 1)))
    if n <= 0:
        raise ValueError("'n' must be positive.")

    T = _resolve_normalization_temperature(cv_obj, options)
    if T is None:
        raise ValueError("normalize requires a nonzero temperature for dimensionless normalization.")

    normalized = deepcopy(cv_obj)
    normalized.data = cv_obj.data.copy(deep=True)
    normalized.units = getattr(cv_obj, "units", {}).copy()

    axes = {"x": None, "y": None}
    values = {"mode": mode, "n": n, "T": T}

    if options.get("e0") is not None:
        E0 = float(options["e0"])
        potential = np.asarray(cv_obj.x({"x axis": "Potential"}), dtype=float)
        normalized.data[DIMENSIONLESS_POTENTIAL_AXIS] = n * F * (potential - E0) / (R * T)
        normalized.units[DIMENSIONLESS_POTENTIAL_AXIS] = ""
        axes["x"] = DIMENSIONLESS_POTENTIAL_AXIS
        values["E0"] = E0

    D = options.get("d")
    C, C_text, C_species = _resolve_normalization_concentration(cv_obj, options)
    S = _resolve_normalization_area(cv_obj, options)
    v = _resolve_normalization_scan_rate(cv_obj, options)
    current_requested = any(
        key in options and options.get(key) is not None
        for key in ("d", "c", "c unit", "species", "area", "s", "electrode area", "scan rate", "v")
    )

    if current_requested:
        missing = []
        if D is None:
            missing.append("D")
        if C is None:
            missing.append("C and C unit")
        if S is None:
            missing.append("area")
        if v is None:
            missing.append("scan rate")
        if missing:
            raise ValueError(
                "Current normalization requires explicit/available " + ", ".join(missing) + "."
            )

        D = float(D)
        if D <= 0 or C <= 0 or S <= 0 or v <= 0:
            raise ValueError("D, C, area, and scan rate must be positive for current normalization.")

        current = np.asarray(cv_obj.y({"y axis": "Current"}), dtype=float)
        current_unit = getattr(cv_obj, "units", {}).get("Current", "A")
        current = current * get_conversion_factor(current_unit, "A")
        denominator = n * F * S * C * np.sqrt(D * n * F * v / (R * T))
        normalized.data[DIMENSIONLESS_CURRENT_AXIS] = current / denominator
        normalized.units[DIMENSIONLESS_CURRENT_AXIS] = ""
        axes["y"] = DIMENSIONLESS_CURRENT_AXIS
        values.update(
            {
                "D": D,
                "C": C,
                "C text": C_text,
                "S": S,
                "v": v,
                "denominator": denominator,
            }
        )
        if C_species is not None:
            values["species"] = C_species

        if mode == "homogeneous" and options.get("k homo") is not None:
            values["lambda"] = float(options["k homo"]) / (n * F * v / (R * T))
        if mode == "heterogeneous" and options.get("k0") is not None:
            values["psi"] = float(options["k0"]) / np.sqrt(np.pi * D * (n * F * v / (R * T)))

    if axes["x"] is None and axes["y"] is None:
        raise ValueError(
            "normalize could not create any dimensionless axes. Provide E0 for potential "
            "normalization or D, C, C unit, area, and scan rate for current normalization."
        )

    normalized.plot_mode = "normalized"
    normalized.normalization_mode = mode
    normalized.normalization_axes = axes
    normalized.normalization_axis_labels = {
        axis: _normalization_axis_label(mode, axis)
        for axis, column in axes.items()
        if column is not None
    }
    normalized.normalization_parameters = values.copy()
    normalized.dimensionless_parameters = {
        key: values[key] for key in ("psi", "lambda") if key in values
    }

    if options.get("print", False):
        _print_cv_normalization_summary([normalized], options)

    return normalized


def normalize(cvs, options=None):
    """Return CV copy/copies with physical dimensionless normalization axes.
    
    Parameters
    ----------
    cvs : cv or sequence of cv
        CV object or objects to copy and normalize.
    options : dict or NormalizeOptions, optional
        Dimensionless CV normalization options. See ``e.describe_options("normalize")``.
    
    Returns
    -------
    cv or list of cv
        Normalized copy or copies; the input objects are not mutated.
    
    Examples
    --------
    >>> normalized = e.normalize(cvs, {"E0": 0.0, "D": 1e-5, "C": 10, "C unit": "mM"})
    """
    typed_options = NormalizeOptions.from_options(options)
    options = _legacy_normalize_option_keys(typed_options.to_legacy_dict())
    if isinstance(cvs, cv):
        pass
    elif isinstance(cvs, (list, tuple)):
        if not all(isinstance(item, cv) for item in cvs):
            raise TypeError("normalize currently supports cv objects only.")
    else:
        raise TypeError("normalize currently supports cv objects only.")
    single_input = isinstance(cvs, cv)
    cv_list = _coerce_cv_list(cvs)
    print_series = bool(options.get("print", False)) and not single_input
    worker_options = options.copy()
    if print_series:
        worker_options["print"] = False
    normalized = [_normalize_single_cv(cv_obj, worker_options) for cv_obj in cv_list]
    if print_series:
        _print_cv_normalization_summary(normalized, options)
    return normalized[0] if single_input else normalized


def _find_column_by_text(columns, column_name):
    lookup = str(column_name).strip().lower()
    for col in columns:
        if str(col).strip().lower() == lookup:
            return col
    return None


def _has_ip0_column(echem_object):
    return _find_column_by_text(getattr(echem_object, "data", pd.DataFrame()).columns, "i/ip0") is not None


def _resolve_manual_ip0_values(options, n_items):
    ip0_value = options.get("ip0")
    non_catalytic_current = options.get("non-catalytic current")

    if ip0_value is not None and non_catalytic_current is not None:
        raise ValueError("Use either 'ip0' or 'non-catalytic current', not both.")

    manual_value = ip0_value if ip0_value is not None else non_catalytic_current
    option_name = "ip0" if ip0_value is not None else "non-catalytic current"
    return _resolve_fowa_scalar_or_sequence(
        manual_value,
        n_items,
        option_name,
        allow_none=True,
    )


def _reference_ip0_options(options, segment=None):
    if isinstance(options, NormalizationOptions):
        return options.for_peak_current()

    ref_options = {}
    for field in fields(PeakCurrentOptions):
        option_key = field.name.replace("_", " ")
        if option_key in options:
            ref_options[option_key] = options[option_key]
        elif field.name in options:
            ref_options[field.name] = options[field.name]

    nc_guess = options.get("reference guess potential")
    if nc_guess is None:
        nc_guess = options.get("non-catalytic guess potential")
    if nc_guess is None:
        nc_guess = options.get("guess potential")
    if nc_guess is None:
        nc_guess = options.get("redox potential")
    if nc_guess is not None:
        ref_options["guess potential"] = nc_guess

    if segment is not None:
        ref_options.pop("segments", None)
        ref_options["segment"] = segment

    ref_options["y axis"] = "Current"
    ref_options["plot"] = False
    ref_options["plot all"] = False
    ref_options["print"] = False
    ref_options["print all"] = False
    ref_options["internal call"] = True
    ref_options["new plot"] = False
    return PeakCurrentOptions.from_options(ref_options)


def _resolve_reference_cvs(cvs, options, *, option_name="reference"):
    shared_ref = options.get("reference cv")
    ref_list = options.get("reference cvs")
    ref_index = options.get("reference index", 0)

    if shared_ref is not None and ref_list is not None:
        raise ValueError("Use either 'reference cv' or 'reference cvs', not both.")

    if shared_ref is not None:
        if not isinstance(shared_ref, cv):
            raise TypeError("'reference cv' must be a cv object.")
        return [shared_ref] * len(cvs)

    if ref_list is not None:
        if not isinstance(ref_list, (list, tuple)):
            raise TypeError("'reference cvs' must be a list or tuple of cv objects.")
        if len(ref_list) != len(cvs):
            raise ValueError("'reference cvs' must have the same length as the CV list.")
        if not all(isinstance(item, cv) for item in ref_list):
            raise TypeError("All entries in 'reference cvs' must be cv objects.")
        return list(ref_list)

    if not isinstance(ref_index, int):
        raise ValueError("'reference index' must be an integer.")
    if ref_index < 0:
        ref_index += len(cvs)
    if ref_index < 0 or ref_index >= len(cvs):
        raise IndexError(
            f"'reference index' = {options.get('reference index', 0)} is out of range "
            f"for a list of length {len(cvs)}."
        )
    return [cvs[ref_index]] * len(cvs)


def _count_current_reference_sources(options, scalar_key):
    count = 0
    if options.get(scalar_key) is not None:
        count += 1
    if options.get("reference cv") is not None:
        count += 1
    if options.get("reference cvs") is not None:
        count += 1
    if options.get("reference index", 0) != 0:
        count += 1
    return count


def _resolve_reference_ip0(ref_cv, options, segment=None):
    ref_options = _reference_ip0_options(options, segment=segment)
    try:
        ref_Ep = ref_cv.peak_potential(ref_options)["Ep"]
    except Exception:
        ref_Ep = None
    ref_current = ref_cv.peak_current(ref_options)
    ip0 = ref_current["ip"]
    ref_tanline = ref_current["tangent line"]
    return float(ip0), ref_cv.name, ref_Ep, ref_tanline


def _scale_reference_segments(options):
    segments = options.get("segments")
    if segments is None:
        segment = options.get("segment", 1)
        if segment is None:
            segment = 1
        segments = [segment, segment + 1]
    if isinstance(segments, int):
        segments = [segments]
    if not isinstance(segments, (list, tuple)) or len(segments) == 0:
        raise ValueError("'segments' must provide at least one segment for reference mode 'both'.")
    return list(segments)


def _resolve_reference_current_vector(cv_obj, options):
    mode = str(options.get("reference mode", "single")).strip().lower()
    if mode == "single":
        ip, _source, _ref_Ep, _ref_tanline = _resolve_reference_ip0(cv_obj, options)
        return np.array([_validate_ip0_value(ip, f"CV '{cv_obj.name}'")], dtype=float)
    if mode == "both":
        values = [
            _resolve_reference_ip0(cv_obj, options, segment=segment)[0]
            for segment in _scale_reference_segments(options)
        ]
        return np.array([
            _validate_ip0_value(value, f"CV '{cv_obj.name}' segment {segment}")
            for value, segment in zip(values, _scale_reference_segments(options))
        ], dtype=float)
    raise ValueError("'reference mode' must be 'single' or 'both' for scale_current.")


def _scale_from_reference_currents(target_values, measured_values, source):
    target_values = np.asarray(target_values, dtype=float)
    measured_values = np.asarray(measured_values, dtype=float)
    if target_values.shape != measured_values.shape:
        raise ValueError("Reference and measured current vectors must have the same length.")
    if len(target_values) == 1:
        return _validate_ip0_value(target_values[0], source) / _validate_ip0_value(measured_values[0], source)
    denominator = float(np.dot(measured_values, measured_values))
    if not np.isfinite(denominator) or denominator == 0:
        raise ValueError(f"The reference-wave current vector for {source} is zero, so scaling cannot be computed.")
    scale = float(np.dot(measured_values, target_values) / denominator)
    if not np.isfinite(scale) or scale == 0:
        raise ValueError(f"Resolved scale factor for {source} must be nonzero and finite.")
    return scale


def _validate_ip0_value(ip0, source):
    if ip0 is None or not np.isfinite(float(ip0)) or float(ip0) == 0:
        raise ValueError(f"Resolved ip0 from {source} must be nonzero and finite.")
    return float(ip0)


def _resolve_ip0_values(cvs, options):
    cvs = _coerce_cv_list(cvs)
    if _count_current_reference_sources(options, "ip0") > 1:
        raise OptionError("Use only one ip0/reference source for normalize_current.")
    manual_ip0_values = _resolve_manual_ip0_values(options, len(cvs))
    ref_cvs = _resolve_reference_cvs(cvs, options)

    ip0_values = []
    for i, (cv_obj, ref_cv) in enumerate(zip(cvs, ref_cvs)):
        manual_ip0 = manual_ip0_values[i]
        if manual_ip0 is not None:
            ip0_values.append(_validate_ip0_value(manual_ip0, "'ip0'"))
            continue

        try:
            ip0, _source, _ref_Ep, _ref_tanline = _resolve_reference_ip0(ref_cv, options)
        except Exception as exc:
            raise ValueError(
                f"Could not determine ip0 from reference CV '{ref_cv.name}' "
                f"for '{cv_obj.name}'."
            ) from exc
        ip0_values.append(_validate_ip0_value(ip0, f"reference CV '{ref_cv.name}'"))

    return ip0_values


def _apply_normalized_current_axis(cv_obj, ip0, options, column_name="i/ip0"):
    ip0 = _validate_ip0_value(ip0, "normalization current")

    full_options = options.copy()
    for key in ("segment", "segments", "plot segment", "plot segments"):
        full_options.pop(key, None)
    full_options["y axis"] = "Current"

    raw_current_col = _find_column_by_text(cv_obj.data.columns, "Current")
    if raw_current_col is not None:
        raw_current = cv_obj.data[raw_current_col]
    else:
        raw_current = cv_obj.y(full_options)
    cv_obj.data[column_name] = np.asarray(raw_current, dtype=float) / ip0
    cv_obj.units[column_name] = ""
    cv_obj.plot_mode = "normalized"
    cv_obj.normalization_mode = "i/ip0"
    cv_obj.normalization_axes = {"x": None, "y": column_name}
    cv_obj.normalization_axis_labels = {"y": "$i / i_p^0$"}
    cv_obj.normalization_parameters = {"ip0": ip0}
    return cv_obj


def _copy_cv_with_normalized_current_axis(cv_obj, ip0, options, column_name="i/ip0"):
    cv_copy = deepcopy(cv_obj)
    cv_copy.data = cv_obj.data.copy(deep=True)
    cv_copy.units = getattr(cv_obj, "units", {}).copy()
    return _apply_normalized_current_axis(cv_copy, ip0, options, column_name=column_name)


def _normalize_current_source_label(cvs, options, index):
    if options.get("ip0") is not None:
        return "manual ip0"
    if options.get("non-catalytic current") is not None:
        return "manual non-catalytic current"
    if options.get("reference cv") is not None:
        ref = options.get("reference cv")
        return f"reference CV: {getattr(ref, 'name', 'CV')}"
    if options.get("reference cvs") is not None:
        ref = options.get("reference cvs")[index]
        return f"reference CV: {getattr(ref, 'name', 'CV')}"

    ref_index = options.get("reference index", 0)
    if not isinstance(ref_index, int):
        return "reference CV"
    if ref_index < 0:
        ref_index += len(cvs)
    if 0 <= ref_index < len(cvs):
        return f"reference CV: {getattr(cvs[ref_index], 'name', 'CV')}"
    return "reference CV"


def _format_ip0_summary_value(ip0, sig_figs):
    return f"{round_sigfigs(float(ip0), sig_figs):g} A"


def _print_normalize_current_summary(cv_list, ip0_values, options):
    sig_figs = options.get("sig figs", 4)
    source_labels = [
        _normalize_current_source_label(cv_list, options, index)
        for index in range(len(cv_list))
    ]
    shared_ip0 = len(ip0_values) > 0 and np.allclose(ip0_values, ip0_values[0])
    shared_source = len(source_labels) > 0 and len(set(source_labels)) == 1

    print("Current Normalization:")

    if len(cv_list) == 1:
        print(f"CV: {getattr(cv_list[0], 'name', 'CV')}")
        print(f"ip0: {_format_ip0_summary_value(ip0_values[0], sig_figs)}")
        print(f"Source: {source_labels[0]}")
        return

    if shared_ip0 and shared_source:
        print(f"CVs: {len(cv_list)}")
        print(f"ip0: {_format_ip0_summary_value(ip0_values[0], sig_figs)}")
        print(f"Source: {source_labels[0]}")
        return

    rows = pd.DataFrame(
        {
            "CV": [getattr(cv_obj, "name", "CV") for cv_obj in cv_list],
            "Source": source_labels,
            "ip0": [
                _format_ip0_summary_value(ip0, sig_figs)
                for ip0 in ip0_values
            ],
        }
    )

    if options.get("pretty print", True):
        display_object_table(rows, options)
    else:
        print(rows.to_string(index=False))


def _raw_current_columns(cv_obj):
    columns = []
    for col in cv_obj.data.columns[cv_obj.num_x_cols:]:
        name = col[-1] if isinstance(col, tuple) else col
        unit = getattr(cv_obj, "units", {}).get(name, getattr(cv_obj, "units", {}).get(col, ""))
        if str(name).strip().lower() == "i/ip0":
            continue
        if str(name).strip().lower().startswith("dimensionless"):
            continue
        if str(name).strip().lower().startswith("current"):
            columns.append(col)
            continue
        try:
            _prefix, base = extract_prefix_and_base(unit)
        except Exception:
            base = ""
        if base == "A":
            columns.append(col)
    return columns


def _apply_current_scale(cv_obj, scale, options):
    scale = float(scale)
    if not np.isfinite(scale) or scale == 0:
        raise ValueError("'scale' must be nonzero and finite.")
    columns = _raw_current_columns(cv_obj)
    if not columns:
        raise ValueError("scale_current could not find a raw current column to scale.")
    for column in columns:
        cv_obj.data[column] = np.asarray(cv_obj.data[column], dtype=float) * scale
    cv_obj.current_scale_factor = scale
    return cv_obj


def normalize_current(cvs, options=None):
    """Return CV copies with an ``i/ip0`` current-normalization column.
    
    Parameters
    ----------
    cvs : cv or sequence of cv
        CV object or objects to copy and normalize.
    options : dict or NormalizationOptions, optional
        ip0 resolution, print, and diagnostic-plot options. See ``e.describe_options("normalize_current")``.
    
    Returns
    -------
    cv or list of cv
        Normalized copy or copies; the input objects are not mutated.
    
    Examples
    --------
    >>> normalized = e.normalize_current(cvs, {"reference cv": blank_cv})
    """
    typed_options = NormalizationOptions.from_options(options)
    options = typed_options.to_legacy_dict()
    single_input = isinstance(cvs, cv)
    cv_list = _coerce_cv_list(cvs)
    ip0_values = _resolve_ip0_values(cv_list, options)
    normalized = [
        deepcopy(cv_obj).normalize_current(ip0, options)
        for cv_obj, ip0 in zip(cv_list, ip0_values)
    ]

    if options.get("print", True):
        _print_normalize_current_summary(cv_list, ip0_values, options)

    if options.get("plot all", False):
        plot_options = _multiplot_options_from_mapping(options)
        plot_options["y axis"] = "i/ip0"
        plot_options["ylabel"] = "$i / i_p^0$"
        plot_options["print"] = False
        multiplot(normalized, plot_options)

    return normalized[0] if single_input else normalized


def _resolve_scale_values(cvs, options):
    cvs = _coerce_cv_list(cvs)
    if _count_current_reference_sources(options, "scale") > 1:
        raise OptionError("Use only one scale/reference source for scale_current.")

    manual_scale = options.get("scale")
    if manual_scale is not None:
        return _resolve_fowa_scalar_or_sequence(
            manual_scale,
            len(cvs),
            "scale",
            allow_none=False,
        ), [None] * len(cvs), [None] * len(cvs)

    ref_cvs = _resolve_reference_cvs(cvs, options)
    target_ips = []
    measured_ips = []
    scales = []
    target_cache = {}
    for cv_obj, ref_cv in zip(cvs, ref_cvs):
        target_key = id(ref_cv)
        if target_key not in target_cache:
            target_cache[target_key] = _resolve_reference_current_vector(ref_cv, options)
        target_ip = target_cache[target_key]
        measured_ip = _resolve_reference_current_vector(cv_obj, options)
        target_ips.append(target_ip)
        measured_ips.append(measured_ip)
        scales.append(_scale_from_reference_currents(target_ip, measured_ip, f"CV '{cv_obj.name}'"))
    return scales, target_ips, measured_ips


def scale_current(cvs, options=None):
    """Return CV copies with raw current columns multiplied by scale factors.
    
    Parameters
    ----------
    cvs : cv or sequence of cv
        CV object or objects to copy and scale.
    options : dict or ScaleCurrentOptions, optional
        Scale or reference-wave options. See ``e.describe_options("scale_current")``.
    
    Returns
    -------
    cv or list of cv
        Scaled copy or copies; the input objects are not mutated.
    
    Examples
    --------
    >>> scaled = e.scale_current(cvs, {"reference index": 0})
    """
    typed_options = ScaleCurrentOptions.from_options(options)
    options = typed_options.to_legacy_dict()
    single_input = isinstance(cvs, cv)
    cv_list = _coerce_cv_list(cvs)
    scales, target_ips, measured_ips = _resolve_scale_values(cv_list, options)
    scaled = [
        deepcopy(cv_obj).scale_current(scale, options)
        for cv_obj, scale in zip(cv_list, scales)
    ]

    for scaled_cv, scale, target_ip, measured_ip in zip(scaled, scales, target_ips, measured_ips):
        scaled_cv.current_scale_factor = scale
        scaled_cv.current_scale_target_ip = target_ip
        scaled_cv.current_scale_source_ip = measured_ip

    if options.get("print", False):
        print("Current scaling summary:")
        if options.get("pretty print", True):
            print_options = options.copy()
            print_options["print conditions"] = options.get("print conditions", True)
            display_df, _meta = build_object_table(cv_list, print_options)
            display_df["Scale Factor"] = [
                round_sigfigs(scale, options.get("sig figs", 4))
                for scale in scales
            ]
            display_object_table(display_df, print_options)
        else:
            for cv_obj, scale in zip(cv_list, scales):
                print(f"  {cv_obj.name}: scale = {scale:.6g}")

    if options.get("plot all", False):
        plot_options = _multiplot_options_from_mapping(options)
        plot_options["print"] = False
        multiplot(scaled, plot_options)

    return scaled[0] if single_input else scaled




def _display_analysis_equation(*args, **kwargs):
    from .analysis_batch import _display_analysis_equation as impl
    return impl(*args, **kwargs)


def _resolve_fowa_scalar_or_sequence(*args, **kwargs):
    from .analysis_batch import _resolve_fowa_scalar_or_sequence as impl
    return impl(*args, **kwargs)


def _multiplot_options_from_mapping(*args, **kwargs):
    from .plotting import _multiplot_options_from_mapping as impl
    return impl(*args, **kwargs)


def build_object_table(*args, **kwargs):
    from .plotting import build_object_table as impl
    return impl(*args, **kwargs)


def display_object_table(*args, **kwargs):
    from .plotting import display_object_table as impl
    return impl(*args, **kwargs)


def multiplot(*args, **kwargs):
    from .plotting import multiplot as impl
    return impl(*args, **kwargs)

__all__ = ["normalize", "normalize_current", "scale_current"]
