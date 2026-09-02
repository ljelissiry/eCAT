"""Scan-rate-series reversibility and surface-coverage analyses."""

from __future__ import annotations

from collections.abc import Sequence
import warnings as python_warnings

import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter, ScalarFormatter
import numpy as np
import pandas as pd

from .options import ReversibilityAnalysisOptions, SurfaceCoverageAnalysisOptions
from .options import PeakCurrentOptions, _project_options
from .plotting import (
    ScatterFitResult,
    _conditional_analysis_name_column,
    _display_table,
    _pretty_table_header_html_label,
)
from .results import AnalysisResult
from ._cv_direction import cv_segment_scan_direction
from .metadata import concentration_to_float
from .utils import scale_value


R = 8.31446261815324
F = 96485.33212


def _finite_float(value, default=np.nan):
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return float(default)
    return numeric if np.isfinite(numeric) else float(default)


def _format_result_value(value, *, unit="", sig_figs=4, prefix=""):
    numeric = _finite_float(value)
    if not np.isfinite(numeric):
        return "Unavailable"
    text = f"{numeric:.{int(sig_figs)}g}"
    if prefix:
        text = f"{prefix}{text}"
    return f"{text} {unit}".strip()


def _current_display_scale(values, selected_unit="auto"):
    array = np.asarray(values, dtype=float)
    finite = array[np.isfinite(array)]
    if finite.size == 0 or selected_unit is None:
        return 1.0, "A"
    reference = float(np.nanmax(np.abs(finite)))
    if reference == 0:
        return 1.0, "A"
    scaled_reference, display_unit = scale_value(
        reference,
        "A",
        selected_unit=selected_unit,
    )
    return float(scaled_reference) / reference, display_unit


def _surface_reversibility_equation_bundle():
    return {
        "symbolic latex": (
            r"i_p=\frac{n^2F^2S\Gamma\nu}{4RT},\quad "
            r"\Delta E_p=E_{p,\mathrm{a}}-E_{p,\mathrm{c}}"
        ),
        "resolved latex": (
            r"|\Delta E_p|\leq\Delta E_{p,\mathrm{tol}}\ \mathrm{(reversible)},\quad "
            r"n\Delta E_p>200\ \mathrm{mV}\ \mathrm{(Laviron\ eligibility)}"
        ),
        "compact latex": "",
        "definitions latex": (
            r"r_i=\left|\frac{i_{p,\mathrm{a}}}{i_{p,\mathrm{c}}}\right|"
        ),
        "symbolic": "ip = n^2 F^2 S Gamma v / (4 RT); Delta Ep = Epa - Epc",
        "resolved": "reversible when |Delta Ep| <= configured tolerance; Laviron eligibility requires n Delta Ep > 200 mV",
        "compact": "",
        "definitions": "ri = |ipa / ipc|",
    }


def _surface_coverage_equation_bundle():
    return {
        "symbolic latex": r"i_p=\frac{n^2F^2S\Gamma\nu}{4RT}",
        "resolved latex": (
            r"Q=nFS\Gamma,\quad "
            r"n_{\mathrm{loading}}=S\Gamma=\frac{Q}{nF}"
        ),
        "compact latex": "",
        "definitions latex": "",
        "symbolic": "ip = n^2 F^2 S Gamma v / (4 RT)",
        "resolved": "Q = n F S Gamma; loading = S Gamma = Q / (n F)",
        "compact": "",
        "definitions": "",
    }


def _bulk_reversibility_equation_sections(
    *,
    include_sevcik=False,
    include_rate=False,
    include_irreversible=False,
):
    """Return labeled bulk equations for the evidence paths actually used."""
    sections = [
        (
            "Nicholson Peak-Separation Conversion",
            {
                "symbolic latex": (
                    r"\psi=\frac{-0.6288+0.0021(n\Delta E_p/\mathrm{mV})}"
                    r"{1-0.017(n\Delta E_p/\mathrm{mV})}"
                ),
                "resolved latex": r"\Lambda=\sqrt{\pi}\,\psi",
                "compact latex": "",
                "definitions latex": "",
                "symbolic": (
                    "psi = [-0.6288 + 0.0021(n Delta Ep / mV)] / "
                    "[1 - 0.017(n Delta Ep / mV)]"
                ),
                "resolved": "Lambda = sqrt(pi) * psi",
                "compact": "",
                "definitions": "",
            },
        ),
        (
            "Matsuda-Ayabe Classification",
            {
                "symbolic latex": (
                    r"\mathrm{region}(\Lambda)=\begin{cases}"
                    r"\mathrm{reversible},&\Lambda\geq15\\"
                    r"\mathrm{quasi\!\text{-}\!reversible},&"
                    r"10^{-2(1+\alpha)}<\Lambda<15\\"
                    r"\mathrm{irreversible},&\Lambda\leq10^{-2(1+\alpha)}"
                    r"\end{cases}"
                ),
                "resolved latex": (
                    r"0.1\leq\psi\leq7\quad"
                    r"\mathrm{for\ Nicholson\ }k^0\mathrm{\ estimation}"
                ),
                "compact latex": "",
                "definitions latex": "",
                "symbolic": (
                    "reversible: Lambda >= 15; quasi-reversible: "
                    "10^[-2(1+alpha)] < Lambda < 15; irreversible: "
                    "Lambda <= 10^[-2(1+alpha)]"
                ),
                "resolved": "Nicholson k0 estimation range: 0.1 <= psi <= 7",
                "compact": "",
                "definitions": "",
            },
        ),
    ]
    if include_sevcik:
        sections.append(
            (
                "Sevcik Diffusion Estimate",
                {
                    "symbolic latex": (
                        r"|i_p|=0.4463\,nFSC"
                        r"\left(\frac{nFD\nu}{RT}\right)^{1/2},\quad"
                        r"D=\frac{RT}{(nF)^3}"
                        r"\left(\frac{m}{0.4463\,SC}\right)^2"
                    ),
                    "resolved latex": r"m=\frac{d|i_p|}{d\sqrt{\nu}}",
                    "compact latex": "",
                    "definitions latex": "",
                    "symbolic": (
                        "|ip| = 0.4463 n F S C sqrt(n F D v / RT); "
                        "D = RT/(nF)^3 * [m/(0.4463 S C)]^2"
                    ),
                    "resolved": "m = d|ip|/d sqrt(v)",
                    "compact": "",
                    "definitions": "",
                },
            )
        )
    if include_rate:
        sections.append(
            (
                "Electron-Transfer Rate Conversion",
                {
                    "symbolic latex": (
                        r"k^0=\Lambda"
                        r"\left(\frac{DnF\nu}{RT}\right)^{1/2}"
                    ),
                    "resolved latex": r"\text{This conversion requires }D.",
                    "compact latex": "",
                    "definitions latex": "",
                    "symbolic": "k0 = Lambda * sqrt(D n F v / RT)",
                    "resolved": "This conversion requires D.",
                    "compact": "",
                    "definitions": "",
                },
            )
        )
    if include_irreversible:
        sections.append(
            (
                "Irreversible-Asymptote Verification",
                {
                    "symbolic latex": (
                        r"\left|E_p-E_{p/2}\right|="
                        r"\frac{1.857RT}{\alpha nF}"
                    ),
                    "resolved latex": (
                        r"\left|\frac{dE_p}{d\log_{10}\nu}\right|="
                        r"\frac{2.303RT}{2\alpha nF}"
                    ),
                    "compact latex": "",
                    "definitions latex": "",
                    "symbolic": "|Ep - Ep/2| = 1.857 RT / (alpha n F)",
                    "resolved": "|dEp/d log10(v)| = 2.303 RT / (2 alpha n F)",
                    "compact": "",
                    "definitions": "",
                },
            )
        )
    return sections


def _reversibility_equation_sections(result):
    phase = result.summary.get("phase", "bulk")
    if phase == "surface":
        return [
            (
                "Surface-Confined Reversible Response",
                _surface_reversibility_equation_bundle(),
            )
        ]

    decision_tree = result.diagnostics.get("decision tree", {})
    sevcik = decision_tree.get("Sevcik Dapp", {})
    electrochemical = decision_tree.get("electrochemical", {})
    regions = electrochemical.get("regions", [])
    conclusion = str(result.summary.get("electrochemical conclusion", "")).lower()
    include_irreversible = any(
        "irreversible" in str(region).lower() for region in regions
    ) or "irreversible" in conclusion
    diffusion = _finite_float(result.summary.get("D / cm^2 s^-1"))
    return _bulk_reversibility_equation_sections(
        include_sevcik=bool(sevcik),
        include_rate=bool(np.isfinite(diffusion) and diffusion > 0),
        include_irreversible=include_irreversible,
    )


def _display_equation_bundle(title, bundle, options):
    include_definitions = bool(
        bundle.get("definitions latex") or bundle.get("definitions")
    )
    if options.get("pretty print", True):
        from .analysis_batch import _display_analysis_equation

        return _display_analysis_equation(
            rf"\text{{{title}:}}",
            title,
            bundle,
            resolved=True,
            compact=False,
            include_definitions=include_definitions,
        )
    print(f"{title}:")
    print(f"  {bundle['symbolic']}")
    print(f"  {bundle['resolved']}")
    if include_definitions:
        print(f"  {bundle['definitions']}")
    return bundle


def _display_reversibility_equations(result, options):
    phase = result.summary.get("phase", "bulk")
    title = (
        "Reversibility Analysis Equations"
        if phase == "bulk"
        else "Surface Reversibility Equations"
    )
    sections = _reversibility_equation_sections(result)
    if options.get("pretty print", True):
        from .analysis_batch import Math, display, _display_analysis_equation

        if display is not None and Math is not None:
            display(Math(rf"\text{{{title}:}}"))
        else:
            print(f"{title}:")
        for label, bundle in sections:
            _display_analysis_equation(
                rf"\text{{{label}:}}",
                label,
                bundle,
                resolved=True,
                compact=False,
                include_definitions=bool(
                    bundle.get("definitions latex") or bundle.get("definitions")
                ),
            )
        return sections

    print(f"{title}:")
    for label, bundle in sections:
        print(f"[{label}]")
        print(f"  {bundle['symbolic']}")
        if bundle.get("resolved"):
            print(f"  {bundle['resolved']}")
        if bundle.get("definitions"):
            print(f"  {bundle['definitions']}")
    return sections


def _rich_scientific_text(value):
    if not isinstance(value, str):
        return value
    replacements = (
        ("|ipa/ipc|", "|i<sub>p,a</sub>/i<sub>p,c</sub>|") ,
        ("nDeltaEp", "nΔE<sub>p</sub>"),
        ("Ep-Ep/2", "|E<sub>p</sub>-E<sub>p/2</sub>|") ,
        ("Ep-v", "E<sub>p</sub>-ν"),
        ("k0", "k<sup>0</sup>"),
        ("Lambda", "Λ"),
        ("psi", "ψ"),
        ("Gamma", "Γ"),
        ("Dapp", "D<sub>app</sub>"),
        ("cm^2", "cm<sup>2</sup>"),
        ("cm^-2", "cm<sup>-2</sup>"),
        ("s^-1", "s<sup>-1</sup>"),
        ("V^-1", "V<sup>-1</sup>"),
    )
    text = value
    for old, new in replacements:
        text = text.replace(old, new)
    return text


def _display_symbol_table(
    table,
    options,
    *,
    title,
    index=False,
    symbol_columns=(),
    rich_text_columns=(),
):
    rich_table = table.copy().rename(columns=_pretty_table_header_html_label)
    for column in symbol_columns:
        rich_column = _pretty_table_header_html_label(column)
        if rich_column in rich_table:
            rich_table[rich_column] = rich_table[rich_column].map(
                lambda value: _pretty_table_header_html_label(value)
                if isinstance(value, str)
                else value
            )
    for column in rich_text_columns:
        rich_column = _pretty_table_header_html_label(column)
        if rich_column in rich_table:
            rich_table[rich_column] = rich_table[rich_column].map(
                _rich_scientific_text
            )
    return _display_table(
        table,
        options,
        title=title,
        rich_table=rich_table,
        escape=None,
        index=index,
    )


def _format_scan_rate_axis(axis, scan_rates, *, sig_figs=4):
    rates = np.unique(np.asarray(scan_rates, dtype=float))
    rates = rates[np.isfinite(rates) & (rates > 0)]
    axis.set_xscale("log")
    axis.xaxis.set_major_locator(FixedLocator(rates))
    axis.xaxis.set_major_formatter(
        FuncFormatter(lambda value, _position: f"{value:.{int(sig_figs)}g}")
    )
    axis.xaxis.set_minor_formatter(NullFormatter())
    return axis


def _agreement_status(first, second, *, tolerance):
    """Return agreement state and symmetric fractional disagreement."""
    a = _finite_float(first)
    b = _finite_float(second)
    if not (np.isfinite(a) and np.isfinite(b)):
        return "unavailable", np.nan
    if a == 0 and b == 0:
        return "agree", 0.0
    if (a == 0) != (b == 0):
        return "disagree", np.inf
    disagreement = abs(a - b) / ((abs(a) + abs(b)) / 2)
    return ("agree" if disagreement <= float(tolerance) else "disagree"), disagreement


def _rate_mean_table(table, value_columns):
    """Aggregate trend inputs by scan rate while retaining replicate spread."""
    rate_column = "scan rate / V s^-1"
    if rate_column not in table:
        raise ValueError(f"Rate table requires '{rate_column}'.")
    rows = []
    for scan_rate, group in table.groupby(rate_column, sort=True, dropna=False):
        row = {rate_column: scan_rate, "replicate count": int(len(group))}
        for column in value_columns:
            values = pd.to_numeric(group[column], errors="coerce")
            row[column] = float(values.mean()) if values.notna().any() else np.nan
            row[f"{column} std"] = (
                float(values.std(ddof=1)) if values.notna().sum() > 1 else np.nan
            )
        rows.append(row)
    return pd.DataFrame(rows)


def _matsuda_ayabe_region(lambda_value, *, alpha=0.5):
    value = _finite_float(lambda_value)
    if not np.isfinite(value) or value < 0:
        return "unavailable"
    if value >= 15:
        return "reversible"
    lower = 10 ** (-2 * (1 + float(alpha)))
    if value <= lower:
        return "irreversible"
    return "quasi-reversible"


def _linear_regression(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    finite = np.isfinite(x) & np.isfinite(y)
    if finite.sum() < 2:
        return {"slope": np.nan, "intercept": np.nan, "r2": np.nan, "points": int(finite.sum())}
    x_fit = x[finite]
    y_fit = y[finite]
    slope, intercept = np.polyfit(x_fit, y_fit, 1)
    predicted = slope * x_fit + intercept
    ss_res = float(np.sum((y_fit - predicted) ** 2))
    ss_tot = float(np.sum((y_fit - np.mean(y_fit)) ** 2))
    r2 = 1.0 if ss_tot == 0 and ss_res == 0 else 1 - ss_res / ss_tot if ss_tot else np.nan
    return {
        "slope": float(slope),
        "intercept": float(intercept),
        "r2": float(r2),
        "points": int(len(x_fit)),
        "x": x_fit,
        "y": y_fit,
        "predicted": predicted,
    }


def _nicholson_evidence(n_delta_ep_mv):
    """Resolve Nicholson psi and Matsuda-Ayabe Lambda from n*Delta Ep."""
    value = _finite_float(n_delta_ep_mv)
    if not np.isfinite(value) or value <= 0:
        return {"psi": np.nan, "Lambda": np.nan, "eligible": False, "reason": "invalid nDeltaEp"}
    if value < 61:
        return {
            "psi": np.inf,
            "Lambda": np.inf,
            "eligible": False,
            "reason": f"too reversible for Nicholson k0 (nDeltaEp={value:.4g} mV < 61 mV)",
        }
    psi = (-0.6288 + 0.0021 * value) / (1 - 0.017 * value)
    if not np.isfinite(psi) or psi <= 0:
        return {
            "psi": np.nan,
            "Lambda": np.nan,
            "eligible": False,
            "reason": "Nicholson conversion did not produce a positive finite psi",
        }
    lambda_value = float(np.sqrt(np.pi) * psi)
    eligible = bool(0.1 <= psi <= 7 and value <= 212)
    if eligible:
        reason = "Nicholson eligible"
    elif psi > 7:
        reason = f"too reversible for Nicholson k0 (psi={psi:.4g} > 7)"
    else:
        reason = (
            "too irreversible for Nicholson k0 "
            f"(psi={psi:.4g} < 0.1 or nDeltaEp={value:.4g} mV > 212 mV)"
        )
    return {
        "psi": float(psi),
        "Lambda": lambda_value,
        "eligible": eligible,
        "reason": reason,
    }


def _k0_from_lambda(lambda_value, *, diffusion, scan_rate, num_electrons, temperature):
    values = [lambda_value, diffusion, scan_rate, num_electrons, temperature]
    if not all(np.isfinite(_finite_float(value)) and float(value) > 0 for value in values):
        return np.nan
    return float(lambda_value) * np.sqrt(
        float(diffusion) * float(num_electrons) * F * float(scan_rate)
        / (R * float(temperature))
    )


def _branch_sevcik_diffusion(
    scan_rates,
    peak_currents,
    *,
    num_electrons,
    temperature,
    electrode_area,
    concentration,
):
    x = np.sqrt(np.asarray(scan_rates, dtype=float))
    y = np.abs(np.asarray(peak_currents, dtype=float))
    fit = _linear_regression(x, y)
    n = float(num_electrons)
    temperature = float(temperature)
    area = float(electrode_area)
    concentration = float(concentration)
    if n <= 0 or temperature <= 0 or area <= 0 or concentration <= 0:
        raise ValueError("Sevcik Dapp estimation requires positive n, T, area, and concentration.")
    slope = fit["slope"]
    diffusion = (R * temperature / (F * n) ** 3) * (
        slope / (0.4463 * area * concentration)
    ) ** 2
    return {**fit, "D / cm^2 s^-1": float(diffusion)}


def _chemical_reversibility_decision(scan_rates, current_ratios, *, tolerance, min_r2):
    rates = np.asarray(scan_rates, dtype=float)
    ratios = np.asarray(current_ratios, dtype=float)
    finite = np.isfinite(rates) & (rates > 0) & np.isfinite(ratios) & (ratios >= 0)
    if finite.sum() < 3:
        return {
            "conclusion": "indeterminate",
            "reason": "fewer than three finite |ipa/ipc| ratios",
            "trend toward unity": False,
            "ratio mean": np.nan,
            "ratio fit": {},
        }
    rates = rates[finite]
    ratios = ratios[finite]
    deviations = np.abs(ratios - 1)
    fit = _linear_regression(np.log10(rates), deviations)
    ratio_min = float(np.min(ratios))
    ratio_max = float(np.max(ratios))
    maximum_deviation = float(np.max(deviations))
    within_tolerance = int(np.count_nonzero(deviations <= float(tolerance)))
    evidence = (
        f"range {ratio_min:.4g} to {ratio_max:.4g}; maximum deviation from unity "
        f"is {maximum_deviation:.4g}; current-ratio tolerance is {float(tolerance):.4g}"
    )
    if np.all(deviations <= float(tolerance)):
        conclusion = "chemically reversible over observed timescale"
        reason = (
            f"all {len(ratios)} tangent-corrected |ipa/ipc| ratios are within "
            f"tolerance of unity ({evidence})"
        )
    elif float(np.nanmean(deviations)) > float(tolerance):
        conclusion = "coupled chemistry indicated"
        reason = (
            f"{len(ratios) - within_tolerance} of {len(ratios)} tangent-corrected "
            f"|ipa/ipc| ratios exceed tolerance ({evidence})"
        )
    else:
        conclusion = "indeterminate"
        reason = (
            f"current-ratio evidence straddles the configured tolerance: "
            f"{within_tolerance} of {len(ratios)} ratios are within tolerance ({evidence})"
        )
    trend_toward_unity = bool(
        np.isfinite(fit.get("slope", np.nan))
        and fit["slope"] < 0
        and np.isfinite(fit.get("r2", np.nan))
        and fit["r2"] >= float(min_r2)
    )
    if trend_toward_unity and conclusion != "chemically reversible over observed timescale":
        conclusion = "coupled chemistry indicated"
        reason += "; the ratio approaches unity at faster scan rates"
    return {
        "conclusion": conclusion,
        "reason": reason,
        "trend toward unity": trend_toward_unity,
        "ratio mean": float(np.mean(ratios)),
        "ratio minimum": ratio_min,
        "ratio maximum": ratio_max,
        "maximum deviation": maximum_deviation,
        "current ratio tolerance": float(tolerance),
        "points within tolerance": within_tolerance,
        "points": int(len(ratios)),
        "ratio fit": fit,
    }


def _surface_coverage_from_slope(
    scan_rates,
    peak_currents,
    *,
    num_electrons,
    temperature,
    electrode_area,
):
    scan_rates = np.asarray(scan_rates, dtype=float)
    peak_currents = np.asarray(peak_currents, dtype=float)
    finite = np.isfinite(scan_rates) & np.isfinite(peak_currents)
    if finite.sum() < 2:
        raise ValueError("Surface coverage slope analysis requires at least two finite points.")
    x = scan_rates[finite]
    y = peak_currents[finite]
    slope, intercept = np.polyfit(x, y, 1)
    fitted = slope * x + intercept
    ss_res = float(np.sum((y - fitted) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 if ss_tot == 0 and ss_res == 0 else 1 - ss_res / ss_tot if ss_tot else np.nan
    n = float(num_electrons)
    temperature = float(temperature)
    loading = 4 * R * temperature * abs(float(slope)) / (n**2 * F**2)
    area = _finite_float(electrode_area)
    coverage = loading / area if np.isfinite(area) and area > 0 else np.nan
    return {
        "slope / A s V^-1": float(slope),
        "intercept / A": float(intercept),
        "r2": float(r2),
        "coverage / mol cm^-2": float(coverage),
        "loading / mol": float(loading),
        "x": x,
        "y": y,
        "y fit": fitted,
    }


def _surface_charge_from_potential(
    potential,
    corrected_current,
    *,
    scan_rate,
    num_electrons,
    electrode_area,
):
    potential = np.asarray(potential, dtype=float)
    corrected_current = np.asarray(corrected_current, dtype=float)
    finite = np.isfinite(potential) & np.isfinite(corrected_current)
    if finite.sum() < 2:
        raise ValueError("Surface charge integration requires at least two finite points.")
    scan_rate = float(scan_rate)
    if not np.isfinite(scan_rate) or scan_rate <= 0:
        raise ValueError("Surface charge integration requires a positive scan rate in V/s.")
    trapezoid = np.trapezoid
    charge = abs(float(trapezoid(corrected_current[finite], potential[finite]))) / scan_rate
    loading = charge / (float(num_electrons) * F)
    area = _finite_float(electrode_area)
    coverage = loading / area if np.isfinite(area) and area > 0 else np.nan
    return {
        "charge / C": charge,
        "coverage / mol cm^-2": float(coverage),
        "loading / mol": float(loading),
    }


def _coerce_cv_series(cvs, *, analysis_name):
    if isinstance(cvs, (str, bytes)) or not isinstance(cvs, Sequence):
        raise TypeError(f"{analysis_name} requires a sequence of CV objects.")
    values = list(cvs)
    if not values:
        raise ValueError(f"{analysis_name} requires at least one CV.")
    for value in values:
        if not hasattr(value, "analysis_segment_data") or not hasattr(value, "scan_rate"):
            raise TypeError(f"{analysis_name} accepts CV objects only.")
    return values


def _condition_signature(cv_obj):
    compounds = tuple(str(value) for value in (getattr(cv_obj, "compounds", None) or ()))
    concentrations = tuple(
        str(value) for value in (getattr(cv_obj, "concentrations", None) or ())
    )
    return (
        str(getattr(cv_obj, "solvent", None) or ""),
        str(getattr(cv_obj, "gas", None) or ""),
        compounds,
        concentrations,
    )


def _validate_series_input(cvs, *, analysis_name):
    cv_list = _coerce_cv_series(cvs, analysis_name=analysis_name)
    signatures = {}
    for cv_obj in cv_list:
        signatures.setdefault(_condition_signature(cv_obj), []).append(
            str(getattr(cv_obj, "name", "unnamed"))
        )
    if len(signatures) > 1:
        details = "; ".join(
            f"condition {index + 1}: {len(names)} CV(s)"
            for index, names in enumerate(signatures.values())
        )
        raise ValueError(
            f"{analysis_name} requires one chemical condition; detected {len(signatures)} groups "
            f"({details}). Use e.group(...) and analyze each group separately."
        )
    rates = np.asarray([_finite_float(cv_obj.scan_rate) for cv_obj in cv_list], dtype=float)
    distinct = np.unique(rates[np.isfinite(rates) & (rates > 0)])
    if len(distinct) < 3:
        raise ValueError(
            f"{analysis_name} requires at least three distinct scan rates; found {len(distinct)}."
        )
    return cv_list, distinct


def _shared_positive_metadata(cv_list, attribute, explicit=None, *, fallback=np.nan):
    value = _finite_float(explicit)
    if np.isfinite(value) and value > 0:
        return value, "option"
    values = [_finite_float(getattr(cv_obj, attribute, None)) for cv_obj in cv_list]
    values = [value for value in values if np.isfinite(value) and value > 0]
    if values and np.allclose(values, values[0], rtol=1e-6, atol=0):
        return float(values[0]), f"CV {attribute} metadata"
    return float(fallback), "fallback" if np.isfinite(_finite_float(fallback)) else "unavailable"


def _resolve_bulk_concentration(cv_list, options):
    explicit = _finite_float(options.get("C"))
    if np.isfinite(explicit) and explicit > 0:
        return explicit, "option"
    species = options.get("species")
    if species in (None, ""):
        return np.nan, "unavailable"
    concentrations = []
    for cv_obj in cv_list:
        compounds = [str(value) for value in (getattr(cv_obj, "compounds", None) or [])]
        matches = [index for index, value in enumerate(compounds) if value == str(species)]
        if len(matches) != 1:
            raise ValueError(
                f"Species {species!r} must exact-match one compound in every CV when C is omitted. "
                f"Available compounds for '{getattr(cv_obj, 'name', 'unnamed')}': {compounds or 'none'}."
            )
        stored = list(getattr(cv_obj, "concentrations", None) or [])
        if matches[0] >= len(stored):
            raise ValueError(f"Species {species!r} has no paired concentration metadata.")
        concentrations.append(concentration_to_float(stored[matches[0]]) / 1000)
    if not np.allclose(concentrations, concentrations[0], rtol=1e-6, atol=0):
        raise ValueError("reversibility_analysis requires one concentration for automatic Sevcik Dapp estimation.")
    return float(concentrations[0]), f"species metadata ({species})"


def _wave_child_options(typed, *, cv_index, cv_count):
    allowed = set(PeakCurrentOptions().to_options_dict())
    options = {
        key: value
        for key, value in typed.to_options_dict().items()
        if key in allowed
    }
    options.update({"plot": False, "print": False, "plot all": False, "print all": False, "internal call": True, "new plot": False})
    for key in ("guess potential", "exact potential", "tangent potential"):
        value = options.get(key)
        if isinstance(value, (list, tuple, np.ndarray)) and len(value) == cv_count:
            options[key] = value[cv_index]
    return options


def _extract_wave_table(cv_list, typed, *, num_electrons):
    rows = []
    for index, cv_obj in enumerate(cv_list):
        with python_warnings.catch_warnings():
            python_warnings.filterwarnings(
                "ignore",
                message=r"W1/2 is unavailable.*",
                category=UserWarning,
            )
            wave = cv_obj.wave_info(
                _wave_child_options(typed, cv_index=index, cv_count=len(cv_list))
            )
        delta_ep = abs(float(wave["Epa"]) - float(wave["Epc"]))
        n_delta_ep_mv = float(num_electrons) * delta_ep * 1000
        nicholson = _nicholson_evidence(n_delta_ep_mv)
        row = {
            "name": getattr(cv_obj, "name", f"CV {index + 1}"),
            "scan rate / V s^-1": float(cv_obj.scan_rate),
            "E1/2 / V": _finite_float(wave.get("E(1/2)")),
            "Delta Ep / V": delta_ep,
            "n Delta Ep / mV": n_delta_ep_mv,
            "cathodic segment": wave.get("cathodic segment"),
            "Epc / V": _finite_float(wave.get("Epc")),
            "ipc / A": _finite_float(wave.get("ipc")),
            "Epc-Epc/2 / V": abs(_finite_float(wave.get("Δ(Epc - Epc/2)"))),
            "W1/2,c / V": _finite_float(wave.get("W1/2,c")),
            "cathodic width status": wave.get("cathodic width status", "Unavailable"),
            "anodic segment": wave.get("anodic segment"),
            "Epa / V": _finite_float(wave.get("Epa")),
            "ipa / A": _finite_float(wave.get("ipa")),
            "Epa-Epa/2 / V": abs(_finite_float(wave.get("Δ(Epa - Epa/2)"))),
            "W1/2,a / V": _finite_float(wave.get("W1/2,a")),
            "anodic width status": wave.get("anodic width status", "Unavailable"),
            "|ipa/ipc|": _finite_float(wave.get("|ipa/ipc|")),
            "psi": nicholson["psi"],
            "Lambda": nicholson["Lambda"],
            "Nicholson eligible": nicholson["eligible"],
            "Nicholson status": nicholson["reason"],
        }
        rows.append(row)
    return pd.DataFrame(rows)


def _irreversible_asymptote_evidence(rate_means, *, num_electrons, temperature, tolerance):
    n = float(num_electrons)
    estimates = []
    details = {}
    branch_columns = {
        "cathodic": ("Epc-Epc/2 / V", "Epc / V"),
        "anodic": ("Epa-Epa/2 / V", "Epa / V"),
    }
    for branch, (half_column, potential_column) in branch_columns.items():
        half_values = pd.to_numeric(rate_means[half_column], errors="coerce")
        finite_half = half_values[np.isfinite(half_values) & (half_values > 0)]
        alpha_half = (
            1.857 * R * float(temperature) / (n * F * float(finite_half.mean()))
            if len(finite_half)
            else np.nan
        )
        fit = _linear_regression(
            np.log10(rate_means["scan rate / V s^-1"]),
            rate_means[potential_column],
        )
        alpha_slope = (
            2.303 * R * float(temperature) / (2 * n * F * abs(fit["slope"]))
            if np.isfinite(fit["slope"]) and fit["slope"] != 0
            else np.nan
        )
        status, disagreement = _agreement_status(alpha_half, alpha_slope, tolerance=tolerance)
        physical = all(np.isfinite(value) and 0 < value <= 1 for value in (alpha_half, alpha_slope))
        verified = bool(physical and status == "agree")
        details[branch] = {
            "alpha from Ep-Ep/2": alpha_half,
            "alpha from Ep-v": alpha_slope,
            "agreement": status,
            "disagreement fraction": disagreement,
            "peak-potential fit": fit,
            "verified": verified,
        }
        if verified:
            estimates.extend([alpha_half, alpha_slope])
    return {
        "verified": bool(estimates),
        "alpha": float(np.mean(estimates)) if estimates else np.nan,
        "branches": details,
    }


def _electrochemical_decision(rate_means, *, alpha, asymptote):
    regions = []
    for _, row in rate_means.iterrows():
        lam = row.get("Lambda", np.nan)
        if np.isposinf(lam):
            regions.append("reversible")
        elif np.isfinite(lam):
            regions.append(_matsuda_ayabe_region(lam, alpha=alpha))
        elif row.get("n Delta Ep / mV", np.nan) > 212:
            regions.append("irreversible candidate")
        else:
            regions.append("unavailable")
    resolved = set(regions) - {"unavailable"}
    counts = {region: regions.count(region) for region in sorted(resolved)}
    rates = pd.to_numeric(rate_means["scan rate / V s^-1"], errors="coerce")
    lambdas = pd.to_numeric(rate_means["Lambda"], errors="coerce")
    finite_rates = rates[np.isfinite(rates) & (rates > 0)]
    finite_lambdas = lambdas[np.isfinite(lambdas)]
    rate_text = (
        f"{float(finite_rates.min()):.4g} to {float(finite_rates.max()):.4g} V/s"
        if len(finite_rates)
        else "an unavailable scan-rate range"
    )
    if len(finite_lambdas):
        lambda_text = (
            f"Lambda={float(finite_lambdas.min()):.4g} to "
            f"{float(finite_lambdas.max()):.4g}"
        )
        if np.isposinf(lambdas).any():
            lambda_text = f"{lambda_text}, with additional values above the reversible limit"
    elif np.isposinf(lambdas).any():
        lambda_text = "Lambda is above the reversible limit"
    else:
        lambda_text = "Lambda is unavailable"
    count_text = " and ".join(
        f"{count} {region}"
        for region, count in counts.items()
    )
    quantitative_reason = (
        f"Matsuda-Ayabe {lambda_text} across {rate_text}: "
        f"{count_text} scan-rate mean{'s' if len(regions) != 1 else ''}"
    )
    if resolved == {"reversible"}:
        conclusion = "reversible"
        reason = quantitative_reason
    elif resolved == {"quasi-reversible"}:
        conclusion = "quasi-reversible"
        reason = quantitative_reason
    elif resolved <= {"reversible", "quasi-reversible"} and resolved == {
        "reversible",
        "quasi-reversible",
    }:
        conclusion = "reversible-to-quasi-reversible transition"
        reason = quantitative_reason
    elif resolved & {"irreversible", "irreversible candidate"} and asymptote["verified"]:
        conclusion = "irreversible behavior indicated"
        reason = (
            f"{quantitative_reason}; large peak separation is corroborated by "
            "irreversible Ep-Ep/2 and Ep-v asymptotes"
        )
    elif resolved & {"irreversible", "irreversible candidate"}:
        conclusion = "quasi-reversible/irreversible transition"
        reason = (
            f"{quantitative_reason}; irreversible asymptotic verification is incomplete"
        )
    else:
        conclusion = "indeterminate"
        reason = f"insufficient dimensionless or asymptotic evidence ({quantitative_reason})"
    return {
        "conclusion": conclusion,
        "reason": reason,
        "regions": regions,
        "region counts": counts,
        "Lambda minimum": float(finite_lambdas.min()) if len(finite_lambdas) else np.nan,
        "Lambda maximum": float(finite_lambdas.max()) if len(finite_lambdas) else np.nan,
    }


def _trumpet_k0_estimate(rate_means, *, diffusion, num_electrons, temperature):
    log_rate = np.log10(rate_means["scan rate / V s^-1"].to_numpy(dtype=float))
    fits = {
        "cathodic": _linear_regression(log_rate, rate_means["Epc / V"]),
        "anodic": _linear_regression(log_rate, rate_means["Epa / V"]),
    }
    cathodic = fits["cathodic"]
    anodic = fits["anodic"]
    slopes = [cathodic["slope"], anodic["slope"]]
    if not (np.isfinite(slopes).all() and cathodic["slope"] < 0 < anodic["slope"]):
        return {"k0 / cm s^-1": np.nan, "eligible": False, "reason": "branch slopes do not diverge", "fits": fits}
    alpha = -R * float(temperature) * np.log(10) / (2 * cathodic["slope"] * F)
    beta = R * float(temperature) * np.log(10) / (2 * anodic["slope"] * F)
    denominator = cathodic["slope"] - anodic["slope"]
    intercept_x = (anodic["intercept"] - cathodic["intercept"]) / denominator if denominator else np.nan
    eligible = all(np.isfinite(value) and value > 0 for value in (diffusion, alpha, beta)) and np.isfinite(intercept_x)
    k0 = (
        10 ** (0.78 + intercept_x / 2)
        / np.sqrt(R * float(temperature) / (alpha * F * float(diffusion)))
        if eligible
        else np.nan
    )
    return {
        "k0 / cm s^-1": float(k0),
        "eligible": bool(eligible),
        "alpha": float(alpha),
        "beta": float(beta),
        "intersection log10 V/s": float(intercept_x),
        "fits": fits,
        "reason": "eligible" if eligible else "unphysical or incomplete trumpet fit",
    }


def _effective_surface_separation_tolerance(cv_list, configured):
    increments = []
    for cv_obj in cv_list:
        potential, _current = cv_obj.analysis_segment_data({})
        differences = np.abs(np.diff(np.asarray(potential, dtype=float)))
        increments.extend(differences[np.isfinite(differences) & (differences > 0)])
    resolution_limit = 3 * float(np.median(increments)) if increments else 0.0
    return max(float(configured), resolution_limit)


def _surface_laviron_estimate(rate_means, *, num_electrons, temperature):
    high = rate_means.loc[rate_means["n Delta Ep / mV"] > 200].copy()
    if len(high) < 2:
        return {"eligible": False, "k0 / s^-1": np.nan, "reason": "fewer than two points have nDeltaEp > 200 mV"}
    log_rate = np.log(high["scan rate / V s^-1"].to_numpy(dtype=float))
    fits = {
        "cathodic": _linear_regression(log_rate, high["Epc / V"]),
        "anodic": _linear_regression(log_rate, high["Epa / V"]),
    }
    cathodic = fits["cathodic"]
    anodic = fits["anodic"]
    slopes = [cathodic["slope"], anodic["slope"]]
    if not (np.isfinite(slopes).all() and cathodic["slope"] < 0 < anodic["slope"]):
        return {"eligible": False, "k0 / s^-1": np.nan, "reason": "high-separation branch slopes do not diverge", "fits": fits}
    n = float(num_electrons)
    temperature = float(temperature)
    alpha = R * temperature / (n * F * anodic["slope"])
    beta = -R * temperature / (n * F * cathodic["slope"])
    physical = all(np.isfinite(value) and 0 < value < 1 for value in (alpha, beta))
    if not physical:
        return {"eligible": False, "k0 / s^-1": np.nan, "reason": "Laviron alpha/beta are not physical", "alpha": alpha, "beta": beta, "fits": fits}
    anodic_scale = R * temperature / (alpha * n * F)
    cathodic_scale = R * temperature / (beta * n * F)
    log_a = np.log(alpha * n * F / (R * temperature))
    log_b = np.log(beta * n * F / (R * temperature))
    log_k0 = (
        anodic_scale * log_a
        + cathodic_scale * log_b
        - (anodic["intercept"] - cathodic["intercept"])
    ) / (anodic_scale + cathodic_scale)
    return {
        "eligible": True,
        "k0 / s^-1": float(np.exp(log_k0)),
        "alpha": float(alpha),
        "beta": float(beta),
        "fits": fits,
        "reason": "Laviron high-separation region",
        "points": int(len(high)),
    }


def _surface_phase_evidence(cv_list, rate_means, resolved, *, num_electrons, temperature):
    effective_tolerance = _effective_surface_separation_tolerance(
        cv_list, resolved["peak separation tolerance"]
    )
    peak_fits = {
        branch: _linear_regression(
            rate_means["scan rate / V s^-1"],
            np.abs(rate_means[current_column]),
        )
        for branch, current_column in {
            "cathodic": "ipc / A",
            "anodic": "ipa / A",
        }.items()
    }
    linear = all(
        np.isfinite(fit["r2"]) and fit["r2"] >= float(resolved["min r2"])
        for fit in peak_fits.values()
    )
    maximum_separation = float(np.nanmax(rate_means["Delta Ep / V"]))
    laviron = _surface_laviron_estimate(
        rate_means,
        num_electrons=num_electrons,
        temperature=temperature,
    )
    if maximum_separation <= effective_tolerance and linear:
        conclusion = "reversible"
        reason = "peak separation is within the effective zero-separation tolerance and ip is linear in scan rate"
    elif laviron["eligible"]:
        conclusion = "irreversible behavior indicated"
        reason = "the high-separation surface series satisfies the Laviron asymptotic branch test"
    elif linear:
        conclusion = "surface quasi-reversible transition"
        reason = "ip remains linear in scan rate but peak separation exceeds the effective zero-separation tolerance"
    else:
        conclusion = "indeterminate"
        reason = "surface ip-v or peak-separation evidence is insufficient"
    return {
        "electrochemical": {"conclusion": conclusion, "reason": reason, "regions": []},
        "surface": {
            "effective peak separation tolerance / V": effective_tolerance,
            "maximum peak separation / V": maximum_separation,
            "ip-v fits": peak_fits,
            "ip-v linear": linear,
            "Laviron": laviron,
        },
        "D / cm^2 s^-1": np.nan,
        "D source": "not applicable",
        "C / mol cm^-3": np.nan,
        "concentration source": "not applicable",
        "Sevcik Dapp": {},
        "irreversible asymptote": {},
        "Nicholson k0 values": [],
        "Nicholson eligible points": 0,
        "Nicholson k0 / cm s^-1": np.nan,
        "trumpet": {},
        "trumpet k0 / cm s^-1": np.nan,
        "k0 / cm s^-1": np.nan,
        "k0 / s^-1": laviron.get("k0 / s^-1", np.nan),
        "k0 lower bound / cm s^-1": np.nan,
        "preferred k0 source": "Laviron" if laviron["eligible"] else "unresolved",
        "warnings": [],
    }


def _bulk_phase_evidence(cv_list, rate_means, resolved, *, num_electrons, temperature, area, tolerance):
    asymptote = _irreversible_asymptote_evidence(
        rate_means,
        num_electrons=num_electrons,
        temperature=temperature,
        tolerance=tolerance,
    )
    electrochemical = _electrochemical_decision(
        rate_means,
        alpha=float(resolved["alpha"]),
        asymptote=asymptote,
    )
    warning_messages = []
    diffusion = _finite_float(resolved.get("D"))
    diffusion_source = "option" if np.isfinite(diffusion) and diffusion > 0 else "unavailable"
    sevcik = {}
    concentration, concentration_source = _resolve_bulk_concentration(cv_list, resolved)
    if not (np.isfinite(diffusion) and diffusion > 0) and np.isfinite(area) and np.isfinite(concentration):
        branch_estimates = {
            branch: _branch_sevcik_diffusion(
                rate_means["scan rate / V s^-1"],
                rate_means[current_column],
                num_electrons=num_electrons,
                temperature=temperature,
                electrode_area=area,
                concentration=concentration,
            )
            for branch, current_column in {
                "cathodic": "ipc / A",
                "anodic": "ipa / A",
            }.items()
        }
        status, difference = _agreement_status(
            branch_estimates["cathodic"]["D / cm^2 s^-1"],
            branch_estimates["anodic"]["D / cm^2 s^-1"],
            tolerance=tolerance,
        )
        sevcik = {"branches": branch_estimates, "agreement": status, "disagreement fraction": difference}
        if status == "agree" and all(
            fit["r2"] >= float(resolved["min r2"]) for fit in branch_estimates.values()
        ):
            diffusion = float(np.mean([fit["D / cm^2 s^-1"] for fit in branch_estimates.values()]))
            diffusion_source = "branch Sevcik Dapp"
        else:
            warning_messages.append("Branch Sevcik Dapp estimates did not provide an agreed, high-R2 diffusion coefficient.")
    nicholson_rows = []
    if np.isfinite(diffusion) and diffusion > 0:
        psi_values = pd.to_numeric(rate_means["psi"], errors="coerce")
        nicholson_mask = (
            np.isfinite(psi_values)
            & (psi_values >= 0.1)
            & (psi_values <= 7)
        )
        for _, row in rate_means.loc[nicholson_mask].iterrows():
            k0 = _k0_from_lambda(
                row["Lambda"],
                diffusion=diffusion,
                scan_rate=row["scan rate / V s^-1"],
                num_electrons=num_electrons,
                temperature=temperature,
            )
            if np.isfinite(k0):
                nicholson_rows.append(k0)
    nicholson_k0 = float(np.mean(nicholson_rows)) if nicholson_rows else np.nan
    trumpet = _trumpet_k0_estimate(
        rate_means,
        diffusion=diffusion,
        num_electrons=num_electrons,
        temperature=temperature,
    ) if np.isfinite(diffusion) and diffusion > 0 else {"eligible": False, "k0 / cm s^-1": np.nan, "reason": "D unavailable"}
    k0 = np.nan
    lower_bound = np.nan
    preferred_source = "unresolved"
    if np.isfinite(nicholson_k0):
        k0 = nicholson_k0
        preferred_source = "Nicholson"
    if trumpet.get("eligible"):
        if np.isfinite(k0):
            status, _difference = _agreement_status(k0, trumpet["k0 / cm s^-1"], tolerance=tolerance)
            if status == "disagree":
                warning_messages.append("Nicholson and trumpet k0 estimates disagree beyond the configured tolerance.")
        else:
            k0 = trumpet["k0 / cm s^-1"]
            preferred_source = "trumpet"
    if electrochemical["conclusion"] == "reversible" and np.isfinite(diffusion) and diffusion > 0:
        lower_bound = _k0_from_lambda(
            15,
            diffusion=diffusion,
            scan_rate=float(np.nanmax(rate_means["scan rate / V s^-1"])),
            num_electrons=num_electrons,
            temperature=temperature,
        )
        k0 = np.nan
        preferred_source = "reversible lower bound"
    return {
        "electrochemical": electrochemical,
        "surface": {},
        "D / cm^2 s^-1": diffusion,
        "D source": diffusion_source,
        "C / mol cm^-3": concentration,
        "concentration source": concentration_source,
        "Sevcik Dapp": sevcik,
        "irreversible asymptote": asymptote,
        "Nicholson k0 values": nicholson_rows,
        "Nicholson eligible points": int(len(nicholson_rows)),
        "Nicholson k0 / cm s^-1": nicholson_k0,
        "trumpet": trumpet,
        "trumpet k0 / cm s^-1": trumpet.get("k0 / cm s^-1", np.nan),
        "k0 / cm s^-1": k0,
        "k0 / s^-1": np.nan,
        "k0 lower bound / cm s^-1": lower_bound,
        "preferred k0 source": preferred_source,
        "warnings": warning_messages,
    }


def _reversibility_display_data_table(result, options=None):
    """Return the compact, phase-specific evidence view used by print all."""
    table = result.table.copy()
    phase = result.summary.get("phase", "bulk")
    if phase == "surface":
        surface = result.diagnostics.get("decision tree", {}).get("surface", {})
        tolerance = _finite_float(
            surface.get("effective peak separation tolerance / V")
        )

        def surface_region(row):
            delta_ep = _finite_float(row.get("Delta Ep / V"))
            n_delta_ep = _finite_float(row.get("n Delta Ep / mV"))
            if np.isfinite(tolerance) and np.isfinite(delta_ep) and delta_ep <= tolerance:
                return "reversible"
            if np.isfinite(n_delta_ep) and n_delta_ep > 200:
                return "Laviron-eligible"
            if np.isfinite(delta_ep):
                return "quasi-reversible"
            return "unavailable"

        table["Surface Region"] = [
            surface_region(row) for _, row in table.iterrows()
        ]
        table["|ipc| / A"] = pd.to_numeric(table["ipc / A"], errors="coerce").abs()
        table["|ipa| / A"] = pd.to_numeric(table["ipa / A"], errors="coerce").abs()
        columns = [
            "name",
            "scan rate / V s^-1",
            "E1/2 / V",
            "n Delta Ep / mV",
            "|ipc| / A",
            "|ipa| / A",
            "|ipa/ipc|",
            "Surface Region",
        ]
        return _conditional_analysis_name_column(
            table.loc[:, columns],
            ["scan rate / V s^-1"],
            options,
        )

    rate_means = result.diagnostics.get("rate means", pd.DataFrame())
    regions = (
        result.diagnostics.get("decision tree", {})
        .get("electrochemical", {})
        .get("regions", [])
    )
    region_by_rate = {
        round(float(rate), 12): region
        for rate, region in zip(
            rate_means.get("scan rate / V s^-1", []),
            regions,
        )
    }
    table["ET Region"] = [
        region_by_rate.get(round(float(rate), 12), "unavailable")
        for rate in table["scan rate / V s^-1"]
    ]

    def nicholson_use(status):
        text = str(status or "").strip()
        lower = text.lower()
        if lower == "nicholson eligible":
            return "Nicholson eligible"
        if "too reversible" in lower:
            return "Not used: too reversible"
        if "too irreversible" in lower:
            return "Not used: too irreversible"
        return "Not used: unavailable"

    table["Nicholson Use"] = table["Nicholson status"].map(nicholson_use)
    columns = [
        "name",
        "scan rate / V s^-1",
        "E1/2 / V",
        "n Delta Ep / mV",
        "|ipa/ipc|",
        "psi",
        "Lambda",
        "ET Region",
        "Nicholson Use",
    ]
    return _conditional_analysis_name_column(
        table.loc[:, columns],
        ["scan rate / V s^-1"],
        options,
    )


def _print_reversibility(result, options):
    sig_figs = options.get("sig figs", 4)
    phase = result.summary.get("phase")
    params = pd.DataFrame(
        [
            {"Parameter": "Phase", "Symbol": "", "Value": result.summary.get("phase")},
            {"Parameter": "Electron Count", "Symbol": "n", "Value": result.summary.get("num electrons")},
            {
                "Parameter": "Temperature",
                "Symbol": "T",
                "Value": _format_result_value(
                    result.summary.get("temperature / K"), unit="K", sig_figs=sig_figs
                ),
            },
            {
                "Parameter": "Diffusion Coefficient",
                "Symbol": "D",
                "Value": _format_result_value(
                    result.summary.get("D / cm^2 s^-1"),
                    unit="cm^2/s",
                    sig_figs=sig_figs,
                ),
            },
            {
                "Parameter": "Electrode Area",
                "Symbol": "S",
                "Value": _format_result_value(
                    result.summary.get("electrode area / cm^2"),
                    unit="cm^2",
                    sig_figs=sig_figs,
                ),
            },
            {"Parameter": "Scan Rates", "Symbol": "ν", "Value": result.summary.get("scan-rate range")},
            {"Parameter": "Segments", "Symbol": "", "Value": result.summary.get("segments")},
            {
                "Parameter": "Agreement Tolerance",
                "Symbol": "",
                "Value": _format_result_value(
                    result.summary.get("agreement tolerance"), sig_figs=sig_figs
                ),
            },
            {
                "Parameter": "Current Ratio Tolerance",
                "Symbol": "",
                "Value": _format_result_value(
                    result.summary.get("current ratio tolerance"), sig_figs=sig_figs
                ),
            },
        ]
    )
    if np.isfinite(_finite_float(result.summary.get("k0 / cm s^-1"))):
        rate_evidence = _format_result_value(
            result.summary.get("k0 / cm s^-1"), unit="cm/s", sig_figs=sig_figs
        )
        if result.summary.get("preferred k0 source") == "Nicholson":
            count = int(result.summary.get("Nicholson eligible points", 0))
            rate_evidence += (
                f" from {count} Nicholson-eligible scan rate"
                f"{'s' if count != 1 else ''}"
            )
    elif np.isfinite(_finite_float(result.summary.get("k0 / s^-1"))):
        rate_evidence = _format_result_value(
            result.summary.get("k0 / s^-1"), unit="s^-1", sig_figs=sig_figs
        )
    elif np.isfinite(_finite_float(result.summary.get("k0 lower bound / cm s^-1"))):
        rate_evidence = _format_result_value(
            result.summary.get("k0 lower bound / cm s^-1"),
            unit="cm/s",
            sig_figs=sig_figs,
            prefix=">= ",
        )
    elif phase == "bulk" and not np.isfinite(
        _finite_float(result.summary.get("D / cm^2 s^-1"))
    ):
        rate_evidence = (
            "k0 unresolved: D is required; provide D, or provide electrode area "
            "and concentration/species metadata so eCAT can estimate Dapp by "
            "Sevcik analysis."
        )
    elif phase == "surface":
        rate_evidence = (
            "k0 unresolved: the scan-rate series does not contain a sufficient "
            "Laviron high-separation region."
        )
    else:
        rate_evidence = (
            "k0 unresolved: no eligible Nicholson or trumpet kinetic region was "
            "resolved from the supplied scan rates."
        )
    conclusions = pd.DataFrame(
        [
            {"Assessment": "Electron Transfer", "Conclusion": result.summary.get("electrochemical conclusion"), "Evidence": result.summary.get("electrochemical reason")},
            {"Assessment": "Chemical Reversibility", "Conclusion": result.summary.get("chemical conclusion"), "Evidence": result.summary.get("chemical reason")},
            {
                "Assessment": "Electron-Transfer Rate",
                "Conclusion": result.summary.get("preferred k0 source"),
                "Evidence": rate_evidence,
            },
        ]
    )
    _display_symbol_table(
        params,
        options,
        title="Reversibility Analysis Parameters",
        index=False,
        symbol_columns=("Symbol",),
        rich_text_columns=("Value",),
    )
    _display_reversibility_equations(result, options)
    _display_symbol_table(
        conclusions,
        options,
        title="Reversibility Analysis Summary",
        index=False,
        rich_text_columns=("Evidence",),
    )
    if options.get("print all", False):
        display_table = _reversibility_display_data_table(result, options)
        _display_symbol_table(
            display_table,
            options,
            title="Reversibility Analysis Data",
            index=False,
        )


def reversibility_analysis(cvs, options=None):
    """Assess electron-transfer and chemical reversibility across one scan-rate series.

    ``phase='bulk'`` applies Matsuda-Ayabe, Nicholson, Sevcik, trumpet, and
    irreversible-asymptote evidence as eligible. ``phase='surface'`` instead
    uses peak separation, peak-current linearity in scan rate, and the Laviron
    high-separation limit. Chemical reversibility is reported independently
    from tangent-corrected anodic/cathodic peak-current ratios.
    """
    typed = ReversibilityAnalysisOptions.from_options(options)
    resolved = typed.to_options_dict()
    cv_list, distinct_rates = _validate_series_input(
        cvs, analysis_name="reversibility_analysis"
    )
    if len(distinct_rates) < 5:
        python_warnings.warn(
            "Reversibility analysis has fewer than five distinct scan rates; trend and asymptotic fits are fragile.",
            UserWarning,
            stacklevel=2,
        )
    n = float(resolved["num electrons"])
    tolerance = float(resolved["agreement tolerance"])
    temperature, temperature_source = _shared_positive_metadata(
        cv_list, "temperature", resolved.get("temperature"), fallback=298.15
    )
    area, area_source = _shared_positive_metadata(
        cv_list, "electrode_area", resolved.get("electrode area")
    )
    table = _extract_wave_table(cv_list, typed, num_electrons=n)
    mean_columns = [
        "E1/2 / V", "Delta Ep / V", "n Delta Ep / mV", "Epc / V", "ipc / A",
        "Epc-Epc/2 / V", "W1/2,c / V", "Epa / V", "ipa / A",
        "Epa-Epa/2 / V", "W1/2,a / V", "|ipa/ipc|", "psi", "Lambda",
    ]
    rate_means = _rate_mean_table(table, mean_columns)
    chemical = _chemical_reversibility_decision(
        rate_means["scan rate / V s^-1"],
        rate_means["|ipa/ipc|"],
        tolerance=float(resolved["current ratio tolerance"]),
        min_r2=float(resolved["min r2"]),
    )
    if resolved["phase"] == "surface":
        kinetic = _surface_phase_evidence(
            cv_list,
            rate_means,
            resolved,
            num_electrons=n,
            temperature=temperature,
        )
    else:
        kinetic = _bulk_phase_evidence(
            cv_list,
            rate_means,
            resolved,
            num_electrons=n,
            temperature=temperature,
            area=area,
            tolerance=tolerance,
        )
    electrochemical = kinetic["electrochemical"]
    warning_messages = kinetic["warnings"]

    segments = sorted(
        set(pd.to_numeric(table["cathodic segment"], errors="coerce").dropna().astype(int))
        | set(pd.to_numeric(table["anodic segment"], errors="coerce").dropna().astype(int))
    )
    summary = {
        "analysis": "reversibility",
        "phase": resolved["phase"],
        "num electrons": n,
        "temperature / K": temperature,
        "temperature source": temperature_source,
        "electrode area / cm^2": area,
        "electrode area source": area_source,
        "C / mol cm^-3": kinetic["C / mol cm^-3"],
        "concentration source": kinetic["concentration source"],
        "D / cm^2 s^-1": kinetic["D / cm^2 s^-1"],
        "D source": kinetic["D source"],
        "distinct scan rates": int(len(distinct_rates)),
        "scan-rate range": f"{float(np.min(distinct_rates)):g} to {float(np.max(distinct_rates)):g} V/s",
        "segments": segments,
        "electrochemical conclusion": electrochemical["conclusion"],
        "electrochemical reason": electrochemical["reason"],
        "chemical conclusion": chemical["conclusion"],
        "chemical reason": chemical["reason"],
        "preferred k0 source": kinetic["preferred k0 source"],
        "agreement tolerance": tolerance,
        "current ratio tolerance": float(resolved["current ratio tolerance"]),
        "k0 / cm s^-1": kinetic["k0 / cm s^-1"],
        "k0 / s^-1": kinetic["k0 / s^-1"],
        "k0 lower bound / cm s^-1": kinetic["k0 lower bound / cm s^-1"],
        "Nicholson eligible points": kinetic["Nicholson eligible points"],
        "Nicholson k0 / cm s^-1": kinetic["Nicholson k0 / cm s^-1"],
        "trumpet k0 / cm s^-1": kinetic["trumpet k0 / cm s^-1"],
    }
    figures = []
    axes = []
    if resolved.get("plot", True):
        figure, plot_axes = plt.subplots(
            2,
            1,
            sharex=True,
            layout="constrained",
        )
        scan_rates = rate_means["scan rate / V s^-1"]
        plot_axes[0].scatter(scan_rates, rate_means["n Delta Ep / mV"])
        plot_axes[0].set_ylabel(r"$n\Delta E_p$ (mV)")
        plot_axes[1].scatter(scan_rates, rate_means["|ipa/ipc|"])
        plot_axes[1].axhline(1, color="black", linestyle="--", linewidth=1)
        _format_scan_rate_axis(
            plot_axes[1], scan_rates, sig_figs=resolved.get("sig figs", 4)
        )
        ratio_formatter = ScalarFormatter(useOffset=False)
        ratio_formatter.set_scientific(False)
        plot_axes[1].yaxis.set_major_formatter(ratio_formatter)
        plot_axes[1].set_xlabel(r"Scan Rate (V s$^{-1}$)")
        plot_axes[1].set_ylabel(r"$|i_{p,\mathrm{a}}/i_{p,\mathrm{c}}|$")
        figures.append(figure)
        axes.extend(plot_axes)
    if resolved.get("plot all", False):
        figure, axis = plt.subplots(layout="constrained")
        scan_rates = rate_means["scan rate / V s^-1"]
        axis.scatter(scan_rates, rate_means["Epc / V"], label=r"$E_{p,\mathrm{c}}$")
        axis.scatter(scan_rates, rate_means["Epa / V"], label=r"$E_{p,\mathrm{a}}$")
        _format_scan_rate_axis(axis, scan_rates, sig_figs=resolved.get("sig figs", 4))
        axis.set_xlabel(r"Scan Rate (V s$^{-1}$)")
        axis.set_ylabel(r"Peak Potential, $E_p$ (V)")
        axis.legend()
        figures.append(figure)
        axes.append(axis)

    result = AnalysisResult(
        {"data": table, "summary": summary},
        table=table,
        summary=summary,
        fits={"Nicholson k0 values": kinetic["Nicholson k0 values"], "trumpet": kinetic["trumpet"]},
        fit_table=rate_means,
        diagnostics={
            "rate means": rate_means,
            "decision tree": {
                "electrochemical": electrochemical,
                "chemical": chemical,
                "irreversible asymptote": kinetic["irreversible asymptote"],
                "Sevcik Dapp": kinetic["Sevcik Dapp"],
                "surface": kinetic["surface"],
            },
        },
        warnings=warning_messages,
        units={
            "scan rate / V s^-1": "V/s",
            "Delta Ep / V": "V",
            "n Delta Ep / mV": "mV",
            "ipc / A": "A",
            "ipa / A": "A",
        },
        figure=figures[0] if figures else None,
        axes=axes[0] if axes else None,
        figures=figures,
        analysis="reversibility",
    )
    for message in warning_messages:
        python_warnings.warn(message, UserWarning, stacklevel=2)
    if resolved.get("print", True):
        _print_reversibility(result, resolved)
    return result


def _surface_segments(options):
    segments = options.get("segments")
    if segments is None:
        segment = options.get("segment")
        return [1 if segment is None else int(segment)]
    if isinstance(segments, (int, np.integer)):
        return [int(segments)]
    values = [int(segment) for segment in segments]
    if not values:
        raise ValueError("surface_coverage_analysis requires at least one segment.")
    return values


def _surface_guesses(options, segments):
    exact = options.get("exact potential")
    guess = options.get("guess potential")
    selected = exact if exact is not None else guess
    if isinstance(selected, (list, tuple, np.ndarray)):
        values = list(selected)
        if len(values) == 1:
            values *= len(segments)
        if len(values) != len(segments):
            raise ValueError(
                "surface_coverage_analysis requires one potential or one potential per selected segment."
            )
    else:
        values = [selected] * len(segments)
    key = "exact potential" if exact is not None else "guess potential"
    return key, values


def _surface_peak_options(typed, *, segment, potential_key, potential):
    potential_overrides = {
        "guess_potential": None,
        "exact_potential": None,
    }
    if potential is not None:
        potential_overrides[potential_key] = potential
    return _project_options(
        PeakCurrentOptions,
        typed,
        plot=False,
        print=False,
        plot_all=False,
        print_all=False,
        internal_call=True,
        new_plot=False,
        plot_cv=False,
        segments=None,
        segment=int(segment),
        **potential_overrides,
    ).to_options_dict()


def _surface_integration_indices(potential, corrected, peak_index, integration_range):
    potential = np.asarray(potential, dtype=float)
    corrected = np.asarray(corrected, dtype=float)
    if integration_range != "auto":
        lower, upper = sorted(float(value) for value in integration_range)
        selected = np.flatnonzero(
            np.isfinite(potential)
            & np.isfinite(corrected)
            & (potential >= lower)
            & (potential <= upper)
        )
        if len(selected) < 2:
            raise ValueError(
                "No usable points fall within the requested integration range."
            )
        return int(selected[0]), int(selected[-1]), "explicit"

    peak_index = int(peak_index)
    peak_height = abs(float(corrected[peak_index]))
    if not np.isfinite(peak_height) or peak_height <= 0:
        raise ValueError("Automatic integration requires a finite nonzero corrected peak.")
    edge_n = max(5, min(25, len(corrected) // 10))
    edge_values = np.concatenate([corrected[:edge_n], corrected[-edge_n:]])
    edge_values = edge_values[np.isfinite(edge_values)]
    edge_center = float(np.median(edge_values)) if len(edge_values) else 0.0
    noise = (
        1.4826 * float(np.median(np.abs(edge_values - edge_center)))
        if len(edge_values)
        else 0.0
    )
    threshold = max(3 * noise, 0.01 * peak_height, np.finfo(float).eps)

    left_candidates = np.flatnonzero(np.abs(corrected[:peak_index] - edge_center) <= threshold)
    right_candidates = np.flatnonzero(
        np.abs(corrected[peak_index + 1 :] - edge_center) <= threshold
    )
    if len(left_candidates) == 0 or len(right_candidates) == 0:
        raise ValueError(
            "Automatic integration could not resolve both tangent-baseline returns. "
            "Pass an explicit 'integration range'."
        )
    left = int(left_candidates[-1])
    right = int(peak_index + 1 + right_candidates[0])
    if right - left < 2:
        raise ValueError("Automatic integration resolved too few points.")
    return left, right, "auto"


def _surface_combined_estimate(values, *, tolerance):
    finite = [float(value) for value in values if np.isfinite(_finite_float(value))]
    if not finite:
        return np.nan, "unavailable", np.nan
    if len(finite) == 1:
        return finite[0], "single estimate", np.nan
    statuses = []
    disagreements = []
    for index, first in enumerate(finite):
        for second in finite[index + 1 :]:
            status, disagreement = _agreement_status(
                first, second, tolerance=tolerance
            )
            statuses.append(status)
            disagreements.append(disagreement)
    if any(status == "disagree" for status in statuses):
        return np.nan, "disagree", float(np.nanmax(disagreements))
    return float(np.mean(finite)), "agree", float(np.nanmax(disagreements))


def _surface_fit_display_table(fit_table, options):
    """Use a compact vertical result table for a single fitted branch."""
    if len(fit_table) != 1:
        return fit_table.copy()
    row = fit_table.iloc[0]
    sig_figs = options.get("sig figs", 4)
    values = [
        ("Branch", row.get("branch", "Unavailable")),
        ("Segment", int(row["segment"]) if np.isfinite(_finite_float(row.get("segment"))) else "Unavailable"),
        ("Slope", _format_result_value(row.get("slope / A s V^-1"), unit="A s V^-1", sig_figs=sig_figs)),
        ("Intercept", _format_result_value(row.get("intercept / A"), unit="A", sig_figs=sig_figs)),
        ("R2", _format_result_value(row.get("R2"), sig_figs=sig_figs)),
        ("Gamma slope", _format_result_value(row.get("Gamma slope / mol cm^-2"), unit="mol cm^-2", sig_figs=sig_figs)),
        ("Loading slope", _format_result_value(row.get("Loading slope / mol"), unit="mol", sig_figs=sig_figs)),
        ("Fit points", int(row["fit points"]) if np.isfinite(_finite_float(row.get("fit points"))) else "Unavailable"),
    ]
    return pd.DataFrame(values, columns=["Metric", "Value"])


def _surface_coverage_display_data_table(table, options=None):
    """Return per-CV surface evidence with names only for duplicate contexts."""
    return _conditional_analysis_name_column(
        table,
        ["scan rate / V s^-1", "segment"],
        options,
    )


def _print_surface_coverage(result, options):
    sig_figs = options.get("sig figs", 4)
    parameters = pd.DataFrame(
        [
            {"Parameter": "Electron Count", "Symbol": "n", "Value": result.summary.get("num electrons")},
            {
                "Parameter": "Temperature",
                "Symbol": "T",
                "Value": _format_result_value(
                    result.summary.get("temperature / K"), unit="K", sig_figs=sig_figs
                ),
            },
            {
                "Parameter": "Electrode Area",
                "Symbol": "S",
                "Value": _format_result_value(
                    result.summary.get("electrode area / cm^2"),
                    unit="cm^2",
                    sig_figs=sig_figs,
                ),
            },
            {"Parameter": "Agreement Tolerance", "Symbol": "", "Value": result.summary.get("agreement tolerance")},
        ]
    )
    _display_symbol_table(
        parameters,
        options,
        title="Surface Coverage Parameters",
        index=False,
        symbol_columns=("Symbol",),
        rich_text_columns=("Value",),
    )
    _display_equation_bundle(
        "Surface Coverage Equations",
        _surface_coverage_equation_bundle(),
        options,
    )
    fit_display = _surface_fit_display_table(result.fit_table, options)
    _display_symbol_table(
        fit_display,
        options,
        title="Surface Coverage Summary",
        index=False,
        symbol_columns=("Metric",) if "Metric" in fit_display else (),
        rich_text_columns=("Value",) if "Value" in fit_display else (),
    )
    if options.get("print all", False):
        display_data = _surface_coverage_display_data_table(result.table, options)
        _display_symbol_table(
            display_data,
            options,
            title="Surface Coverage Data",
            index=False,
        )


def surface_coverage_analysis(cvs, options=None):
    """Estimate surface coverage and electroactive loading from a CV scan-rate series.

    The slope method fits tangent-corrected peak current against scan rate.
    The independent charge method integrates tangent-corrected current with
    respect to potential and divides by scan rate. Areal coverage requires an
    electrode area; total electroactive loading does not.
    """
    typed = SurfaceCoverageAnalysisOptions.from_options(options)
    resolved = typed.to_options_dict()
    cv_list, _distinct_rates = _validate_series_input(
        cvs, analysis_name="surface_coverage_analysis"
    )
    segments = _surface_segments(resolved)
    potential_key, potentials = _surface_guesses(resolved, segments)
    n = float(resolved.get("num electrons", 1))
    tolerance = float(resolved.get("agreement tolerance", 0.25))
    area = _finite_float(resolved.get("electrode area"))
    if not np.isfinite(area) or area <= 0:
        areas = [_finite_float(getattr(cv_obj, "electrode_area", None)) for cv_obj in cv_list]
        finite_areas = [value for value in areas if np.isfinite(value) and value > 0]
        area = finite_areas[0] if finite_areas and np.allclose(finite_areas, finite_areas[0]) else np.nan
    temperature = _finite_float(resolved.get("temperature"))
    if not np.isfinite(temperature) or temperature <= 0:
        temperatures = [_finite_float(getattr(cv_obj, "temperature", None)) for cv_obj in cv_list]
        finite_temperatures = [value for value in temperatures if np.isfinite(value) and value > 0]
        temperature = float(np.mean(finite_temperatures)) if finite_temperatures else 298.15

    rows = []
    integration_diagnostics = []
    warning_messages = []
    overlay_records = []
    for cv_obj in cv_list:
        scan_rate = _finite_float(getattr(cv_obj, "scan_rate", None))
        if not np.isfinite(scan_rate) or scan_rate <= 0:
            raise ValueError(
                f"surface_coverage_analysis requires a positive scan rate for '{getattr(cv_obj, 'name', 'unnamed')}'."
            )
        for segment, potential in zip(segments, potentials):
            peak_options = _surface_peak_options(
                typed,
                segment=segment,
                potential_key=potential_key,
                potential=potential,
            )
            peak_result = cv_obj.peak_current(peak_options)
            x, y = cv_obj.analysis_segment_data({"segment": segment})
            x = np.asarray(x, dtype=float)
            y = np.asarray(y, dtype=float)
            slope, intercept = peak_result["tangent line"]
            corrected = y - (float(slope) * x + float(intercept))
            charge_result = {
                "charge / C": np.nan,
                "coverage / mol cm^-2": np.nan,
                "loading / mol": np.nan,
            }
            integration_status = "ok"
            integration_source = ""
            left = right = None
            try:
                left, right, integration_source = _surface_integration_indices(
                    x,
                    corrected,
                    peak_result["Ep index"],
                    resolved.get("integration range", "auto"),
                )
                charge_result = _surface_charge_from_potential(
                    x[left : right + 1],
                    corrected[left : right + 1],
                    scan_rate=scan_rate,
                    num_electrons=n,
                    electrode_area=area,
                )
            except ValueError as exc:
                integration_status = "Unavailable"
                message = (
                    f"Surface charge integration is unavailable for '{getattr(cv_obj, 'name', 'unnamed')}', "
                    f"segment {segment}: {exc}"
                )
                warning_messages.append(message)
                python_warnings.warn(message, UserWarning, stacklevel=2)

            try:
                direction = cv_segment_scan_direction(cv_obj, segment)
                branch = "anodic" if direction == "increasing" else "cathodic"
            except ValueError:
                direction = "unknown"
                branch = "unknown"
            row = {
                "name": getattr(cv_obj, "name", "unnamed"),
                "scan rate / V s^-1": scan_rate,
                "segment": int(segment),
                "branch": branch,
                "scan direction": direction,
                "Ep / V": float(peak_result["Ep"]),
                "ip / A": float(peak_result["ip"]),
                "Q / C": charge_result["charge / C"],
                "Gamma charge / mol cm^-2": charge_result["coverage / mol cm^-2"],
                "Loading charge / mol": charge_result["loading / mol"],
                "integration status": integration_status,
                "integration source": integration_source,
            }
            rows.append(row)
            integration_diagnostics.append(
                {
                    **row,
                    "left index": left,
                    "right index": right,
                    "left potential / V": float(x[left]) if left is not None else np.nan,
                    "right potential / V": float(x[right]) if right is not None else np.nan,
                    "tangent slope / A V^-1": float(slope),
                    "tangent intercept / A": float(intercept),
                }
            )
            overlay_records.append((cv_obj, segment, x, y, slope, intercept, left, right))

    table = pd.DataFrame(rows)
    fits = {}
    fit_rows = []
    branch_coverage = []
    branch_loading = []
    for segment in segments:
        branch_table = table.loc[table["segment"] == segment].sort_values(
            "scan rate / V s^-1"
        )
        fit = _surface_coverage_from_slope(
            branch_table["scan rate / V s^-1"],
            np.abs(branch_table["ip / A"]),
            num_electrons=n,
            temperature=temperature,
            electrode_area=area,
        )
        branch = str(branch_table["branch"].iloc[0])
        label = f"{branch.title()} (segment {segment})"
        fits[label] = fit
        branch_coverage.append(fit["coverage / mol cm^-2"])
        branch_loading.append(fit["loading / mol"])
        fit_rows.append(
            {
                "branch": branch,
                "segment": segment,
                "slope / A s V^-1": fit["slope / A s V^-1"],
                "intercept / A": fit["intercept / A"],
                "R2": fit["r2"],
                "Gamma slope / mol cm^-2": fit["coverage / mol cm^-2"],
                "Loading slope / mol": fit["loading / mol"],
                "fit points": len(branch_table),
            }
        )
        if fit["r2"] < float(resolved.get("min r2", 0.98)):
            message = f"Surface coverage {branch} slope fit has R2 = {fit['r2']:.4g}, below the configured threshold."
            warning_messages.append(message)
            python_warnings.warn(message, UserWarning, stacklevel=2)

    coverage_slope, branch_slope_status, slope_branch_difference = _surface_combined_estimate(
        branch_coverage, tolerance=tolerance
    )
    loading_slope, loading_branch_status, _ = _surface_combined_estimate(
        branch_loading, tolerance=tolerance
    )
    charge_coverage_by_segment = [
        float(table.loc[table["segment"] == segment, "Gamma charge / mol cm^-2"].mean())
        for segment in segments
    ]
    charge_loading_by_segment = [
        float(table.loc[table["segment"] == segment, "Loading charge / mol"].mean())
        for segment in segments
    ]
    coverage_charge, branch_charge_status, charge_branch_difference = _surface_combined_estimate(
        charge_coverage_by_segment, tolerance=tolerance
    )
    loading_charge, _, _ = _surface_combined_estimate(
        charge_loading_by_segment, tolerance=tolerance
    )
    method_status, method_difference = _agreement_status(
        coverage_slope, coverage_charge, tolerance=tolerance
    )
    if method_status == "disagree":
        message = (
            "Slope-derived and charge-derived surface coverages disagree beyond the "
            f"configured {tolerance:g} tolerance; both estimates are retained."
        )
        warning_messages.append(message)
        python_warnings.warn(message, UserWarning, stacklevel=2)

    fit_table = pd.DataFrame(fit_rows)
    summary = {
        "analysis": "surface coverage",
        "num electrons": n,
        "temperature / K": temperature,
        "electrode area / cm^2": area,
        "agreement tolerance": tolerance,
        "coverage slope / mol cm^-2": coverage_slope,
        "loading slope / mol": loading_slope,
        "coverage charge / mol cm^-2": coverage_charge,
        "loading charge / mol": loading_charge,
        "coverage agreement": method_status,
        "coverage disagreement fraction": method_difference,
        "slope branch agreement": branch_slope_status,
        "charge branch agreement": branch_charge_status,
    }

    figures = []
    axes = []
    if resolved.get("plot", True):
        fit_currents = np.concatenate(
            [np.asarray(fit["y"], dtype=float) for fit in fits.values()]
        )
        current_scale, current_unit = _current_display_scale(
            fit_currents,
            resolved.get("y unit", "auto"),
        )
        figure, axis = plt.subplots(layout="constrained")
        for label, fit in fits.items():
            axis.scatter(fit["x"], np.asarray(fit["y"]) * current_scale, label=label)
            if resolved.get("plot fit", True):
                axis.plot(
                    fit["x"],
                    np.asarray(fit["y fit"]) * current_scale,
                    linestyle=resolved.get("fit linestyle", "--"),
                    linewidth=resolved.get("fit linewidth", 1),
                )
        axis.set_xlabel(r"Scan Rate (V s$^{-1}$)")
        axis.set_ylabel(rf"$|i_p|$ ({current_unit})")
        if len(fits) > 1 or resolved.get("fit label", False):
            axis.legend()
        figures.append(figure)
        axes.append(axis)
    if resolved.get("plot all", False):
        overlay_currents = np.concatenate(
            [np.asarray(record[3], dtype=float) for record in overlay_records]
        )
        current_scale, current_unit = _current_display_scale(
            overlay_currents,
            resolved.get("y unit", "auto"),
        )
        figure, axis = plt.subplots(layout="constrained")
        for cv_obj, segment, x, y, slope, intercept, left, right in overlay_records:
            scaled_y = np.asarray(y) * current_scale
            scaled_tangent = (slope * x + intercept) * current_scale
            axis.plot(
                x,
                scaled_y,
                label=f"{getattr(cv_obj, 'scan_rate', ''):g} V/s, seg {segment}",
            )
            axis.plot(
                x,
                scaled_tangent,
                linestyle="--",
                color="tab:red",
                alpha=0.5,
            )
            if left is not None and right is not None:
                axis.fill_between(
                    x[left : right + 1],
                    scaled_tangent[left : right + 1],
                    scaled_y[left : right + 1],
                    alpha=0.15,
                )
        axis.set_xlabel("Potential (V)")
        axis.set_ylabel(f"Current ({current_unit})")
        figures.append(figure)
        axes.append(axis)

    result = ScatterFitResult(
        table=table,
        fits=fits,
        fit_table=fit_table,
        summary=summary,
        diagnostics={
            "charge integrations": pd.DataFrame(integration_diagnostics),
            "slope branch disagreement": slope_branch_difference,
            "charge branch disagreement": charge_branch_difference,
            "loading branch agreement": loading_branch_status,
        },
        warnings=warning_messages,
        units={
            "scan rate / V s^-1": "V/s",
            "Ep / V": "V",
            "ip / A": "A",
            "Q / C": "C",
            "Gamma charge / mol cm^-2": "mol/cm^2",
            "Loading charge / mol": "mol",
        },
        figure=figures[0] if figures else None,
        axes=axes[0] if axes else None,
        figures=figures,
    )
    if resolved.get("print", True):
        _print_surface_coverage(result, resolved)
    return result


__all__ = ["reversibility_analysis", "surface_coverage_analysis"]
