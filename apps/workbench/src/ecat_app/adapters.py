"""Adapters between Dash state and the public eCAT API."""

from __future__ import annotations

import base64
import contextlib
from copy import deepcopy
from dataclasses import dataclass, field
import io
import importlib.util
from pathlib import Path
import re
import uuid

import pandas as pd
import matplotlib.pyplot as plt

import ecat as e

from .figures import figure_to_data_uri
from .workflow import AppWorkflow


SIMULATION_INSTALL_MESSAGE = (
    "Model simulation requires ElectroKitty. Install simulation support with "
    '`python -m pip install "ecat-electrochemistry[simulation]"`.'
)


def simulation_backend_available() -> bool:
    return importlib.util.find_spec("electrokitty") is not None


@dataclass
class LoadResult:
    objects: list[object] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    workflow: AppWorkflow = field(default_factory=AppWorkflow)
    status: str = ""


def validate_simulation_mechanism(source="preset", preset="E", custom_text=None) -> dict[str, object]:
    source = str(source or "preset").strip().lower()
    mechanism = custom_text if source == "custom" else preset
    mechanism = str(mechanism or "").strip()
    if not mechanism:
        return {
            "ok": False,
            "mechanism": "",
            "message": "Enter a custom mechanism or choose a preset.",
        }
    try:
        compiled = e.simulation.compile_mechanism(mechanism)
    except Exception as exc:
        return {
            "ok": False,
            "mechanism": mechanism,
            "message": f"Mechanism error: {exc}",
            "mechanism_details": [],
            "formatted_equations": [],
        }
    compiled_text = getattr(compiled, "mechanism", str(compiled))
    return {
        "ok": True,
        "mechanism": mechanism,
        "compiled": compiled_text,
        "mechanism_details": simulation_mechanism_detail_rows(compiled_text),
        "formatted_equations": formatted_simulation_equations(compiled_text),
        "message": "Mechanism ready.",
    }


def simulation_mechanism_detail_rows(mechanism_text: str) -> list[dict[str, object]]:
    """Return editable browser rows for ElectroKitty-style mechanism text."""
    rows: list[dict[str, object]] = []
    electron_index = 0
    reaction_index = 0
    for line in str(mechanism_text or "").splitlines():
        equation = line.strip()
        if not equation:
            continue
        prefix, _, body = equation.partition(":")
        kind_key = prefix.strip().lower()
        body = body.strip()
        reactants, products = _split_mechanism_body(body)
        if kind_key.startswith("e"):
            electron_index += 1
            parameter = f"kinetics.{electron_index - 1}"
            kind = "electron transfer"
            electrons = _electron_count_from_prefix(prefix)
        else:
            reaction_index += 1
            parameter = f"reactions.{reaction_index - 1}"
            kind = "chemical step"
            electrons = ""
        rows.append(
            {
                "index": len(rows) + 1,
                "kind": kind,
                "equation": equation,
                "reactants": reactants,
                "products": products,
                "electrons": electrons,
                "parameter": parameter,
                "notes": "",
            }
        )
    return rows


def formatted_simulation_equations(mechanism_text: str) -> list[str]:
    formatted = []
    for row in simulation_mechanism_detail_rows(mechanism_text):
        equation = str(row["equation"])
        prefix, _, _body = equation.partition(":")
        arrow = "⇌" if row["kind"] == "electron transfer" else "→"
        reactants = _display_species_label(row["reactants"])
        products = _display_species_label(row["products"])
        formatted.append(f"{prefix}: {reactants} {arrow} {products}")
    return formatted


def _display_species_label(value: object) -> str:
    """Use display-friendly species labels without rewriting chemical names."""
    text = str(value or "")
    return re.sub(r"\b[a-z]\b", lambda match: match.group(0).upper(), text)


def _split_mechanism_body(body: str) -> tuple[str, str]:
    for separator in ("<=>", "=>", "=", ">", "<"):
        if separator in body:
            left, right = body.split(separator, 1)
            return left.strip(), right.strip()
    return body.strip(), ""


def _electron_count_from_prefix(prefix: str) -> int | str:
    match = re.search(r"\((\d+)\)", prefix or "")
    if not match:
        return ""
    return int(match.group(1))


def simulation_params_from_table_rows(parameter_rows=None, cell_parameter_rows=None, setup_parameter_rows=None) -> dict[str, object]:
    values = {str(row.get("key") or row.get("name", "")): row.get("initial") for row in parameter_rows or []}
    row_by_name = {str(row.get("key") or row.get("name", "")): row for row in parameter_rows or []}
    kinetics_by_index: dict[int, dict[str, object]] = {}
    reactions_by_index: dict[int, dict[str, object]] = {}
    concentration_values = {"bulk": {}, "surface": {}}
    diffusion_values = {}
    for row in parameter_rows or []:
        path = str(row.get("path") or "")
        parts = path.split(".")
        if len(parts) == 3 and parts[0] == "kinetics":
            try:
                index = int(parts[1])
            except ValueError:
                continue
            kinetics_by_index.setdefault(index, {})[parts[2]] = _coerce_simulation_value(row.get("initial"))
        elif len(parts) == 3 and parts[0] == "reactions":
            try:
                index = int(parts[1])
            except ValueError:
                continue
            reactions_by_index.setdefault(index, {})[parts[2]] = _coerce_simulation_value(row.get("initial"))
        elif len(parts) >= 2 and parts[0] == "diffusion":
            species = str(row.get("species") or parts[1]).split(",")[0].strip() or parts[1]
            diffusion = _coerce_simulation_value(row.get("initial"))
            diffusion_unit = str(row.get("unit") or "")
            if "cm" in diffusion_unit and isinstance(diffusion, (int, float)):
                diffusion = diffusion * 1e-4
            diffusion_values[species] = diffusion
        elif len(parts) >= 3 and parts[0] == "concentrations":
            phase = str(row.get("phase") or parts[1] or "bulk").strip().lower() or "bulk"
            if phase not in concentration_values:
                concentration_values[phase] = {}
            species = str(row.get("species") or parts[2]).split(",")[0].strip() or parts[2]
            concentration_values[phase][species] = _coerce_simulation_value(row.get("initial"))

    concentration_row = row_by_name.get("C", {})
    concentration_species = str(concentration_row.get("species") or "a")
    concentration_phase = str(concentration_row.get("phase") or "bulk").strip().lower() or "bulk"
    concentration = _coerce_simulation_value(values.get("C", 1e-3))
    if not diffusion_values:
        for key, default_species in (("D_a", "a"), ("D_b", "b"), ("D_A", "A"), ("D_B", "B")):
            row = row_by_name.get(key, {})
            if not row and key not in values:
                continue
            species = str(row.get("species") or default_species).split(",")[0].strip() or default_species
            diffusion = _coerce_simulation_value(values.get(key, 1e-5))
            diffusion_unit = str(row.get("unit") or "")
            if "cm" in diffusion_unit and isinstance(diffusion, (int, float)):
                diffusion = diffusion * 1e-4
            diffusion_values[species] = diffusion
    if not diffusion_values and "D" in values:
        diffusion = _coerce_simulation_value(values.get("D", 1e-5))
        diffusion_unit = str(row_by_name.get("D", {}).get("unit") or "")
        if "cm" in diffusion_unit and isinstance(diffusion, (int, float)):
            diffusion = diffusion * 1e-4
        diffusion_values = {concentration_species: diffusion, "b": diffusion}

    concentrations = concentration_values
    if not any(concentrations.values()):
        concentrations = {"bulk": {"b": 0.0}}
        concentrations.setdefault(concentration_phase, {})[concentration_species] = concentration

    kinetics = [
        {
            "E0": _coerce_simulation_value(values.get("E0", -0.5)),
            "k0": _coerce_simulation_value(values.get("k0", 1e-3)),
            "alpha": _coerce_simulation_value(values.get("alpha", 0.5)),
        }
    ]
    if kinetics_by_index:
        kinetics = []
        for index in sorted(kinetics_by_index):
            entry = kinetics_by_index[index]
            kinetics.append(
                {
                    "E0": entry.get("E0", 0.0),
                    "k0": entry.get("k0", 1e-3),
                    "alpha": entry.get("alpha", 0.5),
                }
            )

    params = {
        "concentrations": concentrations,
        "diffusion": diffusion_values or {concentration_species: 1e-9, "b": 1e-9},
        "kinetics": kinetics,
        "cell": {},
        "spatial": "fast",
    }
    if reactions_by_index:
        params["reactions"] = [
            {
                "kf": entry.get("kf", 1.0),
                "kb": entry.get("kb", 0.0),
            }
            for _index, entry in sorted(reactions_by_index.items())
        ]
    for row in setup_parameter_rows or []:
        name = str(row.get("key") or row.get("name") or "")
        if name == "spatial":
            spatial_value = _coerce_simulation_value(row.get("initial"))
            if spatial_value not in (None, ""):
                params["spatial"] = spatial_value
        elif name.startswith("spatial."):
            spatial = params.setdefault("spatial", {})
            if not isinstance(spatial, dict):
                spatial = {}
                params["spatial"] = spatial
            spatial[name.split(".", 1)[1]] = _coerce_simulation_value(row.get("initial"))
    for row in cell_parameter_rows or []:
        name = row.get("key") or row.get("name")
        if not name:
            continue
        params["cell"][str(name)] = _coerce_simulation_value(row.get("initial"))
    return params


def fit_spec_from_table_rows(parameter_rows=None, cell_parameter_rows=None, setup_parameter_rows=None) -> dict[str, object]:
    """Build a public eCAT simulation fit spec from editable browser rows."""
    del setup_parameter_rows
    fit_paths: list[str] = []
    bounds: dict[str, list[float]] = {}
    for row in [*(parameter_rows or []), *(cell_parameter_rows or [])]:
        if not _truthy_fit_value(row.get("vary")):
            continue
        path = _simulation_fit_path_from_row(row)
        if not path:
            continue
        fit_paths.append(path)
        row_bounds = _fit_bounds_from_row(row)
        if row_bounds is not None:
            bounds[path] = row_bounds
    fit_paths = list(dict.fromkeys(fit_paths))
    fit: dict[str, object] = {"vary": fit_paths}
    if bounds:
        fit["bounds"] = {path: bounds[path] for path in fit_paths if path in bounds}
    return fit


def _truthy_fit_value(value) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "fit", "on"}
    return bool(value)


def _simulation_fit_path_from_row(row) -> str:
    group = str(row.get("group") or "").strip().lower()
    key = str(row.get("key") or "").strip()
    if group == "concentration":
        phase = str(row.get("phase") or "bulk").strip() or "bulk"
        species = str(row.get("species") or "a").split(",")[0].strip() or "a"
        return f"concentrations.{phase}.{species}"
    if group == "diffusion":
        species = str(row.get("species") or "").split(",")[0].strip()
        if not species and key.startswith("D_"):
            species = key.split("_", 1)[1]
        return f"diffusion.{species or 'a'}"
    if group == "cell" or key in {"T", "Ru", "Cdl", "A"}:
        return f"cell.{key or row.get('name')}"
    path = str(row.get("path") or "").strip()
    return path


def _fit_bounds_from_row(row) -> list[float] | None:
    lower = row.get("lower")
    upper = row.get("upper")
    if lower in (None, "") or upper in (None, ""):
        return None
    return [float(lower), float(upper)]


def _coerce_simulation_value(value):
    if value in (None, ""):
        return None
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.lower() == "auto":
            return "auto"
        try:
            return float(stripped)
        except ValueError:
            return stripped
    return value


def run_browser_simulate_cv(
    *,
    mode="scratch",
    mechanism="E",
    parameter_rows=None,
    cell_parameter_rows=None,
    setup_parameter_rows=None,
    program_settings=None,
    cv_data_settings=None,
    objects=None,
    plot_options=None,
    over_conditions=False,
    condition_settings=None,
) -> dict[str, object]:
    try:
        input_obj = _simulation_input_from_browser(mode, program_settings, cv_data_settings, objects)
        params = simulation_params_from_table_rows(parameter_rows, cell_parameter_rows, setup_parameter_rows)
        if str(mode or "scratch").strip().lower() != "cv" and params.get("cell", {}).get("Cdl") == "auto":
            params["cell"]["Cdl"] = 0.0
        if over_conditions:
            return _run_browser_simulate_cv_sweep(
                input_obj=input_obj,
                mode=mode,
                mechanism=mechanism,
                params=params,
                program_settings=program_settings,
                plot_options=plot_options,
                condition_settings=condition_settings,
            )
        result = e.simulation.simulate_cv(
            input_obj,
            mechanism,
            params,
            options={"plot": False, **dict(plot_options or {})},
        )
        plot_uri = _simulation_result_plot_uri(result, plot_options)
        return {
            "status": "ok",
            "message": "Simulation complete.",
            "summary": dict(getattr(result, "summary", {}) or {}),
            "plot": plot_uri,
            "params": params,
        }
    except Exception as exc:
        return {
            "status": "error",
            "message": f"Simulation error: {exc}",
            "summary": {},
            "plot": None,
            "params": simulation_params_from_table_rows(parameter_rows, cell_parameter_rows, setup_parameter_rows),
        }


def run_browser_fit_cv(
    *,
    fit_mode="single",
    fit_cv_index=0,
    mechanism="E",
    parameter_rows=None,
    cell_parameter_rows=None,
    setup_parameter_rows=None,
    cv_data_settings=None,
    objects=None,
    plot_options=None,
    method="least squares",
) -> dict[str, object]:
    if str(fit_mode or "single").strip().lower() != "single":
        return {
            "status": "blocked",
            "message": (
                "Multiple-CV fitting is not available in the app yet. "
                "Use the group-fitting API or notebook workflow."
            ),
            "summary": {},
            "plot": None,
            "parameter_rows": list(parameter_rows or []),
            "cell_parameter_rows": list(cell_parameter_rows or []),
            "params": simulation_params_from_table_rows(parameter_rows, cell_parameter_rows, setup_parameter_rows),
            "fit": fit_spec_from_table_rows(parameter_rows, cell_parameter_rows, setup_parameter_rows),
        }
    try:
        objects = list(objects or [])
        index = int(float(fit_cv_index or 0))
        if index < 0 or index >= len(objects):
            raise ValueError(f"CV index {index} is not loaded.")
        obj = objects[index]
        if type(obj).__name__ != "cv":
            raise ValueError(f"Object at index {index} is not a CV.")

        params = simulation_params_from_table_rows(parameter_rows, cell_parameter_rows, setup_parameter_rows)
        fit = fit_spec_from_table_rows(parameter_rows, cell_parameter_rows, setup_parameter_rows)
        if not fit.get("vary"):
            raise ValueError("Select at least one parameter in the Fit? column.")
        fit_cv_data_options = dict(cv_data_settings or {})
        fit_cv_data_options.pop("cv_index", None)
        fit_options = {
            "plot": False,
            "progress": False,
            "print setup": False,
            "print params": False,
            "print stats": False,
            "print corrections": False,
            "cv data": fit_cv_data_options,
        }
        result = e.simulation.fit_cv(
            obj,
            mechanism,
            params,
            fit=fit,
            options=fit_options,
            method=method,
        )
        plot_uri = _fit_result_plot_uri(result, plot_options)
        best_params = dict(getattr(result, "best_params", {}) or {})
        fitted_parameter_rows = _rows_with_fit_result(parameter_rows, best_params)
        fitted_cell_rows = _rows_with_fit_result(cell_parameter_rows, best_params)
        return {
            "status": "ok",
            "message": "Fit complete.",
            "summary": dict(getattr(result, "summary", {}) or {}),
            "plot": plot_uri,
            "params": best_params,
            "fit": fit,
            "fit_cv_index": index,
            "parameter_rows": fitted_parameter_rows,
            "cell_parameter_rows": fitted_cell_rows,
        }
    except Exception as exc:
        return {
            "status": "error",
            "message": f"Fit error: {exc}",
            "summary": {},
            "plot": None,
            "params": simulation_params_from_table_rows(parameter_rows, cell_parameter_rows, setup_parameter_rows),
            "fit": fit_spec_from_table_rows(parameter_rows, cell_parameter_rows, setup_parameter_rows),
            "parameter_rows": list(parameter_rows or []),
            "cell_parameter_rows": list(cell_parameter_rows or []),
        }


def _run_browser_simulate_cv_sweep(
    *,
    input_obj,
    mode,
    mechanism,
    params,
    program_settings=None,
    plot_options=None,
    condition_settings=None,
) -> dict[str, object]:
    settings = dict(condition_settings or {})
    axis = _normalize_simulation_condition_axis(settings.get("condition_axis"))
    values = _simulation_condition_values(settings)
    if not values:
        raise ValueError("Enter at least one condition value for the sweep.")

    results = []
    for value in values:
        condition_input, condition_params = _simulation_condition_input_and_params(
            input_obj,
            params,
            axis,
            value,
            mode=mode,
            program_settings=program_settings,
            condition_settings=settings,
        )
        result = e.simulation.simulate_cv(
            condition_input,
            mechanism,
            condition_params,
            options={"plot": False, **dict(plot_options or {})},
        )
        results.append((value, result))

    first_summary = dict(getattr(results[0][1], "summary", {}) or {})
    return {
        "status": "ok",
        "message": "Simulation sweep complete.",
        "summary": {
            **first_summary,
            "condition_axis": axis,
            "condition_values": values,
            "count": len(results),
        },
        "plot": _simulation_sweep_plot_uri(results, axis, plot_options),
        "params": params,
    }


def _normalize_simulation_condition_axis(axis) -> str:
    axis = str(axis or "scan_rate").strip().lower().replace(" ", "_").replace("-", "_")
    aliases = {
        "scanrate": "scan_rate",
        "scan": "scan_rate",
        "v": "scan_rate",
        "temperature": "temperature",
        "temp": "temperature",
        "t": "temperature",
        "concentration": "concentration",
        "conc": "concentration",
        "c": "concentration",
    }
    axis = aliases.get(axis, axis)
    if axis not in {"scan_rate", "temperature", "concentration"}:
        raise ValueError(f"Unsupported condition axis {axis!r}.")
    return axis


def _simulation_condition_values(raw_values) -> list[float]:
    if isinstance(raw_values, dict):
        settings = raw_values
        if settings.get("condition_values") not in (None, ""):
            return _simulation_condition_values(settings.get("condition_values"))
        start = _coerce_simulation_value(settings.get("condition_min"))
        end = _coerce_simulation_value(settings.get("condition_max"))
        count = settings.get("condition_count", 3)
        if start in (None, "") or end in (None, ""):
            return []
        count = max(1, int(float(count or 1)))
        if count == 1:
            return [float(start)]
        step = (float(end) - float(start)) / (count - 1)
        return [float(start) + step * index for index in range(count)]
    if raw_values is None:
        return []
    if isinstance(raw_values, (list, tuple)):
        tokens = raw_values
    else:
        tokens = re.split(r"[,\n;]+", str(raw_values))
    values = []
    for token in tokens:
        if token in (None, ""):
            continue
        values.append(float(str(token).strip()))
    return values


def _simulation_condition_input_and_params(
    input_obj,
    params,
    axis,
    value,
    *,
    mode,
    program_settings=None,
    condition_settings=None,
):
    condition_params = deepcopy(params)
    if axis == "temperature":
        condition_params.setdefault("cell", {})["T"] = value
        return input_obj, condition_params

    if axis == "concentration":
        species = (condition_settings or {}).get("condition_species") or _simulation_bulk_concentration_species(condition_params)
        condition_params.setdefault("concentrations", {}).setdefault("bulk", {})[species] = value
        return input_obj, condition_params

    if axis == "scan_rate":
        return _simulation_input_with_scan_rate(input_obj, value, mode=mode, program_settings=program_settings), condition_params

    raise ValueError(f"Unsupported condition axis {axis!r}.")


def _simulation_bulk_concentration_species(params) -> str:
    bulk = dict(((params or {}).get("concentrations") or {}).get("bulk") or {})
    for species in bulk:
        if str(species).lower() != "b":
            return species
    return next(iter(bulk), "a")


def _simulation_input_with_scan_rate(input_obj, scan_rate, *, mode, program_settings=None):
    if str(mode or "scratch").strip().lower() == "scratch":
        settings = dict(program_settings or {})
        settings["scan_rate"] = scan_rate
        return _simulation_input_from_browser("scratch", settings, None, None)
    if hasattr(input_obj, "with_scan_rate"):
        return input_obj.with_scan_rate(scan_rate)
    raise ValueError("The selected simulation input does not support scan-rate sweeps.")


def _simulation_sweep_plot_uri(results, axis, plot_options=None) -> str | None:
    fig, ax = plt.subplots()
    for value, result in results:
        data = getattr(result, "data", None)
        if data is None or "Potential" not in data or "Current" not in data:
            continue
        ax.plot(data["Potential"], data["Current"], label=f"{axis}={value:g}")
    ax.set_xlabel("Potential (V)")
    ax.set_ylabel("Current (A)")
    if results:
        ax.legend()
    try:
        return figure_to_data_uri(fig, format="png", dpi=150)
    finally:
        plt.close(fig)


def _simulation_input_from_browser(mode, program_settings=None, cv_data_settings=None, objects=None):
    mode = str(mode or "scratch").strip().lower()
    if mode == "cv":
        objects = list(objects or [])
        settings = dict(cv_data_settings or {})
        index = int(settings.pop("cv_index", 0) or 0)
        if index < 0 or index >= len(objects):
            raise ValueError(f"CV index {index} is not loaded.")
        obj = objects[index]
        if type(obj).__name__ != "cv":
            raise ValueError(f"Object at index {index} is not a CV.")
        if settings.get("scan rate") in (None, "") and settings.get("scan_rate") in (None, ""):
            program_scan_rate = (program_settings or {}).get("scan_rate")
            if program_scan_rate not in (None, ""):
                settings["scan rate"] = program_scan_rate
        return e.simulation.cv_data(obj, settings)

    settings = dict(program_settings or {})
    return e.simulation.cv_program(
        settings.get("Ei", 0.0),
        E_low=settings.get("E_low", -1.5),
        E_high=settings.get("E_high"),
        Ef=settings.get("Ef"),
        scan_rate=settings.get("scan_rate", 0.1),
        segments=int(settings.get("segments", 2) or 2),
        points_per_segment=int(settings.get("points_per_segment", 300) or 300),
        quiet_time=float(settings.get("quiet_time", 0) or 0),
    )


def render_browser_cv_data_program_plot(objects=None, cv_data_settings=None, program_settings=None) -> str:
    input_obj = _simulation_input_from_browser(
        "cv",
        program_settings=program_settings,
        cv_data_settings=cv_data_settings,
        objects=objects,
    )
    ax = input_obj.plot({"title": "Simulation Input", "plot": False})
    try:
        return figure_to_data_uri(ax.figure, format="png", dpi=150)
    finally:
        plt.close(ax.figure)


def _simulation_result_plot_uri(result, plot_options=None) -> str | None:
    ax = result.plot({"plot": False, **dict(plot_options or {})})
    try:
        return figure_to_data_uri(ax.figure, format="png", dpi=150)
    finally:
        plt.close(ax.figure)


def _fit_result_plot_uri(result, plot_options=None) -> str | None:
    ax = result.plot({"plot": False, **dict(plot_options or {})})
    try:
        return figure_to_data_uri(ax.figure, format="png", dpi=150)
    finally:
        plt.close(ax.figure)


def _rows_with_fit_result(rows, best_params) -> list[dict[str, object]]:
    fitted = []
    for row in rows or []:
        next_row = dict(row)
        path = _simulation_fit_path_from_row(next_row)
        value = _simulation_param_value(best_params, path)
        if value is not None:
            next_row["final"] = value
            next_row.setdefault("stderr", "")
            next_row["comment"] = _fit_row_comment(next_row, value)
        fitted.append(next_row)
    return fitted


def _simulation_param_value(params, path):
    if not path:
        return None
    current = params
    for token in str(path).split("."):
        if isinstance(current, list):
            try:
                current = current[int(token)]
            except (ValueError, IndexError):
                return None
        elif isinstance(current, dict):
            if token not in current:
                return None
            current = current[token]
        else:
            return None
    return current


def _fit_row_comment(row, value) -> str:
    bounds = _fit_bounds_from_row(row)
    if bounds is None:
        return ""
    lower, upper = bounds
    try:
        value = float(value)
    except (TypeError, ValueError):
        return ""
    tolerance = max(1e-12, abs(value) * 1e-9)
    if abs(value - lower) <= tolerance:
        return "hit lower bound"
    if abs(value - upper) <= tolerance:
        return "hit upper bound"
    return ""


def _import_options(import_options: dict | None) -> dict[str, object]:
    options = {
        "print": False,
        "reference mode": "none",
    }
    options.update(import_options or {})
    return options


def _load_one_file(path: Path, import_options: dict | None = None) -> tuple[object | None, str | None]:
    try:
        return e.echem.from_file(str(path), _import_options(import_options)), None
    except Exception as exc:
        return None, f"Warning: could not convert {path.name}: {exc}"


def _folder_load_status(objects) -> str:
    count = len(objects or [])
    if count == 0:
        return ""
    noun = "file" if count == 1 else "files"
    return f"{count} supported text {noun} found."


def load_local_path(path, recursive: bool = False, import_options: dict | None = None) -> LoadResult:
    path = Path(path).expanduser()
    import_options = _import_options(import_options)
    workflow = AppWorkflow(
        source_kind="local_path",
        source_path=str(path),
        recursive=recursive,
        import_options=import_options,
    )
    warnings: list[str] = []

    if path.is_dir():
        try:
            objects = e.get_data(
                {
                    "folder path": str(path),
                    "recursive search": recursive,
                    "sort keys": ["subfolder", "timestamp"],
                    **import_options,
                }
            )
        except Exception as exc:
            return LoadResult([], [f"Warning: could not load folder {path}: {exc}"], workflow)
        return LoadResult(
            list(objects or []),
            warnings,
            workflow,
            status=_folder_load_status(objects),
        )

    obj, warning = _load_one_file(path, import_options)
    if warning:
        warnings.append(warning)
    status = "1 file loaded." if obj is not None else ""
    return LoadResult([obj] if obj is not None else [], warnings, workflow, status=status)


def _safe_upload_name(filename: str) -> str:
    name = Path(filename or "uploaded.txt").name
    return re.sub(r"[^A-Za-z0-9._ -]+", "_", name) or "uploaded.txt"


def _decode_dash_upload(contents: str) -> bytes:
    if "," in contents:
        _, encoded = contents.split(",", 1)
    else:
        encoded = contents
    return base64.b64decode(encoded)


def load_uploaded_files(uploads, session_root=None, import_options: dict | None = None) -> LoadResult:
    session_root = Path(session_root or Path.cwd() / ".ecat-app-sessions")
    session_dir = session_root / uuid.uuid4().hex
    session_dir.mkdir(parents=True, exist_ok=True)
    import_options = _import_options(import_options)

    objects: list[object] = []
    warnings: list[str] = []

    for upload in uploads or []:
        filename = _safe_upload_name(upload.get("filename", "uploaded.txt"))
        destination = session_dir / filename
        try:
            destination.write_bytes(_decode_dash_upload(upload.get("contents", "")))
            obj, warning = _load_one_file(destination, import_options)
            if obj is None:
                warnings.append(warning or f"Warning: could not convert {filename}")
            else:
                objects.append(obj)
        except Exception as exc:
            warnings.append(f"Warning: could not convert {filename}: {exc}")

    workflow = AppWorkflow(
        source_kind="upload",
        source_path=str(session_dir),
        recursive=False,
        import_options=import_options,
    )
    count = len(objects)
    noun = "file" if count == 1 else "files"
    return LoadResult(objects, warnings, workflow, status=f"{count} uploaded {noun} loaded.")


def reload_workflow(workflow: AppWorkflow | dict) -> LoadResult:
    workflow = AppWorkflow.from_dict(workflow) if isinstance(workflow, dict) else workflow
    result = load_local_path(
        workflow.source_path,
        recursive=workflow.recursive if workflow.source_kind != "upload" else False,
        import_options=workflow.import_options,
    )
    result.workflow.source_kind = workflow.source_kind
    result.workflow.source_path = workflow.source_path
    result.workflow.reference_settings = dict(workflow.reference_settings)
    result.workflow.included_row_ids = list(workflow.included_row_ids)
    result.workflow.app_mode = workflow.app_mode
    return result


def _source_index_lookup(objects) -> dict[str, int]:
    lookup: dict[str, int] = {}
    for index, obj in enumerate(objects or []):
        filepath = getattr(obj, "filepath", None)
        if not filepath:
            continue
        path = Path(filepath)
        for key in {str(filepath), path.name}:
            lookup[key] = index
        try:
            lookup[str(path.expanduser().resolve())] = index
        except OSError:
            lookup[str(path.expanduser().absolute())] = index
    return lookup


def _reference_source_display(obj, source_lookup: dict[str, int]) -> int | str | None:
    source = getattr(obj, "reference_source_file", None)
    if not source:
        return None
    path = Path(source)
    keys = [str(source), path.name]
    try:
        keys.append(str(path.expanduser().resolve()))
    except OSError:
        keys.append(str(path.expanduser().absolute()))
    for key in keys:
        if key in source_lookup:
            return source_lookup[key]
    return path.name


def default_included_row_ids(objects, workflow: AppWorkflow | None = None) -> list[str]:
    objects = list(objects or [])
    if _is_default_fe_phoh_source(workflow):
        selected = [
            f"row-{index}"
            for index, obj in enumerate(objects)
            if _has_ferrocene(obj) and _has_default_fe_phoh_scan_window(obj)
        ]
        if selected:
            return selected
    return [f"row-{index}" for index in range(len(objects))]


def _is_default_fe_phoh_source(workflow: AppWorkflow | None) -> bool:
    source_path = str(getattr(workflow, "source_path", "") or "").lower()
    return "fe_phoh_cv" in source_path


def _has_ferrocene(obj) -> bool:
    compounds = getattr(obj, "compounds", None) or []
    if any(str(compound).strip().lower() == "fc" for compound in compounds):
        return True
    return "fc" in str(getattr(obj, "name", "") or "").lower()


def _has_default_fe_phoh_scan_window(obj) -> bool:
    name = str(getattr(obj, "name", "") or "").lower()
    return "-1.2_to_1v" in name or "-1.2 to 1" in name


def summarize_objects(objects) -> list[dict[str, object]]:
    rows = []
    source_lookup = _source_index_lookup(objects)
    for index, obj in enumerate(objects or []):
        filepath = getattr(obj, "filepath", None)
        rows.append(
            {
                "id": f"row-{index}",
                "index": index,
                "name": getattr(obj, "name", None),
                "filename": Path(filepath).name if filepath else None,
                "class": type(obj).__name__,
                "type": getattr(obj, "type", None),
                "software": getattr(obj, "software", None),
                "gas": getattr(obj, "gas", None),
                "solvent": getattr(obj, "solvent", None),
                "scan rate": getattr(obj, "scan_rate", None),
                "segments": getattr(obj, "segments", None),
                "reference shift": getattr(obj, "reference_shift", None),
                "reference mode": getattr(obj, "reference_mode", None),
                "reference label": getattr(obj, "reference_label", None),
                "reference source": _reference_source_display(obj, source_lookup),
            }
        )
    return rows


def available_filter_values(objects) -> dict[str, list[object]]:
    if not objects:
        return {}
    try:
        return e.get_available_filter_values(objects)
    except Exception:
        keys = ("gas", "solvent", "scan rate", "type", "software")
        values = {}
        for key in keys:
            attr = key.replace(" ", "_")
            unique = sorted({getattr(obj, attr, None) for obj in objects if getattr(obj, attr, None) is not None})
            if unique:
                values[key] = unique
        return values


def filter_and_group(objects, workflow: AppWorkflow):
    current = objects_for_analysis(objects, workflow)
    if workflow.filters:
        current = e.filter(current, workflow.filters, {"print": False})
    if workflow.sort_keys or workflow.group_keys:
        return e.sort_and_group(
            current,
            sort_keys=workflow.sort_keys or None,
            group_keys=workflow.group_keys or None,
            options={"print": False},
        )
    return current


def objects_for_analysis(objects, workflow: AppWorkflow):
    current = list(objects or [])
    if workflow.included_row_ids:
        included_indices = {
            int(str(row_id).replace("row-", ""))
            for row_id in workflow.included_row_ids
            if str(row_id).startswith("row-")
        }
        current = [obj for index, obj in enumerate(current) if index in included_indices]
    if workflow.sort_keys:
        current = e.sort(current, workflow.sort_keys, {"print": False})
    return current


def run_single_cv_analysis(obj, analyses: list[str] | None = None, options: dict | None = None) -> dict[str, object]:
    if type(obj).__name__ != "cv":
        return {
            "status": "skipped",
            "message": "Single-object analysis is CV-only in the eCAT app v1.",
            "results": [],
        }

    analyses = analyses or ["peak_potential", "peak_current"]
    options = dict(options or {})
    analysis_options = {
        key: value
        for key, value in {
            "guess potential": options.get("guess potential"),
            "tangent potential": options.get("tangent potential"),
            "segment": options.get("segment"),
        }.items()
        if value not in (None, "")
    }
    supported = {
        "peak_potential",
        "peak_current",
        "half_peak_potential",
        "half_wave_potential",
    }
    rows = []
    before_figures = set(plt.get_fignums())
    try:
        obj.plot({"new plot": True, "print": False, "legend": False})
    except Exception as exc:
        rows.append({"analysis": "cv_plot", "status": "error", "value": None, "message": str(exc)})
    for analysis in analyses:
        if analysis not in supported:
            rows.append({"analysis": analysis, "status": "skipped", "value": None, "message": "Unsupported"})
            continue
        method = getattr(obj, analysis, None)
        if not callable(method):
            rows.append({"analysis": analysis, "status": "skipped", "value": None, "message": "Unavailable"})
            continue
        try:
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                value = method(
                    {
                        **analysis_options,
                        "plot": True,
                        "plot CV": False,
                        "new plot": False,
                        "plot all": True,
                        "print": True,
                    }
                )
            rows.append(
                {
                    "analysis": analysis,
                    "status": "ok",
                    "value": _coerce_value(value),
                    "output": stdout.getvalue().strip(),
                    "message": "",
                }
            )
        except Exception as exc:
            rows.append({"analysis": analysis, "status": "error", "value": None, "message": str(exc)})
    plot = None
    after_figures = set(plt.get_fignums())
    created_figures = sorted(after_figures - before_figures)
    figure_number = created_figures[-1] if created_figures else (plt.get_fignums()[-1] if plt.get_fignums() else None)
    if figure_number is not None:
        fig = plt.figure(figure_number)
        try:
            plot = figure_to_data_uri(fig)
        finally:
            plt.close(fig)
    return {"status": "ok", "message": "", "results": rows, "plot": plot}


def _run_object_method_analyses(obj, analyses, supported: dict[str, str], default_analyses, expected_class: str):
    if type(obj).__name__ != expected_class:
        return {
            "status": "skipped",
            "message": f"{expected_class.upper()} analysis requires a {expected_class.upper()} object.",
            "results": [],
            "plot": None,
        }

    rows = []
    before_figures = set(plt.get_fignums())
    returned_figure = None
    for analysis in analyses or default_analyses:
        method_name = supported.get(analysis)
        if method_name is None:
            rows.append({"analysis": analysis, "status": "skipped", "value": None, "message": "Unsupported"})
            continue
        method = getattr(obj, method_name, None)
        if not callable(method):
            rows.append({"analysis": analysis, "status": "skipped", "value": None, "message": "Unavailable"})
            continue
        try:
            value = method({"plot": True, "new plot": False, "print": False}) if analysis.startswith("plot") else method()
            if analysis.startswith("plot"):
                if hasattr(value, "figure"):
                    returned_figure = value.figure
                elif hasattr(value, "savefig"):
                    returned_figure = value
            rows.append({"analysis": analysis, "status": "ok", "value": _coerce_value(value), "message": ""})
        except Exception as exc:
            rows.append({"analysis": analysis, "status": "error", "value": None, "message": str(exc)})

    plot = None
    new_figures = [num for num in plt.get_fignums() if num not in before_figures]
    if returned_figure is not None or new_figures:
        fig = returned_figure or plt.figure(new_figures[-1])
        try:
            plot = figure_to_data_uri(fig)
        finally:
            for num in new_figures:
                plt.close(num)
            if returned_figure is not None:
                plt.close(returned_figure)
    return {"status": "ok", "message": "", "results": rows, "plot": plot}


def _figure_from_value(value):
    if isinstance(value, tuple) and value and hasattr(value[0], "savefig"):
        return value[0]
    axes = getattr(value, "axes", None)
    if axes is not None and hasattr(axes, "figure"):
        return axes.figure
    if hasattr(value, "figure"):
        return value.figure
    if hasattr(value, "savefig"):
        return value
    return None


def _capture_plot_from_value(value, before_figures):
    fig = _figure_from_value(value)
    new_figures = [num for num in plt.get_fignums() if num not in before_figures]
    if fig is None and new_figures:
        fig = plt.figure(new_figures[-1])
    if fig is None:
        return None
    try:
        return figure_to_data_uri(fig)
    finally:
        for num in new_figures:
            plt.close(num)
        if _figure_from_value(value) is not None:
            plt.close(fig)


def _chrono_result_summary(value):
    if not isinstance(value, dict):
        return _coerce_value(value)
    summary = {}
    for key, item in value.items():
        if key in {"time", "charge"}:
            continue
        summary[key] = _coerce_value(item)
    return summary


def _chrono_plot_options(options):
    filtered = dict(options or {})
    for key in ("deduplicate labels", "plot style", "_format", "_dpi"):
        filtered.pop(key, None)
    return filtered


def run_ca_analysis(obj, analyses: list[str] | None = None, options: dict | None = None) -> dict[str, object]:
    if type(obj).__name__ != "ca":
        return {
            "status": "skipped",
            "message": "CA analysis requires a CA object.",
            "results": [],
            "plot": None,
        }

    analyses = analyses or ["plot", "charge", "current_charge_overlay"]
    options = dict(options or {})
    plot_options = _chrono_plot_options(options.get("plot options"))
    target_charge = options.get("target charge", 0.75)
    plot_ca = bool(options.get("plot ca", True))
    baseline_tail_fraction = options.get("baseline tail fraction", 0.05)
    rows = []
    before_figures = set(plt.get_fignums())
    returned_figure = None

    for analysis in analyses:
        try:
            if analysis == "stats":
                value = obj.stats()
            elif analysis == "plot":
                value = obj.plot({**plot_options, "print": False})
            elif analysis == "charge":
                value = obj.charge({**plot_options, "plot": True, "print": False})
            elif analysis == "current_charge_overlay":
                value = obj.plot(
                    {
                        **plot_options,
                        "plot charge": True,
                        "print": False,
                    }
                )
            elif analysis == "baseline_charge":
                value = obj.charge(
                    {
                        **plot_options,
                        "baseline correction": True,
                        "baseline tail fraction": baseline_tail_fraction,
                        "plot": True,
                        "print": False,
                    }
                )
            elif analysis == "time_at_charge":
                value = obj.time_at_charge(
                    {
                        **plot_options,
                        "target charge": target_charge,
                        "plot": True,
                        "plot ca": plot_ca,
                        "print": False,
                    }
                )
            else:
                rows.append({"analysis": analysis, "status": "skipped", "value": None, "message": "Unsupported"})
                continue

            returned_figure = _figure_from_value(value) or returned_figure
            rows.append(
                {
                    "analysis": analysis,
                    "status": "ok",
                    "value": _chrono_result_summary(value),
                    "message": "",
                }
            )
        except Exception as exc:
            rows.append({"analysis": analysis, "status": "error", "value": None, "message": str(exc)})

    plot = None
    new_figures = [num for num in plt.get_fignums() if num not in before_figures]
    if returned_figure is not None or new_figures:
        fig = returned_figure or plt.figure(new_figures[-1])
        try:
            plot = figure_to_data_uri(fig)
        finally:
            for num in new_figures:
                plt.close(num)
            if returned_figure is not None:
                plt.close(returned_figure)
    return {"status": "ok", "message": "", "results": rows, "plot": plot}


def run_cp_analysis(obj, analyses: list[str] | None = None, options: dict | None = None) -> dict[str, object]:
    if type(obj).__name__ != "cp":
        return {
            "status": "skipped",
            "message": "CP analysis requires a CP object.",
            "results": [],
            "plot": None,
            "plots": [],
        }

    analyses = analyses or ["stats", "cycle_info", "plot"]
    options = dict(options or {})
    plot_options = _chrono_plot_options(options.get("plot options"))
    percent_capacity = bool(options.get("percent capacity", False))
    cycles = options.get("cycles", (1, 100, 10))
    supported = {
        "stats": "stats",
        "cycle_info": "cycle_info",
        "plot": "plot",
        "cycling_plot": "cycling_plot",
        "plot_cycles": "plot_cycles",
    }
    plot_labels = {
        "plot": "Potential Plot",
        "cycling_plot": "Cycling Performance",
        "plot_cycles": "Cycle Plot",
    }
    rows = []
    plots = []
    for analysis in analyses:
        method_name = supported.get(analysis)
        if method_name is None:
            rows.append({"analysis": analysis, "status": "skipped", "value": None, "message": "Unsupported"})
            continue
        method = getattr(obj, method_name, None)
        if not callable(method):
            rows.append({"analysis": analysis, "status": "skipped", "value": None, "message": "Unavailable"})
            continue
        before_figures = set(plt.get_fignums())
        try:
            if analysis == "cycle_info":
                value = method({"percent capacity": percent_capacity})
            elif analysis == "plot":
                value = method({**plot_options, "print": False})
            elif analysis == "cycling_plot":
                value = method(
                    {
                        **plot_options,
                        "percent capacity": percent_capacity,
                        "capacity mode": options.get("capacity mode", "both"),
                        "efficiency mode": options.get("efficiency mode", "both"),
                    }
                )
            elif analysis == "plot_cycles":
                value = method(
                    {
                        **plot_options,
                        "cycles": cycles,
                        "segment": options.get("segment", "both"),
                        "x axis": options.get("x axis", "capacity"),
                    }
                )
            else:
                value = method()
            if analysis in plot_labels:
                plot = _capture_plot_from_value(value, before_figures)
                if plot:
                    plots.append({"label": plot_labels.get(analysis, analysis), "src": plot})
            rows.append({"analysis": analysis, "status": "ok", "value": _coerce_value(value), "message": ""})
        except Exception as exc:
            for num in [num for num in plt.get_fignums() if num not in before_figures]:
                plt.close(num)
            rows.append({"analysis": analysis, "status": "error", "value": None, "message": str(exc)})

    return {
        "status": "ok",
        "message": "",
        "results": rows,
        "plot": plots[0]["src"] if plots else None,
        "plots": plots,
    }


MULTI_CV_ANALYSES = {
    "fit_peak_potential": "fit_peak_potential",
    "fit_peak_current": "fit_peak_current",
    "sevcik_analysis": "sevcik_analysis",
    "trumpet_analysis": "trumpet_analysis",
    "fowa": "fowa",
    "tafel_analysis": "tafel_analysis",
}


def _cv_object_by_table_index(objects, index):
    if index in (None, ""):
        return None
    try:
        obj = list(objects or [])[int(index)]
    except (TypeError, ValueError, IndexError):
        return None
    return obj if type(obj).__name__ == "cv" else None


def run_multi_cv_analysis(
    objects,
    analysis: str,
    options: dict[str, object] | None = None,
    all_objects=None,
) -> dict[str, object]:
    if analysis not in MULTI_CV_ANALYSES:
        return {
            "analysis": analysis,
            "status": "skipped",
            "message": f"Unsupported multiple CV analysis: {analysis}",
            "value": None,
        }

    cv_objects = [obj for obj in objects or [] if type(obj).__name__ == "cv"]
    if len(cv_objects) == 0:
        return {
            "analysis": analysis,
            "status": "skipped",
            "message": "Multiple CV analysis requires at least one CV object.",
            "value": None,
        }

    run_options = dict(options or {})
    reference_index = run_options.pop("non-catalytic cv index", None)
    tafel_index = run_options.pop("cv index", None)
    if analysis == "fowa" and reference_index is not None:
        reference_cv = _cv_object_by_table_index(all_objects or objects, reference_index)
        if reference_cv is None:
            return {
                "analysis": analysis,
                "status": "error",
                "message": f"Could not find CV at table index {reference_index} for non-catalytic reference.",
                "value": None,
                "options": run_options,
            }
        run_options["non-catalytic cv"] = reference_cv

    if analysis != "tafel_analysis":
        run_options.update(
            {
                "plot": True,
                "plot all": True,
                "new plot": False,
                "print": False,
            }
        )
    before_figures = set(plt.get_fignums())
    try:
        if analysis == "tafel_analysis":
            for key in ("plot", "plot all", "plot fit", "new plot", "print"):
                run_options.pop(key, None)
            selected_cv = _cv_object_by_table_index(all_objects or objects, tafel_index) if tafel_index is not None else cv_objects[0]
            if selected_cv is None:
                return {
                    "analysis": analysis,
                    "status": "error",
                    "message": f"Could not find CV at table index {tafel_index} for Tafel analysis.",
                    "value": None,
                    "options": run_options,
                }
            required = ["TOF max", "thermodynamic potential", "redox potential"]
            missing = [key for key in required if run_options.get(key) in (None, "")]
            if missing:
                return {
                    "analysis": analysis,
                    "status": "error",
                    "message": f"Tafel analysis requires {', '.join(missing)}.",
                    "value": None,
                    "options": run_options,
                }
            value = e.tafel_analysis(
                selected_cv,
                run_options.pop("TOF max"),
                run_options.pop("thermodynamic potential"),
                run_options.pop("redox potential"),
                run_options,
            )
        else:
            value = getattr(e, MULTI_CV_ANALYSES[analysis])(cv_objects, run_options)
    except Exception as exc:
        return {
            "analysis": analysis,
            "status": "error",
            "message": str(exc),
            "value": None,
            "options": run_options,
        }
    plot = None
    plots = []
    new_figures = [num for num in plt.get_fignums() if num not in before_figures]
    if new_figures:
        for index, num in enumerate(new_figures):
            fig = plt.figure(num)
            label = "Output" if index == len(new_figures) - 1 else f"Diagnostic {index + 1}"
            try:
                plots.append({"label": label, "src": figure_to_data_uri(fig)})
            finally:
                plt.close(fig)
        plot = plots[-1]["src"] if plots else None

    return {
        "analysis": analysis,
        "status": "ok",
        "message": "",
        "value": _coerce_value(value),
        "plot": plot,
        "plots": plots,
        "options": run_options,
    }


def _coerce_value(value):
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    table = getattr(value, "table", None)
    if isinstance(table, pd.DataFrame):
        return table.to_dict("records")
    if isinstance(value, pd.DataFrame):
        return value.to_dict("records")
    if isinstance(value, pd.Series):
        return value.to_dict()
    if isinstance(value, dict):
        return {key: _coerce_value(item) for key, item in value.items()}
    return repr(value)
