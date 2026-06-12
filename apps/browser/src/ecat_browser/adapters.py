"""Adapters between Dash state and the public eCAT API."""

from __future__ import annotations

import base64
import contextlib
from dataclasses import dataclass, field
import io
from pathlib import Path
import re
import uuid

import pandas as pd
import matplotlib.pyplot as plt

import ecat as e

from .figures import figure_to_data_uri
from .workflow import BrowserWorkflow


@dataclass
class LoadResult:
    objects: list[object] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    workflow: BrowserWorkflow = field(default_factory=BrowserWorkflow)
    status: str = ""


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


def _status_from_stdout(output: str) -> str:
    for line in output.splitlines():
        stripped = line.strip()
        if ".txt file" in stripped and "found" in stripped:
            return stripped
        if stripped.startswith("No files were found") or stripped.startswith("Found "):
            return stripped
    return ""


def load_local_path(path, recursive: bool = False, import_options: dict | None = None) -> LoadResult:
    path = Path(path).expanduser()
    import_options = _import_options(import_options)
    workflow = BrowserWorkflow(
        source_kind="local_path",
        source_path=str(path),
        recursive=recursive,
        import_options=import_options,
    )
    warnings: list[str] = []

    if path.is_dir():
        stdout = io.StringIO()
        try:
            with contextlib.redirect_stdout(stdout):
                objects = e.get_data(
                    {
                        "folder path": str(path),
                        "recursive search": recursive,
                        "sort keys": ["timestamp"],
                        **import_options,
                    }
                )
        except Exception as exc:
            return LoadResult([], [f"Warning: could not load folder {path}: {exc}"], workflow)
        return LoadResult(list(objects or []), warnings, workflow, status=_status_from_stdout(stdout.getvalue()))

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
    session_root = Path(session_root or Path.cwd() / ".ecat-browser-sessions")
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

    workflow = BrowserWorkflow(
        source_kind="upload",
        source_path=str(session_dir),
        recursive=False,
        import_options=import_options,
    )
    count = len(objects)
    noun = "file" if count == 1 else "files"
    return LoadResult(objects, warnings, workflow, status=f"{count} uploaded {noun} loaded.")


def reload_workflow(workflow: BrowserWorkflow | dict) -> LoadResult:
    workflow = BrowserWorkflow.from_dict(workflow) if isinstance(workflow, dict) else workflow
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


def default_included_row_ids(objects, workflow: BrowserWorkflow | None = None) -> list[str]:
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


def _is_default_fe_phoh_source(workflow: BrowserWorkflow | None) -> bool:
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


def filter_and_group(objects, workflow: BrowserWorkflow):
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


def objects_for_analysis(objects, workflow: BrowserWorkflow):
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
            "message": "Single-object analysis is CV-only in the browser app v1.",
            "results": [],
        }

    analyses = analyses or ["peak_potential", "peak_current"]
    options = dict(options or {})
    analysis_options = {
        key: value
        for key, value in {
            "guess potential": options.get("guess potential"),
            "tangent potential": options.get("tangent potential"),
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
            value = method(
                {
                    **analysis_options,
                    "plot": True,
                    "plot CV": False,
                    "new plot": False,
                    "plot all": True,
                    "print": False,
                }
            )
            rows.append({"analysis": analysis, "status": "ok", "value": _coerce_value(value), "message": ""})
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
    for key in ("deduplicate labels", "_format", "_dpi"):
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
    new_figures = [num for num in plt.get_fignums() if num not in before_figures]
    if new_figures:
        fig = plt.figure(new_figures[-1])
        try:
            plot = figure_to_data_uri(fig)
        finally:
            for num in new_figures:
                plt.close(num)

    return {
        "analysis": analysis,
        "status": "ok",
        "message": "",
        "value": _coerce_value(value),
        "plot": plot,
        "options": run_options,
    }


def _coerce_value(value):
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, pd.DataFrame):
        return value.to_dict("records")
    if isinstance(value, pd.Series):
        return value.to_dict()
    if isinstance(value, dict):
        return {key: _coerce_value(item) for key, item in value.items()}
    return repr(value)
