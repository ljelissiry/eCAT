"""Dash callback registration for the eCAT app."""

import math
from pathlib import Path
import time

import ecat as e
from dash import html

from .adapters import (
    default_included_row_ids,
    filter_and_group,
    load_local_path,
    load_uploaded_files,
    objects_for_analysis,
    reload_workflow,
    run_ca_analysis,
    run_cp_analysis,
    run_browser_fit_cv,
    run_browser_simulate_cv,
    render_browser_cv_data_program_plot,
    run_multi_cv_analysis,
    run_single_cv_analysis,
    summarize_objects,
    validate_simulation_mechanism,
)
from .codegen import generate_python
from .defaults import default_workflow, example_folder_path
from .execution import run_user_code
from .figures import render_animation, render_model_program_plot, render_multiplot, render_object_plot
from .notebook import generate_notebook
from .references import build_reference_options, reference_field_visibility
from .state import registry as default_registry
from .table import ag_grid_column_defs, build_browser_table, selection_toggle_state, toggle_filtered_selection_state
from .table import (
    available_column_options,
    displayed_selected_row_ids,
    import_conditions_summary,
    reset_column_selection,
    selected_grid_rows_for_ids,
    selected_column_values,
    selected_row_ids_from_grid_rows,
)
from .workflow import AppWorkflow


DEFAULT_PLOT_OPTIONS = {
    "legend": True,
    "legend mode": "colorbar",
    "color mode": "auto",
    "title": "auto",
    "plot style": "notebook",
    "plot convention": "IUPAC",
    "_format": "svg",
    "_dpi": 300,
}


def empty_state():
    workflow = AppWorkflow()
    return {
        "objects": [],
        "summary": [],
        "warnings": [],
        "dataset_id": None,
        "included_row_ids": [],
        "table": {"data": [], "columns": []},
        "column_options": [],
        "workflow": workflow.to_dict(),
        "conditions": [],
        "plot": None,
        "status": "",
        "code": generate_python(workflow),
    }


def _state_from_load_result(result, registry=default_registry) -> dict[str, object]:
    dataset_id = registry.put(result.objects, result.warnings)
    snapshot = registry.snapshot(dataset_id)
    workflow = result.workflow
    included_row_ids = workflow.included_row_ids or default_included_row_ids(result.objects, workflow)
    workflow.included_row_ids = included_row_ids
    plot = render_default_multiplot(result.objects, workflow, plot_options=display_plot_options(DEFAULT_PLOT_OPTIONS))
    save_plot = render_default_multiplot(result.objects, workflow, plot_options=DEFAULT_PLOT_OPTIONS)
    table = build_browser_table(result.objects)
    column_options = available_column_options(result.objects)
    conditions = import_conditions_summary(result.objects)
    return {
        **snapshot,
        "included_row_ids": included_row_ids,
        "table": table,
        "column_options": column_options,
        "conditions": conditions,
        "plot": plot,
        "save_plot": save_plot,
        "status": result.status,
        "workflow": workflow.to_dict(),
        "code": generate_python(workflow),
    }


def handle_local_path_load(path, recursive=False, registry=default_registry) -> dict[str, object]:
    if not path:
        return empty_state()
    return _state_from_load_result(load_local_path(path, recursive=recursive), registry)


def handle_upload_load(filenames, contents, registry=default_registry) -> dict[str, object]:
    uploads = [
        {"filename": filename, "contents": content}
        for filename, content in zip(filenames or [], contents or [])
    ]
    return _state_from_load_result(load_uploaded_files(uploads), registry)


def handle_default_load(repo_root=None, registry=default_registry) -> dict[str, object]:
    workflow = default_workflow(repo_root)
    return _state_from_load_result(reload_workflow(workflow), registry)


def handle_example_folder_load(example_key, repo_root=None, registry=default_registry) -> dict[str, object]:
    path = example_folder_path(example_key, repo_root)
    if path is None:
        return empty_state()
    return handle_local_path_load(str(path), recursive=True, registry=registry)


def handle_apply_reference(workflow_data, reference_settings, registry=default_registry) -> dict[str, object]:
    workflow = AppWorkflow.from_dict(workflow_data)
    workflow.reference_settings = dict(reference_settings or {})
    workflow.import_options = build_reference_options(workflow.reference_settings)
    return _state_from_load_result(reload_workflow(workflow), registry)


def header_session_label(state) -> str:
    """Return a compact top-bar label for the currently loaded data/session."""
    if not state:
        return "No data loaded"
    rows = list((state or {}).get("summary") or [])
    count = len(rows)
    if count == 0:
        return "No data loaded"
    workflow = AppWorkflow.from_dict((state or {}).get("workflow"))
    if workflow.source_kind == "upload":
        source_label = "Uploaded files"
    else:
        source_label = Path(str(workflow.source_path or "")).name or "Loaded data"
    noun = "item" if count == 1 else "items"
    return f"{source_label} · {count} {noun}"


def reference_file_options(rows) -> list[dict[str, object]]:
    options = []
    for row in rows or []:
        index = row.get("index")
        label = row.get("filename") or row.get("name") or str(index)
        options.append({"label": f"{index}: {label}", "value": index})
    return options


def reference_file_path_from_index(objects_state, reference_file_index, registry=default_registry):
    if reference_file_index in (None, ""):
        return reference_file_index
    dataset_id = (objects_state or {}).get("dataset_id")
    objects = registry.get(dataset_id)
    try:
        index = int(reference_file_index)
        return getattr(objects[index], "filepath", None) or reference_file_index
    except (TypeError, ValueError, IndexError):
        return reference_file_index


def handle_select_folder(config_data=None) -> str | None:
    config_data = config_data or {}
    if config_data.get("enable_folder_picker") is False:
        return None
    try:
        import tkinter as tk
        from tkinter import filedialog
    except Exception:
        return None

    try:
        root = tk.Tk()
        root.withdraw()
        selected = filedialog.askdirectory()
        root.destroy()
    except Exception:
        return None
    return selected or None


def handle_include_rows(workflow_data, included_row_ids) -> dict[str, object]:
    workflow = AppWorkflow.from_dict(workflow_data)
    workflow.included_row_ids = list(included_row_ids or [])
    return workflow.to_dict()


def update_workflow_plot_options(workflow_data, plot_options) -> dict[str, object]:
    workflow = AppWorkflow.from_dict(workflow_data)
    workflow.plot_options = dict(plot_options or {})
    return workflow.to_dict()


def _analysis_action_key(action: dict[str, object]) -> str:
    kind = str(action.get("kind") or "")
    if kind == "cv_multi":
        return f"{kind}:{action.get('analysis') or ''}"
    return kind


def _upsert_analysis_action(workflow: AppWorkflow, action: dict[str, object]) -> None:
    key = _analysis_action_key(action)
    actions = [dict(existing) for existing in workflow.analysis_actions]
    for index, existing in enumerate(actions):
        if _analysis_action_key(existing) == key:
            actions[index] = action
            break
    else:
        actions.append(action)
    workflow.analysis_actions = actions


def update_workflow_single_analysis(workflow_data, selected_index, analyses, options=None) -> dict[str, object]:
    workflow = AppWorkflow.from_dict(workflow_data)
    workflow.selected_index = None if selected_index is None else int(selected_index)
    workflow.analyses = list(analyses or [])
    _upsert_analysis_action(
        workflow,
        {
            "kind": "cv_single",
            "selected_index": workflow.selected_index,
            "analyses": list(analyses or []),
            "options": dict(options or {}),
        },
    )
    return workflow.to_dict()


def update_workflow_multi_analysis(workflow_data, analysis, options) -> dict[str, object]:
    workflow = AppWorkflow.from_dict(workflow_data)
    workflow.analyses = list(dict.fromkeys([*workflow.analyses, analysis]))
    workflow.plot_options = {**workflow.plot_options, f"{analysis} options": dict(options or {})}
    _upsert_analysis_action(
        workflow,
        {
            "kind": "cv_multi",
            "analysis": analysis,
            "analyses": [analysis],
            "options": dict(options or {}),
        },
    )
    return workflow.to_dict()


def update_workflow_chrono_analysis(workflow_data, kind, selected_index, analyses, options) -> dict[str, object]:
    workflow = AppWorkflow.from_dict(workflow_data)
    _upsert_analysis_action(
        workflow,
        {
            "kind": kind,
            "selected_index": None if selected_index is None else int(selected_index),
            "analyses": list(analyses or []),
            "options": dict(options or {}),
        },
    )
    return workflow.to_dict()


def clear_workflow_analysis_actions(workflow_data) -> dict[str, object]:
    workflow = AppWorkflow.from_dict(workflow_data)
    workflow.analysis_actions = []
    workflow.analyses = []
    return workflow.to_dict()


def first_analysis_index(dataset_id, class_name, registry=default_registry):
    for index, obj in enumerate(registry.get(dataset_id)):
        if type(obj).__name__ == class_name:
            return index
    return None


def analysis_index_defaults(objects_state, registry=default_registry):
    dataset_id = (objects_state or {}).get("dataset_id")
    values = {}
    disabled = {}
    messages = {}
    labels = {"cv": "CV", "ca": "CA", "cp": "CP"}
    for class_name, label in labels.items():
        index = first_analysis_index(dataset_id, class_name, registry=registry)
        values[class_name] = "" if index is None else str(index)
        disabled[class_name] = index is None
        messages[class_name] = f"No {label} objects loaded." if index is None else ""
    return values, disabled, messages


def analysis_card_open_state(objects_state):
    classes = {
        str(row.get("class") or "").lower()
        for row in (objects_state or {}).get("summary", [])
    }
    return "cv" in classes, "ca" in classes, "cp" in classes


def model_main_cards_open_state(active_tab):
    is_model = str(active_tab or "").strip().lower() == "model"
    return is_model, is_model


def _segment_count_from_object(obj):
    raw = getattr(obj, "segments", None)
    if raw in (None, ""):
        return None
    if isinstance(raw, int):
        return raw if raw > 0 else None
    try:
        return int(raw)
    except (TypeError, ValueError):
        try:
            return len(raw)
        except TypeError:
            return None


def single_cv_segment_control_state(objects_state, selected_index, registry=default_registry):
    dataset_id = (objects_state or {}).get("dataset_id")
    objects = registry.get(dataset_id)
    base = {
        "slider_min": 1,
        "slider_max": 1,
        "slider_value": 1,
        "slider_marks": {1: "1"},
        "slider_disabled": True,
        "slider_style": {"display": "none"},
        "text_style": {"display": "none"},
        "text_value": "",
        "status": "Segment count unknown.",
    }
    try:
        obj = objects[int(selected_index)]
    except (TypeError, ValueError, IndexError):
        return base
    if type(obj).__name__ != "cv":
        return base
    count = _segment_count_from_object(obj)
    if count == 1:
        base.update(
            {
                "slider_max": 1,
                "slider_value": 1,
                "slider_marks": {1: "1"},
                "slider_disabled": True,
                "slider_style": {},
                "status": "One segment detected.",
            }
        )
    elif count is not None and 2 <= count <= 12:
        base.update(
            {
                "slider_max": count,
                "slider_value": 1,
                "slider_marks": {index: str(index) for index in range(1, count + 1)},
                "slider_disabled": False,
                "slider_style": {},
                "status": f"{count} segments detected.",
            }
        )
    else:
        base.update(
            {
                "text_style": {},
                "status": "Enter segment manually.",
            }
        )
    return base


def single_cv_options_from_controls(segment_slider=None, segment_text=None):
    value = segment_slider if segment_slider not in (None, "") else segment_text
    if value in (None, ""):
        return {}
    return {"segment": int(float(value))}


def single_cv_dimensionless_normalization_from_controls(
    enabled_values=None,
    mode="homogeneous",
    e0=None,
    n=None,
    temperature=None,
    d=None,
    c=None,
    area_mode="area_cm2",
    area=None,
):
    if "dimensionless" not in (enabled_values or []):
        return {}
    options = {"mode": mode or "homogeneous", "print": False}
    for key, value in {
        "E0": e0,
        "n": n,
        "temperature": temperature,
        "D": d,
        "C": c,
    }.items():
        if value not in (None, ""):
            options[key] = float(value)
    if c not in (None, ""):
        options["C unit"] = "M"
    resolved_area = _dimensionless_area_cm2(area_mode, area)
    if resolved_area is not None:
        options["area"] = resolved_area
    return {"dimensionless normalization": options}


def single_cv_dimensionless_visibility(enabled_values=None) -> dict[str, str]:
    return {} if "dimensionless" in (enabled_values or []) else {"display": "none"}


def _dimensionless_area_cm2(area_mode="area_cm2", value=None):
    if value in (None, ""):
        return None
    value = float(value)
    if str(area_mode or "area_cm2") == "radius_mm":
        radius_cm = value / 10
        return math.pi * radius_cm * radius_cm
    return value


def handle_extra_columns(dataset_id, visible_columns, registry=default_registry) -> dict[str, object]:
    return build_browser_table(registry.get(dataset_id), visible_columns=visible_columns or [])


def handle_reset_columns(dataset_id, registry=default_registry) -> dict[str, object]:
    return reset_column_selection(registry.get(dataset_id))


def render_default_multiplot(objects, workflow=None, plot_options=None) -> str | None:
    workflow = workflow or AppWorkflow()
    ordered = objects_for_analysis(objects, workflow)
    cv_objects = [obj for obj in ordered if type(obj).__name__ == "cv"]
    if not cv_objects:
        return None
    try:
        if dict(plot_options or {}).get("_animate"):
            return render_animation(cv_objects, plot_options)
        return render_multiplot(cv_objects, plot_options)
    except Exception:
        return None


def handle_replot(dataset_id, selected_row_ids=None, registry=default_registry, plot_options=None) -> dict[str, object]:
    objects = registry.get_by_row_ids(dataset_id, selected_row_ids)
    plot = render_default_multiplot(objects, AppWorkflow(), plot_options=plot_options)
    return {"plot": plot}


def display_plot_options(plot_options):
    options = dict(plot_options or {})
    if options.get("_animate"):
        options["_format"] = "html"
    else:
        options["_format"] = "png"
        options["_dpi"] = 150
    return options


def analysis_plot_options(plot_options):
    options = dict(plot_options or {})
    options.pop("_format", None)
    options.pop("_dpi", None)
    return options


def multi_cv_preprocessing_from_controls(
    scale_values=None,
    scale_type="reference",
    scale_factor=None,
    scale_reference_index=None,
    scale_reference_mode="single",
    scale_segment=None,
    scale_guess_potential=None,
    normalize_mode="none",
    dimensionless_mode="homogeneous",
    dimensionless_e0=None,
    dimensionless_n=None,
    dimensionless_temperature=None,
    dimensionless_d=None,
    dimensionless_c=None,
    dimensionless_area_mode="area_cm2",
    dimensionless_area=None,
    current_type="reference",
    current_reference_index=None,
    current_segment=None,
    current_guess_potential=None,
    current_ip0=None,
):
    preprocessing = {}
    if "scale" in (scale_values or []):
        scale_options = {"print": False, "plot all": False}
        scale_type = str(scale_type or "reference").strip().lower()
        if scale_type == "manual" and scale_factor not in (None, ""):
            scale_options["scale"] = float(scale_factor)
        if scale_type != "manual" and scale_reference_index not in (None, ""):
            scale_options["reference index"] = int(float(scale_reference_index))
        if scale_type != "manual" and scale_reference_mode:
            scale_options["reference mode"] = scale_reference_mode
        if scale_type != "manual" and scale_segment not in (None, ""):
            scale_options["segment"] = int(float(scale_segment))
        if scale_type != "manual" and scale_guess_potential not in (None, ""):
            scale_options["guess potential"] = float(scale_guess_potential)
        preprocessing["scale current"] = scale_options

    normalize_mode = str(normalize_mode or "none").strip().lower()
    if normalize_mode == "dimensionless":
        normalize_options = {
            "mode": dimensionless_mode or "homogeneous",
            "print": False,
        }
        optional_float_fields = {
            "E0": dimensionless_e0,
            "n": dimensionless_n,
            "temperature": dimensionless_temperature,
            "D": dimensionless_d,
            "C": dimensionless_c,
        }
        for key, value in optional_float_fields.items():
            if value not in (None, ""):
                normalize_options[key] = float(value)
        if dimensionless_c not in (None, ""):
            normalize_options["C unit"] = "M"
        resolved_area = _dimensionless_area_cm2(dimensionless_area_mode, dimensionless_area)
        if resolved_area is not None:
            normalize_options["area"] = resolved_area
        preprocessing["normalize"] = {"mode": "dimensionless", "options": normalize_options}
    elif normalize_mode == "current":
        normalize_options = {"print": False, "plot all": False}
        current_type = str(current_type or "reference").strip().lower()
        if current_type == "manual" and current_ip0 not in (None, ""):
            normalize_options["ip0"] = float(current_ip0)
        if current_type != "manual" and current_reference_index not in (None, ""):
            normalize_options["reference index"] = int(float(current_reference_index))
        if current_type != "manual" and current_segment not in (None, ""):
            normalize_options["segment"] = int(float(current_segment))
        if current_type != "manual" and current_guess_potential not in (None, ""):
            normalize_options["guess potential"] = float(current_guess_potential)
        preprocessing["normalize"] = {"mode": "current", "options": normalize_options}
    return preprocessing


def multi_cv_preprocessing_visibility(
    scale_values=None,
    normalize_mode="none",
    scale_type="reference",
    current_type="reference",
) -> tuple[dict[str, str], dict[str, str], dict[str, str], dict[str, str], dict[str, str], dict[str, str], dict[str, str]]:
    scale_style = {} if "scale" in (scale_values or []) else {"display": "none"}
    scale_type = str(scale_type or "reference").strip().lower()
    scale_reference_style = {} if scale_type != "manual" else {"display": "none"}
    scale_manual_style = {} if scale_type == "manual" else {"display": "none"}
    normalize_mode = str(normalize_mode or "none").strip().lower()
    dimensionless_style = {} if normalize_mode == "dimensionless" else {"display": "none"}
    current_style = {} if normalize_mode == "current" else {"display": "none"}
    current_type = str(current_type or "reference").strip().lower()
    current_reference_style = {} if current_type != "manual" else {"display": "none"}
    current_manual_style = {} if current_type == "manual" else {"display": "none"}
    return (
        scale_style,
        scale_reference_style,
        scale_manual_style,
        dimensionless_style,
        current_style,
        current_reference_style,
        current_manual_style,
    )


def apply_multi_cv_preprocessing(objects, preprocessing):
    cvs = list(objects or [])
    preprocessing = preprocessing or {}
    scale_options = preprocessing.get("scale current")
    if scale_options:
        cvs = e.scale_current(cvs, scale_options)
    normalize = preprocessing.get("normalize") or {}
    normalize_mode = normalize.get("mode")
    normalize_options = normalize.get("options") or {}
    if normalize_mode == "dimensionless":
        cvs = e.normalize(cvs, normalize_options)
    elif normalize_mode == "current":
        cvs = e.normalize_current(cvs, normalize_options)
    return list(cvs)


def toggle_sidebar_class(class_name: str | None) -> str:
    classes = [part for part in str(class_name or "ecat-app").split() if part]
    if "ecat-app" not in classes:
        classes.insert(0, "ecat-app")
    if "ecat-sidebar-collapsed" in classes:
        classes = [part for part in classes if part != "ecat-sidebar-collapsed"]
    else:
        classes.append("ecat-sidebar-collapsed")
    return " ".join(classes)


def expand_sidebar_class(class_name: str | None) -> str:
    classes = [part for part in str(class_name or "ecat-app").split() if part]
    if "ecat-app" not in classes:
        classes.insert(0, "ecat-app")
    return " ".join(part for part in classes if part != "ecat-sidebar-collapsed")


def handle_single_cv(dataset_id, selected_index, analyses, options=None, registry=default_registry) -> dict[str, object]:
    objects = registry.get(dataset_id)
    if not objects:
        return {"message": "No loaded objects.", "plot": None, "results": []}
    if selected_index in (None, ""):
        return {"message": "No CV objects loaded.", "plot": None, "results": []}
    index = int(selected_index or 0)
    if index < 0 or index >= len(objects):
        return {"message": f"CV index {index} is out of range.", "plot": None, "results": []}
    obj = objects[index]
    options = dict(options or {})
    normalization_options = options.pop("dimensionless normalization", None)
    if normalization_options:
        try:
            obj = e.normalize(obj, normalization_options)
        except Exception as exc:
            return {"message": str(exc), "plot": None, "results": []}
    analysis = run_single_cv_analysis(obj, analyses or ["peak_potential", "peak_current"], options)
    plot = analysis.get("plot")
    message = analysis["message"] or f"CV index {index}"
    return {"message": message, "plot": plot, "results": analysis["results"]}


def handle_single_object_analysis(dataset_id, selected_index, analyses, expected_class, runner, registry=default_registry, row_ids=None):
    objects = registry.get(dataset_id)
    if not objects:
        return {"message": "No loaded objects.", "plot": None, "results": []}
    if selected_index in (None, ""):
        return {"message": f"No {expected_class.upper()} objects loaded.", "plot": None, "results": []}
    index = int(selected_index or 0)
    if index < 0 or index >= len(objects):
        return {"message": f"Object index {index} is out of range.", "plot": None, "results": []}
    obj = objects[index]
    if type(obj).__name__ != expected_class:
        return {
            "message": f"Object index {index} is {type(obj).__name__}, not {expected_class}.",
            "plot": None,
            "results": [],
        }
    return runner(obj, analyses)


def handle_multi_cv(dataset_id, workflow_data, registry=default_registry) -> dict[str, object]:
    workflow = AppWorkflow.from_dict(workflow_data)
    objects = registry.get_by_row_ids(dataset_id, workflow.included_row_ids)
    if not objects:
        return {"message": "No loaded objects.", "plot": None, "summary": []}
    grouped = filter_and_group(objects, workflow)
    plot_target = grouped[0] if grouped and isinstance(grouped[0], list) else grouped
    plot = render_multiplot(plot_target) if plot_target else None
    return {"message": "", "plot": plot, "summary": summarize_objects(plot_target)}


def _parse_segments(value):
    if value in (None, ""):
        return None
    if isinstance(value, (list, tuple)):
        raw_parts = value
    else:
        raw_parts = str(value).replace(";", ",").split(",")
    parts = []
    for part in raw_parts:
        text = str(part).strip()
        if not text:
            continue
        parts.append(int(float(text)))
    return parts or None


def _parse_optional_float(value):
    if value in (None, ""):
        return None
    return float(value)


def _parse_optional_int(value):
    if value in (None, ""):
        return None
    return int(float(value))


def _parse_range(start, end):
    if start in (None, "") or end in (None, ""):
        return None
    return [float(start), float(end)]


def multi_cv_options_from_controls(
    analysis=None,
    segment=None,
    segments=None,
    guess_potential=None,
    sevcik_mode="homogeneous",
    x_axis="auto",
    fit_model="linear",
    toggles=None,
    fowa_reference_index=None,
    fowa_redox_mode="manual",
    fowa_redox_potential=None,
    fowa_fit_basis="y",
    fowa_fit_range_start="0.1",
    fowa_fit_range_end="0.5",
    fowa_diagnostic_y_axis="i/ip0",
    fowa_min_fit_points="50",
    fowa_min_r2="0.95",
    tafel_index=None,
    tafel_tof_max=None,
    tafel_thermo_potential=None,
    tafel_redox_potential=None,
    tafel_overpotential_start="0",
    tafel_overpotential_end="1",
    tafel_color="black",
):
    toggles = toggles or []
    options = {} if analysis == "tafel_analysis" else {"plot fit": "plot_fit" in toggles, "plot all": "plot_all" in toggles}
    if analysis not in {"sevcik_analysis", "fowa", "tafel_analysis"}:
        options["fit"] = "fit" in toggles
    parsed_segments = _parse_segments(segments)
    if parsed_segments:
        options["segments"] = parsed_segments
    elif segment not in (None, ""):
        options["segment"] = int(segment)
    if guess_potential not in (None, ""):
        options["guess potential"] = float(guess_potential)
    if analysis == "sevcik_analysis":
        options["scan dependence"] = 1.0 if str(sevcik_mode or "homogeneous") == "heterogeneous" else 0.5
    if analysis not in {"sevcik_analysis", "fowa", "tafel_analysis"} and x_axis and x_axis != "auto":
        options["x axis"] = x_axis
    if analysis not in {"sevcik_analysis", "fowa", "tafel_analysis"} and fit_model:
        options["fit model"] = fit_model
    if analysis == "fowa":
        options["redox mode"] = fowa_redox_mode or "manual"
        options["fit basis"] = fowa_fit_basis or "y"
        options["diagnostic y axis"] = fowa_diagnostic_y_axis or "i/ip0"
        fit_range = _parse_range(fowa_fit_range_start, fowa_fit_range_end)
        if fit_range is not None:
            options["fit range"] = fit_range
        min_fit_points = _parse_optional_int(fowa_min_fit_points)
        if min_fit_points is not None:
            options["min fit points"] = min_fit_points
        min_r2 = _parse_optional_float(fowa_min_r2)
        if min_r2 is not None:
            options["min r2"] = min_r2
        reference_index = _parse_optional_int(fowa_reference_index)
        if reference_index is not None:
            options["non-catalytic cv index"] = reference_index
        redox_potential = _parse_optional_float(fowa_redox_potential)
        if redox_potential is not None:
            options["redox potential"] = redox_potential
    if analysis == "tafel_analysis":
        cv_index = _parse_optional_int(tafel_index)
        if cv_index is not None:
            options["cv index"] = cv_index
        tof_max = _parse_optional_float(tafel_tof_max)
        if tof_max is not None:
            options["TOF max"] = tof_max
        thermo_potential = _parse_optional_float(tafel_thermo_potential)
        if thermo_potential is not None:
            options["thermodynamic potential"] = thermo_potential
        redox_potential = _parse_optional_float(tafel_redox_potential)
        if redox_potential is not None:
            options["redox potential"] = redox_potential
        overpotential_range = _parse_range(tafel_overpotential_start, tafel_overpotential_end)
        if overpotential_range is not None:
            options["overpotential range"] = overpotential_range
        if tafel_color:
            options["color"] = tafel_color
    return options


def multi_analysis_option_state(analysis):
    if not analysis or analysis == "none":
        return {"display": "none"}, "", "", {"display": "none"}, {"display": "none"}, {"display": "none"}, {}, {}, [], []
    labels = {
        "fit_peak_potential": "Fit Peak Potential Options",
        "fit_peak_current": "Fit Peak Current Options",
        "sevcik_analysis": "Sevcik Analysis Options",
        "trumpet_analysis": "Trumpet Analysis Options",
        "fowa": "FOWA Options",
        "tafel_analysis": "Tafel Analysis Options",
    }
    default_segments = "1, 2" if analysis == "trumpet_analysis" else ""
    sevcik_style = {} if analysis == "sevcik_analysis" else {"display": "none"}
    fowa_style = {} if analysis == "fowa" else {"display": "none"}
    tafel_style = {} if analysis == "tafel_analysis" else {"display": "none"}
    fit_style = {"display": "none"} if analysis in {"fowa", "tafel_analysis", "sevcik_analysis"} else {}
    guess_style = {"display": "none"} if analysis == "tafel_analysis" else {}
    if analysis == "tafel_analysis":
        toggle_options = []
        toggle_values = []
    elif analysis in {"fowa", "sevcik_analysis"}:
        toggle_options = [
            {"label": "Plot fit", "value": "plot_fit"},
            {"label": "Plot diagnostics", "value": "plot_all"},
        ]
        toggle_values = ["plot_fit", "plot_all"]
    else:
        toggle_options = [
            {"label": "Fit", "value": "fit"},
            {"label": "Plot fit", "value": "plot_fit"},
            {"label": "Plot diagnostics", "value": "plot_all"},
        ]
        toggle_values = ["fit", "plot_fit", "plot_all"]
    return (
        {},
        labels.get(analysis, "Analysis Options"),
        default_segments,
        sevcik_style,
        fowa_style,
        tafel_style,
        fit_style,
        guess_style,
        toggle_options,
        toggle_values,
    )


def multi_analysis_equation_content(analysis, sevcik_mode="homogeneous"):
    analysis = str(analysis or "none")
    if analysis == "none":
        return ""
    equations = {
        "fit_peak_potential": [
            html.Span(["E", html.Sub("p"), " = m x + b"]),
            html.Span("Fit selected peak potential against the chosen x axis."),
        ],
        "fit_peak_current": [
            html.Span(["i", html.Sub("p"), " = m x + b"]),
            html.Span("Fit selected peak current against the chosen x axis."),
        ],
        "sevcik_analysis": _sevcik_equation_lines(sevcik_mode),
        "trumpet_analysis": [
            html.Span(["E", html.Sub("p"), " = E", html.Sup("0"), " ± (R T / n F) ln(k", html.Sub("s"), " / v)"]),
            html.Span("Use paired oxidative/reductive peak positions across scan rates."),
        ],
        "fowa": [
            html.Span(["x", html.Sub("FOWA"), " = 1 / (1 + exp[(F / RT)(E - E", html.Sub("redox"), ")])"]),
            html.Span(["y", html.Sub("FOWA"), " = i / i", html.Sub("p"), html.Sup("0"), " = m x", html.Sub("FOWA"), " + b"]),
            html.Span(["k", html.Sub("obs"), " = (m · 0.4463 · n", html.Sub("ref"), " / n", html.Sub("cat"), html.Sup("σ"), ")", html.Sup("2"), " · n", html.Sub("ref"), "Fν / RT"]),
        ],
        "tafel_analysis": [
            html.Span(["η = E", html.Sub("redox"), " - E", html.Sub("thermo"), " - overpotential"]),
            html.Span(["TOF = 2 TOF", html.Sub("max"), " / (1 + exp[(F / RT)(E", html.Sub("thermo"), " - E", html.Sub("redox"), " - η)])"]),
            html.Span(["Tafel plot: log", html.Sub("10"), "(TOF) vs η"]),
        ],
    }
    lines = equations.get(analysis, [])
    if not lines:
        return ""
    return html.Div(
        className="ecat-equation-card",
        children=[html.Div(line, className="ecat-equation-line") for line in lines],
    )


def _sevcik_equation_lines(sevcik_mode="homogeneous"):
    heterogeneous = str(sevcik_mode or "homogeneous") == "heterogeneous"
    if heterogeneous:
        return [
            html.Span(["i", html.Sub("p"), " = mν + b"]),
            html.Span(["Heterogeneous mode uses scan-rate dependence ", html.Span(["ν", html.Sup("1")]), "."]),
        ]
    return [
        html.Span(["i", html.Sub("p"), " = 0.4463 n F A C (n F ν D / R T)", html.Sup("1/2")]),
        html.Span(["D from slope of ", html.Span(["i", html.Sub("p")]), " versus ", html.Span(["ν", html.Sup("1/2")]), "."]),
    ]


def handle_multi_cv_analysis(
    dataset_id,
    selected_row_ids,
    displayed_rows,
    analysis,
    options,
    preprocessing=None,
    registry=default_registry,
) -> dict[str, object]:
    row_ids = displayed_selected_row_ids(displayed_rows, selected_row_ids)
    objects = registry.get_by_row_ids(dataset_id, row_ids)
    if preprocessing:
        try:
            cv_objects = [obj for obj in objects if type(obj).__name__ == "cv"]
            objects = apply_multi_cv_preprocessing(cv_objects, preprocessing)
        except Exception as exc:
            return {
                "analysis": analysis,
                "status": "error",
                "message": str(exc),
                "value": None,
                "plot": None,
                "plots": [],
                "options": options or {},
            }
    options = dict(options or {})
    options.pop("preprocessing", None)
    if analysis == "none":
        cv_objects = [obj for obj in objects if type(obj).__name__ == "cv"]
        try:
            plot = render_multiplot(cv_objects) if cv_objects else None
        except Exception as exc:
            return {
                "analysis": "none",
                "status": "error",
                "message": str(exc),
                "value": None,
                "plot": None,
                "plots": [],
                "options": options,
            }
        plots = [{"label": "Pre-processing", "src": plot}] if plot else []
        return {
            "analysis": "none",
            "status": "ok",
            "message": "Pre-processing plot",
            "value": None,
            "plot": plot,
            "plots": plots,
            "options": options,
        }
    return run_multi_cv_analysis(objects, analysis, options, all_objects=registry.get(dataset_id))


def handle_run_code(code, cwd=None) -> dict[str, object]:
    result = run_user_code(code or "", cwd=cwd)
    return {
        "executed": result.executed,
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }


def render_single_cv_results(result):
    from dash import html

    return render_object_analysis_results(result, "Single CV Analysis")


PLOT_ACTION_ICONS = {
    "refresh": "/assets/ecat_icon_plot_refresh.svg",
    "copy": "/assets/ecat_icon_plot_copy.svg",
    "save": "/assets/ecat_icon_plot_save.svg",
}


def _plot_action_button(html, action, label):
    return html.Button(
        html.Img(src=PLOT_ACTION_ICONS[action], alt="", className="ecat-plot-action-icon"),
        title=label,
        className="ecat-plot-action",
        **{"aria-label": label, "data-ecat-plot-action": action},
    )


def _plot_frame(src, include_refresh=False, save_src=None):
    from dash import html

    if not src:
        return ""
    if str(src).startswith("data:text/html"):
        plot_child = html.Iframe(src=src, className="ecat-animation-frame")
    else:
        plot_child = html.Img(src=src, className="ecat-plot")
    actions = []
    if include_refresh:
        actions.append(_plot_action_button(html, "refresh", "Refresh plot"))
    actions.extend(
        [
            _plot_action_button(html, "copy", "Copy plot"),
            _plot_action_button(html, "save", "Save plot"),
        ]
    )
    return html.Div(
        className="ecat-plot-frame",
        **{"data-ecat-save-src": save_src or src},
        children=[
            plot_child,
            html.Div(
                className="ecat-plot-actions",
                children=[
                    *actions,
                    html.Span(
                        "",
                        className="ecat-plot-action-status",
                        **{"aria-live": "polite"},
                    ),
                ],
            ),
        ],
    )


def _format_analysis_scalar(value):
    if value is None:
        return ""
    try:
        import numpy as np

        if isinstance(value, np.generic):
            value = value.item()
    except Exception:
        pass
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _analysis_value_node(value):
    from dash import html

    try:
        import pandas as pd
    except Exception:
        pd = None

    if pd is not None and isinstance(value, pd.DataFrame):
        rows = value.to_dict("records")
        return _analysis_records_table(rows)
    if pd is not None and isinstance(value, pd.Series):
        return _analysis_mapping_table(value.to_dict())
    if isinstance(value, dict):
        return _analysis_mapping_table(value)
    if isinstance(value, (list, tuple)) and value and all(isinstance(item, dict) for item in value):
        return _analysis_records_table(value)
    if isinstance(value, (list, tuple)):
        return _analysis_records_table([{"index": index, "value": item} for index, item in enumerate(value)])
    return html.Span(_format_analysis_scalar(value), className="ecat-analysis-result-value")


def _analysis_value_is_structured(value) -> bool:
    try:
        import pandas as pd
    except Exception:
        pd = None

    if pd is not None and isinstance(value, (pd.DataFrame, pd.Series)):
        return True
    if isinstance(value, dict):
        return bool(value)
    return isinstance(value, (list, tuple)) and bool(value)


def _analysis_row_output_node(row):
    from dash import html

    output = _clean_analysis_output(row.get("output"))
    value = row.get("value")
    if output and _analysis_value_is_structured(value):
        return html.Div(
            [
                html.Pre(output, className="ecat-analysis-result-output"),
                _analysis_value_node(value),
            ],
            className="ecat-analysis-output-stack",
        )
    if output:
        return html.Pre(output, className="ecat-analysis-result-output")
    return _analysis_value_node(value)


def _clean_analysis_output(output):
    if not output:
        return ""
    lines = [
        line
        for line in str(output).splitlines()
        if not line.strip().startswith("<pandas.io.formats.style.Styler object at")
    ]
    return "\n".join(lines).strip()


def _analysis_mapping_table(mapping):
    from dash import html

    rows = [
        html.Tr(
            [
                html.Th(str(key)),
                html.Td(_format_analysis_scalar(value)),
            ]
        )
        for key, value in (mapping or {}).items()
    ]
    return html.Table(html.Tbody(rows), className="ecat-analysis-value-table")


def _analysis_records_table(records):
    from dash import html

    records = list(records or [])
    if not records:
        return html.Span("", className="ecat-analysis-result-value")
    columns = list(dict.fromkeys(key for record in records for key in record.keys()))
    return html.Table(
        [
            html.Thead(html.Tr([html.Th(str(column)) for column in columns])),
            html.Tbody(
                [
                    html.Tr([html.Td(_format_analysis_scalar(record.get(column))) for column in columns])
                    for record in records
                ]
            ),
        ],
        className="ecat-analysis-value-table",
    )


def render_object_analysis_results(result, title):
    from dash import html

    rows = result.get("results", [])

    def status_nodes(row):
        status = str(row.get("status") or "")
        if not status or status.lower() == "ok":
            return []
        return [html.Span(status, className="ecat-analysis-result-status")]

    entries = [
        html.Div(
            className="ecat-analysis-result-entry",
            children=[
                html.Div(row.get("analysis"), className="ecat-analysis-result-name"),
                html.Div(
                    [
                        *status_nodes(row),
                        _analysis_row_output_node(row),
                        html.Span(str(row.get("message") or ""), className="ecat-analysis-result-message"),
                    ],
                    className="ecat-analysis-result-detail",
                ),
            ],
        )
        for row in rows
    ]
    plots = [plot_item for plot_item in (result.get("plots") or []) if plot_item.get("src")]
    if plots:
        output_plot = next((plot_item for plot_item in plots if plot_item.get("label") == "Output"), plots[-1])
        diagnostic_plots = [plot_item for plot_item in plots if plot_item is not output_plot]

        def plot_block(plot_item):
            return html.Div(
                [
                    html.Div(plot_item.get("label", "Plot"), className="ecat-analysis-plot-label"),
                    _plot_frame(plot_item["src"]),
                ],
                className="ecat-analysis-plot-block",
            )

        plot = html.Div(
            [
                html.Div([plot_block(plot_item) for plot_item in diagnostic_plots], className="ecat-analysis-diagnostics"),
                html.Div(plot_block(output_plot), className="ecat-analysis-output-plot"),
            ],
            className="ecat-analysis-plot-grid",
        )
    else:
        plot = _plot_frame(result.get("plot"))
    return html.Details(
        className="ecat-card ecat-analysis-run",
        open=True,
        children=[
            html.Summary(title, className="ecat-card-summary"),
            html.Div(result.get("message", ""), className="ecat-status"),
            plot,
            html.Div(entries, className="ecat-analysis-result-list"),
        ],
    )


def upsert_analysis_result_store(store, key, result, title):
    updated = dict(store or {})
    updated[str(key)] = {
        "title": title,
        "result": dict(result or {}),
    }
    return updated


def render_analysis_results_store(store):
    from dash import html

    if not store:
        return ""
    technique_order = ["cv_single", "cv_multi", "cv", "ca", "cp"]
    ordered_keys = [key for key in technique_order if key in store]
    ordered_keys.extend(key for key in store if key not in ordered_keys)
    return html.Div(
        [
            render_object_analysis_results(
                store[key].get("result") or {},
                store[key].get("title") or str(key).upper(),
            )
            for key in ordered_keys
        ],
        className="ecat-analysis-result-stack",
    )


def render_multi_cv_results(result):
    from dash import html

    title = str(result.get("analysis") or "multiple_cv").replace("_", " ").title()
    option_lines = [
        html.Div(f"{key}: {value}", className="ecat-analysis-result-detail")
        for key, value in (result.get("options") or {}).items()
        if key not in {"plot", "new plot", "print"}
    ]
    value = result.get("value")
    plot = _plot_frame(result.get("plot"))
    return html.Details(
        className="ecat-card ecat-analysis-run",
        open=True,
        children=[
            html.Summary(title, className="ecat-card-summary"),
            html.Div(result.get("status", ""), className="ecat-status"),
            html.Div(result.get("message", ""), className="ecat-status"),
            plot,
            html.Div(option_lines, className="ecat-analysis-result-list"),
            html.Pre(str(value), className="ecat-analysis-result-output") if value is not None else "",
        ],
    )


def plot_options_from_controls(
    legend_values,
    title_mode="auto",
    convention="IUPAC",
    custom_title=None,
    display_values=None,
    gradient_values=None,
    colorbar_values=None,
    label_values=None,
    trim_values=None,
    trim_min=None,
    trim_max=None,
    trim_mode="expand",
    offset=None,
    scale_bar_height=None,
    scale_bar_location="upper left",
    offset_axis_values=None,
    save_format="svg",
    dpi=300,
    plot_style="notebook",
    axis_label_mode="auto",
    x_axis_label=None,
    y_axis_label=None,
    output_mode="static",
    animation_fps=20,
    animation_stride=1,
    animation_trace_mode="draw",
    animation_schedule="staggered",
    animation_stagger_time=0.5,
    animation_timing_mode="duration",
    animation_timing_value=2,
    animation_advanced=None,
    animation_end_hold=2,
    animation_arrow_potential=None,
    animation_arrow_segment=None,
    animation_scale_bar_values=None,
    animation_scale_bar_length=None,
    animation_scale_bar_location="upper left",
):
    title_mode = str(title_mode or "auto").lower()
    if title_mode == "manual":
        title = custom_title if custom_title else True
    elif title_mode == "none":
        title = False
    else:
        title = "auto"
    display_values = display_values or []
    gradient_values = gradient_values or []
    colorbar_values = colorbar_values or []
    label_values = label_values or []
    allow_gradients = "gradients" in gradient_values
    allow_colorbars = allow_gradients and "colorbar" in colorbar_values
    options = {
        "legend": "legend" in (legend_values or []),
        "legend mode": "colorbar" if allow_colorbars else "discrete",
        "title": title,
        "plot style": plot_style or "notebook",
        "plot convention": convention or "IUPAC",
        "grid": "grid" in display_values,
        "deduplicate labels": "deduplicate" in label_values,
        "_format": save_format or "svg",
        "_dpi": int(dpi or 300),
    }
    options["color mode"] = "auto" if allow_gradients else "discrete"
    axis_label_mode = str(axis_label_mode or "auto").lower()
    if axis_label_mode == "manual":
        if x_axis_label not in (None, ""):
            options["x label"] = x_axis_label
        if y_axis_label not in (None, ""):
            options["y label"] = y_axis_label
    elif axis_label_mode == "none":
        options["x label"] = False
        options["y label"] = False
    if offset not in (None, ""):
        offset_value = float(offset)
        options["offset"] = offset_value
        scale_bar_length = abs(offset_value) if scale_bar_height in (None, "") else float(scale_bar_height)
        options["scale bar"] = {
            "length": scale_bar_length,
            "loc": scale_bar_location or "upper left",
            "remove y ticks": "hide_y_numbers" in (offset_axis_values or []),
        }
    if "trim" in (trim_values or []):
        lower = None if trim_min in (None, "") else float(trim_min)
        upper = None if trim_max in (None, "") else float(trim_max)
        options["potential window"] = [lower, upper]
        options["trim mode"] = trim_mode or "expand"
    if animation_arrow_potential not in (None, ""):
        arrow = {"potential": float(animation_arrow_potential)}
        if animation_arrow_segment not in (None, ""):
            arrow["segment"] = int(animation_arrow_segment)
        options["directional arrows"] = arrow
    if "scale_bar" in (animation_scale_bar_values or []):
        scale_bar = {
            "loc": animation_scale_bar_location or "upper left",
        }
        if animation_scale_bar_length not in (None, ""):
            scale_bar["length"] = float(animation_scale_bar_length)
        options["scale bar"] = scale_bar
    if _plot_output_mode_is_animated(output_mode):
        advanced = animation_advanced or []
        options["_animate"] = True
        options["fps"] = int(animation_fps or 20)
        options["stride"] = int(animation_stride or 1)
        options["trace mode"] = animation_trace_mode or "draw"
        options["schedule"] = animation_schedule or "staggered"
        options["stagger time"] = float(animation_stagger_time or 0)
        options["end hold"] = float(animation_end_hold or 0)
        options["include quiet time"] = "include_quiet_time" in advanced
        options["loop"] = "loop" in advanced
        timing_value = float(animation_timing_value or 1)
        if str(animation_timing_mode or "duration") == "rate":
            options["timing mode"] = "physical"
            options["speedup"] = timing_value
        else:
            options["timing mode"] = "normalized"
            options["normalized duration"] = timing_value
    return options


def _plot_output_mode_is_animated(output_mode) -> bool:
    return str(output_mode or "static").lower() in {"animate", "animated"}


def plot_control_visibility(legend_values, gradient_values=None, title_mode="auto") -> dict[str, dict[str, str]]:
    legend_visible = "legend" in (legend_values or [])
    gradients_visible = legend_visible
    colorbar_visible = legend_visible and "gradients" in (gradient_values or [])
    return {
        "legend_options": {} if legend_visible else {"display": "none"},
        "colorbar": {} if colorbar_visible else {"display": "none"},
        "custom_title": {} if str(title_mode or "auto").lower() == "manual" else {"display": "none"},
    }


def axis_label_controls_visibility(axis_label_mode="auto") -> dict[str, str]:
    return {} if str(axis_label_mode or "auto").lower() == "manual" else {"display": "none"}


def scroll_signal(target: str) -> str:
    return f"{target}:{time.monotonic_ns()}"


def trim_bounds_visibility(trim_values) -> dict[str, str]:
    return {} if "trim" in (trim_values or []) else {"display": "none"}


def offset_controls_visibility(offset) -> dict[str, str]:
    try:
        offset_value = float(offset)
    except (TypeError, ValueError):
        return {"display": "none"}
    return {} if offset_value != 0 else {"display": "none"}


def animation_controls_visibility(output_mode, schedule="staggered", scale_bar_values=None):
    animate = _plot_output_mode_is_animated(output_mode)
    stagger = animate and str(schedule or "") == "staggered"
    scale_bar = "scale_bar" in (scale_bar_values or [])
    return (
        {} if animate else {"display": "none"},
        {} if stagger else {"display": "none"},
        {} if scale_bar else {"display": "none"},
    )


def directional_arrow_options_visibility(clicks=None):
    return {} if clicks and int(clicks) % 2 == 1 else {"display": "none"}


def plot_format_options(output_mode):
    if _plot_output_mode_is_animated(output_mode):
        return (
            [
                {"label": "HTML", "value": "html"},
                {"label": "GIF", "value": "gif"},
                {"label": "MP4", "value": "mp4"},
            ],
            "html",
        )
    return (
        [
            {"label": "PNG", "value": "png"},
            {"label": "SVG", "value": "svg"},
            {"label": "PDF", "value": "pdf"},
        ],
        "svg",
    )


def model_mechanism_visibility(source="preset", preset=None):
    custom = str(source or "preset").strip().lower() == "custom" or str(preset or "").strip().lower() == "custom"
    return {}, {} if custom else {"display": "none"}


def model_mechanism_options_from_controls(source="preset", preset="E", custom_text=None):
    if str(preset or "").strip().lower() == "custom":
        source = "custom"
    validation = validate_simulation_mechanism(source, preset, custom_text)
    return {
        "mechanism_source": str(source or "preset"),
        "mechanism_preset": preset or "E",
        "mechanism_custom": custom_text or "",
        "mechanism": validation.get("mechanism") or "",
        "mechanism_valid": bool(validation.get("ok")),
        "compiled_mechanism": validation.get("compiled") or "",
        "mechanism_details": validation.get("mechanism_details") or [],
        "formatted_equations": validation.get("formatted_equations") or [],
        "simulation_ready": False,
    }


def model_formatted_equations_content(equations):
    equations = [str(equation) for equation in (equations or []) if str(equation or "").strip()]
    if not equations:
        return ""
    return html.Div(
        className="ecat-equation-card",
        children=[html.Div(equation, className="ecat-equation-line") for equation in equations],
    )


def _optional_float(value):
    if value in (None, ""):
        return None
    return float(value)


def _optional_int(value):
    if value in (None, ""):
        return None
    return int(float(value))


def model_program_settings_from_controls(
    ei=0.0,
    e_low=-1.5,
    e_high=0.0,
    ef=None,
    scan_rate=0.1,
    segments=2,
    points_per_segment=300,
    quiet_time=0,
    plot_options=None,
):
    settings = {
        "Ei": 0.0 if ei in (None, "") else float(ei),
        "E_low": _optional_float(e_low),
        "E_high": _optional_float(e_high),
        "Ef": _optional_float(ef),
        "scan_rate": 0.1 if scan_rate in (None, "") else float(scan_rate),
        "segments": 2 if segments in (None, "") else int(float(segments)),
        "points_per_segment": 300 if points_per_segment in (None, "") else int(float(points_per_segment)),
        "quiet_time": 0.0 if quiet_time in (None, "") else float(quiet_time),
        "plot quiet time": "quiet_time" in (plot_options or []),
    }
    return settings


def model_cv_data_settings_from_controls(
    cv_index=0,
    trim_mode="expand",
    window_min=None,
    window_max=None,
    segments=None,
    stride=20,
    estimate_cdl="auto",
):
    trim_mode = str(trim_mode or "none").strip().lower()
    settings = {
        "cv_index": 0 if cv_index in (None, "") else int(float(cv_index)),
        "stride": 20 if stride in (None, "") else int(float(stride)),
    }
    if trim_mode != "none":
        settings["trim mode"] = trim_mode or "expand"
    if trim_mode != "none" and (window_min not in (None, "") or window_max not in (None, "")):
        settings["potential window"] = [_optional_float(window_min), _optional_float(window_max)]
    if segments not in (None, ""):
        text = str(segments).strip()
        if "," in text:
            settings["segments"] = [int(float(part.strip())) for part in text.split(",") if part.strip()]
        elif "-" in text:
            start, end = [int(float(part.strip())) for part in text.split("-", 1)]
            settings["segments"] = list(range(start, end + 1))
        else:
            settings["segment"] = int(float(text))
    if estimate_cdl and estimate_cdl != "off":
        settings["estimate Cdl"] = estimate_cdl
    return settings


def model_action_button_labels(simulate_mode="scratch"):
    return "Plot CV Program", "Simulate CV"


def model_cv_source_state(objects_state, current_mode="scratch", registry=default_registry):
    dataset_id = (objects_state or {}).get("dataset_id")
    cv_index = first_analysis_index(dataset_id, "cv", registry=registry)
    has_cv = cv_index is not None
    options = [
        {"label": "From Scratch", "value": "scratch"},
        {"label": "From CV", "value": "cv", "disabled": not has_cv},
    ]
    mode = str(current_mode or "scratch").strip().lower()
    if mode == "cv" and not has_cv:
        mode = "scratch"
    elif mode not in {"scratch", "cv"}:
        mode = "scratch"
    value = "" if cv_index is None else str(cv_index)
    disabled = not has_cv
    status = "No CV objects loaded." if not has_cv else ""
    return options, mode, value, disabled, status, value, disabled


def model_fit_index_from_cv_index(cv_index, simulate_mode="scratch", current_fit_index=None):
    if str(simulate_mode or "scratch").strip().lower() == "cv" and cv_index not in (None, ""):
        return str(cv_index)
    return current_fit_index


def model_input_card_visibility(simulate_mode="scratch"):
    mode = str(simulate_mode or "scratch").strip().lower()
    return (
        {} if mode == "scratch" else {"display": "none"},
        {} if mode == "cv" else {"display": "none"},
    )


def model_over_conditions_visibility(values=None):
    return {} if "conditions" in (values or []) else {"display": "none"}


def model_condition_axis_controls(axis="scan_rate"):
    axis = str(axis or "scan_rate").strip().lower()
    defaults = {
        "scan_rate": (0, 1, 0.01, [0.05, 0.2], {0.05: "0.05", 0.2: "0.2"}),
        "concentration": (0, 0.01, 0.0001, [0.001, 0.003], {0.001: "1 mM", 0.003: "3 mM"}),
        "temperature": (250, 350, 1, [280, 320], {280: "280 K", 320: "320 K"}),
    }
    return defaults.get(axis, defaults["scan_rate"])


def model_condition_species_visibility(axis="scan_rate"):
    return {} if str(axis or "scan_rate").strip().lower() == "concentration" else {"display": "none"}


def model_condition_settings_from_controls(
    axis="scan_rate",
    condition_range=None,
    condition_count=3,
    condition_species="a",
):
    axis = str(axis or "scan_rate").strip().lower()
    settings = {"condition_axis": axis}
    if isinstance(condition_range, str):
        settings["condition_values"] = condition_range
    elif condition_range is not None:
        _axis_min, _axis_max, _step, default_range, _marks = model_condition_axis_controls(axis)
        try:
            range_values = list(condition_range)
        except TypeError:
            range_values = []
        raw_min = range_values[0] if len(range_values) > 0 else None
        raw_max = range_values[1] if len(range_values) > 1 else None
        condition_min = _optional_float(raw_min)
        condition_max = _optional_float(raw_max)
        settings["condition_min"] = default_range[0] if condition_min is None else condition_min
        settings["condition_max"] = default_range[1] if condition_max is None else condition_max
        settings["condition_count"] = 3 if condition_count in (None, "") else int(float(condition_count))
    else:
        _axis_min, _axis_max, _step, default_range, _marks = model_condition_axis_controls(axis)
        settings.update({"condition_min": default_range[0], "condition_max": default_range[1], "condition_count": 3})
    if axis == "concentration":
        settings["condition_species"] = condition_species or "a"
    return settings


def model_cv_window_visibility(trim_mode="expand"):
    return {"display": "none"} if str(trim_mode or "none").strip().lower() == "none" else {}


def _model_row_field_has_value(row_data, field):
    for row in row_data or []:
        if not isinstance(row, dict):
            continue
        value = row.get(field)
        if value is None:
            continue
        if isinstance(value, str):
            if value.strip():
                return True
            continue
        try:
            if math.isnan(value):
                continue
        except (TypeError, ValueError):
            pass
        return True
    return False


def model_bound_column_defs(show_values=None, section="basic", fit_enabled=True, row_data=None):
    show = "bounds" in (show_values or [])
    columns = []
    if section == "setup":
        columns.append({"field": "path", "headerName": "Path", "editable": False})
    if section == "mechanism":
        columns.append({"field": "step", "headerName": "Step", "editable": False})
    if section == "species":
        columns.extend(
            [
                {
                    "field": "phase",
                    "headerName": "Phase",
                    "editable": True,
                    "cellClass": "ecat-editable-cell",
                    "cellEditor": "agSelectCellEditor",
                    "cellEditorParams": {"values": ["bulk", "surface"]},
                },
                {
                    "field": "species",
                    "headerName": "Species",
                    "editable": True,
                    "cellClass": "ecat-editable-cell",
                    "cellEditor": "agTextCellEditor",
                },
            ]
        )
    columns.extend(
        [
            {"field": "name", "headerName": "Name", "editable": False},
            {
                "field": "initial",
                "headerName": "Initial" if fit_enabled else "Value",
                "editable": True,
                "cellClass": "ecat-editable-cell",
                "cellEditor": "agTextCellEditor",
            },
        ]
    )
    if fit_enabled:
        columns.extend(
            [
                {
                    "field": "lower",
                    "headerName": "Lower",
                    "editable": True,
                    "cellClass": "ecat-editable-cell",
                    "cellEditor": "agTextCellEditor",
                    "hide": not show,
                },
                {
                    "field": "upper",
                    "headerName": "Upper",
                    "editable": True,
                    "cellClass": "ecat-editable-cell",
                    "cellEditor": "agTextCellEditor",
                    "hide": not show,
                },
                {
                    "field": "vary",
                    "headerName": "Fit?",
                    "editable": True,
                    "cellRenderer": "agCheckboxCellRenderer",
                    "cellEditor": "agCheckboxCellEditor",
                },
            ]
        )
        if _model_row_field_has_value(row_data, "final"):
            columns.append({"field": "final", "headerName": "Fitted Value", "editable": False})
        if _model_row_field_has_value(row_data, "stderr"):
            columns.append({"field": "stderr", "headerName": "Std. Error", "editable": False})
        if _model_row_field_has_value(row_data, "comment"):
            columns.append({"field": "comment", "headerName": "Comment", "editable": False})
    return columns


def model_fit_table_state(active_tab="simulate", model_options=None):
    visible = str(active_tab or "simulate") == "fit" and bool((model_options or {}).get("simulation_ready"))
    return visible, ({} if visible else {"display": "none"})


def _format_model_viscosity(value):
    text = f"{float(value):.3g}"
    return text.replace("e-0", "e-").replace("e+0", "e+")


def model_viscosity_dropdown_options():
    viscosities = getattr(e.simulation, "SOLVENT_VISCOSITIES", {})
    return [
        {"label": "Custom", "value": "custom"},
        *[
            {"label": f"{solvent} ({_format_model_viscosity(viscosity)} m²/s)", "value": solvent}
            for solvent, viscosity in viscosities.items()
        ],
    ]


def model_viscosity_custom_visibility(viscosity_source="custom"):
    return {} if str(viscosity_source or "custom") == "custom" else {"display": "none"}


def model_parameter_rows(mechanism="E", simulate_mode="scratch", over_conditions=False, condition_settings=None, mechanism_details=None):
    rows = [
        *model_mechanism_parameter_rows(
            mechanism,
            simulate_mode,
            over_conditions=False,
            condition_settings=condition_settings,
            mechanism_details=mechanism_details,
        ),
        *model_species_parameter_rows(
            mechanism,
            simulate_mode,
            over_conditions=False,
            condition_settings=condition_settings,
            mechanism_details=mechanism_details,
        ),
    ]
    if over_conditions:
        condition_settings = dict(condition_settings or model_condition_settings_from_controls())
        rows = [
            {"key": "condition_axis", "name": "condition axis", "initial": condition_settings.get("condition_axis", "scan_rate"), "unit": "", "lower": "", "upper": "", "vary": False},
            {"key": "condition_values", "name": "condition values", "initial": condition_settings.get("condition_values", ""), "unit": "", "lower": "", "upper": "", "vary": False},
            *rows,
        ]
    return [
        *rows,
    ]


def model_setup_parameter_rows():
    return [
        {"key": "spatial", "path": "spatial", "name": "Spatial grid", "initial": "fast", "unit": "", "lower": "", "upper": "", "vary": False},
    ]


def model_setup_custom_visibility(spatial_mode="fast"):
    return {} if str(spatial_mode or "fast").strip().lower() == "custom" else {"display": "none"}


def model_setup_parameter_rows_from_controls(
    spatial_mode="fast",
    spatial_nx=None,
    spatial_dx_fraction=None,
    spatial_viscosity=None,
    spatial_rotation=None,
    spatial_viscosity_source=None,
    spatial_solvent=None,
):
    if str(spatial_mode or "fast").strip().lower() == "custom":
        rows = []
        viscosity_source = spatial_viscosity_source if spatial_viscosity_source not in (None, "") else spatial_solvent
        viscosity_source = str(viscosity_source or "custom").strip()
        for key, name, value, unit, parser in [
            ("dx_fraction", "Spatial dx fraction", spatial_dx_fraction, "", float),
            ("nx", "Spatial nx", spatial_nx, "points", lambda item: int(float(item))),
            ("rotation", "Rotation", spatial_rotation, "Hz", float),
        ]:
            if value not in (None, ""):
                rows.append(
                    {
                        "key": f"spatial.{key}",
                        "path": f"spatial.{key}",
                        "name": name,
                        "initial": parser(value),
                        "unit": unit,
                        "lower": "",
                        "upper": "",
                        "vary": False,
                    }
                )
        if viscosity_source == "custom":
            if spatial_viscosity not in (None, ""):
                rows.append(
                    {
                        "key": "spatial.viscosity",
                        "path": "spatial.viscosity",
                        "name": "Viscosity",
                        "initial": float(spatial_viscosity),
                        "unit": "m² s⁻¹",
                        "lower": "",
                        "upper": "",
                        "vary": False,
                    }
                )
        else:
            rows.append(
                {
                    "key": "spatial.solvent",
                    "path": "spatial.solvent",
                    "name": "Viscosity",
                    "initial": viscosity_source,
                    "unit": "",
                    "lower": "",
                    "upper": "",
                    "vary": False,
                }
            )
        return rows
    return [
        {"key": "spatial", "path": "spatial", "name": "Spatial grid", "initial": spatial_mode or "fast", "unit": "", "lower": "", "upper": "", "vary": False}
    ]


def model_species_parameter_rows(
    mechanism="E",
    simulate_mode="scratch",
    over_conditions=False,
    condition_settings=None,
    mechanism_details=None,
):
    del simulate_mode
    species = _model_species_symbols_for_rows(mechanism, mechanism_details)
    rows = []
    for symbol, phase in species:
        key_symbol = _model_species_key(symbol)
        rows.append(
            {
                "key": f"D_{key_symbol}",
                "path": f"diffusion.{symbol}",
                "phase": phase,
                "name": "D (cm² s⁻¹)",
                "group": "diffusion",
                "species": symbol,
                "initial": 1e-5,
                "unit": "cm^2 s^-1",
                "lower": 0.0,
                "upper": "",
                "vary": False,
            }
        )
    for index, (symbol, phase) in enumerate(species):
        key_symbol = _model_species_key(symbol)
        rows.append(
            {
                "key": f"C_{key_symbol}",
                "path": f"concentrations.{phase}.{symbol}",
                "phase": phase,
                "name": "C (M)",
                "group": "concentration",
                "species": symbol,
                "initial": 1e-3 if index == 0 else 0.0,
                "unit": "M",
                "lower": 0.0,
                "upper": "",
                "vary": False,
            }
        )
    if not rows:
        rows = [
            {"key": "D_A", "path": "diffusion.A", "phase": "bulk", "name": "D (cm² s⁻¹)", "group": "diffusion", "species": "A", "initial": 1e-5, "unit": "cm^2 s^-1", "lower": 0.0, "upper": "", "vary": False},
            {"key": "D_B", "path": "diffusion.B", "phase": "bulk", "name": "D (cm² s⁻¹)", "group": "diffusion", "species": "B", "initial": 1e-5, "unit": "cm^2 s^-1", "lower": 0.0, "upper": "", "vary": False},
            {"key": "C_A", "path": "concentrations.bulk.A", "phase": "bulk", "name": "C (M)", "group": "concentration", "species": "A", "initial": 1e-3, "unit": "M", "lower": 0.0, "upper": "", "vary": False},
            {"key": "C_B", "path": "concentrations.bulk.B", "phase": "bulk", "name": "C (M)", "group": "concentration", "species": "B", "initial": 0.0, "unit": "M", "lower": 0.0, "upper": "", "vary": False},
        ]
    if over_conditions:
        condition_settings = dict(condition_settings or model_condition_settings_from_controls())
        rows = [
            {"key": "condition_axis", "name": "condition axis", "initial": condition_settings.get("condition_axis", "scan_rate"), "unit": "", "lower": "", "upper": "", "vary": False},
            {"key": "condition_values", "name": "condition values", "initial": condition_settings.get("condition_values", ""), "unit": "", "lower": "", "upper": "", "vary": False},
            *rows,
        ]
    return rows


def model_mechanism_parameter_rows(mechanism="E", simulate_mode="scratch", over_conditions=False, condition_settings=None, mechanism_details=None):
    del simulate_mode, over_conditions, condition_settings
    rows = []
    kinetics_index = 0
    reaction_index = 0
    for detail in _model_mechanism_details_for_rows(mechanism, mechanism_details):
        step = _model_mechanism_step_label(detail)
        kind = str(detail.get("kind") or "").strip().lower()
        species = _model_mechanism_species_label(detail)
        if kind.startswith("electron"):
            rows.extend(
                [
                    {"key": f"E0_{kinetics_index}", "path": f"kinetics.{kinetics_index}.E0", "step": step, "name": "E⁰ (V)", "group": "kinetics", "species": species, "initial": _model_default_e0(kinetics_index), "unit": "V", "lower": "", "upper": "", "vary": True},
                    {"key": f"k0_{kinetics_index}", "path": f"kinetics.{kinetics_index}.k0", "step": step, "name": "k₀ (m s⁻¹)", "group": "kinetics", "species": species, "initial": 1e-3, "unit": "m s^-1", "lower": 1e-30, "upper": "", "vary": True},
                    {"key": f"alpha_{kinetics_index}", "path": f"kinetics.{kinetics_index}.alpha", "step": step, "name": "α", "group": "kinetics", "species": species, "initial": 0.5, "unit": "", "lower": 0.0, "upper": 1.0, "vary": True},
                ]
            )
            if kinetics_index == 0:
                rows[-3]["key"] = "E0"
                rows[-2]["key"] = "k0"
                rows[-1]["key"] = "alpha"
            kinetics_index += 1
        else:
            rows.extend(
                [
                    {"key": f"kf_{reaction_index}", "path": f"reactions.{reaction_index}.kf", "step": step, "name": f"k{_subscript_number(reaction_index + 1)} (s⁻¹)", "group": "reactions", "species": species, "initial": 1.0, "unit": "s^-1", "lower": 0.0, "upper": "", "vary": True},
                    {"key": f"kb_{reaction_index}", "path": f"reactions.{reaction_index}.kb", "step": step, "name": f"k₋{_subscript_number(reaction_index + 1)} (s⁻¹)", "group": "reactions", "species": species, "initial": 0.0, "unit": "s^-1", "lower": 0.0, "upper": "", "vary": False},
                ]
            )
            reaction_index += 1
    return rows


def _model_mechanism_details_for_rows(mechanism="E", mechanism_details=None):
    if mechanism_details:
        return list(mechanism_details)
    mechanism_text = str(mechanism or "E")
    source = "custom" if ":" in mechanism_text or "\n" in mechanism_text else "preset"
    validation = validate_simulation_mechanism(source, mechanism_text, mechanism_text)
    return list(validation.get("mechanism_details") or [])


def _model_mechanism_step_label(detail):
    equation = str(detail.get("equation") or "")
    prefix = equation.split(":", 1)[0].strip()
    return prefix or str(detail.get("index") or "")


def _model_mechanism_species_label(detail):
    reactants = str(detail.get("reactants") or "").strip()
    products = str(detail.get("products") or "").strip()
    if reactants and products:
        return f"{_model_display_species_expression(reactants)}/{_model_display_species_expression(products)}"
    return _model_display_species_expression(reactants or products or "")


def _model_species_symbols_for_rows(mechanism="E", mechanism_details=None):
    symbols = []
    for detail in _model_mechanism_details_for_rows(mechanism, mechanism_details):
        for field in ("reactants", "products"):
            for token in str(detail.get(field) or "").split("+"):
                symbol, phase = _model_species_symbol_and_phase(token)
                if symbol and (symbol, phase) not in symbols:
                    symbols.append((symbol, phase))
    return symbols


def _model_species_symbol_and_phase(value):
    text = str(value or "").strip()
    if not text:
        return "", "bulk"
    phase = "surface" if text.endswith("*") else "bulk"
    text = text.rstrip("*").strip()
    if len(text) == 1 and text.islower():
        text = text.upper()
    return text, phase


def _model_display_species_expression(value):
    parts = []
    for token in str(value or "").split("+"):
        symbol, phase = _model_species_symbol_and_phase(token)
        if symbol:
            parts.append(f"{symbol}*" if phase == "surface" else symbol)
    return "+".join(parts)


def _model_species_key(symbol):
    text = str(symbol or "").strip().rstrip("*") or "species"
    return "".join(ch if ch.isalnum() else "_" for ch in text)


def _model_default_e0(index):
    return -0.5 - 0.5 * int(index)


def merge_model_parameter_rows(default_rows, existing_rows):
    existing_by_path = {str(row.get("path")): row for row in existing_rows or [] if row.get("path")}
    existing_by_key = {str(row.get("key")): row for row in existing_rows or [] if row.get("key")}
    editable_fields = {"initial", "lower", "upper", "vary", "phase", "species"}
    merged = []
    for default_row in default_rows or []:
        row = dict(default_row)
        existing = existing_by_path.get(str(row.get("path"))) or existing_by_key.get(str(row.get("key")))
        if existing:
            for field in editable_fields:
                if field in existing:
                    row[field] = existing[field]
        merged.append(row)
    return merged


def _subscript_number(value) -> str:
    return str(value).translate(str.maketrans("0123456789-", "₀₁₂₃₄₅₆₇₈₉₋"))


def model_cell_parameter_rows():
    return [
        {"key": "T", "name": "T (K)", "initial": 298.15, "unit": "K", "lower": 0.0, "upper": "", "vary": False},
        {"key": "Ru", "name": "Rᵤ (Ω)", "initial": 0.0, "unit": "ohm", "lower": 0.0, "upper": "", "vary": False},
        {"key": "Cdl", "name": "Cdl (F)", "initial": "auto", "unit": "F", "lower": 0.0, "upper": "", "vary": False},
        {"key": "A", "name": "A (m²)", "initial": 1e-5, "unit": "m^2", "lower": 0.0, "upper": "", "vary": False},
    ]


def build_model_simulation_state(
    model_options,
    simulate_mode="scratch",
    program_settings=None,
    cv_data_settings=None,
    over_conditions=None,
    condition_settings=None,
):
    model_options = dict(model_options or {})
    if not model_options.get("mechanism_valid"):
        model_options["simulation_ready"] = False
        model_options["simulation_result"] = {
            "status": "blocked",
            "message": "Choose a valid mechanism before simulating.",
        }
        return model_options
    mechanism = model_options.get("mechanism") or model_options.get("mechanism_preset") or "E"
    mechanism_details = model_options.get("mechanism_details") or []
    over_conditions_enabled = "conditions" in (over_conditions or [])
    model_options.update(
        {
            "simulate_mode": simulate_mode or "scratch",
            "over_conditions": over_conditions_enabled,
            "program_settings": dict(program_settings or model_program_settings_from_controls()),
            "cv_data_settings": dict(cv_data_settings or model_cv_data_settings_from_controls()),
            "condition_settings": dict(condition_settings or model_condition_settings_from_controls()),
            "simulation_ready": True,
            "setup_parameters": model_setup_parameter_rows(),
            "species_parameters": model_species_parameter_rows(
                mechanism,
                simulate_mode,
                over_conditions_enabled,
                condition_settings,
                mechanism_details,
            ),
            "mechanism_parameters": model_mechanism_parameter_rows(mechanism, simulate_mode, over_conditions_enabled, condition_settings, mechanism_details),
            "parameters": model_parameter_rows(mechanism, simulate_mode, over_conditions_enabled, condition_settings, mechanism_details),
            "cell_parameters": model_cell_parameter_rows(),
            "simulation_result": {
                "status": "placeholder",
                "message": "Simulation starting state generated. Fit is now available.",
            },
        }
    )
    model_options.pop("fit_requested", None)
    model_options.pop("fit_result", None)
    return model_options


def attach_model_simulation_result(model_options, result):
    model_options = dict(model_options or {})
    result = dict(result or {})
    model_options["simulation_result"] = result
    model_options["simulation_ready"] = result.get("status") == "ok"
    if result.get("params"):
        model_options["simulation_params"] = result.get("params")
    return model_options


def model_rows_for_simulation(model_options, parameter_rows=None, cell_parameter_rows=None):
    model_options = dict(model_options or {})
    parameters = list(parameter_rows or model_parameter_row_data(model_options))
    cell_parameters = list(cell_parameter_rows or model_cell_parameter_row_data(model_options))
    return parameters, cell_parameters


def model_split_rows_for_simulation(
    model_options,
    mechanism_rows=None,
    cell_parameter_rows=None,
    species_rows=None,
    setup_rows=None,
):
    model_options = dict(model_options or {})
    setup_parameters = list(setup_rows or model_setup_parameter_row_data(model_options))
    species_parameters = list(species_rows or model_species_parameter_row_data(model_options))
    mechanism_parameters = list(mechanism_rows or model_mechanism_parameter_row_data(model_options))
    cell_parameters = list(cell_parameter_rows or model_cell_parameter_row_data(model_options))
    return [*species_parameters, *mechanism_parameters], cell_parameters, setup_parameters


def model_fit_compatibility(model_options, fit_mode="single", fit_cv_index=None, objects=None):
    if str(fit_mode or "single") != "single":
        return "ok", ""
    if objects is None:
        return "ok", ""
    try:
        index = 0 if fit_cv_index in (None, "") else int(float(fit_cv_index))
    except (TypeError, ValueError):
        return "blocked", "Choose a valid CV index before fitting."
    objects = list(objects or [])
    if not objects or index < 0 or index >= len(objects):
        return "blocked", f"CV index {index} is not loaded."
    obj = objects[index]
    if type(obj).__name__ != "cv":
        return "blocked", f"Object at index {index} is not a CV."

    warnings = []
    program_scan_rate = (model_options.get("program_settings") or {}).get("scan_rate")
    object_scan_rate = getattr(obj, "scan_rate", None)
    try:
        if program_scan_rate not in (None, "") and object_scan_rate not in (None, ""):
            program_scan_rate = float(program_scan_rate)
            object_scan_rate = float(object_scan_rate)
            if abs(program_scan_rate - object_scan_rate) > max(1e-12, abs(program_scan_rate) * 1e-6):
                warnings.append(f"Scan-rate mismatch: program {program_scan_rate:g} V s⁻¹, CV {object_scan_rate:g} V s⁻¹.")
    except (TypeError, ValueError):
        pass
    return "warning" if warnings else "ok", " ".join(warnings)


def build_model_fit_state(
    model_options,
    fit_mode="single",
    parameter_rows=None,
    cell_parameter_rows=None,
    fit_cv_index=None,
    objects=None,
):
    model_options = dict(model_options or {})
    if not model_options.get("simulation_ready"):
        model_options["fit_requested"] = False
        model_options["fit_result"] = {
            "status": "blocked",
            "message": "Run a simulation first to create the fit starting guess.",
        }
        return model_options
    compatibility_status, compatibility_message = model_fit_compatibility(
        model_options,
        fit_mode,
        fit_cv_index,
        objects,
    )
    if compatibility_status == "blocked":
        model_options["fit_requested"] = False
        model_options["fit_result"] = {
            "status": "blocked",
            "message": compatibility_message,
        }
        return model_options
    if parameter_rows:
        model_options["parameters"] = list(parameter_rows)
    if cell_parameter_rows:
        model_options["cell_parameters"] = list(cell_parameter_rows)
    if fit_cv_index not in (None, ""):
        model_options["fit_cv_index"] = int(float(fit_cv_index))
    model_options.update(
        {
            "fit_requested": True,
            "fit_mode": fit_mode or "single",
            "fit_result": {
                "status": "placeholder",
                "message": (
                    f"Fit request captured. {compatibility_message} The simulator/fitter engine is not wired yet."
                    if compatibility_message
                    else "Fit request captured. The simulator/fitter engine is not wired yet."
                ),
            },
        }
    )
    return model_options


def model_results_content(model_options):
    model_options = dict(model_options or {})
    parameter_content = ""
    cell_content = ""

    entries = model_result_entries(model_options)
    if entries:
        return parameter_content, cell_content, render_model_result_entries(entries)

    result = model_options.get("fit_result") or model_options.get("simulation_result") or {}
    message_lines = []
    if model_options.get("simulate_mode"):
        message_lines.append(f"Simulation mode: {model_options.get('simulate_mode')}")
    if model_options.get("over_conditions"):
        message_lines.append("Over conditions: enabled")
    if model_options.get("fit_mode"):
        message_lines.append(f"Fit mode: {model_options.get('fit_mode')}")
    message_lines.append(result.get("message") or "Simulation overlays, residuals, and fit summaries will appear here.")
    message = "\n".join(message_lines)
    return parameter_content, cell_content, message


def model_result_entries(model_options):
    model_options = dict(model_options or {})
    entries = []

    program_result = dict(model_options.get("program_result") or {})
    if program_result.get("plot") or program_result.get("message"):
        entries.append(
            {
                "key": "program",
                "title": "CV Program",
                "plot": program_result.get("plot"),
                "message": program_result.get("message") or "CV program plotted.",
            }
        )

    simulation_result = dict(model_options.get("simulation_result") or {})
    if simulation_result.get("plot") or simulation_result.get("message"):
        message_lines = []
        if model_options.get("simulate_mode"):
            message_lines.append(f"Simulation mode: {model_options.get('simulate_mode')}")
        if model_options.get("over_conditions"):
            message_lines.append("Over conditions: enabled")
        message_lines.append(simulation_result.get("message") or "Simulation complete.")
        entries.append(
            {
                "key": "simulation",
                "title": "Simulated CV",
                "plot": simulation_result.get("plot"),
                "message": "\n".join(message_lines),
            }
        )

    fit_result = dict(model_options.get("fit_result") or {})
    if fit_result.get("plot") or fit_result.get("message"):
        message_lines = []
        if model_options.get("fit_mode"):
            message_lines.append(f"Fit mode: {model_options.get('fit_mode')}")
        message_lines.append(fit_result.get("message") or "Fit complete.")
        entries.append(
            {
                "key": "fit",
                "title": "Fit Result",
                "plot": fit_result.get("plot"),
                "message": "\n".join(message_lines),
            }
        )
    return entries


def render_model_result_entries(entries):
    return html.Div(
        className="ecat-model-result-list",
        children=[
            html.Details(
                id=f"ecat-model-result-{entry['key']}",
                className="ecat-card ecat-model-result-entry",
                open=True,
                children=[
                    html.Summary(entry["title"], className="ecat-card-summary"),
                    _plot_frame(entry.get("plot")),
                    html.Pre(entry.get("message") or "", className="ecat-muted-note"),
                ],
            )
            for entry in entries
        ],
    )


def model_parameter_row_data(model_options):
    return list((model_options or {}).get("parameters") or [])


def model_setup_parameter_row_data(model_options):
    return list((model_options or {}).get("setup_parameters") or model_setup_parameter_rows())


def model_species_parameter_row_data(model_options):
    model_options = dict(model_options or {})
    rows = model_options.get("species_parameters")
    if rows:
        return list(rows)
    return [row for row in model_parameter_row_data(model_options) if row.get("group") in {"concentration", "diffusion"}]


def model_mechanism_parameter_row_data(model_options):
    model_options = dict(model_options or {})
    rows = model_options.get("mechanism_parameters")
    if rows:
        return list(rows)
    return [row for row in model_parameter_row_data(model_options) if row.get("group") not in {"concentration", "diffusion"}]


def model_cell_parameter_row_data(model_options):
    return list((model_options or {}).get("cell_parameters") or [])


def model_result_plot(model_options):
    model_options = dict(model_options or {})
    fit_result = dict(model_options.get("fit_result") or {})
    if fit_result.get("plot"):
        return fit_result.get("plot")
    simulation_result = dict(model_options.get("simulation_result") or {})
    if simulation_result.get("plot"):
        return simulation_result.get("plot")
    if not model_options.get("simulation_ready"):
        return None
    try:
        return render_model_program_plot(
            model_parameter_row_data(model_options),
            {"_format": "png", "_dpi": 150},
            program_settings=model_options.get("program_settings") or {},
        )
    except Exception:
        return None


def model_program_plot_from_controls(
    ei=0.0,
    e_low=-1.5,
    e_high=0.0,
    ef=None,
    scan_rate=0.1,
    segments=2,
    points_per_segment=300,
    quiet_time=0,
    plot_options=None,
):
    settings = model_program_settings_from_controls(
        ei,
        e_low,
        e_high,
        ef,
        scan_rate,
        segments,
        points_per_segment,
        quiet_time,
        plot_options,
    )
    return render_model_program_plot([], {"_format": "png", "_dpi": 150}, program_settings=settings)


def model_cv_program_plot_from_controls(
    objects_state,
    cv_index=0,
    trim_mode="expand",
    window_min=None,
    window_max=None,
    segments=None,
    stride=20,
    estimate_cdl="auto",
    *,
    program_scan_rate=0.1,
    registry=default_registry,
):
    cv_data_settings = model_cv_data_settings_from_controls(
        cv_index,
        trim_mode,
        window_min,
        window_max,
        segments,
        stride,
        estimate_cdl,
    )
    program_settings = {"scan_rate": 0.1 if program_scan_rate in (None, "") else float(program_scan_rate)}
    return render_browser_cv_data_program_plot(
        registry.get((objects_state or {}).get("dataset_id")),
        cv_data_settings,
        program_settings,
    )


def update_workflow_model_options(workflow_data, model_options):
    workflow = AppWorkflow.from_dict(workflow_data)
    workflow.model_options = dict(model_options or {})
    return workflow.to_dict()


def update_model_options_with_mechanism_details(model_options, mechanism_details):
    model_options = dict(model_options or {})
    details = list(mechanism_details or [])
    mechanism = model_options.get("mechanism") or model_options.get("mechanism_preset") or "E"
    model_options["mechanism_details"] = details
    default_mechanism_rows = model_mechanism_parameter_rows(
        mechanism,
        model_options.get("simulate_mode") or "scratch",
        model_options.get("over_conditions") or False,
        model_options.get("condition_settings") or None,
        details,
    )
    model_options["mechanism_parameters"] = merge_model_parameter_rows(
        default_mechanism_rows,
        model_mechanism_parameter_row_data(model_options),
    )
    default_species_rows = model_species_parameter_rows(
        mechanism,
        model_options.get("simulate_mode") or "scratch",
        model_options.get("over_conditions") or False,
        model_options.get("condition_settings") or None,
        details,
    )
    species_rows = merge_model_parameter_rows(
        default_species_rows,
        model_species_parameter_row_data(model_options),
    )
    model_options["species_parameters"] = species_rows
    model_options["parameters"] = [
        *species_rows,
        *model_options["mechanism_parameters"],
    ]
    model_options["simulation_ready"] = False
    model_options.pop("fit_requested", None)
    model_options.pop("fit_result", None)
    return model_options


def model_fit_gate(model_options):
    ready = bool((model_options or {}).get("simulation_ready"))
    message = "" if ready else "Run a simulation first to create the fit starting guess."
    return (not ready, not ready, message)


def model_simulate_gate(model_options):
    valid = bool((model_options or {}).get("mechanism_valid"))
    message = "" if valid else "Choose a valid mechanism before simulating."
    return (not valid, message)


def toggle_about_hidden(_clicks, hidden=True) -> bool:
    if not _clicks:
        return bool(hidden)
    return not bool(hidden)


def toggle_about_state(_clicks, hidden=True) -> tuple[bool, str]:
    next_hidden = toggle_about_hidden(_clicks, hidden)
    return next_hidden, "About" if next_hidden else "X"


def register_callbacks(app):
    try:
        from dash import Input, Output, State, ctx, html
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "The eCAT app requires Dash. Reinstall or upgrade eCAT with `pip install -e .`."
        ) from exc

    @app.callback(
        Output("ecat-reference-offset-wrap", "style"),
        Output("ecat-reference-label-wrap", "style"),
        Output("ecat-reference-file-wrap", "style"),
        Output("ecat-reference-keyword-wrap", "style"),
        Output("ecat-reference-keywords-wrap", "style"),
        Output("ecat-reference-guess-wrap", "style"),
        Output("ecat-allow-self-reference-wrap", "style"),
        Input("ecat-reference-mode", "value"),
    )
    def update_reference_field_visibility(mode):
        visible = reference_field_visibility(mode)

        def style(show):
            return {} if show else {"display": "none"}

        return (
            style(visible["manual"]),
            style(visible["label"]),
            style(visible["file"]),
            style(visible["keyword"]),
            style(visible["auto"]),
            style(visible["guess"]),
            style(visible["allow_self_reference"]),
        )

    @app.callback(
        Output("ecat-objects-store", "data"),
        Output("ecat-workflow-store", "data"),
        Output("ecat-import-warnings", "children"),
        Output("ecat-import-conditions", "children"),
        Output("ecat-object-table", "rowData"),
        Output("ecat-object-table", "columnDefs"),
        Output("ecat-object-table", "selectedRows"),
        Output("ecat-selected-row-ids-store", "data"),
        Output("ecat-default-plot", "children"),
        Output("ecat-reference-file", "options"),
        Output("ecat-status", "children"),
        Output("ecat-table-extra-columns", "options"),
        Output("ecat-table-extra-columns", "value"),
        Output("ecat-scroll-target", "children"),
        Input("ecat-load-path", "n_clicks"),
        Input("ecat-upload", "contents"),
        Input("ecat-apply-reference", "n_clicks"),
        Input("ecat-example-folder", "value"),
        State("ecat-local-path", "value"),
        State("ecat-recursive", "value"),
        State("ecat-upload", "filename"),
        State("ecat-objects-store", "data"),
        State("ecat-workflow-store", "data"),
        State("ecat-reference-mode", "value"),
        State("ecat-reference-offset", "value"),
        State("ecat-reference-label", "value"),
        State("ecat-reference-file", "value"),
        State("ecat-reference-keyword", "value"),
        State("ecat-reference-keywords", "value"),
        State("ecat-reference-guess", "value"),
        State("ecat-allow-self-reference", "value"),
        prevent_initial_call=True,
    )
    def load_input_callback(
        _clicks,
        upload_contents,
        _apply_clicks,
        example_folder,
        path,
        recursive_values,
        upload_filenames,
        objects_state,
        workflow_data,
        reference_mode,
        reference_offset,
        reference_label,
        reference_file,
        reference_keyword,
        reference_keywords,
        reference_guess,
        allow_self_reference,
    ):
        if ctx.triggered_id == "ecat-upload":
            state = handle_upload_load(upload_filenames, upload_contents)
        elif ctx.triggered_id == "ecat-example-folder":
            state = handle_example_folder_load(example_folder)
        elif ctx.triggered_id == "ecat-apply-reference":
            resolved_reference_file = reference_file_path_from_index(objects_state, reference_file)
            settings = {
                "mode": reference_mode,
                "offset": reference_offset,
                "label": reference_label,
                "file": resolved_reference_file,
                "file_index": reference_file,
                "keyword": reference_keyword,
                "keywords": reference_keywords,
                "guess": reference_guess,
                "allow_self_reference": "allow" in (allow_self_reference or []),
            }
            state = handle_apply_reference(workflow_data, settings)
        else:
            state = handle_local_path_load(path, recursive="recursive" in (recursive_values or []))
        warnings = [html.Div(warning) for warning in state["warnings"]]
        condition_values = state.get("conditions", [])
        conditions = []
        if condition_values:
            conditions.append(html.Div("Shared Conditions", className="ecat-condition-heading"))
            conditions.extend(
                html.Div(condition, className="ecat-condition-item")
                for condition in condition_values
            )
        rows = state.get("summary", [])
        table = state.get("table", {"data": rows, "columns": []})
        plot = _plot_frame(state.get("plot"), include_refresh=True, save_src=state.get("save_plot"))
        file_options = reference_file_options(rows)
        object_options = [
            {"label": row.get("filename") or row.get("name") or str(row.get("index")), "value": row.get("index")}
            for row in rows
        ]
        return (
            state,
            state["workflow"],
            warnings,
            conditions,
            table["data"],
            ag_grid_column_defs(table),
            selected_grid_rows_for_ids(table["data"], state.get("included_row_ids", [])),
            state.get("included_row_ids", []),
            plot,
            file_options,
            state.get("status", ""),
            available_column_options(default_registry.get(state.get("dataset_id"))),
            selected_column_values(table),
            scroll_signal("import"),
        )

    @app.callback(
        Output("ecat-session-label", "children"),
        Input("ecat-objects-store", "data"),
    )
    def header_session_label_callback(state):
        return header_session_label(state)

    @app.callback(
        Output("ecat-workflow-store", "data", allow_duplicate=True),
        Input("ecat-selected-row-ids-store", "data"),
        State("ecat-workflow-store", "data"),
        prevent_initial_call=True,
    )
    def include_rows_callback(selected_row_ids, workflow_data):
        return handle_include_rows(workflow_data, selected_row_ids)

    @app.callback(
        Output("ecat-selected-row-ids-store", "data", allow_duplicate=True),
        Input("ecat-object-table", "selectedRows"),
        prevent_initial_call=True,
    )
    def grid_selection_callback(selected_rows):
        return selected_row_ids_from_grid_rows(selected_rows)

    @app.callback(
        Output("ecat-object-table", "rowData", allow_duplicate=True),
        Output("ecat-object-table", "columnDefs", allow_duplicate=True),
        Output("ecat-table-extra-columns", "value", allow_duplicate=True),
        Input("ecat-table-extra-columns", "value"),
        State("ecat-objects-store", "data"),
        prevent_initial_call=True,
    )
    def update_extra_columns(visible_columns, state):
        if not state:
            return [], [], []
        table = handle_extra_columns(state.get("dataset_id"), visible_columns)
        return table["data"], ag_grid_column_defs(table), selected_column_values(table)

    @app.callback(
        Output("ecat-object-table", "rowData", allow_duplicate=True),
        Output("ecat-object-table", "columnDefs", allow_duplicate=True),
        Output("ecat-table-extra-columns", "value", allow_duplicate=True),
        Input("ecat-reset-columns", "n_clicks"),
        State("ecat-objects-store", "data"),
        prevent_initial_call=True,
    )
    def reset_columns_callback(_clicks, state):
        if not state:
            return [], [], []
        table = handle_reset_columns(state.get("dataset_id"))
        return table["data"], ag_grid_column_defs(table), selected_column_values(table)

    @app.callback(
        Output("ecat-plot-options-store", "data"),
        Input("ecat-plot-legend", "value"),
        Input("ecat-plot-title-mode", "value"),
        Input("ecat-plot-convention", "value"),
        Input("ecat-plot-custom-title", "value"),
        Input("ecat-plot-style", "value"),
        Input("ecat-plot-axis-label-mode", "value"),
        Input("ecat-plot-x-label", "value"),
        Input("ecat-plot-y-label", "value"),
        Input("ecat-plot-output-mode", "value"),
        Input("ecat-plot-display-options", "value"),
        Input("ecat-plot-gradients", "value"),
        Input("ecat-plot-colorbar", "value"),
        Input("ecat-plot-label-options", "value"),
        Input("ecat-plot-trim-enabled", "value"),
        Input("ecat-plot-trim-mode", "value"),
        Input("ecat-plot-trim-min", "value"),
        Input("ecat-plot-trim-max", "value"),
        Input("ecat-plot-offset", "value"),
        Input("ecat-plot-scale-bar-height", "value"),
        Input("ecat-plot-scale-bar-location", "value"),
        Input("ecat-plot-offset-axis-options", "value"),
        Input("ecat-plot-format", "value"),
        Input("ecat-plot-dpi", "value"),
        Input("ecat-animation-fps", "value"),
        Input("ecat-animation-stride", "value"),
        Input("ecat-animation-trace-mode", "value"),
        Input("ecat-animation-schedule", "value"),
        Input("ecat-animation-stagger-time", "value"),
        Input("ecat-animation-timing-mode", "value"),
        Input("ecat-animation-timing-value", "value"),
        Input("ecat-animation-advanced", "value"),
        Input("ecat-animation-end-hold", "value"),
        Input("ecat-animation-arrow-potential", "value"),
        Input("ecat-animation-arrow-segment", "value"),
        Input("ecat-animation-scale-bar-enabled", "value"),
        Input("ecat-animation-scale-bar-length", "value"),
        Input("ecat-animation-scale-bar-location", "value"),
    )
    def update_plot_options_store(
        legend,
        title_values,
        convention,
        custom_title,
        plot_style,
        axis_label_mode,
        x_axis_label,
        y_axis_label,
        output_mode,
        display_values,
        gradient_values,
        colorbar_values,
        label_values,
        trim_values,
        trim_mode,
        trim_min,
        trim_max,
        offset,
        scale_bar_height,
        scale_bar_location,
        offset_axis_values,
        save_format,
        dpi,
        animation_fps,
        animation_stride,
        animation_trace_mode,
        animation_schedule,
        animation_stagger_time,
        animation_timing_mode,
        animation_timing_value,
        animation_advanced,
        animation_end_hold,
        animation_arrow_potential,
        animation_arrow_segment,
        animation_scale_bar_values,
        animation_scale_bar_length,
        animation_scale_bar_location,
    ):
        return plot_options_from_controls(
            legend,
            title_values,
            convention,
            custom_title,
            display_values,
            gradient_values,
            colorbar_values,
            label_values,
            trim_values=trim_values,
            trim_min=trim_min,
            trim_max=trim_max,
            offset=offset,
            scale_bar_height=scale_bar_height,
            scale_bar_location=scale_bar_location,
            offset_axis_values=offset_axis_values,
            save_format=save_format,
            dpi=dpi,
            plot_style=plot_style,
            axis_label_mode=axis_label_mode,
            x_axis_label=x_axis_label,
            y_axis_label=y_axis_label,
            trim_mode=trim_mode,
            output_mode=output_mode,
            animation_fps=animation_fps,
            animation_stride=animation_stride,
            animation_trace_mode=animation_trace_mode,
            animation_schedule=animation_schedule,
            animation_stagger_time=animation_stagger_time,
            animation_timing_mode=animation_timing_mode,
            animation_timing_value=animation_timing_value,
            animation_advanced=animation_advanced,
            animation_end_hold=animation_end_hold,
            animation_arrow_potential=animation_arrow_potential,
            animation_arrow_segment=animation_arrow_segment,
            animation_scale_bar_values=animation_scale_bar_values,
            animation_scale_bar_length=animation_scale_bar_length,
            animation_scale_bar_location=animation_scale_bar_location,
        )

    @app.callback(
        Output("ecat-workflow-store", "data", allow_duplicate=True),
        Input("ecat-plot-options-store", "data"),
        State("ecat-workflow-store", "data"),
        prevent_initial_call=True,
    )
    def update_workflow_plot_options_callback(plot_options, workflow_data):
        return update_workflow_plot_options(workflow_data, plot_options)

    @app.callback(
        Output("ecat-plot-legend-options-wrap", "style"),
        Output("ecat-plot-colorbar-wrap", "style"),
        Output("ecat-plot-custom-title-wrap", "style"),
        Output("ecat-plot-axis-label-inputs", "style"),
        Output("ecat-plot-trim-bounds", "style"),
        Output("ecat-plot-offset-controls", "style"),
        Input("ecat-plot-legend", "value"),
        Input("ecat-plot-gradients", "value"),
        Input("ecat-plot-title-mode", "value"),
        Input("ecat-plot-axis-label-mode", "value"),
        Input("ecat-plot-trim-enabled", "value"),
        Input("ecat-plot-offset", "value"),
    )
    def update_plot_control_visibility(legend_values, gradient_values, title_mode, axis_label_mode, trim_values, offset):
        visible = plot_control_visibility(legend_values, gradient_values, title_mode)
        return (
            visible["legend_options"],
            visible["colorbar"],
            visible["custom_title"],
            axis_label_controls_visibility(axis_label_mode),
            trim_bounds_visibility(trim_values),
            offset_controls_visibility(offset),
        )

    @app.callback(
        Output("ecat-animation-options", "style"),
        Output("ecat-animation-stagger-wrap", "style"),
        Output("ecat-animation-scale-bar-options", "style"),
        Input("ecat-plot-output-mode", "value"),
        Input("ecat-animation-schedule", "value"),
        Input("ecat-animation-scale-bar-enabled", "value"),
    )
    def update_animation_control_visibility(output_mode, schedule, scale_bar_values):
        return animation_controls_visibility(output_mode, schedule, scale_bar_values)

    @app.callback(
        Output("ecat-directional-arrow-options", "style"),
        Input("ecat-add-directional-arrow", "n_clicks"),
    )
    def update_directional_arrow_options_visibility(clicks):
        return directional_arrow_options_visibility(clicks)

    @app.callback(
        Output("ecat-model-program-card", "style"),
        Output("ecat-model-cv-data-card", "style"),
        Output("ecat-model-over-conditions-card", "style"),
        Output("ecat-model-plot-program", "children"),
        Output("ecat-model-run-simulate", "children"),
        Input("ecat-model-simulate-mode", "value"),
        Input("ecat-model-over-conditions", "value"),
    )
    def update_model_input_card_visibility(simulate_mode, over_conditions):
        program_style, cv_style = model_input_card_visibility(simulate_mode)
        plot_label, simulate_label = model_action_button_labels(simulate_mode)
        return program_style, cv_style, model_over_conditions_visibility(over_conditions), plot_label, simulate_label

    @app.callback(
        Output("ecat-model-simulate-mode", "options"),
        Output("ecat-model-simulate-mode", "value"),
        Output("ecat-model-cv-index", "value"),
        Output("ecat-model-cv-index", "disabled"),
        Output("ecat-model-cv-index-status", "children"),
        Output("ecat-model-fit-cv-index", "value"),
        Output("ecat-model-fit-cv-index", "disabled"),
        Input("ecat-objects-store", "data"),
        State("ecat-model-simulate-mode", "value"),
    )
    def update_model_cv_source_state(objects_state, simulate_mode):
        return model_cv_source_state(objects_state, simulate_mode)

    @app.callback(
        Output("ecat-model-fit-cv-index", "value", allow_duplicate=True),
        Input("ecat-model-cv-index", "value"),
        State("ecat-model-simulate-mode", "value"),
        State("ecat-model-fit-cv-index", "value"),
        prevent_initial_call=True,
    )
    def update_model_fit_cv_index_from_source(cv_index, simulate_mode, current_fit_index):
        return model_fit_index_from_cv_index(cv_index, simulate_mode, current_fit_index)

    @app.callback(
        Output("ecat-model-condition-range", "min"),
        Output("ecat-model-condition-range", "max"),
        Output("ecat-model-condition-range", "step"),
        Output("ecat-model-condition-range", "value"),
        Output("ecat-model-condition-range", "marks"),
        Output("ecat-model-condition-species-wrap", "style"),
        Input("ecat-model-condition-axis", "value"),
    )
    def update_model_condition_axis_controls(axis):
        slider_min, slider_max, step, value, marks = model_condition_axis_controls(axis)
        return slider_min, slider_max, step, value, marks, model_condition_species_visibility(axis)

    @app.callback(
        Output("ecat-model-spatial-custom-fields", "style"),
        Input("ecat-model-spatial-mode", "value"),
    )
    def update_model_spatial_custom_visibility(spatial_mode):
        return model_setup_custom_visibility(spatial_mode)

    @app.callback(
        Output("ecat-model-spatial-viscosity-custom", "style"),
        Input("ecat-model-spatial-viscosity-source", "value"),
    )
    def update_model_viscosity_custom_visibility(viscosity_source):
        return model_viscosity_custom_visibility(viscosity_source)

    @app.callback(
        Output("ecat-model-cv-window-fields", "style"),
        Input("ecat-model-cv-trim-mode", "value"),
    )
    def update_model_cv_window_visibility(trim_mode):
        return model_cv_window_visibility(trim_mode)

    @app.callback(
        Output("ecat-model-mechanism-parameters-grid", "columnDefs"),
        Input("ecat-model-bound-columns", "value"),
        State("ecat-model-tabs", "value"),
        State("ecat-workflow-store", "data"),
        State("ecat-model-mechanism-parameters-grid", "rowData"),
    )
    def update_model_bound_columns(values, active_tab, workflow_data, row_data):
        workflow = AppWorkflow.from_dict(workflow_data)
        fit_enabled, _style = model_fit_table_state(active_tab, workflow.model_options)
        return model_bound_column_defs(values, "mechanism", fit_enabled, row_data)

    @app.callback(
        Output("ecat-model-species-parameters-grid", "columnDefs"),
        Input("ecat-model-species-bound-columns", "value"),
        State("ecat-model-tabs", "value"),
        State("ecat-workflow-store", "data"),
        State("ecat-model-species-parameters-grid", "rowData"),
    )
    def update_model_species_bound_columns(values, active_tab, workflow_data, row_data):
        workflow = AppWorkflow.from_dict(workflow_data)
        fit_enabled, _style = model_fit_table_state(active_tab, workflow.model_options)
        return model_bound_column_defs(values, "species", fit_enabled, row_data)

    @app.callback(
        Output("ecat-model-cell-parameters-grid", "columnDefs"),
        Input("ecat-model-cell-bound-columns", "value"),
        State("ecat-model-tabs", "value"),
        State("ecat-workflow-store", "data"),
        State("ecat-model-cell-parameters-grid", "rowData"),
    )
    def update_model_cell_bound_columns(values, active_tab, workflow_data, row_data):
        workflow = AppWorkflow.from_dict(workflow_data)
        fit_enabled, _style = model_fit_table_state(active_tab, workflow.model_options)
        return model_bound_column_defs(values, "cell", fit_enabled, row_data)

    @app.callback(
        Output("ecat-model-cell-bound-columns", "style", allow_duplicate=True),
        Output("ecat-model-species-bound-columns", "style", allow_duplicate=True),
        Output("ecat-model-bound-columns", "style", allow_duplicate=True),
        Output("ecat-model-cell-parameters-grid", "columnDefs", allow_duplicate=True),
        Output("ecat-model-species-parameters-grid", "columnDefs", allow_duplicate=True),
        Output("ecat-model-mechanism-parameters-grid", "columnDefs", allow_duplicate=True),
        Input("ecat-model-tabs", "value"),
        State("ecat-workflow-store", "data"),
        State("ecat-model-cell-bound-columns", "value"),
        State("ecat-model-species-bound-columns", "value"),
        State("ecat-model-bound-columns", "value"),
        State("ecat-model-cell-parameters-grid", "rowData"),
        State("ecat-model-species-parameters-grid", "rowData"),
        State("ecat-model-mechanism-parameters-grid", "rowData"),
        prevent_initial_call=True,
    )
    def update_model_fit_table_visibility(
        active_tab,
        workflow_data,
        cell_bounds,
        species_bounds,
        mechanism_bounds,
        cell_rows,
        species_rows,
        mechanism_rows,
    ):
        workflow = AppWorkflow.from_dict(workflow_data)
        fit_enabled, style = model_fit_table_state(active_tab, workflow.model_options)
        return (
            style,
            style,
            style,
            model_bound_column_defs(cell_bounds, "cell", fit_enabled, cell_rows),
            model_bound_column_defs(species_bounds, "species", fit_enabled, species_rows),
            model_bound_column_defs(mechanism_bounds, "mechanism", fit_enabled, mechanism_rows),
        )

    @app.callback(
        Output("ecat-plot-format", "options"),
        Output("ecat-plot-format", "value"),
        Input("ecat-plot-output-mode", "value"),
    )
    def update_plot_format_options(output_mode):
        return plot_format_options(output_mode)

    @app.callback(
        Output("ecat-model-preset-wrap", "style"),
        Output("ecat-model-custom-wrap", "style"),
        Output("ecat-model-mechanism-status", "children"),
        Output("ecat-model-formatted-equations", "children"),
        Output("ecat-model-species-parameters-grid", "rowData", allow_duplicate=True),
        Output("ecat-model-mechanism-parameters-grid", "rowData", allow_duplicate=True),
        Output("ecat-workflow-store", "data", allow_duplicate=True),
        Output("ecat-model-fit-tab", "disabled"),
        Output("ecat-model-run-fit", "disabled"),
        Output("ecat-model-run-fit", "title"),
        Output("ecat-model-run-simulate", "disabled"),
        Output("ecat-model-run-simulate", "title"),
        Output("ecat-model-results", "style"),
        Input("ecat-model-mechanism-source", "data"),
        Input("ecat-model-mechanism-preset", "value"),
        Input("ecat-model-mechanism-custom", "value"),
        State("ecat-model-species-parameters-grid", "rowData"),
        State("ecat-model-mechanism-parameters-grid", "rowData"),
        State("ecat-workflow-store", "data"),
        prevent_initial_call=True,
    )
    def update_model_mechanism_callback(source, preset, custom_text, species_rows, mechanism_rows, workflow_data):
        source = "custom" if str(preset or "").strip().lower() == "custom" else source
        preset_style, custom_style = model_mechanism_visibility(source, preset)
        validation = validate_simulation_mechanism(source, preset, custom_text)
        model_options = model_mechanism_options_from_controls(source, preset, custom_text)
        if validation.get("ok"):
            model_options["mechanism_details"] = validation.get("mechanism_details") or []
            model_options["species_parameters"] = merge_model_parameter_rows(
                model_species_parameter_rows(
                    model_options.get("mechanism") or model_options.get("mechanism_preset") or "E",
                    mechanism_details=model_options["mechanism_details"],
                ),
                species_rows or [],
            )
            model_options["mechanism_parameters"] = merge_model_parameter_rows(
                model_mechanism_parameter_rows(
                    model_options.get("mechanism") or model_options.get("mechanism_preset") or "E",
                    mechanism_details=model_options["mechanism_details"],
                ),
                mechanism_rows or [],
            )
            model_options["parameters"] = [
                *model_options["species_parameters"],
                *model_options["mechanism_parameters"],
            ]
        else:
            model_options["species_parameters"] = list(species_rows or [])
            model_options["mechanism_parameters"] = list(mechanism_rows or [])
        workflow = update_workflow_model_options(workflow_data, model_options)
        disabled, run_disabled, disabled_title = model_fit_gate(model_options)
        simulate_disabled, simulate_title = model_simulate_gate(model_options)
        if validation.get("ok"):
            status = f"{validation.get('message')} Run Simulate CV to enable Fit."
        else:
            status = validation.get("message", "Mechanism is not ready.")
        return (
            preset_style,
            custom_style,
            status,
            model_formatted_equations_content(validation.get("formatted_equations") or []),
            model_options.get("species_parameters") or [],
            model_options.get("mechanism_parameters") or [],
            workflow,
            disabled,
            run_disabled,
            disabled_title,
            simulate_disabled,
            simulate_title,
            {"display": "none"},
        )

    @app.callback(
        Output("ecat-workflow-store", "data", allow_duplicate=True),
        Output("ecat-model-results", "style", allow_duplicate=True),
        Output("ecat-model-results-content", "children", allow_duplicate=True),
        Output("ecat-scroll-target", "children", allow_duplicate=True),
        Input("ecat-model-plot-program", "n_clicks"),
        Input("ecat-model-plot-cv-program", "n_clicks"),
        State("ecat-model-simulate-mode", "value"),
        State("ecat-model-program-ei", "value"),
        State("ecat-model-program-e-low", "value"),
        State("ecat-model-program-e-high", "value"),
        State("ecat-model-program-ef", "value"),
        State("ecat-model-program-scan-rate", "value"),
        State("ecat-model-program-segments", "value"),
        State("ecat-model-program-points", "value"),
        State("ecat-model-program-quiet-time", "value"),
        State("ecat-model-program-plot-options", "value"),
        State("ecat-model-cv-index", "value"),
        State("ecat-model-cv-trim-mode", "value"),
        State("ecat-model-cv-window-min", "value"),
        State("ecat-model-cv-window-max", "value"),
        State("ecat-model-cv-segments", "value"),
        State("ecat-model-cv-stride", "value"),
        State("ecat-model-cv-estimate-cdl", "value"),
        State("ecat-objects-store", "data"),
        State("ecat-workflow-store", "data"),
        prevent_initial_call=True,
    )
    def plot_model_program_callback(
        _scratch_clicks,
        _cv_clicks,
        simulate_mode,
        program_ei,
        program_e_low,
        program_e_high,
        program_ef,
        program_scan_rate,
        program_segments,
        program_points,
        program_quiet_time,
        program_plot_options,
        cv_index,
        cv_trim_mode,
        cv_window_min,
        cv_window_max,
        cv_segments,
        cv_stride,
        cv_estimate_cdl,
        objects_state,
        workflow_data,
    ):
        workflow = AppWorkflow.from_dict(workflow_data)
        model_options = dict(workflow.model_options or {})
        program_settings = model_program_settings_from_controls(
            program_ei,
            program_e_low,
            program_e_high,
            program_ef,
            program_scan_rate,
            program_segments,
            program_points,
            program_quiet_time,
            program_plot_options,
        )
        mode = str(simulate_mode or "scratch").strip().lower()
        try:
            if mode == "cv":
                plot = model_cv_program_plot_from_controls(
                    objects_state,
                    cv_index,
                    cv_trim_mode,
                    cv_window_min,
                    cv_window_max,
                    cv_segments,
                    cv_stride,
                    cv_estimate_cdl,
                    program_scan_rate=program_settings.get("scan_rate"),
                )
                program_result = {
                    "status": "ok",
                    "message": "CV-derived program plotted.",
                    "plot": plot,
                }
            else:
                plot = model_program_plot_from_controls(
                    program_ei,
                    program_e_low,
                    program_e_high,
                    program_ef,
                    program_scan_rate,
                    program_segments,
                    program_points,
                    program_quiet_time,
                    program_plot_options,
                )
                program_result = {
                    "status": "ok",
                    "message": "CV program plotted.",
                    "plot": plot,
                }
        except Exception as exc:
            program_result = {
                "status": "error",
                "message": f"CV program error: {exc}",
                "plot": None,
            }
        model_options["program_settings"] = program_settings
        model_options["program_result"] = program_result
        workflow.model_options = model_options
        _parameter_content, _cell_content, result_content = model_results_content(model_options)
        return workflow.to_dict(), {}, result_content, scroll_signal("model-program")

    @app.callback(
        Output("ecat-workflow-store", "data", allow_duplicate=True),
        Output("ecat-model-fit-tab", "disabled", allow_duplicate=True),
        Output("ecat-model-run-fit", "disabled", allow_duplicate=True),
        Output("ecat-model-run-fit", "title", allow_duplicate=True),
        Output("ecat-model-results", "style", allow_duplicate=True),
        Output("ecat-model-cell-parameters-grid", "rowData"),
        Output("ecat-model-cell-parameters-content", "children"),
        Output("ecat-model-species-parameters-grid", "rowData"),
        Output("ecat-model-species-parameters-content", "children"),
        Output("ecat-model-mechanism-parameters-grid", "rowData"),
        Output("ecat-model-mechanism-parameters-content", "children"),
        Output("ecat-model-cell-bound-columns", "style", allow_duplicate=True),
        Output("ecat-model-species-bound-columns", "style", allow_duplicate=True),
        Output("ecat-model-bound-columns", "style", allow_duplicate=True),
        Output("ecat-model-cell-parameters-grid", "columnDefs", allow_duplicate=True),
        Output("ecat-model-species-parameters-grid", "columnDefs", allow_duplicate=True),
        Output("ecat-model-mechanism-parameters-grid", "columnDefs", allow_duplicate=True),
        Output("ecat-model-results-content", "children"),
        Output("ecat-model-simulation-progress-anchor", "children"),
        Output("ecat-scroll-target", "children", allow_duplicate=True),
        Input("ecat-model-run-simulate", "n_clicks"),
        State("ecat-model-simulate-mode", "value"),
        State("ecat-model-program-ei", "value"),
        State("ecat-model-program-e-low", "value"),
        State("ecat-model-program-e-high", "value"),
        State("ecat-model-program-ef", "value"),
        State("ecat-model-program-scan-rate", "value"),
        State("ecat-model-program-segments", "value"),
        State("ecat-model-program-points", "value"),
        State("ecat-model-program-quiet-time", "value"),
        State("ecat-model-program-plot-options", "value"),
        State("ecat-model-spatial-mode", "value"),
        State("ecat-model-spatial-nx", "value"),
        State("ecat-model-spatial-dx-fraction", "value"),
        State("ecat-model-spatial-viscosity", "value"),
        State("ecat-model-spatial-viscosity-source", "value"),
        State("ecat-model-spatial-rotation", "value"),
        State("ecat-model-cv-index", "value"),
        State("ecat-model-cv-trim-mode", "value"),
        State("ecat-model-cv-window-min", "value"),
        State("ecat-model-cv-window-max", "value"),
        State("ecat-model-cv-segments", "value"),
        State("ecat-model-cv-stride", "value"),
        State("ecat-model-cv-estimate-cdl", "value"),
        State("ecat-model-over-conditions", "value"),
        State("ecat-model-condition-axis", "value"),
        State("ecat-model-condition-range", "value"),
        State("ecat-model-condition-count", "value"),
        State("ecat-model-condition-species", "value"),
        State("ecat-model-cell-parameters-grid", "rowData"),
        State("ecat-model-species-parameters-grid", "rowData"),
        State("ecat-model-mechanism-parameters-grid", "rowData"),
        State("ecat-model-mechanism-source", "data"),
        State("ecat-model-mechanism-preset", "value"),
        State("ecat-model-mechanism-custom", "value"),
        State("ecat-objects-store", "data"),
        State("ecat-model-tabs", "value"),
        State("ecat-workflow-store", "data"),
        prevent_initial_call=True,
    )
    def run_model_simulation_callback(
        _clicks,
        simulate_mode,
        program_ei,
        program_e_low,
        program_e_high,
        program_ef,
        program_scan_rate,
        program_segments,
        program_points,
        program_quiet_time,
        program_plot_options,
        spatial_mode,
        spatial_nx,
        spatial_dx_fraction,
        spatial_viscosity,
        spatial_viscosity_source,
        spatial_rotation,
        cv_index,
        cv_trim_mode,
        cv_window_min,
        cv_window_max,
        cv_segments,
        cv_stride,
        cv_estimate_cdl,
        over_conditions,
        condition_axis,
        condition_range,
        condition_count,
        condition_species,
        cell_parameter_rows,
        species_parameter_rows,
        mechanism_parameter_rows,
        source,
        preset,
        custom_text,
        objects_state,
        active_tab,
        workflow_data,
    ):
        workflow = AppWorkflow.from_dict(workflow_data)
        model_options = dict(workflow.model_options or {})
        if not model_options:
            model_options = model_mechanism_options_from_controls(source, preset, custom_text)
        program_settings = model_program_settings_from_controls(
            program_ei,
            program_e_low,
            program_e_high,
            program_ef,
            program_scan_rate,
            program_segments,
            program_points,
            program_quiet_time,
            program_plot_options,
        )
        condition_settings = model_condition_settings_from_controls(
            condition_axis,
            condition_range,
            condition_count,
            condition_species,
        )
        cv_data_settings = model_cv_data_settings_from_controls(
            cv_index,
            cv_trim_mode,
            cv_window_min,
            cv_window_max,
            cv_segments,
            cv_stride,
            cv_estimate_cdl,
        )
        model_options = build_model_simulation_state(
            model_options,
            simulate_mode,
            program_settings,
            cv_data_settings,
            over_conditions,
            condition_settings,
        )
        setup_parameter_rows = model_setup_parameter_rows_from_controls(
            spatial_mode,
            spatial_nx,
            spatial_dx_fraction,
            spatial_viscosity,
            spatial_rotation,
            spatial_viscosity_source,
        )
        parameter_rows, cell_parameter_rows, setup_parameter_rows = model_split_rows_for_simulation(
            model_options,
            mechanism_parameter_rows,
            cell_parameter_rows,
            species_parameter_rows,
            setup_parameter_rows,
        )
        model_options["parameters"] = parameter_rows
        model_options["setup_parameters"] = setup_parameter_rows
        model_options["species_parameters"] = species_parameter_rows
        model_options["mechanism_parameters"] = mechanism_parameter_rows
        model_options["cell_parameters"] = cell_parameter_rows
        simulation_result = run_browser_simulate_cv(
            mode=simulate_mode,
            mechanism=model_options.get("mechanism") or model_options.get("mechanism_preset") or "E",
            parameter_rows=parameter_rows,
            cell_parameter_rows=cell_parameter_rows,
            setup_parameter_rows=setup_parameter_rows,
            program_settings=program_settings,
            cv_data_settings=cv_data_settings,
            objects=default_registry.get((objects_state or {}).get("dataset_id")),
            over_conditions=model_options.get("over_conditions"),
            condition_settings=condition_settings,
        )
        model_options = attach_model_simulation_result(model_options, simulation_result)
        workflow.model_options = model_options
        disabled, run_disabled, disabled_title = model_fit_gate(model_options)
        parameter_content, cell_content, result_content = model_results_content(model_options)
        fit_enabled, fit_control_style = model_fit_table_state(active_tab, model_options)
        cell_rows = model_cell_parameter_row_data(model_options)
        species_rows = model_species_parameter_row_data(model_options)
        mechanism_rows = model_mechanism_parameter_row_data(model_options)
        return (
            workflow.to_dict(),
            disabled,
            run_disabled,
            disabled_title,
            {},
            cell_rows,
            cell_content,
            species_rows,
            "",
            mechanism_rows,
            parameter_content,
            fit_control_style,
            fit_control_style,
            fit_control_style,
            model_bound_column_defs([], "cell", fit_enabled, cell_rows),
            model_bound_column_defs([], "species", fit_enabled, species_rows),
            model_bound_column_defs([], "mechanism", fit_enabled, mechanism_rows),
            result_content,
            "",
            scroll_signal("model-simulation"),
        )

    @app.callback(
        Output("ecat-workflow-store", "data", allow_duplicate=True),
        Output("ecat-model-results", "style", allow_duplicate=True),
        Output("ecat-model-cell-parameters-grid", "rowData", allow_duplicate=True),
        Output("ecat-model-cell-parameters-content", "children", allow_duplicate=True),
        Output("ecat-model-species-parameters-grid", "rowData", allow_duplicate=True),
        Output("ecat-model-species-parameters-content", "children", allow_duplicate=True),
        Output("ecat-model-mechanism-parameters-grid", "rowData", allow_duplicate=True),
        Output("ecat-model-mechanism-parameters-content", "children", allow_duplicate=True),
        Output("ecat-model-cell-parameters-grid", "columnDefs", allow_duplicate=True),
        Output("ecat-model-species-parameters-grid", "columnDefs", allow_duplicate=True),
        Output("ecat-model-mechanism-parameters-grid", "columnDefs", allow_duplicate=True),
        Output("ecat-model-results-content", "children", allow_duplicate=True),
        Output("ecat-scroll-target", "children", allow_duplicate=True),
        Input("ecat-model-run-fit", "n_clicks"),
        State("ecat-model-fit-mode", "value"),
        State("ecat-model-fit-cv-index", "value"),
        State("ecat-model-spatial-mode", "value"),
        State("ecat-model-spatial-nx", "value"),
        State("ecat-model-spatial-dx-fraction", "value"),
        State("ecat-model-spatial-viscosity", "value"),
        State("ecat-model-spatial-viscosity-source", "value"),
        State("ecat-model-spatial-rotation", "value"),
        State("ecat-model-cell-parameters-grid", "rowData"),
        State("ecat-model-species-parameters-grid", "rowData"),
        State("ecat-model-mechanism-parameters-grid", "rowData"),
        State("ecat-model-cell-bound-columns", "value"),
        State("ecat-model-species-bound-columns", "value"),
        State("ecat-model-bound-columns", "value"),
        State("ecat-objects-store", "data"),
        State("ecat-workflow-store", "data"),
        prevent_initial_call=True,
    )
    def run_model_fit_callback(
        _clicks,
        fit_mode,
        fit_cv_index,
        spatial_mode,
        spatial_nx,
        spatial_dx_fraction,
        spatial_viscosity,
        spatial_viscosity_source,
        spatial_rotation,
        cell_parameter_rows,
        species_parameter_rows,
        mechanism_parameter_rows,
        cell_bounds,
        species_bounds,
        mechanism_bounds,
        objects_state,
        workflow_data,
    ):
        workflow = AppWorkflow.from_dict(workflow_data)
        setup_parameter_rows = model_setup_parameter_rows_from_controls(
            spatial_mode,
            spatial_nx,
            spatial_dx_fraction,
            spatial_viscosity,
            spatial_rotation,
            spatial_viscosity_source,
        )
        parameter_rows, cell_parameter_rows, setup_parameter_rows = model_split_rows_for_simulation(
            workflow.model_options,
            mechanism_parameter_rows,
            cell_parameter_rows,
            species_parameter_rows,
            setup_parameter_rows,
        )
        model_options = dict(workflow.model_options or {})
        model_options["setup_parameters"] = setup_parameter_rows
        model_options["species_parameters"] = species_parameter_rows
        model_options["mechanism_parameters"] = mechanism_parameter_rows
        model_options = build_model_fit_state(
            model_options,
            fit_mode,
            parameter_rows,
            cell_parameter_rows,
            fit_cv_index,
            default_registry.get((objects_state or {}).get("dataset_id")),
        )
        if model_options.get("fit_requested") and str(model_options.get("fit_mode") or "single") == "single":
            fit_result = run_browser_fit_cv(
                fit_mode=fit_mode,
                fit_cv_index=fit_cv_index,
                mechanism=model_options.get("mechanism") or model_options.get("mechanism_preset") or "E",
                parameter_rows=parameter_rows,
                cell_parameter_rows=cell_parameter_rows,
                setup_parameter_rows=setup_parameter_rows,
                cv_data_settings=model_options.get("cv_data_settings") or {},
                objects=default_registry.get((objects_state or {}).get("dataset_id")),
            )
            fitted_parameters = list(fit_result.get("parameter_rows") or parameter_rows)
            model_options["fit_result"] = fit_result
            model_options["fit_spec"] = fit_result.get("fit") or {}
            model_options["fit_params"] = fit_result.get("params") or {}
            model_options["parameters"] = fitted_parameters
            model_options["species_parameters"] = [
                row for row in fitted_parameters if row.get("group") in {"concentration", "diffusion"}
            ]
            model_options["mechanism_parameters"] = [
                row for row in fitted_parameters if row.get("group") not in {"concentration", "diffusion"}
            ]
            model_options["cell_parameters"] = list(fit_result.get("cell_parameter_rows") or cell_parameter_rows)
        workflow.model_options = model_options
        parameter_content, cell_content, result_content = model_results_content(model_options)
        cell_rows = model_cell_parameter_row_data(model_options)
        species_rows = model_species_parameter_row_data(model_options)
        mechanism_rows = model_mechanism_parameter_row_data(model_options)
        return (
            workflow.to_dict(),
            {},
            cell_rows,
            cell_content,
            species_rows,
            "",
            mechanism_rows,
            parameter_content,
            model_bound_column_defs(cell_bounds, "cell", True, cell_rows),
            model_bound_column_defs(species_bounds, "species", True, species_rows),
            model_bound_column_defs(mechanism_bounds, "mechanism", True, mechanism_rows),
            result_content,
            scroll_signal("model-fit"),
        )

    @app.callback(
        Output("ecat-default-plot", "children", allow_duplicate=True),
        Output("ecat-scroll-target", "children", allow_duplicate=True),
        Input("ecat-replot", "n_clicks"),
        Input("ecat-plotting-replot", "n_clicks"),
        State("ecat-objects-store", "data"),
        State("ecat-selected-row-ids-store", "data"),
        State("ecat-object-table", "virtualRowData"),
        State("ecat-plot-options-store", "data"),
        prevent_initial_call=True,
    )
    def replot_callback(_clicks, _plotting_clicks, state, selected_row_ids, displayed_rows, plot_options):
        if not state:
            return "", scroll_signal("plot")
        row_ids = displayed_selected_row_ids(displayed_rows, selected_row_ids)
        result = handle_replot(state.get("dataset_id"), row_ids, plot_options=display_plot_options(plot_options))
        save_result = handle_replot(state.get("dataset_id"), row_ids, plot_options=plot_options)
        plot = _plot_frame(result.get("plot"), include_refresh=True, save_src=save_result.get("plot"))
        return plot, scroll_signal("plot")

    @app.callback(
        Output("ecat-download-plot", "data"),
        Input("ecat-save-plot", "n_clicks"),
        Input("ecat-plotting-save-plot", "n_clicks"),
        State("ecat-objects-store", "data"),
        State("ecat-selected-row-ids-store", "data"),
        State("ecat-object-table", "virtualRowData"),
        State("ecat-plot-options-store", "data"),
        prevent_initial_call=True,
    )
    def save_plot_callback(_clicks, _plotting_clicks, state, selected_row_ids, displayed_rows, plot_options):
        if not state:
            return None
        row_ids = displayed_selected_row_ids(displayed_rows, selected_row_ids)
        result = handle_replot(state.get("dataset_id"), row_ids, plot_options=plot_options)
        plot = result.get("plot")
        if not plot:
            return None
        _prefix, encoded = plot.split(",", 1)
        extension = str((plot_options or {}).get("_format") or "png").lower()
        return {"content": encoded, "filename": f"ecat_multiplot.{extension}", "base64": True}

    @app.callback(
        Output("ecat-about-panel", "hidden"),
        Output("ecat-about-button", "children"),
        Input("ecat-about-button", "n_clicks"),
        State("ecat-about-panel", "hidden"),
        prevent_initial_call=True,
    )
    def about_toggle_callback(clicks, hidden):
        return toggle_about_state(clicks, hidden)

    @app.callback(
        Output("ecat-app-shell", "className"),
        Input("ecat-sidebar-toggle", "n_clicks"),
        State("ecat-app-shell", "className"),
        prevent_initial_call=True,
    )
    def sidebar_toggle_callback(_clicks, class_name):
        return toggle_sidebar_class(class_name)

    @app.callback(
        Output("ecat-app-shell", "className", allow_duplicate=True),
        Input("ecat-left-tabs", "value"),
        State("ecat-app-shell", "className"),
        prevent_initial_call=True,
    )
    def sidebar_tab_expand_callback(_value, class_name):
        return expand_sidebar_class(class_name)

    @app.callback(
        Output("ecat-model-settings-card", "open"),
        Output("ecat-model-results-card", "open"),
        Input("ecat-left-tabs", "value"),
    )
    def model_main_cards_open_callback(active_tab):
        return model_main_cards_open_state(active_tab)

    @app.callback(
        Output("ecat-cv-analysis-card", "open"),
        Output("ecat-ca-analysis-card", "open"),
        Output("ecat-cp-analysis-card", "open"),
        Input("ecat-objects-store", "data"),
    )
    def analysis_card_open_callback(state):
        return analysis_card_open_state(state)

    @app.callback(
        Output("ecat-single-index", "value"),
        Output("ecat-single-index", "disabled"),
        Output("ecat-single-index-status", "children"),
        Output("ecat-ca-index", "value"),
        Output("ecat-ca-index", "disabled"),
        Output("ecat-ca-index-status", "children"),
        Output("ecat-cp-index", "value"),
        Output("ecat-cp-index", "disabled"),
        Output("ecat-cp-index-status", "children"),
        Input("ecat-objects-store", "data"),
    )
    def analysis_index_defaults_callback(state):
        values, disabled, messages = analysis_index_defaults(state)
        return (
            values["cv"],
            disabled["cv"],
            messages["cv"],
            values["ca"],
            disabled["ca"],
            messages["ca"],
            values["cp"],
            disabled["cp"],
            messages["cp"],
        )

    @app.callback(
        Output("ecat-single-segment-slider", "min"),
        Output("ecat-single-segment-slider", "max"),
        Output("ecat-single-segment-slider", "value"),
        Output("ecat-single-segment-slider", "marks"),
        Output("ecat-single-segment-slider", "disabled"),
        Output("ecat-single-segment-slider-wrap", "style"),
        Output("ecat-single-segment-text-wrap", "style"),
        Output("ecat-single-segment-text", "value"),
        Output("ecat-single-segment-status", "children"),
        Input("ecat-objects-store", "data"),
        Input("ecat-single-index", "value"),
    )
    def single_cv_segment_callback(state, selected_index):
        segment_state = single_cv_segment_control_state(state, selected_index)
        return (
            segment_state["slider_min"],
            segment_state["slider_max"],
            segment_state["slider_value"],
            segment_state["slider_marks"],
            segment_state["slider_disabled"],
            segment_state["slider_style"],
            segment_state["text_style"],
            segment_state["text_value"],
            segment_state["status"],
        )

    @app.callback(
        Output("ecat-analysis-results-store", "data"),
        Output("ecat-workflow-store", "data", allow_duplicate=True),
        Output("ecat-scroll-target", "children", allow_duplicate=True),
        Input("ecat-run-single", "n_clicks"),
        State("ecat-objects-store", "data"),
        State("ecat-analysis-results-store", "data"),
        State("ecat-workflow-store", "data"),
        State("ecat-single-index", "value"),
        State("ecat-single-analyses", "value"),
        State("ecat-single-guess-potential", "value"),
        State("ecat-single-tangent-potential", "value"),
        State("ecat-single-segment-slider", "value"),
        State("ecat-single-segment-text", "value"),
        State("ecat-single-dimensionless-normalize", "value"),
        State("ecat-single-dimensionless-mode", "value"),
        State("ecat-single-dimensionless-e0", "value"),
        State("ecat-single-dimensionless-n", "value"),
        State("ecat-single-dimensionless-temperature", "value"),
        State("ecat-single-dimensionless-d", "value"),
        State("ecat-single-dimensionless-c", "value"),
        State("ecat-single-dimensionless-area-mode", "value"),
        State("ecat-single-dimensionless-area", "value"),
        prevent_initial_call=True,
    )
    def single_cv_callback(
        _clicks,
        state,
        results_store,
        workflow_data,
        selected_index,
        analyses,
        guess_potential,
        tangent_potential,
        segment_slider,
        segment_text,
        dimensionless_enabled,
        dimensionless_mode,
        dimensionless_e0,
        dimensionless_n,
        dimensionless_temperature,
        dimensionless_d,
        dimensionless_c,
        dimensionless_area_mode,
        dimensionless_area,
    ):
        workflow = AppWorkflow.from_dict(workflow_data).to_dict()
        if not state:
            result = {"message": "No loaded objects.", "plot": None, "results": []}
            return upsert_analysis_result_store(results_store, "cv_single", result, "Single CV Analysis"), workflow, scroll_signal("analysis")
        cv_options = {}
        if guess_potential not in (None, ""):
            cv_options["guess potential"] = float(guess_potential)
        if tangent_potential not in (None, ""):
            cv_options["tangent potential"] = float(tangent_potential)
        cv_options.update(single_cv_options_from_controls(segment_slider, segment_text))
        cv_options.update(
            single_cv_dimensionless_normalization_from_controls(
                dimensionless_enabled,
                dimensionless_mode,
                dimensionless_e0,
                dimensionless_n,
                dimensionless_temperature,
                dimensionless_d,
                dimensionless_c,
                dimensionless_area_mode,
                dimensionless_area,
            )
        )
        workflow = update_workflow_single_analysis(workflow_data, selected_index, analyses, cv_options)
        result = handle_single_cv(state.get("dataset_id"), selected_index, analyses, cv_options)
        return upsert_analysis_result_store(results_store, "cv_single", result, "Single CV Analysis"), workflow, scroll_signal("analysis")

    @app.callback(
        Output("ecat-single-dimensionless-options", "style"),
        Input("ecat-single-dimensionless-normalize", "value"),
    )
    def single_dimensionless_visibility_callback(enabled_values):
        return single_cv_dimensionless_visibility(enabled_values)

    @app.callback(
        Output("ecat-analysis-results-store", "data", allow_duplicate=True),
        Output("ecat-workflow-store", "data", allow_duplicate=True),
        Output("ecat-scroll-target", "children", allow_duplicate=True),
        Input("ecat-run-ca", "n_clicks"),
        State("ecat-objects-store", "data"),
        State("ecat-analysis-results-store", "data"),
        State("ecat-workflow-store", "data"),
        State("ecat-ca-index", "value"),
        State("ecat-ca-analyses", "value"),
        State("ecat-ca-target-charge", "value"),
        State("ecat-ca-target-options", "value"),
        State("ecat-ca-baseline-tail-fraction", "value"),
        State("ecat-plot-options-store", "data"),
        prevent_initial_call=True,
    )
    def ca_callback(
        _clicks,
        state,
        results_store,
        workflow_data,
        selected_index,
        analyses,
        target_charge,
        target_options,
        baseline_tail_fraction,
        plot_options,
    ):
        workflow = AppWorkflow.from_dict(workflow_data).to_dict()
        if not state:
            result = {"message": "No loaded objects.", "plot": None, "results": []}
            return upsert_analysis_result_store(results_store, "ca", result, "CA Analysis"), workflow, scroll_signal("analysis")
        ca_options = {
            "target charge": 0.75 if target_charge in (None, "") else float(target_charge),
            "plot ca": "plot_ca" in (target_options or []),
            "baseline tail fraction": 0.05 if baseline_tail_fraction in (None, "") else float(baseline_tail_fraction),
            "plot options": analysis_plot_options(plot_options),
        }
        workflow = update_workflow_chrono_analysis(workflow_data, "ca", selected_index, analyses, ca_options)
        result = handle_single_object_analysis(
            state.get("dataset_id"),
            selected_index,
            analyses,
            "ca",
            lambda obj, selected_analyses: run_ca_analysis(obj, selected_analyses, ca_options),
        )
        return upsert_analysis_result_store(results_store, "ca", result, "CA Analysis"), workflow, scroll_signal("analysis")

    @app.callback(
        Output("ecat-analysis-results-store", "data", allow_duplicate=True),
        Output("ecat-workflow-store", "data", allow_duplicate=True),
        Output("ecat-scroll-target", "children", allow_duplicate=True),
        Input("ecat-run-cp", "n_clicks"),
        State("ecat-objects-store", "data"),
        State("ecat-analysis-results-store", "data"),
        State("ecat-workflow-store", "data"),
        State("ecat-cp-index", "value"),
        State("ecat-cp-analyses", "value"),
        State("ecat-cp-percent-capacity", "value"),
        State("ecat-cp-capacity-mode", "value"),
        State("ecat-cp-efficiency-mode", "value"),
        State("ecat-cp-cycles-start", "value"),
        State("ecat-cp-cycles-end", "value"),
        State("ecat-cp-cycles-step", "value"),
        State("ecat-cp-cycle-segment", "value"),
        State("ecat-cp-cycle-x-axis", "value"),
        State("ecat-plot-options-store", "data"),
        prevent_initial_call=True,
    )
    def cp_callback(
        _clicks,
        state,
        results_store,
        workflow_data,
        selected_index,
        analyses,
        percent_capacity_values,
        capacity_mode,
        efficiency_mode,
        cycles_start,
        cycles_end,
        cycles_step,
        cycle_segment,
        cycle_x_axis,
        plot_options,
    ):
        workflow = AppWorkflow.from_dict(workflow_data).to_dict()
        if not state:
            result = {"message": "No loaded objects.", "plot": None, "results": []}
            return upsert_analysis_result_store(results_store, "cp", result, "CP Analysis"), workflow, scroll_signal("analysis")
        cp_options = {
            "percent capacity": "percent_capacity" in (percent_capacity_values or []),
            "capacity mode": capacity_mode or "both",
            "efficiency mode": efficiency_mode or "both",
            "cycles": (
                int(cycles_start or 1),
                int(cycles_end or 100),
                int(cycles_step or 10),
            ),
            "segment": cycle_segment or "both",
            "x axis": cycle_x_axis or "capacity",
            "plot options": analysis_plot_options(plot_options),
        }
        workflow = update_workflow_chrono_analysis(workflow_data, "cp", selected_index, analyses, cp_options)
        result = handle_single_object_analysis(
            state.get("dataset_id"),
            selected_index,
            analyses,
            "cp",
            lambda obj, selected_analyses: run_cp_analysis(obj, selected_analyses, cp_options),
        )
        return upsert_analysis_result_store(results_store, "cp", result, "CP Analysis"), workflow, scroll_signal("analysis")

    @app.callback(
        Output("ecat-analysis-results-store", "data", allow_duplicate=True),
        Output("ecat-workflow-store", "data", allow_duplicate=True),
        Output("ecat-scroll-target", "children", allow_duplicate=True),
        Input("ecat-run-multi-analysis", "n_clicks"),
        State("ecat-objects-store", "data"),
        State("ecat-analysis-results-store", "data"),
        State("ecat-workflow-store", "data"),
        State("ecat-selected-row-ids-store", "data"),
        State("ecat-object-table", "virtualRowData"),
        State("ecat-multi-analysis", "value"),
        State("ecat-multi-segment", "value"),
        State("ecat-multi-segments", "value"),
        State("ecat-multi-guess-potential", "value"),
        State("ecat-sevcik-mode", "value"),
        State("ecat-multi-x-axis", "value"),
        State("ecat-multi-fit-model", "value"),
        State("ecat-multi-toggles", "value"),
        State("ecat-fowa-reference-index", "value"),
        State("ecat-fowa-redox-mode", "value"),
        State("ecat-fowa-redox-potential", "value"),
        State("ecat-fowa-fit-basis", "value"),
        State("ecat-fowa-fit-range-start", "value"),
        State("ecat-fowa-fit-range-end", "value"),
        State("ecat-fowa-diagnostic-y-axis", "value"),
        State("ecat-fowa-min-fit-points", "value"),
        State("ecat-fowa-min-r2", "value"),
        State("ecat-tafel-index", "value"),
        State("ecat-tafel-tof-max", "value"),
        State("ecat-tafel-thermo-potential", "value"),
        State("ecat-tafel-redox-potential", "value"),
        State("ecat-tafel-overpotential-start", "value"),
        State("ecat-tafel-overpotential-end", "value"),
        State("ecat-tafel-color", "value"),
        State("ecat-multi-preprocess-scale", "value"),
        State("ecat-multi-scale-type", "value"),
        State("ecat-multi-scale-factor", "value"),
        State("ecat-multi-scale-reference-index", "value"),
        State("ecat-multi-scale-reference-mode", "value"),
        State("ecat-multi-scale-segment", "value"),
        State("ecat-multi-scale-guess-potential", "value"),
        State("ecat-multi-normalize-mode", "value"),
        State("ecat-multi-dimensionless-mode", "value"),
        State("ecat-multi-dimensionless-e0", "value"),
        State("ecat-multi-dimensionless-n", "value"),
        State("ecat-multi-dimensionless-temperature", "value"),
        State("ecat-multi-dimensionless-d", "value"),
        State("ecat-multi-dimensionless-c", "value"),
        State("ecat-multi-dimensionless-area-mode", "value"),
        State("ecat-multi-dimensionless-area", "value"),
        State("ecat-multi-current-normalization-type", "value"),
        State("ecat-multi-current-reference-index", "value"),
        State("ecat-multi-current-segment", "value"),
        State("ecat-multi-current-guess-potential", "value"),
        State("ecat-multi-current-ip0", "value"),
        prevent_initial_call=True,
    )
    def multi_cv_callback(
        _clicks,
        state,
        results_store,
        workflow_data,
        selected_row_ids,
        displayed_rows,
        analysis,
        segment,
        segments,
        guess_potential,
        sevcik_mode,
        x_axis,
        fit_model,
        toggles,
        fowa_reference_index,
        fowa_redox_mode,
        fowa_redox_potential,
        fowa_fit_basis,
        fowa_fit_range_start,
        fowa_fit_range_end,
        fowa_diagnostic_y_axis,
        fowa_min_fit_points,
        fowa_min_r2,
        tafel_index,
        tafel_tof_max,
        tafel_thermo_potential,
        tafel_redox_potential,
        tafel_overpotential_start,
        tafel_overpotential_end,
        tafel_color,
        preprocess_scale,
        scale_type,
        scale_factor,
        scale_reference_index,
        scale_reference_mode,
        scale_segment,
        scale_guess_potential,
        normalize_mode,
        dimensionless_mode,
        dimensionless_e0,
        dimensionless_n,
        dimensionless_temperature,
        dimensionless_d,
        dimensionless_c,
        dimensionless_area_mode,
        dimensionless_area,
        current_type,
        current_reference_index,
        current_segment,
        current_guess_potential,
        current_ip0,
    ):
        if not analysis:
            result = {"message": "Select a CV analysis.", "plot": None, "results": []}
            return (
                upsert_analysis_result_store(results_store, "cv_multi", result, "Multiple CV Analysis"),
                AppWorkflow.from_dict(workflow_data).to_dict(),
                scroll_signal("analysis"),
            )
        options = multi_cv_options_from_controls(
            analysis,
            segment,
            segments,
            guess_potential,
            sevcik_mode,
            x_axis,
            fit_model,
            toggles,
            fowa_reference_index,
            fowa_redox_mode,
            fowa_redox_potential,
            fowa_fit_basis,
            fowa_fit_range_start,
            fowa_fit_range_end,
            fowa_diagnostic_y_axis,
            fowa_min_fit_points,
            fowa_min_r2,
            tafel_index,
            tafel_tof_max,
            tafel_thermo_potential,
            tafel_redox_potential,
            tafel_overpotential_start,
            tafel_overpotential_end,
            tafel_color,
        )
        preprocessing = multi_cv_preprocessing_from_controls(
            preprocess_scale,
            scale_type,
            scale_factor,
            scale_reference_index,
            scale_reference_mode,
            scale_segment,
            scale_guess_potential,
            normalize_mode,
            dimensionless_mode,
            dimensionless_e0,
            dimensionless_n,
            dimensionless_temperature,
            dimensionless_d,
            dimensionless_c,
            dimensionless_area_mode,
            dimensionless_area,
            current_type,
            current_reference_index,
            current_segment,
            current_guess_potential,
            current_ip0,
        )
        if preprocessing:
            options["preprocessing"] = preprocessing
        workflow = update_workflow_multi_analysis(workflow_data, analysis, options)
        if not state:
            result = {"message": "No loaded objects.", "plot": None, "results": []}
            return upsert_analysis_result_store(results_store, "cv_multi", result, "Multiple CV Analysis"), workflow, scroll_signal("analysis")
        result = handle_multi_cv_analysis(
            state.get("dataset_id"),
            selected_row_ids,
            displayed_rows,
            analysis,
            options,
            preprocessing,
        )
        normalized = {
            "message": result.get("message", ""),
            "plot": result.get("plot"),
            "plots": result.get("plots") or [],
            "results": [
                {
                    "analysis": result.get("analysis"),
                    "status": result.get("status"),
                    "value": result.get("value"),
                    "message": result.get("message", ""),
                }
            ],
        }
        return upsert_analysis_result_store(results_store, "cv_multi", normalized, "Multiple CV Analysis"), workflow, scroll_signal("analysis")

    @app.callback(
        Output("ecat-analysis-results-content", "children", allow_duplicate=True),
        Input("ecat-analysis-results-store", "data"),
        prevent_initial_call=True,
    )
    def render_analysis_results_callback(results_store):
        return render_analysis_results_store(results_store)

    @app.callback(
        Output("ecat-multi-analysis-equations", "children"),
        Input("ecat-multi-analysis", "value"),
        Input("ecat-sevcik-mode", "value"),
    )
    def multi_analysis_equations_callback(analysis, sevcik_mode):
        return multi_analysis_equation_content(analysis, sevcik_mode)

    @app.callback(
        Output("ecat-multi-analysis-options", "style"),
        Output("ecat-multi-analysis-title", "children"),
        Output("ecat-multi-segments", "value"),
        Output("ecat-sevcik-options", "style"),
        Output("ecat-fowa-options", "style"),
        Output("ecat-tafel-options", "style"),
        Output("ecat-multi-fit-wrap", "style"),
        Output("ecat-multi-guess-wrap", "style"),
        Output("ecat-multi-toggles", "options"),
        Output("ecat-multi-toggles", "value"),
        Input("ecat-multi-analysis", "value"),
    )
    def multi_analysis_options_callback(analysis):
        return multi_analysis_option_state(analysis)

    @app.callback(
        Output("ecat-multi-scale-options", "style"),
        Output("ecat-multi-scale-reference-fields", "style"),
        Output("ecat-multi-scale-manual-fields", "style"),
        Output("ecat-multi-dimensionless-options", "style"),
        Output("ecat-multi-current-normalization-options", "style"),
        Output("ecat-multi-current-reference-fields", "style"),
        Output("ecat-multi-current-manual-fields", "style"),
        Input("ecat-multi-preprocess-scale", "value"),
        Input("ecat-multi-normalize-mode", "value"),
        Input("ecat-multi-scale-type", "value"),
        Input("ecat-multi-current-normalization-type", "value"),
    )
    def multi_preprocessing_visibility_callback(scale_values, normalize_mode, scale_type, current_type):
        return multi_cv_preprocessing_visibility(scale_values, normalize_mode, scale_type, current_type)

    @app.callback(
        Output("ecat-analysis-results-store", "data", allow_duplicate=True),
        Output("ecat-workflow-store", "data", allow_duplicate=True),
        Input("ecat-reset-analysis-results", "n_clicks"),
        State("ecat-workflow-store", "data"),
        prevent_initial_call=True,
    )
    def reset_analysis_results_callback(_clicks, workflow_data):
        return {}, clear_workflow_analysis_actions(workflow_data)

    @app.callback(
        Output("ecat-code-preview", "value"),
        Input("ecat-workflow-store", "data"),
        Input("ecat-plot-options-store", "data"),
    )
    def update_code_preview(workflow_data, plot_options):
        workflow = AppWorkflow.from_dict(workflow_data)
        workflow.plot_options = dict(plot_options or workflow.plot_options or {})
        return generate_python(workflow)

    @app.callback(
        Output("ecat-download-code", "data"),
        Input("ecat-download-code-button", "n_clicks"),
        State("ecat-code-preview", "value"),
        State("ecat-analysis-results-store", "data"),
        prevent_initial_call=True,
    )
    def download_code_callback(_clicks, code, results_store):
        return {
            "content": generate_notebook(code or "", results_store),
            "filename": "ecat_workflow.ipynb",
            "type": "application/x-ipynb+json",
        }

    return app
