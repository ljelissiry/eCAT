"""Dash callback registration for the eCAT browser app."""

from .adapters import (
    default_included_row_ids,
    filter_and_group,
    load_local_path,
    load_uploaded_files,
    objects_for_analysis,
    reload_workflow,
    run_ca_analysis,
    run_cp_analysis,
    run_multi_cv_analysis,
    run_single_cv_analysis,
    summarize_objects,
)
from .codegen import generate_python
from .defaults import default_workflow, example_folder_path
from .execution import run_user_code
from .figures import render_multiplot, render_object_plot
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
from .workflow import BrowserWorkflow


def empty_state():
    workflow = BrowserWorkflow()
    return {
        "objects": [],
        "warnings": [],
        "workflow": workflow.to_dict(),
        "conditions": [],
        "code": generate_python(workflow),
    }


def _state_from_load_result(result, registry=default_registry) -> dict[str, object]:
    dataset_id = registry.put(result.objects, result.warnings)
    snapshot = registry.snapshot(dataset_id)
    workflow = result.workflow
    included_row_ids = workflow.included_row_ids or default_included_row_ids(result.objects, workflow)
    workflow.included_row_ids = included_row_ids
    plot = render_default_multiplot(result.objects, workflow)
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
    workflow = BrowserWorkflow.from_dict(workflow_data)
    workflow.reference_settings = dict(reference_settings or {})
    workflow.import_options = build_reference_options(workflow.reference_settings)
    return _state_from_load_result(reload_workflow(workflow), registry)


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
    workflow = BrowserWorkflow.from_dict(workflow_data)
    workflow.included_row_ids = list(included_row_ids or [])
    return workflow.to_dict()


def update_workflow_plot_options(workflow_data, plot_options) -> dict[str, object]:
    workflow = BrowserWorkflow.from_dict(workflow_data)
    workflow.plot_options = dict(plot_options or {})
    return workflow.to_dict()


def update_workflow_single_analysis(workflow_data, selected_index, analyses) -> dict[str, object]:
    workflow = BrowserWorkflow.from_dict(workflow_data)
    workflow.selected_index = None if selected_index is None else int(selected_index)
    workflow.analyses = list(analyses or [])
    return workflow.to_dict()


def update_workflow_multi_analysis(workflow_data, analysis, options) -> dict[str, object]:
    workflow = BrowserWorkflow.from_dict(workflow_data)
    workflow.analyses = list(dict.fromkeys([*workflow.analyses, analysis]))
    workflow.plot_options = {**workflow.plot_options, f"{analysis} options": dict(options or {})}
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


def handle_extra_columns(dataset_id, visible_columns, registry=default_registry) -> dict[str, object]:
    return build_browser_table(registry.get(dataset_id), visible_columns=visible_columns or [])


def handle_reset_columns(dataset_id, registry=default_registry) -> dict[str, object]:
    return reset_column_selection(registry.get(dataset_id))


def render_default_multiplot(objects, workflow=None, plot_options=None) -> str | None:
    workflow = workflow or BrowserWorkflow()
    ordered = objects_for_analysis(objects, workflow)
    cv_objects = [obj for obj in ordered if type(obj).__name__ == "cv"]
    if not cv_objects:
        return None
    try:
        return render_multiplot(cv_objects, plot_options)
    except Exception:
        return None


def handle_replot(dataset_id, selected_row_ids=None, registry=default_registry, plot_options=None) -> dict[str, object]:
    objects = registry.get_by_row_ids(dataset_id, selected_row_ids)
    plot = render_default_multiplot(objects, BrowserWorkflow(), plot_options=plot_options)
    return {"plot": plot}


def display_plot_options(plot_options):
    options = dict(plot_options or {})
    options["_format"] = "png"
    options["_dpi"] = 150
    return options


def analysis_plot_options(plot_options):
    options = dict(plot_options or {})
    options.pop("_format", None)
    options.pop("_dpi", None)
    return options


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
    workflow = BrowserWorkflow.from_dict(workflow_data)
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
    if not analysis:
        return {"display": "none"}, "", "", {"display": "none"}, {"display": "none"}, {}, {}, [], []
    labels = {
        "fit_peak_potential": "Fit Peak Potential Options",
        "fit_peak_current": "Fit Peak Current Options",
        "sevcik_analysis": "Sevcik Analysis Options",
        "trumpet_analysis": "Trumpet Analysis Options",
        "fowa": "FOWA Options",
        "tafel_analysis": "Tafel Analysis Options",
    }
    default_segments = "1, 2" if analysis == "trumpet_analysis" else ""
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
        fowa_style,
        tafel_style,
        fit_style,
        guess_style,
        toggle_options,
        toggle_values,
    )


def handle_multi_cv_analysis(dataset_id, selected_row_ids, displayed_rows, analysis, options, registry=default_registry) -> dict[str, object]:
    row_ids = displayed_selected_row_ids(displayed_rows, selected_row_ids)
    objects = registry.get_by_row_ids(dataset_id, row_ids)
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
    entries = [
        html.Div(
            className="ecat-analysis-result-entry",
            children=[
                html.Div(row.get("analysis"), className="ecat-analysis-result-name"),
                html.Div(
                    [
                        html.Span(str(row.get("status") or ""), className="ecat-analysis-result-status"),
                        _analysis_value_node(row.get("value")),
                        html.Span(str(row.get("message") or ""), className="ecat-analysis-result-message"),
                    ],
                    className="ecat-analysis-result-detail",
                ),
            ],
        )
        for row in rows
    ]
    plots = result.get("plots") or []
    if plots:
        plot = html.Div(
            [
                html.Div(
                    [
                        html.Div(plot_item.get("label", "Plot"), className="ecat-analysis-plot-label"),
                        html.Img(src=plot_item["src"], className="ecat-plot"),
                    ],
                    className="ecat-analysis-plot-block",
                )
                for plot_item in plots
                if plot_item.get("src")
            ],
            className="ecat-analysis-plot-list",
        )
    else:
        plot = html.Img(src=result["plot"], className="ecat-plot") if result.get("plot") else ""
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
    technique_order = ["cv", "ca", "cp"]
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
    plot = html.Img(src=result["plot"], className="ecat-plot") if result.get("plot") else ""
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
    convention="US",
    custom_title=None,
    display_values=None,
    gradient_values=None,
    colorbar_values=None,
    label_values=None,
    trim_values=None,
    trim_min=None,
    trim_max=None,
    offset=None,
    scale_bar_height=None,
    scale_bar_location="upper left",
    offset_axis_values=None,
    save_format="png",
    dpi=150,
    plot_style="line",
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
        "plot style": plot_style or "line",
        "plot convention": convention or "US",
        "grid": "grid" in display_values,
        "deduplicate labels": "deduplicate" in label_values,
        "_format": save_format or "png",
        "_dpi": int(dpi or 150),
    }
    options["color mode"] = "auto" if allow_gradients else "discrete"
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
    return options


def plot_control_visibility(legend_values, gradient_values=None, title_mode="auto") -> dict[str, dict[str, str]]:
    legend_visible = "legend" in (legend_values or [])
    gradients_visible = legend_visible
    colorbar_visible = legend_visible and "gradients" in (gradient_values or [])
    return {
        "legend_options": {} if legend_visible else {"display": "none"},
        "colorbar": {} if colorbar_visible else {"display": "none"},
        "custom_title": {} if str(title_mode or "auto").lower() == "manual" else {"display": "none"},
    }


def trim_bounds_visibility(trim_values) -> dict[str, str]:
    return {} if "trim" in (trim_values or []) else {"display": "none"}


def offset_controls_visibility(offset) -> dict[str, str]:
    try:
        offset_value = float(offset)
    except (TypeError, ValueError):
        return {"display": "none"}
    return {} if offset_value != 0 else {"display": "none"}


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
            "The eCAT browser app requires Dash. Install with `pip install -e .[app]`."
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
        plot = html.Img(src=state["plot"], className="ecat-plot") if state.get("plot") else ""
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
            selected_grid_rows_for_ids(table["data"], state["included_row_ids"]),
            state["included_row_ids"],
            plot,
            file_options,
            state.get("status", ""),
            available_column_options(default_registry.get(state.get("dataset_id"))),
            selected_column_values(table),
        )

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
        Input("ecat-plot-display-options", "value"),
        Input("ecat-plot-gradients", "value"),
        Input("ecat-plot-colorbar", "value"),
        Input("ecat-plot-label-options", "value"),
        Input("ecat-plot-trim-enabled", "value"),
        Input("ecat-plot-trim-min", "value"),
        Input("ecat-plot-trim-max", "value"),
        Input("ecat-plot-offset", "value"),
        Input("ecat-plot-scale-bar-height", "value"),
        Input("ecat-plot-scale-bar-location", "value"),
        Input("ecat-plot-offset-axis-options", "value"),
        Input("ecat-plot-format", "value"),
        Input("ecat-plot-dpi", "value"),
    )
    def update_plot_options_store(
        legend,
        title_values,
        convention,
        custom_title,
        plot_style,
        display_values,
        gradient_values,
        colorbar_values,
        label_values,
        trim_values,
        trim_min,
        trim_max,
        offset,
        scale_bar_height,
        scale_bar_location,
        offset_axis_values,
        save_format,
        dpi,
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
            trim_values,
            trim_min,
            trim_max,
            offset,
            scale_bar_height,
            scale_bar_location,
            offset_axis_values,
            save_format,
            dpi,
            plot_style=plot_style,
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
        Output("ecat-plot-trim-bounds", "style"),
        Output("ecat-plot-offset-controls", "style"),
        Input("ecat-plot-legend", "value"),
        Input("ecat-plot-gradients", "value"),
        Input("ecat-plot-title-mode", "value"),
        Input("ecat-plot-trim-enabled", "value"),
        Input("ecat-plot-offset", "value"),
    )
    def update_plot_control_visibility(legend_values, gradient_values, title_mode, trim_values, offset):
        visible = plot_control_visibility(legend_values, gradient_values, title_mode)
        return (
            visible["legend_options"],
            visible["colorbar"],
            visible["custom_title"],
            trim_bounds_visibility(trim_values),
            offset_controls_visibility(offset),
        )

    @app.callback(
        Output("ecat-default-plot", "children", allow_duplicate=True),
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
            return ""
        row_ids = displayed_selected_row_ids(displayed_rows, selected_row_ids)
        result = handle_replot(state.get("dataset_id"), row_ids, plot_options=display_plot_options(plot_options))
        return html.Img(src=result["plot"], className="ecat-plot") if result.get("plot") else ""

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
        Output("ecat-analysis-results-store", "data"),
        Output("ecat-workflow-store", "data", allow_duplicate=True),
        Input("ecat-run-single", "n_clicks"),
        State("ecat-objects-store", "data"),
        State("ecat-analysis-results-store", "data"),
        State("ecat-workflow-store", "data"),
        State("ecat-single-index", "value"),
        State("ecat-single-analyses", "value"),
        State("ecat-single-guess-potential", "value"),
        State("ecat-single-tangent-potential", "value"),
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
    ):
        workflow = update_workflow_single_analysis(workflow_data, selected_index, analyses)
        if not state:
            result = {"message": "No loaded objects.", "plot": None, "results": []}
            return upsert_analysis_result_store(results_store, "cv", result, "CV Analysis"), workflow
        cv_options = {}
        if guess_potential not in (None, ""):
            cv_options["guess potential"] = float(guess_potential)
        if tangent_potential not in (None, ""):
            cv_options["tangent potential"] = float(tangent_potential)
        result = handle_single_cv(state.get("dataset_id"), selected_index, analyses, cv_options)
        return upsert_analysis_result_store(results_store, "cv", result, "CV Analysis"), workflow

    @app.callback(
        Output("ecat-analysis-results-store", "data", allow_duplicate=True),
        Input("ecat-run-ca", "n_clicks"),
        State("ecat-objects-store", "data"),
        State("ecat-analysis-results-store", "data"),
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
        selected_index,
        analyses,
        target_charge,
        target_options,
        baseline_tail_fraction,
        plot_options,
    ):
        if not state:
            result = {"message": "No loaded objects.", "plot": None, "results": []}
            return upsert_analysis_result_store(results_store, "ca", result, "CA Analysis")
        ca_options = {
            "target charge": 0.75 if target_charge in (None, "") else float(target_charge),
            "plot ca": "plot_ca" in (target_options or []),
            "baseline tail fraction": 0.05 if baseline_tail_fraction in (None, "") else float(baseline_tail_fraction),
            "plot options": analysis_plot_options(plot_options),
        }
        result = handle_single_object_analysis(
            state.get("dataset_id"),
            selected_index,
            analyses,
            "ca",
            lambda obj, selected_analyses: run_ca_analysis(obj, selected_analyses, ca_options),
        )
        return upsert_analysis_result_store(results_store, "ca", result, "CA Analysis")

    @app.callback(
        Output("ecat-analysis-results-store", "data", allow_duplicate=True),
        Input("ecat-run-cp", "n_clicks"),
        State("ecat-objects-store", "data"),
        State("ecat-analysis-results-store", "data"),
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
        if not state:
            result = {"message": "No loaded objects.", "plot": None, "results": []}
            return upsert_analysis_result_store(results_store, "cp", result, "CP Analysis")
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
        result = handle_single_object_analysis(
            state.get("dataset_id"),
            selected_index,
            analyses,
            "cp",
            lambda obj, selected_analyses: run_cp_analysis(obj, selected_analyses, cp_options),
        )
        return upsert_analysis_result_store(results_store, "cp", result, "CP Analysis")

    @app.callback(
        Output("ecat-analysis-results-store", "data", allow_duplicate=True),
        Output("ecat-workflow-store", "data", allow_duplicate=True),
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
    ):
        if not analysis:
            result = {"message": "Select a CV analysis.", "plot": None, "results": []}
            return upsert_analysis_result_store(results_store, "cv", result, "CV Analysis"), BrowserWorkflow.from_dict(workflow_data).to_dict()
        options = multi_cv_options_from_controls(
            analysis,
            segment,
            segments,
            guess_potential,
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
        workflow = update_workflow_multi_analysis(workflow_data, analysis, options)
        if not state:
            result = {"message": "No loaded objects.", "plot": None, "results": []}
            return upsert_analysis_result_store(results_store, "cv", result, "CV Analysis"), workflow
        result = handle_multi_cv_analysis(
            state.get("dataset_id"),
            selected_row_ids,
            displayed_rows,
            analysis,
            options,
        )
        normalized = {
            "message": result.get("message", ""),
            "plot": result.get("plot"),
            "results": [
                {
                    "analysis": result.get("analysis"),
                    "status": result.get("status"),
                    "value": result.get("value"),
                    "message": result.get("message", ""),
                }
            ],
        }
        return upsert_analysis_result_store(results_store, "cv", normalized, "CV Analysis"), workflow

    @app.callback(
        Output("ecat-analysis-results-content", "children", allow_duplicate=True),
        Input("ecat-analysis-results-store", "data"),
        prevent_initial_call=True,
    )
    def render_analysis_results_callback(results_store):
        return render_analysis_results_store(results_store)

    @app.callback(
        Output("ecat-multi-analysis-options", "style"),
        Output("ecat-multi-analysis-title", "children"),
        Output("ecat-multi-segments", "value"),
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
        Output("ecat-analysis-results-store", "data", allow_duplicate=True),
        Input("ecat-reset-analysis-results", "n_clicks"),
        prevent_initial_call=True,
    )
    def reset_analysis_results_callback(_clicks):
        return {}

    @app.callback(
        Output("ecat-code-preview", "value"),
        Input("ecat-workflow-store", "data"),
        Input("ecat-plot-options-store", "data"),
    )
    def update_code_preview(workflow_data, plot_options):
        workflow = BrowserWorkflow.from_dict(workflow_data)
        workflow.plot_options = dict(plot_options or workflow.plot_options or {})
        return generate_python(workflow)

    @app.callback(
        Output("ecat-download-code", "data"),
        Input("ecat-download-code-button", "n_clicks"),
        State("ecat-code-preview", "value"),
        prevent_initial_call=True,
    )
    def download_code_callback(_clicks, code):
        return {"content": code or "", "filename": "ecat_workflow.py"}

    return app
