"""Dash layout for the eCAT app."""

import ecat as e

from .adapters import SIMULATION_INSTALL_MESSAGE, simulation_backend_available, validate_simulation_mechanism
from .config import AppConfig
from .table import ag_grid_column_defs, selected_column_values, selected_grid_rows_for_ids

TAB_IDS = ("data", "plot", "analyze", "model")


DEFAULT_MODEL_PARAMETER_ROWS = [
    {"key": "E0", "path": "kinetics.0.E0", "step": "E(1)", "name": "E⁰ (V)", "group": "kinetics", "species": "A/B", "initial": -0.5, "unit": "V", "lower": "", "upper": "", "vary": True},
    {"key": "k0", "path": "kinetics.0.k0", "step": "E(1)", "name": "k₀ (m s⁻¹)", "group": "kinetics", "species": "A/B", "initial": 1e-3, "unit": "m s^-1", "lower": 1e-30, "upper": "", "vary": True},
    {"key": "alpha", "path": "kinetics.0.alpha", "step": "E(1)", "name": "α", "group": "kinetics", "species": "A/B", "initial": 0.5, "unit": "", "lower": 0.0, "upper": 1.0, "vary": True},
    {"key": "D_A", "path": "diffusion.A", "phase": "bulk", "name": "D (cm² s⁻¹)", "group": "diffusion", "species": "A", "initial": 1e-5, "unit": "cm^2 s^-1", "lower": 0.0, "upper": "", "vary": False},
    {"key": "D_B", "path": "diffusion.B", "phase": "bulk", "name": "D (cm² s⁻¹)", "group": "diffusion", "species": "B", "initial": 1e-5, "unit": "cm^2 s^-1", "lower": 0.0, "upper": "", "vary": False},
    {"key": "C_A", "path": "concentrations.bulk.A", "phase": "bulk", "name": "C (M)", "group": "concentration", "species": "A", "initial": 1e-3, "unit": "M", "lower": 0.0, "upper": "", "vary": False},
    {"key": "C_B", "path": "concentrations.bulk.B", "phase": "bulk", "name": "C (M)", "group": "concentration", "species": "B", "initial": 0.0, "unit": "M", "lower": 0.0, "upper": "", "vary": False},
]

DEFAULT_MODEL_SETUP_PARAMETER_ROWS = [
    {"key": "spatial", "path": "spatial", "name": "Spatial grid", "initial": "fast", "unit": "", "lower": "", "upper": "", "vary": False},
]

DEFAULT_MODEL_SPECIES_PARAMETER_ROWS = [
    row for row in DEFAULT_MODEL_PARAMETER_ROWS if row.get("group") in {"concentration", "diffusion"}
]

DEFAULT_MODEL_MECHANISM_PARAMETER_ROWS = [
    row for row in DEFAULT_MODEL_PARAMETER_ROWS if row.get("group") not in {"concentration", "diffusion"}
]

DEFAULT_MODEL_CELL_PARAMETER_ROWS = [
    {"key": "T", "name": "T (K)", "initial": 298.15, "unit": "K", "lower": 0.0, "upper": "", "vary": False},
    {"key": "Ru", "name": "Rᵤ (Ω)", "initial": 0.0, "unit": "ohm", "lower": 0.0, "upper": "", "vary": False},
    {"key": "Cdl", "name": "Cdl (F)", "initial": "auto", "unit": "F", "lower": 0.0, "upper": "", "vary": False},
    {"key": "A", "name": "A (m²)", "initial": 1e-5, "unit": "m^2", "lower": 0.0, "upper": "", "vary": False},
]


def _dash():
    try:
        import dash_ag_grid as dag
        from dash import dcc, html
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "The eCAT app requires optional app dependencies. Install them with "
            '`python -m pip install "ecat[app]"`. For a source checkout, use '
            '`python -m pip install -e ".[app]"`.'
        ) from exc
    return dag, dcc, html


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


def _plot_frame(html, src, include_refresh=False, save_src=None):
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


def _model_equations_content(html, equations):
    equations = [str(equation) for equation in (equations or []) if str(equation or "").strip()]
    if not equations:
        return ""
    return html.Div(
        className="ecat-equation-card",
        children=[html.Div(equation, className="ecat-equation-line") for equation in equations],
    )


def _default_model_equations(html):
    validation = validate_simulation_mechanism("preset", "E")
    return _model_equations_content(html, validation.get("formatted_equations") or [])


def _zoom_button(html, label, action, title, icon_src=None):
    child = html.Img(src=icon_src, alt="", className="ecat-zoom-button-icon") if icon_src else label
    return html.Button(
        child,
        className="ecat-zoom-button",
        title=title,
        **{
            "aria-label": title,
            "data-ecat-zoom-action": action,
            "type": "button",
        },
    )


def create_layout(config: AppConfig | None = None, initial_state: dict | None = None):
    config = config or AppConfig.from_env()
    initial_state = initial_state or {}
    initial_table = initial_state.get("table", {"columns": [], "data": []})
    initial_column_options = initial_state.get("column_options")
    initial_plot = initial_state.get("plot")
    initial_save_plot = initial_state.get("save_plot")
    initial_status = initial_state.get("status", "")
    initial_conditions = initial_state.get("conditions", [])
    initial_reference_options = _reference_file_options_from_state(initial_state)
    initial_selected_row_ids = initial_state.get("included_row_ids") or [
        row.get("id")
        for row in initial_table.get("data", [])
        if row.get("id") is not None
    ]
    dag, dcc, html = _dash()
    return html.Div(
        id="ecat-app-shell",
        className="ecat-app",
        children=[
            dcc.Store(id="ecat-objects-store", data=initial_state or None),
            dcc.Store(id="ecat-workflow-store", data=initial_state.get("workflow")),
            dcc.Store(
                id="ecat-selected-row-ids-store",
                data=initial_selected_row_ids,
            ),
            dcc.Store(id="ecat-analysis-results-store", data={}),
            html.Div(id="ecat-scroll-target", className="ecat-scroll-target", hidden=True),
            dcc.Store(
                id="ecat-plot-options-store",
                data={
                    "legend": True,
                    "legend mode": "colorbar",
                    "color mode": "auto",
                    "title": "auto",
                    "plot style": "notebook",
                    "plot convention": "IUPAC",
                    "_format": "svg",
                    "_dpi": 300,
                },
            ),
            dcc.Store(
                id="ecat-config-store",
                data={
                    "mode": config.mode,
                    "enable_folder_picker": config.enable_folder_picker,
                    "allow_code_execution": config.allow_code_execution,
                },
            ),
            html.Header(
                id="ecat-app-header",
                className="ecat-app-header",
                children=[
                    html.Div(
                        className="ecat-header-title-group",
                        children=[
                            html.Img(
                                src="/assets/ecat-logo_2_accent.svg",
                                className="ecat-header-logo",
                                alt="eCAT logo",
                            ),
                            html.Div(
                                className="ecat-header-copy",
                                children=[
                                    html.Div("eCAT Workbench", className="ecat-app-title"),
                                    html.Div("Visual analysis workspace", className="ecat-app-subtitle"),
                                ],
                            ),
                        ],
                    ),
                    html.Div(
                        className="ecat-header-center",
                        children=[
                            html.Div(
                                "No data loaded",
                                id="ecat-session-label",
                                className="ecat-session-label",
                                title="Current data session",
                            ),
                            dcc.Loading(
                                id="ecat-global-busy-loading",
                                target_components={
                                    "ecat-objects-store": "data",
                                    "ecat-default-plot": "children",
                                    "ecat-analysis-results-store": "data",
                                    "ecat-model-results-content": "children",
                                },
                                delay_show=150,
                                delay_hide=200,
                                custom_spinner=html.Div(
                                    className="ecat-global-busy",
                                    children=[
                                        html.Span(className="ecat-global-busy-dot"),
                                        html.Span("Working"),
                                    ],
                                ),
                                children=html.Div(id="ecat-global-busy-anchor", className="ecat-global-busy-anchor"),
                            ),
                        ],
                    ),
                    html.Div(
                        className="ecat-header-actions",
                        children=[
                            html.Div(
                                className="ecat-zoom-controls",
                                role="group",
                                **{"aria-label": "Zoom controls"},
                                children=[
                                    html.Img(src="/assets/ecat_icon_zoom.svg", alt="", className="ecat-zoom-icon"),
                                    html.Span("Zoom", className="ecat-zoom-label"),
                                    _zoom_button(html, "-", "out", "Zoom out"),
                                    html.Span("100%", id="ecat-zoom-value", className="ecat-zoom-value"),
                                    _zoom_button(html, "+", "in", "Zoom in"),
                                ],
                            ),
                            html.Button("About", id="ecat-about-button", className="ecat-header-button"),
                        ],
                    ),
                ],
            ),
            html.Div(
                id="ecat-about-panel",
                className="ecat-about-panel",
                hidden=True,
                children=[
                    html.Div("About eCAT Workbench", className="ecat-about-title"),
                    html.P(
                        "eCAT, short for electroChemical Analysis Tools, is a Python package for loading, organizing, plotting, and analyzing electrochemical data from common lab workflows."
                    ),
                    html.P(
                        "This app is a local-first interface for the eCAT package. It keeps the notebook-facing API intact while adding import tables, plotting controls, CV/CA/CP analysis panels, and reproducible Python export."
                    ),
                    html.Div(
                        className="ecat-about-meta",
                        children=[
                            html.Div([html.Strong("Package"), html.Span(f"ecat {e.__version__}")]),
                            html.Div([html.Strong("Author"), html.Span("Luke Elissiry")]),
                            html.Div([html.Strong("License"), html.Span("MIT License, copyright 2026 Luke Elissiry")]),
                            html.Div(
                                [
                                    html.Strong("Simulation backend"),
                                    html.Span("Optional ElectroKitty backend by Ožbej Vodeb, BSD 3-Clause License"),
                                ]
                            ),
                            html.Div(
                                [
                                    html.Strong("GitHub"),
                                    html.A(
                                        "github.com/ljelissiry/eCAT",
                                        href="https://github.com/ljelissiry/eCAT",
                                        target="_blank",
                                        rel="noopener noreferrer",
                                    ),
                                ]
                            ),
                        ],
                    ),
                    html.Div(
                        className="ecat-about-shortcuts",
                        children=[
                            html.Div("Keyboard Shortcuts", className="ecat-about-subtitle"),
                            html.Div(
                                className="ecat-shortcut-row",
                                children=[html.Kbd("Cmd/Ctrl + +"), html.Span("Zoom in")],
                            ),
                            html.Div(
                                className="ecat-shortcut-row",
                                children=[html.Kbd("Cmd/Ctrl + -"), html.Span("Zoom out")],
                            ),
                            html.Div(
                                className="ecat-shortcut-row",
                                children=[html.Kbd("Cmd/Ctrl + 0"), html.Span("Reset zoom")],
                            ),
                        ],
                    ),
                    html.P(
                        "The lab beta focuses on trustworthy CV workflows, with growing CA and CP support where parsers and analyses are covered by tests."
                    ),
                    html.P(
                        "Local mode can read local folders and optionally run trusted local code. Remote mode is intended for code generation, preview, and download only."
                    ),
                ],
            ),
            html.Div(
                className="ecat-workspace",
                children=[
                    html.Aside(
                        className="ecat-sidebar",
                        children=[
                            html.Div(id="ecat-sidebar-resizer", className="ecat-sidebar-resizer", title="Resize sidebar"),
                            html.Div(
                                className="ecat-sidebar-nav",
                                children=[
                                    dcc.Tabs(
                                        id="ecat-left-tabs",
                                        value="data",
                                        children=[
                                            dcc.Tab(
                                                label=_tab_label(html, "Data", "ecat_icon_data.svg"),
                                                value="data",
                                                children=_data_tab(
                                                    dcc,
                                                    html,
                                                    config,
                                                    initial_status,
                                                    initial_conditions,
                                                    initial_reference_options,
                                                    initial_state,
                                                ),
                                            ),
                                            dcc.Tab(
                                                label=_tab_label(html, "Plot", "ecat_icon_plotting.svg"),
                                                value="plot",
                                                children=_plotting_tab(dcc, html),
                                            ),
                                            dcc.Tab(
                                                label=_tab_label(html, "Analyze", "ecat_icon_analysis.svg"),
                                                value="analyze",
                                                children=_analysis_tab(dcc, html),
                                            ),
                                            dcc.Tab(
                                                label=_tab_label(html, "Model", "ecat_icon_model.svg"),
                                                value="model",
                                                children=_model_tab(dcc, html),
                                            ),
                                        ],
                                    ),
                                    html.Button(
                                        "‹",
                                        id="ecat-sidebar-toggle",
                                        className="ecat-icon-button ecat-sidebar-toggle",
                                        title="Collapse sidebar",
                                    ),
                                ],
                            ),
                        ],
                    ),
                    html.Main(
                        className="ecat-main",
                        children=[
                            html.Details(
                                id="ecat-main-panel-card",
                                className="ecat-panel ecat-card ecat-table-card",
                                open=True,
                                children=[
                                    html.Summary("Imported Data", className="ecat-card-summary"),
                                    html.Div(
                                        id="ecat-main-panel",
                                        children=_object_table(
                                            dag,
                                            dcc,
                                            html,
                                            initial_table,
                                            initial_column_options,
                                            initial_selected_row_ids,
                                        ),
                                    ),
                                ],
                            ),
                            html.Details(
                                id="ecat-plot-card",
                                className="ecat-panel ecat-card",
                                open=True,
                                children=[
                                    html.Summary("Multiplot", className="ecat-card-summary"),
                                    html.Div(
                                        hidden=True,
                                        children=[
                                            html.Button("", id="ecat-replot", hidden=True),
                                            html.Button("", id="ecat-save-plot", hidden=True),
                                            dcc.Download(id="ecat-download-plot"),
                                        ],
                                    ),
                                    dcc.Loading(
                                        id="ecat-plot-loading",
                                        type="circle",
                                        children=html.Div(
                                        id="ecat-default-plot",
                                        children=_plot_frame(
                                            html,
                                            initial_plot,
                                            include_refresh=True,
                                            save_src=initial_save_plot,
                                        ),
                                        ),
                                    ),
                                ],
                            ),
                            html.Div(
                                id="ecat-analysis-results",
                                className="ecat-panel ecat-card",
                                children=html.Details(
                                    className="ecat-card",
                                    open=True,
                                    children=[
                                        html.Summary("Analysis Results", className="ecat-card-summary"),
                                        html.Div(id="ecat-analysis-results-content"),
                                    ],
                                ),
                            ),
                            html.Div(
                                id="ecat-model-settings",
                                className="ecat-panel ecat-card",
                                children=html.Details(
                                    id="ecat-model-settings-card",
                                    className="ecat-card",
                                    children=[
                                        html.Summary("Model Settings", className="ecat-card-summary"),
                                        html.Div(
                                            className="ecat-model-grid",
                                            children=[
                                                html.Div(
                                                    className="ecat-model-placeholder ecat-model-grid-full",
                                                    children=[
                                                        html.Div("Cell", className="ecat-model-placeholder-title"),
                                                        dcc.Checklist(
                                                            id="ecat-model-cell-bound-columns",
                                                            className="ecat-model-bound-toggle",
                                                            options=[{"label": "Show bounds", "value": "bounds"}],
                                                            value=[],
                                                            style={"display": "none"},
                                                        ),
                                                        _model_parameter_grid(
                                                            dag,
                                                            "ecat-model-cell-parameters-grid",
                                                            DEFAULT_MODEL_CELL_PARAMETER_ROWS,
                                                            section="cell",
                                                            fit_enabled=False,
                                                        ),
                                                        html.Div(
                                                            id="ecat-model-cell-parameters-content",
                                                            className="ecat-muted-note",
                                                            style={"display": "none"},
                                                        ),
                                                    ],
                                                ),
                                                html.Div(
                                                    className="ecat-model-placeholder ecat-model-grid-full",
                                                    children=[
                                                        html.Div("Species", className="ecat-model-placeholder-title"),
                                                        dcc.Checklist(
                                                            id="ecat-model-species-bound-columns",
                                                            className="ecat-model-bound-toggle",
                                                            options=[{"label": "Show bounds", "value": "bounds"}],
                                                            value=[],
                                                            style={"display": "none"},
                                                        ),
                                                        _model_parameter_grid(
                                                            dag,
                                                            "ecat-model-species-parameters-grid",
                                                            DEFAULT_MODEL_SPECIES_PARAMETER_ROWS,
                                                            section="species",
                                                            fit_enabled=False,
                                                        ),
                                                        html.Div(
                                                            id="ecat-model-species-parameters-content",
                                                            className="ecat-muted-note",
                                                            style={"display": "none"},
                                                        ),
                                                    ],
                                                ),
                                                html.Div(
                                                    className="ecat-model-placeholder ecat-model-grid-full",
                                                    children=[
                                                        html.Div("Mechanism", className="ecat-model-placeholder-title"),
                                                        dcc.Checklist(
                                                            id="ecat-model-bound-columns",
                                                            className="ecat-model-bound-toggle",
                                                            options=[{"label": "Show bounds", "value": "bounds"}],
                                                            value=[],
                                                            style={"display": "none"},
                                                        ),
                                                        _model_parameter_grid(
                                                            dag,
                                                            "ecat-model-mechanism-parameters-grid",
                                                            DEFAULT_MODEL_MECHANISM_PARAMETER_ROWS,
                                                            section="mechanism",
                                                            fit_enabled=False,
                                                        ),
                                                        html.Div(
                                                            id="ecat-model-mechanism-parameters-content",
                                                            className="ecat-muted-note",
                                                            style={"display": "none"},
                                                        ),
                                                    ],
                                                ),
                                            ],
                                        ),
                                    ],
                                ),
                            ),
                            html.Div(
                                id="ecat-model-results",
                                className="ecat-panel ecat-card",
                                children=html.Details(
                                    id="ecat-model-results-card",
                                    className="ecat-card",
                                    children=[
                                        html.Summary("Model Results", className="ecat-card-summary"),
                                        html.Div(
                                            className="ecat-model-grid",
                                            children=[
                                                html.Div(
                                                    className="ecat-model-placeholder ecat-model-grid-full",
                                                    children=[
                                                        html.Div("Result Plot", className="ecat-model-placeholder-title"),
                                                        html.Div(
                                                            "Simulation overlays, residuals, and fit summaries will appear here.",
                                                            id="ecat-model-results-content",
                                                        ),
                                                        dcc.Loading(
                                                            id="ecat-model-simulation-progress-loading",
                                                            target_components={"ecat-model-simulation-progress-anchor": "children"},
                                                            delay_show=150,
                                                            delay_hide=250,
                                                            custom_spinner=html.Div(
                                                                className="ecat-model-progress",
                                                                children=[
                                                                    html.Div(
                                                                        className="ecat-model-progress-label",
                                                                        children="Running simulation",
                                                                    ),
                                                                    html.Div(
                                                                        className="ecat-model-progress-track",
                                                                        children=html.Div(className="ecat-model-progress-bar"),
                                                                    ),
                                                                ],
                                                            ),
                                                            children=html.Div(
                                                                id="ecat-model-simulation-progress-anchor",
                                                                className="ecat-model-progress-anchor",
                                                            ),
                                                        ),
                                                    ],
                                                ),
                                            ],
                                        ),
                                    ],
                                ),
                            ),
                        ],
                    ),
                ],
            ),
        ],
    )


def _tab_label(html, name, icon_filename):
    return html.Span(
        className="ecat-tab-label",
        title=name,
        children=[
            html.Span(
                className="ecat-tab-icon-tile",
                children=html.Img(
                    src=f"/assets/{icon_filename}",
                    className="ecat-tab-symbol",
                    alt=f"{name} symbol",
                    title=name,
                ),
            ),
            html.Span(name, className="ecat-tab-text"),
        ],
    )


def _subtab_label(html, name, icon_filename):
    return html.Span(
        className="ecat-subtab-label",
        title=name,
        children=[
            html.Img(
                src=f"/assets/{icon_filename}",
                className="ecat-subtab-symbol",
                alt=f"{name} symbol",
                title=name,
            ),
            html.Span(name),
        ],
    )


def _model_parameter_grid(dag, grid_id, row_data=None, section="mechanism", fit_enabled=True):
    return dag.AgGrid(
        id=grid_id,
        columnSize="responsiveSizeToFit",
        rowData=list(row_data or []),
        columnDefs=_model_parameter_column_defs(section, fit_enabled=fit_enabled, row_data=row_data),
        defaultColDef={
            "sortable": True,
            "filter": True,
            "resizable": True,
        },
        dashGridOptions={
            "domLayout": "autoHeight",
            "enableCellTextSelection": True,
            "ensureDomOrder": True,
            "singleClickEdit": True,
            "stopEditingWhenCellsLoseFocus": True,
        },
        className="ag-theme-alpine ecat-ag-grid ecat-model-parameters-grid",
        style={"width": "100%"},
    )


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
            if value != value:
                continue
        except TypeError:
            pass
        return True
    return False


def _model_parameter_column_defs(section="mechanism", fit_enabled=True, row_data=None):
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
                    "hide": True,
                },
                {
                    "field": "upper",
                    "headerName": "Upper",
                    "editable": True,
                    "cellClass": "ecat-editable-cell",
                    "cellEditor": "agTextCellEditor",
                    "hide": True,
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


def _format_model_viscosity(value):
    text = f"{float(value):.3g}"
    return text.replace("e-0", "e-").replace("e+0", "e+")


def _model_viscosity_options():
    viscosities = getattr(e.simulation, "SOLVENT_VISCOSITIES", {})
    return [
        {"label": "Custom", "value": "custom"},
        *[
            {"label": f"{solvent} ({_format_model_viscosity(viscosity)} m²/s)", "value": solvent}
            for solvent, viscosity in viscosities.items()
        ],
    ]


def _model_labeled_input(dcc, html, label, input_id, unit="", value=None, placeholder="", **props):
    input_mode = props.pop("inputMode", "decimal")
    return html.Label(
        className="ecat-model-field-row",
        children=[
            html.Span(
                className="ecat-model-field-label",
                children=html.Span(label),
            ),
            dcc.Input(
                id=input_id,
                type="text",
                inputMode=input_mode,
                value=value,
                placeholder=placeholder,
                className="ecat-model-float-input",
                **props,
            ),
            html.Span(unit, className="ecat-model-field-unit") if unit else "",
        ],
    )


def _analysis_labeled_input(dcc, html, label, input_id, unit="", value=None, placeholder="", **props):
    input_mode = props.pop("inputMode", "decimal")
    return html.Label(
        className="ecat-control-row ecat-inline-control-row ecat-multi-option-row",
        htmlFor=input_id,
        children=[
            html.Span(label, className="ecat-control-label"),
            dcc.Input(
                id=input_id,
                type="text",
                inputMode=input_mode,
                value=value,
                placeholder=placeholder,
                **props,
            ),
            html.Span(unit, className="ecat-model-field-unit") if unit else "",
        ],
    )


def _analysis_labeled_control(html, label, control):
    return html.Label(
        className="ecat-control-row ecat-inline-control-row ecat-multi-option-row",
        children=[
            html.Span(label, className="ecat-control-label"),
            html.Div(control, className="ecat-multi-option-control"),
        ],
    )


def _data_tab(dcc, html, config, initial_status="", initial_conditions=None, initial_reference_options=None, initial_state=None):
    return html.Div(
        className="ecat-tab-body ecat-data-tab-body",
        children=[
            dcc.Tabs(
                id="ecat-data-tabs",
                value="import",
                className="ecat-data-tabs",
                children=[
                    dcc.Tab(
                        label=_subtab_label(html, "Import", "ecat_icon_import.svg"),
                        value="import",
                        children=_import_tab(
                            dcc,
                            html,
                            config,
                            initial_status,
                            initial_conditions,
                            initial_reference_options,
                        ),
                    ),
                    dcc.Tab(
                        label=_subtab_label(html, "Export", "ecat_icon_export.svg"),
                        value="export",
                        children=_export_tab(dcc, html, initial_state),
                    ),
                ],
            ),
        ],
    )


def _import_tab(dcc, html, config, initial_status="", initial_conditions=None, initial_reference_options=None):
    from .defaults import example_folder_options

    folder_disabled = config.mode == "remote"
    initial_conditions = initial_conditions or []
    initial_reference_options = initial_reference_options or []
    return html.Div(
        className="ecat-tab-body",
        children=[
            html.Div("Load files or folders and apply reference settings before analysis.", className="ecat-tab-intro"),
            html.Div(
                className="ecat-source-panel",
                children=[
                    dcc.Tabs(
                        id="ecat-import-source-tabs",
                        value="files",
                        children=[
                            dcc.Tab(
                                label="Files",
                                value="files",
                                children=[
                                    dcc.Upload(
                                        id="ecat-upload",
                                        className="ecat-upload-zone",
                                        children=html.Div(
                                            className="ecat-upload-content",
                                            children=[
                                                html.Div("⇧", className="ecat-upload-icon"),
                                                html.Div("Drag&Drop files here", className="ecat-upload-title"),
                                                html.Div("or", className="ecat-upload-or"),
                                                html.Button("Browse Files", className="ecat-upload-button"),
                                            ],
                                        ),
                                        multiple=True,
                                    ),
                                ],
                            ),
                            dcc.Tab(
                                label="Folder",
                                value="folder",
                                disabled=folder_disabled,
                                children=[
                                    html.Div(
                                        className="ecat-folder-zone",
                                        children=[
                                            html.Div("Example data", className="ecat-folder-label"),
                                            dcc.Dropdown(
                                                id="ecat-example-folder",
                                                options=example_folder_options(),
                                                placeholder="Load packaged example folder",
                                                clearable=True,
                                                searchable=False,
                                                disabled=folder_disabled,
                                            ),
                                            html.Div("Folder path", className="ecat-folder-label"),
                                            dcc.Input(
                                                id="ecat-local-path",
                                                type="text",
                                                placeholder="Folder path",
                                                disabled=folder_disabled,
                                            ),
                                            dcc.Checklist(
                                                id="ecat-recursive",
                                                options=[{"label": "Search Subfolders", "value": "recursive"}],
                                                value=["recursive"],
                                            ),
                                            html.Button(
                                                "Load Folder",
                                                id="ecat-load-path",
                                                className="ecat-upload-button",
                                                disabled=folder_disabled,
                                            ),
                                        ],
                                    ),
                                ],
                            ),
                        ],
                    ),
                ],
            ),
            html.Div(id="ecat-status", className="ecat-status", children=initial_status),
            html.Div(
                id="ecat-import-conditions",
                className="ecat-import-conditions",
                children=(
                    [html.Div("Shared Conditions", className="ecat-condition-heading")]
                    + [
                        html.Div(condition, className="ecat-condition-item")
                        for condition in initial_conditions
                    ]
                    if initial_conditions
                    else []
                ),
            ),
            html.H3("Reference"),
            dcc.Dropdown(
                id="ecat-reference-mode",
                value="none",
                clearable=False,
                searchable=False,
                options=[
                    {"label": "None", "value": "none"},
                    {"label": "Manual", "value": "manual"},
                    {"label": "File", "value": "file"},
                    {"label": "Keyword", "value": "keyword"},
                    {"label": "Auto", "value": "auto"},
                ],
            ),
            html.Div(
                id="ecat-reference-offset-wrap",
                children=dcc.Input(id="ecat-reference-offset", type="number", placeholder="Reference offset"),
            ),
            html.Div(
                id="ecat-reference-label-wrap",
                children=dcc.Input(id="ecat-reference-label", type="text", placeholder="Reference label"),
            ),
            html.Div(
                id="ecat-reference-file-wrap",
                children=dcc.Dropdown(id="ecat-reference-file", options=initial_reference_options),
            ),
            html.Div(
                id="ecat-reference-keyword-wrap",
                children=dcc.Input(id="ecat-reference-keyword", type="text", placeholder="Reference keyword"),
            ),
            html.Div(
                id="ecat-reference-keywords-wrap",
                children=dcc.Input(id="ecat-reference-keywords", type="text", placeholder="Auto keywords, comma-separated"),
            ),
            html.Div(
                id="ecat-reference-guess-wrap",
                children=dcc.Input(id="ecat-reference-guess", type="text", placeholder="Reference guess"),
            ),
            html.Div(
                id="ecat-allow-self-reference-wrap",
                children=dcc.Checklist(
                    id="ecat-allow-self-reference",
                    options=[{"label": "Allow self reference", "value": "allow"}],
                    value=["allow"],
                ),
            ),
            html.Button("Apply Reference", id="ecat-apply-reference"),
            html.Div(id="ecat-import-warnings"),
        ],
    )


def _object_table(dag, dcc, html, initial_table=None, initial_column_options=None, initial_selected_row_ids=None):
    initial_table = initial_table or {}
    columns = [
        "index",
        "filename",
        "class",
        "type",
        "software",
        "gas",
        "solvent",
        "scan rate",
        "segments",
        "reference shift",
        "reference mode",
        "reference label",
        "reference source",
    ]
    return html.Div(
        children=[
            html.Div(
                className="ecat-table-toolbar",
                children=[
                    dcc.Dropdown(
                        id="ecat-table-extra-columns",
                        multi=True,
                        placeholder="Visible columns",
                        value=selected_column_values(initial_table),
                        options=initial_column_options or _column_options_from_table(initial_table),
                    ),
                    html.Button("Reset", id="ecat-reset-columns"),
                ],
            ),
            html.Div(
                className="ecat-table-scroll",
                children=dag.AgGrid(
                    id="ecat-object-table",
                    columnSize="autoSize",
                    rowData=initial_table.get("data") or [],
                    columnDefs=ag_grid_column_defs(
                        initial_table
                        if initial_table.get("columns")
                        else {"columns": [{"name": col, "id": col} for col in columns]}
                    ),
                    selectedRows=selected_grid_rows_for_ids(
                        initial_table.get("data", []),
                        initial_selected_row_ids,
                    ),
                    getRowId="params.data.id",
                    defaultColDef={
                        "sortable": True,
                        "filter": True,
                        "resizable": True,
                        "suppressMovable": False,
                    },
                    dashGridOptions={
                        "rowSelection": "multiple",
                        "suppressRowClickSelection": True,
                        "pagination": True,
                        "paginationPageSize": 20,
                        "animateRows": False,
                    },
                    className="ag-theme-alpine ecat-ag-grid",
                    style={"height": "430px", "width": "100%"},
                ),
            ),
        ],
    )


def _column_options_from_table(table):
    return [
        {"label": column.get("name"), "value": column.get("id")}
        for column in (table or {}).get("columns", [])
        if column.get("id") != "index"
    ]


def _reference_file_options_from_state(initial_state):
    options = []
    for row in (initial_state or {}).get("summary", []):
        index = row.get("index")
        label = row.get("filename") or row.get("name") or str(index)
        options.append({"label": f"{index}: {label}", "value": index})
    return options


def _control_row(html, label, control, class_name=""):
    label_node = html.Label(label, className="ecat-control-label") if label else ""
    return html.Div(
        className=f"ecat-control-row {class_name}".strip(),
        children=[
            label_node,
            html.Div(control, className="ecat-control-field"),
        ],
    )


def _plot_section(html, title):
    return html.Div(title, className="ecat-plot-section-heading")


def _plot_disclosure(html, title, children, *, open=True):
    return html.Details(
        className="ecat-plot-section",
        open=open,
        children=[
            html.Summary(title, className="ecat-plot-section-heading"),
            html.Div(children=children, className="ecat-plot-section-body"),
        ],
    )


def _plot_subheading(html, title):
    return html.Div(title, className="ecat-control-subheading")


def _plotting_tab(dcc, html):
    return html.Div(
        className="ecat-tab-body",
        children=[
            html.Div("Tune static plots, animations, labels, legends, and export settings.", className="ecat-tab-intro"),
            html.Div(
                className="ecat-action-row",
                children=[
                    html.Button("Refresh plots", id="ecat-plotting-replot", className="ecat-primary-button"),
                    html.Button("Save all plots", id="ecat-plotting-save-plot", className="ecat-primary-button"),
                ],
            ),
            _plot_disclosure(
                html,
                "Labels",
                [
                    _plot_subheading(html, "Axis Labels"),
                    _control_row(
                        html,
                        "",
                        dcc.RadioItems(
                            id="ecat-plot-axis-label-mode",
                            options=[
                                {"label": "Auto", "value": "auto"},
                                {"label": "Manual", "value": "manual"},
                                {"label": "None", "value": "none"},
                            ],
                            value="auto",
                            className="ecat-segmented",
                            inline=True,
                        ),
                    ),
                    html.Div(
                        id="ecat-plot-axis-label-inputs",
                        style={"display": "none"},
                        className="ecat-two-column-controls",
                        children=[
                            dcc.Input(id="ecat-plot-x-label", type="text", placeholder="X label"),
                            dcc.Input(id="ecat-plot-y-label", type="text", placeholder="Y label"),
                        ],
                    ),
                    _plot_subheading(html, "Title"),
                    _control_row(
                        html,
                        "",
                        dcc.RadioItems(
                            id="ecat-plot-title-mode",
                            options=[
                                {"label": "Auto", "value": "auto"},
                                {"label": "Manual", "value": "manual"},
                                {"label": "None", "value": "none"},
                            ],
                            value="auto",
                            className="ecat-segmented",
                            inline=True,
                        ),
                    ),
                    html.Div(
                        id="ecat-plot-custom-title-wrap",
                        style={"display": "none"},
                        children=dcc.Input(
                            id="ecat-plot-custom-title",
                            type="text",
                            placeholder="Custom title",
                            className="ecat-full-width",
                        ),
                    ),
                ],
            ),
            _plot_disclosure(
                html,
                "Legend",
                [
                    _control_row(
                        html,
                        "",
                        dcc.Checklist(
                            id="ecat-plot-legend",
                            options=[
                                {"label": "Show legend", "value": "legend"},
                            ],
                            value=["legend"],
                        ),
                    ),
                    html.Div(
                        id="ecat-plot-legend-options-wrap",
                        children=[
                            dcc.Checklist(
                                id="ecat-plot-gradients",
                                className="ecat-control-field",
                                options=[{"label": "Allow gradients", "value": "gradients"}],
                                value=["gradients"],
                            ),
                            html.Div(
                                id="ecat-plot-colorbar-wrap",
                                children=dcc.Checklist(
                                    id="ecat-plot-colorbar",
                                    className="ecat-control-field",
                                    options=[{"label": "Allow colorbars", "value": "colorbar"}],
                                    value=["colorbar"],
                                ),
                            ),
                            dcc.Checklist(
                                id="ecat-plot-label-options",
                                className="ecat-control-field",
                                options=[{"label": "Deduplicate labels", "value": "deduplicate"}],
                                value=[],
                            ),
                        ],
                    ),
                ],
            ),
            _plot_disclosure(
                html,
                "Display",
                [
                    _control_row(
                        html,
                        "Convention",
                        dcc.RadioItems(
                            id="ecat-plot-convention",
                            options=[
                                {"label": "IUPAC", "value": "IUPAC"},
                                {"label": "US", "value": "US"},
                            ],
                            value="IUPAC",
                            className="ecat-segmented",
                            inline=True,
                        ),
                    ),
                    _control_row(
                        html,
                        "",
                        dcc.Checklist(
                            id="ecat-plot-invert-y-axis",
                            className="ecat-control-field",
                            options=[{"label": "Invert y axis", "value": "invert_y_axis"}],
                            value=[],
                        ),
                    ),
                    _control_row(
                        html,
                        "Plot Style",
                        dcc.Dropdown(
                            id="ecat-plot-style",
                            value="notebook",
                            clearable=False,
                            searchable=False,
                            options=[
                                {"label": "Notebook", "value": "notebook"},
                                {"label": "Publication", "value": "publication"},
                                {"label": "Saveant", "value": "saveant"},
                                {"label": "Matplotlib", "value": "matplotlib"},
                            ],
                        ),
                    ),
                    _control_row(
                        html,
                        "Mode",
                        dcc.RadioItems(
                            id="ecat-plot-output-mode",
                            options=[
                                {"label": "Static", "value": "static"},
                                {"label": "Animated", "value": "animated"},
                            ],
                            value="static",
                            className="ecat-segmented",
                            inline=True,
                        ),
                    ),
                    html.Div(
                        id="ecat-animation-options",
                        className="ecat-panel ecat-card ecat-animation-card",
                        style={"display": "none"},
                        children=[
                            html.Div("Animation", className="ecat-card-title"),
                            html.Div(
                                className="ecat-two-column-controls",
                                children=[
                                    _control_row(
                                        html,
                                        "Framerate (FPS)",
                                        dcc.Input(id="ecat-animation-fps", type="number", min=1, step=1, value=20),
                                    ),
                                    _control_row(
                                        html,
                                        "Stride",
                                        dcc.Input(id="ecat-animation-stride", type="number", min=1, step=1, value=1),
                                    ),
                                ],
                            ),
                            _plot_subheading(html, "Trace"),
                            dcc.RadioItems(
                                id="ecat-animation-trace-mode",
                                options=[
                                    {"label": "Draw", "value": "draw"},
                                    {"label": "Instant", "value": "instant"},
                                ],
                                value="draw",
                                className="ecat-segmented",
                                inline=True,
                            ),
                            _plot_subheading(html, "Sequence"),
                            dcc.RadioItems(
                                id="ecat-animation-schedule",
                                options=[
                                    {"label": "Sequential", "value": "sequential"},
                                    {"label": "Staggered", "value": "staggered"},
                                    {"label": "Together", "value": "simultaneous"},
                                ],
                                value="staggered",
                                className="ecat-segmented",
                                inline=True,
                            ),
                            html.Div(
                                id="ecat-animation-stagger-wrap",
                                children=_control_row(
                                    html,
                                    "Stagger time (s)",
                                    dcc.Input(
                                        id="ecat-animation-stagger-time",
                                        type="number",
                                        min=0,
                                        step=0.1,
                                        value=0.5,
                                    ),
                                ),
                            ),
                            _plot_subheading(html, "Timing"),
                            dcc.RadioItems(
                                id="ecat-animation-timing-mode",
                                options=[
                                    {"label": "Duration", "value": "duration"},
                                    {"label": "Rate", "value": "rate"},
                                ],
                                value="duration",
                                className="ecat-segmented",
                                inline=True,
                            ),
                            _control_row(
                                html,
                                "Duration / rate",
                                dcc.Input(
                                    id="ecat-animation-timing-value",
                                    type="number",
                                    min=0,
                                    step=0.1,
                                    value=2,
                                ),
                            ),
                            _plot_subheading(html, "Advanced Animation"),
                            dcc.Checklist(
                                id="ecat-animation-advanced",
                                options=[
                                    {"label": "Include quiet time", "value": "include_quiet_time"},
                                    {"label": "Loop preview", "value": "loop"},
                                ],
                                value=["loop"],
                            ),
                            _control_row(
                                html,
                                "End hold (s)",
                                dcc.Input(
                                    id="ecat-animation-end-hold",
                                    type="number",
                                    min=0,
                                    step=0.1,
                                    value=2,
                                ),
                            ),
                        ],
                    ),
                    _control_row(
                        html,
                        "",
                        html.Div(
                            children=[
                                dcc.Checklist(
                                    id="ecat-plot-display-options",
                                    options=[
                                        {"label": "Grid", "value": "grid"},
                                    ],
                                    value=[],
                                ),
                                dcc.Checklist(
                                    id="ecat-plot-trim-enabled",
                                    options=[{"label": "Trim potential window", "value": "trim"}],
                                    value=[],
                                ),
                            ],
                        ),
                    ),
                    html.Div(
                        id="ecat-plot-trim-bounds",
                        style={"display": "none"},
                        className="ecat-three-column-controls",
                        children=[
                            dcc.Dropdown(
                                id="ecat-plot-trim-mode",
                                value="expand",
                                clearable=False,
                                searchable=False,
                                options=[
                                    {"label": "Expand", "value": "expand"},
                                    {"label": "Pointwise", "value": "pointwise"},
                                    {"label": "Strict", "value": "strict"},
                                ],
                            ),
                            dcc.Input(id="ecat-plot-trim-min", type="number", placeholder="Min potential"),
                            dcc.Input(id="ecat-plot-trim-max", type="number", placeholder="Max potential"),
                        ],
                    ),
                    _plot_subheading(html, "Offset"),
                    _control_row(
                        html,
                        "",
                        dcc.Input(
                            id="ecat-plot-offset",
                            type="number",
                            placeholder="Offset",
                            className="ecat-full-width",
                        ),
                    ),
                    html.Div(
                        id="ecat-plot-offset-controls",
                        style={"display": "none"},
                        children=[
                            html.Div(
                                className="ecat-two-column-controls",
                                children=[
                                    dcc.Input(
                                        id="ecat-plot-scale-bar-height",
                                        type="number",
                                        placeholder="Scale bar height",
                                    ),
                                    dcc.Dropdown(
                                        id="ecat-plot-scale-bar-location",
                                        value="upper left",
                                        clearable=False,
                                        searchable=False,
                                        options=[
                                            {"label": "Upper left", "value": "upper left"},
                                            {"label": "Upper right", "value": "upper right"},
                                            {"label": "Lower left", "value": "lower left"},
                                            {"label": "Lower right", "value": "lower right"},
                                        ],
                                    ),
                                ],
                            ),
                            dcc.Checklist(
                                id="ecat-plot-offset-axis-options",
                                className="ecat-control-field",
                                options=[{"label": "Hide y-axis numbers", "value": "hide_y_numbers"}],
                                value=["hide_y_numbers"],
                            ),
                        ],
                    ),
                ],
            ),
            _plot_disclosure(
                html,
                "Annotations",
                [
                    html.Div(
                        className="ecat-inline-subheading-action",
                        children=[
                            _plot_subheading(html, "Directional Arrows"),
                            html.Button(
                                "+",
                                id="ecat-add-directional-arrow",
                                className="ecat-plus-button",
                                title="Add directional arrow",
                            ),
                        ],
                    ),
                    html.Div(
                        id="ecat-directional-arrow-options",
                        style={"display": "none"},
                        className="ecat-two-column-controls",
                        children=[
                            dcc.Input(
                                id="ecat-animation-arrow-potential",
                                type="number",
                                placeholder="Arrow potential / V",
                            ),
                            dcc.Input(
                                id="ecat-animation-arrow-segment",
                                type="number",
                                min=1,
                                step=1,
                                placeholder="Segment",
                            ),
                        ],
                    ),
                    _plot_subheading(html, "Scale Bar"),
                    dcc.Checklist(
                        id="ecat-animation-scale-bar-enabled",
                        options=[{"label": "Show scale bar", "value": "scale_bar"}],
                        value=[],
                    ),
                    html.Div(
                        id="ecat-animation-scale-bar-options",
                        style={"display": "none"},
                        className="ecat-two-column-controls",
                        children=[
                            dcc.Input(
                                id="ecat-animation-scale-bar-length",
                                type="number",
                                placeholder="Length",
                            ),
                            dcc.Dropdown(
                                id="ecat-animation-scale-bar-location",
                                value="upper left",
                                clearable=False,
                                searchable=False,
                                options=[
                                    {"label": "Upper left", "value": "upper left"},
                                    {"label": "Upper right", "value": "upper right"},
                                    {"label": "Lower left", "value": "lower left"},
                                    {"label": "Lower right", "value": "lower right"},
                                ],
                            ),
                        ],
                    ),
                ],
            ),
            _plot_disclosure(
                html,
                "Save",
                [
                    _control_row(
                        html,
                        "",
                        html.Div(
                            className="ecat-two-column-controls",
                            children=[
                                dcc.Dropdown(
                                    id="ecat-plot-format",
                                    value="svg",
                                    clearable=False,
                                    searchable=False,
                                    options=[
                                        {"label": "PNG", "value": "png"},
                                        {"label": "SVG", "value": "svg"},
                                        {"label": "PDF", "value": "pdf"},
                                    ],
                                ),
                                dcc.Input(id="ecat-plot-dpi", type="number", min=72, step=1, value=300, placeholder="DPI"),
                            ],
                        ),
                    ),
                ],
            ),
        ],
    )


def _analysis_tab(dcc, html):
    return html.Div(
        className="ecat-tab-body",
        children=[
            html.Div("Run CV, CA, and CP analyses on the selected imported data.", className="ecat-tab-intro"),
            html.Button("Reset Results", id="ecat-reset-analysis-results", className="ecat-full-width"),
            html.Details(
                id="ecat-cv-analysis-card",
                className="ecat-analysis-class-card",
                open=True,
                children=[
                    html.Summary("Cyclic Voltammetry", className="ecat-card-summary"),
                    dcc.Tabs(
                        id="ecat-cv-analysis-tabs",
                        className="ecat-nested-tabs",
                        value="single",
                        children=[
                            dcc.Tab(label="Single", value="single", children=_single_cv_controls(dcc, html)),
                            dcc.Tab(label="Multiple", value="multiple", children=_multi_cv_controls(dcc, html)),
                        ],
                    ),
                ],
            ),
            html.Details(
                id="ecat-ca-analysis-card",
                className="ecat-analysis-class-card",
                children=[
                    html.Summary("Chronoamperometry", className="ecat-card-summary"),
                    _ca_controls(dcc, html),
                ],
            ),
            html.Details(
                id="ecat-cp-analysis-card",
                className="ecat-analysis-class-card",
                children=[
                    html.Summary("Chronopotentiometry", className="ecat-card-summary"),
                    _cp_controls(dcc, html),
                ],
            ),
        ],
    )


def _model_tab(dcc, html):
    simulation_notice = ""
    if not simulation_backend_available():
        simulation_notice = html.Div(
            SIMULATION_INSTALL_MESSAGE,
            className="ecat-model-status",
        )
    return html.Div(
        className="ecat-tab-body",
        children=[
            html.Div("Build a mechanism once, then simulate or fit it against CV data.", className="ecat-tab-intro"),
            simulation_notice,
            _plot_section(html, "Mechanism"),
            dcc.Store(id="ecat-model-mechanism-source", data="preset"),
            html.Div(
                id="ecat-model-preset-wrap",
                children=dcc.Dropdown(
                    id="ecat-model-mechanism-preset",
                    value="E",
                    clearable=False,
                    searchable=False,
                    options=[
                        {"label": "E", "value": "E"},
                        {"label": "EE", "value": "EE"},
                        {"label": "EC", "value": "EC"},
                        {"label": "ECE", "value": "ECE"},
                        {"label": "ECAT", "value": "ECAT"},
                        {"label": "Square Scheme", "value": "Square"},
                        {"label": "Custom", "value": "custom"},
                    ],
                ),
            ),
            html.Div(
                id="ecat-model-custom-wrap",
                style={"display": "none"},
                children=[
                    html.Div(
                        "Enter one eCAT reaction per line. Use coefficients such as 2A or repeated terms such as A+A; eCAT compiles backend syntax automatically.",
                        className="ecat-control-help",
                    ),
                    dcc.Textarea(
                        id="ecat-model-mechanism-custom",
                        className="ecat-code ecat-model-mechanism-text",
                        placeholder="E(1):a=b\nC:b=c",
                        value="",
                    ),
                    html.Div(
                        "Example: E(1):Fe2=Fe1, then C:Fe1>Fe0 on the next line.",
                        className="ecat-control-help",
                    ),
                ],
            ),
            html.Div(
                "Mechanism ready. Run Simulate CV to enable Fit.",
                id="ecat-model-mechanism-status",
                className="ecat-model-status",
            ),
            html.Div(
                id="ecat-model-formatted-equations",
                className="ecat-model-equations",
                children=_default_model_equations(html),
            ),
            dcc.Tabs(
                id="ecat-model-tabs",
                className="ecat-nested-tabs",
                value="simulate",
                children=[
                    dcc.Tab(
                        label="Simulate",
                        value="simulate",
                        children=[
                            _control_row(
                                html,
                                "",
                                dcc.RadioItems(
                                    id="ecat-model-simulate-mode",
                                    options=[
                                        {"label": "From Scratch", "value": "scratch"},
                                        {"label": "From CV", "value": "cv", "disabled": True},
                                    ],
                                    value="scratch",
                                    className="ecat-segmented ecat-segmented-wrap",
                                    inline=False,
                                ),
                            ),
                            html.Div(
                                id="ecat-model-program-card",
                                className="ecat-model-input-card ecat-model-program-card",
                                children=[
                                    html.Div("CV Program", className="ecat-card-title"),
                                    _model_labeled_input(dcc, html, "Initial potential", "ecat-model-program-ei", "V", 0.0),
                                    _model_labeled_input(dcc, html, "Low potential", "ecat-model-program-e-low", "V", -1.5),
                                    _model_labeled_input(dcc, html, "High potential", "ecat-model-program-e-high", "V", 0.0),
                                    _model_labeled_input(dcc, html, "Final potential", "ecat-model-program-ef", "V"),
                                    _model_labeled_input(dcc, html, "Scan rate", "ecat-model-program-scan-rate", "V s⁻¹", 0.1),
                                    _model_labeled_input(dcc, html, "Segments", "ecat-model-program-segments", "", 2, inputMode="numeric"),
                                    _model_labeled_input(dcc, html, "Points per segment", "ecat-model-program-points", "", 300, inputMode="numeric"),
                                    _model_labeled_input(dcc, html, "Quiet time", "ecat-model-program-quiet-time", "s", 0),
                                    dcc.Checklist(
                                        id="ecat-model-program-plot-options",
                                        options=[{"label": "Include quiet time in program plot", "value": "quiet_time"}],
                                        value=["quiet_time"],
                                    ),
                                    html.Div(
                                        className="ecat-model-program-actions",
                                        children=[
                                            html.Button(
                                                "Plot CV Program",
                                                id="ecat-model-plot-program",
                                                className="ecat-primary-button",
                                            ),
                                        ],
                                    ),
                                ],
                            ),
                            html.Div(
                                id="ecat-model-cv-data-card",
                                className="ecat-model-input-card",
                                style={"display": "none"},
                                children=[
                                    html.Div("CV Data", className="ecat-card-title"),
                                    _model_labeled_input(
                                        dcc,
                                        html,
                                        "CV index",
                                        "ecat-model-cv-index",
                                        "",
                                        "",
                                        inputMode="numeric",
                                        disabled=True,
                                    ),
                                    html.Div(id="ecat-model-cv-index-status", className="ecat-control-help"),
                                    _control_row(
                                        html,
                                        "Trim mode",
                                        dcc.Dropdown(
                                            id="ecat-model-cv-trim-mode",
                                            value="expand",
                                            clearable=False,
                                            searchable=False,
                                            options=[
                                                {"label": "None", "value": "none"},
                                                {"label": "Expand", "value": "expand"},
                                                {"label": "Pointwise", "value": "pointwise"},
                                                {"label": "Strict", "value": "strict"},
                                            ],
                                        ),
                                    ),
                                    html.Div(
                                        id="ecat-model-cv-window-fields",
                                        children=[
                                            _model_labeled_input(dcc, html, "Window min", "ecat-model-cv-window-min", "V"),
                                            _model_labeled_input(dcc, html, "Window max", "ecat-model-cv-window-max", "V"),
                                        ],
                                    ),
                                    _model_labeled_input(dcc, html, "Segment(s)", "ecat-model-cv-segments", "", placeholder="blank = all"),
                                    _model_labeled_input(dcc, html, "Stride", "ecat-model-cv-stride", "points", 20, inputMode="numeric"),
                                    _control_row(
                                        html,
                                        "Estimate Cdl",
                                        dcc.Dropdown(
                                            id="ecat-model-cv-estimate-cdl",
                                            value="auto",
                                            clearable=False,
                                            searchable=False,
                                            options=[
                                                {"label": "Auto", "value": "auto"},
                                                {"label": "Off", "value": "off"},
                                            ],
                                        ),
                                    ),
                                    html.Div(
                                        className="ecat-model-program-actions",
                                        children=[
                                            html.Button(
                                                "Plot CV Program",
                                                id="ecat-model-plot-cv-program",
                                                className="ecat-primary-button",
                                            ),
                                        ],
                                    ),
                                ],
                            ),
                            html.Div(
                                id="ecat-model-simulation-setup-card",
                                className="ecat-model-input-card ecat-model-setup-card",
                                children=[
                                    html.Div("Simulation Setup", className="ecat-card-title"),
                                    _control_row(
                                        html,
                                        "Spatial grid",
                                        dcc.Dropdown(
                                            id="ecat-model-spatial-mode",
                                            value="fast",
                                            clearable=False,
                                            searchable=False,
                                            options=[
                                                {"label": "Fast", "value": "fast"},
                                                {"label": "Balanced", "value": "balanced"},
                                                {"label": "Accurate", "value": "accurate"},
                                                {"label": "Custom", "value": "custom"},
                                            ],
                                        ),
                                    ),
                                    html.Div(
                                        id="ecat-model-spatial-custom-fields",
                                        style={"display": "none"},
                                        children=[
                                            _model_labeled_input(
                                                dcc,
                                                html,
                                                "dx fraction",
                                                "ecat-model-spatial-dx-fraction",
                                                "",
                                                0.001 / 36,
                                            ),
                                            _model_labeled_input(
                                                dcc,
                                                html,
                                                "nx",
                                                "ecat-model-spatial-nx",
                                                "points",
                                                20,
                                                inputMode="numeric",
                                            ),
                                            _control_row(
                                                html,
                                                "Viscosity",
                                                dcc.Dropdown(
                                                    id="ecat-model-spatial-viscosity-source",
                                                    value="custom",
                                                    clearable=False,
                                                    searchable=False,
                                                    options=_model_viscosity_options(),
                                                ),
                                                "ecat-model-field-row",
                                            ),
                                            html.Div(
                                                id="ecat-model-spatial-viscosity-custom",
                                                children=_model_labeled_input(
                                                    dcc,
                                                    html,
                                                    "custom viscosity",
                                                    "ecat-model-spatial-viscosity",
                                                    "m² s⁻¹",
                                                    1e-6,
                                                ),
                                            ),
                                            _model_labeled_input(
                                                dcc,
                                                html,
                                                "rotation",
                                                "ecat-model-spatial-rotation",
                                                "Hz",
                                                0,
                                            ),
                                        ],
                                    ),
                                    dcc.Checklist(
                                        id="ecat-model-over-conditions",
                                        options=[{"label": "Over Conditions", "value": "conditions"}],
                                        value=[],
                                    ),
                                    html.Div(
                                        id="ecat-model-over-conditions-card",
                                        style={"display": "none"},
                                        className="ecat-model-input-card",
                                        children=[
                                            html.Div("Over Conditions", className="ecat-card-title"),
                                            _control_row(
                                                html,
                                                "Condition axis",
                                                dcc.Dropdown(
                                                    id="ecat-model-condition-axis",
                                                    value="scan_rate",
                                                    clearable=False,
                                                    searchable=False,
                                                    options=[
                                                        {"label": "Scan rate", "value": "scan_rate"},
                                                        {"label": "Concentration", "value": "concentration"},
                                                        {"label": "Temperature", "value": "temperature"},
                                                    ],
                                                ),
                                            ),
                                            _model_labeled_input(
                                                dcc,
                                                html,
                                                "Number",
                                                "ecat-model-condition-count",
                                                "",
                                                3,
                                                placeholder="3",
                                                inputMode="numeric",
                                            ),
                                            html.Div(
                                                className="ecat-model-slider-block",
                                                children=[
                                                    html.Label("Range", className="ecat-control-label"),
                                                    dcc.RangeSlider(
                                                        id="ecat-model-condition-range",
                                                        min=0,
                                                        max=1,
                                                        step=0.01,
                                                        value=[0.05, 0.2],
                                                        marks={0.05: "0.05", 0.2: "0.2"},
                                                        tooltip={"placement": "bottom", "always_visible": False},
                                                        allowCross=False,
                                                    ),
                                                ],
                                            ),
                                            html.Div(
                                                id="ecat-model-condition-species-wrap",
                                                style={"display": "none"},
                                                children=[
                                                    _control_row(
                                                        html,
                                                        "Species",
                                                        dcc.Dropdown(
                                                            id="ecat-model-condition-species",
                                                            value="A",
                                                            clearable=False,
                                                            searchable=False,
                                                            options=[
                                                                {"label": "A", "value": "A"},
                                                                {"label": "B", "value": "B"},
                                                            ],
                                                        ),
                                                    ),
                                                ],
                                            ),
                                        ],
                                    ),
                                    html.Div(
                                        className="ecat-model-program-actions",
                                        children=[
                                            html.Button(
                                                "Simulate CV",
                                                id="ecat-model-run-simulate",
                                                className="ecat-primary-button",
                                                disabled=False,
                                                title="",
                                            ),
                                        ],
                                    ),
                                ],
                            ),
                        ],
                    ),
                    dcc.Tab(
                        id="ecat-model-fit-tab",
                        label=html.Span(
                            "Fit",
                            title="Run a simulation first to create the fit starting guess.",
                        ),
                        value="fit",
                        disabled=True,
                        children=[
                            _control_row(
                                html,
                                "",
                                dcc.RadioItems(
                                    id="ecat-model-fit-mode",
                                    options=[
                                        {"label": "Single CV", "value": "single"},
                                        {"label": "Multiple CVs", "value": "multiple"},
                                    ],
                                    value="single",
                                    className="ecat-segmented",
                                    inline=True,
                                ),
                            ),
                            _model_labeled_input(
                                dcc,
                                html,
                                "CV index",
                                "ecat-model-fit-cv-index",
                                "",
                                "",
                                inputMode="numeric",
                                disabled=True,
                            ),
                            html.Div(id="ecat-model-fit-compatibility", className="ecat-control-help"),
                            html.Button(
                                "Fit",
                                id="ecat-model-run-fit",
                                className="ecat-full-width",
                                disabled=True,
                                title="Run a simulation first to create the fit starting guess.",
                            ),
                        ],
                    ),
                ],
            ),
            html.Div(
                "Model workflows will use e.simulation and exported notebook cells.",
                className="ecat-muted-note",
            ),
        ],
    )


def _single_cv_controls(dcc, html):
    return html.Div(
        className="ecat-tab-body",
        children=[
            html.Div(
                className="ecat-control-row ecat-inline-control-row",
                children=[
                    html.Label("CV Index:", htmlFor="ecat-single-index", className="ecat-control-label"),
                    dcc.Input(
                        id="ecat-single-index",
                        type="text",
                        inputMode="numeric",
                        value=0,
                        placeholder="0",
                    ),
                ],
            ),
            html.Div(id="ecat-single-index-status", className="ecat-analysis-index-status"),
            html.Label("Segment", className="ecat-control-label"),
            html.Div(id="ecat-single-segment-mode", hidden=True),
            html.Div(
                id="ecat-single-segment-slider-wrap",
                style={"display": "none"},
                children=dcc.Slider(
                    id="ecat-single-segment-slider",
                    min=1,
                    max=1,
                    step=1,
                    value=1,
                    marks={1: "1"},
                    disabled=True,
                ),
            ),
            html.Div(
                id="ecat-single-segment-text-wrap",
                style={"display": "none"},
                children=dcc.Input(
                    id="ecat-single-segment-text",
                    type="text",
                    inputMode="numeric",
                    placeholder="Segment",
                    className="ecat-full-width",
                ),
            ),
            html.Div(id="ecat-single-segment-status", className="ecat-analysis-index-status"),
            html.Div("Pre-processing", className="ecat-analysis-option-title"),
            dcc.Checklist(
                id="ecat-single-dimensionless-normalize",
                options=[{"label": "Dimensionless normalization", "value": "dimensionless"}],
                value=[],
            ),
            html.Div(
                id="ecat-single-dimensionless-options",
                style={"display": "none"},
                children=[
                    dcc.Dropdown(
                        id="ecat-single-dimensionless-mode",
                        value="homogeneous",
                        clearable=False,
                        searchable=False,
                        options=[
                            {"label": "Homogeneous", "value": "homogeneous"},
                            {"label": "Heterogeneous", "value": "heterogeneous"},
                        ],
                    ),
                    html.Div(
                        className="ecat-two-column-controls",
                        children=[
                            dcc.Input(id="ecat-single-dimensionless-e0", type="text", placeholder="E⁰ / V"),
                            dcc.Input(id="ecat-single-dimensionless-n", type="text", placeholder="n / e-"),
                            dcc.Input(id="ecat-single-dimensionless-temperature", type="text", placeholder="Temperature / K"),
                            dcc.Input(id="ecat-single-dimensionless-d", type="text", placeholder="D / cm² s⁻¹"),
                            dcc.Input(id="ecat-single-dimensionless-c", type="text", placeholder="C / M"),
                            dcc.Dropdown(
                                id="ecat-single-dimensionless-area-mode",
                                value="area_cm2",
                                clearable=False,
                                searchable=False,
                                options=[
                                    {"label": "Area / cm²", "value": "area_cm2"},
                                    {"label": "Radius / mm", "value": "radius_mm"},
                                ],
                            ),
                            dcc.Input(id="ecat-single-dimensionless-area", type="text", placeholder="Area value"),
                        ],
                    ),
                ],
            ),
            dcc.Checklist(
                id="ecat-single-analyses",
                className="ecat-analysis-checklist",
                options=[
                    {"label": "Peak potential", "value": "peak_potential"},
                    {"label": "Peak current", "value": "peak_current"},
                    {"label": "Half peak potential", "value": "half_peak_potential"},
                    {"label": "Half wave potential", "value": "half_wave_potential"},
                ],
                value=["peak_potential", "peak_current", "half_peak_potential", "half_wave_potential"],
            ),
            html.Div(
                className="ecat-control-row ecat-inline-control-row ecat-potential-control-row",
                children=[
                    html.Label(
                        [html.Span("GUESS"), html.Span("POTENTIAL")],
                        htmlFor="ecat-single-guess-potential",
                        className="ecat-control-label ecat-stacked-control-label",
                    ),
                    dcc.Input(
                        id="ecat-single-guess-potential",
                        type="text",
                        inputMode="decimal",
                        placeholder="Optional",
                    ),
                ],
            ),
            html.Div(
                className="ecat-control-row ecat-inline-control-row ecat-potential-control-row",
                children=[
                    html.Label(
                        [html.Span("TANGENT"), html.Span("POTENTIAL")],
                        htmlFor="ecat-single-tangent-potential",
                        className="ecat-control-label ecat-stacked-control-label",
                    ),
                    dcc.Input(
                        id="ecat-single-tangent-potential",
                        type="text",
                        inputMode="decimal",
                        placeholder="Optional",
                    ),
                ],
            ),
            html.Button("Run Single CV", id="ecat-run-single"),
        ],
    )


def _multi_cv_controls(dcc, html):
    return html.Div(
        className="ecat-tab-body",
        children=[
            html.Div(
                className="ecat-preprocessing-panel",
                children=[
                    html.Div("Pre-processing", className="ecat-analysis-option-title"),
                    dcc.Checklist(
                        id="ecat-multi-preprocess-scale",
                        options=[{"label": "Scale current", "value": "scale"}],
                        value=[],
                    ),
                    html.Div(
                        id="ecat-multi-scale-options",
                        style={"display": "none"},
                        children=[
                            html.Label("Scale current", className="ecat-control-label"),
                            dcc.RadioItems(
                                id="ecat-multi-scale-type",
                                options=[
                                    {"label": "Reference", "value": "reference"},
                                    {"label": "Set scale", "value": "manual"},
                                ],
                                value="reference",
                                className="ecat-segmented",
                                inline=True,
                            ),
                            html.Div(
                                id="ecat-multi-scale-reference-fields",
                                children=[
                                    dcc.Input(
                                        id="ecat-multi-scale-reference-index",
                                        type="text",
                                        placeholder="Reference index",
                                        className="ecat-full-width",
                                    ),
                                    dcc.RadioItems(
                                        id="ecat-multi-scale-reference-mode",
                                        options=[
                                            {"label": "Single", "value": "single"},
                                            {"label": "Both", "value": "both"},
                                        ],
                                        value="single",
                                        className="ecat-segmented",
                                        inline=True,
                                    ),
                                    dcc.Input(
                                        id="ecat-multi-scale-segment",
                                        type="text",
                                        placeholder="Reference segment",
                                        className="ecat-full-width",
                                    ),
                                    dcc.Input(
                                        id="ecat-multi-scale-guess-potential",
                                        type="text",
                                        placeholder="Guess potential",
                                        className="ecat-full-width",
                                    ),
                                ],
                            ),
                            html.Div(
                                id="ecat-multi-scale-manual-fields",
                                style={"display": "none"},
                                children=dcc.Input(
                                    id="ecat-multi-scale-factor",
                                    type="text",
                                    placeholder="Scale factor",
                                    className="ecat-full-width",
                                ),
                            ),
                        ],
                    ),
                    html.Label("Normalize", className="ecat-control-label"),
                    dcc.Dropdown(
                        id="ecat-multi-normalize-mode",
                        value="none",
                        clearable=False,
                        searchable=False,
                        options=[
                            {"label": "None", "value": "none"},
                            {"label": "Dimensionless", "value": "dimensionless"},
                            {"label": "Current", "value": "current"},
                        ],
                    ),
                    html.Div(
                        id="ecat-multi-dimensionless-options",
                        style={"display": "none"},
                        children=[
                            html.Label("Dimensionless normalization", className="ecat-control-label"),
                            dcc.Dropdown(
                                id="ecat-multi-dimensionless-mode",
                                value="homogeneous",
                                clearable=False,
                                searchable=False,
                                options=[
                                    {"label": "Homogeneous", "value": "homogeneous"},
                                    {"label": "Heterogeneous", "value": "heterogeneous"},
                                ],
                            ),
                            html.Div(
                                className="ecat-two-column-controls",
                                children=[
                                    dcc.Input(id="ecat-multi-dimensionless-e0", type="text", placeholder="E⁰ / V"),
                                    dcc.Input(id="ecat-multi-dimensionless-n", type="text", placeholder="n / e-"),
                                    dcc.Input(id="ecat-multi-dimensionless-temperature", type="text", placeholder="Temperature / K"),
                                    dcc.Input(id="ecat-multi-dimensionless-d", type="text", placeholder="D / cm² s⁻¹"),
                                    dcc.Input(id="ecat-multi-dimensionless-c", type="text", placeholder="C / M"),
                                    dcc.Dropdown(
                                        id="ecat-multi-dimensionless-area-mode",
                                        value="area_cm2",
                                        clearable=False,
                                        searchable=False,
                                        options=[
                                            {"label": "Area / cm²", "value": "area_cm2"},
                                            {"label": "Radius / mm", "value": "radius_mm"},
                                        ],
                                    ),
                                    dcc.Input(id="ecat-multi-dimensionless-area", type="text", placeholder="Area value"),
                                ],
                            ),
                        ],
                    ),
                    html.Div(
                        id="ecat-multi-current-normalization-options",
                        style={"display": "none"},
                        children=[
                            html.Label("Current normalization", className="ecat-control-label"),
                            dcc.RadioItems(
                                id="ecat-multi-current-normalization-type",
                                options=[
                                    {"label": "Reference", "value": "reference"},
                                    {
                                        "label": html.Span(["Set ", html.I("i"), html.Sub("p"), html.Sup("0")]),
                                        "value": "manual",
                                    },
                                ],
                                value="reference",
                                className="ecat-segmented",
                                inline=True,
                            ),
                            html.Div(
                                id="ecat-multi-current-reference-fields",
                                children=[
                                    dcc.Input(
                                        id="ecat-multi-current-reference-index",
                                        type="text",
                                        placeholder="Reference index",
                                        className="ecat-full-width",
                                    ),
                                    html.Div(
                                        className="ecat-two-column-controls",
                                        children=[
                                            dcc.Input(id="ecat-multi-current-segment", type="text", placeholder="Segment"),
                                            dcc.Input(
                                                id="ecat-multi-current-guess-potential",
                                                type="text",
                                                placeholder="Guess potential",
                                            ),
                                        ],
                                    ),
                                ],
                            ),
                            html.Div(
                                id="ecat-multi-current-manual-fields",
                                style={"display": "none"},
                                children=dcc.Input(
                                    id="ecat-multi-current-ip0",
                                    type="text",
                                    placeholder="iₚ⁰",
                                    className="ecat-full-width",
                                ),
                            ),
                        ],
                    ),
                ],
            ),
            html.Label("Analysis", className="ecat-control-label"),
            dcc.Dropdown(
                id="ecat-multi-analysis",
                value="none",
                clearable=False,
                searchable=False,
                options=[
                    {"label": "None", "value": "none"},
                    {"label": "Fit Peak Potential", "value": "fit_peak_potential"},
                    {"label": "Fit Peak Current", "value": "fit_peak_current"},
                    {"label": "Sevcik Analysis", "value": "sevcik_analysis"},
                    {"label": "Trumpet Analysis", "value": "trumpet_analysis"},
                    {"label": "FOWA", "value": "fowa"},
                    {"label": "Tafel Analysis", "value": "tafel_analysis"},
                ],
            ),
            html.Div(id="ecat-multi-analysis-equations", className="ecat-multi-analysis-equations"),
            html.Div(
                id="ecat-multi-analysis-options",
                style={"display": "none"},
                children=[
                    html.Div(id="ecat-multi-analysis-title", className="ecat-analysis-option-title"),
                    dcc.Input(id="ecat-multi-segment", type="number", min=1, step=1, style={"display": "none"}),
                    _analysis_labeled_input(
                        dcc,
                        html,
                        "Segment(s)",
                        "ecat-multi-segments",
                        placeholder="comma-separated",
                    ),
                    html.Div(
                        id="ecat-multi-guess-wrap",
                        children=[
                            _analysis_labeled_input(
                                dcc,
                                html,
                                "Guess potential",
                                "ecat-multi-guess-potential",
                                "V",
                                placeholder="optional",
                            ),
                        ],
                    ),
                    html.Div(
                        id="ecat-sevcik-options",
                        style={"display": "none"},
                        children=[
                            _analysis_labeled_control(
                                html,
                                "Mode",
                                dcc.Dropdown(
                                    id="ecat-sevcik-mode",
                                    value="homogeneous",
                                    clearable=False,
                                    searchable=False,
                                    options=[
                                        {"label": "Homogeneous", "value": "homogeneous"},
                                        {"label": "Heterogeneous", "value": "heterogeneous"},
                                    ],
                                ),
                            ),
                        ],
                    ),
                    html.Div(
                        id="ecat-multi-fit-wrap",
                        children=[
                            _analysis_labeled_control(
                                html,
                                "X axis",
                                dcc.Dropdown(
                                    id="ecat-multi-x-axis",
                                    value="auto",
                                    clearable=False,
                                    searchable=False,
                                    options=[
                                        {"label": "Auto", "value": "auto"},
                                        {"label": "Scan rate", "value": "scan rate"},
                                        {"label": "Concentration", "value": "concentration"},
                                    ],
                                ),
                            ),
                            _analysis_labeled_control(
                                html,
                                "Fit model",
                                dcc.Dropdown(
                                    id="ecat-multi-fit-model",
                                    value="linear",
                                    clearable=False,
                                    searchable=False,
                                    options=[
                                        {"label": "Linear", "value": "linear"},
                                        {"label": "Power", "value": "power"},
                                        {"label": "Exponential", "value": "exponential"},
                                    ],
                                ),
                            ),
                        ],
                    ),
                    html.Div(
                        id="ecat-fowa-options",
                        style={"display": "none"},
                        children=[
                            _analysis_labeled_input(dcc, html, "Non-catalytic CV index", "ecat-fowa-reference-index", placeholder="table index"),
                            _analysis_labeled_control(
                                html,
                                "Redox mode",
                                dcc.Dropdown(
                                    id="ecat-fowa-redox-mode",
                                    value="manual",
                                    clearable=False,
                                    searchable=False,
                                    options=[
                                        {"label": "Manual", "value": "manual"},
                                        {"label": "Half wave", "value": "half wave"},
                                        {"label": "Half peak", "value": "half peak"},
                                    ],
                                ),
                            ),
                            _analysis_labeled_input(dcc, html, "Redox potential", "ecat-fowa-redox-potential", "V", placeholder="optional"),
                            _analysis_labeled_control(
                                html,
                                "Fit basis",
                                dcc.Dropdown(
                                    id="ecat-fowa-fit-basis",
                                    value="y",
                                    clearable=False,
                                    searchable=False,
                                    options=[
                                        {"label": "y", "value": "y"},
                                        {"label": "x", "value": "x"},
                                    ],
                                ),
                            ),
                            html.Div(
                                className="ecat-two-column-controls",
                                children=[
                                    _analysis_labeled_input(dcc, html, "Fit start", "ecat-fowa-fit-range-start", value="0.1"),
                                    _analysis_labeled_input(dcc, html, "Fit end", "ecat-fowa-fit-range-end", value="0.5"),
                                ],
                            ),
                            _analysis_labeled_control(
                                html,
                                "Diagnostic y axis",
                                dcc.Dropdown(
                                    id="ecat-fowa-diagnostic-y-axis",
                                    value="i/ip0",
                                    clearable=False,
                                    searchable=False,
                                    options=[
                                        {"label": "i/ip0", "value": "i/ip0"},
                                        {"label": "Current", "value": "current"},
                                    ],
                                ),
                            ),
                            _analysis_labeled_input(dcc, html, "Minimum fit points", "ecat-fowa-min-fit-points", value="50"),
                            _analysis_labeled_input(dcc, html, "Minimum R²", "ecat-fowa-min-r2", value="0.95"),
                        ],
                    ),
                    html.Div(
                        id="ecat-tafel-options",
                        style={"display": "none"},
                        children=[
                            _analysis_labeled_input(dcc, html, "CV index", "ecat-tafel-index", placeholder="first selected CV"),
                            _analysis_labeled_input(dcc, html, "TOF max", "ecat-tafel-tof-max"),
                            _analysis_labeled_input(dcc, html, "Thermodynamic potential", "ecat-tafel-thermo-potential", "V"),
                            _analysis_labeled_input(dcc, html, "Redox potential", "ecat-tafel-redox-potential", "V"),
                            html.Div(
                                className="ecat-two-column-controls",
                                children=[
                                    _analysis_labeled_input(dcc, html, "η start", "ecat-tafel-overpotential-start", "V", value="0"),
                                    _analysis_labeled_input(dcc, html, "η end", "ecat-tafel-overpotential-end", "V", value="1"),
                                ],
                            ),
                            _analysis_labeled_input(dcc, html, "Color", "ecat-tafel-color", value="black"),
                        ],
                    ),
                    dcc.Checklist(
                        id="ecat-multi-toggles",
                        options=[
                            {"label": "Fit", "value": "fit"},
                            {"label": "Plot fit", "value": "plot_fit"},
                            {"label": "Plot diagnostics", "value": "plot_all"},
                        ],
                        value=["fit", "plot_fit", "plot_all"],
                    ),
                ],
            ),
            html.Button("Run Analysis", id="ecat-run-multi-analysis", className="ecat-full-width"),
        ],
    )


def _ca_controls(dcc, html):
    return html.Div(
        className="ecat-tab-body",
        children=[
            html.Div(
                className="ecat-control-row ecat-inline-control-row",
                children=[
                    html.Label("CA Index:", htmlFor="ecat-ca-index", className="ecat-control-label"),
                    dcc.Input(
                        id="ecat-ca-index",
                        type="text",
                        inputMode="numeric",
                        value="",
                        placeholder="0",
                    ),
                ],
            ),
            html.Div(id="ecat-ca-index-status", className="ecat-analysis-index-status"),
            dcc.Checklist(
                id="ecat-ca-analyses",
                className="ecat-analysis-checklist",
                options=[
                    {"label": "Stats", "value": "stats"},
                    {"label": "Current plot", "value": "plot"},
                    {"label": "Cumulative charge", "value": "charge"},
                    {"label": "Current + charge overlay", "value": "current_charge_overlay"},
                    {"label": "Baseline-corrected charge", "value": "baseline_charge"},
                    {"label": "Time at charge", "value": "time_at_charge"},
                ],
                value=["plot", "charge", "current_charge_overlay"],
            ),
            html.Div(
                className="ecat-control-row ecat-inline-control-row",
                children=[
                    html.Label(
                        "Baseline tail fraction:",
                        htmlFor="ecat-ca-baseline-tail-fraction",
                        className="ecat-control-label",
                    ),
                    dcc.Input(
                        id="ecat-ca-baseline-tail-fraction",
                        type="number",
                        min=0,
                        max=1,
                        step=0.01,
                        value=0.05,
                        placeholder="0.05",
                    ),
                ],
            ),
            html.Div(
                className="ecat-control-row ecat-inline-control-row",
                children=[
                    html.Label("Target charge (C):", htmlFor="ecat-ca-target-charge", className="ecat-control-label"),
                    dcc.Input(
                        id="ecat-ca-target-charge",
                        type="number",
                        min=0,
                        step=0.01,
                        value=0.75,
                        placeholder="0.75",
                    ),
                ],
            ),
            dcc.Checklist(
                id="ecat-ca-target-options",
                options=[{"label": "Show CA trace with target", "value": "plot_ca"}],
                value=["plot_ca"],
            ),
            html.Button("Run CA", id="ecat-run-ca"),
        ],
    )


def _cp_controls(dcc, html):
    return html.Div(
        className="ecat-tab-body",
        children=[
            html.Div(
                className="ecat-control-row ecat-inline-control-row",
                children=[
                    html.Label("CP Index:", htmlFor="ecat-cp-index", className="ecat-control-label"),
                    dcc.Input(
                        id="ecat-cp-index",
                        type="text",
                        inputMode="numeric",
                        value="",
                        placeholder="0",
                    ),
                ],
            ),
            html.Div(id="ecat-cp-index-status", className="ecat-analysis-index-status"),
            dcc.Checklist(
                id="ecat-cp-analyses",
                className="ecat-analysis-checklist",
                options=[
                    {"label": "Stats", "value": "stats"},
                    {"label": "Cycle info", "value": "cycle_info"},
                    {"label": "Potential plot", "value": "plot"},
                    {"label": "Cycling performance", "value": "cycling_plot"},
                    {"label": "Cycle plot", "value": "plot_cycles"},
                ],
                value=["stats", "cycle_info", "plot"],
            ),
            dcc.Checklist(
                id="ecat-cp-percent-capacity",
                options=[{"label": "Percent capacity", "value": "percent_capacity"}],
                value=[],
            ),
            html.Div(
                className="ecat-control-row ecat-inline-control-row",
                children=[
                    html.Label("Capacity mode:", htmlFor="ecat-cp-capacity-mode", className="ecat-control-label"),
                    dcc.Dropdown(
                        id="ecat-cp-capacity-mode",
                        value="both",
                        clearable=False,
                        searchable=False,
                        options=[
                            {"label": "Both", "value": "both"},
                            {"label": "Charge", "value": "charge"},
                            {"label": "Discharge", "value": "discharge"},
                        ],
                    ),
                ],
            ),
            html.Div(
                className="ecat-control-row ecat-inline-control-row",
                children=[
                    html.Label("Efficiency mode:", htmlFor="ecat-cp-efficiency-mode", className="ecat-control-label"),
                    dcc.Dropdown(
                        id="ecat-cp-efficiency-mode",
                        value="both",
                        clearable=False,
                        searchable=False,
                        options=[
                            {"label": "Both", "value": "both"},
                            {"label": "Coulombic", "value": "coulombic"},
                            {"label": "Energy", "value": "energy"},
                        ],
                    ),
                ],
            ),
            html.Div(
                className="ecat-control-row ecat-inline-control-row",
                children=[
                    html.Label("Cycles:", className="ecat-control-label"),
                    dcc.Input(id="ecat-cp-cycles-start", type="number", min=1, step=1, value=1, placeholder="Start"),
                    dcc.Input(id="ecat-cp-cycles-end", type="number", min=1, step=1, value=100, placeholder="End"),
                    dcc.Input(id="ecat-cp-cycles-step", type="number", min=1, step=1, value=10, placeholder="Step"),
                ],
            ),
            html.Div(
                className="ecat-control-row ecat-inline-control-row",
                children=[
                    html.Label("Segment:", htmlFor="ecat-cp-cycle-segment", className="ecat-control-label"),
                    dcc.Dropdown(
                        id="ecat-cp-cycle-segment",
                        value="both",
                        clearable=False,
                        searchable=False,
                        options=[
                            {"label": "Both", "value": "both"},
                            {"label": "Charge", "value": "charge"},
                            {"label": "Discharge", "value": "discharge"},
                            {"label": "Full", "value": "full"},
                        ],
                    ),
                ],
            ),
            html.Div(
                className="ecat-control-row ecat-inline-control-row",
                children=[
                    html.Label("X axis:", htmlFor="ecat-cp-cycle-x-axis", className="ecat-control-label"),
                    dcc.Dropdown(
                        id="ecat-cp-cycle-x-axis",
                        value="capacity",
                        clearable=False,
                        searchable=False,
                        options=[
                            {"label": "Capacity", "value": "capacity"},
                            {"label": "Time", "value": "time"},
                        ],
                    ),
                ],
            ),
            html.Button("Run CP", id="ecat-run-cp"),
        ],
    )


def _export_tab(dcc, html, initial_state=None):
    initial_state = initial_state or {}
    return html.Div(
        className="ecat-tab-body",
        children=[
            html.Div("Export selected data, plots, and reproducible notebook code.", className="ecat-tab-intro"),
            _plot_section(html, "Data Export"),
            dcc.Input(
                id="ecat-export-filename",
                type="text",
                value="",
                placeholder="Export CSV filename",
                className="ecat-full-width",
            ),
            html.Button("Export CSV", id="ecat-export-csv", className="ecat-full-width"),
            _plot_section(html, "Notebook Export"),
            html.Button("Download Python Notebook", id="ecat-download-code-button", className="ecat-full-width"),
            dcc.Download(id="ecat-download-code"),
            _plot_section(html, "Python Preview"),
            html.Div(
                id="ecat-export-code",
                children=dcc.Textarea(
                    id="ecat-code-preview",
                    className="ecat-code",
                    value=initial_state.get("code", ""),
                ),
            ),
        ],
    )
