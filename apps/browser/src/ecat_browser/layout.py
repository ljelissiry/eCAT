"""Dash layout for the eCAT browser app."""

from .config import BrowserAppConfig
from .table import ag_grid_column_defs, selected_column_values, selected_grid_rows_for_ids

TAB_IDS = ("import", "plotting", "analysis", "export")


def _dash():
    try:
        import dash_ag_grid as dag
        from dash import dcc, html
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "The eCAT browser app requires Dash and dash-ag-grid. Install with `pip install -e .[app]`."
        ) from exc
    return dag, dcc, html


def create_layout(config: BrowserAppConfig | None = None, initial_state: dict | None = None):
    config = config or BrowserAppConfig.from_env()
    initial_state = initial_state or {}
    initial_table = initial_state.get("table", {"columns": [], "data": []})
    initial_column_options = initial_state.get("column_options")
    initial_plot = initial_state.get("plot")
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
            dcc.Store(
                id="ecat-plot-options-store",
                data={
                    "legend": True,
                    "legend mode": "colorbar",
                    "color mode": "auto",
                    "title": "auto",
                    "plot style": "line",
                    "plot convention": "US",
                    "_format": "png",
                    "_dpi": 150,
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
                                src="/assets/ecat-logo.svg",
                                className="ecat-header-logo",
                                alt="eCAT logo",
                            ),
                            html.Div(
                                className="ecat-header-copy",
                                children=[
                                    html.Div("eCAT Workbench", className="ecat-app-title"),
                                    html.Div("Browser analysis workspace", className="ecat-app-subtitle"),
                                ],
                            ),
                        ],
                    ),
                    html.Button("About", id="ecat-about-button", className="ecat-header-button"),
                ],
            ),
            html.Div(
                id="ecat-about-panel",
                className="ecat-about-panel",
                hidden=True,
                children=[
                    html.Div("About eCAT Workbench", className="ecat-about-title"),
                    html.P(
                        "eCAT, short for electroCatalysis Analysis Tools, is a Python package for loading, organizing, plotting, and analyzing electrochemical data from common lab workflows."
                    ),
                    html.P(
                        "This browser app is a local-first Dash interface for the eCAT package. It keeps the notebook-facing API intact while adding import tables, plotting controls, CV/CA/CP analysis panels, and reproducible Python export."
                    ),
                    html.Div(
                        className="ecat-about-meta",
                        children=[
                            html.Div([html.Strong("Package"), html.Span("ecat 0.1.0b2")]),
                            html.Div([html.Strong("Author"), html.Span("Luke Elissiry")]),
                            html.Div([html.Strong("License"), html.Span("MIT License, copyright 2026 Luke Elissiry")]),
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
                                        value="import",
                                        children=[
                                            dcc.Tab(
                                                label=_tab_label(html, "Import", "ecat_icon_import.svg"),
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
                                                label=_tab_label(html, "Plotting", "ecat_icon_plotting.svg"),
                                                value="plotting",
                                                children=_plotting_tab(dcc, html),
                                            ),
                                            dcc.Tab(
                                                label=_tab_label(html, "Analysis", "ecat_icon_analysis.svg"),
                                                value="analysis",
                                                children=_analysis_tab(dcc, html),
                                            ),
                                            dcc.Tab(
                                                label=_tab_label(html, "Export", "ecat_icon_export.svg"),
                                                value="export",
                                                children=_export_tab(dcc, html, initial_state),
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
                                        className="ecat-plot-toolbar",
                                        children=[
                                            html.Button("Replot", id="ecat-replot"),
                                            html.Button("Save", id="ecat-save-plot"),
                                            dcc.Download(id="ecat-download-plot"),
                                        ],
                                    ),
                                    dcc.Loading(
                                        id="ecat-plot-loading",
                                        type="circle",
                                        children=html.Div(
                                        id="ecat-default-plot",
                                        children=html.Img(src=initial_plot, className="ecat-plot") if initial_plot else "",
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
            html.Img(
                src=f"/assets/{icon_filename}",
                className="ecat-tab-symbol",
                alt=f"{name} symbol",
                title=name,
            ),
            html.Span(name, className="ecat-tab-text"),
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
                        "paginationPageSize": 12,
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
    return html.Div(
        className=f"ecat-control-row {class_name}".strip(),
        children=[
            html.Label(label, className="ecat-control-label"),
            html.Div(control, className="ecat-control-field"),
        ],
    )


def _plot_section(html, title):
    return html.Div(title, className="ecat-plot-section-heading")


def _plotting_tab(dcc, html):
    return html.Div(
        className="ecat-tab-body",
        children=[
            _plot_section(html, "Plotting Style"),
            _control_row(
                html,
                "Trace style",
                dcc.Dropdown(
                    id="ecat-plot-style",
                    value="line",
                    clearable=False,
                    searchable=False,
                    options=[
                        {"label": "Line", "value": "line"},
                        {"label": "Scatter", "value": "scatter"},
                        {"label": "Line + markers", "value": "line+markers"},
                    ],
                ),
            ),
            _control_row(
                html,
                "Convention",
                dcc.RadioItems(
                    id="ecat-plot-convention",
                    options=[
                        {"label": "US", "value": "US"},
                        {"label": "IUPAC", "value": "IUPAC"},
                    ],
                    value="US",
                    className="ecat-segmented",
                    inline=True,
                ),
            ),
            _plot_section(html, "Title"),
            _control_row(
                html,
                "Mode",
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
            _plot_section(html, "Legend"),
            _control_row(
                html,
                "Visibility",
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
            _plot_section(html, "Display"),
            _control_row(
                html,
                "Options",
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
                className="ecat-two-column-controls",
                children=[
                    dcc.Input(id="ecat-plot-trim-min", type="number", placeholder="Min potential"),
                    dcc.Input(id="ecat-plot-trim-max", type="number", placeholder="Max potential"),
                ],
            ),
            _plot_section(html, "Offset"),
            _control_row(
                html,
                "Vertical offset",
                dcc.Input(
                    id="ecat-plot-offset",
                    type="number",
                    placeholder="Vertical offset",
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
            _plot_section(html, "Save"),
            _control_row(
                html,
                "Output",
                html.Div(
                    className="ecat-two-column-controls",
                    children=[
                        dcc.Dropdown(
                            id="ecat-plot-format",
                            value="png",
                            clearable=False,
                            searchable=False,
                            options=[
                                {"label": "PNG", "value": "png"},
                                {"label": "SVG", "value": "svg"},
                                {"label": "PDF", "value": "pdf"},
                            ],
                        ),
                        dcc.Input(id="ecat-plot-dpi", type="number", min=72, step=1, value=150, placeholder="DPI"),
                    ],
                ),
            ),
            html.Div(
                className="ecat-plot-toolbar",
                children=[
                    html.Button("Replot", id="ecat-plotting-replot"),
                    html.Button("Save", id="ecat-plotting-save-plot"),
                ],
            ),
        ],
    )


def _analysis_tab(dcc, html):
    return html.Div(
        className="ecat-tab-body",
        children=[
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
            html.Label("Analysis", className="ecat-control-label"),
            dcc.Dropdown(
                id="ecat-multi-analysis",
                value=None,
                clearable=True,
                searchable=False,
                placeholder="Select analysis",
                options=[
                    {"label": "Fit Peak Potential", "value": "fit_peak_potential"},
                    {"label": "Fit Peak Current", "value": "fit_peak_current"},
                    {"label": "Sevcik Analysis", "value": "sevcik_analysis"},
                    {"label": "Trumpet Analysis", "value": "trumpet_analysis"},
                    {"label": "FOWA", "value": "fowa"},
                    {"label": "Tafel Analysis", "value": "tafel_analysis"},
                ],
            ),
            html.Div(
                id="ecat-multi-analysis-options",
                style={"display": "none"},
                children=[
                    html.Div(id="ecat-multi-analysis-title", className="ecat-analysis-option-title"),
                    html.Label("Segment(s)", className="ecat-control-label"),
                    dcc.Input(id="ecat-multi-segment", type="number", min=1, step=1, style={"display": "none"}),
                    dcc.Input(
                        id="ecat-multi-segments",
                        type="text",
                        placeholder="Segment(s), comma-separated",
                    ),
                    html.Div(
                        id="ecat-multi-guess-wrap",
                        children=[
                            html.Label("Guess potential", className="ecat-control-label"),
                            dcc.Input(id="ecat-multi-guess-potential", type="text", placeholder="Guess potential"),
                        ],
                    ),
                    html.Div(
                        id="ecat-multi-fit-wrap",
                        children=[
                            html.Label("X axis", className="ecat-control-label"),
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
                            html.Label("Fit model", className="ecat-control-label"),
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
                        ],
                    ),
                    html.Div(
                        id="ecat-fowa-options",
                        style={"display": "none"},
                        children=[
                            html.Label("Non-catalytic CV index", className="ecat-control-label"),
                            dcc.Input(id="ecat-fowa-reference-index", type="text", placeholder="Table index"),
                            html.Label("Redox mode", className="ecat-control-label"),
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
                            html.Label("Redox potential", className="ecat-control-label"),
                            dcc.Input(id="ecat-fowa-redox-potential", type="text", placeholder="Redox potential"),
                            html.Label("Fit basis", className="ecat-control-label"),
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
                            html.Label("Fit range", className="ecat-control-label"),
                            html.Div(
                                className="ecat-two-column-controls",
                                children=[
                                    dcc.Input(id="ecat-fowa-fit-range-start", type="text", value="0.1", placeholder="Start"),
                                    dcc.Input(id="ecat-fowa-fit-range-end", type="text", value="0.5", placeholder="End"),
                                ],
                            ),
                            html.Label("Diagnostic y axis", className="ecat-control-label"),
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
                            html.Label("Minimum fit points", className="ecat-control-label"),
                            dcc.Input(id="ecat-fowa-min-fit-points", type="text", value="50", placeholder="Minimum fit points"),
                            html.Label("Minimum R2", className="ecat-control-label"),
                            dcc.Input(id="ecat-fowa-min-r2", type="text", value="0.95", placeholder="Minimum R2"),
                        ],
                    ),
                    html.Div(
                        id="ecat-tafel-options",
                        style={"display": "none"},
                        children=[
                            html.Label("CV index", className="ecat-control-label"),
                            dcc.Input(id="ecat-tafel-index", type="text", placeholder="First selected CV"),
                            html.Label("TOF max", className="ecat-control-label"),
                            dcc.Input(id="ecat-tafel-tof-max", type="text", placeholder="TOF max"),
                            html.Label("Thermodynamic potential", className="ecat-control-label"),
                            dcc.Input(id="ecat-tafel-thermo-potential", type="text", placeholder="Thermodynamic potential"),
                            html.Label("Redox potential", className="ecat-control-label"),
                            dcc.Input(id="ecat-tafel-redox-potential", type="text", placeholder="Redox potential"),
                            html.Label("Overpotential range", className="ecat-control-label"),
                            html.Div(
                                className="ecat-two-column-controls",
                                children=[
                                    dcc.Input(id="ecat-tafel-overpotential-start", type="text", value="0", placeholder="Start"),
                                    dcc.Input(id="ecat-tafel-overpotential-end", type="text", value="1", placeholder="End"),
                                ],
                            ),
                            html.Label("Color", className="ecat-control-label"),
                            dcc.Input(id="ecat-tafel-color", type="text", value="black", placeholder="Color"),
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
                    html.Button("Run Analysis", id="ecat-run-multi-analysis", className="ecat-full-width"),
                ],
            ),
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
            dcc.Input(
                id="ecat-export-filename",
                type="text",
                value="",
                placeholder="Export CSV filename",
                className="ecat-full-width",
            ),
            html.Button("Export CSV", id="ecat-export-csv", className="ecat-full-width"),
            html.Button("Download Python", id="ecat-download-code-button", className="ecat-full-width"),
            dcc.Download(id="ecat-download-code"),
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
