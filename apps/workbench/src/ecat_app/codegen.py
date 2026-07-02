"""Generate reproducible public-API eCAT Python workflows."""

from __future__ import annotations

from pprint import pformat
import json

from .workflow import AppWorkflow


def _literal(value) -> str:
    if isinstance(value, str):
        return json.dumps(value)
    return pformat(value, width=88, sort_dicts=False)


SINGLE_CV_ANALYSES = {"peak_potential", "peak_current", "half_peak_potential", "half_wave_potential"}
MULTI_CV_ANALYSES = {"fit_peak_potential", "fit_peak_current", "sevcik_analysis", "trumpet_analysis", "fowa", "tafel_analysis"}


def _model_action_was_run(result) -> bool:
    result = dict(result or {})
    status = str(result.get("status") or "").strip().lower()
    return bool(result) and status not in {"", "placeholder", "blocked"}


def _append_cv_program_code(lines: list[str], program_settings: dict[str, object] | None) -> None:
    lines.extend(
        [
            "",
            "# CV Program",
            "program_input = e.simulation.cv_program(",
            "    program_settings.get('Ei', 0.0),",
            "    E_low=program_settings.get('E_low', -1.5),",
            "    E_high=program_settings.get('E_high'),",
            "    Ef=program_settings.get('Ef'),",
            "    scan_rate=program_settings.get('scan_rate', 0.1),",
            "    segments=program_settings.get('segments', 2),",
            "    points_per_segment=program_settings.get('points_per_segment', 300),",
            "    quiet_time=program_settings.get('quiet_time', 0.0),",
            ")",
            'program_ax = program_input.plot({"title": "Simulation Input", "plot": False})',
        ]
    )


def _append_model_code(lines: list[str], model_options: dict[str, object] | None) -> None:
    options = dict(model_options or {})
    if not options.get("mechanism_valid"):
        return

    mechanism = options.get("mechanism") or options.get("mechanism_preset") or "E"
    parameters = options.get("parameters") or []
    cell_parameters = options.get("cell_parameters") or []
    simulation_params = options.get("simulation_params") or options.get("params") or {}
    program_settings = options.get("program_settings") or {}
    cv_data_settings = options.get("cv_data_settings") or {}
    condition_settings = options.get("condition_settings") or {}
    if not parameters:
        parameters = [
            {
                "name": "mechanism",
                "initial": mechanism,
                "lower": "",
                "upper": "",
                "vary": False,
            }
        ]

    lines.extend(
        [
            "",
            "# Model Setup",
            f"mechanism_text = {_literal(mechanism)}",
            "compiled_mechanism = e.simulation.compile_mechanism(mechanism_text)",
            f"program_settings = {_literal(program_settings)}",
            f"cv_data_settings = {_literal(cv_data_settings)}",
            f"condition_settings = {_literal(condition_settings)}",
            f"model_parameters = {_literal(parameters)}",
            f"cell_parameters = {_literal(cell_parameters)}",
            f"simulation_params = {_literal(simulation_params)}",
        ]
    )

    if _model_action_was_run(options.get("program_result")):
        _append_cv_program_code(lines, program_settings)

    if _model_action_was_run(options.get("simulation_result")):
        simulation_mode = options.get("simulate_mode") or "scratch"
        lines.extend(
            [
                "",
                "# Simulation",
                f"simulation_mode = {_literal(simulation_mode)}",
                f"over_conditions = {_literal(bool(options.get('over_conditions')))}",
            ]
        )
        if str(simulation_mode).strip().lower() == "cv":
            lines.extend(
                [
                    "cv_data_options = dict(cv_data_settings)",
                    "cv_index = cv_data_options.pop('cv_index', 0)",
                    "simulation_input = e.simulation.cv_data(data[cv_index], cv_data_options)",
                ]
            )
        else:
            lines.extend(
                [
                    "simulation_input = e.simulation.cv_program(",
                    "    program_settings.get('Ei', 0.0),",
                    "    E_low=program_settings.get('E_low', -1.5),",
                    "    E_high=program_settings.get('E_high'),",
                    "    Ef=program_settings.get('Ef'),",
                    "    scan_rate=program_settings.get('scan_rate', 0.1),",
                    "    segments=program_settings.get('segments', 2),",
                    "    points_per_segment=program_settings.get('points_per_segment', 300),",
                    "    quiet_time=program_settings.get('quiet_time', 0.0),",
                    ")",
                ]
            )
        if options.get("over_conditions"):
            lines.extend(
                [
                    "from copy import deepcopy",
                    "",
                    "condition_axis = condition_settings.get('condition_axis', 'scan_rate')",
                    "if condition_settings.get('condition_values'):",
                    "    condition_values = [",
                    "        float(value.strip())",
                    "        for value in str(condition_settings.get('condition_values', '')).replace(';', ',').split(',')",
                    "        if value.strip()",
                    "    ]",
                    "else:",
                    "    condition_min = float(condition_settings.get('condition_min', 0.05))",
                    "    condition_max = float(condition_settings.get('condition_max', 0.2))",
                    "    condition_count = max(1, int(condition_settings.get('condition_count', 3)))",
                    "    if condition_count == 1:",
                    "        condition_values = [condition_min]",
                    "    else:",
                    "        condition_values = [",
                    "            condition_min + (condition_max - condition_min) * index / (condition_count - 1)",
                    "            for index in range(condition_count)",
                    "        ]",
                    "simulation_results = []",
                    "for condition_value in condition_values:",
                    "    condition_input = simulation_input",
                    "    condition_params = deepcopy(simulation_params)",
                    "    if condition_axis == 'temperature':",
                    "        condition_params.setdefault('cell', {})['T'] = condition_value",
                    "    elif condition_axis == 'concentration':",
                    "        bulk = condition_params.setdefault('concentrations', {}).setdefault('bulk', {})",
                    "        species = condition_settings.get('condition_species') or next((key for key in bulk if str(key).lower() != 'b'), next(iter(bulk), 'a'))",
                    "        bulk[species] = condition_value",
                    "    elif condition_axis == 'scan_rate':",
                    "        condition_input = simulation_input.with_scan_rate(condition_value)",
                    "    simulation_results.append(",
                    "        e.simulation.simulate_cv(",
                    "            condition_input,",
                    "            mechanism_text,",
                    "            condition_params,",
                    "            options={'plot': True},",
                    "        )",
                    "    )",
                ]
            )
        else:
            lines.extend(
                [
                    "simulation_result = e.simulation.simulate_cv(",
                    "    simulation_input,",
                    "    mechanism_text,",
                    "    simulation_params,",
                    "    options={'plot': True},",
                    ")",
                ]
            )

    if _model_action_was_run(options.get("fit_result")):
        fit_mode = options.get("fit_mode") or "single"
        fit_spec = options.get("fit_spec") or options.get("fit") or {}
        fit_cv_index = int(options.get("fit_cv_index", 0) or 0)
        lines.extend(
            [
                "",
                "# Fit",
                f"fit_mode = {_literal(fit_mode)}",
                f"fit_spec = {_literal(fit_spec)}",
            ]
        )
        if str(fit_mode).strip().lower() == "single":
            lines.extend(
                [
                    f"fit_cv_index = {fit_cv_index}",
                    "fit_cv_data_options = dict(cv_data_settings)",
                    "fit_cv_data_options.pop('cv_index', None)",
                    "fit_result = e.simulation.fit_cv(",
                    "    data[fit_cv_index],",
                    "    mechanism_text,",
                    "    simulation_params,",
                    "    fit=fit_spec,",
                    "    options={",
                    "        \"cv data\": fit_cv_data_options,",
                    "        \"plot\": True,",
                    "    },",
                    ")",
                ]
            )
        else:
            lines.extend(
                [
                    "# Multiple-CV fitting is not wired in the eCAT app yet.",
                    "fit_result = None",
                ]
            )


def _analysis_actions_for_codegen(workflow: AppWorkflow, raw_plot_options: dict[str, object]) -> list[dict[str, object]]:
    if workflow.analysis_actions:
        return [dict(action) for action in workflow.analysis_actions]
    actions: list[dict[str, object]] = []
    single = [analysis for analysis in workflow.analyses if analysis in SINGLE_CV_ANALYSES]
    if single:
        actions.append(
            {
                "kind": "cv_single",
                "selected_index": workflow.selected_index,
                "analyses": single,
                "options": {},
            }
        )
    for analysis in workflow.analyses:
        if analysis in MULTI_CV_ANALYSES:
            actions.append(
                {
                    "kind": "cv_multi",
                    "analysis": analysis,
                    "analyses": [analysis],
                    "options": dict(raw_plot_options.get(f"{analysis} options", {})),
                }
            )
    return actions


def _analysis_options_without_preprocessing(options: dict[str, object] | None) -> dict[str, object]:
    options = dict(options or {})
    options.pop("preprocessing", None)
    return options


def _append_single_cv_analysis_code(lines: list[str], action: dict[str, object], plot_options: dict[str, object]) -> None:
    selected_index = int(action.get("selected_index", 0) or 0)
    options = {
        key: value
        for key, value in dict(action.get("options") or {}).items()
        if key in {"guess potential", "tangent potential", "segment"}
    }
    lines.extend(
        [
            "",
            "# Single CV Analysis",
            f"single_cv = data[{selected_index}]",
            "single_cv_ax = single_cv.plot(plot_options)",
        ]
    )
    x_label = plot_options.get("x label")
    y_label = plot_options.get("y label")
    if x_label is False:
        lines.append('single_cv_ax.set_xlabel("")')
    elif x_label not in (None, ""):
        lines.append(f"single_cv_ax.set_xlabel({_literal(x_label)})")
    if y_label is False:
        lines.append('single_cv_ax.set_ylabel("")')
    elif y_label not in (None, ""):
        lines.append(f"single_cv_ax.set_ylabel({_literal(y_label)})")
    method_options = {
        **options,
        "plot": True,
        "plot CV": False,
        "new plot": False,
        "plot all": True,
        "print": True,
    }
    for analysis in action.get("analyses") or []:
        if analysis not in SINGLE_CV_ANALYSES:
            continue
        lines.extend(
            [
                "",
                f"# Analysis: {analysis.replace('_', ' ').title()}",
                f"{analysis}_result = single_cv.{analysis}({_literal(method_options)})",
            ]
        )


def _append_multi_cv_analysis_code(lines: list[str], action: dict[str, object], current_name: str) -> None:
    analysis = str(action.get("analysis") or (action.get("analyses") or [""])[0])
    if analysis == "none" or analysis not in MULTI_CV_ANALYSES:
        return
    options = _analysis_options_without_preprocessing(action.get("options"))
    lines.extend(["", f"# Analysis: {analysis.replace('_', ' ').title()}"])
    if analysis == "fowa" and "non-catalytic cv index" in options:
        reference_index = options.pop("non-catalytic cv index")
        lines.append(f"non_catalytic_cv = data[{int(reference_index)}]")
        options["non-catalytic cv"] = "non_catalytic_cv"
    if analysis == "tafel_analysis":
        cv_index = int(options.pop("cv index", 0))
        tof_max = options.pop("TOF max", "TOF_max")
        thermodynamic_potential = options.pop("thermodynamic potential", "thermodynamic_potential")
        redox_potential = options.pop("redox potential", "redox_potential")
        for key in ("plot", "plot all", "plot fit", "new plot", "print"):
            options.pop(key, None)
        lines.extend(
            [
                f"tafel_cv = data[{cv_index}]",
                "tafel_result = e.tafel_analysis(",
                "    tafel_cv,",
                f"    {_literal(tof_max)},",
                f"    {_literal(thermodynamic_potential)},",
                f"    {_literal(redox_potential)},",
                "    {",
            ]
        )
        for key, value in options.items():
            lines.append(f'        "{key}": {_literal(value)},')
        lines.extend(["    },", ")"])
        return
    lines.append(f"{analysis}_result = e.{analysis}({current_name}, {{")
    for key, value in options.items():
        rendered = value if key == "non-catalytic cv" and value == "non_catalytic_cv" else _literal(value)
        lines.append(f'    "{key}": {rendered},')
    lines.extend(['    "print": False,', "})"])


def _append_ca_analysis_code(lines: list[str], action: dict[str, object]) -> None:
    selected_index = int(action.get("selected_index", 0) or 0)
    options = dict(action.get("options") or {})
    plot_options = dict(options.get("plot options") or {})
    target_charge = options.get("target charge", 0.75)
    plot_ca = bool(options.get("plot ca", True))
    baseline_tail_fraction = options.get("baseline tail fraction", 0.05)
    lines.extend(
        [
            "",
            "# CA Analysis",
            f"ca_obj = data[{selected_index}]",
            f"ca_plot_options = {_literal(plot_options)}",
        ]
    )
    for analysis in action.get("analyses") or []:
        if analysis == "stats":
            lines.append("ca_stats_result = ca_obj.stats()")
        elif analysis == "plot":
            lines.append('ca_plot_result = ca_obj.plot({**ca_plot_options, "print": False})')
        elif analysis == "charge":
            lines.append('ca_charge_result = ca_obj.charge({**ca_plot_options, "plot": True, "print": False})')
        elif analysis == "current_charge_overlay":
            lines.append('ca_current_charge_overlay_result = ca_obj.plot({**ca_plot_options, "plot charge": True, "print": False})')
        elif analysis == "baseline_charge":
            lines.extend(
                [
                    "ca_baseline_charge_result = ca_obj.charge({",
                    "    **ca_plot_options,",
                    '    "baseline correction": True,',
                    f'    "baseline tail fraction": {_literal(baseline_tail_fraction)},',
                    '    "plot": True,',
                    '    "print": False,',
                    "})",
                ]
            )
        elif analysis == "time_at_charge":
            lines.extend(
                [
                    "ca_time_at_charge_result = ca_obj.time_at_charge({",
                    "    **ca_plot_options,",
                    f'    "target charge": {_literal(target_charge)},',
                    f'    "plot ca": {_literal(plot_ca)},',
                    '    "plot": True,',
                    '    "print": False,',
                    "})",
                ]
            )


def _append_cp_analysis_code(lines: list[str], action: dict[str, object]) -> None:
    selected_index = int(action.get("selected_index", 0) or 0)
    options = dict(action.get("options") or {})
    plot_options = dict(options.get("plot options") or {})
    lines.extend(
        [
            "",
            "# CP Analysis",
            f"cp_obj = data[{selected_index}]",
            f"cp_plot_options = {_literal(plot_options)}",
        ]
    )
    for analysis in action.get("analyses") or []:
        if analysis == "stats":
            lines.append("cp_stats_result = cp_obj.stats()")
        elif analysis == "cycle_info":
            lines.append(f'cp_cycle_info_result = cp_obj.cycle_info({{"percent capacity": {_literal(bool(options.get("percent capacity", False)))}}})')
        elif analysis == "plot":
            lines.append('cp_plot_result = cp_obj.plot({**cp_plot_options, "print": False})')
        elif analysis == "cycling_plot":
            lines.extend(
                [
                    "cp_cycling_plot_result = cp_obj.cycling_plot({",
                    "    **cp_plot_options,",
                    f'    "percent capacity": {_literal(bool(options.get("percent capacity", False)))},',
                    f'    "capacity mode": {_literal(options.get("capacity mode", "both"))},',
                    f'    "efficiency mode": {_literal(options.get("efficiency mode", "both"))},',
                    "})",
                ]
            )
        elif analysis == "plot_cycles":
            lines.extend(
                [
                    "cp_plot_cycles_result = cp_obj.plot_cycles({",
                    "    **cp_plot_options,",
                    f'    "cycles": {_literal(options.get("cycles", (1, 100, 10)))},',
                    f'    "segment": {_literal(options.get("segment", "both"))},',
                    f'    "x axis": {_literal(options.get("x axis", "capacity"))},',
                    "})",
                ]
            )


def generate_python(workflow: AppWorkflow | dict | None) -> str:
    workflow = AppWorkflow.from_dict(workflow) if isinstance(workflow, dict) else workflow
    workflow = workflow or AppWorkflow()

    source_path = workflow.source_path or "path/to/data"
    lines = [
        "# Setup",
        "import ecat as e",
        "",
        "# Load Data",
        "data = e.get_data({",
        f'    "folder path": {_literal(source_path)},',
        f'    "recursive search": {bool(workflow.recursive)},',
        '    "print": False,',
    ]
    import_options = dict(workflow.import_options or {})
    import_options.setdefault("reference mode", "none")
    for key, value in import_options.items():
        if key in {"folder path", "recursive search", "print"}:
            continue
        lines.append(f'    "{key}": {_literal(value)},')
    lines.append("})")

    current_name = "data"
    if workflow.included_row_ids:
        included_indices = [
            int(str(row_id).replace("row-", ""))
            for row_id in workflow.included_row_ids
            if str(row_id).startswith("row-")
        ]
        lines.extend(
            [
                "",
                "# Select Data",
                f"included_indices = {_literal(included_indices)}",
                "included = [data[i] for i in included_indices]",
            ]
        )
        current_name = "included"

    if workflow.filters:
        lines.extend(
            [
                "",
                "# Filter Data",
                f"filtered = e.filter({current_name}, {_literal(workflow.filters)}, {{\"print\": False}})",
            ]
        )
        current_name = "filtered"

    if workflow.sort_keys or workflow.group_keys:
        lines.extend(
            [
                "",
                "# Sort And Group Data",
                f"grouped = e.sort_and_group(",
                f"    {current_name},",
                f"    sort_keys={_literal(workflow.sort_keys or None)},",
                f"    group_keys={_literal(workflow.group_keys or None)},",
                '    options={"print": False},',
                ")",
            ]
        )
        current_name = "grouped[0]"

    raw_plot_options = dict(workflow.plot_options or {})
    analysis_actions = _analysis_actions_for_codegen(workflow, raw_plot_options)

    preprocessing = {}
    for action in analysis_actions:
        if action.get("kind") != "cv_multi":
            continue
        options = dict(action.get("options") or {})
        if options.get("preprocessing"):
            preprocessing = dict(options["preprocessing"])
            break
    if preprocessing:
        scale_options = preprocessing.get("scale current")
        normalize = preprocessing.get("normalize") or {}
        if scale_options:
            lines.extend(
                [
                    "",
                    "# Pre-process: Scale Current",
                    f"scaled = e.scale_current({current_name}, {_literal(scale_options)})",
                ]
            )
            current_name = "scaled"
        normalize_mode = normalize.get("mode")
        normalize_options = normalize.get("options") or {}
        if normalize_mode == "dimensionless":
            lines.extend(
                [
                    "",
                    "# Pre-process: Dimensionless Normalize",
                    f"normalized = e.normalize({current_name}, {_literal(normalize_options)})",
                ]
            )
            current_name = "normalized"
        elif normalize_mode == "current":
            lines.extend(
                [
                    "",
                    "# Pre-process: Normalize Current",
                    f"normalized = e.normalize_current({current_name}, {_literal(normalize_options)})",
                ]
            )
            current_name = "normalized"

    x_label = raw_plot_options.get("x label")
    y_label = raw_plot_options.get("y label")
    animation_enabled = bool(raw_plot_options.get("_animate"))
    animation_keys = {
        "trace mode",
        "schedule",
        "timing mode",
        "normalized duration",
        "speedup",
        "fps",
        "stride",
        "stagger time",
        "end hold",
        "loop",
        "include quiet time",
        "directional arrows",
        "scale bar",
    }
    plot_options = {
        key: value
        for key, value in raw_plot_options.items()
        if not str(key).startswith("_") and not str(key).endswith(" options")
        and key not in {"plot style", "x label", "y label"}
        and (not animation_enabled or key not in animation_keys)
    }
    if not plot_options:
        plot_options = {"legend": "auto", "title": "auto"}
    lines.extend(["", "# Plot Options", "plot_options = {"])
    for key, value in plot_options.items():
        lines.append(f'    "{key}": {_literal(value)},')
    lines.append("}")

    lines.extend(
        [
            "",
            "# Plot",
            f"overlay_ax = e.multiplot({current_name}, plot_options)",
        ]
    )
    if x_label is False:
        lines.append('overlay_ax.set_xlabel("")')
    elif x_label not in (None, ""):
        lines.append(f"overlay_ax.set_xlabel({_literal(x_label)})")
    if y_label is False:
        lines.append('overlay_ax.set_ylabel("")')
    elif y_label not in (None, ""):
        lines.append(f"overlay_ax.set_ylabel({_literal(y_label)})")

    if animation_enabled:
        animation_options = {
            key: value
            for key, value in raw_plot_options.items()
            if key in animation_keys and value not in (None, "")
        }
        lines.extend(["", "# Animation Options", "animation_options = {"])
        for key, value in animation_options.items():
            lines.append(f'    "{key}": {_literal(value)},')
        lines.append("}")
        lines.extend(
            [
                "",
                "# Animation",
                f"animation = e.animate({current_name}, animation_options)",
            ]
        )

    _append_model_code(lines, workflow.model_options)

    single_actions = [action for action in analysis_actions if action.get("kind") == "cv_single"]
    selected_index = workflow.selected_index
    if selected_index is None and single_actions:
        selected_index = single_actions[0].get("selected_index")
    if selected_index is not None:
        lines.extend(["", "# Select Single Object", f"selected = data[{int(selected_index)}]"])
        selected_name = "selected"
    else:
        lines.extend(["", "# Select Single Object", "selected = data[0]"])
        selected_name = "selected"

    lines.extend(
        [
            "",
            "# Single Object Plot",
            f"ax = {selected_name}.plot(plot_options)",
        ]
    )
    if x_label is False:
        lines.append('ax.set_xlabel("")')
    elif x_label not in (None, ""):
        lines.append(f"ax.set_xlabel({_literal(x_label)})")
    if y_label is False:
        lines.append('ax.set_ylabel("")')
    elif y_label not in (None, ""):
        lines.append(f"ax.set_ylabel({_literal(y_label)})")

    for action in analysis_actions:
        kind = action.get("kind")
        if kind == "cv_single":
            _append_single_cv_analysis_code(lines, action, raw_plot_options)
        elif kind == "cv_multi":
            _append_multi_cv_analysis_code(lines, action, current_name)
        elif kind == "ca":
            _append_ca_analysis_code(lines, action)
        elif kind == "cp":
            _append_cp_analysis_code(lines, action)

    if workflow.export_filename:
        lines.extend(
            [
                "",
                "# Export Data",
                "e.save_data(data, {",
                '    "folder path": "outputs",',
                f'    "file name": {_literal(workflow.export_filename)},',
                "})",
            ]
        )

    return "\n".join(lines) + "\n"
