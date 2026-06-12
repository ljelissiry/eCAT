"""Generate reproducible public-API eCAT Python workflows."""

from __future__ import annotations

from pprint import pformat
import json

from .workflow import BrowserWorkflow


def _literal(value) -> str:
    if isinstance(value, str):
        return json.dumps(value)
    return pformat(value, width=88, sort_dicts=False)


def generate_python(workflow: BrowserWorkflow | dict | None) -> str:
    workflow = BrowserWorkflow.from_dict(workflow) if isinstance(workflow, dict) else workflow
    workflow = workflow or BrowserWorkflow()

    source_path = workflow.source_path or "path/to/data"
    lines = [
        "import ecat as e",
        "",
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
                f"included_indices = {_literal(included_indices)}",
                "included = [data[i] for i in included_indices]",
            ]
        )
        current_name = "included"

    if workflow.filters:
        lines.extend(
            [
                "",
                f"filtered = e.filter({current_name}, {_literal(workflow.filters)}, {{\"print\": False}})",
            ]
        )
        current_name = "filtered"

    if workflow.sort_keys or workflow.group_keys:
        lines.extend(
            [
                "",
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
    plot_options = {
        key: value
        for key, value in raw_plot_options.items()
        if not str(key).startswith("_") and not str(key).endswith(" options")
    }
    if not plot_options:
        plot_options = {"legend": "auto", "title": "auto"}
    lines.extend(["", "plot_options = {"])
    for key, value in plot_options.items():
        lines.append(f'    "{key}": {_literal(value)},')
    lines.append("}")

    if workflow.selected_index is not None:
        lines.extend(["", f"selected = data[{int(workflow.selected_index)}]"])
        selected_name = "selected"
    else:
        lines.extend(["", "selected = data[0]"])
        selected_name = "selected"

    lines.extend(
        [
            "",
            f"ax = {selected_name}.plot(plot_options)",
        ]
    )

    for analysis in workflow.analyses:
        if analysis in {"peak_potential", "peak_current", "half_peak_potential", "half_wave_potential"}:
            lines.extend(
                [
                    "",
                    f"{analysis}_result = {selected_name}.{analysis}({{",
                    '    "plot": False,',
                    '    "print": False,',
                    "})",
                ]
            )
        if analysis in {"fit_peak_potential", "fit_peak_current", "sevcik_analysis", "trumpet_analysis", "fowa"}:
            options = dict(raw_plot_options.get(f"{analysis} options", {}))
            if analysis == "fowa" and "non-catalytic cv index" in options:
                reference_index = options.pop("non-catalytic cv index")
                lines.extend(["", f"non_catalytic_cv = data[{int(reference_index)}]"])
                options["non-catalytic cv"] = "non_catalytic_cv"
            lines.extend(["", f"{analysis}_result = e.{analysis}({current_name}, {{"])
            for key, value in options.items():
                rendered = value if key == "non-catalytic cv" and value == "non_catalytic_cv" else _literal(value)
                lines.append(f'    "{key}": {rendered},')
            lines.extend(['    "print": False,', "})"])
        if analysis == "tafel_analysis":
            options = dict(raw_plot_options.get("tafel_analysis options", {}))
            cv_index = int(options.pop("cv index", 0))
            tof_max = options.pop("TOF max", "TOF_max")
            thermodynamic_potential = options.pop("thermodynamic potential", "thermodynamic_potential")
            redox_potential = options.pop("redox potential", "redox_potential")
            for key in ("plot", "plot all", "plot fit", "new plot", "print"):
                options.pop(key, None)
            lines.extend(
                [
                    "",
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

    lines.extend(
        [
            "",
            f"overlay_ax = e.multiplot({current_name}, plot_options)",
        ]
    )

    if workflow.export_filename:
        lines.extend(
            [
                "",
                "e.save_data(data, {",
                '    "folder path": "outputs",',
                f'    "file name": {_literal(workflow.export_filename)},',
                "})",
            ]
        )

    return "\n".join(lines) + "\n"
