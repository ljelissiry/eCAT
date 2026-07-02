"""Browser table helpers aligned with eCAT object-list display logic."""

from __future__ import annotations

from pathlib import Path

from ecat.plotting import (
    build_object_table,
    echem_similar_different,
    format_chemical_formulas,
    pretty_table_column_label,
)


AVAILABLE_COLUMNS = [
    "filename",
    "class",
    "software",
    "timestamp",
    "creation time",
    "temperature",
    "electrode area",
    "ir comp resistance",
    "ir uncomp resistance",
    "ir comp percent",
    "name",
    "subfolder",
    "exp type",
    "solvent",
    "gas",
    "compounds",
    "scan window",
    "scan rate",
    "segments",
    "reference shift",
    "reference label",
    "reference mode",
    "reference source",
]

APP_IDENTITY_COLUMNS = [
    "Filename",
    "Class",
    "Software",
]

REFERENCE_COLUMNS = {
    "Reference Shift",
    "Reference Label",
    "Reference Mode",
    "Reference Source",
}

CHEMICAL_VALUE_COLUMNS = {"Gas", "Solvent", "Compounds"}
CONDITION_ORDER = [
    "exp type",
    "solvent",
    "gas",
    "compounds",
    "scan window",
    "scan rate",
    "segments",
]


def _pretty_columns(columns):
    return [pretty_table_column_label(col) for col in columns]


def _unpretty_column(column_id: str) -> str:
    lookup = {pretty_table_column_label(col): col for col in AVAILABLE_COLUMNS}
    return lookup.get(column_id, column_id)


def _format_timestamp(value):
    if value is None:
        return None
    return str(value)


def _is_blank(value):
    if value is None:
        return True
    if isinstance(value, str) and value.strip() == "":
        return True
    if isinstance(value, (list, tuple, set, dict)) and len(value) == 0:
        return True
    return False


def _format_condition_value(key, value):
    if _is_blank(value):
        return None
    if isinstance(value, (list, tuple, set)):
        values = [
            format_chemical_formulas(item, mode="unicode")
            if isinstance(item, str)
            else str(item)
            for item in value
            if not _is_blank(item)
        ]
        return ", ".join(values) if values else None
    if isinstance(value, str) and str(key).strip().lower() in {"gas", "solvent", "compounds"}:
        return format_chemical_formulas(value, mode="unicode")
    return str(value)


def _format_table_value(column, value):
    if value is None:
        return value
    if column in CHEMICAL_VALUE_COLUMNS:
        return _format_condition_value(column, value)
    return value


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


def _object_extra_row(obj, source_lookup=None):
    source_lookup = source_lookup or {}
    filepath = getattr(obj, "filepath", None)
    filename = Path(filepath).name if filepath else None
    return {
        "Filename": getattr(obj, "filename", None) or filename,
        "Class": type(obj).__name__,
        "Software": getattr(obj, "software", None),
        "Timestamp": _format_timestamp(getattr(obj, "timestamp", None)),
        "Creation Time": _format_timestamp(getattr(obj, "creation_time", None)),
        "Temperature": getattr(obj, "temperature", None),
        "Electrode Area": getattr(obj, "electrode_area", None),
        "IR Comp Resistance": getattr(obj, "ir_comp_resistance", None),
        "IR Uncomp Resistance": getattr(obj, "ir_uncomp_resistance", None),
        "IR Comp Percent": getattr(obj, "ir_comp_percent", None),
        "Reference Shift": getattr(obj, "reference_shift", None),
        "Reference Label": getattr(obj, "reference_label", None),
        "Reference Mode": getattr(obj, "reference_mode", None),
        "Reference Source": _reference_source_display(obj, source_lookup),
    }


def _has_any_value(rows, column):
    return any(row.get(column) not in (None, "", "None") for row in rows)


def _has_active_reference(rows):
    for row in rows:
        shift = row.get("Reference Shift")
        source = row.get("Reference Source")
        mode = str(row.get("Reference Mode") or "").strip().lower()
        if shift not in (None, "", "None"):
            return True
        if source not in (None, "", "None"):
            return True
        if mode and mode != "none":
            return True
    return False


def default_visible_columns(objects) -> list[str]:
    objects = list(objects or [])
    display_df, _meta = build_object_table(objects, {"print conditions": False})
    columns = list(display_df.columns)
    source_lookup = _source_index_lookup(objects)
    rows = [_object_extra_row(obj, source_lookup) for obj in objects]
    if _has_active_reference(rows):
        for column in REFERENCE_COLUMNS:
            if column not in columns:
                columns.append(column)
    return columns


def _all_object_table_columns(objects) -> tuple[list[dict[str, object]], list[str]]:
    objects = list(objects or [])
    source_lookup = _source_index_lookup(objects)
    display_df, _meta = build_object_table(
        objects,
        {"columns": "all", "print conditions": False},
    )
    display_df = display_df.reset_index(drop=True)
    rows = []
    for index, row in display_df.iterrows():
        record = {"id": f"row-{index}", "index": index}
        record.update(
            {
                column: _format_table_value(column, value)
                for column, value in row.to_dict().items()
            }
        )
        if index < len(objects):
            record.update(_object_extra_row(objects[index], source_lookup))
        rows.append(record)
    columns = list(display_df.columns)
    for column in [
        "Filename",
        "Class",
        "Software",
        "Timestamp",
        "Creation Time",
        "Temperature",
        "Electrode Area",
        "IR Comp Resistance",
        "IR Uncomp Resistance",
        "IR Comp Percent",
        *sorted(REFERENCE_COLUMNS),
    ]:
        if column not in columns:
            columns.append(column)
    return rows, columns


def import_conditions_summary(objects) -> list[str]:
    objects = list(objects or [])
    if not objects:
        return []
    common_values, _different_values, _stats_rows = echem_similar_different(
        objects,
        options={},
        return_values=True,
    )
    ordered_keys = [key for key in CONDITION_ORDER if key in common_values] + [
        key for key in common_values if key not in CONDITION_ORDER
    ]
    conditions = []
    for key in ordered_keys:
        formatted = _format_condition_value(key, common_values[key])
        if formatted is not None:
            conditions.append(f"{pretty_table_column_label(key)}: {formatted}")
    return conditions


def available_column_options(objects) -> list[dict[str, str]]:
    rows, columns = _all_object_table_columns(list(objects or []))
    options = []
    for column in columns:
        if column in REFERENCE_COLUMNS and not _has_active_reference(rows):
            continue
        options.append({"label": column, "value": column})
    return options


def build_browser_table(objects, visible_columns=None, extra_columns=None) -> dict[str, object]:
    objects = list(objects or [])
    rows, available_columns = _all_object_table_columns(objects)
    if visible_columns is None:
        if extra_columns is not None:
            visible_columns = default_visible_columns(objects) + _pretty_columns(extra_columns)
        else:
            visible_columns = default_visible_columns(objects)
    ordered_columns = [column for column in visible_columns if column in available_columns]
    data = []
    for row in rows:
        record = {"id": row["id"], "index": row["index"]}
        for column in ordered_columns:
            record[column] = row.get(column)
        data.append(record)

    columns = [{"name": "index", "id": "index"}]
    columns.extend({"name": col, "id": col} for col in ordered_columns)
    return {"columns": columns, "data": data}


def selected_column_values(table: dict[str, object]) -> list[str]:
    values = []
    for column in table.get("columns", []):
        column_id = column.get("id")
        if column_id == "index":
            continue
        values.append(column_id)
    return values


def ag_grid_column_defs(table: dict[str, object]) -> list[dict[str, object]]:
    column_defs = []
    for column in table.get("columns", []):
        column_id = column.get("id")
        column_def = {
            "field": column_id,
            "headerName": column.get("name", column_id),
            "sortable": True,
            "filter": True,
            "resizable": True,
        }
        if column_id == "index":
            column_def.update(
                {
                    "checkboxSelection": True,
                    "headerCheckboxSelection": True,
                    "headerCheckboxSelectionFilteredOnly": True,
                    "pinned": "left",
                    "width": 112,
                    "minWidth": 112,
                    "maxWidth": 132,
                }
            )
        elif column_id == "Filename":
            column_def["minWidth"] = 340
        column_defs.append(column_def)
    return column_defs


def selected_row_ids_from_grid_rows(rows) -> list[str]:
    return [row.get("id") for row in rows or [] if row.get("id") is not None]


def selected_grid_rows_for_ids(rows, selected_row_ids) -> dict[str, list[str]]:
    row_ids = {row.get("id") for row in rows or []}
    selected = [row_id for row_id in (selected_row_ids or []) if row_id in row_ids]
    return {"ids": selected}


def reset_column_selection(objects) -> dict[str, object]:
    return build_browser_table(objects, visible_columns=default_visible_columns(objects))


def selected_rows_for_table(table: dict[str, object]) -> list[int]:
    return list(range(len(table.get("data", []))))


def selection_toggle_state(rows, selected_row_ids) -> str:
    row_ids = [row.get("id") for row in rows or []]
    selected = set(selected_row_ids or [])
    selected_visible = selected.intersection(row_ids)
    if not row_ids or len(selected_visible) == 0:
        return "All"
    if selected_visible == set(row_ids):
        return "None"
    return "-"


def toggle_all_selection(rows, selected_row_ids) -> list[str]:
    row_ids = [row.get("id") for row in rows or []]
    selected = set(selected_row_ids or [])
    if row_ids and selected == set(row_ids):
        return []
    return row_ids


def toggle_all_selection_state(rows, selected_row_ids) -> tuple[list[str], list[int]]:
    row_ids = toggle_all_selection(rows, selected_row_ids)
    selected_rows = [index for index, row in enumerate(rows or []) if row.get("id") in set(row_ids)]
    return row_ids, selected_rows


def toggle_filtered_selection_state(
    all_rows, displayed_rows, selected_row_ids
) -> tuple[list[str], list[int]]:
    all_ids = [row.get("id") for row in all_rows or [] if row.get("id") is not None]
    displayed_ids = [row.get("id") for row in displayed_rows or [] if row.get("id") is not None]
    selected = set(selected_row_ids or [])
    if displayed_ids and selected.intersection(displayed_ids) == set(displayed_ids):
        next_selected = [row_id for row_id in all_ids if row_id in selected and row_id not in set(displayed_ids)]
    else:
        next_selected = [row_id for row_id in all_ids if row_id in selected]
        for row_id in displayed_ids:
            if row_id not in next_selected:
                next_selected.append(row_id)
    selected_rows = [index for index, row in enumerate(all_rows or []) if row.get("id") in set(next_selected)]
    return next_selected, selected_rows


def displayed_selected_row_ids(displayed_rows, selected_row_ids=None) -> list[str]:
    displayed_ids = [row.get("id") for row in displayed_rows or [] if row.get("id") is not None]
    if selected_row_ids is None:
        return displayed_ids
    selected = set(selected_row_ids or [])
    return [row_id for row_id in displayed_ids if row_id in selected]
