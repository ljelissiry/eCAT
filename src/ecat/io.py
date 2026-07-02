"""Data-loading helpers."""

import glob
import os
import re

import numpy as np
import pandas as pd

from .metadata import get_file_times
from .options import ImportOptions, import_options_to_legacy_dict
from .utils import (
    _format_path_for_display,
    apply_text_alterations,
    count_segments,
    get_conversion_factor,
    resolve_electrode_area_option,
    round_sigfigs,
)
from .objects import _normalize_parser_settings, ca, cp, cv, dpv, echem
from .plotting import _coerce_display_columns, show_objects
from .collection import sort
from .reference import (
    _apply_reference_map,
    _compute_explicit_reference_file_shift,
    _contains_string_case_insensitive,
    _format_reference_display,
    _normalize_reference_map,
    _print_reference_correction_summary,
    _print_reference_usage_list,
    _print_reference_usage_summary,
    _print_reference_usage_troubleshoot,
    _resolve_reference_shifts,
    canonical_reference_label,
    normalize_legacy_reference_options,
    resolve_reference_options,
)


def _relative_folderpath(folder, root):
    rel_folder = os.path.relpath(os.path.abspath(folder), os.path.abspath(root))
    return "" if rel_folder == "." else rel_folder


def _clean_excel_header_value(value):
    if value is None:
        return ""
    if isinstance(value, float) and pd.isna(value):
        return ""
    return str(value).strip()


def _normalize_excel_column(col):
    """
    Return a normalized 2-tuple header: (top_header, sub_header)
    """
    if isinstance(col, tuple):
        return tuple(_clean_excel_header_value(v) for v in col)
    return (_clean_excel_header_value(col), "")


def _excel_column_role(col):
    """
    Classify an Excel column as:
    - 'current'
    - 'potential'
    - 'reference'
    - 'other'
    """
    top, sub = _normalize_excel_column(col)
    sub_lower = sub.lower()

    if "current" in sub_lower:
        return "current"

    if ("potential" in sub_lower and "vs" in sub_lower) or sub_lower.startswith("v vs"):
        return "reference"

    if "potential" in sub_lower:
        return "potential"

    return "other"


def _reference_label_from_header(header_text):
    """
    Examples
    --------
    'V vs Fc/Fc+' -> 'Fc/Fc+'
    'Potential vs Ag/AgCl' -> 'Ag/AgCl'
    """
    text = _clean_excel_header_value(header_text)
    if text == "":
        return None

    text = re.sub(r"(?i)^potential\s*", "", text).strip()
    text = re.sub(r"(?i)^v\s*vs\s*", "", text).strip()

    match = re.search(r"(?i)\bvs\b\s*(.+)$", text)
    if match:
        return match.group(1).strip()

    return text


def _infer_reference_shift(potential_series, reference_series):
    """
    Infer a constant shift such that:
        E_ref = E_raw - shift
    so:
        shift = E_raw - E_ref
    """
    potential = pd.to_numeric(potential_series, errors="coerce")
    reference = pd.to_numeric(reference_series, errors="coerce")

    diff = (potential - reference).dropna()
    if diff.empty:
        return None, None

    shift = float(diff.median())
    spread = float((diff - shift).abs().max())
    return shift, spread


def _parse_scan_rate_from_name(name):
    """
    Parse scan rate from names like:
    '100mVs', '100 mV/s', '0.1Vs', '500uV/s'
    Returns scan rate in V/s or None.
    """
    text = str(name)

    match = re.search(
        r"(\d+(?:\.\d+)?)\s*([numμm]?)\s*V\s*/?\s*s",
        text,
        flags=re.IGNORECASE,
    )
    if not match:
        return None

    value = float(match.group(1))
    prefix = match.group(2).lower()

    factor = {
        "": 1.0,
        "m": 1e-3,
        "u": 1e-6,
        "μ": 1e-6,
        "n": 1e-9,
    }.get(prefix, 1.0)

    return value * factor


def _parse_scan_rate_from_text_lines(lines):
    """
    Parse scan-rate metadata from common text-export headers.
    """
    for line in lines:
        match = re.search(
            r"scan\s*rate\s*(?:\([^)]*\))?\s*=\s*([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)",
            line,
            flags=re.IGNORECASE,
        )
        if match:
            return float(match.group(1))
    return None


def _parse_unit_from_text_column(column_name, default=None):
    """
    Extract a unit from labels like 'Potential/V', 'Current (A)', or
    'WE(1).Current (uA)'.
    """
    text = str(column_name).strip()
    paren = re.search(r"\(([^()]+)\)\s*$", text)
    if paren:
        return paren.group(1).strip()

    slash = re.search(r"/\s*([^\s,/;]+)\s*$", text)
    if slash:
        return slash.group(1).strip()

    return default


def _standardize_cv_text_columns(df):
    """
    Pick potential/current columns from a real text CV table and normalize to
    eCAT's standard Potential/Current columns in V and A when units are known.
    """
    potential_cols = [
        col for col in df.columns
        if "potential" in str(col).lower()
    ]
    current_cols = [
        col for col in df.columns
        if "current" in str(col).lower()
        and "range" not in str(col).lower()
    ]

    if not potential_cols or not current_cols:
        raise ValueError(
            "Could not identify potential/current columns. "
            f"Available columns: {list(df.columns)}"
        )

    potential_col = potential_cols[0]
    current_col = current_cols[0]

    potential_unit = _parse_unit_from_text_column(potential_col, default="V")
    current_unit = _parse_unit_from_text_column(current_col, default="A")

    out = pd.DataFrame(
        {
            "Potential": pd.to_numeric(df[potential_col], errors="coerce"),
            "Current": pd.to_numeric(df[current_col], errors="coerce"),
        }
    ).dropna(how="any").reset_index(drop=True)

    if out.empty:
        raise ValueError("No numeric potential/current rows were found.")

    try:
        out["Potential"] *= get_conversion_factor(potential_unit, "V")
        potential_unit = "V"
    except Exception:
        pass

    try:
        out["Current"] *= get_conversion_factor(current_unit, "A")
        current_unit = "A"
    except Exception:
        pass

    return out, {"Potential": potential_unit, "Current": current_unit}


def _read_cv_text_table(filepath):
    """
    Read a CV text export with automatic table-start, delimiter, and decimal
    detection. Handles European semicolon/comma tables, tab-delimited tables,
    and CH-style metadata followed by a potential/current table.
    """
    with open(filepath, "r", encoding="utf-8", errors="replace") as handle:
        lines = handle.read().splitlines()

    header_idx = None
    for idx, line in enumerate(lines):
        lower = line.lower()
        if "potential" in lower and "current" in lower:
            header_idx = idx
            break

    if header_idx is None:
        raise ValueError("Could not find a potential/current table header.")

    header_line = lines[header_idx]

    if ";" in header_line:
        sep = ";"
        decimal = ","
        engine = None
    elif "\t" in header_line:
        sep = "\t"
        decimal = "."
        engine = None
    elif "," in header_line:
        # Some CH exports use a comma-delimited header followed by tab data.
        sep = r"[,\t]+"
        decimal = "."
        engine = "python"
    else:
        sep = r"\s+"
        decimal = "."
        engine = "python"

    read_kwargs = {
        "sep": sep,
        "skiprows": header_idx,
        "decimal": decimal,
        "encoding": "utf-8",
    }
    if engine is not None:
        read_kwargs["engine"] = engine

    df = pd.read_csv(filepath, **read_kwargs)
    df.columns = [str(col).strip() for col in df.columns]
    df = df.dropna(how="all", axis=1)

    return _standardize_cv_text_columns(df)


def _make_cv_object_from_text_file(filepath, options=None, root_abs=None):
    """
    Build a file-backed cv object from a flexible real text CV export.
    """
    options = import_options_to_legacy_dict(options)
    filepath = os.path.abspath(filepath)
    root_abs = os.path.abspath(root_abs) if root_abs is not None else os.path.dirname(filepath)

    data, units = _read_cv_text_table(filepath)
    with open(filepath, "r", encoding="utf-8", errors="replace") as handle:
        lines = handle.read().splitlines()

    obj = cv.__new__(cv)
    obj.filepath = filepath
    obj.options = dict(options)
    obj.timestamp = None
    try:
        obj.creation_time, obj.modification_time = get_file_times(filepath)
    except OSError:
        obj.creation_time = None
        obj.modification_time = None

    obj.name = os.path.basename(filepath[:filepath.rindex(".")])
    obj.name = apply_text_alterations(
        obj.name,
        options.get("name alterations")
    )
    obj.type = "Cyclic Voltammetry"
    obj.software = "Text CV"
    obj.num_x_cols = 1
    obj.data = data
    obj.units = units

    obj.temperature = options.get("temperature", 298)
    obj.electrode_area = resolve_electrode_area_option(options)
    obj.gas = options.get("gas")
    obj.solvent = options.get("solvent")

    obj.reference_shift = None
    obj.reference_label = None
    obj.reference_mode = "none"
    obj.reference_source_file = None
    obj.reference_failure_message = None

    obj.folderpath = _relative_folderpath(os.path.dirname(filepath), root_abs)

    parser_settings = options.get("parser settings")
    obj.get_data_from_name(parser_settings)
    compounds, concentrations = obj.extract_compounds_and_concentrations(
        options.get("compounds"),
        parser_settings=parser_settings,
    )
    metadata = {
        "gas": obj.gas,
        "solvent": obj.solvent,
        "compounds": compounds,
        "concentrations": concentrations,
    }
    metadata = obj._apply_custom_parser_metadata(metadata, options)
    obj.gas = metadata.get("gas", obj.gas)
    obj.solvent = metadata.get("solvent", obj.solvent)
    obj.compounds = metadata.get("compounds", [])
    obj.concentrations = metadata.get("concentrations", [])

    scan_rate = _parse_scan_rate_from_text_lines(lines)
    if scan_rate is None:
        scan_rate = _parse_scan_rate_from_name(obj.name)
    if scan_rate is None:
        scan_rate = options.get("scan rate")
    prefer_file_metadata = _normalize_parser_settings(parser_settings)["prefer file metadata"]
    custom_scan_rate = metadata.get("scan rate", None)
    if custom_scan_rate is not None and (scan_rate is None or not prefer_file_metadata):
        scan_rate = float(custom_scan_rate)
    obj.scan_rate = scan_rate

    x = obj.data["Potential"].to_numpy(dtype=float)
    obj.init_E = float(x[0]) if len(x) else None
    obj.final_E = float(x[-1]) if len(x) else None
    obj.min_E = float(np.min(x)) if len(x) else None
    obj.max_E = float(np.max(x)) if len(x) else None
    obj.segments = count_segments(x) if len(x) else 0
    obj.delta_x = float(abs(x[1] - x[0])) if len(x) > 1 else None

    if options.get("convert current", False):
        obj.current_to(options["convert current"])
    if options.get("invert current", False):
        obj.invert_current()
    if options.get("shift potential"):
        obj.potential_shift(options)

    obj._refresh_parse_result(parser="Text CV")
    return obj


def get_CVs(options=None):
    """
    Load CV text files from a folder as eCAT cv objects.

    This is a CV-specific companion to get_data() for practical notebook
    workflows where text exports use mixed delimiters, decimal conventions,
    and minimal instrument metadata.
    """
    typed_options = ImportOptions.from_options(options)
    options = typed_options.to_legacy_dict()

    folder_path = os.path.expanduser(options.get("folder path", "."))
    root_abs = os.path.abspath(folder_path)
    glob_root = glob.escape(root_abs)
    recursive = options.get("recursive search", True)
    recursive_search = "recursively" if recursive else "exclusively"

    if not os.path.exists(root_abs):
        print(f"Folder does not exist:\n{_format_path_for_display(root_abs)}")
        return []
    if not os.path.isdir(root_abs):
        print(f"Path exists but is not a folder:\n {_format_path_for_display(root_abs)}")
        return []

    print(
        "Searching "
        f"{recursive_search} for CV text files through:\n "
        f"{_format_path_for_display(root_abs)}"
    )

    search_pattern = os.path.join(glob_root, "**", "*.txt") if recursive else os.path.join(glob_root, "*.txt")
    file_paths = sorted(
        (os.path.abspath(p) for p in glob.glob(search_pattern, recursive=recursive)),
        key=lambda p: os.path.normcase(os.path.relpath(p, root_abs)),
    )

    if not file_paths:
        print(f"No .txt files were found in the folder:\n {_format_path_for_display(root_abs)}")
        return []

    cvs = []
    failures = []
    for filepath in file_paths:
        try:
            cvs.append(_make_cv_object_from_text_file(filepath, options, root_abs=root_abs))
        except Exception as exc:
            failures.append((filepath, exc))

    print(f"{len(cvs)} CV file(s) loaded.")
    if failures:
        print(f"{len(failures)} file(s) skipped because no CV table could be loaded.")
        if options.get("troubleshoot"):
            for filepath, exc in failures:
                rel = os.path.relpath(filepath, root_abs)
                print(f"  {rel}: {type(exc).__name__}: {exc}")

    if not cvs:
        return []

    if options.get("print", False):
        show_objects(cvs, options)

    return cvs


def _make_cv_object_from_dataframe(
    display_name,
    cv_data,
    options,
    reference_shift=None,
    reference_label=None,
    metadata_name=None,
):
    """
    Build a cv object from an in-memory dataframe without going through
    the file-based cv.__init__ path.
    """
    obj = cv.__new__(cv)

    # Core attributes expected by echem/cv methods
    obj.filepath = None
    obj.options = dict(options)
    obj.timestamp = None
    obj.creation_time = None
    obj.modification_time = None

    obj.name = str(display_name)
    obj.type = "Cyclic Voltammetry"
    obj.software = "Excel"
    obj.num_x_cols = 1

    obj.data = cv_data.copy()
    obj.data.columns = ["Potential", "Current"]
    obj.data = (
        obj.data.apply(pd.to_numeric, errors="coerce")
        .dropna(how="any")
        .reset_index(drop=True)
    )

    obj.units = {"Potential": "V", "Current": "A"}

    obj.temperature = options.get("temperature", 298)
    obj.electrode_area = resolve_electrode_area_option(options)
    obj.gas = options.get("gas")
    obj.solvent = options.get("solvent")

    obj.reference_shift = reference_shift
    obj.reference_label = reference_label
    obj.reference_mode = "manual" if reference_shift is not None else "none"
    obj.reference_source_file = None
    obj.reference_failure_message = None

    # Parse metadata from the raw first-row header, not the display name
    metadata = _extract_excel_name_metadata(
        metadata_name if metadata_name is not None else display_name,
        options.get("compounds"),
        options,
    )

    if metadata["gas"] is not None:
        obj.gas = metadata["gas"]
    if metadata["solvent"] is not None:
        obj.solvent = metadata["solvent"]

    obj.compounds = metadata["compounds"]
    obj.concentrations = metadata["concentrations"]
    obj.scan_rate = metadata["scan_rate"]

    x = obj.data["Potential"].to_numpy(dtype=float)
    obj.init_E = float(x[0]) if len(x) else None
    obj.final_E = float(x[-1]) if len(x) else None
    obj.min_E = float(np.min(x)) if len(x) else None
    obj.max_E = float(np.max(x)) if len(x) else None
    obj.segments = count_segments(x) if len(x) else 0
    obj.delta_x = float(abs(x[1] - x[0])) if len(x) > 1 else None

    if options.get("convert current", False):
        obj.current_to(options["convert current"])

    if options.get("invert current", False):
        obj.invert_current()

    obj._refresh_parse_result(parser="Excel")
    return obj

def _extract_excel_name_metadata(name, extra_compounds=None, options=None):
    """
    Reuse the existing filename-style metadata parsing on an Excel header.
    """
    parsed_name = _clean_excel_header_value(name)

    probe = echem.__new__(echem)
    probe.name = parsed_name
    probe.gas = None
    probe.solvent = None

    parser_settings = options.get("parser settings") if isinstance(options, dict) else None
    probe.get_data_from_name(parser_settings)
    compounds, concentrations = probe.extract_compounds_and_concentrations(
        extra_compounds,
        parser_settings=parser_settings,
    )

    metadata = {
        "name": parsed_name,
        "gas": probe.gas,
        "solvent": probe.solvent,
        "compounds": compounds,
        "concentrations": concentrations,
        "scan rate": _parse_scan_rate_from_name(parsed_name),
    }
    metadata = probe._apply_custom_parser_metadata(metadata, options if isinstance(options, dict) else {})

    return {
        "name": parsed_name,
        "gas": metadata.get("gas"),
        "solvent": metadata.get("solvent"),
        "compounds": metadata.get("compounds", []),
        "concentrations": metadata.get("concentrations", []),
        "scan_rate": metadata.get("scan rate"),
    }


def _manifest_column_lookup(columns):
    return {str(col).strip().lower(): col for col in columns}


def _manifest_value(row, lookup, *names, default=None):
    for name in names:
        col = lookup.get(str(name).strip().lower())
        if col is None:
            continue
        value = row.get(col)
        if value is None:
            continue
        try:
            if pd.isna(value):
                continue
        except Exception:
            pass
        if isinstance(value, str) and value.strip() == "":
            continue
        return value
    return default


def _manifest_float(row, lookup, *names, default=None):
    value = _manifest_value(row, lookup, *names, default=None)
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _manifest_int(row, lookup, *names, default=None):
    value = _manifest_float(row, lookup, *names, default=None)
    if value is None:
        return default
    return int(value)


def _split_manifest_compounds(value):
    if value in (None, ""):
        return []
    try:
        if pd.isna(value):
            return []
    except Exception:
        pass
    if isinstance(value, (list, tuple, set)):
        return [str(v).strip() for v in value if str(v).strip()]
    return [part.strip() for part in str(value).split(",") if part.strip()]


def _type_for_excel_class(class_key):
    class_key = str(class_key or "").strip().lower()
    mapping = {
        "cv": (cv, "Cyclic Voltammetry"),
        "ca": (ca, "Chronoamperometry"),
        "cp": (cp, "Chronopotentiometry"),
        "dpv": (dpv, "Differential Pulse Voltammetry"),
    }
    return mapping.get(class_key, (echem, class_key or None))


def _leading_excel_axis_column_count(columns):
    count = 0
    for column in columns:
        name = str(column).strip().lower()
        is_axis = (
            "potential" in name
            or name in {"e", "time", "t"}
            or name.startswith("time ")
        )
        if not is_axis:
            break
        count += 1
    return max(count, 1)


def _read_ecat_workbook_sheet(workbook, sheet_name):
    raw = pd.read_excel(workbook, sheet_name=sheet_name, header=None)
    if raw.shape[0] < 3:
        return raw, [], [], []
    group_row = [_clean_excel_header_value(v) for v in raw.iloc[0].tolist()]
    column_row = [_clean_excel_header_value(v) for v in raw.iloc[1].tolist()]
    unit_row = [_clean_excel_header_value(v) for v in raw.iloc[2].tolist()]
    values = raw.iloc[3:].reset_index(drop=True)
    return values, group_row, column_row, unit_row


def _columns_for_manifest_object(values, group_row, column_row, unit_row, object_id, x_group):
    object_indices = [
        idx for idx, group in enumerate(group_row)
        if group == object_id and column_row[idx] != ""
    ]
    shared_indices = [
        idx for idx, group in enumerate(group_row)
        if x_group not in (None, "") and group == x_group and column_row[idx] != ""
    ]

    selected = []
    object_has_x = any(
        "potential" in column_row[idx].lower() or "time" in column_row[idx].lower()
        for idx in object_indices
    )
    if object_indices and not object_has_x:
        selected.extend(shared_indices)
    selected.extend(object_indices)
    if not selected:
        selected = object_indices or shared_indices

    data = values.iloc[:, selected].copy() if selected else pd.DataFrame()
    columns = [column_row[idx] for idx in selected]
    units = {column_row[idx]: unit_row[idx] for idx in selected if column_row[idx] != ""}
    data.columns = columns
    data = data.apply(pd.to_numeric, errors="coerce").dropna(how="all").reset_index(drop=True)
    data = data.dropna(axis=1, how="all")
    units = {col: units.get(col, "") for col in data.columns}
    return data, units


def _make_object_from_excel_manifest(class_key, name, data, units, row, lookup, options):
    cls, default_type = _type_for_excel_class(class_key)
    obj = cls.__new__(cls)
    obj.filepath = None
    obj.folderpath = _manifest_value(row, lookup, "Subfolder", default=".")
    obj.options = dict(options)
    obj.timestamp = None
    obj.creation_time = None
    obj.modification_time = None
    obj.name = str(name)
    type_value = _manifest_value(row, lookup, "Exp Type", "Type", default=default_type)
    if str(type_value).strip().lower() == class_key:
        type_value = default_type
    obj.type = type_value
    obj.software = _manifest_value(row, lookup, "Software", default="Excel")
    obj.data = data.reset_index(drop=True)
    obj.units = units
    obj.num_x_cols = _leading_excel_axis_column_count(obj.data.columns)

    obj.temperature = _manifest_float(row, lookup, "Temperature", default=options.get("temperature", 298))
    obj.electrode_area = _manifest_float(row, lookup, "Electrode Area", default=resolve_electrode_area_option(options))
    obj.gas = _manifest_value(row, lookup, "Gas", default=options.get("gas"))
    obj.solvent = _manifest_value(row, lookup, "Solvent", default=options.get("solvent"))
    obj.compounds = _split_manifest_compounds(_manifest_value(row, lookup, "Compounds", default=""))
    obj.concentrations = []
    obj.scan_rate = _manifest_float(row, lookup, "Scan Rate", default=None)
    obj.segments = _manifest_int(row, lookup, "Segments", default=None)
    obj.reference_shift = _manifest_float(row, lookup, "Reference Shift", default=None)
    obj.reference_label = _manifest_value(row, lookup, "Reference Label", default=None)
    obj.reference_mode = _manifest_value(row, lookup, "Reference Mode", default="none")
    obj.reference_source_file = _manifest_value(row, lookup, "Reference Source", default=None)
    obj.reference_failure_message = None
    obj.ir_comp_resistance = _manifest_float(row, lookup, "IR Comp Resistance", default=None)
    obj.ir_uncomp_resistance = _manifest_float(row, lookup, "IR Uncomp Resistance", default=None)
    obj.ir_comp_percent = _manifest_float(row, lookup, "IR Comp Percent", default=None)

    parsed = _extract_excel_name_metadata(obj.name, options.get("compounds"), options)
    if not obj.gas and parsed["gas"] is not None:
        obj.gas = parsed["gas"]
    if not obj.solvent and parsed["solvent"] is not None:
        obj.solvent = parsed["solvent"]
    if not obj.compounds:
        obj.compounds = parsed["compounds"]
        obj.concentrations = parsed["concentrations"]
    if obj.scan_rate is None:
        obj.scan_rate = parsed["scan_rate"]

    if not obj.data.empty:
        x = pd.to_numeric(obj.data.iloc[:, 0], errors="coerce").dropna()
        if not x.empty:
            obj.init_E = float(x.iloc[0])
            obj.final_E = float(x.iloc[-1])
            obj.min_E = float(x.min())
            obj.max_E = float(x.max())
            obj.delta_x = float(abs(x.iloc[1] - x.iloc[0])) if len(x) > 1 else None
            if getattr(obj, "segments", None) is None and class_key == "cv":
                obj.segments = count_segments(x.to_numpy())

    if class_key == "ca":
        obj.run_time = _manifest_float(row, lookup, "Run Time", default=None)
        obj.sample_interval = _manifest_float(row, lookup, "Sample Interval", default=None)
        obj.quiet_time = _manifest_float(row, lookup, "Quiet Time", default=None)
    elif class_key in {"cp", "dpv"}:
        obj.quiet_time = _manifest_float(row, lookup, "Quiet Time", default=None)
    obj._refresh_parse_result(parser="Excel manifest")
    return obj


def _create_objects_from_ecat_workbook(file_path, options):
    workbook = pd.ExcelFile(file_path, engine="openpyxl")
    manifest_sheet = next(
        sheet for sheet in workbook.sheet_names
        if str(sheet).strip().lower() == "manifest"
    )
    manifest = pd.read_excel(workbook, sheet_name=manifest_sheet)
    lookup = _manifest_column_lookup(manifest.columns)
    sheet_cache = {}
    objects = []

    for _, row in manifest.iterrows():
        object_id = _manifest_value(row, lookup, "object_id")
        sheet = _manifest_value(row, lookup, "sheet")
        class_key = str(_manifest_value(row, lookup, "class", default="echem")).strip().lower()
        x_group = _manifest_value(row, lookup, "x group", default=object_id)
        name = _manifest_value(row, lookup, "Name", "name", default=object_id)
        if object_id is None or sheet is None:
            continue
        if sheet not in sheet_cache:
            sheet_cache[sheet] = _read_ecat_workbook_sheet(workbook, sheet)
        values, group_row, column_row, unit_row = sheet_cache[sheet]
        data, units = _columns_for_manifest_object(
            values,
            group_row,
            column_row,
            unit_row,
            str(object_id),
            str(x_group),
        )
        obj = _make_object_from_excel_manifest(
            class_key,
            name,
            data,
            units,
            row,
            lookup,
            options,
        )
        objects.append(obj)

    if options.get("print", False):
        show_objects(objects, options)
    return objects


def _create_data_objects_from_excel(file_path, options=None):
    """
    Create CV objects from an Excel workbook with flexible support for:

    - paired columns: [Potential, Current]
    - shared axes: one Potential column used by several Current columns
    - referenced axes: e.g. 'V vs Fc/Fc+' alongside raw Potential
    - multiple sheets

    Expected layout
    ---------------
    Uses a 2-row header:
      level 0 -> curve/group names
      level 1 -> axis labels such as:
                 'Potential/V', 'V vs Fc/Fc+', 'Current/A'

    Rules
    -----
    1. Any column with a level-1 header containing 'Current' is treated as a CV current column.
    2. Any column with 'Potential' but not 'vs' is treated as the raw potential axis.
    3. Any column with 'Potential ... vs ...' or 'V vs ...' is treated as a referenced axis.
    4. A current column first looks for local axes immediately to its right.
    5. If none exist, it inherits the most recent axis block to its left.
    """
    options = import_options_to_legacy_dict(options)

    try:
        workbook_info = pd.ExcelFile(file_path, engine="openpyxl")
    except ImportError:
        workbook = pd.read_excel(
            file_path,
            header=[0, 1],
            sheet_name=None,
            engine="openpyxl",
        )
        if isinstance(workbook, dict) and not workbook:
            return []
        raise

    if any(str(sheet).strip().lower() == "manifest" for sheet in workbook_info.sheet_names):
        return _create_objects_from_ecat_workbook(file_path, options)

    workbook = pd.read_excel(
        workbook_info,
        header=[0, 1],
        sheet_name=None,
    )

    cv_objects = []
    num_sheets = len(workbook)

    for sheet_name, sheet_data in workbook.items():
        normalized_columns = [_normalize_excel_column(col) for col in sheet_data.columns]

        last_axis_block = None

        for idx, col in enumerate(normalized_columns):
            role = _excel_column_role(col)

            # Cache axis blocks as we move left-to-right
            if role == "potential":
                ref_idx = None
                if idx + 1 < len(normalized_columns):
                    if _excel_column_role(normalized_columns[idx + 1]) == "reference":
                        ref_idx = idx + 1

                last_axis_block = {
                    "potential_idx": idx,
                    "reference_idx": ref_idx,
                    "reference_label": (
                        _reference_label_from_header(normalized_columns[ref_idx][1])
                        if ref_idx is not None else None
                    ),
                }
                continue

            if role != "current":
                continue

            curve_name = col[0] or f"curve_{idx + 1}"
            if num_sheets > 1:
                curve_name = f"{sheet_name}_{curve_name}"

            potential_idx = None
            reference_idx = None
            reference_label = None

            # First: see whether this current has local axes immediately to the right
            j = idx + 1
            while j < len(normalized_columns):
                next_role = _excel_column_role(normalized_columns[j])
                if next_role not in {"potential", "reference"}:
                    break

                if next_role == "potential" and potential_idx is None:
                    potential_idx = j
                elif next_role == "reference" and reference_idx is None:
                    reference_idx = j

                j += 1

            # Otherwise inherit the last axis block on the left
            if potential_idx is None and last_axis_block is not None:
                potential_idx = last_axis_block["potential_idx"]
                reference_idx = last_axis_block["reference_idx"]
                reference_label = last_axis_block["reference_label"]
            else:
                if reference_idx is not None:
                    reference_label = _reference_label_from_header(
                        normalized_columns[reference_idx][1]
                    )

            if potential_idx is None:
                if options.get("troubleshoot", False):
                    print(f"Skipping '{curve_name}': no potential axis found.")
                continue

            raw_curve_name = _clean_excel_header_value(col[0])
            axis_header_name = _clean_excel_header_value(normalized_columns[potential_idx][0])

            # metadata comes from the first-row header for the trace
            metadata_name = raw_curve_name or axis_header_name or f"curve_{idx + 1}"

            # display name can still include sheet prefix for uniqueness
            display_name = metadata_name
            if num_sheets > 1:
                display_name = f"{sheet_name}_{display_name}"

            cv_data = pd.concat(
                [
                    sheet_data.iloc[:, potential_idx],
                    sheet_data.iloc[:, idx],
                ],
                axis=1,
            ).apply(pd.to_numeric, errors="coerce")

            cv_data = cv_data.dropna(how="any").reset_index(drop=True)
            if cv_data.empty:
                if options.get("troubleshoot", False):
                    print(f"Skipping '{display_name}': no numeric data after cleanup.")
                continue

            reference_shift = None
            if reference_idx is not None:
                reference_shift, spread = _infer_reference_shift(
                    sheet_data.iloc[:, potential_idx],
                    sheet_data.iloc[:, reference_idx],
                )

                # Reject non-constant shifts
                if spread is not None and spread > 1e-4:
                    if options.get("troubleshoot", False):
                        print(
                            f"Reference axis for '{display_name}' was not a constant shift "
                            f"(max spread = {spread:.3g} V); ignoring reference axis."
                        )
                    reference_shift = None
                    reference_label = None

            cv_object = _make_cv_object_from_dataframe(
                display_name,
                cv_data,
                options,
                reference_shift=reference_shift,
                reference_label=reference_label,
                metadata_name=metadata_name,
            )
            cv_objects.append(cv_object)

    if options.get("print", False):
        show_objects(cv_objects, options)

    return cv_objects


def get_data_from_excel(file_path, options=None):
    """Create eCAT objects from an Excel workbook.

    Parameters
    ----------
    file_path : str or path-like
        eCAT Excel workbook with a ``manifest`` sheet, or a curated Excel
        workbook with two header rows and potential/current columns.
    options : dict or ImportOptions, optional
        Import and metadata options. See ``e.describe_options("get_data")``.

    Returns
    -------
    list
        eCAT objects created from workbook traces.

    Examples
    --------
    >>> objects = e.get_data_from_excel("processed_data.xlsx", {"print": False})
    """
    return _create_data_objects_from_excel(file_path, options)

__all__ = [
    "get_data",
    "get_CVs",
    "get_data_from_excel",
]

def get_data(options=None):
    """Read electrochemistry text files from a folder into eCAT objects.
    
    Parameters
    ----------
    options : dict or ImportOptions, optional
        Import, parsing, sorting, and reference-shift options. See ``e.describe_options("get_data")``.
    
    Returns
    -------
    list of echem
        Imported CV, DPV, CA, CP, CPE, or generic electrochemistry objects.
    
    Examples
    --------
    >>> cvs = e.get_data({"folder path": folder, "reference mode": "keyword"})
    """
    ### Add option recursive which is used interchangably with recursive search
    # ---------------------------
    # 1. Normalize user options
    # ---------------------------
    typed_options = ImportOptions.from_options(options)
    options = typed_options.to_legacy_dict()
    options = normalize_legacy_reference_options(options)
    reference_config = resolve_reference_options(options)
    reference_label = reference_config.get("label", None)

    user_supplied_reference_label = (
            "reference label" in options or "shift label" in options
    )

    folder_path = os.path.expanduser(options.get("folder path", "."))
    root_abs = os.path.abspath(folder_path)
    glob_root = glob.escape(root_abs)

    recursive = options.get("recursive search", True)
    recursive_search = "recursively" if recursive else "exclusively"

    # Validate folder
    if not os.path.exists(root_abs):
        print(f"Folder does not exist:\n{_format_path_for_display(root_abs)}")
        return []

    if not os.path.isdir(root_abs):
        print(f"Path exists but is not a folder:\n {_format_path_for_display(root_abs)}")
        return []

    print(f"Searching {recursive_search} through:\n {_format_path_for_display(root_abs)}")

    # ---------------------------
    # 2. Find candidate files
    # ---------------------------
    try:
        if recursive:
            search_pattern = os.path.join(glob_root, "**", "*")
        else:
            search_pattern = os.path.join(glob_root, "*")

        candidates = glob.glob(search_pattern, recursive=recursive)
        all_files = [os.path.abspath(p) for p in candidates if os.path.isfile(p)]
        txt_files = [p for p in all_files if os.path.splitext(p)[1].lower() == ".txt"]

    except Exception as exc:
        print(f"Error while searching folder:\n {_format_path_for_display(root_abs)}\n{exc}")
        return []

    file_paths = sorted(
        txt_files,
        key=lambda p: os.path.normcase(os.path.relpath(p, root_abs)),
    )

    if len(file_paths) == 0:
        if len(all_files) == 0:
            print(f"No files were found in the folder:\n {_format_path_for_display(root_abs)}")
        else:
            suffix_counts = {}
            for p in all_files:
                suffix = os.path.splitext(p)[1]
                if suffix == "":
                    suffix = "[no extension]"
                suffix_counts[suffix] = suffix_counts.get(suffix, 0) + 1

            suffix_summary = ", ".join(
                f"{ext}: {count}" for ext, count in sorted(suffix_counts.items())
            )

            print(
                f"Found {len(all_files)} file(s) in the folder, but none were .txt files.\n"
                f"Folder:\n {_format_path_for_display(root_abs)}\n"
                f"File types found: {suffix_summary}"
            )
        return []
    else:
        s = ""
        if len(file_paths) > 1:
            s = "s"
        print(f"{len(file_paths)} .txt file{s} found.\n")

    # ---------------------------
    # 3. Build raw objects and sort
    # ---------------------------
    raw_file_options = options.copy()
    raw_file_options["shift label"] = reference_label
    raw_file_options["shift guess"] = None
    raw_file_options["shift potential"] = False
    raw_file_options["print"] = False

    object_list = []
    for filepath in file_paths:
        if options.get("troubleshoot", False):
            print("Getting data from", _format_path_for_display(filepath, root_abs))

        try:
            echem_object = echem.from_file(filepath, raw_file_options.copy())
        except Exception as exc:
            display_name = _format_reference_display(filepath, root_abs)
            print(f"Warning: could not convert {display_name}: {exc}")
            continue

        file_folder = os.path.dirname(filepath)
        echem_object.folderpath = _relative_folderpath(file_folder, folder_path)
        object_list.append(echem_object)

    if not object_list:
        print("No .txt files could be converted into eCAT objects.")
        return []

    sort_keys = options.get("sort keys", ["timestamp"])
    if isinstance(sort_keys, str):
        sort_keys = [sort_keys]
    if sort_keys:
        sort_options = {"print": False}
        object_list = sort(object_list.copy(), sort_keys, options=sort_options)

    file_paths = [os.path.abspath(obj.filepath) for obj in object_list]

    # ---------------------------
    # 4. Reference mode summary
    # ---------------------------
    reference_mode = reference_config.get("mode", "none")
    reference_label = reference_config.get("label", None)
    reference_guess = reference_config.get("guess", "auto")
    reference_active = reference_mode != "none" or bool(options.get("reference map"))
    manual_shift = reference_mode == "manual"

    # ---------------------------
    # 5. Resolve reference assignments
    # ---------------------------
    # Keep current _resolve_reference_shifts() machinery for now by feeding it
    # a legacy-compatible shim.
    reference_info = {
        "use_reference_files": False,
        "ref_name": None,
        "ref_mapping": {},
        "ref_shift_guess": {},
        "self_ref_shift_guess": {},
        "self_ref_failures": {},
        "chosen_keyword": None,
    }
    explicit_reference_file = None
    explicit_reference_shift = None

    if reference_mode == "keyword":
        legacy_ref_options = options.copy()
        legacy_ref_options["shift potential"] = reference_config.get("keyword")
        legacy_ref_options["shift guess"] = reference_guess
        legacy_ref_options["shift label"] = reference_label
        legacy_ref_options["allow self reference"] = reference_config.get("allow self reference", True)

        reference_info = _resolve_reference_shifts(
            file_paths=file_paths,
            root_abs=root_abs,
            options=legacy_ref_options,
        )
        reference_info["chosen_keyword"] = reference_config.get("keyword")

    elif reference_mode == "auto":
        keywords = reference_config.get("keywords") or []
        last_error = None

        for keyword in keywords:
            legacy_ref_options = options.copy()
            legacy_ref_options["shift potential"] = keyword
            legacy_ref_options["shift guess"] = reference_guess
            legacy_ref_options["shift label"] = reference_label
            legacy_ref_options["allow self reference"] = reference_config.get("allow self reference", True)

            try:
                candidate_info = _resolve_reference_shifts(
                    file_paths=file_paths,
                    root_abs=root_abs,
                    options=legacy_ref_options,
                )

                # Accept first successful keyword that actually produced a mapping
                if candidate_info.get("use_reference_files", False):
                    reference_info = candidate_info
                    reference_info["chosen_keyword"] = keyword

                    if not user_supplied_reference_label:
                        reference_label = canonical_reference_label(
                            keyword,
                            default=reference_label,
                        )
                    break

            except Exception as exc:
                last_error = exc

        # Optional: print a soft warning if auto found nothing
        if not reference_info.get("use_reference_files", False):
            if last_error is not None and options.get("troubleshoot", False):
                print("Automatic reference search did not resolve any keyword.")
                print(last_error)

    elif reference_mode == "file":
        explicit_reference_file, explicit_reference_shift = (
            _compute_explicit_reference_file_shift(
                reference_config=reference_config,
                root_abs=root_abs,
                options=options,
            )
        )

    use_reference_files = reference_info["use_reference_files"]
    ref_name = reference_info["ref_name"]
    ref_mapping = reference_info["ref_mapping"]
    ref_shift_guess = reference_info["ref_shift_guess"]
    self_ref_shift_guess = reference_info["self_ref_shift_guess"]
    self_ref_failures = reference_info["self_ref_failures"]

    # ---------------------------
    # ---------------------------
    # 6. Attach reference metadata
    # ---------------------------
    reference_records = []

    for i, echem_object in enumerate(object_list):
        filepath_abs = os.path.abspath(echem_object.filepath)

        record = {
            "index": i,
            "filepath": filepath_abs,
            "display_name": _format_reference_display(filepath_abs, root_abs),
            "mode": "none",
            "ref_file": None,
            "shift": None,
            "failure_message": None,
        }

        if manual_shift:
            record["mode"] = "manual"
            record["shift"] = float(reference_config["offset"])

        elif reference_mode == "file":
            record["mode"] = "file"
            record["ref_file"] = explicit_reference_file
            record["shift"] = explicit_reference_shift

        if use_reference_files:
            folder_abs = os.path.abspath(os.path.dirname(filepath_abs))
            basename = os.path.basename(filepath_abs)

            assigned_ref_file = ref_mapping.get(folder_abs)
            is_reference_like = _contains_string_case_insensitive(basename, ref_name)
            is_folder_reference = (
                    assigned_ref_file is not None
                    and os.path.abspath(filepath_abs) == os.path.abspath(assigned_ref_file)
            )
            is_self_reference = (
                    reference_config.get("allow self reference", True)
                    and is_reference_like
                    and not is_folder_reference
            )

            if is_self_reference:
                # Try self-reference first
                if filepath_abs in self_ref_shift_guess:
                    shift_guess = self_ref_shift_guess[filepath_abs]

                    record["mode"] = "self"
                    record["ref_file"] = filepath_abs
                    record["shift"] = shift_guess
                    
                else:
                    # Self-reference failed -> fall back to the designated nearest-ancestor reference
                    if assigned_ref_file is None:
                        raise ValueError(
                            f"No designated folder/ancestor reference file was available for fallback: {filepath_abs}"
                        )

                    fallback_shift = ref_shift_guess[assigned_ref_file]
                    failure_message = self_ref_failures.get(
                        filepath_abs,
                        "Self-reference failed for an unknown reason."
                    )

                    record["mode"] = "fallback"
                    record["ref_file"] = assigned_ref_file
                    record["shift"] = fallback_shift
                    record["failure_message"] = failure_message

            else:
                # Standard designated folder/ancestor reference path
                if assigned_ref_file is not None:
                    try:
                        shift_guess = ref_shift_guess[assigned_ref_file]

                        record["mode"] = "folder"
                        record["ref_file"] = assigned_ref_file
                        record["shift"] = shift_guess
                    except KeyError as exc:
                        display_name = _format_reference_display(assigned_ref_file, root_abs)
                        raise ValueError(
                            f"Reference shift assignment failed for: {display_name}\n"
                            "The designated folder/ancestor reference file was selected, but no "
                            "reference voltage was successfully computed for it."
                        ) from exc
        record["is_cv"] = getattr(echem_object, "type", None) == "Cyclic Voltammetry"

        # Store stable reference metadata on the object
        echem_object.reference_shift = record["shift"]
        echem_object.reference_label = reference_label
        echem_object.reference_mode = record["mode"]
        echem_object.reference_source_file = record["ref_file"]
        echem_object.reference_failure_message = record["failure_message"]

        reference_records.append(record)

    # ---------------------------
    # 6. Print summaries
    # ---------------------------
    _apply_reference_map(
        object_list=object_list,
        reference_records=reference_records,
        reference_map=options.get("reference map"),
        reference_config=reference_config,
        reference_label=reference_label,
        options=options,
    )

    _print_reference_correction_summary(
        reference_config=reference_config,
        reference_info=reference_info,
        reference_records=reference_records,
        root_abs=root_abs,
        explicit_reference_file=explicit_reference_file,
        explicit_reference_shift=explicit_reference_shift,
    )

    if options.get("troubleshoot", False):
        _print_reference_usage_troubleshoot(reference_records, root_abs)

    elif options.get("print", False):
        print_options = options.copy()

        if reference_active:
            extra_cols = [
                "reference shift",
                "reference label",
                "reference mode",
                "reference source",
            ]

            existing_cols = _coerce_display_columns(print_options.get("columns", []))

            for col in extra_cols:
                if col not in existing_cols:
                    existing_cols.append(col)

            print_options["columns"] = existing_cols

        show_objects(object_list, print_options)

    return object_list


def parse_file(filepath, options=None):
    """Load one file and return its standardized parser contract."""
    obj = echem.from_file(filepath, options)
    if getattr(obj, "parse_result", None) is None:
        obj._refresh_parse_result()
    return obj.parse_result
