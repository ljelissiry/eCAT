"""Parser helper functions for electrochemical text formats."""

from dataclasses import dataclass, field
from datetime import datetime
from io import StringIO
import math
import os
import re
import warnings as _warnings

import pandas as pd


def parse_ch_timestamp(time_str):
    """
    Parse CH Instruments timestamp strings.

    CH exports commonly use both ``Aug. 27, 2023`` and ``May 7, 2026`` style
    month text, depending on the month/export path.
    """
    text = str(time_str).strip()
    candidates = [text]
    normalized_text = re.sub(r"\s+", " ", text)
    normalized_text = re.sub(r"^Sept\.", "Sep.", normalized_text, flags=re.IGNORECASE)
    if normalized_text not in candidates:
        candidates.append(normalized_text)

    for candidate in candidates:
        for fmt in (
            "%B %d, %Y   %H:%M:%S",
            "%b. %d, %Y   %H:%M:%S",
            "%b %d, %Y   %H:%M:%S",
            "%B %d, %Y %H:%M:%S",
            "%b. %d, %Y %H:%M:%S",
            "%b %d, %Y %H:%M:%S",
        ):
            try:
                return datetime.strptime(candidate, fmt)
            except ValueError:
                continue
    return time_str


def parse_duration_seconds(text):
    text = str(text).strip()
    hms = re.search(r"(\d+)\s*:\s*(\d+)\s*:\s*([\d.]+)", text)
    if hms:
        hours = float(hms.group(1))
        minutes = float(hms.group(2))
        seconds = float(hms.group(3))
        return hours * 3600 + minutes * 60 + seconds

    assignment = re.search(
        r"(?:=|:)\s*([\d.eE+\-]+)\s*([a-zA-Zµμ]+)?",
        text,
    )
    numeric = assignment or re.search(r"([\d.eE+\-]+)\s*([a-zA-Zµμ]+)?", text)
    if not numeric:
        return None

    value = float(numeric.group(1))
    unit = "" if numeric.group(2) is None else numeric.group(2).strip().lower()
    unit = unit.replace("µ", "u").replace("μ", "u")
    if unit in {"", "s", "sec", "secs", "second", "seconds"}:
        return value
    if unit in {"ms", "millisecond", "milliseconds"}:
        return value / 1000
    if unit in {"min", "mins", "minute", "minutes"}:
        return value * 60
    if unit in {"h", "hr", "hrs", "hour", "hours"}:
        return value * 3600
    return value


def parse_quiet_time_from_lines(lines):
    for line in lines:
        text = str(line).strip()
        lower = text.lower()
        is_quiet_field = any(
            key in lower
            for key in (
                "quiet time",
                "rest time",
                "equilibration time",
                "conditioning time",
            )
        )
        is_eclab_rest_row = re.match(r"^tr\s*(?:\(|\s)", lower) is not None
        if not is_quiet_field and not is_eclab_rest_row:
            continue
        value = parse_duration_seconds(text)
        if value is not None:
            return float(value)
    return None


def exp_type_short(exp_type):
    text = str(exp_type or "").strip().lower()
    if "cyclic voltammetry" in text:
        return "CV"
    mapping = {
        "Cyclic Voltammetry": "CV",
        "Chronoamperometry": "CA",
        "Chronopotentiometry": "CP",
        "Differential Pulse Voltammetry": "DPV",
    }
    return mapping.get(exp_type, exp_type)


def _read_text_lines(filepath):
    for encoding in ("utf-8-sig", "utf-8", "ISO-8859-1"):
        try:
            with open(filepath, "r", encoding=encoding) as handle:
                return handle.read().splitlines(), encoding
        except UnicodeDecodeError:
            continue
    with open(filepath, "r", encoding="ISO-8859-1", errors="replace") as handle:
        return handle.read().splitlines(), "ISO-8859-1"


def _parse_timestamp(text):
    text = str(text).strip()
    text = re.sub(r"(\.\d{6})\d+$", r"\1", text)
    for fmt in (
        "%m/%d/%Y %H:%M:%S.%f",
        "%m/%d/%Y %H:%M:%S",
        "%Y-%m-%d %H:%M:%S",
        "%d/%m/%Y %H:%M:%S.%f",
        "%d/%m/%Y %H:%M:%S",
    ):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    parsed_ch = parse_ch_timestamp(text)
    return parsed_ch if isinstance(parsed_ch, datetime) else None


def _parse_assignment_float(line):
    match = re.search(r"([-+]?\d+(?:[.,]\d+)?(?:[eE][-+]?\d+)?)", str(line))
    if not match:
        return None
    try:
        return float(match.group(1).replace(",", "."))
    except ValueError:
        return None


def _line_starts_with_label(line, label):
    text = str(line).strip()
    return re.match(rf"^{re.escape(label)}(?:\b|\s|\()", text, flags=re.IGNORECASE) is not None


def _parse_labeled_float(lines, labels, *, target_unit=None):
    labels = (labels,) if isinstance(labels, str) else tuple(labels)
    for line in lines:
        text = str(line).strip()
        matched_label = next(
            (label for label in labels if _line_starts_with_label(text, label)),
            None,
        )
        if matched_label is None:
            continue
        tail = re.sub(
            rf"^{re.escape(matched_label)}(?:\s*\([^)]*\))?\s*(?:=|:)?\s*",
            "",
            text,
            count=1,
            flags=re.IGNORECASE,
        )
        value = _parse_assignment_float(tail)
        if value is None:
            continue
        if target_unit is None:
            return value
        return value * _factor_from_line_unit(text, target_unit)
    return None


def _parse_labeled_string(lines, labels):
    labels = (labels,) if isinstance(labels, str) else tuple(labels)
    for line in lines:
        text = str(line).strip()
        matched_label = next(
            (label for label in labels if _line_starts_with_label(text, label)),
            None,
        )
        if matched_label is None:
            continue
        if "=" in text:
            return text.split("=", 1)[1].strip()
        if ":" in text:
            return text.split(":", 1)[1].strip()
        return text[len(matched_label) :].strip() or None
    return None


def _parse_scan_rate_from_filename(name):
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


def _format_scan_rate_for_warning(scan_rate):
    value = abs(float(scan_rate))
    if value < 1:
        return f"{value * 1000:g} mV/s"
    return f"{value:g} V/s"


def _format_file_for_warning(filepath, display_root=None):
    filepath_abs = os.path.abspath(filepath)
    display = os.path.basename(filepath_abs)
    if display_root is not None:
        root_abs = os.path.abspath(display_root)
        try:
            if os.path.commonpath([root_abs, filepath_abs]) == root_abs:
                display = os.path.relpath(filepath_abs, root_abs)
        except ValueError:
            display = os.path.basename(filepath_abs)
    return f"`{display.replace(os.sep, '/')}`"


def _scan_rate_mismatch_warning(filepath, header_scan_rate, display_root=None):
    filename_scan_rate = _parse_scan_rate_from_filename(os.path.basename(filepath))
    if header_scan_rate is None or filename_scan_rate is None:
        return None
    try:
        header_value = float(header_scan_rate)
        filename_value = float(filename_scan_rate)
    except (TypeError, ValueError):
        return None
    if not (math.isfinite(header_value) and math.isfinite(filename_value)):
        return None
    if math.isclose(header_value, filename_value, rel_tol=1e-9, abs_tol=1e-12):
        return None
    display_name = _format_file_for_warning(filepath, display_root)
    return (
        f"Scan rate mismatch for {display_name}: header reports "
        f"{_format_scan_rate_for_warning(header_value)}, but filename suggests "
        f"{_format_scan_rate_for_warning(filename_value)}; using header value."
    )


def _factor_from_line_unit(line, target_unit):
    text = str(line).replace("µ", "u").replace("μ", "u")
    target_unit = str(target_unit)
    unit_patterns = {
        "V": [
            (r"\bmV\b", "mV"),
            (r"\buV\b", "uV"),
            (r"\bnV\b", "nV"),
            (r"\bV\b", "V"),
        ],
        "A": [
            (r"\bmA\b", "mA"),
            (r"\buA\b", "uA"),
            (r"\bnA\b", "nA"),
            (r"\bpA\b", "pA"),
            (r"\bA\b", "A"),
        ],
        "s": [
            (r"\bms\b", "ms"),
            (r"\bmin\b", "min"),
            (r"\bh\b", "h"),
            (r"\bsec(?:ond)?s?\b", "s"),
            (r"\bs\b", "s"),
        ],
    }
    for pattern, source_unit in unit_patterns.get(target_unit, []):
        if re.search(pattern, text):
            return _unit_factor(source_unit, target_unit)
    return 1.0


def _unit_factor(unit, target):
    unit = "" if unit is None else str(unit).strip()
    unit = unit.replace("µ", "u").replace("μ", "u")
    target = str(target).strip()
    aliases = {
        "": 1.0,
        "V": 1.0,
        "mV": 1e-3,
        "uV": 1e-6,
        "nV": 1e-9,
        "A": 1.0,
        "mA": 1e-3,
        "uA": 1e-6,
        "nA": 1e-9,
        "pA": 1e-12,
        "s": 1.0,
        "sec": 1.0,
        "secs": 1.0,
        "second": 1.0,
        "seconds": 1.0,
        "ms": 1e-3,
        "min": 60.0,
        "mins": 60.0,
        "h": 3600.0,
        "hr": 3600.0,
        "C": 1.0,
        "mC": 1e-3,
        "uC": 1e-6,
    }
    if unit == target:
        return 1.0
    if unit in aliases and target in {"V", "A", "s", "C"}:
        return aliases[unit]
    raise ValueError(f"Unsupported unit conversion from {unit!r} to {target!r}.")


def _split_column_unit(column):
    text = str(column).strip()
    paren = re.search(r"\(([^()]+)\)\s*$", text)
    if paren:
        name = text[: paren.start()].strip()
        return name, paren.group(1).strip()
    slash = re.search(r"/\s*([^\s,/;]+)\s*$", text)
    if slash:
        name = text[: slash.start()].strip()
        return name, slash.group(1).strip()
    return text, None


def _normalize_column_label(column):
    name, _unit = _split_column_unit(column)
    text = name.strip().lower()
    text = text.replace("<", "").replace(">", "")
    text = re.sub(r"[^a-z0-9]+", " ", text).strip()
    return text


def _column_role(column):
    label = _normalize_column_label(column)
    if label in {"time", "t"} or label.startswith("time "):
        return "Time"
    if "cycle" in label:
        return "Cycle"
    if label in {"ns", "step"} or label.endswith(" ns"):
        return "Step"
    if "charge" in label or label in {"dq", "q qo", "q"}:
        return "Charge"
    if "capacity" in label:
        return "Capacity"
    if "current range" in label or label == "i range":
        return None
    if "current" in label or label == "i" or label.endswith(" i"):
        return "Current"
    if "potential" in label or label in {"e", "ewe", "we potential"} or "ewe" in label:
        return "Potential"
    return None


def _canonical_type_and_technique(raw_type):
    text = str(raw_type or "").strip()
    lower = text.lower()
    if "differential" in lower and "pulse" in lower:
        return "Differential Pulse Voltammetry", "DPV"
    if "cyclic" in lower or lower in {"cv", "cva"}:
        return text if text and lower not in {"cv", "cva"} else "Cyclic Voltammetry", "CV"
    if "chronoamperometry" in lower or "amperometric" in lower:
        return "Chronoamperometry", "CA"
    if "chronopotentiometry" in lower or "galvanostatic" in lower or "gcpl" in lower:
        return "Chronopotentiometry", "CP"
    return None, "unknown"


def _potential_reverses_direction(data):
    if data is None:
        return False
    potential_col = next(
        (col for col in data.columns if _column_role(col) == "Potential"),
        None,
    )
    if potential_col is None:
        return False
    potential = pd.to_numeric(data[potential_col], errors="coerce").dropna()
    if len(potential) < 3:
        return False
    delta = potential.diff().dropna()
    tolerance = max(float(potential.abs().max()), 1.0) * 1e-12
    signs = delta[delta.abs() > tolerance].map(lambda value: 1 if value > 0 else -1)
    return bool(len(signs) >= 2 and (signs != signs.iloc[0]).any())


def _technique_from_columns(columns, warnings, data=None):
    roles = {_column_role(col) for col in columns}
    if {"Time", "Potential", "Current"}.issubset(roles):
        if _potential_reverses_direction(data):
            warnings.append(
                "Generic text parser inferred CV-like axes because the potential trajectory reverses direction."
            )
            return "Cyclic Voltammetry", "CV"
        warnings.append(
            "Generic text parser found time, potential, and current axes without a vendor technique marker; technique was left unknown."
        )
        return "Unknown", "unknown"
    if {"Potential", "Current"}.issubset(roles):
        warnings.append(
            "Generic text parser inferred CV-like potential/current axes without a vendor technique marker."
        )
        return "Cyclic Voltammetry", "CV"
    if {"Time", "Current"}.issubset(roles):
        warnings.append(
            "Generic text parser inferred CA-like time/current axes without a vendor technique marker."
        )
        return "Chronoamperometry", "CA"
    if {"Time", "Potential"}.issubset(roles):
        warnings.append(
            "Generic text parser inferred CP-like time/potential axes without a vendor technique marker."
        )
        return "Chronopotentiometry", "CP"
    return "Unknown", "unknown"


def _detect_delimiter(header_line):
    if "\t" in header_line:
        return "\t"
    if ";" in header_line:
        return ";"
    if "," in header_line:
        return ","
    return r"\s+"


def _decimal_separator(lines, delimiter):
    if delimiter == ";":
        data_text = "\n".join(lines[:10])
        if re.search(r"\d,\d", data_text):
            return ","
    if any(re.search(r"\d,\d+[Ee][+\-]?\d+", line) for line in lines[:10]):
        return ","
    return "."


def _read_table(lines, header_idx, delimiter, decimal, *, has_header=True):
    text = "\n".join(lines[header_idx:]) + "\n"
    kwargs = {
        "sep": delimiter,
        "decimal": decimal,
        "header": 0 if has_header else None,
    }
    if delimiter == r"\s+":
        kwargs["engine"] = "python"
    df = pd.read_csv(StringIO(text), **kwargs)
    if delimiter == "," and any("\t" in line for line in lines[header_idx + 1 :]):
        parsed_numeric_rows = df.iloc[1:] if has_header and len(df) > 1 else df
        numeric_rows_need_retry = (
            df.shape[1] <= 1
            or (
                not parsed_numeric_rows.empty
                and df.shape[1] > 1
                and parsed_numeric_rows.iloc[:, 1:].isna().all().all()
            )
        )
        if numeric_rows_need_retry:
            retry_kwargs = dict(kwargs)
            retry_kwargs["sep"] = r"[,\t]+"
            retry_kwargs["engine"] = "python"
            df = pd.read_csv(StringIO(text), **retry_kwargs)
    df = df.dropna(how="all", axis=1).dropna(how="all").reset_index(drop=True)
    df.columns = [str(col).strip() for col in df.columns]
    return df


def _canonicalize_data(df_raw, technique, warnings):
    output = {}
    units = {}
    original_units = {}
    used_roles = set()

    for col in df_raw.columns:
        original_name, original_unit = _split_column_unit(col)
        if original_unit is not None:
            original_units[original_name] = original_unit
        role = _column_role(col)
        if role is None or role in used_roles:
            continue

        target_unit = {
            "Potential": "V",
            "Current": "A",
            "Time": "s",
            "Charge": "C",
        }.get(role)
        values = pd.to_numeric(df_raw[col], errors="coerce")
        if target_unit is not None:
            try:
                values = values * _unit_factor(original_unit or target_unit, target_unit)
                units[role] = target_unit
            except ValueError:
                units[role] = original_unit
                warnings.append(
                    f"Could not convert {role} unit {original_unit!r}; values were preserved."
                )
        elif original_unit is not None:
            units[role] = original_unit

        output[role] = values
        used_roles.add(role)

    if technique == "unknown":
        for col in df_raw.columns:
            if _column_role(col) is not None:
                continue
            original_name, original_unit = _split_column_unit(col)
            values = pd.to_numeric(df_raw[col], errors="coerce")
            if values.dropna().empty:
                continue
            clean_name = str(original_name).strip() or str(col).strip()
            output[clean_name] = values
            if original_unit is not None:
                units[clean_name] = original_unit
                original_units[clean_name] = original_unit

    ordered = {
        "CV": ["Potential", "Current", "Time", "Cycle", "Segment"],
        "DPV": ["Potential", "Current", "Time", "Cycle", "Segment"],
        "CA": ["Time", "Current", "Potential", "Step", "Cycle", "Charge"],
        "CP": ["Time", "Potential", "Current", "Step", "Cycle", "Charge", "Capacity"],
    }.get(technique, list(output))
    ordered += [name for name in output if name not in ordered]
    data = pd.DataFrame({name: output[name] for name in ordered if name in output})
    if data.empty and not output:
        data = df_raw.copy()
        data.columns = [str(col).strip() for col in data.columns]
        data = data.apply(pd.to_numeric, errors="coerce").dropna(how="all").reset_index(drop=True)
        return data, units, original_units
    required = {
        "CV": ["Potential", "Current"],
        "DPV": ["Potential", "Current"],
        "CA": ["Time", "Current"],
        "CP": ["Time", "Potential"],
    }.get(technique, [])
    if required:
        data = data.dropna(subset=required).reset_index(drop=True)
    else:
        data = data.dropna(how="all").reset_index(drop=True)
    return data, units, original_units


def _parse_eclab_header_count(lines):
    for line in lines[:20]:
        if "Nb header lines" in line:
            value = _parse_assignment_float(line)
            if value is not None:
                return int(value)
            raise ValueError("Could not parse number of header lines in EC-Lab file.")
    return None


def _parse_eclab_technique(lines):
    for line in lines[:20]:
        text = str(line).strip()
        if not text or "EC-Lab" in text or "Nb header lines" in text:
            continue
        if any(
            key in text.lower()
            for key in ("voltammetry", "chronoamperometry", "chronopotentiometry", "galvanostatic")
        ):
            return text
    return None


def _parse_header_settings(lines):
    settings = {}
    for line in lines:
        text = str(line).strip()
        if not text:
            continue
        if "=" in text:
            key, value = text.split("=", 1)
            settings[key.strip()] = value.strip()
            continue
        if ":" in text:
            key, value = text.split(":", 1)
            settings[key.strip()] = value.strip()
            continue
        match = re.match(r"^([A-Za-z0-9_<>/\-'(). ]+?)\s{2,}(.+?)\s*$", text)
        if match:
            settings[match.group(1).strip()] = match.group(2).strip()
    return settings


def _parse_acquisition_start(lines):
    for line in lines:
        if "Acquisition started" in line:
            raw = line.split(":", 1)[1].strip() if ":" in line else line
            return _parse_timestamp(raw), raw
    return None, None


def _parse_scan_rate(lines):
    for idx, line in enumerate(lines):
        lower = str(line).lower()
        if "scan rate" in lower or "de/dt" in lower or "de/dt" in lower.replace(" ", ""):
            value = _parse_assignment_float(line)
            if value is None:
                continue
            unit = None
            if "mv/s" in lower:
                unit = "mV/s"
            elif "v/s" in lower:
                unit = "V/s"
            elif idx + 1 < len(lines):
                next_line = str(lines[idx + 1]).strip()
                if re.search(r"(?i)\bmV\s*/?\s*s\b", next_line):
                    unit = "mV/s"
                elif re.search(r"(?i)\bV\s*/?\s*s\b", next_line):
                    unit = "V/s"
            if unit == "mV/s":
                return value * 1e-3
            return value
    return None


def _parse_segments(lines, df_raw=None):
    for line in lines:
        text = str(line).strip()
        if re.match(r"^(?:N|Segment|Number of segments)\b", text, flags=re.IGNORECASE):
            value = _parse_assignment_float(text)
            if value is not None:
                return int(value)
        if "nc cycles" in text.lower():
            value = _parse_assignment_float(text)
            if value is not None and value > 0:
                return int(value)
    if df_raw is not None:
        for col in df_raw.columns:
            if _column_role(col) == "Cycle":
                values = pd.to_numeric(df_raw[col], errors="coerce").dropna()
                if not values.empty:
                    return int(values.nunique())
    return None


def _parse_reference_electrode(lines):
    for line in lines:
        if "reference electrode" in str(line).lower() and ":" in str(line):
            return str(line).split(":", 1)[1].strip()
    return None


def _parse_eclab_potential_metadata(lines):
    values = {}
    key_map = {
        "ei": "init_E",
        "ef": "final_E",
        "e1": "E1",
        "e2": "E2",
    }
    for line in lines:
        text = str(line).strip()
        match = re.match(r"^(Ei|Ef|E1|E2)\s*\(V\)\s+([-+]?\d+(?:[.,]\d+)?)", text)
        if match:
            values[key_map[match.group(1).lower()]] = float(match.group(2).replace(",", "."))
    extrema = [value for key, value in values.items() if key in {"init_E", "final_E", "E1", "E2"}]
    if extrema:
        values["min_E"] = min(extrema)
        values["max_E"] = max(extrema)
    return values


def _parse_ir_compensation(lines):
    comp_r = _parse_labeled_float(lines, ("Comp R", "IR-Comp. Value"))
    uncomp_r = _parse_labeled_float(lines, ("UC R", "Uncomp R"))
    if comp_r is not None and uncomp_r is None:
        uncomp_r = 0.0
    percent = None
    if comp_r is not None and uncomp_r is not None:
        total_r = comp_r + uncomp_r
        if total_r != 0:
            percent = 100 * comp_r / total_r
    return {
        "ir_comp_resistance": comp_r,
        "ir_uncomp_resistance": uncomp_r,
        "ir_comp_percent": percent,
    }


def _time_step_from_data(data):
    if "Time" not in data or len(data) < 2:
        return None
    diffs = pd.to_numeric(data["Time"], errors="coerce").diff().dropna()
    diffs = diffs[diffs > 0]
    if diffs.empty:
        return None
    return float(diffs.median())


def _axis_delta_from_data(data, axis):
    if axis not in data or len(data) < 2:
        return None
    values = pd.to_numeric(data[axis], errors="coerce").dropna()
    if len(values) < 2:
        return None
    return float(abs(values.iloc[1] - values.iloc[0]))


def _potential_extrema_from_data(data):
    if "Potential" not in data or data.empty:
        return {}
    values = pd.to_numeric(data["Potential"], errors="coerce").dropna()
    if values.empty:
        return {}
    return {
        "init_E": float(values.iloc[0]),
        "final_E": float(values.iloc[-1]),
        "min_E": float(values.min()),
        "max_E": float(values.max()),
    }


def _parse_ch_metadata(header_lines, data, technique):
    metadata = {}
    if header_lines:
        parsed_timestamp = parse_ch_timestamp(header_lines[0])
        if isinstance(parsed_timestamp, datetime):
            metadata["acquisition_start"] = parsed_timestamp
        else:
            metadata["timestamp"] = parsed_timestamp

    metadata.update(
        {
            key: value
            for key, value in _parse_ir_compensation(header_lines).items()
            if value is not None
        }
    )

    if technique in {"CV", "DPV"}:
        metadata["init_E"] = _parse_labeled_float(header_lines, "Init E", target_unit="V")
        metadata["final_E"] = _parse_labeled_float(header_lines, "Final E", target_unit="V")
        high_E = _parse_labeled_float(header_lines, "High E", target_unit="V")
        low_E = _parse_labeled_float(header_lines, "Low E", target_unit="V")
        extrema = [
            value
            for value in (
                metadata.get("init_E"),
                metadata.get("final_E"),
                high_E,
                low_E,
            )
            if value is not None
        ]
        data_extrema = _potential_extrema_from_data(data)
        if metadata.get("final_E") is None:
            metadata["final_E"] = data_extrema.get("final_E")
        if extrema:
            metadata["min_E"] = min(extrema)
            metadata["max_E"] = max(extrema)
        else:
            metadata.update(data_extrema)
        metadata["sample_int"] = _parse_labeled_float(
            header_lines,
            "Sample Interval",
            target_unit="V" if technique in {"CV", "DPV"} else "s",
        )
        metadata["sensitivity"] = _parse_labeled_float(header_lines, "Sensitivity")
        if technique == "DPV":
            metadata["incr_E"] = _parse_labeled_float(header_lines, "Incr E", target_unit="V")
            metadata["amplitude"] = _parse_labeled_float(header_lines, "Amplitude", target_unit="V")
            metadata["pulse_width"] = _parse_labeled_float(header_lines, "Pulse Width", target_unit="s")
            metadata["sample_width"] = _parse_labeled_float(header_lines, "Sample Width", target_unit="s")
            metadata["pulse_period"] = _parse_labeled_float(header_lines, "Pulse Period", target_unit="s")
            metadata["comp_R"] = metadata.get("ir_comp_resistance")

    if technique == "CA":
        metadata["init_E"] = _parse_labeled_float(header_lines, "Init E", target_unit="V")
        metadata["sample_interval"] = _parse_labeled_float(
            header_lines,
            "Sample Interval",
            target_unit="s",
        )
        metadata["sample_int"] = metadata.get("sample_interval")
        metadata["run_time"] = _parse_labeled_float(header_lines, "Run Time", target_unit="s")
        metadata["sensitivity"] = _parse_labeled_float(header_lines, "Sensitivity")

    if technique == "CP":
        metadata["cathodic_current"] = _parse_labeled_float(
            header_lines,
            "Cathodic Current",
            target_unit="A",
        )
        metadata["anodic_current"] = _parse_labeled_float(
            header_lines,
            "Anodic Current",
            target_unit="A",
        )
        metadata["init_PN"] = _parse_labeled_string(header_lines, "Init P/N")
        metadata["sample_int"] = _parse_labeled_float(
            header_lines,
            "Data Storage Interval",
            target_unit="s",
        )
        metadata["sample_interval"] = metadata.get("sample_int")
        metadata["high_E_limit"] = _parse_labeled_float(header_lines, "High E Limit", target_unit="V")
        metadata["low_E_limit"] = _parse_labeled_float(header_lines, "Low E Limit", target_unit="V")
        metadata["cathodic_time"] = _parse_labeled_float(header_lines, "Cathodic Time", target_unit="s")
        metadata["anodic_time"] = _parse_labeled_float(header_lines, "Anodic Time", target_unit="s")
        metadata.update(_potential_extrema_from_data(data))

    return {key: value for key, value in metadata.items() if value is not None}


def _parse_basi_metadata(header_lines, data, technique):
    metadata = {}
    metadata.update(
        {
            key: value
            for key, value in _parse_ir_compensation(header_lines).items()
            if value is not None
        }
    )
    if technique == "CV":
        initial = _parse_labeled_float(header_lines, "Initial Potential", target_unit="V")
        switch_1 = _parse_labeled_float(header_lines, "Switching Potential 1", target_unit="V")
        switch_2 = _parse_labeled_float(header_lines, "Switching Potential 2", target_unit="V")
        final = _parse_labeled_float(header_lines, "Final Potential", target_unit="V")
        metadata["init_E"] = initial
        metadata["final_E"] = final
        extrema = [value for value in (initial, switch_1, switch_2, final) if value is not None]
        if extrema:
            metadata["min_E"] = min(extrema)
            metadata["max_E"] = max(extrema)
        else:
            metadata.update(_potential_extrema_from_data(data))
        metadata["sample_int"] = _parse_labeled_float(
            header_lines,
            "Sample Interval",
            target_unit="V",
        )
    if technique in {"CP", "CA"}:
        metadata.update(_potential_extrema_from_data(data))
        sample_interval = _time_step_from_data(data)
        metadata["sample_interval"] = sample_interval
        metadata["sample_int"] = sample_interval
    return {key: value for key, value in metadata.items() if value is not None}


def _cp_currents_from_step_table(step_table):
    is_step = step_table.get("Is", {})
    values = is_step.get("values", [])
    units = is_step.get("units", [])
    converted = []
    for idx, value in enumerate(values):
        unit = units[idx] if idx < len(units) else "A"
        try:
            converted.append(value * _unit_factor(unit, "A"))
        except ValueError:
            converted.append(value)
    nonzero = [value for value in converted if value != 0]
    positives = [value for value in nonzero if value > 0]
    negatives = [value for value in nonzero if value < 0]
    metadata = {}
    if positives:
        metadata["anodic_current"] = positives[0]
    if negatives:
        metadata["cathodic_current"] = negatives[0]
    return metadata


def _parse_step_table(lines):
    table = {}
    for idx, line in enumerate(lines):
        text = str(line).strip()
        if not re.match(r"^(Is|dI|Ei|dt|dts|ts|t1|dt1)\b", text):
            continue
        label = text.split()[0]
        values = []
        for token in re.split(r"\s+", text)[1:]:
            try:
                values.append(float(token.replace(",", ".")))
            except ValueError:
                pass
        units = []
        if idx + 1 < len(lines):
            next_line = str(lines[idx + 1]).strip()
            if next_line.lower().startswith(f"unit {label.lower()}"):
                units = re.split(r"\s+", next_line)[2:]
        table[label] = {"values": values, "units": units}
    return table


def _count_nonzero_step_values(step_table, label):
    values = step_table.get(label, {}).get("values", [])
    nonzero = [value for value in values if value != 0]
    return len(nonzero) or None


def _base_metadata(filepath, software, exp_type, technique):
    path = os.path.abspath(filepath)
    return {
        "name": os.path.splitext(os.path.basename(path))[0],
        "filepath": path,
        "folderpath": os.path.dirname(path),
        "type": exp_type,
        "software": software,
        "experiment_subtype": None,
        "scan_rate": None,
        "segments": None,
        "quiet_time": None,
        "sample_interval": None,
        "sample_int": None,
        "delta_x": None,
        "init_E": None,
        "final_E": None,
        "min_E": None,
        "max_E": None,
        "run_time": None,
        "sensitivity": None,
        "incr_E": None,
        "amplitude": None,
        "pulse_width": None,
        "sample_width": None,
        "pulse_period": None,
        "comp_R": None,
        "cathodic_current": None,
        "anodic_current": None,
        "init_PN": None,
        "high_E_limit": None,
        "low_E_limit": None,
        "cathodic_time": None,
        "anodic_time": None,
        "acquisition_start": None,
        "working_electrode": None,
        "counter_electrode": None,
        "reference_electrode": None,
        "ir_comp_resistance": None,
        "ir_uncomp_resistance": None,
        "ir_comp_percent": None,
        "gas": None,
        "solvent": None,
        "compounds": [],
        "concentrations": [],
        "reference_shift": None,
        "reference_label": None,
        "reference_mode": "none",
        "reference_source_file": None,
    }


def _add_required_warnings(metadata, technique, warnings):
    if metadata.get("acquisition_start") is None:
        warnings.append("Parser metadata is missing an acquisition timestamp.")
    if technique in {"CV", "DPV"} and metadata.get("scan_rate") is None:
        warnings.append(f"Parser metadata is missing scan rate for {technique}.")
    if technique == "CA":
        warnings.append("CA parser metadata may be missing applied-potential step structure.")
    if technique == "CP":
        warnings.append("CP parser metadata may be missing applied-current step structure.")


def _make_parse_result(
    *,
    filepath,
    software,
    parser,
    technique_raw,
    lines,
    header_lines,
    header_idx,
    df_raw,
    delimiter,
    decimal,
    encoding,
    display_root=None,
    inferred=False,
):
    warnings = []
    exp_type, technique = _canonical_type_and_technique(technique_raw)
    if technique == "unknown":
        if str(technique_raw or "").strip():
            exp_type = str(technique_raw).strip()
        else:
            exp_type, technique = _technique_from_columns(df_raw.columns, warnings, df_raw)
            inferred = True

    data, units, original_units = _canonicalize_data(df_raw, technique, warnings)
    step_table = _parse_step_table(header_lines)
    metadata = _base_metadata(filepath, software, exp_type, technique)
    metadata["experiment_subtype"] = technique_raw
    header_scan_rate = _parse_scan_rate(header_lines)
    filename_scan_rate = _parse_scan_rate_from_filename(os.path.basename(filepath))
    metadata["scan_rate"] = header_scan_rate if header_scan_rate is not None else filename_scan_rate
    scan_rate_warning = _scan_rate_mismatch_warning(
        filepath,
        header_scan_rate,
        display_root=display_root,
    )
    if scan_rate_warning is not None:
        warnings.append(scan_rate_warning)
        _warnings.warn(scan_rate_warning, UserWarning, stacklevel=3)
    metadata["segments"] = _parse_segments(header_lines, df_raw)
    if technique == "CP":
        metadata["segments"] = _count_nonzero_step_values(step_table, "Is") or metadata["segments"]
    metadata["quiet_time"] = parse_quiet_time_from_lines(header_lines)
    metadata["acquisition_start"], acquisition_start_raw = _parse_acquisition_start(header_lines)
    metadata["reference_electrode"] = _parse_reference_electrode(header_lines)
    if software == "EC-Lab":
        metadata.update(_parse_eclab_potential_metadata(header_lines))
    if software == "CH":
        metadata.update(_parse_ch_metadata(header_lines, data, technique))
        acquisition_start_raw = header_lines[0] if header_lines else acquisition_start_raw
    elif software == "BASI":
        metadata.update(_parse_basi_metadata(header_lines, data, technique))
    if technique in {"CA", "CP"} and "Time" in data and len(data) >= 2:
        sample_interval = _time_step_from_data(data)
        if sample_interval is not None:
            metadata["sample_interval"] = sample_interval
            metadata["sample_int"] = sample_interval
        if technique == "CA":
            time_values = pd.to_numeric(data["Time"], errors="coerce").dropna()
            if not time_values.empty and metadata.get("run_time") is None:
                metadata["run_time"] = float(time_values.max() - time_values.min())
    if technique in {"CV", "DPV"} and metadata.get("delta_x") is None:
        metadata["delta_x"] = _axis_delta_from_data(data, "Potential")
    elif technique in {"CA", "CP"} and metadata.get("delta_x") is None:
        metadata["delta_x"] = _time_step_from_data(data)
    if technique == "CP":
        metadata.update(_cp_currents_from_step_table(step_table))
    if technique == "CV" and metadata["segments"] is None:
        metadata["segments"] = 1
        warnings.append("CV segment count was inferred from the data table.")
    if inferred:
        metadata["experiment_subtype"] = None

    _add_required_warnings(metadata, technique, warnings)
    data.attrs["units"] = dict(units)
    raw_metadata = {
        "header_lines": list(header_lines),
        "data_header_line": lines[header_idx] if header_idx < len(lines) else "",
        "original_columns": [str(col).strip() for col in df_raw.columns],
        "original_units": original_units,
        "delimiter": delimiter,
        "decimal_separator": decimal,
        "encoding": encoding,
        "header_line_count": header_idx + 1,
        "acquisition_start_raw": acquisition_start_raw,
        "technique_raw": technique_raw,
        "settings": _parse_header_settings(header_lines),
        "technique_params": {},
        "loop_table": {},
        "step_table": step_table,
        "external_channels": {},
        "parser_notes": "Technique inferred from canonical axes." if inferred else "",
    }
    return ParseResult(
        data=data,
        units=units,
        technique=technique,
        software=software,
        metadata=metadata,
        warnings=list(dict.fromkeys(warnings)),
        raw_metadata=raw_metadata,
        source=os.path.abspath(filepath),
        parser=parser,
    )


def parse_text_file_to_result(filepath, options=None):
    """Parse a text electrochemistry export into a pre-promotion ParseResult."""
    options = {} if options is None else dict(options)
    lines, encoding = _read_text_lines(filepath)
    nonempty_header = "\n".join(lines[:20])
    software = options.get("software")
    if software is None:
        if "EC-Lab" in nonempty_header or "EC-LAB" in nonempty_header:
            software = "EC-Lab"
        elif any("Instrument Model" in line and "CH" in line for line in lines[:20]):
            software = "CH"
        elif any("Experiment Type" in line for line in lines[:20]) or any(
            str(line).strip().upper() == "BASI" for line in lines[:20]
        ):
            software = "BASI"
        elif "NOVA" in nonempty_header or any("WE(1)." in line for line in lines[:20]):
            software = "NOVA"
        elif "Ivium" in nonempty_header:
            software = "IviumSoft"
        else:
            software = "Generic Text"

    if software == "EC-Lab":
        header_count = _parse_eclab_header_count(lines)
        if header_count is None:
            raise ValueError("Header line count not found in EC-Lab file.")
        header_idx = header_count - 1
        header_lines = lines[:header_idx]
        technique_raw = _parse_eclab_technique(lines) or options.get("experiment type")
    else:
        header_idx = None
        numeric_only = False
        for idx, line in enumerate(lines):
            roles = {_column_role(token) for token in re.split(r"\t|;|,|\s+", line.strip()) if token}
            lower = line.lower()
            looks_like_table_header = (
                "/" in line
                or "\t" in line
                or "," in line
                or ";" in line
                or re.search(r"\([^)]*(?:v|a|s|sec|c)[^)]*\)", lower) is not None
            )
            if (
                looks_like_table_header
                and (
                    ("potential" in lower and "current" in lower)
                    or (
                        "potential" in lower
                        and re.search(r"/\s*(?:m?a|u?a|n?a|p?a)\b", lower) is not None
                    )
                    or ("time" in lower and ("current" in lower or "potential" in lower))
                    or {"Potential", "Current"}.issubset(roles)
                    or {"Time", "Current"}.issubset(roles)
                    or {"Time", "Potential"}.issubset(roles)
                )
            ):
                header_idx = idx
                break
            tokens = [token for token in re.split(r"\t|;|,|\s+", line.strip()) if token]
            if tokens:
                try:
                    [float(token) for token in tokens]
                except ValueError:
                    pass
                else:
                    header_idx = idx
                    numeric_only = True
                    break
        if header_idx is None:
            raise ValueError("Could not find a text data table header.")
        header_lines = lines[:header_idx]
        technique_raw = options.get("experiment type")
        if technique_raw is None and software == "CH" and len(header_lines) >= 2:
            technique_raw = str(header_lines[1]).strip()
        elif technique_raw is None and software == "BASI":
            for line in header_lines:
                if "Experiment Type" in line:
                    technique_raw = line.split(":", 1)[1].strip()
                    break
        if technique_raw is None:
            for line in header_lines:
                if "technique" in line.lower() and ":" in line:
                    technique_raw = line.split(":", 1)[1].strip()
                    break

    delimiter = _detect_delimiter(lines[header_idx])
    decimal = _decimal_separator(lines[header_idx + 1 :], delimiter)
    df_raw = _read_table(
        lines,
        header_idx,
        delimiter,
        decimal,
        has_header=not locals().get("numeric_only", False),
    )
    return _make_parse_result(
        filepath=filepath,
        software=software,
        parser=software,
        technique_raw=technique_raw,
        lines=lines,
        header_lines=header_lines,
        header_idx=header_idx,
        df_raw=df_raw,
        delimiter=delimiter,
        decimal=decimal,
        encoding=encoding,
        display_root=options.get("_display root"),
        inferred=technique_raw is None,
    )


def _copy_if_dataframe(value):
    if isinstance(value, pd.DataFrame):
        return value.copy()
    return value


def _copy_dict(value):
    if isinstance(value, dict):
        return dict(value)
    return {}


def _object_metadata(obj):
    keys = [
        "name",
        "filepath",
        "folderpath",
        "type",
        "timestamp",
        "scan_rate",
        "temperature",
        "electrode_area",
        "delta_x",
        "init_E",
        "final_E",
        "min_E",
        "max_E",
        "gas",
        "solvent",
        "compounds",
        "concentrations",
        "zero_concentration_compounds",
        "zero_concentrations",
        "segments",
        "quiet_time",
        "sample_int",
        "sample_interval",
        "run_time",
        "sensitivity",
        "incr_E",
        "amplitude",
        "pulse_width",
        "sample_width",
        "pulse_period",
        "comp_R",
        "cathodic_current",
        "anodic_current",
        "init_PN",
        "high_E_limit",
        "low_E_limit",
        "cathodic_time",
        "anodic_time",
        "reference_shift",
        "reference_label",
        "reference_mode",
        "reference_source_file",
        "reference_failure_message",
        "ir_comp_resistance",
        "ir_uncomp_resistance",
        "ir_comp_percent",
    ]
    metadata = {}
    for key in keys:
        if hasattr(obj, key):
            metadata[key] = getattr(obj, key)
    return metadata


@dataclass(slots=True)
class ParseResult:
    """Standard parser contract for one loaded electrochemistry file/object."""

    data: pd.DataFrame
    units: dict = field(default_factory=dict)
    technique: str | None = None
    software: str | None = None
    metadata: dict = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    raw_metadata: dict = field(default_factory=dict)
    source: str | None = None
    parser: str | None = None

    @classmethod
    def from_object(
        cls,
        obj,
        *,
        parser=None,
        warnings=None,
        raw_metadata=None,
        copy_data=True,
    ):
        data = getattr(obj, "data", pd.DataFrame())
        if copy_data:
            data = _copy_if_dataframe(data)
        units = _copy_dict(getattr(obj, "units", {}))
        metadata = _object_metadata(obj)
        technique = exp_type_short(getattr(obj, "type", None))
        if technique == getattr(obj, "type", None):
            technique = getattr(obj, "type", None)
        return cls(
            data=data,
            units=units,
            technique=technique,
            software=getattr(obj, "software", None),
            metadata=metadata,
            warnings=[] if warnings is None else list(warnings),
            raw_metadata={} if raw_metadata is None else dict(raw_metadata),
            source=getattr(obj, "filepath", None),
            parser=parser or getattr(obj, "software", None),
        )


__all__ = [
    "ParseResult",
    "parse_text_file_to_result",
    "parse_ch_timestamp",
    "parse_duration_seconds",
    "parse_quiet_time_from_lines",
    "exp_type_short",
]
