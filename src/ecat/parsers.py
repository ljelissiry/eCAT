"""Parser helper functions for electrochemical text formats."""

from dataclasses import dataclass, field
from datetime import datetime
import re

import pandas as pd


def parse_ch_timestamp(time_str):
    """
    Parse CH Instruments timestamp strings.

    CH exports commonly use both ``Aug. 27, 2023`` and ``May 7, 2026`` style
    month text, depending on the month/export path.
    """
    text = str(time_str).strip()
    for fmt in ("%b. %d, %Y   %H:%M:%S", "%b %d, %Y   %H:%M:%S"):
        try:
            return datetime.strptime(text, fmt)
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
    mapping = {
        "Cyclic Voltammetry": "CV",
        "Chronoamperometry": "CA",
        "Chronopotentiometry": "CP",
        "Differential Pulse Voltammetry": "DPV",
    }
    return mapping.get(exp_type, exp_type)


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
        "scan_rate",
        "temperature",
        "electrode_area",
        "gas",
        "solvent",
        "compounds",
        "concentrations",
        "segments",
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
    "parse_ch_timestamp",
    "parse_duration_seconds",
    "parse_quiet_time_from_lines",
    "exp_type_short",
]
