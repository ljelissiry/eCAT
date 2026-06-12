"""Parser helper functions for electrochemical text formats."""

from datetime import datetime
import re


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


__all__ = [
    "parse_ch_timestamp",
    "parse_duration_seconds",
    "parse_quiet_time_from_lines",
    "exp_type_short",
]
