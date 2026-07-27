"""Metadata and label-formatting helpers."""

from datetime import datetime
import os
import re


_SI_PREFIX_EXPONENTS = {
    "": 0,
    "n": -9,
    "u": -6,
    "μ": -6,
    "m": -3,
}


def _conversion_factor(unit_str, to_unit_str=None):
    from_unit = str(unit_str).replace("μ", "u")
    from_prefix = from_unit[:-1] if from_unit.endswith("M") and len(from_unit) > 1 else ""
    from_exp = _SI_PREFIX_EXPONENTS.get(from_prefix, 0)

    if to_unit_str is None:
        return 10**from_exp

    to_unit = str(to_unit_str).replace("μ", "u")
    to_prefix = to_unit[:-1] if to_unit.endswith("M") and len(to_unit) > 1 else ""
    to_exp = _SI_PREFIX_EXPONENTS.get(to_prefix, 0)
    return 10 ** (from_exp - to_exp)


def get_file_times(filepath):
    stat = os.stat(filepath)
    modification_time = datetime.fromtimestamp(stat.st_mtime)

    birthtime = getattr(stat, "st_birthtime", None)
    if birthtime is not None:
        creation_time = datetime.fromtimestamp(birthtime)
    else:
        creation_time = modification_time

    return creation_time, modification_time


def parse_concentration_value_and_unit(concentration_str):
    """
    Return a numeric value plus display/base unit for one stored concentration.

    Molar units are converted to M. Dimensionless composition units such as
    mole fraction are kept as their displayed numeric values.
    """
    text = str(concentration_str).strip().replace("μ", "u")
    match = re.match(
        r"^([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*([nmu]?M|L|%|equiv|x)$",
        text,
    )
    if match is None:
        raise ValueError(
            "Invalid concentration string format. Example valid formats: "
            "'100nM', '50uM', '0.1M', '1L', '5%', '2equiv', '0.8x'"
        )

    number = float(match.group(1))
    unit = match.group(2).replace("u", "μ")

    if unit.endswith("M"):
        return number * _conversion_factor(unit, "M"), "M"

    return number * _conversion_factor(unit), unit


def concentration_to_float(concentration_str):
    value, _unit = parse_concentration_value_and_unit(concentration_str)
    return value


SUBSCRIPT_TRANSLATION = str.maketrans("0123456789+-()", "₀₁₂₃₄₅₆₇₈₉⁺⁻₍₎")
SUPERSCRIPT_TRANSLATION = str.maketrans("0123456789+-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻")


_CHARGE_SIGN_RE = r"[+\-−]"
_CHARGE_PIECE_RE = rf"(?:\d+{_CHARGE_SIGN_RE}|{_CHARGE_SIGN_RE}\d+|{_CHARGE_SIGN_RE}|0)"
_CHARGE_TOKEN_RE = re.compile(
    rf"(?P<charge>(?:{_CHARGE_PIECE_RE})(?:/{_CHARGE_PIECE_RE})+|"
    r"\d+[+-]|[+-]\d+|[+-])(?=$|[\s,;)])"
)


def _format_charge_token(token, mode):
    if mode == "html":
        return f"<sup>{token.replace('-', '−')}</sup>"
    if mode == "unicode":
        return token.replace("−", "-").translate(SUPERSCRIPT_TRANSLATION)
    if mode == "mathtext":
        return "$^{" + token.replace("−", "-").replace("-", r"-") + "}$"
    raise ValueError("mode must be one of: 'mathtext', 'html', 'unicode', 'plain'")


def _charge_has_formula_prefix(text, start):
    if start <= 0:
        return False
    before = text[:start]
    token = re.split(r"[\s,;/]+", before)[-1]
    if not token:
        return False
    if token[-1] in "])":
        return True
    return bool(re.search(r"[A-Z]", token))


def _protect_charge_notation(label, mode):
    replacements = {}
    parts = []
    last = 0
    index = 0
    for match in _CHARGE_TOKEN_RE.finditer(label):
        if not _charge_has_formula_prefix(label, match.start()):
            continue
        placeholder = f"\ue000ECAT_CHARGE_{index}\ue001"
        replacements[placeholder] = _format_charge_token(match.group("charge"), mode)
        parts.append(label[last:match.start()])
        parts.append(placeholder)
        last = match.end()
        index += 1
    if not replacements:
        return label, replacements
    parts.append(label[last:])
    return "".join(parts), replacements


def format_chemical_formulas(label, mode="mathtext"):
    """
    Format chemical formulas using one of several output modes.

    Parameters
    ----------
    label : str | object
    mode : {'mathtext', 'html', 'unicode', 'plain'}
    """
    if not isinstance(label, str):
        label = str(label)

    label = re.sub(r"(?<=\d)\s+%", "%", label)

    if mode == "plain":
        return label

    label, charge_replacements = _protect_charge_notation(label, mode)

    formula_pattern = r"([A-Za-z\(\)\]\-\+]+)(\d*)"

    def repl(match):
        element, count = match.groups()
        if (
            not count
            or "-" in element
            or not re.search(r"[A-Za-z]", element)
            or element.endswith("(")
        ):
            return element + count

        if mode == "mathtext":
            return f"{element}$_{count}$"
        if mode == "html":
            return f"{element}<sub>{count}</sub>"
        if mode == "unicode":
            return element + count.translate(SUBSCRIPT_TRANSLATION)

        raise ValueError(
            "mode must be one of: 'mathtext', 'html', 'unicode', 'plain'"
        )

    formatted = re.sub(formula_pattern, repl, label)
    for placeholder, replacement in charge_replacements.items():
        formatted = formatted.replace(placeholder, replacement)
    return formatted

__all__ = [
    "concentration_to_float",
    "format_chemical_formulas",
    "get_file_times",
    "parse_concentration_value_and_unit",
]
