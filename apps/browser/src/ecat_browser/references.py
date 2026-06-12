"""Reference option helpers for browser-driven eCAT imports."""

from __future__ import annotations


def _blank_to_none(value):
    if value is None:
        return None
    if isinstance(value, str) and value.strip() == "":
        return None
    return value


def _coerce_float_or_auto(value):
    value = _blank_to_none(value)
    if value is None:
        return None
    if isinstance(value, str) and value.strip().lower() == "auto":
        return "auto"
    return float(value)


def _split_keywords(value) -> list[str] | None:
    value = _blank_to_none(value)
    if value is None:
        return None
    if isinstance(value, str):
        values = [part.strip() for part in value.split(",")]
    else:
        values = [str(part).strip() for part in value]
    return [part for part in values if part]


def _coerce_reference_map(rows) -> dict[int, int] | None:
    mapping: dict[int, int] = {}
    for row in rows or []:
        target = _blank_to_none(row.get("target"))
        reference = _blank_to_none(row.get("reference"))
        if target is None or reference is None:
            continue
        mapping[int(target)] = int(reference)
    return mapping or None


def build_reference_options(settings: dict | None) -> dict[str, object]:
    settings = settings or {}
    mode = str(settings.get("mode", "none")).strip().lower() or "none"
    options: dict[str, object] = {
        "reference mode": mode,
        "reference label": settings.get("label") or "Fc/Fc+",
    }

    guess = _coerce_float_or_auto(settings.get("guess", "auto"))
    if guess is not None:
        options["reference guess"] = guess

    if mode == "manual":
        offset = _coerce_float_or_auto(settings.get("offset"))
        if offset is not None and offset != "auto":
            options["reference offset"] = offset
    elif mode == "file":
        reference_file = _blank_to_none(settings.get("file"))
        if reference_file is not None:
            options["reference file"] = str(reference_file)
    elif mode == "keyword":
        keyword = _blank_to_none(settings.get("keyword"))
        if keyword is not None:
            options["reference keyword"] = str(keyword)
        options["allow self reference"] = bool(settings.get("allow_self_reference", True))
    elif mode == "auto":
        keywords = _split_keywords(settings.get("keywords"))
        if keywords is not None:
            options["reference keywords"] = keywords
        options["allow self reference"] = bool(settings.get("allow_self_reference", True))

    reference_map = _coerce_reference_map(settings.get("map"))
    if reference_map is not None:
        options["reference map"] = reference_map

    return options


def reference_field_visibility(mode: str | None) -> dict[str, bool]:
    mode = str(mode or "none").strip().lower()
    return {
        "manual": mode == "manual",
        "file": mode == "file",
        "keyword": mode == "keyword",
        "auto": mode == "auto",
        "guess": mode in {"file", "keyword", "auto"},
        "label": mode in {"manual", "file", "keyword", "auto"},
        "allow_self_reference": mode in {"keyword", "auto"},
    }
