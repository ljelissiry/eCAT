"""Shared file-format boundaries for eCAT import workflows."""

from pathlib import Path


SUPPORTED_TEXT_SUFFIXES = frozenset({".dat", ".mpt", ".txt"})
UNSUPPORTED_BINARY_SUFFIXES = frozenset({".mpr"})

_BIOLOGIC_MPR_MAGIC = b"BIO-LOGIC MODULAR FILE"


class UnsupportedFileFormatError(ValueError):
    """Raised when eCAT's built-in importer receives a known unsupported format."""


def normalized_suffix(filepath):
    return Path(filepath).suffix.lower()


def is_supported_text_file(filepath):
    return normalized_suffix(filepath) in SUPPORTED_TEXT_SUFFIXES


def is_known_unsupported_binary_file(filepath):
    return normalized_suffix(filepath) in UNSUPPORTED_BINARY_SUFFIXES


def has_biologic_mpr_magic(filepath):
    try:
        with open(filepath, "rb") as handle:
            return handle.read(len(_BIOLOGIC_MPR_MAGIC)).startswith(_BIOLOGIC_MPR_MAGIC)
    except OSError:
        return False


def validate_default_text_input(filepath):
    """Reject BioLogic binary input before it reaches a delimited-text parser."""
    if not (
        is_known_unsupported_binary_file(filepath)
        or has_biologic_mpr_magic(filepath)
    ):
        return

    filename = Path(filepath).name
    raise UnsupportedFileFormatError(
        f"Unsupported BioLogic EC-Lab binary `.mpr` file `{filename}`. "
        "eCAT's built-in importer reads EC-Lab ASCII `.mpt` exports, not `.mpr` "
        "binaries. Export the experiment as `.mpt`, convert it externally, or "
        "provide a custom reader."
    )


__all__ = [
    "SUPPORTED_TEXT_SUFFIXES",
    "UNSUPPORTED_BINARY_SUFFIXES",
    "UnsupportedFileFormatError",
]
