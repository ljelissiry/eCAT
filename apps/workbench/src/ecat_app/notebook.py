"""Generate compact reproducible Jupyter notebooks from browser workflows."""

from __future__ import annotations

import json


def _source_lines(text: str) -> list[str]:
    if not text:
        return []
    return text.splitlines()


def _markdown_cell(text: str) -> dict[str, object]:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": _source_lines(text),
    }


def _code_cell(text: str) -> dict[str, object]:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [text],
    }


def _split_code_sections(code: str) -> list[tuple[str, str]]:
    sections = []
    title = "Workflow"
    buffer = []
    for line in (code or "").splitlines():
        stripped = line.strip()
        if stripped.startswith("# ") and buffer:
            sections.append((title, "\n".join(buffer).strip()))
            title = stripped[2:].strip() or "Workflow"
            buffer = [line]
            continue
        if stripped.startswith("# ") and not buffer:
            title = stripped[2:].strip() or "Workflow"
            buffer.append(line)
            continue
        buffer.append(line)
    if buffer or not sections:
        sections.append((title, "\n".join(buffer).strip()))
    return [(section_title, section_code) for section_title, section_code in sections if section_code]


def _analysis_markdown(results_store: dict | None) -> str:
    lines = ["## Analysis Results", ""]
    if not results_store:
        lines.append("No analyses were present in the browser results when this notebook was exported.")
        return "\n".join(lines)
    for entry in (results_store or {}).values():
        title = entry.get("title") or "Analysis"
        lines.extend([f"### {title}", ""])
        result = entry.get("result") or {}
        rows = result.get("results") or []
        if not rows:
            lines.append("No tabular result rows were available.")
        for row in rows:
            analysis = row.get("analysis") or "analysis"
            status = row.get("status") or ""
            message = row.get("message") or ""
            detail = f"- `{analysis}`: {status}".rstrip()
            if message:
                detail += f" - {message}"
            lines.append(detail)
        lines.append("")
    return "\n".join(lines).strip()


def generate_notebook(code: str, results_store: dict | None = None) -> str:
    """Return a Jupyter notebook JSON string for the visible browser workflow."""

    cells = [
        _markdown_cell("# eCAT Browser Workflow\n\nReproducible notebook exported from eCAT Workbench."),
    ]
    for title, section_code in _split_code_sections(code or ""):
        cells.append(_markdown_cell(f"## {title}"))
        cells.append(_code_cell(section_code))
    cells.append(_markdown_cell(_analysis_markdown(results_store)))
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "pygments_lexer": "ipython3",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    return json.dumps(notebook, indent=2)
