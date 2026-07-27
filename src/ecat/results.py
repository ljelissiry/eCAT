"""Shared result containers for eCAT analysis workflows."""

from __future__ import annotations

from collections.abc import Mapping

import pandas as pd


class AnalysisResult(dict):
    """Dictionary-compatible base class for eCAT analysis outputs.

    The public contract is intentionally small: analysis workflows expose their
    primary table through ``.table`` and related metadata through attributes such
    as ``.summary``, ``.fits``, ``.fit_table``, ``.warnings``, and ``.units``.
    Dictionary-style access is reserved for named payloads such as
    ``result["data"]`` or established scalar keys on single-CV results; tabular
    access should go through ``result.table``.
    """

    def __init__(
        self,
        values=None,
        *,
        table=None,
        summary=None,
        fits=None,
        fit_table=None,
        fit_model_results=None,
        raw_table=None,
        transformed_table=None,
        diagnostics=None,
        warnings=None,
        units=None,
        figure=None,
        axes=None,
        figures=None,
        analysis=None,
    ):
        super().__init__({} if values is None else dict(values))
        self.table = table
        self.summary = {} if summary is None else summary
        if analysis is not None:
            self.summary.setdefault("analysis", analysis)
        self.fits = fits
        self.fit_table = fit_table
        self.fit_model_results = {} if fit_model_results is None else fit_model_results
        self.raw_table = raw_table if raw_table is not None else table
        self.transformed_table = transformed_table
        self.diagnostics = {} if diagnostics is None else diagnostics
        self.warnings = [] if warnings is None else warnings
        self.units = units if units is not None else self._units_from_table(table)
        self.figure = figure
        self.axes = axes
        self.figures = [] if figures is None else figures

    @staticmethod
    def _units_from_table(table):
        attrs = getattr(table, "attrs", None)
        if isinstance(attrs, dict):
            units = attrs.get("units")
            if units is not None:
                return units
        return {}

    @property
    def data(self):
        """Primary result table, matching older dict-style ``result["data"]``."""
        return self.table

    def show(self, options=None):
        """Print the main table and fit table; return the table when requested."""
        options = {} if options is None else dict(options)
        if self.summary:
            for key, value in self.summary.items():
                print(f"{key}: {value}")
        if self.table is not None:
            try:
                from .plotting import _display_table

                _display_table(self.table, options, title="Result Table")
            except Exception:
                print(self.table)
        if self.fit_table is not None:
            try:
                from .plotting import _display_table

                _display_table(self.fit_table, options, title="Fit Table")
            except Exception:
                print(self.fit_table)
        if options.get("return", False):
            return self.table
        return None

    def to_dataframe(self, kind="table"):
        """Return one of the tabular payloads by name."""
        key = str(kind).strip().lower().replace("-", " ").replace("_", " ")
        if key in {"table", "data", "primary"}:
            return self.table
        if key in {"fit table", "fits table"}:
            return self.fit_table
        if key == "raw":
            return self.raw_table
        if key in {"transformed", "transformed table"}:
            return self.transformed_table
        raise ValueError(
            "kind must be 'table', 'fit table', 'raw', or 'transformed'."
        )

    def to_csv(self, path_or_buf=None, *args, **kwargs):
        """Write the primary result table as CSV, matching pandas semantics."""
        table = self._require_primary_table()
        return table.to_csv(path_or_buf, *args, **kwargs)

    def to_excel(
        self,
        excel_writer,
        *,
        sheet_name="table",
        index=False,
        include_metadata=True,
        **kwargs,
    ):
        """Write the result table and metadata sheets to an Excel workbook."""
        table = self._require_primary_table()
        if isinstance(excel_writer, pd.ExcelWriter):
            self._write_excel_workbook(
                excel_writer,
                table=table,
                sheet_name=sheet_name,
                index=index,
                include_metadata=include_metadata,
                to_excel_kwargs=kwargs,
            )
            return None

        with pd.ExcelWriter(excel_writer) as writer:
            self._write_excel_workbook(
                writer,
                table=table,
                sheet_name=sheet_name,
                index=index,
                include_metadata=include_metadata,
                to_excel_kwargs=kwargs,
            )
        return None

    def _require_primary_table(self):
        if self.table is None:
            raise ValueError("AnalysisResult export requires a primary table.")
        return self.table

    def _write_excel_workbook(
        self,
        writer,
        *,
        table,
        sheet_name,
        index,
        include_metadata,
        to_excel_kwargs,
    ):
        used_sheet_names = set()
        primary_sheet = _excel_sheet_name(sheet_name, used_sheet_names)
        table.to_excel(
            writer,
            sheet_name=primary_sheet,
            index=index,
            **to_excel_kwargs,
        )
        if not include_metadata:
            return

        metadata_tables = [
            ("summary", _mapping_to_frame(self.summary)),
            ("fit_table", self.fit_table),
            ("fits", _mapping_to_frame(self.fits)),
            ("warnings", _sequence_to_frame(self.warnings, "Warning")),
            ("units", _mapping_to_frame(self.units)),
            ("diagnostics", _mapping_to_frame(self.diagnostics)),
        ]
        for name, metadata_table in metadata_tables:
            if _is_empty_table(metadata_table):
                continue
            metadata_table.to_excel(
                writer,
                sheet_name=_excel_sheet_name(name, used_sheet_names),
                index=False,
            )


def analysis_result_from_table(
    table,
    *,
    analysis,
    summary=None,
    values=None,
    axes=None,
    figure=None,
    diagnostics=None,
    warnings=None,
):
    """Build a table-like AnalysisResult from a former DataFrame return."""
    summary_data = {"analysis": analysis}
    if summary:
        summary_data.update(summary)
    values_data = {"data": table, "summary": summary_data}
    if values:
        values_data.update(values)
    return AnalysisResult(
        values_data,
        table=table,
        summary=summary_data,
        diagnostics=diagnostics,
        warnings=warnings,
        units=AnalysisResult._units_from_table(table),
        axes=axes,
        figure=figure,
        analysis=analysis,
    )


def _excel_sheet_name(name, used):
    invalid_chars = set('[]:*?/\\')
    cleaned = "".join("_" if char in invalid_chars else char for char in str(name))
    cleaned = cleaned.strip() or "sheet"
    base = cleaned[:31]
    candidate = base
    suffix = 1
    while candidate in used:
        suffix_text = f"_{suffix}"
        candidate = f"{base[: 31 - len(suffix_text)]}{suffix_text}"
        suffix += 1
    used.add(candidate)
    return candidate


def _mapping_to_frame(value):
    if not value:
        return None
    rows = []
    _append_mapping_rows(rows, "", value)
    return pd.DataFrame(rows, columns=["Field", "Value"])


def _append_mapping_rows(rows, prefix, value):
    if isinstance(value, Mapping):
        for key, item in value.items():
            field = f"{prefix}.{key}" if prefix else str(key)
            _append_mapping_rows(rows, field, item)
        return
    rows.append({"Field": prefix, "Value": _excel_cell_value(value)})


def _sequence_to_frame(value, column):
    if not value:
        return None
    return pd.DataFrame({column: [_excel_cell_value(item) for item in value]})


def _excel_cell_value(value):
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def _is_empty_table(table):
    return table is None or getattr(table, "empty", False)


__all__ = ["AnalysisResult", "analysis_result_from_table"]
