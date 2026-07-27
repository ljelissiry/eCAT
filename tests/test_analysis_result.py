import pandas as pd
import pytest


def test_analysis_result_keeps_tabular_access_explicit(ecat_module):
    table = pd.DataFrame({"x": [1.0, 2.0], "kobs": [3.0, 6.0]})
    table.attrs["units"] = {"kobs": "s^-1"}

    result = ecat_module.AnalysisResult(
        {"data": table},
        table=table,
        summary={"analysis": "example"},
    )

    assert result["data"] is table
    assert result.table["kobs"].tolist() == [3.0, 6.0]
    assert result.table.loc[1, "kobs"] == pytest.approx(6.0)
    assert result.units["kobs"] == "s^-1"
    assert len(result) == 1
    assert result.to_dataframe() is table
    with pytest.raises(KeyError):
        result["kobs"]
    with pytest.raises(AttributeError):
        _ = result.loc


def test_analysis_result_to_csv_exports_primary_table(ecat_module, tmp_path):
    table = pd.DataFrame({"x": [1.0, 2.0], "kobs": [3.0, 6.0]})
    result = ecat_module.AnalysisResult({"data": table}, table=table)

    csv_text = result.to_csv(index=False)
    output_path = tmp_path / "fowa.csv"
    result.to_csv(output_path, index=False)

    assert csv_text.startswith("x,kobs")
    saved = pd.read_csv(output_path)
    assert saved.columns.tolist() == ["x", "kobs"]
    assert saved["kobs"].tolist() == [3.0, 6.0]


def test_analysis_result_to_excel_writes_table_and_metadata_sheets(
    ecat_module, tmp_path
):
    table = pd.DataFrame({"Scan Rate": [0.1, 0.2], "kobs": [1.5, 3.0]})
    fit_table = pd.DataFrame({"parameter": ["m"], "value": [2.0]})
    result = ecat_module.AnalysisResult(
        {"data": table},
        table=table,
        summary={"analysis": "fowa", "n": 2},
        fits={"parameters": {"m": 2.0}},
        fit_table=fit_table,
        warnings=["check the fit window"],
        units={"kobs": "s^-1"},
    )

    output_path = tmp_path / "fowa.xlsx"
    result.to_excel(output_path)

    workbook = pd.ExcelFile(output_path)
    assert workbook.sheet_names == [
        "table",
        "summary",
        "fit_table",
        "fits",
        "warnings",
        "units",
    ]
    saved_table = pd.read_excel(output_path, sheet_name="table")
    summary = pd.read_excel(output_path, sheet_name="summary")
    units = pd.read_excel(output_path, sheet_name="units")

    assert saved_table["kobs"].tolist() == [1.5, 3.0]
    assert summary.loc[summary["Field"] == "analysis", "Value"].iloc[0] == "fowa"
    assert units.loc[units["Field"] == "kobs", "Value"].iloc[0] == "s^-1"


def test_analysis_result_export_requires_primary_table(ecat_module, tmp_path):
    result = ecat_module.AnalysisResult({"summary": {"analysis": "empty"}})

    with pytest.raises(ValueError, match="primary table"):
        result.to_csv(tmp_path / "missing.csv")
    with pytest.raises(ValueError, match="primary table"):
        result.to_excel(tmp_path / "missing.xlsx")


def test_existing_result_classes_share_analysis_result_parent(ecat_module, cv_factory):
    cv_obj = cv_factory()
    peak = cv_obj.peak_potential({"plot": False, "print": False})
    fit = ecat_module.fit_model(
        [1.0, 2.0, 3.0],
        [2.0, 4.0, 6.0],
        options={"plot": False, "print": False},
    )
    chrono = ecat_module.ChronoAnalysisResult({"time": 1.0})

    assert isinstance(peak, ecat_module.AnalysisResult)
    assert isinstance(peak, ecat_module.CVAnalysisResult)
    assert isinstance(fit, ecat_module.AnalysisResult)
    assert isinstance(fit, ecat_module.ScatterFitResult)
    assert isinstance(chrono, ecat_module.AnalysisResult)


def test_fit_rate_accepts_analysis_result_table(ecat_module):
    table = pd.DataFrame({"Scan Rate": [1.0, 2.0, 3.0], "kobs": [2.0, 4.0, 6.0]})
    result = ecat_module.AnalysisResult(
        {"data": table},
        table=table,
        summary={"analysis": "fowa"},
    )

    fit = ecat_module.fit_rate(result, {"plot": False, "print": False})

    assert isinstance(fit, ecat_module.AnalysisResult)
    assert fit.fits["parameters"]["m"] == pytest.approx(2.0)


def test_plateau_current_returns_explicit_table_analysis_result(ecat_module):
    result = ecat_module.plateau_current(
        [],
        {
            "plot": False,
            "print": False,
            "ic": 5e-5,
            "ip0": 1e-5,
            "ip0 scan rate": 0.1,
            "formula mode": "normalized",
        },
    )

    assert isinstance(result, ecat_module.AnalysisResult)
    assert result["data"] is result.table
    assert list(result.table.columns) == ["Metric", "Value"]
    assert "kobs" in set(result.table["Metric"])
    assert result.summary["formula mode"] == "normalized"
    assert result.summary["kobs"] > 0
    assert result.diagnostics["plateau details"].loc[0, "formula mode"] == "normalized"
    with pytest.raises(KeyError):
        result["kobs"]


def test_tafel_analysis_returns_analysis_result_with_legacy_keys(ecat_module, cv_factory):
    cv_obj = cv_factory()

    result = ecat_module.tafel_analysis(
        cv_obj,
        10.0,
        -0.8,
        -1.2,
        {"plot": False, "print": False},
    )

    assert isinstance(result, ecat_module.AnalysisResult)
    assert result["data"] is result.table
    assert "TOFmax" in result.table.columns
    assert "summary" in result
    assert result["axes"] is result.axes
