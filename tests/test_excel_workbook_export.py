import numpy as np
import pandas as pd
import pytest


pytest.importorskip("openpyxl")


def _chrono_object(ecat_module, blank_echem_factory):
    obj = blank_echem_factory(ecat_module.ca)
    obj.name = "CO2_MeCN_CA_-1V"
    obj.type = "Chronoamperometry"
    obj.software = "manual"
    obj.data = pd.DataFrame(
        {
            "Time": [0.0, 1.0, 2.0],
            "Current": [-1.0e-6, -1.2e-6, -1.4e-6],
            "Potential": [-1.0, -1.0, -1.0],
        }
    )
    obj.units = {"Time": "s", "Current": "A", "Potential": "V"}
    obj.gas = "CO2"
    obj.solvent = "MeCN"
    obj.run_time = 2.0
    obj.sample_interval = 1.0
    obj.quiet_time = 0.0
    return obj


def test_save_data_xlsx_writes_manifest_and_class_sheets_with_shared_x(
    ecat_module,
    cv_factory,
    blank_echem_factory,
    tmp_path,
    capsys,
):
    cv1 = cv_factory(name="100mVs_MeCN_Ar_1mMFe")
    cv2 = cv_factory(name="100mVs_MeCN_CO2_1mMFe")
    cv3 = cv_factory(
        name="50mVs_MeCN_CO2_1mMFe",
        potential=[-0.2, -0.1, 0.0, 0.1],
        current=[0.0, 1e-6, 0.5e-6, 0.0],
    )
    cv1.reference_source_file = "/tmp/reference_cv.txt"
    cv1.reference_mode = "manual"
    cv1.reference_shift = 0.4
    ca = _chrono_object(ecat_module, blank_echem_factory)

    exported = ecat_module.save_data(
        [cv1, cv2, cv3, ca],
        {
            "format": "xlsx",
            "folder path": str(tmp_path),
            "file name": "ecat_export",
            "metadata columns": ["reference source"],
            "share x axes": True,
        },
    )

    path = tmp_path / "ecat_export.xlsx"
    captured = capsys.readouterr()
    workbook = pd.ExcelFile(path)
    manifest = pd.read_excel(path, sheet_name="manifest")
    cv_sheet = pd.read_excel(path, sheet_name="cv", header=None)

    assert path.exists()
    assert workbook.sheet_names[:3] == ["manifest", "cv", "ca"]
    assert "Saved 4 echem objects" in captured.out
    assert isinstance(exported, dict)
    assert set(exported) == {"manifest", "cv", "ca"}

    assert list(manifest["class"]) == ["cv", "cv", "cv", "ca"]
    assert list(manifest["sheet"]) == ["cv", "cv", "cv", "ca"]
    assert "Reference Source" in manifest.columns
    assert manifest.loc[0, "Reference Source"] != ""

    group_row = list(cv_sheet.iloc[0].fillna(""))
    column_row = list(cv_sheet.iloc[1].fillna(""))
    assert group_row.count(manifest.loc[0, "x group"]) == 1
    assert group_row.count(manifest.loc[2, "object_id"]) >= 2
    assert column_row.count("Potential") == 2
    assert column_row.count("Current") == 3


def test_get_cvs_from_excel_reads_manifest_workbook_for_multiple_classes(
    ecat_module,
    cv_factory,
    blank_echem_factory,
    tmp_path,
):
    cv_obj = cv_factory(name="100mVs_MeCN_CO2_1mMFe")
    cv_obj.reference_label = "Fc/Fc+"
    cv_obj.reference_mode = "manual"
    cv_obj.reference_shift = 0.5
    ca_obj = _chrono_object(ecat_module, blank_echem_factory)

    ecat_module.save_data(
        [cv_obj, ca_obj],
        {
            "format": "xlsx",
            "folder path": str(tmp_path),
            "file name": "roundtrip",
            "metadata columns": ["reference label", "reference shift", "reference mode"],
            "share x axes": True,
        },
    )

    imported = ecat_module.get_data_from_excel(tmp_path / "roundtrip.xlsx", {"print": False})

    assert [type(obj).__name__ for obj in imported] == ["cv", "ca"]
    assert [obj.name for obj in imported] == [cv_obj.name, ca_obj.name]

    imported_cv, imported_ca = imported
    np.testing.assert_allclose(imported_cv.data["Potential"], cv_obj.data["Potential"])
    np.testing.assert_allclose(imported_cv.data["Current"], cv_obj.data["Current"])
    assert imported_cv.units == cv_obj.units
    assert imported_cv.reference_label == "Fc/Fc+"
    assert imported_cv.reference_mode == "manual"
    assert imported_cv.reference_shift == 0.5

    np.testing.assert_allclose(imported_ca.data["Time"], ca_obj.data["Time"])
    np.testing.assert_allclose(imported_ca.data["Current"], ca_obj.data["Current"])
    assert imported_ca.units["Time"] == "s"
    assert imported_ca.type == "Chronoamperometry"


def test_get_data_from_excel_public_name_reads_manifest_workbook(
    ecat_module,
    cv_factory,
    tmp_path,
):
    cv_obj = cv_factory(name="100mVs_MeCN_CO2_1mMFe")
    ecat_module.save_data(
        [cv_obj],
        {
            "format": "xlsx",
            "folder path": str(tmp_path),
            "file name": "data_import",
        },
    )

    imported = ecat_module.get_data_from_excel(tmp_path / "data_import.xlsx", {"print": False})

    assert [type(obj).__name__ for obj in imported] == ["cv"]
    assert imported[0].name == cv_obj.name


def test_xlsx_data_columns_default_includes_raw_and_referenced_potential(
    ecat_module,
    cv_factory,
    tmp_path,
):
    cv_obj = cv_factory(name="100mVs_MeCN_CO2_1mMFe")
    cv_obj.reference_label = "Fc/Fc+"
    cv_obj.reference_shift = 0.5
    cv_obj.reference_mode = "manual"

    ecat_module.save_data(
        [cv_obj],
        {
            "format": "xlsx",
            "folder path": str(tmp_path),
            "file name": "reference_axes",
        },
    )

    cv_sheet = pd.read_excel(tmp_path / "reference_axes.xlsx", sheet_name="cv", header=None)
    column_row = list(cv_sheet.iloc[1].fillna(""))
    unit_row = list(cv_sheet.iloc[2].fillna(""))

    assert column_row[:3] == ["Potential", "Potential vs Fc/Fc+", "Current"]
    assert unit_row[:3] == ["V", "V", "A"]


def test_xlsx_data_columns_exact_list_is_strict(
    ecat_module,
    cv_factory,
    tmp_path,
):
    cv_obj = cv_factory(name="100mVs_MeCN_CO2_1mMFe")
    cv_obj.reference_label = "Fc/Fc+"
    cv_obj.reference_shift = 0.5
    cv_obj.reference_mode = "manual"

    ecat_module.save_data(
        [cv_obj],
        {
            "format": "xlsx",
            "folder path": str(tmp_path),
            "file name": "strict_columns",
            "data columns": ["Potential vs Fc/Fc+", "Current"],
        },
    )

    cv_sheet = pd.read_excel(tmp_path / "strict_columns.xlsx", sheet_name="cv", header=None)
    column_row = [value for value in cv_sheet.iloc[1].fillna("").tolist() if value != ""]
    assert column_row == ["Potential vs Fc/Fc+", "Current"]


def test_xlsx_data_columns_unknown_column_raises_clear_error(
    ecat_module,
    cv_factory,
    tmp_path,
):
    cv_obj = cv_factory(name="100mVs_MeCN_CO2_1mMFe")

    with pytest.raises(ValueError, match="Unknown data column"):
        ecat_module.save_data(
            [cv_obj],
            {
                "format": "xlsx",
                "folder path": str(tmp_path),
                "file name": "bad_columns",
                "data columns": ["Not A Column"],
            },
        )


def test_xlsx_metadata_columns_can_request_blank_reference_columns(
    ecat_module,
    cv_factory,
    tmp_path,
):
    cv_obj = cv_factory(name="100mVs_MeCN_CO2_1mMFe")

    ecat_module.save_data(
        [cv_obj],
        {
            "format": "xlsx",
            "folder path": str(tmp_path),
            "file name": "blank_reference_metadata",
            "metadata columns": ["reference source", "reference shift", "reference label"],
        },
    )

    manifest = pd.read_excel(tmp_path / "blank_reference_metadata.xlsx", sheet_name="manifest")
    assert ["Reference Source", "Reference Shift", "Reference Label"] == [
        col for col in ["Reference Source", "Reference Shift", "Reference Label"]
        if col in manifest.columns
    ]
    assert manifest.loc[0, "Reference Source"] in ("", np.nan) or pd.isna(manifest.loc[0, "Reference Source"])
