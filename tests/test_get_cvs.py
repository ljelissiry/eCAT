from pathlib import Path

import pytest


def test_get_cvs_loads_semicolon_decimal_comma_cv_as_cv(ecat_module, tmp_path):
    from ecat import io as ecat_io

    data = "\n".join(
        [
            "Scan;Index;Time (s);WE(1).Potential (V);WE(1).Current (A);Current range",
            "1;1;0,0;-0,100;-1,0E-06;10 uA",
            "1;2;0,1;-0,050;-5,0E-07;10 uA",
            "1;3;0,2;-0,100;-1,2E-06;10 uA",
        ]
    )
    (tmp_path / "an_100mVs_CO2_DMF.txt").write_text(data, encoding="utf-8")

    objects = ecat_io.get_CVs(
        {
            "folder path": str(tmp_path),
            "recursive search": False,
            "reference mode": "none",
            "print": False,
        }
    )

    assert len(objects) == 1
    obj = objects[0]
    assert type(obj).__name__ == "cv"
    assert obj.name == "an_100mVs_CO2_DMF"
    assert obj.gas == "CO2"
    assert obj.solvent == "DMF"
    assert obj.scan_rate == pytest.approx(0.1)
    assert obj.data["Potential"].tolist() == pytest.approx([-0.1, -0.05, -0.1])
    assert obj.data["Current"].tolist() == pytest.approx([-1.0e-6, -5.0e-7, -1.2e-6])


def test_get_cvs_loads_ch_metadata_with_mixed_header_and_data_delimiters(
    ecat_module,
    tmp_path,
):
    from ecat import io as ecat_io

    data = "\n".join(
        [
            "June 21, 2023   16:01:40",
            "Cyclic Voltammetry",
            "Instrument Model:  CHI660D",
            "Scan Rate (V/s) = 1",
            "",
            "Potential/V, Current/A",
            "",
            "-0.700\t-6.114e-6",
            "-0.701\t-6.786e-6",
            "-0.702\t-7.061e-6",
        ]
    )
    (tmp_path / "1000mv_co2_100mmphoh.txt").write_text(data, encoding="utf-8")

    objects = ecat_io.get_CVs(
        {
            "folder path": str(tmp_path),
            "recursive search": False,
            "reference mode": "none",
            "print": False,
        }
    )

    assert len(objects) == 1
    obj = objects[0]
    assert type(obj).__name__ == "cv"
    assert obj.gas == "CO2"
    assert obj.compounds == ["phoh"]
    assert obj.concentrations == ["100 mM"]
    assert obj.scan_rate == pytest.approx(1.0)
    assert obj.units == {"Potential": "V", "Current": "A"}
    assert obj.data["Potential"].tolist() == pytest.approx([-0.700, -0.701, -0.702])


def test_get_cvs_parses_l_unit_compounds_from_filename(ecat_module, tmp_path):
    from ecat import io as ecat_io

    data = "\n".join(
        [
            "Potential (V)\tCurrent (A)",
            "0.00\t0.0",
            "0.10\t1.0e-6",
            "0.00\t0.0",
        ]
    )
    (tmp_path / "100mVs_CO2_1L_Fc.txt").write_text(data, encoding="utf-8")

    objects = ecat_io.get_CVs(
        {
            "folder path": str(tmp_path),
            "recursive search": False,
            "reference mode": "none",
            "print": False,
        }
    )

    assert len(objects) == 1
    assert objects[0].compounds == ["Fc"]
    assert objects[0].concentrations == ["1 L"]


def test_get_cvs_uses_default_scan_rate_option_when_filename_has_none(
    ecat_module,
    tmp_path,
):
    from ecat import io as ecat_io

    data = "\n".join(
        [
            "WE(1).Potential (V)\tWE(1).Current (A)",
            "0.00\t0.0",
            "0.10\t1.0e-6",
            "0.00\t0.0",
        ]
    )
    (tmp_path / "GOx_FcMeth.txt").write_text(data, encoding="utf-8")

    objects = ecat_io.get_CVs(
        {
            "folder path": str(tmp_path),
            "recursive search": False,
            "reference mode": "none",
            "scan rate": 0.05,
            "print": False,
        }
    )

    assert len(objects) == 1
    assert objects[0].scan_rate == pytest.approx(0.05)


def test_get_data_custom_parser_merge_fills_missing_name_metadata(
    ecat_module,
    tmp_path,
):
    from ecat import io as ecat_io

    data = "\n".join(
        [
            "WE(1).Potential (V)\tWE(1).Current (A)",
            "0.00\t0.0",
            "0.10\t1.0e-6",
            "0.00\t0.0",
        ]
    )
    (tmp_path / "GOx_FcMeth.txt").write_text(data, encoding="utf-8")

    def parser(name, path=None, options=None):
        assert name == "GOx_FcMeth"
        assert str(path).endswith("GOx_FcMeth.txt")
        assert options["custom parser mode"] == "merge"
        return {
            "solvent": "MeCN",
            "compounds": ["GOx", "FcMeth"],
            "concentrations": ["5 mM", "1 mM"],
            "scan rate": 0.2,
        }

    objects = ecat_io.get_data(
        {
            "folder path": str(tmp_path),
            "recursive search": False,
            "reference mode": "none",
            "custom parser": parser,
            "custom parser mode": "merge",
            "print": False,
        }
    )

    assert len(objects) == 1
    obj = objects[0]
    assert obj.solvent == "MeCN"
    assert obj.compounds == ["GOx", "FcMeth"]
    assert obj.concentrations == ["5 mM", "1 mM"]
    assert obj.scan_rate == pytest.approx(0.2)


def test_get_data_custom_parser_override_replaces_name_metadata_but_not_file_metadata(
    ecat_module,
    tmp_path,
):
    from ecat import io as ecat_io

    data = "\n".join(
        [
            "June 21, 2023   16:01:40",
            "Cyclic Voltammetry",
            "Instrument Model:  CHI660D",
            "Scan Rate (V/s) = 1",
            "",
            "Potential/V, Current/A",
            "",
            "-0.700\t-6.114e-6",
            "-0.701\t-6.786e-6",
            "-0.702\t-7.061e-6",
        ]
    )
    (tmp_path / "1000mv_co2_100mmphoh.txt").write_text(data, encoding="utf-8")

    def parser(name, path=None, options=None):
        return {
            "gas": "Ar",
            "compounds": ["Fc"],
            "concentrations": ["1 mM"],
            "scan rate": 0.2,
        }

    objects = ecat_io.get_data(
        {
            "folder path": str(tmp_path),
            "recursive search": False,
            "reference mode": "none",
            "custom parser": parser,
            "custom parser mode": "override",
            "print": False,
        }
    )

    assert len(objects) == 1
    obj = objects[0]
    assert obj.gas == "Ar"
    assert obj.compounds == ["Fc"]
    assert obj.concentrations == ["1 mM"]
    assert obj.scan_rate == pytest.approx(1.0)


def test_get_data_custom_parser_can_override_file_metadata_when_enabled(
    ecat_module,
    tmp_path,
):
    from ecat import io as ecat_io

    data = "\n".join(
        [
            "June 21, 2023   16:01:40",
            "Cyclic Voltammetry",
            "Instrument Model:  CHI660D",
            "Scan Rate (V/s) = 1",
            "",
            "Potential/V, Current/A",
            "",
            "-0.700\t-6.114e-6",
            "-0.701\t-6.786e-6",
            "-0.702\t-7.061e-6",
        ]
    )
    (tmp_path / "1000mv_co2_100mmphoh.txt").write_text(data, encoding="utf-8")

    def parser(name, path=None, options=None):
        return {
            "scan rate": 0.2,
        }

    objects = ecat_io.get_data(
        {
            "folder path": str(tmp_path),
            "recursive search": False,
            "reference mode": "none",
            "custom parser": parser,
            "custom parser mode": "override",
            "parser settings": {"prefer file metadata": False},
            "print": False,
        }
    )

    assert len(objects) == 1
    assert objects[0].scan_rate == pytest.approx(0.2)


def test_import_eCAT_alias_exposes_get_cvs(repo_root):
    import sys

    src_path = str(repo_root / "src")
    if src_path not in sys.path:
        sys.path.insert(0, src_path)

    import eCAT as e

    assert callable(e.get_CVs)
