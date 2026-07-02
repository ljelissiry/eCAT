from pathlib import Path
from datetime import datetime

import pytest


def test_ch_parser_handles_tab_delimiter_uA_units_and_raw_timestamp(ecat_module, fixtures_dir):
    obj = ecat_module.echem.from_file(
        str(fixtures_dir / "ch_cv_tab_uA_bad_timestamp.txt"),
        {},
    )

    assert type(obj).__name__ == "cv"
    assert obj.timestamp == "not-a-real-timestamp"
    assert list(obj.data.columns) == ["Potential", "Current"]
    assert obj.units == {"Potential": "V", "Current": "uA"}
    assert obj.scan_rate == pytest.approx(0.05)
    assert obj.segments == 2
    assert obj.delta_x == pytest.approx(0.25)


def test_ch_parser_accepts_month_timestamp_without_period(ecat_module, tmp_path):
    path = tmp_path / "may_timestamp.txt"
    path.write_text(
        "\n".join(
            [
                "May 7, 2026   15:59:59",
                "Cyclic Voltammetry",
                "Instrument Model: CHI760E",
                "Init E = -0.30",
                "High E = 0.30",
                "Low E = -0.30",
                "Scan Rate = 0.05",
                "Segment = 2",
                "Sample Interval = 0.05",
                "Sensitivity = 1e-6",
                "Potential/V,Current/A",
                "-0.30,-1e-7",
                "0.00,0",
                "0.30,1e-7",
            ]
        )
        + "\n",
        encoding="ISO-8859-1",
    )

    obj = ecat_module.echem.from_file(str(path), {})

    assert obj.timestamp == datetime(2026, 5, 7, 15, 59, 59)


def test_basi_parser_handles_tab_delimiter_and_missing_switching_potential_2(
    ecat_module,
    fixtures_dir,
):
    obj = ecat_module.echem.from_file(str(fixtures_dir / "basi_cv_tab_uA.txt"), {})

    assert type(obj).__name__ == "cv"
    assert obj.type == "Cyclic Voltammetry"
    assert obj.units == {"Potential": "V", "Current": "uA"}
    assert obj.init_E == pytest.approx(-0.25)
    assert obj.min_E == pytest.approx(-0.25)
    assert obj.max_E == pytest.approx(0.25)
    assert obj.final_E == pytest.approx(-0.25)
    assert obj.scan_rate == pytest.approx(0.05)
    assert obj.segments == 2


def test_eclab_parser_converts_mA_to_A_and_reads_cv_metadata(ecat_module, fixtures_dir):
    obj = ecat_module.echem.from_file(str(fixtures_dir / "eclab_cv_mA_metadata.txt"), {})

    assert type(obj).__name__ == "cv"
    assert obj.units == {"Potential": "V", "Current": "A"}
    assert obj.scan_rate == pytest.approx(0.05)
    assert obj.init_E == pytest.approx(-0.25)
    assert obj.max_E == pytest.approx(0.25)
    assert obj.min_E == pytest.approx(-0.25)
    assert obj.final_E == pytest.approx(-0.25)
    assert obj.data["Current"].tolist() == pytest.approx([-1.0e-4, 2.0e-4, 3.0e-4])


def test_cv_parsers_expose_same_core_data_and_metadata(ecat_module, tmp_path):
    files = {
        "CH": tmp_path / "ch_cv.txt",
        "BASI": tmp_path / "basi_cv.txt",
        "EC-Lab": tmp_path / "eclab_cv.txt",
    }
    files["CH"].write_text(
        "\n".join(
            [
                "Aug. 27, 2023   16:05:21",
                "Cyclic Voltammetry",
                "Instrument Model: CHI760E",
                "Init E = -0.25",
                "High E = 0.25",
                "Low E = -0.25",
                "Scan Rate = 0.05",
                "Segment = 2",
                "Quiet Time (sec) = 2",
                "Sample Interval = 0.05",
                "Sensitivity = 1e-6",
                "Potential/V,Current/A",
                "-0.25,-1.0e-7",
                "-0.10,2.0e-7",
                "0.10,6.0e-7",
                "0.25,3.0e-7",
            ]
        )
        + "\n",
        encoding="ISO-8859-1",
    )
    files["BASI"].write_text(
        "\n".join(
            [
                "BASI",
                "Experiment Type: Cyclic Voltammetry",
                "Initial Potential : -250 mV",
                "Switching Potential 1 : 250 mV",
                "Switching Potential 2 : -250 mV",
                "Final Potential : -250 mV",
                "Scan Rate : 50 mV/s",
                "Number of segments : 2",
                "Sample Interval : 150 mV",
                "Quiet Time : 3 s",
                "IR-Comp. Value : 0 Ohm",
                "[Begin Data]",
                "Potential/V,Current/A",
                "-0.25,-1.0e-7",
                "-0.10,2.0e-7",
                "0.10,6.0e-7",
                "0.25,3.0e-7",
            ]
        )
        + "\n",
        encoding="ISO-8859-1",
    )
    files["EC-Lab"].write_text(
        "\n".join(
            [
                "EC-Lab ASCII FILE",
                "Metadata",
                "Metadata",
                "Cyclic Voltammetry",
                "Nb header lines : 14",
                "Ei (V) -0.25",
                "E1 (V) 0.25",
                "E2 (V) -0.25",
                "Ef (V) 0.25",
                "dE/dt 50",
                "mV/s",
                "tR (h:m:s) 0:00:4.0000",
                "N 2",
                "Ewe/V\t<I>/A",
                "-0.25\t-1.0e-7",
                "-0.10\t2.0e-7",
                "0.10\t6.0e-7",
                "0.25\t3.0e-7",
            ]
        )
        + "\n",
        encoding="ISO-8859-1",
    )

    parsed = {
        software: ecat_module.echem.from_file(str(path), {})
        for software, path in files.items()
    }

    expected_potential = [-0.25, -0.10, 0.10, 0.25]
    expected_current = [-1.0e-7, 2.0e-7, 6.0e-7, 3.0e-7]

    for software, obj in parsed.items():
        assert type(obj).__name__ == "cv", software
        assert obj.software == software
        assert obj.type == "Cyclic Voltammetry"
        assert list(obj.data.columns) == ["Potential", "Current"]
        assert obj.units == {"Potential": "V", "Current": "A"}
        assert obj.data["Potential"].tolist() == pytest.approx(expected_potential)
        assert obj.data["Current"].tolist() == pytest.approx(expected_current)
        assert obj.x().tolist() == pytest.approx(expected_potential)
        assert obj.y().tolist() == pytest.approx(expected_current)
        assert obj.init_E == pytest.approx(-0.25)
        assert obj.min_E == pytest.approx(-0.25)
        assert obj.max_E == pytest.approx(0.25)
        assert obj.scan_rate == pytest.approx(0.05)
        assert obj.segments == 2
        assert obj.delta_x == pytest.approx(0.15)

    assert parsed["CH"].sample_int == pytest.approx(0.05)
    assert parsed["CH"].quiet_time == pytest.approx(2.0)
    assert parsed["BASI"].sample_int == pytest.approx(0.15)
    assert parsed["BASI"].quiet_time == pytest.approx(3.0)
    assert parsed["EC-Lab"].quiet_time == pytest.approx(4.0)


def test_cp_parsers_expose_same_core_data_and_metadata(ecat_module, tmp_path):
    files = {
        "CH": tmp_path / "ch_cp.txt",
        "BASI": tmp_path / "basi_cp.txt",
        "EC-Lab": tmp_path / "eclab_gcpl.txt",
    }
    files["CH"].write_text(
        "\n".join(
            [
                "Jan. 27, 2022   13:43:22",
                "Chronopotentiometry",
                "Instrument Model:  CHI760E",
                "Cathodic Current (A) = -0.025",
                "Anodic Current (A) = 0.025",
                "Init P/N = N",
                "Data Storage Interval (s) = 2.0",
                "High E Limit (V) = 1.08",
                "Low E Limit (V) = 0.38",
                "Cathodic Time (s) = 10",
                "Anodic Time (s) = 10",
                "Quiet Time (sec) = 2",
                "Segment = 2",
                "Time/sec, Potential/V",
                "0.0000, 0.700",
                "2.0000, 0.750",
                "4.0000, 0.690",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    files["BASI"].write_text(
        "\n".join(
            [
                "BASI",
                "Experiment Type: Chronopotentiometry",
                "Quiet Time : 3 s",
                "[Begin Data]",
                "Time/s,Potential/V",
                "0.0000,0.700",
                "2.0000,0.750",
                "4.0000,0.690",
            ]
        )
        + "\n",
        encoding="ISO-8859-1",
    )
    files["EC-Lab"].write_text(
        "\n".join(
            [
                "EC-Lab ASCII FILE",
                "Nb header lines : 20",
                "",
                "Galvanostatic Cycling with Potential Limitation",
                "",
                "Run on channel : 1",
                "Acquisition started on : 04/17/2025 12:47:42.264",
                "Technique started on : 04/17/2025 12:47:42.312",
                "Cycle Definition : Charge/Discharge alternance",
                "Ns                  0                   1                   2",
                "Set I/C             I                   I                   I",
                "Is                  0.000               25.000              -25.000",
                "unit Is             mA                  mA                  mA",
                "t1 (h:m:s)          0:00:0.0000         10:00:0.0000        10:00:0.0000",
                "dt1 (s)             0.0000              2.0000              2.0000",
                "tR (h:m:s)          0:00:4.0000",
                "EM (V)              0.000               1.080               0.380",
                "nc cycles           0                   0                   1",
                "",
                "mode\tNs\ttime/s\tEwe/V\t<I>/mA\tcycle number",
                "1\t1\t0.0000\t0.700\t25.0\t1",
                "1\t1\t2.0000\t0.750\t25.0\t1",
                "1\t2\t4.0000\t0.690\t-25.0\t1",
            ]
        )
        + "\n",
        encoding="ISO-8859-1",
    )

    parsed = {
        software: ecat_module.echem.from_file(str(path), {})
        for software, path in files.items()
    }

    expected_time = [0.0, 2.0, 4.0]
    expected_potential = [0.700, 0.750, 0.690]

    for software, obj in parsed.items():
        assert type(obj).__name__ == "cp", software
        assert obj.type == "Chronopotentiometry"
        assert list(obj.data.columns) == ["Time", "Potential"]
        assert obj.units == {"Time": "s", "Potential": "V"}
        assert obj.data["Time"].tolist() == pytest.approx(expected_time)
        assert obj.data["Potential"].tolist() == pytest.approx(expected_potential)
        assert obj.x().tolist() == pytest.approx(expected_time)
        assert obj.y().tolist() == pytest.approx(expected_potential)
        assert obj.init_E == pytest.approx(0.700)
        assert obj.final_E == pytest.approx(0.690)
        assert obj.min_E == pytest.approx(0.690)
        assert obj.max_E == pytest.approx(0.750)
        assert obj.delta_x == pytest.approx(2.0)

    assert parsed["CH"].software == "CH"
    assert parsed["CH"].sample_int == pytest.approx(2.0)
    assert parsed["CH"].anodic_current == pytest.approx(0.025)
    assert parsed["CH"].cathodic_current == pytest.approx(-0.025)
    assert parsed["CH"].segments == 2
    assert parsed["CH"].quiet_time == pytest.approx(2.0)

    assert parsed["BASI"].software == "BASI"
    assert parsed["BASI"].sample_int == pytest.approx(2.0)
    assert parsed["BASI"].quiet_time == pytest.approx(3.0)

    assert parsed["EC-Lab"].software == "EC-Lab"
    assert parsed["EC-Lab"].sample_int == pytest.approx(2.0)
    assert parsed["EC-Lab"].anodic_current == pytest.approx(0.025)
    assert parsed["EC-Lab"].cathodic_current == pytest.approx(-0.025)
    assert parsed["EC-Lab"].segments == 2
    assert parsed["EC-Lab"].quiet_time == pytest.approx(4.0)


def test_beta_exp_type_parser_matrix_exposes_expected_public_axes(ecat_module, fixtures_dir):
    cases = [
        ("ch_cv.txt", "cv", "Cyclic Voltammetry", "Potential", "Current"),
        ("basi_cv.txt", "cv", "Cyclic Voltammetry", "Potential", "Current"),
        ("eclab_cv.txt", "cv", "Cyclic Voltammetry", "Potential", "Current"),
        ("ch_ca_tiny.txt", "ca", "Chronoamperometry", "Time", "Current"),
        ("ch_cp_tiny.txt", "cp", "Chronopotentiometry", "Time", "Potential"),
        ("eclab_gcpl_tiny.txt", "cp", "Chronopotentiometry", "Time", "Potential"),
    ]

    for filename, expected_class, expected_type, expected_x, expected_y in cases:
        obj = ecat_module.echem.from_file(str(fixtures_dir / filename), {})

        assert type(obj).__name__ == expected_class, filename
        assert obj.type == expected_type
        assert obj.x().name == expected_x
        assert obj.y().name == expected_y
        assert list(obj.data.columns[:2]) == [expected_x, expected_y]
        assert expected_x in obj.units
        assert expected_y in obj.units


def test_loaded_objects_expose_standard_parse_result_contract(ecat_module, fixtures_dir):
    cases = [
        ("ch_cv.txt", "CV", "CH", ["Potential", "Current"]),
        ("basi_cv.txt", "CV", "BASI", ["Potential", "Current"]),
        ("eclab_cv.txt", "CV", "EC-Lab", ["Potential", "Current"]),
        ("ch_ca_tiny.txt", "CA", "CH", ["Time", "Current"]),
        ("ch_cp_tiny.txt", "CP", "CH", ["Time", "Potential"]),
        ("eclab_gcpl_tiny.txt", "CP", "EC-Lab", ["Time", "Potential"]),
    ]

    for filename, technique, software, columns in cases:
        obj = ecat_module.echem.from_file(str(fixtures_dir / filename), {})
        parsed = obj.parse_result

        assert isinstance(parsed, ecat_module.ParseResult), filename
        assert parsed.technique == technique
        assert parsed.software == software
        assert parsed.parser == software
        assert list(parsed.data.columns[:2]) == columns
        assert parsed.units == obj.units
        assert parsed.metadata["name"] == obj.name
        assert parsed.metadata["type"] == obj.type
        assert parsed.source == obj.filepath


def test_parse_file_returns_standard_parse_result_without_exposing_object(ecat_module, fixtures_dir):
    parsed = ecat_module.parse_file(str(fixtures_dir / "ch_cv.txt"), {})

    assert isinstance(parsed, ecat_module.ParseResult)
    assert parsed.technique == "CV"
    assert parsed.software == "CH"
    assert list(parsed.data.columns[:2]) == ["Potential", "Current"]
    assert parsed.metadata["scan_rate"] == pytest.approx(0.05)


def test_ch_dpv_parser_exposes_expected_public_axis_contract(ecat_module, repo_root):
    obj = ecat_module.echem.from_file(
        str(repo_root / "tests" / "tmp_real_examples" / "DPV_MeCN_CO2_0.1MTBAPF6_3mMFc_1mMFe-tpyPY2Me_-0.7_to_-1.2V.txt"),
        {},
    )

    assert type(obj).__name__ == "dpv"
    assert obj.software == "CH"
    assert obj.type == "Differential Pulse Voltammetry"
    assert obj.x().name == "Potential"
    assert obj.y().name == "Current"
    assert list(obj.data.columns) == ["Potential", "Current"]
    assert obj.units == {"Potential": "V", "Current": "A"}
    assert obj.init_E == pytest.approx(-0.7)
    assert obj.final_E == pytest.approx(-1.2)
    assert obj.delta_x is not None

def test_basi_cp_parser_handles_header_and_begin_data_export(ecat_module, tmp_path):
    path = tmp_path / "basi_cp.txt"
    path.write_text(
        "\n".join(
            [
                "BASI",
                "Experiment Type: Chronopotentiometry",
                "[Begin Data]",
                "Time/s,Potential/V",
                "0.0,0.700",
                "2.0,0.750",
                "4.0,0.690",
            ]
        )
        + "\n",
        encoding="ISO-8859-1",
    )

    obj = ecat_module.echem.from_file(str(path), {})

    assert type(obj).__name__ == "cp"
    assert obj.software == "BASI"
    assert obj.type == "Chronopotentiometry"
    assert list(obj.data.columns) == ["Time", "Potential"]
    assert obj.units == {"Time": "s", "Potential": "V"}
    assert obj.data["Time"].tolist() == pytest.approx([0.0, 2.0, 4.0])
    assert obj.data["Potential"].tolist() == pytest.approx([0.700, 0.750, 0.690])
    assert obj.init_E == pytest.approx(0.700)
    assert obj.final_E == pytest.approx(0.690)
    assert obj.min_E == pytest.approx(0.690)
    assert obj.max_E == pytest.approx(0.750)
    assert obj.sample_int == pytest.approx(2.0)
    assert obj.delta_x == pytest.approx(2.0)


def test_generic_fallback_preserves_header_row_and_units_when_present(ecat_module, fixtures_dir):
    obj = ecat_module.echem.from_file(str(fixtures_dir / "generic_header_units.txt"), {})

    assert type(obj).__name__ == "echem"
    assert obj.type == "Unknown"
    assert list(obj.data.columns) == ["Potential", "Current"]
    assert obj.units == {"Potential": "V", "Current": "mA"}
    assert obj.data["Potential"].tolist() == pytest.approx([-0.25, 0.0, 0.25])


def test_generic_fallback_exposes_filename_gas_and_solvent_metadata(ecat_module, tmp_path):
    filepath = tmp_path / "generic_CO2_MeCN_trace.txt"
    filepath.write_text(
        "Potential/V,Current/A\n"
        "0.00,0.0\n"
        "0.10,1.0e-6\n",
        encoding="ISO-8859-1",
    )

    obj = ecat_module.echem.from_file(str(filepath), {})

    assert type(obj).__name__ == "echem"
    assert obj.gas == "CO2"
    assert obj.solvent == "MeCN"
    assert obj.stats()["gas"] == "CO2"
    assert obj.txt_stats()["solvent"] == "MeCN"

    sorted_objects = ecat_module.sort([obj], ["gas"])
    assert sorted_objects == [obj]


def test_generic_fallback_keeps_numeric_columns_when_no_header_is_present(ecat_module, fixtures_dir):
    obj = ecat_module.echem.from_file(str(fixtures_dir / "generic_numeric_only.txt"), {})

    assert type(obj).__name__ == "echem"
    assert obj.type == "Unknown"
    assert list(obj.data.columns) == ["0", "1"]
    assert obj.units == {}
    assert obj.data["0"].tolist() == pytest.approx([-0.25, 0.0, 0.25])


def test_detect_experiment_type_honors_explicit_override(ecat_module, fixtures_dir):
    filepath = fixtures_dir / "generic_unknown.txt"

    exp_type = ecat_module.echem.detect_experiment_type(
        str(filepath),
        {"experiment type": "Chronopotentiometry"},
    )

    assert exp_type == "Chronopotentiometry"


def test_from_file_honors_explicit_software_override_for_loading(ecat_module, fixtures_dir):
    obj = ecat_module.echem.from_file(
        str(fixtures_dir / "generic_header_units.txt"),
        {"software": None},
    )

    assert type(obj).__name__ == "echem"
    assert obj.type == "Unknown"


def test_get_data_respects_recursive_search_flag_on_small_folder_tree(
    ecat_module,
    fixtures_dir,
    tmp_path,
):
    root_file = tmp_path / "root_ch_cv.txt"
    nested_dir = tmp_path / "nested"
    nested_dir.mkdir()
    nested_file = nested_dir / "nested_generic.txt"

    root_file.write_text(
        (fixtures_dir / "ch_cv.txt").read_text(encoding="ISO-8859-1"),
        encoding="ISO-8859-1",
    )
    nested_file.write_text(
        (fixtures_dir / "generic_header_units.txt").read_text(encoding="ISO-8859-1"),
        encoding="ISO-8859-1",
    )

    non_recursive = ecat_module.get_data(
        {
            "folder path": str(tmp_path),
            "recursive search": False,
            "print": False,
            "shift potential": False,
        }
    )
    recursive = ecat_module.get_data(
        {
            "folder path": str(tmp_path),
            "recursive search": True,
            "print": False,
            "shift potential": False,
        }
    )

    assert [Path(obj.filepath).name for obj in non_recursive] == ["root_ch_cv.txt"]
    recursive_names = [Path(obj.filepath).name for obj in recursive]
    assert set(recursive_names) == {"root_ch_cv.txt", "nested_generic.txt"}

    folderpaths_by_name = {
        Path(obj.filepath).name: obj.folderpath
        for obj in recursive
    }
    assert folderpaths_by_name == {
        "root_ch_cv.txt": "",
        "nested_generic.txt": "nested",
    }


def test_from_file_promotes_ch_ca_exports_to_ca(ecat_module, fixtures_dir):
    obj = ecat_module.echem.from_file(str(fixtures_dir / "ch_ca_tiny.txt"), {})

    assert type(obj).__name__ == "ca"
    assert obj.type == "Chronoamperometry"
    assert list(obj.data.columns) == ["Time", "Current"]
    assert obj.units == {"Time": "s", "Current": "A"}
    assert obj.timestamp.year == 2025


def test_ch_ca_parser_reads_small_realistic_header_fields(ecat_module, fixtures_dir):
    obj = ecat_module.echem.from_file(str(fixtures_dir / "ch_ca_tiny.txt"), {})

    assert obj.init_E == pytest.approx(-1.4)
    assert obj.sample_interval == pytest.approx(0.1)
    assert obj.run_time == pytest.approx(3.0)
    assert obj.quiet_time == pytest.approx(2.0)
    assert obj.sensitivity == pytest.approx(0.01)
    assert obj.ir_comp_resistance == pytest.approx(52.8)
    assert obj.ir_uncomp_resistance == pytest.approx(13.2)
    assert obj.ir_comp_percent == pytest.approx(80.0)
    assert obj.delta_x == pytest.approx(0.1)
    assert obj.data["Time"].tolist() == pytest.approx([0.1, 0.2, 0.3])
    assert obj.data["Current"].tolist() == pytest.approx([-3.675e-3, -3.676e-3, -3.632e-3])

    stats = obj.txt_stats()
    assert stats["ir comp resistance"] == pytest.approx(52.8)
    assert stats["ir uncomp resistance"] == pytest.approx(13.2)
    assert stats["ir comp percent"] == pytest.approx(80.0)

    table, _meta = ecat_module.build_object_table(
        [obj],
        {"columns": ["ir comp resistance", "ir uncomp resistance", "ir comp percent"]},
    )
    assert table["IR Comp Resistance"].tolist() == ["52.8"]
    assert table["IR Uncomp Resistance"].tolist() == ["13.2"]
    assert table["IR Comp Percent"].tolist() == ["80"]


def test_real_ch_exports_parse_ir_compensation_for_cv_dpv_and_cpe(ecat_module):
    examples_dir = Path(__file__).resolve().parents[1] / "tests" / "tmp_real_examples"

    cv_obj = ecat_module.echem.from_file(
        str(examples_dir / "CV_post_MeCN_Ar_0.1MTBAPF6_1mMDiipThzOAc_50mMPhAc_100mMPhOH_0_to_-1.6V_100mVs.txt"),
        {},
    )
    dpv_obj = ecat_module.echem.from_file(
        str(examples_dir / "DPV_MeCN_CO2_0.1MTBAPF6_3mMFc_1mMFe-tpyPY2Me_-0.7_to_-1.2V.txt"),
        {},
    )
    cpe_obj = ecat_module.echem.from_file(
        str(examples_dir / "CPE_MeCN_Ar_0.1MTBAPF6_1mMDiipThzOAc_50mMPhAc_100mMPhOH_-1.4V.txt"),
        {},
    )

    assert cv_obj.ir_comp_resistance == pytest.approx(49.0)
    assert cv_obj.ir_uncomp_resistance == pytest.approx(12.3)
    assert cv_obj.ir_comp_percent == pytest.approx(100 * 49.0 / (49.0 + 12.3))

    assert dpv_obj.ir_comp_resistance == pytest.approx(96.0)
    assert dpv_obj.ir_uncomp_resistance == pytest.approx(0.0)
    assert dpv_obj.ir_comp_percent == pytest.approx(100.0)

    assert cpe_obj.ir_comp_resistance == pytest.approx(52.8)
    assert cpe_obj.ir_uncomp_resistance == pytest.approx(0.0)
    assert cpe_obj.ir_comp_percent == pytest.approx(100.0)

    assert "ir comp resistance" not in cv_obj.txt_stats()
    assert "ir comp percent" not in cv_obj.txt_stats()

    table, _meta = ecat_module.build_object_table(
        [cv_obj],
        {"columns": ["ir comp resistance", "ir uncomp resistance", "ir comp percent"]},
    )
    assert table["IR Comp Resistance"].tolist() == ["49"]
    assert table["IR Uncomp Resistance"].tolist() == ["12.3"]
    assert table["IR Comp Percent"].tolist() == ["79.9"]

    missing_uc_table, _meta = ecat_module.build_object_table(
        [dpv_obj, cpe_obj],
        {"columns": ["ir comp resistance", "ir uncomp resistance", "ir comp percent"]},
    )
    assert missing_uc_table["IR Comp Resistance"].tolist() == ["96", "52.8"]
    assert missing_uc_table["IR Uncomp Resistance"].tolist() == ["0", "0"]
    assert missing_uc_table["IR Comp Percent"].tolist() == ["100", "100"]

    grouped = ecat_module.sort_and_group(
        [cpe_obj, dpv_obj, cv_obj],
        sort_keys="ir comp resistance",
        group_keys="ir comp percent",
        options={"print": False},
    )
    sorted_names = [obj.name for group in grouped for obj in group]
    assert sorted_names == [cv_obj.name, cpe_obj.name, dpv_obj.name]


def test_show_formats_ir_compensation_values_with_units(ecat_module, fixtures_dir):
    examples_dir = Path("tests/tmp_real_examples")
    objects = [
        ecat_module.echem.from_file(str(fixtures_dir / "ch_ca_tiny.txt"), {}),
        ecat_module.echem.from_file(
            str(
                examples_dir
                / "CV_post_MeCN_Ar_0.1MTBAPF6_1mMDiipThzOAc_50mMPhAc_100mMPhOH_0_to_-1.6V_100mVs.txt"
            ),
            {},
        ),
        ecat_module.echem.from_file(
            str(examples_dir / "DPV_MeCN_CO2_0.1MTBAPF6_3mMFc_1mMFe-tpyPY2Me_-0.7_to_-1.2V.txt"),
            {},
        ),
        ecat_module.echem.from_file(
            str(examples_dir / "CPE_MeCN_Ar_0.1MTBAPF6_1mMDiipThzOAc_50mMPhAc_100mMPhOH_-1.4V.txt"),
            {},
        ),
    ]

    for obj in objects:
        table = ecat_module.show(obj, {"pretty print": False, "return": True})
        values = dict(zip(table["Metric"], table["Value"]))

        assert values["IR Comp Resistance"].endswith(" ohm")
        assert values["IR Uncomp Resistance"].endswith(" ohm")
        assert values["IR Comp Percent"].endswith(" %")


def test_from_file_promotes_ch_cp_exports_to_cp(ecat_module, fixtures_dir):
    obj = ecat_module.echem.from_file(str(fixtures_dir / "ch_cp_tiny.txt"), {})

    assert type(obj).__name__ == "cp"
    assert obj.type == "Chronopotentiometry"
    assert list(obj.data.columns) == ["Time", "Potential"]
    assert obj.units == {"Time": "s", "Potential": "V"}
    assert obj.timestamp.year == 2022


def test_ch_cp_parser_reads_small_realistic_header_fields(ecat_module, fixtures_dir):
    obj = ecat_module.echem.from_file(str(fixtures_dir / "ch_cp_tiny.txt"), {})

    assert obj.cathodic_current == pytest.approx(0.005)
    assert obj.anodic_current == pytest.approx(0.005)
    assert obj.init_PN == "N"
    assert obj.sample_int == pytest.approx(0.5)
    assert obj.high_E_limit == pytest.approx(1.08)
    assert obj.low_E_limit == pytest.approx(0.38)
    assert obj.cathodic_time == pytest.approx(482.0)
    assert obj.anodic_time == pytest.approx(482.0)
    assert obj.segments == 201
    assert obj.delta_x == pytest.approx(0.5)
    assert obj.data["Potential"].tolist() == pytest.approx([0.718, 0.7173, 0.7169])


def test_from_file_promotes_eclab_gcpl_exports_to_cp(ecat_module, fixtures_dir):
    obj = ecat_module.echem.from_file(str(fixtures_dir / "eclab_gcpl_tiny.txt"), {})

    assert type(obj).__name__ == "cp"
    assert obj.type == "Chronopotentiometry"
    assert list(obj.data.columns) == ["Time", "Potential"]
    assert obj.units == {"Time": "s", "Potential": "V"}
    assert obj.data["Time"].tolist() == pytest.approx([0.0, 2.0, 4.0])
    assert obj.data["Potential"].tolist() == pytest.approx([0.700, 0.750, 0.690])
    assert obj.sample_int == pytest.approx(2.0)
    assert obj.cathodic_current == pytest.approx(-0.025)
    assert obj.anodic_current == pytest.approx(0.025)
    assert obj.segments == 2
    assert obj.delta_x == pytest.approx(2.0)


def test_get_data_loads_ca_and_cp_subclasses_from_small_folder_tree(
    ecat_module,
    fixtures_dir,
    tmp_path,
):
    ca_file = tmp_path / "ca_example.txt"
    cp_dir = tmp_path / "nested"
    cp_dir.mkdir()
    cp_file = cp_dir / "cp_example.txt"

    ca_file.write_text(
        (fixtures_dir / "ch_ca_tiny.txt").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    cp_file.write_text(
        (fixtures_dir / "ch_cp_tiny.txt").read_text(encoding="utf-8"),
        encoding="utf-8",
    )

    objects = ecat_module.get_data(
        {
            "folder path": str(tmp_path),
            "recursive search": True,
            "print": False,
            "shift potential": False,
        }
    )

    summary = {
        Path(obj.filepath).name: (type(obj).__name__, obj.folderpath)
        for obj in objects
    }
    assert summary == {
        "ca_example.txt": ("ca", ""),
        "cp_example.txt": ("cp", "nested"),
    }


def test_get_data_returns_empty_list_when_small_folder_tree_has_no_txt_files(ecat_module, tmp_path):
    (tmp_path / "notes.csv").write_text("a,b\n1,2\n", encoding="utf-8")

    result = ecat_module.get_data(
        {
            "folder path": str(tmp_path),
            "recursive search": False,
            "print": False,
            "shift potential": False,
        }
    )

    assert result == []
