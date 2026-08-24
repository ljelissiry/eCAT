from datetime import datetime

import pytest


def test_eclab_cv_parse_result_preserves_raw_metadata_and_canonical_units(
    ecat_module,
    tmp_path,
):
    filepath = tmp_path / "eclab_cv.mpt"
    filepath.write_text(
        "\n".join(
            [
                "EC-Lab ASCII FILE",
                "Nb header lines : 17",
                "",
                "Cyclic Voltammetry",
                "Acquisition started on : 01/10/2018 17:01:24.000",
                "dE/dt               50.000",
                "dE/dt unit          mV/s",
                "N                   2",
                "tR (h:m:s)          0:00:4.0000",
                "Electrode surface area : 0.001 cm²",
                "E range min (V)     -1.000",
                "E range max (V)     1.000",
                "",
                "",
                "",
                "",
                "mode\ttime/s\tEwe/V\t<I>/mA\tcycle number",
                "2\t0.0\t-0.25\t-0.10\t1",
                "2\t1.0\t0.00\t0.20\t1",
                "2\t2.0\t0.25\t0.30\t2",
            ]
        )
        + "\n",
        encoding="ISO-8859-1",
    )

    parsed = ecat_module.echem.parse_file_to_result(str(filepath), {})

    assert parsed.technique == "CV"
    assert parsed.software == "EC-Lab"
    assert parsed.parser == "EC-Lab"
    assert list(parsed.data.columns[:2]) == ["Potential", "Current"]
    assert parsed.units["Potential"] == "V"
    assert parsed.units["Current"] == "A"
    assert parsed.data["Current"].tolist() == pytest.approx([-1e-4, 2e-4, 3e-4])
    assert parsed.metadata["scan_rate"] == pytest.approx(0.05)
    assert parsed.metadata["segments"] == 2
    assert parsed.metadata["acquisition_start"] == datetime(2018, 1, 10, 17, 1, 24)
    assert parsed.raw_metadata["header_line_count"] == 17
    assert parsed.raw_metadata["data_header_line"] == "mode\ttime/s\tEwe/V\t<I>/mA\tcycle number"
    assert parsed.raw_metadata["original_units"]["<I>"] == "mA"


def test_eclab_ca_parse_result_promotes_to_ca_and_preserves_step_metadata(
    ecat_module,
    tmp_path,
):
    filepath = tmp_path / "eclab_ca.mpt"
    filepath.write_text(
        "\n".join(
            [
                "EC-Lab ASCII FILE",
                "Nb header lines : 14",
                "",
                "Chronoamperometry / Chronocoulometry",
                "Acquisition started on : 04/29/2019 15:43:07.000",
                "Ei (V)              0.300",
                "dt (s)              60.0000",
                "dI                  0.200",
                "unit dI             mA",
                "E range min (V)     -0.500",
                "E range max (V)     0.500",
                "",
                "",
                "mode\ttime/s\tcontrol/V\tEwe/V\tI/mA\tdQ/C",
                "2\t0.0\t0.300\t0.146\t0.0186\t0.0",
                "2\t60.0\t0.300\t0.299\t0.0000289\t0.0000245",
                "2\t120.0\t0.300\t0.298\t0.0000195\t0.0000258",
            ]
        )
        + "\n",
        encoding="ISO-8859-1",
    )

    parsed = ecat_module.echem.parse_file_to_result(str(filepath), {})
    obj = ecat_module.echem.from_file(str(filepath), {})

    assert parsed.technique == "CA"
    assert parsed.metadata["type"] == "Chronoamperometry"
    assert list(parsed.data.columns[:3]) == ["Time", "Current", "Potential"]
    assert parsed.units == {"Time": "s", "Current": "A", "Potential": "V", "Charge": "C"}
    assert parsed.metadata["sample_interval"] == pytest.approx(60.0)
    assert parsed.metadata["quiet_time"] is None
    assert parsed.raw_metadata["step_table"]["dI"]["values"] == [0.2]
    assert parsed.raw_metadata["step_table"]["dI"]["units"] == ["mA"]
    assert not any("timestamp" in warning.lower() for warning in parsed.warnings)
    assert type(obj).__name__ == "ca"
    assert obj.parse_result.parser == parsed.parser
    assert obj.parse_result.raw_metadata["step_table"] == parsed.raw_metadata["step_table"]
    assert obj.x().tolist() == pytest.approx([0.0, 60.0, 120.0])
    assert obj.y().tolist() == pytest.approx([1.86e-5, 2.89e-8, 1.95e-8])


def test_eclab_cp_parse_result_preserves_reference_and_promotes_to_cp(
    ecat_module,
    tmp_path,
):
    filepath = tmp_path / "eclab_cp.mpt"
    filepath.write_text(
        "\n".join(
            [
                "EC-Lab ASCII FILE",
                "Nb header lines : 15",
                "",
                "Chronopotentiometry",
                "Acquisition started on : 03/02/2021 16:17:59.000",
                "Reference electrode : SCE Saturated Calomel Electrode (0.241 V)",
                "Is                  -100.000",
                "unit Is             mA",
                "dts (s)             1.0000",
                "E range min (V)     -10.000",
                "E range max (V)     10.000",
                "",
                "",
                "",
                "mode\ttime/s\tcontrol/mA\t<Ewe>/V\tI/mA\tcycle number",
                "1\t0.0\t-100.0\t-3.246\t-99.888\t1",
                "1\t1.0\t-100.0\t-3.449\t-99.915\t1",
                "1\t2.0\t-100.0\t-3.455\t-99.926\t1",
            ]
        )
        + "\n",
        encoding="ISO-8859-1",
    )

    parsed = ecat_module.echem.parse_file_to_result(str(filepath), {})
    obj = ecat_module.echem.from_file(str(filepath), {})

    assert parsed.technique == "CP"
    assert parsed.metadata["type"] == "Chronopotentiometry"
    assert parsed.metadata["reference_electrode"] == "SCE Saturated Calomel Electrode (0.241 V)"
    assert list(parsed.data.columns[:3]) == ["Time", "Potential", "Current"]
    assert parsed.units == {"Time": "s", "Potential": "V", "Current": "A"}
    assert parsed.data["Current"].tolist() == pytest.approx([-0.099888, -0.099915, -0.099926])
    assert parsed.metadata["sample_interval"] == pytest.approx(1.0)
    assert parsed.raw_metadata["step_table"]["Is"]["values"] == [-100.0]
    assert parsed.raw_metadata["step_table"]["Is"]["units"] == ["mA"]
    assert type(obj).__name__ == "cp"
    assert obj.parse_result.parser == parsed.parser
    assert obj.parse_result.raw_metadata["step_table"] == parsed.raw_metadata["step_table"]
    assert list(obj.data.columns[:3]) == ["Time", "Potential", "Current"]
    assert obj.units["Current"] == "A"
    assert obj.parse_result.data["Current"].tolist() == pytest.approx(
        [-0.099888, -0.099915, -0.099926]
    )
    assert obj.y().tolist() == pytest.approx([-3.246, -3.449, -3.455])


def test_nova_ascii_cv_parse_result_and_promotion(ecat_module, tmp_path):
    filepath = tmp_path / "nova_cv.txt"
    filepath.write_text(
        "\n".join(
            [
                "NOVA ASCII export",
                "Technique: Cyclic Voltammetry",
                "Time (s);WE(1).Potential (V);WE(1).Current (uA)",
                "0.0;-0.10;-2.0",
                "1.0;0.00;0.0",
                "2.0;0.10;2.5",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    parsed = ecat_module.echem.parse_file_to_result(str(filepath), {})
    obj = ecat_module.echem.from_file(str(filepath), {})

    assert parsed.technique == "CV"
    assert parsed.software == "NOVA"
    assert parsed.raw_metadata["delimiter"] == ";"
    assert parsed.raw_metadata["original_units"]["WE(1).Current"] == "uA"
    assert parsed.data["Current"].tolist() == pytest.approx([-2e-6, 0.0, 2.5e-6])
    assert any("scan rate" in warning.lower() for warning in parsed.warnings)
    assert type(obj).__name__ == "cv"
    assert obj.parse_result.parser == parsed.parser
    assert obj.parse_result.raw_metadata["original_units"] == parsed.raw_metadata["original_units"]


def test_generic_text_parse_result_warns_for_inferred_cv_like_axes(
    ecat_module,
    tmp_path,
):
    filepath = tmp_path / "generic_cv_like.txt"
    filepath.write_text(
        "Instrument export without technique\n"
        "Potential/V Current/mA\n"
        "-0.25 -0.10\n"
        "0.00 0.20\n"
        "0.25 0.30\n",
        encoding="utf-8",
    )

    parsed = ecat_module.echem.parse_file_to_result(str(filepath), {})
    obj = ecat_module.echem.from_file(str(filepath), {})

    assert parsed.technique == "CV"
    assert parsed.software == "Generic Text"
    assert parsed.metadata["type"] == "Cyclic Voltammetry"
    assert parsed.raw_metadata["data_header_line"] == "Potential/V Current/mA"
    assert parsed.data["Current"].tolist() == pytest.approx([-1e-4, 2e-4, 3e-4])
    assert any("inferred" in warning.lower() for warning in parsed.warnings)
    assert any("scan rate" in warning.lower() for warning in parsed.warnings)
    assert type(obj).__name__ == "cv"
    assert obj.parse_result.parser == parsed.parser
    assert obj.parse_result.raw_metadata["parser_notes"] == parsed.raw_metadata["parser_notes"]


def test_generic_text_with_time_current_and_potential_stays_generic(
    ecat_module,
    tmp_path,
):
    filepath = tmp_path / "generic_ambiguous_chrono.txt"
    filepath.write_text(
        "Instrument export without technique\n"
        "Time/s Current/mA Potential/V\n"
        "0.0 0.10 -0.25\n"
        "1.0 0.20 -0.20\n"
        "2.0 0.30 -0.15\n",
        encoding="utf-8",
    )

    parsed = ecat_module.echem.parse_file_to_result(str(filepath), {})
    obj = ecat_module.echem.from_file(str(filepath), {})

    assert parsed.technique == "unknown"
    assert parsed.metadata["type"] == "Unknown"
    assert list(parsed.data.columns) == ["Time", "Current", "Potential"]
    assert parsed.units == {"Time": "s", "Current": "A", "Potential": "V"}
    assert any("technique was left unknown" in warning.lower() for warning in parsed.warnings)
    assert type(obj).__name__ == "echem"
    assert obj.type == "Unknown"
    assert obj.parse_result.data["Current"].tolist() == pytest.approx([1e-4, 2e-4, 3e-4])
