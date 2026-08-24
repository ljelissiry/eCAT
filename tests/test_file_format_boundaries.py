from pathlib import Path
import warnings

import pandas as pd
import pytest


def _write_eclab_cv(path, *, scan_rate_mv_s=50.0):
    path.write_text(
        "\n".join(
            [
                "EC-Lab ASCII FILE",
                "Nb header lines : 12",
                "",
                "Cyclic Voltammetry",
                "Acquisition started on : 01/10/2018 17:01:24.000",
                f"dE/dt               {scan_rate_mv_s}",
                "dE/dt unit          mV/s",
                "N                   2",
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


def _write_ch_cv(path, *, header_scan_rate):
    path.write_text(
        "\n".join(
            [
                "June 16, 2026   14:41:35",
                "Cyclic Voltammetry",
                "Instrument Model: CHI760E",
                "Init E = -0.30",
                "High E = 0.30",
                "Low E = -0.30",
                f"Scan Rate (V/s) = {header_scan_rate}",
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


@pytest.mark.parametrize(
    ("suffix", "contents"),
    [
        (".mpr", b"binary data without the expected magic bytes"),
        (".bin", b"BIO-LOGIC MODULAR FILE\x1a\x00\xff\x00" + b"X" * 80),
    ],
)
def test_from_file_rejects_biologic_binary_before_text_parsing(
    ecat_module,
    tmp_path,
    suffix,
    contents,
):
    path = tmp_path / f"binary_cv{suffix}"
    path.write_bytes(contents)

    with pytest.raises(
        ecat_module.UnsupportedFileFormatError,
        match=r"BioLogic EC-Lab binary.*\.mpr.*ASCII.*\.mpt",
    ):
        ecat_module.echem.from_file(
            str(path),
            {"print": False, "reference mode": "none"},
        )


def test_get_data_discovers_txt_and_mpt_and_skips_mpr(
    ecat_module,
    tmp_path,
    capsys,
):
    _write_eclab_cv(tmp_path / "first_50mVs.mpt")
    _write_eclab_cv(tmp_path / "second_50mVs.txt")
    (tmp_path / "binary.mpr").write_bytes(
        b"BIO-LOGIC MODULAR FILE\x1a\x00\xff\x00" + b"X" * 80
    )

    objects = ecat_module.get_data(
        {
            "folder path": str(tmp_path),
            "recursive search": False,
            "reference mode": "none",
            "print": False,
        }
    )

    assert [Path(obj.filepath).name for obj in objects] == [
        "first_50mVs.mpt",
        "second_50mVs.txt",
    ]
    assert all(type(obj).__name__ == "cv" for obj in objects)
    output = capsys.readouterr().out
    assert "2 supported text files found." in output
    assert "Unsupported binary files skipped (1):" in output
    assert "`binary.mpr`" in output
    assert "EC-Lab ASCII `.mpt`" in output


def test_direct_custom_reader_can_handle_mpr(ecat_module, tmp_path):
    path = tmp_path / "custom_binary.mpr"
    path.write_bytes(b"BIO-LOGIC MODULAR FILE\x1a\x00\xff\x00" + b"X" * 80)

    def reader(obj, filepath, options):
        assert Path(filepath) == path
        return pd.DataFrame(
            {"Potential": [-0.1, 0.1], "Current": [-1e-6, 1e-6]}
        )

    obj = ecat_module.echem.from_file(
        str(path),
        {
            "custom reader": reader,
            "reference mode": "none",
            "print": False,
        },
    )

    assert obj.data["Current"].tolist() == pytest.approx([-1e-6, 1e-6])
    assert obj.parse_result.parser == "custom reader"


def test_eclab_mpt_support_is_consistent_across_detection_and_loading(
    ecat_module,
    tmp_path,
):
    path = tmp_path / "sample_50mVs.mpt"
    _write_eclab_cv(path)

    assert ecat_module.echem.detect_software(str(path), {}) == "EC-Lab"
    assert ecat_module.echem.detect_experiment_type(str(path), {}) == "Cyclic Voltammetry"
    assert type(ecat_module.echem.from_file(path, {"print": False})).__name__ == "cv"


@pytest.mark.parametrize(
    ("header_scan_rate", "filename_rate"),
    [
        (0.0249997, "25mVs"),
        (0.0499994, "50mVs"),
        (0.299999, "300mVs"),
        (0.499999, "500mVs"),
        (4.99999, "5Vs"),
    ],
)
def test_scan_rate_rounding_differences_do_not_warn(
    ecat_module,
    tmp_path,
    header_scan_rate,
    filename_rate,
):
    path = tmp_path / f"sample_{filename_rate}.txt"
    _write_ch_cv(path, header_scan_rate=header_scan_rate)

    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        obj = ecat_module.echem.from_file(str(path), {"print": False})

    assert obj.scan_rate == pytest.approx(header_scan_rate)
    assert not any("Scan rate mismatch" in str(item.message) for item in captured)
    assert not any("Scan rate mismatch" in item for item in obj.parse_result.warnings)


def test_genuine_scan_rate_mismatch_warns_and_keeps_header_value(
    ecat_module,
    tmp_path,
):
    path = tmp_path / "sample_500mVs.txt"
    _write_ch_cv(path, header_scan_rate=0.4)

    with pytest.warns(
        UserWarning,
        match=r"header reports 400 mV/s.*filename suggests 500 mV/s.*using header value",
    ):
        obj = ecat_module.echem.from_file(str(path), {"print": False})

    assert obj.scan_rate == pytest.approx(0.4)
