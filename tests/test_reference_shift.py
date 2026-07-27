import shutil
import warnings
from datetime import datetime
from pathlib import Path

import pytest


def _write_ch_cv(path, rows, timestamp="Aug. 27, 2023   16:05:21", scan_rate=0.05):
    lines = [
        timestamp,
        "Cyclic Voltammetry",
        "Instrument Model: CHI760E",
        "Init E = -0.30",
        "High E = 0.30",
        "Low E = -0.30",
        f"Scan Rate = {scan_rate}",
        "Segment = 2",
        "Sample Interval = 0.05",
        "Sensitivity = 1e-6",
        "Potential/V,Current/A",
    ]
    lines.extend(f"{x:.2f},{y}" for x, y in rows)
    path.write_text("\n".join(lines) + "\n", encoding="ISO-8859-1")


REFERENCE_ROWS = [
    (-0.30, -1.0e-7),
    (-0.25, -2.0e-7),
    (-0.20, -5.0e-7),
    (-0.15, -1.5e-6),
    (-0.10, -4.0e-6),
    (-0.05, -2.0e-6),
    (0.00, 0.0),
    (0.05, 2.0e-6),
    (0.10, 4.5e-6),
    (0.15, 1.8e-6),
    (0.20, 4.0e-7),
    (0.25, 1.0e-7),
    (0.30, 0.0),
]

SAMPLE_ROWS = [
    (-0.30, -1.0e-7),
    (-0.20, 0.0),
    (-0.10, 2.0e-7),
    (0.00, 4.0e-7),
    (0.10, 8.0e-7),
    (0.20, 3.0e-7),
    (0.30, 0.0),
]

FAILED_SELF_REFERENCE_ROWS = [
    (-0.30, 0.0),
    (-0.20, 1.0e-7),
    (-0.10, 2.0e-7),
    (0.00, 3.0e-7),
    (0.10, 2.0e-7),
    (0.20, 1.0e-7),
    (0.30, 0.0),
]


def test_potential_shift_exposes_virtual_reference_axis(cv_factory):
    obj = cv_factory()

    obj.potential_shift({"reference offset": 0.12, "reference label": "Fc/Fc+"})

    shifted_x = obj.x()
    raw_x = obj.x({"x axis": "Potential"})
    all_x = obj.x({"one column": False})

    assert shifted_x.name == "Potential vs Fc/Fc+"
    assert shifted_x.iloc[0] == pytest.approx(raw_x.iloc[0] - 0.12)
    assert list(all_x.columns) == ["Potential", "Potential vs Fc/Fc+"]


def test_get_data_applies_manual_reference_shift_metadata(ecat_module, fixtures_dir, tmp_path):
    shutil.copy(fixtures_dir / "ch_cv.txt", tmp_path / "sample_cv.txt")

    objects = ecat_module.get_data(
        {
            "folder path": str(tmp_path),
            "recursive search": False,
            "print": False,
            "reference mode": "manual",
            "reference offset": 0.12,
            "reference label": "Fc/Fc+",
        }
    )

    obj = objects[0]

    assert obj.reference_shift == pytest.approx(0.12)
    assert obj.reference_mode == "manual"
    assert obj.x().name == "Potential vs Fc/Fc+"


def test_get_data_reference_manual_requires_reference_offset(ecat_module, tmp_path):
    _write_ch_cv(tmp_path / "sample.txt", SAMPLE_ROWS)

    with pytest.raises(ValueError, match="'reference offset' is required"):
        ecat_module.get_data(
            {
                "folder path": str(tmp_path),
                "recursive search": False,
                "print": False,
                "reference mode": "manual",
            }
        )


def test_get_data_reference_keyword_requires_reference_keyword(ecat_module, tmp_path):
    _write_ch_cv(tmp_path / "sample.txt", SAMPLE_ROWS)
    _write_ch_cv(tmp_path / "Fc_reference.txt", REFERENCE_ROWS)

    with pytest.raises(ValueError, match="reference keyword.*required"):
        ecat_module.get_data(
            {
                "folder path": str(tmp_path),
                "recursive search": False,
                "print": False,
                "reference mode": "keyword",
            }
        )


def test_get_data_reference_file_missing_uses_formatted_relative_path(ecat_module, tmp_path):
    nested = tmp_path / "nested"
    nested.mkdir()
    _write_ch_cv(nested / "sample.txt", SAMPLE_ROWS)

    with pytest.raises(ValueError) as exc_info:
        ecat_module.get_data(
            {
                "folder path": str(tmp_path),
                "recursive search": True,
                "print": False,
                "reference mode": "file",
                "reference file": "nested/missing_reference.txt",
            }
        )

    message = str(exc_info.value)
    assert "Reference file does not exist:" in message
    assert "`nested/missing_reference.txt`" in message
    assert str(tmp_path) not in message


def test_get_data_prints_unique_import_warnings_without_python_warning_context(
    ecat_module,
    tmp_path,
    capsys,
):
    nested = tmp_path / "nested"
    nested.mkdir()
    _write_ch_cv(
        nested / "MeCN_Ar_Fc_500mVs.txt",
        REFERENCE_ROWS,
        scan_rate=0.2,
    )

    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        objects = ecat_module.get_data(
            {
                "folder path": str(tmp_path),
                "recursive search": True,
                "print": True,
                "reference mode": "keyword",
                "reference keyword": "Fc",
                "reference guess": 0.0,
                "peak prominence": 1e-6,
                "reference smooth": False,
                "troubleshoot": True,
            }
        )

    output = capsys.readouterr().out
    mismatch_warnings = [
        warning for warning in captured
        if "Scan rate mismatch" in str(warning.message)
    ]

    assert len(objects) == 1
    assert not mismatch_warnings
    assert output.count("Scan rate mismatch") == 1
    assert "Scan rate mismatch for `nested/MeCN_Ar_Fc_500mVs.txt`" in output
    assert "Scan rate mismatch for `MeCN_Ar_Fc_500mVs.txt`" not in output
    assert "Getting data from `nested/MeCN_Ar_Fc_500mVs.txt`" in output
    assert "UserWarning" not in output
    assert "objects.py:" not in output
    assert "parsers.py:" not in output
    assert any(
        "Scan rate mismatch for `nested/MeCN_Ar_Fc_500mVs.txt`" in warning
        for warning in objects[0].parse_result.warnings
    )


def test_get_data_reference_mode_rejects_empty_reference_keyword(ecat_module, tmp_path):
    _write_ch_cv(tmp_path / "sample.txt", SAMPLE_ROWS)

    with pytest.raises(ValueError, match="non-empty 'reference keyword'"):
        ecat_module.get_data(
            {
                "folder path": str(tmp_path),
                "recursive search": False,
                "print": False,
                "reference mode": "keyword",
                "reference keyword": "   ",
            }
        )


def test_get_data_reference_accepts_numeric_reference_keyword(ecat_module, tmp_path):
    _write_ch_cv(tmp_path / "1_ref.txt", REFERENCE_ROWS)
    _write_ch_cv(tmp_path / "sample.txt", SAMPLE_ROWS)

    objects = ecat_module.get_data(
        {
            "folder path": str(tmp_path),
            "recursive search": False,
            "print": False,
            "reference mode": "keyword",
            "reference keyword": 1,
            "reference guess": 0.0,
            "peak prominence": 1e-6,
            "reference smooth": False,
        }
    )

    sample = next(obj for obj in objects if obj.filepath.endswith("sample.txt"))
    assert sample.reference_mode in {"folder", "self", "fallback"}
    assert sample.reference_shift == pytest.approx(0.0, abs=1e-12)


def test_get_data_reference_file_requires_reference_file(ecat_module, tmp_path):
    _write_ch_cv(tmp_path / "sample.txt", SAMPLE_ROWS)

    with pytest.raises(ValueError, match="'reference file' is required"):
        ecat_module.get_data(
            {
                "folder path": str(tmp_path),
                "recursive search": False,
                "print": False,
                "reference mode": "file",
            }
        )


def test_get_data_prints_reference_columns_when_reference_shift_active(
    ecat_module,
    fixtures_dir,
    tmp_path,
    monkeypatch,
):
    from ecat import io as ecat_io

    shutil.copy(fixtures_dir / "ch_cv.txt", tmp_path / "sample_cv.txt")
    printed_options = []

    def spy_show_objects(object_list, options):
        printed_options.append(dict(options))

    monkeypatch.setattr(ecat_io, "show_objects", spy_show_objects)

    objects = ecat_module.get_data(
        {
            "folder path": str(tmp_path),
            "recursive search": False,
            "print": True,
            "reference mode": "manual",
            "reference offset": 0.12,
            "reference label": "Fc/Fc+",
        }
    )

    assert len(objects) == 1
    assert objects[0].reference_shift == pytest.approx(0.12)
    assert printed_options
    assert "reference shift" in printed_options[0]["columns"]
    assert "reference mode" in printed_options[0]["columns"]
    assert "reference source" in printed_options[0]["columns"]


def test_get_data_keyword_reference_suppresses_internal_reference_show(
    ecat_module,
    tmp_path,
    monkeypatch,
):
    _write_ch_cv(tmp_path / "Fc_reference.txt", REFERENCE_ROWS)
    _write_ch_cv(tmp_path / "sample.txt", SAMPLE_ROWS)
    calls = {"show": 0, "show_objects": 0}

    def fake_show(*args, **kwargs):
        calls["show"] += 1

    def fake_show_objects(*args, **kwargs):
        calls["show_objects"] += 1

    monkeypatch.setattr(ecat_module.plotting, "show", fake_show)
    monkeypatch.setattr(ecat_module.io, "show_objects", fake_show_objects)

    objects = ecat_module.get_data(
        {
            "folder path": str(tmp_path),
            "recursive search": False,
            "print": True,
            "reference mode": "keyword",
            "reference keyword": "Fc",
            "reference guess": 0.0,
            "peak prominence": 1e-6,
            "reference smooth": False,
        }
    )

    assert len(objects) == 2
    assert calls == {"show": 0, "show_objects": 1}


def test_get_data_applies_explicit_reference_file_to_imported_cvs(
    ecat_module,
    monkeypatch,
    tmp_path,
):
    _write_ch_cv(tmp_path / "sample_cv.txt", SAMPLE_ROWS)
    _write_ch_cv(tmp_path / "explicit_reference.txt", REFERENCE_ROWS)

    calls = []

    def fake_midpoint(ref_cv, **kwargs):
        calls.append(Path(ref_cv.filepath).name)
        return 0.222, {}

    monkeypatch.setattr(ecat_module, "find_reference_midpoint_from_cv", fake_midpoint)

    objects = ecat_module.get_data(
        {
            "folder path": str(tmp_path),
            "recursive search": False,
            "print": True,
            "reference mode": "file",
            "reference file": "explicit_reference.txt",
            "reference guess": 0.0,
        }
    )

    sample = next(obj for obj in objects if obj.filepath.endswith("sample_cv.txt"))

    assert calls == ["explicit_reference.txt"]
    assert sample.reference_mode == "file"
    assert sample.reference_source_file.endswith("explicit_reference.txt")
    assert sample.reference_shift == pytest.approx(0.222)
    assert sample.x().name == "Potential vs Fc/Fc+"


def test_get_data_defaults_to_instrument_timestamp_order(ecat_module, tmp_path):
    _write_ch_cv(
        tmp_path / "a_late.txt",
        SAMPLE_ROWS,
        timestamp="Aug. 27, 2023   16:10:21",
    )
    _write_ch_cv(
        tmp_path / "z_early.txt",
        SAMPLE_ROWS,
        timestamp="Aug. 27, 2023   16:05:21",
    )

    objects = ecat_module.get_data(
        {
            "folder path": str(tmp_path),
            "recursive search": False,
            "print": False,
            "reference mode": "none",
        }
    )

    assert [Path(obj.filepath).name for obj in objects] == [
        "z_early.txt",
        "a_late.txt",
    ]


def test_get_data_defaults_to_subfolder_then_timestamp_order(ecat_module, tmp_path):
    folder1 = tmp_path / "folder1"
    folder2 = tmp_path / "folder2"
    folder1.mkdir()
    folder2.mkdir()

    _write_ch_cv(
        folder1 / "z_time3.txt",
        SAMPLE_ROWS,
        timestamp="Aug. 27, 2023   16:03:21",
    )
    _write_ch_cv(
        folder1 / "a_time2.txt",
        SAMPLE_ROWS,
        timestamp="Aug. 27, 2023   16:02:21",
    )
    _write_ch_cv(
        folder2 / "z_time1.txt",
        SAMPLE_ROWS,
        timestamp="Aug. 27, 2023   16:01:21",
    )
    _write_ch_cv(
        folder2 / "a_time4.txt",
        SAMPLE_ROWS,
        timestamp="Aug. 27, 2023   16:04:21",
    )

    objects = ecat_module.get_data(
        {
            "folder path": str(tmp_path),
            "print": False,
            "reference mode": "none",
        }
    )

    assert [Path(obj.filepath).relative_to(tmp_path).as_posix() for obj in objects] == [
        "folder1/a_time2.txt",
        "folder1/z_time3.txt",
        "folder2/z_time1.txt",
        "folder2/a_time4.txt",
    ]


def test_get_data_sort_keys_override_timestamp_order(ecat_module, tmp_path):
    _write_ch_cv(
        tmp_path / "a_late.txt",
        SAMPLE_ROWS,
        timestamp="Aug. 27, 2023   16:10:21",
    )
    _write_ch_cv(
        tmp_path / "z_early.txt",
        SAMPLE_ROWS,
        timestamp="Aug. 27, 2023   16:05:21",
    )

    objects = ecat_module.get_data(
        {
            "folder path": str(tmp_path),
            "recursive search": False,
            "print": False,
            "reference mode": "none",
            "sort keys": ["name"],
        }
    )

    assert [Path(obj.filepath).name for obj in objects] == [
        "a_late.txt",
        "z_early.txt",
    ]


def test_get_data_timestamp_order_falls_back_when_timestamp_is_unparseable(
    ecat_module,
    monkeypatch,
    tmp_path,
):
    bad_time = tmp_path / "bad_timestamp.txt"
    valid_time = tmp_path / "valid_timestamp.txt"
    _write_ch_cv(bad_time, SAMPLE_ROWS, timestamp="not-a-real-timestamp")
    _write_ch_cv(valid_time, SAMPLE_ROWS, timestamp="Aug. 27, 2023   16:05:21")

    fallback_times = {
        str(bad_time): datetime(2023, 8, 27, 16, 0, 0),
        str(valid_time): datetime(2023, 8, 27, 17, 0, 0),
    }

    def fake_file_times(filepath):
        value = fallback_times[str(filepath)]
        return value, value

    monkeypatch.setattr(ecat_module, "get_file_times", fake_file_times)

    objects = ecat_module.get_data(
        {
            "folder path": str(tmp_path),
            "recursive search": False,
            "print": False,
            "reference mode": "none",
        }
    )

    assert [Path(obj.filepath).name for obj in objects] == [
        "bad_timestamp.txt",
        "valid_timestamp.txt",
    ]


def test_get_data_uses_nearest_ancestor_reference_source(ecat_module, tmp_path):
    subdir = tmp_path / "sub"
    subdir.mkdir()

    _write_ch_cv(tmp_path / "Fc_reference.txt", REFERENCE_ROWS)
    _write_ch_cv(subdir / "sample_cv.txt", SAMPLE_ROWS, timestamp="Aug. 27, 2023   16:10:21")

    objects = ecat_module.get_data(
        {
            "folder path": str(tmp_path),
            "recursive search": True,
            "print": False,
            "reference mode": "keyword",
            "reference keyword": "Fc",
            "reference guess": 0.0,
            "reference window": 0.3,
            "peak prominence": 1e-6,
            "reference smooth": False,
        }
    )

    sample = next(obj for obj in objects if obj.filepath.endswith("sample_cv.txt"))

    assert sample.reference_mode == "folder"
    assert sample.reference_source_file.endswith("Fc_reference.txt")
    assert sample.reference_shift == pytest.approx(0.0, abs=1e-12)
    assert sample.x().name == "Potential vs Fc/Fc+"


def test_get_data_prints_structured_reference_correction_summary(ecat_module, tmp_path, capsys):
    _write_ch_cv(tmp_path / "Fc_reference.txt", REFERENCE_ROWS)
    _write_ch_cv(
        tmp_path / "sample_cv.txt",
        SAMPLE_ROWS,
        timestamp="Aug. 27, 2023   16:10:21",
    )

    ecat_module.get_data(
        {
            "folder path": str(tmp_path),
            "recursive search": False,
            "print": True,
            "reference mode": "keyword",
            "reference keyword": "Fc",
            "reference guess": 0.0,
            "reference window": 0.3,
            "peak prominence": 1e-6,
            "reference smooth": False,
            "allow self reference": False,
        }
    )

    output = capsys.readouterr().out

    assert "Reference correction:\n" in output
    assert "  Mode: keyword\n" in output
    assert "  Keyword: Fc\n" in output
    assert "  Guess: 0 V\n" in output
    assert "  Folder reference: `Fc_reference.txt` = 0 V\n" in output
    assert "  Usage:\n    folder/ancestor reference: 2\n" in output
    assert "\n\n[Conditions]" in output
    assert "Finding reference couple" not in output
    assert "Folder reference assignment" not in output
    assert "Reference usage summary" not in output


def test_get_data_prefers_local_folder_reference_over_parent(ecat_module, tmp_path):
    subdir = tmp_path / "sub"
    subdir.mkdir()

    _write_ch_cv(tmp_path / "Fc_parent.txt", REFERENCE_ROWS)
    _write_ch_cv(subdir / "Fc_local.txt", REFERENCE_ROWS, timestamp="Aug. 27, 2023   16:07:21")
    _write_ch_cv(subdir / "sample_cv.txt", SAMPLE_ROWS, timestamp="Aug. 27, 2023   16:10:21")

    objects = ecat_module.get_data(
        {
            "folder path": str(tmp_path),
            "recursive search": True,
            "print": False,
            "reference mode": "keyword",
            "reference keyword": "Fc",
            "reference guess": 0.0,
            "reference window": 0.3,
            "peak prominence": 1e-6,
            "reference smooth": False,
        }
    )

    sample = next(obj for obj in objects if obj.filepath.endswith("sample_cv.txt"))

    assert sample.reference_mode == "folder"
    assert sample.reference_source_file.endswith("Fc_local.txt")
    assert sample.reference_shift == pytest.approx(0.0, abs=1e-12)


def test_get_data_labels_designated_reference_file_as_self_when_allowed(
    ecat_module,
    tmp_path,
    capsys,
):
    _write_ch_cv(tmp_path / "Fc_reference.txt", REFERENCE_ROWS)
    _write_ch_cv(
        tmp_path / "sample_cv.txt",
        SAMPLE_ROWS,
        timestamp="Aug. 27, 2023   16:10:21",
    )

    objects = ecat_module.get_data(
        {
            "folder path": str(tmp_path),
            "recursive search": False,
            "print": True,
            "reference mode": "keyword",
            "reference keyword": "Fc",
            "reference guess": 0.0,
            "reference window": 0.3,
            "peak prominence": 1e-6,
            "reference smooth": False,
        }
    )

    reference = next(obj for obj in objects if obj.filepath.endswith("Fc_reference.txt"))
    sample = next(obj for obj in objects if obj.filepath.endswith("sample_cv.txt"))

    assert reference.reference_mode == "self"
    assert reference.reference_source_file == reference.filepath
    assert reference.reference_shift == pytest.approx(0.0, abs=1e-12)
    assert sample.reference_mode == "folder"
    assert sample.reference_source_file == reference.filepath
    assert sample.reference_shift == pytest.approx(0.0, abs=1e-12)

    output = capsys.readouterr().out
    assert "    folder/ancestor reference: 1\n" in output
    assert "    self-referenced successfully: 1\n" in output


def test_get_data_tries_multiple_reference_candidates_in_timestamp_order(
    ecat_module,
    monkeypatch,
    tmp_path,
):
    _write_ch_cv(
        tmp_path / "a_good_Fc.txt",
        REFERENCE_ROWS,
        timestamp="Aug. 27, 2023   16:10:21",
    )
    _write_ch_cv(
        tmp_path / "z_bad_Fc.txt",
        REFERENCE_ROWS,
        timestamp="Aug. 27, 2023   16:05:21",
    )
    _write_ch_cv(
        tmp_path / "sample_cv.txt",
        SAMPLE_ROWS,
        timestamp="Aug. 27, 2023   16:20:21",
    )

    calls = []

    def fake_midpoint(ref_cv, **kwargs):
        filename = Path(ref_cv.filepath).name
        calls.append(filename)
        if filename == "z_bad_Fc.txt":
            raise ValueError("bad reference")
        return 0.123, {}

    monkeypatch.setattr(ecat_module, "find_reference_midpoint_from_cv", fake_midpoint)

    objects = ecat_module.get_data(
        {
            "folder path": str(tmp_path),
            "recursive search": False,
            "print": False,
            "reference mode": "keyword",
            "reference keyword": "Fc",
            "reference guess": 0.0,
            "allow self reference": False,
        }
    )

    sample = next(obj for obj in objects if obj.filepath.endswith("sample_cv.txt"))

    assert calls[:2] == ["z_bad_Fc.txt", "a_good_Fc.txt"]
    assert sample.reference_source_file.endswith("a_good_Fc.txt")
    assert sample.reference_shift == pytest.approx(0.123)


def test_get_data_reference_map_overrides_auto_reference_assignment(
    ecat_module,
    monkeypatch,
    tmp_path,
):
    _write_ch_cv(
        tmp_path / "target_sample.txt",
        SAMPLE_ROWS,
        timestamp="Aug. 27, 2023   16:05:21",
    )
    _write_ch_cv(
        tmp_path / "auto_Fc.txt",
        REFERENCE_ROWS,
        timestamp="Aug. 27, 2023   16:10:21",
    )
    _write_ch_cv(
        tmp_path / "manual_Fc.txt",
        REFERENCE_ROWS,
        timestamp="Aug. 27, 2023   16:15:21",
    )

    def fake_midpoint(ref_cv, **kwargs):
        filename = Path(ref_cv.filepath).name
        if filename == "auto_Fc.txt":
            return 0.111, {}
        if filename == "manual_Fc.txt":
            return 0.456, {}
        raise AssertionError(f"unexpected reference file: {filename}")

    monkeypatch.setattr(ecat_module, "find_reference_midpoint_from_cv", fake_midpoint)

    objects = ecat_module.get_data(
        {
            "folder path": str(tmp_path),
            "recursive search": False,
            "print": False,
            "reference mode": "keyword",
            "reference keyword": "Fc",
            "reference guess": 0.0,
            "allow self reference": False,
            "sort keys": ["timestamp"],
            "reference map": {0: 2},
        }
    )

    assert [Path(obj.filepath).name for obj in objects] == [
        "target_sample.txt",
        "auto_Fc.txt",
        "manual_Fc.txt",
    ]
    assert objects[0].reference_mode == "map"
    assert objects[0].reference_source_file.endswith("manual_Fc.txt")
    assert objects[0].reference_shift == pytest.approx(0.456)


def test_get_data_falls_back_to_parent_reference_when_local_reference_fails(ecat_module, tmp_path):
    subdir = tmp_path / "sub"
    subdir.mkdir()

    _write_ch_cv(tmp_path / "Fc_parent.txt", REFERENCE_ROWS)
    _write_ch_cv(
        subdir / "DPV_sample_Fc.txt",
        FAILED_SELF_REFERENCE_ROWS[:4],
        timestamp="Aug. 27, 2023   16:07:21",
    )
    _write_ch_cv(subdir / "sample_cv.txt", SAMPLE_ROWS, timestamp="Aug. 27, 2023   16:10:21")

    objects = ecat_module.get_data(
        {
            "folder path": str(tmp_path),
            "recursive search": True,
            "print": False,
            "reference mode": "keyword",
            "reference keyword": "Fc",
            "reference guess": 0.0,
            "reference window": 0.3,
            "peak prominence": 1e-6,
            "reference smooth": False,
        }
    )

    sample = next(obj for obj in objects if obj.filepath.endswith("sample_cv.txt"))

    assert sample.reference_mode == "folder"
    assert sample.reference_source_file.endswith("Fc_parent.txt")
    assert sample.reference_shift == pytest.approx(0.0, abs=1e-12)


def test_reference_like_sample_falls_back_to_folder_reference_when_self_reference_fails(
    ecat_module,
    tmp_path,
):
    _write_ch_cv(tmp_path / "Fc_reference.txt", REFERENCE_ROWS)
    _write_ch_cv(
        tmp_path / "sample_Fc_cv.txt",
        FAILED_SELF_REFERENCE_ROWS,
        timestamp="Aug. 27, 2023   16:10:21",
    )

    objects = ecat_module.get_data(
        {
            "folder path": str(tmp_path),
            "recursive search": False,
            "print": False,
            "reference mode": "keyword",
            "reference keyword": "Fc",
            "reference guess": 0.0,
            "reference window": 0.3,
            "peak prominence": 1e-6,
            "reference smooth": False,
        }
    )

    sample = next(obj for obj in objects if obj.filepath.endswith("sample_Fc_cv.txt"))

    assert sample.reference_mode == "fallback"
    assert sample.reference_source_file.endswith("Fc_reference.txt")
    assert sample.reference_shift == pytest.approx(0.0, abs=1e-12)
    assert "Failed to identify a reference couple" in sample.reference_failure_message
    assert "File: `sample_Fc_cv.txt`" in sample.reference_failure_message
