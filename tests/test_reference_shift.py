import shutil
import warnings
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
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
    (-0.25, -5.0e-8),
    (-0.20, 0.0),
    (-0.15, 1.0e-7),
    (-0.10, 3.0e-7),
    (-0.05, 8.0e-7),
    (0.00, 1.5e-6),
    (0.05, 2.8e-6),
    (0.10, 4.5e-6),
    (0.15, 2.6e-6),
    (0.20, 1.0e-6),
    (0.25, 3.0e-7),
    (0.30, 0.0),
    (0.25, -1.0e-7),
    (0.20, -3.0e-7),
    (0.15, -6.0e-7),
    (0.10, -1.0e-6),
    (0.05, -2.4e-6),
    (0.00, -3.6e-6),
    (-0.05, -4.0e-6),
    (-0.10, -4.5e-6),
    (-0.15, -4.0e-7),
    (-0.20, -1.0e-7),
    (-0.25, -5.0e-8),
    (-0.30, 0.0),
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


def _segmented_reference_cv(cv_factory, segments, *, name="100mVs_segmented_Fc_reference"):
    potential = []
    current = []
    for segment_index, (start, stop, points, current_function) in enumerate(segments):
        segment_potential = np.linspace(start, stop, points)
        if segment_index:
            segment_potential = segment_potential[1:]
        potential.extend(segment_potential)
        current.extend(current_function(segment_potential))
    return cv_factory(name=name, potential=potential, current=current)


def _lje_reference_regression_cv(cv_factory, *, invert=False):
    def partial_current(E):
        return 2.0e-7 * E

    def anodic_current(E):
        reference_peak = 6.0e-6 * np.exp(-0.5 * ((E - 0.529) / 0.018) ** 2)
        unrelated_minimum = -4.5e-6 * np.exp(-0.5 * ((E - 0.343) / 0.015) ** 2)
        return reference_peak + unrelated_minimum + 1.0e-7 * E

    def cathodic_current(E):
        return -5.5e-6 * np.exp(-0.5 * ((E - 0.470) / 0.018) ** 2) + 1.0e-7 * E

    obj = _segmented_reference_cv(
        cv_factory,
        [
            (0.10, -0.20, 301, partial_current),
            (-0.20, 0.70, 901, anodic_current),
            (0.70, -0.20, 901, cathodic_current),
        ],
    )
    if invert:
        obj.data["Current"] = -obj.data["Current"]
    return obj


def _reference_pair_options(**overrides):
    options = {
        "guess": 0.4,
        "window": 0.35,
        "prominence": 5e-7,
        "max_delta_ep": 0.20,
        "target_delta_ep": 0.08,
        "smooth": False,
    }
    options.update(overrides)
    return options


def test_reference_pairing_uses_adjacent_opposite_direction_segments(
    ecat_module,
    cv_factory,
):
    obj = _lje_reference_regression_cv(cv_factory)

    midpoint, details = ecat_module.find_reference_midpoint_from_cv(
        obj,
        **_reference_pair_options(),
    )

    assert midpoint == pytest.approx(0.4995, abs=1e-3)
    assert details["Epa"] == pytest.approx(0.529, abs=1e-3)
    assert details["Epc"] == pytest.approx(0.470, abs=1e-3)
    assert details["anodic_segment"] == 2
    assert details["cathodic_segment"] == 3
    assert details["anodic_scan_direction"] == "increasing"
    assert details["cathodic_scan_direction"] == "decreasing"
    assert details["selection_mode"] == "ranked adjacent segments"
    assert details["sequential_pairs_examined"] == 2
    assert "Ep_ox" not in details
    assert "Ep_red" not in details


def test_reference_pairing_is_invariant_to_current_inversion(ecat_module, cv_factory):
    normal = _lje_reference_regression_cv(cv_factory)
    inverted = _lje_reference_regression_cv(cv_factory, invert=True)

    normal_midpoint, normal_details = ecat_module.find_reference_midpoint_from_cv(
        normal,
        **_reference_pair_options(),
    )
    inverted_midpoint, inverted_details = ecat_module.find_reference_midpoint_from_cv(
        inverted,
        **_reference_pair_options(),
    )

    for key in ("Epa", "Epc", "midpoint", "delta_ep"):
        assert inverted_details[key] == pytest.approx(normal_details[key], abs=1e-12)
    assert inverted_midpoint == pytest.approx(normal_midpoint, abs=1e-12)
    assert inverted_details["anodic_segment"] == normal_details["anodic_segment"]
    assert inverted_details["cathodic_segment"] == normal_details["cathodic_segment"]
    assert inverted_details["anodic_extremum_kind"] != normal_details["anodic_extremum_kind"]


def test_reference_pairing_excludes_partial_segment_without_reference_guess(
    ecat_module,
    cv_factory,
):
    obj = _lje_reference_regression_cv(cv_factory)

    _midpoint, details = ecat_module.find_reference_midpoint_from_cv(
        obj,
        **_reference_pair_options(),
    )

    assert details["anodic_segment"] == 2
    assert details["cathodic_segment"] == 3
    assert details["rejected_pair_counts"]["guess not sampled"] >= 1


def test_reference_pairing_never_combines_nonadjacent_segments(ecat_module, cv_factory):
    def first(E):
        return -5e-6 * np.exp(-0.5 * ((E - 0.10) / 0.02) ** 2)

    def middle(E):
        return np.zeros_like(E)

    def last(E):
        return 5e-6 * np.exp(-0.5 * ((E - 0.16) / 0.02) ** 2)

    obj = _segmented_reference_cv(
        cv_factory,
        [(-0.3, 0.3, 241, first), (0.3, -0.3, 241, middle), (-0.3, 0.3, 241, last)],
    )

    with pytest.raises(ValueError, match="adjacent.*segments|opposite-extremum"):
        ecat_module.find_reference_midpoint_from_cv(
            obj,
            **_reference_pair_options(guess=0.13, window=0.2),
        )


def _two_cycle_reference_cv(cv_factory, *, first_delta=0.10, second_delta=0.06):
    centers = [
        (0.07 + first_delta / 2, 0.07 - first_delta / 2),
        (-0.13 + second_delta / 2, -0.13 - second_delta / 2),
    ]
    segments = []
    for Epa, Epc in centers:
        segments.extend(
            [
                (-0.30, 0.30, 241, lambda E, Epa=Epa: 5e-6 * np.exp(-0.5 * ((E - Epa) / 0.02) ** 2)),
                (0.30, -0.30, 241, lambda E, Epc=Epc: -5e-6 * np.exp(-0.5 * ((E - Epc) / 0.02) ** 2)),
            ]
        )
    return _segmented_reference_cv(cv_factory, segments)


def test_reference_pairing_ranks_all_cycles_and_keeps_each_pair_adjacent(
    ecat_module,
    cv_factory,
):
    obj = _two_cycle_reference_cv(cv_factory)

    midpoint, details = ecat_module.find_reference_midpoint_from_cv(
        obj,
        **_reference_pair_options(guess="auto", window=0.3, target_delta_ep=0.06),
    )

    assert midpoint == pytest.approx(-0.13, abs=0.004)
    assert {details["anodic_segment"], details["cathodic_segment"]} == {3, 4}


def test_reference_pairing_rejects_epa_not_greater_than_epc(ecat_module, cv_factory):
    obj = _segmented_reference_cv(
        cv_factory,
        [
            (-0.3, 0.3, 241, lambda E: 5e-6 * np.exp(-0.5 * ((E - 0.05) / 0.02) ** 2)),
            (0.3, -0.3, 241, lambda E: -5e-6 * np.exp(-0.5 * ((E - 0.12) / 0.02) ** 2)),
        ],
    )

    with pytest.raises(ValueError, match="Epa.*greater than Epc"):
        ecat_module.find_reference_midpoint_from_cv(
            obj,
            **_reference_pair_options(guess=0.08, window=0.2),
        )


def test_reference_pairing_rejects_same_kind_extrema(ecat_module, cv_factory):
    obj = _segmented_reference_cv(
        cv_factory,
        [
            (-0.3, 0.3, 241, lambda E: 5e-6 * np.exp(-0.5 * ((E - 0.10) / 0.02) ** 2)),
            (0.3, -0.3, 241, lambda E: 4e-6 * np.exp(-0.5 * ((E - 0.04) / 0.02) ** 2)),
        ],
    )

    with pytest.raises(ValueError, match="same extremum kind"):
        ecat_module.find_reference_midpoint_from_cv(
            obj,
            **_reference_pair_options(guess=0.07, window=0.2),
        )


def test_reference_pairing_auto_raises_for_indistinguishable_couples(
    ecat_module,
    cv_factory,
):
    obj = _two_cycle_reference_cv(cv_factory, first_delta=0.06, second_delta=0.06)

    with pytest.raises(ValueError, match="ambiguous.*reference couples"):
        ecat_module.find_reference_midpoint_from_cv(
            obj,
            **_reference_pair_options(guess="auto", window=0.3, target_delta_ep=0.06),
        )

    midpoint, _details = ecat_module.find_reference_midpoint_from_cv(
        obj,
        **_reference_pair_options(guess=0.07, window=0.20, target_delta_ep=0.06),
    )
    assert midpoint == pytest.approx(0.07, abs=0.004)


def test_reference_pairing_uses_raw_potential_axis(ecat_module, cv_factory):
    obj = _lje_reference_regression_cv(cv_factory)
    obj.potential_shift({"reference offset": 0.25, "reference label": "Fc/Fc+"})

    midpoint, _details = ecat_module.find_reference_midpoint_from_cv(
        obj,
        **_reference_pair_options(),
    )

    assert midpoint == pytest.approx(0.4995, abs=1e-3)


def test_reference_troubleshoot_prints_tables_and_plots_segments(
    ecat_module,
    cv_factory,
    capsys,
):
    obj = _lje_reference_regression_cv(cv_factory)

    ecat_module.find_reference_midpoint_from_cv(
        obj,
        **_reference_pair_options(troubleshoot=True),
    )

    output = capsys.readouterr().out
    assert "Reference Segment Summary:" in output
    assert "Reference Candidate Summary:" in output
    assert "Reference Pair Selected:" in output
    assert plt.get_fignums()
    figure = plt.gcf()
    assert len(figure.axes) == 1
    axis = figure.axes[0]
    assert len(axis.lines) == 3
    assert axis.get_xlabel() == "Potential (V)"
    assert axis.get_ylabel() == "Current (A)"
    assert axis.get_title() == "Reference Pair Diagnostic"
    assert len(axis.collections) == 1


def test_reference_troubleshoot_handles_segment_outside_search_window(
    ecat_module,
    cv_factory,
    capsys,
):
    obj = _lje_reference_regression_cv(cv_factory)

    midpoint, details = ecat_module.find_reference_midpoint_from_cv(
        obj,
        **_reference_pair_options(
            guess=0.5,
            window=0.1,
            prominence=None,
            troubleshoot=True,
        ),
    )

    output = capsys.readouterr().out
    assert midpoint == pytest.approx(0.4995, abs=1e-3)
    assert details["anodic_segment"] == 2
    assert details["cathodic_segment"] == 3
    assert "not estimated" in output


def test_reference_troubleshoot_reports_failed_selection(
    ecat_module,
    cv_factory,
    capsys,
):
    obj = _segmented_reference_cv(
        cv_factory,
        [
            (-0.3, 0.3, 241, lambda E: 5e-6 * np.exp(-0.5 * ((E - 0.10) / 0.02) ** 2)),
            (0.3, -0.3, 241, lambda E: 4e-6 * np.exp(-0.5 * ((E - 0.04) / 0.02) ** 2)),
        ],
    )

    with pytest.raises(ValueError):
        ecat_module.find_reference_midpoint_from_cv(
            obj,
            **_reference_pair_options(guess=0.07, window=0.2, troubleshoot=True),
        )

    output = capsys.readouterr().out
    assert "Reference Pair Rejections:" in output
    assert "Reference Pair Selection:" in output
    assert "No valid adjacent" in output
    assert plt.get_fignums()


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
    assert obj.reference_pair_details is None
    assert obj.x().name == "Potential vs Fc/Fc+"


def test_get_data_retains_selected_reference_pair_provenance(ecat_module, tmp_path):
    _write_ch_cv(tmp_path / "Fc_reference.txt", REFERENCE_ROWS)
    _write_ch_cv(tmp_path / "sample_cv.txt", SAMPLE_ROWS)

    objects = ecat_module.get_data(
        {
            "folder path": str(tmp_path),
            "recursive search": False,
            "print": False,
            "reference mode": "keyword",
            "reference keyword": "Fc",
            "reference guess": 0.0,
            "peak prominence": 1e-6,
            "reference smooth": False,
        }
    )

    sample = next(obj for obj in objects if obj.filepath.endswith("sample_cv.txt"))
    details = sample.reference_pair_details
    assert details["Epa"] == pytest.approx(0.10)
    assert details["Epc"] == pytest.approx(-0.10)
    assert details["delta_ep"] == pytest.approx(0.20)
    assert details["anodic_segment"] == 1
    assert details["cathodic_segment"] == 2
    assert details["selection_mode"] == "ranked adjacent segments"
    assert sample.parse_result.metadata["reference_pair_details"] == details


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

    pair_details = {
        "Epa": 0.252,
        "Epc": 0.192,
        "delta_ep": 0.060,
        "anodic_segment": 2,
        "cathodic_segment": 3,
        "selection_mode": "ranked adjacent segments",
    }

    def fake_midpoint(ref_cv, **kwargs):
        calls.append(Path(ref_cv.filepath).name)
        return 0.222, pair_details

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
    assert sample.reference_pair_details == pair_details
    assert sample.parse_result.metadata["reference_pair_details"] == pair_details
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

    auto_details = {"selection_mode": "auto reference"}
    mapped_details = {
        "Epa": 0.486,
        "Epc": 0.426,
        "delta_ep": 0.060,
        "anodic_segment": 2,
        "cathodic_segment": 3,
        "selection_mode": "ranked adjacent segments",
    }

    def fake_midpoint(ref_cv, **kwargs):
        filename = Path(ref_cv.filepath).name
        if filename == "auto_Fc.txt":
            return 0.111, auto_details
        if filename == "manual_Fc.txt":
            return 0.456, mapped_details
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
    assert objects[0].reference_pair_details == mapped_details
    assert objects[0].parse_result.metadata["reference_pair_details"] == mapped_details


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
