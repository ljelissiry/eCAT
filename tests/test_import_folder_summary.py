"""Behavioral checks for get_data's direct-folder import accounting."""

import pytest


def _write_cv(path):
    path.write_text(
        "Aug. 27, 2023   16:03:21\nCyclic Voltammetry\n"
        "Instrument Model: CHI760E\nScan Rate = 0.05\n"
        "Potential/V,Current/A\n-0.10,-1e-7\n0.00,0\n0.10,1e-7\n",
        encoding="utf-8",
    )


@pytest.fixture
def captured_tables(ecat_module, monkeypatch):
    from ecat import io

    tables = []
    monkeypatch.setattr(io, "_display_table", lambda table, *args, **kwargs: tables.append(table.copy()), raising=False)
    monkeypatch.setattr(io, "show_objects", lambda *args, **kwargs: None)
    return tables


def _load(root, **options):
    from ecat import io

    return io.get_data({"folder path": str(root), "reference mode": "none", **options})


def test_recursive_summary_includes_every_folder_and_direct_counts(tmp_path, captured_tables, capsys):
    (tmp_path / "empty").mkdir()
    (tmp_path / "unrelated").mkdir()
    (tmp_path / "nested" / "deep").mkdir(parents=True)
    _write_cv(tmp_path / "root.txt")
    _write_cv(tmp_path / "nested" / "deep" / "sample.txt")
    (tmp_path / "nested" / "broken.txt").write_text("invalid table\n")
    (tmp_path / "unrelated" / "notes.pdf").write_text("notes")
    (tmp_path / "root.mpr").write_bytes(b"binary")

    objects = _load(tmp_path)

    assert len(objects) == 2
    assert len(captured_tables) == 1
    table = captured_tables[0].set_index("Folder")
    assert list(table.index) == [".", "empty", "nested", "nested/deep", "unrelated"]
    assert table.to_dict("index") == {
        ".": {"Candidates": 1, "Loaded": 1, "Failed": 0, "Unsupported": 1, "References used": 0},
        "empty": {"Candidates": 0, "Loaded": 0, "Failed": 0, "Unsupported": 0, "References used": 0},
        "nested": {"Candidates": 1, "Loaded": 0, "Failed": 1, "Unsupported": 0, "References used": 0},
        "nested/deep": {"Candidates": 1, "Loaded": 1, "Failed": 0, "Unsupported": 0, "References used": 0},
        "unrelated": {"Candidates": 0, "Loaded": 0, "Failed": 0, "Unsupported": 1, "References used": 0},
    }
    output = capsys.readouterr().out
    assert "Candidates: 3; Loaded: 2; Failed: 1; Unsupported: 2; References used: 0" in output
    assert "nested/broken.txt" in output
    assert "ValueError" in output


@pytest.mark.parametrize("kind", ["empty", "unsupported", "failed"])
def test_early_returns_still_report_all_visited_folders(tmp_path, captured_tables, capsys, kind):
    sub = tmp_path / "sub"
    sub.mkdir()
    if kind == "unsupported":
        (sub / "notes.pdf").write_text("notes")
    elif kind == "failed":
        (sub / "broken.txt").write_text("invalid")

    assert _load(tmp_path) == []
    table = captured_tables[0].set_index("Folder")
    assert list(table.index) == [".", "sub"]
    assert table.loc["sub", "Failed"] == int(kind == "failed")
    assert table.loc["sub", "Unsupported"] == int(kind == "unsupported")
    assert "Loaded: 0" in capsys.readouterr().out


@pytest.mark.parametrize("recursive", [True, False])
def test_no_subfolders_only_prints_totals(tmp_path, captured_tables, capsys, recursive):
    _write_cv(tmp_path / "sample.txt")
    assert len(_load(tmp_path, **{"recursive search": recursive})) == 1
    assert captured_tables == []
    output = capsys.readouterr().out
    assert "Candidates: 1; Loaded: 1; Failed: 0; Unsupported: 0; References used: 0" in output
    assert "supported text file found" not in output


@pytest.mark.parametrize("troubleshoot", [False, True])
def test_print_false_is_quiet_unless_troubleshooting(tmp_path, captured_tables, capsys, troubleshoot):
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "bad.txt").write_text("invalid")
    assert _load(tmp_path, print=False, troubleshoot=troubleshoot) == []
    output = capsys.readouterr().out
    assert bool(output) == troubleshoot
    assert bool(captured_tables) == troubleshoot


def test_unsupported_filename_and_extension_previews_are_bounded(tmp_path, captured_tables, capsys):
    for index in range(12):
        (tmp_path / f"binary{index:02d}.mpr").write_bytes(b"binary")
        (tmp_path / f"notes{index:02d}.extension{index:02d}").write_text("notes")

    assert _load(tmp_path) == []
    output = capsys.readouterr().out
    assert "Unsupported binary files skipped (12):" in output
    assert "binary04.mpr" in output
    assert "binary05.mpr" not in output
    assert "7 more" in output
    assert "extension11" not in output
    assert "more extension" in output
    assert "Export EC-Lab ASCII `.mpt`" in output
    assert "Unsupported: 24" in output


@pytest.mark.parametrize("external", [False, True])
def test_reference_sources_count_once_without_phantom_folders(tmp_path, captured_tables, monkeypatch, capsys, external):
    from ecat import io

    root = tmp_path / "root"
    (root / "sub").mkdir(parents=True)
    _write_cv(root / "sample.txt")
    _write_cv(root / "sub" / "sample.txt")
    reference = (tmp_path if external else root) / "reference.txt"
    _write_cv(reference)
    monkeypatch.setattr(io, "_compute_explicit_reference_file_shift", lambda **kwargs: (str(reference), 0.123, {}))

    objects = _load(root, **{"reference mode": "file", "reference file": str(reference)})

    assert len(objects) == (2 if external else 3)
    assert all(obj.reference_source_file == str(reference) for obj in objects)
    table = captured_tables[0].set_index("Folder")
    assert list(table.index) == [".", "sub"]
    assert table["References used"].sum() == int(not external)
    assert table.loc["sub", "References used"] == 0
    output = capsys.readouterr().out
    assert "References used: 1" in output
    assert ("External reference sources: 1" in output) == external


def test_nonrecursive_search_excludes_existing_subfolders(tmp_path, captured_tables, capsys):
    (tmp_path / "sub").mkdir()
    _write_cv(tmp_path / "sample.txt")
    _write_cv(tmp_path / "sub" / "other.txt")

    assert len(_load(tmp_path, **{"recursive search": False})) == 1
    assert captured_tables == []
    output = capsys.readouterr().out
    assert "Candidates: 1; Loaded: 1" in output
    assert "supported text file found" not in output


def test_reference_aliases_count_one_physical_source(tmp_path, captured_tables, monkeypatch, capsys):
    from ecat import io

    (tmp_path / "sub").mkdir()
    reference = tmp_path / "reference.txt"
    alias = tmp_path / "alias.txt"
    _write_cv(reference)
    alias.symlink_to(reference)
    _write_cv(tmp_path / "sub" / "sample.txt")

    def apply_references(*, reference_records, **kwargs):
        for index, record in enumerate(reference_records):
            record["ref_file"] = str(reference if index % 2 else alias)
            record["shift"] = 0.123

    monkeypatch.setattr(io, "_apply_reference_map", apply_references)
    assert len(_load(tmp_path)) == 3
    table = captured_tables[0].set_index("Folder")
    assert table["References used"].sum() == 1
    assert "References used: 1" in capsys.readouterr().out


def test_plain_summary_renders_every_visited_folder(ecat_module, tmp_path, capsys):
    for index in range(65):
        (tmp_path / f"empty{index:02d}").mkdir()

    assert _load(tmp_path, **{"pretty print": False}) == []
    output = capsys.readouterr().out
    assert "Import by folder (direct counts)" in output
    for index in range(65):
        assert output.count(f"empty{index:02d}") == 1


@pytest.mark.parametrize("unsupported", [False, True])
def test_no_candidates_reports_search_root_only_once(ecat_module, tmp_path, capsys, unsupported):
    if unsupported:
        (tmp_path / "notes.pdf").write_text("notes")
    assert _load(tmp_path) == []
    assert capsys.readouterr().out.count(tmp_path.as_posix()) == 1


def test_reference_import_preserves_all_display_columns(ecat_module, tmp_path, monkeypatch):
    import pandas as pd
    from ecat import io, plotting

    _write_cv(tmp_path / "sample.txt")
    actual_tables = []

    def build_display_table(objects, options):
        actual_tables.append(plotting.build_object_table(objects, options))

    monkeypatch.setattr(io, "show_objects", build_display_table)
    objects = _load(
        tmp_path,
        **{"reference mode": "manual", "reference offset": 0.12, "display columns": "all"},
    )
    expected_table, expected_metadata = plotting.build_object_table(objects, {"display columns": "all"})
    actual_table, actual_metadata = actual_tables[0]
    pd.testing.assert_frame_equal(actual_table, expected_table)
    assert actual_metadata == expected_metadata


@pytest.mark.parametrize("print_output", [True, False])
def test_failed_auto_reference_search_still_returns_imports(ecat_module, tmp_path, monkeypatch, capsys, print_output):
    from ecat import io

    _write_cv(tmp_path / "sample.txt")

    def unresolved_reference(**kwargs):
        raise ValueError("No reference peak was resolved")

    monkeypatch.setattr(io, "_resolve_reference_shifts", unresolved_reference)
    objects = _load(
        tmp_path,
        **{"reference mode": "auto", "print": print_output, "pretty print": False},
    )
    assert len(objects) == 1
    assert objects[0].reference_shift is None
    assert objects[0].reference_source_file is None
    output = capsys.readouterr().out
    if print_output:
        assert "Loaded: 1" in output
        assert "sample" in output
    else:
        assert output == ""
