from pathlib import Path


def _import_example_cv_with_absolute_paths(ecat_module, repo_root):
    data_dir = repo_root / "examples" / "data" / "fe_phoh_cv"
    filepath = data_dir / "MeCN_Ar_0.1MTBAPF6_-1.2_to_1V_100mVs.txt"
    obj = ecat_module.echem.from_file(
        filepath,
        {"print": False, "reference mode": "none"},
    )
    obj.folderpath = str(data_dir)
    return obj, filepath, data_dir


def test_format_path_for_display_uses_relative_path_under_base(ecat_module, tmp_path):
    base = tmp_path / "project"
    nested = base / "notebooks" / "_outputs" / "result.csv"
    nested.parent.mkdir(parents=True)
    nested.write_text("", encoding="utf-8")

    display_path = ecat_module._format_path_for_display(nested, relative_to=base)

    assert display_path == "notebooks/_outputs/result.csv"
    assert str(tmp_path) not in display_path


def test_format_path_for_display_normalizes_windows_separators(ecat_module, tmp_path, monkeypatch):
    base = tmp_path / "project"
    nested = base / "notebooks" / "_outputs" / "result.csv"
    nested.parent.mkdir(parents=True)
    nested.write_text("", encoding="utf-8")
    monkeypatch.setattr(
        ecat_module.os.path,
        "relpath",
        lambda path, start: r"notebooks\_outputs\result.csv",
    )

    display_path = ecat_module._format_path_for_display(nested, relative_to=base)

    assert display_path == "notebooks/_outputs/result.csv"


def test_get_data_prints_search_folder_relative_to_current_directory(
    ecat_module,
    tmp_path,
    monkeypatch,
    capsys,
):
    project = tmp_path / "project"
    data_dir = project / "examples" / "data"
    data_dir.mkdir(parents=True)
    monkeypatch.chdir(project)

    result = ecat_module.get_data({"folder path": str(data_dir)})

    output = capsys.readouterr().out
    assert result == []
    assert "examples/data" in output
    assert str(tmp_path) not in output


def test_get_data_prints_search_folder_relative_to_project_root_from_notebooks(
    ecat_module,
    tmp_path,
    monkeypatch,
    capsys,
):
    project = tmp_path / "project"
    data_dir = project / "examples" / "data"
    notebook_dir = project / "notebooks"
    data_dir.mkdir(parents=True)
    notebook_dir.mkdir()
    (project / "pyproject.toml").write_text("[project]\nname = 'demo'\n", encoding="utf-8")
    monkeypatch.chdir(notebook_dir)

    result = ecat_module.get_data({"folder path": str(data_dir)})

    output = capsys.readouterr().out
    assert result == []
    assert "examples/data" in output
    assert str(tmp_path) not in output


def test_save_data_prints_export_path_relative_to_current_directory(
    ecat_module,
    cv_factory,
    tmp_path,
    monkeypatch,
    capsys,
):
    project = tmp_path / "project"
    export_dir = project / "notebooks" / "_outputs"
    export_dir.mkdir(parents=True)
    monkeypatch.chdir(project)

    ecat_module.save_data(
        [cv_factory()],
        {"folder path": str(export_dir), "file name": "processed_cv"},
    )

    output = capsys.readouterr().out
    assert "notebooks/_outputs/processed_cv.csv" in output
    assert str(tmp_path) not in output


def test_show_rich_table_formats_object_paths_without_mutating_metadata(
    ecat_module,
    repo_root,
    monkeypatch,
):
    monkeypatch.chdir(repo_root)
    obj, filepath, data_dir = _import_example_cv_with_absolute_paths(
        ecat_module,
        repo_root,
    )
    metadata_before = obj.metadata()

    table = ecat_module.show(obj, {"pretty print": True, "return": True})
    values = dict(zip(table["Metric"], table["Value"]))

    assert values["File Path"] == filepath.relative_to(repo_root).as_posix()
    assert values["Folder Path"] == data_dir.relative_to(repo_root).as_posix()
    assert str(repo_root) not in values["File Path"]
    assert str(repo_root) not in values["Folder Path"]
    assert obj.metadata() == metadata_before
    assert obj.metadata()["file path"] == str(filepath)
    assert obj.metadata()["folder path"] == str(data_dir)


def test_show_plain_output_formats_object_paths(
    ecat_module,
    repo_root,
    monkeypatch,
    capsys,
):
    monkeypatch.chdir(repo_root)
    obj, filepath, data_dir = _import_example_cv_with_absolute_paths(
        ecat_module,
        repo_root,
    )

    table = ecat_module.show(
        obj,
        {"pretty print": False, "return": True},
    )

    output = capsys.readouterr().out
    relative_file = filepath.relative_to(repo_root).as_posix()
    relative_folder = data_dir.relative_to(repo_root).as_posix()
    values = dict(zip(table["Metric"], table["Value"]))
    assert values["File Path"] == relative_file
    assert values["Folder Path"] == relative_folder
    assert relative_file in output
    assert relative_folder in output
    assert str(repo_root) not in output
