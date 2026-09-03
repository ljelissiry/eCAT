from pathlib import Path


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
