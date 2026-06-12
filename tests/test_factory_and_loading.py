from pathlib import Path

import pytest


@pytest.mark.parametrize(
    ("filename", "expected_software", "expected_type", "expected_cls_name"),
    [
        ("ch_cv.txt", "CH", "Cyclic Voltammetry", "cv"),
        ("basi_cv.txt", "BASI", "Cyclic Voltammetry", "cv"),
        ("eclab_cv.txt", "EC-Lab", "Cyclic Voltammetry", "cv"),
        ("generic_unknown.txt", None, "Unknown", "echem"),
    ],
)
def test_factory_detects_software_and_promotes_expected_class(
    ecat_module,
    fixtures_dir,
    filename,
    expected_software,
    expected_type,
    expected_cls_name,
):
    filepath = fixtures_dir / filename

    assert ecat_module.echem.detect_software(str(filepath), {}) == expected_software
    assert ecat_module.echem.detect_experiment_type(str(filepath), {}) == expected_type

    obj = ecat_module.echem.from_file(str(filepath), {})

    assert type(obj).__name__ == expected_cls_name
    assert len(obj.data) >= 4


def test_factory_loading_sets_common_cv_metadata(ecat_module, fixtures_dir):
    obj = ecat_module.echem.from_file(str(fixtures_dir / "ch_cv.txt"), {})

    assert obj.type == "Cyclic Voltammetry"
    assert obj.units["Potential"] == "V"
    assert obj.scan_rate == pytest.approx(0.05)
    assert obj.segments == 2


@pytest.mark.parametrize(
    ("options", "expected_area"),
    [
        ({"electrode diameter": 0.3}, pytest.approx(0.0706858347)),
        ({"electrode diameter": 0.3, "electrode area": None}, pytest.approx(0.0706858347)),
        ({"electrode diameter": 0.3, "electrode area": 0}, 0),
    ],
)
def test_from_file_resolves_electrode_area_from_diameter_when_area_is_omitted_or_none(
    ecat_module,
    fixtures_dir,
    options,
    expected_area,
):
    obj = ecat_module.echem.from_file(str(fixtures_dir / "ch_cv.txt"), options)

    assert obj.electrode_area == expected_area


def test_from_file_print_option_delegates_to_show(ecat_module, fixtures_dir, monkeypatch):
    shown = {}

    def fake_show(obj, options=None):
        shown["type"] = type(obj).__name__
        shown["options"] = options
        return "shown"

    monkeypatch.setattr(ecat_module.plotting, "show", fake_show)

    obj = ecat_module.echem.from_file(
        str(fixtures_dir / "ch_ca_tiny.txt"),
        {"print": True},
    )

    assert type(obj).__name__ == "ca"
    assert shown == {"type": "ca", "options": {"print": True}}
    assert obj.units["Current"] == "A"


def test_get_data_loads_selected_fixture_txt_files(ecat_module, fixtures_dir, tmp_path):
    selected = ["basi_cv.txt", "ch_cv.txt", "eclab_cv.txt", "generic_unknown.txt"]
    for name in selected:
        (tmp_path / name).write_text(
            (fixtures_dir / name).read_text(encoding="ISO-8859-1"),
            encoding="ISO-8859-1",
        )

    objects = ecat_module.get_data(
        {
            "folder path": str(tmp_path),
            "recursive search": False,
            "print": False,
            "shift potential": False,
        }
    )

    loaded_names = {Path(obj.filepath).name for obj in objects}

    assert len(objects) == 4
    assert loaded_names == set(selected)


def test_get_data_derives_electrode_area_from_diameter_when_area_is_omitted(
    ecat_module,
    fixtures_dir,
    tmp_path,
):
    (tmp_path / "ch_cv.txt").write_text(
        (fixtures_dir / "ch_cv.txt").read_text(encoding="ISO-8859-1"),
        encoding="ISO-8859-1",
    )

    objects = ecat_module.get_data(
        {
            "folder path": str(tmp_path),
            "recursive search": False,
            "print": False,
            "reference mode": "none",
            "electrode diameter": 0.3,
        }
    )

    assert len(objects) == 1
    assert objects[0].electrode_area == pytest.approx(0.0706858347)


def test_get_data_root_files_store_empty_folderpath(
    ecat_module,
    fixtures_dir,
    repo_root,
    tmp_path,
):
    sources = {
        "ch_cv.txt": fixtures_dir / "ch_cv.txt",
        "ch_ca_tiny.txt": fixtures_dir / "ch_ca_tiny.txt",
        "ch_cp_tiny.txt": fixtures_dir / "ch_cp_tiny.txt",
        "ch_dpv.txt": repo_root
        / "tests"
        / "tmp_real_examples"
        / "DPV_MeCN_CO2_0.1MTBAPF6_3mMFc_1mMFe-tpyPY2Me_-0.7_to_-1.2V.txt",
    }
    for name, source in sources.items():
        (tmp_path / name).write_text(
            source.read_text(encoding="ISO-8859-1"),
            encoding="ISO-8859-1",
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

    assert {Path(obj.filepath).name for obj in objects} == set(sources)
    assert {getattr(obj, "type", None) for obj in objects} >= {
        "Cyclic Voltammetry",
        "Differential Pulse Voltammetry",
        "Chronoamperometry",
        "Chronopotentiometry",
    }
    assert all(obj.folderpath == "" for obj in objects)


@pytest.mark.parametrize("fixture_name", ["ch_ca_tiny.txt", "ch_cp_tiny.txt"])
def test_get_data_print_true_uses_only_folder_summary(
    ecat_module,
    fixtures_dir,
    tmp_path,
    monkeypatch,
    fixture_name,
):
    (tmp_path / fixture_name).write_text(
        (fixtures_dir / fixture_name).read_text(encoding="ISO-8859-1"),
        encoding="ISO-8859-1",
    )
    calls = {"show": 0, "show_objects": 0}

    def fake_show(*args, **kwargs):
        calls["show"] += 1

    def fake_show_objects(objects, options=None):
        calls["show_objects"] += 1
        return objects

    monkeypatch.setattr(ecat_module.plotting, "show", fake_show)
    monkeypatch.setattr(ecat_module.io, "show_objects", fake_show_objects)

    loaded = ecat_module.get_data(
        {
            "folder path": str(tmp_path),
            "recursive search": False,
            "print": True,
            "reference mode": "none",
        }
    )

    assert len(loaded) == 1
    assert calls == {"show": 0, "show_objects": 1}


def test_get_data_warns_and_skips_files_that_cannot_be_converted(
    ecat_module,
    fixtures_dir,
    tmp_path,
    capsys,
    monkeypatch,
):
    good_file = tmp_path / "good_ch_cv.txt"
    bad_file = tmp_path / "bad_export.txt"
    good_file.write_text(
        (fixtures_dir / "ch_cv.txt").read_text(encoding="ISO-8859-1"),
        encoding="ISO-8859-1",
    )
    bad_file.write_text("not an electrochemistry export\n", encoding="ISO-8859-1")

    original_from_file = ecat_module.echem.from_file

    def fake_from_file(filepath, options=None):
        if Path(filepath).name == "bad_export.txt":
            raise ValueError("unsupported parser shape")
        return original_from_file(filepath, options)

    monkeypatch.setattr(ecat_module.echem, "from_file", fake_from_file)
    objects = ecat_module.get_data(
        {
            "folder path": str(tmp_path),
            "recursive search": False,
            "print": False,
            "reference mode": "none",
            "sort keys": ["name"],
        }
    )

    captured = capsys.readouterr()

    assert [Path(obj.filepath).name for obj in objects] == ["good_ch_cv.txt"]
    assert "Warning: could not convert bad_export.txt" in captured.out
    assert "unsupported parser shape" in captured.out


def test_get_data_returns_empty_list_when_no_files_convert(
    ecat_module,
    tmp_path,
    capsys,
    monkeypatch,
):
    bad_file = tmp_path / "bad_export.txt"
    bad_file.write_text("not an electrochemistry export\n", encoding="ISO-8859-1")

    def fake_from_file(filepath, options=None):
        raise ValueError("unsupported parser shape")

    monkeypatch.setattr(ecat_module.echem, "from_file", fake_from_file)

    objects = ecat_module.get_data(
        {
            "folder path": str(tmp_path),
            "recursive search": False,
            "print": False,
            "reference mode": "none",
        }
    )

    captured = capsys.readouterr()

    assert objects == []
    assert "Warning: could not convert bad_export.txt" in captured.out
    assert "No .txt files could be converted into eCAT objects." in captured.out
