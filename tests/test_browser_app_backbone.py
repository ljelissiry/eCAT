import base64
import sys
from pathlib import Path

import pytest


APP_SRC = Path(__file__).resolve().parents[1] / "apps" / "browser" / "src"
if str(APP_SRC) not in sys.path:
    sys.path.insert(0, str(APP_SRC))


def _copy_fixture(fixtures_dir, tmp_path, filename):
    destination = tmp_path / filename
    destination.write_text(
        (fixtures_dir / filename).read_text(encoding="ISO-8859-1"),
        encoding="ISO-8859-1",
    )
    return destination


def _dash_upload_contents(path):
    raw = path.read_bytes()
    return "data:text/plain;base64," + base64.b64encode(raw).decode("ascii")


def test_browser_local_path_loading_summarizes_supported_objects(fixtures_dir, tmp_path):
    from ecat_browser.adapters import load_local_path, summarize_objects

    _copy_fixture(fixtures_dir, tmp_path, "ch_cv.txt")
    _copy_fixture(fixtures_dir, tmp_path, "ch_cp_tiny.txt")

    result = load_local_path(tmp_path, recursive=False)
    summary = summarize_objects(result.objects)

    assert result.warnings == []
    by_filename = {row["filename"]: row for row in summary}
    assert by_filename["ch_cv.txt"]["class"] == "cv"
    assert by_filename["ch_cv.txt"]["software"] == "CH"
    assert by_filename["ch_cp_tiny.txt"]["class"] == "cp"
    assert by_filename["ch_cp_tiny.txt"]["type"] == "Chronopotentiometry"


def test_browser_upload_loading_preserves_filenames_and_reports_failures(fixtures_dir, tmp_path):
    from ecat_browser.adapters import load_uploaded_files, summarize_objects

    good = _copy_fixture(fixtures_dir, tmp_path, "ch_cv.txt")
    uploads = [
        {"filename": "uploaded_cv.txt", "contents": _dash_upload_contents(good)},
        {
            "filename": "bad_export.txt",
            "contents": "data:text/plain;base64,"
            + base64.b64encode(b"not an electrochemistry export\n").decode("ascii"),
        },
    ]

    result = load_uploaded_files(uploads, session_root=tmp_path / "session")
    summary = summarize_objects(result.objects)

    assert [row["filename"] for row in summary] == ["uploaded_cv.txt"]
    assert "bad_export.txt" in result.warnings[0]
    assert Path(result.objects[0].filepath).name == "uploaded_cv.txt"


def test_browser_upload_reload_uses_existing_session_folder(fixtures_dir, tmp_path):
    from ecat_browser.adapters import load_uploaded_files, reload_workflow

    good = _copy_fixture(fixtures_dir, tmp_path, "ch_cv.txt")
    result = load_uploaded_files(
        [{"filename": "uploaded_cv.txt", "contents": _dash_upload_contents(good)}],
        session_root=tmp_path / "session",
    )

    reloaded = reload_workflow(result.workflow)

    assert result.workflow.source_kind == "upload"
    assert Path(result.workflow.source_path).is_dir()
    assert [Path(obj.filepath).name for obj in reloaded.objects] == ["uploaded_cv.txt"]


def test_browser_uploaded_files_preserve_ecat_load_order(monkeypatch, cv_factory, tmp_path):
    from ecat_browser import adapters

    loaded = [
        cv_factory(name="100mVs_z_later"),
        cv_factory(name="100mVs_a_earlier"),
    ]

    def fake_load_one_file(path, import_options=None):
        return loaded.pop(0), None

    monkeypatch.setattr(adapters, "_load_one_file", fake_load_one_file)

    result = adapters.load_uploaded_files(
        [
            {"filename": "z.txt", "contents": "c2FtcGxl"},
            {"filename": "a.txt", "contents": "c2FtcGxl"},
        ],
        session_root=tmp_path,
    )

    assert [obj.name for obj in result.objects] == ["100mVs_z_later", "100mVs_a_earlier"]


def test_browser_single_cv_analysis_skips_non_cv_objects(ecat_module, fixtures_dir):
    from ecat_browser.adapters import load_local_path, run_single_cv_analysis

    cp_obj = load_local_path(fixtures_dir / "ch_cp_tiny.txt").objects[0]

    result = run_single_cv_analysis(cp_obj)

    assert result["status"] == "skipped"
    assert "CV" in result["message"]


def test_browser_single_cv_analysis_supports_half_metrics(cv_factory):
    from ecat_browser.adapters import run_single_cv_analysis

    obj = cv_factory(name="100mVs_sample_CO2_MeCN_10mM_Fc_run01")

    result = run_single_cv_analysis(
        obj,
        ["peak_potential", "peak_current", "half_peak_potential", "half_wave_potential"],
    )

    names = [row["analysis"] for row in result["results"]]
    assert names == [
        "peak_potential",
        "peak_current",
        "half_peak_potential",
        "half_wave_potential",
    ]
    assert all(row["status"] in {"ok", "error"} for row in result["results"])


def test_browser_single_cv_analysis_requests_plot_all_true():
    from ecat_browser.adapters import run_single_cv_analysis
    import matplotlib.pyplot as plt

    class cv:
        def __init__(self):
            self.options = None
            self.calls = []

        def plot(self, options):
            self.calls.append(("plot", dict(options)))
            plt.figure()
            plt.plot([0, 1], [1, 0])

        def peak_potential(self, options):
            self.calls.append(("peak_potential", dict(options)))
            self.options = dict(options)
            plt.plot([0, 1], [0, 1])
            return {"Ep": 0.1}

    obj = cv()

    run_single_cv_analysis(obj, ["peak_potential"])

    assert [name for name, _options in obj.calls] == ["plot", "peak_potential"]
    assert obj.calls[0][1]["new plot"] is True
    assert obj.options["plot all"] is True
    assert obj.options["plot CV"] is False
    assert obj.options["plot"] is True
    assert obj.options["new plot"] is False
    assert obj.options["print"] is False


def test_browser_single_cv_analysis_passes_guess_and_tangent_potentials():
    from ecat_browser.adapters import run_single_cv_analysis
    import matplotlib.pyplot as plt

    class cv:
        def __init__(self):
            self.options = None

        def plot(self, options):
            plt.figure()
            plt.plot([0, 1], [1, 0])

        def peak_current(self, options):
            self.options = dict(options)
            plt.plot([0, 1], [0, 1])
            return {"ip": 1e-6}

    obj = cv()

    run_single_cv_analysis(obj, ["peak_current"], {"guess potential": -1.2, "tangent potential": -0.8})

    assert obj.options["guess potential"] == -1.2
    assert obj.options["tangent potential"] == -0.8


def test_browser_single_cv_analysis_returns_plot_image():
    from ecat_browser.adapters import run_single_cv_analysis
    import matplotlib.pyplot as plt

    class cv:
        def plot(self, options):
            plt.figure()
            plt.plot([0, 1], [1, 0])

        def peak_potential(self, options):
            plt.plot([0, 1], [0, 1])
            return {"Ep": 0.1}

    result = run_single_cv_analysis(cv(), ["peak_potential"])

    assert result["plot"].startswith("data:image/png;base64,")


def test_browser_single_cv_uses_table_index_not_selected_rows(monkeypatch):
    from ecat_browser import callbacks
    from ecat_browser.callbacks import handle_single_cv
    from ecat_browser.state import SessionRegistry

    class cv:
        pass

    first = cv()
    second = cv()

    def fake_run_single_cv_analysis(obj, analyses, options=None):
        return {
            "message": getattr(obj, "name", ""),
            "plot": "data:image/png;base64,diagnostic",
            "results": [{"analysis": "marker", "status": "ok", "value": getattr(obj, "name", ""), "message": ""}],
        }

    first.name = "table-row-0"
    second.name = "table-row-1"
    registry = SessionRegistry()
    dataset_id = registry.put([first, second])
    monkeypatch.setattr(callbacks, "run_single_cv_analysis", fake_run_single_cv_analysis)

    result = handle_single_cv(dataset_id, 0, ["marker"], registry=registry)

    assert result["message"] == "table-row-0"
    assert result["plot"] == "data:image/png;base64,diagnostic"
    assert result["results"][0]["value"] == "table-row-0"


def test_browser_analysis_index_defaults_use_first_loaded_class():
    from ecat_browser.callbacks import analysis_index_defaults, handle_single_cv, handle_single_object_analysis
    from ecat_browser.state import SessionRegistry

    class cv:
        pass

    class ca:
        pass

    class cp:
        pass

    registry = SessionRegistry()
    dataset_id = registry.put([ca(), cv(), cp(), cv()])

    values, disabled, messages = analysis_index_defaults({"dataset_id": dataset_id}, registry=registry)

    assert values == {"cv": "1", "ca": "0", "cp": "2"}
    assert disabled == {"cv": False, "ca": False, "cp": False}
    assert messages == {"cv": "", "ca": "", "cp": ""}

    ca_only_dataset_id = registry.put([ca()])
    values, disabled, messages = analysis_index_defaults({"dataset_id": ca_only_dataset_id}, registry=registry)

    assert values["cv"] == ""
    assert disabled["cv"] is True
    assert messages["cv"] == "No CV objects loaded."
    assert handle_single_cv(ca_only_dataset_id, "", [], registry=registry)["message"] == "No CV objects loaded."
    assert handle_single_object_analysis(ca_only_dataset_id, "", [], "cp", lambda obj, analyses: {}, registry=registry)["message"] == "No CP objects loaded."


def test_browser_analysis_cards_open_for_loaded_techniques():
    from ecat_browser.callbacks import analysis_card_open_state

    assert analysis_card_open_state({"summary": [{"class": "cv"}]}) == (True, False, False)
    assert analysis_card_open_state({"summary": [{"class": "ca"}, {"class": "cp"}]}) == (False, True, True)
    assert analysis_card_open_state({"summary": [{"class": "cv"}, {"class": "ca"}, {"class": "cp"}]}) == (True, True, True)
    assert analysis_card_open_state({"summary": []}) == (False, False, False)


def test_browser_ca_analysis_matches_quickstart_options():
    from ecat_browser.adapters import run_ca_analysis
    import matplotlib.pyplot as plt

    class Result(dict):
        def __init__(self, values, axes=None):
            super().__init__(values)
            self.axes = axes

    class ca:
        def __init__(self):
            self.plot_options = []
            self.charge_options = []
            self.time_options = None

        def stats(self):
            return {"Avg Current (A)": 1e-6}

        def charge(self, options=None):
            self.charge_options.append(dict(options or {}))
            fig, ax = plt.subplots()
            ax.plot([0, 1], [0, 1])
            return Result({"time": [0, 1], "charge": [0, 1e-6], "final charge": 1e-6}, axes=ax)

        def plot(self, options=None):
            self.plot_options.append(dict(options or {}))
            fig, ax = plt.subplots()
            ax.plot([0, 1], [1, 0])
            return ax

        def time_at_charge(self, options=None):
            self.time_options = dict(options or {})
            fig, ax = plt.subplots()
            ax.plot([0, 1], [1, 0])
            return Result({"time": 42.0, "target charge": options["target charge"], "time unit": "s"}, axes=ax)

    obj = ca()
    result = run_ca_analysis(
        obj,
        ["stats", "plot", "charge", "current_charge_overlay", "baseline_charge", "time_at_charge"],
        {
            "target charge": 0.75,
            "plot ca": True,
            "baseline correction": True,
            "baseline tail fraction": 0.05,
            "plot options": {
                "legend": True,
                "title": "Manual CA",
                "charge color": "tab:red",
                "grid": True,
                "deduplicate labels": True,
            },
        },
    )

    assert result["status"] == "ok"
    assert [row["analysis"] for row in result["results"]] == [
        "stats",
        "plot",
        "charge",
        "current_charge_overlay",
        "baseline_charge",
        "time_at_charge",
    ]
    charge_value = result["results"][2]["value"]
    assert charge_value == {"final charge": 1e-6}
    assert obj.plot_options[0]["legend"] is True
    assert obj.plot_options[0]["title"] == "Manual CA"
    assert obj.plot_options[0]["grid"] is True
    assert "deduplicate labels" not in obj.plot_options[0]
    assert obj.plot_options[1]["plot charge"] is True
    assert obj.plot_options[1]["charge color"] == "tab:red"
    assert obj.plot_options[1]["grid"] is True
    assert "deduplicate labels" not in obj.plot_options[1]
    assert obj.charge_options[0]["grid"] is True
    assert "deduplicate labels" not in obj.charge_options[0]
    assert obj.charge_options[1]["baseline correction"] is True
    assert obj.charge_options[1]["baseline tail fraction"] == 0.05
    assert obj.charge_options[1]["grid"] is True
    assert "deduplicate labels" not in obj.charge_options[1]
    assert obj.time_options["target charge"] == 0.75
    assert obj.time_options["plot ca"] is True
    assert obj.time_options["legend"] is True
    assert obj.time_options["grid"] is True
    assert "deduplicate labels" not in obj.time_options
    assert result["plot"].startswith("data:image/png;base64,")


def test_browser_cp_analysis_runs_stats_cycle_info_and_cycles_plot():
    from ecat_browser.adapters import run_cp_analysis
    import matplotlib.pyplot as plt

    class cp:
        def __init__(self):
            self.cycle_info_options = []
            self.plot_options = []
            self.cycling_plot_options = []
            self.plot_cycles_options = []

        def stats(self):
            return {"Segments": 2}

        def cycle_info(self, options=None):
            self.cycle_info_options.append(dict(options or {}))
            return {"Cycle": [1]}

        def plot(self, options=None):
            self.plot_options.append(dict(options or {}))
            fig, ax = plt.subplots()
            ax.plot([0, 1], [1, 0])
            return ax

        def cycling_plot(self, options=None):
            self.cycling_plot_options.append(dict(options or {}))
            fig, ax = plt.subplots()
            ax.plot([0, 1], [0, 1])
            return fig, (ax, ax)

        def plot_cycles(self, options=None):
            self.plot_cycles_options.append(dict(options or {}))
            fig, ax = plt.subplots()
            ax.plot([0, 1], [0, 1])
            return ax

    obj = cp()
    result = run_cp_analysis(
        obj,
        ["stats", "cycle_info", "plot", "cycling_plot", "plot_cycles"],
        {
            "percent capacity": False,
            "capacity mode": "both",
            "efficiency mode": "both",
            "cycles": (1, 100, 10),
            "segment": "both",
            "x axis": "capacity",
            "plot options": {
                "legend": True,
                "legend mode": "colorbar",
                "color mode": "auto",
                "gradient colormap": "viridis",
                "title": "auto",
                "grid": True,
                "deduplicate labels": True,
            },
        },
    )

    assert result["status"] == "ok"
    assert [row["analysis"] for row in result["results"]] == ["stats", "cycle_info", "plot", "cycling_plot", "plot_cycles"]
    assert obj.cycle_info_options[0]["percent capacity"] is False
    assert obj.plot_options[0]["legend"] is True
    assert obj.plot_options[0]["grid"] is True
    assert "deduplicate labels" not in obj.plot_options[0]
    assert obj.cycling_plot_options[0]["capacity mode"] == "both"
    assert obj.cycling_plot_options[0]["efficiency mode"] == "both"
    assert obj.cycling_plot_options[0]["grid"] is True
    assert "deduplicate labels" not in obj.cycling_plot_options[0]
    assert obj.plot_cycles_options[0]["cycles"] == (1, 100, 10)
    assert obj.plot_cycles_options[0]["segment"] == "both"
    assert obj.plot_cycles_options[0]["x axis"] == "capacity"
    assert obj.plot_cycles_options[0]["legend mode"] == "colorbar"
    assert obj.plot_cycles_options[0]["color mode"] == "auto"
    assert obj.plot_cycles_options[0]["grid"] is True
    assert "deduplicate labels" not in obj.plot_cycles_options[0]
    assert result["plots"][0]["label"] == "Potential Plot"
    assert result["plots"][1]["label"] == "Cycling Performance"
    assert result["plots"][2]["label"] == "Cycle Plot"
    assert all(plot["src"].startswith("data:image/png;base64,") for plot in result["plots"])


def test_browser_multi_cv_analysis_dispatches_public_ecat_function(monkeypatch):
    from ecat_browser.adapters import run_multi_cv_analysis

    calls = []

    def fake_fit_peak_current(cvs, options):
        calls.append((cvs, dict(options)))
        return {"fit": "ok"}

    import ecat_browser.adapters as adapters

    monkeypatch.setattr(adapters.e, "fit_peak_current", fake_fit_peak_current)
    class cv:
        pass

    objects = [cv(), cv()]

    result = run_multi_cv_analysis(
        objects,
        "fit_peak_current",
        {"segment": 1, "guess potential": -0.2, "x axis": "scan rate", "fit": True},
    )

    assert result["status"] == "ok"
    assert result["analysis"] == "fit_peak_current"
    assert calls == [
        (
            objects,
            {
                "segment": 1,
                "guess potential": -0.2,
                "x axis": "scan rate",
                "fit": True,
                    "plot": True,
                    "plot all": True,
                    "new plot": False,
                    "print": False,
                },
            )
        ]


def test_browser_multi_cv_options_filter_sevcik_unsupported_fit_options():
    from ecat_browser.callbacks import multi_cv_options_from_controls

    options = multi_cv_options_from_controls(
        "sevcik_analysis",
        segments="1, 2",
        guess_potential=-0.2,
        x_axis="scan rate",
        fit_model="power",
        toggles=["fit", "plot_fit", "plot_all"],
    )

    assert options == {
        "plot fit": True,
        "plot all": True,
        "segments": [1, 2],
        "guess potential": -0.2,
    }


def test_browser_multi_cv_options_include_fowa_notebook_07_controls():
    from ecat_browser.callbacks import multi_cv_options_from_controls

    options = multi_cv_options_from_controls(
        "fowa",
        segments="1",
        guess_potential="-1.5",
        toggles=["plot_fit", "plot_all"],
        fowa_reference_index="0",
        fowa_redox_mode="manual",
        fowa_redox_potential="-1.46",
        fowa_fit_basis="y",
        fowa_fit_range_start="0.1",
        fowa_fit_range_end="0.5",
        fowa_diagnostic_y_axis="i/ip0",
        fowa_min_fit_points="50",
        fowa_min_r2="0.95",
    )

    assert options == {
        "plot fit": True,
        "plot all": True,
        "segments": [1],
        "guess potential": -1.5,
        "redox mode": "manual",
        "fit basis": "y",
        "diagnostic y axis": "i/ip0",
        "fit range": [0.1, 0.5],
        "min fit points": 50,
        "min r2": 0.95,
        "non-catalytic cv index": 0,
        "redox potential": -1.46,
    }


def test_browser_multi_cv_options_include_tafel_positional_controls():
    from ecat_browser.callbacks import multi_cv_options_from_controls

    options = multi_cv_options_from_controls(
        "tafel_analysis",
        toggles=["plot_all", "plot_fit"],
        tafel_index="3",
        tafel_tof_max="1000",
        tafel_thermo_potential="-0.2",
        tafel_redox_potential="-1.0",
        tafel_overpotential_start="0",
        tafel_overpotential_end="1",
        tafel_color="black",
    )

    assert options == {
        "cv index": 3,
        "TOF max": 1000.0,
        "thermodynamic potential": -0.2,
        "redox potential": -1.0,
        "overpotential range": [0.0, 1.0],
        "color": "black",
    }


def test_browser_multi_cv_fowa_resolves_reference_index(monkeypatch):
    from ecat_browser.adapters import run_multi_cv_analysis

    calls = []

    def fake_fowa(cvs, options):
        calls.append((cvs, dict(options)))
        return [{"FOWA": "ok"}]

    import ecat_browser.adapters as adapters

    monkeypatch.setattr(adapters.e, "fowa", fake_fowa)

    class cv:
        pass

    reference = cv()
    catalytic = cv()

    result = run_multi_cv_analysis(
        [catalytic],
        "fowa",
        {"non-catalytic cv index": 0, "redox mode": "half wave"},
        all_objects=[reference, catalytic],
    )

    assert result["status"] == "ok"
    assert calls[0][0] == [catalytic]
    assert calls[0][1]["non-catalytic cv"] is reference
    assert calls[0][1]["redox mode"] == "half wave"
    assert calls[0][1]["plot all"] is True


def test_browser_multi_cv_tafel_uses_positional_public_call(monkeypatch):
    from ecat_browser.adapters import run_multi_cv_analysis

    calls = []

    def fake_tafel_analysis(cv_obj, tof_max, thermodynamic_potential, redox_potential, options):
        calls.append((cv_obj, tof_max, thermodynamic_potential, redox_potential, dict(options)))
        return {"tafel": "ok"}

    import ecat_browser.adapters as adapters

    monkeypatch.setattr(adapters.e, "tafel_analysis", fake_tafel_analysis)

    class cv:
        pass

    first = cv()
    selected = cv()

    result = run_multi_cv_analysis(
        [first],
        "tafel_analysis",
        {
            "cv index": 1,
            "TOF max": 1000,
            "thermodynamic potential": -0.2,
            "redox potential": -1.0,
            "overpotential range": [0, 1],
            "color": "black",
        },
        all_objects=[first, selected],
    )

    assert result["status"] == "ok"
    assert calls == [(selected, 1000, -0.2, -1.0, {"overpotential range": [0, 1], "color": "black"})]


def test_browser_multi_cv_analysis_rejects_unsupported_analysis():
    from ecat_browser.adapters import run_multi_cv_analysis

    result = run_multi_cv_analysis([object()], "made_up_analysis", {})

    assert result["status"] == "skipped"
    assert "Unsupported" in result["message"]


def test_browser_multi_cv_results_include_plot():
    from ecat_browser.callbacks import render_multi_cv_results

    rendered = repr(
        render_multi_cv_results(
            {
                "analysis": "sevcik_analysis",
                "status": "ok",
                "message": "",
                "plot": "data:image/png;base64,abc",
                "value": {"ok": True},
            }
        )
    )

    assert "data:image/png;base64,abc" in rendered
    assert "ecat-plot" in rendered


def test_browser_reference_settings_build_existing_ecat_options():
    from ecat_browser.references import build_reference_options

    manual = build_reference_options(
        {
            "mode": "manual",
            "offset": "0.12",
            "label": "Fc/Fc+",
        }
    )
    file_mode = build_reference_options(
        {
            "mode": "file",
            "file": "Fc_reference.txt",
            "guess": "0.0",
            "label": "Fc/Fc+",
        }
    )
    keyword = build_reference_options(
        {
            "mode": "keyword",
            "keyword": "Fc",
            "guess": "auto",
            "label": "Fc/Fc+",
            "allow_self_reference": False,
        }
    )
    auto = build_reference_options(
        {
            "mode": "auto",
            "keywords": "Fc, Cp2Fe",
            "guess": "auto",
        }
    )
    mapped = build_reference_options(
        {
            "mode": "keyword",
            "keyword": "Fc",
            "map": [{"target": "0", "reference": "2"}],
        }
    )

    assert manual["reference mode"] == "manual"
    assert manual["reference offset"] == 0.12
    assert file_mode["reference file"] == "Fc_reference.txt"
    assert keyword["allow self reference"] is False
    assert auto["reference keywords"] == ["Fc", "Cp2Fe"]
    assert mapped["reference map"] == {0: 2}


def test_browser_reference_file_options_use_row_indices():
    from ecat_browser.callbacks import reference_file_options

    options = reference_file_options(
        [
            {"index": 0, "filename": "sample.txt"},
            {"index": 7, "filename": "reference.txt"},
        ]
    )

    assert options == [
        {"label": "0: sample.txt", "value": 0},
        {"label": "7: reference.txt", "value": 7},
    ]


def test_browser_reference_file_index_resolves_to_loaded_filepath(cv_factory, tmp_path):
    from ecat_browser.callbacks import reference_file_path_from_index
    from ecat_browser.state import SessionRegistry

    reference = cv_factory(name="100mVs_reference")
    reference.filepath = str(tmp_path / "reference.txt")
    registry = SessionRegistry()
    dataset_id = registry.put([reference], [])

    resolved = reference_file_path_from_index({"dataset_id": dataset_id}, 0, registry=registry)

    assert resolved == str(tmp_path / "reference.txt")


def test_browser_local_reload_applies_manual_reference_options(fixtures_dir, tmp_path):
    from ecat_browser.adapters import load_local_path, reload_workflow
    from ecat_browser.references import build_reference_options

    _copy_fixture(fixtures_dir, tmp_path, "ch_cv.txt")
    result = load_local_path(tmp_path, recursive=False)
    result.workflow.reference_settings = {
        "mode": "manual",
        "offset": "0.12",
        "label": "Fc/Fc+",
    }
    result.workflow.import_options = build_reference_options(result.workflow.reference_settings)

    reloaded = reload_workflow(result.workflow)

    assert reloaded.objects[0].reference_mode == "manual"
    assert reloaded.objects[0].reference_shift == pytest.approx(0.12)


def test_browser_codegen_uses_public_ecat_api_only():
    from ecat_browser.codegen import generate_python
    from ecat_browser.workflow import BrowserWorkflow

    workflow = BrowserWorkflow(
        source_kind="local_path",
        source_path="/data/cvs",
        recursive=True,
        selected_index=0,
        filters={"gas": "CO2"},
        group_keys=["gas"],
        sort_keys=["scan rate"],
        analyses=["peak_potential", "peak_current"],
        export_filename="processed_cv",
        plot_options={"legend": True, "title": "auto", "plot convention": "IUPAC"},
        included_row_ids=["row-0", "row-2"],
        import_options={
            "reference mode": "manual",
            "reference offset": 0.12,
            "reference label": "Fc/Fc+",
        },
    )

    code = generate_python(workflow)

    assert "import ecat as e" in code
    assert "e.get_data" in code
    assert "e.filter" in code
    assert "e.sort_and_group" in code
    assert ".peak_potential" in code
    assert ".peak_current" in code
    assert "plot_options = {" in code
    assert "\"plot convention\": \"IUPAC\"" in code
    assert "overlay_ax = e.multiplot" in code
    assert "e.save_data" in code
    assert '"reference mode": "manual"' in code
    assert "included_indices = [0, 2]" in code
    assert "dash" not in code.lower()
    assert "ecat_browser" not in code


def test_browser_code_execution_is_disabled_without_explicit_trusted_flag(monkeypatch):
    from ecat_browser.execution import code_execution_allowed, run_user_code

    monkeypatch.delenv("ECAT_BROWSER_ALLOW_CODE_EXECUTION", raising=False)

    assert code_execution_allowed() is False
    result = run_user_code("print('should not run')")
    assert result.executed is False
    assert "disabled" in result.stderr.lower()


def test_browser_code_execution_runs_when_trusted_flag_is_set(monkeypatch, tmp_path):
    from ecat_browser.execution import code_execution_allowed, run_user_code

    monkeypatch.setenv("ECAT_BROWSER_ALLOW_CODE_EXECUTION", "1")

    assert code_execution_allowed() is True
    result = run_user_code("print('ran locally')", cwd=tmp_path, timeout_seconds=2)
    assert result.executed is True
    assert result.returncode == 0
    assert result.stdout.strip() == "ran locally"


def test_browser_session_registry_stores_objects_outside_dash_json(ecat_module, cv_factory):
    from ecat_browser.state import SessionRegistry

    registry = SessionRegistry()
    obj = cv_factory()

    dataset_id = registry.put([obj], warnings=["note"])
    snapshot = registry.snapshot(dataset_id)

    assert snapshot["dataset_id"] == dataset_id
    assert snapshot["warnings"] == ["note"]
    assert snapshot["summary"][0]["class"] == "cv"
    assert registry.get(dataset_id)[0] is obj


def test_browser_session_registry_returns_included_objects_by_stable_row_id(ecat_module, cv_factory):
    from ecat_browser.state import SessionRegistry

    registry = SessionRegistry()
    first = cv_factory(name="50mVs_first_CO2_MeCN")
    second = cv_factory(name="100mVs_second_CO2_MeCN")
    dataset_id = registry.put([first, second])

    included = registry.get_included(dataset_id, ["row-1"])

    assert included == [second]


def test_browser_callback_helpers_return_serializable_import_state(fixtures_dir, tmp_path):
    from ecat_browser.callbacks import handle_local_path_load
    from ecat_browser.state import SessionRegistry

    _copy_fixture(fixtures_dir, tmp_path, "ch_cv.txt")
    registry = SessionRegistry()

    state = handle_local_path_load(str(tmp_path), recursive=False, registry=registry)

    assert state["dataset_id"]
    assert state["summary"][0]["class"] == "cv"
    assert state["included_row_ids"] == ["row-0"]
    assert state["workflow"]["source_kind"] == "local_path"
    assert "import ecat as e" in state["code"]
    assert registry.get(state["dataset_id"])


def test_browser_table_rows_follow_ecat_build_object_table_columns(cv_factory):
    from ecat_browser.table import build_browser_table

    first = cv_factory(name="50mVs_sample_CO2_MeCN_10mM_Fc_run01")
    second = cv_factory(name="100mVs_sample_Ar_MeCN_10mM_Fc_run02")

    table = build_browser_table([first, second], visible_columns=["Name", "Gas", "Scan Rate"])

    column_ids = [column["id"] for column in table["columns"]]
    assert "Gas" in column_ids
    assert "Scan Rate" in column_ids
    assert column_ids[-1] == "Scan Rate"
    assert table["data"][0]["id"] == "row-0"
    assert table["data"][1]["id"] == "row-1"


def test_browser_table_pretty_formats_chemical_values(cv_factory):
    from ecat_browser.table import build_browser_table

    first = cv_factory(name="50mVs_sample_CO2_MeCN_10mM_Fc_run01")

    table = build_browser_table([first], visible_columns=["Gas", "Solvent"])

    assert table["data"][0]["Gas"] == "CO₂"
    assert table["data"][0]["Solvent"] == "MeCN"


def test_browser_table_displays_reference_source_as_loaded_index(cv_factory, tmp_path):
    from ecat_browser.table import build_browser_table

    reference = cv_factory(name="100mVs_reference_Fc")
    target = cv_factory(name="100mVs_target_Fc")
    reference.filepath = str(tmp_path / "reference.txt")
    target.filepath = str(tmp_path / "target.txt")
    target.reference_mode = "file"
    target.reference_source_file = str(tmp_path / "reference.txt")

    table = build_browser_table([reference, target], visible_columns=["Reference Source"])

    assert table["data"][1]["Reference Source"] == 0


def test_browser_table_columns_are_true_visible_selection(cv_factory):
    from ecat_browser.table import available_column_options, build_browser_table, default_visible_columns, selected_column_values
    from ecat.plotting import build_object_table, pretty_table_column_label, show_objects

    first = cv_factory(name="50mVs_sample_CO2_MeCN_10mM_Fc_run01")
    second = cv_factory(name="100mVs_sample_Ar_DMF_10mM_Fc_run02")
    first.timestamp = "2024-01-01 12:00:00"
    first.temperature = 298
    first.ir_comp_resistance = 42

    table = build_browser_table([first, second])
    default_columns = default_visible_columns([first, second])
    available_columns = [option["value"] for option in available_column_options([first, second])]
    selected = selected_column_values(table)
    ecat_default, _meta = build_object_table([first, second], {"print conditions": False})
    ecat_available = show_objects(
        [first, second],
        {"columns": "available", "print conditions": False},
    )

    assert default_columns == list(ecat_default.columns)
    assert "Gas" in default_columns
    assert "Scan Rate" in default_columns
    assert "Type" not in default_columns
    assert "Reference Shift" not in default_columns
    assert "Temperature" not in default_columns
    assert "IR Comp Resistance" not in default_columns
    for column in ecat_available:
        assert pretty_table_column_label(column) in available_columns
    assert "Filename" in available_columns
    assert "Class" in available_columns
    assert "Software" in available_columns
    assert "Timestamp" in available_columns
    assert "Temperature" in available_columns
    assert "IR Comp Resistance" in available_columns
    assert selected == [column["id"] for column in table["columns"] if column["id"] != "index"]

    reduced = build_browser_table([first], visible_columns=["Filename", "Timestamp"])
    assert [column["id"] for column in reduced["columns"]] == ["index", "Filename", "Timestamp"]


def test_browser_loaded_state_carries_all_available_column_options(cv_factory):
    from ecat_browser.callbacks import _state_from_load_result
    from ecat_browser.state import SessionRegistry
    from ecat_browser.workflow import BrowserWorkflow

    class Result:
        objects = [
            cv_factory(name="50mVs_sample_CO2_MeCN_10mM_Fc_run01"),
            cv_factory(name="100mVs_sample_Ar_DMF_10mM_Fc_run02"),
        ]
        warnings = []
        workflow = BrowserWorkflow()
        status = ""

    state = _state_from_load_result(Result(), registry=SessionRegistry())
    selected_columns = {column["id"] for column in state["table"]["columns"] if column["id"] != "index"}
    available_columns = {option["value"] for option in state["column_options"]}

    assert selected_columns < available_columns
    assert "Temperature" in available_columns
    assert "IR Comp Resistance" in available_columns


def test_browser_default_fe_phoh_load_selects_scan_window_with_ferrocene(cv_factory, tmp_path):
    from ecat_browser.adapters import default_included_row_ids
    from ecat_browser.workflow import BrowserWorkflow

    objects = []
    for index in range(12):
        if index in {1, 7, 8, 9, 10, 11}:
            name = f"sample_{index}_3mMFc_-1.2_to_1V_100mVs"
        elif index == 0:
            name = "sample_0_-1.2_to_1V_100mVs"
        else:
            name = f"sample_{index}_3mMFc_-1.7_to_1V_100mVs"
        objects.append(cv_factory(name=name))
    workflow = BrowserWorkflow(source_path=str(tmp_path / "examples" / "data" / "fe_phoh_cv"))

    included = default_included_row_ids(objects, workflow)

    assert included == ["row-1", "row-7", "row-8", "row-9", "row-10", "row-11"]


def test_browser_loaded_state_includes_pretty_conditions(cv_factory):
    from ecat_browser.callbacks import _state_from_load_result
    from ecat_browser.state import SessionRegistry
    from ecat_browser.workflow import BrowserWorkflow

    class Result:
        objects = [
            cv_factory(name="50mVs_sample_CO2_MeCN_10mM_Fc_run01"),
            cv_factory(name="100mVs_sample_CO2_MeCN_10mM_Fc_run02"),
        ]
        warnings = []
        workflow = BrowserWorkflow()
        status = ""

    state = _state_from_load_result(Result(), registry=SessionRegistry())

    assert "Gas: CO₂" in state["conditions"]
    assert "Solvent: MeCN" in state["conditions"]


def test_browser_layout_initial_column_selector_uses_available_options():
    pytest.importorskip("dash")
    from ecat_browser.layout import create_layout

    initial_state = {
        "summary": [{"index": 0, "filename": "reference.txt"}],
        "table": {
            "columns": [{"name": "index", "id": "index"}, {"name": "Gas", "id": "Gas"}],
            "data": [{"id": "row-0", "index": 0, "Gas": "CO2"}],
        },
        "column_options": [
            {"label": "Gas", "value": "Gas"},
            {"label": "Temperature", "value": "Temperature"},
        ],
        "conditions": ["Gas: CO₂"],
    }

    rendered = repr(create_layout(initial_state=initial_state))

    assert "Temperature" in rendered
    assert "Shared Conditions" in rendered
    assert "0: reference.txt" in rendered


def test_browser_reference_visibility_depends_on_mode():
    from ecat_browser.references import reference_field_visibility

    assert reference_field_visibility("none")["manual"] is False
    assert reference_field_visibility("manual")["manual"] is True
    assert reference_field_visibility("file")["file"] is True
    assert reference_field_visibility("keyword")["keyword"] is True
    assert reference_field_visibility("auto")["auto"] is True


def test_browser_default_source_points_to_fe_phoh_data(repo_root):
    from ecat_browser.defaults import default_workflow, example_folder_options, example_folder_path

    workflow = default_workflow(repo_root)

    assert workflow.source_kind == "local_path"
    assert workflow.source_path.endswith("examples/data/fe_phoh_cv")
    assert workflow.recursive is True
    assert workflow.import_options["sort keys"] == ["timestamp"]
    assert [option["value"] for option in example_folder_options()] == [
        "fe_phoh_cv",
        "chrono_ca",
        "chrono_cp",
    ]
    assert example_folder_path("chrono_ca", repo_root).as_posix().endswith("examples/data/chrono_ca")
    assert example_folder_path("chrono_cp", repo_root).as_posix().endswith("examples/data/chrono_cp")
    assert example_folder_path("missing", repo_root) is None


def test_browser_local_path_loading_reports_txt_file_status(fixtures_dir, tmp_path):
    from ecat_browser.adapters import load_local_path

    _copy_fixture(fixtures_dir, tmp_path, "ch_cv.txt")
    _copy_fixture(fixtures_dir, tmp_path, "basi_cv.txt")

    result = load_local_path(tmp_path, recursive=True)

    assert result.status == "2 .txt files found."


def test_browser_callback_state_includes_import_status(fixtures_dir, tmp_path):
    from ecat_browser.callbacks import handle_local_path_load
    from ecat_browser.state import SessionRegistry

    _copy_fixture(fixtures_dir, tmp_path, "ch_cv.txt")
    registry = SessionRegistry()

    state = handle_local_path_load(str(tmp_path), recursive=True, registry=registry)

    assert state["status"] == "1 .txt file found."


def test_browser_import_returns_default_multiplot(fixtures_dir, tmp_path):
    from ecat_browser.callbacks import handle_local_path_load
    from ecat_browser.state import SessionRegistry

    _copy_fixture(fixtures_dir, tmp_path, "ch_cv.txt")
    _copy_fixture(fixtures_dir, tmp_path, "basi_cv.txt")
    registry = SessionRegistry()

    state = handle_local_path_load(str(tmp_path), recursive=False, registry=registry)

    assert state["plot"].startswith("data:image/png;base64,")


def test_browser_sort_keys_reorder_included_objects_for_analysis(cv_factory):
    from ecat_browser.adapters import objects_for_analysis
    from ecat_browser.workflow import BrowserWorkflow

    slow = cv_factory(name="50mVs_sample_CO2_MeCN_10mM_Fc_run01")
    fast = cv_factory(name="100mVs_sample_CO2_MeCN_10mM_Fc_run02")
    workflow = BrowserWorkflow(
        included_row_ids=["row-0", "row-1"],
        sort_keys=["scan rate"],
    )

    ordered = objects_for_analysis([fast, slow], workflow)

    assert [obj.scan_rate for obj in ordered] == [0.05, 0.1]


def test_browser_select_all_state_cycles_between_all_and_none():
    from ecat_browser.table import selection_toggle_state, toggle_all_selection

    rows = [{"id": "row-0"}, {"id": "row-1"}]

    assert selection_toggle_state(rows, ["row-0"]) == "-"
    assert selection_toggle_state(rows, ["row-0", "row-1", "row-hidden"]) == "None"
    assert toggle_all_selection(rows, ["row-0"]) == ["row-0", "row-1"]
    assert toggle_all_selection(rows, ["row-0", "row-1"]) == []


def test_browser_table_selection_defaults_include_dash_rows(cv_factory):
    from ecat_browser.table import selected_rows_for_table, toggle_all_selection_state

    rows = [{"id": "row-0"}, {"id": "row-1"}]

    assert selected_rows_for_table({"data": rows}) == [0, 1]
    assert toggle_all_selection_state(rows, ["row-0", "row-1"]) == ([], [])
    assert toggle_all_selection_state(rows, []) == (["row-0", "row-1"], [0, 1])


def test_browser_filtered_selection_toggle_preserves_hidden_rows():
    from ecat_browser.table import toggle_filtered_selection_state

    all_rows = [{"id": "row-0"}, {"id": "row-1"}, {"id": "row-2"}]
    displayed_rows = [{"id": "row-2"}, {"id": "row-0"}]

    assert toggle_filtered_selection_state(all_rows, displayed_rows, ["row-1"]) == (
        ["row-1", "row-2", "row-0"],
        [0, 1, 2],
    )
    assert toggle_filtered_selection_state(all_rows, displayed_rows, ["row-0", "row-1", "row-2"]) == (
        ["row-1"],
        [1],
    )


def test_browser_column_selector_tracks_visible_columns_and_resets(cv_factory):
    from ecat_browser.table import (
        build_browser_table,
        selected_column_values,
        reset_column_selection,
    )

    first = cv_factory(name="50mVs_sample_CO2_MeCN_10mM_Fc_run01")
    second = cv_factory(name="100mVs_sample_Ar_MeCN_10mM_Fc_run02")

    table = build_browser_table([first, second], extra_columns=["name"])

    assert selected_column_values(table)[-1] == "Name"
    reset = reset_column_selection([first, second])
    assert "Name" not in [column["id"] for column in reset["columns"]]


def test_browser_table_builds_ag_grid_column_defs(cv_factory):
    from ecat_browser.table import ag_grid_column_defs, build_browser_table

    first = cv_factory(name="50mVs_sample_CO2_MeCN_10mM_Fc_run01")
    table = build_browser_table([first])

    column_defs = ag_grid_column_defs(table)

    assert column_defs[0]["field"] == "index"
    assert column_defs[0]["checkboxSelection"] is True
    assert column_defs[0]["headerCheckboxSelection"] is True
    assert column_defs[0]["minWidth"] >= 110
    assert column_defs[1]["field"] == "Name"
    assert all(column["sortable"] is True for column in column_defs)
    assert all(column["filter"] is True for column in column_defs)


def test_browser_table_maps_grid_selection_to_stable_row_ids():
    from ecat_browser.table import selected_grid_rows_for_ids, selected_row_ids_from_grid_rows

    rows = [{"id": "row-0", "Filename": "a.txt"}, {"id": "row-1", "Filename": "b.txt"}]

    assert selected_row_ids_from_grid_rows([rows[1]]) == ["row-1"]
    assert selected_grid_rows_for_ids(rows, ["row-1", "row-missing"]) == {"ids": ["row-1"]}


def test_browser_replot_uses_current_selection(cv_factory):
    from ecat_browser.callbacks import handle_replot
    from ecat_browser.state import SessionRegistry

    registry = SessionRegistry()
    first = cv_factory(name="50mVs_sample_CO2_MeCN_10mM_Fc_run01")
    second = cv_factory(name="100mVs_sample_CO2_MeCN_10mM_Fc_run02")
    dataset_id = registry.put([first, second])

    result = handle_replot(dataset_id, ["row-1"], registry=registry)

    assert result["plot"].startswith("data:image/png;base64,")


def test_browser_displayed_row_ids_preserve_table_order():
    from ecat_browser.table import displayed_selected_row_ids

    displayed = [{"id": "row-2"}, {"id": "row-0"}, {"id": "row-1"}]

    assert displayed_selected_row_ids(displayed, ["row-1", "row-2"]) == ["row-2", "row-1"]
    assert displayed_selected_row_ids(displayed, None) == ["row-2", "row-0", "row-1"]


def test_browser_sidebar_class_toggles_collapsed_state():
    from ecat_browser.callbacks import expand_sidebar_class, toggle_sidebar_class

    assert toggle_sidebar_class("ecat-app") == "ecat-app ecat-sidebar-collapsed"
    assert toggle_sidebar_class("ecat-app ecat-sidebar-collapsed") == "ecat-app"
    assert expand_sidebar_class("ecat-app ecat-sidebar-collapsed") == "ecat-app"
    assert expand_sidebar_class("ecat-app") == "ecat-app"


def test_browser_callback_helpers_apply_reference_by_reloading(fixtures_dir, tmp_path):
    from ecat_browser.callbacks import handle_apply_reference, handle_local_path_load
    from ecat_browser.state import SessionRegistry

    _copy_fixture(fixtures_dir, tmp_path, "ch_cv.txt")
    registry = SessionRegistry()
    state = handle_local_path_load(str(tmp_path), recursive=False, registry=registry)

    updated = handle_apply_reference(
        state["workflow"],
        {"mode": "manual", "offset": "0.12", "label": "Fc/Fc+"},
        registry=registry,
    )

    assert updated["dataset_id"] != state["dataset_id"]
    assert updated["summary"][0]["reference mode"] == "manual"
    assert updated["summary"][0]["reference shift"] == pytest.approx(0.12)


def test_browser_app_mode_controls_local_folder_and_code_execution(monkeypatch):
    from ecat_browser.config import BrowserAppConfig

    monkeypatch.setenv("ECAT_BROWSER_MODE", "remote")
    monkeypatch.setenv("ECAT_BROWSER_ALLOW_CODE_EXECUTION", "1")

    remote = BrowserAppConfig.from_env()
    local = BrowserAppConfig(mode="local", allow_code_execution=True)

    assert remote.enable_folder_picker is False
    assert remote.allow_code_execution is False
    assert local.enable_folder_picker is True
    assert local.allow_code_execution is True


def test_browser_dash_layout_contains_expected_tabs():
    pytest.importorskip("dash")
    from ecat_browser.layout import TAB_IDS, create_layout

    layout = create_layout()
    rendered = repr(layout)

    assert TAB_IDS == ("import", "plotting", "analysis", "export")
    assert layout is not None
    assert "ecat-import-source-tabs" in rendered
    assert "ecat-import-conditions" in rendered
    assert "Files" in rendered
    assert "Folder" in rendered
    assert "Example data" in rendered
    assert "ecat-example-folder" in rendered
    assert "Fe/PhOH CV" in rendered
    assert "CA/CPE" in rendered
    assert "CP Cycling" in rendered
    assert "Folder path" in rendered
    assert "Load Folder" in rendered
    assert "ecat-select-folder" not in rendered
    assert "Search Subfolders" in rendered
    assert "ecat-folder-pick-store" not in rendered
    assert "ecat-upload" in rendered
    assert "ecat-object-table" in rendered
    assert "AgGrid" in rendered
    assert "columnSize='autoSize'" in rendered
    assert "columnDefs" in rendered
    assert "rowData" in rendered
    assert "selectedRows" in rendered
    assert "virtualRowData" not in rendered
    assert "ecat-selected-row-ids-store" in rendered
    assert "dash_table" not in rendered
    assert "ecat-table-extra-columns" in rendered
    assert "ecat-reset-columns" in rendered
    assert "ecat-replot" in rendered
    assert "ecat-save-plot" in rendered
    assert "ecat-sidebar-toggle" in rendered
    assert "ecat-sidebar-resizer" in rendered
    assert "ecat-left-tabs" in rendered
    assert "ecat-tab-symbol" in rendered
    assert "/assets/ecat_icon_import.svg" in rendered
    assert "/assets/ecat_icon_plotting.svg" in rendered
    assert "/assets/ecat_icon_analysis.svg" in rendered
    assert "/assets/ecat_icon_export.svg" in rendered
    assert "ecat-app-header" in rendered
    assert "ecat-header-logo" in rendered
    assert "/assets/ecat-logo.svg" in rendered
    assert "eCAT Workbench" in rendered
    assert "Browser analysis workspace" in rendered
    assert "ecat-about-button" in rendered
    assert "ecat-about-panel" in rendered
    assert "About eCAT Workbench" in rendered
    assert "ecat 0.1.0b2" in rendered
    assert "Luke Elissiry" in rendered
    assert "MIT License" in rendered
    assert "https://github.com/ljelissiry/eCAT" in rendered
    assert "ecat-brand" not in rendered
    assert "ecat-main-panel-card" in rendered
    assert "Imported Data" in rendered
    assert "ecat-plot-card" in rendered
    assert "Multiplot" in rendered
    assert "ecat-plot-loading" in rendered
    assert "ecat-single-index" in rendered
    assert "CV Index:" in rendered
    assert "ecat-analysis-checklist" in rendered
    assert "ecat-single-guess-potential" in rendered
    assert "GUESS" in rendered
    assert "POTENTIAL" in rendered
    assert "ecat-single-tangent-potential" in rendered
    assert "TANGENT" in rendered
    assert "ecat-potential-control-row" in rendered
    assert "type='text'" in rendered
    assert "inputMode='decimal'" in rendered
    assert "inputMode='numeric'" in rendered
    assert "ecat-single-index-status" in rendered
    assert "value=0" in rendered
    assert "Half wave potential" in rendered
    assert "ecat-analysis-results" in rendered
    assert "Analysis Results" in rendered
    assert "Plotting Style" in rendered
    assert "ecat-plot-style" in rendered
    assert "Line + markers" in rendered
    assert "ecat-plot-legend" in rendered
    assert "Visibility" in rendered
    assert "ecat-plot-gradients" in rendered
    assert "ecat-plot-colorbar-wrap" in rendered
    assert "ecat-plot-title-mode" in rendered
    assert "Mode" in rendered
    assert "ecat-plot-custom-title-wrap" in rendered
    assert "ecat-plot-convention" in rendered
    assert "IUPAC" in rendered
    assert "Options" in rendered
    assert "ecat-plot-trim-enabled" in rendered
    assert "ecat-plot-trim-bounds" in rendered
    assert "ecat-plot-trim-min" in rendered
    assert "ecat-plot-trim-max" in rendered
    assert "ecat-plot-format" in rendered
    assert "ecat-plot-dpi" in rendered
    assert "ecat-plot-colorbar" in rendered
    assert "Allow colorbars" in rendered
    assert "ecat-plot-label-options" in rendered
    assert "value=['colorbar']" in rendered
    assert "ecat-plot-offset" in rendered
    assert "ecat-plot-offset-controls" in rendered
    assert "ecat-plot-scale-bar-height" in rendered
    assert "ecat-plot-scale-bar-location" in rendered
    assert "ecat-plot-offset-axis-options" in rendered
    assert "Output" in rendered
    assert "Cyclic Voltammetry" in rendered
    assert "Chronoamperometry" in rendered
    assert "Chronopotentiometry" in rendered
    assert "ecat-ca-index" in rendered
    assert "CA Index:" in rendered
    assert "ecat-ca-index-status" in rendered
    assert "ecat-ca-analyses" in rendered
    assert "Cumulative charge" in rendered
    assert "Current + charge overlay" in rendered
    assert "Baseline-corrected charge" in rendered
    assert "Time at charge" in rendered
    assert "ecat-ca-target-charge" in rendered
    assert "Target charge (C):" in rendered
    assert "ecat-ca-baseline-tail-fraction" in rendered
    assert "Baseline tail fraction:" in rendered
    assert "Show CA trace with target" in rendered
    assert "ecat-run-ca" in rendered
    assert "ecat-cp-index" in rendered
    assert "CP Index:" in rendered
    assert "ecat-cp-index-status" in rendered
    assert "ecat-cp-analyses" in rendered
    assert "Cycling performance" in rendered
    assert "ecat-cp-percent-capacity" in rendered
    assert "ecat-cp-capacity-mode" in rendered
    assert "ecat-cp-efficiency-mode" in rendered
    assert "ecat-cp-cycles-start" in rendered
    assert "ecat-cp-cycles-end" in rendered
    assert "ecat-cp-cycles-step" in rendered
    assert "ecat-cp-cycle-segment" in rendered
    assert "ecat-cp-cycle-x-axis" in rendered
    assert "ecat-run-cp" in rendered
    assert "ecat-cv-analysis-card" in rendered
    assert "ecat-ca-analysis-card" in rendered
    assert "ecat-cp-analysis-card" in rendered
    assert "Single" in rendered
    assert "Multiple" in rendered
    assert "ecat-multi-analysis" in rendered
    assert "ecat-multi-analysis-options" in rendered
    assert "ecat-run-multi-analysis" in rendered
    assert "ecat-filter-key" not in rendered
    assert "ecat-sort-keys" not in rendered
    assert "ecat-group-keys" not in rendered
    assert "fit_peak_potential" in rendered
    assert "fit_peak_current" in rendered
    assert "sevcik_analysis" in rendered
    assert "trumpet_analysis" in rendered
    assert "Guess potential" in rendered
    assert "Segment(s)" in rendered
    assert "X axis" in rendered
    assert "ecat-reference-mode" in rendered
    assert "ecat-export-code" in rendered
    assert "Textarea" in rendered
    assert "Run Edited Code" not in rendered
    assert "ecat-run-code" not in rendered
    assert "Export CSV filename" in rendered


def test_browser_analysis_results_render_indented_entries():
    from ecat_browser.callbacks import render_single_cv_results

    rendered = repr(
        render_single_cv_results(
            {
                "message": "Object 0",
                "plot": None,
                "results": [
                    {"analysis": "peak_potential", "status": "ok", "value": 0.1, "message": ""},
                    {"analysis": "peak_current", "status": "error", "value": None, "message": "failed"},
                ],
            }
        )
    )

    assert "Single CV Analysis" in rendered
    assert "ecat-analysis-result-list" in rendered
    assert rendered.count("ecat-analysis-result-entry") == 2


def test_browser_analysis_results_render_structured_values_as_tables():
    from ecat_browser.callbacks import render_single_cv_results

    rendered = repr(
        render_single_cv_results(
            {
                "message": "CV index 0",
                "plot": None,
                "results": [
                    {"analysis": "peak_potential", "status": "ok", "value": {"Ep": 0.1, "Segment": 1}, "message": ""},
                    {
                        "analysis": "peak_current",
                        "status": "ok",
                        "value": [{"Segment": 1, "Ip": 2.5e-6}, {"Segment": 2, "Ip": 2.1e-6}],
                        "message": "",
                    },
                ],
            }
        )
    )

    assert "ecat-analysis-value-table" in rendered
    assert "Ep" in rendered
    assert "Segment" in rendered
    assert "Ip" in rendered


def test_browser_analysis_results_render_multiple_plots():
    from ecat_browser.callbacks import render_object_analysis_results

    rendered = repr(
        render_object_analysis_results(
            {
                "message": "",
                "plots": [
                    {"label": "Potential Plot", "src": "data:image/png;base64,aaa"},
                    {"label": "Cycle Plot", "src": "data:image/png;base64,bbb"},
                ],
                "results": [],
            },
            "CP Analysis",
        )
    )

    assert "ecat-analysis-plot-list" in rendered
    assert "Potential Plot" in rendered
    assert "Cycle Plot" in rendered


def test_browser_analysis_results_store_overwrites_matching_technique():
    from ecat_browser.callbacks import upsert_analysis_result_store

    store = upsert_analysis_result_store(
        {},
        "cv",
        {"message": "old cv", "plot": None, "results": []},
        "CV Analysis",
    )
    store = upsert_analysis_result_store(
        store,
        "ca",
        {"message": "ca result", "plot": None, "results": []},
        "CA Analysis",
    )
    store = upsert_analysis_result_store(
        store,
        "cv",
        {"message": "new cv", "plot": None, "results": []},
        "CV Analysis",
    )

    assert list(store) == ["cv", "ca"]
    assert store["cv"]["result"]["message"] == "new cv"
    assert store["ca"]["result"]["message"] == "ca result"


def test_browser_analysis_results_store_renders_one_dropdown_per_technique():
    from ecat_browser.callbacks import render_analysis_results_store, upsert_analysis_result_store

    store = upsert_analysis_result_store(
        {},
        "cv",
        {"message": "new cv", "plot": None, "results": [{"analysis": "peak_current", "status": "ok", "value": 1, "message": ""}]},
        "CV Analysis",
    )
    store = upsert_analysis_result_store(
        store,
        "cp",
        {"message": "cp result", "plot": None, "results": [{"analysis": "cycle_info", "status": "ok", "value": 2, "message": ""}]},
        "CP Analysis",
    )

    rendered = repr(render_analysis_results_store(store))

    assert rendered.count("ecat-analysis-run") == 2
    assert "CV Analysis" in rendered
    assert "CP Analysis" in rendered
    assert "new cv" in rendered
    assert "cp result" in rendered


def test_browser_workflow_updates_with_plot_and_analysis_choices():
    from ecat_browser.callbacks import update_workflow_plot_options, update_workflow_single_analysis

    workflow = update_workflow_plot_options({"source_path": "/data"}, {"legend": True})
    workflow = update_workflow_single_analysis(workflow, 2, ["peak_current"])

    assert workflow["plot_options"] == {"legend": True}
    assert workflow["selected_index"] == 2
    assert workflow["analyses"] == ["peak_current"]


def test_browser_multi_analysis_visibility_and_defaults():
    from ecat_browser.callbacks import multi_analysis_option_state

    hidden = multi_analysis_option_state(None)
    trumpet = multi_analysis_option_state("trumpet_analysis")
    peak = multi_analysis_option_state("fit_peak_current")

    assert hidden[0] == {"display": "none"}
    assert "Trumpet" in trumpet[1]
    assert trumpet[2] == "1, 2"
    assert peak[0] == {}
    assert peak[2] == ""


def test_browser_plot_controls_map_to_ecat_options():
    from ecat_browser.callbacks import plot_options_from_controls

    options = plot_options_from_controls(
        ["legend"],
        "manual",
        "IUPAC",
        "My plot",
        ["grid"],
        ["gradients"],
        ["colorbar", "deduplicate"],
        ["deduplicate"],
        ["trim"],
        -0.2,
        0.4,
        0.12,
        0.05,
        "upper right",
        ["hide_y_numbers"],
        "svg",
        240,
        plot_style="line+markers",
    )

    assert options == {
        "legend": True,
        "legend mode": "colorbar",
        "color mode": "auto",
        "title": "My plot",
        "plot style": "line+markers",
        "plot convention": "IUPAC",
        "grid": True,
        "deduplicate labels": True,
        "potential window": [-0.2, 0.4],
        "offset": 0.12,
        "scale bar": {
            "length": 0.05,
            "loc": "upper right",
            "remove y ticks": True,
        },
        "_format": "svg",
        "_dpi": 240,
    }


def test_browser_plot_controls_use_discrete_legend_when_colorbar_unchecked():
    from ecat_browser.callbacks import plot_options_from_controls

    options = plot_options_from_controls(["legend"], "auto", gradient_values=["gradients"], colorbar_values=[])

    assert options["legend mode"] == "discrete"
    assert options["color mode"] == "auto"
    assert "min gradient entries" not in options


def test_browser_plot_controls_disable_gradient_coloring_when_unchecked():
    from ecat_browser.callbacks import plot_options_from_controls

    options = plot_options_from_controls(["legend"], "auto", gradient_values=[], colorbar_values=["colorbar"])

    assert options["legend mode"] == "discrete"
    assert options["color mode"] == "discrete"


def test_browser_plot_css_centers_images(repo_root):
    css = (repo_root / "apps" / "browser" / "src" / "ecat_browser" / "assets" / "app.css").read_text()

    assert ".ecat-plot" in css
    assert "margin: 0 auto" in css


def test_browser_analysis_controls_align_labels_with_inputs(repo_root):
    css = (repo_root / "apps" / "browser" / "src" / "ecat_browser" / "assets" / "app.css").read_text()

    assert ".ecat-tab-body label" in css
    assert "align-items: center" in css
    assert ".ecat-analysis-checklist label" in css
    assert "display: flex" in css
    assert ".ecat-potential-control-row" in css
    assert ".ecat-stacked-control-label" in css
    assert "align-self: center" in css
    assert "min-height: 44px" in css


def test_browser_sidebar_resize_script_clears_width_when_collapsed(repo_root):
    script = (repo_root / "apps" / "browser" / "src" / "ecat_browser" / "assets" / "sidebar-resize.js").read_text()

    assert "ecat-sidebar-collapsed" in script
    assert "sidebar.style.width = \"\"" in script
    assert "ecatLastWidth" in script


def test_browser_sidebar_tabs_show_icons_above_text_without_empty_header_row(repo_root):
    css = (repo_root / "apps" / "browser" / "src" / "ecat_browser" / "assets" / "app.css").read_text()

    assert ".ecat-sidebar-nav" in css
    assert "position: relative" in css
    assert ".ecat-sidebar-toggle" in css
    assert "height: 64px" in css
    assert "padding-top: 0" in css
    assert ".ecat-tab-label" in css
    assert "flex-direction: column" in css
    assert ".ecat-tab-symbol" in css
    assert "object-fit: contain" in css
    assert "grid-template-columns: minmax(0, 1fr) 44px" in css
    assert "flex: 0 0 25%" in css
    assert "width: 25%" in css
    assert ".tab.tab--selected" in css
    assert "padding: 8px 6px !important" in css
    assert "border-bottom: 1px solid #d9e0e6 !important" in css


def test_browser_plot_controls_accept_open_ended_trim_bounds():
    from ecat_browser.callbacks import plot_options_from_controls

    options = plot_options_from_controls(
        ["legend"],
        "auto",
        trim_values=["trim"],
        trim_max=0.2,
    )

    assert options["potential window"] == [None, 0.2]


def test_browser_plot_title_modes_map_to_ecat_options():
    from ecat_browser.callbacks import plot_options_from_controls

    assert plot_options_from_controls(["legend"], "auto")["title"] == "auto"
    assert plot_options_from_controls(["legend"], "none")["title"] is False
    assert plot_options_from_controls(["legend"], "manual", custom_title="A")["title"] == "A"
    assert plot_options_from_controls(["legend"], "manual")["title"] is True
    assert plot_options_from_controls(["legend"], "auto")["plot style"] == "line"


def test_browser_display_plot_options_ignore_save_format_and_dpi():
    from ecat_browser.callbacks import display_plot_options

    display = display_plot_options({"_format": "svg", "_dpi": 300, "legend": True})

    assert display["_format"] == "png"
    assert display["_dpi"] == 150
    assert display["legend"] is True


def test_browser_plot_control_visibility_depends_on_modes():
    from ecat_browser.callbacks import plot_control_visibility

    visible = plot_control_visibility(["legend"], ["gradients"], "manual")
    no_gradients = plot_control_visibility(["legend"], [], "auto")
    hidden = plot_control_visibility([], ["gradients"], "auto")

    assert visible["legend_options"] == {}
    assert visible["colorbar"] == {}
    assert visible["custom_title"] == {}
    assert no_gradients["legend_options"] == {}
    assert no_gradients["colorbar"] == {"display": "none"}
    assert hidden["legend_options"] == {"display": "none"}
    assert hidden["colorbar"] == {"display": "none"}
    assert hidden["custom_title"] == {"display": "none"}


def test_browser_trim_bounds_visibility_depends_on_trim_toggle():
    from ecat_browser.callbacks import trim_bounds_visibility

    assert trim_bounds_visibility(["trim"]) == {}
    assert trim_bounds_visibility([]) == {"display": "none"}


def test_browser_offset_controls_visibility_depends_on_offset_value():
    from ecat_browser.callbacks import offset_controls_visibility

    assert offset_controls_visibility(0.12) == {}
    assert offset_controls_visibility("") == {"display": "none"}
    assert offset_controls_visibility(None) == {"display": "none"}
    assert offset_controls_visibility(0) == {"display": "none"}


def test_browser_about_panel_toggles():
    from ecat_browser.callbacks import toggle_about_hidden, toggle_about_state

    assert toggle_about_hidden(None, True) is True
    assert toggle_about_hidden(1, True) is False
    assert toggle_about_hidden(2, False) is True
    assert toggle_about_state(None, True) == (True, "About")
    assert toggle_about_state(1, True) == (False, "X")
    assert toggle_about_state(2, False) == (True, "About")


def test_browser_import_controls_use_placeholders_without_reference_defaults():
    pytest.importorskip("dash")
    from ecat_browser.layout import create_layout

    rendered = repr(create_layout())

    assert "Reference label" in rendered
    assert "Reference guess" in rendered
    assert "Fc/Fc+" not in rendered
    assert "id='ecat-reference-guess', value='auto'" not in rendered
    assert "searchable=False" in rendered


def test_browser_render_options_trim_cv_objects(cv_factory):
    from ecat_browser.figures import prepare_plot_objects_and_options

    obj = cv_factory(name="100mVs_sample_CO2_MeCN_10mM_Fc_run01")

    objects, options = prepare_plot_objects_and_options(
        [obj],
        {"potential window": [-0.2, 0.2], "legend": True},
    )

    assert objects[0] is not obj
    assert "potential window" not in options


def test_browser_trim_uses_expand_and_fills_missing_bounds():
    from ecat_browser.figures import prepare_plot_objects_and_options

    class FakeCV:
        def __init__(self):
            self.trim_options = None

        def x(self):
            return [-0.6, -0.1, 0.4]

        def trim(self, options):
            self.trim_options = dict(options)
            return self

    obj = FakeCV()

    objects, options = prepare_plot_objects_and_options(
        [obj],
        {"potential window": [None, 0.2], "legend": True},
    )

    assert objects == [obj]
    assert options == {"legend": True}
    assert obj.trim_options == {
        "potential window": [-0.6, 0.2],
        "mode": "expand",
    }


def test_browser_offset_multiplot_adds_scale_bar_and_hides_y_tick_labels(monkeypatch):
    from ecat_browser import figures
    import matplotlib.pyplot as plt

    captured = {}

    def fake_multiplot(objects, options):
        captured.update(options)
        fig, ax = plt.subplots()
        ax.plot([0, 1], [0, 1])
        return ax

    monkeypatch.setattr(figures.e, "multiplot", fake_multiplot)

    uri = figures.render_multiplot([object()], {"offset": 0.12})
    fig = plt.gcf()
    ax = fig.axes[0] if fig.axes else None

    assert uri.startswith("data:image/png;base64,")
    assert captured["scale bar"]["length"] == 0.12
    assert captured["scale bar"]["loc"] == "upper left"
    assert captured["scale bar"]["remove y ticks"] is True


def test_browser_offset_multiplot_respects_custom_scale_bar(monkeypatch):
    from ecat_browser import figures
    import matplotlib.pyplot as plt

    captured = {}

    def fake_multiplot(objects, options):
        captured.update(options)
        fig, ax = plt.subplots()
        ax.plot([0, 1], [0, 1])
        return ax

    monkeypatch.setattr(figures.e, "multiplot", fake_multiplot)

    figures.render_multiplot(
        [object()],
        {
            "offset": 0.12,
            "scale bar": {"length": 0.04, "loc": "lower right", "remove y ticks": False},
        },
    )

    assert captured["scale bar"] == {"length": 0.04, "loc": "lower right", "remove y ticks": False}
