import base64
import ast
import sys
from pathlib import Path

import pytest


pytest.importorskip(
    "dash", reason="app tests require the optional ecat-electrochemistry[app] extra"
)


APP_SRC = Path(__file__).resolve().parents[1] / "apps" / "workbench" / "src"
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


def _find_component(component, component_id):
    if getattr(component, "id", None) == component_id:
        return component
    children = getattr(component, "children", None)
    if children is None:
        return None
    if not isinstance(children, (list, tuple)):
        children = [children]
    for child in children:
        found = _find_component(child, component_id)
        if found is not None:
            return found
    return None


def test_browser_package_data_includes_static_assets():
    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    package_data_line = next(
        line.strip()
        for line in pyproject.read_text().splitlines()
        if line.strip().startswith("ecat_app = ")
    )
    package_data = ast.literal_eval(package_data_line.split("=", 1)[1].strip())

    assert "assets/*.css" in package_data
    assert "assets/*.svg" in package_data
    assert "assets/*.js" in package_data


def test_browser_local_path_loading_summarizes_supported_objects(fixtures_dir, tmp_path):
    from ecat_app.adapters import load_local_path, summarize_objects

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
    from ecat_app.adapters import load_uploaded_files, summarize_objects

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
    from ecat_app.adapters import load_uploaded_files, reload_workflow

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
    from ecat_app import adapters

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
    from ecat_app.adapters import load_local_path, run_single_cv_analysis

    cp_obj = load_local_path(fixtures_dir / "ch_cp_tiny.txt").objects[0]

    result = run_single_cv_analysis(cp_obj)

    assert result["status"] == "skipped"
    assert "CV" in result["message"]


def test_browser_single_cv_analysis_supports_half_metrics(cv_factory):
    from ecat_app.adapters import run_single_cv_analysis

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
    from ecat_app.adapters import run_single_cv_analysis
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
    assert obj.options["print"] is True


def test_browser_single_cv_analysis_passes_guess_and_tangent_potentials():
    from ecat_app.adapters import run_single_cv_analysis
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


def test_browser_single_cv_analysis_passes_manual_segment():
    from ecat_app.adapters import run_single_cv_analysis
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

    run_single_cv_analysis(obj, ["peak_current"], {"segment": 2})

    assert obj.options["segment"] == 2


def test_browser_single_cv_analysis_captures_printed_output():
    from ecat_app.adapters import run_single_cv_analysis
    import matplotlib.pyplot as plt

    class cv:
        def plot(self, options):
            plt.figure()
            plt.plot([0, 1], [1, 0])

        def peak_potential(self, options):
            print("Peak potential: 0.10 V")
            plt.plot([0, 1], [0, 1])
            return {"Ep": 0.1}

    result = run_single_cv_analysis(cv(), ["peak_potential"])

    assert result["results"][0]["output"] == "Peak potential: 0.10 V"


def test_browser_single_cv_analysis_uses_display_table_from_result():
    import pandas as pd

    from ecat_app.adapters import run_single_cv_analysis

    class FakeCVResult(dict):
        def __init__(self):
            super().__init__({"Ep": 0.1, "index": 4, "current": 2.5})
            self.table = pd.DataFrame(
                [
                    {"Metric": "Ep", "Value": "0.1 V"},
                    {"Metric": "Segment", "Value": 1},
                ]
            )

    class cv:
        def plot(self, options):
            return None

        def peak_potential(self, options):
            return FakeCVResult()

    result = run_single_cv_analysis(cv(), ["peak_potential"])

    assert result["results"][0]["value"] == [
        {"Metric": "Ep", "Value": "0.1 V"},
        {"Metric": "Segment", "Value": 1},
    ]
    assert "index" not in result["results"][0]["value"][0]


def test_browser_single_cv_analysis_returns_plot_image():
    from ecat_app.adapters import run_single_cv_analysis
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
    from ecat_app import callbacks
    from ecat_app.callbacks import handle_single_cv
    from ecat_app.state import SessionRegistry

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
    from ecat_app.callbacks import analysis_index_defaults, handle_single_cv, handle_single_object_analysis
    from ecat_app.state import SessionRegistry

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


def test_browser_single_cv_segment_control_state_uses_selected_cv_segments():
    from ecat_app.callbacks import single_cv_segment_control_state
    from ecat_app.state import SessionRegistry

    class cv:
        def __init__(self, segments):
            self.segments = segments

    registry = SessionRegistry()
    dataset_id = registry.put([cv(1), cv(3), cv(None)])
    state = {"dataset_id": dataset_id}

    one = single_cv_segment_control_state(state, "0", registry=registry)
    three = single_cv_segment_control_state(state, "1", registry=registry)
    unknown = single_cv_segment_control_state(state, "2", registry=registry)

    assert one["slider_disabled"] is True
    assert three["slider_max"] == 3
    assert three["slider_marks"] == {1: "1", 2: "2", 3: "3"}
    assert unknown["slider_style"] == {"display": "none"}
    assert unknown["text_style"] == {}


def test_browser_single_cv_segment_option_from_controls():
    from ecat_app.callbacks import single_cv_options_from_controls

    assert single_cv_options_from_controls(2, "") == {"segment": 2}
    assert single_cv_options_from_controls(None, "3") == {"segment": 3}


def test_browser_analysis_cards_open_for_loaded_techniques():
    from ecat_app.callbacks import analysis_card_open_state

    assert analysis_card_open_state({"summary": [{"class": "cv"}]}) == (True, False, False)
    assert analysis_card_open_state({"summary": [{"class": "ca"}, {"class": "cp"}]}) == (False, True, True)
    assert analysis_card_open_state({"summary": [{"class": "cv"}, {"class": "ca"}, {"class": "cp"}]}) == (True, True, True)
    assert analysis_card_open_state({"summary": []}) == (False, False, False)


def test_browser_model_main_cards_open_only_on_model_tab():
    from ecat_app.callbacks import model_main_cards_open_state

    assert model_main_cards_open_state("data") == (False, False)
    assert model_main_cards_open_state("plot") == (False, False)
    assert model_main_cards_open_state("analyze") == (False, False)
    assert model_main_cards_open_state("model") == (True, True)


def test_browser_ca_analysis_matches_quickstart_options():
    from ecat_app.adapters import run_ca_analysis
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
                "plot style": "notebook",
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
    assert "plot style" not in obj.plot_options[0]
    assert "deduplicate labels" not in obj.plot_options[0]
    assert obj.plot_options[1]["plot charge"] is True
    assert obj.plot_options[1]["charge color"] == "tab:red"
    assert obj.plot_options[1]["grid"] is True
    assert "plot style" not in obj.plot_options[1]
    assert "deduplicate labels" not in obj.plot_options[1]
    assert obj.charge_options[0]["grid"] is True
    assert "plot style" not in obj.charge_options[0]
    assert "deduplicate labels" not in obj.charge_options[0]
    assert obj.charge_options[1]["baseline correction"] is True
    assert obj.charge_options[1]["baseline tail fraction"] == 0.05
    assert obj.charge_options[1]["grid"] is True
    assert "plot style" not in obj.charge_options[1]
    assert "deduplicate labels" not in obj.charge_options[1]
    assert obj.time_options["target charge"] == 0.75
    assert obj.time_options["plot ca"] is True
    assert obj.time_options["legend"] is True
    assert obj.time_options["grid"] is True
    assert "plot style" not in obj.time_options
    assert "deduplicate labels" not in obj.time_options
    assert result["plot"].startswith("data:image/png;base64,")


def test_browser_cp_analysis_runs_stats_cycle_info_and_cycles_plot():
    from ecat_app.adapters import run_cp_analysis
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
                "plot style": "notebook",
                "deduplicate labels": True,
            },
        },
    )

    assert result["status"] == "ok"
    assert [row["analysis"] for row in result["results"]] == ["stats", "cycle_info", "plot", "cycling_plot", "plot_cycles"]
    assert obj.cycle_info_options[0]["percent capacity"] is False
    assert obj.plot_options[0]["legend"] is True
    assert obj.plot_options[0]["grid"] is True
    assert "plot style" not in obj.plot_options[0]
    assert "deduplicate labels" not in obj.plot_options[0]
    assert obj.cycling_plot_options[0]["capacity mode"] == "both"
    assert obj.cycling_plot_options[0]["efficiency mode"] == "both"
    assert obj.cycling_plot_options[0]["grid"] is True
    assert "plot style" not in obj.cycling_plot_options[0]
    assert "deduplicate labels" not in obj.cycling_plot_options[0]
    assert obj.plot_cycles_options[0]["cycles"] == (1, 100, 10)
    assert obj.plot_cycles_options[0]["segment"] == "both"
    assert obj.plot_cycles_options[0]["x axis"] == "capacity"
    assert obj.plot_cycles_options[0]["legend mode"] == "colorbar"
    assert obj.plot_cycles_options[0]["color mode"] == "auto"
    assert obj.plot_cycles_options[0]["grid"] is True
    assert "plot style" not in obj.plot_cycles_options[0]
    assert "deduplicate labels" not in obj.plot_cycles_options[0]
    assert result["plots"][0]["label"] == "Potential Plot"
    assert result["plots"][1]["label"] == "Cycling Performance"
    assert result["plots"][2]["label"] == "Cycle Plot"
    assert all(plot["src"].startswith("data:image/png;base64,") for plot in result["plots"])


def test_browser_multi_cv_analysis_dispatches_public_ecat_function(monkeypatch):
    from ecat_app.adapters import run_multi_cv_analysis

    calls = []

    def fake_fit_peak_current(cvs, options):
        calls.append((cvs, dict(options)))
        return {"fit": "ok"}

    import ecat_app.adapters as adapters

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


def test_browser_multi_cv_analysis_captures_diagnostics_and_output_plots(monkeypatch):
    from ecat_app.adapters import run_multi_cv_analysis
    import matplotlib.pyplot as plt

    def fake_fit_peak_current(cvs, options):
        plt.figure()
        plt.plot([0, 1], [0, 1])
        plt.figure()
        plt.plot([0, 1], [1, 0])
        return {"fit": "ok"}

    import ecat_app.adapters as adapters

    monkeypatch.setattr(adapters.e, "fit_peak_current", fake_fit_peak_current)

    class cv:
        pass

    result = run_multi_cv_analysis([cv(), cv()], "fit_peak_current", {})

    assert result["status"] == "ok"
    assert [plot["label"] for plot in result["plots"]] == ["Diagnostic 1", "Output"]
    assert all(plot["src"].startswith("data:image/png;base64,") for plot in result["plots"])
    assert result["plot"] == result["plots"][-1]["src"]


def test_browser_multi_cv_options_filter_sevcik_unsupported_fit_options():
    from ecat_app.callbacks import multi_cv_options_from_controls

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


def test_browser_model_custom_setup_controls_build_spatial_params():
    from ecat_app.callbacks import (
        model_setup_custom_visibility,
        model_setup_parameter_rows_from_controls,
        model_viscosity_custom_visibility,
        model_viscosity_dropdown_options,
    )

    hidden = model_setup_custom_visibility("fast")
    visible = model_setup_custom_visibility("custom")
    custom_rows = model_setup_parameter_rows_from_controls(
        "custom",
        spatial_dx_fraction="0.002",
        spatial_nx="16",
        spatial_viscosity="2e-6",
        spatial_rotation="3",
        spatial_viscosity_source="custom",
    )
    mecn_rows = model_setup_parameter_rows_from_controls(
        "custom",
        spatial_dx_fraction="0.002",
        spatial_nx="16",
        spatial_viscosity="2e-6",
        spatial_rotation="3",
        spatial_viscosity_source="MeCN",
    )
    fast_rows = model_setup_parameter_rows_from_controls("fast")

    assert hidden == {"display": "none"}
    assert visible == {}
    assert model_viscosity_custom_visibility("custom") == {}
    assert model_viscosity_custom_visibility("MeCN") == {"display": "none"}
    assert {"label": "MeCN (4.7e-7 m²/s)", "value": "MeCN"} in model_viscosity_dropdown_options()
    assert {"label": "Custom", "value": "custom"} in model_viscosity_dropdown_options()
    assert {row["key"]: row["initial"] for row in custom_rows} == {
        "spatial.dx_fraction": 0.002,
        "spatial.nx": 16,
        "spatial.viscosity": 2e-06,
        "spatial.rotation": 3.0,
    }
    assert {row["key"]: row["initial"] for row in mecn_rows} == {
        "spatial.dx_fraction": 0.002,
        "spatial.nx": 16,
        "spatial.rotation": 3.0,
        "spatial.solvent": "MeCN",
    }
    assert fast_rows == [
        {"key": "spatial", "path": "spatial", "name": "Spatial grid", "initial": "fast", "unit": "", "lower": "", "upper": "", "vary": False}
    ]


def test_browser_multi_cv_options_include_fowa_notebook_07_controls():
    from ecat_app.callbacks import multi_cv_options_from_controls

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
    from ecat_app.callbacks import multi_cv_options_from_controls

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


def test_browser_multi_cv_preprocessing_options_scale_then_dimensionless():
    from ecat_app.callbacks import multi_cv_preprocessing_from_controls

    options = multi_cv_preprocessing_from_controls(
        scale_values=["scale"],
        scale_type="manual",
        scale_factor="2",
        normalize_mode="dimensionless",
        dimensionless_mode="heterogeneous",
        dimensionless_e0="-0.5",
        dimensionless_n="2",
        dimensionless_temperature="298",
        dimensionless_d="1e-5",
        dimensionless_c="0.1",
        dimensionless_area_mode="radius_mm",
        dimensionless_area="1.5",
    )

    assert list(options) == ["scale current", "normalize"]
    assert options["scale current"]["scale"] == 2.0
    assert options["normalize"] == {
        "mode": "dimensionless",
        "options": {
            "mode": "heterogeneous",
            "print": False,
            "E0": -0.5,
            "n": 2.0,
            "temperature": 298.0,
            "D": 1e-5,
            "C": 0.1,
            "C unit": "M",
            "area": pytest.approx(0.0706858347),
        },
    }


def test_browser_multi_cv_preprocessing_options_reference_scale_mode():
    from ecat_app.callbacks import multi_cv_preprocessing_from_controls

    options = multi_cv_preprocessing_from_controls(
        scale_values=["scale"],
        scale_type="reference",
        scale_reference_index="2",
        scale_reference_mode="both",
        scale_segment="1",
        scale_guess_potential="-0.8",
    )

    assert options == {
        "scale current": {
            "print": False,
            "plot all": False,
            "reference index": 2,
            "reference mode": "both",
            "segment": 1,
            "guess potential": -0.8,
        }
    }


def test_browser_multi_cv_preprocessing_options_current_normalization():
    from ecat_app.callbacks import multi_cv_preprocessing_from_controls

    options = multi_cv_preprocessing_from_controls(
        normalize_mode="current",
        current_type="reference",
        current_reference_index="1",
        current_segment="2",
        current_guess_potential="-0.9",
    )

    assert options == {
        "normalize": {
            "mode": "current",
            "options": {
                "print": False,
                "plot all": False,
                "reference index": 1,
                "segment": 2,
                "guess potential": -0.9,
            },
        }
    }


def test_browser_multi_cv_preprocessing_options_manual_current_normalization():
    from ecat_app.callbacks import multi_cv_preprocessing_from_controls

    options = multi_cv_preprocessing_from_controls(
        normalize_mode="current",
        current_type="manual",
        current_ip0="4.2e-5",
        current_reference_index="1",
    )

    assert options == {
        "normalize": {
            "mode": "current",
            "options": {
                "print": False,
                "plot all": False,
                "ip0": 4.2e-5,
            },
        }
    }


def test_browser_multi_cv_preprocessing_visibility_toggles_reference_and_manual_fields():
    from ecat_app.callbacks import multi_cv_preprocessing_visibility

    hidden = multi_cv_preprocessing_visibility([], "none", "reference", "reference")
    scale_manual = multi_cv_preprocessing_visibility(["scale"], "current", "manual", "manual")

    assert hidden == (
        {"display": "none"},
        {},
        {"display": "none"},
        {"display": "none"},
        {"display": "none"},
        {},
        {"display": "none"},
    )
    assert scale_manual == (
        {},
        {"display": "none"},
        {},
        {"display": "none"},
        {},
        {"display": "none"},
        {},
    )


def test_browser_single_cv_dimensionless_normalization_options():
    from ecat_app.callbacks import single_cv_dimensionless_normalization_from_controls

    options = single_cv_dimensionless_normalization_from_controls(
        ["dimensionless"],
        mode="homogeneous",
        e0="-0.2",
        n="1",
        temperature="298",
        c="0.25",
        area_mode="area_cm2",
        area="0.071",
    )

    assert options == {
        "dimensionless normalization": {
            "mode": "homogeneous",
            "print": False,
            "E0": -0.2,
            "n": 1.0,
            "temperature": 298.0,
            "C": 0.25,
            "C unit": "M",
            "area": 0.071,
        }
    }


def test_browser_multi_cv_fowa_resolves_reference_index(monkeypatch):
    from ecat_app.adapters import run_multi_cv_analysis

    calls = []

    def fake_fowa(cvs, options):
        calls.append((cvs, dict(options)))
        return [{"FOWA": "ok"}]

    import ecat_app.adapters as adapters

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


def test_browser_multi_cv_analysis_applies_preprocessing_before_dispatch(monkeypatch):
    import ecat_app.callbacks as callbacks

    events = []

    class cv:
        pass

    raw = cv()
    scaled = cv()
    normalized = cv()

    class Registry:
        def get_by_row_ids(self, dataset_id, row_ids):
            return [raw]

        def get(self, dataset_id):
            return [raw]

    def fake_scale_current(cvs, options):
        events.append(("scale", cvs, dict(options)))
        return [scaled]

    def fake_normalize_current(cvs, options):
        events.append(("normalize_current", cvs, dict(options)))
        return [normalized]

    def fake_run_multi_cv_analysis(objects, analysis, options, all_objects=None):
        events.append(("analysis", objects, dict(options)))
        return {"analysis": analysis, "status": "ok", "message": "", "value": None}

    monkeypatch.setattr(callbacks.e, "scale_current", fake_scale_current)
    monkeypatch.setattr(callbacks.e, "normalize_current", fake_normalize_current)
    monkeypatch.setattr(callbacks, "run_multi_cv_analysis", fake_run_multi_cv_analysis)

    result = callbacks.handle_multi_cv_analysis(
        "dataset",
        ["row-0"],
        [{"id": "row-0"}],
        "fit_peak_current",
        {"plot all": True, "preprocessing": {"app": "only"}},
        {
            "scale current": {"scale": 2.0, "print": False, "plot all": False},
            "normalize": {"mode": "current", "options": {"reference index": 0, "print": False}},
        },
        registry=Registry(),
    )

    assert result["status"] == "ok"
    assert [event[0] for event in events] == ["scale", "normalize_current", "analysis"]
    assert events[0][1] == [raw]
    assert events[1][1] == [scaled]
    assert events[2][1] == [normalized]
    assert "preprocessing" not in events[2][2]


def test_browser_multi_cv_none_analysis_renders_preprocessing_plot(monkeypatch):
    import ecat_app.callbacks as callbacks

    events = []

    class cv:
        pass

    raw = cv()
    scaled = cv()

    class Registry:
        def get_by_row_ids(self, dataset_id, row_ids):
            return [raw]

        def get(self, dataset_id):
            return [raw]

    def fake_scale_current(cvs, options):
        events.append(("scale", cvs, dict(options)))
        return [scaled]

    def fake_render_multiplot(objects, options=None):
        events.append(("plot", objects, options))
        return "data:image/png;base64,preprocessed"

    def fail_run_multi_cv_analysis(*args, **kwargs):
        raise AssertionError("None analysis should not call an analysis function")

    monkeypatch.setattr(callbacks.e, "scale_current", fake_scale_current)
    monkeypatch.setattr(callbacks, "render_multiplot", fake_render_multiplot)
    monkeypatch.setattr(callbacks, "run_multi_cv_analysis", fail_run_multi_cv_analysis)

    result = callbacks.handle_multi_cv_analysis(
        "dataset",
        ["row-0"],
        [{"id": "row-0"}],
        "none",
        {"preprocessing": {"app": "only"}},
        {"scale current": {"scale": 2.0, "print": False, "plot all": False}},
        registry=Registry(),
    )

    assert result["status"] == "ok"
    assert result["analysis"] == "none"
    assert result["plot"] == "data:image/png;base64,preprocessed"
    assert result["plots"] == [{"label": "Pre-processing", "src": "data:image/png;base64,preprocessed"}]
    assert [event[0] for event in events] == ["scale", "plot"]
    assert events[1][1] == [scaled]


def test_browser_multi_cv_tafel_uses_positional_public_call(monkeypatch):
    from ecat_app.adapters import run_multi_cv_analysis

    calls = []

    def fake_tafel_analysis(cv_obj, tof_max, thermodynamic_potential, redox_potential, options):
        calls.append((cv_obj, tof_max, thermodynamic_potential, redox_potential, dict(options)))
        return {"tafel": "ok"}

    import ecat_app.adapters as adapters

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
    from ecat_app.adapters import run_multi_cv_analysis

    result = run_multi_cv_analysis([object()], "made_up_analysis", {})

    assert result["status"] == "skipped"
    assert "Unsupported" in result["message"]


def test_browser_multi_cv_results_include_plot():
    from ecat_app.callbacks import render_multi_cv_results

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
    from ecat_app.references import build_reference_options

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
    from ecat_app.callbacks import reference_file_options

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
    from ecat_app.callbacks import reference_file_path_from_index
    from ecat_app.state import SessionRegistry

    reference = cv_factory(name="100mVs_reference")
    reference.filepath = str(tmp_path / "reference.txt")
    registry = SessionRegistry()
    dataset_id = registry.put([reference], [])

    resolved = reference_file_path_from_index({"dataset_id": dataset_id}, 0, registry=registry)

    assert resolved == str(tmp_path / "reference.txt")


def test_browser_local_reload_applies_manual_reference_options(fixtures_dir, tmp_path):
    from ecat_app.adapters import load_local_path, reload_workflow
    from ecat_app.references import build_reference_options

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
    from ecat_app.codegen import generate_python
    from ecat_app.workflow import AppWorkflow

    workflow = AppWorkflow(
        source_kind="local_path",
        source_path="/data/cvs",
        recursive=True,
        selected_index=0,
        filters={"gas": "CO2"},
        group_keys=["gas"],
        sort_keys=["scan rate"],
        analyses=["peak_potential", "peak_current"],
        export_filename="processed_cv",
        plot_options={"legend": True, "title": "auto", "plot convention": "IUPAC", "x label": "Potential / V", "y label": "Current / A"},
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
    assert '"x label"' not in code
    assert '"y label"' not in code
    assert 'overlay_ax.set_xlabel("Potential / V")' in code
    assert 'overlay_ax.set_ylabel("Current / A")' in code
    assert "overlay_ax = e.multiplot" in code
    assert code.index("overlay_ax = e.multiplot") < code.index("peak_potential_result")
    assert "e.save_data" in code
    assert '"reference mode": "manual"' in code
    assert "included_indices = [0, 2]" in code
    assert "dash" not in code.lower()
    assert "ecat_app" not in code


def test_browser_codegen_emits_multi_cv_preprocessing_before_plot():
    from ecat_app.codegen import generate_python
    from ecat_app.workflow import AppWorkflow

    workflow = AppWorkflow(
        included_row_ids=["row-0", "row-1"],
        analyses=["fit_peak_current"],
        plot_options={
            "fit_peak_current options": {
                "fit": True,
                "preprocessing": {
                    "scale current": {"scale": 2.0, "print": False, "plot all": False},
                    "normalize": {
                        "mode": "current",
                        "options": {"reference index": 0, "print": False, "plot all": False},
                    },
                },
            }
        },
    )

    code = generate_python(workflow)

    assert "scaled = e.scale_current(included" in code
    assert "normalized = e.normalize_current(scaled" in code
    assert "overlay_ax = e.multiplot(normalized, plot_options)" in code
    assert code.index("e.scale_current") < code.index("e.normalize_current") < code.index("e.multiplot")
    assert '"preprocessing"' not in code
    assert "dash" not in code.lower()
    assert "ecat_app" not in code


def test_browser_codegen_emits_animation_after_multiplot():
    from ecat_app.codegen import generate_python
    from ecat_app.workflow import AppWorkflow

    workflow = AppWorkflow(
        included_row_ids=["row-0", "row-1"],
        plot_options={
            "legend": True,
            "title": "auto",
            "_animate": True,
            "_format": "gif",
            "fps": 20,
            "stride": 3,
            "trace mode": "draw",
            "schedule": "sequential",
            "timing mode": "normalized",
            "normalized duration": 2,
            "end hold": 2,
            "include quiet time": False,
        },
    )

    code = generate_python(workflow)

    assert "overlay_ax = e.multiplot(included, plot_options)" in code
    assert "animation = e.animate(included, animation_options)" in code
    assert code.index("overlay_ax = e.multiplot") < code.index("animation = e.animate")
    assert '"stride": 3' in code
    assert '"fps": 20' in code
    assert "_animate" not in code
    assert "ecat_app" not in code


def test_browser_codegen_emits_model_setup_and_simulation_calls():
    from ecat_app.codegen import generate_python
    from ecat_app.workflow import AppWorkflow

    workflow = AppWorkflow(
        model_options={
            "mechanism_source": "custom",
            "mechanism": "E(1):Fe2=Fe1\nC:Fe1>Fe0",
            "mechanism_valid": True,
            "simulate_mode": "scratch",
            "simulation_ready": True,
            "program_settings": {
                "Ei": 0.0,
                "E_low": -1.5,
                "E_high": None,
                "scan_rate": 0.1,
                "segments": 2,
                "points_per_segment": 300,
                "quiet_time": 0.0,
            },
            "simulation_params": {
                "concentrations": {"bulk": {"Fe2": 1e-3, "Fe1": 0.0}},
                "diffusion": {"Fe2": 1e-9, "Fe1": 1e-9},
                "kinetics": [{"E0": 0.0, "k0": 1e-3, "alpha": 0.5}],
                "cell": {"T": 298.15, "Ru": 0.0, "Cdl": 0.0, "A": 1e-5},
                "spatial": "fast",
            },
            "fit_mode": "multiple",
            "fit_requested": True,
            "simulation_result": {"status": "ok", "message": "Simulation complete."},
            "fit_result": {"status": "ok", "message": "Fit requested."},
        }
    )

    code = generate_python(workflow)

    assert "# Model Setup" in code
    assert 'mechanism_text = "E(1):Fe2=Fe1\\nC:Fe1>Fe0"' in code
    assert "compiled_mechanism = e.simulation.compile_mechanism(mechanism_text)" in code
    assert "program_settings =" in code
    assert "cv_data_settings =" in code
    assert "cell_parameters =" in code
    assert "# Simulation" in code
    assert 'simulation_mode = "scratch"' in code
    assert "simulation_input = e.simulation.cv_program(" in code
    assert "simulation_result = e.simulation.simulate_cv(" in code
    assert "simulation_params =" in code
    assert "# TODO: run e.simulation" not in code
    assert "# Fit" in code
    assert 'fit_mode = "multiple"' in code
    assert "mechanism_source" not in code
    assert "simulation_ready" not in code
    assert "ecat_app" not in code


def test_browser_codegen_waits_for_model_run_results_before_execution_calls():
    from ecat_app.codegen import generate_python
    from ecat_app.workflow import AppWorkflow

    workflow = AppWorkflow(
        model_options={
            "mechanism": "E",
            "mechanism_valid": True,
            "simulate_mode": "scratch",
            "simulation_ready": True,
            "program_settings": {"Ei": 0.0, "E_low": -1.0, "scan_rate": 0.1},
            "simulation_params": {
                "concentrations": {"bulk": {"A": 1e-3, "B": 0.0}},
                "diffusion": {"A": 1e-9, "B": 1e-9},
                "kinetics": [{"E0": 0.0, "k0": 1e-3, "alpha": 0.5}],
            },
            "fit_requested": True,
            "fit_spec": {"vary": ["kinetics.0.E0"]},
        }
    )

    code = generate_python(workflow)

    assert "# Model Setup" in code
    assert "# CV Program" not in code
    assert "# Simulation" not in code
    assert "simulation_result = e.simulation.simulate_cv(" not in code
    assert "# Fit" not in code
    assert "fit_result = e.simulation.fit_cv(" not in code


def test_browser_codegen_emits_model_program_after_program_run():
    from ecat_app.codegen import generate_python
    from ecat_app.workflow import AppWorkflow

    workflow = AppWorkflow(
        model_options={
            "mechanism": "E",
            "mechanism_valid": True,
            "program_settings": {"Ei": 0.0, "E_low": -1.0, "scan_rate": 0.1, "segments": 2},
            "program_result": {"status": "ok", "message": "CV program plotted."},
        }
    )

    code = generate_python(workflow)

    assert "# CV Program" in code
    assert "program_input = e.simulation.cv_program(" in code
    assert "program_ax = program_input.plot" in code
    assert "# Simulation" not in code
    assert "simulation_result = e.simulation.simulate_cv(" not in code
    assert "# Fit" not in code


def test_browser_codegen_emits_model_single_cv_fit_call():
    from ecat_app.codegen import generate_python
    from ecat_app.workflow import AppWorkflow

    workflow = AppWorkflow(
        model_options={
            "mechanism": "E",
            "mechanism_valid": True,
            "simulate_mode": "cv",
            "simulation_ready": True,
            "cv_data_settings": {"cv_index": 2, "stride": 5, "segment": 2},
            "simulation_params": {
                "concentrations": {"bulk": {"a": 1e-3, "b": 0.0}},
                "diffusion": {"a": 1e-9, "b": 1e-9},
                "kinetics": [{"E0": -0.3, "k0": 1e-3, "alpha": 0.5}],
                "cell": {"T": 298.15, "Cdl": 1e-6},
            },
            "fit_requested": True,
            "fit_mode": "single",
            "fit_cv_index": 3,
            "fit_spec": {
                "vary": ["kinetics.0.E0"],
                "bounds": {"kinetics.0.E0": [-1.2, -0.2]},
            },
            "simulation_result": {"status": "ok", "message": "Simulation complete."},
            "fit_result": {"status": "ok", "message": "Fit complete."},
        }
    )

    code = generate_python(workflow)

    assert "# Fit" in code
    assert 'fit_mode = "single"' in code
    assert "fit_spec =" in code
    assert "fit_cv_index = 3" in code
    assert "fit_cv_data_options = dict(cv_data_settings)" in code
    assert "fit_cv_data_options.pop('cv_index', None)" in code
    assert "fit_result = e.simulation.fit_cv(" in code
    assert "data[fit_cv_index]," in code
    assert "fit=fit_spec," in code
    assert '"cv data": fit_cv_data_options,' in code
    assert "# TODO: fit imported CV data" not in code
    assert "ecat_app" not in code


def test_browser_codegen_emits_model_condition_sweep_loop():
    from ecat_app.codegen import generate_python
    from ecat_app.workflow import AppWorkflow

    workflow = AppWorkflow(
        model_options={
            "mechanism": "E",
            "mechanism_valid": True,
            "simulate_mode": "scratch",
            "simulation_ready": True,
            "over_conditions": True,
            "condition_settings": {
                "condition_axis": "temperature",
                "condition_min": 280,
                "condition_max": 300,
                "condition_count": 3,
            },
            "program_settings": {"Ei": 0.0, "E_low": -1.0, "scan_rate": 0.1},
            "simulation_params": {
                "concentrations": {"bulk": {"a": 1e-3, "b": 0.0}},
                "diffusion": {"a": 1e-9, "b": 1e-9},
                "kinetics": [{"E0": 0.0, "k0": 1e-3, "alpha": 0.5}],
                "cell": {"T": 298.15},
            },
            "simulation_result": {"status": "ok", "message": "Condition sweep complete."},
        }
    )

    code = generate_python(workflow)

    assert "condition_settings =" in code
    assert "condition_min" in code
    assert "condition_values = [" in code
    assert "range(condition_count)" in code
    assert "for condition_value in condition_values:" in code
    assert "condition_params.setdefault('cell', {})['T'] = condition_value" in code
    assert "simulation_results.append(" in code
    assert "simulation_result = e.simulation.simulate_cv(" not in code
    assert "ecat_app" not in code


def test_browser_codegen_emits_completed_chrono_analysis_actions():
    from ecat_app.codegen import generate_python
    from ecat_app.workflow import AppWorkflow

    workflow = AppWorkflow(
        analysis_actions=[
            {
                "kind": "ca",
                "selected_index": 2,
                "analyses": ["charge", "time_at_charge"],
                "options": {
                    "target charge": 0.5,
                    "plot ca": True,
                    "baseline tail fraction": 0.1,
                    "plot options": {"grid": True},
                },
            },
            {
                "kind": "cp",
                "selected_index": 3,
                "analyses": ["cycle_info", "plot_cycles"],
                "options": {
                    "percent capacity": True,
                    "cycles": (1, 5, 1),
                    "segment": "charge",
                    "x axis": "time",
                    "plot options": {"grid": True},
                },
            },
        ]
    )

    code = generate_python(workflow)

    assert "# CA Analysis" in code
    assert "ca_obj = data[2]" in code
    assert "ca_charge_result = ca_obj.charge(" in code
    assert "ca_time_at_charge_result = ca_obj.time_at_charge(" in code
    assert '"target charge": 0.5' in code
    assert "# CP Analysis" in code
    assert "cp_obj = data[3]" in code
    assert "cp_cycle_info_result = cp_obj.cycle_info(" in code
    assert "cp_plot_cycles_result = cp_obj.plot_cycles(" in code
    assert '"cycles": (1, 5, 1)' in code


def test_browser_notebook_export_uses_code_preview_and_visible_results():
    import json

    from ecat_app.notebook import generate_notebook

    code = "import ecat as e\n\n# Load Data\ndata = e.get_data({})\n\n# Plot\ne.multiplot(data, {})\n"
    results_store = {
        "cv": {
            "title": "CV Analysis",
            "result": {
                "results": [
                    {"analysis": "peak_current", "status": "ok", "value": 1.2, "message": ""},
                ],
            },
        }
    }

    notebook_text = generate_notebook(code, results_store)
    notebook = json.loads(notebook_text)

    assert notebook["nbformat"] == 4
    markdown = "\n".join(
        "\n".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell.get("cell_type") == "markdown"
    )
    code_cells = [
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
    ]
    assert "Load Data" in markdown
    assert "Plot" in markdown
    assert "CV Analysis" in markdown
    assert "peak_current" in markdown
    assert "\n\n".join(code_cells).strip() == code.strip()


def test_browser_code_execution_is_disabled_without_explicit_trusted_flag(monkeypatch):
    from ecat_app.execution import code_execution_allowed, run_user_code

    monkeypatch.delenv("ECAT_APP_ALLOW_CODE_EXECUTION", raising=False)
    monkeypatch.delenv("ECAT_BROWSER_ALLOW_CODE_EXECUTION", raising=False)

    assert code_execution_allowed() is False
    result = run_user_code("print('should not run')")
    assert result.executed is False
    assert "disabled" in result.stderr.lower()
    assert "ECAT_APP_ALLOW_CODE_EXECUTION" in result.stderr


def test_browser_code_execution_runs_when_trusted_flag_is_set(monkeypatch, tmp_path):
    from ecat_app.execution import code_execution_allowed, run_user_code

    monkeypatch.setenv("ECAT_APP_ALLOW_CODE_EXECUTION", "1")

    assert code_execution_allowed() is True
    result = run_user_code("print('ran locally')", cwd=tmp_path, timeout_seconds=2)
    assert result.executed is True
    assert result.returncode == 0
    assert result.stdout.strip() == "ran locally"


def test_browser_session_registry_stores_objects_outside_dash_json(ecat_module, cv_factory):
    from ecat_app.state import SessionRegistry

    registry = SessionRegistry()
    obj = cv_factory()

    dataset_id = registry.put([obj], warnings=["note"])
    snapshot = registry.snapshot(dataset_id)

    assert snapshot["dataset_id"] == dataset_id
    assert snapshot["warnings"] == ["note"]
    assert snapshot["summary"][0]["class"] == "cv"
    assert registry.get(dataset_id)[0] is obj


def test_browser_session_registry_returns_included_objects_by_stable_row_id(ecat_module, cv_factory):
    from ecat_app.state import SessionRegistry

    registry = SessionRegistry()
    first = cv_factory(name="50mVs_first_CO2_MeCN")
    second = cv_factory(name="100mVs_second_CO2_MeCN")
    dataset_id = registry.put([first, second])

    included = registry.get_included(dataset_id, ["row-1"])

    assert included == [second]


def test_browser_callback_helpers_return_serializable_import_state(fixtures_dir, tmp_path):
    from ecat_app.callbacks import handle_local_path_load
    from ecat_app.state import SessionRegistry

    _copy_fixture(fixtures_dir, tmp_path, "ch_cv.txt")
    registry = SessionRegistry()

    state = handle_local_path_load(str(tmp_path), recursive=False, registry=registry)

    assert state["dataset_id"]
    assert state["summary"][0]["class"] == "cv"
    assert state["included_row_ids"] == ["row-0"]
    assert state["workflow"]["source_kind"] == "local_path"
    assert "import ecat as e" in state["code"]
    assert registry.get(state["dataset_id"])


def test_browser_empty_local_path_state_has_import_callback_shape():
    from ecat_app.callbacks import handle_local_path_load

    state = handle_local_path_load("")

    assert state["included_row_ids"] == []
    assert state["table"] == {"data": [], "columns": []}
    assert state["summary"] == []
    assert state["plot"] is None
    assert state["status"] == ""


def test_browser_table_handles_empty_object_list():
    from ecat_app.table import available_column_options, build_browser_table, default_visible_columns

    assert default_visible_columns([]) == []
    assert available_column_options([]) == [
        {"label": "Filename", "value": "Filename"},
        {"label": "Class", "value": "Class"},
        {"label": "Software", "value": "Software"},
        {"label": "Timestamp", "value": "Timestamp"},
        {"label": "Creation Time", "value": "Creation Time"},
        {"label": "Temperature", "value": "Temperature"},
        {"label": "Electrode Area", "value": "Electrode Area"},
        {"label": "IR Comp Resistance", "value": "IR Comp Resistance"},
        {"label": "IR Uncomp Resistance", "value": "IR Uncomp Resistance"},
        {"label": "IR Comp Percent", "value": "IR Comp Percent"},
    ]
    assert build_browser_table([]) == {"columns": [{"name": "index", "id": "index"}], "data": []}


def test_browser_cli_open_browser_helper_respects_enabled_flag(monkeypatch):
    import ecat_app.app as browser_app

    opened = []
    timers = []

    class FakeTimer:
        def __init__(self, delay, func, args=None):
            self.delay = delay
            self.func = func
            self.args = list(args or [])
            self.daemon = False
            timers.append(self)

        def start(self):
            self.func(*self.args)

    monkeypatch.setattr(browser_app.threading, "Timer", FakeTimer)
    monkeypatch.setattr(browser_app.webbrowser, "open_new_tab", lambda url: opened.append(url))

    timer = browser_app._open_browser_later("http://127.0.0.1:8050/", enabled=True, delay=0.25)
    disabled = browser_app._open_browser_later("http://127.0.0.1:8051/", enabled=False)

    assert timer is timers[0]
    assert timers[0].delay == 0.25
    assert timers[0].daemon is True
    assert disabled is None
    assert opened == ["http://127.0.0.1:8050/"]


def test_browser_cli_env_flag_parses_false_values(monkeypatch):
    import ecat_app.app as browser_app

    monkeypatch.delenv("ECAT_APP_OPEN", raising=False)
    monkeypatch.delenv("ECAT_BROWSER_OPEN", raising=False)
    assert browser_app._env_flag("ECAT_APP_OPEN", True, legacy_name="ECAT_BROWSER_OPEN") is True
    monkeypatch.setenv("ECAT_APP_OPEN", "0")
    assert browser_app._env_flag("ECAT_APP_OPEN", True, legacy_name="ECAT_BROWSER_OPEN") is False
    monkeypatch.setenv("ECAT_APP_OPEN", "no")
    assert browser_app._env_flag("ECAT_APP_OPEN", True, legacy_name="ECAT_BROWSER_OPEN") is False
    monkeypatch.setenv("ECAT_APP_OPEN", "1")
    assert browser_app._env_flag("ECAT_APP_OPEN", False, legacy_name="ECAT_BROWSER_OPEN") is True
    monkeypatch.delenv("ECAT_APP_OPEN", raising=False)
    monkeypatch.setenv("ECAT_BROWSER_OPEN", "0")
    assert browser_app._env_flag("ECAT_APP_OPEN", True, legacy_name="ECAT_BROWSER_OPEN") is False


def test_browser_cli_exposes_only_ecat_app_command(repo_root):
    pyproject = (repo_root / "pyproject.toml").read_text()
    assert 'ecat-app = "ecat_app.app:main"' in pyproject
    assert "ecat-browser =" not in pyproject
    assert "ecat-app-window =" not in pyproject
    assert '"pywebview"' in pyproject
    assert "ecat-browser-server" not in pyproject


def test_browser_cli_defaults_to_native_window(monkeypatch):
    import ecat_app.app as browser_app

    calls = {}

    class FakeApp:
        def run(self, **_kwargs):
            calls["dash_run"] = True

    monkeypatch.setattr(browser_app, "create_app", lambda config=None: FakeApp())
    monkeypatch.setattr(
        browser_app,
        "_run_window_app",
        lambda app, host, port, **kwargs: calls.update(
            {"window": (app, host, port, kwargs)}
        )
        or f"http://{host}:5173/",
    )
    monkeypatch.setattr(browser_app, "_open_browser_later", lambda *_args, **_kwargs: calls.update({"browser": True}))

    result = browser_app.main(["--port", "0", "--width", "1200", "--height", "800"])

    assert calls["window"][1:3] == ("127.0.0.1", 0)
    assert calls["window"][3]["width"] == 1200
    assert calls["window"][3]["height"] == 800
    assert calls["window"][3]["title"] == "eCAT Workbench"
    assert result is None
    assert "dash_run" not in calls
    assert "browser" not in calls


def test_browser_cli_browser_mode_uses_browser_tab(monkeypatch):
    import ecat_app.app as browser_app

    calls = {}

    class FakeApp:
        def run(self, **kwargs):
            calls["dash_run"] = kwargs

    monkeypatch.setattr(browser_app, "create_app", lambda config=None: FakeApp())
    monkeypatch.setattr(browser_app, "_run_window_app", lambda *_args, **_kwargs: calls.update({"window": True}))
    monkeypatch.setattr(browser_app, "_open_browser_later", lambda url, **kwargs: calls.update({"browser": (url, kwargs)}))

    result = browser_app.main(["--browser", "--port", "8123", "--no-open"])

    assert result is None
    assert calls["browser"] == ("http://127.0.0.1:8123/", {"enabled": False})
    assert calls["dash_run"] == {"host": "127.0.0.1", "port": 8123, "debug": False}
    assert "window" not in calls


def test_browser_native_window_runner_starts_server_and_shuts_down(monkeypatch):
    import ecat_app.app as browser_app

    events = []

    class FakeDash:
        server = object()

    class FakeServer:
        server_port = 61234

        def serve_forever(self):
            events.append("serve")

        def shutdown(self):
            events.append("shutdown")

    class FakeThread:
        daemon = False

        def __init__(self, target, daemon=False, name=None):
            self.target = target
            self.daemon = daemon
            self.name = name

        def start(self):
            events.append(("thread", self.daemon, self.name))
            self.target()

        def join(self, timeout=None):
            events.append(("join", timeout))

    class FakeWebview:
        @staticmethod
        def create_window(title, url, width=None, height=None):
            events.append(("window", title, url, width, height))
            return object()

        @staticmethod
        def start():
            events.append("webview-start")

    fake_server = FakeServer()
    monkeypatch.setattr(browser_app, "_make_server", lambda host, port, server: fake_server)
    monkeypatch.setattr(browser_app.threading, "Thread", FakeThread)
    monkeypatch.setattr(browser_app, "_load_webview", lambda: FakeWebview)

    url = browser_app._run_window_app(FakeDash(), "127.0.0.1", 0, title="eCAT Workbench", width=1200, height=800)

    assert url == "http://127.0.0.1:61234/"
    assert events == [
        ("thread", True, "ecat-window-61234"),
        "serve",
        ("window", "eCAT Workbench", "http://127.0.0.1:61234/", 1200, 800),
        "webview-start",
        "shutdown",
        ("join", 2),
    ]


def test_browser_double_click_launchers_exist(repo_root):
    command_launcher = repo_root / "apps" / "workbench" / "launchers" / "eCAT Workbench.command"
    windows_launcher = repo_root / "apps" / "workbench" / "launchers" / "eCAT Workbench.cmd"
    shared_launcher = repo_root / "apps" / "workbench" / "launchers" / "ecat-workbench-launcher.sh"
    app_launcher = (
        repo_root
        / "apps"
        / "workbench"
        / "launchers"
        / "eCAT Workbench.app"
        / "Contents"
        / "MacOS"
        / "ecat-workbench"
    )
    app_launcher_source = repo_root / "apps" / "workbench" / "launchers" / "ecat-workbench-launcher.c"
    legacy_spaced_app_launcher = (
        repo_root
        / "apps"
        / "workbench"
        / "launchers"
        / "eCAT Workbench.app"
        / "Contents"
        / "MacOS"
        / "eCAT Workbench"
    )
    plist = (
        repo_root
        / "apps"
        / "workbench"
        / "launchers"
        / "eCAT Workbench.app"
        / "Contents"
        / "Info.plist"
    )
    app_icon = (
        repo_root
        / "apps"
        / "workbench"
        / "launchers"
        / "eCAT Workbench.app"
        / "Contents"
        / "Resources"
        / "ecat-logo.icns"
    )
    bundled_launcher = (
        repo_root
        / "apps"
        / "workbench"
        / "launchers"
        / "eCAT Workbench.app"
        / "Contents"
        / "Resources"
        / "ecat-workbench-launcher.sh"
    )
    accent_logo = repo_root / "apps" / "workbench" / "src" / "ecat_app" / "assets" / "ecat-logo_2_accent.svg"
    app_icon_source = repo_root / "apps" / "workbench" / "src" / "ecat_app" / "assets" / "ecat-logo-app-icon.svg"

    assert command_launcher.exists()
    assert windows_launcher.exists()
    assert shared_launcher.exists()
    assert app_launcher.exists()
    assert app_launcher_source.exists()
    assert bundled_launcher.exists()
    assert not legacy_spaced_app_launcher.exists()
    assert plist.exists()
    shared_text = shared_launcher.read_text()
    plist_text = plist.read_text()
    assert "ecat-app --port 0" in shared_text
    assert "--window" not in shared_text
    assert "--port 0" in shared_text
    assert "MPLCONFIGDIR" in shared_text
    assert "ecat-workbench-launch.log" in shared_text
    assert "display alert" in shared_text
    assert "/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin" in shared_text
    assert "/Library/Frameworks/Python.framework/Versions" in shared_text
    assert 'pip install -e ".[app]"' in shared_text
    assert "cd \\\"$REPO_ROOT\\\"" in shared_text
    assert "eCAT repository folder" in shared_text
    assert "ecat-workbench-launcher.sh" in command_launcher.read_text()
    app_magic = app_launcher.read_bytes()[:4]
    assert app_magic in {
        b"\xcf\xfa\xed\xfe",
        b"\xfe\xed\xfa\xcf",
        b"\xca\xfe\xba\xbe",
        b"\xbe\xba\xfe\xca",
    }
    app_source_text = app_launcher_source.read_text()
    assert "_NSGetExecutablePath" in app_source_text
    assert "ecat-workbench-launcher.sh" in app_source_text
    assert "bundled_script_path" in app_source_text
    assert "access(source_script_path, X_OK)" in app_source_text
    assert "ECAT_LAUNCHER_ENTRY" in app_source_text
    assert "/bin/zsh" in app_source_text
    windows_text = windows_launcher.read_text()
    assert "ecat-app --port 0" in windows_text
    assert "ecat-app --window" not in windows_text
    assert "ECAT_PYTHON" in windows_text
    assert "apps\\workbench\\app.py" in windows_text
    assert "py -3" in windows_text
    assert "ecat-workbench-launch.log" in windows_text
    assert "System.Windows.MessageBox" in windows_text
    assert 'python -m pip install -e ".[app]"' in windows_text
    assert 'cd /d "%REPO_ROOT%"' in windows_text
    assert "eCAT repository folder" in windows_text
    assert "eCAT Workbench" in plist_text
    assert "<key>CFBundleIdentifier</key>" in plist_text
    assert "<string>org.ecat.workbench.native</string>" in plist_text
    assert "<key>CFBundleExecutable</key>" in plist_text
    assert "<string>ecat-workbench</string>" in plist_text
    assert "<key>CFBundleIconFile</key>" in plist_text
    assert "<string>ecat-logo</string>" in plist_text
    assert app_icon.exists()
    assert app_icon.stat().st_size > 0
    accent_text = accent_logo.read_text().lower()
    assert "#1ba6a6" in accent_text
    assert "#f2a900" in accent_text
    app_icon_text = app_icon_source.read_text().lower()
    assert "#e9eef3" in app_icon_text
    assert "#b8c4cf" not in app_icon_text
    assert 'fill="#e9eef3"/>' in app_icon_text
    assert "#1ba6a6" in app_icon_text
    assert "#f2a900" in app_icon_text
    assert 'transform="translate(96 68) scale(1.38)"' in app_icon_text


def test_browser_table_rows_follow_ecat_build_object_table_columns(cv_factory):
    from ecat_app.table import build_browser_table

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
    from ecat_app.table import build_browser_table

    first = cv_factory(name="50mVs_sample_CO2_MeCN_10mM_Fc_run01")

    table = build_browser_table([first], visible_columns=["Gas", "Solvent"])

    assert table["data"][0]["Gas"] == "CO₂"
    assert table["data"][0]["Solvent"] == "MeCN"


def test_browser_table_displays_reference_source_as_loaded_index(cv_factory, tmp_path):
    from ecat_app.table import build_browser_table

    reference = cv_factory(name="100mVs_reference_Fc")
    target = cv_factory(name="100mVs_target_Fc")
    reference.filepath = str(tmp_path / "reference.txt")
    target.filepath = str(tmp_path / "target.txt")
    target.reference_mode = "file"
    target.reference_source_file = str(tmp_path / "reference.txt")

    table = build_browser_table([reference, target], visible_columns=["Reference Source"])

    assert table["data"][1]["Reference Source"] == 0


def test_browser_table_columns_are_true_visible_selection(cv_factory):
    from ecat_app.table import available_column_options, build_browser_table, default_visible_columns, selected_column_values
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
    from ecat_app.callbacks import _state_from_load_result
    from ecat_app.state import SessionRegistry
    from ecat_app.workflow import AppWorkflow

    class Result:
        objects = [
            cv_factory(name="50mVs_sample_CO2_MeCN_10mM_Fc_run01"),
            cv_factory(name="100mVs_sample_Ar_DMF_10mM_Fc_run02"),
        ]
        warnings = []
        workflow = AppWorkflow()
        status = ""

    state = _state_from_load_result(Result(), registry=SessionRegistry())
    selected_columns = {column["id"] for column in state["table"]["columns"] if column["id"] != "index"}
    available_columns = {option["value"] for option in state["column_options"]}

    assert selected_columns < available_columns
    assert "Temperature" in available_columns
    assert "IR Comp Resistance" in available_columns


def test_browser_default_fe_phoh_load_selects_scan_window_with_ferrocene(cv_factory, tmp_path):
    from ecat_app.adapters import default_included_row_ids
    from ecat_app.workflow import AppWorkflow

    objects = []
    for index in range(12):
        if index in {1, 7, 8, 9, 10, 11}:
            name = f"sample_{index}_3mMFc_-1.2_to_1V_100mVs"
        elif index == 0:
            name = "sample_0_-1.2_to_1V_100mVs"
        else:
            name = f"sample_{index}_3mMFc_-1.7_to_1V_100mVs"
        objects.append(cv_factory(name=name))
    workflow = AppWorkflow(source_path=str(tmp_path / "examples" / "data" / "fe_phoh_cv"))

    included = default_included_row_ids(objects, workflow)

    assert included == ["row-1", "row-7", "row-8", "row-9", "row-10", "row-11"]


def test_browser_loaded_state_includes_pretty_conditions(cv_factory):
    from ecat_app.callbacks import _state_from_load_result
    from ecat_app.state import SessionRegistry
    from ecat_app.workflow import AppWorkflow

    class Result:
        objects = [
            cv_factory(name="50mVs_sample_CO2_MeCN_10mM_Fc_run01"),
            cv_factory(name="100mVs_sample_CO2_MeCN_10mM_Fc_run02"),
        ]
        warnings = []
        workflow = AppWorkflow()
        status = ""

    state = _state_from_load_result(Result(), registry=SessionRegistry())

    assert "Gas: CO₂" in state["conditions"]
    assert "Solvent: MeCN" in state["conditions"]


def test_browser_layout_initial_column_selector_uses_available_options():
    pytest.importorskip("dash")
    from ecat_app.layout import create_layout

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
    from ecat_app.references import reference_field_visibility

    assert reference_field_visibility("none")["manual"] is False
    assert reference_field_visibility("manual")["manual"] is True
    assert reference_field_visibility("file")["file"] is True
    assert reference_field_visibility("keyword")["keyword"] is True
    assert reference_field_visibility("auto")["auto"] is True


def test_browser_default_source_points_to_fe_phoh_data(repo_root):
    from ecat_app.defaults import default_workflow, example_folder_options, example_folder_path

    workflow = default_workflow(repo_root)

    assert workflow.source_kind == "local_path"
    assert Path(workflow.source_path).as_posix().endswith("examples/data/fe_phoh_cv")
    assert workflow.recursive is True
    assert workflow.import_options["sort keys"] == ["subfolder", "timestamp"]
    assert [option["value"] for option in example_folder_options()] == [
        "fe_phoh_cv",
        "chrono_ca",
        "chrono_cp",
    ]
    assert example_folder_path("chrono_ca", repo_root).as_posix().endswith("examples/data/chrono_ca")
    assert example_folder_path("chrono_cp", repo_root).as_posix().endswith("examples/data/chrono_cp")
    assert example_folder_path("missing", repo_root) is None


def test_browser_default_fe_phoh_multiscan_sorts_after_main_series(repo_root):
    import ecat as e

    objects = e.get_data(
        {
            "folder path": str(repo_root / "examples" / "data" / "fe_phoh_cv"),
            "recursive search": False,
            "sort keys": ["timestamp"],
            "print": False,
        }
    )

    names = [Path(obj.filepath).name for obj in objects]
    assert names[-1] == "MeCN_Ar_0.1MTBAPF6_1mMFe-tpyPY2Me_-1.5_to_1.5V_100mVs_multiscan.txt"
    assert names[1] == "MeCN_Ar_0.1MTBAPF6_3mMFc_1mMFe-tpyPY2Me_-1.2_to_1V_100mVs.txt"
    assert names[7].startswith("MeCN_CO2_0.1MTBAPF6_3mMFc_1mMFe-tpyPY2Me_-1.2_to_1V")


def test_browser_local_path_loading_reports_supported_text_file_status(fixtures_dir, tmp_path):
    from ecat_app.adapters import load_local_path

    _copy_fixture(fixtures_dir, tmp_path, "ch_cv.txt")
    _copy_fixture(fixtures_dir, tmp_path, "basi_cv.txt")

    result = load_local_path(tmp_path, recursive=True)

    assert result.status == "2 supported text files found."


def test_browser_local_path_loading_discovers_eclab_mpt(fixtures_dir, tmp_path):
    from ecat_app.adapters import load_local_path

    destination = tmp_path / "eclab_cv.mpt"
    destination.write_text(
        (fixtures_dir / "eclab_cv.txt").read_text(encoding="ISO-8859-1"),
        encoding="ISO-8859-1",
    )

    result = load_local_path(tmp_path, recursive=False)

    assert result.status == "1 supported text file found."
    assert len(result.objects) == 1
    assert type(result.objects[0]).__name__ == "cv"


def test_browser_callback_state_includes_import_status(fixtures_dir, tmp_path):
    from ecat_app.callbacks import handle_local_path_load
    from ecat_app.state import SessionRegistry

    _copy_fixture(fixtures_dir, tmp_path, "ch_cv.txt")
    registry = SessionRegistry()

    state = handle_local_path_load(str(tmp_path), recursive=True, registry=registry)

    assert state["status"] == "1 supported text file found."


def test_browser_import_returns_default_multiplot(fixtures_dir, tmp_path):
    from ecat_app.callbacks import handle_local_path_load
    from ecat_app.state import SessionRegistry

    _copy_fixture(fixtures_dir, tmp_path, "ch_cv.txt")
    _copy_fixture(fixtures_dir, tmp_path, "basi_cv.txt")
    registry = SessionRegistry()

    state = handle_local_path_load(str(tmp_path), recursive=False, registry=registry)

    assert state["plot"].startswith("data:image/png;base64,")


def test_browser_sort_keys_reorder_included_objects_for_analysis(cv_factory):
    from ecat_app.adapters import objects_for_analysis
    from ecat_app.workflow import AppWorkflow

    slow = cv_factory(name="50mVs_sample_CO2_MeCN_10mM_Fc_run01")
    fast = cv_factory(name="100mVs_sample_CO2_MeCN_10mM_Fc_run02")
    workflow = AppWorkflow(
        included_row_ids=["row-0", "row-1"],
        sort_keys=["scan rate"],
    )

    ordered = objects_for_analysis([fast, slow], workflow)

    assert [obj.scan_rate for obj in ordered] == [0.05, 0.1]


def test_browser_select_all_state_cycles_between_all_and_none():
    from ecat_app.table import selection_toggle_state, toggle_all_selection

    rows = [{"id": "row-0"}, {"id": "row-1"}]

    assert selection_toggle_state(rows, ["row-0"]) == "-"
    assert selection_toggle_state(rows, ["row-0", "row-1", "row-hidden"]) == "None"
    assert toggle_all_selection(rows, ["row-0"]) == ["row-0", "row-1"]
    assert toggle_all_selection(rows, ["row-0", "row-1"]) == []


def test_browser_table_selection_defaults_include_dash_rows(cv_factory):
    from ecat_app.table import selected_rows_for_table, toggle_all_selection_state

    rows = [{"id": "row-0"}, {"id": "row-1"}]

    assert selected_rows_for_table({"data": rows}) == [0, 1]
    assert toggle_all_selection_state(rows, ["row-0", "row-1"]) == ([], [])
    assert toggle_all_selection_state(rows, []) == (["row-0", "row-1"], [0, 1])


def test_browser_filtered_selection_toggle_preserves_hidden_rows():
    from ecat_app.table import toggle_filtered_selection_state

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
    from ecat_app.table import (
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
    from ecat_app.table import ag_grid_column_defs, build_browser_table

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
    from ecat_app.table import selected_grid_rows_for_ids, selected_row_ids_from_grid_rows

    rows = [{"id": "row-0", "Filename": "a.txt"}, {"id": "row-1", "Filename": "b.txt"}]

    assert selected_row_ids_from_grid_rows([rows[1]]) == ["row-1"]
    assert selected_grid_rows_for_ids(rows, ["row-1", "row-missing"]) == {"ids": ["row-1"]}


def test_browser_replot_uses_current_selection(cv_factory):
    from ecat_app.callbacks import handle_replot
    from ecat_app.state import SessionRegistry

    registry = SessionRegistry()
    first = cv_factory(name="50mVs_sample_CO2_MeCN_10mM_Fc_run01")
    second = cv_factory(name="100mVs_sample_CO2_MeCN_10mM_Fc_run02")
    dataset_id = registry.put([first, second])

    result = handle_replot(dataset_id, ["row-1"], registry=registry)
    save_result = handle_replot(dataset_id, ["row-1"], registry=registry, plot_options={"_format": "svg", "_dpi": 300})

    assert result["plot"].startswith("data:image/png;base64,")
    assert save_result["plot"].startswith("data:image/svg+xml;base64,")


def test_browser_displayed_row_ids_preserve_table_order():
    from ecat_app.table import displayed_selected_row_ids

    displayed = [{"id": "row-2"}, {"id": "row-0"}, {"id": "row-1"}]

    assert displayed_selected_row_ids(displayed, ["row-1", "row-2"]) == ["row-2", "row-1"]
    assert displayed_selected_row_ids(displayed, None) == ["row-2", "row-0", "row-1"]


def test_browser_sidebar_class_toggles_collapsed_state():
    from ecat_app.callbacks import expand_sidebar_class, toggle_sidebar_class

    assert toggle_sidebar_class("ecat-app") == "ecat-app ecat-sidebar-collapsed"
    assert toggle_sidebar_class("ecat-app ecat-sidebar-collapsed") == "ecat-app"
    assert expand_sidebar_class("ecat-app ecat-sidebar-collapsed") == "ecat-app"
    assert expand_sidebar_class("ecat-app") == "ecat-app"


def test_browser_callback_helpers_apply_reference_by_reloading(fixtures_dir, tmp_path):
    from ecat_app.callbacks import handle_apply_reference, handle_local_path_load
    from ecat_app.state import SessionRegistry

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
    from ecat_app.config import AppConfig

    monkeypatch.setenv("ECAT_APP_MODE", "remote")
    monkeypatch.setenv("ECAT_APP_ALLOW_CODE_EXECUTION", "1")

    remote = AppConfig.from_env()
    local = AppConfig(mode="local", allow_code_execution=True)

    assert remote.enable_folder_picker is False
    assert remote.allow_code_execution is False
    assert local.enable_folder_picker is True
    assert local.allow_code_execution is True


def test_browser_workflow_roundtrips_model_options():
    from ecat_app.workflow import AppWorkflow

    workflow = AppWorkflow(
        model_options={
            "mechanism_source": "custom",
            "mechanism": "E(1):a=b",
            "mechanism_valid": True,
        }
    )

    restored = AppWorkflow.from_dict(workflow.to_dict())

    assert restored.model_options == {
        "mechanism_source": "custom",
        "mechanism": "E(1):a=b",
        "mechanism_valid": True,
    }


def test_browser_validates_preset_and_custom_model_mechanisms():
    from ecat_app.adapters import validate_simulation_mechanism

    preset = validate_simulation_mechanism("preset", "EC")
    square = validate_simulation_mechanism("preset", "Square")
    custom = validate_simulation_mechanism(
        "custom",
        "E",
        "E(1):Fe2=Fe1\nC:Fe1>Fe0",
    )
    invalid = validate_simulation_mechanism("preset", "not-a-mechanism")

    assert preset["ok"] is True
    assert preset["compiled"] == "E(1):a=b\nC:b=c"
    assert preset["formatted_equations"] == [
        "E(1): A ⇌ B",
        "C: B → C",
    ]
    assert square["ok"] is True
    assert square["compiled"] == "E(1):a=b\nC:a=c\nE(1):c=d\nC:b=d"
    assert square["formatted_equations"] == [
        "E(1): A ⇌ B",
        "C: A → C",
        "E(1): C ⇌ D",
        "C: B → D",
    ]
    assert custom["ok"] is True
    assert custom["mechanism"] == "E(1):Fe2=Fe1\nC:Fe1>Fe0"
    assert invalid["ok"] is False
    assert "Unknown simulation mechanism preset" in invalid["message"]


def test_browser_model_mechanism_validation_generates_editable_detail_rows():
    from ecat_app.adapters import validate_simulation_mechanism

    validation = validate_simulation_mechanism(
        "custom",
        "E",
        "E(1):FeII=FeI\nC:FeI>Fe0",
    )

    assert validation["ok"] is True
    assert validation["formatted_equations"] == [
        "E(1): FeII ⇌ FeI",
        "C: FeI → Fe0",
    ]
    assert validation["mechanism_details"] == [
        {
            "index": 1,
            "kind": "electron transfer",
            "equation": "E(1):FeII=FeI",
            "reactants": "FeII",
            "products": "FeI",
            "electrons": 1,
            "parameter": "kinetics.0",
            "notes": "",
        },
        {
            "index": 2,
            "kind": "chemical step",
            "equation": "C:FeI>Fe0",
            "reactants": "FeI",
            "products": "Fe0",
            "electrons": "",
            "parameter": "reactions.0",
            "notes": "",
        },
    ]


def test_browser_model_mechanism_details_generate_mechanism_parameter_rows():
    from ecat_app.adapters import validate_simulation_mechanism
    from ecat_app.callbacks import model_mechanism_parameter_rows

    details = validate_simulation_mechanism("preset", "EC")["mechanism_details"]
    rows = model_mechanism_parameter_rows("EC", "scratch", mechanism_details=details)

    assert [row["path"] for row in rows] == [
        "kinetics.0.E0",
        "kinetics.0.k0",
        "kinetics.0.alpha",
        "reactions.0.kf",
        "reactions.0.kb",
    ]
    assert next(row for row in rows if row["path"] == "reactions.0.kf")["name"] == "k₁ (s⁻¹)"
    assert next(row for row in rows if row["path"] == "reactions.0.kb")["initial"] == 0.0


def test_browser_model_species_rows_follow_preset_species_symbols():
    from ecat_app.callbacks import model_species_parameter_rows

    rows = model_species_parameter_rows("Square", "scratch")
    diffusion_rows = [row for row in rows if row["group"] == "diffusion"]
    concentration_rows = [row for row in rows if row["group"] == "concentration"]

    assert [row["species"] for row in diffusion_rows] == ["A", "B", "C", "D"]
    assert [row["path"] for row in diffusion_rows] == [
        "diffusion.A",
        "diffusion.B",
        "diffusion.C",
        "diffusion.D",
    ]
    assert [row["species"] for row in concentration_rows] == ["A", "B", "C", "D"]
    assert [row["initial"] for row in concentration_rows] == [1e-3, 0.0, 0.0, 0.0]


def test_browser_model_row_merge_preserves_existing_values_when_mechanism_changes():
    from ecat_app.callbacks import (
        merge_model_parameter_rows,
        model_mechanism_parameter_rows,
        model_species_parameter_rows,
    )

    species_rows = model_species_parameter_rows("E", "scratch")
    mechanism_rows = model_mechanism_parameter_rows("E", "scratch")
    next(row for row in species_rows if row["path"] == "diffusion.A")["initial"] = 2e-5
    next(row for row in species_rows if row["path"] == "concentrations.bulk.A")["initial"] = 4e-3
    next(row for row in mechanism_rows if row["path"] == "kinetics.0.E0")["initial"] = -0.55

    merged_species = merge_model_parameter_rows(model_species_parameter_rows("EC", "scratch"), species_rows)
    merged_mechanism = merge_model_parameter_rows(model_mechanism_parameter_rows("EC", "scratch"), mechanism_rows)

    assert next(row for row in merged_species if row["path"] == "diffusion.A")["initial"] == 2e-5
    assert next(row for row in merged_species if row["path"] == "concentrations.bulk.A")["initial"] == 4e-3
    assert next(row for row in merged_species if row["path"] == "diffusion.C")["initial"] == 1e-5
    assert next(row for row in merged_species if row["path"] == "concentrations.bulk.C")["initial"] == 0.0
    assert next(row for row in merged_mechanism if row["path"] == "kinetics.0.E0")["initial"] == -0.55
    assert next(row for row in merged_mechanism if row["path"] == "reactions.0.kf")["initial"] == 1.0


def test_browser_model_simulation_params_include_reaction_rows():
    from ecat_app.adapters import simulation_params_from_table_rows
    from ecat_app.callbacks import model_cell_parameter_rows, model_mechanism_parameter_rows, model_species_parameter_rows

    mechanism_rows = model_mechanism_parameter_rows("EC", "scratch")
    species_rows = model_species_parameter_rows("EC", "scratch")
    kf_row = next(row for row in mechanism_rows if row["path"] == "reactions.0.kf")
    kf_row["initial"] = 2.5

    params = simulation_params_from_table_rows([*species_rows, *mechanism_rows], model_cell_parameter_rows())

    assert params["reactions"] == [{"kf": 2.5, "kb": 0.0}]
    assert params["kinetics"] == [{"E0": -0.5, "k0": 0.001, "alpha": 0.5}]


def test_browser_model_mechanism_detail_edits_refresh_parameter_rows():
    from ecat_app.callbacks import update_model_options_with_mechanism_details

    model_options = {
        "mechanism": "E(1):a=b",
        "mechanism_valid": True,
        "simulation_ready": True,
        "fit_requested": True,
        "fit_result": {"status": "ok"},
    }
    edited_details = [
        {
            "index": 1,
            "kind": "electron transfer",
            "equation": "E(1):a=b",
            "reactants": "a",
            "products": "b",
        },
        {
            "index": 2,
            "kind": "chemical step",
            "equation": "C:b=c",
            "reactants": "b",
            "products": "c",
        },
    ]

    updated = update_model_options_with_mechanism_details(model_options, edited_details)

    assert updated["simulation_ready"] is False
    assert "fit_requested" not in updated
    assert "fit_result" not in updated
    assert [row["path"] for row in updated["mechanism_parameters"]] == [
        "kinetics.0.E0",
        "kinetics.0.k0",
        "kinetics.0.alpha",
        "reactions.0.kf",
        "reactions.0.kb",
    ]
    assert any(row["path"] == "diffusion.A" for row in updated["parameters"])
    assert any(row["path"] == "concentrations.bulk.A" for row in updated["parameters"])


def test_browser_model_simulate_button_is_gated_by_mechanism_parse(monkeypatch):
    import ecat_app.callbacks as callbacks

    monkeypatch.setattr(callbacks, "simulation_backend_available", lambda: True)

    assert callbacks.model_simulate_gate({"mechanism_valid": False}) == (
        True,
        "Choose a valid mechanism before simulating.",
    )
    assert callbacks.model_simulate_gate({"mechanism_valid": True}) == (False, "")


def test_browser_model_builds_simulation_params_from_tables():
    from ecat_app.adapters import simulation_params_from_table_rows
    from ecat_app.callbacks import model_cell_parameter_rows, model_parameter_rows

    parameter_rows = model_parameter_rows("E", "scratch")
    for row in parameter_rows:
        if row["key"] == "E0":
            row["initial"] = -0.45
        if row["key"] == "C_A":
            row["species"] = "A"
            row["initial"] = 0.002
    cell_rows = model_cell_parameter_rows()

    params = simulation_params_from_table_rows(parameter_rows, cell_rows)

    assert params["concentrations"]["bulk"]["A"] == 0.002
    assert params["concentrations"]["bulk"]["B"] == 0.0
    assert params["diffusion"]["A"] == 1e-9
    assert params["diffusion"]["B"] == 1e-9
    assert params["kinetics"] == [{"E0": -0.45, "k0": 0.001, "alpha": 0.5}]
    assert params["cell"]["T"] == 298.15
    assert params["cell"]["Cdl"] == "auto"

    for row in parameter_rows:
        if row["key"] == "C_A":
            row["phase"] = "surface"
    surface_params = simulation_params_from_table_rows(parameter_rows, cell_rows)
    assert surface_params["concentrations"]["surface"]["A"] == 0.002


def test_browser_model_builds_fit_spec_from_fit_table_rows():
    from ecat_app.adapters import fit_spec_from_table_rows
    from ecat_app.callbacks import model_cell_parameter_rows, model_mechanism_parameter_rows, model_species_parameter_rows

    species_rows = model_species_parameter_rows("E", "scratch")
    mechanism_rows = model_mechanism_parameter_rows("E", "scratch")
    cell_rows = model_cell_parameter_rows()
    for row in [*species_rows, *mechanism_rows]:
        row["vary"] = False
        if row["key"] == "E0":
            row["vary"] = True
            row["lower"] = -1.2
            row["upper"] = -0.2
        if row["key"] == "C_A":
            row["vary"] = True
            row["phase"] = "surface"
            row["species"] = "cat"
            row["lower"] = 0
            row["upper"] = 1e-8
    for row in cell_rows:
        if row["key"] == "Cdl":
            row["vary"] = True
            row["lower"] = 0
            row["upper"] = 1e-5

    fit = fit_spec_from_table_rows([*species_rows, *mechanism_rows], cell_rows)

    assert fit == {
        "vary": ["concentrations.surface.cat", "kinetics.0.E0", "cell.Cdl"],
        "bounds": {
            "concentrations.surface.cat": [0.0, 1e-08],
            "kinetics.0.E0": [-1.2, -0.2],
            "cell.Cdl": [0.0, 1e-05],
        },
    }


def test_browser_model_parameter_rows_use_formatted_names_with_units():
    from ecat_app.callbacks import model_bound_column_defs, model_parameter_rows

    rows = model_parameter_rows("E", "scratch")
    columns = model_bound_column_defs([])
    value_columns = model_bound_column_defs([], fit_enabled=False)
    empty_fit_columns = model_bound_column_defs(
        [],
        fit_enabled=True,
        row_data=[
            {"final": "", "stderr": None, "comment": ""},
            {"final": float("nan"), "stderr": "", "comment": None},
        ],
    )
    result_columns = model_bound_column_defs(
        [],
        fit_enabled=True,
        row_data=[
            {"final": 0, "stderr": "", "comment": ""},
            {"final": "", "stderr": 0.002, "comment": "hit upper bound"},
        ],
    )
    mechanism_columns = model_bound_column_defs([], "mechanism")
    species_columns = model_bound_column_defs([], "species")
    phase_column = next(column for column in species_columns if column["field"] == "phase")

    assert [column["field"] for column in columns][:2] == ["name", "initial"]
    assert [column["field"] for column in mechanism_columns][:2] == ["step", "name"]
    assert "path" not in [column["field"] for column in mechanism_columns]
    assert next(column for column in value_columns if column["field"] == "initial")["headerName"] == "Value"
    assert next(column for column in columns if column["field"] == "initial")["headerName"] == "Initial"
    assert not {"final", "stderr", "comment"}.intersection({column["field"] for column in empty_fit_columns})
    assert next(column for column in result_columns if column["field"] == "final")["headerName"] == "Fitted Value"
    assert next(column for column in result_columns if column["field"] == "stderr")["headerName"] == "Std. Error"
    assert next(column for column in result_columns if column["field"] == "comment")["headerName"] == "Comment"
    assert any(row.get("path") == "kinetics.0.E0" and row["key"] == "E0" for row in rows)
    assert "unit" not in [column["field"] for column in columns]
    assert any(column["field"] == "vary" and column["headerName"] == "Fit?" for column in columns)
    assert next(column for column in columns if column["field"] == "initial")["cellClass"] == "ecat-editable-cell"
    assert next(column for column in result_columns if column["field"] == "lower")["cellClass"] == "ecat-editable-cell"
    assert next(column for column in result_columns if column["field"] == "final").get("cellClass") is None
    assert not any(column["field"] == "vary" for column in model_bound_column_defs([], fit_enabled=False))
    assert not any(column["field"] == "final" for column in value_columns)
    assert not {"final", "stderr", "comment"}.intersection({column["field"] for column in columns})
    assert {"final", "stderr", "comment"}.issubset({column["field"] for column in result_columns})
    assert "mechanism" not in [row["key"] for row in rows]
    assert any(row["name"] == "E⁰ (V)" and row["key"] == "E0" for row in rows)
    assert any(row["name"] == "k₀ (m s⁻¹)" and row["key"] == "k0" for row in rows)
    assert any(row["name"] == "D (cm² s⁻¹)" and row["key"] == "D_A" for row in rows)
    assert any(row["name"] == "D (cm² s⁻¹)" and row["key"] == "D_B" for row in rows)
    assert phase_column["editable"] is True
    assert phase_column["cellEditor"] == "agSelectCellEditor"
    assert phase_column["cellEditorParams"]["values"] == ["bulk", "surface"]


def test_browser_multi_cv_analysis_equations_and_labeled_inputs():
    from ecat_app.callbacks import multi_analysis_equation_content

    fowa = multi_analysis_equation_content("fowa")
    tafel = multi_analysis_equation_content("tafel_analysis")
    sevcik = multi_analysis_equation_content("sevcik_analysis")
    none = multi_analysis_equation_content("none")

    assert "k" in repr(fowa)
    assert "obs" in repr(fowa)
    assert "x" in repr(fowa)
    assert "fowa" in repr(fowa).lower()
    assert "TOF" in repr(tafel)
    assert "exp" in repr(tafel)
    assert "i" in repr(sevcik)
    assert "ν" in repr(sevcik)
    assert "1/2" in repr(sevcik)
    assert none == ""


def test_browser_model_simulation_uses_edited_grid_rows():
    from ecat_app.callbacks import model_cell_parameter_rows, model_parameter_rows, model_rows_for_simulation

    model_options = {"parameters": model_parameter_rows("E", "scratch"), "cell_parameters": model_cell_parameter_rows()}
    edited_parameters = [dict(row) for row in model_options["parameters"]]
    edited_cells = [dict(row) for row in model_options["cell_parameters"]]
    for row in edited_parameters:
        if row["key"] == "E0":
            row["initial"] = -0.5
    for row in edited_cells:
        if row["key"] == "T":
            row["initial"] = 310

    parameters, cell_parameters = model_rows_for_simulation(model_options, edited_parameters, edited_cells)

    assert next(row for row in parameters if row["key"] == "E0")["initial"] == -0.5
    assert next(row for row in cell_parameters if row["key"] == "T")["initial"] == 310


def test_browser_model_split_tables_merge_for_simulation_params():
    from ecat_app.adapters import simulation_params_from_table_rows
    from ecat_app.callbacks import (
        model_cell_parameter_rows,
        model_mechanism_parameter_rows,
        model_setup_parameter_rows,
        model_species_parameter_rows,
        model_split_rows_for_simulation,
    )

    setup_rows = model_setup_parameter_rows()
    species_rows = model_species_parameter_rows("E", "scratch")
    mechanism_rows = model_mechanism_parameter_rows("E", "scratch")
    cell_rows = model_cell_parameter_rows()
    setup_rows[0]["initial"] = "accurate"
    next(row for row in species_rows if row["key"] == "C_A")["species"] = "cat"
    next(row for row in species_rows if row["key"] == "C_A")["initial"] = 0.004
    next(row for row in species_rows if row["key"] == "D_A")["initial"] = 2e-5
    next(row for row in species_rows if row["key"] == "D_B")["initial"] = 3e-5
    next(row for row in mechanism_rows if row["key"] == "E0")["initial"] = -0.52

    parameter_rows, cell_parameter_rows, setup_parameter_rows = model_split_rows_for_simulation(
        {},
        mechanism_rows,
        cell_rows,
        species_rows,
        setup_rows,
    )
    params = simulation_params_from_table_rows(parameter_rows, cell_parameter_rows, setup_parameter_rows)

    assert [row["key"] for row in parameter_rows] == ["D_A", "D_B", "C_A", "C_B", "E0", "k0", "alpha"]
    assert params["spatial"] == "accurate"
    assert params["concentrations"]["bulk"]["cat"] == 0.004
    assert params["diffusion"]["A"] == pytest.approx(2e-9)
    assert params["diffusion"]["B"] == pytest.approx(3e-9)
    assert params["kinetics"][0]["E0"] == -0.52


def test_browser_model_runs_simulate_cv_from_scratch(monkeypatch):
    import pandas as pd

    from ecat_app.adapters import run_browser_simulate_cv
    from ecat_app.callbacks import model_cell_parameter_rows, model_parameter_rows

    calls = {}

    class FakeSimulation:
        def __init__(self):
            self.data = pd.DataFrame({"Potential": [0.0, -0.5], "Current": [0.0, 1e-6]})
            self.summary = {"backend": "fake", "mechanism": "E(1):a=b"}

        def plot(self, options=None):
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots()
            ax.plot(self.data["Potential"], self.data["Current"])
            return ax

    def fake_simulate_cv(input_obj, mechanism, params, options=None, backend="electrokitty"):
        calls["input"] = input_obj
        calls["mechanism"] = mechanism
        calls["params"] = params
        calls["options"] = options
        calls["backend"] = backend
        return FakeSimulation()

    import ecat as e

    monkeypatch.setattr(e.simulation, "simulate_cv", fake_simulate_cv)

    result = run_browser_simulate_cv(
        mode="scratch",
        mechanism="E",
        parameter_rows=model_parameter_rows("E", "scratch"),
        cell_parameter_rows=model_cell_parameter_rows(),
        program_settings={
            "Ei": 0.0,
            "E_low": -1.0,
            "E_high": None,
            "scan_rate": 0.1,
            "segments": 2,
            "points_per_segment": 20,
            "quiet_time": 0,
        },
    )

    assert result["status"] == "ok"
    assert result["plot"].startswith("data:image/png;base64,")
    assert result["summary"]["backend"] == "fake"
    assert calls["mechanism"] == "E"
    assert calls["params"]["kinetics"][0]["E0"] == -0.5
    assert calls["params"]["cell"]["Cdl"] == 0.0
    assert calls["options"]["plot"] is False


def test_browser_model_runs_simulate_cv_from_cv_with_program_scan_rate_fallback(monkeypatch, cv_factory):
    import pandas as pd

    from ecat_app.adapters import run_browser_simulate_cv
    from ecat_app.callbacks import model_cell_parameter_rows, model_parameter_rows

    cv_obj = cv_factory()
    cv_obj.scan_rate = None
    calls = {}

    class FakeSimulation:
        def __init__(self):
            self.data = pd.DataFrame({"Potential": [0.0, -0.5], "Current": [0.0, 1e-6]})
            self.summary = {"backend": "fake"}

        def plot(self, options=None):
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots()
            ax.plot(self.data["Potential"], self.data["Current"])
            return ax

    def fake_simulate_cv(input_obj, mechanism, params, options=None, backend="electrokitty"):
        calls["input"] = input_obj
        calls["mechanism"] = mechanism
        calls["params"] = params
        return FakeSimulation()

    import ecat as e

    monkeypatch.setattr(e.simulation, "simulate_cv", fake_simulate_cv)

    result = run_browser_simulate_cv(
        mode="cv",
        mechanism="E",
        parameter_rows=model_parameter_rows("E", "cv"),
        cell_parameter_rows=model_cell_parameter_rows(),
        program_settings={"scan_rate": 0.25},
        cv_data_settings={"cv_index": 0, "stride": 1},
        objects=[cv_obj],
    )

    assert result["status"] == "ok"
    assert calls["input"].metadata["scan_rate"] == pytest.approx(0.25)
    assert result["plot"].startswith("data:image/png;base64,")


def test_browser_model_plots_cv_program_from_selected_cv_data(cv_factory):
    from ecat_app.callbacks import model_cv_program_plot_from_controls
    from ecat_app.state import SessionRegistry

    cv_obj = cv_factory()
    cv_obj.scan_rate = None
    registry = SessionRegistry()
    dataset_id = registry.put([cv_obj])

    plot = model_cv_program_plot_from_controls(
        {"dataset_id": dataset_id},
        0,
        "none",
        None,
        None,
        "",
        1,
        "off",
        program_scan_rate=0.25,
        registry=registry,
    )

    assert plot.startswith("data:image/png;base64,")


def test_browser_model_cv_program_plot_passes_trim_options(monkeypatch, cv_factory):
    from ecat_app.callbacks import model_cv_program_plot_from_controls
    from ecat_app.state import SessionRegistry

    cv_obj = cv_factory()
    captured = {}
    registry = SessionRegistry()
    dataset_id = registry.put([cv_obj])

    def fake_render(objects, cv_data_settings, program_settings):
        captured["objects"] = objects
        captured["cv_data_settings"] = cv_data_settings
        captured["program_settings"] = program_settings
        return "data:image/png;base64,trimmed"

    monkeypatch.setattr("ecat_app.callbacks.render_browser_cv_data_program_plot", fake_render)

    plot = model_cv_program_plot_from_controls(
        {"dataset_id": dataset_id},
        0,
        "pointwise",
        -0.1,
        0.1,
        "1-2",
        5,
        "off",
        program_scan_rate=0.25,
        registry=registry,
    )

    assert plot == "data:image/png;base64,trimmed"
    assert captured["objects"] == [cv_obj]
    assert captured["cv_data_settings"] == {
        "cv_index": 0,
        "stride": 5,
        "trim mode": "pointwise",
        "potential window": [-0.1, 0.1],
        "segments": [1, 2],
    }
    assert captured["program_settings"] == {"scan_rate": 0.25}


def test_browser_model_runs_single_cv_fit_with_public_ecat_api(monkeypatch, cv_factory):
    import pandas as pd

    from ecat_app.adapters import run_browser_fit_cv
    from ecat_app.callbacks import model_cell_parameter_rows, model_mechanism_parameter_rows, model_species_parameter_rows

    cv_obj = cv_factory()
    calls = {}

    class FakeFitResult:
        def __init__(self):
            self.best_params = {
                "concentrations": {"bulk": {"a": 0.001, "b": 0.0}},
                "diffusion": {"a": 1e-9, "b": 1e-9},
                "kinetics": [{"E0": -0.2, "k0": 1e-3, "alpha": 0.5}],
                "cell": {"T": 298.15, "Cdl": 2e-6},
            }
            self.summary = {"success": True, "rmse": 0.0}
            self.data = pd.DataFrame({"Potential": [0.0, -0.5], "Current": [0.0, 1e-6]})

        def plot(self, options=None):
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots()
            ax.plot(self.data["Potential"], self.data["Current"])
            return ax

    def fake_fit_cv(input_or_result, mechanism=None, params=None, fit=None, options=None, backend="electrokitty", method="least squares"):
        calls["input"] = input_or_result
        calls["mechanism"] = mechanism
        calls["params"] = params
        calls["fit"] = fit
        calls["options"] = options
        calls["backend"] = backend
        calls["method"] = method
        return FakeFitResult()

    import ecat as e

    monkeypatch.setattr(e.simulation, "fit_cv", fake_fit_cv)

    species_rows = model_species_parameter_rows("E", "cv")
    mechanism_rows = model_mechanism_parameter_rows("E", "cv")
    cell_rows = model_cell_parameter_rows()
    for row in [*species_rows, *mechanism_rows, *cell_rows]:
        row["vary"] = False
    e0_row = next(row for row in mechanism_rows if row["key"] == "E0")
    e0_row["vary"] = True
    e0_row["lower"] = -1.2
    e0_row["upper"] = -0.2
    cdl_row = next(row for row in cell_rows if row["key"] == "Cdl")
    cdl_row["vary"] = True
    cdl_row["lower"] = 0
    cdl_row["upper"] = 1e-5

    result = run_browser_fit_cv(
        fit_mode="single",
        fit_cv_index=0,
        mechanism="E",
        parameter_rows=[*species_rows, *mechanism_rows],
        cell_parameter_rows=cell_rows,
        cv_data_settings={"cv_index": 9, "stride": 5, "segment": 2},
        objects=[cv_obj],
    )

    assert result["status"] == "ok"
    assert result["plot"].startswith("data:image/png;base64,")
    assert result["summary"] == {"success": True, "rmse": 0.0}
    assert calls["input"] is cv_obj
    assert calls["mechanism"] == "E"
    assert calls["fit"] == {
        "vary": ["kinetics.0.E0", "cell.Cdl"],
        "bounds": {"kinetics.0.E0": [-1.2, -0.2], "cell.Cdl": [0.0, 1e-05]},
    }
    assert calls["options"]["plot"] is False
    assert calls["options"]["progress"] is False
    assert calls["options"]["cv data"] == {"stride": 5, "segment": 2}
    fitted_e0 = next(row for row in result["parameter_rows"] if row["key"] == "E0")
    fitted_cdl = next(row for row in result["cell_parameter_rows"] if row["key"] == "Cdl")
    assert fitted_e0["final"] == -0.2
    assert fitted_e0["comment"] == "hit upper bound"
    assert fitted_cdl["final"] == 2e-6
    assert fitted_cdl["comment"] == ""


def test_browser_model_runs_temperature_condition_sweep(monkeypatch):
    import pandas as pd

    from ecat_app.adapters import run_browser_simulate_cv
    from ecat_app.callbacks import model_cell_parameter_rows, model_parameter_rows

    temperatures = []

    class FakeSimulation:
        def __init__(self, temperature):
            self.data = pd.DataFrame({"Potential": [0.0, -0.5], "Current": [0.0, temperature * 1e-9]})
            self.summary = {"backend": "fake", "temperature": temperature}

    def fake_simulate_cv(_input_obj, _mechanism, params, options=None, backend="electrokitty"):
        temperature = params["cell"]["T"]
        temperatures.append(temperature)
        return FakeSimulation(temperature)

    import ecat as e

    monkeypatch.setattr(e.simulation, "simulate_cv", fake_simulate_cv)

    result = run_browser_simulate_cv(
        mode="scratch",
        mechanism="E",
        parameter_rows=model_parameter_rows("E", "scratch"),
        cell_parameter_rows=model_cell_parameter_rows(),
        program_settings={"Ei": 0.0, "E_low": -1.0, "scan_rate": 0.1, "points_per_segment": 20},
        over_conditions=True,
        condition_settings={"condition_axis": "temperature", "condition_values": "280, 300"},
    )

    assert result["status"] == "ok"
    assert result["summary"]["condition_axis"] == "temperature"
    assert result["summary"]["condition_values"] == [280.0, 300.0]
    assert result["plot"].startswith("data:image/png;base64,")
    assert temperatures == [280.0, 300.0]


def test_browser_model_runs_scan_rate_and_concentration_sweeps(monkeypatch):
    import pandas as pd

    from ecat_app.adapters import run_browser_simulate_cv
    from ecat_app.callbacks import model_cell_parameter_rows, model_parameter_rows

    calls = []

    class FakeSimulation:
        def __init__(self, label):
            self.data = pd.DataFrame({"Potential": [0.0, -0.5], "Current": [0.0, 1e-6]})
            self.summary = {"backend": "fake", "label": label}

    def fake_simulate_cv(input_obj, _mechanism, params, options=None, backend="electrokitty"):
        calls.append(
            {
                "scan_rate": getattr(input_obj, "metadata", {}).get("scan_rate"),
                    "concentration": params["concentrations"]["bulk"]["A"],
            }
        )
        return FakeSimulation(len(calls))

    import ecat as e

    monkeypatch.setattr(e.simulation, "simulate_cv", fake_simulate_cv)

    run_browser_simulate_cv(
        mode="scratch",
        mechanism="E",
        parameter_rows=model_parameter_rows("E", "scratch"),
        cell_parameter_rows=model_cell_parameter_rows(),
        program_settings={"Ei": 0.0, "E_low": -1.0, "scan_rate": 0.1, "points_per_segment": 20},
        over_conditions=True,
        condition_settings={"condition_axis": "scan_rate", "condition_values": "0.05, 0.2"},
    )
    run_browser_simulate_cv(
        mode="scratch",
        mechanism="E",
        parameter_rows=model_parameter_rows("E", "scratch"),
        cell_parameter_rows=model_cell_parameter_rows(),
        program_settings={"Ei": 0.0, "E_low": -1.0, "scan_rate": 0.1, "points_per_segment": 20},
        over_conditions=True,
        condition_settings={"condition_axis": "concentration", "condition_values": "0.001, 0.002"},
    )

    assert [call["scan_rate"] for call in calls[:2]] == [0.05, 0.2]
    assert [call["concentration"] for call in calls[2:]] == [0.001, 0.002]


def test_browser_model_condition_range_settings_generate_values():
    from ecat_app.adapters import _simulation_condition_values
    from ecat_app.callbacks import model_condition_settings_from_controls

    settings = model_condition_settings_from_controls(
        "concentration",
        [0.001, 0.003],
        3,
        "cat",
    )

    assert settings == {
        "condition_axis": "concentration",
        "condition_min": 0.001,
        "condition_max": 0.003,
        "condition_count": 3,
        "condition_species": "cat",
    }
    assert _simulation_condition_values(settings) == [0.001, 0.002, 0.003]

    missing_range = model_condition_settings_from_controls("scan_rate", [None, None], None)

    assert missing_range == {
        "condition_axis": "scan_rate",
        "condition_min": 0.05,
        "condition_max": 0.2,
        "condition_count": 3,
    }


def test_browser_model_condition_axis_defaults_and_species_visibility():
    from ecat_app.callbacks import model_condition_axis_controls, model_condition_species_visibility

    scan_rate = model_condition_axis_controls("scan_rate")
    concentration = model_condition_axis_controls("concentration")
    temperature = model_condition_axis_controls("temperature")

    assert scan_rate[:5] == (0, 1, 0.01, [0.05, 0.2], {0.05: "0.05", 0.2: "0.2"})
    assert concentration[:4] == (0, 0.01, 0.0001, [0.001, 0.003])
    assert temperature[:4] == (250, 350, 1, [280, 320])
    assert model_condition_species_visibility("concentration") == {}
    assert model_condition_species_visibility("scan_rate") == {"display": "none"}


def test_browser_model_concentration_sweep_uses_selected_species(monkeypatch):
    import pandas as pd

    from ecat_app.adapters import run_browser_simulate_cv
    from ecat_app.callbacks import model_cell_parameter_rows, model_parameter_rows

    concentrations = []

    class FakeSimulation:
        def __init__(self):
            self.data = pd.DataFrame({"Potential": [0.0, -0.5], "Current": [0.0, 1e-6]})
            self.summary = {"backend": "fake"}

    def fake_simulate_cv(_input_obj, _mechanism, params, options=None, backend="electrokitty"):
        concentrations.append(params["concentrations"]["bulk"]["cat"])
        return FakeSimulation()

    import ecat as e

    monkeypatch.setattr(e.simulation, "simulate_cv", fake_simulate_cv)

    rows = model_parameter_rows("E", "scratch")
    for row in rows:
        if row["key"] == "C":
            row["species"] = "cat"

    result = run_browser_simulate_cv(
        mode="scratch",
        mechanism="E",
        parameter_rows=rows,
        cell_parameter_rows=model_cell_parameter_rows(),
        program_settings={"Ei": 0.0, "E_low": -1.0, "scan_rate": 0.1, "points_per_segment": 20},
        over_conditions=True,
        condition_settings={
            "condition_axis": "concentration",
            "condition_min": 0.001,
            "condition_max": 0.003,
            "condition_count": 3,
            "condition_species": "cat",
        },
    )

    assert result["status"] == "ok"
    assert result["summary"]["condition_values"] == [0.001, 0.002, 0.003]
    assert concentrations == [0.001, 0.002, 0.003]


def test_browser_model_mechanism_helpers_gate_fit_until_simulated():
    from ecat_app.callbacks import (
        build_model_simulation_state,
        model_fit_gate,
        model_mechanism_options_from_controls,
        model_mechanism_visibility,
        update_workflow_model_options,
    )

    preset_style, custom_style = model_mechanism_visibility("custom")
    options = model_mechanism_options_from_controls("custom", "E", "E(1):a=b")
    workflow = update_workflow_model_options({}, options)

    assert preset_style == {}
    assert custom_style == {}
    assert options["mechanism_valid"] is True
    assert options["mechanism"] == "E(1):a=b"
    assert model_fit_gate(options) == (
        True,
        True,
        "Run a simulation first to create the fit starting guess.",
    )
    assert workflow["model_options"]["mechanism_valid"] is True

    simulated = build_model_simulation_state(options, "cv")

    assert simulated["simulation_ready"] is True
    assert simulated["simulate_mode"] == "cv"
    assert simulated["simulation_result"]["status"] == "placeholder"
    assert [row["key"] for row in simulated["parameters"]] == [
        "E0",
        "k0",
        "alpha",
        "D_A",
        "D_B",
        "C_A",
        "C_B",
    ]
    assert [row["key"] for row in simulated["cell_parameters"]] == ["T", "Ru", "Cdl", "A"]
    assert simulated["cv_data_settings"]["cv_index"] == 0
    assert model_fit_gate(simulated) == (False, False, "")


def test_browser_model_parameter_rows_depend_on_simulation_mode():
    from ecat_app.callbacks import model_parameter_rows

    scratch = model_parameter_rows("E", "scratch")
    cv = model_parameter_rows("E", "cv")
    conditions = model_parameter_rows("E", "scratch", over_conditions=True)

    assert [row["key"] for row in scratch] == [
        "E0",
        "k0",
        "alpha",
        "D_A",
        "D_B",
        "C_A",
        "C_B",
    ]
    assert [row["key"] for row in cv] == [row["key"] for row in scratch]
    assert any(row["key"] == "condition_axis" and row["initial"] == "scan_rate" for row in conditions)
    assert any(row["key"] == "condition_values" for row in conditions)


def test_browser_model_input_settings_parse_sidebar_controls():
    from ecat_app.callbacks import (
        model_bound_column_defs,
        model_action_button_labels,
        model_cv_source_state,
        model_cv_data_settings_from_controls,
        model_cv_window_visibility,
        model_fit_index_from_cv_index,
        model_fit_table_state,
        model_input_card_visibility,
        model_program_settings_from_controls,
    )
    from ecat_app.state import SessionRegistry

    class cv:
        pass

    class ca:
        pass

    program = model_program_settings_from_controls(0, -1.5, 1, 0.25, 0.2, 3, 400, 5)
    default_program = model_program_settings_from_controls()
    cv_data = model_cv_data_settings_from_controls(2, "strict", -1.7, -1.0, "2-4", 15, "auto")
    registry = SessionRegistry()
    dataset_id = registry.put([ca(), cv()])
    no_cv_dataset_id = registry.put([ca()])

    assert default_program["E_high"] == 0.0
    assert program == {
        "Ei": 0.0,
        "E_low": -1.5,
        "E_high": 1.0,
        "Ef": 0.25,
        "scan_rate": 0.2,
        "segments": 3,
        "points_per_segment": 400,
        "quiet_time": 5.0,
        "plot quiet time": False,
    }
    assert cv_data["cv_index"] == 2
    assert cv_data["trim mode"] == "strict"
    assert cv_data["potential window"] == [-1.7, -1.0]
    assert cv_data["segments"] == [2, 3, 4]
    assert cv_data["stride"] == 15
    assert cv_data["estimate Cdl"] == "auto"
    no_trim = model_cv_data_settings_from_controls(0, "none", -1.7, -1.0)
    assert no_trim == {"cv_index": 0, "stride": 20, "estimate Cdl": "auto"}
    assert "trim mode" not in no_trim
    assert "potential window" not in no_trim
    assert model_cv_window_visibility("none") == {"display": "none"}
    assert model_cv_window_visibility("expand") == {}
    assert model_input_card_visibility("scratch") == ({}, {"display": "none"})
    assert model_input_card_visibility("cv") == ({"display": "none"}, {})
    assert model_fit_index_from_cv_index("3", "cv", "1") == "3"
    assert model_fit_index_from_cv_index("3", "scratch", "1") == "1"
    cv_options, mode, cv_index, cv_disabled, cv_status, fit_index, fit_disabled = model_cv_source_state(
        {"dataset_id": dataset_id},
        "cv",
        registry=registry,
    )
    assert cv_options[1] == {"label": "From CV", "value": "cv", "disabled": False}
    assert mode == "cv"
    assert cv_index == "1"
    assert cv_disabled is False
    assert cv_status == ""
    assert fit_index == "1"
    assert fit_disabled is False
    no_cv_options, mode, cv_index, cv_disabled, cv_status, fit_index, fit_disabled = model_cv_source_state(
        {"dataset_id": no_cv_dataset_id},
        "cv",
        registry=registry,
    )
    assert no_cv_options[1] == {"label": "From CV", "value": "cv", "disabled": True}
    assert mode == "scratch"
    assert cv_index == ""
    assert cv_disabled is True
    assert cv_status == "No CV objects loaded."
    assert fit_index == ""
    assert fit_disabled is True
    assert model_action_button_labels("scratch") == ("Plot CV Program", "Simulate CV")
    assert model_action_button_labels("cv") == ("Plot CV Program", "Simulate CV")
    assert model_fit_table_state("simulate", {"simulation_ready": True}) == (False, {"display": "none"})
    assert model_fit_table_state("fit", {"simulation_ready": False}) == (False, {"display": "none"})
    assert model_fit_table_state("fit", {"simulation_ready": True}) == (True, {})
    assert [column["hide"] for column in model_bound_column_defs([]) if column["field"] in {"lower", "upper"}] == [True, True]
    assert [column["hide"] for column in model_bound_column_defs(["bounds"]) if column["field"] in {"lower", "upper"}] == [False, False]


def test_browser_model_fit_request_uses_existing_simulation_state():
    from ecat_app.callbacks import build_model_fit_state

    model_options = {
        "mechanism": "EC",
        "mechanism_valid": True,
        "simulation_ready": True,
        "simulation_result": {"status": "placeholder"},
        "parameters": [{"name": "mechanism", "initial": "EC", "vary": False}],
    }

    fit_state = build_model_fit_state(model_options, "multiple")

    assert fit_state["fit_requested"] is False
    assert fit_state["fit_mode"] == "multiple"
    assert fit_state["fit_result"]["status"] == "blocked"
    assert "group-fitting" in fit_state["fit_result"]["message"]
    assert fit_state["simulation_result"] == {"status": "placeholder"}


def test_browser_model_fit_request_preserves_edited_parameter_rows():
    from ecat_app.callbacks import build_model_fit_state

    model_options = {
        "mechanism": "E",
        "mechanism_valid": True,
        "simulation_ready": True,
        "parameters": [{"name": "E0", "initial": -0.4, "unit": "V", "vary": True}],
    }
    edited_rows = [{"name": "E0", "initial": -0.55, "unit": "V", "vary": False}]

    fit_state = build_model_fit_state(model_options, "single", edited_rows)

    assert fit_state["fit_requested"] is True
    assert fit_state["parameters"] == edited_rows
    assert fit_state["fit_result"] == {"status": "pending", "message": "Fit request ready."}


def test_browser_model_fit_single_cv_selection_checks_compatibility(cv_factory):
    from ecat_app.callbacks import build_model_fit_state

    cv_obj = cv_factory()
    cv_obj.scan_rate = 0.2
    model_options = {
        "mechanism": "E",
        "mechanism_valid": True,
        "simulation_ready": True,
        "simulate_mode": "scratch",
        "program_settings": {"scan_rate": 0.1},
    }

    fit_state = build_model_fit_state(model_options, "single", fit_cv_index=0, objects=[cv_obj])

    assert fit_state["fit_cv_index"] == 0
    assert fit_state["fit_result"]["status"] == "pending"
    assert "Scan-rate mismatch" in fit_state["fit_result"]["message"]

    blocked = build_model_fit_state(model_options, "single", fit_cv_index=2, objects=[cv_obj])

    assert blocked["fit_requested"] is False
    assert blocked["fit_result"]["status"] == "blocked"


def test_browser_model_results_content_formats_parameter_rows():
    from ecat_app.callbacks import build_model_simulation_state, model_results_content

    model_options = {
        "mechanism": "E",
        "mechanism_valid": True,
    }
    simulated = build_model_simulation_state(model_options, "scratch")

    parameters, cell_parameters, result = model_results_content(simulated)

    assert parameters == ""
    assert cell_parameters == ""
    assert "Simulated CV" in repr(result)
    assert "Simulation mode: scratch" in repr(result)


def test_browser_model_results_keep_program_and_simulated_cv_entries():
    from ecat_app.callbacks import model_result_entries, model_results_content

    model_options = {
        "simulate_mode": "scratch",
        "program_result": {
            "status": "ok",
            "message": "CV program plotted.",
            "plot": "data:image/png;base64,program",
        },
        "simulation_result": {
            "status": "ok",
            "message": "Simulation complete.",
            "plot": "data:image/png;base64,simulation",
        },
    }

    entries = model_result_entries(model_options)
    rendered = repr(model_results_content(model_options)[2])

    assert [entry["key"] for entry in entries] == ["program", "simulation"]
    assert entries[0]["title"] == "CV Program"
    assert entries[1]["title"] == "Simulated CV"
    assert "id='ecat-model-result-program'" in rendered
    assert "id='ecat-model-result-simulation'" in rendered
    assert rendered.index("CV Program") < rendered.index("Simulated CV")
    assert "data:image/png;base64,program" in rendered
    assert "data:image/png;base64,simulation" in rendered
    assert rendered.count("/assets/ecat_icon_plot_copy.svg") >= 2
    assert rendered.count("/assets/ecat_icon_plot_save.svg") >= 2


def test_browser_model_layout_shows_default_scratch_parameter_rows():
    pytest.importorskip("dash")
    from ecat_app.layout import create_layout

    rendered = repr(create_layout())

    assert "id='ecat-model-results'" in rendered
    assert "style={'display': 'none'}" not in rendered[rendered.index("id='ecat-model-results'"):rendered.index("id='ecat-model-results-content'")]
    assert "Setup" in rendered
    assert "Cell" in rendered
    assert "Species" in rendered
    assert "Mechanism" in rendered
    assert "ecat-model-spatial-mode" in rendered
    assert "ecat-model-spatial-nx" in rendered
    assert "ecat-model-spatial-viscosity-source" in rendered
    assert "MeCN (4.7e-7 m²/s)" in rendered
    assert "ecat-model-spatial-viscosity-custom" in rendered
    assert "custom viscosity" in rendered
    assert "ecat-model-spatial-solvent" not in rendered
    assert "ecat-model-setup-parameters-grid" not in rendered
    assert "ecat-model-cell-parameters-grid" in rendered
    assert "ecat-model-species-parameters-grid" in rendered
    assert "ecat-model-mechanism-parameters-grid" in rendered
    assert "'key': 'T'" in rendered
    assert "'key': 'E0'" in rendered
    assert "'key': 'C_A'" in rendered
    assert "'key': 'C_B'" in rendered
    assert "'key': 'D_A'" in rendered
    assert "'key': 'D_B'" in rendered
    assert "'name': 'E⁰ (V)'" in rendered
    mechanism_grid = rendered[rendered.index("id='ecat-model-mechanism-parameters-grid'"):]
    mechanism_columns = mechanism_grid[:mechanism_grid.index("defaultColDef")]
    assert "'field': 'path'" not in mechanism_columns
    assert mechanism_columns.index("'field': 'step'") < mechanism_columns.index("'field': 'name'")


def test_browser_model_settings_grids_auto_height_to_row_count():
    dag = pytest.importorskip("dash_ag_grid")
    from ecat_app.layout import _model_parameter_grid

    parameter_grid = _model_parameter_grid(
        dag,
        "ecat-model-test-parameters-grid",
        row_data=[{"key": "T"}, {"key": "Ru"}],
    )

    assert parameter_grid.dashGridOptions["domLayout"] == "autoHeight"
    assert parameter_grid.dashGridOptions["enableCellTextSelection"] is True
    assert parameter_grid.dashGridOptions["ensureDomOrder"] is True
    assert parameter_grid.style == {"width": "100%"}
    assert parameter_grid.rowData == [{"key": "T"}, {"key": "Ru"}]


def test_browser_model_labeled_inputs_keep_units_in_input_row():
    css = Path("apps/workbench/src/ecat_app/assets/app.css").read_text()

    field_row_block = css[css.index(".ecat-model-field-row"):css.index(".ecat-model-field-label")]
    assert "grid-template-columns: minmax(0, 0.9fr) minmax(84px, 1fr) minmax(max-content, auto);" in field_row_block
    assert "min-width: 0;" in field_row_block
    label_block = css[css.index(".ecat-model-field-label"):css.index(".ecat-model-field-unit")]
    assert "min-width: 0;" in label_block
    assert "overflow-wrap: anywhere;" in label_block
    actions_block = css[css.index(".ecat-model-program-actions"):css.index(".ecat-model-program-actions button")]
    assert "grid-template-columns: repeat(auto-fit, minmax(128px, 1fr));" in actions_block
    assert "justify-self: end;" in css[css.index(".ecat-model-field-unit"):css.index(".ecat-model-float-input")]
    assert ".ecat-app" in css
    assert "user-select: text;" in css[css.index(".ecat-app"):css.index(".ecat-status")]
    model_selection_block = css[css.index(".ecat-model-program-card"):css.index(".ecat-status")]
    assert "user-select: text;" in model_selection_block


def test_browser_directional_arrow_plus_toggles_options():
    from ecat_app.callbacks import directional_arrow_options_visibility

    assert directional_arrow_options_visibility(None) == {"display": "none"}
    assert directional_arrow_options_visibility(1) == {}
    assert directional_arrow_options_visibility(2) == {"display": "none"}


def test_browser_model_program_plot_renders_data_uri():
    from ecat_app.callbacks import build_model_simulation_state, model_result_plot

    simulated = build_model_simulation_state({"mechanism": "E", "mechanism_valid": True}, "scratch")

    uri = model_result_plot(simulated)

    assert uri.startswith("data:image/png;base64,")


def test_browser_header_session_label_summarizes_loaded_state():
    from ecat_app.callbacks import header_session_label

    assert header_session_label(None) == "No data loaded"
    assert header_session_label({"summary": []}) == "No data loaded"
    assert (
        header_session_label(
            {
                "summary": [{"index": index} for index in range(13)],
                "workflow": {
                    "source_kind": "local_path",
                    "source_path": "/tmp/Fe_PhOH_testing_data",
                },
            }
        )
        == "Fe_PhOH_testing_data · 13 items"
    )
    assert (
        header_session_label(
            {
                "summary": [{"index": 0}],
                "workflow": {"source_kind": "upload", "source_path": "/tmp/session/abc"},
            }
        )
        == "Uploaded files · 1 item"
    )


def test_browser_header_contains_center_session_label_busy_indicator_and_shortcuts():
    pytest.importorskip("dash")
    from ecat_app.layout import create_layout

    layout = create_layout()
    rendered = repr(layout)
    session_label = _find_component(layout, "ecat-session-label")
    busy_loading = _find_component(layout, "ecat-global-busy-loading")
    busy_anchor = _find_component(layout, "ecat-global-busy-anchor")

    assert session_label is not None
    assert session_label.children == "No data loaded"
    assert "ecat-header-center" in rendered
    assert "ecat-global-busy" in rendered
    assert "Working" in rendered
    assert busy_loading is not None
    assert busy_loading.target_components["ecat-objects-store"] == "data"
    assert busy_loading.target_components["ecat-default-plot"] == "children"
    assert busy_loading.target_components["ecat-analysis-results-store"] == "data"
    assert busy_loading.target_components["ecat-model-results-content"] == "children"
    assert busy_anchor is not None
    assert getattr(busy_anchor, "hidden", None) is None
    assert busy_anchor.className == "ecat-global-busy-anchor"
    assert "Keyboard Shortcuts" in rendered
    assert "Zoom in" in rendered
    assert "Zoom out" in rendered
    assert "Reset zoom" in rendered
    assert "Cmd/Ctrl" in rendered


def test_browser_dash_layout_contains_expected_tabs():
    pytest.importorskip("dash")
    from ecat_app.layout import TAB_IDS, create_layout

    layout = create_layout()
    rendered = repr(layout)

    def find_component(component, component_id):
        if getattr(component, "id", None) == component_id:
            return component
        children = getattr(component, "children", None)
        if children is None:
            return None
        if not isinstance(children, (list, tuple)):
            children = [children]
        for child in children:
            found = find_component(child, component_id)
            if found is not None:
                return found
        return None

    assert TAB_IDS == ("data", "plot", "analyze", "model")
    assert layout is not None
    assert "Data" in rendered
    assert "ecat-data-tabs" in rendered
    assert "ecat-subtab-label" in rendered
    assert "Load data, set references, and export reproducible outputs." not in rendered
    assert "Load files or folders and apply reference settings before analysis." in rendered
    assert "Export selected data, plots, and reproducible notebook code." in rendered
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
    assert "ecat-import-invert-current" not in rendered
    assert "Invert current" not in rendered
    assert "ecat-folder-pick-store" not in rendered
    assert "ecat-upload" in rendered
    assert "ecat-object-table" in rendered
    assert "AgGrid" in rendered
    assert "columnSize='autoSize'" in rendered
    assert "columnDefs" in rendered
    assert "rowData" in rendered
    assert "'paginationPageSize': 20" in rendered
    assert "selectedRows" in rendered
    assert "virtualRowData" not in rendered
    assert "ecat-selected-row-ids-store" in rendered
    assert "dash_table" not in rendered
    assert "ecat-table-extra-columns" in rendered
    assert "ecat-reset-columns" in rendered
    assert "ecat-replot" in rendered
    assert "ecat-save-plot" in rendered
    assert "ecat-plotting-replot" in rendered
    assert "ecat-plotting-save-plot" in rendered
    assert "Refresh plots" in rendered
    assert "Save all plots" in rendered
    assert "ecat-primary-button" in rendered
    assert "ecat-sidebar-toggle" in rendered
    assert "ecat-sidebar-resizer" in rendered
    assert "ecat-left-tabs" in rendered
    assert "ecat-tab-symbol" in rendered
    assert "/assets/ecat_icon_data.svg" in rendered
    assert "/assets/ecat_icon_import.svg" in rendered
    assert "/assets/ecat_icon_plotting.svg" in rendered
    assert "/assets/ecat_icon_analysis.svg" in rendered
    assert "/assets/ecat_icon_model.svg" in rendered
    assert "/assets/ecat_icon_export.svg" in rendered
    assert "ecat-app-header" in rendered
    assert "ecat-header-logo" in rendered
    assert "/assets/ecat-logo_2_accent.svg" in rendered
    assert "/assets/ecat-logo.svg" not in rendered
    assert "eCAT Workbench" in rendered
    assert "Visual analysis workspace" in rendered
    assert "Browser analysis workspace" not in rendered
    assert "ecat-header-actions" in rendered
    assert "ecat-zoom-controls" in rendered
    assert "data-ecat-zoom-action" in rendered
    assert "Zoom" in rendered
    assert "100%" in rendered
    assert "Zoom out" in rendered
    assert "Zoom in" in rendered
    assert "Toggle full screen" not in rendered
    assert "/assets/ecat_icon_fullscreen.svg" not in rendered
    assert "data-ecat-zoom-action='fullscreen'" not in rendered
    assert "ecat-window-resize-controls" not in rendered
    assert "data-ecat-window-size" not in rendered
    assert "ecat-about-button" in rendered
    assert "ecat-about-panel" in rendered
    assert "About eCAT Workbench" in rendered
    import ecat

    assert f"ecat {ecat.__version__}" in rendered
    assert "Luke Elissiry" in rendered
    assert "MIT License" in rendered
    assert "ElectroKitty" in rendered
    assert "BSD 3-Clause" in rendered
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
    assert "ecat-single-segment-mode" in rendered
    assert "ecat-single-segment-slider" in rendered
    assert "ecat-single-segment-text" in rendered
    assert "ecat-single-dimensionless-normalize" in rendered
    assert "ecat-single-dimensionless-options" in rendered
    assert "ecat-single-dimensionless-e0" in rendered
    assert "Dimensionless normalization" in rendered
    assert "E⁰ / V" in rendered
    assert "Temperature / K" in rendered
    assert "D / cm² s⁻¹" in rendered
    assert "C / M" in rendered
    assert "ecat-single-dimensionless-area-mode" in rendered
    assert "Area / cm²" in rendered
    assert "Radius / mm" in rendered
    assert "Species / label" not in rendered
    assert "C unit / M" not in rendered
    assert "value=0" in rendered
    assert "Half wave potential" in rendered
    assert "ecat-analysis-results" in rendered
    assert "Analysis Results" in rendered
    assert "Tune static plots, animations, labels, legends, and export settings." in rendered
    assert "Labels" in rendered
    assert "Axis Labels" in rendered
    assert "ecat-plot-axis-label-mode" in rendered
    assert "ecat-plot-axis-label-inputs" in rendered
    assert "ecat-plot-x-label" in rendered
    assert "ecat-plot-y-label" in rendered
    assert "Legend" in rendered
    assert "ecat-plot-style" in rendered
    assert "Plot Style" in rendered
    assert "Notebook" in rendered
    assert "Publication" in rendered
    assert "Saveant" in rendered
    assert "Matplotlib" in rendered
    assert "Line + markers" not in rendered
    assert "ecat-plot-legend" in rendered
    assert "Visibility" not in rendered
    assert "ecat-plot-gradients" in rendered
    assert "ecat-plot-colorbar-wrap" in rendered
    assert "ecat-plot-title-mode" in rendered
    assert "ecat-plot-custom-title-wrap" in rendered
    assert "ecat-plot-convention" in rendered
    assert "Convention" in rendered
    assert "ecat-plot-invert-y-axis" in rendered
    assert "Invert y axis" in rendered
    assert rendered.index("ecat-plot-convention") < rendered.index("ecat-plot-invert-y-axis")
    assert rendered.index("Labels") < rendered.index("Axis Labels") < rendered.index("Title") < rendered.index("Legend") < rendered.index("Display")
    assert "IUPAC" in rendered
    assert "value='IUPAC'" in rendered
    assert "ecat-plot-trim-enabled" in rendered
    assert "ecat-plot-trim-bounds" in rendered
    assert "ecat-plot-trim-mode" in rendered
    assert "ecat-plot-trim-min" in rendered
    assert "ecat-plot-trim-max" in rendered
    assert "ecat-plot-format" in rendered
    assert "ecat-plot-dpi" in rendered
    assert "ecat-plot-colorbar" in rendered
    assert "Allow colorbars" in rendered
    assert "ecat-plot-label-options" in rendered
    assert "value=['colorbar']" in rendered
    assert "ecat-plot-output-mode" in rendered
    assert "Animated" in rendered
    assert "Animate</" not in rendered
    assert "ecat-animation-options" in rendered
    assert "ecat-animation-card" in rendered
    assert "Output" not in rendered[rendered.index("Tune static plots"):rendered.index("Labels")]
    assert rendered.index("Display") < rendered.index("ecat-plot-output-mode") < rendered.index("ecat-animation-options")
    assert "ecat-plot-section" in rendered
    assert "ecat-control-subheading" in rendered
    assert "ecat-animation-fps" in rendered
    assert "ecat-animation-stride" in rendered
    assert "Framerate (FPS)" in rendered
    assert "Stagger time (s)" in rendered
    assert "Duration / rate" in rendered
    assert "End hold (s)" in rendered
    assert "Advanced Animation" in rendered
    assert "Annotations" in rendered
    assert "ecat-add-directional-arrow" in rendered
    assert "ecat-directional-arrow-options" in rendered
    assert "ecat-animation-arrow-potential" in rendered
    assert "ecat-animation-scale-bar-enabled" in rendered
    assert rendered.index("ecat-plot-offset-controls") < rendered.index("Annotations") < rendered.index("ecat-plot-format")
    assert "ecat-plot-offset" in rendered
    assert "ecat-plot-offset-controls" in rendered
    assert "ecat-plot-scale-bar-height" in rendered
    assert "ecat-plot-scale-bar-location" in rendered
    assert "ecat-plot-offset-axis-options" in rendered
    plot_slice = rendered[rendered.index("Tune static plots"):rendered.index("Data Export")]
    assert "ecat-option-disclosure" not in plot_slice
    assert "Data Export" in rendered
    assert rendered.index("Data Export") < rendered.index("ecat-export-filename")
    assert "Notebook Export" in rendered
    assert "Python Preview" in rendered
    assert "Cyclic Voltammetry" in rendered
    assert "Run CV, CA, and CP analyses on the selected imported data." in rendered
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
    assert "ecat-multi-analysis-equations" in rendered
    assert rendered.index("ecat-multi-analysis") < rendered.index("ecat-multi-analysis-equations") < rendered.index("ecat-multi-analysis-options")
    assert "value='none'" in rendered
    assert "ecat-multi-analysis-options" in rendered
    assert "ecat-multi-option-row" in rendered
    assert "ecat-sevcik-options" in rendered
    assert "ecat-sevcik-mode" not in rendered
    assert "htmlFor='ecat-multi-segments'" in rendered
    assert "htmlFor='ecat-fowa-redox-potential'" in rendered
    assert "htmlFor='ecat-tafel-tof-max'" in rendered
    assert "Pre-processing" in rendered
    assert "ecat-multi-preprocess-scale" in rendered
    assert "ecat-preprocessing-panel" in rendered
    assert "ecat-multi-scale-options" in rendered
    assert "ecat-multi-scale-type" in rendered
    assert "ecat-multi-scale-reference-mode" in rendered
    assert "ecat-multi-scale-reference-fields" in rendered
    assert "ecat-multi-scale-guess-potential" in rendered
    assert "ecat-multi-scale-manual-fields" in rendered
    assert "ecat-multi-normalize-mode" in rendered
    assert "ecat-multi-dimensionless-options" in rendered
    assert "ecat-multi-dimensionless-area-mode" in rendered
    assert "ecat-multi-current-normalization-options" in rendered
    assert "ecat-multi-current-normalization-type" in rendered
    assert "ecat-multi-current-reference-fields" in rendered
    assert "ecat-multi-current-manual-fields" in rendered
    assert "ecat-multi-current-ip0" in rendered
    assert "iₚ⁰" in rendered
    assert rendered.index("ecat-multi-analysis-options") < rendered.index("ecat-run-multi-analysis")
    assert "ecat-run-multi-analysis" in rendered
    assert "ecat-filter-key" not in rendered
    assert "ecat-sort-keys" not in rendered
    assert "ecat-group-keys" not in rendered
    assert "Model Settings" in rendered
    assert "Model Results" in rendered
    assert "id='ecat-model-settings'" in rendered
    assert "id='ecat-model-results'" in rendered
    assert "id='ecat-model-settings-card'" in rendered
    assert "id='ecat-model-results-card'" in rendered
    progress_loading = find_component(layout, "ecat-model-simulation-progress-loading")
    progress_anchor = find_component(layout, "ecat-model-simulation-progress-anchor")
    settings_card = rendered[rendered.index("id='ecat-model-settings-card'"):rendered.index("id='ecat-model-results'")]
    results_card = rendered[rendered.index("id='ecat-model-results-card'"):rendered.index("id='ecat-model-results-content'")]
    assert "open=True" not in settings_card
    assert "open=True" not in results_card
    assert rendered.index("id='ecat-model-settings'") < rendered.index("id='ecat-model-results'")
    model_results_block = rendered[rendered.index("id='ecat-model-results'"):]
    assert "ecat-model-cell-parameters-grid" not in model_results_block
    assert "ecat-model-species-parameters-grid" not in model_results_block
    assert "ecat-model-mechanism-parameters-grid" not in model_results_block
    assert "style={'display': 'none'}" not in rendered[rendered.index("id='ecat-model-results'"):rendered.index("id='ecat-model-results-content'")]
    assert progress_loading is not None
    assert progress_loading.target_components == {"ecat-model-simulation-progress-anchor": "children"}
    assert progress_anchor is not None
    assert "ecat-model-progress" in rendered
    assert "Running simulation" in rendered
    assert "Build a mechanism once, then simulate or fit it against CV data." in rendered
    assert "ecat-model-simulate-mode" in rendered
    assert "From Scratch" in rendered
    assert "Over Conditions" in rendered
    assert "V s⁻¹" in rendered
    assert "ecat-model-program-actions" in rendered
    assert rendered.index("ecat-model-plot-program") < rendered.index("ecat-model-run-simulate")
    assert "id='ecat-model-simulation-setup-card'" in rendered
    program_card = repr(find_component(layout, "ecat-model-program-card"))
    cv_data_card = repr(find_component(layout, "ecat-model-cv-data-card"))
    setup_card = repr(find_component(layout, "ecat-model-simulation-setup-card"))
    assert "Simulation Setup" not in program_card
    assert "ecat-model-spatial-mode" not in program_card
    assert "ecat-model-run-simulate" not in program_card
    assert "ecat-model-plot-cv-program" not in program_card
    assert "CV Data" in cv_data_card
    assert "ecat-model-plot-cv-program" in cv_data_card
    assert "Simulation Setup" in setup_card
    assert "ecat-model-spatial-mode" in setup_card
    assert "ecat-model-over-conditions" in setup_card
    assert "ecat-model-over-conditions-card" in setup_card
    assert "ecat-model-run-simulate" in setup_card
    assert "Simulate CV" in rendered
    assert "Simulate CV ·" not in rendered
    assert "Plot CV Program ·" not in rendered
    assert "ecat-model-condition-range" in rendered
    assert "ecat-model-condition-count" in rendered
    assert "ecat-model-condition-species" in rendered
    assert "ecat-model-fit-mode" in rendered
    assert "ecat-model-fit-cv-index" in rendered
    assert "ecat-model-fit-compatibility" in rendered
    assert "ecat-model-mechanism-source" in rendered
    assert "ecat-model-mechanism-preset" in rendered
    assert "Square Scheme" in rendered
    assert "ecat-model-mechanism-custom" in rendered
    assert "Enter one eCAT reaction per line." in rendered
    assert "2A or repeated terms such as A+A" in rendered
    assert "Example: E(1):Fe2=Fe1, then C:Fe1>Fe0 on the next line." in rendered
    assert "ecat-model-mechanism-status" in rendered
    assert "ecat-model-formatted-equations" in rendered
    assert "E(1): A ⇌ B" in rendered
    assert "ecat-model-tabs" in rendered
    assert "ecat-model-fit-tab" in rendered
    assert "disabled=True" in rendered
    assert "ecat-model-cell-parameters-grid" in rendered
    assert "ecat-model-species-parameters-grid" in rendered
    assert "Mechanism Details" not in rendered
    assert "ecat-model-mechanism-details-grid" not in rendered
    assert "Reactants" not in rendered
    assert "Products" not in rendered
    assert "ecat-model-mechanism-parameters-grid" in rendered
    assert "cellEditor" in rendered
    assert "agCheckboxCellRenderer" not in rendered
    assert "ecat-model-mechanism-parameters-content" in rendered
    assert "ecat-model-results-content" in rendered
    assert "Run a simulation first to create the fit starting guess." in rendered
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
    assert "Download Python Notebook" in rendered


def test_browser_model_default_preset_starts_with_simulate_enabled():
    pytest.importorskip("dash")
    from ecat_app.layout import create_layout

    def find_component(component, component_id):
        if getattr(component, "id", None) == component_id:
            return component
        children = getattr(component, "children", None)
        if children is None:
            return None
        if not isinstance(children, (list, tuple)):
            children = [children]
        for child in children:
            found = find_component(child, component_id)
            if found is not None:
                return found
        return None

    button = find_component(create_layout(), "ecat-model-run-simulate")

    assert button is not None
    assert button.disabled is False


def test_browser_layout_wraps_existing_plot_with_copy_and_save_actions():
    pytest.importorskip("dash")
    from ecat_app.layout import create_layout

    rendered = repr(
        create_layout(
            initial_state={
                "plot": "data:image/png;base64,abc",
                "save_plot": "data:image/svg+xml;base64,def",
            }
        )
    )

    assert "ecat-plot-frame" in rendered
    assert "data-ecat-plot-action" in rendered
    assert "Refresh plot" in rendered
    assert "Copy plot" in rendered
    assert "Save plot" in rendered
    assert "/assets/ecat_icon_plot_refresh.svg" in rendered
    assert "/assets/ecat_icon_plot_copy.svg" in rendered
    assert "/assets/ecat_icon_plot_save.svg" in rendered
    assert "ecat-plot-action-status" in rendered
    assert "data-ecat-save-src='data:image/svg+xml;base64,def'" in rendered
    assert "aria-live='polite'" in rendered
    assert "↻" not in rendered
    assert "⧉" not in rendered
    assert "⇩" not in rendered


def test_browser_multiplot_card_uses_inline_refresh_instead_of_visible_toolbar():
    pytest.importorskip("dash")
    from ecat_app.layout import create_layout

    def find_component(component, component_id):
        if getattr(component, "id", None) == component_id:
            return component
        children = getattr(component, "children", None)
        if children is None:
            return None
        if not isinstance(children, (list, tuple)):
            children = [children]
        for child in children:
            found = find_component(child, component_id)
            if found is not None:
                return found
        return None

    layout = create_layout(initial_state={"plot": "data:image/png;base64,abc"})
    rendered = repr(layout)
    default_plot = find_component(layout, "ecat-default-plot")

    assert "ecat-plot-toolbar" not in rendered
    assert find_component(layout, "ecat-replot").hidden is True
    assert find_component(layout, "ecat-save-plot").hidden is True
    assert "data-ecat-plot-action='refresh'" in repr(default_plot)
    assert "Copy plot" in rendered
    assert "Save plot" in rendered
    assert "ecat-plot-action-status" in repr(default_plot)


def test_browser_analysis_results_render_indented_entries():
    from ecat_app.callbacks import render_single_cv_results

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


def test_browser_analysis_results_hide_ok_status_but_keep_errors():
    from ecat_app.callbacks import render_single_cv_results

    def collect_by_class(component, class_name):
        matches = []
        if getattr(component, "className", None) == class_name:
            matches.append(component)
        children = getattr(component, "children", None)
        if children is None:
            return matches
        if not isinstance(children, (list, tuple)):
            children = [children]
        for child in children:
            matches.extend(collect_by_class(child, class_name))
        return matches

    component = render_single_cv_results(
        {
            "message": "Object 0",
            "plot": None,
            "results": [
                {"analysis": "peak_potential", "status": "ok", "value": 0.1, "message": ""},
                {"analysis": "peak_current", "status": "error", "value": None, "message": "failed"},
            ],
        }
    )

    status_text = [node.children for node in collect_by_class(component, "ecat-analysis-result-status")]
    message_text = [node.children for node in collect_by_class(component, "ecat-analysis-result-message")]

    assert "ok" not in status_text
    assert "error" in status_text
    assert "failed" in message_text


def test_browser_analysis_results_render_structured_values_as_tables():
    from ecat_app.callbacks import render_single_cv_results

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


def test_browser_analysis_results_prefers_captured_print_output():
    from ecat_app.callbacks import render_object_analysis_results

    rendered = repr(
        render_object_analysis_results(
            {
                "message": "",
                "plot": None,
                "results": [
                    {
                        "analysis": "peak_potential",
                        "status": "ok",
                        "value": {"Ep": 0.1},
                        "output": "Peak potential: 0.10 V",
                        "message": "",
                    },
                ],
            },
            "Single CV Analysis",
        )
    )

    assert "Peak potential: 0.10 V" in rendered
    assert "ecat-analysis-result-output" in rendered
    assert "ecat-analysis-value-table" in rendered
    assert "Ep" in rendered


def test_browser_analysis_results_suppresses_captured_styler_repr_when_table_exists():
    from ecat_app.callbacks import render_object_analysis_results

    rendered = repr(
        render_object_analysis_results(
            {
                "message": "",
                "plot": None,
                "results": [
                    {
                        "analysis": "peak_potential",
                        "status": "ok",
                        "value": [{"Metric": "Ep", "Value": "0.1 V"}],
                        "output": "<pandas.io.formats.style.Styler object at 0x11055b250>",
                        "message": "",
                    },
                ],
            },
            "Single CV Analysis",
        )
    )

    assert "pandas.io.formats.style.Styler" not in rendered
    assert "ecat-analysis-value-table" in rendered
    assert "0.1 V" in rendered


def test_browser_analysis_results_render_multiple_plots():
    from ecat_app.callbacks import render_object_analysis_results

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

    assert "ecat-analysis-plot-grid" in rendered
    assert "ecat-analysis-diagnostics" in rendered
    assert "ecat-analysis-output-plot" in rendered
    assert "Potential Plot" in rendered
    assert "Cycle Plot" in rendered
    assert rendered.count("/assets/ecat_icon_plot_copy.svg") >= 2
    assert rendered.count("/assets/ecat_icon_plot_save.svg") >= 2


def test_browser_analysis_results_store_overwrites_matching_technique():
    from ecat_app.callbacks import upsert_analysis_result_store

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
    from ecat_app.callbacks import render_analysis_results_store, upsert_analysis_result_store

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


def test_browser_analysis_results_store_keeps_single_and_multiple_cv_separate():
    from ecat_app.callbacks import render_analysis_results_store, upsert_analysis_result_store

    store = upsert_analysis_result_store(
        {},
        "cv_single",
        {"message": "single result", "plot": None, "results": [{"analysis": "peak_current", "status": "ok", "value": 1, "message": ""}]},
        "Single CV Analysis",
    )
    store = upsert_analysis_result_store(
        store,
        "cv_multi",
        {"message": "multi result", "plot": None, "results": [{"analysis": "sevcik_analysis", "status": "ok", "value": 2, "message": ""}]},
        "Multiple CV Analysis",
    )

    rendered = repr(render_analysis_results_store(store))

    assert rendered.count("ecat-analysis-run") == 2
    assert "Single CV Analysis" in rendered
    assert "Multiple CV Analysis" in rendered
    assert "single result" in rendered
    assert "multi result" in rendered


def test_browser_workflow_updates_with_plot_and_analysis_choices():
    from ecat_app.callbacks import update_workflow_plot_options, update_workflow_single_analysis

    workflow = update_workflow_plot_options({"source_path": "/data"}, {"legend": True})
    workflow = update_workflow_single_analysis(workflow, 2, ["peak_current"])

    assert workflow["plot_options"] == {"legend": True}
    assert workflow["selected_index"] == 2
    assert workflow["analyses"] == ["peak_current"]


def test_browser_multi_analysis_visibility_and_defaults():
    from ecat_app.callbacks import multi_analysis_option_state

    hidden = multi_analysis_option_state(None)
    trumpet = multi_analysis_option_state("trumpet_analysis")
    peak = multi_analysis_option_state("fit_peak_current")

    assert hidden[0] == {"display": "none"}
    assert "Trumpet" in trumpet[1]
    assert trumpet[2] == "1, 2"
    assert peak[0] == {}
    assert peak[2] == ""


def test_browser_plot_controls_map_to_ecat_options():
    from ecat_app.callbacks import plot_options_from_controls

    options = plot_options_from_controls(
        legend_values=["legend"],
        title_mode="manual",
        convention="IUPAC",
        custom_title="My plot",
        invert_y_axis_values=["invert_y_axis"],
        display_values=["grid"],
        gradient_values=["gradients"],
        colorbar_values=["colorbar", "deduplicate"],
        label_values=["deduplicate"],
        trim_values=["trim"],
        trim_min=-0.2,
        trim_max=0.4,
        offset=0.12,
        scale_bar_height=0.05,
        scale_bar_location="upper right",
        offset_axis_values=["hide_y_numbers"],
        save_format="svg",
        dpi=240,
        plot_style="line+markers",
    )

    assert options == {
        "legend": True,
        "legend mode": "colorbar",
        "color mode": "auto",
        "title": "My plot",
        "plot style": "line+markers",
        "plot convention": "IUPAC",
        "invert y axis": True,
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
        "trim mode": "expand",
    }


def test_browser_animation_controls_map_to_ecat_options():
    from ecat_app.callbacks import plot_format_options, plot_options_from_controls

    options = plot_options_from_controls(
        ["legend"],
        output_mode="animate",
        save_format="gif",
        animation_fps=24,
        animation_stride=4,
        animation_trace_mode="instant",
        animation_schedule="staggered",
        animation_stagger_time=0.25,
        animation_timing_mode="duration",
        animation_timing_value=3.5,
        animation_advanced=["include_quiet_time", "loop"],
        animation_end_hold=2,
        animation_arrow_potential=-0.8,
        animation_arrow_segment=2,
        animation_scale_bar_values=["scale_bar"],
        animation_scale_bar_length=25,
        animation_scale_bar_location="upper right",
    )

    assert options["_animate"] is True
    assert options["_format"] == "gif"
    assert options["fps"] == 24
    assert options["stride"] == 4
    assert options["trace mode"] == "instant"
    assert options["schedule"] == "staggered"
    assert options["stagger time"] == 0.25
    assert options["timing mode"] == "normalized"
    assert options["normalized duration"] == 3.5
    assert options["include quiet time"] is True
    assert options["loop"] is True
    assert options["end hold"] == 2
    assert options["directional arrows"] == {"potential": -0.8, "segment": 2}
    assert options["scale bar"] == {"loc": "upper right", "length": 25}
    assert plot_options_from_controls(["legend"], output_mode="animated")["_animate"] is True
    assert plot_format_options("animate")[1] == "html"
    assert plot_format_options("animated")[1] == "html"
    assert [item["value"] for item in plot_format_options("animate")[0]] == ["html", "gif", "mp4"]
    assert plot_format_options("static")[1] == "svg"


def test_browser_annotation_controls_apply_to_static_plots():
    from ecat_app.callbacks import animation_controls_visibility, plot_options_from_controls

    options = plot_options_from_controls(
        ["legend"],
        output_mode="static",
        animation_arrow_potential=-0.7,
        animation_arrow_segment=2,
        animation_scale_bar_values=["scale_bar"],
        animation_scale_bar_length=5,
        animation_scale_bar_location="lower left",
    )

    assert "_animate" not in options
    assert options["directional arrows"] == {"potential": -0.7, "segment": 2}
    assert options["scale bar"] == {"loc": "lower left", "length": 5}
    assert animation_controls_visibility("static", "staggered", ["scale_bar"])[2] == {}


def test_browser_animation_render_drops_private_browser_flag(monkeypatch, cv_factory):
    from ecat_app import figures

    captured = {}

    class FakeAnimation:
        figure = None

        def to_html(self, options=None):
            return "<html>animation</html>"

    def fake_animate(objects, options=None):
        captured["objects"] = objects
        captured["options"] = dict(options or {})
        return FakeAnimation()

    monkeypatch.setattr(figures.e, "animate", fake_animate)
    monkeypatch.setattr(figures.plt, "close", lambda _figure: None)

    uri = figures.render_animation([cv_factory()], {"_animate": True, "_format": "html", "stride": 2})

    assert uri.startswith("data:text/html;base64,")
    assert captured["options"] == {"stride": 2}


def test_browser_plot_controls_map_axis_label_modes():
    from ecat_app.callbacks import plot_options_from_controls

    manual = plot_options_from_controls(
        ["legend"],
        axis_label_mode="manual",
        x_axis_label="Potential / V",
        y_axis_label="Current / A",
    )
    hidden = plot_options_from_controls(["legend"], axis_label_mode="none")
    auto = plot_options_from_controls(["legend"], axis_label_mode="auto")

    assert manual["x label"] == "Potential / V"
    assert manual["y label"] == "Current / A"
    assert hidden["x label"] is False
    assert hidden["y label"] is False
    assert "x label" not in auto
    assert "y label" not in auto


def test_browser_axis_label_inputs_visibility_depends_on_manual_mode():
    from ecat_app.callbacks import axis_label_controls_visibility

    assert axis_label_controls_visibility("manual") == {}
    assert axis_label_controls_visibility("auto") == {"display": "none"}
    assert axis_label_controls_visibility("none") == {"display": "none"}


def test_browser_plot_controls_use_discrete_legend_when_colorbar_unchecked():
    from ecat_app.callbacks import plot_options_from_controls

    options = plot_options_from_controls(["legend"], "auto", gradient_values=["gradients"], colorbar_values=[])

    assert options["legend mode"] == "discrete"
    assert options["color mode"] == "auto"
    assert "min gradient entries" not in options


def test_browser_plot_controls_disable_gradient_coloring_when_unchecked():
    from ecat_app.callbacks import plot_options_from_controls

    options = plot_options_from_controls(["legend"], "auto", gradient_values=[], colorbar_values=["colorbar"])

    assert options["legend mode"] == "discrete"
    assert options["color mode"] == "discrete"


def test_browser_plot_css_centers_images(repo_root):
    css = (repo_root / "apps" / "workbench" / "src" / "ecat_app" / "assets" / "app.css").read_text()

    assert ".ecat-plot" in css
    assert "margin: 0 auto" in css


def test_browser_workspace_uses_independent_scroll_regions(repo_root):
    css = (repo_root / "apps" / "workbench" / "src" / "ecat_app" / "assets" / "app.css").read_text()

    assert "html,\nbody {\n  margin: 0;\n  height: 100%;\n  overflow: hidden;" in css
    workspace_block = css[css.index(".ecat-workspace"):css.index(".ecat-app.ecat-sidebar-collapsed .ecat-sidebar > *")]
    sidebar_start = css.index("\n.ecat-sidebar {\n")
    sidebar_block = css[sidebar_start:css.index(".ecat-sidebar-nav", sidebar_start)]
    main_block = css[css.index(".ecat-main {"):css.index(".ecat-panel")]
    assert "flex: 1 1 auto;" in workspace_block
    assert "overflow: hidden;" in workspace_block
    assert "height: 100%;" in sidebar_block
    assert "overflow-y: auto;" in sidebar_block
    assert "overflow-x: hidden;" in sidebar_block
    assert "height: 100%;" in main_block
    assert "overflow-y: auto;" in main_block
    assert "overflow-x: hidden;" in main_block


def test_browser_zoom_controls_have_css_and_script(repo_root):
    css = (repo_root / "apps" / "workbench" / "src" / "ecat_app" / "assets" / "app.css").read_text()
    script = (repo_root / "apps" / "workbench" / "src" / "ecat_app" / "assets" / "zoom-controls.js").read_text()
    app_block = css[css.index(".ecat-app {"):css.index(".ecat-app.ecat-sidebar-collapsed")]
    workspace_block = css[css.index(".ecat-workspace {"):css.index(".ecat-app.ecat-sidebar-collapsed .ecat-sidebar > *")]

    assert ".ecat-header-actions" in css
    assert ".ecat-zoom-controls" in css
    assert ".ecat-zoom-button" in css
    assert ".ecat-zoom-value" in css
    assert "--ecat-ui-zoom" in css
    assert "zoom:" not in app_block
    assert "zoom: var(--ecat-ui-zoom" in workspace_block
    assert "[data-ecat-zoom-action]" in script
    assert "localStorage.setItem(\"ecat-ui-zoom\"" in script
    assert "ecat-zoom-value" in script
    assert "document.querySelector(\".ecat-workspace\")" in script
    assert "fullscreen" not in script
    assert ".style.width" not in script
    assert ".style.height" not in script
    assert "event.metaKey" in script
    assert "event.ctrlKey" in script
    assert "event.preventDefault()" in script
    assert "window.resizeTo" not in script


def test_browser_analysis_controls_align_labels_with_inputs(repo_root):
    css = (repo_root / "apps" / "workbench" / "src" / "ecat_app" / "assets" / "app.css").read_text()

    assert ".ecat-tab-body label" in css
    assert "align-items: center" in css
    assert ".ecat-analysis-checklist label" in css
    assert "display: flex" in css
    assert ".ecat-potential-control-row" in css
    assert ".ecat-stacked-control-label" in css
    assert "align-self: center" in css
    assert "min-height: 44px" in css


def test_browser_sidebar_resize_script_clears_width_when_collapsed(repo_root):
    script = (repo_root / "apps" / "workbench" / "src" / "ecat_app" / "assets" / "sidebar-resize.js").read_text()

    assert "ecat-sidebar-collapsed" in script
    assert "sidebar.style.width = \"\"" in script
    assert "ecatLastWidth" in script
    assert "startX" in script
    assert "startWidth" in script
    assert "(event.clientX - startX) / currentZoom()" in script
    assert "clamp(event.clientX, 260, 520)" not in script


def test_browser_sidebar_tabs_show_icons_above_text_without_empty_header_row(repo_root):
    css = (repo_root / "apps" / "workbench" / "src" / "ecat_app" / "assets" / "app.css").read_text()

    assert ".ecat-sidebar-nav" in css
    assert "position: relative" in css
    assert ".ecat-sidebar-toggle" in css
    assert "height: 70px" in css
    assert "padding-top: 0" in css
    assert ".ecat-tab-label" in css
    assert "flex-direction: column" in css
    assert ".ecat-tab-icon-tile" in css
    assert ".ecat-tab-symbol" in css
    assert "object-fit: contain" in css
    assert "grid-template-columns: minmax(0, 1fr)" in css
    assert "width: calc(100% - 44px)" in css
    assert "background: #f6f8fa" in css
    assert "flex: 0 0 25%" in css
    assert "width: 25%" in css
    assert ".tab.tab--selected" in css
    assert "background: transparent" in css
    assert "padding: 8px 6px !important" in css
    assert "border-bottom: 1px solid #d9e0e6 !important" in css
    assert ".ecat-sidebar-resizer" in css
    assert "z-index: 30" in css
    assert "width: 12px" in css
    assert "width: 100% !important" in css
    assert "flex: 0 0 auto !important" in css
    assert "height: 48px !important" in css
    assert "min-height: 48px !important" in css


def test_browser_column_picker_keeps_trigger_compact_and_makes_menu_taller(repo_root):
    css = (repo_root / "apps" / "workbench" / "src" / "ecat_app" / "assets" / "app.css").read_text()

    assert "#ecat-table-extra-columns.dash-dropdown" in css
    assert "#ecat-table-extra-columns .dash-dropdown-value" in css
    assert "#ecat-table-extra-columns .dash-dropdown-value-item" in css
    assert "#ecat-table-extra-columns .Select-control" in css
    assert "height: 36px !important" in css
    assert "min-height: 36px !important" in css
    assert "max-height: 36px !important" in css
    assert "flex-direction: row !important" in css
    assert "flex-wrap: nowrap" in css
    assert "text-overflow: ellipsis" in css
    assert "[data-radix-popper-content-wrapper] {" in css
    assert "max-height: 60vh !important" in css
    assert "[data-radix-popper-content-wrapper] .dash-dropdown-menu" in css
    assert "[data-radix-popper-content-wrapper] .dash-dropdown-content" in css
    assert "[data-radix-popper-content-wrapper] .dash-dropdown-options" in css
    modern_menu_block = css[
        css.index("[data-radix-popper-content-wrapper] .dash-dropdown-menu"):
        css.index(".Select-menu,", css.index("[data-radix-popper-content-wrapper] .dash-dropdown-menu"))
    ]
    assert "display: flex !important" in modern_menu_block
    assert "flex-direction: column !important" in modern_menu_block
    assert "align-items: stretch !important" in modern_menu_block
    assert "width: 100% !important" in modern_menu_block
    assert ".dash-dropdown-option" in css
    assert "[data-radix-popper-content-wrapper] [role=\"option\"]" in css
    assert ".Select-option" in css
    assert ".VirtualizedSelectOption" in css
    assert "flex: 0 0 auto !important" in css
    option_block_start = css.index("\n[data-radix-popper-content-wrapper] .dash-dropdown-option,")
    compact_option_block = css[
        option_block_start:
        css.index(".Select-menu,", option_block_start)
    ]
    assert "min-height: 36px !important" in compact_option_block
    assert "padding: 7px 12px !important" in compact_option_block
    assert "line-height: 1.25 !important" in compact_option_block
    assert ".dash-dropdown-content" in css
    assert ".dash-dropdown-options" in css
    assert "flex-direction: column !important" in css
    assert ".Select-menu," in css
    assert ".Select-menu-outer" in css
    assert "overflow-y: auto !important" in css
    assert "display: flex !important" in css
    assert "flex-direction: column;" in css
    assert "align-items: center" in css
    assert "width: 100%;" in css


def test_browser_layout_changes_notify_dropdown_positioners(repo_root):
    assets = repo_root / "apps" / "workbench" / "src" / "ecat_app" / "assets"
    zoom_script = (assets / "zoom-controls.js").read_text()
    sidebar_script = (assets / "sidebar-resize.js").read_text()
    dropdown_script = (assets / "dropdown-position.js").read_text()

    assert "ecat:layout-resized" in zoom_script
    assert "window.dispatchEvent(new Event(\"resize\"));" in zoom_script
    assert "window.requestAnimationFrame" in zoom_script
    assert "ecat:layout-resized" in sidebar_script
    assert "window.dispatchEvent(new Event(\"resize\"));" in sidebar_script
    assert "[data-radix-popper-content-wrapper]" in dropdown_script
    assert ".dash-dropdown" in dropdown_script
    assert ".Select" in dropdown_script
    assert "positionDropdowns" in dropdown_script
    assert "coordinateScaleForWrapper" in dropdown_script
    assert "wrapper.closest(\".ecat-workspace\")" in dropdown_script
    assert "left / scale" in dropdown_script
    assert "top / scale" in dropdown_script
    assert "anchorRect.width / scale" in dropdown_script
    assert "lastInteractedDropdownRoot" in dropdown_script
    assert "rememberDropdownRoot" in dropdown_script
    assert "document.addEventListener(\"pointerdown\", handleDropdownInteraction, true)" in dropdown_script
    assert "document.addEventListener(\"focusin\", handleDropdownInteraction, true)" in dropdown_script
    assert "setImportantStyle(wrapper, \"position\", \"fixed\")" in dropdown_script
    assert "setImportantStyle(wrapper, \"transform\", \"none\")" in dropdown_script
    assert "element.style.setProperty(property, value, \"important\")" in dropdown_script
    assert "document.addEventListener(\"ecat:layout-resized\", schedulePositioning)" in dropdown_script


def test_browser_plot_action_script_handles_copy_and_save(repo_root):
    script = (repo_root / "apps" / "workbench" / "src" / "ecat_app" / "assets" / "plot-actions.js").read_text()
    css = (repo_root / "apps" / "workbench" / "src" / "ecat_app" / "assets" / "app.css").read_text()

    assert "data-ecat-plot-action" in script
    assert "ecatPlotAction === \"refresh\"" in script
    assert "document.getElementById(\"ecat-replot\")" in script
    assert "navigator.clipboard.write" in script
    assert "/ecat-app/save-plot" in script
    assert "saveSourceFor(button)" in script
    assert "dataset.ecatSaveSrc" in script
    assert "link.download = \"ecat-plot.\" +" in script
    assert "setButtonState(button, \"working\"" in script
    assert "setButtonState(button, \"success\"" in script
    assert "setButtonState(button, \"error\"" in script
    assert "Saved to Downloads" in script
    assert "result.filename" in script
    assert "Download requested" in script
    assert "aria-live" in script
    assert "iframe.ecat-animation-frame" in script
    assert ".ecat-plot-actions" in css
    assert "grid-template-columns: minmax(0, auto) auto" in css
    assert ".ecat-plot-action-icon" in css
    assert ".ecat-plot-action.is-success" in css
    assert ".ecat-plot-action.is-error" in css
    assert ".ecat-plot-action-status" in css
    assert "position: static" in css
    assert "opacity: 1" in css


def test_browser_server_plot_save_writes_safe_local_file(tmp_path):
    from ecat_app.app import _plot_download_dir, _save_plot_payload

    saved = _save_plot_payload(
        "data:image/svg+xml;base64,PHN2Zz48L3N2Zz4=",
        "../unsafe.svg",
        download_dir=tmp_path,
    )

    assert saved.parent == tmp_path
    assert saved.name == "unsafe.svg"
    assert saved.read_text() == "<svg></svg>"
    assert _plot_download_dir({}) == Path.home() / "Downloads"

    second = _save_plot_payload(
        "data:image/svg+xml;base64,PHN2Zz48L3N2Zz4=",
        "../unsafe.svg",
        download_dir=tmp_path,
    )

    assert second.parent == tmp_path
    assert second.name == "unsafe-2.svg"
    assert saved.read_text() == "<svg></svg>"
    assert second.read_text() == "<svg></svg>"


def test_browser_plot_action_icons_are_svg_assets(repo_root):
    asset_dir = repo_root / "apps" / "workbench" / "src" / "ecat_app" / "assets"

    for filename in [
        "ecat_icon_plot_refresh.svg",
        "ecat_icon_plot_copy.svg",
        "ecat_icon_plot_save.svg",
    ]:
        svg = (asset_dir / filename).read_text()
        assert "<svg" in svg
        assert 'stroke="#231f20"' in svg
        assert 'fill="none"' in svg

    assert "floppy" in (asset_dir / "ecat_icon_plot_save.svg").read_text().lower()


def test_browser_scroll_target_script_handles_model_result_slots(repo_root):
    script = (repo_root / "apps" / "workbench" / "src" / "ecat_app" / "assets" / "scroll-target.js").read_text()

    assert "model-program" in script
    assert "ecat-model-result-program" in script
    assert "model-simulation" in script
    assert "ecat-model-result-simulation" in script
    assert "model-fit" in script
    assert "ecat-model-result-fit" in script
    assert "document.querySelector(\".ecat-main\")" in script
    assert "main.scrollTo" in script
    assert "delta / currentZoom()" in script
    assert "scrollIntoView" not in script


def test_browser_plot_controls_accept_open_ended_trim_bounds():
    from ecat_app.callbacks import plot_options_from_controls

    options = plot_options_from_controls(
        ["legend"],
        "auto",
        trim_values=["trim"],
        trim_max=0.2,
        trim_mode="pointwise",
    )

    assert options["potential window"] == [None, 0.2]
    assert options["trim mode"] == "pointwise"


def test_browser_plot_controls_default_save_options_are_svg_300():
    from ecat_app.callbacks import plot_options_from_controls

    options = plot_options_from_controls(["legend"], "auto")

    assert options["_format"] == "svg"
    assert options["_dpi"] == 300


def test_browser_plot_title_modes_map_to_ecat_options():
    from ecat_app.callbacks import plot_options_from_controls

    assert plot_options_from_controls(["legend"], "auto")["title"] == "auto"
    assert plot_options_from_controls(["legend"], "none")["title"] is False
    assert plot_options_from_controls(["legend"], "manual", custom_title="A")["title"] == "A"
    assert plot_options_from_controls(["legend"], "manual")["title"] is True
    assert plot_options_from_controls(["legend"], "auto")["plot style"] == "notebook"


def test_browser_display_plot_options_ignore_save_format_and_dpi():
    from ecat_app.callbacks import display_plot_options

    display = display_plot_options({"_format": "svg", "_dpi": 300, "legend": True})

    assert display["_format"] == "png"
    assert display["_dpi"] == 150
    assert display["legend"] is True


def test_browser_plot_control_visibility_depends_on_modes():
    from ecat_app.callbacks import plot_control_visibility

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
    from ecat_app.callbacks import trim_bounds_visibility

    assert trim_bounds_visibility(["trim"]) == {}
    assert trim_bounds_visibility([]) == {"display": "none"}


def test_browser_offset_controls_visibility_depends_on_offset_value():
    from ecat_app.callbacks import offset_controls_visibility

    assert offset_controls_visibility(0.12) == {}
    assert offset_controls_visibility("") == {"display": "none"}
    assert offset_controls_visibility(None) == {"display": "none"}
    assert offset_controls_visibility(0) == {"display": "none"}


def test_browser_about_panel_toggles():
    from ecat_app.callbacks import toggle_about_hidden, toggle_about_state

    assert toggle_about_hidden(None, True) is True
    assert toggle_about_hidden(1, True) is False
    assert toggle_about_hidden(2, False) is True
    assert toggle_about_state(None, True) == (True, "About")
    assert toggle_about_state(1, True) == (False, "X")
    assert toggle_about_state(2, False) == (True, "About")


def test_readme_credits_optional_electrokitty_backend():
    readme = (Path(__file__).resolve().parents[1] / "README.md").read_text(encoding="utf-8")

    assert "ElectroKitty" in readme
    assert "BSD 3-Clause" in readme
    assert "Ožbej Vodeb" in readme


def test_third_party_notices_include_electrokitty_license(repo_root):
    notices = (repo_root / "THIRD_PARTY_NOTICES.md").read_text(encoding="utf-8")

    assert "ElectroKitty" in notices
    assert "BSD 3-Clause License" in notices
    assert "Copyright (c) 2024, Ožbej Vodeb" in notices
    assert "RedrumKid/ElectroKitty" in notices
    assert "d1c5f37b442321f8b5bcf48fd9fd76cdd69daef4" in notices
    assert "1.0.11.5" in notices
    assert "Neither the name of the copyright holder" in notices
    assert "dash-ag-grid" in notices


def test_browser_import_controls_use_placeholders_without_reference_defaults():
    pytest.importorskip("dash")
    from ecat_app.layout import create_layout

    rendered = repr(create_layout())

    assert "Reference label" in rendered
    assert "Reference guess" in rendered
    assert "Fc/Fc+" not in rendered
    assert "id='ecat-reference-guess', value='auto'" not in rendered
    assert "searchable=False" in rendered


def test_browser_render_options_trim_cv_objects(cv_factory):
    from ecat_app.figures import prepare_plot_objects_and_options

    obj = cv_factory(name="100mVs_sample_CO2_MeCN_10mM_Fc_run01")

    objects, options = prepare_plot_objects_and_options(
        [obj],
        {"potential window": [-0.2, 0.2], "legend": True},
    )

    assert objects[0] is not obj
    assert "potential window" not in options


def test_browser_render_options_strip_browser_only_plot_style(cv_factory):
    from ecat_app.figures import prepare_plot_objects_and_options

    obj = cv_factory(name="100mVs_sample")

    objects, options = prepare_plot_objects_and_options(
        [obj],
        {"plot style": "line", "legend": True},
    )

    assert objects == [obj]
    assert "plot style" not in options
    assert options["legend"] is True


def test_browser_render_options_apply_ecat_plot_style(monkeypatch, cv_factory):
    from ecat_app import figures

    calls = []
    monkeypatch.setattr(figures.e, "plotting_style", lambda style: calls.append(style))
    obj = cv_factory(name="100mVs_sample")

    objects, options = figures.prepare_plot_objects_and_options(
        [obj],
        {"plot style": "saveant", "legend": True},
    )

    assert objects == [obj]
    assert "plot style" not in options
    assert calls == ["saveant"]


def test_browser_trim_uses_expand_and_fills_missing_bounds():
    from ecat_app.figures import prepare_plot_objects_and_options

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


def test_browser_trim_uses_selected_trim_mode():
    from ecat_app.figures import prepare_plot_objects_and_options

    class FakeCV:
        def __init__(self):
            self.trim_options = None

        def x(self):
            return [-0.6, -0.1, 0.4]

        def trim(self, options):
            self.trim_options = dict(options)
            return self

    obj = FakeCV()

    prepare_plot_objects_and_options(
        [obj],
        {"potential window": [-0.2, None], "trim mode": "strict", "legend": True},
    )

    assert obj.trim_options == {
        "potential window": [-0.2, 0.4],
        "mode": "strict",
    }


def test_browser_multiplot_applies_axis_labels_with_matplotlib(monkeypatch):
    from ecat_app import figures
    import matplotlib.pyplot as plt

    captured = {}

    def fake_multiplot(objects, options):
        captured["options"] = dict(options)
        fig, ax = plt.subplots()
        ax.plot([0, 1], [0, 1])
        return ax

    def fake_data_uri(fig, format="png", dpi=150):
        ax = fig.axes[0]
        captured["x_label"] = ax.get_xlabel()
        captured["y_label"] = ax.get_ylabel()
        return "data:image/png;base64,abc"

    monkeypatch.setattr(figures.e, "multiplot", fake_multiplot)
    monkeypatch.setattr(figures, "figure_to_data_uri", fake_data_uri)

    figures.render_multiplot(
        [object()],
        {"x label": "Potential / V", "y label": "Current / A", "legend": True},
    )

    assert "x label" not in captured["options"]
    assert "y label" not in captured["options"]
    assert captured["x_label"] == "Potential / V"
    assert captured["y_label"] == "Current / A"


def test_browser_offset_multiplot_adds_scale_bar_and_hides_y_tick_labels(monkeypatch):
    from ecat_app import figures
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
    from ecat_app import figures
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
