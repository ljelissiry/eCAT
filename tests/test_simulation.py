import sys
import types
from copy import deepcopy

import numpy as np
import pandas as pd
import pytest
from matplotlib import pyplot as plt


def test_simulation_namespace_is_exported():
    import ecat

    assert hasattr(ecat, "simulation")
    assert hasattr(ecat.simulation, "cv_program")
    assert hasattr(ecat.simulation, "cv_data")
    assert hasattr(ecat.simulation, "simulate_cv")
    assert hasattr(ecat.simulation, "fit_cv")
    assert hasattr(ecat.simulation, "fit_cvs")
    assert hasattr(ecat.simulation, "get_backend")
    assert hasattr(ecat.simulation, "SimulatedCVInput")
    assert hasattr(ecat.simulation, "SimulatedCV")
    assert hasattr(ecat.simulation, "SimulationGroupFitResult")
    assert not hasattr(ecat.simulation, "SimulationInput")
    assert not hasattr(ecat.simulation, "SimulationResult")


def test_cv_program_returns_simulation_input(ecat_module):
    program = ecat_module.simulation.cv_program(
        Ei=0.2,
        E_low=-0.2,
        scan_rate=0.1,
        points_per_segment=5,
        quiet_time=1.0,
    )

    assert isinstance(program, ecat_module.simulation.SimulatedCVInput)
    assert program.i is None
    assert program.source == "program"
    assert program.E[0] == pytest.approx(0.2)
    assert program.E[-1] == pytest.approx(0.2)
    assert np.min(program.E) == pytest.approx(-0.2)
    assert np.all(np.diff(program.t) >= 0)
    assert program.metadata["scan_rate"] == pytest.approx(0.1)
    assert program.metadata["segments"] == 2
    assert program.metadata["direction"] == "negative"
    assert program.metadata["quiet_time"] == pytest.approx(1.0)
    assert program.metadata["quiet_time_applied"] is False


def test_simulated_cv_input_with_scan_rate_returns_copy_with_new_timebase(ecat_module):
    sim = ecat_module.simulation
    program = sim.cv_program(
        Ei=0.2,
        E_low=-0.2,
        scan_rate=0.1,
        points_per_segment=5,
        quiet_time=1.0,
    )

    updated = program.with_scan_rate(0.2)

    assert isinstance(updated, sim.SimulatedCVInput)
    assert updated is not program
    np.testing.assert_allclose(updated.E, program.E)
    assert updated.i is None
    assert updated.source == program.source
    assert updated.metadata["scan_rate"] == pytest.approx(0.2)
    assert program.metadata["scan_rate"] == pytest.approx(0.1)
    assert updated.metadata["quiet_time"] == pytest.approx(1.0)
    assert updated.metadata["quiet_time_applied"] is False
    np.testing.assert_allclose(updated.t[0:2], [0.0, 0.5])
    np.testing.assert_allclose(updated.t[-1], 4.0)
    np.testing.assert_allclose(program.t[-1], 8.0)


def test_simulated_cv_with_scan_rate_reruns_with_updated_input(ecat_module, monkeypatch):
    sim = ecat_module.simulation
    calls = []

    class Backend:
        name = "fake"

        def simulate(self, input_obj, mechanism_spec, params, options):
            calls.append((input_obj, mechanism_spec.mechanism, params, options))
            return input_obj.E, input_obj.t.copy(), input_obj.t, {"scan_rate": input_obj.metadata["scan_rate"]}

    monkeypatch.setattr(sim, "get_backend", lambda name="electrokitty": Backend())

    program = sim.cv_program(Ei=0.2, E_low=-0.2, scan_rate=0.1, points_per_segment=5)
    result = sim.simulate_cv(program, "E", _basic_params(), options={"plot": False})

    updated = result.with_scan_rate(0.2)

    assert isinstance(updated, sim.SimulatedCV)
    assert updated is not result
    assert updated.input is not result.input
    assert result.input.metadata["scan_rate"] == pytest.approx(0.1)
    assert updated.input.metadata["scan_rate"] == pytest.approx(0.2)
    np.testing.assert_allclose(updated.input.E, result.input.E)
    np.testing.assert_allclose(updated.data["Time"], updated.input.t)
    np.testing.assert_allclose(updated.data["Current"], updated.input.t)
    assert calls[-1][0] is updated.input


def test_simulated_cv_with_params_deep_merges_and_reruns(ecat_module, monkeypatch):
    sim = ecat_module.simulation
    calls = []

    class Backend:
        name = "fake"

        def simulate(self, input_obj, mechanism_spec, params, options):
            calls.append((input_obj, mechanism_spec.mechanism, deepcopy(params), dict(options)))
            cdl = float(params["cell"]["Cdl"])
            substrate = float(params["concentrations"]["bulk"]["Substrate"])
            return input_obj.E, np.full_like(input_obj.E, cdl + substrate), input_obj.t, {"params": params}

    monkeypatch.setattr(sim, "get_backend", lambda name="electrokitty": Backend())
    program = sim.cv_program(Ei=0.2, E_low=-0.2, scan_rate=0.1, points_per_segment=3)
    params = _basic_params()
    params["cell"]["Cdl"] = 1.0
    params["concentrations"]["bulk"]["Substrate"] = 2.0
    result = sim.simulate_cv(program, "E", params, options={"plot": False})

    updated = result.with_params({"cell": {"Cdl": 3.0}})

    assert updated is not result
    assert updated.params["cell"]["Cdl"] == pytest.approx(3.0)
    assert updated.params["cell"]["A"] == pytest.approx(result.params["cell"]["A"])
    assert updated.params["concentrations"]["bulk"]["Substrate"] == pytest.approx(2.0)
    assert result.params["cell"]["Cdl"] == pytest.approx(1.0)
    np.testing.assert_allclose(updated.data["Current"], np.full(len(program.E), 5.0))
    assert calls[-1][3]["plot"] is False


def test_simulated_cv_with_param_updates_path_without_mutating_original(ecat_module, monkeypatch):
    sim = ecat_module.simulation

    class Backend:
        name = "fake"

        def simulate(self, input_obj, mechanism_spec, params, options):
            substrate = float(params["concentrations"]["bulk"]["Substrate"])
            return input_obj.E, np.full_like(input_obj.E, substrate), input_obj.t, {"params": params}

    monkeypatch.setattr(sim, "get_backend", lambda name="electrokitty": Backend())
    program = sim.cv_program(Ei=0.2, E_low=-0.2, scan_rate=0.1, points_per_segment=3)
    params = _basic_params()
    params["concentrations"]["bulk"]["Substrate"] = 2.0
    result = sim.simulate_cv(program, "E", params, options={"plot": False})

    updated = result.with_param("concentrations.bulk.Substrate", 7.0)

    assert updated.params["concentrations"]["bulk"]["Substrate"] == pytest.approx(7.0)
    assert result.params["concentrations"]["bulk"]["Substrate"] == pytest.approx(2.0)
    np.testing.assert_allclose(updated.data["Current"], np.full(len(program.E), 7.0))


def test_simulated_cv_with_input_and_with_mechanism_rerun(ecat_module, monkeypatch):
    sim = ecat_module.simulation
    calls = []

    class Backend:
        name = "fake"

        def simulate(self, input_obj, mechanism_spec, params, options):
            calls.append((input_obj, mechanism_spec.mechanism, deepcopy(params)))
            current = np.full(len(input_obj.E), len(mechanism_spec.mechanism), dtype=float)
            return input_obj.E, current, input_obj.t, {"mechanism": mechanism_spec.mechanism}

    monkeypatch.setattr(sim, "get_backend", lambda name="electrokitty": Backend())
    program = sim.cv_program(Ei=0.2, E_low=-0.2, scan_rate=0.1, points_per_segment=3)
    result = sim.simulate_cv(program, "E", _basic_params(), options={"plot": False})
    new_program = sim.cv_program(Ei=0.1, E_low=-0.3, scan_rate=0.2, points_per_segment=4)

    with_input = result.with_input(new_program)
    with_mechanism = result.with_mechanism("EC", params={**result.params, "concentrations": {"bulk": {"a": 1, "b": 0, "c": 0}}})

    assert with_input.input is new_program
    np.testing.assert_allclose(with_input.data["Potential"], new_program.E)
    assert result.input is program
    assert with_mechanism.mechanism.preset == "EC"
    assert "C:" in with_mechanism.mechanism.mechanism
    assert calls[-2][0] is new_program
    assert calls[-1][1] == "E(1):a=b\nC:b=c"


def test_simulated_cv_exposes_cv_like_data_access_and_segments(ecat_module):
    sim = ecat_module.simulation
    input_obj = sim.SimulatedCVInput(
        E=np.array([0.2, 0.0, -0.2, 0.0, 0.2]),
        t=np.array([0.0, 1.0, 2.0, 3.0, 4.0]),
        metadata={"scan_rate": 0.2, "segments": 2},
        source="synthetic",
    )
    result = sim.SimulatedCV(
        data=pd.DataFrame(
            {
                "Potential": [0.2, 0.0, -0.2, 0.0, 0.2],
                "Current": [1e-6, 2e-6, 3e-6, 4e-6, 5e-6],
                "Time": [0.0, 1.0, 2.0, 3.0, 4.0],
                "Backend Current": [1e-6, 2e-6, 3e-6, 4e-6, 5e-6],
            }
        ),
        params={},
        mechanism=None,
        input=input_obj,
        backend_result=None,
    )

    assert result.type == "Simulated Cyclic Voltammetry"
    assert result.scan_rate == pytest.approx(0.2)
    assert result.segments == 2
    assert result.units["Potential"] == "V"
    assert result.units["Current"] == "A"
    assert result.x().name == "Potential"
    assert result.y().name == "Current"
    np.testing.assert_allclose(result.xy()[0], input_obj.E)
    segment_x, segment_y = result.analysis_segment_data({"segment": 2})
    np.testing.assert_allclose(segment_x, [-0.2, 0.0, 0.2])
    np.testing.assert_allclose(segment_y, [3e-6, 4e-6, 5e-6])


def test_simulated_cv_can_use_cv_analysis_methods(ecat_module):
    sim = ecat_module.simulation
    result = sim.SimulatedCV(
        data=pd.DataFrame(
            {
                "Potential": [0.2, 0.0, -0.2, 0.0, 0.2],
                "Current": [1e-6, 2e-6, 5e-6, 2e-6, 1e-6],
                "Time": [0.0, 1.0, 2.0, 3.0, 4.0],
                "Backend Current": [1e-6, 2e-6, 5e-6, 2e-6, 1e-6],
            }
        ),
        params={},
        mechanism=None,
        input=sim.SimulatedCVInput(
            E=np.array([0.2, 0.0, -0.2, 0.0, 0.2]),
            t=np.array([0.0, 1.0, 2.0, 3.0, 4.0]),
            metadata={"scan_rate": 0.2},
        ),
        backend_result=None,
    )

    assert result.current_at_potential(-0.2, {"segment": 2, "plot": False, "print": False})[2] == pytest.approx((-0.2, 5e-6))


def test_simulated_cv_objects_can_be_multiplotted(ecat_module):
    sim = ecat_module.simulation
    results = []
    for scan_rate, scale in [(0.1, 1.0), (0.2, 2.0)]:
        program = sim.cv_program(Ei=0.2, E_low=-0.2, scan_rate=scan_rate, points_per_segment=5)
        results.append(
            sim.SimulatedCV(
                data=pd.DataFrame(
                    {
                        "Potential": program.E,
                        "Current": scale * np.linspace(1e-6, 5e-6, len(program.E)),
                        "Time": program.t,
                        "Backend Current": scale * np.linspace(1e-6, 5e-6, len(program.E)),
                    }
                ),
                params={},
                mechanism=None,
                input=program,
                backend_result=None,
            )
        )

    ax = ecat_module.multiplot(results, {"legend": True, "title": False})

    assert len(ax.lines) == 2
    assert ax.get_xlabel() == "Potential (V)"
    assert ax.get_ylabel() == "Current (μA)"


def test_simulated_cv_input_show_prints_setup_by_default(ecat_module, capsys):
    program = ecat_module.simulation.cv_program(
        Ei=0.2,
        E_low=-0.2,
        scan_rate=0.1,
        points_per_segment=5,
        quiet_time=1.0,
    )

    program.show()

    out = capsys.readouterr().out
    assert "Simulated CV Input Setup:" in out
    assert "Parameter" in out
    assert "Value" in out
    assert "Scan Rate" in out
    assert "0.1 V/s" in out
    assert "Points" in out
    assert "Has Current" in out
    assert "False" in out


def test_simulated_cv_input_show_can_be_quiet_or_raw(ecat_module, capsys):
    program = ecat_module.simulation.cv_program(0.2, E_low=-0.2, points_per_segment=5)

    program.show({"print setup": False})
    assert capsys.readouterr().out == ""

    program.show({"print setup": "raw"})
    out = capsys.readouterr().out
    assert "Simulated CV Input Setup:" in out
    assert "'E':" in out
    assert "'metadata':" in out


def test_simulated_cv_input_show_pretty_print_false_avoids_rich_display(ecat_module, monkeypatch, capsys):
    sim = ecat_module.simulation
    program = sim.cv_program(0.2, E_low=-0.2, points_per_segment=5)

    import IPython.display as ipd

    rich_calls = []

    monkeypatch.setattr(sim, "_can_rich_display", lambda: True)
    monkeypatch.setattr(ipd, "display", lambda obj: rich_calls.append(obj))
    monkeypatch.setattr(ipd, "Markdown", lambda text: text)

    program.show({"pretty print": False})

    out = capsys.readouterr().out
    assert "Simulated CV Input Setup:" in out
    assert "Parameter" in out
    assert "Value" in out
    assert rich_calls == []


def test_simulated_cv_input_plot_draws_program_with_existing_style_helpers(ecat_module):
    program = ecat_module.simulation.cv_program(
        Ei=0.2,
        E_low=-0.2,
        scan_rate=0.1,
        points_per_segment=5,
    )

    ax = program.plot({"label": "program"})

    assert len(ax.lines) == 1
    assert ax.get_xlabel() == "Time (s)"
    assert ax.get_ylabel() == "Potential (V)"
    np.testing.assert_allclose(ax.lines[0].get_xdata(), program.t)
    np.testing.assert_allclose(ax.lines[0].get_ydata(), program.E)
    plt.close(ax.figure)


def test_simulated_cv_input_plot_can_include_quiet_time_without_mutating_input(ecat_module):
    program = ecat_module.simulation.cv_program(
        Ei=0.2,
        E_low=-0.2,
        scan_rate=0.1,
        points_per_segment=5,
        quiet_time=1.0,
    )
    original_len = len(program.E)

    ax = program.plot({"plot quiet time": True})

    assert len(program.E) == original_len
    assert program.metadata["quiet_time_applied"] is False
    assert len(ax.lines[0].get_xdata()) > original_len
    np.testing.assert_allclose(ax.lines[0].get_xdata()[0:2], [-1.0, 0.0])
    np.testing.assert_allclose(ax.lines[0].get_xdata()[-1], program.t[-1])
    np.testing.assert_allclose(ax.lines[0].get_ydata()[0:2], [0.2, 0.2])
    plt.close(ax.figure)


def test_simulated_cv_input_plot_can_draw_measured_current_vs_potential(ecat_module):
    sim = ecat_module.simulation
    input_obj = sim.SimulatedCVInput(
        E=np.array([0.2, 0.0, -0.2]),
        t=np.array([0.0, 1.0, 2.0]),
        i=np.array([1e-6, 2e-6, 3e-6]),
        metadata={"potential_unit": "V", "current_unit": "A"},
        source="measured",
    )

    ax = input_obj.plot({"x axis": "potential", "y axis": "current", "y unit": "uA"})

    assert len(ax.lines) == 1
    assert ax.get_xlabel() == "Potential (V)"
    assert ax.get_ylabel() == "Current (μA)"
    np.testing.assert_allclose(ax.lines[0].get_xdata(), input_obj.E)
    np.testing.assert_allclose(ax.lines[0].get_ydata(), [1.0, 2.0, 3.0])
    assert ax.get_xlim()[0] > ax.get_xlim()[1]
    plt.close(ax.figure)


def test_simulated_cv_show_prints_two_column_setup(ecat_module, capsys):
    sim = ecat_module.simulation
    program = sim.cv_program(Ei=0.2, E_low=-0.2, scan_rate=0.1, points_per_segment=5)
    result = sim.SimulatedCV(
        data=pd.DataFrame(
            {
                "Potential": program.E,
                "Current": np.linspace(1e-6, 5e-6, len(program.E)),
                "Time": program.t,
                "Backend Current": np.linspace(1e-6, 5e-6, len(program.E)),
            }
        ),
        params=_basic_params(),
        mechanism=sim.compile_mechanism("E", _basic_params()),
        input=program,
        backend_result=None,
        current_sign=-1,
        summary={"backend": "fake", "preset": "E", "current_sign": -1},
    )

    result.show({"print setup": True})

    out = capsys.readouterr().out
    assert "Simulated CV Setup:" in out
    assert "Parameter" in out
    assert "Value" in out
    assert "Backend" in out
    assert "fake" in out
    assert "Mechanism" in out
    assert "E(1):a=b" in out
    assert "Current Sign" in out
    assert "-1" in out
    assert "Scan Rate" in out
    assert "0.1 V/s" in out
    assert "Potential Range" in out
    assert "Current Range" in out
    assert "Simulation Params:" not in out


def test_simulated_cv_show_can_include_params_and_data(ecat_module, capsys):
    sim = ecat_module.simulation
    program = sim.cv_program(Ei=0.2, E_low=-0.2, scan_rate=0.1, points_per_segment=5)
    result = sim.SimulatedCV(
        data=pd.DataFrame(
            {
                "Potential": program.E,
                "Current": np.linspace(1e-6, 5e-6, len(program.E)),
                "Time": program.t,
                "Backend Current": np.linspace(1e-6, 5e-6, len(program.E)),
            }
        ),
        params=_basic_params(),
        mechanism=sim.compile_mechanism("E", _basic_params()),
        input=program,
        backend_result=None,
        summary={"backend": "fake", "preset": "E", "current_sign": 1},
    )

    result.show({"print setup": False, "print params": True, "print data": True})

    out = capsys.readouterr().out
    assert "Simulated CV Setup:" not in out
    assert "Simulation Params:" in out
    assert "[parameters]" in out
    assert "cell.T" in out
    assert "Simulated CV Data:" in out
    assert "Potential" in out
    assert "Current" in out


def test_simulated_cv_show_print_params_compact_uses_compact_tables(ecat_module, capsys):
    sim = ecat_module.simulation
    program = sim.cv_program(Ei=0.2, E_low=-0.2, scan_rate=0.1, points_per_segment=5)
    result = sim.SimulatedCV(
        data=pd.DataFrame(
            {
                "Potential": program.E,
                "Current": np.linspace(1e-6, 5e-6, len(program.E)),
                "Time": program.t,
                "Backend Current": np.linspace(1e-6, 5e-6, len(program.E)),
            }
        ),
        params={
            "concentrations": {"bulk": {"a": 1.0, "b": 0.0}},
            "diffusion": {"a": 1e-9, "b": 2e-9},
            "kinetics": [{"alpha": 0.5, "k0": 1e-3, "E0": -1.4}],
        },
        mechanism=sim.compile_mechanism("E", _basic_params()),
        input=program,
        backend_result=None,
        summary={"backend": "fake", "preset": "E", "current_sign": 1},
    )

    result.show({"print setup": False, "print params": "compact"})

    out = capsys.readouterr().out
    assert "Simulation Params:" in out
    assert "Phase" in out
    assert "Amount" in out
    assert "Diffusion" in out
    assert "concentrations.bulk.a" not in out


def test_cv_program_uses_segments_and_direction_aliases(ecat_module):
    cathodic = ecat_module.simulation.cv_program(
        Ei=0.0,
        E_low=-1.0,
        E_high=0.5,
        direction="cathodic",
        segments=3,
        points_per_segment=3,
    )
    anodic = ecat_module.simulation.cv_program(
        Ei=0.0,
        E_low=-1.0,
        E_high=0.5,
        direction="+",
        segments=3,
        points_per_segment=3,
    )

    np.testing.assert_allclose(cathodic.E[[0, 2, 4, 6]], [0.0, -1.0, 0.5, -1.0])
    np.testing.assert_allclose(anodic.E[[0, 2, 4, 6]], [0.0, 0.5, -1.0, 0.5])
    assert cathodic.metadata["segments"] == 3
    assert cathodic.metadata["direction"] == "negative"
    assert anodic.metadata["direction"] == "positive"


def test_cv_program_rejects_removed_e_vertex_and_cycles_arguments(ecat_module):
    with pytest.raises(TypeError):
        ecat_module.simulation.cv_program(0.0, E_vertex=-1.0)
    with pytest.raises(TypeError):
        ecat_module.simulation.cv_program(0.0, E_low=-1.0, cycles=2)


def test_cv_data_extracts_segments_window_stride_and_time(ecat_module, cv_factory):
    cv_obj = cv_factory()
    data = ecat_module.simulation.cv_data(
        cv_obj,
        {
            "segments": [1, 2],
            "potential window": [-0.15, 0.15],
            "trim mode": "pointwise",
            "stride": 2,
        },
    )

    expected_E, expected_i = cv_obj.analysis_segment_data({"segments": [1, 2]})
    mask = (expected_E >= -0.15) & (expected_E <= 0.15)
    expected_E = np.asarray(expected_E[mask], dtype=float)
    expected_i = np.asarray(expected_i[mask], dtype=float)
    expected_indices = np.r_[np.arange(0, len(expected_E), 2), len(expected_E) - 1]
    expected_indices = np.unique(expected_indices)
    expected_E = expected_E[expected_indices]
    expected_i = expected_i[expected_indices]

    assert isinstance(data, ecat_module.simulation.SimulatedCVInput)
    np.testing.assert_allclose(data.E, expected_E)
    np.testing.assert_allclose(data.i, expected_i)
    assert data.source is cv_obj
    assert data.metadata["scan_rate"] == pytest.approx(cv_obj.scan_rate)
    assert data.metadata["potential_window"] == [-0.15, 0.15]
    assert data.metadata["stride"] == 2
    assert data.metadata["stride_mode"] == "manual"
    assert data.metadata["stride_basis"] == "manual"
    assert data.metadata["original_points"] == int(mask.sum())
    assert data.metadata["selected_points"] == len(expected_E)
    assert data.t[0] == pytest.approx(0.0)
    assert np.all(np.diff(data.t) >= 0)


def test_cv_data_preserves_imported_quiet_time_metadata(ecat_module, tmp_path):
    path = tmp_path / "ch_cv_quiet_time.txt"
    path.write_text(
        "\n".join(
            [
                "Aug. 27, 2023   16:05:21",
                "Cyclic Voltammetry",
                "Instrument Model: CHI760E",
                "Init E = 0",
                "High E = 0",
                "Low E = -0.2",
                "Scan Rate = 0.1",
                "Segment = 2",
                "Quiet Time (sec) = 3",
                "Sample Interval = 0.01",
                "Sensitivity = 1e-6",
                "Potential/V,Current/A",
                "0,-1e-7",
                "-0.1,-2e-7",
                "-0.2,-3e-7",
                "-0.1,-2e-7",
                "0,-1e-7",
            ]
        )
        + "\n",
        encoding="ISO-8859-1",
    )

    cv_obj = ecat_module.echem.from_file(str(path), {})
    data = ecat_module.simulation.cv_data(cv_obj, {"stride": 1})

    assert cv_obj.quiet_time == pytest.approx(3.0)
    assert data.metadata["quiet_time"] == pytest.approx(3.0)
    assert data.metadata["quiet_time_applied"] is False


def test_simulate_cv_applies_imported_quiet_time_to_backend_only(ecat_module, monkeypatch):
    sim = ecat_module.simulation

    class Backend:
        name = "quiet-time-backend"

        def simulate(self, input_obj, mechanism_spec, params, options):
            self.input_obj = input_obj
            return input_obj.E, np.arange(len(input_obj.E), dtype=float), input_obj.t, self

    backend = Backend()
    monkeypatch.setattr(sim, "get_backend", lambda name="electrokitty": backend)
    input_obj = sim.SimulatedCVInput(
        E=np.array([0.0, -0.1, -0.2]),
        i=np.array([1.0, 2.0, 3.0]),
        t=np.array([0.0, 1.0, 2.0]),
        metadata={"kind": "cv_data", "quiet_time": 2.0, "quiet_time_applied": False},
        source="synthetic",
    )

    result = sim.simulate_cv(input_obj, "E", _basic_params(), options={"plot": False})

    assert len(backend.input_obj.E) > len(input_obj.E)
    assert backend.input_obj.E[0] == pytest.approx(0.0)
    assert backend.input_obj.E[1] == pytest.approx(0.0)
    assert backend.input_obj.t[-len(input_obj.t)] == pytest.approx(2.0)
    assert len(result.data) == len(input_obj.E)
    np.testing.assert_allclose(result.data["Potential"], input_obj.E)
    np.testing.assert_allclose(result.data["Time"], input_obj.t)
    assert "Measured Current" not in result.data
    np.testing.assert_allclose(result.input.i, input_obj.i)
    assert result.summary["quiet_time"] == pytest.approx(2.0)
    assert result.summary["quiet_time_applied"] is True


def test_simulate_cv_applies_cv_program_quiet_time_to_backend_only(ecat_module, monkeypatch):
    sim = ecat_module.simulation

    class Backend:
        name = "program-quiet-time-backend"

        def simulate(self, input_obj, mechanism_spec, params, options):
            self.input_obj = input_obj
            return input_obj.E, np.arange(len(input_obj.E), dtype=float), input_obj.t, self

    backend = Backend()
    monkeypatch.setattr(sim, "get_backend", lambda name="electrokitty": backend)
    input_obj = sim.cv_program(0.0, E_low=-0.2, scan_rate=0.1, points_per_segment=3, quiet_time=2.0)

    result = sim.simulate_cv(input_obj, "E", _basic_params(), options={"plot": False})

    assert input_obj.metadata["quiet_time_applied"] is False
    assert len(backend.input_obj.E) > len(input_obj.E)
    assert backend.input_obj.metadata["quiet_time_applied"] is True
    assert len(result.data) == len(input_obj.E)
    np.testing.assert_allclose(result.data["Potential"], input_obj.E)
    np.testing.assert_allclose(result.data["Time"], input_obj.t)
    assert result.summary["quiet_time"] == pytest.approx(2.0)
    assert result.summary["quiet_time_applied"] is True


def test_cv_data_potential_window_expands_disconnected_selection_by_default(ecat_module, cv_factory):
    potential = np.array([0.0, -0.4, -0.8, -0.4, 0.0, 0.4, 0.8, 0.4, 0.0])
    current = np.arange(len(potential), dtype=float)
    cv_obj = cv_factory(name="100mVs_sample_CO2_MeCN_10mM_Fc_run01", potential=potential, current=current)

    data = ecat_module.simulation.cv_data(
        cv_obj,
        {"potential window": [-0.2, 0.2], "stride": 1},
    )

    np.testing.assert_allclose(data.E, potential)
    np.testing.assert_allclose(data.i, current)
    assert data.metadata["trim_mode"] == "expand"
    assert "mode" not in data.metadata
    assert "window_mode" not in data.metadata
    assert data.metadata["window_expanded"] is True
    assert data.metadata["window_break_count"] == 2
    assert data.metadata["potential_window_requested"] == [-0.2, 0.2]
    assert data.metadata["potential_window_effective"] == [float(np.min(potential)), float(np.max(potential))]


def test_cv_data_potential_window_strict_mode_rejects_disconnected_selection(ecat_module, cv_factory):
    potential = np.array([0.0, -0.4, -0.8, -0.4, 0.0, 0.4, 0.8, 0.4, 0.0])
    current = np.arange(len(potential), dtype=float)
    cv_obj = cv_factory(
        name="100mVs_sample_CO2_MeCN_10mM_Fc_run01",
        potential=potential,
        current=current,
    )

    with pytest.raises(ValueError, match="disconnect"):
        ecat_module.simulation.cv_data(
            cv_obj,
            {"potential window": [-0.2, 0.2], "trim mode": "strict"},
        )


def test_cv_data_potential_window_pointwise_mode_preserves_pointwise_trim(ecat_module, cv_factory):
    potential = np.array([0.0, -0.4, -0.8, -0.4, 0.0, 0.4, 0.8, 0.4, 0.0])
    current = np.arange(len(potential), dtype=float)
    cv_obj = cv_factory(
        name="100mVs_sample_CO2_MeCN_10mM_Fc_run01",
        potential=potential,
        current=current,
    )

    data = ecat_module.simulation.cv_data(
        cv_obj,
        {"potential window": [-0.2, 0.2], "trim mode": "pointwise", "stride": 1},
    )

    np.testing.assert_allclose(data.E, [0.0, 0.0, 0.0])
    np.testing.assert_allclose(data.i, [0.0, 4.0, 8.0])
    assert data.metadata["trim_mode"] == "pointwise"
    assert "mode" not in data.metadata
    assert "window_mode" not in data.metadata
    assert data.metadata["window_expanded"] is False


def test_cv_data_rejects_removed_generic_mode_option(ecat_module, cv_factory):
    cv_obj = cv_factory()

    with pytest.raises(ValueError, match="Use 'trim mode'"):
        ecat_module.simulation.cv_data(
            cv_obj,
            {"potential window": [-0.2, 0.2], "mode": "pointwise"},
        )


def test_cv_data_rejects_removed_window_mode_option(ecat_module, cv_factory):
    cv_obj = cv_factory()

    with pytest.raises(ValueError, match="Use 'trim mode'"):
        ecat_module.simulation.cv_data(
            cv_obj,
            {"potential window": [-0.2, 0.2], "window mode": "pointwise"},
        )


def test_cv_data_rejects_removed_allow_mode_value(ecat_module, cv_factory):
    cv_obj = cv_factory()

    with pytest.raises(ValueError, match="trim mode must be"):
        ecat_module.simulation.cv_data(
            cv_obj,
            {"potential window": [-0.2, 0.2], "trim mode": "allow"},
        )


def test_cv_data_auto_stride_defaults_to_points_per_volt(ecat_module, cv_factory):
    potential = np.linspace(-1.0, 1.0, 5001)
    current = np.sin(potential)
    cv_obj = cv_factory(potential=potential, current=current)

    data = ecat_module.simulation.cv_data(cv_obj)

    assert data.metadata["stride_mode"] == "auto"
    assert data.metadata["stride_basis"] == "points_per_volt"
    assert data.metadata["points_per_volt"] == pytest.approx(1000)
    assert data.metadata["target_points"] == 2000
    assert data.metadata["stride"] == 3
    assert data.metadata["original_points"] == 5001
    assert data.metadata["selected_points"] == len(data.E)
    assert data.E[0] == pytest.approx(potential[0])
    assert data.E[-1] == pytest.approx(potential[-1])
    assert len(data.E) == 1668


def test_cv_data_auto_stride_preserves_internal_potential_vertex(ecat_module, cv_factory):
    forward = np.linspace(0.0, -1.0, 1201)
    reverse = np.linspace(-1.0, 0.0, 1201)[1:]
    potential = np.r_[forward, reverse]
    current = np.linspace(0.0, 1.0, len(potential))
    cv_obj = cv_factory(potential=potential, current=current)

    data = ecat_module.simulation.cv_data(
        cv_obj,
        {"potential window": [-1.0, 0.0], "points": 300},
    )

    assert data.E[0] == pytest.approx(0.0)
    assert data.E[-1] == pytest.approx(0.0)
    assert np.min(data.E) == pytest.approx(-1.0)


def test_cv_data_points_option_sets_auto_stride_target(ecat_module, cv_factory):
    potential = np.linspace(-0.5, 0.5, 1001)
    current = np.cos(potential)
    cv_obj = cv_factory(potential=potential, current=current)

    data = ecat_module.simulation.cv_data(cv_obj, {"points": 250})

    assert data.metadata["stride_mode"] == "auto"
    assert data.metadata["stride_basis"] == "points"
    assert data.metadata["target_points"] == 300
    assert data.metadata["stride"] == 4
    assert data.E[0] == pytest.approx(potential[0])
    assert data.E[-1] == pytest.approx(potential[-1])


def test_cv_data_points_per_volt_and_min_max_options(ecat_module, cv_factory):
    potential = np.linspace(-0.1, 0.1, 1201)
    current = np.zeros_like(potential)
    cv_obj = cv_factory(potential=potential, current=current)

    data = ecat_module.simulation.cv_data(
        cv_obj,
        {"points per volt": 4000, "min points": 50, "max points": 500},
    )

    assert data.metadata["stride_basis"] == "points_per_volt"
    assert data.metadata["points_per_volt"] == pytest.approx(4000)
    assert data.metadata["target_points"] == 500
    assert data.metadata["stride"] == 3
    assert data.E[-1] == pytest.approx(potential[-1])


def test_cv_data_can_preserve_all_points_with_manual_stride_one(ecat_module, cv_factory):
    cv_obj = cv_factory()

    data = ecat_module.simulation.cv_data(cv_obj, {"stride": 1})
    expected_E, expected_i = cv_obj.analysis_segment_data({})

    assert data.metadata["stride_mode"] == "manual"
    assert data.metadata["stride"] == 1
    np.testing.assert_allclose(data.E, expected_E)
    np.testing.assert_allclose(data.i, expected_i)


def test_cv_data_can_estimate_cdl_from_start_potential_region(ecat_module, cv_factory):
    scan_rate = 0.1
    cdl = 20e-6
    forward = np.linspace(0.0, -0.2, 9)
    reverse = np.linspace(-0.2, 0.0, 9)[1:]
    potential = np.r_[forward, reverse]
    current = np.r_[
        np.full(len(forward), -cdl * scan_rate),
        np.full(len(reverse), cdl * scan_rate),
    ]
    cv_obj = cv_factory(
        name="100mVs_capacitance_test",
        potential=potential,
        current=current,
    )

    data = ecat_module.simulation.cv_data(
        cv_obj,
        {"stride": 1, "estimate Cdl": True, "Cdl window": 0.075},
    )

    assert data.metadata["estimated_cdl"] == pytest.approx(cdl)
    assert data.metadata["estimated_Cdl"] == pytest.approx(cdl)
    diagnostics = data.metadata["estimated_cdl_diagnostics"]
    assert diagnostics["method"] == "median"
    assert diagnostics["n_pairs"] >= 2
    assert diagnostics["scan_rate"] == pytest.approx(scan_rate)
    assert diagnostics["potential_window"][1] == pytest.approx(0.0)


def test_cv_data_converts_measured_current_to_amps_before_auto_cdl(ecat_module, cv_factory):
    scan_rate = 0.1
    cdl = 20e-6
    forward = np.linspace(0.0, -0.2, 9)
    reverse = np.linspace(-0.2, 0.0, 9)[1:]
    potential = np.r_[forward, reverse]
    current_amps = np.r_[
        np.full(len(forward), -cdl * scan_rate),
        np.full(len(reverse), cdl * scan_rate),
    ]
    cv_obj = cv_factory(
        name="100mVs_capacitance_milliamp_test",
        potential=potential,
        current=current_amps * 1000,
    )
    cv_obj.units["Current"] = "mA"

    data = ecat_module.simulation.cv_data(
        cv_obj,
        {"stride": 1, "estimate Cdl": "auto", "Cdl window": 0.075},
    )

    assert data.metadata["estimated_Cdl"] == pytest.approx(cdl)
    assert data.metadata["current_unit"] == "A"
    np.testing.assert_allclose(data.i, current_amps)


def test_cv_data_estimates_cdl_by_default_before_window_trimming(ecat_module, cv_factory):
    scan_rate = 0.1
    cdl = 15e-6
    forward = np.linspace(0.0, -0.2, 9)
    reverse = np.linspace(-0.2, 0.0, 9)[1:]
    potential = np.r_[forward, reverse]
    current = np.r_[
        np.full(len(forward), -cdl * scan_rate),
        np.full(len(reverse), cdl * scan_rate),
    ]
    cv_obj = cv_factory(
        name="100mVs_default_capacitance_test",
        potential=potential,
        current=current,
    )

    data = ecat_module.simulation.cv_data(
        cv_obj,
        {"stride": 1, "potential window": [-0.16, -0.04], "trim mode": "pointwise"},
    )

    assert data.metadata["estimated_cdl"] == pytest.approx(cdl)
    assert data.metadata["estimated_Cdl"] == pytest.approx(cdl)
    assert data.metadata["potential_window"] == [-0.16, -0.04]
    assert np.nanmax(data.E) <= -0.04 + 1e-12


def test_cv_data_can_disable_default_cdl_estimation(ecat_module, cv_factory):
    cv_obj = cv_factory()

    data = ecat_module.simulation.cv_data(cv_obj, {"estimate Cdl": False})

    assert "estimated_cdl" not in data.metadata
    assert "estimated_cdl_error" not in data.metadata


def test_simulate_cv_fills_cell_defaults_and_expands_single_diffusion(ecat_module, monkeypatch):
    sim = ecat_module.simulation

    class Source:
        temperature = 305.0
        electrode_area = 0.071
        ir_uncomp_resistance = 42.0

    class Backend:
        name = "source-default-backend"

        def simulate(self, input_obj, mechanism_spec, params, options):
            self.params = params
            return input_obj.E, np.zeros_like(input_obj.E), input_obj.t, self

    backend = Backend()
    monkeypatch.setattr(sim, "get_backend", lambda name="electrokitty": backend)
    input_obj = sim.SimulatedCVInput(
        E=np.array([0.0, -0.1]),
        t=np.array([0.0, 1.0]),
        metadata={"scan_rate": 0.1},
        source=Source(),
    )
    params = {
        "cell": {"T": None, "Ru": None, "Cdl": 0, "A": 0},
        "spatial": {},
        "diffusion": {"D": 2e-9},
        "concentrations": {"bulk": {"FeII": 1e-3, "FeI": 0.0, "Fe0": 0.0}},
        "kinetics": [{"alpha": 0.5, "k0": 1e-3, "E0": -1.0}, {"alpha": 0.5, "k0": 1e-3, "E0": -1.2}],
        "isotherm": [],
    }

    result = sim.simulate_cv(input_obj, "EE", params, options={"plot": False})

    assert result.params["cell"]["T"] == pytest.approx(305.0)
    assert result.params["cell"]["A"] == pytest.approx(0.071e-4)
    assert result.params["cell"]["Ru"] == pytest.approx(42.0)
    assert result.params["diffusion"] == {"FeII": 2e-9, "FeI": 2e-9, "Fe0": 2e-9}
    assert result.mechanism.mechanism == "E(1):FeII=FeI\nE(1):FeI=Fe0"


def test_simulate_cv_preserves_explicit_zero_ru(ecat_module, monkeypatch):
    sim = ecat_module.simulation

    class Source:
        ir_uncomp_resistance = 55.0

    class Backend:
        name = "ru-default-backend"

        def simulate(self, input_obj, mechanism_spec, params, options):
            self.params = params
            return input_obj.E, np.zeros_like(input_obj.E), input_obj.t, self

    backend = Backend()
    monkeypatch.setattr(sim, "get_backend", lambda name="electrokitty": backend)
    input_obj = sim.SimulatedCVInput(
        E=np.array([0.0, -0.1]),
        t=np.array([0.0, 1.0]),
        metadata={"scan_rate": 0.1},
        source=Source(),
    )
    params = _basic_params()
    params["cell"]["Ru"] = 0.0

    result = sim.simulate_cv(input_obj, "E", params, options={"plot": False})

    assert result.params["cell"]["Ru"] == pytest.approx(0.0)
    assert backend.params["cell"]["Ru"] == pytest.approx(0.0)


def test_simulate_cv_uses_source_ru_when_requested_with_auto(ecat_module, monkeypatch):
    sim = ecat_module.simulation

    class Source:
        ir_uncomp_resistance = 55.0

    class Backend:
        name = "ru-auto-backend"

        def simulate(self, input_obj, mechanism_spec, params, options):
            self.params = params
            return input_obj.E, np.zeros_like(input_obj.E), input_obj.t, self

    backend = Backend()
    monkeypatch.setattr(sim, "get_backend", lambda name="electrokitty": backend)
    input_obj = sim.SimulatedCVInput(
        E=np.array([0.0, -0.1]),
        t=np.array([0.0, 1.0]),
        metadata={"scan_rate": 0.1},
        source=Source(),
    )
    params = _basic_params()
    params["cell"]["Ru"] = "auto"

    result = sim.simulate_cv(input_obj, "E", params, options={"plot": False})

    assert result.params["cell"]["Ru"] == pytest.approx(55.0)
    assert backend.params["cell"]["Ru"] == pytest.approx(55.0)


def test_simulate_cv_preserves_nondefault_user_ru(ecat_module, monkeypatch):
    sim = ecat_module.simulation

    class Source:
        ir_uncomp_resistance = 55.0

    class Backend:
        name = "ru-user-backend"

        def simulate(self, input_obj, mechanism_spec, params, options):
            return input_obj.E, np.zeros_like(input_obj.E), input_obj.t, self

    monkeypatch.setattr(sim, "get_backend", lambda name="electrokitty": Backend())
    input_obj = sim.SimulatedCVInput(
        E=np.array([0.0, -0.1]),
        t=np.array([0.0, 1.0]),
        metadata={"scan_rate": 0.1},
        source=Source(),
    )
    params = _basic_params()
    params["cell"]["Ru"] = 12.0

    result = sim.simulate_cv(input_obj, "E", params, options={"plot": False})

    assert result.params["cell"]["Ru"] == pytest.approx(12.0)


def test_simulate_cv_uses_shared_kinetic_fallbacks_and_k0_strings(ecat_module, monkeypatch):
    sim = ecat_module.simulation

    class Backend:
        name = "kinetic-fallback-backend"

        def simulate(self, input_obj, mechanism_spec, params, options):
            self.params = params
            return input_obj.E, np.zeros_like(input_obj.E), input_obj.t, self

    backend = Backend()
    monkeypatch.setattr(sim, "get_backend", lambda name="electrokitty": backend)
    input_obj = sim.cv_program(0.0, E_low=-0.2, points_per_segment=4)
    params = _basic_params()
    params["kinetics"] = [
        {"E0": -0.9, "k0": "fast"},
        {"E0": -1.1},
    ]

    result = sim.simulate_cv(input_obj, "EE", params, options={"plot": False})

    assert result.params["kinetics"] == [
        {"E0": -0.9, "k0": pytest.approx(1e-3), "alpha": pytest.approx(0.5)},
        {"E0": -1.1, "k0": pytest.approx(1e-3), "alpha": pytest.approx(0.5)},
    ]
    assert backend.params["kinetics"] == result.params["kinetics"]
    assert result.summary["parameter_fallbacks"]
    assert result.summary["parameter fallbacks"] == result.summary["parameter_fallbacks"]
    assert result.data.attrs["parameter fallbacks"] == result.summary["parameter_fallbacks"]
    assert any("kinetics.0.k0" in note and "fast" in note for note in result.summary["parameter_fallbacks"])
    assert any("kinetics.1.alpha" in note for note in result.summary["parameter_fallbacks"])
    assert any("kinetics.1.k0" in note for note in result.summary["parameter_fallbacks"])


def test_simulate_cv_check_params_prints_diagnostic_checks(ecat_module, monkeypatch, capsys):
    sim = ecat_module.simulation

    class Backend:
        name = "check-params-backend"

        def simulate(self, input_obj, mechanism_spec, params, options):
            return input_obj.E, np.zeros_like(input_obj.E), input_obj.t, self

    monkeypatch.setattr(sim, "get_backend", lambda name="electrokitty": Backend())
    input_obj = sim.cv_program(0.0, E_low=-0.2, points_per_segment=4)
    params = _basic_params()
    params["concentrations"] = {
        "bulk": {"CatOx": 1.0, "CatRed": 0.0, "Substrate": 2.0, "Product": 0.0}
    }
    params["diffusion"] = {"CatOx": 1e-9, "extra": 2e-9}

    sim.simulate_cv(input_obj, "Ecat", params, options={"plot": False, "check params": True})

    out = capsys.readouterr().out
    assert "Simulation Parameter Checks:" in out
    assert "WARN" in out
    assert "diffusion.CatRed" in out
    assert "diffusion.extra" in out
    assert "INFO" in out
    assert "CatOx, CatRed, Substrate, Product" in out


def test_simulated_cv_show_print_checks_uses_existing_result_params(ecat_module, monkeypatch, capsys):
    sim = ecat_module.simulation

    class Backend:
        name = "show-checks-backend"

        def simulate(self, input_obj, mechanism_spec, params, options):
            return input_obj.E, np.zeros_like(input_obj.E), input_obj.t, self

    monkeypatch.setattr(sim, "get_backend", lambda name="electrokitty": Backend())
    input_obj = sim.cv_program(0.0, E_low=-0.2, points_per_segment=4)
    params = _basic_params()
    params["diffusion"] = {"a": 1e-9, "unused": 2e-9}
    result = sim.simulate_cv(input_obj, "E", params, options={"plot": False})

    result.show({"print setup": False, "print checks": True})

    out = capsys.readouterr().out
    assert "Simulation Parameter Checks:" in out
    assert "diffusion.unused" in out
    assert "a, b" in out


def test_fit_cv_suppresses_check_params_during_optimizer_simulations(ecat_module, monkeypatch, capsys):
    sim = ecat_module.simulation

    class Backend:
        name = "quiet-check-backend"

        def simulate(self, input_obj, mechanism_spec, params, options):
            return input_obj.E, np.zeros_like(input_obj.E), input_obj.t, self

    monkeypatch.setattr(sim, "get_backend", lambda name="electrokitty": Backend())
    input_obj = sim.SimulatedCVInput(
        E=np.array([0.0, 0.1, 0.2]),
        t=np.array([0.0, 1.0, 2.0]),
        i=np.zeros(3),
        metadata={"kind": "manual", "scan_rate": 0.1},
        source="synthetic",
    )
    params = _basic_params()
    params["diffusion"] = {"a": 1e-9, "unused": 2e-9}

    sim.fit_cv(input_obj, "E", params, fit=[], options={"plot": False, "check params": True})

    out = capsys.readouterr().out
    assert "Simulation Parameter Checks:" not in out


def test_simulate_cv_rejects_unknown_k0_string(ecat_module, monkeypatch):
    sim = ecat_module.simulation

    class Backend:
        name = "unused-backend"

        def simulate(self, input_obj, mechanism_spec, params, options):
            raise AssertionError("backend should not be reached")

    monkeypatch.setattr(sim, "get_backend", lambda name="electrokitty": Backend())
    input_obj = sim.cv_program(0.0, E_low=-0.2, points_per_segment=4)
    params = _basic_params()
    params["kinetics"] = [{"E0": -0.9, "k0": "instant"}]

    with pytest.raises(ValueError, match="Unknown k0 preset"):
        sim.simulate_cv(input_obj, "E", params, options={"plot": False})


def test_simulate_cv_uses_auto_cdl_from_cv_data_metadata(ecat_module, monkeypatch):
    sim = ecat_module.simulation

    class Backend:
        name = "auto-cdl-backend"

        def simulate(self, input_obj, mechanism_spec, params, options):
            self.params = params
            return input_obj.E, np.zeros_like(input_obj.E), input_obj.t, self

    backend = Backend()
    monkeypatch.setattr(sim, "get_backend", lambda name="electrokitty": backend)
    input_obj = sim.SimulatedCVInput(
        E=np.array([0.0, -0.1]),
        t=np.array([0.0, 1.0]),
        metadata={"estimated_cdl": 2.5e-6},
        source="synthetic",
    )
    params = _basic_params()
    params["cell"]["Cdl"] = "auto"

    result = sim.simulate_cv(input_obj, "E", params, options={"plot": False})

    assert result.params["cell"]["Cdl"] == pytest.approx(2.5e-6)

    missing = sim.SimulatedCVInput(
        E=np.array([0.0, -0.1]),
        t=np.array([0.0, 1.0]),
        metadata={},
        source="synthetic",
    )
    with pytest.raises(ValueError, match="estimate Cdl"):
        sim.simulate_cv(missing, "E", params, options={"plot": False})


def test_simulate_cv_can_lazily_estimate_auto_cdl_from_measured_input(ecat_module, cv_factory, monkeypatch):
    sim = ecat_module.simulation

    class Backend:
        name = "lazy-cdl-backend"

        def simulate(self, input_obj, mechanism_spec, params, options):
            self.params = params
            return input_obj.E, np.zeros_like(input_obj.E), input_obj.t, self

    backend = Backend()
    monkeypatch.setattr(sim, "get_backend", lambda name="electrokitty": backend)
    scan_rate = 0.1
    cdl = 12e-6
    forward = np.linspace(0.0, -0.2, 9)
    reverse = np.linspace(-0.2, 0.0, 9)[1:]
    potential = np.r_[forward, reverse]
    current = np.r_[
        np.full(len(forward), -cdl * scan_rate),
        np.full(len(reverse), cdl * scan_rate),
    ]
    cv_obj = cv_factory(
        name="100mVs_sample_CO2_MeCN_10mM_Fc_run01",
        potential=potential,
        current=current,
    )
    input_obj = sim.cv_data(cv_obj, {"stride": 1})
    params = _basic_params()
    params["cell"]["Cdl"] = "auto"

    result = sim.simulate_cv(input_obj, "E", params, options={"plot": False, "Cdl window": 0.075})

    assert result.params["cell"]["Cdl"] == pytest.approx(cdl)
    assert input_obj.metadata["estimated_cdl"] == pytest.approx(cdl)


def test_print_params_defaults_to_pretty_and_raw_mode_is_available(ecat_module, capsys):
    sim = ecat_module.simulation
    params = {
        "cell": {"T": 298.15, "Ru": 0, "Cdl": 2.5e-5, "A": 1e-5},
        "spatial": {"nx": 20},
        "diffusion": {"FeII": 1e-9, "FeI": 1e-9},
        "concentrations": {"bulk": {"FeII": 1.0, "FeI": 0.0}},
        "kinetics": [{"alpha": 0.5, "k0": 1e3, "E0": -0.9}],
        "reactions": [{"kf": 1.0, "kb": 0.0}],
    }

    sim._maybe_print_simulation_params(params, {"print params": True})

    out = capsys.readouterr().out
    assert "Simulation Params:" in out
    assert "[parameters]" in out
    assert "Group" in out
    assert "Symbol" not in out
    assert "cell" in out
    assert "spatial" in out
    assert "diffusion.FeII" in out
    assert "Rᵤ" in out
    assert "Ω" in out
    assert "D(FeII)" in out
    assert "m²/s" in out
    assert "Cdl" in out
    assert "F" in out
    assert "[species]" in out
    assert "[FeII]" in out
    assert "mol/m³" in out
    assert "[mechanism]" in out
    assert "α" in out
    assert "k⁰" in out
    assert "m/s" in out
    assert "E⁰" in out
    assert "V" in out
    assert "k₁" in out
    assert "k₋₁" in out
    assert "s⁻¹" in out
    assert "kᶠ" not in out
    assert "kᵇ" not in out
    assert "k_f" not in out
    assert "k_b" not in out
    assert "{'cell'" not in out
    frames = sim._simulation_param_dataframes(params)
    assert all(isinstance(frame, pd.DataFrame) for _, frame in frames)

    sim._maybe_print_simulation_params(params, {"print params": "raw"})
    raw = capsys.readouterr().out
    assert "{'cell'" in raw


def test_print_params_compact_combines_species_and_mechanism_tables(ecat_module, capsys):
    sim = ecat_module.simulation
    params = {
        "concentrations": {"bulk": {"a": 1.0, "b": 0.0}, "surface": {"cat*": 1e-9}},
        "diffusion": {"a": 1e-9, "b": 2e-9},
        "kinetics": [{"alpha": 0.5, "k0": 1e-3, "E0": -1.4}],
        "reactions": [{"kf": 1.0, "kb": 0.0}],
    }

    sim._maybe_print_simulation_params(params, {"print params": "compact"})

    out = capsys.readouterr().out
    assert "Simulation Params:" in out
    assert "[species]" in out
    assert "Phase" in out
    assert "Species" in out
    assert "Amount" in out
    assert "Diffusion" in out
    assert "1e-09 mol/m²" in out
    assert "2e-09 m²/s" in out
    assert "[mechanism]" in out
    assert "Step" in out
    assert "α" in out
    assert "k⁰" in out
    assert "E⁰" in out
    assert "k₁" in out
    assert "k₋₁" in out


def test_mechanism_presets_compile_expected_strings(ecat_module):
    sim = ecat_module.simulation

    assert sim.compile_mechanism("E").mechanism == "E(1):a=b"
    assert sim.compile_mechanism("EE").mechanism == "E(1):a=b\nE(1):b=c"
    assert sim.compile_mechanism("E,E").mechanism == "E(1):a=b\nE(1):b=c"
    assert sim.compile_mechanism("EC").mechanism == "E(1):a=b\nC:b=c"
    assert sim.compile_mechanism("ECE").mechanism == "E(1):a=b\nC:b=c\nE(1):c=d"

    with pytest.warns(UserWarning, match="inferred catalysis form"):
        homogeneous = sim.compile_mechanism(
            "EC'",
            {"concentrations": {"bulk": {"CatOx": 1, "CatRed": 0, "Substrate": 10, "Product": 0}}},
        )
    assert homogeneous.mechanism == "E(1):CatOx=CatRed\nC:CatRed+Substrate>CatOx+Product"
    assert "inferred" in homogeneous.note

    with pytest.warns(UserWarning, match="inferred catalysis form"):
        surface = sim.compile_mechanism(
            "Ecat",
            {"concentrations": {"surface": {"CatOx*": 1e-5, "CatRed*": 0}}},
        )
    assert surface.mechanism == "E(1):CatOx*=CatRed*\nC:CatRed*=CatOx*"
    assert "inferred" in surface.note

    raw = sim.compile_mechanism("E(1):Fe2=Fe1\nC:Fe1>Fe0")
    assert raw.mechanism == "E(1):Fe2=Fe1\nC:Fe1>Fe0"
    assert raw.preset == "raw"


def test_concentration_normalization_defaults_plain_mapping_to_bulk(ecat_module):
    normalized = ecat_module.simulation.normalize_concentrations({"a": 1, "b": 0})

    assert normalized["bulk"] == {"a": 1.0, "b": 0.0}
    assert normalized["surface"] == {}


def test_surface_species_keys_with_or_without_star_are_equivalent(ecat_module):
    sim = ecat_module.simulation
    params_without_star = {"concentrations": {"surface": {"a": 1e-5, "b": 0}}}
    params_with_star = {"concentrations": {"surface": {"a*": 1e-5, "b*": 0}}}

    assert sim.normalize_concentrations(params_without_star["concentrations"]) == sim.normalize_concentrations(
        params_with_star["concentrations"]
    )
    assert sim.compile_mechanism("E", params_without_star).mechanism == "E(1):a*=b*"
    assert sim.compile_mechanism("E", params_with_star).mechanism == "E(1):a*=b*"


def test_star_shorthand_forces_surface_presets(ecat_module):
    sim = ecat_module.simulation

    assert sim.compile_mechanism("E*").mechanism == "E(1):a*=b*"
    assert sim.compile_mechanism("EE*").mechanism == "E(1):a*=b*\nE(1):b*=c*"
    assert sim.compile_mechanism("EC*").mechanism == "E(1):a*=b*\nC:b*=c*"
    assert sim.compile_mechanism("ECE*").mechanism == "E(1):a*=b*\nC:b*=c*\nE(1):c*=d*"
    with pytest.warns(UserWarning, match="inferred catalysis form"):
        assert sim.compile_mechanism("EC'*").mechanism == "E(1):CatOx*=CatRed*\nC:CatRed*=CatOx*"
    with pytest.warns(UserWarning, match="inferred catalysis form"):
        assert sim.compile_mechanism("Ecat*").mechanism == "E(1):CatOx*=CatRed*\nC:CatRed*=CatOx*"


def test_mixed_star_shorthand_is_not_supported(ecat_module):
    with pytest.raises(ValueError, match="Mixed surface/bulk shorthand"):
        ecat_module.simulation.compile_mechanism("E*,E")


def test_raw_mechanism_passes_through_but_species_populates_adapter(ecat_module):
    raw = "E(1):Fe2=Fe1\nC:Fe1>Fe0"
    spec = ecat_module.simulation.compile_mechanism(
        raw,
        {"concentrations": {"Fe2": 1, "Fe1": 0, "Fe0": 0}},
    )
    adapter = ecat_module.simulation._electrokitty_parameters(
        {"concentrations": {"Fe2": 1, "Fe1": 0, "Fe0": 0}}
    )

    assert spec.mechanism == raw
    assert adapter["species_information"] == [[], [1.0, 0.0, 0.0]]


def test_species_input_sugar_normalizes_to_concentrations_and_diffusion(ecat_module):
    sim = ecat_module.simulation
    program = sim.cv_program(0.1, -0.1, points_per_segment=3)
    params = {
        **_basic_params(),
        "concentrations": {"bulk": {"a": 2.0}},
        "diffusion": {"a": 2e-9},
        "species": {
            "a": {"type": "bulk", "C": 1.0, "D": 1e-9},
            "b": {"type": "solution", "concentration": 0.0, "diffusion": 3e-9},
            "cat": {"type": "adsorbed", "C": 1e-9},
        },
    }

    prepared = sim._prepare_simulation_params(program, params, {})

    assert "species" not in prepared
    assert prepared["concentrations"]["bulk"]["a"] == pytest.approx(2.0)
    assert prepared["concentrations"]["bulk"]["b"] == pytest.approx(0.0)
    assert prepared["concentrations"]["surface"]["cat"] == pytest.approx(1e-9)
    assert prepared["diffusion"]["a"] == pytest.approx(2e-9)
    assert prepared["diffusion"]["b"] == pytest.approx(3e-9)


def test_spatial_preset_string_resolves_for_simulation_params(ecat_module):
    sim = ecat_module.simulation
    program = sim.cv_program(0.1, -0.1, points_per_segment=3)
    params = {**_basic_params(), "spatial": "fast"}

    prepared = sim._prepare_simulation_params(program, params, {})

    assert prepared["spatial"] == {
        "dx_fraction": pytest.approx(0.005),
        "nx": 8,
        "viscosity": pytest.approx(1e-6),
        "rotation": pytest.approx(0.0),
    }


def test_spatial_preset_with_solvent_alias_sets_viscosity(ecat_module):
    sim = ecat_module.simulation
    program = sim.cv_program(0.1, -0.1, points_per_segment=3)
    params = {
        **_basic_params(),
        "spatial": {"preset": "fast", "solvent": "MeCN"},
    }

    prepared = sim._prepare_simulation_params(program, params, {})

    assert prepared["spatial"]["dx_fraction"] == pytest.approx(0.005)
    assert prepared["spatial"]["nx"] == 8
    assert prepared["spatial"]["viscosity"] == pytest.approx(sim.SOLVENT_VISCOSITIES["MeCN"])


def test_source_cv_solvent_sets_spatial_viscosity_by_default(ecat_module):
    sim = ecat_module.simulation
    program = sim.cv_program(0.1, -0.1, points_per_segment=3)
    program.source = types.SimpleNamespace(solvent="MeCN")
    params = {**_basic_params(), "spatial": "fast"}

    prepared = sim._prepare_simulation_params(program, params, {})

    assert prepared["spatial"]["viscosity"] == pytest.approx(sim.SOLVENT_VISCOSITIES["MeCN"])


def test_cell_auto_string_expands_to_auto_cdl_and_source_defaults(ecat_module):
    sim = ecat_module.simulation
    program = sim.SimulatedCVInput(
        E=np.array([0.0, -0.1]),
        t=np.array([0.0, 1.0]),
        metadata={"estimated_cdl": 2.5e-6},
        source=types.SimpleNamespace(electrode_area=0.071, temperature=295.0, Ru=42.0),
    )
    params = {**_basic_params(), "cell": "auto"}

    prepared = sim._prepare_simulation_params(program, params, {})

    assert prepared["cell"]["Cdl"] == pytest.approx(2.5e-6)
    assert prepared["cell"]["A"] == pytest.approx(0.071e-4)
    assert prepared["cell"]["T"] == pytest.approx(295.0)
    assert prepared["cell"]["Ru"] == pytest.approx(42.0)


def test_unknown_cell_string_raises_helpful_error(ecat_module):
    sim = ecat_module.simulation
    program = sim.cv_program(0.1, -0.1, points_per_segment=3)

    with pytest.raises(ValueError, match="cell params must be a mapping"):
        sim._prepare_simulation_params(program, {**_basic_params(), "cell": "default"}, {})


def test_explicit_spatial_solvent_overrides_source_cv_solvent(ecat_module):
    sim = ecat_module.simulation
    program = sim.cv_program(0.1, -0.1, points_per_segment=3)
    program.source = types.SimpleNamespace(solvent="MeCN")
    params = {**_basic_params(), "spatial": {"preset": "fast", "solvent": "DMSO"}}

    prepared = sim._prepare_simulation_params(program, params, {})

    assert prepared["spatial"]["viscosity"] == pytest.approx(sim.SOLVENT_VISCOSITIES["DMSO"])


def test_spatial_numeric_viscosity_overrides_solvent_alias(ecat_module):
    sim = ecat_module.simulation
    program = sim.cv_program(0.1, -0.1, points_per_segment=3)
    params = {
        **_basic_params(),
        "spatial": {"preset": "fast", "solvent": "MeCN", "viscosity": 2e-6},
    }

    prepared = sim._prepare_simulation_params(program, params, {})

    assert prepared["spatial"]["viscosity"] == pytest.approx(2e-6)


def test_unknown_spatial_aliases_raise_helpful_errors(ecat_module):
    sim = ecat_module.simulation
    program = sim.cv_program(0.1, -0.1, points_per_segment=3)

    with pytest.raises(ValueError, match="Unknown spatial preset"):
        sim._prepare_simulation_params(program, {**_basic_params(), "spatial": "very-fancy"}, {})
    with pytest.raises(ValueError, match="Unknown solvent viscosity alias"):
        sim._prepare_simulation_params(
            program,
            {**_basic_params(), "spatial": {"preset": "fast", "solvent": "mystery-solvent"}},
            {},
        )


def test_backend_dispatch_default_and_unknown_errors(ecat_module):
    assert ecat_module.simulation.get_backend("electrokitty").name == "electrokitty"
    with pytest.raises(ValueError, match="Unknown simulation backend"):
        ecat_module.simulation.get_backend("not-a-backend")
    with pytest.raises(ValueError, match="Supported backend is 'electrokitty'"):
        ecat_module.simulation.get_backend("pyecsim")


def test_simulate_cv_missing_electrokitty_has_friendly_error(ecat_module, monkeypatch):
    monkeypatch.setitem(sys.modules, "electrokitty", None)
    program = ecat_module.simulation.cv_program(0.1, -0.1, points_per_segment=3)

    with pytest.raises(ImportError, match=r"pip install .*ecat\[simulation\]"):
        ecat_module.simulation.simulate_cv(program, "E", _basic_params(), options={"plot": False})


def test_simulate_cv_uses_fake_backend_and_auto_matches_current_sign(ecat_module, monkeypatch):
    fake_module = types.ModuleType("electrokitty")
    fake_module.ElectroKitty = _FakeElectroKitty
    monkeypatch.setitem(sys.modules, "electrokitty", fake_module)

    data = ecat_module.simulation.SimulatedCVInput(
        E=np.array([0.1, 0.0, -0.1]),
        t=np.array([0.0, 1.0, 2.0]),
        i=np.array([-1.0, -2.0, -3.0]),
        metadata={"kind": "cv_data"},
        source="synthetic",
    )

    result = ecat_module.simulation.simulate_cv(
        data,
        "E",
        _basic_params(),
        options={"plot": False},
    )

    assert isinstance(result, ecat_module.simulation.SimulatedCV)
    assert result.backend_result is not None
    assert result.summary["backend"] == "electrokitty"
    assert result.current_sign == -1
    assert result.summary["current_sign"] == -1
    assert result.summary["current sign"] == -1
    np.testing.assert_allclose(result.data["Current"], [-1.0, -2.0, -3.0])
    assert "Measured Current" not in result.data
    np.testing.assert_allclose(result.input.i, data.i)
    np.testing.assert_allclose(result.data["Backend Current"], [1.0, 2.0, 3.0])


def test_electrokitty_adapter_orders_reactions_by_mechanism_steps(ecat_module, monkeypatch):
    fake_module = types.ModuleType("electrokitty")
    fake_module.ElectroKitty = _FakeElectroKitty
    monkeypatch.setitem(sys.modules, "electrokitty", fake_module)
    program = ecat_module.simulation.cv_program(0.1, -0.1, points_per_segment=3)
    params = {
        **_basic_params(),
        "concentrations": {"a": 1.0, "b": 0.0, "c": 0.0, "d": 0.0},
        "diffusion": {"a": 1e-9, "b": 1e-9, "c": 1e-9, "d": 1e-9},
        "kinetics": [
            {"alpha": 0.5, "k0": 1e2, "E0": -0.1},
            {"alpha": 0.4, "k0": 1e3, "E0": -0.2},
        ],
        "reactions": [{"kf": 7.0, "kb": 3.0}],
    }

    result = ecat_module.simulation.simulate_cv(
        program,
        "ECE",
        params,
        options={"plot": False},
    )

    assert result.backend_result.created["kin"] == [
        [0.5, 1e2, -0.1],
        [7.0, 3.0],
        [0.4, 1e3, -0.2],
    ]


def test_electrokitty_adapter_includes_raw_eecat_reactions(ecat_module, monkeypatch):
    fake_module = types.ModuleType("electrokitty")
    fake_module.ElectroKitty = _FakeElectroKitty
    monkeypatch.setitem(sys.modules, "electrokitty", fake_module)
    program = ecat_module.simulation.cv_program(0.1, -0.1, points_per_segment=3)
    params = {
        **_basic_params(),
        "concentrations": {"a": 1.0, "b": 0.0, "c": 0.0, "Substrate": 1.0, "Product": 0.0},
        "diffusion": {
            "a": 1e-9,
            "b": 1e-9,
            "c": 1e-9,
            "Substrate": 1e-9,
            "Product": 1e-9,
        },
        "kinetics": [
            {"alpha": 0.5, "k0": 1e2, "E0": -0.1},
            {"alpha": 0.4, "k0": 1e3, "E0": -0.2},
        ],
        "reactions": [{"kf": 11.0, "kb": 0.0}],
    }

    result = ecat_module.simulation.simulate_cv(
        program,
        "E(1):a=b\nE(1):b=c\nC:c+Substrate>a+Product",
        params,
        options={"plot": False},
    )

    assert result.backend_result.created["kin"] == [
        [0.5, 1e2, -0.1],
        [0.4, 1e3, -0.2],
        [11.0, 0.0],
    ]


@pytest.mark.parametrize(
    ("mechanism", "params", "expected_mechanism"),
    [
        (
            "EC",
            {
                "concentrations": {"a": 1.0, "b": 0.0, "c": 0.0},
                "diffusion": {"a": 1e-9, "b": 1e-9, "c": 1e-9},
            },
            "E(1):a=b\nC:b=c",
        ),
        (
            "EC'",
            {
                "concentrations": {"a": 1.0, "b": 0.0, "Substrate": 1.0, "Product": 0.0},
                "diffusion": {"a": 1e-9, "b": 1e-9, "Substrate": 1e-9, "Product": 1e-9},
            },
            "E(1):a=b\nC:b+Substrate>a+Product",
        ),
        (
            "EC*",
            {
                "concentrations": {"surface": {"a": 1.0, "b": 0.0, "c": 0.0}},
                "diffusion": {"a": 1e-9, "b": 1e-9, "c": 1e-9},
            },
            "E(1):a*=b*\nC:b*=c*",
        ),
    ],
)
def test_electrokitty_adapter_orders_reactions_for_ec_variants(
    ecat_module,
    monkeypatch,
    mechanism,
    params,
    expected_mechanism,
):
    fake_module = types.ModuleType("electrokitty")
    fake_module.ElectroKitty = _FakeElectroKitty
    monkeypatch.setitem(sys.modules, "electrokitty", fake_module)
    program = ecat_module.simulation.cv_program(0.1, -0.1, points_per_segment=3)

    result = ecat_module.simulation.simulate_cv(
        program,
        mechanism,
        {
            **_basic_params(),
            **params,
            "kinetics": [{"alpha": 0.5, "k0": 1e2, "E0": -0.1}],
            "reactions": [{"kf": 13.0, "kb": 2.0}],
        },
        options={"plot": False},
    )

    assert result.mechanism.mechanism == expected_mechanism
    assert result.backend_result.created["kin"] == [[0.5, 1e2, -0.1], [13.0, 2.0]]


def test_electrokitty_adapter_requires_reaction_for_chemical_step(ecat_module, monkeypatch):
    fake_module = types.ModuleType("electrokitty")
    fake_module.ElectroKitty = _FakeElectroKitty
    monkeypatch.setitem(sys.modules, "electrokitty", fake_module)
    program = ecat_module.simulation.cv_program(0.1, -0.1, points_per_segment=3)

    with pytest.raises(ValueError, match="chemical step"):
        ecat_module.simulation.simulate_cv(
            program,
            "EC",
            {
                **_basic_params(),
                    "concentrations": {"a": 1.0, "b": 0.0, "c": 0.0},
                "diffusion": {"a": 1e-9, "b": 1e-9, "c": 1e-9},
                "kinetics": [{"alpha": 0.5, "k0": 1e2, "E0": -0.1}],
                "reactions": [],
            },
            options={"plot": False},
        )


def test_simulation_result_plot_smoke(ecat_module, monkeypatch):
    fake_module = types.ModuleType("electrokitty")
    fake_module.ElectroKitty = _FakeElectroKitty
    monkeypatch.setitem(sys.modules, "electrokitty", fake_module)

    program = ecat_module.simulation.cv_program(0.1, -0.1, points_per_segment=3)
    result = ecat_module.simulation.simulate_cv(
        program,
        "E",
        _basic_params(),
        options={"plot": True},
    )

    assert result.axes is not None
    assert len(result.axes.lines) == 1
    assert result.axes.lines[0].get_linestyle() == "--"
    assert result.axes.lines[0].get_label() == "Simulation"
    assert result.axes.get_xlabel() == "Potential (V)"
    assert result.axes.get_ylabel() == "Current (mA)"


def test_simulation_result_plot_uses_ecat_axis_units(ecat_module):
    sim = ecat_module.simulation
    result = sim.SimulatedCV(
        data=pd.DataFrame(
            {
                "Potential": [0.0, -0.1],
                "Current": [0.0, 2e-6],
                "Backend Current": [0.0, 3e-6],
            }
        ),
        params={},
        mechanism=None,
        input=sim.SimulatedCVInput(
            E=np.array([0.0, -0.1]),
            i=np.array([0.0, 1e-6]),
            t=np.array([0.0, 1.0]),
            metadata={
                "potential_axis_name": "Potential vs Fc/Fc+",
                "potential_unit": "V",
                "current_axis_name": "Current",
                "current_unit": "A",
            },
            source="synthetic",
        ),
        backend_result=None,
    )

    ax = result.plot({"legend": False, "plot all": True})

    assert ax.get_xlabel() == r"Potential (V vs $\mathrm{Fc/Fc^{+}}$)"
    assert ax.get_ylabel() == "Current (μA)"
    np.testing.assert_allclose(ax.lines[0].get_ydata(), [0.0, 2.0])
    np.testing.assert_allclose(ax.lines[1].get_ydata(), [0.0, 3.0])
    plt.close(ax.figure)


def test_simulation_result_plot_does_not_overlay_input_current(ecat_module):
    sim = ecat_module.simulation
    result = sim.SimulatedCV(
        data=pd.DataFrame(
            {
                "Potential": [0.0, -0.1],
                "Current": [0.0, 2e-6],
                "Backend Current": [0.0, 2e-6],
            }
        ),
        params={},
        mechanism=None,
        input=sim.SimulatedCVInput(
            E=np.array([0.0, -0.1]),
            i=np.array([0.0, 1e-6]),
            t=np.array([0.0, 1.0]),
            metadata={"current_unit": "A"},
            source="synthetic",
        ),
        backend_result=None,
    )

    ax = result.plot({"legend": False})

    assert len(ax.lines) == 1
    assert ax.lines[0].get_label() == "Simulation"
    plt.close(ax.figure)


def test_simulation_plot_uses_ecat_axis_style_helper(ecat_module, monkeypatch):
    sim = ecat_module.simulation
    calls = []

    def fake_axis_style(ax, options=None):
        calls.append((ax, options))
        return ax

    monkeypatch.setattr(sim, "_apply_ecat_axis_style", fake_axis_style)
    result = sim.SimulatedCV(
        data=pd.DataFrame(
            {
                "Potential": [0.0, -0.1],
                "Current": [0.0, 1e-6],
                "Backend Current": [0.0, 1e-6],
            }
        ),
        params={},
        mechanism=None,
        input=None,
        backend_result=None,
    )

    ax = result.plot({"legend": False})

    assert calls == [(ax, {"legend": False})]


def test_simulation_plot_forces_autoscale_and_preserves_inverted_x(ecat_module):
    sim = ecat_module.simulation
    _, ax = plt.subplots()
    ax.set_xlim(10, 9)
    ax.set_ylim(10, 11)

    result = sim.SimulatedCV(
        data=pd.DataFrame(
            {
                "Potential": [0.0, -2.0],
                "Current": [-1e-6, 3e-6],
                "Backend Current": [-1e-6, 3e-6],
            }
        ),
        params={},
        mechanism=None,
        input=None,
        backend_result=None,
    )

    result.plot({"ax": ax, "legend": False})

    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    assert x0 > x1
    assert x0 >= 0.0
    assert x1 <= -2.0
    assert y0 <= -1e-6
    assert y1 >= 3e-6


def test_fit_spec_compact_aliases_auto_init_bounds_and_transform(ecat_module):
    sim = ecat_module.simulation
    program = sim.cv_program(0.1, -0.1, points_per_segment=3)
    spec = sim._normalize_fit_spec(
        ["E0_0", "k0_0"],
        _basic_params(),
        program,
    )

    assert spec["vary"] == [("kinetics", 0, "E0"), ("kinetics", 0, "k0")]
    assert spec["entries"][("kinetics", 0, "E0")]["init"] == pytest.approx(-0.1)
    assert spec["entries"][("kinetics", 0, "E0")]["bounds"] == pytest.approx((-0.1, 0.1))
    assert spec["entries"][("kinetics", 0, "E0")]["transform"] == "linear"
    assert spec["entries"][("kinetics", 0, "k0")]["init"] == pytest.approx(1e-3)
    assert spec["entries"][("kinetics", 0, "k0")]["bounds"] == pytest.approx((1e-30, 1e12))
    assert spec["entries"][("kinetics", 0, "k0")]["transform"] == "log10"


def test_fit_spec_wildcard_and_bare_kinetic_aliases(ecat_module):
    sim = ecat_module.simulation
    program = sim.cv_program(0.1, -0.1, points_per_segment=3)
    params = _basic_params()
    params["kinetics"] = [
        {"alpha": 0.5, "k0": 1e-3, "E0": -1.1},
        {"alpha": 0.4, "k0": 2e-3, "E0": -1.3},
    ]

    spec = sim._normalize_fit_spec(
        {
            "vary": ["E0_*"],
            "fixed": {"alpha_*": 0.5},
            "bounds": {"E0_*": [-1.8, -0.6]},
        },
        params,
        program,
    )

    assert spec["vary"] == [("kinetics", 0, "E0"), ("kinetics", 1, "E0")]
    assert spec["base_params"]["kinetics"][0]["alpha"] == pytest.approx(0.5)
    assert spec["base_params"]["kinetics"][1]["alpha"] == pytest.approx(0.5)
    assert spec["entries"][("kinetics", 0, "E0")]["bounds"] == pytest.approx((-1.8, -0.6))
    assert spec["entries"][("kinetics", 1, "E0")]["bounds"] == pytest.approx((-1.8, -0.6))

    single = deepcopy(params)
    single["kinetics"] = [single["kinetics"][0]]
    assert sim._normalize_fit_spec(["E0"], single, program)["vary"] == [("kinetics", 0, "E0")]
    with pytest.raises(ValueError, match="ambiguous"):
        sim._normalize_fit_spec(["E0"], params, program)


def test_fit_spec_accepts_friendly_kinetic_alias_spellings(ecat_module):
    sim = ecat_module.simulation
    program = sim.cv_program(0.1, -0.1, points_per_segment=3)
    params = _basic_params()
    params["kinetics"] = [
        {"alpha": 0.5, "k0": 1e-3, "E0": -0.2},
        {"alpha": 0.5, "k0": 1e-3, "E0": -0.4},
    ]

    spec = sim._normalize_fit_spec(
        {
            "vary": ["E 0", "E_1", "k0 0"],
            "fixed": {"α 1": 0.45},
            "bounds": {"E *": [-1.0, 0.0], "k0 0": [1e-6, 1e-1]},
        },
        params,
        program,
    )

    assert spec["vary"] == [
        ("kinetics", 0, "E0"),
        ("kinetics", 1, "E0"),
        ("kinetics", 0, "k0"),
    ]
    assert spec["fixed"] == {("kinetics", 1, "alpha"): 0.45}
    assert spec["entries"][("kinetics", 0, "E0")]["bounds"] == (-1.0, 0.0)
    assert spec["entries"][("kinetics", 1, "E0")]["bounds"] == (-1.0, 0.0)
    assert spec["entries"][("kinetics", 0, "k0")]["bounds"] == (1e-6, 1e-1)


def test_fit_spec_diffusion_aliases(ecat_module):
    sim = ecat_module.simulation
    program = sim.cv_program(0.1, -0.1, points_per_segment=3)
    params = {
        **_basic_params(),
        "diffusion": {"FeII": 1e-9, "FeI": 2e-9, "Fe0": 3e-9},
    }

    spec = sim._normalize_fit_spec(
        {
            "vary": ["D_0", "D2"],
            "bounds": {"D_*": [1e-11, 1e-8]},
            "init": {"D_0": 5e-10},
        },
        params,
        program,
    )

    assert spec["vary"] == [("diffusion", "FeII"), ("diffusion", "Fe0")]
    assert spec["entries"][("diffusion", "FeII")]["init"] == pytest.approx(5e-10)
    assert spec["entries"][("diffusion", "FeII")]["bounds"] == pytest.approx((1e-11, 1e-8))
    assert spec["entries"][("diffusion", "Fe0")]["bounds"] == pytest.approx((1e-11, 1e-8))

    all_diffusion = sim._normalize_fit_spec(["D_*"], params, program)
    assert all_diffusion["vary"] == [
        ("diffusion", "FeII"),
        ("diffusion", "FeI"),
        ("diffusion", "Fe0"),
    ]

    shared = {**_basic_params(), "diffusion": {"D": 1e-9}}
    assert sim._normalize_fit_spec(["D"], shared, program)["vary"] == [("diffusion", "D")]
    with pytest.raises(ValueError, match="ambiguous"):
        sim._normalize_fit_spec(["D"], params, program)

    spaced = sim._normalize_fit_spec(["D *"], params, program)
    assert spaced["vary"] == [("diffusion", "FeII"), ("diffusion", "FeI"), ("diffusion", "Fe0")]


def test_fit_spec_bare_alias_ties_equal_expanded_values(ecat_module):
    sim = ecat_module.simulation
    program = sim.cv_program(0.1, -0.1, points_per_segment=3)
    params = {
        **_basic_params(),
        "diffusion": {"FeII": 1e-9, "FeI": 1e-9, "Fe0": 1e-9},
        "kinetics": [
            {"alpha": 0.5, "k0": 1e3, "E0": -0.9},
            {"alpha": 0.5, "k0": 1e3, "E0": -0.9},
        ],
    }

    spec = sim._normalize_fit_spec(
        {
            "vary": ["D", "E0", "k0", "alpha"],
            "bounds": {"D": [1e-10, 1e-8], "E0": [-1.2, -0.4]},
        },
        params,
        program,
    )

    assert spec["vary"][0] == (("diffusion", "FeII"), ("diffusion", "FeI"), ("diffusion", "Fe0"))
    assert spec["vary"][1] == (("kinetics", 0, "E0"), ("kinetics", 1, "E0"))
    assert spec["entries"][spec["vary"][0]]["bounds"] == pytest.approx((1e-10, 1e-8))
    assert spec["entries"][spec["vary"][1]]["bounds"] == pytest.approx((-1.2, -0.4))

    target_values = [2e-9, -1.0, 1e4, 0.4]
    optimizer_values = [
        sim._external_to_optimizer(value, spec["entries"][target]["transform"])
        for value, target in zip(target_values, spec["vary"])
    ]
    updated = sim._params_from_fit_vector(spec["base_params"], spec, optimizer_values)
    assert updated["diffusion"] == {"FeII": 2e-9, "FeI": 2e-9, "Fe0": 2e-9}
    assert [entry["E0"] for entry in updated["kinetics"]] == [-1.0, -1.0]
    assert [entry["k0"] for entry in updated["kinetics"]] == [1e4, 1e4]
    assert [entry["alpha"] for entry in updated["kinetics"]] == [0.4, 0.4]

    params["kinetics"][1]["E0"] = -1.1
    with pytest.raises(ValueError, match="same value"):
        sim._normalize_fit_spec(["E0"], params, program)


def test_fit_spec_fixed_overrides_params_and_tuple_paths(ecat_module):
    sim = ecat_module.simulation
    program = sim.cv_program(0.1, -0.1, points_per_segment=3)

    spec = sim._normalize_fit_spec(
        {
            "vary": [("kinetics", 0, "E0")],
            "fixed": {"alpha_0": 0.25},
            "bounds": {("kinetics", 0, "E0"): [-1.5, -1.2]},
        },
        _basic_params(),
        program,
    )

    assert spec["base_params"]["kinetics"][0]["alpha"] == pytest.approx(0.25)
    assert spec["entries"][("kinetics", 0, "E0")]["bounds"] == pytest.approx((-1.5, -1.2))

    with pytest.raises(ValueError, match="both fixed and varied"):
        sim._normalize_fit_spec(
            {"vary": ["alpha_0"], "fixed": {"alpha_0": 0.5}},
            _basic_params(),
            program,
        )


def test_fit_spec_omitted_uses_all_fit_safe_numeric_paths(ecat_module):
    sim = ecat_module.simulation
    program = sim.cv_program(0.1, -0.1, points_per_segment=3)
    params = {
        **_basic_params(),
        "cell": {"T": 298.15, "Ru": 0, "Cdl": 0, "A": 1e-5},
        "diffusion": {"a": 1e-9, "b": 1e-9},
        "concentrations": {"bulk": {"a": 1, "b": 0}},
        "reactions": [{"kf": 2.0, "kb": 0.0}],
    }

    spec = sim._normalize_fit_spec(None, params, program)

    assert ("kinetics", 0, "E0") in spec["vary"]
    assert ("reactions", 0, "kf") in spec["vary"]
    assert ("cell", "A") in spec["vary"]
    assert ("diffusion", "a") in spec["vary"]
    assert ("concentrations", "bulk", "a") in spec["vary"]
    assert not any(path[0] == "spatial" for path in spec["vary"])
    assert ("cell", "T") not in spec["vary"]


def test_fit_spec_accepts_concentrations_paths_and_rejects_species_paths(ecat_module):
    sim = ecat_module.simulation
    program = sim.cv_program(0.1, -0.1, points_per_segment=3)
    params = {
        **_basic_params(),
        "concentrations": {"bulk": {"a": 1.0, "b": 0.0}},
    }

    spec = sim._normalize_fit_spec(["concentrations.bulk.a"], params, program)

    assert spec["vary"] == [("concentrations", "bulk", "a")]
    with pytest.raises(ValueError, match="species"):
        sim._normalize_fit_spec(["species.bulk.a"], params, program)


def test_fit_spec_invalid_log_transform_bounds(ecat_module):
    sim = ecat_module.simulation
    program = sim.cv_program(0.1, -0.1, points_per_segment=3)

    with pytest.raises(ValueError, match="log10 transform requires positive"):
        sim._normalize_fit_spec(
            {
                "vary": ["k0_0"],
                "bounds": {"k0_0": [0, 1]},
                "transform": {"k0_0": "log10"},
            },
            _basic_params(),
            program,
        )


def test_fit_method_normalization_strings_and_structured_options(ecat_module):
    sim = ecat_module.simulation

    legacy = sim._normalize_fit_method("least_squares", {"max_nfev": 11})
    assert legacy["strategy"] == "least_squares"
    assert legacy["max_nfev"] == 11

    structured = sim._normalize_fit_method(
        {
            "strategy": "multistart",
            "optimizer": "least_squares",
            "starts": 7,
            "start_strategy": "latin_hypercube",
            "seed": 3,
            "max_nfev": 5,
            "polish_with": "least_squares",
            "polish_max_nfev": 13,
        },
        {"max_nfev": 99},
    )

    assert structured["strategy"] == "multistart"
    assert structured["optimizer"] == "least_squares"
    assert structured["starts"] == 7
    assert structured["start_strategy"] == "latin_hypercube"
    assert structured["seed"] == 3
    assert structured["max_nfev"] == 5
    assert structured["polish_with"] == "least_squares"
    assert structured["polish_max_nfev"] == 13
    assert sim._fit_strategy_progress_total(structured, n_parameters=3) == 192

    with pytest.raises(ValueError, match="requires an optimizer"):
        sim._normalize_fit_method({"strategy": "multistart"}, {})
    with pytest.raises(ValueError, match="optimizer is only valid"):
        sim._normalize_fit_method({"strategy": "least_squares", "optimizer": "least_squares"}, {})
    with pytest.raises(ValueError, match="optimizer is only valid"):
        sim._normalize_fit_method({"strategy": "cma_es", "optimizer": "least_squares"}, {})
    with pytest.raises(NotImplementedError, match="CMA-ES"):
        sim._normalize_fit_method({"strategy": "cma_es"}, {})


def test_fit_method_normalization_accepts_spaced_public_names(ecat_module):
    sim = ecat_module.simulation

    legacy = sim._normalize_fit_method("least squares", {"max nfev": 11})
    assert legacy["strategy"] == "least_squares"
    assert legacy["max_nfev"] == 11

    structured = sim._normalize_fit_method(
        {
            "strategy": "multi start",
            "optimizer": "least squares",
            "starts": 7,
            "start strategy": "latin hypercube",
            "seed": 3,
            "max nfev": 5,
            "polish with": "least squares",
            "polish max nfev": 13,
        },
        {"max nfev": 99},
    )

    assert structured["strategy"] == "multistart"
    assert structured["optimizer"] == "least_squares"
    assert structured["start_strategy"] == "latin_hypercube"
    assert structured["max_nfev"] == 5
    assert structured["polish_with"] == "least_squares"
    assert structured["polish_max_nfev"] == 13


def test_fit_start_generation_includes_nominal_and_is_deterministic(ecat_module):
    sim = ecat_module.simulation
    program = sim.cv_program(0.1, -0.1, points_per_segment=3)
    spec = sim._normalize_fit_spec(
        {
            "vary": ["E0_0", "D"],
            "bounds": {"E0_0": [-2.0, 1.0], "D": [1e-11, 1e-7]},
        },
        {**_basic_params(), "diffusion": {"D": 1e-9}},
        program,
    )
    x0, lower, upper = sim._fit_vectors(spec)
    method = sim._normalize_fit_method(
        {
            "strategy": "multistart",
            "optimizer": "least_squares",
            "starts": 6,
            "start_strategy": "latin_hypercube",
            "seed": 123,
        },
        {},
    )

    starts_a = sim._generate_fit_starts(method, x0, lower, upper)
    starts_b = sim._generate_fit_starts(method, x0, lower, upper)

    assert starts_a.shape == (6, 2)
    np.testing.assert_allclose(starts_a[0], x0)
    np.testing.assert_allclose(starts_a, starts_b)
    assert np.all(starts_a >= lower)
    assert np.all(starts_a <= upper)


def test_fit_cv_least_squares_direct_with_fake_backend(ecat_module, monkeypatch):
    sim = ecat_module.simulation
    input_obj = sim.SimulatedCVInput(
        E=np.array([-1.0, -0.5, 0.0]),
        t=np.array([0.0, 1.0, 2.0]),
        i=np.array([-1.5, -1.0, -0.5]),
        metadata={"kind": "cv_data", "scan_rate": 0.1},
        source="synthetic",
    )
    params = _basic_params()
    params["kinetics"][0]["E0"] = -0.2
    monkeypatch.setattr(sim, "get_backend", lambda name="electrokitty": _LinearE0Backend())

    result = sim.fit_cv(
        input_obj,
        "E",
        params,
        fit={"vary": ["E0_0"], "bounds": {"E0_0": [-2.0, 1.0]}, "fixed": {"alpha_0": 0.5}},
        options={"plot": False, "residual": "direct"},
    )

    assert isinstance(result, sim.SimulationFitResult)
    assert result.best_params["kinetics"][0]["E0"] == pytest.approx(-0.5, abs=1e-6)
    assert result.corrections == {}
    np.testing.assert_allclose(result.residuals, np.zeros(3), atol=1e-6)
    assert "Residual" in result.simulation_result.data
    assert result.summary["rmse"] == pytest.approx(0.0, abs=1e-6)
    assert result.summary["max_abs_residual"] == pytest.approx(0.0, abs=1e-6)
    assert result.summary["n_points"] == 3
    assert result.summary["degrees_of_freedom"] == 2


def test_fit_cv_multistart_can_escape_bad_nominal_basin(ecat_module, monkeypatch):
    sim = ecat_module.simulation
    input_obj = sim.SimulatedCVInput(
        E=np.array([0.0, 1.0]),
        t=np.array([0.0, 1.0]),
        i=np.array([0.0, 0.0]),
        metadata={"kind": "cv_data", "scan_rate": 0.1},
        source="synthetic",
    )
    params = _basic_params()
    params["kinetics"][0]["E0"] = 0.0
    monkeypatch.setattr(sim, "get_backend", lambda name="electrokitty": _BimodalE0Backend())

    nominal = sim.fit_cv(
        input_obj,
        "E",
        params,
        fit={"vary": ["E0_0"], "bounds": {"E0_0": [-2.0, 2.0]}},
        options={"plot": False, "residual": "direct", "max_nfev": 5},
    )
    multistart = sim.fit_cv(
        input_obj,
        "E",
        params,
        fit={"vary": ["E0_0"], "bounds": {"E0_0": [-2.0, 2.0]}},
        method={
            "strategy": "multistart",
            "optimizer": "least_squares",
            "starts": 8,
            "start_strategy": "latin_hypercube",
            "seed": 2,
            "max_nfev": 20,
            "polish_with": "least_squares",
            "polish_max_nfev": 20,
        },
        options={"plot": False, "residual": "direct"},
    )

    assert abs(nominal.best_params["kinetics"][0]["E0"]) < 0.1
    assert abs(abs(multistart.best_params["kinetics"][0]["E0"]) - 1.0) < 1e-4
    assert multistart.summary["strategy"] == "multistart"
    assert multistart.summary["optimizer"] == "least_squares"
    assert multistart.summary["starts"] == 8
    assert multistart.summary["start_strategy"] == "latin_hypercube"
    assert multistart.summary["best_start_index"] != 0
    assert multistart.summary["best_pre_polish_cost"] < nominal.summary["cost"]
    assert multistart.summary["final_cost"] <= multistart.summary["best_pre_polish_cost"] + 1e-12


def test_fit_cv_differential_evolution_smoke_with_polish(ecat_module, monkeypatch):
    sim = ecat_module.simulation
    input_obj = sim.SimulatedCVInput(
        E=np.array([0.0, 1.0]),
        t=np.array([0.0, 1.0]),
        i=np.array([0.0, 0.0]),
        metadata={"kind": "cv_data", "scan_rate": 0.1},
        source="synthetic",
    )
    params = _basic_params()
    params["kinetics"][0]["E0"] = 0.0
    monkeypatch.setattr(sim, "get_backend", lambda name="electrokitty": _BimodalE0Backend())

    result = sim.fit_cv(
        input_obj,
        "E",
        params,
        fit={"vary": ["E0_0"], "bounds": {"E0_0": [-2.0, 2.0]}},
        method={
            "strategy": "differential_evolution",
            "seed": 1,
            "maxiter": 3,
            "popsize": 5,
            "polish_with": "least_squares",
            "polish_max_nfev": 20,
        },
        options={"plot": False, "residual": "direct"},
    )

    assert result.summary["strategy"] == "differential_evolution"
    assert result.summary["polish_with"] == "least_squares"
    assert abs(abs(result.best_params["kinetics"][0]["E0"]) - 1.0) < 1e-4


def test_fit_cv_normalizes_optimizer_residuals_but_reports_amp_residuals(ecat_module, monkeypatch):
    sim = ecat_module.simulation
    input_obj = sim.SimulatedCVInput(
        E=np.array([-1.0, -0.5, 0.0]),
        t=np.array([0.0, 1.0, 2.0]),
        i=np.array([-2.0, -1.0, 0.0]),
        metadata={"kind": "cv_data", "scan_rate": 0.1},
        source="synthetic",
    )
    params = _basic_params()
    params["kinetics"][0]["E0"] = 0.0
    objective_norms = []
    monkeypatch.setattr(sim, "get_backend", lambda name="electrokitty": _LinearE0Backend())

    result = sim.fit_cv(
        input_obj,
        "E",
        params,
        fit=[],
        options={
            "plot": False,
            "residual": "direct",
            "residual normalization": "max_abs_measured",
            "progress": lambda event: objective_norms.append(event["cost"]),
        },
    )

    assert result.summary["residual_normalization"] == "max_abs_measured"
    np.testing.assert_allclose(result.residuals, np.array([1.0, 0.5, 0.0]))
    assert objective_norms[-1] == pytest.approx(0.15625)
    assert result.summary["rmse"] == pytest.approx(np.sqrt(np.mean([1.0, 0.25, 0.0])))
    assert result.summary["normalized_rmse"] == pytest.approx(
        np.sqrt(np.mean([(1.0 / 2.0) ** 2, (0.5 / 2.0) ** 2, 0.0]))
    )
    assert result.summary["max_abs_residual"] == pytest.approx(1.0)
    assert result.summary["normalized_max_abs_residual"] == pytest.approx(0.5)
    assert result.summary["normalized_cost"] == pytest.approx(0.15625)
    assert result.summary["normalized_cost_per_point"] == pytest.approx(0.15625 / 3)


def test_fit_cv_can_print_fitting_setup_and_initial_final_params(ecat_module, monkeypatch, capsys):
    sim = ecat_module.simulation
    input_obj = sim.SimulatedCVInput(
        E=np.array([-1.0, -0.5, 0.0]),
        t=np.array([0.0, 1.0, 2.0]),
        i=np.array([-1.5, -1.0, -0.5]),
        metadata={"kind": "cv_data", "scan_rate": 0.1},
        source="synthetic",
    )
    params = _basic_params()
    params["kinetics"][0]["E0"] = -0.2
    monkeypatch.setattr(sim, "get_backend", lambda name="electrokitty": _LinearE0Backend())

    result = sim.fit_cv(
        input_obj,
        "E",
        params,
        fit={"vary": ["E0_0"], "bounds": {"E0_0": [-2.0, 1.0]}, "fixed": {"alpha_0": 0.5}},
        options={
            "plot": False,
            "residual": "direct",
            "max_nfev": 2,
            "print fitting": True,
            "print params": True,
        },
    )

    out = capsys.readouterr().out
    assert "Fitting Setup:" in out
    assert "[fit]" in out
    assert "[vary]" in out
    assert "Control" in out
    assert 'options["residual"]' in out
    assert 'options["max nfev"]' in out
    assert 'fit["vary"]' in out
    assert "Residual mode" in out
    assert "method" in out
    assert "least squares" in out
    assert "Step" not in out
    assert "Step?" not in out
    assert "Path" in out
    assert "cell.T" in out
    assert "diffusion.a" in out
    assert "kinetics.0.E0" in out
    assert "kinetics.0.alpha" in out
    assert "Target" not in out
    assert "E⁰" in out
    assert "E⁰_0" not in out
    assert "α" in out
    assert "V" in out
    assert "Group" in out
    assert "Symbol" not in out
    assert "[fixed]" in out
    assert "Fitting Params:" in out
    assert "Fit Status" in out
    assert "fit" in out
    assert "fixed" in out
    assert "unchanged" not in out
    assert "Initial Value" in out
    assert "Final Value" in out
    assert "-0.2 V" in out
    assert "-0.4 V" in out
    fixed_rows = [
        frame
        for _name, frame in sim._simulation_param_comparison_dataframes(
            result.fit_spec["base_params"],
            result.best_params,
            result.fit_spec,
        )
        if "Fit Status" in frame.columns
    ]
    fixed = pd.concat(fixed_rows, ignore_index=True)
    assert fixed.loc[fixed["Path"] == "cell.T", "Final Value"].iloc[0] == ""
    assert fixed.loc[fixed["Path"] == "kinetics.0.E0", "Final Value"].iloc[0] != ""
    assert "Simulation Params:" not in out
    assert out.index("Fitting Setup:") < out.index("Fitting Params:")
    assert out.count("Fitting Params:") == 1


def test_fit_param_print_shows_step_only_when_multiple_steps(ecat_module, monkeypatch, capsys):
    sim = ecat_module.simulation
    input_obj = sim.SimulatedCVInput(
        E=np.array([-1.0, -0.5, 0.0]),
        t=np.array([0.0, 1.0, 2.0]),
        i=np.array([-1.5, -1.0, -0.5]),
        metadata={"kind": "cv_data", "scan_rate": 0.1},
        source="synthetic",
    )
    params = _basic_params()
    params["concentrations"] = {"bulk": {"a": 1.0, "b": 0.0, "c": 0.0}}
    params["diffusion"] = {"a": 1e-9, "b": 1e-9, "c": 1e-9}
    params["kinetics"] = [
        {"alpha": 0.5, "k0": 1e3, "E0": -0.2},
        {"alpha": 0.5, "k0": 1e3, "E0": -0.4},
    ]
    monkeypatch.setattr(sim, "get_backend", lambda name="electrokitty": _LinearE0Backend())

    sim.fit_cv(
        input_obj,
        "EE",
        params,
        fit={"vary": ["E0_0", "E0_1"], "bounds": {"E0_0": [-2.0, 1.0], "E0_1": [-2.0, 1.0]}},
        options={
            "plot": False,
            "residual": "direct",
            "max_nfev": 1,
            "print fitting": True,
            "print params": True,
        },
    )

    out = capsys.readouterr().out
    assert "Step" in out
    assert "kinetics.0.E0" in out
    assert "kinetics.1.E0" in out


def test_fit_cv_can_print_progress_audit_table(ecat_module, monkeypatch, capsys):
    sim = ecat_module.simulation
    input_obj = sim.SimulatedCVInput(
        E=np.array([-1.0, -0.5, 0.0]),
        t=np.array([0.0, 1.0, 2.0]),
        i=np.array([-1.5, -1.0, -0.5]),
        metadata={"kind": "cv_data", "scan_rate": 0.1},
        source="synthetic",
    )
    params = _basic_params()
    params["kinetics"][0]["E0"] = -0.2
    monkeypatch.setattr(sim, "get_backend", lambda name="electrokitty": _LinearE0Backend())

    sim.fit_cv(
        input_obj,
        "E",
        params,
        fit={"vary": ["E0_0"], "bounds": {"E0_0": [-2.0, 1.0]}},
        options={
            "plot": False,
            "residual": "direct",
            "max_nfev": 2,
            "print progress": True,
        },
    )

    out = capsys.readouterr().out
    assert "Fitting Progression:" in out
    assert "[progress]" not in out
    assert "Eval" in out
    assert "Cost" in out
    assert "ΔCost" in out
    assert "Residual Norm" in out
    assert "Max |Residual|" in out
    assert "Current Scale" not in out
    assert "Baseline Intercept" not in out
    assert "Baseline Slope" not in out
    assert "Changed Params" in out
    assert "E⁰_0=" in out
    assert "E⁰_0=0 V" not in out


def test_fit_cv_multistart_progress_prints_strategy_context(ecat_module, monkeypatch, capsys):
    sim = ecat_module.simulation
    input_obj = sim.SimulatedCVInput(
        E=np.array([0.0, 1.0]),
        t=np.array([0.0, 1.0]),
        i=np.array([0.0, 0.0]),
        metadata={"kind": "cv_data", "scan_rate": 0.1},
        source="synthetic",
    )
    params = _basic_params()
    params["kinetics"][0]["E0"] = 0.0
    monkeypatch.setattr(sim, "get_backend", lambda name="electrokitty": _BimodalE0Backend())

    sim.fit_cv(
        input_obj,
        "E",
        params,
        fit={"vary": ["E0_0"], "bounds": {"E0_0": [-2.0, 2.0]}},
        method={
            "strategy": "multistart",
            "optimizer": "least_squares",
            "starts": 3,
            "start_strategy": "latin_hypercube",
            "seed": 4,
            "max_nfev": 2,
        },
        options={
            "plot": False,
            "residual": "direct",
            "print progress": True,
        },
    )

    out = capsys.readouterr().out
    assert "Fitting Progression:" in out
    assert "Start" in out


def test_fit_progress_print_compacts_long_tables_by_default(ecat_module, capsys):
    sim = ecat_module.simulation
    rows = [
        {
            "Phase": "multistart",
            "Start": index // 10,
            "Eval": index,
            "Cost": float(index),
            "ΔCost": 1.0,
            "Residual Norm": float(index),
            "Max |Residual|": 0.1,
            "Current Scale": "",
            "Baseline Intercept": "",
            "Baseline Slope": "",
            "Changed Params": f"E⁰_0={index}",
        }
        for index in range(40)
    ]

    sim._maybe_print_fit_progress(rows, {"print progress": True})

    out = capsys.readouterr().out
    assert "Fitting Progression:" in out
    assert "Current Scale" not in out
    assert "Baseline Intercept" not in out
    assert "Baseline Slope" not in out
    assert "... " in out or "..." in out
    assert len([line for line in out.splitlines() if "E⁰_0=" in line]) < len(rows)
    assert "E⁰_0=0" in out
    assert "E⁰_0=39" in out


def test_fit_progress_print_all_rows_from_single_option(ecat_module, capsys):
    sim = ecat_module.simulation
    rows = [
        {
            "Eval": index,
            "Cost": float(index),
            "ΔCost": 1.0,
            "Residual Norm": float(index),
            "Max |Residual|": 0.1,
            "Changed Params": f"E⁰_0={index}",
        }
        for index in range(30)
    ]

    sim._maybe_print_fit_progress(rows, {"print progress": "all"})

    out = capsys.readouterr().out
    assert "Fitting Progression:" in out
    assert "..." not in out
    assert len([line for line in out.splitlines() if "E⁰_0=" in line]) == len(rows)


def test_fit_cv_least_squares_reports_progress_callback(ecat_module, monkeypatch):
    sim = ecat_module.simulation
    input_obj = sim.SimulatedCVInput(
        E=np.array([-1.0, -0.5, 0.0]),
        t=np.array([0.0, 1.0, 2.0]),
        i=np.array([-1.5, -1.0, -0.5]),
        metadata={"kind": "cv_data", "scan_rate": 0.1},
        source="synthetic",
    )
    params = _basic_params()
    params["kinetics"][0]["E0"] = -0.2
    progress_events = []
    monkeypatch.setattr(sim, "get_backend", lambda name="electrokitty": _LinearE0Backend())

    result = sim.fit_cv(
        input_obj,
        "E",
        params,
        fit={"vary": ["E0_0"], "bounds": {"E0_0": [-2.0, 1.0]}},
        options={
            "plot": False,
            "residual": "direct",
            "max_nfev": 5,
            "progress": progress_events.append,
        },
    )

    assert progress_events
    assert progress_events[-1]["n_evaluations"] == result.summary["n_evaluations"]
    assert progress_events[-1]["cost"] >= 0


def test_notebook_fit_progress_display_updates(ecat_module, monkeypatch):
    updates = []
    fake_display_module = types.ModuleType("IPython.display")

    class FakeHTML:
        def __init__(self, data):
            self.data = data

    class FakeHandle:
        def update(self, obj):
            updates.append(obj.data)

    def fake_display(obj, display_id=False):
        updates.append(obj.data)
        return FakeHandle()

    fake_display_module.HTML = FakeHTML
    fake_display_module.display = fake_display
    monkeypatch.setitem(sys.modules, "IPython.display", fake_display_module)

    bar = ecat_module.simulation._NotebookFitProgressDisplay(total=3, label="Fit EE", leave=False)
    bar.update(1, cost=0.25)
    bar.update(4, cost=0.125)
    bar.close()

    assert "Fit EE" in updates[0]
    assert "1/~3 eval" in updates[1]
    assert "4/~3 eval" in updates[2]
    assert "(133%)" in updates[2]
    assert "cost 0.125" in updates[2]
    assert "elapsed" in updates[1]
    assert "remaining" in updates[1]
    assert "remaining ~" in updates[1]
    assert "elapsed" in updates[2]
    assert "remaining" not in updates[2]
    assert updates[-1] == ""


def test_notebook_progress_display_can_exceed_100_percent_and_omits_done_remaining(ecat_module):
    sim = ecat_module.simulation

    html_over = sim._NotebookFitProgressDisplay._html(
        7,
        1.23,
        5,
        "Fitting",
        elapsed=10.0,
        remaining=0.0,
    )
    assert "7/~5 eval (140%)" in html_over
    assert "remaining" not in html_over
    assert 'value="5"' in html_over

    html_mid = sim._NotebookFitProgressDisplay._html(
        2,
        None,
        5,
        "Fitting",
        elapsed=4.0,
        remaining=6.0,
    )
    assert "2/~5 eval (40%)" in html_mid
    assert "remaining ~6s" in html_mid


def test_notebook_progress_display_stops_indeterminate_bar_on_close(ecat_module, monkeypatch):
    updates = []
    fake_display_module = types.ModuleType("IPython.display")

    class FakeHTML:
        def __init__(self, data):
            self.data = data

    class FakeHandle:
        def update(self, obj):
            updates.append(obj.data)

    def fake_display(obj, display_id=False):
        updates.append(obj.data)
        return FakeHandle()

    fake_display_module.HTML = FakeHTML
    fake_display_module.display = fake_display
    monkeypatch.setitem(sys.modules, "IPython.display", fake_display_module)

    bar = ecat_module.simulation._NotebookFitProgressDisplay(total=None, label="Fit EE", leave=True)
    bar.update(3, cost=0.25)
    bar.close()

    assert "<progress" in updates[-2]
    assert "3 eval" in updates[-1]
    assert "done" in updates[-1]
    assert "<progress" not in updates[-1]


def test_fit_cv_residual_scale_and_linear_baseline_corrections(ecat_module, monkeypatch):
    sim = ecat_module.simulation
    input_obj = sim.SimulatedCVInput(
        E=np.array([-1.0, 0.0, 1.0]),
        t=np.array([0.0, 1.0, 2.0]),
        i=np.array([2.5, 2.0, 1.5]),
        metadata={"kind": "cv_data", "scan_rate": 0.1},
        source="synthetic",
    )
    params = _basic_params()
    monkeypatch.setattr(sim, "get_backend", lambda name="electrokitty": _UnitSlopeBackend())

    result = sim.fit_cv(
        input_obj,
        "E",
        params,
        fit=[],
        options={"plot": False, "residual": "scale_linear_baseline"},
    )

    assert result.corrections["current_scale"] == pytest.approx(0.0, abs=1e-10)
    assert result.corrections["baseline_intercept"] == pytest.approx(2.0)
    assert result.corrections["baseline_slope"] == pytest.approx(-0.5)
    np.testing.assert_allclose(result.residuals, np.zeros(3), atol=1e-10)
    assert "Measured Current" not in result.simulation_result.data
    np.testing.assert_allclose(result.measured_current, input_obj.i)

    scale_result = sim.fit_cv(
        input_obj,
        "E",
        params,
        fit=[],
        options={"plot": False, "residual": "scale"},
    )
    assert set(scale_result.corrections) == {"current_scale"}


def test_fit_cv_residual_and_post_correction_offset(ecat_module, monkeypatch):
    sim = ecat_module.simulation
    input_obj = sim.SimulatedCVInput(
        E=np.array([-1.0, 0.0, 1.0]),
        t=np.array([0.0, 1.0, 2.0]),
        i=np.array([6.0, 5.0, 6.0]),
        metadata={"kind": "cv_data", "scan_rate": 0.1},
        source="synthetic",
    )
    params = _basic_params()
    monkeypatch.setattr(sim, "get_backend", lambda name="electrokitty": _UnitSlopeBackend())

    offset_result = sim.fit_cv(
        input_obj,
        "E",
        params,
        fit=[],
        options={"plot": False, "residual": "vertical translation"},
    )

    assert offset_result.summary["residual"] == "offset"
    assert set(offset_result.corrections) == {"baseline_intercept"}
    assert offset_result.corrections["baseline_intercept"] == pytest.approx(5.0)
    np.testing.assert_allclose(offset_result.residuals, np.zeros(3), atol=1e-12)

    post_result = sim.fit_cv(
        input_obj,
        "E",
        params,
        fit=[],
        options={"plot": False, "residual": "direct", "post correction": "offset"},
    )

    assert post_result.summary["residual"] == "direct"
    assert post_result.summary["post_correction"] == "offset"
    assert set(post_result.corrections) == {"baseline_intercept"}
    assert post_result.corrections["baseline_intercept"] == pytest.approx(5.0)
    np.testing.assert_allclose(post_result.residuals, np.zeros(3), atol=1e-12)


def test_fit_cv_prints_statistics_and_corrections(ecat_module, monkeypatch, capsys):
    sim = ecat_module.simulation
    input_obj = sim.SimulatedCVInput(
        E=np.array([-1.0, 0.0, 1.0]),
        t=np.array([0.0, 1.0, 2.0]),
        i=np.array([2.5, 2.0, 1.5]),
        metadata={"kind": "cv_data", "scan_rate": 0.1},
        source="synthetic",
    )
    params = _basic_params()
    monkeypatch.setattr(sim, "get_backend", lambda name="electrokitty": _UnitSlopeBackend())

    sim.fit_cv(
        input_obj,
        "E",
        params,
        fit=[],
        options={
            "plot": False,
            "residual": "scale linear baseline",
            "print stats": True,
        },
    )

    out = capsys.readouterr().out
    assert "Fitting Statistics:" in out
    assert "RMSE" in out
    assert "Normalized RMSE" in out
    assert "Degrees of Freedom" in out
    assert "Fitting Corrections:" in out
    assert "Current Scale" in out
    assert "Baseline Intercept" in out
    assert "Baseline Slope" in out


def test_fit_cv_print_corrections_without_stats(ecat_module, monkeypatch, capsys):
    sim = ecat_module.simulation
    input_obj = sim.SimulatedCVInput(
        E=np.array([-1.0, 0.0, 1.0]),
        t=np.array([0.0, 1.0, 2.0]),
        i=np.array([2.5, 2.0, 1.5]),
        metadata={"kind": "cv_data", "scan_rate": 0.1},
        source="synthetic",
    )
    params = _basic_params()
    monkeypatch.setattr(sim, "get_backend", lambda name="electrokitty": _UnitSlopeBackend())

    sim.fit_cv(
        input_obj,
        "E",
        params,
        fit=[],
        options={
            "plot": False,
            "residual": "scale linear baseline",
            "print corrections": True,
        },
    )

    out = capsys.readouterr().out
    assert "Fitting Statistics:" not in out
    assert "Fitting Corrections:" in out
    assert "Baseline Intercept" in out


def test_simulation_fit_result_show_uses_polished_sections(ecat_module, monkeypatch, capsys):
    sim = ecat_module.simulation
    input_obj = sim.SimulatedCVInput(
        E=np.array([-1.0, 0.0, 1.0]),
        t=np.array([0.0, 1.0, 2.0]),
        i=np.array([2.5, 2.0, 1.5]),
        metadata={"kind": "cv_data", "scan_rate": 0.1},
        source="synthetic",
    )
    params = _basic_params()
    monkeypatch.setattr(sim, "get_backend", lambda name="electrokitty": _UnitSlopeBackend())
    fit_result = sim.fit_cv(
        input_obj,
        "E",
        params,
        fit=[],
        options={
            "plot": False,
            "residual": "scale linear baseline",
            "print setup": False,
            "print params": False,
            "progress": False,
        },
    )

    capsys.readouterr()
    fit_result.show({
        "print setup": True,
        "print stats": True,
        "print corrections": True,
        "print params": True,
        "print simulation": True,
    })

    out = capsys.readouterr().out
    assert "Fitting Setup:" in out
    assert "Fitting Statistics:" in out
    assert "Fitting Corrections:" in out
    assert "Fitting Params:" in out


def test_simulation_fit_result_show_pretty_print_false_avoids_rich_display(ecat_module, monkeypatch, capsys):
    sim = ecat_module.simulation
    input_obj = sim.SimulatedCVInput(
        E=np.array([-1.0, 0.0, 1.0]),
        t=np.array([0.0, 1.0, 2.0]),
        i=np.array([2.5, 2.0, 1.5]),
        metadata={"kind": "cv_data", "scan_rate": 0.1},
        source="synthetic",
    )
    params = _basic_params()
    monkeypatch.setattr(sim, "get_backend", lambda name="electrokitty": _UnitSlopeBackend())

    import IPython.display as ipd

    rich_calls = []

    monkeypatch.setattr(sim, "_can_rich_display", lambda: True)
    monkeypatch.setattr(ipd, "display", lambda obj: rich_calls.append(obj))
    monkeypatch.setattr(ipd, "Markdown", lambda text: text)

    fit_result = sim.fit_cv(
        input_obj,
        "E",
        params,
        fit=[],
        options={
            "plot": False,
            "residual": "scale linear baseline",
            "print setup": False,
            "print params": False,
            "progress": False,
        },
    )

    capsys.readouterr()
    fit_result.show({
        "pretty print": False,
        "print stats": True,
        "print corrections": True,
    })

    out = capsys.readouterr().out
    assert "Fitting Statistics:" in out
    assert "Fitting Corrections:" in out
    assert "Parameter" in out
    assert "Value" in out
    assert rich_calls == []
    assert "RMSE" in out
    assert "Current Scale" in out
    assert "cell.T" in out


def test_fit_cv_accepts_spaced_option_keys_and_values(ecat_module, monkeypatch):
    sim = ecat_module.simulation
    input_obj = sim.SimulatedCVInput(
        E=np.array([-1.0, -0.5, 0.0]),
        t=np.array([0.0, 1.0, 2.0]),
        i=np.array([-1.5, -1.0, -0.5]),
        metadata={"kind": "cv_data", "scan_rate": 0.1},
        source="synthetic",
    )
    params = _basic_params()
    params["kinetics"][0]["E0"] = -0.2
    monkeypatch.setattr(sim, "get_backend", lambda name="electrokitty": _LinearE0Backend())

    result = sim.fit_cv(
        input_obj,
        "E",
        params,
        fit={"vary": ["E0_0"], "bounds": {"E0_0": [-2.0, 1.0]}, "fixed": {"alpha_0": 0.5}},
        method="least squares",
        options={
            "plot": False,
            "residual": "direct",
            "post correction": "scale linear baseline",
            "residual normalization": "max abs measured",
            "max nfev": 5,
            "print progress": False,
        },
    )

    assert result.method == "least_squares"
    assert result.summary["post_correction"] == "scale_linear_baseline"
    assert result.summary["residual_normalization"] == "max_abs_measured"
    assert result.optimizer_result.nfev <= 5


def test_fit_cv_post_correction_is_not_used_during_optimization(ecat_module, monkeypatch):
    sim = ecat_module.simulation
    input_obj = sim.SimulatedCVInput(
        E=np.array([-1.0, -0.5, 0.0]),
        t=np.array([0.0, 1.0, 2.0]),
        i=np.array([-1.5, -1.0, -0.5]),
        metadata={"kind": "cv_data", "scan_rate": 0.1},
        source="synthetic",
    )
    params = _basic_params()
    params["kinetics"][0]["E0"] = -0.2
    monkeypatch.setattr(sim, "get_backend", lambda name="electrokitty": _LinearE0Backend())

    result = sim.fit_cv(
        input_obj,
        "E",
        params,
        fit={"vary": ["E0_0"], "bounds": {"E0_0": [-2.0, 1.0]}},
        options={"plot": False, "residual": "direct", "post correction": "scale_linear_baseline"},
    )

    assert result.best_params["kinetics"][0]["E0"] == pytest.approx(-0.5, abs=1e-6)
    assert result.summary["residual"] == "direct"
    assert result.summary["post_correction"] == "scale_linear_baseline"
    assert set(result.corrections) == {"current_scale", "baseline_intercept", "baseline_slope"}
    assert result.simulation_result.data.attrs["post_correction"] == "scale_linear_baseline"


def test_fit_cv_plot_uses_fit_labels_by_default(ecat_module, monkeypatch):
    sim = ecat_module.simulation
    input_obj = sim.SimulatedCVInput(
        E=np.array([-1.0, 0.0, 1.0]),
        t=np.array([0.0, 1.0, 2.0]),
        i=np.array([-1.0, 0.0, 1.0]),
        metadata={"kind": "cv_data", "scan_rate": 0.1},
        source="synthetic",
    )
    monkeypatch.setattr(sim, "get_backend", lambda name="electrokitty": _UnitSlopeBackend())

    result = sim.fit_cv(
        input_obj,
        "E",
        _basic_params(),
        fit=[],
        options={"plot": True, "plot all": True},
    )

    labels = [line.get_label() for line in result.simulation_result.axes.lines]
    assert labels == ["Simulated Fit", "Measured Data", "Raw Simulated Fit"]


def test_fit_cv_accepts_initial_simulation_result_and_plot_all(ecat_module, monkeypatch):
    sim = ecat_module.simulation
    input_obj = sim.SimulatedCVInput(
        E=np.array([-1.0, 0.0, 1.0]),
        t=np.array([0.0, 1.0, 2.0]),
        i=np.array([-1.0, 0.0, 1.0]),
        metadata={"kind": "cv_data", "scan_rate": 0.1},
        source="synthetic",
    )
    params = _basic_params()
    monkeypatch.setattr(sim, "get_backend", lambda name="electrokitty": _UnitSlopeBackend())
    initial = sim.simulate_cv(input_obj, "E", params, options={"plot": False})

    fit_result = sim.fit_cv(
        initial,
        fit=[],
        options={"plot": True, "plot all": True},
    )

    assert fit_result.initial_result is initial
    assert fit_result.simulation_result.mechanism.mechanism == "E(1):a=b"
    assert fit_result.simulation_result.axes is not None
    assert len(fit_result.simulation_result.axes.lines) == 3


def test_fit_cv_requires_measured_current(ecat_module):
    sim = ecat_module.simulation
    program = sim.cv_program(0.1, -0.1, points_per_segment=3)

    with pytest.raises(ValueError, match="measured current"):
        sim.fit_cv(program, "E", _basic_params(), fit=[], options={"plot": False})


def test_fit_cv_accepts_real_cv_input(ecat_module, cv_factory, monkeypatch):
    sim = ecat_module.simulation
    potential = np.array([-1.0, -0.5, 0.0])
    cv_obj = cv_factory(
        name="100mVs_real_fit_cv",
        potential=potential,
        current=potential - 0.5,
    )
    params = _basic_params()
    params["kinetics"][0]["E0"] = -0.2
    monkeypatch.setattr(sim, "get_backend", lambda name="electrokitty": _LinearE0Backend())

    result = sim.fit_cv(
        cv_obj,
        "E",
        params,
        fit={"vary": ["E0_0"], "bounds": {"E0_0": [-2.0, 1.0]}},
        options={"plot": False, "cv data": {"stride": 1}},
    )

    assert isinstance(result, sim.SimulationFitResult)
    assert result.simulation_result.input.source is cv_obj
    assert result.best_params["kinetics"][0]["E0"] == pytest.approx(-0.5, abs=1e-6)


def test_fit_cv_defaults_print_setup_params_and_progress_bar(ecat_module, monkeypatch, capsys):
    sim = ecat_module.simulation
    progress_options = []

    class Reporter:
        def __init__(self, options):
            progress_options.append(dict(options))
            self.count = 0

        def update(self, *, residuals=None, params=None):
            self.count += 1

        def close(self):
            pass

    input_obj = sim.SimulatedCVInput(
        E=np.array([-1.0, -0.5, 0.0]),
        t=np.array([0.0, 1.0, 2.0]),
        i=np.array([-1.0, -0.5, 0.0]),
        metadata={"kind": "manual", "scan_rate": 0.1},
        source="synthetic",
    )
    monkeypatch.setattr(sim, "get_backend", lambda name="electrokitty": _UnitSlopeBackend())
    monkeypatch.setattr(sim, "_FitProgressReporter", Reporter)

    sim.fit_cv(input_obj, "E", _basic_params(), fit=[], options={"plot": False})

    out = capsys.readouterr().out
    assert "Fitting Setup:" in out
    assert "Fitting Params:" in out
    assert "Fitting Progression:" not in out
    assert progress_options
    assert progress_options[0]["progress"] is True


def test_fit_cv_explicit_false_suppresses_default_fit_printing_and_progress(ecat_module, monkeypatch, capsys):
    sim = ecat_module.simulation
    progress_options = []

    class Reporter:
        def __init__(self, options):
            progress_options.append(dict(options))
            self.count = 0

        def update(self, *, residuals=None, params=None):
            self.count += 1

        def close(self):
            pass

    input_obj = sim.SimulatedCVInput(
        E=np.array([-1.0, -0.5, 0.0]),
        t=np.array([0.0, 1.0, 2.0]),
        i=np.array([-1.0, -0.5, 0.0]),
        metadata={"kind": "manual", "scan_rate": 0.1},
        source="synthetic",
    )
    monkeypatch.setattr(sim, "get_backend", lambda name="electrokitty": _UnitSlopeBackend())
    monkeypatch.setattr(sim, "_FitProgressReporter", Reporter)

    sim.fit_cv(
        input_obj,
        "E",
        _basic_params(),
        fit=[],
        options={"plot": False, "print setup": False, "print params": False, "progress": False},
    )

    out = capsys.readouterr().out
    assert "Fitting Setup:" not in out
    assert "Fitting Params:" not in out
    assert progress_options[0]["progress"] is False


def test_fit_cv_real_cv_uses_trimmed_cv_data_for_fit_and_plot(ecat_module, cv_factory, monkeypatch):
    sim = ecat_module.simulation
    potential = np.array([0.5, 0.0, -0.5, -1.0, -1.5, -2.0])
    cv_obj = cv_factory(
        name="100mVs_real_fit_cv_window",
        potential=potential,
        current=potential - 0.5,
    )
    monkeypatch.setattr(sim, "get_backend", lambda name="electrokitty": _UnitSlopeBackend())

    result = sim.fit_cv(
        cv_obj,
        "E",
        _basic_params(),
        fit=[],
        options={
            "plot": True,
            "cv data": {
                "potential window": [0.0, -1.5],
                "trim mode": "pointwise",
                "stride": 1,
            },
        },
    )

    expected_E = np.array([0.0, -0.5, -1.0, -1.5])
    np.testing.assert_allclose(result.simulation_result.input.E, expected_E)
    np.testing.assert_allclose(result.measured_current, expected_E - 0.5)
    ax = result.simulation_result.axes
    np.testing.assert_allclose(ax.lines[0].get_xdata(), expected_E)
    np.testing.assert_allclose(ax.lines[1].get_xdata(), expected_E)
    assert all(np.nanmin(line.get_xdata()) >= -1.5 for line in ax.lines[:2])
    assert all(np.nanmax(line.get_xdata()) <= 0.0 for line in ax.lines[:2])
    plt.close(ax.figure)


def test_fit_cvs_fits_shared_parameter_from_real_cvs(ecat_module, cv_factory, monkeypatch):
    sim = ecat_module.simulation
    potential = np.array([-1.0, -0.5, 0.0])
    cvs = [
        cv_factory(name="100mVs_group_fit_1", potential=potential, current=potential - 0.5),
        cv_factory(name="200mVs_group_fit_2", potential=potential, current=potential - 0.5),
    ]
    params = _basic_params()
    params["kinetics"][0]["E0"] = -0.2
    monkeypatch.setattr(sim, "get_backend", lambda name="electrokitty": _LinearE0Backend())

    result = sim.fit_cvs(
        cvs,
        "E",
        params,
        fit={"vary": ["E0_0"], "bounds": {"E0_0": [-2.0, 1.0]}},
        per_cv=[],
        options={"plot": False, "cv data": {"stride": 1}},
    )

    assert isinstance(result, sim.SimulationGroupFitResult)
    assert result.best_params["kinetics"][0]["E0"] == pytest.approx(-0.5, abs=1e-6)
    assert len(result.simulation_results) == 2
    assert len(result.measured_currents) == 2
    assert all("Measured Current" not in item.data for item in result.simulation_results)


def test_fit_cvs_group_setup_uses_multiplot_labels(ecat_module, cv_factory, monkeypatch, capsys):
    sim = ecat_module.simulation
    potential = np.array([-1.0, -0.5, 0.0])
    cvs = [
        cv_factory(name="100mVs_sample_CO2_MeCN_1mM_PhOH_run01", potential=potential, current=potential),
        cv_factory(name="100mVs_sample_CO2_MeCN_2mM_PhOH_run02", potential=potential, current=potential),
    ]
    monkeypatch.setattr(sim, "get_backend", lambda name="electrokitty": _UnitSlopeBackend())

    result = sim.fit_cvs(
        cvs,
        "E",
        _basic_params(),
        fit=[],
        per_cv=[],
        options={"plot": False, "print params": False, "progress": False, "cv data": {"stride": 1}},
    )

    out = capsys.readouterr().out
    labels = [dataset["label"] for dataset in result.datasets]
    assert labels == ["1 mM PhOH", "2 mM PhOH"]
    assert "Group Fitting Setup:" in out
    assert "1 mM PhOH" in out
    assert "2 mM PhOH" in out


def test_fit_cvs_accepts_mixed_resolved_inputs(ecat_module, cv_factory, monkeypatch):
    sim = ecat_module.simulation
    potential = np.array([-1.0, -0.5, 0.0])
    cv_obj = cv_factory(name="100mVs_group_real", potential=potential, current=potential - 0.5)
    input_obj = sim.SimulatedCVInput(
        E=potential,
        t=np.array([0.0, 1.0, 2.0]),
        i=potential - 0.5,
        metadata={"kind": "manual", "scan_rate": 0.1},
        source="manual",
    )
    params = _basic_params()
    params["kinetics"][0]["E0"] = -0.2
    monkeypatch.setattr(sim, "get_backend", lambda name="electrokitty": _LinearE0Backend())
    initial = sim.simulate_cv(input_obj, "E", params, options={"plot": False})

    result = sim.fit_cvs(
        [cv_obj, input_obj, initial],
        "E",
        params,
        fit={"vary": ["E0_0"], "bounds": {"E0_0": [-2.0, 1.0]}},
        per_cv=[],
        options={"plot": False, "cv data": {"stride": 1}},
    )

    assert len(result.datasets) == 3
    assert result.best_params["kinetics"][0]["E0"] == pytest.approx(-0.5, abs=1e-6)


def test_fit_cvs_expands_varied_per_cv_parameter(ecat_module, monkeypatch):
    sim = ecat_module.simulation
    inputs = [
        sim.SimulatedCVInput(
            E=np.array([0.0, 1.0]),
            t=np.array([0.0, 1.0]),
            i=np.array([2.0, 2.0]),
            metadata={"kind": "manual", "scan_rate": 0.1},
            source="cv 1",
        ),
        sim.SimulatedCVInput(
            E=np.array([0.0, 1.0]),
            t=np.array([0.0, 1.0]),
            i=np.array([4.0, 4.0]),
            metadata={"kind": "manual", "scan_rate": 0.1},
            source="cv 2",
        ),
    ]
    params = _basic_params()
    params["cell"]["Cdl"] = 1.0
    monkeypatch.setattr(sim, "get_backend", lambda name="electrokitty": _CdlBackend())

    result = sim.fit_cvs(
        inputs,
        "E",
        params,
        fit={"vary": ["cell.Cdl"], "bounds": {"cell.Cdl": [0.1, 10.0]}, "transform": {"cell.Cdl": "linear"}},
        per_cv=["cell.Cdl"],
        options={"plot": False, "residual": "direct"},
    )

    assert result.per_cv_paths == [("cell", "Cdl")]
    assert result.best_params_by_cv[0]["cell"]["Cdl"] == pytest.approx(2.0, abs=1e-6)
    assert result.best_params_by_cv[1]["cell"]["Cdl"] == pytest.approx(4.0, abs=1e-6)
    assert result.best_params["cell"]["Cdl"] == pytest.approx(1.0)


def test_simulation_group_fit_show_prints_corrections_without_missing_helper(ecat_module, capsys):
    sim = ecat_module.simulation
    result = sim.SimulationGroupFitResult(
        best_params={},
        best_params_by_cv=[],
        fit_spec={"base_params": {}},
        per_cv=[],
        method="least_squares",
        backend="fake",
        corrections_by_cv=[
            {"baseline_intercept": 1e-6, "current_scale": 2.0},
            {},
        ],
        datasets=[
            {"label": "CV A"},
            {"label": "CV B"},
        ],
    )

    result.show({"print setup": False, "print stats": False, "print corrections": True})

    out = capsys.readouterr().out
    assert "Group Fitting Corrections:" in out
    assert not out.startswith("Fitting Corrections:")
    assert "\nFitting Corrections:" not in out
    assert "Baseline Intercept" in out
    assert "Current Scale" in out
    assert "(none)" in out


def test_fit_cvs_auto_maps_differing_metadata_concentrations(ecat_module, monkeypatch):
    sim = ecat_module.simulation
    inputs = [
        sim.SimulatedCVInput(
            E=np.array([0.0, 1.0]),
            t=np.array([0.0, 1.0]),
            i=np.array([0.0, 1.0]),
            metadata={"kind": "manual", "scan_rate": 0.1, "concentrations": {"Substrate": 2.0}},
            source="cv 1",
        ),
        sim.SimulatedCVInput(
            E=np.array([0.0, 1.0]),
            t=np.array([0.0, 1.0]),
            i=np.array([0.0, 1.0]),
            metadata={"kind": "manual", "scan_rate": 0.2, "concentrations": {"Substrate": 4.0}},
            source="cv 2",
        ),
    ]
    params = _basic_params()
    params["concentrations"] = {"bulk": {"a": 1.0, "b": 0.0, "Substrate": 1.0}}
    params["diffusion"]["Substrate"] = 1e-9
    monkeypatch.setattr(sim, "get_backend", lambda name="electrokitty": _UnitSlopeBackend())

    result = sim.fit_cvs(inputs, "E", params, fit=[], per_cv="auto", options={"plot": False})

    assert ("concentrations", "bulk", "Substrate") in result.per_cv_paths
    assert result.best_params_by_cv[0]["concentrations"]["bulk"]["Substrate"] == pytest.approx(2.0)
    assert result.best_params_by_cv[1]["concentrations"]["bulk"]["Substrate"] == pytest.approx(4.0)
    assert result.datasets[0]["scan_rate"] == pytest.approx(0.1)
    assert result.datasets[1]["scan_rate"] == pytest.approx(0.2)


def test_fit_cvs_uses_concentration_mapping_option(ecat_module, monkeypatch):
    sim = ecat_module.simulation
    inputs = [
        sim.SimulatedCVInput(
            E=np.array([0.0, 1.0]),
            t=np.array([0.0, 1.0]),
            i=np.array([0.0, 1.0]),
            metadata={"kind": "manual", "scan_rate": 0.1, "concentrations": {"PhOH": 2.0}},
            source="cv 1",
        ),
        sim.SimulatedCVInput(
            E=np.array([0.0, 1.0]),
            t=np.array([0.0, 1.0]),
            i=np.array([0.0, 1.0]),
            metadata={"kind": "manual", "scan_rate": 0.1, "concentrations": {"PhOH": 4.0}},
            source="cv 2",
        ),
    ]
    params = _basic_params()
    params["concentrations"] = {"bulk": {"a": 1.0, "b": 0.0, "Substrate": 1.0}}
    params["diffusion"]["Substrate"] = 1e-9
    monkeypatch.setattr(sim, "get_backend", lambda name="electrokitty": _UnitSlopeBackend())

    result = sim.fit_cvs(
        inputs,
        "E",
        params,
        fit=[],
        per_cv="auto",
        options={"plot": False, "concentration mapping": {"PhOH": "Substrate"}},
    )

    assert result.best_params_by_cv[0]["concentrations"]["bulk"]["Substrate"] == pytest.approx(2.0)
    assert result.best_params_by_cv[1]["concentrations"]["bulk"]["Substrate"] == pytest.approx(4.0)


def test_fit_cvs_setup_shows_mapped_concentrations_not_repeated_scan_rate(ecat_module, monkeypatch, capsys):
    sim = ecat_module.simulation

    def source(name, concentration):
        return types.SimpleNamespace(
            name=f"{concentration.replace(' ', '')}_PhOH",
            compounds=["PhOH"],
            concentrations=[concentration],
            txt_stats=lambda options=None, c=concentration: {
                "gas": "CO2",
                "compounds": f"{c} PhOH",
                "scan rate": "0.1 V/s",
            },
        )

    inputs = [
        sim.SimulatedCVInput(
            E=np.array([0.0, 1.0]),
            t=np.array([0.0, 1.0]),
            i=np.array([0.0, 1.0]),
            metadata={"kind": "manual", "scan_rate": 0.1},
            source=source("cv 1", "100 mM"),
        ),
        sim.SimulatedCVInput(
            E=np.array([0.0, 1.0]),
            t=np.array([0.0, 1.0]),
            i=np.array([0.0, 1.0]),
            metadata={"kind": "manual", "scan_rate": 0.1, "concentrations": {}},
            source=source("cv 2", "560 mM"),
        ),
    ]
    params = _basic_params()
    params["concentrations"] = {"bulk": {"a": 1.0, "b": 0.0, "Substrate": 1.0}}
    params["diffusion"]["Substrate"] = 1e-9
    monkeypatch.setattr(sim, "get_backend", lambda name="electrokitty": _UnitSlopeBackend())

    result = sim.fit_cvs(
        inputs,
        "E",
        params,
        fit=[],
        per_cv=["concentrations.bulk.Substrate"],
        options={
            "plot": False,
            "concentration mapping": {"PhOH": "Substrate"},
            "print setup": True,
            "print params": False,
            "progress": False,
        },
    )

    out = capsys.readouterr().out
    assert "Input Concentrations" in out
    assert "Mapped Concentrations" in out
    assert "100 mM PhOH" in out
    assert "560 mM PhOH" in out
    assert "PhOH → Substrate" in out
    assert "100 mol/m³" in out
    assert "560 mol/m³" in out
    assert "Scan Rate" not in out
    assert result.best_params_by_cv[0]["concentrations"]["bulk"]["Substrate"] == pytest.approx(100.0)
    assert result.best_params_by_cv[1]["concentrations"]["bulk"]["Substrate"] == pytest.approx(560.0)


def test_fit_cvs_print_params_includes_per_cv_concentration_values(ecat_module, monkeypatch, capsys):
    sim = ecat_module.simulation
    inputs = [
        sim.SimulatedCVInput(
            E=np.array([0.0, 1.0]),
            t=np.array([0.0, 1.0]),
            i=np.array([0.0, 1.0]),
            metadata={"kind": "manual", "scan_rate": 0.1, "concentrations": {"PhOH": 2.0}},
            source="cv 1",
        ),
        sim.SimulatedCVInput(
            E=np.array([0.0, 1.0]),
            t=np.array([0.0, 1.0]),
            i=np.array([0.0, 1.0]),
            metadata={"kind": "manual", "scan_rate": 0.1, "concentrations": {"PhOH": 4.0}},
            source="cv 2",
        ),
    ]
    params = _basic_params()
    params["concentrations"] = {"bulk": {"a": 1.0, "b": 0.0, "Substrate": 1.0}}
    params["diffusion"]["Substrate"] = 1e-9
    monkeypatch.setattr(sim, "get_backend", lambda name="electrokitty": _UnitSlopeBackend())

    sim.fit_cvs(
        inputs,
        "E",
        params,
        fit=[],
        per_cv=["concentrations.bulk.Substrate"],
        options={
            "plot": False,
            "concentration mapping": {"PhOH": "Substrate"},
            "print params": True,
            "print setup": False,
            "progress": False,
        },
    )

    out = capsys.readouterr().out
    assert "Per-CV Fitting Params:" in out
    assert "concentrations.bulk.Substrate" in out
    assert "2 mol/m³" in out
    assert "4 mol/m³" in out


def test_fit_cvs_plot_overlays_measured_per_cv(ecat_module, monkeypatch):
    sim = ecat_module.simulation
    inputs = [
        sim.SimulatedCVInput(
            E=np.array([-1.0, 0.0, 1.0]),
            t=np.array([0.0, 1.0, 2.0]),
            i=np.array([-1.0, 0.0, 1.0]),
            metadata={"kind": "manual", "scan_rate": 0.1},
            source="cv 1",
        ),
        sim.SimulatedCVInput(
            E=np.array([-1.0, 0.0, 1.0]),
            t=np.array([0.0, 1.0, 2.0]),
            i=np.array([-1.0, 0.0, 1.0]),
            metadata={"kind": "manual", "scan_rate": 0.1},
            source="cv 2",
        ),
    ]
    monkeypatch.setattr(sim, "get_backend", lambda name="electrokitty": _UnitSlopeBackend())
    result = sim.fit_cvs(inputs, "E", _basic_params(), fit=[], per_cv=[], options={"plot": False})

    axes = result.plot({"plot all": True})

    assert len(axes) == 2
    assert [line.get_label() for line in axes[0].lines] == ["Simulated Fit", "Measured Data", "Raw Simulated Fit"]
    assert len(result.simulation_results[0].plot({"legend": False}).lines) == 1
    for ax in axes:
        plt.close(ax.figure)


def test_fit_cvs_rejects_electrokitty_native_fitter(ecat_module):
    sim = ecat_module.simulation
    input_obj = sim.SimulatedCVInput(
        E=np.array([0.0, 1.0]),
        t=np.array([0.0, 1.0]),
        i=np.array([0.0, 1.0]),
        metadata={"kind": "manual", "scan_rate": 0.1},
        source="cv 1",
    )

    with pytest.raises(ValueError, match="group fitting"):
        sim.fit_cvs([input_obj, input_obj], "E", _basic_params(), fit=[], options={"plot": False}, method="electrokitty")


def test_simulation_group_fit_result_show_includes_dataset_sources(ecat_module, monkeypatch, capsys):
    sim = ecat_module.simulation
    inputs = [
        sim.SimulatedCVInput(
            E=np.array([0.0, 1.0]),
            t=np.array([0.0, 1.0]),
            i=np.array([0.0, 1.0]),
            metadata={"kind": "manual", "scan_rate": 0.1},
            source="source one",
        ),
        sim.SimulatedCVInput(
            E=np.array([0.0, 1.0]),
            t=np.array([0.0, 1.0]),
            i=np.array([0.0, 1.0]),
            metadata={"kind": "manual", "scan_rate": 0.2},
            source="source two",
        ),
    ]
    monkeypatch.setattr(sim, "get_backend", lambda name="electrokitty": _UnitSlopeBackend())
    result = sim.fit_cvs(inputs, "E", _basic_params(), fit=[], per_cv=[], options={"plot": False})

    capsys.readouterr()
    result.show({"print setup": True})

    out = capsys.readouterr().out
    assert "Group Fitting Setup:" in out
    assert "Source" in out
    assert "source one" in out
    assert "source two" in out


def test_fit_cv_electrokitty_method_passes_options(ecat_module, monkeypatch):
    sim = ecat_module.simulation
    fake_module = types.ModuleType("electrokitty")
    fake_module.ElectroKitty = _FakeFittingElectroKitty
    monkeypatch.setitem(sys.modules, "electrokitty", fake_module)
    input_obj = sim.SimulatedCVInput(
        E=np.array([0.1, 0.0, -0.1]),
        t=np.array([0.0, 1.0, 2.0]),
        i=np.array([1.0, 2.0, 3.0]),
        metadata={"kind": "cv_data", "scan_rate": 0.1},
        source="synthetic",
    )

    result = sim.fit_cv(
        input_obj,
        "E",
        _basic_params(),
        method="electrokitty",
        options={"plot": False, "electrokitty": {"fit_kin": True, "fit_Cdl": True, "algorithm": "CMA-ES"}},
    )

    assert result.method == "electrokitty"
    assert result.backend_result.fit_kwargs["fit_Cdl"] is True
    assert result.backend_result.fit_kwargs["algorithm"] == "CMA-ES"

    with pytest.raises(ValueError, match="does not honor arbitrary"):
        sim.fit_cv(
            input_obj,
            "E",
            _basic_params(),
            fit=["E0_0"],
            method="electrokitty",
            options={"plot": False},
        )


def _basic_params():
    return {
        "cell": {"T": 298.15, "Ru": 0, "Cdl": 0, "A": 1e-5},
        "spatial": {"dx_fraction": 0.001 / 36, "nx": 20, "viscosity": 1e-5, "rotation": 0},
        "diffusion": {"a": 1e-9, "b": 1e-9},
        "concentrations": {"surface": {}, "bulk": {"a": 1, "b": 0}},
        "kinetics": [{"alpha": 0.5, "k0": 1e-3, "E0": -1.4}],
        "isotherm": [],
    }


def _surface_params():
    return {
        **_basic_params(),
        "concentrations": {"surface": {"a": 1e-5, "b": 0}},
    }


class _FakeElectroKitty:
    def __init__(self, mechanism):
        self.mechanism = mechanism
        self.created = None
        self.E_generated = None
        self.current = None
        self.t = None
        self.I_data = None

    def set_data(self, E, i, t):
        self.E_generated = np.asarray(E, dtype=float)
        self.I_data = np.asarray(i, dtype=float)
        self.t = np.asarray(t, dtype=float)

    def create_simulation(
        self,
        kin,
        cell_const,
        diffusion_const,
        isotherm,
        spatial_info,
        species_information,
        kinetic_model="BV",
    ):
        self.created = {
            "kin": kin,
            "cell_const": cell_const,
            "diffusion_const": diffusion_const,
            "isotherm": isotherm,
            "spatial_info": spatial_info,
            "species_information": species_information,
            "kinetic_model": kinetic_model,
        }

    def simulate(self):
        self.current = np.arange(1, len(self.E_generated) + 1, dtype=float)
        return self.E_generated, self.current, self.t


class _FakeFittingElectroKitty(_FakeElectroKitty):
    def fit_to_data(self, **kwargs):
        self.fit_kwargs = kwargs
        self.current = np.asarray(self.I_data, dtype=float)
        self.E_Corr = np.asarray(self.E_generated, dtype=float)
        self.fit_score = 0.0

    def print_fitting_parameters(self):
        return (
            self.created["kin"],
            self.created["species_information"],
            self.created["cell_const"],
            self.created["isotherm"],
        )


class _LinearE0Backend:
    name = "fake"

    def simulate(self, input, mechanism_spec, params, options):
        e0 = float(params["kinetics"][0]["E0"])
        E = np.asarray(input.E, dtype=float)
        current = E + e0
        return E, current, np.asarray(input.t, dtype=float), {"e0": e0}


class _BimodalE0Backend:
    name = "fake-bimodal"

    def simulate(self, input, mechanism_spec, params, options):
        e0 = float(params["kinetics"][0]["E0"])
        E = np.asarray(input.E, dtype=float)
        response = 0.5 if abs(e0) < 0.2 else e0**2 - 1.0
        current = np.full_like(E, response, dtype=float)
        return E, current, np.asarray(input.t, dtype=float), {"e0": e0}


class _UnitSlopeBackend:
    name = "fake"

    def simulate(self, input, mechanism_spec, params, options):
        E = np.asarray(input.E, dtype=float)
        return E, E**2, np.asarray(input.t, dtype=float), {"unit": True}


class _CdlBackend:
    name = "fake-cdl"

    def simulate(self, input, mechanism_spec, params, options):
        E = np.asarray(input.E, dtype=float)
        cdl = float(params["cell"]["Cdl"])
        return E, np.full_like(E, cdl, dtype=float), np.asarray(input.t, dtype=float), {"Cdl": cdl}
