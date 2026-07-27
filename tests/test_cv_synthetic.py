import numpy as np
import pandas as pd
import pytest
import sys
import matplotlib.pyplot as plt
from io import BytesIO, TextIOWrapper


def _make_dual_peak_cv(blank_echem_factory, ecat_module):
    seg1 = np.round(np.linspace(-0.3, 0.3, 25), 3)
    seg2 = np.round(np.linspace(0.3, -0.3, 25)[1:], 3)
    potential = np.concatenate([seg1, seg2])

    current = []
    for x in potential:
        baseline = 1.5e-6 * x + 2e-7
        anodic = 4.0e-6 * np.exp(-((x - 0.12) / 0.05) ** 2)
        cathodic = -3.0e-6 * np.exp(-((x + 0.11) / 0.05) ** 2)
        shoulder = 1.2e-6 * np.exp(-((x - 0.23) / 0.03) ** 2)
        current.append(baseline + anodic + cathodic + shoulder)

    data = pd.DataFrame(
        {
            ("raw", "Potential (V)"): potential,
            ("raw", "Current (A)"): current,
        }
    )

    obj = blank_echem_factory(ecat_module.cv)
    obj.manual_init("100mVs_dual_peak_Fc", data, options={})
    return obj


def _make_synthetic_cv(blank_echem_factory, ecat_module, potential, current, name):
    data = pd.DataFrame(
        {
            ("raw", "Potential (V)"): potential,
            ("raw", "Current (A)"): current,
        }
    )

    obj = blank_echem_factory(ecat_module.cv)
    obj.manual_init(name, data, options={})
    return obj


def test_cv_manual_init_builds_basic_in_memory_object(cv_factory):
    obj = cv_factory()

    assert obj.type == "Cyclic Voltammetry"
    assert obj.scan_rate == pytest.approx(0.05)
    assert obj.segments == 2
    assert list(obj.data.columns) == ["Potential", "Current"]
    assert obj.units == {"Potential": "V", "Current": "A"}
    assert obj.data.attrs["units"] == obj.units


def test_cv_trim_returns_copy_with_expand_window_default(cv_factory):
    potential = np.array([0.0, -0.4, -0.8, -0.4, 0.0, 0.4, 0.8, 0.4, 0.0])
    current = np.arange(len(potential), dtype=float)
    obj = cv_factory(potential=potential, current=current)

    trimmed = obj.trim({"potential window": [-0.2, 0.2]})

    assert trimmed is not obj
    assert len(obj.data) == len(potential)
    assert len(trimmed.data) == len(potential)
    assert trimmed.trim_metadata["mode"] == "expand"
    assert "trim_mode" not in trimmed.trim_metadata
    assert "window_mode" not in trimmed.trim_metadata
    assert trimmed.trim_metadata["window_expanded"] is True
    assert trimmed.trim_metadata["window_break_count"] == 2
    np.testing.assert_allclose(trimmed.x().to_numpy(dtype=float), potential)


def test_cv_trim_pointwise_mode_trims_and_updates_metadata(cv_factory):
    potential = np.array([0.0, -0.4, -0.8, -0.4, 0.0, 0.4, 0.8, 0.4, 0.0])
    current = np.arange(len(potential), dtype=float)
    obj = cv_factory(potential=potential, current=current)

    trimmed = obj.trim({"potential window": [-0.2, 0.2], "mode": "pointwise"})

    np.testing.assert_allclose(trimmed.x().to_numpy(dtype=float), [0.0, 0.0, 0.0])
    np.testing.assert_allclose(trimmed.y().to_numpy(dtype=float), [0.0, 4.0, 8.0])
    assert trimmed.trim_metadata["mode"] == "pointwise"
    assert "trim_mode" not in trimmed.trim_metadata
    assert "window_mode" not in trimmed.trim_metadata
    assert trimmed.trim_metadata["window_expanded"] is False
    assert trimmed.init_E == pytest.approx(0.0)
    assert trimmed.final_E == pytest.approx(0.0)
    assert trimmed.min_E == pytest.approx(0.0)
    assert trimmed.max_E == pytest.approx(0.0)


def test_cv_filter_returns_copy_and_records_resolved_metadata(cv_factory):
    potential = np.linspace(-0.5, 0.5, 31)
    clean = np.sin(4 * potential)
    noisy = clean.copy()
    noisy[::2] += 0.2
    obj = cv_factory(potential=potential, current=noisy)

    filtered = obj.filter(
        {
            "method": "savgol",
            "window": 7,
            "polyorder": 2,
            "print": False,
        }
    )

    assert filtered is not obj
    np.testing.assert_allclose(obj.data["Current"], noisy)
    assert not np.allclose(filtered.data["Current"], noisy)
    assert filtered.filter_metadata == {
        "method": "savgol",
        "column": "Current",
        "window": 7,
        "polyorder": 2,
    }
    assert filtered.processing_history[-1]["operation"] == "filter"
    assert filtered.info()["filter"] == "savgol (window=7, polyorder=2)"


@pytest.mark.parametrize(
    ("method", "method_options"),
    [
        ("gaussian", {"sigma": 1.0}),
        ("median", {"size": 3}),
        ("butterworth", {"cutoff": 0.2, "order": 2}),
        ("moving average", {"window": 5}),
    ],
)
def test_cv_filter_supported_scipy_methods_preserve_shape(
    cv_factory,
    method,
    method_options,
):
    potential = np.linspace(-0.5, 0.5, 41)
    current = np.sin(5 * potential) + 0.05 * np.cos(37 * potential)
    obj = cv_factory(potential=potential, current=current)

    filtered = obj.filter(
        {"method": method, "print": False, **method_options}
    )

    assert len(filtered.data) == len(obj.data)
    assert np.isfinite(filtered.data["Current"]).all()
    assert filtered.filter_metadata["method"] == method


def test_cv_filter_inplace_returns_same_object(cv_factory):
    obj = cv_factory()
    original = obj.data["Current"].copy()

    result = obj.filter(
        {"method": "moving average", "window": 3, "inplace": True, "print": False}
    )

    assert result is obj
    assert not np.allclose(result.data["Current"], original)


def test_cv_filter_plot_uses_raw_and_filtered_overlay(cv_factory):
    obj = cv_factory()

    filtered = obj.filter(
        {
            "method": "moving average",
            "window": 3,
            "plot": True,
            "print": False,
            "legend": True,
        }
    )

    ax = plt.gca()
    assert filtered is not obj
    assert len(ax.lines) == 2
    assert [line.get_label() for line in ax.lines] == ["Raw", "Filtered"]
    plt.close(ax.figure)


def test_cv_filter_rejects_unknown_method(cv_factory):
    obj = cv_factory()

    with pytest.raises(Exception, match="method.*savgol.*gaussian.*median.*butterworth.*moving average"):
        obj.filter({"method": "magic", "print": False})


def test_cv_trim_accepts_keyword_options(cv_factory):
    potential = np.array([0.0, -0.4, -0.8, -0.4, 0.0])
    current = np.arange(len(potential), dtype=float)
    obj = cv_factory(potential=potential, current=current)

    trimmed = obj.trim([-0.2, 0.2], mode="pointwise")

    np.testing.assert_allclose(trimmed.x().to_numpy(dtype=float), [0.0, 0.0])


def test_public_trim_trims_flat_and_grouped_cv_lists(ecat_module, cv_factory):
    cvs = [
        cv_factory(potential=[-0.4, -0.1, 0.1, 0.4], current=[0, 1, 2, 3]),
        cv_factory(potential=[-0.5, -0.2, 0.0, 0.2, 0.5], current=[0, 1, 2, 3, 4]),
    ]

    trimmed = ecat_module.trim(cvs, [-0.25, 0.25], mode="pointwise")

    assert len(trimmed) == 2
    assert trimmed[0] is not cvs[0]
    np.testing.assert_allclose(trimmed[0].x().to_numpy(dtype=float), [-0.1, 0.1])
    np.testing.assert_allclose(trimmed[1].x().to_numpy(dtype=float), [-0.2, 0.0, 0.2])

    grouped = ecat_module.trim(
        [[cvs[0]], [cvs[1]]],
        {"potential window": [-0.15, 0.15], "mode": "pointwise"},
    )

    assert len(grouped) == 2
    assert len(grouped[0]) == 1
    np.testing.assert_allclose(grouped[0][0].x().to_numpy(dtype=float), [-0.1, 0.1])
    np.testing.assert_allclose(grouped[1][0].x().to_numpy(dtype=float), [0.0])


def test_cv_trim_rejects_removed_window_mode_option(ecat_module, cv_factory):
    potential = np.array([0.0, -0.4, -0.8, -0.4, 0.0])
    current = np.arange(len(potential), dtype=float)
    obj = cv_factory(potential=potential, current=current)

    with pytest.raises(ecat_module.OptionError, match="Unknown option 'window_mode'"):
        obj.trim([-0.2, 0.2], window_mode="allow")


def test_cv_trim_rejects_removed_allow_mode_value(ecat_module, cv_factory):
    obj = cv_factory()

    with pytest.raises(ecat_module.OptionError, match="'mode' must be"):
        obj.trim([-0.2, 0.2], mode="allow")


def test_cv_trim_accepts_trim_options_dataclass(ecat_module, cv_factory):
    potential = np.array([0.0, -0.4, -0.8, -0.4, 0.0])
    current = np.arange(len(potential), dtype=float)
    obj = cv_factory(potential=potential, current=current)
    options = ecat_module.TrimOptions.from_options(
        {"potential window": [-0.2, 0.2], "mode": "pointwise"}
    )

    trimmed = obj.trim(options)

    np.testing.assert_allclose(trimmed.x().to_numpy(dtype=float), [0.0, 0.0])


def test_cv_trim_rejects_unknown_option_with_suggestion(ecat_module, cv_factory):
    obj = cv_factory()

    with pytest.raises(ecat_module.OptionError, match="potential window"):
        obj.trim({"potential windo": [-0.2, 0.2]})


def test_trim_options_validate_potential_window_shape(ecat_module):
    with pytest.raises(ecat_module.OptionError, match="exactly two"):
        ecat_module.TrimOptions.from_options({"potential window": [-0.2, 0.0, 0.2]})


def test_cv_peak_analysis_with_synthetic_trace(cv_factory):
    obj = cv_factory()
    peak_options = {
        "plot": False,
        "print": False,
        "noise window": 5,
        "noise polyorder": 2,
        "peak prominence": 1e-7,
    }
    current_options = {
        **peak_options,
        "tangent range": [0.05, 0.3],
        "percent threshold": 100,
    }

    peak_result = obj.peak_potential(peak_options)
    current_result = obj.peak_current(current_options)
    half_peak_result = obj.half_peak_potential(current_options)

    assert peak_result["Ep"] == pytest.approx(0.2, abs=1e-12)
    assert peak_result["index"] == 9
    assert np.isfinite(current_result["ip"])
    assert len(current_result["tangent line"]) == 2
    assert current_result["tangent start"] >= 0
    assert half_peak_result["Ep/2"] == pytest.approx(0.1, abs=1e-12)


def test_cv_peak_analysis_accepts_guess_potential_shorthand(cv_factory):
    obj = cv_factory()
    peak_options = {
        "plot": False,
        "print": False,
        "noise window": 5,
        "noise polyorder": 2,
        "peak prominence": 1e-7,
    }
    current_options = {
        **peak_options,
        "tangent range": [0.05, 0.3],
        "percent threshold": 100,
    }

    expected_peak = obj.peak_potential({**peak_options, "guess potential": 0.2})
    shorthand_peak = obj.peak_potential(0.2, peak_options)
    shorthand_current = obj.peak_current(0.2, current_options)
    shorthand_half_peak = obj.half_peak_potential(0.2, current_options)
    shorthand_peak_info = obj.peak_info(0.2, current_options)

    assert shorthand_peak["Ep"] == expected_peak["Ep"]
    assert shorthand_current["Ep"] == expected_peak["Ep"]
    assert shorthand_half_peak["Ep"] == expected_peak["Ep"]
    assert shorthand_peak_info["Ep"] == expected_peak["Ep"]
    assert shorthand_half_peak["Δ(Ep - Ep/2)"] == pytest.approx(0.1, abs=1e-12)


def test_peak_potential_peak_kind_filters_nearest_opposite_extremum(cv_factory):
    potential = np.linspace(-0.2, 0.2, 161)
    current = (
        1.2e-6 * np.exp(-((potential + 0.08) / 0.018) ** 2)
        - 1.0e-6 * np.exp(-((potential - 0.01) / 0.018) ** 2)
    )
    obj = cv_factory(potential=potential, current=current)
    options = {
        "plot": False,
        "print": False,
        "guess potential": 0.0,
        "noise window": 7,
        "noise polyorder": 2,
        "peak prominence": 1e-8,
    }

    min_result = obj.peak_potential({**options, "peak kind": "min"})
    max_result = obj.peak_potential({**options, "peak kind": "max"})

    assert min_result["Ep"] == pytest.approx(0.01, abs=0.005)
    assert min_result["extremum kind"] == "min"
    assert max_result["Ep"] == pytest.approx(-0.08, abs=0.005)
    assert max_result["extremum kind"] == "max"


def test_peak_potential_default_peak_kind_considers_both_extrema(
    blank_echem_factory,
    ecat_module,
):
    obj = _make_dual_peak_cv(blank_echem_factory, ecat_module)
    base_options = {
        "plot": False,
        "print": False,
        "noise window": 7,
        "noise polyorder": 2,
        "peak prominence": 5e-7,
    }

    increasing_segment = obj.peak_potential(
        {**base_options, "segment": 1, "guess potential": -0.11}
    )
    decreasing_segment = obj.peak_potential(
        {**base_options, "segment": 2, "guess potential": 0.12}
    )

    assert increasing_segment["Ep"] == pytest.approx(-0.1)
    assert increasing_segment["extremum kind"] == "min"
    assert decreasing_segment["Ep"] == pytest.approx(0.125)
    assert decreasing_segment["extremum kind"] == "max"


def test_peak_potential_infer_peak_kind_uses_segment_current_change(
    blank_echem_factory,
    ecat_module,
):
    obj = _make_dual_peak_cv(blank_echem_factory, ecat_module)
    base_options = {
        "plot": False,
        "print": False,
        "peak kind": "infer",
        "noise window": 7,
        "noise polyorder": 2,
        "peak prominence": 5e-7,
    }

    increasing_current_segment = obj.peak_potential(
        {**base_options, "segment": 1, "guess potential": -0.11}
    )
    decreasing_current_segment = obj.peak_potential(
        {**base_options, "segment": 2, "guess potential": 0.12}
    )

    assert increasing_current_segment["Ep"] == pytest.approx(0.125)
    assert increasing_current_segment["extremum kind"] == "max"
    assert decreasing_current_segment["Ep"] == pytest.approx(-0.1)
    assert decreasing_current_segment["extremum kind"] == "min"


def test_peak_potential_guess_uses_local_auto_prominence(cv_factory):
    potential = np.linspace(-1.0, 1.0, 801)
    current = -1e-7 * np.exp(-((potential + 0.05) / 0.025) ** 2)
    distant_region = potential > 0.35
    current[distant_region] += 8e-7 * np.sin(120 * (potential[distant_region] - 0.35))
    obj = cv_factory(potential=potential, current=current)

    result = obj.peak_potential(
        -0.05,
        {
            "plot": False,
            "print": False,
            "segment": 1,
            "peak kind": "min",
            "noise window": 5,
            "noise polyorder": 2,
        },
    )

    assert result["Ep"] == pytest.approx(-0.05, abs=0.01)
    assert result["extremum kind"] == "min"


def test_single_cv_analysis_result_preserves_dict_access_and_adds_table(cv_factory, ecat_module):
    obj = cv_factory()
    result = obj.peak_potential(
        {
            "plot": False,
            "print": False,
            "noise window": 5,
            "noise polyorder": 2,
            "peak prominence": 1e-7,
        }
    )

    assert isinstance(result, ecat_module.CVAnalysisResult)
    assert result["Ep"] == pytest.approx(0.2)
    assert result.primary == pytest.approx(0.2)
    assert list(result.table.columns) == ["Metric", "Value"]
    assert result.table.to_dict("records")[0] == {"Metric": "Ep", "Value": "0.2 V"}
    assert "Segment" not in result.table.columns


def test_single_cv_analysis_result_scales_display_current_and_show_plain(
    cv_factory,
    ecat_module,
    capsys,
):
    obj = cv_factory()
    result = obj.peak_current(
        {
            "plot": False,
            "print": False,
            "noise window": 5,
            "noise polyorder": 2,
            "peak prominence": 1e-7,
            "tangent range": [0.05, 0.3],
            "percent threshold": 100,
        }
    )

    assert isinstance(result, ecat_module.CVAnalysisResult)
    assert result.primary == result["ip"]
    assert abs(result.primary) < 1e-3
    rows = result.table.set_index("Metric")["Value"].to_dict()
    assert rows["ip"].endswith("μA")
    assert rows["Ep"].endswith("V")

    result.show({"pretty print": False})
    out = capsys.readouterr().out
    assert "Peak Current:" in out
    assert "Metric" in out
    assert "Value" in out
    assert "μA" in out


def test_cv_analysis_result_pretty_show_formats_metric_labels_as_html(
    cv_factory,
    ecat_module,
    monkeypatch,
):
    import ecat.objects as ecat_objects

    obj = cv_factory()
    result = obj.peak_current(
        {
            "plot": False,
            "print": False,
            "noise window": 5,
            "noise polyorder": 2,
            "peak prominence": 1e-7,
            "tangent range": [0.05, 0.3],
            "percent threshold": 100,
        }
    )

    displayed = {}

    def capture_display(styler):
        displayed["html"] = styler.to_html()

    monkeypatch.setattr(ecat_objects, "display", capture_display)

    result.show()

    assert "Ep" in result
    assert "ip" in result
    assert result.table["Metric"].tolist() == ["Ep", "ip"]
    assert "E<sub>p</sub>" in displayed["html"]
    assert "i<sub>p</sub>" in displayed["html"]


def test_cv_analysis_result_pretty_show_formats_half_peak_label_as_single_subscript(
    cv_factory,
    monkeypatch,
):
    import ecat.objects as ecat_objects

    obj = cv_factory()
    result = obj.half_peak_potential(
        {
            "plot": False,
            "print": False,
            "noise window": 5,
            "noise polyorder": 2,
            "peak prominence": 1e-7,
            "tangent range": [0.05, 0.3],
            "percent threshold": 100,
        }
    )

    displayed = {}

    def capture_display(styler):
        displayed["html"] = styler.to_html()

    monkeypatch.setattr(ecat_objects, "display", capture_display)

    result.show()

    assert "Ep/2" in result
    assert "Δ(Ep - Ep/2)" in result
    assert "E<sub>p/2</sub>" in displayed["html"]
    assert "Δ(E<sub>p</sub> - E<sub>p/2</sub>)" in displayed["html"]
    assert "E<sub>p</sub>/2" not in displayed["html"]


def test_wave_info_analysis_result_includes_segment_when_multiple_segments(
    blank_echem_factory,
    ecat_module,
):
    obj = _make_dual_peak_cv(blank_echem_factory, ecat_module)
    result = obj.wave_info(
        {
            "plot": False,
            "print": False,
            "noise window": 7,
            "noise polyorder": 2,
            "peak prominence": 5e-7,
            "guess potential": [0.12, -0.11],
            "tangent range": [0.05, 0.3],
            "percent threshold": 100,
        }
    )

    assert isinstance(result, ecat_module.CVAnalysisResult)
    assert result.primary == result["E(1/2)"]
    assert "Segment" in result.table.columns
    assert set(result.table["Segment"].dropna().astype(str)) >= {"1", "2"}
    assert result["P1 Ep"] == pytest.approx(0.125)
    assert result["P2 Ep"] == pytest.approx(-0.1)


def test_half_peak_potential_plots_peak_and_tangent_annotations(cv_factory):
    obj = cv_factory()

    obj.half_peak_potential(
        {
            "plot": True,
            "print": False,
            "noise window": 5,
            "noise polyorder": 2,
            "peak prominence": 1e-7,
            "tangent range": [0.05, 0.3],
            "percent threshold": 100,
        }
    )

    ax = plt.gca()
    assert len(ax.lines) == 2
    assert len(ax.collections) >= 3
    assert ax.get_legend() is None


def test_peak_current_manual_tangent_potential_regression(ecat_module, blank_echem_factory):
    potential = [round(-0.30 + i * 0.02, 2) for i in range(31)]
    potential += [round(0.30 - i * 0.02, 2) for i in range(1, 31)]

    current = []
    for x in potential:
        baseline = 1.5e-6 * x + 2e-7
        peak = 4.0e-6 * np.exp(-((x - 0.12) / 0.05) ** 2)
        current.append(baseline + peak)

    data = pd.DataFrame(
        {
            ("raw", "Potential (V)"): potential,
            ("raw", "Current (A)"): current,
        }
    )

    obj = blank_echem_factory(ecat_module.cv)
    obj.manual_init("100mVs_sample_CO2_MeCN_10mM_Fc_run01", data, options={})

    current_result = obj.peak_current(
        {
            "plot": False,
            "print": False,
            "noise window": 7,
            "noise polyorder": 2,
            "peak prominence": 5e-7,
            "tangent range": [0.06, 0.18],
            "tangent potential": -0.18,
        }
    )

    assert current_result["ip"] == pytest.approx(4.0e-6, rel=1e-3)
    assert current_result["tangent line"][0] == pytest.approx(1.5e-6, rel=1e-6)
    assert current_result["tangent line"][1] == pytest.approx(2.0e-7, rel=1e-6)
    assert current_result["tangent start"] == 4


def test_peak_current_auto_tangent_handles_noisy_synthetic_wave(
    blank_echem_factory,
    ecat_module,
):
    potential = np.linspace(-0.45, 0.35, 161)
    noise = (
        2.0e-8 * np.sin(np.linspace(0, 24 * np.pi, potential.size))
        + 1.0e-8 * np.cos(np.linspace(0, 13 * np.pi, potential.size))
    )
    baseline = 1.2e-6 * potential + 0.2e-6
    peak = 3.2e-6 * np.exp(-((potential - 0.08) / 0.055) ** 2)
    obj = _make_synthetic_cv(
        blank_echem_factory,
        ecat_module,
        potential,
        baseline + peak + noise,
        "100mVs_noisy_auto_tangent",
    )

    current_result = obj.peak_current(
        {
            "plot": False,
            "print": False,
            "noise window": 11,
            "noise polyorder": 2,
            "peak prominence": 1e-7,
            "guess potential": 0.08,
            "tangent range": "auto",
            "percent threshold": 50,
        }
    )

    assert current_result["ip"] == pytest.approx(3.2e-6, rel=0.04)
    assert current_result["tangent line"][0] == pytest.approx(1.2e-6, rel=0.08)
    assert current_result["tangent line"][1] == pytest.approx(0.2e-6, abs=2.5e-8)
    assert current_result["tangent start"] >= 0


def test_peak_current_auto_tangent_handles_shallow_peak(
    blank_echem_factory,
    ecat_module,
):
    potential = np.linspace(-0.45, 0.35, 161)
    baseline = -0.8e-6 * potential + 0.1e-6
    peak = 0.7e-6 * np.exp(-((potential + 0.02) / 0.07) ** 2)
    ripple = 1.0e-8 * np.sin(np.linspace(0, 10 * np.pi, potential.size))
    obj = _make_synthetic_cv(
        blank_echem_factory,
        ecat_module,
        potential,
        baseline + peak + ripple,
        "100mVs_shallow_auto_tangent",
    )

    current_result = obj.peak_current(
        {
            "plot": False,
            "print": False,
            "noise window": 11,
            "noise polyorder": 2,
            "peak prominence": 8e-8,
            "guess potential": -0.02,
            "tangent range": "auto",
            "percent threshold": 50,
        }
    )

    assert current_result["ip"] == pytest.approx(0.7e-6, rel=0.20)
    assert current_result["tangent line"][0] == pytest.approx(-0.8e-6, rel=0.50)
    assert current_result["tangent line"][1] == pytest.approx(0.1e-6, abs=1.2e-7)
    assert current_result["tangent start"] >= 0


def test_peak_current_auto_tangent_handles_irreversible_like_broad_wave(
    blank_echem_factory,
    ecat_module,
):
    potential = np.linspace(-0.7, 0.2, 181)
    sigmoid_background = -1.5e-6 / (1 + np.exp((potential + 0.23) / 0.04))
    sloped_background = 0.25e-6 * potential + 0.05e-6
    broad_peak = 1.1e-6 * np.exp(-((potential + 0.34) / 0.09) ** 2)
    obj = _make_synthetic_cv(
        blank_echem_factory,
        ecat_module,
        potential,
        sigmoid_background + sloped_background + broad_peak,
        "100mVs_irreversible_like_auto_tangent",
    )

    current_result = obj.peak_current(
        {
            "plot": False,
            "print": False,
            "noise window": 11,
            "noise polyorder": 2,
            "peak prominence": 2e-8,
            "guess potential": -0.34,
            "tangent range": "auto",
            "percent threshold": 50,
        }
    )

    assert current_result["ip"] == pytest.approx(1.1e-6, rel=0.15)
    assert np.isfinite(current_result["tangent line"][0])
    assert np.isfinite(current_result["tangent line"][1])
    assert current_result["tangent start"] >= 0


def test_peak_current_auto_tangent_uses_selected_segment_not_neighboring_segment(
    blank_echem_factory,
    ecat_module,
):
    segment1_potential = np.linspace(-0.4, 0.3, 121)
    segment2_potential = np.linspace(0.3, -0.4, 121)[1:]
    segment1_current = (
        0.8e-6 * segment1_potential
        + 0.1e-6
        + 2.5e-6 * np.exp(-((segment1_potential - 0.05) / 0.05) ** 2)
    )
    segment2_current = (
        8.0e-6
        - 1.0e-6 * segment2_potential
        - 2.0e-6 * np.exp(-((segment2_potential + 0.08) / 0.05) ** 2)
    )
    obj = _make_synthetic_cv(
        blank_echem_factory,
        ecat_module,
        np.concatenate([segment1_potential, segment2_potential]),
        np.concatenate([segment1_current, segment2_current]),
        "100mVs_two_segment_auto_tangent",
    )

    current_result = obj.peak_current(
        {
            "plot": False,
            "print": False,
            "noise window": 11,
            "noise polyorder": 2,
            "peak prominence": 1e-7,
            "segment": 1,
            "guess potential": 0.05,
            "tangent range": "auto",
            "percent threshold": 50,
        }
    )

    assert current_result["ip"] == pytest.approx(2.5e-6, rel=0.02)
    assert current_result["tangent line"][0] == pytest.approx(0.8e-6, rel=0.02)
    assert current_result["tangent line"][1] == pytest.approx(0.1e-6, abs=1e-12)
    assert current_result["tangent start"] < len(segment1_potential)


def test_peak_analysis_public_methods_accept_option_dataclasses(cv_factory, ecat_module):
    obj = cv_factory()
    peak_options = ecat_module.PeakPotentialOptions.from_options(
        {
            "plot": False,
            "print": False,
            "noise window": 5,
            "noise polyorder": 2,
            "peak prominence": 1e-7,
        }
    )
    current_options = ecat_module.PeakCurrentOptions.from_options(
        {
            "plot": False,
            "print": False,
            "noise window": 5,
            "noise polyorder": 2,
            "peak prominence": 1e-7,
            "tangent range": [0.05, 0.3],
            "percent threshold": 100,
        }
    )

    peak_result = obj.peak_potential(peak_options)
    current_result = obj.peak_current(current_options)

    assert peak_result["Ep"] == pytest.approx(0.2)
    assert peak_result["index"] == 9
    assert np.isfinite(current_result["ip"])
    assert len(current_result["tangent line"]) == 2
    assert current_result["tangent start"] >= 0


def test_peak_potential_public_method_rejects_peak_current_only_option(cv_factory, ecat_module):
    obj = cv_factory()

    with pytest.raises(ecat_module.OptionError, match="tangent range"):
        obj.peak_potential({"plot": False, "print": False, "tangent range": "auto"})


def test_peak_current_rejects_unknown_peak_fallback(cv_factory, ecat_module):
    obj = cv_factory()

    with pytest.raises(ecat_module.OptionError, match="peak fallback"):
        obj.peak_current({"plot": False, "print": False, "peak fallback": "banana"})


def test_analysis_segment_data_selects_requested_segments(blank_echem_factory, ecat_module):
    obj = _make_dual_peak_cv(blank_echem_factory, ecat_module)

    x1, y1 = obj.analysis_segment_data({"segment": 1})
    x2, y2 = obj.analysis_segment_data({"segment": 2})
    x_both, y_both = obj.analysis_segment_data({"segments": [1, 2]})

    assert len(x1) == len(y1) == 24
    assert len(x2) == len(y2) == 25
    assert x1[0] == pytest.approx(-0.3)
    assert x1[-1] == pytest.approx(0.275)
    assert x2[0] == pytest.approx(0.3)
    assert x2[-1] == pytest.approx(-0.3)
    assert np.concatenate([x1, x2]).tolist() == pytest.approx(x_both.tolist())
    assert np.concatenate([y1, y2]).tolist() == pytest.approx(y_both.tolist())


def test_analysis_segment_data_returns_empty_arrays_for_out_of_range_segment(
    blank_echem_factory,
    ecat_module,
):
    obj = _make_dual_peak_cv(blank_echem_factory, ecat_module)

    x, y = obj.analysis_segment_data({"segment": 3})

    assert x.size == 0
    assert y.size == 0


def test_analysis_segment_data_out_of_range_segment_is_console_encoding_safe(
    blank_echem_factory,
    ecat_module,
    monkeypatch,
):
    obj = _make_dual_peak_cv(blank_echem_factory, ecat_module)
    cp1252_stdout = TextIOWrapper(BytesIO(), encoding="cp1252")
    monkeypatch.setattr(sys, "stdout", cp1252_stdout)

    x, y = obj.analysis_segment_data({"segment": 3})

    assert x.size == 0
    assert y.size == 0


def test_peak_potential_supports_automatic_guess_and_exact_options(
    blank_echem_factory,
    ecat_module,
):
    obj = _make_dual_peak_cv(blank_echem_factory, ecat_module)
    base_options = {
        "plot": False,
        "print": False,
        "noise window": 7,
        "noise polyorder": 2,
        "peak prominence": 5e-7,
    }

    automatic_result = obj.peak_potential(base_options)
    guessed_result = obj.peak_potential(
        {**base_options, "guess potential": -0.11}
    )
    exact_result = obj.peak_potential(
        {**base_options, "exact potential": 0.0}
    )

    assert automatic_result["Ep"] == pytest.approx(0.125)
    assert automatic_result["index"] == 17
    assert guessed_result["Ep"] == pytest.approx(-0.1)
    assert guessed_result["index"] == 8
    assert exact_result["Ep"] == pytest.approx(0.0)
    assert exact_result["index"] == 12


def test_peak_current_and_half_peak_potential_follow_selected_segment_tangent(
    blank_echem_factory,
    ecat_module,
):
    obj = _make_dual_peak_cv(blank_echem_factory, ecat_module)
    options = {
        "plot": False,
        "print": False,
        "noise window": 7,
        "noise polyorder": 2,
        "peak prominence": 5e-7,
        "segment": 1,
        "tangent range": [0.05, 0.3],
        "percent threshold": 100,
    }

    current_result = obj.peak_current(options)
    half_peak_result = obj.half_peak_potential(options)

    assert np.isfinite(current_result["ip"])
    assert len(current_result["tangent line"]) == 2
    assert current_result["tangent start"] >= 0
    assert np.isfinite(half_peak_result["Ep/2"])
    assert np.isfinite(half_peak_result["Δ(Ep - Ep/2)"])


def test_peak_current_defaults_to_highest_current_fallback_when_no_peak(
    blank_echem_factory,
    ecat_module,
):
    potential = np.linspace(0.0, -1.0, 61)
    baseline = 0.2e-6 * potential
    catalytic_tail = -2.0e-6 / (1.0 + np.exp((potential + 0.82) / 0.04))
    current = baseline + catalytic_tail
    obj = _make_synthetic_cv(
        blank_echem_factory,
        ecat_module,
        potential,
        current,
        "100mVs_monotonic_tail",
    )

    with pytest.raises(ValueError, match="could not locate any extrema"):
        obj.peak_potential({"plot": False, "print": False, "segment": 1})

    result = obj.peak_current(
        {
            "plot": False,
            "print": False,
            "segment": 1,
            "tangent range": [0.05, 0.3],
            "percent threshold": 100,
        }
    )

    expected_idx = int(np.argmax(np.abs(current)))
    assert result["Ep index"] == expected_idx
    assert result["Ep"] == pytest.approx(potential[expected_idx])
    assert result["peak source"] == "highest current fallback"
    assert result.diagnostics["peak source"] == "highest current fallback"
    assert "peak potential error" in result.diagnostics
    assert np.isfinite(result["ip"])


def test_peak_current_can_disable_no_peak_fallback(
    blank_echem_factory,
    ecat_module,
):
    potential = np.linspace(0.0, -1.0, 61)
    current = -np.linspace(0.0, 2.0e-6, 61)
    obj = _make_synthetic_cv(
        blank_echem_factory,
        ecat_module,
        potential,
        current,
        "100mVs_monotonic_no_fallback",
    )

    with pytest.raises(ValueError, match="could not locate any extrema"):
        obj.peak_current(
            {
                "plot": False,
                "print": False,
                "segment": 1,
                "peak fallback": None,
                "tangent range": [0.05, 0.3],
                "percent threshold": 100,
            }
        )


def test_peak_current_can_fallback_to_guess_potential(
    blank_echem_factory,
    ecat_module,
):
    potential = np.linspace(0.0, -1.0, 61)
    current = -np.linspace(0.0, 2.0e-6, 61)
    obj = _make_synthetic_cv(
        blank_echem_factory,
        ecat_module,
        potential,
        current,
        "100mVs_monotonic_guess_fallback",
    )

    result = obj.peak_current(
        {
            "plot": False,
            "print": False,
            "segment": 1,
            "guess potential": -0.75,
            "peak fallback": "guess potential",
            "tangent range": [0.05, 0.3],
            "percent threshold": 100,
        }
    )

    expected_idx = int(np.argmin(np.abs(potential + 0.75)))
    assert result["Ep index"] == expected_idx
    assert result["Ep"] == pytest.approx(potential[expected_idx])
    assert result["peak source"] == "guess potential fallback"
    assert result.diagnostics["peak source"] == "guess potential fallback"


def test_peak_potential_segment_selects_analysis_but_plots_full_cv(
    blank_echem_factory,
    ecat_module,
):
    obj = _make_dual_peak_cv(blank_echem_factory, ecat_module)

    peak_result = obj.peak_potential(
        {
            "plot": True,
            "print": False,
            "noise window": 7,
            "noise polyorder": 2,
            "peak prominence": 5e-7,
            "segment": 1,
            "guess potential": 0.12,
        }
    )

    ax = plt.gca()

    assert peak_result["Ep"] == pytest.approx(0.125)
    assert peak_result["index"] == 17
    assert len(ax.lines[0].get_xdata()) == len(obj.x())


def test_peak_potential_accepts_derivative_for_diagnostic_plot(cv_factory):
    obj = cv_factory()

    peak_result = obj.peak_potential(
        {
            "plot": True,
            "print": False,
            "exact potential": 0.2,
            "derivative": 1,
            "noise window": 5,
            "noise polyorder": 2,
        }
    )

    ax = plt.gca()

    assert peak_result["Ep"] == pytest.approx(0.2)
    assert peak_result["index"] == 9
    assert len(ax.lines) == 1
    assert len(ax.collections) == 0
    assert ax.lines[0].get_ydata()[9] != pytest.approx(obj.y().iloc[9])


def test_peak_potential_exact_potential_does_not_plot_peak_marker(
    blank_echem_factory,
    ecat_module,
):
    obj = _make_dual_peak_cv(blank_echem_factory, ecat_module)

    result = obj.peak_potential(
        {
            "plot": True,
            "print": False,
            "exact potential": 0.0,
            "noise window": 7,
            "noise polyorder": 2,
            "peak prominence": 5e-7,
        }
    )

    ax = plt.gca()

    assert result["Ep"] == pytest.approx(0.0)
    assert result["source"] == "exact potential"
    assert len(ax.lines[0].get_xdata()) == len(obj.x())
    assert len(ax.collections) == 0


def test_peak_potential_detected_peak_keeps_peak_marker(
    blank_echem_factory,
    ecat_module,
):
    obj = _make_dual_peak_cv(blank_echem_factory, ecat_module)

    result = obj.peak_potential(
        {
            "plot": True,
            "print": False,
            "guess potential": 0.12,
            "noise window": 7,
            "noise polyorder": 2,
            "peak prominence": 5e-7,
        }
    )

    ax = plt.gca()

    assert result["Ep"] == pytest.approx(0.125)
    assert result["source"] == "peak"
    assert len(ax.collections) == 1


def test_peak_current_segment_selects_analysis_but_plots_full_cv(
    blank_echem_factory,
    ecat_module,
):
    obj = _make_dual_peak_cv(blank_echem_factory, ecat_module)

    current_result = obj.peak_current(
        {
            "plot": True,
            "print": False,
            "noise window": 7,
            "noise polyorder": 2,
            "peak prominence": 5e-7,
            "segment": 1,
            "guess potential": 0.12,
            "tangent range": [0.05, 0.3],
            "percent threshold": 100,
        }
    )

    ax = plt.gca()

    assert np.isfinite(current_result["ip"])
    assert len(current_result["tangent line"]) == 2
    assert current_result["tangent start"] >= 0
    assert len(ax.lines[0].get_xdata()) == len(obj.x())


def test_peak_current_new_plot_keeps_peak_marker_and_tangent_on_one_axes(
    blank_echem_factory,
    ecat_module,
):
    obj = _make_dual_peak_cv(blank_echem_factory, ecat_module)
    plt.close("all")

    current_result = obj.peak_current(
        {
            "plot": True,
            "new plot": True,
            "print": False,
            "noise window": 7,
            "noise polyorder": 2,
            "peak prominence": 5e-7,
            "segment": 1,
            "guess potential": 0.12,
            "tangent range": [0.05, 0.3],
            "percent threshold": 100,
        }
    )

    assert len(plt.get_fignums()) == 1
    ax = current_result.axes
    assert ax is plt.gca()
    assert len(ax.lines) >= 2
    assert len(ax.collections) >= 2

    plt.close("all")


def test_peak_current_plot_cv_false_adds_diagnostics_without_redrawing_trace(
    blank_echem_factory,
    ecat_module,
):
    obj = _make_dual_peak_cv(blank_echem_factory, ecat_module)
    plt.close("all")

    ax = obj.plot({"print": False, "new plot": True, "legend": False, "title": False})
    initial_line_count = len(ax.lines)
    initial_collection_count = len(ax.collections)

    current_result = obj.peak_current(
        {
            "plot": True,
            "plot CV": False,
            "new plot": False,
            "print": False,
            "noise window": 7,
            "noise polyorder": 2,
            "peak prominence": 5e-7,
            "segment": 1,
            "guess potential": 0.12,
            "tangent range": [0.05, 0.3],
            "percent threshold": 100,
        }
    )

    assert current_result.axes is ax
    assert len(plt.get_fignums()) == 1
    assert len(ax.lines) == initial_line_count + 1
    assert len(ax.collections) >= initial_collection_count + 2

    plt.close("all")


@pytest.mark.parametrize("method_name", ["half_peak_potential", "peak_info", "half_wave_potential", "wave_info"])
def test_coupled_cv_analysis_new_plot_is_owned_by_top_level_call(
    method_name,
    blank_echem_factory,
    ecat_module,
):
    obj = _make_dual_peak_cv(blank_echem_factory, ecat_module)
    plt.close("all")

    options = {
        "plot": True,
        "new plot": True,
        "print": False,
        "print all": False,
        "noise window": 7,
        "noise polyorder": 2,
        "peak prominence": 5e-7,
        "tangent range": [0.05, 0.3],
        "percent threshold": 100,
    }
    if method_name in {"half_wave_potential", "wave_info"}:
        options["guess potential"] = [0.12, -0.11]
        options["segments"] = [1, 2]
    else:
        options["guess potential"] = 0.12
        options["segment"] = 1

    result = getattr(obj, method_name)(options)

    assert len(plt.get_fignums()) == 1
    assert result.axes is plt.gca()

    plt.close("all")


def test_reverse_scan_peak_analysis_works_with_explicit_segment_and_guess(
    blank_echem_factory,
    ecat_module,
):
    obj = _make_dual_peak_cv(blank_echem_factory, ecat_module)
    peak_options = {
        "plot": False,
        "print": False,
        "noise window": 7,
        "noise polyorder": 2,
        "peak prominence": 5e-7,
        "segment": 2,
        "guess potential": -0.11,
    }
    current_options = {
        **peak_options,
        "tangent range": [0.05, 0.3],
        "percent threshold": 100,
    }

    peak_result = obj.peak_potential(peak_options)
    current_result = obj.peak_current(current_options)
    half_peak_result = obj.half_peak_potential(current_options)

    assert peak_result["Ep"] == pytest.approx(-0.1)
    assert peak_result["index"] == 16
    assert np.isfinite(current_result["ip"])
    assert len(current_result["tangent line"]) == 2
    assert current_result["tangent start"] >= 0
    assert np.isfinite(half_peak_result["Ep/2"])
    assert np.isfinite(half_peak_result["Δ(Ep - Ep/2)"])


def test_half_wave_potential_preserves_return_shape_with_explicit_guesses(
    blank_echem_factory,
    ecat_module,
):
    obj = _make_dual_peak_cv(blank_echem_factory, ecat_module)

    half_wave_result = obj.half_wave_potential(
        {
            "plot": False,
            "print": False,
            "noise window": 7,
            "noise polyorder": 2,
            "peak prominence": 5e-7,
            "guess potential": [0.12, -0.11],
        }
    )

    assert half_wave_result["E(1/2)"] == pytest.approx(0.0125)
    assert half_wave_result["ΔE"] == pytest.approx(0.225)
    assert half_wave_result["peak 1"] == {
        "segment": 1,
        "Ep": pytest.approx(0.125),
        "Ep y": pytest.approx(4.347705076372728e-06),
    }
    assert half_wave_result["peak 2"] == {
        "segment": 2,
        "Ep": pytest.approx(-0.1),
        "Ep y": pytest.approx(-2.832368301821216e-06),
    }


def test_half_wave_potential_single_segment_guess_chooses_adjacent_segment_containing_guess(
    cv_factory,
):
    obj = cv_factory()
    obj.segments = 3
    segment_x = {
        1: np.array([-1.2, -0.9, 0.0]),
        2: np.array([1.0, -0.9, -1.2]),
        3: np.array([0.2, 0.6, 1.0]),
    }
    segment_y = {
        1: np.array([-2.0e-6, -4.0e-6, -1.0e-6]),
        2: np.array([1.0e-6, 3.0e-6, 0.0]),
        3: np.array([0.5e-6, 1.0e-6, 0.7e-6]),
    }
    peak_ep = {1: -0.95, 2: -0.85, 3: 0.6}
    calls = []

    def fake_analysis_segment_data(options=None):
        segment = (options or {}).get("segment")
        return segment_x[segment], segment_y[segment]

    def fake_peak_potential(options=None):
        segment = getattr(options, "segment", None)
        if segment is None and isinstance(options, dict):
            segment = options.get("segment")
        calls.append(segment)
        return {"Ep": peak_ep[segment], "index": 1}

    obj.analysis_segment_data = fake_analysis_segment_data
    obj.peak_potential = fake_peak_potential

    result = obj.half_wave_potential(
        {
            "plot": False,
            "print": False,
            "segment": 2,
            "guess potential": -0.9,
        }
    )

    assert calls == [2, 1]
    assert result["peak 1"]["segment"] == 2
    assert result["peak 2"]["segment"] == 1


def test_half_wave_potential_plots_single_trace_without_default_legend(
    blank_echem_factory,
    ecat_module,
):
    obj = _make_dual_peak_cv(blank_echem_factory, ecat_module)

    obj.half_wave_potential(
        {
            "plot": True,
            "print": False,
            "title": False,
            "noise window": 7,
            "noise polyorder": 2,
            "peak prominence": 5e-7,
            "guess potential": [0.12, -0.11],
        }
    )

    ax = plt.gca()
    assert len(ax.lines) == 1
    assert ax.get_legend() is None


def test_wave_info_routes_tangent_options_to_peak_current_only(
    blank_echem_factory,
    ecat_module,
):
    obj = _make_dual_peak_cv(blank_echem_factory, ecat_module)

    result = obj.wave_info(
        {
            "plot": False,
            "print": False,
            "noise window": 7,
            "noise polyorder": 2,
            "peak prominence": 5e-7,
            "guess potential": [0.12, -0.11],
            "tangent range": [0.05, 0.3],
            "percent threshold": 100,
        }
    )

    assert result["E(1/2)"] == pytest.approx(0.0125)
    assert np.isfinite(result["P1 ip"])
    assert np.isfinite(result["P2 ip"])


def test_current_at_potential_reports_each_segment_and_missing_values(
    blank_echem_factory,
    ecat_module,
):
    obj = _make_dual_peak_cv(blank_echem_factory, ecat_module)
    base_options = {
        "plot": False,
        "print": False,
        "noise window": 7,
        "noise polyorder": 2,
        "peak prominence": 5e-7,
    }

    all_segments = obj.current_at_potential(0.12, base_options)
    reverse_segment = obj.current_at_potential(
        -0.11,
        {**base_options, "segment": 2},
    )
    missing = obj.current_at_potential(0.8, base_options)

    assert all_segments == {
        1: (pytest.approx(0.125), pytest.approx(4.347705076372728e-06)),
        2: (pytest.approx(0.125), pytest.approx(4.347705076372728e-06)),
    }
    assert reverse_segment == {
        2: (pytest.approx(-0.1), pytest.approx(-2.832368301821216e-06))
    }
    assert missing == {1: None, 2: None}


def test_current_at_potential_can_plot_selected_segment(
    blank_echem_factory,
    ecat_module,
):
    obj = _make_dual_peak_cv(blank_echem_factory, ecat_module)

    result = obj.current_at_potential(
        0.12,
        {
            "segment": 1,
            "plot": True,
            "print": False,
            "noise window": 7,
            "noise polyorder": 2,
            "peak prominence": 5e-7,
        },
    )

    assert result == {
        1: (pytest.approx(0.125), pytest.approx(4.347705076372728e-06))
    }
    assert len(plt.gca().collections) == 1


def test_current_at_potential_plot_cv_false_adds_marker_without_redrawing_trace(
    blank_echem_factory,
    ecat_module,
):
    obj = _make_dual_peak_cv(blank_echem_factory, ecat_module)
    plt.close("all")

    ax = obj.plot({"print": False, "new plot": True, "legend": False, "title": False})
    initial_line_count = len(ax.lines)
    initial_collection_count = len(ax.collections)

    result = obj.current_at_potential(
        0.12,
        {
            "segment": 1,
            "plot": True,
            "plot CV": False,
            "print": False,
        },
    )

    assert result.axes is ax
    assert len(plt.get_fignums()) == 1
    assert len(ax.lines) == initial_line_count
    assert len(ax.collections) == initial_collection_count + 1

    plt.close("all")
    plt.close("all")


def test_scale_current_pretty_print_adds_scale(
    ecat_module,
    blank_echem_factory,
    monkeypatch,
):
    ref_cv = _make_dual_peak_cv(blank_echem_factory, ecat_module)
    sample_cv = _make_dual_peak_cv(blank_echem_factory, ecat_module)
    sample_cv.name = "100mVs_sample_run02"
    sample_cv.data.iloc[:, sample_cv.num_x_cols:] *= 0.5
    cvs = [ref_cv, sample_cv]

    displayed = {}

    def capture_display(df, options=None, **kwargs):
        displayed["df"] = df.copy()
        displayed["kwargs"] = kwargs

    monkeypatch.setattr(ecat_module, "display_object_table", capture_display)

    scaled = ecat_module.scale_current(
        cvs,
        {
            "print": True,
            "pretty print": True,
            "segment": 1,
            "guess potential": 0.15,
            "tangent range": [0.0, 0.25],
            "percent threshold": 100,
        },
    )

    table = displayed["df"]

    assert len(scaled) == 2
    assert displayed["kwargs"]["title"] == "Current Scaling Summary"
    assert "Scale Factor" in table.columns
    assert table["Scale Factor"].iloc[0] == pytest.approx(1.0)
    assert table["Scale Factor"].iloc[1] == pytest.approx(
        scaled[1].current_scale_factor
    )


def test_scale_current_plain_print_keeps_text_summary(
    ecat_module,
    blank_echem_factory,
    capsys,
):
    cvs = [
        _make_dual_peak_cv(blank_echem_factory, ecat_module),
        _make_dual_peak_cv(blank_echem_factory, ecat_module),
    ]
    cvs[1].name = "100mVs_sample_run02"
    cvs[1].data.iloc[:, cvs[1].num_x_cols:] *= 0.5

    ecat_module.scale_current(
        cvs,
        {
            "print": True,
            "pretty print": False,
            "segment": 1,
            "guess potential": 0.15,
            "tangent range": [0.0, 0.25],
            "percent threshold": 100,
        },
    )

    printed = capsys.readouterr().out

    assert "Current scaling summary:" in printed
    assert "scale =" in printed


def test_scale_current_accepts_option_dataclass(
    ecat_module,
    blank_echem_factory,
):
    cvs = [
        _make_dual_peak_cv(blank_echem_factory, ecat_module),
        _make_dual_peak_cv(blank_echem_factory, ecat_module),
    ]
    cvs[1].name = "100mVs_sample_run02"
    cvs[1].data.iloc[:, cvs[1].num_x_cols:] *= 0.5
    options = ecat_module.ScaleCurrentOptions.from_options(
        {
            "print": False,
            "segment": 1,
            "guess potential": 0.15,
            "tangent range": [0.0, 0.25],
            "percent threshold": 100,
        }
    )

    scaled = ecat_module.scale_current(cvs, options)

    assert len(scaled) == 2
    assert scaled[1].current_scale_factor == pytest.approx(2.0)


def test_scale_current_reference_mode_both_uses_two_segment_peak_currents(
    ecat_module,
    blank_echem_factory,
    monkeypatch,
):
    ref = _make_dual_peak_cv(blank_echem_factory, ecat_module)
    sample = _make_dual_peak_cv(blank_echem_factory, ecat_module)
    ref.name = "reference_cv"
    sample.name = "sample_cv"
    original_sample_current = sample.data["Current"].copy()
    calls = []

    def fake_peak_current(self, options=None):
        opt = options.to_options_dict() if hasattr(options, "to_options_dict") else dict(options or {})
        segment = opt.get("segment")
        calls.append((self.name, segment))
        values = {
            ("reference_cv", 1): 2.0,
            ("reference_cv", 2): -4.0,
            ("sample_cv", 1): 1.0,
            ("sample_cv", 2): -1.0,
        }
        return {"ip": values[(self.name, segment)], "tangent line": [0.0, 0.0], "tangent start": 0}

    monkeypatch.setattr(ecat_module.cv, "peak_current", fake_peak_current)

    scaled = ecat_module.scale_current(
        [ref, sample],
        {
            "reference mode": "both",
            "segments": [1, 2],
            "print": False,
        },
    )

    assert calls == [
        ("reference_cv", 1),
        ("reference_cv", 2),
        ("reference_cv", 1),
        ("reference_cv", 2),
        ("sample_cv", 1),
        ("sample_cv", 2),
    ]
    assert scaled[1].current_scale_factor == pytest.approx(3.0)
    assert scaled[1].data["Current"].to_numpy() == pytest.approx(
        (original_sample_current * 3.0).to_numpy()
    )


def test_scale_current_reference_mode_both_defaults_to_segment_and_next(
    ecat_module,
    blank_echem_factory,
    monkeypatch,
):
    ref = _make_dual_peak_cv(blank_echem_factory, ecat_module)
    sample = _make_dual_peak_cv(blank_echem_factory, ecat_module)
    ref.name = "reference_cv"
    sample.name = "sample_cv"
    calls = []

    def fake_peak_current(self, options=None):
        opt = options.to_options_dict() if hasattr(options, "to_options_dict") else dict(options or {})
        segment = opt.get("segment")
        calls.append((self.name, segment))
        values = {
            ("reference_cv", 2): -4.0,
            ("reference_cv", 3): 2.0,
            ("sample_cv", 2): -2.0,
            ("sample_cv", 3): 1.0,
        }
        return {"ip": values[(self.name, segment)], "tangent line": [0.0, 0.0], "tangent start": 0}

    monkeypatch.setattr(ecat_module.cv, "peak_current", fake_peak_current)

    scaled = ecat_module.scale_current(
        [ref, sample],
        {
            "reference mode": "both",
            "segment": 2,
            "print": False,
        },
    )

    assert calls == [
        ("reference_cv", 2),
        ("reference_cv", 3),
        ("reference_cv", 2),
        ("reference_cv", 3),
        ("sample_cv", 2),
        ("sample_cv", 3),
    ]
    assert scaled[1].current_scale_factor == pytest.approx(2.0)


def test_scale_current_plot_all_resolves_reference_current_without_child_plots(
    ecat_module,
    blank_echem_factory,
    monkeypatch,
):
    cvs = [
        _make_dual_peak_cv(blank_echem_factory, ecat_module),
        _make_dual_peak_cv(blank_echem_factory, ecat_module),
    ]
    cvs[1].name = "100mVs_sample_run02"
    calls = []

    def fake_peak_current(self, options=None):
        calls.append(options.to_options_dict() if hasattr(options, "to_options_dict") else dict(options or {}))
        return {"ip": 1.0, "tangent line": [0.0, 0.0], "tangent start": 0}

    monkeypatch.setattr(ecat_module.cv, "peak_current", fake_peak_current)

    scaled = ecat_module.scale_current(
        cvs,
        {
            "plot all": True,
            "print": False,
            "segment": 1,
            "guess potential": 0.15,
        },
    )

    assert len(scaled) == 2
    assert calls
    compute_calls = calls[:3]
    diagnostic_calls = calls[3:]
    assert compute_calls
    assert all(call["plot"] is False for call in compute_calls)
    assert all(call["plot all"] is False for call in compute_calls)
    assert diagnostic_calls
    assert all(call["plot"] is True for call in diagnostic_calls)
    assert all(call["plot all"] is True for call in diagnostic_calls)
    assert all(call["plot cv"] is False for call in diagnostic_calls)
    assert all(call["new plot"] is False for call in diagnostic_calls)


def test_scale_current_plot_all_adds_peak_current_diagnostics_to_scaled_overlay(
    ecat_module,
    blank_echem_factory,
):
    cvs = [
        _make_dual_peak_cv(blank_echem_factory, ecat_module),
        _make_dual_peak_cv(blank_echem_factory, ecat_module),
    ]
    cvs[1].name = "100mVs_sample_run02"
    cvs[1].data.iloc[:, cvs[1].num_x_cols:] *= 0.5
    plt.close("all")

    ecat_module.scale_current(
        cvs,
        {
            "plot all": True,
            "print": False,
            "segment": 1,
            "guess potential": 0.15,
            "tangent range": [0.0, 0.25],
            "percent threshold": 100,
        },
    )

    try:
        assert len(plt.get_fignums()) == 1
        ax = plt.gca()
        assert len(ax.lines) >= len(cvs) + len(cvs)
        vertical_segments = [
            np.asarray(segment)
            for collection in ax.collections
            for segment in getattr(collection, "get_segments", lambda: [])()
            if len(segment) == 2
            and np.isclose(segment[0][0], segment[1][0])
        ]
        assert vertical_segments
        blue_peak_markers = [
            np.asarray(collection.get_offsets(), dtype=float)
            for collection in ax.collections
            if len(collection.get_offsets())
            and len(collection.get_facecolors())
            and np.allclose(collection.get_facecolors()[0][:3], (0.12156863, 0.46666667, 0.70588235))
        ]
        assert blue_peak_markers
    finally:
        plt.close("all")


def test_scale_current_rejects_unknown_option_with_suggestion(
    ecat_module,
    blank_echem_factory,
):
    cvs = [_make_dual_peak_cv(blank_echem_factory, ecat_module)]

    with pytest.raises(ecat_module.OptionError, match="reference index"):
        ecat_module.scale_current(cvs, {"referenc index": 0})
