import numpy as np
import pandas as pd
import pytest
import matplotlib.pyplot as plt


def _synthetic_dpv(
    ecat_module,
    blank_echem_factory,
    *,
    name="DPV_synthetic_MeCN_CO2",
    centers=(-0.82, -0.93),
    amplitudes=(-2.4e-6, -1.6e-6),
    sigmas=(0.018, 0.022),
):
    x = np.linspace(-0.7, -1.05, 351)
    y = -2.0e-7 + 1.0e-7 * x
    for center, amplitude, sigma in zip(centers, amplitudes, sigmas):
        y = y + amplitude * np.exp(-((x - center) ** 2) / (2 * sigma ** 2))

    obj = blank_echem_factory(ecat_module.dpv)
    obj.name = name
    obj.type = "Differential Pulse Voltammetry"
    obj.data = pd.DataFrame({"Potential": x, "Current": y})
    obj.units = {"Potential": "V", "Current": "A"}
    obj.delta_x = abs(x[1] - x[0])
    obj.segments = 1
    obj.min_E = float(np.min(x))
    obj.max_E = float(np.max(x))
    return obj


def test_from_file_promotes_ch_dpv_export_to_dpv(ecat_module, repo_root):
    filepath = (
        repo_root
        / "tests"
        / "tmp_real_examples"
        / "DPV_MeCN_CO2_0.1MTBAPF6_3mMFc_1mMFe-tpyPY2Me_-0.7_to_-1.2V.txt"
    )

    obj = ecat_module.echem.from_file(str(filepath), {})

    assert type(obj).__name__ == "dpv"
    assert isinstance(obj, ecat_module.DPV)
    assert obj.type == "Differential Pulse Voltammetry"
    assert obj.software == "CH"
    assert list(obj.data.columns) == ["Potential", "Current"]
    assert obj.units == {"Potential": "V", "Current": "A"}
    assert obj.x().iloc[0] == pytest.approx(-0.701)
    assert obj.y().iloc[0] == pytest.approx(-4.549e-7)


def test_ch_dpv_parser_reads_header_fields_without_saving_reported_peak_fields(ecat_module, repo_root):
    filepath = (
        repo_root
        / "tests"
        / "tmp_real_examples"
        / "DPV_MeCN_CO2_0.1MTBAPF6_3mMFc_1mMFe-tpyPY2Me_-0.7_to_-1.2V.txt"
    )

    obj = ecat_module.echem.from_file(str(filepath), {})

    assert obj.init_E == pytest.approx(-0.7)
    assert obj.final_E == pytest.approx(-1.2)
    assert obj.incr_E == pytest.approx(0.001)
    assert obj.amplitude == pytest.approx(0.01)
    assert obj.pulse_width == pytest.approx(0.05)
    assert obj.sample_width == pytest.approx(0.0167)
    assert obj.pulse_period == pytest.approx(0.5)
    assert obj.quiet_time == pytest.approx(5)
    assert obj.sensitivity == pytest.approx(1e-5)
    assert obj.comp_R == pytest.approx(96.0)
    assert not hasattr(obj, "peak_current")
    assert not hasattr(obj, "peak_area")
    assert callable(obj.peak_potential)
    assert obj.delta_x == pytest.approx(0.001)
    assert obj.min_E == pytest.approx(-1.2)
    assert obj.max_E == pytest.approx(-0.701)


def test_dpv_stats_include_common_name_metadata_and_dpv_fields(ecat_module, repo_root):
    filepath = (
        repo_root
        / "tests"
        / "tmp_real_examples"
        / "DPV_MeCN_CO2_0.1MTBAPF6_3mMFc_1mMFe-tpyPY2Me_-0.7_to_-1.2V.txt"
    )

    obj = ecat_module.echem.from_file(str(filepath), {})
    stats = obj.stats()

    assert stats["solvent"] == "MeCN"
    assert stats["gas"] == "CO2"
    assert "Fc" in stats["compounds"]
    assert "peak potential (V)" not in stats
    assert "peak current (A)" not in stats
    assert "peak area (VA)" not in stats
    assert stats["amplitude"] == pytest.approx(0.01)
    assert stats["pulse width"] == pytest.approx(0.05)
    assert stats["sample width"] == pytest.approx(0.0167)
    assert stats["pulse period"] == pytest.approx(0.5)
    assert "amplitude (V)" not in stats
    assert "pulse period (s)" not in stats


def test_dpv_txt_stats_uses_cv_style_scan_window_without_verbose_fields(ecat_module, repo_root):
    filepath = (
        repo_root
        / "tests"
        / "tmp_real_examples"
        / "DPV_MeCN_CO2_0.1MTBAPF6_3mMFc_1mMFe-tpyPY2Me_-0.7_to_-1.2V.txt"
    )

    obj = ecat_module.echem.from_file(str(filepath), {})
    stats = obj.txt_stats({})

    assert stats["exp type"] == "DPV"
    assert stats["scan window"] == "[-1.2, -0.7]"
    assert stats["amplitude"] == "10 mV"
    assert stats["pulse width"] == "50 ms"
    assert stats["sample width"] == "16.7 ms"
    assert stats["pulse period"] == "500 ms"

    excluded_keys = {
        "start_x",
        "end_x",
        "min_x",
        "max_x",
        "delta_x",
        "init E",
        "final E",
        "increment E",
        "quiet time",
        "sensitivity",
        "comp R",
        "amplitude (V)",
        "pulse width (s)",
        "sample width (s)",
        "pulse period (s)",
        "peak potential (V)",
        "peak current (A)",
        "peak area (VA)",
    }
    assert excluded_keys.isdisjoint(stats)


def test_dpv_show_formats_unitless_pulse_metadata_with_units(ecat_module, repo_root):
    filepath = (
        repo_root
        / "tests"
        / "tmp_real_examples"
        / "DPV_MeCN_CO2_0.1MTBAPF6_3mMFc_1mMFe-tpyPY2Me_-0.7_to_-1.2V.txt"
    )

    obj = ecat_module.echem.from_file(str(filepath), {})

    table = ecat_module.show(obj, {"pretty print": False, "return": True})
    values = dict(zip(table["Metric"], table["Value"]))

    assert values["Amplitude"] == "10 mV"
    assert values["Pulse Width"] == "50 ms"
    assert values["Sample Width"] == "16.7 ms"
    assert values["Pulse Period"] == "500 ms"
    assert "Amplitude (v)" not in values
    assert "Pulse Width (s)" not in values


def test_dpv_plot_subtitle_uses_symbolic_pulse_context(ecat_module, repo_root):
    filepath = (
        repo_root
        / "tests"
        / "tmp_real_examples"
        / "DPV_MeCN_CO2_0.1MTBAPF6_3mMFc_1mMFe-tpyPY2Me_-0.7_to_-1.2V.txt"
    )

    obj = ecat_module.echem.from_file(str(filepath), {})

    ax = obj.plot({"new plot": True, "legend": False})
    fig = ax.figure
    subtitle = ax.get_title()

    assert r"$\Delta E_p$=10 mV" in subtitle
    assert r"$t_p$=50 ms" in subtitle
    assert r"$t_s$=16.7 ms" in subtitle
    assert "T=500 ms" in subtitle
    assert fig is not None


def test_dpv_peak_potential_finds_cathodic_peak(ecat_module, blank_echem_factory):
    obj = _synthetic_dpv(
        ecat_module,
        blank_echem_factory,
        centers=(-0.84,),
        amplitudes=(-2.0e-6,),
        sigmas=(0.02,),
    )

    peak_potential, peak_index = obj.peak_potential(
        {"guess potential": -0.83, "plot": False, "print": False}
    )

    assert peak_potential == pytest.approx(-0.84, abs=0.003)
    assert obj.x().iloc[peak_index] == pytest.approx(-0.84, abs=0.003)


def test_dpv_fit_overlapping_peaks_resolves_two_close_cathodic_peaks(
    ecat_module,
    blank_echem_factory,
):
    obj = _synthetic_dpv(
        ecat_module,
        blank_echem_factory,
        centers=(-0.84, -0.89),
        amplitudes=(-2.0e-6, -1.5e-6),
        sigmas=(0.018, 0.02),
    )

    result = obj.fit_overlapping_peaks(
        {
            "guess potentials": [-0.835, -0.895],
            "plot": False,
            "print": False,
        }
    )

    assert list(result["component"]) == ["peak 1", "peak 2"]
    assert result["potential (V)"].tolist() == pytest.approx([-0.84, -0.89], abs=0.004)
    assert result["amplitude (A)"].tolist() == pytest.approx([-2.0e-6, -1.5e-6], rel=0.15)


def test_dpv_fit_overlapping_peaks_can_plot_overlay(ecat_module, blank_echem_factory):
    obj = _synthetic_dpv(
        ecat_module,
        blank_echem_factory,
        centers=(-0.84, -0.89),
        amplitudes=(-2.0e-6, -1.5e-6),
        sigmas=(0.018, 0.02),
    )

    obj.fit_overlapping_peaks(
        {
            "guess potentials": [-0.835, -0.895],
            "plot": True,
            "print": False,
        }
    )

    ax = plt.gca()
    labels = [text.get_text() for text in ax.get_legend().get_texts()]
    assert len(ax.lines) >= 4
    assert "DPV Data" in labels
    assert "Total Fit" in labels
    assert "Peak 1" in labels
    assert "Peak 2" in labels
    plt.close(ax.figure)


def test_dpv_fit_overlapping_peaks_accepts_custom_overlay_labels(ecat_module, blank_echem_factory):
    obj = _synthetic_dpv(
        ecat_module,
        blank_echem_factory,
        centers=(-0.84, -0.89),
        amplitudes=(-2.0e-6, -1.5e-6),
        sigmas=(0.018, 0.02),
    )

    obj.fit_overlapping_peaks(
        {
            "guess potentials": [-0.835, -0.895],
            "data label": "Measured DPV",
            "fit label": "Model",
            "peak 1 label": "Left Peak",
            "peak 2 label": "Right Peak",
            "plot": True,
            "print": False,
        }
    )

    ax = plt.gca()
    labels = [text.get_text() for text in ax.get_legend().get_texts()]
    assert "Measured DPV" in labels
    assert "Model" in labels
    assert "Left Peak" in labels
    assert "Right Peak" in labels
    plt.close(ax.figure)


def test_dpv_fit_overlapping_peaks_can_hide_overlay_legend(ecat_module, blank_echem_factory):
    obj = _synthetic_dpv(
        ecat_module,
        blank_echem_factory,
        centers=(-0.84, -0.89),
        amplitudes=(-2.0e-6, -1.5e-6),
        sigmas=(0.018, 0.02),
    )

    obj.fit_overlapping_peaks(
        {
            "guess potentials": [-0.835, -0.895],
            "legend": False,
            "plot": True,
            "print": False,
        }
    )

    ax = plt.gca()
    assert ax.get_legend() is None
    plt.close(ax.figure)


def test_dpv_fit_overlapping_peaks_honors_fit_and_center_windows(
    ecat_module,
    blank_echem_factory,
):
    obj = _synthetic_dpv(
        ecat_module,
        blank_echem_factory,
        centers=(-0.84, -0.89),
        amplitudes=(-2.0e-6, -1.5e-6),
        sigmas=(0.018, 0.02),
    )

    result = obj.fit_overlapping_peaks(
        {
            "guess potentials": [-0.835, -0.895],
            "fit window": [-0.93, -0.80],
            "center window": 0.015,
            "plot": False,
            "print": False,
        }
    )

    assert result["potential (V)"].tolist() == pytest.approx([-0.84, -0.89], abs=0.004)
    assert abs(result["potential (V)"].iloc[0] - -0.835) <= 0.015
    assert abs(result["potential (V)"].iloc[1] - -0.895) <= 0.015


def test_dpv_fit_overlapping_peaks_rejects_guesses_outside_fit_window(
    ecat_module,
    blank_echem_factory,
):
    obj = _synthetic_dpv(ecat_module, blank_echem_factory)

    with pytest.raises(ValueError, match="fit window"):
        obj.fit_overlapping_peaks(
            {
                "guess potentials": [-0.84, -0.89],
                "fit window": [-0.86, -0.82],
                "plot": False,
                "print": False,
            }
        )


def test_dpv_fit_overlapping_peaks_uses_shifted_axis_guesses(
    ecat_module,
    blank_echem_factory,
):
    obj = _synthetic_dpv(
        ecat_module,
        blank_echem_factory,
        centers=(-0.84, -0.89),
        amplitudes=(-2.0e-6, -1.5e-6),
        sigmas=(0.018, 0.02),
    )
    obj.reference_shift = 0.45
    obj.reference_label = "Fc/Fc+"

    result = obj.fit_overlapping_peaks(
        {
            "guess potentials": [-1.285, -1.345],
            "plot": False,
            "print": False,
        }
    )

    assert result["potential (V)"].tolist() == pytest.approx([-1.29, -1.34], abs=0.004)


def test_dpv_fit_overlapping_peaks_converts_raw_guesses_on_shifted_axis(
    ecat_module,
    blank_echem_factory,
):
    obj = _synthetic_dpv(
        ecat_module,
        blank_echem_factory,
        centers=(-0.84, -0.89),
        amplitudes=(-2.0e-6, -1.5e-6),
        sigmas=(0.018, 0.02),
    )
    obj.reference_shift = 0.45
    obj.reference_label = "Fc/Fc+"

    result = obj.fit_overlapping_peaks(
        {
            "guess potentials": [-0.835, -0.895],
            "plot": False,
            "print": False,
        }
    )

    assert result["potential (V)"].tolist() == pytest.approx([-1.29, -1.34], abs=0.004)


def test_dpv_fit_overlapping_peaks_rejects_guesses_outside_active_axis(
    ecat_module,
    blank_echem_factory,
):
    obj = _synthetic_dpv(ecat_module, blank_echem_factory)

    with pytest.raises(ValueError, match="guess potentials.*outside"):
        obj.fit_overlapping_peaks(
            {
                "guess potentials": [-0.84, -1.4],
                "plot": False,
                "print": False,
            }
        )
