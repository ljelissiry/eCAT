import numpy as np
import pytest
import matplotlib.pyplot as plt


class FakeCV:
    def __init__(self, name, scan_rate, ep1, ep2, temperature=298.0):
        self.name = name
        self.scan_rate = scan_rate
        self.temperature = temperature
        self.ep1 = ep1
        self.ep2 = ep2
        self.half_wave_options = []

    def half_wave_potential(self, options=None):
        self.half_wave_options.append(dict(options or {}))
        e_half = (self.ep1 + self.ep2) / 2
        delta_e = abs(self.ep1 - self.ep2)
        if options and options.get("plot", False):
            plt.scatter([self.ep1, self.ep2], [self.scan_rate, -self.scan_rate])
        return {
            "E(1/2)": e_half,
            "ΔE": delta_e,
            "peak 1": {"Ep": self.ep1},
            "peak 2": {"Ep": self.ep2},
        }


def _base_options(**overrides):
    options = {
        "D": 1e-5,
        "plot": False,
        "print": False,
        "warn ir drop": False,
    }
    options.update(overrides)
    return options


def test_nicholson_lavagnini_uses_millivolts(ecat_module):
    expected = (-0.6288 + 0.0021 * 100) / (1 - 0.017 * 100)

    assert ecat_module._nicholson_psi_lavagnini(100) == pytest.approx(expected)


def test_nicholson_agarwal_table_lookup_uses_table_4_value(ecat_module):
    assert ecat_module._nicholson_psi_agarwal(100) == pytest.approx(0.558)
    assert ecat_module._nicholson_psi_agarwal(100.5) == pytest.approx((0.558 + 0.544) / 2)


def test_nicholson_delta_ep_below_range_is_excluded(ecat_module):
    cvs = [
        FakeCV("too reversible", 0.1, 0.100, 0.050),
        FakeCV("valid 1", 0.2, 0.100, 0.000),
        FakeCV("valid 2", 0.4, 0.120, 0.000),
    ]

    result = ecat_module.nicholson_analysis(cvs, _base_options())
    data = result["data"]
    summary = result["summary"]

    row = data.loc[data["name"] == "too reversible"].iloc[0]
    assert row["included"] == False
    assert "too reversible" in row["exclusion reason"]
    assert summary["num included"] == 2
    assert summary["num excluded"] == 1


def test_nicholson_delta_ep_above_range_is_excluded(ecat_module):
    cvs = [
        FakeCV("outside", 0.1, 0.250, 0.000),
        FakeCV("valid 1", 0.2, 0.100, 0.000),
        FakeCV("valid 2", 0.4, 0.120, 0.000),
    ]

    result = ecat_module.nicholson_analysis(cvs, _base_options())
    data = result["data"]
    summary = result["summary"]

    row = data.loc[data["name"] == "outside"].iloc[0]
    assert row["included"] == False
    assert "outside Nicholson range" in row["exclusion reason"]
    assert summary["num included"] == 2
    assert summary["num excluded"] == 1


def test_nicholson_through_origin_fit_recovers_known_k0(ecat_module):
    x = np.array([0.1, 0.2, 0.4, 0.8])
    y = 0.35 * x

    fit = ecat_module._nicholson_fit(x, y, through_origin=True)

    assert fit["slope"] == pytest.approx(0.35)
    assert fit["intercept"] == pytest.approx(0.0)
    assert fit["r2"] == pytest.approx(1.0)


def test_nicholson_analysis_returns_dict_with_data_and_summary(ecat_module):
    cvs = [
        FakeCV("valid 1", 0.2, 0.100, 0.000),
        FakeCV("valid 2", 0.4, 0.120, 0.000),
    ]

    result = ecat_module.nicholson_analysis(cvs, _base_options())
    data = result["data"]
    summary = result["summary"]

    assert set(result) == {"data", "summary"}
    assert list(data.columns) == [
        "name",
        "scan rate / V s^-1",
        "temperature / K",
        "Ep1 / V",
        "Ep2 / V",
        "E1/2 / V",
        "ΔEp / V",
        "nΔEp / mV",
        "ψ",
        "Nicholson x / s cm^-1",
        "k0 point / cm s^-1",
        "included",
        "exclusion reason",
        "psi source",
    ]
    assert summary["num points"] == 2
    assert summary["num included"] == 2
    assert summary["D / cm^2 s^-1"] == pytest.approx(1e-5)
    assert summary["psi source"] == "agarwal table"
    assert "equation" in summary
    assert "x definition" in summary


def test_nicholson_analysis_accepts_single_cv(ecat_module):
    cv = FakeCV("valid", 0.2, 0.100, 0.000)

    with pytest.warns(UserWarning, match="requires at least 2 included points"):
        with pytest.raises(ValueError, match="requires at least 2 included points"):
            ecat_module.nicholson_analysis(cv, _base_options())


def test_nicholson_analysis_failure_message_reports_counts_and_reasons(ecat_module):
    cvs = [
        FakeCV("too reversible", 0.1, 0.050, 0.000),
        FakeCV("valid 1", 0.2, 0.100, 0.000),
    ]

    with pytest.warns(UserWarning, match="requires at least 2 included points"):
        with pytest.raises(ValueError, match="Found 1 included out of 2 total") as excinfo:
            ecat_module.nicholson_analysis(cvs, _base_options())

    message = str(excinfo.value)
    assert "too reversible" in message
    assert "nΔEp between 61 and 212 mV" in message
    assert "\n  - too reversible:" in message
    assert "nΔEp = 50" in message
    assert "'exclude invalid delta ep': False" in message


def test_nicholson_analysis_accepts_fit_model_linear(ecat_module):
    cvs = [
        FakeCV("valid 1", 0.2, 0.100, 0.000),
        FakeCV("valid 2", 0.4, 0.120, 0.000),
        FakeCV("valid 3", 0.8, 0.130, 0.000),
    ]

    result = ecat_module.nicholson_analysis(cvs, _base_options(**{"fit model": "linear"}))

    assert result["summary"]["fit model"] == "linear"
    assert result["summary"]["fit through origin"] is False


def test_nicholson_display_tables_respect_sig_figs_and_autoscaled_units(ecat_module):
    cvs = [
        FakeCV("valid 1", 0.025, 0.100, -0.020),
        FakeCV("valid 2", 0.050, 0.120, 0.000),
    ]

    result = ecat_module.nicholson_analysis(cvs, _base_options(**{"sig figs": 3}))
    display_data = ecat_module._nicholson_display_data_table(result["data"], {"sig figs": 3})
    summary_table = ecat_module._nicholson_summary_display_table(result["summary"], {"sig figs": 3})

    assert "scan rate / mV/s" in display_data.columns
    assert "ΔEp / mV" in display_data.columns
    assert any(col.startswith("k0 point / ") for col in display_data.columns)
    assert summary_table.loc[summary_table["Setting"] == "k0", "Value"].iloc[0] != ""


def test_nicholson_analysis_pretty_prints_equation_summary_and_data(ecat_module, capsys):
    cvs = [
        FakeCV("valid 1", 0.2, 0.100, 0.000),
        FakeCV("valid 2", 0.4, 0.120, 0.000),
    ]

    ecat_module.nicholson_analysis(cvs, _base_options(print=True, pretty_print=False))

    printed = capsys.readouterr().out
    assert "Nicholson Analysis Equation" in printed
    assert "Nicholson Parameters" in printed
    assert "Nicholson Analysis Summary" in printed
    assert "Nicholson Analysis Data" in printed
    assert "psi = k0 x" in printed
    assert "x = (R T / (pi D n F v))^1/2" not in printed


def test_nicholson_plot_all_uses_diagnostic_and_fit_figures(ecat_module):
    cvs = [
        FakeCV("valid 1", 0.2, 0.100, 0.000),
        FakeCV("valid 2", 0.4, 0.120, 0.000),
        FakeCV("valid 3", 0.8, 0.130, 0.000),
    ]

    result = ecat_module.nicholson_analysis(
        cvs,
        _base_options(plot=True, plot_all=True, print=False),
    )

    assert set(result) == {"data", "summary"}
    assert len(plt.get_fignums()) == 2
    for cv in cvs:
        assert cv.half_wave_options[-1]["plot"] is True
        assert cv.half_wave_options[-1]["new plot"] is False
