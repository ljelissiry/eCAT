import numpy as np
import pandas as pd
import pytest


def _ca_object(ecat_module, blank_echem_factory):
    obj = blank_echem_factory(ecat_module.ca)
    obj.name = "CA_test"
    obj.type = "Chronoamperometry"
    obj.data = pd.DataFrame(
        {
            "Time": [0.0, 1.0, 2.0, 3.0],
            "Current": [0.0, 2.0, 4.0, 6.0],
        }
    )
    obj.units = {"Time": "s", "Current": "A"}
    return obj


def test_ca_current_at_time_interpolates_current(ecat_module, blank_echem_factory):
    obj = _ca_object(ecat_module, blank_echem_factory)

    result = obj.current_at_time(1.5, {"plot": False, "print": False})

    assert result["time"] == pytest.approx(1.5)
    assert result["current"] == pytest.approx(3.0)
    assert result["current source"] == "current"


def test_ca_current_at_time_can_use_corrected_current(ecat_module, blank_echem_factory):
    obj = _ca_object(ecat_module, blank_echem_factory)

    result = obj.current_at_time(
        1.5,
        {
            "plot": False,
            "print": False,
            "corrected current": True,
            "baseline correction": "tail",
            "baseline tail fraction": 0.25,
        },
    )

    assert result["baseline current"] == pytest.approx(6.0)
    assert result["current"] == pytest.approx(-3.0)
    assert result["current source"] == "corrected current"


def test_ca_average_current_uses_time_weighted_window(ecat_module, blank_echem_factory):
    obj = _ca_object(ecat_module, blank_echem_factory)

    result = obj.average_current(
        [1.0, 3.0],
        {"plot": False, "print": False},
    )

    assert result["time range"] == (1.0, 3.0)
    assert result["average current"] == pytest.approx(4.0)


def test_ca_current_at_time_prints_equation_and_metric_table(ecat_module, blank_echem_factory, capsys):
    obj = _ca_object(ecat_module, blank_echem_factory)

    result = obj.current_at_time(
        1.5,
        {"plot": False, "print": True, "pretty print": False},
    )

    output = capsys.readouterr().out
    assert "Current At Time:" in output
    assert "i(t) = interp(i, t)" in output
    assert list(result.table.columns) == ["Metric", "Value"]
    assert result.table["Metric"].tolist() == ["Time", "Current", "Current Source"]


def test_ca_plot_can_show_corrected_current(ecat_module, blank_echem_factory):
    obj = _ca_object(ecat_module, blank_echem_factory)

    ax = obj.plot(
        {
            "corrected current": True,
            "baseline correction": "tail",
            "baseline tail fraction": 0.25,
            "title": False,
            "legend": False,
        }
    )

    np.testing.assert_allclose(ax.lines[0].get_ydata(), [-6.0, -4.0, -2.0, 0.0])
    assert ax.get_ylabel() == "Current (A)"


def test_ca_rate_at_time_reports_electron_flow(ecat_module, blank_echem_factory):
    obj = _ca_object(ecat_module, blank_echem_factory)

    result = obj.rate_at_time(2.0, {"plot": False, "print": False})

    assert result["time"] == pytest.approx(2.0)
    assert result["current"] == pytest.approx(4.0)
    assert result["electron flow"] == pytest.approx(4.0 / ecat_module.F)
    assert result["absolute electron flow"] == pytest.approx(4.0 / ecat_module.F)
    assert "TOF" not in result


def test_ca_rate_at_time_reports_tof_from_explicit_catalyst_moles(ecat_module, blank_echem_factory):
    obj = _ca_object(ecat_module, blank_echem_factory)

    result = obj.rate_at_time(
        2.0,
        {
            "plot": False,
            "print": False,
            "num electrons": 2,
            "catalyst moles": 1e-6,
        },
    )

    expected_molecular_rate = 4.0 / (2 * ecat_module.F)
    assert result["molecular rate"] == pytest.approx(expected_molecular_rate)
    assert result["TOF"] == pytest.approx(expected_molecular_rate / 1e-6)
    assert result["catalyst moles"] == pytest.approx(1e-6)


def test_ca_average_rate_resolves_tof_from_species_and_volume(ecat_module, blank_echem_factory):
    obj = _ca_object(ecat_module, blank_echem_factory)
    obj.compounds = ["Cat", "Substrate"]
    obj.concentrations = ["1 mM", "50 mM"]

    result = obj.average_rate(
        [1.0, 3.0],
        {
            "plot": False,
            "print": False,
            "num electrons": 2,
            "species": "Cat",
            "volume": "2 mL",
        },
    )

    expected_molecular_rate = 4.0 / (2 * ecat_module.F)
    expected_catalyst_moles = 1e-3 * 2e-3
    assert result["average current"] == pytest.approx(4.0)
    assert result["molecular rate"] == pytest.approx(expected_molecular_rate)
    assert result["catalyst moles"] == pytest.approx(expected_catalyst_moles)
    assert result["TOF"] == pytest.approx(expected_molecular_rate / expected_catalyst_moles)
    assert result["TOF unit"] == "s^-1"


def test_ca_rate_at_time_prints_equations_and_metric_table(ecat_module, blank_echem_factory, capsys):
    obj = _ca_object(ecat_module, blank_echem_factory)

    result = obj.rate_at_time(
        2.0,
        {
            "plot": False,
            "print": True,
            "pretty print": False,
            "num electrons": 2,
            "catalyst moles": 1e-6,
        },
    )

    output = capsys.readouterr().out
    assert "Rate At Time:" in output
    assert "electron flow = i / F" in output
    assert "TOF = molecular rate / catalyst moles" in output
    assert list(result.table.columns) == ["Metric", "Value"]
    assert "Electron Flow" in result.table["Metric"].tolist()
    assert "TOF" in result.table["Metric"].tolist()
