import pandas as pd
import pytest
from matplotlib import colors as mpl_colors


def test_ca_charge_returns_result_and_plots_by_default(ecat_module, blank_echem_factory):
    obj = blank_echem_factory(ecat_module.ca)
    obj.type = "Chronoamperometry"
    obj.data = pd.DataFrame({"Time": [0.0, 1.0, 2.0], "Current": [0.0, 0.5, 1.0]})
    obj.units = {"Time": "s", "Current": "A"}
    obj.num_x_cols = 1

    result = obj.charge({"print": False})

    assert result["charge"].tolist() == pytest.approx([0.0, 0.5, 1.5])
    assert result["final charge"] == pytest.approx(1.5)
    assert result.axes is not None
    assert result.axes.get_xlabel() == "Time s"
    assert result.axes.get_ylabel() == "Charge (C)"


def test_ca_charge_can_resolve_target_from_moles_and_electrons(
    ecat_module,
    blank_echem_factory,
):
    obj = blank_echem_factory(ecat_module.ca)
    obj.type = "Chronoamperometry"
    obj.data = pd.DataFrame({"Time": [0.0, 1.0, 2.0], "Current": [0.0, 0.5, 1.0]})
    obj.units = {"Time": "s", "Current": "A"}
    obj.num_x_cols = 1

    result = obj.charge(
        {
            "target moles": 1 / ecat_module.objects.F,
            "target electrons": 1,
            "plot": False,
            "print": False,
        }
    )

    assert result["target charge"] == pytest.approx(1.0)
    assert result["time at target charge"] == pytest.approx(1.5)
    assert result.axes is None


def test_ca_time_at_charge_uses_target_options_and_can_plot_ca(
    ecat_module,
    blank_echem_factory,
):
    obj = blank_echem_factory(ecat_module.ca)
    obj.type = "Chronoamperometry"
    obj.data = pd.DataFrame({"Time": [0.0, 1.0, 2.0], "Current": [0.0, 0.5, 1.0]})
    obj.units = {"Time": "s", "Current": "A"}
    obj.num_x_cols = 1

    result = obj.time_at_charge(
        {
            "target charge": 1.0,
            "plot": True,
            "plot ca": True,
            "print": False,
        }
    )

    assert result["time"] == pytest.approx(1.5)
    assert result["target charge"] == pytest.approx(1.0)
    assert result.axes is not None
    assert result.axes.get_ylabel() == "Current (mA)"


def test_ca_plot_accepts_plot_charge_overlay(ecat_module, blank_echem_factory):
    obj = blank_echem_factory(ecat_module.ca)
    obj.type = "Chronoamperometry"
    obj.data = pd.DataFrame({"Time": [0.0, 1.0, 2.0], "Current": [0.0, 0.5, 1.0]})
    obj.units = {"Time": "s", "Current": "A"}
    obj.num_x_cols = 1

    ax = obj.plot({"plot charge": True, "print": False})

    assert ax.get_ylabel() == "Current (mA)"
    assert len(ax.figure.axes) == 2
    assert ax.figure.axes[1].get_ylabel() == "Charge (C)"


def test_ca_plot_charge_overlay_defaults_current_trace_to_black(
    ecat_module,
    blank_echem_factory,
):
    obj = blank_echem_factory(ecat_module.ca)
    obj.type = "Chronoamperometry"
    obj.data = pd.DataFrame({"Time": [0.0, 1.0, 2.0], "Current": [0.0, 0.5, 1.0]})
    obj.units = {"Time": "s", "Current": "A"}
    obj.num_x_cols = 1

    ax = obj.plot({"plot charge": True, "print": False})

    assert ax.lines[0].get_color() == "black"


def test_ca_plot_accepts_grid_option(ecat_module, blank_echem_factory):
    obj = blank_echem_factory(ecat_module.ca)
    obj.type = "Chronoamperometry"
    obj.data = pd.DataFrame({"Time": [0.0, 1.0, 2.0], "Current": [0.0, 0.5, 1.0]})
    obj.units = {"Time": "s", "Current": "A"}
    obj.num_x_cols = 1

    ax = obj.plot({"grid": True, "print": False})

    assert any(line.get_visible() for line in ax.xaxis.get_gridlines())
    assert any(line.get_visible() for line in ax.yaxis.get_gridlines())


def test_ca_integrate_and_plot_charge_are_not_public_methods(ecat_module):
    assert not hasattr(ecat_module.ca, "integrate")
    assert not hasattr(ecat_module.ca, "plot_charge")


def test_ca_objects_sort_and_filter_by_chrono_keys(ecat_module, blank_echem_factory):
    first = blank_echem_factory(ecat_module.ca)
    first.type = "Chronoamperometry"
    first.data = pd.DataFrame({"Time": [0.0, 1.0], "Current": [0.0, 0.5]})
    first.units = {"Time": "s", "Current": "A"}
    first.num_x_cols = 1
    first.init_E = -1.0
    first.run_time = 60.0
    first.sample_interval = 1.0

    second = blank_echem_factory(ecat_module.ca)
    second.type = "Chronoamperometry"
    second.data = pd.DataFrame({"Time": [0.0, 1.0], "Current": [0.0, 2.0]})
    second.units = {"Time": "s", "Current": "A"}
    second.num_x_cols = 1
    second.init_E = -1.2
    second.run_time = 120.0
    second.sample_interval = 1.0

    sorted_objects = ecat_module.sort([first, second], "final charge", {"print": False})
    filtered_objects = ecat_module.filter(
        [first, second],
        {"applied potential": -1.2},
        {"print": False},
    )

    assert sorted_objects == [first, second]
    assert filtered_objects == [second]


def test_ca_current_stats_are_not_sort_filter_group_keys(ecat_module, blank_echem_factory):
    obj = blank_echem_factory(ecat_module.ca)
    obj.type = "Chronoamperometry"
    obj.data = pd.DataFrame({"Time": [0.0, 1.0], "Current": [0.0, 0.5]})
    obj.units = {"Time": "s", "Current": "A"}
    obj.num_x_cols = 1

    valid_keys = ecat_module.get_sort_group_dict()

    assert "min current" not in valid_keys
    assert "max current" not in valid_keys
    assert "avg current" not in valid_keys
    assert ecat_module.sort([obj], "avg current", {"print": False}) == [obj]
    assert ecat_module.filter([obj], {"avg current": 0.25}, {"print": False}) == [obj]


def test_ca_plot_includes_zero_on_y_axis_by_default(ecat_module, blank_echem_factory):
    obj = blank_echem_factory(ecat_module.ca)
    obj.type = "Chronoamperometry"
    obj.data = pd.DataFrame({"Time": [0.0, 1.0, 2.0], "Current": [0.5, 0.75, 1.0]})
    obj.units = {"Time": "s", "Current": "A"}
    obj.num_x_cols = 1

    ax = obj.plot({"print": False})

    y0, y1 = ax.get_ylim()
    assert y0 <= 0 <= y1


def test_ca_plot_default_title_uses_chrono_metadata(ecat_module, blank_echem_factory):
    obj = blank_echem_factory(ecat_module.ca)
    obj.type = "Chronoamperometry"
    obj.name = "sample_ca"
    obj.compounds = ["Fe-2imm"]
    obj.concentrations = ["10 mM"]
    obj.gas = "CO2"
    obj.solvent = "H2O"
    obj.init_E = -1.0
    obj.run_time = 3600.0
    obj.data = pd.DataFrame({"Time": [0.0, 1.0], "Current": [0.0, 0.5]})
    obj.units = {"Time": "s", "Current": "A"}
    obj.num_x_cols = 1

    ax = obj.plot({"legend": False})

    assert ax.figure._suptitle is not None
    assert ax.figure._suptitle.get_text() == "10 mM Fe-2imm"
    assert ax.get_title() == "CA, H$_2$O, CO$_2$, -1 V, 60 min"


def test_ca_charge_plots_black_by_default(ecat_module, blank_echem_factory):
    obj = blank_echem_factory(ecat_module.ca)
    obj.type = "Chronoamperometry"
    obj.data = pd.DataFrame({"Time": [0.0, 1.0], "Current": [0.0, 1.0]})
    obj.units = {"Time": "s", "Current": "A"}
    obj.num_x_cols = 1

    result = obj.charge({"print": False})

    assert result.axes.lines[0].get_color() == "black"


def test_ca_charge_accepts_plot_title(ecat_module, blank_echem_factory):
    obj = blank_echem_factory(ecat_module.ca)
    obj.type = "Chronoamperometry"
    obj.data = pd.DataFrame({"Time": [0.0, 1.0], "Current": [0.0, 1.0]})
    obj.units = {"Time": "s", "Current": "A"}
    obj.num_x_cols = 1

    result = obj.charge({"title": "Corrected Charge", "print": False})

    assert result.axes.get_title() == "Corrected Charge"


def test_ca_charge_colors_second_axis_from_charge_color(ecat_module, blank_echem_factory):
    obj = blank_echem_factory(ecat_module.ca)
    obj.type = "Chronoamperometry"
    obj.data = pd.DataFrame({"Time": [0.0, 1.0], "Current": [0.0, 1.0]})
    obj.units = {"Time": "s", "Current": "A"}
    obj.num_x_cols = 1

    ax = obj.plot({"plot charge": True, "charge color": "tab:green", "print": False})
    charge_ax = ax.figure.axes[1]

    assert charge_ax.lines[0].get_color() == "tab:green"
    assert charge_ax.yaxis.label.get_color() == "tab:green"
    assert charge_ax.spines["right"].get_edgecolor() == pytest.approx(
        mpl_colors.to_rgba("tab:green")
    )


def test_ca_charge_threshold_baseline_correction_detects_baseline_time(
    ecat_module,
    blank_echem_factory,
):
    obj = blank_echem_factory(ecat_module.ca)
    obj.type = "Chronoamperometry"
    obj.data = pd.DataFrame(
        {
            "Time": [0.0, 1.0, 2.0, 3.0],
            "Current": [-0.5, -0.2, -0.04, -0.03],
        }
    )
    obj.units = {"Time": "s", "Current": "A"}
    obj.num_x_cols = 1

    result = obj.charge(
        {
            "baseline correction": True,
            "baseline threshold": 0.05,
            "plot": False,
            "print": False,
        }
    )

    assert result["charge"].tolist() == pytest.approx([0.0, -0.2, -0.24, -0.27])
    assert result["corrected charge"].tolist() == pytest.approx([0.0, -0.15, -0.14, -0.12])
    assert result["final charge"] == pytest.approx(-0.27)
    assert result["final corrected charge"] == pytest.approx(-0.12)
    assert result["baseline current"] == pytest.approx(-0.05)
    assert result["baseline time"] == pytest.approx(2.0)
    assert result["removed charge"].tolist() == pytest.approx([0.0, -0.05, -0.1, -0.15])


def test_ca_charge_tail_baseline_correction_uses_final_five_percent_by_default(
    ecat_module,
    blank_echem_factory,
):
    obj = blank_echem_factory(ecat_module.ca)
    obj.type = "Chronoamperometry"
    obj.data = pd.DataFrame(
        {
            "Time": [0.0, 1.0, 2.0, 3.0, 4.0],
            "Current": [-1.0, -0.5, -0.2, -0.1, -0.1],
        }
    )
    obj.units = {"Time": "s", "Current": "A"}
    obj.num_x_cols = 1

    result = obj.charge({"baseline correction": True, "plot": False, "print": False})

    assert result["baseline correction"] == "tail"
    assert result["baseline tail fraction"] == pytest.approx(0.05)
    assert result["baseline current"] == pytest.approx(-0.1)
    assert result["corrected charge"].tolist() == pytest.approx([0.0, -0.4, -0.5, -0.5, -0.5])


def test_time_at_charge_prints_scaled_time_unit(
    ecat_module,
    blank_echem_factory,
    capsys,
):
    obj = blank_echem_factory(ecat_module.ca)
    obj.type = "Chronoamperometry"
    obj.data = pd.DataFrame({"Time": [0.0, 600.0, 1200.0], "Current": [0.0, 0.1, 0.1]})
    obj.units = {"Time": "s", "Current": "A"}
    obj.num_x_cols = 1

    result = obj.time_at_charge({"target charge": 60.0, "plot": False})

    assert result["time"] == pytest.approx(600.0)
    assert result["display time"] == pytest.approx(10.0)
    assert "10.0 min" in capsys.readouterr().out


def test_time_at_charge_does_not_add_legend_by_default(ecat_module, blank_echem_factory):
    obj = blank_echem_factory(ecat_module.ca)
    obj.type = "Chronoamperometry"
    obj.data = pd.DataFrame({"Time": [0.0, 1.0, 2.0], "Current": [0.0, 0.5, 1.0]})
    obj.units = {"Time": "s", "Current": "A"}
    obj.num_x_cols = 1

    result = obj.time_at_charge({"target charge": 1.0, "plot": True, "print": False})

    assert result.axes.get_legend() is None
