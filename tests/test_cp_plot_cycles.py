import pandas as pd
import numpy as np
import warnings
from matplotlib.axes import Axes


def _make_cp_for_cycle_plot(ecat_module, blank_echem_factory):
    obj = blank_echem_factory(ecat_module.cp)
    obj.type = "Chronopotentiometry"
    obj.data = pd.DataFrame(
        {
            "Time": list(range(16)),
            "Potential": [0, -1, -2, -3, -3, -2, -1, 0, 0, -1, -2, -3, -3, -2, -1, 0],
        }
    )
    obj.units = {"Time": "s", "Potential": "V"}
    obj.sample_int = 1.0
    obj.segments = 4
    obj.cathodic_current = -0.001
    obj.anodic_current = 0.001
    obj.compounds = ["BatteryMol"]
    obj.concentrations = ["5 mM"]
    obj.gas = "N2"
    obj.solvent = "MeCN"
    obj.high_E_limit = 1.0
    obj.low_E_limit = -1.0
    return obj


def test_cp_plot_cycles_can_color_cycles_by_gradient(ecat_module, blank_echem_factory):
    obj = _make_cp_for_cycle_plot(ecat_module, blank_echem_factory)

    ax = obj.plot_cycles(
        {
            "cycles": [1, 2],
            "color mode": "gradient",
            "legend": False,
            "print": False,
        }
    )

    colors = [line.get_color() for line in ax.lines]
    assert len(colors) == 2
    assert len(set(colors)) == 2


def test_cp_plot_default_title_uses_chrono_metadata(ecat_module, blank_echem_factory):
    obj = _make_cp_for_cycle_plot(ecat_module, blank_echem_factory)

    ax = obj.plot({"legend": False})

    assert ax.figure._suptitle is not None
    assert ax.figure._suptitle.get_text() == "5 mM BatteryMol"
    assert ax.get_title() == "CP, MeCN, N$_2$, 2 cycles, -1 to 1 V"


def test_cp_plot_accepts_grid_option(ecat_module, blank_echem_factory):
    obj = _make_cp_for_cycle_plot(ecat_module, blank_echem_factory)

    ax = obj.plot({"grid": True, "legend": False})

    assert any(line.get_visible() for line in ax.xaxis.get_gridlines())
    assert any(line.get_visible() for line in ax.yaxis.get_gridlines())


def test_cp_plot_cycles_default_title_describes_cycle_selection(
    ecat_module,
    blank_echem_factory,
):
    obj = _make_cp_for_cycle_plot(ecat_module, blank_echem_factory)

    ax = obj.plot_cycles({"cycles": [1, 2], "legend": False})

    assert ax.figure._suptitle is not None
    assert ax.figure._suptitle.get_text() == "5 mM BatteryMol"
    assert ax.get_title() == "CP cycles 1, 2, capacity axis"


def test_cp_cycling_plot_default_title_describes_performance_overlay(
    ecat_module,
    blank_echem_factory,
):
    obj = _make_cp_for_cycle_plot(ecat_module, blank_echem_factory)

    fig, (ax1, ax2) = obj.cycling_plot({"legend": False})

    assert fig._suptitle is not None
    assert fig._suptitle.get_text() == "5 mM BatteryMol"
    assert ax1.get_title() == "CP cycling performance, capacity and efficiency"


def test_cp_cycling_plot_accepts_grid_option(ecat_module, blank_echem_factory):
    obj = _make_cp_for_cycle_plot(ecat_module, blank_echem_factory)

    _fig, (ax1, ax2) = obj.cycling_plot({"grid": True, "legend": False})

    assert any(line.get_visible() for line in ax1.xaxis.get_gridlines())
    assert any(line.get_visible() for line in ax1.yaxis.get_gridlines())
    assert any(line.get_visible() for line in ax2.xaxis.get_gridlines())
    assert any(line.get_visible() for line in ax2.yaxis.get_gridlines())


def test_cp_cycle_info_uses_nan_for_undefined_efficiencies_without_runtime_warning(
    ecat_module,
    blank_echem_factory,
):
    obj = _make_cp_for_cycle_plot(ecat_module, blank_echem_factory)

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        table = obj.cycle_info({"print": False})

    efficiencies = table[["Coulombic Efficiency (%)", "Energy Efficiency (%)"]]
    assert not np.isinf(efficiencies.to_numpy(dtype=float)).any()


def test_cp_cycling_plot_accepts_cycles_selection(ecat_module, blank_echem_factory):
    obj = _make_cp_for_cycle_plot(ecat_module, blank_echem_factory)
    obj.cycle_info = lambda options=None: pd.DataFrame(
        {
            "Cycle": [1, 2, 3, 4, 5],
            "Discharge Capacity (mA·h)": [1, 2, 3, 4, 5],
            "Charge Capacity (mA·h)": [1.1, 2.1, 3.1, 4.1, 5.1],
            "Coulombic Efficiency (%)": [90, 91, 92, 93, 94],
            "Energy Efficiency (%)": [80, 81, 82, 83, 84],
        }
    )

    _fig, (ax1, ax2) = obj.cycling_plot({"cycles": (2, 4), "legend": False})

    plotted_cycles = set()
    for axis in (ax1, ax2):
        for collection in axis.collections:
            offsets = collection.get_offsets()
            if len(offsets):
                plotted_cycles.update(offsets[:, 0].astype(int).tolist())
    assert plotted_cycles == {2, 3, 4}


def test_cp_cycling_plot_supports_percent_capacity_with_cycle_selection(
    ecat_module,
    repo_root,
):
    obj = ecat_module.echem.from_file(
        str(
            repo_root
            / "examples"
            / "data"
            / "chrono_cp"
            / "CP-100-cycles_MeCN_500mVs_100mMTBAPF6_gcWE_ptCE_AgAgbf4-5mM.txt"
        ),
        {"print": False},
    )

    fig, (ax1, ax2) = obj.cycling_plot(
        {
            "cycles": (1, 100, 5),
            "percent capacity": True,
            "capacity mode": "both",
            "efficiency mode": "both",
            "legend": True,
        }
    )

    assert fig is not None
    assert ax1.get_ylabel() == "Capacity (%)"
    assert len(ax1.collections) >= 2
    assert len(ax2.collections) >= 2


def test_cp_cycling_plot_title_spacing_leaves_room_between_title_and_subtitle(
    ecat_module,
    blank_echem_factory,
):
    obj = _make_cp_for_cycle_plot(ecat_module, blank_echem_factory)

    fig, _axes = obj.cycling_plot({"legend": False})

    assert fig._suptitle is not None
    assert 0.97 <= fig._suptitle.get_position()[1] <= 0.99


def test_cp_plot_cycles_uses_multiplot_style_cycle_colorbar(ecat_module, blank_echem_factory):
    obj = _make_cp_for_cycle_plot(ecat_module, blank_echem_factory)

    ax = obj.plot_cycles(
        {
            "cycles": [1, 2],
            "color mode": "gradient",
            "legend": True,
            "legend mode": "colorbar",
            "print": False,
        }
    )

    assert len(ax.figure.axes) == 1
    assert any(isinstance(child, Axes) for child in ax.get_children())
    panel_ax = next(child for child in ax.get_children() if isinstance(child, Axes))
    panel_text = [text.get_text() for text in panel_ax.texts]
    assert "Cycle 1" in panel_text
    assert "Cycle 2" in panel_text
    assert "Cycle" not in panel_text


def test_cp_plot_cycles_colorbar_ticks_use_selected_cycle_values(
    ecat_module,
    blank_echem_factory,
):
    obj = _make_cp_for_cycle_plot(ecat_module, blank_echem_factory)
    t = np.arange(20, dtype=float)
    v = np.linspace(0.0, 1.0, 20)
    obj.get_cycles = lambda options=None: {
        "t": t,
        "v": v,
        "seg_idxs": [np.arange(i * 2, i * 2 + 2) for i in range(10)],
    }

    ax = obj.plot_cycles(
        {
            "cycles": [1, 5],
            "color mode": "gradient",
            "legend": True,
            "legend mode": "colorbar",
            "print": False,
        }
    )
    panel_ax = next(child for child in ax.get_children() if isinstance(child, Axes))
    cbar_ax = next(child for child in panel_ax.get_children() if isinstance(child, Axes))

    assert list(cbar_ax.get_yticks()) == [1.0, 5.0]


def test_cp_plot_cycles_gradient_colors_both_segments_by_cycle(
    ecat_module,
    blank_echem_factory,
):
    obj = _make_cp_for_cycle_plot(ecat_module, blank_echem_factory)

    ax = obj.plot_cycles(
        {
            "cycles": [1, 2],
            "segment": "both",
            "color mode": "gradient",
            "legend": True,
            "legend mode": "colorbar",
            "print": False,
        }
    )

    assert len(ax.lines) == 4
    assert ax.lines[0].get_color() == ax.lines[1].get_color()
    assert ax.lines[2].get_color() == ax.lines[3].get_color()
    assert ax.lines[0].get_color() != ax.lines[2].get_color()
