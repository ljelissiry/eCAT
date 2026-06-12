import warnings

import numpy as np
import pandas as pd
import pytest

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection


def _plot_cv_triplet(cv_factory):
    return [
        cv_factory(name="50mVs_sample_CO2_MeCN_10mM_Fc_run01"),
        cv_factory(
            name="100mVs_sample_N2_DMF_5mM_Fc_run01",
            current=[
                -0.07e-6, -0.06e-6, -0.04e-6, 0.0, 0.2e-6, 0.6e-6, 1.2e-6,
                2.8e-6, 5.0e-6, 4.7e-6, 4.0e-6, 3.2e-6, 2.2e-6, 1.3e-6,
                0.8e-6, 0.3e-6, 0.1e-6, 0.0, -0.02e-6, -0.04e-6, -0.05e-6,
            ],
        ),
        cv_factory(
            name="200mVs_sample_Ar_MeCN_1mM_Fc_run02",
            current=[
                -0.12e-6, -0.10e-6, -0.06e-6, 0.0, 0.4e-6, 1.0e-6, 2.2e-6,
                4.8e-6, 8.6e-6, 7.5e-6, 6.3e-6, 4.9e-6, 3.5e-6, 2.1e-6,
                1.2e-6, 0.5e-6, 0.1e-6, 0.0, -0.04e-6, -0.06e-6, -0.09e-6,
            ],
        ),
    ]


def _multi_segment_cv(cv_factory, n_segments=4):
    up = [0.0, 0.4, 0.8, 1.2]
    down = [0.8, 0.4, 0.0, -0.4]
    potential = []
    current = []
    for seg_idx in range(n_segments):
        segment_x = up if seg_idx % 2 == 0 else down
        potential.extend(segment_x)
        current.extend([seg_idx + j / 10 for j in range(len(segment_x))])
    return cv_factory(potential=potential, current=current)


def test_cv_plot_smoke_with_agg_backend(cv_factory):
    obj = cv_factory()

    ax = obj.plot({"legend": False, "title": False})
    fig = ax.figure

    assert fig is not None
    assert ax is not None
    assert len(ax.lines) == 1
    plt.close(fig)


def test_cv_plot_defaults_to_new_figure(cv_factory):
    obj = cv_factory()
    existing_fig, existing_ax = plt.subplots()

    ax = obj.plot({"legend": False, "title": False})
    fig = ax.figure

    assert fig is not existing_fig
    assert ax is not existing_ax
    assert len(existing_ax.lines) == 0
    assert len(ax.lines) == 1
    plt.close(existing_fig)
    plt.close(fig)


def test_cv_plot_handles_read_only_axis_arrays(cv_factory, monkeypatch):
    obj = cv_factory()
    x_values = np.asarray(obj.x().to_numpy(), dtype=float)
    y_values = np.asarray(obj.y().to_numpy(), dtype=float)
    x_values.setflags(write=False)
    y_values.setflags(write=False)
    x_series = pd.Series(x_values, name=obj.x().name)
    y_series = pd.Series(y_values, name=obj.y().name)

    monkeypatch.setattr(obj, "x", lambda options=None: x_series)
    monkeypatch.setattr(obj, "y", lambda options=None: y_series)

    ax = obj.plot(
        {
            "legend": False,
            "title": False,
            "plot convention": "IUPAC",
            "x unit": None,
            "y unit": None,
        }
    )

    assert len(ax.lines) == 1
    np.testing.assert_allclose(ax.lines[0].get_xdata(), x_values)
    np.testing.assert_allclose(ax.lines[0].get_ydata(), y_values)
    plt.close(ax.figure)


def test_cv_plot_respects_explicit_new_plot_false(cv_factory):
    obj = cv_factory()
    existing_fig, existing_ax = plt.subplots()

    ax = obj.plot({"new plot": False, "legend": False, "title": False})
    fig = ax.figure

    assert fig is existing_fig
    assert ax is existing_ax
    assert len(existing_ax.lines) == 1
    plt.close(existing_fig)


def test_plotting_style_can_restore_matplotlib_defaults(ecat_module):
    ecat_module.plotting_style(False)

    assert mpl.rcParams["font.size"] == mpl.rcParamsDefault["font.size"]
    assert mpl.rcParams["axes.linewidth"] == mpl.rcParamsDefault["axes.linewidth"]

    ecat_module.plotting_style(True)

    assert mpl.rcParams["font.size"] == 11
    assert mpl.rcParams["axes.linewidth"] == 1.1


def test_plotting_style_uses_notebook_profile_by_default(ecat_module):
    ecat_module.plotting_style(True)

    assert list(mpl.rcParams["figure.figsize"]) == [5.0, 3.75]
    assert mpl.rcParams["figure.dpi"] == 150
    assert mpl.rcParams["savefig.dpi"] == 300
    assert mpl.rcParams["font.size"] == 11
    assert mpl.rcParams["legend.fontsize"] == 9
    assert mpl.rcParams["figure.subplot.top"] == 0.88
    assert mpl.rcParams["xtick.minor.ndivs"] == 2
    assert mpl.rcParams["ytick.minor.ndivs"] == 2


def test_plotting_style_supports_publication_profile(ecat_module):
    ecat_module.plotting_style("publication")

    assert list(mpl.rcParams["figure.figsize"]) == [6.5, 4.875]
    assert mpl.rcParams["figure.dpi"] == 300
    assert mpl.rcParams["font.size"] == 16
    assert mpl.rcParams["legend.fontsize"] == 14
    assert mpl.rcParams["axes.linewidth"] == 1.5
    assert mpl.rcParams["xtick.minor.ndivs"] == 2
    assert mpl.rcParams["ytick.minor.ndivs"] == 2

    ecat_module.plotting_style("notebook")


def test_plotting_style_supports_saveant_profile(ecat_module):
    profile = ecat_module.plotting_style("saveant")

    assert profile == "saveant"
    assert list(mpl.rcParams["figure.figsize"]) == [5.2, 3.75]
    assert mpl.rcParams["figure.dpi"] == 150
    assert mpl.rcParams["axes.facecolor"] == "#b8bad8"
    assert mpl.rcParams["figure.facecolor"] == "#d8d8dc"
    assert mpl.rcParams["axes.linewidth"] == 1.8
    assert mpl.rcParams["font.size"] == 15
    assert mpl.rcParams["axes.labelsize"] == 16
    assert mpl.rcParams["xtick.labelsize"] == 14
    assert mpl.rcParams["ytick.labelsize"] == 14
    assert mpl.rcParams["xtick.direction"] == "out"
    assert mpl.rcParams["ytick.direction"] == "out"
    assert mpl.rcParams["xtick.top"] is False
    assert mpl.rcParams["ytick.right"] is False
    assert mpl.rcParams["xtick.minor.ndivs"] == 5
    assert mpl.rcParams["ytick.minor.ndivs"] == 5

    ecat_module.plotting_style("notebook")


def test_old_plot_style_function_names_are_removed(ecat_module):
    assert not hasattr(ecat_module, "default_plotting")
    assert not hasattr(ecat_module, "use_ecat_plot_style")
    assert not hasattr(ecat_module, "scale_bar")


def test_plot_scale_bar_draws_bar_and_removes_y_ticks(cv_factory):
    obj = cv_factory()

    ax = obj.plot(
        {
            "legend": False,
            "title": False,
            "scale bar": {"loc": (-0.1, 0.0), "length": 1e-6},
        }
    )

    assert len(ax.collections) >= 3
    assert any("1e-06" in text.get_text() and "A" in text.get_text() for text in ax.texts)
    assert len(ax.get_yticks()) == 0
    plt.close(ax.figure)


def test_plot_scale_bar_label_is_to_right_of_bar(cv_factory):
    obj = cv_factory()

    ax = obj.plot(
        {
            "legend": False,
            "title": False,
            "scale bar": {"loc": "upper left", "length": 1e-6},
        }
    )

    text = ax.texts[-1]
    cap_max_x = max(
        segment[:, 0].max()
        for collection in ax.collections
        for segment in collection.get_segments()
    )

    assert text.get_ha() == "left"
    assert text.get_position()[0] > cap_max_x
    plt.close(ax.figure)


def test_scale_bar_label_defaults_to_active_legend_fontsize(ecat_module, cv_factory):
    ecat_module.plotting_style("notebook")
    obj = cv_factory()

    ax = obj.plot(
        {
            "legend": False,
            "title": False,
            "scale bar": {"loc": "upper left", "length": 1e-6},
        }
    )

    assert ax.texts[-1].get_fontsize() == pytest.approx(
        ecat_module._active_plot_style_value("legend fontsize")
    )
    plt.close(ax.figure)


def test_scale_bar_label_accepts_explicit_fontsize(cv_factory):
    obj = cv_factory()

    ax = obj.plot(
        {
            "legend": False,
            "title": False,
            "scale bar": {"loc": "upper left", "length": 1e-6, "fontsize": 6.5},
        }
    )

    assert ax.texts[-1].get_fontsize() == pytest.approx(6.5)
    plt.close(ax.figure)


def test_multiplot_scale_bar_draws_once(ecat_module, cv_factory):
    objects = [cv_factory(name="100mVs"), cv_factory(name="200mVs")]

    ax = ecat_module.multiplot(
        objects,
        {
            "legend": False,
            "title": False,
            "scale bar": {"loc": "lower right", "length": 1e-6},
        },
    )

    assert len(ax.texts) == 1
    assert len(ax.get_yticks()) == 0
    plt.close(ax.figure)


def test_multiplot_scale_bar_uses_visual_right_after_us_convention(ecat_module, cv_factory):
    objects = [cv_factory(name="100mVs"), cv_factory(name="200mVs")]

    ax = ecat_module.multiplot(
        objects,
        {
            "legend": False,
            "title": False,
            "plot convention": "US",
            "scale bar": {"loc": "upper right", "length": 1e-6},
        },
    )

    bar_x = ax.collections[0].get_segments()[0][0, 0]
    xmin, xmax = sorted(ax.get_xlim())

    assert ax.xaxis_inverted()
    assert bar_x < (xmin + xmax) / 2
    plt.close(ax.figure)


def test_scale_bar_upper_right_places_label_to_visual_left_without_overlap(cv_factory):
    obj = cv_factory()

    ax = obj.plot(
        {
            "legend": False,
            "title": False,
            "plot convention": "US",
            "x unit": "mV",
            "y unit": "uA",
            "scale bar": {"loc": "upper right", "length": 100},
        }
    )
    ax.figure.canvas.draw()
    renderer = ax.figure.canvas.get_renderer()

    text_bbox = ax.texts[-1].get_window_extent(renderer)
    vertical_bar = ax.collections[0].get_segments()[0]
    bar_display_x = ax.transData.transform(vertical_bar[0])[0]

    assert ax.texts[-1].get_ha() == "right"
    assert text_bbox.x1 < bar_display_x
    plt.close(ax.figure)


def test_scale_bar_upper_left_places_label_to_visual_right_without_overlap(cv_factory):
    obj = cv_factory()

    ax = obj.plot(
        {
            "legend": False,
            "title": False,
            "plot convention": "US",
            "x unit": "mV",
            "y unit": "uA",
            "scale bar": {"loc": "upper left", "length": 100},
        }
    )
    ax.figure.canvas.draw()
    renderer = ax.figure.canvas.get_renderer()

    text_bbox = ax.texts[-1].get_window_extent(renderer)
    vertical_bar = ax.collections[0].get_segments()[0]
    bar_display_x = ax.transData.transform(vertical_bar[0])[0]

    assert ax.texts[-1].get_ha() == "left"
    assert text_bbox.x0 > bar_display_x
    plt.close(ax.figure)


def test_scale_bar_upper_location_respects_inverted_y_axis(ecat_module, cv_factory):
    obj = cv_factory()

    ax = obj.plot({"legend": False, "title": False, "scale bar": False})
    ax.set_ylim(200, -200)
    ecat_module._add_scale_bar(ax, {"scale bar": {"loc": "upper right", "length": 100}}, unit="uA")

    vertical_bar = ax.collections[0].get_segments()[0]
    bar_display_y = ax.transData.transform(vertical_bar.mean(axis=0))[1]
    y_mid = ax.bbox.y0 + ax.bbox.height / 2

    assert ax.yaxis_inverted()
    assert bar_display_y > y_mid
    plt.close(ax.figure)


def _find_vertical_segment_with_height(ax, expected_height):
    for collection in ax.collections:
        for segment in collection.get_segments():
            dx = abs(segment[1, 0] - segment[0, 0])
            dy = abs(segment[1, 1] - segment[0, 1])
            if dx < 1e-12 and dy == pytest.approx(expected_height):
                return segment
    raise AssertionError(f"No vertical segment with height {expected_height} found.")


def test_scale_bar_repositions_after_later_inverted_ylim_change(ecat_module):
    fig, ax = plt.subplots()
    ax.set_xlim(-1, 1)
    ax.set_ylim(-10, 10)
    ecat_module._add_scale_bar(
        ax,
        {"scale bar": {"loc": "upper right", "length": 100}},
        unit="uA",
    )
    ax.set_ylim(200, -200)
    ax.figure.canvas.draw()

    vertical_bar = _find_vertical_segment_with_height(ax, 100)
    expected_x, expected_y = ecat_module._scale_bar_position(ax, "upper right", 100)
    actual_y = vertical_bar.mean(axis=0)[1]
    bar_display_y = ax.transData.transform(vertical_bar.mean(axis=0))[1]
    y_mid = ax.bbox.y0 + ax.bbox.height / 2

    assert ax.yaxis_inverted()
    assert actual_y == pytest.approx(expected_y)
    assert bar_display_y > y_mid
    plt.close(fig)


def test_scale_bar_lower_location_respects_inverted_y_axis(ecat_module, cv_factory):
    obj = cv_factory()

    ax = obj.plot({"legend": False, "title": False, "scale bar": False})
    ax.set_ylim(200, -200)
    ecat_module._add_scale_bar(ax, {"scale bar": {"loc": "lower right", "length": 100}}, unit="uA")

    vertical_bar = ax.collections[0].get_segments()[0]
    bar_display_y = ax.transData.transform(vertical_bar.mean(axis=0))[1]
    y_mid = ax.bbox.y0 + ax.bbox.height / 2

    assert ax.yaxis_inverted()
    assert bar_display_y < y_mid
    plt.close(ax.figure)


def test_adaptive_legend_fontsize_is_capped_by_active_profile(ecat_module):
    ecat_module.plotting_style("notebook")
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1], label="Trace 1")

    legend_fs, _legend_loc, legend_outside = ecat_module._resolve_adaptive_legend_layout(
        ax,
        {"plot labels": ["Trace 1"], "gradient groups": []},
        {"legend loc": "best", "legend fontsize": "auto", "legend outside": False},
    )

    assert legend_fs <= mpl.rcParams["legend.fontsize"]
    assert legend_fs <= mpl.rcParams["axes.labelsize"]
    assert legend_outside is False
    plt.close(fig)


def test_default_legend_fontsize_falls_back_to_axis_label_size(ecat_module, monkeypatch):
    monkeypatch.setattr(ecat_module, "_active_plot_style_value", lambda key: None)

    with mpl.rc_context({"axes.labelsize": 13, "legend.fontsize": 7}):
        assert ecat_module._default_legend_fontsize() == pytest.approx(13)
        assert ecat_module._legend_fontsize(["short"]) == pytest.approx(13)
        assert ecat_module._legend_fontsize_from_color_spec({"plot labels": ["short"]}) == pytest.approx(13)
        assert ecat_module._legend_font_candidates() == [13, 11, 9, 7]


def test_discrete_legend_overlap_score_ignores_unlabeled_artists(ecat_module):
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])

    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        score = ecat_module._discrete_legend_overlap_score(ax, 9, "upper right")

    assert score == 0
    assert not any("No artists with labels found" in str(warning.message) for warning in captured)
    plt.close(fig)


def test_cv_plot_minor_ticks_option_controls_axis_locators(cv_factory):
    obj = cv_factory()

    ax_default = obj.plot({"legend": False, "title": False})
    assert isinstance(ax_default.xaxis.get_minor_locator(), mpl.ticker.AutoMinorLocator)
    assert ax_default.xaxis.get_minor_locator().ndivs == 2
    assert ax_default.yaxis.get_minor_locator().ndivs == 2
    plt.close(ax_default.figure)

    ax_off = obj.plot({"legend": False, "title": False, "minor ticks": False})
    assert isinstance(ax_off.xaxis.get_minor_locator(), mpl.ticker.NullLocator)
    plt.close(ax_off.figure)

    ax_on = obj.plot({"legend": False, "title": False, "minor ticks": 3})
    assert isinstance(ax_on.xaxis.get_minor_locator(), mpl.ticker.AutoMinorLocator)
    assert isinstance(ax_on.yaxis.get_minor_locator(), mpl.ticker.AutoMinorLocator)
    plt.close(ax_on.figure)


def test_saveant_plot_style_places_axis_labels_inside_and_snaps_bounds(ecat_module, cv_factory):
    ecat_module.plotting_style("saveant")
    obj = cv_factory()

    ax = obj.plot({"legend": False, "title": False})
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    xticks = ax.get_xticks()
    yticks = ax.get_yticks()

    assert ax.get_xlabel() == ""
    assert ax.get_ylabel() == ""
    inside_labels = [text.get_text() for text in ax.texts]
    assert "$E$ (V)" in inside_labels
    assert "$i$ (μA)" in inside_labels
    assert all(text.get_fontstyle() == "normal" for text in ax.texts if text.get_text() in {"$E$ (V)", "$i$ (μA)"})
    assert not any("Potential" in label for label in inside_labels)
    assert not any("Current" in label for label in inside_labels)
    assert np.any(np.isclose(xticks, xlim[0]))
    assert np.any(np.isclose(xticks, xlim[1]))
    assert np.any(np.isclose(yticks, ylim[0]))
    assert np.any(np.isclose(yticks, ylim[1]))
    assert 5 <= len(xticks) - 1 <= 7
    assert 5 <= len(yticks) - 1 <= 7
    assert min(xticks) <= min(obj.x())
    assert max(xticks) >= max(obj.x())
    assert min(yticks) <= min(obj.y())
    assert max(yticks) >= max(obj.y())
    assert ax.xaxis.get_minor_locator().ndivs == 5
    assert ax.yaxis.get_minor_locator().ndivs == 5

    plt.close(ax.figure)
    ecat_module.plotting_style("notebook")


def test_saveant_plot_style_honors_session_minor_tick_default(ecat_module, cv_factory):
    ecat_module.plotting_style("saveant")
    ecat_module.set_defaults("plot", {"minor ticks": 4})
    obj = cv_factory()

    ax = obj.plot({"legend": False, "title": False})

    assert ax.xaxis.get_minor_locator().ndivs == 4
    assert ax.yaxis.get_minor_locator().ndivs == 4

    plt.close(ax.figure)
    ecat_module.set_defaults("plot", {"minor ticks": 2})
    ecat_module.plotting_style("notebook")


def test_saveant_plot_style_separates_suptitle_and_subtitle(ecat_module, cv_factory):
    ecat_module.plotting_style("saveant")
    obj = cv_factory()

    ax = obj.plot({
        "legend": False,
        "title": "Main title",
        "subtitle": "Subtitle line",
    })
    fig = ax.figure

    assert fig._suptitle is not None
    assert fig._suptitle.get_text() == "Main title"
    assert fig.subplotpars.top == pytest.approx(0.86)
    assert fig._suptitle.get_position()[1] == pytest.approx(0.975)
    assert ax.get_title() == "Subtitle line"
    title_gap = fig._suptitle.get_position()[1] - ax.get_position().y1
    assert 0.08 <= title_gap <= 0.13

    plt.close(fig)
    ecat_module.plotting_style("notebook")


def test_saveant_scale_bar_lower_right_shifts_above_inside_xlabel(ecat_module):
    fig, ax = plt.subplots()
    ax.set_xlim(-1, 1)
    ax.set_ylim(-10, 10)

    ecat_module.plotting_style("notebook")
    _x_default, y_default = ecat_module._scale_bar_position(ax, "lower right", 2)

    ecat_module.plotting_style("saveant")
    _x_saveant, y_saveant = ecat_module._scale_bar_position(ax, "lower right", 2)

    assert y_saveant > y_default

    plt.close(fig)
    ecat_module.plotting_style("notebook")


def test_saveant_upper_left_legend_shifts_below_inside_ylabel(ecat_module, cv_factory):
    ecat_module.plotting_style("saveant")
    cvs = _plot_cv_triplet(cv_factory)

    ax = ecat_module.multiplot(cvs, {
        "print": False,
        "title": False,
        "legend": True,
        "legend loc": "upper left",
        "legend outside": False,
    })
    legend = ax.get_legend()
    ax.figure.canvas.draw()
    bbox_axes = legend.get_window_extent(
        ax.figure.canvas.get_renderer()
    ).transformed(ax.transAxes.inverted())

    assert bbox_axes.y1 < 0.93

    plt.close(ax.figure)
    ecat_module.plotting_style("notebook")


def test_saveant_multiplot_snap_bounds_waits_for_all_traces(ecat_module, cv_factory):
    ecat_module.plotting_style("saveant")
    cvs = _plot_cv_triplet(cv_factory)

    ax = ecat_module.multiplot(cvs, {
        "print": False,
        "title": False,
        "legend": False,
        "y unit": "uA",
    })

    line_y = np.concatenate([
        np.asarray(line.get_ydata(), dtype=float)
        for line in ax.lines
    ])
    ymin, ymax = np.nanmin(line_y), np.nanmax(line_y)
    ylim = ax.get_ylim()

    assert min(ylim) <= ymin
    assert max(ylim) >= ymax

    plt.close(ax.figure)
    ecat_module.plotting_style("notebook")


def test_saveant_upper_left_colorbar_panel_shifts_below_inside_ylabel(ecat_module):
    ecat_module.plotting_style("saveant")

    _x0, y0 = ecat_module._resolve_legend_panel_position(
        "upper left",
        panel_width=0.25,
        panel_height=0.30,
        outside=False,
        pad=0.02,
    )

    assert y0 + 0.30 <= 0.86 + 1e-12

    ecat_module.plotting_style("notebook")


def test_cv_plot_segment_color_mode_discrete_defaults_to_groups_of_two(cv_factory):
    obj = _multi_segment_cv(cv_factory, n_segments=4)

    ax = obj.plot(
        {
            "segment color mode": "discrete",
            "legend": True,
            "title": False,
            "plot convention": "IUPAC",
        }
    )
    fig = ax.figure

    assert len(ax.lines) == 4
    assert [line.get_label() for line in ax.lines] == [
        "Segment 1-2",
        "_nolegend_",
        "Segment 3-4",
        "_nolegend_",
    ]
    assert ax.lines[0].get_color() == ax.lines[1].get_color()
    assert ax.lines[2].get_color() == ax.lines[3].get_color()
    assert ax.lines[0].get_color() != ax.lines[2].get_color()
    assert [text.get_text() for text in ax.get_legend().get_texts()] == [
        "Segment 1-2",
        "Segment 3-4",
    ]
    plt.close(fig)


def test_cv_plot_segment_color_groups_absorb_odd_remainder(cv_factory):
    obj = _multi_segment_cv(cv_factory, n_segments=5)

    ax = obj.plot(
        {
            "segment color mode": "discrete",
            "segment color groups": 2,
            "legend": True,
            "title": False,
            "plot convention": "IUPAC",
        }
    )
    fig = ax.figure

    assert [line.get_label() for line in ax.lines] == [
        "Segment 1-2",
        "_nolegend_",
        "Segment 3-5",
        "_nolegend_",
        "_nolegend_",
    ]
    assert len(ax.get_legend().get_texts()) == 2
    plt.close(fig)


def test_cv_plot_segment_color_groups_one_colors_each_segment(cv_factory):
    obj = _multi_segment_cv(cv_factory, n_segments=4)

    ax = obj.plot(
        {
            "segment color mode": "discrete",
            "segment color groups": 1,
            "legend": True,
            "title": False,
            "plot convention": "IUPAC",
        }
    )
    fig = ax.figure

    assert [line.get_label() for line in ax.lines] == [
        "Segment 1",
        "Segment 2",
        "Segment 3",
        "Segment 4",
    ]
    assert len({line.get_color() for line in ax.lines}) == 4
    plt.close(fig)


def test_cv_plot_segment_color_explicit_groups_and_selected_segments(cv_factory):
    obj = _multi_segment_cv(cv_factory, n_segments=5)

    ax = obj.plot(
        {
            "segment color mode": "discrete",
            "segment color groups": [[2], [4, 5]],
            "plot segments": [2, 4, 5],
            "legend": True,
            "title": False,
            "plot convention": "IUPAC",
        }
    )
    fig = ax.figure

    assert [line.get_label() for line in ax.lines] == [
        "Segment 2",
        "Segment 4-5",
        "_nolegend_",
    ]
    assert [text.get_text() for text in ax.get_legend().get_texts()] == [
        "Segment 2",
        "Segment 4-5",
    ]
    plt.close(fig)


def test_cv_plot_segment_color_discrete_gradient_uses_colorbar(cv_factory):
    obj = _multi_segment_cv(cv_factory, n_segments=4)

    ax = obj.plot(
        {
            "segment color mode": "discrete gradient",
            "segment color groups": 1,
            "legend": True,
            "legend mode": "auto",
            "gradient colormap": "viridis",
            "title": False,
            "plot convention": "IUPAC",
        }
    )
    fig = ax.figure

    assert len(ax.lines) == 4
    assert len({line.get_color() for line in ax.lines}) == 4
    assert [line.get_label() for line in ax.lines] == ["_nolegend_"] * 4
    assert len(fig.axes) == 1
    assert len(ax.child_axes) == 1
    assert ax.get_legend() is None
    panel_text = [text.get_text() for text in ax.child_axes[0].texts]
    assert "Segments" in panel_text
    assert "1" in panel_text
    assert "4" in panel_text
    plt.close(fig)


def test_cv_plot_segment_discrete_gradient_colorbar_style_auto_uses_swatch_norm(ecat_module):
    spec = ecat_module.cv._segment_colorbar_spec(
        ["tab:blue", "tab:red", "tab:green"],
        ["Segment 1", "Segment 2", "Segment 3"],
        {"colorbar style": "auto"},
        segment_color_mode="discrete gradient",
    )

    group = spec["gradient groups"][0]
    assert isinstance(group["cmap"], mpl.colors.ListedColormap)
    assert isinstance(group["norm"], mpl.colors.BoundaryNorm)
    assert group["legend context line"] == "Segments"
    assert group["ticklabels"] == ["1", "", "3"]
    assert group["endpoint ticklabels"] == ["1", "3"]


def test_cv_plot_segment_colorbar_style_discrete_uses_swatch_norm(ecat_module):
    spec = ecat_module.cv._segment_colorbar_spec(
        ["tab:blue", "tab:red", "tab:green"],
        ["Segment 1", "Segment 2", "Segment 3"],
        {"colorbar style": "discrete"},
        segment_color_mode="discrete gradient",
    )

    group = spec["gradient groups"][0]
    assert isinstance(group["cmap"], mpl.colors.ListedColormap)
    assert isinstance(group["norm"], mpl.colors.BoundaryNorm)


def test_cv_plot_segment_discrete_colorbar_labels_align_to_ticks(cv_factory):
    obj = _multi_segment_cv(cv_factory, n_segments=4)

    ax = obj.plot(
        {
            "segment color mode": "discrete gradient",
            "segment color groups": 1,
            "colorbar style": "discrete",
            "legend": True,
            "legend mode": "auto",
            "title": False,
            "plot convention": "IUPAC",
        }
    )
    fig = ax.figure

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    panel_ax = ax.child_axes[0]
    cax = panel_ax.child_axes[0]
    labels_by_text = {text.get_text(): text for text in panel_ax.texts}

    for tick_value, label in [(1, "1"), (4, "4")]:
        tick_y = cax.transData.transform((0, tick_value))[1]
        label_bbox = labels_by_text[label].get_window_extent(renderer=renderer)
        label_y = 0.5 * (label_bbox.y0 + label_bbox.y1)
        assert abs(label_y - tick_y) < 2.0

    plt.close(fig)


def test_cv_plot_segment_colorbar_style_continuous_uses_smooth_norm(ecat_module):
    spec = ecat_module.cv._segment_colorbar_spec(
        ["tab:blue", "tab:red", "tab:green"],
        ["Segment 1", "Segment 2", "Segment 3"],
        {"colorbar style": "continuous"},
        segment_color_mode="discrete gradient",
    )

    group = spec["gradient groups"][0]
    assert isinstance(group["cmap"], mpl.colors.LinearSegmentedColormap)
    assert isinstance(group["norm"], mpl.colors.Normalize)
    assert not isinstance(group["norm"], mpl.colors.BoundaryNorm)


def test_custom_colorbar_legend_loc_auto_matches_best(ecat_module):
    fig, ax = plt.subplots()
    ax.plot([0.7, 1.0], [0.7, 1.0], label="_nolegend_")
    color_spec = ecat_module.cv._segment_colorbar_spec(
        ["tab:blue", "tab:red", "tab:green"],
        ["Segment 1", "Segment 2", "Segment 3"],
        {"colorbar style": "continuous"},
        segment_color_mode="discrete gradient",
    )
    base_options = {
        "legend mode": "auto",
        "legend outside": False,
        "legend pad": 0.02,
        "legend sample length": "auto",
        "colorbar height scale": 1.0,
        "colorbar reverse": True,
        "colorbar tick length": 5,
        "colorbar tick pad": 8,
        "colorbar tick labels": "endpoints",
        "colorbar trace ticks": True,
    }

    best = ecat_module._resolve_adaptive_legend_layout(
        ax,
        color_spec,
        {**base_options, "legend loc": "best", "legend fontsize": "auto"},
    )
    auto = ecat_module._resolve_adaptive_legend_layout(
        ax,
        color_spec,
        {**base_options, "legend loc": "auto", "legend fontsize": "auto"},
    )

    assert auto == best
    plt.close(fig)


def test_cv_plot_segment_color_continuous_gradient_uses_line_collection(cv_factory):
    obj = _multi_segment_cv(cv_factory, n_segments=4)

    ax = obj.plot(
        {
            "segment color mode": "continuous gradient",
            "segment color groups": 1,
            "legend": True,
            "title": False,
            "plot convention": "IUPAC",
        }
    )
    fig = ax.figure

    collections = [collection for collection in ax.collections if isinstance(collection, LineCollection)]
    assert len(ax.lines) == 0
    assert len(collections) == 1
    assert len(fig.axes) == 1
    assert len(ax.child_axes) == 1
    plt.close(fig)


def test_cv_plot_segment_color_legend_false_suppresses_colorbar(cv_factory):
    obj = _multi_segment_cv(cv_factory, n_segments=4)

    ax = obj.plot(
        {
            "segment color mode": "discrete gradient",
            "legend": False,
            "title": False,
            "plot convention": "IUPAC",
        }
    )
    fig = ax.figure

    assert len(fig.axes) == 1
    assert len(ax.child_axes) == 0
    assert ax.get_legend() is None
    plt.close(fig)


def test_cv_plot_segment_color_auto_defaults_to_discrete_gradient_above_three_segments(cv_factory):
    obj = _multi_segment_cv(cv_factory, n_segments=4)

    ax = obj.plot({"title": False, "plot convention": "IUPAC"})
    fig = ax.figure

    assert len(ax.lines) == 4
    assert len({line.get_color() for line in ax.lines}) == 2
    assert len(ax.child_axes) == 1
    assert ax.get_legend() is None
    plt.close(fig)


def test_cv_plot_segment_colorbar_is_drawn_after_axis_flips(ecat_module, cv_factory, monkeypatch):
    obj = _multi_segment_cv(cv_factory, n_segments=4)
    observed = {}
    original_draw = ecat_module._draw_multiplot_legend_and_colorbars

    def capture_draw(ax, color_spec, options, legend_fs):
        observed["x_inverted_at_draw"] = ax.xaxis_inverted()
        return original_draw(ax, color_spec, options, legend_fs)

    monkeypatch.setattr(ecat_module, "_draw_multiplot_legend_and_colorbars", capture_draw)

    ax = obj.plot(
        {
            "segment color mode": "discrete gradient",
            "segment color groups": 1,
            "legend": True,
            "legend mode": "auto",
            "title": False,
            "plot convention": "US",
        }
    )
    fig = ax.figure

    assert observed["x_inverted_at_draw"] == True
    assert ax.xaxis_inverted()
    plt.close(fig)


def test_multiplot_does_not_apply_cv_segment_coloring(ecat_module, cv_factory):
    objects = [_multi_segment_cv(cv_factory, n_segments=4) for _ in range(2)]
    objects[0].name = "cv_a"
    objects[1].name = "cv_b"

    ax = ecat_module.multiplot(
        objects,
        {"legend": False, "title": False, "plot convention": "IUPAC"},
    )

    fig = ax.figure
    assert len(ax.lines) == 2
    assert len(ax.child_axes) == 0
    plt.close(fig)


def test_cv_plot_segment_color_auto_preserves_single_line_for_three_or_fewer_segments(cv_factory):
    obj = _multi_segment_cv(cv_factory, n_segments=3)

    ax = obj.plot({"title": False, "plot convention": "IUPAC"})
    fig = ax.figure

    assert len(ax.lines) == 1
    assert len(ax.child_axes) == 0
    assert ax.get_legend() is None
    plt.close(fig)


def test_cv_plot_legend_auto_suppresses_single_entry_legend(cv_factory):
    obj = cv_factory()

    ax = obj.plot({"legend": "auto", "title": False})
    fig = ax.figure

    assert len(ax.lines) == 1
    assert ax.get_legend() is None
    plt.close(fig)


def test_cv_plot_accepts_plot_options_dataclass(ecat_module, cv_factory):
    obj = cv_factory()
    options = ecat_module.PlotOptions.from_options({"legend": False, "title": False})

    ax = obj.plot(options)
    fig = ax.figure

    assert len(ax.lines) == 1
    assert ax.get_title() == ""
    plt.close(fig)


def test_cv_plot_accepts_derivative_and_scale_options(cv_factory):
    obj = cv_factory()

    ax0 = obj.plot({"plot segment": 1, "derivative": 0, "new plot": False, "title": False})
    fig0 = ax0.figure
    ax1 = obj.plot(
        {
            "plot segments": 1,
            "derivative": 1,
            "y scale": 0.1,
            "color": "blue",
            "new plot": False,
            "title": False,
        },
    )
    fig1 = ax1.figure
    ax2 = obj.plot(
        {
            "plot segments": 1,
            "derivative": 2,
            "y scale": 5,
            "color": "red",
            "new plot": False,
            "title": False,
        },
    )
    fig2 = ax2.figure

    assert ax0 is ax1 is ax2
    assert len(ax1.lines) == 3
    assert len(ax2.lines) == 3
    assert ax2.lines[-2].get_color() == "blue"
    assert ax2.lines[-1].get_color() == "red"
    plt.close(fig0)
    plt.close(fig1)
    plt.close(fig2)


def test_cv_plot_labels_first_derivative_axis(cv_factory):
    obj = cv_factory()

    ax = obj.plot(
        {
            "plot segments": 1,
            "derivative": 1,
            "noise window": 5,
            "noise polyorder": 2,
            "title": False,
        },
    )
    fig = ax.figure

    assert "di/dE" in ax.get_ylabel()
    assert "/V" in ax.get_ylabel()
    plt.close(fig)


def test_cv_plot_labels_second_derivative_axis(cv_factory):
    obj = cv_factory()

    ax = obj.plot(
        {
            "plot segments": 1,
            "derivative": 2,
            "noise window": 5,
            "noise polyorder": 2,
            "title": False,
        },
    )
    fig = ax.figure

    assert "d$^2$i/dE$^2$" in ax.get_ylabel()
    assert "/V$^2$" in ax.get_ylabel()
    plt.close(fig)


def test_cv_plot_rejects_unknown_option_with_suggestion(ecat_module, cv_factory):
    obj = cv_factory()

    with pytest.raises(ecat_module.OptionError, match="legend"):
        obj.plot({"legned": False})


def test_cv_plot_default_title_matches_multiplot_auto_style(cv_factory):
    obj = cv_factory()

    ax = obj.plot({"legend": False})
    fig = ax.figure

    assert fig._suptitle is not None
    assert fig._suptitle.get_text() == "10 mM Fc"
    assert fig._suptitle.get_position()[1] == pytest.approx(0.98)
    assert ax.get_title() == "MeCN, CO$_2$, 50 mV/s, 2 seg."
    plt.close(fig)


def test_cv_plot_reference_axis_label_uses_compact_redox_mathtext(cv_factory):
    obj = cv_factory()
    obj.potential_shift({"shift guess": 0.0, "shift label": "Fc/Fc+"})

    ax = obj.plot({"legend": False, "title": False})

    assert ax.get_xlabel() == r"Potential (V vs $\mathrm{Fc/Fc^{+}}$)"
    plt.close(ax.figure)


def test_cv_plot_explicit_title_can_use_cv_name(cv_factory):
    obj = cv_factory()

    ax = obj.plot({"legend": False, "title": obj.name})
    fig = ax.figure

    assert fig._suptitle is None
    assert ax.get_title() == obj.name
    plt.close(fig)


def test_cv_plot_false_title_remains_untitled(cv_factory):
    obj = cv_factory()

    ax = obj.plot({"legend": False, "title": False})
    fig = ax.figure

    assert fig._suptitle is None
    assert ax.get_title() == ""
    plt.close(fig)


def test_cv_current_density_is_plot_time_y_axis_view(cv_factory):
    obj = cv_factory(options={"electrode area": 2.0, "current density": True})

    assert "Current Density" not in " ".join(str(col) for col in obj.data.columns)

    density = obj.y({"y axis": "current density"})

    np.testing.assert_allclose(density.to_numpy(), obj.y().to_numpy() / 2.0)
    assert density.name == "Current Density"
    assert "Current Density" not in " ".join(str(col) for col in obj.data.columns)


def test_cv_current_density_requires_electrode_area(cv_factory):
    obj = cv_factory()

    with pytest.raises(ValueError, match="electrode area"):
        obj.y({"y axis": "current density"})


def test_cv_ip0_is_plot_time_y_axis_view(cv_factory):
    obj = cv_factory()

    normalized = obj.y({"y axis": "i/ip0", "ip0": 2e-6})

    np.testing.assert_allclose(normalized.to_numpy(), obj.y().to_numpy() / 2e-6)
    assert normalized.name == "i/ip0"
    assert "i/ip0" not in obj.data.columns


def test_cv_plot_can_use_ip0_y_axis(cv_factory):
    obj = cv_factory()

    ax = obj.plot(
        {
            "y axis": "i/ip0",
            "ip0": 2e-6,
            "legend": False,
            "title": False,
        },
    )
    fig = ax.figure

    np.testing.assert_allclose(ax.lines[0].get_ydata(), obj.y().to_numpy() / 2e-6)
    assert ax.get_ylabel() == "$i / i_p^0$"
    plt.close(fig)


def test_cv_ip0_requires_nonzero_reference_current(cv_factory):
    obj = cv_factory()

    with pytest.raises(ValueError, match="ip0"):
        obj.y({"y axis": "i/ip0"})

    with pytest.raises(ValueError, match="ip0"):
        obj.y({"y axis": "i/ip0", "ip0": 0})


def test_multiplot_returns_axes_and_draws_all_curves(ecat_module, cv_factory):
    objects = _plot_cv_triplet(cv_factory)

    ax = ecat_module.multiplot(
        objects,
        {"print": False, "title": False, "legend": True},
    )

    fig = ax.figure

    assert isinstance(ax, mpl.axes.Axes)
    assert len(ax.lines) == 3
    assert fig._suptitle is None
    plt.close(fig)


def test_multiplot_draws_custom_subtitle(ecat_module, cv_factory):
    objects = _plot_cv_triplet(cv_factory)[:2]

    ax = ecat_module.multiplot(
        objects,
        {
            "print": False,
            "title": "CO$_2$ series",
            "subtitle": "Custom labels, outside legend, and scale bar",
            "legend": False,
            "plot convention": "IUPAC",
        },
    )

    fig = ax.figure

    assert fig._suptitle.get_text() == "CO$_2$ series"
    assert ax.get_title() == "Custom labels, outside legend, and scale bar"
    plt.close(fig)


def test_multiplot_us_plot_convention_inverts_cv_axis(ecat_module, cv_factory):
    objects = _plot_cv_triplet(cv_factory)[:2]

    ax = ecat_module.multiplot(
        objects,
        {
            "print": False,
            "title": False,
            "legend": False,
            "plot convention": "US",
        },
    )

    assert ax.xaxis_inverted()
    plt.close(ax.figure)


def test_multiplot_accepts_multiplot_options_dataclass(ecat_module, cv_factory):
    objects = _plot_cv_triplet(cv_factory)[:2]
    options = ecat_module.MultiplotOptions.from_options(
        {"print": False, "title": False, "legend": False}
    )

    ax = ecat_module.multiplot(objects, options)

    fig = ax.figure

    assert isinstance(ax, mpl.axes.Axes)
    assert len(ax.lines) == 2
    plt.close(fig)


def test_multiplot_rejects_unknown_option_with_suggestion(ecat_module, cv_factory):
    objects = _plot_cv_triplet(cv_factory)[:2]

    with pytest.raises(ecat_module.OptionError, match="legend"):
        ecat_module.multiplot(objects, {"legned": False})


def test_multiplot_can_plot_current_density_y_axis(ecat_module, cv_factory):
    objects = _plot_cv_triplet(cv_factory)[:2]
    for obj in objects:
        obj.electrode_area = 2.0

    ecat_module.multiplot(
        objects,
        {
            "print": False,
            "title": False,
            "legend": False,
            "y axis": "current density",
        },
    )

    fig = plt.gcf()
    ax = plt.gca()

    np.testing.assert_allclose(
        ax.lines[0].get_ydata(),
        objects[0].y().to_numpy() / 2.0 * 1e6,
    )
    assert ax.get_ylabel() == "Current Density (μA/cm$^2$)"
    plt.close(fig)


def test_multiplot_plain_current_rejects_density_y_unit(ecat_module, cv_factory):
    objects = _plot_cv_triplet(cv_factory)[:2]

    with pytest.raises(ValueError, match="incompatible units"):
        ecat_module.multiplot(
            objects,
            {
                "print": False,
                "title": False,
                "legend": False,
                "y axis": "Current",
                "y unit": "uA/cm^2",
            },
        )
    plt.close("all")


def test_cv_plot_current_density_auto_scales_y_unit(cv_factory):
    obj = cv_factory(options={"electrode area": 2.0})

    ax = obj.plot(
        {
            "y axis": "current density",
            "legend": False,
            "title": False,
        }
    )
    fig = ax.figure

    np.testing.assert_allclose(
        ax.lines[0].get_ydata(),
        obj.y().to_numpy() / 2.0 * 1e6,
    )
    assert ax.get_ylabel() == "Current Density (μA/cm$^2$)"
    plt.close(fig)


def test_symbol_labels_option_uses_physical_axis_symbols(ecat_module, cv_factory):
    obj = cv_factory(options={"electrode area": 2.0})

    ax = obj.plot(
        {
            "y axis": "current density",
            "symbol labels": True,
            "legend": False,
            "title": False,
        }
    )

    assert ax.get_xlabel() == "$E$ (V)"
    assert ax.get_ylabel() == "$j$ (μA/cm$^2$)"
    assert ecat_module.echem.format_axis_label("Time", "s", symbol_labels=True) == "$t$ (s)"
    assert ecat_module.echem.format_axis_label("Charge", "C", symbol_labels=True) == "$Q$ (C)"
    plt.close(ax.figure)


def test_saveant_auto_symbol_labels_include_current_density(ecat_module, cv_factory):
    ecat_module.plotting_style("saveant")
    obj = cv_factory(options={"electrode area": 2.0})

    ax = obj.plot({
        "y axis": "current density",
        "legend": False,
        "title": False,
    })

    inside_labels = [text.get_text() for text in ax.texts]

    assert "$E$ (V)" in inside_labels
    assert "$j$ (μA/cm$^2$)" in inside_labels

    plt.close(ax.figure)
    ecat_module.plotting_style("notebook")


def test_cv_plot_current_density_accepts_area_unit_shorthand(cv_factory):
    obj = cv_factory(options={"electrode area": 2.0})

    ax = obj.plot(
        {
            "y axis": "current density",
            "y unit": "mm",
            "legend": False,
            "title": False,
        }
    )
    fig = ax.figure

    np.testing.assert_allclose(
        ax.lines[0].get_ydata(),
        obj.y().to_numpy() / 2.0 * 1e7,
    )
    assert ax.get_ylabel() == "Current Density (nA/mm$^2$)"
    plt.close(fig)


def test_cv_plot_current_density_accepts_full_unit_without_auto_rescaling(cv_factory):
    obj = cv_factory(options={"electrode area": 2.0})

    ax = obj.plot(
        {
            "y axis": "current density",
            "y unit": "mA/mm^2",
            "legend": False,
            "title": False,
        }
    )
    fig = ax.figure

    np.testing.assert_allclose(
        ax.lines[0].get_ydata(),
        obj.y().to_numpy() / 2.0 * 10,
    )
    assert ax.get_ylabel() == "Current Density (mA/mm$^2$)"
    plt.close(fig)


def test_multiplot_can_plot_ip0_y_axis_with_per_cv_values(ecat_module, cv_factory):
    objects = _plot_cv_triplet(cv_factory)[:2]

    ecat_module.multiplot(
        objects,
        {
            "print": False,
            "title": False,
            "legend": False,
            "y axis": "i/ip0",
            "ip0": [1e-6, 2e-6],
        },
    )

    fig = plt.gcf()
    ax = plt.gca()

    np.testing.assert_allclose(ax.lines[0].get_ydata(), objects[0].y().to_numpy() / 1e-6)
    np.testing.assert_allclose(ax.lines[1].get_ydata(), objects[1].y().to_numpy() / 2e-6)
    assert ax.get_ylabel() == "$i / i_p^0$"
    plt.close(fig)


def test_multiplot_can_plot_ip0_y_axis_with_broadcast_ip0(
    ecat_module,
    cv_factory,
):
    objects = _plot_cv_triplet(cv_factory)[:2]

    ecat_module.multiplot(
        objects,
        {
            "print": False,
            "title": False,
            "legend": False,
            "y axis": "i/ip0",
            "ip0": 2e-6,
        },
    )

    fig = plt.gcf()
    ax = plt.gca()

    np.testing.assert_allclose(ax.lines[0].get_ydata(), objects[0].y().to_numpy() / 2e-6)
    np.testing.assert_allclose(ax.lines[1].get_ydata(), objects[1].y().to_numpy() / 2e-6)
    plt.close(fig)


def test_multiplot_existing_ip0_column_does_not_resolve_reference_again(
    ecat_module,
    cv_factory,
):
    objects = _plot_cv_triplet(cv_factory)[:2]
    normalized = ecat_module.normalize_current(objects, {"ip0": 2e-6})
    ref = cv_factory(name="100mVs_ref_run01")

    def fail_peak_current(options=None):
        raise AssertionError("multiplot should use the existing i/ip0 column")

    ref.peak_current = fail_peak_current

    ecat_module.multiplot(
        normalized,
        {
            "print": False,
            "title": False,
            "legend": False,
            "y axis": "i/ip0",
            "reference cv": ref,
        },
    )

    fig = plt.gcf()
    ax = plt.gca()

    np.testing.assert_allclose(ax.lines[0].get_ydata(), objects[0].y().to_numpy() / 2e-6)
    np.testing.assert_allclose(ax.lines[1].get_ydata(), objects[1].y().to_numpy() / 2e-6)
    assert ax.get_ylabel() == "$i / i_p^0$"
    plt.close(fig)


def test_normalize_current_output_can_be_multiplotted_without_ip0_options(
    ecat_module,
    cv_factory,
):
    objects = _plot_cv_triplet(cv_factory)[:2]
    normalized = ecat_module.normalize_current(objects, {"ip0": 2e-6})

    ecat_module.multiplot(
        normalized,
        {
            "print": False,
            "title": False,
            "legend": False,
            "y axis": "i/ip0",
        },
    )

    fig = plt.gcf()
    ax = plt.gca()

    np.testing.assert_allclose(ax.lines[0].get_ydata(), objects[0].y().to_numpy() / 2e-6)
    np.testing.assert_allclose(ax.lines[1].get_ydata(), objects[1].y().to_numpy() / 2e-6)
    plt.close(fig)


def test_normalize_current_returns_copied_single_cv(ecat_module, cv_factory):
    obj = cv_factory()

    normalized = ecat_module.normalize_current(obj, {"ip0": 2e-6, "print": False})

    assert isinstance(normalized, ecat_module.cv)
    assert normalized is not obj
    assert "i/ip0" in normalized.data.columns
    assert "i/ip0" not in obj.data.columns
    np.testing.assert_allclose(normalized.y({"y axis": "i/ip0"}), obj.y().to_numpy() / 2e-6)


def test_normalize_current_accepts_option_dataclass(ecat_module, cv_factory):
    obj = cv_factory()
    options = ecat_module.NormalizationOptions.from_options({"ip0": 2e-6, "print": False})

    normalized = ecat_module.normalize_current(obj, options)

    assert "i/ip0" in normalized.data.columns
    np.testing.assert_allclose(normalized.y({"y axis": "i/ip0"}), obj.y().to_numpy() / 2e-6)


def test_normalize_current_rejects_unknown_option_with_suggestion(ecat_module, cv_factory):
    obj = cv_factory()

    with pytest.raises(ecat_module.OptionError, match="reference cv"):
        ecat_module.normalize_current(obj, {"referenc cv": obj})


def test_normalize_current_uses_reference_cv_for_ip0_without_returning_reference(
    ecat_module,
    cv_factory,
):
    objects = _plot_cv_triplet(cv_factory)[:2]
    ref = cv_factory(name="100mVs_ref_run01")
    ref.peak_current = lambda options=None: {"ip": 2e-6, "tangent line": [0, 0], "tangent start": 0}

    normalized = ecat_module.normalize_current(
        objects,
        {
            "reference cv": ref,
            "segment": 1,
            "print": False,
        },
    )

    assert isinstance(normalized, list)
    assert len(normalized) == 2
    assert all(item is not ref for item in normalized)
    np.testing.assert_allclose(normalized[0].y({"y axis": "i/ip0"}), objects[0].y().to_numpy() / 2e-6)
    np.testing.assert_allclose(normalized[1].y({"y axis": "i/ip0"}), objects[1].y().to_numpy() / 2e-6)


def test_normalize_current_prints_summary_by_default(ecat_module, cv_factory, capsys):
    obj = cv_factory(name="100mVs_sample_run01")

    ecat_module.normalize_current(obj, {"ip0": 2e-6})

    printed = capsys.readouterr().out
    assert "Current Normalization:" in printed
    assert "100mVs_sample_run01" in printed
    assert "ip0" in printed
    assert "manual ip0" in printed


def test_normalize_current_shared_ip0_print_collapses_repeated_rows(
    ecat_module,
    cv_factory,
    capsys,
):
    objects = _plot_cv_triplet(cv_factory)[:2]

    ecat_module.normalize_current(
        objects,
        {
            "ip0": 2e-6,
            "pretty print": False,
        },
    )

    printed = capsys.readouterr().out
    assert "Current Normalization:" in printed
    assert "CVs: 2" in printed
    assert "ip0: 2e-06 A" in printed
    assert "Source: manual ip0" in printed
    assert "50mVs_sample_CO2_MeCN_10mM_Fc_run01" not in printed
    assert "100mVs_sample_N2_DMF_5mM_Fc_run01" not in printed


def test_normalize_current_plot_all_shows_normalized_multiplot(ecat_module, cv_factory):
    objects = _plot_cv_triplet(cv_factory)[:2]

    normalized = ecat_module.normalize_current(
        objects,
        {
            "ip0": 2e-6,
            "print": False,
            "plot all": True,
            "legend": False,
            "title": False,
        },
    )

    fig = plt.gcf()
    ax = plt.gca()
    assert len(ax.lines) == 2
    assert ax.get_ylabel() == "$i / i_p^0$"
    np.testing.assert_allclose(ax.lines[0].get_ydata(), normalized[0].y().to_numpy())
    np.testing.assert_allclose(ax.lines[1].get_ydata(), normalized[1].y().to_numpy())
    plt.close(fig)


def test_ip0_and_reference_cv_conflict_is_explicit(ecat_module, cv_factory):
    obj = cv_factory()

    with pytest.raises(ecat_module.OptionError, match="only one"):
        obj.y({"y axis": "i/ip0", "ip0": 1e-6, "reference cv": obj})


def _seed_minimal_echem_object(obj, name, data, units):
    obj.filepath = None
    obj.options = {}
    obj.timestamp = None
    obj.creation_time = None
    obj.modification_time = None
    obj.name = name
    obj.data = data
    obj.software = None
    obj.num_x_cols = 1
    obj.temperature = 298
    obj.electrode_area = 0
    obj.delta_x = None
    obj.units = units
    obj.segments = 1
    obj.reference_shift = None
    obj.reference_label = None
    obj.reference_mode = "none"
    obj.reference_source_file = None
    obj.reference_failure_message = None
    obj.ir_comp_resistance = None
    obj.ir_uncomp_resistance = None
    obj.ir_comp_percent = None


def _minimal_dpv(ecat_module, name, gas):
    data = pd.DataFrame(
        {
            "Potential": [-1.0, -0.5, 0.0],
            "Current": [0.0, -1.0e-6, 0.0],
        }
    )
    obj = ecat_module.dpv.__new__(ecat_module.dpv)
    _seed_minimal_echem_object(obj, name, data, {"Potential": "V", "Current": "A"})
    obj.type = "Differential Pulse Voltammetry"
    obj.gas = gas
    obj.solvent = "MeCN"
    obj.compounds = ["TBAPF6"]
    obj.concentrations = ["0.1 M"]
    obj.init_E = -1.0
    obj.final_E = 0.0
    obj.incr_E = 0.01
    obj.amplitude = 0.05
    obj.pulse_width = 0.05
    obj.sample_width = 0.02
    obj.pulse_period = 0.5
    obj.quiet_time = 2.0
    obj.sensitivity = None
    obj.comp_R = None
    obj.min_E = -1.0
    obj.max_E = 0.0
    return obj


def _minimal_ca(ecat_module, name, gas):
    data = pd.DataFrame(
        {
            "Time": [0.0, 1.0, 2.0],
            "Current": [1.0e-6, 0.8e-6, 0.6e-6],
        }
    )
    obj = ecat_module.ca.__new__(ecat_module.ca)
    _seed_minimal_echem_object(obj, name, data, {"Time": "s", "Current": "A"})
    obj.type = "Chronoamperometry"
    obj.gas = gas
    obj.solvent = "MeCN"
    obj.compounds = ["TBAPF6"]
    obj.concentrations = ["0.1 M"]
    obj.init_E = -1.2
    obj.run_time = 2.0
    obj.sample_interval = 1.0
    return obj


def _minimal_cp(ecat_module, name, gas):
    data = pd.DataFrame(
        {
            "Time": [0.0, 1.0, 2.0],
            "Potential": [-0.2, -0.3, -0.4],
        }
    )
    obj = ecat_module.cp.__new__(ecat_module.cp)
    _seed_minimal_echem_object(obj, name, data, {"Time": "s", "Potential": "V"})
    obj.type = "Chronopotentiometry"
    obj.gas = gas
    obj.solvent = "MeCN"
    obj.compounds = ["TBAPF6"]
    obj.concentrations = ["0.1 M"]
    obj.segments = 2
    obj.low_E_limit = -1.0
    obj.high_E_limit = 0.0
    return obj


def test_multiplot_accepts_plot_labels_alias(ecat_module, cv_factory):
    objects = _plot_cv_triplet(cv_factory)[:2]

    ax = ecat_module.multiplot(
        objects,
        {
            "print": False,
            "title": False,
            "legend": True,
            "plot labels": ["Ar", "CO$_2$"],
        },
    )

    assert [text.get_text() for text in ax.get_legend().get_texts()] == ["Ar", "CO$_2$"]
    plt.close(ax.figure)


def test_multiplot_auto_labels_dpv_by_differing_metadata(ecat_module):
    objects = [
        _minimal_dpv(ecat_module, "dpv_ar", "Ar"),
        _minimal_dpv(ecat_module, "dpv_co2", "CO2"),
    ]
    options = ecat_module.MultiplotOptions.from_options(
        {"print": False, "title": False, "legend": True}
    ).to_legacy_dict()

    style = ecat_module._prepare_multiplot_style(objects, options)

    assert style["display labels"] == ["Ar", "CO2"]
    plt.close(style["ax"].figure)


def test_multiplot_auto_labels_ca_and_cp_by_differing_metadata(ecat_module):
    cases = [
        [_minimal_ca(ecat_module, "ca_ar", "Ar"), _minimal_ca(ecat_module, "ca_co2", "CO2")],
        [_minimal_cp(ecat_module, "cp_ar", "Ar"), _minimal_cp(ecat_module, "cp_co2", "CO2")],
    ]
    options = ecat_module.MultiplotOptions.from_options(
        {"print": False, "title": False, "legend": True}
    ).to_legacy_dict()

    for objects in cases:
        style = ecat_module._prepare_multiplot_style(objects, options)
        assert style["display labels"] == ["Ar", "CO2"]
        plt.close(style["ax"].figure)


def test_multiplot_legend_toggle_controls_legend_presence(ecat_module, cv_factory):
    objects = _plot_cv_triplet(cv_factory)[:2]

    ecat_module.multiplot(
        objects,
        {"print": False, "title": False, "legend": True},
    )
    fig_with_legend = plt.gcf()
    ax_with_legend = plt.gca()

    legend = ax_with_legend.get_legend()
    assert legend is not None
    assert len(legend.get_texts()) == 2
    plt.close(fig_with_legend)

    ecat_module.multiplot(
        objects,
        {"print": False, "title": False, "legend": False},
    )
    fig_without_legend = plt.gcf()
    ax_without_legend = plt.gca()

    assert ax_without_legend.get_legend() is None
    plt.close(fig_without_legend)


def test_multiplot_deduplicate_labels_true_uses_scan_window_only_when_segments_match(
    ecat_module,
    cv_factory,
):
    objects = [
        cv_factory(name="100mVs_sample_CO2_MeCN_10mM_Fc_run01"),
        cv_factory(name="100mVs_sample_CO2_MeCN_10mM_Fc_run02"),
        cv_factory(name="100mVs_sample_Ar_MeCN_10mM_Fc_run03"),
    ]
    objects[0].min_E, objects[0].max_E = -2, 1
    objects[1].min_E, objects[1].max_E = -2.25, 1
    objects[2].min_E, objects[2].max_E = -2, 1
    objects[0].segments = 3
    objects[1].segments = 3
    objects[2].segments = 3
    options = ecat_module.MultiplotOptions.from_options(
        {"print": False, "title": False, "legend": True, "deduplicate labels": True}
    ).to_legacy_dict()

    style = ecat_module._prepare_multiplot_style(objects, options)

    assert style["display labels"] == [
        "CO2 ([-2, 1])",
        "CO2 ([-2.25, 1])",
        "Ar",
    ]


def test_multiplot_deduplicate_labels_true_uses_segments_when_segments_differ(
    ecat_module,
    cv_factory,
):
    objects = [
        cv_factory(name="100mVs_sample_CO2_MeCN_10mM_Fc_run01"),
        cv_factory(name="100mVs_sample_CO2_MeCN_10mM_Fc_run02"),
        cv_factory(name="100mVs_sample_Ar_MeCN_10mM_Fc_run03"),
    ]
    objects[0].min_E, objects[0].max_E = -2, 1
    objects[1].min_E, objects[1].max_E = -2, 1
    objects[2].min_E, objects[2].max_E = -2, 1
    objects[0].segments = 3
    objects[1].segments = 9
    objects[2].segments = 3
    options = ecat_module.MultiplotOptions.from_options(
        {"print": False, "title": False, "legend": True, "deduplicate labels": True}
    ).to_legacy_dict()

    style = ecat_module._prepare_multiplot_style(objects, options)

    assert style["display labels"] == [
        "CO2 (3 seg.)",
        "CO2 (9 seg.)",
        "Ar",
    ]


def test_multiplot_deduplicate_labels_falls_back_to_replicate_numbering(
    ecat_module,
    cv_factory,
):
    objects = [
        cv_factory(name="100mVs_sample_CO2_MeCN_10mM_Fc_run01"),
        cv_factory(name="100mVs_sample_CO2_MeCN_10mM_Fc_run02"),
        cv_factory(name="100mVs_sample_Ar_MeCN_10mM_Fc_run03"),
    ]
    objects[0].min_E, objects[0].max_E = -2, 1
    objects[1].min_E, objects[1].max_E = -2, 1
    objects[2].min_E, objects[2].max_E = -2, 1
    objects[0].segments = 3
    objects[1].segments = 3
    objects[2].segments = 3
    options = ecat_module.MultiplotOptions.from_options(
        {"print": False, "title": False, "legend": True, "deduplicate labels": True}
    ).to_legacy_dict()

    style = ecat_module._prepare_multiplot_style(objects, options)

    assert style["display labels"] == [
        "CO2 (rep 1)",
        "CO2 (rep 2)",
        "Ar",
    ]


def test_multiplot_deduplicate_labels_accepts_field_list(ecat_module, cv_factory):
    objects = [
        cv_factory(name="100mVs_sample_CO2_MeCN_10mM_Fc_run01"),
        cv_factory(name="100mVs_sample_CO2_MeCN_10mM_Fc_run02"),
        cv_factory(name="100mVs_sample_Ar_MeCN_10mM_Fc_run03"),
    ]
    objects[0].min_E, objects[0].max_E = -2, 1
    objects[1].min_E, objects[1].max_E = -2.25, 1
    objects[2].min_E, objects[2].max_E = -2, 1
    objects[0].segments = 3
    objects[1].segments = 9
    objects[2].segments = 3
    options = ecat_module.MultiplotOptions.from_options(
        {
            "print": False,
            "title": False,
            "legend": True,
            "deduplicate labels": ["segments"],
        }
    ).to_legacy_dict()

    style = ecat_module._prepare_multiplot_style(objects, options)

    assert style["display labels"] == [
        "CO2 (3 seg.)",
        "CO2 (9 seg.)",
        "Ar",
    ]


def test_multiplot_duplicate_labels_warn_by_default(ecat_module, cv_factory, capsys):
    objects = [
        cv_factory(name="100mVs_sample_CO2_MeCN_10mM_Fc_run01"),
        cv_factory(name="100mVs_sample_CO2_MeCN_10mM_Fc_run02"),
        cv_factory(name="100mVs_sample_Ar_MeCN_10mM_Fc_run03"),
    ]
    objects[0].min_E, objects[0].max_E = -2, 1
    objects[1].min_E, objects[1].max_E = -2.25, 1
    objects[2].min_E, objects[2].max_E = -2, 1
    options = ecat_module.MultiplotOptions.from_options(
        {"print": False, "title": False, "legend": True}
    ).to_legacy_dict()

    style = ecat_module._prepare_multiplot_style(objects, options)
    captured = capsys.readouterr()

    assert style["display labels"] == ["CO2", "CO2", "Ar"]
    assert "Duplicate multiplot labels detected" in captured.out
    assert '"CO2"' in captured.out
    assert "'deduplicate labels': True" in captured.out


def test_multiplot_duplicate_label_warning_is_suppressed_by_explicit_false(
    ecat_module,
    cv_factory,
    capsys,
):
    objects = [
        cv_factory(name="100mVs_sample_CO2_MeCN_10mM_Fc_run01"),
        cv_factory(name="100mVs_sample_CO2_MeCN_10mM_Fc_run02"),
        cv_factory(name="100mVs_sample_Ar_MeCN_10mM_Fc_run03"),
    ]
    options = ecat_module.MultiplotOptions.from_options(
        {
            "print": False,
            "title": False,
            "legend": True,
            "deduplicate labels": False,
        }
    ).to_legacy_dict()

    style = ecat_module._prepare_multiplot_style(objects, options)
    captured = capsys.readouterr()

    assert style["display labels"] == ["CO2", "CO2", "Ar"]
    assert "Duplicate multiplot labels detected" not in captured.out


def test_multiplot_duplicate_label_warning_is_not_emitted_when_deduplicated(
    ecat_module,
    cv_factory,
    capsys,
):
    objects = [
        cv_factory(name="100mVs_sample_CO2_MeCN_10mM_Fc_run01"),
        cv_factory(name="100mVs_sample_CO2_MeCN_10mM_Fc_run02"),
        cv_factory(name="100mVs_sample_Ar_MeCN_10mM_Fc_run03"),
    ]
    objects[0].min_E, objects[0].max_E = -2, 1
    objects[1].min_E, objects[1].max_E = -2.25, 1
    objects[2].min_E, objects[2].max_E = -2, 1
    options = ecat_module.MultiplotOptions.from_options(
        {
            "print": False,
            "title": False,
            "legend": True,
            "deduplicate labels": True,
        }
    ).to_legacy_dict()

    style = ecat_module._prepare_multiplot_style(objects, options)
    captured = capsys.readouterr()

    assert style["display labels"] == ["CO2 ([-2, 1])", "CO2 ([-2.25, 1])", "Ar"]
    assert "Duplicate multiplot labels detected" not in captured.out


@pytest.mark.parametrize("deduplicate_field", ["auto", "true"])
def test_multiplot_deduplicate_labels_string_modes_are_invalid_fields(
    ecat_module,
    cv_factory,
    capsys,
    deduplicate_field,
):
    objects = [
        cv_factory(name="100mVs_sample_CO2_MeCN_10mM_Fc_run01"),
        cv_factory(name="100mVs_sample_CO2_MeCN_10mM_Fc_run02"),
        cv_factory(name="100mVs_sample_Ar_MeCN_10mM_Fc_run03"),
    ]
    options = ecat_module.MultiplotOptions.from_options(
        {
            "print": False,
            "title": False,
            "legend": True,
            "deduplicate labels": deduplicate_field,
        }
    ).to_legacy_dict()

    style = ecat_module._prepare_multiplot_style(objects, options)
    captured = capsys.readouterr()

    assert style["display labels"] == ["CO2", "CO2", "Ar"]
    assert "Invalid deduplicate label key" in captured.out
    assert deduplicate_field in captured.out
    assert "Valid deduplicate label options are:" in captured.out


def test_multiplot_deduplicate_labels_invalid_field_prints_valid_options(
    ecat_module,
    cv_factory,
    capsys,
):
    objects = [
        cv_factory(name="100mVs_sample_CO2_MeCN_10mM_Fc_run01"),
        cv_factory(name="100mVs_sample_CO2_MeCN_10mM_Fc_run02"),
        cv_factory(name="100mVs_sample_Ar_MeCN_10mM_Fc_run03"),
    ]
    options = ecat_module.MultiplotOptions.from_options(
        {
            "print": False,
            "title": False,
            "legend": True,
            "deduplicate labels": "not a stat",
        }
    ).to_legacy_dict()

    style = ecat_module._prepare_multiplot_style(objects, options)
    captured = capsys.readouterr()

    assert style["display labels"] == ["CO2", "CO2", "Ar"]
    assert "Invalid deduplicate label key" in captured.out
    assert "Valid deduplicate label options are:" in captured.out
    assert "scan window" in captured.out
    assert "segments" in captured.out


def test_multiplot_min_gradient_entries_controls_auto_colorbar_threshold(
    ecat_module,
    cv_factory,
):
    objects = [
        cv_factory(name="50mVs_sample_CO2_MeCN_10mM_Fc_run01"),
        cv_factory(name="100mVs_sample_CO2_MeCN_10mM_Fc_run02"),
        cv_factory(name="200mVs_sample_CO2_MeCN_10mM_Fc_run03"),
    ]

    high_options = ecat_module.MultiplotOptions.from_options(
        {
            "print": False,
            "title": False,
            "legend": True,
            "legend mode": "auto",
            "min gradient entries": 4,
        },
    ).to_legacy_dict()
    high_threshold = ecat_module._prepare_multiplot_style(
        objects,
        high_options,
    )["color spec"]

    assert high_threshold["gradient groups"] == []
    assert high_threshold["discrete indices"] == [0, 1, 2]
    plt.close("all")

    met_options = ecat_module.MultiplotOptions.from_options(
        {
            "print": False,
            "title": False,
            "legend": True,
            "legend mode": "auto",
            "min gradient entries": 3,
        },
    ).to_legacy_dict()
    threshold_met = ecat_module._prepare_multiplot_style(
        objects,
        met_options,
    )["color spec"]

    assert len(threshold_met["gradient groups"]) == 1
    assert threshold_met["discrete indices"] == []
    plt.close("all")


def test_multiplot_min_gradient_entries_applies_per_detected_colorbar_group(
    ecat_module,
    cv_factory,
):
    small_group = [
        cv_factory(name="50mVs_small_CO2_MeCN_10mM_Fc_run01"),
        cv_factory(name="100mVs_small_CO2_MeCN_10mM_Fc_run02"),
    ]
    large_group = [
        cv_factory(name="50mVs_large_N2_MeCN_10mM_Fc_run01"),
        cv_factory(name="100mVs_large_N2_MeCN_10mM_Fc_run02"),
        cv_factory(name="200mVs_large_N2_MeCN_10mM_Fc_run03"),
    ]
    objects = small_group + large_group
    options = ecat_module.MultiplotOptions.from_options(
        {
            "print": False,
            "title": False,
            "legend": True,
            "legend mode": "colorbar",
            "min gradient entries": 3,
        },
    ).to_legacy_dict()

    color_spec = ecat_module._prepare_multiplot_style(objects, options)["color spec"]

    assert [group["indices"] for group in color_spec["gradient groups"]] == [[2, 3, 4]]
    assert color_spec["discrete indices"] == [0, 1]


def test_multiplot_scan_rate_gradient_does_not_merge_different_segment_counts(
    ecat_module,
    cv_factory,
):
    objects = [
        cv_factory(name="50mVs_sample_Ar_MeCN_1mMZn(cyclen)(OTf)2_run01"),
        cv_factory(name="100mVs_sample_Ar_MeCN_1mMZn(cyclen)(OTf)2_run02"),
        cv_factory(name="200mVs_sample_Ar_MeCN_1mMZn(cyclen)(OTf)2_run03"),
    ]
    objects[0].segments = 3
    objects[1].segments = 3
    objects[2].segments = 9
    options = ecat_module.MultiplotOptions.from_options(
        {
            "print": False,
            "title": False,
            "legend": True,
            "legend mode": "colorbar",
            "min gradient entries": 3,
        },
    ).to_legacy_dict()

    color_spec = ecat_module._prepare_multiplot_style(objects, options)["color spec"]

    assert color_spec["gradient groups"] == []
    assert color_spec["discrete indices"] == [0, 1, 2]


def test_multiplot_min_gradient_entries_applies_per_concentration_colorbar_group(
    ecat_module,
    cv_factory,
):
    small_group = [
        cv_factory(name="100mVs_small_CO2_MeCN_5mM_H2O_10mM_Fc_run01"),
        cv_factory(name="100mVs_small_CO2_MeCN_10mM_H2O_10mM_Fc_run02"),
    ]
    large_group = [
        cv_factory(name="100mVs_large_N2_MeCN_5mM_H2O_10mM_Fc_run01"),
        cv_factory(name="100mVs_large_N2_MeCN_10mM_H2O_10mM_Fc_run02"),
        cv_factory(name="100mVs_large_N2_MeCN_20mM_H2O_10mM_Fc_run03"),
    ]
    objects = small_group + large_group
    options = ecat_module.MultiplotOptions.from_options(
        {
            "print": False,
            "title": False,
            "legend": True,
            "gradient by": "concentration",
            "legend mode": "colorbar",
            "min gradient entries": 3,
        },
    ).to_legacy_dict()

    color_spec = ecat_module._prepare_multiplot_style(objects, options)["color spec"]

    assert [group["indices"] for group in color_spec["gradient groups"]] == [[2, 3, 4]]
    assert color_spec["discrete indices"] == [0, 1]


def test_multiplot_auto_gradient_keeps_scan_rate_and_concentration_groups(
    ecat_module,
    cv_factory,
):
    scan_rate_group = [
        cv_factory(name="25mVs_scan_Ar_MeCN_1mMFe_3mMFc_run01"),
        cv_factory(name="50mVs_scan_Ar_MeCN_1mMFe_3mMFc_run02"),
        cv_factory(name="100mVs_scan_Ar_MeCN_1mMFe_3mMFc_run03"),
        cv_factory(name="500mVs_scan_Ar_MeCN_1mMFe_3mMFc_run04"),
    ]
    concentration_group = [
        cv_factory(name="100mVs_phoh_CO2_MeCN_1mMFe_3mMFc_100mMPhOH_run01"),
        cv_factory(name="100mVs_phoh_CO2_MeCN_1mMFe_3mMFc_560mMPhOH_run02"),
        cv_factory(name="100mVs_phoh_CO2_MeCN_1mMFe_3mMFc_1MPhOH_run03"),
        cv_factory(name="100mVs_phoh_CO2_MeCN_1mMFe_3mMFc_2.8MPhOH_run04"),
    ]
    objects = scan_rate_group + concentration_group
    options = ecat_module.MultiplotOptions.from_options(
        {
            "print": False,
            "title": False,
            "legend": True,
            "legend mode": "colorbar",
            "color mode": "auto",
            "min gradient entries": 3,
        },
    ).to_legacy_dict()

    color_spec = ecat_module._prepare_multiplot_style(objects, options)["color spec"]

    assert [
        (group["gradient by"], group.get("gradient species"), group["indices"])
        for group in color_spec["gradient groups"]
    ] == [
        ("scan rate", None, [0, 1, 2, 3]),
        ("concentration", "PhOH", [4, 5, 6, 7]),
    ]
    assert color_spec["discrete indices"] == []


def test_multiplot_min_gradient_entries_counts_distinct_gradient_values(
    ecat_module,
    cv_factory,
):
    objects = [
        cv_factory(name="100mVs_a_Ar_MeCN_100mM_H2O_1mMZn(cyclen)(OTf)2_run01"),
        cv_factory(name="100mVs_b_Ar_MeCN_100mM_H2O_1mMZn(cyclen)(OTf)2_run02"),
        cv_factory(name="100mVs_c_Ar_MeCN_2.8M_H2O_1mMZn(cyclen)(OTf)2_run03"),
        cv_factory(name="100mVs_d_Ar_MeCN_2.8M_H2O_1mMZn(cyclen)(OTf)2_run04"),
    ]
    options = ecat_module.MultiplotOptions.from_options(
        {
            "print": False,
            "title": False,
            "legend": True,
            "gradient by": "concentration",
            "legend mode": "colorbar",
            "min gradient entries": 3,
        },
    ).to_legacy_dict()

    color_spec = ecat_module._prepare_multiplot_style(objects, options)["color spec"]

    assert color_spec["gradient groups"] == []
    assert color_spec["discrete indices"] == [0, 1, 2, 3]


def test_multiplot_min_gradient_entries_applies_to_explicit_gradient_mode(
    ecat_module,
    cv_factory,
):
    objects = _plot_cv_triplet(cv_factory)
    options = ecat_module.MultiplotOptions.from_options(
        {
            "print": False,
            "title": False,
            "legend": True,
            "color mode": "gradient",
            "legend mode": "colorbar",
            "min gradient entries": 4,
        },
    ).to_legacy_dict()

    color_spec = ecat_module._prepare_multiplot_style(objects, options)["color spec"]

    assert color_spec["gradient groups"] == []
    assert color_spec["discrete indices"] == [0, 1, 2]
    plt.close("all")


def test_multiplot_colorbar_context_has_clearance_and_compact_percent(ecat_module):
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1], label="Trace A")
    ax.plot([0, 1], [1, 0], label="Trace B")
    color_spec = {
        "gradient groups": [
            {
                "indices": [0, 1],
                "norm": mpl.colors.Normalize(vmin=0, vmax=10),
                "cmap": plt.get_cmap("viridis"),
                "ticks": [0, 10],
                "ticklabels": ["5 % CO2", "20 mM Mg"],
                "endpoint ticks": [0, 10],
                "endpoint ticklabels": ["+5 % CO2", "+20 mM Mg"],
                "legend context line": "20 mM Mg",
            }
        ]
    }
    options = {
        "legend mode": "colorbar",
        "legend loc": "upper left",
        "legend outside": False,
        "legend pad": 0.02,
        "colorbar height scale": 1.0,
        "colorbar reverse": True,
        "colorbar tick length": 5,
        "colorbar tick pad": 8,
        "colorbar trace ticks": True,
        "colorbar tick labels": "endpoints",
    }

    panel_ax = ecat_module._draw_multiplot_legend_and_colorbars(
        ax,
        color_spec,
        options,
        legend_fs=10,
    )
    text_by_plain = {
        ecat_module._plain_formula_text(text.get_text()): text
        for text in panel_ax.texts
    }

    assert "+5% CO2" in text_by_plain
    assert "20 mM Mg" in text_by_plain
    assert (
        text_by_plain["20 mM Mg"].get_position()[1]
        - text_by_plain["+5% CO2"].get_position()[1]
    ) >= 0.08

    plt.close(fig)


def test_multiplot_colorbar_endpoint_labels_do_not_subscript_signed_concentrations(ecat_module):
    assert ecat_module._format_already_or_chemical("+5 % CO2") == "+5% CO$_2$"


def test_chemical_formatter_does_not_subscript_segment_counts(ecat_module):
    assert ecat_module.format_chemical_formulas("Background (3 seg.)") == "Background (3 seg.)"
    assert ecat_module.format_chemical_formulas("CO2 (9 seg.)") == "CO$_2$ (9 seg.)"
    assert ecat_module.format_chemical_formulas("[Fe(CN)6] (3 seg.)") == "[Fe(CN)$_6$] (3 seg.)"
    assert ecat_module.format_chemical_formulas("Zn(cyclen) (3 seg.)") == "Zn(cyclen) (3 seg.)"


def test_multiplot_colorbar_mole_fraction_labels_do_not_use_plus_prefix(ecat_module):
    color_spec = {
        "gradient groups": [
            {
                "indices": [0, 1],
                "gradient by": "concentration",
                "gradient species": "D2O",
                "legend title": "D2O",
                "legend unit": "x",
                "ticklabels": ["0.2 x", "0.8 x"],
            }
        ]
    }
    labels = [
        "2.8 M D2O, 0.2 x D2O",
        "2.8 M D2O, 0.8 x D2O",
    ]

    ecat_module._attach_gradient_legend_text(color_spec, labels)

    assert color_spec["gradient groups"][0]["endpoint ticklabels"] == [
        "χ(D$_2$O) = 0.2",
        "χ(D$_2$O) = 0.8",
    ]


def test_multiplot_gradient_context_preserves_common_disambiguator_block(ecat_module):
    color_spec = {
        "gradient groups": [
            {
                "indices": [0, 1, 2],
                "gradient by": "concentration",
                "gradient species": "Zn(cyclen)",
                "legend title": "Zn(cyclen)",
                "legend unit": "M",
                "ticklabels": ["1 mM", "5 mM", "20 mM"],
            }
        ]
    }
    labels = [
        "CO2, 2.8 M H2O, 1 mM Zn(cyclen) ([-1.9, 1.4], 3 seg.)",
        "CO2, 2.8 M H2O, 5 mM Zn(cyclen) ([-1.9, 1.4], 3 seg.)",
        "CO2, 2.8 M H2O, 20 mM Zn(cyclen) ([-1.9, 1.4], 3 seg.)",
    ]

    ecat_module._attach_gradient_legend_text(color_spec, labels)

    assert (
        color_spec["gradient groups"][0]["legend context line"]
        == "CO2, 2.8 M H2O ([-1.9, 1.4], 3 seg.)"
    )


def test_multiplot_gradient_context_keeps_only_common_disambiguator_parts(ecat_module):
    color_spec = {
        "gradient groups": [
            {
                "indices": [0, 1],
                "gradient by": "concentration",
                "gradient species": "Zn(cyclen)",
                "legend title": "Zn(cyclen)",
                "legend unit": "M",
                "ticklabels": ["5 mM", "20 mM"],
            }
        ]
    }
    labels = [
        "CO2, 2.8 M H2O, 5 mM Zn(cyclen) ([-1.9, 1.4], 3 seg.)",
        "CO2, 2.8 M H2O, 20 mM Zn(cyclen) ([-1.9, 1.4], 9 seg.)",
    ]

    ecat_module._attach_gradient_legend_text(color_spec, labels)

    assert (
        color_spec["gradient groups"][0]["legend context line"]
        == "CO2, 2.8 M H2O ([-1.9, 1.4])"
    )


def test_multiplot_stacked_colorbars_keep_context_after_matching_discrete_line(ecat_module):
    fig, ax = plt.subplots()
    for idx, label in enumerate([
        "Ar",
        "CO2",
        "Ar, 2.8 M H2O",
        "CO2, 2.8 M H2O",
    ]):
        ax.plot([0, 1], [idx, idx + 0.2], label=label)

    color_spec = {
        "gradient groups": [
            {
                "indices": [4, 6, 8, 10, 12],
                "norm": mpl.colors.Normalize(vmin=0.1, vmax=2.0),
                "cmap": plt.get_cmap("viridis"),
                "ticks": [0.1, 0.25, 0.5, 1.0, 2.0],
                "ticklabels": ["0.1 mM", "0.25 mM", "0.5 mM", "1 mM", "2 mM"],
                "endpoint ticks": [0.1, 2.0],
                "endpoint ticklabels": ["0.1 mM", "2 mM"],
                "legend context line": "Ar, 2.8 M H2O",
            },
            {
                "indices": [5, 7, 9, 11, 13],
                "norm": mpl.colors.Normalize(vmin=0.1, vmax=2.0),
                "cmap": plt.get_cmap("plasma"),
                "ticks": [0.1, 0.25, 0.5, 1.0, 2.0],
                "ticklabels": ["0.1 mM", "0.25 mM", "0.5 mM", "1 mM", "2 mM"],
                "endpoint ticks": [0.1, 2.0],
                "endpoint ticklabels": ["0.1 mM", "2 mM"],
                "legend context line": "CO2, 2.8 M H2O",
            },
        ]
    }
    options = {
        "legend mode": "colorbar",
        "legend loc": "upper right",
        "legend outside": False,
        "legend pad": 0.02,
        "colorbar height scale": 1.0,
        "colorbar reverse": True,
        "colorbar tick length": 5,
        "colorbar tick pad": 8,
        "colorbar trace ticks": True,
        "colorbar tick labels": "endpoints",
    }

    panel_ax = ecat_module._draw_multiplot_legend_and_colorbars(
        ax,
        color_spec,
        options,
        legend_fs=10,
    )
    plain_texts = [
        ecat_module._plain_formula_text(text.get_text())
        for text in panel_ax.texts
    ]

    assert plain_texts.count("CO2, 2.8 M H2O") == 2
    assert "Ar, 2.8 M H2O" in plain_texts
    plt.close(fig)


def test_multiplot_discrete_legend_can_be_placed_outside(ecat_module, cv_factory):
    objects = _plot_cv_triplet(cv_factory)[:2]

    ecat_module.multiplot(
        objects,
        {
            "print": False,
            "title": False,
            "legend": True,
            "legend mode": "discrete",
            "legend outside": True,
            "legend loc": "upper right",
        },
    )

    fig = plt.gcf()
    ax = plt.gca()
    legend = ax.get_legend()
    assert legend is not None

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    legend_bbox = legend.get_window_extent(renderer=renderer)
    axes_bbox = ax.get_window_extent(renderer=renderer)

    assert legend_bbox.x0 >= axes_bbox.x1
    plt.close(fig)


def test_multiplot_auto_legend_sample_length_scales_with_font_and_tick(ecat_module):
    fig, ax = plt.subplots()
    defaults = ecat_module.MultiplotOptions.from_options({}).to_legacy_dict()
    options = {
        "legend sample length": defaults["legend sample length"],
        "colorbar tick length": 5,
    }
    layout_cache = ecat_module._build_layout_cache(ax)

    small = ecat_module._legend_sample_length_axes(
        ax,
        options,
        legend_fs=8,
        layout_cache=layout_cache,
    )
    large = ecat_module._legend_sample_length_axes(
        ax,
        options,
        legend_fs=14,
        layout_cache=layout_cache,
    )
    expected_small = ecat_module._points_to_axes_x(
        ax,
        8 * mpl.rcParams["legend.handlelength"],
        layout_cache=layout_cache,
    )
    fixed = ecat_module._legend_sample_length_axes(
        ax,
        {"legend sample length": 0.05, "colorbar tick length": 5},
        legend_fs=8,
        layout_cache=layout_cache,
    )
    layout = ecat_module._custom_legend_layout(
        ax,
        panel_width=0.3,
        has_text=True,
        options=options,
        legend_fs=8,
        layout_cache=layout_cache,
    )

    line_len = (layout["line_x1"] - layout["line_x0"]) * 0.3
    bar_plus_tick = (
        layout["bar_w"] * 0.3
        + layout["endpoint_tick_len"]
    )

    assert defaults["legend sample length"] == "auto"
    assert small == pytest.approx(expected_small)
    assert large > small
    assert fixed == pytest.approx(0.05)
    assert line_len == pytest.approx(bar_plus_tick)
    assert layout["bar_w"] * 0.3 == pytest.approx(line_len * (2.0 / 3.0))
    assert layout["endpoint_tick_len"] == pytest.approx(line_len * (1.0 / 3.0))
    assert layout["bar_w"] * 0.3 < line_len
    plt.close(fig)


def test_multiplot_custom_legend_spacing_uses_matplotlib_labelspacing(ecat_module):
    fig, ax = plt.subplots()
    layout_cache = ecat_module._build_layout_cache(ax)

    with mpl.rc_context({"legend.labelspacing": 0.2}):
        row_h, tight_gap = ecat_module._custom_legend_row_metrics(
            ax,
            legend_fs=10,
            layout_cache=layout_cache,
        )

    with mpl.rc_context({"legend.labelspacing": 0.8}):
        _row_h, loose_gap = ecat_module._custom_legend_row_metrics(
            ax,
            legend_fs=10,
            layout_cache=layout_cache,
        )

    expected_row_h = ecat_module._points_to_axes_y(
        ax,
        10 * max(1.0, mpl.rcParams["legend.handleheight"]),
        layout_cache=layout_cache,
    )

    assert row_h == pytest.approx(expected_row_h)
    assert loose_gap > tight_gap
    plt.close(fig)


def test_multiplot_custom_legend_auto_font_uses_uncapped_panel_height(ecat_module):
    fig, ax = plt.subplots()
    for idx in range(24):
        ax.plot([0, 1], [idx, idx + 0.2], label=f"Trace {idx}")

    color_spec = {
        "gradient groups": [
            {
                "indices": [0, 1, 2],
                "cmap": mpl.colormaps["viridis"],
                "norm": mpl.colors.Normalize(vmin=0, vmax=1),
                "ticks": [0, 0.5, 1],
                "ticklabels": ["", "", ""],
                "endpoint ticks": [0, 1],
                "endpoint ticklabels": ["0", "1"],
                "legend context line": "Concentration",
            }
        ],
        "plot labels": [f"Trace {idx}" for idx in range(24)],
    }
    options = {
        "legend mode": "colorbar",
        "legend loc": "best",
        "legend fontsize": "auto",
        "legend sample length": "auto",
        "legend pad": 0.02,
        "colorbar height scale": 1.0,
    }
    layout_cache = ecat_module._build_layout_cache(ax)

    _width, capped_height = ecat_module._estimate_custom_panel_size(
        ax,
        color_spec,
        options,
        legend_fs=14,
        layout_cache=layout_cache,
    )
    _width, required_height = ecat_module._estimate_custom_panel_size(
        ax,
        color_spec,
        options,
        legend_fs=14,
        layout_cache=layout_cache,
        cap_height=False,
    )
    legend_fs, _legend_loc, legend_outside = ecat_module._resolve_adaptive_legend_layout(
        ax,
        color_spec,
        options,
    )

    assert capped_height == pytest.approx(0.82)
    assert required_height > capped_height
    assert legend_fs < 14
    assert legend_outside is True
    plt.close(fig)


def test_multiplot_colorbar_endpoint_ticks_are_longer_than_intermediate_ticks(ecat_module):
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1], label="Trace A")
    ax.plot([0, 1], [1, 0], label="Trace B")
    color_spec = {
        "gradient groups": [
            {
                "indices": [0, 1],
                "norm": mpl.colors.Normalize(vmin=0, vmax=10),
                "cmap": plt.get_cmap("viridis"),
                "ticks": [0, 5, 10],
                "ticklabels": ["5 % CO2", "10 % CO2", "20 % CO2"],
                "endpoint ticks": [0, 10],
                "endpoint ticklabels": ["+5 % CO2", "+20 % CO2"],
                "legend context line": "CO2",
            }
        ]
    }
    options = {
        "legend mode": "colorbar",
        "legend loc": "upper left",
        "legend outside": False,
        "legend pad": 0.02,
        "colorbar height scale": 1.0,
        "colorbar reverse": True,
        "colorbar tick length": 5,
        "colorbar tick pad": 8,
        "colorbar trace ticks": True,
        "colorbar tick labels": "endpoints",
    }

    panel_ax = ecat_module._draw_multiplot_legend_and_colorbars(
        ax,
        color_spec,
        options,
        legend_fs=10,
    )
    cax = panel_ax.child_axes[0]
    tick_lengths = {
        tick.get_loc(): tick.tick1line.get_markersize()
        for tick in cax.yaxis.get_major_ticks()
    }

    assert tick_lengths[0] == pytest.approx(tick_lengths[10])
    assert tick_lengths[0] > tick_lengths[5]
    plt.close(fig)


def test_multiplot_colorbar_all_tick_labels_does_not_duplicate_endpoints(ecat_module):
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1], label="Trace A")
    ax.plot([0, 1], [1, 0], label="Trace B")
    color_spec = {
        "gradient groups": [
            {
                "indices": [0, 1],
                "norm": mpl.colors.Normalize(vmin=0, vmax=10),
                "cmap": plt.get_cmap("viridis"),
                "ticks": [0, 5, 10],
                "ticklabels": ["0 mV/s", "5 mV/s", "10 mV/s"],
                "endpoint ticks": [0, 10],
                "endpoint ticklabels": ["0 mV/s", "10 mV/s"],
                "legend context line": "Scan Rate",
            }
        ]
    }
    options = {
        "legend mode": "colorbar",
        "legend loc": "upper left",
        "legend outside": False,
        "legend pad": 0.02,
        "colorbar height scale": 1.0,
        "colorbar trace ticks": True,
        "colorbar tick labels": "all",
    }

    panel_ax = ecat_module._draw_multiplot_legend_and_colorbars(
        ax,
        color_spec,
        options,
        legend_fs=10,
    )

    axis_tick_labels = [
        text.get_text()
        for child_ax in panel_ax.child_axes
        for text in child_ax.get_yticklabels()
    ]
    manual_labels = [text.get_text() for text in panel_ax.texts]

    assert axis_tick_labels.count("0 mV/s") == 1
    assert axis_tick_labels.count("10 mV/s") == 1
    assert manual_labels.count("0 mV/s") == 0
    assert manual_labels.count("10 mV/s") == 0
    plt.close(fig)


def test_multiplot_plot_segment_restricts_each_trace(ecat_module, cv_factory):
    objects = _plot_cv_triplet(cv_factory)[:2]

    ecat_module.multiplot(
        objects,
        {"print": False, "title": False, "legend": False, "plot segment": 1},
    )

    fig = plt.gcf()
    ax = plt.gca()

    assert len(ax.lines) == 2
    assert [len(line.get_xdata()) for line in ax.lines] == [10, 10]
    assert ax.get_legend() is None
    plt.close(fig)


def test_multimultiplot_creates_one_figure_per_group_with_expected_line_counts(
    ecat_module,
    cv_factory,
):
    objects = _plot_cv_triplet(cv_factory)
    grouped = [[objects[0], objects[1]], [objects[2]]]

    result = ecat_module.multimultiplot(
        grouped,
        {"print": False, "legend": True},
    )

    fig_numbers = plt.get_fignums()
    figures = [plt.figure(number) for number in fig_numbers]

    assert result is None
    assert len(figures) == 2
    assert [len(fig.axes[0].lines) for fig in figures] == [2, 1]
    assert figures[0].axes[0].get_legend() is not None
    assert figures[1].axes[0].get_legend() is None
    assert figures[0]._suptitle is not None
    assert figures[1]._suptitle is not None

    for fig in figures:
        plt.close(fig)


def test_multimultiplot_accepts_multimultiplot_options_dataclass(ecat_module, cv_factory):
    objects = _plot_cv_triplet(cv_factory)
    grouped = [[objects[0]], [objects[1], objects[2]]]
    options = ecat_module.MultiMultiplotOptions.from_options(
        {"print": False, "legend": True, "titles": ["Forward Set", "Mixed Set"]}
    )

    result = ecat_module.multimultiplot(grouped, options)

    figures = [plt.figure(number) for number in plt.get_fignums()]

    assert result is None
    assert [fig.axes[0].get_title() for fig in figures] == ["Forward Set", "Mixed Set"]

    for fig in figures:
        plt.close(fig)


def test_multimultiplot_rejects_unknown_option_with_suggestion(ecat_module, cv_factory):
    objects = _plot_cv_triplet(cv_factory)

    with pytest.raises(ecat_module.OptionError, match="titles"):
        ecat_module.multimultiplot([[objects[0]], [objects[1]]], {"titelz": ["A", "B"]})


def test_multimultiplot_handles_mixed_named_groups_without_crashing(
    ecat_module,
    cv_factory,
):
    objects = _plot_cv_triplet(cv_factory)
    grouped = [[objects[0]], [objects[1], objects[2]]]

    result = ecat_module.multimultiplot(
        grouped,
        {"print": False, "legend": True, "titles": ["Forward Set", "Mixed Set"]},
    )

    fig_numbers = plt.get_fignums()
    figures = [plt.figure(number) for number in fig_numbers]

    assert result is None
    assert len(figures) == 2
    assert [fig.axes[0].get_title() for fig in figures] == ["Forward Set", "Mixed Set"]
    assert [len(fig.axes[0].lines) for fig in figures] == [1, 2]

    for fig in figures:
        plt.close(fig)
