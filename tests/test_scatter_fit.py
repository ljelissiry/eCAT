import numpy as np
import pandas as pd
import pytest
import matplotlib.pyplot as plt
import warnings
from matplotlib.collections import PolyCollection


def _fit_band_collections(ax):
    return [collection for collection in ax.collections if isinstance(collection, PolyCollection)]


def _band_width_near_x(collection, x_value):
    vertices = collection.get_paths()[0].vertices
    x = vertices[:, 0]
    y = vertices[:, 1]
    x_unique = np.unique(x[np.isfinite(x)])
    nearest = x_unique[np.argmin(np.abs(x_unique - float(x_value)))]
    y_at_x = y[np.isclose(x, nearest)]
    return float(np.nanmax(y_at_x) - np.nanmin(y_at_x))


def test_complex_potential_series_helper_normalizes_scalar_and_per_cv_lists(ecat_module):
    assert ecat_module._resolve_complex_potential_series(
        {"guess potential": -0.1},
        {"guess potential": -0.1},
        n_cvs=3,
        option_name="guess potential",
        analysis_name="fit_peak_current",
    ) == [-0.1, -0.1, -0.1]

    assert ecat_module._resolve_complex_potential_series(
        {"guess potentials": [-0.1, -0.2, -0.3]},
        {"guess potential": [-0.1, -0.2, -0.3]},
        n_cvs=3,
        option_name="guess potential",
        analysis_name="fit_peak_current",
    ) == [-0.1, -0.2, -0.3]


@pytest.mark.parametrize(
    ("option_name", "plural_name"),
    [
        ("guess potential", "guess potentials"),
        ("exact potential", "exact potentials"),
        ("tangent potential", "tangent potentials"),
        ("peak potential", "peak potentials"),
        ("non-catalytic guess potential", "non-catalytic guess potentials"),
        ("redox potential", "redox potentials"),
    ],
)
def test_complex_potential_series_helper_resolves_every_plural_alias(
    ecat_module,
    option_name,
    plural_name,
):
    values = [-0.1, -0.2, -0.3]

    assert ecat_module._resolve_complex_potential_series(
        {plural_name: values},
        {option_name: values},
        n_cvs=3,
        option_name=option_name,
        analysis_name="alias contract",
    ) == values


def test_complex_potential_series_helper_handles_paired_guess_shapes(ecat_module):
    assert ecat_module._resolve_complex_potential_series(
        {"guess potential": [-0.1, 0.1]},
        {"guess potential": [-0.1, 0.1]},
        n_cvs=3,
        option_name="guess potential",
        analysis_name="trumpet_analysis",
        paired=True,
    ) == [[-0.1, 0.1], [-0.1, 0.1], [-0.1, 0.1]]

    assert ecat_module._resolve_complex_potential_series(
        {"guess potential": [-0.1, -0.2]},
        {"guess potential": [-0.1, -0.2]},
        n_cvs=2,
        option_name="guess potential",
        analysis_name="trumpet_analysis",
        paired=True,
    ) == [-0.1, -0.2]

    assert ecat_module._resolve_complex_potential_series(
        {"guess potential": [[-0.1, 0.1], [-0.2, 0.2]]},
        {"guess potential": [[-0.1, 0.1], [-0.2, 0.2]]},
        n_cvs=2,
        option_name="guess potential",
        analysis_name="trumpet_analysis",
        paired=True,
    ) == [[-0.1, 0.1], [-0.2, 0.2]]

    assert ecat_module._resolve_complex_potential_series(
        {"guess potential": [[-0.1, 0.1]]},
        {"guess potential": [[-0.1, 0.1]]},
        n_cvs=3,
        option_name="guess potential",
        analysis_name="trumpet_analysis",
        paired=True,
    ) == [[-0.1, 0.1], [-0.1, 0.1], [-0.1, 0.1]]


@pytest.mark.parametrize(
    ("option_name", "plural_name"),
    [
        ("guess potential", "guess potentials"),
        ("exact potential", "exact potentials"),
        ("tangent potential", "tangent potentials"),
        ("peak potential", "peak potentials"),
        ("non-catalytic guess potential", "non-catalytic guess potentials"),
        ("redox potential", "redox potentials"),
    ],
)
def test_complex_potential_series_helper_reports_mismatched_lengths(
    ecat_module,
    option_name,
    plural_name,
):
    with pytest.raises(
        ValueError,
        match=f"{option_name}.*fit_peak_current.*1 scalar value or 3 scalar values.*2 entries",
    ):
        ecat_module._resolve_complex_potential_series(
            {plural_name: [-0.1, -0.2]},
            {option_name: [-0.1, -0.2]},
            n_cvs=3,
            option_name=option_name,
            analysis_name="fit_peak_current",
        )


def test_other_plural_length_errors_describe_scalar_values_not_cv_objects(ecat_module):
    with pytest.raises(
        ValueError,
        match="scan rate.*1 scalar value or 3 scalar values.*2 entries",
    ):
        ecat_module._resolve_nicholson_scan_rates(
            [object(), object(), object()],
            {"scan rate": [0.1, 0.2]},
        )

    with pytest.raises(
        ValueError,
        match="redox potentials.*1 scalar value or 3 scalar values.*2 entries",
    ):
        ecat_module._resolve_fowa_scalar_or_sequence(
            [0.1, 0.2],
            3,
            "redox potentials",
        )

    with pytest.raises(
        ValueError,
        match="fit range.*1 range or 3 ranges.*2 entries",
    ):
        ecat_module._resolve_fowa_range_or_sequence(
            [[0.1, 0.2], [0.2, 0.3]],
            3,
            option_name="fit range",
        )


def test_fit_default_return_shape_and_unlabeled_plot(ecat_module):
    coeffs, r2 = ecat_module.fit([1, 2, 3], [2, 4, 6], label="Calibration")

    assert coeffs.tolist() == pytest.approx([2.0, 0.0])
    assert r2 == pytest.approx(1.0)

    ax = plt.gca()
    assert len(ax.lines) == 1
    assert ax.lines[0].get_label().startswith("_")
    assert ax.get_legend() is None


def test_fit_label_and_style_options(ecat_module):
    ecat_module.fit(
        [1, 2, 3],
        [2, 4, 6],
        label="Calibration",
        options={
            "fit label": True,
            "fit color": "black",
            "fit linestyle": ":",
            "fit linewidth": 2,
            "fit alpha": 0.5,
        },
    )

    line = plt.gca().lines[0]
    assert line.get_label().startswith("Calibration")
    assert line.get_color() == "black"
    assert line.get_linestyle() == ":"
    assert line.get_linewidth() == pytest.approx(2)
    assert line.get_alpha() == pytest.approx(0.5)


def test_fit_accepts_fit_color_list_and_uses_first_color(ecat_module):
    ecat_module.fit(
        [1, 2, 3],
        [2, 4, 6],
        options={"fit color": ["tab:green", "tab:orange"]},
    )

    line = plt.gca().lines[0]
    assert ecat_module.mpl.colors.to_hex(line.get_color()) == ecat_module.mpl.colors.to_hex("tab:green")
    plt.close("all")


def test_fit_prints_statistics_and_can_return_stats(ecat_module, capsys):
    coeffs, stats = ecat_module.fit(
        [1, 2, 3],
        [3, 5, 7],
        plot_fit=False,
        options={"print": True, "return stats": True},
    )

    printed = capsys.readouterr().out

    assert coeffs.tolist() == pytest.approx([2.0, 1.0])
    assert stats["slope"] == pytest.approx(2.0)
    assert stats["intercept"] == pytest.approx(1.0)
    assert stats["r2"] == pytest.approx(1.0)
    assert stats["rmse"] == pytest.approx(0.0)
    assert stats["n"] == 3
    assert stats["degree"] == 1
    assert stats["y fit"].tolist() == pytest.approx([3.0, 5.0, 7.0])
    assert "R2:" in printed
    assert "RMSE:" in printed


def test_fit_can_use_multiple_fit_index_windows(ecat_module):
    x = np.arange(8, dtype=float)
    y = 2 * x + 1
    y[3:5] = [100, -100]

    coeffs, stats = ecat_module.fit(
        x,
        y,
        plot_fit=False,
        options={"fit indices": [[0, 3], [5, 8]], "return stats": True},
    )

    assert coeffs.tolist() == pytest.approx([2.0, 1.0])
    assert stats["r2"] == pytest.approx(1.0)
    assert stats["n"] == 6
    assert stats["x fit"].tolist() == pytest.approx([0, 1, 2, 5, 6, 7])
    assert stats["y fit"].tolist() == pytest.approx([1, 3, 5, 11, 13, 15])


def test_fit_plot_uses_selected_indices(ecat_module):
    x = np.arange(6, dtype=float)
    y = 3 * x - 2
    y[2:4] = [50, -50]

    ecat_module.fit(
        x,
        y,
        options={"fit indices": [[0, 2], [4, 6]]},
    )

    line = plt.gca().lines[0]
    np.testing.assert_allclose(line.get_xdata(), [0, 1, 4, 5])
    np.testing.assert_allclose(line.get_ydata(), [-2, 1, 10, 13])


def test_fit_line_range_extends_low_level_fit_plot_without_changing_stats(ecat_module):
    x = np.arange(6, dtype=float)
    y = 2 * x + 1
    y[2:4] = [50, -50]

    _coeffs, stats = ecat_module.fit(
        x,
        y,
        options={
            "fit indices": [[0, 2], [4, 6]],
            "fit line range": [0, 10],
            "return stats": True,
        },
    )

    line = plt.gca().lines[0]
    assert stats["x fit"].tolist() == pytest.approx([0, 1, 4, 5])
    np.testing.assert_allclose([line.get_xdata()[0], line.get_xdata()[-1]], [0, 10])
    np.testing.assert_allclose([line.get_ydata()[0], line.get_ydata()[-1]], [1, 21])


def test_transform_values_accepts_numeric_and_text_powers(ecat_module):
    values = np.asarray([2.0, 3.0, 4.0])

    squared, label, meta = ecat_module._transform_values(values, 2)
    assert squared.tolist() == pytest.approx([4.0, 9.0, 16.0])
    assert label == "^2"
    assert meta["power"] == 2.0

    cubed, label, meta = ecat_module._transform_values(values, "^3")
    assert cubed.tolist() == pytest.approx([8.0, 27.0, 64.0])
    assert label == "^3"
    assert meta["power"] == 3.0

    square_root, label, meta = ecat_module._transform_values(values, "square root")
    assert square_root.tolist() == pytest.approx(np.sqrt(values))
    assert label == "sqrt"
    assert meta["power"] == 0.5

    reciprocal, label, meta = ecat_module._transform_values(values, "reciprocal")
    assert reciprocal.tolist() == pytest.approx([0.5, 1 / 3, 0.25])
    assert label == "1/"


def test_fit_rate_lineweaver_burk_uses_independent_xy_transforms(ecat_module):
    df = pd.DataFrame(
        {
            "Substrate Concentration (M)": [1.0, 2.0, 4.0, 8.0],
            "kobs": [2.0, 4.0, 8.0, 16.0],
        }
    )

    result = ecat_module.fit_rate(
        df,
        {
            "plot": False,
            "print": False,
            "x column": "Substrate Concentration (M)",
            "metric": "kobs",
            "transform mode": "lineweaver-burk",
        },
    )
    data = result.table
    fitline = result.fits

    assert data["x transformed"].tolist() == pytest.approx([1.0, 0.5, 0.25, 0.125])
    assert data["y transformed"].tolist() == pytest.approx([0.5, 0.25, 0.125, 0.0625])
    assert fitline["parameters"]["m"] == pytest.approx(0.5)
    assert fitline["parameters"]["b"] == pytest.approx(0.0, abs=1e-12)


def test_fit_rate_lineweaver_burk_plot_formats_reciprocal_axis_labels(ecat_module):
    df = pd.DataFrame(
        {
            "CO2 (%)": [5.0, 10.0, 20.0, 40.0],
            "kobs": [1.0, 1.7, 2.6, 3.4],
        }
    )

    ecat_module.fit_rate(
        df,
        {
            "plot": True,
            "print": False,
            "x column": "CO2 (%)",
            "metric": "kobs",
            "transform mode": "lineweaver-burk",
        },
    )

    ax = plt.gca()
    assert "1/" in ax.get_xlabel()
    assert "CO" in ax.get_xlabel()
    assert "1/" in ax.get_ylabel()
    assert "k" in ax.get_ylabel()


def test_fit_rate_y_mode_enhancement_adjusts_before_fit(ecat_module):
    df = pd.DataFrame(
        {
            "Scan Rate": [1.0, 2.0, 3.0],
            "kobs": [2.0, 4.0, 8.0],
        }
    )

    result = ecat_module.fit_rate(
        df,
        {
            "plot": False,
            "print": False,
            "metric": "kobs",
            "y mode": "enhancement",
        },
    )
    data = result.table
    fitline = result.fits

    assert data["kobs"].tolist() == pytest.approx([2.0, 4.0, 8.0])
    assert data["y raw"].tolist() == pytest.approx([2.0, 4.0, 8.0])
    assert data["y adjusted"].tolist() == pytest.approx([0.0, 1.0, 3.0])
    assert data["y transformed"].tolist() == pytest.approx([0.0, 1.0, 3.0])
    assert data["y0"].unique().tolist() == pytest.approx([2.0])
    assert data["y mode"].unique().tolist() == ["enhancement"]
    assert fitline["parameters"]["m"] == pytest.approx(1.5)


@pytest.mark.parametrize("alias", ["fractional", "relative change", "fractional change"])
def test_fit_rate_y_mode_enhancement_aliases(ecat_module, alias):
    df = pd.DataFrame({"Scan Rate": [1.0, 2.0, 3.0], "kobs": [2.0, 4.0, 8.0]})

    result = ecat_module.fit_rate(
        df,
        {
            "plot": False,
            "print": False,
            "metric": "kobs",
            "y mode": alias,
        },
    )

    assert result.table["y adjusted"].tolist() == pytest.approx([0.0, 1.0, 3.0])
    assert result.table["y mode"].unique().tolist() == ["enhancement"]


def test_fit_rate_y_modes_and_y0_overrides(ecat_module):
    df = pd.DataFrame({"Scan Rate": [1.0, 2.0, 3.0], "kobs": [2.0, 4.0, 8.0]})

    delta = ecat_module.fit_rate(
        df,
        {"plot": False, "print": False, "y mode": "delta", "y0": 1.0},
    ).table
    negative = ecat_module.fit_rate(
        df,
        {"plot": False, "print": False, "y mode": "negative delta", "y0": {"kobs": 10.0}},
    ).table
    ratio = ecat_module.fit_rate(
        df,
        {"plot": False, "print": False, "y mode": "ratio", "y0": {"default": 2.0}},
    ).table

    assert delta["y adjusted"].tolist() == pytest.approx([1.0, 3.0, 7.0])
    assert negative["y adjusted"].tolist() == pytest.approx([8.0, 6.0, 2.0])
    assert ratio["y adjusted"].tolist() == pytest.approx([1.0, 2.0, 4.0])


def test_fit_rate_y_mode_adjustment_happens_before_transform_drop(ecat_module, capsys):
    df = pd.DataFrame({"Scan Rate": [1.0, 2.0, 3.0], "kobs": [2.0, 4.0, 8.0]})

    result = ecat_module.fit_rate(
        df,
        {
            "plot": False,
            "print": True,
            "metric": "kobs",
            "y mode": "enhancement",
            "transform mode": "log-log",
        },
    )
    data = result.table
    fitline = result.fits

    printed = capsys.readouterr().out

    assert "excluded 1" in printed.lower()
    assert data["y adjusted"].tolist() == pytest.approx([1.0, 3.0])
    assert data["y transformed"].tolist() == pytest.approx([0.0, np.log10(3.0)])
    assert set(fitline["parameters"]) == {"m", "b"}


def test_fit_rate_floor_true_defaults_to_relative_for_log_transform(ecat_module):
    df = pd.DataFrame({"Scan Rate": [0.0, 1.0, 10.0], "kobs": [0.0, 2.0, 20.0]})

    result = ecat_module.fit_rate(
        df,
        {
            "plot": False,
            "print": False,
            "metric": "kobs",
            "transform mode": "log-log",
            "floor": True,
        },
    )
    data = result.table
    fitline = result.fits

    assert data["x transform input"].tolist() == pytest.approx([0.1, 1.0, 10.0])
    assert data["y transform input"].tolist() == pytest.approx([0.2, 2.0, 20.0])
    assert data["x transformed"].tolist() == pytest.approx([-1.0, 0.0, 1.0])
    assert data["y transformed"].tolist() == pytest.approx([np.log10(0.2), np.log10(2.0), np.log10(20.0)])
    assert data.loc[0, "x transform note"] == "floor: 0.1"
    assert data.loc[0, "y transform note"] == "floor: 0.2"
    assert set(fitline["parameters"]) == {"m", "b"}


def test_fit_rate_floor_false_drops_nonpositive_values(ecat_module):
    df = pd.DataFrame({"Scan Rate": [0.0, 1.0, 10.0], "kobs": [0.0, 2.0, 20.0]})

    result = ecat_module.fit_rate(
        df,
        {
            "plot": False,
            "print": False,
            "metric": "kobs",
            "transform mode": "log-log",
            "floor": False,
        },
    )
    data = result.table
    fitline = result.fits

    assert data["x raw"].tolist() == pytest.approx([1.0, 10.0])
    assert "x transform input" not in data
    assert "x transform note" not in data
    assert data["x transformed"].tolist() == pytest.approx([0.0, 1.0])
    assert set(fitline["parameters"]) == {"m", "b"}


def test_fit_rate_floor_absolute_can_be_axis_specific(ecat_module):
    df = pd.DataFrame({"Scan Rate": [0.0, 1.0, 10.0], "kobs": [0.0, 2.0, 20.0]})

    result = ecat_module.fit_rate(
        df,
        {
            "plot": False,
            "print": False,
            "metric": "kobs",
            "transform mode": "log-log",
            "x floor": 1e-6,
            "y floor": 1e-9,
        },
    )
    data = result.table

    assert data["x transform input"].tolist() == pytest.approx([1e-6, 1.0, 10.0])
    assert data["y transform input"].tolist() == pytest.approx([1e-9, 2.0, 20.0])
    assert data.loc[0, "x transform note"] == "floor: 1e-06"
    assert data.loc[0, "y transform note"] == "floor: 1e-09"


def test_fit_rate_floor_rescues_values_below_threshold_not_only_zero(ecat_module):
    df = pd.DataFrame({"Scan Rate": [0.0, 0.5, 1.0, 10.0], "kobs": [0.0, 1.0, 2.0, 20.0]})

    result = ecat_module.fit_rate(
        df,
        {
            "plot": False,
            "print": False,
            "fit": False,
            "metric": "kobs",
            "transform mode": "log-log",
            "x floor": 1.0,
            "y floor": 2.0,
        },
    )
    data = result.table

    assert data["x transform input"].tolist() == pytest.approx([1.0, 1.0, 1.0, 10.0])
    assert data["y transform input"].tolist() == pytest.approx([2.0, 2.0, 2.0, 20.0])
    assert data["x transform note"].tolist() == ["floor: 1", "floor: 1", "", ""]
    assert data["y transform note"].tolist() == ["floor: 2", "floor: 2", "", ""]


def test_fit_rate_plot_scale_uses_axis_scale_without_transforming_values(ecat_module):
    df = pd.DataFrame({"Scan Rate": [1.0, 10.0, 100.0], "kobs": [2.0, 20.0, 200.0]})

    result = ecat_module.fit_rate(
        df,
        {
            "plot": True,
            "print": False,
            "metric": "kobs",
            "plot scale": "log-log",
            "legend": False,
        },
    )

    ax = plt.gca()
    assert ax.get_xscale() == "log"
    assert ax.get_yscale() == "log"
    assert result.table["x transform"].unique().tolist() == ["identity"]
    assert result.table["y transform"].unique().tolist() == ["identity"]
    assert result.table["x transformed"].tolist() == pytest.approx([1.0, 10.0, 100.0])
    assert result.table["y transformed"].tolist() == pytest.approx([2.0, 20.0, 200.0])
    assert "log" not in ax.get_xlabel().lower()
    assert "log" not in ax.get_ylabel().lower()
    plt.close(ax.figure)


def test_fit_rate_log_axis_default_fit_line_range_skips_zero_for_display(ecat_module):
    df = pd.DataFrame({"Scan Rate": [0.0, 0.1, 1.0, 10.0], "kobs": [9.5, 8.0, 4.5, 1.0]})

    result = ecat_module.fit_rate(
        df,
        {
            "plot": True,
            "print": False,
            "metric": "kobs",
            "fit indices": [0, None],
            "fit": True,
            "plot scale": "log-log",
            "transform mode": None,
            "legend": False,
        },
    )

    ax = plt.gca()
    line = ax.lines[0]
    assert ax.get_xscale() == "log"
    assert result.fit_table.loc[0, "fit x min"] == pytest.approx(0.0)
    assert line.get_xdata()[0] == pytest.approx(0.1)
    assert line.get_xdata()[-1] == pytest.approx(10.0)
    plt.close(ax.figure)


def test_fit_rate_log_axis_explicit_nonpositive_fit_line_range_errors(ecat_module):
    df = pd.DataFrame({"Scan Rate": [0.0, 0.1, 1.0, 10.0], "kobs": [9.5, 8.0, 4.5, 1.0]})

    with pytest.raises(ValueError, match="fit line range.*positive"):
        ecat_module.fit_rate(
            df,
            {
                "plot": True,
                "print": False,
                "metric": "kobs",
                "fit indices": [0, None],
                "plot scale": "log-log",
                "fit line range": [0, 10],
                "legend": False,
            },
        )
    plt.close("all")


def test_fit_rate_log_log_transform_does_not_apply_plot_scale_to_transformed_coordinates(ecat_module):
    df = pd.DataFrame({"Scan Rate": [0.01, 0.1, 1.0], "kobs": [0.02, 0.2, 2.0]})

    result = ecat_module.fit_rate(
        df,
        {
            "plot": True,
            "print": False,
            "metric": "kobs",
            "transform mode": "log-log",
            "plot scale": "log-log",
            "legend": False,
        },
    )

    ax = plt.gca()
    assert ax.get_xscale() == "linear"
    assert ax.get_yscale() == "linear"
    assert result.table["x transformed"].tolist() == pytest.approx([-2.0, -1.0, 0.0])
    assert result.table["y transformed"].tolist() == pytest.approx(
        [np.log10(0.02), np.log10(0.2), np.log10(2.0)]
    )
    np.testing.assert_allclose(
        [ax.lines[0].get_xdata()[0], ax.lines[0].get_xdata()[-1]],
        [-2.0, 0.0],
    )
    plt.close(ax.figure)


def test_fit_rate_y_mode_label_is_adjusted_before_transform_wrapping(ecat_module):
    df = pd.DataFrame({"Scan Rate": [1.0, 2.0, 3.0], "kobs": [2.0, 4.0, 8.0]})

    ecat_module.fit_rate(
        df,
        {
            "plot": True,
            "print": False,
            "metric": "kobs",
            "y mode": "enhancement",
            "y0": 1.0,
            "y transform": "log10",
        },
    )

    label = plt.gca().get_ylabel()
    assert "log" in label
    assert "/" in label
    assert "^{0}" in label
    assert "- 1" in label
    plt.close()


def test_y_mode_label_superscripts_mathtext_baseline_zero(ecat_module):
    label = ecat_module._format_y_mode_axis_label(
        ecat_module._format_fit_rate_metric_label("kobs"),
        "enhancement",
    )

    assert label == r"$k_{\mathrm{obs}}$/$k_{\mathrm{obs}}^{0}$ - 1"


def test_fit_rate_kobs_plot_uses_dataframe_unit_metadata(ecat_module):
    df = pd.DataFrame({"Scan Rate": [1.0, 2.0, 3.0], "kobs": [2.0, 4.0, 8.0]})
    df.attrs["units"] = {"kobs": "s^-1"}

    ecat_module.fit_rate(
        df,
        {"plot": True, "print": False, "fit": False, "metric": "kobs"},
    )

    assert plt.gca().get_ylabel() == r"$k_{\mathrm{obs}}$ ($\mathrm{s^{-1}}$)"
    plt.close()


def test_fit_rate_forwards_fit_label_and_style_options(ecat_module):
    df = pd.DataFrame(
        {
            "Scan Rate": [1.0, 2.0, 3.0],
            "kobs": [2.0, 4.0, 6.0],
        }
    )

    ecat_module.fit_rate(
        df,
        {
            "plot": True,
            "print": False,
            "metric": "kobs",
            "fit label": True,
            "fit color": "black",
            "fit linestyle": ":",
        },
    )

    line = plt.gca().lines[0]
    assert line.get_label().startswith("kobs Fit")
    assert line.get_color() == "black"
    assert line.get_linestyle() == ":"
    assert plt.gca().get_legend() is not None


def test_fit_rate_fit_color_matches_points_by_default(ecat_module):
    result = ecat_module.fit_rate(
        pd.DataFrame({"Scan Rate": [1.0, 2.0, 3.0], "kobs": [2.0, 4.0, 6.0]}),
        {"print": False},
    )

    ax = plt.gca()
    point_color = ecat_module.mpl.colors.to_hex(ax.collections[0].get_facecolors()[0])
    fit_color = ecat_module.mpl.colors.to_hex(ax.lines[0].get_color())

    assert fit_color == point_color
    plt.close(ax.figure)


def test_fit_rate_explicit_fit_color_still_overrides_point_color(ecat_module):
    result = ecat_module.fit_rate(
        pd.DataFrame({"Scan Rate": [1.0, 2.0, 3.0], "kobs": [2.0, 4.0, 6.0]}),
        {"print": False, "fit color": "black"},
    )

    ax = plt.gca()
    point_color = ecat_module.mpl.colors.to_hex(ax.collections[0].get_facecolors()[0])
    fit_color = ecat_module.mpl.colors.to_hex(ax.lines[0].get_color())

    assert fit_color == "#000000"
    assert fit_color != point_color
    plt.close(ax.figure)


def test_fit_rate_fit_label_auto_enables_style_sized_legend(ecat_module):
    df = pd.DataFrame(
        {
            "Scan Rate": [1.0, 2.0, 3.0],
            "kobs": [2.0, 4.0, 6.0],
        }
    )

    ecat_module.fit_rate(
        df,
        {
            "plot": True,
            "print": False,
            "metric": "kobs",
            "fit label": True,
        },
    )

    legend = plt.gca().get_legend()
    assert legend is not None
    assert legend.get_texts()[0].get_fontsize() == pytest.approx(
        ecat_module._default_legend_fontsize()
    )


def test_fit_rate_fit_label_respects_explicit_legend_false(ecat_module):
    df = pd.DataFrame(
        {
            "Scan Rate": [1.0, 2.0, 3.0],
            "kobs": [2.0, 4.0, 6.0],
        }
    )

    ecat_module.fit_rate(
        df,
        {
            "plot": True,
            "print": False,
            "metric": "kobs",
            "fit label": True,
            "legend": False,
        },
    )

    assert plt.gca().get_legend() is None


def test_fit_rate_fit_line_range_does_not_change_fit_selection_metadata(ecat_module):
    df = pd.DataFrame({"Scan Rate": [0.0, 1.0, 2.0, 3.0, 4.0], "kobs": [1.0, 3.0, 5.0, 7.0, 9.0]})

    result = ecat_module.fit_rate(
        df,
        {
            "plot": True,
            "print": False,
            "fit indices": [1, 3],
            "fit line range": [0.0, 5.0],
        },
    )

    ax = plt.gca()
    line = ax.lines[0]
    fit_table = result.fit_table

    assert fit_table["fit x min"].iloc[0] == pytest.approx(1.0)
    assert fit_table["fit x max"].iloc[0] == pytest.approx(2.0)
    np.testing.assert_allclose([line.get_xdata()[0], line.get_xdata()[-1]], [0.0, 5.0])
    np.testing.assert_allclose([line.get_ydata()[0], line.get_ydata()[-1]], [1.0, 11.0])


def test_fit_rate_fit_line_range_dict_maps_to_named_fit_indices(ecat_module):
    df = pd.DataFrame({"Scan Rate": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0], "kobs": [1.0, 3.0, 5.0, 7.0, 9.0, 11.0]})

    result = ecat_module.fit_rate(
        df,
        {
            "plot": True,
            "print": False,
            "fit indices": {
                "early": [0, 2],
                "late": [3, 5],
            },
            "fit line range": {
                "early": [0.0, 3.0],
                "late": [2.0, 6.0],
            },
        },
    )

    lines = plt.gca().lines
    fit_table = result.fit_table

    assert fit_table.loc[fit_table["series"] == "early", "fit x max"].iloc[0] == pytest.approx(1.0)
    assert fit_table.loc[fit_table["series"] == "late", "fit x min"].iloc[0] == pytest.approx(3.0)
    np.testing.assert_allclose([lines[0].get_xdata()[0], lines[0].get_xdata()[-1]], [0.0, 3.0])
    np.testing.assert_allclose([lines[1].get_xdata()[0], lines[1].get_xdata()[-1]], [2.0, 6.0])


def test_fit_model_fit_line_range_extends_line_but_not_fit_range_table(ecat_module):
    x = np.asarray([0.0, 1.0, 2.0, 3.0, 4.0])
    y = 2.0 * x + 1.0

    result = ecat_module.fit_model(
        x,
        y,
        options={
            "plot": True,
            "print": False,
            "fit indices": [1, 3],
            "fit line range": [0.0, 5.0],
        },
    )

    line = plt.gca().lines[0]
    assert result.fit_table["fit x min"].iloc[0] == pytest.approx(1.0)
    assert result.fit_table["fit x max"].iloc[0] == pytest.approx(2.0)
    np.testing.assert_allclose([line.get_xdata()[0], line.get_xdata()[-1]], [0.0, 5.0])
    np.testing.assert_allclose([line.get_ydata()[0], line.get_ydata()[-1]], [1.0, 11.0])


def test_fit_model_confidence_band_uses_fit_line_range(ecat_module):
    x = np.asarray([0.0, 1.0, 2.0, 3.0, 4.0])
    y = np.asarray([1.1, 2.8, 5.2, 7.1, 8.9])

    ecat_module.fit_model(
        x,
        y,
        options={
            "plot": True,
            "print": False,
            "fit band": "confidence",
            "fit line range": [0.0, 6.0],
        },
    )

    ax = plt.gca()
    bands = _fit_band_collections(ax)

    assert len(bands) == 1
    vertices = bands[0].get_paths()[0].vertices
    assert np.nanmin(vertices[:, 0]) == pytest.approx(0.0)
    assert np.nanmax(vertices[:, 0]) == pytest.approx(6.0)
    assert _band_width_near_x(bands[0], 3.0) > 0


def test_fit_model_prediction_band_is_wider_than_confidence_band(ecat_module):
    x = np.asarray([0.0, 1.0, 2.0, 3.0, 4.0])
    y = np.asarray([1.1, 2.8, 5.2, 7.1, 8.9])

    ecat_module.fit_model(
        x,
        y,
        options={"plot": True, "print": False, "fit band": "confidence"},
    )
    confidence_band = _fit_band_collections(plt.gca())[0]
    confidence_width = _band_width_near_x(confidence_band, 2.0)
    plt.close("all")

    ecat_module.fit_model(
        x,
        y,
        options={"plot": True, "print": False, "fit band": "prediction"},
    )
    prediction_band = _fit_band_collections(plt.gca())[0]
    prediction_width = _band_width_near_x(prediction_band, 2.0)

    assert prediction_width > confidence_width


def test_fit_model_power_recovers_parameters_and_can_plot_and_print(ecat_module, capsys):
    x = np.asarray([1.0, 2.0, 4.0, 8.0, 16.0])
    y = 3.0 * x ** 1.5

    result = ecat_module.fit_model(
        x,
        y,
        model="power",
        options={"plot": True, "print": True, "fit label": True},
    )

    printed = capsys.readouterr().out
    params = result.summary["parameters"]

    assert isinstance(result, ecat_module.ScatterFitResult)
    assert result.summary["model"] == "power"
    assert params["A"] == pytest.approx(3.0, rel=1e-5)
    assert params["n"] == pytest.approx(1.5, rel=1e-5)
    assert set(["Predicted", "Residual"]).issubset(result.table.columns)
    assert "Fit Model:" in printed
    assert "y = A x^n" in printed
    assert "y = 3x^1.5" not in printed
    assert any("y = 3.000x^{1.500}" in text.get_text() for text in plt.gca().get_legend().get_texts())
    assert plt.gca().lines[0].get_label().startswith("Power Fit")
    assert plt.gca().get_legend() is not None


def test_fit_model_supports_unified_fit_model_option(ecat_module):
    x = np.asarray([1.0, 2.0, 4.0, 8.0])
    y = 2.5 * x ** 0.75

    result = ecat_module.fit_model(
        x,
        y,
        options={"fit model": "power", "plot": False},
    )

    assert result.summary["model"] == "power"
    assert result.fits["parameters"]["A"] == pytest.approx(2.5, rel=1e-5)
    assert result.fits["parameters"]["n"] == pytest.approx(0.75, rel=1e-5)


def test_fit_model_overlay_fit_label_includes_equation_and_r2(ecat_module):
    df = pd.DataFrame(
        {
            "Substrate Concentration (M)": [0.1, 0.2, 0.4, 0.8, 1.6, 2.4],
            "kobs": [1.0, 1.2, 1.8, 3.8, 11.0, 26.0],
        }
    )
    base = ecat_module.fit_rate(
        df,
        {
            "plot": True,
            "print": False,
            "transform mode": None,
            "y mode": "raw",
            "fit": False,
        },
    )

    ecat_module.fit_model(
        base.table.iloc[2:],
        model="power offset",
        options={
            "plot": True,
            "new plot": False,
            "print": False,
            "fit label": True,
        },
    )

    legend = plt.gca().get_legend()
    assert legend is not None
    labels = [text.get_text() for text in legend.get_texts()]
    assert any("Power Offset Fit" in label for label in labels)
    assert any("y = b + A x^n" not in label and "y =" in label for label in labels)
    assert any("R^2" in label for label in labels)


def test_fit_model_plot_label_uses_sig_figs_and_braced_decimal_exponent(ecat_module):
    x = np.asarray([1.0, 2.0, 4.0, 8.0])
    y = 2.12345 * x ** 0.54321

    ecat_module.fit_model(
        x,
        y,
        model="power",
        options={"plot": True, "fit label": True, "sig figs": 3},
    )

    legend = plt.gca().get_legend()
    labels = [text.get_text() for text in legend.get_texts()]
    assert any("y = 2.12x^{0.543}" in label for label in labels)


def test_fit_model_plot_label_uses_default_sig_figs_when_option_omitted(ecat_module):
    x = np.asarray([1.0, 2.0, 4.0, 8.0])
    y = 2.12345 * x ** 0.54321

    try:
        ecat_module.set_defaults("fit_rate", {"sig figs": 3})
        ecat_module.fit_model(
            x,
            y,
            model="power",
            options={"plot": True, "fit label": True},
        )

        legend = plt.gca().get_legend()
        labels = [text.get_text() for text in legend.get_texts()]
        assert any("y = 2.12x^{0.543}" in label for label in labels)
    finally:
        ecat_module.reset_defaults()


def test_fit_model_multiline_fit_label_aligns_handle_to_top_line(ecat_module):
    x = np.asarray([1.0, 2.0, 4.0, 8.0])
    y = 2.0 * x ** 0.5

    ecat_module.fit_model(
        x,
        y,
        model="power",
        options={"plot": True, "fit label": True},
    )

    fig = plt.gcf()
    legend = plt.gca().get_legend()
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    fit_text = [text for text in legend.get_texts() if "\n" in text.get_text()][0]
    fit_handle = [handle for handle in legend.legend_handles if hasattr(handle, "get_ydata")][0]
    text_bbox = fit_text.get_window_extent(renderer)
    handle_bbox = fit_handle.get_window_extent(renderer)
    line_count = fit_text.get_text().count("\n") + 1
    first_line_center = text_bbox.y1 - text_bbox.height / line_count / 2
    block_center = (text_bbox.y0 + text_bbox.y1) / 2
    handle_center = (handle_bbox.y0 + handle_bbox.y1) / 2

    assert abs(handle_center - first_line_center) < abs(handle_center - block_center)


def test_fit_model_fit_label_respects_explicit_legend_false(ecat_module):
    x = np.asarray([1.0, 2.0, 4.0, 8.0])
    y = 2.0 * x ** 0.5

    ecat_module.fit_model(
        x,
        y,
        model="power",
        options={"plot": True, "fit label": True, "legend": False},
    )

    assert plt.gca().get_legend() is None


def test_fit_model_log_residual_reports_residual_space_and_raw_r2(ecat_module):
    x = np.asarray([1.0, 2.0, 3.0, 4.0])
    y = np.exp(0.5 + 0.4 * x)

    result = ecat_module.fit_model(
        x,
        y,
        model="exponential",
        options={"plot": False, "fit residual": "log"},
    )

    stats = result.fit_model_results["Model"]["stats"]
    assert stats["residual space"] == "log"
    assert "R2 raw" in stats
    assert stats["R2"] == pytest.approx(1.0)
    assert stats["R2 raw"] == pytest.approx(1.0)


def test_fit_model_log_residual_display_uses_ln_r2_label(ecat_module, monkeypatch):
    displayed = []

    def capture_display(table):
        displayed.append(table)

    monkeypatch.setattr(ecat_module, "display", capture_display)

    ecat_module.fit_model(
        [1.0, 2.0, 3.0, 4.0],
        np.exp(0.5 + 0.4 * np.asarray([1.0, 2.0, 3.0, 4.0])),
        model="exponential",
        options={"plot": False, "print": True, "fit residual": "log"},
    )

    summary_table = next(
        getattr(table, "data", table)
        for table in displayed
        if "Setting" in getattr(table, "data", table).columns
    )
    assert "ln R²" in summary_table["Setting"].tolist()
    assert "R²" not in summary_table["Setting"].tolist()


def test_fit_model_print_uses_pretty_details_and_parameter_tables(ecat_module, capsys):
    x = np.asarray([1.0, 2.0, 4.0, 8.0])
    y = 2.0 * x ** 0.5

    ecat_module.fit_model(
        x,
        y,
        model="power",
        options={"print": True},
    )

    printed = capsys.readouterr().out
    assert "Fit Model:" in printed
    assert "Fit Model Details:" not in printed
    assert "Fit Model Parameters:" not in printed
    assert "Field" in printed
    assert "Value" in printed
    assert "y = A x^n" in printed
    assert "Fit Parameters" in printed
    assert "A, n (2)" in printed
    assert "y = 2x^0.5" not in printed
    assert "±" in printed
    assert "A" in printed
    assert "n" in printed
    assert "Model Fit Statistics:" not in printed
    assert "fit x min" not in printed


def test_fit_model_pretty_print_false_prints_dictionary_summary(ecat_module, capsys):
    x = np.asarray([1.0, 2.0, 4.0, 8.0])
    y = 2.0 * x ** 0.5

    ecat_module.fit_model(
        x,
        y,
        model="power",
        options={"print": True, "pretty print": False},
    )

    printed = capsys.readouterr().out
    assert "Fit Model Summary:" in printed
    assert "'model': 'power'" in printed
    assert "'parameters':" in printed
    assert "Fit Model Details:" not in printed


def test_fit_model_details_print_for_bounded_power_offset(ecat_module, capsys):
    x = np.asarray([0.32, 0.6, 1.0, 1.6, 2.2, 2.8])
    y = 10.0 + 3.0 * x ** 2.0

    ecat_module.fit_model(
        x,
        y,
        model="power offset",
        options={
            "print": True,
            "fit init": {"b": 1.0, "A": 1.0, "n": 1.0},
            "fit bounds": {"b": [0, np.inf], "A": [0, np.inf], "n": [0, 8]},
        },
    )

    printed = capsys.readouterr().out
    assert "Fit Model Details:" in printed
    assert "Setting" in printed
    assert "Value" in printed
    assert "Equation" in printed
    assert "y = b + A x^n" in printed
    assert "Fit Parameters" in printed
    assert "b, A, n (3)" in printed
    assert "Fit Model Parameters:" in printed
    assert "Initial" in printed
    assert "Lower Bound" in printed
    assert "Upper Bound" in printed
    assert "Fit Value" in printed
    assert "Std. Error" in printed


def test_fit_model_curve_fit_method_and_passthrough_print_when_used(ecat_module, capsys):
    x = np.linspace(0.0, 5.0, 20)
    y = 1.0 + 2.0 * x

    result = ecat_module.fit_model(
        x,
        y,
        model="linear",
        options={
            "plot": False,
            "print": True,
            "print fit": "details",
            "fit method": "trf",
            "curve fit options": {"loss": "soft_l1", "x_scale": "jac"},
        },
    )

    printed = capsys.readouterr().out
    model_result = result.fit_model_results["Model"]
    assert model_result["fit method"] == "curve_fit / trf"
    assert model_result["curve fit options"]["loss"] == "soft_l1"
    assert model_result["curve fit options"]["x_scale"] == "jac"
    assert "Fit Method" in printed
    assert "curve_fit / trf" in printed


def test_fit_model_translates_scipy_covariance_warning(ecat_module, capsys):
    from scipy.optimize import OptimizeWarning

    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        result = ecat_module.fit_model(
            [1.0, 2.0, 3.0],
            [2.0, 4.0, 6.0],
            options={"plot": False, "print": True},
        )

    model_result = result.fit_model_results["Model"]
    assert not any(issubclass(item.category, OptimizeWarning) for item in captured)
    assert model_result["covariance warning"] is True
    assert "standard errors are unavailable" in model_result["covariance message"]
    assert "standard errors are unavailable" in capsys.readouterr().out


def test_fit_model_rejects_curve_fit_passthrough_owned_by_ecat(ecat_module):
    x = np.asarray([0.0, 1.0, 2.0])
    y = 1.0 + 2.0 * x

    with pytest.raises(ValueError, match="fit init.*fit bounds"):
        ecat_module.fit_model(
            x,
            y,
            model="linear",
            options={
                "plot": False,
                "print": False,
                "curve fit options": {"p0": [1.0, 0.0]},
            },
        )


def test_fit_model_warns_when_fit_points_equal_fit_parameters(ecat_module):
    x = np.asarray([0.0, 1.0])
    y = 1.0 + 2.0 * x

    with pytest.warns(UserWarning, match="2 points and 2 fitted parameters"):
        ecat_module.fit_model(
            x,
            y,
            model="linear",
            options={"plot": False, "print": False},
        )


def test_fit_model_strongly_warns_when_underdetermined(ecat_module):
    x = np.asarray([0.0, 1.0])
    y = np.asarray([1.0, 2.0])

    with pytest.warns(UserWarning, match="the model is underdetermined"):
        ecat_module.fit_model(
            x,
            y,
            model="k0 + k1*x + k2*x^2",
            options={"plot": False, "print": False, "fit method": "trf"},
        )


def test_fit_rate_accepts_curve_fit_method_options(ecat_module):
    x = np.linspace(0.0, 1.0, 6)
    df = pd.DataFrame({"H2O Concentration (M)": x, "kobs": 1.0 + 2.0 * x})

    result = ecat_module.fit_rate(
        df,
        {
            "plot": False,
            "print": False,
            "x column": "H2O Concentration (M)",
            "metric": "kobs",
            "fit model": "linear",
            "fit method": "trf",
            "curve fit options": {"loss": "soft_l1"},
        },
    )

    assert result.fit_model_results["kobs"]["fit method"] == "curve_fit / trf"
    assert result.fit_model_results["kobs"]["curve fit options"]["loss"] == "soft_l1"


def test_fit_model_print_fit_summary_forces_one_table(ecat_module, capsys):
    x = np.asarray([0.32, 0.6, 1.0, 1.6, 2.2, 2.8])
    y = 10.0 + 3.0 * x ** 2.0

    ecat_module.fit_model(
        x,
        y,
        model="power offset",
        options={"print": True, "print fit": "summary"},
    )

    printed = capsys.readouterr().out
    assert "Fit Model:" in printed
    assert "Fit Model Details:" not in printed
    assert "Fit Model Parameters:" not in printed


def test_show_scatter_fit_result_uses_pretty_fit_model_output(ecat_module, cv_factory, capsys):
    cvs = [
        cv_factory(name="50mVs_sample_CO2_MeCN_10mM_Fc_run01"),
        cv_factory(name="100mVs_sample_CO2_MeCN_10mM_Fc_run02"),
        cv_factory(name="200mVs_sample_CO2_MeCN_10mM_Fc_run03"),
    ]
    for obj, ip in zip(cvs, [2.0e-6, 4.0e-6, 8.0e-6]):
        obj.peak_current = lambda _options, value=ip: {"ip": value, "index": 0}

    result = ecat_module.fit_peak_current(
        cvs,
        {
            "plot": False,
            "print": False,
            "x transform": "identity",
            "fit model": "linear",
        },
    )

    ecat_module.show(result, {"pretty print": True})

    printed = capsys.readouterr().out
    assert "Fit Model:" in printed
    assert "Peak Current Fit Statistics:" not in printed
    assert "Field" in printed
    assert "Value" in printed
    assert "Model" in printed
    assert "Equation" in printed


def test_fit_model_fit_indices_selects_points_without_dropping_output_rows(ecat_module):
    x = np.asarray([0.0, 1.0, 2.0, 3.0, 4.0])
    y = np.asarray([1.0, 3.0, 5.0, 100.0, 200.0])

    result = ecat_module.fit_model(
        x,
        y,
        model="linear",
        options={"fit indices": [0, 3]},
    )

    assert len(result.table) == 5
    assert result.fit_table["Fit Points"].iloc[0] == 3
    first_key = next(iter(result.fits))
    assert first_key == "Model"
    assert result.fits[first_key]["parameters"]["m"] == pytest.approx(2.0)
    assert result.fits[first_key]["parameters"]["b"] == pytest.approx(1.0)
    assert result.table.loc[4, "Predicted"] == pytest.approx(9.0)


def test_fit_model_fit_indices_selects_row_window(ecat_module):
    x = np.asarray([0.0, 1.0, 2.0, 3.0, 4.0])
    y = np.asarray([1.0, 3.0, 5.0, 100.0, 200.0])

    result = ecat_module.fit_model(
        x,
        y,
        model="linear",
        options={"fit indices": [0, 2]},
    )

    assert result.fit_table["Fit Points"].iloc[0] == 2
    model_fit = result.fits[next(iter(result.fits))]
    assert model_fit["parameters"]["m"] == pytest.approx(2.0)
    assert model_fit["parameters"]["b"] == pytest.approx(1.0)
    assert result.fit_table["fit x min"].iloc[0] == pytest.approx(0.0)
    assert result.fit_table["fit x max"].iloc[0] == pytest.approx(1.0)


def test_fit_model_bounds_accept_none_as_unbounded(ecat_module):
    x = np.asarray([0.0, 1.0, 2.0, 3.0])
    y = 2.0 * x - 1.0

    result = ecat_module.fit_model(
        x,
        y,
        model="linear",
        options={
            "plot": False,
            "print": False,
            "fit bounds": {
                "m": [0.0, None],
                "b": [None, 0.0],
            },
        },
    )

    assert result.fits["parameters"]["m"] == pytest.approx(2.0)
    assert result.fits["parameters"]["b"] == pytest.approx(-1.0)
    assert result.fit_model_results["Model"]["bounds"]["m"] == (0.0, np.inf)
    assert result.fit_model_results["Model"]["bounds"]["b"] == (-np.inf, 0.0)


def test_fit_rate_custom_model_bounds_accept_none_as_unbounded(ecat_module):
    h2o = np.asarray([0.0, 0.1, 0.2, 0.5, 1.0])
    kobs = 8.5 + 50.0 * h2o + 100.0 * h2o**2
    df = pd.DataFrame({"H2O Concentration (M)": h2o, "kobs": kobs})

    result = ecat_module.fit_rate(
        df,
        {
            "plot": False,
            "print": False,
            "x column": "H2O Concentration (M)",
            "metric": "kobs",
            "fit model": "k0 + k1*x + k2*x^2",
            "fit bounds": {
                "k0": [0.0, None],
                "k1": [0.0, None],
                "k2": [0.0, None],
            },
        },
    )

    params = result.fit_model_results["kobs"]["parameters"]
    assert params["k0"] == pytest.approx(8.5)
    assert params["k1"] == pytest.approx(50.0)
    assert params["k2"] == pytest.approx(100.0)
    assert result.fit_model_results["kobs"]["bounds"]["k2"] == (0.0, np.inf)


def test_fit_model_fit_indices_list_of_pair_ranges_makes_multiple_fits(ecat_module):
    x = np.arange(10, dtype=float)
    y = 2.0 * x + 1.0

    result = ecat_module.fit_model(
        x,
        y,
        model="linear",
        options={"fit indices": [[0, 2], [4, 7]], "print": False, "plot": False},
    )

    assert isinstance(result.fits, dict)
    assert len(result.fits) == 2
    assert set(result.fits.keys()) == {"Fit 1", "Fit 2"}
    for model_key in ("Fit 1", "Fit 2"):
        params = result.fits[model_key]["parameters"]
        assert params["m"] == pytest.approx(2.0, rel=1e-5)
        assert params["b"] == pytest.approx(1.0, rel=1e-5)


def test_fit_model_fit_indices_nested_windows_as_unnamed_fit(ecat_module):
    x = np.arange(10, dtype=float)
    y = 2.0 * x + 1.0

    result = ecat_module.fit_model(
        x,
        y,
        model="linear",
        options={"fit indices": [[[0, 2], [4, 7]]], "print": False, "plot": False},
    )

    assert isinstance(result.fits, dict)
    assert len(result.fits) == 1
    model_key = next(iter(result.fits.keys()))
    params = result.fits[model_key]["parameters"]
    assert params["m"] == pytest.approx(2.0, rel=1e-5)
    assert params["b"] == pytest.approx(1.0, rel=1e-5)


def test_fit_model_fit_indices_nested_windows_inside_named_entry(ecat_module):
    x = np.arange(10, dtype=float)
    y = 2.5 * x + 3.0

    result = ecat_module.fit_model(
        x,
        y,
        model="linear",
        options={
            "fit indices": {"tail": [[0, 2], [4, None]], "head": [0, 2]},
            "print": False,
            "plot": False,
        },
    )

    assert set(result.fits.keys()) == {"tail", "head"}
    tail_params = result.fits["tail"]["parameters"]
    head_params = result.fits["head"]["parameters"]
    assert tail_params["m"] == pytest.approx(2.5, rel=1e-5)
    assert tail_params["b"] == pytest.approx(3.0, rel=1e-5)
    assert head_params["m"] == pytest.approx(2.5, rel=1e-5)
    assert head_params["b"] == pytest.approx(3.0, rel=1e-5)


def test_fit_model_fit_indices_rejects_mixed_nested_and_index_inputs(ecat_module):
    x = np.arange(10, dtype=float)
    y = np.arange(10, dtype=float)

    with pytest.raises(ValueError, match="'fit indices'"):
        ecat_module.fit_model(
            x,
            y,
            model="linear",
            options={
                "fit indices": [[0, 2], 4],
                "plot": False,
            },
        )


def test_fit_model_fit_indices_multi_list_dicts(ecat_module):
    x = np.arange(10, dtype=float)
    y = 2.0 * x + 1.0

    result = ecat_module.fit_model(
        x,
        y,
        model="linear",
        options={"fit indices": [[0, 2], [4, 7], [4, None]], "print": False, "plot": False},
    )

    assert isinstance(result.fits, dict)
    assert len(result.fits) == 3
    assert set(result.fits.keys()) == {"Fit 1", "Fit 2", "Fit 3"}
    for model_key in ("Fit 1", "Fit 2", "Fit 3"):
        params = result.fits[model_key]["parameters"]
        assert params["m"] == pytest.approx(2.0, rel=1e-5)
        assert params["b"] == pytest.approx(1.0, rel=1e-5)


def test_fit_model_overlay_does_not_replot_data_by_default(ecat_module):
    x = np.asarray([1.0, 2.0, 4.0, 8.0])
    y = 2.0 * x ** 0.5
    plt.figure()
    plt.scatter(x, y, label="Existing Data")
    ax = plt.gca()
    initial_collections = len(ax.collections)

    ecat_module.fit_model(
        x,
        y,
        model="power",
        options={"plot": True, "new plot": False},
    )

    assert plt.gca() is ax
    assert len(ax.collections) == initial_collections
    assert len(ax.lines) == 1


def test_fit_model_michaelis_menten_accepts_scatter_fit_result(ecat_module):
    x = np.asarray([0.5, 1.0, 2.0, 5.0, 10.0])
    y = 12.0 * x / (2.5 + x)
    base = ecat_module.fit_rate(
        pd.DataFrame({"Substrate Concentration (M)": x, "kobs": y}),
        {
            "plot": False,
            "print": False,
            "fit": False,
            "x column": "Substrate Concentration (M)",
            "metric": "kobs",
        },
    )

    result = ecat_module.fit_model(
        base,
        model="michaelis-menten",
        options={"plot": False, "print": False},
    )

    params = result.summary["parameters"]
    assert result.summary["model"] == "michaelis_menten"
    assert params["Vmax"] == pytest.approx(12.0, rel=1e-5)
    assert params["Km"] == pytest.approx(2.5, rel=1e-5)


def test_fit_model_accepts_custom_callable_with_inferred_parameters(ecat_module):
    def quadratic_model(x, k0, k1, k2):
        return k0 + k1 * x + k2 * x ** 2

    x = np.asarray([0.0, 1.0, 2.0, 3.0, 4.0])
    y = 1.5 + 2.0 * x + 0.25 * x ** 2

    result = ecat_module.fit_model(x, y, model=quadratic_model, options={"plot": False})

    assert result.summary["model"] == "custom"
    assert result.summary["equation"] == "quadratic_model"
    assert result.fits["parameters"]["k0"] == pytest.approx(1.5)
    assert result.fits["parameters"]["k1"] == pytest.approx(2.0)
    assert result.fits["parameters"]["k2"] == pytest.approx(0.25)


def test_fit_model_callable_fit_label_omits_function_name_by_default(ecat_module):
    def quadratic_model(x, k0, k1, k2):
        return k0 + k1 * x + k2 * x ** 2

    x = np.asarray([0.0, 1.0, 2.0, 3.0, 4.0])
    y = 1.5 + 2.0 * x + 0.25 * x ** 2

    ecat_module.fit_model(
        x,
        y,
        model=quadratic_model,
        options={"plot": True, "print": False, "fit label": True},
    )

    labels = [text.get_text() for text in plt.gca().get_legend().get_texts()]
    fit_labels = [label for label in labels if "Custom Fit" in label]
    assert fit_labels
    assert not any("quadratic_model" in label for label in fit_labels)
    assert any("R^2" in label for label in fit_labels)


def test_fit_model_rejects_removed_fit_equation_label_option(ecat_module):
    with pytest.raises(ecat_module.OptionError, match="fit equation label"):
        ecat_module.fit_model(
            [1.0, 2.0, 3.0],
            [2.0, 4.0, 6.0],
            model="linear",
            options={
                "plot": False,
                "fit equation label": "y = m*x + b",
            },
        )


def test_fit_model_accepts_formula_string_with_caret_power(ecat_module):
    x = np.asarray([0.0, 1.0, 2.0, 3.0, 4.0])
    y = 1.5 + 2.0 * x + 0.25 * x ** 2

    result = ecat_module.fit_model(
        x,
        y,
        options={
            "plot": False,
            "fit model": "k0 + k1*x + k2*x^2",
            "init": {"k0": 1, "k1": 1, "k2": 0},
        },
    )

    assert result.summary["model"] == "custom"
    assert result.summary["equation"] == "k0 + k1*x + k2*x^2"
    assert result.fits["parameters"]["k0"] == pytest.approx(1.5)
    assert result.fits["parameters"]["k1"] == pytest.approx(2.0)
    assert result.fits["parameters"]["k2"] == pytest.approx(0.25)


def test_fit_model_custom_formula_plot_label_braces_decimal_power(ecat_module):
    x = np.asarray([1.0, 2.0, 4.0, 8.0])
    y = 2.5 * x ** 1.5

    ecat_module.fit_model(
        x,
        y,
        options={
            "plot": True,
            "print": False,
            "fit label": True,
            "fit model": "A*x^1.5",
            "fit init": {"A": 1.0},
            "sig figs": 3,
        },
    )

    labels = [text.get_text() for text in plt.gca().get_legend().get_texts()]
    assert any("x^{1.5}" in label for label in labels)
    assert not any("x^1.5" in label for label in labels)


def test_fit_model_custom_formula_eval_errors_are_user_friendly(ecat_module, monkeypatch):
    def broken_function(_x):
        raise KeyError("__import__")

    monkeypatch.setitem(
        ecat_module._FIT_MODEL_ALLOWED_FORMULA_FUNCTIONS,
        "broken",
        broken_function,
    )

    with pytest.raises(ValueError, match="Could not evaluate custom fit model"):
        ecat_module.fit_model(
            [1.0, 2.0, 3.0],
            [2.0, 4.0, 6.0],
            options={
                "plot": False,
                "fit model": "a + broken(x)",
                "fit init": {"a": 1.0},
            },
        )


def test_fit_rate_fit_model_accepts_custom_formula_and_model_params(ecat_module):
    x = np.asarray([0.0, 1.0, 2.0, 3.0, 4.0])
    df = pd.DataFrame({"Scan Rate": x, "kobs": 1.5 + 2.0 * x + 0.25 * x ** 2})

    result = ecat_module.fit_rate(
        df,
        {
            "plot": False,
            "print": False,
            "metric": "kobs",
            "fit model": "k0 + k1*x + k2*x^2",
            "fit params": ["k0", "k1", "k2"],
            "fit init": {"k0": 1, "k1": 1, "k2": 0},
        },
    )

    assert result.fit_table["model"].unique().tolist() == ["custom"]
    assert result.fits["parameters"]["k2"] == pytest.approx(0.25)


def test_fit_rate_rejects_retired_model_option_prefix(ecat_module):
    df = pd.DataFrame({"Scan Rate": [1.0, 2.0, 3.0], "kobs": [2.0, 4.0, 6.0]})

    with pytest.raises(ecat_module.OptionError, match="model init"):
        ecat_module.fit_rate(
            df,
            {
                "plot": False,
                "print": False,
                "fit model": "linear",
                "model init": {"m": 1, "b": 0},
            },
        )


def test_fit_rate_fit_model_uses_requested_nonlinear_model(ecat_module):
    x = np.asarray([1.0, 2.0, 4.0, 8.0])
    df = pd.DataFrame({"Scan Rate": x, "kobs": 2.5 * x ** 0.75})

    result = ecat_module.fit_rate(
        df,
        {
            "plot": False,
            "print": False,
            "metric": "kobs",
            "fit model": "power",
        },
    )

    assert isinstance(result, ecat_module.ScatterFitResult)
    assert result.fit_table["model"].unique().tolist() == ["power"]
    assert result.fits["parameters"]["n"] == pytest.approx(0.75, rel=1e-5)


def test_fit_rate_default_fit_uses_shared_linear_model_result(ecat_module):
    df = pd.DataFrame({"Scan Rate": [1.0, 2.0, 3.0], "kobs": [2.0, 4.0, 6.0]})

    result = ecat_module.fit_rate(df, {"plot": False, "print": False})

    assert isinstance(result, ecat_module.ScatterFitResult)
    assert result.fit_table["model"].unique().tolist() == ["linear"]
    assert result.fits["parameters"]["m"] == pytest.approx(2.0)
    assert result.fits["parameters"]["b"] == pytest.approx(0.0)


def test_fit_rate_fit_false_returns_modern_result_shape_without_fit(ecat_module):
    df = pd.DataFrame({"Scan Rate": [1.0, 2.0, 3.0], "kobs": [2.0, 4.0, 6.0]})

    result = ecat_module.fit_rate(df, {"plot": False, "print": False, "fit": False})

    assert isinstance(result, ecat_module.ScatterFitResult)
    assert result.fits == {}
    assert result.fit_table.empty
    assert "x raw" in result.table
    assert "y adjusted" in result.table


def test_fit_rate_hides_single_entry_legend_by_default(ecat_module):
    df = pd.DataFrame(
        {
            "Scan Rate": [1.0, 2.0, 3.0],
            "kobs": [2.0, 4.0, 6.0],
        }
    )

    ecat_module.fit_rate(
        df,
        {
            "plot": True,
            "print": False,
            "metric": "kobs",
            "plot fit": False,
        },
    )

    assert plt.gca().get_legend() is None


def test_fit_rate_can_show_data_legend_when_requested(ecat_module):
    df = pd.DataFrame(
        {
            "Scan Rate": [1.0, 2.0, 3.0],
            "kobs": [2.0, 4.0, 6.0],
        }
    )

    ecat_module.fit_rate(
        df,
        {
            "plot": True,
            "print": False,
            "metric": "kobs",
            "plot fit": False,
            "legend": True,
        },
    )

    legend = plt.gca().get_legend()
    assert legend is not None
    assert [text.get_text() for text in legend.get_texts()] == [r"$k_{\mathrm{obs}}$"]


def test_fit_rate_loglog_falls_back_to_abs_y_when_all_y_negative(ecat_module):
    df = pd.DataFrame(
        {
            "Scan Rate": [0.1, 1.0, 10.0],
            "kobs": [-1.0, -10.0, -100.0],
        }
    )

    result = ecat_module.fit_rate(
        df,
        {
            "plot": False,
            "print": False,
            "metric": "kobs",
            "transform mode": "log-log",
        },
    )
    data = result.table
    fitline = result.fits

    assert data["y transform note"].unique().tolist() == ["abs fallback"]
    assert data["y transformed"].tolist() == pytest.approx([0.0, 1.0, 2.0])
    assert fitline["parameters"]["m"] == pytest.approx(1.0)


def test_fit_rate_accepts_dataclass_options(ecat_module):
    df = pd.DataFrame(
        {
            "Scan Rate": [1.0, 2.0, 3.0],
            "kobs": [2.0, 4.0, 6.0],
        }
    )
    options = ecat_module.FitRateOptions.from_options(
        {"plot": False, "print": False, "metric": "kobs"}
    )

    result = ecat_module.fit_rate(df, options)
    data = result.table
    fitline = result.fits

    assert data["y transformed"].tolist() == pytest.approx([2.0, 4.0, 6.0])
    assert fitline["parameters"]["m"] == pytest.approx(2.0)


def test_fit_rate_unknown_option_suggests_metric(ecat_module):
    df = pd.DataFrame(
        {
            "Scan Rate": [1.0, 2.0, 3.0],
            "kobs": [2.0, 4.0, 6.0],
        }
    )

    with pytest.raises(ValueError, match="metric"):
        ecat_module.fit_rate(df, {"metrc": "kobs", "plot": False, "print": False})


def test_fit_peak_potential_accepts_dataclass_options_at_public_boundary(ecat_module):
    options = ecat_module.FitPeakPotentialOptions.from_options({"plot": False, "print": False})

    result = ecat_module.fit_peak_potential([], options)

    assert result.table is None
    assert result.fits is None


def test_fit_peak_potential_unknown_option_suggests_segment(ecat_module):
    with pytest.raises(ValueError, match="segment"):
        ecat_module.fit_peak_potential([], {"segmnt": 1, "plot": False, "print": False})


def test_fit_peak_potential_excludes_nonpositive_x_before_log_transform(ecat_module, cv_factory):
    cvs = [
        cv_factory(name="50mVs_sample_CO2_MeCN_0mMH2O_run01"),
        cv_factory(name="50mVs_sample_CO2_MeCN_10mMH2O_run02"),
        cv_factory(name="50mVs_sample_CO2_MeCN_20mMH2O_run03"),
    ]

    result = ecat_module.fit_peak_potential(
        cvs,
        {
            "plot": False,
            "print": False,
            "exact potential": 0.2,
        },
    )
    data = result.table
    fits = result.fits

    assert np.isnan(data.loc[0, "x transformed"])
    assert data.loc[1:, "x transformed"].tolist() == pytest.approx([-2.0, np.log10(0.02)])
    assert "Ep" in fits or set(fits.get("parameters", {})) == {"m", "b"}


def test_fit_peak_potential_fit_model_uses_requested_model(ecat_module, cv_factory):
    cvs = [
        cv_factory(name="50mVs_sample_CO2_MeCN_10mM_Fc_run01"),
        cv_factory(name="100mVs_sample_CO2_MeCN_10mM_Fc_run02"),
        cv_factory(name="200mVs_sample_CO2_MeCN_10mM_Fc_run03"),
    ]
    for obj, ep in zip(cvs, [-0.1, -0.2, -0.4]):
        obj.peak_potential = lambda _options, value=ep: {"Ep": value, "index": 0}

    result = ecat_module.fit_peak_potential(
        cvs,
        {
            "plot": False,
            "print": False,
            "fit model": "linear",
        },
    )

    assert result.fit_table["model"].unique().tolist() == ["linear"]
    assert set(result.fits["parameters"]) == {"m", "b"}
    assert result.fit_table["Fit Points"].iloc[0] == 3


def test_fit_peak_potential_handles_multiple_segments(ecat_module, cv_factory):
    forward_potential = np.linspace(-0.25, 0.25, 31)
    reverse_potential = np.linspace(0.25, -0.25, 31)
    potential = np.concatenate([forward_potential, reverse_potential])
    current = np.concatenate(
        [
            np.exp(-((forward_potential - 0.05) / 0.06) ** 2),
            -np.exp(-((reverse_potential + 0.05) / 0.06) ** 2),
        ]
    )
    cvs = [
        cv_factory(
            name="50mVs_sample_CO2_MeCN_10mM_Fc_run01",
            potential=potential,
            current=current,
        ),
        cv_factory(
            name="100mVs_sample_CO2_MeCN_10mM_Fc_run02",
            potential=potential,
            current=current * 1.1,
        ),
    ]

    result = ecat_module.fit_peak_potential(
        cvs,
        {
            "plot": False,
            "print": False,
            "plot all": True,
            "segments": [1, 2],
            "exact potential": 0.2,
            "follow e1/2": True,
            "noise window": 5,
            "noise polyorder": 2,
        },
    )

    data = result.table
    fits = result.fits

    assert isinstance(result, ecat_module.ScatterFitResult)
    assert "Seg 1 Ep (V)" in data.columns
    assert "Seg 2 Ep (V)" in data.columns
    assert "Seg 1-2 E1/2 (V)" in data.columns
    assert "Seg 1 Ep" in fits


def test_fit_peak_potential_ehalf_fit_uses_point_color(ecat_module, cv_factory):
    cvs = [
        cv_factory(name="50mVs_sample_CO2_MeCN_10mM_Fc_run01"),
        cv_factory(name="100mVs_sample_CO2_MeCN_10mM_Fc_run02"),
        cv_factory(name="200mVs_sample_CO2_MeCN_10mM_Fc_run03"),
    ]

    ecat_module.fit_peak_potential(
        cvs,
        {
            "plot": True,
            "print": False,
            "segments": [1, 2],
            "exact potential": 0.2,
            "follow e1/2": True,
        },
    )
    ax = plt.gca()

    point_colors = [
        ecat_module.mpl.colors.to_hex(collection.get_facecolors()[0])
        for collection in ax.collections
        if len(collection.get_facecolors())
    ]
    fit_colors = [
        ecat_module.mpl.colors.to_hex(line.get_color())
        for line in ax.lines
    ]

    assert fit_colors == point_colors


def test_fit_peak_potential_plot_all_legend_defaults_to_raw_multiplot_only(
    ecat_module,
    cv_factory,
):
    cvs = [
        cv_factory(name="50mVs_sample_CO2_MeCN_5mMH2O_run01"),
        cv_factory(name="50mVs_sample_CO2_MeCN_10mMH2O_run02"),
        cv_factory(name="50mVs_sample_CO2_MeCN_20mMH2O_run03"),
    ]

    ecat_module.fit_peak_potential(
        cvs,
        {
            "plot": True,
            "print": False,
            "plot all": True,
            "exact potential": 0.2,
        },
    )

    figures = [plt.figure(number) for number in plt.get_fignums()]
    raw_ax = figures[0].axes[0]
    fit_ax = figures[-1].axes[0]

    custom_legend_axes = [
        ax for ax in getattr(raw_ax, "child_axes", [])
        if not ax.axison
    ]
    assert raw_ax.get_legend() is not None or custom_legend_axes
    assert fit_ax.get_legend() is None
    assert fit_ax.get_xlabel() == r"$\log_{10}$([H$_2$O] / M)"


def test_fit_peak_potential_plot_all_accepts_plot_segment_without_analysis_leak(
    ecat_module,
    cv_factory,
    monkeypatch,
):
    cvs = [
        cv_factory(name="50mVs_sample_CO2_MeCN_5mMH2O_run01"),
        cv_factory(name="50mVs_sample_CO2_MeCN_10mMH2O_run02"),
    ]
    multiplot_calls = []
    peak_calls = []

    def capture_multiplot(echem_list, options=None):
        multiplot_calls.append(dict(options or {}))
        return []

    def fake_peak_potential(self, options=None):
        peak_calls.append(dict(options or {}))
        return {"Ep": 0.2, "index": 8}

    monkeypatch.setattr(ecat_module, "multiplot", capture_multiplot)
    monkeypatch.setattr(ecat_module.cv, "peak_potential", fake_peak_potential)

    ecat_module.fit_peak_potential(
        cvs,
        {
            "plot": False,
            "print": False,
            "plot all": True,
            "segment": 1,
            "plot segment": 2,
        },
    )

    assert multiplot_calls[0]["plot segment"] == 2
    assert "segment" not in multiplot_calls[0]
    assert peak_calls[0]["segment"] == 1
    assert "plot segment" not in peak_calls[0]


def test_fit_peak_potential_per_cv_guesses_seed_running_guess(
    ecat_module,
    cv_factory,
    monkeypatch,
):
    cvs = [
        cv_factory(name="50mVs_sample_CO2_MeCN_10mM_Fc_run01"),
        cv_factory(name="100mVs_sample_CO2_MeCN_10mM_Fc_run02"),
        cv_factory(name="200mVs_sample_CO2_MeCN_10mM_Fc_run03"),
    ]
    observed = []

    def fake_peak_potential(self, options=None):
        opts = dict(options or {})
        observed.append((self.name, opts.get("segment"), opts.get("guess potential")))
        guess = float(opts.get("guess potential", 0.0))
        ep = guess + (0.01 if opts.get("segment") == 1 else 0.02)
        return {"Ep": ep, "index": 0, "current": 0.0}

    monkeypatch.setattr(ecat_module.cv, "peak_potential", fake_peak_potential)

    ecat_module.fit_peak_potential(
        cvs,
        {
            "plot": False,
            "print": False,
            "segments": [1, 2],
            "guess potentials": [-0.11, -0.22, -0.33],
        },
    )

    assert observed[0] == (cvs[0].name, 1, pytest.approx(-0.11))
    assert observed[1] == (cvs[0].name, 2, pytest.approx(-0.10))
    assert observed[2] == (cvs[1].name, 1, pytest.approx(-0.22))
    assert observed[3] == (cvs[1].name, 2, pytest.approx(-0.21))
    assert observed[4] == (cvs[2].name, 1, pytest.approx(-0.33))
    assert observed[5] == (cvs[2].name, 2, pytest.approx(-0.32))


def test_fit_peak_potential_scalar_guess_does_not_track_across_cvs(
    ecat_module,
    cv_factory,
    monkeypatch,
):
    cvs = [
        cv_factory(name="50mVs_sample_CO2_MeCN_10mM_Fc_run01"),
        cv_factory(name="100mVs_sample_CO2_MeCN_10mM_Fc_run02"),
        cv_factory(name="200mVs_sample_CO2_MeCN_10mM_Fc_run03"),
    ]
    observed = []
    returned_eps = iter([-0.11, -0.22, -0.33])

    def fake_peak_potential(self, options=None):
        opts = dict(options or {})
        observed.append((self.name, opts.get("guess potential"), opts.get("peak kind")))
        return {
            "Ep": next(returned_eps),
            "index": 0,
            "current": 0.0,
            "extremum kind": "max",
        }

    monkeypatch.setattr(ecat_module.cv, "peak_potential", fake_peak_potential)

    ecat_module.fit_peak_potential(
        cvs,
        {
            "plot": False,
            "print": False,
            "segment": 1,
            "guess potentials": -0.10,
        },
    )

    assert observed == [
        (cvs[0].name, pytest.approx(-0.10), None),
        (cvs[1].name, pytest.approx(-0.10), None),
        (cvs[2].name, pytest.approx(-0.10), None),
    ]


def test_fit_peak_potential_forwards_peak_prominence_none(
    ecat_module,
    cv_factory,
    monkeypatch,
):
    cvs = [
        cv_factory(name="50mVs_sample_CO2_MeCN_10mM_Fc_run01"),
        cv_factory(name="100mVs_sample_CO2_MeCN_10mM_Fc_run02"),
    ]
    observed = []

    def fake_peak_potential(self, options=None):
        opts = dict(options or {})
        observed.append(opts)
        return {
            "Ep": -0.1,
            "index": 0,
            "current": 0.0,
            "extremum kind": "max",
        }

    monkeypatch.setattr(ecat_module.cv, "peak_potential", fake_peak_potential)

    ecat_module.fit_peak_potential(
        cvs,
        {
            "plot": False,
            "print": False,
            "segment": 1,
            "guess potential": -0.10,
            "peak prominence": None,
        },
    )

    assert observed
    assert all("peak prominence" in opts for opts in observed)
    assert all(opts["peak prominence"] is None for opts in observed)


def test_fit_peak_potential_forwards_infer_peak_kind(
    ecat_module,
    cv_factory,
    monkeypatch,
):
    cvs = [
        cv_factory(name="50mVs_sample_CO2_MeCN_10mM_Fc_run01"),
        cv_factory(name="100mVs_sample_CO2_MeCN_10mM_Fc_run02"),
    ]
    observed = []
    returned_eps = iter([-0.11, -0.22])

    def fake_peak_potential(self, options=None):
        opts = dict(options or {})
        observed.append(opts.get("peak kind"))
        return {
            "Ep": next(returned_eps),
            "index": 0,
            "current": 0.0,
            "extremum kind": "max",
        }

    monkeypatch.setattr(ecat_module.cv, "peak_potential", fake_peak_potential)

    ecat_module.fit_peak_potential(
        cvs,
        {
            "plot": False,
            "print": False,
            "segment": 1,
            "guess potentials": -0.10,
            "peak kind": "inferred",
        },
    )

    assert observed == ["infer", "infer"]


def test_fit_peak_current_formats_concentration_transform_symbolically(
    ecat_module,
    cv_factory,
):
    cvs = [
        cv_factory(name="50mVs_sample_CO2_MeCN_5mMH2O_run01"),
        cv_factory(name="50mVs_sample_CO2_MeCN_10mMH2O_run02"),
        cv_factory(name="50mVs_sample_CO2_MeCN_20mMH2O_run03"),
    ]

    ecat_module.fit_peak_current(
        cvs,
        {
            "plot": True,
            "print": False,
            "plot all": True,
            "legend": False,
            "exact potential": 0.2,
            "x transform": "^0.5",
            "noise window": 5,
            "noise polyorder": 2,
            "peak prominence": 1e-7,
            "tangent range": [0.05, 0.3],
            "percent threshold": 100,
        },
    )

    figures = [plt.figure(number) for number in plt.get_fignums()]
    raw_ax = figures[0].axes[0]
    fit_ax = figures[-1].axes[0]

    assert raw_ax.get_legend() is None
    assert fit_ax.get_legend() is None
    assert fit_ax.get_xlabel() == r"[H$_2$O]$^{1/2}$ (M$^{1/2}$)"


def test_fit_peak_current_fit_model_uses_requested_model(ecat_module, cv_factory):
    cvs = [
        cv_factory(name="50mVs_sample_CO2_MeCN_10mM_Fc_run01"),
        cv_factory(name="100mVs_sample_CO2_MeCN_10mM_Fc_run02"),
        cv_factory(name="200mVs_sample_CO2_MeCN_10mM_Fc_run03"),
    ]
    for obj, ip in zip(cvs, [2.0e-6, 4.0e-6, 8.0e-6]):
        obj.peak_current = lambda _options, value=ip: {"ip": value, "index": 0}

    result = ecat_module.fit_peak_current(
        cvs,
        {
            "plot": False,
            "print": False,
            "x transform": "identity",
            "fit model": "power",
        },
    )

    assert result.fit_table["model"].unique().tolist() == ["power"]
    assert result.fits["parameters"]["n"] == pytest.approx(1.0, rel=1e-5)


def test_fit_peak_current_print_uses_shared_fit_model_output(ecat_module, cv_factory, capsys):
    cvs = [
        cv_factory(name="50mVs_sample_CO2_MeCN_10mM_Fc_run01"),
        cv_factory(name="100mVs_sample_CO2_MeCN_10mM_Fc_run02"),
        cv_factory(name="200mVs_sample_CO2_MeCN_10mM_Fc_run03"),
    ]
    for obj, ip in zip(cvs, [2.0e-6, 4.0e-6, 8.0e-6]):
        obj.peak_current = lambda _options, value=ip: {"ip": value, "index": 0}

    ecat_module.fit_peak_current(
        cvs,
        {
            "plot": False,
            "print": True,
            "x transform": "identity",
            "fit model": "linear",
        },
    )

    printed = capsys.readouterr().out
    assert "Fit Model:" in printed
    assert "Peak Current Fit Statistics:" not in printed
    assert "Field" in printed
    assert "Value" in printed


def test_fit_peak_current_multi_series_detailed_print_uses_two_tables(ecat_module, cv_factory, capsys):
    cvs = [
        cv_factory(name="50mVs_sample_CO2_MeCN_10mM_Fc_run01"),
        cv_factory(name="100mVs_sample_CO2_MeCN_10mM_Fc_run02"),
        cv_factory(name="200mVs_sample_CO2_MeCN_10mM_Fc_run03"),
    ]

    for obj in cvs:
        scan_rate = float(obj.scan_rate)
        obj.peak_current = (
            lambda _options, rate=scan_rate: {
                "ip": (2.0e-5 if _options.get("segment", 1) == 1 else 3.0e-5) * rate,
                "index": 0,
            }
        )

    ecat_module.fit_peak_current(
        cvs,
        {
            "plot": False,
            "print": True,
            "segments": [1, 2],
            "x transform": "identity",
            "fit model": "linear",
            "fit bounds": {"m": [-np.inf, np.inf], "b": [-np.inf, np.inf]},
            "fit init": {"m": 1.0, "b": 0.0},
        },
    )

    printed = capsys.readouterr().out
    assert "Fit Model Details:" in printed
    assert "Fit Model Parameters:" in printed
    assert "Setting" in printed
    assert "Seg 1" in printed
    assert "Seg 2" in printed
    assert "Series" in printed
    assert "Peak Current Fit Statistics:" not in printed


def test_fit_peak_current_plot_all_accepts_plot_segment_without_analysis_leak(
    ecat_module,
    cv_factory,
    monkeypatch,
):
    cvs = [
        cv_factory(name="50mVs_sample_CO2_MeCN_5mMH2O_run01"),
        cv_factory(name="50mVs_sample_CO2_MeCN_10mMH2O_run02"),
    ]
    multiplot_calls = []
    peak_calls = []

    def capture_multiplot(echem_list, options=None):
        multiplot_calls.append(dict(options or {}))
        return []

    def fake_peak_current(self, options=None):
        peak_calls.append(dict(options or {}))
        return {"ip": 1e-6, "tangent line": [0.0, 0.0], "tangent start": 0}

    monkeypatch.setattr(ecat_module, "multiplot", capture_multiplot)
    monkeypatch.setattr(ecat_module.cv, "peak_current", fake_peak_current)

    ecat_module.fit_peak_current(
        cvs,
        {
            "plot": False,
            "print": False,
            "plot all": True,
            "segment": 1,
            "plot segment": 2,
            "exact potential": 0.2,
        },
    )

    assert multiplot_calls[0]["plot segment"] == 2
    assert "segment" not in multiplot_calls[0]
    assert peak_calls[0]["segment"] == 1
    assert "plot segment" not in peak_calls[0]


def test_fit_peak_current_passes_per_cv_potential_options(
    ecat_module,
    cv_factory,
    monkeypatch,
):
    cvs = [
        cv_factory(name="50mVs_sample_CO2_MeCN_10mM_Fc_run01"),
        cv_factory(name="100mVs_sample_CO2_MeCN_10mM_Fc_run02"),
    ]
    observed = []

    def fake_peak_current(self, options=None):
        observed.append(dict(options or {}))
        return {"ip": float(self.scan_rate) * 1e-6, "tangent line": [0.0, 0.0], "tangent start": 0}

    monkeypatch.setattr(ecat_module.cv, "peak_current", fake_peak_current)

    ecat_module.fit_peak_current(
        cvs,
        {
            "plot": False,
            "print": False,
            "guess potentials": [-0.11, -0.22],
            "tangent potentials": [-0.31, -0.42],
            "peak kind": "max",
            "peak fallback": None,
            "plot peak potential": False,
            "x transform": "identity",
        },
    )

    assert [opts["guess potential"] for opts in observed] == pytest.approx([-0.11, -0.22])
    assert [opts["tangent potential"] for opts in observed] == pytest.approx([-0.31, -0.42])
    assert all(opts["peak kind"] == "max" for opts in observed)
    assert all(opts["peak fallback"] is None for opts in observed)
    assert all(opts["plot peak potential"] is False for opts in observed)


def test_fit_peak_current_infers_varying_mole_fraction_with_constant_molar_species(
    ecat_module,
    cv_factory,
):
    cvs = [
        cv_factory(name="50mVs_sample_CO2_MeCN_2.8MD2O_0.2xD2O_run01"),
        cv_factory(name="50mVs_sample_CO2_MeCN_2.8MD2O_0.5xD2O_run02"),
        cv_factory(name="50mVs_sample_CO2_MeCN_2.8MD2O_0.8xD2O_run03"),
    ]

    result = ecat_module.fit_peak_current(
        cvs,
        {
            "plot": False,
            "print": False,
            "exact potential": 0.2,
            "noise window": 5,
            "noise polyorder": 2,
            "peak prominence": 1e-7,
            "tangent range": [0.05, 0.3],
            "percent threshold": 100,
        },
    )

    table = result.table
    assert table["x raw"].tolist() == pytest.approx([0.2, 0.5, 0.8])
    assert table["x label"].iloc[0] == "D2O"
    assert table["x unit"].iloc[0] == "x"
    assert table["x kind"].iloc[0] == "mole fraction"


def test_fit_rate_infers_varying_mole_fraction_with_duplicate_species_tokens(
    ecat_module,
):
    df = pd.DataFrame(
        {
            "Compounds": [
                "2.8 M D2O, 0.2 x D2O",
                "2.8 M D2O, 0.5 x D2O",
                "2.8 M D2O, 0.8 x D2O",
            ],
            "kobs": [1.0, 2.0, 4.0],
        }
    )

    result = ecat_module.fit_rate(
        df,
        {"plot": False, "print": False, "plot log-log": True},
    )
    data = result.table

    assert data["x raw"].tolist() == pytest.approx([0.2, 0.5, 0.8])
    assert data["x label"].iloc[0] == "D2O"
    assert data["x unit"].iloc[0] == "x"
    assert data["x kind"].iloc[0] == "mole fraction"


def test_mole_fraction_axis_label_uses_chi_not_concentration_units(ecat_module):
    assert (
        ecat_module._format_symbolic_axis_label(
            "D2O",
            unit="x",
            x_kind="mole fraction",
        )
        == r"$\chi$(D$_2$O)"
    )


def test_symbolic_axis_label_formats_unit_exponents(ecat_module):
    assert (
        ecat_module._format_symbolic_axis_label("kobs", unit="s^-1")
        == r"kobs (s^{-1})"
    )


def test_sevcik_unknown_option_suggests_num_electrons(ecat_module):
    with pytest.raises(ValueError, match="num electrons"):
        ecat_module.sevcik_analysis([], {"num electons": 1, "plot": False, "print": False})


def test_sevcik_plot_all_routes_only_multiplot_options(
    ecat_module,
    cv_factory,
    monkeypatch,
):
    cvs = [
        cv_factory(name="50mVs_sample_CO2_MeCN_10mM_Fc_run01"),
        cv_factory(name="100mVs_sample_CO2_MeCN_10mM_Fc_run02"),
    ]
    multiplot_calls = []
    peak_current_calls = []

    def capture_multiplot(echem_list, options=None):
        multiplot_calls.append(dict(options or {}))
        return []

    def fake_peak_current(self, options=None):
        peak_current_calls.append(dict(options or {}))
        return {"ip": 1e-6, "tangent line": [0.0, 0.0], "tangent start": 0}

    monkeypatch.setattr(ecat_module, "multiplot", capture_multiplot)
    monkeypatch.setattr(ecat_module.cv, "peak_current", fake_peak_current)

    ecat_module.sevcik_analysis(
        cvs,
        {
            "plot": False,
            "print": False,
            "plot all": True,
            "guess potential": -1.6,
            "segment": 1,
        },
    )

    assert len(multiplot_calls) == 1
    assert "segment" not in multiplot_calls[0]
    assert "guess potential" not in multiplot_calls[0]
    assert "segment" in peak_current_calls[0]
    assert "guess potential" in peak_current_calls[0]


def test_sevcik_plot_all_diagnostics_use_multiplot_axis_scaling(
    ecat_module,
    cv_factory,
):
    plt.close("all")
    potential = np.linspace(-0.25, 0.25, 51)
    current = (
        0.4e-6
        + 0.5e-6 * potential
        + 6.0e-6 * np.exp(-((potential - 0.15) / 0.035) ** 2)
    )
    cvs = [
        cv_factory(
            name="50mVs_sample_CO2_MeCN_10mM_Fc_run01",
            potential=potential,
            current=current,
            options={"electrode area": 0.5},
        ),
        cv_factory(
            name="100mVs_sample_CO2_MeCN_10mM_Fc_run02",
            potential=potential,
            current=current,
            options={"electrode area": 0.5},
        ),
        cv_factory(
            name="200mVs_sample_CO2_MeCN_10mM_Fc_run03",
            potential=potential,
            current=current,
            options={"electrode area": 0.5},
        ),
    ]

    ecat_module.sevcik_analysis(
        cvs,
        {
            "plot": False,
            "print": False,
            "plot all": True,
            "segment": 1,
            "exact potential": 0.15,
            "tangent range": [0.1, 0.25],
            "percent threshold": 100,
            "x unit": "mV",
            "y unit": "nA",
        },
    )

    ax = plt.gcf().axes[0]
    try:
        vertical_segments = [
            np.asarray(segment)
            for collection in ax.collections
            for segment in getattr(collection, "get_segments", lambda: [])()
            if len(segment) == 2
            and np.isclose(segment[0][0], segment[1][0])
            and collection.get_linestyle()
        ]
        peak_segment = max(
            vertical_segments,
            key=lambda segment: abs(segment[1][1] - segment[0][1]),
        )

        assert peak_segment[0][0] == pytest.approx(150.0)
        assert np.nanmax(peak_segment[:, 1]) > 1000.0
    finally:
        plt.close("all")


def test_sevcik_plot_all_diagnostics_use_common_auto_y_unit(
    ecat_module,
    cv_factory,
):
    plt.close("all")
    potential = np.linspace(-0.25, 0.25, 51)
    current_large = (
        0.4e-6
        + 0.5e-6 * potential
        + 6.0e-6 * np.exp(-((potential - 0.15) / 0.035) ** 2)
    )
    current_small = current_large * 1e-3
    cvs = [
        cv_factory(
            name="50mVs_sample_CO2_MeCN_10mM_Fc_run01",
            potential=potential,
            current=current_small,
        ),
        cv_factory(
            name="100mVs_sample_CO2_MeCN_10mM_Fc_run02",
            potential=potential,
            current=current_large,
        ),
    ]

    ecat_module.sevcik_analysis(
        cvs,
        {
            "plot": False,
            "print": False,
            "plot all": True,
            "segment": 1,
            "exact potential": 0.15,
            "tangent range": [0.1, 0.25],
            "percent threshold": 100,
        },
    )

    ax = plt.gcf().axes[0]
    try:
        vertical_lengths = [
            abs(segment[1][1] - segment[0][1])
            for collection in ax.collections
            for segment in getattr(collection, "get_segments", lambda: [])()
            if len(segment) == 2
            and np.isclose(segment[0][0], segment[1][0])
            and collection.get_linestyle()
        ]

        assert min(vertical_lengths) < 0.02
    finally:
        plt.close("all")


def test_fit_peak_current_plot_all_diagnostics_use_multiplot_axis_scaling(
    ecat_module,
    cv_factory,
):
    plt.close("all")
    potential = np.linspace(-0.25, 0.25, 51)
    current = (
        0.4e-6
        + 0.5e-6 * potential
        + 6.0e-6 * np.exp(-((potential - 0.15) / 0.035) ** 2)
    )
    cvs = [
        cv_factory(
            name="50mVs_sample_CO2_MeCN_10mM_Fc_run01",
            potential=potential,
            current=current,
        ),
        cv_factory(
            name="100mVs_sample_CO2_MeCN_10mM_Fc_run02",
            potential=potential,
            current=current,
        ),
        cv_factory(
            name="200mVs_sample_CO2_MeCN_10mM_Fc_run03",
            potential=potential,
            current=current,
        ),
    ]

    ecat_module.fit_peak_current(
        cvs,
        {
            "plot": False,
            "print": False,
            "plot all": True,
            "segment": 1,
            "exact potential": 0.15,
            "tangent range": [0.1, 0.25],
            "percent threshold": 100,
            "x unit": "mV",
            "y unit": "nA",
        },
    )

    ax = plt.gcf().axes[0]
    try:
        vertical_segments = [
            np.asarray(segment)
            for collection in ax.collections
            for segment in getattr(collection, "get_segments", lambda: [])()
            if len(segment) == 2
            and np.isclose(segment[0][0], segment[1][0])
            and collection.get_linestyle()
        ]
        peak_segment = max(
            vertical_segments,
            key=lambda segment: abs(segment[1][1] - segment[0][1]),
        )

        assert peak_segment[0][0] == pytest.approx(150.0)
        assert np.nanmax(peak_segment[:, 1]) > 1000.0
    finally:
        plt.close("all")


def test_fit_peak_potential_plot_all_markers_use_common_auto_y_unit(
    ecat_module,
    cv_factory,
):
    plt.close("all")
    potential = np.linspace(-0.25, 0.25, 51)
    current_large = (
        0.4e-6
        + 0.5e-6 * potential
        + 6.0e-6 * np.exp(-((potential - 0.15) / 0.035) ** 2)
    )
    current_small = current_large * 1e-3
    cvs = [
        cv_factory(
            name="50mVs_sample_CO2_MeCN_10mM_Fc_run01",
            potential=potential,
            current=current_small,
        ),
        cv_factory(
            name="100mVs_sample_CO2_MeCN_10mM_Fc_run02",
            potential=potential,
            current=current_large,
        ),
    ]

    ecat_module.fit_peak_potential(
        cvs,
        {
            "plot": False,
            "print": False,
            "plot all": True,
            "segment": 1,
            "exact potential": 0.15,
        },
    )

    try:
        marker_offsets = [
            np.asarray(collection.get_offsets())
            for fig_num in plt.get_fignums()
            for ax in plt.figure(fig_num).axes
            for collection in ax.collections
            if len(collection.get_offsets()) > 0
        ]
        y_offsets = np.concatenate([offsets[:, 1] for offsets in marker_offsets])

        assert np.nanmin(np.abs(y_offsets[np.isfinite(y_offsets)])) < 0.02
    finally:
        plt.close("all")


def test_sevcik_passes_per_cv_potential_options(
    ecat_module,
    cv_factory,
    monkeypatch,
):
    cvs = [
        cv_factory(name="50mVs_sample_CO2_MeCN_10mM_Fc_run01"),
        cv_factory(name="100mVs_sample_CO2_MeCN_10mM_Fc_run02"),
    ]
    observed = []

    def fake_peak_current(self, options=None):
        observed.append(dict(options or {}))
        return {"ip": float(self.scan_rate) * 1e-6, "tangent line": [0.0, 0.0], "tangent start": 0}

    monkeypatch.setattr(ecat_module.cv, "peak_current", fake_peak_current)

    ecat_module.sevcik_analysis(
        cvs,
        {
            "plot": False,
            "print": False,
            "guess potentials": [-0.11, -0.22],
            "tangent potentials": [-0.31, -0.42],
        },
    )

    assert [opts["guess potential"] for opts in observed] == pytest.approx([-0.11, -0.22])
    assert [opts["tangent potential"] for opts in observed] == pytest.approx([-0.31, -0.42])


def test_sevcik_forwards_peak_selection_and_fallback_options(
    ecat_module,
    cv_factory,
    monkeypatch,
):
    cvs = [
        cv_factory(name="50mVs_sample_CO2_MeCN_10mM_Fc_run01"),
        cv_factory(name="100mVs_sample_CO2_MeCN_10mM_Fc_run02"),
    ]
    observed = []

    def fake_peak_current(self, options=None):
        observed.append(dict(options or {}))
        return {
            "ip": float(self.scan_rate) * 1e-6,
            "tangent line": [0.0, 0.0],
            "tangent start": 0,
        }

    monkeypatch.setattr(ecat_module.cv, "peak_current", fake_peak_current)

    ecat_module.sevcik_analysis(
        cvs,
        {
            "plot": False,
            "print": False,
            "segment": 1,
            "peak kind": "max",
            "peak fallback": None,
            "plot peak potential": False,
        },
    )

    assert len(observed) == len(cvs)
    assert all(options["peak kind"] == "max" for options in observed)
    assert all(options["peak fallback"] is None for options in observed)
    assert all(options["plot peak potential"] is False for options in observed)


def test_sevcik_auto_selects_one_segment_from_anchor(
    ecat_module,
    cv_factory,
    monkeypatch,
):
    cvs = [
        cv_factory(name="50mVs_sample_CO2_MeCN_10mM_Fc_run01"),
        cv_factory(name="100mVs_sample_CO2_MeCN_10mM_Fc_run02"),
        cv_factory(name="200mVs_sample_CO2_MeCN_10mM_Fc_run03"),
    ]
    peak_current_calls = []
    peak_potential_calls = []

    def fake_peak_potential(self, options=None):
        opts = dict(options or {})
        segment = opts.get("segment")
        peak_potential_calls.append(segment)
        ep = -0.15 if segment == 1 else 0.05
        return {"Ep": ep, "index": 5, "current": 1e-6}

    def fake_peak_current(self, options=None):
        opts = dict(options or {})
        segment = opts.get("segment")
        peak_current_calls.append(segment)
        ep = -0.15 if segment == 1 else 0.05
        return {
            "ip": float(self.scan_rate) * 1e-6,
            "Ep": ep,
            "tangent line": [0.0, 0.0],
            "tangent start": 0,
        }

    monkeypatch.setattr(ecat_module.cv, "peak_potential", fake_peak_potential)
    monkeypatch.setattr(ecat_module.cv, "peak_current", fake_peak_current)

    result = ecat_module.sevcik_analysis(
        cvs,
        {
            "plot": False,
            "print": False,
            "guess potential": -0.15,
        },
    )

    assert "Seg 1 Peaks" in result.table.columns
    assert "Name" in result.table.columns
    assert "Seg 2 Peaks" not in result.table.columns
    assert result.summary["segment selection"]["mode"] == "auto"
    assert result.summary["segment selection"]["selected segment"] == 1
    assert result.summary["segment selection"]["support"] == {1: 3}
    assert peak_potential_calls
    assert peak_current_calls == [1, 1, 1]


def test_sevcik_auto_segment_selection_routes_only_peak_options(
    ecat_module,
    cv_factory,
    monkeypatch,
):
    cvs = [
        cv_factory(name="50mVs_sample_CO2_MeCN_10mM_Fc_run01"),
        cv_factory(name="100mVs_sample_CO2_MeCN_10mM_Fc_run02"),
    ]
    forbidden = {"num electrons", "fit label", "return stats"}
    observed_peak_options = []

    def fake_peak_potential(self, options=None):
        opts = dict(options or {})
        observed_peak_options.append(opts)
        assert not forbidden.intersection(opts)
        segment = opts.get("segment")
        ep = -0.15 if segment == 1 else 0.05
        return {"Ep": ep, "index": 5, "current": 1e-6}

    def fake_peak_current(self, options=None):
        opts = dict(options or {})
        segment = opts.get("segment")
        ep = -0.15 if segment == 1 else 0.05
        return {
            "ip": float(self.scan_rate) * 1e-6,
            "Ep": ep,
            "tangent line": [0.0, 0.0],
            "tangent start": 0,
        }

    monkeypatch.setattr(ecat_module.cv, "peak_potential", fake_peak_potential)
    monkeypatch.setattr(ecat_module.cv, "peak_current", fake_peak_current)

    result = ecat_module.sevcik_analysis(
        cvs,
        {
            "plot": False,
            "print": False,
            "guess potential": -0.15,
            "num electrons": 1,
            "fit label": True,
            "return stats": True,
        },
    )

    assert result.summary["segment selection"]["selected segment"] == 1
    assert observed_peak_options
    assert all(opts["guess potential"] == pytest.approx(-0.15) for opts in observed_peak_options)


def test_sevcik_fit_color_matches_points_by_default(
    ecat_module,
    cv_factory,
    monkeypatch,
):
    cvs = [
        cv_factory(name="50mVs_sample_CO2_MeCN_10mM_Fc_run01"),
        cv_factory(name="100mVs_sample_CO2_MeCN_10mM_Fc_run02"),
        cv_factory(name="200mVs_sample_CO2_MeCN_10mM_Fc_run03"),
    ]

    def fake_peak_current(self, options=None):
        segment = dict(options or {}).get("segment", 1)
        scale = 1.0 if segment == 1 else 1.5
        return {
            "ip": float(self.scan_rate) * scale * 1e-6,
            "tangent line": [0.0, 0.0],
            "tangent start": 0,
        }

    monkeypatch.setattr(ecat_module.cv, "peak_current", fake_peak_current)

    ecat_module.sevcik_analysis(cvs, {"print": False})
    ax = plt.gca()

    point_colors = [
        ecat_module.mpl.colors.to_hex(collection.get_facecolors()[0])
        for collection in ax.collections
    ]
    fit_colors = [
        ecat_module.mpl.colors.to_hex(line.get_color())
        for line in ax.lines
    ]

    assert fit_colors == point_colors
    plt.close(ax.figure)


def test_sevcik_accepts_fit_color_list(ecat_module, cv_factory, monkeypatch):
    cvs = [
        cv_factory(name="50mVs_sample_CO2_MeCN_10mM_Fc_run01"),
        cv_factory(name="100mVs_sample_CO2_MeCN_10mM_Fc_run02"),
        cv_factory(name="200mVs_sample_CO2_MeCN_10mM_Fc_run03"),
    ]

    def fake_peak_current(self, options=None):
        segment = dict(options or {}).get("segment", 1)
        scale = 1.0 if segment == 1 else 1.5
        return {
            "ip": float(self.scan_rate) * scale * 1e-6,
            "tangent line": [0.0, 0.0],
            "tangent start": 0,
        }

    monkeypatch.setattr(ecat_module.cv, "peak_current", fake_peak_current)

    ecat_module.sevcik_analysis(
        cvs,
        {"print": False, "segments": [1, 2], "fit color": ["black", "tab:orange"]},
    )
    ax = plt.gca()

    assert [ecat_module.mpl.colors.to_hex(line.get_color()) for line in ax.lines] == [
        "#000000",
        ecat_module.mpl.colors.to_hex("tab:orange"),
    ]
    plt.close(ax.figure)


def test_sevcik_prints_symbolic_equation_shared_params_and_fit_results(
    ecat_module,
    cv_factory,
    monkeypatch,
    capsys,
):
    cvs = [
        cv_factory(name="50mVs_sample_CO2_MeCN_10mM_Fc_run01", options={"electrode area": 0.071}),
        cv_factory(name="100mVs_sample_CO2_MeCN_10mM_Fc_run02", options={"electrode area": 0.071}),
        cv_factory(name="200mVs_sample_CO2_MeCN_10mM_Fc_run03", options={"electrode area": 0.071}),
    ]

    def fake_peak_current(self, options=None):
        return {"ip": float(self.scan_rate) * 1e-6, "tangent line": [0.0, 0.0], "tangent start": 0}

    monkeypatch.setattr(ecat_module.cv, "peak_current", fake_peak_current)
    monkeypatch.setattr(ecat_module.analysis_batch, "display", None)
    monkeypatch.setattr(ecat_module.analysis_batch, "Math", None)

    result = ecat_module.sevcik_analysis(
        cvs,
        {
            "plot": False,
            "print": True,
            "segment": 1,
            "C": 1e-6,
            "num electrons": 1,
        },
    )

    output = capsys.readouterr().out
    assert "Sevcik Analysis Parameters:" in output
    assert "[Sevcik Analysis Equations]" in output
    assert "D = (R * T / (F * n)^3) * (m / (0.4463 * S * C))^2" in output
    assert "D = (8.31446" not in output
    assert "m = converted fit slope" not in output
    assert "Parameter" in output
    assert "Symbol" in output
    assert "Value" in output
    assert "Electron Count" in output
    assert " n " in output or " n\n" in output
    assert "298.0 K" in output
    assert "0.07100 cm^2" in output
    assert "1.000e-06 mol/cm^3" in output
    assert "Segment   | Diffusion Coef" not in output
    assert "Sevcik Analysis Summary:" in output
    assert "Sevcik Fit Statistics:" not in output
    assert "Diffusion Coefficient" in output
    assert "cm^2/s" in output
    assert "slope" not in output
    assert "intercept" not in output
    assert output.index("Sevcik Analysis Parameters:") < output.index(
        "[Sevcik Analysis Equations]"
    )
    assert output.index("[Sevcik Analysis Equations]") < output.index(
        "Sevcik Analysis Summary:"
    )

    assert "Diffusion Coefficient" in result.fit_table.columns
    assert "slope" not in result.fit_table.columns
    assert "intercept" not in result.fit_table.columns
    assert all("cm^2/s" in value for value in result.fit_table["Diffusion Coefficient"])


def test_sevcik_missing_electrode_area_raises_clear_error(ecat_module, cv_factory, monkeypatch):
    cvs = [
        cv_factory(name="50mVs_sample_CO2_MeCN_10mM_Fc_run01"),
        cv_factory(name="100mVs_sample_CO2_MeCN_10mM_Fc_run02"),
    ]
    for obj in cvs:
        obj.electrode_area = None

    def fake_peak_current(self, options=None):
        return {"ip": float(self.scan_rate) * 1e-6}

    monkeypatch.setattr(ecat_module.cv, "peak_current", fake_peak_current)

    with pytest.raises(ValueError, match="electrode area"):
        ecat_module.sevcik_analysis(
            cvs,
            {"plot": False, "print": False, "segment": 1, "C": 1e-6},
        )


def test_sevcik_legend_uses_active_plot_style_fontsize(
    ecat_module,
    cv_factory,
    monkeypatch,
):
    ecat_module.plotting_style(True)
    cvs = [
        cv_factory(name="50mVs_sample_CO2_MeCN_10mM_Fc_run01"),
        cv_factory(name="100mVs_sample_CO2_MeCN_10mM_Fc_run02"),
    ]

    def fake_peak_current(self, options=None):
        return {"ip": float(self.scan_rate) * 1e-6, "tangent line": [0.0, 0.0], "tangent start": 0}

    monkeypatch.setattr(ecat_module.cv, "peak_current", fake_peak_current)

    ecat_module.sevcik_analysis(
        cvs,
        {
            "plot": True,
            "print": False,
            "legend": True,
            "fit label": True,
            "segment": 1,
        },
    )

    legend = plt.gca().get_legend()
    assert legend is not None
    assert legend.get_texts()[0].get_fontsize() == pytest.approx(
        ecat_module._default_legend_fontsize()
    )


def test_plot_all_multiplot_options_preserve_default_legend_mode(ecat_module):
    routed = ecat_module._plot_all_multiplot_options(
        {"legend mode": "auto", "legend": False},
        raw_options={},
    )

    assert routed["legend"] is True
    assert routed["legend mode"] == "auto"


def test_plot_all_multiplot_options_respect_explicit_legend_mode(ecat_module):
    routed = ecat_module._plot_all_multiplot_options(
        {"legend mode": "colorbar", "legend": False},
        raw_options={"legend mode": "colorbar"},
    )

    assert routed["legend mode"] == "colorbar"


def test_fit_peak_current_accepts_dataclass_options(ecat_module, cv_factory):
    cvs = [
        cv_factory(name="50mVs_sample_CO2_MeCN_10mM_Fc_run01"),
        cv_factory(name="100mVs_sample_CO2_MeCN_10mM_Fc_run02"),
    ]
    options = ecat_module.FitPeakCurrentOptions.from_options(
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

    result = ecat_module.fit_peak_current(cvs, options)
    data = result.table
    fitline = result.fits

    assert "x transformed" in data.columns
    assert "y transformed" in data.columns
    assert set(fitline["parameters"]) == {"m", "b"}


def test_fit_peak_current_uses_normalized_current_axis_for_normalized_cvs(
    ecat_module,
    cv_factory,
):
    cvs = [
        cv_factory(name="50mVs_sample_CO2_MeCN_10mM_Fc_run01"),
        cv_factory(name="100mVs_sample_CO2_MeCN_10mM_Fc_run02"),
    ]
    normalized = ecat_module.normalize_current(cvs, {"ip0": 1e-6})

    result = ecat_module.fit_peak_current(
        normalized,
        {
            "segment": 1,
            "plot": True,
            "plot all": True,
            "print": False,
            "legend": False,
            "exact potential": 0.2,
            "noise window": 5,
            "noise polyorder": 2,
            "peak prominence": 1e-7,
            "tangent range": [0.05, 0.3],
            "percent threshold": 100,
        },
    )

    table = result.table
    assert "ip" in table.columns
    assert table["ip"].iloc[0] == pytest.approx(
        cvs[0].peak_current(
            {
                "plot": False,
                "print": False,
                "exact potential": 0.2,
                "noise window": 5,
                "noise polyorder": 2,
                "peak prominence": 1e-7,
                "tangent range": [0.05, 0.3],
                "percent threshold": 100,
            }
        )["ip"] / 1e-6
    )
    fit_ax = plt.figure(plt.get_fignums()[-1]).axes[0]
    assert fit_ax.get_ylabel() == "$i_p / i_p^0$"
    plt.close("all")


def test_peak_current_defaults_to_ip0_axis_for_normalized_current_copy(
    ecat_module,
    cv_factory,
):
    cv_obj = cv_factory()
    normalized = ecat_module.normalize_current(cv_obj, {"ip0": 1e-6})

    normalized_ip = normalized.peak_current(
        {
            "plot": False,
            "print": False,
            "exact potential": 0.2,
            "noise window": 5,
            "noise polyorder": 2,
            "peak prominence": 1e-7,
            "tangent range": [0.05, 0.3],
            "percent threshold": 100,
        }
    )["ip"]

    raw_ip = cv_obj.peak_current(
        {
            "plot": False,
            "print": False,
            "exact potential": 0.2,
            "noise window": 5,
            "noise polyorder": 2,
            "peak prominence": 1e-7,
            "tangent range": [0.05, 0.3],
            "percent threshold": 100,
        }
    )["ip"]

    assert normalized_ip == pytest.approx(raw_ip / 1e-6)


def test_fit_peak_current_unknown_option_suggests_unknown(ecat_module):
    with pytest.raises(ValueError, match="Unknown option"):
        ecat_module.fit_peak_current([], {"x powr": 0.5, "plot": False, "print": False})


def test_trumpet_unknown_option_suggests_fit_indices(ecat_module):
    with pytest.raises(ValueError, match="fit indices"):
        ecat_module.trumpet_analysis([], {"fit indice": [0, 2], "plot": False, "print": False})


def test_trumpet_analysis_uses_temperature_attribute_not_legacy_T(ecat_module):
    class FakeCV:
        def __init__(self, scan_rate, ep1, ep2):
            self.scan_rate = scan_rate
            self.temperature = 305
            self._ep1 = ep1
            self._ep2 = ep2

        def half_wave_potential(self, options=None):
            return {
                "E(1/2)": 0.0,
                "ΔE": self._ep2 - self._ep1,
                "peak 1": {"Ep": self._ep1},
                "peak 2": {"Ep": self._ep2},
            }

    cvs = [
        FakeCV(0.1, 0.30, -0.30),
        FakeCV(1.0, 0.20, -0.20),
        FakeCV(10.0, 0.10, -0.10),
    ]

    result = ecat_module.trumpet_analysis(
        cvs,
        {"plot": False, "print": False, "segment": 1},
    )

    data = result.table
    fits = result.fits

    assert list(data["Scan Rates (V/s)"]) == [0.1, 1.0, 10.0]
    assert "Name" in data.columns
    assert "Cathodic Peak Potential (V)" in data.columns
    assert "Anodic Peak Potential (V)" in data.columns
    assert len(fits) == 2
    k0_row = result.fit_table.loc[result.fit_table["Parameter"] == "k0", "Value"].iloc[0]
    assert k0_row == "not computed (D not provided)"


def test_trumpet_analysis_preserves_shared_flat_paired_guess_for_more_than_two_cvs(ecat_module):
    class FakeCV:
        def __init__(self, scan_rate):
            self.name = f"fake-{scan_rate:g}"
            self.scan_rate = scan_rate
            self.temperature = 298
            self.observed_options = None

        def analysis_segment_data(self, options=None):
            segment = int((options or {}).get("segment", 1))
            potential = np.linspace(-0.5, 0.5, 21)
            if segment == 1:
                potential = potential[::-1]
            return potential, np.zeros_like(potential)

        def half_wave_potential(self, options=None):
            self.observed_options = dict(options or {})
            return {
                "E(1/2)": 0.0,
                "ΔE": 0.1,
                "peak 1": {"Ep": 0.30},
                "peak 2": {"Ep": -0.30},
            }

    cvs = [FakeCV(0.1), FakeCV(1.0), FakeCV(10.0)]

    ecat_module.trumpet_analysis(
        cvs,
        {"plot": False, "print": False, "segment": 1, "guess potential": [-0.1, 0.1]},
    )

    assert [cv.observed_options["guess potential"] for cv in cvs] == [
        [-0.1, 0.1],
        [-0.1, 0.1],
        [-0.1, 0.1],
    ]


def test_trumpet_analysis_two_cv_flat_guess_is_per_cv_scalar(ecat_module):
    class FakeCV:
        def __init__(self, scan_rate):
            self.name = f"fake-{scan_rate:g}"
            self.scan_rate = scan_rate
            self.temperature = 298
            self.observed_options = None

        def analysis_segment_data(self, options=None):
            segment = int((options or {}).get("segment", 1))
            potential = np.linspace(-0.5, 0.5, 21)
            if segment == 1:
                potential = potential[::-1]
            return potential, np.zeros_like(potential)

        def half_wave_potential(self, options=None):
            self.observed_options = dict(options or {})
            return {
                "E(1/2)": 0.0,
                "ΔE": 0.1,
                "peak 1": {"Ep": 0.30},
                "peak 2": {"Ep": -0.30},
            }

    cvs = [FakeCV(0.1), FakeCV(1.0)]

    ecat_module.trumpet_analysis(
        cvs,
        {"plot": False, "print": False, "segment": 1, "guess potential": [-0.1, -0.2]},
    )

    assert [cv.observed_options["guess potential"] for cv in cvs] == pytest.approx([-0.1, -0.2])


def test_trumpet_analysis_supports_nested_per_cv_paired_guesses(ecat_module):
    class FakeCV:
        def __init__(self, scan_rate):
            self.name = f"fake-{scan_rate:g}"
            self.scan_rate = scan_rate
            self.temperature = 298
            self.observed_options = None

        def analysis_segment_data(self, options=None):
            segment = int((options or {}).get("segment", 1))
            potential = np.linspace(-0.5, 0.5, 21)
            if segment == 1:
                potential = potential[::-1]
            return potential, np.zeros_like(potential)

        def half_wave_potential(self, options=None):
            self.observed_options = dict(options or {})
            return {
                "E(1/2)": 0.0,
                "ΔE": 0.1,
                "peak 1": {"Ep": 0.30},
                "peak 2": {"Ep": -0.30},
            }

    cvs = [FakeCV(0.1), FakeCV(1.0)]

    ecat_module.trumpet_analysis(
        cvs,
        {
            "plot": False,
            "print": False,
            "segment": 1,
            "guess potentials": [[-0.1, 0.1], [-0.2, 0.2]],
        },
    )

    assert [cv.observed_options["guess potential"] for cv in cvs] == [
        [-0.1, 0.1],
        [-0.2, 0.2],
    ]


def test_trumpet_analysis_is_public_scatterfit_wrapper(ecat_module):
    class FakeCV:
        def __init__(self, scan_rate, ep1, ep2):
            self.scan_rate = scan_rate
            self.temperature = 298
            self._ep1 = ep1
            self._ep2 = ep2

        def half_wave_potential(self, options=None):
            return {
                "E(1/2)": 0.0,
                "ΔE": self._ep2 - self._ep1,
                "peak 1": {"Ep": self._ep1},
                "peak 2": {"Ep": self._ep2},
            }

    cvs = [
        FakeCV(0.1, 0.30, -0.30),
        FakeCV(1.0, 0.20, -0.20),
        FakeCV(10.0, 0.10, -0.10),
    ]

    result = ecat_module.trumpet_analysis(cvs, {"plot": False, "print": False, "segment": 1})

    assert isinstance(result, ecat_module.ScatterFitResult)
    assert result.summary["analysis"] == "trumpet"
    assert len(result.fits) == 2


def test_trumpet_analysis_maps_mirrored_scan_orders_to_same_alpha_beta(ecat_module):
    class DirectionalCV:
        def __init__(self, scan_rate, first_branch):
            self.name = f"{first_branch}-{scan_rate:g}"
            self.scan_rate = scan_rate
            self.temperature = 298
            self.segments = 2
            self.first_branch = first_branch

        def analysis_segment_data(self, options=None):
            segment = int((options or {}).get("segment", 1))
            first_increases = self.first_branch == "anodic"
            increases = first_increases if segment == 1 else not first_increases
            potential = np.linspace(-0.5, 0.5, 51)
            if not increases:
                potential = potential[::-1]
            return potential, np.zeros_like(potential)

        def half_wave_potential(self, options=None):
            segments = list((options or {}).get("segments", [1, 2]))
            logv = np.log10(float(self.scan_rate))
            branch_ep = {
                "cathodic": -0.30 - 0.05 * logv,
                "anodic": 0.30 + 0.05 * logv,
            }
            first = self.first_branch
            second = "cathodic" if first == "anodic" else "anodic"
            return {
                "E(1/2)": 0.0,
                "ΔE": abs(branch_ep[first] - branch_ep[second]),
                "peak 1": {"segment": segments[0], "Ep": branch_ep[first]},
                "peak 2": {"segment": segments[1], "Ep": branch_ep[second]},
            }

    scan_rates = [0.1, 1.0, 10.0]
    reduction_first = [DirectionalCV(rate, "cathodic") for rate in scan_rates]
    oxidation_first = [DirectionalCV(rate, "anodic") for rate in scan_rates]
    mixed_order = [
        DirectionalCV(0.1, "cathodic"),
        DirectionalCV(1.0, "anodic"),
        DirectionalCV(10.0, "cathodic"),
    ]

    reduction_result = ecat_module.trumpet_analysis(
        reduction_first,
        {"plot": False, "print": False, "segments": [1, 2]},
    )
    oxidation_result = ecat_module.trumpet_analysis(
        oxidation_first,
        {"plot": False, "print": False, "segments": [1, 2]},
    )
    mixed_result = ecat_module.trumpet_analysis(
        mixed_order,
        {"plot": False, "print": False, "segments": [1, 2]},
    )

    def coefficient(result, symbol):
        return float(result.fit_table.loc[result.fit_table["Parameter"] == symbol, "Value"].iloc[0])

    assert coefficient(reduction_result, "α") == pytest.approx(
        coefficient(oxidation_result, "α")
    )
    assert coefficient(reduction_result, "β") == pytest.approx(
        coefficient(oxidation_result, "β")
    )
    assert coefficient(reduction_result, "α") == pytest.approx(
        coefficient(mixed_result, "α")
    )
    assert coefficient(reduction_result, "β") == pytest.approx(
        coefficient(mixed_result, "β")
    )
    assert reduction_result.fits[0][0] == pytest.approx(-0.05)
    assert oxidation_result.fits[0][0] == pytest.approx(-0.05)
    assert reduction_result.fits[1][0] == pytest.approx(0.05)
    assert oxidation_result.fits[1][0] == pytest.approx(0.05)
    assert reduction_result.summary["segment selection"]["cathodic segment"] == 1
    assert oxidation_result.summary["segment selection"]["cathodic segment"] == 2
    assert reduction_result.summary["segment selection"]["anodic segment"] == 2
    assert oxidation_result.summary["segment selection"]["anodic segment"] == 1
    assert mixed_result.summary["segment selection"]["cathodic segment"] is None
    assert mixed_result.summary["segment selection"]["anodic segment"] is None
    assert mixed_result.summary["segment selection"]["cathodic segments"] == [1, 2, 1]
    assert mixed_result.summary["segment selection"]["anodic segments"] == [2, 1, 2]


def test_trumpet_analysis_rejects_ambiguous_same_direction_segment_pair(ecat_module):
    class AmbiguousCV:
        name = "ambiguous scan"
        scan_rate = 0.1
        temperature = 298
        segments = 2

        def analysis_segment_data(self, options=None):
            potential = np.linspace(-0.5, 0.5, 51)
            return potential, np.zeros_like(potential)

        def half_wave_potential(self, options=None):
            return {
                "E(1/2)": 0.0,
                "ΔE": 0.6,
                "peak 1": {"segment": 1, "Ep": -0.3},
                "peak 2": {"segment": 2, "Ep": 0.3},
            }

    with pytest.raises(
        ValueError,
        match=r"trumpet_analysis could not assign cathodic and anodic branches.*ambiguous scan.*segments 1 and 2",
    ):
        ecat_module.trumpet_analysis(
            [AmbiguousCV(), AmbiguousCV(), AmbiguousCV()],
            {"plot": False, "print": False, "segments": [1, 2]},
        )


def test_trumpet_auto_segment_pair_uses_closest_adjacent_segment(ecat_module, cv_factory):
    class ThreeSegmentCV:
        def __init__(self, name, scan_rate):
            self.name = name
            self.scan_rate = scan_rate
            self.temperature = 298
            self.segments = 3
            self._segment_x = {
                1: np.linspace(-0.6, -0.2, 100),
                2: np.linspace(-0.2, -0.6, 100),
                3: np.linspace(-0.6, -0.2, 100),
            }

        def analysis_segment_data(self, options=None):
            segment = int((options or {}).get("segment", 1))
            x = self._segment_x[segment]
            return x, np.zeros_like(x)

        def peak_potential(self, options=None):
            segment = int((options or {}).get("segment", 1))
            if segment == 2:
                return {"Ep": -0.4, "index": 2}
            if segment == 1:
                return {"Ep": -0.42, "index": 99}
            return {"Ep": -0.38, "index": 0}

        def half_wave_potential(self, options=None):
            segments = list((options or {}).get("segments", [2, 1]))
            self.last_half_wave_segments = segments
            logv = np.log10(float(self.scan_rate))
            return {
                "E(1/2)": 0.0,
                "ΔE": 0.1,
                "peak 1": {"segment": segments[0], "Ep": 0.30 - 0.05 * logv},
                "peak 2": {"segment": segments[1], "Ep": -0.30 + 0.05 * logv},
            }

    cvs = [
        ThreeSegmentCV("cv1", 0.1),
        ThreeSegmentCV("cv2", 1.0),
        ThreeSegmentCV("cv3", 10.0),
    ]

    result = ecat_module.trumpet_analysis(
        cvs,
        {"plot": False, "print": False, "guess potential": -0.4},
    )

    selection = result.summary["segment selection"]
    assert selection["selected segment"] == 2
    assert selection["paired segment"] == 1
    assert [cv.last_half_wave_segments for cv in cvs] == [[2, 1], [2, 1], [2, 1]]


def test_trumpet_analysis_print_uses_summary_and_shared_fit_model_output(
    ecat_module,
    capsys,
    monkeypatch,
):
    class FakeCV:
        def __init__(self, scan_rate, ep1, ep2):
            self.scan_rate = scan_rate
            self.temperature = 298
            self._ep1 = ep1
            self._ep2 = ep2

        def half_wave_potential(self, options=None):
            return {
                "E(1/2)": 0.0,
                "ΔE": self._ep2 - self._ep1,
                "peak 1": {"Ep": self._ep1},
                "peak 2": {"Ep": self._ep2},
            }

    cvs = [
        FakeCV(0.1, 0.30, -0.30),
        FakeCV(1.0, 0.22, -0.22),
        FakeCV(10.0, 0.12, -0.12),
    ]

    monkeypatch.setattr(ecat_module.analysis_batch, "display", None)
    monkeypatch.setattr(ecat_module.analysis_batch, "Math", None)

    result = ecat_module.trumpet_analysis(
        cvs,
        {"plot": False, "print": True, "segment": 1, "D": 1e-5},
    )

    output = capsys.readouterr().out
    assert "Trumpet Analysis Parameters:" in output
    assert "[Trumpet Analysis Equations]" in output
    assert "Parameter" in output
    assert "Symbol" in output
    assert "Value" in output
    assert "Alpha" not in output
    assert "Beta" not in output
    assert "k_s" not in output
    assert "α" in output or "&alpha;" in output
    assert "β" in output or "&beta;" in output
    assert "k0" in output
    assert "Fit Model:" in output
    assert "Peak Potential Fit Statistics:" not in output
    assert "y = m x + b" in output
    assert "α" in result.fit_table["Parameter"].tolist()
    assert "β" in result.fit_table["Parameter"].tolist()
    assert "k0" in result.fit_table["Parameter"].tolist()
    assert output.index("Trumpet Analysis Parameters:") < output.index(
        "[Trumpet Analysis Equations]"
    )
    assert output.index("[Trumpet Analysis Equations]") < output.index(
        "Trumpet Analysis Summary:"
    )


def test_trumpet_analysis_plot_all_uses_multiplot_for_diagnostics(ecat_module, monkeypatch):
    multiplot_calls = []

    class FakeCV:
        def __init__(self, scan_rate, ep1, ep2):
            self.scan_rate = scan_rate
            self.temperature = 298
            self._ep1 = ep1
            self._ep2 = ep2

        def half_wave_potential(self, options=None):
            assert options.get("plot", False) is False
            return {
                "E(1/2)": 0.0,
                "ΔE": self._ep2 - self._ep1,
                "peak 1": {"Ep": self._ep1},
                "peak 2": {"Ep": self._ep2},
            }

    cvs = [
        FakeCV(0.1, 0.30, -0.30),
        FakeCV(1.0, 0.20, -0.20),
        FakeCV(10.0, 0.10, -0.10),
    ]

    def capture_multiplot(objects, options=None):
        multiplot_calls.append((objects, dict(options or {})))
        return []

    monkeypatch.setattr(ecat_module.analysis_batch, "multiplot", capture_multiplot)

    ecat_module.trumpet_analysis(
        cvs,
        {"plot": False, "plot all": True, "print": False, "segment": 1},
    )

    assert len(multiplot_calls) == 1
    assert multiplot_calls[0][0] == cvs


def test_trumpet_analysis_fit_indices_use_python_exclusive_stop(ecat_module):
    class FakeCV:
        def __init__(self, scan_rate, ep1, ep2):
            self.scan_rate = scan_rate
            self.temperature = 298
            self._ep1 = ep1
            self._ep2 = ep2

        def half_wave_potential(self, options=None):
            return {
                "E(1/2)": 0.0,
                "ΔE": self._ep2 - self._ep1,
                "peak 1": {"Ep": self._ep1},
                "peak 2": {"Ep": self._ep2},
            }

    cvs = [
        FakeCV(0.1, 0.30, -0.30),
        FakeCV(1.0, 0.25, -0.25),
        FakeCV(10.0, 0.10, -0.10),
    ]

    result = ecat_module.trumpet_analysis(
        cvs,
        {"plot": False, "print": False, "segment": 1, "fit indices": [0, 2]},
    )

    assert result.fits[0][0] == pytest.approx(-0.05, rel=1e-6)
    assert result.fits[1][0] == pytest.approx(0.05, rel=1e-6)


def test_trumpet_analysis_reports_untrusted_alpha_beta_region(ecat_module):
    class FakeCV:
        def __init__(self, scan_rate, ep1, ep2):
            self.scan_rate = scan_rate
            self.temperature = 298
            self._ep1 = ep1
            self._ep2 = ep2

        def half_wave_potential(self, options=None):
            return {
                "E(1/2)": 0.0,
                "ΔE": self._ep2 - self._ep1,
                "peak 1": {"Ep": self._ep1},
                "peak 2": {"Ep": self._ep2},
            }

    cvs = [
        FakeCV(0.1, 0.300, -0.300),
        FakeCV(1.0, 0.295, -0.295),
        FakeCV(10.0, 0.290, -0.290),
    ]

    result = ecat_module.trumpet_analysis(
        cvs,
        {"plot": False, "print": False, "segment": 1},
    )

    warning_row = result.fit_table[result.fit_table["Parameter"] == "Warning"]
    assert len(warning_row) == 1
    assert "may not be reliable" in warning_row.iloc[0]["Value"]


def test_trumpet_accepts_fit_color_list(ecat_module):
    class FakeCV:
        def __init__(self, scan_rate, ep1, ep2):
            self.scan_rate = scan_rate
            self.temperature = 298
            self._ep1 = ep1
            self._ep2 = ep2

        def half_wave_potential(self, options=None):
            return {
                "E(1/2)": 0.0,
                "ΔE": self._ep2 - self._ep1,
                "peak 1": {"Ep": self._ep1},
                "peak 2": {"Ep": self._ep2},
            }

    cvs = [
        FakeCV(0.1, 0.30, -0.30),
        FakeCV(1.0, 0.20, -0.20),
        FakeCV(10.0, 0.10, -0.10),
    ]

    ecat_module.trumpet_analysis(
        cvs,
        {"print": False, "segment": 1, "fit color": ["black", "tab:orange"]},
    )
    ax = plt.gca()

    assert [ecat_module.mpl.colors.to_hex(line.get_color()) for line in ax.lines] == [
        "#000000",
        ecat_module.mpl.colors.to_hex("tab:orange"),
    ]
    plt.close(ax.figure)


def test_legacy_trumpet_plot_public_name_removed(ecat_module):
    assert not hasattr(ecat_module, "trumpet_plot")


def test_nicholson_unknown_option_suggests_num_electrons(ecat_module):
    with pytest.raises(ValueError, match="num electrons"):
        ecat_module.nicholson_analysis([], {"num electons": 1, "plot": False, "print": False})


def test_nicholson_auto_segment_pair_is_reported_in_summary(ecat_module):
    class ThreeSegmentCV:
        def __init__(self, name, scan_rate):
            self.name = name
            self.scan_rate = scan_rate
            self.temperature = 298
            self.segments = 3
            self._segment_x = {
                1: np.linspace(-0.6, -0.2, 100),
                2: np.linspace(-0.6, -0.2, 100),
                3: np.linspace(-0.6, -0.2, 100),
            }

        def analysis_segment_data(self, options=None):
            segment = int((options or {}).get("segment", 1))
            x = self._segment_x[segment]
            return x, np.zeros_like(x)

        def peak_potential(self, options=None):
            segment = int((options or {}).get("segment", 1))
            if segment == 2:
                return {"Ep": -0.4, "index": 2}
            if segment == 1:
                return {"Ep": -0.42, "index": 99}
            return {"Ep": -0.38, "index": 0}

        def half_wave_potential(self, options=None):
            segments = list((options or {}).get("segments", [2, 1]))
            self.last_half_wave_segments = segments
            return {
                "E(1/2)": -0.45,
                "ΔE": 0.1,
                "peak 1": {"segment": segments[0], "Ep": -0.50},
                "peak 2": {"segment": segments[1], "Ep": -0.40},
            }

    cvs = [
        ThreeSegmentCV("cv1", 0.1),
        ThreeSegmentCV("cv2", 0.2),
        ThreeSegmentCV("cv3", 0.5),
    ]

    result = ecat_module.nicholson_analysis(
        cvs,
        {
            "plot": False,
            "print": False,
            "D": 2e-5,
            "guess potential": -0.4,
            "warn ir drop": False,
        },
    )

    selection = result["summary"]["segment selection"]
    assert selection["selected segment"] == 2
    assert selection["paired segment"] == 1
    assert [cv.last_half_wave_segments for cv in cvs] == [[2, 1], [2, 1], [2, 1]]


def test_nicholson_analysis_supports_nested_per_cv_paired_guesses(ecat_module):
    class FakeCV:
        def __init__(self, name, scan_rate):
            self.name = name
            self.scan_rate = scan_rate
            self.temperature = 298
            self.observed_options = None

        def half_wave_potential(self, options=None):
            self.observed_options = dict(options or {})
            return {
                "E(1/2)": 0.0,
                "ΔE": 0.1,
                "peak 1": {"Ep": 0.05},
                "peak 2": {"Ep": -0.05},
            }

    cvs = [FakeCV("cv1", 0.1), FakeCV("cv2", 1.0)]

    ecat_module.nicholson_analysis(
        cvs,
        {
            "plot": False,
            "print": False,
            "D": 2e-5,
            "warn ir drop": False,
            "segment": 1,
            "guess potentials": [[-0.1, 0.1], [-0.2, 0.2]],
            "exclude invalid delta ep": False,
        },
    )

    assert [cv.observed_options["guess potential"] for cv in cvs] == [
        [-0.1, 0.1],
        [-0.2, 0.2],
    ]


def test_tafel_accepts_options_and_preserves_user_color(ecat_module):
    class DummyCV:
        temperature = 298

    result = ecat_module.tafel_analysis(
        DummyCV(),
        TOF_max=10,
        thermodynamic_potential=0.2,
        redox_potential=0.0,
        options=ecat_module.TafelAnalysisOptions.from_options(
            {"overpotential range": [0, 0.5], "color": "tab:green"}
        ),
    )

    line = plt.gca().lines[0]
    assert line.get_color() == "tab:green"
    assert line.get_xdata()[0] == pytest.approx(0)
    assert line.get_xdata()[-1] == pytest.approx(0.5)
    assert result["data"]["TOFmax"].iloc[0] == pytest.approx(10)
    assert result["axes"] is plt.gca()


def test_tafel_raw_mapping_preserves_case_and_underscore_option_spellings(ecat_module):
    class DummyCV:
        temperature = 298

    result = ecat_module.tafel_analysis(
        DummyCV(),
        TOF_max=10,
        thermodynamic_potential=0.2,
        redox_potential=0.0,
        options={
            "Overpotential_Range": [0.1, 0.4],
            "Color": "tab:purple",
            "plot": True,
            "print": False,
        },
    )

    line = plt.gca().lines[0]
    assert line.get_color() == "tab:purple"
    assert line.get_xdata()[0] == pytest.approx(0.1)
    assert line.get_xdata()[-1] == pytest.approx(0.4)
    assert result["axes"] is plt.gca()


def test_tafel_accepts_multiple_cvs_and_tof_values(ecat_module):
    class DummyCV:
        def __init__(self, name, temperature=298):
            self.name = name
            self.temperature = temperature
            self.compounds = []
            self.gas = "CO2"
            self.solvent = "MeCN"

        def txt_stats(self, options=None):
            return {"Name": self.name, "Gas": self.gas, "Solvent": self.solvent}

    cvs = [DummyCV("low"), DummyCV("high")]

    result = ecat_module.tafel_analysis(
        cvs,
        TOF_max=[10, 100],
        thermodynamic_potential=-0.7,
        redox_potential=-1.47,
        options={
            "overpotential range": [0, 0.4],
            "labels": ["10 mM PhOH", "100 mM PhOH"],
            "legend": True,
        },
    )

    ax = result["axes"]
    assert len(ax.lines) == 2
    assert [line.get_label() for line in ax.lines] == ["10 mM PhOH", "100 mM PhOH"]
    assert result["summary"]["TOFmax"].tolist() == pytest.approx([10, 100])
    assert set(result["data"]["Label"]) == {"10 mM PhOH", "100 mM PhOH"}
    plt.close(ax.figure)


def test_tafel_pretty_output_uses_analysis_report_sections(ecat_module, capsys):
    class DummyCV:
        name = "cat"
        temperature = 298
        compounds = []
        gas = "CO2"
        solvent = "MeCN"

        def txt_stats(self, options=None):
            return {"Name": self.name, "Gas": self.gas, "Solvent": self.solvent}

    ecat_module.tafel_analysis(
        DummyCV(),
        TOF_max=10,
        thermodynamic_potential=-0.7,
        redox_potential=-1.47,
        options={"plot": False, "print": True, "pretty print": False},
    )

    printed = capsys.readouterr().out
    assert "Tafel Analysis Parameters:" in printed
    assert "Tafel Analysis Equations:" in printed
    assert "Tafel Analysis Summary:" in printed
    assert "Parameter" in printed
    assert "Symbol" in printed
    assert "Metric" in printed
    assert "TOFmax" in printed
    assert "exp" in printed
    assert printed.index("Tafel Analysis Parameters:") < printed.index(
        "Tafel Analysis Equations:"
    )
    assert printed.index("Tafel Analysis Equations:") < printed.index(
        "Tafel Analysis Summary:"
    )


def test_tafel_uses_shared_multiplot_style_helpers(ecat_module, monkeypatch):
    calls = {"prepare": 0, "finish": 0}

    class DummyCV:
        temperature = 298
        name = "cv"
        compounds = []

    def fake_prepare(echem_list, options):
        calls["prepare"] += 1
        fig, ax = plt.subplots()
        return {
            "ax": ax,
            "labels": ["a", "b"],
            "display labels": ["a", "b"],
            "title": "title",
            "subtitle": "",
            "title fontsize": 12,
            "subtitle fontsize": 10,
            "color spec": {
                "line colors": ["tab:blue", "tab:orange"],
                "labels": ["a", "b"],
                "gradient groups": [],
                "discrete indices": [0, 1],
            },
        }

    def fake_finish(echem_list, options, style):
        calls["finish"] += 1

    monkeypatch.setattr(ecat_module, "_prepare_multiplot_style", fake_prepare)
    monkeypatch.setattr(ecat_module, "_finish_multiplot_style", fake_finish)

    result = ecat_module.tafel_analysis(
        [DummyCV(), DummyCV()],
        TOF_max=[1, 2],
        thermodynamic_potential=-0.7,
        redox_potential=-1.47,
        options={"legend": True},
    )

    assert calls == {"prepare": 1, "finish": 1}
    assert [line.get_color() for line in result["axes"].lines] == ["tab:blue", "tab:orange"]
    plt.close(result["axes"].figure)


def test_fit_rate_returns_novice_friendly_result_object(ecat_module):
    df = pd.DataFrame(
        {
            "Scan Rate": [1.0, 2.0, 3.0],
            "kobs": [2.0, 4.0, 6.0],
        }
    )

    result = ecat_module.fit_rate(
        df,
        {"plot": False, "print": False, "metric": "kobs"},
    )

    assert result.table["y transformed"].tolist() == pytest.approx([2.0, 4.0, 6.0])
    assert result.fit_table.loc[0, "slope"] == pytest.approx(2.0)
    assert result.summary["analysis"] == "rate fit"

    assert result.fits["parameters"]["m"] == pytest.approx(2.0)


def test_fit_rate_aligns_hidden_full_results_to_visible_slice(ecat_module):
    full_results = pd.DataFrame(
        {
            "Scan Rate": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
            "kobs": [2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0],
            "Warning Details": [""] * 7,
        }
    )
    display_df = pd.DataFrame(
        {
            "Plot Label": [f"Trace {idx}" for idx in range(7)],
        }
    )
    display_df.attrs["full_results_df"] = full_results

    result = ecat_module.fit_rate(
        display_df.iloc[-5:],
        {"plot": False, "print": False, "metric": "kobs"},
    )

    assert result.table["Scan Rate"].tolist() == pytest.approx([3.0, 4.0, 5.0, 6.0, 7.0])
    assert result.table["kobs"].tolist() == pytest.approx([6.0, 8.0, 10.0, 12.0, 14.0])
    assert result.fit_table.loc[0, "Fit Points"] == 5
    assert result.fit_table.loc[0, "fit x min"] == pytest.approx(3.0)
    assert result.fit_table.loc[0, "fit x max"] == pytest.approx(7.0)


def test_fit_rate_result_includes_fit_statistics_and_prints_summary(
    ecat_module,
    capsys,
):
    df = pd.DataFrame(
        {
            "Scan Rate": [1.0, 2.0, 3.0],
            "kobs": [2.0, 4.0, 6.0],
        }
    )

    result = ecat_module.fit_rate(
        df,
        {"plot": False, "print": True, "metric": "kobs"},
    )

    printed = capsys.readouterr().out

    assert "Fit Model:" in printed
    assert "Fit Statistics:" not in printed
    assert "Field" in printed
    assert "Value" in printed
    assert "Equation" in printed
    assert "R²" in printed
    assert "RMSE" in printed
    assert "Fit Points" in printed
    assert "m" in printed
    assert "b" in printed
    assert result.fit_table.loc[0, "R2"] == pytest.approx(1.0)
    assert result.fit_table.loc[0, "RMSE"] == pytest.approx(0.0)
    assert result.fit_table.loc[0, "Fit Points"] == 3


def test_fit_rate_loglog_default_prints_linear_model_summary(
    ecat_module,
    capsys,
):
    df = pd.DataFrame(
        {
            "Substrate Concentration (M)": [0.1, 0.2, 0.4, 0.8, 1.6, 2.8],
            "kobs": 2.0 * np.asarray([0.1, 0.2, 0.4, 0.8, 1.6, 2.8]) ** 1.5,
        }
    )

    ecat_module.fit_rate(
        df,
        {
            "plot": False,
            "print": True,
            "x column": "Substrate Concentration (M)",
            "metric": "kobs",
            "transform mode": "log-log",
        },
    )

    printed = capsys.readouterr().out
    assert "Fit Model:" in printed
    assert "Fit Statistics:" not in printed
    assert "Model" in printed
    assert "linear" in printed
    assert "Equation" in printed
    assert "y = m x + b" in printed
    assert "y = A x^n" not in printed
    assert "±" in printed
    assert "X Range" in printed
    assert "-1.000 to 0.4472" in printed
    assert "m" in printed
    assert "b" in printed


def test_fit_rate_loglog_explicit_linear_model_prints_linear_summary(
    ecat_module,
    capsys,
):
    df = pd.DataFrame(
        {
            "Substrate Concentration (M)": [0.1, 0.2, 0.4, 0.8, 1.6, 2.8],
            "kobs": 2.0 * np.asarray([0.1, 0.2, 0.4, 0.8, 1.6, 2.8]) ** 1.5,
        }
    )

    result = ecat_module.fit_rate(
        df,
        {
            "plot": False,
            "print": True,
            "x column": "Substrate Concentration (M)",
            "metric": "kobs",
            "transform mode": "log-log",
            "fit model": "linear",
        },
    )

    printed = capsys.readouterr().out
    assert "Fit Model:" in printed
    assert "linear" in printed
    assert "y = m x + b" in printed
    assert "y = A x^n" not in printed
    assert result.fit_table["model"].unique().tolist() == ["linear"]


def test_fit_rate_loglog_fit_label_uses_selected_linear_model_and_sig_figs(ecat_module):
    x = np.asarray([0.1, 0.2, 0.4, 0.8, 1.6, 2.8])
    df = pd.DataFrame(
        {
            "Substrate Concentration (M)": x,
            "kobs": 2.12345 * x ** 0.54321,
        }
    )

    ecat_module.fit_rate(
        df,
        {
            "plot": True,
            "print": False,
            "x column": "Substrate Concentration (M)",
            "metric": "kobs",
            "transform mode": "log-log",
            "fit label": True,
            "sig figs": 3,
        },
    )

    labels = [text.get_text() for text in plt.gca().get_legend().get_texts()]
    assert any("y = 0.543x + 0.327" in label for label in labels)


def test_fit_rate_fit_label_uses_sig_figs_for_r2(ecat_module):
    df = pd.DataFrame(
        {
            "Substrate Concentration (M)": [0.18, 0.35, 0.7, 1.4, 2.8],
            "kobs": [1.0, 2.0, 3.5, 8.0, 15.0],
        }
    )

    ecat_module.fit_rate(
        df,
        {
            "plot": True,
            "print": False,
            "x column": "Substrate Concentration (M)",
            "metric": "kobs",
            "fit model": "linear",
            "fit label": True,
            "sig figs": 5,
        },
    )

    labels = [text.get_text() for text in plt.gca().get_legend().get_texts()]
    assert any("R^2 = 0.99779" in label for label in labels)


def test_fit_rate_log_residual_fit_label_uses_ln_r2(ecat_module):
    x = np.asarray([1.0, 2.0, 3.0, 4.0])
    df = pd.DataFrame({"x": x, "kobs": np.exp(0.5 + 0.4 * x)})

    ecat_module.fit_rate(
        df,
        {
            "plot": True,
            "print": False,
            "x column": "x",
            "metric": "kobs",
            "fit model": "exponential",
            "fit residual": "log",
            "fit label": True,
        },
    )

    labels = [text.get_text() for text in plt.gca().get_legend().get_texts()]
    assert any(r"\ln R^2 =" in label for label in labels)
    assert not any("\n$R^2 =" in label for label in labels)


def test_fit_rate_log10_residual_fit_label_uses_log_r2(ecat_module):
    x = np.asarray([1.0, 2.0, 3.0, 4.0])
    df = pd.DataFrame({"x": x, "kobs": 10 ** (0.5 + 0.4 * x)})

    ecat_module.fit_rate(
        df,
        {
            "plot": True,
            "print": False,
            "x column": "x",
            "metric": "kobs",
            "fit model": "exponential",
            "fit residual": "log10",
            "fit label": True,
        },
    )

    labels = [text.get_text() for text in plt.gca().get_legend().get_texts()]
    assert any(r"\log R^2 =" in label for label in labels)
    assert not any("\n$R^2 =" in label for label in labels)


def test_fit_rate_fit_label_uses_display_friendly_default_sig_figs(ecat_module):
    x = np.asarray([0.18, 0.35, 0.7, 1.4, 2.8])
    df = pd.DataFrame(
        {
            "Substrate Concentration (M)": x,
            "kobs": 3.14159265 * x + 2.718281828,
        }
    )

    ecat_module.fit_rate(
        df,
        {
            "plot": True,
            "print": False,
            "x column": "Substrate Concentration (M)",
            "metric": "kobs",
            "fit model": "linear",
            "fit label": True,
        },
    )

    labels = [text.get_text() for text in plt.gca().get_legend().get_texts()]
    assert any("y = 3.142x + 2.718" in label for label in labels)
    assert not any("3.14159" in label or "2.71828" in label for label in labels)


def test_fit_rate_fit_label_formats_scientific_notation_as_mathtext(ecat_module):
    x = np.asarray([0.0, 1.0, 2.0, 3.0])
    df = pd.DataFrame(
        {
            "x": x,
            "kobs": 3.2e5 * x + 1.1e-5,
        }
    )

    ecat_module.fit_rate(
        df,
        {
            "plot": True,
            "print": False,
            "x column": "x",
            "metric": "kobs",
            "fit model": "linear",
            "fit label": True,
            "sig figs": 3,
        },
    )

    labels = [text.get_text() for text in plt.gca().get_legend().get_texts()]
    assert any(r"3.20\times 10^{5}" in label for label in labels)
    assert any(r"1.10\times 10^{-5}" in label for label in labels)
    assert not any("e+05" in label or "e-05" in label for label in labels)


def test_fit_rate_accepts_sig_fig_alias_for_fit_label(ecat_module):
    x = np.asarray([0.18, 0.35, 0.7, 1.4, 2.8])
    df = pd.DataFrame(
        {
            "Substrate Concentration (M)": x,
            "kobs": 3.14159265 * x + 2.718281828,
        }
    )

    ecat_module.fit_rate(
        df,
        {
            "plot": True,
            "print": False,
            "x column": "Substrate Concentration (M)",
            "metric": "kobs",
            "fit model": "linear",
            "fit label": True,
            "significant figures": 3,
        },
    )

    labels = [text.get_text() for text in plt.gca().get_legend().get_texts()]
    assert any("y = 3.14x + 2.72" in label for label in labels)


def test_fit_rate_named_fit_indices_produce_multiple_loglog_fits(ecat_module):
    df = pd.DataFrame(
        {
            "Scan Rate": np.arange(1, 13, dtype=float),
            "kobs": np.arange(1, 13, dtype=float) ** 2,
        }
    )

    result = ecat_module.fit_rate(
        df,
        {
            "plot": False,
            "print": False,
            "transform mode": "log-log",
            "fit indices": {
                "early": [0, 5],
                "tail split": [[6, 9], [9, None]],
            },
        },
    )

    assert list(result.fits) == ["early", "tail split"]
    assert result.fit_table["series"].drop_duplicates().tolist() == ["early", "tail split"]
    assert result.fit_table.drop_duplicates("series")["slope"].tolist() == pytest.approx([2.0, 2.0])
    assert result.fit_table.drop_duplicates("series")["Fit Points"].tolist() == [5, 6]


def test_fit_rate_named_fit_indices_produce_multiple_fits_with_open_ended_windows(ecat_module):
    df = pd.DataFrame(
        {
            "Scan Rate": np.arange(1, 9, dtype=float),
            "kobs": 3.0 * np.arange(1, 9, dtype=float) + 2.0,
        }
    )

    result = ecat_module.fit_rate(
        df,
        {
            "plot": False,
            "print": False,
            "metric": "kobs",
            "fit indices": {
                "early": [0, 3],
                "late": [4, None],
            },
        },
    )

    assert list(result.fits) == ["early", "late"]
    assert result.fit_table["series"].drop_duplicates().tolist() == ["early", "late"]
    assert result.fit_table.drop_duplicates("series")["slope"].tolist() == pytest.approx([3.0, 3.0])
    assert result.fit_table.drop_duplicates("series")["Fit Points"].tolist() == [3, 4]


def test_fit_rate_named_fit_indices_support_model_fits_and_plot_lines(ecat_module):
    df = pd.DataFrame(
        {
            "Scan Rate": np.arange(1, 9, dtype=float),
            "kobs": np.arange(1, 9, dtype=float) ** 2,
        }
    )

    result = ecat_module.fit_rate(
        df,
        {
            "plot": True,
            "print": False,
            "fit model": "power",
            "fit indices": {
                "early": [0, 4],
                "late": [4, None],
            },
        },
    )

    ax = plt.gca()
    assert list(result.fits) == ["early", "late"]
    assert result.fit_table["series"].drop_duplicates().tolist() == ["early", "late"]
    assert result.fit_table["model"].unique().tolist() == ["power"]
    assert [result.fits[key]["parameters"]["n"] for key in ["early", "late"]] == pytest.approx([2.0, 2.0])
    assert len(ax.lines) == 2


def test_fit_rate_named_fit_indices_accept_fit_color_list(ecat_module):
    df = pd.DataFrame(
        {
            "Scan Rate": np.arange(1, 9, dtype=float),
            "kobs": np.arange(1, 9, dtype=float) ** 2,
        }
    )

    ecat_module.fit_rate(
        df,
        {
            "plot": True,
            "print": False,
            "fit model": "power",
            "fit indices": {
                "early": [0, 4],
                "late": [4, None],
            },
            "fit color": ["black", "tab:orange"],
        },
    )

    ax = plt.gca()
    assert [ecat_module.mpl.colors.to_hex(line.get_color()) for line in ax.lines] == [
        "#000000",
        ecat_module.mpl.colors.to_hex("tab:orange"),
    ]
    plt.close(ax.figure)


def test_fit_rate_unnamed_fit_indices_get_generated_labels_and_plot_lines(ecat_module):
    df = pd.DataFrame(
        {
            "Scan Rate": np.arange(1, 7, dtype=float),
            "kobs": np.arange(1, 7, dtype=float) * 3,
        }
    )

    result = ecat_module.fit_rate(
        df,
        {
            "plot": True,
            "print": False,
            "fit indices": [[0, 3], [3, None]],
        },
    )

    ax = plt.gca()
    assert list(result.fits) == ["Fit 1", "Fit 2"]
    assert result.fit_table["series"].drop_duplicates().tolist() == ["Fit 1", "Fit 2"]
    assert len(ax.lines) == 2
    assert [line.get_label() for line in ax.lines] == ["Fit 1", "Fit 2"]
    plt.close("all")


def test_fit_rate_displays_fit_statistics_dataframe(
    ecat_module,
    monkeypatch,
    capsys,
):
    displayed = {}

    def capture_display(table):
        displayed["object"] = table

    monkeypatch.setattr(ecat_module, "display", capture_display)

    df = pd.DataFrame(
        {
            "Scan Rate": [1.0, 2.0, 3.0],
            "kobs": [2.0, 4.0, 6.0],
        }
    )

    result = ecat_module.fit_rate(
        df,
        {"plot": False, "print": True, "metric": "kobs"},
    )

    printed = capsys.readouterr().out

    displayed_table = getattr(displayed["object"], "data", displayed["object"])

    assert "Fit Model:" not in printed
    assert "Fit Statistics:" not in printed
    assert getattr(displayed["object"], "caption", None) == "Fit Model"
    assert list(displayed_table.columns) == ["Field", "Value"]
    assert displayed_table.set_index("Field").loc["Model", "Value"] == "linear"
    assert "R²" in displayed_table["Field"].tolist()
    assert "R2" in result.fit_table.columns


def test_fit_peak_current_result_includes_fit_statistics(
    ecat_module,
    cv_factory,
):
    cvs = [
        cv_factory(name="50mVs_sample_CO2_MeCN_10mM_Fc_run01"),
        cv_factory(name="100mVs_sample_CO2_MeCN_10mM_Fc_run02"),
    ]

    result = ecat_module.fit_peak_current(
        cvs,
        {
            "plot": False,
            "print": False,
            "exact potential": 0.2,
            "noise window": 5,
            "noise polyorder": 2,
            "peak prominence": 1e-7,
            "tangent range": [0.05, 0.3],
            "percent threshold": 100,
        },
    )

    assert result.fit_table.loc[0, "series"] == "ip"
    assert "R2" in result.fit_table.columns
    assert "RMSE" in result.fit_table.columns
    assert result.fit_table.loc[0, "Fit Points"] == 2


def test_fit_peak_current_applies_y_mode_to_peak_series(
    ecat_module,
    cv_factory,
):
    base_current = np.asarray(
        [
            -0.1e-6, -0.08e-6, -0.05e-6, 0.0, 0.3e-6, 0.8e-6, 1.8e-6,
            4.0e-6, 7.2e-6, 6.5e-6, 5.5e-6, 4.2e-6, 3.0e-6, 1.8e-6,
            1.0e-6, 0.4e-6, 0.1e-6, 0.0, -0.03e-6, -0.05e-6, -0.07e-6,
        ]
    )
    cvs = [
        cv_factory(name="50mVs_sample_CO2_MeCN_10mM_Fc_run01", current=base_current),
        cv_factory(name="100mVs_sample_CO2_MeCN_10mM_Fc_run02", current=2 * base_current),
        cv_factory(name="200mVs_sample_CO2_MeCN_10mM_Fc_run03", current=4 * base_current),
    ]

    result = ecat_module.fit_peak_current(
        cvs,
        {
            "plot": False,
            "print": False,
            "exact potential": 0.2,
            "noise window": 5,
            "noise polyorder": 2,
            "peak prominence": 1e-7,
            "tangent range": [0.05, 0.3],
            "percent threshold": 100,
            "y mode": "ratio",
        },
    )

    assert result.table["y adjusted"].tolist() == pytest.approx([1.0, 2.0, 4.0])
    assert result.table["y transformed"].tolist() == pytest.approx([1.0, 2.0, 4.0])
    assert result.table["y mode"].unique().tolist() == ["ratio"]


def test_fit_peak_current_y_mode_label_is_adjusted_before_transform_wrapping(
    ecat_module,
    cv_factory,
):
    base_current = np.asarray(
        [
            -0.1e-6, -0.08e-6, -0.05e-6, 0.0, 0.3e-6, 0.8e-6, 1.8e-6,
            4.0e-6, 7.2e-6, 6.5e-6, 5.5e-6, 4.2e-6, 3.0e-6, 1.8e-6,
            1.0e-6, 0.4e-6, 0.1e-6, 0.0, -0.03e-6, -0.05e-6, -0.07e-6,
        ]
    )
    cvs = [
        cv_factory(name="50mVs_sample_CO2_MeCN_10mM_Fc_run01", current=base_current),
        cv_factory(name="100mVs_sample_CO2_MeCN_10mM_Fc_run02", current=2 * base_current),
        cv_factory(name="200mVs_sample_CO2_MeCN_10mM_Fc_run03", current=4 * base_current),
    ]

    ecat_module.fit_peak_current(
        cvs,
        {
            "plot": True,
            "print": False,
            "exact potential": 0.2,
            "noise window": 5,
            "noise polyorder": 2,
            "peak prominence": 1e-7,
            "tangent range": [0.05, 0.3],
            "percent threshold": 100,
            "y mode": "ratio",
            "y transform": "log10",
        },
    )

    label = plt.gca().get_ylabel()
    assert "log" in label
    assert "/" in label
    assert "Peak" in label
    plt.close()


def test_fit_peak_potential_applies_y_mode_to_ep_series(ecat_module, cv_factory):
    cvs = [
        cv_factory(name="50mVs_sample_CO2_MeCN_10mM_Fc_run01"),
        cv_factory(name="100mVs_sample_CO2_MeCN_10mM_Fc_run02"),
        cv_factory(name="200mVs_sample_CO2_MeCN_10mM_Fc_run03"),
    ]

    result = ecat_module.fit_peak_potential(
        cvs,
        {
            "plot": False,
            "print": False,
            "exact potential": 0.2,
            "y mode": "delta",
            "y0": 0.1,
        },
    )

    assert result.table["y adjusted"].tolist() == pytest.approx([0.1, 0.1, 0.1])
    assert result.table["y transformed"].tolist() == pytest.approx([0.1, 0.1, 0.1])
    assert result.table["y0"].unique().tolist() == pytest.approx([0.1])


def test_fit_peak_potential_y_mode_label_shows_adjusted_potential(ecat_module, cv_factory):
    cvs = [
        cv_factory(name="50mVs_sample_CO2_MeCN_10mM_Fc_run01"),
        cv_factory(name="100mVs_sample_CO2_MeCN_10mM_Fc_run02"),
        cv_factory(name="200mVs_sample_CO2_MeCN_10mM_Fc_run03"),
    ]

    ecat_module.fit_peak_potential(
        cvs,
        {
            "plot": True,
            "print": False,
            "exact potential": 0.2,
            "y mode": "delta",
        },
    )

    label = plt.gca().get_ylabel()
    assert "Peak" in label
    assert " - " in label
    plt.close()


def test_sevcik_analysis_is_canonical_and_legacy_wrapper_removed(ecat_module):
    assert hasattr(ecat_module, "sevcik_analysis")
    assert not hasattr(ecat_module, "Sevcik")


def test_multi_scatterplot_plots_labeled_dataframes_with_explicit_columns(ecat_module):
    datasets = {
        "Fe only": pd.DataFrame({"x": [1.0, 2.0, 3.0], "kobs": [2.0, 4.0, 6.0]}),
        "cyclen": pd.DataFrame({"x": [1.0, 2.0, 3.0], "kobs": [3.0, 6.0, 9.0]}),
    }

    result = ecat_module.multi_scatterplot(
        datasets,
        {
            "x column": "x",
            "y column": "kobs",
            "legend": False,
            "title": False,
        },
    )

    ax = result.axes
    assert result.figure is ax.figure
    assert len(ax.collections) == 2
    assert [collection.get_label() for collection in ax.collections] == ["Fe only", "cyclen"]
    assert ax.get_xlabel() == "x"
    assert ax.get_ylabel() == ecat_module._format_fit_rate_metric_label("kobs")
    assert result.table["series"].tolist() == ["Fe only"] * 3 + ["cyclen"] * 3


def test_multi_scatterplot_uses_scatterfit_raw_table_as_default_source(ecat_module):
    result = ecat_module.ScatterFitResult(
        table=pd.DataFrame({"x raw": [999.0], "kobs": [999.0]}),
        raw_table=pd.DataFrame({"concentration": [1.0, 2.0], "kobs": [10.0, 20.0]}),
        transformed_table=pd.DataFrame({"x transformed": [100.0], "y transformed": [200.0]}),
        fit_table=None,
        fit_model_results=None,
    )

    plotted = ecat_module.multi_scatterplot(
        {"sample": result},
        {
            "x column": "concentration",
            "y column": "kobs",
            "fit": False,
            "legend": False,
            "title": False,
        },
    )

    assert plotted.raw_table["x raw"].tolist() == [1.0, 2.0]
    assert plotted.raw_table["y raw"].tolist() == [10.0, 20.0]
    assert plotted.table["x"].tolist() == [1.0, 2.0]
    assert plotted.table["y"].tolist() == [10.0, 20.0]
    plt.close(plotted.figure)


def test_scatter_fit_result_is_not_tuple_iterable(ecat_module):
    result = ecat_module.ScatterFitResult(table=pd.DataFrame({"x": [1.0]}), fits={})

    with pytest.raises(TypeError):
        iter(result)


def test_multi_scatterplot_accepts_dataclass_options(ecat_module):
    datasets = {
        "Fe only": pd.DataFrame({"x": [1.0, 2.0, 3.0], "kobs": [2.0, 4.0, 6.0]}),
        "cyclen": pd.DataFrame({"x": [1.0, 2.0, 3.0], "kobs": [3.0, 6.0, 9.0]}),
    }
    options = ecat_module.MultiScatterplotOptions.from_options(
        {
            "x column": "x",
            "y column": "kobs",
            "legend": False,
            "title": False,
        }
    )

    result = ecat_module.multi_scatterplot(datasets, options)

    assert len(result.axes.collections) == 2
    assert result.table["series"].tolist() == ["Fe only"] * 3 + ["cyclen"] * 3


def test_multi_scatterplot_uses_matplotlib_xscale_yscale_options(ecat_module):
    datasets = {
        "Fe only": pd.DataFrame({"x": [1.0, 10.0, 100.0], "kobs": [2.0, 20.0, 200.0]}),
    }

    result = ecat_module.multi_scatterplot(
        datasets,
        {
            "x column": "x",
            "y column": "kobs",
            "xscale": "log",
            "yscale": "log",
            "legend": False,
            "title": False,
        },
    )

    assert result.axes.get_xscale() == "log"
    assert result.axes.get_yscale() == "log"
    assert result.axes.get_xlabel() == "x"
    assert result.axes.get_ylabel() == ecat_module._format_fit_rate_metric_label("kobs")
    assert result.table["x"].tolist() == pytest.approx([1.0, 10.0, 100.0])
    assert result.table["y"].tolist() == pytest.approx([2.0, 20.0, 200.0])
    plt.close(result.figure)


def test_multi_scatterplot_plot_scale_log_log_sets_both_axes(ecat_module):
    datasets = {
        "Fe only": pd.DataFrame({"x": [1.0, 10.0, 100.0], "kobs": [2.0, 20.0, 200.0]}),
    }

    result = ecat_module.multi_scatterplot(
        datasets,
        {
            "x column": "x",
            "y column": "kobs",
            "plot scale": "log-log",
            "legend": False,
            "title": False,
        },
    )

    assert result.axes.get_xscale() == "log"
    assert result.axes.get_yscale() == "log"
    plt.close(result.figure)


def test_multi_scatterplot_accepts_scatterfit_modes_for_loglog_kobs(ecat_module):
    datasets = {
        "sample": pd.DataFrame({"x raw": [1.0, 10.0, 100.0], "kobs": [2.0, 20.0, 200.0]}),
    }

    result = ecat_module.multi_scatterplot(
        datasets,
        {
            "x column": "x raw",
            "y column": "kobs",
            "y mode": "raw",
            "transform mode": "log-log",
            "fit": True,
            "legend": False,
            "title": False,
        },
    )

    assert result.table["x"].tolist() == pytest.approx([0.0, 1.0, 2.0])
    assert result.table["y"].tolist() == pytest.approx(np.log10([2.0, 20.0, 200.0]))
    assert result.fit_table.loc[0, "slope"] == pytest.approx(1.0)
    plt.close(result.figure)


def test_multi_scatterplot_scatterfit_modes_use_scatterfit_axis_labels(ecat_module):
    df = pd.DataFrame({"Scan Rate": [1.0, 2.0, 3.0], "kobs": [2.0, 4.0, 8.0]})
    rate_result = ecat_module.fit_rate(
        df,
        {
            "plot": False,
            "print": False,
            "metric": "kobs",
            "y mode": "enhancement",
            "transform mode": "log-log",
            "fit": False,
        },
    )

    result = ecat_module.multi_scatterplot(
        {"rate": rate_result},
        {
            "x column": "x raw",
            "y column": "kobs",
            "y mode": "enhancement",
            "transform mode": "log-log",
            "fit": False,
            "legend": False,
            "title": False,
        },
    )

    assert result.axes.get_xlabel() == r"$\log_{10}$(Scan Rate / V/s)"
    assert result.axes.get_ylabel() == (
        r"$\log_{10}$($k_{\mathrm{obs}}$/$k_{\mathrm{obs}}^{0}$ - 1)"
    )
    plt.close(result.figure)


def test_multi_scatterplot_print_defaults_to_true(ecat_module):
    assert ecat_module.MultiScatterplotOptions.from_options({}).print is True


def test_multi_scatterplot_rejects_unknown_option_with_suggestion(ecat_module):
    datasets = {"trace": pd.DataFrame({"x": [1.0, 2.0], "kobs": [2.0, 4.0]})}

    with pytest.raises(ValueError, match="plot style"):
        ecat_module.multi_scatterplot(datasets, {"plot stile": "line"})


def test_multi_scatterplot_reuses_fits_from_scatter_result(ecat_module):
    df = pd.DataFrame({"Scan Rate": [1.0, 2.0, 3.0], "kobs": [2.0, 4.0, 6.0]})
    rate_result = ecat_module.fit_rate(
        df,
        {"plot": False, "print": False, "metric": "kobs"},
    )

    result = ecat_module.multi_scatterplot(
        {"rate": rate_result},
        {"legend": False, "title": False},
    )

    ax = result.axes
    assert len(ax.collections) == 1
    assert len(ax.lines) == 1
    np.testing.assert_allclose(ax.lines[0].get_ydata(), 2 * ax.lines[0].get_xdata())
    assert result.fit_table.loc[0, "slope"] == pytest.approx(2.0)


def test_multi_scatterplot_fit_line_range_extends_overlay(ecat_module):
    df = pd.DataFrame({"Scan Rate": [1.0, 2.0, 3.0], "kobs": [2.0, 4.0, 6.0]})
    rate_result = ecat_module.fit_rate(
        df,
        {"plot": False, "print": False, "metric": "kobs"},
    )

    result = ecat_module.multi_scatterplot(
        {"rate": rate_result},
        {"legend": False, "title": False, "fit line range": [0.0, 5.0]},
    )

    line = result.axes.lines[0]
    np.testing.assert_allclose([line.get_xdata()[0], line.get_xdata()[-1]], [0.0, 5.0])
    np.testing.assert_allclose([line.get_ydata()[0], line.get_ydata()[-1]], [0.0, 10.0], atol=1e-12)
    assert result.fit_table.loc[0, "fit x min"] == pytest.approx(1.0)
    assert result.fit_table.loc[0, "fit x max"] == pytest.approx(3.0)


def test_multi_scatterplot_result_input_confidence_band(ecat_module):
    df = pd.DataFrame(
        {
            "Scan Rate": [1.0, 2.0, 3.0, 4.0, 5.0],
            "kobs": [2.2, 3.8, 6.1, 8.3, 9.7],
        }
    )
    rate_result = ecat_module.fit_rate(
        df,
        {"plot": False, "print": False, "metric": "kobs"},
    )

    result = ecat_module.multi_scatterplot(
        {"rate": rate_result},
        {
            "legend": False,
            "title": False,
            "fit band": "confidence",
            "fit line range": [0.0, 6.0],
        },
    )

    bands = _fit_band_collections(result.axes)
    assert len(bands) == 1
    vertices = bands[0].get_paths()[0].vertices
    assert np.nanmin(vertices[:, 0]) == pytest.approx(0.0)
    assert np.nanmax(vertices[:, 0]) == pytest.approx(6.0)
    assert _band_width_near_x(bands[0], 3.0) > 0


def test_multi_scatterplot_uses_multiplot_colors_for_points_and_fits(ecat_module):
    first = ecat_module.fit_rate(
        pd.DataFrame({"Scan Rate": [1.0, 2.0, 3.0], "kobs": [2.0, 4.0, 6.0]}),
        {"plot": False, "print": False, "metric": "kobs"},
    )
    second = ecat_module.fit_rate(
        pd.DataFrame({"Scan Rate": [1.0, 2.0, 3.0], "kobs": [3.0, 6.0, 9.0]}),
        {"plot": False, "print": False, "metric": "kobs"},
    )

    result = ecat_module.multi_scatterplot(
        {"Fe only": first, "cyclen": second},
        {"legend": False, "title": False},
    )

    expected = [ecat_module.mpl.colors.to_hex(color) for color in ["black", "tab:blue"]]
    point_colors = [
        ecat_module.mpl.colors.to_hex(collection.get_facecolors()[0])
        for collection in result.axes.collections
    ]
    fit_colors = [
        ecat_module.mpl.colors.to_hex(line.get_color())
        for line in result.axes.lines
    ]

    assert point_colors == expected
    assert fit_colors == expected
    plt.close(result.figure)


def test_multi_scatterplot_print_true_displays_reused_fit_table(
    ecat_module,
    monkeypatch,
    capsys,
):
    displayed = {}

    def capture_display(table):
        displayed["object"] = table

    monkeypatch.setattr(ecat_module, "display", capture_display)

    df = pd.DataFrame({"Scan Rate": [1.0, 2.0, 3.0], "kobs": [2.0, 4.0, 6.0]})
    rate_result = ecat_module.fit_rate(
        df,
        {"plot": False, "print": False, "metric": "kobs"},
    )

    result = ecat_module.multi_scatterplot(
        {"rate": rate_result},
        {"print": True, "legend": False, "title": False},
    )

    printed = capsys.readouterr().out

    displayed_table = getattr(displayed["object"], "data", displayed["object"])

    assert "Multi Scatterplot Fit Statistics:" not in printed
    assert getattr(displayed["object"], "caption", None) == "Multi Scatterplot Fit Statistics"
    assert displayed_table.equals(result.fit_table)
    assert displayed_table.loc[0, "slope"] == pytest.approx(2.0)


def test_multi_scatterplot_fits_from_scatter_result_use_plotted_points(ecat_module):
    df = pd.DataFrame({"Scan Rate": [1.0, 2.0, 3.0, 4.0], "kobs": [2.0, 4.0, 20.0, 30.0]})
    rate_result = ecat_module.fit_rate(
        df,
        {
            "plot": False,
            "print": False,
            "metric": "kobs",
            "fit indices": [0, 2],
        },
    )

    result = ecat_module.multi_scatterplot(
        {"rate": rate_result},
        {"legend": False, "title": False},
    )

    line_x = result.axes.lines[0].get_xdata()
    assert np.nanmin(line_x) == pytest.approx(1.0)
    assert np.nanmax(line_x) == pytest.approx(4.0)
    assert result.fit_table.loc[0, "slope"] == pytest.approx(10.0)
    plt.close(result.figure)


def test_multi_scatterplot_can_disable_reused_fit_plotting(ecat_module):
    df = pd.DataFrame({"Scan Rate": [1.0, 2.0, 3.0], "kobs": [2.0, 4.0, 6.0]})
    rate_result = ecat_module.fit_rate(
        df,
        {"plot": False, "print": False, "metric": "kobs"},
    )

    result = ecat_module.multi_scatterplot(
        {"rate": rate_result},
        {"plot fit": False, "legend": False, "title": False},
    )

    assert len(result.axes.collections) == 1
    assert len(result.axes.lines) == 0
    assert result.fit_table.loc[0, "slope"] == pytest.approx(2.0)
    plt.close(result.figure)


def test_multi_scatterplot_auto_resolves_transformed_columns(ecat_module):
    df = pd.DataFrame(
        {
            "x raw": [1.0, 2.0, 3.0],
            "x transformed": [0.0, 0.3, 0.48],
            "kobs": [2.0, 4.0, 6.0],
            "y transformed": [0.3, 0.6, 0.78],
        }
    )

    result = ecat_module.multi_scatterplot(
        {"rate": df},
        {"legend": False, "title": False},
    )

    assert result.table["x column"].unique().tolist() == ["x transformed"]
    assert result.table["y column"].unique().tolist() == ["y transformed"]
    np.testing.assert_allclose(result.axes.collections[0].get_offsets()[:, 0], [0.0, 0.3, 0.48])


def test_multi_scatterplot_plot_fits_from_explicit_raw_columns(ecat_module):
    df = pd.DataFrame({"Scan Rate": [1.0, 2.0, 4.0], "kobs": [3.0, 12.0, 48.0]})
    rate_result = ecat_module.fit_rate(
        df,
        {
            "plot": False,
            "print": False,
            "metric": "kobs",
            "transform mode": "log-log",
        },
    )

    result = ecat_module.multi_scatterplot(
        {"rate": rate_result},
        {
            "x column": "x raw",
            "y column": "kobs",
            "legend": False,
            "title": False,
        },
    )

    assert result.table["x column"].unique().tolist() == ["x raw"]
    assert result.table["y column"].unique().tolist() == ["kobs"]
    line = result.axes.lines[0]
    assert len(line.get_xdata()) >= 2
    expected = np.poly1d(np.polyfit([1.0, 2.0, 4.0], [3.0, 12.0, 48.0], 1))
    np.testing.assert_allclose(line.get_ydata(), expected(line.get_xdata()), rtol=1e-6)
    plt.close(result.figure)


def test_multi_scatterplot_selects_transformed_or_adjusted_columns(ecat_module):
    df = pd.DataFrame({"Scan Rate": [1.0, 2.0, 3.0], "kobs": [2.0, 4.0, 8.0]})
    rate_result = ecat_module.fit_rate(
        df,
        {
            "plot": False,
            "print": False,
            "metric": "kobs",
            "y mode": "enhancement",
            "transform mode": "log-log",
        },
    )

    transformed = ecat_module.multi_scatterplot(
        {"rate": rate_result},
        {
            "x column": "x transformed",
            "y column": "y transformed",
            "plot fit": False,
            "legend": False,
            "title": False,
        },
    )
    adjusted = ecat_module.multi_scatterplot(
        {"rate": rate_result},
        {
            "x column": "x raw",
            "y column": "y adjusted",
            "plot fit": False,
            "legend": False,
            "title": False,
        },
    )

    assert transformed.table["x column"].unique().tolist() == ["x transformed"]
    assert transformed.table["y column"].unique().tolist() == ["y transformed"]
    assert adjusted.table["x column"].unique().tolist() == ["x raw"]
    assert adjusted.table["y column"].unique().tolist() == ["y adjusted"]
    np.testing.assert_allclose(adjusted.axes.collections[0].get_offsets()[:, 1], [1.0, 3.0])
    assert adjusted.axes.get_ylabel() == r"$k_{\mathrm{obs}}$/$k_{\mathrm{obs}}^{0}$ - 1"
    assert transformed.axes.get_ylabel() == (
        r"$\log_{10}$($k_{\mathrm{obs}}$/$k_{\mathrm{obs}}^{0}$ - 1)"
    )
    plt.close(transformed.figure)
    plt.close(adjusted.figure)


def test_multi_scatterplot_raw_points_from_metric_series(
    ecat_module,
):
    df = pd.DataFrame({"Scan Rate": [1.0, 2.0, 3.0], "kobs": [2.0, 4.0, 8.0]})
    rate_result = ecat_module.fit_rate(
        df,
        {
            "plot": False,
            "print": False,
            "metric": "kobs",
            "y mode": "enhancement",
            "transform mode": "log-log",
        },
    )

    result = ecat_module.multi_scatterplot(
        {"rate": rate_result},
        {
            "x column": "x raw",
            "y column": "kobs",
            "plot fit": False,
            "print": False,
            "legend": False,
            "title": False,
        },
    )

    assert result.table["x column"].unique().tolist() == ["x raw"]
    assert result.table["y column"].unique().tolist() == ["kobs"]
    assert len(result.axes.lines) == 0
    np.testing.assert_allclose(result.axes.collections[0].get_offsets()[:, 0], [2.0, 3.0])
    np.testing.assert_allclose(result.axes.collections[0].get_offsets()[:, 1], [4.0, 8.0])
    plt.close(result.figure)


def test_multi_scatterplot_infers_axis_labels_from_scatter_fit_result(
    ecat_module,
    cv_factory,
):
    cvs = [
        cv_factory(name="50mVs_sample_CO2_MeCN_10mM_Fc_run01"),
        cv_factory(name="100mVs_sample_CO2_MeCN_10mM_Fc_run02"),
        cv_factory(name="200mVs_sample_CO2_MeCN_10mM_Fc_run03"),
    ]

    peak_result = ecat_module.fit_peak_current(
        cvs,
        {
            "plot": False,
            "print": False,
            "exact potential": 0.2,
            "noise window": 5,
            "noise polyorder": 2,
            "peak prominence": 1e-7,
            "tangent range": [0.05, 0.3],
            "percent threshold": 100,
        },
    )
    result = ecat_module.multi_scatterplot(
        {"Peak currents": peak_result},
        {"legend": False, "title": False},
    )

    assert result.axes.get_xlabel() != "x transformed"
    assert result.axes.get_ylabel() != "y transformed"
    assert "Scan Rate" in result.axes.get_xlabel()
    assert "Peak" in result.axes.get_ylabel()
    assert "Current" in result.axes.get_ylabel()
    plt.close(result.figure)


def test_multi_scatterplot_raw_peak_current_label(ecat_module):
    ip = pd.DataFrame({"x raw": [1.0, 2.0], "ip": [1e-6, 2e-6]})

    result = ecat_module.multi_scatterplot(
        {"trace": ip},
        {
            "x column": "x raw",
            "y column": "ip",
            "legend": False,
            "title": False,
        },
    )

    assert result.table["y column"].unique().tolist() == ["ip"]
    assert result.axes.get_ylabel() == r"$i_p$"
    plt.close(result.figure)


def test_multi_scatterplot_auto_resolves_common_rate_ip_ep_columns(ecat_module):
    rate = pd.DataFrame({"x raw": [1.0, 2.0], "kobs": [4.0, 8.0]})
    ip = pd.DataFrame({"x raw": [1.0, 2.0], "ip": [1e-6, 2e-6]})
    ep = pd.DataFrame({"x raw": [1.0, 2.0], "Ep": [0.1, 0.2]})

    for df, expected in [(rate, "kobs"), (ip, "ip"), (ep, "Ep")]:
        result = ecat_module.multi_scatterplot(
            {"trace": df},
            {"legend": False, "title": False},
        )
        assert result.table["y column"].unique().tolist() == [expected]
        plt.close(result.figure)


def test_multi_scatterplot_raw_axis_labels_use_chemical_formula_formatting(ecat_module):
    df = pd.DataFrame({"Scan Rate": [1.0, 2.0], "Fe(OH)2 current": [1e-6, 2e-6]})

    result = ecat_module.multi_scatterplot(
        {"trace": df},
        {
            "x column": "Scan Rate",
            "y column": "Fe(OH)2 current",
            "legend": False,
            "title": False,
        },
    )

    assert result.axes.get_ylabel() == "Fe(OH)$_2$ current"
    plt.close(result.figure)


def test_multi_scatterplot_ambiguous_y_columns_raise_clear_error(ecat_module):
    df = pd.DataFrame(
        {
            "x raw": [1.0, 2.0],
            "Seg 1 ip": [1.0, 2.0],
            "Seg 2 ip": [3.0, 4.0],
        }
    )

    with pytest.raises(ValueError, match="multiple candidates"):
        ecat_module.multi_scatterplot({"trace": df}, {"legend": False, "title": False})


def test_multi_scatterplot_supports_line_and_line_marker_styles(ecat_module):
    df = pd.DataFrame({"x raw": [1.0, 2.0, 3.0], "kobs": [2.0, 4.0, 6.0]})

    line_result = ecat_module.multi_scatterplot(
        {"line": df},
        {"plot style": "line", "fit": False, "legend": False, "title": False},
    )
    assert len(line_result.axes.lines) == 1
    assert line_result.axes.lines[0].get_marker() == "None"
    plt.close(line_result.figure)

    marker_result = ecat_module.multi_scatterplot(
        {"markers": df},
        {"plot style": "line+markers", "fit": False, "legend": False, "title": False},
    )
    assert len(marker_result.axes.lines) == 1
    assert marker_result.axes.lines[0].get_marker() == "o"
