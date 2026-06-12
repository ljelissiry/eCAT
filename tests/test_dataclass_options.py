import pytest


def test_peak_current_unknown_option_suggests_tangent_range(ecat_module):
    with pytest.raises(ecat_module.OptionError, match="tangent range"):
        ecat_module.PeakCurrentOptions.from_options({"tanget range": "auto"})


def test_peak_current_unknown_option_suggests_segment(ecat_module):
    with pytest.raises(ecat_module.OptionError, match="segment"):
        ecat_module.PeakCurrentOptions.from_options({"segmnt": 1})


def test_peak_potential_rejects_peak_current_only_option(ecat_module):
    with pytest.raises(ecat_module.OptionError, match="tangent range"):
        ecat_module.PeakPotentialOptions.from_options({"tangent range": "auto"})


def test_cv_option_value_validation(ecat_module):
    with pytest.raises(ecat_module.OptionError, match="odd integer"):
        ecat_module.PeakCurrentOptions.from_options({"noise window": 4})
    with pytest.raises(ecat_module.OptionError, match="less than"):
        ecat_module.PeakCurrentOptions.from_options({"noise window": 5, "noise polyorder": 5})
    with pytest.raises(ecat_module.OptionError, match="either 'segment' or 'segments'"):
        ecat_module.PeakCurrentOptions.from_options({"segment": 1, "segments": [1, 2]})


def test_trumpet_segments_override_default_segment(ecat_module):
    opts = ecat_module.TrumpetAnalysisOptions.from_options({"segments": [1, 2]})
    legacy = opts.to_legacy_dict()

    assert opts.segment is None
    assert opts.segments == [1, 2]
    assert legacy["segment"] == 1
    assert legacy["segments"] is None


def test_trumpet_segments_must_be_consecutive_pair(ecat_module):
    with pytest.raises(ecat_module.OptionError, match="consecutive"):
        ecat_module.TrumpetAnalysisOptions.from_options({"segments": [1, 3]})


def test_fowa_option_validation(ecat_module):
    with pytest.raises(ecat_module.OptionError, match="fit basis"):
        ecat_module.FOWAOptions.from_options({"fit basis": "z"})
    with pytest.raises(ecat_module.OptionError, match="redox mode"):
        ecat_module.FOWAOptions.from_options({"redox mode": "peakish"})


def test_fowa_accepts_multiplot_style_legend_options(ecat_module):
    opts = ecat_module.FOWAOptions.from_options(
        {
            "legend": True,
            "legend loc": "lower left",
            "legend mode": "colorbar",
            "legend outside": True,
        }
    )

    legacy = opts.to_legacy_dict()

    assert opts.legend_loc == "lower left"
    assert opts.legend_mode == "colorbar"
    assert opts.legend_outside is True
    assert legacy["legend loc"] == "lower left"
    assert legacy["legend mode"] == "colorbar"
    assert legacy["legend outside"] is True


def test_legend_mode_auto_is_default_and_colorbar_alias(ecat_module):
    multiplot_defaults = ecat_module.MultiplotOptions.from_options({}).to_legacy_dict()
    fowa_defaults = ecat_module.FOWAOptions.from_options({}).to_legacy_dict()

    assert multiplot_defaults["legend mode"] == "auto"
    assert fowa_defaults["legend mode"] == "auto"
    assert ecat_module._normalize_multiplot_legend_mode("auto") == "colorbar"
    assert ecat_module._normalize_multiplot_legend_mode("colorbar") == "colorbar"
    assert ecat_module._normalize_multiplot_legend_mode("discrete") == "discrete"


def test_fowa_accepts_fit_plot_options(ecat_module):
    opts = ecat_module.FOWAOptions.from_options(
        {
            "plot fit": False,
            "fit color": "tab:green",
            "fit linestyle": ":",
        }
    )

    legacy = opts.to_legacy_dict()

    assert opts.plot_fit is False
    assert legacy["plot fit"] is False
    assert legacy["fit color"] == "tab:green"
    assert legacy["fit linestyle"] == ":"


def test_fit_rate_accepts_print_all_as_common_verbosity_option(ecat_module):
    opts = ecat_module.FitRateOptions.from_options({"print all": True})

    assert opts.print_all is True
    assert opts.to_legacy_dict()["print all"] is True


def test_peak_current_routes_only_peak_potential_fields(ecat_module):
    opts = ecat_module.PeakCurrentOptions.from_options(
        {"guess potential": 0.1, "tangent range": [0.05, 0.3]}
    )

    routed = opts.for_peak_potential()

    assert isinstance(routed, ecat_module.PeakPotentialOptions)
    assert routed.guess_potential == pytest.approx(0.1)
    assert not hasattr(routed, "tangent_range")


def test_peak_current_routes_plot_axis_fields_to_peak_potential(ecat_module):
    opts = ecat_module.PeakCurrentOptions.from_options(
        {
            "guess potential": 0.1,
            "y axis": "i/ip0",
            "y unit": None,
            "ylabel": "$i / i_p^0$",
            "new plot": False,
        }
    )

    routed = opts.for_peak_potential()
    legacy = routed.to_legacy_dict()

    assert routed.y_axis == "i/ip0"
    assert legacy["y axis"] == "i/ip0"
    assert legacy["y unit"] is None
    assert legacy["ylabel"] == "$i / i_p^0$"
    assert legacy["new plot"] is False


def test_peak_potential_accepts_derivative_plot_option(ecat_module):
    opts = ecat_module.PeakPotentialOptions.from_options(
        {"plot": True, "print": False, "derivative": 1}
    )

    assert opts.derivative == 1
    assert opts.to_legacy_dict()["derivative"] == 1


def test_defaults_precedence_and_reset(ecat_module, tmp_path):
    ecat_module.reset_defaults()
    assert ecat_module.PeakCurrentOptions.from_options({}).tangent_range == "auto"

    ecat_module.set_defaults("peak_current", {"tangent range": [0.1, 0.4]})
    assert ecat_module.PeakCurrentOptions.from_options({}).tangent_range == [0.1, 0.4]
    assert ecat_module.PeakCurrentOptions.from_options({"tangent range": [0.2, 0.5]}).tangent_range == [0.2, 0.5]

    defaults_file = tmp_path / "defaults.toml"
    defaults_file.write_text("[peak_current]\ntangent_range = [0.3, 0.6]\n", encoding="utf-8")
    ecat_module.reset_defaults()
    ecat_module.load_defaults(defaults_file)
    assert ecat_module.PeakCurrentOptions.from_options({}).tangent_range == [0.3, 0.6]

    ecat_module.reset_defaults()
    assert ecat_module.PeakCurrentOptions.from_options({}).tangent_range == "auto"


def test_set_defaults_option_shorthand_applies_to_supporting_models(ecat_module):
    ecat_module.reset_defaults()

    ecat_module.set_defaults("print", False)

    assert ecat_module.ImportOptions.from_options({}).print is False
    assert ecat_module.PlotOptions.from_options({}).print is False
    assert ecat_module.FOWAOptions.from_options({}).print is False
    assert ecat_module.FitRateOptions.from_options({}).print is False
    assert ecat_module.FilterOptions.from_options({}).print is False
    assert ecat_module.FOWAOptions.from_options({"print": True}).print is True

    ecat_module.reset_defaults()


def test_set_defaults_section_override_wins_after_global_option(ecat_module):
    ecat_module.reset_defaults()

    ecat_module.set_defaults("print", False)
    ecat_module.set_defaults("fowa", {"print": True})

    assert ecat_module.PlotOptions.from_options({}).print is False
    assert ecat_module.FOWAOptions.from_options({}).print is True

    ecat_module.reset_defaults()


def test_set_defaults_option_shorthand_rejects_misspellings(ecat_module):
    ecat_module.reset_defaults()

    with pytest.raises(ecat_module.OptionError, match="print"):
        ecat_module.set_defaults("pritn", False)

    ecat_module.reset_defaults()


def test_reset_defaults_option_shorthand_and_ambiguous_plot_reset(ecat_module):
    ecat_module.reset_defaults()

    ecat_module.set_defaults("print", False)
    ecat_module.reset_defaults("print")
    assert ecat_module.PlotOptions.from_options({}).print is True
    assert ecat_module.FitRateOptions.from_options({}).print is True

    ecat_module.set_defaults("plot", False)
    assert ecat_module.PlotOptions.from_options({}).plot is False
    ecat_module.reset_defaults("plot")
    assert ecat_module.PlotOptions.from_options({}).plot is False
    ecat_module.reset_defaults_option("plot")
    assert ecat_module.PlotOptions.from_options({}).plot is True

    ecat_module.reset_defaults()
