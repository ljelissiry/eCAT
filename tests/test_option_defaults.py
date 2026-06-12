import builtins
import os

import pandas as pd
import pytest


def _option_row(df, option):
    matches = df.loc[df["Option"] == option]
    assert len(matches) == 1
    return matches.iloc[0]


def test_dataclass_options_expose_expected_legacy_defaults(ecat_module):
    import_defaults = ecat_module.ImportOptions.from_options({}).to_legacy_dict()
    plot_defaults = ecat_module.PlotOptions.from_options({}).to_legacy_dict()
    multiplot_defaults = ecat_module.MultiplotOptions.from_options({}).to_legacy_dict()
    multimultiplot_defaults = ecat_module.MultiMultiplotOptions.from_options({}).to_legacy_dict()
    filter_defaults = ecat_module.FilterOptions.from_options({}).to_legacy_dict()
    cv_analysis_defaults = ecat_module.PeakCurrentOptions.from_options({}).to_legacy_dict()

    assert import_defaults["recursive search"] is True
    assert import_defaults["reference mode"] == "auto"
    assert import_defaults["reference guess"] == "auto"
    assert import_defaults["reference label"] == "Fc/Fc+"

    assert plot_defaults["legend"] == "auto"
    assert plot_defaults["legend loc"] == "auto"
    assert plot_defaults["grid"] is False
    assert plot_defaults["segment color mode"] == "auto"
    assert plot_defaults["new plot"] is False
    assert "normalize" in plot_defaults

    assert multiplot_defaults["title"] == "auto"
    assert multiplot_defaults["legend"] is True
    assert multiplot_defaults["legend loc"] == "auto"
    assert multiplot_defaults["deduplicate labels"] is False
    assert multiplot_defaults["print"] is False
    assert multiplot_defaults["legend outside"] is False
    assert multiplot_defaults["colorbar tick labels"] == "endpoints"
    assert multiplot_defaults["colorbar trace ticks"] is True
    assert "gradient tick labels" not in multiplot_defaults
    assert "gradient show trace ticks" not in multiplot_defaults

    assert multimultiplot_defaults["titles"] == "auto"
    assert multimultiplot_defaults["subtitles"] == "auto"
    assert multimultiplot_defaults["print"] is False
    assert multimultiplot_defaults["legend"] is True

    assert filter_defaults["print"] is True
    assert filter_defaults["mode"] == "include"
    assert filter_defaults["logic"] is None

    assert cv_analysis_defaults["noise window"] == "auto"
    assert cv_analysis_defaults["noise polyorder"] == "auto"
    assert cv_analysis_defaults["plot cv"] is True
    assert cv_analysis_defaults["tangent range"] == "auto"
    assert cv_analysis_defaults["peak fallback"] == "highest current"
    assert cv_analysis_defaults["print all"] is False


def test_package_defaults_load_null_values_as_none(ecat_module):
    defaults = ecat_module.get_defaults("plot")

    assert defaults["label"] is None
    assert defaults["gradient_colormap"] is None


def test_describe_options_returns_human_readable_dataframe_when_requested(ecat_module):
    df = ecat_module.describe_options(
        ecat_module.PeakCurrentOptions,
        {"print": False, "return": True},
    )

    segment = _option_row(df, "segment")
    assert list(df.columns) == ["Category", "Option", "Default", "Type", "Description"]
    assert segment["Category"] == "Selection/filtering"
    assert segment["Default"] is None
    assert "int" in segment["Type"]
    assert "Choices" not in df.columns
    assert segment["Description"] == "CV segment to analyze."
    assert _option_row(df, "noise window")["Description"] == "Savitzky-Golay smoothing window."
    assert "underlying CV trace" in _option_row(df, "plot CV")["Description"]


def test_describe_options_pretty_print_displays_styled_dataframe_without_returning_dataframe(ecat_module, monkeypatch):
    displayed = {}

    def capture_display(obj):
        displayed["obj"] = obj
        displayed["df"] = obj.data.copy() if hasattr(obj, "data") else obj.copy()

    monkeypatch.setattr(ecat_module, "display", capture_display)

    result = ecat_module.describe_options("multiplot")

    assert result is None
    assert list(displayed["df"].columns) == ["Category", "Option", "Default", "Type", "Choices", "Description"]
    if hasattr(displayed["obj"], "table_styles"):
        assert displayed["obj"].table_styles[0]["props"] == [("text-align", "left")]
    assert "legend mode" in displayed["df"]["Option"].tolist()
    assert displayed["df"].loc[
        displayed["df"]["Option"] == "legend mode",
        "Default",
    ].iloc[0] == "auto"
    assert displayed["df"].loc[
        displayed["df"]["Option"] == "legend mode",
        "Choices",
    ].iloc[0] == "auto, colorbar, discrete"


def test_describe_options_pretty_print_can_also_return(ecat_module, monkeypatch):
    displayed = {}

    def capture_display(obj):
        displayed["df"] = obj.data.copy() if hasattr(obj, "data") else obj.copy()

    monkeypatch.setattr(ecat_module, "display", capture_display)

    df = ecat_module.describe_options(
        "multiplot",
        {"return": True},
    )

    assert df.equals(displayed["df"])
    assert _option_row(df, "legend mode")["Default"] == "auto"


def test_describe_options_pretty_print_hides_empty_display_columns(ecat_module, monkeypatch):
    displayed = {}

    def capture_display(obj):
        displayed["df"] = obj.data.copy() if hasattr(obj, "data") else obj.copy()

    monkeypatch.setattr(ecat_module, "display", capture_display)

    result = ecat_module.describe_options("peak_current")

    assert result is None
    assert "Choices" not in displayed["df"].columns


def test_describe_options_pretty_print_false_uses_plain_output_without_display(ecat_module, monkeypatch, capsys):
    def fail_display(_df):
        raise AssertionError("describe_options should not display when pretty is False")

    monkeypatch.setattr(ecat_module, "display", fail_display)

    result = ecat_module.describe_options("multiplot", {"pretty print": False})
    captured = capsys.readouterr()

    assert result is None
    assert "legend mode" in captured.out
    assert "Category" in captured.out


def test_describe_options_print_false_suppresses_output_without_returning_dataframe(ecat_module, monkeypatch, capsys):
    def fail_display(_df):
        raise AssertionError("describe_options should not display when print is False")

    monkeypatch.setattr(ecat_module, "display", fail_display)

    result = ecat_module.describe_options("multiplot", {"print": False})
    captured = capsys.readouterr()

    assert result is None
    assert captured.out == ""


def test_describe_options_accepts_return_option(ecat_module, monkeypatch):
    def fail_display(_df):
        raise AssertionError("describe_options should not display when print is False")

    monkeypatch.setattr(ecat_module, "display", fail_display)

    schema = ecat_module.describe_options(
        "multiplot",
        {"print": False, "return": True},
    )

    legend_mode = _option_row(schema, "legend mode")
    assert legend_mode["Category"] == "Legend"
    assert legend_mode["Default"] == "auto"
    assert legend_mode["Choices"] == "auto, colorbar, discrete"


def test_describe_options_multiplot_uses_specific_plotting_categories(ecat_module):
    schema = ecat_module.describe_options(
        "multiplot",
        {"print": False, "return": True},
    )

    assert _option_row(schema, "plot labels")["Category"] == "Labels/titles"
    assert _option_row(schema, "legend mode")["Category"] == "Legend"
    assert _option_row(schema, "gradient by")["Category"] == "Color mapping"
    assert _option_row(schema, "gradient scale")["Choices"] == "auto, linear, sqrt, log, index"
    assert _option_row(schema, "gradient reverse")["Category"] == "Color mapping"
    assert _option_row(schema, "colorbar reverse")["Category"] == "Colorbar"
    assert _option_row(schema, "colorbar tick labels")["Choices"] == "endpoints, all, none"
    assert _option_row(schema, "colorbar tick labels")["Category"] == "Colorbar"
    assert _option_row(schema, "colorbar trace ticks")["Category"] == "Colorbar"
    assert "gradient tick labels" not in set(schema["Option"])
    assert "gradient show trace ticks" not in set(schema["Option"])


def test_multiplot_rejects_old_gradient_colorbar_option_names(ecat_module):
    with pytest.raises(ecat_module.OptionError, match="gradient tick labels"):
        ecat_module.MultiplotOptions.from_options({"gradient tick labels": "all"})

    with pytest.raises(ecat_module.OptionError, match="gradient show trace ticks"):
        ecat_module.MultiplotOptions.from_options({"gradient show trace ticks": False})


def test_describe_options_lists_iupac_plot_convention_choice(ecat_module):
    df = ecat_module.describe_options("plot", {"print": False, "return": True})

    assert _option_row(df, "plot convention")["Choices"] == "US, IUPAC"


def test_describe_options_plot_includes_derivative(ecat_module):
    df = ecat_module.describe_options("plot", {"print": False, "return": True})

    derivative = _option_row(df, "derivative")
    assert derivative["Default"] == 0
    assert derivative["Type"] == "int or float or None"


def test_describe_options_normalize_documents_auto_collected_metadata(ecat_module):
    df = ecat_module.describe_options("normalize", {"print": False, "return": True})

    assert "E0" in set(df["Option"])
    assert "e0" not in set(df["Option"])
    assert "CV's temperature" in _option_row(df, "temperature")["Description"]
    assert "CV's electrode_area" in _option_row(df, "electrode area")["Description"]
    assert "CV's scan_rate" in _option_row(df, "scan rate")["Description"]
    assert "exact-matches cv.compounds" in _option_row(df, "species")["Description"]


def test_describe_options_uses_function_specific_metadata_for_shared_option_names(ecat_module):
    normalize_schema = ecat_module.describe_options("normalize", {"print": False, "return": True})
    filter_schema = ecat_module.describe_options("filter", {"print": False, "return": True})

    normalize_mode = _option_row(normalize_schema, "mode")
    filter_mode = _option_row(filter_schema, "mode")

    assert normalize_mode["Choices"] == "homogeneous, heterogeneous"
    assert "Normalization family" in normalize_mode["Description"]
    assert filter_mode["Choices"] == "include, exclude"
    assert "include or exclude" in filter_mode["Description"]


def test_describe_options_all_uses_function_specific_metadata(ecat_module):
    df = ecat_module.describe_options("all", {"print": False, "return": True})
    mode_rows = df.loc[df["Option"] == "mode"]

    normalize_mode = mode_rows.loc[mode_rows["Function"] == "normalize"].iloc[0]
    filter_mode = mode_rows.loc[mode_rows["Function"] == "filter"].iloc[0]

    assert normalize_mode["Choices"] == "homogeneous, heterogeneous"
    assert "Normalization family" in normalize_mode["Description"]
    assert filter_mode["Choices"] == "include, exclude"
    assert "include or exclude" in filter_mode["Description"]


def test_describe_options_dataclass_uses_function_specific_metadata_when_unambiguous(ecat_module):
    df = ecat_module.describe_options(
        ecat_module.NormalizeOptions,
        {"print": False, "return": True},
    )

    mode = _option_row(df, "mode")
    assert mode["Choices"] == "homogeneous, heterogeneous"
    assert "Normalization family" in mode["Description"]


def test_describe_options_documents_auto_retrieval_and_algorithmic_auto_behavior(ecat_module):
    get_data_schema = ecat_module.describe_options("get_data", {"print": False, "return": True})
    plot_schema = ecat_module.describe_options("plot", {"print": False, "return": True})
    cv_analysis_schema = ecat_module.describe_options("cv_analysis", {"print": False, "return": True})
    peak_schema = ecat_module.describe_options("peak_current", {"print": False, "return": True})
    plateau_schema = ecat_module.describe_options("plateau_current", {"print": False, "return": True})
    nicholson_schema = ecat_module.describe_options("nicholson", {"print": False, "return": True})
    trumpet_schema = ecat_module.describe_options("trumpet_analysis", {"print": False, "return": True})

    assert "searches imported files" in _option_row(get_data_schema, "reference mode")["Description"]
    assert "locates the reference wave automatically" in _option_row(get_data_schema, "reference guess")["Description"]

    assert "auto scales from the displayed data" in _option_row(plot_schema, "x unit")["Description"]
    assert "auto chooses colorbar" in _option_row(plot_schema, "legend mode")["Description"]
    assert "auto colors multi-segment CVs" in _option_row(plot_schema, "segment color mode")["Description"]

    assert "'auto' chooses an odd Savitzky-Golay window" in _option_row(cv_analysis_schema, "noise window")["Description"]
    assert "'auto' chooses a pre-peak baseline region" in _option_row(peak_schema, "tangent range")["Description"]
    assert "largest absolute current" in _option_row(peak_schema, "peak fallback")["Description"]

    assert "uses CV or reference-CV temperature metadata" in _option_row(plateau_schema, "temperature")["Description"]
    assert "uses the catalytic CV's electrode_area" in _option_row(plateau_schema, "electrode area")["Description"]
    assert "chooses direct, slope-normalized, or normalized" in _option_row(plateau_schema, "formula mode")["Description"]

    assert "uses each CV's scan_rate" in _option_row(nicholson_schema, "scan rate")["Description"]
    assert "uses each CV's temperature" in _option_row(trumpet_schema, "temperature")["Description"]


def test_describe_options_documents_not_inferred_scientific_inputs(ecat_module):
    normalize_schema = ecat_module.describe_options("normalize", {"print": False, "return": True})
    plateau_schema = ecat_module.describe_options("plateau_current", {"print": False, "return": True})
    trumpet_schema = ecat_module.describe_options("trumpet_analysis", {"print": False, "return": True})

    assert "not inferred" in _option_row(normalize_schema, "D")["Description"]
    assert "not inferred" in _option_row(normalize_schema, "E0")["Description"]
    assert "not inferred" in _option_row(plateau_schema, "D")["Description"]
    assert "not inferred" in _option_row(trumpet_schema, "D")["Description"]


def test_describe_options_documents_auto_result_column_resolution(ecat_module):
    scatter_schema = ecat_module.describe_options("multi_scatterplot", {"print": False, "return": True})
    fit_rate_schema = ecat_module.describe_options("fit_rate", {"print": False, "return": True})

    assert "auto prefers transformed/raw x columns" in _option_row(scatter_schema, "x column")["Description"]
    assert "auto prefers transformed, metric, kobs, TOFmax, ip, then Ep" in _option_row(scatter_schema, "y column")["Description"]
    assert "auto chooses a sensible x column" in _option_row(fit_rate_schema, "x column")["Description"]


def test_minimum_gradient_entries_alias_routes_to_multiplot_threshold(ecat_module):
    options = ecat_module.MultiplotOptions.from_options({"minimum gradient entries": 5})

    assert options.min_gradient_entries == 5


def test_reference_mode_choices_are_context_specific(ecat_module):
    get_data_schema = ecat_module.describe_options("get_data", {"print": False, "return": True})
    scale_schema = ecat_module.describe_options("scale_current", {"print": False, "return": True})

    assert _option_row(get_data_schema, "reference mode")["Choices"] == "auto, manual, keyword, file, none"
    assert _option_row(scale_schema, "reference mode")["Choices"] == "single, both"


def test_explain_options_is_not_public_api(ecat_module):
    assert not hasattr(ecat_module, "explain_options")


def test_all_registered_option_fields_have_descriptions():
    from dataclasses import fields

    from ecat import options as option_module

    missing = []
    for cls, _sections in option_module._option_default_registry():
        for field in fields(cls):
            if not option_module.OPTION_DESCRIPTIONS.get(option_module.normalize_key(field.name)):
                missing.append(f"{cls.__name__}.{field.name}")

    assert missing == []


def test_all_default_options_have_descriptions():
    from ecat import options as option_module

    missing = []
    for section, values in option_module.get_defaults().items():
        for key in values:
            if not option_module.OPTION_DESCRIPTIONS.get(option_module.normalize_key(key)):
                missing.append(f"{section}.{key}")

    assert missing == []


def test_describe_options_no_argument_returns_menu_with_all_first(ecat_module):
    menu = ecat_module.describe_options(None, {"print": False, "return": True})
    functions = menu["Function"].tolist()

    assert list(menu.columns) == ["Function", "Description"]
    assert functions[0] == "all"
    assert "cv.peak_current" in functions
    assert "cv.peak_potential" in functions
    assert "dpv.peak_potential" in functions
    assert "ca.charge" in functions
    assert "cp.cycle_info" in functions
    assert "get_data" in functions
    assert "fit_peak_current" in functions
    assert "fit_peak_potential" in functions
    assert "fowa" in functions
    assert "sevcik_analysis" in functions
    assert "trumpet_analysis" in functions
    assert "normalize" in functions
    assert "normalize_current" in functions
    assert "scale_current" in functions
    assert "standardize_current_by_reference_wave" not in functions
    assert "import data" not in functions
    assert "fit ip" not in functions
    assert "ep fit" not in functions
    assert "sevcik" not in functions
    assert "trumpet" not in functions


def test_describe_options_accepts_public_method_names(ecat_module):
    cases = {
        "cv.peak_current": ("tangent range", "peak fallback"),
        "cv.half_wave_potential": ("tangent range", "peak fallback"),
        "cv.peak_potential": ("guess potential", "noise window"),
        "cv.half_peak_potential": ("guess potential", "noise window"),
        "cv.plateau_current": ("ilim", "formula mode"),
        "cv.normalize": ("area", "C"),
        "cv.normalize_current": ("ip0", "reference cv"),
        "cv.scale_current": ("reference index", "reference cv"),
        "dpv.peak_potential": ("guess potential", "noise window"),
        "ca.charge": ("plot", "x unit"),
        "ca.time_at_charge": ("plot", "x unit"),
        "cp.cycle_info": ("plot segment", "plot"),
        "cp.plot_cycles": ("plot segment", "plot"),
        "cp.cycling_plot": ("cycles", "plot"),
    }

    for method_name, expected_options in cases.items():
        df = ecat_module.describe_options(method_name, {"print": False, "return": True})
        options = set(df["Option"])
        for expected in expected_options:
            assert expected in options, method_name


def test_describe_options_method_name_matches_underlying_section(ecat_module):
    method_df = ecat_module.describe_options("cv.peak_current", {"print": False, "return": True})
    section_df = ecat_module.describe_options("peak_current", {"print": False, "return": True})

    assert method_df.equals(section_df)


def test_describe_options_all_returns_all_option_rows(ecat_module):
    df = ecat_module.describe_options("all", {"print": False, "return": True})

    assert "Function" in df.columns
    assert "Category" in df.columns
    assert "Option" in df.columns
    assert "get_data" in df["Function"].tolist()
    assert "plot" in df["Function"].tolist()
    assert "sort keys" in df.loc[df["Function"] == "get_data", "Option"].tolist()


def test_describe_options_sorts_by_function_category_and_option(ecat_module):
    from ecat import options as option_module

    df = ecat_module.describe_options("all", {"print": False, "return": True})
    category_order = {
        category: index
        for index, category in enumerate(option_module.OPTION_CATEGORY_ORDER)
    }
    observed = [
        (
            row["Function"],
            category_order[row["Category"]],
            row["Option"],
        )
        for _, row in df.iterrows()
    ]

    assert observed == sorted(observed)


def test_describe_options_function_sorts_by_category_and_option(ecat_module):
    from ecat import options as option_module

    df = ecat_module.describe_options("plot", {"print": False, "return": True})
    category_order = {
        category: index
        for index, category in enumerate(option_module.OPTION_CATEGORY_ORDER)
    }
    observed = [
        (
            category_order[row["Category"]],
            row["Option"],
        )
        for _, row in df.iterrows()
    ]

    assert observed == sorted(observed)


def test_describe_options_fowa_includes_tangent_background_controls(ecat_module):
    df = ecat_module.describe_options("fowa", {"print": False, "return": True})

    expected_options = [
        "background correction",
        "tangent potential",
        "tangent range",
        "tangent min points",
        "percent threshold",
    ]

    for option in expected_options:
        assert _option_row(df, option)["Category"] == "Fitting/analysis"

    assert _option_row(df, "troubleshoot")["Category"] == "Output/display"
    assert "tangent activity fraction" not in df["Option"].tolist()


def test_plot_options_accepts_grid_display_option(ecat_module):
    options = ecat_module.PlotOptions.from_options({"grid": True}).to_legacy_dict()

    assert options["grid"] is True
    schema = ecat_module.describe_options("plot", {"print": False, "return": True})
    assert _option_row(schema, "grid")["Category"] == "Plotting"


def test_describe_options_all_registered_options_have_allowed_categories(ecat_module):
    from ecat import options as option_module

    df = ecat_module.describe_options("all", {"print": False, "return": True})

    assert df["Category"].isna().sum() == 0
    assert (df["Category"].astype(str).str.strip() != "").all()
    assert set(df["Category"]).issubset(option_module.OPTION_CATEGORIES)
    assert set(df["Category"]) == option_module.OPTION_CATEGORIES


def test_describe_options_invalid_section_shows_friendly_menu(ecat_module, monkeypatch, capsys):
    displayed = {}

    def capture_display(obj):
        displayed["df"] = obj.data.copy() if hasattr(obj, "data") else obj.copy()

    monkeypatch.setattr(ecat_module, "display", capture_display)

    result = ecat_module.describe_options("mulitplot")
    captured = capsys.readouterr()

    assert result is None
    assert "Unknown options function 'mulitplot'. Did you mean 'multiplot'?" in captured.out
    assert displayed["df"]["Function"].tolist()[0] == "all"
    assert "multiplot" in displayed["df"]["Function"].tolist()


def test_describe_options_invalid_section_without_suggestion_returns_possible_sections(ecat_module, capsys):
    menu = ecat_module.describe_options("zzz", {"print": False, "return": True})
    captured = capsys.readouterr()

    assert captured.out == ""
    assert menu["Function"].tolist()[0] == "all"
    assert "get_data" in menu["Function"].tolist()


def test_describe_options_documents_its_own_print_controls(ecat_module):
    df = ecat_module.describe_options("describe_options", {"print": False, "return": True})

    assert bool(_option_row(df, "print")["Default"]) is True
    assert bool(_option_row(df, "pretty print")["Default"]) is True
    assert bool(_option_row(df, "return")["Default"]) is False
    assert "False suppresses output" in _option_row(df, "print")["Description"]
    assert "does not suppress output" in _option_row(df, "pretty print")["Description"]


def test_fit_analysis_legend_fontsize_defaults_to_plot_style(ecat_module):
    for function in (
        "fit_peak_potential",
        "sevcik_analysis",
        "fit_peak_current",
        "trumpet_analysis",
    ):
        df = ecat_module.describe_options(function, {"print": False, "return": True})
        assert _option_row(df, "legend fontsize")["Default"] is None


def test_friendly_default_section_aliases_route_to_canonical_sections(ecat_module):
    ecat_module.reset_defaults()

    friendly_to_current = {
        "sevcik analysis": "sevcik_analysis",
        "sevcik": "sevcik_analysis",
        "trumpet": "trumpet_analysis",
        "trumpet analysis": "trumpet_analysis",
        "tafel analysis": "tafel_analysis",
        "tafel": "tafel_analysis",
        "peak current": "peak_current",
        "fit peak current": "fit_peak_current",
        "fit peak potential": "fit_peak_potential",
    }

    for section, new_section in friendly_to_current.items():
        schema = ecat_module.describe_options(section, {"print": False, "return": True})
        assert "Default" in schema.columns, section
        assert schema.attrs.get("section") in {None, new_section}

    ecat_module.reset_defaults()


def test_option_choices_registry_is_described_and_registered():
    from dataclasses import fields

    from ecat import options as option_module

    field_names = set()
    default_names = set()
    for cls, _sections in option_module._option_default_registry():
        field_names.update(field.name for field in fields(cls))
    for values in option_module.get_defaults().values():
        default_names.update(values)

    missing_description = []
    missing_registration = []
    for key in option_module.OPTION_CHOICES:
        if not option_module.OPTION_DESCRIPTIONS.get(key):
            missing_description.append(key)
        if key not in field_names and key not in default_names:
            missing_registration.append(key)

    assert missing_description == []
    assert missing_registration == []


def test_cv_analysis_option_helper_applies_defaults_and_tangent_range_normalization(cv_factory):
    obj = cv_factory()

    defaulted = obj._cv_analysis_options({})
    normalized = obj._cv_analysis_options({"tangent range": 0.5})
    normalize_warped = obj._cv_analysis_options({"normalize": True})

    assert defaulted["noise window"] == "auto"
    assert defaulted["noise polyorder"] == "auto"
    assert defaulted["tangent range"] == "auto"
    assert defaulted["segment"] is None
    assert defaulted["plot"] is True

    assert normalized["tangent range"] == [0.05, 0.5]
    assert normalize_warped["normalize"] is True
    assert normalize_warped["tangent range"] == "auto"


def test_multiplot_passes_default_expanded_options_to_each_curve(
    ecat_module,
    cv_factory,
    monkeypatch,
):
    import matplotlib as mpl

    objects = [
        cv_factory(name="50mVs_sample_CO2_MeCN_10mM_Fc_run01"),
        cv_factory(name="100mVs_sample_N2_DMF_5mM_Fc_run01"),
    ]
    observed_options = []

    def spy_plot(self, options=None):
        captured = dict(options or {})
        observed_options.append(captured)
        return f"ax-{len(observed_options)}"

    monkeypatch.setattr(ecat_module.cv, "plot", spy_plot)
    monkeypatch.setattr(
        ecat_module,
        "_draw_multiplot_legend_and_colorbars",
        lambda *args, **kwargs: None,
    )

    ax = ecat_module.multiplot(objects, {})

    assert isinstance(ax, mpl.axes.Axes)
    assert len(observed_options) == 2
    assert all(opts["legend"] is False for opts in observed_options)
    assert all(opts["title"] == "auto" for opts in observed_options)
    assert all(opts["subtitle"] == "auto" for opts in observed_options)
    assert all(opts["legend outside"] is False for opts in observed_options)
    assert all(opts["print"] is False for opts in observed_options)
    assert observed_options[0]["offset"] == pytest.approx(0)
    assert observed_options[1]["offset"] == pytest.approx(
        ecat_module.MultiplotOptions.from_options({}).offset
    )


def test_multimultiplot_forwards_default_expanded_options_into_multiplot(
    ecat_module,
    cv_factory,
    monkeypatch,
):
    objects = [
        cv_factory(name="50mVs_sample_CO2_MeCN_10mM_Fc_run01"),
        cv_factory(name="100mVs_sample_N2_DMF_5mM_Fc_run01"),
    ]
    observed_calls = []

    def spy_multiplot(group, options=None):
        observed_calls.append((group, dict(options or {})))
        return ["sentinel"]

    monkeypatch.setattr(ecat_module, "multiplot", spy_multiplot)

    result = ecat_module.multimultiplot([[objects[0]], [objects[1]]], {})

    assert result is None
    assert len(observed_calls) == 2
    assert observed_calls[0][0] == [objects[0]]
    assert observed_calls[1][0] == [objects[1]]
    assert observed_calls[0][1]["titles"] == "auto"
    assert observed_calls[0][1]["subtitles"] == "auto"
    assert observed_calls[0][1]["legend"] is True
    assert observed_calls[0][1]["print"] is False
    assert observed_calls[0][1]["title"] == "Group 0: GENERATE"
    assert observed_calls[0][1]["subtitle"] == "auto"
    assert observed_calls[1][1]["title"] == "Group 1: GENERATE"


def test_filter_applies_default_logic_and_respects_override(
    ecat_module,
    cv_factory,
    monkeypatch,
):
    objects = [
        cv_factory(name="50mVs_sample_CO2_MeCN_10mM_Fc_run01"),
        cv_factory(name="100mVs_sample_N2_DMF_5mM_Fc_run01"),
    ]
    recorded_messages = []

    def fake_print(*args, **kwargs):
        recorded_messages.append(" ".join(str(arg) for arg in args))

    monkeypatch.setattr(builtins, "print", fake_print)

    filtered_default = ecat_module.filter(objects, {"gas": "CO2"}, options={})
    filtered_override = ecat_module.filter(
        objects,
        {"gas": "CO2", "solvent": "DMF"},
        options={"logic": "OR", "print": False},
    )

    assert filtered_default == [objects[0]]
    assert filtered_override == objects
    assert any("Filtered Objects (Include)" in message for message in recorded_messages)


def test_get_data_uses_import_defaults_before_filesystem_work(
    ecat_module,
    monkeypatch,
    tmp_path,
):
    observed = {}

    def fake_exists(path):
        observed["checked_path"] = path
        return False

    monkeypatch.setattr(os.path, "exists", fake_exists)

    result = ecat_module.get_data({"folder path": str(tmp_path / "missing-folder")})

    assert result is None
    assert observed["checked_path"].endswith("missing-folder")


def test_get_data_accepts_import_options_dataclass(ecat_module, monkeypatch, tmp_path):
    observed = {}

    def fake_exists(path):
        observed["checked_path"] = path
        return False

    monkeypatch.setattr(os.path, "exists", fake_exists)
    options = ecat_module.ImportOptions.from_options({"folder path": str(tmp_path / "missing-folder")})

    result = ecat_module.get_data(options)

    assert result is None
    assert observed["checked_path"].endswith("missing-folder")


def test_describe_options_get_data_includes_sort_keys(ecat_module):
    df = ecat_module.describe_options("get_data", {"print": False, "return": True})

    assert _option_row(df, "sort keys")["Default"] == ["timestamp"]


def test_create_cv_objects_from_excel_accepts_import_options_dataclass(ecat_module, monkeypatch):
    from ecat import io as ecat_io

    monkeypatch.setattr(ecat_io.pd, "read_excel", lambda *args, **kwargs: {})
    options = ecat_module.ImportOptions.from_options({"print": False})

    assert ecat_io.create_cv_objects_from_excel("fake.xlsx", options) == []


def test_get_cvs_from_excel_alias_accepts_import_options_dataclass(ecat_module, monkeypatch):
    from ecat import io as ecat_io

    monkeypatch.setattr(ecat_io.pd, "read_excel", lambda *args, **kwargs: {})
    options = ecat_module.ImportOptions.from_options({"print": False})

    assert ecat_io.get_CVs_from_excel("fake.xlsx", options) == []


def test_create_cv_objects_from_excel_rejects_unknown_import_option(ecat_module):
    from ecat import io as ecat_io

    with pytest.raises(ecat_module.OptionError, match="troubleshoot"):
        ecat_io.create_cv_objects_from_excel("fake.xlsx", {"troubleshot": True})


def test_get_data_rejects_unknown_import_option_with_suggestion(ecat_module, tmp_path):
    with pytest.raises(ecat_module.OptionError, match="recursive search"):
        ecat_module.get_data(
            {
                "folder path": str(tmp_path),
                "recursive serch": False,
            }
        )


def test_get_data_import_defaults_and_overrides_flow_into_reference_normalization(
    ecat_module,
    monkeypatch,
    tmp_path,
):
    observed_options = []

    monkeypatch.setattr(os.path, "exists", lambda path: True)
    monkeypatch.setattr(os.path, "isdir", lambda path: True)
    monkeypatch.setattr(ecat_module.glob, "glob", lambda pattern, recursive=False: [])
    monkeypatch.setattr(builtins, "print", lambda *args, **kwargs: None)

    original_normalize = ecat_module.normalize_legacy_reference_options

    def spy_normalize(options):
        observed_options.append(dict(options))
        return original_normalize(options)

    monkeypatch.setattr(ecat_module, "normalize_legacy_reference_options", spy_normalize)

    result_default = ecat_module.get_data({"folder path": str(tmp_path)})
    result_manual = ecat_module.get_data(
        {
            "folder path": str(tmp_path),
            "recursive search": False,
            "reference mode": "manual",
            "reference offset": 0.123,
        }
    )

    assert result_default is None
    assert result_manual is None

    default_options = observed_options[0]
    manual_options = next(
        opts for opts in observed_options if opts.get("recursive search") is False
    )

    assert default_options["recursive search"] is True
    assert default_options["reference mode"] == "auto"
    assert default_options["reference guess"] == "auto"
    assert default_options["reference label"] == "Fc/Fc+"
    assert default_options["print"] is True
    assert manual_options["recursive search"] is False
    assert manual_options["reference mode"] == "manual"
    assert manual_options["reference offset"] == pytest.approx(0.123)


def test_build_object_table_ignores_import_parser_columns_integer(ecat_module, cv_factory):
    obj = cv_factory()

    table, _metadata = ecat_module.build_object_table([obj], {"columns": 3})

    assert "Name" in table.columns


def test_build_object_table_keeps_display_columns_string(ecat_module, cv_factory):
    obj = cv_factory()

    table, _metadata = ecat_module.build_object_table(
        [obj],
        {"columns": "gas", "print conditions": False},
    )

    assert "Gas" in table.columns


def test_show_objects_columns_available_returns_canonical_columns(ecat_module, cv_factory):
    obj = cv_factory()

    columns = ecat_module.show_objects(
        [obj],
        {"columns": "available", "print conditions": False},
    )

    assert isinstance(columns, list)
    assert "name" in columns
    assert "gas" in columns
    assert "scan rate" in columns
    assert "reference shift" not in columns
    assert "reference label" not in columns
    assert "reference mode" not in columns
    assert "reference source" not in columns
    assert "ir comp resistance" in columns
    assert "ir uncomp resistance" in columns
    assert "ir comp percent" in columns
    assert "temperature" in columns
    assert "electrode area" in columns
    assert all(column == column.lower() for column in columns)


def test_show_objects_columns_available_includes_reference_columns_when_reference_exists(ecat_module, cv_factory):
    obj = cv_factory()
    obj.reference_shift = 0.401
    obj.reference_mode = "manual"
    obj.reference_label = "Fc/Fc+"

    columns = ecat_module.show_objects(
        [obj],
        {"columns": "available", "print conditions": False},
    )

    assert "reference shift" in columns
    assert "reference label" in columns
    assert "reference mode" in columns
    assert "reference source" in columns


def test_show_single_echem_displays_info_stats_table(ecat_module, cv_factory, monkeypatch):
    obj = cv_factory()
    displayed = {}

    def capture_display(table):
        displayed["table"] = table.copy()

    monkeypatch.setattr(ecat_module, "display_object_table", capture_display)

    result = ecat_module.show(obj)

    assert result is None
    table = displayed["table"]
    assert list(table.columns) == ["Metric", "Value"]
    assert "Name" in table["Metric"].tolist()
    assert "Scan Rate" in table["Metric"].tolist()
    assert "Segments" in table["Metric"].tolist()
    assert "Reference Shift" not in table["Metric"].tolist()
    assert "Reference Label" not in table["Metric"].tolist()
    assert "Reference Mode" not in table["Metric"].tolist()
    assert "Reference Source File" not in table["Metric"].tolist()
    assert "Folder Path" not in table["Metric"].tolist()


def test_show_single_echem_return_option_returns_info_stats_table(ecat_module, cv_factory):
    obj = cv_factory()

    table = ecat_module.show(obj, {"pretty print": False, "return": True})

    assert list(table.columns) == ["Metric", "Value"]
    assert table.loc[table["Metric"] == "Name", "Value"].iloc[0] == obj.name
    assert "Reference Mode" not in table["Metric"].tolist()


def test_show_single_echem_tolerates_missing_folderpath(ecat_module, cv_factory):
    obj = cv_factory()
    delattr(obj, "folderpath")

    table = ecat_module.show(obj, {"pretty print": False, "return": True})

    assert "Folder Path" not in table["Metric"].tolist()


def test_show_single_echem_displays_reference_source_as_relative_path(
    ecat_module,
    cv_factory,
    tmp_path,
):
    sample_dir = tmp_path / "samples"
    reference_dir = tmp_path / "references"
    sample_dir.mkdir()
    reference_dir.mkdir()
    target_file = sample_dir / "target.txt"
    reference_file = reference_dir / "fc.txt"
    target_file.write_text("", encoding="utf-8")
    reference_file.write_text("", encoding="utf-8")

    obj = cv_factory()
    obj.filepath = str(target_file)
    obj.folderpath = "samples"
    obj.reference_shift = 0.40123
    obj.reference_mode = "folder"
    obj.reference_label = "Fc/Fc+"
    obj.reference_source_file = str(reference_file)

    table = ecat_module.show(obj, {"pretty print": False, "return": True})
    values = dict(zip(table["Metric"], table["Value"]))

    assert values["Reference Source File"] == "references/fc.txt"


def test_show_single_echem_respects_sig_figs_for_numeric_values(
    ecat_module,
    cv_factory,
):
    obj = cv_factory()
    obj.scan_rate = 0.123456
    obj.reference_shift = 0.401234
    obj.reference_mode = "manual"

    table = ecat_module.show(
        obj,
        {"pretty print": False, "return": True, "sig figs": 2},
    )
    values = dict(zip(table["Metric"], table["Value"]))

    assert values["Scan Rate"] == "0.12"
    assert values["Reference Shift"] == "0.4"


def test_show_routes_flat_list_to_show_objects(ecat_module, cv_factory):
    obj = cv_factory()

    table = ecat_module.show(
        [obj],
        {
            "columns": ["gas"],
            "print conditions": False,
            "pretty print": False,
            "return": True,
        },
    )

    assert "Gas" in table.columns


def test_show_group_delegates_to_show_objects(ecat_module, cv_factory, monkeypatch):
    group = [cv_factory()]
    seen = {}

    def fake_show_objects(objects, options=None):
        seen["objects"] = objects
        seen["options"] = options
        return "object table"

    monkeypatch.setattr(ecat_module, "show_objects", fake_show_objects)

    result = ecat_module.show(group, {"return": True})

    assert result == "object table"
    assert seen["objects"] is group
    assert seen["options"] == {"return": True}


def test_show_routes_grouped_list_to_show_groups(ecat_module, cv_factory):
    obj = cv_factory()

    tables = ecat_module.show(
        [[obj]],
        {
            "columns": ["gas"],
            "print conditions": False,
            "pretty print": False,
            "return": True,
        },
    )

    assert len(tables) == 1
    assert "Gas" in tables[0].columns


def test_show_groups_delegates_to_show_groups(ecat_module, cv_factory, monkeypatch):
    groups = [[cv_factory()]]
    seen = {}

    def fake_show_groups(grouped_objects, options=None):
        seen["groups"] = grouped_objects
        seen["options"] = options
        return "group tables"

    monkeypatch.setattr(ecat_module, "show_groups", fake_show_groups)

    result = ecat_module.show(groups, {"return": True})

    assert result == "group tables"
    assert seen["groups"] is groups
    assert seen["options"] == {"return": True}


def test_show_delegates_to_objects_with_show_method(ecat_module):
    class Result:
        def __init__(self):
            self.options = None

        def show(self, options=None):
            self.options = options
            return "shown"

    result = Result()

    assert ecat_module.show(result, {"return": True}) == "shown"
    assert result.options == {"return": True}


def test_show_delegates_options_to_scatter_fit_result(ecat_module):
    import pandas as pd

    table = pd.DataFrame({"x": [1], "y": [2]})
    result = ecat_module.ScatterFitResult(table=table)

    returned = ecat_module.show(result, {"return": True})

    assert returned is table


def test_show_invalid_input_raises_clear_type_error(ecat_module):
    with pytest.raises(TypeError, match="show\\(\\) expected an eCAT object"):
        ecat_module.show(object())


def test_build_object_table_can_request_optional_cv_metadata_without_defaulting_to_it(ecat_module, cv_factory):
    low = cv_factory(name="50mVs")
    high = cv_factory(name="100mVs")
    low.ir_comp_resistance = 50.0
    high.ir_comp_resistance = 100.0
    low.temperature = 298
    high.temperature = 305
    low.electrode_area = 0.071
    high.electrode_area = 0.071

    default_table, _metadata = ecat_module.build_object_table(
        [low, high],
        {"print conditions": False},
    )
    assert "IR Comp Resistance" not in default_table.columns
    assert "Temperature" not in default_table.columns
    assert "Electrode Area" not in default_table.columns

    requested_table, _metadata = ecat_module.build_object_table(
        [low, high],
        {
            "columns": ["ir comp resistance", "temperature", "electrode_area"],
            "print conditions": False,
        },
    )
    assert requested_table["IR Comp Resistance"].tolist() == ["50", "100"]
    assert requested_table["Temperature"].tolist() == ["298", "305"]
    assert requested_table["Electrode Area"].tolist() == ["0.071", "0.071"]


def test_build_object_table_keeps_ir_compensation_out_of_default_columns(ecat_module, cv_factory):
    low = cv_factory(name="50mVs")
    high = cv_factory(name="100mVs")
    low.ir_comp_resistance = 40.0
    high.ir_comp_resistance = 80.0
    low.ir_uncomp_resistance = 10.0
    high.ir_uncomp_resistance = 20.0
    low.ir_comp_percent = 80.0
    high.ir_comp_percent = 80.0

    default_table, _metadata = ecat_module.build_object_table(
        [low, high],
        {"print conditions": False},
    )
    requested_table, _metadata = ecat_module.build_object_table(
        [low, high],
        {
            "columns": ["ir comp resistance", "ir uncomp resistance", "ir comp percent"],
            "print conditions": False,
        },
    )

    assert "IR Comp Resistance" not in default_table.columns
    assert "IR Uncomp Resistance" not in default_table.columns
    assert "IR Comp Percent" not in default_table.columns
    assert "IR Comp Resistance" in requested_table.columns
    assert "IR Uncomp Resistance" in requested_table.columns
    assert "IR Comp Percent" in requested_table.columns


def test_ca_object_table_display_headers_omit_unit_suffixes(ecat_module, blank_echem_factory):
    obj = blank_echem_factory(ecat_module.ca)
    obj.type = "Chronoamperometry"
    obj.data = pd.DataFrame({"Time": [0.0, 1.0], "Current": [0.0, 0.5]})
    obj.units = {"Time": "s", "Current": "A"}
    obj.num_x_cols = 1
    obj.init_E = -1.0
    obj.run_time = 60.0
    obj.sample_interval = 1.0

    table, _metadata = ecat_module.build_object_table(
        [obj],
        {
            "columns": ["Init Potential (V)", "Run Time (s)", "Sample Interval (s)"],
            "print conditions": False,
        },
    )

    assert "Applied Potential" in table.columns
    assert "Run Time" in table.columns
    assert "Sample Interval" in table.columns
    assert "Init Potential (V)" not in table.columns
    assert "Init Potential" not in table.columns
    assert "Run Time (s)" not in table.columns
    assert "Sample Interval (s)" not in table.columns


def test_ca_show_table_includes_basic_metadata(ecat_module, blank_echem_factory):
    obj = blank_echem_factory(ecat_module.ca)
    obj.type = "Chronoamperometry"
    obj.data = pd.DataFrame({"Time": [0.0, 1.0], "Current": [0.0, 0.5]})
    obj.units = {"Time": "s", "Current": "A"}
    obj.solvent = "MeCN"
    obj.gas = "CO2"
    obj.compounds = ["Fc"]
    obj.concentrations = ["1 mM"]

    table = ecat_module.show(obj, {"pretty print": False, "return": True})

    values = dict(zip(table["Metric"], table["Value"]))
    assert values["Solvent"] == "MeCN"
    assert values["Gas"] == "CO2"
    assert values["Compounds"] == ["Fc"]
    assert values["Concentrations"] == ["1 mM"]


def test_ca_show_current_stats_put_units_in_values_and_respect_sig_figs(
    ecat_module,
    blank_echem_factory,
):
    obj = blank_echem_factory(ecat_module.ca)
    obj.type = "Chronoamperometry"
    obj.data = pd.DataFrame(
        {
            "Time": [0.0, 1.0, 2.0],
            "Current": [1.23456e-6, 2.34567e-6, 3.45678e-6],
        }
    )
    obj.units = {"Time": "s", "Current": "A"}

    table = ecat_module.show(
        obj,
        {"pretty print": False, "return": True, "sig figs": 3},
    )
    values = dict(zip(table["Metric"], table["Value"]))

    assert "Min Current (A)" not in values
    assert "Max Current (A)" not in values
    assert "Avg Current (A)" not in values
    assert values["Min Current"] == "1.23 μA"
    assert values["Max Current"] == "3.46 μA"
    assert values["Avg Current"] == "2.35 μA"


def test_cp_show_table_includes_basic_metadata(ecat_module, blank_echem_factory):
    obj = blank_echem_factory(ecat_module.cp)
    obj.type = "Chronopotentiometry"
    obj.data = pd.DataFrame({"Time": [0.0, 1.0], "Potential": [0.0, -0.5]})
    obj.units = {"Time": "s", "Potential": "V"}
    obj.solvent = "MeCN"
    obj.gas = "N2"
    obj.compounds = ["TBAPF6"]
    obj.concentrations = ["100 mM"]

    table = ecat_module.show(obj, {"pretty print": False, "return": True})

    values = dict(zip(table["Metric"], table["Value"]))
    assert values["Solvent"] == "MeCN"
    assert values["Gas"] == "N2"
    assert values["Compounds"] == ["TBAPF6"]
    assert values["Concentrations"] == ["100 mM"]


def test_ca_current_stats_are_not_object_table_columns(ecat_module, blank_echem_factory):
    obj = blank_echem_factory(ecat_module.ca)
    obj.type = "Chronoamperometry"
    obj.data = pd.DataFrame({"Time": [0.0, 1.0], "Current": [0.0, 0.5]})
    obj.units = {"Time": "s", "Current": "A"}
    obj.num_x_cols = 1

    available = ecat_module.show_objects(
        [obj],
        {"columns": "available", "print": False},
    )

    assert "Min Current (A)" not in available
    assert "Max Current (A)" not in available
    assert "Avg Current (A)" not in available
    with pytest.raises(ValueError, match="Avg Current"):
        ecat_module.build_object_table(
            [obj],
            {"columns": ["Avg Current (A)"], "print conditions": False},
        )


def test_build_object_table_columns_all_shows_available_non_internal_columns(ecat_module, cv_factory):
    obj = cv_factory()

    table, _metadata = ecat_module.build_object_table(
        [obj],
        {"columns": "all", "print conditions": False},
    )

    assert "Name" in table.columns
    assert "Gas" in table.columns
    assert "Scan Rate" in table.columns
    assert "Reference Shift" not in table.columns


def test_build_object_table_accepts_column_display_and_underscore_aliases(ecat_module, cv_factory):
    obj = cv_factory()
    obj.reference_shift = 0.401
    obj.reference_mode = "manual"

    table, _metadata = ecat_module.build_object_table(
        [obj],
        {"columns": ["Scan Rate", "reference_shift"], "print conditions": False},
    )

    assert "Scan Rate" in table.columns
    assert "Reference Shift" in table.columns


def test_build_object_table_invalid_column_suggests_and_lists_available_columns(ecat_module, cv_factory):
    obj = cv_factory()

    with pytest.raises(ValueError) as excinfo:
        ecat_module.build_object_table(
            [obj],
            {"columns": ["scan_rate_v_per_s"], "print conditions": False},
        )

    message = str(excinfo.value)
    assert 'Unknown show_objects column: "scan_rate_v_per_s"' in message
    assert 'Did you mean: "scan rate"?' in message
    assert "Available columns:" in message
    assert "scan rate" in message
    assert "ir comp resistance" in message
