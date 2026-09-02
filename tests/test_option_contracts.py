from dataclasses import fields

import pandas as pd
import pytest

from ecat import options as option_models


OPTION_MODEL_CASES = [
    ("get_data", option_models.ImportOptions),
    ("trim", option_models.TrimOptions),
    ("plot", option_models.PlotOptions),
    ("multiplot", option_models.MultiplotOptions),
    ("multimultiplot", option_models.MultiMultiplotOptions),
    ("multi_scatterplot", option_models.MultiScatterplotOptions),
    ("cv.peak_potential", option_models.PeakPotentialOptions),
    ("cv.peak_current", option_models.PeakCurrentOptions),
    ("cv.peak_width", option_models.PeakWidthOptions),
    ("normalize", option_models.NormalizeOptions),
    ("normalize_current", option_models.NormalizationOptions),
    ("scale_current", option_models.ScaleCurrentOptions),
    ("fowa", option_models.FOWAOptions),
    ("plateau_current", option_models.PlateauCurrentOptions),
    ("fit_rate", option_models.FitRateOptions),
    ("fit_peak_potential", option_models.FitPeakPotentialOptions),
    ("sevcik_analysis", option_models.SevcikAnalysisOptions),
    ("fit_peak_current", option_models.FitPeakCurrentOptions),
    ("trumpet_analysis", option_models.TrumpetAnalysisOptions),
    ("nicholson", option_models.NicholsonOptions),
    ("tafel_analysis", option_models.TafelAnalysisOptions),
    ("filter", option_models.FilterOptions),
    ("sort_group", option_models.SortGroupOptions),
    ("group_summary", option_models.GroupSummaryOptions),
]


REQUIRED_OPTION_BASELINES = {
    option_models.TrimOptions: {"potential window": [-1.0, 1.0]},
}


def _public_field_names(cls):
    return {field.name for field in fields(cls) if not field.name.startswith("_")}


def _display_candidates(option_label):
    label = str(option_label)
    candidates = [label]
    if "(s)" in label:
        singular = label.replace("(s)", "")
        candidates.extend([singular, f"{singular}s"])
    return list(dict.fromkeys(candidates))


def _is_missing(value):
    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _sample_value(option_label, default):
    canonical = option_models._canonical_option_key(option_label)
    if canonical == "potential_window":
        return [-1.0, 1.0]
    if canonical == "overpotential_range":
        return [0.0, 1.0]
    if canonical == "custom_parser":
        return lambda result, settings=None: {}
    if canonical == "parser_settings":
        return {}
    if canonical == "plot_options":
        return {}
    if canonical == "colors":
        return []
    if canonical == "directional_arrows":
        return False
    if canonical == "scale_bar":
        return False
    if canonical == "references":
        return None
    if _is_missing(default):
        return None
    return default


@pytest.mark.parametrize("target, option_cls", OPTION_MODEL_CASES)
def test_described_option_model_fields_are_documented_for_public_function(
    ecat_module,
    target,
    option_cls,
):
    schema = ecat_module.describe_options(target, {"print": False, "return": True})
    documented = set(schema["Option"])

    missing = []
    for field in fields(option_cls):
        if field.name.startswith("_"):
            continue
        label = option_models._display_option_key(field.name, section=target)
        if label not in documented:
            missing.append(label)

    assert missing == []


def test_typed_options_use_neutral_resolved_dictionary_api(ecat_module):
    from ecat import options as option_module

    resolved = ecat_module.PlotOptions.from_options({"invert y": True})

    assert resolved.to_options_dict()["invert y axis"] is True
    assert not hasattr(resolved, "to_legacy_dict")
    assert hasattr(option_module, "resolve_import_options")
    assert not hasattr(option_module, "import_options_to_legacy_dict")


@pytest.mark.parametrize("target, option_cls", OPTION_MODEL_CASES)
def test_documented_public_options_are_accepted_by_their_option_model(
    ecat_module,
    target,
    option_cls,
):
    schema = ecat_module.describe_options(target, {"print": False, "return": True})
    valid_fields = _public_field_names(option_cls)
    baseline = dict(REQUIRED_OPTION_BASELINES.get(option_cls, {}))
    failures = {}

    for row in schema.to_dict("records"):
        option_label = row["Option"]
        value = _sample_value(option_label, row.get("Default"))
        accepted = False
        errors = []
        for candidate in _display_candidates(option_label):
            canonical = option_models._canonical_option_key(candidate)
            if canonical not in valid_fields:
                continue
            try:
                option_cls.from_options({**baseline, candidate: value})
            except option_models.OptionError as exc:
                errors.append(f"{candidate}: {exc}")
            else:
                accepted = True
                break
        if not accepted:
            failures[option_label] = errors or ["no documented spelling maps to a public field"]

    assert failures == {}


def test_option_models_accept_empty_options_except_declared_required_inputs():
    failures = {}
    for _target, option_cls in OPTION_MODEL_CASES:
        if option_cls in REQUIRED_OPTION_BASELINES:
            continue
        try:
            option_cls.from_options({})
        except Exception as exc:  # noqa: BLE001 - contract should report all failures at once
            failures[option_cls.__name__] = repr(exc)

    assert failures == {}


@pytest.mark.parametrize(
    "alias, qualified",
    [
        ("cv_data", "simulation.cv_data"),
        ("simulate_cv", "simulation.simulate_cv"),
        ("fit_cv", "simulation.fit_cv"),
    ],
)
def test_describe_options_accepts_bare_simulation_function_aliases(ecat_module, alias, qualified):
    bare = ecat_module.describe_options(alias, {"print": False, "return": True})
    qualified_table = ecat_module.describe_options(qualified, {"print": False, "return": True})

    assert list(bare["Option"]) == list(qualified_table["Option"])


@pytest.mark.parametrize("target, option_cls", OPTION_MODEL_CASES)
def test_common_package_options_smoke_where_supported(target, option_cls):
    baseline = dict(REQUIRED_OPTION_BASELINES.get(option_cls, {}))
    supported = _public_field_names(option_cls)
    common_values = {
        "print": False,
        "plot": False,
        "pretty print": False,
        "sig figs": 3,
    }

    for option_name, value in common_values.items():
        if option_models._canonical_option_key(option_name) in supported:
            option_cls.from_options({**baseline, option_name: value})


def test_plot_options_use_invert_y_axis_not_y_flip(ecat_module):
    options = ecat_module.PlotOptions.from_options({"invert y axis": True}).to_options_dict()

    assert options["invert y axis"] is True
    assert "y flip" not in options
    assert "invert y" not in options

    alias_options = ecat_module.PlotOptions.from_options({"invert y": True}).to_options_dict()
    assert alias_options["invert y axis"] is True
    assert "invert y" not in alias_options

    with pytest.raises(ecat_module.OptionError, match="invert y axis"):
        ecat_module.PlotOptions.from_options({"y flip": True})


def test_ca_axis_inversion_options_are_optional_overrides(ecat_module):
    options = ecat_module.PlotOptions.from_options(
        {
            "invert y axis": True,
            "invert current axis": False,
            "invert charge axis": True,
        }
    ).to_options_dict()

    assert options["invert y axis"] is True
    assert options["invert current axis"] is False
    assert options["invert charge axis"] is True

    defaults = ecat_module.PlotOptions.from_options({}).to_options_dict()
    assert defaults["invert current axis"] is None
    assert defaults["invert charge axis"] is None


def test_ca_plot_describes_specific_axis_inversion_controls(ecat_module):
    table = ecat_module.describe_options(
        "ca.plot",
        {"print": False, "return": True},
    ).set_index("Option")

    assert table.loc["invert current axis", "Default"] is None
    assert table.loc["invert charge axis", "Default"] is None
    assert "inherit" in table.loc["invert current axis", "Description"].lower()
    assert "inherit" in table.loc["invert charge axis", "Description"].lower()


def test_fowa_and_plateau_accept_n_cat_n_turn_aliases(ecat_module):
    fowa = ecat_module.FOWAOptions.from_options({"n_cat": 2, "n_turn": 4}).to_options_dict()
    plateau = ecat_module.PlateauCurrentOptions.from_options({"n_cat": 3, "n_turn": 6}).to_options_dict()

    assert fowa["catalyst electrons"] == 2
    assert fowa["turnover electrons"] == 4
    assert plateau["catalyst electrons"] == 3
    assert plateau["turnover electrons"] == 6


def test_import_options_use_invert_current_for_data_sign(ecat_module):
    options = ecat_module.ImportOptions.from_options({"invert current": True}).to_options_dict()

    assert options["invert current"] is True
    assert "invert y axis" not in options


def test_peak_width_options_accept_common_public_keys(ecat_module):
    opts = ecat_module.PeakWidthOptions.from_options(
        {
            "plot": False,
            "print": False,
            "pretty print": False,
            "sig figs": 5,
            "level": 0.5,
            "guess potential": -0.4,
            "segment": 1,
            "peak kind": "max",
            "tangent range": "auto",
        }
    )

    data = opts.to_options_dict()
    assert data["level"] == 0.5
    assert data["guess potential"] == -0.4
    assert data["peak kind"] == "max"


def test_peak_width_options_reject_side_mode_and_interpolate(ecat_module):
    with pytest.raises(ecat_module.OptionError, match="Unknown option 'side'"):
        ecat_module.PeakWidthOptions.from_options({"side": "both"})

    with pytest.raises(ecat_module.OptionError, match="Unknown option 'mode'"):
        ecat_module.PeakWidthOptions.from_options({"mode": "raw"})

    with pytest.raises(ecat_module.OptionError, match="Unknown option 'interpolate'"):
        ecat_module.PeakWidthOptions.from_options({"interpolate": False})


def test_multiplot_rejects_removed_stacking_option(ecat_module):
    with pytest.raises(ecat_module.OptionError, match="Unknown option 'stacking'"):
        ecat_module.MultiplotOptions.from_options({"stacking": True})

    described = ecat_module.describe_options(
        "multiplot",
        {"print": False, "return": True},
    )
    assert "stacking" not in described["Option"].tolist()


@pytest.mark.parametrize("level", [0, -0.1, 1.0, 1.2])
def test_peak_width_options_require_fractional_level(ecat_module, level):
    with pytest.raises(
        ecat_module.OptionError,
        match="'level' must be greater than 0 and less than 1",
    ):
        ecat_module.PeakWidthOptions.from_options({"level": level})


@pytest.mark.parametrize(
    "option_cls, option_name, value, expected_key, expected_value",
    [
        (option_models.PlotOptions, "plot convention", "iupac", "plot convention", "IUPAC"),
        (option_models.PlotOptions, "Plot Convention", "us", "plot convention", "US"),
        (option_models.PlotOptions, "plot_convention", "IUPAC", "plot convention", "IUPAC"),
        (option_models.MultiplotOptions, "colorbar tick labels", "All", "colorbar tick labels", "all"),
        (option_models.FilterOptions, "logic", "and", "logic", "AND"),
        (option_models.FilterOptions, "mode", "EXCLUDE", "mode", "exclude"),
        (option_models.ImportOptions, "reference mode", None, "reference mode", "none"),
        (option_models.ImportOptions, "reference mode", "None", "reference mode", "none"),
        (option_models.ImportOptions, "reference_mode", False, "reference mode", "none"),
        (option_models.ScaleCurrentOptions, "reference mode", "BOTH", "reference mode", "both"),
    ],
)
def test_choice_values_are_canonicalized_case_insensitively(
    option_cls,
    option_name,
    value,
    expected_key,
    expected_value,
):
    options = option_cls.from_options(option_name and {option_name: value}).to_options_dict()

    assert options[expected_key] == expected_value


@pytest.mark.parametrize(
    "option_cls, options, message",
    [
        (option_models.PeakCurrentOptions, {"segment": 1, "segments": [1, 2]}, "segment.*segments"),
        (
            option_models.PeakCurrentOptions,
            {"guess potential": -0.1, "exact potential": -0.1},
            "exact potential.*guess potential|guess potential.*exact potential",
        ),
        (
            option_models.NormalizationOptions,
            {"ip0": 1e-6, "reference cv": object()},
            "ip0/reference source",
        ),
        (
            option_models.ScaleCurrentOptions,
            {"scale": 2.0, "reference cv": object()},
            "scale/reference source",
        ),
        (
            option_models.FOWAOptions,
            {"ip0": 1e-6, "non-catalytic current": 2e-6},
            "ip0.*non-catalytic current",
        ),
        (
            option_models.FOWAOptions,
            {"non-catalytic cv": object(), "non-catalytic cvs": [object()]},
            "non-catalytic cv.*non-catalytic cvs",
        ),
        (option_models.PlateauCurrentOptions, {"ilim": 1e-6, "ic": 2e-6}, "ilim.*ic"),
    ],
)
def test_mutually_exclusive_option_combinations_reject_early(option_cls, options, message):
    with pytest.raises(option_models.OptionError, match=message):
        option_cls.from_options(options)


def test_explicit_segments_none_does_not_erase_default_segment():
    trumpet = option_models.TrumpetAnalysisOptions.from_options({"segments": None})
    nicholson = option_models.NicholsonOptions.from_options({"segments": None})

    assert trumpet.segment == 1
    assert trumpet.segments is None
    assert nicholson.segment == 1
    assert nicholson.segments is None


def test_nicholson_accepts_documented_explicit_segment_pair():
    options = option_models.NicholsonOptions.from_options({"segments": [1, 2]})
    resolved = options.to_options_dict()

    assert options.segment is None
    assert options.segments == [1, 2]
    assert resolved["segment"] is None
    assert resolved["segments"] == [1, 2]


@pytest.mark.parametrize(
    "option_cls",
    [
        option_models.NormalizationOptions,
        option_models.ScaleCurrentOptions,
        option_models.SevcikAnalysisOptions,
        option_models.FitPeakCurrentOptions,
    ],
)
def test_option_forwarding_helpers_preserve_selector_contracts(option_cls):
    options = option_cls.from_options({"guess potential": -0.25})
    forwarded = options.for_peak_current()
    forwarded_options = forwarded.to_options_dict()

    assert forwarded_options["guess potential"] == pytest.approx(-0.25)
    assert forwarded_options["exact potential"] is None
    assert forwarded_options["segment"] == options.segment


@pytest.mark.parametrize(
    "option_cls",
    [
        option_models.PeakWidthOptions,
        option_models.SevcikAnalysisOptions,
        option_models.FitPeakCurrentOptions,
    ],
)
def test_peak_current_subclass_forwarding_preserves_peak_selection_controls(option_cls):
    options = option_cls.from_options(
        {
            "peak kind": "max",
            "peak fallback": None,
            "plot peak potential": False,
        }
    )

    forwarded = options.for_peak_current().to_options_dict()

    assert forwarded["peak kind"] == "max"
    assert forwarded["peak fallback"] is None
    assert forwarded["plot peak potential"] is False


def test_fowa_peak_current_forwarding_preserves_supported_fallback_control():
    options = option_models.FOWAOptions.from_options({"peak fallback": None})

    forwarded = options.for_peak_current().to_options_dict()

    assert forwarded["peak fallback"] is None


@pytest.mark.parametrize(
    "option_cls",
    [
        option_models.PeakCurrentOptions,
        option_models.PeakWidthOptions,
        option_models.FOWAOptions,
        option_models.PlateauCurrentOptions,
        option_models.SevcikAnalysisOptions,
        option_models.ReversibilityAnalysisOptions,
        option_models.SurfaceCoverageAnalysisOptions,
        option_models.FitPeakCurrentOptions,
        option_models.NicholsonOptions,
    ],
)
def test_peak_current_based_analyses_default_to_highest_current_fallback(option_cls):
    assert option_cls.from_options({}).peak_fallback == "highest current"


def test_trumpet_analysis_does_not_expose_peak_current_fallback():
    options = option_models.TrumpetAnalysisOptions.from_options({})
    resolved = options.to_options_dict()

    assert not hasattr(options, "peak_fallback")
    assert "peak fallback" not in resolved
    with pytest.raises(option_models.OptionError, match="Unknown option 'peak fallback'"):
        option_models.TrumpetAnalysisOptions.from_options({"peak fallback": None})


def test_fit_peak_potential_accepts_documented_peak_tracking_modes():
    default_options = option_models.FitPeakPotentialOptions.from_options({})
    within_cv = option_models.FitPeakPotentialOptions.from_options({"peak tracking": "within CV"})
    series = option_models.FitPeakPotentialOptions.from_options({"peak tracking": "series"})
    strict = option_models.FitPeakPotentialOptions.from_options({"peak tracking": "series-strict"})

    assert default_options.peak_tracking is None
    assert within_cv.peak_tracking == "within cv"
    assert series.peak_tracking == "series"
    assert strict.peak_tracking == "series strict"

    with pytest.raises(option_models.OptionError, match="'peak tracking' must be"):
        option_models.FitPeakPotentialOptions.from_options({"peak tracking": "previous cv"})


def test_option_projection_preserves_aliases_and_rejects_duplicate_spellings():
    projected = option_models._project_options(
        option_models.PeakCurrentOptions,
        {
            "Peak Kind": "max",
            "Peak Fallback": None,
            "sig_fig": 6,
        },
    )

    assert projected.peak_kind == "max"
    assert projected.peak_fallback is None
    assert projected.sig_figs == 6

    with pytest.raises(option_models.OptionError, match="both resolve to 'sig figs'"):
        option_models._project_options(
            option_models.PeakCurrentOptions,
            {"sig figs": 4, "sig_fig": 6},
        )


@pytest.mark.parametrize(
    "option_cls, intentional_overrides",
    [
        (option_models.PeakWidthOptions, set()),
        (option_models.NormalizationOptions, {"print"}),
        (option_models.ScaleCurrentOptions, {"print"}),
        (option_models.FOWAOptions, {"plot", "print"}),
        (option_models.SevcikAnalysisOptions, set()),
        (option_models.FitPeakCurrentOptions, set()),
    ],
)
def test_peak_current_forwarding_preserves_every_shared_field(
    option_cls,
    intentional_overrides,
):
    sample_values = {
        "plot": True,
        "print": True,
        "pretty_print": False,
        "plot_all": True,
        "print_all": True,
        "x_axis": "Potential",
        "y_axis": "Current",
        "x_unit": "mV",
        "y_unit": "uA",
        "xlabel": "Potential test",
        "ylabel": "Current test",
        "new_plot": True,
        "plot_cv": False,
        "derivative": 1,
        "plot_segment": 2,
        "plot_segments": None,
        "segment": 2,
        "segments": None,
        "noise_window": 7,
        "noise_polyorder": 2,
        "sig_figs": 6,
        "peak_prominence": 1e-7,
        "peak_kind": "max",
        "guess_potential": -0.25,
        "exact_potential": None,
        "troubleshoot": True,
        "internal_call": True,
        "offset": 0.2,
        "tangent_range": [0.1, 0.2],
        "tangent_min_points": 5,
        "tangent_potential": -0.4,
        "percent_threshold": 0.2,
        "plot_peak_potential": False,
        "peak_fallback": None,
    }
    source_fields = {field.name for field in fields(option_cls)}
    source = option_cls.from_options(
        {
            key: value
            for key, value in sample_values.items()
            if key in source_fields
        }
    )
    forwarded = source.for_peak_current()

    for field in fields(option_models.PeakCurrentOptions):
        if field.name not in source_fields or field.name in intentional_overrides:
            continue
        assert getattr(forwarded, field.name) == getattr(source, field.name), field.name


def test_peak_current_forwarding_to_peak_potential_drops_peak_current_only_options():
    options = option_models.PeakCurrentOptions.from_options(
        {"guess potential": -0.25, "tangent range": [0.1, 0.2]}
    )

    forwarded = options.for_peak_potential().to_options_dict()

    assert forwarded["guess potential"] == pytest.approx(-0.25)
    assert "tangent range" not in forwarded


def test_simulation_option_normalizer_supports_spaces_and_underscores():
    from ecat import simulation as sim

    normalized = sim._normalize_options(
        {
            "plot_all": True,
            "post_correction": "vertical shift",
            "residual_normalization": "max_abs_measured",
            "current_sign": "flip",
            "print_params": True,
            "trim_mode": "expand",
        }
    )

    assert normalized["plot all"] is True
    assert normalized["plot_all"] is True
    assert normalized["post correction"] == "vertical shift"
    assert normalized["post_correction"] == "vertical shift"
    assert normalized["residual normalization"] == "max_abs_measured"
    assert normalized["current sign"] == "flip"
    assert normalized["print params"] is True
    assert normalized["trim mode"] == "expand"
    assert sim._normalize_post_correction_mode(normalized) == "offset"
    assert sim._normalize_residual_normalization(normalized) == "max_abs_measured"
