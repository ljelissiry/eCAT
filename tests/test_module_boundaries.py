import importlib


def test_split_module_boundaries_preserve_public_object_identity():
    import ecat

    expected = {
        "ecat.objects": [
            "ChronoAnalysisResult",
            "CVAnalysisResult",
            "echem",
            "cv",
            "ca",
            "cp",
            "dpv",
        ],
        "ecat.io": [
            "get_data",
            "get_CVs",
            "get_data_from_excel",
        ],
        "ecat.plotting": [
            "ScatterFitResult",
            "multiplot",
            "multimultiplot",
            "multi_scatterplot",
            "plotting_style",
            "show",
            "show_groups",
            "show_objects",
        ],
        "ecat.analysis_cv": ["normalize", "normalize_current", "scale_current"],
        "ecat.analysis_batch": [
            "fowa",
            "sevcik_analysis",
            "trumpet_analysis",
            "nicholson_analysis",
            "tafel_analysis",
            "fit_model",
            "fit_rate",
            "plateau_current",
            "fit_peak_potential",
            "fit_peak_current",
        ],
        "ecat.export": ["save_data"],
        "ecat.metadata": [
            "concentration_to_float",
            "format_chemical_formulas",
            "get_file_times",
            "parse_concentration_value_and_unit",
        ],
        "ecat.reference": [
            "midpoint_potential",
            "find_reference_midpoint_from_cv",
            "normalize_legacy_reference_options",
            "canonical_reference_label",
            "resolve_reference_options",
        ],
        "ecat.results": [
            "AnalysisResult",
            "analysis_result_from_table",
        ],
    }

    for module_name, names in expected.items():
        module = importlib.import_module(module_name)
        assert module.__all__ == names
        for name in names:
            if hasattr(ecat, name):
                assert getattr(ecat, name) is getattr(module, name)


def test_parser_module_exposes_parser_helpers_used_by_implementation():
    from ecat import parsers

    expected = [
        "ParseResult",
        "parse_ch_timestamp",
        "parse_duration_seconds",
        "parse_quiet_time_from_lines",
        "exp_type_short",
    ]

    assert parsers.__all__ == expected


def test_metadata_module_owns_concentration_parser_used_by_implementation():
    from ecat import metadata

    assert metadata.parse_concentration_value_and_unit("250uM") == (0.00025, "M")


def test_plotting_module_owns_plot_style_helpers_used_by_implementation():
    from ecat import plotting

    assert plotting.plotting_style.__module__ == "ecat._plot_style"
    assert plotting._active_plot_style_value.__module__ == "ecat._plot_style"


def test_plotting_module_owns_plot_helper_functions_used_by_implementation():
    from ecat import plotting

    expected_private_helpers = [
        "_normalize_scale_bar_options",
        "_scale_bar_position",
        "_add_scale_bar",
        "_apply_ecat_axis_style",
        "_format_reference_label_mathtext",
        "_apply_plot_titles",
    ]

    for name in expected_private_helpers:
        assert getattr(plotting, name).__module__ == "ecat._plot_helpers"


def test_export_module_owns_save_data_used_by_implementation():
    from ecat import export

    assert export.save_data.__module__ == "ecat.export"


def test_io_module_owns_cv_specific_loading_helpers_used_by_implementation():
    from ecat import io

    expected = [
        "get_CVs",
        "get_data_from_excel",
    ]

    for name in expected:
        assert getattr(io, name).__module__ == "ecat.io"


def test_split_modules_own_remaining_large_public_bodies():
    from ecat import analysis_batch
    from ecat import analysis_cv
    from ecat import objects
    from ecat import plotting

    expected_modules = {
        objects: ["echem", "cv", "ca", "cp", "dpv"],
        plotting: ["multiplot", "multimultiplot", "multi_scatterplot", "show", "show_groups", "show_objects"],
        analysis_cv: ["normalize", "normalize_current", "scale_current"],
        analysis_batch: [
            "fowa",
            "sevcik_analysis",
            "trumpet_analysis",
            "nicholson_analysis",
            "tafel_analysis",
            "fit_rate",
            "plateau_current",
            "fit_peak_potential",
            "fit_peak_current",
        ],
    }

    for module, names in expected_modules.items():
        for name in names:
            assert getattr(module, name).__module__ == module.__name__


def test_historical_core_facade_is_not_exposed_before_beta():
    assert importlib.util.find_spec("ecat.core") is None


def test_core_impl_temporary_module_is_removed_before_beta():
    assert importlib.util.find_spec("ecat._core_impl") is None


def test_package_imports_do_not_use_internal_module_wiring():
    import ecat

    assert not hasattr(ecat, "_wire_internal_modules")
    assert not hasattr(ecat, "_INTERNAL_MODULES")
