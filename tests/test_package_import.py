import re
from pathlib import Path


def test_package_import_exposes_public_api():
    import ecat

    assert ecat.__version__ == "0.1.0b2"
    expected_public = {
        "echem",
        "cv",
        "ca",
        "cp",
        "dpv",
        "get_data",
        "get_CVs",
        "get_CVs_from_excel",
        "create_cv_objects_from_excel",
        "multiplot",
        "multimultiplot",
        "multi_scatterplot",
        "filter",
        "sort",
        "group",
        "sort_and_group",
        "group_summary",
        "show",
        "show_groups",
        "show_objects",
        "save_data",
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
        "normalize",
        "normalize_current",
        "scale_current",
        "plotting_style",
        "open_app",
        "describe_options",
        "get_defaults",
        "set_defaults",
        "reset_defaults",
        "load_defaults",
        "simulation",
    }

    assert expected_public <= set(ecat.__all__)
    for name in expected_public:
        assert hasattr(ecat, name), name


def test_package_import_hides_obvious_internals_from_public_api():
    import ecat

    hidden = {
        "np",
        "pd",
        "plt",
        "mpl",
        "curve_fit",
        "AutoMinorLocator",
        "Path",
        "datetime",
        "animate",
        "save_animation",
        "default_plotting",
        "use_ecat_plot_style",
        "scale_bar",
        "display_object_table",
        "print_groups",
        "print_object_list",
        "get_sort_group_dict",
        "FOWA",
        "Tafel",
        "Sevcik",
        "Nicholson",
        "peak_current_fit",
        "peak_potential_fit",
    }

    assert hidden.isdisjoint(ecat.__all__)
    for name in hidden:
        assert not hasattr(ecat, name), name


def test_package_help_docstring_points_to_discovery_tools():
    import ecat

    doc = ecat.__doc__
    assert "Common starting points" in doc
    assert "describe_options()" in doc
    assert "ecat.simulation" in doc


def test_package_metadata_version_matches_import_version():
    import ecat

    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    text = pyproject.read_text(encoding="utf-8")
    version = re.search(r'^version = "([^"]+)"$', text, flags=re.MULTILINE).group(1)

    assert version == ecat.__version__
