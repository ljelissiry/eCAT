import inspect
import re
from pathlib import Path


def test_package_import_exposes_public_api():
    import ecat

    assert re.fullmatch(r"\d+\.\d+\.\d+(?:[a-z]+\d+)?", ecat.__version__)
    expected_public = {
        "echem",
        "cv",
        "ca",
        "cp",
        "dpv",
        "get_data",
        "get_data_from_excel",
        "multiplot",
        "multimultiplot",
        "multi_scatterplot",
        "animate",
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


def test_public_options_defaults_are_not_mutable_dicts():
    import ecat

    public_callables = [
        ecat.get_data,
        ecat.get_data_from_excel,
        ecat.save_data,
        ecat.multiplot,
        ecat.multimultiplot,
        ecat.multi_scatterplot,
        ecat.fowa,
        ecat.sevcik_analysis,
        ecat.trumpet_analysis,
        ecat.nicholson_analysis,
        ecat.tafel_analysis,
        ecat.fit_rate,
        ecat.plateau_current,
        ecat.fit_peak_potential,
        ecat.fit_peak_current,
        ecat.normalize_current,
        ecat.scale_current,
        ecat.echem.x,
        ecat.echem.y,
        ecat.echem.xy,
        ecat.echem.plot,
        ecat.cv.x,
        ecat.cv.y,
        ecat.cv.xy,
        ecat.cv.plot,
        ecat.cv.normalize,
        ecat.cv.normalize_current,
        ecat.cv.scale_current,
        ecat.cv.current_at_potential,
        ecat.cv.peak_potential,
        ecat.cv.peak_current,
        ecat.cv.plateau_current,
        ecat.cv.half_peak_potential,
        ecat.cv.peak_info,
        ecat.cv.half_wave_potential,
        ecat.cv.wave_info,
        ecat.dpv.peak_potential,
        ecat.dpv.fit_overlapping_peaks,
        ecat.cp.get_cycles,
        ecat.cp.cycle_info,
        ecat.cp.plot_cycles,
        ecat.cp.cycling_plot,
        ecat.ca.charge,
        ecat.ca.plot,
        ecat.ca.time_at_charge,
    ]

    for func in public_callables:
        signature = inspect.signature(func)
        for parameter in signature.parameters.values():
            assert parameter.default != {}, f"{func.__qualname__}.{parameter.name}"


def test_legacy_public_looking_methods_are_removed():
    import ecat
    from ecat import utils

    assert not hasattr(utils, "animate")
    assert not hasattr(utils, "save_animation")
    assert not hasattr(ecat.cv, "peak_potential_old")
    assert not hasattr(ecat.cp, "cycling_plot_old")


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
    project_block = text.split("[project]", 1)[1].split("\n[", 1)[0]

    assert not re.search(r"(?m)^version\s*=", project_block)
    assert re.search(r'(?m)^dynamic\s*=\s*\["version"\]$', project_block)
    assert 'version = { attr = "ecat._version.__version__" }' in text
    assert re.fullmatch(r"\d+\.\d+\.\d+(?:[a-z]+\d+)?", ecat.__version__)
