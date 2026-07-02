import inspect
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import ecat as e


PUBLIC_FUNCTION_OPTIONS = {
    "describe_options": 'describe_options("',
    "set_defaults": 'describe_options("',
    "get_defaults": 'describe_options("',
    "reset_defaults": 'describe_options("',
    "load_defaults": 'describe_options("',
    "plotting_style": 'describe_options("',
    "animate": 'describe_options("animate")',
    "get_data": 'describe_options("get_data")',
    "get_data_from_excel": 'describe_options("get_data")',
    "multiplot": 'describe_options("multiplot")',
    "multimultiplot": 'describe_options("multimultiplot")',
    "multi_scatterplot": 'describe_options("multi_scatterplot")',
    "fowa": 'describe_options("fowa")',
    "normalize_current": 'describe_options("normalize_current")',
    "scale_current": 'describe_options("scale_current")',
    "fit_peak_current": 'describe_options("fit_peak_current")',
    "fit_peak_potential": 'describe_options("fit_peak_potential")',
    "fit_rate": 'describe_options("fit_rate")',
    "sevcik_analysis": 'describe_options("sevcik_analysis")',
    "trumpet_analysis": 'describe_options("trumpet_analysis")',
    "nicholson_analysis": 'describe_options("nicholson")',
    "tafel_analysis": 'describe_options("tafel_analysis")',
    "plateau_current": 'describe_options("plateau_current")',
    "filter": 'describe_options("filter")',
    "sort": 'describe_options("sort_group")',
    "group": 'describe_options("sort_group")',
    "sort_and_group": 'describe_options("sort_group")',
    "group_summary": 'describe_options("group_summary")',
}


PUBLIC_METHOD_OPTIONS = {
    e.echem.from_file: 'describe_options("get_data")',
    e.echem.x: 'describe_options("plot")',
    e.echem.y: 'describe_options("plot")',
    e.echem.xy: 'describe_options("plot")',
    e.echem.plot: 'describe_options("plot")',
    e.echem.animate: 'describe_options("animate")',
    e.cv.x: 'describe_options("plot")',
    e.cv.y: 'describe_options("plot")',
    e.cv.xy: 'describe_options("plot")',
    e.cv.plot: 'describe_options("plot")',
    e.cv.peak_potential: 'describe_options("cv.peak_potential")',
    e.cv.peak_current: 'describe_options("cv.peak_current")',
    e.cv.plateau_current: 'describe_options("cv.plateau_current")',
    e.cv.half_peak_potential: 'describe_options("cv.half_peak_potential")',
    e.cv.half_wave_potential: 'describe_options("cv.half_wave_potential")',
    e.cv.normalize: 'describe_options("cv.normalize")',
    e.cv.normalize_current: 'describe_options("cv.normalize_current")',
    e.cv.scale_current: 'describe_options("cv.scale_current")',
    e.dpv.peak_potential: 'describe_options("dpv.peak_potential")',
    e.cp.get_cycles: 'describe_options("cp.get_cycles")',
    e.cp.plot_cycles: 'describe_options("cp.plot_cycles")',
    e.ca.charge: 'describe_options("ca.charge")',
    e.ca.plot: 'describe_options("plot")',
    e.ca.time_at_charge: 'describe_options("ca.time_at_charge")',
}


def _doc(obj):
    return inspect.getdoc(obj) or ""


def test_curated_public_functions_have_numpy_style_docstrings():
    for name, options_reference in PUBLIC_FUNCTION_OPTIONS.items():
        doc = _doc(getattr(e, name))
        assert doc, name
        assert "Parameters\n----------" in doc, name
        assert "Returns\n-------" in doc, name
        assert "Examples\n--------" in doc, name
        assert options_reference in doc, name


def test_curated_public_methods_have_options_references():
    for method, options_reference in PUBLIC_METHOD_OPTIONS.items():
        doc = _doc(method)
        assert doc, method
        assert "Parameters\n----------" in doc, method
        assert options_reference in doc, method


def test_core_public_classes_have_docstrings():
    for cls in [e.echem, e.cv, e.dpv, e.cp, e.ca]:
        doc = _doc(cls)
        assert doc, cls.__name__
        assert "Parameters\n----------" in doc, cls.__name__
        assert "Examples\n--------" in doc, cls.__name__


def test_new_public_docstrings_use_canonical_fit_peak_potential_name():
    public_docs = [
        _doc(getattr(e, name))
        for name in PUBLIC_FUNCTION_OPTIONS
    ]
    public_docs.extend(_doc(method) for method in PUBLIC_METHOD_OPTIONS)
    assert "Ep_fit" not in "\n".join(public_docs)
    assert "peak_potential_fit" not in "\n".join(public_docs)
    assert "peak_current_fit" not in "\n".join(public_docs)
    assert "fit_trumpet" not in "\n".join(public_docs)
    assert "fit_sevcik" not in "\n".join(public_docs)
    assert "FOWA(" not in "\n".join(public_docs)


def test_beta_api_removes_legacy_analysis_aliases():
    for name in [
        "rate_fit",
        "Ep_fit",
        "Nicholson",
        "PlateauCurrent",
        "Tafel",
        "fit_ip",
        "Sevcik",
        "FOWA",
        "peak_potential_fit",
        "peak_current_fit",
        "fit_trumpet",
        "fit_sevcik",
        "EpFitOptions",
        "FitIpOptions",
        "FitTrumpetOptions",
        "FitSevcikOptions",
        "TafelOptions",
        "TrumpetOptions",
        "SevcikOptions",
        "default_plotting",
        "use_ecat_plot_style",
        "scale_bar",
    ]:
        assert not hasattr(e, name)
