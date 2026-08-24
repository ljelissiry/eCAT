"""eCAT: electroChemical Analysis Tools.

Common starting points:
    echem.from_file(path)
    get_data({"folder path": path})
    multiplot(objects)
    animate(obj_or_list)
    open_app()
    describe_options()

Core objects:
    echem, cv, ca, cp, dpv

Common analyses:
    fowa(cvs)
    fit_peak_current(cvs)
    tafel_analysis(cv_or_list, TOF_max, E_thermo, E_redox)

Plot styling:
    plotting_style("notebook")
    plotting_style("publication")
    plotting_style("saveant")
    plotting_style(False)

Simulation helpers live under:
    ecat.simulation
"""

from . import (
    analysis_batch,
    analysis_cv,
    collection,
    export,
    io,
    metadata,
    objects,
    parsers,
    plotting,
    reference,
    reversibility,
    results,
    simulation,
    utils,
)
from ._version import __version__
from ._file_formats import UnsupportedFileFormatError
from .analysis_batch import (
    fit_model,
    fit_peak_current,
    fit_peak_potential,
    fit_rate,
    fowa,
    nicholson_analysis,
    plateau_current,
    sevcik_analysis,
    tafel_analysis,
    trumpet_analysis,
)
from .reversibility import reversibility_analysis, surface_coverage_analysis
from .analysis_cv import normalize, normalize_current, scale_current, trim
from .animation import animate
from .app import open_app
from .collection import (
    filter,
    get_available_filter_values,
    group,
    group_summary,
    sort,
    sort_and_group,
)
from .utils import describe_options
from .export import save_data
from .io import get_data, get_data_from_excel, parse_file
from .objects import CVAnalysisResult, ChronoAnalysisResult, ca, cp, cv, dpv, echem
from .plotting import (
    ScatterFitResult,
    multimultiplot,
    multiplot,
    multi_scatterplot,
    plotting_style,
    show,
    show_groups,
    show_objects,
)
from .results import AnalysisResult
from .parsers import ParseResult
from .options import (
    CVFilterOptions,
    FilterOptions,
    FitPeakCurrentOptions,
    FitPeakPotentialOptions,
    FitRateOptions,
    FOWAOptions,
    GroupSummaryOptions,
    ImportOptions,
    MultiMultiplotOptions,
    MultiScatterplotOptions,
    MultiplotOptions,
    NicholsonOptions,
    NormalizationOptions,
    NormalizeOptions,
    OptionError,
    PeakCurrentOptions,
    PeakPotentialOptions,
    PeakWidthOptions,
    PlateauCurrentOptions,
    PlotOptions,
    ScaleCurrentOptions,
    SevcikAnalysisOptions,
    ReversibilityAnalysisOptions,
    SurfaceCoverageAnalysisOptions,
    SortGroupOptions,
    TafelAnalysisOptions,
    TrimOptions,
    TrumpetAnalysisOptions,
    get_defaults,
    load_defaults,
    reset_defaults,
    set_defaults,
)

__all__ = [
    "__version__",
    "UnsupportedFileFormatError",
    "simulation",
    "AnalysisResult",
    "ParseResult",
    "CVAnalysisResult",
    "ChronoAnalysisResult",
    "CVFilterOptions",
    "ScatterFitResult",
    "echem",
    "cv",
    "ca",
    "cp",
    "dpv",
    "get_data",
    "parse_file",
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
    "get_available_filter_values",
    "show",
    "show_groups",
    "show_objects",
    "save_data",
    "open_app",
    "fowa",
    "sevcik_analysis",
    "reversibility_analysis",
    "surface_coverage_analysis",
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
    "trim",
    "plotting_style",
    "describe_options",
    "get_defaults",
    "set_defaults",
    "reset_defaults",
    "load_defaults",
    "OptionError",
    "ImportOptions",
    "TrimOptions",
    "PlotOptions",
    "MultiplotOptions",
    "MultiMultiplotOptions",
    "MultiScatterplotOptions",
    "PeakPotentialOptions",
    "PeakCurrentOptions",
    "PeakWidthOptions",
    "NormalizeOptions",
    "NormalizationOptions",
    "ScaleCurrentOptions",
    "FOWAOptions",
    "PlateauCurrentOptions",
    "FitRateOptions",
    "FitPeakPotentialOptions",
    "SevcikAnalysisOptions",
    "ReversibilityAnalysisOptions",
    "SurfaceCoverageAnalysisOptions",
    "FitPeakCurrentOptions",
    "TrumpetAnalysisOptions",
    "NicholsonOptions",
    "TafelAnalysisOptions",
    "FilterOptions",
    "SortGroupOptions",
    "GroupSummaryOptions",
]
