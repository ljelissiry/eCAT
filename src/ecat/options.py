"""Validated option models for eCAT notebook-facing APIs."""

from __future__ import annotations

from dataclasses import MISSING, dataclass, field, fields
from pathlib import Path
from importlib import resources
import difflib
import re
import types
import typing
from copy import deepcopy

try:
    from IPython.display import display
except Exception:  # pragma: no cover - only used outside notebook/IPython
    display = None

try:
    import tomllib
    from tomllib import TOMLDecodeError
except ModuleNotFoundError:  # pragma: no cover - exercised on Python 3.10
    tomllib = None
    TOMLDecodeError = ValueError


_MISSING = object()
_PACKAGE_DEFAULTS = None
_USER_DEFAULTS = {}
_SESSION_DEFAULTS = {}
_GLOBAL_OPTION_DEFAULTS = {}


class OptionError(ValueError):
    """Raised when user-facing options are unknown, invalid, or conflicting."""


def normalize_key(key):
    return re.sub(r"[\s\-/]+", "_", str(key).strip().lower())


_OPTION_KEY_ALIASES = {
    "colorbar_height": "colorbar_height_scale",
    "in_place": "inplace",
    "minimum_gradient_entries": "min_gradient_entries",
    "sig_fig": "sig_figs",
    "sigfig": "sig_figs",
    "sigfigs": "sig_figs",
    "significant_figure": "sig_figs",
    "significant_figures": "sig_figs",
}

_SECTION_ALIASES = {
    "import_data": "get_data",
    "peak_potential": "cv_analysis",
    "sevcik": "sevcik_analysis",
    "trumpet": "trumpet_analysis",
    "tafel": "tafel_analysis",
}

_METHOD_SECTION_ALIASES = {
    "echem.from_file": "get_data",
    "echem.x": "plot",
    "echem.y": "plot",
    "echem.xy": "plot",
    "echem.plot": "plot",
    "cv.x": "plot",
    "cv.y": "plot",
    "cv.xy": "plot",
    "cv.plot": "plot",
    "cv.current_at_potential": "cv_analysis",
    "cv.peak_potential": "cv_analysis",
    "cv.peak_current": "peak_current",
    "cv.peak_info": "peak_current",
    "cv.plateau_current": "plateau_current",
    "cv.half_peak_potential": "cv_analysis",
    "cv.half_wave_potential": "peak_current",
    "cv.wave_info": "peak_current",
    "cv.normalize": "normalize",
    "cv.normalize_current": "normalize_current",
    "cv.scale_current": "scale_current",
    "dpv.x": "plot",
    "dpv.y": "plot",
    "dpv.xy": "plot",
    "dpv.plot": "plot",
    "dpv.peak_potential": "cv_analysis",
    "cp.plot": "plot",
    "cp.get_cycles": "plot",
    "cp.plot_cycles": "plot",
    "cp.cycling_plot": "plot",
    "cp.cycle_info": "plot",
    "ca.plot": "plot",
    "ca.charge": "plot",
    "ca.time_at_charge": "plot",
}


def _canonical_option_key(key):
    norm = normalize_key(key)
    return _OPTION_KEY_ALIASES.get(norm, norm)


def _canonical_section_key(key):
    norm = normalize_key(key)
    if norm in _METHOD_SECTION_ALIASES:
        return _METHOD_SECTION_ALIASES[norm]
    return _SECTION_ALIASES.get(norm, norm)


def _friendly_key(key):
    return str(key).replace("_", " ")


_DISPLAY_KEY_OVERRIDES = {
    "c": "C",
    "d": "D",
    "e0": "E0",
    "ehalf": "Ehalf",
    "follow_e1_2": "follow e1/2",
    "plot_cv": "plot CV",
}


OPTION_CATEGORY_ORDER = (
    "Data/input",
    "Selection/filtering",
    "Reference/correction",
    "Units/normalization",
    "Axes",
    "Labels/titles",
    "Color mapping",
    "Colorbar",
    "Legend",
    "Plotting",
    "Fitting/analysis",
    "Output/display",
    "Advanced",
)

OPTION_CATEGORIES = set(OPTION_CATEGORY_ORDER)


_OPTION_CATEGORY_BY_KEY = {
    # Data and metadata inputs
    "folder_path": "Data/input",
    "delimiter": "Data/input",
    "decimal": "Data/input",
    "columns": "Data/input",
    "software": "Data/input",
    "experiment_type": "Data/input",
    "custom_reader": "Data/input",
    "recursive_search": "Data/input",
    "name_alterations": "Data/input",
    "sort_keys": "Data/input",
    "compounds": "Data/input",
    "gas": "Data/input",
    "solvent": "Data/input",
    "temperature": "Data/input",
    "metric": "Data/input",
    "x_column": "Data/input",
    "y_column": "Data/input",
    "y_columns": "Data/input",
    "group_keys": "Data/input",
    "data_mode": "Data/input",
    "scan_rate": "Data/input",
    "scan_rates": "Data/input",
    "species": "Data/input",

    # Selection and filtering
    "segment": "Selection/filtering",
    "segments": "Selection/filtering",
    "cycles": "Selection/filtering",
    "plot_segment": "Selection/filtering",
    "plot_segments": "Selection/filtering",
    "guess_potential": "Selection/filtering",
    "exact_potential": "Selection/filtering",
    "wave_range": "Selection/filtering",
    "fit_range": "Selection/filtering",
    "fit_indices": "Selection/filtering",
    "fit_ranges": "Selection/filtering",
    "log_fit_indices": "Selection/filtering",
    "mode": "Selection/filtering",
    "logic": "Selection/filtering",
    "overpotential_range": "Selection/filtering",
    "potential_window": "Selection/filtering",

    # Reference/correction
    "reference_mode": "Reference/correction",
    "reference_keywords": "Reference/correction",
    "reference_keyword": "Reference/correction",
    "reference_file": "Reference/correction",
    "reference_map": "Reference/correction",
    "reference_offset": "Reference/correction",
    "reference_guess": "Reference/correction",
    "reference_label": "Reference/correction",
    "allow_self_reference": "Reference/correction",
    "shift_potential": "Reference/correction",
    "shift_guess": "Reference/correction",
    "shift_label": "Reference/correction",
    "shift_window": "Reference/correction",
    "shift_smooth": "Reference/correction",
    "shift_max_delta_ep": "Reference/correction",
    "shift_target_delta_ep": "Reference/correction",
    "background_correction": "Fitting/analysis",
    "ecat_shift_warning_threshold": "Reference/correction",
    "ip0": "Reference/correction",
    "reference_index": "Reference/correction",
    "reference_cv": "Reference/correction",
    "reference_cvs": "Reference/correction",
    "reference_guess_potential": "Reference/correction",
    "scale": "Reference/correction",

    # Units and normalization
    "convert_current": "Units/normalization",
    "current_density": "Units/normalization",
    "invert_current": "Units/normalization",
    "electrode_diameter": "Units/normalization",
    "electrode_area": "Units/normalization",
    "area": "Units/normalization",
    "x_unit": "Units/normalization",
    "y_unit": "Units/normalization",
    "c": "Units/normalization",
    "c_unit": "Units/normalization",
    "d": "Units/normalization",
    "e0": "Units/normalization",
    "s": "Units/normalization",
    "v": "Units/normalization",
    "n": "Units/normalization",
    "num_electrons": "Units/normalization",
    "normalize": "Units/normalization",
    "normalize_params": "Units/normalization",
    "k_homo": "Units/normalization",
    "k0": "Units/normalization",
    "formula_mode": "Units/normalization",

    # Plotting
    "plot": "Plotting",
    "plot_all": "Plotting",
    "plot_fit": "Plotting",
    "plot_log_log": "Plotting",
    "log_log_plot": "Plotting",
    "plot_local_slopes": "Plotting",
    "plot_diagnostic": "Plotting",
    "new_plot": "Plotting",
    "label": "Plotting",
    "labels": "Plotting",
    "plot_labels": "Plotting",
    "deduplicate_labels": "Plotting",
    "plot_convention": "Plotting",
    "offset": "Plotting",
    "legend": "Plotting",
    "legend_mode": "Plotting",
    "legend_loc": "Plotting",
    "legend_outside": "Plotting",
    "legend_pad": "Plotting",
    "legend_bbox_to_anchor": "Plotting",
    "legend_sample_length": "Plotting",
    "legend_fontsize": "Plotting",
    "grid": "Plotting",
    "title": "Plotting",
    "subtitle": "Plotting",
    "titles": "Plotting",
    "subtitles": "Plotting",
    "title_fontsize": "Plotting",
    "subtitle_fontsize": "Plotting",
    "color": "Plotting",
    "default_colors": "Plotting",
    "default_discrete_colormap": "Plotting",
    "default_gradient_colormap": "Plotting",
    "color_mode": "Plotting",
    "colorbar_height_scale": "Plotting",
    "colorbar_reverse": "Plotting",
    "colorbar_style": "Plotting",
    "colorbar_tick_length": "Plotting",
    "colorbar_tick_pad": "Plotting",
    "colorbar_tick_labels": "Plotting",
    "colorbar_trace_ticks": "Plotting",
    "gradient_by": "Plotting",
    "gradient_species": "Plotting",
    "gradient_scale": "Plotting",
    "gradient_colormap": "Plotting",
    "gradient_colormaps": "Plotting",
    "gradient_colors": "Plotting",
    "gradient_gamma": "Plotting",
    "gradient_reverse": "Plotting",
    "min_gradient_entries": "Plotting",
    "plot_style": "Plotting",
    "fit_color": "Plotting",
    "fit_linestyle": "Plotting",
    "fit_linewidth": "Plotting",
    "fit_alpha": "Plotting",
    "fit_label": "Plotting",
    "y_col": "Plotting",
    "y_flip": "Plotting",
    "invert_y": "Plotting",
    "stacking": "Plotting",
    "label_alterations": "Plotting",
    "xlabel": "Axes",
    "ylabel": "Axes",
    "x_axis": "Axes",
    "y_axis": "Axes",
    "one_column": "Plotting",
    "segment_color_mode": "Plotting",
    "segment_color_groups": "Plotting",
    "animate": "Plotting",
    "animate_minrate": "Plotting",
    "animate_repeat": "Plotting",
    "plot_options": "Plotting",
    "minor_ticks": "Plotting",
    "symbol_labels": "Axes",
    "scale_bar": "Plotting",
    "xscale": "Plotting",
    "yscale": "Plotting",
    "plot_scale": "Plotting",

    # Analysis and fitting behavior
    "noise_window": "Fitting/analysis",
    "noise_polyorder": "Fitting/analysis",
    "peak_prominence": "Fitting/analysis",
    "tangent_range": "Fitting/analysis",
    "tangent_min_points": "Fitting/analysis",
    "tangent_potential": "Fitting/analysis",
    "percent_threshold": "Fitting/analysis",
    "fit": "Fitting/analysis",
    "fit_basis": "Fitting/analysis",
    "min_r2": "Fitting/analysis",
    "min_fit_points": "Fitting/analysis",
    "redox_mode": "Fitting/analysis",
    "diagnostic_y_axis": "Fitting/analysis",
    "catalyst_electrons": "Fitting/analysis",
    "turnover_electrons": "Fitting/analysis",
    "x_power": "Fitting/analysis",
    "x_transform": "Fitting/analysis",
    "y_transform": "Fitting/analysis",
    "transform_mode": "Fitting/analysis",
    "floor": "Fitting/analysis",
    "x_floor": "Fitting/analysis",
    "y_floor": "Fitting/analysis",
    "y_mode": "Fitting/analysis",
    "y0": "Fitting/analysis",
    "follow_e1_2": "Fitting/analysis",
    "scan_dependence": "Fitting/analysis",
    "exclude_warnings": "Fitting/analysis",
    "exclude_low_r2": "Fitting/analysis",
    "local_slope_mode": "Fitting/analysis",
    "exclude_invalid_delta_ep": "Fitting/analysis",
    "fit_through_origin": "Fitting/analysis",
    "num_electrons": "Fitting/analysis",
    "ilim": "Fitting/analysis",
    "ic": "Fitting/analysis",
    "ip0_scan_rate": "Fitting/analysis",
    "ip0_sqrt_scan_rate_slope": "Fitting/analysis",
    "plateau_slope_tolerance": "Fitting/analysis",
    "plateau_min_cvs": "Fitting/analysis",
    "plateau_average_method": "Fitting/analysis",
    "plateau_selection_mode": "Fitting/analysis",
    "validate_plateau": "Fitting/analysis",
    "require_plateau": "Fitting/analysis",
    "integrate": "Fitting/analysis",

    # Output and diagnostics
    "print": "Output/display",
    "pretty_print": "Output/display",
    "print_conditions": "Output/display",
    "print_all": "Output/display",
    "return": "Output/display",
    "return_stats": "Output/display",
    "sig_figs": "Output/display",
    "troubleshoot": "Output/display",
    "warnings": "Output/display",
    "print_local_slopes": "Output/display",
    "analysis": "Output/display",

    # Advanced/scientific-model details
    "mechanism": "Advanced",
    "sigma": "Advanced",
    "gaussian_weight": "Advanced",
    "gaussian_skew": "Advanced",
    "psi_source": "Advanced",
    "nicholson_delta_ep_min_mv": "Advanced",
    "nicholson_delta_ep_max_mv": "Advanced",
    "empirical_psi_equation": "Advanced",
    "warn_ir_drop": "Advanced",
}


_OPTION_CATEGORY_BY_SECTION = {
    "get_data": {
        "peak_prominence": "Reference/correction",
        "temperature": "Data/input",
    },
    "filter": {
        "mode": "Selection/filtering",
    },
    "group_summary": {
        "columns": "Selection/filtering",
    },
    "plot": {
        "scan_rate": "Data/input",
        "derivative": "Fitting/analysis",
    },
    "multiplot": {
        "x_axis": "Axes",
        "y_axis": "Axes",
        "x_unit": "Axes",
        "y_unit": "Axes",
        "xlabel": "Axes",
        "ylabel": "Axes",
        "plot_labels": "Labels/titles",
        "labels": "Labels/titles",
        "deduplicate_labels": "Labels/titles",
        "label_alterations": "Labels/titles",
        "title": "Labels/titles",
        "subtitle": "Labels/titles",
        "titles": "Labels/titles",
        "subtitles": "Labels/titles",
        "title_fontsize": "Labels/titles",
        "subtitle_fontsize": "Labels/titles",
        "legend": "Legend",
        "legend_mode": "Legend",
        "legend_loc": "Legend",
        "legend_outside": "Legend",
        "legend_pad": "Legend",
        "legend_bbox_to_anchor": "Legend",
        "legend_sample_length": "Legend",
        "legend_fontsize": "Legend",
        "color_mode": "Color mapping",
        "default_colors": "Color mapping",
        "default_discrete_colormap": "Color mapping",
        "default_gradient_colormap": "Color mapping",
        "gradient_by": "Color mapping",
        "gradient_species": "Color mapping",
        "gradient_scale": "Color mapping",
        "gradient_colormap": "Color mapping",
        "gradient_colormaps": "Color mapping",
        "gradient_colors": "Color mapping",
        "gradient_gamma": "Color mapping",
        "gradient_reverse": "Color mapping",
        "min_gradient_entries": "Color mapping",
        "colorbar_height_scale": "Colorbar",
        "colorbar_reverse": "Colorbar",
        "colorbar_style": "Colorbar",
        "colorbar_tick_length": "Colorbar",
        "colorbar_tick_pad": "Colorbar",
        "colorbar_tick_labels": "Colorbar",
        "colorbar_trace_ticks": "Colorbar",
    },
    "tafel_analysis": {
        "overpotential_range": "Fitting/analysis",
    },
}


OPTION_DESCRIPTIONS = {
    "allow_self_reference": "Allow a CV to be considered as its own reference candidate during automatic reference shifting.",
    "analysis": "Enable additional analysis output for multi-panel plotting workflows.",
    "animate": "Animate the CV scan instead of making a static plot.",
    "animate_minrate": "Minimum frame rate used for CV animation playback.",
    "animate_repeat": "Repeat the animation after the final frame.",
    "baseline_correction": "Baseline-current correction for CA charge integration: False, True, tail, or threshold.",
    "baseline_tail_fraction": "Final fraction of a CA trace used for tail baseline correction.",
    "baseline_threshold": "Current threshold in A used for threshold CA baseline correction.",
    "background_correction": "Background correction method used before kinetic analysis.",
    "c": "Analyte or catalyst concentration; strings are parsed as concentration units when supported. For normalize, explicit C overrides species-based lookup.",
    "c_unit": "Unit for numeric concentration values. Required for numeric C; not needed when C is a concentration string or normalize resolves C from species.",
    "catalyst_electrons": "Number of catalyst redox-wave electron-transfer processes used in kinetic equations.",
    "charge_color": "Color used for cumulative charge overlays and target markers.",
    "color": "Primary plot color.",
    "color_mode": "Color assignment mode for multi-trace plots.",
    "cycles": "Cycle selection for CP cycle plots: int, list of ints, (start, end), or (start, end, step).",
    "colorbar_height_scale": "Scale factor for colorbar height in gradient legends.",
    "colorbar_reverse": "Reverse colorbar order.",
    "colorbar_style": "Colorbar rendering style, such as auto, continuous, or discrete swatches.",
    "colorbar_tick_length": "Length of colorbar ticks.",
    "colorbar_tick_labels": "Which colorbar ticks receive text labels.",
    "colorbar_tick_pad": "Padding between colorbar ticks and labels.",
    "colorbar_trace_ticks": "Show tick marks for each trace value on gradient colorbars.",
    "columns": "Number of columns expected in imported data files.",
    "compounds": "Compound names associated with the electrochemical object.",
    "convert_current": "Convert current values to a requested display or analysis unit.",
    "current_density": "Normalize current by electrode area to report current density.",
    "custom_formula": "Callable custom kinetic formula used instead of the built-in formula.",
    "custom_reader": "User-provided file reader for custom import formats.",
    "d": "Diffusion coefficient in cm^2/s.",
    "data_mode": "Whether multi_scatterplot uses automatic, raw, adjusted, or transformed result-table columns.",
    "decimal": "Decimal separator used in imported text files.",
    "default_colors": "Default discrete colors used for multi-trace plots.",
    "default_discrete_colormap": "Default colormap used for discrete color legends.",
    "default_gradient_colormap": "Default colormap used for gradient legends.",
    "deduplicate_labels": "Append distinguishing metadata to duplicate multiplot labels; True uses scan window and segments.",
    "delimiter": "Column delimiter used in imported text files.",
    "diagnostic_y_axis": "Y axis used for FOWA diagnostic multiplot output.",
    "e0": "Formal potential used for physical dimensionless CV normalization.",
    "ecat_shift_warning_threshold": "Potential-shift threshold for warning that catalytic and reference waves may not align.",
    "electrode_area": "Electrode area in cm^2.",
    "electrode_diameter": "Electrode diameter in cm.",
    "ehalf": "Half-wave potential used for display or multi-panel analysis.",
    "empirical_psi_equation": "Empirical Nicholson psi equation used when not using the table lookup.",
    "exact_potential": "Exact potential to use for peak or current extraction.",
    "exclude_invalid_delta_ep": "Exclude Nicholson points outside the valid nDelta Ep range from fitting.",
    "exclude_low_r2": "Exclude fits whose R2 is below the requested threshold.",
    "exclude_warnings": "Exclude rows or fits that emitted analysis warnings.",
    "experiment_type": "Experiment type to assume or require during import.",
    "fit": "Whether to fit the analysis result.",
    "fit_alpha": "Alpha transparency for plotted fit lines.",
    "fit_basis": "Axis or quantity used to select the fit region.",
    "fit_color": "Color of plotted fit lines.",
    "fit_colors": "Sequence of colors for plotted fit lines when multiple fits are drawn.",
    "fit_indices": "Indices, mask, or index windows included in a fit.",
    "fit_label": "Whether to label the fit line, or custom label text.",
    "fit_model": "Fit model name, callable, or formula string.",
    "fit_params": "Parameter names for a custom callable or formula fit model.",
    "fit_equation": "Optional display equation for custom fit model printouts.",
    "fit_init": "Initial parameter guesses for a fit model.",
    "fit_bounds": "Lower and upper bounds for a fit model.",
    "fit_residual": "Residual mode used for model fitting.",
    "fit_max_evals": "Maximum function evaluations for model fitting.",
    "print_fit": "Fit print style: auto, summary, or details.",
    "print_fit_details": "If True, force detailed two-table fit printing.",
    "fit_ranges": "Named or unnamed x-value windows for multiple fits. Use a dict for named fits or a list for generated labels.",
    "fit_linestyle": "Line style for plotted fit lines.",
    "fit_linewidth": "Line width for plotted fit lines.",
    "fit_range": "Range of the transformed or raw axis included in fitting.",
    "fit_through_origin": "Fit the model through the origin rather than fitting an intercept.",
    "folder_path": "Folder path searched for electrochemical data files.",
    "follow_e1_2": "Use or follow E1/2 values when fitting peak-potential trends.",
    "formula_label": "Display label for a custom kinetic formula.",
    "formula_mode": "Formula-selection mode for plateau-current analysis.",
    "gas": "Gas condition metadata for the electrochemical object.",
    "gaussian_skew": "Skew parameter for Gaussian-style smoothing or weighting.",
    "gaussian_weight": "Weight parameter for Gaussian-style smoothing or weighting.",
    "gradient_by": "Metadata field used to assign gradient colors.",
    "gradient_colormap": "Colormap used for a gradient legend.",
    "gradient_colormaps": "List of colormaps used for multiple gradient legends.",
    "gradient_colors": "Explicit colors used for gradient mapping.",
    "gradient_gamma": "Gamma adjustment applied to gradient color interpolation.",
    "gradient_reverse": "Reverse gradient color order.",
    "gradient_scale": "Scale used for gradient color mapping.",
    "gradient_species": "Species used to resolve concentration-based gradient coloring.",
    "grid": "Whether to show major grid lines on plots.",
    "group_keys": "Metadata field or fields used to group objects before summarizing.",
    "guess_potential": "Initial potential guess for peak or wave selection.",
    "ic": "Manual catalytic plateau current.",
    "ilim": "Manual limiting or plateau current.",
    "integrate": "Integrate the selected signal when supported.",
    "internal_call": "Mark an internal helper call to suppress user-facing side effects.",
    "invert_current": "Multiply current by -1.",
    "inplace": "Mutate the existing object instead of returning a copied result.",
    "invert_y": "Invert the plotted y-axis.",
    "ip0": "Non-catalytic reference peak current.",
    "ip0_scan_rate": "Scan rate of the CV used to measure ip0.",
    "ip0_sqrt_scan_rate_slope": "Forced-origin slope for ip0 versus sqrt(scan rate).",
    "label": "Label for a plotted trace or analysis output.",
    "label_alterations": "Text replacements applied to generated labels.",
    "labels": "Explicit labels for plotted objects.",
    "legend": "Whether to show a plot legend.",
    "legend_bbox_to_anchor": "Matplotlib legend anchor box.",
    "legend_fontsize": "Font size for plot legends.",
    "legend_loc": "Matplotlib legend location.",
    "legend_mode": "Legend mode for discrete or gradient color encodings.",
    "legend_outside": "Place the legend outside the axes.",
    "legend_pad": "Padding used when placing legends outside the axes.",
    "legend_sample_length": "Visual sample length used for legend handles.",
    "local_slope_mode": "Method for calculating local slopes.",
    "log_fit_indices": "Fit indices used for log-transformed fits.",
    "log_log_plot": "Plot the fit result on log-log axes.",
    "logic": "Logical rule used to combine filter criteria.",
    "mechanism": "Electrocatalytic mechanism label or model choice.",
    "metric": "Metric column or quantity to analyze.",
    "min_fit_points": "Minimum recommended number of fit points.",
    "min_gradient_entries": "Minimum number of entries before using gradient color mapping.",
    "minor_ticks": "Minor tick behavior for plots; use True for eCAT defaults, False to disable, or an integer locator subdivision count.",
    "symbol_labels": "Use compact electrochemistry symbols for axis labels, such as E, i, j, t, and Q; 'auto' follows the active plotting style.",
    "min_r2": "Minimum recommended R2 threshold.",
    "mode": "Mode selector for the current operation.",
    "name_alterations": "Filename text replacements applied during import.",
    "new_plot": "Create a new figure before plotting.",
    "nicholson_delta_ep_max_mv": "Maximum valid nDelta Ep in mV for Nicholson analysis.",
    "nicholson_delta_ep_min_mv": "Minimum valid nDelta Ep in mV for Nicholson analysis.",
    "noise_polyorder": "Savitzky-Golay smoothing polynomial order.",
    "noise_window": "Savitzky-Golay smoothing window.",
    "non_catalytic_current": "Manual non-catalytic reference current.",
    "non_catalytic_cv": "Single non-catalytic reference CV.",
    "non_catalytic_cvs": "List of non-catalytic reference CVs.",
    "non_catalytic_guess_potential": "Potential guess used for non-catalytic reference extraction.",
    "normalize": "Whether to normalize current or axes during processing.",
    "normalize_params": "Parameters used for legacy normalization.",
    "n": "Number of electrons used in dimensionless or kinetic equations.",
    "num_electrons": "Number of electrons in the redox event.",
    "offset": "Vertical offset applied to plotted traces.",
    "one_column": "Use a one-column plot or document layout when supported.",
    "area": "Electrode area used for physical dimensionless CV normalization.",
    "overpotential_range": "Potential range used for Tafel or overpotential analysis.",
    "peak_potential": "Manual peak potential.",
    "peak_fallback": "Fallback used by peak_current when peak_potential cannot find a local extremum.",
    "peak_prominence": "Minimum peak prominence for automatic peak detection.",
    "percent_threshold": "Percent threshold used in peak or tangent selection.",
    "plateau_average_method": "Average method used to combine accepted plateau currents.",
    "plateau_min_cvs": "Minimum number of CVs required in a plateau-validation subset.",
    "plateau_selection_mode": "Strategy used to select scan-rate-independent plateau subsets.",
    "plateau_slope_tolerance": "Maximum fractional slope metric accepted as scan-rate independent.",
    "plot": "Whether to plot the result.",
    "plot_all": "Whether to show diagnostic or child plots.",
    "plot_ca": "Whether CA charge helpers include the current-vs-time trace before adding diagnostics.",
    "plot_charge": "Overlay cumulative charge on a secondary axis when plotting chronoamperometry data.",
    "plot_convention": "Electrochemical plotting convention for axis orientation and signs.",
    "plot_cv": "Whether CV analysis helpers redraw the underlying CV trace before adding diagnostics.",
    "plot_diagnostic": "Whether to show diagnostic plots for the analysis.",
    "plot_fit": "Whether to overlay fitted curves or lines.",
    "plot_labels": "Explicit labels for plotted traces.",
    "plot_local_slopes": "Whether to plot local slope diagnostics.",
    "plot_log_log": "Whether to include a log-log plot.",
    "plot_scale": "Convenience axis-scale preset for scatter plots, such as log-log, semilogx, semilogy, symlog, or linear. Uses Matplotlib axis scaling and does not transform fit values.",
    "plot_options": "Nested plotting options passed through to plot helpers.",
    "scale_bar": "Scale-bar options for plots. Use False to hide, True or a dict to draw, with length in displayed y-axis units; dicts may include loc, label, fontsize, color, linewidth, cap width, and label pad.",
    "potential_window": "Two-value potential window used to select or trim CV data.",
    "_provided_options": "Internal record of explicitly provided options.",
    "plot_peak_potential": "Whether peak-potential diagnostics are plotted during peak-current extraction.",
    "plot_segment": "Segment to emphasize or plot.",
    "plot_segments": "Segments to emphasize or plot.",
    "plot_style": "Plot style such as scatter or line.",
    "plot_target": "Mark the requested charge target on charge or chronoamperometry plots.",
    "pretty_print": "Use rich table display when printing object lists or summaries. False uses plain-text output when print is True.",
    "print": "Whether to emit output. False suppresses output; it is independent of pretty print.",
    "print_all": "Whether child helper calls should print their own summaries.",
    "print_conditions": "Whether to include condition columns in printed object tables.",
    "print_local_slopes": "Whether to print local slope diagnostics.",
    "psi_source": "Source used to resolve Nicholson psi values.",
    "recursive_search": "Recursively search subfolders during import.",
    "redox_mode": "Method used to resolve the reference redox potential.",
    "redox_potential": "Manual redox reference potential.",
    "reference_file": "Explicit file used as a reference-shift source.",
    "reference_guess": "Potential guess used to locate a reference wave.",
    "reference_cv": "Single reference CV used for current normalization or scaling.",
    "reference_cvs": "List of reference CVs used for current normalization or scaling.",
    "reference_guess_potential": "Potential guess used when extracting current from reference CVs.",
    "reference_index": "Index of the reference CV used for current normalization or scaling.",
    "reference_keyword": "Single filename keyword used to identify reference files.",
    "reference_keywords": "Filename keywords used to identify reference files.",
    "reference_label": "Axis label used after reference shifting.",
    "reference_map": "Explicit imported-object index mapping, e.g. {45: 54}, where the key is the object to shift and the value is the object used as its reference source.",
    "reference_mode": "Reference mode for the current operation.",
    "reference_offset": "Manual reference potential offset.",
    "require_plateau": "Raise an error if no scan-rate-independent plateau subset is found.",
    "return_stats": "Return a statistics dictionary instead of only a compact fit result.",
    "return": "Return the options table or menu as a pandas DataFrame.",
    "scan_dependence": "Exponent applied to scan rate.",
    "scan_rate": "Scan rate in V/s.",
    "scan_rates": "Manual scan rates in V/s.",
    "s": "Electrode area alias used for physical dimensionless CV normalization.",
    "scale": "Multiplier applied to raw current columns by scale_current.",
    "segment": "CV segment to analyze.",
    "segment_color_groups": "Segment grouping used for cv.plot segment coloring; integer minimum size or explicit segment groups.",
    "segment_color_mode": "Segment color mode for cv.plot, such as off, discrete, discrete gradient, or continuous gradient.",
    "segments": "One or more CV segments to analyze.",
    "shift_guess": "Legacy potential guess for reference shifting.",
    "shift_label": "Legacy label used after potential shifting.",
    "shift_max_delta_ep": "Maximum reference peak separation allowed during reference matching.",
    "shift_potential": "Legacy flag or value for shifting the potential axis.",
    "shift_smooth": "Smooth data before locating reference-shift peaks.",
    "shift_target_delta_ep": "Target reference peak separation used during reference matching.",
    "shift_window": "Potential window used to locate reference-shift peaks.",
    "sig_figs": "Significant figures used for reported values.",
    "sigma": "Stoichiometric or pathway exponent used in FOWA kinetics.",
    "software": "Instrument software or parser to use during import.",
    "solvent": "Solvent metadata for the electrochemical object.",
    "sort_keys": "Sort keys used to order imported objects after parsing and before reference assignment.",
    "species": "Chemical species used to resolve concentration or metadata. For normalize, exact-matches cv.compounds and pulls the paired cv.concentrations value when C/C unit are omitted.",
    "stacking": "Vertically stack plotted traces.",
    "subtitle": "Plot subtitle text or subtitle mode.",
    "subtitle_fontsize": "Font size for subtitles.",
    "subtitles": "Subtitles for multiple plots or panels.",
    "tangent_min_points": "Minimum number of points used for tangent baseline fitting.",
    "tangent_potential": "Potential at which to anchor tangent baseline fitting.",
    "tangent_range": "Potential range for tangent baseline fitting.",
    "temperature": "Temperature in K.",
    "target_charge": "Target cumulative charge in Coulombs.",
    "target_electrons": "Number of electrons per molecule used to convert target moles to Coulombs.",
    "target_label": "Custom label for a plotted charge target marker.",
    "target_moles": "Target amount in moles used with target electrons to calculate charge.",
    "title": "Plot title text or title mode.",
    "title_fontsize": "Font size for titles.",
    "titles": "Titles for multiple plots or panels.",
    "transform_mode": "Transform mode applied to x and/or y data.",
    "troubleshoot": "Print additional troubleshooting output.",
    "turnover_electrons": "Catalyst equivalents or electrons used per turnover in kinetic equations.",
    "validate_plateau": "Validate plateau-current scan-rate independence.",
    "v": "Scan-rate alias used for physical dimensionless CV normalization.",
    "k_homo": "Homogeneous rate constant used for EC' dimensionless CV normalization.",
    "k0": "Heterogeneous rate constant used for electrochemical dimensionless CV normalization.",
    "warn_ir_drop": "Warn when iR compensation or background subtraction cannot be verified.",
    "warnings": "Whether to emit Python warnings for non-fatal analysis quality issues.",
    "wave_range": "Manual FOWA catalytic-wave potential range; may be one range or one range per CV.",
    "x_axis": "Column or axis used as x data.",
    "x_column": "DataFrame column used as x data.",
    "x_column_index": "Column index used as x data.",
    "x_power": "Power applied to x values.",
    "x_scale": "Scale factor applied to x values for plotting.",
    "xscale": "Matplotlib x-axis scale such as linear, log, symlog, or logit. This changes axis spacing only; use x transform to change fit values.",
    "x_transform": "Transform applied to x values.",
    "x_unit": "Requested x-axis unit.",
    "xlabel": "Custom x-axis label.",
    "y_axis": "Column or axis used as y data.",
    "y_col": "Column index used as y data.",
    "y_column": "DataFrame column used as y data.",
    "y_column_index": "Column index used as y data.",
    "y_columns": "Multiple DataFrame columns used as y data.",
    "y_flip": "Flip the sign of y values.",
    "y_mode": "Y-series adjustment mode before y transforms, such as raw, delta, ratio, or enhancement.",
    "y0": "Baseline y value for y-series adjustment; may be a scalar or keyed mapping.",
    "y_scale": "Scale factor applied to y values for plotting.",
    "yscale": "Matplotlib y-axis scale such as linear, log, symlog, or logit. This changes axis spacing only; use y transform to change fit values.",
    "y_transform": "Transform applied to y values.",
    "y_unit": "Requested y-axis unit.",
    "ylabel": "Custom y-axis label.",
    "derivative": "Derivative order or mode used for plotted data.",
    "smooth": "Smooth plotted data before display.",
    "floor": "Floor threshold for positive-only transforms. True uses 0.1x the smallest positive axis value, a number is absolute, and strings like '0.1x' are relative. Values below the threshold are replaced before transforming.",
    "x_floor": "X-axis floor threshold overriding floor for positive-only x transforms.",
    "y_floor": "Y-axis floor threshold overriding floor for positive-only y transforms.",
}


OPTION_CHOICES = {
    "background_correction": ["tangent", "start current"],
    "color_mode": ["auto", "gradient", "discrete"],
    "diagnostic_y_axis": ["i/ip0", "current"],
    "fit_basis": ["x", "y"],
    "formula_mode": ["auto", "normalized", "slope normalized", "direct"],
    "gradient_by": ["auto", "scan rate", "concentration"],
    "gradient_scale": ["auto", "linear", "sqrt", "log", "index"],
    "colorbar_tick_labels": ["endpoints", "all", "none"],
    "legend_mode": ["auto", "colorbar", "discrete"],
    "colorbar_style": ["auto", "continuous", "discrete"],
    "local_slope_mode": ["adjacent", "gradient"],
    "logic": ["AND", "OR"],
    "mode": ["include", "exclude"],
    "plateau_average_method": ["mean", "median"],
    "plateau_selection_mode": ["high scan suffix"],
    "plot_convention": ["US", "IUPAC"],
    "plot_style": ["scatter", "line", "line+markers"],
    "plot_scale": ["linear", "log-log", "semilogx", "semilogy", "symlog"],
    "psi_source": ["agarwal table", "empirical"],
    "redox_mode": ["half wave", "half peak", "manual"],
    "reference_mode": ["auto", "manual", "keyword", "file", "none"],
    "segment_color_mode": ["auto", "off", "discrete", "discrete gradient", "continuous gradient"],
    "transform_mode": ["log-log", "lineweaver-burk"],
    "y_mode": ["raw", "delta", "negative delta", "ratio", "enhancement"],
}

OPTION_CHOICES_BY_SECTION = {
    "scale_current": {
        "reference_mode": ["single", "both"],
    },
    "nicholson": {
        "fit_model": ["origin", "linear"],
    },
}


def _build_option_metadata():
    keys = set(_OPTION_CATEGORY_BY_KEY) | set(OPTION_DESCRIPTIONS) | set(OPTION_CHOICES)
    metadata = {
        "*": {
            key: {
                field: value
                for field, value in {
                    "category": _OPTION_CATEGORY_BY_KEY.get(key),
                    "choices": OPTION_CHOICES.get(key),
                    "description": OPTION_DESCRIPTIONS.get(key),
                }.items()
                if value not in (None, "")
            }
            for key in sorted(keys)
        }
    }

    for section, section_categories in _OPTION_CATEGORY_BY_SECTION.items():
        section_key = _canonical_section_key(section)
        metadata.setdefault(section_key, {})
        for key, category in section_categories.items():
            metadata[section_key].setdefault(normalize_key(key), {})["category"] = category

    for section, section_choices in OPTION_CHOICES_BY_SECTION.items():
        section_key = _canonical_section_key(section)
        metadata.setdefault(section_key, {})
        for key, choices in section_choices.items():
            metadata[section_key].setdefault(normalize_key(key), {})["choices"] = list(choices)

    def update(section, entries):
        section_key = _canonical_section_key(section)
        section_metadata = metadata.setdefault(section_key, {})
        for key, values in entries.items():
            section_metadata.setdefault(normalize_key(key), {}).update(values)

    update("get_data", {
        "temperature": {
            "description": "Temperature in K assigned to imported objects when the file does not provide one.",
        },
        "electrode_area": {
            "description": "Electrode area in cm^2 assigned to imported objects; may be computed from electrode diameter when current density is requested.",
        },
        "electrode_diameter": {
            "description": "Electrode diameter in cm used to compute electrode area for current-density conversion when electrode area is not provided.",
        },
        "scan_rate": {
            "description": "Scan rate in V/s assigned to imported objects when the parser or file metadata does not provide one.",
        },
        "reference_mode": {
            "description": "'auto' searches imported files with reference keywords; 'keyword', 'file', and 'manual' use the corresponding explicit reference source.",
        },
        "reference_guess": {
            "description": "'auto' locates the reference wave automatically; a numeric value searches near that potential.",
        },
        "shift_guess": {
            "description": "Legacy reference-shift guess. 'auto' locates the shifted reference wave automatically.",
        },
        "reference_keywords": {
            "description": "Reference keywords tried by reference mode 'auto'; the first successful keyword defines the shift source.",
        },
        "reference_keyword": {
            "description": "Single keyword used by reference mode 'keyword' to choose reference files.",
        },
        "reference_file": {
            "description": "Explicit reference file used by reference mode 'file'; no file matching is inferred.",
        },
        "reference_map": {
            "description": "Explicit target-to-reference object index mapping; overrides automatic reference assignment for those objects.",
        },
        "peak_prominence": {
            "description": "Minimum peak prominence for automatic reference-wave detection; None lets peak detection use its internal default.",
        },
        "software": {
            "description": "Instrument software/parser. If omitted, import tries to infer the parser from file contents and extension.",
        },
        "experiment_type": {
            "description": "Experiment type to require or assign. If omitted, eCAT promotes objects from parser/file metadata when possible.",
        },
    })

    update("filter", {
        "mode": {
            "choices": ["include", "exclude"],
            "description": "Mode selector: include or exclude objects matching the filter criteria.",
        },
    })

    update("plot", {
        "x_unit": {
            "description": "'auto' auto scales from the displayed data and object units; explicit units force the requested x-axis unit.",
        },
        "y_unit": {
            "description": "'auto' auto scales from the displayed data and object units; explicit units force the requested y-axis unit.",
        },
        "legend": {
            "description": "'auto' suppresses single-entry legends and shows legends when multiple plotted entries need labels.",
        },
        "legend_loc": {
            "description": "'auto' uses Matplotlib's best location for in-axes legends.",
        },
        "legend_mode": {
            "description": "'auto' auto chooses colorbar-style legends when gradient coloring is active; otherwise discrete legend behavior is used.",
        },
        "legend_sample_length": {
            "description": "'auto' sizes custom legend line samples from the legend font and tick geometry.",
        },
        "colorbar_style": {
            "description": "'auto' uses discrete swatches for small segment groups and continuous bars for continuous gradients.",
        },
        "segment_color_mode": {
            "description": "'auto' auto colors multi-segment CVs with discrete gradients when enough segments are present and keeps simple line styling for short CVs.",
        },
        "noise_window": {
            "description": "'auto' chooses an odd Savitzky-Golay window from the selected data length when smoothing is requested.",
        },
        "noise_polyorder": {
            "description": "'auto' chooses a Savitzky-Golay polynomial order compatible with the resolved smoothing window.",
        },
        "title": {
            "description": "Plot title text or title mode. 'auto' derives a concise title from object metadata.",
        },
        "subtitle": {
            "description": "Plot subtitle text or subtitle mode. 'auto' uses object metadata that is useful context but not already in the title.",
        },
        "scan_rate": {
            "description": "Scan rate in V/s used for animation or normalization helpers; plotting otherwise uses the object's stored scan_rate when needed.",
        },
    })

    update("multiplot", {
        "title": {
            "description": "'auto' builds a shared title from metadata common to the plotted objects.",
        },
        "subtitle": {
            "description": "'auto' builds shared context from metadata common to the plotted objects.",
        },
        "titles": {
            "description": "'auto' builds panel titles from grouped-object metadata.",
        },
        "subtitles": {
            "description": "'auto' builds panel subtitles from grouped-object metadata.",
        },
        "labels": {
            "description": "Explicit labels for plotted objects. If omitted, labels are generated from object metadata.",
        },
        "plot_labels": {
            "description": "Explicit labels for plotted objects. If omitted, labels are generated from object metadata.",
        },
        "color_mode": {
            "description": "'auto' detects scan-rate or concentration gradients and colors those groups by gradient; remaining traces use discrete colors.",
        },
        "gradient_by": {
            "description": "'auto' first looks for scan-rate gradients, then concentration gradients.",
        },
        "gradient_species": {
            "description": "'auto' uses the species whose concentration varies within the detected concentration-gradient group.",
        },
        "gradient_scale": {
            "description": "'auto' uses log scale for scan rate, square-root scale for concentration, and linear scale otherwise.",
        },
        "gradient_reverse": {
            "description": "Reverse trace color assignment within the gradient; unlike colorbar reverse, this changes which colors are applied to traces.",
        },
        "legend_mode": {
            "description": "'auto' uses colorbar legends for detected gradients and discrete labels for non-gradient traces.",
        },
        "legend_loc": {
            "description": "'auto' uses Matplotlib's best location for in-axes legends.",
        },
        "legend_sample_length": {
            "description": "'auto' sizes custom legend line samples from the legend font and tick geometry.",
        },
        "colorbar_reverse": {
            "description": "Reverse the displayed colorbar direction without changing trace color assignment.",
        },
        "colorbar_tick_labels": {
            "description": "'endpoints' labels only min/max colorbar values, 'all' labels every trace-value tick, and 'none' hides tick text.",
        },
        "colorbar_trace_ticks": {
            "description": "If True, draw tick marks at each trace's gradient value; if False, draw only endpoint ticks.",
        },
    })

    update("multimultiplot", {
        "titles": {
            "description": "'auto' derives one title per panel/group from grouped-object metadata.",
        },
        "subtitles": {
            "description": "'auto' derives one subtitle per panel/group from grouped-object metadata.",
        },
    })

    update("multi_scatterplot", {
        "data_mode": {
            "description": "'auto' lets x/y column resolution prefer transformed result columns while falling back to raw or common metric columns. This selects existing result data; it does not recompute upstream fits.",
        },
        "x_column": {
            "description": "'auto' auto prefers transformed/raw x columns and falls back to the first sensible x column. Explicit columns control plotted points; stored fit overlays are reused only when compatible.",
        },
        "y_column": {
            "description": "'auto' auto prefers transformed, metric, kobs, TOFmax, ip, then Ep result columns. Explicit columns control plotted points; stored fit overlays are reused only when compatible.",
        },
        "y_columns": {
            "description": "Explicit y columns to plot; when omitted, y-column auto-resolution is used. Selecting a raw metric from a non-raw upstream fit requires 'plot fit': False or a matching upstream fit.",
        },
        "metric": {
            "description": "Preferred metric column used during y-column auto-resolution.",
        },
    })

    cv_auto = {
        "x_unit": {
            "description": "'auto' auto scales from the selected CV data and stored units.",
        },
        "y_unit": {
            "description": "'auto' auto scales from the selected CV data and stored units.",
        },
        "noise_window": {
            "description": "'auto' chooses an odd Savitzky-Golay window from the selected data length.",
        },
        "noise_polyorder": {
            "description": "'auto' chooses a Savitzky-Golay polynomial order compatible with the resolved smoothing window.",
        },
        "peak_prominence": {
            "description": "Minimum peak prominence for automatic peak detection; None uses the peak-detection default.",
        },
        "peak_fallback": {
            "description": "Fallback used by peak_current when no local peak is detected. 'highest current' uses the largest absolute current in the selected segment; 'guess potential' treats the guess as an exact potential; None/'none' keeps the strict error.",
        },
        "guess_potential": {
            "description": "Initial potential guess for automatic peak or wave selection; omitted values let eCAT choose from the selected segment.",
        },
        "exact_potential": {
            "description": "Exact potential for current extraction; when provided it bypasses peak-location auto-selection.",
        },
        "plot_cv": {
            "description": "When plotting CV analysis diagnostics, draw the underlying CV trace before adding markers and lines.",
        },
    }
    update("cv_analysis", cv_auto)
    update("peak_current", {
        **cv_auto,
        "tangent_range": {
            "description": "'auto' chooses a pre-peak baseline region from derivative/curvature behavior before the peak.",
        },
        "tangent_min_points": {
            "description": "Minimum points for tangent fitting; when omitted, eCAT derives a minimum from the pre-peak data length.",
        },
        "tangent_potential": {
            "description": "Manual tangent anchor potential; when omitted, eCAT anchors from the resolved tangent region.",
        },
    })

    update("normalize", {
        "mode": {
            "category": "Units/normalization",
            "choices": ["homogeneous", "heterogeneous"],
            "description": "Normalization family: homogeneous uses Φ for current; heterogeneous uses χ.",
        },
        "temperature": {
            "category": "Units/normalization",
            "description": "Temperature in K. If omitted, uses the CV's temperature.",
        },
        "electrode_area": {
            "category": "Units/normalization",
            "description": "Electrode area in cm^2. If omitted for Dimensionless Current, uses the CV's electrode_area.",
        },
        "area": {
            "category": "Units/normalization",
            "description": "Electrode area alias in cm^2. If omitted for Dimensionless Current, uses the CV's electrode_area.",
        },
        "s": {
            "category": "Units/normalization",
            "description": "Electrode area alias in cm^2. If omitted for Dimensionless Current, uses the CV's electrode_area.",
        },
        "scan_rate": {
            "category": "Units/normalization",
            "description": "Scan rate in V/s. If omitted for Dimensionless Current, uses the CV's scan_rate.",
        },
        "v": {
            "category": "Units/normalization",
            "description": "Scan-rate alias in V/s. If omitted for Dimensionless Current, uses the CV's scan_rate.",
        },
        "species": {
            "category": "Units/normalization",
            "description": "If C/C unit are omitted, exact-matches cv.compounds and uses the paired cv.concentrations value.",
        },
        "c": {
            "category": "Units/normalization",
            "description": "Analyte or catalyst concentration. Explicit C overrides species-based lookup.",
        },
        "c_unit": {
            "category": "Units/normalization",
            "description": "Unit for numeric C. Required for numeric C; not needed when C is a concentration string or species resolves it.",
        },
        "d": {
            "category": "Units/normalization",
            "description": "Diffusion coefficient in cm^2/s. Required for Dimensionless Current; not inferred.",
        },
        "e0": {
            "category": "Units/normalization",
            "description": "Formal potential in V. Required for Dimensionless Potential; not inferred.",
        },
    })

    update("normalize_current", {
        "ip0": {
            "description": "Manual non-catalytic reference peak current; if omitted, eCAT extracts ip0 from the selected reference CV.",
        },
        "reference_index": {
            "description": "Reference CV index used when ip0/reference CVs are omitted; default 0 uses the first input CV.",
        },
        "reference_cv": {
            "description": "Single reference CV used to extract ip0 automatically with peak_current.",
        },
        "reference_cvs": {
            "description": "Reference CVs paired with input CVs to extract ip0 automatically with peak_current.",
        },
        "reference_guess_potential": {
            "description": "Guess potential used when extracting ip0 from reference CVs; omitted values use the peak-current guess potential.",
        },
        "tangent_range": {
            "description": "'auto' is passed to peak_current so ip0 extraction uses automatic tangent-baseline selection.",
        },
    })

    update("scale_current", {
        "scale": {
            "description": "Manual current multiplier. If omitted, eCAT computes a scale factor from reference CV peak currents.",
        },
        "reference_mode": {
            "choices": ["single", "both"],
            "description": "'single' scales by one reference peak current; 'both' extracts paired segment currents and computes a best scale factor.",
        },
        "reference_index": {
            "description": "Reference CV index used when scale/reference CVs are omitted; default 0 uses the first input CV.",
        },
        "reference_cv": {
            "description": "Single reference CV used to compute scale automatically with peak_current.",
        },
        "reference_cvs": {
            "description": "Reference CVs paired with input CVs to compute scale automatically with peak_current.",
        },
        "tangent_range": {
            "description": "'auto' is passed to peak_current so reference-current extraction uses automatic tangent-baseline selection.",
        },
    })

    update("fowa", {
        "redox_mode": {
            "description": "Method used to resolve redox potential: half-wave, half-peak, or manual redox potential.",
        },
        "redox_potential": {
            "description": "Manual redox potential; required only when redox mode is manual.",
        },
        "ip0": {
            "description": "Manual non-catalytic peak current. If omitted, FOWA can extract ip0 from non-catalytic CV inputs.",
        },
        "non_catalytic_current": {
            "description": "Manual non-catalytic current alternative to ip0; no CV reference extraction is used when provided.",
        },
        "non_catalytic_cv": {
            "description": "Single non-catalytic CV used to extract the reference current automatically.",
        },
        "non_catalytic_cvs": {
            "description": "Non-catalytic CVs paired with catalytic CVs to extract reference currents automatically.",
        },
        "non_catalytic_guess_potential": {
            "description": "Guess potential used only for non-catalytic reference-current extraction; omitted values fall back to guess potential.",
        },
        "wave_range": {
            "description": "Manual catalytic-wave window. If omitted, FOWA auto-selects a wave region from the transformed diagnostic curve.",
        },
        "tangent_range": {
            "description": "'auto' is passed to peak_current so background/current extraction uses automatic tangent-baseline selection.",
        },
        "background_correction": {
            "description": "Background correction method used before FOWA; tangent uses the auto/manual tangent-baseline controls.",
        },
    })

    update("plateau_current", {
        "temperature": {
            "description": "Temperature in K. If omitted, uses CV or reference-CV temperature metadata before falling back to the option default.",
        },
        "electrode_area": {
            "description": "Electrode area in cm^2. If omitted for direct mode, uses the catalytic CV's electrode_area metadata.",
        },
        "scan_rate": {
            "description": "Manual scan rate in V/s used for normalized kobs when ip0 scan rate is omitted; otherwise catalytic/reference CV scan_rate metadata are used.",
        },
        "species": {
            "description": "Species used to resolve concentration from the catalytic CV's compounds/concentrations metadata when C is omitted.",
        },
        "c": {
            "description": "Catalyst concentration. If omitted, plateau_current can resolve it from species metadata; otherwise not inferred.",
        },
        "c_unit": {
            "description": "Unit for numeric C. If omitted for numeric C, plateau_current assumes M and converts to mol/cm^3.",
        },
        "d": {
            "description": "Diffusion coefficient in cm^2/s for direct plateau-current mode; not inferred.",
        },
        "ilim": {
            "description": "Manual catalytic limiting current. If omitted, plateau_current extracts it from catalytic CVs with peak_current.",
        },
        "ic": {
            "description": "Manual catalytic plateau current alias for ilim. If omitted, plateau_current extracts it from catalytic CVs with peak_current.",
        },
        "ip0": {
            "description": "Manual non-catalytic peak current. If omitted, plateau_current can extract it from non-catalytic CVs.",
        },
        "non_catalytic_current": {
            "description": "Manual non-catalytic reference current alternative to ip0.",
        },
        "non_catalytic_cv": {
            "description": "Single non-catalytic CV used to extract ip0 automatically.",
        },
        "non_catalytic_cvs": {
            "description": "Non-catalytic CVs used to extract ip0 or an ip0-versus-sqrt(scan rate) slope automatically.",
        },
        "ip0_scan_rate": {
            "description": "Scan rate associated with manual ip0; if omitted, scan rate may supply it.",
        },
        "ip0_sqrt_scan_rate_slope": {
            "description": "Manual forced-origin slope of ip0 versus sqrt(scan rate); if omitted, it can be fitted from multiple non-catalytic CVs.",
        },
        "formula_mode": {
            "description": "'auto' chooses direct, slope-normalized, or normalized kobs formula based on which inputs are available.",
        },
        "tangent_range": {
            "description": "'auto' is passed to peak_current so catalytic/reference current extraction uses automatic tangent-baseline selection.",
        },
    })

    scatter_fit_auto = {
        "x_unit": {
            "description": "'auto' auto scales the resolved x data using available units.",
        },
        "y_unit": {
            "description": "'auto' auto scales the resolved y data using available units.",
        },
        "species": {
            "description": "Species used when auto-resolving concentration-based x data from CV compounds/concentrations metadata.",
        },
        "fit_indices": {
            "description": "Explicit rows or index windows included in the fit; when omitted, all resolved points are included.",
        },
        "fit_range": {
            "description": "Single x-value window included in the fit, [x_min, x_max].",
        },
        "fit_ranges": {
            "description": "Named or unnamed x-value windows for multiple fits; each range is fitted and plotted separately.",
        },
        "fit_model": {
            "description": "Model fit on the resolved x/y values. Supported models are linear, power, power offset, exponential, michaelis menten, logistic, callables, and restricted formulas such as k0 + k1*x + k2*x^2.",
        },
        "fit_params": {
            "description": "Parameter names for a custom callable or formula model; formula strings infer names when omitted.",
        },
        "fit_equation": {
            "description": "Optional display equation for custom model printouts.",
        },
        "fit_init": {
            "description": "'auto' chooses model-specific initial guesses from the resolved data; a list or dict overrides one or more parameter guesses.",
        },
        "fit_bounds": {
            "description": "'auto' chooses broad model-specific bounds; a [lower, upper] pair or dict by parameter name overrides them.",
        },
        "fit_residual": {
            "description": "Residual used for direct model fitting: direct, relative, log, or log10.",
        },
        "fit_max_evals": {
            "description": "Maximum function evaluations for direct model fitting.",
        },
        "print_fit": {
            "description": "Fit print style: auto, summary, or details. Auto uses details for explicit init/bounds, custom or constrained models, 3+ parameters, or bound-hitting fits.",
        },
        "print_fit_details": {
            "description": "If True, force detailed two-table fit printing.",
        },
    }
    update("sevcik_analysis", {
        **cv_auto,
        **scatter_fit_auto,
        "c": {
            "description": "Concentration used for diffusion coefficient reporting; not inferred unless species-based concentration metadata are available to the analysis.",
        },
        "tangent_range": {
            "description": "'auto' is passed to peak_current so peak-current extraction uses automatic tangent-baseline selection.",
        },
    })
    update("fit_peak_current", {
        **cv_auto,
        **scatter_fit_auto,
        "x_power": {
            "description": "Power applied to auto-resolved x values; default 0.5 gives sqrt(scan rate) behavior.",
        },
        "tangent_range": {
            "description": "'auto' is passed to peak_current so peak-current extraction uses automatic tangent-baseline selection.",
        },
    })
    update("fit_peak_potential", {
        **cv_auto,
        **scatter_fit_auto,
        "follow_e1_2": {
            "description": "When True, follows paired peak-potential behavior using resolved E1/2-style values.",
        },
    })
    update("trumpet_analysis", {
        **cv_auto,
        "temperature": {
            "description": "Temperature in K. If omitted, uses each CV's temperature metadata before falling back to the option default.",
        },
        "d": {
            "description": "Diffusion coefficient in cm^2/s for ks reporting; not inferred.",
        },
        "tangent_range": {
            "description": "'auto' is passed to peak_current so peak-potential/current extraction uses automatic tangent-baseline selection.",
        },
    })
    update("nicholson", {
        **cv_auto,
        "fit_model": {
            "description": "Nicholson fit model. 'origin' uses the theoretical ψ = k0 x relation; 'linear' adds a fitted intercept as a diagnostic backup.",
        },
        "scan_rate": {
            "description": "Manual scan rate in V/s. If omitted, nicholson_analysis uses each CV's scan_rate metadata.",
        },
        "scan_rates": {
            "description": "Manual scan rates in V/s. If omitted, nicholson_analysis uses each CV's scan_rate metadata.",
        },
        "temperature": {
            "description": "Temperature in K. If omitted, nicholson_analysis uses each CV's temperature metadata before the option default.",
        },
        "d": {
            "description": "Diffusion coefficient in cm^2/s for heterogeneous rate reporting; not inferred.",
        },
        "fit_through_origin": {
            "description": "Legacy Nicholson fit toggle. When both are provided, 'fit model' wins: 'origin' constrains through the origin and 'linear' fits an intercept.",
        },
        "psi_source": {
            "description": "Source for Nicholson psi values: Agarwal table or empirical Lavagnini equation.",
        },
        "tangent_range": {
            "description": "'auto' is passed to peak_current so half-wave extraction uses automatic tangent-baseline selection.",
        },
    })
    update("fit_rate", {
        "x_column": {
            "description": "'auto' auto chooses a sensible x column: varying scan rate first, then concentration columns/compound metadata.",
        },
        "species": {
            "description": "Species used to filter concentration-column auto-resolution when x column is auto.",
        },
        "metric": {
            "description": "Metric column to fit; if omitted from plotting helpers, common rate/output metric columns are preferred.",
        },
        "fit_indices": {
            "description": "Explicit point indices included in the fit; when omitted, all resolved points are included unless fit ranges are supplied.",
        },
        "fit_ranges": {
            "description": "Named or unnamed x-value fit windows; when supplied, each window is fitted separately.",
        },
        "fit_model": {
            "description": "Model fit on resolved x/y values. Supported models are linear, power, power offset, exponential, michaelis menten, logistic, callables, and restricted formulas such as k0 + k1*x + k2*x^2.",
        },
        "fit_params": {
            "description": "Parameter names for a custom callable or formula model; formula strings infer names when omitted.",
        },
        "fit_equation": {
            "description": "Optional display equation for custom model printouts.",
        },
        "fit_init": {
            "description": "'auto' chooses model-specific initial guesses from the resolved data; a list or dict overrides one or more parameter guesses.",
        },
        "fit_bounds": {
            "description": "'auto' chooses broad model-specific bounds; a [lower, upper] pair or dict by parameter name overrides them.",
        },
        "fit_residual": {
            "description": "Residual used for direct model fitting: direct, relative, log, or log10.",
        },
        "fit_max_evals": {
            "description": "Maximum function evaluations for direct model fitting.",
        },
        "print_fit": {
            "description": "Fit print style: auto, summary, or details. Auto uses details for explicit init/bounds, custom or constrained models, 3+ parameters, or bound-hitting fits.",
        },
        "print_fit_details": {
            "description": "If True, force detailed two-table fit printing.",
        },
    })

    update("describe_options", {
        "print": {
            "description": "Whether to emit the options table or menu. False suppresses output but still allows dataframe return.",
        },
        "pretty_print": {
            "description": "Whether emitted output uses rich notebook display. False prints a plain text table; it does not suppress output.",
        },
        "return": {
            "description": "Whether to return the options table/menu as a pandas DataFrame in addition to any printed output.",
        },
    })

    return metadata


OPTION_METADATA = _build_option_metadata()


def _simple_toml_loads(text):
    data = {}
    section = None

    def parse_value(raw):
        raw = raw.strip()
        if raw == "null":
            return None
        if raw == "true":
            return True
        if raw == "false":
            return False
        if raw.startswith('"') and raw.endswith('"'):
            return raw[1:-1]
        if raw.startswith("[") and raw.endswith("]"):
            inner = raw[1:-1].strip()
            if not inner:
                return []
            return [parse_value(part.strip()) for part in inner.split(",")]
        try:
            if any(ch in raw for ch in ".eE"):
                return float(raw)
            return int(raw)
        except ValueError:
            return raw

    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("[") and line.endswith("]"):
            section = line[1:-1].strip()
            data[section] = {}
            continue
        if "=" in line and section is not None:
            key, value = line.split("=", 1)
            data[section][key.strip()] = parse_value(value)
    return data


def _toml_loads(text):
    if tomllib is None:
        return _simple_toml_loads(text)
    try:
        return tomllib.loads(text)
    except TOMLDecodeError:
        if re.search(r"=\s*null\b", text):
            return _simple_toml_loads(text)
        raise


def _load_toml_path(path):
    path = Path(path)
    return _toml_loads(path.read_text(encoding="utf-8"))


def _load_package_defaults():
    global _PACKAGE_DEFAULTS
    if _PACKAGE_DEFAULTS is None:
        default_file = resources.files("ecat").joinpath("defaults.toml")
        _PACKAGE_DEFAULTS = _toml_loads(default_file.read_text(encoding="utf-8"))
    return deepcopy(_PACKAGE_DEFAULTS)


def _normalize_mapping(mapping):
    normalized = {}
    for section, values in (mapping or {}).items():
        section_key = _canonical_section_key(section)
        normalized[section_key] = {
            _canonical_option_key(key): value
            for key, value in (values or {}).items()
        }
    return normalized


def _deep_merge(base, override):
    merged = deepcopy(base)
    for section, values in (override or {}).items():
        merged.setdefault(section, {})
        merged[section].update(values or {})
    return merged


def _option_default_registry():
    return [
        (ImportOptions, ["get_data"]),
        (PlotOptions, ["plot", "normalize_current"]),
        (MultiplotOptions, ["plot", "multiplot", "normalize_current"]),
        (MultiMultiplotOptions, ["plot", "multiplot", "multimultiplot", "normalize_current"]),
        (MultiScatterplotOptions, ["plot", "multiplot", "multi_scatterplot"]),
        (PeakPotentialOptions, ["plot", "cv_selection", "cv_analysis"]),
        (PeakCurrentOptions, ["plot", "cv_selection", "cv_analysis", "peak_current"]),
        (NormalizeOptions, ["normalize"]),
        (NormalizationOptions, ["cv_selection", "cv_analysis", "peak_current", "normalize_current"]),
        (ScaleCurrentOptions, ["cv_analysis", "peak_current", "scale_current"]),
        (FOWAOptions, ["plot", "multiplot", "cv_selection", "cv_analysis", "peak_current", "fowa"]),
        (FitRateOptions, ["fit_rate"]),
        (FitPeakPotentialOptions, ["plot", "cv_selection", "cv_analysis", "fit_peak_potential"]),
        (SevcikAnalysisOptions, ["plot", "cv_selection", "cv_analysis", "peak_current", "sevcik_analysis"]),
        (FitPeakCurrentOptions, ["plot", "cv_selection", "cv_analysis", "peak_current", "fit_peak_current"]),
        (TrumpetAnalysisOptions, ["plot", "cv_selection", "cv_analysis", "peak_current", "trumpet_analysis"]),
        (NicholsonOptions, ["plot", "cv_selection", "cv_analysis", "peak_current", "nicholson"]),
        (PlateauCurrentOptions, ["plot", "cv_selection", "cv_analysis", "peak_current", "fowa", "plateau_current"]),
        (SortGroupOptions, ["sort_group"]),
        (GroupSummaryOptions, ["group_summary"]),
        (TafelAnalysisOptions, ["tafel_analysis"]),
        (FilterOptions, ["filter"]),
    ]


def _known_global_option_names():
    names = set()
    for cls, _sections in _option_default_registry():
        names.update(_field_names(cls))
    return names


def _default_section_names():
    sections = set(_normalize_mapping(_load_package_defaults()).keys())
    sections.update(_USER_DEFAULTS.keys())
    sections.update(_SESSION_DEFAULTS.keys())
    return sections


def _unknown_global_option_error(key):
    norm = _canonical_option_key(key)
    suggestions = difflib.get_close_matches(norm, sorted(_known_global_option_names()), n=3)
    if len(suggestions) == 1:
        hint = f" Did you mean '{_friendly_key(suggestions[0])}'?"
    elif suggestions:
        friendly = "', '".join(_friendly_key(suggestion) for suggestion in suggestions)
        hint = f" Did you mean one of '{friendly}'?"
    else:
        hint = ""
    return OptionError(f"Unknown global option '{key}'.{hint}")


def _sections_for_global_option(option_key):
    option_key = _canonical_option_key(option_key)
    sections = []
    for cls, model_sections in _option_default_registry():
        if option_key not in _field_names(cls):
            continue
        for section in model_sections:
            if section not in sections:
                sections.append(section)
    if not sections:
        raise _unknown_global_option_error(option_key)
    return sections


def _expanded_global_option_defaults():
    expanded = {}
    for option_key, value in _GLOBAL_OPTION_DEFAULTS.items():
        for section in _sections_for_global_option(option_key):
            expanded.setdefault(section, {})[option_key] = value
    return expanded


def load_defaults(path):
    """Load user defaults from a TOML file for this Python session.
    
    Parameters
    ----------
    path : str or path-like
        TOML file containing eCAT defaults sections.
    
    Returns
    -------
    dict
        Merged defaults after loading the file.
    
    Examples
    --------
    >>> e.load_defaults("lab_ecat_defaults.toml")
    >>> e.describe_options("get_data")
    """
    global _USER_DEFAULTS
    _USER_DEFAULTS = _normalize_mapping(_load_toml_path(path))
    return get_defaults()


def set_defaults(section_or_mapping, updates=None):
    """Set runtime defaults for an eCAT options section or global option name.
    
    Parameters
    ----------
    section_or_mapping : str or mapping
        Section name, global option name, or mapping of sections to updates.
    updates : object, optional
        Updates for the section or value for a global option shorthand.
    
    Returns
    -------
    dict
        Current merged defaults after applying the update.
    
    Examples
    --------
    >>> e.set_defaults("plot", {"legend mode": "colorbar"})
    >>> e.describe_options("multiplot")
    """
    global _SESSION_DEFAULTS, _GLOBAL_OPTION_DEFAULTS
    if updates is None:
        mapping = _normalize_mapping(section_or_mapping)
    elif isinstance(updates, dict):
        mapping = _normalize_mapping({section_or_mapping: updates})
    else:
        option_key = _canonical_option_key(section_or_mapping)
        _sections_for_global_option(option_key)
        _GLOBAL_OPTION_DEFAULTS[option_key] = updates
        return get_defaults()
    _SESSION_DEFAULTS = _deep_merge(_SESSION_DEFAULTS, mapping)
    return get_defaults()


def get_defaults(section=None):
    """Return merged eCAT defaults for all sections or one section.
    
    Parameters
    ----------
    section : str, optional
        Public defaults section name, such as ``"get_data"`` or ``"multiplot"``.
    
    Returns
    -------
    dict
        Defaults after package, user, session, and global overrides are merged.
    
    Examples
    --------
    >>> e.get_defaults("fowa")
    >>> e.describe_options("fowa")
    """
    defaults = _load_package_defaults()
    defaults = _deep_merge(defaults, _USER_DEFAULTS)
    defaults = _deep_merge(defaults, _expanded_global_option_defaults())
    defaults = _deep_merge(defaults, _SESSION_DEFAULTS)
    defaults = _normalize_mapping(defaults)
    if section is None:
        return defaults
    return deepcopy(defaults.get(_canonical_section_key(section), {}))


def reset_defaults(section=None):
    """Reset user-loaded and session defaults for all sections or one target.
    
    Parameters
    ----------
    section : str, optional
        Section or global option shorthand to reset.
    
    Returns
    -------
    dict
        Current defaults after reset.
    
    Examples
    --------
    >>> e.reset_defaults()
    >>> e.describe_options("plot")
    """
    global _USER_DEFAULTS, _SESSION_DEFAULTS, _GLOBAL_OPTION_DEFAULTS
    if section is None:
        _USER_DEFAULTS = {}
        _SESSION_DEFAULTS = {}
        _GLOBAL_OPTION_DEFAULTS = {}
        return get_defaults()
    else:
        section_key = _canonical_section_key(section)
        if section_key in _default_section_names():
            _USER_DEFAULTS.pop(section_key, None)
            _SESSION_DEFAULTS.pop(section_key, None)
        else:
            reset_defaults_option(section_key)
        return get_defaults(section_key)


def reset_defaults_option(option):
    """Reset a runtime shorthand default from every section that supports an option."""
    global _GLOBAL_OPTION_DEFAULTS, _SESSION_DEFAULTS
    option_key = _canonical_option_key(option)
    _sections_for_global_option(option_key)
    _GLOBAL_OPTION_DEFAULTS.pop(option_key, None)
    for section_values in _SESSION_DEFAULTS.values():
        section_values.pop(option_key, None)
    return get_defaults()


def reset_defaults_section(section):
    """Explicitly reset one defaults section."""
    global _USER_DEFAULTS, _SESSION_DEFAULTS
    section_key = _canonical_section_key(section)
    _USER_DEFAULTS.pop(section_key, None)
    _SESSION_DEFAULTS.pop(section_key, None)
    return get_defaults(section_key)


def _field_names(cls):
    return {field.name for field in fields(cls)}


def _coerce_options(cls, raw, sections):
    if isinstance(raw, cls):
        return raw
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise TypeError(f"{cls.__name__} accepts a dict or {cls.__name__}, not {type(raw).__name__}.")

    valid = _field_names(cls)
    normalized = {}
    original_keys = {}
    for key, value in raw.items():
        if str(key).startswith("_") and str(key) != "_provided_options":
            continue
        norm = _canonical_option_key(key)
        if norm not in valid:
            suggestions = difflib.get_close_matches(norm, sorted(valid), n=3)
            if len(suggestions) == 1:
                hint = f" Did you mean '{_friendly_key(suggestions[0])}'?"
            elif suggestions:
                friendly = "', '".join(_friendly_key(suggestion) for suggestion in suggestions)
                hint = f" Did you mean one of '{friendly}'?"
            else:
                hint = ""
            raise OptionError(f"Unknown option '{key}' for {cls.__name__}.{hint}")
        normalized[norm] = value
        original_keys[norm] = key

    defaults = {}
    for section in sections:
        defaults.update(get_defaults(section))

    kwargs = {
        name: defaults.get(name, field.default)
        for name, field in ((field.name, field) for field in fields(cls))
        if field.default is not MISSING
    }
    kwargs.update(normalized)
    if "_provided_options" in valid:
        kwargs["_provided_options"] = tuple(normalized)
    if "segments" in normalized and "segment" not in normalized and "segment" in valid:
        kwargs["segment"] = None
    if "segment" in normalized and "segments" not in normalized and "segments" in valid:
        kwargs["segments"] = None
    opts = cls(**kwargs)
    opts.validate()
    return opts


def _validate_common_cv(opts):
    if opts.segment is not None and opts.segments is not None:
        raise OptionError("Use either 'segment' or 'segments', not both.")
    if opts.exact_potential is not None and opts.guess_potential is not None:
        raise OptionError("Use either 'exact potential' or 'guess potential', not both.")
    if opts.noise_window != "auto":
        if not isinstance(opts.noise_window, int) or opts.noise_window < 3 or opts.noise_window % 2 == 0:
            raise OptionError("'noise window' must be 'auto' or an odd integer >= 3.")
    if (
        opts.noise_polyorder != "auto"
        and opts.noise_window != "auto"
        and int(opts.noise_polyorder) >= int(opts.noise_window)
    ):
        raise OptionError("'noise polyorder' must be less than 'noise window'.")


@dataclass(frozen=True, slots=True)
class ImportOptions:
    folder_path: str = "."
    delimiter: str = ","
    decimal: str = "."
    columns: int = 3
    software: str | None = None
    experiment_type: str | None = None
    custom_reader: object | None = None
    print: bool = False
    troubleshoot: bool = False
    recursive_search: bool = True
    name_alterations: object | None = None
    pretty_print: bool = True
    sort_keys: object = field(default_factory=lambda: ["timestamp"])
    reference_mode: str = "auto"
    reference_keywords: list[str] | None = None
    reference_keyword: str | None = None
    reference_file: str | None = None
    reference_map: dict | None = None
    reference_offset: float | None = None
    reference_guess: float | str | None = "auto"
    reference_label: str = "Fc/Fc+"
    allow_self_reference: bool = True
    shift_potential: bool | str | float = False
    shift_guess: float | str | None = "auto"
    shift_label: str = "Fc/Fc+"
    shift_window: float = 0.3
    shift_smooth: bool = True
    shift_max_delta_ep: float = 0.20
    shift_target_delta_ep: float = 0.08
    peak_prominence: float | None = None
    compounds: object | None = None
    gas: str | None = None
    solvent: str | None = None
    temperature: float = 298
    convert_current: bool = False
    electrode_diameter: float = 0
    electrode_area: float = 0
    current_density: bool = False
    invert_current: bool = False
    scan_rate: float | None = None
    _provided_options: tuple[str, ...] = field(default=(), repr=False, compare=False)

    @classmethod
    def from_options(cls, options=None):
        return _coerce_options(cls, options, ["get_data"])

    def validate(self):
        if not isinstance(self.recursive_search, bool):
            raise OptionError("'recursive search' must be True or False.")
        if self.sort_keys is not None and not isinstance(self.sort_keys, (str, list, tuple)):
            raise OptionError("'sort keys' must be a string, list, tuple, or None.")
        allowed_modes = {"none", "off", "false", "manual", "keyword", "auto", "file"}
        if str(self.reference_mode).strip().lower() not in allowed_modes:
            raise OptionError("'reference mode' must be 'auto', 'manual', 'keyword', 'file', or 'none'.")
        if self.reference_map is not None:
            if not isinstance(self.reference_map, dict):
                raise OptionError("'reference map' must be a dictionary such as {45: 54}.")
            for target_idx, reference_idx in self.reference_map.items():
                if not isinstance(target_idx, int) or not isinstance(reference_idx, int):
                    raise OptionError("'reference map' keys and values must be integer object indices.")
        if self.current_density and not (self.electrode_area or self.electrode_diameter):
            raise OptionError("'current density' requires 'electrode area' or 'electrode diameter'.")

    def to_legacy_dict(self):
        data = {
            field.name.replace("_", " "): getattr(self, field.name)
            for field in fields(self)
            if not field.name.startswith("_")
        }
        data["folder path"] = self.folder_path
        data["experiment type"] = self.experiment_type
        data["custom reader"] = self.custom_reader
        data["recursive search"] = self.recursive_search
        data["name alterations"] = self.name_alterations
        data["pretty print"] = self.pretty_print
        data["sort keys"] = self.sort_keys
        data["gas"] = self.gas
        data["solvent"] = self.solvent
        data["reference mode"] = self.reference_mode
        data["reference keywords"] = self.reference_keywords
        data["reference keyword"] = self.reference_keyword
        data["reference file"] = self.reference_file
        data["reference map"] = self.reference_map
        data["reference offset"] = self.reference_offset
        data["reference guess"] = self.reference_guess
        data["reference label"] = self.reference_label
        data["allow self reference"] = self.allow_self_reference
        data["shift potential"] = self.shift_potential
        data["shift guess"] = self.shift_guess
        data["shift label"] = self.shift_label
        data["shift window"] = self.shift_window
        data["shift smooth"] = self.shift_smooth
        data["shift max delta ep"] = self.shift_max_delta_ep
        data["shift target delta ep"] = self.shift_target_delta_ep
        data["peak prominence"] = self.peak_prominence
        data["convert current"] = self.convert_current
        data["electrode diameter"] = self.electrode_diameter
        data["electrode area"] = self.electrode_area
        data["_electrode area provided"] = "electrode_area" in self._provided_options
        data["current density"] = self.current_density
        data["invert current"] = self.invert_current
        data["scan rate"] = self.scan_rate
        return data


def import_options_to_legacy_dict(options=None):
    if isinstance(options, dict) and "_electrode area provided" in options:
        return dict(options)
    return ImportOptions.from_options(options).to_legacy_dict()


@dataclass(frozen=True, slots=True)
class TrimOptions:
    potential_window: object | None = None
    mode: str | None = None
    inplace: bool = False
    x_axis: object | None = None

    @classmethod
    def from_options(cls, options=None):
        return _coerce_options(cls, options, [])

    @staticmethod
    def _canonical_mode(value):
        return str(value).strip().lower().replace("_", " ").replace("-", " ")

    def validate(self):
        if self.potential_window is None:
            raise OptionError("cv.trim requires 'potential window'.")
        try:
            values = list(self.potential_window)
        except TypeError as exc:
            raise OptionError("'potential window' must be a two-value sequence like [start, stop].") from exc
        if len(values) != 2:
            raise OptionError("'potential window' must contain exactly two values.")
        try:
            float(values[0])
            float(values[1])
        except (TypeError, ValueError) as exc:
            raise OptionError("'potential window' values must be numeric.") from exc

        mode = self._canonical_mode("expand" if self.mode is None else self.mode)
        if mode not in {"expand", "pointwise", "strict"}:
            raise OptionError("'mode' must be 'expand', 'pointwise', or 'strict'.")

    def to_legacy_dict(self):
        mode = self._canonical_mode("expand" if self.mode is None else self.mode)
        return {
            "potential window": self.potential_window,
            "mode": mode,
            "inplace": self.inplace,
            "x axis": self.x_axis,
        }


@dataclass(frozen=True, slots=True)
class PlotOptions:
    plot: bool = True
    print: bool = True
    plot_all: bool = False
    print_all: bool = False
    new_plot: bool = False
    label: str | None = None
    legend: bool | str = "auto"
    title: bool | str = True
    subtitle: bool | str | None = "auto"
    color: str | None = "black"
    default_colors: list[str] = field(
        default_factory=lambda: ["black", "tab:blue", "tab:red", "tab:green", "tab:orange", "tab:purple"]
    )
    default_gradient_colormap: str = "viridis"
    color_mode: str = "auto"
    gradient_by: str = "auto"
    gradient_species: str = "auto"
    gradient_scale: str = "auto"
    gradient_colormap: str | None = None
    gradient_colormaps: list[str] = field(default_factory=list)
    gradient_colors: list[str] = field(default_factory=list)
    gradient_gamma: float = 1.0
    gradient_reverse: bool = False
    min_gradient_entries: int = 3
    colorbar_height_scale: float = 1.0
    colorbar_reverse: bool = True
    colorbar_style: str = "auto"
    colorbar_tick_length: int = 5
    colorbar_tick_pad: int = 8
    colorbar_tick_labels: str = "endpoints"
    colorbar_trace_ticks: bool = True
    legend_mode: str = "auto"
    y_col: int = -1
    y_flip: bool = False
    invert_y: bool = False
    plot_convention: str = "US"
    sig_figs: int = 4
    offset: float = 0
    stacking: bool = False
    label_alterations: object | None = None
    legend_sample_length: float | str = "auto"
    legend_fontsize: int | None = None
    legend_loc: str = "auto"
    legend_outside: bool = False
    legend_pad: float = 0.02
    grid: bool = False
    title_fontsize: int | str | None = None
    subtitle_fontsize: int | str | None = None
    x_unit: str | None = "auto"
    y_unit: str | None = "auto"
    x_scale: float | int = 1
    y_scale: float | int = 1
    xlabel: str | None = None
    ylabel: str | None = None
    x_axis: str | None = None
    y_axis: str | None = None
    x_column_index: int = -1
    y_column_index: int = -1
    derivative: int | float | None = 0
    smooth: bool = False
    noise_window: int | str = "auto"
    noise_polyorder: int | str = "auto"
    one_column: bool = True
    cycles: object = None
    plot_segment: int | None = None
    plot_segments: list[int] | int | None = None
    segment_color_mode: str = "auto"
    segment_color_groups: object = 2
    integrate: bool = False
    plot_charge: bool = False
    plot_ca: bool = True
    plot_target: bool = True
    target_charge: float | None = None
    target_moles: float | None = None
    target_electrons: float | None = None
    target_label: str | None = None
    charge_color: str = "tab:red"
    baseline_correction: bool | str = False
    baseline_threshold: float | None = None
    baseline_tail_fraction: float = 0.05
    animate: bool = False
    animate_minrate: float | int = 0
    animate_repeat: bool = False
    scan_rate: float | None = None
    normalize: bool = False
    normalize_params: dict | None = None
    ip0: float | list[float] | None = None
    reference_index: int = 0
    reference_cv: object | None = None
    reference_cvs: object | None = None
    plot_options: dict | None = None
    scale_bar: object = False
    minor_ticks: bool | int = 2
    symbol_labels: bool | str = "auto"

    @classmethod
    def from_options(cls, options=None):
        return _coerce_options(cls, options, ["plot", "normalize_current"])

    def validate(self):
        sources = [
            self.ip0 is not None,
            self.reference_cv is not None,
            self.reference_cvs is not None,
            self.reference_index != 0,
        ]
        if sum(bool(source) for source in sources) > 1:
            raise OptionError("Use only one ip0/reference source for i/ip0 plotting.")
        mode = str(self.segment_color_mode).strip().lower().replace("_", " ").replace("-", " ")
        if mode in {"none", "false", "0"}:
            mode = "off"
        if mode not in {"auto", "off", "discrete", "discrete gradient", "continuous gradient"}:
            raise OptionError(
                "'segment color mode' must be 'auto', 'off', 'discrete', "
                "'discrete gradient', or 'continuous gradient'."
            )
        legend_mode = str(self.legend_mode).strip().lower()
        if legend_mode not in {"auto", "colorbar", "discrete"}:
            raise OptionError("'legend mode' must be 'auto', 'colorbar', or 'discrete'.")
        color_mode = str(self.color_mode).strip().lower()
        if color_mode not in {"auto", "gradient", "discrete"}:
            raise OptionError("'color mode' must be 'auto', 'gradient', or 'discrete'.")
        gradient_scale = str(self.gradient_scale).strip().lower()
        if gradient_scale not in {"auto", "linear", "sqrt", "log", "index"}:
            raise OptionError("'gradient scale' must be 'auto', 'linear', 'sqrt', 'log', or 'index'.")
        colorbar_style = str(self.colorbar_style).strip().lower().replace("_", " ").replace("-", " ")
        if colorbar_style not in {"auto", "continuous", "discrete", "swatch", "swatches"}:
            raise OptionError("'colorbar style' must be 'auto', 'continuous', or 'discrete'.")
        baseline_correction = self.baseline_correction
        if isinstance(baseline_correction, str):
            baseline_correction = baseline_correction.strip().lower().replace("_", " ").replace("-", " ")
            if baseline_correction not in {"tail", "threshold", "true", "false", "on", "off"}:
                raise OptionError(
                    "'baseline correction' must be True, False, 'tail', or 'threshold'."
                )
        try:
            if float(self.baseline_tail_fraction) <= 0 or float(self.baseline_tail_fraction) > 1:
                raise ValueError
        except (TypeError, ValueError):
            raise OptionError("'baseline tail fraction' must be a number between 0 and 1.")
        if self.minor_ticks is not True and self.minor_ticks is not False:
            try:
                if int(self.minor_ticks) < 0:
                    raise ValueError
            except (TypeError, ValueError):
                raise OptionError("'minor ticks' must be True, False, or a non-negative integer.")
        symbol_labels = str(self.symbol_labels).strip().lower()
        if self.symbol_labels is not True and self.symbol_labels is not False and symbol_labels != "auto":
            raise OptionError("'symbol labels' must be True, False, or 'auto'.")

    def to_legacy_dict(self):
        data = {
            item.name.replace("_", " "): getattr(self, item.name)
            for item in fields(self)
            if not item.name.startswith("_")
        }
        data["plot all"] = self.plot_all
        data["print all"] = self.print_all
        data["new plot"] = self.new_plot
        data["y col"] = self.y_col
        data["y flip"] = self.y_flip
        data["invert y"] = self.invert_y
        data["plot convention"] = self.plot_convention
        data["sig figs"] = self.sig_figs
        data["label alterations"] = self.label_alterations
        data["legend sample length"] = self.legend_sample_length
        data["legend fontsize"] = self.legend_fontsize
        data["legend loc"] = self.legend_loc
        data["legend mode"] = self.legend_mode
        data["legend outside"] = self.legend_outside
        data["legend pad"] = self.legend_pad
        data["default colors"] = list(self.default_colors)
        data["default gradient colormap"] = self.default_gradient_colormap
        data["color mode"] = self.color_mode
        data["gradient by"] = self.gradient_by
        data["gradient species"] = self.gradient_species
        data["gradient scale"] = self.gradient_scale
        data["gradient colormap"] = self.gradient_colormap
        data["gradient colormaps"] = list(self.gradient_colormaps)
        data["gradient colors"] = list(self.gradient_colors)
        data["gradient gamma"] = self.gradient_gamma
        data["gradient reverse"] = self.gradient_reverse
        data["min gradient entries"] = self.min_gradient_entries
        data["colorbar height scale"] = self.colorbar_height_scale
        data["colorbar reverse"] = self.colorbar_reverse
        data["colorbar style"] = self.colorbar_style
        data["colorbar tick length"] = self.colorbar_tick_length
        data["colorbar tick pad"] = self.colorbar_tick_pad
        data["colorbar tick labels"] = self.colorbar_tick_labels
        data["colorbar trace ticks"] = self.colorbar_trace_ticks
        data["title fontsize"] = self.title_fontsize
        data["subtitle fontsize"] = self.subtitle_fontsize
        data["x unit"] = self.x_unit
        data["y unit"] = self.y_unit
        data["x scale"] = self.x_scale
        data["y scale"] = self.y_scale
        data["x axis"] = self.x_axis
        data["y axis"] = self.y_axis
        data["x column index"] = self.x_column_index
        data["y column index"] = self.y_column_index
        data["noise window"] = self.noise_window
        data["noise polyorder"] = self.noise_polyorder
        data["one column"] = self.one_column
        data["cycles"] = self.cycles
        data["plot segment"] = self.plot_segment
        data["plot segments"] = self.plot_segments
        data["segment color mode"] = self.segment_color_mode
        data["segment color groups"] = self.segment_color_groups
        data["plot charge"] = self.plot_charge
        data["plot ca"] = self.plot_ca
        data["plot target"] = self.plot_target
        data["target charge"] = self.target_charge
        data["target moles"] = self.target_moles
        data["target electrons"] = self.target_electrons
        data["target label"] = self.target_label
        data["charge color"] = self.charge_color
        data["baseline correction"] = self.baseline_correction
        data["baseline threshold"] = self.baseline_threshold
        data["baseline tail fraction"] = self.baseline_tail_fraction
        data["animate minrate"] = self.animate_minrate
        data["animate repeat"] = self.animate_repeat
        data["scan rate"] = self.scan_rate
        data["normalize params"] = self.normalize_params
        data["reference index"] = self.reference_index
        data["reference cv"] = self.reference_cv
        data["reference cvs"] = self.reference_cvs
        data["plot options"] = {} if self.plot_options is None else dict(self.plot_options)
        return data


@dataclass(frozen=True, slots=True)
class MultiplotOptions(PlotOptions):
    _provided_options: tuple[str, ...] = field(default=(), repr=False, compare=False)
    print: bool = False
    legend: bool = True
    title: bool | str = "auto"
    subtitle: bool | str | None = "auto"
    labels: list[str] | None = None
    plot_labels: list[str] | None = None
    deduplicate_labels: object = False
    legend_bbox_to_anchor: object | None = None
    titles: list[str] | str | None = "auto"
    subtitles: list[str] | str | None = "auto"
    default_colors: list[str] = field(
        default_factory=lambda: ["black", "tab:blue", "tab:red", "tab:green", "tab:orange", "tab:purple"]
    )
    default_discrete_colormap: str = "tab20"
    default_gradient_colormap: str = "viridis"
    color_mode: str = "auto"
    legend_mode: str = "auto"
    min_gradient_entries: int = 3
    colorbar_height_scale: float = 1.0
    colorbar_reverse: bool = True
    colorbar_tick_length: int = 5
    colorbar_tick_pad: int = 8
    gradient_by: str = "auto"
    gradient_species: str = "auto"
    gradient_scale: str = "auto"
    gradient_colormap: str | None = None
    gradient_colormaps: list[str] = field(default_factory=list)
    gradient_colors: list[str] = field(default_factory=list)
    gradient_gamma: float = 1.0
    gradient_reverse: bool = False
    colorbar_tick_labels: str = "endpoints"
    colorbar_trace_ticks: bool = True

    @classmethod
    def from_options(cls, options=None):
        return _coerce_options(cls, options, ["plot", "multiplot", "normalize_current"])

    def to_legacy_dict(self):
        data = PlotOptions.to_legacy_dict(self)
        data["_deduplicate labels explicit"] = "deduplicate_labels" in self._provided_options
        data["plot labels"] = self.plot_labels
        data["deduplicate labels"] = self.deduplicate_labels
        data["legend loc"] = self.legend_loc
        data["legend outside"] = self.legend_outside
        data["legend pad"] = self.legend_pad
        data["legend bbox to anchor"] = self.legend_bbox_to_anchor
        data["titles"] = self.titles
        data["subtitles"] = self.subtitles
        data["default colors"] = list(self.default_colors)
        data["default discrete colormap"] = self.default_discrete_colormap
        data["default gradient colormap"] = self.default_gradient_colormap
        data["color mode"] = self.color_mode
        data["legend mode"] = self.legend_mode
        data["min gradient entries"] = self.min_gradient_entries
        data["colorbar height scale"] = self.colorbar_height_scale
        data["colorbar reverse"] = self.colorbar_reverse
        data["colorbar tick length"] = self.colorbar_tick_length
        data["colorbar tick pad"] = self.colorbar_tick_pad
        data["gradient by"] = self.gradient_by
        data["gradient species"] = self.gradient_species
        data["gradient scale"] = self.gradient_scale
        data["gradient colormap"] = self.gradient_colormap
        data["gradient colormaps"] = list(self.gradient_colormaps)
        data["gradient colors"] = list(self.gradient_colors)
        data["gradient gamma"] = self.gradient_gamma
        data["gradient reverse"] = self.gradient_reverse
        data["colorbar tick labels"] = self.colorbar_tick_labels
        data["colorbar trace ticks"] = self.colorbar_trace_ticks
        return data


@dataclass(frozen=True, slots=True)
class MultiMultiplotOptions(MultiplotOptions):
    titles: list[str] | str | None = "auto"
    subtitles: list[str] | str | None = "auto"
    analysis: bool = False
    Ehalf: float | None = None

    @classmethod
    def from_options(cls, options=None):
        return _coerce_options(cls, options, ["plot", "multiplot", "multimultiplot", "normalize_current"])

    def to_legacy_dict(self):
        data = MultiplotOptions.to_legacy_dict(self)
        data["titles"] = self.titles
        data["subtitles"] = self.subtitles
        data["analysis"] = self.analysis
        data["Ehalf"] = self.Ehalf
        return data


@dataclass(frozen=True, slots=True)
class MultiScatterplotOptions(MultiplotOptions):
    print: bool = True
    x_column: str | int | None = "auto"
    y_column: str | int | None = "auto"
    y_columns: object | None = None
    metric: str | None = None
    data_mode: str = "auto"
    plot_style: str = "scatter"
    xscale: str | None = None
    yscale: str | None = None
    plot_scale: str | None = None
    plot_fit: bool = True
    fit_linestyle: str = "--"
    fit_linewidth: int | float = 1
    fit_alpha: int | float = 1

    @classmethod
    def from_options(cls, options=None):
        return _coerce_options(cls, options, ["plot", "multiplot", "multi_scatterplot"])

    def validate(self):
        data_mode = str(self.data_mode).strip().lower()
        if data_mode not in {"auto", "raw", "adjusted", "transformed"}:
            raise OptionError("'data mode' must be 'auto', 'raw', 'adjusted', or 'transformed'.")
        style = str(self.plot_style).strip().lower()
        if style not in {"scatter", "line", "line+markers", "line markers", "line-and-markers"}:
            raise OptionError("'plot style' must be 'scatter', 'line', or 'line+markers'.")
        if self.plot_scale is not None:
            scale = str(self.plot_scale).strip().lower().replace("_", "-").replace(" ", "-")
            if scale not in {"linear", "log-log", "loglog", "semilogx", "semi-log-x", "semilogy", "semi-log-y", "symlog"}:
                raise OptionError("'plot scale' must be 'linear', 'log-log', 'semilogx', 'semilogy', or 'symlog'.")
        if not 0 <= float(self.fit_alpha) <= 1:
            raise OptionError("'fit alpha' must be between 0 and 1.")
        if float(self.fit_linewidth) < 0:
            raise OptionError("'fit linewidth' must be non-negative.")

    def to_legacy_dict(self):
        data = MultiplotOptions.to_legacy_dict(self)
        data["x column"] = self.x_column
        data["y column"] = self.y_column
        data["y columns"] = self.y_columns
        data["metric"] = self.metric
        data["data mode"] = self.data_mode
        data["plot style"] = self.plot_style
        data["xscale"] = self.xscale
        data["yscale"] = self.yscale
        data["plot scale"] = self.plot_scale
        data["plot fit"] = self.plot_fit
        data["fit linestyle"] = self.fit_linestyle
        data["fit linewidth"] = self.fit_linewidth
        data["fit alpha"] = self.fit_alpha
        return data


@dataclass(frozen=True, slots=True)
class PeakPotentialOptions:
    plot: bool = True
    print: bool = True
    pretty_print: bool = True
    plot_all: bool = False
    print_all: bool = False
    x_axis: str | None = None
    y_axis: str | None = None
    x_unit: str | None = "auto"
    y_unit: str | None = "auto"
    xlabel: str | None = None
    ylabel: str | None = None
    new_plot: bool = False
    plot_cv: bool = True
    derivative: int | float | None = 0
    plot_segment: int | None = None
    plot_segments: list[int] | int | None = None
    segment: int | None = None
    segments: list[int] | None = None
    noise_window: int | str = "auto"
    noise_polyorder: int | str = "auto"
    sig_figs: int = 4
    peak_prominence: float | None = None
    guess_potential: float | None = None
    exact_potential: float | None = None
    troubleshoot: bool = False
    internal_call: bool = False
    offset: float = 0
    normalize: bool = False

    @classmethod
    def from_options(cls, options=None):
        return _coerce_options(cls, options, ["plot", "cv_selection", "cv_analysis"])

    def validate(self):
        _validate_common_cv(self)

    def to_legacy_dict(self):
        data = {field.name.replace("_", " "): getattr(self, field.name) for field in fields(self)}
        data["sig figs"] = self.sig_figs
        data["peak prominence"] = self.peak_prominence
        data["guess potential"] = self.guess_potential
        data["exact potential"] = self.exact_potential
        data["noise window"] = self.noise_window
        data["noise polyorder"] = self.noise_polyorder
        data["x axis"] = self.x_axis
        data["y axis"] = self.y_axis
        data["x unit"] = self.x_unit
        data["y unit"] = self.y_unit
        data["new plot"] = self.new_plot
        data["plot segment"] = self.plot_segment
        data["plot segments"] = self.plot_segments
        data["internal call"] = self.internal_call
        data["pretty print"] = self.pretty_print
        data["plot all"] = self.plot_all
        data["print all"] = self.print_all
        return data


@dataclass(frozen=True, slots=True)
class PeakCurrentOptions(PeakPotentialOptions):
    tangent_range: str | float | list[float] | tuple[float, float] = "auto"
    tangent_min_points: int | None = None
    tangent_potential: float | None = None
    percent_threshold: float | None = None
    plot_peak_potential: bool = True
    peak_fallback: str | None = "highest current"

    @classmethod
    def from_options(cls, options=None):
        return _coerce_options(cls, options, ["plot", "cv_selection", "cv_analysis", "peak_current"])

    def validate(self):
        _validate_common_cv(self)
        if self.peak_fallback is None:
            return
        fallback = str(self.peak_fallback).strip().lower().replace("_", " ").replace("-", " ")
        valid = {
            "highest current",
            "highest absolute current",
            "max current",
            "max absolute current",
            "abs current",
            "guess potential",
            "exact potential",
            "none",
            "error",
            "raise",
        }
        if fallback not in valid:
            raise OptionError(
                "'peak fallback' must be 'highest current', 'guess potential', or None/'none'."
            )

    def for_peak_potential(self):
        return PeakPotentialOptions(
            **{
                field.name: getattr(self, field.name)
                for field in fields(PeakPotentialOptions)
            }
        )

    def to_legacy_dict(self):
        data = PeakPotentialOptions.to_legacy_dict(self)
        data["tangent range"] = self.tangent_range
        data["tangent min points"] = self.tangent_min_points
        data["tangent potential"] = self.tangent_potential
        data["percent threshold"] = self.percent_threshold
        data["plot peak potential"] = self.plot_peak_potential
        data["peak fallback"] = self.peak_fallback
        return data


@dataclass(frozen=True, slots=True)
class NormalizeOptions:
    print: bool = False
    pretty_print: bool = True
    mode: str = "homogeneous"
    e0: float | None = None
    n: float = 1
    num_electrons: float | None = None
    temperature: float | None = None
    d: float | None = None
    c: float | str | None = None
    c_unit: str | None = None
    species: str | None = None
    area: float | None = None
    s: float | None = None
    electrode_area: float | None = None
    scan_rate: float | None = None
    v: float | None = None
    k_homo: float | None = None
    k0: float | None = None

    @classmethod
    def from_options(cls, options=None):
        return _coerce_options(cls, options, ["normalize"])

    def validate(self):
        mode = str(self.mode).strip().lower()
        if mode not in {"homogeneous", "heterogeneous"}:
            raise OptionError("'mode' must be 'homogeneous' or 'heterogeneous'.")
        n_value = self.num_electrons if self.num_electrons is not None else self.n
        if n_value is None or float(n_value) <= 0:
            raise OptionError("'n' must be positive.")

    def to_legacy_dict(self):
        data = {
            field.name.replace("_", " "): getattr(self, field.name)
            for field in fields(self)
            if not field.name.startswith("_")
        }
        data["E0"] = self.e0
        data["D"] = self.d
        data["C"] = self.c
        data["C unit"] = self.c_unit
        data["electrode area"] = self.electrode_area
        data["scan rate"] = self.scan_rate
        data["num electrons"] = self.num_electrons
        data["k homo"] = self.k_homo
        return data


@dataclass(frozen=True, slots=True)
class NormalizationOptions:
    print: bool = True
    plot_all: bool = False
    pretty_print: bool = True
    print_conditions: bool = True
    legend: bool = True
    title: bool | str = "auto"
    subtitle: bool | str | None = "auto"
    labels: list[str] | None = None
    plot_labels: list[str] | None = None
    segment: int | None = None
    segments: list[int] | int | None = None
    noise_window: int | str = "auto"
    noise_polyorder: int | str = "auto"
    sig_figs: int = 4
    peak_prominence: float | None = None
    guess_potential: float | None = None
    exact_potential: float | None = None
    tangent_range: str | float | list[float] | tuple[float, float] = "auto"
    tangent_min_points: int | None = None
    tangent_potential: float | None = None
    percent_threshold: float | None = None
    ip0: float | list[float] | None = None
    reference_index: int = 0
    reference_cv: object | None = None
    reference_cvs: object | None = None
    reference_guess_potential: float | None = None

    @classmethod
    def from_options(cls, options=None):
        return _coerce_options(cls, options, ["cv_selection", "cv_analysis", "peak_current", "normalize_current"])

    def validate(self):
        _validate_common_cv(self)
        if not isinstance(self.reference_index, int):
            raise OptionError("'reference index' must be an integer.")
        sources = [
            self.ip0 is not None,
            self.reference_cv is not None,
            self.reference_cvs is not None,
            self.reference_index != 0,
        ]
        if sum(bool(source) for source in sources) > 1:
            raise OptionError("Use only one ip0/reference source for normalize_current.")

    def for_peak_current(self):
        return PeakCurrentOptions.from_options(
            {
                "plot": False,
                "print": False,
                "segment": self.segment,
                "segments": self.segments,
                "noise window": self.noise_window,
                "noise polyorder": self.noise_polyorder,
                "sig figs": self.sig_figs,
                "peak prominence": self.peak_prominence,
                "guess potential": (
                    self.reference_guess_potential
                    if self.reference_guess_potential is not None
                    else self.guess_potential
                ),
                "exact potential": self.exact_potential,
                "tangent range": self.tangent_range,
                "tangent min points": self.tangent_min_points,
                "tangent potential": self.tangent_potential,
                "percent threshold": self.percent_threshold,
                "internal call": True,
            }
        )

    def to_legacy_dict(self):
        data = {field.name.replace("_", " "): getattr(self, field.name) for field in fields(self)}
        data["print"] = self.print
        data["plot all"] = self.plot_all
        data["pretty print"] = self.pretty_print
        data["print conditions"] = self.print_conditions
        data["legend"] = self.legend
        data["title"] = self.title
        data["subtitle"] = self.subtitle
        data["labels"] = self.labels
        data["plot labels"] = self.plot_labels
        data["noise window"] = self.noise_window
        data["noise polyorder"] = self.noise_polyorder
        data["sig figs"] = self.sig_figs
        data["peak prominence"] = self.peak_prominence
        data["guess potential"] = self.guess_potential
        data["exact potential"] = self.exact_potential
        data["tangent range"] = self.tangent_range
        data["tangent min points"] = self.tangent_min_points
        data["tangent potential"] = self.tangent_potential
        data["percent threshold"] = self.percent_threshold
        data["reference index"] = self.reference_index
        data["reference cv"] = self.reference_cv
        data["reference cvs"] = self.reference_cvs
        data["reference guess potential"] = self.reference_guess_potential
        return data


@dataclass(frozen=True, slots=True)
class ScaleCurrentOptions:
    print: bool = False
    plot_all: bool = False
    pretty_print: bool = True
    print_conditions: bool = True
    scale: float | list[float] | None = None
    reference_mode: str = "single"
    reference_index: int = 0
    reference_cv: object | None = None
    reference_cvs: object | None = None
    segment: int | None = 1
    segments: list[int] | int | None = None
    noise_window: int | str = "auto"
    noise_polyorder: int | str = "auto"
    sig_figs: int = 4
    peak_prominence: float | None = None
    guess_potential: float | None = None
    exact_potential: float | None = None
    tangent_range: str | float | list[float] | tuple[float, float] = "auto"
    tangent_min_points: int | None = None
    tangent_potential: float | None = None
    percent_threshold: float | None = None

    @classmethod
    def from_options(cls, options=None):
        return _coerce_options(cls, options, ["cv_analysis", "peak_current", "scale_current"])

    def validate(self):
        _validate_common_cv(self)
        if str(self.reference_mode).strip().lower() not in {"single", "both"}:
            raise OptionError("'reference mode' must be 'single' or 'both' for scale_current.")
        if not isinstance(self.reference_index, int):
            raise OptionError("'reference index' must be an integer.")
        sources = [
            self.scale is not None,
            self.reference_cv is not None,
            self.reference_cvs is not None,
            self.reference_index != 0,
        ]
        if sum(bool(source) for source in sources) > 1:
            raise OptionError("Use only one scale/reference source for scale_current.")

    def for_peak_current(self, segment=None):
        return PeakCurrentOptions.from_options(
            {
                "plot": self.plot_all,
                "plot all": self.plot_all,
                "print": False,
                "print all": False,
                "segment": self.segment if segment is None else segment,
                "noise window": self.noise_window,
                "noise polyorder": self.noise_polyorder,
                "sig figs": self.sig_figs,
                "peak prominence": self.peak_prominence,
                "guess potential": self.guess_potential,
                "exact potential": self.exact_potential,
                "tangent range": self.tangent_range,
                "tangent min points": self.tangent_min_points,
                "tangent potential": self.tangent_potential,
                "percent threshold": self.percent_threshold,
            }
        )

    def to_legacy_dict(self):
        data = {field.name.replace("_", " "): getattr(self, field.name) for field in fields(self)}
        data["pretty print"] = self.pretty_print
        data["plot all"] = self.plot_all
        data["print conditions"] = self.print_conditions
        data["scale"] = self.scale
        data["reference mode"] = str(self.reference_mode).strip().lower()
        data["reference index"] = self.reference_index
        data["reference cv"] = self.reference_cv
        data["reference cvs"] = self.reference_cvs
        data["noise window"] = self.noise_window
        data["noise polyorder"] = self.noise_polyorder
        data["sig figs"] = self.sig_figs
        data["peak prominence"] = self.peak_prominence
        data["guess potential"] = self.guess_potential
        data["exact potential"] = self.exact_potential
        data["tangent range"] = self.tangent_range
        data["tangent min points"] = self.tangent_min_points
        data["tangent potential"] = self.tangent_potential
        data["percent threshold"] = self.percent_threshold
        return data


@dataclass(frozen=True, slots=True)
class FOWAOptions:
    plot: bool = True
    print: bool = True
    plot_all: bool = False
    print_all: bool = False
    legend: bool = False
    legend_fontsize: int | float | None = None
    legend_loc: str = "best"
    legend_outside: bool = False
    legend_pad: float = 0.02
    legend_bbox_to_anchor: object | None = None
    legend_mode: str = "auto"
    min_gradient_entries: int = 3
    colorbar_height_scale: float = 1.0
    title: bool | str = True
    offset: float = 0
    color: str = "black"
    plot_labels: list[str] | None = None
    labels: list[str] | None = None
    new_plot: bool = False
    segment: int | None = None
    segments: list[int] | int | None = None
    plot_segments: list[int] | int | None = None
    noise_window: int | str = "auto"
    noise_polyorder: int | str = "auto"
    sig_figs: int = 4
    peak_prominence: float | None = None
    guess_potential: float | list[float] | None = None
    exact_potential: float | None = None
    tangent_range: str | float | list[float] | tuple[float, float] = "auto"
    tangent_min_points: int | None = None
    tangent_potential: float | None = None
    percent_threshold: float | None = None
    peak_fallback: str | None = "highest current"
    peak_potential: float | None = None
    troubleshoot: bool = False
    fit_basis: str = "x"
    fit_range: list[float] | tuple[float, float] = (0.0, 0.2)
    wave_range: list[float] | tuple[float, float] | None = None
    plot_fit: bool = True
    fit_label: bool | str = False
    fit_color: object | None = None
    fit_colors: object | None = None
    fit_linestyle: str = "--"
    fit_linewidth: int | float = 1.5
    fit_alpha: int | float = 1
    min_r2: float = 0.98
    redox_mode: str = "half wave"
    min_fit_points: int = 20
    background_correction: str | None = "tangent"
    diagnostic_y_axis: str = "i/ip0"
    ecat_shift_warning_threshold: float | bool | None = 0.05
    warnings: bool = True
    mechanism: str = "EC'"
    custom_formula: object | None = None
    formula_label: str | None = None
    catalyst_electrons: float = 1
    turnover_electrons: float = 1
    sigma: float = 1.0
    gaussian_weight: float = 0.4
    gaussian_skew: float = -3
    redox_potential: float | list[float] | None = None
    ip0: float | list[float] | None = None
    non_catalytic_current: float | None = None
    non_catalytic_cv: object | None = None
    non_catalytic_cvs: object | None = None
    non_catalytic_guess_potential: float | None = None

    @classmethod
    def from_options(cls, options=None):
        return _coerce_options(
            cls,
            options,
            ["plot", "multiplot", "cv_selection", "cv_analysis", "peak_current", "fowa"],
        )

    def validate(self):
        if isinstance(self.fit_basis, str):
            fit_basis_values = [self.fit_basis]
        elif isinstance(self.fit_basis, (list, tuple)):
            fit_basis_values = list(self.fit_basis)
        else:
            raise OptionError("'fit basis' must be 'x', 'y', or a list of those values.")

        bad_fit_basis = [value for value in fit_basis_values if str(value).strip().lower() not in {"x", "y"}]
        if bad_fit_basis:
            raise OptionError("'fit basis' must be 'x' or 'y'.")
        redox_mode = str(self.redox_mode).strip().lower()
        if redox_mode not in {"half wave", "half peak", "manual"}:
            raise OptionError("'redox mode' must be 'half wave', 'half peak', or 'manual'.")
        if self.redox_potential is not None and redox_mode != "manual":
            raise OptionError("Use 'redox mode': 'manual' when providing 'redox potential'.")
        diagnostic_y_axis = str(self.diagnostic_y_axis).strip().lower()
        if diagnostic_y_axis not in {"i/ip0", "current"}:
            raise OptionError("'diagnostic y axis' must be 'i/ip0' or 'current'.")
        if self.peak_fallback is not None:
            fallback = str(self.peak_fallback).strip().lower().replace("_", " ").replace("-", " ")
            valid = {
                "highest current",
                "highest absolute current",
                "max current",
                "max absolute current",
                "abs current",
                "guess potential",
                "exact potential",
                "none",
                "error",
                "raise",
            }
            if fallback not in valid:
                raise OptionError(
                    "'peak fallback' must be 'highest current', 'guess potential', or None/'none'."
                )
        if self.ip0 is not None and self.non_catalytic_current is not None:
            raise OptionError("Use either 'ip0' or 'non-catalytic current', not both.")
        _validate_common_cv(self)

    def for_peak_current(self):
        return PeakCurrentOptions.from_options({
            "plot": False,
            "print": False,
            "plot all": self.plot_all,
            "print all": self.print_all,
        })

    def to_legacy_dict(self):
        data = {field.name.replace("_", " "): getattr(self, field.name) for field in fields(self)}
        data["plot all"] = self.plot_all
        data["print all"] = self.print_all
        data["plot labels"] = self.plot_labels
        data["colorbar height scale"] = self.colorbar_height_scale
        data["min gradient entries"] = self.min_gradient_entries
        data["new plot"] = self.new_plot
        data["noise window"] = self.noise_window
        data["noise polyorder"] = self.noise_polyorder
        data["sig figs"] = self.sig_figs
        data["peak prominence"] = self.peak_prominence
        data["guess potential"] = self.guess_potential
        data["exact potential"] = self.exact_potential
        data["tangent range"] = self.tangent_range
        data["tangent min points"] = self.tangent_min_points
        data["tangent potential"] = self.tangent_potential
        data["percent threshold"] = self.percent_threshold
        data["peak fallback"] = self.peak_fallback
        data["peak potential"] = self.peak_potential
        data["fit basis"] = self.fit_basis
        data["fit range"] = self.fit_range
        data["wave range"] = self.wave_range
        data["min fit points"] = self.min_fit_points
        data["min r2"] = self.min_r2
        data["redox mode"] = str(self.redox_mode).strip().lower()
        data["background correction"] = self.background_correction
        data["diagnostic y axis"] = str(self.diagnostic_y_axis).strip().lower()
        data["ecat shift warning threshold"] = self.ecat_shift_warning_threshold
        data["warnings"] = self.warnings
        data["custom formula"] = self.custom_formula
        data["formula label"] = self.formula_label
        data["catalyst electrons"] = self.catalyst_electrons
        data["turnover electrons"] = self.turnover_electrons
        data["gaussian weight"] = self.gaussian_weight
        data["gaussian skew"] = self.gaussian_skew
        data["redox potential"] = self.redox_potential
        data["non-catalytic current"] = self.non_catalytic_current
        data["non-catalytic cv"] = self.non_catalytic_cv
        data["non-catalytic cvs"] = self.non_catalytic_cvs
        data["non-catalytic guess potential"] = self.non_catalytic_guess_potential
        return data


@dataclass(frozen=True, slots=True)
class PlateauCurrentOptions(FOWAOptions):
    ilim: float | list[float] | None = None
    ic: float | list[float] | None = None
    ip0_scan_rate: float | None = None
    ip0_sqrt_scan_rate_slope: float | None = None
    plateau_slope_tolerance: float = 0.10
    plateau_min_cvs: int = 2
    plateau_average_method: str = "mean"
    plateau_selection_mode: str = "high scan suffix"
    validate_plateau: bool = True
    require_plateau: bool = True
    formula_mode: str = "auto"
    c: float | str | None = None
    c_unit: str | None = None
    species: str | None = None
    d: float | None = None
    electrode_area: float | None = None
    scan_rate: float | None = None
    temperature: float = 298
    warn_ir_drop: bool = True

    @classmethod
    def from_options(cls, options=None):
        return _coerce_options(
            cls,
            options,
            ["plot", "cv_selection", "cv_analysis", "peak_current", "fowa", "plateau_current"],
        )

    def validate(self):
        FOWAOptions.validate(self)
        if self.ilim is not None and self.ic is not None:
            raise OptionError("Use either 'ilim' or 'ic', not both.")
        if float(self.plateau_slope_tolerance) < 0:
            raise OptionError("'plateau slope tolerance' must be non-negative.")
        if int(self.plateau_min_cvs) < 1:
            raise OptionError("'plateau min cvs' must be at least 1.")
        if str(self.plateau_average_method).strip().lower() not in {"mean", "median"}:
            raise OptionError("'plateau average method' must be 'mean' or 'median'.")
        if str(self.plateau_selection_mode).strip().lower() != "high scan suffix":
            raise OptionError("Only 'high scan suffix' plateau selection mode is currently supported.")
        mode = str(self.formula_mode).strip().lower()
        if mode not in {"auto", "normalized", "slope normalized", "slope-normalized", "direct"}:
            raise OptionError("'formula mode' must be 'auto', 'normalized', 'slope normalized', or 'direct'.")
        if self.d is not None and float(self.d) <= 0:
            raise OptionError("'D' must be positive when provided.")
        if self.electrode_area is not None and float(self.electrode_area) <= 0:
            raise OptionError("'electrode area' must be positive when provided.")

    def to_legacy_dict(self):
        data = FOWAOptions.to_legacy_dict(self)
        data["ilim"] = self.ilim
        data["ic"] = self.ic
        data["ip0 scan rate"] = self.ip0_scan_rate
        data["ip0 sqrt scan rate slope"] = self.ip0_sqrt_scan_rate_slope
        data["plateau slope tolerance"] = self.plateau_slope_tolerance
        data["plateau min cvs"] = self.plateau_min_cvs
        data["plateau average method"] = self.plateau_average_method
        data["plateau selection mode"] = self.plateau_selection_mode
        data["validate plateau"] = self.validate_plateau
        data["require plateau"] = self.require_plateau
        data["formula mode"] = self.formula_mode
        data["C"] = self.c
        data["C unit"] = self.c_unit
        data["species"] = self.species
        data["D"] = self.d
        data["electrode area"] = self.electrode_area
        data["scan rate"] = self.scan_rate
        data["temperature"] = self.temperature
        data["warn ir drop"] = self.warn_ir_drop
        return data


@dataclass(frozen=True, slots=True)
class FitRateOptions:
    plot: bool = True
    print: bool = True
    print_all: bool = False
    fit: bool = True
    metric: str = "kobs"
    x_column: str | None = "auto"
    species: str | None = None
    x_transform: str | float | int | None = None
    y_transform: str | float | int | None = None
    transform_mode: str | None = None
    floor: bool | str | float | int | None = None
    x_floor: bool | str | float | int | None = None
    y_floor: bool | str | float | int | None = None
    y_mode: str | None = None
    y0: object | None = None
    xscale: str | None = None
    yscale: str | None = None
    plot_scale: str | None = None
    plot_log_log: bool = False
    log_log_plot: bool = False
    fit_indices: object | None = None
    fit_range: object | None = None
    fit_ranges: object | None = None
    log_fit_indices: object | None = None
    fit_model: object | None = "linear"
    fit_params: object | None = None
    fit_equation: str | None = None
    fit_init: object | None = "auto"
    fit_bounds: object | None = "auto"
    fit_residual: str = "direct"
    fit_max_evals: int = 10000
    print_fit: str = "auto"
    print_fit_details: bool = False
    sig_figs: int = 4
    plot_fit: bool = True
    exclude_warnings: bool = False
    exclude_low_r2: bool = False
    min_r2: float = 0.95
    local_slope_mode: str = "adjacent"
    print_local_slopes: bool = False
    plot_local_slopes: bool = False
    legend: bool = False
    legend_fontsize: int | float | None = None
    fit_label: bool | str = False
    fit_color: object = "tab:red"
    fit_colors: object | None = None
    fit_linestyle: str = "--"
    fit_linewidth: int | float = 1
    fit_alpha: int | float = 1
    return_stats: bool = False

    @classmethod
    def from_options(cls, options=None):
        return _coerce_options(cls, options, ["fit_rate"])

    def validate(self):
        mode = str(self.local_slope_mode).strip().lower()
        if mode not in {"adjacent", "gradient"}:
            raise OptionError("'local slope mode' must be 'adjacent' or 'gradient'.")
        if self.plot_scale is not None:
            scale = str(self.plot_scale).strip().lower().replace("_", "-").replace(" ", "-")
            if scale not in {"linear", "log-log", "loglog", "semilogx", "semi-log-x", "semilogy", "semi-log-y", "symlog"}:
                raise OptionError("'plot scale' must be 'linear', 'log-log', 'semilogx', 'semilogy', or 'symlog'.")
        if not 0 <= float(self.min_r2) <= 1:
            raise OptionError("'min r2' must be between 0 and 1.")
        if not 0 <= float(self.fit_alpha) <= 1:
            raise OptionError("'fit alpha' must be between 0 and 1.")
        if float(self.fit_linewidth) < 0:
            raise OptionError("'fit linewidth' must be non-negative.")

    def to_legacy_dict(self):
        data = {field.name.replace("_", " "): getattr(self, field.name) for field in fields(self)}
        data["x column"] = self.x_column
        data["x transform"] = self.x_transform
        data["y transform"] = self.y_transform
        data["transform mode"] = self.transform_mode
        data["floor"] = self.floor
        data["x floor"] = self.x_floor
        data["y floor"] = self.y_floor
        data["y mode"] = self.y_mode
        data["y0"] = self.y0
        data["xscale"] = self.xscale
        data["yscale"] = self.yscale
        data["plot scale"] = self.plot_scale
        data["plot log-log"] = self.plot_log_log
        data["log log plot"] = self.log_log_plot
        data["fit indices"] = self.fit_indices
        data["fit range"] = self.fit_range
        data["fit ranges"] = self.fit_ranges
        data["log fit indices"] = self.log_fit_indices
        data["fit model"] = self.fit_model
        data["fit params"] = self.fit_params
        data["fit equation"] = self.fit_equation
        data["fit init"] = self.fit_init
        data["fit bounds"] = self.fit_bounds
        data["fit residual"] = self.fit_residual
        data["fit max evals"] = self.fit_max_evals
        data["print fit"] = self.print_fit
        data["print fit details"] = self.print_fit_details
        data["sig figs"] = self.sig_figs
        data["plot fit"] = self.plot_fit
        data["exclude warnings"] = self.exclude_warnings
        data["exclude low r2"] = self.exclude_low_r2
        data["min r2"] = self.min_r2
        data["local slope mode"] = self.local_slope_mode
        data["print local slopes"] = self.print_local_slopes
        data["plot local slopes"] = self.plot_local_slopes
        data["legend"] = self.legend
        data["legend fontsize"] = self.legend_fontsize
        data["fit label"] = self.fit_label
        data["fit color"] = self.fit_color
        data["fit colors"] = self.fit_colors
        data["fit linestyle"] = self.fit_linestyle
        data["fit linewidth"] = self.fit_linewidth
        data["fit alpha"] = self.fit_alpha
        data["return stats"] = self.return_stats
        return data


@dataclass(frozen=True, slots=True)
class FitPeakPotentialOptions(PeakPotentialOptions):
    follow_e1_2: bool = False
    fit: bool = True
    x_unit: str | None = "auto"
    y_unit: str | None = "auto"
    species: str | None = None
    y_mode: str | None = None
    y0: object | None = None
    fit_indices: object | None = None
    fit_range: object | None = None
    fit_ranges: object | None = None
    fit_model: object | None = "linear"
    fit_params: object | None = None
    fit_equation: str | None = None
    fit_init: object | None = "auto"
    fit_bounds: object | None = "auto"
    fit_residual: str = "direct"
    fit_max_evals: int = 10000
    print_fit: str = "auto"
    print_fit_details: bool = False
    sig_figs: int = 6
    plot_fit: bool = True
    legend: bool = False
    legend_fontsize: int | float | None = None
    fit_label: bool | str = False
    fit_color: object = "tab:red"
    fit_colors: object | None = None
    fit_linestyle: str = "--"
    fit_linewidth: int | float = 1
    fit_alpha: int | float = 1
    return_stats: bool = False

    @classmethod
    def from_options(cls, options=None):
        return _coerce_options(cls, options, ["plot", "cv_selection", "cv_analysis", "fit_peak_potential"])

    def validate(self):
        _validate_common_cv(self)
        if not 0 <= float(self.fit_alpha) <= 1:
            raise OptionError("'fit alpha' must be between 0 and 1.")

    def to_legacy_dict(self):
        data = PeakPotentialOptions.to_legacy_dict(self)
        data["follow e1/2"] = self.follow_e1_2
        data["fit"] = self.fit
        data["x unit"] = self.x_unit
        data["y unit"] = self.y_unit
        data["species"] = self.species
        data["y mode"] = self.y_mode
        data["y0"] = self.y0
        data["fit indices"] = self.fit_indices
        data["fit range"] = self.fit_range
        data["fit ranges"] = self.fit_ranges
        data["fit model"] = self.fit_model
        data["fit params"] = self.fit_params
        data["fit equation"] = self.fit_equation
        data["fit init"] = self.fit_init
        data["fit bounds"] = self.fit_bounds
        data["fit residual"] = self.fit_residual
        data["fit max evals"] = self.fit_max_evals
        data["print fit"] = self.print_fit
        data["print fit details"] = self.print_fit_details
        data["sig figs"] = self.sig_figs
        data["plot fit"] = self.plot_fit
        data["legend"] = self.legend
        data["legend fontsize"] = self.legend_fontsize
        data["fit label"] = self.fit_label
        data["fit color"] = self.fit_color
        data["fit colors"] = self.fit_colors
        data["fit linestyle"] = self.fit_linestyle
        data["fit linewidth"] = self.fit_linewidth
        data["fit alpha"] = self.fit_alpha
        data["return stats"] = self.return_stats
        return data


@dataclass(frozen=True, slots=True)
class SevcikAnalysisOptions(PeakCurrentOptions):
    x_unit: str | None = "auto"
    y_unit: str | None = "auto"
    species: str | None = None
    fit_indices: object | None = None
    plot_fit: bool = True
    legend: bool = False
    legend_fontsize: int | float | None = None
    fit_label: bool | str = False
    fit_color: object = "tab:red"
    fit_colors: object | None = None
    fit_linestyle: str = "--"
    fit_linewidth: int | float = 1
    fit_alpha: int | float = 1
    return_stats: bool = False
    c: float | None = None
    num_electrons: int | float = 1
    scan_dependence: int | float = 0.5

    @classmethod
    def from_options(cls, options=None):
        return _coerce_options(cls, options, ["plot", "cv_selection", "cv_analysis", "peak_current", "sevcik_analysis"])

    def validate(self):
        _validate_common_cv(self)
        if float(self.num_electrons) <= 0:
            raise OptionError("'num electrons' must be positive.")
        if float(self.scan_dependence) <= 0:
            raise OptionError("'scan dependence' must be positive.")
        if not 0 <= float(self.fit_alpha) <= 1:
            raise OptionError("'fit alpha' must be between 0 and 1.")

    def to_legacy_dict(self):
        data = PeakCurrentOptions.to_legacy_dict(self)
        data["x unit"] = self.x_unit
        data["y unit"] = self.y_unit
        data["species"] = self.species
        data["fit indices"] = self.fit_indices
        data["plot fit"] = self.plot_fit
        data["legend"] = self.legend
        data["legend fontsize"] = self.legend_fontsize
        data["fit label"] = self.fit_label
        data["fit color"] = self.fit_color
        data["fit colors"] = self.fit_colors
        data["fit linestyle"] = self.fit_linestyle
        data["fit linewidth"] = self.fit_linewidth
        data["fit alpha"] = self.fit_alpha
        data["return stats"] = self.return_stats
        data["C"] = self.c
        data["num electrons"] = self.num_electrons
        data["scan dependence"] = self.scan_dependence
        return data

    def for_peak_current(self, segment=None):
        return PeakCurrentOptions.from_options(
            {
                "plot": self.plot,
                "print": self.print,
                "plot all": self.plot_all,
                "print all": self.print_all,
                "segment": self.segment if segment is None else segment,
                "noise window": self.noise_window,
                "noise polyorder": self.noise_polyorder,
                "sig figs": self.sig_figs,
                "peak prominence": self.peak_prominence,
                "guess potential": self.guess_potential,
                "exact potential": self.exact_potential,
                "tangent range": self.tangent_range,
                "tangent min points": self.tangent_min_points,
                "tangent potential": self.tangent_potential,
                "percent threshold": self.percent_threshold,
            }
        )


@dataclass(frozen=True, slots=True)
class FitPeakCurrentOptions(PeakCurrentOptions):
    fit: bool = True
    x_unit: str | None = "auto"
    y_unit: str | None = "auto"
    species: str | None = None
    x_power: int | float | None = 0.5
    x_transform: str | float | int | None = None
    y_transform: str | float | int | None = None
    transform_mode: str | None = None
    floor: bool | str | float | int | None = None
    x_floor: bool | str | float | int | None = None
    y_floor: bool | str | float | int | None = None
    y_mode: str | None = None
    y0: object | None = None
    xscale: str | None = None
    yscale: str | None = None
    plot_scale: str | None = None
    fit_indices: object | None = None
    fit_range: object | None = None
    fit_ranges: object | None = None
    fit_model: object | None = "linear"
    fit_params: object | None = None
    fit_equation: str | None = None
    fit_init: object | None = "auto"
    fit_bounds: object | None = "auto"
    fit_residual: str = "direct"
    fit_max_evals: int = 10000
    print_fit: str = "auto"
    print_fit_details: bool = False
    sig_figs: int = 6
    plot_fit: bool = True
    legend: bool = False
    legend_fontsize: int | float | None = None
    fit_label: bool | str = False
    fit_color: object = "tab:red"
    fit_colors: object | None = None
    fit_linestyle: str = "--"
    fit_linewidth: int | float = 1
    fit_alpha: int | float = 1
    return_stats: bool = False

    @classmethod
    def from_options(cls, options=None):
        return _coerce_options(cls, options, ["plot", "cv_selection", "cv_analysis", "peak_current", "fit_peak_current"])

    def validate(self):
        _validate_common_cv(self)
        if self.plot_scale is not None:
            scale = str(self.plot_scale).strip().lower().replace("_", "-").replace(" ", "-")
            if scale not in {"linear", "log-log", "loglog", "semilogx", "semi-log-x", "semilogy", "semi-log-y", "symlog"}:
                raise OptionError("'plot scale' must be 'linear', 'log-log', 'semilogx', 'semilogy', or 'symlog'.")
        if not 0 <= float(self.fit_alpha) <= 1:
            raise OptionError("'fit alpha' must be between 0 and 1.")
        if float(self.fit_linewidth) < 0:
            raise OptionError("'fit linewidth' must be non-negative.")

    def to_legacy_dict(self):
        data = PeakCurrentOptions.to_legacy_dict(self)
        data["fit"] = self.fit
        data["x unit"] = self.x_unit
        data["y unit"] = self.y_unit
        data["species"] = self.species
        data["x power"] = self.x_power
        data["x transform"] = self.x_transform
        data["y transform"] = self.y_transform
        data["transform mode"] = self.transform_mode
        data["floor"] = self.floor
        data["x floor"] = self.x_floor
        data["y floor"] = self.y_floor
        data["y mode"] = self.y_mode
        data["y0"] = self.y0
        data["xscale"] = self.xscale
        data["yscale"] = self.yscale
        data["plot scale"] = self.plot_scale
        data["fit indices"] = self.fit_indices
        data["fit range"] = self.fit_range
        data["fit ranges"] = self.fit_ranges
        data["fit model"] = self.fit_model
        data["fit params"] = self.fit_params
        data["fit equation"] = self.fit_equation
        data["fit init"] = self.fit_init
        data["fit bounds"] = self.fit_bounds
        data["fit residual"] = self.fit_residual
        data["fit max evals"] = self.fit_max_evals
        data["print fit"] = self.print_fit
        data["print fit details"] = self.print_fit_details
        data["sig figs"] = self.sig_figs
        data["plot fit"] = self.plot_fit
        data["legend"] = self.legend
        data["legend fontsize"] = self.legend_fontsize
        data["fit label"] = self.fit_label
        data["fit color"] = self.fit_color
        data["fit colors"] = self.fit_colors
        data["fit linestyle"] = self.fit_linestyle
        data["fit linewidth"] = self.fit_linewidth
        data["fit alpha"] = self.fit_alpha
        data["return stats"] = self.return_stats
        return data

    def for_peak_current(self, segment=None):
        return PeakCurrentOptions.from_options(
            {
                "plot": self.plot,
                "print": self.print,
                "plot all": self.plot_all,
                "print all": self.print_all,
                "segment": self.segment if segment is None else segment,
                "noise window": self.noise_window,
                "noise polyorder": self.noise_polyorder,
                "sig figs": self.sig_figs,
                "peak prominence": self.peak_prominence,
                "guess potential": self.guess_potential,
                "exact potential": self.exact_potential,
                "tangent range": self.tangent_range,
                "tangent min points": self.tangent_min_points,
                "tangent potential": self.tangent_potential,
                "percent threshold": self.percent_threshold,
            }
        )


@dataclass(frozen=True, slots=True)
class TrumpetAnalysisOptions(PeakCurrentOptions):
    segment: int | None = 1
    fit_indices: object | None = None
    plot_fit: bool = True
    legend: bool = False
    legend_fontsize: int | float | None = None
    fit_label: bool | str = False
    fit_color: object = "tab:red"
    fit_colors: object | None = None
    fit_linestyle: str = "--"
    fit_linewidth: int | float = 1
    fit_alpha: int | float = 1
    return_stats: bool = False
    d: float | None = None
    temperature: float = 298

    @classmethod
    def from_options(cls, options=None):
        return _coerce_options(cls, options, ["plot", "cv_selection", "cv_analysis", "peak_current", "trumpet_analysis"])

    def validate(self):
        _validate_common_cv(self)
        if _resolve_trumpet_base_segment(self.segment, self.segments) is None:
            raise OptionError("'segment' or consecutive 'segments' is required for trumpet_analysis.")
        if self.d is not None and float(self.d) <= 0:
            raise OptionError("'D' must be positive when provided.")
        if float(self.temperature) <= 0:
            raise OptionError("'temperature' must be positive.")
        if not 0 <= float(self.fit_alpha) <= 1:
            raise OptionError("'fit alpha' must be between 0 and 1.")

    def to_legacy_dict(self):
        data = PeakCurrentOptions.to_legacy_dict(self)
        if self.segment is None and self.segments is not None:
            data["segment"] = _resolve_trumpet_base_segment(self.segment, self.segments)
            data["segments"] = None
        data["fit indices"] = self.fit_indices
        data["plot fit"] = self.plot_fit
        data["legend"] = self.legend
        data["legend fontsize"] = self.legend_fontsize
        data["fit label"] = self.fit_label
        data["fit color"] = self.fit_color
        data["fit colors"] = self.fit_colors
        data["fit linestyle"] = self.fit_linestyle
        data["fit linewidth"] = self.fit_linewidth
        data["fit alpha"] = self.fit_alpha
        data["return stats"] = self.return_stats
        data["D"] = self.d
        data["temperature"] = self.temperature
        return data


def _resolve_trumpet_base_segment(segment, segments):
    if segment is not None:
        return segment
    if segments is None:
        return None
    if isinstance(segments, int):
        return segments
    try:
        values = list(segments)
    except TypeError as exc:
        raise OptionError("'segments' for trumpet_analysis must be an int or a consecutive 2-element sequence.") from exc
    if len(values) != 2:
        raise OptionError("'segments' for trumpet_analysis must be an int or a consecutive 2-element sequence.")
    first, second = values
    if not isinstance(first, int) or not isinstance(second, int) or second != first + 1:
        raise OptionError("'segments' for trumpet_analysis must be an int or a consecutive 2-element sequence.")
    return first


@dataclass(frozen=True, slots=True)
class NicholsonOptions(PeakCurrentOptions):
    segment: int | None = 1
    plot_fit: bool = True
    fit_model: object | None = "origin"
    num_electrons: int | float = 1
    d: float | None = None
    psi_source: str = "agarwal table"
    nicholson_delta_ep_min_mv: int | float = 61
    nicholson_delta_ep_max_mv: int | float = 212
    fit_through_origin: bool = True
    exclude_invalid_delta_ep: bool = True
    plot_diagnostic: bool = True
    empirical_psi_equation: str = "lavagnini"
    warn_ir_drop: bool = True
    scan_rate: object | None = None
    scan_rates: object | None = None

    @classmethod
    def from_options(cls, options=None):
        return _coerce_options(cls, options, ["plot", "cv_selection", "cv_analysis", "peak_current", "nicholson"])

    def validate(self):
        _validate_common_cv(self)
        if self.segment is None:
            raise OptionError("'segment' is required for Nicholson.")
        model = str(self.fit_model).strip().lower().replace("_", " ").replace("-", " ")
        if model not in {"origin", "linear"}:
            raise OptionError("'fit model' for Nicholson must be 'origin' or 'linear'.")
        if float(self.num_electrons) <= 0:
            raise OptionError("'num electrons' must be positive.")
        if self.d is not None and float(self.d) <= 0:
            raise OptionError("'D' must be positive when provided.")
        if float(self.nicholson_delta_ep_min_mv) <= 0:
            raise OptionError("'nicholson delta ep min mv' must be positive.")
        if float(self.nicholson_delta_ep_max_mv) <= float(self.nicholson_delta_ep_min_mv):
            raise OptionError("'nicholson delta ep max mv' must be greater than the minimum.")

    def to_legacy_dict(self):
        data = PeakCurrentOptions.to_legacy_dict(self)
        data["plot fit"] = self.plot_fit
        data["fit model"] = self.fit_model
        data["num electrons"] = self.num_electrons
        data["D"] = self.d
        data["psi source"] = self.psi_source
        data["nicholson delta ep min mv"] = self.nicholson_delta_ep_min_mv
        data["nicholson delta ep max mv"] = self.nicholson_delta_ep_max_mv
        data["fit through origin"] = self.fit_through_origin
        data["exclude invalid delta ep"] = self.exclude_invalid_delta_ep
        data["plot diagnostic"] = self.plot_diagnostic
        data["empirical psi equation"] = self.empirical_psi_equation
        data["warn ir drop"] = self.warn_ir_drop
        data["scan rate"] = self.scan_rate
        data["scan rates"] = self.scan_rates
        return data


@dataclass(frozen=True, slots=True)
class TafelAnalysisOptions:
    overpotential_range: list[float] | tuple[float, float] = (0, 1)
    color: str = "black"

    @classmethod
    def from_options(cls, options=None):
        return _coerce_options(cls, options, ["tafel_analysis"])

    def validate(self):
        if len(self.overpotential_range) != 2:
            raise OptionError("'overpotential range' must contain [start, end].")

    def to_legacy_dict(self):
        return {
            "overpotential range": list(self.overpotential_range),
            "color": self.color,
        }


@dataclass(frozen=True, slots=True)
class FilterOptions:
    print: bool = True
    pretty_print: bool = True
    print_conditions: bool = True
    mode: str = "include"
    logic: str | None = None

    @classmethod
    def from_options(cls, options=None):
        return _coerce_options(cls, options, ["filter"])

    def validate(self):
        mode = str(self.mode).strip().lower()
        if mode not in {"include", "exclude"}:
            raise OptionError("'mode' must be 'include' or 'exclude'.")
        if self.logic is not None and str(self.logic).strip().upper() not in {"AND", "OR"}:
            raise OptionError("'logic' must be 'AND', 'OR', or None.")

    def to_legacy_dict(self):
        return {
            "print": self.print,
            "pretty print": self.pretty_print,
            "print conditions": self.print_conditions,
            "mode": str(self.mode).strip().lower(),
            "logic": None if self.logic is None else str(self.logic).strip().upper(),
        }


@dataclass(frozen=True, slots=True)
class SortGroupOptions:
    print: bool = True
    pretty_print: bool = True
    print_conditions: bool = True

    @classmethod
    def from_options(cls, options=None):
        return _coerce_options(cls, options, ["sort_group"])

    def validate(self):
        return None

    def to_legacy_dict(self):
        return {
            "print": self.print,
            "pretty print": self.pretty_print,
            "print conditions": self.print_conditions,
        }


@dataclass(frozen=True, slots=True)
class GroupSummaryOptions:
    print: bool = True
    pretty_print: bool = True
    sig_figs: int = 3
    group_keys: object = None
    columns: object = field(default_factory=list)

    @classmethod
    def from_options(cls, options=None):
        return _coerce_options(cls, options, ["group_summary"])

    def validate(self):
        return None

    def to_legacy_dict(self):
        return {
            "print": self.print,
            "pretty print": self.pretty_print,
            "sig figs": self.sig_figs,
            "group keys": self.group_keys,
            "columns": self.columns,
        }


def _display_option_key(key):
    key = normalize_key(key)
    return _DISPLAY_KEY_OVERRIDES.get(key, _friendly_key(key))


def _display_section_key(key):
    return _canonical_section_key(key)


def _type_to_string(annotation):
    if isinstance(annotation, str):
        return annotation.replace("NoneType", "None").replace("typing.", "")

    if annotation is None or annotation is type(None):
        return "None"
    if annotation is object:
        return "object"

    origin = typing.get_origin(annotation)
    args = typing.get_args(annotation)
    if origin in {typing.Union, types.UnionType}:
        return " or ".join(_type_to_string(arg) for arg in args)
    if origin in {list, tuple, dict, set}:
        name = origin.__name__
        if not args:
            return name
        return f"{name}[{', '.join(_type_to_string(arg) for arg in args)}]"
    if hasattr(annotation, "__name__"):
        return annotation.__name__
    return str(annotation).replace("typing.", "")


def _type_hints_for_model(cls):
    hints = {}
    for base in reversed(cls.__mro__):
        try:
            hints.update(typing.get_type_hints(base))
        except Exception:  # pragma: no cover - defensive fallback for unusual annotations
            hints.update(getattr(base, "__annotations__", {}))
    return hints


def _option_type_lookup():
    lookup = {}
    for cls, _sections in _option_default_registry():
        hints = _type_hints_for_model(cls)
        for field in fields(cls):
            lookup.setdefault(field.name, hints.get(field.name, field.type))
    return lookup


def _default_for_field(field):
    if field.default is not MISSING:
        return field.default
    if field.default_factory is not MISSING:  # type: ignore[attr-defined]
        return field.default_factory()  # type: ignore[misc]
    return None


def _sections_for_option_model(cls):
    for registered_cls, sections in _option_default_registry():
        if registered_cls is cls:
            return sections
    return []


def _metadata_for_option(key, section=None):
    norm = normalize_key(key)
    metadata = dict(OPTION_METADATA.get("*", {}).get(norm, {}))
    if section is not None:
        section_norm = _canonical_section_key(section)
        metadata.update(OPTION_METADATA.get(section_norm, {}).get(norm, {}))
    return metadata


def _description_for_option(key, section=None):
    return _metadata_for_option(key, section=section).get("description", "")


def _choices_for_option(key, section=None):
    return list(_metadata_for_option(key, section=section).get("choices", []))


def _category_for_option(key, section=None):
    return _metadata_for_option(key, section=section).get("category", "Advanced")


def _schema_entry(key, default, annotation, section=None):
    return {
        "category": _category_for_option(key, section=section),
        "default": deepcopy(default),
        "type": _type_to_string(annotation),
        "choices": _choices_for_option(key, section=section),
        "description": _description_for_option(key, section=section),
    }


def _format_choices_for_display(choices):
    if not choices:
        return ""
    return ", ".join(str(choice) for choice in choices)


def _options_schema_to_dataframe(schema):
    import pandas as pd

    rows = []

    def add_schema_rows(section, section_schema):
        for option, entry in section_schema.items():
            rows.append({
                "Function": section,
                "Category": entry.get("category", "Advanced"),
                "Option": option,
                "Default": entry.get("default"),
                "Type": entry.get("type", ""),
                "Choices": _format_choices_for_display(entry.get("choices", [])),
                "Description": entry.get("description", ""),
            })

    def sort_option_rows(df):
        if df.empty or "Option" not in df.columns:
            return df
        sort_columns = []
        if "Function" in df.columns:
            sort_columns.append("Function")
        if "Category" in df.columns:
            category_order = {
                category: index
                for index, category in enumerate(OPTION_CATEGORY_ORDER)
            }
            df = df.assign(
                _category_order=df["Category"].map(category_order).fillna(len(category_order))
            )
            sort_columns.append("_category_order")
        sort_columns.append("Option")
        sorted_df = df.sort_values(sort_columns, kind="stable").reset_index(drop=True)
        if "_category_order" in sorted_df.columns:
            sorted_df = sorted_df.drop(columns=["_category_order"])
        return sorted_df

    if schema and all(
        isinstance(value, dict)
        and "default" not in value
        and "type" not in value
        and "choices" not in value
        and "description" not in value
        and "category" not in value
        for value in schema.values()
    ):
        for section, section_schema in schema.items():
            add_schema_rows(section, section_schema)
        return _drop_empty_display_columns(sort_option_rows(pd.DataFrame(rows)))

    add_schema_rows("", schema)
    df = pd.DataFrame(rows)
    return _drop_empty_display_columns(sort_option_rows(df.drop(columns=["Function"])))


def _drop_empty_display_columns(df):
    protected_columns = {"Function", "Category", "Option"}
    empty_columns = []
    for column in df.columns:
        if column in protected_columns:
            continue
        values = df[column]
        if values.map(lambda value: value is None or value == "").all():
            empty_columns.append(column)
    if empty_columns:
        return df.drop(columns=empty_columns)
    return df


def _display_options_schema(schema, display_function=None):
    df = _options_schema_to_dataframe(schema)
    _display_options_dataframe(df, display_function=display_function)


def _display_options_dataframe(df, display_function=None):
    display_func = display_function if display_function is not None else display
    if display_func is not None:
        try:
            styled = (
                df.style
                .set_properties(**{
                    "text-align": "left",
                    "white-space": "pre-wrap",
                    "vertical-align": "top",
                })
                .set_table_styles([
                    {"selector": "th", "props": [("text-align", "left")]},
                    {"selector": "td", "props": [("text-align", "left")]},
                ])
            )
            display_func(styled)
        except Exception:
            display_func(df)
    else:
        print(df.to_string(index=False))


def _describe_options_config(options=None):
    config = {} if options is None else dict(options)
    config.setdefault("print", True)
    config.setdefault("pretty print", True)
    config.setdefault("return", False)
    return config


def _describe_options_should_return_table(config):
    return bool(config.get("return", False))


def _describe_options_section_names():
    return [_display_section_key(section) for section in sorted(_default_section_names())]


def _describe_options_method_names():
    return sorted(_METHOD_SECTION_ALIASES)


def _describe_options_function_names():
    return sorted(set(_describe_options_section_names()) | set(_describe_options_method_names()))


def _describe_options_menu_dataframe():
    import pandas as pd

    rows = [
        {
            "Function": "all",
            "Description": "Show options for every function.",
        }
    ]
    rows.extend(
        {
            "Function": function,
            "Description": f'Show options for "{function}".',
        }
        for function in _describe_options_function_names()
    )
    return pd.DataFrame(rows)


def _describe_options_schema_for_section(section, type_lookup):
    defaults = get_defaults(section)
    return {
        _display_option_key(key): _schema_entry(
            key,
            value,
            type_lookup.get(key, type(value)),
            section=section,
        )
        for key, value in defaults.items()
    }


def _describe_options_all_schema(type_lookup):
    return {
        _display_section_key(section): _describe_options_schema_for_section(section, type_lookup)
        for section in sorted(_default_section_names())
    }


def _describe_options_invalid_section_message(section):
    valid_inputs = ["all", *_describe_options_function_names()]
    section_text = str(section)
    matches = difflib.get_close_matches(
        normalize_key(section_text),
        valid_inputs,
        n=1,
        cutoff=0.6,
    )
    if matches:
        return f"Unknown options function '{section_text}'. Did you mean '{matches[0]}'?"
    return (
        f"Unknown options function '{section_text}'. "
        "Possible functions are listed below."
    )


def describe_options(option_model_or_section=None, options=None):
    """Display eCAT option functions or option tables.

    With no argument, display valid option function names. Pass ``"all"`` for
    every option function, a function name such as ``"multiplot"``, a public
    method name such as ``"cv.peak_current"`` or ``"ca.charge"``, or an option
    dataclass such as ``PeakCurrentOptions``.

    Output is enabled by default. ``options={"pretty print": False}`` emits a
    plain text table instead of rich notebook display. Pass
    ``options={"print": False}`` to suppress output, and
    ``options={"return": True}`` to receive the dataframe.
    """
    config = _describe_options_config(options)
    type_lookup = _option_type_lookup()
    print_output = bool(config.get("print", True))
    pretty_print = bool(config.get("pretty print", True))
    return_table = _describe_options_should_return_table(config)

    def display_and_return(df):
        if print_output:
            if pretty_print:
                _display_options_dataframe(df, config.get("display function"))
            else:
                print(df.to_string(index=False))
        if return_table:
            return df
        return None

    if option_model_or_section is None:
        return display_and_return(_describe_options_menu_dataframe())

    if isinstance(option_model_or_section, str):
        section = option_model_or_section
        section_key = _canonical_section_key(section)

        if section_key == "all":
            return display_and_return(
                _options_schema_to_dataframe(_describe_options_all_schema(type_lookup))
            )

        if section_key not in _default_section_names():
            if print_output:
                print(_describe_options_invalid_section_message(section))
            return display_and_return(_describe_options_menu_dataframe())

        schema = _describe_options_schema_for_section(section_key, type_lookup)
        return display_and_return(_options_schema_to_dataframe(schema))

    cls = option_model_or_section
    if not hasattr(cls, "__dataclass_fields__"):
        cls = type(option_model_or_section)
    if not hasattr(cls, "__dataclass_fields__"):
        raise TypeError("describe_options() accepts a defaults section name or option dataclass.")

    sections = _sections_for_option_model(cls)
    metadata_section = sections[0] if len(sections) == 1 else None
    defaults = {}
    for section in sections:
        defaults.update(get_defaults(section))

    hints = _type_hints_for_model(cls)
    schema = {}
    for field in fields(cls):
        default = defaults.get(field.name, _default_for_field(field))
        schema[_display_option_key(field.name)] = _schema_entry(
            field.name,
            default,
            hints.get(field.name, field.type),
            section=metadata_section,
        )
    return display_and_return(_options_schema_to_dataframe(schema))

