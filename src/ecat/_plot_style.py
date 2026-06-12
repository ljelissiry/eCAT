"""Matplotlib style profiles used by eCAT plotting helpers."""

import matplotlib as mpl
import matplotlib.pyplot as plt


_ECAT_PLOT_STYLE_KEYS = [
    "figure.figsize",
    "figure.dpi",
    "figure.subplot.top",
    "savefig.dpi",
    "legend.fontsize",
    "figure.titlesize",
    "axes.titlesize",
    "axes.labelsize",
    "font.size",
    "font.family",
    "font.serif",
    "mathtext.fontset",
    "lines.linewidth",
    "axes.edgecolor",
    "axes.linewidth",
    "legend.framealpha",
    "legend.labelspacing",
    "legend.borderpad",
    "figure.facecolor",
    "axes.facecolor",
    "xtick.minor.visible",
    "ytick.minor.visible",
    "xtick.minor.ndivs",
    "ytick.minor.ndivs",
    "xtick.direction",
    "ytick.direction",
    "xtick.top",
    "ytick.right",
    "xtick.labelsize",
    "ytick.labelsize",
    "xtick.major.size",
    "ytick.major.size",
    "xtick.major.width",
    "ytick.major.width",
    "xtick.minor.size",
    "ytick.minor.size",
    "xtick.minor.width",
    "ytick.minor.width",
]
_MATPLOTLIB_PLOT_STYLE_DEFAULTS = {
    key: mpl.rcParamsDefault[key]
    for key in _ECAT_PLOT_STYLE_KEYS
}
_ECAT_PLOT_STYLE_PROFILES = {
    "notebook": {
        "rcParams": {
            "figure.figsize": [5.0, 3.75],
            "figure.dpi": 150,
            "figure.subplot.top": 0.88,
            "savefig.dpi": 300,
            "legend.fontsize": 9,
            "figure.titlesize": 14,
            "axes.titlesize": 11,
            "axes.labelsize": 12,
            "font.size": 11,
            "font.family": "Arial",
            "axes.linewidth": 1.1,
            "legend.framealpha": 0,
            "legend.labelspacing": 0.25,
            "legend.borderpad": 0.2,
            "figure.facecolor": "none",
            "axes.facecolor": "none",
            "xtick.minor.visible": True,
            "ytick.minor.visible": True,
            "xtick.minor.ndivs": 2,
            "ytick.minor.ndivs": 2,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "xtick.major.size": 5,
            "ytick.major.size": 5,
            "xtick.major.width": 1.1,
            "ytick.major.width": 1.1,
            "xtick.minor.size": 3,
            "ytick.minor.size": 3,
            "xtick.minor.width": 1.1,
            "ytick.minor.width": 1.1,
        },
        "title fontsize": 14,
        "subtitle fontsize": 10,
        "legend fontsize": 9,
        "minor ticks": 2,
        "symbol labels": False,
    },
    "publication": {
        "rcParams": {
            "figure.figsize": [6.5, 4.875],
            "figure.dpi": 300,
            "figure.subplot.top": 0.87,
            "savefig.dpi": 300,
            "legend.fontsize": 14,
            "figure.titlesize": 20,
            "axes.titlesize": 14,
            "axes.labelsize": 16,
            "font.size": 16,
            "font.family": "Arial",
            "axes.linewidth": 1.5,
            "legend.framealpha": 0,
            "legend.labelspacing": 0.25,
            "legend.borderpad": 0.2,
            "figure.facecolor": "none",
            "axes.facecolor": "none",
            "xtick.minor.visible": True,
            "ytick.minor.visible": True,
            "xtick.minor.ndivs": 2,
            "ytick.minor.ndivs": 2,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "xtick.major.size": 7,
            "ytick.major.size": 7,
            "xtick.major.width": 1.5,
            "ytick.major.width": 1.5,
            "xtick.minor.size": 4,
            "ytick.minor.size": 4,
            "xtick.minor.width": 1.5,
        },
        "title fontsize": 18,
        "subtitle fontsize": 14,
        "legend fontsize": 14,
        "minor ticks": 2,
        "symbol labels": False,
    },
    "saveant": {
        "rcParams": {
            "figure.figsize": [5.2, 3.75],
            "figure.dpi": 150,
            "figure.subplot.top": 0.92,
            "savefig.dpi": 300,
            "legend.fontsize": 11,
            "figure.titlesize": 14,
            "axes.titlesize": 12,
            "axes.labelsize": 16,
            "font.size": 15,
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "lines.linewidth": 2.0,
            "axes.edgecolor": "black",
            "axes.linewidth": 1.8,
            "legend.framealpha": 0,
            "legend.labelspacing": 0.25,
            "legend.borderpad": 0.2,
            "figure.facecolor": "#d8d8dc",
            "axes.facecolor": "#b8bad8",
            "xtick.minor.visible": True,
            "ytick.minor.visible": True,
            "xtick.minor.ndivs": 5,
            "ytick.minor.ndivs": 5,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "xtick.top": False,
            "ytick.right": False,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "xtick.major.size": 8,
            "ytick.major.size": 8,
            "xtick.major.width": 1.6,
            "ytick.major.width": 1.6,
            "xtick.minor.size": 4,
            "ytick.minor.size": 4,
            "xtick.minor.width": 1.2,
            "ytick.minor.width": 1.2,
        },
        "title fontsize": 14,
        "subtitle fontsize": 11,
        "legend fontsize": 11,
        "minor ticks": 5,
        "axis labels": "inside",
        "compact axis labels": True,
        "symbol labels": True,
        "snap bounds to ticks": True,
        "title axes top": 0.86,
        "suptitle y": 0.975,
        "subtitle pad": 8,
    },
}
_ECAT_ACTIVE_PLOT_STYLE_PROFILE = "notebook"


def _normalize_plot_style_profile(profile):
    profile_key = str(profile).strip().lower().replace("_", "-")
    aliases = {
        "notebook": "notebook",
        "jupyter": "notebook",
        "vscode": "notebook",
        "vs-code": "notebook",
        "publication": "publication",
        "pub": "publication",
        "paper": "publication",
        "saveant": "saveant",
        "savéant": "saveant",
        "saveant-style": "saveant",
        "savéant-style": "saveant",
    }
    if profile_key not in aliases:
        choices = ", ".join(sorted(_ECAT_PLOT_STYLE_PROFILES))
        raise ValueError(f"Unknown eCAT plot style profile '{profile}'. Choose one of: {choices}.")
    return aliases[profile_key]


def _active_plot_style_value(key):
    profile = _ECAT_PLOT_STYLE_PROFILES.get(_ECAT_ACTIVE_PLOT_STYLE_PROFILE, {})
    return profile.get(key)


def _apply_plotting_style(profile="notebook"):
    global _ECAT_ACTIVE_PLOT_STYLE_PROFILE
    profile = _normalize_plot_style_profile(profile)
    _ECAT_ACTIVE_PLOT_STYLE_PROFILE = profile
    plt.rcParams.update(_ECAT_PLOT_STYLE_PROFILES[profile]["rcParams"])


def plotting_style(style=True):
    """Enable, disable, or switch eCAT's package-level Matplotlib style.

    Parameters
    ----------
    style : bool or str, optional
        ``True`` enables the default notebook profile, ``False`` restores
        Matplotlib defaults, and a string selects ``"notebook"``,
        ``"publication"``, ``"saveant"``, ``"default"``, or ``"matplotlib"``. See
        ``e.describe_options("plot")`` for related plotting options.

    Returns
    -------
    str
        Active eCAT plotting style profile, or ``"matplotlib"`` when disabled.

    Examples
    --------
    >>> e.plotting_style("publication")
    """
    global _ECAT_ACTIVE_PLOT_STYLE_PROFILE
    if isinstance(style, str):
        style_key = style.strip().lower().replace("_", "-")
        if style_key in {"default", "matplotlib", "off", "false", "none"}:
            plt.rcParams.update(_MATPLOTLIB_PLOT_STYLE_DEFAULTS)
            _ECAT_ACTIVE_PLOT_STYLE_PROFILE = "notebook"
            return "matplotlib"
        profile = _normalize_plot_style_profile(style)
        _apply_plotting_style(profile)
        return profile
    if style:
        _apply_plotting_style("notebook")
        return "notebook"

    plt.rcParams.update(_MATPLOTLIB_PLOT_STYLE_DEFAULTS)
    _ECAT_ACTIVE_PLOT_STYLE_PROFILE = "notebook"
    return "matplotlib"
