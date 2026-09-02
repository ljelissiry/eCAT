"""Simulation helpers for eCAT CV workflows.

The public API is intentionally small: build a simulation input from a synthetic
CV program or an existing eCAT CV, then run an ElectroKitty-backed simulation.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
import pprint
import re
from typing import Any
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ._progress import NotebookProgressDisplay, progress_enabled
from .objects import cv as _EcatCV


_DEFAULT_CV_DATA_POINTS_PER_VOLT = 1000
_DEFAULT_CV_DATA_MIN_POINTS = 300
_DEFAULT_CV_DATA_MAX_POINTS = 3000
_DEFAULT_FAST_K0_INIT = 1e-3
_MIN_POSITIVE_FIT_BOUND = 1e-30
_DEFAULT_ACTIVITY_STANDARD_CONCENTRATION = 1000.0
_DEFAULT_ACTIVITY_STANDARD_COVERAGE = 1.0
_ACTIVITY_DISPLAY_RTOL = 1e-12

KINETIC_FALLBACKS = {
    "E0": 0.0,
    "k0": 1e-3,
    "alpha": 0.5,
}

K0_PRESETS = {
    # SI units, m/s. These are practical starting values, not universal regime boundaries.
    "fast": 1e-3,
    "reversible": 1e-3,
    "rev": 1e-3,
    "quasi": 1e-5,
    "quasireversible": 1e-5,
    "quasi reversible": 1e-5,
    "slow": 1e-8,
    "irreversible": 1e-8,
    "irrev": 1e-8,
}

SPATIAL_PRESETS = {
    "fast": {"dx_fraction": 0.005, "nx": 8, "viscosity": 1e-6, "rotation": 0.0},
    "balanced": {"dx_fraction": 0.001, "nx": 12, "viscosity": 1e-6, "rotation": 0.0},
    "accurate": {"dx_fraction": 0.001 / 36, "nx": 20, "viscosity": 1e-6, "rotation": 0.0},
}

_SPATIAL_PRESET_ALIASES = {
    "quick": "fast",
    "explore": "fast",
    "exploration": "fast",
    "phoh": "fast",
    "default": "accurate",
    "standard": "accurate",
}

SOLVENT_VISCOSITIES = {
    # Approximate kinematic viscosities at room temperature in m^2/s.
    "MeCN": 4.7e-7,
    "DMF": 8.5e-7,
    "DMSO": 1.82e-6,
    "H2O": 8.9e-7,
    "DCM": 3.1e-7,
    "DME": 5.3e-7,
    "THF": 5.1e-7,
    "MeOH": 6.9e-7,
    "EtOH": 1.36e-6,
    "PC": 2.1e-6,
    "toluene": 6.5e-7,
}

_SOLVENT_ALIASES = {
    "acetonitrile": "MeCN",
    "mecn": "MeCN",
    "ch3cn": "MeCN",
    "n,n-dimethylformamide": "DMF",
    "n,n dimethylformamide": "DMF",
    "dimethylformamide": "DMF",
    "dmf": "DMF",
    "dimethyl sulfoxide": "DMSO",
    "dmso": "DMSO",
    "water": "H2O",
    "h2o": "H2O",
    "dichloromethane": "DCM",
    "methylene chloride": "DCM",
    "dcm": "DCM",
    "ch2cl2": "DCM",
    "dimethoxyethane": "DME",
    "1,2 dimethoxyethane": "DME",
    "glyme": "DME",
    "dme": "DME",
    "tetrahydrofuran": "THF",
    "thf": "THF",
    "methanol": "MeOH",
    "meoh": "MeOH",
    "ethanol": "EtOH",
    "etoh": "EtOH",
    "propylene carbonate": "PC",
    "pc": "PC",
    "toluene": "toluene",
}

_ECAT_SIMULATION_INSTALL_MESSAGE = (
    "ElectroKitty is required for eCAT simulation. Install the optional "
    "dependency with `pip install 'ecat[simulation]'` or install "
    "`electrokitty` in the active environment."
)

@dataclass
class SimulatedCVInput:
    """Potential/time/current arrays used as simulation input."""

    E: np.ndarray
    t: np.ndarray
    i: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    source: Any = None

    def __post_init__(self):
        self.E = np.asarray(self.E, dtype=float)
        self.t = np.asarray(self.t, dtype=float)
        self.metadata = dict(self.metadata or {})
        incubation_time = float(self.metadata.get("incubation_time", 0.0) or 0.0)
        if incubation_time < 0:
            raise ValueError("incubation_time cannot be negative.")
        self.metadata["incubation_time"] = incubation_time
        if self.i is not None:
            self.i = np.asarray(self.i, dtype=float)
            if len(self.i) != len(self.E):
                raise ValueError("SimulatedCVInput current array must match the potential array length.")
        if len(self.t) != len(self.E):
            raise ValueError("SimulatedCVInput time array must match the potential array length.")

    @property
    def has_current(self):
        return self.i is not None

    def show(self, options=None):
        """Display a compact setup summary for this simulated CV input."""
        _maybe_print_simulated_cv_input_setup(self, options, default=True)

    def plot(self, options=None):
        """Plot the input waveform, usually potential versus time."""
        return _plot_simulated_cv_input(self, options)

    def with_scan_rate(self, scan_rate):
        """Return a copy of this CV input with the same waveform at a new scan rate."""
        scan_rate = float(scan_rate)
        if scan_rate <= 0:
            raise ValueError("scan_rate must be positive.")

        metadata = dict(self.metadata or {})
        metadata["scan_rate"] = scan_rate
        metadata["quiet_time_applied"] = False
        return SimulatedCVInput(
            E=self.E.copy(),
            t=_time_from_potential(self.E, scan_rate),
            i=None if self.i is None else self.i.copy(),
            metadata=metadata,
            source=self.source,
        )

    def with_incubation_time(self, incubation_time):
        """Return a copy of this input with a new chemical incubation time."""
        incubation_time = float(incubation_time)
        if incubation_time < 0:
            raise ValueError("incubation_time cannot be negative.")
        metadata = dict(self.metadata or {})
        metadata["incubation_time"] = incubation_time
        return SimulatedCVInput(
            E=self.E.copy(),
            t=self.t.copy(),
            i=None if self.i is None else self.i.copy(),
            metadata=metadata,
            source=self.source,
        )


@dataclass
class MechanismSpec:
    """Compiled mechanism metadata used by the ElectroKitty adapter."""

    mechanism: str
    preset: str
    note: str | None = None
    surface_confined: bool = False
    raw: bool = False
    steps: list[dict[str, Any]] = field(default_factory=list)


class SimulatedCV(_EcatCV):
    """Notebook-friendly result returned by :func:`simulate_cv`."""

    def __new__(cls, *args, **kwargs):
        return object.__new__(cls)

    def __init__(
        self,
        *,
        data,
        params,
        input_params=None,
        mechanism,
        input,
        backend_result=None,
        current_sign=1,
        figure=None,
        axes=None,
        summary=None,
    ):
        self.data = data
        self.params = params
        self.input_params = deepcopy(params if input_params is None else input_params)
        self.mechanism = mechanism
        self.input = input
        self.backend_result = backend_result
        self.current_sign = current_sign
        self.figure = figure
        self.axes = axes
        self.summary = {} if summary is None else summary
        self.filepath = None
        self.options = {}
        self.timestamp = None
        self.creation_time = None
        self.modification_time = None
        self.name = self._simulation_name()
        self.type = "Simulated Cyclic Voltammetry"
        self.software = self.summary.get("backend")
        self.num_x_cols = 1
        self.temperature = self._cell_value("T", 298)
        self.electrode_area = self._cell_value("A", 0)
        self.units = {
            "Potential": _simulation_axis_unit(input, "potential", "V"),
            "Current": _simulation_axis_unit(input, "current", "A"),
            "Time": "s",
            "Backend Current": _simulation_axis_unit(input, "current", "A"),
        }
        self.gas = None
        self.solvent = None
        self.compounds = None
        self.concentrations = None
        self.reference_shift = None
        self.reference_label = None
        self.reference_mode = "none"
        self.reference_source_file = None
        self.reference_pair_details = None
        self.reference_failure_message = None
        self.folderpath = "."
        self.scan_rate = self._scan_rate()
        self.init_E = float(self.data["Potential"].iloc[0]) if len(self.data) else np.nan
        self.final_E = float(self.data["Potential"].iloc[-1]) if len(self.data) else np.nan
        self.min_E = float(np.nanmin(self.data["Potential"])) if len(self.data) else np.nan
        self.max_E = float(np.nanmax(self.data["Potential"])) if len(self.data) else np.nan
        self.segments = self._segments()
        self.delta_x = self._delta_x()

    def with_scan_rate(self, scan_rate, mechanism=None, params=None, backend=None, options=None):
        """Rerun this simulated CV at a new scan rate without modifying the original."""
        input_obj = self.input.with_scan_rate(scan_rate)
        mechanism = self.mechanism if mechanism is None else mechanism
        params = deepcopy(self.input_params if params is None else params)
        backend = self.summary.get("backend", "electrokitty") if backend is None else backend
        options = {"plot": False} if options is None else options
        return simulate_cv(input_obj, mechanism, params, options=options, backend=backend)

    def with_incubation_time(self, incubation_time, mechanism=None, params=None, backend=None, options=None):
        """Rerun this simulated CV after a different chemical incubation time."""
        input_obj = self.input.with_incubation_time(incubation_time)
        mechanism = self.mechanism if mechanism is None else mechanism
        params = deepcopy(self.input_params if params is None else params)
        backend = self.summary.get("backend", "electrokitty") if backend is None else backend
        options = {"plot": False} if options is None else options
        return simulate_cv(input_obj, mechanism, params, options=options, backend=backend)

    def with_params(self, params=None, *, set=None, mechanism=None, input=None, backend=None, options=None):
        """Rerun this simulated CV with deep-merged parameter updates."""
        next_params = deepcopy(self.input_params)
        if params:
            next_params = _deep_merge_dicts(next_params, params)
        if set:
            for path, value in dict(set).items():
                _set_param_path(next_params, _coerce_user_param_path(path), value)
        mechanism = self.mechanism if mechanism is None else mechanism
        input_obj = self.input if input is None else input
        backend = self.summary.get("backend", "electrokitty") if backend is None else backend
        options = {"plot": False} if options is None else options
        return simulate_cv(input_obj, mechanism, next_params, options=options, backend=backend)

    def with_param(self, path, value, *, mechanism=None, input=None, backend=None, options=None):
        """Rerun this simulated CV with one parameter path updated."""
        return self.with_params(
            set={path: value},
            mechanism=mechanism,
            input=input,
            backend=backend,
            options=options,
        )

    def with_input(self, input, *, mechanism=None, params=None, backend=None, options=None):
        """Rerun this simulated CV with a replacement simulation input."""
        mechanism = self.mechanism if mechanism is None else mechanism
        params = deepcopy(self.input_params if params is None else params)
        backend = self.summary.get("backend", "electrokitty") if backend is None else backend
        options = {"plot": False} if options is None else options
        return simulate_cv(input, mechanism, params, options=options, backend=backend)

    def with_mechanism(self, mechanism, *, params=None, input=None, backend=None, options=None):
        """Rerun this simulated CV with a replacement mechanism."""
        params = deepcopy(self.input_params if params is None else params)
        input_obj = self.input if input is None else input
        backend = self.summary.get("backend", "electrokitty") if backend is None else backend
        options = {"plot": False} if options is None else options
        return simulate_cv(input_obj, mechanism, params, options=options, backend=backend)

    def _cell_value(self, key, default):
        cell = self.params.get("cell", {}) if isinstance(self.params, dict) else {}
        return cell.get(key, default)

    def _scan_rate(self):
        metadata = getattr(self.input, "metadata", {}) if self.input is not None else {}
        return float(metadata.get("scan_rate", _scan_rate_from_input(self.input))) if self.input is not None else np.nan

    def _segments(self):
        metadata = getattr(self.input, "metadata", {}) if self.input is not None else {}
        segments = metadata.get("segments")
        if isinstance(segments, (list, tuple, np.ndarray)):
            return len(segments)
        if segments is not None:
            try:
                return int(segments)
            except (TypeError, ValueError):
                pass
        if len(self.data) < 2 or "Potential" not in self.data:
            return 1
        x = self.data["Potential"].to_numpy(dtype=float)
        dx = np.diff(x)
        dx = dx[np.isfinite(dx) & (dx != 0)]
        if len(dx) == 0:
            return 1
        return int(np.count_nonzero(np.diff(np.sign(dx))) + 1)

    def _delta_x(self):
        if len(self.data) < 2 or "Potential" not in self.data:
            return np.nan
        diffs = np.abs(np.diff(self.data["Potential"].to_numpy(dtype=float)))
        diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
        return float(np.nanmedian(diffs)) if len(diffs) else np.nan

    def _simulation_name(self):
        metadata = getattr(self.input, "metadata", {}) if self.input is not None else {}
        name = metadata.get("name")
        if name:
            return f"Simulation of {name}"
        scan_rate = metadata.get("scan_rate")
        if scan_rate is not None:
            return f"Simulated CV {float(scan_rate):g} V/s"
        return "Simulated CV"

    def x(self, options=None):
        """Return simulated CV potential data as a pandas Series."""
        options = _normalize_options(options)
        column = options.get("x axis", options.get("x_axis", "Potential")) or "Potential"
        if str(column).strip().lower() in {"potential", "e"}:
            column = "Potential"
        scale = options.get("x scale", options.get("x_scale", 1))
        series = self.data[column].copy()
        return series * scale if scale != 1 else series

    def y(self, options=None):
        """Return simulated CV current data as a pandas Series."""
        options = _normalize_options(options)
        column = options.get("y axis", options.get("y_axis", "Current")) or "Current"
        if str(column).strip().lower() in {"current", "i"}:
            column = "Current"
        scale = options.get("y scale", options.get("y_scale", 1))
        series = self.data[column].copy()
        return series * scale if scale != 1 else series

    def xy(self, options=None):
        """Return simulated CV potential and current arrays."""
        return self.x(options).to_numpy(dtype=float), self.y(options).to_numpy(dtype=float)

    def plot(self, options=None):
        """Plot simulated current only."""
        return _plot_simulated_cv(self, options)

    def show(self, options=None):
        """Display setup, optional parameters, and optional data for this simulated CV."""
        options = _normalize_options(options)
        _maybe_print_simulated_cv_setup(self, options, default=True)
        _maybe_print_simulation_concentration_states(self, options)
        print_params = options.get("print params", options.get("print_params", False))
        if _truthy_option(print_params):
            param_options = dict(options)
            param_options["print params"] = print_params
            _maybe_print_simulation_params(self.params, param_options)
        _maybe_print_simulation_param_checks(self.params, self.input, self.mechanism, options)
        if _truthy_option(options.get("print data", options.get("print_data", False))):
            _display_dataframe_sections("Simulated CV Data:", [(None, self.data)], options=options)


class SimulationFitResult:
    """Notebook-friendly result returned by :func:`fit_cv`."""

    def __init__(
        self,
        *,
        best_params,
        fit_spec,
        method,
        backend,
        optimizer_result=None,
        initial_result=None,
        simulation_result=None,
        residuals=None,
        corrections=None,
        summary=None,
        backend_result=None,
        measured_current=None,
    ):
        self.best_params = best_params
        self.fit_spec = fit_spec
        self.method = method
        self.backend = backend
        self.optimizer_result = optimizer_result
        self.initial_result = initial_result
        self.simulation_result = simulation_result
        self.residuals = np.asarray([] if residuals is None else residuals, dtype=float)
        self.corrections = {} if corrections is None else corrections
        self.summary = {} if summary is None else summary
        self.backend_result = backend_result
        if measured_current is None:
            measured_current = _measured_current_for_result(simulation_result, required=False)
        self.measured_current = None if measured_current is None else np.asarray(measured_current, dtype=float)

    @property
    def data(self):
        return None if self.simulation_result is None else self.simulation_result.data

    def plot(self, options=None):
        if self.simulation_result is None:
            raise ValueError("No final simulation result is available to plot.")
        return _plot_simulated_cv(self.simulation_result, options, measured_current=self.measured_current)

    def show(self, options=None):
        """Display fit setup, statistics, parameters, and final simulation on request."""
        options = _normalize_options(options)
        defaults = {
            "print stats": True,
            "print corrections": True,
            "print params": True,
            "print simulation": True,
        }
        defaults.update(options)
        options = _normalize_options(defaults)
        if _truthy_option(options.get("print setup", options.get("print_setup", False))):
            self._show_fit_setup(options)
        _maybe_print_fit_statistics(self.summary, self.corrections, options)
        if _truthy_option(options.get("print params", options.get("print_params", False))):
            initial_params = (
                self.initial_result.params
                if self.initial_result is not None
                else (self.fit_spec or {}).get("base_params", {})
            )
            _maybe_print_fit_params(initial_params, self.best_params, options, self.fit_spec)
        if _truthy_option(options.get("print simulation", options.get("print_simulation", False))):
            if self.simulation_result is not None:
                self.simulation_result.show(
                    {
                        "print setup": True,
                        "pretty print": options.get("pretty print", True),
                    }
                )

    def _show_fit_setup(self, options):
        if self.simulation_result is None or not isinstance(self.fit_spec, dict):
            return
        setup_options = dict(options)
        setup_options["print fitting"] = options.get("print setup", options.get("print_setup", True))
        _maybe_print_fitting_setup(
            self.simulation_result.input,
            self.simulation_result.mechanism,
            self.fit_spec,
            setup_options,
            backend=self.backend,
            method=self.method,
            residual_mode=self.summary.get("residual", self.summary.get("residual_mode", "")),
            post_correction_mode=self.summary.get(
                "post_correction",
                self.summary.get("post correction", ""),
            ),
            residual_normalization=self.summary.get(
                "residual_normalization",
                self.summary.get("residual normalization", ""),
            ),
        )


class SimulationGroupFitResult:
    """Notebook-friendly result returned by :func:`fit_cvs`."""

    def __init__(
        self,
        *,
        best_params,
        best_params_by_cv,
        fit_spec,
        per_cv,
        method,
        backend,
        optimizer_result=None,
        initial_results=None,
        simulation_results=None,
        measured_currents=None,
        residuals=None,
        residuals_by_cv=None,
        corrections_by_cv=None,
        summary=None,
        datasets=None,
    ):
        self.best_params = best_params
        self.best_params_by_cv = list(best_params_by_cv or [])
        self.fit_spec = fit_spec
        self.per_cv = per_cv
        self.per_cv_paths = list(per_cv or [])
        self.method = method
        self.backend = backend
        self.optimizer_result = optimizer_result
        self.initial_results = list(initial_results or [])
        self.simulation_results = list(simulation_results or [])
        self.measured_currents = [
            np.asarray(item, dtype=float) for item in (measured_currents or [])
        ]
        self.residuals = np.asarray([] if residuals is None else residuals, dtype=float)
        self.residuals_by_cv = [
            np.asarray(item, dtype=float) for item in (residuals_by_cv or [])
        ]
        self.corrections_by_cv = list(corrections_by_cv or [])
        self.summary = {} if summary is None else summary
        self.datasets = list(datasets or [])

    @property
    def data(self):
        return [result.data for result in self.simulation_results]

    def plot(self, options=None):
        axes = []
        for index, result in enumerate(self.simulation_results):
            plot_options = _normalize_options(options)
            plot_options.setdefault("simulation label", "Simulated Fit")
            plot_options.setdefault("backend label", "Raw Simulated Fit")
            plot_options.setdefault("data label", "Measured Data")
            label = self.datasets[index].get("label") if index < len(self.datasets) else None
            if label and not plot_options.get("title"):
                plot_options["title"] = str(label)
            measured = self.measured_currents[index] if index < len(self.measured_currents) else None
            axes.append(_plot_simulated_cv(result, plot_options, measured_current=measured))
        return axes

    def show(self, options=None):
        options = _normalize_options(options)
        if not options:
            options = {"print setup": True, "print stats": True, "print corrections": True}
        if _truthy_option(options.get("print setup", options.get("print_setup", False))):
            _display_dataframe_sections(
                "Group Fitting Setup:",
                [(None, _group_fit_datasets_dataframe(self.datasets))],
                options=options,
            )
        stats_options = dict(options)
        stats_options["print corrections"] = False
        stats_options["print_corrections"] = False
        _maybe_print_fit_statistics(self.summary, {}, stats_options)
        if _truthy_option(options.get("print corrections", options.get("print_corrections", False))):
            _display_dataframe_sections(
                "Group Fitting Corrections:",
                [(None, _group_fit_corrections_dataframe(self))],
                options=options,
            )
        if _truthy_option(options.get("print params", options.get("print_params", False))):
            _maybe_print_group_fit_params(
                self.fit_spec,
                self.best_params,
                self.best_params_by_cv,
                self.datasets,
                options,
            )
        if _truthy_option(options.get("print simulation", options.get("print_simulation", False))):
            for result in self.simulation_results:
                result.show(
                    {
                        "print setup": True,
                        "pretty print": options.get("pretty print", True),
                    }
                )


def _plot_simulated_cv_input(input_obj, options=None):
    options = _normalize_options(options)
    ax = options.get("ax")
    if ax is None:
        if options.get("new plot", options.get("new_plot", True)):
            _, ax = _ecat_pyplot().subplots()
        else:
            ax = _ecat_pyplot().gca()

    plot_input = _input_with_plot_quiet_time(input_obj, options)
    x_key = _normalize_input_plot_axis(options.get("x axis", options.get("x_axis", "time")))
    y_key = _normalize_input_plot_axis(options.get("y axis", options.get("y_axis", "potential")))
    x, x_name, x_unit = _simulated_cv_input_axis(plot_input, x_key)
    y, y_name, y_unit = _simulated_cv_input_axis(plot_input, y_key)

    x_scale, x_unit = _ecat_plot_axis_scale(x, x_name, x_unit, options.get("x unit", "auto"))
    y_scale, y_unit = _ecat_plot_axis_scale(y, y_name, y_unit, options.get("y unit", "auto"))

    ax.plot(
        x * x_scale,
        y * y_scale,
        color=options.get("color"),
        label=options.get("label", options.get("program label", options.get("input label", "CV Program"))),
        linestyle=options.get("linestyle", options.get("program linestyle", "-")),
    )

    ax.set_xlabel(options.get("xlabel") or _ecat_plot_axis_label(x_name, x_unit))
    ax.set_ylabel(options.get("ylabel") or _ecat_plot_axis_label(y_name, y_unit))
    _apply_ecat_axis_style(ax, options)
    if options.get("autoscale", True):
        ax.relim()
        ax.autoscale(enable=True, axis="both", tight=False)
        ax.margins(
            x=options.get("x margin", options.get("x_margin", 0.05)),
            y=options.get("y margin", options.get("y_margin", 0.05)),
        )
        ax.autoscale_view()
    invert_default = x_key == "potential"
    if options.get("invert x", options.get("invert_x", invert_default)):
        x0, x1 = ax.get_xlim()
        if x0 < x1:
            ax.invert_xaxis()
    if options.get("legend", False):
        ax.legend()
    return ax


def _input_with_plot_quiet_time(input_obj, options):
    if not _truthy_option(options.get("plot quiet time", options.get("plot_quiet_time", False))):
        return input_obj
    quiet_time = _input_quiet_time(input_obj, options)
    if quiet_time <= 0 or len(input_obj.E) == 0:
        return input_obj

    t = np.asarray(input_obj.t, dtype=float)
    E = np.asarray(input_obj.E, dtype=float)
    dt = _representative_dt(t)
    hold_points = max(2, int(np.ceil(quiet_time / dt)) + 1)
    t_hold = np.linspace(-quiet_time, 0.0, hold_points)
    E_hold = np.full(hold_points, E[0], dtype=float)
    extended_E = np.concatenate([E_hold, E[1:]])
    extended_t = np.concatenate([t_hold, t[1:]])
    extended_i = None
    if input_obj.i is not None:
        measured = np.asarray(input_obj.i, dtype=float)
        i_hold = np.full(hold_points, measured[0], dtype=float)
        extended_i = np.concatenate([i_hold, measured[1:]])
    metadata = dict(getattr(input_obj, "metadata", {}) or {})
    metadata["quiet_time"] = quiet_time
    metadata["quiet_time_applied"] = True
    return SimulatedCVInput(
        E=extended_E,
        t=extended_t,
        i=extended_i,
        metadata=metadata,
        source=input_obj.source,
    )


def _normalize_input_plot_axis(value):
    key = _canonical_token(value)
    aliases = {
        "e": "potential",
        "potential": "potential",
        "voltage": "potential",
        "v": "potential",
        "t": "time",
        "time": "time",
        "i": "current",
        "current": "current",
    }
    if key not in aliases:
        raise ValueError("SimulatedCVInput plot axes must be 'time', 'potential', or 'current'.")
    return aliases[key]


def _simulated_cv_input_axis(input_obj, axis):
    if axis == "time":
        return np.asarray(input_obj.t, dtype=float), "Time", "s"
    if axis == "potential":
        return (
            np.asarray(input_obj.E, dtype=float),
            _simulation_axis_name(input_obj, "potential", "Potential"),
            _simulation_axis_unit(input_obj, "potential", "V"),
        )
    if axis == "current":
        if input_obj.i is None:
            raise ValueError("SimulatedCVInput has no current data to plot.")
        return (
            np.asarray(input_obj.i, dtype=float),
            _simulation_axis_name(input_obj, "current", "Current"),
            _simulation_axis_unit(input_obj, "current", "A"),
        )
    raise ValueError("SimulatedCVInput plot axes must be 'time', 'potential', or 'current'.")


def _plot_simulated_cv(result, options=None, measured_current=None):
    options = _normalize_options(options)
    ax = options.get("ax")
    if ax is None:
        if options.get("new plot", options.get("new_plot", True)):
            _, ax = _ecat_pyplot().subplots()
        else:
            ax = _ecat_pyplot().gca()

    x = result.data["Potential"].to_numpy(dtype=float)
    x_name = _simulation_axis_name(result.input, "potential", "Potential")
    x_unit = _simulation_axis_unit(result.input, "potential", "V")
    x_scale, x_unit = _ecat_plot_axis_scale(x, x_name, x_unit, options.get("x unit", "auto"))
    x_plot = x * x_scale

    current_arrays = [result.data["Current"].to_numpy(dtype=float)]
    measured = None
    if measured_current is not None:
        measured = np.asarray(measured_current, dtype=float)
        if len(measured) != len(x):
            raise ValueError("Measured current length must match the simulated CV length.")
        current_arrays.append(measured)
    if "Backend Current" in result.data:
        current_arrays.append(result.data["Backend Current"].to_numpy(dtype=float))
    y_for_scale = np.concatenate(current_arrays) if current_arrays else result.data["Current"].to_numpy(dtype=float)
    y_name = _simulation_axis_name(result.input, "current", "Current")
    y_unit = _simulation_axis_unit(result.input, "current", "A")
    y_scale, y_unit = _ecat_plot_axis_scale(y_for_scale, y_name, y_unit, options.get("y unit", "auto"))

    y = result.data["Current"].to_numpy(dtype=float)
    simulation_linestyle = options.get("linestyle")
    if simulation_linestyle is None:
        simulation_linestyle = options.get(
            "simulation linestyle",
            options.get("simulation_linestyle", "--"),
        )
    ax.plot(
        x_plot,
        y * y_scale,
        color=options.get("color"),
        label=options.get("label", options.get("simulation label", "Simulation")),
        linestyle=simulation_linestyle,
    )

    if measured is not None:
        ax.plot(
            x_plot,
            measured * y_scale,
            label=options.get("data label", "Data"),
            linestyle=options.get("data linestyle", options.get("data_linestyle", "-")),
        )

    if options.get("plot all", options.get("plot_all", False)) and "Backend Current" in result.data:
        backend = result.data["Backend Current"].to_numpy(dtype=float)
        ax.plot(
            x_plot,
            backend * y_scale,
            label=options.get("backend label", "Backend Current"),
            linestyle=options.get("backend linestyle", options.get("backend_linestyle", ":")),
        )

    ax.set_xlabel(options.get("xlabel") or _ecat_plot_axis_label(x_name, x_unit))
    ax.set_ylabel(options.get("ylabel") or _ecat_plot_axis_label(y_name, y_unit))
    _apply_ecat_axis_style(ax, options)
    if options.get("autoscale", True):
        ax.relim()
        ax.autoscale(enable=True, axis="both", tight=False)
        ax.margins(
            x=options.get("x margin", options.get("x_margin", 0.05)),
            y=options.get("y margin", options.get("y_margin", 0.05)),
        )
        ax.autoscale_view()
    if options.get("invert x", True):
        x0, x1 = ax.get_xlim()
        if x0 < x1:
            ax.invert_xaxis()
    if options.get("legend", "auto") == "auto":
        if measured is not None:
            ax.legend()
    elif options.get("legend"):
        ax.legend()

    result.axes = ax
    result.figure = ax.figure
    return ax


class _FitStrategyOptimizerResult:
    """Small strategy-level wrapper around scipy optimizer results."""

    def __init__(
        self,
        *,
        x=None,
        cost=None,
        success=None,
        message="",
        nfev=0,
        strategy=None,
        optimizer=None,
        raw_result=None,
        children=None,
        best_start_index=None,
        pre_polish_result=None,
        polish_result=None,
    ):
        self.x = np.asarray([] if x is None else x, dtype=float)
        self.cost = cost
        self.success = success
        self.message = message
        self.nfev = int(0 if nfev is None else nfev)
        self.strategy = strategy
        self.optimizer = optimizer
        self.raw_result = raw_result
        self.children = [] if children is None else list(children)
        self.best_start_index = best_start_index
        self.pre_polish_result = pre_polish_result
        self.polish_result = polish_result


def cv_program(
    Ei,
    E_low=None,
    E_high=None,
    Ef=None,
    scan_rate=0.1,
    direction="negative",
    segments=2,
    points_per_segment=1000,
    quiet_time=0.0,
    incubation_time=0.0,
):
    """Create a synthetic CV potential program as a :class:`SimulatedCVInput`."""
    Ei = float(Ei)
    E_low = None if E_low is None else float(E_low)
    E_high = None if E_high is None else float(E_high)
    Ef = None if Ef is None else float(Ef)
    scan_rate = float(scan_rate)
    direction = _normalize_cv_direction(direction)
    segments = int(segments)
    points_per_segment = int(points_per_segment)
    quiet_time = float(quiet_time)
    incubation_time = float(incubation_time)

    if E_low is None and E_high is None:
        raise ValueError("cv_program requires E_low, E_high, or both.")
    if scan_rate <= 0:
        raise ValueError("scan_rate must be positive.")
    if segments < 1:
        raise ValueError("segments must be at least 1.")
    if points_per_segment < 2:
        raise ValueError("points_per_segment must be at least 2.")
    if quiet_time < 0:
        raise ValueError("quiet_time cannot be negative.")
    if incubation_time < 0:
        raise ValueError("incubation_time cannot be negative.")

    vertices = _cv_program_vertices(Ei, E_low, E_high, Ef, direction, segments)
    E = _joined_segments(
        *[
            np.linspace(start, stop, points_per_segment)
            for start, stop in zip(vertices[:-1], vertices[1:])
        ]
    )

    t = _time_from_potential(E, scan_rate)

    return SimulatedCVInput(
        E=E,
        t=t,
        i=None,
        metadata={
            "kind": "cv_program",
            "Ei": Ei,
            "E_low": E_low,
            "E_high": E_high,
            "Ef": vertices[-1],
            "scan_rate": scan_rate,
            "direction": direction,
            "segments": segments,
            "points_per_segment": points_per_segment,
            "quiet_time": quiet_time,
            "quiet_time_applied": False,
            "incubation_time": incubation_time,
            "potential_axis_name": "Potential",
            "potential_unit": "V",
            "current_axis_name": "Current",
            "current_unit": "A",
        },
        source="program",
    )


def _normalize_cv_direction(direction):
    text = str(direction).strip().lower().replace("_", " ").replace("-", " ")
    aliases = {
        "negative": "negative",
        "neg": "negative",
        "-": "negative",
        "-1": "negative",
        "cathodic": "negative",
        "cathode": "negative",
        "reduction": "negative",
        "reducing": "negative",
        "positive": "positive",
        "pos": "positive",
        "+": "positive",
        "+1": "positive",
        "anodic": "positive",
        "anode": "positive",
        "oxidation": "positive",
        "oxidizing": "positive",
    }
    if text not in aliases:
        raise ValueError("direction must be 'negative'/'cathodic'/'-' or 'positive'/'anodic'/'+'.")
    return aliases[text]


def _cv_program_vertices(Ei, E_low, E_high, Ef, direction, segments):
    first = E_low if direction == "negative" else E_high
    second = E_high if direction == "negative" else E_low
    if first is None:
        raise ValueError(
            "cv_program direction requires the matching first vertex: "
            "E_low for negative/cathodic scans or E_high for positive/anodic scans."
        )
    if second is None:
        second = Ei

    vertices = [Ei]
    for index in range(segments):
        vertices.append(first if index % 2 == 0 else second)
    if Ef is not None:
        vertices[-1] = Ef
    return vertices


def cv_data(cv, options=None):
    """Extract measured eCAT CV data into a :class:`SimulatedCVInput`."""
    options = _normalize_options(options)
    segment_options = {}
    if "segments" in options:
        segment_options["segments"] = options["segments"]
    if "segment" in options:
        segment_options["segment"] = options["segment"]

    E, i = cv.analysis_segment_data(segment_options)
    E = np.asarray(E, dtype=float)
    i = np.asarray(i, dtype=float)
    E_for_estimate = E.copy()
    i_for_estimate = i.copy()
    potential_axis_name = "Potential"
    current_axis_name = "Current"
    potential_unit = "V"
    current_unit = "A"
    try:
        source_x = cv.x(segment_options)
        potential_axis_name = getattr(source_x, "name", None) or potential_axis_name
        potential_unit = getattr(cv, "units", {}).get(potential_axis_name, potential_unit)
    except Exception:
        pass
    try:
        source_y = cv.y(segment_options)
        current_axis_name = getattr(source_y, "name", None) or current_axis_name
        current_unit = getattr(cv, "units", {}).get(current_axis_name, current_unit)
    except Exception:
        pass
    source_current_unit = current_unit
    i_for_estimate, current_unit = _current_values_in_amps(i_for_estimate, current_axis_name, source_current_unit)
    i, current_unit = _current_values_in_amps(i, current_axis_name, source_current_unit)

    potential_window = options.get("potential window", options.get("potential_window"))
    window_info = _cv_data_window_info(E, potential_window, options)
    if potential_window is not None:
        mask = window_info["mask"]
        E = E[mask]
        i = i[mask]

    if len(E) == 0:
        raise ValueError("cv_data selected no points.")

    i, background_info = _cv_data_background_corrected_current(cv, E, i, options)

    original_points = len(E)
    stride_info = _cv_data_stride_info(E, options)
    E = E[stride_info["indices"]]
    i = i[stride_info["indices"]]

    scan_rate = float(options.get("scan rate", getattr(cv, "scan_rate", np.nan)))
    if not np.isfinite(scan_rate) or scan_rate <= 0:
        raise ValueError("cv_data requires a positive scan rate.")

    metadata = {
        "kind": "cv_data",
        "name": getattr(cv, "name", None),
        "scan_rate": scan_rate,
        "segments": options.get("segments", options.get("segment")),
        "potential_window": list(potential_window) if potential_window is not None else None,
        "potential_window_requested": window_info["requested"],
        "potential_window_effective": window_info["effective"],
        "trim_mode": window_info["mode"],
        "window_expanded": window_info["expanded"],
        "window_break_count": window_info["break_count"],
        "quiet_time": _source_quiet_time(cv),
        "quiet_time_applied": False,
        "incubation_time": _cv_data_incubation_time(options),
        "potential_axis_name": potential_axis_name,
        "potential_unit": potential_unit,
        "current_axis_name": current_axis_name,
        "current_unit": current_unit,
        "stride": stride_info["stride"],
        "stride_mode": stride_info["mode"],
        "stride_basis": stride_info["basis"],
        "points_per_volt": stride_info["points_per_volt"],
        "target_points": stride_info["target_points"],
        "original_points": original_points,
        "selected_points": len(E),
    }
    metadata.update(background_info)
    estimate_cdl = options.get("estimate Cdl", options.get("estimate_cdl", "auto"))
    if not _falsey_option(estimate_cdl):
        try:
            cdl, diagnostics = estimate_cdl_from_cv_arrays(E_for_estimate, i_for_estimate, scan_rate, options)
            metadata["estimated_cdl"] = cdl
            metadata["estimated_Cdl"] = cdl
            metadata["estimated_cdl_diagnostics"] = diagnostics
        except Exception as exc:
            if not _is_auto_value(estimate_cdl) and _truthy_option(estimate_cdl):
                raise
            metadata["estimated_cdl_error"] = str(exc)

    input_obj = SimulatedCVInput(
        E=E,
        i=i,
        t=_time_from_potential(E, scan_rate),
        metadata=metadata,
        source=cv,
    )
    _maybe_print_simulated_cv_input_setup(input_obj, options, default=False)
    return input_obj


def _cv_data_incubation_time(options):
    value = options.get("incubation time", options.get("incubation_time", 0.0))
    value = float(0.0 if value is None else value)
    if value < 0:
        raise ValueError("incubation_time cannot be negative.")
    return value


def _cv_data_background_corrected_current(cv, E, i, options):
    mode = _cv_data_background_correction_mode(options)
    if mode is None:
        return i, {}

    E = np.asarray(E, dtype=float)
    i = np.asarray(i, dtype=float)
    if len(i) == 0:
        return i, {}

    if mode == "start current":
        start_current = float(i[0])
        return i - start_current, {
            "background_correction": "start current",
            "background_correction_applied": True,
            "background_correction_points": int(len(i)),
            "background_current": start_current,
        }

    tangent_potential = options.get("tangent potential", options.get("tangent_potential"))
    target_potential = _cv_data_background_target_potential(options)
    tangent_options = dict(options)
    tangent_options["print"] = False
    tangent_options["internal call"] = True
    tangent_options["new plot"] = False
    tangent_options["plot all"] = False

    try:
        if tangent_potential is not None:
            tangent_meta = cv._fit_tangent_line(
                E,
                i,
                idx_target=None,
                tangent_potential=float(tangent_potential),
                options=tangent_options,
            )
            anchor = float(tangent_potential)
        elif target_potential is not None:
            idx_target = int(np.argmin(np.abs(E - float(target_potential))))
            tangent_meta = cv._fit_tangent_line(
                E,
                i,
                idx_target=idx_target,
                tangent_potential=None,
                options=tangent_options,
            )
            anchor = float(target_potential)
        else:
            raise ValueError(
                "'background correction': 'tangent' requires 'tangent potential', "
                "'peak potential', 'exact potential', or 'guess potential'."
            )
    except Exception as exc:
        if isinstance(exc, ValueError) and "background correction" in str(exc):
            raise
        raise ValueError(
            "Could not fit tangent background for simulation.cv_data. "
            "Set 'tangent potential' near a baseline region or use "
            "{'background correction': 'start current'}."
        ) from exc

    slope = float(tangent_meta["slope"])
    intercept = float(tangent_meta["intercept"])
    background = slope * E + intercept
    return i - background, {
        "background_correction": "tangent",
        "background_correction_applied": True,
        "background_correction_points": int(len(i)),
        "background_tangent_potential": anchor,
        "background_slope": slope,
        "background_intercept": intercept,
        "background_fit_indices": [int(idx) for idx in np.asarray(tangent_meta.get("fit_indices", []), dtype=int)],
    }


def _cv_data_background_correction_mode(options):
    value = options.get("background correction", options.get("background_correction"))
    if value is None or _falsey_option(value):
        return None
    token = _canonical_token(value)
    aliases = {
        "start_current": "start current",
        "start": "start current",
        "initial_current": "start current",
        "initial": "start current",
        "first_current": "start current",
        "first": "start current",
        "t0": "start current",
        "t_0": "start current",
        "time_0": "start current",
        "time_zero": "start current",
        "tangent": "tangent",
        "tangent_line": "tangent",
        "linear_tangent": "tangent",
    }
    if token in aliases:
        return aliases[token]
    raise ValueError("'background correction' must be None, 'start current', or 'tangent'.")


def _cv_data_background_target_potential(options):
    for key in ("peak potential", "peak_potential", "exact potential", "exact_potential", "guess potential", "guess_potential"):
        if key in options and options[key] is not None:
            return options[key]
    return None


def _cv_data_window_info(E, potential_window, options):
    E = np.asarray(E, dtype=float)
    if potential_window is None:
        return {
            "mask": np.ones(len(E), dtype=bool),
            "requested": None,
            "effective": None,
            "mode": None,
            "expanded": False,
            "break_count": 0,
        }

    requested = [float(potential_window[0]), float(potential_window[1])]
    lo, hi = sorted(requested)
    mask = (E >= lo) & (E <= hi)
    selected = np.flatnonzero(mask)
    if len(selected) == 0:
        return {
            "mask": mask,
            "requested": requested,
            "effective": None,
            "mode": _cv_data_window_mode(options),
            "expanded": False,
            "break_count": 0,
        }

    breaks = np.where(np.diff(selected) > 1)[0]
    break_count = int(len(breaks))
    mode = _cv_data_window_mode(options)
    expanded = False
    if break_count:
        if mode in {"strict", "error", "reject"}:
            raise ValueError(
                "The requested potential window would disconnect the CV scan history. "
                "Use mode='expand' to preserve a connected waveform, or "
                "mode='pointwise' to keep the pointwise trim."
            )
        if mode == "expand":
            connected_mask = np.zeros(len(E), dtype=bool)
            connected_mask[selected[0] : selected[-1] + 1] = True
            mask = connected_mask
            expanded = True

    effective = [float(np.nanmin(E[mask])), float(np.nanmax(E[mask]))] if np.any(mask) else None
    return {
        "mask": mask,
        "requested": requested,
        "effective": effective,
        "mode": mode,
        "expanded": expanded,
        "break_count": break_count,
    }


def _cv_data_window_mode(options):
    removed = [key for key in ("mode", "window mode", "window_mode") if key in options]
    if removed:
        raise ValueError(
            "Removed CV trim option(s): "
            + ", ".join(repr(key) for key in removed)
            + ". Use 'trim mode' with 'expand', 'pointwise', or 'strict'."
        )
    mode = options.get("trim mode", options.get("trim_mode", "expand"))
    mode = _canonical_token(mode)
    if mode not in {"expand", "strict", "pointwise"}:
        raise ValueError("trim mode must be 'expand', 'pointwise', or 'strict'.")
    return mode


def _mechanism_spec(mechanism, preset, **kwargs):
    return MechanismSpec(
        mechanism=mechanism,
        preset=preset,
        steps=_mechanism_steps(mechanism),
        **kwargs,
    )


def _mechanism_steps(mechanism):
    steps = []
    counts = {"E": 0, "C": 0}
    for global_index, line in enumerate(str(mechanism or "").splitlines()):
        text = line.strip()
        if not text or ":" not in text:
            continue
        prefix, equation = text.split(":", 1)
        kind = prefix.strip()[:1].upper()
        if kind not in {"E", "C"}:
            continue
        group_index = counts[kind]
        counts[kind] += 1
        equation = equation.strip()
        steps.append(
            {
                "kind": kind,
                "index": group_index,
                "global_index": global_index,
                "line": text,
                "equation": equation,
                "key": _reaction_key(equation),
            }
        )
    return steps


def _reaction_key(reaction):
    left, separator, right = _split_reaction_equation(reaction)
    return (
        f"{_canonical_stoichiometric_side(left)}"
        f"{separator}"
        f"{_canonical_stoichiometric_side(right)}"
    ).lower()


def _normalize_reaction_arrows(reaction):
    text = str(reaction or "").strip()
    text = text.replace("⇌", "=").replace("<=>", "=").replace("<->", "=")
    return text.replace("→", ">").replace("->", ">")


def _split_reaction_equation(reaction):
    normalized = _normalize_reaction_arrows(reaction)
    separators = [separator for separator in ("=", "<", ">") if normalized.count(separator) == 1]
    if len(separators) != 1:
        raise ValueError("reaction must contain exactly one '=', '<', or '>' separator.")
    separator = separators[0]
    left, right = normalized.split(separator, 1)
    if not left.strip() or not right.strip():
        raise ValueError("reaction must have reactants and products.")
    return left, separator, right


def _parse_stoichiometric_term(term):
    text = str(term or "").strip()
    if not text:
        raise ValueError("Reaction species terms cannot be empty.")
    if re.match(r"^\d+\.\d+", text):
        raise ValueError("Only positive integer stoichiometric coefficients are supported.")
    match = re.match(r"^(\d+)?\s*(.+?)\s*$", text)
    if not match:
        raise ValueError(f"Could not parse reaction term {text!r}.")
    coefficient_text, species = match.groups()
    coefficient = int(coefficient_text) if coefficient_text else 1
    species = species.replace(" ", "")
    if coefficient <= 0 or not species:
        raise ValueError("Only positive integer stoichiometric coefficients are supported.")
    return species, coefficient


def _canonical_stoichiometric_side(side):
    coefficients = {}
    for raw_term in str(side).split("+"):
        species, coefficient = _parse_stoichiometric_term(raw_term)
        coefficients[species] = coefficients.get(species, 0) + coefficient
    return "+".join(
        f"{coefficient if coefficient != 1 else ''}{species}"
        for species, coefficient in sorted(coefficients.items(), key=lambda item: item[0].lower())
    )


def _mechanism_steps_of_kind(mechanism_spec, kind):
    kind = str(kind).upper()
    steps = getattr(mechanism_spec, "steps", None) or _mechanism_steps(getattr(mechanism_spec, "mechanism", ""))
    return [step for step in steps if step.get("kind") == kind]


def _source_quiet_time(source):
    quiet_time = _first_finite_attr(source, ["quiet_time", "quiet time", "quiet_time_s"])
    if quiet_time is None:
        return 0.0
    return float(quiet_time)


def compile_mechanism(mechanism, params=None):
    """Compile an eCAT mechanism preset or custom eCAT mechanism string."""
    if isinstance(mechanism, MechanismSpec):
        return mechanism
    params = {} if params is None else params
    text = str(mechanism).strip()
    normalized = text.lower().replace(" ", "")
    concentrations = normalize_concentrations(_concentrations_from_params(params))

    if ":" in text:
        return _mechanism_spec(text, "raw", raw=True)

    if "," in normalized and "*" in normalized:
        raise ValueError(
            "Mixed surface/bulk shorthand labels are not supported. "
            "Use a custom eCAT mechanism string for mixed mechanisms."
        )

    force_surface = normalized.endswith("*")
    if force_surface:
        normalized = normalized[:-1]

    surface_confined = force_surface or (
        bool(concentrations["surface"]) and not bool(concentrations["bulk"])
    )

    if normalized == "e":
        return _mechanism_spec(
            _compile_e_preset(concentrations, surface_confined),
            "E",
            surface_confined=surface_confined,
        )
    if normalized in {"ee", "e,e"}:
        return _mechanism_spec(
            _compile_ee_preset(concentrations, surface_confined),
            "EE",
            surface_confined=surface_confined,
        )
    if normalized == "ec":
        return _mechanism_spec(
            _compile_ec_preset(concentrations, surface_confined),
            "EC",
            surface_confined=surface_confined,
        )
    if normalized == "ece":
        return _mechanism_spec(
            _compile_ece_preset(concentrations, surface_confined),
            "ECE",
            surface_confined=surface_confined,
        )
    if normalized in {"square", "squarescheme", "square-scheme", "square_scheme", "sq"}:
        return _mechanism_spec(
            _compile_square_preset(concentrations, surface_confined),
            "Square",
            surface_confined=surface_confined,
        )
    if normalized in {"ec'", "ecprime", "ecat"}:
        compiled = _compile_ecat_preset(params, force_surface=force_surface)
        note = (
            "EC'/Ecat preset inferred catalysis form from '*' adsorbed-species notation; "
            "use a custom eCAT mechanism string for unusual catalytic cycles."
        )
        warnings.warn(note, UserWarning, stacklevel=2)
        return _mechanism_spec(
            compiled,
            "Ecat",
            note=note,
            surface_confined=force_surface or bool(concentrations["surface"]) and not bool(concentrations["bulk"]),
        )

    raise ValueError(f"Unknown simulation mechanism preset: {mechanism!r}")


def get_backend(name="electrokitty"):
    """Return the named simulation backend adapter."""
    key = str(name or "electrokitty").strip().lower().replace("-", "").replace("_", "")
    if key == "electrokitty":
        return _ElectroKittyBackend()
    raise ValueError(
        f"Unknown simulation backend {name!r}. Supported backend is 'electrokitty'."
    )


def simulate_cv(input, mechanism, params, options=None, backend="electrokitty"):
    """Run a backend CV simulation."""
    options = _normalize_options(options)
    plot = bool(options.get("plot", True))
    input_params = _prepare_simulation_params(
        input,
        params,
        options,
        mechanism=mechanism,
        expand_parameter_model=False,
    )
    params = _prepare_simulation_params(input, input_params, options, mechanism=mechanism)

    backend_adapter = get_backend(backend)
    mechanism_spec = compile_mechanism(mechanism, params)
    _maybe_print_simulation_params(params, options)
    _maybe_print_simulation_param_checks(params, input, mechanism_spec, options)
    backend_input, quiet_info = _backend_input_with_quiet_time(input, options)
    E_generated, backend_current, t, backend_result = backend_adapter.simulate(
        backend_input,
        mechanism_spec,
        params,
        options,
    )

    E_generated = np.asarray(E_generated, dtype=float)
    backend_current = np.asarray(backend_current, dtype=float)
    t = np.asarray(t, dtype=float)
    E_generated, backend_current, t = _trim_backend_quiet_time_output(
        E_generated,
        backend_current,
        t,
        quiet_info,
    )
    current_sign = _resolve_current_sign(backend_current, input.i, options)
    display_current = current_sign * backend_current

    data = pd.DataFrame(
        {
            "Potential": E_generated,
            "Current": display_current,
            "Time": t,
            "Backend Current": backend_current,
        }
    )

    data.attrs["params"] = params
    data.attrs["mechanism"] = mechanism_spec.mechanism
    data.attrs["mechanism preset"] = mechanism_spec.preset
    data.attrs["backend"] = backend_adapter.name
    if mechanism_spec.note:
        data.attrs["mechanism note"] = mechanism_spec.note
    fallback_notes = list(params.get("_fallbacks", []) or [])
    if fallback_notes:
        data.attrs["parameter fallbacks"] = fallback_notes
    parameter_model_report = deepcopy(params.get("_parameter_model", {}))
    incubation_report = parameter_model_report.get("incubation", {}) or {}
    if parameter_model_report:
        data.attrs["parameter model"] = parameter_model_report

    result = SimulatedCV(
        data=data,
        params=params,
        input_params=input_params,
        mechanism=mechanism_spec,
        input=input,
        backend_result=backend_result,
        current_sign=current_sign,
        summary={
            "backend": backend_adapter.name,
            "mechanism": mechanism_spec.mechanism,
            "preset": mechanism_spec.preset,
            "current_sign": current_sign,
            "current sign": current_sign,
            "quiet_time": quiet_info["quiet_time"],
            "quiet_time_applied": quiet_info["applied"],
            "incubation_time": float(incubation_report.get("time", _input_incubation_time(input))),
            "incubation_applied": bool(incubation_report.get("applied", False)),
            "parameter_fallbacks": fallback_notes,
            "parameter fallbacks": fallback_notes,
            "parameter_model": parameter_model_report,
            "parameter model": parameter_model_report,
        },
    )
    _maybe_print_simulation_concentration_states(result, options)
    if plot:
        result.plot(options.get("plot options", options))
    return result


def _prepare_simulation_params(
    input_obj,
    params,
    options=None,
    mechanism=None,
    *,
    expand_parameter_model=True,
):
    options = _normalize_options(options)
    prepared = deepcopy(params)
    prepared.pop("_compiled", None)
    prepared.pop("_parameter_model", None)
    prepared = _normalize_simulation_species_params(prepared)
    prepared = _normalize_simulation_activity_params(prepared)
    prepared["cell"] = _cell_with_source_defaults(input_obj, prepared.get("cell", {}), options)
    prepared["spatial"] = _spatial_with_aliases(
        prepared.get("spatial", {}),
        source=getattr(input_obj, "source", None),
    )
    mechanism_spec = _compile_mechanism_for_preparation(mechanism, prepared) if mechanism is not None else None
    prepared = _normalize_mechanism_parameter_sections(prepared, mechanism_spec)
    prepared["kinetics"], fallback_notes = _kinetics_with_fallbacks(prepared.get("kinetics", []))
    if fallback_notes:
        prepared["_fallbacks"] = fallback_notes
    if expand_parameter_model:
        prepared = _expand_parameter_model(prepared, input_obj, options, mechanism_spec=mechanism_spec)
    prepared["diffusion"] = _diffusion_with_species_defaults(
        prepared.get("diffusion", {}),
        prepared.get("concentrations", {}),
    )
    return prepared


def _compile_mechanism_for_preparation(mechanism, params):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return compile_mechanism(mechanism, params)


def _expand_parameter_model(params, input_obj=None, options=None, mechanism_spec=None):
    params = deepcopy(params or {})
    if "pools" in params:
        raise ValueError(
            "Simulation pools are no longer supported. Enter every initial concentration; "
            "eCAT derives conservation constraints from reaction stoichiometry."
        )
    if "equilibria" in params:
        raise ValueError(
            "Top-level equilibria are no longer supported. Put K directly in the matching "
            "reactions entry."
        )
    concentrations = normalize_concentrations(params.get("concentrations", {}) or {})
    reaction_equilibria = _reaction_local_equilibria(params, mechanism_spec)
    params["concentrations"] = concentrations
    activity = _activity_config(params)
    parsed_equilibria = {}
    initial_state = _parameter_model_concentration_state(concentrations)
    report = {
        "equilibria": [],
        "warnings": [],
        "states": {
            "initial": deepcopy(initial_state),
            "equilibrated": deepcopy(initial_state),
        },
    }
    for name, entry in reaction_equilibria.items():
        parsed = _parse_equilibrium_reaction(entry["reaction"])
        phase = _equilibrium_reaction_phase(parsed, concentrations, entry)
        parsed_equilibria[str(name)] = {
            **entry,
            "phase": phase,
            "_name": str(name),
            "_parsed": parsed,
        }

    equilibrated_entries = [
        entry for entry in parsed_equilibria.values() if _reaction_uses_pre_equilibrium(entry)
    ]
    if equilibrated_entries:
        concentrations, equilibrium_report = _solve_stoichiometric_pre_equilibrium(
            concentrations,
            equilibrated_entries,
            activity,
            options,
        )
        params["concentrations"] = concentrations
        report["equilibrium"] = equilibrium_report
        report["states"]["equilibrated"] = _parameter_model_concentration_state(concentrations)

    reactions = list(params.get("reactions", []) or [])
    compiled_reactions = _base_compiled_reactions(reactions)
    for name, entry in parsed_equilibria.items():
        derived = _expand_equilibrium_rate(entry, compiled_reactions, concentrations, activity)
        if derived:
            report["equilibria"].append(derived)
    incubation_time = _input_incubation_time(input_obj)
    concentrations, incubation_report = _apply_chemical_incubation(
        concentrations,
        mechanism_spec,
        compiled_reactions,
        incubation_time,
    )
    params["concentrations"] = concentrations
    report["incubation"] = incubation_report
    report["states"]["incubated"] = _parameter_model_concentration_state(concentrations)
    params["_compiled"] = _compiled_parameter_sections(params, {"reactions": compiled_reactions})

    params["_parameter_model"] = report
    return params


def _input_incubation_time(input_obj):
    metadata = getattr(input_obj, "metadata", {}) or {}
    value = float(metadata.get("incubation_time", 0.0) or 0.0)
    if value < 0:
        raise ValueError("incubation_time cannot be negative.")
    return value


def _apply_chemical_incubation(concentrations, mechanism_spec, compiled_reactions, incubation_time):
    incubation_time = float(incubation_time)
    if incubation_time <= 0:
        return deepcopy(concentrations), {
            "time": incubation_time,
            "applied": False,
            "reaction_count": 0,
            "skipped_reaction_count": 0,
            "skipped_reactions": [],
            "phases": [],
        }
    steps = _mechanism_steps_of_kind(mechanism_spec, "C") if mechanism_spec is not None else []
    if not steps:
        return deepcopy(concentrations), {
            "time": incubation_time,
            "applied": False,
            "reaction_count": 0,
            "skipped_reaction_count": 0,
            "skipped_reactions": [],
            "phases": [],
        }
    if len(compiled_reactions) < len(steps):
        raise ValueError("Chemical incubation requires rate parameters for every chemical mechanism step.")

    parsed_steps = []
    skipped_reactions = []
    for step in steps:
        parsed = _parse_equilibrium_reaction(step["equation"])
        if not _chemical_incubation_is_bulk(parsed, concentrations):
            skipped_reactions.append(step["equation"])
            continue
        rates = compiled_reactions[int(step["index"])]
        if not isinstance(rates, dict) or not any(key in rates for key in ("k", "kf", "kb")):
            raise ValueError(
                f"Chemical incubation requires k or kf/kb for reaction {step['equation']!r}."
            )
        parsed_steps.append({"parsed": parsed, "phase": "bulk", "rates": rates})

    out = deepcopy(concentrations)
    phase_reports = []
    for phase in ("bulk",):
        phase_steps = [entry for entry in parsed_steps if entry["phase"] == phase]
        if not phase_steps:
            continue
        out[phase], phase_report = _integrate_chemical_incubation_phase(
            phase,
            out.get(phase, {}),
            phase_steps,
            incubation_time,
        )
        phase_reports.append(phase_report)
    return out, {
        "time": incubation_time,
        "applied": bool(phase_reports),
        "reaction_count": len(parsed_steps),
        "skipped_reaction_count": len(skipped_reactions),
        "skipped_reactions": skipped_reactions,
        "phases": phase_reports,
    }


def _chemical_incubation_is_bulk(parsed, concentrations):
    phases = set()
    for species, _coefficient in list(parsed["reactants"]) + list(parsed["products"]):
        present = {
            phase
            for phase in ("bulk", "surface")
            if species in ((concentrations or {}).get(phase, {}) or {})
        }
        if not present:
            raise ValueError(
                f"Chemical reaction {parsed['reaction']!r} references species {species!r}, "
                "but no entered concentration is defined. Enter zero when the species is "
                "initially absent."
            )
        phases.update(present)
    return phases == {"bulk"}


def _integrate_chemical_incubation_phase(phase, phase_concentrations, entries, incubation_time):
    species = []
    for entry in entries:
        parsed = entry["parsed"]
        for name, _coefficient in list(parsed["reactants"]) + list(parsed["products"]):
            if name not in species:
                species.append(name)
    index = {name: position for position, name in enumerate(species)}
    stoichiometry = np.zeros((len(species), len(entries)), dtype=float)
    initial = np.asarray([float(phase_concentrations[name]) for name in species], dtype=float)
    if np.any(~np.isfinite(initial)) or np.any(initial < 0):
        raise ValueError("Chemical incubation concentrations must be finite and nonnegative.")

    for reaction_index, entry in enumerate(entries):
        parsed = entry["parsed"]
        for name, coefficient in parsed["reactants"]:
            stoichiometry[index[name], reaction_index] -= float(coefficient)
        for name, coefficient in parsed["products"]:
            stoichiometry[index[name], reaction_index] += float(coefficient)

    def rhs(_time, values):
        nonnegative = np.maximum(values, 0.0)
        rates = []
        for entry in entries:
            parsed = entry["parsed"]
            constants = entry["rates"]
            forward_constant = float(constants.get("k", constants.get("kf", 0.0)))
            forward = forward_constant * np.prod(
                [nonnegative[index[name]] ** coefficient for name, coefficient in parsed["reactants"]]
            )
            reverse = 0.0
            if "kb" in constants:
                reverse = float(constants["kb"]) * np.prod(
                    [nonnegative[index[name]] ** coefficient for name, coefficient in parsed["products"]]
                )
            rates.append(float(forward - reverse))
        return stoichiometry @ np.asarray(rates, dtype=float)

    from scipy.integrate import solve_ivp

    result = solve_ivp(
        rhs,
        (0.0, float(incubation_time)),
        initial,
        method="BDF",
        rtol=1e-8,
        atol=1e-12,
    )
    if not result.success:
        raise ValueError(f"Chemical incubation failed: {result.message}")
    final = np.asarray(result.y[:, -1], dtype=float)
    negative_tolerance = 1e-9 * max(1.0, float(np.max(initial)) if initial.size else 1.0)
    if np.any(final < -negative_tolerance):
        raise ValueError("Chemical incubation produced materially negative concentrations.")
    final = np.maximum(final, 0.0)
    out = dict(phase_concentrations)
    out.update({name: float(value) for name, value in zip(species, final)})
    return out, {
        "phase": phase,
        "species": list(species),
        "reaction_count": len(entries),
        "evaluations": int(result.nfev),
        "success": bool(result.success),
    }


def _parameter_model_concentration_state(concentrations):
    return {
        phase: {str(name): float(value) for name, value in ((concentrations or {}).get(phase, {}) or {}).items()}
        for phase in ("bulk", "surface")
    }


def _reaction_uses_pre_equilibrium(entry):
    value = entry.get("equilibrate", True)
    return not _falsey_option(value)


def _equilibrium_reaction_phase(parsed, concentrations, entry=None):
    entry = {} if entry is None else entry
    explicit = entry.get("phase")
    phases = set()
    for species, _coefficient in list(parsed["reactants"]) + list(parsed["products"]):
        present = [
            phase
            for phase in ("bulk", "surface")
            if species in ((concentrations or {}).get(phase, {}) or {})
        ]
        if not present:
            raise ValueError(
                f"Equilibrium reaction {parsed['reaction']!r} references species {species!r}, "
                "but no entered concentration is defined. Enter zero when the species is "
                "initially absent."
            )
        if len(present) > 1:
            raise ValueError(
                f"Equilibrium species {species!r} exists in both bulk and surface concentrations; "
                "use distinct species names."
            )
        phases.add(present[0])
    if explicit not in (None, ""):
        explicit = _normalize_concentration_phase(explicit)
        if phases != {explicit}:
            raise ValueError(
                f"Equilibrium reaction {parsed['reaction']!r} does not match its declared {explicit} phase."
            )
    if len(phases) != 1:
        raise ValueError(
            f"Equilibrium reaction {parsed['reaction']!r} mixes bulk and surface species. "
            "Cross-phase pre-equilibrium is not supported."
        )
    return next(iter(phases))


def _solve_stoichiometric_pre_equilibrium(concentrations, entries, activity, options=None):
    solved = deepcopy(concentrations)
    phase_reports = []
    for phase in ("bulk", "surface"):
        phase_entries = [entry for entry in entries if entry.get("phase") == phase]
        if not phase_entries:
            continue
        solved[phase], phase_report = _solve_stoichiometric_equilibrium_phase(
            phase,
            solved.get(phase, {}),
            phase_entries,
            activity,
            options,
        )
        phase_reports.append(phase_report)
    max_residual = max((report["max_equilibrium_residual"] for report in phase_reports), default=0.0)
    return solved, {
        "reaction_count": int(sum(report["reaction_count"] for report in phase_reports)),
        "reaction_rank": int(sum(report["reaction_rank"] for report in phase_reports)),
        "dependent_reactions": int(sum(report["dependent_reactions"] for report in phase_reports)),
        "conservation_rank": int(sum(report["conservation_rank"] for report in phase_reports)),
        "max_equilibrium_residual": float(max_residual),
        "phases": phase_reports,
    }


def _solve_stoichiometric_equilibrium_phase(phase, phase_concentrations, entries, activity, options=None):
    species = []
    for entry in entries:
        for name, _coefficient in list(entry["_parsed"]["reactants"]) + list(entry["_parsed"]["products"]):
            if name not in species:
                species.append(name)
    index = {name: position for position, name in enumerate(species)}
    stoichiometry = np.zeros((len(species), len(entries)), dtype=float)
    for reaction_index, entry in enumerate(entries):
        for name, coefficient in entry["_parsed"]["reactants"]:
            stoichiometry[index[name], reaction_index] -= float(coefficient)
        for name, coefficient in entry["_parsed"]["products"]:
            stoichiometry[index[name], reaction_index] += float(coefficient)

    initial = np.asarray([float(phase_concentrations[name]) for name in species], dtype=float)
    if np.any(~np.isfinite(initial)) or np.any(initial < 0):
        raise ValueError("Pre-equilibrium concentrations must be finite and nonnegative.")

    from scipy.linalg import null_space
    from scipy.optimize import least_squares

    reaction_rank = int(np.linalg.matrix_rank(stoichiometry))
    conservation = null_space(stoichiometry.T)
    conserved_totals = conservation.T @ initial
    conservation_scales = np.maximum(
        np.sum(np.abs(conservation.T) * np.maximum(initial, 1e-30)[None, :], axis=1),
        1e-30,
    )
    standard = _activity_standard(activity, phase)
    positive_initial = np.maximum(initial, max(standard, 1.0) * 1e-12)

    def residual(log_values):
        values = np.exp(log_values)
        value_map = {name: float(value) for name, value in zip(species, values)}
        equilibrium_rows = [
            _log_activity_quotient(entry["_parsed"], value_map, phase, {phase: value_map}, activity)
            - np.log(_dimensionless_equilibrium_constant(entry, entry["_parsed"], activity))
            for entry in entries
        ]
        conservation_rows = (
            (conservation.T @ values - conserved_totals) / conservation_scales
            if conservation.shape[1]
            else np.asarray([], dtype=float)
        )
        return np.r_[equilibrium_rows, conservation_rows]

    result = least_squares(residual, np.log(positive_initial), max_nfev=8000)
    final_rows = residual(result.x)
    equilibrium_rows = final_rows[: len(entries)]
    conservation_rows = final_rows[len(entries) :]
    tolerance = float(_normalize_options(options).get("parameter model tolerance", 1e-7))
    max_equilibrium = float(np.max(np.abs(equilibrium_rows))) if equilibrium_rows.size else 0.0
    max_conservation = float(np.max(np.abs(conservation_rows))) if conservation_rows.size else 0.0
    if not result.success or max(max_equilibrium, max_conservation) > tolerance:
        raise ValueError(
            "Stoichiometric pre-equilibrium has inconsistent equilibrium constraints or "
            f"failed to converge; residual {max(max_equilibrium, max_conservation):.3g} "
            f"exceeds tolerance {tolerance:.3g}."
        )
    values = np.exp(result.x)
    out = dict(phase_concentrations)
    out.update({name: float(value) for name, value in zip(species, values)})
    return out, {
        "phase": phase,
        "species": list(species),
        "reaction_count": len(entries),
        "reaction_rank": reaction_rank,
        "dependent_reactions": len(entries) - reaction_rank,
        "conservation_rank": int(conservation.shape[1]),
        "max_equilibrium_residual": max_equilibrium,
        "max_conservation_residual": max_conservation,
        "success": bool(result.success),
        "evaluations": int(result.nfev),
    }


def _reaction_local_equilibria(params, mechanism_spec=None):
    reactions = params.get("reactions", []) or []
    if not isinstance(reactions, list):
        return {}
    steps = ((params.get("_mechanism_steps", {}) or {}).get("reactions") or [])
    if not steps and mechanism_spec is not None:
        steps = _mechanism_steps_of_kind(mechanism_spec, "C")
    out = {}
    for index, entry in enumerate(reactions):
        if not isinstance(entry, dict) or "K" not in entry:
            continue
        normalized = _normalize_equilibrium_param_entry(entry)
        reaction = normalized.get("reaction")
        if reaction in (None, ""):
            if index >= len(steps):
                raise ValueError(f"reactions.{index}.K requires a mechanism chemical step or explicit reaction.")
            reaction = steps[index]["equation"]
        normalized["reaction"] = reaction
        normalized.setdefault("target", f"reactions.{index}")
        out[f"reactions.{index}"] = normalized
    return out


def _normalize_equilibrium_param_entry(entry):
    out = dict(entry)
    if "K" in out:
        out.update(_coerce_equilibrium_constant_fields(out["K"], out))
    return out


def _compiled_parameter_sections(params, overrides=None):
    compiled = dict((params or {}).get("_compiled", {}) or {})
    compiled["kinetics"] = deepcopy((params or {}).get("kinetics", []) or [])
    compiled["reactions"] = _base_compiled_reactions((params or {}).get("reactions", []) or [])
    for key, value in (overrides or {}).items():
        compiled[key] = deepcopy(value)
    return compiled


def _base_compiled_reactions(reactions):
    out = []
    for entry in reactions or []:
        if isinstance(entry, dict):
            compiled = {
                key: value
                for key, value in entry.items()
                if key in {"k", "kf", "kb"} or (isinstance(key, str) and key.startswith("_"))
            }
            out.append(compiled)
        else:
            out.append(deepcopy(entry))
    return out


def _normalize_concentration_phase(value):
    phase = _canonical_token(value or "bulk")
    if phase in {"bulk", "solution", "soluble"}:
        return "bulk"
    if phase in {"surface", "adsorbed", "ads"}:
        return "surface"
    raise ValueError("concentration phase must be bulk/solution/soluble or surface/adsorbed/ads.")



def _parse_equilibrium_reaction(reaction):
    text = str(reaction or "").strip()
    try:
        left, separator, right = _split_reaction_equation(text)
    except ValueError as exc:
        raise ValueError("equilibrium reaction must contain exactly one '=' or reaction arrow.") from exc
    if separator == "<":
        left, right = right, left
    reactants = _parse_equilibrium_side(left)
    products = _parse_equilibrium_side(right)
    if not reactants or not products:
        raise ValueError("equilibrium reaction must have reactants and products.")
    return {"reactants": reactants, "products": products, "reaction": text}


def _parse_equilibrium_side(side):
    terms = []
    for raw in str(side).split("+"):
        species, coefficient = _parse_stoichiometric_term(raw)
        terms.append((species.rstrip("*"), float(coefficient)))
    return terms


def _log_activity_quotient(parsed, variable_values, phase, concentrations, activity):
    log_products = sum(
        coeff * _log_species_activity(species, coeff, variable_values, phase, concentrations, activity)
        for species, coeff in parsed["products"]
    )
    log_reactants = sum(
        coeff * _log_species_activity(species, coeff, variable_values, phase, concentrations, activity)
        for species, coeff in parsed["reactants"]
    )
    return float(log_products - log_reactants)


def _log_species_activity(species, coeff, variable_values, phase, concentrations, activity):
    del coeff
    concentration = _species_concentration(species, variable_values, phase, concentrations)
    if concentration <= 0:
        raise ValueError(f"Species {species!r} has nonpositive concentration in an equilibrium.")
    gamma = _activity_gamma(activity, phase, species)
    if gamma <= 0:
        raise ValueError(f"Activity coefficient for {species!r} must be positive.")
    return float(np.log(gamma) + np.log(concentration) - np.log(_activity_standard(activity, phase)))


def _species_concentration(species, variable_values, phase, concentrations):
    clean = str(species).rstrip("*")
    if clean in variable_values:
        return float(variable_values[clean])
    phase_values = concentrations.get(phase, {}) or {}
    if clean in phase_values:
        return float(phase_values[clean])
    bulk_values = concentrations.get("bulk", {}) or {}
    if clean in bulk_values:
        return float(bulk_values[clean])
    raise ValueError(f"Equilibrium references species {clean!r}, but no concentration is defined.")


def _expand_equilibrium_rate(entry, reactions, concentrations, activity):
    target = entry.get("target")
    if target in (None, "", False):
        return None
    index = _equilibrium_target_reaction_index(target)
    while len(reactions) <= index:
        reactions.append({})
    parsed = entry["_parsed"]
    k_concentration = _equilibrium_concentration_quotient(entry, parsed, activity)
    external_reference_factor = 1.0
    if "k_exchange" in entry and entry.get("k_exchange") is not None:
        k_exchange = float(entry["k_exchange"])
        if k_exchange <= 0 or not np.isfinite(k_exchange):
            raise ValueError("reaction k_exchange must be positive and finite.")
        phase = entry.get("phase", "bulk")
        reference_key = "reference_coverage" if phase == "surface" else "reference_concentration"
        reference = float(
            entry.get(
                reference_key,
                entry.get("reference_amount", _activity_standard(activity, phase)),
            )
        )
        if reference <= 0 or not np.isfinite(reference):
            raise ValueError(f"reaction {reference_key} must be positive and finite.")
        reactant_order = _total_stoich(parsed["reactants"])
        product_order = _total_stoich(parsed["products"])
        forward_factor = reference ** max(reactant_order - 1, 0)
        reverse_factor = reference ** max(product_order - 1, 0)
        external_reference_factor = forward_factor
        kb = k_exchange / (k_concentration * forward_factor + reverse_factor)
        kf = k_concentration * kb
        derived_from = "K, k_exchange"
    elif "koff" in entry and entry.get("koff") is not None:
        kb = float(entry["koff"])
        kf = k_concentration * kb
        derived_from = "K, koff"
    else:
        return None
    reactions[index] = {**(reactions[index] if isinstance(reactions[index], dict) else {}), "kf": float(kf), "kb": float(kb)}
    return {
        "equilibrium": entry["_name"],
        "target": f"reactions.{index}",
        "kf": float(kf),
        "kb": float(kb),
        "K_concentration": float(k_concentration),
        "derived_from": derived_from,
        "external_reference_factor": float(external_reference_factor),
    }


def _equilibrium_target_reaction_index(target):
    if isinstance(target, (tuple, list)) and len(target) >= 2 and target[0] == "reactions":
        return int(target[1])
    text = str(target).strip()
    if text.startswith("reactions."):
        return int(text.split(".")[1])
    raise ValueError("equilibrium target must be a reactions.N path.")


def _total_stoich(terms):
    return int(sum(coeff for _species, coeff in terms))


def _dimensionless_equilibrium_constant(entry, parsed, activity):
    value = float(entry.get("K"))
    if value <= 0 or not np.isfinite(value):
        raise ValueError("equilibrium K must be positive.")
    unit = _canonical_k_unit(entry.get("K_unit", entry.get("K unit", entry.get("K units", "dimensionless"))))
    if unit in {"", "dimensionless", "activity"}:
        return value
    if entry.get("phase", "bulk") == "surface":
        raise ValueError(
            "Unit-bearing K is only supported for bulk reactions; use dimensionless activity K "
            "with activity.standard_coverage for surface equilibria."
        )
    reactant_order = _total_stoich(parsed["reactants"])
    product_order = _total_stoich(parsed["products"])
    power = reactant_order - product_order
    if power <= 0:
        raise ValueError("concentration-quotient K_unit is only supported when reactant order exceeds product order.")
    if unit in {"m^-1"}:
        return value
    if unit in {"mm^-1"}:
        return value * (1000.0**power)
    if unit in {"native", "m^3/mol", "m3/mol", "m3mol"}:
        return value * (_activity_standard(activity, entry.get("phase", "bulk")) ** power)
    raise ValueError(f"Unknown K_unit {entry.get('K_unit')!r}.")


def _canonical_k_unit(value):
    text = str(value or "dimensionless").strip().lower()
    text = text.replace(" ", "").replace("−", "-")
    if text in {"", "dimensionless", "activity"}:
        return "dimensionless"
    if text in {"m^-1", "m-1", "1/m"}:
        return "m^-1"
    if text in {"mm^-1", "mm-1", "1/mm"}:
        return "mm^-1"
    if text in {"native", "m^3/mol", "m3/mol", "m3mol"}:
        return text
    return text


def _equilibrium_concentration_quotient(entry, parsed, activity):
    k_dimless = _dimensionless_equilibrium_constant(entry, parsed, activity)
    reactant_gamma = np.prod([
        _activity_gamma(activity, entry.get("phase", "bulk"), species) ** coeff
        for species, coeff in parsed["reactants"]
    ])
    product_gamma = np.prod([
        _activity_gamma(activity, entry.get("phase", "bulk"), species) ** coeff
        for species, coeff in parsed["products"]
    ])
    reactant_order = _total_stoich(parsed["reactants"])
    product_order = _total_stoich(parsed["products"])
    return float(
        k_dimless
        * reactant_gamma
        / product_gamma
        * (_activity_standard(activity, entry.get("phase", "bulk")) ** (product_order - reactant_order))
    )


def _kinetics_with_fallbacks(kinetics):
    if kinetics is None:
        return [], []
    if not isinstance(kinetics, list):
        return kinetics, []

    prepared = []
    notes = []
    for index, entry in enumerate(kinetics):
        if not isinstance(entry, dict):
            prepared.append(entry)
            continue

        kinetic = dict(entry)
        has_chemical_rate = any(key in kinetic for key in ("kf", "kb", "k"))
        if has_chemical_rate and not any(key in kinetic for key in ("E0", "k0", "alpha")):
            prepared.append(kinetic)
            continue

        if "k0" in kinetic and isinstance(kinetic["k0"], str):
            original = kinetic["k0"]
            kinetic["k0"] = _resolve_k0_preset(original)
            notes.append(f"kinetics.{index}.k0 used preset {original!r} -> {kinetic['k0']:.6g} m/s")

        for key, value in KINETIC_FALLBACKS.items():
            if key not in kinetic or kinetic[key] is None:
                kinetic[key] = value
                notes.append(f"kinetics.{index}.{key} defaulted to {value:.6g}")

        prepared.append(kinetic)
    return prepared, notes


def _resolve_k0_preset(value):
    key = _normalize_alias_key(value)
    if key in K0_PRESETS:
        return float(K0_PRESETS[key])
    raise ValueError(
        f"Unknown k0 preset {value!r}. Supported presets are: "
        f"{', '.join(sorted(K0_PRESETS))}."
    )


def _backend_input_with_quiet_time(input_obj, options):
    quiet_time = _input_quiet_time(input_obj, options)
    metadata = dict(getattr(input_obj, "metadata", {}) or {})
    use_quiet_time = _truthy_option(options.get("use quiet time", options.get("use_quiet_time", True)))
    already_applied = bool(metadata.get("quiet_time_applied", False))
    if quiet_time <= 0 or not use_quiet_time or already_applied or len(input_obj.E) == 0:
        return input_obj, {"applied": False, "quiet_time": quiet_time, "trim_points": 0}

    t = np.asarray(input_obj.t, dtype=float)
    E = np.asarray(input_obj.E, dtype=float)
    dt = _representative_dt(t)
    hold_points = max(2, int(np.ceil(quiet_time / dt)) + 1)
    t_hold = np.linspace(0.0, quiet_time, hold_points)
    E_hold = np.full(hold_points, E[0], dtype=float)

    extended_E = np.concatenate([E_hold, E[1:]])
    extended_t = np.concatenate([t_hold, quiet_time + t[1:]])
    extended_i = None
    if input_obj.i is not None:
        measured = np.asarray(input_obj.i, dtype=float)
        i_hold = np.full(hold_points, measured[0], dtype=float)
        extended_i = np.concatenate([i_hold, measured[1:]])

    backend_metadata = dict(metadata)
    backend_metadata["quiet_time"] = quiet_time
    backend_metadata["quiet_time_applied"] = True
    backend_input = SimulatedCVInput(
        E=extended_E,
        t=extended_t,
        i=extended_i,
        metadata=backend_metadata,
        source=input_obj.source,
    )
    return backend_input, {
        "applied": True,
        "quiet_time": quiet_time,
        "trim_points": hold_points - 1,
    }


def _input_quiet_time(input_obj, options):
    quiet_time = options.get("quiet time", options.get("quiet_time", None))
    if quiet_time is None:
        metadata = getattr(input_obj, "metadata", {}) or {}
        quiet_time = metadata.get("quiet_time", metadata.get("quiet time", None))
    if quiet_time is None:
        quiet_time = _source_quiet_time(getattr(input_obj, "source", None))
    quiet_time = 0.0 if quiet_time is None else float(quiet_time)
    if quiet_time < 0:
        raise ValueError("quiet time cannot be negative.")
    return quiet_time


def _trim_backend_quiet_time_output(E, current, t, quiet_info):
    trim_points = int(quiet_info.get("trim_points", 0))
    if trim_points <= 0:
        return E, current, t
    E = np.asarray(E, dtype=float)[trim_points:]
    current = np.asarray(current, dtype=float)[trim_points:]
    t = np.asarray(t, dtype=float)[trim_points:] - float(quiet_info.get("quiet_time", 0.0))
    if len(t):
        t = t - t[0]
    return E, current, t


def _spatial_with_aliases(spatial, source=None):
    if spatial is None:
        return {}
    if isinstance(spatial, str):
        base = deepcopy(_resolve_spatial_preset(spatial))
        _apply_source_solvent_viscosity(base, source)
        return base
    if not isinstance(spatial, dict):
        raise ValueError("spatial params must be a mapping, a preset name, or None.")

    preset_name = spatial.get("preset", spatial.get("mode", spatial.get("profile")))
    base = deepcopy(_resolve_spatial_preset(preset_name)) if preset_name else {}
    explicit = {
        key: value
        for key, value in spatial.items()
        if key not in {"preset", "mode", "profile", "solvent"}
    }
    solvent = spatial.get("solvent")
    if solvent is not None and "viscosity" not in explicit:
        base["viscosity"] = _resolve_solvent_viscosity(solvent)
    elif "viscosity" not in explicit:
        _apply_source_solvent_viscosity(base, source)
    base.update(explicit)
    if isinstance(base.get("viscosity"), str):
        base["viscosity"] = _resolve_solvent_viscosity(base["viscosity"])
    return base


def _apply_source_solvent_viscosity(spatial, source):
    solvent = getattr(source, "solvent", None)
    if solvent in (None, ""):
        return
    viscosity = _try_resolve_solvent_viscosity(solvent)
    if viscosity is not None:
        spatial["viscosity"] = viscosity


def _resolve_spatial_preset(name):
    key = _normalize_alias_key(name)
    key = _SPATIAL_PRESET_ALIASES.get(key, key)
    if key in SPATIAL_PRESETS:
        return SPATIAL_PRESETS[key]
    raise ValueError(
        f"Unknown spatial preset {name!r}. Supported presets are: "
        f"{', '.join(sorted(SPATIAL_PRESETS))}."
    )


def _resolve_solvent_viscosity(name):
    value = _try_resolve_solvent_viscosity(name)
    if value is not None:
        return value
    raise ValueError(
        f"Unknown solvent viscosity alias {name!r}. Supported aliases include: "
        f"{', '.join(sorted(SOLVENT_VISCOSITIES))}."
    )


def _try_resolve_solvent_viscosity(name):
    key = _normalize_alias_key(name)
    canonical = _SOLVENT_ALIASES.get(key)
    if canonical is None:
        return None
    return float(SOLVENT_VISCOSITIES[canonical])


def _normalize_alias_key(value):
    return str(value).strip().lower().replace("_", " ").replace("-", " ")


def _cell_with_source_defaults(input_obj, cell, options):
    cell = _normalize_cell_params(cell)
    cell = _cell_with_auto_cdl(input_obj, cell, options)
    source = getattr(input_obj, "source", None)
    if source is None:
        return cell

    defaults = {
        "T": _first_finite_attr(source, ["temperature", "T"]),
        "A": _source_electrode_area_m2(source, options),
        "Ru": _first_finite_attr(
            source,
            ["ir_uncomp_resistance", "uncompensated_resistance", "uncomp_R", "Ru", "comp_R", "IR_comp"],
        ),
    }
    allow_default_override = bool(
        options.get("source defaults override", options.get("source_defaults_override", False))
    )
    default_values = {"T": 298.15, "A": 1e-5, "Ru": 0.0}
    for key, value in defaults.items():
        if value is None:
            continue
        current = cell.get(key)
        if current is None or _is_auto_value(current) or (key == "A" and float(current) == 0.0):
            if key != "A" or value != 0.0:
                cell[key] = value
        elif allow_default_override and key in default_values and float(current) == default_values[key]:
            cell[key] = value
    for key, value in default_values.items():
        current = cell.get(key)
        if current is None or _is_auto_value(current) or (key == "A" and float(current) == 0.0):
            cell[key] = value
    return cell


def _normalize_cell_params(cell):
    if cell is None:
        return {}
    if _is_auto_value(cell):
        return {"Cdl": "auto"}
    if isinstance(cell, str):
        raise ValueError("cell params must be a mapping, 'auto', or None.")
    if isinstance(cell, dict):
        return dict(cell)
    try:
        return dict(cell)
    except (TypeError, ValueError) as exc:
        raise ValueError("cell params must be a mapping, 'auto', or None.") from exc


def _cell_with_auto_cdl(input_obj, cell, options=None):
    if not _is_auto_value(cell.get("Cdl")):
        return cell
    options = _normalize_options(options)
    metadata = getattr(input_obj, "metadata", {}) or {}
    cdl = metadata.get("estimated_cdl", metadata.get("estimated_Cdl"))
    if cdl is None:
        current = getattr(input_obj, "i", None)
        scan_rate = metadata.get("scan_rate")
        if current is not None and scan_rate is not None:
            cdl, diagnostics = estimate_cdl_from_cv_arrays(input_obj.E, current, scan_rate, options)
            metadata["estimated_cdl"] = cdl
            metadata["estimated_Cdl"] = cdl
            metadata["estimated_cdl_diagnostics"] = diagnostics
        else:
            raise ValueError(
                "cell['Cdl']='auto' requires measured CV data with a positive scan rate, "
                "or a precomputed Cdl estimate from cv_data(cv, {'estimate Cdl': True})."
            )
    cell["Cdl"] = float(cdl)
    return cell


def _source_electrode_area_m2(source, options):
    area = _first_finite_attr(source, ["electrode_area", "area", "A"])
    if area is None:
        return None
    units = str(
        options.get(
            "source electrode area units",
            options.get("source_electrode_area_units", "cm^2"),
        )
    ).strip().lower().replace("²", "^2")
    if units in {"cm2", "cm^2", "cm 2"}:
        return area * 1e-4
    if units in {"m2", "m^2", "m 2"}:
        return area
    raise ValueError("source electrode area units must be 'cm^2' or 'm^2'.")


def _normalize_simulation_species_params(params):
    params = {} if params is None else dict(params)
    if "pools" in params:
        raise ValueError(
            "Simulation pools are no longer supported. Enter every initial concentration; "
            "eCAT derives conservation constraints from reaction stoichiometry."
        )
    species_input = params.pop("species", None)
    concentrations = normalize_concentrations(params.get("concentrations", {}) or {})
    diffusion = dict(params.get("diffusion", {}) or {})

    if species_input:
        species_concentrations, species_diffusion = _species_input_sugar(species_input)
        for phase in ("bulk", "surface"):
            for name, value in species_concentrations[phase].items():
                concentrations[phase].setdefault(name, value)
        for name, value in species_diffusion.items():
            diffusion.setdefault(name, value)

    params["concentrations"] = concentrations
    params["diffusion"] = diffusion
    return params


def _normalize_simulation_activity_params(params):
    params = {} if params is None else dict(params)
    sugar = params.pop("activity_coefficients", None)
    activity = _activity_config({"activity": params.get("activity", {}), "activity_coefficients": sugar})
    has_activity = "activity" in params or sugar not in (None, {})
    if has_activity:
        params["activity"] = activity
    return params


def _activity_config(params):
    params = {} if params is None else params
    activity = params.get("activity", {}) if isinstance(params.get("activity", {}), dict) else {}
    standard = activity.get(
        "standard_concentration",
        activity.get("standard concentration", activity.get("C_standard", _DEFAULT_ACTIVITY_STANDARD_CONCENTRATION)),
    )
    standard_coverage = activity.get(
        "standard_coverage",
        activity.get("standard coverage", activity.get("Gamma_standard", _DEFAULT_ACTIVITY_STANDARD_COVERAGE)),
    )
    gamma_raw = activity.get(
        "gamma",
        activity.get("gammas", activity.get("coefficients", activity.get("activity_coefficients", {}))),
    )
    if not gamma_raw and params.get("activity_coefficients") not in (None, {}):
        gamma_raw = params.get("activity_coefficients")
    standard = float(standard)
    standard_coverage = float(standard_coverage)
    if standard <= 0 or not np.isfinite(standard):
        raise ValueError("activity standard_concentration must be positive and finite.")
    if standard_coverage <= 0 or not np.isfinite(standard_coverage):
        raise ValueError("activity standard_coverage must be positive and finite.")
    return {
        "standard_concentration": standard,
        "standard_coverage": standard_coverage,
        "gamma": _normalize_gamma_mapping(gamma_raw),
    }


def _normalize_gamma_mapping(gamma):
    out = {"bulk": {}, "surface": {}}
    if not isinstance(gamma, dict) or not gamma:
        return out
    if any(key in gamma for key in ("bulk", "surface")):
        for phase in ("bulk", "surface"):
            values = gamma.get(phase, {}) or {}
            if not isinstance(values, dict):
                raise ValueError("activity gamma phase entries must be mappings.")
            for name, value in values.items():
                clean = _strip_surface_star(name) if phase == "surface" else str(name).rstrip("*")
                out[phase][clean] = float(value)
    else:
        for name, value in gamma.items():
            out["bulk"][str(name).rstrip("*")] = float(value)
    for phase, values in out.items():
        for name, value in values.items():
            if value <= 0 or not np.isfinite(value):
                raise ValueError(f"activity coefficient gamma for {phase}.{name} must be positive.")
    return out


def _activity_gamma(activity, phase, species):
    phase = _normalize_concentration_phase(phase)
    clean = _strip_surface_star(species) if phase == "surface" else str(species).rstrip("*")
    return float(((activity or {}).get("gamma", {}).get(phase, {}) or {}).get(clean, 1.0))


def _activity_standard(activity, phase):
    phase = _normalize_concentration_phase(phase)
    key = "standard_coverage" if phase == "surface" else "standard_concentration"
    default = (
        _DEFAULT_ACTIVITY_STANDARD_COVERAGE
        if phase == "surface"
        else _DEFAULT_ACTIVITY_STANDARD_CONCENTRATION
    )
    return float((activity or {}).get(key, default))


def _species_activity(concentration, activity, phase, species):
    return float(_activity_gamma(activity, phase, species) * float(concentration) / _activity_standard(activity, phase))


def _has_nonideal_activity(params):
    activity = _activity_config(params)
    for phase in ("bulk", "surface"):
        for value in (activity.get("gamma", {}).get(phase, {}) or {}).values():
            if not np.isclose(float(value), 1.0, rtol=_ACTIVITY_DISPLAY_RTOL, atol=0.0):
                return True
    return False


def _normalize_mechanism_parameter_sections(params, mechanism_spec=None):
    params = {} if params is None else dict(params)
    if mechanism_spec is None:
        return params
    params["kinetics"] = _normalize_step_parameter_section(
        params.get("kinetics", []),
        _mechanism_steps_of_kind(mechanism_spec, "E"),
        "kinetics",
    )
    params["reactions"] = _normalize_step_parameter_section(
        params.get("reactions", []),
        _mechanism_steps_of_kind(mechanism_spec, "C"),
        "reactions",
    )
    params["_mechanism_steps"] = {
        "kinetics": _mechanism_steps_of_kind(mechanism_spec, "E"),
        "reactions": _mechanism_steps_of_kind(mechanism_spec, "C"),
    }
    return params


def _normalize_step_parameter_section(values, steps, section):
    if values is None:
        return []
    if isinstance(values, list):
        normalized = [deepcopy(entry) for entry in values]
    elif isinstance(values, tuple):
        normalized = [deepcopy(entry) for entry in values]
    elif isinstance(values, dict):
        normalized = []
        for key, entry in values.items():
            index = _resolve_step_parameter_key(key, steps, section)
            while len(normalized) <= index:
                normalized.append({})
            if normalized[index] not in ({}, None):
                raise ValueError(f"Duplicate {section} entry for step {index}.")
            normalized[index] = deepcopy(entry)
    else:
        raise ValueError(f"{section} params must be a list or mapping.")
    if section == "reactions":
        normalized = [_normalize_reaction_param_entry(entry) for entry in normalized]
    return normalized


def _resolve_step_parameter_key(key, steps, section):
    if isinstance(key, (int, np.integer)):
        index = int(key)
        if index < 0:
            raise ValueError(f"{section} step index cannot be negative.")
        return index
    text = str(key).strip()
    if text.isdigit():
        return int(text)
    lowered = text.lower()
    for prefix in (f"{section}.", f"{section[:-1]}."):
        if lowered.startswith(prefix):
            tail = text.split(".", 1)[1]
            if tail.isdigit():
                return int(tail)
    key_text = _reaction_key(text)
    matches = [int(step["index"]) for step in steps if step.get("key") == key_text]
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise ValueError(f"Could not match {section} key {key!r} to a mechanism step.")
    raise ValueError(f"{section} key {key!r} matches multiple mechanism steps; use a numeric key.")


def _normalize_reaction_param_entry(entry):
    if not isinstance(entry, dict):
        return entry
    out = dict(entry)
    if "pool" in out:
        raise ValueError(
            "Reaction pool fields are no longer supported. Use equilibrate=False for a "
            "reversible reaction that should remain dynamic-only."
        )
    if "mode" in out:
        raise ValueError(
            "Reaction mode is no longer supported. Equilibrium reactions equilibrate by "
            "default; use equilibrate=False to opt out."
        )
    if "k_exchange_ref" in out:
        raise ValueError(
            "reaction k_exchange_ref is no longer supported; use the standard-state "
            "k_exchange parameter."
        )
    if "K" in out:
        out.update(_coerce_equilibrium_constant_fields(out["K"], out))
    return out


def _coerce_equilibrium_constant_fields(value, entry=None):
    entry = {} if entry is None else entry
    if isinstance(value, str):
        parsed_value, parsed_unit = _parse_equilibrium_constant_string(value)
        unit = entry.get("K_unit", entry.get("K unit", entry.get("K units", parsed_unit)))
        out = {"K": parsed_value}
        if unit not in (None, ""):
            out["K_unit"] = str(unit)
        return out
    return {"K": float(value)}


def _parse_equilibrium_constant_string(value):
    text = str(value).strip()
    match = re.match(r"^\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s*(.*?)\s*$", text)
    if not match:
        raise ValueError(f"Could not parse equilibrium constant {value!r}.")
    number, unit = match.groups()
    return float(number), unit.strip()


def _concentrations_from_params(params):
    params = {} if params is None else params
    if "species" in params:
        return _normalize_simulation_species_params(params).get("concentrations", {})
    return params.get("concentrations", {})


def _species_input_sugar(species):
    if not isinstance(species, dict):
        raise ValueError("species input must be a mapping of species names to setup dictionaries.")
    if any(key in species for key in ("bulk", "surface")):
        raise ValueError("Use concentrations.bulk/surface for grouped species; species is now input sugar.")

    concentrations = {"surface": {}, "bulk": {}}
    diffusion = {}
    for name, entry in species.items():
        if not isinstance(entry, dict):
            raise ValueError("species input sugar entries must be dictionaries with type/C/D fields.")
        phase = _normalize_species_input_type(entry.get("type", "bulk"))
        clean_name = _strip_surface_star(name) if phase == "surface" else str(name).rstrip("*")
        amount, has_amount = _species_input_amount(entry)
        if has_amount:
            concentrations[phase][clean_name] = float(amount)
        d_value, has_diffusion = _species_input_diffusion(entry)
        if has_diffusion:
            diffusion[str(name).rstrip("*")] = float(d_value)
    return concentrations, diffusion


def _normalize_species_input_type(value):
    text = _canonical_token("bulk" if value is None else value)
    if text in {"bulk", "solution", "soluble"}:
        return "bulk"
    if text in {"surface", "adsorbed", "ads"}:
        return "surface"
    raise ValueError("species type must be bulk/solution/soluble or surface/adsorbed/ads.")


def _species_input_amount(entry):
    if "C" in entry:
        return entry["C"], True
    if "concentration" in entry:
        return entry["concentration"], True
    return None, False


def _species_input_diffusion(entry):
    if "D" in entry:
        return entry["D"], True
    if "diffusion" in entry:
        return entry["diffusion"], True
    return None, False


def _first_finite_attr(obj, names):
    for name in names:
        value = getattr(obj, name, None)
        if value is None:
            continue
        try:
            value = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(value):
            return value
    return None


def _diffusion_with_species_defaults(diffusion, species):
    diffusion = {} if diffusion is None else dict(diffusion)
    normalized_concentrations = normalize_concentrations(species)
    species_names = list(normalized_concentrations["bulk"])
    if not species_names:
        return diffusion
    if "D" in diffusion:
        default = float(diffusion["D"])
        overrides = {str(name).rstrip("*"): float(value) for name, value in diffusion.items() if name != "D"}
        return {name: overrides.get(name, default) for name in species_names}
    if len(diffusion) == 1 and len(species_names) > 1:
        value = float(next(iter(diffusion.values())))
        return {name: value for name in species_names}
    return diffusion


def _maybe_print_simulation_params(params, options):
    mode = options.get("print params", options.get("print_params", False))
    if not _truthy_option(mode):
        return
    if isinstance(mode, str) and mode.strip().lower().replace("-", "_") in {"raw", "dict", "pprint"}:
        print("Simulation Params:")
        pprint.pprint(params, sort_dicts=False)
    elif _compact_param_mode(mode):
        _display_dataframe_sections(
            "Simulation Params:",
            _simulation_param_dataframes(params, compact=True),
            options=options,
        )
    else:
        _display_dataframe_sections(
            "Simulation Params:",
            _simulation_param_dataframes(params),
            options=options,
        )


def _maybe_print_simulation_param_checks(params, input_obj=None, mechanism_spec=None, options=None):
    options = _normalize_options(options)
    mode = options.get(
        "print checks",
        options.get("print_checks", options.get("check params", options.get("check_params", False))),
    )
    if not _truthy_option(mode):
        return
    checks = _simulation_param_check_dataframe(params, input_obj=input_obj, mechanism_spec=mechanism_spec)
    if _is_raw_display_mode(mode):
        print("Simulation Parameter Checks:")
        pprint.pprint(checks.to_dict("records"), sort_dicts=False)
        return
    _display_dataframe_sections("Simulation Parameter Checks:", [(None, checks)], options=options)


def _simulation_param_check_dataframe(params, input_obj=None, mechanism_spec=None):
    rows = _simulation_param_check_rows(params, input_obj=input_obj, mechanism_spec=mechanism_spec)
    if not rows:
        rows = [{"Status": "OK", "Path": "", "Check": "No parameter issues detected", "Detail": ""}]
    return pd.DataFrame(rows, columns=["Status", "Path", "Check", "Detail"])


def _simulation_param_check_rows(params, input_obj=None, mechanism_spec=None):
    params = {} if params is None else params
    rows = []
    concentrations = normalize_concentrations(_concentrations_from_params(params))
    diffusion = dict(params.get("diffusion", {}) or {})
    clean_diffusion = {str(name).rstrip("*"): value for name, value in diffusion.items()}

    for name in concentrations["bulk"]:
        clean_name = str(name).rstrip("*")
        if clean_name not in clean_diffusion:
            rows.append(
                {
                    "Status": "WARN",
                    "Path": _fit_target_path_label(("diffusion", clean_name)),
                    "Check": "Missing diffusion",
                    "Detail": f"Bulk/mobile species {clean_name!r} has no diffusion coefficient.",
                }
            )

    bulk_names = {str(name).rstrip("*") for name in concentrations["bulk"]}
    for name in diffusion:
        clean_name = str(name).rstrip("*")
        if clean_name not in bulk_names:
            rows.append(
                {
                    "Status": "WARN",
                    "Path": _fit_target_path_label(("diffusion", clean_name)),
                    "Check": "Unused diffusion",
                    "Detail": f"No bulk concentration is defined for {clean_name!r}.",
                }
            )

    if concentrations["surface"]:
        surface_names = ", ".join(str(name).rstrip("*") for name in concentrations["surface"])
        rows.append(
            {
                "Status": "INFO",
                "Path": "concentrations.surface",
                "Check": "Surface amount",
                "Detail": f"Surface species use coverage units and do not require diffusion: {surface_names}.",
            }
        )

    order_detail = _mechanism_species_order_detail(mechanism_spec, concentrations)
    if order_detail:
        rows.append(
            {
                "Status": "INFO",
                "Path": "mechanism",
                "Check": "Preset species order",
                "Detail": order_detail,
            }
        )

    for note in params.get("_fallbacks", []) or []:
        rows.append(
            {
                "Status": "INFO",
                "Path": "",
                "Check": "Parameter fallback",
                "Detail": str(note),
            }
        )

    cell = params.get("cell", {}) if isinstance(params, dict) else {}
    if _is_auto_value(cell.get("Cdl")) and getattr(input_obj, "i", None) is None:
        rows.append(
            {
                "Status": "WARN",
                "Path": "cell.Cdl",
                "Check": "Cdl auto unavailable",
                "Detail": "cell.Cdl='auto' requires measured current in the simulation input.",
            }
        )

    return rows


def _mechanism_species_order_detail(mechanism_spec, concentrations):
    if mechanism_spec is None:
        return ""
    preset = str(getattr(mechanism_spec, "preset", "") or "").strip().lower()
    surface_confined = bool(getattr(mechanism_spec, "surface_confined", False))
    if preset in {"", "raw"}:
        return ""
    if preset == "e":
        names = _first_two_species(_species_group(concentrations, surface_confined), fallback=("a", "b"))
    elif preset in {"ee", "ec"}:
        names = _first_three_species(_species_group(concentrations, surface_confined), fallback=("a", "b", "c"))
    elif preset in {"ece", "square"}:
        names = _first_four_species(_species_group(concentrations, surface_confined), fallback=("a", "b", "c", "d"))
    elif preset == "ecat":
        if surface_confined and concentrations["surface"]:
            names = _first_two_species(concentrations["surface"], fallback=("CatOx", "CatRed"))
        else:
            names = _first_four_species(
                concentrations["bulk"],
                fallback=("CatOx", "CatRed", "Substrate", "Product"),
            )
    else:
        return ""
    clean_names = [str(name).rstrip("*") for name in names]
    return f"{getattr(mechanism_spec, 'preset', preset)} preset uses species in order: {', '.join(clean_names)}."


def _maybe_print_simulated_cv_input_setup(input_obj, options, default=False):
    options = _normalize_options(options)
    mode = options.get("print setup", options.get("print_setup", default))
    if not _truthy_option(mode):
        return
    if _is_raw_display_mode(mode):
        print("Simulated CV Input Setup:")
        pprint.pprint(
            {
                "E": input_obj.E,
                "t": input_obj.t,
                "i": input_obj.i,
                "metadata": input_obj.metadata,
                "source": _simulation_source_label(input_obj.source),
            },
            sort_dicts=False,
        )
        return
    _display_dataframe_sections(
        "Simulated CV Input Setup:",
        [(None, _simulated_cv_input_setup_dataframe(input_obj))],
        options=options,
    )


def _maybe_print_simulated_cv_setup(result, options, default=False):
    options = _normalize_options(options)
    mode = options.get("print setup", options.get("print_setup", default))
    if not _truthy_option(mode):
        return
    if _is_raw_display_mode(mode):
        print("Simulated CV Setup:")
        pprint.pprint(
            {
                "summary": result.summary,
                "params": result.params,
                "mechanism": result.mechanism,
                "input": result.input,
                "data": result.data,
            },
            sort_dicts=False,
        )
        return
    _display_dataframe_sections(
        "Simulated CV Setup:",
        [(None, _simulated_cv_setup_dataframe(result))],
        options=options,
    )


def _simulated_cv_input_setup_dataframe(input_obj):
    metadata = getattr(input_obj, "metadata", {}) or {}
    rows = [
        {"Parameter": "Source", "Value": _simulation_source_label(getattr(input_obj, "source", None))},
        {"Parameter": "Kind", "Value": metadata.get("kind", "")},
        {"Parameter": "Points", "Value": _format_param_value(len(input_obj.E))},
        {"Parameter": "Scan Rate", "Value": _format_scan_rate_value(metadata.get("scan_rate", _scan_rate_from_input(input_obj)))},
        {"Parameter": "Segments", "Value": _format_param_value(_input_segment_count(metadata.get("segments")))},
        {"Parameter": "Incubation Time", "Value": _format_param_value(metadata.get("incubation_time", 0.0), "s")},
        {"Parameter": "Quiet Time", "Value": _format_param_value(metadata.get("quiet_time", 0.0), "s")},
        {"Parameter": "Potential Range", "Value": _format_array_range(input_obj.E, _simulation_axis_unit(input_obj, "potential", "V"))},
        {"Parameter": "Time Range", "Value": _format_array_range(input_obj.t, "s")},
        {"Parameter": "Potential Unit", "Value": _simulation_axis_unit(input_obj, "potential", "V")},
        {"Parameter": "Current Unit", "Value": _simulation_axis_unit(input_obj, "current", "A")},
        {"Parameter": "Has Current", "Value": str(input_obj.has_current)},
    ]
    if metadata.get("background_correction"):
        rows.append({"Parameter": "Background Correction", "Value": metadata.get("background_correction")})
    if input_obj.i is not None:
        rows.insert(
            8,
            {"Parameter": "Current Range", "Value": _format_array_range(input_obj.i, _simulation_axis_unit(input_obj, "current", "A"))},
        )
    return pd.DataFrame(rows, columns=["Parameter", "Value"])


def _simulated_cv_setup_dataframe(result):
    data = getattr(result, "data", pd.DataFrame())
    input_obj = getattr(result, "input", None)
    mechanism = getattr(result, "mechanism", None)
    mechanism_text = getattr(mechanism, "mechanism", mechanism)
    mechanism_preset = (
        getattr(mechanism, "preset", None)
        or (getattr(result, "summary", {}) or {}).get("preset")
        or (getattr(result, "summary", {}) or {}).get("mechanism preset")
        or ""
    )
    summary = getattr(result, "summary", {}) or {}
    units = getattr(result, "units", {}) or {}
    rows = [
        {"Parameter": "Backend", "Value": summary.get("backend", getattr(result, "software", ""))},
        {"Parameter": "Mechanism Preset", "Value": mechanism_preset},
        {"Parameter": "Mechanism", "Value": str(mechanism_text)},
        {"Parameter": "Current Sign", "Value": _format_param_value(summary.get("current_sign", getattr(result, "current_sign", "")))},
        {"Parameter": "Points", "Value": _format_param_value(len(data))},
        {"Parameter": "Scan Rate", "Value": _format_scan_rate_value(getattr(result, "scan_rate", np.nan))},
        {"Parameter": "Segments", "Value": _format_param_value(getattr(result, "segments", ""))},
        {"Parameter": "Incubation Time", "Value": _format_param_value(summary.get("incubation_time", _metadata_value(input_obj, "incubation_time", 0.0)), "s")},
        {"Parameter": "Quiet Time", "Value": _format_param_value(summary.get("quiet_time", _metadata_value(input_obj, "quiet_time", 0.0)), "s")},
        {"Parameter": "Potential Range", "Value": _format_column_range(data, "Potential", units.get("Potential", _simulation_axis_unit(input_obj, "potential", "V")))},
        {"Parameter": "Current Range", "Value": _format_column_range(data, "Current", units.get("Current", _simulation_axis_unit(input_obj, "current", "A")))},
        {"Parameter": "Time Range", "Value": _format_column_range(data, "Time", units.get("Time", "s"))},
        {"Parameter": "Units", "Value": _format_units_mapping(units)},
        {"Parameter": "Source", "Value": _simulation_source_label(getattr(input_obj, "source", None))},
        {"Parameter": "Input Has Measured Current", "Value": str(input_obj is not None and getattr(input_obj, "i", None) is not None)},
    ]
    return pd.DataFrame(rows, columns=["Parameter", "Value"])


def _maybe_print_simulation_concentration_states(result, options):
    options = _normalize_options(options)
    mode = options.get(
        "print states",
        options.get("print_states", options.get("print concentration states", False)),
    )
    if not _truthy_option(mode):
        return
    frame = _simulation_concentration_states_dataframe(result)
    if frame.empty:
        return
    _display_dataframe_sections(
        "Simulation Concentration States:",
        [(None, frame)],
        options=options,
    )


def _simulation_concentration_states_dataframe(result):
    report = (getattr(result, "summary", {}) or {}).get("parameter_model", {}) or {}
    states = report.get("states", {}) or {}
    initial = states.get("initial", {}) or {}
    equilibrated = states.get("equilibrated", initial) or initial
    incubated = states.get("incubated", equilibrated) or equilibrated
    rows = []
    for phase in ("bulk", "surface"):
        names = []
        for state in (initial, equilibrated, incubated):
            for name in ((state.get(phase, {}) or {})):
                if name not in names:
                    names.append(name)
        for name in names:
            entered = (initial.get(phase, {}) or {}).get(name, np.nan)
            equilibrium_value = (equilibrated.get(phase, {}) or {}).get(name, entered)
            incubation_value = (incubated.get(phase, {}) or {}).get(name, equilibrium_value)
            values = np.asarray([entered, equilibrium_value, incubation_value], dtype=float)
            if np.allclose(values, values[0], rtol=1e-9, atol=1e-30, equal_nan=True):
                continue
            unit = _concentration_unit(phase)
            rows.append(
                {
                    "Phase": phase,
                    "Species": name,
                    "Entered": _format_param_value(entered, unit),
                    "Equilibrated": _format_param_value(equilibrium_value, unit),
                    "Incubated": _format_param_value(incubation_value, unit),
                }
            )
    return pd.DataFrame(rows, columns=["Phase", "Species", "Entered", "Equilibrated", "Incubated"])


def _is_raw_display_mode(value):
    return isinstance(value, str) and value.strip().lower().replace("-", "_") in {"raw", "dict", "pprint"}


def _format_scan_rate_value(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not np.isfinite(value):
        return ""
    return _format_param_value(value, "V/s")


def _format_array_range(values, unit=""):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return ""
    return f"{_format_param_value(float(np.nanmin(values)), unit)} to {_format_param_value(float(np.nanmax(values)), unit)}"


def _format_column_range(frame, column, unit=""):
    if not isinstance(frame, pd.DataFrame) or column not in frame:
        return ""
    return _format_array_range(frame[column].to_numpy(dtype=float), unit)


def _format_units_mapping(units):
    if not units:
        return ""
    return ", ".join(f"{key}: {value}" for key, value in units.items())


def _metadata_value(input_obj, key, default=""):
    metadata = getattr(input_obj, "metadata", {}) or {}
    return metadata.get(key, default)


def _input_segment_count(segments):
    if isinstance(segments, (list, tuple, np.ndarray)):
        return len(segments)
    if segments in (None, ""):
        return ""
    try:
        return int(segments)
    except (TypeError, ValueError):
        return segments


def _simulation_source_label(source):
    if source is None:
        return ""
    if isinstance(source, str):
        return source
    name = getattr(source, "name", None)
    if name:
        return name
    return type(source).__name__


def _maybe_print_fit_params(initial_params, final_params, options, fit_spec=None):
    mode = options.get("print params", options.get("print_params", False))
    if not _truthy_option(mode):
        return
    if isinstance(mode, str) and mode.strip().lower().replace("-", "_") in {"raw", "dict", "pprint"}:
        print("Fitting Params:")
        pprint.pprint(
            {"initial": initial_params, "final": final_params},
            sort_dicts=False,
        )
    else:
        _display_dataframe_sections(
            "Fitting Params:",
            _simulation_param_comparison_dataframes(initial_params, final_params, fit_spec),
            options=options,
        )


def _maybe_print_group_fit_params(group_fit_spec, best_params, best_params_by_cv, datasets, options):
    _maybe_print_fit_params(group_fit_spec.get("base_params", {}), best_params, options, group_fit_spec)
    mode = options.get("print params", options.get("print_params", False))
    if not _truthy_option(mode):
        return
    per_cv_frame = _group_fit_per_cv_params_dataframe(group_fit_spec, best_params_by_cv, datasets)
    if per_cv_frame.empty:
        return
    _display_dataframe_sections(
        "Per-CV Fitting Params:",
        [(None, per_cv_frame)],
        options=options,
    )


def _maybe_print_fit_progress(rows, options):
    mode = options.get(
        "print progress",
        options.get("print_progress", options.get("print fit progress", options.get("print_fit_progress", False))),
    )
    if not _truthy_option(mode):
        return
    frame = pd.DataFrame(rows or [{"Eval": "", "Cost": "", "ΔCost": "", "Residual Norm": "", "Max |Residual|": "", "Current Scale": "", "Baseline Intercept": "", "Baseline Slope": "", "Changed Params": ""}])
    frame = _fit_progress_display_frame(frame, options)
    _display_dataframe_sections("Fitting Progression:", [(None, frame)], options=options)


def _maybe_print_fit_statistics(summary, corrections, options):
    stats_mode = options.get(
        "print stats",
        options.get("print_stats", options.get("print statistics", options.get("print_statistics", False))),
    )
    corrections_mode = options.get(
        "print corrections",
        options.get("print_corrections", options.get("print fit corrections", options.get("print_fit_corrections", False))),
    )
    print_stats = _truthy_option(stats_mode)
    print_corrections = _truthy_option(corrections_mode)

    if print_stats:
        _display_dataframe_sections(
            "Fitting Statistics:",
            [(None, _fit_statistics_dataframe(summary))],
            options=options,
        )
    if print_corrections or (print_stats and _fit_has_reportable_corrections(summary, corrections)):
        _display_dataframe_sections(
            "Fitting Corrections:",
            [(None, _fit_corrections_dataframe(summary, corrections))],
            options=options,
        )


def _fit_has_reportable_corrections(summary, corrections):
    if corrections:
        return True
    current_sign = summary.get("current_sign", summary.get("current sign"))
    return current_sign not in (None, "", 1, 1.0)


def _fit_statistics_dataframe(summary):
    rows = []
    for key, label, unit in [
        ("n_points", "Data Points", ""),
        ("n_parameters", "Fit Parameters", ""),
        ("degrees_of_freedom", "Degrees of Freedom", ""),
        ("n_evaluations", "Objective Evaluations", ""),
        ("cost", "Optimizer Cost", ""),
        ("final_cost", "Final Optimizer Cost", ""),
        ("residual_norm", "Residual Norm", "A"),
        ("rmse", "RMSE", "A"),
        ("mae", "MAE", "A"),
        ("max_abs_residual", "Max |Residual|", "A"),
        ("physical_cost", "Physical Cost", "A²"),
        ("physical_cost_per_point", "Physical Cost / Point", "A²"),
        ("reduced_physical_cost", "Reduced Physical Cost", "A²"),
        ("normalized_residual_norm", "Normalized Residual Norm", ""),
        ("normalized_rmse", "Normalized RMSE", ""),
        ("normalized_max_abs_residual", "Normalized Max |Residual|", ""),
        ("normalized_cost", "Normalized Cost", ""),
        ("normalized_cost_per_point", "Normalized Cost / Point", ""),
        ("reduced_normalized_cost", "Reduced Normalized Cost", ""),
    ]:
        value = summary.get(key)
        if value is None:
            continue
        rows.append(
            {
                "Parameter": label,
                "Value": _format_param_value(value, unit),
            }
        )
    return _rows_dataframe(rows, empty_message="(none)")


def _fit_corrections_dataframe(summary, corrections):
    corrections = {} if corrections is None else dict(corrections)
    rows = []
    current_sign = summary.get("current_sign", summary.get("current sign"))
    if current_sign not in (None, ""):
        rows.append({"Parameter": "Current Sign", "Value": _format_param_value(current_sign)})
    for key, label, unit in [
        ("current_scale", "Current Scale", ""),
        ("baseline_intercept", "Baseline Intercept", "A"),
        ("baseline_slope", "Baseline Slope", "A/V"),
    ]:
        if key in corrections:
            rows.append(
                {
                    "Parameter": label,
                    "Value": _format_param_value(corrections[key], unit),
                }
            )
    return _rows_dataframe(rows, empty_message="(none)")


def _fit_correction_label_and_unit(key):
    labels = {
        "current_scale": ("Current Scale", ""),
        "baseline_intercept": ("Baseline Intercept", "A"),
        "baseline_slope": ("Baseline Slope", "A/V"),
        "current_sign": ("Current Sign", ""),
        "current sign": ("Current Sign", ""),
    }
    if key in labels:
        return labels[key]
    return str(key).replace("_", " ").replace("-", " ").title(), ""


def _fit_progress_display_frame(frame, options):
    frame = _drop_uninformative_progress_columns(frame)
    return _compact_progress_rows(frame, options)


def _drop_uninformative_progress_columns(frame):
    keep = []
    for column in frame.columns:
        values = frame[column]
        if _progress_column_has_information(values):
            keep.append(column)
    return frame.loc[:, keep] if keep else frame


def _progress_column_has_information(values):
    normalized = [_normalize_progress_cell(value) for value in values]
    nonempty = [value for value in normalized if value != ""]
    if not nonempty:
        return False
    return len(set(nonempty)) > 1


def _normalize_progress_cell(value):
    if value is None:
        return ""
    if isinstance(value, float) and np.isnan(value):
        return ""
    return str(value)


def _compact_progress_rows(frame, options):
    limit = _progress_row_limit(options)
    if limit is None or len(frame) <= limit:
        return frame
    head_count = min(5, max(1, limit // 4))
    tail_count = max(1, limit - head_count - 1)
    if tail_count + head_count >= len(frame):
        return frame
    ellipsis = {column: "..." for column in frame.columns}
    return pd.concat(
        [
            frame.head(head_count),
            pd.DataFrame([ellipsis], columns=frame.columns),
            frame.tail(tail_count),
        ],
        ignore_index=True,
    )


def _progress_row_limit(options):
    print_progress = options.get(
        "print progress",
        options.get("print_progress", options.get("print fit progress", options.get("print_fit_progress", False))),
    )
    if _progress_row_value_is_all(print_progress):
        return None
    if _progress_row_value_is_limit(print_progress):
        return max(3, int(print_progress))

    value = options.get(
        "progress rows",
        options.get("progress_rows", options.get("print progress rows", options.get("print_progress_rows", 25))),
    )
    if _progress_row_value_is_all(value):
        return None
    if value in (None, False):
        return 25
    return max(3, int(value))


def _progress_row_value_is_all(value):
    if value in (None, False, "all", "All", "ALL"):
        return value not in (None, False)
    return isinstance(value, str) and value.strip().lower() == "all"


def _progress_row_value_is_limit(value):
    if isinstance(value, bool):
        return False
    if isinstance(value, (int, np.integer)):
        return True
    if isinstance(value, str):
        return value.strip().isdigit()
    return False


def _format_simulation_params_pretty(params):
    return _format_dataframe_sections_text(_simulation_param_dataframes(params))


def _simulation_param_dataframes(params, compact=False):
    sections = []
    setup_rows = []
    spatial = params.get("spatial")
    if isinstance(spatial, dict) and spatial:
        setup_rows.extend(_mapping_param_rows("spatial", spatial))
    if setup_rows:
        sections.append(("simulation setup", _parameter_dataframe(setup_rows)))

    cell = params.get("cell")
    if isinstance(cell, dict) and cell:
        sections.append(("cell", _parameter_dataframe(_mapping_param_rows("cell", cell))))

    if compact:
        species_frame = _compact_species_dataframe(params)
        if not species_frame.empty:
            sections.append(("species", species_frame))
    else:
        species_rows = _species_param_rows(params)
        if species_rows:
            sections.append(("species", _parameter_dataframe(species_rows)))

    mechanism_rows = []
    for group in ("kinetics", "reactions", "isotherm"):
        values = params.get(group)
        if isinstance(values, list) and values:
            mechanism_rows.extend(_list_param_rows(group, values))
    if mechanism_rows:
        if compact:
            sections.append(("mechanism", _compact_mechanism_dataframe(mechanism_rows)))
        else:
            sections.append(("mechanism", _parameter_dataframe(mechanism_rows, keep_step=True)))

    fallback_notes = list(params.get("_fallbacks", []) or [])
    if fallback_notes:
        sections.append(("notes", pd.DataFrame({"Note": fallback_notes})))

    if not sections:
        sections.append(("simulation setup", _parameter_dataframe([{"Group": "", "Path": "", "Parameter": "(none)", "Value": ""}])))
    return sections


def _simulation_param_comparison_dataframes(initial_params, final_params, fit_spec=None):
    fit_status = _fit_status_lookup(fit_spec)
    sections = []
    setup_rows = _mapping_param_comparison_rows(
        "spatial",
        (initial_params or {}).get("spatial"),
        (final_params or {}).get("spatial"),
        fit_status=fit_status,
    )
    if setup_rows:
        sections.append(("simulation setup", _parameter_dataframe(setup_rows)))

    cell_rows = _mapping_param_comparison_rows(
        "cell",
        (initial_params or {}).get("cell"),
        (final_params or {}).get("cell"),
        fit_status=fit_status,
    )
    if cell_rows:
        sections.append(("cell", _parameter_dataframe(cell_rows)))

    diffusion_rows = _mapping_param_comparison_rows(
        "diffusion",
        (initial_params or {}).get("diffusion"),
        (final_params or {}).get("diffusion"),
        group_label="bulk",
        fit_status=fit_status,
    )

    species_rows = list(diffusion_rows)
    initial_species = _concentration_sections((initial_params or {}).get("concentrations"))
    final_species = _concentration_sections((final_params or {}).get("concentrations"))
    for phase in ("bulk", "surface"):
        species_rows.extend(
            _mapping_param_comparison_rows(
                f"concentrations.{phase}",
                initial_species.get(phase),
                final_species.get(phase),
                group_label=phase,
                fit_status=fit_status,
            )
        )
    if species_rows:
        sections.append(("species", _parameter_dataframe(species_rows)))

    mechanism_rows = []
    for group in ("kinetics", "reactions", "isotherm"):
        mechanism_rows.extend(
            _list_param_comparison_rows(
                group,
                (initial_params or {}).get(group),
                (final_params or {}).get(group),
                fit_status=fit_status,
            )
        )
    if mechanism_rows:
        sections.append(("mechanism", _parameter_dataframe(mechanism_rows, keep_step=True)))

    if not sections:
        sections.append(("simulation setup", _parameter_dataframe([{"Group": "", "Path": "", "Parameter": "(none)", "Fit Status": "", "Initial Value": "", "Final Value": ""}])))
    return sections


def _compact_param_mode(mode):
    return isinstance(mode, str) and _canonical_token(mode) == "compact"


def _species_param_rows(params):
    rows = []
    concentrations = normalize_concentrations((params or {}).get("concentrations", {}) or {})
    diffusion = (params or {}).get("diffusion", {}) or {}
    activity = _activity_config(params or {})
    show_activity = _has_nonideal_activity(params or {})
    for phase in ("bulk", "surface"):
        for name, value in concentrations[phase].items():
            row = {
                "Group": phase,
                "Path": _fit_target_path_label(("concentrations", phase, name)),
                "Species": name,
                "Parameter": _concentration_symbol(phase, name),
                "Value": _format_param_value(value, _concentration_unit(phase)),
            }
            if show_activity:
                row["gamma"] = _format_param_value(_activity_gamma(activity, phase, name))
                row["Activity"] = _format_param_value(_species_activity(value, activity, phase, name))
            rows.append(row)
            if phase == "bulk" and name in diffusion:
                rows.append(_diffusion_param_row(name, diffusion[name], group_label=phase, show_activity=show_activity))
    for name, value in diffusion.items():
        clean_name = str(name).rstrip("*")
        if clean_name not in concentrations["bulk"]:
            rows.append(_diffusion_param_row(name, value, group_label="bulk", show_activity=show_activity))
    return rows


def _diffusion_param_row(name, value, group_label="bulk", show_activity=False):
    clean_name = str(name).rstrip("*")
    row = {
        "Group": group_label,
        "Path": _fit_target_path_label(("diffusion", clean_name)),
        "Species": clean_name,
        "Parameter": _parameter_symbol("diffusion", clean_name),
        "Value": _format_param_value(value, _parameter_unit("diffusion", clean_name)),
    }
    if show_activity:
        row["gamma"] = ""
        row["Activity"] = ""
    return row


def _compact_species_dataframe(params):
    concentrations = normalize_concentrations((params or {}).get("concentrations", {}) or {})
    diffusion = (params or {}).get("diffusion", {}) or {}
    activity = _activity_config(params or {})
    show_activity = _has_nonideal_activity(params or {})
    rows = []
    for phase in ("bulk", "surface"):
        for name, value in concentrations[phase].items():
            clean_name = str(name).rstrip("*")
            row = {
                "Phase": phase,
                "Species": clean_name,
                "Amount": _format_param_value(value, _concentration_unit(phase)),
                "Diffusion": (
                    _format_param_value(diffusion[clean_name], _parameter_unit("diffusion", clean_name))
                    if phase == "bulk" and clean_name in diffusion
                    else ""
                ),
            }
            if show_activity:
                row["gamma"] = _format_param_value(_activity_gamma(activity, phase, clean_name))
                row["Activity"] = _format_param_value(_species_activity(value, activity, phase, clean_name))
            rows.append(row)
    for name, value in diffusion.items():
        clean_name = str(name).rstrip("*")
        if clean_name not in concentrations["bulk"]:
            row = {
                "Phase": "bulk",
                "Species": clean_name,
                "Amount": "",
                "Diffusion": _format_param_value(value, _parameter_unit("diffusion", clean_name)),
            }
            if show_activity:
                row["gamma"] = ""
                row["Activity"] = ""
            rows.append(row)
    columns = ["Phase", "Species", "Amount", "Diffusion"]
    if show_activity:
        columns.extend(["gamma", "Activity"])
    return pd.DataFrame(rows, columns=columns)


def _compact_mechanism_dataframe(rows):
    if not rows:
        return _rows_dataframe([], empty_message="(none)")
    compact_rows = {}
    for row in rows:
        key = (row.get("Group", ""), row.get("Step", ""))
        compact_rows.setdefault(key, {"Group": key[0], "Step": key[1]})
        compact_rows[key][row.get("Parameter", "")] = row.get("Value", "")
    frame = pd.DataFrame(list(compact_rows.values()))
    if frame.empty:
        return frame
    ordered = ["Group", "Step"]
    ordered.extend(column for column in frame.columns if column not in ordered)
    frame = frame.loc[:, ordered]
    return frame.fillna("")


def _concentration_symbol(phase, name):
    return f"Γ({name})" if phase == "surface" else f"[{name}]"


def _concentration_unit(phase):
    return "mol/m²" if phase == "surface" else "mol/m³"


def _fit_status_lookup(fit_spec):
    status = {}
    if not isinstance(fit_spec, dict):
        return status
    for target in fit_spec.get("vary", []) or []:
        paths = _fit_target_paths(target)
        label = "fit-tied" if len(paths) > 1 else "fit"
        for path in paths:
            status[path] = label
    for path in (fit_spec.get("fixed", {}) or {}):
        status.setdefault(path, "fixed")
    return status


def _fit_status_for_path(path, fit_status):
    return fit_status.get(tuple(path), "fixed")


def _concentration_sections(concentrations):
    if not isinstance(concentrations, dict) or not concentrations:
        return {"bulk": {}, "surface": {}}
    if any(key in concentrations for key in ("bulk", "surface")):
        return {
            "bulk": concentrations.get("bulk") if isinstance(concentrations.get("bulk"), dict) else {},
            "surface": concentrations.get("surface") if isinstance(concentrations.get("surface"), dict) else {},
        }
    return {"bulk": concentrations, "surface": {}}


def _mapping_param_rows(name, values, group_label=None):
    group = str(group_label or name)
    return [
        {
            "Group": group,
            "Path": _fit_target_path_label(_comparison_mapping_path(name, key)),
            "Parameter": _parameter_symbol(name, key),
            "Value": _format_param_value(value, _parameter_unit(name, key)),
        }
        for key, value in values.items()
    ]


def _mapping_param_comparison_rows(name, initial_values, final_values, group_label=None, fit_status=None):
    initial_values = initial_values if isinstance(initial_values, dict) else {}
    final_values = final_values if isinstance(final_values, dict) else {}
    keys = list(initial_values)
    keys.extend(key for key in final_values if key not in initial_values)
    group = str(group_label or name)
    fit_status = {} if fit_status is None else fit_status
    rows = []
    for key in keys:
        path = _comparison_mapping_path(name, key)
        status = _fit_status_for_path(path, fit_status)
        rows.append(
            {
            "Group": group,
            "Path": _fit_target_path_label(path),
            "Parameter": _parameter_symbol(name, key),
            "Fit Status": status,
            "Initial Value": _format_param_value(initial_values.get(key, ""), _parameter_unit(name, key)),
            "Final Value": _comparison_final_value(status, final_values.get(key, ""), _parameter_unit(name, key)),
            }
        )
    return rows


def _comparison_final_value(status, value, unit):
    return "" if status == "fixed" else _format_param_value(value, unit)


def _comparison_mapping_path(name, key):
    if name == "cell":
        return ("cell", key)
    if name == "diffusion":
        return ("diffusion", key)
    if name == "concentrations.bulk":
        return ("concentrations", "bulk", key)
    if name == "concentrations.surface":
        return ("concentrations", "surface", key)
    return (name, key)


def _list_param_rows(name, values):
    rows = []
    for index, entry in enumerate(values):
        if isinstance(entry, dict):
            for key, value in entry.items():
                rows.append(
                    {
                        "Group": name,
                        "Path": _fit_target_path_label((name, index, key)),
                        "Step": index,
                        "Parameter": _list_parameter_symbol(name, key, index),
                        "Value": _format_param_value(value, _parameter_unit(name, key)),
                    }
                )
        else:
            rows.append(
                {
                    "Group": name,
                    "Path": _fit_target_path_label((name, index)),
                    "Step": index,
                    "Parameter": "value",
                    "Value": _format_param_value(entry, _parameter_unit(name, "value")),
                }
            )
    return rows


def _list_param_comparison_rows(name, initial_values, final_values, fit_status=None):
    initial_values = initial_values if isinstance(initial_values, list) else []
    final_values = final_values if isinstance(final_values, list) else []
    fit_status = {} if fit_status is None else fit_status
    rows = []
    for index in range(max(len(initial_values), len(final_values))):
        initial_entry = initial_values[index] if index < len(initial_values) else {}
        final_entry = final_values[index] if index < len(final_values) else {}
        if isinstance(initial_entry, dict) or isinstance(final_entry, dict):
            initial_entry = initial_entry if isinstance(initial_entry, dict) else {}
            final_entry = final_entry if isinstance(final_entry, dict) else {}
            keys = list(initial_entry)
            keys.extend(key for key in final_entry if key not in initial_entry)
            for key in keys:
                status = _fit_status_for_path((name, index, key), fit_status)
                rows.append(
                    {
                        "Group": name,
                        "Path": _fit_target_path_label((name, index, key)),
                        "Step": index,
                        "Parameter": _list_parameter_symbol(name, key, index),
                        "Fit Status": status,
                        "Initial Value": _format_param_value(initial_entry.get(key, ""), _parameter_unit(name, key)),
                        "Final Value": _comparison_final_value(status, final_entry.get(key, ""), _parameter_unit(name, key)),
                    }
                )
        else:
            status = _fit_status_for_path((name, index), fit_status)
            rows.append(
                {
                    "Group": name,
                    "Path": _fit_target_path_label((name, index)),
                    "Step": index,
                    "Parameter": "value",
                    "Fit Status": status,
                    "Initial Value": _format_param_value(initial_entry, _parameter_unit(name, "value")),
                    "Final Value": _comparison_final_value(status, final_entry, _parameter_unit(name, "value")),
                }
            )
    return rows


def _list_parameter_symbol(section, key, index):
    if str(section) == "reactions":
        return _reaction_rate_symbol(key, index)
    return _parameter_symbol(section, key)


def _reaction_rate_symbol(key, index):
    lower = str(key).lower()
    step = int(index) + 1
    if lower == "kf":
        return f"k{_subscript_int(step)}"
    if lower == "kb":
        return f"k{_subscript_int(-step)}"
    if lower == "k":
        return f"k{_subscript_int(step)}"
    return _parameter_symbol("reactions", key)


def _subscript_int(value):
    digits = str(abs(int(value))).translate(str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉"))
    return f"₋{digits}" if int(value) < 0 else digits


def _parameter_symbol(section, key):
    section = str(section)
    key_text = str(key)
    lower = key_text.lower()
    symbols = {
        "t": "T",
        "ru": "Rᵤ",
        "cdl": "Cdl",
        "a": "A" if section == "cell" else key_text,
        "dx_fraction": "Δx/xmax",
        "fraction": "Δx/xmax",
        "nx": "nₓ",
        "viscosity": "η",
        "rotation": "ω",
        "rot_freq": "ω",
        "alpha": "α",
        "k0": "k⁰",
        "e0": "E⁰",
        "kf": "k₁",
        "kb": "k₋₁",
        "k": "k",
        "k_exchange": "k_exchange",
        "koff": "koff",
    }
    if section == "diffusion":
        return "D" if key_text == "D" else f"D({key_text})"
    if section == "concentrations.bulk":
        return f"[{key_text}]"
    if section == "concentrations.surface":
        return f"Γ({key_text})"
    return symbols.get(lower, key_text)


def _parameter_unit(section, key):
    section = str(section)
    lower = str(key).lower()
    if section == "cell":
        return {"t": "K", "ru": "Ω", "cdl": "F", "a": "m²"}.get(lower, "")
    if section == "spatial":
        return {"viscosity": "m²/s", "rotation": "Hz", "rot_freq": "Hz"}.get(lower, "")
    if section == "diffusion":
        return "m²/s"
    if section == "concentrations.bulk":
        return "mol/m³"
    if section == "concentrations.surface":
        return "mol/m²"
    if section == "kinetics":
        return {"k0": "m/s", "e0": "V"}.get(lower, "")
    if section == "reactions":
        return {"kf": "s⁻¹", "kb": "s⁻¹", "k": "s⁻¹", "koff": "s⁻¹", "k_exchange": "s⁻¹"}.get(lower, "")
    return ""


def _format_param_value(value, unit=""):
    if isinstance(value, float):
        text = f"{value:.6g}"
    elif isinstance(value, (np.floating,)):
        text = f"{float(value):.6g}"
    elif isinstance(value, (int, np.integer)) and not isinstance(value, bool):
        text = str(int(value))
    else:
        text = str(value)
    if unit and _value_should_show_unit(value):
        return f"{text} {unit}"
    return text


def _value_should_show_unit(value):
    if value is None or isinstance(value, bool):
        return False
    if isinstance(value, (int, float, np.integer, np.floating)):
        return True
    if isinstance(value, str):
        try:
            float(value)
        except ValueError:
            return False
        return True
    return False


def _display_dataframe_sections(title, sections, options=None):
    options = _normalize_options(options)
    if options.get("pretty print", True) and _can_rich_display():
        from IPython.display import display

        for name, frame in sections:
            caption = f"{title} {name}" if name else title
            display(_left_justified_dataframe(frame, caption=caption))
        return
    print(title)
    print(_format_dataframe_sections_text(sections))


def _left_justified_dataframe(frame, caption=None):
    if frame is None:
        return frame
    try:
        styled = frame.style.set_properties(**{"text-align": "left", "white-space": "pre-line"}).set_table_styles(
            [
                {
                    "selector": "caption",
                    "props": [
                        ("caption-side", "top"),
                        ("text-align", "left"),
                        ("font-weight", "600"),
                        ("color", "inherit"),
                        ("margin-bottom", "0.35em"),
                    ],
                },
                {"selector": "th", "props": [("text-align", "left")]},
                {"selector": "td", "props": [("text-align", "left"), ("white-space", "pre-line")]},
            ]
        )
        if caption:
            styled = styled.set_caption(str(caption).strip())
        return styled
    except Exception:
        return frame


def _can_rich_display():
    try:
        from IPython import get_ipython

        return get_ipython() is not None
    except Exception:
        return False


def _format_dataframe_sections_text(sections):
    parts = []
    for name, frame in sections:
        if frame is None or frame.empty:
            body = "(none)"
        else:
            body = frame.to_string(index=False)
        parts.append(f"[{name}]\n{body}" if name else body)
    return "\n\n".join(parts)


def _fit_simulation_options(options):
    sim_options = dict(options)
    sim_options["print params"] = False
    sim_options["print_params"] = False
    sim_options["check params"] = False
    sim_options["check_params"] = False
    sim_options["print checks"] = False
    sim_options["print_checks"] = False
    return sim_options


def _maybe_print_fitting_setup(
    input_obj,
    mechanism,
    fit_spec,
    options,
    *,
    backend,
    method,
    residual_mode,
    post_correction_mode,
    residual_normalization,
):
    if (
        ("print setup" in options or "print_setup" in options)
        and not _truthy_option(options.get("print setup", options.get("print_setup", False)))
    ):
        return
    mode = options.get(
        "print fitting",
        options.get("print_fitting", options.get("print setup", options.get("print_setup", False))),
    )
    if not _truthy_option(mode):
        return
    if isinstance(mode, str) and mode.strip().lower().replace("-", "_") in {"raw", "dict", "pprint"}:
        print("Fitting Setup:")
        pprint.pprint(
            {
                "method": method,
                "backend": backend,
                "mechanism": getattr(mechanism, "mechanism", mechanism),
                "fit_spec": fit_spec,
                "options": options,
            },
            sort_dicts=False,
        )
        return
    _display_dataframe_sections(
        "Fitting Setup:",
        _fitting_setup_dataframes(
            input_obj,
            mechanism,
            fit_spec,
            options,
            backend=backend,
            method=method,
            residual_mode=residual_mode,
            post_correction_mode=post_correction_mode,
            residual_normalization=residual_normalization,
        ),
        options=options,
    )


def _format_fitting_setup_pretty(
    input_obj,
    mechanism,
    fit_spec,
    options,
    *,
    backend,
    method,
    residual_mode,
    post_correction_mode,
    residual_normalization,
):
    return _format_dataframe_sections_text(
        _fitting_setup_dataframes(
            input_obj,
            mechanism,
            fit_spec,
            options,
            backend=backend,
            method=method,
            residual_mode=residual_mode,
            post_correction_mode=post_correction_mode,
            residual_normalization=residual_normalization,
        )
    )


def _fitting_setup_dataframes(
    input_obj,
    mechanism,
    fit_spec,
    options,
    *,
    backend,
    method,
    residual_mode,
    post_correction_mode,
    residual_normalization,
):
    mechanism_spec = compile_mechanism(mechanism, fit_spec.get("base_params", {}))
    rows = {
        "method": method,
        "backend": backend,
        "mechanism": mechanism_spec.mechanism.replace("\n", " ; "),
        "preset": mechanism_spec.preset,
        "residual": residual_mode,
        "post correction": post_correction_mode,
        "residual normalization": residual_normalization,
        "max_nfev": options.get("max_nfev"),
        "input points": len(input_obj.E),
        "fit parameters": len(fit_spec.get("vary", [])),
    }
    sections = [("fit", _settings_dataframe(rows))]

    vary_rows = []
    for target in fit_spec.get("vary", []):
        entry = fit_spec["entries"][target]
        lower, upper = entry["bounds"]
        unit = _fit_target_unit(target)
        vary_rows.append(
            {
                "Path": _fit_target_path_label(target),
                "Step": _fit_target_step(target),
                "Parameter": _fit_target_setup_symbol(target),
                "Initial": _format_param_value(entry["init"], unit),
                "Lower": _format_param_value(lower, unit),
                "Upper": _format_param_value(upper, unit),
                "Transform": entry["transform"],
            }
        )
    sections.append(("vary", _rows_dataframe(vary_rows, empty_message="(none)")))

    fixed_rows = [
        {
            "Path": _fit_target_path_label(path),
            "Step": _fit_target_step(path),
            "Parameter": _fit_target_setup_symbol(path),
            "Value": _format_param_value(value, _fit_target_unit(path)),
        }
        for path, value in (fit_spec.get("fixed", {}) or {}).items()
    ]
    sections.append(("fixed", _rows_dataframe(fixed_rows, empty_message="(none explicitly fixed)")))
    return sections


def _settings_dataframe(rows):
    return pd.DataFrame(
        [
            {
                "Parameter": _fit_setting_label(key),
                "Control": _fit_setting_control(key),
                "Value": _fit_setting_value(key, value),
            }
            for key, value in rows.items()
        ]
    )


def _fit_setting_value(key, value):
    if isinstance(value, str) and key in {
        "method",
        "residual",
        "post correction",
        "residual normalization",
    }:
        return value.replace("_", " ")
    return _format_param_value(value)


def _rows_dataframe(rows, empty_message="(none)"):
    if not rows:
        return _parameter_dataframe([{"Path": "", "Parameter": empty_message, "Value": ""}])
    return _parameter_dataframe(rows)


def _parameter_dataframe(rows, keep_step=False):
    frame = pd.DataFrame(rows)
    if "Step" in frame.columns and not keep_step and not _should_show_step_column(frame["Step"]):
        frame = frame.drop(columns=["Step"])
    return frame


def _should_show_step_column(values):
    nonblank = []
    for value in values:
        if value in ("", None):
            continue
        if isinstance(value, float) and np.isnan(value):
            continue
        nonblank.append(value)
    return len(set(nonblank)) > 1


def _fit_setting_label(key):
    symbols = {
        "method": "Fit strategy",
        "backend": "Simulation backend",
        "mechanism": "Mechanism",
        "preset": "Mechanism preset",
        "residual": "Residual mode",
        "post correction": "Final current correction",
        "residual normalization": "Residual normalization",
        "max_nfev": "Evaluation budget",
        "input points": "Data points",
        "fit parameters": "Fit targets",
    }
    return symbols.get(str(key), str(key))


def _fit_setting_control(key):
    controls = {
        "method": "method",
        "backend": "backend",
        "mechanism": "mechanism",
        "preset": "compile_mechanism(...).preset",
        "residual": 'options["residual"]',
        "post correction": 'options["post correction"]',
        "residual normalization": 'options["residual normalization"]',
        "max_nfev": 'options["max nfev"]',
        "input points": "input.E",
        "fit parameters": 'fit["vary"]',
    }
    return controls.get(str(key), str(key))


def _fit_target_label(target):
    paths = _fit_target_paths(target)
    if len(paths) > 1:
        return _tied_fit_target_label(paths)
    path = paths[0]
    if len(path) == 3 and path[0] == "kinetics" and path[2] in {"E0", "k0", "alpha"}:
        return f"{path[2]}_{path[1]}"
    if len(path) == 3 and path[0] == "reactions" and path[2] in {"kf", "kb", "k"}:
        return f"{path[2]}_{path[1]}"
    if len(path) == 2 and path[0] == "diffusion":
        return "D" if path[1] == "D" else f"D_{path[1]}"
    if len(path) == 2 and path[0] == "cell":
        return "Area" if path[1] == "A" else str(path[1])
    return ".".join(str(part) for part in path)


def _fit_target_symbol(target):
    paths = _fit_target_paths(target)
    if len(paths) > 1:
        return _tied_fit_target_symbol(paths)
    path = paths[0]
    if len(path) == 3 and path[0] == "kinetics":
        base = _parameter_symbol("kinetics", path[2])
        return f"{base}_{path[1]}" if path[2] in {"E0", "k0", "alpha"} else base
    if len(path) == 3 and path[0] == "reactions":
        return _reaction_rate_symbol(path[2], path[1])
    if len(path) == 2 and path[0] == "diffusion":
        return _parameter_symbol("diffusion", path[1])
    if len(path) == 2 and path[0] == "cell":
        return _parameter_symbol("cell", path[1])
    if len(path) >= 2 and path[0] == "concentrations":
        section = f"concentrations.{path[1]}" if len(path) > 2 else "concentrations"
        return _parameter_symbol(section, path[-1])
    return _fit_target_label(target)


def _fit_target_setup_symbol(target):
    paths = _fit_target_paths(target)
    if len(paths) > 1:
        return _tied_fit_target_symbol(paths)
    path = paths[0]
    if len(path) == 3 and path[0] == "kinetics":
        return _parameter_symbol("kinetics", path[2])
    if len(path) == 3 and path[0] == "reactions":
        return _reaction_rate_symbol(path[2], path[1])
    if len(path) == 3 and path[0] == "isotherm":
        return _parameter_symbol("isotherm", path[2])
    return _fit_target_symbol(target)


def _fit_target_step(target):
    paths = _fit_target_paths(target)
    steps = []
    for path in paths:
        if len(path) >= 3 and path[0] in {"kinetics", "reactions", "isotherm"}:
            steps.append(path[1])
    if not steps:
        return ""
    unique_steps = []
    for step in steps:
        if step not in unique_steps:
            unique_steps.append(step)
    return unique_steps[0] if len(unique_steps) == 1 else ", ".join(str(step) for step in unique_steps)


def _tied_fit_target_symbol(paths):
    if all(len(path) == 2 and path[0] == "diffusion" for path in paths):
        return "D (tied)"
    if all(len(path) == 3 and path[0] == "kinetics" and path[2] == paths[0][2] for path in paths):
        return f"{_parameter_symbol('kinetics', paths[0][2])} (tied)"
    if all(len(path) == 3 and path[0] == "reactions" and path[2] == paths[0][2] for path in paths):
        return f"{_reaction_rate_symbol(paths[0][2], paths[0][1])}... (tied)"
    return "tied"


def _tied_fit_target_label(paths):
    if all(len(path) == 2 and path[0] == "diffusion" for path in paths):
        return "D"
    if all(len(path) == 3 and path[0] == "kinetics" and path[2] == paths[0][2] for path in paths):
        return str(paths[0][2])
    if all(len(path) == 3 and path[0] == "reactions" and path[2] == paths[0][2] for path in paths):
        return str(paths[0][2])
    return " + ".join(_fit_target_label(path) for path in paths)


def _fit_target_unit(target):
    paths = _fit_target_paths(target)
    units = {_fit_path_unit(path) for path in paths}
    units.discard("")
    return next(iter(units)) if len(units) == 1 else ""


def _fit_path_unit(path):
    if len(path) == 2 and path[0] == "cell":
        return _parameter_unit("cell", path[1])
    if len(path) == 2 and path[0] == "diffusion":
        return _parameter_unit("diffusion", path[1])
    if len(path) == 3 and path[0] == "kinetics":
        return _parameter_unit("kinetics", path[2])
    if len(path) == 3 and path[0] == "reactions":
        return _parameter_unit("reactions", path[2])
    if len(path) >= 3 and path[0] == "concentrations":
        return _parameter_unit(f"concentrations.{path[1]}", path[-1])
    return ""


def _fit_target_path_label(target):
    paths = _fit_target_paths(target)
    labels = [".".join(str(part) for part in path) for path in paths]
    return ", ".join(labels)


def _normalize_fit_method(method, options=None):
    options = _normalize_options(options)
    if method is None:
        method = "least_squares"

    if isinstance(method, str):
        strategy = _normalize_fit_method_name(method)
        if strategy == "cma_es":
            raise NotImplementedError("CMA-ES strategy is not implemented yet; install/support will be added in a future release.")
        return _validated_fit_method(
            {
                "strategy": strategy,
                "optimizer": None,
                "starts": 1,
                "start_strategy": "nominal",
                "seed": None,
                "max_nfev": options.get("max_nfev"),
                "polish_with": None,
                "polish_max_nfev": None,
            },
            options,
        )

    if not isinstance(method, dict):
        raise ValueError("fit_cv method must be a string or a method specification dict.")

    spec = _normalize_options(method)
    strategy = _normalize_fit_method_name(spec.get("strategy", spec.get("name", "least_squares")))
    optimizer = spec.get("optimizer")
    optimizer = None if optimizer in (None, False, "none", "None") else _normalize_fit_method_name(optimizer)
    polish_with = spec.get("polish_with", spec.get("polish with", spec.get("polish")))
    polish_with = None if polish_with in (None, False, "none", "None", "off", "Off") else _normalize_fit_method_name(polish_with)
    start_strategy = _canonical_token(spec.get("start_strategy", "latin_hypercube" if strategy == "multistart" else "nominal"))

    normalized = {
        "strategy": strategy,
        "optimizer": optimizer,
        "starts": int(spec.get("starts", 10 if strategy == "multistart" else 1)),
        "start_strategy": start_strategy,
        "seed": spec.get("seed"),
        "max_nfev": spec.get("max_nfev", options.get("max_nfev")),
        "polish_with": polish_with,
        "polish_max_nfev": spec.get("polish_max_nfev", spec.get("polish_maxiter")),
    }
    for key in ("maxiter", "popsize", "tol", "xtol", "ftol", "gtol", "start_scale"):
        if key in spec:
            normalized[key] = spec[key]
    return _validated_fit_method(normalized, options)


def _normalize_fit_method_name(value):
    text = _canonical_token(value)
    aliases = {
        "scipy": "least_squares",
        "least_squares_fit": "least_squares",
        "lsq": "least_squares",
        "multi_start": "multistart",
        "de": "differential_evolution",
        "differential_evolution_fit": "differential_evolution",
        "nelder_mead_fit": "nelder_mead",
        "nelder": "nelder_mead",
        "nm": "nelder_mead",
        "cmaes": "cma_es",
        "ekitty": "electrokitty",
        "electro_kitty": "electrokitty",
    }
    return aliases.get(text, text)


def _validated_fit_method(spec, options):
    strategy = spec["strategy"]
    valid_strategies = {"least_squares", "multistart", "differential_evolution", "nelder_mead", "cma_es", "electrokitty"}
    if strategy not in valid_strategies:
        raise ValueError(
            "Unknown fit strategy. Use 'least_squares', 'multistart', 'differential_evolution', "
            "'nelder_mead', 'cma_es', or 'electrokitty'."
        )

    if spec["starts"] < 1:
        raise ValueError("method['starts'] must be at least 1.")

    valid_polish = {None, "least_squares", "nelder_mead"}
    if spec["polish_with"] not in valid_polish:
        raise ValueError("method['polish_with'] must be None, 'least_squares', or 'nelder_mead'.")

    if strategy == "multistart":
        if spec["optimizer"] is None:
            raise ValueError("strategy='multistart' requires an optimizer, such as 'least_squares' or 'nelder_mead'.")
        if spec["optimizer"] not in {"least_squares", "nelder_mead"}:
            raise ValueError("method['optimizer'] for multistart must be 'least_squares' or 'nelder_mead'.")
        if spec["start_strategy"] not in {"nominal", "latin_hypercube", "random", "sobol", "around_init", "grid"}:
            raise ValueError("Unknown multistart start_strategy.")
    else:
        if spec["optimizer"] is not None:
            raise ValueError("optimizer is only valid for strategy='multistart'.")
        spec["optimizer"] = None

    if strategy == "cma_es":
        raise NotImplementedError("CMA-ES strategy is not implemented yet; install/support will be added in a future release.")

    if strategy != "multistart":
        spec["starts"] = int(spec.get("starts") or 1)
        if strategy != "least_squares":
            spec["start_strategy"] = spec.get("start_strategy") or "nominal"

    return spec


def _fit_method_label(method_spec):
    if isinstance(method_spec, str):
        return method_spec
    strategy = method_spec.get("strategy", "least_squares")
    if strategy == "multistart":
        return f"multistart/{method_spec.get('optimizer')}"
    return strategy


def _fit_vectors(fit_spec):
    vary_paths = fit_spec.get("vary", []) or []
    if not vary_paths:
        empty = np.asarray([], dtype=float)
        return empty, empty.copy(), empty.copy()
    x0 = np.asarray(
        [
            _external_to_optimizer(fit_spec["entries"][path]["init"], fit_spec["entries"][path]["transform"])
            for path in vary_paths
        ],
        dtype=float,
    )
    lower = np.asarray(
        [
            _external_to_optimizer(fit_spec["entries"][path]["bounds"][0], fit_spec["entries"][path]["transform"])
            for path in vary_paths
        ],
        dtype=float,
    )
    upper = np.asarray(
        [
            _external_to_optimizer(fit_spec["entries"][path]["bounds"][1], fit_spec["entries"][path]["transform"])
            for path in vary_paths
        ],
        dtype=float,
    )
    return x0, lower, upper


def _generate_fit_starts(method_spec, x0, lower, upper):
    x0 = np.asarray(x0, dtype=float)
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    if x0.size == 0:
        return np.zeros((1, 0), dtype=float)

    n_starts = int(method_spec.get("starts", 1) or 1)
    starts = np.empty((n_starts, x0.size), dtype=float)
    starts[0] = np.clip(x0, lower, upper)
    if n_starts == 1:
        return starts

    if not np.all(np.isfinite(lower)) or not np.all(np.isfinite(upper)):
        raise ValueError("Multistart strategies require finite optimizer-space bounds.")

    strategy = _canonical_token(method_spec.get("start_strategy", "latin_hypercube"))
    seed = method_spec.get("seed")
    rng = np.random.default_rng(seed)
    count = n_starts - 1

    if strategy == "nominal":
        starts[1:] = starts[0]
    elif strategy == "random":
        starts[1:] = lower + rng.random((count, x0.size)) * (upper - lower)
    elif strategy == "latin_hypercube":
        starts[1:] = _latin_hypercube_starts(count, x0.size, lower, upper, seed, rng)
    elif strategy == "sobol":
        starts[1:] = _sobol_starts(count, x0.size, lower, upper, seed, rng)
    elif strategy == "around_init":
        scale = float(method_spec.get("start_scale", 0.25))
        starts[1:] = np.clip(x0 + rng.normal(size=(count, x0.size)) * (upper - lower) * scale, lower, upper)
    elif strategy == "grid":
        starts[1:] = _grid_starts(count, x0.size, lower, upper)
    else:
        raise ValueError("Unknown start_strategy.")
    return starts


def _latin_hypercube_starts(count, dim, lower, upper, seed, rng):
    try:
        from scipy.stats import qmc

        sampler = qmc.LatinHypercube(d=dim, seed=seed)
        return qmc.scale(sampler.random(count), lower, upper)
    except Exception:
        return lower + rng.random((count, dim)) * (upper - lower)


def _sobol_starts(count, dim, lower, upper, seed, rng):
    try:
        from scipy.stats import qmc

        sampler = qmc.Sobol(d=dim, scramble=True, seed=seed)
        power = int(np.ceil(np.log2(max(count, 1))))
        sample = sampler.random_base2(power)[:count]
        return qmc.scale(sample, lower, upper)
    except Exception:
        return lower + rng.random((count, dim)) * (upper - lower)


def _grid_starts(count, dim, lower, upper):
    if dim > 2:
        raise ValueError("start_strategy='grid' is only supported for one- or two-parameter fits.")
    if dim == 1:
        return np.linspace(lower[0], upper[0], count, dtype=float).reshape(count, 1)
    side = int(np.ceil(np.sqrt(count)))
    axes = [np.linspace(lower[i], upper[i], side, dtype=float) for i in range(dim)]
    mesh = np.meshgrid(*axes, indexing="ij")
    points = np.column_stack([axis.ravel() for axis in mesh])
    return points[:count]


def fit_cv(input_or_result, mechanism=None, params=None, fit=None, options=None, backend="electrokitty", method="least squares"):
    """Fit a simulated CV to measured current."""
    options = _fit_default_display_options(_normalize_options(options))
    method_spec = _normalize_fit_method(method, options)

    input_obj, mechanism_value, params_value, backend_value, initial_result = _resolve_fit_inputs(
        input_or_result,
        mechanism,
        params,
        backend,
        options,
    )
    if getattr(input_obj, "i", None) is None:
        raise ValueError("fit_cv requires measured current. Use cv_data, not cv_program.")

    if method_spec["strategy"] == "electrokitty":
        return _fit_cv_electrokitty(
            input_obj,
            mechanism_value,
            params_value,
            fit,
            options,
            initial_result,
        )
    return _fit_cv_strategy(
        input_obj,
        mechanism_value,
        params_value,
        fit,
        options,
        backend_value,
        initial_result,
        method_spec,
    )


def fit_cvs(inputs_or_results, mechanism=None, params=None, fit=None, per_cv="auto", options=None, backend="electrokitty", method="least squares"):
    """Fit one shared simulation mechanism to multiple measured CV datasets."""
    options = _fit_default_display_options(_normalize_options(options))
    method_spec = _normalize_fit_method(method, options)
    if method_spec["strategy"] == "electrokitty":
        raise ValueError("fit_cvs group fitting does not support method='electrokitty'; use an eCAT optimizer strategy.")
    if isinstance(inputs_or_results, (SimulatedCVInput, SimulatedCV, _EcatCV)) or not isinstance(inputs_or_results, (list, tuple)):
        raise ValueError("fit_cvs requires a list or tuple of CV inputs/results.")
    if len(inputs_or_results) == 0:
        raise ValueError("fit_cvs requires at least one CV dataset.")

    datasets = [
        _resolve_fit_dataset(item, mechanism, params, backend, options, index=index)
        for index, item in enumerate(inputs_or_results)
    ]
    _apply_group_fit_multiplot_labels(datasets, options)
    for dataset in datasets:
        if getattr(dataset["input"], "i", None) is None:
            raise ValueError("fit_cvs requires measured current for every dataset. Use cv_data, not cv_program.")

    mechanism_value = datasets[0]["mechanism"]
    backend_value = datasets[0]["backend"]
    base_params = deepcopy(datasets[0]["params"])
    return _fit_cvs_strategy(
        datasets,
        mechanism_value,
        base_params,
        fit,
        per_cv,
        options,
        backend_value,
        method_spec,
    )


def _fit_default_display_options(options):
    options = dict(options or {})
    options.setdefault("print setup", True)
    options.setdefault("print params", True)
    options.setdefault("progress", True)
    return options


def _fit_cv_least_squares(input_obj, mechanism, params, fit, options, backend, initial_result):
    return _fit_cv_strategy(
        input_obj,
        mechanism,
        params,
        fit,
        options,
        backend,
        initial_result,
        _normalize_fit_method("least_squares", options),
    )


def _fit_cv_strategy(input_obj, mechanism, params, fit, options, backend, initial_result, method_spec):

    params = _prepare_simulation_params(
        input_obj,
        params,
        options,
        mechanism=mechanism,
        expand_parameter_model=False,
    )
    fit_spec = _normalize_fit_spec(fit, params, input_obj)
    base_params = fit_spec["base_params"]
    vary_paths = fit_spec["vary"]

    if initial_result is None:
        initial_result = simulate_cv(
            input_obj,
            mechanism,
            base_params,
            options={**_fit_simulation_options(options), "plot": False},
            backend=backend,
        )

    x0, lower, upper = _fit_vectors(fit_spec)
    progress_total = _fit_strategy_progress_total(method_spec, n_parameters=len(x0))

    residual_mode = _normalize_residual_mode(options)
    post_correction_mode = _normalize_post_correction_mode(options)
    residual_normalization = _normalize_residual_normalization(options)
    setup_options = dict(options)
    if progress_total is not None:
        setup_options["max_nfev"] = f"~{progress_total}"
    _maybe_print_fitting_setup(
        input_obj,
        mechanism,
        fit_spec,
        setup_options,
        backend=backend,
        method=_fit_method_label(method_spec),
        residual_mode=residual_mode,
        post_correction_mode=post_correction_mode,
        residual_normalization=residual_normalization,
    )

    last = {}
    progress_options = dict(options)
    progress_options.setdefault("max_nfev", progress_total)
    progress = _FitProgressReporter(progress_options)
    progress_rows = []
    objective_context = {"start_index": None, "phase": method_spec["strategy"]}

    def objective(x):
        trial_params = _params_from_fit_vector(base_params, fit_spec, x)
        sim_result = simulate_cv(
            input_obj,
            mechanism,
            trial_params,
            options={**_fit_simulation_options(options), "plot": False},
            backend=backend,
        )
        residuals, corrections, corrected_current = _fit_residual_components(sim_result, residual_mode)
        optimizer_residuals = _optimizer_residuals(residuals, sim_result, residual_normalization)
        last["params"] = trial_params
        last["simulation_result"] = sim_result
        last["residuals"] = residuals
        last["optimizer_residuals"] = optimizer_residuals
        last["corrections"] = corrections
        last["corrected_current"] = corrected_current
        progress.update(residuals=optimizer_residuals, params=trial_params)
        _record_fit_progress_row(
            progress_rows,
            progress.count,
            optimizer_residuals,
            corrections,
            trial_params,
            fit_spec,
            start_index=objective_context.get("start_index"),
            phase=objective_context.get("phase"),
        )
        return optimizer_residuals

    try:
        optimizer_result = _run_fit_strategy(
            method_spec,
            objective,
            x0,
            lower,
            upper,
            options,
            objective_context,
        )
        best_x = optimizer_result.x
    finally:
        progress.close()

    best_params = _params_from_fit_vector(base_params, fit_spec, best_x)
    final_result = simulate_cv(
        input_obj,
        mechanism,
        best_params,
        options={**_fit_simulation_options(options), "plot": False},
        backend=backend,
    )
    final_correction_mode = post_correction_mode or residual_mode
    residuals, corrections, corrected_current = _fit_residual_components(final_result, final_correction_mode)
    _apply_fit_current(
        final_result,
        corrected_current,
        residuals,
        corrections,
        residual_mode,
        post_correction_mode=post_correction_mode,
    )
    fit_metrics = _fit_quality_metrics(
        final_result,
        residuals,
        n_parameters=len(vary_paths),
        residual_normalization=residual_normalization,
    )
    final_result.summary.update(fit_metrics)
    fit_summary = {
        "method": method_spec["strategy"],
        "strategy": method_spec["strategy"],
        "optimizer": method_spec.get("optimizer"),
        "polish_with": method_spec.get("polish_with"),
        "starts": method_spec.get("starts"),
        "start_strategy": method_spec.get("start_strategy"),
        "seed": method_spec.get("seed"),
        "backend": backend,
        "residual": residual_mode,
        "post_correction": post_correction_mode,
        "residual_normalization": residual_normalization,
        "n_parameters": len(vary_paths),
        "cost": getattr(optimizer_result, "cost", None),
        "best_start_index": getattr(optimizer_result, "best_start_index", None),
        "best_pre_polish_cost": _fit_result_cost(getattr(optimizer_result, "pre_polish_result", None), getattr(optimizer_result, "cost", None)),
        "final_cost": getattr(optimizer_result, "cost", None),
        "success": getattr(optimizer_result, "success", None),
        "n_evaluations": progress.count,
        "current_sign": final_result.summary.get("current_sign"),
        "current sign": final_result.summary.get("current sign"),
        **fit_metrics,
    }
    _maybe_print_fit_progress(progress_rows, options)
    _maybe_print_fit_statistics(fit_summary, corrections, options)
    _maybe_print_fit_params(
        _prepare_simulation_params(input_obj, base_params, options, mechanism=mechanism),
        final_result.params,
        options,
        fit_spec,
    )

    fit_result = SimulationFitResult(
        best_params=best_params,
        fit_spec=fit_spec,
        method=method_spec["strategy"],
        backend=backend,
        optimizer_result=optimizer_result,
        initial_result=initial_result,
        simulation_result=final_result,
        residuals=residuals,
        corrections=corrections,
        summary=fit_summary,
        measured_current=input_obj.i,
    )
    if bool(options.get("plot", True)):
        plot_options = dict(options.get("plot options", options))
        plot_options.setdefault("simulation label", "Simulated Fit")
        plot_options.setdefault("backend label", "Raw Simulated Fit")
        plot_options.setdefault("data label", "Measured Data")
        fit_result.plot(plot_options)
    return fit_result


def _fit_cvs_strategy(datasets, mechanism, params, fit, per_cv, options, backend, method_spec):
    dataset_params = [
        _prepare_simulation_params(
            dataset["input"],
            _params_with_dataset_inference(dataset["params"], dataset["input"], options),
            options,
            mechanism=dataset["mechanism"],
            expand_parameter_model=False,
        )
        for dataset in datasets
    ]
    for dataset, params_i in zip(datasets, dataset_params):
        dataset["input_concentrations"] = _input_concentration_label(dataset["input"])
        dataset["mapped_concentrations"] = _mapped_concentration_label(dataset["input"], params_i, options)
    shared_params = _prepare_simulation_params(
        datasets[0]["input"],
        params,
        options,
        mechanism=mechanism,
        expand_parameter_model=False,
    )
    fit_spec = _normalize_fit_spec(fit, shared_params, datasets[0]["input"])
    fixed_paths = fit_spec.get("fixed", {}) or {}
    for params_i in dataset_params:
        for path, value in fixed_paths.items():
            _set_param_path(params_i, path, value)

    per_cv_paths = _normalize_per_cv_paths(per_cv, fit_spec["base_params"], dataset_params, datasets, options)
    group_fit_spec = _group_fit_spec(fit_spec, per_cv_paths, dataset_params, datasets)
    x0, lower, upper = _fit_vectors(group_fit_spec)
    progress_total = _fit_strategy_progress_total(method_spec, n_parameters=len(x0))

    if _truthy_option(options.get("print setup", options.get("print_setup", False))):
        _display_dataframe_sections(
            "Group Fitting Setup:",
            [(None, _group_fit_datasets_dataframe(datasets))],
            options=options,
        )

    residual_mode = _normalize_residual_mode(options)
    post_correction_mode = _normalize_post_correction_mode(options)
    residual_normalization = _normalize_residual_normalization(options)
    progress_options = dict(options)
    progress_options.setdefault("max_nfev", progress_total)
    progress = _FitProgressReporter(progress_options)
    progress_rows = []
    objective_context = {"start_index": None, "phase": method_spec["strategy"]}

    initial_results = []
    for dataset, params_i in zip(datasets, dataset_params):
        if dataset["initial_result"] is not None:
            initial_results.append(dataset["initial_result"])
        else:
            initial_results.append(
                simulate_cv(
                    dataset["input"],
                    dataset["mechanism"],
                    params_i,
                    options={**_fit_simulation_options(options), "plot": False},
                    backend=dataset["backend"],
                )
            )

    def objective(x):
        trial_shared, trial_by_cv = _group_params_from_fit_vector(group_fit_spec, x)
        residual_chunks = []
        corrections_by_cv = []
        simulation_results = []
        raw_residuals_by_cv = []
        for dataset, params_i in zip(datasets, trial_by_cv):
            sim_result = simulate_cv(
                dataset["input"],
                dataset["mechanism"],
                params_i,
                options={**_fit_simulation_options(options), "plot": False},
                backend=dataset["backend"],
            )
            residuals, corrections, _corrected_current = _fit_residual_components(sim_result, residual_mode)
            optimizer_residuals = _optimizer_residuals(residuals, sim_result, residual_normalization)
            residual_chunks.append(optimizer_residuals)
            corrections_by_cv.append(corrections)
            simulation_results.append(sim_result)
            raw_residuals_by_cv.append(residuals)
        optimizer_residuals = np.concatenate(residual_chunks) if residual_chunks else np.asarray([], dtype=float)
        progress.update(residuals=optimizer_residuals, params=trial_shared)
        _record_fit_progress_row(
            progress_rows,
            progress.count,
            optimizer_residuals,
            _combine_group_corrections(corrections_by_cv),
            trial_shared,
            group_fit_spec,
            start_index=objective_context.get("start_index"),
            phase=objective_context.get("phase"),
        )
        return optimizer_residuals

    try:
        optimizer_result = _run_fit_strategy(method_spec, objective, x0, lower, upper, options, objective_context)
        best_x = optimizer_result.x
    finally:
        progress.close()

    best_params, best_params_by_cv = _group_params_from_fit_vector(group_fit_spec, best_x)
    final_results = []
    residuals_by_cv = []
    corrections_by_cv = []
    corrected_chunks = []
    final_correction_mode = post_correction_mode or residual_mode
    for dataset, params_i in zip(datasets, best_params_by_cv):
        final_result = simulate_cv(
            dataset["input"],
            dataset["mechanism"],
            params_i,
            options={**_fit_simulation_options(options), "plot": False},
            backend=dataset["backend"],
        )
        residuals, corrections, corrected_current = _fit_residual_components(final_result, final_correction_mode)
        _apply_fit_current(
            final_result,
            corrected_current,
            residuals,
            corrections,
            residual_mode,
            post_correction_mode=post_correction_mode,
        )
        final_results.append(final_result)
        residuals_by_cv.append(residuals)
        corrections_by_cv.append(corrections)
        corrected_chunks.append(residuals)
    all_residuals = np.concatenate(corrected_chunks) if corrected_chunks else np.asarray([], dtype=float)
    fit_summary = {
        "method": method_spec["strategy"],
        "strategy": method_spec["strategy"],
        "optimizer": method_spec.get("optimizer"),
        "backend": backend,
        "residual": residual_mode,
        "post_correction": post_correction_mode,
        "residual_normalization": residual_normalization,
        "n_parameters": len(group_fit_spec.get("vary", []) or []),
        "n_datasets": len(datasets),
        "n_points": int(len(all_residuals)),
        "cost": getattr(optimizer_result, "cost", None),
        "final_cost": getattr(optimizer_result, "cost", None),
        "success": getattr(optimizer_result, "success", None),
        "n_evaluations": progress.count,
        **_group_fit_quality_metrics(all_residuals, len(group_fit_spec.get("vary", []) or [])),
    }
    for result in final_results:
        result.summary.update(fit_summary)

    _maybe_print_fit_progress(progress_rows, options)
    stats_options = dict(options)
    stats_options["print corrections"] = False
    stats_options["print_corrections"] = False
    _maybe_print_fit_statistics(fit_summary, {}, stats_options)
    if _truthy_option(options.get("print corrections", options.get("print_corrections", False))):
        _display_dataframe_sections(
            "Group Fitting Corrections:",
            [(None, _group_fit_corrections_dataframe_from_parts(corrections_by_cv, datasets))],
            options=options,
        )
    _maybe_print_group_fit_params(group_fit_spec, best_params, best_params_by_cv, datasets, options)

    group_result = SimulationGroupFitResult(
        best_params=best_params,
        best_params_by_cv=best_params_by_cv,
        fit_spec=group_fit_spec,
        per_cv=per_cv_paths,
        method=method_spec["strategy"],
        backend=backend,
        optimizer_result=optimizer_result,
        initial_results=initial_results,
        simulation_results=final_results,
        measured_currents=[dataset["input"].i for dataset in datasets],
        residuals=all_residuals,
        residuals_by_cv=residuals_by_cv,
        corrections_by_cv=corrections_by_cv,
        summary=fit_summary,
        datasets=datasets,
    )
    if bool(options.get("plot", True)):
        group_result.plot(options.get("plot options", options))
    return group_result


def _normalize_per_cv_paths(per_cv, base_params, dataset_params, datasets, options):
    if per_cv in (None, False):
        return []
    if isinstance(per_cv, str) and per_cv.strip().lower() == "auto":
        return _auto_per_cv_paths(base_params, dataset_params)
    raw_paths = [per_cv] if isinstance(per_cv, (str, tuple)) and not isinstance(per_cv, list) else list(per_cv or [])
    out = []
    for path in raw_paths:
        for resolved in _resolve_fit_paths(path, base_params):
            if resolved not in out:
                out.append(resolved)
    return out


def _auto_per_cv_paths(base_params, dataset_params):
    candidates = []
    for key in ("A", "T", "Cdl", "Ru"):
        if "cell" in base_params and key in (base_params.get("cell", {}) or {}):
            candidates.append(("cell", key))
    concentrations = base_params.get("concentrations", {}) or {}
    if any(key in concentrations for key in ("bulk", "surface")):
        for phase in ("bulk", "surface"):
            for name in (concentrations.get(phase, {}) or {}):
                candidates.append(("concentrations", phase, name))
    else:
        for name in concentrations:
            candidates.append(("concentrations", "bulk", name))
    return [path for path in candidates if _dataset_param_values_differ(dataset_params, path)]


def _dataset_param_values_differ(dataset_params, path):
    values = []
    for params in dataset_params:
        try:
            values.append(_get_param_path(params, path))
        except Exception:
            values.append(None)
    if len(values) < 2:
        return False
    first = values[0]
    return any(not _fit_values_equal(first, value) for value in values[1:])


def _group_fit_spec(fit_spec, per_cv_paths, dataset_params, datasets):
    per_cv_set = set(per_cv_paths)
    group_spec = {
        "vary": [],
        "fixed": fit_spec.get("fixed", {}),
        "entries": {},
        "base_params": deepcopy(fit_spec["base_params"]),
        "per_cv_paths": list(per_cv_paths),
        "dataset_base_params": [deepcopy(item) for item in dataset_params],
        "shared_targets": [],
        "per_cv_targets": [],
    }
    for target in fit_spec.get("vary", []) or []:
        target_paths = _fit_target_paths(target)
        if any(path in per_cv_set for path in target_paths):
            for path in target_paths:
                if path not in per_cv_set:
                    group_spec["vary"].append(path)
                    group_spec["entries"][path] = deepcopy(fit_spec["entries"][target])
                    group_spec["shared_targets"].append(path)
                    continue
                for index, params_i in enumerate(dataset_params):
                    group_target = ("per_cv", index, path)
                    entry = deepcopy(fit_spec["entries"][target])
                    try:
                        init = _coerce_param_path_value(path, _get_param_path(params_i, path))
                        entry["init"] = _fit_init_within_bounds(init, entry["bounds"], allow_clip=True)
                    except Exception:
                        pass
                    group_spec["vary"].append(group_target)
                    group_spec["entries"][group_target] = entry
                    group_spec["per_cv_targets"].append(group_target)
        else:
            group_spec["vary"].append(target)
            group_spec["entries"][target] = deepcopy(fit_spec["entries"][target])
            group_spec["shared_targets"].append(target)
    return group_spec


def _group_params_from_fit_vector(group_fit_spec, x):
    shared_params = deepcopy(group_fit_spec["base_params"])
    params_by_cv = [deepcopy(item) for item in group_fit_spec["dataset_base_params"]]
    for value, target in zip(x, group_fit_spec.get("vary", []) or []):
        entry = group_fit_spec["entries"][target]
        external_value = _optimizer_to_external(value, entry["transform"])
        if _is_per_cv_fit_target(target):
            _, index, path = target
            _set_param_path(params_by_cv[index], path, external_value)
            continue
        for path in _fit_target_paths(target):
            _set_param_path(shared_params, path, external_value)
            for params_i in params_by_cv:
                _set_param_path(params_i, path, external_value)
    return shared_params, params_by_cv


def _is_per_cv_fit_target(target):
    return isinstance(target, tuple) and len(target) == 3 and target[0] == "per_cv" and isinstance(target[2], tuple)


def _params_with_dataset_inference(params, input_obj, options):
    params = deepcopy(params)
    notes = []
    for name, value in _input_concentration_values(input_obj).items():
        mapped = _mapped_concentration_species(name, params, options)
        if mapped is None:
            notes.append(f"No simulation species matched concentration {name!r}.")
            continue
        phase, species = mapped
        params.setdefault("concentrations", {}).setdefault(phase, {})[species] = float(value)
    if notes:
        params.setdefault("_inference_warnings", []).extend(notes)
    return params


def _input_concentration_values(input_obj):
    metadata = getattr(input_obj, "metadata", {}) or {}
    out = {}
    source = getattr(input_obj, "source", None)
    compounds = getattr(source, "compounds", None) or []
    concentrations = getattr(source, "concentrations", None) or []
    for name, concentration in zip(compounds, concentrations):
        parsed = _parse_concentration_value(concentration)
        if parsed is not None:
            out[str(name)] = parsed
    values = metadata.get("concentrations", metadata.get("concentration", {}))
    if isinstance(values, dict):
        out.update({str(name): float(value) for name, value in values.items() if _is_numeric(value)})
    return out


def _input_concentration_label(input_obj):
    source = getattr(input_obj, "source", None)
    compounds = getattr(source, "compounds", None) or []
    concentrations = getattr(source, "concentrations", None) or []
    labels = []
    for name, concentration in zip(compounds, concentrations):
        if concentration in (None, ""):
            labels.append(str(name))
        else:
            labels.append(f"{concentration} {name}")
    if labels:
        return "; ".join(labels)
    values = _input_concentration_values(input_obj)
    return "; ".join(
        f"{name}={_format_param_value(value, 'mol/m³')}"
        for name, value in values.items()
    )


def _mapped_concentration_label(input_obj, params, options):
    pieces = []
    for name, value in _input_concentration_values(input_obj).items():
        mapped = _mapped_concentration_species(name, params, options)
        if mapped is None:
            continue
        phase, species = mapped
        pieces.append(f"{name} → {species}: {_format_param_value(value, _concentration_unit(phase))}")
    return "; ".join(pieces)


def _mapped_concentration_species(name, params, options):
    mapping = options.get("concentration mapping", options.get("concentration_mapping", {})) or {}
    species_name = mapping.get(name, mapping.get(str(name), name))
    concentrations = normalize_concentrations(_concentrations_from_params(params))
    for phase in ("bulk", "surface"):
        for candidate in concentrations.get(phase, {}) or {}:
            if str(candidate) == str(species_name):
                return phase, candidate
    return None


def _parse_concentration_value(value):
    if _is_numeric(value):
        return float(value)
    text = str(value or "").strip()
    if not text:
        return None
    parts = text.split()
    try:
        number = float(parts[0])
    except (TypeError, ValueError):
        return None
    unit = parts[1] if len(parts) > 1 else ""
    unit_key = unit.lower().replace("μ", "u")
    factors = {
        "mol/m3": 1.0,
        "mol/m^3": 1.0,
        "m": 1000.0,
        "mm": 1.0,
        "um": 1e-3,
        "µm": 1e-3,
        "nm": 1e-6,
    }
    return number * factors.get(unit_key, 1.0)


def _group_fit_quality_metrics(residuals, n_parameters=0):
    residuals = np.asarray(residuals, dtype=float)
    n_points = int(len(residuals))
    dof = max(n_points - int(n_parameters or 0), 1)
    cost = float(0.5 * np.dot(residuals, residuals))
    abs_residuals = np.abs(residuals)
    return {
        "degrees_of_freedom": dof,
        "residual_norm": float(np.linalg.norm(residuals)),
        "rmse": float(np.sqrt(np.nanmean(residuals ** 2))) if n_points else np.nan,
        "mae": float(np.nanmean(abs_residuals)) if n_points else np.nan,
        "max_abs_residual": float(np.nanmax(abs_residuals)) if n_points else np.nan,
        "physical_cost": cost,
        "physical_cost_per_point": cost / n_points if n_points else np.nan,
        "reduced_physical_cost": cost / dof,
    }


def _combine_group_corrections(corrections_by_cv):
    combined = {}
    for index, corrections in enumerate(corrections_by_cv):
        for key, value in (corrections or {}).items():
            combined[f"CV {index + 1} {key}"] = value
    return combined


def _group_fit_datasets_dataframe(datasets):
    rows = []
    for dataset in datasets or []:
        input_obj = dataset.get("input")
        points = len(getattr(input_obj, "E", []))
        rows.append(
            {
                "CV": dataset.get("index", 0) + 1,
                "Label": dataset.get("label", ""),
                "Source": _simulation_source_label(dataset.get("source")),
                "Scan Rate": _format_scan_rate_value(dataset.get("scan_rate", np.nan)),
                "Input Concentrations": dataset.get("input_concentrations", ""),
                "Mapped Concentrations": dataset.get("mapped_concentrations", ""),
                "Points": points,
            }
        )
    frame = pd.DataFrame(rows)
    if not frame.empty and "Scan Rate" in frame.columns:
        scan_rates = frame["Scan Rate"].fillna("").astype(str)
        if scan_rates.nunique(dropna=False) <= 1:
            frame = frame.drop(columns=["Scan Rate"])
    for column in ("Input Concentrations", "Mapped Concentrations"):
        if column in frame.columns and not frame[column].fillna("").astype(str).str.strip().any():
            frame = frame.drop(columns=[column])
    return frame


def _group_fit_per_cv_params_dataframe(group_fit_spec, best_params_by_cv, datasets):
    per_cv_paths = list((group_fit_spec or {}).get("per_cv_paths", []) or [])
    dataset_base_params = list((group_fit_spec or {}).get("dataset_base_params", []) or [])
    if not per_cv_paths or not best_params_by_cv:
        return pd.DataFrame()
    rows = []
    vary_targets = set((group_fit_spec or {}).get("vary", []) or [])
    for index, final_params in enumerate(best_params_by_cv or []):
        initial_params = dataset_base_params[index] if index < len(dataset_base_params) else {}
        dataset = datasets[index] if index < len(datasets or []) else {}
        label = dataset.get("label", f"CV {index + 1}") if isinstance(dataset, dict) else f"CV {index + 1}"
        source = _simulation_source_label(dataset.get("source")) if isinstance(dataset, dict) else ""
        for path in per_cv_paths:
            path = tuple(path)
            try:
                initial_value = _get_param_path(initial_params, path)
            except Exception:
                initial_value = ""
            try:
                final_value = _get_param_path(final_params, path)
            except Exception:
                final_value = ""
            status = "fit" if ("per_cv", index, path) in vary_targets else "fixed"
            row = {
                "CV": index + 1,
                "Label": label,
                "Path": _fit_target_path_label(path),
                "Parameter": _fit_target_setup_symbol(path),
                "Fit Status": status,
                "Initial Value": _format_param_value(initial_value, _fit_target_unit(path)),
                "Final Value": _comparison_final_value(status, final_value, _fit_target_unit(path)),
            }
            if source and source != label:
                row["Source"] = source
            rows.append(row)
    frame = pd.DataFrame(rows)
    if not frame.empty and "Source" in frame.columns:
        source_values = frame["Source"].fillna("").astype(str)
        if source_values.nunique(dropna=False) <= 1:
            frame = frame.drop(columns=["Source"])
    return frame


def _group_fit_corrections_dataframe(result):
    return _group_fit_corrections_dataframe_from_parts(
        result.corrections_by_cv,
        result.datasets,
    )


def _group_fit_corrections_dataframe_from_parts(corrections_by_cv, datasets):
    rows = []
    datasets = list(datasets or [])
    for index, corrections in enumerate(corrections_by_cv or []):
        label = datasets[index].get("label", f"CV {index + 1}") if index < len(datasets) else f"CV {index + 1}"
        if not corrections:
            rows.append({"CV": index + 1, "Label": label, "Correction": "(none)", "Value": ""})
            continue
        for key, value in corrections.items():
            correction_label, correction_unit = _fit_correction_label_and_unit(key)
            rows.append(
                {
                    "CV": index + 1,
                    "Label": label,
                    "Correction": correction_label,
                    "Value": _format_param_value(value, correction_unit),
                }
            )
    return pd.DataFrame(rows)


def _fit_strategy_progress_total(method_spec, n_parameters=0):
    max_nfev = method_spec.get("max_nfev")
    if max_nfev is None:
        return None
    try:
        max_nfev = int(max_nfev)
    except (TypeError, ValueError):
        return None
    strategy = method_spec.get("strategy")
    if strategy == "multistart":
        local_multiplier = _fit_optimizer_eval_multiplier(method_spec.get("optimizer"), n_parameters)
        total = max_nfev * int(method_spec.get("starts", 1) or 1) * local_multiplier
        if method_spec.get("polish_with") and method_spec.get("polish_max_nfev"):
            total += int(method_spec["polish_max_nfev"]) * _fit_optimizer_eval_multiplier(
                method_spec.get("polish_with"),
                n_parameters,
            )
        return total
    total = max_nfev * _fit_optimizer_eval_multiplier(strategy, n_parameters)
    return total


def _fit_optimizer_eval_multiplier(optimizer, n_parameters=0):
    if optimizer == "least_squares":
        return max(1, int(n_parameters) + 1)
    return 1


def _run_fit_strategy(method_spec, objective, x0, lower, upper, options, objective_context):
    strategy = method_spec["strategy"]
    if strategy == "least_squares":
        objective_context.update({"start_index": 0, "phase": "least_squares"})
        return _run_local_fit("least_squares", objective, x0, lower, upper, method_spec, options)
    if strategy == "nelder_mead":
        objective_context.update({"start_index": 0, "phase": "nelder_mead"})
        return _run_local_fit("nelder_mead", objective, x0, lower, upper, method_spec, options)
    if strategy == "multistart":
        return _run_multistart_fit(method_spec, objective, x0, lower, upper, options, objective_context)
    if strategy == "differential_evolution":
        return _run_differential_evolution_fit(method_spec, objective, x0, lower, upper, options, objective_context)
    raise ValueError(f"Unsupported fit strategy {strategy!r}.")


def _run_multistart_fit(method_spec, objective, x0, lower, upper, options, objective_context):
    starts = _generate_fit_starts(method_spec, x0, lower, upper)
    children = []
    best = None
    best_index = None
    for index, start in enumerate(starts):
        objective_context.update({"start_index": index, "phase": "multistart"})
        child = _run_local_fit(
            method_spec["optimizer"],
            objective,
            start,
            lower,
            upper,
            method_spec,
            options,
            max_nfev=method_spec.get("max_nfev"),
        )
        children.append(child)
        if best is None or _fit_result_cost(child) < _fit_result_cost(best):
            best = child
            best_index = index

    pre_polish = best
    final = best
    polish_with = method_spec.get("polish_with")
    if polish_with:
        polish_spec = dict(method_spec)
        polish_spec["max_nfev"] = method_spec.get("polish_max_nfev")
        objective_context.update({"start_index": best_index, "phase": "polish"})
        polished = _run_local_fit(polish_with, objective, best.x, lower, upper, polish_spec, options)
        if _fit_result_cost(polished) <= _fit_result_cost(best) + 1e-12:
            final = polished
        children.append(polished)
    return _FitStrategyOptimizerResult(
        x=final.x,
        cost=_fit_result_cost(final),
        success=getattr(final, "success", None),
        message=getattr(final, "message", ""),
        nfev=sum(getattr(child, "nfev", 0) for child in children),
        strategy="multistart",
        optimizer=method_spec.get("optimizer"),
        children=children,
        best_start_index=best_index,
        pre_polish_result=pre_polish,
        polish_result=None if final is pre_polish else final,
    )


def _run_differential_evolution_fit(method_spec, objective, x0, lower, upper, options, objective_context):
    if x0.size == 0:
        objective_context.update({"start_index": 0, "phase": "differential_evolution"})
        return _run_noop_fit(objective, x0, strategy="differential_evolution")
    if not np.all(np.isfinite(lower)) or not np.all(np.isfinite(upper)):
        raise ValueError("differential_evolution requires finite optimizer-space bounds.")
    from scipy.optimize import differential_evolution

    objective_context.update({"start_index": None, "phase": "differential_evolution"})

    def scalar_objective(x):
        residuals = objective(x)
        return float(0.5 * np.dot(residuals, residuals))

    raw = differential_evolution(
        scalar_objective,
        bounds=list(zip(lower, upper)),
        seed=method_spec.get("seed"),
        maxiter=int(method_spec.get("maxiter", method_spec.get("max_nfev", 100) or 100)),
        popsize=int(method_spec.get("popsize", 15)),
        tol=float(method_spec.get("tol", 0.01)),
        polish=False,
    )
    pre_polish = _FitStrategyOptimizerResult(
        x=raw.x,
        cost=float(raw.fun),
        success=bool(raw.success),
        message=str(raw.message),
        nfev=int(getattr(raw, "nfev", 0)),
        strategy="differential_evolution",
        raw_result=raw,
    )
    final = pre_polish
    polish_result = None
    if method_spec.get("polish_with"):
        polish_spec = dict(method_spec)
        polish_spec["max_nfev"] = method_spec.get("polish_max_nfev")
        objective_context.update({"start_index": None, "phase": "polish"})
        polish_result = _run_local_fit(method_spec["polish_with"], objective, raw.x, lower, upper, polish_spec, options)
        if _fit_result_cost(polish_result) <= _fit_result_cost(pre_polish) + 1e-12:
            final = polish_result
    return _FitStrategyOptimizerResult(
        x=final.x,
        cost=_fit_result_cost(final),
        success=getattr(final, "success", None),
        message=getattr(final, "message", ""),
        nfev=int(getattr(pre_polish, "nfev", 0) + (0 if polish_result is None else getattr(polish_result, "nfev", 0))),
        strategy="differential_evolution",
        optimizer=None,
        raw_result=raw,
        children=[] if polish_result is None else [polish_result],
        best_start_index=None,
        pre_polish_result=pre_polish,
        polish_result=polish_result,
    )


def _run_local_fit(optimizer, objective, x0, lower, upper, method_spec, options, max_nfev=None):
    if x0.size == 0:
        return _run_noop_fit(objective, x0, strategy=optimizer)
    if optimizer == "least_squares":
        return _run_least_squares_fit(objective, x0, lower, upper, method_spec, options, max_nfev=max_nfev)
    if optimizer == "nelder_mead":
        return _run_nelder_mead_fit(objective, x0, lower, upper, method_spec, options, max_nfev=max_nfev)
    raise ValueError("Local optimizer must be 'least_squares' or 'nelder_mead'.")


def _run_least_squares_fit(objective, x0, lower, upper, method_spec, options, max_nfev=None):
    from scipy.optimize import least_squares

    raw = least_squares(
        objective,
        x0,
        bounds=(lower, upper),
        xtol=method_spec.get("xtol", options.get("xtol", 1e-8)),
        ftol=method_spec.get("ftol", options.get("ftol", 1e-8)),
        gtol=method_spec.get("gtol", options.get("gtol", 1e-8)),
        max_nfev=max_nfev if max_nfev is not None else method_spec.get("max_nfev", options.get("max_nfev")),
    )
    return _FitStrategyOptimizerResult(
        x=raw.x,
        cost=float(raw.cost),
        success=bool(raw.success),
        message=str(raw.message),
        nfev=int(getattr(raw, "nfev", 0)),
        strategy="least_squares",
        optimizer="least_squares",
        raw_result=raw,
    )


def _run_nelder_mead_fit(objective, x0, lower, upper, method_spec, options, max_nfev=None):
    from scipy.optimize import minimize

    def scalar_objective(x):
        clipped = np.clip(np.asarray(x, dtype=float), lower, upper)
        residuals = objective(clipped)
        return float(0.5 * np.dot(residuals, residuals))

    limit = max_nfev if max_nfev is not None else method_spec.get("max_nfev", options.get("max_nfev"))
    minimize_options = {}
    if limit is not None:
        minimize_options["maxfev"] = int(limit)
        minimize_options["maxiter"] = int(limit)
    raw = minimize(
        scalar_objective,
        np.asarray(x0, dtype=float),
        method="Nelder-Mead",
        bounds=list(zip(lower, upper)),
        options=minimize_options,
    )
    x = np.clip(np.asarray(raw.x, dtype=float), lower, upper)
    return _FitStrategyOptimizerResult(
        x=x,
        cost=float(raw.fun),
        success=bool(raw.success),
        message=str(raw.message),
        nfev=int(getattr(raw, "nfev", 0)),
        strategy="nelder_mead",
        optimizer="nelder_mead",
        raw_result=raw,
    )


def _run_noop_fit(objective, x0, strategy="least_squares"):
    residuals = objective(x0)
    cost = float(0.5 * np.dot(residuals, residuals))
    return _FitStrategyOptimizerResult(
        x=x0,
        cost=cost,
        success=True,
        message="No fitted parameters; evaluated residual model once.",
        nfev=1,
        strategy=strategy,
        optimizer=strategy,
    )


def _fit_result_cost(result, default=np.inf):
    if result is None:
        return default
    cost = getattr(result, "cost", default)
    if cost is None:
        return default
    return float(cost)


def _fit_cv_electrokitty(input_obj, mechanism, params, fit, options, initial_result):
    if fit not in (None, "all"):
        raise ValueError("method='electrokitty' does not honor arbitrary fit specs; use options['electrokitty'] flags instead.")

    input_params = _prepare_simulation_params(
        input_obj,
        params,
        options,
        mechanism=mechanism,
        expand_parameter_model=False,
    )
    params = _prepare_simulation_params(input_obj, input_params, options, mechanism=mechanism)
    initial_params = deepcopy(params)
    mechanism_spec = compile_mechanism(mechanism, params)
    ElectroKitty = _import_electrokitty()
    backend_mechanism = _electrokitty_backend_mechanism(mechanism_spec)
    ek = ElectroKitty(backend_mechanism)
    species_order = _electrokitty_runtime_species_order(ek)
    adapter = _electrokitty_parameters(params, mechanism_spec, species_order=species_order)
    backend_input, quiet_info = _backend_input_with_quiet_time(input_obj, options)
    ek.set_data(
        np.asarray(backend_input.E, dtype=float),
        np.asarray(backend_input.i, dtype=float),
        np.asarray(backend_input.t, dtype=float),
    )
    ek.create_simulation(
        adapter["kin"],
        adapter["cell_const"],
        adapter["diffusion_const"],
        adapter["isotherm"],
        adapter["spatial_info"],
        adapter["species_information"],
        kinetic_model=options.get("kinetic model", options.get("kinetic_model", "BV")),
    )
    ek_options = {
        "fit_kin": True,
        "fit_Cdl": False,
        "fit_Ru": False,
        "fit_gamamax": False,
        "fit_A": False,
        "fit_iso": False,
        "algorithm": "Nelder-Mead",
    }
    ek_options.update(options.get("electrokitty", {}) or {})
    ek.fit_to_data(**ek_options)

    E_generated = np.asarray(getattr(ek, "E_Corr", backend_input.E), dtype=float)
    backend_current = np.asarray(getattr(ek, "current", backend_input.i), dtype=float)
    t = np.asarray(getattr(ek, "t", backend_input.t), dtype=float)[: len(E_generated)]
    E_generated, backend_current, t = _trim_backend_quiet_time_output(
        E_generated,
        backend_current,
        t,
        quiet_info,
    )
    current_sign = _resolve_current_sign(backend_current, input_obj.i, options)
    display_current = current_sign * backend_current
    measured_current = np.asarray(input_obj.i, dtype=float)
    if len(display_current) != len(measured_current):
        raise ValueError(
            "ElectroKitty native fit returned a different number of CV points after quiet-time trimming."
        )
    data = pd.DataFrame(
        {
            "Potential": E_generated,
            "Current": display_current,
            "Time": t,
            "Backend Current": backend_current,
        }
    )
    data["Residual"] = data["Current"].to_numpy(dtype=float) - measured_current
    final_result = SimulatedCV(
        data=data,
        params=params,
        input_params=input_params,
        mechanism=mechanism_spec,
        input=input_obj,
        backend_result=ek,
        current_sign=current_sign,
        summary={
            "method": "electrokitty",
            "backend": "electrokitty",
            "mechanism": mechanism_spec.mechanism,
            "preset": mechanism_spec.preset,
            "current_sign": current_sign,
            "current sign": current_sign,
            "quiet_time": quiet_info["quiet_time"],
            "quiet_time_applied": quiet_info["applied"],
            "incubation_time": float(
                (params.get("_parameter_model", {}).get("incubation", {}) or {}).get(
                    "time",
                    _input_incubation_time(input_obj),
                )
            ),
            "incubation_applied": bool(
                (params.get("_parameter_model", {}).get("incubation", {}) or {}).get("applied", False)
            ),
            "parameter_model": deepcopy(params.get("_parameter_model", {})),
            "parameter model": deepcopy(params.get("_parameter_model", {})),
        },
    )
    fit_metrics = _fit_quality_metrics(
        final_result,
        data["Residual"].to_numpy(dtype=float),
        n_parameters=0,
        residual_normalization=_normalize_residual_normalization(options),
    )
    final_result.summary.update(fit_metrics)
    fit_summary = {
        "method": "electrokitty",
        "backend": "electrokitty",
        "fit_score": getattr(ek, "fit_score", None),
        "n_parameters": 0,
        "current_sign": current_sign,
        "current sign": current_sign,
        **fit_metrics,
    }
    fit_result = SimulationFitResult(
        best_params=input_params,
        fit_spec={"method": "electrokitty", "options": ek_options},
        method="electrokitty",
        backend="electrokitty",
        optimizer_result=getattr(ek, "optimizer", None),
        initial_result=initial_result,
        simulation_result=final_result,
        residuals=data["Residual"].to_numpy(dtype=float),
        corrections={},
        summary=fit_summary,
        backend_result=ek,
        measured_current=measured_current,
    )
    _maybe_print_fit_statistics(fit_summary, {}, options)
    _maybe_print_fit_params(initial_params, final_result.params, options)
    if bool(options.get("plot", True)):
        fit_result.plot(options.get("plot options", options))
    return fit_result


class _ElectroKittyBackend:
    name = "electrokitty"

    def simulate(self, input, mechanism_spec, params, options):
        measured_current = input.i if input.i is not None else np.zeros(len(input.E), dtype=float)

        ElectroKitty = _import_electrokitty()
        backend_mechanism = _electrokitty_backend_mechanism(mechanism_spec)
        ek = ElectroKitty(backend_mechanism)
        species_order = _electrokitty_runtime_species_order(ek)
        adapter = _electrokitty_parameters(params, mechanism_spec, species_order=species_order)
        ek.set_data(
            np.asarray(input.E, dtype=float),
            np.asarray(measured_current, dtype=float),
            np.asarray(input.t, dtype=float),
        )
        ek.create_simulation(
            adapter["kin"],
            adapter["cell_const"],
            adapter["diffusion_const"],
            adapter["isotherm"],
            adapter["spatial_info"],
            adapter["species_information"],
            kinetic_model=options.get("kinetic model", options.get("kinetic_model", "BV")),
        )
        E_generated, backend_current, t = ek.simulate()
        return E_generated, backend_current, t, ek


def _resolve_fit_inputs(input_or_result, mechanism, params, backend, options):
    dataset = _resolve_fit_dataset(input_or_result, mechanism, params, backend, options, index=0)
    return (
        dataset["input"],
        dataset["mechanism"],
        dataset["params"],
        dataset["backend"],
        dataset["initial_result"],
    )


def _resolve_fit_dataset(input_or_result, mechanism, params, backend, options, *, index=0):
    if isinstance(input_or_result, SimulatedCV):
        result = input_or_result
        input_obj = result.input
        mechanism_value = result.mechanism
        params_value = deepcopy(result.input_params)
        backend_value = result.summary.get("backend", backend)
        if mechanism is not None:
            mechanism_value = mechanism
        if params is not None:
            params_value = deepcopy(params)
        return _fit_dataset_record(index, input_obj, mechanism_value, params_value, backend_value, result)

    if isinstance(input_or_result, _EcatCV):
        if mechanism is None or params is None:
            raise ValueError("fit_cv requires mechanism and params unless input_or_result is a SimulatedCV.")
        input_obj = cv_data(input_or_result, _fit_cv_data_options(options))
        return _fit_dataset_record(index, input_obj, mechanism, deepcopy(params), backend, None)

    if mechanism is None or params is None:
        raise ValueError("fit_cv requires mechanism and params unless input_or_result is a SimulatedCV.")
    return _fit_dataset_record(index, input_or_result, mechanism, deepcopy(params), backend, None)


def _fit_dataset_record(index, input_obj, mechanism, params, backend, initial_result):
    label = _simulation_source_label(getattr(input_obj, "source", None))
    if not label:
        label = _metadata_value(input_obj, "name", "") or f"CV {int(index) + 1}"
    return {
        "index": int(index),
        "label": label,
        "input": input_obj,
        "mechanism": mechanism,
        "params": params,
        "backend": backend,
        "initial_result": initial_result,
        "source": getattr(input_obj, "source", None),
        "scan_rate": _metadata_value(input_obj, "scan_rate", _scan_rate_from_input(input_obj)),
        "metadata": dict(getattr(input_obj, "metadata", {}) or {}),
    }


def _apply_group_fit_multiplot_labels(datasets, options):
    sources = [dataset.get("source") for dataset in datasets]
    if len(sources) < 2 or not all(callable(getattr(source, "txt_stats", None)) for source in sources):
        return
    try:
        from .plotting import (
            _deduplicate_multiplot_labels,
            _resolve_multiplot_labels_title_subtitle,
            _warn_duplicate_multiplot_labels,
        )

        labels, *_ = _resolve_multiplot_labels_title_subtitle(sources, dict(options or {}))
        labels = _deduplicate_multiplot_labels(sources, labels, dict(options or {}))
        _warn_duplicate_multiplot_labels(labels, dict(options or {}))
    except Exception:
        return
    if len(labels) != len(datasets):
        return
    for dataset, label in zip(datasets, labels):
        if label not in (None, ""):
            dataset["label"] = str(label)


def _fit_cv_data_options(options):
    cv_options = options.get("cv data", options.get("cv_data", {}))
    if cv_options in (None, False):
        return {}
    return dict(cv_options)


def _normalize_fit_spec(fit, params, input_obj):
    base_params = _normalize_simulation_species_params(deepcopy(params))
    base_params = _normalize_simulation_activity_params(base_params)
    fixed = _fit_mapping(fit, "fixed")
    fixed_paths = _expand_fit_mapping(fixed, base_params)
    for path, value in fixed_paths.items():
        _set_param_path(base_params, path, value)

    vary = _fit_vary_paths(fit, base_params)
    vary_paths = _expand_fit_paths(vary, base_params)
    flat_vary_paths = _flatten_fit_targets(vary_paths)
    if len(set(flat_vary_paths)) != len(flat_vary_paths):
        raise ValueError("fit vary parameters contain duplicates.")

    conflict = set(flat_vary_paths).intersection(fixed_paths)
    if conflict:
        raise ValueError(f"Parameters cannot be both fixed and varied: {sorted(conflict)!r}")

    init_map = _fit_mapping(fit, "init")
    bounds_map = _fit_mapping(fit, "bounds")
    transform_map = _fit_mapping(fit, "transform")
    init_paths = _expand_fit_mapping(init_map, base_params)
    bounds_paths = _expand_fit_mapping(bounds_map, base_params)
    transform_paths = _expand_fit_mapping(transform_map, base_params)
    init_auto = _fit_field_is_auto(fit, "init")
    bounds_auto = _fit_field_is_auto(fit, "bounds")
    transform_auto = _fit_field_is_auto(fit, "transform")

    entries = {}
    for target in vary_paths:
        target_paths = _fit_target_paths(target)
        primary_path = target_paths[0]
        init_value, has_init = _fit_mapping_value_for_target(target_paths, init_paths)
        bounds_value, has_bounds = _fit_mapping_value_for_target(target_paths, bounds_paths)
        transform_value, has_transform = _fit_mapping_value_for_target(target_paths, transform_paths)
        init = (
            _auto_fit_init(primary_path, _get_param_path(base_params, primary_path), input_obj)
            if init_auto or not has_init
            else _coerce_param_path_value(primary_path, init_value)
        )
        bounds = (
            _auto_fit_bounds(primary_path, init, input_obj)
            if bounds_auto or not has_bounds
            else tuple(bounds_value)
        )
        transform = (
            _auto_fit_transform(primary_path, bounds)
            if transform_auto or not has_transform
            else str(transform_value).lower()
        )
        init = _fit_init_within_bounds(init, bounds, allow_clip=init_auto or not has_init)
        if transform not in {"linear", "log10"}:
            raise ValueError("fit transform must be 'auto', 'linear', or 'log10'.")
        if transform == "log10" and (float(bounds[0]) <= 0 or float(bounds[1]) <= 0 or float(init) <= 0):
            raise ValueError("log10 transform requires positive init and bounds.")
        entries[target] = {
            "init": float(init),
            "bounds": (float(bounds[0]), float(bounds[1])),
            "transform": transform,
        }

    return {
        "vary": vary_paths,
        "fixed": fixed_paths,
        "entries": entries,
        "base_params": base_params,
    }


def _fit_vary_paths(fit, params):
    if fit is None or fit == "all":
        return _fit_safe_numeric_paths(params)
    if isinstance(fit, (list, tuple)):
        if _is_path_tuple(fit):
            return [fit]
        return list(fit)
    if isinstance(fit, dict):
        vary = fit.get("vary", fit.get("varied", None))
        if vary is None:
            return _fit_safe_numeric_paths(params)
        if vary == "all":
            return _fit_safe_numeric_paths(params)
        if isinstance(vary, (str, tuple)) and not isinstance(vary, list):
            return [vary]
        return list(vary)
    if isinstance(fit, str):
        if fit == "all":
            return _fit_safe_numeric_paths(params)
        return [fit]
    raise ValueError("fit must be None, 'all', a list of parameters, or a fit specification dict.")


def _fit_mapping(fit, key):
    if not isinstance(fit, dict):
        return {}
    value = fit.get(key, "auto")
    return {} if value in (None, "auto") else dict(value)


def _fit_field_is_auto(fit, key):
    if not isinstance(fit, dict):
        return True
    return fit.get(key, "auto") in (None, "auto")


def _resolve_fit_path(path, params=None):
    paths = _resolve_fit_paths(path, params)
    if len(paths) != 1:
        raise ValueError(f"Fit parameter alias {path!r} resolved to {len(paths)} parameters; use a wildcard-aware field or tuple paths.")
    return paths[0]


def _expand_fit_paths(paths, params=None):
    out = []
    for path in paths:
        out.extend(_resolve_fit_targets(path, params))
    return out


def _expand_fit_mapping(mapping, params=None):
    out = {}
    for path, value in mapping.items():
        for resolved in _resolve_fit_paths(path, params):
            out[resolved] = value
    return out


def _resolve_fit_targets(path, params=None):
    paths = _resolve_fit_paths(path, params)
    if _is_tied_bare_alias(path) and len(paths) > 1:
        return [tuple(paths)]
    return paths


def _is_tied_bare_alias(path):
    if not isinstance(path, str):
        return False
    lower = _canonical_fit_alias(path)
    return lower in {"e0", "k0", "alpha", "d", "kf", "kb", "k"}


def _resolve_fit_paths(path, params=None):
    if _is_path_tuple(path):
        if path and path[0] == "species":
            raise ValueError("Use concentrations.* fit paths; species is input sugar only.")
        return [tuple(path)]
    if not isinstance(path, str):
        raise ValueError(f"Invalid fit parameter path {path!r}.")
    text = path.strip()
    if "." in text:
        if text.lower() == "diffusion.*":
            return _diffusion_paths(params)
        out = []
        for part in text.split("."):
            out.append(int(part) if part.isdigit() else part)
        if out and out[0] == "species":
            raise ValueError("Use concentrations.* fit paths; species is input sugar only.")
        return [tuple(out)]
    lower = _canonical_fit_alias(text)
    aliases = {
        "ru": ("cell", "Ru"),
        "cdl": ("cell", "Cdl"),
        "area": ("cell", "A"),
        "a": ("cell", "A"),
    }
    if lower in aliases:
        return [aliases[lower]]
    if lower in {"e0", "k0", "alpha"}:
        return _single_kinetic_paths(lower, text, params)
    if lower in {"kf", "kb", "k"}:
        return _single_rate_paths(lower, text, params)
    if lower == "d":
        return _single_diffusion_paths(text, params)
    if lower in {"e0_*", "k0_*", "alpha_*"}:
        key = {"e0_*": "E0", "k0_*": "k0", "alpha_*": "alpha"}[lower]
        return _kinetic_paths(key, params)
    if lower in {"d_*", "diffusion_*"}:
        return _diffusion_paths(params)
    if lower.startswith("d") and lower != "diffusion" and len(lower) > 1 and lower[1] != "_":
        return _indexed_or_named_diffusion_path(lower[1:], text, params)
    if "_" in lower:
        name, index_text = lower.rsplit("_", 1)
        lower_key = name.lower()
        if lower_key in {"e0", "k0", "alpha"} and index_text == "*":
            return _kinetic_paths({"e0": "E0", "k0": "k0", "alpha": "alpha"}[lower_key], params)
        if lower_key in {"kf", "kb", "k"} and index_text == "*":
            return _rate_paths({"kf": "kf", "kb": "kb", "k": "k"}[lower_key], params)
        if lower_key in {"d", "diffusion"}:
            return _indexed_or_named_diffusion_path(index_text, text, params)
        if index_text.isdigit():
            index = int(index_text)
            if lower_key in {"e0", "k0", "alpha"}:
                return [("kinetics", index, {"e0": "E0", "k0": "k0", "alpha": "alpha"}[lower_key])]
            if lower_key in {"kf", "kb", "k"}:
                if params and "reactions" in params and len(params.get("reactions", []) or []) > index:
                    return [("reactions", index, {"kf": "kf", "kb": "kb", "k": "k"}[lower_key])]
                return [("kinetics", index, {"kf": "kf", "kb": "kb", "k": "k"}[lower_key])]
    raise ValueError(f"Unknown fit parameter alias {path!r}. Use a tuple path for nested parameters.")


def _canonical_fit_alias(value):
    text = str(value).strip().replace("α", "alpha").replace("Α", "alpha")
    text = text.lower().replace("-", "_")
    text = "_".join(text.replace("_", " ").split())
    if text == "e0" or text.startswith("e0_"):
        return text
    if text == "e":
        return "e0"
    if text.startswith("e_"):
        return "e0_" + text[2:]
    if len(text) > 1 and text[0] == "e" and (text[1:].isdigit() or text[1:] == "*"):
        return "e0_" + text[1:]
    return text


def _single_kinetic_paths(lower_key, alias, params):
    key = {"e0": "E0", "k0": "k0", "alpha": "alpha"}[lower_key]
    paths = _kinetic_paths(key, params)
    if len(paths) == 1:
        return paths
    if not paths:
        raise ValueError(f"Unknown fit parameter alias {alias!r}; no kinetics entries contain {key!r}.")
    _require_same_fit_values(alias, paths, params, f"{alias}_*")
    return paths


def _kinetic_paths(key, params):
    paths = []
    for index, entry in enumerate((params or {}).get("kinetics", []) or []):
        if isinstance(entry, dict) and key in entry:
            paths.append(("kinetics", index, key))
    return paths


def _rate_paths(key, params):
    group = "reactions" if (params or {}).get("reactions") else "kinetics"
    paths = []
    for index, entry in enumerate((params or {}).get(group, []) or []):
        if isinstance(entry, dict) and key in entry:
            paths.append((group, index, key))
    return paths


def _single_rate_paths(lower_key, alias, params):
    paths = _rate_paths({"kf": "kf", "kb": "kb", "k": "k"}[lower_key], params)
    if len(paths) == 1:
        return paths
    if not paths:
        raise ValueError(f"Unknown fit parameter alias {alias!r}; no rate entries contain {lower_key!r}.")
    _require_same_fit_values(alias, paths, params, f"{alias}_*")
    return paths


def _single_diffusion_paths(alias, params):
    diffusion = (params or {}).get("diffusion", {}) or {}
    if "D" in diffusion:
        return [("diffusion", "D")]
    if len(diffusion) == 1:
        return [("diffusion", next(iter(diffusion)))]
    paths = _diffusion_paths(params)
    _require_same_fit_values(alias, paths, params, "D_*")
    return paths


def _diffusion_paths(params):
    return [("diffusion", key) for key in ((params or {}).get("diffusion", {}) or {})]


def _indexed_or_named_diffusion_path(index_text, alias, params):
    index_text = str(index_text).strip().lower()
    diffusion = (params or {}).get("diffusion", {}) or {}
    keys = list(diffusion)
    if index_text == "*":
        return _diffusion_paths(params)
    if index_text.isdigit():
        index = int(index_text)
        if index >= len(keys):
            raise ValueError(f"Fit parameter alias {alias!r} is out of range for diffusion parameters.")
        return [("diffusion", keys[index])]
    for key in keys:
        if str(key).lower() == index_text.lower():
            return [("diffusion", key)]
    raise ValueError(f"Unknown diffusion fit parameter alias {alias!r}.")


def _require_same_fit_values(alias, paths, params, wildcard_hint):
    if not paths:
        return
    values = [_get_param_path(params, path) for path in paths]
    first = values[0]
    for value in values[1:]:
        if not _fit_values_equal(first, value):
            raise ValueError(
                f"Fit parameter alias {alias!r} is ambiguous because matching parameters do not all "
                f"have the same value. Use {wildcard_hint!r} to fit them independently or indexed "
                "aliases/tuple paths to choose specific parameters."
            )


def _fit_values_equal(a, b):
    if _is_numeric(a) and _is_numeric(b):
        return bool(np.isclose(float(a), float(b), rtol=1e-12, atol=0.0))
    return a == b


def _fit_target_paths(target):
    if _is_tied_fit_target(target):
        return list(target)
    return [target]


def _is_tied_fit_target(target):
    return isinstance(target, tuple) and bool(target) and all(_is_path_tuple(path) for path in target)


def _flatten_fit_targets(targets):
    out = []
    for target in targets:
        out.extend(_fit_target_paths(target))
    return out


def _fit_mapping_value_for_target(paths, mapping):
    values = [mapping[path] for path in paths if path in mapping]
    if not values:
        return None, False
    first = values[0]
    for value in values[1:]:
        if value != first:
            raise ValueError("Tied fit parameters cannot use conflicting init, bounds, or transform values.")
    return first, True


def _is_path_tuple(value):
    return isinstance(value, tuple)


def _original_path_key(resolved_path, mapping, params):
    for key in mapping:
        if _resolve_fit_path(key, params) == resolved_path:
            return key
    return resolved_path


def _fit_safe_numeric_paths(params):
    out = []
    for group in ("kinetics", "reactions"):
        for i, entry in enumerate(params.get(group, []) or []):
            if isinstance(entry, dict):
                for key, value in entry.items():
                    if _is_numeric(value):
                        out.append((group, i, key))
    for key in ("Ru", "Cdl", "A"):
        if _is_numeric((params.get("cell", {}) or {}).get(key)):
            out.append(("cell", key))
    for group in ("diffusion",):
        for key, value in (params.get(group, {}) or {}).items():
            if _is_numeric(value):
                out.append((group, key))
    concentrations = params.get("concentrations", {}) or {}
    if any(key in concentrations for key in ("bulk", "surface")):
        for phase in ("bulk", "surface"):
            for key, value in (concentrations.get(phase, {}) or {}).items():
                if _is_numeric(value):
                    out.append(("concentrations", phase, key))
    else:
        for key, value in concentrations.items():
            if _is_numeric(value):
                out.append(("concentrations", "bulk", key))
    activity = _activity_config(params)
    out.extend(_safe_numeric_nested_paths(activity.get("gamma", {}) or {}, ("activity", "gamma")))
    return out


def _safe_numeric_nested_paths(value, prefix):
    out = []
    if isinstance(value, dict):
        for key, nested in value.items():
            out.extend(_safe_numeric_nested_paths(nested, tuple(prefix) + (key,)))
    elif _is_numeric(value):
        out.append(tuple(prefix))
    return out


def _is_numeric(value):
    return isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool)


def _get_param_path(params, path):
    value = params
    for part in path:
        value = value[part]
    return value


def _set_param_path(params, path, value):
    target = params
    for part in path[:-1]:
        target = target[part]
    target[path[-1]] = float(_coerce_param_path_value(path, value))


def _coerce_user_param_path(path):
    if _is_path_tuple(path):
        return path
    if isinstance(path, str):
        return tuple(int(part) if part.isdigit() else part for part in path.split("."))
    raise ValueError(f"Invalid parameter path {path!r}.")


def _deep_merge_dicts(base, updates):
    result = deepcopy(base)
    for key, value in dict(updates).items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge_dicts(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def _coerce_param_path_value(path, value):
    if tuple(path)[-1] == "k0" and isinstance(value, str):
        return _resolve_k0_preset(value)
    if str(tuple(path)[-1]).lower() == "k" and isinstance(value, str):
        parsed, _unit = _parse_equilibrium_constant_string(value)
        return parsed
    return value


def _auto_fit_init(path, value, input_obj):
    key = str(path[-1]).lower()
    value = float(value)
    if key == "k0":
        return _DEFAULT_FAST_K0_INIT
    if key in {"kf", "kb", "k"} and value <= 0:
        return 1.0
    return value


def _fit_init_within_bounds(init, bounds, allow_clip=False):
    init = float(init)
    lower, upper = float(bounds[0]), float(bounds[1])
    if lower > upper:
        raise ValueError("fit bounds lower limit cannot exceed upper limit.")
    if lower <= init <= upper:
        return init
    if allow_clip:
        return float(np.clip(init, lower, upper))
    raise ValueError("fit init must be within fit bounds.")


def _auto_fit_bounds(path, init, input_obj):
    key = str(path[-1]).lower()
    init = float(init)
    if key == "alpha":
        return (0.0, 1.0)
    if key == "e0":
        E = np.asarray(input_obj.E, dtype=float)
        return (float(np.nanmin(E)), float(np.nanmax(E)))
    if key == "k0":
        return (_MIN_POSITIVE_FIT_BOUND, 1e12)
    if key in {"kf", "kb", "k"}:
        return (_MIN_POSITIVE_FIT_BOUND, 1e12)
    if (
        key in {"a", "cdl", "ru", "total", "k", "k_exchange", "koff"}
        or path[0] in {"diffusion", "concentrations"}
        or (path[0] == "activity" and "gamma" in path)
    ):
        if init > 0:
            return (max(init / 100.0, 1e-30), init * 100.0)
        return (0.0, 1000.0)
    return (init - max(abs(init), 1.0), init + max(abs(init), 1.0))


def _auto_fit_transform(path, bounds):
    key = str(path[-1]).lower()
    if (
        key in {"k0", "kf", "kb", "k", "a", "cdl", "ru", "total", "k_exchange", "koff"}
        or path[0] in {"diffusion", "concentrations"}
        or (path[0] == "activity" and "gamma" in path)
    ):
        if float(bounds[0]) > 0 and float(bounds[1]) > 0:
            return "log10"
    return "linear"


def _external_to_optimizer(value, transform):
    value = float(value)
    return float(np.log10(value)) if transform == "log10" else value


def _optimizer_to_external(value, transform):
    value = float(value)
    return float(10 ** value) if transform == "log10" else value


def _params_from_fit_vector(base_params, fit_spec, x):
    params = deepcopy(base_params)
    for value, target in zip(x, fit_spec["vary"]):
        entry = fit_spec["entries"][target]
        external_value = _optimizer_to_external(value, entry["transform"])
        for path in _fit_target_paths(target):
            _set_param_path(params, path, external_value)
    return params


def _normalize_residual_mode(options):
    residual = _canonical_token(options.get("residual", "direct"))
    residual = _correction_mode_alias(residual)
    if residual not in {"direct", "offset", "scale", "scale_linear_baseline"}:
        raise ValueError("residual must be 'direct', 'offset', 'scale', or 'scale linear baseline'.")
    return residual


def _normalize_post_correction_mode(options):
    correction = options.get("post correction", options.get("correction", options.get("correction mode")))
    if correction in (None, False, "none", "None", "off", "Off"):
        return None
    correction = _canonical_token(correction)
    correction = _correction_mode_alias(correction)
    if correction not in {"offset", "scale", "scale_linear_baseline"}:
        raise ValueError("post correction must be 'offset', 'scale', 'scale linear baseline', or None.")
    return correction


def _correction_mode_alias(value):
    aliases = {
        "vertical": "offset",
        "vertical_shift": "offset",
        "vertical_translation": "offset",
        "constant": "offset",
        "constant_offset": "offset",
        "constant_baseline": "offset",
        "intercept": "offset",
        "baseline_intercept": "offset",
    }
    return aliases.get(value, value)


def _normalize_residual_normalization(options):
    value = options.get("residual normalization", "max_abs_measured")
    if value in (None, False, "none", "None", "off", "Off"):
        return None
    value = _canonical_token(value)
    aliases = {
        "max": "max_abs_measured",
        "max_abs": "max_abs_measured",
        "max_current": "max_abs_measured",
        "measured": "max_abs_measured",
    }
    value = aliases.get(value, value)
    if value not in {"max_abs_measured", "rms_measured"}:
        raise ValueError("residual normalization must be 'max abs measured', 'rms measured', or None.")
    return value


def _optimizer_residuals(residuals, sim_result, normalization):
    residuals = np.asarray(residuals, dtype=float)
    if normalization is None:
        return residuals
    measured = _measured_current_for_result(sim_result)
    if normalization == "max_abs_measured":
        scale = float(np.nanmax(np.abs(measured))) if len(measured) else 0.0
    elif normalization == "rms_measured":
        scale = float(np.sqrt(np.nanmean(measured ** 2))) if len(measured) else 0.0
    else:
        raise ValueError("Unknown residual normalization.")
    if not np.isfinite(scale) or scale <= 0:
        return residuals
    return residuals / scale


def _fit_residual_components(sim_result, residual_mode):
    simulated = sim_result.data["Current"].to_numpy(dtype=float)
    measured = _measured_current_for_result(sim_result)
    E = sim_result.data["Potential"].to_numpy(dtype=float)

    if residual_mode == "direct":
        corrected = simulated
        corrections = {}
    elif residual_mode == "offset":
        offset = float(np.nanmean(measured - simulated))
        corrected = simulated + offset
        corrections = {"baseline_intercept": offset}
    elif residual_mode == "scale":
        denom = float(np.dot(simulated, simulated))
        scale = 0.0 if denom == 0 else float(np.dot(measured, simulated) / denom)
        corrected = scale * simulated
        corrections = {"current_scale": scale}
    else:
        X = np.column_stack([simulated, np.ones_like(simulated), E - np.nanmean(E)])
        coeffs, *_ = np.linalg.lstsq(X, measured, rcond=None)
        corrected = X @ coeffs
        corrections = {
            "current_scale": float(coeffs[0]),
            "baseline_intercept": float(coeffs[1]),
            "baseline_slope": float(coeffs[2]),
        }

    residuals = corrected - measured
    return residuals, corrections, corrected


def _measured_current_for_result(sim_result, *, required=True):
    measured = None
    input_obj = getattr(sim_result, "input", None)
    if input_obj is not None and getattr(input_obj, "i", None) is not None:
        measured = np.asarray(input_obj.i, dtype=float)
    elif sim_result is not None and getattr(sim_result, "data", None) is not None and "Measured Current" in sim_result.data:
        measured = sim_result.data["Measured Current"].to_numpy(dtype=float)

    if measured is None:
        if required:
            raise ValueError("fit_cv requires measured current in the simulation input.")
        return None

    data = getattr(sim_result, "data", None)
    if data is not None and len(measured) != len(data):
        raise ValueError("Measured current length must match the simulated CV length.")
    return measured


def _apply_fit_current(sim_result, corrected_current, residuals, corrections, residual_mode, post_correction_mode=None):
    sim_result.data["Current"] = np.asarray(corrected_current, dtype=float)
    sim_result.data["Residual"] = np.asarray(residuals, dtype=float)
    sim_result.data.attrs["residual"] = residual_mode
    sim_result.data.attrs["post_correction"] = post_correction_mode
    sim_result.data.attrs["corrections"] = corrections
    sim_result.summary["residual"] = residual_mode
    sim_result.summary["post_correction"] = post_correction_mode
    sim_result.summary["corrections"] = corrections


def _fit_quality_metrics(sim_result, residuals, *, n_parameters=0, residual_normalization=None):
    residuals = np.asarray(residuals, dtype=float)
    n_points = int(len(residuals))
    n_parameters = int(n_parameters or 0)
    dof = max(n_points - n_parameters, 1)
    abs_residuals = np.abs(residuals)
    physical_cost = float(0.5 * np.dot(residuals, residuals))
    normalized = _optimizer_residuals(residuals, sim_result, residual_normalization)
    normalized_cost = float(0.5 * np.dot(normalized, normalized))

    metrics = {
        "n_points": n_points,
        "degrees_of_freedom": dof,
        "residual_norm": float(np.linalg.norm(residuals)),
        "rmse": float(np.sqrt(np.nanmean(residuals ** 2))) if n_points else np.nan,
        "mae": float(np.nanmean(abs_residuals)) if n_points else np.nan,
        "max_abs_residual": float(np.nanmax(abs_residuals)) if n_points else np.nan,
        "physical_cost": physical_cost,
        "physical_cost_per_point": physical_cost / n_points if n_points else np.nan,
        "reduced_physical_cost": physical_cost / dof,
        "normalized_residual_norm": float(np.linalg.norm(normalized)),
        "normalized_rmse": float(np.sqrt(np.nanmean(normalized ** 2))) if n_points else np.nan,
        "normalized_max_abs_residual": float(np.nanmax(np.abs(normalized))) if n_points else np.nan,
        "normalized_cost": normalized_cost,
        "normalized_cost_per_point": normalized_cost / n_points if n_points else np.nan,
        "reduced_normalized_cost": normalized_cost / dof,
    }
    return metrics


def _record_fit_progress_row(rows, eval_count, residuals, corrections, params, fit_spec, start_index=None, phase=None):
    residuals = np.asarray(residuals, dtype=float)
    cost = float(0.5 * np.dot(residuals, residuals))
    previous_cost = rows[-1]["Cost"] if rows else None
    delta_cost = "" if previous_cost in (None, "") else cost - float(previous_cost)
    row = {
        "Eval": int(eval_count),
        "Cost": cost,
        "ΔCost": delta_cost,
        "Residual Norm": float(np.linalg.norm(residuals)),
        "Max |Residual|": float(np.nanmax(np.abs(residuals))) if residuals.size else 0.0,
        "Current Scale": _format_progress_value((corrections or {}).get("current_scale", "")),
        "Baseline Intercept": _format_progress_value((corrections or {}).get("baseline_intercept", "")),
        "Baseline Slope": _format_progress_value((corrections or {}).get("baseline_slope", "")),
        "Changed Params": _format_progress_params(params, fit_spec),
    }
    if start_index is not None:
        row = {"Start": int(start_index), **row}
    if phase is not None:
        row = {"Phase": phase, **row}
    rows.append(row)


def _format_progress_value(value):
    if value == "":
        return ""
    return _format_param_value(value)


def _format_progress_params(params, fit_spec):
    if not isinstance(fit_spec, dict):
        return ""
    pieces = []
    for target in fit_spec.get("vary", []) or []:
        if _is_per_cv_fit_target(target):
            continue
        paths = _fit_target_paths(target)
        if not paths:
            continue
        primary_path = paths[0]
        value = _get_param_path(params, primary_path)
        pieces.append(f"{_fit_target_symbol(target)}={_format_param_value(value)}")
    return "; ".join(pieces)


class _FitProgressReporter:
    def __init__(self, options):
        self.count = 0
        self.callback = options.get("progress callback", options.get("progress_callback"))
        self._bar = None
        self._bar_kind = None

        progress = options.get("progress", options.get("fit progress", options.get("fit_progress", False)))
        self.enabled = self._progress_enabled(progress)
        if not self.enabled:
            return

        self.total = options.get("max_nfev")
        self.label = options.get("progress label", options.get("progress_label", "Fitting CV"))
        self.leave = bool(options.get("progress leave", options.get("progress_leave", True)))
        self._create_bar(progress)

    def update(self, *, residuals=None, params=None):
        self.count += 1
        if callable(self.callback):
            payload = {
                "n_evaluations": self.count,
                "cost": None if residuals is None else float(0.5 * np.dot(residuals, residuals)),
                "params": params,
            }
            self.callback(payload)
        if self._bar is None:
            return
        cost = None if residuals is None else float(0.5 * np.dot(residuals, residuals))
        if self._bar_kind == "tqdm":
            self._bar.update(1)
            if cost is not None:
                self._bar.set_postfix(cost=f"{cost:.3g}")
        elif self._bar_kind == "notebook":
            self._bar.update(self.count, cost=cost)

    def close(self):
        if self._bar_kind == "tqdm" and self._bar is not None:
            self._bar.close()
        elif self._bar_kind == "notebook" and self._bar is not None:
            self._bar.close()

    @staticmethod
    def _progress_enabled(progress):
        return progress_enabled(progress)

    def _create_bar(self, progress):
        if callable(progress):
            self.callback = progress
            return

        preference = str(progress).strip().lower() if isinstance(progress, str) else "auto"
        if preference == "notebook":
            self._create_notebook_bar()
            if self._bar is not None:
                return

        tqdm_imports = []
        if preference in {"terminal", "cli"}:
            tqdm_imports = [("tqdm", "tqdm")]
        else:
            tqdm_imports = [("tqdm.auto", "tqdm"), ("tqdm.notebook", "tqdm"), ("tqdm", "tqdm")]

        for module_name, attr in tqdm_imports:
            try:
                module = __import__(module_name, fromlist=[attr])
                tqdm = getattr(module, attr)
                self._bar = tqdm(total=self.total, desc=self.label, unit="eval", leave=self.leave)
                self._bar_kind = "tqdm"
                return
            except Exception:
                continue

        self._create_notebook_bar()

    def _create_notebook_bar(self):
        try:
            self._bar = _NotebookFitProgressDisplay(
                total=self.total,
                label=self.label,
                leave=self.leave,
            )
            self._bar_kind = "notebook"
        except Exception:
            self._bar = None
            self._bar_kind = None


class _NotebookFitProgressDisplay:
    def __init__(self, *, total=None, label="Fitting CV", leave=True):
        self._impl = NotebookProgressDisplay(
            total=total,
            label=label,
            leave=leave,
            unit="evals",
            approx_total=True,
            metric_label="cost",
        )
        self.total = self._impl.total
        self.label = label
        self.leave = leave

    def update(self, count, *, cost=None):
        self._impl.update(count, metric=cost)

    def close(self):
        self._impl.close()

    @staticmethod
    def _html(count, cost, total, label, *, elapsed=0.0, remaining=None):
        return NotebookProgressDisplay._html(
            count,
            cost,
            total,
            label,
            elapsed=elapsed,
            remaining=remaining,
            unit="evals",
            approx_total=True,
            metric_label="cost",
        )

    @staticmethod
    def _done_html(count, cost, total, label, *, elapsed=0.0):
        return NotebookProgressDisplay._done_html(
            count,
            cost,
            total,
            label,
            elapsed=elapsed,
            unit="evals",
            approx_total=True,
            metric_label="cost",
        )


def _normalize_options(options):
    if options is None:
        return {}
    normalized = {}
    for key, value in dict(options).items():
        normalized[key] = value
        if isinstance(key, str):
            spaced = _canonical_option_key(key)
            underscored = spaced.replace(" ", "_")
            normalized[spaced] = value
            normalized[underscored] = value
    return normalized


def _canonical_option_key(key):
    text = str(key).strip().lower().replace("_", " ").replace("-", " ")
    return " ".join(text.split())


def _canonical_token(value):
    text = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    while "__" in text:
        text = text.replace("__", "_")
    return text


def _truthy_option(value):
    if isinstance(value, str):
        return value.strip().lower() not in {"", "0", "false", "no", "none", "off"}
    return bool(value)


def _falsey_option(value):
    if isinstance(value, str):
        return value.strip().lower() in {"", "0", "false", "no", "none", "off"}
    return not bool(value)


def _is_auto_value(value):
    return isinstance(value, str) and value.strip().lower() == "auto"


def _ecat_pyplot():
    try:
        from . import core

        return core.plt
    except Exception:
        return plt


def _simulation_axis_name(input_obj, axis, default):
    metadata = getattr(input_obj, "metadata", {}) or {}
    return metadata.get(f"{axis}_axis_name", default)


def _simulation_axis_unit(input_obj, axis, default):
    metadata = getattr(input_obj, "metadata", {}) or {}
    return metadata.get(f"{axis}_unit", default)


def _ecat_plot_axis_scale(values, axis_name, axis_unit, selected_unit):
    try:
        from .objects import echem

        return echem.scale_axis(values, axis_name, axis_unit, selected_unit)
    except Exception:
        return 1, axis_unit


def _current_values_in_amps(values, axis_name, axis_unit):
    values = np.asarray(values, dtype=float)
    try:
        from .utils import extract_prefix_and_base, get_conversion_factor

        prefix, base = extract_prefix_and_base(axis_unit)
        if base != "A":
            return values, axis_unit
        scale = get_conversion_factor(prefix + base, "A")
    except Exception:
        return values, axis_unit
    return values * scale, "A"


def _ecat_plot_axis_label(axis_name, axis_unit):
    try:
        from .objects import echem

        return echem.format_axis_label(axis_name, axis_unit)
    except Exception:
        return f"{axis_name} ({axis_unit})" if axis_unit else str(axis_name)


def _apply_ecat_axis_style(ax, options=None):
    try:
        from . import core

        return core._apply_ecat_axis_style(ax, options)
    except Exception:
        return ax


def _cv_data_stride_info(E, options):
    E = np.asarray(E, dtype=float)
    n_points = len(E)
    stride_option = options.get("stride", "auto")

    if isinstance(stride_option, str) and stride_option.lower() == "auto":
        target_points, basis, points_per_volt = _cv_data_target_points(E, options)
        stride = max(1, int(np.ceil(n_points / target_points)))
        mode = "auto"
    else:
        if stride_option is None:
            stride_option = "auto"
            target_points, basis, points_per_volt = _cv_data_target_points(E, options)
            stride = max(1, int(np.ceil(n_points / target_points)))
            mode = "auto"
        else:
            stride = int(stride_option)
            if stride < 1:
                raise ValueError("stride must be at least 1.")
            target_points = n_points
            basis = "manual"
            points_per_volt = None
            mode = "manual"

    indices = np.arange(0, n_points, stride, dtype=int)
    if len(indices) == 0:
        indices = np.asarray([0], dtype=int)
    if indices[-1] != n_points - 1:
        indices = np.r_[indices, n_points - 1]
    if np.any(np.isfinite(E)):
        indices = np.r_[indices, int(np.nanargmin(E)), int(np.nanargmax(E))]
    indices = np.unique(indices)

    return {
        "indices": indices,
        "stride": stride,
        "mode": mode,
        "basis": basis,
        "points_per_volt": points_per_volt,
        "target_points": target_points,
    }


def _cv_data_target_points(E, options):
    requested_points = options.get("points", options.get("target points", options.get("target_points")))
    min_points = int(options.get("min points", options.get("min_points", _DEFAULT_CV_DATA_MIN_POINTS)))
    max_points = options.get("max points", options.get("max_points", _DEFAULT_CV_DATA_MAX_POINTS))
    max_points = None if max_points is None else int(max_points)

    if min_points < 1:
        raise ValueError("min points must be at least 1.")
    if max_points is not None and max_points < min_points:
        raise ValueError("max points must be greater than or equal to min points.")

    if requested_points is not None:
        target_points = int(requested_points)
        basis = "points"
        points_per_volt = None
    else:
        points_per_volt = float(
            options.get(
                "points per volt",
                options.get("points_per_volt", _DEFAULT_CV_DATA_POINTS_PER_VOLT),
            )
        )
        if points_per_volt <= 0:
            raise ValueError("points per volt must be positive.")
        span = float(np.nanmax(E) - np.nanmin(E)) if len(E) else 0.0
        target_points = int(np.ceil(points_per_volt * span))
        basis = "points_per_volt"

    if target_points < 1:
        raise ValueError("points must be at least 1.")
    target_points = max(min_points, target_points)
    if max_points is not None:
        target_points = min(max_points, target_points)
    target_points = max(1, target_points)
    return target_points, basis, points_per_volt


def estimate_cdl_from_cv_arrays(E, current, scan_rate, options=None):
    """Estimate double-layer capacitance from forward/reverse current separation near Ei."""
    options = _normalize_options(options)
    E = np.asarray(E, dtype=float)
    current = np.asarray(current, dtype=float)
    scan_rate = float(scan_rate)
    if len(E) != len(current):
        raise ValueError("Cdl estimation requires matching potential and current arrays.")
    if len(E) < 4:
        raise ValueError("Cdl estimation requires both forward and reverse CV branches.")
    if not np.isfinite(scan_rate) or scan_rate <= 0:
        raise ValueError("Cdl estimation requires a positive scan rate.")

    start = float(E[0])
    finite = np.isfinite(E) & np.isfinite(current)
    if np.count_nonzero(finite) < 4:
        raise ValueError("Cdl estimation requires finite potential and current values.")
    E = E[finite]
    current = current[finite]

    vertex_index = int(np.nanargmax(np.abs(E - start)))
    forward_E = E[: vertex_index + 1]
    forward_i = current[: vertex_index + 1]
    reverse_E = E[vertex_index:]
    reverse_i = current[vertex_index:]
    if len(forward_E) < 2 or len(reverse_E) < 2:
        raise ValueError("Cdl estimation requires both forward and reverse CV branches.")

    span = float(np.nanmax(E) - np.nanmin(E))
    width = options.get("Cdl window", options.get("cdl window", options.get("cdl_window")))
    if width is None:
        fraction = float(
            options.get(
                "Cdl window fraction",
                options.get("cdl_window_fraction", 0.05),
            )
        )
        minimum = float(
            options.get(
                "minimum Cdl window",
                options.get("minimum_cdl_window", 0.025),
            )
        )
        width = max(minimum, span * fraction)
    width = float(width)
    if width <= 0:
        raise ValueError("Cdl window must be positive.")

    forward_mask = np.abs(forward_E - start) <= width
    reverse_mask = np.abs(reverse_E - start) <= width
    fE, fi = _sorted_unique_xy(forward_E[forward_mask], forward_i[forward_mask])
    rE, ri = _sorted_unique_xy(reverse_E[reverse_mask], reverse_i[reverse_mask])
    if len(fE) < 2 or len(rE) < 2:
        raise ValueError("Cdl estimation found too few paired points near the start potential.")

    lo = max(float(np.nanmin(fE)), float(np.nanmin(rE)))
    hi = min(float(np.nanmax(fE)), float(np.nanmax(rE)))
    if lo >= hi:
        raise ValueError("Cdl estimation found no overlapping forward/reverse potentials near the start.")

    grid = np.unique(np.r_[fE[(fE >= lo) & (fE <= hi)], rE[(rE >= lo) & (rE <= hi)]])
    grid = grid[np.isfinite(grid)]
    if len(grid) < 2:
        raise ValueError("Cdl estimation found too few paired potentials near the start.")

    forward_interp = np.interp(grid, fE, fi)
    reverse_interp = np.interp(grid, rE, ri)
    delta_i = np.abs(reverse_interp - forward_interp)
    cdl_values = delta_i / (2.0 * scan_rate)
    cdl_values = cdl_values[np.isfinite(cdl_values)]
    if len(cdl_values) == 0:
        raise ValueError("Cdl estimation produced no finite values.")

    method = str(options.get("Cdl method", options.get("cdl_method", "median"))).strip().lower()
    if method == "median":
        cdl = float(np.nanmedian(cdl_values))
    elif method == "mean":
        cdl = float(np.nanmean(cdl_values))
    else:
        raise ValueError("Cdl method must be 'median' or 'mean'.")
    diagnostics = {
        "method": method,
        "scan_rate": scan_rate,
        "start_potential": start,
        "window_width": width,
        "potential_window": [float(np.nanmin(grid)), float(np.nanmax(grid))],
        "n_pairs": int(len(grid)),
        "median_current_difference": float(np.nanmedian(delta_i)),
        "mean_current_difference": float(np.nanmean(delta_i)),
    }
    return cdl, diagnostics


def _sorted_unique_xy(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    finite = np.isfinite(x) & np.isfinite(y)
    x = x[finite]
    y = y[finite]
    if len(x) == 0:
        return x, y
    order = np.argsort(x)
    x = x[order]
    y = y[order]
    unique_x, inverse = np.unique(x, return_inverse=True)
    if len(unique_x) == len(x):
        return x, y
    unique_y = np.asarray([np.nanmedian(y[inverse == i]) for i in range(len(unique_x))], dtype=float)
    return unique_x, unique_y


def _joined_segments(*segments):
    arrays = []
    for i, segment in enumerate(segments):
        segment = np.asarray(segment, dtype=float)
        arrays.append(segment if i == 0 else segment[1:])
    return np.concatenate(arrays)


def _time_from_potential(E, scan_rate):
    E = np.asarray(E, dtype=float)
    if len(E) == 0:
        return np.asarray([], dtype=float)
    dE = np.abs(np.diff(E, prepend=E[0]))
    t = np.cumsum(dE / float(scan_rate))
    t -= t[0]
    return t


def _representative_dt(t):
    diffs = np.diff(np.asarray(t, dtype=float))
    diffs = diffs[diffs > 0]
    if len(diffs) == 0:
        return 1.0
    return float(np.nanmedian(diffs))


def normalize_concentrations(concentrations):
    """Normalize concentration mappings into explicit bulk/surface groups."""
    concentrations = {} if concentrations is None else dict(concentrations)
    if "pools" in concentrations:
        raise ValueError(
            "Simulation pools are no longer supported. Enter every initial concentration; "
            "eCAT derives conservation constraints from reaction stoichiometry."
        )
    has_groups = any(key in concentrations for key in ("bulk", "surface"))
    if has_groups:
        bulk = concentrations.get("bulk", {}) or {}
        surface = concentrations.get("surface", {}) or {}
    else:
        bulk = concentrations
        surface = {}

    return {
        "surface": {_strip_surface_star(key): float(value) for key, value in surface.items()},
        "bulk": {str(key).rstrip("*"): float(value) for key, value in bulk.items()},
    }


def _compile_e_preset(species, surface_confined):
    a, b = _first_two_species(_species_group(species, surface_confined), fallback=("a", "b"))
    return f"E(1):{_species_name(a, surface_confined)}={_species_name(b, surface_confined)}"


def _compile_ee_preset(species, surface_confined):
    a, b, c = _first_three_species(_species_group(species, surface_confined), fallback=("a", "b", "c"))
    return "\n".join(
        [
            f"E(1):{_species_name(a, surface_confined)}={_species_name(b, surface_confined)}",
            f"E(1):{_species_name(b, surface_confined)}={_species_name(c, surface_confined)}",
        ]
    )


def _compile_ec_preset(species, surface_confined):
    a, b, c = _first_three_species(_species_group(species, surface_confined), fallback=("a", "b", "c"))
    return "\n".join(
        [
            f"E(1):{_species_name(a, surface_confined)}={_species_name(b, surface_confined)}",
            f"C:{_species_name(b, surface_confined)}={_species_name(c, surface_confined)}",
        ]
    )


def _compile_ece_preset(species, surface_confined):
    a, b, c, d = _first_four_species(_species_group(species, surface_confined), fallback=("a", "b", "c", "d"))
    return "\n".join(
        [
            f"E(1):{_species_name(a, surface_confined)}={_species_name(b, surface_confined)}",
            f"C:{_species_name(b, surface_confined)}={_species_name(c, surface_confined)}",
            f"E(1):{_species_name(c, surface_confined)}={_species_name(d, surface_confined)}",
        ]
    )


def _compile_square_preset(species, surface_confined):
    a, b, c, d = _first_four_species(_species_group(species, surface_confined), fallback=("a", "b", "c", "d"))
    return "\n".join(
        [
            f"E(1):{_species_name(a, surface_confined)}={_species_name(b, surface_confined)}",
            f"C:{_species_name(a, surface_confined)}={_species_name(c, surface_confined)}",
            f"E(1):{_species_name(c, surface_confined)}={_species_name(d, surface_confined)}",
            f"C:{_species_name(b, surface_confined)}={_species_name(d, surface_confined)}",
        ]
    )


def _compile_ecat_preset(params, force_surface=False):
    concentrations = normalize_concentrations(_concentrations_from_params(params))
    surface_confined = force_surface or bool(concentrations["surface"]) and not bool(concentrations["bulk"])

    if surface_confined:
        ox, red = _first_two_species(concentrations["surface"], fallback=("CatOx", "CatRed"))
        ox = _species_name(ox, True)
        red = _species_name(red, True)
        return f"E(1):{ox}={red}\nC:{red}={ox}"

    ox, red, substrate, product = _first_four_species(
        concentrations["bulk"],
        fallback=("CatOx", "CatRed", "Substrate", "Product"),
    )
    return f"E(1):{ox}={red}\nC:{red}+{substrate}>{ox}+{product}"


def _strip_surface_star(name):
    return str(name).rstrip("*")


def _species_name(name, surface_confined):
    name = _strip_surface_star(name)
    return f"{name}*" if surface_confined else name


def _species_group(species, surface_confined):
    return species["surface"] if surface_confined else species["bulk"]


def _first_two_species(species, fallback):
    names = list(species)
    while len(names) < 2:
        names.append(fallback[len(names)])
    return names[0], names[1]


def _first_three_species(species, fallback):
    names = list(species)
    while len(names) < 3:
        names.append(fallback[len(names)])
    return names[0], names[1], names[2]


def _first_four_species(species, fallback):
    names = list(species)
    while len(names) < 4:
        names.append(fallback[len(names)])
    return names[0], names[1], names[2], names[3]


def _electrokitty_parameters(params, mechanism_spec=None, species_order=None):
    cell = params.get("cell", {}) or {}
    spatial = _spatial_with_aliases(params.get("spatial", {}) or {})
    concentrations = normalize_concentrations(_concentrations_from_params(params))
    if species_order is None:
        surface_order, bulk_order = _electrokitty_species_order(mechanism_spec)
    else:
        surface_order, bulk_order = species_order
    area = float(cell.get("A", 1e-5))
    total_cdl = float(cell.get("Cdl", 0.0))

    cell_const = [
        float(cell.get("T", 298.15)),
        float(cell.get("Ru", 0.0)),
        _electrokitty_areal_cdl(total_cdl, area),
        area,
    ]
    spatial_info = [
        float(spatial.get("dx_fraction", spatial.get("fraction", 0.001 / 36))),
        int(spatial.get("nx", 20)),
        float(spatial.get("viscosity", 1e-6)),
        float(spatial.get("rotation", spatial.get("rot_freq", 0.0))),
    ]

    return {
        "kin": _electrokitty_kinetics(params, mechanism_spec),
        "cell_const": cell_const,
        "diffusion_const": _mapping_values(params.get("diffusion", {}), bulk_order),
        "isotherm": _mapping_values(params.get("isotherm", []), surface_order, surface=True),
        "spatial_info": spatial_info,
        "species_information": [
            _mapping_values(concentrations["surface"], surface_order, default=0.0, surface=True),
            _mapping_values(concentrations["bulk"], bulk_order, default=0.0),
        ],
    }


def _electrokitty_areal_cdl(total_cdl, area):
    total_cdl = float(total_cdl)
    area = float(area)
    if total_cdl == 0.0:
        return 0.0
    if not np.isfinite(area) or area <= 0:
        raise ValueError("ElectroKitty simulation requires positive cell.A when cell.Cdl is nonzero.")
    return total_cdl / area


def _electrokitty_backend_mechanism(mechanism_spec):
    mechanism = getattr(mechanism_spec, "mechanism", mechanism_spec)
    backend_lines = []
    for raw_line in str(mechanism or "").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if ":" not in line:
            raise ValueError(f"Mechanism line {line!r} must contain ':' before the reaction equation.")
        prefix, equation = line.split(":", 1)
        left, separator, right = _split_reaction_equation(equation)
        backend_lines.append(
            f"{prefix.replace(' ', '')}:"
            f"{_electrokitty_expand_stoichiometric_side(left)}"
            f"{separator}"
            f"{_electrokitty_expand_stoichiometric_side(right)}"
        )
    return "\n".join(backend_lines)


def _electrokitty_expand_stoichiometric_side(side):
    expanded = []
    for raw_term in str(side).split("+"):
        species, coefficient = _parse_stoichiometric_term(raw_term)
        expanded.extend([species] * coefficient)
    return "+".join(expanded)


def _electrokitty_species_order(mechanism_spec=None):
    if mechanism_spec is None:
        return None, None
    mechanism = _electrokitty_backend_mechanism(mechanism_spec)
    surface = []
    bulk = []
    for raw_line in str(mechanism or "").splitlines():
        line = raw_line.strip().replace(" ", "")
        if not line or ":" not in line:
            continue
        equation = line.split(":", 1)[1]
        equation = equation.replace("⇌", "=").replace("<=>", "=").replace("<->", "=")
        equation = equation.replace("→", ">").replace("->", ">")
        if "=" in equation:
            sides = equation.split("=")
        elif "<" in equation:
            sides = equation.split("<")
        elif ">" in equation:
            sides = equation.split(">")
        else:
            continue
        for side in sides:
            for term in side.split("+"):
                name = term.strip()
                if not name:
                    raise ValueError("ElectroKitty mechanism contains an empty species term.")
                if name.endswith("*"):
                    if name not in surface:
                        surface.append(name)
                elif name not in bulk:
                        bulk.append(name)
    return surface, bulk


def _electrokitty_runtime_species_order(electrokitty):
    parser = getattr(electrokitty, "Parser", None)
    parse = getattr(parser, "Parse_mechanism", None)
    if not callable(parse):
        raise RuntimeError(
            "The installed ElectroKitty version does not expose Parser.Parse_mechanism(); "
            "eCAT cannot safely align backend species parameter arrays."
        )
    try:
        parsed = parse()
        species = parsed[0]
        surface = list(species[0])
        bulk = list(species[1])
    except Exception as exc:
        raise RuntimeError(
            "The installed ElectroKitty parser returned an unsupported mechanism structure; "
            "eCAT cannot safely align backend species parameter arrays."
        ) from exc
    if any(not isinstance(name, str) or not name for name in [*surface, *bulk]):
        raise RuntimeError("The installed ElectroKitty parser returned invalid species names.")
    if len(surface) != len(set(surface)) or len(bulk) != len(set(bulk)):
        raise RuntimeError("The installed ElectroKitty parser returned duplicate species names.")
    return surface, bulk


def _electrokitty_kinetics(params, mechanism_spec=None):
    compiled = params.get("_compiled", {}) or {}
    kinetics = list(compiled.get("kinetics", params.get("kinetics", [])) or [])
    reactions = list(compiled.get("reactions", params.get("reactions", [])) or [])
    if mechanism_spec is None:
        return _kinetics_to_electrokitty(kinetics)

    step_kinds = _electrokitty_step_kinds(mechanism_spec.mechanism)
    if not reactions:
        if "C" in step_kinds and len(kinetics) < len(step_kinds):
            raise ValueError(
                "ElectroKitty mechanism contains a chemical step, but no matching reaction rate was found. "
                "Add params['reactions'] entries such as {'kf': ..., 'kb': ...}."
            )
        return _kinetics_to_electrokitty(kinetics)

    electrochemical = []
    chemical_from_kinetics = []
    for entry in kinetics:
        if _is_chemical_kinetic_entry(entry):
            chemical_from_kinetics.append(entry)
        else:
            electrochemical.append(entry)

    chemical = list(reactions) + chemical_from_kinetics
    ordered = []
    e_index = 0
    c_index = 0
    for kind in step_kinds:
        if kind == "E":
            if e_index >= len(electrochemical):
                raise ValueError(
                    "ElectroKitty mechanism contains more electrochemical steps than params['kinetics'] entries."
                )
            ordered.append(electrochemical[e_index])
            e_index += 1
        elif kind == "C":
            if c_index >= len(chemical):
                raise ValueError(
                    "ElectroKitty mechanism contains a chemical step, but no matching reaction rate was found. "
                    "Add params['reactions'] entries such as {'kf': ..., 'kb': ...}."
                )
            ordered.append(chemical[c_index])
            c_index += 1

    ordered.extend(electrochemical[e_index:])
    ordered.extend(chemical[c_index:])
    return _kinetics_to_electrokitty(ordered)


def _is_chemical_kinetic_entry(entry):
    if isinstance(entry, dict):
        return any(key in entry for key in ("kf", "kb", "k")) and not any(
            key in entry for key in ("E0", "k0", "alpha")
        )
    return False


def _electrokitty_step_kinds(mechanism):
    kinds = []
    for line in str(mechanism).splitlines():
        text = line.strip()
        if not text:
            continue
        first = text[0].upper()
        if first in {"E", "C"}:
            kinds.append(first)
    return kinds


def _kinetics_to_electrokitty(kinetics):
    out = []
    for entry in kinetics:
        if isinstance(entry, dict):
            if {"alpha", "k0", "E0"}.issubset(entry):
                out.append([float(entry["alpha"]), float(entry["k0"]), float(entry["E0"])])
            elif {"kf", "kb"}.issubset(entry):
                out.append([float(entry["kf"]), float(entry["kb"])])
            elif "k" in entry:
                out.append([float(entry["k"])])
            else:
                raise ValueError("Kinetic entries must define alpha/k0/E0, kf/kb, or k.")
        else:
            out.append([float(value) for value in entry])
    return out


def _mapping_values(value, order=None, default=None, surface=False):
    if isinstance(value, dict):
        if order is not None:
            lookup = {
                (_strip_surface_star(key) if surface else str(key).rstrip("*")): item
                for key, item in value.items()
            }
            out = []
            for name in order:
                key = _strip_surface_star(name) if surface else str(name).rstrip("*")
                if key in lookup:
                    out.append(float(lookup[key]))
                elif default is not None:
                    out.append(float(default))
                else:
                    raise ValueError(f"Missing ElectroKitty parameter value for species {key!r}.")
            return out
        return [float(v) for v in value.values()]
    return [float(v) for v in value]


def _isotherm_to_electrokitty(value):
    if isinstance(value, dict):
        return [float(v) for v in value.values()]
    return [float(v) for v in value]


def _kinetic_at(kinetics, index):
    if index < len(kinetics) and isinstance(kinetics[index], dict):
        return kinetics[index]
    return {}


def _program_bounds(input):
    E = np.asarray(input.E, dtype=float)
    if len(E) == 0:
        raise ValueError("Simulation input has no potential points.")
    first = float(E[0])
    last = float(E[-1])
    vertex = float(E[np.argmax(np.abs(E - first))])
    return first, vertex, last


def _scan_rate_from_input(input):
    E = np.asarray(input.E, dtype=float)
    t = np.asarray(input.t, dtype=float)
    if len(E) < 2 or len(t) < 2:
        return 0.1
    dt = np.diff(t)
    dE = np.abs(np.diff(E))
    mask = dt > 0
    if np.count_nonzero(mask) == 0:
        return 0.1
    return float(np.nanmedian(dE[mask] / dt[mask]))


def _resolve_current_sign(backend_current, measured_current, options):
    explicit = options.get("current sign", options.get("current_sign"))
    if explicit is not None:
        if explicit in (1, "+", "backend", "native"):
            return 1
        if explicit in (-1, "-", "flip", "flipped"):
            return -1
        raise ValueError("current sign must be 'backend', 'native', 'flip', 1, or -1.")

    if measured_current is None:
        return 1

    backend_current = np.asarray(backend_current, dtype=float)
    measured_current = np.asarray(measured_current, dtype=float)
    mask = np.isfinite(backend_current) & np.isfinite(measured_current)
    if np.count_nonzero(mask) == 0:
        return 1

    same = np.sum((backend_current[mask] - measured_current[mask]) ** 2)
    flipped = np.sum((-backend_current[mask] - measured_current[mask]) ** 2)
    return -1 if flipped < same else 1


def _import_electrokitty():
    try:
        from electrokitty import ElectroKitty
    except Exception as exc:  # pragma: no cover - exact import failure varies by environment
        raise ImportError(_ECAT_SIMULATION_INSTALL_MESSAGE) from exc
    return ElectroKitty
