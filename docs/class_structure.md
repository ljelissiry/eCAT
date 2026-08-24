# eCAT Class Structure

This document summarizes the main object model in eCAT. It is intended as a
developer and advanced-user map of class responsibilities, owned variables, and
the public methods that notebook workflows usually depend on.

## Core Object Model

### `echem`

`echem` is the base class for imported electrochemistry time-series objects.
It owns the shared file metadata, parsed data table, units, reference-shift
metadata, and common data access methods.

Owned variables include:

- `filepath`, `options`, `timestamp`, `creation_time`, `modification_time`
- `name`, `data`, `type`, `software`, `num_x_cols`, `units`
- `temperature`, `electrode_area`, `delta_x`, `segments`
- `gas`, `solvent`, `compounds`, `concentrations`
- `ir_comp_resistance`, `ir_uncomp_resistance`, `ir_comp_percent`
- `reference_shift`, `reference_label`, `reference_mode`,
  `reference_source_file`, `reference_failure_message`

Key public methods include:

- `from_file()`, `detect_software()`, `detect_experiment_type()`
- `x()`, `y()`, `xy()`
- `stats()`, `info()`
- `plot()`, plus shared helpers used by analysis and plotting functions

### `cv`

`cv` subclasses `echem` and represents cyclic voltammetry loaded from files or
manual data. It adds CV-specific setup fields and analysis behavior.

Owned variables include the inherited `echem` state plus:

- `init_E`, `final_E`, `min_E`, `max_E`
- `scan_rate`, `segments`, `delta_x`

Key public methods include:

- `x()`, `y()`, `xy()`, `analysis_segment_data()`
- `peak_potential()`, `peak_current()`, `peak_width()`, `half_peak_potential()`,
  `half_wave_potential()`
- `current_at_potential()` and related CV analysis helpers
- `stats()`, `plot()`

### `ca`

`ca` subclasses `echem` and represents chronoamperometry.

Owned variables include the inherited `echem` state plus:

- `init_E`, `sample_interval`, `run_time`, `quiet_time`, `sensitivity`

Key public methods include:

- inherited `x()`, `y()`, `xy()`, `plot()`
- `stats()`
- CA-specific charge/current analysis helpers

### `cp`

`cp` subclasses `echem` and represents chronopotentiometry and related
galvanostatic cycling data.

Owned variables include the inherited `echem` state plus:

- `cathodic_current`, `anodic_current`, `init_PN`
- `high_E_limit`, `low_E_limit`
- `cathodic_time`, `anodic_time`, `sample_int`, `quiet_time`
- `init_E`, `final_E`, `min_E`, `max_E`, `segments`

Key public methods include:

- inherited `x()`, `y()`, `xy()`, `plot()`
- `stats()`
- CP cycle and segment plotting/analysis helpers

### `dpv`

`dpv` subclasses `echem` and represents differential pulse voltammetry.

Owned variables include the inherited `echem` state plus:

- `init_E`, `final_E`, `incr_E`, `amplitude`
- `pulse_width`, `sample_width`, `pulse_period`, `quiet_time`
- `sensitivity`, `comp_R`, `min_E`, `max_E`

DPV timing and pulse attributes are stored as unitless SI values. Public stats
use unitless keys such as `amplitude`, `pulse width`, `sample width`, and
`pulse period`; printed summaries and plot subtitles put autoscaled units in the
displayed values.

Key public methods include:

- inherited `x()`, `y()`, `xy()`, `plot()`
- `stats()`, `txt_stats()`
- DPV peak analysis helpers

## Simulation Object Model

Simulation classes live under `e.simulation`.

### `SimulatedCVInput`

`SimulatedCVInput` is the potential/time/current input object passed to CV
simulation and fitting routines. It is not a subclass of `cv`; it is the
simulation program or measured CV extract that a backend consumes.

Owned variables:

- `E`: potential array
- `t`: time array
- `i`: optional measured current array
- `metadata`: setup metadata such as scan rate, segments, `incubation_time`,
  quiet time, units, stride, and source-window details
- `source`: original source object or source label

Key public methods and properties:

- `has_current`
- `plot(options=None)`: plots the input waveform. Defaults to time vs
  potential; use `{"x axis": "potential", "y axis": "current"}` for inputs
  that include measured current, or `{"plot quiet time": True}` to draw a
  metadata quiet hold at negative time
- `with_scan_rate(scan_rate)`: returns a new input with the same waveform and a
  new timebase
- `with_incubation_time(seconds)`: returns a new input with a changed
  chemical-only incubation duration; the default is `0`
- `show(options=None)`: displays setup by default; use
  `{"print setup": False}` to suppress or `{"print setup": "raw"}` for a raw
  debug dump

### `SimulatedCV`

`SimulatedCV` subclasses `cv` so simulated CV results can use CV analysis and
plotting workflows, including `multiplot`. It is constructed from backend
simulation output, not from file parsing. Measured current stays on
`SimulatedCVInput` and `SimulationFitResult`; `SimulatedCV.data` contains
simulated current only.

Quiet time is stored as `SimulatedCVInput.metadata["quiet_time"]` and is applied
when constructing backend simulation input, not inserted into the stored
potential/time arrays.
`incubation_time` is also input metadata, but it evolves bulk homogeneous
chemical reactions before the quiet-time electrode hold. Surface and
mixed-phase steps remain backend-only during incubation.

Owned variables:

- `data`: simulated data table with `Potential`, `Current`, `Time`, and
  `Backend Current`
- `params`: prepared simulation parameter dictionary
- `input_params`: normalized entered parameters used for immutable reruns;
  concentrations here have not been equilibrated or incubated
- `mechanism`: compiled `MechanismSpec`
- `input`: source `SimulatedCVInput`
- `backend_result`: raw backend result object, when available
- `current_sign`, `figure`, `axes`, `summary`
- CV-compatible fields such as `name`, `type`, `software`, `units`,
  `scan_rate`, `init_E`, `final_E`, `min_E`, `max_E`, `segments`, and `delta_x`

Key public methods:

- `x()`, `y()`, `xy()`, `plot()`
- inherited CV analysis methods such as `analysis_segment_data()` and
  `current_at_potential()`
- `with_scan_rate(scan_rate, ...)`: reruns the simulation using a copied input
  with an updated scan rate
- `with_incubation_time(seconds, ...)`: reruns from the entered parameters with
  a changed chemical incubation duration
- `with_params(params=None, set=None, ...)`: deep-merges parameter updates,
  applies optional path updates, and reruns the simulation
- `with_param(path, value, ...)`: updates one parameter path and reruns
- `with_input(input, ...)` and `with_mechanism(mechanism, ...)`: rerun with a
  replacement input program or mechanism
- `show(options=None)`: displays setup by default. `{"print params": True}`
  adds simulation parameter tables, `{"print checks": True}` adds diagnostic
  parameter checks, `{"print states": True}` compares entered, equilibrated,
  and incubated concentrations, and `{"print data": True}` adds the full
  simulated data table.

Simulation parameter tables use canonical `concentrations` and `diffusion`
inputs. `species` may be used only as input sugar with per-species `type`, `C`,
and `D` fields; prepared/output params normalize that shape into
`concentrations` and `diffusion`. `{"print params": "compact"}` combines
amounts and diffusion coefficients into a compact human-facing `species` table.
Simulation params use SI-derived public units: potentials in `V`, time in `s`,
current in `A`, scan rate in `V/s`, bulk concentrations in `mol/m³`, surface
coverages in `mol/m²`, diffusion in
`m²/s`, electrode area in `m²`, resistance in `Ω`, temperature in `K`, and
`cell.Cdl` as total capacitance in `F`. Chemical `k`/`kf`/`kb` units depend on
reaction order on the `mol/m³` concentration basis; first-order rates are
`s⁻¹`, second-order rates are `m³ mol⁻¹ s⁻¹`, and third-order rates are
`m⁶ mol⁻² s⁻¹`.
`kinetics` and `reactions` are user-facing physical parameter sections and may
be lists, integer-keyed dictionaries, or dictionaries keyed by mechanism
reaction strings. `reactions` supports irreversible `k`, reversible `kf`/`kb`,
and equilibrium-derived `K` with `k_exchange` or `koff`;
backend-ready rates are compiled under `_compiled`. Fits can optimize physical
paths such as `reactions.0.K` and `reactions.0.k_exchange`; eCAT recompiles
those trial values into backend `kf`/`kb` rates for each simulation.
`k_exchange` is a pool-free standard-state rate scale. For reactant order
`n_r`, product order `n_p`, and phase-appropriate activity standard `X°`, eCAT
uses `k_exchange = kf*X°**(n_r-1) + kb*X°**(n_p-1)` together with the
activity-based `K` to compile backend rates. `X°` comes from
`activity.standard_concentration` for bulk reactions or
`activity.standard_coverage` for surface reactions. The actual mass-action rate
still uses each dataset's current amounts.

Every species in an explicit-`K` reaction has an entered concentration, with
zero used for an initially absent species. eCAT builds a stoichiometric matrix,
infers its conservation equations, and solves the pre-equilibrium before
backend simulation. `equilibrate=False` excludes a reversible reaction from
that algebraic solve while retaining it for incubation and CV dynamics.
`concentrations.pools`, top-level `pools`, and top-level `equilibria` are
rejected migration inputs. Consistent dependent cycles are allowed;
inconsistent cycles raise an equilibrium-residual error. Mixed bulk/surface
pre-equilibria remain unsupported because no volume-to-area conservation
convention is defined.

After pre-equilibrium, eCAT optionally integrates the bulk homogeneous chemical
network for `SimulatedCVInput.metadata["incubation_time"]`. Surface and
mixed-phase chemical steps remain backend-only during this stage. The resulting
concentrations are then passed to the backend, which applies quiet time at the
initial potential and runs the CV. Numerical rank and residual diagnostics are
reported, but species names are not interpreted as molecular formulas and eCAT
does not claim elemental or charge balance.

Activity coefficients live under `activity.gamma`; they default to `1` and
display as extra species-table columns only when at least one gamma is
non-ideal.
Cell constants may be supplied as a mapping or as `"cell": "auto"`; the
shorthand requests auto Cdl estimation and fills temperature, resistance, and
electrode area from the source CV or simulation defaults.
`cell.Cdl` is owned and displayed as total capacitance in `F`. Backend adapters
that need area-normalized capacitance perform the conversion from
`cell.Cdl / cell.A` internally, so users should not pass `F/m²` as `cell.Cdl`.
`simulate_cv(..., options={"check params": True})` and
`SimulatedCV.show({"print checks": True})` print likely interpretation issues
such as missing/unused diffusion entries, parameter fallbacks, and preset
mechanism species order.

### `SimulationFitResult`

`SimulationFitResult` is the notebook-facing result returned by
`e.simulation.fit_cv()`. `fit_cv()` accepts real eCAT `cv` objects, fit-ready
`SimulatedCVInput` objects with measured current, and `SimulatedCV` objects.

Owned variables:

- `best_params`, `fit_spec`, `method`, `backend`
- `optimizer_result`, `initial_result`, `simulation_result`
- `residuals`, `corrections`, `summary`, `backend_result`

Key public methods and properties:

- `data`: final simulated CV data table, or `None` if unavailable
- `plot(options=None)`: delegates to the final `SimulatedCV`
- `show(options=None)`: displays requested fitting sections using the same
  table machinery as fit-time printing. Supported display options include
  `print setup`, `print stats`, `print corrections`, `print params`, and
  `print simulation`.

### `SimulationGroupFitResult`

`SimulationGroupFitResult` is returned by `e.simulation.fit_cvs()`, which fits
one shared mechanism across multiple real CVs, fit-ready `SimulatedCVInput`
objects, or existing `SimulatedCV` results.

Owned variables:

- `best_params`: shared/global best-fit params
- `best_params_by_cv`: final prepared params for each CV dataset
- `simulation_results`, `measured_currents`, `residuals`, `residuals_by_cv`
- `corrections_by_cv`, `summary`, `fit_spec`, `per_cv`, `datasets`

Key public methods and properties:

- `data`: list of final simulated CV data tables
- `plot(options=None)`: plots per-CV measured-vs-simulated diagnostics
- `show(options=None)`: displays group setup, statistics, corrections, params,
  and optional final simulated CV setup tables

`fit_cvs()` keeps the existing `fit` spec as the fixed/varied-parameter
contract. The `per_cv` argument is only a dataset-specific path selector: paths
listed in both `fit["vary"]` and `per_cv` are fitted separately for each CV;
paths listed only in `per_cv` are fixed separately for each CV.

### `MechanismSpec`

`MechanismSpec` is the compiled mechanism metadata used by backend adapters.

Owned variables:

- `mechanism`: user-facing eCAT mechanism string
- `preset`: eCAT preset name or `raw`
- `note`: optional warning or inference note
- `surface_confined`: whether the compiled mechanism is surface-confined
- `raw`: whether the user supplied a custom eCAT mechanism rather than a preset;
  this historical field name does not mean the string is forwarded unchanged

Key use:

- Returned by `compile_mechanism()` and stored on `SimulatedCV`.
- Consumed by backend adapters when translating eCAT parameters into backend
  simulation calls.

Custom eCAT mechanisms accept conventional coefficients (`2A`) and repeated
species (`A+A`) as equivalent stoichiometry. The original form is retained for
display and reaction paths. The ElectroKitty adapter privately expands
coefficients into repeated terms and uses the installed ElectroKitty parser's
surface/bulk species order when constructing positional parameter arrays.

## Design Notes

`SimulatedCVInput` and `SimulatedCV` are intentionally separate. The input owns
the waveform and measured-current context; the result owns simulated data,
backend metadata, and CV-compatible analysis behavior.

`SimulatedCV` subclasses `cv` because simulated CVs should participate in the
same CV analysis surface as imported CVs. The construction path is different:
file-backed `cv` objects parse instrument files, while `SimulatedCV` objects are
assembled from simulation output and prepared parameters.
