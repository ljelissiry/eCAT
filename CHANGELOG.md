# Changelog

## 0.1.0b8 - Release candidate

- Normalize filter/sort/group keys across case and underscore spellings; expose WE/RE/CE aliases and `filter()` inspection with bounded candidate printing.
- Replace `get_available_filter_values()` with `filter(objects, fields)` or `filter(objects, "available", {"print": False})`; migrate the Workbench caller so all metadata choices remain available.
- Preserve solvent labels from filenames, group known aliases by identity, and use the most frequent supplied spelling within a plot.
- Report import totals and per-folder outcomes, keep subfolder paths relative to the import root, and avoid repeating reference-folder paths.
- Show shared electrodes once above object tables and differing electrodes in per-file columns; add explicit display-column controls.

- Use three significant figures for automatic scan-window matching, retaining five for other continuous metadata and preserving measured potentials.
- Draw polynomial and Sevcik fit lines in increasing x order without changing input-order fit selections or statistics.
- Identify the curve and segment pair when trumpet peak extraction fails.
- Allow collection filtering for directly loaded objects without a folder attribute.
- Update GitHub artifact actions to Node.js 24 versions.

## 0.1.0b7 - 2026-10-19

- Added filename decimal aliases, preserved spaced species names and DMA/DMAc spelling, and supported scan-rate units such as mV_s.
- Prefer reliable waveform scan rates over filename rates when no instrument-header rate is supplied; retain provenance and flag disagreements.
- Unified CV sweep segmentation and added significant-figure matching controls for filtering, sorting, and grouping.
- Improved species-aware concentration analysis, peak-current transform help, and trumpet diagnostics, with updated tutorials.
- Fixed CA current/rate plot annotations to use the displayed current units and offset, while retaining numerical results in amperes. Corrected the Fc/Fc+ segment selection and current-at-potential label in tutorial 03.
- Retain the established public simulation interface with standard ElectroKitty >=1.0.11.6. Experimental time-grid, repaired-build, and local-equilibrium work is excluded.


- Added compact animation display selection through `display format`: `auto`
  prefers HTML5 video when FFmpeg is available and falls back to JSHTML, while
  explicit HTML5, JSHTML, and GIF modes remain available for notebook display
  and HTML export.
- Simplified `AnimationResult` so `show()` is the sole public method for inline
  animation rendering.
- Reworked the README around a beginner-first installation and quickstart
  workflow, added the eCAT logo and full package name, documented every
  numbered quickstart notebook, and moved optional app/simulation guidance
  after the core workflow.
- Replaced the ambiguous object `info()` method with `metadata()` for
  machine-readable metadata snapshots, made `show()` the sole public object
  presentation API, and privatized the compact statistics/compound-label
  helpers used internally by plotting. Metadata snapshots now expose common
  identity, provenance, units, parser, reference, and processing fields across
  CV, CA, CP, and DPV objects, and nested values are independent copies.
  Human-readable `show()` tables now format source file and folder paths
  relative to the active project when possible, without changing stored paths.
- Replaced the retired CV-only `cv.fft()` diagnostic with inherited
  `obj.spectrum()` for CV, CA, CP, DPV, and generic electrochemistry objects.
  The typed workflow provides Welch power spectral density and calibrated
  one-sided FFT amplitude/phase results, records measured or reconstructed
  timing and sampling provenance, and returns standard tables, summaries,
  diagnostics, warnings, and spectrum/time-domain figures.

## 0.1.0b6 - TestPyPI Packaging Trial - 2026-09-03

- Renamed the installable distribution to `ecat-electrochemistry` for its first
  PyPI beta while preserving `import ecat as e`, the eCAT brand, and existing
  application commands.
- Prepared separate TestPyPI and PyPI Trusted Publishing paths so release
  artifacts are built from a clean tagged checkout and uploaded without a
  long-lived API token.

## 0.1.0b5 - Beta Reliability, Peak Tracking, And CI - 2026-09-02

- Made automatic reference correction segment-aware: reference couples now require adjacent, opposite-direction sweeps, remain invariant to current inversion, use resolution-aware physical ranking, fail explicitly on ambiguous automatic choices, retain selected-pair provenance, and expose segment/candidate diagnostics through `troubleshoot=True`.
- Fixed delegated option routing so Sevcik and peak-current fitting preserve `peak kind`, `peak fallback`, and peak-marker controls; Tafel and nested plot/analysis helpers now also preserve accepted case, underscore, and registered-alias spellings instead of silently reverting them to defaults.
- Added explicit `fit_peak_potential()` tracking modes for independent, within-CV, series-consensus, and strict series-consensus peak selection, with per-row tracking diagnostics.
- Hardened installed-package and app behavior: packaged example folders now ship in wheels, `ecat-app --version` uses the canonical package version, folder status no longer scrapes console output, and `get_data(..., {"print": False})` is quiet unless troubleshooting is requested.
- Fixed CP cycle-efficiency division by zero, translated SciPy covariance warnings into fit metadata and readable output, removed duplicate option metadata and mutable defaults, and cleaned dead progress formatting.
- Polished reversibility diagnostics so nearly constant peak-separation series use readable plain axes without scientific offset clutter.
- Updated package metadata to PEP 639 license fields and added tested source/wheel packaging for the app example datasets.
- Added Linux Python 3.10-3.14, app, simulation, ElectroKitty, and release-metadata CI coverage plus native Windows core and installed-wheel app jobs.
- Normalized user-facing relative paths to forward slashes across operating systems and made release/app tests explicitly UTF-8-safe on Windows.
- Added a staged Ruff policy that blocks high-confidence correctness regressions while reporting broader existing lint debt without blocking the beta.
- Re-executed and visually audited every numbered quickstart notebook with embedded b5 outputs.

## 0.1.0b4 - Beta Parser, Plotting, Analysis, Simulation, And App Refresh - 2026-08-24

- Added old BASI-Epsilon `.dat` CV text import support, including folder discovery and headerless potential/current table handling.
- Added `reversibility_analysis()` for cautious, series-level bulk or surface-confined electron-transfer assessment plus a separate chemical-reversibility conclusion, with documented Matsuda-Ayabe, Nicholson, Sevcik, trumpet/Laviron, and irreversible-asymptote eligibility rules.
- Added `surface_coverage_analysis()` with independent peak-slope and tangent-corrected charge estimates of surface coverage and total electroactive loading, branch/method agreement diagnostics, and configurable agreement tolerance.
- Added tangent-corrected full width at half peak current to `peak_info()` and both peak widths plus direction-resolved cathodic/anodic peak currents and `|ipa/ipc|` to `wave_info()`.
- Reversibility and surface-coverage equations now use the shared LaTeX pretty-print path; reversibility diagnostics use constrained, vertically stacked panels with physical branch symbols.
- Kept Matsuda-Ayabe `Lambda` classification independent from Nicholson `k0` eligibility, added quantitative region/current-ratio evidence, and separated the `current ratio tolerance` default (`0.10`) from cross-method `agreement tolerance` (`0.25`).
- Made single-branch surface-coverage fit output a compact vertical result table and kept units in result values rather than equation definitions.
- Made Sevcik analysis strictly diffusion-controlled with the fixed square-root scan-rate relationship and removed the configurable scan-rate exponent from the package and app.
- Added consistent EC-Lab ASCII `.mpt` support to direct and folder imports while rejecting BioLogic binary `.mpr` input before delimited-text parsing with actionable conversion guidance.
- Made filename/header scan-rate reconciliation tolerant of normal acquisition and rounding precision while retaining header-first behavior for genuine mismatches.
- Kept canonical CV current selection through typed analysis defaults when imported tables also contain time, cycle, segment, or other auxiliary columns.
- Normalized path-like inputs at the object-construction boundary so `pathlib.Path` works as documented with `echem.from_file()`.
- Made trumpet analysis scan-direction independent by mapping paired segments to cathodic/anodic branches before calculating alpha, beta, and k0; ambiguous branch pairs now raise targeted errors and resolved branch assignments are reported.
- Removed the inert `stacking` plot option; use numeric `offset` to separate traces vertically in `multiplot()`.
- Allowed `multiplot()` `offset` to accept explicit per-trace offset lists while preserving scalar values as constant trace-to-trace steps.
- Applied the existing `sig figs` setting to scaled scan-rate values in object summaries and gradient/colorbar labels so instrument precision noise does not leak into display text.
- Made directional-arrow orientation use eCAT's default smoothed derivative, with the existing local point slope retained only as a short-trace fallback.
- Standardized notebook-facing analysis reports as Parameters, Equations,
  Summary, then Data only with `print all=True`; nested peak helpers no longer
  leak their own reports into Sevcik or Nicholson output.
- Standardized row identity across reversibility, surface coverage, Nicholson,
  Sevcik, trumpet, FOWA, and plateau reports: unique scan-rate/concentration
  context omits long names, replicate rows add `Name` first, and grouped rows
  use `Condition`; returned data retains names for traceability.
- Added compact phase-specific reversibility evidence tables; detailed peak
  and eligibility columns remain available in the returned result rather than
  crowding the printed report.
- Labeled reversibility equations by scientific role and made Sevcik,
  electron-transfer-rate, and irreversible-asymptote equations conditional on
  the evidence paths actually used; unresolved bulk rates now explain how to
  supply or estimate the required diffusion coefficient.
- Added separate real-data reversibility and imported-data surface-confined
  quickstarts, shifted advanced analysis and simulation/fitting to notebooks
  09-12, and added a reproducible surface-series Excel fixture generated outside
  the notebook.
- Applied automatic current-unit scaling to both surface-coverage diagnostic
  plots while retaining explicit `y unit` control.

- Accept `wave ranges` as the explicit per-CV alias for FOWA catalytic-wave windows and document it as `wave range(s)`.
- Add `fit=True/False` to FOWA; enabled fits now warn and skip only unusable per-CV regressions instead of aborting the complete analysis.
- Improved CA current/charge overlays with automatic charge-unit scaling, shared or independent current/charge-axis inversion, inversion-safe zero bounds, and shared secondary-axis styling.
- Standardized membership filtering so compounds, concentrations, and species use all-of matching by default while explicit `any` / `all` requests remain available and clearly printed.
- Improved import/export behavior with deduplicated warnings, quoted paths relative to the requested import folder, SI-unit canonicalization, axis-aware CSV unit conversion, and zero-concentration species provenance without misleading plot/colorbar entries.
- Refined FOWA, plateau-current, current-normalization, and peak-selection diagnostics with more consistent units, labels, equations, grouped output, and optional reference plots.

- Consolidated CV text import on `get_data()`, removed the redundant CV-only loader, and preserved flexible delimiter/header parsing plus header-first scan-rate handling in the unified parser.
- Added copy-first `cv.filter()` preprocessing with recorded Savitzky-Golay, Gaussian, median, Butterworth, and moving-average settings; filter history is shown on single objects and preserved by Excel workbook round trips.
- Removed the retired `eCAT` import shim, beta option spellings, and plot-time animation/normalization switches; `import ecat as e`, `labels`, `fit indices`, `fit color`, `fit line range`, and reference-prefixed import options are now the canonical surface.
- Added shared SciPy `curve_fit` method controls, fitted-parameter counts, and warnings for exactly determined or underdetermined model fits.
- Simplified `cv.peak_info()` child orchestration and removed the abandoned wave-analysis implementation.
- Added GitHub Actions coverage for Python 3.10-3.14 plus app and optional-simulation smoke jobs.
- Made `src/ecat/_version.py` the package version source, added CI checks for app/shortcut/standalone release metadata, and renamed internal typed-option dictionary adapters to neutral resolved-option terminology.
- Replaced explicit simulation concentration pools with reaction-stoichiometry-derived pre-equilibrium, added reaction-local `equilibrate=False`, and added chemical-only `incubation_time` on `SimulatedCVInput` with entered/equilibrated/incubated state reporting.
- Added immutable `with_incubation_time(...)` reruns, preserved normalized entered parameters on `SimulatedCV.input_params`, and updated fitting so every objective evaluation starts from the unchanged entered composition.
- Added phase-specific activity standards for bulk concentration and surface coverage, kept pre-run incubation bulk-homogeneous only, and fixed stoichiometric species parsing plus entered-state/quiet-time handling in native ElectroKitty fitting.
- Standardized FOWA and plateau-current option aliases around `n_cat` / `n_turn`, moved plateau/FOWA display units into result-table headers, and switched displayed equations to literature-style `n` / `n′` notation with explicit definitions.
- Corrected the transformed FOWA plot x-axis label so it reflects the resolved redox-reference mode and the `n` exponent convention.

- Added ParseResult-first parser coverage for tested EC-Lab-style CV/CA/CP text, limited NOVA ASCII CV text, and generic CV-like text fallback, with raw metadata and parser warnings preserved on loaded objects.
- Expanded filename metadata parsing with custom parser hooks/settings, stricter concentration phrase handling, case-insensitive gas/solvent detection, concentration-less compound support, and electrode metadata extraction.
- Improved import/reference error messaging, reference-label chemical formatting, object display summaries, and Excel workbook round-tripping through `save_data(..., {"format": "xlsx"})` and `get_data_from_excel(...)`.
- Added and documented CA current/rate helpers, including `current_at_time()`, `average_current()`, `rate_at_time()`, and `average_rate()` with equation and metric-table output.
- Improved CV plotting and analysis ergonomics, including CV default axis selection for multi-column data, `cv.plot_program()`, connected-window trimming semantics, peak fallback diagnostics, and more consistent pretty-print result tables.
- Expanded scatter-fit/model workflows with shared fit-model plumbing, custom formula fitting, flexible fit-index selection, clearer fit labels, and more consistent fit statistics output.
- Refined Sevcik, trumpet, Nicholson, FOWA, plateau-current, Tafel, and related notebook-facing analysis output/plotting behavior.
- Added plotting polish for public `colors`, colorbar labels/ticks, scale bars, directional arrows, subtitles, unit scaling, animation progress/display/save behavior, and updated plotting/animation quickstart examples.
- Updated the eCAT Workbench app packaging, optional app dependency messaging, launchers, layout behavior, UI controls, and app tests for the current beta workflow.
- Updated the quickstart notebooks, API reference, beta scope, troubleshooting notes, README, and eCAT Guide for the current beta surface.

## 0.1.0b3 - Beta App And Notebook Refresh - 2026-07-01

- Expanded the eCAT app backbone, layout, workflow wiring, code generation, plot callbacks, styling, and tests.
- Added packaged app assets for plot actions and app controls, including updated SVG icons and JavaScript helpers.
- Added notebook-facing app launch support and updated setup, plotting/animation, and simulation notebooks for the beta workflow.
- Simplified app launch behavior around `ecat-app`, made the native pywebview window the default, added `ecat-app --browser` / `e.open_app(browser=True)` browser mode, and removed the old `ecat-browser` command.
- Renamed the internal app source tree from `apps/browser/src/ecat_browser` to `apps/workbench/src/ecat_app` and renamed the matching app docs/tests.
- Removed the inactive fullscreen button from the app plot controls.
- Restored executed outputs in the public quickstart notebooks while keeping notebook outputs free of local absolute paths.
- Advanced the animation API and option plumbing, including timing/default behavior, validation, placeholder behavior, plotting smoke tests, and documentation coverage.
- Improved simulation and fitting workflows, including grouped fitting coverage, model/fit-option validation, fit index handling, and fit/analysis output consistency tests.
- Added `AnalysisResult.to_csv(...)` for primary-table CSV export and `AnalysisResult.to_excel(...)` for table plus metadata workbook export.
- Fixed FOWA diagnostic plot labeling so automatic diagnostic labels are delegated to `multiplot`, preserving gradient/colorbar label behavior while retaining explicit labels.
- Improved reference-shift handling, path display behavior, plotting smoke coverage, option defaults, package import checks, and public API reference cleanup.
- Updated the eCAT Guide, beta scope, API reference, and package metadata for the current beta state.

## 0.1.0b1 - Initial Beta Readiness

- Documented the beta scope, supported file/technique matrix, limitations, quickstart, beta tester guide, troubleshooting notes, and bug-report template.
- Added beta-readiness tests for existing object accessors, summaries, CA/CP plotting smoke behavior, figure saving, and CSV export.
- Marked package metadata as beta release `0.1.0b1`.
- No core workflow behavior, public return shapes, or scientific algorithms were intentionally changed in this readiness pass.
