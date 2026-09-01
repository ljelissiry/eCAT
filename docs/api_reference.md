# eCAT API Reference

This page is a quick index of the public eCAT functions, classes, and common object methods. It is meant for users who want to discover what is available from Python without reading every tutorial first.

## Getting Help In Python

```python
import ecat as e

help(e)
help(e.cv)
help(e.cv.peak_current)

e.describe_options()
e.describe_options("plot")
e.describe_options("cv.peak_current")
e.describe_options("simulation.fit_cv")
```

Use `describe_options()` when you know the workflow but need the available options. Use `help(...)` when you want function arguments, return values, and a short example.

Option names accept spaces, underscores, and hyphens in most notebook-facing APIs, and registered choice values are case-insensitive. For choice-like options, spaces, underscores, and hyphens are also treated interchangeably: for example, `"scale_linear_baseline"` and `"scale linear baseline"` resolve the same way where that choice is valid. If an option explicitly lists `none` as a valid choice, Python `None`, `"None"`, `"off"`, `"false"`, `"no"`, and `"0"` are interpreted as the canonical string `"none"`; omitting the option still uses the default.

## Module Organization

Most users should still start with:

```python
import ecat as e
```

The top-level API remains the stable notebook-facing surface. The package also exposes focused modules so the implementation can be split without changing user workflows:

| Module | Contains |
|---|---|
| `ecat.objects` | `echem`, `cv`, `ca`, `cp`, `dpv` |
| `ecat.io` | General loading access plus CV-specific text and Excel import helpers |
| `ecat.parsers` | Shared parser helpers for timestamps, durations, quiet-time rows, and experiment labels |
| `ecat.reference` | Reference-potential helper functions |
| `ecat.collection` | Filtering, sorting, grouping, and collection summary helpers |
| `ecat.plotting` | Plotting, plotting style, scale-bar helpers, and object-list display helpers |
| `ecat.analysis_cv` | CV normalization and current-scaling helpers |
| `ecat.analysis_batch` | Batch and advanced analysis workflows |
| `ecat.export` | Data/table export helpers |
| `ecat.app` | Notebook-friendly launcher for the local eCAT app |
| `ecat.metadata` | Metadata and label-formatting helpers |
| `ecat.options` | Option dataclasses, defaults, and option discovery |
| `ecat.utils` | Shared internal numeric, unit, and display helpers |
| `ecat.simulation` | Preview simulation and fitting helpers |

Public function names, argument names, and return shapes should match the top-level API. New notebooks and beta documentation should use `import ecat as e` or the focused modules listed above; `ecat.core` is not part of the beta API.

## Core Objects

| Name | Purpose | Common use |
|---|---|---|
| `echem` | Base electrochemistry object for imported time-series experiments. | `e.echem.from_file(path)` |
| `cv` | Cyclic voltammetry object with CV-specific analysis methods. | `cv_obj.peak_current(...)` |
| `ca` | Chronoamperometry object. | `ca_obj.plot()` |
| `cp` | Chronopotentiometry object. | `cp_obj.plot()` |
| `dpv` | Differential pulse voltammetry object. | `dpv_obj.peak_potential(...)` |

## Loading Data

| Name | Purpose | Options |
|---|---|---|
| `echem.from_file(path, options=None)` | Load one file and promote it to the detected object type when supported. | `describe_options("get_data")` |
| `parse_file(path, options=None)` | Load one file and return the standardized parser contract (`ParseResult`) without returning the eCAT object. | `describe_options("get_data")` |
| `get_data(options=None)` | Load supported electrochemistry files from a folder. | `describe_options("get_data")` |
| `get_data_from_excel(file_path, options=None)` | Create eCAT objects from an eCAT Excel workbook, or fall back to curated Excel CV header parsing when no `manifest` sheet is present. | `describe_options("get_data")` |

The built-in text importer accepts `.txt`, old BASI-Epsilon `.dat`, and EC-Lab ASCII `.mpt` files. BioLogic binary `.mpr` files raise `UnsupportedFileFormatError` before they reach the table parser; export `.mpt`, convert externally, or provide a direct custom reader. Folder loaders skip `.mpr` files with a clear relative-path diagnostic and continue loading supported text files.

Folder loaders return an empty list (`[]`) when no supported files are found or no files can be converted, so notebook loops and filters can safely consume the result without a separate `None` check. By default, folder imports keep subfolders together and order objects within each subfolder by acquisition timestamp using `sort keys = ["subfolder", "timestamp"]`; pass explicit `sort keys` to override that ordering.

`get_data()` and `echem.from_file()` now support a `custom parser` hook for filename-derived metadata and a `parser settings` dictionary for parser behavior. Use `custom parser mode="merge"` to fill only missing filename metadata, or `custom parser mode="override"` to replace the built-in filename parser result. File-derived metadata still wins by default; set `parser settings={"prefer file metadata": False}` only when you explicitly want the custom parser to replace file metadata such as scan rate. Parser settings also accept canonical `gases` and `solvents` lists plus `compound stopwords`.

Every loaded object exposes `obj.parse_result`, a `ParseResult` with a consistent parser contract: `.data`, `.units`, `.technique`, `.software`, `.metadata`, `.raw_metadata`, `.warnings`, `.source`, and `.parser`. Use `parse_file(...)` when you want that contract directly for parser debugging or importer tests. Normal analysis workflows should use `echem.from_file(...)` or `get_data(...)`.

Text importers use the parser contract before object promotion for tested CH, BASI, EC-Lab-style, limited NOVA ASCII, and generic numeric/header text paths. These importers preserve raw header/column metadata and nonfatal warnings on `obj.parse_result`. Header and filename scan rates are normalized to V/s before comparison; differences within 0.1% or `1e-6 V/s` are treated as rounding noise, while genuine mismatches warn and retain the measured header value. Generic files with only potential/current columns can promote as CV-like fallbacks; ambiguous generic files containing time, potential, and current columns remain generic `echem` objects unless a vendor or user-supplied technique marker resolves the technique. IviumSoft text and DPV text beyond existing CH/private-fixture coverage still need representative files before they are treated as beta-supported workflows.

## eCAT App

| Name | Purpose |
|---|---|
| `open_app(host="127.0.0.1", port=0, browser=False, inline=False)` | Start the local eCAT app from Python or a notebook and return the local URL. |

The app uses optional dependencies. Install them with:

```bash
python -m pip install "ecat[app]"
```

For a local source checkout, use:

```bash
python -m pip install -e ".[app]"
```

Terminal users can also run:

```bash
ecat-app
```

Notebook users can run:

```python
import ecat as e

url = e.open_app()
```

By default, `e.open_app()` opens the native app window. Use `ecat-app --browser` or `e.open_app(browser=True)` to open a browser tab instead. Use `e.open_app(inline=True)` to display browser mode in a notebook iframe when the notebook environment supports it. Browser mode uses `8050` by default and automatically chooses the next available port if that port is busy; native-window mode uses a private available port by default.

### Filename Parsing And Recommended Names

For text imports, eCAT tries to recover useful metadata from filenames when the file itself does not provide it cleanly. The built-in filename parser looks for:

- `gas`, such as `Ar`, `N2`, `CO`, or `CO2`
- `solvent`, such as `MeCN`, `DMF`, `DMSO`, `DCM`, `THF`, or `H2O`
- `compounds` and paired `concentrations`
- `scan rate`

Recognized filename concentration units include:

- `M`
- `mM`
- `uM` or `μM`
- `nM`
- `L`
- `%`
- `equiv`
- `x`

Recognized pressure units for metadata and display include `Pa`, `bar`, `atm`, `Torr`, `mmHg`, and `psi`. SI prefixes are supported where they are natural for pressure, such as `kPa`, `MPa`, and `mbar`. eCAT preserves pressure units by default and converts between pressure units only when a target/display unit is requested explicitly.

Recognized scan-rate patterns include compact and spaced forms such as:

- `100mVs`
- `100 mV/s`
- `0.1Vs`
- `500uV/s`

Recommended filename style is compact, ordered, and explicit:

```text
100mVs_CO2_MeCN_1mMFe-tpyPY2Me_100mMPhOH_run01
CV_MeCN_Ar_0.1MTBAPF6_3mMFc_1mMFe_100mVs
```

Practical recommendations:

- use explicit concentration units in every concentration token
- keep each concentration immediately attached to its species when possible, for example `100mMPhOH`
- use canonical gas and solvent names consistently across a project
- prefer underscores or similarly clean separators over long free-text filenames
- treat human-readable space-delimited names as a fallback, not the primary naming convention

Important parser rules:

- bare lowercase `m` is not treated as molar because it is too ambiguous
- gas-fraction tokens such as `0.1CO2` are converted to `%`-style metadata
- mole-fraction tokens such as `0.8xD2O` are supported
- zero-concentration species such as `0mM HCO3` are treated as absent from normal `compounds` / `concentrations`; eCAT retains them separately as zero-concentration metadata for provenance, but concentration colorbars start at the first positive added concentration
- if the built-in parser is not enough, use `compounds`, `custom parser`, and `parser settings`

## Inspecting Objects

These methods are available on eCAT objects unless a technique-specific object documents otherwise.

| Method | Purpose |
|---|---|
| `obj.info()` | Print object metadata and basic import details. |
| `obj.stats()` | Return metadata and numeric ranges as a dictionary. |
| `obj.x(options=None)` | Return the selected x-axis data. |
| `obj.y(options=None)` | Return the selected y-axis data. |
| `obj.xy(options=None)` | Return selected x and y arrays or series together. |
| `obj.animate(options=None)` | Build an animation for one object using the shared plotting surface. |

| Name | Purpose |
|---|---|
| `show(value, options=None)` | Smart notebook display for one eCAT object, a flat object list, grouped objects, or a result object with `.show()`. |
| `show_objects(objects, options=None)` | Display a compact table of loaded objects. |
| `show_groups(groups, options=None)` | Display already-created groups from `group()` or `sort_and_group()`. |

Display helpers return `None` by default to avoid duplicate notebook output. `show(group)` delegates to `show_objects(group)` for one group, while `show(groups)` delegates to `show_groups(groups)` for nested grouped lists. Pass `{"return": True}` to return the displayed table when you need it for further work. Rich notebook output uses table captions for titled tables, while terminal/plain output keeps readable heading lines before the table. Numeric object-display values honor `sig figs`; recognized unit suffixes such as `(A)`, `(V)`, and `(s)` are displayed in the Value column. Use `show_objects(objects, {"columns": "available"})` to return the available column keys, `{"columns": "all"}` to show every non-internal column, or pass labels such as `"Scan Rate"` and aliases such as `"scan_rate"` in the column list.

## Plotting

| Name | Purpose | Options |
|---|---|---|
| `obj.plot(options=None)` | Plot one eCAT object. | `describe_options("plot")` |
| `cv_obj.plot_program(options=None)` | Plot a CV potential program as potential versus time; if no time column is stored, eCAT reconstructs time from scan rate and can prepend quiet time as a negative-time hold. | `describe_options("cv.plot_program")` |
| `multiplot(objects, options=None)` | Overlay multiple objects on one axes. | `describe_options("multiplot")` |
| `multimultiplot(groups, options=None)` | Plot multiple groups as separate multiplots. | `describe_options("multimultiplot")` |
| `multi_scatterplot(data, options=None)` | Plot one or more result-table metrics. | `describe_options("multi_scatterplot")` |
| `plotting_style(style=True)` | Apply notebook, publication, Savéant-inspired, or Matplotlib-default plotting styles. | `describe_options("plot")` |

Available style values are `"notebook"`, `"publication"`, `"saveant"` / `"savéant"`, and `False` / `"matplotlib"` / `"default"` to restore Matplotlib defaults.

Axis labels can use descriptive text or electrochemistry symbols through the plot option `symbol labels`. Use `{"symbol labels": True}` for compact labels such as `E`, `i`, `j`, `t`, and `Q`; use `False` for descriptive labels; use `"auto"` to follow the active plotting style. The Savéant style enables symbol labels by default. Normalized axes such as `i/ip0` and dimensionless normalization labels remain symbolic.

The package default plot convention is `IUPAC`. For CVs, pass `{"plot convention": "US"}` for the left-to-right axis orientation instead.

`multiplot()` auto-generates labels from differing object metadata for CV, DPV, CA, and CP overlays. Pass `{"labels": [...]}` when you want explicit trace labels instead.

Use a numeric `offset` as a constant vertical step between traces, or pass one explicit absolute offset per trace. Offset values use the displayed y-axis unit. For example, `{"y unit": "uA", "offset": 2}` plots offsets of `0`, `2`, and `4` uA for three traces, while `{"y unit": "uA", "offset": [0, 1, 4]}` uses those three offsets directly.

Scale bars are available through plot options rather than as a standalone top-level helper:

```python
cv_obj.plot({"scale bar": {"loc": "lower right", "length": 1e-6}})
e.multiplot(cvs, {"scale bar": {"loc": (-0.5, 0.0), "length": 5e-6}})
```

The scale-bar `length` is in the displayed y-axis unit. A tuple `loc=(x, y)` uses displayed data coordinates after unit conversion and scaling. Scale bars remove y ticks by default.

For gradient-colored multiplots, `gradient ...` options control how trace metadata maps to colors, while `colorbar ...` options control the displayed colorbar legend. Use `colorbar tick labels` (`endpoints`, `all`, or `none`) and `colorbar trace ticks` to control tick text and per-trace tick marks. With `gradient scale = "auto"`, scan-rate and positive concentration gradients use log spacing. Zero-concentration entries such as `0mM HCO3` are treated as absent/background traces rather than `+0` colorbar endpoints. Percent and equivalent concentration series follow the same concentration-gradient default. `gradient reverse` reverses trace color assignment; `colorbar reverse` flips only the displayed colorbar direction.

`multi_scatterplot()` works on analysis result tables and supports direct column control:

- `x column`: explicit x-axis column name or `"auto"` (default).
  - `auto` prefers `"x transformed"` then raw `"x"` columns before falling back to sensible x metadata.
- `y column`: explicit y-axis column name or `"auto"` (default).
  - `auto` prefers transformed and metric columns (`"kobs"`, `"TOFmax"`, `"ip"`, `"Ep"`).
- `y columns`: optional list of y columns to plot at once.
- `metric`: preferred metric name used by auto-resolution when `y column = "auto"`.
- `labels`: optional explicit trace labels, useful for concentration or scan-rate series.

Important compatibility detail: for fit-table inputs, `auto` can use transformed values when available (`x transformed`, `y transformed`), but this is data-internal and does not change the result of upstream fitting options (for example, `transform mode` is still controlled by the source fit call).

Notebook-style usage:

```python
# plot raw kobs from fit_rate output
e.multi_scatterplot(results_raw, {"x column": "H2O", "y column": "kobs", "metric": "kobs"})

# plot transformed data from fit helpers
e.multi_scatterplot(results_transformed, {"x column": "x transformed", "y column": "y transformed"})

# plot multiple columns simultaneously
e.multi_scatterplot(results, {"y columns": ["kobs", "TOFmax"], "labels": ["kobs", "TOFmax"]})
```

## Animation

| Name | Purpose | Options |
|---|---|---|
| `animate(obj_or_list, options=None)` | Animate one object or a list of objects through the normal eCAT plotting pipeline. | `describe_options("animate")` |
| `obj.animate(options=None)` | Convenience method for animating one object directly. | `describe_options("animate")` |
| `AnimationResult.show()` | Display the animation inline in a notebook when supported. | none |
| `AnimationResult.save(path, format=None, options=None)` | Save the animation to GIF or MP4. | none |

Animation is object-first: prefer `e.animate(cv_obj)` or `cv_obj.animate()` instead of overloading `plot()` or `multiplot()` for animation. Fully rendered frames reuse the existing plot and multiplot helpers, so titles, units, plot conventions, legends, colorbars, and scale bars match the static plotting surface.

Default behavior:

- single object: `trace mode="draw"`
- multiple objects with different scan rates: `schedule="simultaneous"` and `trace mode="draw"`
- multiple objects with matched scan rates: `schedule="staggered"` and `trace mode="instant"`
- `timing mode="auto"` uses physical timing when scan-rate or time metadata is available; otherwise it falls back to normalized timing
- `cycle=True`, `fps=20`, `stagger time=0.5`, and `end hold=2`
- quiet time is excluded unless `{"include quiet time": True}` is passed

Notebook example:

```python
result = e.animate(cv_obj, {"trace mode": "draw", "plot convention": "IUPAC"})
result.show()
result.save("cv.gif")
```

## Filtering And Grouping

| Name | Purpose | Options |
|---|---|---|
| `filter(objects, filter_keys, options=None)` | Include or exclude objects by metadata. | `describe_options("filter")` |
| `sort(objects, sort_keys, options=None)` | Sort objects by metadata. | `describe_options("sort_group")` |
| `group(objects, group_keys, options=None)` | Group objects by metadata. | `describe_options("sort_group")` |
| `sort_and_group(objects, sort_keys=None, group_keys=None, options=None)` | Sort and group in one workflow. | `describe_options("sort_group")` |
| `group_summary(objects, options=None)` | Summarize grouped object metadata. | `describe_options("group_summary")` |
| `get_available_filter_values(objects, keys=None)` | Inspect possible values before filtering. | none |

Filter `logic` combines different filter keys. For example, `{"logic": "AND"}`
requires each top-level key such as `gas`, `species`, and `replicate` to match.
Within membership-style keys, eCAT now uses concentration-series-friendly
defaults:

- `compounds`, `concentrations`, and `species` lists require all listed values
  by default.
- Scalar metadata keys such as `gas`, `solvent`, `type`, and `scan rate` keep
  list-as-any behavior.
- Use `{"any": [...]}` or `{"all": [...]}` inside one key when the per-key
  logic should be explicit.

Examples:

```python
# Must contain all listed compound identities.
baseline = e.filter(cvs, {"compounds": ["[Co]", "Fc", "Zn(cyclen)", "H2O"]})

# Concentration-aware species matching; all listed species must be present.
series = e.filter(cvs, {
    "species": ["1mM[Co]", "3mMFc", "1mMZn(cyclen)", "2.8MH2O"],
    "replicate": -1,
})

# Match any one of several possible added species.
with_additive = e.filter(cvs, {"species": {"any": ["PhOH", "H2O"]}})

# Scalar metadata lists remain any-of.
gas_subset = e.filter(cvs, {"gas": ["Ar", "CO2"]})
```

## CV Analysis

| Name | Purpose | Options |
|---|---|---|
| `cv_obj.peak_potential(guess_or_options=None, options=None)` | Locate a selected peak potential. A numeric first argument is shorthand for `guess potential`. | `describe_options("cv.peak_potential")` |
| `cv_obj.peak_current(guess_or_options=None, options=None)` | Measure peak current with baseline/tangent handling. A numeric first argument is shorthand for `guess potential`. | `describe_options("cv.peak_current")` |
| `cv_obj.peak_width(guess_or_options=None, options=None)` | Measure tangent-corrected full peak width at a fractional peak-current level. A numeric first argument is shorthand for `guess potential`. | `describe_options("cv.peak_width")` |
| `cv_obj.half_peak_potential(guess_or_options=None, options=None)` | Estimate half-peak potential. A numeric first argument is shorthand for `guess potential`. | `describe_options("cv.half_peak_potential")` |
| `cv_obj.half_wave_potential(guess_or_options=None, options=None)` | Estimate half-wave potential from paired peaks. A numeric first argument is shorthand for `guess potential`. | `describe_options("cv.half_wave_potential")` |
| `cv_obj.current_at_potential(potential, options=None)` | Extract current at a requested potential. | `describe_options("cv.current_at_potential")` |
| `cv_obj.plateau_current(options=None)` | Analyze plateau current for one CV. | `describe_options("cv.plateau_current")` |
| `cv_obj.trim(window_or_options=None, options=None)` | Return one trimmed CV copy. A two-value first argument is shorthand for `potential window`. | `describe_options("trim")` |
| `trim(cvs, window_or_options=None, options=None)` | Return trimmed CV copy/copies while preserving list or grouped-list shape. | `describe_options("trim")` |
| `normalize(cvs, options=None)` | Return normalized CV copy/copies. CV-only. | `describe_options("normalize")` |
| `cv_obj.normalize(options=None)` | Normalize one CV in place and return itself. | `describe_options("cv.normalize")` |
| `normalize_current(cvs, options=None)` | Return CV copy/copies with `i/ip0` current normalization. | `describe_options("normalize_current")` |
| `scale_current(cvs, options=None)` | Scale currents against reference currents or manual values. | `describe_options("scale_current")` |
| `cv_obj.filter(options=None)` | Return a filtered CV copy using a recorded SciPy-backed filter. | `describe_options("cv.filter")` |

Single-CV analysis helpers such as `peak_potential`, `peak_current`, `peak_width`, `half_peak_potential`, `half_wave_potential`, `wave_info`, and `current_at_potential` return `CVAnalysisResult`, an `AnalysisResult` child that remains dictionary-compatible. Existing notebook access such as `result["ip"]`, `result["Ep"]`, or `result["width"]` is preserved. New code can also use `result.primary` for the main base-unit scalar, `result.table` for a tidy display table with columns such as `Metric` and `Value`, and `result.show()` for pretty or plain printing. Human-facing analysis tables are formatted at display time: `sig figs`, `x unit`, and `y unit` passed to `result.show(...)` control the rendered table without changing the base-unit scalar values. Display formatting preserves trailing significant zeros, so default four-sig-fig values can render as `100.0 mV` or `8.000 μA`. Automatic display scaling chooses a unit only when the largest finite absolute displayed value falls in a readable range, `1 <= abs(value) < 1000`; for example, `0.16 V` can display as `160.0 mV`, but `-1.474 V` stays in V instead of becoming `-1474 mV`. Plot axes keep their separate plotting convention and do not autoscale potential axes by default.

Peak and wave defaults are intended for exploratory use. Single-feature helpers such as `peak_potential()`, `peak_current()`, `peak_width()`, `half_peak_potential()`, and `peak_info()` choose the largest detected feature when `segment` and `guess potential` are omitted. Paired-wave helpers such as `half_wave_potential()` and `wave_info()` default to segment `1` paired with segment `2` when `segments` is omitted. For final analysis, specify `segment`, `segments`, `guess potential`, or `exact potential` so the notebook records the intended feature. The default `peak kind = "both"` considers maxima and minima because current sign conventions determine which extrema are cathodic or anodic; set `peak kind = "infer"` to map increasing selected current to maxima and decreasing selected current to minima, or set `peak kind = "max"` / `"min"` to force one kind. CV and DPV peak/wave helpers accept a numeric first argument as shorthand for `guess potential`, so `cv.peak_current(-1.5, {"segment": 1})` is equivalent to `cv.peak_current({"guess potential": -1.5, "segment": 1})`.

`peak_current()` defaults to `peak fallback = "highest current"`: when no eligible local extremum is detected, it uses the largest absolute current in the selected segment and reports the fallback source. This is useful for exploratory extraction but can select a scan vertex or boundary when the intended peak is unresolved. Use `peak fallback = None` (or `"none"`) for strict analysis that should fail instead, or `peak fallback = "guess potential"` to measure at the supplied guess when detection fails. Batch workflows that delegate to `peak_current()`, including `sevcik_analysis()` and `fit_peak_current()`, preserve `peak kind`, `peak fallback`, and `plot peak potential`; per-CV locations can be supplied with `guess potentials` or `exact potentials`.

`peak_width()` reports tangent-corrected full peak width. It reuses `peak_current()` peak and tangent-baseline options, then subtracts the tangent and finds the scan-order `E leading` and `E trailing` crossings at `level` times the tangent-corrected peak current. The default `level = 0.5` gives full width at half peak current. Crossings are linearly interpolated internally; there is no public `interpolate`, `side`, or `mode` option.

Automatic peak prominence uses a conservative global estimate when no `guess potential` is provided. With a guess, eCAT estimates the automatic prominence from a local potential window centered on the guess, using 20% of the selected potential span by default; explicit `peak prominence` values still override this behavior. Set `noise window = None` to disable Savitzky-Golay smoothing for peak detection.

`trim(...)` follows the same options-dict pattern with a small shorthand for the common case. `cv.trim([-1.5, 0])` trims one CV, while `e.trim(cvs, [-1.5, 0])` trims a list or grouped list of CVs. The default `mode="expand"` preserves connected CV waveforms; use `mode="pointwise"` for a hard pointwise crop or `mode="strict"` to raise when a requested window would disconnect the scan.

`cv.filter(...)` is copy-first and never runs automatically. The default is Savitzky-Golay filtering; supported methods are `savgol`, `gaussian`, `median`, `butterworth`, and `moving average`. The returned CV records the resolved filter settings in `filter_metadata` and `processing_history`, single-object `show()` reports the active filter, and eCAT Excel workbooks preserve the processing history across import. `inplace=True` is available when mutation is intentional. Filtering can shift peak positions, alter peak currents and derivatives, and bias FOWA or kinetic fits, so keep the raw CV and report the filter settings whenever filtered data are analyzed.

Imported numeric data are stored in canonical units whenever eCAT knows the source units: potential in `V`, current in `A`, time in `s`, and charge in `C`. This applies to vendor text parsers, generic/header text parsing, and eCAT Excel manifest workbooks with edited unit rows; the source units are retained in parser metadata when available. Display scaling belongs in plotting and printed tables, not import. For example, use `cv.plot({"y unit": "uA"})` or `e.multiplot(cvs, {"y unit": "uA"})` for microamp axes. Current density is also derived at display/analysis time: keep `electrode area` on the object, then request `{"y axis": "current density"}`. Import-time `"convert current"` and `"current density"` options are intentionally not supported.

Complex CV analyses accept per-CV potential lists through plural aliases. `describe_options` shows these as one row with `(s)`, such as `guess potential(s)`, `tangent potential(s)`, `non-catalytic cv(s)`, and `scan rate(s)`. Use `guess potentials`, `exact potentials`, `tangent potentials`, `peak potentials`, `non-catalytic guess potentials`, or FOWA `redox potentials` when each CV needs its own value. FOWA also accepts `wave ranges` as one `[min, max]` raw-potential window per catalytic CV; singular `wave range` remains the shared-window form. Scalar values under either singular or plural spelling are broadcast. For `fit_peak_potential`, a scalar `guess potential` or scalar `guess potentials` seed follows the ordered CV series, while a list must contain one scalar value per CV. For paired-peak analyses such as `trumpet_analysis` and `nicholson_analysis`, a flat two-value `guess potential` is treated as a shared paired guess when there are more than two CVs; with exactly two CVs, use nested pairs like `[[g1a, g1b], [g2a, g2b]]` when each CV needs a paired guess. If both singular and plural spellings are supplied for the same option, eCAT raises an option error.

FOWA uses `fit=True` by default. Each transformed trace is fitted independently; if a trace has fewer than five usable fit points or its regression fails, eCAT warns, marks that row as `fit skipped`, leaves its kinetic values unavailable, and continues with the remaining CVs. Set `fit=False` to return and plot transformed FOWA traces without performing any regressions.

When overlaying analysis diagnostics on an existing CV plot or `multiplot`, use `{"plot": True, "plot CV": False, "new plot": False}`. This adds markers, tangent lines, and current-distance diagnostics to the active axes without redrawing the underlying CV trace.

`normalize(...)` is intentionally CV-only. It raises a clear error for CA, CP, DPV, or generic objects because physical dimensionless normalization is not the same workflow for those techniques.

For dimensionless CV normalization, `normalize(...)` automatically uses stored CV metadata when explicit options are omitted. Temperature defaults to `cv.temperature` for either normalized axis. For current-axis normalization, electrode area defaults to `cv.electrode_area` and scan rate defaults to `cv.scan_rate`. Concentration is explicit by default. If `C`/`C unit` are omitted, pass `species` to resolve the concentration from the CV's exact `compounds` / `concentrations` metadata pair:

```python
normalized = e.normalize(cv, {"E0": -1.0, "D": 1e-5, "species": "[Co]"})
```

Species matching is exact; use the same spelling, case, and brackets shown in `cv.compounds`. Explicit `C` and `C unit` override `species`.

Normalized CVs store plain data columns named `Dimensionless Potential` and `Dimensionless Current`. Default axis labels use compact symbols: `θ` for dimensionless potential, `Φ` for homogeneous dimensionless current, and `χ` for heterogeneous dimensionless current. The full equations are shown by `normalize(..., {"print": True})`.

`normalize_current(...)` is separate from physical dimensionless normalization. It creates `i/ip0` from the raw `Current` column; it does not compute `Φ/Φp0`. With `{"plot all": True}`, eCAT plots the normalized `i/ip0` overlay. Add `{"plot reference diagnostic": True}` to first plot the reference-CV peak-current diagnostic used to determine `ip0` when `ip0` is extracted from reference CVs. Multiple distinct reference CVs are shown together on one reference diagnostic plot. Manual `ip0` values skip the reference diagnostic.

For CA current/charge overlays, `ca.plot({"plot charge": True})` keeps current on the primary axis and cumulative charge on a styled secondary axis. A scalar `y unit`, such as `"uA"`, controls current while charge remains automatically scaled; use `{"y unit": ["uA", "mC"]}` to set both axes explicitly. `invert y axis` inverts both axes. The optional `invert current axis` and `invert charge axis` settings override that shared choice independently; their default `None` means inherit the shared setting.

## Other Technique Methods

| Name | Purpose | Options |
|---|---|---|
| `dpv_obj.peak_potential(guess_or_options=None, options=None)` | Locate a DPV peak potential. A numeric first argument is shorthand for `guess potential`. | `describe_options("dpv.peak_potential")` |
| `ca_obj.charge(options=None)` | Integrate CA current to cumulative charge and optionally plot target-charge diagnostics. | `describe_options("ca.charge")` |
| `ca_obj.time_at_charge(charge=None, options=None)` | Find when a CA trace reaches a target charge. | `describe_options("ca.time_at_charge")` |
| `ca_obj.current_at_time(time=None, options=None)` | Interpolate CA current at a requested time, with optional corrected-current handling and a printed metric table. | `describe_options("ca.current_at_time")` |
| `ca_obj.average_current(time_range=None, options=None)` | Compute the time-weighted average CA current over a window. | `describe_options("ca.average_current")` |
| `ca_obj.rate_at_time(time=None, options=None)` | Convert CA current at a time to electron flow and, when stoichiometry/catalyst amount are provided, molecular rate and TOF. | `describe_options("ca.rate_at_time")` |
| `ca_obj.average_rate(time_range=None, options=None)` | Convert average CA current over a window to electron flow and optional molecular rate/TOF. | `describe_options("ca.average_rate")` |
| `cp_obj.get_cycles(options=None)` | Split CP data into charge/discharge cycles. | `describe_options("cp.get_cycles")` |
| `cp_obj.cycle_info(options=None)` | Summarize CP cycle capacity, efficiency, and potential metrics. | `describe_options("cp.cycle_info")` |
| `cp_obj.plot_cycles(options=None)` | Plot selected CP cycles. | `describe_options("cp.plot_cycles")` |
| `cp_obj.cycling_plot(options=None)` | Plot CP capacity and efficiency versus cycle number; accepts the same `cycles` selection forms as `plot_cycles()`. | `describe_options("cp.cycling_plot")` |

DPV pulse metadata follows the same unit convention as CA/CP metadata: object
attributes and `stats()` keys are unitless SI values, while printed tables and
auto plot subtitles put autoscaled units in the values, such as `10 mV` or
`50 ms`.

## Advanced Analysis

Notebook-facing analysis reports follow one shared order: Parameters,
Equations, Summary, then Data only when `print all=True`. Empty sections are
omitted, while complete numeric and diagnostic detail remains available on the
returned result. See the [analysis output contract](analysis_output_contract.md).

| Name | Purpose | Options |
|---|---|---|
| `fowa(cvs, options=None)` | Foot-of-the-wave analysis for catalytic CVs. | `describe_options("fowa")` |
| `sevcik_analysis(cvs, options=None)` | Sevcik-style peak-current trend analysis. | `describe_options("sevcik_analysis")` |
| `reversibility_analysis(cvs, options=None)` | Series-level electron-transfer and chemical-reversibility assessment for one bulk or surface-confined condition. | `describe_options("reversibility_analysis")` |
| `surface_coverage_analysis(cvs, options=None)` | Surface coverage and electroactive loading from independent peak-slope and tangent-corrected charge methods. | `describe_options("surface_coverage_analysis")` |
| `trumpet_analysis(cvs, options=None)` | Trumpet analysis from paired peak potentials. | `describe_options("trumpet_analysis")` |
| `nicholson_analysis(cvs, options=None)` | Nicholson-style heterogeneous electron-transfer analysis. | `describe_options("nicholson_analysis")` |
| `tafel_analysis(cv_or_list, TOF_max, thermodynamic_potential, redox_potential, options=None)` | Tafel-style turnover-frequency curves for one CV or a CV series; multi-CV calls reuse the shared `multiplot()` label, color, gradient, legend, and colorbar options. | `describe_options("tafel_analysis")`; also accepts common `multiplot` styling options |
| `fit_model(x_or_result, y=None, model=None, options=None)` | Fit direct scatter models such as linear, power, power offset, exponential, Michaelis-Menten, and logistic. | standalone options |
| `fit_rate(df_or_result, options=None)` | Fit rate or transformed result tables; accepts a DataFrame or an `AnalysisResult` and uses `.table` by default. | `describe_options("fit_rate")` |
| `plateau_current(cvs, options=None)` | Batch plateau-current workflow. | `describe_options("plateau_current")` |
| `fit_peak_potential(cvs, options=None)` | Fit peak-potential trends. | `describe_options("fit_peak_potential")` |
| `fit_peak_current(cvs, options=None)` | Fit peak-current trends. | `describe_options("fit_peak_current")` |

### Reversibility Decision Tree

`reversibility_analysis()` accepts one chemical condition measured at at least
three distinct scan rates. Five or more rates are recommended. Replicates are
retained in `result.table`; trend calculations use the mean at each scan rate
and retain replicate counts and standard deviations in
`result.diagnostics["rate means"]`. Mixed conditions raise an error with a
prompt to group first with `e.group(...)`.

The required `phase` model is explicit. It defaults to `"bulk"`; use
`"surface"` for adsorbed or surface-confined couples. There is no automatic
phase inference.

For `phase="bulk"`, eCAT evaluates evidence in this order:

1. Extract paired tangent-corrected peaks, half-peak potentials, full
   half-peak widths, `E1/2`, `Delta Ep`, cathodic/anodic segment numbers, and
   `|ipa/ipc|` for every CV. Physical branch names are resolved from potential
   scan direction, so segment 1 may be cathodic or anodic.
2. Convert `n Delta Ep` values to Nicholson `psi`, then calculate
   `Lambda = sqrt(pi) psi`. Matsuda-Ayabe labels are `reversible` for
   `Lambda >= 15`, `quasi-reversible` for
   `10^[-2(1+alpha)] < Lambda < 15`, and `irreversible` below that lower
   boundary. `psi` and `Lambda` remain available outside the recommended
   Nicholson rate-estimation range whenever the conversion is defined.
   Nicholson-derived rate estimates are used only for `0.1 <= psi <= 7`;
   the Matsuda-Ayabe boundary, not the Nicholson fitting window, controls the
   region label. The printed equations explicitly label the Nicholson
   peak-separation approximation
   `psi = [-0.6288 + 0.0021(n Delta Ep / mV)] /
   [1 - 0.017(n Delta Ep / mV)]` and the separate Matsuda-Ayabe
   classification thresholds.
3. Resolve `D` from the explicit `D` option first. If `D` is omitted and exact
   `species`, concentration, electrode area, and temperature information are
   available, fit both peak-current branches against `sqrt(scan rate)` and
   calculate two Sevcik `Dapp` estimates. They must have `R2 >= min r2` and
   agree within `agreement tolerance` before their mean is used. The Sevcik
   diffusion equation is printed only when this automatic estimate is
   attempted.
4. Report Nicholson `k0` in the eligible region. A high-scan-rate trumpet
   estimate is also retained when its branch slopes are physical. If both
   estimates exist, both are reported and disagreement beyond the configured
   tolerance produces a warning. A series that remains reversible through its
   fastest scan reports a lower bound based on `Lambda = 15`, not a falsely
   precise fitted value. The `k0 = Lambda sqrt(D n F scan_rate / RT)`
   conversion is printed only when `D` is available. Otherwise the summary
   explains that `D`, or the area and concentration/species metadata needed
   for a Sevcik estimate, must be supplied.
5. A candidate irreversible series is only promoted to
   `irreversible behavior indicated` when the independent asymptotes agree:
   `|Ep-Ep/2| = 1.857 RT/(alpha nF)` and an `Ep` shift of
   `2.303 RT/(2 alpha nF)` per decade of scan rate. Otherwise the cautious
   conclusion is a quasi-reversible/irreversible transition. These
   irreversible-asymptote equations are printed only when the series reaches
   that candidate branch of the decision tree.

For `phase="surface"`, diffusion, Sevcik, and Nicholson are not used. eCAT
requires peak current to be linear in scan rate and compares peak separation
with an effective zero-separation tolerance of
`max(peak separation tolerance, 3 * median potential increment)`. Surface
Laviron fitting is eligible only when at least two scan rates have
`n Delta Ep > 200 mV`; its standard electron-transfer rate is reported in
`s^-1`, not `cm/s`.

Chemical reversibility is always reported separately from electron-transfer
kinetics. All tangent-corrected $|i_{p,\mathrm{a}}/i_{p,\mathrm{c}}|$ ratios within
`current ratio tolerance` of unity give
`chemically reversible over observed timescale`. Ratios outside that band,
especially when they move toward unity at faster scan rates, give
`coupled chemistry indicated`. Sparse or contradictory evidence gives
`indeterminate`. The default current-ratio tolerance is `0.10`; it is
independent of the `0.25` default `agreement tolerance` used to compare two
estimates of quantities such as `D` or surface coverage. Both can be changed
globally with `set_defaults(...)`. The result records the observed ratio range,
maximum deviation from unity, Matsuda-Ayabe `Lambda` range and region counts,
and number of Nicholson-eligible scan rates. These labels describe the
observed timescale and do not assign a chemical mechanism.

With `plot=True`, the default reversibility figure contains two vertically
stacked panels sharing a logarithmic scan-rate axis: $n\Delta E_p$ and
$|i_{p,\mathrm{a}}/i_{p,\mathrm{c}}|$. `plot all=True` adds a separate
$E_{p,\mathrm{c}}$/$E_{p,\mathrm{a}}$ scan-rate diagnostic. Pretty printing
renders all analysis equations and scientific table symbols through the same
LaTeX/HTML display path used by Sevcik and FOWA; `pretty print=False` uses
plain-text equations and column names. Bulk reports always label the
Nicholson Peak-Separation Conversion and Matsuda-Ayabe Classification;
Sevcik Diffusion Estimate, Electron-Transfer Rate Conversion, and
Irreversible-Asymptote Verification blocks appear only when their evidence
paths apply.

With `print all=True`, bulk reversibility prints a compact evidence table with
scan rate, `E1/2`, `n Delta Ep`, `|ipa/ipc|`, `psi`, `Lambda`, electron-transfer
region, and Nicholson-use status. Surface-confined mode instead prints scan
rate, `E1/2`, `n Delta Ep`, absolute cathodic and anodic peak currents,
`|ipa/ipc|`, and the surface region. `Name` is added as the first column only
when replicate rows have the same displayed scan rate. Detailed peak,
half-peak-width, branch, eligibility, and replicate columns remain in
`result.table` and `result.diagnostics` without crowding the printed table.

`surface_coverage_analysis()` mirrors the scan-rate-series structure of
Sevcik analysis but uses the surface-confined equations
`ip = n^2 F^2 A Gamma scan_rate / (4 RT)` and `Q = n F A Gamma`.
The slope method fits each selected branch independently. The charge method
integrates tangent-corrected current with respect to potential and divides by
scan rate. The result reports `Gamma` in `mol/cm^2` and total electroactive
loading in `mol`; loading remains available when electrode area is omitted.
Slope, charge, and branch estimates are all retained, and disagreements beyond
`agreement tolerance` warn rather than being silently averaged. Use an
explicit `integration range` when the automatic tangent-baseline return search
cannot resolve both sides of a peak.

The public quickstarts keep the two physical models separate. Notebook
`07_reversibility_analysis.ipynb` uses bundled real Fe/Fc CVs; notebook
`08_surface_confined_cv.ipynb` imports the pre-generated
`examples/data/surface_coverage_cv/surface_confined_cv_series.xlsx` workbook.
The workbook's adjacent generator records its known coverage and loading but is
not executed by the notebook.

`sevcik_analysis()` is now strictly the bulk diffusion-controlled Sevcik
workflow. Scan-rate series always fit peak current against `sqrt(scan rate)`;
the former configurable scan-rate exponent is not an option. Use
`surface_coverage_analysis()` for the linear-in-scan-rate surface-confined
relationship.

`trumpet_analysis()` is independent of the initial scan direction and segment
order. It classifies decreasing-potential segments as cathodic and
increasing-potential segments as anodic, then applies the cathodic slope to
`alpha` and the anodic slope to `beta`. The result table and fit labels use
those physical branch names, while `result.summary["segment selection"]`
records the resolved cathodic/anodic segment numbers and assignment source.
Mixed initial scan directions within one series are mapped per CV. Flat,
nonmonotonic, same-direction, or otherwise unresolvable segment pairs raise a
targeted branch-assignment error. CV-like custom objects without accessible
segment data fall back only when their two fitted peak-potential slopes have
unambiguous opposite signs.

`plateau_current()` accepts one CV, a flat list of CVs, or nested lists of CVs. With the default `{"group mode": "auto"}`, a flat list is grouped with `e.group(..., "species")`, so concentration/composition series produce one plateau row per condition while scan-rate series within a condition are used for plateau validation. Use `{"group mode": "as given"}` to force one flat list to be treated as a single validation group, `{"group mode": "each"}` to analyze each CV independently, or pass nested lists when you want explicit validation groups. Printed output contains Plateau Current Parameters, the symbolic Plateau Current Equations, and a compact Plateau Current Summary built from object-summary differences plus plateau-analysis columns. `print all=True` adds Plateau Current Data. Plateau-current equations stay symbolic; the displayed parameter and result labels carry symbols such as `Catalyst Electrons (n)`, `Turnover Electrons (n′)`, `Temperature (T)`, `Diffusion Coefficient (D)`, `Electrode Area (S)`, and `Catalyst Concentration (C)`. Resolved parameter values stay in the parameter/summary tables. Peak-current extraction details are retained in `result.diagnostics` rather than printed as separate peak-potential/current tables. In `plot all` diagnostics, CV overlays default to a single combined `i/ip0` plot for the catalytic CVs and available non-catalytic reference CVs when `ip0` can be resolved; otherwise eCAT falls back to current and records the fallback in `result.warnings`. Peak-current guide marks are redrawn on the normalized overlay using the same diagnostic style as FOWA. Plateau validation plots are emitted once per condition with enough scan-rate points and still use current versus `sqrt(scan rate)` because that diagnostic tests scan-rate independence of the limiting current. Multi-condition `plateau_current()` display tables keep a hidden numeric `attrs["full_results_df"]` table with `kobs` and concentration columns, so the result can be passed directly to `fit_rate(...)` even when context columns are not visible.

`fowa(..., {"print": True})` prints a vertical FOWA Parameters table for shared settings and reference information, the symbolic FOWA Equations with definitions, and the compact FOWA Summary table. `print all=True` adds the fuller FOWA Data table. It does not print a second equation with the `n` values substituted. The transformed FOWA plot labels the x-axis with the actual redox-reference convention used for the calculation: `E1/2` for `"redox mode": "half wave"`, `Ep/2` for `"half peak"`, `Eredox` for `"manual"`, and a generic `Eref` only when modes are mixed. The x-axis uses literature-style `n` in the exponent, and the printed equation defines `n = n_cat = catalyst redox-wave electron count`; `n′ = n_turn` appears in the slope-to-`kobs` equation rather than in the x-axis transform. `{"pretty print": False}` preserves the same section order using plain text.

Advanced analysis helpers return `AnalysisResult`-style objects. Use `.table` for the primary table, `.summary` for workflow metadata, `.fits` / `.fit_table` where fitting applies, `.axes` for plots, `.units` for result units, and `.warnings` / `.diagnostics` for extra detail. FOWA and plateau-current results no longer pretend to be DataFrames; use `result.table.columns`, `result.table.loc[...]`, `result.table.attrs`, or `result.summary["kobs"]` for scalar plateau values. When an analysis has a specialized display formatter, `result.show({"sig figs": ...})` applies that formatter at display time while leaving exported/raw tables numeric where they were numeric. For export, `result.to_csv(...)` writes the primary table using pandas CSV semantics, while `result.to_excel(...)` writes a workbook with the primary `table` sheet plus available metadata sheets such as `summary`, `fit_table`, `fits`, `warnings`, `units`, and `diagnostics`.

Row-level analysis reports use visible context as their identity. A unique scan
rate or concentration series therefore omits long object names; if duplicate
displayed contexts remain, `Name` is inserted first. Numeric identity is
evaluated with the active `sig figs` setting, so acquisition values that print
as the same scan rate are treated as replicates. Aggregated plateau and grouped
results use `Condition`, `Group`, or `Branch` rather than `Name`. Returned data
can still retain names for traceability even when the printed table omits them.

`nicholson_analysis()` returns an `AnalysisResult` with `result["data"]` and `result.table` for the Nicholson point table and `result["summary"]` / `result.summary` for equation metadata, fit statistics, and kinetic values. With `{"print": True}`, it prints Nicholson Analysis Parameters, Equations, and Summary; `print all=True` adds the compact Nicholson Analysis Data table. With `{"plot all": True}`, it produces one CV diagnostic plot and one Nicholson fit plot.

`fit_rate`, `fit_peak_current`, and `fit_peak_potential` use the same direct model fitter as `fit_model`. Use `fit model` to choose the model and `fit init`, `fit bounds`, `fit residual`, `fit max evals`, and `fit indices` to control the fit. Use `fit line range` only to extend or shorten the plotted fit line after fitting; it does not change selected points, fitted parameters, residuals, R2, RMSE, or fit-table `fit x min` / `fit x max` values. Standalone `fit_model` also accepts concise context-specific names such as `model`, `init`, `bounds`, `residual`, `band`, and `indices`.

Scatter-fit helpers return `ScatterFitResult`, an `AnalysisResult` child, rather than tuples. Use `.table` for plotted/result data, `.fits` for fit parameters, `.fit_table` for the human fit-statistics table, and `.summary` for workflow metadata. Tuple unpacking such as `data, fits = e.fit_rate(...)` is not part of the beta API.

Scatter-fit plots use matching point and fit-line colors by default. This applies to `fit_model`, `fit_rate`, `fit_peak_current`, `fit_peak_potential`, `sevcik_analysis`, `trumpet_analysis`, and `multi_scatterplot`. Pass `fit color` to override the fit line explicitly; it may be a scalar or a list consumed in plotted-fit order. `fit line range` accepts one `[x_min, x_max]` pair, a list consumed in fit order, or a dict keyed by fit label; `None` leaves a side open. Use `fit band` to add shaded uncertainty around fitted model lines: `None` / `"none"` disables bands, `"confidence"` draws the uncertainty in the fitted mean curve, `"prediction"` includes residual scatter for a future observation, and `"both"` draws both. `fit band level` defaults to `0.95` for 95% bands. Bands use the same x-domain as the plotted fit line.

When `fit_model(..., {"print": True})` is used, the default pretty output is adaptive. Simple fits print a concise `Field`/`Value` table with model settings, fit statistics, fitted parameter names/count, and fitted values with standard errors inline when available. More constrained or complex fits print separate `Fit Model Details` and `Fit Model Parameters` tables, including initial values, bounds, final values, and standard errors. Auto switches to details for explicit `fit init`, explicit `fit bounds`, custom formula/callable models, constrained models such as power offset/logistic/Michaelis-Menten, three or more parameters, or parameters at a bound. Use `{"print fit": "summary"}` or `{"print fit": "details"}` to force the style. Printed equations are the general model equations so parameter meanings remain visible; plot labels use the fitted equation with numeric values, formatted by `sig figs`. `fit_rate(..., {"print": True})` uses the same human table. Transforms such as `{"transform mode": "log-log"}` only change the coordinates being fit; they do not change or relabel the selected fit model. Use `{"fit model": "power"}` when you want a direct power-law model. Use `{"pretty print": False}` for a compact dictionary-style summary. eCAT warns when fit points equal the fitted-parameter count and gives a stronger underdetermined warning when fewer points than parameters are supplied.

Fitting helpers keep row selection and fit-line drawing separate. `fit indices` is row/position based; `[start, stop]` follows Python slicing, so it includes `start`, excludes `stop`, and accepts `None` on either side. A nested list such as `[[0, 3], [6, 9]]` performs one fit across disconnected row windows. A dictionary creates separate named fits, and each value may itself contain disconnected windows, for example `{"early": [0, 3], "tail": [[6, 9], [9, None]]}`. The selected rows determine fitted parameters, while result tables still report predictions and residuals for the original rows where applicable. `fit line range` changes only the displayed x-domain.

SciPy optimizer controls remain explicit. `fit method` accepts `auto`, `lm`, `trf`, or `dogbox`; common controls include `fit sigma`, `fit absolute sigma`, `fit check finite`, `fit nan policy`, and `fit jac`. Advanced `scipy.optimize.curve_fit` keywords can be supplied in `curve fit options`. eCAT retains ownership of `p0`, `bounds`, and `full_output`, which are controlled through `fit init`, `fit bounds`, and the result object.

`fit_model` also accepts custom models. Pass a callable with signature `f(x, param1, param2, ...)`, or pass a restricted formula string using `x` and fitted parameter names:

```python
quad = e.fit_model(
    scatter_result,
    model="k0 + k1*x + k2*x^2",
    options={"init": {"k0": 0, "k1": 1, "k2": 0}},
)
```

The same custom model forms work through `fit_rate`, `fit_peak_current`, and `fit_peak_potential` with the `fit model` option. Formula strings support arithmetic, `^` for powers, and selected math functions such as `exp`, `log`, `log10`, and `sqrt`; arbitrary Python code is not evaluated.

## Export

| Name | Purpose | Options |
|---|---|---|
| `save_data(objects, options=None)` | Export processed eCAT data to CSV or an eCAT Excel workbook. | `describe_options("save_data")` |

CSV export remains the simple flat-table path:

```python
e.save_data(objects, {
    "format": "csv",
    "folder path": "outputs",
    "file name": "processed_cv",
})
```

Excel export uses `format="xlsx"` and writes a round-trip-friendly workbook. The first sheet is `manifest`, a one-row-per-object metadata table using `show_objects`-style metadata columns. Additional sheets are grouped by object class, such as `cv`, `ca`, `cp`, and `dpv`. Each class sheet has three header rows before values: object/group id, column name, and unit. By default, objects with identical x-axis values share one x column inside their class sheet; pass `{"share x axes": False}` to write each object as a separate block. Referenced CVs export both the stored `Potential` axis and the active `Potential vs ...` axis by default.

```python
e.save_data(objects, {
    "format": "xlsx",
    "folder path": "outputs",
    "file name": "experiment_export",
    "metadata columns": ["reference source", "ir comp percent"],
    "data columns": "all",
    "share x axes": True,
})
```

`metadata columns` follows `show_objects` semantics: `"used"` is the default, `"all"` includes every available manifest column, and a list adds those columns to the default used-column selection. `data columns` is stricter: `"all"` is the default, `"stored"` exports only columns physically stored on the object, and a list exports exactly those data columns or raises a clear error if any are missing. If an Excel workbook has no `manifest` sheet, `get_data_from_excel(...)` falls back to header parsing and the same filename/header metadata parser settings used elsewhere in eCAT.

## Options And Defaults

| Name | Purpose |
|---|---|
| `describe_options(section=None, options=None)` | Show available option sections or a specific option table. |
| `get_defaults(section=None)` | Inspect current defaults. |
| `set_defaults(section_or_values, values=None)` | Set session/user defaults depending on the input form. |
| `reset_defaults(...)` | Reset default values. |
| `load_defaults(path=None)` | Load defaults from a TOML file. |

Option dataclasses such as `ImportOptions`, `PlotOptions`, `MultiplotOptions`, `PeakCurrentOptions`, and `FOWAOptions` are public for users who prefer typed options over dictionaries.

`describe_options("function_name")` reports function-specific option metadata: current defaults from eCAT's defaults system, dataclass-derived types, and function-specific choices/descriptions. Shared option names can therefore display different meanings by function, such as `mode` for `filter` versus `normalize`. Class-method names are accepted for notebook discoverability, for example `describe_options("cv.peak_current")`, `describe_options("dpv.peak_potential")`, `describe_options("ca.charge")`, and `describe_options("cp.get_cycles")`.

For `describe_options` itself, `print` controls whether anything is emitted, while `pretty print` only controls output style: `{"pretty print": False}` prints a plain text table, and `{"print": False}` suppresses output.

When an option is resolved automatically, the `Description` column distinguishes the source. Metadata retrieval is described explicitly, for example using a CV's stored `scan_rate`, `temperature`, `electrode_area`, `compounds`, or `concentrations`. Algorithmic `auto` behavior is described as automatic selection from data or plot context, such as unit scaling, tangent-region selection, gradient-color grouping, reference-wave detection, or result-column selection. Scientific inputs that are not inferred, such as some diffusion coefficients or formal potentials, are marked that way in the function-specific option table.

## Simulation Namespace

Simulation and fitting helpers live under the namespace:

```python
e.simulation
```

These helpers are intentionally namespaced rather than exported as top-level names. The documented CV simulation and fitting workflows are supported when the optional ElectroKitty dependency is installed.

Simulation uses ElectroKitty as an optional backend. Importing `ecat` and `ecat.simulation` works without ElectroKitty installed, but `simulate_cv()` and fitting calls that need a backend raise a friendly install message:

```bash
python -m pip install "ecat[simulation]"
```

Simulation option dictionaries are discoverable through:

```python
e.describe_options("simulation.cv_data")
e.describe_options("simulation.simulate_cv")
e.describe_options("simulation.fit_cv")
```

The bare names `cv_data`, `simulate_cv`, and `fit_cv` are accepted as short
aliases for those simulation option tables.

`simulation.cv_data()` can subtract measured-current background before fitting
with `{"background correction": "start current"}` or
`{"background correction": "tangent", "tangent potential": ...}`. The
`"start current"` name matches the FOWA option and subtracts the first selected
current point after segment/window trimming and before stride. Tangent mode uses
the existing tangent controls that are meaningful in this path
(`tangent potential`, `tangent range`, and `percent threshold`) and records the
applied correction in the returned `SimulatedCVInput.metadata`.

Mechanism presets accepted by `e.simulation.compile_mechanism()` and simulation calls include `E`, `EE`/`E,E`, `EC`, `ECE`, `EC'`/`Ecat`, and `Square`. The square-scheme preset compiles to `E(1):a=b`, `C:a=c`, `E(1):c=d`, and `C:b=d`; use `Square*` for the surface-confined shorthand.

Custom mechanisms use eCAT mechanism strings, which are a compatible superset
of ElectroKitty mechanism syntax. Write one `E:` or `C:` step per line. eCAT
accepts either conventional positive-integer coefficients or repeated species:

```python
mechanism = "C:A+2B=C"      # conventional eCAT form
equivalent = "C:A+B+B=C"   # ElectroKitty-compatible repeated form
```

Both forms produce the same eCAT stoichiometry, reaction-key matching,
activity quotient, and conservation equations. eCAT preserves the entered
equation for `.show()` and reports, then privately compiles coefficients to
repeated terms before calling ElectroKitty. Backend concentration and diffusion
arrays follow the species order returned by the installed ElectroKitty parser.
A leading positive integer is therefore always a coefficient in eCAT; literal
species names beginning with digits are not supported.

CV simulations use `e.simulation.SimulatedCVInput` for potential/time programs and `e.simulation.SimulatedCV` for backend results. `incubation_time` and quiet time are stored as input metadata and are not inserted into stored `E`/`t` arrays. Incubation evolves bulk homogeneous chemical reactions before the electrochemical program; surface and mixed-phase steps remain backend-only. Quiet time is materialized later as a backend hold at the starting potential. `SimulatedCVInput.plot()` plots the input waveform, using time vs potential by default; pass `{"plot quiet time": True}` to draw the metadata quiet hold at negative time, or pass `{"x axis": "potential", "y axis": "current"}` for fit-ready inputs with measured current. Simulated CVs expose the normal CV data-access surface (`x()`, `y()`, `xy()`, `analysis_segment_data()`) and can be overlaid with `e.multiplot`. `SimulatedCV.data` contains simulated current only; measured-vs-simulated overlays belong to `SimulationFitResult.plot()` and `SimulationGroupFitResult.plot()` as fit diagnostics. To return changed copies without modifying the original objects:

```python
program_fast = program.with_scan_rate(0.5)
aged_program = program.with_incubation_time(30.0)
result_fast = result.with_scan_rate(0.5)
faster_rxn = result.with_param("reactions.0.kf", 10.0)
more_substrate = result.with_params({"concentrations": {"bulk": {"Substrate": 2800}}})
```

`SimulatedCV.with_incubation_time(...)`, `with_params(...)`, `with_param(path, value)`, `with_input(...)`, and `with_mechanism(...)` all rerun the simulation and return a new `SimulatedCV`; the original object is not modified. Dict parameter updates are deep-merged, while lists and scalar values replace the existing value. Reruns use `SimulatedCV.input_params`, the normalized entered parameters, rather than reusing equilibrated or incubated backend concentrations.

Simulation objects use `.show()` for notebook-friendly setup and result display:

```python
program.show()
program.plot()
result.show({"print setup": True, "print params": True})
result.show({"print setup": False, "print states": True})
result.show({"print setup": False, "print checks": True})
fit_result.show({"print stats": True, "print corrections": True, "print params": True})
```

`print setup` is the canonical setup-display option. `SimulatedCVInput.show()` and `SimulatedCV.show()` display setup by default; pass `{"print setup": False}` to suppress setup or `{"print setup": "raw"}` for a raw debug dump.

Simulation fitting functions display the fitting setup, a live progress bar, and final fitting parameters by default. Pass `{"print setup": False}`, `{"progress": False}`, or `{"print params": False}` to suppress those pieces. The detailed `Fitting Progression:` audit table remains opt-in with `{"print progress": True}`.

For `fit_cvs()`, shared parameters are printed under `Fitting Params:` and dataset-specific paths from `per_cv` are printed under `Per-CV Fitting Params:`. The group setup table includes source and mapped concentration summaries when available; repeated scan rates are suppressed there so concentration-series fits emphasize the varying compound/concentration. Concentrations inferred from source CV metadata or `options["concentration mapping"]` appear in that per-CV table and in `result.best_params_by_cv`. In fitting parameter tables, fixed rows leave `Final Value` blank because the fixed initial value is the operative value; fitted rows show the optimizer result. Group-fit residual corrections are per-CV and print under `Group Fitting Corrections:`.

Parameter checks are available without a separate validation function. Use `simulate_cv(..., options={"check params": True})` to print checks during simulation, or `result.show({"print checks": True})` to inspect an existing simulated CV. Checks are diagnostic tables for likely interpretation issues such as missing diffusion for mobile bulk species, diffusion entries with no matching bulk concentration, parameter fallbacks, and preset mechanism species order.

Use `{"print states": True}` with `simulate_cv()` or `SimulatedCV.show()` to
display one table containing entered, equilibrated, and post-incubation amounts
for species whose concentration changed.

`fit_cv()` accepts a real eCAT `cv`, a fit-ready `SimulatedCVInput` with measured current, or a `SimulatedCV`. `fit_cvs()` fits one shared mechanism across multiple CV datasets and accepts a mixed list of those same input forms. The existing `fit` spec still controls which parameters are fixed or varied; `per_cv` only marks which parameter paths are dataset-specific:

```python
group_fit = e.simulation.fit_cvs(
    [cv_25, cv_50, cv_100],
    "E",
    params,
    fit={"vary": ["E0_0", "cell.Cdl"], "bounds": "auto"},
    per_cv=["cell.Cdl"],
    options={"cv data": {"stride": 1}, "concentration mapping": {"PhOH": "Substrate"}},
)
```

Simulation parameters use `concentrations` and `diffusion` for species setup:

```python
params = {
    "concentrations": {
        "bulk": {"a": 1.0, "b": 0.0},
        "surface": {"cat*": 1e-9},
    },
    "diffusion": {"a": 1e-9, "b": 1e-9},
}
```

Simulation parameter values use SI-derived units unless a documented string
alias or unit-bearing string is supplied:

| Path or quantity | eCAT public unit | Notes |
| --- | --- | --- |
| `SimulatedCVInput.E`, potential limits, `kinetics.*.E0` | `V` | Potentials are stored in volts. |
| `SimulatedCVInput.t`, `incubation_time`, quiet time | `s` | Incubation is chemical-only; quiet time is a backend hold at the starting potential. |
| simulated or measured current | `A` | Plotting may scale display units, but stored currents are amps. |
| `scan_rate` | `V/s` | Programmatic and CV-derived inputs use volts per second. |
| `cell.T` | `K` | `"auto"` pulls source CV metadata when available. |
| `cell.Ru` | `Ω` | Uncompensated resistance. |
| `cell.Cdl` | `F` | Total double-layer capacitance. `cv_data(..., {"estimate Cdl": "auto"})` estimates total farads from current separation. |
| `cell.A` | `m²` | Electrode area. Backend adapters that need areal capacitance compute `cell.Cdl / cell.A` internally. |
| `concentrations.bulk.*` | `mol/m³` | Use `1000 mol/m³` for `1 M`; entered values are pre-equilibration amounts. |
| `concentrations.surface.*` | `mol/m²` | Surface amounts are coverages. |
| `diffusion.*` | `m²/s` | Applies to mobile bulk species. |
| `kinetics.*.k0` | `m/s` | String presets such as `"fast"`, `"quasi"`, and `"slow"` resolve to SI values. |
| `kinetics.*.alpha` | dimensionless | Charge-transfer coefficient. |
| `spatial.dx_fraction` | dimensionless | Spatial mesh fraction. |
| `spatial.nx` | count | Spatial mesh size setting. |
| `spatial.viscosity` | `m²/s` | Kinematic viscosity; solvent aliases resolve to approximate room-temperature values. |
| `spatial.rotation` | `Hz` | Rotating-disk frequency when used. |
| `activity.standard_concentration` | `mol/m³` | Defaults to `1000 mol/m³`, i.e. `1 M`. |
| `activity.standard_coverage` | `mol/m²` | Surface-activity standard; defaults to `1 mol/m²`. |
| `activity.gamma.*` | dimensionless | Activity coefficients; omitted values behave as `1`. |
| `reactions.*.K` | dimensionless by default | Dimensionless `K` is activity-based. Unit-bearing strings such as `"2 M^-1"` or `"1e-3 m^3/mol"` are converted using the activity standard concentration and gammas. |
| `reactions.*.k`, `kf`, `kb` | reaction-order dependent | First-order rates are `s⁻¹`; second-order rates are `m³ mol⁻¹ s⁻¹`; third-order rates are `m⁶ mol⁻² s⁻¹`, using the concentration units above. |
| `reactions.*.k_exchange`, `koff` | `s⁻¹` | Exchange or reverse rate scales used with `K` before compiling backend `kf`/`kb`. |

`kinetics` and `reactions` are user-facing mechanism parameter sections. They
may be lists, integer-keyed dictionaries, or dictionaries keyed by mechanism
reaction strings. `kinetics` entries describe electrochemical `E:` steps with
`E0`, `k0`, and `alpha`. `reactions` entries describe chemical `C:` steps and
may use irreversible `k`, reversible `kf`/`kb`, or equilibrium-derived
`K` plus `k_exchange` or `koff`. A string `K` may include
units, such as `"2 M^-1"` or `"1e-3 m^3/mol"`. eCAT compiles these physical
inputs into private backend-ready rates under `params["_compiled"]`. Fitting
can still vary physical paths such as `reactions.0.K` and
`reactions.0.k_exchange`; for each trial simulation, eCAT converts the current
physical values into the `kf`/`kb` rates required by the backend.
Reaction-string dictionary keys may use either coefficient or repeated-species
notation; for example, a `"A+A=B"` key matches a `C:2A=B` mechanism step.
For a reaction with total reactant order `n_r`, product order `n_p`, and the
phase-appropriate activity standard `X°`, eCAT defines the pool-free
standard-state exchange scale as:

```text
k_exchange = kf * X°**(n_r - 1) + kb * X°**(n_p - 1)
```

`X°` is `activity.standard_concentration` for bulk reactions and
`activity.standard_coverage` for surface reactions. For first-order equilibria
this reduces to `kf + kb`; for bulk `A + B = C` it is `kf C° + kb`. The
activity-derived `kf/kb` ratio and this equation uniquely
determine the backend rates. Those intrinsic rates do not change across a
concentration series, while the actual mass-action rate still changes with the
current concentrations. `reference_concentration` or `reference_coverage` may
replace the corresponding phase standard for this rate-scale definition; the
reference must be positive and finite.

Every species in an explicit-`K` reaction must have an entered concentration;
use zero for an initially absent species. By default eCAT solves the complete
pre-equilibrium from the reaction stoichiometric matrix, dimensionless activity
quotients, and conservation laws inferred from that matrix. No pool declaration
is required. A reversible reaction with `equilibrate=False` is excluded from
the algebraic starting-state solve but still participates in finite incubation
and backend dynamics. Consistent dependent equilibrium cycles are accepted;
inconsistent cycles raise a residual-based error.

The preparation order is:

1. Normalize entered concentrations and physical reaction parameters.
2. Solve explicit-`K` reactions unless `equilibrate=False`.
3. Integrate bulk homogeneous chemical reactions for `incubation_time` when it is greater than zero. Surface and mixed-phase steps remain backend-only.
4. Apply quiet time as an ElectroKitty hold at the starting potential.
5. Run the CV waveform.

`concentrations.pools`, top-level `pools`, and top-level `equilibria` are not
supported. Put `K` directly in the matching `reactions` entry. Pre-equilibrium
currently requires every participating reaction to remain within one phase;
mixed bulk/surface equilibria raise an error because eCAT has no volume-to-area
capacity convention for a shared conservation equation.

eCAT reports numerical reaction rank, inferred conservation rank, dependent
reaction count, and equilibrium residuals. It does not claim elemental or
charge balance from species labels because names such as `Substrate` or
coordination-complex abbreviations are not reliable molecular formulas.

Activity coefficients are `activity["gamma"]` values, default to `1`, and are
displayed inside the species table only when at least one gamma differs from
`1`:

```python
params["concentrations"] = {"bulk": {"A": 10.0, "B": 0.0}}
params["reactions"] = {"A=B": {"K": 4.0, "k_exchange": 10.0}}
params["activity"] = {"gamma": {"bulk": {"A": 0.9}}}
```

For cell constants, use an explicit mapping when teaching or auditing values,
or use `"cell": "auto"` once the workflow is established. The shorthand expands
to `{"Cdl": "auto"}` and then fills `T`, `Ru`, and `A` from the source CV when
available, falling back to `298.15 K`, `0 Ω`, and `1e-5 m²`. `Cdl` requires
measured current or a prior `cv_data(..., {"estimate Cdl": "auto"})` estimate.
In eCAT's public params, `cell.Cdl` is total double-layer capacitance in `F`.
The ElectroKitty adapter converts that total value to the backend's
area-normalized capacitance internally using `cell.Cdl / cell.A`; users should
not divide by area themselves or pass `F/m²` as `cell.Cdl`.

As input sugar, `species` may be supplied as a mapping of species names to `type`/`C`/`D` fields; eCAT normalizes it immediately into `concentrations` and `diffusion`, and prepared simulation results do not retain `species`. Use `{"print params": "compact"}` for compact species and mechanism parameter tables; fitting comparison tables remain path-rich.

## Internal Helpers

The curated top-level API intentionally omits imported dependencies, plotting internals, parser helpers, and legacy low-level animation helpers.

Examples of categories that are not part of the beta public API:

- `np`, `pd`, `plt`, `mpl`
- legacy animation helpers
- parser internals
- plotting internals
- compatibility wrappers

Prefer the names listed in this reference for new notebooks.
