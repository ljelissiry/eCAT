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
| `get_CVs(options=None)` | Load CV-like text files from a folder using flexible text-table detection. | `describe_options("get_data")` |
| `get_data_from_excel(file_path, options=None)` | Create eCAT objects from an eCAT Excel workbook, or fall back to curated Excel CV header parsing when no `manifest` sheet is present. | `describe_options("get_data")` |

Folder loaders return an empty list (`[]`) when no supported files are found or no files can be converted, so notebook loops and filters can safely consume the result without a separate `None` check.

`get_data()` and `echem.from_file()` now support a `custom parser` hook for filename-derived metadata and a `parser settings` dictionary for parser behavior. Use `custom parser mode="merge"` to fill only missing filename metadata, or `custom parser mode="override"` to replace the built-in filename parser result. File-derived metadata still wins by default; set `parser settings={"prefer file metadata": False}` only when you explicitly want the custom parser to replace file metadata such as scan rate. Parser settings also accept canonical `gases` and `solvents` lists plus `compound stopwords`.

Every loaded object exposes `obj.parse_result`, a `ParseResult` with a consistent parser contract: `.data`, `.units`, `.technique`, `.software`, `.metadata`, `.raw_metadata`, `.warnings`, `.source`, and `.parser`. Use `parse_file(...)` when you want that contract directly for parser debugging or importer tests. Normal analysis workflows should still use `echem.from_file(...)`, `get_data(...)`, or `get_CVs(...)`.

## eCAT App

| Name | Purpose |
|---|---|
| `open_app(host="127.0.0.1", port=0, browser=False, inline=False)` | Start the local eCAT app from Python or a notebook and return the local URL. |

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

Display helpers return `None` by default to avoid duplicate notebook output. `show(group)` delegates to `show_objects(group)` for one group, while `show(groups)` delegates to `show_groups(groups)` for nested grouped lists. Pass `{"return": True}` to return the displayed table when you need it for further work. Numeric object-display values honor `sig figs`; recognized unit suffixes such as `(A)`, `(V)`, and `(s)` are displayed in the Value column. Use `show_objects(objects, {"columns": "available"})` to return the available column keys, `{"columns": "all"}` to show every non-internal column, or pass labels such as `"Scan Rate"` and aliases such as `"scan_rate"` in the column list.

## Plotting

| Name | Purpose | Options |
|---|---|---|
| `obj.plot(options=None)` | Plot one eCAT object. | `describe_options("plot")` |
| `multiplot(objects, options=None)` | Overlay multiple objects on one axes. | `describe_options("multiplot")` |
| `multimultiplot(groups, options=None)` | Plot multiple groups as separate multiplots. | `describe_options("multimultiplot")` |
| `multi_scatterplot(data, options=None)` | Plot one or more result-table metrics. | `describe_options("multi_scatterplot")` |
| `plotting_style(style=True)` | Apply notebook, publication, Savéant-inspired, or Matplotlib-default plotting styles. | `describe_options("plot")` |

Available style values are `"notebook"`, `"publication"`, `"saveant"` / `"savéant"`, and `False` / `"matplotlib"` / `"default"` to restore Matplotlib defaults.

Axis labels can use descriptive text or electrochemistry symbols through the plot option `symbol labels`. Use `{"symbol labels": True}` for compact labels such as `E`, `i`, `j`, `t`, and `Q`; use `False` for descriptive labels; use `"auto"` to follow the active plotting style. The Savéant style enables symbol labels by default. Normalized axes such as `i/ip0` and dimensionless normalization labels remain symbolic.

The package default plot convention is `IUPAC`. For CVs, pass `{"plot convention": "US"}` for the left-to-right axis orientation instead.

`multiplot()` auto-generates labels from differing object metadata for CV, DPV, CA, and CP overlays. Pass `{"labels": [...]}` when you want explicit trace labels instead. The legacy alias `plot labels` is accepted, but do not supply both spellings.

Scale bars are available through plot options rather than as a standalone top-level helper:

```python
cv_obj.plot({"scale bar": {"loc": "lower right", "length": 1e-6}})
e.multiplot(cvs, {"scale bar": {"loc": (-0.5, 0.0), "length": 5e-6}})
```

The scale-bar `length` is in the displayed y-axis unit. A tuple `loc=(x, y)` uses displayed data coordinates after unit conversion and scaling. Scale bars remove y ticks by default.

For gradient-colored multiplots, `gradient ...` options control how trace metadata maps to colors, while `colorbar ...` options control the displayed colorbar legend. Use `colorbar tick labels` (`endpoints`, `all`, or `none`) and `colorbar trace ticks` to control tick text and per-trace tick marks. `gradient reverse` reverses trace color assignment; `colorbar reverse` flips only the displayed colorbar direction.

`multi_scatterplot()` works on analysis result tables and supports direct column control:

- `x column`: explicit x-axis column name or `"auto"` (default).
  - `auto` prefers `"x transformed"` then raw `"x"` columns before falling back to sensible x metadata.
- `y column`: explicit y-axis column name or `"auto"` (default).
  - `auto` prefers transformed and metric columns (`"kobs"`, `"TOFmax"`, `"ip"`, `"Ep"`).
- `y columns`: optional list of y columns to plot at once.
- `metric`: preferred metric name used by auto-resolution when `y column = "auto"`.
- `labels`: optional explicit trace labels (useful for concentration or scan-rate series; `plot labels` is accepted as a legacy alias).

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

## CV Analysis

| Name | Purpose | Options |
|---|---|---|
| `cv_obj.peak_potential(options=None)` | Locate a selected peak potential. | `describe_options("cv.peak_potential")` |
| `cv_obj.peak_current(options=None)` | Measure peak current with baseline/tangent handling. | `describe_options("cv.peak_current")` |
| `cv_obj.half_peak_potential(options=None)` | Estimate half-peak potential. | `describe_options("cv.half_peak_potential")` |
| `cv_obj.half_wave_potential(options=None)` | Estimate half-wave potential from paired peaks. | `describe_options("cv.half_wave_potential")` |
| `cv_obj.current_at_potential(potential, options=None)` | Extract current at a requested potential. | `describe_options("cv.current_at_potential")` |
| `cv_obj.plateau_current(options=None)` | Analyze plateau current for one CV. | `describe_options("cv.plateau_current")` |
| `normalize(cvs, options=None)` | Return normalized CV copy/copies. CV-only. | `describe_options("normalize")` |
| `cv_obj.normalize(options=None)` | Normalize one CV in place and return itself. | `describe_options("cv.normalize")` |
| `normalize_current(cvs, options=None)` | Return CV copy/copies with `i/ip0` current normalization. | `describe_options("normalize_current")` |
| `scale_current(cvs, options=None)` | Scale currents against reference currents or manual values. | `describe_options("scale_current")` |

Single-CV analysis helpers such as `peak_potential`, `peak_current`, `half_peak_potential`, `half_wave_potential`, `wave_info`, and `current_at_potential` return `CVAnalysisResult`, an `AnalysisResult` child that remains dictionary-compatible. Existing notebook access such as `result["ip"]` or `result["Ep"]` is preserved. New code can also use `result.primary` for the main base-unit scalar, `result.table` for a tidy display table with columns such as `Metric` and `Value`, and `result.show()` for pretty or plain printing. The table scales values to readable display units, while `result.primary` remains in base units.

Peak and wave defaults are intended for exploratory use. Single-feature helpers such as `peak_potential()`, `peak_current()`, `half_peak_potential()`, and `peak_info()` choose the largest detected feature when `segment` and `guess potential` are omitted. Paired-wave helpers such as `half_wave_potential()` and `wave_info()` default to segment `1` paired with segment `2` when `segments` is omitted. For final analysis, specify `segment`, `segments`, `guess potential`, or `exact potential` so the notebook records the intended feature.

Complex CV analyses accept per-CV potential lists through plural aliases. `describe_options` shows these as one row with `(s)`, such as `guess potential(s)`, `tangent potential(s)`, `non-catalytic cv(s)`, and `scan rate(s)`. Use `guess potentials`, `exact potentials`, `tangent potentials`, `peak potentials`, `non-catalytic guess potentials`, or FOWA `redox potentials` when each CV needs its own value. Scalar `guess potential` keeps the existing running-guess behavior within multi-segment analyses. For paired-peak analyses such as `trumpet_analysis` and `nicholson_analysis`, a flat two-value `guess potential` is treated as a shared paired guess when there are more than two CVs; with exactly two CVs, use nested pairs like `[[g1a, g1b], [g2a, g2b]]` when each CV needs a paired guess. If both singular and plural spellings are supplied for the same option, eCAT raises an option error.

When overlaying analysis diagnostics on an existing CV plot or `multiplot`, use `{"plot": True, "plot CV": False, "new plot": False}`. This adds markers, tangent lines, and current-distance diagnostics to the active axes without redrawing the underlying CV trace.

`normalize(...)` is intentionally CV-only. It raises a clear error for CA, CP, DPV, or generic objects because physical dimensionless normalization is not the same workflow for those techniques.

For dimensionless CV normalization, `normalize(...)` automatically uses stored CV metadata when explicit options are omitted. Temperature defaults to `cv.temperature` for either normalized axis. For current-axis normalization, electrode area defaults to `cv.electrode_area` and scan rate defaults to `cv.scan_rate`. Concentration is explicit by default. If `C`/`C unit` are omitted, pass `species` to resolve the concentration from the CV's exact `compounds` / `concentrations` metadata pair:

```python
normalized = e.normalize(cv, {"E0": -1.0, "D": 1e-5, "species": "[Co]"})
```

Species matching is exact; use the same spelling, case, and brackets shown in `cv.compounds`. Explicit `C` and `C unit` override `species`.

Normalized CVs store plain data columns named `Dimensionless Potential` and `Dimensionless Current`. Default plot labels use compact symbols: `θ` for dimensionless potential, `Φ` for homogeneous dimensionless current, and `χ` for heterogeneous dimensionless current. The full equations are shown by `normalize(..., {"print": True})`.

`normalize_current(...)` is separate from physical dimensionless normalization. It creates `i/ip0` from the raw `Current` column; it does not compute `Φ/Φp0`.

## Other Technique Methods

| Name | Purpose | Options |
|---|---|---|
| `dpv_obj.peak_potential(options=None)` | Locate a DPV peak potential. | `describe_options("dpv.peak_potential")` |
| `ca_obj.charge(options=None)` | Integrate CA current to cumulative charge and optionally plot target-charge diagnostics. | `describe_options("ca.charge")` |
| `ca_obj.time_at_charge(charge=None, options=None)` | Find when a CA trace reaches a target charge. | `describe_options("ca.time_at_charge")` |
| `cp_obj.get_cycles(options=None)` | Split CP data into charge/discharge cycles. | `describe_options("cp.get_cycles")` |
| `cp_obj.cycle_info(options=None)` | Summarize CP cycle capacity, efficiency, and potential metrics. | `describe_options("cp.cycle_info")` |
| `cp_obj.plot_cycles(options=None)` | Plot selected CP cycles. | `describe_options("cp.plot_cycles")` |
| `cp_obj.cycling_plot(options=None)` | Plot CP capacity and efficiency versus cycle number; accepts the same `cycles` selection forms as `plot_cycles()`. | `describe_options("cp.cycling_plot")` |

DPV pulse metadata follows the same unit convention as CA/CP metadata: object
attributes and `stats()` keys are unitless SI values, while printed tables and
auto plot subtitles put autoscaled units in the values, such as `10 mV` or
`50 ms`.

## Advanced Analysis

| Name | Purpose | Options |
|---|---|---|
| `fowa(cvs, options=None)` | Foot-of-the-wave analysis for catalytic CVs. | `describe_options("fowa")` |
| `sevcik_analysis(cvs, options=None)` | Sevcik-style peak-current trend analysis. | `describe_options("sevcik_analysis")` |
| `trumpet_analysis(cvs, options=None)` | Trumpet analysis from paired peak potentials. | `describe_options("trumpet_analysis")` |
| `nicholson_analysis(cvs, options=None)` | Nicholson-style heterogeneous electron-transfer analysis. | `describe_options("nicholson_analysis")` |
| `tafel_analysis(cv_or_list, TOF_max, thermodynamic_potential, redox_potential, options=None)` | Tafel-style turnover-frequency curves for one CV or a CV series; multi-CV calls reuse the shared `multiplot()` label, color, gradient, legend, and colorbar options. | `describe_options("tafel_analysis")`; also accepts common `multiplot` styling options |
| `fit_model(x_or_result, y=None, model=None, options=None)` | Fit direct scatter models such as linear, power, power offset, exponential, Michaelis-Menten, and logistic. | standalone options |
| `fit_rate(df_or_result, options=None)` | Fit rate or transformed result tables; accepts a DataFrame or an `AnalysisResult` and uses `.table` by default. | `describe_options("fit_rate")` |
| `plateau_current(cvs, options=None)` | Batch plateau-current workflow. | `describe_options("plateau_current")` |
| `fit_peak_potential(cvs, options=None)` | Fit peak-potential trends. | `describe_options("fit_peak_potential")` |
| `fit_peak_current(cvs, options=None)` | Fit peak-current trends. | `describe_options("fit_peak_current")` |

`fowa(..., {"print": True})` prints a vertical `Field`/`Value` summary table for shared settings and reference information, the symbolic FOWA `kobs` equation with definitions, and the FOWA result table. It does not print a second equation with the `n` values substituted. `{"pretty print": False}` uses the same vertical summary shape as plain text.

Advanced analysis helpers return `AnalysisResult`-style objects. Use `.table` for the primary table, `.summary` for workflow metadata, `.fits` / `.fit_table` where fitting applies, `.axes` for plots, `.units` for result units, and `.warnings` / `.diagnostics` for extra detail. FOWA and plateau-current results no longer pretend to be DataFrames; use `result.table.columns`, `result.table.loc[...]`, `result.table.attrs`, or `result.table["kobs"]`. For export, `result.to_csv(...)` writes the primary table using pandas CSV semantics, while `result.to_excel(...)` writes a workbook with the primary `table` sheet plus available metadata sheets such as `summary`, `fit_table`, `fits`, `warnings`, `units`, and `diagnostics`.

`nicholson_analysis()` returns an `AnalysisResult` with `result["data"]` and `result.table` for the Nicholson point table and `result["summary"]` / `result.summary` for equation metadata, fit statistics, and kinetic values. With `{"print": True}`, it prints the Nicholson equation, a summary table, and the input/result table. With `{"plot all": True}`, it produces one CV diagnostic plot and one Nicholson fit plot.

`fit_rate`, `fit_peak_current`, and `fit_peak_potential` use the same direct model fitter as `fit_model`. Use `fit model` to choose the model and `fit init`, `fit bounds`, `fit residual`, `fit max evals`, `fit range`, `fit ranges`, and `fit indices` to control the fit. Standalone `fit_model` also accepts bare aliases such as `model`, `init`, `bounds`, `residual`, `range`, `ranges`, and `indices` because the function context is already fitting.

Scatter-fit helpers return `ScatterFitResult`, an `AnalysisResult` child, rather than tuples. Use `.table` for plotted/result data, `.fits` for fit parameters, `.fit_table` for the human fit-statistics table, and `.summary` for workflow metadata. Tuple unpacking such as `data, fits = e.fit_rate(...)` is not part of the beta API.

Scatter-fit plots use matching point and fit-line colors by default. This applies to `fit_model`, `fit_rate`, `fit_peak_current`, `fit_peak_potential`, `sevcik_analysis`, `trumpet_analysis`, and `multi_scatterplot`. Pass `fit color` to override the fit line explicitly. For functions that draw multiple fits, `fit color` may be a list; `fit colors` is accepted as the plural alias, and `describe_options` displays this as `fit color(s)`. Colors are consumed in plotted fit order, and the last color repeats if the list is shorter than the number of fits. If both spellings are supplied, eCAT raises an option error.

When `fit_model(..., {"print": True})` is used, the default pretty output is adaptive. Simple fits print a concise `Field`/`Value` table with model settings, fit statistics, and fitted parameter values with standard errors inline when available. More constrained or complex fits print separate `Fit Model Details` and `Fit Model Parameters` tables, including initial values, bounds, final values, and standard errors. Auto switches to details for explicit `fit init`, explicit `fit bounds`, custom formula/callable models, constrained models such as power offset/logistic/Michaelis-Menten, three or more parameters, or parameters at a bound. Use `{"print fit": "summary"}` or `{"print fit": "details"}` to force the style. Printed equations are the general model equations so parameter meanings remain visible; plot labels use the fitted equation with numeric values, formatted by `sig figs`. `fit_rate(..., {"print": True})` uses the same human table. Transforms such as `{"transform mode": "log-log"}` only change the coordinates being fit; they do not change or relabel the selected fit model. Use `{"fit model": "power"}` when you want a direct power-law model. Use `{"pretty print": False}` for a compact dictionary-style summary.

Fitting helpers keep row/index selection and x-value-window selection separate. Use `fit indices` for row/position-based selection; index windows use Python-style stops, so `[start, stop]` includes `start` and excludes `stop`, and `None` leaves either side open-ended. Use `fit range` for one x-value window on the resolved/transformed x axis. Use `fit ranges` when you want multiple named or generated x-value-window fits; nested windows let one fit use disconnected x regions. Dictionaries create named fits, for example `{"early": [0, 3], "tail": [4, None]}`. The selected points determine the fitted parameters, while result tables still report predictions and residuals for the original input rows where applicable.

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

These helpers are intentionally not exported as top-level names during the beta pass. Treat simulation workflows as preview or experimental unless a notebook or guide section says otherwise.

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

Mechanism presets accepted by `e.simulation.compile_mechanism()` and simulation calls include `E`, `EE`/`E,E`, `EC`, `ECE`, `EC'`/`Ecat`, and `Square`. The square-scheme preset compiles to `E(1):a=b`, `C:a=c`, `E(1):c=d`, and `C:b=d`; use `Square*` for the surface-confined shorthand.

CV simulations use `e.simulation.SimulatedCVInput` for potential/time programs and `e.simulation.SimulatedCV` for backend results. Quiet time is stored as input metadata and is materialized only when preparing backend simulation input; it is not inserted into stored `E`/`t` arrays. `SimulatedCVInput.plot()` plots the input waveform, using time vs potential by default; pass `{"plot quiet time": True}` to draw the metadata quiet hold at negative time, or pass `{"x axis": "potential", "y axis": "current"}` for fit-ready inputs with measured current. Simulated CVs expose the normal CV data-access surface (`x()`, `y()`, `xy()`, `analysis_segment_data()`) and can be overlaid with `e.multiplot`. `SimulatedCV.data` contains simulated current only; measured-vs-simulated overlays belong to `SimulationFitResult.plot()` and `SimulationGroupFitResult.plot()` as fit diagnostics. To rerun the same simulated waveform at a new scan rate without modifying the original object, use `with_scan_rate()`:

```python
program_fast = program.with_scan_rate(0.5)
result_fast = result.with_scan_rate(0.5)
faster_rxn = result.with_param("reactions.0.kf", 10.0)
more_substrate = result.with_params({"concentrations": {"bulk": {"Substrate": 2800}}})
```

`SimulatedCV.with_params(...)`, `with_param(path, value)`, `with_input(...)`, and `with_mechanism(...)` all rerun the simulation and return a new `SimulatedCV`; the original object is not modified. Dict parameter updates are deep-merged, while lists and scalar values replace the existing value.

Simulation objects use `.show()` for notebook-friendly setup and result display:

```python
program.show()
program.plot()
result.show({"print setup": True, "print params": True})
result.show({"print setup": False, "print checks": True})
fit_result.show({"print stats": True, "print corrections": True, "print params": True})
```

`print setup` is the canonical setup-display option. `SimulatedCVInput.show()` and `SimulatedCV.show()` display setup by default; pass `{"print setup": False}` to suppress setup or `{"print setup": "raw"}` for a raw debug dump.

Simulation fitting functions display the fitting setup, a live progress bar, and final fitting parameters by default. Pass `{"print setup": False}`, `{"progress": False}`, or `{"print params": False}` to suppress those pieces. The detailed `Fitting Progression:` audit table remains opt-in with `{"print progress": True}`.

For `fit_cvs()`, shared parameters are printed under `Fitting Params:` and dataset-specific paths from `per_cv` are printed under `Per-CV Fitting Params:`. The group setup table includes source and mapped concentration summaries when available; repeated scan rates are suppressed there so concentration-series fits emphasize the varying compound/concentration. Concentrations inferred from source CV metadata or `options["concentration mapping"]` appear in that per-CV table and in `result.best_params_by_cv`. In fitting parameter tables, fixed rows leave `Final Value` blank because the fixed initial value is the operative value; fitted rows show the optimizer result. Group-fit residual corrections are per-CV and print under `Group Fitting Corrections:`.

Parameter checks are available without a separate validation function. Use `simulate_cv(..., options={"check params": True})` to print checks during simulation, or `result.show({"print checks": True})` to inspect an existing simulated CV. Checks are diagnostic tables for likely interpretation issues such as missing diffusion for mobile bulk species, diffusion entries with no matching bulk concentration, parameter fallbacks, and preset mechanism species order.

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

For cell constants, use an explicit mapping when teaching or auditing values,
or use `"cell": "auto"` once the workflow is established. The shorthand expands
to `{"Cdl": "auto"}` and then fills `T`, `Ru`, and `A` from the source CV when
available, falling back to `298.15 K`, `0 Ω`, and `1e-5 m²`. `Cdl` requires
measured current or a prior `cv_data(..., {"estimate Cdl": "auto"})` estimate.

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
