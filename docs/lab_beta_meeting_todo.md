# Lab Beta Meeting Todo

This is the working checklist for getting eCAT ready to introduce to lab users. The goal is a clear, useful beta meeting: users should understand what the package can do, what is still limited, how to try it on their own data, and how to report useful feedback.

## Highest Priority Before The Meeting

1. Prepare a meeting-ready tutorial. **Status: pending**
   - Build one short live-demo path: install/import, load data, plot, filter or group, run one analysis, export a figure or table.
   - Use a tiny example dataset that is known to work.
   - Keep one backup dataset ready in case uploaded files expose parser issues during the meeting.
   - Add a short `demo_script.md` or meeting notes file with exact cells/files/options to use live.

2. Create quickstart notebooks. **Status: drafted; execution check still needed**
   - Make notebooks that users can run top-to-bottom without private local paths.
   - Keep each notebook focused on one workflow.
   - Label limited or experimental workflows clearly.
   - Run every quickstart notebook from top to bottom before the meeting.

3. Create a default template notebook. **Status: drafted; execution check still needed**
   - This should be the notebook beta users copy first for their own data.
   - It should use obvious placeholders for data folders, file patterns, reference files, and export folders.
   - It should include common cells for loading, inspecting, plotting, filtering, analysis, and saving outputs.
   - It should include short notes where users are expected to edit values.

4. Triage uploaded data. **Status: pending**
   - Go through representative files from lab users.
   - Record what loads, plots, analyzes, exports, or fails.
   - Capture parser failures and surprising metadata behavior as known limitations or post-beta bugs.
   - Identify one known-good demo dataset and one backup dataset from real or example data.

5. Review and update documentation. **Status: mostly complete; final consistency pass needed**
   - Make sure README, quickstart, beta scope, troubleshooting, and beta tester guide agree with observed behavior.
   - Make sure terminology is consistent: CV, CA, CP, DPV, EIS, FOWA, EC-Lab, CH Instruments, BASI, reference shifting, current density, normalization.
   - Document `save_data` as the beta-supported top-level export helper.
   - Document `plotting_style(...)` as the plotting-style control function.
   - Document scale bars through `plot` and `multiplot` options, not as a standalone helper.
   - Document `get_CVs_from_excel(...)` for spreadsheet-assembled CV datasets.
   - Document that top-level `normalize(...)` is CV-only and returns copies.
   - Keep animation helpers out of the curated top-level beta API for now.
   - Remove deprecated aliases from `docs/api_reference.md`; only document current public names.

6. Verify release basics. **Status: mostly complete; final clean install still needed**
   - Confirm the beta version, preferably `0.1.0b2`.
   - Run a clean install check.
   - Run `pytest -q`.
   - Make sure example notebooks execute from top to bottom.
   - Confirm docs do not promise unsupported workflows.
   - Confirm beta users can install from the public GitHub link.
   - Confirm the default branch or release tag points at the beta-ready code.

7. Prepare meeting logistics. **Status: pending**
   - Pick the lab beta meeting time.
   - Make a short presentation or slide deck.
   - Include install instructions, supported-scope matrix, demo workflow, known limitations, and feedback form link.
   - Send beta users the GitHub link, notebook starting point, and what files to bring.

## Completed Since This Checklist Was Created

- Package version is set to `0.1.0b2`.
- The supported-scope matrix exists in `docs/beta_scope.md`.
- README, quickstart, troubleshooting notes, beta tester guide, and API reference exist.
- The main beta notebook set exists, including the default template notebook.
- A tracked Fe/PhOH example dataset exists under `examples/data/fe_phoh_cv`.
- The package has been split into focused modules while retaining the notebook-facing API.
- Top-level API cleanup has been mostly implemented; simulation remains namespaced under `e.simulation`.
- Current tests have passed locally, most recently `569 passed, 1 warning`.
- A local beta wheel exists at `dist/ecat-0.1.0b2-py3-none-any.whl`.

## Suggested Notebook Set

| Notebook | Purpose | Beta Status |
|---|---|---|
| `00_install_and_setup.ipynb` | Environment setup, imports, package/version check, test data load | Required |
| `01_default_template.ipynb` | Copy-first notebook for users analyzing their own data | Required |
| `02_basics_load_plot_filter.ipynb` | `from_file`, `get_data`, plotting, filtering, grouping, simple exports | Required |
| `03_cv_analysis.ipynb` | Reference shifting, peak potential/current, current-at-potential, half-peak, and paired-wave metrics | Required |
| `04_plotting_multiplot.ipynb` | Trim, polished `multiplot`, gradients/colorbars, scale bars, and `multimultiplot` grouped overlays | Required |
| `05_dpv_basics.ipynb` | DPV loading, pulse metadata, Ar/CO2 overlays, peak potential, and overlapping peak fits | Recommended |
| `06_ca_cp_basics.ipynb` | CA/CP load, plot, and export workflows where currently supported | Recommended |
| `07_advanced_analysis.ipynb` | Normalization, standardized current, `i/ip0`, FOWA, Sevcik-style workflows, Tafel-style comparisons, summary tables | Recommended |
| `08_simulation_intro.ipynb` | Basic simulation, single-CV Ar/EEC' fits, and simulated scan-rate sweeps | Optional/advanced |
| `09_group_fitting.ipynb` | Group fitting across multiple CVs with shared and per-CV parameters | Optional/advanced |
| `99_troubleshooting.ipynb` | Parser failures, missing peaks, units, references, normalization, unsupported files | Recommended |

## Default Template Notebook Outline

The default template notebook should be practical rather than explanatory. It should help users replace a few paths and immediately analyze data.

Suggested sections:

1. Setup
   - Import eCAT and common scientific Python packages.
   - Print package version.
   - Set plotting backend/style if needed.

2. User inputs
   - `DATA_DIR`
   - `FILE_PATTERN`
   - `REFERENCE_FILE` or `REFERENCE_MODE`
   - `EXPORT_DIR`
   - technique or expected file type, if useful

3. Load data
   - Load a single file.
   - Load a folder with `get_data()`.
   - Print a short summary of loaded objects.

4. Inspect data
   - Show `info()`, `stats()`, `x()`, `y()`, and `xy()` examples.
   - Display filenames and parsed metadata.

5. Plot
   - Plot one object.
   - Plot grouped or overlaid data.
   - Save one figure.

6. Filter and group
   - Filter by metadata or filename-derived fields.
   - Group by scan rate, concentration, gas, solvent, or other common metadata.

7. Analysis
   - CV: peak potential/current and optional current density or normalization.
   - CA/CP: load/plot/export only unless analysis is validated for beta.
   - Advanced cells should be clearly marked optional.

8. Export
   - Save processed data table.
   - Save summary table.
   - Save figures.

9. Notes for bug reports
   - Record package version.
   - Record filename and instrument.
   - Record expected behavior and actual behavior.
   - Include traceback or screenshot if available.

## Uploaded Data Triage Table

Use this table while reviewing lab-user files.

| File | Source/User | Instrument/Format | Technique | Loads | Plots | Analysis Works | Export Works | Notes | Fixture Candidate |
|---|---|---|---|---:|---:|---:|---:|---|---:|
|  |  |  |  |  |  |  |  |  |  |

Suggested categories for notes:

- Parser failure
- Metadata parsing issue
- Units unclear
- Reference shift unclear
- Normalization issue
- Missing peak
- CA/CP limited support
- Unsupported binary file
- Good beta example

## Test And Debug Before Meeting

These are specific beta-risk areas to test or debug before inviting broad lab use.

| Area | Why It Matters | Suggested Check | Status |
|---|---|---|---|
| Quickstart notebooks | Users will likely copy notebook workflows directly | Run `00` through `09` and `99` top-to-bottom in a clean kernel | Pending |
| Unit storage, `df` vs `.units` | Unit confusion can silently corrupt interpretation | Confirm loaded/exported objects keep unit metadata consistent with DataFrame columns and printed summaries | Pending |
| Plateau current | Advanced benchmarking workflow should not silently mislead users | Run existing tests, run one example workflow, document limitations | Pending |
| Animation | Previously de-emphasized; decide whether it works or stays non-public | Smoke-test existing animation helpers or explicitly document as out of beta scope | Pending |
| Deprecated aliases in API docs | Users should learn current names only | Remove old aliases from API reference and notebooks | Completed |
| FOWA tangent diagnostics | Tangent-baseline failures should be actionable | Confirm `describe_options("fowa")` shows tangent controls and errors report current settings | Completed |
| Reference shifting with printed summaries | This recently regressed after module split | Confirm `get_data(..., reference mode=..., print=True)` works | Completed |
| CV wave and half-wave plotting | Recent fix should stay covered | Keep regression tests passing | Completed |

## Simulation Feature Suggestions

Simulation should not block the beta meeting unless the documentation, tutorial, and uploaded-data triage are already ready. Treat simulation and group fitting as advanced workflows with clear setup, assumptions, and limitations.

### Good Pre-Beta Simulation Features

- Simulate simple CV-like traces with adjustable parameters.
- Sweep scan rate, concentration/current scale, formal potential, and noise.
- Overlay simulated and experimental traces.
- Plot residuals between simulation and experiment.
- Export simulated data in an eCAT-compatible table shape.
- Fit one experimental trace to a small parameter set.
- Fit grouped simulated/experimental CV sets with shared and per-trace parameters.
- Make simulated CV objects compatible with the same public analysis options as imported CVs wherever scientifically meaningful.
- Provide clear warnings about assumptions and non-uniqueness.

### Group Fit Features

Start with a small, transparent global fit rather than a broad simulation engine.

Possible fit design:

- Shared parameters:
  - formal potential or midpoint potential
  - kinetic/shape parameter
  - diffusion/current scale, if appropriate
- Per-trace parameters:
  - baseline offset
  - current scale or concentration factor
  - optional potential offset
- Outputs:
  - best-fit parameter table
  - confidence or fit-quality diagnostics, if available
  - overlay plot
  - residual plot
  - warning when the fit is underdetermined

### Simulation And Fitting Items To Add

- Grouped simulation fitting for related CV sets.
- Simulated CV objects should support the same analysis options as real CV objects where the object has equivalent data and metadata.
- Document which analysis options are meaningful for simulated CVs and which are ignored or invalid.
- Unify fitting and analysis outputs so result tables, fit summaries, diagnostics, plots, and export metadata follow consistent naming and structure across workflows.
- Add Tafel comparison workflows for comparing multiple catalysts, conditions, or derived quantities on one benchmarking figure/table, with assumptions and input sources clearly documented.

### Simulation Features To Defer

- Broad mechanism-aware simulation framework.
- Claims of mechanistic uniqueness.
- Complex ECEC/EECC fitting without validation.
- GUI-based fit window selection.
- Large-scale automated model selection.

## Meeting Agenda

Suggested 45-60 minute structure:

1. What eCAT is for: 5 minutes.
2. What is supported in this beta: 5 minutes.
3. Live demo with known-good data: 10-15 minutes.
4. Show the default template notebook: 10 minutes.
5. Explain how to try it on lab data: 5 minutes.
6. Explain limitations and bug reporting: 5 minutes.
7. Collect user questions and target workflows: remaining time.

## User Instructions To Prepare

Send beta users a short message before the meeting:

- Please install Python or bring questions about installation.
- Please bring one or two representative exported text files from your instrument.
- Prefer `.txt`, `.csv`, or other text exports when possible.
- Binary files may not be supported in the beta.
- If you use CV, CA, CP, DPV, EIS, or another technique, note which workflows you want to analyze.
- If you already use Jupyter notebooks, bring an example of how you currently analyze data.

## Do Not Let These Block The Meeting

- Full simulation framework.
- Polished global fitting.
- Support for every instrument format.
- GUI or dashboard features.
- Complete CA/CP analysis parity with CV.
- Public animation workflow or top-level animation API.
- Manuscript-ready figures.

## Meeting-Ready Definition Of Done

The package is ready to introduce to lab users when:

- A new user can follow one quickstart path.
- The default template notebook exists.
- At least one known-good demo dataset works.
- Uploaded data has been triaged enough to identify obvious failures.
- Unsupported workflows are documented.
- `pytest -q` passes locally.
- Installation instructions are accurate.
- Users know what files to bring, what to test, and how to report problems.

## Public GitHub Beta Handoff

Goal: beta users should be able to install directly from the public GitHub repository.

Minimum steps:

1. Make sure the beta-ready branch is merged to the public-facing default branch, or create a clearly named beta branch/tag such as `v0.1.0b2`.
2. Confirm the repository visibility is public on GitHub.
3. Add/update README install instructions for GitHub install:
   - editable local clone for users who may inspect notebooks:
     `git clone https://github.com/ljelissiry/eCAT.git`
     `cd eCAT`
     `python -m pip install -e .`
   - direct pip install:
     `python -m pip install "git+https://github.com/ljelissiry/eCAT.git@v0.1.0b2"`
4. Push the committed beta code and tag/release.
5. Test install from the public URL in a clean environment.
6. Send users the GitHub link, setup notebook, beta scope, and feedback form.

Current readiness:

- Local package, docs, tests, and wheel are close to beta-ready.
- Public GitHub handoff is not complete until the beta branch/tag is pushed, repository visibility is confirmed public, and a clean install from the GitHub URL succeeds.
