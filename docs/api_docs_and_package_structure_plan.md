# API, Documentation, And Package Structure Plan

This plan captures the agreed direction for making eCAT easier for new users to discover, easier to document, and easier to maintain without changing notebook-facing workflows before the lab beta.

## Goals

- Make first-time user help easier to find even if users skip the quickstart notebooks.
- Keep a fairly rich top-level API for existing analysis workflows.
- Keep simulation available through the `ecat.simulation` namespace rather than exporting simulation helpers at top level.
- Add a curated function and class index to the documentation.
- Clean the `dir(ecat)` and `help(ecat)` experience.
- Split the large implementation into focused `.py` files in small, tested stages.
- Rename the Word package details document to `eCAT Guide` and make it useful while using Markdown docs as the maintainable source for repo-based documentation.

## Recommendation Summary

Use both the Word document and Markdown docs.

- The Word document should be renamed from `eCAT Package Details` to `eCAT Guide` and become the polished lab-facing guide.
- Markdown docs should be the source-of-truth working docs that are easier to diff, review, and update with code changes.
- The API/function/class index should exist in Markdown first, then be incorporated into the Word document.
- The top-level API should stay generous for data loading, plotting, options, CV analysis, and batch analysis, but simulation should remain under `ecat.simulation`.
- File splitting should happen after the public API index is written, because the index clarifies which names must remain stable.

## Proposed Order

### 1. Write The API Reference Markdown Page

Create `docs/api_reference.md`.

Purpose:

- Give users a readable function and class index.
- Define the intended public API before changing `__init__.py`.
- Provide text that can be copied or adapted into the Word document.

Suggested sections:

1. Getting help inside Python
   - `help(ecat)`
   - `help(ecat.cv)`
   - `help(ecat.cv.peak_current)`
   - `ecat.describe_options()`
   - `ecat.describe_options("plot")`

2. Core objects
   - `echem`
   - `cv`
   - `ca`
   - `cp`
   - `dpv`

3. Loading data
   - `echem.from_file`
   - `get_data`
   - `get_CVs`
   - `get_CVs_from_excel`

4. Inspecting objects
   - `info`
   - `stats`
   - `x`
   - `y`
   - `xy`

5. Plotting
   - object-level `plot`
   - `multiplot`
   - `multimultiplot`
   - `multi_scatterplot`
   - `plotting_style`
   - `scale bar` plot option

6. Filtering and grouping
   - `filter`
   - `sort`
   - `group`
   - `sort_and_group`
   - `group_summary`

7. CV analysis
   - `peak_potential`
   - `peak_current`
   - `half_peak_potential`
   - `half_wave_potential`
   - `current_at_potential`
   - `normalize`
   - `normalize_current`
   - `scale_current`

8. Advanced analysis
   - `fowa`
   - `sevcik_analysis`
   - `trumpet_analysis`
   - `nicholson_analysis`
   - `tafel_analysis`
   - `fit_rate`
   - `plateau_current`
   - `fit_peak_potential`
   - `fit_peak_current`

9. Options and defaults
   - `describe_options`
   - `get_defaults`
   - `set_defaults`
   - `reset_defaults`
   - `load_defaults`

10. Simulation
   - `ecat.simulation`
   - Keep simulation helpers documented as experimental or preview.
   - Do not export simulation helpers directly at top level unless later requested.

11. Legacy or internal helpers
   - Explain that not every object visible in older notebooks is part of the stable public API.

### 2. Add The Function And Class Index To The eCAT Guide

Rename `eCAT Package Details.docx` to `eCAT Guide.docx`, then update it after `docs/api_reference.md` is drafted.

Recommended `eCAT Guide` structure:

1. Quick orientation
2. Installation
3. Five-minute usage path
4. Function and class index
5. Loading data
6. Plotting and grouping
7. CV analysis
8. CA/CP basics and limitations
9. Advanced analysis
10. Simulation preview
11. Troubleshooting
12. Beta scope and limitations
13. Detailed function dictionary

The Word doc should not be only a dictionary. It should have a short workflow-first front section, then a compact function/class index, then the current detailed per-function dictionary as an appendix-style reference near the end. That keeps the guide approachable for new users while preserving the detailed material.

Recommended placement for the current dictionary:

- Put a compact index in section 4 so users can orient quickly.
- Move or retain the detailed per-function entries in section 13, `Detailed function dictionary`.
- Cross-reference from workflow chapters to the detailed dictionary instead of making users read the dictionary first.
- Mark functions as `Core`, `Advanced`, `Experimental`, `Legacy`, or `Internal/helper` where useful.

### 3. Clean The Top-Level API

Change `src/ecat/__init__.py` so `dir(ecat)` and `help(ecat)` are useful.

Preserve a rich top-level API for common workflows:

- Core classes: `echem`, `cv`, `ca`, `cp`, `dpv`
- Loading: `get_data`, `get_CVs`, `get_CVs_from_excel`, `create_cv_objects_from_excel`
- Plotting: `multiplot`, `multimultiplot`, `multi_scatterplot`
- Filtering/grouping: `filter`, `sort`, `group`, `sort_and_group`, `group_summary`
- Object display: `show`, `show_objects`, `show_groups`
- Export: `save_data`
- CV and batch analysis: `fowa`, `sevcik_analysis`, `trumpet_analysis`, `nicholson_analysis`, `tafel_analysis`, `fit_rate`, `plateau_current`, `fit_peak_potential`, `fit_peak_current`
- Options/defaults/style: `describe_options`, `get_defaults`, `set_defaults`, `reset_defaults`, `load_defaults`, `plotting_style`
- Option dataclasses that users may reasonably pass directly.

Do not export at top level:

- `np`
- `pd`
- `plt`
- Matplotlib classes
- Imported standard-library modules
- Low-level formatting helpers
- Internal parser utilities
- Animation helpers for now: `animate`, `save_animation`
- Removed plotting-style helper names: `default_plotting`, `use_ecat_plot_style`, `scale_bar`
- Legacy aliases that tests already say are removed: `rate_fit`, `Ep_fit`, `Nicholson`, `PlateauCurrent`, `fit_ip`, `Sevcik`, `FOWA`, `Tafel`, `peak_potential_fit`, `peak_current_fit`, `fit_trumpet`, `fit_sevcik`
- Simulation helper functions, except the `simulation` namespace itself

Current object display functions:

- `show` is the smart user-facing function for displaying one eCAT object, a flat object list, grouped objects, or result objects with `.show()`.
- `show_objects` is the explicit user-facing function for displaying a collection of eCAT objects.
- `show_groups` is the explicit user-facing function for displaying already-created groups.
- `build_object_table` constructs the object summary table and is useful for tests and possibly advanced notebook users.
- `display_object_table` handles pretty display of a table and is more likely an internal/helper function.

Proposed default:

- Keep `show`, `show_objects`, and `show_groups` at top level.
- Ask before deciding whether `build_object_table` should remain top-level public or move to an advanced/internal category.
- Hide `display_object_table` from curated `__all__` unless you want users to customize pretty display directly.

Functions to ask about before hiding from top level:

- `build_object_table`: useful advanced table builder, but helper-ish.
- `save_data`: user-facing export helper; keep top-level public.
- `save_animation`: hide from top-level public API for now; animation is not beta-core.
- `animate`: hide from top-level public API for now; animation is not beta-core.
- `plotting_style`: canonical top-level plotting-style helper that enables, disables, or switches style profiles.
- `scale bar`: incorporated as a `plot`/`multiplot` option instead of a standalone top-level helper.
- `echem_similar_different`: unclear whether this is a user workflow or a helper.
- `get_available_filter_values`: useful discovery helper for filtering; likely worth keeping public.
- `get_sort_group_dict`: helper for grouping; likely hide unless users already call it.
- `get_CVs_from_excel`: clearer top-level public name for spreadsheet-created CV datasets.
- `create_cv_objects_from_excel`: older compatibility name for Excel CV import.
- `normalize`: public batch helper for CV objects only.

Current decisions:

- Keep `save_data` top-level.
- Do not include animation helpers in the curated top-level API for this beta pass.
- Use `tafel_analysis` as the canonical Tafel helper name.
- Use `plotting_style` as the canonical plotting-style API; remove `default_plotting` and `use_ecat_plot_style`.
- Include scale bars through `plot` and `multiplot` options; remove the standalone `scale_bar` function.
- Keep spreadsheet CV import top-level as `get_CVs_from_excel`.
- Keep top-level `normalize`, but have it raise a friendly error for non-CV objects.

Plotting/style helper recommendation:

- `plotting_style("notebook")` applies the notebook style.
- `plotting_style("publication")` applies the publication style.
- `plotting_style(False)`, `plotting_style("default")`, or `plotting_style("matplotlib")` restores Matplotlib defaults.
- `scale bar` is a plotting option. `length` is interpreted in the currently displayed y-axis unit. A `loc` tuple is interpreted as `(x, y)` in the currently displayed data coordinates after unit conversion/scaling.
- Scale bars should work through ordinary `echem.plot()` paths, including CV/CA/CP-style plots, and through `multiplot`.

Excel and normalization recommendation:

- `create_cv_objects_from_excel` is a specialized importer for curated Excel workbooks with two-row headers, paired/shared potential-current columns, and optional referenced axes. It is useful, but it is not the main user loading path.
- Use `get_CVs_from_excel` as the clearer public name.
- Keep `create_cv_objects_from_excel` as a compatibility name unless there is a later explicit removal pass.
- Keep top-level `normalize` public. It returns normalized CV copy/copies without mutating the inputs, while `cv.normalize()` mutates one CV object. It should reject non-CV objects with a clear CV-only error.

Testing:

- Add a test that required public names are present in `ecat.__all__`.
- Add a test that obvious internals are absent from `ecat.__all__`.
- Add a test that `import ecat as e` still supports common beta workflows.

Compatibility note:

- Cleaning `__all__` affects `from ecat import *`.
- It does not necessarily need to remove legacy attributes immediately.
- The safest beta approach is to curate `__all__` and top-level docs first, then decide whether to remove direct access to internals after beta.

### 4. Improve `help(ecat)`

`help(ecat)` is already the package help entry point. The improvement comes from:

- a better package docstring,
- a curated `__all__`,
- less namespace noise,
- clear public classes/functions,
- and a pointer to `describe_options()`.

Suggested package docstring content:

- one-sentence package purpose,
- common first calls,
- options discovery,
- simulation namespace note,
- documentation pointers.

Example:

```python
"""eCAT: electroCatalysis Analysis Tools.

Common starting points:
    echem.from_file(path)
    get_data({"folder path": path})
    multiplot(objects)
    describe_options()

Core objects:
    cv, ca, cp, dpv

Analysis examples:
    fowa(cvs)
    fit_peak_current(cvs)
    tafel_analysis(cv_obj, TOF_max, E_thermo, E_redox)

Simulation helpers live under:
    ecat.simulation
"""
```

### 5. Split Files In Small Tested Slices

Do not do a broad move all at once. Use the API reference and `__all__` tests as guardrails.

Recommended split order:

1. `options.py`
   - Already exists and should remain the options/defaults home.
   - Keep improving this first because many public workflows depend on option discovery.

2. `metadata.py`
   - Filename parsing, concentration/species/gas/solvent extraction, chemical formatting that is metadata-specific.
   - Good first split because it is testable and user-facing.

3. `parsers.py`
   - CH, BASI, EC-Lab, generic parser helpers.
   - Keep public loading functions stable.
   - Add parser equivalence tests as guardrails.

4. `io.py`
   - `from_file` dispatch and `get_data`.
   - Reference file discovery may stay here initially or move to `reference.py` once stable.

5. `reference.py`
   - Reference shifting, reference-source selection, midpoint/reference helpers.
   - Move only after reference-shift regression tests are strong.

6. `plotting.py`
   - `multiplot`, `multimultiplot`, plot styling helpers.
   - Move after plotting smoke tests are green.

7. `objects.py`
   - `echem`, `cv`, `ca`, `cp`, `dpv`.
   - This is higher risk because many methods currently live on classes.
   - Consider moving helper functions first, then classes later.

8. `analysis_cv.py`
   - CV-specific object methods and helper routines.
   - Move carefully because return shapes and notebook workflows matter.

9. `analysis_batch.py`
   - `fowa`, `sevcik_analysis`, `trumpet_analysis`, `nicholson_analysis`, `tafel_analysis`, `fit_rate`, `plateau_current`, `fit_peak_potential`, and `fit_peak_current`.
   - Move after public API and doc references are stable.

10. `export.py`
   - CSV/table/figure export helpers.

11. `utils.py`
   - Units, labels, generic formatting helpers.
   - Avoid making this a miscellaneous junk drawer; move only stable, clearly shared helpers.

12. Historical `core.py`
   - Do not keep as a beta-facing compatibility layer.
   - Remove before lab beta once imports and docs no longer depend on it.

### 6. Notebook Work

Keep the beta notebook plan from `docs/lab_beta_meeting_todo.md`.

Add the default template notebook as a required deliverable:

- `notebooks/01_default_template.ipynb`

Purpose:

- Users copy it for their own data.
- It should contain obvious variables to edit.
- It should load data, inspect objects, plot, filter/group, run basic analysis, and export results.

The template should complement the API reference:

- The API reference answers "what exists?"
- The default notebook answers "what do I run?"

## Suggested Implementation Sequence

1. Draft `docs/api_reference.md`.
2. Rename/update the Word doc as `eCAT Guide` with a workflow-first structure and the function/class index.
3. Add tests for the intended top-level public API.
4. Curate `src/ecat/__init__.py` and `__all__`.
5. Improve the package docstring so `help(ecat)` is more useful.
6. Run `pytest -q`.
7. Create `notebooks/01_default_template.ipynb`.
8. Begin file splitting with metadata/parser/options-adjacent code.
9. Split loading/reference/plotting only with focused regression tests.
10. Split object classes and CV analysis later, after beta-critical docs and notebooks are stable.

Status after the split passes:

- Added focused modules for `objects`, `io`, `parsers`, `reference`, `plotting`, `analysis_cv`, `analysis_batch`, `export`, and `metadata`.
- Updated top-level imports to flow through those modules while preserving the existing public API and object identities.
- Added module-boundary regression tests so future physical moves out of `core.py` can happen one slice at a time.
- Physically moved low-risk metadata helpers, parser timestamp/quiet-time helpers, midpoint/reference math, plotting-style profile handling, shared plotting helpers such as scale bars/title/reference-label formatting, export behavior, and CV-specific text/Excel loading helpers out of `core.py`.
- Removed the historical `ecat.core` compatibility facade before beta so new notebooks and docs point to `import ecat as e` and the focused modules.
- Physically moved object classes into `ecat.objects`, main plotting/object-list bodies into `ecat.plotting`, CV normalization/current-scaling bodies into `ecat.analysis_cv`, and batch/advanced analysis bodies into `ecat.analysis_batch`.
- Removed the temporary `ecat._core_impl` module. Generic helpers now live in `ecat.utils`, filtering/grouping in `ecat.collection`, reference helpers in `ecat.reference`, and loading orchestration in `ecat.io`.

## Decisions To Revisit After Beta

- Whether to remove legacy top-level helper attributes completely.
- Whether simulation helpers should stay namespace-only or become top-level public API.
- Whether the `eCAT Guide` Word document should continue as the main user guide or become a polished export of the Markdown docs.

## Open Implementation Notes

- Any file split should preserve public names, argument names, return types, and notebook-facing behavior.
- Add tests before each risky move.
- Do not use the module split as an opportunity to redesign scientific workflows.
- Update Markdown docs first, then the Word document for user-facing changes.
