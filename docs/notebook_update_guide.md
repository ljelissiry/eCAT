# Notebook Update Guide

This note captures the conventions established while updating
`notebooks/00_install_and_setup.ipynb`. Use it as a reference while revising the
remaining quickstart notebooks so the series feels consistent and beginner
friendly.

## Conventions From Notebook 00

- Start with warm, concrete boilerplate that tells a beginner what the notebook
  checks, what data it uses, and what they should edit for their own workflow.
- Add a short markdown subheader before each new eCAT function or workflow
  concept. The heading should name the function, and the text should explain
  when to use it.
- Prefer eCAT-native helpers over hand-built teaching boilerplate when the
  helper is part of the public API.
- Use `cv` for a single loaded CV object and `cvs` for a list loaded from a
  folder.
- Show `e.echem.from_file()` before `e.get_data()` when the notebook introduces
  loading. The single-file object should be assigned to `cv`.
- After `from_file()`, use `e.show(cv)` before loading the folder instead of
  manually printing object properties.
- Use `e.get_data()` for folder loading and keep loading options explicit enough
  for beginners to copy.
- In notebooks after 02, use a shorter setup block; notebook 00 and 02 already
  teach the full path setup pattern.
- Use `e.show(cv)` for one-object inspection, `e.show_objects(cvs, {...})` for
  folder summaries, and `e.show_groups(groups)` after grouping instead of
  manually building Pandas summary tables.
- Use `e.show_objects(cvs, {"columns": "available"})` before a custom summary
  table when the notebook is teaching column discovery.
- Use `e.describe_options(..., {"print": False, "return": True})` when an
  options table needs to be queried or displayed as a DataFrame.
- Use public eCAT helpers such as `e.filter()`, `e.sort()`, and `e.group()` for
  teaching subsets and groups instead of manual list comprehensions.
- Do not import `matplotlib.pyplot` only to call `plt.show()`. In notebooks,
  leave the plot object as the final expression or assign it to `ax`.
- For simple plot cells, use `cv.plot(...)` or the relevant eCAT plot helper and
  include a short commented save command:

```python
ax = cv.plot({"print": False})
# ax.figure.savefig(EXPORT_DIR / "first_cv.png", dpi=300, bbox_inches="tight")
```

## Notebook 00 Changes

- Replaced terse setup language with beginner-facing context.
- Added markdown sections for import/setup, `describe_options`, `from_file`,
  `get_data`, `show`, `show_objects`, and plotting.
- Changed the single-file import from `obj = ...` to `cv = ...`.
- Moved single-CV inspection before folder loading.
- Replaced the manual `summary_rows` DataFrame construction with
  `e.show_objects()`.
- Removed unused `pandas` and `matplotlib.pyplot` imports.
- Removed `plt.show()` and added a commented figure-save line.

## Notebook 02 Changes

- Kept `reference mode` set to `"none"` so reference correction is introduced
  intentionally in notebook 03.
- Added `e.show_objects(cvs, {"columns": "available"})` before the custom object
  summary table.
- Used only eCAT helpers for selecting subsets: `e.filter()`, `e.sort()`,
  `e.group()`, and `e.show_groups()`.
- Split `multiplot()` teaching into a basic overlay first, then a separate
  customization section for trace labels, legend placement, colorbar gradients,
  scale bars, titles, and subtitles.
- Keep notebook 02 focused on the everyday workflow: load, inspect, filter,
  plot one CV, make one basic `multiplot()`, group, save/export. Detailed
  plotting mechanics now live in notebook 04.
- Demonstrated axis limits with `ax.set_xlim()` / `ax.set_ylim()` on the
  returned Matplotlib axes; these are not currently `multiplot` options.
- Used `EXPORT_DIR` for saved figures and exported the full relevant processed
  subset with `e.save_data(...)`.

## Notebook 03-09 Pass

- Updated notebooks 03-09 to use the shorter post-02 setup style.
- Introduced reference correction in notebook 03 with both `reference mode:
  "file"` and `reference mode: "keyword"` so later CV analysis uses referenced
  potentials intentionally.
- Removed `matplotlib.pyplot` imports and `plt.show()` calls.
- Used `e.show()`, `e.show_objects()`, and eCAT filter/sort/group helpers as the
  default teaching surface.
- In notebook 03, plotted peak-potential and peak-current analysis diagnostics
  on one graph.
- Added beginner examples for `half_peak_potential()`, `peak_info()`,
  `half_wave_potential()`, and `wave_info()` using a single catalytic trace and
  a cleaner Fc/Fc+ paired-wave trace.
- In notebook 03, used `multiplot()` for the CV traces and
  `peak_current({"plot CV": False})` for overlaying peak-current diagnostics
  without redrawing each CV.
- Keep notebook 03 focused on CV metrics and diagnostic helpers. Do not put
  physical normalization, current standardization, or `i/ip0` workflows here.
- Use notebook 04 for focused plotting mechanics: `cv.trim()` for one trace,
  `e.trim()` for lists/groups, `multiplot()`
  labels, titles/subtitles, legend location, outside legends, axis limits on
  the returned `ax`, scale bars, `gradient by`, `legend mode: "colorbar"`,
  `colorbar tick labels`, `colorbar trace ticks`, `gradient colormap`,
  `gradient reverse`, `gradient scale`, `multimultiplot()`, and saving polished
  figures.
- Put DPV basics in notebook 05 after the plotting-focused notebook. Keep it
  focused on loading DPV files, pulse metadata, Ar/CO2 overlays,
  `peak_potential()`, and `fit_overlapping_peaks()` with explicit guesses.
- Move CA/CP basics to notebook 06 after DPV.
- Move normalization workflows to notebook 07: physical dimensionless
  `normalize()` on a scan-rate series, standardized current with
  `scale_current()` on a titration, and normalized current with
  `normalize_current()` / `i/ip0` on the same titration.
- Kept advanced fitting/statistics material in notebook 07 and later.
- Left workflows that need user-supplied chemistry, validated references, or
  optional simulation backends as disabled templates.

## Questions For The Next Notebook Pass

- Notebooks after `02_basics_load_plot_filter.ipynb` can use a shorter loading
  block because the full setup pattern has already been shown.
- `reference mode` should default to `"none"` until notebook 03 explicitly
  introduces reference correction.
- Notebook outputs should be committed rendered so the notebooks are readable on
  GitHub and useful as static examples.
- Filter notebooks can use the full relevant series, not just a single
  representative CV.
- Save/export examples should use `EXPORT_DIR` consistently in every notebook.
- Avoid advanced fitting/statistics terminology until notebook 07.
