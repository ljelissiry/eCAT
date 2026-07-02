# Changelog

## Unreleased

- No changes yet.

## 0.1.0b3 - Internal Lab Beta Refresh - 2026-07-01

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
- Fixed FOWA diagnostic plot labeling so automatic diagnostic labels are delegated to `multiplot`, preserving gradient/colorbar label behavior while retaining explicit `plot labels` support.
- Improved reference-shift handling, path display behavior, plotting smoke coverage, option defaults, package import checks, and public API reference cleanup.
- Updated the eCAT Guide, beta scope, API reference, and package metadata for the current beta state.

## 0.1.0b1 - Internal Lab Beta

- Documented the lab beta scope, supported file/technique matrix, limitations, quickstart, beta tester guide, troubleshooting notes, and bug-report template.
- Added beta-readiness tests for existing object accessors, summaries, CA/CP plotting smoke behavior, figure saving, and CSV export.
- Marked package metadata as beta release `0.1.0b1`.
- No core workflow behavior, public return shapes, or scientific algorithms were intentionally changed in this readiness pass.
