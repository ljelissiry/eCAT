# eCAT Package Audit - 2026-09-01

## Scope

This audit used two passes:

1. Inventory and discovery without source changes.
2. Test-first repair of confirmed defects, followed by full verification.

The audit covered the public Python API, typed options and `describe_options()`,
supported import paths, numerical analyses, plotting, optional simulation and app
extras, packaged examples, notebooks, source and wheel builds, and supported Python
versions available locally.

## Baseline

- Baseline commit: `35c72095` (`Fix nested option forwarding contracts`)
- Package version: `0.1.0b4`
- Full-extra interpreter: CPython 3.14.5
- Additional core interpreters: CPython 3.10.20, 3.11.15, and 3.12.13
- Python 3.13 remains covered by the repository CI matrix.

## Pass 1 Results

### Verification that passed

- Full test suite with all optional extras: 1,708 passed, 11 skipped.
- Core CI-equivalent suite: 1,507 passed and 11 skipped on Python 3.10,
  3.11, and 3.12.
- All numbered quickstart notebooks (`00` through `12`, plus `99`) executed
  top-to-bottom in an isolated environment.
- Executed notebook output contained no private absolute paths, tracebacks, or
  warning output.
- Source distribution and wheel builds completed successfully.
- A clean wheel install imported both `ecat` and `ecat_app`, and the installed
  browser app returned HTTP 200.
- All 74 documented option surfaces returned non-empty schemas with no duplicate
  displayed options or missing descriptions.
- Supported example text folders imported without parser warnings.

### Coverage

The all-extra test run measured 83% statement coverage overall. Core scientific
modules were generally between 83% and 92%. Lower-coverage public areas were the
app callbacks (65%), app core (65%), export/import helpers (71%), result display
(76%), and preprocessing (77%).

## Finding Register

| ID | Severity | Finding | Pass 2 disposition |
|---|---|---|---|
| A-001 | P1 | The installed app advertises packaged example folders, but those folders are absent from wheels. | Fixed: app examples are wheel data and installed-prefix resolution is tested. |
| A-002 | P1 | `get_data({"print": False})` emits search, discovery, unsupported-file, and conversion messages despite its public print contract. | Fixed: ordinary output is silent; explicit troubleshoot output remains available. |
| A-003 | P1 | CP cycle efficiency uses unguarded division and can emit divide-by-zero warnings or infinite values. | Fixed: undefined ratios return `NaN` without runtime warnings. |
| A-004 | P2 | `scipy.optimize.OptimizeWarning` leaks from the shared scatter-fit path even when eCAT already reports an underdetermined fit. | Fixed: the warning is captured as fit metadata and a `Fit Warning` table row. |
| A-005 | P3 | A plain core-only `pytest` run imports app-only tests and fails when Dash is not installed, although the package correctly keeps app dependencies optional. | Fixed: app-only tests skip explicitly without the app extra. |
| A-006 | P3 | Package builds warn that the current setuptools license metadata is deprecated and must be migrated before 2027-02-18. | Fixed: package metadata now uses PEP 639 SPDX/license-file fields and setuptools 77+. |
| A-007 | P3 | The installed `ecat-app` command has no `--version` option. | Fixed: `ecat-app --version` reports the canonical package version. |
| A-008 | P3 | Ruff reported 574 existing findings, led by broad exception handling, imports, and unused names. This is technical debt rather than 574 runtime defects. | Report only; four focused correctness findings were removed, leaving 569 broader lint findings. |
| A-009 | P2 | The app derived folder-load status by scraping `get_data()` console output. | Fixed: status is generated from the loaded-object count. |
| A-010 | P3 | The option metadata contained duplicate literal keys, one object helper used a mutable default, and progress formatting contained a dead branch. | Fixed with focused static checks. |

## Skipped Private-Data Tests

Eleven tests were skipped because their private DPV or real-example source files
were not present. The skips were explicit and did not mask failures in the public
sample-data workflows.

## Pass 2 Verification

- Full all-extra suite with `RuntimeWarning` and SciPy `OptimizeWarning` promoted
  to errors: 1,714 passed, 11 skipped.
- Plain core-only suite on Python 3.12: 1,525 passed, 13 skipped. App-only tests
  skipped instead of failing when Dash was absent.
- Changed-path regression tests on Python 3.10 and 3.11: 46 passed, 1 optional
  app test skipped on each interpreter.
- Focused Ruff correctness rules (`F601`, `RUF034`, `B006`) passed.
- Source distribution and wheel rebuilt without license-metadata deprecation
  warnings.
- Clean wheel install exposed all three advertised app example folders:
  13 Fe/PhOH CV files, 4 CA/CPE files, and 1 CP file.
- Installed `ecat-app --version` reported `ecat 0.1.0b4`.
- The freshly installed wheel app loaded all 13 Fe/PhOH example objects and
  served the `eCAT Workbench` page over localhost with HTTP 200.
- All 14 numbered notebooks executed successfully after the fixes. Captured
  output contained no tracebacks, private user paths, or workflow-only paths.

## Residual Risk And Follow-Up

- No P0 scientific/data-corruption finding was confirmed in this audit.
- Eleven private-data tests remain skipped when their private fixtures are not
  present.
- Python 3.13 is covered by CI but was not available as a local interpreter for
  this run. Python 3.10, 3.11, 3.12, and 3.14 were exercised locally.
- Windows behavior was not exercised locally, and the current CI jobs run only on
  Ubuntu. A native Windows CI job remains necessary for path, launcher, wheel,
  packaged-data, and browser-app coverage.
- The remaining 569 Ruff findings should be reduced incrementally. Broad
  exception handling and silent exception paths deserve the highest priority;
  they should not be mass-fixed without behavior tests.
- App callback, export/import, result-display, and preprocessing coverage remain
  the clearest test-expansion opportunities.
