# eCAT Lab Beta Scope

This document defines what is supported for the internal lab beta of eCAT `0.1.0b3`. The beta goal is to validate real lab workflows without implying that every parser, technique, or analysis path is manuscript-ready.

## Supported Files, Techniques, and Limitations

| Format | Technique | Status | Tested Workflows | Limitations | Beta Guidance |
|---|---|---:|---|---|---|
| CH text | CV | Supported | Load, `get_data()`, recursive folders, plot, filter/group, reference shift, peak potential/current, normalization, save figure/table | Scientific review is still required for tangent-baseline choices and unusual wave shapes | Recommended beta workflow |
| CH text | CA | Limited | Load, class promotion to `ca`, metadata parsing, plot smoke | CA-specific analysis utilities are not broadly validated for beta | Use with caution; report import/plot issues |
| CH text | CP | Limited | Load, class promotion to `cp`, metadata parsing, plot smoke | CP analysis workflows are not broadly validated for beta | Use with caution; treat as load/plot beta only |
| BASI text | CV | Supported | Load, class promotion to `cv`, metadata/units parsing, plot and peak-analysis compatible object behavior | Header/unit variants beyond tested fixtures may need parser fixes | Recommended if the export resembles tested text files |
| BASI text | CP | Limited | Load, class promotion to `cp`, time/potential parsing, plot-compatible object behavior | Validated against a minimal `[Begin Data]` text shape; broader BASI CP header variants still need real lab files | Use with caution; report representative exports that fail |
| EC-Lab text | CV | Supported | Load, class promotion to `cv`, mA-to-A conversion, metadata parsing, reference-shift compatible object behavior | Metadata inference depends on recognizable text headers | Recommended with caution; verify units after loading |
| EC-Lab text | CP/GCPL | Limited | Load, class promotion to `cp`, basic time/potential/current metadata | Analysis beyond loading and plotting is not guaranteed | Document-only beta unless the specific workflow passes local tests |
| Generic numeric/header text | CV-like or unknown | Limited | Fallback loading, basic metadata from filename, sort/filter compatibility | Technique detection and metadata inference are intentionally minimal | Fallback only; prefer vendor text exports |
| Binary files | Any | Unsupported | None for beta scope | Binary parser behavior is not beta-ready | Convert/export to text first |

## Beta Core Workflows

The beta validates these existing workflows without redesigning them:

- `import ecat as e`
- `e.echem.from_file(...)`
- `e.get_data(...)`
- `e.filter(...)`, `e.sort_and_group(...)`, and replicate filtering including `replicate = -1`
- `x()`, `y()`, `xy()`, `stats()`, and `info()`
- CV plotting, `multiplot`, and smoke-level figure export
- CV `peak_potential()` and `peak_current()`
- CV normalization, current density, and `i/ip0` plotting paths
- Reference-shift workflows already covered by tests
- CSV/table export and eCAT Excel workbook export through `save_data(...)`

## Known Limitations

- Beta support means "tested enough for lab feedback," not "fully validated for publication."
- Tangent-baseline peak current is scientifically sensitive; users should inspect fits for noisy, shallow, irreversible-looking, or multi-segment CVs.
- FOWA and other advanced analysis outputs should be treated as mechanistic descriptors only when the assumptions are appropriate.
- CA and CP are limited to import/plot confidence unless a specific analysis function has a passing test and human review.
- Generic text loading is a fallback and may miss technique metadata, scan rate, units, or reference context.
- Binary vendor files are out of scope for this beta.

## Recommended Beta-User Action

Start with a folder of text exports, load with `get_data()`, inspect `obj.info()` and `obj.units`, plot the traces, run one core CV analysis, and export one figure/table. Report any mismatch between expected and observed behavior with the bug-report template.
