# eCAT Beta Scope

This document defines what is supported for the current eCAT `0.1.0b4` beta.
The beta goal is to validate real electrochemistry workflows across friendly
external users and selected collaborating labs without implying that every
parser, technique, or analysis path is manuscript-ready.

## Supported Files, Techniques, and Limitations

| Format | Technique | Status | Tested Workflows | Limitations | Beta Guidance |
|---|---|---:|---|---|---|
| CH text | CV | Supported | Load, `get_data()`, recursive folders, plot, filter/group, reference shift, peak potential/current/width, normalization, save figure/table | Scientific review is still required for tangent-baseline choices and unusual wave shapes | Recommended beta workflow |
| CH text | CA | Limited | Load, class promotion to `ca`, metadata parsing, plot smoke | CA-specific analysis utilities are not broadly validated for beta | Use with caution; report import/plot issues |
| CH text | CP | Limited | Load, class promotion to `cp`, metadata parsing, plot smoke | CP analysis workflows are not broadly validated for beta | Use with caution; treat as load/plot beta only |
| BASI text | CV | Supported | Load, class promotion to `cv`, metadata/units parsing, plot and peak-analysis compatible object behavior | Header/unit variants beyond tested fixtures may need parser fixes | Recommended if the export resembles tested text files |
| BASI text | CP | Limited | Load, class promotion to `cp`, time/potential parsing, plot-compatible object behavior | Validated against a minimal `[Begin Data]` text shape; broader BASI CP header variants still need real lab files | Use with caution; report representative exports that fail |
| EC-Lab ASCII `.mpt` or compatible `.txt` | CV | Supported | Direct and folder load, `ParseResult` pre-parse, class promotion to `cv`, mA-to-A conversion, scan-rate/timestamp parsing, raw header metadata, reference-shift compatible object behavior | Metadata inference depends on recognizable text headers; unusual EC-Lab column names may need parser fixes | Recommended with caution; inspect `obj.parse_result.warnings` and verify units after loading |
| EC-Lab ASCII `.mpt` or compatible `.txt` | CA | Limited | Direct and folder load, `ParseResult` pre-parse, class promotion to `ca`, time/current/potential columns, unit conversion, step-table/raw-header metadata | CA analysis beyond load/inspect/plot/export is not broadly validated | Use with caution; report representative exports that fail |
| EC-Lab ASCII `.mpt` or compatible `.txt` | CP/GCPL | Limited | Direct and folder load, `ParseResult` pre-parse, class promotion to `cp`, time/potential/current columns, unit conversion, reference-electrode and step-table metadata when present | Analysis beyond loading and plotting is not guaranteed | Document-only beta unless the specific workflow passes local tests |
| NOVA ASCII text | CV | Limited | Load, `ParseResult` pre-parse, class promotion to `cv` when potential/current columns are recognizable, current-unit conversion, raw header metadata | Header metadata is often sparse; scan rate/timestamp may be missing; CA/CP/DPV NOVA workflows need representative files | Use with caution; inspect parser warnings and verify metadata manually |
| IviumSoft text | Any | Experimental/unvalidated | Detection and generic text fallback only when the table resembles supported time/potential/current data | No representative local fixture has been validated; technique-specific promotion is not beta-supported | Please provide representative `.txt` exports before relying on this path |
| Generic numeric/header text | CV-like or unknown | Limited | Fallback loading, CV-like potential/current promotion when columns are recognizable, conservative generic loading for ambiguous time/potential/current tables, basic metadata from filename, parser warnings, sort/filter compatibility | Technique detection and metadata inference are intentionally minimal; missing scan rate is common | Fallback only; inspect `obj.parse_result.warnings` and prefer vendor text exports |
| BioLogic EC-Lab binary `.mpr` | Any | Unsupported | Detected before text parsing with a specific `UnsupportedFileFormatError` | Native binary parsing is not beta-ready | Export EC-Lab ASCII `.mpt`, convert externally, or provide a custom reader for direct loading |
| Other binary files | Any | Unsupported | None for beta scope | Binary parser behavior is not beta-ready | Convert/export to a supported text format first |

## Beta Core Workflows

The beta validates these existing workflows without redesigning them:

- `import ecat as e`
- `e.echem.from_file(...)`
- `e.get_data(...)`
- `e.filter(...)`, `e.sort_and_group(...)`, and replicate filtering including `replicate = -1`
- `x()`, `y()`, `xy()`, `stats()`, and `info()`
- CV plotting, `multiplot`, and smoke-level figure export
- CV `peak_potential()`, `peak_current()`, and `peak_width()`
- CV normalization, current density, and `i/ip0` plotting paths
- Reference-shift workflows already covered by tests
- CSV/table export and eCAT Excel workbook export through `save_data(...)`

## Known Limitations

- Beta support means "tested enough for real-user feedback," not "fully validated for publication."
- Tangent-baseline peak current is scientifically sensitive; users should inspect fits for noisy, shallow, irreversible-looking, or multi-segment CVs.
- FOWA and other advanced analysis outputs should be treated as mechanistic descriptors only when the assumptions are appropriate.
- `reversibility_analysis()` reports cautious, evidence-backed labels for one scan-rate series; it does not assign an EC/ECE or other coupled mechanism. Review the recorded decision tree and diagnostics before using the conclusion in a manuscript.
- `surface_coverage_analysis()` assumes a surface-confined peak and reports both slope- and charge-derived estimates. Disagreement warnings should be resolved experimentally rather than hidden by averaging.
- CA and CP are limited to import/plot confidence unless a specific analysis function has a passing test and human review.
- Parser diagnostics are available through `obj.parse_result.warnings` and `obj.parse_result.raw_metadata`. Use these fields when a file loads but metadata looks incomplete.
- Generic text loading is a fallback and may miss technique metadata, scan rate, units, or reference context.
- IviumSoft text and DPV text beyond existing CH/public-fixture coverage still need representative files before they should be treated as beta-supported.
- Binary vendor files are out of scope for this beta. BioLogic `.mpr` files are rejected explicitly instead of being passed to the text parser.
- Simulation pre-equilibrium is stoichiometric and phase-local. Reactions that
  mix bulk concentrations with surface coverages require a physical
  volume-to-area capacity convention and currently raise an error.
- Simulation equilibrium diagnostics verify numerical conservation and
  residuals, not elemental or charge balance inferred from species names.
- The app Model tab supports single-CV simulation and fitting when the
  `simulation` extra is installed. Multi-CV/global fitting is currently an API
  and notebook workflow; app controls for it remain on the release roadmap.

## Recommended Beta-User Action

Start with a folder of text exports, load with `get_data()`, inspect
`obj.info()` and `obj.units`, plot the traces, run one core CV analysis, and
export one figure/table. Report any mismatch between expected and observed
behavior with the beta feedback form.
