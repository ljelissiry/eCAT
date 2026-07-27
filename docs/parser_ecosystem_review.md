# External Electrochemistry Parser Ecosystem Review

Last reviewed: July 2, 2026

This document summarizes how nearby open-source electrochemistry packages parse
instrument files and what eCAT can learn from them. It is a planning reference,
not a commitment that these formats are currently supported in eCAT.

## Bottom Line

- `yadg` is the best current reference for BioLogic EC-Lab `.mpt` and `.mpr`
  parsing, but it is GPL-3.0. Use it as a behavioral reference, not copied source.
- `ixdat` is the best reference for a broad potentiostat-reader interface. It is
  MIT licensed and has readers for BioLogic, AutoLab/NOVA ASCII, Ivium, and
  Nordic TDMS.
- `galvani` is useful for BioLogic binary `.mpr` and Arbin `.res` ideas, but it
  is GPL-3.0 and its Arbin route is a database-conversion workflow rather than a
  simple text parser.
- For eCAT, the safest path is independent parsers built from small user-provided
  fixture files, with tests asserting eCAT's canonical data/metadata contract.

## Package Summary

| Package | License | Relevant Formats | Main Strategy | eCAT Use |
|---|---|---|---|---|
| `yadg` | GPL-3.0 | BioLogic EC-Lab `.mpt`, `.mpr` | Convert EC-Lab text/binary structures into `xarray.DataTree` with rich metadata, units, timestamps, and estimated uncertainty channels | Study behavior and metadata model; do not copy GPL source into MIT eCAT |
| `ixdat` | MIT | BioLogic `.mpt`/`.mpr`, AutoLab/NOVA ASCII, Ivium `.txt`, Nordic `.tdms` | Reader classes convert files into ixdat `ECMeasurement` objects made of time/value series plus aliases and metadata | Best architectural inspiration for pluggable eCAT readers |
| `galvani` | GPL-3.0 | BioLogic `.mpt`, `.mpr`; Arbin `.res` via converter | BioLogic text to NumPy record arrays; BioLogic binary reverse-engineered module parsing; Arbin `.res` to SQLite using `mdbtools` | Useful as reference/test oracle; avoid bundling/copying |
| `eclabfiles` | GPL-3.0, archived | BioLogic EC-Lab files | Older BioLogic parser that returns pandas-style data with attrs | Historical context only; prefer `yadg` for future review |

## Supported File And Experiment Types Observed

| Vendor / Format | Package Support Observed | Experiment / Technique Coverage |
|---|---|---|
| BioLogic EC-Lab `.mpt` text | `yadg`, `ixdat`, `galvani`, historical `eclabfiles` | `yadg` explicitly maps CA, CP, CV, CVA, GCPL, GEIS, LOOP, LSV, MB, OCV, PEIS, WAIT, ZIR, MP, CoV, CoC, and BCD. `ixdat` treats these as electrochemistry measurements and preserves the EC-Lab sub-technique string when it can. |
| BioLogic EC-Lab `.mpr` binary | `yadg`, `galvani`; `ixdat` delegates to `galvani` then `eclabfiles` | Broad EC-Lab binary support, but reverse-engineered and version-sensitive. `ixdat` warns that `.mpr` reading is discouraged and recommends `.mpt` when possible. |
| Arbin `.res` | `galvani` | Battery/cycler tables such as global metadata, resume data, normal channel data, statistics, auxiliary data, event tables, smart-battery tables, and version-specific CAN/BMS or auxiliary tables. This is not a CV-style potentiostat parser. |
| AutoLab NOVA ASCII | `ixdat` | Generic electrochemistry time/potential/current exports with semicolon-delimited ASCII and NOVA column names. Technique-specific metadata appears limited compared with EC-Lab. |
| IviumSoft `.txt` | `ixdat` | Generic electrochemistry exports with timestamp line, column line, whitespace-delimited data, and optional multi-file dataset assembly into cycles. |
| Nordic Electrochemistry `.tdms` | `ixdat` | EC time/potential/current channels plus optional EIS channels when present. Requires `nptdms`; can parse a nearby `.EC_Macro` file for experiment-sequence metadata. |

## Parsing Strategies By Format

### BioLogic EC-Lab `.mpt`

`yadg` expects the `EC-Lab ASCII FILE` magic line, reads `Nb header lines`, splits
the file into header and data regions, and parses:

- settings from header sections
- technique name and technique parameters
- acquisition timestamp and locale/timezone-sensitive dates
- loops and external-device information where present
- column names through an explicit EC-Lab column-unit map
- numeric data into an `xarray.Dataset` inside a `DataTree`
- absolute Unix time (`uts`) by combining relative time with acquisition time
- control/measured potential and current separation
- estimated uncertainty variables for voltage/current channels

`ixdat` uses a simpler line parser for `.mpt`: it searches header lines for
header count, acquisition timestamp, loop definitions, and EC-Lab sub-technique,
then builds a shared time series plus value series from tab-delimited columns.
It uses aliases such as `raw_potential`, `raw_current`, `raw_CE_potential`, `t`,
and `cycle`.

`galvani` exposes lower-level BioLogic text utilities: it checks the EC-Lab/BT-Lab
magic line, skips header comments, converts field names to NumPy dtypes, handles
comma decimal separators, and returns record arrays plus header comments.

### BioLogic EC-Lab `.mpr`

`yadg` parses the binary module structure directly. It checks the BioLogic binary
magic bytes, reads settings/data/log/loop modules, maps binary column IDs to names,
dtypes, and units, handles flags and ambiguous/technique-dependent column IDs, and
stores original metadata under dataset attrs.

`galvani` also reverse-engineers the `.mpr` module structure. It identifies VMP
settings, data, loop, and log modules; derives dtype structure from column IDs;
handles unknown column IDs either strictly or by trying common dtypes; extracts
start/end dates and an OLE timestamp from log modules; and exposes loop indices.

`ixdat` does not implement a full native `.mpr` parser. It tries `galvani` first,
then `eclabfiles`, and warns users to prefer `.mpt` exports because `.mpt` gives
ixdat more useful metadata such as loop numbers.

### Arbin `.res`

`galvani` handles Arbin through `res2sqlite.py`, which is closer to database
migration than instrument-file parsing. It uses `mdbtools`/`mdb-export` to read
Access-style `.res` tables, creates known SQLite schemas, loads text and numeric
tables, detects the Arbin schema version, and builds helper capacity/energy
summary tables.

The supported table strategy is useful for battery/cycler workflows, but eCAT
would need an additional translation layer from SQLite tables to eCAT objects.
The most relevant tables for eCAT-style summaries would be channel normal data,
resume/step/cycle data, channel statistics, auxiliary data, event data, and global
metadata.

### AutoLab NOVA ASCII

`ixdat` reads AutoLab NOVA ASCII with pandas using semicolon delimiters. It expects
columns such as:

- `Time (s)`
- `WE(1).Potential (V)`
- `WE(1).Current (A)`

It extracts units from the final parenthesized text in each column name and builds
time/value series. Timestamp handling is weaker than EC-Lab: the caller can supply
`tstamp` or `timestring`; otherwise ixdat prompts for the timestamp.

### IviumSoft `.txt`

`ixdat` assumes the first line is a timestamp, the second line is column names, and
subsequent rows are whitespace-delimited data. It handles a known Ivium quirk where
the data can contain one more column than the header line by assigning
`unlabeled_0`, etc. It extracts units from text after `/` in column names, uses
`time/s` as the time channel, and aliases `E/V`, `I/A`, and `time/s`.

For exported Ivium datasets made of several files with a shared base name, ixdat
can assemble the component files into a single measurement and assign cycle numbers
based on file order.

### Nordic `.tdms`

`ixdat` reads Nordic files through the optional `nptdms` dependency. It expects an
`EC` group with channels:

- `Time`
- `E`
- `i`

It reads TDMS properties such as `name` and `dateTime`, preserves channel
`unit_string` metadata, converts current from A to mA when the TDMS current unit is
`A`, and creates aliases for time, potential, and current. If channels `Z_E` and
`Phase_E` are present, it adds EIS-like data on an inferred evenly spaced time
axis. It also looks for a nearby `.EC_Macro` file and parses it into a list of
step dictionaries.

## Metadata Extracted

| Metadata / Data Feature | `yadg` | `ixdat` | `galvani` | Current eCAT |
|---|---|---|---|---|
| Raw vendor header | Stores `original_metadata` with settings/params/log where available | Keeps reader state such as header lines for BioLogic `.mpt`; metadata dict for Nordic | Returns comments for `.mpt`; module/header fields for `.mpr`; SQLite tables for Arbin | Mostly stores selected fields on objects; no uniform raw header/metadata contract yet |
| Technique/sub-technique | Explicit BioLogic technique mapping and parameter dtypes | `technique="EC"` plus `ec_technique` for BioLogic | BioLogic column/module data, less eCAT-ready technique abstraction | Promotes to `cv`, `ca`, `cp`, `dpv` when supported; generic `echem` otherwise |
| Units | Unit maps attached to data variables/attrs; estimated uncertainty channels | Unit extracted from column labels or TDMS channel properties | Units encoded through dtype/column maps for BioLogic; Arbin schema columns | Object-level `.units` dict plus `data.attrs["units"]`; selected conversions to V/A/s |
| Timestamps | Acquisition timestamp plus relative time into `uts`; OLE log timestamps for `.mpr` | `tstamp` on time series; BioLogic header timestamp; Nordic TDMS `dateTime`; AutoLab can prompt | BioLogic start/end dates and OLE timestamp; Arbin datetime tables | File creation/modification times and limited vendor scan-rate/header metadata |
| Loops/cycles/steps | Parses EC-Lab loop sections/modules and technique params | BioLogic loop number/cycle aliases; Ivium dataset cycle assembly; Nordic macro steps | BioLogic loop index; Arbin step/cycle tables | Segment counting and replicate/file-name metadata; limited vendor loop/step preservation |
| Reference/electrode metadata | Vendor settings may include cell/settings params | Mostly whatever appears in series/metadata | Present if vendor modules/tables expose it | Strong user-facing reference-shift workflow, but parser metadata is mostly eCAT-specific |
| Warnings/unknown fields | Logs duplicate/unknown columns and missing log/timestamp cases | Raises/warns for missing dependencies, missing columns, Nordic macro/EIS channels | Strict or permissive unknown binary columns; Arbin mdbtools dependency errors | Raises user-facing parser errors; newer parse-result structure can store warnings/raw metadata |

## Differences From eCAT

eCAT currently optimizes for notebook-facing electrochemical workflows rather than
full vendor-file fidelity. That means:

- eCAT normalizes common CV/CA/CP data into familiar object columns like
  `Potential`, `Current`, and `Time`.
- eCAT parses lab-facing metadata from filenames and workbook headers, including
  compounds, concentrations, gases, solvents, scan rate, electrodes, replicate
  labels, and reference-shift information.
- eCAT has stronger user-facing reference-shift semantics than these external
  parsers, but weaker raw vendor metadata preservation.
- eCAT stores units in `.units` and `data.attrs["units"]`; `yadg` and `ixdat`
  keep units closer to each variable/series.
- eCAT does not yet have one parser-result contract for all importers containing
  `data`, `units`, `technique`, `software`, `metadata`, `warnings`, and
  `raw_metadata`, though that is the right direction.

The main lesson is that broader file support should not simply add more ad hoc
`if software == ...` blocks. A future parser layer should return a structured
intermediate object that eCAT can promote into `cv`, `ca`, `cp`, `dpv`, or generic
`echem` objects.

## Recommended eCAT Parser Contract

Each future parser should return the same internal structure:

| Field | Purpose |
|---|---|
| `data` | Canonical pandas table with eCAT-standard column names where possible |
| `units` | Dict mapping canonical and preserved vendor columns to display units |
| `technique` | eCAT technique target such as `CV`, `CA`, `CP`, `DPV`, `EIS`, `battery`, or `unknown` |
| `experiment_subtype` | Vendor-specific technique string such as `Cyclic Voltammetry Advanced` or `GCPL` |
| `software` | Vendor/source label such as `EC-Lab`, `NOVA`, `IviumSoft`, `Nordic`, `Arbin` |
| `metadata` | Clean, eCAT-facing metadata used by `info()`, grouping, filtering, and plotting |
| `raw_metadata` | Original header/settings/module/table metadata for debugging and reproducibility |
| `warnings` | Nonfatal parser concerns: missing timestamp, unknown column, inferred unit, unsupported channel |
| `source` | File path plus parser/version information |

## Implementation Guidance For eCAT

1. Start with text exports before binary/container formats.
2. Add tiny public fixtures for each parser variant, plus optional ignored real-file
   fixtures for private regression testing.
3. Implement BioLogic `.mpt` first because it is both common and text-based.
4. Implement AutoLab NOVA ASCII and Ivium `.txt` next because ixdat shows they can
   be parsed with straightforward column/unit conventions.
5. Treat Nordic `.tdms` as an optional-extra parser because it requires `nptdms`.
6. Treat BioLogic `.mpr` and Arbin `.res` as separate projects because they need
   binary/database strategies and careful licensing boundaries.
7. Keep GPL packages out of eCAT's distributed MIT source unless eCAT's licensing
   strategy intentionally changes.

## Source References

- `yadg`: <https://github.com/dgbowl/yadg>
- `ixdat`: <https://github.com/ixdat/ixdat>
- `galvani`: <https://github.com/echemdata/galvani>
- archived `eclabfiles`: <https://github.com/vetschn/eclabfiles>
- ixdat BioLogic reader: <https://github.com/ixdat/ixdat/blob/main/src/ixdat/readers/biologic.py>
- ixdat AutoLab reader: <https://github.com/ixdat/ixdat/blob/main/src/ixdat/readers/autolab.py>
- ixdat Ivium reader: <https://github.com/ixdat/ixdat/blob/main/src/ixdat/readers/ivium.py>
- ixdat Nordic reader: <https://github.com/ixdat/ixdat/blob/main/src/ixdat/readers/nordic.py>
- yadg EC-Lab extractors: <https://github.com/dgbowl/yadg/tree/main/src/yadg/extractors/eclab>
- galvani BioLogic parser: <https://github.com/echemdata/galvani/blob/master/galvani/BioLogic.py>
- galvani Arbin converter: <https://github.com/echemdata/galvani/blob/master/galvani/res2sqlite.py>
