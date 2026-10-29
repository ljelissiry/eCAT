# 5-Minute eCAT Quickstart

```python
import ecat as e
```

## Load a Folder

```python
data = e.get_data({
    "folder path": "path/to/txt_exports",
    "recursive search": True,
    "print": True,
    "reference mode": "none",
})
```

With printing enabled, eCAT shows the import root once and totals for candidate text files, loaded files, failed imports, unsupported files, and reference sources used. If subfolders are visited, a DataFrame lists **every visited folder**, including empty folders and folders containing only unrelated files. Each row counts files directly in that folder; parent rows do not include descendants. References are an overlapping role, not additional files. External reference sources are reported separately. “Loaded” means parsing succeeded, not that the data have been scientifically validated. Binary filename examples and unsupported extension summaries are capped at five entries. `"print": False` suppresses this reporting; `"troubleshoot": True` explicitly enables diagnostic output.

With `"recursive search": False`, only files directly inside the selected folder are considered. The totals, reference summary, and object/conditions display remain; the folder breakdown is omitted. The same compact output applies to recursive searches that visit only the root folder. There is no separate “supported text files found” count. In the folder-reference summary, each source path is relative to the folder on that line: `series/A: Fc.txt` means a local reference, while `series/A/deep: ../Fc.txt` means its parent's reference. Stored reference-source paths are unchanged.

The legacy numeric `columns` import hint is not needed by built-in text readers, which infer the file structure. It remains available to custom readers for compatibility. To choose additional printed metadata, use `"display columns": ["WE", "RE", "CE"]` (or `"all"`).

Known electrode values shared by every file appear once above the object table:

```text
[Conditions] Exp Type: CV, Scan Rate: 100 mV/s, Segments: 2
[Electrodes] WE: GC, RE: Ag/AgCl, CE: Pt
[Reference] Label: Fc/Fc+
```

Each role is handled separately: if only RE varies, WE and CE stay on the shared line and a **Reference Electrode** column shows each file's RE. Missing entries display as `unspecified` and never count as a shared known electrode. Plain-text output includes differing electrodes beside each file instead. RE describes the physical electrode; `[Reference]` describes the potential reporting scale/correction. Original metadata are unchanged.

Explicit `display columns` requests still show the requested electrode columns, even if they are shared. With `"print conditions": False`, the shared headings are suppressed and known electrode values remain in columns (or beside files in plain text).

Check the first object:

```python
obj = data[0]
e.show(obj)
metadata = obj.metadata()
obj.units
```

`metadata()` works consistently for CV, CA, CP, and DPV objects. It returns an independent dictionary with common provenance and units plus the technique-specific values reported by that object's `stats()` method.

## Plot One CV

```python
ax = obj.plot({
    "legend": False,
    "title": True,
})
```

## Filter and Group

```python
co2 = e.filter(data, {"gas": "CO2"}, {"print": False})

co2_titration = e.filter(
    data,
    {
        "species": ["1mM[Co]", "3mMFc", "2.8MH2O"],
        "gas": "CO2",
        "replicate": -1,
    },
    {"print": False},
)

optional_additive = e.filter(
    data,
    {"species": {"any": ["PhOH", "H2O"]}},
    {"print": False},
)

grouped = e.sort_and_group(
    data,
    sort_keys=["gas", "scan rate"],
    group_keys="gas",
    options={"print": False},
)
```

For `compounds`, `concentrations`, and `species`, a list means all listed values
must be present. Use `{"any": [...]}` inside the key when you want any-of
matching. Scalar keys such as `gas` still treat lists as any-of.

Keys ignore case and accept underscores. Solvent filters recognize aliases and tables preserve filename spelling. Each plot combines aliases under the most-used spelling among its entries (first encountered on a tie); explicit labels remain unchanged. ACN and MeCN group together by default; use `{"solvent grouping": "label"}` to separate labels.

```python
choices = e.filter(data, ["solvent", "scan window", "RE"])
all_choices = e.filter(data, "available", {"print": False})
acn = e.filter(data, {"SOLVENT": "ACN"})
gc = e.filter(data, {"WE": "GC"}, {"display columns": ["WE", "RE", "CE"]})
window = e.filter(data, {"scan window": [-1.0, 0.5]})
endpoint = e.filter(data, {"scan window": -1.0})
# Subfolders are relative to the selected import root; both slash styles work.
series = e.filter(data, {"subfolder": r"series\A"})
```

Automatic numeric matching uses three significant figures for scan windows and five for other numeric fields. This only changes matching keys, never the measured potentials. Failed filters show up to 12 candidate values per field; use `{"candidate limit": None}` to print all values. Inspection via `e.filter(data)` or a list of field names always returns all values in a dictionary, regardless of the print limit; dictionary criteria return selected objects. Use `e.filter(data, ["scan rate"], {"print": False})` in place of the removed standalone `get_available_filter_values()` helper.

## Run Peak Analysis

```python
peak = obj.peak_potential({
    "plot": False,
    "print": False,
})

current = obj.peak_current({
    "plot": False,
    "print": False,
    "tangent range": "auto",
})

width = obj.peak_width({
    "plot": False,
    "print": False,
    "tangent range": "auto",
})

print(peak)
print(current)
print(width)
```

## Export Data

```python
e.save_data(data, {
    "folder path": "outputs",
    "file name": "processed_beta_export",
})
```

For a round-trip-friendly workbook with a `manifest` sheet and class-specific data sheets, use:

```python
e.save_data(data, {
    "folder path": "outputs",
    "file name": "processed_beta_export",
    "format": "xlsx",
    "data columns": "all",
})
```

Referenced CVs include both the stored potential axis and the active referenced potential axis by default. Pass an exact list such as `{"data columns": ["Potential vs Fc/Fc+", "Current"]}` when you want only specific exported data columns.

For beta, verify the exported table and figure manually before using them in a report or manuscript.

## Continue With The Numbered Notebooks

After the core CV, plotting, DPV, and chrono notebooks:

1. `07_reversibility_analysis.ipynb` uses bundled real scan-rate data for
   electrochemical and chemical reversibility.
2. `08_surface_confined_cv.ipynb` imports a checked-in Excel series and
   demonstrates surface-confined reversibility, coverage, and loading.
3. `09_advanced_analysis.ipynb` covers normalization, catalytic analysis, and
   general fitting.
4. `10_simulation_intro.ipynb`, `11_group_fitting.ipynb`, and
   `12_equilibria_fitting.ipynb` form the optional simulation/fitting sequence.

The surface workbook is generated separately and stored under
`examples/data/surface_coverage_cv/`; notebook 08 therefore runs without an
optional simulation backend.
