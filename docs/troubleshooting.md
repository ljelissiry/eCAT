# eCAT Beta Troubleshooting

## No Files Load

- Confirm the files are text exports, usually `.txt`.
- Check `docs/beta_scope.md` to see whether the format is in beta scope.
- Try loading one file directly with `e.echem.from_file("path/to/file.txt", {})`.
- Binary files are not supported for beta; export or convert to text first.

## Units Look Wrong

- Inspect `obj.units` immediately after loading.
- Imported potential, current, and time columns are stored in SI units (`V`, `A`, `s`) whenever the parser can identify the source units.
- Use plot/display options such as `"y unit": "uA"` to change displayed current units; import-time `"convert current"` is not supported.
- For current-density plots, pass `electrode area` during import or set `obj.electrode_area`, then request `"y axis": "current density"`.
- For normalized current, confirm the chosen `ip0`, concentration, diffusion coefficient, and area are the intended values.

## Reference Shift Looks Wrong

- Inspect `obj.reference_mode`, `obj.reference_shift`, `obj.reference_label`, and `obj.reference_source_file`.
- Use `"reference mode": "none"` to compare raw potentials.
- Use `"x axis": "Potential"` to request raw potential explicitly.
- Use the shifted default axis only after confirming the reference source.

## No Peak Is Found

- Plot the trace first and confirm the expected segment and direction.
- Provide a `guess potential` near the expected peak.
- Guessed peaks use a local automatic `peak prominence`; set an explicit lower `peak prominence` for small shoulders or a higher value for noisy traces.
- Use `noise window = None` to disable Savitzky-Golay smoothing during peak detection.
- For `peak_current()`, inspect or constrain `tangent range` when auto selection is not scientifically reasonable.

## Normalization Is Confusing

- Use `e.normalize(...)` when creating normalized CV copies.
- Use `cv.normalize(...)` when you intentionally want to mutate the CV object.
- Check whether the object now uses dimensionless axes by default with `obj.x().name` and `obj.y().name`.
- Override axes explicitly with `"x axis": "Potential"` or `"y axis": "Current"` when needed.

## Generic Text Loads But Metadata Is Missing

- Generic fallback loading is intentionally limited.
- Inspect `obj.parse_result.warnings` for missing or inferred metadata.
- Inspect `obj.parse_result.raw_metadata` to see the original header lines, column names, units, and parser notes that eCAT preserved.
- Rename files with clear gas, solvent, concentration, and scan-rate tokens where possible.
- Prefer CH, BASI, EC-Lab text, or representative NOVA ASCII CV exports for beta feedback.

## Parser Warnings Appear After Loading

- Parser warnings are nonfatal diagnostics. They usually mean eCAT loaded the numeric table but inferred or could not find metadata such as scan rate, timestamp, step structure, or technique.
- Use `e.parse_file(path)` when you want to inspect the parser contract before object promotion.
- Use `obj.parse_result.warnings` and `obj.parse_result.raw_metadata` when an object loaded successfully but `obj.info()` looks incomplete.
- If a CH-style file warns that the header scan rate and filename scan rate disagree, eCAT keeps importing and uses the scan rate embedded in the file header. Treat this as a likely export/renaming issue and inspect the raw file before analysis.
- If a NOVA ASCII or generic text file loads as a CV, confirm the potential/current units and scan rate before analysis.
- If a generic text file contains time, potential, and current columns but no technique marker, eCAT keeps it generic rather than guessing CA, CP, or CV.
- IviumSoft text exports are not yet validated for beta; send representative files if you need that importer.
