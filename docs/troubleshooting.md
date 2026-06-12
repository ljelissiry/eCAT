# eCAT Beta Troubleshooting

## No Files Load

- Confirm the files are text exports, usually `.txt`.
- Check `docs/beta_scope.md` to see whether the format is in beta scope.
- Try loading one file directly with `e.echem.from_file("path/to/file.txt", {})`.
- Binary files are not supported for beta; export or convert to text first.

## Units Look Wrong

- Inspect `obj.units` immediately after loading.
- For EC-Lab CV files, tested mA current headers are converted to A internally.
- For current density, confirm `electrode area` is nonzero.
- For normalized current, confirm the chosen `ip0`, concentration, diffusion coefficient, and area are the intended values.

## Reference Shift Looks Wrong

- Inspect `obj.reference_mode`, `obj.reference_shift`, `obj.reference_label`, and `obj.reference_source_file`.
- Use `"reference mode": "none"` to compare raw potentials.
- Use `"x axis": "Potential"` to request raw potential explicitly.
- Use the shifted default axis only after confirming the reference source.

## No Peak Is Found

- Plot the trace first and confirm the expected segment and direction.
- Provide a `guess potential` near the expected peak.
- Adjust `peak prominence` for small or noisy peaks.
- For `peak_current()`, inspect or constrain `tangent range` when auto selection is not scientifically reasonable.

## Normalization Is Confusing

- Use `e.normalize(...)` when creating normalized CV copies.
- Use `cv.normalize(...)` when you intentionally want to mutate the CV object.
- Check whether the object now uses dimensionless axes by default with `obj.x().name` and `obj.y().name`.
- Override axes explicitly with `"x axis": "Potential"` or `"y axis": "Current"` when needed.

## Generic Text Loads But Metadata Is Missing

- Generic fallback loading is intentionally limited.
- Rename files with clear gas, solvent, concentration, and scan-rate tokens where possible.
- Prefer CH, BASI, or EC-Lab text exports for beta feedback.
