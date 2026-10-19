# eCAT Beta Troubleshooting

## No Files Load

- Confirm the files are supported text exports: `.txt` generally, old BASI-Epsilon `.dat`, or EC-Lab ASCII `.mpt`.
- Check `docs/beta_scope.md` to see whether the format is in beta scope.
- Try loading one file directly with `e.echem.from_file("path/to/file.txt", {})`, `e.echem.from_file("path/to/file.dat", {})`, or `e.echem.from_file("path/to/file.mpt", {})`.
- BioLogic `.mpr` is a binary format, not another spelling of `.mpt`. eCAT raises `UnsupportedFileFormatError` before text parsing; export EC-Lab ASCII `.mpt`, convert externally, or provide a custom reader for direct loading.

## Units Look Wrong

- Inspect `obj.units` immediately after loading.
- Imported potential, current, and time columns are stored in SI units (`V`, `A`, `s`) whenever the parser can identify the source units.
- Use plot/display options such as `"y unit": "uA"` to change displayed current units; import-time `"convert current"` is not supported.
- For current-density plots, pass `electrode area` during import or set `obj.electrode_area`, then request `"y axis": "current density"`.
- For normalized current, confirm the chosen `ip0`, concentration, diffusion coefficient, and area are the intended values.

## Reference Shift Looks Wrong

- Inspect `obj.reference_mode`, `obj.reference_shift`, `obj.reference_label`, `obj.reference_source_file`, and `obj.reference_pair_details`.
- Use `"reference mode": "none"` to compare raw potentials.
- Use `"x axis": "Potential"` to request raw potential explicitly.
- Use the shifted default axis only after confirming the reference source.
- Automatic pairing accepts only extrema from adjacent, opposite-direction CV segments; it never combines two features from one sweep or nonadjacent sweeps.
- If `reference guess="auto"` reports ambiguity, plot the reference scan and supply a numeric `reference guess` near the intended couple.
- Set `troubleshoot=True` to print segment spans, candidate extrema, rejection reasons, and the selected pair and to display the segment-colored reference diagnostic.

## No Peak Is Found

- Plot the trace first and confirm the expected segment and direction.
- Provide a `guess potential` near the expected peak.
- Guessed peaks use a local automatic `peak prominence`; set an explicit lower `peak prominence` for small shoulders or a higher value for noisy traces.
- Use `noise window = None` to disable Savitzky-Golay smoothing during peak detection.
- For `peak_current()` or `peak_width()`, inspect or constrain `tangent range` when auto selection is not scientifically reasonable.
- For `peak_width()`, choose a `level` between 0 and 1; the default `0.5` reports full width at half peak current.

## Normalization Is Confusing

- Use `e.normalize(...)` when creating normalized CV copies.
- Use `cv.normalize(...)` when you intentionally want to mutate the CV object.
- Check whether the object now uses dimensionless axes by default with `obj.x().name` and `obj.y().name`.
- Override axes explicitly with `"x axis": "Potential"` or `"y axis": "Current"` when needed.

## Spectrum Analysis Fails Or Looks Misleading

- Use `obj.spectrum({"method": "welch"})` for averaged power spectral density
  or `obj.spectrum({"method": "fft"})` for one-sided amplitude and phase. Both
  include deterministic electrochemical signal variation as well as noise.
- Inspect `result.summary["time source"]`, the reported sample rate, and the
  Nyquist frequency before interpreting peaks. CA, CP, and generic objects need
  recorded Time; CV can reconstruct time per segment from raw Potential and
  scan rate, and DPV can reconstruct it from pulse period.
- If sampling-interval deviation exceeds the default 1% tolerance, repair the
  timestamps or set `"resample": True` to interpolate onto the median-interval
  grid. Inspect the `plot all=True` time-domain diagnostic after resampling.
- If the trace is too short or Welch reports weak averaging, select a wider
  `"time range"` or use a smaller `"points per window"` value.
- Keep the upper `"frequency range"` at or below the reported Nyquist
  frequency. Frequencies above Nyquist cannot be resolved and may appear as
  lower-frequency aliases.
- DC is excluded by default. Set `"include dc": True` to retain it, but do not
  combine a retained zero-frequency bin with a logarithmic x-axis.
- A finite record can leak power into neighboring frequency bins. Detrending
  and a Hann or other window reduce leakage but do not prove that a feature is
  noise; compare the time-domain trace and vary defensible window settings.
- `"plot scale"` controls spectrum-axis spacing, not numeric multiplication.
  Ordinary `obj.plot()` scaling uses `"x scale"` / `"y scale"` to multiply
  displayed data. Use spectrum `"plot scale": "linear"`, `"log-log"`,
  `"semilogx"`, `"semilogy"`, or `"symlog"`, or direct `xscale` / `yscale`
  overrides.

## Generic Text Loads But Metadata Is Missing

- Generic fallback loading is intentionally limited.
- Inspect `obj.parse_result.warnings` for missing or inferred metadata.
- Inspect `obj.parse_result.raw_metadata` to see the original header lines, column names, units, and parser notes that eCAT preserved.
- Rename files with clear gas, solvent, concentration, and scan-rate tokens where possible.
- Prefer CH, BASI, EC-Lab text, or representative NOVA ASCII CV exports for beta feedback.

## Parser Warnings Appear After Loading

- Parser warnings are nonfatal diagnostics. They usually mean eCAT loaded the numeric table but inferred or could not find metadata such as scan rate, timestamp, step structure, or technique.
- Use `e.parse_file(path)` when you want to inspect the parser contract before object promotion.
- Use `obj.parse_result.warnings` and `obj.parse_result.raw_metadata` when an object loaded successfully but `obj.metadata()` looks incomplete.
- Header and filename scan rates are compared after conversion to V/s. Differences within 0.1% or `1e-6 V/s` are treated as acquisition/rounding noise. Larger disagreements warn, continue importing, and use the measured header value; inspect the filename and raw header before analysis.
- If a NOVA ASCII or generic text file loads as a CV, confirm the potential/current units and scan rate before analysis.
- If a generic text file contains time, potential, and current columns but no technique marker, eCAT keeps it generic rather than guessing CA, CP, or CV.
- IviumSoft text exports are not yet validated for beta; send representative files if you need that importer.
