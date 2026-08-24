# Surface-Confined CV Series

`surface_confined_cv_series.xlsx` is a deterministic tutorial fixture generated
separately from the quickstart notebooks. Notebook 08 imports the workbook with
`e.get_data_from_excel(...)`, just as it would import a user-created eCAT Excel
workbook.

The five CVs represent a reversible, one-electron surface-confined `E*` couple at
0.025, 0.05, 0.1, 0.2, and 0.5 V/s. The generation parameters are:

- formal potential: -0.10 V
- temperature: 298.15 K
- electrode area: 0.10 cm2
- surface coverage: 3e-10 mol/cm2
- total loading: 3e-11 mol

The generator uses eCAT's `cv_program`, `compile_mechanism("E*")`,
`SimulatedCV`, and `save_data` APIs. The reversible surface-wave current is
evaluated analytically because the core tutorial must not require an optional
simulation backend at runtime.

Regenerate from the repository root with:

```bash
python examples/data/surface_coverage_cv/generate_surface_workbook.py
```
