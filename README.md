# eCAT

eCAT, short for electroChemical Analysis Tools, is a Python package for loading, organizing, plotting, and analyzing electrochemical data from common lab workflows. The current beta focuses on trustworthy cyclic voltammetry workflows, with limited CA and CP support where the existing parsers are covered by tests.

## Install

For the beta release, install directly from GitHub. This requires Git to be
installed and available on your `PATH`; check with `git --version` if the
command fails.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install --upgrade "git+https://github.com/ljelissiry/eCAT.git@v0.1.0b5"
```

For local development from a source checkout, install from the repository root:

```bash
python -m pip install -e .
```

Then verify the install:

```bash
python -c "import ecat as e; print(e.__version__)"
pytest -q
```

The beta version is `0.1.0b5`.

## eCAT App

The eCAT app uses optional GUI/web dependencies. Install them with the app extra:

```bash
python -m pip install "ecat[app]"
```

For a local source checkout, use:

```bash
python -m pip install -e ".[app]"
```

Then run:

```bash
ecat-app
```

The installed app includes the public Fe/PhOH CV, CA/CPE, and CP example
folders shown in its example selector. Check the installed package version with
`ecat-app --version`.

From a notebook, launch the native app window with:

```python
import ecat as e

e.open_app()
```

If a native window is not convenient, run the same local app in browser mode:

```bash
ecat-app --browser
```

or from Python:

```python
e.open_app(browser=True)
```

To embed browser mode in a notebook cell when supported:

```python
e.open_app(inline=True)
```

Browser mode runs locally at `http://127.0.0.1:8050` by default and automatically uses the next available port if `8050` is busy.

## Simulation

CV simulation and fitting use the optional ElectroKitty backend:

```bash
python -m pip install "ecat[simulation]"
```

For a local source checkout, use:

```bash
python -m pip install -e ".[simulation]"
```

Without this extra, importing eCAT still works; simulation calls and the app's Model tab will show an install note.

Custom simulations use eCAT mechanism strings. Conventional coefficients and
repeated species are equivalent (`C:A+2B=C` and `C:A+B+B=C`). eCAT preserves
the entered equation for display and compiles a private ElectroKitty-compatible
form before calling the backend.

ElectroKitty is developed by Ožbej Vodeb and is licensed under the BSD 3-Clause License. eCAT uses ElectroKitty as an optional simulation backend; eCAT itself remains MIT licensed.

## Quickstart

```python
import ecat as e

data = e.get_data({
    "folder path": "path/to/exported/txt/files",
    "recursive search": True,
    "print": False,
    "reference mode": "none",
})

cv = data[0]
ax = cv.plot({"legend": False, "title": True})
peak = cv.peak_potential({"plot": False, "print": False})
print(peak)
```

Other single-CV metric helpers include `cv.peak_current()` and
`cv.peak_width()` when you need current or tangent-corrected full-width values.
`cv.peak_info()` and `cv.wave_info()` also report tangent-corrected full width
at half peak current; `wave_info()` identifies the cathodic and anodic segment
numbers and reports $i_{p,\mathrm{c}}$, $i_{p,\mathrm{a}}$, and
$|i_{p,\mathrm{a}}/i_{p,\mathrm{c}}|$ as evidence for chemical reversibility.

For multiple CVs:

```python
grouped = e.sort_and_group(
    data,
    sort_keys=["gas", "scan rate"],
    group_keys="gas",
    options={"print": False},
)

e.multiplot(grouped[0], {"legend": "auto", "title": False})
```

For one chemical condition measured across scan rates, use the cautious
series-level reversibility assessment:

```python
result = e.reversibility_analysis(
    scan_rate_series,
    {
        "phase": "bulk",
        "guess potential": -1.0,
        "num electrons": 1,
        "D": 1e-5,
    },
)
```

Surface-confined loading and coverage use a separate physical workflow:

```python
coverage = e.surface_coverage_analysis(
    scan_rate_series,
    {"segments": [1, 2], "guess potential": [-0.1, -0.1]},
)
```

The numbered quickstarts keep these physical models separate:

- [`07_reversibility_analysis.ipynb`](notebooks/07_reversibility_analysis.ipynb)
  uses a bundled real Fe/Fc scan-rate series to compare reversible and
  quasi-reversible waves.
- [`08_surface_confined_cv.ipynb`](notebooks/08_surface_confined_cv.ipynb)
  imports a pre-generated eCAT Excel workbook and recovers surface coverage
  plus total electroactive loading.
- [`09_advanced_analysis.ipynb`](notebooks/09_advanced_analysis.ipynb) covers
  normalization, catalytic analysis, and general fitting before the simulation
  sequence begins in notebook 10.

See [the API reference](docs/api_reference.md#reversibility-decision-tree)
for the exact bulk/surface decision tree, kinetic eligibility ranges, and
chemical-reversibility labels.

## Beta Scope

See [docs/beta_scope.md](docs/beta_scope.md) for the supported file/technique matrix, known limitations, and recommended beta-user guidance.

In short:

- Recommended beta path: CH `.txt`, BASI `.txt`/old BASI-Epsilon `.dat`, plus EC-Lab ASCII `.mpt` or compatible `.txt` exports.
- Limited path: CH CA, CH CP, EC-Lab CA/CP/GCPL text exports, and NOVA ASCII CV text exports.
- Fallback path: generic numeric/header text files with parser warnings available through `obj.parse_result.warnings`.
- Unsupported for beta: binary files, including BioLogic `.mpr`, and untested vendor formats. eCAT rejects `.mpr` before text parsing and recommends exporting EC-Lab ASCII `.mpt` or converting externally.

## Reporting Issues

Report beta bugs with the [eCAT Beta Bug / Feedback Report Google Form](https://docs.google.com/forms/d/e/1FAIpQLSe5rFOeuQ_qoh5NpULyKKsGGnMWOvqXH011f3dyz3X8YR603g/viewform). Include the smallest file example possible, the code you ran, the expected behavior, the actual behavior, and any traceback or screenshot.

## License And Third-Party Notices

eCAT is MIT licensed. See [LICENSE](LICENSE) for the eCAT license and
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for notices covering optional
and direct third-party dependencies.

## Development

Run tests with:

```bash
pytest -q
```

The test suite uses Matplotlib's Agg backend and checks objects, labels, numeric values, and exported files rather than pixel-perfect images.
