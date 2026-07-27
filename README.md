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
python -m pip install --upgrade "git+https://github.com/ljelissiry/eCAT.git@v0.1.0b4"
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

The beta version is `0.1.0b4`.

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

## Beta Scope

See [docs/beta_scope.md](docs/beta_scope.md) for the supported file/technique matrix, known limitations, and recommended beta-user guidance.

In short:

- Recommended beta path: CH, BASI, and EC-Lab CV text exports.
- Limited path: CH CA, CH CP, EC-Lab CA/CP/GCPL text exports, and NOVA ASCII CV text exports.
- Fallback path: generic numeric/header text files with parser warnings available through `obj.parse_result.warnings`.
- Unsupported for beta: binary files and untested vendor formats.

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
