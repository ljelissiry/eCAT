# eCAT

eCAT, short for electroCatalysis Analysis Tools, is a Python package for loading, organizing, plotting, and analyzing electrochemical data from common lab workflows. The current lab beta focuses on trustworthy cyclic voltammetry workflows, with limited CA and CP support where the existing parsers are covered by tests.

## Install

For the lab beta, install from the repository root in a clean environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e .
```

Then verify the install:

```bash
python -c "import ecat as e; print(e.__version__)"
pytest -q
```

The beta version is `0.1.0b2`.

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
- Limited path: CH CA, CH CP, and EC-Lab GCPL/CP text exports.
- Fallback path: generic numeric/header text files.
- Unsupported for beta: binary files and untested vendor formats.

## Reporting Issues

Report beta bugs with the [eCAT Beta Bug / Feedback Report Google Form](https://docs.google.com/forms/d/e/1FAIpQLSe5rFOeuQ_qoh5NpULyKKsGGnMWOvqXH011f3dyz3X8YR603g/viewform). Include the smallest file example possible, the code you ran, the expected behavior, the actual behavior, and any traceback or screenshot.

## Development

Run tests with:

```bash
pytest -q
```

The test suite uses Matplotlib's Agg backend and checks objects, labels, numeric values, and exported files rather than pixel-perfect images.
