<h1><img src="apps/workbench/src/ecat_app/assets/ecat-logo-app-icon.svg" alt="eCAT logo" width="52" height="52" align="absmiddle"> eCAT: electroChemical Analysis Tools</h1>

eCAT is a Python package and app for loading, organizing, plotting, and analyzing electrochemical data. It supports tested workflows for cyclic voltammetry (CV), chronoamperometry and controlled-potential electrolysis (CA/CPE), chronopotentiometry (CP), and differential-pulse voltammetry (DPV).

The current beta emphasizes readable notebook workflows, explicit metadata and units, consistent plotting, and diagnostics that show how each electrochemical result was obtained.

## Install

The package is distributed as `ecat-electrochemistry`, but imported as `ecat`:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install --pre ecat-electrochemistry
```

For a reproducible install of this exact beta, pin the release:

```bash
python -m pip install "ecat-electrochemistry==0.1.0b8"
```

On Windows PowerShell, activate the environment with:

```powershell
.venv\Scripts\Activate.ps1
```

To install the current tagged beta directly from GitHub instead, first make sure [Git](https://git-scm.com/downloads) is installed:

```bash
python -m pip install --upgrade "git+https://github.com/ljelissiry/eCAT.git@v0.1.0b8"
```

Confirm that the notebook kernel sees the same environment:

```bash
python -c "import ecat as e; print(e.__version__)"
```

The beta version is `0.1.0b8`.

## Quickstart

Import eCAT and load a folder of instrument text exports:

```python
import ecat as e

data = e.get_data({
    "folder path": "path/to/exported/files",
    "recursive search": True,
    "reference mode": "none",
})
```

`get_data()` recognizes tested CH, BASI, and EC-Lab ASCII workflows and preserves parser warnings and metadata on each object. Use `e.describe_options()` to discover public functions, or pass a function name to inspect its options:

```python
e.describe_options("get_data")
e.show(data)
```

Plot one CV and run a peak measurement:

```python
cv = data[0]
cv.plot({"title": True})

peak = cv.peak_current({
    "segment": 1,
    "guess potential": -1.0,
    "plot all": True,
})
```

Filter, sort, and group object collections with their parsed metadata:

```python
co2 = e.filter(data, {"gas": "CO2"}, {"print": False})
co2 = e.sort(co2, ["scan rate"], {"print": False})
by_scan_rate = e.group(co2, ["scan rate"], {"print": False})
```

Use `multiplot()` for an overlay. Labels, colors, units, and colorbars can be inferred from the objects or set explicitly:

```python
e.multiplot(co2, {
    "title": "CO$_2$ Series",
    "legend": "auto",
    "y unit": "uA",
})
```

## Quickstart Notebooks

The numbered notebooks are runnable guides that build from installation through simulation and fitting:

- [00 - Install And Setup](https://github.com/ljelissiry/eCAT/blob/master/notebooks/00_install_and_setup.ipynb): install checks, packaged example data, and help discovery.
- [01 - Default Template](https://github.com/ljelissiry/eCAT/blob/master/notebooks/01_default_template.ipynb): a reusable analysis notebook structure with import, selection, plotting, and export cells.
- [02 - Basics: Load, Plot, Filter, And Group](https://github.com/ljelissiry/eCAT/blob/master/notebooks/02_basics_load_plot_filter.ipynb): the core object workflow and Excel round trip.
- [03 - CV Analysis](https://github.com/ljelissiry/eCAT/blob/master/notebooks/03_cv_analysis.ipynb): reference correction, peak selection, peak current, and wave metrics.
- [04 - Plotting And Animation](https://github.com/ljelissiry/eCAT/blob/master/notebooks/04_plotting_and_animation.ipynb): plot styling, overlays, colorbars, directional arrows, and object-first animation.
- [05 - DPV Basics](https://github.com/ljelissiry/eCAT/blob/master/notebooks/05_dpv_basics.ipynb): loading, inspecting, plotting, and analyzing pulse-voltammetry data.
- [06 - CA And CP Basics](https://github.com/ljelissiry/eCAT/blob/master/notebooks/06_ca_cp_basics.ipynb): chrono plots, charge, current/rate helpers, cycle analysis, and mixed-class Excel export.
- [07 - Reversibility Analysis](https://github.com/ljelissiry/eCAT/blob/master/notebooks/07_reversibility_analysis.ipynb): cautious scan-rate-series assessment using real CV data.
- [08 - Surface-Confined CV](https://github.com/ljelissiry/eCAT/blob/master/notebooks/08_surface_confined_cv.ipynb): surface coverage and electroactive loading from an imported example series.
- [09 - Advanced Analysis](https://github.com/ljelissiry/eCAT/blob/master/notebooks/09_advanced_analysis.ipynb): normalization, kinetic analyses, scatter fitting, and model fitting.
- [10 - Simulation Intro](https://github.com/ljelissiry/eCAT/blob/master/notebooks/10_simulation_intro.ipynb): CV programs, real-CV inputs, mechanisms, simulation, and local/multistart fitting.
- [11 - Group Fitting](https://github.com/ljelissiry/eCAT/blob/master/notebooks/11_group_fitting.ipynb): shared and per-CV parameter fitting across related experiments.
- [12 - Equilibria And Incubation Fitting](https://github.com/ljelissiry/eCAT/blob/master/notebooks/12_equilibria_fitting.ipynb): pre-equilibria, incubation, and coupled simulation workflows.
- [99 - Troubleshooting](https://github.com/ljelissiry/eCAT/blob/master/notebooks/99_troubleshooting.ipynb): common import, metadata, reference, plotting, and environment problems.

## eCAT App

Install the optional app dependencies and launch the local desktop/web app:

```bash
python -m pip install --pre "ecat-electrochemistry[app]"
ecat-app
```

The app can also be launched from Python:

```python
import ecat as e

e.open_app()               # native window
e.open_app(browser=True)   # local browser
e.open_app(inline=True)    # notebook, when supported
```

Browser mode runs locally at `http://127.0.0.1:8050` by default and uses the next available port if needed. The installed app includes public CV, CA/CPE, and CP examples.

## Simulation

CV simulation and fitting use [ElectroKitty](https://github.com/RedrumKid/ElectroKitty), developed by Ožbej Vodeb, as an optional backend:

```bash
python -m pip install --pre "ecat-electrochemistry[simulation]"
```

The simulation extra uses the standard PyPI ElectroKitty package (minimum 1.0.11.6); a fresh installation selects the latest compatible release. To update an existing installation, run `python -m pip install --upgrade --upgrade-strategy eager "ecat-electrochemistry[simulation]"`. Experimental local-equilibrium and patched-backend integrations are not included in this release.

Without this extra, the rest of eCAT still imports and runs. Simulation calls and the app Model tab provide an installation message when the backend is unavailable.

```python
from ecat import simulation as sim

program = sim.cv_program(
    0.0,
    E_low=-1.5,
    E_high=0.5,
    scan_rate=0.1,
    segments=2,
)
```

See notebooks 10-12 for complete simulation and fitting examples. eCAT is MIT licensed; ElectroKitty is distributed under the BSD 3-Clause License.

## Beta Scope

The recommended beta path covers CH `.txt`, BASI `.txt` and older BASI-Epsilon `.dat`, plus EC-Lab ASCII `.mpt` or compatible `.txt` exports. Generic numeric text files have a tested fallback parser, while binary files such as BioLogic `.mpr` are rejected with conversion guidance.

See the [beta scope](https://github.com/ljelissiry/eCAT/blob/master/docs/beta_scope.md), [API reference](https://github.com/ljelissiry/eCAT/blob/master/docs/api_reference.md), and [troubleshooting guide](https://github.com/ljelissiry/eCAT/blob/master/docs/troubleshooting.md) for current support boundaries and function details.

## Feedback

Report beta bugs with the [eCAT Beta Bug / Feedback Report](https://docs.google.com/forms/d/e/1FAIpQLSe5rFOeuQ_qoh5NpULyKKsGGnMWOvqXH011f3dyz3X8YR603g/viewform). Include the smallest example file possible, the code you ran, expected and actual behavior, and any traceback or screenshot.

## License And Development

eCAT is MIT licensed. See the [license](https://github.com/ljelissiry/eCAT/blob/master/LICENSE) and [third-party notices](https://github.com/ljelissiry/eCAT/blob/master/THIRD_PARTY_NOTICES.md).

For local development:

```bash
git clone https://github.com/ljelissiry/eCAT.git
cd eCAT
python -m pip install -e ".[dev]"
python -m ruff check src apps/workbench/src tests packaging
pytest -q
```

Release maintainers should follow the [release guide](https://github.com/ljelissiry/eCAT/blob/master/docs/releasing.md).

## User guide and analysis methods

- [User guide (PDF)](eCAT%20Guide.pdf) and [editable Word guide](eCAT%20Guide.docx)
- [API reference](docs/api_reference.md), [analysis equations and options](docs/analysis_equations.md), and [method references](docs/analysis_reading.md)
- [Guide maintenance](docs/guide-maintenance.md) and [development environments](docs/development-environments.md)
