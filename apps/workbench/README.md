# eCAT App

Dash-based local interface for eCAT. The app is intentionally separate from
the core `ecat` package internals and calls the public notebook-facing API.

## Install

From the repository root:

```bash
pip install -e .
```

## Run

Development path:

```bash
python apps/workbench/app.py
```

Installed command:

```bash
ecat-app
```

By default, `ecat-app` opens the native eCAT Workbench window using pywebview.
The app still runs locally on your computer.

If the native window is not convenient, use browser mode:

```bash
ecat-app --browser
```

Browser mode runs locally by default at `http://127.0.0.1:8050` and opens that
address in your default browser. To print the URL without opening a browser:

```bash
ecat-app --browser --no-open
```

or set:

```bash
ECAT_APP_OPEN=0 ecat-app --browser
```

## Double-Click Launchers

For a double-click launch on macOS, open one of:

- `apps/workbench/launchers/eCAT Workbench.app`
- `apps/workbench/launchers/eCAT Workbench.command`

For a double-click launch on Windows, open:

- `apps/workbench/launchers/eCAT Workbench.cmd`

All launchers start the same local Dash app in a pywebview window. They prefer
an installed `ecat-app` command and fall back to this repository's development
entrypoint when launched from a source checkout.

If the window opens and immediately closes, check the launcher log:

```bash
tail -80 "${TMPDIR:-/tmp}/ecat-workbench-launch.log"
```

The most common cause is a Python environment missing the installed eCAT
dependencies. Reinstall from the repository root:

```bash
python3 -m pip install -e .
```

The double-click launchers show a popup with the same install instructions when
they cannot find a usable Python environment. When launched from this checkout,
the popup includes the exact `cd` command for the repository folder before the
install command.

If you use a specific Python environment, point the launcher at it:

```bash
export ECAT_PYTHON=/path/to/python
open "apps/workbench/launchers/eCAT Workbench.app"
```

## Code Execution

Generated Python can always be previewed and downloaded. Running edited Python
inside the app is disabled by default and is intended only for local trusted
sessions:

```bash
ecat-app --allow-code-execution
```

or in browser mode:

```bash
ecat-app --browser --allow-code-execution
```

or:

```bash
ECAT_APP_ALLOW_CODE_EXECUTION=1 ecat-app
```

Do not enable edited-code execution for internet-deployed or shared instances.
