# eCAT App

Dash-based local interface for eCAT. The app is intentionally separate from
the core `ecat` package internals and calls the public notebook-facing API.

## Install

From the repository root:

```bash
python -m pip install -e ".[app]"
```

For an installed beta package, use
`python -m pip install "ecat-electrochemistry[app]"`.

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
python3 -m pip install -e ".[app]"
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

## Maintainer Update Checklist

When the package is updated, the source launchers should not need logic changes
for ordinary Python edits. They are designed to prefer this checkout's
development entrypoint before any stale globally installed `ecat-app` command.
The package version is assigned only in `src/ecat/_version.py`; package metadata
and the in-app About panel read it dynamically. The macOS shortcut's
`Info.plist`, standalone bundle metadata, and public install text are checked
against that source by CI.

Do refresh the local editable install after changes to console commands,
package metadata, optional app dependencies, or import paths:

```bash
python3 -m pip install -e ".[app]"
ecat-app --help
```

Rebuild and ad-hoc sign the macOS `.app` only when the C wrapper, `Info.plist`,
icon, bundled launcher script, or app bundle resources change. For public
downloads, each released app bundle still needs normal platform signing:
Developer ID signing/notarization/stapling on macOS and Authenticode signing on
Windows. Signing is per released build, not a one-time setup.

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
