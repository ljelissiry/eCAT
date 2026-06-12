# eCAT Browser App

Dash-based browser interface for eCAT. This app is intentionally separate from
the core `ecat` package and calls the public notebook-facing API.

## Install

From the repository root:

```bash
pip install -e ".[app]"
```

## Run

Development path:

```bash
python apps/browser/app.py
```

Installed command:

```bash
ecat-browser
```

The app runs locally by default at `http://127.0.0.1:8050`.

## Code Execution

Generated Python can always be previewed and downloaded. Running edited Python
inside the app is disabled by default and is intended only for local trusted
sessions:

```bash
ecat-browser --allow-code-execution
```

or:

```bash
ECAT_BROWSER_ALLOW_CODE_EXECUTION=1 ecat-browser
```

Do not enable edited-code execution for internet-deployed or shared instances.
