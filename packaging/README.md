# eCAT Workbench Standalone Packaging

This folder contains the rough standalone-app packaging path for eCAT
Workbench. It is intentionally separate from the normal development workflow.

The app code still lives in `apps/workbench` and the scientific package still
lives in `src/ecat`. The standalone build just bundles Python, eCAT, the Dash
app, pywebview, assets, and public example data into a desktop artifact.

## Development Build

From the repository root, install the app dependencies and PyInstaller in the
environment you want to freeze:

```bash
python3 -m pip install -e ".[app]"
python3 -m pip install pyinstaller
```

Then build the current platform:

```bash
python3 packaging/build_standalone.py
```

Outputs are written under `dist/standalone/`.

For release signing on macOS, prefer a non-cloud-synced output folder. Cloud
storage providers can attach FileProvider or Finder metadata to app bundles,
which can make strict `codesign` verification fail even when the same bundle is
valid from a normal local folder:

```bash
python3 packaging/build_standalone.py \
  --dist-dir /private/tmp/ecat-standalone-dist \
  --work-dir /private/tmp/ecat-standalone-build
```

On macOS, the rough output is:

```text
dist/standalone/eCAT Workbench.app
```

On Windows, the rough output is usually:

```text
dist\standalone\eCAT Workbench\eCAT Workbench.exe
```

Build each platform on that platform. A macOS build should be produced on macOS,
and a Windows build should be produced on Windows.

## Security Warnings

Unsigned builds are useful for internal testing, but they will trigger platform
trust warnings when downloaded by someone else.

For macOS distribution, use an Apple Developer account, a Developer ID
Application certificate, hardened runtime signing, Apple notarization, and
stapling. See `packaging/sign_macos.sh`.

For Windows distribution, use an Authenticode code-signing certificate and
timestamp the executable or installer. See `packaging/sign_windows.ps1`.
Unsigned or newly signed Windows apps can still show Microsoft SmartScreen
warnings until the publisher/file reputation is established.

There is no legitimate project flag that makes these warnings disappear for
other users. The trustworthy path is signing, notarization on macOS, timestamped
signing on Windows, and release artifacts distributed from a consistent source.

## Release Policy

- Keep standalone binaries out of Git history.
- Upload `.dmg`, `.zip`, or installer files as GitHub Release assets.
- Keep this folder focused on reproducible packaging scripts and release notes.
- Do not enable edited Python execution in distributed remote builds.
- Read the application version from `src/ecat/_version.py`; do not hard-code a
  separate standalone version. CI verifies the source macOS shortcut and
  PyInstaller bundle metadata against the package version.
