# -*- mode: python ; coding: utf-8 -*-

from pathlib import Path
import os
import re
import sys

from PyInstaller.utils.hooks import collect_submodules


APP_NAME = os.environ.get("ECAT_PYINSTALLER_NAME", "eCAT Workbench")
ROOT = Path(SPECPATH).resolve().parents[1]
# Source roots: apps/workbench/src and src.
APP_SRC = ROOT / "apps" / "workbench" / "src"
CORE_SRC = ROOT / "src"
ENTRY = ROOT / "packaging" / "pyinstaller" / "ecat_workbench_frozen.py"
ICON = ROOT / "apps" / "workbench" / "launchers" / "eCAT Workbench.app" / "Contents" / "Resources" / "ecat-logo.icns"
VERSION_TEXT = (CORE_SRC / "ecat" / "_version.py").read_text(encoding="utf-8")
APP_VERSION = re.search(r'^__version__\s*=\s*["\']([^"\']+)["\']', VERSION_TEXT, re.MULTILINE).group(1)
beta_match = re.search(r"b(\d+)$", APP_VERSION)
BUNDLE_VERSION = beta_match.group(1) if beta_match else APP_VERSION


def data_files():
    datas = []
    assets = APP_SRC / "ecat_app" / "assets"
    for path in assets.glob("*"):
        if path.is_file() and path.name != ".DS_Store":
            datas.append((str(path), "ecat_app/assets"))

    defaults = CORE_SRC / "ecat" / "defaults.toml"
    datas.append((str(defaults), "ecat"))

    examples = ROOT / "examples" / "data"
    if examples.exists():
        for path in examples.rglob("*"):
            if path.is_file() and path.name != ".DS_Store":
                relative_parent = path.relative_to(examples).parent
                datas.append((str(path), str(Path("examples") / "data" / relative_parent)))
    return datas


hiddenimports = []
for package_name in ("dash", "dash_ag_grid", "webview", "flask", "werkzeug"):
    hiddenimports.extend(collect_submodules(package_name))


a = Analysis(
    [str(ENTRY)],
    pathex=[str(APP_SRC), str(CORE_SRC)],
    binaries=[],
    datas=data_files(),
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["pytest"],
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name=APP_NAME,
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=str(ICON) if ICON.exists() else None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name=APP_NAME,
)

if sys.platform == "darwin":
    app = BUNDLE(
        coll,
        name=f"{APP_NAME}.app",
        icon=str(ICON) if ICON.exists() else None,
        bundle_identifier="org.ecat.workbench",
        info_plist={
            "CFBundleName": APP_NAME,
            "CFBundleDisplayName": APP_NAME,
            "CFBundleShortVersionString": APP_VERSION,
            "CFBundleVersion": BUNDLE_VERSION,
            "NSHighResolutionCapable": "True",
        },
    )
