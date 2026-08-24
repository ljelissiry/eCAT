"""Frozen eCAT Workbench entrypoint used by PyInstaller."""

from __future__ import annotations

import os
from pathlib import Path
import sys


def _bundle_root() -> Path:
    if getattr(sys, "frozen", False):
        return Path(getattr(sys, "_MEIPASS", Path(sys.executable).resolve().parent))
    return Path(__file__).resolve().parents[2]


os.environ.setdefault("ECAT_APP_MODE", "local")
os.environ.setdefault("ECAT_APP_REPO_ROOT", str(_bundle_root()))

from ecat_app.app import main


if __name__ == "__main__":
    main(["--mode", "local"])

