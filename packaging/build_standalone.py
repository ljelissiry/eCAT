"""Build a rough standalone eCAT Workbench app with PyInstaller."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import platform
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / "packaging" / "pyinstaller" / "ecat-workbench.spec"
# The spec freezes packaging/pyinstaller/ecat_workbench_frozen.py.
DEFAULT_DIST = ROOT / "dist" / "standalone"
DEFAULT_WORK = ROOT / "build" / "standalone"


def _platform_output(dist_dir: Path, name: str) -> Path:
    if platform.system() == "Darwin":
        return dist_dir / f"{name}.app"
    if platform.system() == "Windows":
        return dist_dir / name / f"{name}.exe"
    return dist_dir / name / name


def build(argv: list[str] | None = None) -> Path:
    parser = argparse.ArgumentParser(
        description="Build a rough standalone eCAT Workbench app for this platform."
    )
    parser.add_argument("--name", default="eCAT Workbench")
    parser.add_argument("--dist-dir", default=str(DEFAULT_DIST))
    parser.add_argument("--work-dir", default=str(DEFAULT_WORK))
    parser.add_argument("--no-clean", action="store_true")
    args = parser.parse_args(argv)

    dist_dir = Path(args.dist_dir).expanduser().resolve()
    work_dir = Path(args.work_dir).expanduser().resolve()
    command = [
        sys.executable,
        "-m",
        "PyInstaller",
        "--noconfirm",
        "--distpath",
        str(dist_dir),
        "--workpath",
        str(work_dir),
        str(SPEC),
    ]
    if not args.no_clean:
        command.insert(3, "--clean")

    config_dir = work_dir / "pyinstaller-config"
    config_dir.mkdir(parents=True, exist_ok=True)
    env = {
        "ECAT_PYINSTALLER_NAME": args.name,
        "PYINSTALLER_CONFIG_DIR": str(config_dir),
        "MPLCONFIGDIR": str(work_dir / "matplotlib-config"),
    }
    completed = subprocess.run(
        command,
        cwd=ROOT,
        env={**os.environ, **env},
        check=False,
    )
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)

    output = _platform_output(dist_dir, args.name)
    print(f"Built eCAT Workbench standalone artifact: {output}")
    return output


if __name__ == "__main__":
    build()
