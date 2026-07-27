"""Default eCAT app data sources."""

from __future__ import annotations

import os
from pathlib import Path
import sys

from .workflow import AppWorkflow


EXAMPLE_FOLDERS = {
    "fe_phoh_cv": {
        "label": "Fe/PhOH CV",
        "relative_path": Path("examples") / "data" / "fe_phoh_cv",
    },
    "chrono_ca": {
        "label": "CA/CPE",
        "relative_path": Path("examples") / "data" / "chrono_ca",
    },
    "chrono_cp": {
        "label": "CP Cycling",
        "relative_path": Path("examples") / "data" / "chrono_cp",
    },
}


def repo_root_path(repo_root=None) -> Path:
    if repo_root is not None:
        return Path(repo_root)
    configured = os.environ.get("ECAT_APP_REPO_ROOT")
    if configured:
        return Path(configured).expanduser()
    bundle_root = getattr(sys, "_MEIPASS", None)
    if bundle_root and (Path(bundle_root) / "examples" / "data").exists():
        return Path(bundle_root)
    return Path(__file__).resolve().parents[4]


def default_fe_phoh_path(repo_root=None) -> Path:
    return example_folder_path("fe_phoh_cv", repo_root)


def example_folder_options() -> list[dict[str, str]]:
    return [
        {"label": config["label"], "value": key}
        for key, config in EXAMPLE_FOLDERS.items()
    ]


def example_folder_path(key, repo_root=None) -> Path | None:
    config = EXAMPLE_FOLDERS.get(key)
    if config is None:
        return None
    return repo_root_path(repo_root) / config["relative_path"]


def default_workflow(repo_root=None) -> AppWorkflow:
    path = default_fe_phoh_path(repo_root)
    return AppWorkflow(
        source_kind="local_path",
        source_path=str(path),
        recursive=True,
        import_options={"sort keys": ["subfolder", "timestamp"]},
    )
