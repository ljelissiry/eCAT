"""Runtime configuration for the eCAT browser app."""

from __future__ import annotations

from dataclasses import dataclass
import os


@dataclass(frozen=True)
class BrowserAppConfig:
    mode: str = "local"
    allow_code_execution: bool = False

    @classmethod
    def from_env(cls, env=None) -> "BrowserAppConfig":
        env = env or os.environ
        mode = str(env.get("ECAT_BROWSER_MODE", "local")).strip().lower()
        if mode not in {"local", "remote"}:
            mode = "local"
        allow_code = env.get("ECAT_BROWSER_ALLOW_CODE_EXECUTION") == "1"
        if mode == "remote":
            allow_code = False
        return cls(mode=mode, allow_code_execution=allow_code)

    @property
    def enable_folder_picker(self) -> bool:
        return self.mode == "local"
