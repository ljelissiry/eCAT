"""Dash app factory and CLI for eCAT browser."""

import argparse
import os

from .callbacks import register_callbacks
from .callbacks import handle_default_load
from .config import BrowserAppConfig
from .layout import create_layout


def create_app(config: BrowserAppConfig | None = None):
    try:
        from dash import Dash
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "The eCAT browser app requires Dash. Reinstall or upgrade eCAT with "
            '`python -m pip install --upgrade "git+https://github.com/ljelissiry/eCAT.git@v0.1.0b2"`.'
        ) from exc

    config = config or BrowserAppConfig.from_env()
    app = Dash(__name__, title="eCAT Browser")
    try:
        initial_state = handle_default_load()
    except Exception:
        initial_state = None
    app.layout = create_layout(config, initial_state=initial_state)
    register_callbacks(app)
    return app


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run the local eCAT browser app.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8050, type=int)
    parser.add_argument("--mode", choices=["local", "remote"], default=os.environ.get("ECAT_BROWSER_MODE", "local"))
    parser.add_argument(
        "--allow-code-execution",
        action="store_true",
        help="Allow edited Python execution. Use only for local trusted sessions.",
    )
    args = parser.parse_args(argv)

    if args.allow_code_execution:
        os.environ["ECAT_BROWSER_ALLOW_CODE_EXECUTION"] = "1"
    os.environ["ECAT_BROWSER_MODE"] = args.mode

    config = BrowserAppConfig.from_env()
    if config.mode == "remote":
        os.environ.pop("ECAT_BROWSER_ALLOW_CODE_EXECUTION", None)

    app = create_app(config)
    app.run(host=args.host, port=args.port, debug=False)
