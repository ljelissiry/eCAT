"""Dash app factory and CLI for the eCAT app."""

import argparse
import base64
from datetime import datetime
import mimetypes
import os
from pathlib import Path
import re
import threading
from urllib.parse import unquote_to_bytes
import webbrowser

from ecat._version import __version__

from .config import AppConfig


_ECAT_APP_INSTALL_MESSAGE = (
    "The eCAT app requires optional app dependencies. Install them with "
    '`python -m pip install "ecat[app]"`. For a source checkout, use '
    '`python -m pip install -e ".[app]"`.'
)


def _require_app_dependencies():
    missing = []
    for module_name, display_name in (
        ("dash", "dash"),
        ("dash_ag_grid", "dash-ag-grid"),
    ):
        try:
            __import__(module_name)
        except ModuleNotFoundError:
            missing.append(display_name)
    if missing:
        raise RuntimeError(
            f"{_ECAT_APP_INSTALL_MESSAGE} Missing: {', '.join(missing)}."
        )


def _plot_download_dir(env=None) -> Path:
    env = env or os.environ
    configured = env.get("ECAT_APP_DOWNLOAD_DIR")
    if configured:
        return Path(configured).expanduser()
    return Path.home() / "Downloads"


def _safe_plot_filename(filename: str | None, extension: str) -> str:
    name = Path(filename or "").name
    if not name:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        name = f"ecat-plot-{stamp}.{extension}"
    name = re.sub(r"[^A-Za-z0-9._ -]+", "_", name).strip(" .")
    if not name:
        name = f"ecat-plot.{extension}"
    if "." not in Path(name).name:
        name = f"{name}.{extension}"
    return name


def _unique_plot_path(destination_dir: Path, filename: str) -> Path:
    candidate = destination_dir / filename
    if not candidate.exists():
        return candidate
    stem = candidate.stem
    suffix = candidate.suffix
    counter = 2
    while True:
        next_candidate = destination_dir / f"{stem}-{counter}{suffix}"
        if not next_candidate.exists():
            return next_candidate
        counter += 1


def _plot_extension_from_mime(mime_type: str) -> str:
    if mime_type == "image/svg+xml":
        return "svg"
    if mime_type == "text/html":
        return "html"
    if mime_type == "image/jpeg":
        return "jpg"
    guessed = mimetypes.guess_extension(mime_type or "")
    return (guessed or ".png").lstrip(".")


def _decode_plot_data_uri(src: str) -> tuple[bytes, str]:
    if not src or not str(src).startswith("data:") or "," not in str(src):
        raise ValueError("Expected a plot data URI.")
    header, payload = str(src).split(",", 1)
    metadata = header[5:]
    parts = metadata.split(";")
    mime_type = parts[0] or "application/octet-stream"
    if "base64" in parts[1:]:
        return base64.b64decode(payload), mime_type
    return unquote_to_bytes(payload), mime_type


def _save_plot_payload(src: str, filename: str | None = None, download_dir: str | Path | None = None) -> Path:
    content, mime_type = _decode_plot_data_uri(src)
    extension = _plot_extension_from_mime(mime_type)
    destination_dir = Path(download_dir).expanduser() if download_dir is not None else _plot_download_dir()
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination = _unique_plot_path(destination_dir, _safe_plot_filename(filename, extension))
    destination.write_bytes(content)
    return destination


def _register_plot_save_route(app, config: AppConfig):
    from flask import jsonify, request

    @app.server.post("/ecat-app/save-plot")
    def save_plot_route():
        if config.mode != "local":
            return jsonify({"ok": False, "message": "Local plot saving is disabled."}), 403
        payload = request.get_json(silent=True) or {}
        try:
            destination = _save_plot_payload(
                payload.get("src", ""),
                payload.get("filename"),
            )
        except Exception as exc:
            return jsonify({"ok": False, "message": str(exc)}), 400
        return jsonify(
            {
                "ok": True,
                "filename": destination.name,
                "path": str(destination),
                "message": "Saved to Downloads",
            }
        )


def create_app(config: AppConfig | None = None):
    _require_app_dependencies()

    from dash import Dash

    from .callbacks import handle_default_load, register_callbacks
    from .layout import create_layout

    config = config or AppConfig.from_env()
    app = Dash(__name__, title="eCAT Workbench")
    try:
        initial_state = handle_default_load()
    except Exception:
        initial_state = None
    app.layout = create_layout(config, initial_state=initial_state)
    _register_plot_save_route(app, config)
    register_callbacks(app)
    return app


def _env_flag(name: str, default: bool = True, *, legacy_name: str | None = None) -> bool:
    value = os.environ.get(name)
    if value is None and legacy_name:
        value = os.environ.get(legacy_name)
    if value is None:
        return default
    return str(value).strip().lower() not in {"0", "false", "no", "off"}


def _open_browser_later(url: str, enabled: bool = True, delay: float = 1.0):
    if not enabled:
        return None
    timer = threading.Timer(float(delay), webbrowser.open_new_tab, args=[url])
    timer.daemon = True
    timer.start()
    return timer


def _make_server(host: str, port: int, server):
    from werkzeug.serving import make_server

    return make_server(host, int(port), server)


def _load_webview():
    try:
        import webview
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "The native eCAT app window requires pywebview from the optional app "
            "dependencies. Install them with `python -m pip install \"ecat[app]\"`, "
            "or run `ecat-app --browser` to use the browser fallback."
        ) from exc
    return webview


def _run_window_app(
    app,
    host: str,
    port: int,
    *,
    title: str = "eCAT Workbench",
    width: int = 1400,
    height: int = 900,
) -> str:
    server = _make_server(host, port, app.server)
    selected_port = int(getattr(server, "server_port", port))
    url = f"http://{host}:{selected_port}/"
    thread = threading.Thread(
        target=server.serve_forever,
        daemon=True,
        name=f"ecat-window-{selected_port}",
    )
    thread.start()

    webview = _load_webview()
    try:
        webview.create_window(title, url, width=int(width), height=int(height))
        webview.start()
        return url
    finally:
        server.shutdown()
        thread.join(timeout=2)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run the local eCAT app.")
    parser.add_argument("--version", action="version", version=f"ecat {__version__}")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=None, type=int)
    parser.add_argument(
        "--mode",
        choices=["local", "remote"],
        default=os.environ.get("ECAT_APP_MODE", os.environ.get("ECAT_BROWSER_MODE", "local")),
    )
    parser.add_argument(
        "--browser",
        action="store_true",
        help="Open the app in the default web browser instead of the native app window.",
    )
    parser.add_argument(
        "--no-open",
        action="store_true",
        help="Do not open a browser tab when using --browser.",
    )
    parser.add_argument("--width", default=1400, type=int, help="Native app window width.")
    parser.add_argument("--height", default=900, type=int, help="Native app window height.")
    parser.add_argument("--title", default="eCAT Workbench", help="Native app window title.")
    parser.add_argument(
        "--allow-code-execution",
        action="store_true",
        help="Allow edited Python execution. Use only for local trusted sessions.",
    )
    args = parser.parse_args(argv)

    if args.allow_code_execution:
        os.environ["ECAT_APP_ALLOW_CODE_EXECUTION"] = "1"
    os.environ["ECAT_APP_MODE"] = args.mode

    config = AppConfig.from_env()
    if config.mode == "remote":
        os.environ.pop("ECAT_APP_ALLOW_CODE_EXECUTION", None)
        os.environ.pop("ECAT_BROWSER_ALLOW_CODE_EXECUTION", None)

    try:
        app = create_app(config)
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from None
    selected_port = args.port if args.port is not None else (8050 if args.browser else 0)
    if not args.browser:
        try:
            _run_window_app(app, args.host, selected_port, title=args.title, width=args.width, height=args.height)
        except RuntimeError as exc:
            raise SystemExit(str(exc)) from None
        return None
    url = f"http://{args.host}:{selected_port}/"
    _open_browser_later(
        url,
        enabled=(
            not args.no_open
            and _env_flag("ECAT_APP_OPEN", True, legacy_name="ECAT_BROWSER_OPEN")
        ),
    )
    app.run(host=args.host, port=selected_port, debug=False)
