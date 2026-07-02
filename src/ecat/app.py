"""Notebook-friendly launcher for the local eCAT app."""

from __future__ import annotations

import os
import socket
import threading
import webbrowser


def _find_available_port(host: str, preferred_port: int, max_tries: int = 50) -> int:
    """Return preferred_port when free, otherwise the next free local port."""
    for port in range(int(preferred_port), int(preferred_port) + int(max_tries)):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind((host, port))
            except OSError:
                continue
            return port
    raise RuntimeError(
        f"Could not find an available port from {preferred_port} to {preferred_port + max_tries - 1}."
    )


def _create_app():
    try:
        from ecat_app.app import create_app
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "The eCAT app dependencies are missing. Reinstall or upgrade eCAT with "
            "`python -m pip install --upgrade \"git+https://github.com/ljelissiry/eCAT.git@v0.1.0b3\"`."
        ) from exc
    return create_app()


def _run_native_app(app, host: str, port: int, *, title: str, width: int, height: int) -> str:
    try:
        from ecat_app.app import _run_window_app
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "The eCAT app window dependencies are missing. Reinstall or upgrade eCAT, "
            "or call e.open_app(browser=True) to use the browser fallback."
        ) from exc
    return _run_window_app(app, host, port, title=title, width=width, height=height)


def _display_notebook_link(url: str, inline: bool, height: int) -> None:
    try:
        from IPython.display import HTML, IFrame, display
    except Exception:
        return

    if inline:
        display(IFrame(src=url, width="100%", height=int(height)))
        return

    display(HTML(f'<a href="{url}" target="_blank">Open eCAT App in Browser</a>'))


def open_app(
    host: str = "127.0.0.1",
    port: int = 0,
    *,
    mode: str = "local",
    browser: bool = False,
    open_browser: bool = True,
    inline: bool = False,
    height: int = 900,
    width: int = 1400,
    title: str = "eCAT Workbench",
    allow_code_execution: bool = False,
    quiet: bool = False,
) -> str:
    """Start the local eCAT app and return its URL.

    This helper is designed for notebooks:

    Examples
    --------
    >>> import ecat as e
    >>> url = e.open_app()

    By default this opens the native eCAT app window. Pass ``browser=True`` to
    open the app in a browser tab instead, or ``inline=True`` to display the
    browser-mode app in a notebook iframe when the environment supports it.
    """

    browser_mode = bool(browser or inline)
    if browser_mode:
        preferred_port = 8050 if int(port) == 0 else int(port)
        selected_port = _find_available_port(host, preferred_port)
        url = f"http://{host}:{selected_port}"
    else:
        selected_port = 0 if int(port) == 0 else _find_available_port(host, int(port))
        url = None

    os.environ["ECAT_APP_MODE"] = str(mode)
    if allow_code_execution and str(mode).strip().lower() != "remote":
        os.environ["ECAT_APP_ALLOW_CODE_EXECUTION"] = "1"
    else:
        os.environ.pop("ECAT_APP_ALLOW_CODE_EXECUTION", None)
        os.environ.pop("ECAT_BROWSER_ALLOW_CODE_EXECUTION", None)

    app = _create_app()

    if not browser_mode:
        url = _run_native_app(
            app,
            host,
            selected_port,
            title=title,
            width=width,
            height=height,
        )
        if not quiet:
            print(f"eCAT App: {url}")
        return url

    thread = threading.Thread(
        target=app.run,
        kwargs={
            "host": host,
            "port": selected_port,
            "debug": False,
            "use_reloader": False,
        },
        daemon=True,
        name=f"ecat-app-browser-{selected_port}",
    )
    thread.start()

    if open_browser:
        webbrowser.open_new_tab(url)

    _display_notebook_link(url, inline=inline, height=height)

    if not quiet:
        print(f"eCAT App Browser Mode: {url}")

    return url


__all__ = ["open_app"]
