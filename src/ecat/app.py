"""Notebook-friendly launcher for the local eCAT browser app."""

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


def _create_browser_app():
    try:
        from ecat_browser.app import create_app
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "The eCAT browser app dependencies are missing. Reinstall or upgrade eCAT with "
            "`python -m pip install --upgrade \"git+https://github.com/ljelissiry/eCAT.git@v0.1.0b2\"`."
        ) from exc
    return create_app()


def _display_notebook_link(url: str, inline: bool, height: int) -> None:
    try:
        from IPython.display import HTML, IFrame, display
    except Exception:
        return

    if inline:
        display(IFrame(src=url, width="100%", height=int(height)))
        return

    display(HTML(f'<a href="{url}" target="_blank">Open eCAT Browser</a>'))


def open_app(
    host: str = "127.0.0.1",
    port: int = 8050,
    *,
    mode: str = "local",
    open_browser: bool = True,
    inline: bool = False,
    height: int = 900,
    allow_code_execution: bool = False,
    quiet: bool = False,
) -> str:
    """Start the local eCAT browser app and return its URL.

    This helper is designed for notebooks:

    Examples
    --------
    >>> import ecat as e
    >>> url = e.open_app()
    >>> url
    'http://127.0.0.1:8050'

    Pass ``inline=True`` to display the app in a notebook iframe when the
    notebook environment supports it.
    """

    selected_port = _find_available_port(host, port)
    url = f"http://{host}:{selected_port}"

    os.environ["ECAT_BROWSER_MODE"] = str(mode)
    if allow_code_execution and str(mode).strip().lower() != "remote":
        os.environ["ECAT_BROWSER_ALLOW_CODE_EXECUTION"] = "1"
    else:
        os.environ.pop("ECAT_BROWSER_ALLOW_CODE_EXECUTION", None)

    app = _create_browser_app()

    thread = threading.Thread(
        target=app.run,
        kwargs={
            "host": host,
            "port": selected_port,
            "debug": False,
            "use_reloader": False,
        },
        daemon=True,
        name=f"ecat-browser-{selected_port}",
    )
    thread.start()

    if open_browser:
        webbrowser.open_new_tab(url)

    _display_notebook_link(url, inline=inline, height=height)

    if not quiet:
        print(f"eCAT Browser: {url}")

    return url


__all__ = ["open_app"]
