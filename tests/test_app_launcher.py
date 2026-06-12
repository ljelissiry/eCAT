import time


class _FakeDashApp:
    def __init__(self):
        self.run_calls = []

    def run(self, **kwargs):
        self.run_calls.append(kwargs)


def _wait_for_run(fake_app):
    deadline = time.monotonic() + 1.0
    while not fake_app.run_calls and time.monotonic() < deadline:
        time.sleep(0.01)


def test_open_app_starts_browser_app_on_local_url(monkeypatch, capsys):
    import ecat
    import ecat.app as app_launcher

    fake_app = _FakeDashApp()
    opened_urls = []
    displayed = []

    monkeypatch.setattr(app_launcher, "_create_browser_app", lambda: fake_app)
    monkeypatch.setattr(app_launcher, "_find_available_port", lambda host, port: port)
    monkeypatch.setattr(app_launcher.webbrowser, "open_new_tab", lambda url: opened_urls.append(url))
    monkeypatch.setattr(
        app_launcher,
        "_display_notebook_link",
        lambda url, inline, height: displayed.append((url, inline, height)),
    )

    url = ecat.open_app(port=8765, open_browser=True, inline=True, height=500)

    assert url == "http://127.0.0.1:8765"
    _wait_for_run(fake_app)
    assert opened_urls == [url]
    assert displayed == [(url, True, 500)]
    assert fake_app.run_calls == [
        {
            "host": "127.0.0.1",
            "port": 8765,
            "debug": False,
            "use_reloader": False,
        }
    ]
    assert "eCAT Browser: http://127.0.0.1:8765" in capsys.readouterr().out


def test_open_app_uses_next_available_port(monkeypatch):
    import ecat.app as app_launcher

    fake_app = _FakeDashApp()
    monkeypatch.setattr(app_launcher, "_create_browser_app", lambda: fake_app)
    monkeypatch.setattr(app_launcher, "_find_available_port", lambda host, port: port + 1)
    monkeypatch.setattr(app_launcher.webbrowser, "open_new_tab", lambda url: None)
    monkeypatch.setattr(app_launcher, "_display_notebook_link", lambda *args, **kwargs: None)

    url = app_launcher.open_app(port=8766, open_browser=False, quiet=True)

    assert url == "http://127.0.0.1:8767"
    _wait_for_run(fake_app)
    assert fake_app.run_calls[0]["port"] == 8767


def test_open_app_sets_browser_environment(monkeypatch):
    import ecat.app as app_launcher

    monkeypatch.setattr(app_launcher, "_create_browser_app", lambda: _FakeDashApp())
    monkeypatch.setattr(app_launcher, "_find_available_port", lambda host, port: port)
    monkeypatch.setattr(app_launcher.webbrowser, "open_new_tab", lambda url: None)
    monkeypatch.setattr(app_launcher, "_display_notebook_link", lambda *args, **kwargs: None)

    app_launcher.open_app(
        port=8768,
        mode="local",
        allow_code_execution=True,
        open_browser=False,
        quiet=True,
    )

    assert app_launcher.os.environ["ECAT_BROWSER_MODE"] == "local"
    assert app_launcher.os.environ["ECAT_BROWSER_ALLOW_CODE_EXECUTION"] == "1"

    app_launcher.open_app(
        port=8769,
        mode="remote",
        allow_code_execution=True,
        open_browser=False,
        quiet=True,
    )

    assert app_launcher.os.environ["ECAT_BROWSER_MODE"] == "remote"
    assert "ECAT_BROWSER_ALLOW_CODE_EXECUTION" not in app_launcher.os.environ


def test_dash_dependencies_are_installed_with_base_package(repo_root):
    pyproject = (repo_root / "pyproject.toml").read_text()

    dependencies_block = pyproject.split("dependencies = [", 1)[1].split("]", 1)[0]

    assert '"dash"' in dependencies_block
    assert '"dash-ag-grid"' in dependencies_block
