import time
import builtins
import os
import plistlib
import re
import stat
import sys

import pytest


class _FakeDashApp:
    def __init__(self):
        self.run_calls = []

    def run(self, **kwargs):
        self.run_calls.append(kwargs)


def _wait_for_run(fake_app):
    deadline = time.monotonic() + 1.0
    while not fake_app.run_calls and time.monotonic() < deadline:
        time.sleep(0.01)


def test_open_app_starts_native_app_window_by_default(monkeypatch, capsys):
    import ecat
    import ecat.app as app_launcher

    fake_app = _FakeDashApp()
    window_calls = []

    monkeypatch.setattr(app_launcher, "_create_app", lambda: fake_app)
    monkeypatch.setattr(app_launcher, "_find_available_port", lambda host, port: port)
    monkeypatch.setattr(
        app_launcher,
        "_run_native_app",
        lambda app, host, port, **kwargs: window_calls.append((app, host, port, kwargs))
        or f"http://{host}:{port}/",
    )
    monkeypatch.setattr(app_launcher.webbrowser, "open_new_tab", lambda _url: pytest.fail("browser tab opened"))
    monkeypatch.setattr(app_launcher, "_display_notebook_link", lambda *_args, **_kwargs: None)

    url = ecat.open_app(port=8765, quiet=False)

    assert url == "http://127.0.0.1:8765/"
    assert window_calls[0][0] is fake_app
    assert window_calls[0][1:3] == ("127.0.0.1", 8765)
    assert window_calls[0][3]["title"] == "eCAT Workbench"
    assert fake_app.run_calls == []
    assert "eCAT App: http://127.0.0.1:8765/" in capsys.readouterr().out


def test_open_app_browser_mode_starts_local_server_and_opens_tab(monkeypatch, capsys):
    import ecat
    import ecat.app as app_launcher

    fake_app = _FakeDashApp()
    opened_urls = []
    displayed = []

    monkeypatch.setattr(app_launcher, "_create_app", lambda: fake_app)
    monkeypatch.setattr(app_launcher, "_find_available_port", lambda host, port: port)
    monkeypatch.setattr(app_launcher.webbrowser, "open_new_tab", lambda url: opened_urls.append(url))
    monkeypatch.setattr(
        app_launcher,
        "_display_notebook_link",
        lambda url, inline, height: displayed.append((url, inline, height)),
    )

    url = ecat.open_app(port=8765, browser=True, open_browser=True, inline=True, height=500)

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
    assert "eCAT App Browser Mode: http://127.0.0.1:8765" in capsys.readouterr().out


def test_open_app_uses_next_available_port_in_browser_mode(monkeypatch):
    import ecat.app as app_launcher

    fake_app = _FakeDashApp()
    monkeypatch.setattr(app_launcher, "_create_app", lambda: fake_app)
    monkeypatch.setattr(app_launcher, "_find_available_port", lambda host, port: port + 1)
    monkeypatch.setattr(app_launcher.webbrowser, "open_new_tab", lambda url: None)
    monkeypatch.setattr(app_launcher, "_display_notebook_link", lambda *args, **kwargs: None)

    url = app_launcher.open_app(port=8766, browser=True, open_browser=False, quiet=True)

    assert url == "http://127.0.0.1:8767"
    _wait_for_run(fake_app)
    assert fake_app.run_calls[0]["port"] == 8767


def test_open_app_sets_browser_environment(monkeypatch):
    import ecat.app as app_launcher

    monkeypatch.setattr(app_launcher, "_create_app", lambda: _FakeDashApp())
    monkeypatch.setattr(app_launcher, "_find_available_port", lambda host, port: port)
    monkeypatch.setattr(app_launcher.webbrowser, "open_new_tab", lambda url: None)
    monkeypatch.setattr(app_launcher, "_display_notebook_link", lambda *args, **kwargs: None)

    app_launcher.open_app(
        port=8768,
        mode="local",
        allow_code_execution=True,
        browser=True,
        open_browser=False,
        quiet=True,
    )

    assert app_launcher.os.environ["ECAT_APP_MODE"] == "local"
    assert app_launcher.os.environ["ECAT_APP_ALLOW_CODE_EXECUTION"] == "1"

    app_launcher.open_app(
        port=8769,
        mode="remote",
        allow_code_execution=True,
        browser=True,
        open_browser=False,
        quiet=True,
    )

    assert app_launcher.os.environ["ECAT_APP_MODE"] == "remote"
    assert "ECAT_APP_ALLOW_CODE_EXECUTION" not in app_launcher.os.environ
    assert "ECAT_BROWSER_ALLOW_CODE_EXECUTION" not in app_launcher.os.environ


def _pyproject_list(text, key):
    pattern = rf"(?m)^{re.escape(key)}\s*=\s*\[(.*?)\]"
    match = re.search(pattern, text, flags=re.S)
    assert match is not None
    return set(re.findall(r'"([^"]+)"', match.group(1)))


def test_app_dependencies_are_optional_extra(repo_root):
    pyproject = (repo_root / "pyproject.toml").read_text()

    base_dependencies = _pyproject_list(pyproject, "dependencies")
    app_dependencies = _pyproject_list(pyproject, "app")

    assert "dash" not in base_dependencies
    assert "dash-ag-grid" not in base_dependencies
    assert "pywebview" not in base_dependencies
    assert {"dash", "dash-ag-grid", "pywebview"}.issubset(app_dependencies)


def test_simulation_dependency_extra_remains_electrokitty(repo_root):
    pyproject = (repo_root / "pyproject.toml").read_text()

    assert _pyproject_list(pyproject, "simulation") == {"electrokitty"}


def test_app_launch_dependency_message_uses_app_extra(monkeypatch):
    import ecat.app as app_launcher

    original_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "ecat_app.app":
            raise ModuleNotFoundError("No module named 'ecat_app'")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(RuntimeError, match=r"ecat\[app\]") as excinfo:
        app_launcher._create_app()
    assert "ecat[simulation]" not in str(excinfo.value)


def test_workbench_create_app_reports_missing_app_extra(monkeypatch, repo_root):
    monkeypatch.syspath_prepend(str(repo_root / "apps" / "workbench" / "src"))
    import ecat_app.app as browser_app

    missing = {"dash", "dash_ag_grid"}
    original_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name in missing:
            raise ModuleNotFoundError(f"No module named {name!r}")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(RuntimeError, match=r"ecat\[app\]"):
        browser_app._require_app_dependencies()


def test_workbench_cli_exits_cleanly_for_missing_app_extra(monkeypatch, repo_root):
    monkeypatch.syspath_prepend(str(repo_root / "apps" / "workbench" / "src"))
    import ecat_app.app as browser_app

    monkeypatch.setattr(
        browser_app,
        "create_app",
        lambda _config=None: (_ for _ in ()).throw(RuntimeError("install ecat[app]")),
    )

    with pytest.raises(SystemExit, match=r"ecat\[app\]"):
        browser_app.main(["--browser", "--no-open"])


def test_workbench_cli_reports_package_version(capsys, repo_root):
    monkeypatch_path = str(repo_root / "apps" / "workbench" / "src")
    if monkeypatch_path not in sys.path:
        sys.path.insert(0, monkeypatch_path)
    import ecat
    import ecat_app.app as browser_app

    with pytest.raises(SystemExit) as excinfo:
        browser_app.main(["--version"])

    assert excinfo.value.code == 0
    assert capsys.readouterr().out.strip() == f"ecat {ecat.__version__}"


def test_source_checkout_launchers_prefer_repo_entrypoint_before_global_command(repo_root):
    launcher_dir = repo_root / "apps" / "workbench" / "launchers"
    mac_launcher = (launcher_dir / "ecat-workbench-launcher.sh").read_text()
    windows_launcher = (launcher_dir / "eCAT Workbench.cmd").read_text()

    installed_call = mac_launcher.index("run_installed_command\ninstalled_status=$?")
    assert mac_launcher.index('REPO_ROOT="$(find_repo_root || true)"') < installed_call
    assert mac_launcher.index('run_repo_python "$REPO_ROOT"') < installed_call
    assert windows_launcher.index(":find_repo") < windows_launcher.index("where ecat-app")
    assert windows_launcher.index('call :try_python "%REPO_ROOT%\\.venv\\Scripts\\python.exe"') < windows_launcher.index("where ecat-app")


def test_model_simulate_gate_reports_missing_simulation_extra(monkeypatch, repo_root):
    pytest.importorskip("dash", reason="app callback tests require ecat[app]")
    monkeypatch.syspath_prepend(str(repo_root / "apps" / "workbench" / "src"))
    import ecat_app.callbacks as callbacks

    monkeypatch.setattr(callbacks, "simulation_backend_available", lambda: False)

    disabled, message = callbacks.model_simulate_gate({"mechanism_valid": True})

    assert disabled is True
    assert "ecat[simulation]" in message


@pytest.mark.skipif(os.name != "posix", reason="macOS launcher permissions are POSIX-specific")
def test_macos_double_click_launchers_are_executable(repo_root):
    launcher_dir = repo_root / "apps" / "workbench" / "launchers"
    launcher_paths = [
        launcher_dir / "eCAT Workbench.app" / "Contents" / "MacOS" / "ecat-workbench",
        launcher_dir / "eCAT Workbench.app" / "Contents" / "Resources" / "ecat-workbench-launcher.sh",
        launcher_dir / "ecat-workbench-launcher.sh",
        launcher_dir / "eCAT Workbench.command",
    ]

    for launcher_path in launcher_paths:
        mode = launcher_path.stat().st_mode
        assert mode & stat.S_IXUSR, f"{launcher_path} must be executable for double-click launch"


def test_release_version_is_single_source_and_app_surfaces_match(repo_root):
    import ecat

    version = ecat.__version__
    beta_match = re.search(r"b(\d+)$", version)
    expected_bundle_version = beta_match.group(1) if beta_match else version

    pyproject_text = (repo_root / "pyproject.toml").read_text(encoding="utf-8")
    project_block = pyproject_text.split("[project]", 1)[1].split("\n[", 1)[0]
    assert not re.search(r"(?m)^version\s*=", project_block)
    assert re.search(r'(?m)^dynamic\s*=\s*\["version"\]$', project_block)
    assert 'version = { attr = "ecat._version.__version__" }' in pyproject_text

    plist_path = (
        repo_root
        / "apps"
        / "workbench"
        / "launchers"
        / "eCAT Workbench.app"
        / "Contents"
        / "Info.plist"
    )
    with plist_path.open("rb") as handle:
        plist = plistlib.load(handle)
    assert plist["CFBundleShortVersionString"] == version
    assert plist["CFBundleVersion"] == expected_bundle_version

    layout_text = (repo_root / "apps" / "workbench" / "src" / "ecat_app" / "layout.py").read_text(
        encoding="utf-8"
    )
    spec_text = (repo_root / "packaging" / "pyinstaller" / "ecat-workbench.spec").read_text(
        encoding="utf-8"
    )
    readme_text = (repo_root / "README.md").read_text(encoding="utf-8")

    assert 'html.Span(f"ecat {e.__version__}")' in layout_text
    assert "APP_VERSION =" in spec_text
    assert '"CFBundleShortVersionString": APP_VERSION' in spec_text
    assert '"CFBundleVersion": BUNDLE_VERSION' in spec_text
    assert f"@v{version}" in readme_text
    assert f"The beta version is `{version}`." in readme_text
