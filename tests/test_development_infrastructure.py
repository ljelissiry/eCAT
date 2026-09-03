import re


def _pyproject_list(text, key):
    pattern = rf"(?m)^{re.escape(key)}\s*=\s*\[(.*?)\]"
    match = re.search(pattern, text, flags=re.S)
    assert match is not None
    return set(re.findall(r'"([^"]+)"', match.group(1)))


def test_development_extra_includes_test_build_and_lint_tools(repo_root):
    pyproject = (repo_root / "pyproject.toml").read_text()

    development_dependencies = _pyproject_list(pyproject, "dev")

    assert {"build", "nbclient", "nbformat", "pytest", "pytest-cov"}.issubset(
        development_dependencies
    )
    assert any(dependency.startswith("ruff") for dependency in development_dependencies)


def test_ci_has_blocking_and_advisory_ruff_checks(repo_root):
    workflow = (repo_root / ".github" / "workflows" / "tests.yml").read_text()
    advisory_config = (repo_root / "ruff-advisory.toml").read_text()

    assert "name: Lint / correctness and advisory report" in workflow
    assert "python -m ruff check" in workflow
    assert "tests packaging scripts" not in workflow
    assert "src apps/workbench/src tests packaging" in workflow
    assert "--config ruff-advisory.toml" in workflow
    assert "--statistics --exit-zero" in workflow
    assert '"BLE001"' in advisory_config
    assert '"I001"' in advisory_config
    assert '"S110"' in advisory_config


def test_ci_has_native_windows_core_and_installed_wheel_app_checks(repo_root):
    workflow = (repo_root / ".github" / "workflows" / "tests.yml").read_text()

    assert workflow.count("runs-on: windows-latest") >= 2
    assert "name: Windows core / Python 3.13" in workflow
    assert "name: Windows wheel + app / Python 3.13" in workflow
    assert "python -m build" in workflow
    assert "ecat-app.exe" in workflow
    assert "test_client().get" in workflow
