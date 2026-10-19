from pathlib import Path


def test_publish_workflow_builds_checks_and_uses_trusted_publishing(repo_root: Path):
    workflow_path = repo_root / ".github" / "workflows" / "publish.yml"
    assert workflow_path.is_file()

    workflow = workflow_path.read_text(encoding="utf-8")
    assert "workflow_dispatch:" in workflow
    assert "release:" in workflow
    assert "python -m build" in workflow
    assert "python -m twine check" in workflow
    assert "actions/upload-artifact@" in workflow
    assert "actions/download-artifact@" in workflow
    assert "pypa/gh-action-pypi-publish@release/v1" in workflow
    assert "repository-url: https://test.pypi.org/legacy/" in workflow
    assert "name: testpypi" in workflow
    assert "name: pypi" in workflow
    assert "id-token: write" in workflow
    assert "github.event_name == 'workflow_dispatch'" in workflow
    assert "github.event_name == 'release'" in workflow
    assert 'name = "ecat-electrochemistry"' in workflow
    assert 'expected_tag = f"v{version_match.group(1)}"' in workflow


def test_readme_simulation_quickstart_uses_public_cv_program_signature(repo_root: Path):
    readme = (repo_root / "README.md").read_text(encoding="utf-8")

    assert "program = sim.cv_program(" in readme
    assert "E_low=-1.5" in readme
    assert "E_high=0.5" in readme
    assert "sim.cv_program({" not in readme
