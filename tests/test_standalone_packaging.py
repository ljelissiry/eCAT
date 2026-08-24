from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
PACKAGING = ROOT / "packaging"


def test_standalone_packaging_files_document_supported_build_flow():
    expected = [
        PACKAGING / "README.md",
        PACKAGING / "build_standalone.py",
        PACKAGING / "pyinstaller" / "ecat_workbench_frozen.py",
        PACKAGING / "pyinstaller" / "ecat-workbench.spec",
        PACKAGING / "sign_macos.sh",
        PACKAGING / "sign_windows.ps1",
    ]

    missing = [path.relative_to(ROOT).as_posix() for path in expected if not path.exists()]

    assert missing == []


def test_standalone_entrypoint_uses_existing_app_cli():
    entrypoint = (PACKAGING / "pyinstaller" / "ecat_workbench_frozen.py").read_text()
    spec = (PACKAGING / "pyinstaller" / "ecat-workbench.spec").read_text()
    builder = (PACKAGING / "build_standalone.py").read_text()

    assert "from ecat_app.app import main" in entrypoint
    assert "main([\"--mode\", \"local\"])" in entrypoint
    assert "apps/workbench/src" in spec
    assert "src" in spec
    assert "ecat_workbench_frozen.py" in builder
    assert "PYINSTALLER_CONFIG_DIR" in builder


def test_security_warning_guidance_uses_signing_not_fake_bypass():
    readme = (PACKAGING / "README.md").read_text()
    mac_sign = (PACKAGING / "sign_macos.sh").read_text()
    windows_sign = (PACKAGING / "sign_windows.ps1").read_text()

    assert "Apple Developer" in readme
    assert "notarization" in readme
    assert "SmartScreen" in readme
    assert "xattr -cr" in mac_sign
    assert "com.apple.FinderInfo" in mac_sign
    assert "codesign" in mac_sign
    assert "notarytool" in mac_sign
    assert "signtool" in windows_sign
    assert "suppress" not in readme.lower()


def test_example_data_root_can_resolve_from_frozen_bundle(monkeypatch, tmp_path):
    app_src = ROOT / "apps" / "workbench" / "src"
    if str(app_src) not in sys.path:
        sys.path.insert(0, str(app_src))
    from ecat_app import defaults

    bundle_root = tmp_path / "bundle"
    (bundle_root / "examples" / "data").mkdir(parents=True)

    monkeypatch.delenv("ECAT_APP_REPO_ROOT", raising=False)
    monkeypatch.setattr(sys, "_MEIPASS", str(bundle_root), raising=False)

    assert defaults.repo_root_path() == bundle_root
