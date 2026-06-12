"""Development entrypoint for the eCAT Dash browser app."""

from pathlib import Path
import sys


APP_SRC = Path(__file__).resolve().parent / "src"
REPO_SRC = Path(__file__).resolve().parents[2] / "src"
for path in (APP_SRC, REPO_SRC):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from ecat_browser.app import main


if __name__ == "__main__":
    main()
