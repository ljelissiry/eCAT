"""Local trusted execution for edited browser-app Python code."""

from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass
from io import StringIO
import os
import signal
from pathlib import Path


@dataclass
class ExecutionResult:
    executed: bool
    returncode: int | None
    stdout: str = ""
    stderr: str = ""


def code_execution_allowed(env=None) -> bool:
    env = env or os.environ
    return env.get("ECAT_BROWSER_ALLOW_CODE_EXECUTION") == "1"


class _Timeout:
    def __init__(self, seconds: int):
        self.seconds = seconds
        self.previous = None

    def __enter__(self):
        if hasattr(signal, "SIGALRM"):
            self.previous = signal.signal(signal.SIGALRM, self._raise)
            signal.alarm(int(self.seconds))
        return self

    def __exit__(self, exc_type, exc, tb):
        if hasattr(signal, "SIGALRM"):
            signal.alarm(0)
            if self.previous is not None:
                signal.signal(signal.SIGALRM, self.previous)
        return False

    @staticmethod
    def _raise(signum, frame):
        raise TimeoutError("Timed out while running edited Python code.")


def run_user_code(code: str, cwd=None, timeout_seconds: int = 10, output_limit: int = 20000) -> ExecutionResult:
    if not code_execution_allowed():
        return ExecutionResult(
            executed=False,
            returncode=None,
            stderr=(
                "Edited Python execution is disabled. Set "
                "ECAT_BROWSER_ALLOW_CODE_EXECUTION=1 only for local trusted use."
            ),
        )

    cwd = Path(cwd or Path.cwd())
    cwd.mkdir(parents=True, exist_ok=True)
    stdout = StringIO()
    stderr = StringIO()
    old_cwd = Path.cwd()
    namespace = {"__name__": "__main__"}

    try:
        os.chdir(cwd)
        with _Timeout(timeout_seconds), redirect_stdout(stdout), redirect_stderr(stderr):
            exec(compile(code, "<ecat-browser-edited-code>", "exec"), namespace)
        return ExecutionResult(True, 0, stdout.getvalue()[:output_limit], stderr.getvalue()[:output_limit])
    except Exception as exc:
        err = stderr.getvalue() + f"{type(exc).__name__}: {exc}\n"
        return ExecutionResult(True, 1, stdout.getvalue()[:output_limit], err[:output_limit])
    finally:
        os.chdir(old_cwd)
