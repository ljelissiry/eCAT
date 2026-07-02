"""Dash app helpers for eCAT.

This package is a client of the public ``ecat`` API. It intentionally does not
extend or rename notebook-facing eCAT functions.
"""

from .workflow import AppWorkflow

__all__ = ["AppWorkflow"]
