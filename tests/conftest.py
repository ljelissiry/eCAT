import os
import sys
from pathlib import Path

import matplotlib
import pandas as pd
import pytest


os.environ.setdefault("MPLBACKEND", "Agg")
matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt


@pytest.fixture(scope="session")
def repo_root():
    return Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session", autouse=True)
def package_src_path(repo_root):
    src_path = str(repo_root / "src")
    if src_path not in sys.path:
        sys.path.insert(0, src_path)


@pytest.fixture(scope="session")
def fixtures_dir():
    return Path(__file__).resolve().parent / "fixtures"


@pytest.fixture(scope="session")
def ecat_module(repo_root):
    import ecat
    from ecat import (
        analysis_batch,
        analysis_cv,
        collection,
        export,
        io,
        metadata,
        objects,
        parsers,
        plotting,
        reference,
        utils,
    )

    internal_modules = (
        utils,
        metadata,
        parsers,
        objects,
        plotting,
        analysis_cv,
        analysis_batch,
        collection,
        reference,
        io,
        export,
    )

    class EcatTestProxy:
        __name__ = "ecat"

        @property
        def _modules(self):
            return internal_modules

        def __getattr__(self, name):
            if hasattr(ecat, name):
                return getattr(ecat, name)
            for module in self._modules:
                if hasattr(module, name):
                    return getattr(module, name)
            raise AttributeError(name)

        def __setattr__(self, name, value):
            if name in getattr(ecat, "__all__", ()) or name in vars(ecat):
                setattr(ecat, name, value)
            for module in self._modules:
                if hasattr(module, name):
                    setattr(module, name, value)

        def __delattr__(self, name):
            if name in vars(ecat):
                delattr(ecat, name)
            for module in self._modules:
                if name in vars(module):
                    delattr(module, name)

    return EcatTestProxy()


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


def _blank_echem_object(ecat_module, cls):
    obj = cls.__new__(cls)
    obj.filepath = None
    obj.options = {}
    obj.timestamp = None
    obj.creation_time = None
    obj.modification_time = None
    obj.name = "Unnamed EChem Object"
    obj.data = pd.DataFrame()
    obj.type = None
    obj.software = None
    obj.num_x_cols = 1
    obj.temperature = 298
    obj.electrode_area = 0
    obj.delta_x = None
    obj.units = {}
    obj.segments = 1
    obj.gas = None
    obj.solvent = None
    obj.reference_shift = None
    obj.reference_label = None
    obj.reference_mode = "none"
    obj.reference_source_file = None
    obj.reference_failure_message = None
    obj.folderpath = "."
    return obj


@pytest.fixture
def blank_echem_factory(ecat_module):
    def factory(cls=None):
        cls = ecat_module.echem if cls is None else cls
        return _blank_echem_object(ecat_module, cls)

    return factory


@pytest.fixture
def cv_factory(ecat_module):
    def factory(
        name="50mVs_sample_CO2_MeCN_10mM_Fc_run01",
        potential=None,
        current=None,
        options=None,
    ):
        if potential is None:
            potential = [
                -0.25, -0.2, -0.15, -0.1, -0.05, 0.0, 0.05, 0.1, 0.15, 0.2,
                0.25, 0.2, 0.15, 0.1, 0.05, 0.0, -0.05, -0.1, -0.15, -0.2, -0.25,
            ]
        if current is None:
            current = [
                -0.1e-6, -0.08e-6, -0.05e-6, 0.0, 0.3e-6, 0.8e-6, 1.8e-6, 4.0e-6,
                7.2e-6, 6.5e-6, 5.5e-6, 4.2e-6, 3.0e-6, 1.8e-6, 1.0e-6, 0.4e-6,
                0.1e-6, 0.0, -0.03e-6, -0.05e-6, -0.07e-6,
            ]

        data = pd.DataFrame(
            {
                ("raw", "Potential (V)"): potential,
                ("raw", "Current (A)"): current,
            }
        )
        obj = _blank_echem_object(ecat_module, ecat_module.cv)
        obj.manual_init(name, data, options=options or {})
        obj.folderpath = "."
        return obj

    return factory
