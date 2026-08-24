"""Generate the imported surface-confined CV tutorial workbook.

The workbook is generated separately from the quickstart notebook so notebook 08
exercises the same Excel import path a user would use with experimental data.
"""

from pathlib import Path

import numpy as np
import pandas as pd

import ecat as e


SCAN_RATES = [0.025, 0.05, 0.1, 0.2, 0.5]
AREA_CM2 = 0.10
COVERAGE_MOL_CM2 = 3e-10
TEMPERATURE_K = 298.15
E0_V = -0.10
R = 8.31446261815324
F = 96485.33212


def build_surface_series():
    """Return deterministic reversible E* CVs as eCAT simulation objects."""
    coverage_mol_m2 = COVERAGE_MOL_CM2 * 1e4
    params = {
        "concentrations": {"surface": {"Ox": coverage_mol_m2, "Red": 0.0}},
        "kinetics": [{"E0": E0_V, "k0": "fast", "alpha": 0.5}],
        "cell": {"T": TEMPERATURE_K, "Ru": 0.0, "Cdl": 0.0, "A": 1e-5},
        "spatial": "fast",
    }
    mechanism = e.simulation.compile_mechanism("E*", params)
    cvs = []

    for scan_rate in SCAN_RATES:
        program = e.simulation.cv_program(
            Ei=0.25,
            E_low=-0.45,
            E_high=0.25,
            scan_rate=scan_rate,
            direction="cathodic",
            segments=2,
            points_per_segment=1000,
        )
        turning_index = int(np.argmin(program.E))
        peak_current = (
            F**2
            * AREA_CM2
            * COVERAGE_MOL_CM2
            * scan_rate
            / (4 * R * TEMPERATURE_K)
        )
        z = F * (program.E - E0_V) / (2 * R * TEMPERATURE_K)
        current = peak_current / np.cosh(z) ** 2
        current[: turning_index + 1] *= -1

        simulated = e.simulation.SimulatedCV(
            data=pd.DataFrame(
                {
                    "Potential": program.E,
                    "Current": current,
                    "Time": program.t,
                }
            ),
            params=params,
            mechanism=mechanism,
            input=program,
            backend_result={"source": "analytic reversible surface E* fixture"},
            summary={
                "backend": "ecat analytic surface fixture",
                "preset": "E*",
                "current_sign": 1,
            },
        )
        simulated.name = f"surface_Estar_{scan_rate:g}Vs"
        simulated.compounds = ["SurfaceOx"]
        simulated.concentrations = []
        simulated.solvent = "MeCN"
        simulated.gas = "Ar"
        simulated.electrode_area = AREA_CM2
        cvs.append(simulated)

    return cvs


def main():
    output_dir = Path(__file__).resolve().parent
    e.save_data(
        build_surface_series(),
        {
            "format": "xlsx",
            "folder path": str(output_dir),
            "file name": "surface_confined_cv_series",
            "metadata columns": "all",
            "data columns": ["Potential", "Current", "Time"],
            "share x axes": False,
            "print": False,
        },
    )


if __name__ == "__main__":
    main()
