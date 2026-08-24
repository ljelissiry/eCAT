import numpy as np
import pandas as pd
import pytest
from pathlib import Path


def _surface_cv_series(ecat_module, blank_echem_factory, *, coverage=3e-10, area=0.07):
    rates = [0.05, 0.1, 0.2, 0.5]
    temperature = 298.15
    sigma = 4 * ecat_module.R * temperature / (
        ecat_module.F * np.sqrt(2 * np.pi)
    )
    values = []
    for rate in rates:
        forward = np.linspace(-0.3, 0.3, 601)
        reverse = np.linspace(0.3, -0.3, 601)[1:]
        potential = np.concatenate([forward, reverse])
        peak_current = (
            ecat_module.F**2 * area * coverage * rate
            / (4 * ecat_module.R * temperature)
        )
        forward_current = peak_current * np.exp(-0.5 * ((forward - 0.02) / sigma) ** 2)
        reverse_current = -peak_current * np.exp(-0.5 * ((reverse + 0.02) / sigma) ** 2)
        current = np.concatenate([forward_current, reverse_current])
        obj = blank_echem_factory(ecat_module.cv)
        obj.manual_init(
            f"{rate:g}Vs_surface_A",
            __import__("pandas").DataFrame(
                {
                    ("raw", "Potential (V)"): potential,
                    ("raw", "Current (A)"): current,
                }
            ),
            options={},
        )
        obj.scan_rate = rate
        obj.temperature = temperature
        obj.electrode_area = area
        obj.compounds = ["SurfaceA"]
        obj.concentrations = [1.0]
        obj.segments = 2
        values.append(obj)
    return values


def test_surface_coverage_public_api_and_option_defaults(ecat_module):
    assert callable(ecat_module.surface_coverage_analysis)
    options = ecat_module.SurfaceCoverageAnalysisOptions.from_options({})
    assert options.integration_range == "auto"
    assert options.agreement_tolerance == pytest.approx(0.25)
    assert options.num_electrons == pytest.approx(1)


def test_surface_coverage_slope_recovers_loading_without_area(ecat_module):
    rates = np.array([0.05, 0.1, 0.2, 0.5])
    loading = 2.5e-10
    temperature = 298.15
    slope = loading * ecat_module.F**2 / (4 * ecat_module.R * temperature)
    currents = slope * rates

    result = ecat_module._surface_coverage_from_slope(
        rates,
        currents,
        num_electrons=1,
        temperature=temperature,
        electrode_area=None,
    )

    assert result["loading / mol"] == pytest.approx(loading, rel=1e-10)
    assert np.isnan(result["coverage / mol cm^-2"])
    assert result["r2"] == pytest.approx(1.0)


def test_surface_coverage_slope_recovers_areal_coverage(ecat_module):
    rates = np.array([0.05, 0.1, 0.2, 0.5])
    area = 0.07
    coverage = 3e-10
    temperature = 298.15
    slope = (
        coverage
        * area
        * ecat_module.F**2
        / (4 * ecat_module.R * temperature)
    )

    result = ecat_module._surface_coverage_from_slope(
        rates,
        slope * rates,
        num_electrons=1,
        temperature=temperature,
        electrode_area=area,
    )

    assert result["coverage / mol cm^-2"] == pytest.approx(coverage, rel=1e-10)
    assert result["loading / mol"] == pytest.approx(coverage * area, rel=1e-10)


def test_surface_charge_uses_potential_and_scan_rate(ecat_module):
    potential = np.linspace(-0.2, 0.2, 2001)
    scan_rate = 0.1
    charge = 4.0e-6
    sigma = 0.03
    normalized = np.exp(-0.5 * (potential / sigma) ** 2)
    current = charge * scan_rate * normalized / np.trapezoid(normalized, potential)

    result = ecat_module._surface_charge_from_potential(
        potential,
        current,
        scan_rate=scan_rate,
        num_electrons=1,
        electrode_area=0.1,
    )

    assert result["charge / C"] == pytest.approx(charge, rel=1e-4)
    assert result["loading / mol"] == pytest.approx(charge / ecat_module.F, rel=1e-4)
    assert result["coverage / mol cm^-2"] == pytest.approx(
        charge / (ecat_module.F * 0.1), rel=1e-4
    )


def test_sevcik_rejects_removed_scan_dependence(ecat_module):
    with pytest.raises(ecat_module.OptionError, match="Unknown option 'scan dependence'"):
        ecat_module.SevcikAnalysisOptions.from_options({"scan dependence": 1})


def test_surface_coverage_analysis_recovers_slope_and_charge_loading(
    ecat_module,
    blank_echem_factory,
):
    coverage = 3e-10
    area = 0.07
    cvs = _surface_cv_series(
        ecat_module,
        blank_echem_factory,
        coverage=coverage,
        area=area,
    )

    result = ecat_module.surface_coverage_analysis(
        cvs,
        {
            "segments": [1, 2],
            "guess potential": [0.02, -0.02],
            "peak kind": "infer",
            "tangent range": [0.15, 0.25],
            "integration range": [-0.18, 0.18],
            "electrode area": area,
            "plot": False,
            "print": False,
        },
    )

    assert isinstance(result, ecat_module.ScatterFitResult)
    assert set(result.table["segment"]) == {1, 2}
    assert result.summary["coverage slope / mol cm^-2"] == pytest.approx(
        coverage, rel=0.03
    )
    assert result.summary["coverage charge / mol cm^-2"] == pytest.approx(
        coverage, rel=0.03
    )
    assert result.summary["loading slope / mol"] == pytest.approx(
        coverage * area, rel=0.03
    )
    assert result.summary["coverage agreement"] == "agree"
    assert len(result.fits) == 2
    assert not result.fit_table.empty
    assert "charge integrations" in result.diagnostics


def test_surface_coverage_auto_integration_and_plot_all(
    ecat_module,
    blank_echem_factory,
):
    cvs = _surface_cv_series(ecat_module, blank_echem_factory)

    result = ecat_module.surface_coverage_analysis(
        cvs,
        {
            "segment": 1,
            "guess potential": 0.02,
            "peak kind": "max",
            "tangent range": [0.15, 0.25],
            "integration range": "auto",
            "plot": True,
            "plot all": True,
            "print": False,
        },
    )

    assert np.isfinite(result.table["Q / C"]).all()
    assert len(result.figures) >= 2
    assert any(axis.collections for figure in result.figures for axis in figure.axes)
    assert result.axes.get_ylabel() == r"$|i_p|$ (μA)"
    assert result.figures[1].axes[0].get_ylabel() == "Current (μA)"
    assert max(result.axes.collections[0].get_offsets()[:, 1]) > 0.1


def test_surface_coverage_print_formats_parameter_units(
    ecat_module,
    blank_echem_factory,
    capsys,
):
    cvs = _surface_cv_series(ecat_module, blank_echem_factory)

    ecat_module.surface_coverage_analysis(
        cvs,
        {
            "segment": 1,
            "guess potential": 0.02,
            "peak kind": "max",
            "tangent range": [0.15, 0.25],
            "integration range": [-0.18, 0.18],
            "plot": False,
            "print": True,
            "pretty print": False,
        },
    )

    output = capsys.readouterr().out
    assert "Surface Coverage Equations:" in output
    assert "Surface Coverage Parameters:" in output
    assert "Parameter" in output
    assert "Symbol" in output
    assert "Electrode Area" in output
    assert "cm^2" in output
    assert output.index("Surface Coverage Parameters:") < output.index(
        "Surface Coverage Equations:"
    )
    assert output.index("Surface Coverage Equations:") < output.index(
        "Surface Coverage Summary:"
    )


def test_single_branch_surface_fit_uses_vertical_metric_value_display(
    ecat_module,
    blank_echem_factory,
):
    cvs = _surface_cv_series(ecat_module, blank_echem_factory)
    result = ecat_module.surface_coverage_analysis(
        cvs,
        {
            "segment": 1,
            "guess potential": 0.02,
            "peak kind": "max",
            "tangent range": [0.15, 0.25],
            "integration range": [-0.18, 0.18],
            "plot": False,
            "print": False,
        },
    )

    display_table = ecat_module._surface_fit_display_table(
        result.fit_table,
        {"sig figs": 4},
    )

    assert list(display_table.columns) == ["Metric", "Value"]
    assert display_table["Metric"].tolist() == [
        "Branch",
        "Segment",
        "Slope",
        "Intercept",
        "R2",
        "Gamma slope",
        "Loading slope",
        "Fit points",
    ]
    assert "mol cm^-2" in display_table.loc[
        display_table["Metric"] == "Gamma slope", "Value"
    ].iloc[0]


def test_surface_coverage_display_only_adds_name_for_replicates(ecat_module):
    unique = pd.DataFrame(
        {
            "name": ["25mVs", "50mVs"],
            "scan rate / V s^-1": [0.025, 0.05],
            "segment": [1, 1],
            "ip / A": [1e-6, 2e-6],
        }
    )
    replicates = pd.concat(
        [unique.iloc[[0]], unique.iloc[[0]].assign(name="25mVs replicate")],
        ignore_index=True,
    )

    unique_display = ecat_module._surface_coverage_display_data_table(unique)
    replicate_display = ecat_module._surface_coverage_display_data_table(replicates)

    assert "Name" not in unique_display.columns
    assert replicate_display.columns[0] == "Name"


def test_bundled_surface_workbook_roundtrip_recovers_expected_coverage(ecat_module):
    workbook = (
        Path(__file__).resolve().parents[1]
        / "examples"
        / "data"
        / "surface_coverage_cv"
        / "surface_confined_cv_series.xlsx"
    )

    assert workbook.exists()
    cvs = ecat_module.get_data_from_excel(workbook, {"print": False})

    assert len(cvs) == 5
    assert [obj.scan_rate for obj in cvs] == pytest.approx(
        [0.025, 0.05, 0.1, 0.2, 0.5]
    )
    assert all(obj.electrode_area == pytest.approx(0.10) for obj in cvs)

    result = ecat_module.surface_coverage_analysis(
        cvs,
        {
            "segments": [1, 2],
            "guess potential": [-0.10, -0.10],
            "electrode area": 0.10,
            "integration range": "auto",
            "plot": False,
            "print": False,
        },
    )

    assert result.summary["coverage slope / mol cm^-2"] == pytest.approx(
        3e-10, rel=0.01
    )
    assert result.summary["coverage charge / mol cm^-2"] == pytest.approx(
        3e-10, rel=0.02
    )
    assert result.summary["loading slope / mol"] == pytest.approx(3e-11, rel=0.01)
