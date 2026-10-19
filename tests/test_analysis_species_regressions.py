import numpy as np
import pytest
import matplotlib.pyplot as plt


def _sevcik_cvs(ecat_module, cv_factory, monkeypatch, concentrations, rates):
    """Synthetic reversible currents with a known diffusion coefficient."""
    diffusion = 7.5e-6
    cvs = []
    for index, (concentration, rate) in enumerate(zip(concentrations, rates)):
        obj = cv_factory(name=f"{rate * 1000:g}mVs_sample_CO2_MeCN_10mMFc_run{index:02d}",
                         options={"electrode area": 0.071})
        obj.compounds = ["Fc", "electrolyte"]
        obj.scan_rate = rate
        obj.concentrations = [f"{concentration:g}M", "0.1M"]
        obj._known_current = (0.4463 * ecat_module.F * obj.electrode_area * concentration / 1000
                              * np.sqrt(ecat_module.F * diffusion * rate
                                        / (ecat_module.R * obj.temperature)))
        cvs.append(obj)

    def peak_current(self, options=None):
        sign = 1 if options.get("segment", 1) == 1 else -1
        area = self.electrode_area if str(options.get("y axis", "")).lower() == "current density" else 1
        return {"ip": sign * self._known_current / area}

    monkeypatch.setattr(ecat_module.cv, "peak_current", peak_current)
    return cvs, diffusion


def _analyze(ecat_module, cvs, **options):
    return ecat_module.sevcik_analysis(cvs, {
        "plot": False, "print": False, "segments": [1, 2], **options,
    })


def test_sevcik_scan_series_selects_species_despite_reordered_metadata(ecat_module, cv_factory, monkeypatch):
    cvs, diffusion = _sevcik_cvs(ecat_module, cv_factory, monkeypatch,
                                [0.001] * 3, [0.05, 0.2, 0.1])
    cvs[1].compounds.reverse()
    cvs[1].concentrations.reverse()
    result = _analyze(ecat_module, cvs, species="Fc")
    assert result.summary["diffusion coefficients / cm^2 s^-1"] == pytest.approx([diffusion, diffusion])
    assert result.summary["species"] == "Fc"
    assert result.summary["concentration source"] == "CV metadata"
    assert result.table["Concentration / M"].tolist() == pytest.approx([0.001] * 3)
    assert result.table["Scan Rate^1/2 / (V s^-1)^1/2"].tolist() == pytest.approx(np.sqrt([0.05, 0.2, 0.1]))


def test_sevcik_concentration_series_recovers_known_diffusion_for_each_segment(ecat_module, cv_factory, monkeypatch):
    cvs, diffusion = _sevcik_cvs(ecat_module, cv_factory, monkeypatch,
                                [0.001, 0.001, 0.003, 0.002], [0.1] * 4)
    cvs[2].compounds.reverse()
    cvs[2].concentrations.reverse()
    result = _analyze(ecat_module, cvs)
    assert result.summary["diffusion coefficients / cm^2 s^-1"] == pytest.approx([diffusion, diffusion])
    assert result.summary["species"] == "Fc"
    assert result.table["Concentration (M)"].tolist() == pytest.approx([0.001, 0.001, 0.003, 0.002])


def test_sevcik_scan_series_requires_species_when_ambiguous(ecat_module, cv_factory, monkeypatch):
    cvs, _ = _sevcik_cvs(ecat_module, cv_factory, monkeypatch, [0.001] * 3, [0.05, 0.1, 0.2])
    with pytest.raises(ValueError, match="species"):
        _analyze(ecat_module, cvs)


def test_sevcik_missing_selected_species_is_not_silent_zero(ecat_module, cv_factory, monkeypatch):
    cvs, _ = _sevcik_cvs(ecat_module, cv_factory, monkeypatch, [0.001, 0.002, 0.003], [0.1] * 3)
    cvs[1].compounds = ["electrolyte"]
    cvs[1].concentrations = ["0.1M"]
    with pytest.raises(ValueError, match="Fc.*missing|missing.*Fc"):
        _analyze(ecat_module, cvs, species="Fc")


def test_sevcik_manual_c_takes_precedence_and_reports_units(ecat_module, cv_factory, monkeypatch):
    cvs, diffusion = _sevcik_cvs(ecat_module, cv_factory, monkeypatch, [0.001] * 3, [0.05, 0.1, 0.2])
    result = _analyze(ecat_module, cvs, C=1e-6)
    assert result.summary["diffusion coefficients / cm^2 s^-1"] == pytest.approx([diffusion, diffusion])
    assert result.summary["concentration source"] == "options['C'] (mol/cm^3)"
    assert result.table["Concentration / M"].tolist() == pytest.approx([0.001] * 3)


@pytest.mark.parametrize("token", ["^-1", "-0.5", "power:-2", "power=-1", "1e-3"])
def test_negative_string_powers_keep_their_sign(ecat_module, token):
    expected = {"^-1": -1, "-0.5": -0.5, "power:-2": -2, "power=-1": -1, "1e-3": 0.001}[token]
    assert ecat_module._normalize_transform_token(token) == ("power", expected)


def test_composite_unit_power_groups_the_whole_unit(ecat_module):
    label = ecat_module._format_symbolic_axis_label("Scan Rate", unit="V/s", transform="sqrt")
    assert r"(V/s)$^{1/2}$" in label


def test_sevcik_concentration_equation_uses_faraday_cubed(ecat_module):
    equation = ecat_module._format_sevcik_diffusion_equation(
        mode="concentration", num_electrons=1, temperature=298, electrode_area=0.071, scan_rate=0.1)
    assert "F^3 * n^3" in equation["symbolic"]


@pytest.mark.parametrize("reference_shift", [0.0, 0.05])
def test_trumpet_plot_all_marks_both_peaks_once_on_shared_scaled_overlay(ecat_module, cv_factory, reference_shift):
    cvs = []
    for index, rate in enumerate([0.01, 0.1, 1.0]):
        forward = np.linspace(-0.5, 0.5, 201)
        reverse = np.linspace(0.495, -0.5, 200)
        peak = 0.2 + 0.03 * np.log10(rate)
        current = np.concatenate([4e-6 * np.exp(-((forward - peak) / 0.03) ** 2),
                                  -3e-6 * np.exp(-((reverse + peak) / 0.03) ** 2)])
        obj = cv_factory(name=f"{rate * 1000:g}mVs_sample_CO2_MeCN_1mMFc_run{index:02d}",
                         potential=np.concatenate([forward, reverse]), current=current)
        obj.scan_rate = rate
        obj.potential_shift({"reference offset": reference_shift, "reference label": "test reference"})
        cvs.append(obj)

    result = ecat_module.trumpet_analysis(cvs, {
        "segment": 1, "plot": False, "print": False, "plot all": True,
        "guess potential": [0.15 - reference_shift, -0.15 - reference_shift],
        "x unit": "mV", "y unit": "nA",
    })
    assert len(plt.get_fignums()) == 1
    ax = plt.gcf().axes[0]
    markers = [collection.get_offsets() for collection in ax.collections
               if hasattr(collection, "get_offsets") and len(collection.get_offsets()) == 2]
    assert len(markers) == 3
    assert len(ax.lines) == 3
    for index, positions in enumerate(markers):
        assert sorted(positions[:, 0]) == pytest.approx(sorted([
            result.table["Anodic Peak Potential (V)"].iloc[index] * 1000,
            result.table["Cathodic Peak Potential (V)"].iloc[index] * 1000,
        ]))
        assert sorted(positions[:, 1]) == pytest.approx([-3000, 4000], rel=0.01)
        peak = 0.2 + 0.03 * np.log10(cvs[index].scan_rate)
        assert sorted(positions[:, 0]) == pytest.approx(
            [(-peak - reference_shift) * 1000, (peak - reference_shift) * 1000], abs=5)


def test_sevcik_explicit_zero_blank_is_allowed(ecat_module, cv_factory, monkeypatch):
    cvs, diffusion = _sevcik_cvs(ecat_module, cv_factory, monkeypatch,
                                [0, 0.001, 0.002], [0.1] * 3)
    cvs[0].compounds = ["electrolyte"]
    cvs[0].concentrations = ["0.1M"]
    cvs[0].zero_concentration_compounds = ["Fc"]
    cvs[0].zero_concentrations = ["0M"]
    result = _analyze(ecat_module, cvs, species="Fc")
    assert result.summary["diffusion coefficients / cm^2 s^-1"] == pytest.approx([diffusion] * 2)


def test_sevcik_scan_series_rejects_varying_selected_concentration(ecat_module, cv_factory, monkeypatch):
    cvs, _ = _sevcik_cvs(ecat_module, cv_factory, monkeypatch, [0.001, 0.002, 0.003], [0.05, 0.1, 0.2])
    with pytest.raises(ValueError, match="constant positive"):
        _analyze(ecat_module, cvs, species="Fc")


def test_sevcik_duplicate_molar_species_requires_unambiguous_metadata(ecat_module, cv_factory, monkeypatch):
    cvs, _ = _sevcik_cvs(ecat_module, cv_factory, monkeypatch, [0.001] * 3, [0.05, 0.1, 0.2])
    cvs[1].compounds.append("Fc")
    cvs[1].concentrations.append("2mM")
    with pytest.raises(ValueError, match="ambiguous"):
        _analyze(ecat_module, cvs, species="Fc")


def test_generic_peak_current_reports_resolved_species(ecat_module, cv_factory, monkeypatch):
    cvs, _ = _sevcik_cvs(ecat_module, cv_factory, monkeypatch, [0.001, 0.002, 0.003], [0.1] * 3)
    result = ecat_module.fit_peak_current(cvs, {
        "plot": False, "print": False, "segment": 1, "species": "Fc",
    })
    assert result.summary["species"] == "Fc"
    assert result.summary["concentration unit"] == "M"


@pytest.mark.parametrize("mode", ["scan rate", "concentration"])
@pytest.mark.parametrize("unit", ["mA/cm^2", "μA/mm^2"])
def test_sevcik_density_and_two_electron_scaling(ecat_module, cv_factory, monkeypatch, mode, unit):
    concentrations = [0.001] * 3 if mode == "scan rate" else [0.001, 0.002, 0.003]
    rates = [0.05, 0.1, 0.2] if mode == "scan rate" else [0.1] * 3
    cvs, diffusion = _sevcik_cvs(ecat_module, cv_factory, monkeypatch, concentrations, rates)
    for obj in cvs:
        obj._known_current *= 2 ** 1.5
    result = _analyze(ecat_module, cvs, species="Fc", **{
        "num electrons": 2, "y axis": "Current Density", "y unit": unit,
    })
    assert result.summary["diffusion coefficients / cm^2 s^-1"] == pytest.approx([diffusion] * 2)
