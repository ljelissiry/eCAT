import numpy as np
import pandas as pd
import pytest
import warnings


def test_reversibility_public_api_and_option_defaults(ecat_module):
    assert callable(ecat_module.reversibility_analysis)
    options = ecat_module.ReversibilityAnalysisOptions.from_options({})
    assert options.phase == "bulk"
    assert options.agreement_tolerance == pytest.approx(0.25)
    assert options.current_ratio_tolerance == pytest.approx(0.10)
    assert options.peak_separation_tolerance == pytest.approx(0.010)
    assert options.num_electrons == pytest.approx(1)


def test_reversibility_phase_is_explicit_and_validated(ecat_module):
    surface = ecat_module.ReversibilityAnalysisOptions.from_options(
        {"phase": "surface"}
    )
    assert surface.phase == "surface"

    with pytest.raises(ecat_module.OptionError, match="phase.*bulk.*surface"):
        ecat_module.ReversibilityAnalysisOptions.from_options({"phase": "auto"})


def test_agreement_tolerance_can_be_changed_globally(ecat_module):
    original = ecat_module.get_defaults("reversibility_analysis")[
        "agreement_tolerance"
    ]
    try:
        ecat_module.set_defaults("agreement tolerance", 0.2)
        assert (
            ecat_module.ReversibilityAnalysisOptions.from_options({}).agreement_tolerance
            == pytest.approx(0.2)
        )
        assert (
            ecat_module.SurfaceCoverageAnalysisOptions.from_options({}).agreement_tolerance
            == pytest.approx(0.2)
        )
    finally:
        ecat_module.set_defaults("agreement tolerance", original)


def test_current_ratio_tolerance_is_independent_and_configurable(ecat_module):
    original = ecat_module.get_defaults("reversibility_analysis")[
        "current_ratio_tolerance"
    ]
    try:
        ecat_module.set_defaults("current ratio tolerance", 0.08)
        options = ecat_module.ReversibilityAnalysisOptions.from_options({})
        assert options.current_ratio_tolerance == pytest.approx(0.08)
        assert options.agreement_tolerance == pytest.approx(0.25)
        with pytest.raises(ecat_module.OptionError, match="current ratio tolerance"):
            ecat_module.ReversibilityAnalysisOptions.from_options(
                {"current ratio tolerance": 1.0}
            )
    finally:
        ecat_module.set_defaults("current ratio tolerance", original)


def test_reversibility_requires_three_distinct_scan_rates(ecat_module, cv_factory):
    cvs = [cv_factory(name="50mVs_A"), cv_factory(name="50mVs_B")]
    for cv_obj in cvs:
        cv_obj.scan_rate = 0.05

    with pytest.raises(ValueError, match="at least three distinct scan rates"):
        ecat_module.reversibility_analysis(
            cvs,
            {"plot": False, "print": False, "guess potential": 0.1},
        )


def test_reversibility_rejects_mixed_conditions(ecat_module, cv_factory):
    cvs = []
    for index, rate in enumerate((0.05, 0.1, 0.2)):
        cv_obj = cv_factory(name=f"{rate:g}Vs_condition_{index}")
        cv_obj.scan_rate = rate
        cv_obj.compounds = ["A" if index < 2 else "B"]
        cv_obj.concentrations = [1.0]
        cvs.append(cv_obj)

    with pytest.raises(ValueError, match="one chemical condition.*group"):
        ecat_module.reversibility_analysis(
            cvs,
            {"plot": False, "print": False, "guess potential": 0.1},
        )


def test_symmetric_agreement_status(ecat_module):
    assert ecat_module._agreement_status(1.0, 1.2, tolerance=0.25)[0] == "agree"
    assert ecat_module._agreement_status(1.0, 2.0, tolerance=0.25)[0] == "disagree"
    assert ecat_module._agreement_status(0.0, 0.0, tolerance=0.25)[0] == "agree"
    assert ecat_module._agreement_status(0.0, 1.0, tolerance=0.25)[0] == "disagree"
    assert ecat_module._agreement_status(np.nan, 1.0, tolerance=0.25)[0] == "unavailable"


def test_replicates_are_retained_but_rate_means_are_fit(ecat_module):
    table = pd.DataFrame(
        {
            "scan rate / V s^-1": [0.1, 0.1, 0.2, 0.5],
            "P1 ip / A": [1.0, 1.2, 2.0, 5.0],
        }
    )

    grouped = ecat_module._rate_mean_table(table, ["P1 ip / A"])

    assert len(table) == 4
    assert len(grouped) == 3
    assert grouped.loc[grouped["scan rate / V s^-1"] == 0.1, "replicate count"].iloc[0] == 2
    assert grouped.loc[grouped["scan rate / V s^-1"] == 0.1, "P1 ip / A"].iloc[0] == pytest.approx(1.1)


def test_matsuda_ayabe_regions_use_literature_boundaries(ecat_module):
    assert ecat_module._matsuda_ayabe_region(15.0, alpha=0.5) == "reversible"
    assert ecat_module._matsuda_ayabe_region(0.1, alpha=0.5) == "quasi-reversible"
    assert ecat_module._matsuda_ayabe_region(1e-3, alpha=0.5) == "irreversible"


def test_nicholson_lambda_and_k0_use_dimensionless_definition(ecat_module):
    evidence = ecat_module._nicholson_evidence(120.0)
    assert evidence["eligible"] is True
    assert evidence["psi"] == pytest.approx(
        (-0.6288 + 0.0021 * 120.0) / (1 - 0.017 * 120.0)
    )
    assert evidence["Lambda"] == pytest.approx(np.sqrt(np.pi) * evidence["psi"])

    k0 = ecat_module._k0_from_lambda(
        evidence["Lambda"],
        diffusion=1e-5,
        scan_rate=0.1,
        num_electrons=1,
        temperature=298.15,
    )
    expected = evidence["Lambda"] * np.sqrt(
        1e-5 * ecat_module.F * 0.1 / (ecat_module.R * 298.15)
    )
    assert k0 == pytest.approx(expected)


def test_matsuda_lambda_survives_above_nicholson_k0_range(ecat_module):
    evidence = ecat_module._nicholson_evidence(62.6)

    assert evidence["psi"] > 7
    assert np.isfinite(evidence["Lambda"])
    assert evidence["eligible"] is False
    assert "too reversible" in evidence["reason"]
    assert ecat_module._matsuda_ayabe_region(evidence["Lambda"]) == "quasi-reversible"


def test_branch_sevcik_diffusion_recovers_input(ecat_module):
    scan_rates = np.array([0.05, 0.1, 0.2, 0.5])
    diffusion = 1.7e-5
    n = 1
    area = 0.071
    concentration = 1e-6
    temperature = 298.15
    slope = (
        0.4463
        * n
        * ecat_module.F
        * area
        * concentration
        * np.sqrt(n * ecat_module.F * diffusion / (ecat_module.R * temperature))
    )
    result = ecat_module._branch_sevcik_diffusion(
        scan_rates,
        slope * np.sqrt(scan_rates),
        num_electrons=n,
        temperature=temperature,
        electrode_area=area,
        concentration=concentration,
    )
    assert result["D / cm^2 s^-1"] == pytest.approx(diffusion)
    assert result["r2"] == pytest.approx(1.0)


def test_chemical_conclusion_uses_physical_branch_current_evidence(ecat_module):
    reversible = ecat_module._chemical_reversibility_decision(
        [0.05, 0.1, 0.2, 0.5],
        [0.98, 1.03, 1.01, 1.00],
        tolerance=0.10,
        min_r2=0.98,
    )
    coupled = ecat_module._chemical_reversibility_decision(
        [0.05, 0.1, 0.2, 0.5],
        [0.35, 0.50, 0.70, 0.90],
        tolerance=0.10,
        min_r2=0.90,
    )
    assert reversible["conclusion"] == "chemically reversible over observed timescale"
    assert "0.98" in reversible["reason"]
    assert "maximum deviation" in reversible["reason"]
    assert "0.1" in reversible["reason"]
    assert coupled["conclusion"] == "coupled chemistry indicated"
    assert coupled["trend toward unity"] is True


def test_bulk_series_reports_quantitative_reversible_to_quasi_transition(ecat_module):
    delta_eps = (0.0610, 0.0626, 0.0626, 0.0634, 0.0658)
    cvs = [
        _WaveCV(rate, delta_ep=delta_ep, ratio=0.98)
        for rate, delta_ep in zip((0.025, 0.05, 0.1, 0.2, 0.5), delta_eps)
    ]

    result = ecat_module.reversibility_analysis(
        cvs,
        {"D": 1e-5, "plot": False, "print": False},
    )

    assert result.summary["electrochemical conclusion"] == (
        "reversible-to-quasi-reversible transition"
    )
    reason = result.summary["electrochemical reason"]
    assert "reversible" in reason
    assert "quasi-reversible" in reason
    assert "Lambda" in reason
    assert "0.025" in reason and "0.5" in reason
    assert result.summary["Nicholson eligible points"] == 2
    assert len(result.fits["Nicholson k0 values"]) == 2


class _WaveCV:
    def __init__(self, rate, *, delta_ep, ratio, peak_center=-0.4, current_power=0.5):
        self.name = f"{rate:g}Vs_A"
        self.scan_rate = rate
        self.temperature = 298.15
        self.electrode_area = 0.071
        self.compounds = ["A"]
        self.concentrations = ["1 mM"]
        self.solvent = "MeCN"
        self.gas = "Ar"
        self._delta_ep = delta_ep
        self._ratio = ratio
        self._peak_center = peak_center
        self._current_power = current_power

    def analysis_segment_data(self, options=None):
        return np.linspace(-0.8, 0.0, 81), np.zeros(81)

    def wave_info(self, options=None):
        current = -2e-5 * self.scan_rate**self._current_power
        return {
            "E(1/2)": self._peak_center,
            "ΔE": self._delta_ep,
            "P1 segment": 1,
            "P1 Ep": self._peak_center - self._delta_ep / 2,
            "P1 ip": current,
            "P1 Ep/2": self._peak_center - self._delta_ep / 2 + 0.05,
            "P1 Δ(Ep - Ep/2)": -0.05,
            "P1 W1/2": 0.09,
            "P1 width status": "ok",
            "P1 branch": "cathodic",
            "P2 segment": 2,
            "P2 Ep": self._peak_center + self._delta_ep / 2,
            "P2 ip": abs(current) * self._ratio,
            "P2 Ep/2": self._peak_center + self._delta_ep / 2 - 0.05,
            "P2 Δ(Ep - Ep/2)": 0.05,
            "P2 W1/2": 0.09,
            "P2 width status": "ok",
            "P2 branch": "anodic",
            "cathodic segment": 1,
            "anodic segment": 2,
            "Epc": self._peak_center - self._delta_ep / 2,
            "Epa": self._peak_center + self._delta_ep / 2,
            "ipc": current,
            "ipa": abs(current) * self._ratio,
            "Δ(Epc - Epc/2)": -0.05,
            "Δ(Epa - Epa/2)": 0.05,
            "W1/2,c": 0.09,
            "W1/2,a": 0.09,
            "cathodic width status": "ok",
            "anodic width status": "ok",
            "|ipa/ipc|": self._ratio,
        }


class _WidthWarningCV(_WaveCV):
    def wave_info(self, options=None):
        warnings.warn(
            "W1/2 is unavailable for this CV: two crossings were not resolved.",
            UserWarning,
        )
        result = super().wave_info(options)
        result["cathodic width status"] = "failed: two crossings were not resolved"
        result["anodic width status"] = "failed: two crossings were not resolved"
        return result


def test_reversibility_retains_width_status_without_leaking_child_warning(ecat_module):
    cvs = [
        _WidthWarningCV(rate, delta_ep=0.12, ratio=0.98)
        for rate in (0.025, 0.05, 0.1, 0.2, 0.5)
    ]

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = ecat_module.reversibility_analysis(
            cvs,
            {"D": 1e-5, "plot": False, "print": False},
        )

    assert not any(str(item.message).startswith("W1/2 is unavailable") for item in caught)
    assert result.table["cathodic width status"].str.startswith("failed").all()
    assert result.table["anodic width status"].str.startswith("failed").all()


def test_bulk_reversibility_reports_kinetic_and_chemical_conclusions(ecat_module):
    cvs = [
        _WaveCV(rate, delta_ep=0.12, ratio=ratio)
        for rate, ratio in zip((0.05, 0.1, 0.2, 0.5), (0.35, 0.50, 0.70, 0.90))
    ]
    result = ecat_module.reversibility_analysis(
        cvs,
        {
            "D": 1e-5,
            "guess potential": -0.4,
            "plot": False,
            "print": False,
            "min r2": 0.90,
        },
    )
    assert isinstance(result, ecat_module.AnalysisResult)
    assert len(result.table) == len(cvs)
    assert result.summary["electrochemical conclusion"] == "quasi-reversible"
    assert result.summary["chemical conclusion"] == "coupled chemistry indicated"
    assert result.summary["preferred k0 source"] == "Nicholson"
    assert result.summary["k0 / cm s^-1"] > 0
    assert "decision tree" in result.diagnostics
    assert "rate means" in result.diagnostics


def test_reversible_series_reports_k0_lower_bound(ecat_module):
    cvs = [
        _WaveCV(rate, delta_ep=0.059, ratio=1.0)
        for rate in (0.05, 0.1, 0.2, 0.5)
    ]
    result = ecat_module.reversibility_analysis(
        cvs,
        {"D": 1e-5, "plot": False, "print": False},
    )
    assert result.summary["electrochemical conclusion"] == "reversible"
    assert result.summary["k0 lower bound / cm s^-1"] > 0
    assert result.summary["preferred k0 source"] == "reversible lower bound"


def test_surface_phase_uses_surface_specific_decision_tree(ecat_module):
    cvs = [
        _WaveCV(rate, delta_ep=0.006, ratio=1.0, current_power=1.0)
        for rate in (0.05, 0.1, 0.2, 0.5)
    ]
    result = ecat_module.reversibility_analysis(
        cvs,
        {"phase": "surface", "plot": False, "print": False},
    )
    assert result.summary["phase"] == "surface"
    assert result.summary["electrochemical conclusion"] == "reversible"
    assert result.summary["D source"] == "not applicable"
    assert result.summary["preferred k0 source"] == "unresolved"
    assert result.diagnostics["decision tree"]["surface"]["ip-v linear"] is True


def test_surface_zero_separation_tolerance_respects_data_resolution(ecat_module):
    cvs = [_WaveCV(rate, delta_ep=0.006, ratio=1.0) for rate in (0.05, 0.1, 0.2)]
    for cv_obj in cvs:
        cv_obj.analysis_segment_data = lambda options=None: (
            np.arange(-0.3, 0.304, 0.004),
            np.zeros(151),
        )
    tolerance = ecat_module._effective_surface_separation_tolerance(cvs, 0.010)
    assert tolerance == pytest.approx(0.012)


def test_reversibility_print_and_plot_report_units_and_diagnostics(
    ecat_module,
    capsys,
):
    cvs = [
        _WaveCV(rate, delta_ep=0.12, ratio=1.0)
        for rate in (0.025, 0.05, 0.1, 0.2, 0.5)
    ]

    result = ecat_module.reversibility_analysis(
        cvs,
        {
            "D": 1e-5,
            "plot": True,
            "plot all": True,
            "print": True,
            "print all": True,
            "pretty print": False,
        },
    )

    output = capsys.readouterr().out
    assert "Reversibility Analysis Equations:" in output
    assert "Reversibility Analysis Parameters:" in output
    assert "Reversibility Analysis Summary:" in output
    assert "Reversibility Analysis Data:" in output
    assert "Parameter" in output
    assert "Symbol" in output
    assert "Electrode Area" in output
    assert "cm^2/s" in output
    assert "Electron-Transfer Rate" in output
    assert output.index("Reversibility Analysis Parameters:") < output.index(
        "Reversibility Analysis Equations:"
    )
    assert output.index("Reversibility Analysis Equations:") < output.index(
        "Reversibility Analysis Summary:"
    )
    assert output.index("Reversibility Analysis Summary:") < output.index(
        "Reversibility Analysis Data:"
    )
    assert len(result.figures) == 2
    assert len(result.figures[0].axes) == 2
    assert result.figures[0].get_layout_engine() is not None
    upper_axis, lower_axis = result.figures[0].axes
    assert upper_axis.get_position().y0 > lower_axis.get_position().y0
    assert upper_axis.get_shared_x_axes().joined(upper_axis, lower_axis)
    assert upper_axis.get_ylabel() == r"$n\Delta E_p$ (mV)"
    assert lower_axis.get_ylabel() == r"$|i_{p,\mathrm{a}}/i_{p,\mathrm{c}}|$"
    assert lower_axis.get_xlabel() == r"Scan Rate (V s$^{-1}$)"
    result.figures[0].canvas.draw()
    assert [label.get_text() for label in lower_axis.get_xticklabels()] == [
        "0.025", "0.05", "0.1", "0.2", "0.5"
    ]
    assert lower_axis.yaxis.get_offset_text().get_text() == ""
    assert result.figures[1].get_layout_engine() is not None
    assert result.figures[1].axes[0].get_xscale() == "log"
    result.figures[1].canvas.draw()
    assert [label.get_text() for label in result.figures[1].axes[0].get_xticklabels()] == [
        "0.025", "0.05", "0.1", "0.2", "0.5"
    ]


def test_bulk_reversibility_equations_are_labeled_and_conditional(
    ecat_module,
    capsys,
):
    cvs = [
        _WaveCV(rate, delta_ep=0.12, ratio=1.0)
        for rate in (0.025, 0.05, 0.1, 0.2, 0.5)
    ]

    result = ecat_module.reversibility_analysis(
        cvs,
        {
            "D": 1e-5,
            "plot": False,
            "print": True,
            "pretty print": False,
        },
    )

    output = capsys.readouterr().out
    assert "[Nicholson Peak-Separation Conversion]" in output
    assert "[Matsuda-Ayabe Classification]" in output
    assert "[Electron-Transfer Rate Conversion]" in output
    assert "[Sevcik Diffusion Estimate]" not in output
    assert "[Irreversible-Asymptote Verification]" not in output
    assert "n Delta Ep / mV" in result.table


def test_reversibility_prints_sevcik_equation_only_when_auto_diffusion_is_attempted(
    ecat_module,
    capsys,
):
    cvs = [
        _WaveCV(rate, delta_ep=0.12, ratio=1.0)
        for rate in (0.025, 0.05, 0.1, 0.2, 0.5)
    ]

    result = ecat_module.reversibility_analysis(
        cvs,
        {
            "species": "A",
            "electrode area": 0.071,
            "plot": False,
            "print": True,
            "pretty print": False,
        },
    )

    output = capsys.readouterr().out
    assert result.summary["D source"] == "branch Sevcik Dapp"
    assert "[Sevcik Diffusion Estimate]" in output
    assert "[Electron-Transfer Rate Conversion]" in output


def test_reversibility_without_diffusion_prints_actionable_rate_evidence(
    ecat_module,
    capsys,
):
    cvs = [
        _WaveCV(rate, delta_ep=0.12, ratio=1.0)
        for rate in (0.025, 0.05, 0.1, 0.2, 0.5)
    ]

    ecat_module.reversibility_analysis(
        cvs,
        {
            "plot": False,
            "print": True,
            "pretty print": False,
        },
    )

    output = capsys.readouterr().out
    assert "[Electron-Transfer Rate Conversion]" not in output
    assert "k0 unresolved: D is required" in output
    assert "provide electrode area and concentration/species metadata" in output


def test_irreversible_asymptote_equation_is_only_shown_for_candidate_series(
    ecat_module,
    capsys,
):
    cvs = [
        _WaveCV(rate, delta_ep=0.299, ratio=1.0)
        for rate in (0.025, 0.05, 0.1, 0.2, 0.5)
    ]

    result = ecat_module.reversibility_analysis(
        cvs,
        {
            "D": 1e-5,
            "plot": False,
            "print": True,
            "pretty print": False,
        },
    )

    output = capsys.readouterr().out
    assert "irreversible" in result.summary["electrochemical conclusion"]
    assert "[Irreversible-Asymptote Verification]" in output


def test_reversibility_table_uses_physical_branches_not_acquisition_order(ecat_module):
    cvs = [
        _WaveCV(rate, delta_ep=0.12, ratio=1.0)
        for rate in (0.025, 0.05, 0.1, 0.2, 0.5)
    ]

    result = ecat_module.reversibility_analysis(
        cvs,
        {"D": 1e-5, "plot": False, "print": False},
    )

    assert {
        "cathodic segment",
        "anodic segment",
        "Epc / V",
        "ipc / A",
        "Epa / V",
        "ipa / A",
        "|ipa/ipc|",
    }.issubset(result.table.columns)
    assert "return/forward ip ratio" not in result.table.columns


def test_bulk_reversibility_print_all_uses_compact_evidence_view(ecat_module):
    cvs = [
        _WaveCV(rate, delta_ep=0.12, ratio=1.0)
        for rate in (0.025, 0.05, 0.1, 0.2, 0.5)
    ]
    result = ecat_module.reversibility_analysis(
        cvs,
        {"D": 1e-5, "plot": False, "print": False},
    )

    display_table = ecat_module._reversibility_display_data_table(result)

    assert display_table.columns.tolist() == [
        "scan rate / V s^-1",
        "E1/2 / V",
        "n Delta Ep / mV",
        "|ipa/ipc|",
        "psi",
        "Lambda",
        "ET Region",
        "Nicholson Use",
    ]
    assert set(display_table["ET Region"]) == {"quasi-reversible"}
    assert all(str(value).startswith("Nicholson") for value in display_table["Nicholson Use"])
    assert "ipc / A" in result.table
    assert "W1/2,c / V" in result.table


def test_surface_reversibility_print_all_uses_surface_evidence_view(ecat_module):
    cvs = [
        _WaveCV(rate, delta_ep=0.006, ratio=1.0, current_power=1.0)
        for rate in (0.025, 0.05, 0.1, 0.2, 0.5)
    ]
    result = ecat_module.reversibility_analysis(
        cvs,
        {"phase": "surface", "plot": False, "print": False},
    )

    display_table = ecat_module._reversibility_display_data_table(result)

    assert display_table.columns.tolist() == [
        "scan rate / V s^-1",
        "E1/2 / V",
        "n Delta Ep / mV",
        "|ipc| / A",
        "|ipa| / A",
        "|ipa/ipc|",
        "Surface Region",
    ]
    assert set(display_table["Surface Region"]) == {"reversible"}
    assert (display_table["|ipc| / A"] >= 0).all()
    assert (display_table["|ipa| / A"] >= 0).all()
    assert "psi" in result.table


def test_reversibility_display_adds_name_for_replicate_scan_rates(ecat_module):
    rates = (0.025, 0.025, 0.05, 0.1, 0.2, 0.5)
    cvs = [
        _WaveCV(rate, delta_ep=0.12, ratio=1.0)
        for rate in rates
    ]
    cvs[0].name = "25mVs_rep1"
    cvs[1].name = "25mVs_rep2"
    result = ecat_module.reversibility_analysis(
        cvs,
        {"D": 1e-5, "plot": False, "print": False},
    )

    display_table = ecat_module._reversibility_display_data_table(result)

    assert display_table.columns[0] == "Name"
    assert display_table["Name"].tolist()[:2] == ["25mVs_rep1", "25mVs_rep2"]
