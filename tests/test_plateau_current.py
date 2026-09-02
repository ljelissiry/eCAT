import numpy as np
import pytest
import matplotlib.pyplot as plt
import warnings
import pandas as pd


class FakePeakCV:
    def __init__(
        self,
        name,
        scan_rate=0.1,
        current=1e-5,
        temperature=298.0,
        fail_without_exact=False,
        call_log=None,
    ):
        self.name = name
        self.scan_rate = scan_rate
        self.current = current
        self.temperature = temperature
        self.electrode_area = 0.07
        self.fail_without_exact = fail_without_exact
        self.calls = []
        self.call_log = call_log
        self.compounds = []
        self.concentrations = []
        self.data = pd.DataFrame({
            "Potential": [-1.0, -0.5, 0.0],
            "Current": [current * 0.5, current, current * 0.25],
        })
        self.units = {"Potential": "V", "Current": "A"}

    def peak_current(self, options=None):
        options = dict(options or {})
        self.calls.append(options)
        if self.call_log is not None:
            self.call_log.append(f"peak:{self.name}")
        if self.fail_without_exact and options.get("exact potential") is None:
            raise ValueError("no local peak")
        source = "peak"
        potential = options.get("exact potential")
        if potential is None and options.get("guess potential") is not None:
            potential = options.get("guess potential")
            source = "guess potential fallback"
        return {
            "ip": self.current,
            "Ep": potential,
            "peak source": source,
            "tangent line": [0.0, 0.0],
            "tangent start": 0,
        }


class FakeScalingPeakCV(FakePeakCV):
    def peak_current(self, options=None):
        result = super().peak_current(options)
        if options and options.get("y unit") in {"uA", "μA"}:
            result["ip"] = result["ip"] * 1e6
            result["tangent line"] = [0.0, 0.0]
        return result


def _opts(**overrides):
    options = {
        "plot": False,
        "print": False,
        "validate plateau": False,
        "require plateau": False,
        "warn ir drop": False,
    }
    options.update(overrides)
    return options


def test_plateau_current_normalized_equation_manual(ecat_module):
    ic = 5e-5
    ip0 = 1e-5
    v = 0.1
    n = 1
    n_prime = 2
    T = 298
    expected = (0.446 * abs(ic / ip0) * np.sqrt(n * ecat_module.F * v / (ecat_module.R * T))) ** 2 / n_prime

    result = ecat_module.plateau_current(
        [],
        _opts(
            ic=ic,
            ip0=ip0,
            **{"ip0 scan rate": v, "temperature": T, "catalyst electrons": n, "turnover electrons": n_prime},
        ),
    )

    assert result.summary["formula mode"] == "normalized"
    assert result.summary["kobs"] == pytest.approx(expected)


def test_plateau_current_accepts_n_cat_and_n_turn_aliases(ecat_module):
    ic = 5e-5
    ip0 = 1e-5
    v = 0.1
    expected = (
        0.446
        * abs(ic / ip0)
        * np.sqrt(2 * ecat_module.F * v / (ecat_module.R * 298))
    ) ** 2 / 4

    result = ecat_module.plateau_current(
        [],
        _opts(
            ic=ic,
            ip0=ip0,
            **{"ip0 scan rate": v, "n_cat": 2, "n_turn": 4},
        ),
    )

    assert result.summary["catalyst electrons"] == pytest.approx(2)
    assert result.summary["turnover electrons"] == pytest.approx(4)
    assert result.summary["kobs"] == pytest.approx(expected)


def test_plateau_equation_is_symbolic_and_labels_carry_symbols(ecat_module):
    values = {
        "catalyst electrons": 2,
        "turnover electrons": 4,
        "temperature": 298,
        "ilim": 5e-5,
        "ip0": 1e-5,
        "ip0 scan rate": 0.1,
    }

    plateau_equation = ecat_module._format_plateau_kobs_equation("normalized", values)
    fowa_equation = ecat_module._format_fowa_kobs_equation(
        {"catalyst electrons": 2, "turnover electrons": 4, "sigma": 1}
    )

    assert plateau_equation["definitions"] == ""
    assert plateau_equation["definitions latex"] == ""
    assert "n_cat" not in plateau_equation["symbolic"]
    assert "n_turn" not in plateau_equation["symbolic"]
    assert "n'" in plateau_equation["symbolic"]
    assert "n_cat" not in fowa_equation["symbolic"]
    assert "n_turn" not in fowa_equation["symbolic"]
    assert "n'" in fowa_equation["symbolic"]
    assert "n = 2 (n_cat" in fowa_equation["definitions"]
    assert "n' = 4 (n_turn" in fowa_equation["definitions"]

    summary_table = ecat_module._plateau_summary_display_table(
        {
            **values,
            "formula mode": "normalized",
            "ilim source": "manual",
            "ip0 source": "manual",
            "plateau validation": "not tested",
        },
        {"valid plateau": None},
        [],
    )
    assert summary_table.columns.tolist() == ["Parameter", "Symbol", "Value"]
    parameter_symbols = dict(zip(summary_table["Parameter"], summary_table["Symbol"]))
    assert parameter_symbols["Catalyst Electrons"] == "n"
    assert parameter_symbols["Turnover Electrons"] == "n'"
    assert parameter_symbols["Temperature"] == "T"

    metric_table = ecat_module._plateau_result_table(
        {
            **values,
            "formula mode": "normalized",
            "kobs": 1.0,
            "ilim source": "manual",
            "ip0 source": "manual",
        },
        transpose=True,
    )
    assert "ip0 Scan Rate (ν_ip0)" in metric_table["Metric"].tolist()


def test_plateau_current_slope_normalized_equation_manual(ecat_module):
    ic = 5e-5
    slope = 2e-5
    T = 298
    expected = (0.446 * abs(ic) / abs(slope) * np.sqrt(ecat_module.F / (ecat_module.R * T))) ** 2

    result = ecat_module.plateau_current(
        [],
        _opts(ic=ic, **{"ip0 sqrt scan rate slope": slope, "temperature": T}),
    )

    assert result.summary["formula mode"] == "slope normalized"
    assert result.summary["kobs"] == pytest.approx(expected)


def test_plateau_current_direct_equation_manual(ecat_module):
    ic = 5e-5
    D = 1e-5
    C = 1e-6
    A = 0.07
    expected = (abs(ic) / (ecat_module.F * A * C)) ** 2 / D

    result = ecat_module.plateau_current(
        [],
        _opts(ic=ic, D=D, C=C, **{"C unit": "mol/cm^3", "electrode area": A}),
    )

    assert result.summary["formula mode"] == "direct"
    assert result.summary["kobs"] == pytest.approx(expected)


def test_plateau_current_auto_uses_shared_non_catalytic_cv_for_ip0(ecat_module):
    cat = FakePeakCV("cat", scan_rate=0.2, current=-5e-5)
    ref = FakePeakCV("ref", scan_rate=0.1, current=-1e-5)
    expected = (
        0.446
        * abs(cat.current / ref.current)
        * np.sqrt(ecat_module.F * ref.scan_rate / (ecat_module.R * cat.temperature))
    ) ** 2

    result = ecat_module.plateau_current(
        cat,
        _opts(
            **{
                "non-catalytic cv": ref,
                "guess potential": -0.8,
                "formula mode": "auto",
            }
        ),
    )

    assert result.summary["formula mode"] == "normalized"
    assert result.summary["ip0 source"] == "non-catalytic cv"
    assert result.summary["ip0"] == pytest.approx(ref.current)
    assert result.summary["ip0 scan rate"] == pytest.approx(ref.scan_rate)
    assert result.summary["kobs"] == pytest.approx(expected)


def test_plateau_non_catalytic_guess_potential_overrides_global_exact_potential(
    ecat_module,
):
    cat = FakePeakCV("cat", scan_rate=0.2, current=-5e-5)
    ref = FakePeakCV("ref", scan_rate=0.1, current=-1e-5)

    result = ecat_module.plateau_current(
        cat,
        _opts(
            **{
                "non-catalytic cv": ref,
                "exact potential": -0.4,
                "non-catalytic guess potential": -0.8,
                "formula mode": "auto",
            }
        ),
    )

    assert result.summary["formula mode"] == "normalized"
    assert cat.calls[-1]["exact potential"] == pytest.approx(-0.4)
    assert "guess potential" not in cat.calls[-1]
    assert ref.calls[-1]["guess potential"] == pytest.approx(-0.8)
    assert "exact potential" not in ref.calls[-1]


def test_plateau_peak_extraction_does_not_forward_display_units(ecat_module):
    cat = FakeScalingPeakCV("cat", scan_rate=0.2, current=-5e-6)

    result = ecat_module._extract_current_with_peak_current(
        cat,
        {"exact potential": -0.8, "y unit": "uA", "x unit": "mV", "y axis": "current"},
        role="catalytic",
    )

    assert result["current"] == pytest.approx(-5e-6)
    assert cat.calls[-1]["y unit"] == "A"
    assert "x unit" not in cat.calls[-1]
    assert cat.calls[-1]["y axis"] == "Current"


def test_plateau_current_auto_mode_error_lists_missing_input_paths(ecat_module):
    with pytest.raises(ValueError) as excinfo:
        ecat_module.plateau_current([], _opts(ic=5e-5))

    message = str(excinfo.value)
    assert "could not resolve formula mode automatically" in message
    assert "direct" in message
    assert "D" in message
    assert "C" in message
    assert "electrode area" in message
    assert "slope-normalized" in message
    assert "ip0 sqrt scan rate slope" in message
    assert "multiple non-catalytic cvs" in message
    assert "normalized" in message
    assert "non-catalytic cv" in message
    assert "Received:" in message


def test_cv_plateau_current_delegates_to_batch_function(ecat_module, cv_factory):
    cv_obj = cv_factory(name="100mVs_cat_MeCN_CO2_1mM_test")
    options = _opts(
        ic=5e-5,
        ip0=1e-5,
        **{"ip0 scan rate": 0.1, "temperature": 298},
    )

    method_result = cv_obj.plateau_current(options)
    function_result = ecat_module.plateau_current(cv_obj, options)

    assert isinstance(method_result, ecat_module.AnalysisResult)
    assert isinstance(function_result, ecat_module.AnalysisResult)
    ecat_module.pd.testing.assert_frame_equal(method_result.table, function_result.table)
    assert method_result.summary["formula mode"] == "normalized"


def test_plateau_forced_origin_fit(ecat_module):
    x = np.array([0.1, 0.2, 0.4])
    y = 3.5 * x

    slope, y_pred, r2, residuals = ecat_module._plateau_fit_forced_origin(x, y)

    assert slope == pytest.approx(3.5)
    assert y_pred.tolist() == pytest.approx(y)
    assert r2 == pytest.approx(1.0)
    assert residuals.tolist() == pytest.approx([0, 0, 0])


def test_plateau_subset_flat_series_passes(ecat_module):
    df = ecat_module.pd.DataFrame(
        {
            "cv": ["a", "b", "c"],
            "scan rate": [0.1, 0.4, 0.9],
            "sqrt scan rate": np.sqrt([0.1, 0.4, 0.9]),
            "ic": [-1e-5, -1.01e-5, -0.99e-5],
            "abs ic": [1e-5, 1.01e-5, 0.99e-5],
        }
    )

    selected = ecat_module._select_plateau_subset(df, _opts(**{"validate plateau": True}))

    assert selected["valid plateau"] is True
    assert selected["ilim"] == pytest.approx(np.mean([-1e-5, -1.01e-5, -0.99e-5]))
    assert selected["accepted indices"] == [0, 1, 2]


def test_plateau_subset_strong_slope_fails(ecat_module):
    df = ecat_module.pd.DataFrame(
        {
            "cv": ["a", "b", "c"],
            "scan rate": [0.1, 0.4, 0.9],
            "sqrt scan rate": np.sqrt([0.1, 0.4, 0.9]),
            "ic": [-1e-5, -2e-5, -4e-5],
            "abs ic": [1e-5, 2e-5, 4e-5],
        }
    )

    with pytest.raises(ValueError, match="No scan-rate-independent plateau current"):
        ecat_module._select_plateau_subset(
            df,
            _opts(**{"validate plateau": True, "require plateau": True, "plateau min cvs": 2}),
        )


def test_plateau_catalytic_extraction_trusts_peak_current_fallback(ecat_module):
    cv = FakePeakCV("cat", current=-2e-5)
    result = ecat_module._extract_current_with_peak_current(
        cv,
        _opts(**{"guess potential": -0.8, "peak fallback": "guess potential"}),
        role="catalytic",
    )

    assert result["current"] == pytest.approx(-2e-5)
    assert result["potential"] == pytest.approx(-0.8)
    assert result["source"] == "guess potential fallback"
    assert len(cv.calls) == 1
    assert cv.calls[-1]["guess potential"] == pytest.approx(-0.8)
    assert "exact potential" not in cv.calls[-1]
    assert cv.calls[-1]["peak fallback"] == "guess potential"


def test_plateau_reference_fallback_potential_still_promotes_to_exact(ecat_module):
    cv = FakePeakCV("ref", current=-1e-5)

    result = ecat_module._extract_current_with_peak_current(
        cv,
        _opts(**{"guess potential": -0.7}),
        role="non-catalytic",
        fallback_potential=-0.9,
    )

    assert result["current"] == pytest.approx(-1e-5)
    assert result["potential"] == pytest.approx(-0.9)
    assert cv.calls[-1]["exact potential"] == pytest.approx(-0.9)
    assert "guess potential" not in cv.calls[-1]


def test_plateau_single_cv_records_warning_without_python_warning(ecat_module):
    cv = FakePeakCV("cat", current=-2e-5)

    with warnings.catch_warnings(record=True) as captured_warnings:
        warnings.simplefilter("always")
        result = ecat_module.plateau_current(
            cv,
            _opts(
                ip0=1e-5,
                **{
                    "ip0 scan rate": 0.1,
                    "validate plateau": True,
                    "require plateau": True,
                    "print": False,
                },
            ),
        )

    assert not captured_warnings
    assert result.summary["valid plateau"] is None
    assert result.summary["plateau validation"] == "not tested"
    assert "scan-rate independence cannot be tested" in result.warnings
    assert "plateau details" in result.diagnostics
    assert "plateau warning" in result.diagnostics["plateau details"].columns


def test_plateau_single_cv_table_is_compact_and_diagnostics_keep_full_details(ecat_module):
    cat = FakePeakCV("cat", scan_rate=0.2, current=-5e-5)
    ref = FakePeakCV("ref", scan_rate=0.1, current=-1e-5)

    result = ecat_module.plateau_current(
        cat,
        _opts(
            **{
                "non-catalytic cv": ref,
                "guess potential": -0.8,
                "validate plateau": True,
            }
        ),
    )

    assert list(result.table.columns) == ["Metric", "Value"]
    assert "kobs" in set(result.table["Metric"])
    assert not any(column.startswith("plateau subset") for column in result.table.columns)
    assert "plateau details" in result.diagnostics
    details = result.diagnostics["plateau details"]
    assert details.loc[0, "formula mode"] == "normalized"
    assert details.loc[0, "ip0"] == pytest.approx(ref.current)
    assert "plateau subset cvs" in details.columns


def test_plateau_print_all_prints_compact_output_and_diagnostics(ecat_module, capsys):
    cat = FakePeakCV("cat", scan_rate=0.2, current=-5e-5)
    ref = FakePeakCV("ref", scan_rate=0.1, current=-1e-5)

    ecat_module.plateau_current(
        cat,
        _opts(
            **{
                "non-catalytic cv": ref,
                "guess potential": -0.8,
                "validate plateau": True,
                "print": True,
                "print all": True,
                "pretty print": False,
            }
        ),
    )

    out = capsys.readouterr().out
    assert "Plateau Current Parameters:" in out
    assert "Plateau Current Summary" in out
    assert "Plateau Current Data" in out
    assert "plateau subset cvs" in out
    assert "Catalytic Current Diagnostics" not in out
    assert "ip0 Current Diagnostics" not in out
    assert "extraction potential" not in out
    assert "kobs" in out


def test_plateau_plot_all_single_catalytic_and_reference_cv_uses_one_overlay(
    ecat_module,
    monkeypatch,
):
    calls = []
    cat = FakePeakCV("cat", scan_rate=0.2, current=-5e-5)
    ref = FakePeakCV("ref", scan_rate=0.1, current=-1e-5)

    def record_multiplot(cvs, options=None):
        calls.append(("multiplot", [getattr(cv, "name", "") for cv in cvs]))

    monkeypatch.setattr(ecat_module, "multiplot", record_multiplot)
    monkeypatch.setattr(
        ecat_module,
        "_plot_ip0_sqrt_fit",
        lambda ip0_df, slope, options: calls.append(("ip0 sqrt fit", [])),
    )
    monkeypatch.setattr(
        ecat_module,
        "_plot_plateau_validation",
        lambda ic_df, selection, options: calls.append(("plateau validation", [])),
    )

    ecat_module.plateau_current(
        cat,
        _opts(
            **{
                "plot all": True,
                "non-catalytic cv": ref,
                "guess potential": -1.6,
                "validate plateau": True,
            }
        ),
    )

    assert calls == [("multiplot", ["ref", "cat"])]


def test_plateau_plot_all_uses_i_over_ip0_diagnostic_when_available(
    ecat_module,
    monkeypatch,
):
    calls = []
    cat = FakePeakCV("cat", scan_rate=0.2, current=-5e-5)
    ref = FakePeakCV("ref", scan_rate=0.1, current=-1e-5)

    def record_multiplot(cvs, options=None):
        calls.append((list(cvs), dict(options or {})))

    monkeypatch.setattr(ecat_module, "multiplot", record_multiplot)

    result = ecat_module.plateau_current(
        cat,
        _opts(
            **{
                "plot all": True,
                "non-catalytic cv": ref,
                "guess potential": -1.6,
                "validate plateau": True,
            }
        ),
    )

    assert calls
    plotted_cvs, plot_options = calls[0]
    assert plot_options["y axis"] == "i/ip0"
    assert plot_options["ylabel"] == "$i / i_p^0$"
    assert all("i/ip0" in cv_obj.data.columns for cv_obj in plotted_cvs)
    assert not any("i/ip0" in warning for warning in result.warnings)


def test_plateau_plot_all_falls_back_to_current_diagnostic_without_ip0(
    ecat_module,
    monkeypatch,
):
    calls = []
    cat = FakePeakCV("cat", scan_rate=0.2, current=-5e-5)

    def record_multiplot(cvs, options=None):
        calls.append((list(cvs), dict(options or {})))

    monkeypatch.setattr(ecat_module, "multiplot", record_multiplot)

    result = ecat_module.plateau_current(
        cat,
        _opts(
            **{
                "plot all": True,
                "formula mode": "direct",
                "D": 1e-5,
                "C": 1e-6,
                "C unit": "mol/cm^3",
                "electrode area": 0.07,
            }
        ),
    )

    assert calls
    assert calls[0][1].get("y axis") == "Current"
    assert "i/ip0 diagnostic requested but ip0 could not be resolved" in result.warnings


def test_plateau_raw_fallback_diagnostic_uses_autoscaled_current_axis(
    ecat_module,
    cv_factory,
):
    plt.close("all")
    potential = np.linspace(-0.25, 0.25, 51)
    current = (
        0.4e-6
        + 0.5e-6 * potential
        + 6.0e-6 * np.exp(-((potential - 0.15) / 0.035) ** 2)
    )
    cv_obj = cv_factory(
        name="100mVs_sample_CO2_MeCN_10mM_Fc_run02",
        potential=potential,
        current=current,
    )

    ecat_module.plateau_current(
        cv_obj,
        _opts(
            **{
                "plot all": True,
                "formula mode": "direct",
                "D": 1e-5,
                "C": 1e-6,
                "C unit": "mol/cm^3",
                "electrode area": 0.07,
                "exact potential": 0.15,
                "tangent range": [0.1, 0.25],
                "percent threshold": 100,
            }
        ),
    )

    ax = plt.gca()
    ax.figure.canvas.draw()
    assert "μA" in ax.get_ylabel()
    assert ax.yaxis.get_offset_text().get_text() == ""


def test_plateau_normalized_diagnostic_disables_y_axis_offset(
    ecat_module,
    cv_factory,
):
    plt.close("all")
    potential = np.linspace(-0.25, 0.25, 51)
    current = (
        0.4e-6
        + 0.5e-6 * potential
        + 6.0e-6 * np.exp(-((potential - 0.15) / 0.035) ** 2)
    )
    cv_obj = cv_factory(
        name="100mVs_sample_CO2_MeCN_10mM_Fc_run01",
        potential=potential,
        current=current,
    )

    ecat_module._plot_plateau_cv_diagnostic(
        [cv_obj],
        _opts(),
        ip0_values=[1e-6],
    )

    ax = plt.gca()
    ax.figure.canvas.draw()
    assert ax.get_ylabel() == "$i / i_p^0$"
    assert ax.yaxis.get_offset_text().get_text() == ""


def test_plateau_validation_plot_autoscales_current_axis_without_offset(ecat_module):
    plt.close("all")
    ic_df = pd.DataFrame(
        {
            "sqrt scan rate": [0.1, 0.2, 0.4],
            "abs ic": [3.85e-6, 7.9e-6, 1.58e-5],
        }
    )
    selection = {"accepted indices": [0, 1, 2]}

    ecat_module._plot_plateau_validation(ic_df, selection, _opts())

    ax = plt.gca()
    ax.figure.canvas.draw()
    assert ax.get_ylabel() == "|i_c| / μA"
    assert ax.yaxis.get_offset_text().get_text() == ""
    plotted_y = np.asarray(ax.collections[0].get_offsets())[:, 1]
    np.testing.assert_allclose(plotted_y, [3.85, 7.9, 15.8])


def test_plateau_auto_groups_flat_list_by_species_and_validates_each_condition(
    ecat_module,
    monkeypatch,
):
    calls = []
    cvs = []
    for concentration, scale in [("1 mM", 1.0), ("2 mM", 1.4)]:
        for scan_rate in [0.1, 0.4]:
            cv = FakePeakCV(
                f"cat {concentration} {scan_rate}",
                scan_rate=scan_rate,
                current=-5e-5 * scale,
            )
            cv.compounds = ["Cat", "Substrate"]
            cv.concentrations = ["1 mM", concentration]
            cvs.append(cv)

    monkeypatch.setattr(
        ecat_module,
        "multiplot",
        lambda cvs, options=None: calls.append(("multiplot", [cv.name for cv in cvs])),
    )
    monkeypatch.setattr(
        ecat_module,
        "_plot_plateau_validation",
        lambda ic_df, selection, options: calls.append(("validation", ic_df["cv"].tolist())),
    )

    result = ecat_module.plateau_current(
        cvs,
        _opts(
            **{
                "plot all": True,
                "ip0": 1e-5,
                "ip0 scan rate": 0.1,
                "validate plateau": False,
                "group mode": "auto",
            }
        ),
    )

    assert len(result.table) == 2
    assert result.summary["groups"] == 2
    assert result.summary["group mode"] == "auto"
    assert result.summary["group by"] == "species"
    assert len(result.diagnostics["groups"]) == 2
    assert "condition" in result.diagnostics["plateau details"].columns
    assert sum(kind == "validation" for kind, _payload in calls) == 2
    assert sum(kind == "multiplot" for kind, _payload in calls) == 1


def test_plateau_grouped_output_feeds_fit_rate(ecat_module, monkeypatch):
    cvs = []
    for concentration, scale in [("1 mM", 1.0), ("2 mM", 1.4), ("3 mM", 1.8)]:
        for scan_rate in [0.1, 0.4]:
            cv = FakePeakCV(
                f"cat {concentration} {scan_rate}",
                scan_rate=scan_rate,
                current=-5e-5 * scale,
            )
            cv.compounds = ["Cat", "Substrate"]
            cv.concentrations = ["1 mM", concentration]
            cvs.append(cv)

    monkeypatch.setattr(ecat_module, "multiplot", lambda cvs, options=None: None)

    plateau = ecat_module.plateau_current(
        cvs,
        _opts(
            **{
                "ip0": 1e-5,
                "ip0 scan rate": 0.1,
                "validate plateau": False,
                "group mode": "auto",
            }
        ),
    )
    fit_input = ecat_module.fit_rate(
        plateau,
        {"plot": False, "print": False, "fit": False},
    )

    assert "kobs" in plateau.table
    assert plateau.table["kobs"].dtype.kind in "fc"
    assert "Substrate Concentration (M)" in plateau.table
    assert "Name" not in plateau.table
    assert "full_results_df" in plateau.table.attrs
    assert "Substrate Concentration (M)" in plateau.table.attrs["full_results_df"]
    assert fit_input.table["x kind"].eq("concentration").all()
    assert fit_input.table["x raw"].tolist() == pytest.approx([0.001, 0.002, 0.003])


def test_plateau_nested_duplicate_context_uses_condition_not_name(ecat_module, monkeypatch):
    groups = []
    for index, current in enumerate((-5e-5, -6e-5), start=1):
        cv = FakePeakCV(
            f"condition {index}",
            scan_rate=0.1,
            current=current,
        )
        cv.compounds = ["Cat", "Substrate"]
        cv.concentrations = ["1 mM", "2 mM"]
        groups.append([cv])

    monkeypatch.setattr(ecat_module, "multiplot", lambda cvs, options=None: None)
    result = ecat_module.plateau_current(
        groups,
        _opts(
            **{
                "ip0": 1e-5,
                "ip0 scan rate": 0.1,
                "validate plateau": False,
                "print": False,
            }
        ),
    )

    assert result.table.columns[0] == "Condition"
    assert "Name" not in result.table.columns


def test_plateau_grouped_prints_summary_equation_and_hides_peak_tables(
    ecat_module,
    monkeypatch,
    capsys,
):
    cvs = []
    for concentration, scale in [("1 %", 1.0), ("2 %", 1.4)]:
        for scan_rate in [0.1, 0.4]:
            cv = FakePeakCV(
                f"cat CO2 {concentration} {scan_rate}",
                scan_rate=scan_rate,
                current=-5e-5 * scale,
            )
            cv.compounds = ["Cat", "CO2"]
            cv.concentrations = ["1 mM", concentration]
            cvs.append(cv)

    monkeypatch.setattr(ecat_module, "multiplot", lambda cvs, options=None: None)
    monkeypatch.setattr(ecat_module, "display", None)
    monkeypatch.setattr(ecat_module, "Math", None)

    plateau = ecat_module.plateau_current(
        cvs,
        _opts(
            **{
                "ip0": 1e-5,
                "ip0 scan rate": 0.1,
                "n_turn": 2,
                "validate plateau": False,
                "group mode": "auto",
                "print": True,
                "print all": True,
                "pretty print": False,
            }
        ),
    )

    out = capsys.readouterr().out
    assert "Plateau Current Parameters:" in out
    assert "[Plateau Current Equations]" in out
    assert "Turnover Electrons" in out
    assert "Symbol" in out
    assert "n'" in out
    assert "n' = n_turn (turnover electron count)" not in out
    assert "n' = 2 (n_turn, turnover electron count)" not in out
    assert "v_ip0 = 0.1 V/s" not in out
    assert "Plateau Current Summary:" in out
    assert "Plateau Current Data:" in out
    assert out.index("Plateau Current Parameters:") < out.index(
        "[Plateau Current Equations]"
    )
    assert out.index("[Plateau Current Equations]") < out.index(
        "Plateau Current Summary:"
    )
    assert out.index("Plateau Current Summary:") < out.index(
        "Plateau Current Data:"
    )
    assert "Catalytic Current Diagnostics" not in out
    assert "ip0 Current Diagnostics" not in out
    assert "extraction potential" not in out
    assert "condition" not in plateau.table.columns
    assert "Compounds" in plateau.table.columns
    assert "CO2 Concentration (%)" not in plateau.table.columns
    assert "CO2 Concentration (%)" in plateau.table.attrs["full_results_df"].columns


def test_plateau_display_table_puts_units_in_headers_not_values(ecat_module):
    table = pd.DataFrame(
        {
            "Compounds": ["1 mM Cat, 10 % CO2"],
            "ilim": [52.64e-6],
            "ilim/ip0": [1.9],
            "kobs": [1.398],
        }
    )
    table.attrs["units"] = ecat_module._plateau_units_map()

    display_table = ecat_module._plateau_results_display_table(
        table,
        {"sig figs": 4},
    )

    assert "ilim / μA" in display_table.columns
    assert "kobs / s⁻¹" in display_table.columns
    assert display_table.loc[0, "ilim / μA"] == "52.64"
    assert display_table.loc[0, "kobs / s⁻¹"] == "1.398"


def test_fowa_display_table_puts_kobs_units_in_header(ecat_module):
    table = pd.DataFrame({"Compounds": ["10 % CO2"], "kobs": [1.576e4]})
    table.attrs["units"] = {"kobs": "s^-1"}

    display_table = ecat_module._fowa_results_display_table(table, {"sig figs": 4})

    assert "kobs / s⁻¹" in display_table.columns
    assert display_table.loc[0, "kobs / s⁻¹"] == "1.576e+04"


def test_plateau_tangent_equations_are_promoted_to_results(ecat_module):
    cat = FakePeakCV("cat", scan_rate=0.2, current=-5e-5)
    ref = FakePeakCV("ref", scan_rate=0.1, current=-1e-5)

    result = ecat_module.plateau_current(
        cat,
        _opts(
            **{
                "non-catalytic cv": ref,
                "guess potential": -0.8,
                "validate plateau": True,
            }
        ),
    )

    metrics = set(result.table["Metric"])
    assert "ilim tangent" in metrics
    assert "ip0 tangent" in metrics
    details = result.diagnostics["plateau details"]
    assert details.loc[0, "ilim tangent"] == "y = 0.000x + 0.000"
    assert details.loc[0, "ip0 tangent"] == "y = 0.000x + 0.000"


def test_plateau_nested_lists_define_explicit_validation_groups(
    ecat_module,
    monkeypatch,
):
    groups = [
        [
            FakePeakCV("group a slow", scan_rate=0.1, current=-5e-5),
            FakePeakCV("group a fast", scan_rate=0.4, current=-5.1e-5),
        ],
        [
            FakePeakCV("group b slow", scan_rate=0.1, current=-7e-5),
            FakePeakCV("group b fast", scan_rate=0.4, current=-7.1e-5),
        ],
    ]
    validation_calls = []

    monkeypatch.setattr(ecat_module, "multiplot", lambda cvs, options=None: None)
    monkeypatch.setattr(
        ecat_module,
        "_plot_plateau_validation",
        lambda ic_df, selection, options: validation_calls.append(ic_df["cv"].tolist()),
    )

    result = ecat_module.plateau_current(
        groups,
        _opts(
            **{
                "plot all": True,
                "ip0": 1e-5,
                "ip0 scan rate": 0.1,
                "validate plateau": False,
            }
        ),
    )

    assert len(result.table) == 2
    assert result.summary["group mode"] == "nested"
    assert len(validation_calls) == 2


def test_plateau_normalized_overlay_uses_recorded_extraction_redraw(
    ecat_module,
    monkeypatch,
):
    cat = FakePeakCV("cat", scan_rate=0.2, current=-5e-5)
    ref = FakePeakCV("ref", scan_rate=0.1, current=-1e-5)
    redraw_calls = []
    plotted = []

    monkeypatch.setattr(
        ecat_module,
        "multiplot",
        lambda cvs, options=None: plotted.append((list(cvs), dict(options or {}))),
    )
    monkeypatch.setattr(
        ecat_module,
        "_plot_fowa_normalized_diagnostics",
        lambda ax, diagnostic_calls, copy_by_original_id, object_offsets, options: redraw_calls.append(
            (diagnostic_calls, copy_by_original_id)
        ),
    )

    ecat_module.plateau_current(
        cat,
        _opts(
            **{
                "plot all": True,
                "non-catalytic cv": ref,
                "guess potential": -1.6,
                "validate plateau": True,
            }
        ),
    )

    assert plotted
    assert redraw_calls
    diagnostic_calls, copy_by_original_id = redraw_calls[0]
    assert {call["kind"] for call in diagnostic_calls} == {"plateau_extraction"}
    assert {getattr(call["obj"], "name", "") for call in diagnostic_calls} == {"cat", "ref"}
    assert all("i/ip0" in copy.data.columns for copy in copy_by_original_id.values())


def test_plateau_current_legacy_alias_removed(ecat_module):
    assert not hasattr(ecat_module, "PlateauCurrent")


def test_plateau_plot_all_runs_reference_diagnostic_before_plateau_validation(ecat_module, monkeypatch):
    order = []
    refs = [
        FakePeakCV("ref slow", scan_rate=0.1, current=-1e-5, call_log=order),
        FakePeakCV("ref fast", scan_rate=0.4, current=-2e-5, call_log=order),
    ]
    cats = [
        FakePeakCV("cat slow", scan_rate=0.1, current=-5e-5, call_log=order),
        FakePeakCV("cat fast", scan_rate=0.4, current=-5.1e-5, call_log=order),
    ]

    def record_multiplot(cvs, options=None):
        names = [getattr(item, "name", "") for item in cvs]
        order.append("reference multiplot" if names and names[0].startswith("ref") else "catalytic multiplot")

    monkeypatch.setattr(ecat_module, "multiplot", record_multiplot)
    monkeypatch.setattr(
        ecat_module,
        "_plot_ip0_sqrt_fit",
        lambda ip0_df, slope, options: order.append("ip0 sqrt fit"),
    )
    monkeypatch.setattr(
        ecat_module,
        "_plot_plateau_validation",
        lambda ic_df, selection, options: order.append("plateau validation"),
    )

    ecat_module.plateau_current(
        cats,
        _opts(
            **{
                "plot all": True,
                "non-catalytic cvs": refs,
                "guess potential": -1.6,
                "validate plateau": False,
            }
        ),
    )

    assert order == [
        "peak:ref slow",
        "peak:ref fast",
        "peak:cat slow",
        "peak:cat fast",
        "reference multiplot",
        "plateau validation",
    ]


def test_plateau_plot_all_peak_current_diagnostics_use_multiplot_axis_scaling(
    ecat_module,
    cv_factory,
):
    plt.close("all")
    potential = np.linspace(-0.25, 0.25, 51)
    current_large = (
        0.4e-6
        + 0.5e-6 * potential
        + 6.0e-6 * np.exp(-((potential - 0.15) / 0.035) ** 2)
    )
    current_small = current_large * 1e-3
    cvs = [
        cv_factory(
            name="50mVs_sample_CO2_MeCN_10mM_Fc_run01",
            potential=potential,
            current=current_small,
        ),
        cv_factory(
            name="100mVs_sample_CO2_MeCN_10mM_Fc_run02",
            potential=potential,
            current=current_large,
        ),
    ]

    ecat_module.plateau_current(
        cvs,
        _opts(
            **{
                "plot": False,
                "print": False,
                "plot all": True,
                "non-catalytic current": 1e-6,
                "ip0 scan rate": 0.1,
                "validate plateau": False,
                "exact potential": 0.15,
                "tangent range": [0.1, 0.25],
                "percent threshold": 100,
            }
        ),
    )

    try:
        normalized_axes = [
            ax
            for fig_num in plt.get_fignums()
            for ax in plt.figure(fig_num).axes
            if ax.get_ylabel() == "$i / i_p^0$"
        ]
        assert normalized_axes
        normalized_axes[0].figure.canvas.draw()
        assert normalized_axes[0].yaxis.get_offset_text().get_text() == ""
        vertical_segments = [
            np.asarray(segment)
            for collection in normalized_axes[0].collections
            for segment in getattr(collection, "get_segments", lambda: [])()
            if len(segment) == 2
            and np.isclose(segment[0][0], segment[1][0])
            and collection.get_linestyle()
        ]
        peak_segment = max(
            vertical_segments,
            key=lambda segment: abs(segment[1][1] - segment[0][1]),
        )

        vertical_lengths = [
            abs(segment[1][1] - segment[0][1])
            for segment in vertical_segments
        ]

        assert peak_segment[0][0] == pytest.approx(0.15)
        assert max(np.nanmax(np.abs(segment[:, 1])) for segment in vertical_segments) < 10
        diagnostic_offsets = [
            np.asarray(collection.get_offsets(), dtype=float)
            for collection in normalized_axes[0].collections
            if len(collection.get_offsets())
        ]
        assert diagnostic_offsets
        assert max(
            np.nanmax(np.abs(offsets[:, 1]))
            for offsets in diagnostic_offsets
        ) < 10
        blue_peak_markers = [
            np.asarray(collection.get_offsets(), dtype=float)
            for collection in normalized_axes[0].collections
            if len(collection.get_offsets())
            and len(collection.get_facecolors())
            and np.allclose(collection.get_facecolors()[0][:3], (0.12156863, 0.46666667, 0.70588235))
        ]
        assert not blue_peak_markers
        assert min(vertical_lengths) < 0.02
    finally:
        plt.close("all")


def test_plateau_normalized_diagnostic_marks_non_catalytic_guess_fallback(
    ecat_module,
    cv_factory,
):
    plt.close("all")
    potential = np.linspace(-0.25, 0.25, 51)
    current = (
        0.4e-6
        + 0.5e-6 * potential
        + 6.0e-6 * np.exp(-((potential - 0.15) / 0.035) ** 2)
    )
    ref_cv = cv_factory(
        name="100mVs_reference_CO2_MeCN_10mM_Fc_run01",
        potential=potential,
        current=current,
    )
    ip0 = 1e-6
    normalized_ref = ecat_module._copy_cv_with_normalized_current_axis(
        ref_cv,
        ip0,
        {},
    )
    fig, ax = plt.subplots()

    ecat_module._plot_plateau_normalized_extraction_diagnostics(
        ax,
        [
            {
                "kind": "plateau_extraction",
                "obj": ref_cv,
                "role": "non-catalytic",
                "options": {"guess potential": 0.15},
                "source": "guess potential fallback",
                "potential": 0.15,
                "current": ip0,
                "baseline current": 0.0,
                "peak index": 40,
                "tangent slope": 0.0,
                "tangent intercept": 0.0,
                "tangent start": 0,
                "fit indices": [],
            }
        ],
        {id(ref_cv): normalized_ref},
        {id(ref_cv): 0.0},
        {id(ref_cv): ip0},
        {},
    )

    try:
        blue_peak_markers = [
            np.asarray(collection.get_offsets(), dtype=float)
            for collection in ax.collections
            if len(collection.get_offsets())
            and len(collection.get_facecolors())
            and np.allclose(collection.get_facecolors()[0][:3], (0.12156863, 0.46666667, 0.70588235))
        ]
        assert blue_peak_markers
        assert any(
            np.any(np.isclose(offsets[:, 0], 0.15))
            for offsets in blue_peak_markers
        )
    finally:
        plt.close(fig)


def test_plateau_extracts_raw_current_from_pre_normalized_cvs(
    ecat_module,
    cv_factory,
):
    potential = np.linspace(-0.25, 0.25, 51)
    current = (
        0.4e-6
        + 0.5e-6 * potential
        + 6.0e-6 * np.exp(-((potential - 0.15) / 0.035) ** 2)
    )
    cv_obj = cv_factory(
        name="100mVs_sample_CO2_MeCN_10mM_Fc_run01",
        potential=potential,
        current=current,
    )
    normalized_cv = ecat_module.normalize_current(
        cv_obj,
        {"ip0": 1e-6, "print": False},
    )

    result = ecat_module.plateau_current(
        normalized_cv,
        _opts(
            **{
                "ip0": 1e-6,
                "ip0 scan rate": 0.1,
                "exact potential": 0.15,
                "tangent potential": 0.0,
                "plot": False,
                "plot all": False,
            }
        ),
    )

    extracted = result.diagnostics["catalytic currents"].iloc[0]
    assert extracted["ic"] == pytest.approx(6e-6, rel=0.1)
    assert abs(extracted["ic"]) < 1e-3


def test_plateau_y_axis_current_aliases_raw_diagnostic_when_explicit(
    ecat_module,
    cv_factory,
    monkeypatch,
):
    cat = FakePeakCV("cat", scan_rate=0.2, current=-5e-5)
    calls = []

    monkeypatch.setattr(
        ecat_module,
        "multiplot",
        lambda cvs, options=None: calls.append(dict(options or {})),
    )

    ecat_module.plateau_current(
        cat,
        _opts(
            **{
                "plot all": True,
                "formula mode": "direct",
                "D": 1e-5,
                "C": 1e-6,
                "C unit": "mol/cm^3",
                "electrode area": 0.07,
                "y axis": "Current",
            }
        ),
    )

    assert calls
    assert calls[0].get("y axis") == "Current"
