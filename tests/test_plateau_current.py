import numpy as np
import pytest
import matplotlib.pyplot as plt


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

    assert result.table.loc[0, "formula mode"] == "normalized"
    assert result.table.loc[0, "kobs"] == pytest.approx(expected)


def test_plateau_current_slope_normalized_equation_manual(ecat_module):
    ic = 5e-5
    slope = 2e-5
    T = 298
    expected = (0.446 * abs(ic) / abs(slope) * np.sqrt(ecat_module.F / (ecat_module.R * T))) ** 2

    result = ecat_module.plateau_current(
        [],
        _opts(ic=ic, **{"ip0 sqrt scan rate slope": slope, "temperature": T}),
    )

    assert result.table.loc[0, "formula mode"] == "slope normalized"
    assert result.table.loc[0, "kobs"] == pytest.approx(expected)


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

    assert result.table.loc[0, "formula mode"] == "direct"
    assert result.table.loc[0, "kobs"] == pytest.approx(expected)


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
    assert method_result.table.loc[0, "formula mode"] == "normalized"


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
    assert cv.calls[-1]["guess potential"] == pytest.approx(-0.7)


def test_plateau_single_cv_warns_but_returns(ecat_module):
    cv = FakePeakCV("cat", current=-2e-5)

    with pytest.warns(UserWarning, match="scan-rate independence cannot be tested"):
        result = ecat_module.plateau_current(
            cv,
            _opts(
                ip0=1e-5,
                **{"ip0 scan rate": 0.1, "validate plateau": True, "require plateau": True},
            ),
        )

    assert result.table.loc[0, "valid plateau"] == True
    assert "scan-rate independence cannot be tested" in result.table.loc[0, "plateau warning"]


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
        "reference multiplot",
        "peak:ref slow",
        "peak:ref fast",
        "ip0 sqrt fit",
        "catalytic multiplot",
        "peak:cat slow",
        "peak:cat fast",
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
        vertical_segments = [
            np.asarray(segment)
            for fig_num in plt.get_fignums()
            for ax in plt.figure(fig_num).axes
            for collection in ax.collections
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
        assert min(vertical_lengths) < 0.02
    finally:
        plt.close("all")
