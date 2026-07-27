import numpy as np
import pandas as pd
import pytest
import matplotlib.pyplot as plt
import warnings


class _PlateauDummyCV:
    def __init__(self, name, scan_rate, current):
        self.name = name
        self.scan_rate = scan_rate
        self.temperature = 298
        self.electrode_area = 0
        self._current = current
        self.peak_current_calls = []

    def peak_current(self, options):
        self.peak_current_calls.append(dict(options or {}))
        return {"ip": self._current, "tangent line": None, "tangent start": None}


def _synthetic_fowa_cv(ecat_module, blank_echem_factory, name, scale):
    potential = np.linspace(-0.2, 0.25, 120)
    current = (
        1e-6 * scale / (1 + np.exp(-30 * (potential - 0.02)))
        + 1e-7 * potential
    )
    data = pd.DataFrame(
        {
            ("raw", "Potential (V)"): potential,
            ("raw", "Current (A)"): current,
        }
    )

    obj = blank_echem_factory(ecat_module.cv)
    obj.manual_init(name, data, options={})
    return obj


def _analysis_table(result):
    return result.table


def _synthetic_multisegment_fowa_cv(ecat_module, blank_echem_factory, name, scale):
    forward_potential = np.linspace(-0.2, 0.25, 80)
    reverse_potential = np.linspace(0.25, -0.2, 80)
    potential = np.concatenate([forward_potential, reverse_potential])
    forward_current = (
        1e-6 * scale / (1 + np.exp(-30 * (forward_potential - 0.02)))
        + 1e-7 * forward_potential
    )
    reverse_current = (
        0.35e-6 * scale / (1 + np.exp(-25 * (reverse_potential - 0.01)))
        + 0.5e-7 * reverse_potential
    )
    current = np.concatenate([forward_current, reverse_current])
    data = pd.DataFrame(
        {
            ("raw", "Potential (V)"): potential,
            ("raw", "Current (A)"): current,
        }
    )

    obj = blank_echem_factory(ecat_module.cv)
    obj.manual_init(name, data, options={})
    return obj


def _synthetic_reversible_reference_cv(ecat_module, blank_echem_factory, name):
    forward_potential = np.linspace(-0.2, 0.25, 120)
    reverse_potential = np.linspace(0.249, -0.2, 120)
    potential = np.concatenate([forward_potential, reverse_potential])
    forward_current = (
        1.0e-6 * np.exp(-((forward_potential - 0.08) / 0.045) ** 2)
        + 0.08e-6 * forward_potential
    )
    reverse_current = (
        -0.8e-6 * np.exp(-((reverse_potential - 0.01) / 0.045) ** 2)
        + 0.08e-6 * reverse_potential
    )
    data = pd.DataFrame(
        {
            ("raw", "Potential (V)"): potential,
            ("raw", "Current (A)"): np.concatenate([forward_current, reverse_current]),
        }
    )

    obj = blank_echem_factory(ecat_module.cv)
    obj.manual_init(name, data, options={})
    return obj


def test_plateau_current_uses_peak_current_without_fowa_only_options(ecat_module):
    cvs = [
        _PlateauDummyCV("cat low", 0.1, -1.0e-5),
        _PlateauDummyCV("cat high", 1.0, -1.1e-5),
    ]

    result = ecat_module.plateau_current(
        cvs,
        {
            "plot": False,
            "plot all": False,
            "print": False,
            "exact potential": -1.8,
            "require plateau": False,
            "validate plateau": False,
            "formula mode": "normalized",
            "ip0": 1.0e-6,
            "ip0 scan rate": 0.1,
        },
    )

    assert result.summary["ilim source"] == "peak_current"
    assert result.summary["formula mode"] == "normalized"
    assert result.diagnostics["plateau details"]["ilim source"].iloc[0] == "peak_current"


def test_plateau_current_passes_per_cv_potential_options(ecat_module):
    cvs = [
        _PlateauDummyCV("cat low", 0.1, -1.0e-5),
        _PlateauDummyCV("cat high", 1.0, -1.1e-5),
    ]

    ecat_module.plateau_current(
        cvs,
        {
            "plot": False,
            "plot all": False,
            "print": False,
            "guess potentials": [-0.11, -0.22],
            "tangent potentials": [-0.31, -0.42],
            "require plateau": False,
            "validate plateau": False,
            "formula mode": "normalized",
            "ip0": 1.0e-6,
            "ip0 scan rate": 0.1,
        },
    )

    assert [cv.peak_current_calls[0]["guess potential"] for cv in cvs] == pytest.approx([-0.11, -0.22])
    assert [cv.peak_current_calls[0]["tangent potential"] for cv in cvs] == pytest.approx([-0.31, -0.42])


def test_fowa_table_column_labels_preserve_electrochemical_notation(ecat_module):
    labels = {
        "ip0 source": "ip0 Source",
        "reference Ep": "Reference Ep",
        "redox delta E": "Redox Delta E",
        "catalytic Ecat/2": "Catalytic Ecat/2",
        "Ecat/2 shift": "Ecat/2 - E1/2",
        "FOWA fit": "FOWA Fit",
        "kobs": "kobs",
        "R2": "R2",
    }

    for raw, expected in labels.items():
        assert ecat_module.pretty_table_column_label(raw) == expected


def test_fowa_pretty_table_headers_format_electrochemical_notation(ecat_module):
    labels = {
        "ip0 Source": "i<sub>p</sub><sup>0</sup> Source",
        "Reference Ep": "Reference E<sub>p</sub>",
        "Catalytic Ecat/2": "Catalytic E<sub>cat/2</sub>",
        "Ecat/2 - E1/2": "E<sub>cat/2</sub> - E<sub>1/2</sub>",
        "R2": "R<sup>2</sup>",
        "kobs": "k<sub>obs</sub>",
    }

    for raw, expected in labels.items():
        assert ecat_module._pretty_table_header_html_label(raw) == expected


def test_fowa_table_includes_fit_range_when_ranges_differ(
    ecat_module,
    blank_echem_factory,
    monkeypatch,
):
    cvs = [
        _synthetic_fowa_cv(ecat_module, blank_echem_factory, "100mVs_cat_run01", 1.0),
        _synthetic_fowa_cv(ecat_module, blank_echem_factory, "100mVs_cat_run02", 1.2),
    ]
    monkeypatch.setattr(
        ecat_module,
        "build_object_table",
        lambda object_list, options=None: (
            pd.DataFrame({"Name": [obj.name for obj in object_list]}),
            {},
        ),
    )

    with pytest.warns(UserWarning, match="non-positive slope"):
        table = _analysis_table(ecat_module.fowa(
            cvs,
            {
                "plot": False,
                "print": False,
                "non-catalytic current": 1e-6,
                "redox potential": 0.0,
                "redox mode": "manual",
                "background correction": None,
                "fit basis": ["x", "y"],
                "fit range": [[0.1, 0.3], [0.3, 0.5]],
                "min fit points": 5,
                "min r2": 0,
                "ecat shift warning threshold": False,
            },
        ))

    assert "Fit Basis" in table.columns
    assert table["Fit Basis"].tolist() == ["x", "y"]
    assert "Fit Range" in table.columns
    assert table["Fit Range"].tolist() == ["[0.1, 0.3]", "[0.3, 0.5]"]


def test_fowa_status_summarizes_all_issues_and_hides_warning_details(
    ecat_module,
    blank_echem_factory,
    monkeypatch,
):
    cv_obj = _synthetic_fowa_cv(ecat_module, blank_echem_factory, "100mVs_cat_run01", 1.0)
    monkeypatch.setattr(
        ecat_module,
        "build_object_table",
        lambda object_list, options=None: (
            pd.DataFrame({"Name": [obj.name for obj in object_list]}),
            {},
        ),
    )

    with pytest.warns(UserWarning):
        table = _analysis_table(ecat_module.fowa(
            [cv_obj],
            {
                "plot": False,
                "print": False,
                "non-catalytic current": 1e-6,
                "redox potential": 0.0,
                "redox mode": "manual",
                "background correction": None,
                "fit basis": "x",
                "fit range": [0.1, 0.3],
                "min fit points": 200,
                "min r2": 1.01,
                "ecat shift warning threshold": False,
            },
        ))

    status = table["Status"].iloc[0]
    assert "fit points < threshold" in status
    assert "fit R2 < threshold" in status
    assert "nonpositive slope" in status
    assert "Warnings" not in table.columns
    assert "Warning Details" not in table.columns

    full = table.attrs["full_results_df"]
    assert "Warning Details" in full.columns
    assert "fit for" in full["Warning Details"].iloc[0]
    assert table.attrs["warnings"][cv_obj.name] == full["Warning Details"].iloc[0]


def test_fowa_warnings_option_suppresses_python_warnings_but_keeps_status(
    ecat_module,
    blank_echem_factory,
    monkeypatch,
):
    cv_obj = _synthetic_fowa_cv(ecat_module, blank_echem_factory, "100mVs_cat_run01", 1.0)
    monkeypatch.setattr(
        ecat_module,
        "build_object_table",
        lambda object_list, options=None: (
            pd.DataFrame({"Name": [obj.name for obj in object_list]}),
            {},
        ),
    )

    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        table = _analysis_table(ecat_module.fowa(
            [cv_obj],
            {
                "plot": False,
                "print": False,
                "warnings": False,
                "non-catalytic current": 1e-6,
                "redox potential": 0.0,
                "redox mode": "manual",
                "background correction": None,
                "fit basis": "x",
                "fit range": [0.1, 0.3],
                "min fit points": 200,
                "min r2": 1.01,
                "ecat shift warning threshold": False,
            },
        ))

    assert captured == []
    assert "multiple issues" not in table["Status"].iloc[0]
    assert "fit points < threshold" in table["Status"].iloc[0]
    assert "Warning Details" in table.attrs["full_results_df"].columns


def test_fowa_result_tables_include_unit_metadata(
    ecat_module,
    blank_echem_factory,
    monkeypatch,
):
    cv_obj = _synthetic_fowa_cv(ecat_module, blank_echem_factory, "100mVs_cat_run01", 1.0)
    monkeypatch.setattr(
        ecat_module,
        "build_object_table",
        lambda object_list, options=None: (
            pd.DataFrame({"Name": [obj.name for obj in object_list]}),
            {},
        ),
    )

    with pytest.warns(UserWarning, match="non-positive slope"):
        table = _analysis_table(ecat_module.fowa(
            [cv_obj],
            {
                "plot": False,
                "print": False,
                "non-catalytic current": 1e-6,
                "redox potential": 0.0,
                "redox mode": "manual",
                "background correction": None,
                "fit basis": "x",
                "fit range": [0.1, 0.3],
                "min fit points": 5,
                "min r2": 0,
                "ecat shift warning threshold": False,
            },
        ))

    assert table.attrs["units"]["kobs"] == "s^-1"
    assert table.attrs["units"]["ip0"] == "A"
    assert table.attrs["full_results_df"].attrs["units"]["kobs"] == "s^-1"


def test_fowa_rejects_unknown_option_with_suggestion(ecat_module, blank_echem_factory):
    cv_obj = _synthetic_fowa_cv(
        ecat_module,
        blank_echem_factory,
        "100mVs_cat_run01",
        1.0,
    )

    with pytest.raises(ecat_module.OptionError, match="fit range"):
        ecat_module.fowa([cv_obj], {"plot": False, "fit rnage": [0.1, 0.5]})


def test_fowa_redox_mode_defaults_to_half_wave(ecat_module):
    assert ecat_module.FOWAOptions.from_options({}).redox_mode == "half wave"


def test_fowa_x_axis_label_reflects_half_wave_reference_and_n(ecat_module):
    label = ecat_module._format_fowa_x_axis_label(
        [{"redox mode": "half wave", "catalyst electrons": 2}],
        {"catalyst electrons": 2, "turnover electrons": 4},
    )

    assert r"\frac{nF}{RT}" in label
    assert r"E_{1/2}" in label
    assert "turn" not in label


def test_fowa_x_axis_label_reflects_manual_and_half_peak_references(ecat_module):
    manual_label = ecat_module._format_fowa_x_axis_label(
        [{"redox mode": "manual", "catalyst electrons": 1}],
        {"catalyst electrons": 1},
    )
    half_peak_label = ecat_module._format_fowa_x_axis_label(
        [{"redox mode": "half peak", "catalyst electrons": 1}],
        {"catalyst electrons": 1},
    )
    mixed_label = ecat_module._format_fowa_x_axis_label(
        [
            {"redox mode": "half wave", "catalyst electrons": 1},
            {"redox mode": "manual", "catalyst electrons": 1},
        ],
        {"catalyst electrons": 1},
    )

    assert r"E_{\mathrm{redox}}" in manual_label
    assert r"E_{p/2}" in half_peak_label
    assert r"E_{\mathrm{ref}}" in mixed_label


def test_fowa_plot_xlabel_uses_dynamic_reference_label(
    ecat_module,
    blank_echem_factory,
    monkeypatch,
):
    cv_obj = _synthetic_fowa_cv(
        ecat_module,
        blank_echem_factory,
        "100mVs_cat_run01",
        1.0,
    )
    monkeypatch.setattr(
        ecat_module,
        "build_object_table",
        lambda object_list, options=None: (
            pd.DataFrame({"Name": [obj.name for obj in object_list]}),
            {},
        ),
    )

    plt.close("all")
    with pytest.warns(UserWarning, match="non-positive slope"):
        ecat_module.fowa(
            [cv_obj],
            {
                "plot": True,
                "print": False,
                "non-catalytic current": 1e-6,
                "redox potential": 0.0,
                "redox mode": "manual",
                "background correction": None,
                "fit range": [0.1, 0.3],
                "min fit points": 5,
                "min r2": 0,
                "ecat shift warning threshold": False,
                "n_cat": 2,
                "n_turn": 4,
            },
        )

    xlabel = plt.gca().get_xlabel()
    assert r"\frac{nF}{RT}" in xlabel
    assert r"E_{\mathrm{redox}}" in xlabel
    assert "turn" not in xlabel
    plt.close("all")


@pytest.mark.parametrize("redox_mode", [None, "half wave", "half peak"])
def test_fowa_options_require_manual_mode_for_redox_potential(ecat_module, redox_mode):
    options = {"redox potential": -1.46}
    if redox_mode is not None:
        options["redox mode"] = redox_mode

    with pytest.raises(
        ecat_module.OptionError,
        match="Use 'redox mode': 'manual' when providing 'redox potential'",
    ):
        ecat_module.FOWAOptions.from_options(options)


def test_fowa_options_allow_manual_redox_potential(ecat_module):
    options = ecat_module.FOWAOptions.from_options(
        {"redox mode": "manual", "redox potential": -1.46}
    )

    assert options.redox_mode == "manual"
    assert options.redox_potential == pytest.approx(-1.46)


def test_plateau_current_options_inherit_redox_potential_conflict(ecat_module):
    with pytest.raises(
        ecat_module.OptionError,
        match="Use 'redox mode': 'manual' when providing 'redox potential'",
    ):
        ecat_module.PlateauCurrentOptions.from_options({"redox potential": -1.46})


def test_fowa_public_call_rejects_redox_potential_without_manual_mode(
    ecat_module,
    blank_echem_factory,
):
    cv_obj = _synthetic_fowa_cv(
        ecat_module,
        blank_echem_factory,
        "100mVs_cat_run01",
        1.0,
    )

    with pytest.raises(
        ecat_module.OptionError,
        match="Use 'redox mode': 'manual' when providing 'redox potential'",
    ):
        ecat_module.fowa(
            [cv_obj],
            {
                "plot": False,
                "print": False,
                "non-catalytic current": 1e-6,
                "redox potential": -1.46,
            },
        )


def test_fowa_accepts_manual_wave_range(
    ecat_module,
    blank_echem_factory,
    monkeypatch,
):
    cv_obj = _synthetic_fowa_cv(
        ecat_module,
        blank_echem_factory,
        "100mVs_cat_run01",
        1.0,
    )

    def fail_auto_wave(*args, **kwargs):
        raise AssertionError("manual wave range should bypass automatic wave detection")

    monkeypatch.setattr(ecat_module, "_auto_fowa_wave_bounds", fail_auto_wave)
    monkeypatch.setattr(
        ecat_module,
        "build_object_table",
        lambda object_list, options=None: (
            pd.DataFrame({"Name": [obj.name for obj in object_list]}),
            {},
        ),
    )

    with pytest.warns(UserWarning, match="non-positive slope"):
        table = _analysis_table(ecat_module.fowa(
            [cv_obj],
            {
                "plot": False,
                "print": False,
                "non-catalytic current": 1e-6,
                "redox potential": 0.0,
                "redox mode": "manual",
                "background correction": None,
                "wave range": [-0.05, 0.08],
                "colorbar height": 0.42,
                "fit range": [0.1, 0.5],
                "min fit points": 5,
                "min r2": 0,
                "ecat shift warning threshold": False,
            },
    ))

    full = table.attrs["full_results_df"]
    potential = cv_obj.x()
    selected = potential[(potential >= -0.05) & (potential <= 0.08)]
    expected = f"[{selected.iloc[0]:.6g}, {selected.iloc[-1]:.6g}]"
    assert full.loc[0, "Wave Range"] == expected
    assert table.attrs["shared_summary"]["Wave Range"] == expected


def test_fowa_accepts_plural_per_cv_wave_ranges(
    ecat_module,
    blank_echem_factory,
    monkeypatch,
):
    cvs = [
        _synthetic_fowa_cv(ecat_module, blank_echem_factory, "100mVs_cat_run01", 1.0),
        _synthetic_fowa_cv(ecat_module, blank_echem_factory, "100mVs_cat_run02", 1.2),
    ]
    wave_ranges = [[-0.05, 0.08], [-0.02, 0.1]]

    def fail_auto_wave(*args, **kwargs):
        raise AssertionError("per-CV manual wave ranges should bypass automatic wave detection")

    monkeypatch.setattr(ecat_module, "_auto_fowa_wave_bounds", fail_auto_wave)
    monkeypatch.setattr(
        ecat_module,
        "build_object_table",
        lambda object_list, options=None: (
            pd.DataFrame({"Name": [obj.name for obj in object_list]}),
            {},
        ),
    )

    with pytest.warns(UserWarning, match="non-positive slope"):
        table = _analysis_table(ecat_module.fowa(
            cvs,
            {
                "plot": False,
                "print": False,
                "non-catalytic current": 1e-6,
                "redox mode": "manual",
                "redox potentials": [0.0, 0.01],
                "background correction": None,
                "wave ranges": wave_ranges,
                "fit range": [0.1, 0.5],
                "min fit points": 5,
                "min r2": 0,
                "ecat shift warning threshold": False,
            },
        ))

    full = table.attrs["full_results_df"]
    expected = []
    for cv_obj, (lo, hi) in zip(cvs, wave_ranges):
        potential = cv_obj.x()
        selected = potential[(potential >= lo) & (potential <= hi)]
        expected.append(f"[{selected.iloc[0]:.6g}, {selected.iloc[-1]:.6g}]")

    assert full["Wave Range"].tolist() == expected


def test_fowa_fit_false_returns_transformed_data_without_regression(
    ecat_module,
    blank_echem_factory,
    monkeypatch,
):
    cv_obj = _synthetic_fowa_cv(
        ecat_module,
        blank_echem_factory,
        "100mVs_cat_run01",
        1.0,
    )
    monkeypatch.setattr(
        ecat_module,
        "build_object_table",
        lambda object_list, options=None: (
            pd.DataFrame({"Name": [obj.name for obj in object_list]}),
            {},
        ),
    )

    result = ecat_module.fowa(
        [cv_obj],
        {
            "plot": True,
            "print": False,
            "fit": False,
            "non-catalytic current": 1e-6,
            "redox mode": "manual",
            "redox potential": 0.0,
            "background correction": None,
            "wave range": [-0.05, 0.08],
            "ecat shift warning threshold": False,
        },
    )

    full = result.diagnostics["full results"]
    plot_data = result.diagnostics["plot data"][0]

    assert full.loc[0, "Status"] == "not fit"
    assert pd.isna(full.loc[0, "Slope"])
    assert pd.isna(full.loc[0, "Intercept"])
    assert pd.isna(full.loc[0, "R2"])
    assert pd.isna(full.loc[0, "kobs"])
    assert plot_data["x fowa"].size > 0
    assert plot_data["y fowa"].size > 0
    assert plot_data["x fit"].size == 0
    assert plot_data["y fit"].size == 0
    assert plt.gca().lines
    assert all(line.get_linestyle() != "--" for line in plt.gca().lines)


def test_fowa_fit_true_warns_and_skips_only_cv_with_unusable_fit_region(
    ecat_module,
    blank_echem_factory,
    monkeypatch,
):
    cvs = [
        _synthetic_fowa_cv(ecat_module, blank_echem_factory, "100mVs_cat_run01", 1.0),
        _synthetic_fowa_cv(ecat_module, blank_echem_factory, "100mVs_cat_run02", 1.2),
        _synthetic_fowa_cv(ecat_module, blank_echem_factory, "100mVs_cat_run03", 1.4),
    ]
    original_resolver = ecat_module._resolve_fowa_fit_mask
    call_count = 0

    def second_fit_has_no_points(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 2:
            x_fowa = np.asarray(kwargs["x_fowa"], dtype=float)
            return np.zeros_like(x_fowa, dtype=bool), "x_fowa", {}
        return original_resolver(*args, **kwargs)

    monkeypatch.setattr(ecat_module, "_resolve_fowa_fit_mask", second_fit_has_no_points)
    monkeypatch.setattr(
        ecat_module,
        "build_object_table",
        lambda object_list, options=None: (
            pd.DataFrame({"Name": [obj.name for obj in object_list]}),
            {},
        ),
    )

    with pytest.warns(UserWarning, match="Only 0 points.*fit skipped"):
        result = ecat_module.fowa(
            cvs,
            {
                "plot": False,
                "print": False,
                "fit": True,
                "non-catalytic current": 1e-6,
                "redox mode": "manual",
                "redox potentials": [0.0, 0.01, 0.02],
                "background correction": None,
                "wave range": [-0.05, 0.08],
                "fit range": [0.1, 0.5],
                "min fit points": 5,
                "min r2": 0,
                "ecat shift warning threshold": False,
            },
        )

    full = result.diagnostics["full results"]

    assert len(full) == 3
    assert full.loc[1, "Status"] == "fit skipped"
    assert pd.isna(full.loc[1, "kobs"])
    assert np.isfinite(full.loc[0, "kobs"])
    assert np.isfinite(full.loc[2, "kobs"])
    assert result.diagnostics["plot data"][1]["x fowa"].size > 0
    assert result.diagnostics["plot data"][1]["x fit"].size == 0


def test_fowa_manual_redox_still_reports_ecat_half_shift(
    ecat_module,
    blank_echem_factory,
    monkeypatch,
):
    cv_obj = _synthetic_fowa_cv(
        ecat_module,
        blank_echem_factory,
        "100mVs_cat_run01",
        1.0,
    )
    monkeypatch.setattr(
        ecat_module,
        "build_object_table",
        lambda object_list, options=None: (
            pd.DataFrame({"Name": [obj.name for obj in object_list]}),
            {},
        ),
    )
    monkeypatch.setattr(
        ecat_module,
        "_resolve_catalytic_half_peak_for_shift_check",
        lambda **kwargs: (0.12, 0.02),
    )

    with pytest.warns(UserWarning, match="non-positive slope"):
        table = _analysis_table(ecat_module.fowa(
            [cv_obj],
            {
                "plot": False,
                "print": False,
                "non-catalytic current": 1e-6,
                "redox potential": 0.03,
                "redox mode": "manual",
                "background correction": None,
                "fit range": [0.1, 0.5],
                "min fit points": 5,
                "min r2": 0,
                "ecat shift warning threshold": False,
            },
        ))

    full = table.attrs["full_results_df"]
    assert full.loc[0, "Catalytic Ecat/2"] == pytest.approx(0.12)
    assert full.loc[0, "Ecat/2 - E1/2"] == pytest.approx(0.09)


def test_fowa_accepts_colorbar_height_alias(ecat_module):
    options = ecat_module.FOWAOptions.from_options(
        {
            "plot": False,
            "colorbar height": 0.42,
        }
    )

    assert options.colorbar_height_scale == pytest.approx(0.42)
    assert options.to_options_dict()["colorbar height scale"] == pytest.approx(0.42)


def test_fowa_half_peak_redox_ignores_scatter_plot_options(ecat_module):
    class DummyCV:
        def __init__(self, name):
            self.name = name

        def peak_potential(self, options):
            assert "legend" not in options
            return {"Ep": -1.4, "index": 0}

        def half_peak_potential(self, options):
            assert "legend" not in options
            assert options.get("exact potential") == pytest.approx(-1.4)
            assert "guess potential" not in options or options.get("guess potential") is None
            return {"Ep/2": -1.45, "Δ(Ep - Ep/2)": 0.05}

    redox_potential, source, delta, mode = ecat_module._resolve_fowa_redox_potential(
        cat_cv=DummyCV("cat"),
        ref_cv=DummyCV("ref"),
        options={"redox mode": "half peak"},
        internal_options={
            "plot": False,
            "print": False,
            "segment": 1,
            "guess potential": -1.5,
            "legend": True,
            "fit color": "tab:red",
        },
    )

    assert redox_potential == pytest.approx(-1.45)
    assert source == "cat"
    assert delta == pytest.approx(0.05)
    assert mode == "half peak"


def test_fowa_prints_by_default_after_multiplot_defaults(
    ecat_module,
    blank_echem_factory,
    monkeypatch,
    capsys,
):
    cv_obj = _synthetic_fowa_cv(
        ecat_module,
        blank_echem_factory,
        "100mVs_cat_run01",
        1.0,
    )
    monkeypatch.setattr(
        ecat_module,
        "build_object_table",
        lambda object_list, options=None: (
            pd.DataFrame({"Name": [obj.name for obj in object_list]}),
            {},
        ),
    )

    with pytest.warns(UserWarning, match="non-positive slope"):
        ecat_module.fowa(
            [cv_obj],
            {
                "plot": False,
                "non-catalytic current": 1e-6,
                "redox potential": 0.0,
                "redox mode": "manual",
                "background correction": None,
                "fit range": [0.1, 0.5],
                "min fit points": 5,
                "min r2": 0,
                "ecat shift warning threshold": False,
            },
        )

    captured = capsys.readouterr()
    assert "### FOWA Summary ###" in captured.out


def test_fowa_summary_display_table_is_vertical(ecat_module):
    table = ecat_module._fowa_summary_display_table(
        {
            "Reference CV": "blank_cv",
            "ip0 Source": "manual (1e-06)",
            "Segment": 1,
        }
    )

    assert list(table.columns) == ["Field", "Value"]
    assert table.to_dict("records") == [
        {"Field": "Reference CV", "Value": "blank_cv"},
        {"Field": "ip0 Source", "Value": "manual (1e-06)"},
        {"Field": "Segment", "Value": "1"},
    ]


def test_fowa_summary_pretty_print_uses_styled_dataframe(ecat_module, monkeypatch):
    displayed = {}

    def capture_display(obj):
        displayed["object"] = obj

    monkeypatch.setattr(ecat_module, "display", capture_display)

    table = ecat_module._display_fowa_summary_table(
        {
            "ip0 Source": "manual (1e-06)",
            "Reference Ep": "0.1",
            "R2": "0.99",
        },
        {"pretty print": True},
    )

    assert list(table.columns) == ["Field", "Value"]
    rendered = displayed["object"].to_html()
    assert "i<sub>p</sub><sup>0</sup> Source" in rendered
    assert "Reference E<sub>p</sub>" in rendered
    assert "R<sup>2</sup>" in rendered


def test_fowa_results_display_respects_sig_figs_and_formats_kobs_scientific(
    ecat_module,
    monkeypatch,
):
    displayed = {}

    def capture_display(obj):
        displayed["object"] = obj

    monkeypatch.setattr(ecat_module, "display", capture_display)

    table = pd.DataFrame(
        {
            "Name": ["cat"],
            "Redox Potential": [0.123456],
            "R2": [0.987654],
            "kobs": [12345.6789],
        }
    )

    returned = ecat_module._display_fowa_results_table(
        table,
        {"pretty print": True, "sig figs": 3},
    )

    assert returned is table
    rendered = displayed["object"].to_html()
    assert "0.123" in rendered
    assert "0.988" in rendered
    assert "1.23e+04" in rendered
    assert "12345.6789" not in rendered
    assert "k<sub>obs</sub>" in rendered


def test_fowa_summary_plain_print_uses_dataframe_text(ecat_module, monkeypatch, capsys):
    monkeypatch.setattr(ecat_module, "display", None)

    table = ecat_module._display_fowa_summary_table(
        {
            "Segment": 1,
            "Fit Range": "[0.1, 0.5]",
        },
        {"pretty print": False},
    )

    output = capsys.readouterr().out
    assert "Field" in output
    assert "Value" in output
    assert "Segment" in output
    assert "Fit Range" in output
    assert table.to_dict("records") == [
        {"Field": "Segment", "Value": "1"},
        {"Field": "Fit Range", "Value": "[0.1, 0.5]"},
    ]


def test_fowa_kobs_equation_can_skip_resolved_n_substitution(ecat_module, monkeypatch, capsys):
    monkeypatch.setattr(ecat_module, "display", None)

    ecat_module._display_fowa_kobs_equation(
        {
            "catalyst electrons": 2,
            "turnover electrons": 3,
            "sigma": 2,
        },
        resolved=False,
    )

    output = capsys.readouterr().out
    assert "k_obs = (m * 0.4463 * n / n'^sigma)^2" in output
    assert "n = 2 (n_cat, catalyst redox-wave electron count)" in output
    assert "n' = 3 (n_turn, turnover electron count)" in output
    assert "m * 0.4463 * 2 / 3^2" not in output


def test_fowa_accepts_option_dataclass(ecat_module, blank_echem_factory, monkeypatch):
    cv_obj = _synthetic_fowa_cv(
        ecat_module,
        blank_echem_factory,
        "100mVs_cat_run01",
        1.0,
    )
    monkeypatch.setattr(
        ecat_module,
        "build_object_table",
        lambda object_list, options=None: (
            pd.DataFrame({"Name": [obj.name for obj in object_list]}),
            {},
        ),
    )
    options = ecat_module.FOWAOptions.from_options(
        {
            "plot": False,
            "print": False,
            "non-catalytic current": 1e-6,
            "redox potential": 0.0,
            "redox mode": "manual",
            "background correction": None,
            "fit range": [0.1, 0.5],
            "min fit points": 5,
            "min r2": 0,
            "ecat shift warning threshold": False,
        }
    )

    with pytest.warns(UserWarning, match="non-positive slope"):
        table = _analysis_table(ecat_module.fowa([cv_obj], options))

    assert "FOWA Fit" in table.attrs["shared_summary"]
    assert "Slope" in table.attrs["full_results_df"].columns


def test_fowa_table_combines_line_and_wave_range_columns(
    ecat_module,
    blank_echem_factory,
    monkeypatch,
):
    cv_obj = _synthetic_fowa_cv(
        ecat_module,
        blank_echem_factory,
        "100mVs_cat_run01",
        1.0,
    )
    monkeypatch.setattr(
        ecat_module,
        "build_object_table",
        lambda object_list, options=None: (
            pd.DataFrame({"Name": [obj.name for obj in object_list]}),
            {},
        ),
    )

    with pytest.warns(UserWarning, match="non-positive slope"):
        table = _analysis_table(ecat_module.fowa(
            [cv_obj],
            {
                "plot": False,
                "print": False,
                "non-catalytic current": 1e-6,
                "redox potential": 0.0,
                "redox mode": "manual",
                "background correction": "tangent",
                "tangent potential": -0.1,
                "fit range": [0.1, 0.5],
                "min fit points": 5,
                "min r2": 0,
                "ecat shift warning threshold": False,
            },
        ))

    shared_summary = table.attrs["shared_summary"]
    assert "Background Tangent" in shared_summary
    assert "Wave Range" in shared_summary
    assert "FOWA Fit" in shared_summary
    assert shared_summary["Background Tangent"].startswith("y = ")
    assert shared_summary["Wave Range"].startswith("[")
    assert shared_summary["FOWA Fit"].startswith("y = ")
    assert "Fit Points" in shared_summary
    assert "N Fit Points" not in table.columns
    assert "TOFmax" not in table.columns
    assert "Background Tangent Potential" not in table.columns
    assert "Background Slope" not in table.columns
    assert "Background Intercept" not in table.columns
    assert "Wave Start" not in table.columns
    assert "Wave End" not in table.columns
    assert "Slope" not in table.columns
    assert "Intercept" not in table.columns

    full_results = table.attrs["full_results_df"]
    assert "Background Slope" in full_results.columns
    assert "Slope" in full_results.columns
    assert "TOFmax" in full_results.columns


def test_fowa_shared_analysis_columns_move_to_summary(
    ecat_module,
    blank_echem_factory,
    monkeypatch,
):
    cvs = [
        _synthetic_fowa_cv(ecat_module, blank_echem_factory, "100mVs_cat_run01", 1.0),
        _synthetic_fowa_cv(ecat_module, blank_echem_factory, "100mVs_cat_run02", 1.2),
    ]
    monkeypatch.setattr(
        ecat_module,
        "build_object_table",
        lambda object_list, options=None: (
            pd.DataFrame({"Name": [obj.name for obj in object_list]}),
            {},
        ),
    )

    with pytest.warns(UserWarning, match="non-positive slope"):
        table = _analysis_table(ecat_module.fowa(
            cvs,
            {
                "plot": False,
                "print": False,
                "non-catalytic current": 1e-6,
                "redox potential": 0.0,
                "redox mode": "manual",
                "background correction": None,
                "fit basis": "y",
                "fit range": [0.1, 0.5],
                "min fit points": 5,
                "min r2": 0,
                "ecat shift warning threshold": False,
            },
        ))

    shared_summary = table.attrs["shared_summary"]
    assert "Fit Basis" not in table.columns
    assert "Fit Range" not in table.columns
    assert "Redox Mode" not in table.columns
    assert shared_summary["Fit Basis"] == "y"
    assert shared_summary["Fit Range"] == "[0.1, 0.5]"
    assert shared_summary["Redox Mode"] == "manual"


def test_fowa_table_uses_operation_order_for_analysis_columns(
    ecat_module,
    blank_echem_factory,
    monkeypatch,
):
    cv_obj = _synthetic_fowa_cv(
        ecat_module,
        blank_echem_factory,
        "100mVs_cat_run01",
        1.0,
    )
    monkeypatch.setattr(
        ecat_module,
        "build_object_table",
        lambda object_list, options=None: (
            pd.DataFrame({"Name": [obj.name for obj in object_list]}),
            {},
        ),
    )

    with pytest.warns(UserWarning, match="non-positive slope"):
        table = _analysis_table(ecat_module.fowa(
            [cv_obj],
            {
                "plot": False,
                "print": False,
                "non-catalytic current": 1e-6,
                "redox potential": 0.0,
                "redox mode": "manual",
                "background correction": "tangent",
                "tangent potential": -0.1,
                "fit range": [0.1, 0.5],
                "min fit points": 5,
                "min r2": 0,
                "ecat shift warning threshold": 999,
            },
        ))

    expected_order = [
        "Name",
        "Background Tangent",
        "R2",
        "Status",
        "kobs",
    ]

    visible_expected = [col for col in expected_order if col in table.columns]
    assert list(table.columns) == visible_expected


def test_fowa_applies_per_cv_tangent_and_redox_potentials(
    ecat_module,
    blank_echem_factory,
    monkeypatch,
):
    cvs = [
        _synthetic_fowa_cv(ecat_module, blank_echem_factory, "100mVs_cat_run01", 1.0),
        _synthetic_fowa_cv(ecat_module, blank_echem_factory, "100mVs_cat_run02", 1.2),
    ]
    monkeypatch.setattr(
        ecat_module,
        "build_object_table",
        lambda object_list, options=None: (
            pd.DataFrame({"Name": [obj.name for obj in object_list]}),
            {},
        ),
    )

    with pytest.warns(UserWarning, match="non-positive slope"):
        table = _analysis_table(ecat_module.fowa(
            cvs,
            {
                "plot": False,
                "print": False,
                "non-catalytic current": 1e-6,
                "redox mode": "manual",
                "redox potentials": [0.0, 0.01],
                "background correction": "tangent",
                "tangent potentials": [-0.1, -0.05],
                "fit range": [0.1, 0.5],
                "min fit points": 5,
                "min r2": 0,
                "ecat shift warning threshold": False,
            },
        ))

    full_results = table.attrs["full_results_df"]
    assert full_results["Redox Potential"].tolist() == pytest.approx([0.0, 0.01])
    assert full_results["Background Tangent Potential"].tolist() == pytest.approx([-0.1, -0.05])


def test_fowa_print_accepts_per_cv_manual_ip0_values(
    ecat_module,
    blank_echem_factory,
    monkeypatch,
    capsys,
):
    cvs = [
        _synthetic_fowa_cv(ecat_module, blank_echem_factory, "100mVs_cat_run01", 1.0),
        _synthetic_fowa_cv(ecat_module, blank_echem_factory, "100mVs_cat_run02", 1.2),
    ]
    monkeypatch.setattr(
        ecat_module,
        "build_object_table",
        lambda object_list, options=None: (
            pd.DataFrame({"Name": [obj.name for obj in object_list]}),
            {},
        ),
    )

    with pytest.warns(UserWarning, match="non-positive slope"):
        result = ecat_module.fowa(
            cvs,
            {
                "plot": False,
                "print": True,
                "pretty print": False,
                "non-catalytic current": [1e-6, 2e-6],
                "redox mode": "manual",
                "redox potentials": [0.0, 0.01],
                "background correction": None,
                "fit range": [[0.1, 0.3], [0.3, 0.5]],
                "min fit points": 5,
                "min r2": 0,
                "ecat shift warning threshold": False,
            },
        )

    printed = capsys.readouterr().out
    assert "manual (per CV)" in printed
    full_results = result.table.attrs["full_results_df"]
    assert full_results["ip0"].tolist() == pytest.approx([1e-6, 2e-6])


def test_fowa_transformed_plot_respects_multiplot_legend_toggle(
    ecat_module,
    blank_echem_factory,
    monkeypatch,
):
    cvs = [
        _synthetic_fowa_cv(ecat_module, blank_echem_factory, "100mVs_cat_run01", 1.0),
        _synthetic_fowa_cv(ecat_module, blank_echem_factory, "100mVs_cat_run02", 1.2),
    ]
    monkeypatch.setattr(
        ecat_module,
        "build_object_table",
        lambda object_list, options=None: (
            pd.DataFrame({"Name": [obj.name for obj in object_list]}),
            {},
        ),
    )

    with pytest.warns(UserWarning, match="non-positive slope"):
        ecat_module.fowa(
            cvs,
            {
                "plot": True,
                "plot all": False,
                "legend": False,
                "title": False,
                "print": False,
                "non-catalytic current": 1e-6,
                "redox potential": 0.0,
                "redox mode": "manual",
                "background correction": None,
                "fit range": [0.1, 0.5],
                "min fit points": 5,
                "min r2": 0,
                "ecat shift warning threshold": False,
            },
        )

    fig = plt.gcf()
    ax = plt.gca()
    assert ax.get_legend() is None
    plt.close(fig)


def test_fowa_plot_all_defaults_to_fowa_normalized_current(
    ecat_module,
    blank_echem_factory,
    monkeypatch,
):
    cv_obj = _synthetic_fowa_cv(
        ecat_module,
        blank_echem_factory,
        "100mVs_cat_run01",
        1.0,
    )
    monkeypatch.setattr(
        ecat_module,
        "build_object_table",
        lambda object_list, options=None: (
            pd.DataFrame({"Name": [obj.name for obj in object_list]}),
            {},
        ),
    )

    with pytest.warns(UserWarning, match="non-positive slope"):
        ecat_module.fowa(
            [cv_obj],
            {
                "plot": False,
                "plot all": True,
                "legend": False,
                "title": False,
                "print": False,
                "non-catalytic current": 1e-6,
                "redox potential": 0.0,
                "redox mode": "manual",
                "background correction": None,
                "fit range": [0.1, 0.5],
                "min fit points": 5,
                "min r2": 0,
                "ecat shift warning threshold": False,
            },
        )

    fig = plt.gcf()
    ax = plt.gca()
    np.testing.assert_allclose(ax.lines[0].get_ydata(), cv_obj.y().to_numpy() / 1e-6)
    assert ax.get_ylabel() == "$i / i_p^0$"
    plt.close(fig)


def test_fowa_uses_raw_current_when_input_cv_is_already_normalized(
    ecat_module,
    blank_echem_factory,
):
    cv_obj = _synthetic_fowa_cv(
        ecat_module,
        blank_echem_factory,
        "100mVs_cat_run01",
        1.0,
    )
    normalized_cv = ecat_module.normalize_current(
        cv_obj,
        {"ip0": 2e-6, "print": False},
    )

    common_options = {
        "plot": False,
        "plot all": False,
        "legend": False,
        "title": False,
        "print": False,
        "ip0": 1e-6,
        "redox potential": 0.0,
        "redox mode": "manual",
        "background correction": None,
        "fit range": [0.1, 0.5],
        "min fit points": 5,
        "min r2": 0,
        "ecat shift warning threshold": False,
    }

    with pytest.warns(UserWarning, match="non-positive slope"):
        raw_result = _analysis_table(ecat_module.fowa([cv_obj], common_options))
    with pytest.warns(UserWarning, match="non-positive slope"):
        normalized_result = _analysis_table(ecat_module.fowa([normalized_cv], common_options))

    raw_full = raw_result.attrs["full_results_df"]
    normalized_full = normalized_result.attrs["full_results_df"]
    np.testing.assert_allclose(
        normalized_full["Slope"].to_numpy(),
        raw_full["Slope"].to_numpy(),
    )


def test_fowa_accepts_ip0_alias_for_non_catalytic_current(
    ecat_module,
    blank_echem_factory,
    monkeypatch,
):
    cv_obj = _synthetic_fowa_cv(
        ecat_module,
        blank_echem_factory,
        "100mVs_cat_run01",
        1.0,
    )
    monkeypatch.setattr(
        ecat_module,
        "build_object_table",
        lambda object_list, options=None: (
            pd.DataFrame({"Name": [obj.name for obj in object_list]}),
            {},
        ),
    )

    with pytest.warns(UserWarning, match="non-positive slope"):
        table = _analysis_table(ecat_module.fowa(
            [cv_obj],
            {
                "plot": False,
                "print": False,
                "ip0": 1e-6,
                "redox potential": 0.0,
                "redox mode": "manual",
                "background correction": None,
                "fit range": [0.1, 0.5],
                "min fit points": 5,
                "min r2": 0,
                "ecat shift warning threshold": False,
            },
        ))

    assert "FOWA Fit" in table.attrs["shared_summary"]
    assert "Slope" in table.attrs["full_results_df"].columns


def test_fowa_plot_all_current_diagnostic_y_axis_preserves_current_axis(
    ecat_module,
    blank_echem_factory,
    monkeypatch,
):
    cv_obj = _synthetic_fowa_cv(
        ecat_module,
        blank_echem_factory,
        "100mVs_cat_run01",
        1.0,
    )
    monkeypatch.setattr(
        ecat_module,
        "build_object_table",
        lambda object_list, options=None: (
            pd.DataFrame({"Name": [obj.name for obj in object_list]}),
            {},
        ),
    )

    with pytest.warns(UserWarning, match="non-positive slope"):
        ecat_module.fowa(
            [cv_obj],
            {
                "plot": False,
                "plot all": True,
                "y axis": "Current",
                "legend": False,
                "title": False,
                "print": False,
                "non-catalytic current": 1e-6,
                "redox potential": 0.0,
                "redox mode": "manual",
                "background correction": None,
                "fit range": [0.1, 0.5],
                "min fit points": 5,
                "min r2": 0,
                "ecat shift warning threshold": False,
            },
        )

    fig = plt.gcf()
    ax = plt.gca()
    assert "Current" in ax.get_ylabel()
    plt.close(fig)


def test_fowa_plot_all_first_figure_is_normalized_multiplot_with_overlays(
    ecat_module,
    blank_echem_factory,
    monkeypatch,
):
    cv_obj = _synthetic_fowa_cv(
        ecat_module,
        blank_echem_factory,
        "100mVs_cat_run01",
        1.0,
    )
    monkeypatch.setattr(
        ecat_module,
        "build_object_table",
        lambda object_list, options=None: (
            pd.DataFrame({"Name": [obj.name for obj in object_list]}),
            {},
        ),
    )

    with pytest.warns(UserWarning, match="non-positive slope"):
        ecat_module.fowa(
            [cv_obj],
            {
                "plot": True,
                "plot all": True,
                "legend": False,
                "title": False,
                "print": False,
                "non-catalytic current": 1e-6,
                "redox potential": 0.0,
                "redox mode": "manual",
                "background correction": "tangent",
                "fit range": [0.1, 0.5],
                "min fit points": 5,
                "min r2": 0,
                "ecat shift warning threshold": False,
            },
        )

    figures = [plt.figure(number) for number in plt.get_fignums()]

    assert len(figures) == 2
    raw_ax = figures[0].axes[0]
    transformed_ax = figures[1].axes[0]

    assert raw_ax.get_ylabel() == "$i / i_p^0$"
    assert len(raw_ax.lines) >= 3
    np.testing.assert_allclose(
        raw_ax.lines[0].get_ydata(),
        cv_obj.y().to_numpy() / 1e-6,
    )
    assert transformed_ax.get_ylabel() == "$i / i_p^0$"

    for fig in figures:
        plt.close(fig)


def test_fowa_half_wave_redox_calculates_silently_then_redraws_normalized_diagnostic(
    ecat_module,
    blank_echem_factory,
    monkeypatch,
):
    cat_cv = _synthetic_fowa_cv(
        ecat_module,
        blank_echem_factory,
        "100mVs_cat_run01",
        1.0,
    )
    ref_cv = _synthetic_fowa_cv(
        ecat_module,
        blank_echem_factory,
        "100mVs_ref_run01",
        0.4,
    )
    redox_call_options = []

    def half_wave_spy(options=None):
        redox_call_options.append((options or {}).copy())
        return {"E(1/2)": 0.0, "ΔE": 0.05, "peak 1": {}, "peak 2": {}}

    ref_cv.half_wave_potential = half_wave_spy
    monkeypatch.setattr(
        ecat_module,
        "build_object_table",
        lambda object_list, options=None: (
            pd.DataFrame({"Name": [obj.name for obj in object_list]}),
            {},
        ),
    )

    with pytest.warns(UserWarning, match="non-positive slope"):
        ecat_module.fowa(
            [cat_cv],
            {
                "plot": True,
                "plot all": True,
                "legend": False,
                "title": False,
                "print": False,
                "non-catalytic cv": ref_cv,
                "non-catalytic current": 1e-6,
                "redox mode": "half wave",
                "peak prominence": 0,
                "background correction": None,
                "fit range": [0.1, 0.5],
                "min fit points": 5,
                "min r2": 0,
                "ecat shift warning threshold": False,
            },
        )

    assert len(redox_call_options) == 2
    assert redox_call_options[0]["plot"] is False
    assert redox_call_options[0]["plot all"] is False
    assert redox_call_options[1]["plot"] is True
    assert redox_call_options[1]["plot all"] is True
    assert redox_call_options[1]["y axis"] == "i/ip0"


def test_fowa_plot_all_redraws_half_wave_diagnostics_on_normalized_axis(
    ecat_module,
    blank_echem_factory,
    monkeypatch,
):
    cat_cv = _synthetic_fowa_cv(
        ecat_module,
        blank_echem_factory,
        "100mVs_cat_run01",
        1.0,
    )
    ref_cv = _synthetic_reversible_reference_cv(
        ecat_module,
        blank_echem_factory,
        "100mVs_ref_run01",
    )
    monkeypatch.setattr(
        ecat_module,
        "build_object_table",
        lambda object_list, options=None: (
            pd.DataFrame({"Name": [obj.name for obj in object_list]}),
            {},
        ),
    )

    with pytest.warns(UserWarning, match="non-positive slope"):
        ecat_module.fowa(
            [cat_cv],
            {
                "plot": True,
                "plot all": True,
                "legend": False,
                "title": False,
                "print": False,
                "non-catalytic cv": ref_cv,
                "non-catalytic current": 1e-6,
                "redox mode": "half wave",
                "background correction": None,
                "fit range": [0.1, 0.5],
                "min fit points": 5,
                "min r2": 0,
                "ecat shift warning threshold": False,
            },
        )

    figures = [plt.figure(number) for number in plt.get_fignums()]
    raw_ax = figures[0].axes[0]
    diagnostic_offsets = [
        collection.get_offsets()
        for collection in raw_ax.collections
        if hasattr(collection, "get_offsets") and len(collection.get_offsets()) > 0
    ]

    assert len(figures) == 2
    assert raw_ax.get_ylabel() == "$i / i_p^0$"
    assert diagnostic_offsets
    assert any(np.nanmax(np.abs(offsets[:, 1])) > 0.1 for offsets in diagnostic_offsets)

    for fig in figures:
        plt.close(fig)


def test_fowa_plot_all_multiplot_lines_are_full_cvs_and_only_traces_in_legend(
    ecat_module,
    blank_echem_factory,
    monkeypatch,
):
    cvs = [
        _synthetic_fowa_cv(ecat_module, blank_echem_factory, "100mVs_cat_run01", 1.0),
        _synthetic_fowa_cv(ecat_module, blank_echem_factory, "100mVs_cat_run02", 1.2),
    ]
    monkeypatch.setattr(
        ecat_module,
        "build_object_table",
        lambda object_list, options=None: (
            pd.DataFrame({"Name": [obj.name for obj in object_list]}),
            {},
        ),
    )

    with pytest.warns(UserWarning, match="non-positive slope"):
        ecat_module.fowa(
            cvs,
            {
                "plot": False,
                "plot all": True,
                "legend": True,
                "title": False,
                "print": False,
                "labels": ["Trace A", "Trace B"],
                "non-catalytic current": 1e-6,
                "redox potential": 0.0,
                "redox mode": "manual",
                "background correction": "tangent",
                "fit range": [0.1, 0.5],
                "min fit points": 5,
                "min r2": 0,
                "ecat shift warning threshold": False,
            },
        )

    fig = plt.gcf()
    ax = plt.gca()
    legend = ax.get_legend()
    legend_labels = [text.get_text() for text in legend.get_texts()]

    assert len(ax.lines[0].get_xdata()) == len(cvs[0].x())
    assert len(ax.lines[1].get_xdata()) == len(cvs[1].x())
    assert not any(label.startswith("_child") for label in legend_labels)
    assert legend_labels == ["Trace A", "Trace B"]

    plt.close(fig)


def test_fowa_plot_all_shows_multiplot_legend_by_default(
    ecat_module,
    blank_echem_factory,
    monkeypatch,
):
    cvs = [
        _synthetic_fowa_cv(ecat_module, blank_echem_factory, "100mVs_cat_run01", 1.0),
        _synthetic_fowa_cv(ecat_module, blank_echem_factory, "100mVs_cat_run02", 1.2),
    ]
    monkeypatch.setattr(
        ecat_module,
        "build_object_table",
        lambda object_list, options=None: (
            pd.DataFrame({"Name": [obj.name for obj in object_list]}),
            {},
        ),
    )

    with pytest.warns(UserWarning, match="non-positive slope"):
        ecat_module.fowa(
            cvs,
            {
                "plot": False,
                "plot all": True,
                "title": False,
                "print": False,
                "labels": ["Trace A", "Trace B"],
                "non-catalytic current": 1e-6,
                "redox potential": 0.0,
                "redox mode": "manual",
                "background correction": "tangent",
                "fit range": [0.1, 0.5],
                "min fit points": 5,
                "min r2": 0,
                "ecat shift warning threshold": False,
            },
        )

    fig = plt.gcf()
    ax = plt.gca()
    legend = ax.get_legend()
    legend_labels = [text.get_text() for text in legend.get_texts()]

    assert legend is not None
    assert legend_labels == ["Trace A", "Trace B"]

    plt.close(fig)


def test_fowa_plot_all_leaves_auto_labels_to_multiplot(
    ecat_module,
    blank_echem_factory,
    monkeypatch,
):
    ref = _synthetic_fowa_cv(ecat_module, blank_echem_factory, "100mVs_ref", 0.4)
    ref.compounds = ["Fc", "[Zn]"]
    ref.concentrations = ["3 mM", "20 mM"]

    cvs = [
        _synthetic_fowa_cv(ecat_module, blank_echem_factory, "100mVs_cat_10mMH2O", 1.0),
        _synthetic_fowa_cv(ecat_module, blank_echem_factory, "100mVs_cat_20mMH2O", 1.2),
    ]
    for cv_obj, h2o_conc in zip(cvs, ["10 mM", "20 mM"]):
        cv_obj.compounds = ["Fc", "[Zn]", "H2O"]
        cv_obj.concentrations = ["3 mM", "20 mM", h2o_conc]

    captured_options = []

    def fake_multiplot(objects, options=None):
        captured_options.append(dict(options or {}))
        plt.figure()
        ax = plt.gca()
        labels = (options or {}).get("labels") or [obj.name for obj in objects]
        for i, obj in enumerate(objects):
            ax.plot(obj.x(), obj.y({"y axis": "i/ip0"}), label=labels[i])

    monkeypatch.setattr(ecat_module, "multiplot", fake_multiplot)
    monkeypatch.setattr(
        ecat_module,
        "build_object_table",
        lambda object_list, options=None: (
            pd.DataFrame({"Name": [obj.name for obj in object_list]}),
            {},
        ),
    )

    with pytest.warns(UserWarning, match="non-positive slope"):
        ecat_module.fowa(
            cvs,
            {
                "plot": False,
                "plot all": True,
                "legend": True,
                "title": False,
                "print": False,
                "min gradient entries": 4,
                "non-catalytic cv": ref,
                "ip0": 1e-6,
                "redox potential": 0.0,
                "redox mode": "manual",
                "background correction": None,
                "fit range": [0.1, 0.5],
                "min fit points": 5,
                "min r2": 0,
                "ecat shift warning threshold": False,
            },
    )

    assert captured_options
    assert captured_options[-1].get("labels") is None
    assert captured_options[-1]["min gradient entries"] == 4

    plt.close("all")


def test_fowa_plot_all_main_trace_uses_full_cv_not_analysis_segment(
    ecat_module,
    blank_echem_factory,
    monkeypatch,
):
    cv_obj = _synthetic_multisegment_fowa_cv(
        ecat_module,
        blank_echem_factory,
        "100mVs_cat_run01",
        1.0,
    )
    monkeypatch.setattr(
        ecat_module,
        "build_object_table",
        lambda object_list, options=None: (
            pd.DataFrame({"Name": [obj.name for obj in object_list]}),
            {},
        ),
    )

    with pytest.warns(UserWarning, match="non-positive slope"):
        ecat_module.fowa(
            [cv_obj],
            {
                "plot": False,
                "plot all": True,
                "legend": False,
                "title": False,
                "print": False,
                "non-catalytic current": 1e-6,
                "redox potential": 0.0,
                "redox mode": "manual",
                "segment": 1,
                "background correction": None,
                "fit range": [0.1, 0.5],
                "min fit points": 5,
                "min r2": 0,
                "ecat shift warning threshold": False,
            },
        )

    fig = plt.gcf()
    ax = plt.gca()
    segment_x, _segment_y = cv_obj.analysis_segment_data({"segment": 1})

    assert len(segment_x) < len(cv_obj.x())
    assert len(ax.lines[0].get_xdata()) == len(cv_obj.x())
    assert "i/ip0" not in cv_obj.data.columns

    plt.close(fig)


def test_fowa_plot_all_plot_segments_controls_diagnostic_multiplot_only(
    ecat_module,
    blank_echem_factory,
    monkeypatch,
):
    cv_obj = _synthetic_multisegment_fowa_cv(
        ecat_module,
        blank_echem_factory,
        "100mVs_cat_run01",
        1.0,
    )
    captured_multiplot_options = []
    segment_calls = []

    def fake_multiplot(objects, options=None):
        captured_multiplot_options.append(dict(options or {}))
        plt.figure()
        ax = plt.gca()
        for obj in objects:
            ax.plot(obj.x(options or {}), obj.y(options or {}), label=obj.name)

    original_analysis_segment_data = cv_obj.analysis_segment_data

    def capture_analysis_segment_data(options=None):
        segment_calls.append((options or {}).get("segment"))
        return original_analysis_segment_data(options)

    cv_obj.analysis_segment_data = capture_analysis_segment_data
    monkeypatch.setattr(ecat_module, "multiplot", fake_multiplot)
    monkeypatch.setattr(
        ecat_module,
        "build_object_table",
        lambda object_list, options=None: (
            pd.DataFrame({"Name": [obj.name for obj in object_list]}),
            {},
        ),
    )

    with pytest.warns(UserWarning, match="non-positive slope"):
        ecat_module.fowa(
            [cv_obj],
            {
                "plot": False,
                "plot all": True,
                "plot segments": [2],
                "legend": False,
                "title": False,
                "print": False,
                "non-catalytic current": 1e-6,
                "redox potential": 0.0,
                "redox mode": "manual",
                "segment": 1,
                "background correction": None,
                "fit range": [0.1, 0.5],
                "min fit points": 5,
                "min r2": 0,
                "ecat shift warning threshold": False,
            },
        )

    assert captured_multiplot_options
    assert captured_multiplot_options[-1]["plot segments"] == [2]
    assert 1 in segment_calls

    plt.close("all")


def test_fowa_plot_all_half_wave_diagnostics_do_not_duplicate_peak_markers(
    ecat_module,
    blank_echem_factory,
    monkeypatch,
):
    ref_cv = _synthetic_reversible_reference_cv(
        ecat_module,
        blank_echem_factory,
        "100mVs_ref_Fc",
    )
    cat_cv = _synthetic_fowa_cv(
        ecat_module,
        blank_echem_factory,
        "100mVs_cat_run01",
        1.0,
    )
    monkeypatch.setattr(
        ecat_module,
        "build_object_table",
        lambda object_list, options=None: (
            pd.DataFrame({"Name": [obj.name for obj in object_list]}),
            {},
        ),
    )

    with pytest.warns(UserWarning, match="non-positive slope"):
        ecat_module.fowa(
            [cat_cv],
            {
                "plot": False,
                "plot all": True,
                "legend": False,
                "title": False,
                "print": False,
                "non-catalytic cv": ref_cv,
                "guess potential": 0.08,
                "redox mode": "half wave",
                "background correction": None,
                "fit range": [0.1, 0.5],
                "min fit points": 5,
                "min r2": 0,
                "ecat shift warning threshold": False,
            },
        )

    fig = plt.gcf()
    ax = plt.gca()
    blue_scatter_point_count = sum(
        len(collection.get_offsets())
        for collection in ax.collections
        if len(collection.get_facecolors()) > 0
        and np.allclose(collection.get_facecolors()[0][:3], (31 / 255, 119 / 255, 180 / 255))
    )

    assert blue_scatter_point_count == 2

    plt.close(fig)


def test_fowa_plot_all_redraws_peak_current_diagnostic_on_normalized_axis(
    ecat_module,
    blank_echem_factory,
    monkeypatch,
):
    ref_cv = _synthetic_reversible_reference_cv(
        ecat_module,
        blank_echem_factory,
        "100mVs_ref_Fc",
    )
    cat_cv = _synthetic_fowa_cv(
        ecat_module,
        blank_echem_factory,
        "100mVs_cat_run01",
        1.0,
    )
    monkeypatch.setattr(
        ecat_module,
        "build_object_table",
        lambda object_list, options=None: (
            pd.DataFrame({"Name": [obj.name for obj in object_list]}),
            {},
        ),
    )

    with pytest.warns(UserWarning, match="non-positive slope"):
        ecat_module.fowa(
            [cat_cv],
            {
                "plot": False,
                "plot all": True,
                "legend": False,
                "title": False,
                "print": False,
                "non-catalytic cv": ref_cv,
                "guess potential": 0.08,
                "redox mode": "half wave",
                "background correction": None,
                "fit range": [0.1, 0.5],
                "min fit points": 5,
                "min r2": 0,
                "ecat shift warning threshold": False,
            },
        )

    fig = plt.gcf()
    ax = plt.gca()
    red_dashed_lines = [
        line
        for line in ax.lines
        if line.get_color() == "tab:red" and line.get_linestyle() == "--"
    ]

    assert red_dashed_lines

    plt.close(fig)


def test_multiplot_bool_title_and_subtitle_resolve_to_auto_text(ecat_module, cv_factory):
    cvs = [cv_factory(name="100mVs_A_CO2_MeCN_10mM_Fc"), cv_factory(name="200mVs_B_CO2_MeCN_10mM_Fc")]
    options = ecat_module.MultiplotOptions.from_options(
        {"title": True, "subtitle": True}
    ).to_options_dict()

    _labels, title, subtitle, _shared, _similarities = (
        ecat_module._resolve_multiplot_labels_title_subtitle(cvs, options)
    )

    assert title != "True"
    assert subtitle != "True"
    assert title is None or isinstance(title, str)
    assert subtitle is None or isinstance(subtitle, str)


def test_fowa_transformed_plot_uses_user_plot_labels_not_filename_fallback(
    ecat_module,
    blank_echem_factory,
    monkeypatch,
):
    cvs = [
        _synthetic_fowa_cv(ecat_module, blank_echem_factory, "100mVs_cat_run01", 1.0),
        _synthetic_fowa_cv(ecat_module, blank_echem_factory, "100mVs_cat_run02", 1.2),
    ]
    monkeypatch.setattr(
        ecat_module,
        "build_object_table",
        lambda object_list, options=None: (
            pd.DataFrame({"Name": [obj.name for obj in object_list]}),
            {},
        ),
    )

    with pytest.warns(UserWarning, match="non-positive slope"):
        ecat_module.fowa(
            cvs,
            {
                "plot": True,
                "plot all": False,
                "legend": True,
                "title": False,
                "print": False,
                "labels": ["Trace A", "Trace B"],
                "non-catalytic current": 1e-6,
                "redox potential": 0.0,
                "redox mode": "manual",
                "background correction": None,
                "fit range": [0.1, 0.5],
                "min fit points": 5,
                "min r2": 0,
                "ecat shift warning threshold": False,
            },
        )

    fig = plt.gcf()
    ax = plt.gca()
    legend = ax.get_legend()
    legend_labels = [text.get_text() for text in legend.get_texts()]

    assert legend_labels == ["Trace A", "Trace B"]

    plt.close(fig)


def test_fowa_transformed_plot_labels_ignore_reference_cv_when_finding_shared_terms(
    ecat_module,
    blank_echem_factory,
    monkeypatch,
):
    ref = _synthetic_fowa_cv(ecat_module, blank_echem_factory, "100mVs_background", 0.4)
    ref.compounds = ["Fc"]
    ref.concentrations = ["3 mM"]

    cvs = [
        _synthetic_fowa_cv(ecat_module, blank_echem_factory, "100mVs_cat_10mMH2O", 1.0),
        _synthetic_fowa_cv(ecat_module, blank_echem_factory, "100mVs_cat_20mMH2O", 1.2),
    ]
    for cv_obj, h2o_conc in zip(cvs, ["10 mM", "20 mM"]):
        cv_obj.compounds = ["Fc", "[Zn]", "H2O"]
        cv_obj.concentrations = ["3 mM", "20 mM", h2o_conc]

    monkeypatch.setattr(
        ecat_module,
        "build_object_table",
        lambda object_list, options=None: (
            pd.DataFrame({"Name": [obj.name for obj in object_list]}),
            {},
        ),
    )

    with pytest.warns(UserWarning, match="non-positive slope"):
        ecat_module.fowa(
            cvs,
            {
                "plot": True,
                "plot all": False,
                "legend": True,
                "legend mode": "discrete",
                "title": False,
                "print": False,
                "non-catalytic cv": ref,
                "ip0": 1e-6,
                "redox potential": 0.0,
                "redox mode": "manual",
                "background correction": None,
                "fit range": [0.1, 0.5],
                "min fit points": 5,
                "min r2": 0,
                "ecat shift warning threshold": False,
            },
        )

    ax = plt.gca()
    legend = ax.get_legend()
    legend_labels = [text.get_text() for text in legend.get_texts()]

    assert legend_labels == ["10 mM H$_2$O", "20 mM H$_2$O"]
    assert not any("Fc" in label or "Zn" in label for label in legend_labels)

    plt.close("all")


def test_fowa_transformed_plot_generated_colorbar_labels_use_concentration_delta(
    ecat_module,
    blank_echem_factory,
    monkeypatch,
):
    labels_and_metadata = [
        ("100mVs_Ar", "Ar", [], [], "Ar"),
        (
            "100mVs_Ar_CO2_5pctCO2",
            "Ar/CO2",
            ["CO2"],
            ["5 %"],
            "Ar/CO2, 5 % CO2",
        ),
        (
            "100mVs_Ar_CO2_10pctCO2",
            "Ar/CO2",
            ["CO2"],
            ["10 %"],
            "Ar/CO2, 10 % CO2",
        ),
        (
            "100mVs_Ar_CO2_20pctCO2",
            "Ar/CO2",
            ["CO2"],
            ["20 %"],
            "Ar/CO2, 20 % CO2",
        ),
    ]
    cvs = []
    for name, gas, compounds, concentrations, _label in labels_and_metadata:
        cv_obj = _synthetic_fowa_cv(ecat_module, blank_echem_factory, name, 1.0)
        cv_obj.gas = gas
        cv_obj.compounds = compounds
        cv_obj.concentrations = concentrations
        cvs.append(cv_obj)

    plot_data = []
    for cv_obj, (_name, _gas, _compounds, _concentrations, label) in zip(
        cvs,
        labels_and_metadata,
    ):
        plot_data.append({
            "cat cv": cv_obj,
            "x fowa": np.array([0.1, 0.2, 0.3]),
            "y fowa": np.array([1.0, 1.2, 1.4]),
            "x fit": np.array([0.1, 0.2]),
            "y fit": np.array([1.0, 1.2]),
            # Default FOWA should not pass generated labels back through the
            # explicit-label path; multiplot should infer from CV metadata.
            "plot label": f"generated {label}",
        })

    captured_color_specs = []
    original_prepare = ecat_module._prepare_multiplot_style

    def capture_prepare(objects, options):
        style = original_prepare(objects, options)
        captured_color_specs.append(style["color spec"])
        return style

    monkeypatch.setattr(ecat_module, "_prepare_multiplot_style", capture_prepare)
    monkeypatch.setattr(
        ecat_module,
        "_draw_multiplot_legend_and_colorbars",
        lambda *args, **kwargs: None,
    )

    ecat_module._plot_fowa_transformed(
        plot_data,
        pd.DataFrame({"Slope": [1.0] * len(cvs), "Intercept": [0.0] * len(cvs)}),
        {
            "legend": True,
            "legend mode": "colorbar",
            "min gradient entries": 3,
            "title": False,
            "print": False,
            "plot fit": False,
        },
    )

    assert captured_color_specs
    gradient_groups = captured_color_specs[0]["gradient groups"]
    assert len(gradient_groups) == 1
    assert gradient_groups[0]["indices"] == [1, 2, 3]
    assert gradient_groups[0]["endpoint ticklabels"] == [
        "+5% CO$_2$",
        "+20% CO$_2$",
    ]
    assert "Ar/CO2" not in "".join(gradient_groups[0]["endpoint ticklabels"])

    plt.close("all")


def test_fowa_transformed_plot_formats_formula_plot_labels(
    ecat_module,
    blank_echem_factory,
    monkeypatch,
):
    cvs = [
        _synthetic_fowa_cv(ecat_module, blank_echem_factory, "100mVs_cat_run01", 1.0),
        _synthetic_fowa_cv(ecat_module, blank_echem_factory, "100mVs_cat_run02", 1.2),
    ]
    monkeypatch.setattr(
        ecat_module,
        "build_object_table",
        lambda object_list, options=None: (
            pd.DataFrame({"Name": [obj.name for obj in object_list]}),
            {},
        ),
    )

    with pytest.warns(UserWarning, match="non-positive slope"):
        ecat_module.fowa(
            cvs,
            {
                "plot": True,
                "plot all": False,
                "legend": True,
                "title": False,
                "print": False,
                "labels": ["10 mM CO2", "20 mM CO2"],
                "non-catalytic current": 1e-6,
                "redox potential": 0.0,
                "redox mode": "manual",
                "background correction": None,
                "fit range": [0.1, 0.5],
                "min fit points": 5,
                "min r2": 0,
                "ecat shift warning threshold": False,
            },
        )

    fig = plt.gcf()
    ax = plt.gca()
    legend = ax.get_legend()
    legend_labels = [text.get_text() for text in legend.get_texts()]

    assert legend_labels == ["10 mM CO$_2$", "20 mM CO$_2$"]

    plt.close(fig)


def test_fowa_transformed_plot_shows_multiplot_legend_by_default(
    ecat_module,
    blank_echem_factory,
    monkeypatch,
):
    cvs = [
        _synthetic_fowa_cv(ecat_module, blank_echem_factory, "100mVs_cat_run01", 1.0),
        _synthetic_fowa_cv(ecat_module, blank_echem_factory, "100mVs_cat_run02", 1.2),
    ]
    monkeypatch.setattr(
        ecat_module,
        "build_object_table",
        lambda object_list, options=None: (
            pd.DataFrame({"Name": [obj.name for obj in object_list]}),
            {},
        ),
    )

    with pytest.warns(UserWarning, match="non-positive slope"):
        ecat_module.fowa(
            cvs,
            {
                "plot": True,
                "plot all": False,
                "title": False,
                "print": False,
                "labels": ["Trace A", "Trace B"],
                "non-catalytic current": 1e-6,
                "redox potential": 0.0,
                "redox mode": "manual",
                "background correction": None,
                "fit range": [0.1, 0.5],
                "min fit points": 5,
                "min r2": 0,
                "ecat shift warning threshold": False,
            },
        )

    fig = plt.gcf()
    ax = plt.gca()
    legend = ax.get_legend()
    legend_labels = [text.get_text() for text in legend.get_texts()]

    assert legend is not None
    assert legend_labels == ["Trace A", "Trace B"]

    plt.close(fig)


def test_fowa_plot_all_overlays_do_not_expand_main_cv_axis_limits(
    ecat_module,
    blank_echem_factory,
    monkeypatch,
):
    cv_obj = _synthetic_fowa_cv(
        ecat_module,
        blank_echem_factory,
        "100mVs_cat_run01",
        1.0,
    )
    monkeypatch.setattr(
        ecat_module,
        "build_object_table",
        lambda object_list, options=None: (
            pd.DataFrame({"Name": [obj.name for obj in object_list]}),
            {},
        ),
    )

    with pytest.warns(UserWarning, match="non-positive slope"):
        ecat_module.fowa(
            [cv_obj],
            {
                "plot": False,
                "plot all": True,
                "legend": False,
                "title": False,
                "print": False,
                "non-catalytic current": 1e-6,
                "redox potential": 0.0,
                "redox mode": "manual",
                "background correction": "tangent",
                "fit range": [0.1, 0.5],
                "min fit points": 5,
                "min r2": 0,
                "ecat shift warning threshold": False,
            },
        )

    fig = plt.gcf()
    ax = plt.gca()
    main_y = ax.lines[0].get_ydata()
    y0, y1 = ax.get_ylim()
    main_span = float(np.nanmax(main_y) - np.nanmin(main_y))

    assert (y1 - y0) < main_span * 1.5

    plt.close(fig)


def test_fowa_plot_all_manual_tangent_potential_does_not_create_red_only_figure(
    ecat_module,
    blank_echem_factory,
    monkeypatch,
):
    cv_obj = _synthetic_fowa_cv(
        ecat_module,
        blank_echem_factory,
        "100mVs_cat_run01",
        1.0,
    )
    monkeypatch.setattr(
        ecat_module,
        "build_object_table",
        lambda object_list, options=None: (
            pd.DataFrame({"Name": [obj.name for obj in object_list]}),
            {},
        ),
    )

    with pytest.warns(UserWarning, match="non-positive slope"):
        ecat_module.fowa(
            [cv_obj],
            {
                "plot": False,
                "plot all": True,
                "legend": False,
                "title": False,
                "print": False,
                "non-catalytic current": 1e-6,
                "redox potential": 0.0,
                "redox mode": "manual",
                "background correction": "tangent",
                "tangent potential": -0.1,
                "fit range": [0.1, 0.5],
                "min fit points": 5,
                "min r2": 0,
                "ecat shift warning threshold": False,
            },
        )

    assert len(plt.get_fignums()) == 1

    fig = plt.gcf()
    ax = plt.gca()
    assert ax.lines

    plt.close(fig)


def test_fowa_tangent_failure_reports_allowed_current_settings(
    ecat_module,
    blank_echem_factory,
    monkeypatch,
):
    cv_obj = _synthetic_fowa_cv(
        ecat_module,
        blank_echem_factory,
        "100mVs_cat_run01",
        1.0,
    )

    def fail_tangent(*args, **kwargs):
        raise ValueError(
            "Could not find enough tangent-fit points. "
            "Try increasing 'percent threshold' or 'tangent activity fraction'."
        )

    monkeypatch.setattr(cv_obj, "_fit_tangent_line", fail_tangent)

    with pytest.raises(ValueError) as excinfo:
        ecat_module.fowa(
            [cv_obj],
            {
                "plot": False,
                "print": False,
                "non-catalytic current": 1e-6,
                "redox potential": 0.0,
                "redox mode": "manual",
                "background correction": "tangent",
                "tangent range": "auto",
                "tangent potential": None,
                "tangent min points": None,
                "percent threshold": None,
                "fit range": [0.1, 0.5],
                "min fit points": 5,
                "min r2": 0,
                "ecat shift warning threshold": False,
            },
        )

    message = str(excinfo.value)

    assert "Could not fit FOWA tangent background" in message
    assert "100mVs_cat_run01" in message
    assert "Current tangent settings:" in message
    assert "tangent range: auto" in message
    assert "tangent potential: None" in message
    assert "tangent min points: None" in message
    assert "percent threshold: None" in message
    assert "set 'tangent range' manually" in message
    assert "set 'tangent potential' manually" in message
    assert "increase 'percent threshold'" in message
    assert "'background correction': 'start current'" in message
    assert "'troubleshoot': True" in message
    assert "tangent activity fraction" not in message


def test_fowa_transformed_fit_lines_extend_to_wave_y_limit_and_skip_legend(
    ecat_module,
    blank_echem_factory,
    monkeypatch,
):
    cvs = [
        _synthetic_fowa_cv(ecat_module, blank_echem_factory, "100mVs_cat_run01", 1.0),
        _synthetic_fowa_cv(ecat_module, blank_echem_factory, "100mVs_cat_run02", 1.2),
    ]
    monkeypatch.setattr(
        ecat_module,
        "build_object_table",
        lambda object_list, options=None: (
            pd.DataFrame({"Name": [obj.name for obj in object_list]}),
            {},
        ),
    )

    with pytest.warns(UserWarning, match="non-positive slope"):
        ecat_module.fowa(
            cvs,
            {
                "plot": True,
                "plot all": False,
                "legend": True,
                "title": False,
                "print": False,
                "non-catalytic current": 1e-6,
                "redox potential": 0.0,
                "redox mode": "manual",
                "background correction": None,
                "fit range": [0.1, 0.5],
                "min fit points": 5,
                "min r2": 0,
                "ecat shift warning threshold": False,
            },
        )

    fig = plt.gcf()
    ax = plt.gca()
    legend = ax.get_legend()
    legend_labels = [text.get_text() for text in legend.get_texts()]
    dashed_fit_lines = [line for line in ax.lines if line.get_linestyle() == "--"]

    assert legend is not None
    assert len(legend_labels) == 2
    assert all("Fit" not in label for label in legend_labels)
    assert len(dashed_fit_lines) == 2

    solid_wave_lines = [line for line in ax.lines if line.get_linestyle() == "-"]
    for fit_line, wave_line in zip(dashed_fit_lines, solid_wave_lines):
        fit_y = fit_line.get_ydata()
        wave_y = wave_line.get_ydata()
        expected_endpoint = np.nanmax(wave_y) if fit_y[-1] >= fit_y[0] else np.nanmin(wave_y)
        assert fit_y[-1] == pytest.approx(expected_endpoint, rel=1e-3)

    plt.close(fig)
