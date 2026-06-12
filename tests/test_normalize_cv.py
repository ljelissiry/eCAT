import numpy as np
import pytest


def test_normalize_homogeneous_adds_dimensionless_axes_and_plot_defaults(
    ecat_module,
    cv_factory,
):
    obj = cv_factory(options={"electrode area": 0.071})

    normalized = ecat_module.normalize(
        obj,
        {
            "mode": "homogeneous",
            "E0": 0.0,
            "D": 1e-5,
            "C": 10,
            "C unit": "mM",
            "print": False,
        },
    )

    assert isinstance(normalized, ecat_module.cv)
    assert normalized is not obj
    assert "Dimensionless Potential" in normalized.data.columns
    assert "Dimensionless Current" in normalized.data.columns
    assert "Dimensionless Potential" not in obj.data.columns
    assert normalized.plot_mode == "normalized"
    assert normalized.normalization_mode == "homogeneous"
    assert normalized.normalization_axes == {
        "x": "Dimensionless Potential",
        "y": "Dimensionless Current",
    }

    expected_theta = ecat_module.F * (obj.x().to_numpy() - 0.0) / (
        ecat_module.R * obj.temperature
    )
    C_mol_cm3 = 10e-3 / 1000
    denominator = (
        ecat_module.F
        * obj.electrode_area
        * C_mol_cm3
        * np.sqrt(1e-5 * ecat_module.F * obj.scan_rate / (ecat_module.R * obj.temperature))
    )
    expected_phi = obj.y().to_numpy() / denominator

    np.testing.assert_allclose(normalized.x().to_numpy(), expected_theta)
    np.testing.assert_allclose(normalized.y().to_numpy(), expected_phi)

    ax = normalized.plot({"legend": False, "title": False})
    fig = ax.figure
    assert ax.get_xlabel() == r"$\theta$"
    assert ax.get_ylabel() == r"$\Phi$"


def test_normalize_accepts_multiple_cvs_and_returns_same_shape(ecat_module, cv_factory):
    objects = [
        cv_factory(name="50mVs_sample_CO2_MeCN_10mM_Fc_run01", options={"electrode area": 0.071}),
        cv_factory(name="100mVs_sample_CO2_MeCN_10mM_Fc_run02", options={"electrode area": 0.071}),
    ]

    normalized = ecat_module.normalize(
        objects,
        {
            "mode": "homogeneous",
            "E0": 0.0,
            "D": 1e-5,
            "C": 10,
            "C unit": "mM",
        },
    )

    assert isinstance(normalized, list)
    assert len(normalized) == 2
    assert all(item.plot_mode == "normalized" for item in normalized)
    assert normalized[0] is not objects[0]
    assert normalized[1] is not objects[1]


def test_normalize_series_prints_one_symbolic_equation_and_parameter_table(
    ecat_module,
    cv_factory,
    monkeypatch,
    capsys,
):
    import ecat.analysis_cv as analysis_cv

    objects = [
        cv_factory(name="50mVs_sample_CO2_MeCN_10mM_Fc_run01", options={"electrode area": 0.071}),
        cv_factory(name="100mVs_sample_CO2_MeCN_10mM_Fc_run02", options={"electrode area": 0.071}),
    ]
    displayed = {}

    def capture_display(obj):
        if hasattr(obj, "to_html"):
            displayed["html"] = obj.to_html()

    monkeypatch.setattr(analysis_cv, "display", capture_display)
    monkeypatch.setattr(analysis_cv, "Math", None)

    ecat_module.normalize(
        objects,
        {
            "mode": "homogeneous",
            "E0": 0.0,
            "D": 1e-5,
            "C": 10,
            "C unit": "mM",
            "print": True,
            "pretty print": True,
        },
    )

    output = capsys.readouterr().out
    assert output.count("[CV normalization equation]") == 1
    assert "theta = n * F * (E - E0) / (R * T)" in output
    assert "Phi = I / (n * F * S * C* * sqrt(D * n * F * v / (R * T)))" in output
    assert "E0 = 0" not in output
    assert "Mode: homogeneous" in output

    table = analysis_cv._cv_normalization_parameter_table(
        ecat_module.normalize(
            objects,
            {
                "mode": "homogeneous",
                "E0": 0.0,
                "D": 1e-5,
                "C": 10,
                "C unit": "mM",
                "print": False,
            },
        ),
        {"sig figs": 4},
    )
    assert list(table.columns) == [0, 1]
    assert "mode" not in table.index
    assert "v (V/s)" in table.index
    assert "denominator" not in table.index
    assert table.loc["v (V/s)", 0] != table.loc["v (V/s)", 1]
    assert "<i>T</i> / K" in displayed["html"]
    assert "cm<sup>2</sup> s<sup>-1</sup>" in displayed["html"]


def test_normalize_homogeneous_can_add_only_potential_axis(ecat_module, cv_factory):
    obj = cv_factory()

    normalized = ecat_module.normalize(
        obj,
        {
            "mode": "homogeneous",
            "E0": 0.0,
        },
    )

    assert "Dimensionless Potential" in normalized.data.columns
    assert "Dimensionless Current" not in normalized.data.columns
    assert normalized.x().name == "Dimensionless Potential"
    assert normalized.y().name == "Current"


def test_normalize_homogeneous_can_add_only_current_axis(ecat_module, cv_factory):
    obj = cv_factory(options={"electrode area": 0.071})

    normalized = ecat_module.normalize(
        obj,
        {
            "mode": "homogeneous",
            "D": 1e-5,
            "C": 10,
            "C unit": "mM",
        },
    )

    assert "Dimensionless Potential" not in normalized.data.columns
    assert "Dimensionless Current" in normalized.data.columns
    assert normalized.x().name == "Potential"
    assert normalized.y().name == "Dimensionless Current"


def test_normalize_species_exact_match_supplies_concentration(ecat_module, cv_factory):
    obj = cv_factory(options={"electrode area": 0.071})
    obj.compounds = ["Fc", "[Co]", "H2O"]
    obj.concentrations = ["3 mM", "1 mM", "2.8 M"]

    normalized = ecat_module.normalize(
        obj,
        {
            "mode": "homogeneous",
            "D": 1e-5,
            "species": "[Co]",
        },
    )

    C_mol_cm3 = 1e-3 / 1000
    denominator = (
        ecat_module.F
        * obj.electrode_area
        * C_mol_cm3
        * np.sqrt(1e-5 * ecat_module.F * obj.scan_rate / (ecat_module.R * obj.temperature))
    )
    np.testing.assert_allclose(normalized.y().to_numpy(), obj.y().to_numpy() / denominator)
    assert normalized.normalization_parameters["C text"] == "1 mM"
    assert normalized.normalization_parameters["species"] == "[Co]"


def test_normalize_explicit_concentration_overrides_species(ecat_module, cv_factory):
    obj = cv_factory(options={"electrode area": 0.071})
    obj.compounds = ["[Co]"]
    obj.concentrations = ["1 mM"]

    normalized = ecat_module.normalize(
        obj,
        {
            "mode": "homogeneous",
            "D": 1e-5,
            "C": 2,
            "C unit": "mM",
            "species": "[Co]",
        },
    )

    assert normalized.normalization_parameters["C text"] == "2 mM"
    assert "species" not in normalized.normalization_parameters


def test_normalize_species_matching_is_exact(ecat_module, cv_factory):
    obj = cv_factory(options={"electrode area": 0.071})
    obj.compounds = ["[Co]"]
    obj.concentrations = ["1 mM"]

    with pytest.raises(ValueError, match="Species '\\[co\\]' was not found"):
        ecat_module.normalize(
            obj,
            {
                "mode": "homogeneous",
                "D": 1e-5,
                "species": "[co]",
            },
        )


def test_normalize_species_lists_available_compounds_on_missing_match(ecat_module, cv_factory):
    obj = cv_factory(options={"electrode area": 0.071})
    obj.compounds = ["Fc", "[Co]"]
    obj.concentrations = ["3 mM", "1 mM"]

    with pytest.raises(ValueError, match="Available species: Fc, \\[Co\\]"):
        ecat_module.normalize(
            obj,
            {
                "mode": "homogeneous",
                "D": 1e-5,
                "species": "[Fe]",
            },
        )


def test_normalized_cv_axis_options_override_plot_mode(ecat_module, cv_factory):
    obj = cv_factory(options={"electrode area": 0.071})
    normalized = ecat_module.normalize(
        obj,
        {
            "mode": "homogeneous",
            "E0": 0.0,
            "D": 1e-5,
            "C": 10,
            "C unit": "mM",
        },
    )

    assert normalized.x({"x axis": "Potential"}).name == "Potential"
    assert normalized.y({"y axis": "Current"}).name == "Current"


def test_normalize_current_uses_raw_current_after_dimensionless_normalize(ecat_module, cv_factory):
    obj = cv_factory(options={"electrode area": 0.071})
    dimensionless = ecat_module.normalize(
        obj,
        {
            "mode": "homogeneous",
            "E0": 0.0,
            "D": 1e-5,
            "C": 10,
            "C unit": "mM",
        },
    )

    current_normalized = ecat_module.normalize_current(
        dimensionless,
        {"ip0": 2e-6, "print": False},
    )

    assert "Dimensionless Current" in current_normalized.data.columns
    assert "i/ip0" in current_normalized.data.columns
    np.testing.assert_allclose(
        current_normalized.y({"y axis": "i/ip0"}).to_numpy(),
        obj.y().to_numpy() / 2e-6,
    )
    assert current_normalized.normalization_mode == "i/ip0"
    assert current_normalized.normalization_axes == {"x": None, "y": "i/ip0"}
    assert current_normalized.normalization_axis_labels == {"y": "$i / i_p^0$"}


def test_normalize_prints_equations_for_created_axes(ecat_module, cv_factory, monkeypatch, capsys):
    import ecat.analysis_cv as analysis_cv

    obj = cv_factory()
    monkeypatch.setattr(analysis_cv, "display", None)
    monkeypatch.setattr(analysis_cv, "Math", None)

    ecat_module.normalize(
        obj,
        {
            "mode": "homogeneous",
            "E0": 0.0,
            "print": True,
        },
    )

    output = capsys.readouterr().out
    assert "[CV normalization equation]" in output
    assert "theta = n * F * (E - E0) / (R * T)" in output
    assert "Mode: homogeneous" in output
    assert "Parameter" in output
    assert "Value" in output
    assert "E0 (V)" in output
    assert "E0 = 0" not in output


def test_plot_time_normalize_true_is_removed(cv_factory):
    obj = cv_factory()

    with pytest.raises(ValueError, match="normalize"):
        obj.x({"normalize": True})

    with pytest.raises(ValueError, match="normalize"):
        obj.y({"normalize": True})

    with pytest.raises(ValueError, match="normalize"):
        obj.plot({"normalize": True})


def test_cv_normalize_mutates_and_returns_self(ecat_module, cv_factory):
    obj = cv_factory(options={"electrode area": 0.071})

    returned = obj.normalize(
        {
            "mode": "homogeneous",
            "E0": 0.0,
            "D": 1e-5,
            "C": 10,
            "C unit": "mM",
        },
    )

    assert returned is obj
    assert "Dimensionless Potential" in obj.data.columns
    assert "Dimensionless Current" in obj.data.columns


def test_top_level_normalize_rejects_non_cv_objects(ecat_module, blank_echem_factory):
    cp_obj = blank_echem_factory(ecat_module.cp)

    with pytest.raises(TypeError, match="normalize currently supports cv objects only"):
        ecat_module.normalize(cp_obj)
