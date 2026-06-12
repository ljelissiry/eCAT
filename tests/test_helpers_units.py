import numpy as np
import pytest


def test_count_segments_detects_direction_change(ecat_module):
    x_values = np.array([0.0, 1.0, 2.0, 1.0, 0.0])

    assert ecat_module.count_segments(x_values) == 2


def test_concentration_to_float_converts_prefixed_molar_units(ecat_module):
    assert ecat_module.concentration_to_float("250uM") == pytest.approx(250e-6)
    assert ecat_module.concentration_to_float("2mM") == pytest.approx(2e-3)


def test_extract_prefix_and_base_leaves_compound_units_intact(ecat_module):
    assert ecat_module.extract_prefix_and_base("V vs Fc/Fc+") == ("", "V vs Fc/Fc+")


def test_get_conversion_factor_and_scale_axis(ecat_module):
    assert ecat_module.get_conversion_factor("mA", "uA") == pytest.approx(1e3)

    scale_factor, unit = ecat_module.scale_axis(np.array([0.0, 2e-6]), "A")

    assert scale_factor == pytest.approx(1e6)
    assert unit == "μA"


def test_current_density_auto_scales_current_numerator(ecat_module):
    scale_factor, unit = ecat_module.scale_axis(
        np.array([0.0, 2e-6]),
        "A/cm$^2$",
    )

    assert scale_factor == pytest.approx(1e6)
    assert unit == "μA/cm$^2$"


@pytest.mark.parametrize("selected_unit", ["mm", "mm^2", "/mm^2"])
def test_current_density_area_unit_shorthand_keeps_current_auto(ecat_module, selected_unit):
    scale_factor, unit = ecat_module.scale_axis(
        np.array([0.0, 2e-6]),
        "A/cm$^2$",
        selected_unit=selected_unit,
    )

    assert scale_factor == pytest.approx(1e7)
    assert unit == "nA/mm$^2$"


def test_current_density_current_unit_shorthand_keeps_area_unit(ecat_module):
    scale_factor, unit = ecat_module.scale_axis(
        np.array([0.0, 2e-6]),
        "A/cm$^2$",
        selected_unit="uA",
    )

    assert scale_factor == pytest.approx(1e6)
    assert unit == "μA/cm$^2$"


def test_current_density_full_unit_is_exact(ecat_module):
    scale_factor, unit = ecat_module.scale_axis(
        np.array([0.0, 2e-6]),
        "A/cm$^2$",
        selected_unit="uA/mm^2",
    )

    assert scale_factor == pytest.approx(1e4)
    assert unit == "μA/mm$^2$"


def test_plain_current_rejects_density_unit(ecat_module):
    with pytest.raises(ValueError, match="incompatible units"):
        ecat_module.scale_axis(
            np.array([0.0, 2e-6]),
            "A",
            selected_unit="uA/cm^2",
        )
