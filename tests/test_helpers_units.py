import numpy as np
import pytest


def test_count_segments_detects_direction_change(ecat_module):
    x_values = np.array([0.0, 1.0, 2.0, 1.0, 0.0])

    assert ecat_module.count_segments(x_values) == 2


def test_savgol_noise_window_none_disables_smoothing(ecat_module):
    signal = np.array([0.0, 1.0, 0.0, 1.0, 0.0], dtype=float)

    smoothed, meta = ecat_module._savgol_apply(
        signal,
        {"noise window": None, "noise polyorder": "auto"},
    )

    np.testing.assert_allclose(smoothed, signal)
    assert meta["window"] is None
    assert meta["polyorder"] is None


def test_find_extrema_guess_uses_local_prominence_metadata(ecat_module):
    x = np.linspace(-1.0, 1.0, 801)
    signal = -1e-7 * np.exp(-((x + 0.05) / 0.025) ** 2)
    distant_region = x > 0.35
    signal[distant_region] += 8e-7 * np.sin(120 * (x[distant_region] - 0.35))

    _extrema, _smoothed, _prom_map, meta = ecat_module._find_extrema_indices(
        signal,
        {"noise window": 5, "noise polyorder": 2, "guess potential": -0.05},
        x=x,
    )

    assert meta["prominence mode"] == "guess local"
    assert meta["prominence window fraction"] == pytest.approx(0.2)
    assert meta["prominence window"] == pytest.approx([-0.25, 0.15])


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


def test_scale_axis_respects_already_prefixed_current_units(ecat_module):
    scale_factor, unit = ecat_module.scale_axis(np.array([0.0, 35.2]), "μA")

    assert scale_factor == pytest.approx(1.0)
    assert unit == "μA"

    explicit_scale, explicit_unit = ecat_module.scale_axis(
        np.array([0.0, 35.2]),
        "μA",
        selected_unit="uA",
    )

    assert explicit_scale == pytest.approx(1.0)
    assert explicit_unit == "μA"


def test_pressure_units_are_recognized_and_convert_when_requested(ecat_module):
    assert ecat_module.extract_prefix_and_base("Pa") == ("", "Pa")
    assert ecat_module.extract_prefix_and_base("kPa") == ("k", "Pa")
    assert ecat_module.extract_prefix_and_base("mbar") == ("m", "bar")
    assert ecat_module.extract_prefix_and_base("atm") == ("", "atm")
    assert ecat_module.extract_prefix_and_base("Torr") == ("", "Torr")
    assert ecat_module.extract_prefix_and_base("mmHg") == ("", "mmHg")
    assert ecat_module.extract_prefix_and_base("psi") == ("", "psi")
    assert ecat_module.extract_prefix_and_base("mpsi") == ("", "mpsi")

    assert ecat_module.get_conversion_factor("atm", "Pa") == pytest.approx(101325)
    assert ecat_module.get_conversion_factor("bar", "Pa") == pytest.approx(100000)
    assert ecat_module.get_conversion_factor("mbar", "Pa") == pytest.approx(100)
    assert ecat_module.get_conversion_factor("Torr", "Pa") == pytest.approx(101325 / 760)
    assert ecat_module.get_conversion_factor("mmHg", "atm") == pytest.approx(1 / 760)
    assert ecat_module.get_conversion_factor("psi", "Pa") == pytest.approx(6894.757293168)


def test_pressure_values_preserve_units_by_default_and_convert_explicitly(ecat_module):
    value, unit = ecat_module.scale_value(1.0, "atm", selected_unit="auto")
    assert value == pytest.approx(1.0)
    assert unit == "atm"

    value, unit = ecat_module.scale_value(1013.25, "mbar", selected_unit="bar")
    assert value == pytest.approx(1.01325)
    assert unit == "bar"

    scale_factor, unit = ecat_module.scale_axis(
        np.array([0.0, 1.0]),
        "atm",
        selected_unit="Pa",
    )
    assert scale_factor == pytest.approx(101325)
    assert unit == "Pa"

    scale_factor, unit = ecat_module.scale_axis(np.array([0.0, 1013.25]), "mbar")
    assert scale_factor == pytest.approx(1.0)
    assert unit == "mbar"


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
