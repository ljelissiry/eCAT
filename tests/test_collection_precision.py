"""Precision is a shared metadata matching policy, not display rounding."""

import copy

import numpy as np
import pandas as pd
import pytest


def rates(cv_factory, values):
    objects = [cv_factory() for _ in values]
    for obj, value in zip(objects, values):
        obj.scan_rate = value
    return objects


@pytest.mark.parametrize("precision", [5, "auto", " AUTO ", {"SCAN_RATE": 5}])
def test_matching_precision_agrees_across_collection_operations(ecat_module, cv_factory, precision):
    a, b, c = rates(cv_factory, [0.1, 0.0999997, 0.11])
    options = {"sig figs": precision, "print": False}
    assert ecat_module.filter([a, b, c], {"scan rate": 0.1}, options) == [a, b]
    assert ecat_module.group([a, b, c], "scan rate", options) == [[a, b], [c]]
    assert ecat_module.sort([c, a, b], "scan rate", options) == [a, b, c]
    assert ecat_module.sort_and_group([c, a, b], "scan rate", options=options) == [[a, b], [c]]


def test_auto_default_exact_override_and_mapping_omissions(ecat_module, cv_factory):
    a, b = rates(cv_factory, [0.1, 0.0999997])
    for precision in (None, {"concentrations": 4}):
        options = {"sig figs": precision, "print": False}
        assert ecat_module.filter([a, b], {"scan rate": 0.1}, options) == [a]
        assert ecat_module.group([a, b], "scan rate", options) == [[a], [b]]
    assert ecat_module.group([a, b], "scan rate", {"print": False}) == [[a, b]]


@pytest.mark.parametrize("values, expected", [
    ([0.0, -0.0, 1e-12, 1.000001e-12], [[0, 1], [2, 3]]),
    ([-0.1, -0.0999997, 0.1], [[0, 1], [2]]),
    ([0.123444, 0.123446, 0.123447], [[0], [1, 2]]),
    ([9.99997, 10.0, 10.001], [[0, 1], [2]]),
])
def test_decimal_buckets_near_zero_negative_and_boundaries(ecat_module, cv_factory, values, expected):
    objects = rates(cv_factory, values)
    options = {"sig figs": 5, "print": False}
    expected_objects = [[objects[i] for i in indices] for indices in expected]
    assert ecat_module.group(objects, "scan rate", options) == expected_objects
    for group in expected_objects:
        assert ecat_module.filter(objects, {"scan rate": group[0].scan_rate}, options) == group


def test_precision_is_input_order_independent_and_preserves_data(ecat_module, cv_factory):
    objects = rates(cv_factory, [0.1, 0.0999997, 0.11])
    before = [(obj.scan_rate, copy.deepcopy(obj.concentrations), obj.data.copy(deep=True)) for obj in objects]
    options = {"sig figs": "auto", "print": False}
    groups = ecat_module.group(objects, "scan rate", options)
    reversed_groups = ecat_module.group(list(reversed(objects)), "scan rate", options)
    memberships = lambda groups: {frozenset(map(id, group)) for group in groups}
    assert memberships(groups) == memberships(reversed_groups)
    ecat_module.sort(objects.copy(), "scan rate", options)
    ecat_module.filter(objects, {"scan rate": 0.1}, options)
    for obj, (rate, concentrations, data) in zip(objects, before):
        assert obj.scan_rate == rate
        assert obj.concentrations == concentrations
        pd.testing.assert_frame_equal(obj.data, data)


def test_concentration_species_and_scan_window_share_matching_keys(ecat_module, cv_factory):
    a, b = rates(cv_factory, [0.1, 0.1])
    a.concentrations, b.concentrations = ["100 mM"], ["0.0999997 M"]
    a.compounds = b.compounds = ["Fc"]
    a.min_E, b.min_E = -0.1, -0.0999997
    a.max_E = b.max_E = 0.2
    for key, criterion in [("concentrations", "0.1 M"), ("species", "100mM Fc"), ("scan window", [-0.1, 0.2])]:
        options = {"sig figs": {key: 5}, "print": False}
        assert ecat_module.group([a, b], key, options) == [[a, b]]
        assert ecat_module.filter([a, b], {key: criterion}, options) == [a, b]
        assert ecat_module.filter([a, b], {key: criterion}, {"sig figs": None, "print": False}) == [a]


def test_missing_values_and_categorical_counts(ecat_module, cv_factory):
    a, b, c, d = rates(cv_factory, [None, float("nan"), np.float64("nan"), 0.1])
    options = {"sig figs": 1, "print": False}
    assert ecat_module.group([a, b, c, d], "scan rate", options) == [[a], [b, c], [d]]
    assert ecat_module.sort([b, a, d, c], "scan rate", options) == [d, b, a, c]
    assert ecat_module.filter([a, b, c, d], {"scan rate": float("nan")}, options) == [b, c]
    b.segments, c.segments = 11, 12
    assert ecat_module.group([b, c], "segments", options) == [[b], [c]]
    assert ecat_module.filter([b, c], {"segments": 11}, options) == [b]


@pytest.mark.parametrize("value", [True, False, 0, -1, 18, 2.5, 5.0, float("nan"), float("inf"), "5", "invalid", [], {"scan rate": True}, {"segments": 2}, {"unknown": 4}, {"scan rate": {"nested": 2}}])
@pytest.mark.parametrize("model_name", ["FilterOptions", "SortGroupOptions"])
def test_invalid_precision_rejected_early(ecat_module, value, model_name):
    with pytest.raises(ValueError, match="sig figs"):
        getattr(ecat_module, model_name).from_options({"sig figs": value})


@pytest.mark.parametrize("alias", ["sig figs", "sig_figs", "SIGFIGS", "Significant Figures"])
def test_precision_aliases_and_typed_options(ecat_module, cv_factory, alias):
    a, b = rates(cv_factory, [0.1, 0.0999997])
    options = ecat_module.FilterOptions.from_options({alias: "AUTO", "print": False})
    assert ecat_module.filter([a, b], {"scan rate": 0.1}, options) == [a, b]


@pytest.mark.parametrize("precision", [None, "auto", {"scan rate": "auto"}])
@pytest.mark.parametrize("operation", ["filter", "sort", "group"])
def test_precision_printing_uses_resolved_policy(ecat_module, cv_factory, capsys, precision, operation):
    obj = cv_factory()
    key = {"scan rate": obj.scan_rate} if operation == "filter" else "scan rate"
    getattr(ecat_module, operation)([obj], key, {"sig figs": precision, "pretty print": False})
    output = capsys.readouterr().out
    if precision is not None:
        assert "5 significant digits" in output
        assert "instrument resolution" in output


def test_invalid_direct_typed_options_rejected(ecat_module, cv_factory):
    obj = cv_factory()
    with pytest.raises(ValueError, match="sig figs"):
        ecat_module.filter([obj], {"scan rate": 0.1}, ecat_module.FilterOptions(sig_figs=True))


def test_numeric_filter_logic_nested_groups_and_replicates(ecat_module, cv_factory):
    a, b, c = rates(cv_factory, [0.1, 0.0999997, 0.2])
    a.min_E, b.min_E, c.min_E = -1.7, -1.699997, -1.6
    a.max_E = b.max_E = c.max_E = 0.0
    options = {"sig figs": "auto", "print": False}
    assert ecat_module.filter([[a, b], [c]], {"scan rate": [0.1]}, options) == [[a, b]]
    assert ecat_module.filter([a, b, c], {"scan window": -1.7}, options) == [a, b]
    assert ecat_module.filter([a, b, c], {"scan window": [[0, -1.7], [0, -1.6]]}, options) == [a, b, c]
    assert ecat_module.filter([a, b, c], {"scan rate": 0.1}, {**options, "mode": "exclude"}) == [c]
    # Precision never alters replicate identities, even when metadata buckets match.
    assert ecat_module.filter([a, b, c], {"replicate": -1}, options) == [a, b, c]


def test_significant_precision_is_scale_invariant(ecat_module, cv_factory):
    options = {"sig figs": 5, "print": False}
    for scale in (1e-9, 1.0, 1e9):
        a, b, c = rates(cv_factory, [0.1 * scale, 0.0999997 * scale, 0.11 * scale])
        assert ecat_module.group([a, b, c], "scan rate", options) == [[a, b], [c]]


def test_species_precision_supports_scientific_notation(ecat_module, cv_factory):
    a, b = rates(cv_factory, [0.1, 0.1])
    a.compounds = b.compounds = ["Fc"]
    a.concentrations, b.concentrations = ["1e-3 M"], ["0.999997 mM"]
    options = {"sig figs": 5, "print": False}
    assert ecat_module.group([a, b], "species", options) == [[a, b]]
    assert ecat_module.filter([a, b], {"species": "1e-3M Fc"}, options) == [a, b]


@pytest.mark.parametrize("key", ["species", "concentrations"])
@pytest.mark.parametrize("precision", [None, 5, "auto"])
def test_matching_preserves_concentration_unit_identity(ecat_module, cv_factory, key, precision):
    units = ["M", "x", "%", "L", "equiv"]
    objects = rates(cv_factory, [0.1] * len(units))
    for obj, unit in zip(objects, units):
        obj.concentrations = [f"1 {unit}"]
        obj.compounds = ["Fc"]
    options = {"sig figs": precision, "print": False}
    assert ecat_module.group(objects, key, options) == [[obj] for obj in objects]
    for obj, unit in zip(objects, units):
        criterion = f"1 {unit} Fc" if key == "species" else f"1 {unit}"
        assert ecat_module.filter(objects, {key: criterion}, options) == [obj]
    grouped = ecat_module.sort_and_group(objects, key, options=options)
    assert len(grouped) == len(units)
    assert all(len(group) == 1 for group in grouped)
    # A bare species still intentionally selects that compound at any composition.
    assert ecat_module.filter(objects, {"species": "Fc"}, options) == objects


@pytest.mark.parametrize("key", ["species", "concentrations"])
def test_molar_prefixes_match_without_crossing_unit_dimensions(ecat_module, cv_factory, key):
    a, b, c = rates(cv_factory, [0.1] * 3)
    for obj in (a, b, c):
        obj.compounds = ["Fc"]
    a.concentrations, b.concentrations, c.concentrations = ["1 mM"], ["0.001 M"], ["0.001 x"]
    options = {"sig figs": None, "print": False}
    assert ecat_module.group([a, b, c], key, options) == [[a, b], [c]]
    criterion = "1 mM Fc" if key == "species" else "1 mM"
    assert ecat_module.filter([a, b, c], {key: criterion}, options) == [a, b]
    assert ecat_module.filter([a, b, c], {"concentrations": 0.001}, options) == [a, b]
