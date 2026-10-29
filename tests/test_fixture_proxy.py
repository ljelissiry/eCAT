"""The shared test proxy must restore real module attributes after monkeypatching."""

import pytest


@pytest.mark.parametrize(
    "name", ["display", "multiplot", "_prepare_multiplot_style", "get_file_times"]
)
def test_forwarded_attribute_is_restored_after_monkeypatch(ecat_module, name):
    import ecat

    owners = [module for module in (ecat, *ecat_module._modules) if hasattr(module, name)]
    original = getattr(ecat_module, name)
    assert owners
    replacement = lambda *args, **kwargs: None

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(ecat_module, name, replacement)
        assert all(getattr(module, name) is replacement for module in owners)

    # This proxy forwards one value to every alias, including on restoration.
    # The regression is deletion of those aliases, not per-wrapper identity.
    assert getattr(ecat_module, name) is original
    assert all(getattr(module, name, None) is original for module in owners)
