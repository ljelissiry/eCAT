def test_module_import_smoke(ecat_module):
    assert ecat_module.__name__ == "ecat"
    assert ecat_module.__version__
    assert hasattr(ecat_module, "ImportOptions")
    assert hasattr(ecat_module, "PlotOptions")
    assert hasattr(ecat_module, "echem")
    assert hasattr(ecat_module, "cv")
    assert hasattr(ecat_module, "cp")
    assert hasattr(ecat_module, "ca")
    assert hasattr(ecat_module, "get_data")
    assert hasattr(ecat_module, "multiplot")
    assert hasattr(ecat_module, "animate")
    assert hasattr(ecat_module, "filter")
    assert hasattr(ecat_module, "sort_and_group")


def test_package_import_smoke():
    import ecat

    assert ecat.__version__
    assert hasattr(ecat, "echem")
    assert hasattr(ecat, "cv")
    assert hasattr(ecat, "get_data")
    assert hasattr(ecat, "multiplot")
    assert hasattr(ecat, "animate")
    assert hasattr(ecat, "filter")
    assert hasattr(ecat, "sort_and_group")
