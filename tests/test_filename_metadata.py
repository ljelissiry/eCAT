def test_get_data_from_name_extracts_gas_and_solvent(ecat_module, blank_echem_factory):
    obj = blank_echem_factory(ecat_module.echem)
    obj.name = "sample_CO2_MeCN_run01"

    obj.get_data_from_name()

    assert obj.gas == "CO2"
    assert obj.solvent == "MeCN"


def test_extract_compounds_and_concentrations_strips_filename_separators(
    ecat_module,
    blank_echem_factory,
):
    obj = blank_echem_factory(ecat_module.echem)
    obj.name = "sample_250uM_Fc_10mM_TBABF4_run01"

    compounds, concentrations = obj.extract_compounds_and_concentrations()

    assert compounds == ["Fc", "TBABF4"]
    assert concentrations == ["250 uM", "10 mM"]


def test_extract_compounds_and_concentrations_handles_brackets_equiv_and_percent(
    ecat_module,
    blank_echem_factory,
):
    obj = blank_echem_factory(ecat_module.echem)
    obj.name = "sample_250uM_[Fe(CN)6]3-_5equiv_H2O_10mM_Co(bpy)3Cl2_run01"

    compounds, concentrations = obj.extract_compounds_and_concentrations()

    assert compounds == ["[Fe(CN)6]3-", "H2O", "Co(bpy)3Cl2"]
    assert concentrations == ["250 uM", "5 equiv", "10 mM"]


def test_extract_compounds_and_concentrations_parses_percent_components(
    ecat_module,
    blank_echem_factory,
):
    obj = blank_echem_factory(ecat_module.echem)
    obj.name = "sample_50%_H2O_10mM_Fc_run01"

    compounds, concentrations = obj.extract_compounds_and_concentrations()

    assert compounds == ["H2O", "Fc"]
    assert concentrations == ["50 %", "10 mM"]


def test_extract_compounds_and_concentrations_parses_mole_fraction_x(
    ecat_module,
    blank_echem_factory,
):
    obj = blank_echem_factory(ecat_module.echem)
    obj.name = "sample_0.8xD2O_10mM_Fc_run01"

    compounds, concentrations = obj.extract_compounds_and_concentrations()

    assert compounds == ["D2O", "Fc"]
    assert concentrations == ["0.8 x", "10 mM"]


def test_extract_compounds_and_concentrations_preserves_molarity_and_x_for_same_species(
    ecat_module,
    blank_echem_factory,
):
    obj = blank_echem_factory(ecat_module.echem)
    obj.name = (
        "MeCN_CO2_0.1MTBAPF6_3mMFc_1mMFe-tpyPY2Me_"
        "20mMZn(cyclen)(OTf)2_2.8MD2O_0.8xD2O_1_to_-1.7V_100mVs"
    )

    compounds, concentrations = obj.extract_compounds_and_concentrations()

    assert compounds == [
        "TBAPF6",
        "Fc",
        "Fe-tpyPY2Me",
        "Zn(cyclen)(OTf)2",
        "D2O",
        "D2O",
    ]
    assert concentrations == ["0.1 M", "3 mM", "1 mM", "20 mM", "2.8 M", "0.8 x"]


def test_extract_compounds_and_concentrations_requires_explicit_x_endpoints(
    ecat_module,
    blank_echem_factory,
):
    obj = blank_echem_factory(ecat_module.echem)
    obj.name = "sample_0xD2O_1xH2O_D2O_run01"

    compounds, concentrations = obj.extract_compounds_and_concentrations()

    assert compounds == ["D2O", "H2O"]
    assert concentrations == ["0 x", "1 x"]


def test_extract_compounds_and_concentrations_does_not_parse_legacy_n_mole_fraction(
    ecat_module,
    blank_echem_factory,
):
    obj = blank_echem_factory(ecat_module.echem)
    obj.name = "sample_0.8nD2O_10mM_Fc_run01"

    compounds, concentrations = obj.extract_compounds_and_concentrations()

    assert compounds == ["Fc"]
    assert concentrations == ["10 mM"]


def test_extract_compounds_and_concentrations_ignores_bare_lowercase_m_and_ag_reference_salt(
    ecat_module,
    blank_echem_factory,
):
    obj = blank_echem_factory(ecat_module.echem)
    obj.name = "CP-100-cycles_MeCN_500mVs_100mTBAPF6_gcWE_ptCE_AgAgbf4-5mM"

    compounds, concentrations = obj.extract_compounds_and_concentrations()

    assert compounds == []
    assert concentrations == []

    obj.name = "CP-100-cycles_MeCN_500mVs_100mMTBAPF6_gcWE_ptCE_AgAgbf4-5mM"
    compounds, concentrations = obj.extract_compounds_and_concentrations()

    assert compounds == ["TBAPF6"]
    assert concentrations == ["100 mM"]
