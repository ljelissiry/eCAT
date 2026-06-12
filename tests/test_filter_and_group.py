from datetime import datetime, timedelta

import pytest


def test_filter_supports_last_replicate_lookup(ecat_module, cv_factory):
    base_time = datetime(2026, 4, 16, 12, 0, 0)

    rep1 = cv_factory(name="50mVs_sample_CO2_MeCN_10mM_Fc_run01")
    rep1.timestamp = base_time

    rep2 = cv_factory(name="50mVs_sample_CO2_MeCN_10mM_Fc_run02")
    rep2.timestamp = base_time + timedelta(minutes=5)

    other = cv_factory(name="100mVs_sample_N2_DMF_5mM_Fc_run01")
    other.timestamp = base_time + timedelta(minutes=10)

    filtered = ecat_module.filter(
        [rep1, rep2, other],
        {"replicate": -1},
        {"print": False},
    )

    assert filtered == [rep2, other]


def test_filter_replicate_minus_one_preserves_nested_group_shape(ecat_module, cv_factory):
    base_time = datetime(2026, 4, 16, 12, 0, 0)

    a1 = cv_factory(name="50mVs_sample_CO2_MeCN_10mM_Fc_run01")
    a1.timestamp = base_time
    a2 = cv_factory(name="50mVs_sample_CO2_MeCN_10mM_Fc_run02")
    a2.timestamp = base_time + timedelta(minutes=5)

    b1 = cv_factory(name="75mVs_sample_N2_DMF_5mM_Fc_run01")
    b1.timestamp = base_time + timedelta(minutes=10)
    b2 = cv_factory(name="75mVs_sample_N2_DMF_5mM_Fc_run02")
    b2.timestamp = base_time + timedelta(minutes=15)

    filtered = ecat_module.filter(
        [[a1, a2], [b1, b2]],
        {"replicate": -1},
        {"print": False},
    )

    assert filtered == [[a2], [b2]]


def test_filter_accepts_dataclass_options(ecat_module, cv_factory):
    co2 = cv_factory(name="50mVs_sample_CO2_MeCN_10mM_Fc_run01")
    n2 = cv_factory(name="100mVs_sample_N2_DMF_5mM_Fc_run01")
    options = ecat_module.FilterOptions.from_options({"print": False})

    filtered = ecat_module.filter([co2, n2], {"gas": "CO2"}, options)

    assert filtered == [co2]


def test_filter_concentrations_accepts_spaced_and_compact_strings(
    ecat_module,
    cv_factory,
):
    fc_100 = cv_factory(name="50mVs_sample_CO2_MeCN_100mM_Fc_run01")
    fc_10 = cv_factory(name="50mVs_sample_CO2_MeCN_10mM_Fc_run01")

    spaced = ecat_module.filter(
        [fc_100, fc_10],
        {"concentrations": "100 mM"},
        {"print": False},
    )
    compact = ecat_module.filter(
        [fc_100, fc_10],
        {"concentrations": "100mM"},
        {"print": False},
    )

    assert spaced == [fc_100]
    assert compact == [fc_100]


def test_filter_species_accepts_concentration_plus_compound_strings(
    ecat_module,
    cv_factory,
):
    fc_100 = cv_factory(name="50mVs_sample_CO2_MeCN_100mM_Fc_run01")
    fc_10 = cv_factory(name="50mVs_sample_CO2_MeCN_10mM_Fc_run01")
    phoh_100 = cv_factory(name="50mVs_sample_CO2_MeCN_100mM_PhOH_run01")

    spaced = ecat_module.filter(
        [fc_100, fc_10, phoh_100],
        {"species": "100 mM Fc"},
        {"print": False},
    )
    compact = ecat_module.filter(
        [fc_100, fc_10, phoh_100],
        {"species": "100mM Fc"},
        {"print": False},
    )
    compound_only = ecat_module.filter(
        [fc_100, fc_10, phoh_100],
        {"species": "Fc"},
        {"print": False},
    )

    assert spaced == [fc_100]
    assert compact == [fc_100]
    assert compound_only == [fc_100, fc_10]


def test_filter_species_accepts_l_unit_compound_strings(
    ecat_module,
    cv_factory,
):
    one_l = cv_factory(name="50mVs_sample_CO2_MeCN_1L_Fc_run01")
    two_l = cv_factory(name="50mVs_sample_CO2_MeCN_2L_Fc_run01")

    spaced = ecat_module.filter(
        [one_l, two_l],
        {"species": "1 L Fc"},
        {"print": False},
    )
    compact = ecat_module.filter(
        [one_l, two_l],
        {"species": "1L Fc"},
        {"print": False},
    )

    assert one_l.concentrations == ["1 L"]
    assert spaced == [one_l]
    assert compact == [one_l]


def test_filter_species_accepts_mole_fraction_x_compound_strings(
    ecat_module,
    cv_factory,
):
    d2o_08 = cv_factory(name="50mVs_sample_CO2_MeCN_0.8xD2O_run01")
    d2o_10 = cv_factory(name="50mVs_sample_CO2_MeCN_1xD2O_run01")

    spaced = ecat_module.filter(
        [d2o_08, d2o_10],
        {"species": "0.8 x D2O"},
        {"print": False},
    )
    compact = ecat_module.filter(
        [d2o_08, d2o_10],
        {"species": "0.8xD2O"},
        {"print": False},
    )

    assert d2o_08.concentrations == ["0.8 x"]
    assert spaced == [d2o_08]
    assert compact == [d2o_08]


def test_group_concentrations_accepts_mole_fraction_x(ecat_module, cv_factory):
    d2o_08 = cv_factory(name="50mVs_sample_CO2_MeCN_0.8xD2O_run01")
    d2o_10 = cv_factory(name="50mVs_sample_CO2_MeCN_1xD2O_run01")

    grouped = ecat_module.group(
        [d2o_10, d2o_08],
        "concentrations",
        {"print": False},
    )

    assert [[obj.concentrations for obj in grp] for grp in grouped] == [
        [["1 x"]],
        [["0.8 x"]],
    ]


def test_filter_unknown_option_suggests_logic(ecat_module, cv_factory):
    obj = cv_factory(name="50mVs_sample_CO2_MeCN_10mM_Fc_run01")

    with pytest.raises(ValueError, match="logic"):
        ecat_module.filter([obj], {"gas": "CO2"}, {"logc": "AND"})


def test_get_available_filter_values_includes_last_replicate_marker(ecat_module, cv_factory):
    base_time = datetime(2026, 4, 16, 12, 0, 0)

    rep1 = cv_factory(name="50mVs_sample_CO2_MeCN_10mM_Fc_run01")
    rep1.timestamp = base_time
    rep2 = cv_factory(name="50mVs_sample_CO2_MeCN_10mM_Fc_run02")
    rep2.timestamp = base_time + timedelta(minutes=5)

    values = ecat_module.get_available_filter_values([rep1, rep2], keys=["replicate"])

    assert values["replicate"] == [1, 2, -1]


def test_sort_and_group_smoke_behavior(ecat_module, cv_factory):
    co2_a = cv_factory(name="50mVs_sample_CO2_MeCN_10mM_Fc_run01")
    co2_b = cv_factory(name="100mVs_sample_CO2_MeCN_10mM_Fc_run02")
    n2 = cv_factory(name="75mVs_sample_N2_DMF_10mM_Fc_run01")

    grouped = ecat_module.sort_and_group(
        [n2, co2_b, co2_a],
        sort_keys=["gas", "scan rate"],
        group_keys="gas",
        options={"print": False},
    )

    assert len(grouped) == 2
    assert [obj.gas for obj in grouped[0]] == ["CO2", "CO2"]
    assert [obj.scan_rate for obj in grouped[0]] == [0.05, 0.1]
    assert [obj.gas for obj in grouped[1]] == ["N2"]


def test_filter_pretty_print_false_uses_plain_output(ecat_module, cv_factory, monkeypatch, capsys):
    obj = cv_factory(name="50mVs_sample_CO2_MeCN_10mM_Fc_run01")

    def fail_pretty_table(_df, options=None):
        raise AssertionError("pretty table should not be used")

    monkeypatch.setattr(ecat_module, "display_object_table", fail_pretty_table)

    filtered = ecat_module.filter(
        [obj],
        {"gas": "CO2"},
        {"print": True, "pretty print": False},
    )

    assert filtered == [obj]
    assert obj.name in capsys.readouterr().out


def test_sort_and_group_pretty_print_false_flows_to_group_printing(
    ecat_module,
    cv_factory,
    monkeypatch,
    capsys,
):
    co2 = cv_factory(name="50mVs_sample_CO2_MeCN_10mM_Fc_run01")
    n2 = cv_factory(name="75mVs_sample_N2_DMF_10mM_Fc_run01")

    def fail_pretty_table(_df, options=None):
        raise AssertionError("pretty table should not be used")

    monkeypatch.setattr(ecat_module, "display_object_table", fail_pretty_table)

    grouped = ecat_module.sort_and_group(
        [n2, co2],
        sort_keys="gas",
        group_keys="gas",
        options={"print": True, "pretty print": False},
    )

    assert len(grouped) == 2
    printed = capsys.readouterr().out
    assert "### Group" in printed
    assert co2.name in printed
    assert n2.name in printed


def test_sort_and_group_print_does_not_inject_group_key_conditions(
    ecat_module,
    cv_factory,
    capsys,
):
    low = cv_factory(name="50mVs_sample_CO2_MeCN_10mM_Fc_run01")
    high = cv_factory(name="100mVs_sample_CO2_MeCN_10mM_Fc_run02")
    low.ir_comp_resistance = 50.0
    high.ir_comp_resistance = 100.0

    grouped = ecat_module.sort_and_group(
        [high, low],
        sort_keys="ir comp resistance",
        group_keys="ir comp resistance",
        options={"print": True, "pretty print": False},
    )

    assert len(grouped) == 2
    printed = capsys.readouterr().out
    assert "IR Comp Resistance: 50" not in printed
    assert "IR Comp Resistance: 100" not in printed


def test_show_groups_rejects_group_keys_option(ecat_module, cv_factory):
    obj = cv_factory(name="50mVs_sample_CO2_MeCN_10mM_Fc_run01")

    with pytest.raises(ValueError, match="does not accept 'group keys'"):
        ecat_module.show_groups(
            [[obj]],
            {"group keys": "gas", "pretty print": False},
        )


def test_show_objects_does_not_insert_blank_line_after_conditions(
    ecat_module,
    cv_factory,
    capsys,
):
    obj = cv_factory(name="50mVs_sample_CO2_MeCN_10mM_Fc_run01")

    ecat_module.show_objects(
        [obj],
        {
            "columns": ["gas", "scan rate"],
            "pretty print": False,
        },
    )

    printed = capsys.readouterr().out
    assert "[Conditions]" in printed
    assert "\n\n[0]" not in printed


def test_show_objects_conditions_only_include_ir_comp_percent_by_default(
    ecat_module,
    cv_factory,
    capsys,
):
    first = cv_factory(name="50mVs_sample_CO2_MeCN_10mM_Fc_run01")
    second = cv_factory(name="100mVs_sample_CO2_MeCN_10mM_Fc_run02")
    for obj in (first, second):
        obj.ir_comp_resistance = 80.0
        obj.ir_uncomp_resistance = 20.0
        obj.ir_comp_percent = 80.0

    ecat_module.show_objects(
        [first, second],
        {"pretty print": False},
    )

    printed = capsys.readouterr().out
    conditions_line = next(line for line in printed.splitlines() if line.startswith("[Conditions]"))
    assert "IR Comp Percent: 80 %" in conditions_line
    assert "IR Comp Resistance" not in conditions_line
    assert "IR Uncomp Resistance" not in conditions_line


def test_show_objects_conditions_include_requested_ir_comp_columns(
    ecat_module,
    cv_factory,
    capsys,
):
    first = cv_factory(name="50mVs_sample_CO2_MeCN_10mM_Fc_run01")
    second = cv_factory(name="100mVs_sample_CO2_MeCN_10mM_Fc_run02")
    for obj in (first, second):
        obj.ir_comp_resistance = 80.0
        obj.ir_uncomp_resistance = 20.0
        obj.ir_comp_percent = 80.0

    ecat_module.show_objects(
        [first, second],
        {
            "pretty print": False,
            "columns": ["ir comp resistance", "ir uncomp resistance", "ir comp percent"],
        },
    )

    printed = capsys.readouterr().out
    conditions_line = next(line for line in printed.splitlines() if line.startswith("[Conditions]"))
    assert "IR Comp Percent: 80 %" in conditions_line
    assert "IR Comp Resistance: 80 ohm" in conditions_line
    assert "IR Uncomp Resistance: 20 ohm" in conditions_line


def test_group_summary_groups_concentrations_like_group(ecat_module, cv_factory):
    low_50 = cv_factory(name="50mVs_sample_CO2_MeCN_10mM_Fc_run01")
    low_100 = cv_factory(name="100mVs_sample_CO2_MeCN_10mM_Fc_run02")
    high = cv_factory(name="75mVs_sample_CO2_MeCN_100mM_Fc_run01")

    grouped = ecat_module.group(
        [low_50, high, low_100],
        "concentrations",
        {"print": False},
    )
    summary = ecat_module.group_summary(
        [low_50, high, low_100],
        group_keys="concentrations",
        options={"print": False},
    )

    assert summary["Concentrations"].tolist() == [
        ecat_module._format_group_summary_value(
            ecat_module.get_sort_group_dict()["concentrations"](grp[0])
        )
        for grp in grouped
    ]
    assert summary["N Objects"].tolist() == [2, 1]
    assert summary["N CV"].tolist() == [2, 1]


def test_group_summary_accepts_group_keys_option_and_param_precedence(
    ecat_module,
    cv_factory,
):
    co2_low = cv_factory(name="50mVs_sample_CO2_MeCN_10mM_Fc_run01")
    co2_high = cv_factory(name="100mVs_sample_CO2_MeCN_100mM_Fc_run01")
    n2 = cv_factory(name="75mVs_sample_N2_MeCN_10mM_Fc_run01")

    option_summary = ecat_module.group_summary(
        [co2_low, co2_high, n2],
        options={"group keys": "gas", "print": False},
    )
    precedence_summary = ecat_module.group_summary(
        [co2_low, co2_high, n2],
        group_keys="concentrations",
        options={"group keys": "gas", "print": False},
    )

    assert option_summary["Gas"].tolist() == ["CO2", "N2"]
    assert option_summary["N Objects"].tolist() == [2, 1]
    assert "Gas" not in precedence_summary.columns
    assert precedence_summary["Concentrations"].tolist() == ["[0.01]", "[0.1]"]


def test_group_summary_requested_columns_show_distinct_count_and_values(
    ecat_module,
    cv_factory,
):
    slow = cv_factory(name="50mVs_sample_CO2_MeCN_10mM_Fc_run01")
    fast = cv_factory(name="100mVs_sample_CO2_MeCN_10mM_Fc_run02")

    summary = ecat_module.group_summary(
        [slow, fast],
        group_keys="concentrations",
        options={"columns": ["scan rate"], "print": False},
    )

    assert summary["N Scan Rate"].tolist() == [2]
    assert summary["Scan Rate Values"].tolist() == ["[0.05, 0.1]"]


def test_group_summary_summarizes_pregrouped_and_flat_inputs(
    ecat_module,
    cv_factory,
):
    co2 = cv_factory(name="50mVs_sample_CO2_MeCN_10mM_Fc_run01")
    n2 = cv_factory(name="75mVs_sample_N2_MeCN_10mM_Fc_run01")
    grouped = ecat_module.sort_and_group(
        [n2, co2],
        sort_keys="gas",
        group_keys="gas",
        options={"print": False},
    )

    grouped_summary = ecat_module.group_summary(grouped, options={"print": False})
    flat_summary = ecat_module.group_summary([co2, n2], options={"print": False})

    assert grouped_summary["Group"].tolist() == [0, 1]
    assert grouped_summary["N Objects"].tolist() == [1, 1]
    assert flat_summary["N Objects"].tolist() == [2]
    assert "N Gas" in flat_summary.columns
    assert flat_summary["Gas Values"].tolist() == ["[CO2, N2]"]


def test_group_summary_print_modes(ecat_module, cv_factory, monkeypatch, capsys):
    obj = cv_factory(name="50mVs_sample_CO2_MeCN_10mM_Fc_run01")
    displayed = []

    monkeypatch.setattr(ecat_module, "display_object_table", lambda df, options=None: displayed.append(df))

    returned = ecat_module.group_summary([obj])

    assert displayed == [returned]
    assert "N Objects" in returned.columns

    ecat_module.group_summary([obj], options={"print": False})

    assert len(displayed) == 1
    assert capsys.readouterr().out == ""


def test_build_object_table_labels_replicates_before_reference_columns(
    ecat_module,
    cv_factory,
):
    rep1 = cv_factory(name="100mVs_sample_CO2_MeCN_10mM_Fc_run01")
    rep2 = cv_factory(name="100mVs_sample_CO2_MeCN_10mM_Fc_run02")
    rep1.reference_shift = 0.401
    rep2.reference_shift = 0.405
    rep1.reference_mode = rep2.reference_mode = "folder"
    rep1.reference_label = rep2.reference_label = "Fc/Fc+"

    table, _meta = ecat_module.build_object_table(
        [rep1, rep2],
        {"columns": ["reference shift", "reference mode"]},
    )

    assert table["Replicate"].tolist() == ["1", "2"]
    assert table["Reference Shift"].tolist() == ["0.401", "0.405"]
    assert list(table.columns).index("Replicate") < list(table.columns).index("Reference Shift")


def test_build_object_table_places_txt_stats_before_replicate(ecat_module, repo_root):
    filepath = (
        repo_root
        / "tests"
        / "tmp_real_examples"
        / "DPV_MeCN_CO2_0.1MTBAPF6_3mMFc_1mMFe-tpyPY2Me_-0.7_to_-1.2V.txt"
    )
    rep1 = ecat_module.echem.from_file(str(filepath), {})
    rep2 = ecat_module.echem.from_file(str(filepath), {})

    table, _meta = ecat_module.build_object_table(
        [rep1, rep2],
        {"columns": ["sample width"], "print conditions": False},
    )

    assert table["Replicate"].tolist() == ["1", "2"]
    assert list(table.columns).index("Sample Width") < list(table.columns).index("Replicate")


def test_build_object_table_rounds_scalar_numeric_txt_stats(ecat_module, repo_root):
    filepath = (
        repo_root
        / "tests"
        / "tmp_real_examples"
        / "DPV_MeCN_CO2_0.1MTBAPF6_3mMFc_1mMFe-tpyPY2Me_-0.7_to_-1.2V.txt"
    )
    obj = ecat_module.echem.from_file(str(filepath), {})

    table, _meta = ecat_module.build_object_table(
        [obj],
        {
            "columns": [
                "amplitude",
                "pulse width",
                "sample width",
                "pulse period",
            ],
            "print conditions": False,
            "sig figs": 3,
        },
    )

    assert table["Amplitude"].iloc[0] == "10 mV"
    assert table["Pulse Width"].iloc[0] == "50 ms"
    assert table["Sample Width"].iloc[0] == "16.7 ms"
    assert table["Pulse Period"].iloc[0] == "500 ms"

    with pytest.raises(ValueError, match="sample width"):
        ecat_module.build_object_table(
            [obj],
            {"columns": ["sample width (s)"], "print conditions": False},
        )


def test_build_object_table_rounds_numeric_lists_for_display(
    ecat_module,
    cv_factory,
):
    potential = [-2.19818115234375, -1.49993896484375] * 11
    obj = cv_factory(
        potential=potential[:21],
    )

    table, _meta = ecat_module.build_object_table(
        [obj],
        {"columns": ["scan window"], "sig figs": 4},
    )

    scan_window = table["Scan Window"].iloc[0]
    assert scan_window == "[-2.2, -1.5]"
    assert "15234375" not in scan_window


def test_cv_txt_stats_supports_potential_rounding_option(ecat_module, cv_factory):
    potential = [-2.19818115234375, -1.49993896484375] * 11
    obj = cv_factory(potential=potential[:21])

    assert obj.stats()["low E"] == -2.19818115234375

    default_stats = obj.txt_stats({})
    assert default_stats["scan window"] == "[-2.2, -1.5]"

    exactish_display_stats = obj.txt_stats({"potential rounding": None, "sig figs": 6})
    assert exactish_display_stats["scan window"] == "[-2.19818, -1.49994]"
