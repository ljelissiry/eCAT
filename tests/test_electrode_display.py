"""Shared electrodes are context; differing/missing electrodes stay per object."""
import pytest


def _objects(cv_factory):
    objects = [cv_factory(name="50mVs_sample_a"), cv_factory(name="50mVs_sample_b")]
    for obj in objects:
        obj.working_electrode = "GC"
        obj.reference_electrode = "Ag/AgCl (3 M KCl)"
        obj.counter_electrode = "Pt"
        obj.reference_label = "Fc/Fc+"
        obj.reference_shift = 0.2
    return objects


@pytest.mark.parametrize("pretty", [True, False])
def test_shared_electrodes_print_once_between_conditions_and_reference(
    ecat_module, cv_factory, capsys, monkeypatch, pretty,
):
    from ecat import plotting
    objects = _objects(cv_factory)
    monkeypatch.setattr(plotting, "display_object_table", lambda table: print("TABLE"))
    table = ecat_module.show_objects(objects, {
        "pretty print": pretty, "return": True, "columns": ["reference label"],
    })
    output = capsys.readouterr().out
    line = "[Electrodes] WE: GC, RE: Ag/AgCl (3 M KCl), CE: Pt"
    assert output.count(line) == 1
    assert output.index("[Conditions]") < output.index(line) < output.index("[Reference]")
    assert "Label: Fc/Fc+" in output
    if pretty:
        assert output.index(line) < output.index("TABLE")
    assert not {"Working Electrode", "Reference Electrode", "Counter Electrode"} & set(table.columns)
    assert all(obj.reference_electrode == "Ag/AgCl (3 M KCl)" for obj in objects)


def test_differing_electrode_details_are_automatic_columns(ecat_module, cv_factory, capsys):
    objects = _objects(cv_factory)
    objects[1].reference_electrode = "Ag/AgCl (sat. KCl)"
    table, _ = ecat_module.build_object_table(objects)
    output = capsys.readouterr().out
    assert "[Electrodes] WE: GC, CE: Pt" in output
    assert table["Reference Electrode"].tolist() == ["Ag/AgCl (3 M KCl)", "Ag/AgCl (sat. KCl)"]
    assert "Working Electrode" not in table and "Counter Electrode" not in table
    assert "Replicate" not in table  # The electrodes distinguish these entries.


@pytest.mark.parametrize("missing", [None, "", "  ", float("nan")])
def test_missing_value_prevents_shared_claim(ecat_module, cv_factory, capsys, missing):
    objects = _objects(cv_factory)
    objects[1].working_electrode = missing
    table, _ = ecat_module.build_object_table(objects)
    line = next(line for line in capsys.readouterr().out.splitlines() if line.startswith("[Electrodes]"))
    assert "WE:" not in line
    assert table["Working Electrode"].tolist() == ["GC", "unspecified"]


def test_all_missing_electrodes_are_not_a_shared_setup(ecat_module, cv_factory, capsys):
    objects = [cv_factory(), cv_factory()]
    table, _ = ecat_module.build_object_table(objects)
    assert "[Electrodes]" not in capsys.readouterr().out
    assert not any("Electrode" in col for col in table.columns)


@pytest.mark.parametrize("columns", [["WE", "RE", "CE"], "all"])
def test_explicit_shared_electrode_columns_remain_additive(ecat_module, cv_factory, capsys, columns):
    objects = _objects(cv_factory)
    table, _ = ecat_module.build_object_table(objects, {"display columns": columns})
    assert "[Electrodes]" in capsys.readouterr().out
    assert table["Working Electrode"].tolist() == ["GC", "GC"]
    assert table["Counter Electrode"].tolist() == ["Pt", "Pt"]


def test_hidden_conditions_keep_electrode_information_in_table(ecat_module, cv_factory, capsys):
    table, _ = ecat_module.build_object_table(_objects(cv_factory), {"print conditions": False})
    assert capsys.readouterr().out == ""
    assert table["Working Electrode"].tolist() == ["GC", "GC"]
    assert "Reference Electrode" in table and "Counter Electrode" in table


def test_plain_output_also_identifies_differing_electrodes(ecat_module, cv_factory, capsys):
    objects = _objects(cv_factory)
    objects[1].working_electrode = "Au"
    ecat_module.show_objects(objects, {"pretty print": False})
    output = capsys.readouterr().out
    lines = [line for line in output.splitlines() if line.startswith(("[0]", "[1]"))]
    assert "WE: GC" in lines[0]
    assert "WE: Au" in lines[1]


def test_group_summaries_resolve_electrodes_per_group(ecat_module, cv_factory, capsys):
    first, second = _objects(cv_factory), _objects(cv_factory)
    for obj in second:
        obj.working_electrode = "Au"
    ecat_module.show_groups([first, second], {"pretty print": False})
    lines = [line for line in capsys.readouterr().out.splitlines() if line.startswith("[Electrodes]")]
    assert len(lines) == 2
    assert "WE: GC" in lines[0] and "WE: Au" in lines[1]


def test_available_columns_query_is_still_quiet(ecat_module, cv_factory, capsys):
    columns = ecat_module.show_objects(_objects(cv_factory), {"columns": "available"})
    assert "working electrode" in columns
    assert capsys.readouterr().out == ""


def test_one_object_prints_its_known_electrodes(ecat_module, cv_factory, capsys):
    ecat_module.show_objects(_objects(cv_factory)[:1], {"pretty print": False})
    assert "[Electrodes] WE: GC, RE: Ag/AgCl (3 M KCl), CE: Pt" in capsys.readouterr().out
