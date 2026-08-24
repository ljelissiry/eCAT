import pandas as pd
import pytest


def test_show_single_pretty_formats_chemical_metadata_html(
    ecat_module,
    cv_factory,
    monkeypatch,
):
    obj = cv_factory(name="100mVs_sample_CO2_MeCN_10mM_Fc_run01")
    obj.gas = "CO2"
    obj.solvent = "MeCN"
    obj.compounds = ["H2O", "TBAPF6"]
    captured = {}

    def capture_display(styled):
        captured.setdefault("html", []).append(styled.to_html())

    monkeypatch.setattr(ecat_module, "display", capture_display)
    ecat_module.show(obj, {"pretty print": True})
    html = "\n".join(captured["html"])
    assert "CO<sub>2</sub>" in html
    assert "H<sub>2</sub>O" in html
    assert "TBAPF<sub>6</sub>" in html


def test_show_single_pretty_formats_reference_label_html(
    ecat_module,
    cv_factory,
    monkeypatch,
):
    obj = cv_factory(name="100mVs_sample_CO2_MeCN_10mM_Fc_run01")
    obj.reference_label = "[Fe(CN)6]3-/4-"
    captured = {}

    def capture_display(styled):
        captured.setdefault("html", []).append(styled.to_html())

    table = ecat_module.show(obj, {"pretty print": False, "return": True})
    assert "Reference Label" in table["Metric"].tolist()

    import ecat.plotting as ecat_plotting

    monkeypatch.setattr(ecat_plotting, "display", capture_display)
    ecat_plotting.display_object_table(
        pd.DataFrame(
            {
                "Metric": ["Reference Label"],
                "Value": ["[Fe(CN)6]3-/4-"],
            }
        )
    )
    html = "\n".join(captured["html"])
    assert "CN)<sub>6</sub>" in html
    assert "<sup>3−/4−</sup>" in html


def test_chemical_formatter_superscripts_common_charge_notation(ecat_module):
    assert (
        ecat_module.format_chemical_formulas("[Fe(CN)6]3-/4-", mode="html")
        == "[Fe(CN)<sub>6</sub>]<sup>3−/4−</sup>"
    )
    assert (
        ecat_module.format_chemical_formulas("Fc/Fc+", mode="html")
        == "Fc/Fc<sup>+</sup>"
    )
    assert (
        ecat_module.format_chemical_formulas("[Co]0/+", mode="unicode")
        == "[Co]⁰/⁺"
    )
    assert (
        ecat_module.format_chemical_formulas("[Fe(CN)6]3−/4−", mode="html")
        == "[Fe(CN)<sub>6</sub>]<sup>3−/4−</sup>"
    )


def test_object_table_displays_compounds_without_list_brackets(
    ecat_module,
    cv_factory,
):
    low_co2 = cv_factory(name="100mVs_sample_MeCN_5%CO2_10mMFc_run01")
    high_co2 = cv_factory(name="100mVs_sample_MeCN_10%CO2_10mMFc_run02")

    table, _meta = ecat_module.build_object_table(
        [low_co2, high_co2],
        {"columns": ["compounds"], "print conditions": False},
    )

    assert table["Compounds"].tolist() == ["5 % CO2", "10 % CO2"]


def test_fit_rate_parses_bracketed_display_table_compounds(ecat_module):
    df = pd.DataFrame(
        {
            "Compounds": ["[5 % CO2]", "[10 % CO2]", "[20 % CO2]"],
            "kobs": [1.0, 2.0, 4.0],
        }
    )

    result = ecat_module.fit_rate(
        df,
        {
            "species": "CO2",
            "metric": "kobs",
            "plot": False,
            "fit": False,
            "print": False,
        },
    )

    assert result.table["x raw"].tolist() == pytest.approx([5.0, 10.0, 20.0])
    assert result.table["x label"].unique().tolist() == ["CO2"]
