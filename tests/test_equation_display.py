def _sample_equation():
    return {
        "symbolic latex": "symbolic_latex",
        "resolved latex": "resolved_latex",
        "compact latex": "compact_latex",
        "definitions latex": "definitions_latex",
        "symbolic": "symbolic_text",
        "resolved": "resolved_text",
        "compact": "compact_text",
        "definitions": "definitions_text",
    }


def test_display_analysis_equation_plain_text_fallback(ecat_module, monkeypatch, capsys):
    monkeypatch.setattr(ecat_module, "display", None)
    monkeypatch.setattr(ecat_module, "Math", None)

    returned = ecat_module._display_analysis_equation(
        "title_latex",
        "Example equation",
        _sample_equation(),
        resolved=True,
        compact=True,
    )

    output = capsys.readouterr().out
    assert returned["symbolic"] == "symbolic_text"
    assert "[Example equation]" in output
    assert "symbolic_text" in output
    assert "resolved_text" in output
    assert "compact_text" in output
    assert "definitions_text" in output


def test_display_analysis_equation_notebook_path(ecat_module, monkeypatch):
    displayed = []

    class FakeMath:
        def __init__(self, text):
            self.text = text

    monkeypatch.setattr(ecat_module, "Math", FakeMath)
    monkeypatch.setattr(ecat_module, "display", lambda math_obj: displayed.append(math_obj.text))

    ecat_module._display_analysis_equation(
        "title_latex",
        "Example equation",
        _sample_equation(),
        resolved=True,
        compact=False,
    )

    assert displayed == [
        "title_latex",
        "symbolic_latex",
        "resolved_latex",
        "definitions_latex",
    ]


def test_fowa_display_wrapper_returns_equation_dict(ecat_module, monkeypatch):
    monkeypatch.setattr(ecat_module, "display", None)
    monkeypatch.setattr(ecat_module, "Math", None)

    equation = ecat_module._display_fowa_kobs_equation(
        {"num electrons": 2, "turnover electrons": 1, "sigma": 1},
        resolved=False,
        compact=False,
    )

    assert {
        "symbolic latex",
        "resolved latex",
        "compact latex",
        "definitions latex",
        "symbolic",
        "resolved",
        "compact",
        "definitions",
    }.issubset(equation)


def test_sevcik_equation_formatter_scan_rate_mode(ecat_module):
    equation = ecat_module._format_sevcik_diffusion_equation(
        mode="scan rate",
        num_electrons=1,
        temperature=298,
        electrode_area=0.071,
        concentration=1e-5,
        scan_rate=None,
        scan_dependence=0.5,
    )

    assert "D = (R * T / (F * n)^3)" in equation["symbolic"]
    assert "i_p = m * v^0.5 + b" in equation["symbolic"]
    assert "C = 1e-05 mol/cm^3" in equation["definitions"]
    assert "S = 0.071 cm^2" in equation["definitions"]


def test_sevcik_equation_formatter_concentration_mode(ecat_module):
    equation = ecat_module._format_sevcik_diffusion_equation(
        mode="concentration",
        num_electrons=2,
        temperature=298,
        electrode_area=0.071,
        concentration=None,
        scan_rate=0.1,
        scan_dependence=0.5,
    )

    assert "D = (R * T / (F^2 * n^3 * v * S^2))" in equation["symbolic"]
    assert "i_p = m * C + b" in equation["symbolic"]
    assert "v = 0.1 V/s" in equation["definitions"]
    assert "n = 2" in equation["definitions"]
