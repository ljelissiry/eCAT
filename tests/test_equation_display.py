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


def test_reversibility_equation_bundles_use_latex_symbols(ecat_module):
    bulk = dict(ecat_module._bulk_reversibility_equation_sections())
    surface = ecat_module._surface_reversibility_equation_bundle()
    coverage = ecat_module._surface_coverage_equation_bundle()

    assert r"\Lambda" in bulk["Nicholson Peak-Separation Conversion"][
        "resolved latex"
    ]
    assert r"\Lambda" in bulk["Matsuda-Ayabe Classification"][
        "symbolic latex"
    ]
    assert r"\Gamma" in surface["symbolic latex"]
    assert r"\Gamma" in coverage["symbolic latex"]
    assert r"n_{\mathrm{loading}}" in coverage["resolved latex"]
    assert coverage["definitions latex"] == ""
    assert coverage["definitions"] == ""


def test_bulk_reversibility_equation_sections_have_labeled_latex(ecat_module):
    sections = dict(
        ecat_module._bulk_reversibility_equation_sections(
            include_sevcik=True,
            include_rate=True,
            include_irreversible=True,
        )
    )

    assert set(sections) == {
        "Nicholson Peak-Separation Conversion",
        "Matsuda-Ayabe Classification",
        "Sevcik Diffusion Estimate",
        "Electron-Transfer Rate Conversion",
        "Irreversible-Asymptote Verification",
    }
    assert r"n\Delta E_p" in sections["Nicholson Peak-Separation Conversion"][
        "symbolic latex"
    ]
    assert r"\Lambda" in sections["Matsuda-Ayabe Classification"][
        "symbolic latex"
    ]
    assert r"D=" in sections["Sevcik Diffusion Estimate"]["symbolic latex"]
    assert r"k^0" in sections["Electron-Transfer Rate Conversion"][
        "symbolic latex"
    ]
    assert r"E_{p/2}" in sections["Irreversible-Asymptote Verification"][
        "symbolic latex"
    ]


def test_pretty_table_headers_format_reversibility_symbols(ecat_module):
    assert ecat_module._pretty_table_header_html_label("E1/2 / V") == "E<sub>1/2</sub> / V"
    assert ecat_module._pretty_table_header_html_label("Ep / V") == "E<sub>p</sub> / V"
    assert ecat_module._pretty_table_header_html_label("ipc / A") == "i<sub>p,c</sub> / A"
    assert ecat_module._pretty_table_header_html_label("ipa / A") == "i<sub>p,a</sub> / A"
    assert ecat_module._pretty_table_header_html_label("|ipc| / A") == "|i<sub>p,c</sub>| / A"
    assert ecat_module._pretty_table_header_html_label("|ipa| / A") == "|i<sub>p,a</sub>| / A"
    assert ecat_module._pretty_table_header_html_label("|ipa/ipc|") == (
        "|i<sub>p,a</sub>/i<sub>p,c</sub>|"
    )
    assert ecat_module._pretty_table_header_html_label("n Delta Ep / mV") == (
        "nΔE<sub>p</sub> / mV"
    )
    assert ecat_module._pretty_table_header_html_label("Gamma slope / mol cm^-2") == (
        "Γ<sub>slope</sub> / mol cm<sup>-2</sup>"
    )


def test_reversibility_rich_evidence_formats_embedded_symbols(ecat_module):
    rich = ecat_module._rich_scientific_text(
        "Lambda=7.17; psi=4.04; |ipa/ipc|=0.98; k0=0.1 cm^2 s^-1"
    )

    assert "Λ=7.17" in rich
    assert "ψ=4.04" in rich
    assert "|i<sub>p,a</sub>/i<sub>p,c</sub>|=0.98" in rich
    assert "k<sup>0</sup>=0.1" in rich
    assert "cm<sup>2</sup> s<sup>-1</sup>" in rich


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
    )

    assert "D = (R * T / (F^2 * n^3 * v * S^2))" in equation["symbolic"]
    assert "i_p = m * C + b" in equation["symbolic"]
    assert "v = 0.1 V/s" in equation["definitions"]
    assert "n = 2" in equation["definitions"]
