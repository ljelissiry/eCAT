import numpy as np
import pandas as pd


def test_display_table_uses_captioned_styler_for_rich_display(ecat_module, monkeypatch, capsys):
    displayed = []

    def capture_display(obj):
        displayed.append(obj)

    monkeypatch.setattr(ecat_module, "display", capture_display)

    table = pd.DataFrame({"Field": ["Model"], "Value": ["linear"]})

    returned = ecat_module._display_table(
        table,
        {"pretty print": True},
        title="Fit Model",
        index=False,
    )

    assert returned is table
    assert capsys.readouterr().out == ""
    assert len(displayed) == 1
    assert getattr(displayed[0], "caption", None) == "Fit Model"
    assert list(displayed[0].data.columns) == ["Field", "Value"]


def test_display_table_prints_title_and_table_for_plain_display(ecat_module, monkeypatch, capsys):
    monkeypatch.setattr(ecat_module, "display", None)

    table = pd.DataFrame({"Field": ["Model"], "Value": ["linear"]})

    ecat_module._display_table(
        table,
        {"pretty print": True},
        title="Fit Model",
        index=False,
    )

    printed = capsys.readouterr().out
    assert printed.startswith("Fit Model:\n")
    assert "Field" in printed
    assert "Value" in printed
    assert "linear" in printed


def test_fit_model_rich_print_uses_table_captions(ecat_module, monkeypatch, capsys):
    displayed = []

    def capture_display(obj):
        displayed.append(obj)

    monkeypatch.setattr(ecat_module, "display", capture_display)

    x = np.asarray([0.32, 0.6, 1.0, 1.6, 2.2, 2.8])
    y = 10.0 + 3.0 * x ** 2.0

    ecat_module.fit_model(
        x,
        y,
        model="power offset",
        options={
            "print": True,
            "plot": False,
            "fit init": {"b": 1.0, "A": 1.0, "n": 1.0},
            "fit bounds": {"b": [0, np.inf], "A": [0, np.inf], "n": [0, 8]},
        },
    )

    printed = capsys.readouterr().out
    captions = [getattr(obj, "caption", None) for obj in displayed]

    assert "Fit Model Details:" not in printed
    assert "Fit Model Parameters:" not in printed
    assert "Fit Model Details" in captions
    assert "Fit Model Parameters" in captions
