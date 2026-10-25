"""Incomplete EC-Lab column selections must fail with actionable guidance."""

import pytest


@pytest.mark.parametrize("loader", ["parse_file_to_result", "from_file"])
@pytest.mark.parametrize(
    "columns, row, missing",
    [
        ("time/s\tI Range\tPwe/W\tRwe/Ohm", "0\t39\t-8e-6\t140", "Potential, Current"),
        ("Pwe/W\tRwe/Ohm", "-8e-6\t140", "Potential, Current"),
        ("time/s\tEwe/V\tI Range", "0\t0.1\t39", "Current"),
        ("time/s\tcontrol/V\tI/mA", "0\t0.1\t0.2", "Potential"),
    ],
)
def test_incomplete_eclab_cv_names_missing_axes_and_reexport_columns(
    ecat_module, tmp_path, loader, columns, row, missing
):
    path = tmp_path / "incomplete_cv.txt"
    path.write_text(
        "EC-Lab ASCII FILE\nNb header lines : 5\n\nCyclic Voltammetry\n"
        f"{columns}\n{row}\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError) as exc:
        getattr(ecat_module.echem, loader)(str(path), {})

    message = str(exc.value)
    assert f"missing required columns: {missing}" in message
    assert "EC-Lab CV" in message
    assert columns.split("\t")[0] in message
    assert "Re-export" in message
    assert "Ewe/V" in message
    assert "I/mA" in message


def test_eclab_cv_with_auxiliary_columns_preserves_measured_axes(ecat_module, tmp_path):
    path = tmp_path / "complete_cv.txt"
    path.write_text(
        "EC-Lab ASCII FILE\nNb header lines : 7\n\nCyclic Voltammetry\n"
        "dE/dt 50\ndE/dt unit mV/s\n"
        "time/s\tI Range\tPwe/W\tRwe/Ohm\tEwe/V\t<I>/mA\n"
        "0\t39\t-8e-6\t140\t0.1\t0.2\n"
        "1\t39\t-6e-6\t150\t0.15\t0.3\n"
        "2\t39\t-4e-6\t160\t0.2\t0.4\n",
        encoding="utf-8",
    )

    obj = ecat_module.echem.from_file(str(path), {})
    assert type(obj).__name__ == "cv"
    assert obj.data["Potential"].tolist() == pytest.approx([0.1, 0.15, 0.2])
    assert obj.data["Current"].tolist() == pytest.approx([0.0002, 0.0003, 0.0004])
    assert obj.units["Current"] == "A"
    assert obj.scan_rate == pytest.approx(0.05)
