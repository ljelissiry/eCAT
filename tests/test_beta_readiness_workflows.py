from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest


def test_beta_object_accessors_and_summaries_are_notebook_friendly(cv_factory):
    obj = cv_factory(name="50mVs_sample_CO2_MeCN_10mM_Fc_run01")

    x = obj.x()
    y = obj.y()
    xy = obj.xy()
    stats = obj.stats()
    info = obj.info()

    assert x.name == "Potential"
    assert y.name == "Current"
    assert isinstance(xy, tuple)
    assert len(xy) == 2
    np.testing.assert_allclose(xy[0], x.to_numpy())
    np.testing.assert_allclose(xy[1], y.to_numpy())
    assert stats["gas"] == "CO2"
    assert stats["solvent"] == "MeCN"
    assert stats["segments"] == 2
    assert info["name"] == obj.name
    assert info["reference mode"] == "none"
    assert info["gas"] == "CO2"


@pytest.mark.parametrize(
    ("filename", "expected_class", "expected_x", "expected_y"),
    [
        ("ch_ca_tiny.txt", "ca", "Time", "Current"),
        ("ch_cp_tiny.txt", "cp", "Time", "Potential"),
        ("eclab_gcpl_tiny.txt", "cp", "Time", "Potential"),
    ],
)
def test_beta_limited_ca_cp_objects_load_and_plot_smoke(
    ecat_module,
    fixtures_dir,
    filename,
    expected_class,
    expected_x,
    expected_y,
):
    obj = ecat_module.echem.from_file(str(fixtures_dir / filename), {})

    assert type(obj).__name__ == expected_class
    assert obj.x().name == expected_x
    assert obj.y().name == expected_y

    ax = obj.plot({"legend": False, "title": False})

    assert len(ax.lines) == 1
    assert ax.get_xlabel()
    assert ax.get_ylabel()
    plt.close(ax.figure)


def test_beta_figure_export_smoke(cv_factory, tmp_path):
    obj = cv_factory()

    ax = obj.plot({"legend": False, "title": False})
    output = tmp_path / "cv_smoke.png"
    ax.figure.savefig(output)

    assert output.exists()
    assert output.stat().st_size > 0
    plt.close(ax.figure)


def test_beta_save_data_exports_csv_with_units(ecat_module, cv_factory, tmp_path, capsys):
    obj = cv_factory(name="50mVs_sample_CO2_MeCN_10mM_Fc_run01")

    exported = ecat_module.save_data(
        [obj],
        {
            "folder path": str(tmp_path),
            "file name": "processed_cv",
        },
    )

    output = tmp_path / "processed_cv.csv"
    captured = capsys.readouterr()
    csv = pd.read_csv(output, header=[0, 1])

    assert output.exists()
    assert not exported.empty
    assert not csv.empty
    assert "Saved 1 echem objects" in captured.out
    assert Path(obj.name).name in csv.columns.get_level_values(0)
    assert "Potential (V)" in csv.columns.get_level_values(1)
    assert "Current (A)" in csv.columns.get_level_values(1)


def test_beta_save_data_applies_y_unit_only_to_y_columns(
    ecat_module,
    cv_factory,
    tmp_path,
):
    obj = cv_factory(name="50mVs_sample_CO2_MeCN_10mM_Fc_run01")

    ecat_module.save_data(
        [obj],
        {
            "folder path": str(tmp_path),
            "file name": "processed_cv_uA",
            "y unit": "uA",
        },
    )

    csv = pd.read_csv(tmp_path / "processed_cv_uA.csv", header=[0, 1])
    potential = csv.xs("Potential (V)", axis=1, level=1).iloc[:, 0]
    current = csv.xs("Current (μA)", axis=1, level=1).iloc[:, 0]

    np.testing.assert_allclose(potential, obj.data["Potential"])
    np.testing.assert_allclose(current, obj.data["Current"] * 1e6)
