import pandas as pd


def test_cp_cycle_info_uses_numpy_trapezoid_when_trapz_is_unavailable(
    ecat_module,
    monkeypatch,
):
    cp_obj = ecat_module.cp.__new__(ecat_module.cp)
    cp_obj.data = pd.DataFrame(
        {
            "Time": [0.0, 1.0, 2.0, 3.0, 4.0],
            "Potential": [0.7, 0.6, 0.0, 0.1, 0.2],
        }
    )
    cp_obj.units = {"Time": "s", "Potential": "V"}
    cp_obj.sample_int = 1.0
    cp_obj.segments = 2
    cp_obj.cathodic_current = -0.005
    cp_obj.anodic_current = 0.005

    monkeypatch.delattr(ecat_module.objects.np, "trapz", raising=False)

    summary = cp_obj.cycle_info({"percent capacity": False})

    assert summary["Cycle"].tolist() == [1]
    assert summary["Discharge Capacity (mA·h)"].iloc[0] > 0
    assert summary["Charge Capacity (mA·h)"].iloc[0] > 0
    assert summary["Discharge Energy (mWh)"].iloc[0] > 0
