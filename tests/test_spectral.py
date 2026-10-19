from array import array
from collections import OrderedDict
import copy
from dataclasses import FrozenInstanceError, fields
import os
import subprocess
import sys
from types import SimpleNamespace
import warnings

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

import ecat.spectral as spectral
from ecat.options import SpectrumOptions
from ecat.results import AnalysisResult
from ecat.spectral import _SpectrumTrace, _resolve_spectrum_traces


def _options(**overrides):
    values = {"plot": False, "print": False}
    values.update(overrides)
    return SpectrumOptions.from_options(values).to_options_dict()


def _timed_object(blank_echem_factory, cls, *, time=None, time_unit="s"):
    if time is None:
        time = np.arange(12, dtype=float) * 0.1
    time = np.asarray(time, dtype=float)
    obj = blank_echem_factory(cls)
    obj.data = pd.DataFrame(
        {
            "Time": time,
            "Current": np.linspace(1.0, 2.0, len(time)),
            "Potential": np.linspace(-0.4, 0.2, len(time)),
        }
    )
    obj.units = {"Time": time_unit, "Current": "A", "Potential": "V"}
    return obj


def _two_segment_cv(blank_echem_factory, ecat_module, *, recorded_time=False):
    forward = np.linspace(-0.3, 0.3, 12)
    reverse = np.linspace(0.3, -0.3, 12)[1:]
    potential = np.concatenate([forward, reverse])
    obj = blank_echem_factory(ecat_module.cv)
    obj.data = pd.DataFrame(
        {
            "Potential": potential,
            "Current": np.sin(np.arange(len(potential), dtype=float)),
        }
    )
    if recorded_time:
        obj.data.insert(0, "Time", 40.0 + np.arange(len(potential)) * 0.2)
    obj.units = {"Potential": "V", "Current": "A", "Time": "s"}
    obj.scan_rate = 0.1
    return obj


def _resolved_signal(
    blank_echem_factory,
    ecat_module,
    values,
    *,
    sample_rate,
    cls=None,
    column="Current",
    unit="A",
    **option_overrides,
):
    values = np.asarray(values, dtype=float)
    cls = ecat_module.ca if cls is None else cls
    obj = blank_echem_factory(cls)
    obj.data = pd.DataFrame(
        {
            "Time": np.arange(len(values), dtype=float) / sample_rate,
            column: values,
        }
    )
    obj.units = {"Time": "s", column: unit}
    options = _options(column=column, **option_overrides)
    traces, warning_messages = _resolve_spectrum_traces(obj, options)
    return traces[0], options, warning_messages


def _plotting_case(obj, **option_overrides):
    options = _options(plot=True, **option_overrides)
    traces, warning_messages = _resolve_spectrum_traces(obj, options)
    computations = [
        spectral._compute_trace(trace, options, warning_messages) for trace in traces
    ]
    result = AnalysisResult(diagnostics={"traces": traces})
    returned = spectral._attach_spectrum_figures(result, computations, options)
    return returned, traces, computations, options


def _spectrum_analysis_object(
    blank_echem_factory,
    ecat_module,
    *,
    sample_count=128,
    sample_rate=64.0,
    frequency=8.0,
):
    time = np.arange(sample_count, dtype=float) / sample_rate
    obj = _timed_object(
        blank_echem_factory,
        ecat_module.ca,
        time=time,
    )
    obj.data["Current"] = 1.25 * np.cos(2 * np.pi * frequency * time + 0.2)
    return obj


def _public_spectrum_object(blank_echem_factory, ecat_module, class_name):
    cls = getattr(ecat_module, class_name)
    if class_name in {"echem", "ca", "cp"}:
        obj = _timed_object(blank_echem_factory, cls)
        return obj, "recorded", 10.0

    obj = blank_echem_factory(cls)
    obj.data = pd.DataFrame(
        {
            "Potential": np.linspace(-0.3, 0.3, 12),
            "Current": np.cos(np.arange(12, dtype=float)),
        }
    )
    obj.units = {"Potential": "V", "Current": "A"}
    if class_name == "cv":
        obj.scan_rate = 0.2
        return obj, "reconstructed from CV potential and scan rate", 11.0 / 3.0

    obj.pulse_period = 0.25
    return obj, "reconstructed from DPV pulse period", 4.0


@pytest.mark.parametrize(
    ("unit", "stored_scale"),
    [("s", 1.0), ("ms", 1000.0)],
)
def test_recorded_time_is_converted_to_seconds(
    blank_echem_factory,
    ecat_module,
    unit,
    stored_scale,
):
    seconds = np.arange(12, dtype=float) * 0.125
    obj = _timed_object(
        blank_echem_factory,
        ecat_module.ca,
        time=seconds * stored_scale,
        time_unit=unit,
    )

    traces, warning_messages = _resolve_spectrum_traces(obj, _options())

    assert warning_messages == []
    assert traces[0].time_source == "recorded"
    assert traces[0].time_unit == "s"
    assert traces[0].time == pytest.approx(seconds)


@pytest.mark.parametrize(
    ("time_column", "units", "expected_scale"),
    [
        ("time", {"time": "ms", "Time": "s"}, 0.001),
        ("time", {"TIME": "ms", "Time": "s"}, 1.0),
    ],
)
def test_recorded_time_unit_uses_exact_column_then_exact_time_fallback(
    blank_echem_factory,
    ecat_module,
    time_column,
    units,
    expected_scale,
):
    obj = blank_echem_factory(ecat_module.ca)
    obj.data = pd.DataFrame(
        {
            time_column: np.arange(12, dtype=float),
            "Current": np.arange(12, dtype=float),
        }
    )
    obj.units = {**units, "Current": "A"}

    traces, _ = _resolve_spectrum_traces(obj, _options())

    assert traces[0].time == pytest.approx(np.arange(12) * expected_scale)


@pytest.mark.parametrize(
    ("time_column", "units", "expected_sample_rate"),
    [
        pytest.param(
            "TIME",
            {"time": "ms"},
            1000.0,
            id="unique-case-insensitive-match",
        ),
        pytest.param(
            "time",
            {"time": "ms", "Time": "s"},
            1000.0,
            id="exact-actual-column-precedence",
        ),
        pytest.param(
            "time",
            {"TIME": "ms", "Time": "s"},
            1.0,
            id="exact-canonical-time-precedence",
        ),
        pytest.param(
            "TIME",
            {"TIME": None, "time": "ms"},
            1.0,
            id="exact-missing-unit-defaults-to-seconds",
        ),
        pytest.param(
            "time",
            {"Time": pd.NA, "TIME": "ms"},
            1.0,
            id="canonical-missing-unit-defaults-to-seconds",
        ),
    ],
)
def test_public_spectrum_recorded_time_unit_precedence_is_case_robust(
    blank_echem_factory,
    ecat_module,
    time_column,
    units,
    expected_sample_rate,
):
    obj = blank_echem_factory(ecat_module.ca)
    obj.data = pd.DataFrame(
        {
            time_column: np.arange(12, dtype=float),
            "Current": np.cos(np.arange(12, dtype=float)),
        }
    )
    obj.units = {**units, "Current": "A"}

    result = obj.spectrum(
        {
            "method": "fft",
            "window": "boxcar",
            "detrend": None,
            "plot": False,
            "print": False,
        }
    )

    trace = result.diagnostics["traces"][0]
    expected_interval = 1.0 / expected_sample_rate
    assert trace.time == pytest.approx(np.arange(12) * expected_interval)
    assert result.summary["sample interval / s"] == pytest.approx(expected_interval)
    assert result.summary["sample rate / Hz"] == pytest.approx(expected_sample_rate)


def test_public_spectrum_rejects_conflicting_case_insensitive_time_units(
    blank_echem_factory,
    ecat_module,
):
    obj = blank_echem_factory(ecat_module.ca)
    obj.data = pd.DataFrame(
        {
            "TIME": np.arange(12, dtype=float),
            "Current": np.arange(12, dtype=float),
        }
    )
    obj.units = {"time": "ms", "tIME": "s", "Current": "A"}

    with pytest.raises(
        ValueError,
        match=r"(?i)ambiguous.*Time.*time.*ms.*tIME.*s",
    ):
        obj.spectrum(
            {
                "method": "fft",
                "plot": False,
                "print": False,
            }
        )


def test_recorded_time_without_unit_metadata_defaults_to_seconds(
    blank_echem_factory,
    ecat_module,
):
    obj = blank_echem_factory(ecat_module.ca)
    obj.data = pd.DataFrame(
        {
            "TIME": np.arange(12, dtype=float),
            "Current": np.arange(12, dtype=float),
        }
    )
    obj.units = {"Current": "A"}

    traces, _ = _resolve_spectrum_traces(obj, _options())

    assert traces[0].time == pytest.approx(np.arange(12, dtype=float))


@pytest.mark.parametrize(
    ("class_name", "expected_column"),
    [
        ("cv", "Current"),
        ("ca", "Current"),
        ("cp", "Potential"),
        ("dpv", "Current"),
        ("echem", "Current"),
    ],
)
def test_auto_column_uses_technique_default(
    blank_echem_factory,
    ecat_module,
    class_name,
    expected_column,
):
    cls = getattr(ecat_module, class_name)
    obj = _timed_object(blank_echem_factory, cls)

    traces, _ = _resolve_spectrum_traces(obj, _options())

    assert traces[0].signal_column == expected_column
    assert traces[0].signal_unit == obj.units[expected_column]
    assert traces[0].signal == pytest.approx(obj.data[expected_column])


def test_auto_column_is_case_insensitive_and_preserves_actual_key(
    blank_echem_factory,
    ecat_module,
):
    obj = blank_echem_factory(ecat_module.ca)
    obj.data = pd.DataFrame(
        {
            "time": np.arange(12, dtype=float),
            "cUrReNt": np.linspace(0.0, 1.0, 12),
        }
    )
    obj.units = {"Time": "s", "Current": "A"}

    traces, _ = _resolve_spectrum_traces(obj, _options())

    assert traces[0].signal_column == "cUrReNt"
    assert traces[0].signal_unit == "A"


def test_explicit_auxiliary_column_uses_case_insensitive_unit_key(
    blank_echem_factory,
    ecat_module,
):
    obj = _timed_object(blank_echem_factory, ecat_module.ca)
    obj.data["aUx"] = np.arange(12, dtype=float)
    obj.units["Aux"] = "mA"

    traces, _ = _resolve_spectrum_traces(obj, _options(column="AUX"))

    assert traces[0].signal_column == "aUx"
    assert traces[0].signal_unit == "mA"


def test_explicit_auxiliary_column_without_matching_unit_stays_unitless(
    blank_echem_factory,
    ecat_module,
):
    obj = _timed_object(blank_echem_factory, ecat_module.ca)
    obj.data["Aux"] = np.arange(12, dtype=float)

    traces, _ = _resolve_spectrum_traces(obj, _options(column="aux"))

    assert traces[0].signal_column == "Aux"
    assert traces[0].signal_unit == ""


def test_generic_auto_column_falls_back_to_first_finite_numeric_column(
    blank_echem_factory,
    ecat_module,
):
    obj = blank_echem_factory(ecat_module.echem)
    obj.data = pd.DataFrame(
        {
            "Time": np.arange(12, dtype=float),
            "Segment": np.ones(12, dtype=int),
            "Notes": ["ok"] * 12,
            "Response": np.linspace(3.0, 4.0, 12),
            "Other": np.linspace(5.0, 6.0, 12),
        }
    )
    obj.units = {"Time": "s", "Response": "counts"}

    traces, _ = _resolve_spectrum_traces(obj, _options())

    assert traces[0].signal_column == "Response"
    assert traces[0].signal == pytest.approx(obj.data["Response"])


def test_generic_auto_column_prefers_potential_before_arbitrary_numeric(
    blank_echem_factory,
    ecat_module,
):
    obj = blank_echem_factory(ecat_module.echem)
    obj.data = pd.DataFrame(
        {
            "Time": np.arange(12, dtype=float),
            "Response": np.linspace(3.0, 4.0, 12),
            "pOtEnTiAl": np.linspace(-0.4, 0.2, 12),
        }
    )
    obj.units = {"Time": "s", "Potential": "V", "Response": "counts"}

    traces, _ = _resolve_spectrum_traces(obj, _options())

    assert traces[0].signal_column == "pOtEnTiAl"
    assert traces[0].signal_unit == "V"


@pytest.mark.parametrize(("column", "expected"), [("aUx", "Aux"), (3, "Aux")])
def test_explicit_column_accepts_case_insensitive_name_or_position(
    blank_echem_factory,
    ecat_module,
    column,
    expected,
):
    obj = _timed_object(blank_echem_factory, ecat_module.ca)
    obj.data["Aux"] = np.arange(12, dtype=float) ** 2
    obj.units["Aux"] = "arb"

    traces, _ = _resolve_spectrum_traces(obj, _options(column=column))

    assert traces[0].signal_column == expected
    assert traces[0].signal_unit == "arb"
    assert traces[0].signal == pytest.approx(obj.data[expected])


def test_cv_reconstruction_creates_independent_segment_local_time(
    blank_echem_factory,
    ecat_module,
):
    obj = _two_segment_cv(blank_echem_factory, ecat_module)
    obj.x = lambda *args, **kwargs: pytest.fail("CV splitting must use raw Potential")

    traces, warning_messages = _resolve_spectrum_traces(obj, _options())

    assert [trace.segment for trace in traces] == [1, 2]
    assert [len(trace.time) for trace in traces] == [12, 11]
    assert all(trace.time[0] == pytest.approx(0.0) for trace in traces)
    assert all(
        trace.time_source == "reconstructed from CV potential and scan rate"
        for trace in traces
    )
    assert traces[0].time[-1] == pytest.approx(6.0)
    assert traces[1].time[-1] == pytest.approx(6.0 - 6.0 / 11.0)
    assert len(warning_messages) == 1
    assert "scan rate" in warning_messages[0].lower()


def test_cv_reconstructed_time_and_frequency_are_potential_unit_invariant(
    blank_echem_factory,
    ecat_module,
):
    cv_volts = _two_segment_cv(blank_echem_factory, ecat_module)
    cv_millivolts = _two_segment_cv(blank_echem_factory, ecat_module)
    cv_millivolts.data["Potential"] *= 1000.0
    cv_millivolts.units["Potential"] = "mV"
    options = {
        "method": "fft",
        "window": "boxcar",
        "detrend": None,
        "plot": False,
        "print": False,
    }

    with pytest.warns(UserWarning, match="reconstructed"):
        volts_result = cv_volts.spectrum(options)
    with pytest.warns(UserWarning, match="reconstructed"):
        millivolts_result = cv_millivolts.spectrum(options)

    for volts_trace, millivolts_trace in zip(
        volts_result.diagnostics["traces"],
        millivolts_result.diagnostics["traces"],
    ):
        assert millivolts_trace.time == pytest.approx(volts_trace.time)
    np.testing.assert_allclose(
        millivolts_result.table["Frequency / Hz"],
        volts_result.table["Frequency / Hz"],
    )


@pytest.mark.parametrize("potential_unit", ["A", "not-a-unit"])
def test_cv_reconstructed_time_rejects_incompatible_potential_units(
    blank_echem_factory,
    ecat_module,
    potential_unit,
):
    obj = _two_segment_cv(blank_echem_factory, ecat_module)
    obj.units["Potential"] = potential_unit

    with pytest.raises(
        ValueError,
        match=rf"Potential.*{potential_unit}.*convert.*V|convert.*Potential.*{potential_unit}.*V",
    ):
        obj.spectrum({"plot": False, "print": False})


def test_recorded_time_cv_is_split_and_made_segment_local(
    blank_echem_factory,
    ecat_module,
):
    obj = _two_segment_cv(blank_echem_factory, ecat_module, recorded_time=True)

    traces, warning_messages = _resolve_spectrum_traces(obj, _options())

    assert warning_messages == []
    assert [trace.segment for trace in traces] == [1, 2]
    assert all(trace.time_source == "recorded" for trace in traces)
    assert all(trace.time[0] == pytest.approx(0.0) for trace in traces)
    assert traces[0].time[-1] == pytest.approx(2.2)
    assert traces[1].time[-1] == pytest.approx(2.0)


def test_recorded_time_cv_localizes_from_first_finite_timestamp(
    blank_echem_factory,
    ecat_module,
):
    obj = _two_segment_cv(blank_echem_factory, ecat_module, recorded_time=True)
    obj.data.loc[0, "Time"] = np.nan

    traces, _ = _resolve_spectrum_traces(obj, _options(segments=1))

    assert traces[0].source_indices.tolist() == list(range(1, 12))
    assert traces[0].time[0] == pytest.approx(0.0)
    assert traces[0].time[-1] == pytest.approx(2.0)


def test_recorded_time_cv_origin_does_not_move_when_first_signal_is_nonfinite(
    blank_echem_factory,
    ecat_module,
):
    obj = _two_segment_cv(blank_echem_factory, ecat_module, recorded_time=True)
    obj.data.loc[0, "Current"] = np.nan

    traces, _ = _resolve_spectrum_traces(obj, _options(segments=1))

    assert traces[0].source_indices.tolist() == list(range(1, 12))
    assert traces[0].time[0] == pytest.approx(0.2)
    assert traces[0].time[-1] == pytest.approx(2.2)


def test_dpv_reconstruction_uses_pulse_period(
    blank_echem_factory,
    ecat_module,
):
    obj = blank_echem_factory(ecat_module.dpv)
    obj.data = pd.DataFrame(
        {
            "Potential": np.linspace(-0.2, 0.3, 12),
            "Current": np.linspace(0.0, 1.0, 12),
        }
    )
    obj.units = {"Potential": "V", "Current": "A"}
    obj.pulse_period = 0.25

    traces, warning_messages = _resolve_spectrum_traces(obj, _options())

    assert traces[0].time == pytest.approx(np.arange(12) * 0.25)
    assert traces[0].time_source == "reconstructed from DPV pulse period"
    assert len(warning_messages) == 1
    assert "pulse period" in warning_messages[0].lower()


def test_selected_cv_segments_follow_repeated_vertex_ownership(
    blank_echem_factory,
    ecat_module,
):
    forward = np.linspace(-0.3, 0.3, 12)
    reverse = np.linspace(0.3, -0.3, 12)
    potential = np.concatenate([forward, reverse])
    obj = blank_echem_factory(ecat_module.cv)
    obj.data = pd.DataFrame(
        {
            "Time": np.arange(len(potential), dtype=float),
            "Potential": potential,
            "Current": np.arange(len(potential), dtype=float),
        }
    )
    obj.units = {"Time": "s", "Potential": "V", "Current": "A"}

    traces, _ = _resolve_spectrum_traces(obj, _options(segments=[1, 2]))

    assert [trace.segment for trace in traces] == [1, 2]
    assert traces[0].source_indices.tolist() == list(range(13))
    assert traces[1].source_indices.tolist() == list(range(13, 24))


def test_non_cv_numeric_segments_are_grouped_from_stored_column(
    blank_echem_factory,
    ecat_module,
):
    obj = blank_echem_factory(ecat_module.ca)
    obj.data = pd.DataFrame(
        {
            "time": np.arange(24, dtype=float) * 0.1,
            "segment": np.repeat([1, 2], 12),
            "current": np.arange(24, dtype=float),
        }
    )
    obj.units = {"Time": "s", "Current": "A"}

    traces, _ = _resolve_spectrum_traces(obj, _options(segments=[2, 1]))

    assert [trace.segment for trace in traces] == [2, 1]
    assert traces[0].source_indices.tolist() == list(range(12, 24))
    assert traces[1].source_indices.tolist() == list(range(12))


@pytest.mark.parametrize("class_name", ["ca", "cp", "echem"])
@pytest.mark.parametrize("clock", ["local", "continuous"])
def test_public_non_cv_spectrum_all_groups_stored_segments_without_bridging(
    blank_echem_factory,
    ecat_module,
    class_name,
    clock,
):
    sample_count = 12
    local_time = np.arange(sample_count, dtype=float) * 0.1
    time = (
        np.tile(local_time, 2)
        if clock == "local"
        else np.arange(2 * sample_count, dtype=float) * 0.1
    )
    segment_values = np.repeat([2, 1], sample_count)
    obj = blank_echem_factory(getattr(ecat_module, class_name))
    obj.data = pd.DataFrame(
        {
            "Time": time,
            "Current": np.cos(np.arange(2 * sample_count, dtype=float)),
            "Potential": np.sin(np.arange(2 * sample_count, dtype=float)),
            "Segment": segment_values,
        }
    )
    obj.units = {"Time": "s", "Current": "A", "Potential": "V"}

    result = obj.spectrum(
        {
            "method": "fft",
            "window": "boxcar",
            "detrend": None,
            "plot": True,
            "print": False,
        }
    )

    traces = result.diagnostics["traces"]
    expected_times = (
        [local_time, local_time]
        if clock == "local"
        else [time[:sample_count], time[sample_count:]]
    )
    assert [trace.segment for trace in traces] == [2, 1]
    assert [len(trace.time) for trace in traces] == [sample_count, sample_count]
    for trace, expected_time in zip(traces, expected_times):
        assert trace.time == pytest.approx(expected_time)
    assert result.table["Segment"].drop_duplicates().tolist() == [2, 1]
    assert [line.get_label() for line in result.axes.lines] == [
        "Segment 2",
        "Segment 1",
    ]


def test_public_non_cv_spectrum_all_without_segment_column_is_one_trace(
    blank_echem_factory,
    ecat_module,
):
    obj = _timed_object(blank_echem_factory, ecat_module.echem)

    result = obj.spectrum(
        {
            "method": "fft",
            "window": "boxcar",
            "plot": False,
            "print": False,
        }
    )

    assert len(result.diagnostics["traces"]) == 1
    assert result.diagnostics["traces"][0].segment is None
    assert "Segment" not in result.table


@pytest.mark.parametrize(
    ("segment_values", "message"),
    [
        pytest.param(
            [np.nan] * 12,
            r"Segment column.*no valid segment values.*positive integers",
            id="no-valid-values",
        ),
        pytest.param(
            [1] * 11 + ["not-a-segment"],
            r"Segment column.*invalid or missing.*positive integer.*11",
            id="nonnumeric-row",
        ),
        pytest.param(
            [1] * 11 + [None],
            r"Segment column.*invalid or missing.*positive integer.*11",
            id="missing-row",
        ),
    ],
)
def test_public_non_cv_spectrum_rejects_invalid_stored_segment_rows(
    blank_echem_factory,
    ecat_module,
    segment_values,
    message,
):
    obj = _timed_object(blank_echem_factory, ecat_module.ca)
    obj.data["Segment"] = segment_values

    with pytest.raises(ValueError, match=message):
        obj.spectrum(
            {
                "method": "fft",
                "window": "boxcar",
                "plot": False,
                "print": False,
            }
        )


@pytest.mark.parametrize(
    "boolean_value",
    [
        pytest.param(True, id="python-bool"),
        pytest.param(np.bool_(True), id="numpy-bool"),
    ],
)
def test_public_non_cv_spectrum_rejects_boolean_segment_rows(
    blank_echem_factory,
    ecat_module,
    boolean_value,
):
    obj = _timed_object(blank_echem_factory, ecat_module.ca)
    obj.data["Segment"] = pd.Series(
        [1] * 11 + [boolean_value],
        dtype=object,
    )

    with pytest.raises(
        ValueError,
        match=r"Segment column.*invalid or missing.*positive integer.*11",
    ):
        obj.spectrum(
            {
                "method": "fft",
                "window": "boxcar",
                "plot": False,
                "print": False,
            }
        )


def test_time_range_is_inclusive_after_local_time_resolution(
    blank_echem_factory,
    ecat_module,
):
    obj = _timed_object(
        blank_echem_factory,
        ecat_module.ca,
        time=np.arange(12, dtype=float),
    )

    traces, _ = _resolve_spectrum_traces(obj, _options(time_range=[2.0, 9.0]))

    assert traces[0].time.tolist() == list(np.arange(2.0, 10.0))
    assert traces[0].source_indices.tolist() == list(range(2, 10))


def test_cv_time_range_is_inclusive_on_each_local_segment(
    blank_echem_factory,
    ecat_module,
):
    obj = _two_segment_cv(blank_echem_factory, ecat_module, recorded_time=True)
    obj.data["Time"] = 100.0 + np.arange(len(obj.data), dtype=float)

    traces, _ = _resolve_spectrum_traces(obj, _options(time_range=[2.0, 9.0]))

    assert [trace.time.tolist() for trace in traces] == [
        list(np.arange(2.0, 10.0)),
        list(np.arange(2.0, 10.0)),
    ]
    assert traces[0].source_indices.tolist() == list(range(2, 10))
    assert traces[1].source_indices.tolist() == list(range(14, 22))


def test_resampling_rejects_oversized_grid_before_allocation(
    blank_echem_factory,
    ecat_module,
    monkeypatch,
):
    time = np.array([0, 1e-6, 2e-6, 3e-6, 4e-6, 5e-6, 6e-6, 3.0])
    obj = _timed_object(blank_echem_factory, ecat_module.ca, time=time)
    real_arange = np.arange

    def guarded_arange(*args, **kwargs):
        if len(args) >= 3:
            start, stop, step = args[:3]
            if (stop - start) / step > 2_000_000:
                pytest.fail("oversized resampling grid was allocated")
        return real_arange(*args, **kwargs)

    monkeypatch.setattr(spectral.np, "arange", guarded_arange)

    with pytest.raises(
        ValueError,
        match=r"2,000,000.*time range.*acquisition timestamps",
    ):
        _resolve_spectrum_traces(obj, _options(resample=True))


def test_resampling_at_cap_never_allocates_above_checked_count(
    blank_echem_factory,
    ecat_module,
    monkeypatch,
):
    time = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 9.6])
    obj = _timed_object(blank_echem_factory, ecat_module.ca, time=time)
    real_arange = np.arange
    allocation_sizes = []

    def guarded_arange(*args, **kwargs):
        values = real_arange(*args, **kwargs)
        allocation_sizes.append(values.size)
        if values.size > spectral._MAX_RESAMPLED_POINTS:
            pytest.fail("resampling allocated more points than its checked count")
        return values

    monkeypatch.setattr(spectral, "_MAX_RESAMPLED_POINTS", 10)
    monkeypatch.setattr(spectral.np, "arange", guarded_arange)

    traces, _ = _resolve_spectrum_traces(obj, _options(resample=True))

    assert traces[0].time.tolist() == list(range(10))
    assert max(allocation_sizes) == 10


def test_resampling_preserves_integral_decimal_grid_endpoint(
    blank_echem_factory,
    ecat_module,
):
    time = np.array([0.10, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.19])
    obj = _timed_object(blank_echem_factory, ecat_module.ca, time=time)

    traces, _ = _resolve_spectrum_traces(obj, _options(resample=True))

    expected = 0.10 + np.arange(10, dtype=float) * 0.01
    assert traces[0].time == pytest.approx(expected)
    assert len(traces[0].time) == 10
    assert traces[0].time[-1] == 0.19


@pytest.mark.parametrize("class_name", ["ca", "cp", "echem"])
def test_techniques_without_reconstructable_time_require_recorded_time(
    blank_echem_factory,
    ecat_module,
    class_name,
):
    obj = blank_echem_factory(getattr(ecat_module, class_name))
    obj.data = pd.DataFrame({"Current": np.arange(12.0), "Potential": np.arange(12.0)})
    obj.units = {"Current": "A", "Potential": "V"}

    with pytest.raises(ValueError, match="recorded Time"):
        _resolve_spectrum_traces(obj, _options())


@pytest.mark.parametrize(
    ("class_name", "attribute", "message"),
    [("cv", "scan_rate", "scan_rate"), ("dpv", "pulse_period", "pulse_period")],
)
def test_reconstructed_time_requires_positive_timing_metadata(
    blank_echem_factory,
    ecat_module,
    class_name,
    attribute,
    message,
):
    obj = blank_echem_factory(getattr(ecat_module, class_name))
    obj.data = pd.DataFrame(
        {"Potential": np.linspace(-0.2, 0.2, 12), "Current": np.arange(12.0)}
    )
    obj.units = {"Potential": "V", "Current": "A"}
    setattr(obj, attribute, None)

    with pytest.raises(ValueError, match=message):
        _resolve_spectrum_traces(obj, _options())


@pytest.mark.parametrize(
    "time",
    [
        [0, 1, 2, 3, 4, 4, 6, 7, 8, 9, 10, 11],
        [0, 1, 2, 3, 4, 3.5, 6, 7, 8, 9, 10, 11],
    ],
)
def test_duplicate_or_decreasing_time_raises(
    blank_echem_factory,
    ecat_module,
    time,
):
    obj = _timed_object(blank_echem_factory, ecat_module.ca, time=time)

    with pytest.raises(ValueError, match="strictly increasing"):
        _resolve_spectrum_traces(obj, _options(resample=True))


def test_at_least_eight_finite_pairs_are_required(blank_echem_factory, ecat_module):
    obj = _timed_object(blank_echem_factory, ecat_module.ca, time=np.arange(10.0))
    obj.data.loc[[1, 4, 8], "Current"] = np.nan

    with pytest.raises(ValueError, match="at least 8 finite"):
        _resolve_spectrum_traces(obj, _options())


def test_unsupported_recorded_time_unit_is_actionable(blank_echem_factory, ecat_module):
    obj = _timed_object(blank_echem_factory, ecat_module.ca, time_unit="fortnight")

    with pytest.raises(ValueError, match="convert recorded Time.*fortnight.*seconds"):
        _resolve_spectrum_traces(obj, _options())


@pytest.mark.parametrize(
    ("column", "message"),
    [("missing", "Available columns"), (20, "Available columns"), ("Notes", "numeric")],
)
def test_invalid_signal_column_raises_with_available_columns(
    blank_echem_factory,
    ecat_module,
    column,
    message,
):
    obj = _timed_object(blank_echem_factory, ecat_module.ca)
    obj.data["Notes"] = ["x"] * len(obj.data)

    with pytest.raises((ValueError, IndexError), match=message):
        _resolve_spectrum_traces(obj, _options(column=column))


def test_invalid_cv_segment_lists_available_segments(blank_echem_factory, ecat_module):
    obj = _two_segment_cv(blank_echem_factory, ecat_module, recorded_time=True)

    with pytest.raises(ValueError, match=r"available segments.*1.*2"):
        _resolve_spectrum_traces(obj, _options(segments=3))


@pytest.mark.parametrize("include_segment_column", [False, True])
def test_invalid_non_cv_segment_selection_is_actionable(
    blank_echem_factory,
    ecat_module,
    include_segment_column,
):
    obj = _timed_object(blank_echem_factory, ecat_module.ca)
    if include_segment_column:
        obj.data["Segment"] = np.ones(len(obj.data), dtype=int)
        message = "available segments"
    else:
        message = "Segment column"

    with pytest.raises(ValueError, match=message):
        _resolve_spectrum_traces(obj, _options(segments=2))


@pytest.mark.parametrize("class_name", ["cv", "dpv"])
def test_reconstruction_warning_is_returned_once_per_analysis_call(
    blank_echem_factory,
    ecat_module,
    class_name,
):
    if class_name == "cv":
        obj = _two_segment_cv(blank_echem_factory, ecat_module)
    else:
        obj = blank_echem_factory(ecat_module.dpv)
        obj.data = pd.DataFrame(
            {"Potential": np.linspace(-0.2, 0.2, 12), "Current": np.arange(12.0)}
        )
        obj.units = {"Potential": "V", "Current": "A"}
        obj.pulse_period = 0.2

    _, warning_messages = _resolve_spectrum_traces(obj, _options())

    assert len(warning_messages) == 1


def test_spectrum_trace_is_frozen_and_exposes_required_diagnostics(
    blank_echem_factory,
    ecat_module,
):
    obj = _timed_object(blank_echem_factory, ecat_module.ca)

    traces, _ = _resolve_spectrum_traces(obj, _options())
    trace = traces[0]
    trace_fields = {item.name: item for item in fields(_SpectrumTrace)}

    assert {
        "segment",
        "label",
        "time",
        "signal",
        "signal_column",
        "signal_unit",
        "time_source",
        "time_unit",
        "source_indices",
        "original_time",
        "resampled",
        "original_signal",
        "original_intervals",
        "resolved_intervals",
        "median_interval",
        "relative_interval_deviation",
        "sample_rate",
    } <= set(trace_fields)
    assert trace_fields["time_unit"].default == "s"
    assert trace_fields["resampled"].default is False
    with pytest.raises(FrozenInstanceError):
        trace.segment = 2


def test_spectrum_trace_owns_and_protects_every_array_field():
    source_arrays = {
        "time": np.arange(8, dtype=float),
        "signal": np.arange(8, dtype=float) + 10,
        "source_indices": np.arange(8),
        "original_time": np.arange(8, dtype=float) + 20,
        "original_signal": np.arange(8, dtype=float) + 30,
        "original_intervals": np.arange(7, dtype=float) + 40,
        "resolved_intervals": np.arange(7, dtype=float) + 50,
    }
    trace = _SpectrumTrace(
        segment=None,
        label="trace",
        signal_column="Current",
        signal_unit="A",
        time_source="recorded",
        median_interval=1.0,
        relative_interval_deviation=0.0,
        sample_rate=1.0,
        **source_arrays,
    )

    for name, source in source_arrays.items():
        resolved = getattr(trace, name)
        assert not np.shares_memory(resolved, source)
        assert resolved.flags.writeable is False
        with pytest.raises(ValueError, match="read-only"):
            resolved.flat[0] = resolved.flat[0]


def test_resolver_returns_warning_text_without_emitting_python_warnings(
    blank_echem_factory,
    ecat_module,
):
    obj = _two_segment_cv(blank_echem_factory, ecat_module)

    with warnings.catch_warnings(record=True) as emitted:
        warnings.simplefilter("always")
        _, warning_messages = _resolve_spectrum_traces(obj, _options())

    assert len(warning_messages) == 1
    assert emitted == []


def test_welch_recovers_ca_sinusoid_peaks_within_one_frequency_bin(
    blank_echem_factory,
    ecat_module,
):
    sample_rate = 128.0
    time = np.arange(1024, dtype=float) / sample_rate
    values = np.sin(2 * np.pi * 5 * time) + np.sin(2 * np.pi * 17 * time)
    trace, options, warning_messages = _resolved_signal(
        blank_echem_factory,
        ecat_module,
        values,
        sample_rate=sample_rate,
        points_per_window=256,
    )

    computation = spectral._compute_trace(trace, options, warning_messages)

    for expected_frequency in (5.0, 17.0):
        neighborhood = np.abs(computation.frequency - expected_frequency) <= 2.0
        recovered = computation.frequency[neighborhood][
            np.argmax(computation.psd[neighborhood])
        ]
        assert recovered == pytest.approx(
            expected_frequency,
            abs=computation.frequency_bin_spacing,
        )
    assert computation.method == "welch"
    assert computation.nperseg == 256
    assert computation.noverlap == 128
    assert computation.nfft == 256
    assert computation.window_count == 7
    assert computation.sample_rate == pytest.approx(sample_rate)
    assert computation.nyquist == pytest.approx(sample_rate / 2)
    assert computation.frequency_bin_spacing == pytest.approx(0.5)
    assert warning_messages == []


def test_welch_integrated_psd_recovers_zero_mean_sinusoid_variance(
    blank_echem_factory,
    ecat_module,
):
    sample_rate = 128.0
    amplitude = 3.2
    time = np.arange(4096, dtype=float) / sample_rate
    values = amplitude * np.sin(2 * np.pi * 8 * time)
    trace, options, warning_messages = _resolved_signal(
        blank_echem_factory,
        ecat_module,
        values,
        sample_rate=sample_rate,
        detrend=None,
        points_per_window=512,
        include_dc=True,
    )

    computation = spectral._compute_trace(trace, options, warning_messages)
    integrated_power = np.trapezoid(computation.psd, computation.frequency)

    # Exact bin centering limits this ideal case to floating-point integration error.
    assert integrated_power == pytest.approx(amplitude**2 / 2, rel=1e-10)


def test_fft_recovers_one_sided_cosine_amplitude_and_phase(
    blank_echem_factory,
    ecat_module,
):
    sample_rate = 64.0
    sample_count = 64
    frequency = 9.0
    amplitude = 2.75
    phase = 0.4
    time = np.arange(sample_count, dtype=float) / sample_rate
    values = amplitude * np.cos(2 * np.pi * frequency * time + phase)
    trace, options, warning_messages = _resolved_signal(
        blank_echem_factory,
        ecat_module,
        values,
        sample_rate=sample_rate,
        method="fft",
        detrend=None,
        window="boxcar",
    )

    computation = spectral._compute_trace(trace, options, warning_messages)
    peak = np.argmax(computation.amplitude)

    assert computation.frequency[peak] == pytest.approx(frequency)
    assert computation.amplitude[peak] == pytest.approx(amplitude)
    assert computation.phase[peak] == pytest.approx(phase)
    assert computation.coherent_gain == pytest.approx(1.0)
    assert computation.nfft == sample_count
    assert computation.window == "boxcar"
    assert computation.sample_rate == pytest.approx(sample_rate)
    assert computation.nyquist == pytest.approx(sample_rate / 2)
    assert computation.frequency_bin_spacing == pytest.approx(1.0)


def test_fft_hann_window_coherent_gain_recovers_bin_centered_amplitude(
    blank_echem_factory,
    ecat_module,
):
    sample_rate = 256.0
    sample_count = 256
    amplitude = 4.2
    frequency = 21.0
    time = np.arange(sample_count, dtype=float) / sample_rate
    values = amplitude * np.cos(2 * np.pi * frequency * time)
    trace, options, warning_messages = _resolved_signal(
        blank_echem_factory,
        ecat_module,
        values,
        sample_rate=sample_rate,
        method="fft",
        detrend=None,
        window="hann",
    )

    computation = spectral._compute_trace(trace, options, warning_messages)
    selected = np.flatnonzero(np.isclose(computation.frequency, frequency))

    assert computation.coherent_gain == pytest.approx(0.5)
    assert computation.amplitude[selected[0]] == pytest.approx(amplitude)


@pytest.mark.parametrize(
    ("sample_count", "bin_index", "values_factory"),
    [
        (64, 32, lambda n, k: 1.8 * np.cos(np.pi * n)),
        (63, 31, lambda n, k: 1.8 * np.cos(2 * np.pi * k * n / len(n))),
    ],
)
def test_fft_one_sided_scaling_handles_even_nyquist_and_odd_highest_bin(
    blank_echem_factory,
    ecat_module,
    sample_count,
    bin_index,
    values_factory,
):
    samples = np.arange(sample_count, dtype=float)
    trace, options, warning_messages = _resolved_signal(
        blank_echem_factory,
        ecat_module,
        values_factory(samples, bin_index),
        sample_rate=float(sample_count),
        method="fft",
        detrend=None,
        window="boxcar",
    )

    computation = spectral._compute_trace(trace, options, warning_messages)

    selected = np.flatnonzero(np.isclose(computation.frequency, float(bin_index)))
    assert len(selected) == 1
    assert computation.amplitude[selected[0]] == pytest.approx(1.8)


def test_fft_zero_padding_keeps_signal_length_normalization(
    blank_echem_factory,
    ecat_module,
):
    sample_count = 32
    sample_rate = 32.0
    amplitude = 3.25
    frequency = 5.0
    time = np.arange(sample_count, dtype=float) / sample_rate
    trace, options, warning_messages = _resolved_signal(
        blank_echem_factory,
        ecat_module,
        amplitude * np.cos(2 * np.pi * frequency * time),
        sample_rate=sample_rate,
        method="fft",
        detrend=None,
        window="boxcar",
        nfft=128,
    )

    computation = spectral._compute_trace(trace, options, warning_messages)
    selected = np.flatnonzero(np.isclose(computation.frequency, frequency))

    assert computation.nfft == 128
    assert computation.frequency_bin_spacing == pytest.approx(0.25)
    assert computation.amplitude[selected[0]] == pytest.approx(amplitude)


@pytest.mark.parametrize(
    ("detrend", "values", "expected_dc"),
    [
        (None, np.full(64, 3.0), 3.0),
        ("constant", np.full(64, 3.0), 0.0),
        ("linear", 2.0 * np.arange(64, dtype=float) + 3.0, 0.0),
    ],
)
def test_fft_detrend_modes_have_physical_behavior(
    blank_echem_factory,
    ecat_module,
    detrend,
    values,
    expected_dc,
):
    trace, options, warning_messages = _resolved_signal(
        blank_echem_factory,
        ecat_module,
        values,
        sample_rate=64.0,
        method="fft",
        detrend=detrend,
        window="boxcar",
        include_dc=True,
    )

    computation = spectral._compute_trace(trace, options, warning_messages)

    assert computation.amplitude[0] == pytest.approx(expected_dc, abs=1e-12)
    if detrend is not None:
        assert computation.amplitude == pytest.approx(np.zeros(33), abs=1e-12)


def test_welch_passes_none_detrend_to_scipy_as_false(
    blank_echem_factory,
    ecat_module,
    monkeypatch,
):
    observed = []
    original_welch = spectral.signal.welch

    def recording_welch(*args, **kwargs):
        observed.append(kwargs)
        return original_welch(*args, **kwargs)

    monkeypatch.setattr(spectral.signal, "welch", recording_welch)
    trace, options, warning_messages = _resolved_signal(
        blank_echem_factory,
        ecat_module,
        np.cos(2 * np.pi * np.arange(64) / 8),
        sample_rate=64.0,
        detrend=None,
        points_per_window=32,
    )

    spectral._compute_trace(trace, options, warning_messages)

    assert observed[0]["detrend"] is False
    assert observed[0]["return_onesided"] is True
    assert observed[0]["scaling"] == "density"


def test_dc_is_omitted_by_default_and_retained_when_requested(
    blank_echem_factory,
    ecat_module,
):
    values = 4.0 + np.cos(2 * np.pi * np.arange(64) / 8)
    trace, default_options, default_warnings = _resolved_signal(
        blank_echem_factory,
        ecat_module,
        values,
        sample_rate=64.0,
        method="fft",
        detrend=None,
        window="boxcar",
    )
    _, include_options, include_warnings = _resolved_signal(
        blank_echem_factory,
        ecat_module,
        values,
        sample_rate=64.0,
        method="fft",
        detrend=None,
        window="boxcar",
        include_dc=True,
    )

    omitted = spectral._compute_trace(trace, default_options, default_warnings)
    included = spectral._compute_trace(trace, include_options, include_warnings)

    assert np.all(omitted.frequency > 0)
    assert included.frequency[0] == 0
    assert included.amplitude[0] == pytest.approx(4.0)


def test_frequency_range_is_inclusive(
    blank_echem_factory,
    ecat_module,
):
    trace, options, warning_messages = _resolved_signal(
        blank_echem_factory,
        ecat_module,
        np.cos(2 * np.pi * 5 * np.arange(64) / 64),
        sample_rate=64.0,
        method="fft",
        detrend=None,
        window="boxcar",
        frequency_range=[5.0, 7.0],
    )

    computation = spectral._compute_trace(trace, options, warning_messages)

    assert computation.frequency.tolist() == [5.0, 6.0, 7.0]


def test_public_spectrum_frequency_range_uses_ulp_tolerance_at_boundaries(
    blank_echem_factory,
    ecat_module,
):
    sample_rate = 100.0
    sample_count = 100
    obj = _spectrum_analysis_object(
        blank_echem_factory,
        ecat_module,
        sample_count=sample_count,
        sample_rate=sample_rate,
    )

    result = obj.spectrum(
        {
            "method": "fft",
            "window": "boxcar",
            "detrend": None,
            "frequency range": [
                np.nextafter(10.0, np.inf),
                np.nextafter(50.0, np.inf),
            ],
            "plot": False,
            "print": False,
        }
    )

    assert result.table["Frequency / Hz"].iloc[0] == pytest.approx(10.0)
    assert result.table["Frequency / Hz"].iloc[-1] == pytest.approx(50.0)
    assert len(result.table) == 41


def test_maximum_finite_frequency_range_still_raises_without_runtime_warning(
    blank_echem_factory,
    ecat_module,
):
    trace, options, warning_messages = _resolved_signal(
        blank_echem_factory,
        ecat_module,
        np.arange(100, dtype=float),
        sample_rate=100.0,
        method="fft",
        frequency_range=[10.0, np.finfo(float).max],
    )

    with warnings.catch_warnings(record=True) as emitted:
        warnings.simplefilter("always")
        with pytest.raises(ValueError, match="maximum frequency.*Nyquist"):
            spectral._compute_trace(trace, options, warning_messages)

    assert emitted == []


def test_frequency_range_above_nyquist_names_requested_and_available_values(
    blank_echem_factory,
    ecat_module,
):
    trace, options, warning_messages = _resolved_signal(
        blank_echem_factory,
        ecat_module,
        np.arange(64, dtype=float),
        sample_rate=64.0,
        method="fft",
        frequency_range=[1.0, 33.0],
    )

    with pytest.raises(ValueError, match=r"33(?:\.0)?.*Nyquist.*32"):
        spectral._compute_trace(trace, options, warning_messages)


def test_frequency_filtering_errors_when_no_bins_remain(
    blank_echem_factory,
    ecat_module,
):
    trace, options, warning_messages = _resolved_signal(
        blank_echem_factory,
        ecat_module,
        np.arange(64, dtype=float),
        sample_rate=64.0,
        method="fft",
        frequency_range=[0.0, 0.5],
    )

    with pytest.raises(ValueError, match="no frequency bins"):
        spectral._compute_trace(trace, options, warning_messages)


@pytest.mark.parametrize("method", ["welch", "fft"])
def test_public_spectrum_rejects_oversized_nfft_before_transform_backend(
    blank_echem_factory,
    ecat_module,
    monkeypatch,
    method,
):
    obj = _spectrum_analysis_object(blank_echem_factory, ecat_module)
    backend_calls = []

    def fail_backend(*args, **kwargs):
        backend_calls.append((args, kwargs))
        pytest.fail("transform backend received an oversized nfft")

    if method == "welch":
        monkeypatch.setattr(spectral.signal, "welch", fail_backend)
    else:
        monkeypatch.setattr(spectral.np.fft, "rfft", fail_backend)

    with pytest.raises(
        ValueError,
        match=(
            r"Requested nfft 2,000,001.*2,000,000-point limit.*"
            r"[Rr]educe nfft.*narrow.*trace"
        ),
    ):
        obj.spectrum(
            {
                "method": method,
                "window": "boxcar",
                "points per window": 32,
                "nfft": 2_000_001,
                "plot": False,
                "print": False,
            }
        )

    assert backend_calls == []


def test_implicit_fft_size_limit_is_checked_before_signal_materialization():
    class HugeLengthSignal:
        def __len__(self):
            return 2_000_001

        def __array__(self, *args, **kwargs):
            pytest.fail("oversized implicit FFT signal was materialized")

    trace = SimpleNamespace(
        signal=HugeLengthSignal(),
        label="implicit huge FFT",
        median_interval=1.0,
    )

    with pytest.raises(
        ValueError,
        match=(
            r"Implicit FFT nfft 2,000,001.*selected sample count.*"
            r"2,000,000-point limit.*[Rr]educe nfft.*narrow.*trace"
        ),
    ):
        spectral._compute_trace(
            trace,
            _options(method="fft", window="boxcar", detrend=None),
            [],
        )


def test_welch_warns_when_fewer_than_two_complete_windows_contribute(
    blank_echem_factory,
    ecat_module,
):
    trace, options, warning_messages = _resolved_signal(
        blank_echem_factory,
        ecat_module,
        np.arange(64, dtype=float),
        sample_rate=64.0,
    )

    computation = spectral._compute_trace(trace, options, warning_messages)

    assert computation.window_count == 1
    assert len([message for message in warning_messages if "two" in message]) == 1


def test_welch_auto_window_uses_up_to_256_samples(
    blank_echem_factory,
    ecat_module,
):
    trace, options, warning_messages = _resolved_signal(
        blank_echem_factory,
        ecat_module,
        np.arange(300, dtype=float),
        sample_rate=100.0,
    )

    computation = spectral._compute_trace(trace, options, warning_messages)

    assert computation.nperseg == 256
    assert computation.noverlap == 128
    assert computation.window_count == 1


@pytest.mark.parametrize(
    ("method", "cls_name", "column", "unit", "value_column", "value_unit"),
    [
        ("welch", "ca", "Current", "A", "PSD / A^2 Hz^-1", "A^2 Hz^-1"),
        ("welch", "ca", "Current", "mA", "PSD / mA^2 Hz^-1", "mA^2 Hz^-1"),
        (
            "welch",
            "ca",
            "Current",
            "A cm^-2",
            "PSD / (A cm^-2)^2 Hz^-1",
            "(A cm^-2)^2 Hz^-1",
        ),
        ("fft", "cp", "Potential", "V", "Amplitude / V", "V"),
        ("welch", "echem", "Response", "", "PSD / signal^2 Hz^-1", "signal^2 Hz^-1"),
        ("fft", "echem", "Response", "", "Amplitude / signal unit", "signal unit"),
    ],
)
def test_spectrum_table_headers_and_units_follow_the_selected_signal(
    blank_echem_factory,
    ecat_module,
    method,
    cls_name,
    column,
    unit,
    value_column,
    value_unit,
):
    trace, options, warning_messages = _resolved_signal(
        blank_echem_factory,
        ecat_module,
        np.cos(2 * np.pi * np.arange(64) / 8),
        sample_rate=64.0,
        cls=getattr(ecat_module, cls_name),
        column=column,
        unit=unit,
        method=method,
        points_per_window=32,
    )
    computation = spectral._compute_trace(trace, options, warning_messages)

    table = spectral._spectrum_table([computation], method)
    units = spectral._spectrum_units(unit, method)

    assert "Segment" not in table.columns
    assert "Frequency / Hz" in table.columns
    assert value_column in table.columns
    assert table.attrs["units"] == units
    assert units["Frequency / Hz"] == "Hz"
    assert units[value_column] == value_unit
    if method == "fft":
        assert "Phase / rad" in table.columns
        assert units["Phase / rad"] == "rad"


def test_mirrored_cv_segments_compute_independently_and_table_retains_segment(
    blank_echem_factory,
    ecat_module,
):
    obj = _two_segment_cv(blank_echem_factory, ecat_module, recorded_time=True)
    options = _options(method="fft", detrend=None, window="boxcar", include_dc=True)
    traces, warning_messages = _resolve_spectrum_traces(obj, options)

    computations = [
        spectral._compute_trace(trace, options, warning_messages) for trace in traces
    ]
    table = spectral._spectrum_table(computations, "fft")

    assert [computation.segment for computation in computations] == [1, 2]
    assert len(computations[0].frequency) == 7
    assert len(computations[1].frequency) == 6
    assert table.columns[0] == "Segment"
    assert table["Segment"].drop_duplicates().tolist() == [1, 2]
    assert len(table) == 13


@pytest.mark.parametrize("method", ["welch", "fft"])
def test_spectrum_computation_is_frozen_and_owns_read_only_result_arrays(
    blank_echem_factory,
    ecat_module,
    method,
):
    source_window = np.ones(32 if method == "welch" else 64)
    trace, options, warning_messages = _resolved_signal(
        blank_echem_factory,
        ecat_module,
        np.cos(2 * np.pi * np.arange(64) / 8),
        sample_rate=64.0,
        method=method,
        points_per_window=32,
        window=source_window,
    )

    computation = spectral._compute_trace(trace, options, warning_messages)

    assert isinstance(computation, spectral._SpectrumComputation)
    with pytest.raises(FrozenInstanceError):
        computation.nfft = 1
    assert isinstance(computation.window, np.ndarray)
    assert not np.shares_memory(computation.window, source_window)
    assert computation.window.flags.writeable is False
    with pytest.raises(ValueError, match="read-only"):
        computation.window[0] = 2
    arrays = [computation.frequency]
    arrays.extend(
        values
        for values in (computation.psd, computation.amplitude, computation.phase)
        if values is not None
    )
    for values in arrays:
        assert values.flags.writeable is False
        assert not np.shares_memory(values, trace.signal)
        with pytest.raises(ValueError, match="read-only"):
            values.flat[0] = values.flat[0]


@pytest.mark.parametrize(
    "source_window",
    [
        pytest.param([1.0] * 64, id="list"),
        pytest.param(pd.Series(np.ones(64)), id="series"),
    ],
)
def test_spectrum_computation_freezes_custom_array_like_window_metadata(
    blank_echem_factory,
    ecat_module,
    source_window,
):
    trace, options, warning_messages = _resolved_signal(
        blank_echem_factory,
        ecat_module,
        np.cos(2 * np.pi * np.arange(64) / 8),
        sample_rate=64.0,
        method="fft",
        detrend=None,
        window=source_window,
    )

    computation = spectral._compute_trace(trace, options, warning_messages)
    if isinstance(source_window, pd.Series):
        source_window.iloc[0] = 7.0
    else:
        source_window[0] = 7.0

    assert computation.window[0] == pytest.approx(1.0)
    with pytest.raises(ValueError, match="read-only"):
        computation.window[0] = 2.0
    assert isinstance(computation.window, np.ndarray)
    assert computation.window.flags.writeable is False


def test_spectrum_computation_freezes_array_array_window_metadata(
    blank_echem_factory,
    ecat_module,
):
    source_window = array("d", [1.0] * 64)
    trace, options, warning_messages = _resolved_signal(
        blank_echem_factory,
        ecat_module,
        np.cos(2 * np.pi * np.arange(64) / 8),
        sample_rate=64.0,
        method="fft",
        detrend=None,
        window=source_window,
    )

    computation = spectral._compute_trace(trace, options, warning_messages)
    source_window[0] = 7.0

    assert computation.window[0] == pytest.approx(1.0)
    with pytest.raises(ValueError, match="read-only"):
        computation.window[0] = 2.0
    assert isinstance(computation.window, np.ndarray)
    assert computation.window.flags.writeable is False


def test_spectrum_computation_recursively_freezes_named_window_parameters(
    blank_echem_factory,
    ecat_module,
):
    weights = [0.5, 0.5]
    window_specification = ("general_cosine", weights)
    trace, options, warning_messages = _resolved_signal(
        blank_echem_factory,
        ecat_module,
        np.cos(2 * np.pi * np.arange(64) / 8),
        sample_rate=64.0,
        method="fft",
        detrend=None,
        window=window_specification,
    )

    computation = spectral._compute_trace(trace, options, warning_messages)
    weights[0] = 0.75

    assert list(computation.window[1]) == [0.5, 0.5]
    with pytest.raises(TypeError):
        computation.window[1][0] = 0.25
    assert computation.window == ("general_cosine", (0.5, 0.5))


@pytest.mark.parametrize(
    ("method", "expected_scales"),
    [("welch", ("linear", "log")), ("fft", ("linear", "linear"))],
)
def test_spectrum_plot_auto_scale_follows_method(
    blank_echem_factory,
    ecat_module,
    method,
    expected_scales,
):
    obj = _timed_object(blank_echem_factory, ecat_module.ca)

    result, _, _, _ = _plotting_case(obj, method=method)

    assert result.axes.get_xscale() == expected_scales[0]
    assert result.axes.get_yscale() == expected_scales[1]
    assert result.figure.get_layout_engine() is not None


@pytest.mark.parametrize(
    ("plot_scale", "expected_scales"),
    [
        ("linear", ("linear", "linear")),
        ("log-log", ("log", "log")),
        ("semilogx", ("log", "linear")),
        ("semilogy", ("linear", "log")),
        ("symlog", ("symlog", "symlog")),
    ],
)
def test_spectrum_plot_scale_presets_map_to_matplotlib_axes(
    blank_echem_factory,
    ecat_module,
    plot_scale,
    expected_scales,
):
    obj = _timed_object(blank_echem_factory, ecat_module.ca)

    result, _, _, _ = _plotting_case(obj, plot_scale=plot_scale)

    assert (result.axes.get_xscale(), result.axes.get_yscale()) == expected_scales


def test_spectrum_direct_axis_scales_override_only_corresponding_preset_axis(
    blank_echem_factory,
    ecat_module,
):
    obj = _timed_object(blank_echem_factory, ecat_module.ca)

    x_override, _, _, _ = _plotting_case(
        obj,
        plot_scale="log-log",
        xscale="linear",
    )
    y_override, _, _, _ = _plotting_case(
        obj,
        plot_scale="log-log",
        yscale="linear",
    )

    assert (x_override.axes.get_xscale(), x_override.axes.get_yscale()) == (
        "linear",
        "log",
    )
    assert (y_override.axes.get_xscale(), y_override.axes.get_yscale()) == (
        "log",
        "linear",
    )


@pytest.mark.parametrize(
    ("method", "class_name", "column", "unit", "expected_ylabel"),
    [
        ("welch", "ca", "Current", "A", r"PSD ($\mathrm{A}^{2}\,\mathrm{Hz}^{-1}$)"),
        ("welch", "ca", "Current", "mA", r"PSD ($\mathrm{mA}^{2}\,\mathrm{Hz}^{-1}$)"),
        ("fft", "cp", "Potential", "V", r"Amplitude ($\mathrm{V}$)"),
        (
            "welch",
            "echem",
            "Response",
            "",
            r"PSD ($\mathrm{signal\ units}^{2}\,\mathrm{Hz}^{-1}$)",
        ),
        ("fft", "echem", "Response", "", "Amplitude (signal units)"),
    ],
)
def test_spectrum_plot_labels_follow_selected_signal_units(
    blank_echem_factory,
    ecat_module,
    method,
    class_name,
    column,
    unit,
    expected_ylabel,
):
    sample_rate = 64.0
    time = np.arange(64, dtype=float) / sample_rate
    obj = blank_echem_factory(getattr(ecat_module, class_name))
    obj.data = pd.DataFrame(
        {"Time": time, column: np.sin(2 * np.pi * 8 * time)}
    )
    obj.units = {"Time": "s", column: unit}

    result, _, _, _ = _plotting_case(obj, method=method)

    assert result.axes.get_xlabel() == "Frequency (Hz)"
    assert result.axes.get_ylabel() == expected_ylabel
    if unit == "":
        assert "(A)" not in result.axes.get_ylabel()
        assert r"\mathrm{A}" not in result.axes.get_ylabel()
        assert "(V)" not in result.axes.get_ylabel()
        assert r"\mathrm{V}" not in result.axes.get_ylabel()
    result.figure.canvas.draw()


def test_spectrum_plot_honors_custom_text_and_grid(
    blank_echem_factory,
    ecat_module,
):
    obj = _timed_object(blank_echem_factory, ecat_module.ca)

    result, _, _, _ = _plotting_case(
        obj,
        title="Electrode Noise",
        subtitle="Steady-state interval",
        xlabel="Custom frequency",
        ylabel="Custom response",
        grid=True,
    )

    assert result.figure._suptitle.get_text() == "Electrode Noise"
    assert result.axes.get_title() == "Steady-state interval"
    assert result.axes.get_xlabel() == "Custom frequency"
    assert result.axes.get_ylabel() == "Custom response"
    assert any(line.get_visible() for line in result.axes.get_xgridlines())
    assert any(line.get_visible() for line in result.axes.get_ygridlines())


def test_two_cv_segments_make_unique_primary_and_diagnostic_traces(
    blank_echem_factory,
    ecat_module,
):
    obj = _two_segment_cv(blank_echem_factory, ecat_module, recorded_time=True)

    result, traces, _, _ = _plotting_case(obj, method="fft", plot_all=True)
    primary_lines = result.axes.lines
    diagnostic_lines = result.figures[1].axes[0].lines

    assert [line.get_label() for line in primary_lines] == ["Segment 1", "Segment 2"]
    assert [line.get_label() for line in diagnostic_lines] == [
        "Segment 1",
        "Segment 2",
    ]
    assert len(primary_lines) == len(diagnostic_lines) == 2
    for line, trace in zip(diagnostic_lines, traces):
        assert line.get_xdata() == pytest.approx(trace.time)
        assert line.get_ydata() == pytest.approx(trace.signal)


@pytest.mark.parametrize(("plot_all", "figure_count"), [(False, 1), (True, 2)])
def test_spectrum_plot_all_controls_diagnostic_figure(
    blank_echem_factory,
    ecat_module,
    plot_all,
    figure_count,
):
    obj = _timed_object(blank_echem_factory, ecat_module.ca)

    result, _, _, _ = _plotting_case(obj, plot_all=plot_all)

    assert len(result.figures) == figure_count
    assert result.figures[0] is result.figure
    assert result.axes is result.figure.axes[0]
    assert all(figure.get_layout_engine() is not None for figure in result.figures)


def test_resampled_diagnostic_shows_original_markers_and_resolved_line(
    blank_echem_factory,
    ecat_module,
):
    time = np.array([0.0, 1.0, 2.0, 3.2, 4.2, 5.2, 6.2, 7.2, 8.2, 9.2])
    obj = _timed_object(
        blank_echem_factory,
        ecat_module.ca,
        time=time,
    )

    result, traces, _, _ = _plotting_case(
        obj,
        method="fft",
        resample=True,
        sampling_tolerance=0.05,
        plot_all=True,
    )
    trace = traces[0]
    original, resolved = result.figures[1].axes[0].lines

    assert trace.resampled is True
    assert original.get_linestyle() == "None"
    assert original.get_marker() != "None"
    assert original.get_alpha() < 0.5
    assert original.get_xdata() == pytest.approx(trace.original_time)
    assert original.get_ydata() == pytest.approx(trace.original_signal)
    assert resolved.get_linestyle() != "None"
    assert resolved.get_xdata() == pytest.approx(trace.time)
    assert resolved.get_ydata() == pytest.approx(trace.signal)
    assert "original" in original.get_label().casefold()
    assert "resolved" in resolved.get_label().casefold()
    assert result.figures[1].axes[0].get_legend() is not None


def test_spectrum_colors_and_outside_legend_follow_options(
    blank_echem_factory,
    ecat_module,
):
    obj = _two_segment_cv(blank_echem_factory, ecat_module, recorded_time=True)

    listed, _, _, _ = _plotting_case(
        obj,
        method="fft",
        colors=["tab:red", "tab:green"],
        legend_loc="lower right",
        legend_outside=True,
    )
    overridden, _, _, _ = _plotting_case(
        obj,
        method="fft",
        color="black",
        colors=["tab:red", "tab:green"],
        legend_loc="upper left",
    )

    assert [line.get_color() for line in listed.axes.lines] == [
        "tab:red",
        "tab:green",
    ]
    outside_legend = listed.axes.get_legend()
    assert [text.get_text() for text in outside_legend.get_texts()] == [
        "Segment 1",
        "Segment 2",
    ]
    assert outside_legend._loc == mpl.legend.Legend.codes["lower left"]
    assert outside_legend.get_bbox_to_anchor()._bbox.x0 > 1.0
    assert [line.get_color() for line in overridden.axes.lines] == ["black", "black"]
    assert overridden.axes.get_legend()._loc == mpl.legend.Legend.codes["upper left"]


def test_spectrum_plotting_does_not_mutate_options_arrays_or_rcparams(
    blank_echem_factory,
    ecat_module,
):
    obj = _two_segment_cv(blank_echem_factory, ecat_module, recorded_time=True)
    options = _options(
        plot=True,
        method="fft",
        plot_all=True,
        colors=["tab:orange", "tab:cyan"],
    )
    traces, warning_messages = _resolve_spectrum_traces(obj, options)
    computations = [
        spectral._compute_trace(trace, options, warning_messages) for trace in traces
    ]
    result = AnalysisResult(diagnostics={"traces": traces})
    options_before = copy.deepcopy(options)
    computation_arrays = [
        (
            computation.frequency.copy(),
            computation.amplitude.copy(),
            computation.phase.copy(),
        )
        for computation in computations
    ]
    trace_arrays = [
        (
            trace.time.copy(),
            trace.signal.copy(),
            trace.original_time.copy(),
            trace.original_signal.copy(),
        )
        for trace in traces
    ]
    rcparams_before = mpl.rcParams.copy()

    spectral._attach_spectrum_figures(result, computations, options)

    assert options == options_before
    assert mpl.rcParams == rcparams_before
    for computation, expected in zip(computations, computation_arrays):
        for actual, snapshot in zip(
            (computation.frequency, computation.amplitude, computation.phase),
            expected,
        ):
            assert np.array_equal(actual, snapshot)
            assert actual.flags.writeable is False
    for trace, expected in zip(traces, trace_arrays):
        for actual, snapshot in zip(
            (trace.time, trace.signal, trace.original_time, trace.original_signal),
            expected,
        ):
            assert np.array_equal(actual, snapshot)
            assert actual.flags.writeable is False


def test_spectrum_plot_false_leaves_result_without_figures(
    blank_echem_factory,
    ecat_module,
):
    obj = _timed_object(blank_echem_factory, ecat_module.ca)
    options = _options(plot=False)
    traces, warning_messages = _resolve_spectrum_traces(obj, options)
    computations = [spectral._compute_trace(traces[0], options, warning_messages)]
    result = AnalysisResult(diagnostics={"traces": traces})

    returned = spectral._attach_spectrum_figures(result, computations, options)

    assert returned is result
    assert result.figure is None
    assert result.axes is None
    assert result.figures == []
    assert plt.get_fignums() == []


def test_public_spectrum_validates_direct_spectrum_options_instance(
    blank_echem_factory,
    ecat_module,
):
    obj = _spectrum_analysis_object(blank_echem_factory, ecat_module)
    options = SpectrumOptions(
        sampling_tolerance=float("nan"),
        plot=False,
        print=False,
    )

    with pytest.raises(
        ecat_module.OptionError,
        match="sampling tolerance.*finite nonnegative",
    ):
        obj.spectrum(options)


def test_welch_auto_scale_falls_back_to_linear_for_nonpositive_psd(
    blank_echem_factory,
    ecat_module,
):
    obj = _timed_object(blank_echem_factory, ecat_module.ca)
    obj.data["Current"] = 0.0

    result, _, computations, _ = _plotting_case(obj, plot_scale="auto")

    assert np.all(computations[0].psd == 0)
    assert result.axes.get_yscale() == "linear"


@pytest.mark.parametrize(
    "scale_options",
    [
        pytest.param({"plot scale": "semilogy"}, id="preset"),
        pytest.param(
            {"plot scale": "linear", "yscale": "log"},
            id="direct-override",
        ),
    ],
)
def test_explicit_log_y_rejects_nonpositive_spectrum_before_figure_creation(
    blank_echem_factory,
    ecat_module,
    scale_options,
):
    obj = _timed_object(blank_echem_factory, ecat_module.ca)
    obj.data["Current"] = 0.0
    figure_numbers_before = set(plt.get_fignums())

    with pytest.raises(ValueError, match="logarithmic y-axis.*finite positive"):
        _plotting_case(obj, **scale_options)

    assert set(plt.get_fignums()) == figure_numbers_before


def test_symlog_accepts_nonpositive_spectrum(
    blank_echem_factory,
    ecat_module,
):
    obj = _timed_object(blank_echem_factory, ecat_module.ca)
    obj.data["Current"] = 0.0

    result, _, _, _ = _plotting_case(obj, plot_scale="symlog")

    assert result.axes.get_xscale() == "symlog"
    assert result.axes.get_yscale() == "symlog"


@pytest.mark.parametrize(
    "scale_options",
    [
        pytest.param({"plot scale": "semilogx"}, id="semilogx"),
        pytest.param({"plot scale": "log-log"}, id="log-log"),
        pytest.param(
            {"plot scale": "linear", "xscale": "log"},
            id="direct-override",
        ),
    ],
)
def test_log_x_rejects_retained_dc_before_figure_creation(
    blank_echem_factory,
    ecat_module,
    scale_options,
):
    obj = _timed_object(blank_echem_factory, ecat_module.ca)
    figure_numbers_before = set(plt.get_fignums())

    with pytest.raises(
        ValueError,
        match="include dc=False.*non-log x scale",
    ):
        _plotting_case(
            obj,
            method="fft",
            include_dc=True,
            **scale_options,
        )

    assert set(plt.get_fignums()) == figure_numbers_before


@pytest.mark.parametrize(
    "missing_unit",
    [
        pytest.param(None, id="none"),
        pytest.param(np.nan, id="nan"),
        pytest.param(pd.NA, id="pandas-na"),
    ],
)
def test_missing_signal_unit_metadata_uses_generic_plot_labels(
    blank_echem_factory,
    ecat_module,
    missing_unit,
):
    obj = _timed_object(blank_echem_factory, ecat_module.ca)
    obj.units["Current"] = missing_unit

    result, traces, computations, _ = _plotting_case(obj, plot_all=True)
    diagnostic_axis = result.figures[1].axes[0]

    assert traces[0].signal_unit == ""
    assert computations[0].signal_unit == ""
    assert result.axes.get_ylabel() == (
        r"PSD ($\mathrm{signal\ units}^{2}\,\mathrm{Hz}^{-1}$)"
    )
    assert diagnostic_axis.get_ylabel() == "Current (signal units)"
    for label in (result.axes.get_ylabel(), diagnostic_axis.get_ylabel()):
        assert r"\mathrm{A}" not in label
        assert r"\mathrm{V}" not in label


def test_spectrum_label_helpers_treat_none_unit_as_unknown():
    computation = type(
        "Computation",
        (),
        {"signal_unit": None, "method": "fft"},
    )()
    trace = type(
        "Trace",
        (),
        {"signal_unit": None, "signal_column": "Response"},
    )()

    assert spectral._spectrum_axis_ylabel(computation) == "Amplitude (signal units)"
    assert spectral._time_domain_ylabel(trace) == "Response (signal units)"


@pytest.mark.parametrize(
    ("diagnostics", "message"),
    [
        ({}, "resolved traces"),
        ({"traces": []}, "one resolved trace"),
    ],
)
def test_plot_all_validates_diagnostics_before_creating_figures(
    blank_echem_factory,
    ecat_module,
    diagnostics,
    message,
):
    obj = _timed_object(blank_echem_factory, ecat_module.ca)
    options = _options(plot=True, plot_all=True)
    traces, warning_messages = _resolve_spectrum_traces(obj, options)
    computations = [spectral._compute_trace(traces[0], options, warning_messages)]
    result = AnalysisResult(diagnostics=diagnostics)
    existing_figure = plt.figure()
    figure_numbers_before = set(plt.get_fignums())

    with pytest.raises(ValueError, match=message):
        spectral._attach_spectrum_figures(result, computations, options)

    assert set(plt.get_fignums()) == figure_numbers_before
    assert plt.fignum_exists(existing_figure.number)


def test_plotting_exception_closes_all_locally_created_figures(
    blank_echem_factory,
    ecat_module,
    monkeypatch,
):
    obj = _two_segment_cv(blank_echem_factory, ecat_module, recorded_time=True)
    options = _options(plot=True, plot_all=True, method="fft")
    traces, warning_messages = _resolve_spectrum_traces(obj, options)
    computations = [
        spectral._compute_trace(trace, options, warning_messages) for trace in traces
    ]
    result = AnalysisResult(diagnostics={"traces": traces})
    original_apply_style = spectral._apply_ecat_axis_style
    style_calls = 0

    def fail_on_diagnostic(axis, style_options):
        nonlocal style_calls
        style_calls += 1
        if style_calls == 2:
            raise RuntimeError("diagnostic rendering failed")
        return original_apply_style(axis, style_options)

    monkeypatch.setattr(spectral, "_apply_ecat_axis_style", fail_on_diagnostic)
    existing_figure = plt.figure()
    figure_numbers_before = set(plt.get_fignums())

    with pytest.raises(RuntimeError, match="diagnostic rendering failed"):
        spectral._attach_spectrum_figures(result, computations, options)

    assert style_calls == 2
    assert set(plt.get_fignums()) == figure_numbers_before
    assert plt.fignum_exists(existing_figure.number)


@pytest.mark.parametrize(
    ("method", "value_column", "expected_units", "metadata"),
    [
        (
            "welch",
            "PSD / A^2 Hz^-1",
            {"Frequency / Hz": "Hz", "PSD / A^2 Hz^-1": "A^2 Hz^-1"},
            {"nperseg": 64, "noverlap": 32, "window count": 3},
        ),
        (
            "fft",
            "Amplitude / A",
            {
                "Frequency / Hz": "Hz",
                "Amplitude / A": "A",
                "Phase / rad": "rad",
            },
            {"coherent gain": 1.0},
        ),
    ],
)
def test_spectrum_analysis_assembles_standard_result_contract(
    blank_echem_factory,
    ecat_module,
    method,
    value_column,
    expected_units,
    metadata,
):
    obj = _spectrum_analysis_object(blank_echem_factory, ecat_module)
    options = {
        "method": method,
        "window": "boxcar",
        "detrend": None,
        "points per window": 64,
        "plot": False,
        "print": False,
    }

    result = spectral.spectrum_analysis(obj, options)

    assert isinstance(result, AnalysisResult)
    assert result["data"] is result.table
    assert result["summary"] is result.summary
    assert result.raw_table is result.table
    assert value_column in result.table
    assert result.table.attrs["units"] == spectral._spectrum_units("A", method)
    assert result.units == expected_units
    assert result.summary["analysis"] == "spectrum"
    assert result.summary["method"] == method
    assert result.summary["signal column"] == "Current"
    assert result.summary["signal unit"] == "A"
    assert result.summary["selected segments"] == "all"
    assert result.summary["time source"] == "recorded"
    assert result.summary["sample count"] == 128
    assert result.summary["sample rate / Hz"] == pytest.approx(64.0)
    assert result.summary["Nyquist frequency / Hz"] == pytest.approx(32.0)
    expected_bin_spacing = 1.0 if method == "welch" else 0.5
    assert result.summary["frequency bin spacing / Hz"] == pytest.approx(
        expected_bin_spacing
    )
    assert result.summary["resampled"] is False
    assert result.summary["peak frequency / Hz"] == pytest.approx(8.0)
    assert isinstance(result.diagnostics["traces"], tuple)
    assert isinstance(result.diagnostics["computations"], tuple)
    assert len(result.diagnostics["traces"]) == 1
    assert len(result.diagnostics["computations"]) == 1
    assert result.diagnostics["method"] == method
    assert result.diagnostics["window"] == "boxcar"
    assert result.diagnostics["detrend"] is None
    for key, value in metadata.items():
        assert result.diagnostics[key] == pytest.approx(value)
    assert result.figure is None
    assert result.axes is None
    assert result.figures == []


@pytest.mark.parametrize(
    ("unit", "value_column", "value_unit"),
    [
        ("A", "PSD / A^2 Hz^-1", "A^2 Hz^-1"),
        (
            "A cm^-2",
            "PSD / (A cm^-2)^2 Hz^-1",
            "(A cm^-2)^2 Hz^-1",
        ),
        ("", "PSD / signal^2 Hz^-1", "signal^2 Hz^-1"),
    ],
)
def test_spectrum_result_units_match_dynamic_table_schema(
    blank_echem_factory,
    ecat_module,
    unit,
    value_column,
    value_unit,
):
    obj = _spectrum_analysis_object(blank_echem_factory, ecat_module)
    obj.units["Current"] = unit

    result = spectral.spectrum_analysis(
        obj,
        {
            "method": "welch",
            "window": "boxcar",
            "points per window": 64,
            "plot": False,
            "print": False,
        },
    )

    assert value_column in result.table
    assert result.units == result.table.attrs["units"]
    assert result.units == {
        "Frequency / Hz": "Hz",
        value_column: value_unit,
    }


def test_spectrum_peak_frequency_requires_a_finite_positive_ordinate():
    computation = spectral._SpectrumComputation(
        segment=None,
        label="flat",
        signal_unit="A",
        method="fft",
        frequency=np.array([1.0, 2.0, 3.0]),
        amplitude=np.array([np.nan, 0.0, -1.0]),
        sample_count=8,
        sample_rate=8.0,
        nyquist=4.0,
        frequency_bin_spacing=1.0,
        nfft=8,
        window="boxcar",
        detrend=None,
    )

    assert np.isnan(spectral._spectrum_peak_frequency(computation))


def test_spectrum_peak_frequency_tie_uses_lowest_retained_frequency():
    computation = spectral._SpectrumComputation(
        segment=None,
        label="tied",
        signal_unit="A",
        method="welch",
        frequency=np.array([1.0, 2.0, 3.0, 4.0]),
        psd=np.array([0.5, 2.0, 2.0, 1.0]),
        sample_count=8,
        sample_rate=8.0,
        nyquist=4.0,
        frequency_bin_spacing=1.0,
        nfft=8,
        window="boxcar",
        detrend=None,
        nperseg=8,
        noverlap=4,
        window_count=1,
    )

    assert np.all(np.diff(computation.frequency) > 0)
    assert spectral._spectrum_peak_frequency(computation) == 2.0


def test_flat_spectrum_keeps_nan_peak_and_displays_not_detected(
    blank_echem_factory,
    ecat_module,
    monkeypatch,
):
    obj = _spectrum_analysis_object(blank_echem_factory, ecat_module)
    obj.data["Current"] = 0.0
    displayed = {}

    def capture_table(table, options, *, title=None, **kwargs):
        displayed[title] = table.copy()
        return table

    monkeypatch.setattr(spectral, "_display_table", capture_table)

    result = spectral.spectrum_analysis(
        obj,
        {
            "method": "fft",
            "window": "boxcar",
            "plot": False,
            "print": True,
            "pretty print": False,
        },
    )

    assert np.isnan(result.summary["peak frequency / Hz"])
    summary = displayed["Spectrum Analysis Summary"].set_index("Setting")["Value"]
    assert summary["Peak Frequency / Hz"] == "not detected"


@pytest.mark.parametrize("method", ["welch", "fft"])
def test_spectrum_parameter_rows_are_exact_and_method_specific(
    blank_echem_factory,
    ecat_module,
    monkeypatch,
    method,
):
    obj = _spectrum_analysis_object(blank_echem_factory, ecat_module)
    displayed = {}

    def capture_table(table, options, *, title=None, **kwargs):
        displayed[title] = table.copy()
        return table

    monkeypatch.setattr(spectral, "_display_table", capture_table)

    result = spectral.spectrum_analysis(
        obj,
        {
            "method": method,
            "window": "boxcar",
            "points per window": 64,
            "detrend": None,
            "plot": False,
            "print": True,
            "pretty print": False,
        },
    )

    shared_rows = [
        "Method",
        "Signal Column",
        "Signal Unit",
        "Detrend",
        "Window",
        "Resampling",
        "Segments",
        "Time Range",
        "Frequency Range",
        "Include DC",
        "Sampling Tolerance",
        "NFFT",
    ]
    parameter_rows = displayed["Spectrum Analysis Parameters"]["Setting"].tolist()
    welch_rows = ["Points per Window", "Overlap"]
    assert parameter_rows == shared_rows + (welch_rows if method == "welch" else [])
    assert ("Points per Window" in parameter_rows) is (method == "welch")
    assert ("Overlap" in parameter_rows) is (method == "welch")
    assert parameter_rows.count("NFFT") == 1

    parameters = displayed["Spectrum Analysis Parameters"].set_index("Setting")[
        "Value"
    ]
    assert parameters["Time Range"] == "all"
    assert parameters["Frequency Range"] == "all"
    assert parameters["Include DC"] == "No"
    assert parameters["Detrend"] == "None"
    assert parameters["NFFT"].startswith("auto (")


def test_spectrum_consequential_parameters_are_displayed_and_stored_compactly(
    blank_echem_factory,
    ecat_module,
    monkeypatch,
):
    obj = _spectrum_analysis_object(blank_echem_factory, ecat_module)
    displayed = {}

    def capture_table(table, options, *, title=None, **kwargs):
        displayed[title] = table.copy()
        return table

    monkeypatch.setattr(spectral, "_display_table", capture_table)

    result = spectral.spectrum_analysis(
        obj,
        {
            "method": "welch",
            "window": "boxcar",
            "points per window": 64,
            "time range": [0.0, 1.9],
            "frequency range": [1.0, 20.0],
            "include dc": True,
            "sampling tolerance": 0.02,
            "nfft": 256,
            "plot": False,
            "print": True,
            "pretty print": False,
        },
    )

    parameters = displayed["Spectrum Analysis Parameters"].set_index("Setting")[
        "Value"
    ]
    assert parameters["Time Range"] != "all"
    assert parameters["Frequency Range"] != "all"
    assert parameters["Include DC"] == "Yes"
    assert parameters["Sampling Tolerance"] != ""
    assert parameters["NFFT"] == "256"

    resolved = result.diagnostics["parameters"]
    assert resolved["method"] == "welch"
    assert resolved["window"] == "boxcar"
    assert resolved["time range"] == (0.0, 1.9)
    assert resolved["frequency range"] == (1.0, 20.0)
    assert resolved["include dc"] is True
    assert resolved["sampling tolerance"] == pytest.approx(0.02)
    assert resolved["nfft"] == 256
    assert not any(isinstance(value, np.ndarray) for value in resolved.values())


@pytest.mark.parametrize(
    ("window", "expected"),
    [
        (("kaiser", 2), "kaiser (2)"),
        (("kaiser", 14), "kaiser (14)"),
        (("general_cosine", [0.5, 0.5]), "general_cosine (2 values)"),
    ],
)
def test_spectrum_parameterized_window_display_is_compact_and_distinct(
    blank_echem_factory,
    ecat_module,
    monkeypatch,
    window,
    expected,
):
    obj = _spectrum_analysis_object(blank_echem_factory, ecat_module)
    displayed = {}

    def capture_table(table, options, *, title=None, **kwargs):
        displayed[title] = table.copy()
        return table

    monkeypatch.setattr(spectral, "_display_table", capture_table)

    result = spectral.spectrum_analysis(
        obj,
        {
            "method": "fft",
            "window": window,
            "plot": False,
            "print": True,
            "pretty print": False,
        },
    )

    parameters = displayed["Spectrum Analysis Parameters"].set_index("Setting")[
        "Value"
    ]
    assert parameters["Window"] == expected
    assert result.diagnostics["parameters"]["window"] == expected


def test_multi_trace_summary_preserves_unequal_reconstructed_intervals(
    blank_echem_factory,
    ecat_module,
):
    forward = np.linspace(-0.3, 0.3, 12)
    reverse = np.linspace(0.3, -0.3, 9)[1:]
    obj = blank_echem_factory(ecat_module.cv)
    obj.data = pd.DataFrame(
        {
            "Potential": np.concatenate([forward, reverse]),
            "Current": np.cos(np.arange(20, dtype=float)),
        }
    )
    obj.units = {"Potential": "V", "Current": "A"}
    obj.scan_rate = 0.1

    with pytest.warns(UserWarning, match="reconstructed"):
        result = spectral.spectrum_analysis(
            obj,
            {
                "method": "fft",
                "window": "boxcar",
                "detrend": None,
                "plot": False,
                "print": False,
            },
        )

    assert result.summary["selected segments"] == [1, 2]
    assert result.summary["time source"] == (
        "reconstructed from CV potential and scan rate"
    )
    assert result.summary["resampled"] is False
    assert result.summary["sample count"] == OrderedDict(
        [("Segment 1", 12), ("Segment 2", 8)]
    )
    sample_rates = result.summary["sample rate / Hz"]
    assert isinstance(sample_rates, OrderedDict)
    assert list(sample_rates) == ["Segment 1", "Segment 2"]
    assert sample_rates["Segment 1"] == pytest.approx(11.0 / 6.0)
    assert sample_rates["Segment 2"] == pytest.approx(4.0 / 3.0)
    assert isinstance(result.summary["Nyquist frequency / Hz"], OrderedDict)
    assert isinstance(result.summary["frequency bin spacing / Hz"], OrderedDict)
    assert isinstance(result.summary["peak frequency / Hz"], OrderedDict)


def test_multi_trace_summary_does_not_collapse_nearby_physical_sample_rates(
    blank_echem_factory,
    ecat_module,
):
    first_interval = 0.1
    second_interval = 0.10000000000005
    obj = blank_echem_factory(ecat_module.ca)
    obj.data = pd.DataFrame(
        {
            "Time": np.concatenate(
                [
                    np.arange(8, dtype=float) * first_interval,
                    10.0 + np.arange(8, dtype=float) * second_interval,
                ]
            ),
            "Current": np.cos(np.arange(16, dtype=float)),
            "Segment": np.repeat([1, 2], 8),
        }
    )
    obj.units = {"Time": "s", "Current": "A"}

    result = spectral.spectrum_analysis(
        obj,
        {
            "method": "fft",
            "segments": [1, 2],
            "window": "boxcar",
            "detrend": None,
            "plot": False,
            "print": False,
        },
    )

    sample_rates = result.summary["sample rate / Hz"]
    assert isinstance(sample_rates, OrderedDict)
    assert sample_rates["Segment 1"] != sample_rates["Segment 2"]


def test_spectrum_trace_keys_are_unique_against_all_previously_issued_keys():
    traces = [
        SimpleNamespace(segment=None, label="trace"),
        SimpleNamespace(segment=None, label="trace (2)"),
        SimpleNamespace(segment=None, label="trace"),
    ]

    keys = spectral._spectrum_trace_keys(traces)
    mapping = spectral._spectrum_audit_value([10, 20, 30], keys)

    assert keys == ["trace", "trace (2)", "trace (3)"]
    assert isinstance(mapping, OrderedDict)
    assert len(mapping) == 3
    assert list(mapping.values()) == [10, 20, 30]


def test_spectrum_analysis_deduplicates_stored_and_emitted_warnings(
    blank_echem_factory,
    ecat_module,
    monkeypatch,
):
    obj = _spectrum_analysis_object(blank_echem_factory, ecat_module)
    options = _options(method="fft", window="boxcar", detrend=None)
    traces, warning_messages = _resolve_spectrum_traces(obj, options)
    computation = spectral._compute_trace(traces[0], options, warning_messages)
    message = "identical spectrum warning"

    monkeypatch.setattr(
        spectral,
        "_resolve_spectrum_traces",
        lambda analysis_obj, resolved: (traces, [message, message]),
    )

    def duplicate_warning(trace, resolved, messages):
        messages.extend([message, message])
        return computation

    monkeypatch.setattr(spectral, "_compute_trace", duplicate_warning)

    with warnings.catch_warnings(record=True) as emitted:
        warnings.simplefilter("always")
        result = spectral.spectrum_analysis(
            obj,
            {"method": "fft", "plot": False, "print": False},
        )

    assert result.warnings == [message]
    assert [str(item.message) for item in emitted] == [message]
    assert emitted[0].category is UserWarning


def test_warning_as_error_prevents_plot_creation(
    blank_echem_factory,
    ecat_module,
):
    obj = _two_segment_cv(blank_echem_factory, ecat_module)
    figure_numbers_before = set(plt.get_fignums())

    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        with pytest.raises(UserWarning, match="reconstructed"):
            spectral.spectrum_analysis(
                obj,
                {
                    "method": "fft",
                    "window": "boxcar",
                    "plot": True,
                    "print": False,
                },
            )

    assert set(plt.get_fignums()) == figure_numbers_before


def test_warning_is_emitted_once_before_a_later_plotting_error(
    blank_echem_factory,
    ecat_module,
    monkeypatch,
):
    obj = _spectrum_analysis_object(blank_echem_factory, ecat_module)
    options = _options(method="fft", window="boxcar", detrend=None)
    traces, _ = _resolve_spectrum_traces(obj, options)
    message = "warning before plotting"
    emitted = []

    monkeypatch.setattr(
        spectral,
        "_resolve_spectrum_traces",
        lambda analysis_obj, resolved: (traces, [message, message]),
    )
    monkeypatch.setattr(
        spectral.warnings,
        "warn",
        lambda text, category, stacklevel: emitted.append(
            (text, category, stacklevel)
        ),
    )
    monkeypatch.setattr(
        spectral,
        "_attach_spectrum_figures",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("plot failed")),
    )

    with pytest.raises(RuntimeError, match="plot failed"):
        spectral.spectrum_analysis(
            obj,
            {
                "method": "fft",
                "window": "boxcar",
                "plot": True,
                "print": False,
            },
        )

    assert emitted == [(message, UserWarning, 3)]


@pytest.mark.parametrize("print_all", [False, True])
def test_spectrum_plain_report_has_contract_order_and_optional_data(
    blank_echem_factory,
    ecat_module,
    capsys,
    print_all,
):
    obj = _spectrum_analysis_object(blank_echem_factory, ecat_module)

    spectral.spectrum_analysis(
        obj,
        {
            "method": "fft",
            "window": "boxcar",
            "detrend": None,
            "plot": False,
            "print": True,
            "print all": print_all,
            "pretty print": False,
        },
    )

    output = capsys.readouterr().out
    headings = [
        "Spectrum Analysis Parameters:",
        "Spectrum Analysis Equations:",
        "Spectrum Analysis Summary:",
    ]
    assert all(heading in output for heading in headings)
    assert output.index(headings[0]) < output.index(headings[1]) < output.index(
        headings[2]
    )
    assert (
        "Spectrum Analysis Data:" in output
    ) is print_all
    assert (
        "X_k = sum_(n=0)^(N-1) x_n w_n exp(-i 2 pi k n / M); "
        "A_k = c_k |X_k| / (N Gw)"
    ) in output
    assert "c_k is 1 at DC and Nyquist and 2 for interior one-sided bins" in output


@pytest.mark.parametrize(
    ("method", "symbolic", "definitions"),
    [
        (
            "welch",
            "Sxx(f_k) = c_k/K sum_(j=1)^K |sum_(n=0)^(N-1) w_n "
            "x_(j,n) exp(-i 2 pi k n / M)|^2 / "
            "(fs sum_(n=0)^(N-1) w_n^2)",
            "N is samples per window; M is nfft; K is the number of Welch "
            "windows; c_k is 1 at DC and Nyquist and 2 for interior one-sided "
            "bins; fs is the sample rate in Hz; Sxx has signal^2/Hz units",
        ),
        (
            "fft",
            "X_k = sum_(n=0)^(N-1) x_n w_n exp(-i 2 pi k n / M); "
            "A_k = c_k |X_k| / (N Gw)",
            "N is the sample count; M is nfft; Gw = N^-1 sum_n w_n; c_k is "
            "1 at DC and Nyquist and 2 for interior one-sided bins",
        ),
    ],
)
def test_spectrum_plain_equations_observably_match_the_computation(
    blank_echem_factory,
    ecat_module,
    capsys,
    method,
    symbolic,
    definitions,
):
    obj = _spectrum_analysis_object(blank_echem_factory, ecat_module)

    spectral.spectrum_analysis(
        obj,
        {
            "method": method,
            "window": "boxcar",
            "points per window": 64,
            "plot": False,
            "print": True,
            "pretty print": False,
        },
    )

    output = capsys.readouterr().out
    assert "Spectrum Analysis Equations:" in output
    assert symbolic in output
    assert definitions in output


def test_spectrum_pretty_mode_uses_plain_equations_outside_live_ipython(
    blank_echem_factory,
    ecat_module,
    monkeypatch,
    capsys,
):
    from ecat import analysis_batch

    obj = _spectrum_analysis_object(blank_echem_factory, ecat_module)
    monkeypatch.setattr(spectral, "_can_rich_table_display", lambda: False)
    monkeypatch.setattr(
        analysis_batch,
        "_display_analysis_equation",
        lambda *args, **kwargs: pytest.fail("ordinary Python must use plain equations"),
    )

    spectral.spectrum_analysis(
        obj,
        {
            "method": "fft",
            "window": "boxcar",
            "plot": False,
            "print": True,
            "pretty print": True,
        },
    )

    output = capsys.readouterr().out
    assert "Spectrum Analysis Equations:" in output
    assert "A_k = c_k |X_k| / (N Gw)" in output


@pytest.mark.parametrize(
    (
        "shared_capability",
        "shell_name",
        "has_kernel",
        "expected",
    ),
    [
        (False, "ZMQInteractiveShell", True, False),
        (True, "TerminalInteractiveShell", True, False),
        (True, "ZMQInteractiveShell", False, True),
        (True, "UnknownInteractiveShell", False, False),
        (True, "UnknownInteractiveShell", True, True),
    ],
)
def test_spectrum_notebook_rich_display_capability(
    monkeypatch,
    shared_capability,
    shell_name,
    has_kernel,
    expected,
):
    IPython = pytest.importorskip("IPython")
    attributes = {"kernel": object()} if has_kernel else {}
    shell = type(shell_name, (), attributes)()
    monkeypatch.setattr(
        spectral,
        "_can_rich_table_display",
        lambda: shared_capability,
    )
    monkeypatch.setattr(IPython, "get_ipython", lambda: shell)

    assert spectral._spectrum_notebook_rich_display_available() is expected


def test_terminal_ipython_pretty_request_prints_every_section_as_plain_text(
    blank_echem_factory,
    ecat_module,
    monkeypatch,
    capsys,
):
    IPython = pytest.importorskip("IPython")
    from ecat import analysis_batch

    class TerminalInteractiveShell:
        kernel = object()

    obj = _spectrum_analysis_object(blank_echem_factory, ecat_module)
    monkeypatch.setattr(spectral, "_can_rich_table_display", lambda: True)
    monkeypatch.setattr(
        IPython,
        "get_ipython",
        lambda: TerminalInteractiveShell(),
    )
    monkeypatch.setattr(
        analysis_batch,
        "_display_analysis_equation",
        lambda *args, **kwargs: pytest.fail(
            "terminal IPython must use plain equations"
        ),
    )

    spectral.spectrum_analysis(
        obj,
        {
            "method": "fft",
            "window": "boxcar",
            "plot": False,
            "print": True,
            "print all": True,
            "pretty print": True,
        },
    )

    output = capsys.readouterr().out
    headings = [
        "Spectrum Analysis Parameters:",
        "Spectrum Analysis Equations:",
        "Spectrum Analysis Summary:",
        "Spectrum Analysis Data:",
    ]
    assert all(heading in output for heading in headings)
    assert [output.index(heading) for heading in headings] == sorted(
        output.index(heading) for heading in headings
    )
    assert "A_k = c_k |X_k| / (N Gw)" in output
    assert "Styler" not in output
    assert "Math(" not in output
    assert "<IPython" not in output


@pytest.mark.parametrize(
    ("method", "symbolic_latex", "definitions_latex"),
    [
        (
            "welch",
            r"S_{xx}(f_k)=\frac{c_k}{K}\sum_{j=1}^{K}"
            r"\frac{\left|\sum_{n=0}^{N-1}w_nx_{j,n}"
            r"e^{-i2\pi kn/M}\right|^2}{f_s\sum_{n=0}^{N-1}w_n^2}",
            r"N=\mathrm{samples/window};\ M=\mathrm{nfft};\ "
            r"K=\mathrm{Welch\ window\ count};\ c_k=1\ \mathrm{at\ DC\ and\ "
            r"Nyquist},\ 2\ \mathrm{for\ interior\ one-sided\ bins};\ "
            r"f_s\ \mathrm{in\ Hz};\ S_{xx}\ \mathrm{in\ signal^2\,Hz^{-1}}",
        ),
        (
            "fft",
            r"X_k=\sum_{n=0}^{N-1}x_nw_n e^{-i2\pi kn/M},\qquad "
            r"A_k=\frac{c_k|X_k|}{N\,G_w}",
            r"N=\mathrm{sample\ count};\ M=\mathrm{nfft};\ "
            r"G_w=N^{-1}\sum_n w_n;\ c_k=1\ \mathrm{at\ DC\ and\ Nyquist},\ "
            r"2\ \mathrm{for\ interior\ one-sided\ bins}",
        ),
    ],
)
def test_spectrum_live_pretty_path_observably_displays_exact_latex(
    blank_echem_factory,
    ecat_module,
    monkeypatch,
    method,
    symbolic_latex,
    definitions_latex,
):
    IPython = pytest.importorskip("IPython")
    from ecat import analysis_batch

    class ZMQInteractiveShell:
        pass

    class CapturedMath:
        def __init__(self, value):
            self.value = value

    displayed = []
    obj = _spectrum_analysis_object(blank_echem_factory, ecat_module)
    monkeypatch.setattr(spectral, "_can_rich_table_display", lambda: True)
    monkeypatch.setattr(IPython, "get_ipython", lambda: ZMQInteractiveShell())
    monkeypatch.setattr(spectral, "_display_table", lambda *args, **kwargs: None)
    monkeypatch.setattr(analysis_batch, "Math", CapturedMath)
    monkeypatch.setattr(
        analysis_batch,
        "display",
        lambda value: displayed.append(value.value),
    )

    spectral.spectrum_analysis(
        obj,
        {
            "method": method,
            "window": "boxcar",
            "points per window": 64,
            "plot": False,
            "print": True,
            "pretty print": True,
        },
    )

    assert displayed == [
        r"\text{Spectrum analysis equations:}",
        symbolic_latex,
        definitions_latex,
    ]


def test_spectrum_sig_figs_only_change_display_copies(
    blank_echem_factory,
    ecat_module,
    monkeypatch,
):
    obj = _spectrum_analysis_object(
        blank_echem_factory,
        ecat_module,
        sample_rate=8.123456789,
        frequency=1.0,
    )
    displayed = {}

    def capture_table(table, options, *, title=None, **kwargs):
        displayed[title] = table.copy()
        return table

    monkeypatch.setattr(spectral, "_display_table", capture_table)

    result = spectral.spectrum_analysis(
        obj,
        {
            "method": "fft",
            "window": "boxcar",
            "plot": False,
            "print": True,
            "print all": True,
            "pretty print": False,
            "sig figs": 3,
        },
    )

    summary_display = displayed["Spectrum Analysis Summary"]
    sample_rate_display = summary_display.loc[
        summary_display["Setting"] == "Sample Rate / Hz", "Value"
    ].iloc[0]
    assert sample_rate_display == "8.12"
    assert result.summary["sample rate / Hz"] == pytest.approx(8.123456789)
    assert isinstance(result.summary["sample rate / Hz"], float)
    assert all(
        np.issubdtype(dtype, np.number) for dtype in result.table.dtypes
    )
    assert displayed["Spectrum Analysis Data"].map(
        lambda value: isinstance(value, str)
    ).all().all()


def test_spectrum_compact_plain_output_summarizes_custom_window(
    blank_echem_factory,
    ecat_module,
    capsys,
):
    obj = _spectrum_analysis_object(blank_echem_factory, ecat_module)

    result = spectral.spectrum_analysis(
        obj,
        {
            "method": "fft",
            "window": np.ones(128),
            "detrend": None,
            "plot": False,
            "print": True,
            "pretty print": False,
        },
    )

    output = capsys.readouterr().out
    assert "custom window (128 points)" in output
    assert "array(" not in output
    assert "[1. 1. 1." not in output
    assert result.diagnostics["parameters"]["window"] == (
        "custom window (128 points)"
    )
    assert result.diagnostics["window"] == "custom window (128 points)"
    assert not any(
        isinstance(value, np.ndarray)
        for value in result.diagnostics["parameters"].values()
    )


def test_spectrum_excel_diagnostics_use_compact_object_descriptors(
    blank_echem_factory,
    ecat_module,
    tmp_path,
):
    sample_count = 300
    sample_rate = 100.0
    local_time = np.arange(sample_count, dtype=float) / sample_rate
    obj = blank_echem_factory(ecat_module.ca)
    obj.data = pd.DataFrame(
        {
            "Time": np.concatenate([local_time, 10.0 + local_time]),
            "Current": np.sin(2 * np.pi * 8 * np.tile(local_time, 2)),
            "Segment": np.repeat([1, 2], sample_count),
        }
    )
    obj.units = {"Time": "s", "Current": "A"}
    result = spectral.spectrum_analysis(
        obj,
        {
            "method": "fft",
            "segments": [1, 2],
            "window": "boxcar",
            "plot": False,
            "print": False,
        },
    )
    output_path = tmp_path / "spectrum-result.xlsx"

    with warnings.catch_warnings(record=True) as emitted:
        warnings.simplefilter("always")
        result.to_excel(output_path)

    assert not any(
        "32,767" in str(item.message) or "too long" in str(item.message).lower()
        for item in emitted
    )
    diagnostics = pd.read_excel(output_path, sheet_name="diagnostics")
    cell_text = diagnostics.fillna("").astype(str)
    assert cell_text.map(len).to_numpy().max() < 32_767
    trace_descriptor = diagnostics.loc[
        diagnostics["Field"] == "traces", "Value"
    ].iloc[0]
    computation_descriptor = diagnostics.loc[
        diagnostics["Field"] == "computations", "Value"
    ].iloc[0]
    assert "_SpectrumTrace(" in trace_descriptor
    assert "samples=300" in trace_descriptor
    assert "array(" not in trace_descriptor
    assert "_SpectrumComputation(" in computation_descriptor
    assert "bins=" in computation_descriptor
    assert "array(" not in computation_descriptor


def test_many_trace_excel_diagnostics_have_bounded_sequence_descriptors(
    tmp_path,
):
    sample_count = 8
    trace_count = 300
    time = np.arange(sample_count, dtype=float)
    intervals = np.diff(time)
    traces = []
    computations = []
    for index in range(trace_count):
        traces.append(
            spectral._SpectrumTrace(
                segment=index + 1,
                label=f"trace {index + 1}",
                time=time,
                signal=np.arange(sample_count, dtype=float) + index,
                signal_column="Current",
                signal_unit="A",
                time_source="recorded",
                source_indices=np.arange(sample_count),
                original_time=time,
                original_signal=np.arange(sample_count, dtype=float) + index,
                original_intervals=intervals,
                resolved_intervals=intervals,
                median_interval=1.0,
                relative_interval_deviation=0.0,
                sample_rate=1.0,
            )
        )
        computations.append(
            spectral._SpectrumComputation(
                segment=index + 1,
                label=f"trace {index + 1}",
                signal_unit="A",
                method="fft",
                frequency=np.arange(5, dtype=float) / 8,
                amplitude=np.ones(5),
                phase=np.zeros(5),
                sample_count=sample_count,
                sample_rate=1.0,
                nyquist=0.5,
                frequency_bin_spacing=0.125,
                nfft=sample_count,
                window="boxcar",
                detrend=None,
                coherent_gain=1.0,
            )
        )

    resolved = _options(method="fft", window="boxcar", detrend=None)
    diagnostics = spectral._spectrum_diagnostics(computations, traces, resolved)
    table = pd.DataFrame(
        {
            "Frequency / Hz": [0.125],
            "Amplitude / A": [1.0],
            "Phase / rad": [0.0],
        }
    )
    table.attrs["units"] = spectral._spectrum_units("A", "fft")
    result = AnalysisResult(table=table, diagnostics=diagnostics)
    output_path = tmp_path / "many-spectrum-traces.xlsx"

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result.to_excel(output_path)

    trace_sequence = result.diagnostics["traces"]
    computation_sequence = result.diagnostics["computations"]
    assert isinstance(trace_sequence, tuple)
    assert isinstance(computation_sequence, tuple)
    assert len(trace_sequence) == len(computation_sequence) == trace_count
    assert trace_sequence[299].signal[0] == pytest.approx(299.0)
    assert trace_sequence[299].signal.flags.writeable is False
    assert computation_sequence[299].amplitude.flags.writeable is False
    with pytest.raises(AttributeError):
        trace_sequence.extra_metadata = "mutable"
    for sequence, descriptor_name in (
        (trace_sequence, "_SpectrumTrace("),
        (computation_sequence, "_SpectrumComputation("),
    ):
        descriptor = repr(sequence)
        assert str(sequence) == descriptor
        assert "count=300" in descriptor
        assert descriptor.count(descriptor_name) == 3
        assert "..." in descriptor
        assert len(descriptor) < 2_000

    exported = pd.read_excel(output_path, sheet_name="diagnostics")
    exported_text = exported.fillna("").astype(str)
    assert exported_text.map(len).to_numpy().max() < 32_767


def test_spectrum_analysis_attaches_plots_after_diagnostics_exist(
    blank_echem_factory,
    ecat_module,
    monkeypatch,
):
    obj = _spectrum_analysis_object(blank_echem_factory, ecat_module)
    original_attach = spectral._attach_spectrum_figures
    diagnostic_snapshots = []

    def checked_attach(result, computations, options):
        diagnostic_snapshots.append(result.diagnostics.copy())
        assert result.diagnostics["traces"]
        assert result.diagnostics["computations"]
        return original_attach(result, computations, options)

    monkeypatch.setattr(spectral, "_attach_spectrum_figures", checked_attach)

    result = spectral.spectrum_analysis(
        obj,
        {
            "method": "fft",
            "window": "boxcar",
            "plot": True,
            "plot all": True,
            "print": False,
        },
    )

    assert len(diagnostic_snapshots) == 1
    assert len(result.figures) == 2
    assert result.figure is result.figures[0]
    assert result.axes is result.figure.axes[0]


def test_report_exception_does_not_repeat_accumulated_warning(
    blank_echem_factory,
    ecat_module,
    monkeypatch,
):
    obj = _spectrum_analysis_object(blank_echem_factory, ecat_module)
    message = "one report warning"
    original_resolve = spectral._resolve_spectrum_traces

    def resolve_with_duplicate(analysis_obj, resolved):
        traces, _ = original_resolve(analysis_obj, resolved)
        return traces, [message, message]

    monkeypatch.setattr(spectral, "_resolve_spectrum_traces", resolve_with_duplicate)
    monkeypatch.setattr(
        spectral,
        "_print_spectrum_result",
        lambda result, options: (_ for _ in ()).throw(RuntimeError("report failed")),
    )

    with warnings.catch_warnings(record=True) as emitted:
        warnings.simplefilter("always")
        with pytest.raises(RuntimeError, match="report failed"):
            spectral.spectrum_analysis(
                obj,
                {
                    "method": "fft",
                    "window": "boxcar",
                    "plot": False,
                    "print": True,
                },
            )

    assert [str(item.message) for item in emitted] == [message]


def test_task_four_public_module_export_contract():
    assert callable(spectral.spectrum_analysis)
    assert "spectrum_analysis" in spectral.__all__
    assert len(spectral.__all__) == len(set(spectral.__all__))


@pytest.mark.parametrize("class_name", ["echem", "cv", "ca", "cp", "dpv"])
def test_public_spectrum_is_the_exact_inherited_echem_method(ecat_module, class_name):
    cls = getattr(ecat_module, class_name)

    assert cls.spectrum is ecat_module.echem.spectrum
    if cls is not ecat_module.echem:
        assert "spectrum" not in cls.__dict__


@pytest.mark.parametrize(
    "class_name",
    [
        "AnalysisResult",
        "CVAnalysisResult",
        "ChronoAnalysisResult",
        "ScatterFitResult",
        "ParseResult",
    ],
)
def test_public_spectrum_methods_do_not_leak_to_non_echem_results(
    ecat_module,
    class_name,
):
    cls = getattr(ecat_module, class_name)

    assert not hasattr(cls, "spectrum")
    assert not hasattr(cls, "fft")


@pytest.mark.parametrize("class_name", ["echem", "cv", "ca", "cp", "dpv"])
def test_public_spectrum_instances_do_not_expose_cv_fft(
    blank_echem_factory,
    ecat_module,
    class_name,
):
    obj = blank_echem_factory(getattr(ecat_module, class_name))

    assert not hasattr(obj, "fft")


def test_retired_spectrum_surface_has_no_normalized_frequency_fft_fallback(
    blank_echem_factory,
    ecat_module,
):
    obj, _, _ = _public_spectrum_object(blank_echem_factory, ecat_module, "cv")
    obj.scan_rate = None

    assert not hasattr(ecat_module.cv, "fft")
    with pytest.raises(AttributeError, match="fft"):
        getattr(obj, "fft")


def test_importing_ecat_keeps_spectral_module_lazy(repo_root, tmp_path):
    env = os.environ.copy()
    env.update(
        {
            "MPLBACKEND": "Agg",
            "MPLCONFIGDIR": str(tmp_path / "mpl"),
            "PYTHONHASHSEED": "0",
            "PYTHONPATH": str(repo_root / "src"),
        }
    )
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; import ecat; "
                "assert 'ecat.spectral' not in sys.modules, sorted(sys.modules)"
            ),
        ],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr


def test_public_spectrum_delegates_with_original_object_and_options(
    blank_echem_factory,
    ecat_module,
    monkeypatch,
):
    obj = blank_echem_factory(ecat_module.echem)
    options = {"method": "fft", "plot": False}
    sentinel = object()
    observed = {}

    def fake_spectrum_analysis(received_obj, received_options):
        observed["obj"] = received_obj
        observed["options"] = received_options
        return sentinel

    monkeypatch.setattr(spectral, "spectrum_analysis", fake_spectrum_analysis)

    result = obj.spectrum(options)

    assert observed["obj"] is obj
    assert observed["options"] is options
    assert result is sentinel


@pytest.mark.parametrize(
    ("class_name", "method"),
    [
        ("echem", None),
        ("ca", None),
        ("cp", "fft"),
        ("cv", None),
        ("dpv", None),
    ],
)
def test_public_spectrum_calls_return_physical_analysis_results(
    blank_echem_factory,
    ecat_module,
    class_name,
    method,
):
    obj, expected_time_source, expected_sample_rate = _public_spectrum_object(
        blank_echem_factory,
        ecat_module,
        class_name,
    )
    options = {"plot": False, "print": False}
    if method is not None:
        options["method"] = method

    with warnings.catch_warnings(record=True) as emitted:
        warnings.simplefilter("always")
        result = obj.spectrum(options)

    warning_messages = [str(item.message) for item in emitted]
    if method == "fft":
        assert warning_messages == []
    else:
        assert sum("fewer than two complete" in message for message in warning_messages) == 1
        expected_reconstruction_warnings = 1 if class_name in {"cv", "dpv"} else 0
        assert sum("reconstructed" in message for message in warning_messages) == (
            expected_reconstruction_warnings
        )
        assert len(warning_messages) == 1 + expected_reconstruction_warnings

    assert isinstance(result, AnalysisResult)
    assert result.summary["method"] == (method or "welch")
    assert result.summary["time source"] == expected_time_source
    assert result.summary["time unit"] == "s"
    assert result.summary["original sample count"] == 12
    assert result.summary["sample count"] == 12
    assert result.summary["sample interval / s"] == pytest.approx(
        1.0 / expected_sample_rate
    )
    assert result.summary["sample rate / Hz"] == pytest.approx(expected_sample_rate)
    assert result.diagnostics["traces"][0].time == pytest.approx(
        np.arange(12, dtype=float) / expected_sample_rate
    )
    assert result.figure is None
    assert result.axes is None
    assert result.figures == []


@pytest.mark.parametrize("class_name", ["ca", "cp"])
def test_public_spectrum_requires_recorded_time_for_chrono_techniques(
    blank_echem_factory,
    ecat_module,
    class_name,
):
    obj = blank_echem_factory(getattr(ecat_module, class_name))
    obj.data = pd.DataFrame(
        {
            "Current": np.linspace(1.0, 2.0, 12),
            "Potential": np.linspace(-0.4, 0.2, 12),
        }
    )
    obj.units = {"Current": "A", "Potential": "V"}

    with pytest.raises(ValueError, match="recorded Time"):
        obj.spectrum({"plot": False, "print": False})


@pytest.mark.parametrize(
    "time",
    [
        [0, 1, 2, 3, 4, 4, 6, 7, 8, 9, 10, 11],
        [0, 1, 2, 3, 4, 3.5, 6, 7, 8, 9, 10, 11],
    ],
    ids=["duplicate", "non-increasing"],
)
def test_public_spectrum_rejects_duplicate_or_non_increasing_time(
    blank_echem_factory,
    ecat_module,
    time,
):
    obj = _timed_object(blank_echem_factory, ecat_module.ca, time=time)

    with pytest.raises(ValueError, match="strictly increasing"):
        obj.spectrum({"plot": False, "print": False})


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("too few finite pairs", "at least 8 finite"),
        ("invalid column", "Available columns"),
        ("invalid segment", r"Requested segment.*2.*available segments.*1"),
        ("frequency above Nyquist", "maximum frequency.*Nyquist"),
    ],
)
def test_public_spectrum_rejects_invalid_analysis_boundaries(
    blank_echem_factory,
    ecat_module,
    case,
    message,
):
    options = {"plot": False, "print": False}
    if case == "too few finite pairs":
        obj = _timed_object(
            blank_echem_factory,
            ecat_module.ca,
            time=np.arange(8, dtype=float),
        )
        obj.data.loc[0, "Current"] = np.nan
    elif case == "invalid column":
        obj = _timed_object(blank_echem_factory, ecat_module.ca)
        options["column"] = "missing"
    elif case == "invalid segment":
        obj = _timed_object(blank_echem_factory, ecat_module.ca)
        obj.data["Segment"] = np.ones(len(obj.data), dtype=int)
        options["segments"] = 2
    else:
        obj = _timed_object(blank_echem_factory, ecat_module.ca)
        options["frequency range"] = [1.0, 6.0]

    with pytest.raises(ValueError, match=message):
        obj.spectrum(options)


@pytest.mark.parametrize(
    ("tolerance_case", "accepted"),
    [
        pytest.param("exact", True, id="exact-tolerance"),
        pytest.param("just-above", False, id="just-above-tolerance"),
    ],
)
def test_public_spectrum_sampling_tolerance_boundary_is_inclusive(
    blank_echem_factory,
    ecat_module,
    tolerance_case,
    accepted,
):
    intervals = np.ones(11, dtype=float)
    intervals[4:6] = [1.01, 0.99]
    time = np.concatenate(([0.0], np.cumsum(intervals)))
    deviation = float(np.max(np.abs(np.diff(time) - 1.0)))
    tolerance = (
        deviation
        if tolerance_case == "exact"
        else deviation - 1e-12
    )
    obj = _timed_object(blank_echem_factory, ecat_module.ca, time=time)
    options = {
        "method": "fft",
        "window": "boxcar",
        "detrend": None,
        "sampling tolerance": tolerance,
        "plot": False,
        "print": False,
    }

    if not accepted:
        with pytest.raises(ValueError, match=r"resample=True"):
            obj.spectrum(options)
        return

    result = obj.spectrum(options)

    assert result.summary["relative interval deviation"] == pytest.approx(deviation)
    assert result.summary["resampled"] is False
    assert result.diagnostics["traces"][0].time == pytest.approx(time)


def test_public_spectrum_substantial_irregularity_requires_resampling_by_default(
    blank_echem_factory,
    ecat_module,
):
    time = np.arange(12, dtype=float) * 0.1
    time[5] += 0.02
    obj = _timed_object(blank_echem_factory, ecat_module.ca, time=time)

    with pytest.raises(
        ValueError,
        match=r"relative interval deviation.*tolerance 0\.01.*resample=True",
    ):
        obj.spectrum(
            {
                "method": "fft",
                "window": "boxcar",
                "plot": False,
                "print": False,
            }
        )


def test_public_spectrum_resample_true_is_noop_exactly_at_sampling_tolerance(
    blank_echem_factory,
    ecat_module,
):
    intervals = np.ones(11, dtype=float)
    intervals[4:6] = [1.01, 0.99]
    time = np.concatenate(([0.0], np.cumsum(intervals)))
    deviation = float(np.max(np.abs(np.diff(time) - 1.0)))
    obj = _timed_object(blank_echem_factory, ecat_module.ca, time=time)
    original_signal = obj.data["Current"].to_numpy(copy=True)

    result = obj.spectrum(
        {
            "method": "fft",
            "window": "boxcar",
            "detrend": None,
            "resample": True,
            "sampling tolerance": deviation,
            "plot": False,
            "print": False,
        }
    )

    trace = result.diagnostics["traces"][0]
    assert result.summary["resampled"] is False
    assert trace.resampled is False
    assert trace.original_time == pytest.approx(time)
    assert trace.time == pytest.approx(time)
    assert trace.original_signal == pytest.approx(original_signal)
    assert trace.signal == pytest.approx(original_signal)
    assert trace.original_intervals == pytest.approx(np.diff(time))
    assert trace.resolved_intervals == pytest.approx(np.diff(time))


@pytest.mark.parametrize("interval", [1.0, 0.1, 0.01, 1e-6])
def test_public_spectrum_accepts_exact_one_percent_irregularity_at_all_scales(
    blank_echem_factory,
    ecat_module,
    interval,
):
    intervals = np.full(11, interval, dtype=float)
    intervals[4:6] = [interval * 1.01, interval * 0.99]
    time = np.concatenate(([0.0], np.cumsum(intervals)))
    obj = _timed_object(blank_echem_factory, ecat_module.ca, time=time)

    result = obj.spectrum(
        {
            "method": "fft",
            "window": "boxcar",
            "detrend": None,
            "plot": False,
            "print": False,
        }
    )

    assert result.summary["relative interval deviation"] == pytest.approx(0.01)
    assert result.summary["sample interval / s"] == pytest.approx(interval)
    assert result.summary["sample rate / Hz"] == pytest.approx(1.0 / interval)


@pytest.mark.parametrize("interval", [1.0, 0.1, 0.01, 1e-6])
def test_public_spectrum_rejects_just_over_one_percent_at_all_scales(
    blank_echem_factory,
    ecat_module,
    interval,
):
    deviation = 0.010001
    intervals = np.full(11, interval, dtype=float)
    intervals[4:6] = [
        interval * (1.0 + deviation),
        interval * (1.0 - deviation),
    ]
    time = np.concatenate(([0.0], np.cumsum(intervals)))
    obj = _timed_object(blank_echem_factory, ecat_module.ca, time=time)

    with pytest.raises(ValueError, match=r"interval deviation.*resample=True"):
        obj.spectrum(
            {
                "method": "fft",
                "window": "boxcar",
                "plot": False,
                "print": False,
            }
        )


@pytest.mark.parametrize("irregular", [False, True], ids=["uniform", "irregular"])
def test_public_spectrum_accepts_maximum_sampling_tolerance_without_overflow(
    blank_echem_factory,
    ecat_module,
    irregular,
):
    intervals = np.ones(11, dtype=float)
    if irregular:
        intervals[4:6] = [1.5, 0.5]
    time = np.concatenate(([0.0], np.cumsum(intervals)))
    obj = _timed_object(blank_echem_factory, ecat_module.ca, time=time)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = obj.spectrum(
            {
                "method": "fft",
                "window": "boxcar",
                "detrend": None,
                "sampling tolerance": np.finfo(float).max,
                "plot": False,
                "print": False,
            }
        )

    assert result.summary["resampled"] is False
    assert np.isfinite(result.summary["relative interval deviation"])


def test_public_spectrum_converts_time_subtraction_overflow_to_sampling_error(
    blank_echem_factory,
    ecat_module,
):
    maximum = np.finfo(float).max
    time = np.array(
        [
            -maximum,
            -1.5e308,
            -1.0e308,
            -0.9e308,
            0.9e308,
            1.0e308,
            1.5e308,
            maximum,
        ]
    )
    obj = _timed_object(blank_echem_factory, ecat_module.ca, time=time)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(
            ValueError,
            match=r"Sampling intervals.*finite and strictly positive.*Time",
        ):
            obj.spectrum(
                {
                    "method": "fft",
                    "window": "boxcar",
                    "plot": False,
                    "print": False,
                }
            )


def test_public_spectrum_rejects_infinite_sample_rate_without_runtime_warning(
    blank_echem_factory,
    ecat_module,
):
    interval = np.nextafter(0.0, 1.0)
    time = np.arange(8, dtype=float) * interval
    obj = _timed_object(blank_echem_factory, ecat_module.ca, time=time)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(
            ValueError,
            match=r"sample rate.*finite and positive.*Time",
        ):
            obj.spectrum(
                {
                    "method": "fft",
                    "window": "boxcar",
                    "plot": False,
                    "print": False,
                }
            )


def test_public_spectrum_resampling_records_immutable_original_and_resolved_arrays(
    blank_echem_factory,
    ecat_module,
):
    time = np.arange(12, dtype=float) * 0.1
    time[5] += 0.02
    obj = _timed_object(blank_echem_factory, ecat_module.ca, time=time)

    result = obj.spectrum(
        {
            "method": "fft",
            "window": "boxcar",
            "detrend": None,
            "resample": True,
            "plot": False,
            "print": False,
        }
    )

    trace = result.diagnostics["traces"][0]
    assert result.summary["resampled"] is True
    assert result.diagnostics["parameters"]["resampled"] is True
    assert trace.original_time == pytest.approx(time)
    assert trace.original_intervals == pytest.approx(np.diff(time))
    assert trace.time == pytest.approx(np.arange(12, dtype=float) * 0.1)
    assert trace.resolved_intervals == pytest.approx(np.full(11, 0.1))
    for field_name in (
        "time",
        "signal",
        "original_time",
        "original_signal",
        "original_intervals",
        "resolved_intervals",
    ):
        assert getattr(trace, field_name).flags.writeable is False


def test_public_spectrum_accepts_eight_pairs_and_warns_once_for_weak_welch_averaging(
    blank_echem_factory,
    ecat_module,
):
    obj = _timed_object(
        blank_echem_factory,
        ecat_module.ca,
        time=np.arange(8, dtype=float),
    )

    with pytest.warns(UserWarning, match="fewer than two complete") as emitted:
        result = obj.spectrum(
            {
                "window": "boxcar",
                "points per window": 8,
                "plot": False,
                "print": False,
            }
        )

    weak_averaging = [
        message
        for message in result.warnings
        if "fewer than two complete" in message
    ]
    assert result.summary["sample count"] == 8
    assert result.diagnostics["window count"] == 1
    assert len(emitted) == 1
    assert len(weak_averaging) == 1


def test_public_spectrum_rejects_too_small_explicit_welch_window(
    blank_echem_factory,
    ecat_module,
):
    obj = _timed_object(blank_echem_factory, ecat_module.ca)

    with pytest.raises(
        ecat_module.OptionError,
        match=r"points per window.*integer >= 2",
    ):
        obj.spectrum(
            {
                "points per window": 1,
                "plot": False,
                "print": False,
            }
        )


def test_public_spectrum_records_oversized_welch_window_reduction_everywhere(
    blank_echem_factory,
    ecat_module,
):
    obj = _spectrum_analysis_object(
        blank_echem_factory,
        ecat_module,
        sample_count=100,
        sample_rate=100.0,
    )
    options = {
        "window": "boxcar",
        "points per window": 200,
        "plot": False,
        "print": False,
    }

    with pytest.warns(UserWarning) as emitted:
        result = obj.spectrum(options)

    emitted_messages = [str(item.message) for item in emitted]
    reduction_messages = [
        message for message in emitted_messages if "reduc" in message.casefold()
    ]
    parameter_table = spectral._spectrum_parameter_display_table(
        result,
        SpectrumOptions.from_options(options).to_options_dict(),
    ).set_index("Setting")["Value"]
    assert len(reduction_messages) == 1
    assert result.warnings == emitted_messages
    assert result.summary["points per window"] == 100
    assert result.diagnostics["nperseg"] == 100
    assert result.diagnostics["parameters"]["points per window"] == 100
    assert parameter_table["Points per Window"] == "200 requested; 100 used"


@pytest.mark.parametrize(
    "window",
    [
        pytest.param("not-a-scipy-window", id="unknown-name"),
        pytest.param(("kaiser",), id="incomplete-specification"),
    ],
)
def test_public_spectrum_translates_invalid_scipy_window_errors(
    blank_echem_factory,
    ecat_module,
    window,
):
    obj = _spectrum_analysis_object(blank_echem_factory, ecat_module)

    with pytest.raises(ValueError) as exc_info:
        obj.spectrum(
            {
                "method": "fft",
                "window": window,
                "plot": False,
                "print": False,
            }
        )

    message = str(exc_info.value)
    assert "window" in message
    assert repr(window) in message
    assert exc_info.value.__cause__ is None


@pytest.mark.parametrize(
    "window",
    [
        pytest.param(("kaiser", 1e308), id="overflowing-kaiser"),
        pytest.param(("gaussian", 0.0), id="zero-width-gaussian"),
    ],
)
def test_public_spectrum_translates_warning_leaking_scipy_windows(
    blank_echem_factory,
    ecat_module,
    window,
):
    obj = _spectrum_analysis_object(blank_echem_factory, ecat_module)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with np.errstate(all="raise"):
            with pytest.raises(ValueError) as exc_info:
                obj.spectrum(
                    {
                        "method": "fft",
                        "window": window,
                        "plot": False,
                        "print": False,
                    }
                )

    message = str(exc_info.value)
    assert "window" in message
    assert repr(window) in message
    assert exc_info.value.__cause__ is None


@pytest.mark.parametrize("method", ["welch", "fft"])
@pytest.mark.parametrize(
    "window",
    [
        pytest.param(np.full(64, 1.0 + 1.0j), id="array-nonzero-imaginary"),
        pytest.param(
            pd.Series(np.ones(64, dtype=complex)),
            id="series-zero-imaginary",
        ),
    ],
)
def test_public_spectrum_rejects_complex_custom_windows_without_warning(
    blank_echem_factory,
    ecat_module,
    method,
    window,
):
    obj = _spectrum_analysis_object(
        blank_echem_factory,
        ecat_module,
        sample_count=64,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(ValueError) as exc_info:
            obj.spectrum(
                {
                    "method": method,
                    "window": window,
                    "points per window": 32,
                    "plot": False,
                    "print": False,
                }
            )

    message = str(exc_info.value)
    assert "window" in message
    assert "real finite coefficients" in message


def test_public_spectrum_rejects_complex_scipy_window_output(
    blank_echem_factory,
    ecat_module,
    monkeypatch,
):
    obj = _spectrum_analysis_object(blank_echem_factory, ecat_module)
    monkeypatch.setattr(
        spectral.signal,
        "get_window",
        lambda window, length, fftbins: np.ones(length, dtype=complex),
    )

    with pytest.raises(ValueError) as exc_info:
        obj.spectrum(
            {
                "method": "welch",
                "window": "boxcar",
                "points per window": 32,
                "plot": False,
                "print": False,
            }
        )

    message = str(exc_info.value)
    assert "window" in message
    assert "real finite coefficients" in message


@pytest.mark.parametrize(
    ("method", "signal_scale", "window"),
    [
        pytest.param("welch", 2e307, "boxcar", id="welch-huge-signal"),
        pytest.param("fft", 2e307, "boxcar", id="fft-huge-signal"),
    ],
)
def test_public_spectrum_rejects_huge_finite_backend_arithmetic(
    blank_echem_factory,
    ecat_module,
    method,
    signal_scale,
    window,
):
    obj = _spectrum_analysis_object(
        blank_echem_factory,
        ecat_module,
        sample_count=64,
    )
    phase = np.arange(64, dtype=float) * (2 * np.pi / 8)
    obj.data["Current"] = signal_scale * np.cos(phase)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with np.errstate(all="raise"):
            with pytest.raises(ValueError) as exc_info:
                obj.spectrum(
                    {
                        "method": method,
                        "window": window,
                        "points per window": 32,
                        "plot": False,
                        "print": False,
                    }
                )

    message = str(exc_info.value)
    assert method.casefold() in message.casefold()
    assert "Unnamed EChem Object" in message
    assert ("boxcar" if isinstance(window, str) else "custom window") in message
    assert "nonfinite numerical output" in message
    assert exc_info.value.__cause__ is None


def test_public_fft_huge_finite_window_never_returns_nonfinite_output(
    blank_echem_factory,
    ecat_module,
):
    obj = _spectrum_analysis_object(
        blank_echem_factory,
        ecat_module,
        sample_count=64,
    )
    window = np.concatenate([np.full(8, 2e307), np.zeros(56)])

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = obj.spectrum(
            {
                "method": "fft",
                "window": window,
                "detrend": None,
                "plot": False,
                "print": False,
            }
        )

    assert np.all(np.isfinite(result.table["Frequency / Hz"]))
    assert np.all(np.isfinite(result.table["Amplitude / A"]))
    assert np.all(np.isfinite(result.table["Phase / rad"]))


@pytest.mark.parametrize(
    "case",
    [
        "welch-frequency",
        "welch-psd",
        "fft-transform-real",
        "fft-transform-imaginary",
        "fft-amplitude",
        "fft-phase",
        "fft-frequency",
    ],
)
def test_public_spectrum_rejects_monkeypatched_nonfinite_backend_outputs(
    blank_echem_factory,
    ecat_module,
    monkeypatch,
    case,
):
    obj = _spectrum_analysis_object(
        blank_echem_factory,
        ecat_module,
        sample_count=64,
    )
    method = "welch" if case.startswith("welch") else "fft"
    output_length = 17 if method == "welch" else 33
    window = "boxcar"

    if method == "welch":
        frequency = np.arange(output_length, dtype=float)
        psd = np.ones(output_length)
        if case == "welch-frequency":
            frequency[-1] = np.nan
        else:
            psd[-1] = np.inf
        monkeypatch.setattr(
            spectral.signal,
            "welch",
            lambda *args, **kwargs: (frequency, psd),
        )
    elif case == "fft-frequency":
        frequency = np.arange(output_length, dtype=float)
        frequency[-1] = np.inf
        monkeypatch.setattr(
            spectral.np.fft,
            "rfftfreq",
            lambda *args, **kwargs: frequency,
        )
    elif case == "fft-amplitude":
        window = np.full(64, np.nextafter(0.0, 1.0))
        monkeypatch.setattr(
            spectral.np.fft,
            "rfft",
            lambda *args, **kwargs: np.ones(output_length, dtype=complex),
        )
    elif case == "fft-phase":
        monkeypatch.setattr(
            spectral.np,
            "angle",
            lambda values: np.full(len(values), np.nan),
        )
    else:
        transformed = np.ones(output_length, dtype=complex)
        if case == "fft-transform-real":
            transformed[-1] = complex(np.inf, 0.0)
        else:
            transformed[-1] = complex(1.0, np.nan)
        monkeypatch.setattr(
            spectral.np.fft,
            "rfft",
            lambda *args, **kwargs: transformed,
        )

    with pytest.raises(ValueError) as exc_info:
        obj.spectrum(
            {
                "method": method,
                "window": window,
                "points per window": 32,
                "detrend": None,
                "plot": False,
                "print": False,
            }
        )

    message = str(exc_info.value)
    assert method.casefold() in message.casefold()
    assert "Unnamed EChem Object" in message
    assert ("boxcar" if isinstance(window, str) else "custom window") in message
    assert "nonfinite numerical output" in message


@pytest.mark.parametrize("method", ["welch", "fft"])
def test_public_spectrum_translates_backend_runtime_warnings_only(
    blank_echem_factory,
    ecat_module,
    monkeypatch,
    method,
):
    obj = _spectrum_analysis_object(blank_echem_factory, ecat_module)

    def warning_backend(*args, **kwargs):
        warnings.warn("backend overflow", RuntimeWarning)

    backend = "welch" if method == "welch" else "rfft"
    target = spectral.signal if method == "welch" else spectral.np.fft
    monkeypatch.setattr(target, backend, warning_backend)

    with pytest.raises(ValueError) as exc_info:
        obj.spectrum(
            {
                "method": method,
                "window": "boxcar",
                "points per window": 32,
                "detrend": None,
                "plot": False,
                "print": False,
            }
        )

    message = str(exc_info.value)
    assert method.casefold() in message.casefold()
    assert "Unnamed EChem Object" in message
    assert "boxcar" in message
    assert "nonfinite numerical output" in message
    assert exc_info.value.__cause__ is None


@pytest.mark.parametrize("method", ["welch", "fft"])
def test_public_spectrum_translates_backend_floating_point_errors_only(
    blank_echem_factory,
    ecat_module,
    monkeypatch,
    method,
):
    obj = _spectrum_analysis_object(blank_echem_factory, ecat_module)

    def failing_backend(*args, **kwargs):
        raise FloatingPointError("backend floating-point failure")

    backend = "welch" if method == "welch" else "rfft"
    target = spectral.signal if method == "welch" else spectral.np.fft
    monkeypatch.setattr(target, backend, failing_backend)

    with np.errstate(all="raise"):
        with pytest.raises(ValueError) as exc_info:
            obj.spectrum(
                {
                    "method": method,
                    "window": "boxcar",
                    "points per window": 32,
                    "detrend": None,
                    "plot": False,
                    "print": False,
                }
            )

    message = str(exc_info.value)
    assert method.casefold() in message.casefold()
    assert "Unnamed EChem Object" in message
    assert "boxcar" in message
    assert "nonfinite numerical output" in message
    assert exc_info.value.__cause__ is None


@pytest.mark.parametrize("method", ["welch", "fft"])
@pytest.mark.parametrize(
    ("exception_type", "exception_message"),
    [
        pytest.param(UserWarning, "backend advisory", id="user-warning"),
        pytest.param(RuntimeError, "backend bug", id="programmer-exception"),
    ],
)
def test_public_spectrum_backend_exception_translation_is_narrow(
    blank_echem_factory,
    ecat_module,
    monkeypatch,
    method,
    exception_type,
    exception_message,
):
    obj = _spectrum_analysis_object(blank_echem_factory, ecat_module)

    def failing_backend(*args, **kwargs):
        raise exception_type(exception_message)

    backend = "welch" if method == "welch" else "rfft"
    target = spectral.signal if method == "welch" else spectral.np.fft
    monkeypatch.setattr(target, backend, failing_backend)

    with pytest.raises(exception_type, match=exception_message):
        obj.spectrum(
            {
                "method": method,
                "window": "boxcar",
                "points per window": 32,
                "detrend": None,
                "plot": False,
                "print": False,
            }
        )


def test_public_fft_rejects_huge_finite_custom_window_without_warning(
    blank_echem_factory,
    ecat_module,
):
    obj = _spectrum_analysis_object(
        blank_echem_factory,
        ecat_module,
        sample_count=64,
    )
    window = np.full(64, np.finfo(float).max)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(ValueError) as exc_info:
            obj.spectrum(
                {
                    "method": "fft",
                    "window": window,
                    "plot": False,
                    "print": False,
                }
            )

    message = str(exc_info.value)
    assert "FFT window" in message
    assert "coherent gain" in message


@pytest.mark.parametrize(
    ("method", "window", "message"),
    [
        pytest.param(
            "fft",
            np.ones(63),
            r"window.*64 finite values.*received 63",
            id="wrong-length",
        ),
        pytest.param(
            "fft",
            np.concatenate([np.ones(63), [np.nan]]),
            r"window.*64 finite values",
            id="nonfinite",
        ),
        pytest.param(
            "welch",
            np.zeros(64),
            r"Welch window.*finite positive power",
            id="zero-power",
        ),
        pytest.param(
            "welch",
            np.full(64, np.sqrt(np.finfo(float).max)),
            r"Welch window.*finite positive power",
            id="nonfinite-power",
        ),
        pytest.param(
            "fft",
            np.tile([1.0, -1.0], 32),
            r"coherent gain",
            id="zero-coherent-gain",
        ),
        pytest.param(
            "fft",
            -np.ones(64),
            r"coherent gain.*positive",
            id="negative-coherent-gain",
        ),
    ],
)
def test_public_spectrum_rejects_invalid_custom_windows(
    blank_echem_factory,
    ecat_module,
    method,
    window,
    message,
):
    obj = _spectrum_analysis_object(
        blank_echem_factory,
        ecat_module,
        sample_count=64,
    )

    with pytest.raises(ValueError, match=message):
        obj.spectrum(
            {
                "method": method,
                "window": window,
                "points per window": 64,
                "plot": False,
                "print": False,
            }
        )


@pytest.mark.parametrize("method", ["welch", "fft"])
def test_public_spectrum_supports_nfft_zero_padding(
    blank_echem_factory,
    ecat_module,
    method,
):
    obj = _spectrum_analysis_object(
        blank_echem_factory,
        ecat_module,
        sample_count=64,
        sample_rate=64.0,
    )

    result = obj.spectrum(
        {
            "method": method,
            "window": "boxcar",
            "points per window": 32,
            "nfft": 128,
            "plot": False,
            "print": False,
        }
    )

    assert result.diagnostics["nfft"] == 128
    assert result.summary["frequency bin spacing / Hz"] == pytest.approx(0.5)


def test_public_welch_rejects_nfft_shorter_than_points_per_window(
    blank_echem_factory,
    ecat_module,
):
    obj = _spectrum_analysis_object(blank_echem_factory, ecat_module)

    with pytest.raises(
        ValueError,
        match=r"Welch nfft.*nperseg=8.*received 4",
    ):
        obj.spectrum(
            {
                "window": "boxcar",
                "points per window": 8,
                "nfft": 4,
                "plot": False,
                "print": False,
            }
        )


@pytest.mark.parametrize(
    ("options", "message"),
    [
        pytest.param(
            {"method": "fft", "nfft": 8},
            r"nfft.*selected signal length 12",
            id="undersized-nfft",
        ),
        pytest.param(
            {"method": "fft", "frequency range": [0.0, 0.1]},
            r"no frequency bins",
            id="empty-frequency-selection",
        ),
    ],
)
def test_public_spectrum_translates_computation_boundaries(
    blank_echem_factory,
    ecat_module,
    options,
    message,
):
    obj = _timed_object(blank_echem_factory, ecat_module.ca)
    options.update(
        {
            "window": "boxcar",
            "plot": False,
            "print": False,
        }
    )

    with pytest.raises(ValueError, match=message):
        obj.spectrum(options)


@pytest.mark.parametrize(
    "overlap",
    [
        pytest.param(1.0, id="no-progress"),
        pytest.param(np.nan, id="nonfinite"),
    ],
)
def test_public_spectrum_rejects_overlap_without_valid_window_progress(
    blank_echem_factory,
    ecat_module,
    overlap,
):
    obj = _spectrum_analysis_object(blank_echem_factory, ecat_module)

    with pytest.raises(ecat_module.OptionError, match="overlap"):
        obj.spectrum(
            {
                "overlap": overlap,
                "plot": False,
                "print": False,
            }
        )


def test_public_spectrum_near_complete_overlap_still_advances_windows(
    blank_echem_factory,
    ecat_module,
):
    obj = _spectrum_analysis_object(
        blank_echem_factory,
        ecat_module,
        sample_count=64,
    )

    result = obj.spectrum(
        {
            "window": "boxcar",
            "points per window": 8,
            "overlap": np.nextafter(1.0, 0.0),
            "plot": False,
            "print": False,
        }
    )

    assert result.diagnostics["noverlap"] == 7
    assert result.diagnostics["window count"] == 57


def test_public_spectrum_all_zero_signal_has_no_peak_and_uses_linear_plot_fallback(
    blank_echem_factory,
    ecat_module,
):
    obj = _spectrum_analysis_object(blank_echem_factory, ecat_module)
    obj.data["Current"] = 0.0

    result = obj.spectrum(
        {
            "window": "boxcar",
            "points per window": 64,
            "plot": True,
            "print": False,
        }
    )

    assert np.isnan(result.summary["peak frequency / Hz"])
    assert np.all(result.diagnostics["computations"][0].psd == 0)
    assert result.axes.get_yscale() == "linear"


def test_spectrum_excel_export_closes_writer_under_warnings_as_errors(
    blank_echem_factory,
    ecat_module,
    tmp_path,
):
    obj = _spectrum_analysis_object(blank_echem_factory, ecat_module)
    result = obj.spectrum(
        {
            "method": "fft",
            "window": "boxcar",
            "plot": False,
            "print": False,
        }
    )
    output_path = tmp_path / "warning-strict-spectrum.xlsx"
    renamed_path = tmp_path / "reopened-spectrum.xlsx"
    figure_numbers_before = set(plt.get_fignums())

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result.to_excel(output_path)
        with pd.ExcelFile(output_path) as workbook:
            assert workbook.sheet_names
        output_path.rename(renamed_path)
        renamed_path.unlink()

    assert not output_path.exists()
    assert not renamed_path.exists()
    assert set(plt.get_fignums()) == figure_numbers_before


@pytest.mark.parametrize(
    ("class_name", "attribute", "value"),
    [
        ("cv", "scan_rate", None),
        ("cv", "scan_rate", 0),
        ("cv", "scan_rate", np.nan),
        ("cv", "scan_rate", np.inf),
        ("dpv", "pulse_period", None),
        ("dpv", "pulse_period", 0),
        ("dpv", "pulse_period", np.nan),
        ("dpv", "pulse_period", np.inf),
    ],
    ids=[
        "cv-missing-scan-rate",
        "cv-zero-scan-rate",
        "cv-nan-scan-rate",
        "cv-infinite-scan-rate",
        "dpv-missing-pulse-period",
        "dpv-zero-pulse-period",
        "dpv-nan-pulse-period",
        "dpv-infinite-pulse-period",
    ],
)
def test_public_spectrum_requires_valid_reconstruction_metadata(
    blank_echem_factory,
    ecat_module,
    class_name,
    attribute,
    value,
):
    obj, _, _ = _public_spectrum_object(
        blank_echem_factory,
        ecat_module,
        class_name,
    )
    if value is None:
        delattr(obj, attribute)
    else:
        setattr(obj, attribute, value)

    with pytest.raises(ValueError, match=attribute):
        obj.spectrum({"plot": False, "print": False})
