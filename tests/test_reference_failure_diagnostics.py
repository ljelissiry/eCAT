"""Failure guidance uses selector evidence, not an assumed physical diagnosis.

All waveforms and candidate records below are synthetic test inputs.
"""
import inspect

import numpy as np
import pytest

from ecat import reference
from ecat.options import ImportOptions


def _candidate_cv(cv_factory, monkeypatch, *, kinds=None, potentials=None):
    # A partial initial sweep, followed by six complete alternating sweeps.
    ramps = [np.linspace(0.1, -0.2, 21)]
    for i in range(6):
        start, stop = (-0.2, 1.0) if i % 2 == 0 else (1.0, -0.2)
        ramps.append(np.linspace(start, stop, 121)[1:])
    x = np.concatenate(ramps)
    cv = cv_factory(potential=x, current=np.zeros_like(x))
    potentials = potentials or [0.73, 0.45, 0.71, 0.44, 0.72, 0.43]
    kinds = kinds or ["maximum", "minimum"] * 3

    def candidates(x, y, segment, **kwargs):
        number = segment["segment"]
        found = []
        if number > 1:
            found = [{
                "segment": number, "direction": segment["direction"],
                "extremum_kind": kinds[number - 2],
                "potential": potentials[number - 2], "current": 1e-6,
                "prominence": 1e-6, "normalized_prominence": 1.0,
            }]
        return found, {**segment, "search_points": 20, "prominence": 1e-7,
                       "candidate_count": len(found)}

    monkeypatch.setattr(reference, "_reference_segment_candidates", candidates)
    return cv


def test_all_checked_pairs_exceed_cutoff_leads_with_separation_evidence(
    cv_factory, monkeypatch, capsys,
):
    cv = _candidate_cv(cv_factory, monkeypatch)
    before = cv.data.copy(deep=True)
    monkeypatch.setattr(reference, "_plot_reference_troubleshooting", lambda *a, **k: None)
    with pytest.raises(ValueError) as exc:
        reference.find_reference_midpoint_from_cv(
            cv, guess=0.5, window=0.3, smooth=False, troubleshoot=True,
        )
    message = str(exc.value)
    assert "All 5 checked candidate pairs exceed" in message
    assert message.index("All 5") < message.index("Guess:")
    assert "ΔEp = 280 mV > cutoff 200 mV" in message
    assert "segment 2 (increasing" in message and "segment 3 (decreasing" in message
    assert "unaccepted candidate midpoint = 0.59 V" in message
    assert "cycle identity is not inferred" in message
    suggestions = message.split("Suggestions:", 1)[1]
    assert "Move 'reference guess'" not in suggestions
    assert "Increase 'reference window'" not in suggestions
    assert "measured uncompensated resistance" in suggestions
    assert "not a diagnosis" in suggestions
    assert "relaxes the acceptance check" in suggestions
    assert "Record the chosen cutoff" in suggestions
    assert "never relaxed automatically" in suggestions
    output = capsys.readouterr().out
    assert output.index("All 5") < output.index("Reference Segment Summary:")
    assert "unaccepted candidate midpoint = 0.59 V" in output
    assert cv.data.equals(before)


def test_no_peaks_gets_search_guidance_without_separation_override(cv_factory):
    x = np.r_[np.linspace(-0.3, 0.3, 121), np.linspace(0.3, -0.3, 121)[1:]]
    cv = cv_factory(potential=x, current=np.zeros_like(x))
    with pytest.raises(ValueError) as exc:
        reference.find_reference_midpoint_from_cv(cv, guess=0, window=0.2, smooth=False)
    message = str(exc.value)
    assert "No eligible extrema were found" in message
    assert "peak prominence" in message.split("Suggestions:", 1)[1]
    assert "Increase 'reference max delta ep'" not in message
    assert "unaccepted candidate midpoint" not in message


@pytest.mark.parametrize("return_peak", [True, False])
def test_actual_peak_extraction_distinguishes_broad_pair_from_missing_return_peak(
    cv_factory, return_peak,
):
    forward = np.linspace(0.2, 0.9, 701)
    reverse = forward[-2::-1]
    x = np.r_[forward, reverse]
    y = np.r_[
        5e-6 * np.exp(-0.5 * ((forward - 0.73) / 0.02) ** 2),
        -4e-6 * np.exp(-0.5 * ((reverse - 0.45) / 0.02) ** 2)
        if return_peak else np.zeros_like(reverse),
    ]
    cv = cv_factory(potential=x, current=y)
    with pytest.raises(ValueError) as exc:
        reference.find_reference_midpoint_from_cv(
            cv, guess=0.5, window=0.3, smooth=False, prominence=1e-7,
        )
    message = str(exc.value)
    if return_peak:
        assert "The checked candidate pair exceeds" in message
        assert "ΔEp = 280 mV > cutoff 200 mV" in message
    else:
        assert "Eligible extrema are missing from one or more adjacent sweep segments" in message
        assert "Increase 'reference max delta ep'" not in message


def test_unsampled_guess_gets_targeted_guess_guidance(cv_factory, monkeypatch):
    cv = _candidate_cv(cv_factory, monkeypatch)
    with pytest.raises(ValueError) as exc:
        reference.find_reference_midpoint_from_cv(cv, guess=2.0, smooth=False)
    message = str(exc.value)
    assert "No adjacent opposite-direction segment pair sampled the reference guess" in message
    assert "Move 'reference guess'" in message
    assert "Increase 'reference max delta ep'" not in message


@pytest.mark.parametrize("potentials,kinds,expected", [
    (None, ["maximum"] * 6, "same extremum kind"),
    ([0.4, 0.7] * 3, None, "Epa must be greater than Epc"),
    (None, ["maximum", "maximum", "maximum", "minimum", "maximum", "minimum"],
     "multiple selection criteria"),
])
def test_other_pair_failures_are_not_misreported_as_all_separation_rejections(
    cv_factory, monkeypatch, potentials, kinds, expected,
):
    cv = _candidate_cv(cv_factory, monkeypatch, potentials=potentials, kinds=kinds)
    with pytest.raises(ValueError) as exc:
        reference.find_reference_midpoint_from_cv(cv, guess=0.5, smooth=False)
    message = str(exc.value)
    assert expected in message
    assert "All 5 checked candidate pairs exceed" not in message


def test_cutoff_default_and_explicit_override_selection_are_unchanged(cv_factory, monkeypatch):
    cv = _candidate_cv(cv_factory, monkeypatch)
    assert inspect.signature(reference.find_reference_midpoint_from_cv).parameters["max_delta_ep"].default == 0.2
    assert ImportOptions.from_options({}).reference_max_delta_ep == 0.2
    with pytest.raises(ValueError):
        reference.find_reference_midpoint_from_cv(cv, guess=0.5, smooth=False)
    midpoint, details = reference.find_reference_midpoint_from_cv(
        cv, guess=0.5, smooth=False, max_delta_ep=0.3,
    )
    assert midpoint == pytest.approx(0.575)
    assert details["delta_ep"] == pytest.approx(0.27)
    assert (details["anodic_segment"], details["cathodic_segment"]) == (4, 5)
    assert details["candidate_pairs_checked"] == 5
    assert details["rejected_pair_counts"]["delta ep exceeds maximum"] == 0
    assert "max_delta_ep" not in details  # Existing result schema remains unchanged.
