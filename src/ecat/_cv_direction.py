"""Scan-direction helpers shared by CV analyses."""

from __future__ import annotations

import numpy as np


def cv_segment_scan_direction(cv_obj, segment):
    """Return ``"increasing"`` or ``"decreasing"`` for one CV segment."""
    try:
        potential, _current = cv_obj.analysis_segment_data({"segment": int(segment)})
    except Exception as exc:
        raise ValueError(
            f"could not read potential data for segment {int(segment)}"
        ) from exc

    potential = np.asarray(potential, dtype=float)
    potential = potential[np.isfinite(potential)]
    if len(potential) < 2:
        raise ValueError(f"segment {int(segment)} has fewer than two finite potential points")

    differences = np.diff(potential)
    scale = max(float(np.ptp(potential)), float(np.max(np.abs(potential))), 1.0)
    tolerance = max(scale * 1e-9, 1e-12)
    differences = differences[np.abs(differences) > tolerance]
    if len(differences) == 0:
        raise ValueError(f"segment {int(segment)} has no resolvable potential movement")

    increasing = bool(np.any(differences > 0))
    decreasing = bool(np.any(differences < 0))
    if increasing and decreasing:
        raise ValueError(
            f"segment {int(segment)} contains both increasing and decreasing potential steps"
        )
    return "increasing" if increasing else "decreasing"


def cv_segment_branch(cv_obj, segment):
    """Map potential scan direction to the electrochemical branch name."""
    direction = cv_segment_scan_direction(cv_obj, segment)
    return "anodic" if direction == "increasing" else "cathodic"


def resolve_cv_segment_pair_branches(cvs, first_segment, second_segment, *, analysis_name):
    """Resolve one cathodic and one anodic segment across a CV series.

    ``None`` is returned when the supplied objects do not expose segment data;
    callers may then use another physically explicit signal such as branch-fit
    slope. Ambiguous observed directions raise immediately; valid mixed scan
    orders are mapped independently for each CV.
    """
    first_segment = int(first_segment)
    second_segment = int(second_segment)
    assignments = []
    unavailable = []

    for index, cv_obj in enumerate(cvs):
        name = getattr(cv_obj, "name", f"CV {index + 1}")
        if not callable(getattr(cv_obj, "analysis_segment_data", None)):
            unavailable.append({
                "name": name,
                "reason": "object does not expose analysis_segment_data()",
            })
            continue
        try:
            first_direction = cv_segment_scan_direction(cv_obj, first_segment)
            second_direction = cv_segment_scan_direction(cv_obj, second_segment)
        except ValueError as exc:
            raise ValueError(
                f"{analysis_name} could not assign cathodic and anodic branches for "
                f"'{name}' using segments {first_segment} and {second_segment}: {exc}."
            ) from exc

        first_branch = "anodic" if first_direction == "increasing" else "cathodic"
        second_branch = "anodic" if second_direction == "increasing" else "cathodic"
        if first_branch == second_branch:
            raise ValueError(
                f"{analysis_name} could not assign cathodic and anodic branches for "
                f"'{name}': segments {first_segment} and {second_segment} are both "
                f"{first_branch} ({first_direction} potential). Select one increasing "
                "and one decreasing segment."
            )

        cathodic_segment = first_segment if first_branch == "cathodic" else second_segment
        anodic_segment = first_segment if first_branch == "anodic" else second_segment
        assignments.append((index, name, cathodic_segment, anodic_segment))

    if not assignments:
        return None, {
            "source": "unavailable",
            "unavailable": unavailable,
        }

    unique = {
        (cathodic, anodic)
        for _index, _name, cathodic, anodic in assignments
    }
    if unavailable and len(unique) > 1:
        unavailable_names = ", ".join(item["name"] for item in unavailable)
        raise ValueError(
            f"{analysis_name} resolved mixed scan orders across the CV series but "
            f"could not inspect segment direction for {unavailable_names}. Branch "
            "assignment is ambiguous for those CVs."
        )

    if unavailable:
        common_cathodic, common_anodic = next(iter(unique))
        observed_by_index = {
            index: (cathodic, anodic)
            for index, _name, cathodic, anodic in assignments
        }
        assignments = [
            (
                index,
                getattr(cv_obj, "name", f"CV {index + 1}"),
                *observed_by_index.get(
                    index,
                    (common_cathodic, common_anodic),
                ),
            )
            for index, cv_obj in enumerate(cvs)
        ]

    assignments.sort(key=lambda item: item[0])
    cathodic_segments = [int(item[2]) for item in assignments]
    anodic_segments = [int(item[3]) for item in assignments]
    common_cathodic = cathodic_segments[0] if len(set(cathodic_segments)) == 1 else None
    common_anodic = anodic_segments[0] if len(set(anodic_segments)) == 1 else None
    return {
        "cathodic segment": common_cathodic,
        "anodic segment": common_anodic,
        "cathodic segments": cathodic_segments,
        "anodic segments": anodic_segments,
        "branch assignment source": "potential scan direction",
    }, {
        "source": "potential scan direction",
        "assignments": [
            {
                "index": int(index),
                "name": name,
                "cathodic segment": int(cathodic),
                "anodic segment": int(anodic),
            }
            for index, name, cathodic, anodic in assignments
        ],
        "unavailable": unavailable,
    }
