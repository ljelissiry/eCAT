"""Reference-potential helpers."""

import math
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _round_sigfigs(number, sigfigs):
    if number == 0:
        return 0
    return round(number, sigfigs - 1 - int(math.floor(math.log10(abs(number)))))


def midpoint_potential(E1, E2, sig_figs=None):
    """
    Lightweight helper so peak-potential fits and half_wave_potential can share
    the same midpoint logic.
    """
    E_half = (E1 + E2) / 2
    if sig_figs is not None:
        return _round_sigfigs(E_half, sig_figs)
    return E_half

__all__ = ["midpoint_potential"]

from .utils import *  # noqa: F401,F403
from ._cv_direction import split_cv_potential_segments
from .metadata import get_file_times
from .objects import echem
from .options import ImportOptions
from .parsers import exp_type_short as _exp_type_short, _format_file_for_warning

def _is_manual_shift(value):
    """Return True only for real numeric manual shifts, not bools."""
    return isinstance(value, Real) and not isinstance(value, bool)


def _contains_string_case_insensitive(text, pattern):
    return pattern.lower() in text.lower()


def _coerce_reference_mode(mode):
    if isinstance(mode, str):
        normalized = mode.strip().lower().replace("_", " ").replace("-", " ")
    else:
        normalized = str(mode).strip().lower()

    if normalized in {"off", "false", "0", "none", "null", "no"}:
        return "none"
    if normalized in {"auto", "manual", "keyword", "file"}:
        return normalized
    if mode is False:
        return "none"
    return normalized


def _coerce_reference_float(value, *, mode, label):
    if value is None:
        raise ValueError(f"{label} is required when reference mode='{mode}'.")
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be numeric when reference mode='{mode}'.") from exc


def _coerce_reference_keyword(value):
    if value is None:
        raise ValueError("A reference keyword is required when reference mode='keyword'.")
    if isinstance(value, bool):
        raise ValueError("A non-empty 'reference keyword' is required when reference mode='keyword'.")
    keyword = str(value).strip()
    if not keyword:
        raise ValueError("A non-empty 'reference keyword' is required when reference mode='keyword'.")
    return keyword


def _coerce_reference_file(value):
    if value is None:
        raise ValueError("'reference file' is required when reference mode='file'.")
    if not isinstance(value, (str, bytes, os.PathLike)):
        raise ValueError("'reference file' must be a file path string.")
    normalized = str(value).strip()
    if not normalized:
        raise ValueError("'reference file' is required when reference mode='file'.")
    return normalized


def _build_reference_failure_message(
    ref_cv,
    guess,
    window,
    prominence,
    max_delta_ep,
    segment_summaries,
    detected_candidates,
    total_pairs_checked,
    candidates_kept,
    rejected_pair_counts,
    failure_reason,
    smooth,
):
    """
    Build a readable diagnostic message when reference finding fails.
    """
    ref_name = getattr(ref_cv, "name", "Unknown reference file")
    ref_path = getattr(ref_cv, "filepath", None)
    ref_options = getattr(ref_cv, "options", {}) or {}

    lines = [
        f"Failed to identify a reference couple in: {ref_name}",
    ]

    if ref_path:
        lines.append(
            f"  File: {_format_file_for_warning(ref_path, ref_options.get('_display root'))}"
        )

    lines.extend([
        f"  Guess: {guess}",
        f"  Window: {window} V",
        f"  Smoothing: {smooth}",
        (
            "  Prominence: automatic per segment"
            if prominence is None
            else f"  Prominence: {round_sigfigs(float(prominence), 4)}"
        ),
        f"  max_delta_ep: {max_delta_ep} V",
        f"  Segments found: {len(segment_summaries)}",
        f"  Extrema found in search region: {len(detected_candidates)}",
        f"  Pair combinations checked: {total_pairs_checked}",
        f"  Valid adjacent-segment candidates: {candidates_kept}",
        f"  Reason: {failure_reason}",
    ])

    if rejected_pair_counts:
        lines.append("  Rejected pair counts:")
        for reason, count in sorted(rejected_pair_counts.items()):
            if count:
                lines.append(f"    - {reason}: {count}")

    if detected_candidates:
        lines.append("  Detected extrema:")
        for candidate in detected_candidates[:12]:
            lines.append(
                "    - segment "
                f"{candidate['segment']} ({candidate['direction']}), "
                f"{candidate['extremum_kind']} at "
                f"{round_sigfigs(candidate['potential'], 4)} V"
            )
        if len(detected_candidates) > 12:
            lines.append(f"    - ... ({len(detected_candidates)} total)")

    lines.extend([
        "",
        "Suggestions:",
        "  - Confirm that the file contains both sweeps of a reference couple.",
        "  - Move 'reference guess' closer to the expected reference potential.",
        "  - Increase 'reference window' if either sweep misses the search region.",
        "  - Adjust 'peak prominence' if real peaks are being missed or noise is selected.",
        "  - Increase 'reference max delta ep' only for a genuinely broader couple.",
        "  - Set 'troubleshoot': True to inspect segment candidates and the diagnostic plot.",
        "  - Use a better reference file or a manual shift when no physical pair is present.",
    ])

    return "\n".join(lines)


def _reference_current_scale(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return np.finfo(float).eps
    q05, q95 = np.quantile(values, [0.05, 0.95])
    median = float(np.median(values))
    mad_scale = 1.4826 * float(np.median(np.abs(values - median)))
    return max(float(q95 - q05), mad_scale, np.finfo(float).eps)


def _reference_segment_candidates(
    x,
    y,
    segment,
    *,
    guess,
    window,
    prominence,
    smooth,
    sg_options,
):
    start = segment["start"]
    stop = segment["stop"]
    segment_x = np.asarray(x[start:stop], dtype=float)
    segment_y = np.asarray(y[start:stop], dtype=float)
    smoothed_y = segment_y.copy()
    sg_meta = {"window": None, "polyorder": None}
    if smooth:
        smoothed_y, sg_meta = _savgol_apply(segment_y, sg_options, deriv=0)

    if guess is None:
        search_mask = np.ones(len(segment_x), dtype=bool)
    else:
        search_mask = (segment_x >= guess - window) & (segment_x <= guess + window)
    search_y = smoothed_y[search_mask]
    if len(search_y) < 3:
        return [], {
            **segment,
            "search_points": int(len(search_y)),
            "prominence": prominence,
            "current_scale": np.nan,
            "smoothing_window": sg_meta["window"],
            "candidate_count": 0,
        }

    current_scale = _reference_current_scale(search_y)
    segment_prominence = prominence
    if segment_prominence is None:
        dy = np.diff(search_y)
        noise = float(np.std(dy) / np.sqrt(2)) if len(dy) else 0.0
        segment_prominence = max(5.0 * noise, float(np.std(search_y)) * 0.05)

    maxima, maximum_properties = find_peaks(smoothed_y, prominence=segment_prominence)
    minima, minimum_properties = find_peaks(-smoothed_y, prominence=segment_prominence)
    boundary_margin = max(1, int((sg_meta.get("window") or 1) // 2))
    candidates = []
    for extremum_kind, indices, properties in (
        ("maximum", maxima, maximum_properties),
        ("minimum", minima, minimum_properties),
    ):
        for local_index, candidate_prominence in zip(indices, properties["prominences"]):
            local_index = int(local_index)
            if local_index < boundary_margin or local_index >= len(segment_x) - boundary_margin:
                continue
            potential = float(segment_x[local_index])
            if guess is not None and not (guess - window <= potential <= guess + window):
                continue
            candidates.append(
                {
                    "segment": int(segment["segment"]),
                    "direction": segment["direction"],
                    "extremum_kind": extremum_kind,
                    "local_index": local_index,
                    "index": int(start + local_index),
                    "potential": potential,
                    "current": float(segment_y[local_index]),
                    "smoothed_current": float(smoothed_y[local_index]),
                    "prominence": float(candidate_prominence),
                    "normalized_prominence": float(candidate_prominence / current_scale),
                }
            )

    return candidates, {
        **segment,
        "search_points": int(len(search_y)),
        "prominence": float(segment_prominence),
        "current_scale": float(current_scale),
        "smoothing_window": sg_meta["window"],
        "candidate_count": int(len(candidates)),
    }


def _reference_candidates_are_ambiguous(best, other, potential_resolution, numeric_guess):
    if numeric_guess and abs(best["midpoint_error_v"] - other["midpoint_error_v"]) > potential_resolution:
        return False
    if abs(best["delta_error_v"] - other["delta_error_v"]) > 2.0 * potential_resolution:
        return False

    prominence_scale = max(
        best["minimum_normalized_prominence"],
        other["minimum_normalized_prominence"],
        np.finfo(float).eps,
    )
    if (
        abs(
            best["minimum_normalized_prominence"]
            - other["minimum_normalized_prominence"]
        )
        / prominence_scale
        > 0.05
    ):
        return False
    return abs(best["prominence_balance"] - other["prominence_balance"]) <= abs(np.log(1.05))


def _print_reference_troubleshooting(
    segment_summaries,
    candidates,
    rejected_counts,
    best=None,
    selection_message=None,
):
    print("Reference Segment Summary:")
    segment_table = pd.DataFrame(
        [
            {
                "Segment": item["segment"],
                "Direction": item["direction"],
                "Potential Range / V": (
                    f"{round_sigfigs(item['potential_min'], 4)} to "
                    f"{round_sigfigs(item['potential_max'], 4)}"
                ),
                "Points": item["points"],
                "Search Points": item["search_points"],
                "Candidates": item["candidate_count"],
                "Prominence": (
                    "not estimated"
                    if item["prominence"] is None
                    else round_sigfigs(item["prominence"], 4)
                ),
            }
            for item in segment_summaries
        ]
    )
    print(segment_table.to_string(index=False))

    print("Reference Candidate Summary:")
    if candidates:
        candidate_table = pd.DataFrame(
            [
                {
                    "Segment": item["segment"],
                    "Direction": item["direction"],
                    "Extremum": item["extremum_kind"],
                    "Potential / V": round_sigfigs(item["potential"], 5),
                    "Prominence": round_sigfigs(item["prominence"], 4),
                    "Normalized Prominence": round_sigfigs(
                        item["normalized_prominence"], 4
                    ),
                }
                for item in candidates
            ]
        )
        print(candidate_table.to_string(index=False))
    else:
        print("No eligible extrema found.")

    if rejected_counts:
        rejected_table = pd.DataFrame(
            [
                {"Reason": reason, "Count": count}
                for reason, count in sorted(rejected_counts.items())
                if count
            ]
        )
        if not rejected_table.empty:
            print("Reference Pair Rejections:")
            print(rejected_table.to_string(index=False))

    if best is not None:
        print("Reference Pair Selected:")
        print(
            f"Epa = {round_sigfigs(best['Epa'], 5)} V "
            f"(segment {best['anodic_segment']}), "
            f"Epc = {round_sigfigs(best['Epc'], 5)} V "
            f"(segment {best['cathodic_segment']}), "
            f"midpoint = {round_sigfigs(best['midpoint'], 5)} V, "
            f"ΔEp = {round_sigfigs(best['delta_ep'], 5)} V"
        )
    elif selection_message:
        print("Reference Pair Selection:")
        print(selection_message)


def _plot_reference_troubleshooting(
    x,
    y,
    segment_summaries,
    ref_cv,
    *,
    candidates=None,
    best=None,
):
    figure, axis = plt.subplots(constrained_layout=True)
    colors = plt.get_cmap("viridis")(np.linspace(0.1, 0.9, len(segment_summaries)))
    for color, segment in zip(colors, segment_summaries):
        start = segment["start"]
        stop = segment["stop"]
        axis.plot(x[start:stop], y[start:stop], color=color, label=f"Segment {segment['segment']}")
    if best is None:
        for candidate in candidates or []:
            marker = "^" if candidate["extremum_kind"] == "maximum" else "v"
            axis.scatter(
                candidate["potential"],
                candidate["current"],
                color="0.35",
                marker=marker,
                zorder=5,
            )
    else:
        axis.scatter(
            [best["Epa"], best["Epc"]],
            [best["anodic_current"], best["cathodic_current"]],
            color=["tab:red", "tab:blue"],
            zorder=5,
        )
        axis.annotate(
            rf"$E_{{pa}}$, seg. {best['anodic_segment']}",
            (best["Epa"], best["anodic_current"]),
            xytext=(6, -8),
            textcoords="offset points",
            ha="left",
            va="top",
        )
        axis.annotate(
            rf"$E_{{pc}}$, seg. {best['cathodic_segment']}",
            (best["Epc"], best["cathodic_current"]),
            xytext=(6, 8),
            textcoords="offset points",
            ha="left",
            va="bottom",
        )
    axis.set_xlabel("Potential (V)")
    axis.set_ylabel("Current (A)")
    axis.set_title("Reference Pair Diagnostic")
    axis.legend()
    return figure, axis


def find_reference_midpoint_from_cv(
    ref_cv,
    guess=0.4,
    window=0.35,
    prominence=None,
    max_delta_ep=0.20,
    target_delta_ep=0.08,
    smooth=True,
    troubleshoot=False,
):
    """
    Find a reference midpoint from adjacent, opposite-direction CV segments.
    """
    if not isinstance(window, Real) or isinstance(window, bool) or float(window) <= 0:
        raise ValueError("reference window must be a positive number.")
    if not isinstance(max_delta_ep, Real) or isinstance(max_delta_ep, bool) or float(max_delta_ep) <= 0:
        raise ValueError("reference max delta ep must be a positive number.")
    if (
        not isinstance(target_delta_ep, Real)
        or isinstance(target_delta_ep, bool)
        or not 0 <= float(target_delta_ep) <= float(max_delta_ep)
    ):
        raise ValueError("reference target delta ep must be between 0 and reference max delta ep.")
    if prominence is not None and (
        not isinstance(prominence, Real)
        or isinstance(prominence, bool)
        or float(prominence) < 0
    ):
        raise ValueError("peak prominence must be a non-negative number or None.")

    if isinstance(guess, str):
        if guess.strip().lower() != "auto":
            raise ValueError("reference guess must be numeric or 'auto'.")
        guess = None
    elif not isinstance(guess, Real) or isinstance(guess, bool):
        raise ValueError("reference guess must be numeric or 'auto'.")
    else:
        guess = float(guess)

    x = np.asarray(ref_cv.x({"x axis": "Potential"}), dtype=float)
    y = np.asarray(ref_cv.y({"smooth": False}), dtype=float)
    if len(x) != len(y) or len(x) < 5:
        raise ValueError("Reference CV must contain at least five paired potential/current points.")
    finite = np.isfinite(x) & np.isfinite(y)
    x = x[finite]
    y = y[finite]

    segments = split_cv_potential_segments(x)
    default_sg_options = PeakPotentialOptions.from_options({})
    ref_options = getattr(ref_cv, "options", {}) or {}
    sg_options = {
        "noise window": ref_options.get("noise window", default_sg_options.noise_window),
        "noise polyorder": ref_options.get("noise polyorder", default_sg_options.noise_polyorder),
    }

    all_candidates = []
    segment_summaries = []
    candidates_by_segment = {}
    for segment in segments:
        segment_candidates, summary = _reference_segment_candidates(
            x,
            y,
            segment,
            guess=guess,
            window=float(window),
            prominence=prominence,
            smooth=bool(smooth),
            sg_options=sg_options,
        )
        candidates_by_segment[segment["segment"]] = segment_candidates
        all_candidates.extend(segment_candidates)
        segment_summaries.append(summary)

    finite_steps = np.abs(np.diff(x))
    finite_steps = finite_steps[finite_steps > max(float(np.ptp(x)) * 1e-9, 1e-12)]
    potential_resolution = float(np.median(finite_steps)) if len(finite_steps) else 1e-6
    rejected_counts = {
        "guess not sampled": 0,
        "same scan direction": 0,
        "missing eligible extrema": 0,
        "same extremum kind": 0,
        "Epa must be greater than Epc": 0,
        "delta ep exceeds maximum": 0,
    }
    valid_pairs = []
    total_pairs_checked = 0

    for first, second in zip(segments, segments[1:]):
        if first["direction"] == second["direction"]:
            rejected_counts["same scan direction"] += 1
            continue
        if guess is not None and not all(
            item["potential_min"] - potential_resolution
            <= guess
            <= item["potential_max"] + potential_resolution
            for item in (first, second)
        ):
            rejected_counts["guess not sampled"] += 1
            continue

        first_candidates = candidates_by_segment[first["segment"]]
        second_candidates = candidates_by_segment[second["segment"]]
        if not first_candidates or not second_candidates:
            rejected_counts["missing eligible extrema"] += 1
            continue

        for first_candidate in first_candidates:
            for second_candidate in second_candidates:
                total_pairs_checked += 1
                if first_candidate["extremum_kind"] == second_candidate["extremum_kind"]:
                    rejected_counts["same extremum kind"] += 1
                    continue

                if first_candidate["direction"] == "increasing":
                    anodic = first_candidate
                    cathodic = second_candidate
                else:
                    anodic = second_candidate
                    cathodic = first_candidate

                Epa = float(anodic["potential"])
                Epc = float(cathodic["potential"])
                if Epa <= Epc:
                    rejected_counts["Epa must be greater than Epc"] += 1
                    continue
                delta_ep = Epa - Epc
                if delta_ep > float(max_delta_ep):
                    rejected_counts["delta ep exceeds maximum"] += 1
                    continue

                midpoint = 0.5 * (Epa + Epc)
                midpoint_error_v = abs(midpoint - guess) if guess is not None else 0.0
                delta_error_v = abs(delta_ep - float(target_delta_ep))
                minimum_prominence = min(
                    anodic["normalized_prominence"],
                    cathodic["normalized_prominence"],
                )
                prominence_balance = abs(
                    np.log(
                        max(anodic["normalized_prominence"], np.finfo(float).eps)
                        / max(cathodic["normalized_prominence"], np.finfo(float).eps)
                    )
                )
                midpoint_bucket = int(np.floor(midpoint_error_v / potential_resolution + 0.5))
                delta_bucket = int(np.floor(delta_error_v / (2.0 * potential_resolution) + 0.5))
                rank = (
                    (midpoint_bucket, delta_bucket, -minimum_prominence, prominence_balance)
                    if guess is not None
                    else (delta_bucket, -minimum_prominence, prominence_balance)
                )
                valid_pairs.append(
                    {
                        "Epa": Epa,
                        "Epc": Epc,
                        "midpoint": float(midpoint),
                        "delta_ep": float(delta_ep),
                        "anodic_segment": int(anodic["segment"]),
                        "cathodic_segment": int(cathodic["segment"]),
                        "anodic_scan_direction": anodic["direction"],
                        "cathodic_scan_direction": cathodic["direction"],
                        "anodic_extremum_kind": anodic["extremum_kind"],
                        "cathodic_extremum_kind": cathodic["extremum_kind"],
                        "anodic_current": float(anodic["current"]),
                        "cathodic_current": float(cathodic["current"]),
                        "anodic_prominence": float(anodic["prominence"]),
                        "cathodic_prominence": float(cathodic["prominence"]),
                        "anodic_normalized_prominence": float(anodic["normalized_prominence"]),
                        "cathodic_normalized_prominence": float(cathodic["normalized_prominence"]),
                        "minimum_normalized_prominence": float(minimum_prominence),
                        "prominence_balance": float(prominence_balance),
                        "midpoint_error_v": float(midpoint_error_v),
                        "delta_error_v": float(delta_error_v),
                        "potential_resolution": float(potential_resolution),
                        "rank": rank,
                    }
                )

    if not valid_pairs:
        reason = (
            "No valid adjacent, opposite-direction, opposite-extremum reference pair was found."
        )
        if rejected_counts["Epa must be greater than Epc"]:
            reason += " Epa must be greater than Epc for every accepted pair."
        if troubleshoot:
            _print_reference_troubleshooting(
                segment_summaries,
                all_candidates,
                rejected_counts,
                selection_message=reason,
            )
            _plot_reference_troubleshooting(
                x,
                y,
                segment_summaries,
                ref_cv,
                candidates=all_candidates,
            )
        raise ValueError(
            _build_reference_failure_message(
                ref_cv=ref_cv,
                guess="auto" if guess is None else guess,
                window=window,
                prominence=prominence,
                max_delta_ep=max_delta_ep,
                segment_summaries=segment_summaries,
                detected_candidates=all_candidates,
                total_pairs_checked=total_pairs_checked,
                candidates_kept=0,
                rejected_pair_counts=rejected_counts,
                failure_reason=reason,
                smooth=smooth,
            )
        )

    valid_pairs.sort(key=lambda item: item["rank"])
    best = valid_pairs[0]
    if guess is None:
        ambiguous = [
            candidate
            for candidate in valid_pairs[1:]
            if _reference_candidates_are_ambiguous(
                best,
                candidate,
                potential_resolution,
                numeric_guess=False,
            )
        ]
        if ambiguous:
            alternatives = ", ".join(
                f"{round_sigfigs(item['midpoint'], 5)} V "
                f"(segments {item['anodic_segment']}/{item['cathodic_segment']})"
                for item in [best, *ambiguous[:4]]
            )
            ambiguity_message = (
                "Multiple reference couples are indistinguishable within the CV "
                f"potential resolution ({alternatives})."
            )
            if troubleshoot:
                _print_reference_troubleshooting(
                    segment_summaries,
                    all_candidates,
                    rejected_counts,
                    selection_message=ambiguity_message,
                )
                _plot_reference_troubleshooting(
                    x,
                    y,
                    segment_summaries,
                    ref_cv,
                    candidates=all_candidates,
                )
            raise ValueError(
                "Automatic reference selection is ambiguous: multiple reference couples "
                f"are indistinguishable within the CV potential resolution ({alternatives}). "
                "Provide a numeric 'reference guess' to select the intended couple."
            )

    best = dict(best)
    best.pop("rank", None)
    best["sequential_pairs_examined"] = max(0, len(segments) - 1)
    best["candidate_pairs_checked"] = int(total_pairs_checked)
    best["rejected_pair_counts"] = {
        reason: int(count) for reason, count in rejected_counts.items()
    }
    best["selection_mode"] = "ranked adjacent segments"

    if troubleshoot:
        _print_reference_troubleshooting(segment_summaries, all_candidates, rejected_counts, best)
        _plot_reference_troubleshooting(
            x,
            y,
            segment_summaries,
            ref_cv,
            candidates=all_candidates,
            best=best,
        )

    return float(best["midpoint"]), best

def _format_reference_display(ref_file, root_abs, *, quote=False):
    """
    Return a compact display string for a reference file.

    Root-folder references are shown as:
        filename.txt

    Subfolder references are shown as:
        subfolder/filename.txt
    """
    folder_abs = os.path.dirname(ref_file)
    rel_folder = os.path.relpath(folder_abs, root_abs)
    filename = os.path.basename(ref_file)

    if rel_folder == ".":
        display = filename
    else:
        display = os.path.join(rel_folder, filename).replace(os.sep, "/")
    return f"`{display}`" if quote else display

def _resolve_reference_file_path(ref_file, root_abs):
    if not ref_file:
        raise ValueError("'reference file' is required when reference mode='file'.")

    ref_path = os.path.expanduser(str(ref_file))
    if not os.path.isabs(ref_path):
        ref_path = os.path.join(root_abs, ref_path)
    ref_path = os.path.abspath(ref_path)

    if not os.path.exists(ref_path):
        display_path = _format_file_for_warning(ref_path, root_abs)
        raise ValueError(f"Reference file does not exist:\n{display_path}")
    if not os.path.isfile(ref_path):
        display_path = _format_file_for_warning(ref_path, root_abs)
        raise ValueError(f"Reference file is not a file:\n{display_path}")

    return ref_path

def _compute_explicit_reference_file_shift(reference_config, root_abs, options):
    ref_file = _resolve_reference_file_path(reference_config.get("file"), root_abs)

    ref_options = options.copy()
    ref_options["reference mode"] = "none"
    ref_options["print"] = False

    ref_cv = echem.from_file(ref_file, ref_options)
    shift_guess, pair_details = find_reference_midpoint_from_cv(
        ref_cv=ref_cv,
        guess=reference_config.get("guess", 0.4),
        window=options.get("reference window", 0.3),
        prominence=options.get("peak prominence", None),
        max_delta_ep=options.get("reference max delta ep", 0.20),
        target_delta_ep=options.get("reference target delta ep", 0.08),
        smooth=options.get("reference smooth", True),
        troubleshoot=options.get("troubleshoot", False),
    )
    return ref_file, shift_guess, pair_details

def _resolve_reference_shifts(file_paths, root_abs, options):
    """
    Determine nearest-folder / ancestor-folder references, and self-reference
    candidates in best-effort mode.

    Behavior
    --------
    For each folder containing data:
    - first look for a reference file in that folder
    - if none exists, walk upward through parent folders
    - stop at the requested root folder
    - do not search sideways into sibling folders

    Files that contain the reference keyword but are not the designated
    folder/ancestor reference are treated as self-reference candidates.
    """
    result = {
        "use_reference_files": False,
        "ref_name": None,
        "ref_mapping": {},
        "ref_shift_guess": {},        # designated folder/ancestor refs only
        "ref_pair_details": {},       # selected pair provenance for designated refs
        "self_ref_shift_guess": {},   # successful self-reference-only files
        "self_ref_pair_details": {},  # selected pair provenance for self refs
        "self_ref_failures": {},      # failed self-reference-only files
    }

    reference_mode = _coerce_reference_mode(options.get("reference mode", "none"))
    use_reference_files = reference_mode == "keyword"
    if not use_reference_files or len(file_paths) == 0:
        return result

    root_abs = os.path.abspath(root_abs)
    ref_name = str(options.get("reference keyword") or "Fc")
    file_paths_abs = [os.path.abspath(p) for p in file_paths]

    # Collect all reference-like files, indexed by their containing folder
    refs_by_folder = {}
    for fp in file_paths_abs:
        if _contains_string_case_insensitive(os.path.basename(fp), ref_name):
            folder_abs = os.path.dirname(fp)
            refs_by_folder.setdefault(folder_abs, []).append(fp)

    if len(refs_by_folder) == 0:
        print(
            f'Could not find any filenames containing "{ref_name}" in the searched folders.\n'
            "Files will remain unreferenced. Enter a manual shift with "
            "{'reference mode': 'manual', 'reference offset': 0.4} if needed."
        )
        return result

    folder_abs_set = sorted(set(os.path.dirname(fp) for fp in file_paths_abs))
    def _find_reference_candidates(folder_abs):
        """
        Return ordered reference candidates from folder_abs upward to root_abs.
        Local references are preferred, followed by nearest ancestors.
        """
        current = os.path.abspath(folder_abs)
        candidates = []

        while True:
            local_refs = refs_by_folder.get(current, [])
            if local_refs:
                candidates.extend(local_refs)

            if current == root_abs:
                break

            parent = os.path.dirname(current)
            if parent == current:
                break

            # Safety: do not walk above the requested root
            if os.path.commonpath([root_abs, parent]) != root_abs:
                break

            current = parent

        return candidates

    ref_shift_guess = {}
    ref_pair_details = {}
    self_ref_shift_guess = {}
    self_ref_pair_details = {}
    self_ref_failures = {}
    ref_failures = {}

    def _compute_shift(ref_file):
        ref_options = options.copy()
        ref_options["reference mode"] = "none"
        ref_options["print"] = False

        ref_cv = echem.from_file(ref_file, ref_options)
        shift_guess, pair_details = find_reference_midpoint_from_cv(
            ref_cv=ref_cv,
            guess=options.get("reference guess", 0.4),
            window=options.get("reference window", 0.3),
            prominence=options.get("peak prominence", None),
            max_delta_ep=options.get("reference max delta ep", 0.20),
            target_delta_ep=options.get("reference target delta ep", 0.08),
            smooth=options.get("reference smooth", True),
            troubleshoot=options.get("troubleshoot", False),
        )
        return shift_guess, pair_details

    def _compute_shift_cached(ref_file):
        if ref_file not in ref_shift_guess:
            shift_guess, pair_details = _compute_shift(ref_file)
            ref_shift_guess[ref_file] = shift_guess
            ref_pair_details[ref_file] = pair_details
        return ref_shift_guess[ref_file]

    ref_mapping = {}

    # Assign each folder the nearest usable reference. If a local reference-like
    # file cannot produce a midpoint, try other local/ancestor candidates before
    # failing the import.
    for folder_abs in folder_abs_set:
        candidates = _find_reference_candidates(folder_abs)
        ref_mapping[folder_abs] = None

        for ref_file in candidates:
            try:
                _compute_shift_cached(ref_file)
                ref_mapping[folder_abs] = ref_file
                break
            except ValueError as exc:
                ref_failures[ref_file] = str(exc)

        if ref_mapping[folder_abs] is None and candidates:
            display_name = _format_reference_display(candidates[0], root_abs, quote=True)
            raise ValueError(
                f"Reference shift assignment failed for: {display_name}\n\n"
                f"{ref_failures[candidates[0]]}"
            )

    # Best-effort self-reference candidates:
    # files that contain the ref keyword but are NOT the designated
    # folder/ancestor reference for that folder
    self_ref_candidates = []
    for fp in file_paths_abs:
        if not _contains_string_case_insensitive(os.path.basename(fp), ref_name):
            continue

        folder_abs = os.path.dirname(fp)
        assigned_ref = ref_mapping.get(folder_abs)

        if assigned_ref is None:
            continue

        if os.path.abspath(fp) != os.path.abspath(assigned_ref):
            self_ref_candidates.append(fp)

    # Best-effort self references: store failure, do not raise
    for ref_file in self_ref_candidates:
        try:
            shift_guess, pair_details = _compute_shift(ref_file)
            self_ref_shift_guess[ref_file] = shift_guess
            self_ref_pair_details[ref_file] = pair_details
        except ValueError as exc:
            self_ref_failures[ref_file] = str(exc)

    result["use_reference_files"] = True
    result["ref_name"] = ref_name
    result["ref_mapping"] = ref_mapping
    result["ref_shift_guess"] = ref_shift_guess
    result["ref_pair_details"] = ref_pair_details
    result["self_ref_shift_guess"] = self_ref_shift_guess
    result["self_ref_pair_details"] = self_ref_pair_details
    result["self_ref_failures"] = self_ref_failures
    return result

def _reference_usage_counts(reference_records):
    # Only summarize actual CV experiment files, not dedicated reference scans
    records = [
        r for r in reference_records
        if r.get("is_cv", True) and r.get("mode") != "reference_file"
    ]

    counts = {
        "folder/ancestor reference": sum(r["mode"] == "folder" for r in records),
        "explicit reference file": sum(r["mode"] == "file" for r in records),
        "self-referenced successfully": sum(r["mode"] == "self" for r in records),
        "self-reference fallback used": sum(r["mode"] == "fallback" for r in records),
        "explicit reference map": sum(r["mode"] == "map" for r in records),
        "unreferenced": sum(r["mode"] == "none" for r in records)
    }

    return [(label, n) for label, n in counts.items() if n > 0]


def _print_reference_usage_summary(reference_records):
    nonzero = _reference_usage_counts(reference_records)

    if not nonzero:
        return

    if len(nonzero) == 1:
        label, n = nonzero[0]

        if label == "folder/ancestor reference":
            print("All CV files referenced using folder/ancestor reference.")
        elif label == "explicit reference file":
            print("All CV files referenced using the explicit reference file.")
        elif label == "self-referenced successfully":
            print("All eligible CV files self-referenced successfully.")
        elif label == "self-reference fallback used":
            print("All eligible CV files used folder/ancestor fallback after self-reference failed.")
        elif label == "explicit reference map":
            print("All CV files referenced using the explicit reference map.")
        return

    print("Reference usage summary:")
    for label, n in nonzero:
        print(f"  {label}: {n}")


def _print_reference_correction_summary(
    reference_config,
    reference_info=None,
    reference_records=None,
    root_abs=None,
    explicit_reference_file=None,
    explicit_reference_shift=None,
):
    """Print a compact, notebook-friendly reference-correction log."""
    mode = reference_config.get("mode", "none")
    reference_info = {} if reference_info is None else reference_info
    reference_records = [] if reference_records is None else reference_records
    root_abs = os.path.abspath("." if root_abs is None else root_abs)

    if mode == "none" and not any(r.get("mode") not in (None, "none") for r in reference_records):
        return

    print("Reference correction:")
    print(f"  Mode: {mode}")

    if mode == "manual":
        print(f"  Shift: {round_sigfigs(reference_config.get('offset'), 4)} V")

    elif mode == "file":
        if explicit_reference_file is not None:
            print(f"  File: {_format_reference_display(explicit_reference_file, root_abs, quote=True)}")
        if explicit_reference_shift is not None:
            print(f"  Midpoint: {round_sigfigs(explicit_reference_shift, 4)} V")

    elif mode == "keyword":
        print(f"  Keyword: {reference_config.get('keyword')}")
        guess = reference_config.get("guess", "auto")
        if str(guess).lower() != "auto":
            print(f"  Guess: {round_sigfigs(guess, 4)} V")
        else:
            print("  Guess: auto")

    elif mode == "auto":
        keywords = reference_config.get("keywords") or []
        if keywords:
            print(f"  Keywords: {', '.join(map(str, keywords))}")
        chosen_keyword = reference_info.get("chosen_keyword")
        if chosen_keyword:
            print(f"  Chosen keyword: {chosen_keyword}")
        guess = reference_config.get("guess", "auto")
        if str(guess).lower() != "auto":
            print(f"  Guess: {round_sigfigs(guess, 4)} V")
        else:
            print("  Guess: auto")

    ref_mapping = reference_info.get("ref_mapping") or {}
    ref_shift_guess = reference_info.get("ref_shift_guess") or {}
    if ref_mapping:
        folder_items = []
        for folder_abs in sorted(ref_mapping):
            rel_folder = os.path.relpath(folder_abs, root_abs)
            folder_label = "Base Folder" if rel_folder == "." else rel_folder
            ref_file = ref_mapping[folder_abs]
            if ref_file is None:
                folder_items.append((folder_label, "none"))
            else:
                shift = ref_shift_guess.get(ref_file)
                ref_text = _format_reference_display(ref_file, root_abs, quote=True)
                if shift is not None:
                    ref_text = f"{ref_text} = {round_sigfigs(shift, 4)} V"
                folder_items.append((folder_label, ref_text))

        if len(folder_items) == 1 and folder_items[0][0] == "Base Folder":
            print(f"  Folder reference: {folder_items[0][1]}")
        else:
            print("  Folder references:")
            for folder_label, ref_text in folder_items:
                print(f"    {folder_label}: {ref_text}")

    nonzero = _reference_usage_counts(reference_records)
    if nonzero:
        print("  Usage:")
        for label, n in nonzero:
            print(f"    {label}: {n}")

    print()

def _print_reference_usage_list(reference_records, root_abs):
    print("Reference assignments:")
    for r in reference_records:
        idx = r["index"]
        name = _format_reference_display(r["filepath"], root_abs, quote=True)

        if r["mode"] == "self":
            print(f"  [{idx}] {name}  -> self  ({round_sigfigs(r['shift'], 4)} V)")
        elif r["mode"] == "fallback":
            ref_display = _format_reference_display(r["ref_file"], root_abs, quote=True)
            print(
                f"  [{idx}] {name}  -> fallback to {ref_display} "
                f"({round_sigfigs(r['shift'], 4)} V)"
            )
        elif r["mode"] == "folder":
            ref_display = _format_reference_display(r["ref_file"], root_abs, quote=True)
            print(
                f"  [{idx}] {name}  -> {ref_display} "
                f"({round_sigfigs(r['shift'], 4)} V)"
            )
        elif r["mode"] == "file":
            ref_display = _format_reference_display(r["ref_file"], root_abs, quote=True)
            print(
                f"  [{idx}] {name}  -> explicit file {ref_display} "
                f"({round_sigfigs(r['shift'], 4)} V)"
            )
        elif r["mode"] == "manual":
            print(f"  [{idx}] {name}  -> manual shift ({round_sigfigs(r['shift'], 4)} V)")
        elif r["mode"] == "map":
            ref_display = _format_reference_display(r["ref_file"], root_abs, quote=True)
            print(
                f"  [{idx}] {name}  -> mapped to {ref_display} "
                f"({round_sigfigs(r['shift'], 4)} V)"
            )
        else:
            print(f"  [{idx}] {name}")
            
def _print_reference_usage_troubleshoot(reference_records, root_abs):
    print("Detailed reference assignments:")
    for r in reference_records:
        idx = r["index"]
        name = _format_reference_display(r["filepath"], root_abs, quote=True)
        print(f"  [{idx}] {name}")
        print(f"    mode: {r['mode']}")

        if r["ref_file"] is not None:
            print(f"    reference file: {_format_reference_display(r['ref_file'], root_abs, quote=True)}")

        if r["shift"] is not None:
            print(f"    shift: {round_sigfigs(r['shift'], 4)} V")

        if r["failure_message"]:
            print("    self-reference failure:")
            for line in str(r["failure_message"]).splitlines():
                print(f"      {line}")


def _normalize_reference_map(reference_map, object_count):
    if reference_map in (None, {}):
        return {}
    if not isinstance(reference_map, dict):
        raise ValueError("'reference map' must be a dictionary such as {45: 54}.")

    normalized = {}
    for target_idx, reference_idx in reference_map.items():
        if not isinstance(target_idx, int) or not isinstance(reference_idx, int):
            raise ValueError("'reference map' keys and values must be integer object indices.")
        target = target_idx + object_count if target_idx < 0 else target_idx
        reference = reference_idx + object_count if reference_idx < 0 else reference_idx
        if target < 0 or target >= object_count:
            raise IndexError(
                f"'reference map' target index {target_idx} is out of range "
                f"for {object_count} imported objects."
            )
        if reference < 0 or reference >= object_count:
            raise IndexError(
                f"'reference map' reference index {reference_idx} is out of range "
                f"for {object_count} imported objects."
            )
        normalized[target] = reference
    return normalized


def _apply_reference_map(object_list, reference_records, reference_map, reference_config, reference_label, options):
    normalized_map = _normalize_reference_map(reference_map, len(object_list))
    if not normalized_map:
        return

    shift_cache = {}
    pair_details_cache = {}

    def _mapped_shift(reference_idx):
        if reference_idx not in shift_cache:
            ref_obj = object_list[reference_idx]
            existing_shift = getattr(ref_obj, "reference_shift", None)
            try:
                ref_obj.reference_shift = None
                shift_guess, pair_details = find_reference_midpoint_from_cv(
                    ref_cv=ref_obj,
                    guess=reference_config.get("guess", 0.4),
                    window=options.get("reference window", 0.3),
                    prominence=options.get("peak prominence", None),
                    max_delta_ep=options.get("reference max delta ep", 0.20),
                    target_delta_ep=options.get("reference target delta ep", 0.08),
                    smooth=options.get("reference smooth", True),
                    troubleshoot=options.get("troubleshoot", False),
                )
            finally:
                ref_obj.reference_shift = existing_shift
            shift_cache[reference_idx] = shift_guess
            pair_details_cache[reference_idx] = pair_details
        return shift_cache[reference_idx], pair_details_cache[reference_idx]

    for target_idx, reference_idx in normalized_map.items():
        ref_obj = object_list[reference_idx]
        target_obj = object_list[target_idx]
        shift_guess, pair_details = _mapped_shift(reference_idx)

        record = reference_records[target_idx]
        record["mode"] = "map"
        record["ref_file"] = os.path.abspath(ref_obj.filepath)
        record["shift"] = shift_guess
        record["pair_details"] = deepcopy(pair_details)
        record["failure_message"] = None

        target_obj.reference_shift = shift_guess
        target_obj.reference_label = reference_label
        target_obj.reference_mode = "map"
        target_obj.reference_source_file = os.path.abspath(ref_obj.filepath)
        target_obj.reference_pair_details = deepcopy(pair_details)
        target_obj.reference_failure_message = None
        if getattr(target_obj, "parse_result", None) is not None:
            target_obj.parse_result.metadata["reference_pair_details"] = deepcopy(pair_details)


REFERENCE_LABEL_MAP = {
    "fc": "Fc/Fc+",
    "ferrocene": "Fc/Fc+",
    "fc/fc+": "Fc/Fc+",

    "dmfc": "DmFc/DmFc+",
    "decamethylferrocene": "DmFc/DmFc+",
    "dmfc/dmfc+": "DmFc/DmFc+",

    "cocp2": "CoCp2/CoCp2+",
    "cobaltocene": "CoCp2/CoCp2+",
    "cocp2/cocp2+": "CoCp2/CoCp2+",
}

def canonical_reference_label(keyword, default="reference"):
    if keyword is None:
        return default
    return REFERENCE_LABEL_MAP.get(str(keyword).strip().lower(), str(keyword))

def resolve_reference_options(options):
    options = dict(options)

    mode = _coerce_reference_mode(options.get("reference mode", "none"))
    offset = options.get("reference offset")
    ref_file = options.get("reference file")
    ref_keyword = options.get("reference keyword")
    ref_keywords = options.get("reference keywords")
    guess = options.get("reference guess", "auto")
    allow_self = options.get("allow self reference", True)
    label = options.get("reference label", "Fc/Fc+")

    if mode == "none":
        return {"mode": "none", "label": label}

    if mode == "manual":
        return {
            "mode": "manual",
            "offset": _coerce_reference_float(
                offset,
                mode="manual",
                label="'reference offset'",
            ),
            "label": label,
        }

    if mode == "file":
        ref_file = _coerce_reference_file(ref_file)
        return {
            "mode": "file",
            "file": ref_file,
            "guess": guess,
            "label": label,
            "allow self reference": allow_self,
        }

    if mode == "keyword":
        ref_keyword = _coerce_reference_keyword(ref_keyword)
        return {
            "mode": "keyword",
            "keyword": ref_keyword,
            "guess": guess,
            "label": label,
            "allow self reference": allow_self,
        }

    if mode == "auto":
        if offset is not None:
            return {
                "mode": "manual",
                "offset": _coerce_reference_float(
                    offset,
                    mode="auto",
                    label="'reference offset'",
                ),
                "label": label,
            }

        if ref_file:
            return {
                "mode": "file",
                "file": ref_file,
                "guess": guess,
                "label": label,
                "allow self reference": allow_self,
            }

        keyword_list = []
        if ref_keyword:
            keyword_list.append(ref_keyword)

        if ref_keywords:
            keyword_list.extend(ref_keywords)
        if not keyword_list:
            raise ValueError(
                "reference mode='auto' requires at least one entry in 'reference keywords' "
                "or 'reference keyword'."
            )

        return {
            "mode": "auto",
            "keywords": keyword_list,
            "guess": guess,
            "label": label,
            "allow self reference": allow_self,
        }

    raise ValueError(f"Unknown reference mode: {mode}")


__all__ = [
    'midpoint_potential',
    'find_reference_midpoint_from_cv',
    'canonical_reference_label',
    'resolve_reference_options',
]
