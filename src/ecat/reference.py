"""Reference-potential helpers."""

import math
import os

import numpy as np


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


def _format_peak_list(x_vals, idx, max_peaks=6, sig_figs=4):
    """
    Format the first few detected peak potentials for error messages.
    """
    if len(idx) == 0:
        return "None"

    peaks = [round_sigfigs(float(x_vals[i]), sig_figs) for i in idx[:max_peaks]]
    text = ", ".join(f"{p} V" for p in peaks)

    if len(idx) > max_peaks:
        text += f", ... ({len(idx)} total)"
    return text


def _build_reference_failure_message(
    ref_cv,
    guess,
    window,
    prominence,
    max_delta_ep,
    x_use,
    ox_idx,
    red_idx,
    total_pairs_checked,
    candidates_kept,
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

    if len(x_use) > 0:
        lines.append(
            f"  Search region: {round_sigfigs(float(np.min(x_use)), 4)} to "
            f"{round_sigfigs(float(np.max(x_use)), 4)} V"
        )

    lines.extend([
        f"  Guess: {guess}",
        f"  Window: {window} V",
        f"  Smoothing: {smooth}",
        f"  Prominence used: {round_sigfigs(float(prominence), 4) if prominence is not None else prominence}",
        f"  max_delta_ep: {max_delta_ep} V",
        f"  Oxidation peaks found: {len(ox_idx)}",
        f"  Reduction peaks found: {len(red_idx)}",
        f"  Pair combinations checked: {total_pairs_checked}",
        f"  Candidate pairs passing ΔEp filter: {candidates_kept}",
        f"  Reason: {failure_reason}",
        "",
        "Detected peak locations:",
        f"  Oxidation: {_format_peak_list(x_use, ox_idx)}",
        f"  Reduction: {_format_peak_list(x_use, red_idx)}",
        "",
        "Suggestions:",
        "  - Plot this reference file and confirm it actually contains a clean reversible couple.",
        "  - Move 'reference guess' closer to the expected Fc/Fc+ position.",
        "  - Increase 'reference window' if the search region is too narrow.",
        "  - Increase 'reference max delta ep' if the couple is broader than expected.",
        "  - Try setting 'troubleshoot': True to inspect the search behavior.",
        "  - If the file is not a good reference, rename/exclude it or use a manual shift.",
    ])

    return "\n".join(lines)


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
    Find the midpoint of the best reversible-looking redox couple in a CV.
    """
    x = np.asarray(ref_cv.x(), dtype=float)
    y = np.asarray(ref_cv.y({"smooth": False}), dtype=float)
    
    sg_meta = {"window": None, "polyorder": None}

    if smooth:
        default_sg_options = PeakPotentialOptions.from_options({})
        sg_options = {
            "noise window": ref_cv.options.get("noise window", default_sg_options.noise_window),
            "noise polyorder": ref_cv.options.get("noise polyorder", default_sg_options.noise_polyorder),
        }

        y, sg_meta = _savgol_apply(y, sg_options, deriv=0)

        if troubleshoot:
            print(
                f"Reference SG window={sg_meta['window']}, "
                f"polyorder={sg_meta['polyorder']}"
            )

    if isinstance(guess, str) and guess.lower() == "auto":
        guess = None

    if isinstance(guess, Real) and not isinstance(guess, bool):
        mask = (x >= guess - window) & (x <= guess + window)
        x_use = x[mask]
        y_use = y[mask]
    else:
        x_use = x
        y_use = y

    if len(x_use) < 5:
        raise ValueError(
            f"Failed to identify a reference couple in {ref_cv.name}: "
            "not enough data in the search region to identify a reference couple."
        )

    if prominence is None:
        dy = np.diff(y_use)
        noise = np.std(dy) / np.sqrt(2) if len(dy) else 0.0
        prominence = max(5 * noise, np.std(y_use) * 0.05)

    ox_idx, _ = find_peaks(y_use, prominence=prominence)
    red_idx, _ = find_peaks(-y_use, prominence=prominence)

    total_pairs_checked = len(ox_idx) * len(red_idx)

    if len(ox_idx) == 0 or len(red_idx) == 0:
        msg = _build_reference_failure_message(
            ref_cv=ref_cv,
            guess=guess,
            window=window,
            prominence=prominence,
            max_delta_ep=max_delta_ep,
            x_use=x_use,
            ox_idx=ox_idx,
            red_idx=red_idx,
            total_pairs_checked=total_pairs_checked,
            candidates_kept=0,
            failure_reason="Could not find both anodic and cathodic peaks.",
            smooth=smooth,
        )
        raise ValueError(msg)

    candidates = []
    rejected_by_delta_ep = 0

    for i in ox_idx:
        for j in red_idx:
            ep_ox = float(x_use[i])
            ep_red = float(x_use[j])
            delta_ep = abs(ep_ox - ep_red)

            if delta_ep > max_delta_ep:
                rejected_by_delta_ep += 1
                continue

            midpoint = 0.5 * (ep_ox + ep_red)
            i_ox = float(y_use[i])
            i_red = float(y_use[j])

            pair_strength = abs(i_ox - i_red)
            magnitude_mismatch = abs(abs(i_ox) - abs(i_red))

            score = pair_strength - 0.5 * magnitude_mismatch
            score -= abs(delta_ep - target_delta_ep)

            if isinstance(guess, Real) and not isinstance(guess, bool):
                score -= 3.0 * abs(midpoint - guess)

            candidates.append(
                {
                    "score": score,
                    "midpoint": midpoint,
                    "Ep_ox": ep_ox,
                    "Ep_red": ep_red,
                    "delta_ep": delta_ep,
                    "i_ox": i_ox,
                    "i_red": i_red,
                }
            )

    if not candidates:
        msg = _build_reference_failure_message(
            ref_cv=ref_cv,
            guess=guess,
            window=window,
            prominence=prominence,
            max_delta_ep=max_delta_ep,
            x_use=x_use,
            ox_idx=ox_idx,
            red_idx=red_idx,
            total_pairs_checked=total_pairs_checked,
            candidates_kept=0,
            failure_reason=(
                f"No oxidation/reduction peak pairs passed the ΔEp filter "
                f"(rejected {rejected_by_delta_ep} pair(s))."
            ),
            smooth=smooth,
        )
        raise ValueError(msg)

    best = max(candidates, key=lambda d: d["score"])

    if troubleshoot:
        print("Reference pair selected:")
        print(
            f"Epa = {round_sigfigs(best['Ep_ox'], 4)} V, "
            f"Epc = {round_sigfigs(best['Ep_red'], 4)} V, "
            f"midpoint = {round_sigfigs(best['midpoint'], 4)} V, "
            f"ΔEp = {round_sigfigs(best['delta_ep'], 4)} V"
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
    shift_guess, _ = find_reference_midpoint_from_cv(
        ref_cv=ref_cv,
        guess=reference_config.get("guess", 0.4),
        window=options.get("reference window", 0.3),
        prominence=options.get("peak prominence", None),
        max_delta_ep=options.get("reference max delta ep", 0.20),
        target_delta_ep=options.get("reference target delta ep", 0.08),
        smooth=options.get("reference smooth", True),
        troubleshoot=options.get("troubleshoot", False),
    )
    return ref_file, shift_guess

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
        "self_ref_shift_guess": {},   # successful self-reference-only files
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
    self_ref_shift_guess = {}
    self_ref_failures = {}
    ref_failures = {}

    def _compute_shift(ref_file):
        ref_options = options.copy()
        ref_options["reference mode"] = "none"
        ref_options["print"] = False

        ref_cv = echem.from_file(ref_file, ref_options)
        shift_guess, _ = find_reference_midpoint_from_cv(
            ref_cv=ref_cv,
            guess=options.get("reference guess", 0.4),
            window=options.get("reference window", 0.3),
            prominence=options.get("peak prominence", None),
            max_delta_ep=options.get("reference max delta ep", 0.20),
            target_delta_ep=options.get("reference target delta ep", 0.08),
            smooth=options.get("reference smooth", True),
            troubleshoot=options.get("troubleshoot", False),
        )
        return shift_guess

    def _compute_shift_cached(ref_file):
        if ref_file not in ref_shift_guess:
            ref_shift_guess[ref_file] = _compute_shift(ref_file)
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
            self_ref_shift_guess[ref_file] = _compute_shift(ref_file)
        except ValueError as exc:
            self_ref_failures[ref_file] = str(exc)

    result["use_reference_files"] = True
    result["ref_name"] = ref_name
    result["ref_mapping"] = ref_mapping
    result["ref_shift_guess"] = ref_shift_guess
    result["self_ref_shift_guess"] = self_ref_shift_guess
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

    def _mapped_shift(reference_idx):
        if reference_idx not in shift_cache:
            ref_obj = object_list[reference_idx]
            existing_shift = getattr(ref_obj, "reference_shift", None)
            try:
                ref_obj.reference_shift = None
                shift_guess, _ = find_reference_midpoint_from_cv(
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
        return shift_cache[reference_idx]

    for target_idx, reference_idx in normalized_map.items():
        ref_obj = object_list[reference_idx]
        target_obj = object_list[target_idx]
        shift_guess = _mapped_shift(reference_idx)

        record = reference_records[target_idx]
        record["mode"] = "map"
        record["ref_file"] = os.path.abspath(ref_obj.filepath)
        record["shift"] = shift_guess
        record["failure_message"] = None

        target_obj.reference_shift = shift_guess
        target_obj.reference_label = reference_label
        target_obj.reference_mode = "map"
        target_obj.reference_source_file = os.path.abspath(ref_obj.filepath)
        target_obj.reference_failure_message = None


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
