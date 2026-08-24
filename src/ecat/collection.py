"""Filtering, sorting, grouping, and collection-summary helpers."""

from collections.abc import Mapping

from .utils import *  # noqa: F401,F403
from .options import FilterOptions, GroupSummaryOptions, SortGroupOptions
from .parsers import exp_type_short as _exp_type_short
from .plotting import (
    _coerce_display_columns,
    display_object_table,
    pretty_table_column_label,
    show_groups,
    show_objects,
)

def get_sort_group_dict():
    """Returns a dictionary of key functions used for sorting and grouping."""
    from .objects import cv

    def get_name(echem_object):
        return echem_object.name

    def get_folderpath(echem_object):
        return echem_object.folderpath

    def get_compounds(echem_object):
        return getattr(echem_object, "compounds", [])

    def get_concentrations(echem_object):
        concs = []
        for c in getattr(echem_object, "concentrations", []):
            concs.append(concentration_to_float(c))
        return concs

    def get_species(echem_object):
        return [
            f"{conc} {comp}" for conc, comp in zip(
                getattr(echem_object, "concentrations", []) or [],
                getattr(echem_object, "compounds", []) or [],
            )
        ]

    def get_type(echem_object):
        return echem_object.type

    def get_scan_rate(echem_object):
        if isinstance(echem_object, cv):
            return echem_object.scan_rate
        return echem_object.delta_x

    def get_gas(echem_object):
        return getattr(echem_object, "gas", None) or '*None'

    def get_solvent(echem_object):
        return getattr(echem_object, "solvent", None) or '*None'

    def get_ir_comp_resistance(echem_object):
        return getattr(echem_object, "ir_comp_resistance", None)

    def get_ir_uncomp_resistance(echem_object):
        return getattr(echem_object, "ir_uncomp_resistance", None)

    def get_ir_comp_percent(echem_object):
        return getattr(echem_object, "ir_comp_percent", None)

    def get_scan_window(echem_object):
        if isinstance(echem_object, cv):
            return [echem_object.max_E, echem_object.min_E]
        return [0]

    def get_segments(echem_object):
        if isinstance(echem_object, cv):
            return echem_object.segments
        return 0

    def get_creation_time(echem_object):
        return getattr(echem_object, "creation_time", None)

    def get_timestamp(echem_object):
        return _object_time_for_sort(echem_object)

    def get_applied_potential(echem_object):
        return getattr(echem_object, "init_E", None)

    def get_sample_interval(echem_object):
        return (
            getattr(echem_object, "sample_interval", None)
            or getattr(echem_object, "sample_int", None)
        )

    def get_run_time(echem_object):
        run_time = getattr(echem_object, "run_time", None)
        if run_time is not None:
            return run_time
        data = getattr(echem_object, "data", None)
        if data is not None and "Time" in data and len(data["Time"]) > 0:
            return float(data["Time"].iloc[-1] - data["Time"].iloc[0])
        return None

    def get_final_charge(echem_object):
        charge_method = getattr(echem_object, "charge", None)
        if not callable(charge_method):
            return None
        try:
            return charge_method({"plot": False, "print": False})["final charge"]
        except Exception:
            return None

    def get_min_current(echem_object):
        data = getattr(echem_object, "data", None)
        if data is not None and "Current" in data:
            return data["Current"].min()
        return None

    def get_max_current(echem_object):
        data = getattr(echem_object, "data", None)
        if data is not None and "Current" in data:
            return data["Current"].max()
        return None

    def get_avg_current(echem_object):
        data = getattr(echem_object, "data", None)
        if data is not None and "Current" in data:
            return data["Current"].mean()
        return None

    def get_cathodic_current(echem_object):
        return getattr(echem_object, "cathodic_current", None)

    def get_anodic_current(echem_object):
        return getattr(echem_object, "anodic_current", None)

    def get_cathodic_time(echem_object):
        return getattr(echem_object, "cathodic_time", None)

    def get_anodic_time(echem_object):
        return getattr(echem_object, "anodic_time", None)

    def get_high_potential_limit(echem_object):
        return getattr(echem_object, "high_E_limit", None)

    def get_low_potential_limit(echem_object):
        return getattr(echem_object, "low_E_limit", None)

    def get_quiet_time(echem_object):
        return getattr(echem_object, "quiet_time", None)

    return {
        'name': get_name,
        'subfolder': get_folderpath,
        'compounds': get_compounds,
        'concentrations': get_concentrations,
        'species': get_species,
        'type': get_type,
        'scan rate': get_scan_rate,
        'gas': get_gas,
        'solvent': get_solvent,
        'ir comp resistance': get_ir_comp_resistance,
        'ir uncomp resistance': get_ir_uncomp_resistance,
        'ir comp percent': get_ir_comp_percent,
        'scan window': get_scan_window,
        'segments': get_segments,
        'creation time': get_creation_time,
        'timestamp': get_timestamp,
        'applied potential': get_applied_potential,
        'sample interval': get_sample_interval,
        'run time': get_run_time,
        'final charge': get_final_charge,
        'cathodic current': get_cathodic_current,
        'anodic current': get_anodic_current,
        'cathodic time': get_cathodic_time,
        'anodic time': get_anodic_time,
        'high potential limit': get_high_potential_limit,
        'low potential limit': get_low_potential_limit,
        'quiet time': get_quiet_time,
    }


def validate_keys(keys, valid_keys, key_type):
    """Checks if all keys are valid and prints available options if not."""
    invalid_keys = [key for key in keys if key not in valid_keys]
    if invalid_keys:
        print("\033[91m",f"Invalid {key_type} key(s):", invalid_keys,"\033[0m")
        print(f"Valid {key_type} options are:", list(valid_keys))
        return False
    return True


def _flatten_object_list(object_list):
    """
    Flatten a possibly nested object/group list into a simple list of echem objects.
    """
    flat = []
    for item in object_list:
        if isinstance(item, list):
            flat.extend(_flatten_object_list(item))
        else:
            flat.append(item)
    return flat


def _normalize_filter_value(value):
    """
    Normalize strings for matching:
    - remove spaces
    - lowercase

    Lists/tuples are normalized elementwise.
    """
    if isinstance(value, str):
        return value.replace(" ", "").lower()
    elif isinstance(value, (list, tuple)):
        return [_normalize_filter_value(v) for v in value]
    return value


def _unique_preserve_order(values):
    """
    Return unique values while preserving first-seen order.
    """
    unique = []
    seen = set()

    for value in values:
        # Make lists hashable if needed
        if isinstance(value, list):
            key = tuple(value)
        else:
            key = value

        if key not in seen:
            seen.add(key)
            unique.append(value)

    return unique


def _sort_filter_values(values):
    """
    Sort values for display when possible.
    - numeric values sort numerically
    - otherwise sort by string representation
    """
    try:
        if all(isinstance(v, (int, float, np.integer, np.floating)) for v in values):
            return sorted(values)
    except Exception:
        pass

    try:
        return sorted(values, key=lambda x: str(x).lower())
    except Exception:
        return values


def get_available_filter_values(object_list, keys=None):
    """
    Collect the possible filter values present in the provided object set.

    Parameters
    ----------
    object_list : list
        Flat or nested list of echem objects.
    keys : list, tuple, or None
        Keys to collect values for. If None, collect for all valid filter keys.

    Returns
    -------
    dict
        Example:
        {
            'gas': ['Ar', 'CO2'],
            'scan rate': [0.05, 0.1, 0.2],
            'species': ['1 mM Fc', '100 mM H2O']
        }
    """
    flat_objects = _flatten_object_list(object_list)

    opt_dict = get_sort_group_dict()

    if keys is None:
        keys = list(opt_dict.keys()) + ["replicate"]

    available = {}
    rep_num_map, rep_count_map = _build_replicate_lookup(flat_objects)

    for key in keys:
        if key == "replicate":
            values = [v for v in rep_num_map.values() if v is not None]
            values = _unique_preserve_order(values)
            values = _sort_filter_values(values)
            if any(v is not None for v in rep_count_map.values()):
                if -1 not in values:
                    values.append(-1)
            available[key] = values
            continue

        if key not in opt_dict:
            continue

        values = []
        for obj in flat_objects:
            stat_value = opt_dict[key](obj)

            if isinstance(stat_value, list):
                values.extend(stat_value)
            else:
                values.append(stat_value)

        values = _unique_preserve_order(values)
        values = _sort_filter_values(values)
        available[key] = values

    return available


def _format_filter_value_list(values, max_items=12):
    """
    Format a list of possible values for printing.
    """
    if not values:
        return "None"

    values_str = [str(v) for v in values]

    if len(values_str) <= max_items:
        return ", ".join(values_str)

    shown = ", ".join(values_str[:max_items])
    return f"{shown}, ... ({len(values_str)} total)"

def _make_hashable_filter_value(value):
    """
    Convert nested list-like values into hashable tuples for grouping.
    """
    if isinstance(value, list):
        return tuple(_make_hashable_filter_value(v) for v in value)
    if isinstance(value, tuple):
        return tuple(_make_hashable_filter_value(v) for v in value)
    return value


_COLLECTION_FILTER_KEYS = {"compounds", "concentrations", "species"}


def _is_logical_filter_request(value):
    return isinstance(value, Mapping) and any(key in value for key in ("all", "any"))


def _logical_filter_values(value):
    """
    Convert a scalar or sequence into a list of requested filter values.
    Strings remain scalar values.
    """
    if isinstance(value, (list, tuple, set, np.ndarray, pd.Index)) and not isinstance(value, str):
        return list(value)
    return [value]


def _logical_filter_clauses(value, default_logic):
    """
    Normalize implicit list behavior and explicit {'all': ...} / {'any': ...}
    requests into clauses that can be evaluated by one matcher.
    """
    if _is_logical_filter_request(value):
        unknown = set(value) - {"all", "any"}
        if unknown:
            unknown_text = ", ".join(str(key) for key in sorted(unknown))
            raise ValueError(
                "Filter logical requests only support 'all' and 'any' keys; "
                f"got: {unknown_text}."
            )

        clauses = []
        for logic in ("all", "any"):
            if logic not in value:
                continue
            values = _logical_filter_values(value[logic])
            if not values:
                raise ValueError(f"Filter logical request '{logic}' must include at least one value.")
            clauses.append((logic, values))

        if not clauses:
            raise ValueError("Filter logical request must include 'all' or 'any'.")
        return clauses

    values = _logical_filter_values(value)
    if not values:
        raise ValueError("Filter request must include at least one value.")
    return [(default_logic, values)]


def _match_logical_filter_request(value, match_one, default_logic):
    for logic, values in _logical_filter_clauses(value, default_logic):
        matches = [match_one(requested) for requested in values]
        if logic == "all" and not all(matches):
            return False
        if logic == "any" and not any(matches):
            return False
    return True


def _format_filter_value(value):
    if isinstance(value, (list, tuple, set, np.ndarray, pd.Index)) and not isinstance(value, str):
        return "[" + ", ".join(str(v) for v in list(value)) + "]"
    return str(value)


def _format_filter_criterion(key, value):
    if _is_logical_filter_request(value):
        pieces = []
        for logic in ("all", "any"):
            if logic in value:
                pieces.append(f"{key} {logic} {_format_filter_value(value[logic])}")
        return " and ".join(pieces)
    if key in _COLLECTION_FILTER_KEYS and isinstance(value, (list, tuple, set, np.ndarray, pd.Index)) and not isinstance(value, str):
        return f"{key} all {_format_filter_value(value)}"
    return f"{key}: {_format_filter_value(value)}"


def _filter_uses_collection_logic(filter_keys):
    return any(
        key in _COLLECTION_FILTER_KEYS
        and (
            _is_logical_filter_request(value)
            or (
                isinstance(value, (list, tuple, set, np.ndarray, pd.Index))
                and not isinstance(value, str)
            )
        )
        for key, value in filter_keys.items()
    )


def _best_object_time(obj):
    """
    Best available timestamp for ordering replicates.
    """
    return _object_time_for_sort(obj)


def _build_replicate_lookup(items, compare_keys=None):
    """
    Build replicate number and replicate-count lookups for the non-list objects
    in `items`.

    Replicates are defined as objects that match on all compare_keys.
    Ordering follows timestamp/creation/modification time, then input order.
    """
    flat_items = [item for item in items if not isinstance(item, list)]

    rep_num = {id(obj): None for obj in flat_items}
    rep_count = {id(obj): None for obj in flat_items}

    if len(flat_items) < 2:
        return rep_num, rep_count

    opt_dict = get_sort_group_dict()

    if compare_keys is None:
        compare_keys = [
            "subfolder",
            "type",
            "gas",
            "solvent",
            "scan rate",
            "scan window",
            "segments",
            "compounds",
            "concentrations",
        ]

    rows = []
    for i, obj in enumerate(flat_items):
        row = {
            key: _make_hashable_filter_value(opt_dict[key](obj))
            for key in compare_keys
        }
        row["__obj__"] = obj
        row["__sort_time__"] = _best_object_time(obj)
        row["__row_index__"] = i
        rows.append(row)

    df = pd.DataFrame(rows)

    if len(compare_keys) == 0:
        grouped_indices = [list(df.index)] if len(df) > 1 else []
    else:
        duplicate_mask = df[compare_keys].duplicated(keep=False)
        grouped_indices = [
            list(idx)
            for _, idx in df.loc[duplicate_mask].groupby(
                compare_keys,
                dropna=False,
                sort=False,
            ).groups.items()
        ]

    for idx_group in grouped_indices:
        idx_sorted = sorted(
            idx_group,
            key=lambda j: (
                pd.Timestamp.max
                if df.at[j, "__sort_time__"] is None
                else pd.Timestamp(df.at[j, "__sort_time__"]),
                df.at[j, "__row_index__"],
            ),
        )

        n_group = len(idx_sorted)
        if n_group <= 1:
            continue

        for rep_number, j in enumerate(idx_sorted, start=1):
            obj = df.at[j, "__obj__"]
            rep_num[id(obj)] = rep_number
            rep_count[id(obj)] = n_group

    return rep_num, rep_count

def filter(object_list, filter_keys, options=None):
    """Filter electrochemistry objects or groups using metadata keys.
    
    Parameters
    ----------
    object_list : sequence
        Flat or nested list of eCAT objects.
    filter_keys : dict
        Metadata filters to include or exclude.
    options : dict or FilterOptions, optional
        Filter logic and print options. See ``e.describe_options("filter")``.
    
    Returns
    -------
    list
        Filtered objects or groups preserving the input nesting style.
    
    Examples
    --------
    >>> ar_cvs = e.filter(cvs, {"gas": "Ar"})
    """
    options = FilterOptions.from_options(options).to_options_dict()

    if not isinstance(filter_keys, dict):
        print("\033[91mError: filter_keys must be a dictionary where keys are attributes and values are criteria.\033[0m")
        return object_list

    opt_dict = get_sort_group_dict()

    valid_keys = list(opt_dict.keys()) + ['replicate']
    if not validate_keys(filter_keys.keys(), valid_keys, "filter"):
        return object_list

    if options.get('logic') is None:
        options['logic'] = 'AND' if options.get('mode', 'include') == 'include' else 'OR'

    def _is_iterable_filter_value(value):
        return isinstance(value, (list, tuple, set, np.ndarray, pd.Index))

    def _is_numeric_scalar(value):
        return isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool)

    def _is_scan_window_pair(value):
        return (
                isinstance(value, (list, tuple, np.ndarray, pd.Index))
                and len(value) == 2
                and not any(_is_iterable_filter_value(v) for v in value)
        )

    def _is_scan_window_collection(value):
        return (
                isinstance(value, (list, tuple, np.ndarray, pd.Index))
                and len(value) > 0
                and all(_is_scan_window_pair(v) for v in value)
        )

    def _canonicalize_scan_window(window):
        a, b = [float(v) for v in window]
        lo, hi = sorted((a, b))
        return (lo, hi)

    def _scan_window_contains(stat_window, requested_value):
        lo, hi = _canonicalize_scan_window(stat_window)
        req = float(requested_value)
        return np.isclose(lo, req) or np.isclose(hi, req)

    def _scan_window_exact_match(stat_window, requested_window):
        stat_lo, stat_hi = _canonicalize_scan_window(stat_window)
        req_lo, req_hi = _canonicalize_scan_window(requested_window)
        return np.isclose(stat_lo, req_lo) and np.isclose(stat_hi, req_hi)

    def _parse_filter_concentration(value):
        if _is_numeric_scalar(value):
            return float(value)
        if isinstance(value, str):
            try:
                return concentration_to_float(value.replace(" ", ""))
            except ValueError:
                return None
        return None

    def _parse_species_filter_value(value):
        if not isinstance(value, str):
            return None

        text = value.strip()
        match = re.match(
            r"^(\d+(?:\.\d+)?)\s*([nmuμ]?M|L|%|equiv|x)\s*(.+)$",
            text,
            flags=re.IGNORECASE,
        )
        if not match:
            return {"compound": _normalize_filter_value(text), "concentration": None}

        conc = _parse_filter_concentration(match.group(1) + match.group(2))
        compound = _normalize_filter_value(match.group(3))
        return {"compound": compound, "concentration": conc}

    def _parse_species_stat_value(value):
        parsed = _parse_species_filter_value(value)
        return parsed

    def matches_single_criterion(stat_value, requested_value, key=None, replicate_total=None):
        """
        Match one criterion against one object attribute value.
        """
        if key == "replicate":
            effective_rep = 1 if stat_value is None else stat_value
            effective_total = 1 if replicate_total is None else replicate_total

            requested_values = (
                list(requested_value)
                if _is_iterable_filter_value(requested_value)
                   and not isinstance(requested_value, str)
                else [requested_value]
            )

            for req in requested_values:
                if req == -1:
                    if effective_rep == effective_total:
                        return True
                elif effective_rep == req:
                    return True
            return False

        if key == "scan window":
            if not _is_scan_window_pair(stat_value):
                return False

            # single endpoint: -1.5 means any window containing -1.5
            if _is_numeric_scalar(requested_value):
                return _scan_window_contains(stat_value, requested_value)

            # exact window: [-1.5, 0]
            if _is_scan_window_pair(requested_value):
                return _scan_window_exact_match(stat_value, requested_value)

            # OR across exact windows: [[-1.5, 0], [-1.5, 1.5]]
            if _is_scan_window_collection(requested_value):
                return any(
                    _scan_window_exact_match(stat_value, req_window)
                    for req_window in requested_value
                )

            return False

        if key == "concentrations":
            stat_values = stat_value if isinstance(stat_value, list) else [stat_value]
            stat_concs = []
            for value in stat_values:
                parsed = _parse_filter_concentration(value)
                if parsed is not None:
                    stat_concs.append(parsed)

            def concentration_matches_one(requested):
                requested_conc = _parse_filter_concentration(requested)
                if requested_conc is None:
                    return False
                return any(np.isclose(stat_conc, requested_conc) for stat_conc in stat_concs)

            return _match_logical_filter_request(
                requested_value,
                concentration_matches_one,
                default_logic="all",
            )

        if key == "species":
            stat_values = stat_value if isinstance(stat_value, list) else [stat_value]
            stat_species = [
                parsed for parsed in (_parse_species_stat_value(v) for v in stat_values)
                if parsed is not None
            ]

            def species_matches_one(requested_value):
                requested = _parse_species_filter_value(requested_value)
                if requested is None:
                    return False
                for stat in stat_species:
                    compound_matches = stat["compound"] == requested["compound"]
                    if not compound_matches:
                        continue
                    if requested["concentration"] is None:
                        return True
                    if stat["concentration"] is not None and np.isclose(
                        stat["concentration"],
                        requested["concentration"],
                    ):
                        return True
                return False

            return _match_logical_filter_request(
                requested_value,
                species_matches_one,
                default_logic="all",
            )

        requested_norm = _normalize_filter_value(requested_value)

        if isinstance(stat_value, list):
            stat_norm = _normalize_filter_value(stat_value)
            if key in _COLLECTION_FILTER_KEYS:
                def collection_matches_one(requested):
                    return _normalize_filter_value(requested) in stat_norm

                return _match_logical_filter_request(
                    requested_value,
                    collection_matches_one,
                    default_logic="all",
                )
            if isinstance(requested_norm, list):
                return all(v in stat_norm for v in requested_norm)
            return requested_norm in stat_norm

        if isinstance(stat_value, str):
            stat_norm = _normalize_filter_value(stat_value)
            if isinstance(requested_norm, list):
                return stat_norm in requested_norm
            return stat_norm == requested_norm

        if _is_iterable_filter_value(requested_value) and not isinstance(requested_value, str):
            return stat_value in requested_value

        return stat_value == requested_value

    def matches_criteria(echem_object, rep_num_map, rep_count_map):
        results = []
        for key, requested_value in filter_keys.items():
            if key == "replicate":
                stat_value = rep_num_map.get(id(echem_object))
                replicate_total = rep_count_map.get(id(echem_object))
                results.append(
                    matches_single_criterion(
                        stat_value,
                        requested_value,
                        key=key,
                        replicate_total=replicate_total,
                    )
                )
            else:
                stat_value = opt_dict[key](echem_object)
                results.append(
                    matches_single_criterion(
                        stat_value,
                        requested_value,
                        key=key,
                    )
                )

        return all(results) if options.get('logic') == 'AND' else any(results)

    def filter_nested(items):
        """
        Recursively preserve nested group structure while filtering.
        """
        filtered = []

        rep_num_map, rep_count_map = _build_replicate_lookup(items)

        for item in items:
            if isinstance(item, list):
                sub_filtered = filter_nested(item)
                if sub_filtered:
                    filtered.append(sub_filtered)
            else:
                match = matches_criteria(item, rep_num_map, rep_count_map)
                mode = options.get('mode', 'include')

                if (mode == 'include' and match) or (mode == 'exclude' and not match):
                    filtered.append(item)
        return filtered

    filtered_list = filter_nested(object_list)

    if options.get('print', False):
        logic_display = f' - {options.get("logic", "AND")}' if len(filter_keys) > 1 else ''
        print(f"=== Filtered Objects ({options['mode'].capitalize()}{logic_display}) ===")

        flat_input = _flatten_object_list(object_list)
        flat_output = _flatten_object_list(filtered_list)

        total_objects = len(flat_input)
        matched_objects = len(flat_output)

        search_criteria = f" {options.get('logic', 'AND')} ".join(
            f"({_format_filter_criterion(key, value)})" for key, value in filter_keys.items()
        )

        print(f"Filtering Criteria: {search_criteria}")
        if _filter_uses_collection_logic(filter_keys):
            print(
                "Membership filters (compounds, concentrations, species) "
                "require all listed values by default. Use {'any': [...]} for OR "
                "matching or {'all': [...]} to be explicit."
            )
        print(f"Matched Objects: {matched_objects} / {total_objects}")

        if not filtered_list:
            print("\033[91mNo objects matching your criteria were found.\033[0m")

            print("Possible values in this object set for the requested key(s):")
            available = get_available_filter_values(object_list, keys=filter_keys.keys())
            for key, values in available.items():
                print(f"  {key}: {_format_filter_value_list(values)}")

            print("All valid filter keys:")
            print("  " + ", ".join(valid_keys))

        elif isinstance(object_list[0], list):
            show_groups(filtered_list, options)
        else:
            show_objects(filtered_list, options)

    return filtered_list


def sort(object_list, sort_keys, options=None):
    """Sort electrochemistry objects by metadata keys.
    
    Parameters
    ----------
    object_list : list
        Objects to sort.
    sort_keys : str or sequence of str
        Sort/group keys such as ``"timestamp"`` or ``"scan rate"``.
    options : dict or SortGroupOptions, optional
        Print and table-display options. See ``e.describe_options("sort_group")``.
    
    Returns
    -------
    list
        Sorted list of objects.
    
    Examples
    --------
    >>> sorted_cvs = e.sort(cvs, ["gas", "timestamp"])
    """
    options = SortGroupOptions.from_options(options).to_options_dict()

    if isinstance(sort_keys, str):
        sort_keys = [sort_keys]

    opt_dict = get_sort_group_dict()

    if not validate_keys(sort_keys, opt_dict, "sort"):
        return object_list  # Return unsorted list if keys are invalid

    def sortable_value(value):
        if value is None:
            return (9, "")
        if isinstance(value, datetime):
            return (0, pd.Timestamp(value))
        if isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool):
            if pd.isna(value):
                return (9, "")
            return (1, float(value))
        if isinstance(value, (list, tuple)):
            return (2, tuple(sortable_value(item) for item in value))
        if isinstance(value, dict):
            return (3, tuple((str(k).lower(), sortable_value(v)) for k, v in sorted(value.items())))
        return (4, str(value).lower())

    def safe_key(f, default=None):
        return lambda x: sortable_value(f(x) if f(x) is not None else default)

    for skey in reversed(sort_keys):
        object_list.sort(key=safe_key(opt_dict[skey]))

    if options.get('print', True):
        print("\n=== Sorted Objects ===")
        show_objects(object_list, options)

    return object_list


def group(object_list, group_keys, options=None):
    """Group electrochemistry objects by metadata keys.
    
    Parameters
    ----------
    object_list : sequence
        Objects to group.
    group_keys : str or sequence of str
        Sort/group keys used to form groups.
    options : dict or SortGroupOptions, optional
        Print and table-display options. See ``e.describe_options("sort_group")``.
    
    Returns
    -------
    list
        Nested list of grouped objects.
    
    Examples
    --------
    >>> groups = e.group(cvs, ["gas", "compounds"])
    """
    options = SortGroupOptions.from_options(options).to_options_dict()

    opt_dict = get_sort_group_dict()

    if isinstance(group_keys, str):
        group_keys = [group_keys]

    if not validate_keys(group_keys, opt_dict, "group"):
        return object_list

    def get_group_stats(echem_object):
        return {key: opt_dict[key](echem_object) for key in group_keys}

    def make_hashable(value):
        if isinstance(value, list):
            return tuple(make_hashable(v) for v in value)
        if isinstance(value, dict):
            return tuple((k, make_hashable(v)) for k, v in sorted(value.items()))
        return value

    def group_key(echem_object):
        stats = get_group_stats(echem_object)
        return tuple((k, make_hashable(stats[k])) for k in group_keys)

    grouped_dict = {}
    for echem_object in object_list:
        key = group_key(echem_object)
        grouped_dict.setdefault(key, []).append(echem_object)

    groups = list(grouped_dict.values())

    if options.get('print', True):
        print(f"\n=== {len(groups)} Groups Created ===\n")
        show_groups(groups, options)

    return groups


def _format_group_summary_value(value, options=None):
    if options is None:
        options = {}

    sig_figs = options.get("sig figs", 3)

    if value is None:
        return ""

    if isinstance(value, float) and pd.isna(value):
        return ""

    if isinstance(value, (np.integer, int)) and not isinstance(value, bool):
        return str(value)

    if isinstance(value, (np.floating, float)) and not isinstance(value, bool):
        return f"{round_sigfigs(float(value), sig_figs):g}"

    if isinstance(value, (list, tuple, set)):
        return "[" + ", ".join(_format_group_summary_value(v, options) for v in value) + "]"

    if isinstance(value, dict):
        return ", ".join(
            f"{k}: {_format_group_summary_value(v, options)}"
            for k, v in value.items()
        )

    return str(value)


def _unique_group_summary_values(values):
    unique = []
    seen = set()

    for value in values:
        key = _make_hashable_filter_value(value)
        if key in seen:
            continue
        seen.add(key)
        unique.append(value)

    return unique


def _format_group_summary_values(values, options=None):
    unique_values = _unique_group_summary_values(values)

    if len(unique_values) == 0:
        return ""

    if len(unique_values) == 1:
        return _format_group_summary_value(unique_values[0], options)

    return "[" + ", ".join(
        _format_group_summary_value(value, options)
        for value in unique_values
    ) + "]"


def _is_grouped_object_list(object_list):
    return isinstance(object_list, list) and any(isinstance(item, list) for item in object_list)


def group_summary(object_list, group_keys=None, options=None):
    """Summarize metadata shared within flat or nested eCAT object groups.
    
    Parameters
    ----------
    object_list : sequence
        Flat or nested list of eCAT objects.
    group_keys : sequence of str, optional
        Keys that define the groups being summarized.
    options : dict or GroupSummaryOptions, optional
        Summary-column and display options. See ``e.describe_options("group_summary")``.
    
    Returns
    -------
    pandas.DataFrame
        Summary table for the supplied objects or groups.
    
    Examples
    --------
    >>> summary = e.group_summary(groups, group_keys=["gas"])
    """
    options = GroupSummaryOptions.from_options(options).to_options_dict()

    if object_list is None:
        object_list = []

    opt_dict = get_sort_group_dict()

    resolved_group_keys = group_keys
    if resolved_group_keys is None:
        resolved_group_keys = options.get("group keys")

    if isinstance(resolved_group_keys, str):
        resolved_group_keys = [resolved_group_keys]

    requested_columns = _coerce_display_columns(options.get("columns", []))

    if requested_columns and not validate_keys(requested_columns, opt_dict, "summary column"):
        raise ValueError(
            "Invalid summary column(s): "
            + ", ".join([col for col in requested_columns if col not in opt_dict])
            + "\nAvailable columns: "
            + ", ".join(opt_dict.keys())
        )

    group_key_values = None
    if resolved_group_keys is not None:
        if not validate_keys(resolved_group_keys, opt_dict, "group"):
            return pd.DataFrame()

        flat_objects = _flatten_object_list(object_list)
        groups = group(flat_objects, resolved_group_keys, options={"print": False})
        group_key_values = [
            {
                key: opt_dict[key](grp[0]) if len(grp) > 0 else None
                for key in resolved_group_keys
            }
            for grp in groups
        ]
    elif _is_grouped_object_list(object_list):
        groups = object_list
    else:
        groups = [_flatten_object_list(object_list)]

    auto_candidate_keys = [key for key in opt_dict if key not in (resolved_group_keys or [])]

    rows = []
    type_columns = set()
    summary_columns = set()

    for group_index, grp in enumerate(groups):
        row = {}

        if group_key_values is not None:
            for key, value in group_key_values[group_index].items():
                row[key] = _format_group_summary_value(value, options)
        elif _is_grouped_object_list(object_list):
            row["group"] = group_index

        row["n objects"] = len(grp)

        type_counts = {}
        for obj in grp:
            exp_type = _exp_type_short(getattr(obj, "type", ""))
            if exp_type in {"CV", "CA", "CP", "DPV"}:
                type_counts[exp_type] = type_counts.get(exp_type, 0) + 1

        for exp_type, count in type_counts.items():
            key = f"n {exp_type}"
            row[key] = count
            type_columns.add(key)

        columns_for_group = set(requested_columns)
        for key in auto_candidate_keys:
            values = [opt_dict[key](obj) for obj in grp]
            if len(_unique_group_summary_values(values)) > 1:
                columns_for_group.add(key)

        for key in opt_dict:
            if key not in columns_for_group:
                continue

            values = [opt_dict[key](obj) for obj in grp]
            n_key = f"n {key}"
            values_key = f"{key} values"
            row[n_key] = len(_unique_group_summary_values(values))
            row[values_key] = _format_group_summary_values(values, options)
            summary_columns.update({n_key, values_key})

        rows.append(row)

    leading_columns = []
    if group_key_values is not None:
        leading_columns.extend(resolved_group_keys)
    elif _is_grouped_object_list(object_list):
        leading_columns.append("group")
    leading_columns.append("n objects")

    type_order = ["n CV", "n CA", "n CP", "n DPV"]
    ordered_columns = leading_columns + [
        col for col in type_order if col in type_columns
    ]

    for key in opt_dict:
        for col in (f"n {key}", f"{key} values"):
            if col in summary_columns and col not in ordered_columns:
                ordered_columns.append(col)

    for row in rows:
        for col in ordered_columns:
            row.setdefault(col, 0 if col in type_columns else "")

    summary = pd.DataFrame(rows)
    if len(summary) > 0:
        summary = summary[ordered_columns]
    summary = summary.rename(columns={c: pretty_table_column_label(c) for c in summary.columns})

    if options.get("print", True):
        if options.get("pretty print", True):
            display_object_table(summary, options)
        else:
            print(summary.to_string(index=False))

    return summary


def sort_and_group(object_list, sort_keys=None, group_keys=None, options=None):
    """Sort electrochemistry objects and then group them by metadata keys.
    
    Parameters
    ----------
    object_list : sequence
        Objects to sort and group.
    sort_keys : str or sequence of str, optional
        Keys used for sorting before grouping.
    group_keys : str or sequence of str, optional
        Keys used to form groups.
    options : dict or SortGroupOptions, optional
        Print and table-display options. See ``e.describe_options("sort_group")``.
    
    Returns
    -------
    list
        Grouped objects in sorted order.
    
    Examples
    --------
    >>> groups = e.sort_and_group(cvs, sort_keys=["timestamp"], group_keys=["gas"])
    """
    options = SortGroupOptions.from_options(options).to_options_dict()

    if sort_keys is None and group_keys is None:
        print("Provide at least one sort key or group key!")
        return object_list
    if sort_keys is None:
        sort_keys = group_keys
    if group_keys is None:
        group_keys = sort_keys
    if isinstance(sort_keys, str):
        sort_keys = [sort_keys]
    if isinstance(group_keys, str):
        group_keys = [group_keys]

    sort_options = options.copy()
    sort_options["print"] = False
    sorted_objects = sort(object_list.copy(), sort_keys, options=sort_options)
    grouped_objects = group(sorted_objects, group_keys, options)

    return grouped_objects



__all__ = [
    'filter',
    'sort',
    'group',
    'sort_and_group',
    'group_summary',
    'get_available_filter_values',
    'get_sort_group_dict',
]
