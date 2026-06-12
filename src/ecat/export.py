"""Export helpers for processed data and figures."""

import os
import re

import numpy as np
import pandas as pd

from .utils import _format_path_for_display


def save_data(object_list, options={}):
    """
    Export electrochem data with two header rows:
      - Row 1: object/group 'name'
      - Row 2: column name with units in parentheses

    Units policy:
      - Source units are read ONLY from e.units.get(<column_name>).
      - If options request a target unit, values are converted using the
        object's ``scale_axis`` method or the module-level eCAT fallback.
      - If no target is requested, values are kept as-is and labeled with the
        source unit when available.
    """
    debug = bool(options.get("debug", False))
    folder_path = options.get("folder path", ".")
    file_name = options.get("file name", "output")
    output_path = os.path.join(folder_path, f"{file_name}.csv")

    if not object_list:
        raise ValueError("object_list is empty")

    def dprint(*args):
        if debug:
            print(*args)

    paren_re = re.compile(r"\s*\(.*\)\s*$")

    def with_unit(colname: str, unit: str | None) -> str:
        name = str(colname)
        if not unit or paren_re.search(name):
            return name
        return f"{name} ({unit})"

    def norm_units_dict(d):
        return {str(k): (None if v is None else str(v)) for k, v in d.items()} if isinstance(d, dict) else {}

    options_units = norm_units_dict(options.get("units", {}))
    global_x_unit = options.get("x unit", None)
    global_y_unit = options.get("y unit", None)

    def as_df(obj) -> pd.DataFrame:
        if isinstance(obj, pd.DataFrame):
            return obj.reset_index(drop=True)
        if isinstance(obj, pd.Series):
            return obj.to_frame().reset_index(drop=True)
        arr = np.asarray(obj)
        if arr.ndim == 1:
            return pd.DataFrame(arr).reset_index(drop=True)
        return pd.DataFrame(arr).reset_index(drop=True)

    def get_raw_x_series(e) -> pd.Series:
        try:
            x_raw = e.x({"one column": False})
        except TypeError:
            x_raw = e.x()
        X = as_df(x_raw)
        if X.shape[1] > 1:
            X = X.iloc[:, [0]]
        name = X.columns[0] if X.columns[0] is not None and not str(X.columns[0]).startswith("Unnamed") else "x"
        s = X.iloc[:, 0].reset_index(drop=True)
        s.name = str(name)
        return s

    def same_series(a: pd.Series, b: pd.Series, rtol=1e-10, atol=1e-12) -> bool:
        av = np.asarray(a).reshape(-1)
        bv = np.asarray(b).reshape(-1)
        if av.shape != bv.shape:
            return False
        if np.array_equal(av, bv, equal_nan=True):
            return True
        try:
            return np.allclose(av.astype(float), bv.astype(float), rtol=rtol, atol=atol, equal_nan=True)
        except Exception:
            return False

    def select_target_unit(colname: str, is_x: bool) -> str | None:
        if colname in options_units and options_units[colname] is not None:
            return options_units[colname]
        return global_x_unit if is_x else global_y_unit

    def scale_fn_for(e):
        fn = getattr(e, "scale_axis", None)
        if callable(fn):
            return fn
        from .utils import scale_axis

        return scale_axis

    def scale_and_label_series(e, s: pd.Series, is_x: bool) -> pd.DataFrame:
        col = s.name
        units_dict = getattr(e, "units", {}) or {}
        current_unit = units_dict.get(col)
        target_unit = select_target_unit(col, is_x)

        fn = scale_fn_for(e)
        if target_unit is not None:
            scale, out_unit = fn(s.values, col, current_unit, target_unit)
            out_unit = out_unit if out_unit is not None else current_unit
            values = s.astype(float) * float(scale)
            label = with_unit(col, out_unit)
        else:
            values = s
            label = with_unit(col, current_unit)

        return pd.DataFrame({label: values})

    def scale_and_label_frame(e, Y: pd.DataFrame, is_x: bool = False) -> pd.DataFrame:
        parts = []
        for c in Y.columns:
            col_series = pd.Series(Y[c].values, name=str(c))
            parts.append(scale_and_label_series(e, col_series, is_x=is_x))
        return pd.concat(parts, axis=1)

    def is_grouped(seq) -> bool:
        return isinstance(seq[0], (list, tuple))

    echem_object_count = len(object_list)
    grouped = is_grouped(object_list)
    dprint(f"Detected grouped={grouped}")

    if grouped:
        frames = []
        for g_idx, group in enumerate(object_list):
            if not group:
                continue
            echem_object_count += len(group) - 1

            raw_xs = [get_raw_x_series(e) for e in group]
            names = [getattr(e, "name", "Unnamed") for e in group]

            Ys_raw = []
            for e in group:
                Y = as_df(e.y(options))
                Y.columns = [
                    str(c) if c is not None and not str(c).startswith("Unnamed") else f"y{i + 1}"
                    for i, c in enumerate(Y.columns)
                ]
                Ys_raw.append(Y)

            max_len = max(max(len(s) for s in raw_xs), max(y.shape[0] for y in Ys_raw))
            raw_xs = [s.reindex(range(max_len)) for s in raw_xs]
            Ys_raw = [y.reindex(range(max_len)) for y in Ys_raw]

            x_all_same = all(same_series(raw_xs[0], s) for s in raw_xs[1:])
            dprint(f"[Group {g_idx}] names={names}, x_all_same={x_all_same}")

            if x_all_same:
                e0 = group[0]
                X_shared = scale_and_label_series(e0, raw_xs[0], is_x=True)
                y_blocks = [scale_and_label_frame(e, Y) for e, Y in zip(group, Ys_raw)]
                parts = [X_shared] + y_blocks
                block = pd.concat(parts, axis=1)

                top_labels = [""] + [nm for nm, Y in zip(names, y_blocks) for _ in range(Y.shape[1])]
                bottom_labels = [X_shared.columns[0]] + [c for Y in y_blocks for c in Y.columns]
                block.columns = pd.MultiIndex.from_tuples(list(zip(top_labels, bottom_labels)))
            else:
                subs = []
                for nm, e, sX, Y in zip(names, group, raw_xs, Ys_raw):
                    X_conv = scale_and_label_series(e, sX, is_x=True)
                    Y_conv = scale_and_label_frame(e, Y)
                    sub = pd.concat([X_conv, Y_conv], axis=1)
                    sub.columns = pd.MultiIndex.from_product([[nm], list(sub.columns)])
                    subs.append(sub)
                block = pd.concat(subs, axis=1)

            frames.append(block)

        all_data = pd.concat(frames, axis=1)

    else:
        blocks = []
        for i, e in enumerate(object_list):
            nm = getattr(e, "name", "Unnamed")

            if hasattr(e, "data") and isinstance(e.data, (pd.DataFrame, pd.Series)):
                df = as_df(e.data)
                df.columns = [
                    str(c) if c is not None and not str(c).startswith("Unnamed") else f"c{i + 1}"
                    for i, c in enumerate(df.columns)
                ]
                df_conv = scale_and_label_frame(e, df, is_x=False)
                df_conv.columns = pd.MultiIndex.from_product([[nm], list(df_conv.columns)])
                dprint(f"[Ungrouped {i}] using .data with shape {df.shape}")
            else:
                sX = get_raw_x_series(e)
                Y = as_df(e.y(options))
                Y.columns = [
                    str(c) if c is not None and not str(c).startswith("Unnamed") else f"y{i + 1}"
                    for i, c in enumerate(Y.columns)
                ]

                max_len = max(len(sX), Y.shape[0])
                sX = sX.reindex(range(max_len))
                Y = Y.reindex(range(max_len))

                X_conv = scale_and_label_series(e, sX, is_x=True)
                Y_conv = scale_and_label_frame(e, Y)
                df_conv = pd.concat([X_conv, Y_conv], axis=1)
                df_conv.columns = pd.MultiIndex.from_product([[nm], list(df_conv.columns)])
                dprint(f"[Ungrouped {i}] built x|y with shapes X={X_conv.shape}, Y={Y_conv.shape}")

            blocks.append(df_conv)

        all_data = pd.concat(blocks, axis=1)

    os.makedirs(folder_path, exist_ok=True)
    all_data.to_csv(output_path, index=False)
    abs_path = os.path.abspath(output_path)
    print(
        "Saved "
        f"{echem_object_count} echem objects to:\n"
        f"{_format_path_for_display(abs_path)}"
    )
    return all_data

__all__ = ["save_data"]
