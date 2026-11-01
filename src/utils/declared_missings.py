"""Declared (user) missing values: discrete codes and SPSS-style ranges.

pyreadstat reports Stata user missings in ``meta.missing_user_values`` but SPSS ones
only in ``meta.missing_ranges``, where a discrete code is a one-value range
(``{"lo": 8.0, "hi": 8.0}``). The data dictionary works with one map per file::

    {column: [8, 9, {"lo": -9, "hi": -1}]}

Plain entries are discrete codes; dicts are inclusive ranges. Callers may also send
ranges as ``{"min": .., "max": ..}`` (the DDI ``<range>`` spelling).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _range_bounds(entry):
    """(lo, hi) for a range entry, or None when the entry is a discrete code."""
    if not isinstance(entry, dict):
        return None
    lo = entry.get("lo", entry.get("min"))
    hi = entry.get("hi", entry.get("max"))
    if lo is None and hi is None:
        return None
    return lo, hi


def _declared(values, ranges) -> list:
    entries = []
    if values is not None:
        entries = list(values) if isinstance(values, (list, tuple)) else [values]
    for r in ranges or []:
        lo, hi = r.get("lo"), r.get("hi")
        entry = lo if lo == hi else {"lo": lo, "hi": hi}
        if entry not in entries:
            entries.append(entry)
    return entries


def column_declared_missings(meta, col) -> list:
    """Missing values a Stata or SPSS file declares for one column."""
    values = (getattr(meta, "missing_user_values", None) or {}).get(col)
    ranges = (getattr(meta, "missing_ranges", None) or {}).get(col)
    return _declared(values, ranges)


def file_declared_missings(meta) -> dict:
    """Missing values a Stata or SPSS file declares, in the shared map shape."""
    cols = set((getattr(meta, "missing_user_values", None) or {}).keys())
    cols |= set((getattr(meta, "missing_ranges", None) or {}).keys())
    out = {}
    for col in cols:
        entries = column_declared_missings(meta, col)
        if entries:
            out[col] = entries
    return out


def split_missings(values):
    """(discrete codes, [(lo, hi), ...]) for one column's list."""
    discrete, ranges = [], []
    if values is None:
        return discrete, ranges
    if not isinstance(values, (list, tuple)):
        values = [values]
    for v in values:
        bounds = _range_bounds(v)
        if bounds is None:
            discrete.append(v)
        else:
            ranges.append(bounds)
    return discrete, ranges


def discrete_missings_map(missings) -> dict:
    """The map without range entries, for code that matches values one by one."""
    out = {}
    for col, values in (missings or {}).items():
        discrete, _ = split_missings(values)
        if discrete:
            out[col] = discrete
    return out


def missing_ranges_map(missings) -> dict:
    """{column: [(lo, hi), ...]} for columns that declare ranges."""
    out = {}
    for col, values in (missings or {}).items():
        _, ranges = split_missings(values)
        if ranges:
            out[col] = ranges
    return out


def apply_missing_ranges(df: pd.DataFrame, ranges_map: dict) -> pd.DataFrame:
    """Set values inside any declared range to NaN (numeric comparison, inclusive)."""
    for col, ranges in (ranges_map or {}).items():
        if col not in df.columns:
            continue
        numeric = pd.to_numeric(df[col], errors="coerce")
        mask = pd.Series(False, index=df.index)
        for lo, hi in ranges:
            cond = numeric.notna()
            if lo is not None:
                cond &= numeric >= float(lo)
            if hi is not None:
                cond &= numeric <= float(hi)
            mask |= cond
        if mask.any():
            df.loc[mask, col] = np.nan
    return df


def invalrng(values) -> dict:
    """``var_invalrng`` for one column: discrete codes in ``values``, ranges in ``ranges``."""
    discrete, ranges = split_missings(values)
    out = {"values": discrete}
    if ranges:
        out["ranges"] = [{"min": lo, "max": hi} for lo, hi in ranges]
    return out


class MissingTally:
    """
    Counts what the data dictionary turns into missing, per column:

    - sysmiss: values already missing in the file (Stata ".", SPSS sysmis, empty CSV cell)
    - codes:   values blanked because they are declared missing (codes, ranges, Stata
               letters), keyed by value, so they can be reported as frequency rows.

    Call add() with the same rows before and after missing values are applied. Works on
    whole frames and on chunks (counts accumulate).
    """

    def __init__(self):
        self.sysmiss = {}
        self.codes = {}

    def add(self, before: pd.DataFrame, after: pd.DataFrame) -> None:
        for col in before.columns:
            if col not in after.columns:
                continue
            b = before[col]
            a = after[col]
            self.sysmiss[col] = self.sysmiss.get(col, 0) + int(b.isna().sum())
            mask = b.notna().to_numpy() & a.isna().to_numpy()
            if mask.any():
                counts = self.codes.setdefault(col, {})
                for value, n in b[mask].value_counts().items():
                    counts[value] = counts.get(value, 0) + int(n)

    def apply(self, variables: list, value_labels: dict) -> None:
        """
        Add `sysmiss_count` to every variable and, where a frequency list was computed,
        rows for the declared missing values (flagged is_missing, labelled if the file
        labels them).
        """
        from src.weighted_freq_key import merge_category_value_counts, sort_category_items
        from src.utils.categorical import attach_labels

        for var in variables:
            name = var.get("name")
            var["sysmiss_count"] = int(self.sysmiss.get(name, 0))
            if var.get("frequencies_skipped_reason") is not None:
                continue
            counts = self.codes.get(name)
            if not counts:
                continue
            existing = {str(c.get("value")) for c in (var.get("var_catgry") or [])}
            rows = []
            for value, n in sort_category_items(merge_category_value_counts(counts).items()):
                if str(value) in existing:
                    continue
                rows.append({
                    "value": str(value),
                    "stats": [{"type": "freq", "value": str(n)}],
                    "is_missing": 1,
                })
            attach_labels(rows, (value_labels or {}).get(name, {}))
            var["var_catgry"] = list(var.get("var_catgry") or []) + rows
