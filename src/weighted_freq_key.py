"""Canonical keys for weighted frequency dicts (supports non-integer numeric categories)."""

from __future__ import annotations

import math
import re
from typing import Any, Iterable

import numpy as np
import pandas as pd
from fastapi import HTTPException

# Same shape as PHP Variable_service::value_key — no leading-zero codes.
_VALUE_KEY_NUM = re.compile(r"^-?(0|[1-9]\d*)(\.\d+)?$")


def normalize_category_value(v: Any) -> Any:
    """
    Canonical category key as a string (editor / PHP value_key).

    int 1, float 1.0, and str "1" all become "1" so mixed pandas dtypes join.
    Padded codes ("01", "011", "000223") stay as themselves — never rewritten
    as a different number. Matching is always by this string, never by int.
    """
    if v is None:
        return ""

    if isinstance(v, bool):
        return v

    if isinstance(v, float) and math.isnan(v):
        return v

    # Avoid pd.isna on arrays; scalars only. Skip bool (already handled).
    if not isinstance(v, (str, bytes)) and not isinstance(v, (int, float)):
        try:
            if pd.isna(v):
                return v
        except (ValueError, TypeError):
            pass

    s = str(v).strip()
    if _VALUE_KEY_NUM.match(s):
        if "." in s:
            s = s.rstrip("0").rstrip(".")
            if s in ("", "-"):
                s = "0"
            if s == "-0":
                s = "0"
    return s


def category_sort_key(v: Any) -> tuple:
    """Sort key for category values; avoids str/int comparison errors."""
    if v is None or (isinstance(v, float) and math.isnan(v)) or pd.isna(v):
        return (2, "")

    if isinstance(v, bool):
        return (0, float(int(v)))

    num = pd.to_numeric(v, errors="coerce")
    if pd.notna(num):
        return (0, float(num))

    return (1, str(v))


def sort_category_items(items: Iterable[tuple[Any, Any]]) -> list[tuple[Any, Any]]:
    """Sort (category, frequency) pairs with a mixed-type-safe key."""
    return sorted(items, key=lambda item: category_sort_key(item[0]))


def merge_category_value_counts(counts: pd.Series | dict) -> dict[Any, int]:
    """Merge value counts that differ only by type (e.g. 1 vs \"1\" → \"1\")."""
    merged: dict[Any, int] = {}
    for key, freq in counts.items():
        canonical = normalize_category_value(key)
        merged[canonical] = merged.get(canonical, 0) + int(freq)
    return merged


def distinct_category_values(counts: pd.Series | dict) -> set:
    """
    The distinct values merge_category_value_counts would produce, without the counts.

    Uses keys() because iterating a Series yields its values, not its index.
    """
    return {normalize_category_value(key) for key in counts.keys()}


def weighted_freq_category_key(v) -> str:
    """
    Stable string key so groupby values and var_catgry string labels match after
    numeric normalization (e.g. 1, 1.0, "1.0" → same key; 1.3 and "1.3" match).
    Non-numeric categories use an explicit prefix.
    """
    if v is None or (isinstance(v, float) and math.isnan(v)) or pd.isna(v):
        raise HTTPException(
            status_code=400,
            detail="Weighted statistics: cannot use a missing value as a category key.",
        )

    if isinstance(v, bool):
        return f"b:{int(v)}"

    num = pd.to_numeric(v, errors="coerce")
    if pd.notna(num):
        fv = float(num)
        if math.isnan(fv):
            raise HTTPException(
                status_code=400,
                detail=f"Weighted statistics: invalid numeric category value {v!r}.",
            )
        if fv == 0.0:
            fv = 0.0
        return format(fv, ".15g")

    return f"s:{v!s}"
