"""Canonical keys for weighted frequency dicts (supports non-integer numeric categories)."""

from __future__ import annotations

import math
from typing import Any, Iterable

import numpy as np
import pandas as pd
from fastapi import HTTPException


def normalize_category_value(v: Any) -> Any:
    """
    Canonical category value for grouping and display.

    Maps equivalent representations to one hashable value so mixed-type
    columns (e.g. int 1 and str "1") aggregate under a single key.
    """
    if v is None or (isinstance(v, float) and math.isnan(v)) or pd.isna(v):
        return v

    if isinstance(v, bool):
        return v

    num = pd.to_numeric(v, errors="coerce")
    if pd.notna(num):
        fv = float(num)
        if math.isnan(fv):
            return v
        if fv == 0.0:
            fv = 0.0
        if fv == int(fv) and abs(fv) < 2**53:
            return int(fv)
        return fv

    return str(v)


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
    """Merge value counts that differ only by type (e.g. 1 vs \"1\")."""
    merged: dict[Any, int] = {}
    for key, freq in counts.items():
        canonical = normalize_category_value(key)
        merged[canonical] = merged.get(canonical, 0) + int(freq)
    return merged


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
