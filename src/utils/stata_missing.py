"""Detect and replace Stata extended missing values (.a through .z)."""

from __future__ import annotations

import numpy as np
import pandas as pd

STATA_EXTENDED_MISSING_LETTERS = frozenset(chr(i) for i in range(ord("a"), ord("z") + 1))


def _is_numeric_value(value) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return False
    if isinstance(value, (int, float, np.integer, np.floating)):
        return not pd.isna(value)
    return pd.notna(pd.to_numeric(value, errors="coerce"))


def stata_extended_missing_values(series: pd.Series) -> list[str]:
    """
    Return single-letter a-z values that look like Stata extended missings.

    pyreadstat can surface .a-.z as bare letters in object columns that also
    contain numeric values. Pure string categoricals are left unchanged.
    """
    if series.dtype != object and not pd.api.types.is_string_dtype(series):
        return []

    non_null = series.dropna()
    if non_null.empty:
        return []

    letters_present: list[str] = []
    has_numeric = False
    for value in non_null.unique():
        if (
            isinstance(value, str)
            and len(value) == 1
            and value in STATA_EXTENDED_MISSING_LETTERS
        ):
            letters_present.append(value)
        elif _is_numeric_value(value):
            has_numeric = True

    return letters_present if has_numeric and letters_present else []


def replace_stata_extended_missings(
    series: pd.Series, user_missings: list | None = None
) -> pd.Series:
    """Replace declared and inferred Stata extended missings with NaN."""
    out = series.replace(user_missings or [], np.nan)
    extra = stata_extended_missing_values(out)
    if extra:
        out = out.where(~out.isin(extra), np.nan)
    return out
