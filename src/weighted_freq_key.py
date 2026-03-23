"""Canonical keys for weighted frequency dicts (supports non-integer numeric categories)."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from fastapi import HTTPException


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
