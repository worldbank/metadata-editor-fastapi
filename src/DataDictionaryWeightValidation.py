"""Validate dataframe columns before weighted mean/std (DescrStatsW); no dtype coercion."""

import pandas as pd
from pandas.api.types import is_numeric_dtype
from fastapi import HTTPException


def _dtype_label(series: pd.Series) -> str:
    return str(series.dtype)


def _first_value_preview(series: pd.Series, max_len: int = 120) -> str:
    if is_numeric_dtype(series):
        return ""
    non_null = series.dropna()
    if len(non_null) == 0:
        return " Column has no non-null values."
    v = non_null.iloc[0]
    s = repr(v)
    if len(s) > max_len:
        s = s[: max_len - 3] + "..."
    return f" First non-null value (preview): {s}"


def validate_weight_columns_for_descr_stats(df: pd.DataFrame, field: str, weight_field: str) -> None:
    """
    Require numeric dtypes for both the analysis column and the weight column.
    Does not modify the dataframe. Raises HTTPException 400 with field names and dtypes.
    """
    checks = (
        (field, "analysis", "field"),
        (weight_field, "weight", "weight_field"),
    )
    for col, role, param_name in checks:
        if col not in df.columns:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Weighted statistics: column '{col}' not found in data "
                    f"(expected as {param_name} for {role} variable)."
                ),
            )
        s = df[col]
        if not is_numeric_dtype(s):
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Weighted mean, standard deviation, and weighted frequencies require numeric dtypes "
                    f"(values are not coerced automatically). The {role} column '{col}' "
                    f"has dtype '{_dtype_label(s)}', which is not numeric.{_first_value_preview(s)}"
                ),
            )
