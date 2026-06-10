"""Load CSV files for export to Stata/SPSS with optional memory-optimized dtypes."""

from __future__ import annotations

import logging
import os
from types import SimpleNamespace

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_ENCODINGS = [
    None,
    "utf-8",
    "latin1",
    "cp1252",
    "iso-8859-1",
    "cp850",
]

CSV_EXPORT_OPTIMIZE_MIN_BYTES = int(
    os.environ.get("CSV_EXPORT_OPTIMIZE_MIN_BYTES", str(500 * 1024 * 1024))
)
CSV_EXPORT_SAMPLE_ROWS = int(os.environ.get("CSV_EXPORT_SAMPLE_ROWS", "10000"))
CSV_EXPORT_CATEGORY_MAX_CARDINALITY = int(
    os.environ.get("CSV_EXPORT_CATEGORY_MAX_CARDINALITY", "256")
)


def _make_csv_meta(df: pd.DataFrame) -> SimpleNamespace:
    meta = SimpleNamespace()
    meta.column_names = df.columns.tolist()
    meta.column_names_to_labels = {}
    meta.number_rows = df.shape[0]
    meta.number_columns = df.shape[1]
    meta.variable_value_labels = {}
    meta.dtypes = df.dtypes.to_dict()
    return meta


def _read_csv_with_encodings(
    file_path: str,
    *,
    usecols: list[str] | None = None,
    dtype: dict | str | None = None,
    nrows: int | None = None,
) -> tuple[pd.DataFrame, str | None]:
    last_error: Exception | None = None
    for encoding in DEFAULT_ENCODINGS:
        try:
            read_kwargs: dict = {
                "filepath_or_buffer": file_path,
                "usecols": usecols,
                "dtype": dtype,
                "low_memory": False,
            }
            if nrows is not None:
                read_kwargs["nrows"] = nrows
            if encoding is not None:
                read_kwargs["encoding"] = encoding
            df = pd.read_csv(**read_kwargs)
            return df, encoding
        except UnicodeDecodeError as exc:
            last_error = exc
            logger.debug(
                "Failed to read CSV with encoding %r: %s", encoding, exc
            )
        except Exception as exc:
            last_error = exc
            logger.debug(
                "Failed to read CSV with encoding %r: %s", encoding, exc
            )
    raise Exception(
        f"Failed to read CSV file with any encoding. Last error: {last_error}"
    )


def load_csv_legacy(
    file_path: str,
    usecols: list[str] | None = None,
    dtypes: dict | None = None,
) -> tuple[pd.DataFrame, SimpleNamespace]:
    """Load CSV using the original single-pass read (unchanged behavior)."""
    df, encoding = _read_csv_with_encodings(
        file_path, usecols=usecols, dtype=dtypes
    )
    logger.debug(
        "CSV legacy load succeeded with encoding %r, shape: %s",
        encoding,
        df.shape,
    )
    return df, _make_csv_meta(df)


def should_use_optimized_csv_load(
    file_path: str,
    user_dtypes: dict | None = None,
) -> bool:
    """Return True when the file is large enough to use optimized dtype loading."""
    if CSV_EXPORT_OPTIMIZE_MIN_BYTES <= 0:
        return False
    if not os.path.isfile(file_path):
        return False
    file_size = os.path.getsize(file_path)
    if file_size < CSV_EXPORT_OPTIMIZE_MIN_BYTES:
        logger.debug(
            "Using legacy CSV load for %s: size %s < %s",
            file_path,
            file_size,
            CSV_EXPORT_OPTIMIZE_MIN_BYTES,
        )
        return False
    logger.debug(
        "Using optimized CSV load for %s: size %s >= %s",
        file_path,
        file_size,
        CSV_EXPORT_OPTIMIZE_MIN_BYTES,
    )
    return True


def _nullable_int_dtype(min_val: float, max_val: float) -> str:
    if min_val >= np.iinfo(np.int8).min and max_val <= np.iinfo(np.int8).max:
        return "Int8"
    if min_val >= np.iinfo(np.int16).min and max_val <= np.iinfo(np.int16).max:
        return "Int16"
    if min_val >= np.iinfo(np.int32).min and max_val <= np.iinfo(np.int32).max:
        return "Int32"
    return "Int64"


def infer_csv_dtypes(
    sample_df: pd.DataFrame,
    *,
    missings_cols: set[str] | None = None,
    user_dtypes: dict | None = None,
) -> dict[str, str | type]:
    """Infer memory-efficient dtypes from a CSV sample."""
    missings_cols = missings_cols or set()
    user_dtypes = user_dtypes or {}
    inferred: dict[str, str | type] = {}

    for col in sample_df.columns:
        if col in user_dtypes:
            inferred[col] = user_dtypes[col]
            continue
        if col in missings_cols:
            inferred[col] = str
            continue

        series = sample_df[col]
        non_null = series.dropna()
        if non_null.empty:
            inferred[col] = str
            continue

        numeric = pd.to_numeric(non_null, errors="coerce")
        if numeric.notna().all():
            if (numeric % 1 == 0).all():
                inferred[col] = _nullable_int_dtype(float(numeric.min()), float(numeric.max()))
            else:
                inferred[col] = "Float64"
            continue

        as_str = non_null.astype(str)
        if as_str.str.fullmatch(r"-?\d+").all():
            numeric = pd.to_numeric(non_null, errors="coerce")
            inferred[col] = _nullable_int_dtype(float(numeric.min()), float(numeric.max()))
            continue

        nunique = non_null.nunique()
        if (
            nunique <= CSV_EXPORT_CATEGORY_MAX_CARDINALITY
            and nunique < len(non_null)
        ):
            inferred[col] = "category"
            continue

        inferred[col] = str

    return inferred


def load_csv_optimized(
    file_path: str,
    usecols: list[str] | None = None,
    user_dtypes: dict | None = None,
    missings_cols: set[str] | None = None,
) -> tuple[pd.DataFrame, SimpleNamespace, str]:
    """Two-pass CSV load: sample for dtype inference, then full read."""
    missings_cols = missings_cols or set()
    user_dtypes = user_dtypes or {}

    sample_df, encoding = _read_csv_with_encodings(
        file_path,
        usecols=usecols,
        dtype=user_dtypes or None,
        nrows=CSV_EXPORT_SAMPLE_ROWS,
    )
    inferred_dtypes = infer_csv_dtypes(
        sample_df,
        missings_cols=missings_cols,
        user_dtypes=user_dtypes,
    )
    logger.debug(
        "CSV optimized dtype inference for %s (encoding=%r): %s",
        file_path,
        encoding,
        inferred_dtypes,
    )

    read_kwargs: dict = {
        "filepath_or_buffer": file_path,
        "usecols": usecols,
        "dtype": inferred_dtypes,
        "low_memory": False,
    }
    if encoding is not None:
        read_kwargs["encoding"] = encoding

    try:
        df = pd.read_csv(**read_kwargs)
    except Exception as exc:
        logger.warning(
            "Optimized CSV load failed for %s (%s); falling back to legacy load",
            file_path,
            exc,
        )
        df, meta = load_csv_legacy(file_path, usecols=usecols, dtypes=user_dtypes or None)
        return df, meta, "legacy"

    logger.debug(
        "CSV optimized load succeeded with encoding %r, shape: %s",
        encoding,
        df.shape,
    )
    return df, _make_csv_meta(df), "optimized"


def load_csv_for_export(
    file_path: str,
    usecols: list[str] | None = None,
    dtypes: dict | None = None,
    missings: dict | list | None = None,
) -> tuple[pd.DataFrame, SimpleNamespace, str]:
    """Load CSV for export, choosing legacy or optimized path by file size."""
    missings_cols: set[str] = set()
    if isinstance(missings, dict):
        missings_cols = set(missings.keys())

    if should_use_optimized_csv_load(file_path, user_dtypes=dtypes):
        logger.info(
            "Loading CSV (optimized): %s (%s bytes)",
            file_path,
            os.path.getsize(file_path),
        )
        df, meta, loader = load_csv_optimized(
            file_path,
            usecols=usecols,
            user_dtypes=dtypes,
            missings_cols=missings_cols,
        )
        return df, meta, loader

    logger.info(
        "Loading CSV (legacy): %s (%s bytes)",
        file_path,
        os.path.getsize(file_path) if os.path.isfile(file_path) else "unknown",
    )
    df, meta = load_csv_legacy(file_path, usecols=usecols, dtypes=dtypes)
    return df, meta, "legacy"
