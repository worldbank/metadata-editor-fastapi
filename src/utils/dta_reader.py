"""Read Stata .dta files with pyreadstat, falling back to pandas when metadata encoding fails."""

from __future__ import annotations

import logging
from types import SimpleNamespace

import pandas as pd
import pyreadstat

logger = logging.getLogger(__name__)

DEFAULT_ENCODINGS = [
    None,
    "utf-8",
    "latin1",
    "cp1252",
    "iso-8859-1",
    "cp850",
    "windows-1252",
]

_STATA_TYPE_TO_READSTAT = {
    "b": "int8",
    "h": "int16",
    "l": "int32",
    "f": "float",
    "d": "double",
    "byte": "int8",
    "int": "int16",
    "long": "int32",
    "float": "float",
    "double": "double",
}


def _pandas_dtype_to_readstat(dtype) -> str:
    kind = dtype.kind
    if kind == "i":
        if dtype.itemsize <= 1:
            return "int8"
        if dtype.itemsize <= 2:
            return "int16"
        if dtype.itemsize <= 4:
            return "int32"
        return "int64"
    if kind == "f":
        return "double" if dtype.itemsize >= 8 else "float"
    if kind in ("O", "U", "S"):
        return "string"
    return "string"


def _stata_type_to_readstat(typ) -> str:
    if isinstance(typ, int):
        return "string"
    return _STATA_TYPE_TO_READSTAT.get(typ, "string")


def _normalize_value_labels(value_labels: dict | None) -> dict:
    if not value_labels:
        return {}
    normalized = {}
    for var, labels in value_labels.items():
        normalized[var] = {
            (int(k) if hasattr(k, "item") else k): v for k, v in labels.items()
        }
    return normalized


def _meta_from_stata_reader(
    reader: pd.io.stata.StataReader,
    df: pd.DataFrame | None = None,
    usecols: list[str] | None = None,
) -> SimpleNamespace:
    reader._ensure_open()
    varlist = list(reader._varlist)
    fmtlist = list(reader._fmtlist)
    typlist = list(reader._typlist)
    var_labels = reader.variable_labels()
    value_labels = _normalize_value_labels(reader.value_labels())

    if usecols:
        selected = [name for name in varlist if name in usecols]
    else:
        selected = varlist

    index_by_name = {name: idx for idx, name in enumerate(varlist)}
    column_names_to_labels = {
        name: var_labels.get(name, "") for name in selected
    }
    readstat_variable_types = {}
    original_variable_types = {}
    variable_display_width = {}

    for name in selected:
        idx = index_by_name[name]
        readstat_variable_types[name] = _stata_type_to_readstat(typlist[idx])
        original_variable_types[name] = fmtlist[idx]
        if df is not None and name in df.columns:
            readstat_variable_types[name] = _pandas_dtype_to_readstat(df[name].dtype)
            if readstat_variable_types[name] == "string":
                original_variable_types[name] = f"str{df[name].astype(str).str.len().max() or 1}"
        variable_display_width[name] = 0

    number_rows = reader._nobs if df is None else len(df)
    number_columns = len(selected)

    filtered_value_labels = {
        name: value_labels[name]
        for name in selected
        if name in value_labels
    }

    return SimpleNamespace(
        column_names=selected,
        column_names_to_labels=column_names_to_labels,
        number_rows=number_rows,
        number_columns=number_columns,
        variable_value_labels=filtered_value_labels,
        readstat_variable_types=readstat_variable_types,
        original_variable_types=original_variable_types,
        variable_measure={name: "unknown" for name in selected},
        variable_display_width=variable_display_width,
        missing_user_values={},
    )


def _read_dta_pandas(
    file_path: str,
    metadataonly: bool = False,
    usecols: list[str] | None = None,
) -> tuple[pd.DataFrame, SimpleNamespace]:
    columns = list(usecols) if usecols else None

    with pd.io.stata.StataReader(file_path) as reader:
        if metadataonly:
            meta = _meta_from_stata_reader(reader, df=None, usecols=columns)
            df = pd.DataFrame(columns=meta.column_names)
            return df, meta

        df = pd.read_stata(
            file_path,
            convert_categoricals=False,
            convert_missing=False,
            columns=columns,
        )
        meta = _meta_from_stata_reader(reader, df=df, usecols=columns)
        return df, meta


def read_dta(
    file_path: str,
    metadataonly: bool = False,
    usecols: list[str] | None = None,
    user_missing: bool = True,
    encodings: list[str | None] | None = None,
) -> tuple[pd.DataFrame, object]:
    """Read a Stata .dta file, falling back to pandas when pyreadstat cannot decode metadata."""
    encodings_to_try = encodings or DEFAULT_ENCODINGS
    df = None
    meta = None
    last_error: Exception | None = None
    saw_unicode_error = False

    for missing_flag in (True, False) if user_missing else (False,):
        for encoding in encodings_to_try:
            try:
                kwargs = {
                    "metadataonly": metadataonly,
                    "usecols": usecols,
                    "user_missing": missing_flag,
                }
                if encoding is not None:
                    kwargs["encoding"] = encoding
                df, meta = pyreadstat.read_dta(file_path, **kwargs)
                if missing_flag is False and user_missing:
                    logger.debug(
                        "Read DTA file with user_missing=False: %s", file_path
                    )
                return df, meta
            except UnicodeDecodeError as e:
                saw_unicode_error = True
                last_error = e
            except (UnicodeError, pyreadstat.ReadstatError, ValueError) as e:
                last_error = e
                if isinstance(e, UnicodeError):
                    saw_unicode_error = True

    if saw_unicode_error:
        logger.warning(
            "pyreadstat failed to decode DTA metadata for %s (%s); using pandas.read_stata",
            file_path,
            last_error,
        )
        return _read_dta_pandas(file_path, metadataonly=metadataonly, usecols=usecols)

    raise last_error if last_error else RuntimeError(f"Failed to read DTA file: {file_path}")
