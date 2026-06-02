"""Read Stata .dta files with pyreadstat, falling back to pandas when metadata encoding fails."""

from __future__ import annotations

import logging
import os
import shutil
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
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

DEFAULT_CHUNK_SIZE = int(os.environ.get("DTA_CHUNK_SIZE", "50000"))
MAX_ROWS_FULL_READ = int(os.environ.get("DTA_MAX_ROWS_FULL_READ", "500000"))
MAX_FILE_SIZE_FULL_READ = int(
    os.environ.get("DTA_MAX_FILE_SIZE_FULL_READ", str(200 * 1024 * 1024))
)
DEFAULT_MEMORY_BUDGET_BYTES = int(
    os.environ.get("DTA_MEMORY_BUDGET_BYTES", str(1024 * 1024 * 1024))
)
BYTES_PER_CELL_OBJECT = 8
BYTES_PER_CELL_NUMERIC = 8

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


def estimate_dta_memory_bytes(
    rows: int, cols: int, user_missing: bool = True
) -> int:
    """Rough in-memory estimate for a fully loaded DataFrame."""
    bytes_per_cell = BYTES_PER_CELL_OBJECT if user_missing else BYTES_PER_CELL_NUMERIC
    return rows * cols * bytes_per_cell


def should_use_chunked_read(
    file_path: str,
    meta: object | None = None,
    usecols: list[str] | None = None,
    user_missing: bool = True,
) -> bool:
    """Return True when the file should be processed in chunks instead of a full read."""
    if meta is None:
        _, meta = read_dta(
            file_path,
            metadataonly=True,
            usecols=usecols,
            user_missing=user_missing,
        )

    rows = meta.number_rows
    cols = meta.number_columns
    file_size = os.path.getsize(file_path)

    if rows >= MAX_ROWS_FULL_READ:
        logger.debug(
            "Using chunked read for %s: rows %s >= %s",
            file_path,
            rows,
            MAX_ROWS_FULL_READ,
        )
        return True
    if file_size >= MAX_FILE_SIZE_FULL_READ:
        logger.debug(
            "Using chunked read for %s: size %s >= %s",
            file_path,
            file_size,
            MAX_FILE_SIZE_FULL_READ,
        )
        return True
    estimated = estimate_dta_memory_bytes(rows, cols, user_missing)
    if estimated >= DEFAULT_MEMORY_BUDGET_BYTES:
        logger.debug(
            "Using chunked read for %s: estimated memory %s >= %s",
            file_path,
            estimated,
            DEFAULT_MEMORY_BUDGET_BYTES,
        )
        return True
    return False


def _require_dta_file(file_path: str) -> None:
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"File {file_path} does not exist!")


@contextmanager
def dta_read_snapshot(file_path: str):
    """Provide a stable path for chunked reads via hard link or temp copy.

    If the original file is deleted while export is running (e.g. by an external
    process), reads can continue against the linked copy until this context exits.
    """
    _require_dta_file(file_path)
    fd, temp_path = tempfile.mkstemp(suffix=".dta", prefix="dta_read_")
    os.close(fd)
    try:
        try:
            os.link(file_path, temp_path)
            logger.info("Chunked DTA read using hard link: %s", file_path)
        except OSError:
            logger.info(
                "Chunked DTA read copying to temp (hard link unavailable): %s",
                file_path,
            )
            shutil.copy2(file_path, temp_path)
        yield temp_path
    finally:
        try:
            os.unlink(temp_path)
        except OSError:
            logger.warning("Failed to remove temp DTA snapshot: %s", temp_path)


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


def _resolve_pyreadstat_kwargs(
    file_path: str,
    usecols: list[str] | None = None,
    user_missing: bool = True,
    encodings: list[str | None] | None = None,
) -> dict | None:
    """Probe pyreadstat encodings with a metadata-only read; return kwargs that work."""
    encodings_to_try = encodings or DEFAULT_ENCODINGS
    last_error: Exception | None = None
    saw_unicode_error = False

    for missing_flag in (True, False) if user_missing else (False,):
        for encoding in encodings_to_try:
            kwargs = {
                "metadataonly": True,
                "usecols": usecols,
                "user_missing": missing_flag,
            }
            if encoding is not None:
                kwargs["encoding"] = encoding
            try:
                pyreadstat.read_dta(file_path, **kwargs)
                resolved = dict(kwargs)
                resolved["metadataonly"] = False
                return resolved
            except UnicodeDecodeError as e:
                saw_unicode_error = True
                last_error = e
            except (UnicodeError, pyreadstat.ReadstatError, ValueError) as e:
                last_error = e
                if isinstance(e, UnicodeError):
                    saw_unicode_error = True

    if saw_unicode_error:
        return None

    raise last_error if last_error else RuntimeError(f"Failed to read DTA file: {file_path}")


def read_dta(
    file_path: str,
    metadataonly: bool = False,
    usecols: list[str] | None = None,
    user_missing: bool = True,
    encodings: list[str | None] | None = None,
) -> tuple[pd.DataFrame, object]:
    """Read a Stata .dta file, falling back to pandas when pyreadstat cannot decode metadata."""
    kwargs = _resolve_pyreadstat_kwargs(
        file_path,
        usecols=usecols,
        user_missing=user_missing,
        encodings=encodings,
    )
    if kwargs is not None:
        if metadataonly:
            kwargs["metadataonly"] = True
        if kwargs.get("user_missing") is False and user_missing:
            logger.debug("Read DTA file with user_missing=False: %s", file_path)
        return pyreadstat.read_dta(file_path, **kwargs)

    logger.warning(
        "pyreadstat failed to decode DTA metadata for %s; using pandas.read_stata",
        file_path,
    )
    return _read_dta_pandas(file_path, metadataonly=metadataonly, usecols=usecols)


def _convert_mixed_column(series: pd.Series) -> pd.Series:
    def try_convert(x):
        try:
            if isinstance(x, float) and x.is_integer():
                return int(x)
            return int(str(x)) if str(x).lstrip("-").isdigit() else x
        except (ValueError, TypeError):
            return x

    return series.apply(try_convert)


def prepare_dta_dataframe(df: pd.DataFrame, meta: object) -> pd.DataFrame:
    """Apply the same dtype/missing conversions used before CSV export."""
    df = df.convert_dtypes()
    missing_user_values = getattr(meta, "missing_user_values", None) or {}
    for col in df.columns:
        if col in missing_user_values:
            df[col] = _convert_mixed_column(df[col])
    return df


def _iter_dta_chunks_pyreadstat(
    file_path: str,
    chunksize: int,
    read_kwargs: dict,
    usecols: list[str] | None = None,
) -> Iterator[tuple[pd.DataFrame, object]]:
    """Read a DTA file in chunks using explicit row_offset / row_limit."""
    meta_kwargs = dict(read_kwargs)
    meta_kwargs["metadataonly"] = True
    meta_kwargs["usecols"] = usecols
    _, meta = pyreadstat.read_dta(file_path, **meta_kwargs)

    expected_rows = meta.number_rows
    row_offset = 0
    chunk_index = 0

    while row_offset < expected_rows:
        _require_dta_file(file_path)
        chunk_kwargs = dict(read_kwargs)
        chunk_kwargs["metadataonly"] = False
        chunk_kwargs["usecols"] = usecols
        chunk_kwargs["row_offset"] = row_offset
        chunk_kwargs["row_limit"] = chunksize

        df, _chunk_meta = pyreadstat.read_dta(file_path, **chunk_kwargs)
        if df is None or len(df) == 0:
            logger.warning(
                "Empty chunk at offset %s for %s (expected %s rows total)",
                row_offset,
                file_path,
                expected_rows,
            )
            break

        chunk_index += 1
        logger.info(
            "Read DTA chunk %s for %s: %s rows (offset %s, cumulative %s/%s)",
            chunk_index,
            file_path,
            len(df),
            row_offset,
            row_offset + len(df),
            expected_rows,
        )
        yield df, meta

        row_offset += len(df)
        if len(df) < chunksize:
            break

    if row_offset < expected_rows:
        raise RuntimeError(
            f"Incomplete DTA read for {file_path}: read {row_offset} rows, "
            f"expected {expected_rows}"
        )


def iter_dta_chunks(
    file_path: str,
    chunksize: int = DEFAULT_CHUNK_SIZE,
    usecols: list[str] | None = None,
    user_missing: bool = True,
    encodings: list[str | None] | None = None,
) -> Iterator[tuple[pd.DataFrame, object]]:
    """Yield (chunk, meta) pairs for a DTA file."""
    read_kwargs = _resolve_pyreadstat_kwargs(
        file_path,
        usecols=usecols,
        user_missing=user_missing,
        encodings=encodings,
    )
    if read_kwargs is None:
        logger.info("Using pandas StataReader chunks for %s", file_path)
        columns = list(usecols) if usecols else None
        with pd.io.stata.StataReader(file_path) as reader:
            meta = _meta_from_stata_reader(reader, df=None, usecols=columns)
            expected_rows = meta.number_rows
            row_offset = 0
            chunk_index = 0
            while row_offset < expected_rows:
                _require_dta_file(file_path)
                try:
                    chunk = reader.read(chunksize)
                except StopIteration:
                    break
                if chunk is None or chunk.empty:
                    break
                if columns:
                    chunk = chunk[[name for name in columns if name in chunk.columns]]
                chunk_index += 1
                logger.info(
                    "Read DTA chunk %s for %s (pandas): %s rows (offset %s, cumulative %s/%s)",
                    chunk_index,
                    file_path,
                    len(chunk),
                    row_offset,
                    row_offset + len(chunk),
                    expected_rows,
                )
                yield chunk, meta
                row_offset += len(chunk)
                if len(chunk) < chunksize:
                    break
        if row_offset < expected_rows:
            raise RuntimeError(
                f"Incomplete DTA read for {file_path}: read {row_offset} rows, "
                f"expected {expected_rows}"
            )
        return

    yield from _iter_dta_chunks_pyreadstat(
        file_path,
        chunksize,
        read_kwargs,
        usecols=usecols,
    )


def _validate_exported_row_count(
    rows_written: int,
    expected_rows: int,
    file_path: str,
) -> None:
    if rows_written != expected_rows:
        raise RuntimeError(
            f"DTA export incomplete for {file_path}: wrote {rows_written} rows, "
            f"expected {expected_rows}"
        )


def write_dta_to_csv(
    file_path: str,
    csv_filepath: str,
    user_missing: bool = True,
    chunksize: int = DEFAULT_CHUNK_SIZE,
) -> None:
    """Write a DTA file to CSV using a full read or chunked streaming."""
    _require_dta_file(file_path)
    _, meta = read_dta(
        file_path,
        metadataonly=True,
        user_missing=user_missing,
    )
    expected_rows = meta.number_rows

    if should_use_chunked_read(file_path, meta=meta, user_missing=user_missing):
        logger.info(
            "Writing DTA to CSV in chunks: %s (%s rows expected)",
            file_path,
            expected_rows,
        )
        rows_written = 0
        first = True
        try:
            with dta_read_snapshot(file_path) as stable_path:
                for chunk, chunk_meta in iter_dta_chunks(
                    stable_path,
                    chunksize=chunksize,
                    user_missing=user_missing,
                ):
                    chunk = prepare_dta_dataframe(chunk, chunk_meta)
                    chunk.to_csv(
                        csv_filepath,
                        mode="w" if first else "a",
                        header=first,
                        index=False,
                    )
                    rows_written += len(chunk)
                    logger.info(
                        "Wrote DTA CSV chunk for %s: %s rows (cumulative %s/%s)",
                        file_path,
                        len(chunk),
                        rows_written,
                        expected_rows,
                    )
                    first = False
        except Exception:
            if (
                rows_written > 0
                and rows_written < expected_rows
                and os.path.exists(csv_filepath)
            ):
                try:
                    os.unlink(csv_filepath)
                    logger.warning(
                        "Removed incomplete CSV after failed export: %s (%s/%s rows)",
                        csv_filepath,
                        rows_written,
                        expected_rows,
                    )
                except OSError:
                    pass
            raise
        if rows_written == 0:
            raise RuntimeError(f"No data read from DTA file: {file_path}")
        _validate_exported_row_count(rows_written, expected_rows, file_path)
        return

    df, meta = read_dta(file_path, user_missing=user_missing)
    df = prepare_dta_dataframe(df, meta)
    df.to_csv(csv_filepath, index=False)
    _validate_exported_row_count(len(df), expected_rows, file_path)
