"""Read SPSS .sav files with pyreadstat, with chunked reads for large files."""

from __future__ import annotations

import logging
import os
import shutil
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager

import pandas as pd
import pyreadstat

from src.utils.dta_reader import (
    DEFAULT_CHUNK_SIZE,
    DEFAULT_MEMORY_BUDGET_BYTES,
    MAX_FILE_SIZE_FULL_READ,
    MAX_ROWS_FULL_READ,
    estimate_dta_memory_bytes,
)

logger = logging.getLogger(__name__)

DEFAULT_ENCODINGS = [
    None,
    "utf-8",
    "latin1",
    "cp1252",
    "iso-8859-1",
    "cp850",
    "cp437",
    "windows-1252",
    "ascii",
    "utf-16",
    "utf-32",
]


def _require_sav_file(file_path: str) -> None:
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"File {file_path} does not exist!")


def _probe_sav_read(file_path: str, kwargs: dict) -> None:
    """Raise if pyreadstat cannot read metadata and a one-row data sample."""
    meta_kwargs = dict(kwargs)
    meta_kwargs["metadataonly"] = True
    pyreadstat.read_sav(file_path, **meta_kwargs)

    sample_kwargs = dict(kwargs)
    sample_kwargs["metadataonly"] = False
    sample_kwargs["row_limit"] = 1
    pyreadstat.read_sav(file_path, **sample_kwargs)


def resolve_sav_read_kwargs(
    file_path: str,
    usecols: list[str] | None = None,
    user_missing: bool = True,
    encodings: list[str | None] | None = None,
    require_data_sample: bool = False,
) -> dict:
    """Probe encodings and return kwargs that work for pyreadstat.read_sav."""
    encodings_to_try = encodings or DEFAULT_ENCODINGS
    last_error: Exception | None = None

    for missing_flag in (True, False) if user_missing else (False,):
        for encoding in encodings_to_try:
            kwargs: dict = {
                "metadataonly": False,
                "usecols": usecols,
                "user_missing": missing_flag,
            }
            if encoding is not None:
                kwargs["encoding"] = encoding
            try:
                if require_data_sample:
                    _probe_sav_read(file_path, kwargs)
                else:
                    probe_kwargs = dict(kwargs)
                    probe_kwargs["metadataonly"] = True
                    pyreadstat.read_sav(file_path, **probe_kwargs)
                return kwargs
            except (
                pyreadstat.ReadstatError,
                UnicodeDecodeError,
                UnicodeError,
                ValueError,
            ) as e:
                last_error = e
                continue

    raise RuntimeError(
        f"Failed to read SAV file with any encoding "
        f"(tried both user_missing=True and user_missing=False). "
        f"Last error: {last_error}"
    ) from last_error


def read_sav_metadata(
    file_path: str,
    usecols: list[str] | None = None,
    user_missing: bool = True,
    encodings: list[str | None] | None = None,
) -> tuple[object, dict]:
    """Read SPSS metadata only and return (meta, read_kwargs)."""
    read_kwargs = resolve_sav_read_kwargs(
        file_path,
        usecols=usecols,
        user_missing=user_missing,
        encodings=encodings,
        require_data_sample=False,
    )
    meta_kwargs = dict(read_kwargs)
    meta_kwargs["metadataonly"] = True
    meta_kwargs["usecols"] = usecols
    _, meta = pyreadstat.read_sav(file_path, **meta_kwargs)
    return meta, read_kwargs


def should_use_chunked_sav_read(
    file_path: str,
    meta: object | None = None,
    usecols: list[str] | None = None,
    user_missing: bool = True,
) -> bool:
    """Return True when the SAV file should be processed in chunks."""
    if meta is None:
        meta, _ = read_sav_metadata(
            file_path,
            usecols=usecols,
            user_missing=user_missing,
        )

    rows = meta.number_rows
    cols = meta.number_columns
    file_size = os.path.getsize(file_path)

    if rows >= MAX_ROWS_FULL_READ:
        logger.debug(
            "Using chunked SAV read for %s: rows %s >= %s",
            file_path,
            rows,
            MAX_ROWS_FULL_READ,
        )
        return True
    if file_size >= MAX_FILE_SIZE_FULL_READ:
        logger.debug(
            "Using chunked SAV read for %s: size %s >= %s",
            file_path,
            file_size,
            MAX_FILE_SIZE_FULL_READ,
        )
        return True
    estimated = estimate_dta_memory_bytes(rows, cols, user_missing)
    if estimated >= DEFAULT_MEMORY_BUDGET_BYTES:
        logger.debug(
            "Using chunked SAV read for %s: estimated memory %s >= %s",
            file_path,
            estimated,
            DEFAULT_MEMORY_BUDGET_BYTES,
        )
        return True
    return False


@contextmanager
def sav_read_snapshot(file_path: str):
    """Provide a stable path for chunked SAV reads via hard link or temp copy."""
    _require_sav_file(file_path)
    fd, temp_path = tempfile.mkstemp(suffix=".sav", prefix="sav_read_")
    os.close(fd)
    try:
        try:
            os.link(file_path, temp_path)
            logger.info("Chunked SAV read using hard link: %s", file_path)
        except OSError:
            logger.info(
                "Chunked SAV read copying to temp (hard link unavailable): %s",
                file_path,
            )
            shutil.copy2(file_path, temp_path)
        yield temp_path
    finally:
        try:
            os.unlink(temp_path)
        except OSError:
            logger.warning("Failed to remove temp SAV snapshot: %s", temp_path)


def iter_sav_chunks(
    file_path: str,
    chunksize: int = DEFAULT_CHUNK_SIZE,
    usecols: list[str] | None = None,
    user_missing: bool = True,
    read_kwargs: dict | None = None,
    encodings: list[str | None] | None = None,
) -> Iterator[tuple[pd.DataFrame, object]]:
    """Yield (chunk, meta) pairs for an SPSS .sav file."""
    if read_kwargs is None:
        _, read_kwargs = read_sav_metadata(
            file_path,
            usecols=usecols,
            user_missing=user_missing,
            encodings=encodings,
        )
    else:
        _probe_sav_read(
            file_path,
            {**read_kwargs, "usecols": usecols},
        )

    meta_kwargs = dict(read_kwargs)
    meta_kwargs["metadataonly"] = True
    meta_kwargs["usecols"] = usecols
    _, meta = pyreadstat.read_sav(file_path, **meta_kwargs)

    expected_rows = meta.number_rows
    row_offset = 0
    chunk_index = 0

    while row_offset < expected_rows:
        _require_sav_file(file_path)
        chunk_kwargs = dict(read_kwargs)
        chunk_kwargs["metadataonly"] = False
        chunk_kwargs["usecols"] = usecols
        chunk_kwargs["row_offset"] = row_offset
        chunk_kwargs["row_limit"] = chunksize

        df, _chunk_meta = pyreadstat.read_sav(file_path, **chunk_kwargs)
        if df is None or len(df) == 0:
            logger.warning(
                "Empty SAV chunk at offset %s for %s (expected %s rows total)",
                row_offset,
                file_path,
                expected_rows,
            )
            break

        chunk_index += 1
        logger.info(
            "Read SAV chunk %s for %s: %s rows (offset %s, cumulative %s/%s)",
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
            f"Incomplete SAV read for {file_path}: read {row_offset} rows, "
            f"expected {expected_rows}"
        )


def read_sav(
    file_path: str,
    metadataonly: bool = False,
    usecols: list[str] | None = None,
    user_missing: bool = True,
    encodings: list[str | None] | None = None,
) -> tuple[pd.DataFrame, object]:
    """Read an SPSS .sav file with encoding and user_missing fallbacks."""
    read_kwargs = resolve_sav_read_kwargs(
        file_path,
        usecols=usecols,
        user_missing=user_missing,
        encodings=encodings,
        require_data_sample=not metadataonly,
    )
    read_kwargs = dict(read_kwargs)
    read_kwargs["metadataonly"] = metadataonly
    read_kwargs["usecols"] = usecols
    return pyreadstat.read_sav(file_path, **read_kwargs)


def _validate_exported_row_count(
    rows_written: int,
    expected_rows: int,
    file_path: str,
) -> None:
    if rows_written != expected_rows:
        raise RuntimeError(
            f"SAV export incomplete for {file_path}: wrote {rows_written} rows, "
            f"expected {expected_rows}"
        )


def write_sav_to_csv(
    file_path: str,
    csv_filepath: str,
    user_missing: bool = True,
    chunksize: int = DEFAULT_CHUNK_SIZE,
) -> None:
    """Write an SPSS .sav file to CSV using a full read or chunked streaming."""
    _require_sav_file(file_path)
    meta, read_kwargs = read_sav_metadata(
        file_path,
        user_missing=user_missing,
    )
    expected_rows = meta.number_rows

    if should_use_chunked_sav_read(
        file_path,
        meta=meta,
        user_missing=user_missing,
    ):
        logger.info(
            "Writing SAV to CSV in chunks: %s (%s rows expected)",
            file_path,
            expected_rows,
        )
        rows_written = 0
        first = True
        try:
            with sav_read_snapshot(file_path) as stable_path:
                for chunk, _chunk_meta in iter_sav_chunks(
                    stable_path,
                    chunksize=chunksize,
                    read_kwargs=read_kwargs,
                ):
                    # CSV has no types; to_csv stringifies values as read from pyreadstat.
                    chunk.to_csv(
                        csv_filepath,
                        mode="w" if first else "a",
                        header=first,
                        index=False,
                    )
                    rows_written += len(chunk)
                    logger.info(
                        "Wrote SAV CSV chunk for %s: %s rows (cumulative %s/%s)",
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
                        "Removed incomplete CSV after failed SAV export: %s (%s/%s rows)",
                        csv_filepath,
                        rows_written,
                        expected_rows,
                    )
                except OSError:
                    pass
            raise
        if rows_written == 0:
            raise RuntimeError(f"No data read from SAV file: {file_path}")
        _validate_exported_row_count(rows_written, expected_rows, file_path)
        return

    df, _meta = read_sav(file_path, user_missing=user_missing)
    df.to_csv(csv_filepath, index=False)
