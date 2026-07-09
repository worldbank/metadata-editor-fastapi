"""Extract source file format and version from Stata/SPSS/CSV files."""

from __future__ import annotations

import os
import re
from typing import Any, List, Optional

# Stata dta release → user-facing Stata version (inverse of ExportDatafile mapping)
_STATA_RELEASE_TO_VERSION = {
    113: "8",
    114: "10",
    115: "12",
    117: "13",
    118: "14",
    119: "15",
}


def stata_release_to_version(release: Optional[int]) -> Optional[str]:
    if release is None:
        return None
    return _STATA_RELEASE_TO_VERSION.get(int(release), str(release))


def read_stata_release(file_path: str) -> Optional[int]:
    """Read Stata .dta format release from file header (XML or binary)."""
    try:
        with open(file_path, "rb") as fh:
            head = fh.read(256)
    except OSError:
        return None

    if not head:
        return None

    # Modern Stata 13+ XML header: <release>118</release>
    m = re.search(br"<release>(\d+)</release>", head)
    if m:
        return int(m.group(1))

    # Older binary formats: first byte is release (e.g. 113–115)
    first = head[0]
    if 105 <= first <= 119:
        return int(first)

    return None


def read_spss_format_hint(file_path: str) -> Optional[str]:
    """Best-effort SPSS format label from file magic / pyreadstat file_format."""
    try:
        with open(file_path, "rb") as fh:
            magic = fh.read(4)
    except OSError:
        return None

    if magic.startswith(b"$FL2"):
        return "sav"
    if magic.startswith(b"$FL3"):
        return "zsav"
    return None


def build_file_info(
    file_path: str,
    meta: Any = None,
) -> dict:
    """
    Build file_info block for name-labels / inspect responses.

    Returns keys: format, format_release, format_version, format_label, file_label
    """
    ext = os.path.splitext(file_path)[1].lower().lstrip(".")
    file_label = None
    if meta is not None:
        file_label = getattr(meta, "file_label", None) or None

    info: dict = {
        "format": ext or None,
        "format_release": None,
        "format_version": None,
        "format_label": None,
        "file_label": file_label,
    }

    if ext == "csv":
        info["format"] = "csv"
        info["format_label"] = "CSV"
        return info

    if ext == "dta":
        release = read_stata_release(file_path)
        version = stata_release_to_version(release)
        info["format"] = "dta"
        info["format_release"] = release
        info["format_version"] = version
        if version:
            info["format_label"] = f"Stata {version}"
        else:
            info["format_label"] = "Stata"
        return info

    if ext in ("sav", "zsav", "por"):
        hint = read_spss_format_hint(file_path)
        fmt = "sav"
        if meta is not None:
            ff = getattr(meta, "file_format", None)
            if isinstance(ff, str) and ff:
                # e.g. "sav/zsav"
                if "zsav" in ff.lower():
                    fmt = "sav"
                elif "sav" in ff.lower():
                    fmt = "sav"
                elif "por" in ff.lower():
                    fmt = "por"
        elif hint:
            fmt = "sav" if hint in ("sav", "zsav") else hint
        info["format"] = fmt
        info["format_label"] = "SPSS"
        # SPSS file version is not reliably exposed by pyreadstat; leave null
        info["format_version"] = None
        info["format_release"] = None
        return info

    if ext:
        info["format_label"] = ext.upper()
    return info


def compare_columns(
    file_columns: List[str],
    expected_columns: Optional[List[str]],
) -> Optional[dict]:
    """Compare file column names to expected (e.g. DB variable names)."""
    if expected_columns is None:
        return None

    file_set = set(file_columns)
    expected_set = set(expected_columns)
    missing_in_file = [c for c in expected_columns if c not in file_set]
    extra_in_file = [c for c in file_columns if c not in expected_set]
    return {
        "expected_columns_provided": True,
        "missing_in_file": missing_in_file,
        "extra_in_file": extra_in_file,
        "match": len(missing_in_file) == 0 and len(extra_in_file) == 0,
        "match_mode": "exact_name_set",
    }
