"""Storage path configuration and filesystem path validation."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from fastapi import HTTPException

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
load_dotenv(_PROJECT_ROOT / ".env", override=True)

logger = logging.getLogger(__name__)


def _configure_storage_path() -> tuple[Optional[str], bool]:
    """
    Require an explicit STORAGE_PATH choice in the environment:
      - unset: fail startup
      - empty string: disable path validation (local development only)
      - absolute directory path: restrict file operations to that tree
    """
    if "STORAGE_PATH" not in os.environ:
        raise ValueError(
            "STORAGE_PATH must be set explicitly in .env. "
            "Use an absolute directory path for production, or STORAGE_PATH= "
            "(empty) to disable path validation for local development only."
        )

    raw = os.getenv("STORAGE_PATH", "")
    if raw == "":
        logger.warning(
            "Path validation DISABLED (STORAGE_PATH is empty). "
            "Use only on localhost; set STORAGE_PATH to an absolute directory in production."
        )
        return None, False

    if not os.path.isabs(raw):
        raise ValueError("STORAGE_PATH must be an absolute path: " + raw)

    if not os.path.isdir(raw):
        raise ValueError("STORAGE_PATH does not exist or is not a directory: " + raw)

    storage_path = os.path.realpath(raw)
    logger.info("Path validation enabled: %s", storage_path)
    return storage_path, True


STORAGE_PATH, PATH_VALIDATION_ENABLED = _configure_storage_path()


def _canonical_path(path: str) -> str:
    """
    Resolve symlinks and normalize path.
    Works for paths whose final component does not exist yet (uses existing parents).
    """
    return str(Path(path).resolve(strict=False))


def is_safe_path(file_path: str) -> bool:
    """Return True when file_path is allowed under STORAGE_PATH (or validation is disabled)."""
    if not PATH_VALIDATION_ENABLED or STORAGE_PATH is None:
        return True

    try:
        target_path = _canonical_path(file_path)
        storage_root = _canonical_path(STORAGE_PATH)
        common = os.path.commonpath([storage_root, target_path])
    except (OSError, ValueError):
        return False

    return common == storage_root


def ensure_safe_path(path: str, *, label: str = "path") -> None:
    """Raise ValueError when path is outside STORAGE_PATH."""
    if not is_safe_path(path):
        raise ValueError(f"Invalid {label}: {path}")


def ensure_safe_paths(*paths: str, label: str = "path") -> None:
    for path in paths:
        if path:
            ensure_safe_path(path, label=label)


def ensure_safe_path_http(path: str, *, label: str = "path") -> None:
    """Raise HTTP 400 when path is outside STORAGE_PATH."""
    if not is_safe_path(path):
        raise HTTPException(status_code=400, detail=f"Invalid {label}: {path}")


def ensure_safe_paths_http(*paths: str, label: str = "path") -> None:
    for path in paths:
        if path:
            ensure_safe_path_http(path, label=label)
