"""Register optional metadata reviewer routes and app state."""

from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI

logger = logging.getLogger(__name__)

REVIEWER_JOB_TYPE = "metadata-reviewer"
REVIEWER_AVAILABLE = False


def _load_reviewer_env(reviewer_env_path: Path) -> None:
    if reviewer_env_path.is_file():
        load_dotenv(reviewer_env_path, override=False)
        print(f"Environment: loaded reviewer.env from {reviewer_env_path}")
    else:
        print(f"Environment: no reviewer.env at {reviewer_env_path} (optional)")


def _init_app_state(app: FastAPI) -> None:
    concurrency = int(os.getenv("REVIEWER_CONCURRENCY", "10"))
    max_inflight = int(os.getenv("REVIEWER_MAX_INFLIGHT", "200"))
    timeout_sec = int(os.getenv("REVIEWER_JOB_TIMEOUT_SEC", "900"))

    app.reviewer_sem = asyncio.Semaphore(concurrency)
    app.reviewer_max_inflight = max_inflight
    app.reviewer_job_timeout_sec = timeout_sec
    app.reviewer_tasks = {}
    app.reviewer_cancel_flags = {}


def register_reviewer(app: FastAPI, project_root: Path) -> bool:
    """
    Load reviewer.env, mount /review routes when ai4data is installed.

    Returns True if the reviewer package is available.
    """
    global REVIEWER_AVAILABLE

    _load_reviewer_env(project_root / "reviewer.env")

    try:
        import ai4data.metadata.reviewer  # noqa: F401
    except ImportError:
        logger.info(
            "Metadata reviewer not installed (optional). "
            "Install with: pip install -r requirements-reviewer.txt"
        )
        REVIEWER_AVAILABLE = False
        return False

    REVIEWER_AVAILABLE = True
    _init_app_state(app)

    from src.routers.review import router as review_router

    app.include_router(review_router)
    logger.info("Metadata reviewer routes registered at /review")
    return True


def dispose_reviewer_job_if_needed(app: FastAPI, jobid: str, job: dict) -> None:
    """Cancel a metadata-reviewer asyncio task and signal the pipeline to stop."""
    if job.get("jobtype") != REVIEWER_JOB_TYPE:
        return
    if not getattr(app, "reviewer_tasks", None):
        return
    ev = app.reviewer_cancel_flags.pop(jobid, None)
    if ev is not None:
        ev.set()
    t = app.reviewer_tasks.pop(jobid, None)
    if t is not None and not t.done():
        t.cancel()
