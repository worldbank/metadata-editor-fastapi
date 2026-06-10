"""Metadata review API: submit long-running reviews without blocking the FIFO worker."""

from __future__ import annotations

import asyncio
import datetime
import json
import logging
import os
import threading
import time
from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse

from ai4data.metadata.reviewer.core import (
    MetadataReviewerCore,
    _DEFAULT_MANIFEST_FILE,
    _DEFAULT_TEAM_PRESET,
)

from ..models.review_models import ReviewSubmitRequest
from ..reviewer.integration import REVIEWER_AVAILABLE
from ..services.reviewer_service import get_reviewer_model_client

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/review", tags=["review"])

JOB_TYPE = "metadata-reviewer"


@router.get("/manifests")
async def list_review_manifests():
    """List YAML manifest filenames available for POST /review/jobs."""
    if not REVIEWER_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="Metadata reviewer requires: pip install -r requirements-reviewer.txt",
        )
    core = MetadataReviewerCore(model_client=None)  # list_manifests does not use the client
    try:
        return {"manifests": core.list_manifests()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


async def _run_metadata_review(
    app: Any,
    jobid: str,
    metadata: Dict[str, Any],
    manifest_file: Optional[str],
    team_preset: Optional[str],
    cancel_flag: threading.Event,
) -> None:
    sem: asyncio.Semaphore = app.reviewer_sem
    timeout: float = float(app.reviewer_job_timeout_sec)
    jobs_dir = os.path.join(os.getcwd(), "jobs")
    os.makedirs(jobs_dir, exist_ok=True)
    out_path = os.path.join(jobs_dir, f"{jobid}.json")

    try:
        if cancel_flag.is_set():
            app.jobs[jobid]["status"] = "cancelled"
            app.jobs[jobid]["cancelled_at"] = datetime.datetime.now().isoformat()
            app.jobs[jobid]["cancellation_reason"] = "Cancelled before start"
            return

        app.jobs[jobid]["status"] = "waiting"

        async with sem:
            if cancel_flag.is_set():
                app.jobs[jobid]["status"] = "cancelled"
                app.jobs[jobid]["cancelled_at"] = datetime.datetime.now().isoformat()
                app.jobs[jobid]["cancellation_reason"] = "Cancelled while waiting or running"
                return

            app.jobs[jobid]["status"] = "processing"
            app.jobs[jobid]["started_at"] = datetime.datetime.now().isoformat()

            model_client = get_reviewer_model_client()
            core = MetadataReviewerCore(model_client)

            mf = manifest_file or _DEFAULT_MANIFEST_FILE
            tp = team_preset or _DEFAULT_TEAM_PRESET

            result = await asyncio.wait_for(
                core.run(
                    metadata_to_scan=metadata,
                    manifest_file=mf,
                    team_preset=tp,
                    cancel_flag=cancel_flag,
                ),
                timeout=timeout,
            )

            if cancel_flag.is_set():
                app.jobs[jobid]["status"] = "cancelled"
                app.jobs[jobid]["cancelled_at"] = datetime.datetime.now().isoformat()
                app.jobs[jobid]["cancellation_reason"] = "Cancelled during review"
                return

            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(result if result is not None else [], f, default=str)

            app.jobs[jobid]["status"] = "done"
            app.jobs[jobid]["completed_at"] = datetime.datetime.now().isoformat()

    except asyncio.TimeoutError:
        app.jobs[jobid]["status"] = "error"
        app.jobs[jobid]["error"] = f"Reviewer job timed out after {timeout:.0f}s"
        app.jobs[jobid]["completed_at"] = datetime.datetime.now().isoformat()
        logger.warning("Reviewer job %s timed out", jobid)
    except asyncio.CancelledError:
        app.jobs[jobid]["status"] = "cancelled"
        app.jobs[jobid]["cancelled_at"] = datetime.datetime.now().isoformat()
        app.jobs[jobid]["cancellation_reason"] = "Task cancelled"
        logger.info("Reviewer job %s asyncio task cancelled", jobid)
        raise
    except Exception as e:
        logger.exception("Reviewer job %s failed", jobid)
        app.jobs[jobid]["status"] = "error"
        app.jobs[jobid]["error"] = str(e)
        app.jobs[jobid]["completed_at"] = datetime.datetime.now().isoformat()
    finally:
        app.reviewer_tasks.pop(jobid, None)
        app.reviewer_cancel_flags.pop(jobid, None)


@router.post("/jobs")
async def submit_metadata_review(req: ReviewSubmitRequest, request: Request):
    """
    Start a metadata review in the background.

    Poll ``GET /jobs/{job_id}`` and read ``data`` when ``status`` is ``done``.
    Concurrency is capped by ``REVIEWER_CONCURRENCY`` (asyncio.Semaphore).
    """
    if not REVIEWER_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="Metadata reviewer requires: pip install -r requirements-reviewer.txt",
        )

    app = request.app

    if len(app.reviewer_tasks) >= app.reviewer_max_inflight:
        raise HTTPException(
            status_code=429,
            detail="Too many in-flight reviewer jobs; try again later.",
        )

    try:
        get_reviewer_model_client()
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e)) from e

    jobid = "job-" + str(time.time())
    now = datetime.datetime.now().isoformat()
    app.jobs[jobid] = {
        "jobid": jobid,
        "jobtype": JOB_TYPE,
        "status": "queued",
        "created_at": now,
        "completed_at": None,
        "last_accessed": now,
        "info": {
            "manifest_file": req.manifest_file or _DEFAULT_MANIFEST_FILE,
            "team_preset": req.team_preset or _DEFAULT_TEAM_PRESET,
        },
    }

    cancel_flag = threading.Event()
    app.reviewer_cancel_flags[jobid] = cancel_flag

    task = asyncio.create_task(
        _run_metadata_review(
            app,
            jobid,
            req.metadata,
            req.manifest_file,
            req.team_preset,
            cancel_flag,
        ),
        name=f"metadata-reviewer-{jobid}",
    )
    app.reviewer_tasks[jobid] = task

    return JSONResponse(
        status_code=202,
        content={"message": "Reviewer job started", "job_id": jobid},
    )
