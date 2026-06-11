"""FIFO job enqueue helpers, durable audit logging, and SQLite persistence."""

from __future__ import annotations

import datetime
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Callable, Awaitable

from src.DataUtils import DataUtils
from src.job_handlers import build_job_callback

logger = logging.getLogger(__name__)

_AUDIT_PATH: Path | None = None

_FILE_PATH_KEYS = (
    "file_path",
    "csv_path",
    "output_path",
    "output_file",
    "output_csv_path",
    "csv_output_path",
)


def _audit_file_path() -> Path:
    global _AUDIT_PATH
    if _AUDIT_PATH is None:
        logs_dir = Path(os.getenv("JOB_AUDIT_LOG_DIR", "logs"))
        logs_dir.mkdir(parents=True, exist_ok=True)
        _AUDIT_PATH = logs_dir / os.getenv("JOB_AUDIT_LOG_FILE", "job_audit.log")
    return _AUDIT_PATH


def _info_as_dict(info: Any) -> dict:
    if info is None:
        return {}
    if hasattr(info, "model_dump"):
        return info.model_dump()
    if isinstance(info, dict):
        return info
    if hasattr(info, "file_path"):
        return {"file_path": info.file_path}
    return {}


def job_context_from_record(job: dict | None) -> dict[str, Any]:
    """Extract log-friendly fields from an app.jobs entry."""
    job = job or {}
    info = _info_as_dict(job.get("info"))

    file_path = None
    for key in _FILE_PATH_KEYS:
        value = info.get(key)
        if value:
            file_path = value
            break

    file_size_bytes = None
    file_size = None
    if file_path and os.path.exists(file_path):
        try:
            file_size_bytes = os.path.getsize(file_path)
            file_size = DataUtils.sizeof_fmt(file_size_bytes)
        except OSError:
            pass

    return {
        "jobtype": job.get("jobtype"),
        "file_path": file_path,
        "file_size": file_size,
        "file_size_bytes": file_size_bytes,
        "export_format": info.get("export_format"),
        "project_id": info.get("project_id"),
    }


def _write_audit_line(payload: dict[str, Any]) -> None:
    line = json.dumps(payload, default=str, ensure_ascii=False)
    path = _audit_file_path()
    with open(path, "a", encoding="utf-8") as audit_file:
        audit_file.write(line + "\n")
        audit_file.flush()
        os.fsync(audit_file.fileno())


def audit_job_event(event: str, jobid: str, context: dict[str, Any] | None = None, **extra: Any) -> None:
    """Append one durable audit record (flushed to disk immediately)."""
    record = {
        "ts": datetime.datetime.now().isoformat(),
        "event": event,
        "jobid": jobid,
        "pid": os.getpid(),
    }
    if context:
        record.update({k: v for k, v in context.items() if v is not None})
    record.update({k: v for k, v in extra.items() if v is not None})
    try:
        _write_audit_line(record)
    except OSError as exc:
        logger.error("Failed to write job audit log: %s", exc)


def _log_job_message(level: int, message: str, jobid: str, context: dict[str, Any], **extra: Any) -> None:
    parts = [
        message,
        f"jobid={jobid}",
    ]
    if context.get("jobtype"):
        parts.append(f"type={context['jobtype']}")
    if context.get("file_path"):
        parts.append(f"file={context['file_path']}")
    if context.get("file_size"):
        parts.append(f"size={context['file_size']}")
    if context.get("export_format"):
        parts.append(f"format={context['export_format']}")
    if context.get("project_id"):
        parts.append(f"project_id={context['project_id']}")
    for key, value in extra.items():
        if value is not None:
            parts.append(f"{key}={value}")
    logger.log(level, " ".join(parts))


def _queue_size(app: Any) -> int:
    store = getattr(app, "job_store", None)
    if store is not None:
        return store.count_by_status("queued")
    return app.fifo_queue.qsize()


def _is_job_cancelled(app: Any, jobid: str) -> bool:
    job = app.jobs.get(jobid, {})
    if job.get("status") == "cancelled":
        return True
    store = getattr(app, "job_store", None)
    if store is not None:
        stored = store.get_job(jobid)
        if stored and stored.get("status") == "cancelled":
            return True
    return False


def _persist_queued_job(app: Any, job: dict[str, Any]) -> None:
    store = getattr(app, "job_store", None)
    if store is None:
        return
    try:
        store.save_queued_job(job)
    except Exception:
        logger.exception("Failed to persist queued job %s", job.get("jobid"))


def _sync_job_to_store(app: Any, jobid: str) -> None:
    store = getattr(app, "job_store", None)
    job = app.jobs.get(jobid)
    if store is None or not job:
        return
    try:
        store.sync_from_memory(job)
    except Exception:
        logger.exception("Failed to sync job %s to store", jobid)


async def enqueue_fifo_job(
    app: Any,
    jobid: str,
    callback: Callable[[], Awaitable[Any]],
    *,
    persist: bool = True,
) -> None:
    """
    Queue a FIFO job with queued/started/completed/failed audit + INFO logging.

    callback: zero-arg async callable (e.g. functools.partial(async_fn, jobid, ...)).
    persist: write queued state to SQLite (set False when recovering on startup).
    """
    job = app.jobs.get(jobid, {})
    context = job_context_from_record(job)
    context["jobid"] = jobid

    if persist:
        _persist_queued_job(app, job)

    audit_job_event("queued", jobid, context, queue_size=_queue_size(app))
    _log_job_message(
        logging.INFO,
        "Job queued",
        jobid,
        context,
        queue_size=_queue_size(app),
    )

    async def wrapped() -> Any:
        if _is_job_cancelled(app, jobid):
            logger.info("Skipping cancelled job jobid=%s", jobid)
            audit_job_event("cancelled", jobid, context, reason="cancelled_before_start")
            return {"status": "cancelled"}

        start_ctx = job_context_from_record(app.jobs.get(jobid, job))
        start_ctx["jobid"] = jobid
        started = time.monotonic()
        started_at = datetime.datetime.now().isoformat()

        if jobid in app.jobs:
            app.jobs[jobid]["status"] = "processing"
            app.jobs[jobid]["started_at"] = started_at
        store = getattr(app, "job_store", None)
        if store is not None:
            store.update_status(jobid, "processing", started_at=started_at)

        audit_job_event("started", jobid, start_ctx, queue_size=_queue_size(app))
        _log_job_message(logging.INFO, "Job started", jobid, start_ctx)

        try:
            result = await callback()
            duration_s = round(time.monotonic() - started, 2)
            final_job = app.jobs.get(jobid, {})
            job_status = final_job.get("status")
            job_error = final_job.get("error")

            _sync_job_to_store(app, jobid)

            if job_status == "error":
                audit_job_event(
                    "failed",
                    jobid,
                    start_ctx,
                    duration_s=duration_s,
                    error=job_error,
                )
                _log_job_message(
                    logging.ERROR,
                    "Job failed",
                    jobid,
                    start_ctx,
                    duration_s=duration_s,
                    error=job_error,
                )
            elif job_status == "cancelled":
                audit_job_event("cancelled", jobid, start_ctx, duration_s=duration_s)
                _log_job_message(
                    logging.INFO,
                    "Job cancelled",
                    jobid,
                    start_ctx,
                    duration_s=duration_s,
                )
            else:
                audit_job_event("completed", jobid, start_ctx, duration_s=duration_s)
                _log_job_message(
                    logging.INFO,
                    "Job completed",
                    jobid,
                    start_ctx,
                    duration_s=duration_s,
                )
            return result
        except Exception as exc:
            duration_s = round(time.monotonic() - started, 2)
            if jobid in app.jobs:
                app.jobs[jobid]["status"] = "error"
                app.jobs[jobid]["error"] = str(exc)
                app.jobs[jobid]["completed_at"] = datetime.datetime.now().isoformat()
            _sync_job_to_store(app, jobid)
            audit_job_event(
                "failed",
                jobid,
                start_ctx,
                duration_s=duration_s,
                error_type=type(exc).__name__,
                error=str(exc),
            )
            _log_job_message(
                logging.ERROR,
                "Job failed",
                jobid,
                start_ctx,
                duration_s=duration_s,
                error_type=type(exc).__name__,
                error=str(exc),
            )
            raise

    await app.fifo_queue.put(wrapped)


async def recover_pending_jobs(app: Any) -> int:
    """Reload queued jobs from SQLite after a restart."""
    store = getattr(app, "job_store", None)
    if store is None:
        return 0

    interrupted = store.mark_processing_as_interrupted()
    for job in interrupted:
        app.jobs[job["jobid"]] = job

    queued_jobs = store.list_by_status("queued")
    recovered = 0
    for job in queued_jobs:
        jobid = job["jobid"]
        app.jobs[jobid] = job
        try:
            callback = build_job_callback(app, jobid, job["jobtype"], _info_as_dict(job.get("info")))
        except Exception:
            logger.exception("Cannot rebuild callback for job %s type=%s", jobid, job.get("jobtype"))
            job["status"] = "error"
            job["error"] = "unsupported or invalid job type on recovery"
            job["completed_at"] = datetime.datetime.now().isoformat()
            app.jobs[jobid] = job
            store.sync_from_memory(job)
            continue

        await enqueue_fifo_job(app, jobid, callback, persist=False)
        recovered += 1

    if recovered or interrupted:
        logger.info(
            "Job recovery complete recovered=%s interrupted=%s",
            recovered,
            len(interrupted),
        )
    return recovered
