"""SQLite-backed durable storage for FIFO job records."""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import threading
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS jobs (
    jobid TEXT PRIMARY KEY,
    jobtype TEXT NOT NULL,
    status TEXT NOT NULL,
    info_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    started_at TEXT,
    completed_at TEXT,
    last_accessed TEXT,
    error TEXT,
    cancelled_at TEXT,
    cancellation_reason TEXT
);
CREATE INDEX IF NOT EXISTS idx_jobs_status_created ON jobs(status, created_at);
"""


def _now_iso() -> str:
    return datetime.now().isoformat()


def _info_to_json(info: Any) -> str:
    if info is None:
        payload: dict[str, Any] = {}
    elif hasattr(info, "model_dump"):
        payload = info.model_dump()
    elif isinstance(info, dict):
        payload = info
    else:
        payload = {"value": info}
    return json.dumps(payload, default=str, ensure_ascii=False)


def _info_from_json(info_json: str) -> dict[str, Any]:
    data = json.loads(info_json)
    return data if isinstance(data, dict) else {"value": data}


class JobStore:
    """Small SQLite store for job metadata and status."""

    def __init__(self, db_path: str | None = None) -> None:
        path = db_path or os.getenv("JOB_STORE_DB_PATH", "db/jobs.sqlite")
        self._db_path = Path(path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path, timeout=30, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        return conn

    def _init_schema(self) -> None:
        with self._lock:
            with self._connect() as conn:
                conn.executescript(_SCHEMA_SQL)
                conn.commit()

    def _row_to_job(self, row: sqlite3.Row) -> dict[str, Any]:
        job: dict[str, Any] = {
            "jobid": row["jobid"],
            "jobtype": row["jobtype"],
            "status": row["status"],
            "created_at": row["created_at"],
            "completed_at": row["completed_at"],
            "last_accessed": row["last_accessed"] or row["created_at"],
            "info": _info_from_json(row["info_json"]),
        }
        if row["error"]:
            job["error"] = row["error"]
        if row["cancelled_at"]:
            job["cancelled_at"] = row["cancelled_at"]
        if row["cancellation_reason"]:
            job["cancellation_reason"] = row["cancellation_reason"]
        if row["started_at"]:
            job["started_at"] = row["started_at"]
        return job

    def save_queued_job(self, job: dict[str, Any]) -> None:
        """Insert or refresh a queued job from an in-memory app.jobs entry."""
        jobid = job["jobid"]
        created_at = job.get("created_at") or _now_iso()
        last_accessed = job.get("last_accessed") or created_at
        with self._lock:
            with self._connect() as conn:
                conn.execute(
                    """
                    INSERT INTO jobs (
                        jobid, jobtype, status, info_json, created_at, last_accessed
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    ON CONFLICT(jobid) DO UPDATE SET
                        jobtype = excluded.jobtype,
                        status = excluded.status,
                        info_json = excluded.info_json,
                        last_accessed = excluded.last_accessed
                    WHERE jobs.status IN ('queued', 'waiting')
                    """,
                    (
                        jobid,
                        job["jobtype"],
                        job.get("status", "queued"),
                        _info_to_json(job.get("info")),
                        created_at,
                        last_accessed,
                    ),
                )
                conn.commit()

    def get_job(self, jobid: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM jobs WHERE jobid = ?",
                (jobid,),
            ).fetchone()
        return self._row_to_job(row) if row else None

    def list_by_status(self, *statuses: str) -> list[dict[str, Any]]:
        if not statuses:
            return []
        placeholders = ",".join("?" for _ in statuses)
        with self._connect() as conn:
            rows = conn.execute(
                f"""
                SELECT * FROM jobs
                WHERE status IN ({placeholders})
                ORDER BY created_at ASC
                """,
                statuses,
            ).fetchall()
        return [self._row_to_job(row) for row in rows]

    def count_by_status(self, status: str) -> int:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT COUNT(*) AS c FROM jobs WHERE status = ?",
                (status,),
            ).fetchone()
        return int(row["c"]) if row else 0

    def update_status(
        self,
        jobid: str,
        status: str,
        *,
        error: str | None = None,
        started_at: str | None = None,
        completed_at: str | None = None,
        cancelled_at: str | None = None,
        cancellation_reason: str | None = None,
        touch_accessed: bool = False,
    ) -> None:
        fields = ["status = ?"]
        values: list[Any] = [status]

        if error is not None:
            fields.append("error = ?")
            values.append(error)
        if started_at is not None:
            fields.append("started_at = ?")
            values.append(started_at)
        if completed_at is not None:
            fields.append("completed_at = ?")
            values.append(completed_at)
        if cancelled_at is not None:
            fields.append("cancelled_at = ?")
            values.append(cancelled_at)
        if cancellation_reason is not None:
            fields.append("cancellation_reason = ?")
            values.append(cancellation_reason)
        if touch_accessed:
            fields.append("last_accessed = ?")
            values.append(_now_iso())

        values.append(jobid)
        sql = f"UPDATE jobs SET {', '.join(fields)} WHERE jobid = ?"

        with self._lock:
            with self._connect() as conn:
                conn.execute(sql, values)
                conn.commit()

    def mark_processing_as_interrupted(self, message: str = "interrupted_by_restart") -> list[dict[str, Any]]:
        """Mark in-flight jobs as failed after a crash/restart."""
        now = _now_iso()
        with self._lock:
            with self._connect() as conn:
                rows = conn.execute(
                    "SELECT jobid FROM jobs WHERE status = 'processing'"
                ).fetchall()
                if rows:
                    conn.execute(
                        """
                        UPDATE jobs
                        SET status = 'error',
                            error = ?,
                            completed_at = ?
                        WHERE status = 'processing'
                        """,
                        (message, now),
                    )
                    conn.commit()

        interrupted: list[dict[str, Any]] = []
        for row in rows:
            job = self.get_job(row["jobid"])
            if job:
                interrupted.append(job)
        if interrupted:
            logger.info(
                "Marked %s processing job(s) as interrupted after restart",
                len(interrupted),
            )
        return interrupted

    def delete_job(self, jobid: str) -> None:
        with self._lock:
            with self._connect() as conn:
                conn.execute("DELETE FROM jobs WHERE jobid = ?", (jobid,))
                conn.commit()

    def sync_from_memory(self, job: dict[str, Any]) -> None:
        """Update SQLite from an in-memory app.jobs entry."""
        jobid = job.get("jobid")
        status = job.get("status")
        if not jobid or not status:
            return

        kwargs: dict[str, Any] = {"touch_accessed": True}
        if status in ("done", "error", "cancelled"):
            kwargs["completed_at"] = job.get("completed_at") or _now_iso()
        if status == "processing":
            kwargs["started_at"] = job.get("started_at") or _now_iso()
        if job.get("error"):
            kwargs["error"] = job["error"]
        if job.get("cancelled_at"):
            kwargs["cancelled_at"] = job["cancelled_at"]
        if job.get("cancellation_reason"):
            kwargs["cancellation_reason"] = job["cancellation_reason"]

        self.update_status(jobid, status, **kwargs)
