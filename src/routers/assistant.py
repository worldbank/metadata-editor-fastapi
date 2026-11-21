"""
Assistant routes. Internal like every route in this service: the editor's PHP proxy calls them after
its own session and ACL checks; the browser never does.
"""

from __future__ import annotations

import asyncio
import datetime
import json
import logging
import os
import time
import uuid

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from src.assistant.core import PROTOCOL
from src.assistant.core.chat import run_turn
from src.assistant.core.documents import ExtractError, extract_text
from src.assistant.core.knowledge import is_hash
from pydantic import BaseModel, Field

from src.assistant.core.protocol import ChatRequest, ExtractRequest, ReviewRequest, SourceRef, SuggestRequest, sse
from src.assistant.core.review import REVIEW_QUESTION, REVIEW_WITH_DOCUMENT, run_review
from src.assistant.core.tasks import parse_suggestion
from src.assistant.host.editor import parse_bundle
from src.assistant.host.tasks import datafile_description_messages, resource_field_messages
from src.assistant.host.tools import _preview, bundle_corpus, ungrounded_tokens

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/assistant", tags=["Assistant"])

SSE_HEADERS = {
    "Cache-Control": "no-cache",
    # Stops nginx-style proxies from buffering the stream; harmless elsewhere.
    "X-Accel-Buffering": "no",
}


def _gateway(request: Request):
    gateway = getattr(request.app, "assistant_gateway", None)
    if gateway is None:
        raise HTTPException(
            status_code=503,
            detail=getattr(request.app, "assistant_config_error", None) or "assistant is not configured",
        )
    return gateway


class _Disconnect:
    """
    Watches the client connection from a side task, so a turn can stop for a caller that left without
    polling the connection on every token. Callable: True once the client is gone.
    """

    POLL_SEC = 0.5

    def __init__(self, request: Request) -> None:
        self._request = request
        self.gone = False
        self._task: asyncio.Task | None = None

    def __call__(self) -> bool:
        return self.gone

    async def __aenter__(self) -> "_Disconnect":
        self._task = asyncio.create_task(self._watch())
        return self

    async def __aexit__(self, *exc) -> None:
        if self._task is not None:
            self._task.cancel()

    async def _watch(self) -> None:
        try:
            while not self.gone:
                if await self._request.is_disconnected():
                    self.gone = True
                    return
                await asyncio.sleep(self.POLL_SEC)
        except Exception:  # noqa: BLE001 - a broken receive channel means the client is gone
            self.gone = True


async def _acquire(sem: asyncio.Semaphore, gone) -> bool:
    """Wait for a model slot; give up (False) once the caller has gone, instead of queueing for nobody."""
    while True:
        try:
            await asyncio.wait_for(sem.acquire(), timeout=1.0)
            return True
        except asyncio.TimeoutError:
            if gone():
                return False


def _local_spec(registry, model: str | None):
    """The model for a project that must stay on this server: the one asked for if local, else the first local one."""
    try:
        spec = registry.resolve(model)
        if spec.local:
            return spec
    except KeyError:
        pass
    local = [m for m in registry.usable() if m.local]
    if not local:
        raise HTTPException(
            status_code=409,
            detail="This project is set to keep its data on this server, and no local model is configured. "
            "Ask an administrator to set one up, or change the project's setting.",
        )
    return next((m for m in local if m.alias == registry.default), local[0])


def _check_request(request: Request, protocol: str, model: str | None, keep_local: bool = False):
    """The gateway and model spec for a request, or the HTTP error every assistant route answers with."""
    gateway = _gateway(request)
    if protocol != PROTOCOL:
        raise HTTPException(status_code=400, detail=f"unsupported protocol {protocol!r}; expected {PROTOCOL}")
    if keep_local:
        spec = _local_spec(gateway.registry, model)
    else:
        try:
            spec = gateway.registry.resolve(model)
        except KeyError:
            raise HTTPException(status_code=400, detail=f"unknown or unavailable model {model!r}")
    if request.app.assistant_inflight >= request.app.assistant_max_inflight:
        raise HTTPException(status_code=429, detail="too many assistant requests in flight")
    return gateway, spec


class SyncFile(BaseModel):
    id: str = Field(max_length=64)
    kind: str = Field(default="resource", pattern="^(resource|attachment)$")
    name: str = Field(default="", max_length=200)
    path: str = Field(min_length=1, max_length=1024)


class SyncRequest(BaseModel):
    protocol: str = PROTOCOL
    sid: str = Field(min_length=1, max_length=64)
    files: list[SyncFile] = Field(default_factory=list, max_length=300)


class SidRequest(BaseModel):
    protocol: str = PROTOCOL
    sid: str = Field(min_length=1, max_length=64)


class RemoveRequest(SidRequest):
    hashes: list[str] = Field(default_factory=list, max_length=100)


def _knowledge(request: Request):
    store = getattr(request.app, "assistant_knowledge", None)
    if store is None:
        raise HTTPException(status_code=503, detail="the project knowledge index is not available")
    return store


async def _resolve_document(request: Request, body, document):
    """
    What grounds a request in the project's files: the searchable view over its ``sources``, else the raw
    ``document`` text, else None. A source that cannot be read is skipped; if none can, say so.
    """
    refs = list(getattr(body, "sources", None) or [])
    store = getattr(request.app, "assistant_knowledge", None)
    if not refs or store is None:
        text = document if document is not None else getattr(body, "document", None)
        return text if text and str(text).strip() else None
    from pathlib import Path

    from src.utils.path_security import resolve_safe_path_http

    parts = []
    skipped: list[str] = []
    sid = (body.sid or "").strip()
    for ref in sorted(refs, key=lambda r: r.kind != "resource"):  # resources first: they win a duplicate
        label = ref.name or ref.id
        try:
            if ref.kind == "resource":
                if not ref.path:
                    skipped.append(label)
                    continue
                path = resolve_safe_path_http(ref.path, label="source path")
                name = ref.name or Path(path).name
                parts.append((name, await store.ensure_file(path, name)))
            else:
                if not (sid and is_hash(ref.hash) and store.is_registered(sid, ref.hash)):
                    skipped.append(label)
                    continue
                entry = await asyncio.to_thread(store.get, ref.hash)
                if entry is None:
                    skipped.append(label)
                    continue
                parts.append((store.attachment_name(sid, ref.hash) or ref.name or ref.id, entry))
        except (ExtractError, HTTPException) as exc:
            logger.info("assistant: source %s skipped: %s", label, getattr(exc, "detail", exc))
            skipped.append(label)
    view = store.view(parts)
    if view is None:
        text = document if document is not None else getattr(body, "document", None)
        if text and str(text).strip():
            return text
        raise HTTPException(status_code=400, detail="None of the selected sources could be read. Add them again or choose others.")
    if skipped:
        logger.info("assistant: %d of %d sources unavailable for this request", len(skipped), len(refs))
    return view


@router.get("/status")
async def assistant_status(request: Request):
    gateway = getattr(request.app, "assistant_gateway", None)
    if gateway is None:
        return {
            "enabled": False,
            "protocol": PROTOCOL,
            "error": getattr(request.app, "assistant_config_error", None),
        }
    registry = gateway.registry
    return {
        "enabled": True,
        "protocol": PROTOCOL,
        "default": registry.default,
        "local_only": registry.local_only,
        "models": len(registry.usable()),
        "inflight": request.app.assistant_inflight,
        "retrieval": retrieval.status() if (retrieval := getattr(request.app, "assistant_retrieval", None)) else None,
        "knowledge": store.status() if (store := getattr(request.app, "assistant_knowledge", None)) else None,
    }


@router.get("/models")
async def assistant_models(request: Request):
    """Aliases a request may choose, with whether each can be reached right now."""
    gateway = _gateway(request)
    registry = gateway.registry
    out = []
    for spec in registry.usable():
        item = spec.public()
        item["available"] = await gateway.available(spec)
        item["default"] = spec.alias == registry.default
        out.append(item)
    return {"protocol": PROTOCOL, "local_only": registry.local_only, "models": out}


@router.post("/chat/stream")
async def assistant_chat_stream(body: ChatRequest, request: Request):
    app = request.app
    gateway, spec = _check_request(request, body.protocol, body.model, body.keep_local)
    body.set_view(await _resolve_document(request, body, None))

    async def events():
        # Counted from the first iteration: Starlette never starts the body of a response whose client
        # left before the headers went out, and a count taken earlier would never come back down.
        app.assistant_inflight += 1
        try:
            async with _Disconnect(request) as gone:
                if app.assistant_sem.locked():
                    # Every slot is busy with other users' turns; say so instead of a silent spinner.
                    yield sse("queued", {"detail": "waiting for a free model slot"})
                if not await _acquire(app.assistant_sem, gone):
                    return
                try:
                    async for event in run_turn(
                        gateway,
                        app.assistant_host,
                        body,
                        spec,
                        app.assistant_limits,
                        is_cancelled=gone,
                    ):
                        yield event
                finally:
                    app.assistant_sem.release()
        finally:
            app.assistant_inflight -= 1

    return StreamingResponse(events(), media_type="text/event-stream", headers=SSE_HEADERS)


@router.post("/suggest")
async def assistant_suggest(body: SuggestRequest, request: Request):
    app = request.app
    gateway, spec = _check_request(request, body.protocol, body.model, body.keep_local)
    if body.task not in {"datafile-description", "resource-field"}:
        raise HTTPException(status_code=400, detail=f"unsupported task {body.task!r}")
    if body.task == "resource-field":
        # The pointer must be one this project's form declares and can accept a single value for;
        # otherwise the model would be drafting into a field that does not exist.
        fields = parse_bundle(body.bundle or {}).fields
        field = fields.get(body.field or "")
        if not isinstance(field, dict):
            raise HTTPException(
                status_code=400,
                detail=f"{body.field!r} is not a field of this project; use a pointer the form declares",
            )
        if field.get("in_template") is False or not field.get("fillable", True):
            raise HTTPException(
                status_code=400,
                detail=f"{body.field!r} is a list or table field; a single draft cannot fill it",
            )
    doc = await _resolve_document(request, body, None)
    if body.task == "datafile-description":
        messages = await asyncio.to_thread(datafile_description_messages, body.bundle or {}, doc)
    else:
        messages = await asyncio.to_thread(resource_field_messages, body.bundle or {}, body.field, doc)
    app.assistant_inflight += 1
    try:
        async with _Disconnect(request) as gone:
            if not await _acquire(app.assistant_sem, gone):
                raise HTTPException(status_code=499, detail="the client closed the request")
            try:
                # Room for the model's own reasoning before the JSON: LFM2.5 spends a few hundred tokens
                # thinking even with greedy sampling, and a cap below that returns an empty reply.
                text = await gateway.complete(
                    spec, messages, max_tokens=min(2048, spec.max_tokens), temperature=0.1, is_cancelled=gone
                )
            finally:
                app.assistant_sem.release()
    finally:
        app.assistant_inflight -= 1
    parsed = parse_suggestion(text)
    bundle = parse_bundle(body.bundle or {})
    if body.task == "resource-field":
        field = bundle.fields.get(body.field or "") if isinstance(bundle.fields, dict) else None
        field = field if isinstance(field, dict) else {}
        allowed = field.get("options") or field.get("suggestions") or []
        if allowed and parsed["suggestion"]:
            chosen = next(
                (opt for opt in allowed if str(opt).strip().casefold() == parsed["suggestion"].strip().casefold()),
                None,
            )
            if chosen is None:
                return {
                    "protocol": PROTOCOL,
                    "task": body.task,
                    "model": spec.alias,
                    "suggestion": "",
                    "reason": "",
                    "message": f"The draft must be one of: {', '.join(str(o) for o in allowed[:30])}. Try again.",
                }
            parsed["suggestion"] = chosen
    corpus = bundle_corpus(bundle, extra=doc or "")
    ungrounded = ungrounded_tokens(parsed["suggestion"], corpus)
    if ungrounded:
        # Same guard as the propose tool: a draft may not carry a date, number or identifier the
        # project and the attached document never state.
        return {
            "protocol": PROTOCOL,
            "task": body.task,
            "model": spec.alias,
            "suggestion": "",
            "reason": "",
            "message": f"The draft mentioned {', '.join(ungrounded)}, which the project and the document never state. Try again.",
        }
    # The saved value the draft would replace, so the panel can show both and label the card.
    if body.task == "resource-field":
        current = (bundle.fields.get(body.field or "") or {}).get("value")
    else:
        current = (bundle.datafile or {}).get("description")
    return {"protocol": PROTOCOL, "task": body.task, "model": spec.alias, "current": _preview(current, 400), **parsed}


def _new_jobid() -> str:
    # The time keeps ids sortable like the service's other jobs; the random part keeps two reviews
    # started in the same instant apart and makes an id impossible to guess.
    return f"job-{time.time():.6f}-{uuid.uuid4().hex[:12]}"


async def _take_job_slot(app, cancelled):
    """
    Wait for one of the job slots (reviews and fills may use only some of the model slots, so a chat
    question always has one free). Returns the semaphore to release, ``None`` when there is no limit,
    or ``False`` when the job was cancelled while it waited.
    """
    sem = getattr(app, "assistant_job_sem", None)
    if sem is None:
        return None
    while True:
        try:
            await asyncio.wait_for(sem.acquire(), timeout=1.0)
            return sem
        except asyncio.TimeoutError:
            if cancelled():
                return False


async def _run_assistant_job(app, jobid: str, run) -> None:
    """
    Job body for a review or a document fill, run as its own task: the service's FIFO queue runs one
    job at a time, and an LLM review there would hold up every other user's review and data job.
    """
    job = app.jobs[jobid]
    store = getattr(app, "job_store", None)

    def persist() -> None:
        # The result file already outlives a restart; this lets GET /jobs/{id} find the job again.
        if store is None:
            return
        try:
            store.sync_from_memory(job)
        except Exception:  # noqa: BLE001 - the in-memory job still answers; a store failure must not lose it
            logger.exception("assistant: could not store job %s", jobid)

    def cancelled() -> bool:
        # In memory only: DELETE /jobs/{id} flips this dict, and the check runs on every streamed token.
        return job.get("status") == "cancelled"

    def progress(done: int, total: int) -> None:
        job["info"]["progress"] = {"done": done, "total": total}

    job_slot = None
    try:
        job_slot = await _take_job_slot(app, cancelled)
        result = {} if job_slot is False else await run(cancelled, progress)
    except Exception as exc:  # noqa: BLE001 - the job reports its own failure
        logger.exception("assistant: job %s failed", jobid)
        job["status"] = "error"
        job["error"] = f"Review failed: {type(exc).__name__}"
        job["completed_at"] = datetime.datetime.now().isoformat()
        persist()
        return
    finally:
        if job_slot:
            job_slot.release()
        app.assistant_inflight -= 1
        getattr(app, "assistant_jobs", {}).pop(jobid, None)
        getattr(app, "assistant_job_owner", {}).pop(jobid, None)

    now = datetime.datetime.now().isoformat()
    if cancelled():
        job["cancelled_at"] = job.get("cancelled_at") or now
        job["completed_at"] = now
        persist()
        return
    if result.get("error"):
        job["status"] = "error"
        job["error"] = str(result["error"].get("message") or "The model did not answer.")
        job["completed_at"] = now
        persist()
        return

    with open(os.path.join("jobs", f"{jobid}.json"), "w", encoding="utf-8") as fh:
        json.dump(result, fh, ensure_ascii=False)
    job["status"] = "done"
    job["completed_at"] = now
    persist()
    logger.info(
        "assistant job=%s type=%s model=%s proposals=%d steps=%d tool_calls=%s total_ms=%s",
        jobid,
        job.get("jobtype"),
        result.get("model"),
        len(result.get("proposals") or []),
        len(result.get("steps") or []),
        result.get("tool_calls"),
        (result.get("timings") or {}).get("total_ms"),
    )


@router.post("/review/jobs")
async def assistant_review_job(body: ReviewRequest, request: Request):
    """
    Start a whole-record review (kind "record") or a fill from the attached document (kind "document").
    The client polls ``GET /jobs/{job_id}`` (``info.progress`` while a fill runs) and cancels with DELETE.
    """
    app = request.app
    gateway, spec = _check_request(request, body.protocol, body.model, body.keep_local)
    if not hasattr(app, "jobs"):
        raise HTTPException(status_code=503, detail="the job store is not available")
    document = await _resolve_document(request, body, None)
    if body.kind in {"document", "variables"} and not document:
        raise HTTPException(status_code=400, detail="add a source to fill from it")
    # A review of a record that may be filled, against an attached document, is the same work as a fill:
    # check every fillable field against the document. The free-roaming agent spent its tool budget reading
    # and proposed 0 to 7 fields from run to run; the field-by-field pass proposes the same ones each time.
    # Without a document (or for a viewer, who cannot fill) the agent review stays as it was.
    kind = body.kind
    if kind == "record" and document and (body.bundle or {}).get("fill"):
        kind = "document"
    owners = getattr(app, "assistant_job_owner", None)
    if owners is None:
        owners = app.assistant_job_owner = {}
    if body.owner:
        limit = getattr(app, "assistant_max_jobs_per_owner", 2)
        if sum(1 for o in owners.values() if o == body.owner) >= limit:
            raise HTTPException(
                status_code=429,
                detail=f"You already have {limit} assistant jobs running. Wait for one to finish, or stop it first.",
            )

    if kind == "variables":
        from src.assistant.host.varfill import run_variable_fill

        question = "Label variables from the attached document"

        async def run(cancelled, progress):
            return await run_variable_fill(
                gateway, spec, body.bundle, document, attrs=body.attrs, slot=app.assistant_sem,
                is_cancelled=cancelled, progress=progress,
            )

    elif kind == "document":
        from src.assistant.host.docfill import run_document_fill

        question = "Fill fields from the attached document" if body.kind == "document" else "Review this record against the attached document"

        async def run(cancelled, progress):
            return await run_document_fill(
                gateway, spec, body.bundle, document, slot=app.assistant_sem, is_cancelled=cancelled, progress=progress
            )

    else:
        chat = ChatRequest(
            protocol=PROTOCOL,
            model=spec.alias,
            question=body.question or (REVIEW_WITH_DOCUMENT if document else REVIEW_QUESTION),
            bundle=body.bundle,
            max_tokens=body.max_tokens,
            document=body.document,
        )
        chat.set_view(document)
        question = chat.question

        async def run(cancelled, progress):
            # Queued until a model slot frees up; processing from then on.
            while True:
                if cancelled():
                    return {}
                try:
                    await asyncio.wait_for(app.assistant_sem.acquire(), timeout=1.0)
                    break
                except asyncio.TimeoutError:
                    continue
            try:
                app.jobs[jobid]["status"] = "processing"
                return await run_review(
                    gateway, app.assistant_host, chat, spec, app.assistant_limits, is_cancelled=cancelled
                )
            finally:
                app.assistant_sem.release()

    jobid = _new_jobid()
    now = datetime.datetime.now().isoformat()
    jobtype = {"document": "assistant-document-fill", "variables": "assistant-variable-fill"}.get(body.kind, "assistant-review")
    app.jobs[jobid] = {
        "jobid": jobid,
        "jobtype": jobtype,
        "status": "queued" if kind == "record" else "processing",
        "created_at": now,
        "completed_at": None,
        "last_accessed": now,
        "info": {"model": spec.alias, "question": question},
    }
    app.assistant_inflight += 1
    if body.owner:
        owners[jobid] = body.owner
    store = getattr(app, "job_store", None)
    if store is not None:
        try:
            # Stored as processing even while it waits for a slot: after a restart the store marks
            # processing jobs interrupted, and there is nothing to resume (the document and the
            # record travelled with this request, not through the store).
            store.save_queued_job({**app.jobs[jobid], "status": "processing"})
        except Exception:  # noqa: BLE001
            logger.exception("assistant: could not store job %s", jobid)
    if not hasattr(app, "assistant_jobs"):
        app.assistant_jobs = {}
    # Kept referenced until it finishes: the event loop holds tasks only weakly.
    app.assistant_jobs[jobid] = asyncio.create_task(_run_assistant_job(app, jobid, run), name=f"assistant-{jobid}")
    return {"protocol": PROTOCOL, "job_id": jobid, "status": app.jobs[jobid]["status"]}


@router.post("/documents/extract")
async def assistant_documents_extract(body: ExtractRequest, request: Request):
    """Text of an attachment already under STORAGE_PATH (the PHP proxy checks the project ACL first)."""
    _gateway(request)
    if body.protocol != PROTOCOL:
        raise HTTPException(status_code=400, detail=f"unsupported protocol {body.protocol!r}; expected {PROTOCOL}")
    # Imported here, not at module scope: this route tree stays importable without the app's
    # STORAGE_PATH startup check (the assistant is optional).
    from src.utils.path_security import resolve_safe_path_http

    path = resolve_safe_path_http(body.file_path, label="file_path")
    if not os.path.isfile(path):
        raise HTTPException(status_code=400, detail="only a regular file can be read")
    try:
        # pypdf on an 8 MB file takes seconds of CPU; off the event loop, other users' streams keep flowing.
        result = await asyncio.to_thread(extract_text, path, body.max_chars)
    except ExtractError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"protocol": PROTOCOL, "file_path": path, **result}


@router.post("/knowledge/sync")
async def knowledge_sync(body: SyncRequest, request: Request):
    """
    Index the project's files (read in full, cached by content hash) and report each one's state. Called
    by the PHP proxy with the paths it took from the project's own rows, so this is also the status poll.
    """
    _gateway(request)
    store = _knowledge(request)
    if body.protocol != PROTOCOL:
        raise HTTPException(status_code=400, detail=f"unsupported protocol {body.protocol!r}; expected {PROTOCOL}")
    from pathlib import Path

    from src.utils.path_security import resolve_safe_path_http

    out = []
    for f in body.files:
        name = f.name or Path(f.path).name
        item = {"id": f.id, "kind": f.kind, "name": name}
        try:
            path = resolve_safe_path_http(f.path, label="path")
            if not os.path.isfile(path):
                raise ExtractError("The file does not exist.")
            entry = await store.ensure_file(path, name)
            if f.kind == "attachment":
                store.register_attachment(body.sid, entry.hash, name)
                item["id"] = f.id if f.id.startswith("att-") else store.attachment_id(entry.hash)
            out.append(store.describe(entry, id=item["id"], kind=f.kind, name=name))
        except ExtractError as exc:
            out.append({**item, "state": "failed", "error": str(exc)})
        except HTTPException as exc:
            out.append({**item, "state": "failed", "error": str(exc.detail)})
    return {"protocol": PROTOCOL, "sources": out}


@router.post("/knowledge/list")
async def knowledge_list(body: SidRequest, request: Request):
    """The attachments registered under a project, with their current state."""
    _gateway(request)
    store = _knowledge(request)
    out = []
    for digest, info in store.attachments(body.sid):
        entry = await asyncio.to_thread(store.get, digest)
        if entry is None:
            out.append({"id": info["id"], "hash": digest, "kind": "attachment", "name": info["name"], "state": "failed",
                        "error": "No longer indexed. Add the file again."})
        else:
            out.append(store.describe(entry, id=info["id"], kind="attachment", name=info["name"]))
    return {"protocol": PROTOCOL, "sources": out}


@router.post("/knowledge/remove")
async def knowledge_remove(body: RemoveRequest, request: Request):
    _gateway(request)
    store = _knowledge(request)
    hashes = [h for h in body.hashes if is_hash(h)]
    return {"protocol": PROTOCOL, "removed": store.remove(body.sid, hashes)}
