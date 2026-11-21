"""
Wire format between the host's proxy and the assistant: request model and Server-Sent Events.

Versioned as ``pslm-server/1`` because portable-slm's client will speak it once the core is extracted.
Event names: meta, reasoning, token, tool, proposal, warning, done, error (tool and proposal arrive
with the agent loop).
"""

from __future__ import annotations

import json
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, PrivateAttr

from src.assistant.core import PROTOCOL

MAX_QUESTION_CHARS = 4000
MAX_MESSAGE_CHARS = 20000
MAX_HISTORY_MESSAGES = 50


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(max_length=MAX_MESSAGE_CHARS)


class SourceRef(BaseModel):
    """One project source a request may draw on: a resource by path, an attachment by content hash."""

    kind: Literal["resource", "attachment"]
    id: str = Field(max_length=64)
    name: Optional[str] = Field(default=None, max_length=200)
    # Resource: a path under STORAGE_PATH (the proxy takes it from the project's own database rows).
    path: Optional[str] = Field(default=None, max_length=1024)
    # Attachment: SHA-256 of the file, registered under ``sid`` when it was uploaded.
    hash: Optional[str] = Field(default=None, min_length=64, max_length=64)


class SourcesMixin(BaseModel):
    sid: Optional[str] = Field(default=None, max_length=64)
    # The project may not be sent to a hosted model: only a local model may answer (never a hosted fallback).
    keep_local: bool = False
    sources: list[SourceRef] = Field(default_factory=list, max_length=40)
    # The resolved, searchable form of ``sources`` (or of ``document``), set by the route before the turn.
    _view: Any = PrivateAttr(default=None)

    @property
    def doc(self) -> Any:
        """What grounds the turn in an attached document: the indexed sources, else the raw text, else None."""
        if self._view is not None:
            return self._view
        return getattr(self, "document", None) or None

    def set_view(self, view: Any) -> None:
        self._view = view


class SuggestRequest(SourcesMixin):
    protocol: str = PROTOCOL
    model: Optional[str] = None
    task: str = Field(min_length=1, max_length=64)
    bundle: Optional[dict[str, Any]] = None
    field: Optional[str] = None
    # Text of an attachment the curator pointed at (already extracted under STORAGE_PATH).
    document: Optional[str] = Field(default=None, max_length=60000)


class ReviewRequest(SourcesMixin):
    protocol: str = PROTOCOL
    model: Optional[str] = None
    # "record": one agent turn over the whole record. "document": every fillable field checked against
    # the attached document. "variables": the open data file's variables labelled from a codebook
    # (both need ``document``).
    kind: Literal["record", "document", "variables"] = "record"
    # Which variable attributes a "variables" job proposes.
    attrs: list[Literal["label", "description", "question"]] = Field(default_factory=lambda: ["label"], max_length=3)
    # Opaque id of the person who started the job (the proxy sends a hash); caps their running jobs.
    owner: Optional[str] = Field(default=None, max_length=64)
    question: Optional[str] = Field(default=None, max_length=MAX_QUESTION_CHARS)
    bundle: Optional[dict[str, Any]] = None
    max_tokens: Optional[int] = Field(default=None, ge=1, le=16384)
    document: Optional[str] = Field(default=None, max_length=60000)


class ExtractRequest(BaseModel):
    protocol: str = PROTOCOL
    file_path: str = Field(min_length=1, max_length=1024)
    max_chars: int = Field(default=40000, ge=1000, le=200000)


class ChatRequest(SourcesMixin):
    protocol: str = PROTOCOL
    model: Optional[str] = Field(default=None, description="Registry alias; omitted means the default")
    question: str = Field(min_length=1, max_length=MAX_QUESTION_CHARS)
    messages: list[ChatMessage] = Field(default_factory=list, max_length=MAX_HISTORY_MESSAGES)
    mode: Optional[Literal["record", "app"]] = None
    # Host-specific context (for the editor: record id, section, snapshot, fields). The core treats it
    # as opaque and hands it to the host adapter, which validates it.
    bundle: Optional[dict[str, Any]] = None
    max_tokens: Optional[int] = Field(default=None, ge=1, le=16384)
    # Text of an attachment the curator pointed at; the host may use it as grounding.
    document: Optional[str] = Field(default=None, max_length=60000)


def sse(event: str, data: Any) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False, separators=(',', ':'))}\n\n"
