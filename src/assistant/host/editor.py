"""
Metadata Editor host adapter.

The PHP editor builds the bundle (after its own session and ACL checks) and sends it with each turn;
this service never calls back into the editor. Turns are grounded in the help text and the current
view; record data reaches the model through tools (host/tools.py), which it uses agentically.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from src.assistant.core.documents import attached_document
from src.assistant.core.host import HostContext
from src.assistant.core.protocol import ChatRequest
from src.assistant.core.retrieval import RetrievalService
from src.assistant.core.tools import Tool
from src.assistant.host.tools import AGENT_ADDENDUM, DOCUMENT_ADDENDUM, PROPOSE_ADDENDUM, build_editor_tools

logger = logging.getLogger(__name__)

APP_NAME = "Metadata Editor"
HELP_MAX_BYTES = 8192
DOCUMENT_MAX_BYTES = 12000


class EditorBundle(BaseModel):
    """What the editor's proxy sends about the page. Unknown keys are ignored, not rejected."""

    model_config = ConfigDict(extra="ignore")

    record_id: Optional[str] = Field(default=None, max_length=64)
    section: Optional[str] = Field(default=None, max_length=200)
    file_id: Optional[str] = Field(default=None, max_length=64)
    record: Optional[dict[str, Any]] = None
    # Declared fields, keyed by JSON Pointer: {label, value, found, help, in_template}. The proxy sends
    # only allowlisted pointers, so this map is the allowlist the tools enforce.
    fields: dict[str, Any] = Field(default_factory=dict, max_length=400)
    datafile: Optional[dict[str, Any]] = None
    datafiles: list[dict[str, Any]] = Field(default_factory=list, max_length=200)
    # The page opted into proposals (the browser panel's data-fill).
    fill: bool = False


def parse_bundle(raw: Optional[dict[str, Any]]) -> EditorBundle:
    if not raw:
        return EditorBundle()
    try:
        return EditorBundle.model_validate(raw)
    except ValidationError:
        logger.warning("assistant: ignoring malformed editor bundle", exc_info=True)
        return EditorBundle()


class EditorHost:
    app_name = APP_NAME

    def __init__(self, help_path: Optional[Path], retrieval: Optional[RetrievalService] = None) -> None:
        self._help_path = help_path
        self._help_cache: tuple[float, str] | None = None
        self.retrieval = retrieval
        # id(raw bundle) -> (raw, parsed): one turn asks for its bundle from mode, context and tools.
        self._parsed: dict[int, tuple[Any, EditorBundle]] = {}

    def _bundle(self, request: ChatRequest) -> EditorBundle:
        raw = request.bundle
        hit = self._parsed.get(id(raw))
        if hit is not None and hit[0] is raw:
            return hit[1]
        parsed = parse_bundle(raw)
        if len(self._parsed) >= 8:  # a few turns in flight; each entry holds a whole bundle
            self._parsed.clear()
        self._parsed[id(raw)] = (raw, parsed)
        return parsed

    def mode(self, request: ChatRequest) -> str:
        if request.mode:
            return request.mode
        return "record" if self._bundle(request).record_id else "app"

    async def context(self, request: ChatRequest) -> HostContext:
        bundle = self._bundle(request)
        parts: list[str] = []
        sources: list[str] = []
        citations: list[str] = []
        help_text = self._help()
        if help_text:
            parts.append(f"## About this application\n\n{help_text}")
            sources.append("help")
        if bundle.section:
            parts.append(f"## Current view\n\nThe user is on the \"{bundle.section}\" page.")
            sources.append("view")
        doc = request.doc
        if doc:
            # Sources are never sent whole: the passages this question matches (labelled with source and
            # page), and the rest stays reachable through the search_document tool.
            view = attached_document(doc)
            document, cut = await asyncio.to_thread(view.excerpt, request.question, DOCUMENT_MAX_BYTES)
            heading = "Project sources (passages the question matches)" if cut else "Attached document"
            parts.append(f"## {heading}, provided by the curator\n\n{document}")
            sources.append("document:excerpt" if cut else "document")
            if cut and hasattr(view, "search"):
                for chunk in await asyncio.to_thread(view.search, request.question, 4):
                    label = view.label(chunk) if hasattr(view, "label") else f"passage {chunk.index + 1}"
                    if label not in citations:
                        citations.append(label)
        if self.retrieval is not None:
            found, info = await self.retrieval.search(request.question)
            if found:
                parts.append(f"## From the editor's documentation\n\n{found}")
                sources.append("docs" if info.get("semantic") else "docs:keyword")
                citations.extend(h for h in (info.get("hits") or []) if h not in citations)
        return HostContext(text="\n\n".join(parts), sources=sources, citations=citations)

    def tools(self, request: ChatRequest) -> list[Tool]:
        bundle = self._bundle(request)
        if self.mode(request) != "record":
            return []
        search = self.retrieval.search if self.retrieval is not None else None
        return build_editor_tools(bundle, fill=bundle.fill, search_docs=search, document=request.doc)

    def system_addendum(self, request: ChatRequest, tools: list[Tool]) -> str:
        names = {t.name for t in tools}
        parts = []
        if names & {"list_fields", "read_fields", "find_variables"}:
            parts.append(AGENT_ADDENDUM)
        if names & {"propose_field_value", "propose_datafile_description"}:
            parts.append(PROPOSE_ADDENDUM)
        if "search_document" in names:
            parts.append(DOCUMENT_ADDENDUM)
        return "\n".join(parts)

    def _help(self) -> str:
        path = self._help_path
        if path is None or not path.is_file():
            return ""
        mtime = path.stat().st_mtime
        if self._help_cache and self._help_cache[0] == mtime:
            return self._help_cache[1]
        raw = path.read_bytes()[:HELP_MAX_BYTES].decode("utf-8", errors="ignore").strip()
        self._help_cache = (mtime, raw)
        return raw
