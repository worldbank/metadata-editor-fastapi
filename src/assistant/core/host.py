"""
The extension point a host application implements. The core calls it; it never calls the core's caller.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from src.assistant.core.protocol import ChatRequest
from src.assistant.core.tools import Tool


@dataclass
class HostContext:
    """Host text appended after the assistant's own description, plus where it came from."""

    text: str = ""
    sources: list[str] = field(default_factory=list)
    # Human-readable labels of what was retrieved (e.g. "Collaborators — editor usage guide"), so a
    # client can say where an answer was grounded without parsing the prompt.
    citations: list[str] = field(default_factory=list)


class HostAdapter(Protocol):
    app_name: str

    def mode(self, request: ChatRequest) -> str:
        """"record" when a specific record is open, "app" otherwise."""
        ...

    async def context(self, request: ChatRequest) -> HostContext:
        """Text grounding this turn. Must be bounded; the core caps it again."""
        ...

    def tools(self, request: ChatRequest) -> list[Tool]:
        """Host tools for this turn (built-in utilities are added by the core)."""
        ...

    def system_addendum(self, request: ChatRequest, tools: list[Tool]) -> str:
        """Extra system-prompt sentences the host's tools need, appended after the core's own."""
        ...


class NullHost:
    """A host with no context of its own: a general-purpose assistant."""

    app_name = "this application"

    def mode(self, request: ChatRequest) -> str:
        return request.mode or "none"

    async def context(self, request: ChatRequest) -> HostContext:
        return HostContext()

    def tools(self, request: ChatRequest) -> list[Tool]:
        return []

    def system_addendum(self, request: ChatRequest, tools: list[Tool]) -> str:
        return ""
