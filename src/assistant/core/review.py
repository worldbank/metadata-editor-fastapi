"""
Whole-record review as a job: one agent turn with the review question, collected into a result the
editor polls through ``GET /jobs/{id}``.

The turn itself is the ordinary chat path (``run_turn``), so a review reads and proposes with the same
tools, budgets and guards as a chat question. This module only collects what that stream produced.
"""

from __future__ import annotations

import json
from typing import Any, Callable

from src.assistant.core.chat import TurnLimits, run_turn
from src.assistant.core.config import ModelSpec
from src.assistant.core.host import HostAdapter
from src.assistant.core.llm import Gateway
from src.assistant.core.protocol import ChatRequest

REVIEW_QUESTION = (
    "Review this record: find required and recommended fields that are empty, unclear or inconsistent, "
    "propose values where the project's own content supports them, and summarize what you found."
)
REVIEW_WITH_DOCUMENT = (
    "Review this record against the attached document: find required and recommended fields that are empty, "
    "unclear or inconsistent, propose values where the document or the project's own content supports them, "
    "and summarize what you found."
)


def _parse_event(block: str) -> tuple[str, Any] | None:
    name, data = None, ""
    for line in block.splitlines():
        if line.startswith("event: "):
            name = line[7:]
        elif line.startswith("data: "):
            data = line[6:]
    if name is None:
        return None
    try:
        return name, json.loads(data) if data else None
    except json.JSONDecodeError:
        return None


async def run_review(
    gateway: Gateway,
    host: HostAdapter,
    request: ChatRequest,
    spec: ModelSpec,
    limits: TurnLimits,
    *,
    is_cancelled: Callable[[], bool] | None = None,
) -> dict[str, Any]:
    """Run a review turn and return what the job stores: answer, proposals, steps and timings."""
    proposals: list[dict[str, Any]] = []
    steps: list[dict[str, Any]] = []
    done: dict[str, Any] | None = None
    error: dict[str, Any] | None = None
    async for chunk in run_turn(gateway, host, request, spec, limits, is_cancelled=is_cancelled):
        for block in chunk.split("\n\n"):
            parsed = _parse_event(block.strip())
            if parsed is None:
                continue
            name, data = parsed
            if name == "proposal" and isinstance(data, dict):
                proposals.append(data)
            elif name == "tool" and isinstance(data, dict) and data.get("stage") in {"call", "result", "failed"}:
                steps.append(data)
            elif name == "done" and isinstance(data, dict):
                done = data
            elif name == "error" and isinstance(data, dict):
                error = data
    if error is not None:
        return {"text": "", "proposals": proposals, "steps": steps, "error": error}
    done = done or {}
    return {
        "text": done.get("text", ""),
        "proposals": proposals,
        "steps": steps,
        "tool_calls": done.get("tool_calls"),
        "completeness": done.get("completeness"),
        "usage": done.get("usage"),
        "timings": done.get("timings"),
        "model": done.get("model"),
        "served_by": done.get("served_by"),
        "citations": done.get("citations"),
    }
