"""
One chat turn as a stream of protocol events.

Tools run through the agent loop; greeting turns get a plain round. Every turn ends with exactly one
``done`` or ``error`` event, so a client can always tell a finished reply from a dropped connection.
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass
from typing import AsyncIterator, Callable

from src.assistant.core import PROTOCOL
from src.assistant.core.agent import AgentResult, run_agent
from src.assistant.core.completeness import describe_incomplete
from src.assistant.core.config import ModelSpec
from src.assistant.core.host import HostAdapter
from src.assistant.core.llm import Gateway, count_tokens
from src.assistant.core.prompt import (
    DIRECT_REPLY_NOTE,
    build_messages,
    byte_cap,
    compose_context,
    routes_direct,
    system_prompt,
    tool_instructions,
)
from src.assistant.core.protocol import ChatRequest, sse
from src.assistant.core.tools import BUILTIN_TOOL_NAMES, Tool, builtin_tools

logger = logging.getLogger(__name__)


@dataclass
class TurnLimits:
    max_history: int = 6
    context_max_bytes: int = 16384


def resolve_call_mode(spec: ModelSpec) -> str:
    """
    "native" (OpenAI tool_calls) or a text protocol. An explicit ``tool_mode`` wins; otherwise LiteLLM's
    capability table decides, and a model it doesn't know (most local ones) gets the text protocol,
    which every model can follow.
    """
    text = spec.text_protocol or "hermes"
    if spec.tool_mode == "native":
        return "native"
    if spec.tool_mode == "text":
        return text
    try:
        import litellm

        if litellm.supports_function_calling(model=spec.provider_model):
            return "native"
    except Exception:  # noqa: BLE001 - unknown model: fall through to the text protocol
        pass
    return text


# LiteLLM's router appends its whole fallback table to the provider's error ("No fallback model group
# found for original model_group=... Fallbacks=[...]. Received Model Group=..."). That names every
# configured model, hosted ones included, to a curator whose project is kept on local models.
_ROUTER_SUFFIX = re.compile(r"\s*(No fallback model group found|Fallbacks=\[|Received Model Group=|Available Model Group Fallbacks=).*$", re.S)


def _error_message(exc: Exception) -> str:
    # Provider errors can be long and include request details; the class name and first line are enough
    # for the panel. The full error goes to the server log.
    first = str(exc).strip().splitlines()[0] if str(exc).strip() else ""
    first = _ROUTER_SUFFIX.sub("", first).rstrip(" .")
    return f"{type(exc).__name__}: {first[:300]}" if first else type(exc).__name__


async def run_turn(
    gateway: Gateway,
    host: HostAdapter,
    request: ChatRequest,
    spec: ModelSpec,
    limits: TurnLimits,
    *,
    is_cancelled: Callable[[], bool] | None = None,
) -> AsyncIterator[str]:
    started = time.monotonic()
    mode = host.mode(request)
    direct = routes_direct(request.question)

    host_ctx = await host.context(request)
    host_text, host_capped = byte_cap(host_ctx.text, limits.context_max_bytes)
    context = compose_context(spec, host_text)
    system = system_prompt(host.app_name, mode)
    # Greeting and identity turns get no tools: small models over-trigger them on exactly these.
    tools: list[Tool] = [] if direct else [*builtin_tools(), *host.tools(request)]
    call_mode = resolve_call_mode(spec)
    if tools:
        system = f"{system}\n{tool_instructions(tools, call_mode, BUILTIN_TOOL_NAMES)}"
        addendum = host.system_addendum(request, tools)
        if addendum:
            system = f"{system}\n{addendum}"
    if direct:
        system = f"{system}\n\n{DIRECT_REPLY_NOTE}"

    max_tokens = min(request.max_tokens or spec.max_tokens, spec.max_tokens)
    assembled = build_messages(
        spec,
        system=system,
        context=context,
        history=[m.model_dump() for m in request.messages],
        question=request.question,
        max_history=limits.max_history,
        max_tokens=max_tokens,
        count=count_tokens,
    )

    yield sse(
        "meta",
        {
            "protocol": PROTOCOL,
            "model": spec.alias,
            "label": spec.label,
            "local": spec.local,
            "mode": mode,
            "direct": direct,
            "sources": host_ctx.sources,
            "citations": host_ctx.citations,
            "tools": [t.name for t in tools],
        },
    )
    if host_capped:
        yield sse("warning", {"kind": "context_capped", "detail": f"host context cut to {limits.context_max_bytes} bytes"})
    if assembled.dropped_messages or assembled.context_truncated:
        yield sse(
            "warning",
            {
                "kind": "window_fit",
                "detail": {
                    "dropped_messages": assembled.dropped_messages,
                    "context_truncated": assembled.context_truncated,
                },
            },
        )

    first_token_at: float | None = None
    reasoning_chars = 0
    result: AgentResult | None = None
    try:
        async for event, data in run_agent(
            gateway,
            spec,
            assembled.messages,
            tools,
            call_mode=call_mode,
            max_tokens=max_tokens,
            # Greedy when tools are offered, as in the browser: a sampled tool call is a malformed one.
            temperature=0.0 if tools else spec.temperature,
            count=count_tokens,
            repair=bool(tools),
            max_rounds=spec.max_rounds,
            max_tool_calls=spec.max_tool_calls,
            max_calls_per_round=spec.max_calls_per_round,
            is_cancelled=is_cancelled,
        ):
            if event == "result":
                result = data
                break
            if event in ("token", "reasoning"):
                if first_token_at is None:
                    first_token_at = time.monotonic()
                if event == "reasoning":
                    reasoning_chars += len(data["text"])
            yield sse(event, data)
    except Exception as exc:  # noqa: BLE001 - every turn must end with an event the client understands
        logger.warning("assistant: model call failed (%s)", spec.alias, exc_info=True)
        gateway.mark_failed(spec)
        yield sse("error", {"code": "model_error", "message": _error_message(exc)})
        return
    assert result is not None

    if result.finish_reason == "length":
        yield sse("warning", {"kind": "token_cap", "detail": f"reply stopped at max_tokens={max_tokens}"})
    if not result.verdict.ok and result.finish_reason != "cancelled":
        yield sse(
            "warning",
            {
                "kind": "incomplete",
                "reason": result.verdict.reason,
                "evidence": result.verdict.evidence,
                "detail": describe_incomplete(result.verdict),
            },
        )

    ended = time.monotonic()
    usage = result.usage or {}
    # One line per turn for operations: what was asked of which model and what it cost. No question,
    # no record values, no answer text.
    logger.info(
        "assistant turn model=%s mode=%s direct=%s tools=%d tool_calls=%s rounds=%s finish=%s "
        "completion_tokens=%s total_ms=%s",
        spec.alias,
        mode,
        direct,
        len(tools),
        result.tool_calls,
        result.rounds,
        result.finish_reason,
        usage.get("completion_tokens"),
        round((ended - started) * 1000),
    )
    completion_tokens = usage.get("completion_tokens")
    gen_sec = ended - (first_token_at or ended)
    yield sse(
        "done",
        {
            "text": result.text,
            "finish_reason": result.finish_reason,
            "model": spec.alias,
            "provider_model": result.model,
            "served_by": _served_by(gateway, spec, result.model),
            "call_mode": call_mode if tools else None,
            "tools": [t.name for t in tools],
            "tool_calls": result.tool_calls,
            "rounds": result.rounds,
            "completeness": {"ok": result.verdict.ok, "reason": result.verdict.reason},
            "usage": usage,
            "timings": {
                "first_token_ms": round(((first_token_at or ended) - started) * 1000),
                "total_ms": round((ended - started) * 1000),
                "tokens_per_sec": round(completion_tokens / gen_sec, 1) if completion_tokens and gen_sec > 0 else None,
            },
            "reasoning_chars": reasoning_chars,
            "trimmed": {
                "dropped_messages": assembled.dropped_messages,
                "context_truncated": assembled.context_truncated or host_capped,
            },
            "window": {"prompt_tokens": assembled.prompt_tokens, "context_window": spec.context_window},
            "sent": {"system": assembled.system, "context": assembled.context},
        },
    )


def _served_by(gateway, spec, response_model):
    """{alias, label, local} of the model that answered when a fallback replaced the requested one."""
    registry = getattr(gateway, "registry", None)
    other = registry.served_by(spec, response_model) if registry is not None else None
    return {"alias": other.alias, "label": other.label, "local": bool(other.local)} if other else None
