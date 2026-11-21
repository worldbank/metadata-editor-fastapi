"""
LLM gateway: every model call goes through LiteLLM's Router with OpenAI-compatible chat completions.

Local servers (Ollama, llama.cpp ``llama-server``, LM Studio, vLLM) and hosted providers (OpenRouter,
OpenAI, Anthropic, Gemini) differ only in their registry entry; nothing here branches on provider.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Callable

from src.assistant.core.config import ModelSpec, Registry
from src.assistant.core.think import TagSplitter, ThinkSplitter
from src.assistant.core.toolcalls import PROTOCOL_TAGS, parse_calls

logger = logging.getLogger(__name__)

PROBE_TTL_SEC = 30.0


@dataclass
class Delta:
    """
    One normalised streaming event. ``kind`` is "reasoning", "token" or "finish".

    The finish delta carries the round's result: ``content`` is the answer text (no reasoning, no
    tool-call spans), ``raw`` is the reply as the model wrote it minus reasoning (replayed into the
    next round in text-protocol mode), and ``tool_calls`` are OpenAI-shaped calls. ``native`` says
    whether those calls came back as structured ``tool_calls`` or were parsed out of the text.
    """

    kind: str
    text: str = ""
    finish_reason: str | None = None
    usage: dict[str, int] | None = None
    model: str | None = None
    content: str = ""
    raw: str = ""
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    native: bool = False


class _ToolCallAssembler:
    """Stitch streamed ``tool_calls`` fragments (by index) into whole calls."""

    def __init__(self) -> None:
        self._calls: dict[int, dict[str, Any]] = {}

    def feed(self, fragments: Any) -> None:
        for n, frag in enumerate(fragments or []):
            index = getattr(frag, "index", None)
            index = n if index is None else index
            call = self._calls.setdefault(index, {"id": None, "name": "", "arguments": ""})
            if getattr(frag, "id", None):
                call["id"] = frag.id
            fn = getattr(frag, "function", None)
            if fn is not None:
                if getattr(fn, "name", None):
                    call["name"] += fn.name
                if getattr(fn, "arguments", None):
                    call["arguments"] += fn.arguments

    def calls(self) -> list[dict[str, Any]]:
        out = []
        for index in sorted(self._calls):
            c = self._calls[index]
            if not c["name"]:
                continue
            out.append(
                {
                    "id": c["id"] or f"call_{index + 1}",
                    "type": "function",
                    "function": {"name": c["name"], "arguments": c["arguments"] or "{}"},
                }
            )
        return out


def _usage_dict(usage: Any) -> dict[str, int] | None:
    if usage is None:
        return None
    out = {}
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        val = getattr(usage, key, None)
        if val is None and isinstance(usage, dict):
            val = usage.get(key)
        if isinstance(val, int):
            out[key] = val
    return out or None


class Gateway:
    def __init__(
        self,
        registry: Registry,
        *,
        timeout_sec: float = 120.0,
        num_retries: int = 0,
        router: Any = None,
    ) -> None:
        self.registry = registry
        self._probe_cache: dict[str, tuple[float, bool]] = {}
        if router is None:
            from litellm import Router

            router = Router(
                model_list=[
                    {"model_name": m.alias, "litellm_params": dict(m.litellm_params)}
                    for m in registry.usable()
                ],
                fallbacks=registry.router_fallbacks() or None,
                num_retries=num_retries,
                timeout=timeout_sec,
                # One deployment per alias: a cooldown would make an alias vanish for a minute after a
                # single failed call, which reads as a random outage to the user.
                disable_cooldowns=True,
            )
        self.router = router

    async def stream(
        self,
        spec: ModelSpec,
        messages: list[dict[str, Any]],
        *,
        max_tokens: int,
        temperature: float,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | None = None,
        text_protocol: str | None = None,
        is_cancelled: Callable[[], bool] | None = None,
    ) -> AsyncIterator[Delta]:
        """
        One model round. With ``tools`` the model gets OpenAI tool definitions (native mode). With
        ``text_protocol`` the reply is also scanned for that protocol's call spans, which are hidden
        from the token stream: in text mode that is how calls arrive, and in native mode it catches a
        server that returned the model's raw markers instead of parsing them.
        """
        kwargs: dict[str, Any] = {}
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = tool_choice or "auto"
        response = await self.router.acompletion(
            model=spec.alias,
            messages=messages,
            stream=True,
            max_tokens=max_tokens,
            temperature=temperature,
            stream_options={"include_usage": True},
            drop_params=True,
            **kwargs,
        )
        think = ThinkSplitter()
        calls_split = (
            TagSplitter(*PROTOCOL_TAGS[text_protocol], inside="toolcall") if text_protocol else None
        )
        assembler = _ToolCallAssembler()
        raw_parts: list[str] = []
        answer_parts: list[str] = []
        finish_reason: str | None = None
        usage: dict[str, int] | None = None
        model: str | None = None

        def route(pieces: list[tuple[str, str]]) -> list[Delta]:
            out = []
            for kind, text in pieces:
                if kind == "reasoning":
                    out.append(Delta("reasoning", text))
                    continue
                raw_parts.append(text)
                for k2, t2 in calls_split.feed(text) if calls_split else [("token", text)]:
                    if k2 == "token":
                        answer_parts.append(t2)
                        out.append(Delta("token", t2))
            return out

        try:
            async for chunk in response:
                if is_cancelled is not None and is_cancelled():
                    finish_reason = "cancelled"
                    break
                model = model or getattr(chunk, "model", None)
                usage = _usage_dict(getattr(chunk, "usage", None)) or usage
                choices = getattr(chunk, "choices", None) or []
                if not choices:
                    continue
                choice = choices[0]
                finish_reason = getattr(choice, "finish_reason", None) or finish_reason
                delta = getattr(choice, "delta", None)
                if delta is None:
                    continue
                reasoning = getattr(delta, "reasoning_content", None)
                if reasoning:
                    yield Delta("reasoning", reasoning)
                assembler.feed(getattr(delta, "tool_calls", None))
                content = getattr(delta, "content", None)
                if content:
                    for d in route(think.feed(content)):
                        yield d
        finally:
            close = getattr(response, "aclose", None)
            if close is not None:
                try:
                    await close()
                except Exception:  # noqa: BLE001 - closing a dead stream must not mask the real error
                    logger.debug("assistant: closing stream failed", exc_info=True)
        for d in route(think.flush()):
            yield d
        if calls_split:
            for kind, text in calls_split.flush():
                if kind == "token":
                    answer_parts.append(text)
                    yield Delta("token", text)

        raw = "".join(raw_parts)
        native_calls = assembler.calls()
        parsed_calls = [] if native_calls or not text_protocol else parse_calls(raw, text_protocol)
        yield Delta(
            "finish",
            finish_reason=finish_reason,
            usage=usage,
            model=model,
            content="".join(answer_parts).strip(),
            raw=raw.strip(),
            tool_calls=native_calls or parsed_calls,
            native=bool(native_calls),
        )

    async def complete_ex(self, spec, messages, *, max_tokens, temperature=0.1, text_protocol=None, is_cancelled=None):
        """
        One-shot completion: the streaming path, collecting only answer text. Local models that think
        inline spend the first tokens on reasoning; streaming separates it, a non-streaming reply would
        come back empty with finish_reason "length". ``is_cancelled`` stops generation for a caller
        that has gone away. Returns (answer text, finish_reason, tokens spent on reasoning, in characters)
        so a caller can tell "the model said nothing" from "it ran out of tokens while thinking".
        """
        parts: list[str] = []
        reasoning = 0
        finish: str | None = None
        async for delta in self.stream(
            spec, messages, max_tokens=max_tokens, temperature=temperature, text_protocol=text_protocol,
            is_cancelled=is_cancelled,
        ):
            if delta.kind == "token":
                parts.append(delta.text)
            elif delta.kind == "reasoning":
                reasoning += len(delta.text)
            elif delta.kind == "finish":
                finish = delta.finish_reason
        return "".join(parts).strip(), finish, reasoning

    async def complete(self, spec, messages, *, max_tokens, temperature=0.1, text_protocol=None, is_cancelled=None):
        """The answer text of a one-shot completion (see ``complete_ex``)."""
        text, _, _ = await self.complete_ex(
            spec, messages, max_tokens=max_tokens, temperature=temperature, text_protocol=text_protocol,
            is_cancelled=is_cancelled,
        )
        return text

    async def available(self, spec: ModelSpec) -> bool:
        """
        Whether a model can be called right now.

        Hosted models count as available when configured; they are not pinged, since a ping costs a
        request and says little. Local models are probed at their base URL: any HTTP answer means a
        server is listening, a connection error means it is stopped.
        """
        if not spec.configured:
            return False
        if not spec.local:
            return True
        base = spec.api_base or _default_local_base(spec.provider_model)
        if not base:
            return True
        now = time.monotonic()
        cached = self._probe_cache.get(spec.alias)
        if cached and now - cached[0] < PROBE_TTL_SEC:
            return cached[1]
        import httpx

        try:
            async with httpx.AsyncClient(timeout=2.0) as client:
                await client.get(base)
            ok = True
        except httpx.HTTPError:
            ok = False
        self._probe_cache[spec.alias] = (now, ok)
        return ok

    def mark_failed(self, spec: ModelSpec) -> None:
        """Record a failed call so the model list stops reporting a stopped local server as up."""
        if spec.local:
            self._probe_cache[spec.alias] = (time.monotonic(), False)


def _default_local_base(provider_model: str) -> str | None:
    # LiteLLM's own default for Ollama when no api_base is given.
    if provider_model.startswith(("ollama/", "ollama_chat/")):
        return "http://localhost:11434"
    return None


def count_tokens(spec: ModelSpec, messages: list[dict[str, Any]]) -> int:
    """Token estimate for window fitting. Falls back to a byte heuristic if the tokenizer can't load."""
    try:
        import litellm

        return int(litellm.token_counter(model=spec.provider_model, messages=messages))
    except Exception:  # noqa: BLE001
        return sum(len(str(m.get("content", "")).encode("utf-8")) for m in messages) // 3 + 4 * len(messages)
