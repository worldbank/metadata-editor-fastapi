"""
Agent loop: model round, tool calls, results, until the model answers. Ported from portable-slm
``src/agent.js``; same limits, same controller lines, same single repair.

Differences from the browser loop, each because the server serves many models rather than one:

- An unknown tool or invalid arguments is handed back to the model as a failed call instead of failing
  the turn. In the browser that meant host misconfiguration; here it is usually a model that formats
  calls slightly differently, and the turn should degrade, not die.
- Calls arrive natively (OpenAI ``tool_calls``) or through a text protocol, and the transcript is
  written back in the matching shape (``tool`` messages, or one user message of results).
- A user line is merged into a trailing user message rather than appended after it, because several
  chat templates (Gemma's among them) reject two user turns in a row.
- No digest sub-turn: no server tool opts into it yet.

Model output and tool results are untrusted data; only declared, validated tools run.
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Callable, Optional

from src.assistant.core.completeness import Verdict, assess, repair_instruction
from src.assistant.core.config import ModelSpec
from src.assistant.core.llm import Delta, Gateway
from src.assistant.core.tools import Tool, ToolArgumentError, ToolContext, check_args

logger = logging.getLogger(__name__)

# Defaults match the browser loop. A model's registry entry may raise them for agentic work (a review
# reads and proposes across many fields); ROUNDS_CEILING bounds any configuration.
MAX_CALLS_PER_ROUND = 4
MAX_TOOL_CALLS = 5
MAX_ROUNDS = 5
ROUNDS_CEILING = 12
CALLS_CEILING = 40
TOOL_TIMEOUT_SEC = 10.0
PREVIEW_CHARS = 300
SHRUNK_RESULT_CHARS = 512

TOOL_BUDGET_SPENT = (
    "TOOL CONTROLLER: the tool budget for this turn is spent, so no further tool calls are possible. Answer now "
    "using the results you already have. Say plainly which fields you could not check, and do not describe what "
    "you would do next."
)


@dataclass
class AgentResult:
    text: str
    verdict: Verdict
    finish_reason: Optional[str]
    model: Optional[str]
    rounds: int
    tool_calls: int
    usage: dict[str, Any] = field(default_factory=dict)


def push_user(history: list[dict[str, Any]], text: str) -> None:
    if history and history[-1].get("role") == "user" and isinstance(history[-1].get("content"), str):
        history[-1] = {**history[-1], "content": f"{history[-1]['content']}\n\n{text}"}
    else:
        history.append({"role": "user", "content": text})


def _fit_history(history: list[dict[str, Any]], spec: ModelSpec, budget: int, count: Callable) -> int:
    """
    Keep a long tool turn inside the window by shortening the largest tool results first. Returns how
    many results were shortened. The question and the system message are never touched.
    """
    shrunk = 0
    while count(spec, history) > budget:
        candidates = [
            (len(str(m.get("content") or "")), i)
            for i, m in enumerate(history)
            if m.get("role") == "tool" and len(str(m.get("content") or "")) > SHRUNK_RESULT_CHARS
        ]
        if not candidates:
            break
        _, i = max(candidates)
        content = str(history[i]["content"])
        history[i] = {**history[i], "content": f"{content[:SHRUNK_RESULT_CHARS]}...[cut {len(content) - SHRUNK_RESULT_CHARS} chars to fit the window]"}
        shrunk += 1
    return shrunk


def _cap(payload: str, cap: int) -> tuple[str, bool]:
    if len(payload) <= cap:
        return payload, False
    return f"{payload[:cap]}...[cut {len(payload) - cap} bytes]", True


async def run_agent(
    gateway: Gateway,
    spec: ModelSpec,
    messages: list[dict[str, Any]],
    tools: list[Tool],
    *,
    call_mode: str,
    max_tokens: int,
    temperature: float,
    count: Callable[[ModelSpec, list[dict[str, Any]]], int],
    max_rounds: int = MAX_ROUNDS,
    max_tool_calls: int = MAX_TOOL_CALLS,
    max_calls_per_round: int = MAX_CALLS_PER_ROUND,
    repair: bool = True,
    is_cancelled: Callable[[], bool] | None = None,
) -> AsyncIterator[tuple[str, Any]]:
    """
    Yield ``(event, data)`` pairs: reasoning, token, round, tool, warning and any event a tool emits
    (e.g. proposal), then exactly one ``("result", AgentResult)``.

    ``call_mode`` is "native", "lfm" or "hermes". With no tools the turn is a single plain round.
    """
    if not 0 <= max_rounds <= ROUNDS_CEILING:
        raise ValueError(f"max_rounds must be between 0 and {ROUNDS_CEILING}")
    if not 1 <= max_tool_calls <= CALLS_CEILING or not 1 <= max_calls_per_round <= max_tool_calls:
        raise ValueError("tool-call budget out of range")
    by_name: dict[str, Tool] = {}
    for tool in tools:
        if tool.name in by_name:
            raise ValueError(f"duplicate tool {tool.name}")
        by_name[tool.name] = tool
    if not tools:
        max_rounds = 0
    native = call_mode == "native"
    text_protocol = None if native else call_mode
    if native and tools:
        text_protocol = spec.text_protocol  # fallback parser for servers that return raw markers
    specs = [t.spec() for t in tools]

    history = [dict(m) for m in messages]
    budget = spec.context_window - max_tokens
    called: dict[str, str] = {}
    usage: dict[str, Any] = {"peak_prompt_tokens": 0, "prompt_tokens": 0, "completion_tokens": 0, "rounds": []}
    calls_used = 0
    retried = False
    last_repair = ""
    refused_fragment: Optional[str] = None
    replaced: Optional[tuple[str, Verdict]] = None
    model_name: Optional[str] = None

    def finish(text: str, verdict: Verdict, finish_reason: Optional[str], round_no: int) -> tuple[str, AgentResult]:
        return (
            "result",
            AgentResult(
                text=text,
                verdict=verdict,
                finish_reason=finish_reason,
                model=model_name,
                rounds=round_no + 1,
                tool_calls=calls_used,
                usage=usage,
            ),
        )

    for round_no in range(max_rounds + 2):
        if is_cancelled is not None and is_cancelled():
            yield finish("", Verdict(False, "empty"), "cancelled", round_no)
            return
        answer_only = round_no >= max_rounds or calls_used >= max_tool_calls
        shrunk = _fit_history(history, spec, budget, count)
        if shrunk:
            yield ("warning", {"kind": "window_fit", "detail": {"tool_results_shortened": shrunk}})

        round_tools = None
        tool_choice = None
        if native and specs:
            # Offered on answer-only rounds too, with tool_choice "none": some providers reject a
            # transcript that contains tool calls when no tools are declared.
            round_tools = specs
            tool_choice = "none" if answer_only else "auto"

        final: Optional[Delta] = None
        async for delta in gateway.stream(
            spec,
            history,
            max_tokens=max_tokens,
            temperature=temperature,
            tools=round_tools,
            tool_choice=tool_choice,
            text_protocol=text_protocol,
            is_cancelled=is_cancelled,
        ):
            if delta.kind == "finish":
                final = delta
            else:
                yield (delta.kind, {"text": delta.text, "round": round_no})
        assert final is not None
        model_name = model_name or final.model
        u = final.usage or {}
        if isinstance(u.get("prompt_tokens"), int):
            usage["peak_prompt_tokens"] = max(usage["peak_prompt_tokens"], u["prompt_tokens"])
            usage["prompt_tokens"] += u["prompt_tokens"]
        if isinstance(u.get("completion_tokens"), int):
            usage["completion_tokens"] += u["completion_tokens"]
        usage["rounds"].append({"round": round_no, **u})

        if final.finish_reason == "cancelled":
            yield finish(final.content, Verdict(False, "empty"), "cancelled", round_no)
            return

        calls = final.tool_calls
        if not calls:
            text = final.content
            verdict = assess(text, instruction=last_repair)
            if not verdict.ok and repair and not retried and final.finish_reason != "length":
                retried = True
                last_repair = repair_instruction(verdict)
                replaced = (text, verdict)
                if text:
                    history.append({"role": "assistant", "content": text})
                push_user(history, last_repair)
                yield ("round", {"round": round_no + 1, "reason": "repair", "verdict": verdict.reason})
                continue
            if verdict.reason == "narrating" and replaced is not None:
                text, verdict = replaced
                yield ("tool", {"stage": "leaked", "name": "reasoning", "reason": "the repair round narrated the prompt"})
            yield finish(text, verdict, final.finish_reason, round_no)
            return

        if answer_only:
            for call in calls:
                yield ("tool", {"stage": "refused", "name": call["function"]["name"], "reason": "tool budget spent"})
            fragment = final.content
            if not retried:
                retried = True
                refused_fragment = fragment
                if fragment:
                    history.append({"role": "assistant", "content": fragment})
                push_user(history, f"{TOOL_BUDGET_SPENT} {repair_instruction(Verdict(False, 'empty'))}")
                yield ("round", {"round": round_no + 1, "reason": "budget"})
                continue
            text = fragment or refused_fragment or ""
            yield finish(text, assess(text), final.finish_reason, round_no)
            return

        # Replay the assistant turn in the shape the model produced it.
        if final.native:
            history.append({"role": "assistant", "content": final.content or None, "tool_calls": calls})
        else:
            history.append({"role": "assistant", "content": final.raw})

        surplus = calls[max_calls_per_round:]
        if surplus:
            yield ("tool", {"stage": "capped", "dropped": len(surplus)})
        controller: list[str] = []
        results: list[tuple[dict[str, Any], str]] = []
        for call in calls[:max_calls_per_round]:
            name = call["function"]["name"]
            if calls_used >= max_tool_calls:
                results.append((call, json.dumps({"error": "Not executed: the tool budget for this turn is spent."})))
                continue
            raw_args = call["function"].get("arguments") or "{}"
            signature = f"{name}:{raw_args}"
            if signature in called:
                results.append((call, called[signature]))
                controller.append(
                    f"TOOL CONTROLLER: {name} was already called with those exact arguments, and the result above is "
                    "that same result. Do not call it again; use it, or answer."
                )
                yield ("tool", {"stage": "repeat", "name": name})
                continue
            tool = by_name.get(name)
            try:
                if tool is None:
                    raise ToolArgumentError(f"no tool named {name}; available: {', '.join(by_name)}")
                args = check_args(raw_args, tool.parameters)
            except ToolArgumentError as exc:
                calls_used += 1
                message = str(exc)
                # Remembered like a result: the same malformed call again gets the same error without a
                # slot (observed: a model repeating one rejected call three times in a row).
                called[signature] = json.dumps({"error": message})
                results.append((call, called[signature]))
                controller.append(f"TOOL CONTROLLER: {name} was not run ({message}). Fix the call or answer without it.")
                yield ("tool", {"stage": "failed", "name": name, "message": message, "raw_args": raw_args[:PREVIEW_CHARS]})
                continue

            yield ("tool", {"stage": "call", "name": name, "args": args})
            ctx = ToolContext()
            try:
                result = await asyncio.wait_for(tool.run(args, ctx), timeout=TOOL_TIMEOUT_SEC)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - a failed tool is information for the model
                calls_used += 1
                message = f"{name} timed out" if isinstance(exc, asyncio.TimeoutError) else (str(exc) or type(exc).__name__)
                called[signature] = json.dumps({"error": message})
                results.append((call, called[signature]))
                controller.append(
                    f"TOOL CONTROLLER: {name} failed ({message}). Do not call it again; answer from what is already "
                    "available, and say what is missing."
                )
                yield ("tool", {"stage": "failed", "name": name, "message": message})
                continue
            calls_used += 1
            for event in ctx.events:
                yield event
            content = result if isinstance(result, str) else json.dumps(result, ensure_ascii=False, default=str)
            payload, truncated = _cap(content, tool.max_result_bytes)
            if truncated:
                yield ("tool", {"stage": "truncated", "name": name, "cap": tool.max_result_bytes})
            called[signature] = payload
            results.append((call, payload))
            yield (
                "tool",
                {"stage": "result", "name": name, "args": args, "bytes": len(content), "preview": content[:PREVIEW_CHARS]},
            )

        for call in surplus:
            results.append(
                (
                    call,
                    json.dumps({"error": f"Not executed: at most {max_calls_per_round} tool calls run per round. Ask again next round if it is still needed."}),
                )
            )
        if round_no + 1 >= max_rounds or calls_used >= max_tool_calls:
            controller.append(TOOL_BUDGET_SPENT)

        # Every call gets a result, before any controller line, so the round reads: calls, results,
        # then one correction.
        if final.native:
            for call, payload in results:
                history.append({"role": "tool", "tool_call_id": call["id"], "content": payload})
            if controller:
                push_user(history, "\n".join(controller))
        else:
            lines = [f"TOOL RESULT {call['function']['name']}: {payload}" for call, payload in results]
            push_user(history, "\n".join(lines + controller))
        yield ("round", {"round": round_no + 1, "reason": "tool_results"})

    raise RuntimeError("tool loop failed to finish")
