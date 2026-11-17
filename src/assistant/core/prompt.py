"""
Prompt assembly, ported from portable-slm ``integrations/chat-core.js``.

Same contract as the browser version: the assistant's own description comes first and a host can only
append to it, never replace it, so a host cannot make the assistant claim a capability it lacks. The
wording differs where the browser text described running "inside this browser tab".
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Callable

from src.assistant.core.config import ModelSpec

CONTEXT_SEPARATOR = "--- context supplied by this application ---"


def default_context(spec: ModelSpec) -> str:
    where = (
        "on infrastructure run by this application's operators"
        if spec.local
        else "by an external model provider chosen by this application's operators"
    )
    return f"""About this assistant
- This assistant runs a language model ({spec.label}) on a server, {where}. The model can be
  wrong: answer clearly, and say when you are unsure.
- Answer general questions from your own knowledge. Do not refuse just because the supplied text does
  not cover it.
- Facts about this application come from the supplied text or from a tool's result. If neither has the
  fact, say you cannot verify it rather than inventing a value.
- You cannot inspect the page, screen, selection, other tabs, or unsaved form values unless they are
  supplied here. You cannot save, publish, delete, or change anything; never claim that you did."""


def system_prompt(app: str = "this application", mode: str = "none", kind: str = "help") -> str:
    opening = (
        f"You are a helpful general-purpose assistant inside {app}. Answer general questions normally "
        "from your knowledge; be clear about uncertainty."
    )
    if mode == "record":
        return (
            f"{opening} For facts about the open project's saved metadata, use the supplied snapshot or a "
            "relevant available tool. Never guess a project value. If it is absent or a read is denied, say "
            "you cannot verify it. General advice is allowed but distinguish it from facts about this project. "
            "Never claim to save, publish, or change anything."
        )
    if mode == "app" and kind == "content":
        return (
            f"{opening} The supplied text is the material to answer from: answer the question that was asked, "
            "using it, and do not end by asking the user what they want or offering to help. If it does not "
            "cover the answer, say that plainly in one sentence rather than filling the gap from general "
            "knowledge. Name the document or section you drew on. Questions about this application, site or "
            'page itself are answered from the "About this application" text you were given: never say you '
            "cannot see the page, and never ask the user to paste content, because the page is described in "
            "front of you. You have read nothing beyond the supplied text, and you cannot open, save, publish, "
            "or change anything."
        )
    if mode == "app":
        return (
            f"{opening} For questions about how to use {app}, use the supplied help text; if it does not cover "
            "the answer, say so, then offer general guidance clearly labeled as such. You have not read a "
            "project or user data. Never claim to open, save, publish, or change a record."
        )
    return (
        f"{opening} You have not read this application's page or any project data. Do not claim to have "
        "opened, saved, published, or changed anything."
    )


# Identity, capability and greeting turns never need host data, and small models over-trigger tools on
# exactly these questions. Routing them is deterministic, as in the browser version.
_DIRECT_PATTERN = re.compile(
    r"(^|\b)(hi+|hello|henlo|hallo|helo|heya|hiya|hey|yo|sup|good\s(morning|afternoon|evening))\b"
    r"|(what|who)\s+(are|r)\s+(you|u)\b|what\s+can\s+(you|u)\s+do|what\s+do\s+(you|u)\s+do"
    r"|(your|ur)\s+name\b|what\s+are\s+(your|ur)\s+(capabilities|tools|limits)"
    r"|can\s+(you|u)\s+(save|edit|write|publish|delete|modify)",
    re.IGNORECASE,
)

DIRECT_REPLY_NOTE = (
    "The user greeted you or asked about your identity or capabilities. Reply in one or two short sentences "
    "and invite their question. Do not enumerate, list or summarize the supplied context (projects, fields, "
    "records) unless they explicitly ask."
)


def routes_direct(question: str) -> bool:
    text = (question or "").strip()
    return 0 < len(text) <= 120 and bool(_DIRECT_PATTERN.search(text))


def compose_context(spec: ModelSpec, host_text: str) -> str:
    base = default_context(spec)
    host = (host_text or "").strip()
    return f"{base}\n\n{CONTEXT_SEPARATOR}\n{host}" if host else base


def byte_cap(text: str, max_bytes: int) -> tuple[str, bool]:
    """Cut ``text`` to ``max_bytes`` of UTF-8 without splitting a character."""
    raw = text.encode("utf-8")
    if len(raw) <= max_bytes:
        return text, False
    return raw[:max_bytes].decode("utf-8", errors="ignore") + "\n[truncated]", True


@dataclass
class Assembled:
    messages: list[dict[str, Any]]
    system: str
    context: str
    dropped_messages: int
    context_truncated: bool
    prompt_tokens: int


def build_messages(
    spec: ModelSpec,
    *,
    system: str,
    context: str,
    history: list[dict[str, str]],
    question: str,
    max_history: int,
    max_tokens: int,
    count: Callable[[ModelSpec, list[dict[str, Any]]], int],
) -> Assembled:
    """
    ``[system + context, ...last max_history messages, user question]``, fitted to the model's window.

    Over the window it degrades rather than failing: oldest history first (whole exchanges, so the
    model never sees an answer without its question), then the context is cut, and the system message
    says what was dropped.
    """
    history = [m for m in history if m.get("content")][-max_history:] if max_history > 0 else []
    if history and history[0]["role"] == "assistant":
        history = history[1:]
    budget = spec.context_window - max_tokens
    dropped = 0
    truncated = False

    def assemble(hist: list[dict[str, str]], ctx: str) -> list[dict[str, Any]]:
        note = (
            f"\n\n[{dropped} earlier message(s) were dropped to fit the model's context window.]"
            if dropped
            else ""
        )
        head = {"role": "system", "content": f"{system}\n\nContext from the host application:\n{ctx}{note}"}
        return [head, *hist, {"role": "user", "content": question}]

    messages = assemble(history, context)
    tokens = count(spec, messages)
    while tokens > budget and history:
        cut = 2 if len(history) >= 2 else 1
        history = history[cut:]
        dropped += cut
        messages = assemble(history, context)
        tokens = count(spec, messages)
    while tokens > budget and len(context) > 200:
        context, _ = byte_cap(context, len(context.encode("utf-8")) // 2)
        truncated = True
        messages = assemble(history, context)
        tokens = count(spec, messages)
    return Assembled(
        messages=messages,
        system=system,
        context=context,
        dropped_messages=dropped,
        context_truncated=truncated,
        prompt_tokens=tokens,
    )


def tool_instructions(tools: list[Any], call_mode: str, builtin_names: frozenset[str]) -> str:
    """
    Tool rules for the system prompt, ported from ``toolInstructions``. The strict rule (call only for
    an explicitly requested saved value) applies once any host tool is present; built-in utilities
    alone get the short rule.
    """
    from src.assistant.core.toolcalls import PROTOCOL_INSTRUCTIONS

    if not tools:
        return ""
    listing = "\n".join(f"- {t.name}: {t.description}" for t in tools)
    reads_host_data = any(t.name not in builtin_names for t in tools)
    if reads_host_data:
        # The browser's rule was "call a tool ONLY when explicitly asked for a saved value": right for a
        # 2.6B model in a tab with a five-call budget. On the server the tools are cheap reads of data the
        # proxy already sent, and the assistant is expected to work a task through, so the rule is about
        # when tools are not needed rather than when they are allowed.
        rules = (
            "Use the tools whenever the answer depends on this application's saved data or documentation, and use "
            "as many as the task needs. Greetings, identity questions, opinions and general-knowledge questions "
            "need none."
        )
    else:
        rules = "Call a tool only when the answer needs one: a current date or time, or a calculation."
    protocol = PROTOCOL_INSTRUCTIONS.get(call_mode, PROTOCOL_INSTRUCTIONS["native"])
    return (
        f"Tool rules. {rules} Check the supplied text first: if the answer is already in it, answer directly. "
        f"Never invent a tool result, and never claim a request succeeded unless it returned one. Every proposal and "
        f"every factual statement must trace to read tool output or the supplied documentation; if the record is "
        f"silent, say it is missing instead of guessing.\n"
        f"Available tools:\n{listing}\n{protocol}"
    )
