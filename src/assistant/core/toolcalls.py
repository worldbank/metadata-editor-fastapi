"""
Text tool-call protocols, for models whose server does not return OpenAI ``tool_calls``.

``lfm``:    ``<|tool_call_start|>[name(key='value', ...)]<|tool_call_end|>`` (Liquid LFM2/2.5), ported
            from portable-slm ``src/lfm-output.js``: a tiny literal grammar, never eval.
``hermes``: ``<tool_call>{"name": "...", "arguments": {...}}</tool_call>`` (Qwen, Hermes; works as an
            instructed format for Gemma and others).

Parsing is lenient about a cut-off final span (a truncated call is dropped, not a failed turn) and
strict inside a complete span.
"""

from __future__ import annotations

import json
import re
from typing import Any

PROTOCOL_TAGS: dict[str, tuple[str, str]] = {
    "lfm": ("<|tool_call_start|>", "<|tool_call_end|>"),
    "hermes": ("<tool_call>", "</tool_call>"),
}

PROTOCOL_INSTRUCTIONS: dict[str, str] = {
    "lfm": (
        "To call one, output exactly <|tool_call_start|>[tool_name(parameter='value')]<|tool_call_end|> "
        "and no prose. After the result, answer normally."
    ),
    "hermes": (
        'To call one, output exactly <tool_call>{"name": "tool_name", "arguments": {"parameter": "value"}}'
        "</tool_call> and no prose. After the result, answer normally."
    ),
    "native": "Use the tool-call format required by the model when a tool is needed, then answer after its result.",
}

MAX_SPAN_CHARS = 2048
_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]{0,63}")
_TOOL_NAME = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
_NUMBER = re.compile(r"-?(?:0|[1-9]\d*)(?:\.\d+)?(?:[eE][+-]?\d+)?")
_ESCAPES = {"n": "\n", "r": "\r", "t": "\t", "\\": "\\", "'": "'", '"': '"'}


def _spans(text: str, start_tag: str, end_tag: str) -> list[str]:
    out = []
    cursor = 0
    while True:
        start = text.find(start_tag, cursor)
        if start < 0:
            break
        body_start = start + len(start_tag)
        end = text.find(end_tag, body_start)
        if end < 0:
            break  # truncated mid-call
        body = text[body_start:end].strip()
        if len(body) <= MAX_SPAN_CHARS:
            out.append(body)
        cursor = end + len(end_tag)
    return out


def strip_calls(text: str, protocol: str) -> str:
    """The reply with every tool-call span (complete or cut off) removed."""
    start_tag, end_tag = PROTOCOL_TAGS[protocol]
    out = []
    cursor = 0
    while True:
        start = text.find(start_tag, cursor)
        if start < 0:
            out.append(text[cursor:])
            break
        out.append(text[cursor:start])
        end = text.find(end_tag, start + len(start_tag))
        if end < 0:
            break
        cursor = end + len(end_tag)
    return "".join(out).strip()


class _Parser:
    """Exactly one ``name(key='value', ...)`` starting at index 0."""

    def __init__(self, source: str) -> None:
        self.s = source
        self.i = 0

    def ws(self) -> None:
        while self.i < len(self.s) and self.s[self.i].isspace():
            self.i += 1

    def peek(self) -> str:
        return self.s[self.i] if self.i < len(self.s) else ""

    def identifier(self) -> str:
        m = _IDENT.match(self.s, self.i)
        if not m:
            raise ValueError("expected a name")
        self.i = m.end()
        return m.group(0)

    def quoted(self) -> str:
        quote = self.s[self.i]
        self.i += 1
        value = []
        while self.i < len(self.s):
            ch = self.s[self.i]
            self.i += 1
            if ch == quote:
                return "".join(value)
            if ch != "\\":
                value.append(ch)
                continue
            esc = self.peek()
            self.i += 1
            if esc in _ESCAPES:
                value.append(_ESCAPES[esc])
            elif esc == "u":
                hexd = self.s[self.i:self.i + 4]
                if not re.fullmatch(r"[0-9a-fA-F]{4}", hexd):
                    raise ValueError("invalid unicode escape")
                value.append(chr(int(hexd, 16)))
                self.i += 4
            else:
                raise ValueError("invalid escape")
        raise ValueError("unclosed string")

    def literal(self, depth: int = 0) -> Any:
        if self.peek() in ("'", '"'):
            return self.quoted()
        if self.peek() == "[" and depth == 0:
            # One level of list, e.g. read_fields(pointers=['/a', '/b']).
            self.i += 1
            items: list[Any] = []
            self.ws()
            while self.peek() != "]":
                if self.i >= len(self.s) or len(items) >= 64:
                    raise ValueError("unclosed list")
                items.append(self.literal(depth + 1))
                self.ws()
                if self.peek() == ",":
                    self.i += 1
                    self.ws()
                elif self.peek() != "]":
                    raise ValueError("expected , or ]")
            self.i += 1
            return items
        m = _NUMBER.match(self.s, self.i)
        if m:
            self.i = m.end()
            text = m.group(0)
            return float(text) if any(c in text for c in ".eE") else int(text)
        for word, value in (("true", True), ("false", False), ("None", None), ("null", None)):
            end = self.i + len(word)
            if self.s.startswith(word, self.i) and not re.match(r"[A-Za-z0-9_]", self.s[end:end + 1]):
                self.i = end
                return value
        raise ValueError("arguments must be quoted strings, numbers, booleans, null or a list")

    def call(self) -> tuple[str, dict[str, Any], int]:
        self.ws()
        name = self.identifier()
        self.ws()
        if self.peek() != "(":
            raise ValueError("expected (")
        self.i += 1
        args: dict[str, Any] = {}
        self.ws()
        while self.peek() != ")":
            if self.i >= len(self.s):
                raise ValueError("unclosed call")
            if len(args) >= 16:
                raise ValueError("too many arguments")
            key = self.identifier()
            if key in args:
                raise ValueError(f"duplicate argument {key}")
            self.ws()
            if self.peek() != "=":
                raise ValueError("expected =")
            self.i += 1
            self.ws()
            args[key] = self.literal()
            self.ws()
            if self.peek() == ",":
                self.i += 1
                self.ws()
                continue
            if self.peek() != ")":
                raise ValueError("expected , or )")
        self.i += 1
        return name, args, self.i


def _parse_lfm_block(block: str) -> list[tuple[str, dict[str, Any]]]:
    """Lenient: brackets, commas and prose between calls are skipped; a malformed call is dropped."""
    calls = []
    i = 0
    guard = 0
    while i < len(block) and guard < 64:
        guard += 1
        while i < len(block) and (block[i].isspace() or block[i] in "[,]"):
            i += 1
        if i >= len(block):
            break
        if not (block[i].isalpha() or block[i] == "_"):
            i += 1
            continue
        start = i
        try:
            name, args, end = _Parser(block[i:]).call()
            if not _TOOL_NAME.match(name):
                raise ValueError("invalid tool name")
            calls.append((name, args))
            i += end
        except ValueError:
            # Resync past the junk word, not to the next parenthesis, which could skip a good call.
            i = start
            while i < len(block) and not block[i].isspace():
                i += 1
    return calls


def _parse_hermes_block(block: str) -> list[tuple[str, dict[str, Any]]]:
    try:
        data = json.loads(block)
    except json.JSONDecodeError:
        return []
    items = data if isinstance(data, list) else [data]
    calls = []
    for item in items:
        if not isinstance(item, dict):
            continue
        name = item.get("name")
        args = item.get("arguments", item.get("parameters", {}))
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                continue
        if isinstance(name, str) and _TOOL_NAME.match(name) and isinstance(args, dict):
            calls.append((name, args))
    return calls


def parse_calls(text: str, protocol: str) -> list[dict[str, Any]]:
    """OpenAI-shaped tool calls found in ``text``."""
    start_tag, end_tag = PROTOCOL_TAGS[protocol]
    parse_block = _parse_lfm_block if protocol == "lfm" else _parse_hermes_block
    out = []
    for body in _spans(text, start_tag, end_tag):
        for name, args in parse_block(body):
            out.append(
                {
                    "id": f"{protocol}_{len(out) + 1}",
                    "type": "function",
                    "function": {"name": name, "arguments": json.dumps(args, ensure_ascii=False)},
                }
            )
    return out
