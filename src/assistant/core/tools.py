"""
Tool definitions, argument validation, and the built-in offline tools.

Ported from portable-slm ``src/tools.js`` and the argument checks in ``src/agent.js``. Tools are trusted
host code; their arguments come from the model and are untrusted until ``check_args`` passes.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable

TOOL_NAME_RE = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
MAX_STRING_ARG = 500
DEFAULT_RESULT_BYTES = 2048


@dataclass
class ToolContext:
    """What a running tool may do besides return a result: emit protocol events (e.g. a proposal)."""

    events: list[tuple[str, Any]] = field(default_factory=list)

    def emit(self, event: str, data: Any) -> None:
        self.events.append((event, data))


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict[str, Any]
    run: Callable[[dict[str, Any], ToolContext], Awaitable[Any]]
    max_result_bytes: int = DEFAULT_RESULT_BYTES

    def __post_init__(self) -> None:
        if not TOOL_NAME_RE.match(self.name) or self.parameters.get("type") != "object":
            raise ValueError(f"invalid tool definition: {self.name!r}")

    def spec(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {"name": self.name, "description": self.description, "parameters": self.parameters},
        }


class ToolArgumentError(ValueError):
    pass


def check_args(raw: str, schema: dict[str, Any]) -> dict[str, Any]:
    try:
        args = json.loads(raw or "{}")
    except json.JSONDecodeError as exc:
        raise ToolArgumentError("tool arguments must be a JSON object") from exc
    if not isinstance(args, dict):
        raise ToolArgumentError("tool arguments must be a JSON object")
    props = schema.get("properties") or {}
    for key in schema.get("required") or []:
        if key not in args:
            raise ToolArgumentError(f"missing tool argument: {key}")
    for key, value in args.items():
        rule = props.get(key)
        if rule is None:
            raise ToolArgumentError(f"unexpected tool argument: {key}")
        kind = rule.get("type")
        if kind == "string":
            if not isinstance(value, str) or len(value) > int(rule.get("maxLength", MAX_STRING_ARG)):
                raise ToolArgumentError(f"invalid string: {key}")
        elif kind == "number":
            if isinstance(value, bool) or not isinstance(value, (int, float)) or value != value or value in (float("inf"), float("-inf")):
                raise ToolArgumentError(f"invalid number: {key}")
        elif kind == "boolean":
            if not isinstance(value, bool):
                raise ToolArgumentError(f"invalid boolean: {key}")
        elif kind == "array":
            # Lists of short strings only (e.g. several field pointers at once).
            limit = int(rule.get("maxItems", 20))
            if (
                not isinstance(value, list)
                or not value
                or len(value) > limit
                or any(not isinstance(v, str) or len(v) > MAX_STRING_ARG for v in value)
            ):
                raise ToolArgumentError(f"invalid list: {key} (1 to {limit} strings)")
            continue
        else:
            raise ToolArgumentError(f"unsupported argument type: {key}")
        if "enum" in rule and value not in rule["enum"]:
            # Name the valid choices: a model that guessed "/study_desc/abstract" can correct itself in
            # the next round only if it is told what exists.
            choices = ", ".join(map(str, rule["enum"]))
            raise ToolArgumentError(f"invalid value for {key}: {str(value)[:80]!r}; choose one of: {choices[:600]}")
    return args


_CALC_ALLOWED = re.compile(r"^[\d.+\-*/()\s]+$")
_CALC_TOKEN = re.compile(r"\d+(?:\.\d+)?|\.\d+|[()+*/-]")
_CALC_NUMBER = re.compile(r"^(?:\d+(?:\.\d+)?|\.\d+)$")


def calculate(expression: str) -> float | int:
    """Recursive-descent arithmetic over + - * / and parentheses. Never eval."""
    if not isinstance(expression, str) or len(expression) > 80 or not _CALC_ALLOWED.match(expression):
        raise ValueError("Only short arithmetic expressions are allowed")
    tokens = _CALC_TOKEN.findall(expression)
    if "".join(tokens) != re.sub(r"\s", "", expression):
        raise ValueError("Invalid arithmetic expression")
    pos = 0

    def take() -> str | None:
        nonlocal pos
        tok = tokens[pos] if pos < len(tokens) else None
        pos += 1
        return tok

    def peek() -> str | None:
        return tokens[pos] if pos < len(tokens) else None

    def atom(depth: int) -> float:
        if depth > 32:
            raise ValueError("Expression is too deeply nested")
        tok = take()
        if tok in ("-", "+"):
            return (-1 if tok == "-" else 1) * atom(depth + 1)
        if tok == "(":
            value = total(depth + 1)
            if take() != ")":
                raise ValueError("Unclosed parenthesis")
            return value
        if tok is None or not _CALC_NUMBER.match(tok):
            raise ValueError("Expected a number")
        return float(tok)

    def product(depth: int) -> float:
        value = atom(depth)
        while peek() in ("*", "/"):
            op = take()
            rhs = atom(depth)
            if op == "/" and rhs == 0:
                raise ValueError("Division by zero")
            value = value * rhs if op == "*" else value / rhs
        return value

    def total(depth: int) -> float:
        value = product(depth)
        while peek() in ("+", "-"):
            op = take()
            value += (1 if op == "+" else -1) * product(depth)
        return value

    result = total(0)
    if pos != len(tokens) or result != result or abs(result) > 1e12:
        raise ValueError("Invalid arithmetic result")
    return int(result) if result.is_integer() else result


async def _get_datetime(args: dict[str, Any], ctx: ToolContext) -> Any:
    return {"datetime": datetime.now(timezone.utc).isoformat(timespec="seconds")}


async def _calculate(args: dict[str, Any], ctx: ToolContext) -> Any:
    return {"result": calculate(args["expression"])}


BUILTIN_TOOL_NAMES = frozenset({"get_datetime", "calculate"})


def builtin_tools() -> list[Tool]:
    """Offline utilities. Portable-slm's ``wiki_search`` is not ported: this worker makes no web calls."""
    return [
        Tool(
            name="get_datetime",
            description="Read the server's current date and time in UTC. Use for questions about today or the current time.",
            parameters={"type": "object", "properties": {}, "required": []},
            run=_get_datetime,
        ),
        Tool(
            name="calculate",
            description="Calculate an arithmetic expression using +, -, *, /, and parentheses.",
            parameters={
                "type": "object",
                "properties": {"expression": {"type": "string", "description": "Arithmetic expression, e.g. (2+3)*7"}},
                "required": ["expression"],
            },
            run=_calculate,
        ),
    ]
