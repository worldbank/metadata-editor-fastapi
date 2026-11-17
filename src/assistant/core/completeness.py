"""
Is this a finished answer? Ported from portable-slm ``src/completeness.js``.

Small models stop early in four observed ways: reasoning past the budget with no answer (empty),
emitting the plan they were about to act on (plan-shaped), being cut off mid-structure (truncated),
and narrating the prompt back (narrating). Code, not the prompt, refuses to present those as answers.

Deliberately conservative: a false "incomplete" teaches the reader to ignore the marker, so
plan-shaped only fires when the whole reply is one short planning utterance.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

# "let me know" is a closing courtesy, not an action the assistant is about to take.
_PLAN_TAIL = re.compile(
    r"^\s*(?:let me(?!\s+know\b)|let'?s|i(?:'ll| will| should| need to| am going to| can(?: also)?)|next,? i|now i(?:'ll| will))\b",
    re.IGNORECASE,
)
_DANGLING_TAIL = re.compile(r"(?::|,|\band|\bthe|\bwith|\bto|\bbecause|\bso)\s*$", re.IGNORECASE)
WHOLE_REPLY_PLAN_CHARS = 240


@dataclass(frozen=True)
class Verdict:
    ok: bool
    reason: Optional[str] = None
    evidence: Optional[str] = None


def _echo(value: str, instruction: str) -> bool:
    needle = re.sub(r'^[^"]*"', "", instruction, count=1).strip()[:40].lower()
    return len(needle) >= 20 and needle in value.lower()


def assess(text: str, *, instruction: str = "", whole_reply_plan_chars: int = WHOLE_REPLY_PLAN_CHARS) -> Verdict:
    value = (text or "").strip()
    if not value:
        return Verdict(False, "empty")
    if instruction and _echo(value, instruction):
        return Verdict(False, "narrating", value[:160])
    if value.count("```") % 2 == 1:
        return Verdict(False, "truncated", "unclosed code fence")
    tail = value[-160:]
    last_sentence = re.split(r"(?<=[.!?])\s+", tail)[-1].strip()
    is_question = last_sentence.endswith("?")
    if not is_question and len(value) <= whole_reply_plan_chars and _PLAN_TAIL.search(last_sentence):
        return Verdict(False, "plan-shaped", last_sentence)
    if _DANGLING_TAIL.search(value):
        return Verdict(False, "truncated", tail.strip()[-60:])
    return Verdict(True)


def repair_instruction(verdict: Verdict) -> str:
    if verdict.reason == "plan-shaped":
        return (
            f'Your reply ended with "{verdict.evidence}" and stated no answer. Give the answer now, using only '
            "what you already have. Do not describe what you would do next and do not call a tool."
        )
    if verdict.reason == "truncated":
        return "Your reply stopped in the middle. Give the whole answer again in one piece, shorter if it does not fit."
    return "Give your answer now. One or two short sentences, no analysis, no tool calls."


def describe_incomplete(verdict: Verdict) -> Optional[str]:
    if verdict.reason == "plan-shaped":
        return (
            f'The model stopped mid-plan: "{verdict.evidence}" is what it was about to do, not an answer. '
            "Ask again, or raise the tool budget."
        )
    if verdict.reason == "truncated":
        return "The reply was cut off mid-sentence rather than finished."
    if verdict.reason == "empty":
        return "The model returned no answer to this question."
    if verdict.reason == "narrating":
        return "The model described the conversation instead of answering it."
    return None
