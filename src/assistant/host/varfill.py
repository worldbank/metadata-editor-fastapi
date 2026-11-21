"""
Label a data file's variables from an attached codebook, questionnaire or data dictionary.

A file can hold thousands of variables, so the model is the second resort, not the first:

1. **Codebook lines.** A codebook states each variable on a line of its own: a table row, ``name: label``,
   ``name - label``. Those are read directly. No model call, no invention, and it scales to any number of
   variables. A table whose header names its columns (Label, Description, Question) fills each attribute
   from its own column.
2. **The model**, for the variables no line states cleanly (a PDF whose table lost its separators, a
   questionnaire that names a variable inside a question): batches of a dozen, each with the passages
   that mention those variables, one completion per batch, one model slot per batch.

Every proposal carries the document's own words as evidence. A model proposal is kept only when its
evidence is in the document and names the variable (or the label itself is written there). A proposal
for a variable that already has a value is marked as differing, never silently replacing it.

Nothing here writes: the result is a list of proposals the curator reviews and applies.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from dataclasses import dataclass
from typing import Any, Callable, Optional

from src.assistant.core.config import ModelSpec
from src.assistant.core.documents import attached_document
from src.assistant.core.llm import Gateway
from src.assistant.host.editor import parse_bundle
from src.assistant.host.tools import _normalise, evidence_is_grounded, ungrounded_tokens

logger = logging.getLogger(__name__)

ATTRS = ("label", "description", "question")
LIMITS = {"label": 255, "description": 4000, "question": 2000}
MAX_VARIABLES = 5000
BATCH = 12
MAX_MODEL_VARIABLES = 240
EXCERPT_BYTES = 5500
MAX_TOKENS = 4096
# Lines within this distance of another variable line make a run; a lone "Country: Saint Lucia" in
# prose is not a codebook line.
RUN_GAP = 3

_HEADER_NAME = {"variable", "variable name", "var", "var name", "varname", "name", "variable id", "variable code", "field", "column", "item"}
_HEADER_ATTR = {
    "label": "label", "variable label": "label", "var label": "label", "short label": "label", "title": "label",
    "description": "description", "definition": "description", "variable definition": "description",
    "question": "question", "question text": "question", "questionnaire text": "question",
    "literal question": "question", "question wording": "question", "wording": "question",
}
_EMPTY_CELL = {"", "-", "—", "–", "n/a", "na", "none", ".", "…"}
_SEPARATOR_ROW = re.compile(r"^:?-{2,}:?$")
_WORD = re.compile(r"[A-Za-z_][\w.]*")
_TEXT_LINE = re.compile(r"^([A-Za-z_][\w.]*)(?:\s*[:=]\s*|\s+[-–—]+\s+|\t+|\s{2,})(\S.*)$")
_BULLET = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+")


@dataclass
class Candidate:
    line: int
    name: str  # casefolded
    values: dict[str, str]
    headered: bool
    evidence: str


def _cell(text: str) -> str:
    text = re.sub(r"<br\s*/?>", " ", text or "", flags=re.IGNORECASE)
    text = text.replace("`", "").replace("**", "").replace("__", "")
    return re.sub(r"\s+", " ", text).strip().strip('"“”')


def _plausible(name: str, attr: str, value: str) -> bool:
    if value.casefold() in _EMPTY_CELL or value.casefold() == name.casefold():
        return False
    if not re.search(r"[A-Za-zÀ-ɏ]", value):
        return False
    return (2 if attr == "label" else 3) <= len(value) <= LIMITS[attr]


def _header_map(cells: list[str]) -> Optional[dict[int, str]]:
    """Column index -> attribute for a header row, or None when the row is not a header."""
    if _cell(cells[0]).casefold() not in _HEADER_NAME:
        return None
    mapping: dict[int, str] = {}
    for i, text in enumerate(cells[1:], 1):
        attr = _HEADER_ATTR.get(_cell(text).casefold())
        if attr and attr not in mapping.values():
            mapping[i] = attr
    # A codebook whose only text column is "Description" means the label by it.
    if "label" not in mapping.values():
        for i, attr in list(mapping.items()):
            if attr == "description":
                mapping[i] = "label"
                break
    return mapping or None


def parse_codebook(text: str, names: set[str]) -> dict[str, list[Candidate]]:
    """
    Codebook entries for ``names`` (casefolded) found in ``text``, by name. A structured line counts when
    its table has a header, or when another variable line sits within ``RUN_GAP`` lines of it.
    """
    found: list[Candidate] = []
    columns: Optional[dict[int, str]] = None
    for number, raw in enumerate((text or "").splitlines()):
        line = raw.strip()
        if line.count("|") >= 2:
            cells = line.strip("|").split("|")
            if all(_SEPARATOR_ROW.match(c.strip()) for c in cells if c.strip()):
                continue
            header = _header_map(cells)
            if header is not None:
                columns = header
                continue
            name = _cell(cells[0]).casefold()
            if name not in names or len(cells) < 2:
                continue
            mapping = columns or {1: "label"}
            values = {}
            for i, attr in mapping.items():
                if i < len(cells):
                    value = _cell(cells[i])
                    if _plausible(name, attr, value):
                        values[attr] = value
            if values:
                shown = " · ".join(c for c in (_cell(x) for x in cells) if c)
                found.append(Candidate(number, name, values, columns is not None, shown[:400]))
            continue
        columns = None
        # Matched before whitespace is collapsed: two spaces are a separator in plain-text codebooks.
        match = _TEXT_LINE.match(_BULLET.sub("", line).replace("`", "").replace("**", ""))
        if match and match.group(1).casefold() in names:
            name, value = match.group(1).casefold(), _cell(match.group(2))
            if _plausible(name, "label", value):
                found.append(Candidate(number, name, {"label": value}, False, line[:400]))
    out: dict[str, list[Candidate]] = {}
    for cand in found:
        if cand.headered or any(o is not cand and abs(o.line - cand.line) <= RUN_GAP for o in found):
            out.setdefault(cand.name, []).append(cand)
    return out


def _same(a: Any, b: Any) -> bool:
    def norm(x: Any) -> str:
        return re.sub(r"[\s.]+$", "", _normalise(str(x or "")))

    return norm(a) == norm(b)


def _current(variable: dict[str, Any], attr: str) -> str:
    return str(variable.get(attr) or "").strip()


def _words(text: str) -> set[str]:
    return {w.casefold() for w in _WORD.findall(text)}


def batch_messages(variables: list[dict[str, Any]], excerpt: str, attrs: list[str]) -> list[dict[str, str]]:
    wanted = {
        "label": "'label': a short label (under 80 characters unless the document's own is longer)",
        "description": "'description': what the variable measures, in one or two sentences",
        "question": "'question': the question as the document words it for the respondent, ending in a question mark; leave it out when the document gives only a label or an interviewer note",
    }
    lines = []
    for n, v in enumerate(variables, 1):
        have = [f"{a}: {_current(v, a)!r}" for a in attrs if _current(v, a)]
        lines.append(f"{n}. {v['name']}" + (f"  (already has {'; '.join(have)})" if have else ""))
    content = (
        "You label the variables of a data file from a document the curator attached (a codebook, data dictionary "
        "or questionnaire). For each numbered variable below, find where the document describes it and give:\n- "
        + "\n- ".join(wanted[a] for a in attrs)
        + "\n- 'evidence': the document's words about this variable, copied exactly, including the variable's name.\n"
        "Use only what the document states about that exact variable. Copy wording, do not invent it. Leave a "
        "variable out when the document does not describe it. Fill only the attributes the document states.\n"
        'Return ONLY JSON: {"variables": [{"variable": 1, "label": "...", "evidence": "..."}]}. '
        'Return {"variables": []} when the document describes none of them.\n\n'
        "Variables:\n" + "\n".join(lines) + f"\n\nDocument excerpts:\n{excerpt}"
    )
    return [{"role": "user", "content": content}]


def parse_items(text: str) -> list[dict[str, Any]]:
    raw = (text or "").strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```[a-z]*\n?", "", raw).removesuffix("```").strip()
    for candidate in (raw, *(m.group(0) for m in re.finditer(r"\{.*\}|\[.*\]", raw, re.DOTALL))):
        try:
            obj = json.loads(candidate)
        except (json.JSONDecodeError, TypeError):
            continue
        items = (obj.get("variables") or obj.get("proposals")) if isinstance(obj, dict) else obj
        if isinstance(items, list):
            return [i for i in items if isinstance(i, dict)]
    return []


def accept_model_item(
    item: dict[str, Any],
    batch: list[dict[str, Any]],
    attrs: list[str],
    document: str,
    corpus: str,
    labels: Optional[dict[str, str]] = None,
) -> tuple[dict[str, dict[str, str]], Optional[dict[str, Any]], Optional[str]]:
    """({attr: {value, evidence}}, the variable, None) when the item holds up; ({}, variable or None, why) otherwise."""
    ref = item.get("variable", item.get("name"))
    variable = None
    if isinstance(ref, str) and ref.strip().isdigit():
        ref = int(ref.strip())
    if isinstance(ref, int) and 1 <= ref <= len(batch):
        variable = batch[ref - 1]
    elif isinstance(ref, str):
        variable = next((v for v in batch if v["name"].casefold() == ref.strip().casefold()), None)
    if variable is None:
        return {}, None, "unknown variable"
    evidence = str(item.get("evidence") or "").strip()
    if not evidence or not evidence_is_grounded(evidence, document):
        return {}, variable, "evidence is not in the document"
    name = variable["name"].casefold()
    out: dict[str, dict[str, str]] = {}
    for attr in attrs:
        value = _cell(str(item.get(attr) or ""))
        if not value or not _plausible(name, attr, value):
            continue
        if name not in evidence.casefold() and _normalise(value) not in _normalise(document):
            continue  # the evidence is about something else
        if ungrounded_tokens(value, corpus):
            continue
        if attr != "label":
            # Small models restate the label as a description or question; neither is a new fact.
            known = (labels or {}).get(name) or _current(variable, "label") or str(out.get("label", {}).get("value") or "")
            if known and _same(value, known):
                continue
        if attr == "question" and "?" not in value:
            continue  # question wording, not an interviewer note or a label
        out[attr] = {"value": value, "evidence": evidence[:400]}
    return out, variable, None if out else "nothing the document supports"


def _add_item(
    items: list[dict[str, Any]], variable: dict[str, Any], attr: str, value: str, evidence: str, via: str
) -> str:
    """Append a proposal unless the saved value already agrees. Returns "empty", "differs" or "agrees"."""
    current = _current(variable, attr)
    if current and _same(current, value):
        return "agrees"
    status = "differs" if current else "empty"
    items.append(
        {
            "uid": variable.get("uid"),
            "name": variable["name"],
            "field": attr,
            "value": value,
            "current": current,
            "status": status,
            "via": via,
            "evidence": evidence,
        }
    )
    return status


def summarize(stats: dict[str, Any], attrs: list[str], cancelled: bool) -> str:
    what = "labels" if attrs == ["label"] else "labels, descriptions and question text"
    head = "Stopped early. " if cancelled else ""
    if not stats["proposals"]:
        body = f"The document does not state {what} for these variables that the file lacks."
    else:
        bits = []
        if stats["empty"]:
            bits.append(f"**{stats['empty']}** missing {what.split(',')[0] if attrs == ['label'] else 'values'} it can fill")
        if stats["differs"]:
            bits.append(f"**{stats['differs']}** saved value{'s' if stats['differs'] != 1 else ''} that differ{'' if stats['differs'] != 1 else 's'} from the document")
        body = "Found " + " and ".join(bits) + ". Tick the ones you accept, then apply them."
    parts = [f"{head}Read the document for {stats['variables']} variables. {body}"]
    if stats["agreed"]:
        parts.append(f"{stats['agreed']} already match.")
    if stats["not_found"]:
        shown = ", ".join(stats["not_found"][:8]) + ("…" if len(stats["not_found"]) > 8 else "")
        n = len(stats["not_found"])
        parts.append(f"{n} {'is' if n == 1 else 'are'} not in the document ({shown}).")
    if stats["unchecked"]:
        parts.append(f"{stats['unchecked']} more were left unchecked to keep the run short; run it again after applying.")
    if stats["failed_batches"]:
        parts.append(f"{stats['failed_batches']} batch{'es' if stats['failed_batches'] != 1 else ''} could not be read; run it again to retry.")
    return " ".join(parts)


async def run_variable_fill(
    gateway: Gateway,
    spec: ModelSpec,
    bundle_raw: Optional[dict[str, Any]],
    document: str,
    *,
    attrs: list[str],
    slot: asyncio.Semaphore,
    is_cancelled: Callable[[], bool] = lambda: False,
    progress: Callable[[int, int], None] = lambda done, total: None,
) -> dict[str, Any]:
    """Run the labelling and return what a job stores: summary text, variable proposals, steps, timings."""
    started = time.monotonic()
    attrs = [a for a in ATTRS if a in attrs] or ["label"]
    bundle = parse_bundle(bundle_raw)
    datafile = bundle.datafile if isinstance(bundle.datafile, dict) else {}
    variables = [v for v in (datafile.get("variables") or []) if isinstance(v, dict) and str(v.get("name") or "").strip()][:MAX_VARIABLES]
    steps: list[dict[str, Any]] = []
    items: list[dict[str, Any]] = []
    stats: dict[str, Any] = {
        "variables": len(variables), "empty": 0, "differs": 0, "agreed": 0, "codebook": 0, "model": 0,
        "not_found": [], "unchecked": 0, "failed_batches": 0, "proposals": 0,
    }
    if not variables:
        return {"text": "This data file has no variables yet.", "proposals": [], "variables": {"items": [], "stats": stats}, "steps": []}

    by_name = {v["name"].casefold(): v for v in variables}
    progress(0, len(variables))

    # 1. Codebook lines.
    steps.append({"stage": "call", "name": "read_codebook", "args": {"variables": len(variables)}})
    candidates = parse_codebook(document, set(by_name))
    stated: dict[str, set[str]] = {}  # variable -> attributes some line stated
    for name, cands in candidates.items():
        variable = by_name[name]
        for attr in attrs:
            pick = next((c for c in cands if attr in c.values and c.headered), None) or next((c for c in cands if attr in c.values), None)
            if pick is None:
                continue
            stated.setdefault(name, set()).add(attr)
            outcome = _add_item(items, variable, attr, pick.values[attr], pick.evidence, "codebook")
            if outcome == "agrees":
                stats["agreed"] += 1
            else:
                stats["codebook"] += 1
    steps.append({"stage": "result", "name": "read_codebook", "found": len(stated)})

    # 2. The model, for variables with a missing attribute that no line stated.
    doc = attached_document(document)
    mentioned = _words(document)
    needs = [
        v for v in variables
        if any(not _current(v, a) and a not in stated.get(v["name"].casefold(), set()) for a in attrs)
    ]
    askable = [v for v in needs if v["name"].casefold() in mentioned]
    stats["unchecked"] = max(0, len(askable) - MAX_MODEL_VARIABLES)
    askable = askable[:MAX_MODEL_VARIABLES]
    # Progress counts variables: those the codebook settled are done, each batch settles its own.
    settled = len(variables) - len(askable)
    corpus = document + "\n" + " ".join(v["name"] for v in variables)
    chunk_words = [_words(c.text) for c in doc.chunks]
    answered: set[str] = set()
    cancelled = False
    labels = {n: cands[0].values["label"] for n, cands in candidates.items() if cands and "label" in cands[0].values}
    total = len(variables)
    done = settled
    progress(done, total)
    for start in range(0, len(askable), BATCH):
        if is_cancelled():
            cancelled = True
            break
        batch = askable[start : start + BATCH]
        names = {v["name"].casefold() for v in batch}
        ranked = sorted(
            (i for i, words in enumerate(chunk_words) if words & names), key=lambda i: (-len(chunk_words[i] & names), i)
        )
        picked, used = [], 0
        for i in ranked:
            size = len(doc.passage(i).encode("utf-8")) + 2
            if used + size > EXCERPT_BYTES:
                continue
            picked.append(i)
            used += size
        steps.append({"stage": "call", "name": "label_variables", "args": {"names": [v["name"] for v in batch]}})
        try:
            async with slot:
                text = await gateway.complete(
                    spec,
                    batch_messages(batch, "\n\n".join(doc.passage(i) for i in sorted(picked)), attrs),
                    max_tokens=min(MAX_TOKENS, spec.max_tokens),
                    temperature=0.0,
                    is_cancelled=is_cancelled,
                )
        except Exception as exc:  # noqa: BLE001 - one failed batch must not lose the others
            logger.warning("assistant: variable batch failed (%s)", spec.alias, exc_info=True)
            gateway.mark_failed(spec)
            stats["failed_batches"] += 1
            steps.append({"stage": "failed", "name": "label_variables", "message": type(exc).__name__})
            if stats["failed_batches"] >= 2 and not stats["model"] and not stats["codebook"]:
                break  # the model is down, not the batch
            continue
        for item in parse_items(text):
            found, variable, _ = accept_model_item(item, batch, attrs, document, corpus, labels)
            for attr, got in found.items():
                if _current(variable, attr) or attr in stated.get(variable["name"].casefold(), set()):
                    continue  # the model fills what is missing; saved values are only challenged by codebook lines
                _add_item(items, variable, attr, got["value"], got["evidence"], "model")
                stats["model"] += 1
                answered.add(variable["name"].casefold())
        steps.append({"stage": "result", "name": "label_variables", "found": len(answered)})
        done += len(batch)
        progress(done, total)

    # Not found: the document says nothing about the variable (no codebook line, no model reading) and
    # something it was asked for is still missing.
    said = set(stated) | answered
    stats["not_found"] = sorted(v["name"] for v in needs if v["name"].casefold() not in said)
    order = {v["name"].casefold(): i for i, v in enumerate(variables)}
    items.sort(key=lambda it: (order[it["name"].casefold()], ATTRS.index(it["field"])))
    stats["empty"] = sum(1 for it in items if it["status"] == "empty")
    stats["differs"] = sum(1 for it in items if it["status"] == "differs")
    stats["proposals"] = len(items)
    total_ms = round((time.monotonic() - started) * 1000)
    logger.info(
        "assistant varfill model=%s variables=%d codebook=%d model_items=%d agreed=%d not_found=%d failed=%d total_ms=%d",
        spec.alias, len(variables), stats["codebook"], stats["model"], stats["agreed"], len(stats["not_found"]),
        stats["failed_batches"], total_ms,
    )
    if stats["failed_batches"] and not stats["model"] and not stats["codebook"] and askable:
        return {"error": {"code": "model_error", "message": "The model did not answer. Check that it is running, then try again."}}
    return {
        "text": summarize(stats, attrs, cancelled),
        "proposals": [],
        "variables": {
            "file_id": bundle.file_id or datafile.get("file_id"),
            "file_name": datafile.get("file_name"),
            "attrs": attrs,
            "items": items,
            "stats": stats,
        },
        "steps": steps,
        "tool_calls": sum(1 for s in steps if s["stage"] == "call"),
        "timings": {"total_ms": total_ms},
        "model": spec.alias,
        "citations": [],
    }
