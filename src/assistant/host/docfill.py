"""
Fill from a document: every fillable field of the record checked against the attached document, in
small batches, as one job.

The chat agent can do this too, but a small local model given 40 fields spends its rounds listing and
reading them and proposes a handful. Here the fields are known up front, so each batch is one plain
completion: the fields, their saved values and the document passages about them in, a JSON list of
proposals out. Every proposal passes the same guard as ``propose_field_value`` and must quote the
document, so an empty field gets a value only when the document states it, and a saved value is
challenged only when the document says something more complete or different.

Batches take the shared model slot one at a time, so a long fill does not hold it against other users.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
from typing import Any, Callable, Optional

from src.assistant.core.config import ModelSpec
from src.assistant.core.documents import attached_document
from src.assistant.core.llm import Gateway
from src.assistant.host import listfill
from src.assistant.host.editor import parse_bundle
from src.assistant.host.tools import _is_empty, _label, _normalise, _preview, bundle_corpus, check_field_value, evidence_is_grounded

logger = logging.getLogger(__name__)

BATCH = 6
# Batches in flight at once for one fill on a hosted model.
FILL_PARALLEL = max(1, int(os.getenv("ASSISTANT_FILL_PARALLEL", "3")))
# A local model usually has far fewer slots; raise this only to the number the model server runs at once.
FILL_PARALLEL_LOCAL = max(1, int(os.getenv("ASSISTANT_FILL_PARALLEL_LOCAL", "1")))
MAX_FIELDS = 72
EXCERPT_BYTES = int(os.getenv("ASSISTANT_FILL_EXCERPT_BYTES", "6000"))  # document text sent with each batch
CURRENT_PREVIEW = 600
# Local models think before the JSON: LFM2.5-2.6B spends ~1,600 tokens reasoning over six fields, and
# a cap near that returns an empty answer with finish_reason "length".
MAX_TOKENS = 4096
# Added up front for a local model, and to a batch whose first reply was empty or cut off: small reasoning
# models can spend the whole token budget thinking and never write the answer.
BREVITY_NUDGE = "\n\nAnswer now with ONLY the JSON object. Do not explain or think at length; keep each value brief."
FORMATS = {"date": "one date as the document writes it, not a range", "integer": "a whole number"}
_RANGE = re.compile(r"\s(?:to|until|through)\s|\s[-–—]\s|\d\s*[–—]\s*\d", re.IGNORECASE)

PROMPT = (
    "You fill metadata fields from a document the curator attached. For each numbered field below:\n"
    "- if it is empty and the document states its value, propose that value;\n"
    "- if it has a value and the document states it more completely, more precisely or differently, propose "
    "the improved value;\n"
    "- otherwise leave the field out.\n"
    "Copy names, dates, numbers and identifiers exactly as the document writes them. For a field with options, "
    "use exactly one option. Put the document's own words that support the value in 'evidence'.\n"
    "Return ONLY JSON: {\"proposals\": [{\"field\": 1, \"value\": \"...\", \"evidence\": \"words copied from "
    "the document\", \"reason\": \"one short sentence\"}]}. Return {\"proposals\": []} when the document says "
    "nothing about these fields."
)


def fill_targets(fields: dict[str, Any]) -> list[str]:
    """
    Fillable pointers in the order worth spending the model on: empty required, empty recommended,
    other empty, then filled ones in the same order. Template order within each group keeps a batch's
    fields in the same section, which is also where the document tends to discuss them together.
    """
    groups: list[list[str]] = [[] for _ in range(6)]
    for pointer, f in fields.items():
        if not isinstance(f, dict) or f.get("in_template") is False or not f.get("fillable", True):
            continue
        rank = 0 if f.get("required") else 1 if f.get("recommended") else 2
        groups[rank + (0 if _is_empty(f.get("value")) else 3)].append(pointer)
    ordered = [p for group in groups for p in group][:MAX_FIELDS]
    keep = set(ordered)
    # Batches follow the template, not the priority: the cap above already chose which fields to check.
    return [p for p in fields if p in keep]


def batch_messages(fields: dict[str, Any], pointers: list[str], document: str) -> list[dict[str, str]]:
    doc = attached_document(document)
    lines = []
    query_parts = []
    for n, pointer in enumerate(pointers, 1):
        f = fields[pointer]
        label = _label(f, pointer)
        query_parts.append(f"{label} {f.get('help') or ''}".strip())
        lines.append(f"{n}. {label}" + (f" (section: {f['section']})" if f.get("section") else ""))
        if f.get("help"):
            lines.append(f"   help: {f['help']}")
        if f.get("type") in FORMATS:
            lines.append(f"   format: {FORMATS[f['type']]}")
        allowed = f.get("options") or f.get("suggestions")
        if allowed:
            lines.append(f"   options: {'; '.join(str(o) for o in allowed[:30])}")
        value = f.get("value")
        lines.append(f"   current value: {'(empty)' if _is_empty(value) else json.dumps(_preview(value, CURRENT_PREVIEW), ensure_ascii=False)}")
    excerpt, _ = doc.excerpt(query_parts, EXCERPT_BYTES)
    content = f"{PROMPT}\n\nFields:\n" + "\n".join(lines) + f"\n\nDocument:\n{excerpt}"
    return [{"role": "user", "content": content}]


def parse_proposals(text: str) -> list[dict[str, Any]]:
    """The proposals list from a reply, tolerating code fences, prose around the JSON and a bare list."""
    raw = (text or "").strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```[a-z]*\n?", "", raw).removesuffix("```").strip()
    for candidate in (raw, *(m.group(0) for m in re.finditer(r"\{.*\}|\[.*\]", raw, re.DOTALL))):
        try:
            obj = json.loads(candidate)
        except (json.JSONDecodeError, TypeError):
            continue
        items = obj.get("proposals") if isinstance(obj, dict) else obj
        if isinstance(items, list):
            return [item for item in items if isinstance(item, dict)]
    return []


def parse_proposals_strict(text: str) -> Optional[list[dict[str, Any]]]:
    """
    The proposals in a reply, or None when it holds no usable answer: an empty reply, or JSON cut off
    before one complete proposal. ``[]`` means the model answered and found nothing. When a reply was
    cut off part-way through its list, the proposals that are complete are kept.
    """
    if not (text or "").strip():
        return None
    raw = text.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```[a-z]*\n?", "", raw).removesuffix("```").strip()
    for candidate in (raw, *(m.group(0) for m in re.finditer(r"\{.*\}|\[.*\]", raw, re.DOTALL))):
        try:
            obj = json.loads(candidate)
        except (json.JSONDecodeError, TypeError):
            continue
        items = obj.get("proposals") if isinstance(obj, dict) else obj
        if isinstance(items, list):
            return [item for item in items if isinstance(item, dict)]
    # Cut off (the token budget ran out): read the complete objects of the "proposals" list one by one.
    start = raw.find("[", raw.find('"proposals"') if '"proposals"' in raw else 0)
    if start < 0:
        return None
    decoder, pos, kept = json.JSONDecoder(), start + 1, []
    while pos < len(raw):
        while pos < len(raw) and raw[pos] in " \t\r\n,":
            pos += 1
        try:
            item, pos = decoder.raw_decode(raw, pos)
        except json.JSONDecodeError:
            break
        if isinstance(item, dict):
            kept.append(item)
    return kept or None


def _resolve_pointer(item: dict[str, Any], pointers: list[str]) -> Optional[str]:
    ref = item.get("field", item.get("pointer"))
    if isinstance(ref, str) and ref.strip().isdigit():
        ref = int(ref.strip())
    if isinstance(ref, int) and 1 <= ref <= len(pointers):
        return pointers[ref - 1]
    if isinstance(ref, str) and ref in pointers:
        return ref
    return None


def accept(
    item: dict[str, Any], pointers: list[str], fields: dict[str, Any], document: str, corpus: str, record_id: Optional[str]
) -> tuple[Optional[dict[str, Any]], Optional[str]]:
    """A proposal card, or why the item was dropped."""
    pointer = _resolve_pointer(item, pointers)
    if pointer is None:
        return None, "unknown field"
    field = fields[pointer]
    value, error = check_field_value(field, pointer, item.get("value"), None, corpus)
    if error:
        return None, error
    if field.get("type") == "date" and _RANGE.search(value):
        return None, "a range for a single date"
    evidence = str(item.get("evidence") or "").strip()
    # The document must carry the value: quoted words from it, or the value itself written in it.
    if not (evidence and evidence_is_grounded(evidence, document)) and _normalise(value) not in _normalise(document):
        return None, "not supported by the document"
    current = field.get("value")
    if not _is_empty(current) and _normalise(value) == _normalise(current if isinstance(current, str) else json.dumps(current, ensure_ascii=False)):
        return None, "same as the saved value"
    cite = document.locate(evidence) if evidence and hasattr(document, "locate") else None
    return {
        "kind": "field",
        "pointer": pointer,
        "label": _label(field, pointer),
        "section": field.get("section") or "",
        "value": value,
        "reason": str(item.get("reason") or "").strip()[:1000] or "Stated in the attached document.",
        "status": "empty" if _is_empty(current) else "improvement",
        "current": _preview(current, 400),
        "evidence": evidence[:400],
        "source": "document",
        **({"cite": cite} if cite else {}),
        "record_id": record_id,
    }, None


def summarize(checked: int, proposals: list[dict[str, Any]], dropped: int, failed: int, cancelled: bool) -> str:
    rows = sum(len(p.get("rows") or []) for p in proposals if p.get("kind") == "rows")
    proposals_ = [p for p in proposals if p.get("kind") != "rows"]
    empty = sum(1 for p in proposals_ if p["status"] == "empty")
    improved = len(proposals_) - empty
    if cancelled:
        head = f"Stopped after checking {checked} fields against the attached document."
    else:
        head = f"Checked {checked} fields against the attached document."
    if not proposals:
        body = "The document does not state a value for any of them that differs from what is saved."
    else:
        bits = []
        if empty:
            bits.append(f"**{empty}** empty field{'s' if empty != 1 else ''} it can fill")
        if improved:
            bits.append(f"**{improved}** saved value{'s' if improved != 1 else ''} it states more completely or differently")
        if rows:
            lists = len(proposals) - len(proposals_)
            bits.append(f"**{rows}** new table row{'s' if rows != 1 else ''} for {lists} list{'s' if lists != 1 else ''}")
        body = "Found " + " and ".join(bits) + ". Review each card, then fill the ones you accept."
    notes = []
    if dropped:
        notes.append(f"{dropped} draft{'s were' if dropped != 1 else ' was'} left out because the document does not support {'them' if dropped != 1 else 'it'}.")
    if failed:
        notes.append(f"{failed} batch{'es' if failed != 1 else ''} of fields could not be checked; run it again to retry them.")
    return " ".join([head, body, *notes])


async def run_document_fill(
    gateway: Gateway,
    spec: ModelSpec,
    bundle_raw: Optional[dict[str, Any]],
    document: str,
    *,
    slot: asyncio.Semaphore,
    is_cancelled: Callable[[], bool] = lambda: False,
    progress: Callable[[int, int], None] = lambda done, total: None,
) -> dict[str, Any]:
    """Run the fill and return what a review job stores: text, proposals, steps, timings."""
    started = time.monotonic()
    bundle = parse_bundle(bundle_raw)
    fields = {p: f for p, f in bundle.fields.items() if isinstance(f, dict)}
    targets = fill_targets(fields)
    lists = listfill.list_targets(fields)
    corpus = bundle_corpus(bundle, extra=document)
    proposals: list[dict[str, Any]] = []
    steps: list[dict[str, Any]] = []
    dropped = failed = checked = 0
    max_tokens = min(MAX_TOKENS, spec.max_tokens)
    batches = [("fields", targets[i : i + BATCH]) for i in range(0, len(targets), BATCH)]
    batches += [("rows", lists[i : i + listfill.BATCH]) for i in range(0, len(lists), listfill.BATCH)]
    total = len(targets) + len(lists)
    progress(0, total)

    # Batches are independent, so a hosted model takes several at once. A local model does one at a time:
    # its server is the limit. Results are merged in batch order, so the output does not depend on timing.
    width = max(1, min(FILL_PARALLEL_LOCAL if spec.local else FILL_PARALLEL, len(batches)))
    gate = asyncio.Semaphore(width)
    results: list[Optional[dict[str, Any]]] = [None] * len(batches)
    state = {"checked": 0, "failed": 0, "unusable": 0, "good": 0, "down": False}

    async def one(index: int, kind: str, pointers: list[str]) -> None:
        async with gate:
            if is_cancelled() or state["down"]:
                return
            name = "fill_from_document" if kind == "fields" else "fill_lists_from_document"
            build = batch_messages if kind == "fields" else listfill.batch_messages
            # Retrieval embeds each field's query (CPU): off the event loop, other users' streams keep flowing.
            messages = await asyncio.to_thread(build, fields, pointers, document)
            out: dict[str, Any] = {"steps": [{"stage": "call", "name": name, "args": {"pointers": pointers}}], "cards": [], "dropped": 0}
            async def ask(msgs: list[dict[str, str]], tokens: int, temperature: float = 0.0):
                async with slot:  # one call per slot: other users' turns get the model between calls
                    return await gateway.complete_ex(spec, msgs, max_tokens=tokens, temperature=temperature)

            budget = min(max_tokens if kind == "fields" else listfill.MAX_TOKENS, spec.max_tokens)
            if spec.local:
                messages = [{**messages[0], "content": messages[0]["content"] + BREVITY_NUDGE}]
            try:
                text, finish, thought = await ask(messages, budget)
                items = parse_proposals_strict(text)
                if items is None:
                    # No usable answer is not "nothing to propose". Ask once more, briefly, with the full budget.
                    logger.info(
                        "assistant docfill unusable reply model=%s finish=%s reasoning_chars=%d text_chars=%d: retrying",
                        spec.alias, finish, thought, len(text),
                    )
                    nudged = messages if messages[0]["content"].endswith(BREVITY_NUDGE) else [{**messages[0], "content": messages[0]["content"] + BREVITY_NUDGE}]
                    # A little temperature: at 0 the same prompt would fail the same way.
                    text, finish, thought = await ask(nudged, spec.max_tokens, 0.3)
                    items = parse_proposals_strict(text)
            except Exception as exc:  # noqa: BLE001 - one failed batch must not lose the others
                logger.warning("assistant: document fill batch failed (%s)", spec.alias, exc_info=True)
                gateway.mark_failed(spec)
                state["failed"] += 1
                out["steps"].append({"stage": "failed", "name": name, "message": type(exc).__name__})
                if state["failed"] >= 2 and not state["good"]:
                    state["down"] = True  # the model is down, not the batch
                results[index] = out
                return
            if items is None:
                # Still nothing usable: say it was not checked rather than reporting an empty result.
                logger.warning("assistant docfill batch unusable model=%s finish=%s reasoning_chars=%d", spec.alias, finish, thought)
                state["unusable"] += 1
                out["steps"].append({"stage": "failed", "name": name, "message": "NoUsableReply"})
                results[index] = out
                return
            seen: set[str] = set()
            for item in items:
                if kind == "rows":
                    card, why, refused = listfill.accept_rows(item, pointers, fields, document, corpus, bundle.record_id)
                    out["dropped"] += refused
                    if card is None:
                        continue
                else:
                    card, why = accept(item, pointers, fields, document, corpus, bundle.record_id)
                    if card is None:
                        # A value equal to the saved one is agreement, not a rejected draft.
                        if why != "same as the saved value":
                            out["dropped"] += 1
                            logger.info("assistant docfill dropped %s: %s", item.get("field", item.get("pointer")), str(why)[:80])
                        continue
                if card["pointer"] in seen:
                    continue
                seen.add(card["pointer"])
                out["cards"].append(card)
            out["steps"].append({"stage": "result", "name": name, "found": len(out["cards"])})
            state["good"] += 1
            state["checked"] += len(pointers)
            results[index] = out
            progress(state["checked"], total)

    await asyncio.gather(*(one(i, kind, pointers) for i, (kind, pointers) in enumerate(batches)))
    cancelled = is_cancelled() and any(r is None for r in results)
    failed = state["failed"] + state["unusable"]
    checked = state["checked"]
    for out in results:
        if out is None:
            continue
        steps.extend(out["steps"])
        proposals.extend(out["cards"])
        dropped += out["dropped"]

    total_ms = round((time.monotonic() - started) * 1000)
    logger.info(
        "assistant docfill model=%s fields=%d checked=%d proposals=%d dropped=%d failed=%d total_ms=%d",
        spec.alias, len(targets) + len(lists), checked, len(proposals), dropped, failed, total_ms,
    )
    if not checked and failed:
        return {"error": {"code": "model_error", "message": "The model did not answer. Check that it is running, then try again."}}
    return {
        "text": summarize(checked, proposals, dropped, failed, cancelled) if targets or lists else "This record has no field the assistant can fill.",
        "proposals": proposals,
        "steps": steps,
        "tool_calls": sum(1 for s in steps if s["stage"] == "call"),
        "timings": {"total_ms": total_ms},
        "model": spec.alias,
        "citations": [],
    }
