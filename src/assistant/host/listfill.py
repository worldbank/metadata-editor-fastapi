"""
Fill list fields (producers, funders, countries, keywords) from a document.

A list field holds rows of a few text cells. The document fill proposes rows to ADD: existing rows are
never changed or removed, a row whose main cell is already in the list is left out, and a row is kept
only if its main cell (the first column) is written in the document. Any other cell the document does
not carry is left empty rather than invented; dates, numbers and identifiers pass the same grounding
guard as a single value.
"""

from __future__ import annotations

import json
import re
from typing import Any, Optional

from src.assistant.core.documents import attached_document
from src.assistant.host.tools import _is_empty, _label, _normalise, ungrounded_tokens

# The local model thinks for ~3,500 tokens before it answers however few tables it is given, and ~1,000
# more for each: three tables at a 4,096 cap came back empty. Two per call fit under the model's own cap.
BATCH = 2
MAX_TOKENS = 6144
MAX_LISTS = 24
MAX_ROWS = 40
CELL_LIMIT = 300
EXCERPT_BYTES = 6000
CURRENT_PREVIEW = 400
GENERIC_KEYS = ("name", "value", "title", "term", "text", "item")

PROMPT = (
    "You list the people, organizations, places or terms a document names, as rows of a metadata table. "
    "For each numbered table below, return the rows the document states that the table does not already "
    "hold. Each row uses ONLY the column names listed for that table, spelled exactly. The first column is the row's main value "
    "and must be written in the document exactly. Leave a column out when the document does not give it; "
    "never guess a role, abbreviation or affiliation. Skip rows already in the table.\n"
    "Return ONLY JSON: {\"proposals\": [{\"field\": 1, \"rows\": [{\"<column>\": \"...\"}], \"evidence\": "
    "\"words copied from the document\", \"reason\": \"one short sentence\"}]}. Return {\"proposals\": []} "
    "when the document names nothing for these tables."
)


def list_targets(fields: dict[str, Any]) -> list[str]:
    """Pointers of list fields a row can be added to, empty ones first, in template order."""
    found = [p for p, f in fields.items() if isinstance(f, dict) and f.get("in_template") is not False and f.get("columns")]
    found.sort(key=lambda p: (not _is_empty(fields[p].get("value")), not fields[p].get("required"), not fields[p].get("recommended")))
    keep = set(found[:MAX_LISTS])
    return [p for p in fields if p in keep]


def _columns(field: dict[str, Any]) -> list[dict[str, str]]:
    return [c for c in field.get("columns") or [] if isinstance(c, dict) and c.get("key")]


def current_rows(field: dict[str, Any]) -> list[dict[str, Any]]:
    value = field.get("value")
    return [r for r in value if isinstance(r, dict)] if isinstance(value, list) else []


def _row_text(row: dict[str, Any]) -> str:
    return " ".join(str(v) for v in row.values() if v not in (None, ""))


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
        lines.append("   columns: " + "; ".join(f"{c['key']} ({c.get('title') or c['key']})" for c in _columns(f)))
        rows = current_rows(f)
        if rows:
            shown = json.dumps(rows, ensure_ascii=False)
            lines.append(f"   already in the table: {shown[:CURRENT_PREVIEW]}{'…' if len(shown) > CURRENT_PREVIEW else ''}")
        else:
            lines.append("   already in the table: (none)")
    excerpt, _ = doc.excerpt(query_parts, EXCERPT_BYTES)
    return [{"role": "user", "content": f"{PROMPT}\n\nTables:\n" + "\n".join(lines) + f"\n\nDocument:\n{excerpt}"}]


def _already_in(field: dict[str, Any], key: str):
    """A test for "this main value is already in the table"; a value cut short by the size cap is searched as text."""
    value = field.get("value")
    if isinstance(value, str):
        blob = _normalise(value)
        return lambda text: bool(blob) and text in blob
    known = {_normalise(str(r.get(key, ""))) for r in current_rows(field) if str(r.get(key, "")).strip()}
    return lambda text: text in known


def accept_rows(
    item: dict[str, Any], pointers: list[str], fields: dict[str, Any], document: str, corpus: str, record_id: Optional[str]
) -> tuple[Optional[dict[str, Any]], Optional[str], int]:
    """A proposal card for the new rows, why the item was dropped, and how many rows were refused."""
    from src.assistant.host.docfill import _resolve_pointer

    pointer = _resolve_pointer(item, pointers)
    if pointer is None:
        return None, "unknown field", 0
    field = fields[pointer]
    columns = _columns(field)
    if not columns:
        return None, "not a list field", 0
    keys = [c["key"] for c in columns]
    main = keys[0]
    raw = item.get("rows")
    if not isinstance(raw, list):
        return None, "no rows", 0
    haystack = _normalise(document)
    already = _already_in(field, main)
    seen: set[str] = set()
    rows: list[dict[str, str]] = []
    refused = 0
    for entry in raw[: MAX_ROWS * 2]:
        if isinstance(entry, str) and len(keys) == 1:
            entry = {main: entry}
        if not isinstance(entry, dict):
            refused += 1
            continue
        if isinstance(entry, dict) and not str(entry.get(main) or "").strip():
            # Small models answer a keywords table with "name" or "value"; that is the main cell.
            alias = next((g for g in GENERIC_KEYS if g not in keys and str(entry.get(g) or "").strip()), None)
            if alias:
                entry = {**entry, main: entry[alias]}
        row = {k: re.sub(r"\s+", " ", str(entry.get(k) or "")).strip()[:CELL_LIMIT] for k in keys}
        lead = _normalise(row[main])
        if not lead:
            refused += 1
            continue
        if already(lead) or lead in seen:
            continue  # agreement, not a rejected draft
        if lead not in haystack:
            refused += 1
            continue
        # Cells the document does not carry are left empty, not invented.
        for k in keys[1:]:
            if row[k] and (_normalise(row[k]) not in haystack or ungrounded_tokens(row[k], corpus)):
                row[k] = ""
        if ungrounded_tokens(row[main], corpus):
            refused += 1
            continue
        seen.add(lead)
        rows.append({k: v for k, v in row.items() if v})
        if len(rows) >= MAX_ROWS:
            break
    if not rows:
        return None, "no new rows", refused
    existing = current_rows(field)
    evidence = str(item.get("evidence") or "").strip()
    return (
        {
            "kind": "rows",
            "pointer": pointer,
            "label": _label(field, pointer),
            "section": field.get("section") or "",
            "columns": columns,
            "rows": rows,
            "value": "\n".join(_row_text(r) for r in rows),
            "reason": str(item.get("reason") or "").strip()[:1000] or "Named in the attached document.",
            "status": "improvement" if existing or (isinstance(field.get("value"), str) and field["value"].strip()) else "empty",
            "current": f"{len(existing)} row{'s' if len(existing) != 1 else ''} saved" if existing else None,
            "evidence": evidence[:400],
            "source": "document",
            "record_id": record_id,
        },
        None,
        refused,
    )
