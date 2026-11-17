"""Editor-specific one-shot task prompt builders."""

from __future__ import annotations

import json

from src.assistant.core.documents import attached_document
from src.assistant.host.editor import parse_bundle


DOC_MAX_BYTES = 12000


def _document_section(document, query=""):
    """The document, or for a long one its opening plus the passages about ``query`` (the field)."""
    if not (document or "").strip():
        return ""
    text, cut = attached_document(document).excerpt(query, DOC_MAX_BYTES)
    return (
        f"\n\nAttached document ({'excerpts; ' if cut else ''}extracted text; prefer it for facts):\n"
        f"{text}\n"
        "Use only what the document or the saved fields state. Do not add a name, country, period, "
        "threshold or number that neither source contains; say so in the reason when the sources do "
        "not settle the value."
    )


def datafile_description_messages(bundle_raw, document=None):
    bundle = parse_bundle(bundle_raw)
    df = bundle.datafile or {}
    variables = df.get("variables") or []
    names = [str(v.get("name") or "") for v in variables if isinstance(v, dict)]
    labels = [str(v.get("label") or "") for v in variables if isinstance(v, dict) and v.get("label")]
    prompt = (
        "Write a description for this data file in two to four plain sentences: what it contains, its unit of "
        "observation and geographic/time coverage. Base it only on the saved values below. "
        "Return ONLY a JSON object with keys 'suggestion' and 'reason'.\n\n"
        f"File name: {df.get('file_name') or ''}\n"
        f"Variables ({len(names)}): {', '.join(names[:80])}\n"
        f"Labels already saved: {', '.join(labels[:40])}\n"
        f"Existing description: {df.get('description') or '(none)'}\n"
        f"Project record: {json.dumps(bundle.record or {}, ensure_ascii=False)[:4000]}"
        f"{_document_section(document, 'data file contents unit observation coverage ' + ' '.join(labels[:20]))}"
    )
    return [{"role": "user", "content": prompt}]


def resource_field_messages(bundle_raw, pointer, document=None):
    bundle = parse_bundle(bundle_raw)
    field_pointer = pointer or ""
    field = (bundle.fields or {}).get(field_pointer) if isinstance(bundle.fields, dict) else None
    field = field if isinstance(field, dict) else {}
    current = field.get("value")
    help_text = field.get("help")
    options = field.get("options") or []
    suggestions = field.get("suggestions") or []
    empty = current in (None, "", [], {})
    source = "the attached document and this project's saved content" if (document or "").strip() else "this project's own saved content"
    query = " ".join(str(x) for x in (field.get("label"), field.get("section"), help_text, field_pointer) if x)
    out = (
        f"{'Draft a value' if empty else 'Draft an improved value'} for one metadata field of this project, as plain "
        f"text a curator can paste. Base it only on {source}. Return ONLY JSON with keys 'suggestion' and 'reason'.\n\n"
        f"Field pointer: {field_pointer}\n"
        f"Field label: {field.get('label') or field_pointer}\n"
        f"Current value: {'(empty)' if empty else json.dumps(current, ensure_ascii=False)}\n"
        f"Field help: {help_text or '(none)'}\n"
        f"Allowed options: {', '.join(str(o) for o in options[:30]) if options else '(free text)'}\n"
        f"Preferred values: {', '.join(str(o) for o in suggestions[:30]) if suggestions else '(none)'}\n"
        f"Project record: {json.dumps(bundle.record or {}, ensure_ascii=False)[:4000]}"
        f"{_document_section(document, query)}"
    )
    return [{"role": "user", "content": out}]
