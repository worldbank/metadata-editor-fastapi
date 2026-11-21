"""
Metadata Editor tools. They read from the bundle the editor's PHP proxy sends with each turn instead
of fetching: the proxy has already applied the session, the project ACL and private-field exclusion,
and this service holds no credentials to the editor.

The set is built for agentic work: survey what exists (list_fields, list_data_files), read in batches
(read_fields, find_variables, get_current_page), look things up (search_docs), then act through
proposals (propose_field_value, propose_datafile_description). There is still no write path: a proposal
is a `proposal` event the panel shows as a card, and only a curator's click fills the form.
"""

from __future__ import annotations

import asyncio
import functools

import json
import re
from typing import Any, Awaitable, Callable, Optional

from src.assistant.core.documents import attached_document
from src.assistant.core.prompt import byte_cap
from src.assistant.core.tools import Tool, ToolContext
from src.assistant.host import varindex

RECORD_MAX_BYTES = 24576
PAGE_MAX_BYTES = 16000
# Headroom under PAGE_MAX_BYTES so a rounding difference between the two measurements never trims twice.
PAGE_BUDGET = 15600
LIST_MAX_BYTES = 12000
READ_MAX_BYTES = 14000
VALUE_PREVIEW = 1500
MAX_READ = 15
# Models ask for many fields at once in a review (observed: LFM2.5-2.6B listing 20+ pointers in one
# call). The schema accepts up to MAX_ASK and the tool reads MAX_READ, returning the rest as not read
# yet, instead of rejecting the whole call and watching the model repeat it.
MAX_ASK = 60
MAX_LISTED = 120
DOCUMENT_HITS = 4
DOCUMENT_RESULT_BYTES = 6000

AGENT_ADDENDUM = (
    "You can act on this project with tools. For a task (review the record, find what is missing, improve or "
    "draft a field, describe a data file), work in steps: first gather what you need (list_fields to see which "
    "fields exist and which are empty or required, read_fields to read several values at once, find_variables "
    "or get_current_page for data files, search_docs for how the editor or the DDI standard defines something), "
    "then act, then finish with a short summary of what you found and what you proposed. A proposal may target "
    "only a pointer returned by list_fields(filter=\"fillable\"): that list is the controlled set of fields the "
    "record's own form can accept a single value for. Prefer one read of "
    "several fields over many single reads. Do not ask the user for anything a tool can read. Never "
    "invent or guess a value that is not literally written in a tool result or the supplied documentation; "
    "when the record is silent, leave the field or say it is missing."
)

PROPOSE_ADDENDUM = (
    "When the user asks for a suggestion, improvement, draft or review, or your analysis finds an empty required "
    "or recommended field worth filling or saved values that contradict each other, propose the value with "
    "propose_field_value, one call per field, based only on this project's own content, quoting the saved values "
    "you relied on in the reason. For a controlled list, propose one of its options exactly. Never invent a date, "
    "version, name, number, identifier or any other fact that is not written somewhere in this project: when the "
    "project does not say, list the field as missing in your summary instead of proposing a value. Every date, number and "
    "identifier in a proposal must be copied exactly from a value you read; pass the supporting words you copied as "
    "evidence. When a field lists preferred values, propose exactly one of them. A proposal is never applied by "
    "itself; the curator fills or copies it, so never say a value was saved or changed."
)

DOCUMENT_ADDENDUM = (
    "The curator attached a document; for this turn it counts as this project's own content. Use search_document "
    "to find what it says about a field. Propose a value for an empty field the document states, and an improvement "
    "for a saved value the document states more completely or differently (status 'improvement' or 'inconsistent'). "
    "Copy names, dates and numbers exactly as the document writes them and pass the copied words as evidence."
)


def _size(value: Any) -> int:
    return len(json.dumps(value, ensure_ascii=False).encode("utf-8"))


def fit_tool_result(result: dict[str, Any], max_bytes: int) -> dict[str, Any]:
    """
    Fit a page result inside a tool's byte budget. Ported from portable-slm ``fitToolResult``: ``record``
    (text) is cut by quarters, ``page.variables`` loses its tail, and a result that still cannot fit
    becomes an error the model can relay instead of a failed turn.
    """
    out = dict(result)
    if isinstance(out.get("record"), str):
        while len(out["record"]) > 256 and _size(out) > max_bytes:
            out["record"] = out["record"][: max(256, int(len(out["record"]) * 0.75))]
            out["truncated"] = True
    page = out.get("page")
    if isinstance(page, dict) and isinstance(page.get("variables"), list):
        variables = page["variables"]
        keep = len(variables)
        while keep > 0 and _size(out) > max_bytes:
            keep -= max(1, -(-keep // 4))
            out["page"] = {**page, "variables": variables[:keep], "variablesTruncated": True}
    if _size(out) > max_bytes:
        return {"error": "The current view is too large to load in one request; ask about a single field instead."}
    return out


def _label(field: dict[str, Any], pointer: str) -> str:
    return str(field.get("label") or pointer)


def _is_empty(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    if isinstance(value, (list, dict)):
        return not value
    return False


def _preview(value: Any, limit: int = VALUE_PREVIEW) -> Any:
    if isinstance(value, str):
        return value if len(value) <= limit else value[:limit] + "…"
    if isinstance(value, (list, dict)):
        text = json.dumps(value, ensure_ascii=False)
        return value if len(text) <= limit else text[:limit] + "…"
    return value


def _words(text: str) -> list[str]:
    return [w for w in re.split(r"[^a-z0-9]+", (text or "").lower()) if len(w) > 1]


_OPTION_CODE = re.compile(r"\s*[\[(]([^\])]*)[\])]\s*$")


def _match_option(value: str, options: list[str]) -> Optional[str]:
    """
    The option ``value`` names: exactly, or by its label or code alone when the option carries one in
    brackets ("Face-to-face [f2f]" for "face-to-face" or "f2f"), as models drop the code.
    """
    wanted = value.strip().casefold()
    for opt in options:
        if opt.strip().casefold() == wanted:
            return opt
    bare = _OPTION_CODE.sub("", wanted).strip()
    for opt in options:
        m = _OPTION_CODE.search(opt)
        if m and bare in {_OPTION_CODE.sub("", opt).strip().casefold(), m.group(1).strip().casefold()}:
            return opt
    return None


# Tokens that must exist somewhere in the project before a draft may contain them: dates, years, long
# numbers, and identifier-like codes (letters and digits together). This is the mechanical half of "do
# not invent dates, versions or identifiers"; ordinary prose is left to the prompt and the evidence rule.
# Durations are facts too: "six months" was invented in a Universe draft even though every digit in
# the record matched. Spelled numbers before a time unit are checked the same way as dates.
_GROUNDED_DURATION = re.compile(
    r"\b(?:one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\s+"
    r"(?:month|months|year|years|day|days|week|weeks)\b",
    re.IGNORECASE,
)

_GROUNDED_TOKEN = re.compile(
    r"\b\d{4}-\d{2}-\d{2}\b"
    r"|\b(?:19|20)\d{2}\b"
    r"|\b\d{3,}\b"
    r"|\b(?=[A-Za-z0-9_-]{2,}\b)(?=[A-Za-z0-9_-]*\d)(?=[A-Za-z0-9_-]*[A-Z])[A-Za-z0-9_-]+\b"
)


def bundle_corpus(bundle: Any, extra: str = "") -> str:
    """Everything the model may treat as source: the record, the page, every declared field."""
    parts = [extra or "", json.dumps(bundle.record or {}, ensure_ascii=False), json.dumps(bundle.datafile or {}, ensure_ascii=False)]
    parts.append(json.dumps(bundle.datafiles or [], ensure_ascii=False))
    for pointer, field in (bundle.fields or {}).items():
        if not isinstance(field, dict):
            continue
        parts.extend(
            [
                pointer,
                str(field.get("label") or ""),
                str(field.get("section") or ""),
                str(field.get("help") or ""),
                json.dumps(field.get("value"), ensure_ascii=False),
                json.dumps(field.get("options") or [], ensure_ascii=False),
            ]
        )
    return "\n".join(parts)


def _normalise(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").casefold()).strip()


_WORD = re.compile(r"[^\W_]+")
_HYPHEN_BREAK = re.compile(r"(\w)-[ \t]*\n[ \t]*(\w)")
_ELISION = re.compile(r"\.{3,}|\u2026|\[\s*\.{3}\s*\]|\[\s*\u2026\s*\]")
_RUN = 5            # words in a run that must be found in the corpus
_RUN_SHARE = 0.85   # share of a long quote's runs that must be found (page headers, footnote marks and the like)


def _words_of(text: str) -> list[str]:
    return _WORD.findall(_HYPHEN_BREAK.sub(r"\1\2", text).casefold())


@functools.lru_cache(maxsize=4)
def _flat_words(corpus: str) -> str:
    """The corpus as one string of words, for exact phrase tests."""
    return " " + " ".join(_words_of(corpus)) + " "


@functools.lru_cache(maxsize=2)
def _word_runs(corpus: str) -> frozenset[tuple[str, ...]]:
    """The corpus's 5-word runs. Only built when a long quote is not found word for word."""
    words = _words_of(corpus)
    return frozenset(tuple(words[i : i + _RUN]) for i in range(len(words) - _RUN + 1))


def _fragment_is_in(fragment: str, corpus: str) -> bool:
    words = _words_of(fragment)
    if not words:
        return True
    if " " + " ".join(words) + " " in _flat_words(corpus):
        return True  # the words, in order, as written (case, spacing, punctuation and line breaks aside)
    if len(words) < 2 * _RUN:
        return False  # a short quote must be exact
    runs = _word_runs(corpus)
    quote_runs = [tuple(words[i : i + _RUN]) for i in range(len(words) - _RUN + 1)]
    return sum(r in runs for r in quote_runs) / len(quote_runs) >= _RUN_SHARE


def evidence_is_grounded(evidence: str, corpus: str) -> bool:
    """
    Every quoted fragment must come from the project's text. Models often quote several, separated by ``;``
    or by an ellipsis. A fragment counts when its words appear in order (ignoring case, spacing, punctuation
    and the hyphen a PDF line break leaves), or, for a long quote, when nearly all of its 5-word runs do, so
    a page header or footnote number inside the quote does not reject a real passage. A paraphrase or an
    invented sentence shares few runs with the text and is refused.
    """
    corpus = str(corpus or "")
    pieces = [part for chunk in _ELISION.split(evidence or "") for part in re.split(r"[;\n\u2022|]+", chunk)]
    fragments = [part.strip(' \t"\'\u201c\u201d\u2018\u2019.,\u2026') for part in pieces]
    fragments = [part for part in fragments if part]
    return bool(fragments) and all(_fragment_is_in(part, corpus) for part in fragments)


def ungrounded_tokens(value: Any, corpus: str) -> list[str]:
    """Dates, years, long numbers and identifier-like codes in ``value`` that ``corpus`` does not contain."""
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
    haystack = _normalise(corpus)
    out: list[str] = []
    for token in _GROUNDED_TOKEN.findall(text or ""):
        bare = token.rstrip(".,;:)")
        if _normalise(bare) in haystack or _normalise(token) in haystack:
            continue
        if bare and bare not in out:
            out.append(bare)
    for phrase in _GROUNDED_DURATION.findall(text or ""):
        if _normalise(phrase) not in haystack and phrase not in out:
            out.append(phrase)
    return out


def check_field_value(
    field: Optional[dict[str, Any]], pointer: str, value: Any, evidence: Any, corpus: str
) -> tuple[str, Optional[str]]:
    """
    The guard every field proposal passes, from chat or from a document fill: the pointer is a fillable
    field of this record, a controlled list gets one of its options, and quoted evidence, dates, numbers
    and identifiers all appear in ``corpus``. Returns (value to propose, None) or ("", error).
    """
    if field is None:
        return "", f"Pointer {pointer} is not a field of this project. Use list_fields to find the pointer."
    if field.get("in_template") is False:
        return "", (
            f"This record's template has no field {pointer}; it belongs to a different record type. "
            "Propose only fields that exist in this record."
        )
    if not field.get("fillable", True):
        return "", f"'{_label(field, pointer)}' is a list or table field; a single value cannot fill it. Describe the change in your answer instead."
    value = str(value if value is not None else "")
    if not value.strip():
        return "", "The proposed value is empty. Propose the actual text the field should contain."
    options = field.get("options") or []
    allowed = options or field.get("suggestions") or []
    if allowed:
        chosen = _match_option(value, allowed)
        if chosen is None:
            kind = "a controlled list" if options else "preferred values"
            return "", f"'{_label(field, pointer)}' takes {kind}. Propose exactly one of: {', '.join(allowed[:30])}"
        value = chosen
    evidence = str(evidence or "").strip()
    if evidence and not evidence_is_grounded(evidence, corpus):
        return "", (
            "The evidence you quoted does not appear in this project. Copy the supporting words exactly "
            "from a field you read, or drop them and propose without evidence."
        )
    ungrounded = ungrounded_tokens(value, corpus)
    if ungrounded:
        return "", (
            f"The value contains {', '.join(ungrounded)}, which this project never states. Copy dates, "
            "numbers and identifiers exactly from saved values, or leave them out."
        )
    return value, None


def build_editor_tools(
    bundle: Any,
    *,
    fill: bool,
    search_docs: Optional[Callable[[str], Awaitable[tuple[str, dict[str, Any]]]]] = None,
    document: Optional[str] = None,
) -> list[Tool]:
    """Tools for one turn on a record page. ``bundle`` is an ``EditorBundle``."""
    if not bundle.record_id:
        return []
    fields: dict[str, dict[str, Any]] = {p: f for p, f in bundle.fields.items() if isinstance(f, dict)}
    corpus = bundle_corpus(bundle, extra=document or "")
    proposed: set[str] = set()
    tools: list[Tool] = []

    if fields:

        async def list_fields(args: dict[str, Any], ctx: ToolContext) -> Any:
            which = args.get("filter", "all")
            words = _words(args.get("query", ""))
            rows = []
            for pointer, f in fields.items():
                empty = _is_empty(f.get("value"))
                if which == "empty" and not empty:
                    continue
                if which == "filled" and empty:
                    continue
                if which == "required" and not f.get("required"):
                    continue
                if which == "recommended" and not (f.get("recommended") or f.get("required")):
                    continue
                if which == "fillable" and (f.get("in_template") is False or not f.get("fillable", True)):
                    continue
                if words:
                    hay = f"{f.get('label', '')} {f.get('section', '')} {pointer}".lower()
                    if not all(w in hay for w in words):
                        continue
                row = {"pointer": pointer, "label": _label(f, pointer), "section": f.get("section") or "", "empty": empty}
                if f.get("required"):
                    row["required"] = True
                elif f.get("recommended"):
                    row["recommended"] = True
                if not f.get("fillable", True):
                    row["read_only"] = True
                rows.append(row)
            total = len(rows)
            out = {"total": total, "fields": rows[:MAX_LISTED]}
            while len(out["fields"]) > 1 and _size(out) > LIST_MAX_BYTES:
                out["fields"] = out["fields"][: int(len(out["fields"]) * 0.75)]
            if len(out["fields"]) < total:
                out["note"] = f"showing {len(out['fields'])} of {total}; narrow with filter or query"
            return out

        tools.append(
            Tool(
                name="list_fields",
                description=(
                    "List this project's metadata fields with their pointer, label, section, and whether each is empty, "
                    "required or recommended. Use filter='fillable' for the exact set a proposal may target. "
                    "Use it first to see what exists and what is missing, then read_fields."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "filter": {
                            "type": "string",
                            "enum": ["all", "empty", "filled", "required", "recommended", "fillable"],
                            "description": "Which fields to list; fillable means a single value can fill it (recommended includes required)",
                        },
                        "query": {"type": "string", "description": "Optional words to match in the label or section"},
                    },
                    "required": [],
                },
                run=list_fields,
                max_result_bytes=LIST_MAX_BYTES + 400,
            )
        )

        async def read_fields(args: dict[str, Any], ctx: ToolContext) -> Any:
            out = []
            unknown = []
            asked = list(dict.fromkeys(args["pointers"]))
            for pointer in asked[:MAX_READ]:
                f = fields.get(pointer)
                if f is None:
                    unknown.append(pointer)
                    continue
                item = {
                    "pointer": pointer,
                    "label": _label(f, pointer),
                    "value": _preview(f.get("value")),
                    "empty": _is_empty(f.get("value")),
                }
                if f.get("help"):
                    item["help"] = f["help"]
                if f.get("options"):
                    item["options"] = f["options"]
                if f.get("suggestions"):
                    item["suggestions"] = f["suggestions"]
                out.append(item)
            result: dict[str, Any] = {"fields": out}
            if len(asked) > MAX_READ:
                result["not_read_yet"] = asked[MAX_READ:]
                result["note"] = f"read {MAX_READ} at a time; call read_fields again for not_read_yet if still needed"
            if unknown:
                result["unknown"] = unknown
                result["hint"] = "Use list_fields (with a query) to find the right pointer."
            while len(result["fields"]) > 1 and _size(result) > READ_MAX_BYTES:
                result["fields"] = result["fields"][:-1]
                result["note"] = "some fields left out to fit; read them separately"
            return result

        tools.append(
            Tool(
                name="read_fields",
                description=(
                    "Read the saved values of several fields at once (15 per call), with each field's help text and, for a "
                    "controlled list, its options. Pass pointers from list_fields."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "pointers": {
                            "type": "array",
                            "items": {"type": "string"},
                            "maxItems": MAX_ASK,
                            "description": "JSON Pointers, e.g. ['/study_desc/study_info/abstract']",
                        }
                    },
                    "required": ["pointers"],
                },
                run=read_fields,
                max_result_bytes=READ_MAX_BYTES + 400,
            )
        )

    if bundle.datafile or bundle.record:

        async def get_current_page(args: dict[str, Any], ctx: ToolContext) -> Any:
            if bundle.datafile:
                return fit_tool_result({"view": bundle.section or "data file", "page": bundle.datafile}, PAGE_BUDGET)
            record_text, cut = byte_cap(json.dumps(bundle.record, ensure_ascii=False), RECORD_MAX_BYTES)
            out = fit_tool_result({"view": bundle.section or "project", "record": record_text}, PAGE_BUDGET)
            if cut:
                out["truncated"] = True
            return out

        tools.append(
            Tool(
                name="get_current_page",
                description=(
                    "Load what the user is looking at: the open data file's metadata and variables, or otherwise the whole "
                    "project record. Prefer list_fields and read_fields for specific fields."
                ),
                parameters={"type": "object", "properties": {}, "required": []},
                run=get_current_page,
                max_result_bytes=PAGE_MAX_BYTES,
            )
        )

    if bundle.datafiles:

        async def list_data_files(args: dict[str, Any], ctx: ToolContext) -> Any:
            return {"open_file": bundle.file_id, "data_files": bundle.datafiles}

        tools.append(
            Tool(
                name="list_data_files",
                description="List the project's data files with their names, case and variable counts, and descriptions.",
                parameters={"type": "object", "properties": {}, "required": []},
                run=list_data_files,
                max_result_bytes=LIST_MAX_BYTES,
            )
        )

    datafile = bundle.datafile if isinstance(bundle.datafile, dict) else None
    variables = datafile.get("variables") if datafile else None
    if isinstance(variables, list) and variables:

        async def find_variables(args: dict[str, Any], ctx: ToolContext) -> Any:
            query = str(args.get("query", "") or "")
            words = _words(query)
            which = args.get("missing")

            def wanted(v: Any) -> bool:
                if not isinstance(v, dict):
                    return False
                if which == "label" and str(v.get("label") or "").strip():
                    return False
                if which == "description" and str(v.get("description") or "").strip():
                    return False
                return True

            if query.strip():
                # Ranked by meaning and by word (name, label, description, question), best first. Substring
                # matching stays as the fallback for a query the index finds nothing for.
                ranked = await asyncio.to_thread(varindex.for_variables(variables).search, query, len(variables))
                rows = [v for v in ranked if wanted(v)]
                if not rows and words:
                    rows = [
                        v for v in variables
                        if wanted(v) and any(w in " ".join(str(v.get(k) or "") for k in ("name", "label", "description", "question")).lower() for w in words)
                    ]
            else:
                rows = [v for v in variables if wanted(v)]
            out = {"file": datafile.get("file_name"), "matches": len(rows), "variables": rows[:60]}
            while len(out["variables"]) > 1 and _size(out) > LIST_MAX_BYTES:
                out["variables"] = out["variables"][: int(len(out["variables"]) * 0.75)]
            if len(out["variables"]) < len(rows):
                out["note"] = f"showing {len(out['variables'])} of {len(rows)}"
            return out

        tools.append(
            Tool(
                name="find_variables",
                description=(
                    "Search the open data file's variables by topic or words (name, label, description, question; "
                    "'firm size' finds a variable labelled 'number of employees'), best match first, or list the "
                    "ones missing a label or a description."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Words to look for"},
                        "missing": {"type": "string", "enum": ["label", "description"], "description": "Only variables missing this"},
                    },
                    "required": [],
                },
                run=find_variables,
                max_result_bytes=LIST_MAX_BYTES + 400,
            )
        )

    if search_docs is not None:

        async def search_documentation(args: dict[str, Any], ctx: ToolContext) -> Any:
            text, info = await search_docs(args["query"])
            if not text:
                return {"found": False, "note": "No matching documentation section."}
            return {"found": True, "sections": info.get("hits", []), "text": text}

        tools.append(
            Tool(
                name="search_docs",
                description=(
                    "Search the Metadata Editor's documentation, the DDI schema guide and the API reference for how a field "
                    "or feature is defined or used. Each turn already includes a few relevant sections; use this for "
                    "a different or more specific question."
                ),
                parameters={
                    "type": "object",
                    "properties": {"query": {"type": "string", "description": "What to look up"}},
                    "required": ["query"],
                },
                run=search_documentation,
                max_result_bytes=8000,
            )
        )

    if document and document.strip():
        attached = attached_document(document)

        async def search_document(args: dict[str, Any], ctx: ToolContext) -> Any:
            hits = await asyncio.to_thread(attached.search, str(args.get("query") or ""), DOCUMENT_HITS)
            if not hits:
                return {"found": False, "note": "No passage of the project's sources matches; try other words."}
            label = getattr(attached, "label", None)
            passages = []
            for c in hits:
                item = {"passage": c.index + 1, "text": attached.passage(c.index)}
                if label is not None:
                    item["source"] = label(c)
                passages.append(item)
            out = {"found": True, "passages": passages}
            while len(out["passages"]) > 1 and _size(out) > DOCUMENT_RESULT_BYTES:
                out["passages"] = out["passages"][:-1]
            return out

        tools.append(
            Tool(
                name="search_document",
                description=(
                    "Search the sources the curator added (a questionnaire, report, manual or other project file; each "
                    "result names its source and page) for the passages about a field or topic, e.g. 'sampling procedure' "
                    "or 'reference period'. The turn shows only a few passages; use this before saying the sources do "
                    "not state something."
                ),
                parameters={
                    "type": "object",
                    "properties": {"query": {"type": "string", "description": "Words to look for, such as a field's label"}},
                    "required": ["query"],
                },
                run=search_document,
                max_result_bytes=DOCUMENT_RESULT_BYTES + 400,
            )
        )

    if fill and fields:

        async def propose_field_value(args: dict[str, Any], ctx: ToolContext) -> Any:
            pointer = args["pointer"]
            field = fields.get(pointer)
            value, error = check_field_value(field, pointer, args["value"], args.get("evidence"), corpus)
            if error:
                # A correction the model can act on this round, not a refusal.
                return {"error": error}
            proposal = {
                "kind": "field",
                "pointer": pointer,
                "label": _label(field, pointer),
                "section": field.get("section") or "",
                "value": value,
                "reason": args.get("reason", ""),
                "status": args.get("status"),
                "current": _preview(field.get("value"), 400),
                "record_id": bundle.record_id,
            }
            if pointer in proposed:
                # Seen in a review: one field proposed twice, two cards for one decision.
                return {"error": "This field was already proposed in this turn. Do not propose it again."}
            proposed.add(pointer)
            ctx.emit("proposal", proposal)
            return "proposed - shown to the curator as a card with Fill in form and Copy; continue with the task"

        tools.append(
            Tool(
                name="propose_field_value",
                description=(
                    "Propose a value for one field of this project. One call per field. The curator sees each proposal as a "
                    "card and decides; nothing is applied by itself."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "pointer": {"type": "string", "description": "JSON Pointer of the field, from list_fields"},
                        "value": {"type": "string", "maxLength": 6000, "description": "The proposed value, based only on this project's own content"},
                        "reason": {
                            "type": "string",
                            "maxLength": 1000,
                            "description": "One sentence: the saved values you relied on (quote them) and why this value fits",
                        },
                        "status": {
                            "type": "string",
                            "enum": ["empty", "inconsistent", "improvement"],
                            "description": "Whether the field is empty, contradicts other fields, or could be clearer",
                        },
                        "evidence": {
                            "type": "string",
                            "maxLength": 400,
                            "description": "The saved words this value relies on, copied exactly from what you read",
                        },
                    },
                    "required": ["pointer", "value", "reason"],
                },
                run=propose_field_value,
            )
        )

    if fill and datafile and bundle.file_id:

        async def propose_datafile_description(args: dict[str, Any], ctx: ToolContext) -> Any:
            value = str(args["value"])
            if not value.strip():
                return {"error": "The proposed description is empty."}
            ungrounded = ungrounded_tokens(value, corpus)
            if ungrounded:
                return {
                    "error": f"The description contains {', '.join(ungrounded)}, which this project never states. Copy "
                    "dates, numbers and identifiers exactly from the data file, or leave them out."
                }
            ctx.emit(
                "proposal",
                {
                    "kind": "datafile",
                    "file_id": bundle.file_id,
                    "pointer": f"datafile:{bundle.file_id}:description",
                    "label": f"Description of {datafile.get('file_name') or bundle.file_id}",
                    "value": value,
                    "reason": args.get("reason", ""),
                    "status": "empty" if not str(datafile.get("description") or "").strip() else "improvement",
                    "current": _preview(datafile.get("description"), 400),
                    "record_id": bundle.record_id,
                },
            )
            return "proposed - shown to the curator as a card with Fill in form and Copy"

        tools.append(
            Tool(
                name="propose_datafile_description",
                description=(
                    "Propose a description for the open data file (what it contains, its unit and coverage), based on its "
                    "variables. The curator decides; nothing is applied by itself."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "value": {"type": "string", "maxLength": 3000, "description": "Two to four plain sentences"},
                        "reason": {"type": "string", "description": "Which variables or saved values it is based on"},
                    },
                    "required": ["value", "reason"],
                },
                run=propose_datafile_description,
            )
        )
    return tools
