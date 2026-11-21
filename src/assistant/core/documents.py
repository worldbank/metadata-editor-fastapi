"""
Text extraction for an uploaded attachment already stored under STORAGE_PATH, and passage selection
over that text once it is attached to a turn.

Only what the assistant needs to read a document a curator asked about: PDF (pypdf), plain text and
Markdown as-is. The path is validated by the caller (``resolve_safe_path_http``), never here.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

MAX_CHARS = 40000
# Indexed sources are read in full up to this many characters (a 139-page codebook is about 110,000);
# MAX_CHARS above stays the cap for the one-shot ``extract_text`` route that returns the text itself.
SOURCE_MAX_CHARS = 2_000_000
TEXT_SUFFIXES = {".txt", ".md", ".csv", ".tsv", ".json", ".xml", ".html", ".htm"}


class ExtractError(Exception):
    """A file the assistant cannot read; the message is safe to show."""


def extract_text(path: str | Path, max_chars: int = MAX_CHARS) -> dict[str, object]:
    p = Path(path)
    if not p.is_file():
        raise ExtractError("The file does not exist.")
    suffix = p.suffix.lower()
    if suffix == ".pdf":
        text, pages = _pdf_text(p)
    elif suffix in TEXT_SUFFIXES:
        text, pages = p.read_text(encoding="utf-8", errors="replace"), None
    else:
        raise ExtractError(f"Cannot read {suffix or 'this file type'}; use PDF, text or Markdown.")
    cut = len(text) > max_chars
    return {
        "text": text[:max_chars],
        "chars": min(len(text), max_chars),
        "pages": pages,
        "truncated": cut,
    }


def extract_pages(path: str | Path, max_chars: int = SOURCE_MAX_CHARS) -> dict[str, object]:
    """
    Whole-document text kept page by page, for indexing: ``pages`` is [(number, text)], the number 0 for
    a source with no pages. Stops at ``max_chars`` and says so (``truncated``), never silently.
    """
    p = Path(path)
    if not p.is_file():
        raise ExtractError("The file does not exist.")
    suffix = p.suffix.lower()
    pages: list[tuple[int, str]] = []
    total = 0
    cut = False
    count = None
    if suffix == ".pdf":
        try:
            from pypdf import PdfReader
        except ImportError:  # pragma: no cover - optional dependency
            raise ExtractError("PDF reading is not installed on the assistant service.")
        try:
            reader = PdfReader(str(p))
            count = len(reader.pages)
            for number, page in enumerate(reader.pages, 1):
                text = (page.extract_text() or "").strip()
                if not text:
                    continue
                if total + len(text) > max_chars:
                    text, cut = text[: max(0, max_chars - total)], True
                pages.append((number, text))
                total += len(text)
                if cut:
                    break
        except ExtractError:
            raise
        except Exception as exc:  # noqa: BLE001 - a corrupt file is information for the caller
            logger.info("assistant: PDF extraction failed for %s", p.name, exc_info=True)
            raise ExtractError(f"Could not read this PDF ({type(exc).__name__}).") from exc
    elif suffix in TEXT_SUFFIXES:
        text = p.read_text(encoding="utf-8", errors="replace").strip()
        if len(text) > max_chars:
            text, cut = text[:max_chars], True
        if text:
            pages.append((0, text))
        total = len(text)
    else:
        raise ExtractError(f"Cannot read {suffix or 'this file type'}; use PDF, text or Markdown.")
    if not pages:
        raise ExtractError("No text could be read from this file. A scanned PDF has no text layer; it needs OCR first.")
    return {"pages": pages, "page_count": count, "chars": total, "truncated": cut}


def _pdf_text(p: Path) -> tuple[str, int]:
    try:
        from pypdf import PdfReader
    except ImportError:  # pragma: no cover - optional dependency
        raise ExtractError("PDF reading is not installed on the assistant service.")
    try:
        reader = PdfReader(str(p))
        text = "\n\n".join(page.extract_text() or "" for page in reader.pages)
    except Exception as exc:  # noqa: BLE001 - a corrupt file is information for the caller
        logger.info("assistant: PDF extraction failed for %s", p.name, exc_info=True)
        raise ExtractError(f"Could not read this PDF ({type(exc).__name__}).") from exc
    return text.strip(), len(reader.pages)


# ---------------------------------------------------------------- passages of an attached document

PASSAGE_CHARS = 900
_GAP = "\n\n[…]\n\n"


class AttachedDocument:
    """
    An attached document split into passages with a keyword index, so a turn can carry the passages
    that answer it instead of only the first pages. Built once per distinct text (``attached_document``).
    """

    def __init__(self, text: str, passage_chars: int = PASSAGE_CHARS) -> None:
        from src.assistant.core.retrieval import Index, chunk_text

        self.text = (text or "").strip()
        self.chunks = chunk_text(self.text, path="document", target_chars=passage_chars, overlap_chars=0)
        self._index = Index(self.chunks, key="document")

    def passage(self, pos: int) -> str:
        """A passage with its section heading on the section's first passage (the chunker keeps headings apart)."""
        chunk = self.chunks[pos]
        first = pos == 0 or self.chunks[pos - 1].heading != chunk.heading
        if first and chunk.heading and not chunk.text.startswith(chunk.heading):
            return f"{chunk.heading}\n{chunk.text}"
        return chunk.text

    def search(self, query: str, k: int = 4) -> list[Any]:
        """Best passages for ``query`` in score order; empty when no query word occurs."""
        from src.assistant.core.retrieval import RetrievalOptions, bm25

        scores = bm25(self._index, query, RetrievalOptions())
        ranked = sorted(scores, key=lambda pos: (-scores[pos], pos))
        return [self.chunks[pos] for pos in ranked[:k]]

    def excerpt(self, query: "str | list[str]", max_bytes: int) -> tuple[str, bool]:
        """
        The whole text when it fits; otherwise the opening passage (title page, abstract) plus the
        passages ``query`` matches best, in document order, up to ``max_bytes``. Returns (text, cut).
        """
        if isinstance(query, list):
            query = " ".join(query)
        if len(self.text.encode("utf-8")) <= max_bytes:
            return self.text, False
        if not self.chunks:
            return "", True
        picked = [0]
        for chunk in self.search(query, k=len(self.chunks)):
            if chunk.index not in picked:
                picked.append(chunk.index)
        # Unmatched passages follow in reading order, so a vague question still gets the opening pages.
        picked.extend(i for i in range(len(self.chunks)) if i not in picked)
        chosen: list[int] = []
        used = 0
        for pos in picked:
            size = len(self.passage(pos).encode("utf-8")) + len(_GAP)
            if used + size > max_bytes:
                continue
            chosen.append(pos)
            used += size
        chosen.sort()
        parts: list[str] = []
        for n, pos in enumerate(chosen):
            if n and pos != chosen[n - 1] + 1:
                parts.append("[…]")
            parts.append(self.passage(pos))
        return "\n\n".join(parts), True


@lru_cache(maxsize=16)
def _attached_text(text: str) -> AttachedDocument:
    return AttachedDocument(text)


def attached_document(document: Any) -> Any:
    """
    The searchable form of an attachment. Raw text is chunked and indexed once (chat turns resend the
    same attachment); a ``KnowledgeView`` (indexed project sources) is already searchable and returned as is.
    """
    if hasattr(document, "excerpt") and hasattr(document, "passage"):
        return document
    return _attached_text(document)
