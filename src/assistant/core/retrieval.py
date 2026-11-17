"""
Hybrid retrieval over a host's own documents: BM25 always, embeddings when the embedder is loaded.

Ported from portable-slm ``src/retrieval.js`` (chunker, tokenizer, BM25 with the common-term cutoff,
min-max fusion, similarity floor, byte-capped selection) and ``integrations/retrieval-context.js`` (BM25
answers from the first question while embeddings build in the background; a corpus that cannot be read
degrades to no retrieval instead of failing the turn).

The index is frozen to disk with a key over the corpus content, chunker and embedder, so a restart
reuses it and any change rebuilds it.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import math
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)

CHUNKER_VERSION = 1
INDEX_FORMAT = "pslm-server-index/1"


@dataclass(frozen=True)
class RetrievalOptions:
    target_chars: int = 1200
    overlap_chars: int = 150
    max_chunks_per_doc: int = 20000
    top_k: int = 5
    max_bytes: int = 6144
    alpha: float = 0.5
    k1: float = 1.2
    b: float = 0.75
    # Floor for chunks no query term matched. None = the embedder's own measured floor.
    min_similarity: Optional[float] = None


@dataclass
class Chunk:
    path: str
    index: int
    heading: str
    text: str
    tokens: list[str] = field(default_factory=list, repr=False)
    # Page of the source document (1-based); 0 when the source has no pages (text, Markdown).
    page: int = 0


# ---------------------------------------------------------------- chunking and tokens

_HEADING = re.compile(r"^(#{1,6})\s+(.*\S)\s*$")


def chunk_text(text: str, *, path: str = "", target_chars: int = 1200, overlap_chars: int = 150, max_chunks: int = 20000) -> list[Chunk]:
    """Headings are the primary boundary; long sections split on paragraph, line, then word, with overlap."""
    source = re.sub(r"\r\n?", "\n", text or "").strip()
    if not source:
        return []
    target = max(200, target_chars)
    sections: list[tuple[str, str]] = []
    heading = ""
    body: list[str] = []

    def flush() -> None:
        content = "\n".join(body).strip()
        if content or heading:
            sections.append((heading, content))

    for line in source.split("\n"):
        m = _HEADING.match(line)
        if m:
            flush()
            body = []
            heading = m.group(2).strip()
            continue
        body.append(line)
    flush()

    raw: list[tuple[str, str]] = []
    for sec_heading, sec_text in sections:
        if not sec_text:
            if sec_heading:
                raw.append((sec_heading, sec_heading))
            continue
        rest = sec_text
        carry = ""
        while rest:
            room = target - len(carry)
            piece = rest[:room]
            if len(rest) > room:
                at = max(piece.rfind("\n\n"), piece.rfind("\n"), piece.rfind(" "))
                if at > room * 0.5:
                    piece = piece[:at]
            merged = (carry + piece).strip()
            if merged:
                raw.append((sec_heading, merged))
            rest = rest[len(piece):]
            carry = merged[-overlap_chars:] if overlap_chars else ""
            if len(raw) >= max_chunks or not rest.strip():
                break
        if len(raw) >= max_chunks:
            break
    return [Chunk(path=path, index=i, heading=h, text=t, tokens=tokenize(f"{h} {t}")) for i, (h, t) in enumerate(raw)]


# English function words only; short identifiers that carry meaning in metadata (id, no, hh) are kept.
STOPWORDS = frozenset(
    "a about an and are as at be been but by can could did do does for from had has have he her his how if in "
    "into is it its may me my not of on or our she should so than that the their them then there these they "
    "this those to was we were what when where which who why will with would you your".split()
)
_CAMEL = re.compile(r"([a-z0-9])([A-Z])")
_SPLIT = re.compile(r"[^a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Identifier-aware: ``house_hold_id``, ``HHID`` and ``getUserName`` split into parts."""
    lowered = _CAMEL.sub(r"\1 \2", text or "").lower()
    return [t for t in _SPLIT.split(lowered) if t and (len(t) > 1 or t.isdigit()) and t not in STOPWORDS]


# ---------------------------------------------------------------- index

@dataclass
class Index:
    chunks: list[Chunk]
    key: str
    vectors: Optional[np.ndarray] = None  # (n, dims) float32, L2-normalised
    embedder_id: Optional[str] = None
    # Inverted index: term -> {chunk position: count}
    postings: dict[str, dict[int, int]] = field(default_factory=dict, repr=False)
    lengths: list[int] = field(default_factory=list, repr=False)
    avg_length: float = 1.0

    def __post_init__(self) -> None:
        if not self.postings:
            for pos, chunk in enumerate(self.chunks):
                for tok in chunk.tokens:
                    bucket = self.postings.setdefault(tok, {})
                    bucket[pos] = bucket.get(pos, 0) + 1
            self.lengths = [len(c.tokens) for c in self.chunks]
            self.avg_length = (sum(self.lengths) / len(self.lengths)) if self.lengths else 1.0


def corpus_hash(docs: list[tuple[str, str]]) -> str:
    """Identity of the content: sorted by path so listing order does not matter."""
    h = hashlib.sha256()
    for path, text in sorted(docs):
        h.update(path.encode("utf-8"))
        h.update(b"\0")
        h.update(text.encode("utf-8"))
        h.update(b"\0")
    return h.hexdigest()


def index_key(content_hash: str, options: RetrievalOptions, embedder_id: Optional[str], dims: Optional[int]) -> str:
    return "/".join(
        [INDEX_FORMAT, f"chunker{CHUNKER_VERSION}", str(options.target_chars), str(options.overlap_chars), embedder_id or "none", str(dims or 0), content_hash]
    )


def build_chunks(docs: list[tuple[str, str]], options: RetrievalOptions) -> list[Chunk]:
    out: list[Chunk] = []
    for path, text in docs:
        out.extend(
            chunk_text(text, path=path, target_chars=options.target_chars, overlap_chars=options.overlap_chars, max_chunks=options.max_chunks_per_doc)
        )
    return out


def save_vectors(path: Path, key: str, embedder_id: str, vectors: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.npz")
    np.savez(tmp, vectors=vectors.astype(np.float32), meta=np.array(json.dumps({"key": key, "embedder": embedder_id})))
    tmp.replace(path)


def load_vectors(path: Path, key: str, n: int) -> Optional[np.ndarray]:
    if not path.is_file():
        return None
    try:
        with np.load(path, allow_pickle=False) as data:
            meta = json.loads(str(data["meta"]))
            vectors = data["vectors"]
    except Exception:  # noqa: BLE001 - a damaged index is rebuilt, not fatal
        logger.warning("assistant: unreadable index %s, rebuilding", path, exc_info=True)
        return None
    if meta.get("key") != key or vectors.shape[0] != n:
        return None
    return vectors


# ---------------------------------------------------------------- ranking

def bm25(index: Index, question: str, options: RetrievalOptions) -> dict[int, float]:
    """
    BM25 over chunk positions. A term present in more than a third of the chunks (floor 2), or in
    every chunk, carries no signal and neither scores nor produces a hit.
    """
    n = len(index.chunks)
    scores: dict[int, float] = {}
    if not n:
        return scores
    ubiquitous = max(2, n // 3)
    for term in dict.fromkeys(tokenize(question)):
        posting = index.postings.get(term)
        if not posting:
            continue
        df = len(posting)
        if df > ubiquitous or (n > 1 and df >= n):
            continue
        idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
        for pos, count in posting.items():
            length = index.lengths[pos]
            tf = (count * (options.k1 + 1)) / (count + options.k1 * (1 - options.b + options.b * length / index.avg_length))
            scores[pos] = scores.get(pos, 0.0) + idf * tf
    return scores


def _scale(scores: dict[int, float]) -> dict[int, float]:
    if not scores:
        return {}
    lo, hi = min(scores.values()), max(scores.values())
    span = hi - lo
    return {k: ((v - lo) / span if span > 0 else 1.0) for k, v in scores.items()}


@dataclass
class Ranked:
    chunk: Chunk
    lexical: float
    semantic: Optional[float]
    score: float


def rank_chunks(
    index: Index, question: str, options: RetrievalOptions, query_vector: Optional[np.ndarray] = None, *, floor: float = 0.0
) -> list[Ranked]:
    if not index.chunks:
        return []
    lexical = bm25(index, question, options)
    semantic: dict[int, float] = {}
    if query_vector is not None and index.vectors is not None and query_vector.shape[0] == index.vectors.shape[1]:
        # numpy 2.x on macOS Accelerate warns spuriously (divide by zero / overflow) on finite unit
        # vectors; the results are finite. Silence it here, and drop anything non-finite regardless.
        with np.errstate(all="ignore"):
            sims = np.nan_to_num(index.vectors @ query_vector, nan=-1.0, posinf=-1.0, neginf=-1.0)
        # A vector match is a candidate for every chunk; without a floor, an uncovered question still
        # returns its least-bad section. The floor spares chunks a query term matched.
        for pos, sim in enumerate(sims.tolist()):
            if sim >= floor or lexical.get(pos, 0) > 0:
                semantic[pos] = sim
    weight = min(1.0, max(0.0, options.alpha)) if semantic else 0.0
    lex_s, sem_s = _scale(lexical), _scale(semantic)
    out = []
    for pos in set(lexical) | set(semantic):
        out.append(
            Ranked(
                chunk=index.chunks[pos],
                lexical=lexical.get(pos, 0.0),
                semantic=semantic.get(pos),
                score=lex_s.get(pos, 0.0) * (1 - weight) + sem_s.get(pos, 0.0) * weight,
            )
        )
    out.sort(key=lambda r: (-r.score, r.chunk.path, r.chunk.index))
    return out[: max(1, options.top_k)]


@dataclass
class Selection:
    text: str
    labels: list[str]
    omitted: list[str]
    truncated: bool


def select_under_cap(ranked: list[Ranked], max_bytes: int) -> Selection:
    """Prompt text under a byte cap; what did not fit is reported, never silently dropped."""
    size = lambda s: len(s.encode("utf-8"))  # noqa: E731
    marker = "\n[truncated to fit the size limit]"
    included: list[tuple[str, str, bool]] = []
    omitted: list[str] = []
    used = 0
    for entry in ranked:
        heading = (entry.chunk.heading or "").strip()
        source = (entry.chunk.path or "").strip()
        label = (f"{heading} — {source}" if heading and source and heading != source else (heading or source or f"chunk {entry.chunk.index}"))[:120]
        prefix = f"### {label}\n"
        cost = size(prefix) + size(entry.chunk.text) + (2 if included else 0)
        if used + cost <= max_bytes:
            included.append((label, entry.chunk.text, False))
            used += cost
            continue
        if included:
            omitted.append(label)
            continue
        room = max_bytes - size(prefix) - size(marker)
        if room <= 0:
            omitted.append(label)
            continue
        cut = entry.chunk.text.encode("utf-8")[:room].decode("utf-8", errors="ignore")
        included.append((label, cut, True))
        used = size(prefix) + size(cut) + size(marker)
    text = "\n\n".join(f"### {label}\n{body}{marker if cut else ''}" for label, body, cut in included)
    if size(text) > max_bytes:
        text = text.encode("utf-8")[:max_bytes].decode("utf-8", errors="ignore")
    return Selection(text=text, labels=[l for l, _, _ in included], omitted=omitted, truncated=any(c for _, _, c in included) or bool(omitted))


# ---------------------------------------------------------------- service

@dataclass(frozen=True)
class Document:
    path: Path
    label: str


class RetrievalService:
    """
    Owns the index for a fixed list of documents. ``start`` makes BM25 available at once and builds or
    loads embeddings in a worker thread; ``search`` uses whatever is ready.
    """

    def __init__(self, documents: list[Document], options: RetrievalOptions, embedder: Any, index_path: Path) -> None:
        self.documents = documents
        self.options = options
        self.embedder = embedder
        self.index_path = index_path
        self.index: Optional[Index] = None
        self.state = "idle"  # idle | keyword | embedding | ready | failed
        self.detail: Optional[str] = None
        self._task: Optional[asyncio.Task] = None

    def status(self) -> dict[str, Any]:
        return {
            "state": self.state,
            "detail": self.detail,
            "documents": [d.label for d in self.documents],
            "chunks": len(self.index.chunks) if self.index else 0,
            "embedder": self.embedder.id if self.embedder else None,
            "embedded": bool(self.index is not None and self.index.vectors is not None),
        }

    def _read_docs(self) -> list[tuple[str, str]]:
        docs = []
        for doc in self.documents:
            try:
                docs.append((doc.label, doc.path.read_text(encoding="utf-8", errors="replace")))
            except OSError as exc:
                logger.warning("assistant: cannot read %s (%s); skipped", doc.path, exc)
        return docs

    def prepare_keyword(self) -> None:
        docs = self._read_docs()
        if not docs:
            self.state, self.detail = "failed", "no readable documents"
            return
        dims = self.embedder.dims if self.embedder else None
        key = index_key(corpus_hash(docs), self.options, self.embedder.id if self.embedder else None, dims)
        self.index = Index(chunks=build_chunks(docs, self.options), key=key)
        self.state = "keyword"
        self.detail = "keyword search only" + ("; embeddings loading" if self.embedder else "")

    def prepare_embeddings(self) -> None:
        index = self.index
        if index is None or self.embedder is None:
            return
        vectors = load_vectors(self.index_path, index.key, len(index.chunks))
        if vectors is None:
            self.state = "embedding"
            started = time.monotonic()
            texts = [f"{c.heading}\n{c.text}" if c.heading and c.heading != c.text else c.text for c in index.chunks]
            vectors = self.embedder.embed(texts, "document")
            save_vectors(self.index_path, index.key, self.embedder.id, vectors)
            logger.info("Assistant: embedded %d chunks in %.0fs", len(texts), time.monotonic() - started)
        else:
            self.embedder.load()  # queries need it either way
        index.vectors = vectors
        index.embedder_id = self.embedder.id
        self.state, self.detail = "ready", None

    async def start(self) -> None:
        """Non-blocking: returns once keyword search works; embeddings continue in a thread."""
        try:
            await asyncio.to_thread(self.prepare_keyword)
        except Exception as exc:  # noqa: BLE001
            self.state, self.detail = "failed", str(exc)
            logger.warning("assistant: retrieval unavailable", exc_info=True)
            return
        if self.embedder is not None and self.index is not None:

            async def embed() -> None:
                try:
                    await asyncio.to_thread(self.prepare_embeddings)
                except Exception as exc:  # noqa: BLE001 - keyword search keeps working
                    self.state, self.detail = "keyword", f"keyword search only: embeddings failed ({exc})"
                    logger.warning("assistant: embeddings unavailable", exc_info=True)

            self._task = asyncio.create_task(embed())

    async def search(self, question: str) -> tuple[str, dict[str, Any]]:
        """Prompt text (possibly empty) and what was used, for the ``meta`` event and the log."""
        index = self.index
        if index is None or not index.chunks:
            return "", {"state": self.state, "hits": []}
        query_vector = None
        if index.vectors is not None and self.embedder is not None:
            try:
                query_vector = (await asyncio.to_thread(self.embedder.embed, [question], "query"))[0]
            except Exception:  # noqa: BLE001 - answer from keywords
                logger.warning("assistant: query embedding failed", exc_info=True)
        floor = self.options.min_similarity
        if floor is None:
            floor = getattr(getattr(self.embedder, "spec", None), "min_similarity", 0.0)
        ranked = rank_chunks(index, question, self.options, query_vector, floor=floor)
        if not ranked:
            return "", {"state": self.state, "hits": []}
        sel = select_under_cap(ranked, self.options.max_bytes)
        notes = []
        if sel.omitted:
            notes.append(f"{len(sel.omitted)} further matching section(s) did not fit")
        if query_vector is None:
            notes.append("keyword search only")
        text = sel.text + (f"\n\n[{'; '.join(notes)}.]" if notes else "")
        return text, {"state": self.state, "hits": sel.labels, "semantic": query_vector is not None}
