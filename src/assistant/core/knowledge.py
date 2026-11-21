"""
Project knowledge index: the files a curator works from (External resources, chat attachments), read
once, chunked with page numbers, indexed (BM25 and embeddings) and cached by content hash.

Nothing is loaded into a prompt whole. A turn or a fill asks the view for the passages that answer a
question (or one query per field) and gets them labelled ``name p.N``.

* Identity is the SHA-256 of the bytes, so the same file reaching the assistant as a resource and as an
  attachment is indexed once.
* A source is searchable only from the project (``sid``) it was registered under. Resources are
  registered by the caller on every sync (it lists the project's files); attachments are registered at
  upload and remembered across restarts.
* BM25 is ready as soon as the text is extracted; embeddings are built in one background thread and
  picked up by the next search.
"""

from __future__ import annotations

import asyncio
import dataclasses
import hashlib
import json
import logging
import os
import re
import threading
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np

from src.assistant.core.documents import ExtractError, SOURCE_MAX_CHARS, extract_pages
from src.assistant.core.retrieval import (
    CHUNKER_VERSION,
    Chunk,
    Index,
    RetrievalOptions,
    chunk_text,
    load_vectors,
    rank_chunks,
    save_vectors,
    tokenize,
)

logger = logging.getLogger(__name__)

FORMAT = "pslm-source/1"
SOURCE_CHUNK_CHARS = 1000
SOURCE_OVERLAP_CHARS = 100
MAX_CHUNKS_PER_SOURCE = 6000
_HASH = re.compile(r"^[0-9a-f]{64}$")


def is_hash(value: object) -> bool:
    return isinstance(value, str) and bool(_HASH.match(value))


def attachment_id(digest: str) -> str:
    return f"att-{digest[:12]}"


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def sha256_file(path: str | Path) -> str:
    return _sha256_file(Path(path))


def _chunk_pages(pages: list[tuple[int, str]], name: str) -> list[Chunk]:
    out: list[Chunk] = []
    for number, text in pages:
        for c in chunk_text(text, path=name, target_chars=SOURCE_CHUNK_CHARS, overlap_chars=SOURCE_OVERLAP_CHARS, max_chunks=MAX_CHUNKS_PER_SOURCE):
            out.append(Chunk(path=name, index=len(out), heading=c.heading, text=c.text, tokens=c.tokens, page=number))
        if len(out) >= MAX_CHUNKS_PER_SOURCE:
            break
    return out


@dataclass
class SourceEntry:
    """One file's extracted text and chunks, keyed by content hash. Shared by every project that has it."""

    hash: str
    pages: list[tuple[int, str]]
    page_count: Optional[int]
    chars: int
    truncated: bool
    chunks: list[Chunk] = field(default_factory=list)
    vectors: Optional[np.ndarray] = None
    note: Optional[str] = None
    state: str = "keyword"  # keyword (embeddings pending) | ready
    embedding: bool = False

    @property
    def text(self) -> str:
        return "\n\n".join(t for _, t in self.pages)


class KnowledgeStore:
    def __init__(
        self,
        cache_dir: Path,
        embedder: Any,
        options: Optional[RetrievalOptions] = None,
        *,
        max_entries: int = 48,
        max_disk_bytes: int = 2 * 1024**3,
    ) -> None:
        self.dir = Path(cache_dir) / "knowledge"
        self.embedder = embedder
        self.options = options or RetrievalOptions()
        self.max_entries = max_entries
        self.max_disk_bytes = max_disk_bytes
        self._entries: "OrderedDict[str, SourceEntry]" = OrderedDict()
        self._hash_memo: dict[tuple[str, int, int], str] = {}
        self._registry: dict[str, dict[str, dict[str, str]]] = {}  # sid -> hash -> {id, name}
        self._locks: dict[str, asyncio.Lock] = {}
        self._indexes: "OrderedDict[tuple, Index]" = OrderedDict()
        self._qcache: "OrderedDict[str, np.ndarray]" = OrderedDict()
        # Sources are embedded this many at a time (default one: the model already uses the machine's cores).
        self._pool = ThreadPoolExecutor(max_workers=max(1, int(os.getenv("ASSISTANT_EMBED_WORKERS", "1"))), thread_name_prefix="assistant-embed")
        self._tasks: set[asyncio.Task] = set()
        self._mutex = threading.RLock()
        self._load_registry()

    # ------------------------------------------------------------ registry (attachments only)

    @property
    def _registry_path(self) -> Path:
        return self.dir / "registry.json"

    def _load_registry(self) -> None:
        try:
            self._registry = json.loads(self._registry_path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            self._registry = {}
        except Exception:  # noqa: BLE001 - a damaged registry only forgets attachments
            logger.warning("assistant: unreadable source registry, starting empty", exc_info=True)
            self._registry = {}

    def _save_registry(self) -> None:
        try:
            self.dir.mkdir(parents=True, exist_ok=True)
            tmp = self._registry_path.with_suffix(".tmp")
            tmp.write_text(json.dumps(self._registry), encoding="utf-8")
            tmp.replace(self._registry_path)
        except OSError:
            logger.warning("assistant: could not save the source registry", exc_info=True)

    def register_attachment(self, sid: str, digest: str, name: str) -> str:
        with self._mutex:
            self._registry.setdefault(str(sid), {})[digest] = {"id": attachment_id(digest), "name": name[:200]}
            self._save_registry()
        return attachment_id(digest)

    def attachments(self, sid: str) -> list[tuple[str, dict[str, str]]]:
        with self._mutex:
            return list(self._registry.get(str(sid), {}).items())

    def attachment_name(self, sid: str, digest: str) -> Optional[str]:
        with self._mutex:
            return (self._registry.get(str(sid), {}).get(digest) or {}).get("name")

    @staticmethod
    def attachment_id(digest: str) -> str:
        return attachment_id(digest)

    def status(self) -> dict[str, Any]:
        with self._mutex:
            return {
                "cached_in_memory": len(self._entries),
                "embedder": self.embedder.id if self.embedder else None,
                "attachments": sum(len(v) for v in self._registry.values()),
            }

    def is_registered(self, sid: str, digest: str) -> bool:
        with self._mutex:
            return digest in self._registry.get(str(sid), {})

    def remove(self, sid: str, hashes: list[str]) -> int:
        removed = 0
        with self._mutex:
            mine = self._registry.get(str(sid), {})
            for digest in hashes:
                if mine.pop(digest, None) is not None:
                    removed += 1
            if removed:
                if not mine:
                    self._registry.pop(str(sid), None)
                self._save_registry()
        return removed

    # ------------------------------------------------------------ entries

    def _paths(self, digest: str) -> tuple[Path, Path]:
        return self.dir / f"{digest}.json", self.dir / f"{digest}.npz"

    def _vector_key(self, digest: str) -> str:
        e = self.embedder
        return f"{FORMAT}/chunker{CHUNKER_VERSION}/{SOURCE_CHUNK_CHARS}/{SOURCE_OVERLAP_CHARS}/{e.id if e else 'none'}/{digest}"

    def _remember(self, entry: SourceEntry) -> None:
        with self._mutex:
            self._entries[entry.hash] = entry
            self._entries.move_to_end(entry.hash)
            while len(self._entries) > self.max_entries:
                self._entries.popitem(last=False)

    def get(self, digest: str) -> Optional[SourceEntry]:
        """The entry in memory or on disk, or None when this content was never indexed here."""
        with self._mutex:
            entry = self._entries.get(digest)
            if entry is not None:
                self._entries.move_to_end(digest)
                return entry
        json_path, npz_path = self._paths(digest)
        try:
            data = json.loads(json_path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return None
        except Exception:  # noqa: BLE001 - rebuilt from the file on the next sync
            logger.warning("assistant: unreadable source cache %s", json_path.name, exc_info=True)
            return None
        if data.get("format") != FORMAT:
            return None
        pages = [(int(n), str(t)) for n, t in data.get("pages", [])]
        entry = SourceEntry(
            hash=digest,
            pages=pages,
            page_count=data.get("page_count"),
            chars=int(data.get("chars", 0)),
            truncated=bool(data.get("truncated")),
            chunks=_chunk_pages(pages, ""),
        )
        if self.embedder is not None:
            vectors = load_vectors(npz_path, self._vector_key(digest), len(entry.chunks))
            if vectors is not None:
                entry.vectors = vectors
                entry.state = "ready"
        else:
            entry.state = "ready"
        self._remember(entry)
        if entry.state != "ready":
            self._schedule_embedding(entry)
        return entry

    async def ensure_file(self, path: str | Path, name: str) -> SourceEntry:
        """Hash the file, then load or build its entry. Raises ExtractError for a file that cannot be read."""
        p = Path(path)
        st = await asyncio.to_thread(p.stat)
        memo_key = (str(p), st.st_size, st.st_mtime_ns)
        digest = self._hash_memo.get(memo_key)
        if digest is None:
            digest = await asyncio.to_thread(_sha256_file, p)
            if len(self._hash_memo) > 4096:
                self._hash_memo.clear()
            self._hash_memo[memo_key] = digest
        entry = await asyncio.to_thread(self.get, digest)
        if entry is not None:
            return entry
        lock = self._locks.setdefault(digest, asyncio.Lock())
        async with lock:
            entry = await asyncio.to_thread(self.get, digest)
            if entry is not None:
                return entry
            started = time.monotonic()
            result = await asyncio.to_thread(extract_pages, p, SOURCE_MAX_CHARS)
            entry = SourceEntry(
                hash=digest,
                pages=result["pages"],
                page_count=result["page_count"],
                chars=int(result["chars"]),
                truncated=bool(result["truncated"]),
            )
            entry.chunks = await asyncio.to_thread(_chunk_pages, entry.pages, "")
            entry.state = "keyword" if self.embedder is not None else "ready"
            await asyncio.to_thread(self._write, entry)
            self._remember(entry)
            logger.info(
                "assistant: indexed source %s (%s) pages=%s chars=%d chunks=%d in %.1fs",
                digest[:12], name[:60], entry.page_count, entry.chars, len(entry.chunks), time.monotonic() - started,
            )
            if self.embedder is not None:
                self._schedule_embedding(entry)
            return entry

    def _write(self, entry: SourceEntry) -> None:
        try:
            self.dir.mkdir(parents=True, exist_ok=True)
            json_path, _ = self._paths(entry.hash)
            tmp = json_path.with_suffix(".tmp")
            tmp.write_text(
                json.dumps(
                    {
                        "format": FORMAT,
                        "page_count": entry.page_count,
                        "chars": entry.chars,
                        "truncated": entry.truncated,
                        "pages": [[n, t] for n, t in entry.pages],
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            tmp.replace(json_path)
            self._prune()
        except OSError:
            logger.warning("assistant: could not cache source %s", entry.hash[:12], exc_info=True)

    def _prune(self) -> None:
        try:
            files = [f for f in self.dir.iterdir() if f.suffix in {".json", ".npz"} and f.name != "registry.json"]
            total = sum(f.stat().st_size for f in files)
            if total <= self.max_disk_bytes:
                return
            keep = {h for sid in self._registry.values() for h in sid}
            for f in sorted(files, key=lambda f: f.stat().st_mtime):
                if total <= self.max_disk_bytes:
                    break
                if f.stem in keep or f.stem in self._entries:
                    continue
                total -= f.stat().st_size
                f.unlink(missing_ok=True)
        except OSError:
            logger.debug("assistant: source cache prune failed", exc_info=True)

    # ------------------------------------------------------------ embeddings

    def _schedule_embedding(self, entry: SourceEntry) -> None:
        if self.embedder is None or entry.embedding or entry.state == "ready":
            return
        entry.embedding = True
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        if loop is None:
            self._pool.submit(self._embed_entry, entry)
            return
        task = loop.run_in_executor(self._pool, self._embed_entry, entry)
        self._tasks.add(task)  # type: ignore[arg-type]
        task.add_done_callback(self._tasks.discard)

    def _embed_entry(self, entry: SourceEntry) -> None:
        try:
            texts = [f"{c.heading}\n{c.text}" if c.heading and c.heading != c.text else c.text for c in entry.chunks]
            started = time.monotonic()
            vectors = self.embedder.embed(texts, "document")
            _, npz_path = self._paths(entry.hash)
            save_vectors(npz_path, self._vector_key(entry.hash), self.embedder.id, vectors)
            entry.vectors = vectors
            logger.info("assistant: embedded source %s: %d chunks in %.0fs", entry.hash[:12], len(texts), time.monotonic() - started)
        except Exception as exc:  # noqa: BLE001 - keyword search keeps working
            entry.note = f"keyword search only: embeddings failed ({type(exc).__name__})"
            logger.warning("assistant: source embeddings failed for %s", entry.hash[:12], exc_info=True)
        finally:
            entry.state = "ready"
            entry.embedding = False

    # ------------------------------------------------------------ reporting and views

    @staticmethod
    def describe(entry: SourceEntry, *, id: str, kind: str, name: str) -> dict[str, Any]:
        return {
            "id": id,
            "hash": entry.hash,
            "kind": kind,
            "name": name,
            "pages": entry.page_count,
            "chars": entry.chars,
            "chunks": len(entry.chunks),
            "state": entry.state,
            "embedded": entry.vectors is not None,
            "truncated": entry.truncated,
            **({"note": entry.note} if entry.note else {}),
        }

    def _index_for(self, parts: list[tuple[str, SourceEntry]]) -> tuple[list[Chunk], Index]:
        key = tuple((name, e.hash, e.vectors is not None) for name, e in parts)
        with self._mutex:
            hit = self._indexes.get(key)
            if hit is not None:
                self._indexes.move_to_end(key)
                return hit.chunks, hit
        chunks: list[Chunk] = []
        for name, entry in parts:
            for c in entry.chunks:
                chunks.append(Chunk(path=name, index=len(chunks), heading=c.heading, text=c.text, tokens=c.tokens, page=c.page))
        vectors = None
        embedder_id = None
        if self.embedder is not None and parts and all(e.vectors is not None and len(e.vectors) == len(e.chunks) for _, e in parts):
            vectors = np.vstack([e.vectors for _, e in parts]) if len(parts) > 1 else parts[0][1].vectors
            embedder_id = self.embedder.id
        index = Index(chunks=chunks, key="|".join(str(k) for k in key), vectors=vectors, embedder_id=embedder_id)
        with self._mutex:
            self._indexes[key] = index
            while len(self._indexes) > 16:
                self._indexes.popitem(last=False)
        return chunks, index

    def view(self, parts: list[tuple[str, SourceEntry]]) -> Optional["KnowledgeView"]:
        # The same bytes under two names (a resource and an attachment, or two resources) are searched once:
        # the first one given wins, so callers list resources first.
        seen: set[str] = set()
        parts = [p for p in parts if not (p[1].hash in seen or seen.add(p[1].hash))]
        if not parts:
            return None
        chunks, index = self._index_for(parts)
        return KnowledgeView(self, parts, chunks, index)

    def query_vector(self, query: str) -> Optional[np.ndarray]:
        if self.embedder is None:
            return None
        with self._mutex:
            hit = self._qcache.get(query)
            if hit is not None:
                self._qcache.move_to_end(query)
                return hit
        try:
            vec = self.embedder.embed([query], "query")[0]
        except Exception:  # noqa: BLE001 - answer from keywords
            logger.warning("assistant: query embedding failed", exc_info=True)
            return None
        with self._mutex:
            self._qcache[query] = vec
            while len(self._qcache) > 256:
                self._qcache.popitem(last=False)
        return vec

    def floor(self) -> float:
        if self.options.min_similarity is not None:
            return self.options.min_similarity
        return getattr(getattr(self.embedder, "spec", None), "min_similarity", 0.0)


def source_label(chunk: Chunk) -> str:
    return f"{chunk.path} p.{chunk.page}" if chunk.page else chunk.path


class KnowledgeView(str):
    """
    The project sources a request works from, as one searchable document. As a ``str`` it is the full
    text (what grounding checks and the codebook parser read); ``excerpt``, ``search`` and ``passage``
    are what a prompt uses, so the model gets passages and never the whole text.
    """

    def __new__(cls, store: KnowledgeStore, parts: list[tuple[str, SourceEntry]], chunks: list[Chunk], index: Index):
        return super().__new__(cls, "\n\n".join(e.text for _, e in parts))

    def __init__(self, store: KnowledgeStore, parts: list[tuple[str, SourceEntry]], chunks: list[Chunk], index: Index) -> None:
        self._store = store
        self.parts = parts
        self.chunks = chunks
        self._index = index
        self._bytes: Optional[int] = None

    # -- what the request actually used

    @property
    def names(self) -> list[str]:
        return [name for name, _ in self.parts]

    @property
    def semantic(self) -> bool:
        return self._index.vectors is not None

    def label(self, chunk: Chunk) -> str:
        return source_label(chunk)

    # -- search

    def passage(self, pos: int) -> str:
        chunk = self.chunks[pos]
        first = pos == 0 or self.chunks[pos - 1].heading != chunk.heading or self.chunks[pos - 1].path != chunk.path
        if first and chunk.heading and not chunk.text.startswith(chunk.heading):
            return f"{chunk.heading}\n{chunk.text}"
        return chunk.text

    def search(self, query: str, k: int = 4) -> list[Chunk]:
        """Best passages for ``query``, best first; empty when nothing matches by word or meaning."""
        if not self.chunks or not (query or "").strip():
            return []
        options = dataclasses.replace(self._store.options, top_k=max(1, k))
        vector = self._store.query_vector(query) if self.semantic else None
        return [r.chunk for r in rank_chunks(self._index, query, options, vector, floor=self._store.floor())]

    def locate(self, quoted: str) -> Optional[str]:
        """``name p.N`` of the passage that contains these words, so a proposal can say where it comes from."""
        norm = lambda t: re.sub(r"\s+", " ", t).strip().lower()  # noqa: E731
        needle = norm(quoted)[:90]
        if len(needle) < 12:
            return None
        for chunk in self.chunks:
            if needle in norm(chunk.text):
                return source_label(chunk)
        return None

    def excerpt(self, query: "str | list[str]", max_bytes: int) -> tuple[str, bool]:
        """
        The whole text when it fits ``max_bytes``. Otherwise the opening of the first source plus, for each
        query in turn (one per field or variable), its best passages, labelled with source and page, in
        reading order. Returns (text, cut).
        """
        if self._bytes is None:
            self._bytes = len(str.encode(self, "utf-8"))
        if self._bytes <= max_bytes:
            return str(self), False
        if not self.chunks:
            return "", True
        queries = [q for q in (query if isinstance(query, list) else [query]) if q and q.strip()]
        size = lambda c: len(self.passage(c.index).encode("utf-8")) + len(source_label(c)) + 12  # noqa: E731
        picked: list[int] = []
        seen: set[int] = set()
        used = 0

        def take(pos: int) -> bool:
            nonlocal used
            if pos in seen:
                return False
            cost = size(self.chunks[pos])
            if used + cost > max_bytes:
                return False
            seen.add(pos)
            picked.append(pos)
            used += cost
            return True

        take(0)
        rankings = [[c.index for c in self.search(q, k=10)] for q in queries]
        for depth in range(10):
            for ranked in rankings:
                if depth < len(ranked):
                    take(ranked[depth])
        if not rankings or not any(rankings):
            # Nothing matched: reading order from the start, so a vague question still sees the opening pages.
            for pos in range(len(self.chunks)):
                if not take(pos) and used >= max_bytes:
                    break
        parts: list[str] = []
        previous = -2
        for pos in sorted(picked):
            chunk = self.chunks[pos]
            if pos != previous + 1:
                parts.append("[…]")
            parts.append(f"### {source_label(chunk)}\n{self.passage(pos)}")
            previous = pos
        return "\n\n".join(parts), True
