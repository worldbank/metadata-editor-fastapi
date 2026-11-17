"""
Search over the open data file's variables: BM25 over name, label, description, question and concepts,
plus embeddings once they are built, so "firm size" finds ``l1`` labelled "Number of permanent employees".

The variables come in the bundle (the editor builds it per request), so the index is keyed by their
content and kept in memory; embeddings are built in the knowledge store's single background thread and
picked up by the next search.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
import threading
from collections import OrderedDict
from typing import Any, Optional

import numpy as np

from src.assistant.core.retrieval import Chunk, Index, RetrievalOptions, rank_chunks, tokenize

logger = logging.getLogger(__name__)

MAX_INDEXED = 3000
_store: Any = None
_cache: "OrderedDict[str, VariableIndex]" = OrderedDict()
_lock = threading.Lock()


def configure(store: Any) -> None:
    """Called once at startup with the KnowledgeStore (embedder, options, background thread)."""
    global _store
    _store = store


def _text(v: dict[str, Any]) -> str:
    concepts = v.get("concepts") or []
    bits = [str(v.get("name") or ""), str(v.get("label") or ""), str(v.get("description") or ""), str(v.get("question") or ""),
            str(v.get("universe") or ""), " ".join(str(c) for c in concepts if c)]
    return ". ".join(b for b in bits if b)


class VariableIndex:
    def __init__(self, variables: list[dict[str, Any]]) -> None:
        self.variables = variables
        chunks = []
        for i, v in enumerate(variables):
            text = _text(v)
            chunks.append(Chunk(path=str(v.get("name") or ""), index=i, heading="", text=text, tokens=tokenize(text)))
        self.index = Index(chunks=chunks, key="variables")
        self.embedding = False

    def _embed(self, embedder: Any) -> None:
        try:
            self.index.vectors = embedder.embed([c.text for c in self.index.chunks], "document")
            self.index.embedder_id = embedder.id
        except Exception:  # noqa: BLE001 - keyword search keeps working
            logger.warning("assistant: variable embeddings failed", exc_info=True)
        finally:
            self.embedding = False

    def start_embedding(self) -> None:
        store = _store
        if store is None or store.embedder is None or self.embedding or self.index.vectors is not None:
            return
        self.embedding = True
        store._pool.submit(self._embed, store.embedder)

    def search(self, query: str, k: int) -> list[dict[str, Any]]:
        """Variables ranked for ``query`` (best first); empty when nothing matches by word or meaning."""
        store = _store
        options = dataclasses.replace(store.options if store else RetrievalOptions(), top_k=max(1, k))
        vector: Optional[np.ndarray] = None
        floor = 0.0
        if store is not None and self.index.vectors is not None:
            vector = store.query_vector(query)
            floor = store.floor()
        return [self.variables[r.chunk.index] for r in rank_chunks(self.index, query, options, vector, floor=floor)]


def for_variables(variables: list[dict[str, Any]]) -> VariableIndex:
    rows = [v for v in variables if isinstance(v, dict)][:MAX_INDEXED]
    key = hashlib.sha256(json.dumps(rows, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()
    with _lock:
        hit = _cache.get(key)
        if hit is not None:
            _cache.move_to_end(key)
            return hit
    built = VariableIndex(rows)
    built.start_embedding()
    with _lock:
        _cache[key] = built
        while len(_cache) > 8:
            _cache.popitem(last=False)
    return built
