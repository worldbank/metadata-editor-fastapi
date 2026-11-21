"""Project knowledge index: content-hash identity, per-project registry, passages instead of whole text."""

from __future__ import annotations

import asyncio
import hashlib
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from src.assistant.core.documents import attached_document
from src.assistant.core.knowledge import KnowledgeStore, KnowledgeView, attachment_id, is_hash
from src.assistant.core.retrieval import RetrievalOptions
from src.assistant.host import varindex


class FakeEmbedder:
    """Hashes words into a small space: texts sharing words are close. Deterministic, no model download."""

    id = "fake-embed"
    dims = 64
    spec = SimpleNamespace(min_similarity=0.2)

    def embed(self, texts, kind):
        out = np.zeros((len(texts), self.dims), dtype=np.float32)
        for i, text in enumerate(texts):
            for word in text.lower().split():
                out[i, int(hashlib.md5(word.encode()).hexdigest(), 16) % self.dims] += 1.0
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.maximum(norms, 1e-12)


def _long_doc(topic_page: int = 30) -> str:
    """A long Markdown document with one distinctive section deep inside, far beyond any 40k cap."""
    parts = ["# WBES India follow-up\n\nThe survey title page and abstract."]
    for n in range(1, 100):
        body = f"Filler paragraph {n} about enumeration areas and field logistics. " * 18
        if n == topic_page:
            body = "The sampling procedure is stratified random sampling of establishments re-contacted from the baseline. " * 3
        parts.append(f"## Section {n}\n\n{body}")
    return "\n\n".join(parts)


def _store(tmp_path: Path, embedder=None) -> KnowledgeStore:
    return KnowledgeStore(tmp_path / "cache", embedder, RetrievalOptions(), max_entries=8)


def _write(tmp_path: Path, name: str, text: str) -> Path:
    p = tmp_path / name
    p.write_text(text, encoding="utf-8")
    return p


def run(coro):
    return asyncio.run(coro)


def test_full_text_is_indexed_not_cut_at_40k(tmp_path):
    text = _long_doc()
    assert len(text) > 100_000
    store = _store(tmp_path)
    entry = run(store.ensure_file(_write(tmp_path, "report.md", text), "report.md"))
    assert entry.chars == len(text.strip()) and not entry.truncated
    view = store.view([("report.md", entry)])
    hits = view.search("sampling procedure stratified", 3)
    assert hits and "stratified random sampling" in hits[0].text


def test_same_bytes_are_one_entry_whatever_the_path(tmp_path):
    store = _store(tmp_path)
    a = run(store.ensure_file(_write(tmp_path, "a.md", "# T\n\nsame content here"), "a.md"))
    b = run(store.ensure_file(_write(tmp_path, "b.md", "# T\n\nsame content here"), "copy of a.md"))
    assert a is b and a.hash == b.hash and is_hash(a.hash)


def test_entry_survives_restart_from_disk(tmp_path):
    first = _store(tmp_path)
    entry = run(first.ensure_file(_write(tmp_path, "a.md", _long_doc(5)), "a.md"))
    second = _store(tmp_path)
    again = second.get(entry.hash)
    assert again is not None and len(again.chunks) == len(entry.chunks) and again.chars == entry.chars


def test_unknown_or_unreadable_files_are_refused_clearly(tmp_path):
    from src.assistant.core.documents import ExtractError

    store = _store(tmp_path)
    with pytest.raises(ExtractError):
        run(store.ensure_file(_write(tmp_path, "x.bin", "data"), "x.bin"))
    with pytest.raises(ExtractError):
        run(store.ensure_file(_write(tmp_path, "empty.md", "   "), "empty.md"))


def test_attachments_are_scoped_to_their_project_and_persist(tmp_path):
    store = _store(tmp_path)
    entry = run(store.ensure_file(_write(tmp_path, "a.md", "# T\n\nbody"), "a.md"))
    assert not store.is_registered("7", entry.hash)
    assert store.register_attachment("7", entry.hash, "a.md") == attachment_id(entry.hash)
    assert store.is_registered("7", entry.hash) and not store.is_registered("8", entry.hash)
    assert store.attachment_name("7", entry.hash) == "a.md"
    reopened = _store(tmp_path)
    assert reopened.is_registered("7", entry.hash)
    assert reopened.remove("7", [entry.hash]) == 1
    assert not _store(tmp_path).is_registered("7", entry.hash)


def test_view_is_the_full_text_but_excerpt_is_only_passages(tmp_path):
    store = _store(tmp_path)
    entry = run(store.ensure_file(_write(tmp_path, "r.md", _long_doc()), "r.md"))
    view = store.view([("r.md", entry)])
    assert isinstance(view, KnowledgeView) and isinstance(view, str)
    assert "stratified random sampling" in view  # grounding checks read the whole text
    assert attached_document(view) is view
    text, cut = view.excerpt("sampling procedure", 4000)
    assert cut and len(text.encode()) <= 4000 + 400
    assert "stratified random sampling" in text and "### r.md" in text


def test_one_query_per_field_each_gets_its_passage(tmp_path):
    store = _store(tmp_path)
    doc = "\n\n".join(
        [f"# Section {n}\n\nFiller {n}. " + "lorem ipsum dolor sit amet " * 60 for n in range(60)]
        + ["# Funding\n\nThe survey was funded by the Gates Foundation under grant 4711.",
           "# Mode\n\nInterviews were conducted by computer-assisted telephone."]
    )
    entry = run(store.ensure_file(_write(tmp_path, "d.md", doc), "d.md"))
    view = store.view([("d.md", entry)])
    text, cut = view.excerpt(["funded grant foundation", "interviews telephone mode"], 3500)
    assert cut and "Gates Foundation" in text and "telephone" in text


def test_small_documents_are_sent_whole(tmp_path):
    store = _store(tmp_path)
    entry = run(store.ensure_file(_write(tmp_path, "s.md", "# Short\n\nTiny doc."), "s.md"))
    text, cut = store.view([("s.md", entry)]).excerpt("anything", 4000)
    assert not cut and "Tiny doc." in text


def test_several_sources_are_labelled_and_searched_together(tmp_path):
    store = _store(tmp_path)
    a = run(store.ensure_file(_write(tmp_path, "methods.md", "# Methods\n\nThe weights are calibrated to the census."), "methods.md"))
    b = run(store.ensure_file(_write(tmp_path, "access.md", "# Access\n\nData are released under an open licence."), "access.md"))
    view = store.view([("methods.md", a), ("access.md", b)])
    assert view.names == ["methods.md", "access.md"]
    hit = view.search("licence open release", 2)[0]
    assert view.label(hit) == "access.md"


def test_embeddings_are_built_in_the_background_and_used(tmp_path):
    async def go():
        store = _store(tmp_path, FakeEmbedder())
        entry = await store.ensure_file(_write(tmp_path, "r.md", _long_doc()), "r.md")
        for _ in range(100):
            if entry.state == "ready":
                break
            await asyncio.sleep(0.05)
        return store, entry

    store, entry = run(go())
    assert entry.state == "ready" and entry.vectors is not None and entry.vectors.shape[0] == len(entry.chunks)
    view = store.view([("r.md", entry)])
    assert view.semantic
    assert (tmp_path / "cache" / "knowledge" / f"{entry.hash}.npz").is_file()
    # A restart reuses the saved vectors instead of embedding again.
    again = _store(tmp_path, FakeEmbedder()).get(entry.hash)
    assert again.vectors is not None and again.state == "ready"


def test_variable_search_ranks_by_label_and_falls_back_to_keywords(tmp_path):
    store = _store(tmp_path)
    varindex.configure(store)
    variables = [
        {"name": "l1", "label": "Number of permanent employees", "description": "", "question": "How many full-time workers?"},
        {"name": "b7", "label": "Years of experience of the top manager", "description": "", "question": ""},
        {"name": "wt_ES", "label": "Baseline sampling weight", "description": "", "question": ""},
    ]
    idx = varindex.for_variables(variables)
    assert idx.search("employees workers", 3)[0]["name"] == "l1"
    assert idx.search("weight", 3)[0]["name"] == "wt_ES"
    assert varindex.for_variables(variables) is idx  # same content, same index


def test_identical_content_under_two_names_is_searched_once(tmp_path):
    store = _store(tmp_path)
    text = "# Access\n\nData are released under an open licence."
    a = run(store.ensure_file(_write(tmp_path, "access.md", text), "access.md"))
    b = run(store.ensure_file(_write(tmp_path, "copy-of-access.md", text), "copy-of-access.md"))
    assert a.hash == b.hash
    view = store.view([("access.md", a), ("copy-of-access.md", b)])
    assert view.names == ["access.md"]  # the first one given wins
    assert len(view.search("open licence", 5)) == len(store.view([("access.md", a)]).search("open licence", 5))
