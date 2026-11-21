import asyncio
import os
from pathlib import Path

import numpy as np
import pytest

from src.assistant.core.retrieval import (
    Document,
    Index,
    RetrievalOptions,
    RetrievalService,
    bm25,
    build_chunks,
    chunk_text,
    rank_chunks,
    select_under_cap,
    tokenize,
)

DOC = """# Projects

Create a project from the Projects page. Each project has a type such as survey or indicator.

# Collaborators

Share a project with a collaborator by adding their email in the Collaborators tab.

# Data files

Import a data file (Stata, SPSS, CSV) and generate summary statistics for each variable.

# Universe

The universe describes the population covered by the study, for example all households.
"""


def test_tokenize_splits_identifiers_and_drops_function_words():
    assert tokenize("The house_hold_id and getUserName of HHID") == ["house", "hold", "id", "get", "user", "name", "hhid"]
    assert tokenize("a b 7 is") == ["7"]


def test_chunk_text_keeps_headings_and_splits_long_sections():
    chunks = chunk_text(DOC, path="guide")
    assert [c.heading for c in chunks] == ["Projects", "Collaborators", "Data files", "Universe"]
    long = "# Big\n\n" + " ".join(f"word{i}" for i in range(2000))
    parts = chunk_text(long, target_chars=400, overlap_chars=50)
    assert len(parts) > 5 and all(len(p.text) <= 400 for p in parts)
    assert parts[1].text.startswith(parts[0].text[-50:].strip()[:10])  # overlap carried


def index_for(text=DOC, vectors=None):
    chunks = build_chunks([("guide", text)], RetrievalOptions())
    return Index(chunks=chunks, key="k", vectors=vectors)


def test_bm25_finds_the_matching_section_and_ignores_ubiquitous_terms():
    idx = index_for()
    scores = bm25(idx, "How do I share with a collaborator?", RetrievalOptions())
    best = max(scores, key=scores.get)
    assert idx.chunks[best].heading == "Collaborators"
    # "project" appears in 2 of 4 chunks (> floor(4/3)=1 but the floor is 2): still scores.
    assert bm25(idx, "project", RetrievalOptions())
    assert bm25(idx, "capital of Peru", RetrievalOptions()) == {}


def test_fusion_and_similarity_floor():
    idx = index_for()
    n = len(idx.chunks)
    vectors = np.eye(n, 8, dtype=np.float32)
    idx.vectors = vectors
    query = vectors[3]  # semantically the "Universe" chunk
    opts = RetrievalOptions(top_k=2)
    ranked = rank_chunks(idx, "who is covered by the study population", opts, query, floor=0.5)
    assert ranked[0].chunk.heading == "Universe" and ranked[0].semantic == pytest.approx(1.0)
    # Below the floor and no keyword: nothing comes back.
    off = np.zeros(8, dtype=np.float32)
    off[7] = 1.0
    assert rank_chunks(idx, "banana bread recipe", opts, off, floor=0.5) == []


def test_select_under_cap_reports_omissions_and_truncates_first():
    idx = index_for()
    ranked = rank_chunks(idx, "project collaborator data universe", RetrievalOptions(top_k=4))
    sel = select_under_cap(ranked, 250)
    assert len(sel.text.encode()) <= 250 and sel.omitted and sel.labels
    tiny = select_under_cap(ranked, 60)
    assert "[truncated" in tiny.text and len(tiny.text.encode()) <= 60


class FakeEmbedder:
    id = "fake"
    dims = 8

    def __init__(self):
        self.calls = 0

    def load(self):
        pass

    def embed(self, texts, kind):
        self.calls += 1
        out = np.zeros((len(texts), 8), dtype=np.float32)
        for i, t in enumerate(texts):
            out[i, hash(t.split()[0].lower()) % 8] = 1.0
        return out


def make_service(tmp_path, text=DOC, embedder=None):
    doc = tmp_path / "guide.md"
    doc.write_text(text, encoding="utf-8")
    return RetrievalService([Document(doc, "guide")], RetrievalOptions(min_similarity=0.5), embedder, tmp_path / "idx.npz")


def test_service_keyword_then_embeddings_then_reuses_index(tmp_path):
    emb = FakeEmbedder()
    svc = make_service(tmp_path, embedder=emb)

    async def go():
        await svc.start()
        assert svc.status()["state"] in {"keyword", "embedding", "ready"}
        await svc._task
        return await svc.search("share with a collaborator")

    text, info = asyncio.run(go())
    assert svc.status()["state"] == "ready" and svc.status()["embedded"]
    assert "### Collaborators — guide" in text and info["semantic"] is True

    emb2 = FakeEmbedder()
    svc2 = make_service(tmp_path, embedder=emb2)
    svc2.prepare_keyword()
    svc2.prepare_embeddings()
    assert emb2.calls == 0  # loaded from disk, not re-embedded

    svc3 = make_service(tmp_path, text=DOC + "\n# New\n\nA new section.\n", embedder=FakeEmbedder())
    svc3.prepare_keyword()
    svc3.prepare_embeddings()
    assert svc3.embedder.calls == 1  # content changed: rebuilt


def test_service_without_embedder_or_documents(tmp_path):
    svc = make_service(tmp_path)
    asyncio.run(svc.start())
    text, info = asyncio.run(svc.search("collaborator"))
    assert "Collaborators" in text and "keyword search only" in text
    missing = RetrievalService([Document(tmp_path / "nope.md", "x")], RetrievalOptions(), None, tmp_path / "i.npz")
    asyncio.run(missing.start())
    assert missing.status()["state"] == "failed"
    assert asyncio.run(missing.search("anything")) == ("", {"state": "failed", "hits": []})


def test_editor_context_includes_docs(tmp_path):
    from src.assistant.core.protocol import ChatRequest
    from src.assistant.host.editor import EditorHost

    svc = make_service(tmp_path)
    asyncio.run(svc.start())
    host = EditorHost(None, retrieval=svc)
    ctx = asyncio.run(host.context(ChatRequest(question="How do I share with a collaborator?")))
    assert "## From the editor's documentation" in ctx.text and "docs:keyword" in ctx.sources
    assert ctx.citations and ctx.citations[0].startswith("Collaborators")


@pytest.mark.skipif(not os.getenv("ASSISTANT_TEST_EMBEDDER_CACHE"), reason="set ASSISTANT_TEST_EMBEDDER_CACHE to run the real model")
def test_real_ultrabert_batch_matches_single():
    from src.assistant.core.embedder import make_embedder

    emb = make_embedder("ultrabert-embed-150m", Path(os.environ["ASSISTANT_TEST_EMBEDDER_CACHE"]))
    texts = ["short", "a much longer passage about household surveys and the universe field " * 5, "middle length text here"]
    batch = emb.embed(texts, "document")
    single = np.stack([emb.embed([t], "document")[0] for t in texts])
    assert batch.shape == (3, 768)
    assert np.allclose((batch * single).sum(1), 1.0, atol=1e-4)
    q = emb.embed(["What does the universe field mean?"], "query")[0]
    assert float(q @ batch[1]) > float(q @ batch[0])
