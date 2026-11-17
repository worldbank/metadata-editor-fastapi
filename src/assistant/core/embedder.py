"""
Local text embedder: ONNX Runtime + a Hugging Face tokenizer, CPU, any OS, no PyTorch, no remote code.

Embeddings are always computed on this server and never sent to a paid API. Model files are pinned
by revision and their SHA-256 is checked before the first load, as portable-slm does for its weights.

UltraBERT-Embed-150M (dmis-lab, Apache-2.0) is the default. The ONNX file is pastel-labs' export of
the official weights; on 2026-10-20 it was compared with the official sentence-transformers model on 31
texts from the editor's own documentation (3 to 3,058 tokens, with and without the query instruction):
minimum cosine 1.00000, identical token ids.
"""

from __future__ import annotations

import hashlib
import logging
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EmbedderSpec:
    id: str
    repo: str
    revision: str
    onnx_file: str
    onnx_sha256: str
    tokenizer_file: str
    pad_token: str
    dims: int
    pooling: Literal["cls", "mean"]
    query_prefix: str
    document_prefix: str
    max_tokens: int
    # Retrieval floor for chunks no query term matched. Measured per model, because similarity scales
    # differ: a shared constant would be wrong for at least one embedder.
    min_similarity: float
    license: str


EMBEDDERS: dict[str, EmbedderSpec] = {
    "ultrabert-embed-150m": EmbedderSpec(
        id="ultrabert-embed-150m",
        repo="pastel-labs/UltraBERT-Embed-150M-ONNX",
        revision="932b2b7639699d25bb60621b74897d665d238fd5",
        onnx_file="onnx/model.onnx",
        onnx_sha256="2d747633eb4c268a66db287cc1a4918f9d01972d36ef3c2fcf0ab2097e00f397",
        tokenizer_file="tokenizer.json",
        pad_token="<|endoftext|>",  # tokenizer_config.json
        dims=768,
        pooling="cls",
        # From the official config_sentence_transformers.json: queries carry the instruction,
        # documents are embedded raw.
        query_prefix="Instruct: Given a question, retrieve passages that answer the question\nQuery: ",
        document_prefix="",
        # The model takes 32K tokens; chunks are ~1.2K characters, so this only bounds a pathological input.
        max_tokens=2048,
        # On the editor docs (2,563 chunks): off-topic questions peaked at 0.161-0.179, on-topic at
        # 0.305-0.479 (2026-10-20). 0.25 sits between them.
        min_similarity=0.25,
        license="Apache-2.0 (weights: dmis-lab/UltraBERT-Embed-150M)",
    ),
}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


class OnnxEmbedder:
    """Loads lazily and thread-safely; ``embed`` is synchronous (call it from a worker thread)."""

    def __init__(self, spec: EmbedderSpec, cache_dir: Path, *, batch_size: int = 16) -> None:
        self.spec = spec
        self.cache_dir = cache_dir
        self.batch_size = batch_size
        self._lock = threading.Lock()
        self._session = None
        self._tokenizer = None

    @property
    def id(self) -> str:
        return self.spec.id

    @property
    def dims(self) -> int:
        return self.spec.dims

    def _fetch(self, filename: str) -> Path:
        from huggingface_hub import hf_hub_download

        return Path(
            hf_hub_download(self.spec.repo, filename, revision=self.spec.revision, cache_dir=str(self.cache_dir))
        )

    def load(self) -> None:
        with self._lock:
            if self._session is not None:
                return
            import onnxruntime as ort
            from tokenizers import Tokenizer

            onnx_path = self._fetch(self.spec.onnx_file)
            marker = onnx_path.with_name(onnx_path.name + ".sha256-ok")
            if not marker.is_file() or marker.read_text().strip() != self.spec.onnx_sha256:
                digest = _sha256(onnx_path)
                if digest != self.spec.onnx_sha256:
                    raise RuntimeError(f"{self.spec.id}: SHA-256 mismatch for {self.spec.onnx_file} ({digest})")
                try:
                    marker.write_text(digest)
                except OSError:
                    pass  # read-only cache: verify again next start
            tokenizer = Tokenizer.from_file(str(self._fetch(self.spec.tokenizer_file)))
            tokenizer.enable_truncation(self.spec.max_tokens)
            pad_id = tokenizer.token_to_id(self.spec.pad_token)
            if pad_id is None:
                raise RuntimeError(f"{self.spec.id}: pad token {self.spec.pad_token!r} not in tokenizer")
            # Right padding: CLS pooling reads position 0, and the attention mask hides the pad.
            tokenizer.enable_padding(direction="right", pad_id=pad_id, pad_token=self.spec.pad_token)
            options = ort.SessionOptions()
            threads = int(os.getenv("ASSISTANT_EMBEDDER_THREADS", "0"))
            if threads > 0:
                options.intra_op_num_threads = threads
            self._session = ort.InferenceSession(str(onnx_path), options, providers=["CPUExecutionProvider"])
            self._tokenizer = tokenizer
            logger.info("Assistant: embedder %s loaded", self.spec.id)

    def embed(self, texts: list[str], kind: Literal["query", "document"]) -> np.ndarray:
        """L2-normalised float32 vectors, one row per text."""
        self.load()
        prefix = self.spec.query_prefix if kind == "query" else self.spec.document_prefix
        out = np.zeros((len(texts), self.spec.dims), dtype=np.float32)
        # Similar lengths per batch, so padding stays small.
        order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
        for start in range(0, len(order), self.batch_size):
            idx = order[start:start + self.batch_size]
            encs = self._tokenizer.encode_batch([prefix + texts[i] for i in idx])
            ids = np.array([e.ids for e in encs], dtype=np.int64)
            mask = np.array([e.attention_mask for e in encs], dtype=np.int64)
            hidden = self._session.run(None, {"input_ids": ids, "attention_mask": mask})[0]
            if self.spec.pooling == "cls":
                vecs = hidden[:, 0, :]
            else:
                m = mask[:, :, None].astype(np.float32)
                vecs = (hidden * m).sum(1) / np.maximum(m.sum(1), 1.0)
            norms = np.linalg.norm(vecs, axis=1, keepdims=True)
            out[idx] = vecs / np.maximum(norms, 1e-12)
        return out


def make_embedder(name: str, cache_dir: Path) -> Optional[OnnxEmbedder]:
    """``none`` (or empty) disables embeddings: retrieval falls back to BM25 alone."""
    key = (name or "").strip().lower()
    if key in ("", "none", "off"):
        return None
    spec = EMBEDDERS.get(key)
    if spec is None:
        raise ValueError(f"unknown embedder {name!r}; known: {', '.join(EMBEDDERS)} or none")
    return OnnxEmbedder(spec, cache_dir)
