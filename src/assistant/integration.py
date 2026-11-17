"""Register optional assistant routes and app state."""

from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI

logger = logging.getLogger(__name__)


def _load_assistant_env(path: Path) -> None:
    if path.is_file():
        load_dotenv(path, override=False)
        logger.debug("Environment: loaded assistant.env from %s", path)
    else:
        logger.debug("Environment: no assistant.env at %s (optional)", path)


def _resolve(project_root: Path, value: str) -> Path:
    p = Path(value)
    return p if p.is_absolute() else project_root / p


def _parse_docs(project_root: Path, value: str):
    """``path=label;path=label``. A bare path is labelled with its file name."""
    from src.assistant.core.retrieval import Document

    docs = []
    for item in filter(None, (part.strip() for part in value.split(";"))):
        path, _, label = item.partition("=")
        resolved = _resolve(project_root, path.strip())
        docs.append(Document(path=resolved, label=label.strip() or resolved.name))
    return docs


def _make_retrieval(project_root: Path):
    from src.assistant.core.embedder import make_embedder
    from src.assistant.core.retrieval import RetrievalOptions, RetrievalService

    docs = _parse_docs(project_root, os.getenv("ASSISTANT_DOCS", ""))
    if not docs:
        return None
    try:
        embedder = make_embedder(
            os.getenv("ASSISTANT_EMBEDDER", "ultrabert-embed-150m"),
            _resolve(project_root, os.getenv("ASSISTANT_MODEL_CACHE", "db/models")),
        )
    except ValueError as exc:
        logger.warning("Assistant: %s; using keyword search only", exc)
        embedder = None
    defaults = RetrievalOptions()
    options = RetrievalOptions(
        target_chars=int(os.getenv("ASSISTANT_RETRIEVAL_CHUNK_CHARS", defaults.target_chars)),
        top_k=int(os.getenv("ASSISTANT_RETRIEVAL_TOP_K", defaults.top_k)),
        max_bytes=int(os.getenv("ASSISTANT_RETRIEVAL_MAX_BYTES", defaults.max_bytes)),
        alpha=float(os.getenv("ASSISTANT_RETRIEVAL_ALPHA", defaults.alpha)),
        min_similarity=float(v) if (v := os.getenv("ASSISTANT_RETRIEVAL_MIN_SIMILARITY")) else None,
    )
    index_path = _resolve(project_root, os.getenv("ASSISTANT_INDEX_PATH", "db/assistant_index.npz"))
    return RetrievalService(docs, options, embedder, index_path)


def _make_knowledge(project_root: Path):
    """The project sources index. Shares the embedder with the documentation retrieval when there is one."""
    from src.assistant.core.embedder import make_embedder
    from src.assistant.core.knowledge import KnowledgeStore
    from src.assistant.core.retrieval import RetrievalOptions

    cache = _resolve(project_root, os.getenv("ASSISTANT_KNOWLEDGE_DIR", os.getenv("ASSISTANT_MODEL_CACHE", "db/models") + "/../sources"))
    try:
        embedder = make_embedder(
            os.getenv("ASSISTANT_EMBEDDER", "ultrabert-embed-150m"),
            _resolve(project_root, os.getenv("ASSISTANT_MODEL_CACHE", "db/models")),
        )
    except ValueError as exc:
        logger.warning("Assistant: %s; sources use keyword search only", exc)
        embedder = None
    defaults = RetrievalOptions()
    options = RetrievalOptions(
        alpha=float(os.getenv("ASSISTANT_RETRIEVAL_ALPHA", defaults.alpha)),
        min_similarity=float(v) if (v := os.getenv("ASSISTANT_RETRIEVAL_MIN_SIMILARITY")) else None,
    )
    cap_mb = int(os.getenv("ASSISTANT_KNOWLEDGE_MAX_MB", "2048"))
    return KnowledgeStore(cache, embedder, options, max_disk_bytes=cap_mb * 1024 * 1024)


def _init_app_state(app: FastAPI, project_root: Path) -> None:
    from src.assistant.core.chat import TurnLimits
    from src.assistant.core.config import ConfigError, load_registry
    from src.assistant.core.llm import Gateway
    from src.assistant.host.editor import EditorHost

    slots = int(os.getenv("ASSISTANT_CONCURRENCY", "4"))
    app.assistant_sem = asyncio.Semaphore(slots)
    # Reviews and fills hold a model slot for a minute or more; capping how many run at once keeps the
    # rest of the slots for chat. Default: half, so with 4 slots at most 2 jobs run and chat always has 2.
    job_slots = int(os.getenv("ASSISTANT_JOB_SLOTS", str(max(1, slots // 2))))
    # Never all of them: with one model slot a job and a chat simply take turns.
    app.assistant_job_sem = asyncio.Semaphore(max(1, min(job_slots, slots - 1)) if slots > 1 else 1)
    app.assistant_max_inflight = int(os.getenv("ASSISTANT_MAX_INFLIGHT", "32"))
    app.assistant_inflight = 0
    # Running reviews and fills per person (the proxy names the person); more would starve other users.
    app.assistant_max_jobs_per_owner = int(os.getenv("ASSISTANT_MAX_JOBS_PER_USER", "2"))
    app.assistant_limits = TurnLimits(
        max_history=int(os.getenv("ASSISTANT_MAX_HISTORY", "6")),
        context_max_bytes=int(os.getenv("ASSISTANT_CONTEXT_MAX_BYTES", "16384")),
    )
    help_path = os.getenv("ASSISTANT_HELP_PATH", "")
    app.assistant_retrieval = _make_retrieval(project_root)
    app.assistant_knowledge = _make_knowledge(project_root)
    from src.assistant.host import varindex

    varindex.configure(app.assistant_knowledge)
    app.assistant_host = EditorHost(
        _resolve(project_root, help_path) if help_path else None,
        retrieval=app.assistant_retrieval,
    )

    models_path = _resolve(project_root, os.getenv("ASSISTANT_MODELS_PATH", "assistant.models.yaml"))
    app.assistant_gateway = None
    app.assistant_config_error = None
    try:
        registry = load_registry(models_path)
        app.assistant_gateway = Gateway(
            registry,
            timeout_sec=float(os.getenv("ASSISTANT_REQUEST_TIMEOUT_SEC", "120")),
            num_retries=int(os.getenv("ASSISTANT_NUM_RETRIES", "0")),
        )
        unconfigured = [m.alias for m in registry.models.values() if not m.configured]
        logger.info(
            "Assistant: %d model(s) usable from %s%s",
            len(registry.usable()),
            models_path.name,
            f" (unconfigured: {', '.join(unconfigured)})" if unconfigured else "",
        )
    except ConfigError as exc:
        # Routes stay mounted so /assistant/status can say what is wrong; chat returns 503.
        app.assistant_config_error = str(exc)
        logger.warning("Assistant: %s", exc)


def register_assistant(app: FastAPI, project_root: Path) -> bool:
    """
    Load assistant.env and mount /assistant routes when LiteLLM is installed.

    Returns True if the routes were mounted.
    """
    _load_assistant_env(project_root / "assistant.env")
    if os.getenv("ASSISTANT_ENABLED", "1").strip().lower() in {"0", "false", "no", "off"}:
        logger.info("Assistant disabled (ASSISTANT_ENABLED=0)")
        return False
    # LiteLLM downloads its model price map at import unless told to use the bundled copy. The worker
    # must start offline, so this is set before the first import.
    os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    # The Hugging Face client keeps logs and its download cache under $HOME/.cache unless told otherwise.
    # A service user's home is often the (root-owned) project folder, where the embedding download then
    # fails with "Permission denied"; keep it next to the model cache, which the service owns. Read by
    # huggingface_hub at import, so set before anything imports it. Compose sets HF_HOME itself.
    os.environ.setdefault(
        "HF_HOME",
        str(_resolve(project_root, os.getenv("ASSISTANT_MODEL_CACHE", "db/models")).parent / "hf"),
    )
    try:
        import litellm
        import yaml  # noqa: F401
    except ImportError:
        logger.info(
            "Assistant not installed (optional). Install with: pip install -r requirements-assistant.txt"
        )
        return False

    # LiteLLM warns once per model it has no price for, which is every local model.
    litellm.suppress_debug_info = True
    logging.getLogger("LiteLLM").setLevel(logging.ERROR)

    _init_app_state(app, project_root)

    from src.routers.assistant import router as assistant_router

    app.include_router(assistant_router)
    if app.assistant_retrieval is not None:
        # Keyword search is ready when startup finishes; embeddings build or load in a thread after it.
        app.add_event_handler("startup", app.assistant_retrieval.start)
    logger.info("Assistant routes registered at /assistant")
    return True
