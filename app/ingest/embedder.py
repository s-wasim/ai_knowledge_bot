"""Local code embeddings.

Runs `jinaai/jina-embeddings-v2-base-code` in-process via sentence-transformers:
Apache 2.0, 768 dimensions, trained on 30 programming languages, no API key. The
weights are baked into the Docker image at build time, so nothing is downloaded
at request time.

Every entry point returns None rather than raising when the model is unavailable.
Retrieval is built to degrade to lexical and symbolic search in that case, and
`embedding_status()` exists so the degradation is reported instead of silently
pretending semantic search happened — the exact failure this project had with
Voyage, where an empty API key quietly downgraded every query.
"""

import logging
import os
import threading

logger = logging.getLogger(__name__)

MODEL_NAME = os.environ.get("KB_EMBED_MODEL", "jinaai/jina-embeddings-v2-base-code")
EMBED_DIMS = 768
BATCH_SIZE = 32
MAX_SEQ_LENGTH = 1024

_model = None
_load_error: str | None = None
_loaded = False
_lock = threading.Lock()


def _reset_cache() -> None:
    """Test-only: forget the cached model and any recorded load failure."""
    global _model, _load_error, _loaded
    with _lock:
        _model = None
        _load_error = None
        _loaded = False


def _load_model():
    """Construct the SentenceTransformer. Separated so tests can substitute it."""
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(MODEL_NAME, trust_remote_code=True, device="cpu")
    # Files are already split into chunks, so capping the sequence length keeps
    # CPU encoding predictable rather than letting one huge chunk stall an ingest.
    current = getattr(model, "max_seq_length", None) or MAX_SEQ_LENGTH
    model.max_seq_length = min(current, MAX_SEQ_LENGTH)
    return model


def get_model():
    """Return the process-wide model, or None if it could not be loaded.

    A failed load is cached: reloading a 300MB model on every request would turn
    one failure into a stall on every query.
    """
    global _model, _load_error, _loaded

    if _loaded:
        return _model

    with _lock:
        if _loaded:
            return _model
        try:
            _model = _load_model()
            _load_error = None
            logger.info("Loaded embedding model %s", MODEL_NAME)
        except Exception as e:
            _model = None
            _load_error = str(e) or e.__class__.__name__
            logger.error("Could not load embedding model %s: %s", MODEL_NAME, e)
        _loaded = True
        return _model


def is_embedding_available() -> bool:
    return get_model() is not None


def embedding_status() -> dict:
    """Reportable state of the embedder, surfaced by GET /health."""
    available = get_model() is not None
    return {
        "ok": available,
        "model": MODEL_NAME,
        "dims": EMBED_DIMS,
        "error": None if available else (_load_error or "model not loaded"),
    }


def _to_float_lists(vectors) -> list[list[float]]:
    """Convert whatever the encoder returned into plain Python float lists.

    psycopg cannot adapt numpy scalars, so this has to happen before the values
    reach the database.
    """
    return [[float(v) for v in vector] for vector in vectors]


def embed_texts(
    texts: list[str], batch_size: int = BATCH_SIZE
) -> list[list[float]] | None:
    """Embed documents for storage. Returns None if embedding is unavailable."""
    if not texts:
        return []

    model = get_model()
    if model is None:
        return None

    out: list[list[float]] = []
    try:
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            vectors = model.encode(
                batch,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
            out.extend(_to_float_lists(vectors))
    except Exception as e:
        logger.error("Embedding failed after %d/%d texts: %s", len(out), len(texts), e)
        return None

    return out


def embed_query(text: str) -> list[float] | None:
    """Embed a single search query. Returns None if embedding is unavailable."""
    if not text or not text.strip():
        return None

    model = get_model()
    if model is None:
        return None

    try:
        vectors = model.encode(
            [text],
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
    except Exception as e:
        logger.error("Query embedding failed: %s", e)
        return None

    converted = _to_float_lists(vectors)
    return converted[0] if converted else None
