"""Retriever construction and mode reporting.

The mode string is not cosmetic. The previous implementation silently swapped
semantic search for full-text search whenever an API key was missing, and nothing
in the UI said so. Here, a missing embedding model yields `"degraded"`, which the
health endpoint and the UI both surface.
"""

from app.ingest.embedder import is_embedding_available
from app.retrieval.hybrid import HybridRetriever

MODE_HYBRID = "hybrid"
MODE_DEGRADED = "degraded"


def create_retriever(session_factory) -> tuple[HybridRetriever, str]:
    """Build the retriever and report which signals are actually available."""
    mode = MODE_HYBRID if is_embedding_available() else MODE_DEGRADED
    return HybridRetriever(session_factory), mode


def get_mode_display(mode: str) -> str:
    if mode == MODE_HYBRID:
        return "Hybrid (code embeddings + full-text + symbols)"
    return "Degraded (full-text + symbols, no embeddings)"
