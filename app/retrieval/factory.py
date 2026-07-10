from app.ingest.embedder import is_voyage_available
from app.retrieval.fts import FtsRetriever
from app.retrieval.vector import VectorRetriever


def create_retriever(session_factory):
    if is_voyage_available():
        return VectorRetriever(session_factory), "vector"
    else:
        return FtsRetriever(session_factory), "fts"


def get_mode_display(mode: str) -> str:
    if mode == "vector":
        return "Vector (voyage-code-3)"
    return "Full-text search"
