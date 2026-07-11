import threading

from app.db import get_session
from app.graph.build import build_rag_graph
from app.retrieval.factory import create_retriever

_lock = threading.Lock()
_retriever = None
_retrieval_mode = None
_graph = None


def get_retriever_and_mode():
    global _retriever, _retrieval_mode
    if _retriever is None:
        with _lock:
            if _retriever is None:
                _retriever, _retrieval_mode = create_retriever(get_session)
    return _retriever, _retrieval_mode


def get_graph():
    global _graph
    if _graph is None:
        with _lock:
            if _graph is None:
                _graph = build_rag_graph()
    return _graph


def reset_singletons():
    """Test-only: clear cached singletons so tests can rebuild them fresh."""
    global _retriever, _retrieval_mode, _graph
    _retriever = None
    _retrieval_mode = None
    _graph = None
