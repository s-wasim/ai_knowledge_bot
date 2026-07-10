"""Holds the global retriever instance, avoiding circular imports."""

_retriever = None

def set_retriever(retriever):
    global _retriever
    _retriever = retriever

def get_retriever():
    return _retriever
