"""Fetch candidate chunks from the hybrid retriever.

The retriever arrives through the graph config rather than being constructed here,
so tests can substitute one and the API layer can keep a single warm instance.
"""

import logging
from typing import Optional

from langchain_core.runnables import RunnableConfig

from app.graph.state import RagState, effective_query
from app.retrieval.fusion import MAX_CANDIDATES

logger = logging.getLogger(__name__)


def retrieve(state: RagState, config: Optional[RunnableConfig] = None) -> dict:
    # NOTE: do not add `from __future__ import annotations` to this module.
    # LangGraph inspects the runtime annotation to decide whether to inject
    # config; a stringized annotation makes it skip injection, leaving the node
    # with no retriever and every query returning nothing.
    retriever = ((config or {}).get("configurable") or {}).get("retriever")
    if retriever is None:
        logger.warning("No retriever configured; returning no candidates")
        return {"retrieved": []}

    repo_id = state.get("repo_id")
    if repo_id is None:
        return {"retrieved": []}

    query = effective_query(state)

    try:
        results = retriever.search(repo_id=repo_id, query=query, k=MAX_CANDIDATES)
    except Exception:
        # HybridRetriever already guards each sub-retriever, so this catches a
        # total failure such as a dead connection. Degrading to the not-found path
        # beats returning a 500.
        logger.error("Retrieval failed for repo %s", repo_id, exc_info=True)
        return {"retrieved": []}

    return {"retrieved": results}
