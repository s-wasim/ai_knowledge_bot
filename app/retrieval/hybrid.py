"""Hybrid retrieval: three signals, fused by rank.

Each retriever runs inside its own guard. One failing — a missing model, a broken
extension, a malformed query — degrades the result rather than failing the
request, and says so in the log. A query that returns nothing from every retriever
returns an empty list, which routes the graph to its honest not-found path.
"""

from __future__ import annotations

import logging

from app.retrieval.base import ChunkData
from app.retrieval.dense import DenseRetriever
from app.retrieval.fusion import MAX_CANDIDATES, reciprocal_rank_fusion
from app.retrieval.lexical import LexicalRetriever
from app.retrieval.symbolic import SymbolicRetriever

logger = logging.getLogger(__name__)

# Each retriever over-fetches so fusion has room to reward agreement before the
# candidate list is capped.
PER_RETRIEVER_K = 30
SYMBOLIC_K = 20


class HybridRetriever:
    name = "hybrid"

    def __init__(self, session_factory):
        self._dense = DenseRetriever(session_factory)
        self._lexical = LexicalRetriever(session_factory)
        self._symbolic = SymbolicRetriever(session_factory)

    def _safe(self, retriever, repo_id: int, query: str, k: int) -> list[ChunkData]:
        try:
            return retriever.search(repo_id=repo_id, query=query, k=k)
        except Exception:
            logger.warning(
                "Retriever %s failed; continuing without it", retriever.name, exc_info=True
            )
            return []

    def search(self, repo_id: int, query: str, k: int = MAX_CANDIDATES) -> list[ChunkData]:
        ranked = {
            self._dense.name: self._safe(self._dense, repo_id, query, PER_RETRIEVER_K),
            self._lexical.name: self._safe(self._lexical, repo_id, query, PER_RETRIEVER_K),
            self._symbolic.name: self._safe(self._symbolic, repo_id, query, SYMBOLIC_K),
        }

        counts = {name: len(results) for name, results in ranked.items()}
        fused = reciprocal_rank_fusion(ranked, limit=k)
        logger.info(
            "Retrieved %d candidates for repo %s (%s)",
            len(fused),
            repo_id,
            ", ".join(f"{name}={count}" for name, count in counts.items()),
        )

        return fused
