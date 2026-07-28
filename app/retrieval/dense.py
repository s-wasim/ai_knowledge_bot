"""Dense retrieval: cosine similarity over local code embeddings.

Returns nothing when the embedding model is unavailable, so the hybrid retriever
degrades to lexical and symbolic search instead of failing. Rows whose embedding
is NULL are excluded — an ingest that ran without the model leaves them behind,
and including them would silently return arbitrary results.
"""

from __future__ import annotations

import logging

from sqlalchemy import text

from app.ingest.embedder import embed_query
from app.retrieval.base import ChunkData

logger = logging.getLogger(__name__)


class DenseRetriever:
    name = "dense"

    def __init__(self, session_factory):
        self._session_factory = session_factory

    def search(self, repo_id: int, query: str, k: int = 30) -> list[ChunkData]:
        vector = embed_query(query)
        if vector is None:
            logger.debug("Dense retrieval skipped: no embedding available")
            return []

        session = self._session_factory()
        try:
            rows = session.execute(
                text(
                    """
                    SELECT path, start_line, end_line, content, symbol, language,
                           1 - (embedding <=> CAST(:vec AS vector)) AS score
                    FROM chunks
                    WHERE repo_id = :repo_id
                      AND embedding IS NOT NULL
                    ORDER BY embedding <=> CAST(:vec AS vector)
                    LIMIT :k
                    """
                ),
                {"repo_id": repo_id, "vec": str(vector), "k": k},
            ).fetchall()

            return [
                ChunkData(
                    path=row.path,
                    start_line=row.start_line,
                    end_line=row.end_line,
                    content=row.content,
                    score=float(row.score or 0.0),
                    symbol=row.symbol,
                    language=row.language,
                )
                for row in rows
            ]
        finally:
            session.close()
