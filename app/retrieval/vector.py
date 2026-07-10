from sqlalchemy import text
from app.retrieval.base import ChunkData
from app.ingest.embedder import get_voyage_client


class VectorRetriever:
    def __init__(self, session_factory):
        self._session_factory = session_factory

    def search(self, repo_id: int, query: str, k: int = 8) -> list[ChunkData]:
        if not query.strip():
            return []

        client = get_voyage_client()
        if client is None:
            return []

        response = client.embed(
            texts=[query],
            model="voyage-code-3",
            input_type="query",
        )
        query_embedding = response.embeddings[0]

        session = self._session_factory()
        try:
            rows = session.execute(
                text(
                    """
                    SELECT path, start_line, end_line, content,
                           1 - (embedding <=> :query_embedding::vector) AS score
                    FROM chunks
                    WHERE repo_id = :repo_id
                      AND embedding IS NOT NULL
                    ORDER BY embedding <=> :query_embedding
                    LIMIT :k
                    """
                ),
                {
                    "repo_id": repo_id,
                    "query_embedding": "[" + ",".join(str(v) for v in query_embedding) + "]",
                    "k": k,
                },
            ).fetchall()

            return [
                ChunkData(
                    path=row.path,
                    start_line=row.start_line,
                    end_line=row.end_line,
                    content=row.content,
                    score=float(row.score),
                )
                for row in rows
            ]
        finally:
            session.close()
