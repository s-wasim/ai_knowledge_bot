from sqlalchemy import text

from app.retrieval.base import ChunkData


class FtsRetriever:
    def __init__(self, session_factory):
        self._session_factory = session_factory

    def search(self, repo_id: int, query: str, k: int = 8) -> list[ChunkData]:
        if not query.strip():
            return []

        session = self._session_factory()
        try:
            rows = session.execute(
                text(
                    """
                    SELECT path, start_line, end_line, content,
                           ts_rank(tsv, query) AS score
                    FROM chunks,
                         to_tsquery(
                             'english',
                             replace(plainto_tsquery('english', :query)::text, ' & ', ' | ')
                         ) AS query
                    WHERE repo_id = :repo_id
                      AND tsv @@ query
                    ORDER BY score DESC
                    LIMIT :k
                    """
                ),
                {"repo_id": repo_id, "query": query, "k": k},
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
