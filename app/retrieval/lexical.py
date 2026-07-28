"""Lexical retrieval: full-text search tuned for code.

Two deliberate departures from stock Postgres full-text search:

1. The `simple` configuration, not `english`. English stemming mangles code —
   `configured` and `config` collapse together while `getConnection` survives as
   one opaque token.
2. The indexed text includes an identifier-split copy of the content (see
   `_TSV_EXPRESSION` in app/db.py), so `get connection` in a question matches
   `getConnection` in the source.

The AND-to-OR relaxation is retained from the original implementation: a
multi-word question rarely has every term in one chunk, and requiring that
returns nothing.
"""

from __future__ import annotations

from sqlalchemy import text

from app.retrieval.base import ChunkData


class LexicalRetriever:
    name = "text"

    def __init__(self, session_factory):
        self._session_factory = session_factory

    def search(self, repo_id: int, query: str, k: int = 30) -> list[ChunkData]:
        if not query.strip():
            return []

        session = self._session_factory()
        try:
            rows = session.execute(
                text(
                    """
                    SELECT path, start_line, end_line, content, symbol, language,
                           ts_rank_cd(tsv, query) AS score
                    FROM chunks,
                         to_tsquery(
                             'simple',
                             replace(
                                 plainto_tsquery('simple', :query)::text,
                                 ' & ', ' | '
                             )
                         ) AS query
                    WHERE repo_id = :repo_id
                      AND tsv @@ query
                    ORDER BY score DESC, start_line ASC
                    LIMIT :k
                    """
                ),
                {"repo_id": repo_id, "query": _split_identifiers(query), "k": k},
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


def _split_identifiers(query: str) -> str:
    """Mirror the indexing-side identifier split on the query side.

    The stored tsvector contains both the original tokens and a split copy, so
    adding split forms of the query's own identifiers lets `getConnection` in a
    question match `get_connection` in the source, and vice versa.
    """
    import re

    extra: list[str] = []
    for token in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", query):
        pieces = re.findall(r"[A-Z]+(?![a-z])|[A-Z][a-z0-9]*|[a-z0-9]+", token)
        if len(pieces) > 1:
            extra.extend(pieces)

    if not extra:
        return query

    return query + " " + " ".join(extra)
