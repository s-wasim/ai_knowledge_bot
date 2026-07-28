"""Symbolic retrieval: fuzzy matching on identifiers, symbols, and paths.

This is what makes a question naming `get_connection` surface that definition.
Dense embeddings do it unreliably (an identifier is a token, not a concept) and
stemmed full-text search does it badly. Trigram similarity over the `symbol` and
`path` columns handles it directly, including near-misses like `getConnection`
against `get_connection`.

If a query contains no code-shaped tokens, this retriever returns nothing rather
than guessing — arbitrary trigram noise would otherwise be rewarded by fusion.
"""

from __future__ import annotations

import re

from sqlalchemy import text

from app.retrieval.base import ChunkData

# Ordered by specificity. Each pattern targets a shape that is unambiguously
# code rather than prose.
_PATTERNS = (
    re.compile(r"`([^`]+)`"),                            # `backticked`
    re.compile(r"\b([a-z][a-z0-9]*(?:_[a-z0-9]+)+)\b"),  # snake_case
    re.compile(r"\b([A-Z][A-Z0-9]*(?:_[A-Z0-9]+)+)\b"),  # SCREAMING_SNAKE
    re.compile(r"\b([a-z][a-z0-9]*(?:[A-Z][a-z0-9]*)+)\b"),  # camelCase
    re.compile(r"\b([A-Z][a-z0-9]+(?:[A-Z][a-z0-9]*)+)\b"),  # PascalCase
    re.compile(r"\b([\w]+(?:\.[\w]+)+)\b"),              # dotted.path, file.ext
    re.compile(r"\b([\w/]+/[\w./]+)\b"),                 # path/to/file
)

MIN_IDENTIFIER_LENGTH = 3


def extract_identifiers(query: str, limit: int = 8) -> list[str]:
    """Pull code-shaped tokens out of a natural-language query.

    Returns an ordered, deduplicated list. Empty when the query is plain prose.
    """
    if not query:
        return []

    found: list[str] = []
    seen: set[str] = set()

    for pattern in _PATTERNS:
        for match in pattern.finditer(query):
            token = match.group(1).strip()
            if len(token) < MIN_IDENTIFIER_LENGTH:
                continue
            if token.lower() in seen:
                continue
            seen.add(token.lower())
            found.append(token)

    return found[:limit]


class SymbolicRetriever:
    """Trigram similarity over symbol and path, plus a literal content match."""

    name = "symbol"

    def __init__(self, session_factory):
        self._session_factory = session_factory

    def search(self, repo_id: int, query: str, k: int = 20) -> list[ChunkData]:
        identifiers = extract_identifiers(query)
        if not identifiers:
            return []

        needle = " ".join(identifiers)
        # One literal term keeps exact usages (call sites, not just definitions)
        # in play; ILIKE is bounded by the repo_id filter.
        literal = f"%{identifiers[0]}%"

        session = self._session_factory()
        try:
            rows = session.execute(
                text(
                    """
                    SELECT path, start_line, end_line, content, symbol, language,
                           GREATEST(
                               similarity(coalesce(symbol, ''), :needle),
                               similarity(path, :needle),
                               CASE WHEN content ILIKE :literal THEN 0.35 ELSE 0 END
                           ) AS score
                    FROM chunks
                    WHERE repo_id = :repo_id
                      AND (
                          coalesce(symbol, '') %% :needle
                          OR path %% :needle
                          OR content ILIKE :literal
                      )
                    ORDER BY score DESC, start_line ASC
                    LIMIT :k
                    """
                ),
                {"repo_id": repo_id, "needle": needle, "literal": literal, "k": k},
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
