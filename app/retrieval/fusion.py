"""Reciprocal Rank Fusion.

The three retrievers produce scores on incomparable scales — cosine distance,
ts_rank_cd, and trigram similarity. Normalizing them against each other would
require calibration that drifts with every query. RRF sidesteps that by using only
rank position, and it rewards agreement: a chunk two retrievers both surface
outranks one that only the single best-scoring retriever found.
"""

from __future__ import annotations

from dataclasses import replace

from app.retrieval.base import ChunkData

RRF_K = 60
MAX_CANDIDATES = 24


def _richer(existing: ChunkData, candidate: ChunkData) -> ChunkData:
    """Merge metadata across duplicates.

    Retrievers select different columns, so the copy that survives must not lose
    a symbol or language that a sibling retriever supplied.
    """
    return replace(
        existing,
        symbol=existing.symbol or candidate.symbol,
        language=existing.language or candidate.language,
        content=existing.content or candidate.content,
    )


def reciprocal_rank_fusion(
    ranked_lists: dict[str, list[ChunkData] | None],
    k: int = RRF_K,
    limit: int = MAX_CANDIDATES,
) -> list[ChunkData]:
    """Fuse per-retriever ranked lists into one ranked list.

    `ranked_lists` maps a retriever name to its results, best first. A None value
    is tolerated so a retriever that failed does not break the query.
    Returns chunks with `score` set to the RRF score and `sources` set to the
    contributing retriever names, best first, capped at `limit`.
    """
    merged: dict[tuple[str, int], ChunkData] = {}
    scores: dict[tuple[str, int], float] = {}
    sources: dict[tuple[str, int], list[str]] = {}

    for name, results in ranked_lists.items():
        if not results:
            continue
        for rank, chunk in enumerate(results, start=1):
            key = chunk.key
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
            if key in merged:
                merged[key] = _richer(merged[key], chunk)
            else:
                merged[key] = chunk
                sources[key] = []
            if name not in sources[key]:
                sources[key].append(name)

    ordered = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)

    fused: list[ChunkData] = []
    for key, score in ordered[:limit]:
        fused.append(replace(merged[key], score=score, sources=tuple(sources[key])))

    return fused
