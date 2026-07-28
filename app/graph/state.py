from dataclasses import dataclass
from typing import TypedDict, Literal

from app.retrieval.base import ChunkData


@dataclass
class GradedChunk:
    chunk: ChunkData
    keep: bool
    reason: str
    # Claude's confidence that this chunk answers the question, 0-1. Orders the
    # kept chunks and is shown in the UI alongside the retrieval score.
    relevance: float = 0.0


@dataclass
class Citation:
    chunk: ChunkData
    index: int


class RagState(TypedDict):
    question: str
    chat_history: list[dict]
    rewritten_query: str | None
    retrieved: list[ChunkData]
    graded: list[GradedChunk]
    answer: str | None
    citations: list[Citation]
    mode: Literal["hybrid", "degraded"]
    repo_id: int | None


def effective_query(state: RagState) -> str:
    """The query retrieval, grading, and answering should use.

    `rewritten_query` is present in the initial state with a value of None, so
    `state.get("rewritten_query", state["question"])` never reached its default —
    it returned None whenever rewriting was skipped or failed.
    """
    return state.get("rewritten_query") or state["question"]
