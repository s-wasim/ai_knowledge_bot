from dataclasses import dataclass, field
from typing import TypedDict, Literal

from app.retrieval.base import ChunkData


@dataclass
class GradedChunk:
    chunk: ChunkData
    keep: bool
    reason: str


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
    mode: Literal["vector", "fts"]
