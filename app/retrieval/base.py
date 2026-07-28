from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable


@dataclass
class ChunkData:
    path: str
    start_line: int
    end_line: int
    content: str
    score: float
    symbol: str | None = None
    language: str | None = None
    # Names of the retrievers that surfaced this chunk. Populated by fusion and
    # shown in the UI so retrieval provenance is visible rather than guessed at.
    sources: tuple[str, ...] = field(default_factory=tuple)

    @property
    def key(self) -> tuple[str, int]:
        """Identity used for cross-retriever deduplication."""
        return (self.path, self.start_line)


@runtime_checkable
class Retriever(Protocol):
    def search(self, repo_id: int, query: str, k: int = 8) -> list[ChunkData]: ...
