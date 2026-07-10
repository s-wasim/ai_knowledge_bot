from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass
class ChunkData:
    path: str
    start_line: int
    end_line: int
    content: str
    score: float


@runtime_checkable
class Retriever(Protocol):
    def search(self, repo_id: int, query: str, k: int = 8) -> list[ChunkData]: ...
