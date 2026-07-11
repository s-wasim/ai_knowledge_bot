from typing import Optional

from pydantic import BaseModel


class HealthOut(BaseModel):
    db_ok: bool
    db_error: Optional[str] = None
    mode: str
    mode_display: str


class RepoOut(BaseModel):
    id: int
    name: str
    file_count: int
    chunk_count: int
    source_url: Optional[str] = None
    ingested_at: Optional[str] = None


class IngestLocalRequest(BaseModel):
    path: str
    name: Optional[str] = None


class IngestGithubRequest(BaseModel):
    url: str
    branch: Optional[str] = None


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    repo_id: int
    question: str
    history: list[ChatMessage] = []


class BrowseFileOut(BaseModel):
    path: str
    chunk_count: int
    start_line: int
    end_line: int


class BrowseMetricsOut(BaseModel):
    files: int
    chunks: int
    source: str


class BrowseOut(BaseModel):
    metrics: BrowseMetricsOut
    files: list[BrowseFileOut]


class FileChunkOut(BaseModel):
    start_line: int
    end_line: int
    content: str
