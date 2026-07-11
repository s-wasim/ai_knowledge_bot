from fastapi import APIRouter
from sqlalchemy import func

from app.api.schemas import BrowseFileOut, BrowseMetricsOut, BrowseOut, FileChunkOut
from app.db import Chunk, Repo, get_session

router = APIRouter()


@router.get("/repos/{repo_id}/browse", response_model=BrowseOut)
def browse(repo_id: int, keyword: str = "") -> BrowseOut:
    session = get_session()
    repo = session.query(Repo).filter_by(id=repo_id).first()

    query = session.query(
        Chunk.path,
        func.count(Chunk.id).label("chunk_count"),
        func.min(Chunk.start_line).label("start_line"),
        func.max(Chunk.end_line).label("end_line"),
    ).filter(Chunk.repo_id == repo_id)

    if keyword:
        query = query.filter(Chunk.content.ilike(f"%{keyword}%"))

    file_stats = query.group_by(Chunk.path).order_by(Chunk.path).all()

    return BrowseOut(
        metrics=BrowseMetricsOut(
            files=repo.file_count if repo else 0,
            chunks=repo.chunk_count if repo else 0,
            source="GitHub" if repo and repo.source_url else "Local",
        ),
        files=[
            BrowseFileOut(
                path=row.path,
                chunk_count=row.chunk_count,
                start_line=row.start_line,
                end_line=row.end_line,
            )
            for row in file_stats
        ],
    )


@router.get("/repos/{repo_id}/files", response_model=list[FileChunkOut])
def browse_file(repo_id: int, path: str) -> list[FileChunkOut]:
    session = get_session()
    chunks = (
        session.query(Chunk)
        .filter(Chunk.repo_id == repo_id, Chunk.path == path)
        .order_by(Chunk.start_line)
        .all()
    )
    return [
        FileChunkOut(start_line=c.start_line, end_line=c.end_line, content=c.content)
        for c in chunks
    ]
