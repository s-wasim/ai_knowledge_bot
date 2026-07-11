from fastapi import APIRouter

from app.api.schemas import RepoOut
from app.db import Repo, get_session
from app.ingest.walker import DEFAULT_ALLOWLIST

router = APIRouter()


@router.get("/repos", response_model=list[RepoOut])
def list_repos() -> list[RepoOut]:
    session = get_session()
    repos = session.query(Repo).order_by(Repo.ingested_at.desc()).all()
    return [
        RepoOut(
            id=r.id,
            name=r.name,
            file_count=r.file_count,
            chunk_count=r.chunk_count,
            source_url=r.source_url,
            ingested_at=r.ingested_at.isoformat() if r.ingested_at else None,
        )
        for r in repos
    ]


@router.get("/config/allowlist", response_model=list[str])
def get_allowlist() -> list[str]:
    return sorted(DEFAULT_ALLOWLIST)
