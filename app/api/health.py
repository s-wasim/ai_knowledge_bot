from fastapi import APIRouter
from sqlalchemy import text

from app.api.deps import get_retriever_and_mode
from app.api.schemas import HealthOut
from app.db import get_session
from app.ingest.embedder import embedding_status
from app.retrieval.factory import get_mode_display

router = APIRouter()


@router.get("/health", response_model=HealthOut)
def health() -> HealthOut:
    _retriever, mode = get_retriever_and_mode()

    db_ok = True
    db_error = None
    try:
        get_session().execute(text("SELECT 1"))
    except Exception as e:
        db_ok = False
        db_error = str(e)

    # Reported explicitly so a missing embedding model is visible rather than
    # showing up as unexplained retrieval quality.
    embedding = embedding_status()

    return HealthOut(
        db_ok=db_ok,
        db_error=db_error,
        mode=mode,
        mode_display=get_mode_display(mode),
        embed_model_ok=embedding["ok"],
        embed_model=embedding["model"],
        embed_dims=embedding["dims"],
        embed_error=embedding["error"],
    )
