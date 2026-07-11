from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.api.browse import router as browse_router
from app.api.chat import router as chat_router
from app.api.health import router as health_router
from app.api.ingest import router as ingest_router
from app.api.repos import router as repos_router

FRONTEND_DIR = Path(__file__).resolve().parent.parent / "FrontendDesign"

app = FastAPI(title="AI Knowledge Bot")

app.include_router(health_router)
app.include_router(repos_router)
app.include_router(browse_router)
app.include_router(ingest_router)
app.include_router(chat_router)


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(FRONTEND_DIR / "KnowledgeBot.dc.html")


app.mount("/", StaticFiles(directory=str(FRONTEND_DIR)), name="frontend")
