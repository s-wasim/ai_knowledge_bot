from fastapi import FastAPI

from app.api.browse import router as browse_router
from app.api.chat import router as chat_router
from app.api.health import router as health_router
from app.api.ingest import router as ingest_router
from app.api.repos import router as repos_router

app = FastAPI(title="AI Knowledge Bot")

app.include_router(health_router)
app.include_router(repos_router)
app.include_router(browse_router)
app.include_router(ingest_router)
app.include_router(chat_router)
