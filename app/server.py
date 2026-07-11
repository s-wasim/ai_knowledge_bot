from fastapi import FastAPI

from app.api.health import router as health_router
from app.api.repos import router as repos_router

app = FastAPI(title="AI Knowledge Bot")

app.include_router(health_router)
app.include_router(repos_router)
