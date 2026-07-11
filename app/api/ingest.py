# app/api/ingest.py
import json
import queue
import threading
from pathlib import Path
from typing import Callable

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from app.api.schemas import IngestGithubRequest, IngestLocalRequest
from app.ingest.github import ingest_github_url
from app.ingest.pipeline import ingest_repo

router = APIRouter()

_SENTINEL = object()


def _sse_frame(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


def _run_with_progress_sse(target: Callable, source: str):
    """Run `target(progress_callback=...)` on a worker thread, translating its
    synchronous callback calls into SSE frames as they arrive."""
    q: "queue.Queue" = queue.Queue()

    def progress_callback(current, chunk_added, filename):
        q.put(("progress", {"current": current, "filename": filename}))

    def worker():
        try:
            repo = target(progress_callback=progress_callback)
            q.put(
                (
                    "done",
                    {
                        "id": repo.id,
                        "name": repo.name,
                        "file_count": repo.file_count,
                        "chunk_count": repo.chunk_count,
                        "source": source,
                    },
                )
            )
        except Exception as e:
            q.put(("error", {"message": str(e)}))
        finally:
            q.put(_SENTINEL)

    threading.Thread(target=worker, daemon=True).start()

    while True:
        item = q.get()
        if item is _SENTINEL:
            break
        event, data = item
        yield _sse_frame(event, data)


@router.post("/ingest/local")
def ingest_local(body: IngestLocalRequest) -> StreamingResponse:
    path = body.path
    name = body.name or "new-repo"

    def generate():
        root = Path(path) if path else None
        if root is None or not root.exists() or not root.is_dir():
            yield _sse_frame("error", {"message": f"Directory not found: {path}"})
            return
        yield from _run_with_progress_sse(
            lambda progress_callback: ingest_repo(
                repo_name=name, root_dir=str(root), progress_callback=progress_callback
            ),
            source="Local",
        )

    return StreamingResponse(generate(), media_type="text/event-stream")


@router.post("/ingest/github")
def ingest_github(body: IngestGithubRequest) -> StreamingResponse:
    url = body.url
    branch = body.branch or None

    def generate():
        yield from _run_with_progress_sse(
            lambda progress_callback: ingest_github_url(
                url=url, branch=branch, progress_callback=progress_callback
            ),
            source="GitHub",
        )

    return StreamingResponse(generate(), media_type="text/event-stream")
