# FrontendDesign Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the Streamlit UI with a single FastAPI/uvicorn process that serves the `FrontendDesign/` React prototype and the exact REST/SSE API `FrontendDesign/src/api/adapter.js` already documents, calling straight into the existing, unmodified `app/ingest`, `app/graph`, `app/retrieval`, `app/db` modules — deployable as one container on the same port as today.

**Architecture:** New `app/api/` package holds one router module per resource (health, repos, browse, ingest, chat); `app/server.py` wires them into a FastAPI app and, once all routes exist, mounts `FrontendDesign/` as static files with an explicit `/` route for `KnowledgeBot.dc.html`. `app/main.py` and `app/tabs/*.py` are deleted — their orchestration logic moves into the new routers unchanged in substance. `adapter.js`'s mock bodies are replaced with real `fetch`/SSE calls; its exported function signatures and `KnowledgeBot.dc.html`'s `Component` class are untouched.

**Tech Stack:** FastAPI, uvicorn, existing SQLAlchemy/LangGraph/pgvector stack (unchanged), vendored React 18.3.1 / ReactDOM 18.3.1 / Babel Standalone 7.29.0 (unchanged versions, now served locally instead of from unpkg.com).

## Global Constraints

- Do not modify `app/ingest/*.py`, `app/graph/*.py`, `app/retrieval/*.py`, `app/db.py`, `app/highlight.py`, `app/llm.py` — every new route calls these exactly as `app/tabs/*.py` did.
- Do not modify `FrontendDesign/support.js` (generated file) or the `Component` class / template markup inside `FrontendDesign/KnowledgeBot.dc.html` — only its `<head>` gains a small resource-override script.
- Keep port 8501 in `Dockerfile` (`EXPOSE`) and `docker-compose.yml` (`8501:8501`) — only the `CMD` changes.
- Backend is stateless per request: no server-side session store; retriever/graph are process-lifetime singletons.
- Run all commands from the `ai_knowledge_bot/` repo root using `.venv/bin/python` / `.venv/bin/pytest` / `.venv/bin/pip`.

---

### Task 1: Swap Streamlit for FastAPI/uvicorn in dependencies

**Files:**
- Modify: `requirements.txt`

**Interfaces:**
- Produces: `fastapi`, `uvicorn` importable in the venv for all later tasks.

- [ ] **Step 1: Edit requirements.txt**

Remove the `streamlit>=1.38,<2` line and add two lines in its place:

```
fastapi>=0.115,<1
uvicorn[standard]>=0.32,<1
langgraph>=0.2,<1
langchain-anthropic>=0.2,<1
voyageai>=0.2,<1
sqlalchemy>=2.0,<3
psycopg2-binary>=2.9,<3
pgvector>=0.3,<1
httpx>=0.28,<1
pygments>=2.18,<3
anthropic>=0.40,<1
pydantic>=2.0,<3
```

- [ ] **Step 2: Install and verify**

Run: `.venv/bin/pip install -r requirements.txt`
Expected: installs `fastapi` and `uvicorn` with no errors; `streamlit` may remain installed in the venv cache but is no longer a declared dependency.

Run: `.venv/bin/python -c "import fastapi, uvicorn; print(fastapi.__version__, uvicorn.__version__)"`
Expected: prints two version strings, no `ImportError`.

- [ ] **Step 3: Commit**

```bash
git add requirements.txt
git commit -m "build: swap streamlit for fastapi+uvicorn"
```

---

### Task 2: Process-lifetime retriever/graph singletons

**Files:**
- Create: `app/api/__init__.py`
- Create: `app/api/deps.py`
- Test: `tests/test_api_deps.py`

**Interfaces:**
- Produces: `deps.get_retriever_and_mode() -> tuple[Retriever, str]`, `deps.get_graph() -> CompiledGraph`, `deps.reset_singletons() -> None` (test-only cache reset). Consumed by every later route module.

- [ ] **Step 1: Create the empty package marker**

```python
# app/api/__init__.py
```

- [ ] **Step 2: Write the failing test**

```python
# tests/test_api_deps.py
from app.api import deps


class TestGetRetrieverAndMode:
    def setup_method(self):
        deps.reset_singletons()

    def test_returns_fts_retriever_when_no_voyage_key(self, monkeypatch):
        monkeypatch.delenv("VOYAGE_API_KEY", raising=False)
        retriever, mode = deps.get_retriever_and_mode()
        assert mode == "fts"
        assert retriever is not None

    def test_caches_singleton_across_calls(self, monkeypatch):
        monkeypatch.delenv("VOYAGE_API_KEY", raising=False)
        retriever_1, _ = deps.get_retriever_and_mode()
        retriever_2, _ = deps.get_retriever_and_mode()
        assert retriever_1 is retriever_2


class TestGetGraph:
    def setup_method(self):
        deps.reset_singletons()

    def test_returns_compiled_graph(self):
        graph = deps.get_graph()
        assert hasattr(graph, "invoke")
        assert hasattr(graph, "stream")

    def test_caches_singleton_across_calls(self):
        graph_1 = deps.get_graph()
        graph_2 = deps.get_graph()
        assert graph_1 is graph_2
```

- [ ] **Step 3: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_api_deps.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.api.deps'`

- [ ] **Step 4: Write the implementation**

```python
# app/api/deps.py
import threading

from app.db import get_session
from app.graph.build import build_rag_graph
from app.retrieval.factory import create_retriever

_lock = threading.Lock()
_retriever = None
_retrieval_mode = None
_graph = None


def get_retriever_and_mode():
    global _retriever, _retrieval_mode
    if _retriever is None:
        with _lock:
            if _retriever is None:
                _retriever, _retrieval_mode = create_retriever(get_session)
    return _retriever, _retrieval_mode


def get_graph():
    global _graph
    if _graph is None:
        with _lock:
            if _graph is None:
                _graph = build_rag_graph()
    return _graph


def reset_singletons():
    """Test-only: clear cached singletons so tests can rebuild them fresh."""
    global _retriever, _retrieval_mode, _graph
    _retriever = None
    _retrieval_mode = None
    _graph = None
```

- [ ] **Step 5: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_api_deps.py -v`
Expected: 4 passed

- [ ] **Step 6: Commit**

```bash
git add app/api/__init__.py app/api/deps.py tests/test_api_deps.py
git commit -m "feat: add process-lifetime retriever/graph singletons for API layer"
```

---

### Task 3: Pydantic request/response schemas

**Files:**
- Create: `app/api/schemas.py`
- Test: `tests/test_api_schemas.py`

**Interfaces:**
- Produces: `HealthOut`, `RepoOut`, `IngestLocalRequest`, `IngestGithubRequest`, `ChatMessage`, `ChatRequest`, `BrowseFileOut`, `BrowseMetricsOut`, `BrowseOut`, `FileChunkOut` — all Pydantic `BaseModel` subclasses, consumed by every route task below.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_api_schemas.py
from app.api.schemas import ChatMessage, ChatRequest, HealthOut, RepoOut


def test_health_out_defaults_error_to_none():
    health = HealthOut(db_ok=True, mode="fts", mode_display="Full-text search")
    assert health.db_error is None


def test_repo_out_optional_fields_default_to_none():
    repo = RepoOut(id=1, name="x", file_count=0, chunk_count=0)
    assert repo.source_url is None
    assert repo.ingested_at is None


def test_chat_request_history_defaults_to_empty_list():
    req = ChatRequest(repo_id=1, question="hi?")
    assert req.history == []


def test_chat_request_parses_history_dicts_into_chat_message():
    req = ChatRequest(
        repo_id=1, question="hi?", history=[{"role": "user", "content": "x"}]
    )
    assert isinstance(req.history[0], ChatMessage)
    assert req.history[0].role == "user"
    assert req.history[0].content == "x"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_api_schemas.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.api.schemas'`

- [ ] **Step 3: Write the implementation**

```python
# app/api/schemas.py
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_api_schemas.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add app/api/schemas.py tests/test_api_schemas.py
git commit -m "feat: add Pydantic schemas for the HTTP API layer"
```

---

### Task 4: FastAPI app skeleton with `/health` and `/repos`

**Files:**
- Create: `app/api/health.py`
- Create: `app/api/repos.py`
- Create: `app/server.py`
- Test: `tests/test_api_health.py`
- Test: `tests/test_api_repos.py`

**Interfaces:**
- Consumes: `deps.get_retriever_and_mode()` (Task 2), `HealthOut`/`RepoOut` (Task 3), `app.db.get_session`, `app.db.Repo`, `app.retrieval.factory.get_mode_display`, `app.ingest.walker.DEFAULT_ALLOWLIST`.
- Produces: `app.server.app` (the FastAPI instance every later task imports and extends), `health.router`, `repos.router`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_api_health.py
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from app.api import deps
from app.server import app


class TestHealth:
    def setup_method(self):
        deps.reset_singletons()

    @patch("app.api.health.get_session")
    def test_health_ok(self, mock_get_session, monkeypatch):
        monkeypatch.delenv("VOYAGE_API_KEY", raising=False)
        mock_get_session.return_value = MagicMock()

        client = TestClient(app)
        res = client.get("/health")

        assert res.status_code == 200
        body = res.json()
        assert body["db_ok"] is True
        assert body["db_error"] is None
        assert body["mode"] == "fts"
        assert body["mode_display"] == "Full-text search"

    @patch("app.api.health.get_session")
    def test_health_reports_db_error(self, mock_get_session, monkeypatch):
        monkeypatch.delenv("VOYAGE_API_KEY", raising=False)
        broken_session = MagicMock()
        broken_session.execute.side_effect = RuntimeError("connection refused")
        mock_get_session.return_value = broken_session

        client = TestClient(app)
        res = client.get("/health")

        assert res.status_code == 200
        body = res.json()
        assert body["db_ok"] is False
        assert "connection refused" in body["db_error"]
```

```python
# tests/test_api_repos.py
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from app.server import app


class TestListRepos:
    @patch("app.api.repos.get_session")
    def test_returns_repos_sorted_newest_first(self, mock_get_session):
        repo = MagicMock()
        repo.id = 1
        repo.name = "my_repo"
        repo.file_count = 5
        repo.chunk_count = 40
        repo.source_url = None
        repo.ingested_at = datetime(2026, 7, 10, 9, 0, tzinfo=timezone.utc)

        mock_session = MagicMock()
        mock_session.query.return_value.order_by.return_value.all.return_value = [repo]
        mock_get_session.return_value = mock_session

        client = TestClient(app)
        res = client.get("/repos")

        assert res.status_code == 200
        body = res.json()
        assert body[0]["id"] == 1
        assert body[0]["name"] == "my_repo"
        assert body[0]["ingested_at"].startswith("2026-07-10T09:00:00")


class TestAllowlist:
    def test_returns_sorted_allowlist(self):
        client = TestClient(app)
        res = client.get("/config/allowlist")

        assert res.status_code == 200
        body = res.json()
        assert body == sorted(body)
        assert ".py" in body
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_api_health.py tests/test_api_repos.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.server'`

- [ ] **Step 3: Write `app/api/health.py`**

```python
# app/api/health.py
from fastapi import APIRouter
from sqlalchemy import text

from app.api.deps import get_retriever_and_mode
from app.api.schemas import HealthOut
from app.db import get_session
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

    return HealthOut(
        db_ok=db_ok,
        db_error=db_error,
        mode=mode,
        mode_display=get_mode_display(mode),
    )
```

- [ ] **Step 4: Write `app/api/repos.py`**

```python
# app/api/repos.py
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
```

- [ ] **Step 5: Write `app/server.py`**

```python
# app/server.py
from fastapi import FastAPI

from app.api.health import router as health_router
from app.api.repos import router as repos_router

app = FastAPI(title="AI Knowledge Bot")

app.include_router(health_router)
app.include_router(repos_router)
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_api_health.py tests/test_api_repos.py -v`
Expected: 4 passed

- [ ] **Step 7: Commit**

```bash
git add app/api/health.py app/api/repos.py app/server.py tests/test_api_health.py tests/test_api_repos.py
git commit -m "feat: add FastAPI app skeleton with health, repos, and allowlist endpoints"
```

---

### Task 5: `/repos/{id}/browse` and `/repos/{id}/files`

**Files:**
- Create: `app/api/browse.py`
- Modify: `app/server.py`
- Test: `tests/test_api_browse.py`

**Interfaces:**
- Consumes: `BrowseFileOut`/`BrowseMetricsOut`/`BrowseOut`/`FileChunkOut` (Task 3), `app.db.get_session`, `app.db.Repo`, `app.db.Chunk`.
- Produces: `browse.router`, mirroring `app/tabs/browser.py`'s query logic exactly.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_api_browse.py
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from app.server import app


class FakeRow:
    def __init__(self, path, chunk_count, start_line, end_line):
        self.path = path
        self.chunk_count = chunk_count
        self.start_line = start_line
        self.end_line = end_line


class TestBrowse:
    @patch("app.api.browse.get_session")
    def test_returns_metrics_and_files_for_existing_repo(self, mock_get_session):
        repo = MagicMock()
        repo.file_count = 3
        repo.chunk_count = 10
        repo.source_url = "https://github.com/acme/repo"

        rows = [FakeRow("app/main.py", 2, 1, 20)]

        mock_session = MagicMock()
        mock_session.query.return_value.filter_by.return_value.first.return_value = repo
        query_chain = mock_session.query.return_value
        query_chain.filter.return_value.group_by.return_value.order_by.return_value.all.return_value = rows
        mock_get_session.return_value = mock_session

        client = TestClient(app)
        res = client.get("/repos/1/browse")

        assert res.status_code == 200
        body = res.json()
        assert body["metrics"]["files"] == 3
        assert body["metrics"]["chunks"] == 10
        assert body["metrics"]["source"] == "GitHub"
        assert body["files"][0]["path"] == "app/main.py"

    @patch("app.api.browse.get_session")
    def test_missing_repo_returns_zeroed_metrics(self, mock_get_session):
        mock_session = MagicMock()
        mock_session.query.return_value.filter_by.return_value.first.return_value = None
        query_chain = mock_session.query.return_value
        query_chain.filter.return_value.group_by.return_value.order_by.return_value.all.return_value = []
        mock_get_session.return_value = mock_session

        client = TestClient(app)
        res = client.get("/repos/999/browse")

        assert res.status_code == 200
        body = res.json()
        assert body["metrics"]["files"] == 0
        assert body["metrics"]["source"] == "Local"
        assert body["files"] == []


class TestBrowseFile:
    @patch("app.api.browse.get_session")
    def test_returns_chunks_ordered_by_start_line(self, mock_get_session):
        chunk = MagicMock()
        chunk.start_line = 1
        chunk.end_line = 10
        chunk.content = "print('hi')"

        mock_session = MagicMock()
        query_chain = mock_session.query.return_value
        query_chain.filter.return_value.order_by.return_value.all.return_value = [chunk]
        mock_get_session.return_value = mock_session

        client = TestClient(app)
        res = client.get("/repos/1/files", params={"path": "app/main.py"})

        assert res.status_code == 200
        body = res.json()
        assert body[0]["start_line"] == 1
        assert body[0]["content"] == "print('hi')"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_api_browse.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.api.browse'`

- [ ] **Step 3: Write `app/api/browse.py`**

```python
# app/api/browse.py
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
```

- [ ] **Step 4: Modify `app/server.py`**

```python
# app/server.py
from fastapi import FastAPI

from app.api.browse import router as browse_router
from app.api.health import router as health_router
from app.api.repos import router as repos_router

app = FastAPI(title="AI Knowledge Bot")

app.include_router(health_router)
app.include_router(repos_router)
app.include_router(browse_router)
```

- [ ] **Step 5: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_api_browse.py -v`
Expected: 3 passed

- [ ] **Step 6: Commit**

```bash
git add app/api/browse.py app/server.py tests/test_api_browse.py
git commit -m "feat: add repo browse and file endpoints"
```

---

### Task 6: `/ingest/local` and `/ingest/github` (SSE)

**Files:**
- Create: `app/api/ingest.py`
- Modify: `app/server.py`
- Test: `tests/test_api_ingest.py`

**Interfaces:**
- Consumes: `IngestLocalRequest`/`IngestGithubRequest` (Task 3), `app.ingest.pipeline.ingest_repo`, `app.ingest.github.ingest_github_url`.
- Produces: `ingest.router`; a `text/event-stream` body with `progress`/`done`/`error` frames matching `adapter.js`'s documented `POST /ingest/local` and `POST /ingest/github` contracts exactly.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_api_ingest.py
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from app.server import app


class TestIngestLocal:
    def test_missing_directory_emits_error_event(self, tmp_path):
        client = TestClient(app)
        res = client.post(
            "/ingest/local",
            json={"path": str(tmp_path / "does_not_exist"), "name": "x"},
        )

        assert res.status_code == 200
        assert "event: error" in res.text
        assert "Directory not found" in res.text

    @patch("app.api.ingest.ingest_repo")
    def test_success_emits_progress_then_done(self, mock_ingest_repo, tmp_path):
        def fake_ingest(repo_name, root_dir, progress_callback=None):
            progress_callback(1, 3, "a.py")
            progress_callback(2, 2, "b.py")
            repo = MagicMock()
            repo.id = 7
            repo.name = repo_name
            repo.file_count = 2
            repo.chunk_count = 5
            return repo

        mock_ingest_repo.side_effect = fake_ingest

        client = TestClient(app)
        res = client.post(
            "/ingest/local", json={"path": str(tmp_path), "name": "my_repo"}
        )

        assert res.status_code == 200
        assert res.text.count("event: progress") == 2
        assert "event: done" in res.text
        assert '"name": "my_repo"' in res.text
        assert '"source": "Local"' in res.text

    @patch("app.api.ingest.ingest_repo")
    def test_exception_emits_error_event(self, mock_ingest_repo, tmp_path):
        mock_ingest_repo.side_effect = RuntimeError("boom")

        client = TestClient(app)
        res = client.post(
            "/ingest/local", json={"path": str(tmp_path), "name": "x"}
        )

        assert res.status_code == 200
        assert "event: error" in res.text
        assert "boom" in res.text


class TestIngestGithub:
    @patch("app.api.ingest.ingest_github_url")
    def test_success_emits_done_with_github_source(self, mock_ingest_github_url):
        def fake_ingest(url, branch=None, progress_callback=None):
            progress_callback(1, 1, "README.md")
            repo = MagicMock()
            repo.id = 9
            repo.name = "repo"
            repo.file_count = 1
            repo.chunk_count = 1
            return repo

        mock_ingest_github_url.side_effect = fake_ingest

        client = TestClient(app)
        res = client.post(
            "/ingest/github", json={"url": "https://github.com/acme/repo"}
        )

        assert res.status_code == 200
        assert "event: progress" in res.text
        assert "event: done" in res.text
        assert '"source": "GitHub"' in res.text

    @patch("app.api.ingest.ingest_github_url")
    def test_invalid_url_emits_error_event(self, mock_ingest_github_url):
        mock_ingest_github_url.side_effect = ValueError(
            "Invalid GitHub URL: not-a-url. Expected https://github.com/owner/repo"
        )

        client = TestClient(app)
        res = client.post("/ingest/github", json={"url": "not-a-url"})

        assert res.status_code == 200
        assert "event: error" in res.text
        assert "Invalid GitHub URL" in res.text
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_api_ingest.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.api.ingest'`

- [ ] **Step 3: Write `app/api/ingest.py`**

```python
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
```

- [ ] **Step 4: Modify `app/server.py`**

```python
# app/server.py
from fastapi import FastAPI

from app.api.browse import router as browse_router
from app.api.health import router as health_router
from app.api.ingest import router as ingest_router
from app.api.repos import router as repos_router

app = FastAPI(title="AI Knowledge Bot")

app.include_router(health_router)
app.include_router(repos_router)
app.include_router(browse_router)
app.include_router(ingest_router)
```

- [ ] **Step 5: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_api_ingest.py -v`
Expected: 5 passed

- [ ] **Step 6: Commit**

```bash
git add app/api/ingest.py app/server.py tests/test_api_ingest.py
git commit -m "feat: add SSE ingest endpoints for local folders and GitHub URLs"
```

---

### Task 7: `/chat` (SSE)

**Files:**
- Create: `app/api/chat.py`
- Modify: `app/server.py`
- Test: `tests/test_api_chat.py`

**Interfaces:**
- Consumes: `ChatRequest` (Task 3), `deps.get_graph()`/`deps.get_retriever_and_mode()` (Task 2), `app.db.get_session`, `app.graph.state.RagState`.
- Produces: `chat.router`; a `text/event-stream` body with `node`/`token`/`final`/`error` frames matching `adapter.js`'s documented `POST /chat` contract, reusing the exact node-completion / token-streaming mechanism `app/tabs/chat.py`'s `_run_graph_streamed` already used.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_api_chat.py
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from app.graph.state import Citation, GradedChunk
from app.retrieval.base import ChunkData
from app.server import app


def _fake_stream(state, config, stream_mode):
    chunk = ChunkData(path="app/db.py", start_line=1, end_line=10, content="engine = ...", score=0.9)
    graded = GradedChunk(chunk=chunk, keep=True, reason="relevant")
    citation = Citation(chunk=chunk, index=1)

    yield "updates", {"rewrite_query": {"rewritten_query": state["question"]}}
    yield "updates", {"retrieve": {"retrieved": [chunk]}}
    yield "updates", {"grade_chunks": {"graded": [graded]}}
    yield "messages", (SimpleNamespace(content="The "), {"langgraph_node": "generate_answer"})
    yield "messages", (SimpleNamespace(content="answer."), {"langgraph_node": "generate_answer"})
    yield "updates", {"generate_answer": {"answer": "The answer.", "citations": [citation]}}


class TestChat:
    @patch("app.api.chat.get_graph")
    @patch("app.api.chat.get_retriever_and_mode")
    def test_streams_node_token_and_final_events(
        self, mock_get_retriever_and_mode, mock_get_graph
    ):
        mock_get_retriever_and_mode.return_value = (MagicMock(), "fts")
        fake_graph = MagicMock()
        fake_graph.stream.side_effect = _fake_stream
        mock_get_graph.return_value = fake_graph

        client = TestClient(app)
        res = client.post(
            "/chat",
            json={"repo_id": 1, "question": "Where is the DB configured?", "history": []},
        )

        assert res.status_code == 200
        text = res.text
        assert "event: node" in text
        assert '"node": "rewrite_query"' in text
        assert '"node": "generate_answer"' in text
        assert "event: token" in text
        assert '"text": "The "' in text
        assert "event: final" in text
        assert '"answer": "The answer."' in text
        assert '"path": "app/db.py"' in text

    @patch("app.api.chat.get_graph")
    @patch("app.api.chat.get_retriever_and_mode")
    def test_graph_exception_emits_error_event(
        self, mock_get_retriever_and_mode, mock_get_graph
    ):
        mock_get_retriever_and_mode.return_value = (MagicMock(), "fts")
        fake_graph = MagicMock()
        fake_graph.stream.side_effect = RuntimeError("llm unavailable")
        mock_get_graph.return_value = fake_graph

        client = TestClient(app)
        res = client.post(
            "/chat", json={"repo_id": 1, "question": "hi?", "history": []}
        )

        assert res.status_code == 200
        assert "event: error" in res.text
        assert "llm unavailable" in res.text
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_api_chat.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.api.chat'`

- [ ] **Step 3: Write `app/api/chat.py`**

```python
# app/api/chat.py
import json

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from app.api.deps import get_graph, get_retriever_and_mode
from app.api.schemas import ChatRequest
from app.db import get_session
from app.graph.state import RagState

router = APIRouter()


def _sse_frame(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


def _citation_dict(citation) -> dict:
    return {
        "path": citation.chunk.path,
        "start_line": citation.chunk.start_line,
        "end_line": citation.chunk.end_line,
        "content": citation.chunk.content,
    }


def _graded_dict(graded_chunk) -> dict:
    return {
        "path": graded_chunk.chunk.path,
        "start_line": graded_chunk.chunk.start_line,
        "end_line": graded_chunk.chunk.end_line,
        "content": graded_chunk.chunk.content,
        "keep": graded_chunk.keep,
        "reason": graded_chunk.reason,
        "score": graded_chunk.chunk.score,
    }


def _retrieved_dict(chunk) -> dict:
    return {
        "path": chunk.path,
        "start_line": chunk.start_line,
        "end_line": chunk.end_line,
        "content": chunk.content,
        "score": chunk.score,
    }


@router.post("/chat")
def chat(body: ChatRequest) -> StreamingResponse:
    retriever, mode = get_retriever_and_mode()
    graph = get_graph()

    history = [m.model_dump() for m in body.history[-6:]]

    state: RagState = {
        "question": body.question,
        "chat_history": history,
        "rewritten_query": None,
        "retrieved": [],
        "graded": [],
        "answer": None,
        "citations": [],
        "mode": mode,
        "repo_id": body.repo_id,
    }
    config = {"configurable": {"retriever": retriever, "get_session": get_session}}

    def generate():
        final_state: dict = {}
        try:
            for stream_mode, payload in graph.stream(
                state, config=config, stream_mode=["updates", "messages"]
            ):
                if stream_mode == "updates":
                    for node_name, node_output in payload.items():
                        yield _sse_frame("node", {"node": node_name})
                        final_state.update(node_output)
                elif stream_mode == "messages":
                    message_chunk, metadata = payload
                    if metadata.get("langgraph_node") == "generate_answer" and message_chunk.content:
                        yield _sse_frame("token", {"text": message_chunk.content})

            answer = final_state.get("answer") or "No answer generated."
            citations = final_state.get("citations", [])
            graded = final_state.get("graded", [])
            retrieved = final_state.get("retrieved", [])

            yield _sse_frame(
                "final",
                {
                    "answer": answer,
                    "citations": [_citation_dict(c) for c in citations],
                    "graded": [_graded_dict(gc) for gc in graded],
                    "retrieved": [_retrieved_dict(c) for c in retrieved],
                },
            )
        except Exception as e:
            yield _sse_frame("error", {"message": str(e)})

    return StreamingResponse(generate(), media_type="text/event-stream")
```

- [ ] **Step 4: Modify `app/server.py`**

```python
# app/server.py
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
```

- [ ] **Step 5: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_api_chat.py -v`
Expected: 2 passed

- [ ] **Step 6: Commit**

```bash
git add app/api/chat.py app/server.py tests/test_api_chat.py
git commit -m "feat: add SSE chat endpoint streaming node/token/final events"
```

---

### Task 8: Serve `FrontendDesign/` as static files

**Files:**
- Modify: `app/server.py`
- Test: `tests/test_api_static.py`

**Interfaces:**
- Consumes: nothing new.
- Produces: `GET /` returns `KnowledgeBot.dc.html`; `GET /support.js`, `GET /src/api/adapter.js` served from `FrontendDesign/` via `StaticFiles`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_api_static.py
from fastapi.testclient import TestClient

from app.server import app


class TestStaticFrontend:
    def test_root_serves_the_dc_html_entrypoint(self):
        client = TestClient(app)
        res = client.get("/")

        assert res.status_code == 200
        assert "<x-dc>" in res.text

    def test_support_js_is_served(self):
        client = TestClient(app)
        res = client.get("/support.js")

        assert res.status_code == 200
        assert "dc-runtime" in res.text

    def test_adapter_js_is_served(self):
        client = TestClient(app)
        res = client.get("/src/api/adapter.js")

        assert res.status_code == 200

    def test_api_routes_still_resolve_alongside_static_mount(self):
        client = TestClient(app)
        res = client.get("/config/allowlist")

        assert res.status_code == 200
        assert isinstance(res.json(), list)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_api_static.py -v`
Expected: FAIL — `test_root_serves_the_dc_html_entrypoint` gets a 404 (no `/` route registered yet)

- [ ] **Step 3: Modify `app/server.py`**

```python
# app/server.py
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_api_static.py -v`
Expected: 4 passed

- [ ] **Step 5: Run the full test suite to confirm nothing regressed**

Run: `.venv/bin/pytest -v`
Expected: all tests pass (existing backend tests + all `test_api_*` tests added so far)

- [ ] **Step 6: Commit**

```bash
git add app/server.py tests/test_api_static.py
git commit -m "feat: serve FrontendDesign as static files from the FastAPI app"
```

---

### Task 9: Vendor React/ReactDOM/Babel locally

**Files:**
- Create: `FrontendDesign/vendor/react.production.min.js`
- Create: `FrontendDesign/vendor/react-dom.production.min.js`
- Create: `FrontendDesign/vendor/babel.min.js`
- Modify: `FrontendDesign/KnowledgeBot.dc.html`

**Interfaces:**
- Produces: `window.__resources` map read by `support.js`'s existing `cdnScriptFor()` (already reads this global before falling back to unpkg.com — no change to `support.js` itself).

- [ ] **Step 1: Download the pinned versions `support.js` already references**

```bash
mkdir -p FrontendDesign/vendor
curl -sL -o FrontendDesign/vendor/react.production.min.js \
  https://unpkg.com/react@18.3.1/umd/react.production.min.js
curl -sL -o FrontendDesign/vendor/react-dom.production.min.js \
  https://unpkg.com/react-dom@18.3.1/umd/react-dom.production.min.js
curl -sL -o FrontendDesign/vendor/babel.min.js \
  https://unpkg.com/@babel/standalone@7.29.0/babel.min.js
```

- [ ] **Step 2: Verify each download matches the SRI hash `support.js` expects**

```bash
openssl dgst -sha384 -binary FrontendDesign/vendor/react.production.min.js | openssl base64 -A
```
Expected: `DGyLxAyjq0f9SPpVevD6IgztCFlnMF6oW/XQGmfe+IsZ8TqEiDrcHkMLKI6fiB/Z` (matches `REACT_SRI` in `support.js:1074`)

```bash
openssl dgst -sha384 -binary FrontendDesign/vendor/react-dom.production.min.js | openssl base64 -A
```
Expected: `gTGxhz21lVGYNMcdJOyq01Edg0jhn/c22nsx0kyqP0TxaV5WVdsSH1fSDUf5YJj1` (matches `REACT_DOM_SRI` in `support.js:1076`)

```bash
openssl dgst -sha384 -binary FrontendDesign/vendor/babel.min.js | openssl base64 -A
```
Expected: `m08KidiNqLdpJqLq95G/LEi8Qvjl/xUYll3QILypMoQ65QorJ9Lvtp2RXYGBFj1y` (matches `BABEL_SRI` in `support.js:1078`)

If any hash doesn't match, do not proceed — it means unpkg served a different build than the version pinned in `support.js`; stop and re-check the URL/version instead of editing the hash check.

- [ ] **Step 3: Point `support.js`'s resource resolver at the vendored copies**

In `FrontendDesign/KnowledgeBot.dc.html`, replace line 6:

```html
<script src="./support.js"></script>
```

with:

```html
<script>
  window.__resources = {
    "https://unpkg.com/react@18.3.1/umd/react.production.min.js": "./vendor/react.production.min.js",
    "https://unpkg.com/react-dom@18.3.1/umd/react-dom.production.min.js": "./vendor/react-dom.production.min.js",
    "https://unpkg.com/@babel/standalone@7.29.0/babel.min.js": "./vendor/babel.min.js"
  };
</script>
<script src="./support.js"></script>
```

- [ ] **Step 4: Verify the override map is in place and the vendor files are served**

Run: `grep -n "window.__resources" FrontendDesign/KnowledgeBot.dc.html`
Expected: prints the line just added

Run: `.venv/bin/pytest tests/test_api_static.py -v`
Expected: still 4 passed (static mount serves the new `vendor/` files automatically since it serves the whole `FrontendDesign/` directory)

- [ ] **Step 5: Commit**

```bash
git add FrontendDesign/vendor FrontendDesign/KnowledgeBot.dc.html
git commit -m "feat: vendor React/ReactDOM/Babel locally for offline-capable frontend"
```

---

### Task 10: Replace `adapter.js`'s mock bodies with live calls

**Files:**
- Modify: `FrontendDesign/src/api/adapter.js` (full rewrite — the mock seed data, `detectIntent`/`CANNED` canned answers, and `runIngestSimulation` are all deleted; every exported function's *signature* is unchanged)
- Test: `tests/test_api_static.py` (extend)

**Interfaces:**
- Consumes: every endpoint built in Tasks 4–7.
- Produces: no change to the calling contract `KnowledgeBot.dc.html`'s `Component` class relies on — same exported function names, same parameter shapes, same callback names.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_api_static.py`:

```python
class TestAdapterIsLive:
    def test_adapter_no_longer_ships_mock_data(self):
        client = TestClient(app)
        res = client.get("/src/api/adapter.js")

        assert res.status_code == 200
        assert "MOCK MODE" not in res.text
        assert "fetch(" in res.text
```

Run: `.venv/bin/pytest tests/test_api_static.py::TestAdapterIsLive -v`
Expected: FAIL (current file still contains `"MOCK MODE"` in its header comment)

- [ ] **Step 2: Replace `FrontendDesign/src/api/adapter.js` in full**

```javascript
// =============================================================================
// VeloRelAI Knowledge Bot — API Adapter
// -----------------------------------------------------------------------------
// Single point of contact for all backend data. Every UI component calls into
// this module only — never fetch() directly from a component.
//
// LIVE MODE: calls the FastAPI backend served from the same origin as this
// file. Exported function names, parameters, and callback shapes are
// unchanged from the original mock so no caller-side code needs to change.
// =============================================================================

export const API_BASE = window.location.origin;

// -----------------------------------------------------------------------------
// SSE helper — POST a JSON body, parse a `text/event-stream` response as it
// arrives, and dispatch parsed frames to `on<EventName>` handlers. Returns a
// cancel function that aborts the in-flight request.
// -----------------------------------------------------------------------------
function sseFetch(url, body, handlers) {
  const controller = new AbortController();

  (async () => {
    try {
      const res = await fetch(url, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
        signal: controller.signal,
      });

      if (!res.ok || !res.body) {
        const text = await res.text().catch(() => "");
        handlers.onError({ message: text || `HTTP ${res.status}` });
        return;
      }

      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buf = "";

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        buf += decoder.decode(value, { stream: true });

        let idx;
        while ((idx = buf.indexOf("\n\n")) !== -1) {
          const frame = buf.slice(0, idx);
          buf = buf.slice(idx + 2);

          const eventMatch = /^event:\s*(.+)$/m.exec(frame);
          const dataMatch = /^data:\s*(.+)$/m.exec(frame);
          if (!eventMatch || !dataMatch) continue;

          const eventName = eventMatch[1].trim();
          let data;
          try {
            data = JSON.parse(dataMatch[1]);
          } catch {
            continue;
          }

          const handlerName = "on" + eventName[0].toUpperCase() + eventName.slice(1);
          const handler = handlers[handlerName];
          if (handler) handler(data);
        }
      }
    } catch (e) {
      if (e.name !== "AbortError") {
        handlers.onError({ message: e.message || String(e) });
      }
    }
  })();

  return () => controller.abort();
}

// =============================================================================
// GET /health
// =============================================================================
export async function getHealth() {
  const res = await fetch(`${API_BASE}/health`);
  return res.json();
}

// =============================================================================
// GET /repos
// =============================================================================
export async function listRepos() {
  const res = await fetch(`${API_BASE}/repos`);
  return res.json();
}

// =============================================================================
// POST /ingest/local  (Server-Sent Events)
// =============================================================================
export function ingestLocal({ path, name }, onProgress, onDone, onError) {
  return sseFetch(
    `${API_BASE}/ingest/local`,
    { path, name },
    { onProgress, onDone, onError }
  );
}

// =============================================================================
// POST /ingest/github  (Server-Sent Events)
// =============================================================================
export function ingestGithub({ url, branch }, onProgress, onDone, onError) {
  return sseFetch(
    `${API_BASE}/ingest/github`,
    { url, branch },
    { onProgress, onDone, onError }
  );
}

// =============================================================================
// POST /chat  (Server-Sent Events)
// =============================================================================
export function chat({ repo_id, question, history }, { onNode, onToken, onFinal, onError }) {
  const trimmedHistory = (history || []).slice(-6);
  return sseFetch(
    `${API_BASE}/chat`,
    { repo_id, question, history: trimmedHistory },
    { onNode, onToken, onFinal, onError }
  );
}

// =============================================================================
// GET /repos/{repo_id}/browse?keyword=
// =============================================================================
export async function browse(repo_id, keyword) {
  const res = await fetch(
    `${API_BASE}/repos/${repo_id}/browse?keyword=${encodeURIComponent(keyword || "")}`
  );
  return res.json();
}

// =============================================================================
// GET /repos/{repo_id}/files?path=
// =============================================================================
export async function browseFile(repo_id, path) {
  const res = await fetch(
    `${API_BASE}/repos/${repo_id}/files?path=${encodeURIComponent(path)}`
  );
  return res.json();
}

// =============================================================================
// GET /config/allowlist
// =============================================================================
export async function getAllowlist() {
  const res = await fetch(`${API_BASE}/config/allowlist`);
  return res.json();
}
```

- [ ] **Step 3: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_api_static.py -v`
Expected: 5 passed

- [ ] **Step 4: Commit**

```bash
git add FrontendDesign/src/api/adapter.js tests/test_api_static.py
git commit -m "feat: replace adapter.js mock data with live fetch/SSE calls to the FastAPI backend"
```

---

### Task 11: Remove Streamlit and update the Docker entrypoint

**Files:**
- Delete: `app/main.py`
- Delete: `app/tabs/__init__.py`
- Delete: `app/tabs/ingest.py`
- Delete: `app/tabs/chat.py`
- Delete: `app/tabs/browser.py`
- Modify: `Dockerfile`

**Interfaces:** none — this task removes now-dead code and repoints the container's process manager at `app.server:app`.

- [ ] **Step 1: Delete the Streamlit UI files**

```bash
git rm app/main.py app/tabs/__init__.py app/tabs/ingest.py app/tabs/chat.py app/tabs/browser.py
```

- [ ] **Step 2: Update `Dockerfile`**

Replace the final `CMD` line:

```dockerfile
CMD ["streamlit", "run", "app/main.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

with:

```dockerfile
CMD ["uvicorn", "app.server:app", "--host", "0.0.0.0", "--port", "8501"]
```

`EXPOSE 8501` and every other line stay unchanged.

- [ ] **Step 3: Verify the app still imports cleanly and the full suite passes**

Run: `.venv/bin/python -c "from app.server import app; print(app.title)"`
Expected: `AI Knowledge Bot`

Run: `.venv/bin/pytest -v`
Expected: all tests pass (no `test_main.py`/`test_tabs_*.py` existed before this change, so nothing is lost)

- [ ] **Step 4: Commit**

```bash
git add Dockerfile
git commit -m "chore: remove Streamlit UI, run the app via uvicorn app.server:app"
```

---

### Task 12: End-to-end manual verification

**Files:** none — this task only runs and observes the built app.

- [ ] **Step 1: Build and start the full stack**

```bash
docker compose up --build
```

Expected: `app` and `db` services start; `app` logs show uvicorn listening on `0.0.0.0:8501` with no traceback.

- [ ] **Step 2: Verify the frontend loads**

Open `http://localhost:8501` in a browser.
Expected: the VeloRelAI splash animation plays, then the Ingest panel renders with the mode badge showing "Full-text search" (or "Vector (voyage-code-3)" if `VOYAGE_API_KEY` is set in `.env`).

- [ ] **Step 3: Verify ingest works end-to-end**

In the Ingest panel, enter a path under `./repos` or `./fixtures` (mounted volumes per `docker-compose.yml`) and click Ingest.
Expected: progress ticks through filenames in real time, then a success state showing file/chunk counts — sourced from the real `/ingest/local` SSE stream, not simulated data.

- [ ] **Step 4: Verify chat works end-to-end**

Switch to the Chat panel, select the ingested repo, ask one of the quick-ask questions.
Expected: the pipeline stage indicator advances through rewrite → retrieve → grade → answer live, the answer streams token-by-token, and citations/graded chunks shown below are real chunks from the ingested repo (not the canned FastAPI-backend answers from the old mock).

- [ ] **Step 5: Verify the index browser works end-to-end**

Switch to the Index Browser panel, select the repo, expand a file.
Expected: real chunk content and line ranges from the database, matching what was ingested in Step 3.

- [ ] **Step 6: Tear down**

```bash
docker compose down
```
