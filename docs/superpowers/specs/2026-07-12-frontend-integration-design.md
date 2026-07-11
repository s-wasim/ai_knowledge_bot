# FrontendDesign Integration Design

**Date**: 2026-07-12
**Status**: Approved

## Goal

Wire the React prototype in `FrontendDesign/` (`KnowledgeBot.dc.html` + `support.js` + `src/api/adapter.js`) up to the real backend, replacing its mock/dummy data with live calls into the existing, unchanged Python modules — and make the result deployable as a single monolithic container, same as today.

## Decisions

- Replace Streamlit as the UI entirely. Delete `app/main.py` and `app/tabs/*.py`; their orchestration logic (which calls to make, in what order) moves into new API route handlers, unchanged in substance.
- New single-process FastAPI/uvicorn app (`app/server.py`) serves both the static frontend and the REST/SSE API — one container, one process, matching the current monolithic deployment shape.
- Backend is stateless per-request: no server-side session store. Chat history travels with each `/chat` request exactly as `adapter.js` already sends it. Retriever/graph objects are built once at process startup as module-level singletons (not per-session, since Streamlit's per-session pattern goes away).
- `app/ingest/*`, `app/graph/*`, `app/retrieval/*`, `app/db.py` are not modified. New route handlers call them exactly as `app/tabs/*.py` does today.
- Vendor React 18.3.1, ReactDOM 18.3.1, and Babel Standalone 7.29.0 locally (same versions/SRI `support.js` already references) instead of depending on unpkg.com at runtime, so the app works with no outbound internet access from the browser.
- `requirements.txt`: drop `streamlit`, add `fastapi` and `uvicorn[standard]`.
- Keep port 8501 everywhere (Dockerfile `EXPOSE`, `docker-compose.yml` mapping) — only the Dockerfile `CMD` changes, from `streamlit run` to `uvicorn app.server:app`.
- New API layer gets FastAPI `TestClient` test coverage, matching the project's existing test rigor. No changes to existing backend tests since backend modules are untouched.

## Architecture

```
app/server.py            FastAPI app: mounts API routes, then StaticFiles(FrontendDesign) at "/"
app/api/
  schemas.py              Pydantic request/response models mirroring adapter.js's documented contracts
  health.py                GET /health
  repos.py                  GET /repos, GET /config/allowlist
  ingest.py                  POST /ingest/local, POST /ingest/github  (SSE)
  chat.py                     POST /chat  (SSE)
  browse.py                    GET /repos/{id}/browse, GET /repos/{id}/files
```

Route registration order: explicit API routes and an explicit `GET /` (returns `KnowledgeBot.dc.html`) are registered before `StaticFiles(directory="FrontendDesign")` is mounted at `/` as the catch-all for `support.js`, `src/api/adapter.js`, `vendor/*`. Starlette matches literal routes before falling through to a mount, so there is no path collision.

## Endpoint mapping

All endpoints and their exact request/response/event shapes are already specified in `FrontendDesign/src/api/adapter.js`'s own doc comments (written anticipating this swap). Backend calls behind each:

| Endpoint | Backend call (unchanged) |
|---|---|
| `GET /health` | `is_voyage_available()` for mode; a live `SELECT 1` for `db_ok`/`db_error` |
| `GET /repos` | `session.query(Repo).order_by(Repo.ingested_at.desc())` |
| `GET /config/allowlist` | `DEFAULT_ALLOWLIST` |
| `POST /ingest/local` (SSE) | `ingest_repo(repo_name, root_dir, progress_callback=...)` run on a worker thread; the callback pushes onto a queue that the async generator drains into `progress`/`done`/`error` SSE frames |
| `POST /ingest/github` (SSE) | same pattern over `ingest_github_url(...)` |
| `POST /chat` (SSE) | `graph.stream(state, config, stream_mode=["updates","messages"])` — `updates` payloads emit `node` events (node name = dict key: `rewrite_query`/`retrieve`/`grade_chunks`/`generate_answer`/`answer_not_found`); `messages` payloads where `metadata["langgraph_node"]=="generate_answer"` emit `token` events. This is the same mechanism `app/tabs/chat.py`'s `_run_graph_streamed` already uses, serialized to JSON instead of `st.write_stream`. Final state serialized into a `final` event using the same tuple-extraction `chat.py` already performs for citations/graded/retrieved. |
| `GET /repos/{id}/browse?keyword=` | same `Chunk`/`Repo` query `app/tabs/browser.py` runs today |
| `GET /repos/{id}/files?path=` | chunk lookup by path, ordered by `start_line` |

## Frontend changes (only files touched under `FrontendDesign/`)

- **`adapter.js`**: set `API_BASE` to a truthy same-origin value; replace each mock function body with a real `fetch()` (plain GETs) or a `fetch()` + `ReadableStream` reader parsing `event:`/`data:` frames (the three SSE endpoints — `EventSource` can't be used since these are POST requests). Function signatures and calling contract stay identical, so `KnowledgeBot.dc.html`'s `Component` class (the actual interaction/state-machine logic) needs zero edits.
- **`KnowledgeBot.dc.html`** `<head>`: add a `window.__resources = {...}` map, before `support.js` loads, pointing the three pinned CDN URLs to local vendored copies. This is the override hook already built into `support.js` (`cdnScriptFor` checks `window.__resources` before falling back to unpkg), so `support.js` itself (marked "generated, do not edit") stays untouched.
- New `FrontendDesign/vendor/` directory: pinned copies of `react@18.3.1`, `react-dom@18.3.1`, `@babel/standalone@7.29.0` (same versions/SRI already referenced in `support.js`).

## Deployment

- `requirements.txt`: drop `streamlit`, add `fastapi`, `uvicorn[standard]`.
- `Dockerfile` `CMD` → `uvicorn app.server:app --host 0.0.0.0 --port 8501`. Port, `EXPOSE`, and `docker-compose.yml`'s `8501:8501` mapping are unchanged.
- `docker-compose.yml` otherwise unchanged: same single `app` service + `db` service, same volumes.

## Testing

New `tests/test_api_*.py` using FastAPI's `TestClient`, one file per resource area (health/repos, ingest SSE, chat SSE, browse), covering the same scenarios the deleted Streamlit tabs implicitly exercised. Existing 90+ backend tests are untouched since no backend module changes.
