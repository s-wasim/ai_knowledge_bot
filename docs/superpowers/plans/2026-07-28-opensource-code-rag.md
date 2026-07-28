# Open-Source Code RAG Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the Voyage-embedding RAG stack with a local, open-source hybrid retrieval pipeline over Postgres, using Claude only for query rewriting, relevance selection, and answering.

**Architecture:** Three independent Postgres retrievers (pgvector dense, code-aware full-text, pg_trgm symbolic) run per query and fuse by Reciprocal Rank Fusion into ~24 candidates. Claude grades those candidates through a tool-use-validated structured output and keeps only the relevant ones, then streams a cited answer. Chunking becomes AST-aware via tree-sitter so each chunk is a whole function or class.

**Tech Stack:** FastAPI, LangGraph, langchain-anthropic, sentence-transformers (`jinaai/jina-embeddings-v2-base-code`, 768-dim), tree-sitter + tree-sitter-language-pack, Postgres 16 + pgvector + pg_trgm, SQLAlchemy 2.

**Spec:** `docs/superpowers/specs/2026-07-28-opensource-code-rag-design.md`

## Global Constraints

- Embedding model: `jinaai/jina-embeddings-v2-base-code`, 768 dimensions, L2-normalized at write time, CPU only.
- Model is downloaded at Docker build time; runtime sets `HF_HUB_OFFLINE=1` so a missing model fails loudly instead of downloading silently.
- `ANTHROPIC_API_KEY` is the only required external credential. `VOYAGE_API_KEY` is removed entirely.
- Anthropic model id stays `claude-sonnet-5`; `temperature=0` must actually be passed to `ChatAnthropic`.
- Fusion constant `RRF_K = 60`. Candidate cap `MAX_CANDIDATES = 24`. Kept-chunk cap `MAX_KEPT = 8`.
- Chunking: `MAX_CHUNK_LINES = 120`, line-window fallback stays 80 lines / 20 overlap.
- Ingest cap: `MAX_CHUNKS_PER_REPO = 20000`, override via `KB_MAX_CHUNKS`. Truncation must be reported, never silent.
- Stored chunk paths are always relative to the ingest root, using forward slashes.
- No LLM or model failure may produce an HTTP 500. Degrade, and label the degradation.
- App stays on port 8501. Existing 111 tests stay green except where an interface legitimately changes.
- Every LLM-facing structured output uses `with_structured_output`, never manual JSON string parsing.

---

### Task 1: Dependencies, schema, and index reset

**Files:**
- Modify: `requirements.txt`
- Modify: `Dockerfile`
- Modify: `app/db.py`
- Create: `scripts/reset_index.py`
- Test: `tests/test_db.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `Chunk.symbol: str | None`, `Chunk.language: str | None`, `Chunk.embedding: Vector(768)`; `app.db.EMBED_DIMS = 768`; `scripts/reset_index.py` as a `python -m scripts.reset_index` entry point.

- [ ] **Step 1: Write the failing test**

```python
def test_chunk_model_has_symbol_and_language_columns():
    from app.db import Chunk
    cols = {c.name for c in Chunk.__table__.columns}
    assert {"symbol", "language"} <= cols


def test_embedding_dimension_is_768():
    from app.db import Chunk, EMBED_DIMS
    assert EMBED_DIMS == 768
    assert Chunk.__table__.c.embedding.type.dim == 768
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_db.py -v -k "symbol or dimension"`
Expected: FAIL — `ImportError: cannot import name 'EMBED_DIMS'`

- [ ] **Step 3: Implement**

In `app/db.py`: add `EMBED_DIMS = 768`, add `symbol = Column(String, nullable=True)` and `language = Column(String, nullable=True)` to `Chunk`, change `embedding` to `Vector(EMBED_DIMS)`. In `init_db`, create `pg_trgm` alongside `vector`, replace the `tsv` generated column with the code-aware expression, and add the trigram indexes:

```python
conn.execute(DDL("CREATE EXTENSION IF NOT EXISTS pg_trgm"))
conn.execute(DDL(
    "ALTER TABLE chunks ADD COLUMN IF NOT EXISTS tsv tsvector "
    "GENERATED ALWAYS AS (to_tsvector('simple', "
    "  coalesce(content,'') || ' ' || "
    "  translate(regexp_replace(coalesce(content,''),'([a-z0-9])([A-Z])','\\1 \\2','g'),'_.-/','    ')"
    ")) STORED"
))
conn.execute(DDL("CREATE INDEX IF NOT EXISTS ix_chunks_symbol_trgm ON chunks USING gin (symbol gin_trgm_ops)"))
conn.execute(DDL("CREATE INDEX IF NOT EXISTS ix_chunks_path_trgm ON chunks USING gin (path gin_trgm_ops)"))
```

`requirements.txt`: remove `voyageai`, add `sentence-transformers>=3.0,<6`, `torch>=2.2`, `tree-sitter>=0.23,<1`, `tree-sitter-language-pack>=0.7,<1`.

`Dockerfile`: set `ENV HF_HOME=/opt/hf`, download the model in a build layer, then `ENV HF_HUB_OFFLINE=1`.

`scripts/reset_index.py`: `DROP TABLE IF EXISTS chunks CASCADE; DELETE FROM repos;` then `init_db()`.

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_db.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add requirements.txt Dockerfile app/db.py scripts/reset_index.py tests/test_db.py
git commit -m "feat: 768-dim embeddings, code-aware tsv, trigram indexes, index reset script"
```

---

### Task 2: Cheap file counting

**Files:**
- Modify: `app/ingest/walker.py`
- Test: `tests/test_walker.py`

**Interfaces:**
- Produces: `count_files(root_dir: Path | str, allowlist: set[str] | None = None) -> int` — applies the same filters as `walk_directory` but never reads file bodies.

- [ ] **Step 1: Write the failing test**

```python
def test_count_files_matches_walk_without_reading(tmp_path):
    from app.ingest.walker import count_files, walk_directory
    (tmp_path / "a.py").write_text("x = 1")
    (tmp_path / "b.md").write_text("# hi")
    (tmp_path / "c.bin").write_bytes(b"\x00\x01")
    assert count_files(tmp_path) == len(list(walk_directory(tmp_path)))
```

- [ ] **Step 2: Run to verify it fails**

Run: `pytest tests/test_walker.py -k count_files -v`
Expected: FAIL — `ImportError: cannot import name 'count_files'`

- [ ] **Step 3: Implement**

Extract the shared filter into `_is_candidate(filepath) -> bool` (symlink, suffix, size, NUL-byte header) and have both `walk_directory` and `count_files` use it. `count_files` returns a count without calling `read_text`.

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_walker.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add app/ingest/walker.py tests/test_walker.py
git commit -m "feat: add count_files() to avoid reading every file twice during ingest"
```

---

### Task 3: AST-aware chunking

**Files:**
- Create: `app/ingest/chunker/__init__.py`
- Create: `app/ingest/chunker/ast_chunker.py`
- Create: `app/ingest/chunker/text_chunker.py`
- Delete: `app/ingest/chunker.py`
- Test: `tests/test_chunker.py`, `tests/test_ast_chunker.py`

**Interfaces:**
- Produces: `chunk_file(path: str, text: str) -> list[dict]` where each dict has keys `path`, `start_line`, `end_line`, `content`, `symbol` (`str | None`), `language` (`str | None`). `content` is prefixed with a synthetic header line `# <path> — <symbol>`. Also `line_window_chunks(path, text, window=80, overlap=20, min_chunk=10) -> list[dict]` in `text_chunker`, and `LANGUAGE_BY_SUFFIX: dict[str, str]` in `ast_chunker`.

- [ ] **Step 1: Write the failing tests**

```python
def test_python_functions_become_separate_chunks():
    from app.ingest.chunker import chunk_file
    src = "def alpha():\n    return 1\n\n\ndef beta():\n    return 2\n"
    chunks = chunk_file("app/x.py", src)
    symbols = {c["symbol"] for c in chunks}
    assert {"alpha", "beta"} <= symbols
    assert all(c["language"] == "python" for c in chunks)


def test_chunk_content_carries_synthetic_header():
    from app.ingest.chunker import chunk_file
    chunks = chunk_file("app/x.py", "def alpha():\n    return 1\n")
    assert chunks[0]["content"].startswith("# app/x.py — alpha")
    assert "def alpha" in chunks[0]["content"]


def test_unparseable_source_falls_back_to_line_window():
    from app.ingest.chunker import chunk_file
    chunks = chunk_file("app/x.py", "def (((( broken\n" * 300)
    assert len(chunks) > 1
    assert all(c["start_line"] >= 1 for c in chunks)


def test_unknown_extension_uses_line_window():
    from app.ingest.chunker import chunk_file
    chunks = chunk_file("notes.txt", "line\n" * 200)
    assert len(chunks) > 1
    assert chunks[0]["symbol"] is None


def test_markdown_splits_on_headings():
    from app.ingest.chunker import chunk_file
    chunks = chunk_file("README.md", "# One\ntext\n\n# Two\nmore\n")
    assert len(chunks) == 2
    assert chunks[0]["symbol"] == "One"


def test_oversized_function_is_split():
    from app.ingest.chunker import chunk_file
    body = "\n".join(f"    x{i} = {i}" for i in range(300))
    chunks = chunk_file("app/big.py", f"def huge():\n{body}\n")
    assert len(chunks) > 1
```

- [ ] **Step 2: Run to verify they fail**

Run: `pytest tests/test_ast_chunker.py -v`
Expected: FAIL — module not found

- [ ] **Step 3: Implement**

`ast_chunker.py`: `LANGUAGE_BY_SUFFIX = {".py": "python", ".js": "javascript", ".ts": "typescript", ".tsx": "tsx"}`. Load grammars lazily through `tree_sitter_language_pack.get_parser(lang)`, cached per language. Walk top-level children; emit a chunk per `function_definition`, `class_definition`, `decorated_definition`, `function_declaration`, `class_declaration`, `method_definition`, `lexical_declaration`, and group consecutive non-definition siblings into one chunk. Include preceding decorators and comments in the chunk. Nodes longer than `MAX_CHUNK_LINES` recurse one level into their body; if still oversized, hand to `line_window_chunks`. Any exception, missing grammar, or `root_node.has_error` on the whole file returns `None` so the dispatcher falls back.

`text_chunker.py`: `line_window_chunks` (the current logic, plus `symbol=None`), `markdown_chunks` (split on `^#{1,6} `, symbol = heading text), `sql_chunks` (split on `;` at line ends).

`__init__.py`: dispatch on suffix — AST for the four code languages with fallback, markdown for `.md`, SQL for `.sql`, line window otherwise. Prefix every chunk's `content` with the synthetic header. Header is added after line numbers are computed, so `start_line`/`end_line` keep pointing at the real source.

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_ast_chunker.py tests/test_chunker.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add app/ingest/chunker tests/test_ast_chunker.py tests/test_chunker.py
git rm app/ingest/chunker.py
git commit -m "feat: AST-aware chunking with markdown/sql/line-window fallbacks"
```

---

### Task 4: Local embedder

**Files:**
- Rewrite: `app/ingest/embedder.py`
- Test: `tests/test_embedder.py`

**Interfaces:**
- Produces: `MODEL_NAME`, `EMBED_DIMS`, `get_model()` (cached singleton, returns `None` on load failure), `embed_texts(texts, batch_size=32) -> list[list[float]] | None`, `embed_query(text) -> list[float] | None`, `is_embedding_available() -> bool`, `embedding_status() -> dict` with keys `ok`, `model`, `dims`, `error`.

- [ ] **Step 1: Write the failing tests**

```python
@patch("app.ingest.embedder.get_model")
def test_embed_texts_batches_and_normalizes(mock_get_model):
    from app.ingest.embedder import embed_texts
    model = MagicMock()
    model.encode.side_effect = [[[1.0, 0.0]], [[0.0, 1.0]]]
    mock_get_model.return_value = model
    out = embed_texts(["a", "b"], batch_size=1)
    assert len(out) == 2
    assert model.encode.call_count == 2
    assert model.encode.call_args.kwargs["normalize_embeddings"] is True


@patch("app.ingest.embedder.get_model", return_value=None)
def test_embed_texts_returns_none_when_model_unavailable(_):
    from app.ingest.embedder import embed_texts
    assert embed_texts(["a"]) is None


@patch("app.ingest.embedder.get_model", return_value=None)
def test_status_reports_not_ok(_):
    from app.ingest.embedder import embedding_status
    assert embedding_status()["ok"] is False
```

- [ ] **Step 2: Run to verify they fail**

Run: `pytest tests/test_embedder.py -v`
Expected: FAIL — `is_voyage_available` gone, new names missing

- [ ] **Step 3: Implement**

`get_model()` loads `SentenceTransformer(MODEL_NAME, trust_remote_code=True, device="cpu")` once behind a lock, caching both the model and any load error. `embed_texts` batches, calls `model.encode(batch, normalize_embeddings=True, convert_to_numpy=True)`, converts to plain lists, and returns `None` on any exception after logging. Remove `is_voyage_available` and every Voyage import.

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_embedder.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add app/ingest/embedder.py tests/test_embedder.py
git commit -m "feat: replace Voyage client with local jina-embeddings-v2-base-code"
```

---

### Task 5: Ingest pipeline

**Files:**
- Modify: `app/ingest/pipeline.py`
- Modify: `app/api/ingest.py`
- Test: `tests/test_pipeline.py`, `tests/test_api_ingest.py`

**Interfaces:**
- Produces: `ingest_repo(repo_name, root_dir, source_url=None, branch=None, progress_callback=None) -> Repo` with `progress_callback(current: int, total: int, chunks_added: int, filename: str)` where `filename` is repo-relative. Also `MAX_CHUNKS_PER_REPO`.

- [ ] **Step 1: Write the failing tests**

```python
def test_stored_paths_are_relative_to_root(tmp_path, ...):
    # ingest tmp_path containing pkg/mod.py; assert Chunk.path == "pkg/mod.py"


def test_progress_filename_is_relative(tmp_path, ...):
    # assert no reported filename starts with str(tmp_path)


def test_chunk_cap_is_reported(tmp_path, ...):
    # with MAX_CHUNKS_PER_REPO patched to 1, assert the truncation warning is emitted
```

- [ ] **Step 2: Run to verify they fail**

Run: `pytest tests/test_pipeline.py -v`
Expected: FAIL — absolute paths returned

- [ ] **Step 3: Implement**

Compute `root = Path(root_dir).resolve()` once; `rel = filepath.relative_to(root).as_posix()`, with a `ValueError` guard falling back to `filepath.name`. Pass `rel` to `chunk_file` and the progress callback. Use `count_files` for `total_files`. Persist `symbol` and `language`. Embed via `embed_texts`; on `None`, leave embeddings `NULL` and log the count. Stop adding chunks past `MAX_CHUNKS_PER_REPO` and surface the cap through a `warning` SSE frame from `app/api/ingest.py`.

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_pipeline.py tests/test_api_ingest.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add app/ingest/pipeline.py app/api/ingest.py tests/test_pipeline.py tests/test_api_ingest.py
git commit -m "fix: store repo-relative chunk paths; report ingest caps and embedding gaps"
```

---

### Task 6: Hybrid retrieval

**Files:**
- Modify: `app/retrieval/base.py`
- Create: `app/retrieval/dense.py`, `app/retrieval/lexical.py`, `app/retrieval/symbolic.py`, `app/retrieval/fusion.py`, `app/retrieval/hybrid.py`
- Delete: `app/retrieval/vector.py`, `app/retrieval/fts.py`
- Modify: `app/retrieval/factory.py`
- Test: `tests/test_fusion.py`, `tests/test_dense_retriever.py`, `tests/test_lexical_retriever.py`, `tests/test_symbolic_retriever.py`, `tests/test_hybrid_retriever.py`
- Delete: `tests/test_vector_retriever.py`, `tests/test_fts_retriever.py`

**Interfaces:**
- Produces:
  - `ChunkData(path, start_line, end_line, content, score, symbol=None, language=None, sources=())`
  - `extract_identifiers(query: str) -> list[str]`
  - `reciprocal_rank_fusion(ranked_lists: dict[str, list[ChunkData]], k: int = 60, limit: int = 24) -> list[ChunkData]` — sets `score` to the RRF score and `sources` to the contributing retriever names, sorted descending.
  - `DenseRetriever(session_factory).search(repo_id, query, k=30)`, `LexicalRetriever(...)`, `SymbolicRetriever(...)`, all matching the existing `Retriever` protocol.
  - `HybridRetriever(session_factory).search(repo_id, query, k=24)`
  - `create_retriever(session_factory) -> tuple[HybridRetriever, str]` where mode is `"hybrid"` or `"degraded"`; `get_mode_display(mode)` returns `"Hybrid (code embeddings + full-text + symbols)"` or `"Degraded (full-text + symbols only)"`.

- [ ] **Step 1: Write the failing tests**

```python
def test_rrf_ranks_items_found_by_multiple_retrievers_higher():
    from app.retrieval.fusion import reciprocal_rank_fusion
    a = ChunkData("a.py", 1, 9, "x", 0.0)
    b = ChunkData("b.py", 1, 9, "y", 0.0)
    fused = reciprocal_rank_fusion({"dense": [b, a], "text": [a]})
    assert fused[0].path == "a.py"
    assert set(fused[0].sources) == {"dense", "text"}


def test_rrf_dedups_on_path_and_start_line():
    ...


def test_extract_identifiers_finds_snake_camel_and_dotted():
    from app.retrieval.symbolic import extract_identifiers
    ids = extract_identifiers("where is get_connection or getPool in app.db?")
    assert "get_connection" in ids and "getPool" in ids and "app.db" in ids


def test_extract_identifiers_ignores_plain_english():
    from app.retrieval.symbolic import extract_identifiers
    assert extract_identifiers("how does authentication work") == []


def test_dense_retriever_returns_empty_when_embeddings_unavailable():
    ...


def test_hybrid_survives_one_retriever_raising():
    # a retriever that raises must not fail the search; the others still return
    ...
```

- [ ] **Step 2: Run to verify they fail**

Run: `pytest tests/test_fusion.py tests/test_symbolic_retriever.py -v`
Expected: FAIL — modules missing

- [ ] **Step 3: Implement**

`dense.py` embeds the query with `embed_query`, returns `[]` when that is `None`, and orders by `embedding <=> :vec` with `embedding IS NOT NULL`. `lexical.py` keeps the AND-to-OR relaxation but on the `simple` configuration with `ts_rank_cd`. `symbolic.py` extracts identifiers and matches `symbol`/`path` via `similarity()` plus a `content ILIKE` term, returning `[]` when no identifiers are present. `hybrid.py` runs all three inside individual `try/except` blocks so one failure degrades rather than propagates, then fuses. `factory.py` returns `"degraded"` when `is_embedding_available()` is false.

- [ ] **Step 4: Run tests**

Run: `pytest tests/ -k retriever or fusion -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add app/retrieval tests/test_fusion.py tests/test_dense_retriever.py tests/test_lexical_retriever.py tests/test_symbolic_retriever.py tests/test_hybrid_retriever.py
git rm app/retrieval/vector.py app/retrieval/fts.py tests/test_vector_retriever.py tests/test_fts_retriever.py
git commit -m "feat: hybrid dense+lexical+symbolic retrieval fused by RRF"
```

---

### Task 7: Claude relevance selection

**Files:**
- Create: `app/graph/nodes/select.py`
- Delete: `app/graph/nodes/grade.py`
- Modify: `app/graph/state.py`, `app/graph/build.py`, `app/graph/nodes/rewrite.py`, `app/graph/nodes/retrieve.py`, `app/llm.py`
- Test: `tests/test_select.py`, `tests/test_rewrite.py`, `tests/test_graph_wiring.py`
- Delete: `tests/test_grade.py`

**Interfaces:**
- Produces:
  - `SelectionItem` (`index: int`, `keep: bool`, `relevance: float`, `reason: str`) and `SelectionResult` (`items: list[SelectionItem]`) as Pydantic models.
  - `select_chunks(state: RagState) -> dict` returning `{"graded": list[GradedChunk]}`; `GradedChunk` gains `relevance: float = 0.0`.
  - `merge_adjacent(kept: list[GradedChunk]) -> list[GradedChunk]`.
  - `get_llm(model=..., temperature=0)` and `get_llm_streaming(...)` now pass `temperature` through.
  - Graph node name `select_chunks` replaces `grade_chunks`.

- [ ] **Step 1: Write the failing tests**

```python
def test_select_keeps_only_flagged_chunks():
    # structured output returns keep=True for 1, keep=False for 2 -> one GradedChunk kept


def test_select_defaults_unmentioned_candidates_to_dropped():
    ...


def test_select_discards_out_of_range_indices():
    ...


def test_select_falls_back_to_fusion_order_when_llm_fails():
    # llm raises -> top MAX_KEPT kept, reason mentions "LLM grading unavailable"


def test_select_returns_empty_when_nothing_relevant():
    ...


def test_merge_adjacent_joins_contiguous_chunks_from_same_file():
    ...


def test_rewrite_sends_assistant_turns_as_ai_message():
    from langchain_core.messages import AIMessage
    # assert an AIMessage instance appears in the messages passed to the llm
```

- [ ] **Step 2: Run to verify they fail**

Run: `pytest tests/test_select.py -v`
Expected: FAIL — module missing

- [ ] **Step 3: Implement**

`select_chunks` renders candidates (index, `path:start-end`, symbol, language, body truncated to 60 lines), calls `get_llm().with_structured_output(SelectionResult)`, retries once, validates indices, defaults unmentioned to dropped, sorts kept by `relevance`, caps at `MAX_KEPT`, and merges adjacent chunks. On total failure, keeps the top `MAX_KEPT` by fusion score with `reason="kept by fusion score (LLM grading unavailable)"`. `rewrite.py` uses `AIMessage` for assistant turns and returns `{"rewritten_query": ...}`; every `state.get("rewritten_query", state["question"])` becomes `state.get("rewritten_query") or state["question"]`. `build.py` renames the node and routes on `any(g.keep ...)`.

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_select.py tests/test_rewrite.py tests/test_graph_wiring.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add app/graph tests/test_select.py tests/test_rewrite.py tests/test_graph_wiring.py app/llm.py
git rm app/graph/nodes/grade.py tests/test_grade.py
git commit -m "feat: tool-use-validated Claude relevance selection replaces JSON-scraped grading"
```

---

### Task 8: Answer, citations, and API surface

**Files:**
- Modify: `app/graph/nodes/answer.py`, `app/graph/nodes/not_found.py`, `app/api/chat.py`, `app/api/health.py`
- Test: `tests/test_answer.py`, `tests/test_citations.py`, `tests/test_api_chat.py`, `tests/test_api_health.py`

**Interfaces:**
- Produces: `_citation_dict` gains `index`; `_graded_dict` gains `relevance`, `symbol`, `sources`; `/health` gains `embed_model_ok: bool` and `embed_dims: int`.

- [ ] **Step 1: Write the failing tests**

```python
def test_citation_payload_carries_marker_number():
    # answer citing [2] only -> SSE citation dict has index == 2


def test_answer_preserves_indentation():
    from app.graph.nodes.answer import _postprocess_answer
    text = "Example:\n    def f():\n        return 1\n"
    cleaned, _ = _postprocess_answer(text, [])
    assert "    def f():" in cleaned


def test_health_reports_embedding_status():
    # GET /health -> body has embed_model_ok and embed_dims
```

- [ ] **Step 2: Run to verify they fail**

Run: `pytest tests/test_citations.py tests/test_api_health.py -v`
Expected: FAIL

- [ ] **Step 3: Implement**

`_postprocess_answer` only collapses doubled spaces on lines where a marker was removed. `Citation.index` flows into the SSE payload. `not_found.py` filters out empty path segments so suggestions never render blank. `health.py` reports `embedding_status()`.

- [ ] **Step 4: Run tests**

Run: `pytest tests/ -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add app/graph/nodes/answer.py app/graph/nodes/not_found.py app/api tests/
git commit -m "fix: citation numbering, answer whitespace, blank not-found suggestions, health status"
```

---

### Task 9: Frontend fixes

**Files:**
- Modify: `FrontendDesign/KnowledgeBot.dc.html`

**Interfaces:**
- Consumes: SSE `citation.index`, `graded[].relevance`, `graded[].sources`.

- [ ] **Step 1: Replace every `dangerouslySetInnerHTML` attribute**

Six sites: nav icon, chat empty icon, answer body, citation code viewer, graded code viewer, browser chunk body, plus the browser empty icon. Each becomes `sc-camel-dangerously-set-inner-h-t-m-l`, the runtime's `CAMEL_ATTR` escape hatch, with a comment explaining that the HTML parser lowercases the React spelling and `window.__resources` suppresses the raw-source re-fetch that would otherwise restore it.

- [ ] **Step 2: Fix citation chip numbering**

`citationsChips` uses `c.index` for the displayed number, falling back to `i + 1`.

- [ ] **Step 3: Fix score meters**

Add `meterFillStyle` per row using the computed `meterWidth`, and bind it in the template instead of the shared `chatStyles.gradedMeterFill`.

- [ ] **Step 4: Add provenance badges and rename stages**

Render `sources` as small badges on each graded row. Update `PIPELINE_STAGES` and `NODE_INDEX` to `rewrite_query`, `retrieve`, `select_chunks`, `generate_answer`, `answer_not_found`.

- [ ] **Step 5: Verify in a real browser**

Load the app, send a demo question, confirm the answer body renders text, nav icons appear, and a code viewer shows highlighted source.

- [ ] **Step 6: Commit**

```bash
git add FrontendDesign/KnowledgeBot.dc.html
git commit -m "fix: restore innerHTML rendering, correct citation numbers and score meters"
```

---

### Task 10: Eval harness

**Files:**
- Create: `tests/eval/__init__.py`, `tests/eval/test_demo_questions.py`
- Create: `tests/test_retrieval_integration.py`
- Create: `pytest.ini`

**Interfaces:**
- Consumes: the whole stack.
- Produces: `pytest -m eval` target; markers `eval` and `integration` registered.

- [ ] **Step 1: Register markers**

`pytest.ini` declares `markers = eval: end-to-end retrieval quality; integration: needs Postgres`.

- [ ] **Step 2: Write the integration tests**

`tests/test_retrieval_integration.py`, marked `integration`, skipped unless `DATABASE_URL` reaches a live Postgres. Seeds a throwaway repo with three known chunks, then asserts: the lexical retriever matches `get_connection` through the identifier-split tsvector when queried as `"get connection"`; the symbolic retriever finds a chunk by its `symbol` via trigram similarity; the dense retriever returns rows ordered by cosine distance when embeddings are present and `[]` when they are `NULL`. Tears the seeded repo down afterwards.

- [ ] **Step 3: Write the eval**

Ingest `fixtures/mini_repo`, then for each demo question assert the expected file appears among citation paths (`db.py` for the database question, `auth/login.py` for authentication, `config.yaml` for env vars) and that the notification question routes to `answer_not_found` with no citations. Skip the module when `ANTHROPIC_API_KEY` or Postgres is unavailable.

- [ ] **Step 4: Run it**

Run: `pytest -m integration -v` then `pytest -m eval -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/eval tests/test_retrieval_integration.py pytest.ini
git commit -m "test: end-to-end eval harness over the five demo questions"
```

---

### Task 11: Rebuild, reindex, verify

**Files:**
- Modify: `README.md`, `.env.example`

- [ ] **Step 1: Rebuild the image**

Run: `docker compose build app` — expect a slow first build while the model downloads.

- [ ] **Step 2: Reset the index**

Run: `docker compose run --rm app python -m scripts.reset_index`

- [ ] **Step 3: Ingest the demo repo**

Ingest `./repos/document_pipeline` and confirm `/repos` reports plausible counts and `/health` reports `mode: "hybrid"`.

- [ ] **Step 4: Verify the five demo questions in the browser**

Confirm streamed prose, citation chips matching the prose markers, repo-relative paths, working code viewers, and scrolling with a long conversation.

- [ ] **Step 5: Update docs**

`README.md` describes the hybrid pipeline and the reset script; `.env.example` drops `VOYAGE_API_KEY`.

- [ ] **Step 6: Commit**

```bash
git add README.md .env.example
git commit -m "docs: describe hybrid retrieval setup and index reset"
```
