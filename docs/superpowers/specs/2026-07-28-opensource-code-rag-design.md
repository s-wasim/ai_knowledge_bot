# Open-Source Code RAG — Design

**Date:** 2026-07-28
**Status:** Approved
**Scope:** Replace the Voyage-embedding RAG stack with a fully open-source, Postgres-backed
hybrid retrieval pipeline tuned for code search. Claude is used only for query rewriting,
relevance selection, and answer generation — never for embeddings.

---

## 1. Motivation

An audit of the running system established the following, each verified against the live
stack rather than inferred:

| Finding | Evidence |
|---|---|
| The chat UI shows retrieved chunks but never the answer | Headless-browser probe: the answer container renders with `innerHTML === ""` while citation chips populate |
| Root cause is not the backend | `POST /chat` streams 26 `token` frames plus a `final` frame containing a complete answer and 5 citations |
| Root cause is attribute lowercasing | The rendered DOM carries `dangerouslysetinnerhtml="[object Object]"` on 4 elements. The template is read via `x-dc.innerHTML`, so the HTML parser lowercases the attribute; the runtime's raw-source re-fetch is skipped because the page sets `window.__resources` |
| GitHub-ingested paths are unusable | Stored paths look like `/tmp/gh_ingest_xvpsadyu/transformers-main/awesome-transformers.md` — an absolute path into a deleted temp directory |
| Directory suggestions are empty | `path.split("/")[0]` on a leading-slash absolute path yields `""`, producing "You might want to look in  for relevant files" |
| Retrieval was never semantic | `VOYAGE_API_KEY` is empty, so `create_retriever` silently returned `FtsRetriever`; `/health` reported `mode: "fts"` |
| Relevance grading is fragile | `grade_chunks` parses the model's JSON by string-splitting on code fences, then falls back to keeping every chunk on any exception |

The retrieval quality problem and the API-key dependency are the same problem: relevance
rested on `english`-stemmed full-text search over code. This design removes the external
embedding dependency entirely and replaces single-signal retrieval with fused multi-signal
retrieval, with Claude as the relevance judge.

## 2. Non-goals

- No agentic retrieval loop. Retrieval stays deterministic and unit-testable.
- No GPU requirement. Everything runs on CPU.
- No change to the HTTP contract beyond additive SSE fields.
- No unrelated refactoring of the ingest API, browse endpoints, or theming.

## 3. Architecture

```
app/ingest/                  (kept — renaming to app/index/ would churn 6 test modules
                              for a cosmetic gain and is dropped from the design)
  walker.py                  file discovery; adds cheap count_files() (no file reads)
  chunker/__init__.py        chunk_file() dispatcher by extension
  chunker/ast_chunker.py     tree-sitter: .py .js .ts .tsx
  chunker/text_chunker.py    .md headings, .sql statements, line-window fallback
  embedder.py                local SentenceTransformer, process-wide singleton
  pipeline.py                walk -> chunk -> embed -> persist (repo-relative paths)
  github.py                  unchanged download/extract; passes content_dir as the path root

app/retrieval/
  base.py                    ChunkData: path, start_line, end_line, content, score,
                             symbol, language, sources[]
  dense.py                   DenseRetriever   — pgvector cosine
  lexical.py                 LexicalRetriever — code-aware full-text
  symbolic.py                SymbolicRetriever — pg_trgm on symbol/path + identifier match
  fusion.py                  reciprocal_rank_fusion()
  hybrid.py                  HybridRetriever — runs the three, fuses, dedups
  factory.py                 builds HybridRetriever; reports degraded mode

app/graph/nodes/
  rewrite.py                 Claude -> standalone query + extracted identifiers
  retrieve.py                HybridRetriever -> ~24 candidates
  select.py                  Claude -> keeps only relevant chunks (replaces grade.py)
  answer.py                  Claude -> streamed answer + validated citations
  not_found.py               honest "not in this codebase" path
```

Graph shape stays `rewrite_query -> retrieve -> select_chunks -> {generate_answer |
answer_not_found} -> END`. LangGraph is retained because the frontend already renders the
node sequence as a pipeline visualization, and `stream_mode=["updates", "messages"]` is what
drives both the stage indicators and token streaming.

### 3.1 Embedding model

`jinaai/jina-embeddings-v2-base-code` via `sentence-transformers`. Apache 2.0, 161M
parameters, 768 dimensions, 8192-token context, trained on 30 programming languages.

- Loaded once per process behind a double-checked lock, mirroring the existing
  `app/api/deps.py` singleton pattern.
- Downloaded at **image build time** so runtime needs no network. `HF_HOME` is pinned to a
  baked-in path and `HF_HUB_OFFLINE=1` is set at runtime, so a silent runtime download can
  never mask a broken build.
- Embeddings are L2-normalized at write time, so cosine distance and inner product agree
  and `ORDER BY embedding <=> query` is stable.

### 3.2 Chunking

One chunk per semantic unit, each prefixed with a synthetic header so both the embedder and
Claude see context that the raw body lacks:

```
# app/db.py — get_session
<body>
```

- `.py .js .ts .tsx` — tree-sitter. Chunk at function, method, and class definitions,
  keeping signatures and decorators attached. Nodes larger than `MAX_CHUNK_LINES` (120) are
  split on their direct children; module-level code between definitions is grouped into
  synthetic chunks.
- `.md` — split on headings, keeping the heading with its section.
- `.sql` — split on statement boundaries.
- Everything else, plus any parse failure or unsupported grammar — the existing 80/20
  line-window chunker. A tree-sitter exception must never fail an ingest.

`symbol` (the enclosing definition name, nullable) and `language` are persisted per chunk.

### 3.3 Retrieval

Three independent candidate generators run against Postgres, then fuse.

**Dense** — cosine over pgvector, top 30. Skipped when no embeddings exist or the model
failed to load.

**Lexical** — code-aware full-text, top 30. The `english` configuration stems code
identifiers into nonsense, so the generated tsvector column uses `simple` and additionally
indexes identifier-split text, letting `getConnection` match `get connection`:

```sql
tsv GENERATED ALWAYS AS (to_tsvector('simple',
  coalesce(content,'') || ' ' ||
  translate(regexp_replace(coalesce(content,''),'([a-z0-9])([A-Z])','\1 \2','g'),'_.-/','    ')
)) STORED
```

Both operands are immutable, which a generated column requires. Ranking uses `ts_rank_cd`,
retaining the existing AND-to-OR query relaxation so a multi-word question still matches.

**Symbolic** — top 20. Identifier-shaped tokens are extracted from the rewritten query
(`snake_case`, `camelCase`, `dotted.paths`, backtick-quoted spans) and matched against
`symbol` and `path` using `pg_trgm` similarity, with a GIN trigram index on each. This is
what makes a query naming `get_connection` reliably surface its definition — the case
dense-only retrieval handles worst.

**Fusion** — Reciprocal Rank Fusion, `score = Σ 1/(60 + rank_i)`. Rank-based fusion needs no
score calibration between retrievers, which matters because cosine similarity, `ts_rank_cd`,
and trigram similarity are not comparable quantities. Results dedup on
`(path, start_line)`; each survivor records which retrievers found it in `sources`, which is
both a debugging aid and evidence surfaced in the UI. Fused output is capped at 24
candidates.

### 3.4 Claude as relevance judge

`select_chunks` is the node that satisfies the "Claude finds the relevant docs" requirement.

Candidates are rendered as a numbered list of `path:start-end`, symbol, language, and body
truncated to 60 lines. Output is obtained via `with_structured_output(SelectionResult)` —
tool-use-validated rather than parsed out of a code fence — returning per candidate:
`index`, `keep`, `relevance` (0–1), `reason`.

Bulletproofing, in order of application:

1. One retry on validation failure.
2. Indices outside `1..len(candidates)` are discarded, not clamped into a wrong chunk.
3. Candidates the model never mentions default to `keep=False`.
4. Total call failure degrades to the top 8 by fusion score, each labelled
   `"kept by fusion score (LLM grading unavailable)"`, so the failure is visible rather than
   disguised as a judgment.
5. Kept chunks are capped at 8, ordered by `relevance` descending.
6. Zero kept chunks routes to `answer_not_found`.

Contiguous kept chunks from the same file are merged before answering, so a function split
across two chunks reaches the model whole.

### 3.5 Answer generation

Streaming Claude call, `temperature=0` actually passed (today the parameter is accepted and
discarded). Citations are validated against kept-chunk numbering; invalid `[n]` markers are
stripped. Two fixes to how evidence reaches the UI:

- `_citation_dict` gains `index`, so a chip labelled `[2]` refers to the chunk the prose
  cites as `[2]`. Today chips are numbered by array position, so an answer citing `[2]` and
  `[5]` renders chips `[1]` and `[2]`.
- The global `re.sub(r' +', ' ')` collapse is replaced by targeted cleanup applied only
  where a marker was removed, so code indentation in answers survives.

## 4. Failure behaviour

No LLM or model failure produces a 500. Every path yields a degraded answer or an SSE
`error` frame.

| Failure | Behaviour |
|---|---|
| Embedding model won't load | Dense retriever disabled; lexical + symbolic still fuse; `/health` reports `mode: "degraded"` with the reason |
| Embedding call fails mid-ingest | Affected chunks persist with `NULL` embedding; file content is never lost; ingest reports how many chunks lack embeddings |
| Selection LLM fails | Top 8 by fusion score, explicitly labelled as ungraded |
| Rewrite LLM fails | Falls back to the raw question |
| Answer LLM fails | SSE `error` frame; the UI already renders it inside the assistant card |
| No candidates at all | `answer_not_found`, with directory suggestions drawn from the index |
| Ingest exceeds `MAX_CHUNKS_PER_REPO` (default 20000, `KB_MAX_CHUNKS` env override) | Truncated, and the cap is emitted as an SSE warning plus a log line — never silent |

`/health` reports `db_ok`, `db_error`, `mode`, `mode_display`, `embed_model_ok`, and
`embed_dims`.

## 5. Migration

768-dimension vectors are incompatible with the existing 1024-dimension column, and the new
chunker changes chunk boundaries, so the index is rebuilt rather than migrated.

- `scripts/reset_index.py` drops and recreates `chunks` (including the generated `tsv`
  column and all indexes), leaving `repos` rows to be re-ingested.
- `init_db` creates `pg_trgm` alongside `vector`, and adds the `symbol`/`language` columns
  and trigram indexes idempotently.
- `requirements.txt`: add `sentence-transformers`, CPU `torch`, `tree-sitter`,
  `tree-sitter-language-pack`; remove `voyageai`.
- `Dockerfile`: pre-download the model at build; set `HF_HOME` and `HF_HUB_OFFLINE=1`.
  Image grows by roughly 1.5–2 GB and first build is materially slower.
- `docker-compose.yml` is unchanged; the app stays on port 8501.

After the rewrite, `./repos/document_pipeline` is ingested and all five demo questions are
verified end-to-end.

## 6. Frontend fixes

- **Render bug.** Six `dangerouslySetInnerHTML` attributes become
  `sc-camel-dangerously-set-inner-h-t-m-l`, the runtime's own escape hatch
  (`CAMEL_ATTR` in `support.js`), which survives HTML lowercasing because it is already
  lowercase. This restores the answer body, nav icons, empty-state icons, and every code
  viewer. A comment at each site records why the spelling is what it is.
- Citation chips use `citation.index`.
- Score meters use a per-row computed width; today `gradedMeterFill` is a static style and
  every meter renders full.
- Chunk rows show `sources` badges (dense / text / symbol) so retrieval provenance is
  visible.
- Pipeline stage labels track the renamed nodes.

## 7. Other audit fixes in scope

- `pipeline.py` stores paths relative to the ingest root, fixing citations, directory
  suggestions, and the browse panel for GitHub ingests.
- The pre-walk file count uses `count_files()` (stat only) instead of a second
  `walk_directory()`, which currently reads every file's full text just to count.
- `rewrite.py` sends prior assistant turns as `AIMessage`; today they are wrapped in
  `HumanMessage` with an `"Assistant: "` prefix, which corrupts follow-up rewriting — the
  exact path demo question 3 depends on.
- `get_llm`/`get_llm_streaming` pass `temperature` through.
- `state.get("rewritten_query", ...)` becomes `state.get("rewritten_query") or ...`; the key
  always exists with a possible `None`, so the default never applied.

## 8. Testing

**Unit.** AST chunking per language, oversized-node splitting, parse-failure fallback,
synthetic headers; identifier extraction; RRF ordering and dedup; lexical tokenization;
every `select_chunks` fallback branch; citation numbering and marker stripping; relative
path construction.

**Integration.** SQL retrievers against the Docker Postgres, marked `integration` and
skipped when unavailable, covering dense/lexical/symbolic recall on a seeded fixture and the
generated-column behaviour.

**Eval.** `tests/eval/test_demo_questions.py`, marked `eval`, ingests the fixture repo, runs
the five demo questions through the live graph, and asserts that expected files appear in
citations and that the notification question routes to `not_found`. This is what catches
retrieval-quality regressions, which unit tests structurally cannot.

The existing 111-test suite must stay green, with tests updated where interfaces
legitimately change.

## 9. Success criteria

1. Chat renders streamed answer prose, citations, and code viewers in the browser.
2. `/health` reports `mode: "hybrid"` with no API key beyond `ANTHROPIC_API_KEY`.
3. Demo questions 1, 2, 3, and 5 answer with citations pointing at correct
   repo-relative paths; question 4 honestly reports not-found.
4. Citation chip numbers match the `[n]` markers in the prose.
5. Full test suite green, including the new eval harness.
