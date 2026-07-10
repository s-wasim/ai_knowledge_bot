# AI Knowledge Bot

A Streamlit + LangGraph RAG application that ingests codebases and answers onboarding questions with grounded citations.

## Quickstart

### Prerequisites

- Docker & Docker Compose
- Anthropic API key (required)
- Voyage AI API key (optional — enables vector search)

### Setup

```bash
# Clone and enter the directory
cd ai_knowledge_bot

# Configure environment
cp .env.example .env
# Edit .env: add your ANTHROPIC_API_KEY
# Optionally add VOYAGE_API_KEY for vector mode

# Start the stack
docker compose up
```

Open http://localhost:8501 in your browser.

### Ingest a codebase

1. Go to the **📦 Ingest** tab
2. Choose **Local Folder** and enter a path under `./repos/`
3. Click **Ingest** — progress bar shows file-by-file progress
4. Or choose **GitHub URL** and paste a public repo URL

### Ask questions

1. Go to the **💬 Chat** tab
2. Select the ingested repo from the dropdown
3. Type a question (or use the quick-ask buttons in the sidebar)
4. Answers stream with `[n]` citations you can click to view source code

## Configuration

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `ANTHROPIC_API_KEY` | Yes | — | Anthropic API key for chat/rewrite/grading |
| `VOYAGE_API_KEY` | No | — | Voyage AI key for vector embeddings |
| `DATABASE_URL` | No | `postgresql://postgres:postgres@db:5432/knowledgebot` | Postgres connection string |

## Two modes

- **Full-text search** (VOYAGE_API_KEY unset): Uses Postgres FTS with tsvector
- **Vector search** (VOYAGE_API_KEY set): Uses voyage-code-3 embeddings with pgvector HNSW

The active mode is shown as a badge on every tab.

## Project structure

```
├── app/
│   ├── main.py              # Streamlit entry point
│   ├── db.py                # SQLAlchemy models + pgvector setup
│   ├── llm.py               # Claude LLM factory
│   ├── graph/
│   │   ├── state.py         # RagState, GradedChunk, Citation
│   │   ├── build.py         # LangGraph graph compilation
│   │   └── nodes/
│   │       ├── rewrite.py   # Query rewriting with history
│   │       ├── retrieve.py  # Retriever invocation
│   │       ├── grade.py     # Relevance grading
│   │       ├── answer.py    # Answer generation with citations
│   │       └── not_found.py # Honesty path
│   ├── ingest/
│   │   ├── walker.py        # File tree walker with FR-3 filters
│   │   ├── chunker.py       # Line-window chunker (80/20)
│   │   ├── pipeline.py      # Ingest orchestration
│   │   ├── embedder.py      # Voyage AI embedding
│   │   └── github.py        # GitHub zip download
│   ├── retrieval/
│   │   ├── base.py          # Retriever protocol
│   │   ├── fts.py           # Postgres FTS retriever
│   │   ├── vector.py        # pgvector cosine retriever
│   │   └── factory.py       # Mode-selection factory
│   └── tabs/
│       ├── ingest.py        # Ingest tab UI
│       ├── chat.py          # Chat tab UI
│       └── browser.py       # Index Browser tab UI
├── tests/                   # pytest unit tests
├── fixtures/mini_repo/      # Test fixture codebase
├── repos/                   # Mounted repo directory
├── DEMO_QUESTIONS.md        # 5 demo questions
├── DEMO_SCRIPT.md           # Demo walkthrough
└── docker-compose.yml       # App + pgvector containers
```

## Reset

```bash
docker compose down -v
# Removes all data — re-ingest after restart
```

## Testing

```bash
pip install pytest
pytest tests/
```

## Architecture decisions

- **Monolith**: Ingest and chat run in-process in Streamlit (single-user demo)
- **FTS first**: Build with zero extra API keys, then upgrade to vector
- **3 Claude calls max per question**: Rewrite → Grade (batched) → Answer
- **Transparency**: All retrieved chunks + grades visible in the UI
