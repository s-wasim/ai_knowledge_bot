# AI Knowledge Bot — Demo Questions

These 5 questions exercise the full RAG pipeline on a demo codebase.

## Setup

1. Ingest the demo repo (e.g., a small FastAPI or Flask project, or the fixture repo)
2. Select the repo in the Chat tab
3. Ask each question in order

## Questions

### Q1: Basic retrieval — "Where is the database connection configured?"

**Expected answer:** Points to `db.py` or equivalent file with the database configuration.
**Citations:** Should cite the specific file and line range.
**FR covered:** FR-7, FR-8

### Q2: Cross-file understanding — "How does authentication work?"

**Expected answer:** Explains the auth flow across `auth/login.py` and related files.
**Citations:** Should cite multiple chunks.
**FR covered:** FR-7, FR-8

### Q3: Follow-up (pronoun resolution) — "How do I change it to use MySQL instead?"

**Expected answer:** Refers back to the database connection from Q1 and explains how to modify it for MySQL.
**Citations:** Should cite relevant configuration or connection files.
**FR covered:** FR-10

### Q4: Honesty path — "How does the notification system work?"

**Expected answer:** "I couldn't find information about 'notification system'..." since the codebase likely has no notification feature.
**Citations:** None.
**FR covered:** FR-9

### Q5: Code understanding — "What environment variables does the application use?"

**Expected answer:** Lists DATABASE_URL, DB_POOL_SIZE, DB_TIMEOUT, ANTHROPIC_API_KEY, VOYAGE_API_KEY.
**Citations:** Should cite relevant config files.
**FR covered:** FR-7, FR-8
