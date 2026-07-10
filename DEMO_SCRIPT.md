# AI Knowledge Bot — Demo Script

**Prep time:** <5 min (after initial `docker compose up`)
**Demo time:** ~10 min
**Prerequisites:** Docker, ANTHROPIC_API_KEY set in `.env`

---

## Click-path walkthrough

### 0. Setup

```bash
cd ai_knowledge_bot
cp .env.example .env
# Edit .env to add your ANTHROPIC_API_KEY (and optionally VOYAGE_API_KEY)
docker compose up
```

Wait for both containers to be healthy (app at http://localhost:8501).

### 1. Ingest tab (📦)

1. Select "Local Folder" source
2. Enter path: `./fixtures/mini_repo`
3. Repo name auto-fills to `mini_repo`
4. Click **Ingest**
5. ✅ Progress bar fills, summary card shows file/chunk counts
6. **Talking point:** "Notice the mode badge shows Full-text search — the app works with just an Anthropic key"

### 2. Chat tab (💬)

1. Switch to the Chat tab
2. ✅ Repo is auto-selected in the dropdown
3. Type: "Where is the database connection configured?"
4. ✅ Answer streams with citation [1]
5. Click the citation to expand the source chunk
6. ✅ Source code displayed with line numbers
7. Expand "Retrieved chunks & grades" to see grading
8. **Talking point:** "The grading shows which chunks were considered relevant — complete transparency"

### 3. Follow-up question

1. Type: "How do I change it to use MySQL instead?"
2. ✅ Answer still references the database connection (follow-up resolution)
3. **Talking point:** "The rewrite step folded the conversation history into a standalone query"

### 4. Honesty path

1. Type: "How does the notification system work?"
2. ✅ Bot responds it can't find the answer and suggests where to look
3. **Talking point:** "The bot never fabricates — if it doesn't know, it says so"

### 5. Index Browser (🗂)

1. Switch to the Index Browser tab
2. ✅ Repo selected, file list with chunk counts shown
3. Search for "database" in the keyword filter
4. ✅ Only db.py shown
5. Expand a file to see individual chunks
6. **Talking point:** "Full visibility into what the bot knows"

### 6. (Optional) GitHub ingest

1. Go back to Ingest tab
2. Select "GitHub URL"
3. Enter a public repo URL
4. Click **Ingest from GitHub**
5. ✅ Progress bar, summary card, repo name from GitHub

---

## Talking points summary

- **Key value:** Instant codebase onboarding for new engineers
- **Transparency:** Every retrieved chunk visible with relevance grade
- **Safety:** Citation enforcement + honesty path prevent hallucinations
- **Zero-config fallback:** Works with just an Anthropic key (FTS mode)
- **Vector upgrade:** Add VOYAGE_API_KEY for semantic search
