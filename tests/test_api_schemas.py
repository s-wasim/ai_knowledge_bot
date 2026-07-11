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
