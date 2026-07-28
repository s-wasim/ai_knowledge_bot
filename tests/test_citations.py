from unittest.mock import MagicMock, patch

from app.graph.state import RagState, GradedChunk, Citation
from app.retrieval.base import ChunkData
from app.graph.nodes.answer import _postprocess_answer


def test_valid_citation_preserved():
    kept = [
        GradedChunk(chunk=ChunkData(path="db.py", start_line=1, end_line=10, content="x", score=0.9), keep=True, reason="r"),
        GradedChunk(chunk=ChunkData(path="config.py", start_line=5, end_line=15, content="y", score=0.8), keep=True, reason="r"),
    ]
    answer = "The DB is in [1] and config is in [2]."
    cleaned, citations = _postprocess_answer(answer, kept)
    assert "[1]" in cleaned
    assert "[2]" in cleaned
    assert len(citations) == 2


def test_invalid_citation_stripped():
    kept = [
        GradedChunk(chunk=ChunkData(path="db.py", start_line=1, end_line=10, content="x", score=0.9), keep=True, reason="r"),
    ]
    answer = "The DB is in [1] and also see [5]."
    cleaned, citations = _postprocess_answer(answer, kept)
    assert "[1]" in cleaned
    assert "[5]" not in cleaned
    assert len(citations) == 1


def test_no_citations():
    kept = [GradedChunk(chunk=ChunkData(path="db.py", start_line=1, end_line=10, content="x", score=0.9), keep=True, reason="r")]
    answer = "I don't know."
    cleaned, citations = _postprocess_answer(answer, kept)
    assert cleaned == answer
    assert citations == []


def test_not_found_message():
    from app.graph.nodes.not_found import answer_not_found
    state = RagState(
        question="nonexistent feature", chat_history=[], rewritten_query="nonexistent feature",
        retrieved=[ChunkData(path="auth/login.py", start_line=1, end_line=5, content="x", score=0.5)],
        graded=[], answer=None, citations=[], mode="hybrid", repo_id=1,
    )
    result = answer_not_found(state)
    assert "couldn't find" in result["answer"]
    assert result["citations"] == []


def test_not_found_falls_back_without_config():
    """No config/get_session available -> generic fallback, never raises (FR-9)."""
    from app.graph.nodes.not_found import answer_not_found
    state = RagState(
        question="nonexistent feature", chat_history=[], rewritten_query="nonexistent feature",
        retrieved=[], graded=[], answer=None, citations=[], mode="hybrid", repo_id=1,
    )
    result = answer_not_found(state)
    assert "couldn't find" in result["answer"]
    assert "rephrasing" in result["answer"]
    # No blank suggestion artifact. Absolute paths used to yield an empty leading
    # segment, rendering "look in  for relevant files".
    assert "  " not in result["answer"]
    assert "look in ," not in result["answer"]


def test_not_found_suggests_top_two_directories_from_index():
    """FR-9: suggestions come from the whole index, not from discarded chunks."""
    from app.db import Chunk
    from app.graph.nodes.not_found import answer_not_found

    mock_session = MagicMock()
    mock_session.query.return_value.filter.return_value.all.return_value = [
        ("auth/login.py",), ("auth/session.py",), ("auth/tokens.py",),
        ("db/models.py",), ("db/queries.py",),
        ("utils/helpers.py",),
    ]

    state = RagState(
        question="nonexistent feature", chat_history=[], rewritten_query="nonexistent feature",
        retrieved=[ChunkData(path="totally/unrelated.py", start_line=1, end_line=5, content="x", score=0.1)],
        graded=[], answer=None, citations=[], mode="hybrid", repo_id=1,
    )
    config = {"configurable": {"get_session": lambda: mock_session}}
    result = answer_not_found(state, config)

    assert "auth" in result["answer"]
    assert "db" in result["answer"]
    assert "totally" not in result["answer"]
    mock_session.close.assert_called_once()


def test_generate_answer_no_kept_chunks():
    from app.graph.nodes.answer import generate_answer
    state = RagState(
        question="test", chat_history=[], rewritten_query="test",
        retrieved=[], graded=[], answer=None, citations=[], mode="hybrid", repo_id=1,
    )
    result = generate_answer(state)
    assert "couldn't find" in result["answer"]


@patch("app.graph.nodes.answer.get_llm_streaming")
def test_generate_answer_uses_streaming_call(mock_get_llm_streaming):
    """FR-8/TASK-011: generate_answer must stream tokens, not block on invoke()."""
    from app.graph.nodes.answer import generate_answer

    mock_llm = MagicMock()
    chunk1, chunk2 = MagicMock(content="The DB is in "), MagicMock(content="[1].")
    mock_llm.stream.return_value = iter([chunk1, chunk2])
    mock_get_llm_streaming.return_value = mock_llm

    state = RagState(
        question="where is the db?", chat_history=[], rewritten_query="where is the db?",
        retrieved=[], graded=[GradedChunk(
            chunk=ChunkData(path="db.py", start_line=1, end_line=10, content="x", score=0.9),
            keep=True, reason="r",
        )],
        answer=None, citations=[], mode="hybrid", repo_id=1,
    )
    result = generate_answer(state)

    mock_llm.stream.assert_called_once()
    assert result["answer"] == "The DB is in [1]."
    assert len(result["citations"]) == 1
