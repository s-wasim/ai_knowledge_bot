from unittest.mock import MagicMock, patch

from app.graph.state import RagState
from app.retrieval.base import ChunkData


@patch("app.graph.nodes.grade.get_llm")
def test_single_llm_call(mock_get_llm):
    mock_llm = MagicMock()
    mock_llm.invoke.return_value.content = (
        '{"grades": [{"index": 1, "keep": true, "reason": "relevant"}]}'
    )
    mock_get_llm.return_value = mock_llm

    from app.graph.nodes.grade import grade_chunks

    state = RagState(
        question="test",
        chat_history=[],
        rewritten_query="test query",
        retrieved=[ChunkData(path="test.py", start_line=1, end_line=10, content="test content", score=0.9)],
        graded=[],
        answer=None,
        citations=[],
        mode="fts",
        repo_id=1,
    )

    result = grade_chunks(state)
    mock_llm.invoke.assert_called_once()
    assert len(result["graded"]) == 1
    assert result["graded"][0].keep is True


@patch("app.graph.nodes.grade.get_llm")
def test_all_kept(mock_get_llm):
    mock_llm = MagicMock()
    mock_llm.invoke.return_value.content = (
        '{"grades": [{"index": 1, "keep": true, "reason": "relevant"}, '
        '{"index": 2, "keep": true, "reason": "also relevant"}]}'
    )
    mock_get_llm.return_value = mock_llm

    from app.graph.nodes.grade import grade_chunks

    state = RagState(
        question="test",
        chat_history=[],
        rewritten_query="test query",
        retrieved=[
            ChunkData(path="a.py", start_line=1, end_line=10, content="aaa", score=0.9),
            ChunkData(path="b.py", start_line=1, end_line=10, content="bbb", score=0.8),
        ],
        graded=[],
        answer=None,
        citations=[],
        mode="fts",
        repo_id=1,
    )

    result = grade_chunks(state)
    assert len(result["graded"]) == 2
    assert all(g.keep for g in result["graded"])


@patch("app.graph.nodes.grade.get_llm")
def test_some_discarded(mock_get_llm):
    mock_llm = MagicMock()
    mock_llm.invoke.return_value.content = (
        '{"grades": [{"index": 1, "keep": true, "reason": "relevant"}, '
        '{"index": 2, "keep": false, "reason": "irrelevant"}]}'
    )
    mock_get_llm.return_value = mock_llm

    from app.graph.nodes.grade import grade_chunks

    state = RagState(
        question="test",
        chat_history=[],
        rewritten_query="test query",
        retrieved=[
            ChunkData(path="a.py", start_line=1, end_line=10, content="aaa", score=0.9),
            ChunkData(path="b.py", start_line=1, end_line=10, content="bbb", score=0.8),
        ],
        graded=[],
        answer=None,
        citations=[],
        mode="fts",
        repo_id=1,
    )

    result = grade_chunks(state)
    assert len(result["graded"]) == 2
    assert result["graded"][0].keep is True
    assert result["graded"][1].keep is False


@patch("app.graph.nodes.grade.get_llm")
def test_empty_retrieved(mock_get_llm):
    from app.graph.nodes.grade import grade_chunks

    state = RagState(
        question="test",
        chat_history=[],
        rewritten_query="test query",
        retrieved=[],
        graded=[],
        answer=None,
        citations=[],
        mode="fts",
        repo_id=1,
    )

    result = grade_chunks(state)
    assert result["graded"] == []
    mock_get_llm.assert_not_called()


@patch("app.graph.nodes.grade.get_llm")
def test_list_content_response_falls_back(mock_get_llm):
    """response.content can arrive as list-shaped content blocks (e.g. Claude
    extended thinking) instead of a plain str; grade_chunks must not crash."""
    mock_llm = MagicMock()
    mock_llm.invoke.return_value.content = [
        {"type": "thinking", "thinking": "reasoning...", "signature": "abc"}
    ]
    mock_get_llm.return_value = mock_llm

    from app.graph.nodes.grade import grade_chunks

    state = RagState(
        question="test",
        chat_history=[],
        rewritten_query="test query",
        retrieved=[ChunkData(path="test.py", start_line=1, end_line=10, content="test content", score=0.9)],
        graded=[],
        answer=None,
        citations=[],
        mode="fts",
        repo_id=1,
    )

    result = grade_chunks(state)
    assert len(result["graded"]) == 1
    assert result["graded"][0].keep is True


@patch("app.graph.nodes.grade.get_llm")
def test_malformed_response(mock_get_llm):
    mock_llm = MagicMock()
    mock_llm.invoke.return_value.content = "not valid json at all"
    mock_get_llm.return_value = mock_llm

    from app.graph.nodes.grade import grade_chunks

    state = RagState(
        question="test",
        chat_history=[],
        rewritten_query="test query",
        retrieved=[ChunkData(path="test.py", start_line=1, end_line=10, content="test content", score=0.9)],
        graded=[],
        answer=None,
        citations=[],
        mode="fts",
        repo_id=1,
    )

    result = grade_chunks(state)
    assert len(result["graded"]) == 1
    assert result["graded"][0].keep is True
    assert "Fallback after error" in result["graded"][0].reason
