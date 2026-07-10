from unittest.mock import MagicMock, patch

from app.graph.nodes.rewrite import rewrite_query
from app.graph.state import RagState
from app.retrieval.base import ChunkData


@patch("app.graph.nodes.rewrite.get_llm")
def test_rewrite_with_history(mock_get_llm):
    mock_llm = MagicMock()
    mock_llm.invoke.return_value.content = "database connection setup"
    mock_get_llm.return_value = mock_llm

    state = RagState(
        question="how do I change it?",
        chat_history=[
            {"role": "user", "content": "Where is the database connection configured?"},
            {"role": "assistant", "content": "It's in db.py"},
        ],
        rewritten_query=None,
        retrieved=[],
        graded=[],
        answer=None,
        citations=[],
        mode="fts",
        repo_id=1,
    )

    result = rewrite_query(state)
    assert result["rewritten_query"] is not None
    assert "connection" in result["rewritten_query"]
    mock_llm.invoke.assert_called_once()


@patch("app.graph.nodes.rewrite.get_llm")
def test_rewrite_no_history(mock_get_llm):
    state = RagState(
        question="Where is the database connection configured?",
        chat_history=[],
        rewritten_query=None,
        retrieved=[],
        graded=[],
        answer=None,
        citations=[],
        mode="fts",
        repo_id=1,
    )
    result = rewrite_query(state)
    assert result["rewritten_query"] == state["question"]
    mock_get_llm.assert_not_called()


def test_retrieve_node():
    mock_retriever = MagicMock()
    mock_retriever.search.return_value = [
        ChunkData(path="db.py", start_line=1, end_line=10, content="...", score=0.9)
    ]

    state = RagState(
        question="test",
        chat_history=[],
        rewritten_query="database connection",
        retrieved=[],
        graded=[],
        answer=None,
        citations=[],
        mode="fts",
        repo_id=1,
    )

    from app.graph.nodes.retrieve import retrieve

    config = {"configurable": {"retriever": mock_retriever}}
    result = retrieve(state, config)
    assert len(result["retrieved"]) == 1
    assert result["retrieved"][0].path == "db.py"
    mock_retriever.search.assert_called_once_with(
        repo_id=1, query="database connection", k=8
    )


def test_retrieve_node_missing_retriever_in_config():
    state = RagState(
        question="test",
        chat_history=[],
        rewritten_query="database connection",
        retrieved=[],
        graded=[],
        answer=None,
        citations=[],
        mode="fts",
        repo_id=1,
    )

    from app.graph.nodes.retrieve import retrieve

    assert retrieve(state, {}) == {"retrieved": []}
    assert retrieve(state, None) == {"retrieved": []}
