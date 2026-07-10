from unittest.mock import patch, MagicMock

import pytest

from app.retrieval.base import ChunkData, Retriever
from app.retrieval.vector import VectorRetriever


@patch("app.retrieval.vector.get_voyage_client")
def test_vector_search_empty_result(mock_get_client):
    """Test that search returns empty list when DB returns no rows."""
    mock_client = MagicMock()
    embed_response = MagicMock()
    embed_response.embeddings = [[0.1] * 1024]
    mock_client.embed.return_value = embed_response
    mock_get_client.return_value = mock_client

    mock_session = MagicMock()
    mock_result = MagicMock()
    mock_result.fetchall.return_value = []
    mock_session.execute.return_value = mock_result

    session_factory = MagicMock(return_value=mock_session)

    retriever = VectorRetriever(session_factory)
    results = retriever.search(repo_id=1, query="test", k=5)
    assert results == []
    mock_client.embed.assert_called_once_with(
        texts=["test"], model="voyage-code-3", input_type="query"
    )


@patch("app.retrieval.vector.get_voyage_client")
def test_query_embedding_format(mock_get_client):
    """Verify embedding is passed in pgvector-compatible format."""
    mock_client = MagicMock()
    embed_response = MagicMock()
    embed_response.embeddings = [[0.1, 0.2, 0.3]]
    mock_client.embed.return_value = embed_response
    mock_get_client.return_value = mock_client

    mock_session = MagicMock()
    mock_result = MagicMock()
    mock_result.fetchall.return_value = []
    mock_session.execute.return_value = mock_result
    session_factory = MagicMock(return_value=mock_session)

    retriever = VectorRetriever(session_factory)
    retriever.search(repo_id=1, query="test", k=5)

    # Verify the query embedding is formatted correctly with ::vector cast
    call_kwargs = mock_session.execute.call_args[0][1]
    assert "query_embedding" in call_kwargs
    assert call_kwargs["query_embedding"].startswith("[")
    assert call_kwargs["query_embedding"].endswith("]")

    # Verify SQL contains ::vector cast
    sql_text = mock_session.execute.call_args[0][0].text
    assert "::vector" in sql_text


@patch("app.retrieval.vector.get_voyage_client")
def test_empty_query(mock_get_client):
    """Empty query should return empty list without calling Voyage."""
    retriever = VectorRetriever(MagicMock())
    results = retriever.search(repo_id=1, query="", k=5)
    assert results == []
    mock_get_client.assert_not_called()


@patch("app.retrieval.vector.get_voyage_client")
def test_no_client(mock_get_client):
    """No Voyage client should return empty list."""
    mock_get_client.return_value = None
    retriever = VectorRetriever(MagicMock())
    results = retriever.search(repo_id=1, query="test", k=5)
    assert results == []
    mock_get_client.assert_called_once()


@patch("app.retrieval.vector.get_voyage_client")
def test_type_compliance(mock_get_client):
    """VectorRetriever should satisfy the Retriever protocol."""
    from app.retrieval.base import Retriever

    retriever = VectorRetriever(MagicMock())
    assert isinstance(retriever, Retriever)


@patch("app.retrieval.vector.get_voyage_client")
def test_session_closed(mock_get_client):
    """Session should be closed after search."""
    mock_client = MagicMock()
    embed_response = MagicMock()
    embed_response.embeddings = [[0.1] * 1024]
    mock_client.embed.return_value = embed_response
    mock_get_client.return_value = mock_client

    mock_session = MagicMock()
    mock_result = MagicMock()
    mock_result.fetchall.return_value = []
    mock_session.execute.return_value = mock_result
    session_factory = MagicMock(return_value=mock_session)

    retriever = VectorRetriever(session_factory)
    retriever.search(repo_id=1, query="test", k=5)
    mock_session.close.assert_called_once()
