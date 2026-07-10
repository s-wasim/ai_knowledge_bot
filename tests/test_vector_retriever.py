from unittest.mock import patch, MagicMock
from app.retrieval.vector import VectorRetriever


@patch("app.retrieval.vector.get_voyage_client")
def test_vector_search(mock_get_client):
    mock_client = MagicMock()
    mock_client.embed.return_value.embeddings = [[0.1] * 1024]
    mock_get_client.return_value = mock_client

    mock_session = MagicMock()
    mock_result = MagicMock()
    mock_result.fetchall.return_value = []
    mock_session.execute.return_value = mock_result

    session_factory = MagicMock(return_value=mock_session)

    retriever = VectorRetriever(session_factory)
    results = retriever.search(repo_id=1, query="test", k=5)
    assert results == []
    mock_client.embed.assert_called_once()
