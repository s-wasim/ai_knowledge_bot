from unittest.mock import patch, MagicMock
from app.ingest.embedder import embed_texts, is_voyage_available


@patch("app.ingest.embedder.get_voyage_client")
def test_embed_texts_batching(mock_get_client):
    mock_client = MagicMock()
    mock_client.embed.side_effect = [
        MagicMock(embeddings=[[0.1] * 1024, [0.2] * 1024]),
        MagicMock(embeddings=[[0.3] * 1024]),
    ]
    mock_get_client.return_value = mock_client

    texts = ["a", "b", "c"]
    result = embed_texts(texts, batch_size=2)
    assert result is not None
    assert len(result) == 3
    assert mock_client.embed.call_count == 2
