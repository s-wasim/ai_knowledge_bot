"""Local embedding model.

The model runs in-process with no API key. Every failure mode must degrade to
`None` rather than raising, because retrieval is designed to fall back to lexical
and symbolic search when embeddings are unavailable — but only if it is told.
"""

from unittest.mock import MagicMock, patch

import pytest

import app.ingest.embedder as embedder_module
from app.ingest.embedder import (
    EMBED_DIMS,
    MODEL_NAME,
    embed_query,
    embed_texts,
    embedding_status,
    is_embedding_available,
)


def _fake_model(vectors):
    model = MagicMock()
    model.encode.return_value = vectors
    return model


@pytest.fixture(autouse=True)
def _clear_model_cache():
    """The module caches the model process-wide, including load failures, so every
    test needs a clean slate. Autouse so it covers class-based tests too."""
    embedder_module._reset_cache()
    yield
    embedder_module._reset_cache()


class TestConfiguration:
    def test_model_is_the_open_source_code_model(self):
        assert MODEL_NAME == "jinaai/jina-embeddings-v2-base-code"

    def test_dimensions_match_the_database_column(self):
        from app.db import EMBED_DIMS as DB_DIMS

        assert EMBED_DIMS == DB_DIMS == 768


class TestEmbedTexts:
    @patch("app.ingest.embedder.get_model")
    def test_batches_requests(self, mock_get_model):
        model = MagicMock()
        model.encode.side_effect = [[[1.0, 0.0]], [[0.0, 1.0]]]
        mock_get_model.return_value = model

        out = embed_texts(["a", "b"], batch_size=1)

        assert out == [[1.0, 0.0], [0.0, 1.0]]
        assert model.encode.call_count == 2

    @patch("app.ingest.embedder.get_model")
    def test_requests_normalized_embeddings(self, mock_get_model):
        """Normalizing at write time means cosine distance and inner product
        agree, so pgvector ordering is stable."""
        mock_get_model.return_value = _fake_model([[1.0, 0.0]])

        embed_texts(["a"])

        kwargs = mock_get_model.return_value.encode.call_args.kwargs
        assert kwargs["normalize_embeddings"] is True

    @patch("app.ingest.embedder.get_model")
    def test_returns_plain_python_floats(self, mock_get_model):
        """psycopg cannot adapt numpy scalars, so conversion must happen here."""
        import array

        mock_get_model.return_value = _fake_model([array.array("f", [0.5, 0.25])])

        out = embed_texts(["a"])

        assert out == [[0.5, 0.25]]
        assert all(isinstance(v, float) for v in out[0])

    @patch("app.ingest.embedder.get_model", return_value=None)
    def test_returns_none_when_model_unavailable(self, _):
        assert embed_texts(["a"]) is None

    @patch("app.ingest.embedder.get_model")
    def test_returns_none_when_encoding_raises(self, mock_get_model):
        model = MagicMock()
        model.encode.side_effect = RuntimeError("out of memory")
        mock_get_model.return_value = model

        assert embed_texts(["a"]) is None

    def test_empty_input_returns_empty_list_without_loading_model(self):
        assert embed_texts([]) == []


class TestEmbedQuery:
    @patch("app.ingest.embedder.get_model")
    def test_returns_a_single_vector(self, mock_get_model):
        mock_get_model.return_value = _fake_model([[0.1, 0.2]])

        assert embed_query("where is the db") == [0.1, 0.2]

    @patch("app.ingest.embedder.get_model", return_value=None)
    def test_returns_none_when_unavailable(self, _):
        assert embed_query("x") is None

    @patch("app.ingest.embedder.get_model")
    def test_blank_query_returns_none(self, mock_get_model):
        assert embed_query("   ") is None
        mock_get_model.assert_not_called()


class TestStatus:
    @patch("app.ingest.embedder.get_model", return_value=None)
    def test_reports_not_ok_when_model_missing(self, _):
        status = embedding_status()
        assert status["ok"] is False
        assert status["model"] == MODEL_NAME
        assert status["dims"] == EMBED_DIMS
        assert is_embedding_available() is False

    @patch("app.ingest.embedder.get_model")
    def test_reports_ok_when_model_loads(self, mock_get_model):
        mock_get_model.return_value = MagicMock()
        status = embedding_status()
        assert status["ok"] is True
        assert status["error"] is None
        assert is_embedding_available() is True

    def test_surfaces_the_load_error(self):
        """A silent fallback to lexical-only search is the failure mode this
        project already suffered with Voyage. The reason must be reportable."""
        with patch.object(
            embedder_module, "_load_model", side_effect=OSError("weights not found")
        ):
            assert embedder_module.get_model() is None
            assert "weights not found" in embedding_status()["error"]


class TestModelCaching:
    def test_model_is_loaded_once(self):
        sentinel = MagicMock()
        with patch.object(embedder_module, "_load_model", return_value=sentinel) as load:
            assert embedder_module.get_model() is sentinel
            assert embedder_module.get_model() is sentinel
            assert load.call_count == 1

    def test_failed_load_is_not_retried_on_every_call(self):
        """Reloading a 300MB model on every request would turn one failure into a
        stall on every query."""
        with patch.object(
            embedder_module, "_load_model", side_effect=OSError("nope")
        ) as load:
            embedder_module.get_model()
            embedder_module.get_model()
            assert load.call_count == 1
