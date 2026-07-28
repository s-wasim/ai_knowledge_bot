"""GET /health.

Health reports the embedding model's state explicitly. The predecessor reported
only `mode`, which flipped to full-text search whenever an API key was missing —
so a silently degraded deployment looked identical to a healthy one.
"""

from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from app.api import deps
from app.server import app


class TestHealth:
    def setup_method(self):
        deps.reset_singletons()

    def teardown_method(self):
        deps.reset_singletons()

    @patch("app.retrieval.factory.is_embedding_available", return_value=True)
    @patch("app.api.health.embedding_status")
    @patch("app.api.health.get_session")
    def test_health_ok_in_hybrid_mode(self, mock_get_session, mock_status, _available):
        mock_get_session.return_value = MagicMock()
        mock_status.return_value = {
            "ok": True,
            "model": "jinaai/jina-embeddings-v2-base-code",
            "dims": 768,
            "error": None,
        }

        res = TestClient(app).get("/health")

        assert res.status_code == 200
        body = res.json()
        assert body["db_ok"] is True
        assert body["db_error"] is None
        assert body["mode"] == "hybrid"
        assert "embeddings" in body["mode_display"]
        assert body["embed_model_ok"] is True
        assert body["embed_dims"] == 768
        assert body["embed_model"] == "jinaai/jina-embeddings-v2-base-code"

    @patch("app.retrieval.factory.is_embedding_available", return_value=False)
    @patch("app.api.health.embedding_status")
    @patch("app.api.health.get_session")
    def test_health_names_the_degradation(self, mock_get_session, mock_status, _available):
        mock_get_session.return_value = MagicMock()
        mock_status.return_value = {
            "ok": False,
            "model": "jinaai/jina-embeddings-v2-base-code",
            "dims": 768,
            "error": "weights not found",
        }

        body = TestClient(app).get("/health").json()

        assert body["mode"] == "degraded"
        assert body["embed_model_ok"] is False
        assert body["embed_error"] == "weights not found"
        assert "no embeddings" in body["mode_display"]

    @patch("app.retrieval.factory.is_embedding_available", return_value=False)
    @patch("app.api.health.get_session")
    def test_health_reports_db_error(self, mock_get_session, _available):
        broken_session = MagicMock()
        broken_session.execute.side_effect = RuntimeError("connection refused")
        mock_get_session.return_value = broken_session

        res = TestClient(app).get("/health")

        assert res.status_code == 200
        body = res.json()
        assert body["db_ok"] is False
        assert "connection refused" in body["db_error"]
