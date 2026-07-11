from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from app.api import deps
from app.server import app


class TestHealth:
    def setup_method(self):
        deps.reset_singletons()

    @patch("app.api.health.get_session")
    def test_health_ok(self, mock_get_session, monkeypatch):
        monkeypatch.delenv("VOYAGE_API_KEY", raising=False)
        mock_get_session.return_value = MagicMock()

        client = TestClient(app)
        res = client.get("/health")

        assert res.status_code == 200
        body = res.json()
        assert body["db_ok"] is True
        assert body["db_error"] is None
        assert body["mode"] == "fts"
        assert body["mode_display"] == "Full-text search"

    @patch("app.api.health.get_session")
    def test_health_reports_db_error(self, mock_get_session, monkeypatch):
        monkeypatch.delenv("VOYAGE_API_KEY", raising=False)
        broken_session = MagicMock()
        broken_session.execute.side_effect = RuntimeError("connection refused")
        mock_get_session.return_value = broken_session

        client = TestClient(app)
        res = client.get("/health")

        assert res.status_code == 200
        body = res.json()
        assert body["db_ok"] is False
        assert "connection refused" in body["db_error"]
