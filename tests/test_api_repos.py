from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from app.server import app


class TestListRepos:
    @patch("app.api.repos.get_session")
    def test_returns_repos_sorted_newest_first(self, mock_get_session):
        repo = MagicMock()
        repo.id = 1
        repo.name = "my_repo"
        repo.file_count = 5
        repo.chunk_count = 40
        repo.source_url = None
        repo.ingested_at = datetime(2026, 7, 10, 9, 0, tzinfo=timezone.utc)

        mock_session = MagicMock()
        mock_session.query.return_value.order_by.return_value.all.return_value = [repo]
        mock_get_session.return_value = mock_session

        client = TestClient(app)
        res = client.get("/repos")

        assert res.status_code == 200
        body = res.json()
        assert body[0]["id"] == 1
        assert body[0]["name"] == "my_repo"
        assert body[0]["ingested_at"].startswith("2026-07-10T09:00:00")


class TestAllowlist:
    def test_returns_sorted_allowlist(self):
        client = TestClient(app)
        res = client.get("/config/allowlist")

        assert res.status_code == 200
        body = res.json()
        assert body == sorted(body)
        assert ".py" in body
