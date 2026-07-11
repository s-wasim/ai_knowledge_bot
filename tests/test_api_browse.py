from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from app.server import app


class FakeRow:
    def __init__(self, path, chunk_count, start_line, end_line):
        self.path = path
        self.chunk_count = chunk_count
        self.start_line = start_line
        self.end_line = end_line


class TestBrowse:
    @patch("app.api.browse.get_session")
    def test_returns_metrics_and_files_for_existing_repo(self, mock_get_session):
        repo = MagicMock()
        repo.file_count = 3
        repo.chunk_count = 10
        repo.source_url = "https://github.com/acme/repo"

        rows = [FakeRow("app/main.py", 2, 1, 20)]

        mock_session = MagicMock()
        mock_session.query.return_value.filter_by.return_value.first.return_value = repo
        query_chain = mock_session.query.return_value
        query_chain.filter.return_value.group_by.return_value.order_by.return_value.all.return_value = rows
        mock_get_session.return_value = mock_session

        client = TestClient(app)
        res = client.get("/repos/1/browse")

        assert res.status_code == 200
        body = res.json()
        assert body["metrics"]["files"] == 3
        assert body["metrics"]["chunks"] == 10
        assert body["metrics"]["source"] == "GitHub"
        assert body["files"][0]["path"] == "app/main.py"

    @patch("app.api.browse.get_session")
    def test_missing_repo_returns_zeroed_metrics(self, mock_get_session):
        mock_session = MagicMock()
        mock_session.query.return_value.filter_by.return_value.first.return_value = None
        query_chain = mock_session.query.return_value
        query_chain.filter.return_value.group_by.return_value.order_by.return_value.all.return_value = []
        mock_get_session.return_value = mock_session

        client = TestClient(app)
        res = client.get("/repos/999/browse")

        assert res.status_code == 200
        body = res.json()
        assert body["metrics"]["files"] == 0
        assert body["metrics"]["source"] == "Local"
        assert body["files"] == []


class TestBrowseFile:
    @patch("app.api.browse.get_session")
    def test_returns_chunks_ordered_by_start_line(self, mock_get_session):
        chunk = MagicMock()
        chunk.start_line = 1
        chunk.end_line = 10
        chunk.content = "print('hi')"

        mock_session = MagicMock()
        query_chain = mock_session.query.return_value
        query_chain.filter.return_value.order_by.return_value.all.return_value = [chunk]
        mock_get_session.return_value = mock_session

        client = TestClient(app)
        res = client.get("/repos/1/files", params={"path": "app/main.py"})

        assert res.status_code == 200
        body = res.json()
        assert body[0]["start_line"] == 1
        assert body[0]["content"] == "print('hi')"
