# tests/test_api_ingest.py
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from app.server import app


class TestIngestLocal:
    def test_missing_directory_emits_error_event(self, tmp_path):
        client = TestClient(app)
        res = client.post(
            "/ingest/local",
            json={"path": str(tmp_path / "does_not_exist"), "name": "x"},
        )

        assert res.status_code == 200
        assert "event: error" in res.text
        assert "Directory not found" in res.text

    @patch("app.api.ingest.ingest_repo")
    def test_success_emits_progress_then_done(self, mock_ingest_repo, tmp_path):
        def fake_ingest(repo_name, root_dir, progress_callback=None, warning_callback=None):
            progress_callback(1, 2, 3, "a.py")
            progress_callback(2, 2, 2, "b.py")
            repo = MagicMock()
            repo.id = 7
            repo.name = repo_name
            repo.file_count = 2
            repo.chunk_count = 5
            return repo

        mock_ingest_repo.side_effect = fake_ingest

        client = TestClient(app)
        res = client.post(
            "/ingest/local", json={"path": str(tmp_path), "name": "my_repo"}
        )

        assert res.status_code == 200
        assert res.text.count("event: progress") == 2
        assert '"total": 2' in res.text
        assert "event: done" in res.text
        assert '"name": "my_repo"' in res.text
        assert '"source": "Local"' in res.text

    @patch("app.api.ingest.ingest_repo")
    def test_exception_emits_error_event(self, mock_ingest_repo, tmp_path):
        mock_ingest_repo.side_effect = RuntimeError("boom")

        client = TestClient(app)
        res = client.post(
            "/ingest/local", json={"path": str(tmp_path), "name": "x"}
        )

        assert res.status_code == 200
        assert "event: error" in res.text
        assert "boom" in res.text


class TestIngestGithub:
    @patch("app.api.ingest.ingest_github_url")
    def test_success_emits_done_with_github_source(self, mock_ingest_github_url):
        def fake_ingest(url, branch=None, progress_callback=None, warning_callback=None):
            progress_callback(1, 1, 1, "README.md")
            repo = MagicMock()
            repo.id = 9
            repo.name = "repo"
            repo.file_count = 1
            repo.chunk_count = 1
            return repo

        mock_ingest_github_url.side_effect = fake_ingest

        client = TestClient(app)
        res = client.post(
            "/ingest/github", json={"url": "https://github.com/acme/repo"}
        )

        assert res.status_code == 200
        assert "event: progress" in res.text
        assert '"total": 1' in res.text
        assert "event: done" in res.text
        assert '"source": "GitHub"' in res.text

    @patch("app.api.ingest.ingest_github_url")
    def test_invalid_url_emits_error_event(self, mock_ingest_github_url):
        mock_ingest_github_url.side_effect = ValueError(
            "Invalid GitHub URL: not-a-url. Expected https://github.com/owner/repo"
        )

        client = TestClient(app)
        res = client.post("/ingest/github", json={"url": "not-a-url"})

        assert res.status_code == 200
        assert "event: error" in res.text
        assert "Invalid GitHub URL" in res.text


class TestWarningFrames:
    """Chunk caps and missing embeddings reach the client as `warning` frames.
    Logging them alone would let a partially indexed repo look complete."""

    @patch("app.api.ingest.ingest_repo")
    def test_warnings_are_streamed(self, mock_ingest_repo, tmp_path):
        def fake_ingest(repo_name, root_dir, progress_callback=None, warning_callback=None):
            warning_callback("Chunk cap of 1 reached after 1 of 9 files.")
            progress_callback(1, 9, 1, "a.py")
            repo = MagicMock()
            repo.id, repo.name, repo.file_count, repo.chunk_count = 1, "r", 1, 1
            return repo

        mock_ingest_repo.side_effect = fake_ingest

        client = TestClient(app)
        res = client.post("/ingest/local", json={"path": str(tmp_path), "name": "r"})

        assert res.status_code == 200
        assert "event: warning" in res.text
        assert "Chunk cap" in res.text
        assert "event: done" in res.text
