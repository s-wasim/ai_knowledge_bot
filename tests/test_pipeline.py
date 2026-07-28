"""Ingest pipeline.

The behaviour that matters most here is path storage. Chunk paths are the evidence
a user checks an answer against, and they were previously absolute — for GitHub
ingests, absolute into a temp directory that gets deleted, so every citation
pointed at a path that no longer existed.
"""

from unittest.mock import MagicMock, patch

import pytest

from app.db import Chunk, Repo


@pytest.fixture
def session_capture():
    """A mock session that records added objects and assigns repo ids."""
    session = MagicMock()
    added: list = []

    def add(obj):
        added.append(obj)
        if isinstance(obj, Repo):
            obj.id = 1

    session.add.side_effect = add
    session.query.return_value.filter_by.return_value.first.return_value = None
    session.added = added
    return session


@pytest.fixture
def repo_dir(tmp_path):
    """A small real directory tree, so path handling is exercised for real."""
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "mod.py").write_text("def alpha():\n    return 1\n")
    (tmp_path / "top.py").write_text("VALUE = 1\n")
    return tmp_path


@pytest.fixture(autouse=True)
def _no_embeddings():
    """Embedding has its own tests; keep these runs model-free by default."""
    with patch("app.ingest.pipeline.is_embedding_available", return_value=False):
        yield


def _chunks_of(session):
    return [o for o in session.added if isinstance(o, Chunk)]


class TestPathStorage:
    def test_paths_are_relative_to_the_ingest_root(self, session_capture, repo_dir):
        with patch("app.ingest.pipeline.get_session", return_value=session_capture):
            from app.ingest.pipeline import ingest_repo

            ingest_repo(repo_name="r", root_dir=str(repo_dir))

        paths = {c.path for c in _chunks_of(session_capture)}
        assert paths == {"pkg/mod.py", "top.py"}

    def test_no_stored_path_is_absolute(self, session_capture, repo_dir):
        with patch("app.ingest.pipeline.get_session", return_value=session_capture):
            from app.ingest.pipeline import ingest_repo

            ingest_repo(repo_name="r", root_dir=str(repo_dir))

        for chunk in _chunks_of(session_capture):
            assert not chunk.path.startswith("/")
            assert str(repo_dir) not in chunk.path

    def test_progress_filenames_are_relative(self, session_capture, repo_dir):
        seen = []
        with patch("app.ingest.pipeline.get_session", return_value=session_capture):
            from app.ingest.pipeline import ingest_repo

            ingest_repo(
                repo_name="r",
                root_dir=str(repo_dir),
                progress_callback=lambda c, t, n, f: seen.append(f),
            )

        assert seen
        assert all(not f.startswith("/") for f in seen)

    def test_symbol_and_language_are_persisted(self, session_capture, repo_dir):
        with patch("app.ingest.pipeline.get_session", return_value=session_capture):
            from app.ingest.pipeline import ingest_repo

            ingest_repo(repo_name="r", root_dir=str(repo_dir))

        chunks = _chunks_of(session_capture)
        assert any(c.symbol == "alpha" for c in chunks)
        assert all(c.language == "python" for c in chunks)


class TestProgress:
    def test_progress_reports_the_real_total(self, session_capture, repo_dir):
        """The UI renders current/total, and total was once hardcoded to 20."""
        calls = []
        with patch("app.ingest.pipeline.get_session", return_value=session_capture):
            from app.ingest.pipeline import ingest_repo

            ingest_repo(
                repo_name="r",
                root_dir=str(repo_dir),
                progress_callback=lambda c, t, n, f: calls.append((c, t)),
            )

        assert [c for c, _ in calls] == [1, 2]
        assert all(total == 2 for _, total in calls)

    def test_counts_are_recorded_on_the_repo(self, session_capture, repo_dir):
        with patch("app.ingest.pipeline.get_session", return_value=session_capture):
            from app.ingest.pipeline import ingest_repo

            repo = ingest_repo(repo_name="r", root_dir=str(repo_dir))

        assert repo.file_count == 2
        assert repo.chunk_count == len(_chunks_of(session_capture))


class TestChunkCap:
    def test_cap_stops_ingest_and_reports(self, session_capture, repo_dir, monkeypatch):
        monkeypatch.setenv("KB_MAX_CHUNKS", "1")
        warnings = []

        with patch("app.ingest.pipeline.get_session", return_value=session_capture):
            from app.ingest.pipeline import ingest_repo

            ingest_repo(
                repo_name="r",
                root_dir=str(repo_dir),
                warning_callback=warnings.append,
            )

        assert len(_chunks_of(session_capture)) == 1
        assert warnings and "cap" in warnings[0].lower()

    def test_invalid_cap_falls_back_to_the_default(self, monkeypatch):
        from app.ingest.pipeline import DEFAULT_MAX_CHUNKS, max_chunks_per_repo

        monkeypatch.setenv("KB_MAX_CHUNKS", "not-a-number")
        assert max_chunks_per_repo() == DEFAULT_MAX_CHUNKS

        monkeypatch.setenv("KB_MAX_CHUNKS", "0")
        assert max_chunks_per_repo() == DEFAULT_MAX_CHUNKS

    def test_cap_is_read_from_the_environment(self, monkeypatch):
        from app.ingest.pipeline import max_chunks_per_repo

        monkeypatch.setenv("KB_MAX_CHUNKS", "5")
        assert max_chunks_per_repo() == 5


class TestReingest:
    def test_reingest_deletes_the_previous_repo_row(self, session_capture, repo_dir):
        existing = MagicMock(spec=Repo)
        existing.id = 1
        session_capture.query.return_value.filter_by.return_value.first.return_value = existing

        with patch("app.ingest.pipeline.get_session", return_value=session_capture):
            from app.ingest.pipeline import ingest_repo

            ingest_repo(repo_name="r", root_dir=str(repo_dir))

        session_capture.delete.assert_called_once_with(existing)


class TestFailures:
    def test_missing_directory_raises(self, session_capture):
        with patch("app.ingest.pipeline.get_session", return_value=session_capture):
            from app.ingest.pipeline import ingest_repo

            with pytest.raises(FileNotFoundError):
                ingest_repo(repo_name="r", root_dir="/nonexistent-path-xyz")

        session_capture.rollback.assert_called_once()
        session_capture.commit.assert_not_called()

    def test_empty_directory_commits_an_empty_repo(self, session_capture, tmp_path):
        with patch("app.ingest.pipeline.get_session", return_value=session_capture):
            from app.ingest.pipeline import ingest_repo

            repo = ingest_repo(repo_name="r", root_dir=str(tmp_path))

        assert repo.chunk_count == 0
        session_capture.commit.assert_called_once()


class TestEmbedding:
    def test_unavailable_model_warns_and_indexes_without_vectors(
        self, session_capture, repo_dir
    ):
        """Documented degraded mode: keep the content, say so out loud."""
        warnings = []
        with patch("app.ingest.pipeline.get_session", return_value=session_capture):
            from app.ingest.pipeline import ingest_repo

            ingest_repo(
                repo_name="r", root_dir=str(repo_dir), warning_callback=warnings.append
            )

        assert warnings and "without vectors" in warnings[0]
        assert all(c.embedding is None for c in _chunks_of(session_capture))
        session_capture.commit.assert_called_once()

    def test_embedding_failure_fails_the_ingest(self, session_capture, repo_dir):
        """A half-embedded index would keep reporting healthy hybrid search while
        its dense half silently returned nothing for this repo (FR-5)."""
        with patch(
            "app.ingest.pipeline.get_session", return_value=session_capture
        ), patch(
            "app.ingest.pipeline.is_embedding_available", return_value=True
        ), patch(
            "app.ingest.pipeline.embed_texts", return_value=None
        ):
            from app.ingest.pipeline import ingest_repo

            with pytest.raises(RuntimeError, match="Embedding failed"):
                ingest_repo(repo_name="r", root_dir=str(repo_dir))

        session_capture.rollback.assert_called_once()
        session_capture.commit.assert_not_called()

    def test_vector_count_mismatch_fails_the_ingest(self, session_capture, repo_dir):
        with patch(
            "app.ingest.pipeline.get_session", return_value=session_capture
        ), patch(
            "app.ingest.pipeline.is_embedding_available", return_value=True
        ), patch(
            "app.ingest.pipeline.embed_texts", return_value=[[0.1] * 768]
        ):
            from app.ingest.pipeline import ingest_repo

            with pytest.raises(RuntimeError, match="vectors for"):
                ingest_repo(repo_name="r", root_dir=str(repo_dir))

    def test_successful_embedding_is_attached(self, session_capture, repo_dir):
        vectors = [[0.1] * 768, [0.2] * 768]
        with patch(
            "app.ingest.pipeline.get_session", return_value=session_capture
        ), patch(
            "app.ingest.pipeline.is_embedding_available", return_value=True
        ), patch(
            "app.ingest.pipeline.embed_texts", return_value=vectors
        ):
            from app.ingest.pipeline import ingest_repo

            ingest_repo(repo_name="r", root_dir=str(repo_dir))

        chunks = _chunks_of(session_capture)
        assert len(chunks) == 2
        assert all(c.embedding is not None for c in chunks)
