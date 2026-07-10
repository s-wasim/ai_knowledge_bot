from unittest.mock import MagicMock, patch
import pytest
from app.db import Repo, Chunk


@patch("app.ingest.pipeline.walk_directory")
@patch("app.ingest.pipeline.chunk_file")
@patch("app.ingest.pipeline.get_session")
def test_basic_ingest(mock_get_session, mock_chunk_file, mock_walk_directory):
    """Verify pipeline creates repo and chunks."""
    mock_session = MagicMock()
    mock_get_session.return_value = mock_session

    mock_walk_directory.return_value = [
        (MagicMock(__str__=lambda x: "test.py"), "content1"),
    ]
    mock_chunk_file.return_value = [
        {"path": "test.py", "start_line": 1, "end_line": 5, "content": "content1"}
    ]

    captured_objects = []

    def add_side_effect(obj):
        captured_objects.append(obj)
        if isinstance(obj, Repo):
            obj.id = 1

    mock_session.add.side_effect = add_side_effect

    from app.ingest.pipeline import ingest_repo
    result = ingest_repo(repo_name="test_repo", root_dir="/fake/path")

    assert result is not None
    assert any(isinstance(obj, Repo) for obj in captured_objects)
    assert any(isinstance(obj, Chunk) for obj in captured_objects)


@patch("app.ingest.pipeline.walk_directory")
@patch("app.ingest.pipeline.chunk_file")
@patch("app.ingest.pipeline.get_session")
def test_reingest_replaces_chunks(mock_get_session, mock_chunk_file, mock_walk_directory):
    """FR-11: Re-ingesting replaces chunks (delete-then-insert)."""
    mock_session = MagicMock()
    mock_get_session.return_value = mock_session

    # First ingest: no existing repo
    mock_session.query.return_value.filter_by.return_value.first.return_value = None

    mock_walk_directory.return_value = [
        (MagicMock(__str__=lambda x: "test.py"), "content1"),
    ]
    mock_chunk_file.return_value = [
        {"path": "test.py", "start_line": 1, "end_line": 5, "content": "content1"}
    ]

    captured_repos = []

    def add_side_effect(obj):
        if isinstance(obj, Repo):
            obj.id = 1
            captured_repos.append(obj)

    mock_session.add.side_effect = add_side_effect

    from app.ingest.pipeline import ingest_repo
    result1 = ingest_repo(repo_name="test_repo", root_dir="/fake/path")
    assert result1 is not None

    # Second ingest: existing repo found
    existing_repo = MagicMock(spec=Repo)
    existing_repo.id = 1
    mock_session.query.return_value.filter_by.return_value.first.return_value = existing_repo

    def add_side_effect2(obj):
        if isinstance(obj, Repo):
            obj.id = 1
            captured_repos.append(obj)

    mock_session.add.side_effect = add_side_effect2

    result2 = ingest_repo(repo_name="test_repo", root_dir="/fake/path")
    assert result2 is not None

    # Verify delete was called on existing repo
    mock_session.delete.assert_called_once_with(existing_repo)


@patch("app.ingest.pipeline.walk_directory")
@patch("app.ingest.pipeline.chunk_file")
@patch("app.ingest.pipeline.get_session")
def test_bad_path_error(mock_get_session, mock_chunk_file, mock_walk_directory):
    """Bad path should raise an error."""
    mock_session = MagicMock()
    mock_get_session.return_value = mock_session
    mock_walk_directory.side_effect = FileNotFoundError("Directory not found")

    from app.ingest.pipeline import ingest_repo
    with pytest.raises(FileNotFoundError):
        ingest_repo(repo_name="test_repo", root_dir="/nonexistent")
