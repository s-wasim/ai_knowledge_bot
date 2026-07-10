from unittest.mock import ANY, MagicMock, patch, call

import pytest

from app.ingest.pipeline import ingest_repo


def _make_chunk(path, start, end):
    return {"path": path, "start_line": start, "end_line": end, "content": f"lines {start}-{end}"}


@patch("app.ingest.pipeline.chunk_file", return_value=[_make_chunk("/f/a.py", 1, 5), _make_chunk("/f/a.py", 3, 8)])
@patch("app.ingest.pipeline.walk_directory")
@patch("app.ingest.pipeline.get_session")
def test_basic_ingest(mock_get_session, mock_walk_directory, mock_chunk_file):
    mock_session = MagicMock()
    mock_get_session.return_value = mock_session

    mock_repo = MagicMock()
    mock_repo.id = 1
    mock_repo.name = ""
    mock_repo.file_count = 0
    mock_repo.chunk_count = 0

    mock_session.query.return_value.filter_by.return_value.first.return_value = None

    def add_side_effect(obj):
        if isinstance(obj, MagicMock) and hasattr(obj, "id"):
            return
        if obj is mock_repo:
            obj.name = "test_repo"
            obj.file_count = 0
            obj.chunk_count = 0

    mock_session.add.side_effect = add_side_effect

    mock_walk_directory.return_value = iter([("/f/a.py", "line1\nline2\nline3\nline4\nline5\nline6\nline7\nline8")])

    result = ingest_repo("test_repo", "/fake/root", progress_callback=lambda c, a, f: None)

    assert result is not None
    assert mock_session.add.called
    assert mock_session.commit.called


@patch("app.ingest.pipeline.chunk_file", return_value=[_make_chunk("/f/b.py", 1, 3)])
@patch("app.ingest.pipeline.walk_directory")
@patch("app.ingest.pipeline.get_session")
def test_reingest_replaces_chunks(mock_get_session, mock_walk_directory, mock_chunk_file):
    mock_session = MagicMock()
    mock_get_session.return_value = mock_session

    # First call: no existing repo
    existing_repo = MagicMock()
    existing_repo.id = 1
    mock_session.query.return_value.filter_by.return_value.first.side_effect = [None, existing_repo]

    repo_obj = MagicMock()
    repo_obj.id = 1
    repo_obj.name = ""
    repo_obj.file_count = 0
    repo_obj.chunk_count = 0

    def add_side_effect(obj):
        if obj is repo_obj:
            repo_obj.name = "test_repo"
            repo_obj.file_count = 1
            repo_obj.chunk_count = 1

    mock_session.add.side_effect = add_side_effect

    mock_walk_directory.return_value = iter([("/f/b.py", "a\nb\nc")])

    ingest_repo("test_repo", "/fake/root")
    ingest_repo("test_repo", "/fake/root")

    delete_calls = [c for c in mock_session.delete.call_args_list]
    assert len(delete_calls) == 1
    assert mock_session.commit.call_count == 2


@patch("app.ingest.pipeline.walk_directory")
@patch("app.ingest.pipeline.get_session")
def test_bad_path_error(mock_get_session, mock_walk_directory):
    mock_session = MagicMock()
    mock_get_session.return_value = mock_session
    mock_session.query.return_value.filter_by.return_value.first.return_value = None

    mock_walk_directory.side_effect = FileNotFoundError("Directory not found: /bad/path")

    with pytest.raises(FileNotFoundError, match="Directory not found"):
        ingest_repo("bad_repo", "/bad/path")

    mock_session.rollback.assert_called_once()
