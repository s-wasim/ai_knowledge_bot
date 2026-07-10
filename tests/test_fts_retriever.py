from unittest.mock import ANY, MagicMock

from app.retrieval.base import ChunkData, Retriever
from app.retrieval.fts import FtsRetriever


def _mock_row(path, start_line, end_line, content, score):
    row = MagicMock()
    row.path = path
    row.start_line = start_line
    row.end_line = end_line
    row.content = content
    row.score = score
    return row


class TestFtsRetriever:
    def test_search_returns_chunkdata(self):
        mock_session = MagicMock()
        mock_result = MagicMock()
        mock_result.fetchall.return_value = [
            _mock_row("file.py", 1, 10, "code content", 0.5),
        ]
        mock_session.execute.return_value = mock_result
        mock_factory = MagicMock(return_value=mock_session)

        retriever = FtsRetriever(mock_factory)
        results = retriever.search(repo_id=1, query="test function")

        assert len(results) == 1
        chunk = results[0]
        assert isinstance(chunk, ChunkData)
        assert chunk.path == "file.py"
        assert chunk.start_line == 1
        assert chunk.end_line == 10
        assert chunk.content == "code content"
        assert chunk.score == 0.5

    def test_k_respected(self):
        all_rows = [
            _mock_row(f"file{i}.py", 1, 2, f"content{i}", 0.1 * (10 - i))
            for i in range(5)
        ]

        def execute_side_effect(*args, **kwargs):
            params = args[1] if len(args) > 1 else {}
            k = params.get("k", 8)
            mock_result = MagicMock()
            mock_result.fetchall.return_value = all_rows[:k]
            return mock_result

        mock_session = MagicMock()
        mock_session.execute.side_effect = execute_side_effect
        mock_factory = MagicMock(return_value=mock_session)

        retriever = FtsRetriever(mock_factory)
        results = retriever.search(repo_id=1, query="test", k=3)

        assert len(results) == 3

    def test_empty_query(self):
        mock_factory = MagicMock()
        retriever = FtsRetriever(mock_factory)
        results = retriever.search(repo_id=1, query="   ")
        assert results == []
        mock_factory.assert_not_called()

    def test_no_results(self):
        mock_session = MagicMock()
        mock_result = MagicMock()
        mock_result.fetchall.return_value = []
        mock_session.execute.return_value = mock_result
        mock_factory = MagicMock(return_value=mock_session)

        retriever = FtsRetriever(mock_factory)
        results = retriever.search(repo_id=1, query="zzzznotfound")

        assert results == []

    def test_session_closed(self):
        mock_session = MagicMock()
        mock_result = MagicMock()
        mock_result.fetchall.return_value = []
        mock_session.execute.return_value = mock_result
        mock_factory = MagicMock(return_value=mock_session)

        retriever = FtsRetriever(mock_factory)
        retriever.search(repo_id=1, query="test")

        mock_session.close.assert_called_once()

    def test_type_checks_protocol(self):
        mock_factory = MagicMock()
        retriever = FtsRetriever(mock_factory)
        assert isinstance(retriever, Retriever)
