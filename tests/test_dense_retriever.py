"""Dense retrieval.

The important behaviours are the guards: no embedding model means no dense
results (not an exception, and not arbitrary rows), and chunks stored without an
embedding must never be returned by a similarity query.
"""

from unittest.mock import MagicMock, patch

from app.retrieval.dense import DenseRetriever
from app.retrieval.lexical import LexicalRetriever, _split_identifiers


def _factory(rows):
    session = MagicMock()
    result = MagicMock()
    result.fetchall.return_value = rows
    session.execute.return_value = result
    return MagicMock(return_value=session), session


def _row(**kw):
    defaults = dict(
        path="app/db.py",
        start_line=1,
        end_line=10,
        content="def get_session(): ...",
        symbol="get_session",
        language="python",
        score=0.9,
    )
    defaults.update(kw)
    return MagicMock(**defaults)


class TestDense:
    @patch("app.retrieval.dense.embed_query", return_value=None)
    def test_returns_empty_when_embeddings_unavailable(self, _):
        factory, session = _factory([])
        assert DenseRetriever(factory).search(1, "q") == []
        session.execute.assert_not_called()

    @patch("app.retrieval.dense.embed_query", return_value=[0.1, 0.2])
    def test_maps_rows_to_chunkdata(self, _):
        factory, _s = _factory([_row()])
        results = DenseRetriever(factory).search(1, "q")
        assert results[0].path == "app/db.py"
        assert results[0].symbol == "get_session"
        assert results[0].language == "python"
        assert results[0].score == 0.9

    @patch("app.retrieval.dense.embed_query", return_value=[0.1, 0.2])
    def test_excludes_rows_without_embeddings(self, _):
        factory, session = _factory([])
        DenseRetriever(factory).search(1, "q")
        sql = str(session.execute.call_args[0][0])
        assert "embedding IS NOT NULL" in sql

    @patch("app.retrieval.dense.embed_query", return_value=[0.1, 0.2])
    def test_passes_the_vector_as_a_string_literal(self, _):
        """pgvector accepts its text representation; passing a Python list would
        make psycopg guess at an array type."""
        factory, session = _factory([])
        DenseRetriever(factory).search(1, "q")
        params = session.execute.call_args[0][1]
        assert params["vec"] == "[0.1, 0.2]"

    @patch("app.retrieval.dense.embed_query", return_value=[0.1])
    def test_closes_the_session(self, _):
        factory, session = _factory([])
        DenseRetriever(factory).search(1, "q")
        session.close.assert_called_once()


class TestLexical:
    def test_blank_query_returns_empty(self):
        factory, session = _factory([])
        assert LexicalRetriever(factory).search(1, "   ") == []
        session.execute.assert_not_called()

    def test_maps_rows_to_chunkdata(self):
        factory, _s = _factory([_row(score=0.4)])
        results = LexicalRetriever(factory).search(1, "session")
        assert results[0].score == 0.4
        assert results[0].symbol == "get_session"

    def test_uses_the_simple_configuration_not_english(self):
        """English stemming mangles identifiers."""
        factory, session = _factory([])
        LexicalRetriever(factory).search(1, "session")
        sql = str(session.execute.call_args[0][0])
        assert "'simple'" in sql
        assert "'english'" not in sql

    def test_relaxes_and_to_or(self):
        factory, session = _factory([])
        LexicalRetriever(factory).search(1, "database connection config")
        sql = str(session.execute.call_args[0][0])
        assert "' & '" in sql and "' | '" in sql

    def test_closes_the_session(self):
        factory, session = _factory([])
        LexicalRetriever(factory).search(1, "q")
        session.close.assert_called_once()


class TestQuerySideIdentifierSplitting:
    def test_splits_camel_case(self):
        out = _split_identifiers("where is getConnection")
        assert "get" in out.split() and "Connection" in out.split()

    def test_splits_snake_case(self):
        out = _split_identifiers("where is get_connection")
        assert "get" in out.split() and "connection" in out.split()

    def test_keeps_the_original_query(self):
        out = _split_identifiers("where is getConnection")
        assert out.startswith("where is getConnection")

    def test_leaves_plain_prose_untouched(self):
        assert _split_identifiers("how does auth work") == "how does auth work"

    def test_handles_screaming_snake_case(self):
        out = _split_identifiers("DATABASE_URL")
        assert "DATABASE" in out.split() and "URL" in out.split()
