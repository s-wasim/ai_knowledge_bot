"""Symbolic retrieval.

This is the retriever that earns hybrid search its keep on code. A question naming
`get_connection` must surface that definition, which dense embeddings do
unreliably and stemmed full-text search does badly. Identifier extraction is the
whole game: it must fire on code-shaped tokens and stay silent on prose, so plain
English questions do not drag in noise.
"""

from unittest.mock import MagicMock

from app.retrieval.symbolic import SymbolicRetriever, extract_identifiers


def _session_factory(rows):
    session = MagicMock()
    result = MagicMock()
    result.fetchall.return_value = rows
    session.execute.return_value = result
    return MagicMock(return_value=session), session


class TestExtractIdentifiers:
    def test_finds_snake_case(self):
        assert "get_connection" in extract_identifiers("where is get_connection set up?")

    def test_finds_camel_case(self):
        assert "getPool" in extract_identifiers("does getPool cache anything")

    def test_finds_pascal_case(self):
        assert "SessionStore" in extract_identifiers("what does SessionStore do")

    def test_finds_dotted_paths(self):
        assert "app.db" in extract_identifiers("what lives in app.db")

    def test_finds_backticked_spans(self):
        assert "DATABASE_URL" in extract_identifiers("how is `DATABASE_URL` read")

    def test_finds_screaming_snake_case(self):
        assert "MAX_RETRIES" in extract_identifiers("what sets MAX_RETRIES")

    def test_finds_file_paths(self):
        ids = extract_identifiers("what is in auth/login.py")
        assert any("login" in i for i in ids)

    def test_ignores_plain_english(self):
        assert extract_identifiers("how does authentication work") == []

    def test_ignores_ordinary_capitalized_sentences(self):
        """A sentence-initial capital is not a PascalCase identifier."""
        assert extract_identifiers("Where is the database configured?") == []

    def test_ignores_short_noise(self):
        assert extract_identifiers("is it ok") == []

    def test_deduplicates(self):
        ids = extract_identifiers("get_connection and get_connection again")
        assert ids.count("get_connection") == 1

    def test_handles_empty_query(self):
        assert extract_identifiers("") == []


class TestSearch:
    def test_returns_empty_without_identifiers(self):
        """No identifiers means no symbolic signal. Querying anyway would return
        arbitrary trigram noise that fusion would then reward."""
        factory, session = _session_factory([])
        results = SymbolicRetriever(factory).search(1, "how does authentication work")
        assert results == []
        session.execute.assert_not_called()

    def test_maps_rows_to_chunkdata(self):
        row = MagicMock(
            path="app/db.py",
            start_line=10,
            end_line=20,
            content="def get_connection(): ...",
            symbol="get_connection",
            language="python",
            score=0.75,
        )
        factory, _ = _session_factory([row])
        results = SymbolicRetriever(factory).search(1, "where is get_connection")
        assert len(results) == 1
        assert results[0].path == "app/db.py"
        assert results[0].symbol == "get_connection"
        assert results[0].score == 0.75

    def test_closes_the_session(self):
        factory, session = _session_factory([])
        SymbolicRetriever(factory).search(1, "get_connection")
        session.close.assert_called_once()

    def test_closes_the_session_when_the_query_raises(self):
        session = MagicMock()
        session.execute.side_effect = RuntimeError("boom")
        factory = MagicMock(return_value=session)
        try:
            SymbolicRetriever(factory).search(1, "get_connection")
        except RuntimeError:
            pass
        session.close.assert_called_once()

    def test_passes_repo_id_and_limit(self):
        factory, session = _session_factory([])
        SymbolicRetriever(factory).search(7, "get_connection", k=15)
        params = session.execute.call_args[0][1]
        assert params["repo_id"] == 7
        assert params["k"] == 15
