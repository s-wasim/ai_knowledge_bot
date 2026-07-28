"""Retrieval against a live Postgres.

The unit tests mock the session, so they verify shape but not SQL. These run the
real queries, which is the only way to catch a broken generated column, a missing
extension, an unescaped `%` in a psycopg format string, or a pgvector cast that
does not compile.

Skipped when no database is reachable.
"""

import os

import pytest
from sqlalchemy import text

pytestmark = pytest.mark.integration


def _database_url() -> str:
    return os.environ.get(
        "DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/knowledgebot"
    )


@pytest.fixture(scope="module")
def engine():
    from sqlalchemy import create_engine

    try:
        eng = create_engine(_database_url(), pool_pre_ping=True)
        with eng.connect() as conn:
            conn.execute(text("SELECT 1"))
    except Exception as e:
        pytest.skip(f"No Postgres available: {e}")
    return eng


@pytest.fixture(scope="module")
def seeded(engine):
    """Insert a throwaway repo with three known chunks, then remove it."""
    import app.db as db_module

    db_module.init_db()

    session_factory = db_module.get_session()
    session = session_factory()

    repo = db_module.Repo(name="__integration_fixture__", file_count=2, chunk_count=3)
    session.add(repo)
    session.flush()

    rows = [
        db_module.Chunk(
            repo_id=repo.id,
            path="app/db.py",
            start_line=1,
            end_line=12,
            content="def get_connection():\n    return create_engine(DATABASE_URL)\n",
            symbol="get_connection",
            language="python",
            embedding=[0.0] * (db_module.EMBED_DIMS - 1) + [1.0],
        ),
        db_module.Chunk(
            repo_id=repo.id,
            path="app/auth.py",
            start_line=1,
            end_line=9,
            content="def verifyPassword(raw, hashed):\n    return compare(raw, hashed)\n",
            symbol="verifyPassword",
            language="python",
            embedding=[1.0] + [0.0] * (db_module.EMBED_DIMS - 1),
        ),
        db_module.Chunk(
            repo_id=repo.id,
            path="README.md",
            start_line=1,
            end_line=4,
            content="# Docs\nNothing about databases here.\n",
            symbol="Docs",
            language="markdown",
            embedding=None,
        ),
    ]
    session.add_all(rows)
    session.commit()
    repo_id = repo.id
    session.close()

    yield repo_id, session_factory

    session = session_factory()
    victim = session.query(db_module.Repo).filter_by(id=repo_id).first()
    if victim is not None:
        session.delete(victim)
        session.commit()
    session.close()


class TestExtensionsAndSchema:
    def test_required_extensions_exist(self, engine, seeded):
        with engine.connect() as conn:
            names = {
                row[0]
                for row in conn.execute(
                    text("SELECT extname FROM pg_extension")
                ).fetchall()
            }
        assert {"vector", "pg_trgm"} <= names

    def test_generated_tsv_column_exists(self, engine, seeded):
        with engine.connect() as conn:
            generated = conn.execute(
                text(
                    "SELECT is_generated FROM information_schema.columns "
                    "WHERE table_name='chunks' AND column_name='tsv'"
                )
            ).scalar()
        assert generated == "ALWAYS"


class TestLexical:
    def test_matches_a_split_identifier(self, seeded):
        """The whole reason for the custom tsvector: a natural-language query must
        reach a camelCase or snake_case identifier."""
        from app.retrieval.lexical import LexicalRetriever

        repo_id, factory = seeded
        results = LexicalRetriever(factory).search(repo_id, "get connection")
        assert any(r.path == "app/db.py" for r in results)

    def test_matches_camel_case_from_spaced_words(self, seeded):
        from app.retrieval.lexical import LexicalRetriever

        repo_id, factory = seeded
        results = LexicalRetriever(factory).search(repo_id, "verify password")
        assert any(r.path == "app/auth.py" for r in results)

    def test_returns_symbol_and_language(self, seeded):
        from app.retrieval.lexical import LexicalRetriever

        repo_id, factory = seeded
        results = LexicalRetriever(factory).search(repo_id, "get connection")
        hit = next(r for r in results if r.path == "app/db.py")
        assert hit.symbol == "get_connection"
        assert hit.language == "python"

    def test_scopes_to_the_repo(self, seeded):
        from app.retrieval.lexical import LexicalRetriever

        _repo_id, factory = seeded
        assert LexicalRetriever(factory).search(-999, "get connection") == []


class TestSymbolic:
    def test_finds_a_chunk_by_its_symbol(self, seeded):
        from app.retrieval.symbolic import SymbolicRetriever

        repo_id, factory = seeded
        results = SymbolicRetriever(factory).search(repo_id, "where is get_connection")
        assert any(r.path == "app/db.py" for r in results)

    def test_tolerates_a_near_miss(self, seeded):
        """Trigram similarity is what lets getConnection find get_connection."""
        from app.retrieval.symbolic import SymbolicRetriever

        repo_id, factory = seeded
        results = SymbolicRetriever(factory).search(repo_id, "getConnection")
        assert any(r.path == "app/db.py" for r in results)

    def test_prose_query_returns_nothing(self, seeded):
        from app.retrieval.symbolic import SymbolicRetriever

        repo_id, factory = seeded
        assert SymbolicRetriever(factory).search(repo_id, "how does it all work") == []


class TestDense:
    def test_orders_by_cosine_distance(self, seeded):
        from app.retrieval.dense import DenseRetriever
        import app.db as db_module

        repo_id, factory = seeded
        # A query vector aligned with the auth chunk's embedding.
        vector = [1.0] + [0.0] * (db_module.EMBED_DIMS - 1)

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr("app.retrieval.dense.embed_query", lambda _q: vector)
            results = DenseRetriever(factory).search(repo_id, "anything")

        assert results
        assert results[0].path == "app/auth.py"

    def test_excludes_chunks_without_embeddings(self, seeded):
        from app.retrieval.dense import DenseRetriever
        import app.db as db_module

        repo_id, factory = seeded
        vector = [0.5] * db_module.EMBED_DIMS

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr("app.retrieval.dense.embed_query", lambda _q: vector)
            results = DenseRetriever(factory).search(repo_id, "anything")

        assert all(r.path != "README.md" for r in results)


class TestHybrid:
    def test_fuses_all_signals_and_marks_provenance(self, seeded):
        from app.retrieval.hybrid import HybridRetriever

        repo_id, factory = seeded
        results = HybridRetriever(factory).search(repo_id, "where is get_connection")

        assert results
        top = next(r for r in results if r.path == "app/db.py")
        # Lexical and symbolic should both find it; dense depends on the model
        # being present, which this test does not require.
        assert {"text", "symbol"} <= set(top.sources)

    def test_works_without_the_embedding_model(self, seeded):
        from app.retrieval.hybrid import HybridRetriever

        repo_id, factory = seeded
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr("app.retrieval.dense.embed_query", lambda _q: None)
            results = HybridRetriever(factory).search(repo_id, "get_connection")

        assert any(r.path == "app/db.py" for r in results)
