import pytest

from sqlalchemy import create_engine, inspect
from sqlalchemy.orm import sessionmaker

from app.db import Base, Repo, Chunk


@pytest.fixture(scope="function")
def db_session():
    engine = create_engine("sqlite:///:memory:", echo=False)
    Base.metadata.create_all(engine)
    TestSession = sessionmaker(bind=engine)
    session = TestSession()
    yield session
    session.close()
    engine.dispose()


class TestModels:
    def test_repo_columns(self):
        columns = {c.name: c for c in Repo.__table__.columns}
        assert columns["id"].primary_key
        assert not columns["name"].nullable
        assert columns["source_url"].nullable
        assert columns["branch"].nullable
        assert columns["ingested_at"] is not None
        assert columns["file_count"].default.arg == 0
        assert columns["chunk_count"].default.arg == 0

    def test_chunk_columns(self):
        columns = {c.name: c for c in Chunk.__table__.columns}
        assert columns["id"].primary_key
        assert not columns["repo_id"].nullable
        assert not columns["path"].nullable
        assert not columns["start_line"].nullable
        assert not columns["end_line"].nullable
        assert not columns["content"].nullable
        assert columns["embedding"].nullable

    def test_chunk_foreign_key(self):
        fk = next(iter(Chunk.__table__.foreign_keys))
        assert fk.column.table.name == "repos"

    def test_repo_chunk_relationship(self):
        assert hasattr(Repo, "chunks")
        assert hasattr(Chunk, "repo")


class TestCRUD:
    def test_insert_repo(self, db_session):
        repo = Repo(name="test-repo", source_url="https://example.com/repo")
        db_session.add(repo)
        db_session.commit()
        assert repo.id is not None
        fetched = db_session.get(Repo, repo.id)
        assert fetched.name == "test-repo"
        assert fetched.source_url == "https://example.com/repo"

    def test_insert_chunk_without_embedding(self, db_session):
        """FTS-only mode: chunk can be inserted with embedding=NULL"""
        repo = Repo(name="fts-repo")
        db_session.add(repo)
        db_session.commit()

        chunk = Chunk(
            repo_id=repo.id,
            path="src/main.py",
            start_line=1,
            end_line=10,
            content="def hello():\n    print('hello')\n",
            embedding=None,
        )
        db_session.add(chunk)
        db_session.commit()

        assert chunk.id is not None
        assert chunk.embedding is None

        fetched = db_session.get(Chunk, chunk.id)
        assert fetched.path == "src/main.py"
        assert fetched.content == "def hello():\n    print('hello')\n"
        assert fetched.embedding is None

    def test_chunk_repo_relationship(self, db_session):
        repo = Repo(name="relation-test")
        db_session.add(repo)
        db_session.commit()

        chunk1 = Chunk(
            repo_id=repo.id, path="a.py", start_line=1, end_line=5, content="aaa"
        )
        chunk2 = Chunk(
            repo_id=repo.id, path="b.py", start_line=1, end_line=5, content="bbb"
        )
        db_session.add_all([chunk1, chunk2])
        db_session.commit()

        assert len(repo.chunks) == 2
        assert chunk1.repo.name == "relation-test"

    def test_cascade_delete(self, db_session):
        repo = Repo(name="cascade-test")
        db_session.add(repo)
        db_session.commit()
        repo_id = repo.id

        chunk = Chunk(
            repo_id=repo_id, path="c.py", start_line=1, end_line=3, content="ccc"
        )
        db_session.add(chunk)
        db_session.commit()

        db_session.delete(repo)
        db_session.commit()

        assert db_session.get(Repo, repo_id) is None
        assert db_session.get(Chunk, chunk.id) is None
