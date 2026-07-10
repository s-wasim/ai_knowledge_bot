import os
import time
from datetime import datetime, timezone

from sqlalchemy import (
    Column,
    Integer,
    String,
    Text,
    DateTime,
    ForeignKey,
    create_engine,
)
from sqlalchemy.orm import declarative_base, relationship, scoped_session, sessionmaker
from sqlalchemy import DDL

from pgvector.sqlalchemy import Vector

Base = declarative_base()


class Repo(Base):
    __tablename__ = "repos"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String, nullable=False)
    source_url = Column(String, nullable=True)
    branch = Column(String, nullable=True)
    ingested_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    file_count = Column(Integer, default=0)
    chunk_count = Column(Integer, default=0)

    chunks = relationship("Chunk", back_populates="repo", cascade="all, delete-orphan")


class Chunk(Base):
    __tablename__ = "chunks"

    id = Column(Integer, primary_key=True, autoincrement=True)
    repo_id = Column(Integer, ForeignKey("repos.id"), nullable=False)
    path = Column(String, nullable=False)
    start_line = Column(Integer, nullable=False)
    end_line = Column(Integer, nullable=False)
    content = Column(Text, nullable=False)
    embedding = Column(Vector(1024), nullable=True)

    repo = relationship("Repo", back_populates="chunks")


_engine = None
_session_factory = None
_scoped_session = None


def init_db(retries=10, delay=3):
    global _engine, _session_factory, _scoped_session

    database_url = os.environ.get(
        "DATABASE_URL",
        "postgresql://postgres:postgres@db:5432/knowledgebot",
    )

    for attempt in range(1, retries + 1):
        try:
            _engine = create_engine(database_url, pool_pre_ping=True)
            _engine.connect().close()
            break
        except Exception:
            if attempt == retries:
                raise
            time.sleep(delay)

    _session_factory = sessionmaker(bind=_engine)
    _scoped_session = scoped_session(_session_factory)

    if _engine.dialect.name == "postgresql":
        with _engine.connect() as conn:
            conn.execute(DDL("CREATE EXTENSION IF NOT EXISTS vector"))
            conn.commit()

    Base.metadata.create_all(_engine)

    if _engine.dialect.name == "postgresql":
        with _engine.connect() as conn:
            conn.execute(
                DDL(
                    "ALTER TABLE chunks ADD COLUMN IF NOT EXISTS tsv tsvector "
                    "GENERATED ALWAYS AS (to_tsvector('english', coalesce(content, ''))) STORED"
                )
            )
            conn.execute(
                DDL("CREATE INDEX IF NOT EXISTS ix_chunks_tsv ON chunks USING gin (tsv)")
            )
            conn.execute(
                DDL(
                    "CREATE INDEX IF NOT EXISTS ix_chunks_embedding "
                    "ON chunks USING hnsw (embedding vector_cosine_ops)"
                )
            )
            conn.commit()

    return _engine, _scoped_session


def get_session():
    global _scoped_session
    if _scoped_session is None:
        init_db()
    return _scoped_session


__all__ = ["Base", "Repo", "Chunk", "init_db", "get_session"]
