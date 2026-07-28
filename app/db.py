import os
import threading
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

# Dimensionality of jinaai/jina-embeddings-v2-base-code. Changing this requires
# rebuilding the index — see scripts/reset_index.py.
EMBED_DIMS = 768

# Postgres' `english` text-search config stems code identifiers into nonsense, so
# the generated column uses `simple` and additionally indexes an identifier-split
# copy of the content: `getConnection` becomes `get Connection`, and `_ . - /`
# become spaces. That lets a natural-language query match camelCase and
# snake_case names. Both regexp_replace and translate are IMMUTABLE, which a
# generated column requires.
_TSV_EXPRESSION = (
    "to_tsvector('simple', "
    "coalesce(content,'') || ' ' || "
    "translate("
    "regexp_replace(coalesce(content,''), '([a-z0-9])([A-Z])', '\\1 \\2', 'g'),"
    " '_.-/', '    ')"
    ")"
)


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
    symbol = Column(String, nullable=True)
    language = Column(String, nullable=True)
    embedding = Column(Vector(EMBED_DIMS), nullable=True)

    repo = relationship("Repo", back_populates="chunks")


_engine = None
_session_factory = None
_scoped_session = None
_init_lock = threading.Lock()


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
            conn.execute(DDL("CREATE EXTENSION IF NOT EXISTS pg_trgm"))
            conn.commit()

    Base.metadata.create_all(_engine)

    if _engine.dialect.name == "postgresql":
        _create_search_objects(_engine)

    return _engine, _scoped_session


def _create_search_objects(engine) -> None:
    """Create the generated tsvector column and every search index, idempotently.

    Split out of init_db so scripts/reset_index.py can rebuild the index without
    duplicating the DDL.
    """
    with engine.connect() as conn:
        conn.execute(
            DDL(
                "ALTER TABLE chunks ADD COLUMN IF NOT EXISTS tsv tsvector "
                f"GENERATED ALWAYS AS ({_TSV_EXPRESSION}) STORED"
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
        # Trigram indexes back the symbolic retriever's fuzzy identifier matching.
        conn.execute(
            DDL(
                "CREATE INDEX IF NOT EXISTS ix_chunks_symbol_trgm "
                "ON chunks USING gin (symbol gin_trgm_ops)"
            )
        )
        conn.execute(
            DDL(
                "CREATE INDEX IF NOT EXISTS ix_chunks_path_trgm "
                "ON chunks USING gin (path gin_trgm_ops)"
            )
        )
        conn.execute(
            DDL("CREATE INDEX IF NOT EXISTS ix_chunks_repo_id ON chunks (repo_id)")
        )
        conn.commit()


def get_session():
    global _scoped_session
    if _scoped_session is None:
        with _init_lock:
            if _scoped_session is None:
                init_db()
    return _scoped_session


__all__ = [
    "Base",
    "Repo",
    "Chunk",
    "EMBED_DIMS",
    "init_db",
    "get_session",
    "_create_search_objects",
]
