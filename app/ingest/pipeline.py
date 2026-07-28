"""Ingest a directory into the chunk index.

Paths are stored relative to the ingest root. They were previously absolute, which
for GitHub ingests meant citations pointing into a deleted temp directory
(`/tmp/gh_ingest_xvpsadyu/repo-main/file.py`) — unusable as evidence, and enough
to break the not-found path's directory suggestions.

Embedding is best-effort by design: if the model is unavailable, chunks are stored
without vectors and retrieval degrades to lexical and symbolic search. Losing the
file would be worse than losing its embedding, and the shortfall is reported.
"""

import logging
import os
from pathlib import Path
from typing import Callable, Optional

from app.db import get_session, Repo, Chunk
from app.ingest.chunker import chunk_file
from app.ingest.embedder import embed_texts, is_embedding_available
from app.ingest.walker import count_files, walk_directory

logger = logging.getLogger(__name__)

DEFAULT_MAX_CHUNKS = 20000


def max_chunks_per_repo() -> int:
    """Read at call time so tests and operators can change it without a restart."""
    raw = os.environ.get("KB_MAX_CHUNKS")
    if not raw:
        return DEFAULT_MAX_CHUNKS
    try:
        value = int(raw)
    except ValueError:
        logger.warning("Ignoring non-numeric KB_MAX_CHUNKS=%r", raw)
        return DEFAULT_MAX_CHUNKS
    return value if value > 0 else DEFAULT_MAX_CHUNKS


def _relative_path(filepath: Path, root: Path) -> str:
    """Repo-relative, forward-slashed path for storage and display."""
    try:
        return filepath.relative_to(root).as_posix()
    except ValueError:
        # Outside the root: a symlink escape, or a root that moved mid-walk.
        logger.warning("Path %s is outside ingest root %s", filepath, root)
        return filepath.name


def ingest_repo(
    repo_name: str,
    root_dir: str,
    source_url: Optional[str] = None,
    branch: Optional[str] = None,
    progress_callback: Optional[Callable[[int, int, int, str], None]] = None,
    warning_callback: Optional[Callable[[str], None]] = None,
) -> Repo:
    """Walk, chunk, embed, and persist a repository.

    progress_callback receives (files_done, files_total, chunks_added, path) with
    a repo-relative path. warning_callback receives human-readable notices such as
    a chunk cap being hit — surfaced rather than logged and forgotten.
    """
    root = Path(root_dir).resolve()
    cap = max_chunks_per_repo()

    session = get_session()
    try:
        existing = session.query(Repo).filter_by(name=repo_name).first()
        if existing is not None:
            session.delete(existing)
            session.flush()

        repo = Repo(
            name=repo_name,
            source_url=source_url,
            branch=branch,
            file_count=0,
            chunk_count=0,
        )
        session.add(repo)
        session.flush()

        total_files = count_files(root)

        processed_files = 0
        all_chunks: list[Chunk] = []
        truncated = False

        for filepath, text in walk_directory(root):
            rel_path = _relative_path(filepath, root)
            file_chunks = chunk_file(rel_path, text)

            for c in file_chunks:
                if len(all_chunks) >= cap:
                    truncated = True
                    break
                all_chunks.append(
                    Chunk(
                        repo_id=repo.id,
                        path=c["path"],
                        start_line=c["start_line"],
                        end_line=c["end_line"],
                        content=c["content"],
                        symbol=c.get("symbol"),
                        language=c.get("language"),
                    )
                )

            processed_files += 1
            if progress_callback is not None:
                progress_callback(processed_files, total_files, len(file_chunks), rel_path)

            if truncated:
                message = (
                    f"Chunk cap of {cap} reached after {processed_files} of "
                    f"{total_files} files. The rest of the repository was not "
                    f"indexed. Raise KB_MAX_CHUNKS to index more."
                )
                logger.warning(message)
                if warning_callback is not None:
                    warning_callback(message)
                break

        _attach_embeddings(all_chunks, warning_callback)

        for chunk in all_chunks:
            session.add(chunk)

        repo.file_count = processed_files
        repo.chunk_count = len(all_chunks)
        session.commit()
        logger.info(
            "Ingested %s: %d files, %d chunks", repo_name, processed_files, len(all_chunks)
        )
        return repo
    except Exception:
        session.rollback()
        logger.exception("Ingest failed for %s", repo_name)
        raise


def _attach_embeddings(chunks: list, warning_callback) -> None:
    """Embed chunk contents in place.

    Two failures, deliberately handled differently:

    - The model is unavailable at all. This is the documented degraded mode: the
      repo is indexed without vectors, the caller is told, and /health reports it,
      so retrieval falling back to lexical and symbolic search is expected.
    - The model loaded but embedding failed partway. That is an error, and it
      raises. A half-embedded index would keep reporting healthy hybrid search
      while quietly returning nothing from its dense half for this repo — the
      silent degradation FR-5 exists to prevent. Failing here means the operator
      retries instead of inheriting a broken index.
    """
    if not chunks:
        return

    if not is_embedding_available():
        message = (
            "Embedding model unavailable, so this repo was indexed without vectors. "
            "Search will use full-text and symbol matching only."
        )
        logger.warning(message)
        if warning_callback is not None:
            warning_callback(message)
        return

    embeddings = embed_texts([c.content for c in chunks])

    if embeddings is None:
        raise RuntimeError(
            f"Embedding failed for {len(chunks)} chunks. The repository was not "
            f"indexed; re-run the ingest once the embedding model is healthy."
        )

    if len(embeddings) != len(chunks):
        raise RuntimeError(
            f"Embedder returned {len(embeddings)} vectors for {len(chunks)} chunks. "
            f"The repository was not indexed."
        )

    for chunk, embedding in zip(chunks, embeddings):
        chunk.embedding = embedding
