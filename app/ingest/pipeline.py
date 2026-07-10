import logging
from pathlib import Path
from typing import Callable, Optional

from app.db import get_session, Repo, Chunk
from app.ingest.embedder import embed_texts, is_voyage_available
from app.ingest.walker import walk_directory
from app.ingest.chunker import chunk_file

logger = logging.getLogger(__name__)


def ingest_repo(
    repo_name: str,
    root_dir: str,
    source_url: Optional[str] = None,
    branch: Optional[str] = None,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
) -> Repo:
    session = get_session()
    try:
        existing = session.query(Repo).filter_by(name=repo_name).first()
        if existing is not None:
            session.delete(existing)
            session.flush()

        repo = Repo(name=repo_name, source_url=source_url, branch=branch, file_count=0, chunk_count=0)
        session.add(repo)
        session.flush()

        total_files = 0
        total_chunks = 0
        all_chunks: list[Chunk] = []

        for filepath, text in walk_directory(root_dir):
            path_str = str(filepath)
            file_chunks = chunk_file(path_str, text)
            for c in file_chunks:
                all_chunks.append(
                    Chunk(
                        repo_id=repo.id,
                        path=c["path"],
                        start_line=c["start_line"],
                        end_line=c["end_line"],
                        content=c["content"],
                    )
                )
            total_files += 1
            total_chunks += len(file_chunks)
            if progress_callback is not None:
                progress_callback(total_files, len(file_chunks), path_str)

        if is_voyage_available():
            texts_to_embed = [c.content for c in all_chunks]
            embeddings = embed_texts(texts_to_embed)
            if embeddings:
                for chunk, embedding in zip(all_chunks, embeddings):
                    chunk.embedding = embedding

        for chunk in all_chunks:
            session.add(chunk)

        repo.file_count = total_files
        repo.chunk_count = total_chunks
        session.commit()
        logger.info("Ingested %s: %d files, %d chunks", repo_name, total_files, total_chunks)
        return repo
    except Exception:
        session.rollback()
        logger.exception("Ingest failed for %s", repo_name)
        raise
