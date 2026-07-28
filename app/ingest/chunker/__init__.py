"""Chunking entry point.

Dispatches on file extension: syntax-aware for code, structure-aware for markdown
and SQL, fixed line windows for everything else. Every chunk's content is
prefixed with a synthetic header naming the file and enclosing symbol, so a chunk
retrieved in isolation still says where it came from — which matters both to the
embedding model and to Claude when it grades relevance.
"""

from __future__ import annotations

import logging
from pathlib import PurePosixPath

from app.ingest.chunker.ast_chunker import LANGUAGE_BY_SUFFIX, ast_chunks
from app.ingest.chunker.text_chunker import (
    line_window_chunks,
    markdown_chunks,
    sql_chunks,
)

logger = logging.getLogger(__name__)

LANGUAGE_BY_EXTENSION = {
    ".md": "markdown",
    ".sql": "sql",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".toml": "toml",
}

__all__ = ["chunk_file", "header_for", "LANGUAGE_BY_SUFFIX", "LANGUAGE_BY_EXTENSION"]


def header_for(path: str, symbol: str | None) -> str:
    """The synthetic first line prepended to every chunk's content."""
    if symbol:
        return f"# {path} — {symbol}"
    return f"# {path}"


def _suffix(path: str) -> str:
    return PurePosixPath(path).suffix.lower()


def _dispatch(path: str, text: str) -> list[dict]:
    suffix = _suffix(path)

    language = LANGUAGE_BY_SUFFIX.get(suffix)
    if language is not None:
        chunks = ast_chunks(path, text, language)
        if chunks:
            return chunks
        # Unparseable, or an unavailable grammar. Windowing still makes the file
        # searchable, which beats dropping it.
        return line_window_chunks(path, text, language=language)

    if suffix == ".md":
        return markdown_chunks(path, text)

    if suffix == ".sql":
        return sql_chunks(path, text)

    return line_window_chunks(path, text, language=LANGUAGE_BY_EXTENSION.get(suffix))


def chunk_file(path: str, text: str) -> list[dict]:
    """Split one file into chunks.

    Returns dicts with keys: path, start_line, end_line, content, symbol,
    language. Line numbers are 1-based and point at the real source, unaffected
    by the synthetic header added to content.
    """
    if not text or not text.strip():
        return []

    try:
        chunks = _dispatch(path, text)
    except Exception:
        logger.warning("Chunking failed for %s; using line windows", path, exc_info=True)
        chunks = line_window_chunks(path, text)

    for chunk in chunks:
        chunk["content"] = header_for(path, chunk.get("symbol")) + "\n" + chunk["content"]

    return chunks
