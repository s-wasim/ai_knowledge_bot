"""Chunkers for non-code and fallback content.

`line_window_chunks` is the universal fallback: it makes no assumptions about the
input, so it is what unrecognised extensions, unparseable sources, and oversized
AST nodes fall back to.
"""

from __future__ import annotations

import re

HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")


def _chunk(
    path: str,
    start_line: int,
    end_line: int,
    content: str,
    symbol: str | None,
    language: str | None,
) -> dict:
    return {
        "path": path,
        "start_line": start_line,
        "end_line": end_line,
        "content": content,
        "symbol": symbol,
        "language": language,
    }


def line_window_chunks(
    path: str,
    text: str,
    window: int = 80,
    overlap: int = 20,
    min_chunk: int = 10,
    language: str | None = None,
) -> list[dict]:
    """Fixed-size overlapping line windows over the whole file."""
    if not text:
        return []

    lines = text.split('\n')
    total = len(lines)

    if total <= window:
        return [_chunk(path, 1, total, text, None, language)]

    step = max(1, window - overlap)
    chunks: list[dict] = []
    start = 0

    while start < total:
        end = min(start + window, total)
        chunks.append(
            _chunk(path, start + 1, end, '\n'.join(lines[start:end]), None, language)
        )
        if end == total:
            break
        start += step

    if len(chunks) > 1:
        last = chunks[-1]
        last_size = last["end_line"] - last["start_line"] + 1
        if last_size < min_chunk:
            prev = chunks[-2]
            prev["end_line"] = last["end_line"]
            prev["content"] = prev["content"] + '\n' + last["content"]
            chunks.pop()

    return chunks


def markdown_chunks(path: str, text: str) -> list[dict]:
    """Split on ATX headings, keeping each heading with the prose beneath it.

    Prose appearing before the first heading becomes its own chunk rather than
    being dropped.
    """
    if not text.strip():
        return []

    lines = text.split('\n')
    sections: list[tuple[int, str | None]] = []

    for i, line in enumerate(lines):
        match = HEADING_RE.match(line)
        if match:
            sections.append((i, match.group(2).strip() or None))

    if not sections:
        return line_window_chunks(path, text, language="markdown")

    chunks: list[dict] = []

    if sections[0][0] > 0:
        preamble = '\n'.join(lines[: sections[0][0]])
        if preamble.strip():
            chunks.append(
                _chunk(path, 1, sections[0][0], preamble, None, "markdown")
            )

    for idx, (line_idx, symbol) in enumerate(sections):
        end_idx = sections[idx + 1][0] if idx + 1 < len(sections) else len(lines)
        body = '\n'.join(lines[line_idx:end_idx])
        if not body.strip():
            continue
        chunks.append(
            _chunk(path, line_idx + 1, end_idx, body, symbol, "markdown")
        )

    return chunks or line_window_chunks(path, text, language="markdown")


def sql_chunks(path: str, text: str) -> list[dict]:
    """Split on statement terminators, keeping each statement whole."""
    if not text.strip():
        return []

    lines = text.split('\n')
    chunks: list[dict] = []
    start_idx = 0

    for i, line in enumerate(lines):
        if line.rstrip().endswith(';'):
            body = '\n'.join(lines[start_idx : i + 1])
            if body.strip():
                chunks.append(
                    _chunk(path, start_idx + 1, i + 1, body, _sql_symbol(body), "sql")
                )
            start_idx = i + 1

    trailing = '\n'.join(lines[start_idx:])
    if trailing.strip():
        chunks.append(
            _chunk(
                path,
                start_idx + 1,
                len(lines),
                trailing,
                _sql_symbol(trailing),
                "sql",
            )
        )

    return chunks or line_window_chunks(path, text, language="sql")


_SQL_NAME_RE = re.compile(
    r"\b(?:CREATE|ALTER|DROP)\s+(?:TABLE|INDEX|VIEW|FUNCTION|TYPE)\s+"
    r"(?:IF\s+NOT\s+EXISTS\s+)?[\"`]?([\w.]+)",
    re.IGNORECASE,
)


def _sql_symbol(statement: str) -> str | None:
    match = _SQL_NAME_RE.search(statement)
    return match.group(1) if match else None
