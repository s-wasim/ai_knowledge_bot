"""Structure-aware chunking via tree-sitter.

Each chunk is a whole definition — function, method, class — with its signature
and decorators intact, because that is the unit both the embedding model and
Claude's relevance grader are shown. A chunk cut mid-function is evidence nobody
can act on.

Every failure path returns None so the dispatcher falls back to line windows. A
grammar that is missing, a syntax error, or an unexpected node shape must never
fail an ingest.
"""

from __future__ import annotations

import logging
import threading

from app.ingest.chunker.text_chunker import line_window_chunks

logger = logging.getLogger(__name__)

LANGUAGE_BY_SUFFIX = {
    ".py": "python",
    ".js": "javascript",
    ".ts": "typescript",
    ".tsx": "tsx",
}

MAX_CHUNK_LINES = 120

# Node types that represent a nameable definition worth its own chunk.
DEFINITION_TYPES = frozenset(
    {
        # Python
        "function_definition",
        "class_definition",
        "decorated_definition",
        # JavaScript / TypeScript
        "function_declaration",
        "generator_function_declaration",
        "class_declaration",
        "abstract_class_declaration",
        "method_definition",
        "interface_declaration",
        "type_alias_declaration",
        "enum_declaration",
        "lexical_declaration",
        "variable_declaration",
        # Wrappers that carry a definition as their payload.
        "export_statement",
        "ambient_declaration",
    }
)

# Wrappers whose real subject is a child node.
_UNWRAP_TYPES = frozenset({"export_statement", "decorated_definition", "ambient_declaration"})

_COMMENT_PREFIXES = {
    "python": ("#",),
    "javascript": ("//", "/*", "*"),
    "typescript": ("//", "/*", "*"),
    "tsx": ("//", "/*", "*"),
}

_parsers: dict[str, object] = {}
_parser_lock = threading.Lock()


def get_parser_for(language: str):
    """Return a cached tree-sitter parser, or None if the grammar is unavailable."""
    if language in _parsers:
        return _parsers[language]

    with _parser_lock:
        if language in _parsers:
            return _parsers[language]
        try:
            from tree_sitter_language_pack import get_parser

            parser = get_parser(language)
        except Exception:
            logger.warning("tree-sitter grammar unavailable for %s", language, exc_info=True)
            parser = None
        _parsers[language] = parser
        return parser


def _is_comment_line(line: str, language: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    return stripped.startswith(_COMMENT_PREFIXES.get(language, ("#",)))


def _unwrap(node):
    """Follow wrapper nodes down to the definition they carry."""
    current = node
    for _ in range(4):
        if current.type not in _UNWRAP_TYPES:
            return current
        payload = [c for c in current.named_children if c.type in DEFINITION_TYPES]
        if not payload:
            return current
        current = payload[0]
    return current


def _symbol_of(node) -> str | None:
    target = _unwrap(node)

    name = target.child_by_field_name("name")
    if name is not None:
        return name.text.decode("utf-8", "replace")

    # const handler = () => {...}  — the name lives on the declarator.
    for child in target.named_children:
        if child.type in ("variable_declarator", "public_field_definition"):
            inner = child.child_by_field_name("name")
            if inner is not None:
                return inner.text.decode("utf-8", "replace")

    return None


def _is_definition(node) -> bool:
    if node.type not in DEFINITION_TYPES:
        return False
    # A bare `const x = 1` is not a definition worth its own chunk; only treat
    # lexical declarations as definitions when they bind a function or class.
    if node.type in ("lexical_declaration", "variable_declaration"):
        return _binds_callable(node)
    if node.type == "export_statement":
        inner = _unwrap(node)
        return inner is not node and _is_definition(inner)
    return True


def _binds_callable(node) -> bool:
    for declarator in node.named_children:
        if declarator.type != "variable_declarator":
            continue
        value = declarator.child_by_field_name("value")
        if value is not None and value.type in (
            "arrow_function",
            "function_expression",
            "function",
            "class",
        ):
            return True
    return False


def _make(path, start_row, end_row, lines, symbol, language) -> dict:
    return {
        "path": path,
        "start_line": start_row + 1,
        "end_line": end_row + 1,
        "content": "\n".join(lines[start_row : end_row + 1]),
        "symbol": symbol,
        "language": language,
    }


def _window_range(path, start_row, end_row, lines, symbol, language) -> list[dict]:
    """Line-window a row range, keeping absolute line numbers."""
    sub = "\n".join(lines[start_row : end_row + 1])
    out = []
    for chunk in line_window_chunks(path, sub, language=language):
        chunk["start_line"] += start_row
        chunk["end_line"] += start_row
        chunk["symbol"] = symbol
        out.append(chunk)
    return out


def _emit_definition(node, start_row, end_row, path, lines, language, prefix="") -> list[dict]:
    """Emit one definition, splitting it if it exceeds MAX_CHUNK_LINES."""
    symbol = _symbol_of(node)
    qualified = f"{prefix}{symbol}" if symbol else (prefix.rstrip(".") or None)

    if end_row - start_row + 1 <= MAX_CHUNK_LINES:
        return [_make(path, start_row, end_row, lines, qualified, language)]

    target = _unwrap(node)
    body = target.child_by_field_name("body")
    inner_defs = (
        [c for c in body.named_children if _is_definition(c)] if body is not None else []
    )

    if not inner_defs:
        # A single oversized function with no nested definitions. Windowing keeps
        # it retrievable rather than storing one enormous chunk.
        return _window_range(path, start_row, end_row, lines, qualified, language)

    chunks: list[dict] = []
    header_end = inner_defs[0].start_point[0] - 1
    if header_end >= start_row:
        chunks.append(_make(path, start_row, header_end, lines, qualified, language))

    child_prefix = f"{qualified}." if qualified else ""
    cursor = inner_defs[0].start_point[0]
    for child in inner_defs:
        c_start = child.start_point[0]
        c_end = child.end_point[0]
        if c_start > cursor:
            chunks.append(_make(path, cursor, c_start - 1, lines, qualified, language))
        chunks.extend(
            _emit_definition(child, c_start, c_end, path, lines, language, child_prefix)
        )
        cursor = c_end + 1

    if cursor <= end_row:
        chunks.append(_make(path, cursor, end_row, lines, qualified, language))

    return chunks


def ast_chunks(path: str, text: str, language: str) -> list[dict] | None:
    """Chunk source by its syntax tree, or None when that cannot be done safely."""
    parser = get_parser_for(language)
    if parser is None:
        return None

    try:
        tree = parser.parse(text.encode("utf-8", "replace"))
    except Exception:
        logger.warning("tree-sitter parse failed for %s", path, exc_info=True)
        return None

    root = tree.root_node
    if root is None:
        return None

    # A parse error anywhere means node boundaries cannot be trusted, so the
    # whole file falls back rather than risking chunks that straddle definitions.
    if root.has_error:
        return None

    lines = text.split("\n")
    last_row = len(lines) - 1
    chunks: list[dict] = []
    pending_start = 0

    for node in root.named_children:
        if not _is_definition(node):
            continue

        start_row = node.start_point[0]
        end_row = min(node.end_point[0], last_row)

        # Pull contiguous comment lines directly above into the definition.
        attach_start = start_row
        while attach_start > pending_start and _is_comment_line(
            lines[attach_start - 1], language
        ):
            attach_start -= 1

        if attach_start > pending_start:
            chunks.append(
                _make(path, pending_start, attach_start - 1, lines, None, language)
            )

        chunks.extend(
            _emit_definition(node, attach_start, end_row, path, lines, language)
        )
        pending_start = end_row + 1

    if pending_start <= last_row:
        chunks.append(_make(path, pending_start, last_row, lines, None, language))

    kept = [c for c in chunks if c["content"].strip()]
    if not kept:
        return None

    return kept
