"""AST-aware chunking.

The point of these tests is that a chunk is a semantic unit — a whole function or
class with its signature attached — because that is what both the embedding model
and Claude's relevance grader are shown. A chunk cut mid-function is evidence
nobody can act on.
"""

import pytest

from app.ingest.chunker import chunk_file
from app.ingest.chunker.text_chunker import line_window_chunks


REQUIRED_KEYS = {"path", "start_line", "end_line", "content", "symbol", "language"}


def _assert_shape(chunks):
    assert chunks, "expected at least one chunk"
    for c in chunks:
        assert REQUIRED_KEYS <= set(c), f"missing keys: {REQUIRED_KEYS - set(c)}"
        assert c["start_line"] >= 1
        assert c["end_line"] >= c["start_line"]


class TestPython:
    def test_functions_become_separate_chunks(self):
        src = "def alpha():\n    return 1\n\n\ndef beta():\n    return 2\n"
        chunks = chunk_file("app/x.py", src)
        _assert_shape(chunks)
        assert {"alpha", "beta"} <= {c["symbol"] for c in chunks}
        assert all(c["language"] == "python" for c in chunks)

    def test_class_and_methods_are_captured(self):
        src = (
            "class Store:\n"
            "    def get(self, key):\n"
            "        return self._d[key]\n"
            "\n"
            "    def put(self, key, value):\n"
            "        self._d[key] = value\n"
        )
        chunks = chunk_file("app/store.py", src)
        _assert_shape(chunks)
        symbols = {c["symbol"] for c in chunks}
        assert "Store" in symbols or {"Store.get", "Store.put"} <= symbols

    def test_decorator_stays_with_its_function(self):
        src = "@router.get('/health')\ndef health():\n    return {}\n"
        chunks = chunk_file("app/api.py", src)
        body = next(c["content"] for c in chunks if c["symbol"] == "health")
        assert "@router.get" in body

    def test_signature_is_never_orphaned_from_body(self):
        src = "def compute(a, b):\n" + "".join(f"    a += {i}\n" for i in range(40)) + "    return a\n"
        chunks = chunk_file("app/c.py", src)
        owning = [c for c in chunks if "def compute" in c["content"]]
        assert owning, "the def line must live in a chunk"
        assert "return a" in "".join(c["content"] for c in chunks)

    def test_module_level_code_is_not_lost(self):
        src = "import os\n\nDEBUG = True\n\ndef f():\n    return DEBUG\n"
        chunks = chunk_file("app/m.py", src)
        joined = "".join(c["content"] for c in chunks)
        assert "import os" in joined
        assert "DEBUG = True" in joined

    def test_oversized_function_is_split(self):
        body = "".join(f"    x{i} = {i}\n" for i in range(300))
        chunks = chunk_file("app/big.py", f"def huge():\n{body}")
        _assert_shape(chunks)
        assert len(chunks) > 1

    def test_unparseable_source_falls_back_to_line_window(self):
        chunks = chunk_file("app/broken.py", "def (((( broken\n" * 300)
        _assert_shape(chunks)
        assert len(chunks) > 1


class TestJavaScriptFamily:
    def test_js_functions_are_chunked(self):
        src = "function alpha() {\n  return 1;\n}\n\nfunction beta() {\n  return 2;\n}\n"
        chunks = chunk_file("src/x.js", src)
        _assert_shape(chunks)
        assert {"alpha", "beta"} <= {c["symbol"] for c in chunks}
        assert all(c["language"] == "javascript" for c in chunks)

    def test_ts_class_is_chunked(self):
        src = "export class Repo {\n  find(id: number): string {\n    return '';\n  }\n}\n"
        chunks = chunk_file("src/repo.ts", src)
        _assert_shape(chunks)
        assert any(c["symbol"] and "Repo" in c["symbol"] for c in chunks)
        assert all(c["language"] == "typescript" for c in chunks)

    def test_tsx_component_is_chunked(self):
        src = "export function Card() {\n  return <div>hi</div>;\n}\n"
        chunks = chunk_file("src/Card.tsx", src)
        _assert_shape(chunks)
        assert any(c["symbol"] == "Card" for c in chunks)
        assert all(c["language"] == "tsx" for c in chunks)


class TestNonCode:
    def test_markdown_splits_on_headings(self):
        chunks = chunk_file("README.md", "# One\ntext\n\n# Two\nmore\n")
        _assert_shape(chunks)
        assert len(chunks) == 2
        assert chunks[0]["symbol"] == "One"
        assert chunks[1]["symbol"] == "Two"
        assert chunks[0]["language"] == "markdown"

    def test_markdown_preamble_before_first_heading_is_kept(self):
        chunks = chunk_file("README.md", "intro line\n\n# One\ntext\n")
        assert "intro line" in "".join(c["content"] for c in chunks)

    def test_sql_splits_on_statements(self):
        src = "CREATE TABLE a (id int);\n\nCREATE TABLE b (id int);\n"
        chunks = chunk_file("schema.sql", src)
        _assert_shape(chunks)
        assert len(chunks) == 2
        assert all(c["language"] == "sql" for c in chunks)

    def test_unknown_extension_uses_line_window(self):
        chunks = chunk_file("notes.txt", "line\n" * 200)
        _assert_shape(chunks)
        assert len(chunks) > 1
        assert all(c["symbol"] is None for c in chunks)

    def test_yaml_and_toml_are_indexed(self):
        for name in ("config.yaml", "pyproject.toml"):
            chunks = chunk_file(name, "key: value\n")
            _assert_shape(chunks)


class TestHeaderAndLineNumbers:
    def test_content_carries_synthetic_header(self):
        chunks = chunk_file("app/x.py", "def alpha():\n    return 1\n")
        first = chunks[0]
        assert first["content"].startswith("# app/x.py")
        assert "alpha" in first["content"].split("\n")[0]
        assert "def alpha" in first["content"]

    def test_header_does_not_shift_line_numbers(self):
        src = "def alpha():\n    return 1\n\n\ndef beta():\n    return 2\n"
        chunks = chunk_file("app/x.py", src)
        beta = next(c for c in chunks if c["symbol"] == "beta")
        assert src.split("\n")[beta["start_line"] - 1].startswith("def beta")

    def test_line_numbers_stay_within_the_file(self):
        src = "def alpha():\n    return 1\n"
        total = len(src.split("\n"))
        for c in chunk_file("app/x.py", src):
            assert c["end_line"] <= total


class TestEdgeCases:
    def test_empty_file_yields_no_chunks(self):
        assert chunk_file("app/x.py", "") == []

    def test_whitespace_only_file_yields_no_chunks(self):
        assert chunk_file("app/x.py", "\n\n   \n") == []

    def test_single_line_file(self):
        chunks = chunk_file("app/x.py", "x = 1")
        _assert_shape(chunks)

    def test_no_trailing_newline(self):
        chunks = chunk_file("app/x.py", "def f():\n    return 1")
        _assert_shape(chunks)
        assert "return 1" in "".join(c["content"] for c in chunks)

    def test_crlf_line_endings(self):
        chunks = chunk_file("app/x.py", "def f():\r\n    return 1\r\n")
        _assert_shape(chunks)

    def test_line_window_fallback_keeps_original_contract(self):
        chunks = line_window_chunks("a.txt", "line\n" * 200)
        assert chunks[0]["start_line"] == 1
        assert all(c["symbol"] is None for c in chunks)
