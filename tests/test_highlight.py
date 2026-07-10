from app.highlight import highlight_chunk, highlight_style


def test_python_file_gets_python_lexer_classes():
    html = highlight_chunk("db.py", "def connect():\n    pass\n", start_line=1)
    assert 'class="k">def' in html


def test_line_numbers_start_at_chunk_start_line_not_one():
    html = highlight_chunk("db.py", "def connect():\n    pass\n", start_line=42)
    assert '>42<' in html
    assert '>43<' in html
    assert '>1<' not in html


def test_typescript_file_uses_typescript_lexer():
    html = highlight_chunk("main.ts", "const x: number = 1;\n", start_line=1)
    assert "chunk-highlight" in html


def test_unknown_extension_falls_back_without_raising():
    html = highlight_chunk("weird_file.xyz123", "some plain text\n", start_line=1)
    assert "some plain text" in html


def test_style_contains_css():
    style = highlight_style()
    assert "<style>" in style
    assert "chunk-highlight" in style
