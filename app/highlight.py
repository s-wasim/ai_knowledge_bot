from pygments import highlight
from pygments.formatters import HtmlFormatter
from pygments.lexers import TextLexer, get_lexer_for_filename
from pygments.util import ClassNotFound

_CSS_CLASS = "chunk-highlight"


def highlight_style() -> str:
    """CSS for chunk_html() output; safe to include multiple times per page."""
    return f"<style>{HtmlFormatter(cssclass=_CSS_CLASS).get_style_defs('.' + _CSS_CLASS)}</style>"


def highlight_chunk(path: str, content: str, start_line: int) -> str:
    """Render a code chunk as syntax-highlighted HTML, gutter numbered from the
    chunk's real file line (not 1), using the language inferred from `path`."""
    try:
        lexer = get_lexer_for_filename(path, stripnl=False)
    except ClassNotFound:
        lexer = TextLexer(stripnl=False)

    formatter = HtmlFormatter(
        cssclass=_CSS_CLASS, linenos="inline", linenostart=start_line
    )
    return highlight(content, lexer, formatter)
