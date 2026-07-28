import pytest
from pathlib import Path

from app.ingest.walker import walk_directory

FIXTURE_DIR = Path(__file__).parent.parent / "fixtures" / "mini_repo"


def test_valid_files_found():
    result = {p.relative_to(FIXTURE_DIR) for p, _ in walk_directory(FIXTURE_DIR)}
    expected = {
        Path("db.py"),
        Path("auth/login.py"),
        Path("auth/__init__.py"),
        Path("main.ts"),
        Path("utils/helpers.py"),
        Path("utils/__init__.py"),
        Path("config.yaml"),
        Path("README.md"),
        Path("schema.sql"),
        Path("pyproject.toml"),
    }
    assert result == expected


def test_oversized_file_excluded():
    paths = {p for p, _ in walk_directory(FIXTURE_DIR)}
    oversized = FIXTURE_DIR / "junk_large_file.py"
    assert oversized not in paths


def test_node_modules_excluded():
    paths = {p for p, _ in walk_directory(FIXTURE_DIR)}
    assert not any("node_modules" in p.parts for p in paths)


def test_git_excluded():
    paths = {p for p, _ in walk_directory(FIXTURE_DIR)}
    assert not any(".git" in p.parts for p in paths)


def test_dist_excluded():
    paths = {p for p, _ in walk_directory(FIXTURE_DIR)}
    assert not any("dist" in p.parts for p in paths)


def test_venv_excluded():
    paths = {p for p, _ in walk_directory(FIXTURE_DIR)}
    assert not any("venv" in p.parts for p in paths)


def test_non_utf8_skipped():
    """Non-allowlisted extension (.bin) is excluded before reaching the decode step."""
    paths = {p for p, _ in walk_directory(FIXTURE_DIR)}
    non_utf8 = FIXTURE_DIR / "non_utf8_file.bin"
    assert non_utf8 not in paths


def test_non_utf8_allowlisted_extension_skipped_without_raising(caplog):
    """A file with an allowlisted extension (.py) but invalid UTF-8 bytes must be
    tolerated: skipped with a logged warning, not raised (FR-3)."""
    paths = {p for p, _ in walk_directory(FIXTURE_DIR)}
    bad_encoding = FIXTURE_DIR / "bad_encoding.py"
    assert bad_encoding not in paths
    assert any("bad_encoding.py" in message for message in caplog.messages)


def test_custom_allowlist():
    result = {p.relative_to(FIXTURE_DIR) for p, _ in walk_directory(FIXTURE_DIR, allowlist={'.yaml'})}
    assert result == {Path("config.yaml")}


def test_nonexistent_directory():
    with pytest.raises(FileNotFoundError):
        list(walk_directory("/nonexistent/path/12345"))


def test_count_files_matches_walk_directory():
    """count_files must apply exactly the same filters as walk_directory,
    otherwise the ingest progress total drifts from the work actually done."""
    from app.ingest.walker import count_files

    assert count_files(FIXTURE_DIR) == len(list(walk_directory(FIXTURE_DIR)))


def test_count_files_does_not_read_file_bodies(monkeypatch):
    """The pre-walk exists to make progress totals cheap. Reading every file's
    text just to count them doubles ingest I/O on large repos."""
    from pathlib import Path as _Path

    from app.ingest.walker import count_files

    def explode(*args, **kwargs):
        raise AssertionError("count_files must not read file contents")

    monkeypatch.setattr(_Path, "read_text", explode)
    assert count_files(FIXTURE_DIR) > 0


def test_count_files_respects_allowlist():
    from app.ingest.walker import count_files

    assert count_files(FIXTURE_DIR, allowlist={".sql"}) == 1
