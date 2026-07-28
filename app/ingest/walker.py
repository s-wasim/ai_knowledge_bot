import codecs
import logging
import os
from pathlib import Path
from typing import Generator, Set, Tuple, Optional

DEFAULT_ALLOWLIST = {'.py', '.ts', '.tsx', '.js', '.md', '.sql', '.yaml', '.toml'}
EXCLUDED_DIRS = {
    'node_modules',
    '.git',
    'dist',
    'venv',
    '__pycache__',
    '.pytest_cache',
    '.mypy_cache',
    '.ruff_cache',
    'build',
    '.tox',
}
MAX_FILE_SIZE = 200 * 1024
BINARY_CHECK_SIZE = 8 * 1024


def _resolve_root(root_dir: Path | str) -> Path:
    root = Path(root_dir).resolve()
    if not root.exists():
        raise FileNotFoundError(f"Directory not found: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"Not a directory: {root}")
    return root


def _header_is_utf8(header: bytes) -> bool:
    """Whether a leading slice of a file decodes as UTF-8.

    An incremental decoder is used so a multi-byte character straddling the end
    of the slice is treated as incomplete rather than invalid — otherwise large
    valid files would be rejected at random.
    """
    decoder = codecs.getincrementaldecoder("utf-8")()
    try:
        decoder.decode(header, final=False)
    except UnicodeDecodeError:
        return False
    return True


def _is_candidate(filepath: Path, allowlist: Set[str]) -> bool:
    """Every exclusion that does not require decoding the whole file.

    Shared by walk_directory and count_files so an ingest progress total can
    never disagree with the files actually processed.
    """
    if filepath.is_symlink():
        return False

    if filepath.suffix not in allowlist:
        return False

    try:
        size = os.path.getsize(filepath)
    except OSError:
        return False

    if size > MAX_FILE_SIZE:
        return False

    try:
        with open(filepath, 'rb') as f:
            header = f.read(BINARY_CHECK_SIZE)
    except OSError:
        return False

    if b'\0' in header:
        return False

    if not _header_is_utf8(header):
        logging.warning(f"Skipping non-UTF8 file: {filepath}")
        return False

    return True


def _iter_candidates(root: Path, allowlist: Set[str]) -> Generator[Path, None, None]:
    for dirpath, dirnames, filenames in os.walk(root, topdown=True, followlinks=False):
        dirnames[:] = [d for d in dirnames if d not in EXCLUDED_DIRS]

        for filename in sorted(filenames):
            filepath = Path(dirpath) / filename
            if _is_candidate(filepath, allowlist):
                yield filepath


def walk_directory(
    root_dir: Path | str,
    allowlist: Optional[Set[str]] = None,
) -> Generator[Tuple[Path, str], None, None]:
    root = _resolve_root(root_dir)
    allowlist = DEFAULT_ALLOWLIST if allowlist is None else allowlist

    for filepath in _iter_candidates(root, allowlist):
        try:
            text = filepath.read_text(encoding='utf-8')
        except UnicodeDecodeError:
            # Invalid bytes beyond the sampled header. Rare, and skipping keeps
            # ingest going rather than failing the whole repo.
            logging.warning(f"Skipping non-UTF8 file: {filepath}")
            continue
        except OSError:
            continue
        yield (filepath, text)


def count_files(
    root_dir: Path | str,
    allowlist: Optional[Set[str]] = None,
) -> int:
    """Count the files walk_directory would yield, without decoding any of them.

    Used for ingest progress totals. Only the first BINARY_CHECK_SIZE bytes of
    each file are inspected, so a file whose invalid UTF-8 appears past that
    point is counted here and skipped during the real walk. That is preferable to
    reading every file's full text twice.
    """
    root = _resolve_root(root_dir)
    allowlist = DEFAULT_ALLOWLIST if allowlist is None else allowlist

    return sum(1 for _ in _iter_candidates(root, allowlist))
