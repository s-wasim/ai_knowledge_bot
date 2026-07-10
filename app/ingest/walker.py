import logging
import os
from pathlib import Path
from typing import Generator, Set, Tuple, Optional

DEFAULT_ALLOWLIST = {'.py', '.ts', '.tsx', '.js', '.md', '.sql', '.yaml', '.toml'}
EXCLUDED_DIRS = {'node_modules', '.git', 'dist', 'venv', '__pycache__'}
MAX_FILE_SIZE = 200 * 1024
BINARY_CHECK_SIZE = 8 * 1024


def walk_directory(
    root_dir: Path | str,
    allowlist: Optional[Set[str]] = None,
) -> Generator[Tuple[Path, str], None, None]:
    root = Path(root_dir).resolve()
    if not root.exists():
        raise FileNotFoundError(f"Directory not found: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"Not a directory: {root}")

    allowlist = DEFAULT_ALLOWLIST if allowlist is None else allowlist

    for dirpath, dirnames, filenames in os.walk(root, topdown=True, followlinks=False):
        dirnames[:] = [d for d in dirnames if d not in EXCLUDED_DIRS]

        for filename in filenames:
            filepath = Path(dirpath) / filename

            if filepath.is_symlink():
                continue

            if filepath.suffix not in allowlist:
                continue

            try:
                size = os.path.getsize(filepath)
            except OSError:
                continue

            if size > MAX_FILE_SIZE:
                continue

            try:
                with open(filepath, 'rb') as f:
                    header = f.read(BINARY_CHECK_SIZE)
                if b'\0' in header:
                    continue
            except OSError:
                continue

            try:
                text = filepath.read_text(encoding='utf-8')
                yield (filepath, text)
            except UnicodeDecodeError:
                logging.warning(f"Skipping non-UTF8 file: {filepath}")
                continue
            except OSError:
                continue
