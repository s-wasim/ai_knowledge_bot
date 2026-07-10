import logging
import os
import shutil
import tempfile
import zipfile
from io import BytesIO
from urllib.parse import urlparse

import httpx

from app.ingest.pipeline import ingest_repo

logger = logging.getLogger(__name__)


def parse_github_url(url: str) -> tuple[str, str, str]:
    """Parse a GitHub URL into (owner, repo, branch).

    Handles formats:
    - https://github.com/owner/repo
    - https://github.com/owner/repo/
    - https://github.com/owner/repo.git
    - https://github.com/owner/repo/tree/branch
    """
    url = url.rstrip("/")
    if url.endswith(".git"):
        url = url[:-4]

    parsed = urlparse(url)

    parts = parsed.path.strip("/").split("/")

    if len(parts) < 2:
        raise ValueError(
            f"Invalid GitHub URL: {url}. Expected https://github.com/owner/repo"
        )

    owner = parts[0]
    repo = parts[1]
    branch = "main"

    if len(parts) >= 4 and parts[2] == "tree":
        branch = parts[3]

    return owner, repo, branch


def _download_zip(owner: str, repo: str, branch: str) -> bytes:
    """Download a GitHub repo as a zip archive."""
    url = f"https://codeload.github.com/{owner}/{repo}/zip/refs/heads/{branch}"

    with httpx.Client(follow_redirects=True, timeout=120.0) as client:
        response = client.get(url)

        if response.status_code == 404 and branch == "main":
            branch = "master"
            url = f"https://codeload.github.com/{owner}/{repo}/zip/refs/heads/{branch}"
            response = client.get(url)

        response.raise_for_status()
        return response.content


def _extract_zip(zip_content: bytes) -> str:
    """Extract zip content to a temp directory and return the path."""
    temp_dir = tempfile.mkdtemp(prefix="gh_ingest_")
    with zipfile.ZipFile(BytesIO(zip_content)) as zf:
        zf.extractall(temp_dir)

    content_dir = None
    for d in os.listdir(temp_dir):
        full_path = os.path.join(temp_dir, d)
        if os.path.isdir(full_path):
            content_dir = full_path
            break

    if content_dir is None:
        content_dir = temp_dir

    return content_dir


def ingest_github_url(
    url: str,
    branch: str | None = None,
    progress_callback=None,
) -> tuple:
    """Ingest a repo from a GitHub URL.

    Returns (repo_name, file_count, chunk_count).
    """
    owner, repo, parsed_branch = parse_github_url(url)
    effective_branch = branch or parsed_branch

    logger.info(f"Downloading {owner}/{repo} branch={effective_branch}")

    zip_content = _download_zip(owner, repo, effective_branch)

    content_dir = _extract_zip(zip_content)

    try:
        source_url = f"https://github.com/{owner}/{repo}"
        repo_obj = ingest_repo(
            repo_name=repo,
            root_dir=content_dir,
            source_url=source_url,
            branch=effective_branch,
            progress_callback=progress_callback,
        )
        return repo_obj
    finally:
        shutil.rmtree(os.path.dirname(content_dir), ignore_errors=True)
