import pytest
from app.ingest.github import parse_github_url


def test_simple_url():
    owner, repo, branch = parse_github_url("https://github.com/owner/repo")
    assert owner == "owner"
    assert repo == "repo"
    assert branch == "main"


def test_trailing_slash():
    owner, repo, branch = parse_github_url("https://github.com/owner/repo/")
    assert owner == "owner"
    assert repo == "repo"


def test_git_suffix():
    owner, repo, branch = parse_github_url("https://github.com/owner/repo.git")
    assert owner == "owner"
    assert repo == "repo"


def test_branch_url():
    owner, repo, branch = parse_github_url(
        "https://github.com/owner/repo/tree/develop"
    )
    assert owner == "owner"
    assert repo == "repo"
    assert branch == "develop"


def test_invalid_url():
    with pytest.raises(ValueError):
        parse_github_url("https://github.com/owner")
