"""End-to-end retrieval quality over the demo questions.

Unit tests cannot catch a retrieval regression: every stage can behave correctly
in isolation while the pipeline still fails to find the file that answers a
question. These run the real graph against a real index and assert on the
citations, which is the only signal that actually matters to a user.

Needs Postgres, the embedding model, and ANTHROPIC_API_KEY. Skipped otherwise.
"""

import os
from pathlib import Path

import pytest

pytestmark = pytest.mark.eval

FIXTURE_REPO = Path(__file__).resolve().parents[2] / "fixtures" / "mini_repo"
REPO_NAME = "__eval_mini_repo__"


def _requires_stack():
    if not os.environ.get("ANTHROPIC_API_KEY"):
        pytest.skip("ANTHROPIC_API_KEY not set")


@pytest.fixture(scope="module")
def ingested():
    """Ingest the fixture repo once, then remove it."""
    _requires_stack()

    import app.db as db_module

    try:
        db_module.init_db()
    except Exception as e:
        pytest.skip(f"No Postgres available: {e}")

    from app.ingest.pipeline import ingest_repo

    repo = ingest_repo(repo_name=REPO_NAME, root_dir=str(FIXTURE_REPO))
    repo_id = repo.id

    yield repo_id

    session = db_module.get_session()()
    victim = session.query(db_module.Repo).filter_by(id=repo_id).first()
    if victim is not None:
        session.delete(victim)
        session.commit()
    session.close()


def _ask(repo_id, question, history=None):
    """Run the full graph and return (visited_nodes, answer, citation_paths)."""
    from app.api.deps import get_graph, get_retriever_and_mode
    from app.db import get_session

    retriever, mode = get_retriever_and_mode()
    graph = get_graph()

    state = {
        "question": question,
        "chat_history": history or [],
        "rewritten_query": None,
        "retrieved": [],
        "graded": [],
        "answer": None,
        "citations": [],
        "mode": mode,
        "repo_id": repo_id,
    }
    config = {"configurable": {"retriever": retriever, "get_session": get_session}}

    visited = []
    final = {}
    for chunk in graph.stream(state, config=config, stream_mode="updates"):
        for node, output in chunk.items():
            visited.append(node)
            if output:
                final.update(output)

    paths = [c.chunk.path for c in final.get("citations", [])]
    return visited, final.get("answer") or "", paths


class TestDemoQuestions:
    def test_q1_database_connection(self, ingested):
        """FR-7, FR-8: retrieval plus a cited answer."""
        visited, answer, paths = _ask(ingested, "Where is the database connection configured?")
        assert "generate_answer" in visited
        assert paths, "expected at least one citation"
        assert any("db.py" in p for p in paths), f"citations were {paths}"

    def test_q2_authentication_flow(self, ingested):
        visited, answer, paths = _ask(ingested, "How does authentication work?")
        assert "generate_answer" in visited
        assert any("login.py" in p or "auth" in p for p in paths), f"citations were {paths}"

    def test_q3_follow_up_resolves_the_pronoun(self, ingested):
        """FR-10: 'it' refers to the database connection from the previous turn."""
        history = [
            {"role": "user", "content": "Where is the database connection configured?"},
            {
                "role": "assistant",
                "content": "The connection is configured in db.py from DATABASE_URL.",
            },
        ]
        visited, answer, paths = _ask(
            ingested, "How do I change it to use MySQL instead?", history=history
        )
        assert "generate_answer" in visited
        assert any("db.py" in p or "config" in p for p in paths), f"citations were {paths}"

    def test_q4_absent_feature_is_reported_honestly(self, ingested):
        """FR-9: the fixture repo has no notification system."""
        visited, answer, paths = _ask(ingested, "How does the notification system work?")
        assert "answer_not_found" in visited, f"visited {visited}"
        assert paths == []
        assert "couldn't find" in answer.lower()

    def test_q5_environment_variables(self, ingested):
        visited, answer, paths = _ask(
            ingested, "What environment variables does the application use?"
        )
        assert "generate_answer" in visited
        assert paths, f"expected citations, answer was {answer[:200]}"


class TestEvidenceQuality:
    def test_citations_use_repo_relative_paths(self, ingested):
        _visited, _answer, paths = _ask(ingested, "Where is the database connection configured?")
        for path in paths:
            assert not path.startswith("/"), f"{path} is absolute"
            assert "/tmp/" not in path

    def test_citation_numbers_appear_in_the_answer(self, ingested):
        """A citation the prose never references would render as an orphan chip."""
        from app.api.deps import get_graph, get_retriever_and_mode
        from app.db import get_session

        retriever, mode = get_retriever_and_mode()
        state = {
            "question": "Where is the database connection configured?",
            "chat_history": [],
            "rewritten_query": None,
            "retrieved": [],
            "graded": [],
            "answer": None,
            "citations": [],
            "mode": mode,
            "repo_id": ingested,
        }
        config = {"configurable": {"retriever": retriever, "get_session": get_session}}
        final = {}
        for chunk in get_graph().stream(state, config=config, stream_mode="updates"):
            for _node, output in chunk.items():
                if output:
                    final.update(output)

        answer = final.get("answer") or ""
        for citation in final.get("citations", []):
            assert f"[{citation.index}]" in answer
