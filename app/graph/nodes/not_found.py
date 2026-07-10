import logging
from collections import Counter

from langchain_core.runnables import RunnableConfig

from app.db import Chunk
from app.graph.state import RagState

logger = logging.getLogger(__name__)

SUGGESTION_COUNT = 2


def _top_directories(config: RunnableConfig | None, repo_id) -> list[str]:
    """Top directories by chunk count across the whole indexed repo (FR-9:
    suggestions come from the index, not from the chunks the grader discarded)."""
    get_session = ((config or {}).get("configurable") or {}).get("get_session")
    if get_session is None or repo_id is None:
        return []

    try:
        session = get_session()
        try:
            paths = session.query(Chunk.path).filter(Chunk.repo_id == repo_id).all()
        finally:
            session.close()
    except Exception:
        logger.warning("Could not query index for directory suggestions", exc_info=True)
        return []

    counts = Counter()
    for (path,) in paths:
        parts = path.split("/")
        if len(parts) > 1:
            counts[parts[0]] += 1

    return [directory for directory, _ in counts.most_common(SUGGESTION_COUNT)]


def answer_not_found(state: RagState, config: RunnableConfig | None = None) -> dict:
    question = state.get("rewritten_query", state["question"])
    repo_id = state.get("repo_id")

    directories = _top_directories(config, repo_id)
    suggestions = ", ".join(directories) if directories else "the codebase"

    answer = (
        f"I couldn't find information about '{question}' in the indexed codebase. "
        f"You might want to look in {suggestions} for relevant files, "
        f"or try rephrasing your question with more specific terms."
    )

    return {"answer": answer, "citations": []}
