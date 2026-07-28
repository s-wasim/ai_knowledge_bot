"""The honest path: say the codebase does not cover this.

Suggestions come from the index as a whole, not from the chunks the grader just
rejected — those are known to be irrelevant, so pointing at them would be worse
than saying nothing.
"""

import logging
from collections import Counter

from typing import Optional

from langchain_core.runnables import RunnableConfig

from app.db import Chunk
from app.graph.state import RagState, effective_query

logger = logging.getLogger(__name__)

SUGGESTION_COUNT = 2


def _top_directories(config: Optional[RunnableConfig], repo_id) -> list[str]:
    """Top-level directories by chunk count across the whole indexed repo."""
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

    counts: Counter = Counter()
    for (path,) in paths:
        # Empty segments are filtered out. Absolute paths used to be stored here,
        # and "/tmp/x".split("/")[0] is "", which rendered as
        # "look in  for relevant files".
        parts = [part for part in (path or "").split("/") if part]
        if len(parts) > 1:
            counts[parts[0]] += 1

    return [directory for directory, _ in counts.most_common(SUGGESTION_COUNT)]


def answer_not_found(state: RagState, config: Optional[RunnableConfig] = None) -> dict:
    question = effective_query(state)
    repo_id = state.get("repo_id")

    directories = _top_directories(config, repo_id)

    if directories:
        where = (
            "You might want to look in "
            + ", ".join(directories)
            + " for relevant files, or try"
        )
    else:
        where = "Try"

    answer = (
        f"I couldn't find information about '{question}' in the indexed codebase. "
        f"{where} rephrasing your question with more specific terms — a function "
        f"name, file name, or configuration key tends to work best."
    )

    return {"answer": answer, "citations": []}
