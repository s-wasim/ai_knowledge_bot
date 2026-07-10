import logging

from app.graph.state import RagState

logger = logging.getLogger(__name__)


def answer_not_found(state: RagState) -> dict:
    question = state.get("rewritten_query", state["question"])

    retrieved = state.get("retrieved", [])
    directories = set()
    for chunk in retrieved:
        parts = chunk.path.split('/')
        if len(parts) > 1:
            directories.add(parts[0])

    suggestions = ", ".join(sorted(directories)[:3]) if directories else "the codebase"

    answer = (
        f"I couldn't find information about '{question}' in the indexed codebase. "
        f"You might want to look in {suggestions} for relevant files, "
        f"or try rephrasing your question with more specific terms."
    )

    return {"answer": answer, "citations": []}
