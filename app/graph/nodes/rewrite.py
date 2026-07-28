"""Turn a conversational question into a standalone search query.

Follow-ups like "how do I change it to use MySQL instead?" carry their subject in
the history, not the question. Retrieval sees only the query string, so the
pronoun has to be resolved before search happens.

Prior turns are replayed with their real roles. They were previously all wrapped
in HumanMessage with an "Assistant: " text prefix, leaving the model to infer
speaker boundaries from prose — on the exact path this node exists to get right.
"""

from __future__ import annotations

import logging

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from app.graph.state import RagState
from app.llm import extract_text, get_llm

logger = logging.getLogger(__name__)

MAX_HISTORY_TURNS = 6
MAX_TURN_CHARS = 2000

SYSTEM_PROMPT = (
    "You rewrite questions for a codebase search system. Given the conversation "
    "so far and the latest question, produce one standalone search query that "
    "captures what the user wants, resolving any pronouns or references to earlier "
    "turns. Keep identifiers, file names, and technical terms exactly as written. "
    "Return only the query, with no preamble and no quotes."
)


def rewrite_query(state: RagState) -> dict:
    question = state["question"]
    history = state.get("chat_history") or []

    if not history:
        return {"rewritten_query": question}

    messages: list = [SystemMessage(content=SYSTEM_PROMPT)]

    for turn in history[-MAX_HISTORY_TURNS:]:
        content = (turn.get("content") or "")[:MAX_TURN_CHARS]
        if not content:
            continue
        role = turn.get("role")
        if role == "user":
            messages.append(HumanMessage(content=content))
        elif role == "assistant":
            messages.append(AIMessage(content=content))

    messages.append(HumanMessage(content=f"Latest question: {question}"))

    try:
        response = get_llm().invoke(messages)
        rewritten = extract_text(response.content).strip()
    except Exception as e:
        # Searching the raw question is far better than not searching.
        logger.warning("Query rewrite failed, using the original question: %s", e)
        return {"rewritten_query": question}

    if not rewritten:
        return {"rewritten_query": question}

    logger.info("Rewrote %r to %r", question, rewritten)
    return {"rewritten_query": rewritten}
