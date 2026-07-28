"""Generate the answer, grounded in the kept chunks.

The `[n]` markers the model writes are the product: they are what makes an answer
checkable. So they are validated against the chunks actually supplied, invalid
markers are stripped, and each surviving Citation records the number as written —
letting the UI label a chip `[2]` when the prose says `[2]`.
"""

from __future__ import annotations

import logging
import re

from langchain_core.messages import HumanMessage, SystemMessage

from app.graph.state import Citation, RagState, effective_query
from app.llm import extract_text, get_llm_streaming

logger = logging.getLogger(__name__)

CITATION_RE = re.compile(r"\[(\d+)\]")

SYSTEM_PROMPT = (
    "You answer questions about a codebase using only the numbered chunks below.\n\n"
    "Rules:\n"
    "- Cite with [n] markers that refer to the chunk numbers you were given.\n"
    "- Every factual claim about the code needs a citation.\n"
    "- Never invent file paths, line numbers, or chunk numbers.\n"
    "- If the chunks do not contain enough information, say so plainly rather than "
    "filling the gap from general knowledge.\n"
    "- Be concise. Lead with the direct answer, then the supporting detail.\n\n"
    "Example:\n"
    "The database connection is configured in [1] from the DATABASE_URL environment "
    "variable, and the engine is created with pool_pre_ping enabled [1]. Sessions are "
    "handed out by get_session() [2].\n\n"
    "Chunks:\n"
)


def _build_answer_prompt(state: RagState) -> tuple[list, list]:
    """Returns (messages, kept_chunks). Empty messages when there is no evidence."""
    kept = [gc for gc in state.get("graded", []) if gc.keep]

    if not kept:
        return [], []

    chunks_text = ""
    for i, gc in enumerate(kept, 1):
        chunk = gc.chunk
        label = f"[{i}] {chunk.path}:{chunk.start_line}-{chunk.end_line}"
        if chunk.symbol:
            label += f"  ({chunk.symbol})"
        chunks_text += f"\n{label}\n```\n{chunk.content}\n```\n"

    messages = [
        SystemMessage(content=SYSTEM_PROMPT + chunks_text),
        HumanMessage(content=effective_query(state)),
    ]

    return messages, kept


def _postprocess_answer(answer: str, kept_chunks: list) -> tuple[str, list[Citation]]:
    """Strip invalid citations and build the Citation list.

    Only lines that actually lost a marker get whitespace repair. A global
    `re.sub(r' +', ' ')` used to run over the whole answer, flattening the
    indentation of any code the model included.
    """
    valid_numbers: set[int] = set()

    for match in CITATION_RE.finditer(answer):
        number = int(match.group(1))
        if 1 <= number <= len(kept_chunks):
            valid_numbers.add(number)

    citations = [
        Citation(chunk=kept_chunks[number - 1].chunk, index=number)
        for number in sorted(valid_numbers)
    ]

    repaired_lines = []
    for line in answer.split("\n"):
        cleaned = CITATION_RE.sub(
            lambda m: m.group(0) if 1 <= int(m.group(1)) <= len(kept_chunks) else "",
            line,
        )
        if cleaned != line:
            # Removing "[9]" leaves a double space, or a space before punctuation.
            cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
            cleaned = re.sub(r" +([.,;:)])", r"\1", cleaned)
            cleaned = cleaned.rstrip()
        repaired_lines.append(cleaned)

    return "\n".join(repaired_lines).strip(), citations


def generate_answer(state: RagState) -> dict:
    messages, kept_chunks = _build_answer_prompt(state)

    if not kept_chunks:
        return {
            "answer": "I couldn't find any relevant code chunks to answer your question.",
            "citations": [],
        }

    llm = get_llm_streaming()
    # Streamed so the SSE layer can forward tokens as they arrive. The joined text
    # is what gets post-processed and stored.
    raw_answer = "".join(
        extract_text(chunk.content) for chunk in llm.stream(messages)
    ).strip()

    cleaned_answer, citations = _postprocess_answer(raw_answer, kept_chunks)

    if not cleaned_answer:
        logger.warning("Model returned an empty answer for %d chunks", len(kept_chunks))
        cleaned_answer = (
            "I found relevant code but could not produce an answer from it. "
            "The retrieved chunks are listed below."
        )

    return {"answer": cleaned_answer, "citations": citations}
