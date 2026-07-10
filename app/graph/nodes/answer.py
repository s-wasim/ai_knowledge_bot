import re

from langchain_core.messages import HumanMessage, SystemMessage

from app.graph.state import RagState, Citation
from app.llm import get_llm_streaming


def _build_answer_prompt(state: RagState) -> tuple[list, list]:
    """Returns (messages, kept_chunks)"""
    kept = [gc for gc in state.get("graded", []) if gc.keep]

    if not kept:
        return [], []

    chunks_text = ""
    for i, gc in enumerate(kept, 1):
        chunks_text += f"\n[{i}] {gc.chunk.path}:{gc.chunk.start_line}-{gc.chunk.end_line}\n```\n{gc.chunk.content}\n```\n"

    system_prompt = (
        "You are a codebase Q&A assistant. Answer the user's question using ONLY the provided code chunks. "
        "For every factual statement about the code, cite the source using [n] markers that reference the numbered chunks above. "
        "Each citation must be a valid chunk number.\n\n"
        "Rules:\n"
        "- Use [1], [2], etc. to cite chunks\n"
        "- Every citation must map to one of the provided chunk numbers\n"
        "- Do NOT fabricate file paths or line numbers\n"
        "- If the chunks don't contain enough information, say so\n\n"
        "Example response format:\n"
        "The database connection is configured in [1] using the DATABASE_URL environment variable. "
        "The connection is established via the get_connection() function [1] which supports PostgreSQL and SQLite backends [2].\n\n"
        "Chunks:\n" + chunks_text
    )

    user_prompt = state.get("rewritten_query", state["question"])

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ]

    return messages, kept


def _postprocess_answer(answer: str, kept_chunks: list) -> tuple[str, list[Citation]]:
    """Post-process answer: validate citations, strip invalid ones, build Citation list."""
    citations = []
    valid_indices = set()

    pattern = re.compile(r'\[(\d+)\]')

    for match in pattern.finditer(answer):
        idx = int(match.group(1))
        if 1 <= idx <= len(kept_chunks):
            valid_indices.add(idx)

    for idx in sorted(valid_indices):
        gc = kept_chunks[idx - 1]
        citations.append(Citation(chunk=gc.chunk, index=idx))

    def _replace_invalid(match):
        idx = int(match.group(1))
        if 1 <= idx <= len(kept_chunks):
            return match.group(0)
        return ""

    cleaned_answer = pattern.sub(_replace_invalid, answer)
    cleaned_answer = re.sub(r' +', ' ', cleaned_answer)
    cleaned_answer = cleaned_answer.strip()

    return cleaned_answer, citations


def generate_answer(state: RagState) -> dict:
    messages, kept_chunks = _build_answer_prompt(state)

    if not kept_chunks:
        return {"answer": "I couldn't find any relevant code chunks to answer your question.", "citations": []}

    llm = get_llm_streaming()
    raw_answer = "".join(chunk.content for chunk in llm.stream(messages)).strip()

    cleaned_answer, citations = _postprocess_answer(raw_answer, kept_chunks)

    return {"answer": cleaned_answer, "citations": citations}
