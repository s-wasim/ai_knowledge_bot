import json
import logging

from langchain_core.messages import HumanMessage, SystemMessage

from app.graph.state import GradedChunk, RagState
from app.llm import get_llm
from app.retrieval.base import ChunkData

logger = logging.getLogger(__name__)


def grade_chunks(state: RagState) -> dict:
    retrieved = state.get("retrieved", [])
    if not retrieved:
        return {"graded": []}

    question = state.get("rewritten_query", state["question"])

    chunks_text = ""
    for i, chunk in enumerate(retrieved, 1):
        lines = chunk.content.split('\n')
        if len(lines) > 60:
            display_content = '\n'.join(lines[:60]) + "\n... [truncated]"
        else:
            display_content = chunk.content
        chunks_text += f"\n[{i}] {chunk.path}:{chunk.start_line}-{chunk.end_line}\n```\n{display_content}\n```\n"

    system_prompt = (
        "You are a code relevance grader. Given a search query and a set of code chunks, "
        "determine which chunks are relevant to answering the query. "
        "Respond with a JSON object with key 'grades' which is a list of objects, each with:\n"
        "- 'index': int (the chunk number)\n"
        "- 'keep': bool (true if relevant, false if not)\n"
        "- 'reason': str (one-line explanation)\n"
        "Example: {\"grades\": [{\"index\": 1, \"keep\": true, \"reason\": \"Shows DB connection config\"}]}\n"
        "Return ONLY valid JSON, no other text."
    )

    user_prompt = f"Search query: {question}\n\nRelevant chunks:{chunks_text}\n\nRate each chunk [1-{len(retrieved)}] as keep or discard."

    llm = get_llm()
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ]

    try:
        response = llm.invoke(messages)
        content = response.content.strip()

        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()

        data = json.loads(content)
        grade_list = data.get("grades", [])

        graded = []
        for g in grade_list:
            idx = g.get("index", 0) - 1
            if 0 <= idx < len(retrieved):
                graded.append(GradedChunk(
                    chunk=retrieved[idx],
                    keep=bool(g.get("keep", True)),
                    reason=g.get("reason", ""),
                ))

        if not graded:
            logger.warning("Grade parsing returned empty, keeping all chunks")
            graded = [GradedChunk(chunk=c, keep=True, reason="Fallback: keep all") for c in retrieved]

        return {"graded": graded}

    except Exception as e:
        logger.error(f"Grade LLM call failed: {e}")
        graded = [GradedChunk(chunk=c, keep=True, reason=f"Fallback after error: {e}") for c in retrieved]
        return {"graded": graded}
