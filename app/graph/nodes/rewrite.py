from langchain_core.messages import HumanMessage, SystemMessage

from app.graph.state import RagState
from app.llm import get_llm


def rewrite_query(state: RagState) -> dict:
    history = state.get("chat_history", [])
    question = state["question"]

    if not history:
        return {"rewritten_query": question}

    llm = get_llm()

    messages = [
        SystemMessage(
            content="You are a query rewriter for a codebase RAG system. "
            "Given the conversation history and the latest user question, "
            "produce a standalone search query that captures the user's intent. "
            "Return ONLY the rewritten query, nothing else."
        ),
    ]

    for turn in history[-6:]:
        if turn.get("role") == "user":
            messages.append(HumanMessage(content=turn.get("content", "")))
        elif turn.get("role") == "assistant":
            messages.append(HumanMessage(content=f"Assistant: {turn.get('content', '')}"))

    messages.append(HumanMessage(content=f"Current question: {question}"))

    response = llm.invoke(messages)
    rewritten = response.content.strip()

    if not rewritten:
        rewritten = question

    return {"rewritten_query": rewritten}
