from langchain_core.runnables import RunnableConfig

from app.graph.state import RagState


def retrieve(state: RagState, config: RunnableConfig | None = None) -> dict:
    retriever = ((config or {}).get("configurable") or {}).get("retriever")
    if retriever is None:
        return {"retrieved": []}

    repo_id = state.get("repo_id", None)
    if repo_id is None:
        return {"retrieved": []}

    query = state.get("rewritten_query", state["question"])
    results = retriever.search(repo_id=repo_id, query=query, k=8)

    return {"retrieved": results}
