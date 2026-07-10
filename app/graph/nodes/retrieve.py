from app.graph.build import get_retriever
from app.graph.state import RagState


def retrieve(state: RagState) -> dict:
    retriever = get_retriever()
    if retriever is None:
        return {"retrieved": []}

    repo_id = state.get("repo_id", None)
    if repo_id is None:
        return {"retrieved": []}

    query = state.get("rewritten_query", state["question"])
    results = retriever.search(repo_id=repo_id, query=query, k=8)

    return {"retrieved": results}
