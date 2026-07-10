import logging

from langgraph.graph import END, StateGraph

from app.graph.state import RagState

logger = logging.getLogger(__name__)

_retriever = None


def set_retriever(retriever):
    global _retriever
    _retriever = retriever


def get_retriever():
    return _retriever


def rewrite_query(state: RagState) -> dict:
    logger.info("rewrite_query: passing question through as rewritten_query")
    return {"rewritten_query": state["question"]}


def retrieve(state: RagState) -> dict:
    logger.info("retrieve: stub — no-op")
    return {}


def grade_chunks(state: RagState) -> dict:
    logger.info("grade_chunks: stub — no-op")
    return {}


def answer_not_found(state: RagState) -> dict:
    logger.info("answer_not_found: stub — no-op")
    return {}


def generate_answer(state: RagState) -> dict:
    logger.info("generate_answer: stub — no-op")
    return {}


def build_rag_graph():
    workflow = StateGraph(RagState)

    workflow.add_node("rewrite_query", rewrite_query)
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("grade_chunks", grade_chunks)
    workflow.add_node("answer_not_found", answer_not_found)
    workflow.add_node("generate_answer", generate_answer)

    workflow.set_entry_point("rewrite_query")
    workflow.add_edge("rewrite_query", "retrieve")
    workflow.add_edge("retrieve", "grade_chunks")

    workflow.add_conditional_edges(
        "grade_chunks",
        lambda state: "answer_not_found" if not any(g.keep for g in state["graded"]) else "generate_answer",
        {
            "answer_not_found": "answer_not_found",
            "generate_answer": "generate_answer",
        },
    )

    workflow.add_edge("answer_not_found", END)
    workflow.add_edge("generate_answer", END)

    return workflow.compile()
