import logging

from langgraph.graph import END, StateGraph

from app.graph.state import RagState

logger = logging.getLogger(__name__)

from app.graph.retriever_holder import set_retriever, get_retriever

from app.graph.nodes.grade import grade_chunks
from app.graph.nodes.retrieve import retrieve
from app.graph.nodes.rewrite import rewrite_query
from app.graph.nodes.answer import generate_answer
from app.graph.nodes.not_found import answer_not_found


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
