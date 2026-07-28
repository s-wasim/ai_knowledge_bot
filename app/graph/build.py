"""RAG graph wiring.

rewrite_query -> retrieve -> select_chunks -> {generate_answer | answer_not_found}

The node names are part of the contract: the frontend renders them as pipeline
stages as they stream in, so renaming one means updating NODE_INDEX in
FrontendDesign/KnowledgeBot.dc.html.
"""

import logging

from langgraph.graph import END, StateGraph

from app.graph.nodes.answer import generate_answer
from app.graph.nodes.not_found import answer_not_found
from app.graph.nodes.retrieve import retrieve
from app.graph.nodes.rewrite import rewrite_query
from app.graph.nodes.select import select_chunks
from app.graph.state import RagState

logger = logging.getLogger(__name__)


def _has_evidence(state: RagState) -> str:
    """Route to the answer only when at least one chunk survived selection."""
    graded = state.get("graded") or []
    if any(gc.keep for gc in graded):
        return "generate_answer"
    return "answer_not_found"


def build_rag_graph():
    workflow = StateGraph(RagState)

    workflow.add_node("rewrite_query", rewrite_query)
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("select_chunks", select_chunks)
    workflow.add_node("generate_answer", generate_answer)
    workflow.add_node("answer_not_found", answer_not_found)

    workflow.set_entry_point("rewrite_query")
    workflow.add_edge("rewrite_query", "retrieve")
    workflow.add_edge("retrieve", "select_chunks")

    workflow.add_conditional_edges(
        "select_chunks",
        _has_evidence,
        {
            "generate_answer": "generate_answer",
            "answer_not_found": "answer_not_found",
        },
    )

    workflow.add_edge("generate_answer", END)
    workflow.add_edge("answer_not_found", END)

    return workflow.compile()
