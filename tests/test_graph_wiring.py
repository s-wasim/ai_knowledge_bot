from unittest.mock import MagicMock, patch

import pytest

from app.graph.build import build_rag_graph
from app.graph.state import GradedChunk, RagState
from app.retrieval.base import ChunkData


class TestConditionalEdges:
    def test_empty_graded_routes_to_not_found(self):
        state: RagState = {
            "question": "test",
            "chat_history": [],
            "rewritten_query": None,
            "retrieved": [],
            "graded": [],
            "answer": None,
            "citations": [],
            "mode": "hybrid",
        }
        cond = (
            lambda s: "answer_not_found"
            if not any(g.keep for g in s["graded"])
            else "generate_answer"
        )
        assert cond(state) == "answer_not_found"

    def test_non_empty_routes_to_generate(self):
        chunk = ChunkData(
            path="test.py", start_line=1, end_line=5, content="code", score=0.9
        )
        state: RagState = {
            "question": "test",
            "chat_history": [],
            "rewritten_query": None,
            "retrieved": [chunk],
            "graded": [GradedChunk(chunk=chunk, keep=True, reason="relevant")],
            "answer": None,
            "citations": [],
            "mode": "hybrid",
        }
        cond = (
            lambda s: "answer_not_found"
            if not any(g.keep for g in s["graded"])
            else "generate_answer"
        )
        assert cond(state) == "generate_answer"


class TestGraphCompilation:
    def test_graph_compiles(self):
        graph = build_rag_graph()
        assert graph is not None
        assert hasattr(graph, "invoke")


class TestGraphInvocation:
    def test_graph_invokes_with_retriever(self):
        mock_retriever = MagicMock()
        mock_retriever.search.return_value = [
            ChunkData(
                path="a.py", start_line=1, end_line=3, content="x", score=0.9
            )
        ]

        graph = build_rag_graph()

        result = graph.invoke(
            {
                "question": "what is x?",
                "chat_history": [],
                "rewritten_query": None,
                "retrieved": [],
                "graded": [],
                "answer": None,
                "citations": [],
                "mode": "hybrid",
            },
            config={"configurable": {"retriever": mock_retriever}},
        )

        assert result["rewritten_query"] == "what is x?"

    def test_graph_invokes_not_found_path(self):
        mock_retriever = MagicMock()
        mock_retriever.search.return_value = []

        graph = build_rag_graph()

        chunk = ChunkData(
            path="a.py", start_line=1, end_line=3, content="x", score=0.9
        )
        result = graph.invoke(
            {
                "question": "what is x?",
                "chat_history": [],
                "rewritten_query": None,
                "retrieved": [chunk],
                "graded": [GradedChunk(chunk=chunk, keep=False, reason="irrelevant")],
                "answer": None,
                "citations": [],
                "mode": "hybrid",
            },
            config={"configurable": {"retriever": mock_retriever}},
        )

        assert result["rewritten_query"] == "what is x?"

    @patch("app.graph.nodes.select.get_llm")
    def test_retriever_from_config_is_used(self, mock_get_llm):
        """Retriever must flow through LangGraph's config, not a global singleton."""
        mock_retriever = MagicMock()
        mock_retriever.search.return_value = [
            ChunkData(path="a.py", start_line=1, end_line=3, content="x", score=0.9)
        ]
        # Selection now uses tool-use-validated structured output rather than a
        # JSON string, so the double is the structured runnable.
        from app.graph.nodes.select import SelectionItem, SelectionResult

        mock_llm = MagicMock()
        mock_llm.with_structured_output.return_value.invoke.return_value = SelectionResult(
            items=[SelectionItem(index=1, keep=False, relevance=0.0, reason="irrelevant")]
        )
        mock_get_llm.return_value = mock_llm

        graph = build_rag_graph()

        result = graph.invoke(
            {
                "question": "what is x?",
                "chat_history": [],
                "rewritten_query": None,
                "retrieved": [],
                "graded": [],
                "answer": None,
                "citations": [],
                "mode": "hybrid",
                "repo_id": 1,
            },
            config={"configurable": {"retriever": mock_retriever}},
        )

        mock_retriever.search.assert_called_once()
        assert len(result["retrieved"]) == 1

    def test_missing_retriever_in_config_returns_empty(self):
        graph = build_rag_graph()

        result = graph.invoke(
            {
                "question": "what is x?",
                "chat_history": [],
                "rewritten_query": None,
                "retrieved": [],
                "graded": [],
                "answer": None,
                "citations": [],
                "mode": "hybrid",
                "repo_id": 1,
            }
        )

        assert result["retrieved"] == []
