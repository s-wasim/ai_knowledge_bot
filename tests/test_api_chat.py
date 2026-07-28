# tests/test_api_chat.py
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from app.graph.state import Citation, GradedChunk
from app.retrieval.base import ChunkData
from app.server import app


def _fake_stream(state, config, stream_mode):
    chunk = ChunkData(path="app/db.py", start_line=1, end_line=10, content="engine = ...", score=0.9)
    graded = GradedChunk(chunk=chunk, keep=True, reason="relevant")
    citation = Citation(chunk=chunk, index=1)

    yield "updates", {"rewrite_query": {"rewritten_query": state["question"]}}
    yield "updates", {"retrieve": {"retrieved": [chunk]}}
    yield "updates", {"grade_chunks": {"graded": [graded]}}
    yield "messages", (SimpleNamespace(content="The "), {"langgraph_node": "generate_answer"})
    yield "messages", (SimpleNamespace(content="answer."), {"langgraph_node": "generate_answer"})
    yield "updates", {"generate_answer": {"answer": "The answer.", "citations": [citation]}}


class TestChat:
    @patch("app.api.chat.get_graph")
    @patch("app.api.chat.get_retriever_and_mode")
    def test_streams_node_token_and_final_events(
        self, mock_get_retriever_and_mode, mock_get_graph
    ):
        mock_get_retriever_and_mode.return_value = (MagicMock(), "fts")
        fake_graph = MagicMock()
        fake_graph.stream.side_effect = _fake_stream
        mock_get_graph.return_value = fake_graph

        client = TestClient(app)
        res = client.post(
            "/chat",
            json={"repo_id": 1, "question": "Where is the DB configured?", "history": []},
        )

        assert res.status_code == 200
        text = res.text
        assert "event: node" in text
        assert '"node": "rewrite_query"' in text
        assert '"node": "generate_answer"' in text
        assert "event: token" in text
        assert '"text": "The "' in text
        assert "event: final" in text
        assert '"answer": "The answer."' in text
        assert '"path": "app/db.py"' in text

    @patch("app.api.chat.get_graph")
    @patch("app.api.chat.get_retriever_and_mode")
    def test_streams_ignore_thinking_block_content(
        self, mock_get_retriever_and_mode, mock_get_graph
    ):
        """A thinking-block chunk (list-shaped content) must be skipped in the
        SSE token stream instead of being forwarded as-is or crashing."""
        mock_get_retriever_and_mode.return_value = (MagicMock(), "fts")

        def _fake_stream_with_thinking(state, config, stream_mode):
            chunk = ChunkData(path="app/db.py", start_line=1, end_line=10, content="engine = ...", score=0.9)
            yield "updates", {"retrieve": {"retrieved": [chunk]}}
            yield "messages", (
                SimpleNamespace(content=[{"type": "thinking", "thinking": "hmm", "signature": "abc"}]),
                {"langgraph_node": "generate_answer"},
            )
            yield "messages", (SimpleNamespace(content="The answer."), {"langgraph_node": "generate_answer"})
            yield "updates", {"generate_answer": {"answer": "The answer.", "citations": []}}

        fake_graph = MagicMock()
        fake_graph.stream.side_effect = _fake_stream_with_thinking
        mock_get_graph.return_value = fake_graph

        client = TestClient(app)
        res = client.post(
            "/chat",
            json={"repo_id": 1, "question": "Where is the DB configured?", "history": []},
        )

        assert res.status_code == 200
        text = res.text
        assert "event: token" in text
        assert '"text": "The answer."' in text
        assert "thinking" not in text
        assert "event: error" not in text

    @patch("app.api.chat.get_graph")
    @patch("app.api.chat.get_retriever_and_mode")
    def test_graph_exception_emits_error_event(
        self, mock_get_retriever_and_mode, mock_get_graph
    ):
        mock_get_retriever_and_mode.return_value = (MagicMock(), "fts")
        fake_graph = MagicMock()
        fake_graph.stream.side_effect = RuntimeError("llm unavailable")
        mock_get_graph.return_value = fake_graph

        client = TestClient(app)
        res = client.post(
            "/chat", json={"repo_id": 1, "question": "hi?", "history": []}
        )

        assert res.status_code == 200
        assert "event: error" in res.text
        assert "llm unavailable" in res.text
