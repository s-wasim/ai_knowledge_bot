from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from app.graph.nodes.answer import generate_answer
from app.graph.state import GradedChunk, RagState
from app.retrieval.base import ChunkData


def _make_state(chunks):
    graded = [GradedChunk(chunk=c, keep=True, reason="relevant") for c in chunks]
    return RagState(
        question="question",
        chat_history=[],
        rewritten_query="question",
        retrieved=chunks,
        graded=graded,
        answer=None,
        citations=[],
        mode="hybrid",
        repo_id=1,
    )


@patch("app.graph.nodes.answer.get_llm_streaming")
def test_stream_with_thinking_block_does_not_crash(mock_get_llm_streaming):
    """Claude can emit extended-thinking content blocks mid-stream even when
    `thinking` was never requested; langchain_anthropic represents those as
    list-shaped chunk.content instead of str, mixed in with plain str text
    chunks. generate_answer must skip the non-text blocks and still assemble
    the visible answer instead of crashing on `"".join(...)`.
    """
    mock_llm = MagicMock()
    mock_llm.stream.return_value = iter(
        [
            SimpleNamespace(content=""),
            SimpleNamespace(
                content=[
                    {
                        "type": "thinking",
                        "thinking": "reasoning about the answer...",
                        "signature": "abc123",
                        "index": 0,
                    }
                ]
            ),
            SimpleNamespace(content="The answer "),
            SimpleNamespace(content="is [1]."),
            SimpleNamespace(content=""),
        ]
    )
    mock_get_llm_streaming.return_value = mock_llm

    chunk = ChunkData(path="app/db.py", start_line=1, end_line=10, content="engine = ...", score=0.9)
    result = generate_answer(_make_state([chunk]))

    assert result["answer"] == "The answer is [1]."
    assert len(result["citations"]) == 1
