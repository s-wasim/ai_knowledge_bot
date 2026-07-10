from app.graph.state import RagState, GradedChunk, Citation
from app.retrieval.base import ChunkData
from app.graph.nodes.answer import _postprocess_answer


def test_valid_citation_preserved():
    kept = [
        GradedChunk(chunk=ChunkData(path="db.py", start_line=1, end_line=10, content="x", score=0.9), keep=True, reason="r"),
        GradedChunk(chunk=ChunkData(path="config.py", start_line=5, end_line=15, content="y", score=0.8), keep=True, reason="r"),
    ]
    answer = "The DB is in [1] and config is in [2]."
    cleaned, citations = _postprocess_answer(answer, kept)
    assert "[1]" in cleaned
    assert "[2]" in cleaned
    assert len(citations) == 2


def test_invalid_citation_stripped():
    kept = [
        GradedChunk(chunk=ChunkData(path="db.py", start_line=1, end_line=10, content="x", score=0.9), keep=True, reason="r"),
    ]
    answer = "The DB is in [1] and also see [5]."
    cleaned, citations = _postprocess_answer(answer, kept)
    assert "[1]" in cleaned
    assert "[5]" not in cleaned
    assert len(citations) == 1


def test_no_citations():
    kept = [GradedChunk(chunk=ChunkData(path="db.py", start_line=1, end_line=10, content="x", score=0.9), keep=True, reason="r")]
    answer = "I don't know."
    cleaned, citations = _postprocess_answer(answer, kept)
    assert cleaned == answer
    assert citations == []


def test_not_found_message():
    from app.graph.nodes.not_found import answer_not_found
    state = RagState(
        question="nonexistent feature", chat_history=[], rewritten_query="nonexistent feature",
        retrieved=[ChunkData(path="auth/login.py", start_line=1, end_line=5, content="x", score=0.5)],
        graded=[], answer=None, citations=[], mode="fts", repo_id=1,
    )
    result = answer_not_found(state)
    assert "couldn't find" in result["answer"]
    assert result["citations"] == []


def test_generate_answer_no_kept_chunks():
    from app.graph.nodes.answer import generate_answer
    state = RagState(
        question="test", chat_history=[], rewritten_query="test",
        retrieved=[], graded=[], answer=None, citations=[], mode="fts", repo_id=1,
    )
    result = generate_answer(state)
    assert "couldn't find" in result["answer"]
