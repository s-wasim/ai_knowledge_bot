"""Hybrid retrieval and mode reporting.

The contract that matters: one broken retriever degrades the result, it never
fails the request. And the reported mode must tell the truth about which signals
are live, because the previous implementation's silent downgrade is what made
retrieval quality impossible to reason about.
"""

from unittest.mock import MagicMock, patch

from app.retrieval.base import ChunkData
from app.retrieval.factory import MODE_DEGRADED, MODE_HYBRID, create_retriever, get_mode_display
from app.retrieval.hybrid import HybridRetriever


def chunk(path, start=1, score=0.5):
    return ChunkData(path, start, start + 9, f"body {path}", score)


def _retriever_with(dense=None, lexical=None, symbolic=None):
    r = HybridRetriever(MagicMock())
    r._dense = MagicMock(name="dense", search=MagicMock(return_value=dense or []))
    r._dense.name = "dense"
    r._lexical = MagicMock(name="text", search=MagicMock(return_value=lexical or []))
    r._lexical.name = "text"
    r._symbolic = MagicMock(name="symbol", search=MagicMock(return_value=symbolic or []))
    r._symbolic.name = "symbol"
    return r


class TestFanOut:
    def test_queries_all_three_retrievers(self):
        r = _retriever_with()
        r.search(1, "where is the db")
        r._dense.search.assert_called_once()
        r._lexical.search.assert_called_once()
        r._symbolic.search.assert_called_once()

    def test_fuses_results_from_every_retriever(self):
        r = _retriever_with(dense=[chunk("a.py")], lexical=[chunk("b.py")], symbolic=[chunk("c.py")])
        paths = {c.path for c in r.search(1, "q")}
        assert paths == {"a.py", "b.py", "c.py"}

    def test_marks_provenance(self):
        shared = chunk("a.py")
        r = _retriever_with(dense=[shared], lexical=[chunk("a.py")])
        result = r.search(1, "q")[0]
        assert set(result.sources) == {"dense", "text"}

    def test_respects_the_candidate_cap(self):
        many = [chunk(f"{i}.py") for i in range(40)]
        r = _retriever_with(dense=many)
        assert len(r.search(1, "q", k=24)) == 24


class TestDegradation:
    def test_one_retriever_raising_does_not_fail_the_search(self):
        r = _retriever_with(lexical=[chunk("b.py")])
        r._dense.search.side_effect = RuntimeError("model gone")
        results = r.search(1, "q")
        assert [c.path for c in results] == ["b.py"]

    def test_all_retrievers_raising_yields_an_empty_list(self):
        r = _retriever_with()
        for sub in (r._dense, r._lexical, r._symbolic):
            sub.search.side_effect = RuntimeError("down")
        assert r.search(1, "q") == []

    def test_no_results_yields_an_empty_list(self):
        assert _retriever_with().search(1, "q") == []


class TestModeReporting:
    @patch("app.retrieval.factory.is_embedding_available", return_value=True)
    def test_hybrid_when_embeddings_are_available(self, _):
        retriever, mode = create_retriever(MagicMock())
        assert isinstance(retriever, HybridRetriever)
        assert mode == MODE_HYBRID
        assert "embeddings" in get_mode_display(mode)

    @patch("app.retrieval.factory.is_embedding_available", return_value=False)
    def test_degraded_when_embeddings_are_missing(self, _):
        _, mode = create_retriever(MagicMock())
        assert mode == MODE_DEGRADED
        assert "no embeddings" in get_mode_display(mode)
