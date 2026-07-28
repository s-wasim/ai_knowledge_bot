"""Reciprocal Rank Fusion.

Cosine similarity, ts_rank_cd, and trigram similarity are not comparable
quantities, so the retrievers are combined by rank rather than by score. RRF needs
no calibration between them and rewards agreement: a chunk found by two
retrievers outranks one found only by the best-scoring retriever.
"""

from app.retrieval.base import ChunkData
from app.retrieval.fusion import RRF_K, reciprocal_rank_fusion


def chunk(path, start=1, score=0.0):
    return ChunkData(
        path=path,
        start_line=start,
        end_line=start + 9,
        content=f"body of {path}",
        score=score,
    )


class TestRanking:
    def test_agreement_outranks_a_single_top_hit(self):
        a, b = chunk("a.py"), chunk("b.py")
        fused = reciprocal_rank_fusion({"dense": [b, a], "text": [a]})
        assert fused[0].path == "a.py"

    def test_records_which_retrievers_contributed(self):
        a, b = chunk("a.py"), chunk("b.py")
        fused = reciprocal_rank_fusion({"dense": [b, a], "text": [a]})
        by_path = {c.path: c for c in fused}
        assert set(by_path["a.py"].sources) == {"dense", "text"}
        assert set(by_path["b.py"].sources) == {"dense"}

    def test_scores_are_rrf_scores_in_descending_order(self):
        a, b = chunk("a.py"), chunk("b.py")
        fused = reciprocal_rank_fusion({"dense": [a, b]})
        assert fused[0].score == 1 / (RRF_K + 1)
        assert fused[1].score == 1 / (RRF_K + 2)
        assert [c.score for c in fused] == sorted((c.score for c in fused), reverse=True)

    def test_single_retriever_preserves_its_order(self):
        items = [chunk(f"{i}.py") for i in range(5)]
        fused = reciprocal_rank_fusion({"dense": items})
        assert [c.path for c in fused] == [c.path for c in items]


class TestDeduplication:
    def test_dedups_on_path_and_start_line(self):
        fused = reciprocal_rank_fusion(
            {"dense": [chunk("a.py", 1)], "text": [chunk("a.py", 1)]}
        )
        assert len(fused) == 1

    def test_same_file_different_lines_are_distinct(self):
        fused = reciprocal_rank_fusion({"dense": [chunk("a.py", 1), chunk("a.py", 40)]})
        assert len(fused) == 2

    def test_keeps_the_richer_duplicate(self):
        """Retrievers select different columns; the surviving copy must not lose
        the symbol a sibling retriever supplied."""
        bare = chunk("a.py", 1)
        rich = ChunkData("a.py", 1, 10, "body", 0.5, symbol="get_session", language="python")
        fused = reciprocal_rank_fusion({"dense": [bare], "text": [rich]})
        assert fused[0].symbol == "get_session"


class TestLimits:
    def test_applies_the_candidate_cap(self):
        items = [chunk(f"{i}.py") for i in range(50)]
        assert len(reciprocal_rank_fusion({"dense": items}, limit=24)) == 24

    def test_empty_input_yields_nothing(self):
        assert reciprocal_rank_fusion({}) == []

    def test_empty_lists_are_ignored(self):
        assert reciprocal_rank_fusion({"dense": [], "text": []}) == []

    def test_a_failed_retriever_contributing_none_is_tolerated(self):
        """hybrid.py passes through whatever each retriever returned; a None from
        a failed retriever must not break fusion."""
        fused = reciprocal_rank_fusion({"dense": None, "text": [chunk("a.py")]})
        assert [c.path for c in fused] == ["a.py"]
