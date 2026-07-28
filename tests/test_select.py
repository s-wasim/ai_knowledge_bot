"""Claude relevance selection.

This node decides which retrieved chunks become evidence, so its failure modes
matter more than its happy path. The predecessor scraped JSON out of a code fence
and, on any exception, kept every chunk while reporting nothing — turning a failed
judgment into a confident-looking one. Here the output is tool-use-validated, and
every degradation is labelled.
"""

from unittest.mock import MagicMock, patch

import pytest

from app.graph.nodes.select import (
    MAX_KEPT,
    SelectionItem,
    SelectionResult,
    merge_adjacent,
    select_chunks,
)
from app.graph.state import GradedChunk
from app.retrieval.base import ChunkData


def chunk(path="app/db.py", start=1, end=10, score=0.5, content=None, symbol=None):
    return ChunkData(
        path=path,
        start_line=start,
        end_line=end,
        content=content if content is not None else f"code from {path}",
        score=score,
        symbol=symbol,
    )


def state(retrieved, question="where is the database configured"):
    return {
        "question": question,
        "chat_history": [],
        "rewritten_query": question,
        "retrieved": retrieved,
        "graded": [],
        "answer": None,
        "citations": [],
        "mode": "hybrid",
        "repo_id": 1,
    }


def _llm_returning(result):
    llm = MagicMock()
    structured = MagicMock()
    structured.invoke.return_value = result
    llm.with_structured_output.return_value = structured
    return llm, structured


class TestSelection:
    @patch("app.graph.nodes.select.get_llm")
    def test_keeps_only_the_chunks_claude_flags(self, mock_get_llm):
        retrieved = [chunk("a.py", 1), chunk("b.py", 1)]
        llm, _ = _llm_returning(
            SelectionResult(
                items=[
                    SelectionItem(index=1, keep=True, relevance=0.9, reason="defines it"),
                    SelectionItem(index=2, keep=False, relevance=0.1, reason="unrelated"),
                ]
            )
        )
        mock_get_llm.return_value = llm

        graded = select_chunks(state(retrieved))["graded"]

        kept = [g for g in graded if g.keep]
        assert len(kept) == 1
        assert kept[0].chunk.path == "a.py"
        assert kept[0].reason == "defines it"
        assert kept[0].relevance == 0.9

    @patch("app.graph.nodes.select.get_llm")
    def test_reports_dropped_chunks_too(self, mock_get_llm):
        """The UI shows what was considered and rejected, so dropped chunks stay
        in the graded list rather than vanishing."""
        retrieved = [chunk("a.py"), chunk("b.py", 20)]
        llm, _ = _llm_returning(
            SelectionResult(
                items=[
                    SelectionItem(index=1, keep=True, relevance=0.8, reason="yes"),
                    SelectionItem(index=2, keep=False, relevance=0.0, reason="no"),
                ]
            )
        )
        mock_get_llm.return_value = llm

        graded = select_chunks(state(retrieved))["graded"]
        assert len(graded) == 2
        assert [g.keep for g in graded].count(False) == 1

    @patch("app.graph.nodes.select.get_llm")
    def test_unmentioned_candidates_default_to_dropped(self, mock_get_llm):
        """Silence is not endorsement. Defaulting to keep is how the old grader
        turned a partial response into eight irrelevant citations."""
        retrieved = [chunk("a.py"), chunk("b.py", 20), chunk("c.py", 40)]
        llm, _ = _llm_returning(
            SelectionResult(items=[SelectionItem(index=1, keep=True, relevance=0.9, reason="yes")])
        )
        mock_get_llm.return_value = llm

        graded = select_chunks(state(retrieved))["graded"]
        assert [g.keep for g in graded] == [True, False, False]

    @patch("app.graph.nodes.select.get_llm")
    def test_out_of_range_indices_are_discarded(self, mock_get_llm):
        retrieved = [chunk("a.py")]
        llm, _ = _llm_returning(
            SelectionResult(
                items=[
                    SelectionItem(index=99, keep=True, relevance=1.0, reason="hallucinated"),
                    SelectionItem(index=0, keep=True, relevance=1.0, reason="off by one"),
                    SelectionItem(index=-1, keep=True, relevance=1.0, reason="negative"),
                ]
            )
        )
        mock_get_llm.return_value = llm

        graded = select_chunks(state(retrieved))["graded"]
        assert all(not g.keep for g in graded)

    @patch("app.graph.nodes.select.get_llm")
    def test_duplicate_indices_do_not_duplicate_chunks(self, mock_get_llm):
        retrieved = [chunk("a.py")]
        llm, _ = _llm_returning(
            SelectionResult(
                items=[
                    SelectionItem(index=1, keep=True, relevance=0.9, reason="first"),
                    SelectionItem(index=1, keep=True, relevance=0.5, reason="again"),
                ]
            )
        )
        mock_get_llm.return_value = llm

        graded = select_chunks(state(retrieved))["graded"]
        assert len(graded) == 1

    @patch("app.graph.nodes.select.get_llm")
    def test_kept_chunks_are_ordered_by_relevance(self, mock_get_llm):
        retrieved = [chunk("a.py"), chunk("b.py", 20), chunk("c.py", 40)]
        llm, _ = _llm_returning(
            SelectionResult(
                items=[
                    SelectionItem(index=1, keep=True, relevance=0.3, reason="weak"),
                    SelectionItem(index=2, keep=True, relevance=0.9, reason="strong"),
                    SelectionItem(index=3, keep=True, relevance=0.6, reason="medium"),
                ]
            )
        )
        mock_get_llm.return_value = llm

        kept = [g for g in select_chunks(state(retrieved))["graded"] if g.keep]
        assert [g.chunk.path for g in kept] == ["b.py", "c.py", "a.py"]

    @patch("app.graph.nodes.select.get_llm")
    def test_kept_chunks_are_capped(self, mock_get_llm):
        retrieved = [chunk(f"{i}.py") for i in range(20)]
        llm, _ = _llm_returning(
            SelectionResult(
                items=[
                    SelectionItem(index=i + 1, keep=True, relevance=1.0, reason="yes")
                    for i in range(20)
                ]
            )
        )
        mock_get_llm.return_value = llm

        kept = [g for g in select_chunks(state(retrieved))["graded"] if g.keep]
        assert len(kept) == MAX_KEPT


class TestDegradation:
    def test_no_retrieved_chunks_short_circuits(self):
        assert select_chunks(state([]))["graded"] == []

    @patch("app.graph.nodes.select.get_llm")
    def test_retries_once_before_giving_up(self, mock_get_llm):
        retrieved = [chunk("a.py")]
        llm = MagicMock()
        structured = MagicMock()
        structured.invoke.side_effect = [
            ValueError("schema mismatch"),
            SelectionResult(items=[SelectionItem(index=1, keep=True, relevance=0.9, reason="ok")]),
        ]
        llm.with_structured_output.return_value = structured
        mock_get_llm.return_value = llm

        graded = select_chunks(state(retrieved))["graded"]
        assert structured.invoke.call_count == 2
        assert graded[0].keep is True

    @patch("app.graph.nodes.select.get_llm")
    def test_total_failure_falls_back_to_fusion_order(self, mock_get_llm):
        retrieved = [chunk(f"{i}.py", score=1.0 - i / 10) for i in range(12)]
        llm = MagicMock()
        structured = MagicMock()
        structured.invoke.side_effect = RuntimeError("api down")
        llm.with_structured_output.return_value = structured
        mock_get_llm.return_value = llm

        graded = select_chunks(state(retrieved))["graded"]
        kept = [g for g in graded if g.keep]
        assert len(kept) == MAX_KEPT
        assert kept[0].chunk.path == "0.py"

    @patch("app.graph.nodes.select.get_llm")
    def test_fallback_labels_itself_as_ungraded(self, mock_get_llm):
        """A degraded result that looks like a judgment is worse than no result."""
        llm = MagicMock()
        llm.with_structured_output.side_effect = RuntimeError("no tool support")
        mock_get_llm.return_value = llm

        graded = select_chunks(state([chunk("a.py")]))["graded"]
        assert "unavailable" in graded[0].reason.lower()

    @patch("app.graph.nodes.select.get_llm", side_effect=ValueError("no api key"))
    def test_missing_api_key_degrades_instead_of_raising(self, _):
        graded = select_chunks(state([chunk("a.py")]))["graded"]
        assert graded and graded[0].keep is True

    @patch("app.graph.nodes.select.get_llm")
    def test_empty_response_keeps_nothing(self, mock_get_llm):
        """An explicit 'nothing is relevant' must route to the not-found path, not
        silently fall back to keeping everything."""
        llm, _ = _llm_returning(SelectionResult(items=[]))
        mock_get_llm.return_value = llm

        graded = select_chunks(state([chunk("a.py")]))["graded"]
        assert all(not g.keep for g in graded)

    @patch("app.graph.nodes.select.get_llm")
    def test_none_response_falls_back(self, mock_get_llm):
        llm, _ = _llm_returning(None)
        mock_get_llm.return_value = llm

        graded = select_chunks(state([chunk("a.py")]))["graded"]
        assert graded[0].keep is True
        assert "unavailable" in graded[0].reason.lower()


class TestMergeAdjacent:
    def test_joins_contiguous_chunks_from_the_same_file(self):
        a = GradedChunk(chunk("a.py", 1, 10, content="first"), True, "r", 0.9)
        b = GradedChunk(chunk("a.py", 11, 20, content="second"), True, "r", 0.8)
        merged = merge_adjacent([a, b])
        assert len(merged) == 1
        assert merged[0].chunk.start_line == 1
        assert merged[0].chunk.end_line == 20
        assert "first" in merged[0].chunk.content and "second" in merged[0].chunk.content

    def test_keeps_the_higher_relevance_after_merging(self):
        a = GradedChunk(chunk("a.py", 1, 10), True, "weaker", 0.4)
        b = GradedChunk(chunk("a.py", 11, 20), True, "stronger", 0.95)
        assert merge_adjacent([a, b])[0].relevance == 0.95

    def test_does_not_join_across_files(self):
        a = GradedChunk(chunk("a.py", 1, 10), True, "r", 0.9)
        b = GradedChunk(chunk("b.py", 11, 20), True, "r", 0.8)
        assert len(merge_adjacent([a, b])) == 2

    def test_does_not_join_distant_chunks(self):
        a = GradedChunk(chunk("a.py", 1, 10), True, "r", 0.9)
        b = GradedChunk(chunk("a.py", 200, 210), True, "r", 0.8)
        assert len(merge_adjacent([a, b])) == 2

    def test_joins_overlapping_chunks(self):
        a = GradedChunk(chunk("a.py", 1, 20), True, "r", 0.9)
        b = GradedChunk(chunk("a.py", 15, 30), True, "r", 0.8)
        merged = merge_adjacent([a, b])
        assert len(merged) == 1
        assert merged[0].chunk.end_line == 30

    def test_empty_input(self):
        assert merge_adjacent([]) == []

    def test_single_chunk_is_unchanged(self):
        a = GradedChunk(chunk("a.py", 1, 10), True, "r", 0.9)
        assert merge_adjacent([a]) == [a]


class TestPromptConstruction:
    @patch("app.graph.nodes.select.get_llm")
    def test_candidates_are_numbered_from_one(self, mock_get_llm):
        llm, structured = _llm_returning(SelectionResult(items=[]))
        mock_get_llm.return_value = llm

        select_chunks(state([chunk("a.py"), chunk("b.py", 20)]))

        prompt = str(structured.invoke.call_args[0][0])
        assert "[1]" in prompt and "[2]" in prompt

    @patch("app.graph.nodes.select.get_llm")
    def test_candidate_bodies_are_truncated(self, mock_get_llm):
        llm, structured = _llm_returning(SelectionResult(items=[]))
        mock_get_llm.return_value = llm
        huge = "\n".join(f"line {i}" for i in range(500))

        select_chunks(state([chunk("a.py", content=huge)]))

        prompt = str(structured.invoke.call_args[0][0])
        assert "truncated" in prompt
        assert "line 499" not in prompt

    @patch("app.graph.nodes.select.get_llm")
    def test_uses_the_rewritten_query(self, mock_get_llm):
        llm, structured = _llm_returning(SelectionResult(items=[]))
        mock_get_llm.return_value = llm
        s = state([chunk("a.py")], question="original")
        s["rewritten_query"] = "standalone rewritten form"

        select_chunks(s)

        assert "standalone rewritten form" in str(structured.invoke.call_args[0][0])

    @patch("app.graph.nodes.select.get_llm")
    def test_falls_back_to_the_raw_question_when_rewrite_is_missing(self, mock_get_llm):
        llm, structured = _llm_returning(SelectionResult(items=[]))
        mock_get_llm.return_value = llm
        s = state([chunk("a.py")], question="raw question")
        s["rewritten_query"] = None

        select_chunks(s)

        assert "raw question" in str(structured.invoke.call_args[0][0])


class TestSchema:
    def test_relevance_is_clamped_to_the_unit_interval(self):
        assert SelectionItem(index=1, keep=True, relevance=5.0, reason="r").relevance == 1.0
        assert SelectionItem(index=1, keep=True, relevance=-2.0, reason="r").relevance == 0.0

    def test_reason_defaults_to_empty(self):
        assert SelectionItem(index=1, keep=True, relevance=0.5).reason == ""
