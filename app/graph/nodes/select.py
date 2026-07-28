"""Claude relevance selection.

This is the node that decides which retrieved chunks become evidence. Retrieval
casts a wide net on purpose — three signals, over-fetched and fused — and this
node narrows it.

Two properties are load-bearing:

1. The response is obtained through `with_structured_output`, so it is validated
   by the tool-use schema rather than scraped out of a code fence.
2. Every degradation announces itself. The predecessor kept all eight chunks on
   any exception and reported nothing, which made a failed judgment
   indistinguishable from a confident one.
"""

from __future__ import annotations

import logging

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field, field_validator

from app.graph.state import GradedChunk, RagState, effective_query
from app.llm import get_llm

logger = logging.getLogger(__name__)

MAX_KEPT = 8
CANDIDATE_BODY_LINES = 60
MERGE_GAP_LINES = 2
FALLBACK_REASON = "kept by retrieval score (LLM grading unavailable)"


class SelectionItem(BaseModel):
    index: int = Field(description="The candidate number, exactly as shown in brackets.")
    keep: bool = Field(description="True only if this chunk helps answer the question.")
    relevance: float = Field(
        default=0.0, description="Confidence from 0 to 1 that this chunk answers the question."
    )
    reason: str = Field(default="", description="One short line explaining the decision.")

    @field_validator("relevance")
    @classmethod
    def _clamp(cls, v: float) -> float:
        return max(0.0, min(1.0, float(v)))


class SelectionResult(BaseModel):
    items: list[SelectionItem] = Field(
        default_factory=list, description="One entry per candidate you were shown."
    )


SYSTEM_PROMPT = (
    "You select which code chunks are relevant to a question about a codebase.\n\n"
    "Return one entry per candidate, using the exact bracketed number shown.\n"
    "Set keep=true only when the chunk contains information that helps answer the "
    "question — a definition, the configuration being asked about, the logic in "
    "question, or a call site that shows how it is used.\n\n"
    "Be strict. A chunk that merely mentions a matching word is not relevant. "
    "Prefer a small set of chunks that genuinely answer the question over a broad "
    "set that might. If nothing is relevant, keep nothing: an honest empty result "
    "is more useful than a plausible wrong one.\n\n"
    "relevance is your confidence from 0 to 1. reason is one short line."
)


def _render_candidate(number: int, chunk) -> str:
    lines = chunk.content.split("\n")
    if len(lines) > CANDIDATE_BODY_LINES:
        body = "\n".join(lines[:CANDIDATE_BODY_LINES]) + "\n... [truncated]"
    else:
        body = chunk.content

    header = f"[{number}] {chunk.path}:{chunk.start_line}-{chunk.end_line}"
    if chunk.symbol:
        header += f"  symbol={chunk.symbol}"
    if chunk.language:
        header += f"  language={chunk.language}"

    return f"{header}\n```\n{body}\n```"


def _build_messages(question: str, retrieved: list) -> list:
    candidates = "\n\n".join(
        _render_candidate(i, chunk) for i, chunk in enumerate(retrieved, start=1)
    )
    user = (
        f"Question: {question}\n\n"
        f"Candidates:\n\n{candidates}\n\n"
        f"Judge candidates 1 to {len(retrieved)}."
    )
    return [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=user)]


def _fallback(retrieved: list) -> list[GradedChunk]:
    """Keep the best candidates by retrieval score, labelled as ungraded.

    Retrieval already ranked these, so they are a reasonable answer basis. What
    matters is that the UI and the answer prompt can tell this apart from a real
    selection.
    """
    graded: list[GradedChunk] = []
    for position, chunk in enumerate(retrieved):
        keep = position < MAX_KEPT
        graded.append(
            GradedChunk(
                chunk=chunk,
                keep=keep,
                reason=FALLBACK_REASON if keep else "not in the top results",
                relevance=0.0,
            )
        )
    return graded


def _invoke_with_retry(llm, messages) -> SelectionResult | None:
    """One retry, because a schema violation is usually transient."""
    structured = llm.with_structured_output(SelectionResult)

    for attempt in (1, 2):
        try:
            result = structured.invoke(messages)
        except Exception as e:
            logger.warning("Selection attempt %d failed: %s", attempt, e)
            continue
        if isinstance(result, SelectionResult):
            return result
        logger.warning("Selection attempt %d returned %r", attempt, type(result))

    return None


def merge_adjacent(kept: list[GradedChunk]) -> list[GradedChunk]:
    """Join contiguous or overlapping kept chunks from the same file.

    AST chunking splits oversized functions, so a single answer can depend on two
    adjacent chunks. Merging them means the model sees the code whole and the
    citation points at one continuous range instead of two fragments.
    """
    if len(kept) < 2:
        return list(kept)

    from dataclasses import replace

    ordered = sorted(kept, key=lambda g: (g.chunk.path, g.chunk.start_line))
    merged: list[GradedChunk] = []

    for graded in ordered:
        if not merged:
            merged.append(graded)
            continue

        previous = merged[-1]
        same_file = previous.chunk.path == graded.chunk.path
        contiguous = graded.chunk.start_line <= previous.chunk.end_line + MERGE_GAP_LINES + 1

        if not (same_file and contiguous):
            merged.append(graded)
            continue

        if graded.chunk.end_line <= previous.chunk.end_line:
            # Fully contained; keep the wider range but take the better score.
            merged[-1] = replace(
                previous, relevance=max(previous.relevance, graded.relevance)
            )
            continue

        overlap = max(0, previous.chunk.end_line - graded.chunk.start_line + 1)
        tail_lines = graded.chunk.content.split("\n")[overlap:]
        combined_content = previous.chunk.content
        if tail_lines:
            combined_content += "\n" + "\n".join(tail_lines)

        combined_chunk = replace(
            previous.chunk,
            end_line=graded.chunk.end_line,
            content=combined_content,
            score=max(previous.chunk.score, graded.chunk.score),
            sources=tuple(dict.fromkeys(previous.chunk.sources + graded.chunk.sources)),
        )
        best = previous if previous.relevance >= graded.relevance else graded
        merged[-1] = replace(
            best, chunk=combined_chunk, relevance=max(previous.relevance, graded.relevance)
        )

    return merged


def select_chunks(state: RagState) -> dict:
    """Grade retrieved candidates and keep only the relevant ones."""
    retrieved = state.get("retrieved", [])
    if not retrieved:
        return {"graded": []}

    question = effective_query(state)
    messages = _build_messages(question, retrieved)

    try:
        llm = get_llm()
        result = _invoke_with_retry(llm, messages)
    except Exception as e:
        # A missing API key or an unusable client must not surface as a 500.
        logger.error("Selection unavailable: %s", e)
        result = None

    if result is None:
        logger.warning("Falling back to retrieval order for %d candidates", len(retrieved))
        return {"graded": _fallback(retrieved)}

    decisions: dict[int, SelectionItem] = {}
    for item in result.items:
        if not 1 <= item.index <= len(retrieved):
            logger.warning("Discarding out-of-range candidate index %s", item.index)
            continue
        # First decision wins, so a repeated index cannot duplicate a chunk.
        decisions.setdefault(item.index, item)

    kept: list[GradedChunk] = []
    dropped: list[GradedChunk] = []

    for position, chunk in enumerate(retrieved, start=1):
        decision = decisions.get(position)
        if decision is None:
            # Silence is not endorsement.
            dropped.append(
                GradedChunk(chunk=chunk, keep=False, reason="not selected", relevance=0.0)
            )
            continue

        graded = GradedChunk(
            chunk=chunk,
            keep=bool(decision.keep),
            reason=decision.reason or ("relevant" if decision.keep else "not relevant"),
            relevance=decision.relevance,
        )
        (kept if graded.keep else dropped).append(graded)

    kept.sort(key=lambda g: (g.relevance, g.chunk.score), reverse=True)

    if len(kept) > MAX_KEPT:
        for extra in kept[MAX_KEPT:]:
            extra.keep = False
            extra.reason = f"{extra.reason} (beyond the top {MAX_KEPT})"
        dropped.extend(kept[MAX_KEPT:])
        kept = kept[:MAX_KEPT]

    kept = merge_adjacent(kept)
    kept.sort(key=lambda g: (g.relevance, g.chunk.score), reverse=True)

    logger.info("Selected %d of %d candidates", len(kept), len(retrieved))
    return {"graded": kept + dropped}
