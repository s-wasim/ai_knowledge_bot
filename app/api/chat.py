# app/api/chat.py
import json
import logging

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from app.api.deps import get_graph, get_retriever_and_mode
from app.api.schemas import ChatRequest
from app.db import get_session
from app.graph.state import RagState
from app.llm import extract_text

router = APIRouter()

logger = logging.getLogger(__name__)


def _sse_frame(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


def _citation_dict(citation) -> dict:
    return {
        # The number as written in the answer text. Without it the UI numbered
        # chips by array position, so an answer citing [2] and [5] rendered chips
        # labelled [1] and [2].
        "index": citation.index,
        "path": citation.chunk.path,
        "start_line": citation.chunk.start_line,
        "end_line": citation.chunk.end_line,
        "content": citation.chunk.content,
        "symbol": citation.chunk.symbol,
        "language": citation.chunk.language,
    }


def _graded_dict(graded_chunk) -> dict:
    return {
        "path": graded_chunk.chunk.path,
        "start_line": graded_chunk.chunk.start_line,
        "end_line": graded_chunk.chunk.end_line,
        "content": graded_chunk.chunk.content,
        "keep": graded_chunk.keep,
        "reason": graded_chunk.reason,
        "score": graded_chunk.chunk.score,
        "relevance": graded_chunk.relevance,
        "symbol": graded_chunk.chunk.symbol,
        "language": graded_chunk.chunk.language,
        "sources": list(graded_chunk.chunk.sources),
    }


def _retrieved_dict(chunk) -> dict:
    return {
        "path": chunk.path,
        "start_line": chunk.start_line,
        "end_line": chunk.end_line,
        "content": chunk.content,
        "score": chunk.score,
        "symbol": chunk.symbol,
        "language": chunk.language,
        "sources": list(chunk.sources),
    }


@router.post("/chat")
def chat(body: ChatRequest) -> StreamingResponse:
    retriever, mode = get_retriever_and_mode()
    graph = get_graph()

    history = [m.model_dump() for m in body.history[-6:]]

    state: RagState = {
        "question": body.question,
        "chat_history": history,
        "rewritten_query": None,
        "retrieved": [],
        "graded": [],
        "answer": None,
        "citations": [],
        "mode": mode,
        "repo_id": body.repo_id,
    }
    config = {"configurable": {"retriever": retriever, "get_session": get_session}}

    def generate():
        final_state: dict = {}
        try:
            for stream_mode, payload in graph.stream(
                state, config=config, stream_mode=["updates", "messages"]
            ):
                if stream_mode == "updates":
                    for node_name, node_output in payload.items():
                        yield _sse_frame("node", {"node": node_name})
                        final_state.update(node_output)
                elif stream_mode == "messages":
                    message_chunk, metadata = payload
                    if metadata.get("langgraph_node") == "generate_answer":
                        text = extract_text(message_chunk.content)
                        if text:
                            yield _sse_frame("token", {"text": text})

            answer = final_state.get("answer") or "No answer generated."
            citations = final_state.get("citations", [])
            graded = final_state.get("graded", [])
            retrieved = final_state.get("retrieved", [])

            yield _sse_frame(
                "final",
                {
                    "answer": answer,
                    "citations": [_citation_dict(c) for c in citations],
                    "graded": [_graded_dict(gc) for gc in graded],
                    "retrieved": [_retrieved_dict(c) for c in retrieved],
                },
            )
        except Exception as e:
            # The graph guards each node, so reaching here means something outside
            # them failed. Reported as an SSE frame the UI renders inside the
            # assistant card, never as a mid-stream HTTP error.
            logger.exception("Chat stream failed for repo %s", body.repo_id)
            yield _sse_frame("error", {"message": str(e) or e.__class__.__name__})

    return StreamingResponse(generate(), media_type="text/event-stream")
