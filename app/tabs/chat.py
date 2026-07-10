import streamlit as st

from app.db import get_session, Repo
from app.graph.build import build_rag_graph, set_retriever
from app.graph.state import RagState
from app.retrieval.base import ChunkData
from app.retrieval.factory import create_retriever, get_mode_display


def init_session():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "graph" not in st.session_state:
        st.session_state.graph = build_rag_graph()
    if "retriever_initialized" not in st.session_state:
        st.session_state.retriever_initialized = False


def tab_chat():
    init_session()

    st.markdown("### 💬 Chat with your codebase")

    # Mode badge
    retriever, mode = create_retriever(get_session)
    mode_badge = f'<span style="background:#f0f2f6;padding:2px 10px;border-radius:10px;font-size:0.8em">{get_mode_display(mode)}</span>'
    st.markdown(f"**Retrieval mode**: {mode_badge}", unsafe_allow_html=True)

    # Repo selector
    session = get_session()
    repos = session.query(Repo).order_by(Repo.ingested_at.desc()).all()
    session.close()

    if not repos:
        st.info("No repos ingested yet. Go to the **📦 Ingest** tab to ingest a codebase first.")
        return

    repo_options = {f"{r.name} ({r.file_count} files, {r.chunk_count} chunks)": r.id for r in repos}
    selected_label = st.selectbox("Select repo", options=list(repo_options.keys()))
    repo_id = repo_options[selected_label]

    # Initialize retriever for this repo
    if not st.session_state.retriever_initialized:
        retriever, mode = create_retriever(get_session)
        set_retriever(retriever)
        st.session_state.retriever_initialized = True

    # Display chat messages
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and "citations" in msg:
                _show_citations(msg)
            if msg["role"] == "assistant" and "graded" in msg:
                _show_graded_chunks(msg)

    # Quick-ask buttons in sidebar
    with st.sidebar:
        st.markdown("### Quick Questions")

        quick_questions = [
            "Where is the database connection configured?",
            "How does authentication work?",
            "How do I change it to use MySQL instead?",
            "How does the notification system work?",
            "What environment variables does the application use?",
        ]

        for q in quick_questions:
            if st.button(q, key=f"quick_{hash(q)}", use_container_width=True):
                st.session_state._quick_question = q
                st.rerun()

        if st.button("Clear chat history", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

    # Check for pending quick question
    quick_question = st.session_state.pop("_quick_question", None)

    if prompt := quick_question or st.chat_input("Ask a question about the codebase..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Build history
        history = []
        for m in st.session_state.messages[-6:]:
            if m["role"] in ("user", "assistant"):
                history.append({"role": m["role"], "content": m["content"]})

        # Build state
        state = RagState(
            question=prompt,
            chat_history=history,
            rewritten_query=None,
            retrieved=[],
            graded=[],
            answer=None,
            citations=[],
            mode="fts",
            repo_id=repo_id,
        )

        # Run graph
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    result = st.session_state.graph.invoke(state)
                    answer = result.get("answer", "No answer generated.")
                    citations = result.get("citations", [])
                    graded = result.get("graded", [])
                    retrieved = result.get("retrieved", [])

                    st.markdown(answer)

                    msg_data = {
                        "role": "assistant",
                        "content": answer,
                        "citations": [(c.chunk.path, c.chunk.start_line, c.chunk.end_line, c.chunk.content) for c in citations],
                        "graded": [(gc.chunk.path, gc.chunk.start_line, gc.chunk.end_line, gc.chunk.content, gc.keep, gc.reason, gc.chunk.score) for gc in graded],
                        "retrieved": [(c.path, c.start_line, c.end_line, c.content, c.score) for c in retrieved],
                    }
                    st.session_state.messages.append(msg_data)
                    _show_citations(msg_data)
                    _show_graded_chunks(msg_data)

                except Exception as e:
                    st.error(f"Error: {e}")
                    st.session_state.messages.append({"role": "assistant", "content": f"Sorry, an error occurred: {e}"})

        st.rerun()


def _show_citations(msg):
    """Display citation buttons for an assistant message."""
    citations = msg.get("citations", [])
    if citations:
        st.markdown("**Sources:**")
        for i, (path, start, end, content) in enumerate(citations, 1):
            with st.expander(f"[{i}] {path}:{start}-{end}"):
                st.code(content, line_numbers=True, language="python")


def _show_graded_chunks(msg):
    """Display the graded chunks expander."""
    graded = msg.get("graded", [])
    retrieved = msg.get("retrieved", [])
    if not graded:
        graded = [(c[0], c[1], c[2], c[3], True, "Retrieved", c[4]) for c in retrieved]
    if graded:
        with st.expander("Retrieved chunks & grades"):
            for path, start_line, end_line, content, keep, reason, score in graded:
                icon = "✓" if keep else "✗"
                st.markdown(f"{icon} **{path}:{start_line}-{end_line}** (score: {score:.3f})")
                st.caption(f"Grade: {reason}")
                with st.expander("View source"):
                    st.code(content, line_numbers=True)
