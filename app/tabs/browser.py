import streamlit as st
from sqlalchemy import func

from app.db import get_session, Repo, Chunk
from app.highlight import highlight_chunk, highlight_style
from app.ingest.embedder import is_voyage_available
from app.retrieval.factory import get_mode_display


def tab_browser():
    st.markdown("### 🗂 Index Browser")
    st.markdown(highlight_style(), unsafe_allow_html=True)

    mode = "vector" if is_voyage_available() else "fts"
    mode_badge = f'<span style="background:#f0f2f6;padding:2px 10px;border-radius:10px;font-size:0.8em">{get_mode_display(mode)}</span>'
    st.markdown(f"**Retrieval mode**: {mode_badge}", unsafe_allow_html=True)

    session = get_session()

    repos = session.query(Repo).order_by(Repo.ingested_at.desc()).all()
    if not repos:
        st.info("No repos ingested yet.")
        session.close()
        return

    repo_options = {f"{r.name} (ingested {r.ingested_at.strftime('%Y-%m-%d %H:%M') if r.ingested_at else 'N/A'})": r.id for r in repos}
    selected_label = st.selectbox("Select repo", options=list(repo_options.keys()), key="browser_repo")
    repo_id = repo_options[selected_label]

    selected_repo = session.query(Repo).filter_by(id=repo_id).first()

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Files", selected_repo.file_count if selected_repo else 0)
    with col2:
        st.metric("Chunks", selected_repo.chunk_count if selected_repo else 0)
    with col3:
        st.metric("Source", "GitHub" if selected_repo and selected_repo.source_url else "Local")

    keyword = st.text_input("Filter by keyword", placeholder="e.g. database, auth, config...")

    query = session.query(
        Chunk.path,
        func.count(Chunk.id).label("chunk_count"),
        func.min(Chunk.start_line).label("start_line"),
        func.max(Chunk.end_line).label("end_line"),
    ).filter(Chunk.repo_id == repo_id)

    if keyword:
        query = query.filter(Chunk.content.ilike(f"%{keyword}%"))

    query = query.group_by(Chunk.path).order_by(Chunk.path)
    file_stats = query.all()

    if not file_stats:
        st.info("No files match the filter." if keyword else "No files found.")
        session.close()
        return

    st.markdown(f"**{len(file_stats)} files**")

    for row in file_stats:
        with st.expander(f"{row.path} ({row.chunk_count} chunks, lines {row.start_line}-{row.end_line})"):
            chunk_query = session.query(Chunk).filter(
                Chunk.repo_id == repo_id,
                Chunk.path == row.path,
            ).order_by(Chunk.start_line).all()

            for chunk in chunk_query:
                st.markdown(f"**Lines {chunk.start_line}-{chunk.end_line}**")
                st.markdown(
                    highlight_chunk(chunk.path, chunk.content, chunk.start_line),
                    unsafe_allow_html=True,
                )
                st.divider()

    session.close()
