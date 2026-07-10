import os
from pathlib import Path

import streamlit as st

from app.ingest.pipeline import ingest_repo
from app.ingest.walker import DEFAULT_ALLOWLIST


def tab_ingest():
    mode_badge = '<span style="background:#f0f2f6;padding:2px 10px;border-radius:10px;font-size:0.8em">Full-text search</span>'
    st.markdown(f"**Retrieval mode**: {mode_badge}", unsafe_allow_html=True)
    st.divider()

    mode = st.radio("Source type", ["Local Folder", "GitHub URL"], horizontal=True)

    if mode == "GitHub URL":
        st.info("GitHub ingestion coming soon — use Local Folder for now.")
        return

    col1, col2 = st.columns(2)
    with col1:
        path_value = st.text_input("Repo path", value="./repos/my_repo", key="repo_path")
    with col2:
        auto_name = os.path.basename(os.path.normpath(path_value))
        name_value = st.text_input("Repo name", value=auto_name, key="repo_name")

    with st.expander("Extension allowlist"):
        current_exts = "\n".join(sorted(DEFAULT_ALLOWLIST))
        st.text_area("Extensions (one per line)", value=current_exts, height=150, disabled=True)
        st.caption("Changes take effect on next ingest. Edit DEFAULT_ALLOWLIST in app/ingest/walker.py.")

    if st.button("Ingest", type="primary"):
        root = Path(path_value)
        if not root.exists() or not root.is_dir():
            st.error(f"Directory not found: {path_value}")
            return

        progress_bar = st.progress(0, text="Starting ingest...")
        status_text = st.empty()

        files_processed = [0]

        def progress(current, chunk_added, filename):
            files_processed[0] = current
            progress_bar.progress(min(current / max(files_processed[0], 1), 1.0))
            status_text.text(f"Processed {current} files, last: {filename}")

        try:
            repo = ingest_repo(
                repo_name=name_value,
                root_dir=str(root),
                progress_callback=progress,
            )
            progress_bar.progress(1.0)
            status_text.empty()
            st.success(
                f"Ingested **{repo.name}**\n\n"
                f"- **Files**: {repo.file_count}\n"
                f"- **Chunks**: {repo.chunk_count}\n"
                f"- **Source**: Local folder"
            )
        except Exception as e:
            progress_bar.empty()
            status_text.empty()
            st.error(f"Ingest failed: {e}")
