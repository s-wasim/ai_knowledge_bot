import os
from pathlib import Path

import streamlit as st

from app.ingest.embedder import is_voyage_available
from app.ingest.pipeline import ingest_repo
from app.ingest.walker import DEFAULT_ALLOWLIST
from app.retrieval.factory import get_mode_display


def tab_ingest():
    mode = "vector" if is_voyage_available() else "fts"
    mode_badge = f'<span style="background:#f0f2f6;padding:2px 10px;border-radius:10px;font-size:0.8em">{get_mode_display(mode)}</span>'
    st.markdown(f"**Retrieval mode**: {mode_badge}", unsafe_allow_html=True)
    st.divider()

    source_type = st.radio("Source type", ["Local Folder", "GitHub URL"], horizontal=True)

    if source_type == "GitHub URL":
        col1, col2 = st.columns([3, 1])
        with col1:
            url_value = st.text_input(
                "GitHub URL",
                value="https://github.com/owner/repo",
                key="github_url",
            )
        with col2:
            branch_value = st.text_input("Branch (optional)", value="", key="github_branch")

        if st.button("Ingest from GitHub", type="primary"):
            if not url_value or url_value == "https://github.com/owner/repo":
                st.error("Please enter a valid GitHub URL.")
                return

            status_text = st.empty()

            def progress(current, chunk_added, filename):
                status_text.text(f"Processed {current} files, last: {filename}")

            try:
                from app.ingest.github import ingest_github_url

                repo = ingest_github_url(
                    url=url_value,
                    branch=branch_value if branch_value else None,
                    progress_callback=progress,
                )
                status_text.empty()
                st.success(
                    f"Ingested **{repo.name}** from GitHub\n\n"
                    f"- **Files**: {repo.file_count}\n"
                    f"- **Chunks**: {repo.chunk_count}\n"
                    f"- **Source**: {repo.source_url}"
                )
            except Exception as e:
                status_text.empty()
                st.error(f"GitHub ingest failed: {e}")

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

        status_text = st.empty()

        def progress(current, chunk_added, filename):
            status_text.text(f"Processed {current} files, last: {filename}")

        try:
            repo = ingest_repo(
                repo_name=name_value,
                root_dir=str(root),
                progress_callback=progress,
            )
            status_text.empty()
            st.success(
                f"Ingested **{repo.name}**\n\n"
                f"- **Files**: {repo.file_count}\n"
                f"- **Chunks**: {repo.chunk_count}\n"
                f"- **Source**: Local folder"
            )
        except Exception as e:
            status_text.empty()
            st.error(f"Ingest failed: {e}")
