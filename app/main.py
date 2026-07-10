import streamlit as st

st.set_page_config(page_title="AI Knowledge Bot", layout="wide")

_db_error = None
try:
    from app.db import init_db
    init_db()
except Exception as e:
    _db_error = e

from app.tabs.ingest import tab_ingest
from app.tabs.chat import tab_chat
from app.tabs.browser import tab_browser

tab1, tab2, tab3 = st.tabs(["📦 Ingest", "💬 Chat", "🗂 Index Browser"])

if _db_error:
    st.warning(f"Database not available yet: {_db_error}")

with tab1:
    tab_ingest()

with tab2:
    tab_chat()

with tab3:
    tab_browser()
