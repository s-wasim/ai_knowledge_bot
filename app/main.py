import streamlit as st
from app.db import init_db

# Initialize database on startup
try:
    init_db()
except Exception as e:
    st.warning(f"Database not available yet: {e}")

from app.tabs.ingest import tab_ingest
from app.tabs.chat import tab_chat
from app.tabs.browser import tab_browser

st.set_page_config(page_title="AI Knowledge Bot", layout="wide")

tab1, tab2, tab3 = st.tabs(["📦 Ingest", "💬 Chat", "🗂 Index Browser"])

with tab1:
    tab_ingest()

with tab2:
    tab_chat()

with tab3:
    tab_browser()
