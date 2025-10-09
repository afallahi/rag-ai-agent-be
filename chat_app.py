import streamlit as st
import time

from main.pipeline_core import RAGPipeline
from pipeline import get_llm, get_reranker, generate_response

MAX_HISTORY_LENGTH = 10


# === Page Setup ===
st.set_page_config(page_title="Armstrong AI Assistant")
st.title("Armstrong AI Assistant")
st.caption("Ask technical questions about Armstrong products. I'll pull relevant info and answer.")


# === Component Initialization ===
@st.cache_resource
def load_components():
    llm = get_llm()
    rag_pipeline = RAGPipeline(force_index=False)
    rag_pipeline.refresh_index()
    reranker = get_reranker()
    return llm, rag_pipeline, reranker

llm, rag_pipeline, reranker = load_components()


# === Initialize Chat History ===
if "history" not in st.session_state:
    st.session_state.history = []


# === Sidebar with Reset Button ===
with st.sidebar:
    st.header("Options")
    if st.button("Reset Conversation"):
        st.session_state.history.clear()
        st.rerun()


# === Display Chat History ===
for user_msg, assistant_msg in st.session_state.history:
    with st.chat_message("user"):
        st.markdown(user_msg)
    with st.chat_message("assistant"):
        st.markdown(assistant_msg)


# === Chat Input ===
query = st.chat_input("Ask me something about Armstrong products...")

if query:
    with st.chat_message("user"):
        st.markdown(query)

    with st.spinner("Thinking..."):
        start = time.time()
        response = generate_response(rag_pipeline, query, llm, st.session_state.history[-MAX_HISTORY_LENGTH:], reranker)
        duration = time.time() - start

    st.session_state.history.append((query, response))
    if len(st.session_state.history) > MAX_HISTORY_LENGTH:
        st.session_state.history = st.session_state.history[-MAX_HISTORY_LENGTH:]

    with st.chat_message("assistant"):
        st.markdown(response)
        st.caption(f"Response time: {duration:.2f} seconds")
