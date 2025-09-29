import streamlit as st
import time

from pipeline import build_global_index, build_prompt
from main.llm.factory import get_llm_client
from main.intent_detector import IntentDetector
from main.vector_store import faiss_indexer
from main.embedder import embedder


# === Page Setup ===
st.set_page_config(page_title="Armstrong AI Assistant")
st.title("Armstrong AI Assistant")
st.caption("Ask technical questions about Armstrong products. I'll pull relevant info and answer using a local LLM.")


# === Component Initialization ===
@st.cache_resource
def load_components():
    llm = get_llm_client()
    index = build_global_index(force=False)
    intent_detector = IntentDetector()
    return llm, index, intent_detector

llm, index, intent_detector = load_components()


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


# === Core Logic ===
def respond_to_query(query_text: str) -> str:
    intent = intent_detector.detect(query_text)

    if intent == "greeting":
        return "Hello! I'm your Armstrong assistant. Ask me a technical question and I’ll look it up for you."
    elif intent == "thanks":
        return "You're welcome! Let me know if you have more questions."
    elif intent == "goodbye":
        return "Goodbye! Feel free to come back with more questions anytime."
    elif intent == "help":
        return "I can help answer questions about Armstrong products. Ask me something specific!"
    elif intent == "vague":
        return "Could you please rephrase your question or ask something more specific?"
    elif intent == "empty":
        return None

    top_chunks = faiss_indexer.query_faiss_index(index, query_text, embedder.get_model(), k=4)
    if not top_chunks:
        return "Sorry, I couldn't find relevant information in the documents."

    max_score = max(score for _, score in top_chunks)
    if max_score < 0.2:
        return "I looked through the documents but didn’t find anything helpful for that question."

    context = "\n\n".join(chunk for chunk, _ in top_chunks)
    prompt = build_prompt(context, query_text, st.session_state.history[-3:])
    return llm.generate_answer(prompt)


# === Chat Input ===
query = st.chat_input("Ask me something about Armstrong products...")

if query:
    with st.chat_message("user"):
        st.markdown(query)

    with st.spinner("Thinking..."):
        start = time.time()
        response = respond_to_query(query)
        duration = time.time() - start

    if response:
        st.session_state.history.append((query, response))
        with st.chat_message("assistant"):
            st.markdown(response)
            st.caption(f"Response time: {duration:.2f} seconds")
    else:
        st.warning("Please enter a valid question.")
