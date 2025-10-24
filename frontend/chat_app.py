import streamlit as st
import time
import requests

MAX_HISTORY_LENGTH = 10

# === Initialize Chat History ===
st.session_state.history = st.session_state.get("history", [])


def query_backend(query_text: str, history: list[tuple[str, str]]) -> str:
    try:
        payload = {
            "query": query_text,
            "history": history
        }
        response = requests.post("http://localhost:8000/query", json=payload)
        if response.status_code == 200:
            return response.json().get("results", ["No response"])[0]
        else:
            return f"Error {response.status_code}: {response.text}"
    except Exception as e:
        return f"Request failed: {e}"


def setup_page():
    st.set_page_config(page_title="Armstrong AI Assistant")
    st.title("Armstrong AI Assistant")
    st.caption("Ask questions about Armstrong products. I'll pull relevant info and answer.")


def handle_sidebar():
    with st.sidebar:
        st.header("Options")
        if st.button("Reset Conversation"):
            st.session_state.history.clear()
            st.rerun()


def display_chat_history():
    for user_msg, assistant_msg in st.session_state.history:
        with st.chat_message("user"):
            st.markdown(user_msg)
        with st.chat_message("assistant"):
            st.markdown(assistant_msg)


def handle_user_input():
    query: str | None = st.chat_input("Ask me something about Armstrong products...")
    if query:
        with st.chat_message("user"):
            st.markdown(query)

        with st.spinner("Thinking..."):
            start = time.time()
            response = query_backend(query, st.session_state.history[-MAX_HISTORY_LENGTH:])
            duration = time.time() - start

        st.session_state.history.append((query, response))
        if len(st.session_state.history) > MAX_HISTORY_LENGTH:
            st.session_state.history = st.session_state.history[-MAX_HISTORY_LENGTH:]

        with st.chat_message("assistant"):
            st.markdown(response)
            st.caption(f"Response time: {duration:.2f} seconds")


def show_footer():
    st.markdown("---")
    st.caption("Powered by Armstrong RAG Engine.")


def main():
    setup_page()
    handle_sidebar()
    display_chat_history()
    handle_user_input()
    show_footer()


if __name__ == "__main__":
    main()
