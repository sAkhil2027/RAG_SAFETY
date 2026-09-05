import uuid
import requests
import streamlit as st

BACKEND_URL = "http://127.0.0.1:8000/api/chat"

st.set_page_config(page_title="RAG Chatbot", page_icon="🤖", layout="wide")
st.title("🤖 OWASP Security Assistant")

# Initialize persistent session state
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar controls
with st.sidebar:
    st.subheader("Session Controls")
    st.caption(f"Session ID:\n`{st.session_state.session_id}`")
    if st.button("Clear History"):
        st.session_state.messages = []
        st.session_state.session_id = str(uuid.uuid4())
        st.rerun()

# Render previous chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User Chat Input
if user_input := st.chat_input("Ask a security question..."):
    # Append user prompt to UI state
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Stream response into assistant UI container
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_response = ""

        try:
            payload = {
                "session_id": st.session_state.session_id,
                "question": user_input,
                "top_k": 3,
            }

            response = requests.post(BACKEND_URL, json=payload, stream=True, timeout=30)

            if response.status_code == 200:
                for chunk in response.iter_content(chunk_size=None, decode_unicode=True):
                    if chunk:
                        full_response += chunk
                        response_placeholder.markdown(full_response + "▌")

                response_placeholder.markdown(full_response)
                st.session_state.messages.append({"role": "assistant", "content": full_response})
            else:
                st.error(f"Error {response.status_code}: Failed to generate answer.")

        except Exception as e:
            st.error(f"Connection failed: {e}")