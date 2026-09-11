import uuid
import requests
import streamlit as st

BACKEND_URL = "http://127.0.0.1:8000/api/chat"
INGEST_URL = "http://127.0.0.1:8000/api/ingest_pdf"
KB_STATUS_URL = "http://127.0.0.1:8000/api/kb_status"

st.set_page_config(page_title="RAG Chatbot", page_icon="🤖", layout="wide")
st.title("🤖 OWASP Security Assistant")

# Initialize persistent session state
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar controls
with st.sidebar:
    st.subheader("Knowledge Base Status")
    try:
        kb_resp = requests.get(KB_STATUS_URL, timeout=3)
        if kb_resp.status_code == 200:
            kb_data = kb_resp.json()
            if kb_data.get("status") == "ready":
                st.success(f"🟢 **KB Active:** {kb_data.get('indexed_chunks')} chunks indexed via Groq API")
            else:
                st.warning("🔴 **KB Empty:** Please upload an OWASP PDF below")
    except Exception:
        st.caption("KB Status: Backend connecting...")

    st.divider()
    st.subheader("Session Controls")
    st.caption(f"Session ID:\n`{st.session_state.session_id}`")
    if st.button("Clear History"):
        st.session_state.messages = []
        st.session_state.session_id = str(uuid.uuid4())
        st.rerun()

    st.divider()
    st.subheader("📄 Dynamic Document Ingestion")
    st.caption("Upload an OWASP Top 10 vulnerability notes PDF. Embeddings are generated fresh via Groq API.")
    uploaded_pdf = st.file_uploader("Select PDF file", type=["pdf"])

    if uploaded_pdf is not None:
        if st.button("🚀 Ingest & Index Document"):
            with st.spinner("Validating document pattern and indexing..."):
                try:
                    files = {"file": (uploaded_pdf.name, uploaded_pdf.getvalue(), "application/pdf")}
                    res = requests.post(INGEST_URL, files=files, timeout=60)
                    if res.status_code == 200:
                        data = res.json()
                        st.success(f"✅ {data.get('message', 'Document indexed successfully!')}")
                        if "stats" in data:
                            cats = ", ".join(data["stats"].get("categories", []))
                            st.info(f"📋 **Categories Indexed ({len(data['stats'].get('categories', []))}):** {cats}")
                    else:
                        error_detail = res.json().get("detail", res.text)
                        st.error(f"⚠️ **Validation Error**: {error_detail}")
                except Exception as e:
                    st.error(f"❌ Connection failed: {e}")

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