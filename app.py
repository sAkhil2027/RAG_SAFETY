# import os
# import json
# import uvicorn
# from dotenv import load_dotenv
# from fastapi import FastAPI, HTTPException
# from fastapi.middleware.cors import CORSMiddleware
# from fastapi.responses import StreamingResponse, FileResponse
# from fastapi.staticfiles import StaticFiles
# from pydantic import BaseModel
# from typing import List, Optional
# import uuid
# from openai import OpenAI
# from rag_retriever import RAGRetriever
# # Load environment variables
# load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))
# from llm_service import generate_answer as generate_answer

# GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# if not GROQ_API_KEY:
#     print("[WARNING] GROQ_API_KEY environment variable not found in .env!")

# # Initialize OpenAI client with Groq endpoint
# groq_client = OpenAI(
#     base_url="https://api.groq.com/openai/v1",
#     api_key=GROQ_API_KEY or "dummy_key"
# )

# app = FastAPI(title="Groq AI Chatbot")

# # Enable CORS
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# class ChatMessage(BaseModel):
#     role: str
#     content: str

# class ChatRequest(BaseModel):
#     messages: List[ChatMessage]
#     model: Optional[str] = "groq/compound-mini"

# class QueryRequest(BaseModel):
#     question: str
#     top_k: int = 3

# PREFERRED_MODELS = [
#     {"id": "groq/compound-mini", "name": "Groq Compound Mini (Fastest)"},
#     {"id": "groq/compound", "name": "Groq Compound (Standard)"},
#     {"id": "qwen/qwen3.6-27b", "name": "Qwen 3.6 27B"},
#     {"id": "qwen/qwen3.8-27b", "name": "Qwen 3.8 27B"},
#     {"id": "openai/gpt-oss-120b", "name": "GPT-OSS 120B"},
#     {"id": "openai/gpt-oss-20b", "name": "GPT-OSS 20B"},
# ]

# @app.get("/api/models")
# def get_models():
#     """Fetch available models from Groq API or return default fallback list."""
#     try:
#         if GROQ_API_KEY:
#             fetched_models = groq_client.models.list()
#             active_ids = {m.id for m in fetched_models.data}
#             # Filter preferred models that are active, or include all active chat models
#             available = []
#             for p in PREFERRED_MODELS:
#                 if p["id"] in active_ids:
#                     available.append(p)
            
#             # Add any other chat models returned by Groq API
#             preferred_ids = {p["id"] for p in PREFERRED_MODELS}
#             for m in fetched_models.data:
#                 if m.id not in preferred_ids and not m.id.startswith("whisper") and not "guard" in m.id:
#                     available.append({"id": m.id, "name": m.id})
                    
#             if available:
#                 return {"models": available}
#     except Exception as e:
#         print("[WARNING] Failed to fetch Groq models dynamically:", e)
    
#     return {"models": PREFERRED_MODELS}

# @app.post("/answer")
# async def answer_endpoint(request: QueryRequest):
#     """Handle answer generation with RAG and streaming response."""
#     if not GROQ_API_KEY:
#         raise HTTPException(status_code=500, detail="GROQ_API_KEY is not set in backend environment.")
#     retriever = RAGRetriever()
#     chunks = retriever.retrieve(request.question, top_k=request.top_k)
#     request_id = str(uuid.uuid4())
#     return StreamingResponse(
#         generate_answer(request.question, chunks, request_id),
#         media_type="text/event-stream"
#     )
# @app.post("/api/chat")
# async def chat_endpoint(request: QueryRequest):
#     """Alias for /answer endpoint to match frontend expectations."""
#     # Reuse the same logic as the primary answer endpoint
#     return answer_endpoint(request)
# # Static files setup
# static_dir = os.path.join(os.path.dirname(__file__), "static")
# if not os.path.exists(static_dir):
#     os.makedirs(static_dir)

# app.mount("/static", StaticFiles(directory=static_dir), name="static")

# @app.get("/")
# def read_index():
#     index_path = os.path.join(static_dir, "index.html")
#     if os.path.exists(index_path):
#         return FileResponse(index_path)
#     return {"message": "Chatbot backend active. Please create static/index.html"}

# if __name__ == "__main__":
#     print("Starting Groq Chatbot Server on http://127.0.0.1:8000")
#     uvicorn.run(app, host="127.0.0.1", port=8000)








import os
import uuid
from typing import List, Dict, Optional
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, ConfigDict
from groq import Groq

# Initialize FastAPI App
app = FastAPI(title="RAG Chatbot API")

# Enable CORS for Streamlit cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


from dotenv import load_dotenv
load_dotenv()

# Retrieve key safely
api_key = os.environ.get("GROQ_API_KEY")
if not api_key:
    raise ValueError("GROQ_API_KEY is missing! Add it to your .env file or environment.")

groq_client = Groq(api_key=api_key)
# # Initialize Groq Client
# groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# In-memory session store for chat history
CONVERSATION_HISTORY: Dict[str, List[Dict[str, str]]] = {}


class ChatRequest(BaseModel):
    session_id: Optional[str] = None
    question: Optional[str] = None
    prompt: Optional[str] = None
    message: Optional[str] = None
    top_k: int = 3

    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    @property
    def query_text(self) -> str:
        return (self.question or self.prompt or self.message or "").strip()


def mock_qdrant_retriever(query: str, top_k: int = 3) -> List[Dict]:
    """
    Mock retriever simulating Qdrant vector search output.
    Replace or connect this with your actual Qdrant retriever module.
    """
    return [
        {
            "payload": {
                "source": "OWASP_Top_10.pdf",
                "page": 14,
                "text": "Broken Access Control ranks as the #1 web application security risk.",
            }
        },
        {
            "payload": {
                "source": "OWASP_Top_10.pdf",
                "page": 18,
                "text": "Cryptographic Failures relate to sensitive data exposure during transit or rest.",
            }
        },
    ][:top_k]


def extract_page_references(chunks: List[Dict]) -> List[str]:
    """Extract and deduplicate page references from document chunk payloads."""
    refs = set()
    for chunk in chunks:
        payload = chunk.get("payload", chunk)
        page = payload.get("page") or payload.get("page_number") or payload.get("metadata", {}).get("page")
        source = payload.get("source") or payload.get("file_name") or payload.get("metadata", {}).get("source")

        if page and source:
            refs.add(f"{source} (Page {page})")
        elif page:
            refs.add(f"Page {page}")
        elif source:
            refs.add(f"{source}")

    return sorted(list(refs))


@app.post("/api/chat")
async def chat_endpoint(request: ChatRequest):
    user_query = request.query_text
    if not user_query:
        raise HTTPException(status_code=400, detail="Query text cannot be empty.")

    # Manage session and conversation history
    session_id = request.session_id or str(uuid.uuid4())
    if session_id not in CONVERSATION_HISTORY:
        CONVERSATION_HISTORY[session_id] = []

    # 1. Retrieve context chunks from vector DB
    chunks = mock_qdrant_retriever(user_query, top_k=request.top_k)
    page_refs = extract_page_references(chunks)

    # 2. Extract retrieved context text
    context_str = "\n".join([c.get("payload", {}).get("text", "") for c in chunks])

    # 3. Append current user query to session history
    CONVERSATION_HISTORY[session_id].append({"role": "user", "content": user_query})

    # 4. Prepare message list for Groq API including system context
    messages_payload = [
        {
            "role": "system",
            "content": f"You are a helpful security assistant. Answer using the context below:\n{context_str}",
        }
    ] + CONVERSATION_HISTORY[session_id]

    async def stream_response_with_citations():
        full_response = "" 

        # Stream response chunks from Groq API
        completion = groq_client.chat.completions.create(
            model="groq/compound-mini",
            messages=messages_payload,
            temperature=0.3,
            stream=True,
        )

        for chunk in completion:
            token = chunk.choices[0].delta.content or ""
            if token:
                full_response += token
                yield token

        # Append references at the end of output stream
        if page_refs:
            ref_text = "\n\n**References & Sources:**\n" + "\n".join([f"- {ref}" for ref in page_refs])
            full_response += ref_text
            yield ref_text

        # Update assistant response in history
        CONVERSATION_HISTORY[session_id].append({"role": "assistant", "content": full_response})

    return StreamingResponse(
        stream_response_with_citations(),
        media_type="text/event-stream",
        headers={"X-Session-ID": session_id},
    )


@app.get("/api/history/{session_id}")
async def get_history(session_id: str):
    return {"session_id": session_id, "history": CONVERSATION_HISTORY.get(session_id, [])}