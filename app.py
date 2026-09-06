import uuid
import logging
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
import os

# Load environment variables from .env in project root
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))

# Initialize logger (reuse uvicorn logger if available)
logger = logging.getLogger("uvicorn.error")

# Import RAG components
from rag_retriever import RAGRetriever
from llm_service import generate_answer

app = FastAPI(title="OWASP Security Assistant API")

# Allow CORS for any origin (adjust as needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    session_id: str
    question: str
    top_k: int = 3
    # Optional: limit of candidate chunks for reranking (defaults to 10 in retriever)
    candidate_k: int = 10

@app.post("/api/chat")
async def chat_endpoint(request: ChatRequest):
    """Handle a chat request, retrieve relevant chunks, and stream an LLM answer.
    The response is a streaming text payload compatible with the existing Streamlit client.
    """
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    request_id = str(uuid.uuid4())
    logger.info(f"[REQ {request_id}] Received question: {request.question}")

    # Retrieve relevant context chunks using the RAG pipeline
    retriever = RAGRetriever()
    try:
        chunks = retriever.retrieve(request.question, top_k=request.top_k, candidate_k=request.candidate_k)
    except Exception as e:
        logger.error(f"[REQ {request_id}] Retrieval error: {e}")
        raise HTTPException(status_code=500, detail="Error during retrieval")

    # Stream the answer from the LLM service
    async def answer_generator():
        async for token in generate_answer(request.question, chunks, request_id):
            yield token

    return StreamingResponse(answer_generator(), media_type="text/event-stream")
