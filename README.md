# 🛡️ OWASP Security Assistant (RAG Pipeline)

An intelligent, secure **Retrieval-Augmented Generation (RAG)** pipeline and web application designed to ingest OWASP Top 10 security documentation, extract structured vulnerability information, index content into dense vector embeddings, and generate expert remediation guidance using **Groq LLMs**.

---

## 🚀 Key Features

- **Document Ingestion & Chunking (`pdf_ingestion.py`):**
  - Custom PyMuPDF-based parser that chunks documents on a "one chunk per vulnerability" basis (`A01` through `A10`).
  - Preserves subsections (*What It Is*, *Why It Happens*, *How to Fix It*).

- **Data Cleaning & Hybrid Keyword Extraction (`clean_data.py`):**
  - Normalizes PDF formatting, removes Unicode noise, and merges broken paragraphs.
  - Extracts domain-specific security keywords (e.g., IDOR, CSPRNG, Argon2, PBKDF2, SAST, DAST, SQLi).

- **Dense Embeddings & Vector Storage (`vector_storage.py`, `encoder.py`):**
  - Uses `all-MiniLM-L6-v2` via a thread-safe singleton encoder.
  - Stores embeddings and full metadata payloads in Qdrant (with an in-memory fallback for local execution).

- **Two-Stage Retrieval & Hybrid Reranking (`rag_retriever.py`, `retrieval_engine.py`):**
  - Candidate vector retrieval combined with keyword and category code score boosts.
  - Supports paired vulnerability and guidance retrieval.

- **Privacy Guardrails & Sanitization (`sanitizer.py`, `rules.json`):**
  - Configurable regex-based redaction for emails, IP addresses, local user directories, and API keys.
  - Whitelist-aware rules preserving version numbers (e.g., `v10.0.12.100`).
  - Dynamic rule reloading endpoint for administrative configuration.

- **FastAPI Backend & Rate Limiting (`app.py`, `rate_limiter.py`):**
  - Streaming `/api/chat` endpoint using Server-Sent Events (SSE).
  - Specialized `/explain_vulns` endpoint.
  - In-memory sliding window rate limiter (60 req/min per IP).

- **Interactive Streamlit Web UI (`streamlit_app.py`):**
  - Real-time token streaming, session management, and chat history controls.

---

## 📂 Project Structure

```
├── Copy of OWASP Top 10 – Vulnerability Notes_easy.pdf # Source PDF documentation
├── pdf_ingestion.py             # PDF text extraction & vulnerability chunking
├── structured_chunks.json       # Ingested vulnerability topic chunks
├── clean_data.py                # Text normalization & security keyword tagging
├── cleaned_data.json            # Cleaned chunks ready for embedding
├── encoder.py                   # SentenceTransformer lazy singleton
├── vector_storage.py            # Qdrant client & vector storage interface
├── retrieval_engine.py          # Vulnerability and OWASP guidance retrieval
├── rag_retriever.py             # Hybrid reranking and retrieval pipeline
├── context_builder.py           # Tiktoken-based token counter and context truncation
├── rules.json                   # Regex patterns for data sanitization & whitelist
├── sanitizer.py                 # Configurable text sanitizer
├── prompt_builder.py            # Prompt engineering and strict guidance formatting
├── llm_formatter.py             # JSON schema validation for model responses
├── llm_service.py               # Async Groq LLM client integration
├── rate_limiter.py              # In-memory IP-based rate limiting middleware
├── app.py                       # FastAPI application & REST endpoints
├── streamlit_app.py             # Interactive Streamlit frontend UI
├── Dockerfile & docker-compose  # Containerization configurations
└── tests/                       # Pytest test suite
```

---

## 🛠️ Getting Started

### 1. Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/sAkhil2027/shieldove.git
cd shieldove
pip install -r requirements.txt
```

### 2. Environment Variables

Create a `.env` file in the root directory:

```env
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=groq/compound-mini
LLM_SYSTEM_PROMPT=You are a security assistant. Use ONLY the provided OWASP context to answer the user question.
MAX_CONTEXT_TOKENS=1500
ENABLE_ADMIN_ENDPOINT=true
ADMIN_RELOAD_TOKEN=your_secure_admin_token
```

### 3. Run the Backend API

Start the FastAPI application:

```bash
uvicorn app:app --host 127.0.0.1 --port 8000 --reload
```

Interactive API documentation will be available at:
- **Swagger UI:** `http://127.0.0.1:8000/docs`
- **ReDoc:** `http://127.0.0.1:8000/redoc`

### 4. Run the Streamlit UI

In a separate terminal, launch the Streamlit frontend:

```bash
streamlit run streamlit_app.py
```

Access the chat interface at: `http://localhost:8501`.

---

## 🧪 Testing

Run unit and integration tests using `pytest`:

```bash
pytest
```
