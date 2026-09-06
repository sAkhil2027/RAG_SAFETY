
import os
import json
import time
import uuid
import logging
from typing import List, Dict, AsyncGenerator
from dotenv import load_dotenv
from groq import AsyncGroq

# Load environment variables (expects .env at project root)
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MODEL_NAME = os.getenv("GROQ_MODEL", "groq/compound-mini")
SYSTEM_PROMPT = os.getenv( 
    "LLM_SYSTEM_PROMPT",
    "You are a security assistant. Use ONLY the provided OWASP context to answer the user question. "
    'If the answer is not present, respond with "I don\'t have enough information to answer this question."',
)
MAX_CONTEXT_TOKENS = int(os.getenv("MAX_CONTEXT_TOKENS", "1500"))

# Initialize logger – reuse existing application logger
logger = logging.getLogger("uvicorn.error")

# Initialize Groq client (base_url is handled internally by AsyncGroq)
client = AsyncGroq(api_key=GROQ_API_KEY)

# Import context helper functions
from context_builder import _count_tokens, _truncate_context


async def generate_answer(
    question: str, chunks: List[Dict], request_id: str
) -> AsyncGenerator[str, None]:
    """Asynchronously stream answer tokens from Groq API.

    Parameters
    ----------
    question : str
        User's question.
    chunks : List[Dict]
        Retrieved chunks from ``rag_retriever``.
    request_id : str
        Unique identifier for logging audit trails.
    """
    context = _truncate_context(chunks, MAX_CONTEXT_TOKENS)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
    ]

    start_time = time.time()
    token_count = 0
    prompt_tokens = 0
    completion_tokens = 0

    try:
        # Asynchronous non-blocking streaming call
        stream = await client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=0.2,
            stream=True,
            stream_options={"include_usage": True},
        )

        async for chunk in stream:
            # 1. Capture token usage metrics inside loop when usage chunk arrives
            if hasattr(chunk, "usage") and chunk.usage is not None:
                prompt_tokens = chunk.usage.prompt_tokens
                completion_tokens = chunk.usage.completion_tokens

            # 2. Guard against empty choices [] (e.g., usage-only chunk)
            if chunk.choices and len(chunk.choices) > 0:
                delta = chunk.choices[0].delta
                content = delta.content or ""
                if content:
                    token_count += _count_tokens(content)
                    yield content

        elapsed = time.time() - start_time

        logger.info(
            f"request_id={request_id} model={MODEL_NAME} "
            f"est_tokens={token_count} prompt_tokens={prompt_tokens} "
            f"completion_tokens={completion_tokens} latency={elapsed:.2f}s"
        )

    except Exception as e:
        logger.error(f"request_id={request_id} error={e}")
        fallback = "I don't have enough information to answer this question."
        yield fallback