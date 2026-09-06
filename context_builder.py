import os
from typing import List, Dict
from dotenv import load_dotenv

# Load .env from project root
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", ".env"))


def _count_tokens(text: str) -> int:
    """Return an accurate token count for *text* using cl100k_base BPE encoding.

    Falls back to a 4-chars-per-token heuristic if tiktoken is unavailable.
    """
    if not text:
        return 0

    try:
        import tiktoken

        # Use cl100k_base directly to avoid model name lookup errors
        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except Exception:
        # Fallback heuristic: ~4 characters per token
        return max(1, len(text) // 4)


def _truncate_context(chunks: List[Dict], max_tokens: int) -> str:
    """Concatenate chunk contents and truncate to *max_tokens* tokens.

    Supports both top-level 'content' keys and Qdrant-style nested
    'payload.content' structures.
    """
    context_parts = []
    total_tokens = 0

    for chunk in chunks:
        # Robust dictionary key extraction: handles payload dicts or direct dicts
        content = chunk.get("content") or chunk.get("payload", {}).get("content", "")

        if not content:
            continue

        token_len = _count_tokens(content)

        if total_tokens + token_len > max_tokens:
            remaining = max_tokens - total_tokens
            if remaining <= 0:
                break

            # Approximate character truncation for remaining token budget
            approx_chars = remaining * 4
            truncated_content = content[:approx_chars]
            context_parts.append(truncated_content)
            total_tokens = max_tokens
            break
        else:
            context_parts.append(content)
            total_tokens += token_len

    return "\n\n".join(context_parts)


if __name__ == "__main__":
    # Demo that loads actual stored chunks from VectorStore
    # pyrefly: ignore [missing-import]
    from vector_storage import VectorStore
    # Initialize the vector store (defaults to ./qdrant_db)
    store = VectorStore()
    try:
        # Retrieve up to 1000 stored points (adjust limit as needed)
        points, _ = store.client.scroll(
            collection_name=store.COLLECTION_NAME,
            limit=1000,
            with_payload=True,
            with_vectors=False,
        )
        # Extract payload dicts which contain the original chunk metadata
        chunks = [getattr(pt, "payload", {}) for pt in points]
    except Exception as e:
        print(f"[WARN] Unable to retrieve points from VectorStore: {e}")
        chunks = []
    max_tokens = 500  # Adjust token budget as needed
    print("=== Truncated context from stored chunks ===")
    truncated = _truncate_context(chunks, max_tokens)
    print(truncated)