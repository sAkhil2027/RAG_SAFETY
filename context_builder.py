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