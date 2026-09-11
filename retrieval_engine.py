import os
from typing import List, Dict

from vector_storage import VectorStore

# Vulnerability store (default collection)
vuln_store = VectorStore()

# Guidance store – points to OWASP guidance collection
# Assume OWASP vector DB is located at ./owasp_db and collection name is owasp_guidance
guidance_store = VectorStore(db_path='./owasp_db')
guidance_store.COLLECTION_NAME = 'owasp_guidance'

def search_vulnerabilities(query: str, top_k: int = 2) -> List[Dict]:
    """Return up to `top_k` vulnerability payloads matching the query."""
    return vuln_store.search(query, limit=top_k)

def match_owasp_guidance(vuln_payload: Dict) -> str:
    """Fetch the single most relevant OWASP guidance chunk for a vulnerability.
    Uses the vulnerability `title` or `description` as the query.
    Returns the raw guidance text or an empty string if not found.
    """
    query = vuln_payload.get('title') or vuln_payload.get('description', '')
    results = guidance_store.search(query, limit=1)
    if not results and vuln_store:
        results = vuln_store.search(query, limit=1)
    if results:
        return results[0].get('content', '')
    return ''
