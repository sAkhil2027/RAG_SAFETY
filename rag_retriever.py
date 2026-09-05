import os
import json
import re
import sys
from typing import List, Dict, Any, Optional

# Connect with vector_storage module
from vector_storage import VectorStore

# Ensure UTF-8 output encoding for Windows terminal
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass


class RAGRetriever:
    """
    RAG Retriever pipeline:
    User Question -> Query Embedding -> Vector Search (Qdrant) -> Top-K Candidate Chunks -> Hybrid Reranking -> Top-3 Results
    """

    def __init__(self, db_path: str = "./qdrant_db", model_name: str = "all-MiniLM-L6-v2"):
        """Initialize the retriever connected to local Qdrant Vector DB."""
        print(f"Connecting RAG Retriever to Qdrant DB at '{db_path}'...")
        self.vector_store = VectorStore(db_path=db_path, model_name=model_name)

    def retrieve(self, query: str, top_k: int = 1, candidate_k: int = 10) -> List[Dict[str, Any]]:
        """
        Execute RAG retrieval workflow:
        1. Query Embedding & Vector Search in Qdrant (candidate_k initial items)
        2. Hybrid Keyword Reranking
        3. Return Top-K relevant chunks (default top_k = 1)
        """
        query_clean = query.strip()
        if not query_clean:
            return []

        # Step 1 & 2: Embed Query & Run Qdrant Vector Search
        initial_candidates = self.vector_store.search(query_clean, limit=candidate_k)

        if not initial_candidates:
            return []

        # Step 3: Apply Hybrid Reranking to candidates
        reranked_results = self.rerank_chunks(query_clean, initial_candidates)

        # Return top-k items
        return reranked_results[:top_k]

    def rerank_chunks(self, query: str, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Hybrid Reranking:
        Combines Qdrant cosine vector score with exact keyword matching boost.
        """
        query_words = set(re.findall(r'\b\w+\b', query.lower()))
        reranked = []

        for candidate in candidates:
            base_score = candidate.get("score", 0.0)
            category = candidate.get("category_code", "").lower()
            title = candidate.get("vulnerability_title", "").lower()
            metadata = candidate.get("metadata", {})
            keywords = [k.lower() for k in metadata.get("keywords", [])]

            keyword_boost = 0.0

            # 1. Match against metadata keywords (0.05 boost per matched keyword)
            for kw in keywords:
                kw_terms = set(re.findall(r'\b\w+\b', kw))
                if kw_terms and kw_terms.issubset(query_words):
                    keyword_boost += 0.08
                elif any(term in query_words for term in kw_terms if len(term) > 3):
                    keyword_boost += 0.04

            # 2. Match against category code e.g. "a01", "a05" or vulnerability title
            if category and category in query.lower():
                keyword_boost += 0.15
            if title and any(t_word in query_words for t_word in title.split() if len(t_word) > 3):
                keyword_boost += 0.05

            final_score = base_score + keyword_boost

            reranked_item = dict(candidate)
            reranked_item["raw_vector_score"] = base_score
            reranked_item["keyword_boost"] = round(keyword_boost, 4)
            reranked_item["rerank_score"] = round(final_score, 4)
            reranked.append(reranked_item)

        # Sort by rerank_score descending
        reranked.sort(key=lambda item: item["rerank_score"], reverse=True)
        return reranked


if __name__ == "__main__":
    retriever = RAGRetriever()

    # Test Query
    test_question = "How to prevent SQL injection vulnerabilities using parameterized queries?"
    
    print("\n==================================================")
    print(f" USER QUESTION: '{test_question}'")
    print("==================================================")

    # Retrieve top_k = 3 relevant chunks
    top_chunks = retriever.retrieve(test_question, top_k=3, candidate_k=10)

    print(f"\nRetrieved Top-{len(top_chunks)} Relevant Chunks:\n")

    for rank, chunk in enumerate(top_chunks, 1):
        print(f"--- Rank #{rank} [Score: {chunk['rerank_score']} | Vector Score: {chunk['raw_vector_score']}] ---")
        print(f"Category: {chunk['category_code']} - {chunk['vulnerability_title']}")
        print(f"Keywords: {chunk['metadata'].get('keywords', [])}")
        print(f"Content Snippet:\n{chunk['content'][:250]}...\n")
