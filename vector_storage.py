import os
import json
import sys
from typing import List, Dict, Any, Optional

# Connect with clean_data file
from clean_data import run_pipeline

# Import Qdrant and SentenceTransformer libraries
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from sentence_transformers import SentenceTransformer

# Ensure UTF-8 output encoding for Windows terminal
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass


class VectorStore:
    """
    Vector storage and search engine connected to clean_data output, 
    generating dense embeddings and storing points in Qdrant vector database.
    """

    COLLECTION_NAME = "owasp_vulnerabilities"
    VECTOR_DIM = 384  # Dimension of all-MiniLM-L6-v2

    def __init__(self, db_path: str = "./qdrant_db", model_name: str = "all-MiniLM-L6-v2"):
        """Initialize Qdrant local persistent client and embedding model."""
        self.db_path = db_path
        print(f"Initializing Qdrant Vector DB client at: {db_path}...")
        self.client = QdrantClient(path=db_path)

        print(f"Loading Embedding Model ({model_name})...")
        self.encoder = SentenceTransformer(model_name)
        print("Embedding Model loaded successfully.")

    def initialize_collection(self):
        """Re-create or initialize Qdrant collection for vector search."""
        if self.client.collection_exists(self.COLLECTION_NAME):
            self.client.delete_collection(self.COLLECTION_NAME)

        self.client.create_collection(
            collection_name=self.COLLECTION_NAME,
            vectors_config=VectorParams(size=self.VECTOR_DIM, distance=Distance.COSINE)
        )
        print(f"Collection '{self.COLLECTION_NAME}' initialized in Qdrant.")

    def embed_and_store(self, chunks: List[Dict[str, Any]]) -> int:
        """
        Takes cleaned chunks output from clean_data, generates vector embeddings,
        and stores them in Qdrant database.
        """
        self.initialize_collection()

        points: List[PointStruct] = []

        print(f"\nGenerating embeddings and preparing Qdrant points for {len(chunks)} chunks...")
        for c in chunks:
            chunk_id = c.get("chunk_id", 1)
            content_text = c.get("content", "")
            
            # 1. Compute dense vector embedding for chunk content
            embedding = self.encoder.encode(content_text).tolist()

            # 2. Build payload preserving all input chunk properties & metadata
            payload = {
                "chunk_id": chunk_id,
                "category_code": c.get("category_code", ""),
                "vulnerability_title": c.get("vulnerability_title", ""),
                "page_numbers": c.get("page_numbers", [1]),
                "content": content_text,
                "metadata": c.get("metadata", {})
            }

            points.append(PointStruct(
                id=chunk_id,
                vector=embedding,
                payload=payload
            ))

        # 3. Upsert points into Qdrant vector database
        self.client.upsert(
            collection_name=self.COLLECTION_NAME,
            points=points
        )
        print(f"Successfully stored {len(points)} embedded chunks in Qdrant database!")
        return len(points)

    def search(self, query: str, limit: int = 3) -> List[Dict[str, Any]]:
        """Search Qdrant vector database using vector similarity search."""
        query_vector = self.encoder.encode(query).tolist()
        
        search_results = self.client.search(
            collection_name=self.COLLECTION_NAME,
            query_vector=query_vector,
            limit=limit
        )

        formatted_results = []
        for res in search_results:
            formatted_results.append({
                "score": round(res.score, 4),
                "chunk_id": res.payload.get("chunk_id"),
                "category_code": res.payload.get("category_code"),
                "vulnerability_title": res.payload.get("vulnerability_title"),
                "content": res.payload.get("content"),
                "metadata": res.payload.get("metadata")
            })

        return formatted_results


def process_and_store(pdf_path: str, db_path: str = "./qdrant_db") -> VectorStore:
    """
    1. Ingest & Clean via clean_data module (connected to pdf_ingestion).
    2. Generate embeddings and store in Qdrant database.
    """
    print("==================================================")
    print(" PIPELINE STEP 1 & 2: INGESTION & CLEANING")
    print("==================================================")
    cleaned_chunks = run_pipeline(pdf_path)

    print("\n==================================================")
    print(" PIPELINE STEP 3: EMBEDDING & QDRANT STORAGE")
    print("==================================================")
    vector_store = VectorStore(db_path=db_path)
    vector_store.embed_and_store(cleaned_chunks)

    return vector_store


if __name__ == "__main__":
    pdf_file = os.path.join(os.path.dirname(__file__), "Copy of OWASP Top 10 – Vulnerability Notes_easy.pdf")

    if os.path.exists(pdf_file):
        # Run complete pipeline: pdf_ingestion -> clean_data -> vector_storage (Qdrant)
        v_store = process_and_store(pdf_file)

        # Test Vector Search Query
        test_query = "How to prevent SQL injection vulnerabilities using parameterized queries?"
        print(f"\n==================================================")
        print(f" TESTING QDRANT VECTOR SEARCH FOR: '{test_query}'")
        print(f"==================================================")
        
        results = v_store.search(test_query, limit=2)
        for rank, res in enumerate(results, 1):
            print(f"\n--- Match #{rank} (Similarity Score: {res['score']}) ---")
            print(json.dumps(res, indent=2, ensure_ascii=False))
    else:
        print(f"PDF file not found at: {pdf_file}")
