from dns import asyncquery
from mpmath import limit
import os
import json
import sys
from typing import List, Dict, Any, Optional

# Connect with clean_data file
from clean_data import run_pipeline

# Import Qdrant and SentenceTransformer libraries
try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import VectorParams, Distance, PointStruct
except Exception as import_err:
    print("[WARN] qdrant_client import failed:", import_err)
    # Minimal in‑memory fallback client
    # Simple point container mimicking Qdrant's PointStruct
    class SimplePoint:
        def __init__(self, id, vector, payload):
            self.id = id
            self.vector = vector
            self.payload = payload

    # Alias PointStruct to SimplePoint when the real class is unavailable
    if 'PointStruct' not in globals():
        PointStruct = SimplePoint

    class SimpleInMemoryClient:
        def __init__(self, *args, **kwargs):
            # Accept any args (e.g., path) to be compatible with QdrantClient signature
            self.collections = {}
            self.points = {}
            # Ignore unexpected keyword arguments like 'path'
            # No further initialization needed for in‑memory storagelf.collections

        def collection_exists(self, name):
            return name in self.collections

        def delete_collection(self, name):
            self.collections.pop(name, None)
            self.points.pop(name, None)

        def create_collection(self, collection_name, vectors_config=None):
            # vectors_config ignored in in‑memory version
            self.collections[collection_name] = vectors_config
            self.points[collection_name] = []

        def upsert(self, collection_name, points):
            self.points.setdefault(collection_name, []).extend(points)

        def search(self, collection_name, query_vector, limit=5, vector_name=None):
            import math
            def cosine(a, b):
                dot = sum(x * y for x, y in zip(a, b))
                norm_a = math.sqrt(sum(x * x for x in a))
                norm_b = math.sqrt(sum(x * x for x in b))
                return dot / (norm_a * norm_b) if norm_a and norm_b else 0.0

            hits = []
            for pt in self.points.get(collection_name, []):
                score = cosine(pt.vector["embedding"], query_vector)
                hit = type('Res', (), {'score': score, 'payload': pt.payload})
                hits.append(hit)
            hits.sort(key=lambda r: r.score, reverse=True)
            return hits[:limit]

    # Alias QdrantClient to fallback client
    QdrantClient = SimpleInMemoryClient

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
        # Use HTTP mode (requires a running Qdrant server on localhost:6333)
        # If you prefer embedded mode, ensure qdrant-client local extras are installed.
        print(f"Initializing Qdrant Vector DB client at: {db_path} (fallback if needed)...")
        # ==== NEW LOGIC ==== 
        # For this educational demo we always use the in‑memory fallback client.
        # This guarantees that the client has the required `search` method and avoids
        # reliance on an external Qdrant server which may not be running.
        try:
            # Attempt to create a real Qdrant client – if it fails we fall back.
            self.client = QdrantClient(path=db_path)
        except Exception as e:
            print("[WARN] Real Qdrant init failed, switching to SimpleInMemoryClient:", e)
            self.client = SimpleInMemoryClient()
        # ====================

        print(f"Loading Embedding Model ({model_name})...")
        self.encoder = SentenceTransformer(model_name)
        print("Embedding Model loaded successfully.")

    def initialize_collection(self):
        """Re-create or initialize Qdrant collection for vector search."""
        if self.client.collection_exists(self.COLLECTION_NAME):
            self.client.delete_collection(self.COLLECTION_NAME)

        # If the real Qdrant client is available, configure vector params.
        if 'VectorParams' in globals() and hasattr(self.client, 'create_collection'):
            self.client.create_collection(
                collection_name=self.COLLECTION_NAME,
                vectors_config={"embedding": VectorParams(size=self.VECTOR_DIM, distance=Distance.COSINE)}
            )
        else:
            # Fallback client (in‑memory) does not need vector config.
            self.client.create_collection(collection_name=self.COLLECTION_NAME)
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
                vector={"embedding": embedding},
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
            limit=limit,
            vector_name="embedding"
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

    def inspect_qdrant_db(self) -> None:
        """Debug helper to check stored items directly in Qdrant (or fallback).
        Prints collection stats and samples of the first two stored points.
        """
        # 1. Collection statistics (real Qdrant only)
        try:
            info = self.client.get_collection(self.COLLECTION_NAME)
            print("\n--- QDRANT STATS ---")
            print(f"Total Points Stored: {getattr(info, 'points_count', 'N/A')}")
            print(f"Vectors Count: {getattr(info, 'indexed_vectors_count', 'N/A')}")
        except Exception as e:
            print("[WARN] get_collection not available:", e)

        # 2. Retrieve a couple of raw points
        try:
            sample_points, _ = self.client.scroll(
                collection_name=self.COLLECTION_NAME,
                limit=2,
                with_payload=True,
                with_vectors=True,
            )
            for pt in sample_points:
                print(f"\n[POINT ID {getattr(pt, 'id', 'N/A')}]")
                has_vec = getattr(pt, "vector", None) is not None
                print(f"Has Vector: {has_vec}")
                if has_vec:
                    vec = pt.vector.get("embedding") if isinstance(pt.vector, dict) else pt.vector
                    length = len(vec) if vec else None
                    print(f"Vector Length: {length}")
                payload = getattr(pt, "payload", {})
                print(f"Payload Keys: {list(payload.keys())}")
                print(f"Content Sample: {str(payload.get('content', ''))[:100]}")
        except Exception as e:
            print("[WARN] scroll not available:", e)
        """Search Qdrant vector database using vector similarity search."""
        query_vector = self.encoder.encode(query).tolist()
        
        search_results = self.client.search(
            collection_name=self.COLLECTION_NAME,
            query_vector=query_vector,
            limit=limit,
            vector_name="embedding"
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

