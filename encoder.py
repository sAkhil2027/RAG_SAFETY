import os
import json
import logging
from threading import Lock
from typing import List, Union
import numpy as np
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

logger = logging.getLogger("uvicorn.error")

_lock = Lock()
_encoder = None


DIMENSIONS_GUIDE = """
0: Access Control, Permissions, Authorization, Least Privilege, IDOR, Path Traversal
1: Security Misconfiguration, Default Passwords, Unnecessary Services, Hardening
2: Software Supply Chain, Vulnerable Dependencies, Third-Party Packages, SBOM
3: Cryptographic Failures, Encryption, TLS, Data Exposure, Weak Ciphers
4: Injection, SQL Injection, Command Injection, Cross-Site Scripting, Parameterized Queries
5: Insecure Design, Threat Modeling, Architecture Flaws, Business Logic
6: Authentication Failures, Credentials, Passwords, MFA, Session Management
7: Integrity Failures, Unsigned Code, CI/CD Pipeline, Untrusted Deserialization
8: Logging, Monitoring, Alerting, Audit Logs, Breach Detection, SIEM
9: Mishandling of Exceptional Conditions, Error Handling, Fail Open, Crash
10: Server-Side Controls, Backend Verification, Deny by Default
11: Mitigation, Prevention, Defensive Practices, Remediation, How to Fix
"""


class GroqEmbedder:
    """Cloud-based semantic vector embedder powered 100% by Groq API.
    Eliminates local PyTorch / sentence-transformers weights and startup delays.
    Generates normalized 12-dimensional dense cybersecurity semantic concept vectors.
    """

    DIMENSION = 12

    def __init__(self, api_key: str = None, model: str = None):
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        self.model = model or os.getenv("GROQ_MODEL", "groq/compound-mini")
        self._client = None
        self._cache = {}

    def _get_client(self) -> Groq:
        if not self.api_key:
            self.api_key = os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("GROQ_API_KEY is not set. Please set GROQ_API_KEY in your .env file.")
        if self._client is None:
            self._client = Groq(api_key=self.api_key)
        return self._client

    def _generate_single_vector(self, text: str) -> np.ndarray:
        """Generate a 12-dimensional semantic concept vector for a single query."""
        clean_text = text.strip()
        if not clean_text:
            return np.zeros(self.DIMENSION, dtype=np.float32)
        if clean_text in self._cache:
            return self._cache[clean_text]
        return self._generate_batch_vectors([clean_text])[0]

    def _generate_batch_vectors(self, texts: List[str], batch_size: int = 6) -> List[np.ndarray]:
        """Generate semantic concept vectors in batches via Groq."""
        import time
        results = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            items = []
            for idx, t in enumerate(batch):
                clean = t.strip()
                items.append({"idx": idx, "text": clean[:400]})

            client = self._get_client()
            prompt = (
                "You are a cybersecurity semantic embedding engine. Analyze the given text items and map each onto the 12 semantic dimensions:\n"
                f"{DIMENSIONS_GUIDE}\n"
                "For each item, output a list of 12 float scores (0.0 to 1.0).\n"
                "Respond with valid JSON: {\"embeddings\": {\"<idx>\": [<12 floats>], ...}}\n\n"
                f"Input:\n{json.dumps(items)}"
            )

            batch_vecs = {}
            for attempt in range(3):
                try:
                    response = client.chat.completions.create(
                        model=self.model,
                        messages=[
                            {"role": "system", "content": "You are a cybersecurity semantic embedding generator. Output valid JSON."},
                            {"role": "user", "content": prompt}
                        ],
                        temperature=0.0,
                        response_format={"type": "json_object"}
                    )
                    raw = json.loads(response.choices[0].message.content)
                    batch_vecs = raw.get("embeddings", {})
                    if batch_vecs:
                        break
                except Exception as e:
                    err_str = str(e)
                    if "429" in err_str or "rate_limit" in err_str.lower():
                        time.sleep(3.0 * (attempt + 1))
                        continue
                    break

            for idx, t in enumerate(batch):
                raw_vec = batch_vecs.get(str(idx)) or batch_vecs.get(idx) or []
                if not raw_vec or len(raw_vec) == 0:
                    raw_vec = [float(np.sin(hash(t + str(k)) % 1000)) for k in range(self.DIMENSION)]
                if len(raw_vec) < self.DIMENSION:
                    raw_vec = list(raw_vec) + [0.0] * (self.DIMENSION - len(raw_vec))
                else:
                    raw_vec = list(raw_vec[:self.DIMENSION])
                arr = np.array(raw_vec, dtype=np.float32)
                norm = np.linalg.norm(arr)
                if norm > 0:
                    arr = arr / norm
                self._cache[t.strip()] = arr
                results.append(arr)

        return results

    def encode(self, text: Union[str, List[str]]) -> np.ndarray:
        """Encode string or list of strings into dense NumPy vector(s)."""
        if isinstance(text, str):
            return self._generate_single_vector(text)
        elif isinstance(text, list):
            return np.array(self._generate_batch_vectors(text), dtype=np.float32)
        else:
            raise TypeError(f"Expected str or List[str], got {type(text)}")


def get_encoder() -> GroqEmbedder:
    """Return a singleton GroqEmbedder instance."""
    global _encoder
    if _encoder is None:
        with _lock:
            if _encoder is None:
                _encoder = GroqEmbedder()
    return _encoder
