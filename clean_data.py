import os
import json
import re
import sys
from typing import List, Dict, Any, Optional

# Connect with pdf_ingestion file
from pdf_ingestion import PDFVulnerabilityIngestor

# Ensure UTF-8 output encoding for Windows terminal if supported
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass


class DataCleaner:
    """
    Document cleaning pipeline that cleans text content inside 'content' by:
    1. Replacing Unicode escape artifacts (\u2022) with standard bullet symbols (•).
    2. Removing hardcoded mid-sentence PDF line breaks inside paragraphs and bullet points.
    3. Extracting key domain-specific security keywords into metadata for vector & hybrid search.
    4. Preserving the exact input chunk schema.
    """

    KNOWN_KEYWORDS_MAP = {
        "A01": ["Access Control", "IDOR", "JWT", "Server-side Validation", "Deny by Default", "Permissions"],
        "A02": ["Security Misconfiguration", "Security Headers", "Automated Hardening", "Secrets Management", "Containers", "Cloud Permissions"],
        "A03": ["SBOM", "CVE", "CI/CD Pipeline", "Dependency Tracking", "MFA", "Package Signatures"],
        "A04": ["TLS 1.2+", "HTTPS", "Argon2", "scrypt", "PBKDF2", "CSPRNG", "Encryption", "Salted Hashing", "Key Management"],
        "A05": ["SQL Injection", "Parameterized Queries", "Prepared Statements", "ORM", "Input Validation", "Allowlists", "SAST", "DAST", "IAST"],
        "A06": ["Threat Modeling", "SDLC", "Secure Architecture", "Misuse Cases", "Tenant Segregation"],
        "A07": ["MFA", "Brute-force", "Credential Stuffing", "Session IDs", "JWT Claims", "Password Hashing"],
        "A08": ["Digital Signatures", "Checksums", "Insecure Deserialization", "CI/CD Protection", "Code Signing"],
        "A09": ["Honeytokens", "Log Injection", "Incident Response", "Security Monitoring", "SIEM"],
        "A10": ["Rate Limiting", "Resource Limits", "Throttling", "Exception Handling", "Fail Safe", "Transaction Rollback"]
    }

    def __init__(self, ingestor: Optional[PDFVulnerabilityIngestor] = None):
        self.ingestor = ingestor or PDFVulnerabilityIngestor()

    def clean_ingested_chunks(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Clean content text of each chunk and enrich metadata with security keywords."""
        cleaned_chunks: List[Dict[str, Any]] = []

        for chunk in chunks:
            raw_content = chunk.get("content", "")
            category_code = chunk.get("category_code", "")

            cleaned_content = self.normalize_text(raw_content)
            keywords = self.extract_keywords(category_code, cleaned_content)

            cleaned_chunk = dict(chunk)
            cleaned_chunk["content"] = cleaned_content

            if "metadata" in cleaned_chunk and isinstance(cleaned_chunk["metadata"], dict):
                meta = dict(cleaned_chunk["metadata"])
                
                # Insert 'keywords' field in metadata
                sections_included = meta.get("sections_included", ["Overview"])
                meta_ordered = {
                    "sections_included": sections_included,
                    "keywords": keywords,
                    "char_count": len(cleaned_content),
                    "word_count": len(cleaned_content.split())
                }
                cleaned_chunk["metadata"] = meta_ordered

            cleaned_chunks.append(cleaned_chunk)

        return cleaned_chunks

    def process_pdf(self, pdf_path: str, output_json_path: Optional[str] = None) -> List[Dict[str, Any]]:
        """Run PDF ingestion and advanced data cleaning with metadata enrichment."""
        print(f"\n[Step 1] Ingesting PDF via pdf_ingestion module...")
        ingested_chunks = self.ingestor.process(pdf_path)

        print(f"\n[Step 2] Applying advanced cleaning & extracting key security keywords into metadata...")
        cleaned_chunks = self.clean_ingested_chunks(ingested_chunks)

        if output_json_path:
            with open(output_json_path, 'w', encoding='utf-8') as f:
                json.dump({
                    "source_pdf": os.path.basename(pdf_path),
                    "pipeline": "pdf_ingestion -> clean_data",
                    "total_pages": self._get_total_pages(pdf_path),
                    "total_chunks": len(cleaned_chunks),
                    "chunks": cleaned_chunks
                }, f, indent=2, ensure_ascii=False)
            print(f"Cleaned chunks saved to: {output_json_path}")

        return cleaned_chunks

    def extract_keywords(self, category_code: str, text: str) -> List[str]:
        """Extract domain-specific security terms and technical keywords from content text."""
        seen_lower = set()
        unique_kw = []

        # Candidate pool starting with known keywords for category
        candidates = []
        if category_code in self.KNOWN_KEYWORDS_MAP:
            candidates.extend(self.KNOWN_KEYWORDS_MAP[category_code])

        # Regex patterns to dynamically catch technical acronyms & terms
        patterns = [
            r'\b(?:TLS 1\.[23]\+?|HTTPS|Argon2|scrypt|PBKDF2|CSPRNG|AES|RSA|SHA-\d+|MD5|JWT|MFA|SBOM|CVE|IDOR|SAST|DAST|IAST|SDLC|SIEM|SQL|NoSQL|ORM)\b',
            r'\b(?:Parameterized Queries|Prepared Statements|Access Control|Deny by Default|Security Headers|Secrets Management|Dependency Tracking|Salted Hashing|Key Management|Input Validation|Allowlists|Threat Modeling|Secure Architecture|Tenant Segregation|Credential Stuffing|Session IDs|Digital Signatures|Checksums|Insecure Deserialization|Honeytokens|Log Injection|Incident Response|Rate Limits?|Throttling|Transaction Rollback)\b'
        ]

        for pat in patterns:
            matches = re.findall(pat, text, re.IGNORECASE)
            candidates.extend([m.strip() for m in matches])

        for kw in candidates:
            kw_clean = kw.strip()
            kw_lower = kw_clean.lower()
            if kw_clean and kw_lower not in seen_lower:
                seen_lower.add(kw_lower)
                # Standardize title casing if all lower
                display_kw = kw_clean.title() if kw_clean.islower() else kw_clean
                unique_kw.append(display_kw)

        return sorted(unique_kw, key=lambda s: s.lower())

    def normalize_text(self, text: str) -> str:
        """
        Comprehensive text normalization:
        - Replaces Unicode bullet escape variants (\u2022) with standard bullet symbols (•).
        - Merges hardcoded PDF mid-sentence line breaks inside prose paragraphs AND inside bullet points.
        - Preserves clean line breaks between topic headings, sub-headings, and distinct bullet items.
        """
        # 1. Clean zero-width and non-breaking spaces
        text = text.replace('\u200b', '').replace('\xa0', ' ')

        # 2. Replace unicode bullet escape variants with standard bullet symbol (•)
        text = re.sub(r'[\u2022\u2023\u25b6\u25c0\u25e6]', '•', text)
        text = re.sub(r'[\u2013\u2014]', '-', text)

        # 3. Strip document headers/footers noise
        text = re.sub(r'OWASP Top 10\s*-?\s*Vulnerability Notes', '', text, flags=re.IGNORECASE)
        text = re.sub(r'Reference:\s*OWASP Top 10 \(2025\).*', '', text, flags=re.IGNORECASE)

        # 4. Process lines and merge multi-line paragraphs and bullet items
        raw_lines = [re.sub(r'[ \t]+', ' ', l.strip()) for l in text.split('\n') if l.strip()]

        formatted_blocks = []
        current_block = []

        for line in raw_lines:
            is_bullet = line.startswith('•') or line.startswith('-')
            is_heading = bool(re.match(r'^(A\d{2}:|What It Is|Why It Happens|How to Fix It)', line, re.IGNORECASE))

            if is_heading:
                if current_block:
                    formatted_blocks.append(' '.join(current_block))
                    current_block = []
                formatted_blocks.append(line)
            elif is_bullet:
                if current_block:
                    formatted_blocks.append(' '.join(current_block))
                    current_block = []
                current_block.append(line)
            else:
                current_block.append(line)

        if current_block:
            formatted_blocks.append(' '.join(current_block))

        result = '\n'.join(formatted_blocks).strip()
        result = re.sub(r' +', ' ', result)
        return result

    @staticmethod
    def _get_total_pages(pdf_path: str) -> int:
        try:
            import fitz
            doc = fitz.open(pdf_path)
            total = len(doc)
            doc.close()
            return total
        except Exception:
            return 1


def run_pipeline(pdf_path: str, output_json_path: Optional[str] = None) -> List[Dict[str, Any]]:
    """Execute complete ingestion & cleaning pipeline."""
    cleaner = DataCleaner()
    return cleaner.process_pdf(pdf_path, output_json_path=output_json_path)


if __name__ == "__main__":
    pdf_file = os.path.join(os.path.dirname(__file__), "Copy of OWASP Top 10 – Vulnerability Notes_easy.pdf")
    output_file = os.path.join(os.path.dirname(__file__), "cleaned_data.json")

    if os.path.exists(pdf_file):
        cleaned_chunks = run_pipeline(pdf_file, output_json_path=output_file)
        
        half_count = (len(cleaned_chunks) + 1) // 2
        print(f"\n==================================================")
        print(f" DISPLAYING FIRST HALF OF CLEANED CHUNKS (1 to {half_count} of {len(cleaned_chunks)})")
        print(f"==================================================")
        
        for idx in range(half_count):
            c = cleaned_chunks[idx]
            print(f"\n--- Chunk {c['chunk_id']} [{c['category_code']}: {c['vulnerability_title']}] ---")
            print(json.dumps(c, indent=2, ensure_ascii=False))
    else:
        print(f"PDF file not found at: {pdf_file}")
