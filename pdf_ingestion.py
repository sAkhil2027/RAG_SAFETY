import os
import json
import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
import fitz  # PyMuPDF


@dataclass
class VulnerabilityChunk:
    chunk_id: int
    category_code: str
    vulnerability_title: str
    page_numbers: List[int]
    content: str
    metadata: Dict[str, Any]


class PDFVulnerabilityIngestor:
    """PDF Ingestion pipeline that creates One Chunk Per Vulnerability topic."""

    TOPIC_HEADING_PATTERN = r'(?i)([aA]\d{2}:[^\n]+)'
    SUBSECTION_PATTERN = r'(?i)(what it is|why it happens|how to fix it)'

    MIN_REQUIRED_CATEGORIES = 3
    REQUIRED_SUBSECTIONS = ["What It Is", "Why It Happens", "How to Fix It"]

    def extract_pages_from_doc(self, doc: fitz.Document) -> List[Dict[str, Any]]:
        """Extract clean text page by page from a fitz Document."""
        extracted_pages = []
        for page_idx in range(len(doc)):
            page = doc[page_idx]
            raw_text = page.get_text()
            cleaned_text = self._clean_text(raw_text)
            if cleaned_text:
                extracted_pages.append({
                    "page_number": page_idx + 1,
                    "text": cleaned_text
                })
        return extracted_pages

    def extract_pages(self, pdf_path: str) -> List[Dict[str, Any]]:
        """Extract clean text page by page from PDF file."""
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        doc = fitz.open(pdf_path)
        extracted_pages = self.extract_pages_from_doc(doc)
        doc.close()
        return extracted_pages

    def validate_pdf_structure(self, doc: fitz.Document) -> tuple:
        """Validate whether the PDF strictly matches the expected OWASP Top 10 structure:
        1. Must contain topic headings matching TOPIC_HEADING_PATTERN (A01:.. to A10:.., case-insensitive).
        2. Must contain at least MIN_REQUIRED_CATEGORIES (default 3) distinct OWASP categories.
        3. Must contain the canonical subsections (What It Is, Why It Happens, How to Fix It, case-insensitive).
        """
        if len(doc) == 0:
            return False, "PDF document is empty. Please insert the correct OWASP Top 10 vulnerability notes PDF matching the required topic (A01-A10) and subsection format.", {}

        pages = self.extract_pages_from_doc(doc)
        full_text = "\n".join(p["text"] for p in pages)

        topic_matches = list(re.finditer(self.TOPIC_HEADING_PATTERN, full_text))
        if not topic_matches:
            return False, "No OWASP vulnerability topics (A01-A10) found. Please insert the correct OWASP Top 10 vulnerability notes PDF matching the required topic (A01-A10) and subsection format.", {}

        categories = set()
        for m in topic_matches:
            heading = m.group(1).strip()
            cat_code, _ = self._parse_heading(heading)
            if re.match(r'^A\d{2}$', cat_code, re.IGNORECASE):
                categories.add(cat_code.upper())

        if len(categories) < self.MIN_REQUIRED_CATEGORIES:
            return False, f"Document only contains {len(categories)} OWASP category ({', '.join(sorted(categories)) if categories else 'none'}). Expected at least {self.MIN_REQUIRED_CATEGORIES} topics (A01-A10). Please insert the correct OWASP Top 10 vulnerability notes PDF matching the required topic (A01-A10) and subsection format.", {}

        # Check for required subsections case-insensitively
        canonical_map = {s.lower(): s for s in self.REQUIRED_SUBSECTIONS}
        subsections_found_raw = set(re.findall(self.SUBSECTION_PATTERN, full_text))
        subsections_found_lower = {s.lower() for s in subsections_found_raw}
        required_lower = {s.lower() for s in self.REQUIRED_SUBSECTIONS}
        missing_lower = required_lower - subsections_found_lower
        if missing_lower:
            missing_display = [canonical_map.get(m, m) for m in missing_lower]
            return False, f"Document is missing required vulnerability subsections: {', '.join(sorted(missing_display))}. Please insert the correct OWASP Top 10 vulnerability notes PDF matching the required topic (A01-A10) and subsection format.", {}

        stats = {
            "total_pages": len(doc),
            "topics_found": len(topic_matches),
            "categories": sorted(list(categories)),
            "subsections": sorted([canonical_map.get(s, s.title()) for s in subsections_found_lower])
        }
        return True, "Valid OWASP Top 10 vulnerability notes structure", stats

    def validate_pdf_bytes(self, pdf_bytes: bytes) -> tuple:
        """Validate an uploaded PDF file from raw bytes in memory."""
        try:
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            is_valid, msg, stats = self.validate_pdf_structure(doc)
            doc.close()
            return is_valid, msg, stats
        except Exception as e:
            return False, f"Could not read PDF file: {e}. Please insert the correct OWASP Top 10 vulnerability notes PDF.", {}

    def process_bytes(self, pdf_bytes: bytes) -> List[Dict[str, Any]]:
        """Validate and chunk a PDF directly from memory bytes. Raises ValueError if validation fails."""
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        is_valid, msg, stats = self.validate_pdf_structure(doc)
        if not is_valid:
            doc.close()
            raise ValueError(msg)

        pages = self.extract_pages_from_doc(doc)
        doc.close()
        chunks = self.chunk_by_vulnerability(pages)
        return [asdict(c) for c in chunks]

    def chunk_by_vulnerability(self, pages: List[Dict[str, Any]]) -> List[VulnerabilityChunk]:
        """
        Merge sections into a single document chunk per vulnerability topic.
        """
        full_text = ""
        page_map = []
        offset = 0

        for p in pages:
            text = p["text"]
            start = offset
            end = offset + len(text)
            page_map.append((start, end, p["page_number"]))
            full_text += text + "\n"
            offset = end + 1

        topic_matches = list(re.finditer(self.TOPIC_HEADING_PATTERN, full_text))

        if not topic_matches:
            raise ValueError("No OWASP vulnerability topics (A01-A10) found. Please insert the correct OWASP Top 10 vulnerability notes PDF matching the required topic (A01-A10) and subsection format.")

        chunks: List[VulnerabilityChunk] = []
        chunk_id = 1

        # Process document header/overview if available
        if topic_matches[0].start() > 0:
            header_text = full_text[:topic_matches[0].start()].strip()
            if header_text:
                pages_covered = self._get_pages_for_range(0, topic_matches[0].start(), page_map)
                chunks.append(VulnerabilityChunk(
                    chunk_id=chunk_id,
                    category_code="HEADER",
                    vulnerability_title="Document Overview",
                    page_numbers=pages_covered,
                    content=header_text,
                    metadata={
                        "sections_included": ["Header & Reference"],
                        "char_count": len(header_text),
                        "word_count": len(header_text.split())
                    }
                ))
                chunk_id += 1

        # Process each vulnerability topic into 1 merged chunk
        for i in range(len(topic_matches)):
            start_pos = topic_matches[i].start()
            end_pos = topic_matches[i+1].start() if i + 1 < len(topic_matches) else len(full_text)
            
            raw_heading = topic_matches[i].group(1).strip()
            raw_content = full_text[start_pos:end_pos].strip()

            cat_code, vul_title = self._parse_heading(raw_heading)
            pages_covered = self._get_pages_for_range(start_pos, end_pos, page_map)

            # Detect sub-sections present in this vulnerability block (case-insensitive)
            canonical_map = {s.lower(): s for s in self.REQUIRED_SUBSECTIONS}
            raw_sections = re.findall(self.SUBSECTION_PATTERN, raw_content)
            sections_found = list(dict.fromkeys([canonical_map.get(s.lower(), s.title()) for s in raw_sections]))
            if not sections_found:
                sections_found = ["Overview"]

            chunks.append(VulnerabilityChunk(
                chunk_id=chunk_id,
                category_code=cat_code,
                vulnerability_title=vul_title,
                page_numbers=pages_covered,
                content=raw_content,
                metadata={
                    "sections_included": sections_found,
                    "char_count": len(raw_content),
                    "word_count": len(raw_content.split())
                }
            ))
            chunk_id += 1

        return chunks

    def process(self, pdf_path: str, output_json_path: Optional[str] = None) -> List[Dict[str, Any]]:
        """Run full extraction and chunking pipeline."""
        print(f"Extracting and parsing document structure from: {pdf_path}")
        pages = self.extract_pages(pdf_path)
        print(f"Read {len(pages)} pages.")

        chunks = self.chunk_by_vulnerability(pages)
        print(f"Generated {len(chunks)} vulnerability chunks (One Chunk Per Vulnerability).")

        chunks_dict = [asdict(c) for c in chunks]

        if output_json_path:
            with open(output_json_path, 'w', encoding='utf-8') as f:
                json.dump({
                    "source_pdf": os.path.basename(pdf_path),
                    "chunking_strategy": "one_chunk_per_vulnerability",
                    "total_pages": len(pages),
                    "total_chunks": len(chunks),
                    "chunks": chunks_dict
                }, f, indent=2, ensure_ascii=False)
            print(f"Structured chunks saved to: {output_json_path}")

        return chunks_dict

    @staticmethod
    def _parse_heading(raw_heading: str) -> tuple:
        """Extract category code (standardized to uppercase e.g. A01) and clean title."""
        parts = raw_heading.split(':', 1)
        code = parts[0].strip().upper()
        title = parts[1].strip() if len(parts) > 1 else raw_heading
        title = re.sub(r'^\d{4}-', '', title).rstrip(':').strip()
        return code, title

    @staticmethod
    def _get_pages_for_range(start_idx: int, end_idx: int, page_map: List[tuple]) -> List[int]:
        """Map text character indices to page numbers."""
        pages = set()
        for p_start, p_end, p_num in page_map:
            if max(start_idx, p_start) < min(end_idx, p_end):
                pages.add(p_num)
        return sorted(list(pages)) if pages else [1]

    @staticmethod
    def _clean_text(text: str) -> str:
        """Sanitize text formatting."""
        text = text.replace('\u200b', '')
        text = re.sub(r'[ \t]+', ' ', text)
        text = re.sub(r'\n{3,}', '\n\n', text)
        return text.strip()

    def _fallback_chunking(self, full_text: str, page_map: List[tuple]) -> List[VulnerabilityChunk]:
        """Fallback for unstructured documents."""
        paragraphs = [p.strip() for p in full_text.split('\n\n') if p.strip()]
        chunks = []
        for idx, para in enumerate(paragraphs):
            chunks.append(VulnerabilityChunk(
                chunk_id=idx + 1,
                category_code=f"SEC_{idx+1}",
                vulnerability_title="Document Section",
                page_numbers=[1],
                content=para,
                metadata={"sections_included": ["Paragraph"], "char_count": len(para), "word_count": len(para.split())}
            ))
        return chunks


if __name__ == "__main__":
    pdf_file = os.path.join(os.path.dirname(__file__), "Copy of OWASP Top 10 – Vulnerability Notes_easy.pdf")
    output_file = os.path.join(os.path.dirname(__file__), "structured_chunks.json")

    if os.path.exists(pdf_file):
        ingestor = PDFVulnerabilityIngestor()
        chunks = ingestor.process(pdf_file, output_json_path=output_file)
        
        half_count = (len(chunks) + 1) // 2
        print(f"\n==================================================")
        print(f" DISPLAYING FIRST HALF OF CHUNKS (1 to {half_count} of {len(chunks)})")
        print(f"==================================================")
        
        for idx in range(half_count):
            c = chunks[idx]
            print(f"\n--- Chunk {c['chunk_id']} [{c['category_code']}: {c['vulnerability_title']}] ---")
            print(json.dumps(c, indent=2, ensure_ascii=True))
    else:
        print(f"PDF not found at: {pdf_file}")
