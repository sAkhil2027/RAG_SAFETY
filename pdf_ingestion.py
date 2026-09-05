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

    TOPIC_HEADING_PATTERN = r'(A\d{2}:[^\n]+)'
    SUBSECTION_PATTERN = r'(What It Is|Why It Happens|How to Fix It)'

    def extract_pages(self, pdf_path: str) -> List[Dict[str, Any]]:
        """Extract clean text page by page from PDF file."""
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        extracted_pages = []
        doc = fitz.open(pdf_path)

        for page_idx in range(len(doc)):
            page = doc[page_idx]
            raw_text = page.get_text()
            cleaned_text = self._clean_text(raw_text)
            
            if cleaned_text:
                extracted_pages.append({
                    "page_number": page_idx + 1,
                    "text": cleaned_text
                })

        doc.close()
        return extracted_pages

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
            return self._fallback_chunking(full_text, page_map)

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

            # Detect sub-sections present in this vulnerability block
            sections_found = list(dict.fromkeys(re.findall(self.SUBSECTION_PATTERN, raw_content)))
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
        """Extract category code and clean title."""
        parts = raw_heading.split(':', 1)
        code = parts[0].strip()
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
