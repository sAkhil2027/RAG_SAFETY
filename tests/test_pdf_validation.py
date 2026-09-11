import os
import io
import pytest
import fitz  # PyMuPDF
from fastapi.testclient import TestClient
from pdf_ingestion import PDFVulnerabilityIngestor
from app import app

client = TestClient(app)

REAL_PDF_PATH = os.path.join(
    os.path.dirname(__file__), "..", "Copy of OWASP Top 10 – Vulnerability Notes_easy.pdf"
)


def create_dummy_pdf(text_content: str) -> bytes:
    """Create a minimal PDF in memory with specified text."""
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((50, 72), text_content)
    pdf_bytes = doc.write()
    doc.close()
    return pdf_bytes


def test_valid_pdf_structure():
    """Verify that the official OWASP Top 10 notes PDF passes validation."""
    assert os.path.exists(REAL_PDF_PATH), f"Sample PDF not found at {REAL_PDF_PATH}"
    ingestor = PDFVulnerabilityIngestor()
    with open(REAL_PDF_PATH, "rb") as f:
        pdf_bytes = f.read()

    is_valid, msg, stats = ingestor.validate_pdf_bytes(pdf_bytes)
    assert is_valid is True
    assert "Valid" in msg
    assert stats["topics_found"] >= 10
    assert "A01" in stats["categories"]
    assert "A05" in stats["categories"]


def test_random_pdf_rejection_no_topics():
    """Verify that a random PDF without OWASP headings is rejected."""
    random_text = (
        "John Doe Resume\n"
        "Software Engineer with 5 years experience in Python and cloud services.\n"
        "Education: BS Computer Science."
    )
    dummy_bytes = create_dummy_pdf(random_text)
    ingestor = PDFVulnerabilityIngestor()
    is_valid, msg, stats = ingestor.validate_pdf_bytes(dummy_bytes)

    assert is_valid is False
    assert "No OWASP vulnerability topics (A01-A10) found" in msg
    assert "Please insert the correct OWASP Top 10 vulnerability notes PDF" in msg


def test_random_pdf_insufficient_categories():
    """Verify that a PDF with only 1 topic is rejected for not meeting minimum coverage."""
    partial_text = (
        "A01: Broken Access Control\n"
        "What It Is\nSome explanation.\n"
        "Why It Happens\nSome reason.\n"
        "How to Fix It\nSome fix.\n"
    )
    dummy_bytes = create_dummy_pdf(partial_text)
    ingestor = PDFVulnerabilityIngestor()
    is_valid, msg, stats = ingestor.validate_pdf_bytes(dummy_bytes)

    assert is_valid is False
    assert "Expected at least 3 topics (A01-A10)" in msg
    assert "Please insert the correct OWASP Top 10 vulnerability notes PDF" in msg


def test_random_pdf_missing_subsections():
    """Verify that a PDF with headings but missing canonical subsections is rejected."""
    incomplete_text = (
        "A01: Broken Access Control\nJust some text without subsections.\n"
        "A02: Security Misconfiguration\nJust another text.\n"
        "A03: Supply Chain Risk\nYet another text."
    )
    dummy_bytes = create_dummy_pdf(incomplete_text)
    ingestor = PDFVulnerabilityIngestor()
    is_valid, msg, stats = ingestor.validate_pdf_bytes(dummy_bytes)

    assert is_valid is False
    assert "missing required vulnerability subsections" in msg
    assert "Please insert the correct OWASP Top 10 vulnerability notes PDF" in msg


def test_api_ingest_pdf_invalid():
    """Test API endpoint rejecting invalid PDF with HTTP 400."""
    dummy_bytes = create_dummy_pdf("Company Invoice #12345\nTotal Due: $500")
    response = client.post(
        "/api/ingest_pdf",
        files={"file": ("invoice.pdf", io.BytesIO(dummy_bytes), "application/pdf")}
    )
    assert response.status_code == 400
    data = response.json()
    assert "detail" in data
    assert "Please insert the correct OWASP Top 10 vulnerability notes PDF" in data["detail"]


def test_api_ingest_pdf_valid():
    """Test API endpoint successfully processing and indexing valid OWASP PDF."""
    with open(REAL_PDF_PATH, "rb") as f:
        pdf_bytes = f.read()

    response = client.post(
        "/api/ingest_pdf",
        files={"file": ("owasp_notes.pdf", io.BytesIO(pdf_bytes), "application/pdf")}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["indexed_chunks"] >= 10
    assert "Successfully ingested and indexed" in data["message"]
