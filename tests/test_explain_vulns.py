import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch

from app import app

client = TestClient(app)

# Mock vulnerability payload
MOCK_VULN = {
    "title": "Sample Vulnerability",
    "cve_id": "CVE-2023-0001",
    "severity": "high",
    "description": "Test description"
}

def mock_match_owasp_guidance(vuln):
    return "Sample OWASP guidance text."

@pytest.fixture(autouse=True)
def mock_dependencies():
    with patch('retrieval_engine.search_vulnerabilities', return_value=[MOCK_VULN]):
        with patch('retrieval_engine.match_owasp_guidance', side_effect=mock_match_owasp_guidance):
            yield

def test_explain_vulns_streaming():
    response = client.post(
        "/explain_vulns",
        json={"session_id": "test", "question": "How to fix XSS?", "top_k": 5, "candidate_k": 10},
        headers={"accept": "text/event-stream"},
        timeout=10
    )
    assert response.status_code == 200
    content = response.text
    assert "How to fix XSS?" in content
    assert "Sample OWASP guidance" in content
