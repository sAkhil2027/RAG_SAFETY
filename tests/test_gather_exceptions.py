import pytest
from retrieval_engine import search_vulnerabilities, match_owasp_guidance
from unittest import mock

# Mock the guidance store to raise an exception on the second call
@mock.patch('retrieval_engine.guidance_store')
def test_owasp_lookup_exception(mock_guidance_store):
    # First call returns a valid guidance
    mock_guidance_store.search.side_effect = [
        [{'content': 'Guidance for vuln 1'}],  # first vulnerable payload guidance
        Exception('Timeout')  # second call raises exception
    ]
    # Prepare dummy vulnerability payloads
    vulns = [
        {'title': 'Vuln 1', 'cve_id': 'CVE-1111', 'severity': 'high'},
        {'title': 'Vuln 2', 'cve_id': 'CVE-2222', 'severity': 'medium'}
    ]
    enriched = []
    for v in vulns:
        try:
            guidance = match_owasp_guidance(v)
        except Exception:
            guidance = ''
        enriched.append({**v, 'owasp_guidance': guidance})
    assert enriched[0]['owasp_guidance'] == 'Guidance for vuln 1'
    assert enriched[1]['owasp_guidance'] == ''  # fallback on exception
