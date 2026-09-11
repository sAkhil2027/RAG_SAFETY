import textwrap
from typing import List, Dict


def build_llm_prompt(vulns: List[Dict], question: str) -> str:
    """Create a single LLM prompt containing all vulnerability details and
    their associated OWASP guidance.

    ``vulns`` – list of enriched vulnerability dicts (must include ``owasp_guidance``).
    ``question`` – the original user question to be appended at the end of the prompt.
    """
    prompt_parts = []
    for idx, item in enumerate(vulns, 1):
        part = textwrap.dedent(f"""
            --- Vulnerability {idx} ---
            Title: {item.get('title', '')}
            CVE: {item.get('cve_id', '')}
            Severity: {item.get('severity', '')}
            Guidance: {item.get('owasp_guidance', '')}
        """).strip()
        prompt_parts.append(part)
    joined = "\n".join(prompt_parts)
    return (
        "You are a security assistant. Using only the provided guidance, answer the following question.\n"
        + joined
        + f"\nQuestion: {question}"
    )
