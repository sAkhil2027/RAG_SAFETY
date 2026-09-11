import re
import json
from jsonschema import validate, ValidationError

JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "vulnerabilities": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "cve_id": {"type": "string"},
                    "severity": {"type": "string"},
                    "explanation": {"type": "string"},
                    "fix": {"type": "string"}
                },
                "required": ["title", "cve_id", "severity", "explanation", "fix"]
            }
        }
    },
    "required": ["vulnerabilities"]
}

def extract_json(raw: str) -> dict:
    """Extract the first JSON object from a raw LLM response string.
    Looks for the outermost curly braces.
    """
    start = raw.find('{')
    end = raw.rfind('}')
    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON object found in LLM response")
    json_str = raw[start:end+1]
    return json.loads(json_str)

def parse_and_validate(raw: str) -> dict:
    """Parse raw LLM response and validate against the expected schema.
    Raises ``ValueError`` or ``jsonschema.ValidationError`` on failure.
    """
    data = extract_json(raw)
    validate(instance=data, schema=JSON_SCHEMA)  # may raise ValidationError
    return data
