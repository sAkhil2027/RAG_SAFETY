import json
import re
import os
from typing import List, Dict, Pattern


class ConfigurableSanitizer:
    """Load regex‑based sanitization rules from a JSON file and apply them.

    The rules file (default ``rules.json``) follows the format used in the
    repository, e.g.::

        {
            "rules": [
                {"pattern": "[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\\\.[a-zA-Z]{2,}",
                 "replacement": "[REDACTED_EMAIL]"},
                {"type": "whitelist", "pattern": "\\\\b\\\\d+(?:\\\\.\\\\d+){2,3}\\\\b",
                 "description": "Software version numbers – do not redact"}
            ]
        }

    Whitelist rules are applied first – any text matching a whitelist pattern is
    left untouched. All other rules perform a ``re.sub`` replacement on the
    remaining text.
    """

    def __init__(self, config_path: str = "rules.json"):
        self.config_path = os.path.abspath(config_path)
        self.rules: List[Dict] = []
        self._whitelist: List[Pattern] = []
        self._replacements: List[Dict] = []
        self._load_rules()

    # ---------------------------------------------------------------------
    # Rule handling
    # ---------------------------------------------------------------------
    def _load_rules(self) -> None:
        """Read ``self.config_path`` and compile regex objects.

        The method populates ``self._whitelist`` and ``self._replacements``.
        Whitelist entries are stored as compiled patterns; replacement entries
        keep the compiled pattern together with the replacement string.
        """
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Sanitizer config not found: {self.config_path}")
        with open(self.config_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.rules = data.get("rules", [])
        self._whitelist.clear()
        self._replacements.clear()
        for rule in self.rules:
            pattern = rule.get("pattern")
            if not pattern:
                continue
            compiled = re.compile(pattern, re.IGNORECASE)
            if rule.get("type") == "whitelist":
                self._whitelist.append(compiled)
            else:
                replacement = rule.get("replacement", "[REDACTED]")
                self._replacements.append({"pattern": compiled, "replacement": replacement})

    def reload_rules(self) -> None:
        """Public method to re‑load rules at runtime (used by the admin endpoint)."""
        self._load_rules()

    # ---------------------------------------------------------------------
    # Sanitization API
    # ---------------------------------------------------------------------
    def _is_whitelisted(self, span_start: int, span_end: int, text: str) -> bool:
        """Check whether the slice ``text[span_start:span_end]`` matches any whitelist.
        This helper avoids overlapping replacements when a whitelist pattern
        occurs inside a larger match.
        """
        segment = text[span_start:span_end]
        for wl in self._whitelist:
            if wl.fullmatch(segment):
                return True
        return False

    def pseudonymize(self, text: str) -> str:
        """Apply whitelist‑aware regex replacements to ``text``.

        The algorithm works in two passes:
        1. Identify all whitelist spans and remember them.
        2. Iterate over replacement rules; for each match, replace it only if
           the matched span does **not** intersect a whitelist span.
        """
        # 1. Collect whitelist spans (as ``(start, end)`` tuples)
        whitelist_spans: List[tuple] = []
        for wl in self._whitelist:
            for m in wl.finditer(text):
                whitelist_spans.append((m.start(), m.end()))

        # Helper to test overlap
        def overlaps(start: int, end: int) -> bool:
            for ws, we in whitelist_spans:
                if not (end <= ws or start >= we):  # ranges intersect
                    return True
            return False

        # 2. Apply each replacement rule respecting whitelist overlaps
        result = text
        for repl in self._replacements:
            pattern: Pattern = repl["pattern"]
            replacement: str = repl["replacement"]

            def _replacer(match: re.Match) -> str:
                s, e = match.start(), match.end()
                if overlaps(s, e):
                    return match.group(0)
                return replacement

            result = pattern.sub(_replacer, result)
        return result

    # ---------------------------------------------------------------------
    # Convenience shortcut used throughout the codebase
    # ---------------------------------------------------------------------
    def __call__(self, text: str) -> str:
        """Allow the instance to be called directly as ``sanitizer(text)``.
        This forwards to :meth:`pseudonymize`.
        """
        return self.pseudonymize(text)

# End of file
