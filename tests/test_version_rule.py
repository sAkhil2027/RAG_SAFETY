import os
import pytest
from sanitizer import ConfigurableSanitizer

def test_version_not_redacted():
    # Ensure version strings are not altered by the sanitizer
    sanitizer = ConfigurableSanitizer(config_path='rules.json')
    version_str = "v10.0.12.100"
    assert sanitizer.pseudonymize(version_str) == version_str
    version_str2 = "2.4.1.0"
    assert sanitizer.pseudonymize(version_str2) == version_str2
