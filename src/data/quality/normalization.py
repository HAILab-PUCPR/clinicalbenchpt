"""
Text normalization for deterministic comparison and duplicate detection.
Follows Section 3.2 of the ClinicalBenchPT Curation Specification.
"""

import hashlib
import re
import unicodedata
from typing import Any


def normalize_text(text: Any) -> str:
    """
    Apply deterministic text normalization for comparison:
    1. Unicode NFKC normalization
    2. Lowercase with casefold()
    3. Collapse consecutive whitespace (including line breaks)
    4. Strip leading and trailing whitespace
    
    Preserves accents, numbers and punctuation.
    """
    if text is None:
        return ""
    text_str = str(text)
    normalized = unicodedata.normalize("NFKC", text_str)
    lowercased = normalized.casefold()
    collapsed = re.sub(r"\s+", " ", lowercased)
    return collapsed.strip()


def compute_sha256(text: str) -> str:
    """Compute SHA-256 hash of normalized text."""
    normalized = normalize_text(text)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()
