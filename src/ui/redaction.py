from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Union

# Compiled regex patterns for redaction
_PATTERNS = [
    re.compile(r"gsk_[a-zA-Z0-9]{20,}", re.IGNORECASE),
    re.compile(r"sk-[a-zA-Z0-9]{20,}", re.IGNORECASE),
    re.compile(r"sk-ant-[a-zA-Z0-9_\-]{20,}", re.IGNORECASE),
    re.compile(r"Bearer\s+[a-zA-Z0-9_\-\.]{15,}", re.IGNORECASE),
    re.compile(r"(api[_-]?key|authorization|auth[_-]?token|password|secret)[\"']?\s*[:=]\s*[\"']?([^\"'\s,;]+)", re.IGNORECASE),
]


def redact_sensitive_text(text: Optional[str], custom_keys: Optional[List[str]] = None) -> str:
    """Scrub known API keys, authorization tokens, and credentials from string."""
    if not text:
        return ""
    
    redacted = str(text)
    
    # Redact any explicitly supplied user keys
    if custom_keys:
        for k in custom_keys:
            if k and len(k.strip()) >= 6:
                redacted = redacted.replace(k.strip(), "[REDACTED_API_KEY]")
    
    for pat in _PATTERNS:
        # For the key-value pattern, keep the key name and redact the value
        if "api[_-]?key" in pat.pattern:
            redacted = pat.sub(r"\1: [REDACTED_CREDENTIAL]", redacted)
        else:
            redacted = pat.sub("[REDACTED_SECRET]", redacted)
            
    return redacted


def redact_data_structure(data: Any, custom_keys: Optional[List[str]] = None) -> Any:
    """Recursively redact sensitive tokens from dictionaries, lists, and primitives."""
    if isinstance(data, str):
        return redact_sensitive_text(data, custom_keys=custom_keys)
    elif isinstance(data, dict):
        cleaned: Dict[str, Any] = {}
        for k, v in data.items():
            lower_k = str(k).lower()
            if any(term in lower_k for term in ["key", "auth", "secret", "token", "password", "credential"]):
                cleaned[k] = "[REDACTED_CREDENTIAL]"
            else:
                cleaned[k] = redact_data_structure(v, custom_keys=custom_keys)
        return cleaned
    elif isinstance(data, list):
        return [redact_data_structure(item, custom_keys=custom_keys) for item in data]
    return data
