"""VIDURA Cloud Security & Sensitive Data Policy.

Provides deterministic detection and redaction of sensitive credentials, private keys,
API tokens, and protected repository resources to prevent leakage to cloud providers.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

# Protected file extensions and names (aligned with Phase 7 security boundaries)
PROTECTED_PATTERNS = (".git", ".env", ".key", ".pem", ".secret")
SENSITIVE_FILENAME_SUBSTRINGS = ("id_rsa", "credentials", "password", "secret", "private_key", ".env")

# Regex patterns for deterministic secret detection
PRIVATE_KEY_PATTERN = re.compile(r"-----BEGIN [A-Z0-9_\- ]*PRIVATE KEY-----", re.IGNORECASE)
BEARER_TOKEN_PATTERN = re.compile(r"Bearer\s+[A-Za-z0-9_\-\.]{12,}", re.IGNORECASE)
API_KEY_ASSIGNMENT_PATTERN = re.compile(
    r"(?:api[_-]?key|auth[_-]?token|secret[_-]?key|access[_-]?token)\s*[:=]\s*['\"]?[A-Za-z0-9_\-\.]{16,}['\"]?",
    re.IGNORECASE,
)
WELL_KNOWN_KEY_PREFIXES = re.compile(r"\b(?:sk-[A-Za-z0-9]{20,}|ghp_[A-Za-z0-9]{20,}|ollama_[A-Za-z0-9]{16,})\b")


def is_protected_file_target(target_path: str | Path | list[Any] | tuple[Any, ...] | None) -> bool:
    """Checks if a target file path or collection of paths matches protected security patterns."""
    if not target_path:
        return False
    if isinstance(target_path, (list, tuple, set)):
        return any(is_protected_file_target(p) for p in target_path)
    path_str = str(target_path).lower().strip()
    try:
        parts = Path(target_path).parts
    except Exception:
        parts = ()
    for part in parts:
        part_lower = part.lower()
        if part_lower.startswith(".git") or part_lower == ".env":
            return True
    for pat in PROTECTED_PATTERNS:
        if path_str.endswith(pat) or f"/{pat}" in path_str or pat in parts:
            return True
    for sub in SENSITIVE_FILENAME_SUBSTRINGS:
        if sub in path_str:
            return True
    return False


def extract_text_content(item: Any) -> str:
    """Recursively extracts text from nested structures, dicts, and message lists."""
    if item is None:
        return ""
    if isinstance(item, str):
        return item
    if isinstance(item, dict):
        return " ".join(extract_text_content(v) for v in item.values() if v is not None)
    if isinstance(item, (list, tuple, set)):
        return " ".join(extract_text_content(sub) for sub in item if sub is not None)
    return str(item)


def check_sensitive_data(
    messages_or_text: Any,
    target_files: list[str] | str | None = None,
    api_key: str | None = None,
) -> tuple[bool, str]:
    """Deterministically checks if text, message history, or target files contain sensitive content.

    Returns:
        tuple of (is_sensitive: bool, reason: str)
    """
    # 1. Check target files
    if target_files:
        if is_protected_file_target(target_files):
            return True, f"Target file '{target_files}' is a protected credential or sensitive path."

    # 2. Extract full text
    full_text = extract_text_content(messages_or_text)
    if not full_text.strip():
        return False, ""

    # 3. Check for configured API key if provided
    if api_key and api_key.strip() and api_key.strip() in full_text:
        return True, "Request contains configured cloud API key."

    # 4. Check for private keys
    if PRIVATE_KEY_PATTERN.search(full_text):
        return True, "Request contains private key headers (BEGIN PRIVATE KEY)."

    # 5. Check for well-known API key patterns and bearer tokens
    if WELL_KNOWN_KEY_PREFIXES.search(full_text):
        return True, "Request contains known API key token pattern (e.g. sk-..., ghp-..., ollama_...)."

    if BEARER_TOKEN_PATTERN.search(full_text):
        return True, "Request contains authorization Bearer token."

    if API_KEY_ASSIGNMENT_PATTERN.search(full_text):
        return True, "Request contains assigned API key or secret token credentials."

    return False, ""


def contains_sensitive_data(
    messages_or_text: Any,
    target_files: list[str] | str | None = None,
    api_key: str | None = None,
) -> bool:
    """Returns True if messages, text, or target files contain sensitive credentials or protected paths."""
    is_sensitive, _ = check_sensitive_data(messages_or_text, target_files=target_files, api_key=api_key)
    return is_sensitive


def redact_secrets(text: str, api_key: str | None = None) -> str:
    """Redacts private keys, authorization headers, bearer tokens, and API keys from text."""
    if not text:
        return ""
    sanitized = str(text)

    # Redact configured API key if known
    if api_key and api_key.strip():
        sanitized = sanitized.replace(api_key.strip(), "[REDACTED_API_KEY]")

    # Redact full private key blocks
    sanitized = re.sub(
        r"-----BEGIN [A-Z0-9_\- ]*PRIVATE KEY-----[\s\S]*?-----END [A-Z0-9_\- ]*PRIVATE KEY-----",
        "[REDACTED_PRIVATE_KEY]",
        sanitized,
        flags=re.IGNORECASE,
    )

    # Redact standalone private key headers
    sanitized = PRIVATE_KEY_PATTERN.sub("[REDACTED_PRIVATE_KEY_HEADER]", sanitized)

    # Redact Bearer tokens
    sanitized = re.sub(
        r"Bearer\s+[A-Za-z0-9_\-\.]{8,}",
        "[REDACTED_BEARER_TOKEN]",
        sanitized,
        flags=re.IGNORECASE,
    )


    # Redact well-known key prefixes
    sanitized = WELL_KNOWN_KEY_PREFIXES.sub("[REDACTED_KEY]", sanitized)

    # Redact key assignments
    sanitized = re.sub(
        r"((?:api[_-]?key|auth[_-]?token|secret[_-]?key|access[_-]?token|password)\s*[:=]\s*['\"]?)[A-Za-z0-9_\-\.]{8,}(['\"]?)",
        r"\1[REDACTED]\2",
        sanitized,
        flags=re.IGNORECASE,
    )

    return sanitized
