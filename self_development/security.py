import logging
import re
from pathlib import Path
from typing import Any

from models.security import redact_secrets

logger = logging.getLogger("VIDURA.self_development.security")

# Security-critical files and directories that guard permissions and core invariants
SECURITY_CRITICAL_PATTERNS = [
    "permissions/manager.py",
    "permissions/",
    "developer/applier.py",
    "developer/verifier.py",
    "models/security.py",
    "models/usage.py",
    "self_development/security.py",
    "self_development/loop.py",
    "config.py",
]

# Security-critical class and symbol names
SECURITY_CRITICAL_SYMBOLS = [
    "PermissionManager",
    "CodeChangeApplier",
    "CodeChangeVerifier",
    "SelfDevelopmentLoop",
    "grant_write_permission",
    "revoke_write_permission",
    "_validate_workspace_path",
    "_is_protected_target",
    "check_sensitive_data",
    "CloudUsageTracker",
]

UNSUPPORTED_OPERATIONS = {
    "delete", "delete_file", "remove", "destroy", "unlink",
    "rename", "rename_file", "move", "move_file",
    "execute", "shell", "run_shell", "command", "bash", "exec",
}


class SelfDevelopmentSecurityError(ValueError):
    """Base exception for self-development security violations."""
    pass


class SelfDevelopmentDisabledError(SelfDevelopmentSecurityError):
    """Raised when self-development is invoked while disabled via configuration."""
    pass


class SelfDevelopmentScopeError(SelfDevelopmentSecurityError):
    """Raised when a proposal attempts to expand scope beyond the approved plan."""
    pass


class SelfDevelopmentRecursionError(SelfDevelopmentSecurityError):
    """Raised when recursive self-development execution is detected."""
    pass


class ElevatedAuthorizationRequiredError(SelfDevelopmentSecurityError):
    """Raised when modifying security-critical infrastructure without elevated authorization."""
    pass


def is_security_critical_target(path_or_symbol: str) -> bool:
    """Detects whether a file path or symbol name touches security-critical infrastructure."""
    if not path_or_symbol:
        return False
    target_str = str(path_or_symbol).strip()
    target_lower = target_str.lower()

    # Check file patterns
    for pat in SECURITY_CRITICAL_PATTERNS:
        if pat in target_lower or target_lower.endswith(pat):
            return True

    # Check sensitive extensions
    parts = Path(target_str).parts
    for part in parts:
        if part.startswith(".git") or part == ".env":
            return True

    if target_lower.endswith((".key", ".pem", ".secret", ".token", ".crt")):
        return True

    # Check symbol names
    for sym in SECURITY_CRITICAL_SYMBOLS:
        if sym.lower() == target_lower or sym == target_str:
            return True

    return False


def validate_scope(allowed_files: Any, target_file: Any = None) -> tuple[bool, str]:
    """Ensures that a generated proposal targets strictly within the plan's authorized scope."""
    # Handle inverted argument order gracefully if caller passed (target_file, allowed_files)
    if isinstance(allowed_files, (str, Path)) and isinstance(target_file, (list, tuple, set)):
        allowed_files, target_file = list(target_file), allowed_files

    if not isinstance(allowed_files, (list, tuple, set)):
        allowed_files = [str(allowed_files)] if allowed_files else []

    if target_file is None:
        return False, "Target file cannot be empty."

    # If target_file is a list/tuple/set, check all items
    if isinstance(target_file, (list, tuple, set)):
        for tf in target_file:
            ok, err = validate_scope(allowed_files, tf)
            if not ok:
                return False, err
        return True, ""

    target_norm = str(Path(target_file)).strip()
    if not target_norm:
        return False, "Target file cannot be empty."

    if not allowed_files:
        return False, f"Scope expansion rejected: No allowed files authorized in plan; cannot target '{target_file}'."

    allowed_norms = [str(Path(f)).strip() for f in allowed_files if f]

    if target_norm in allowed_norms or any(target_norm.endswith(a) or a.endswith(target_norm) for a in allowed_norms):
        return True, ""

    return False, (
        f"Scope expansion rejected: Target file '{target_file}' is not permitted by approved scope "
        f"(authorized files: {allowed_files})."
    )


def validate_operation(operation: str) -> tuple[bool, str]:
    """Validates that requested operation conforms to supported create/modify semantics."""
    if not operation:
        return False, "Operation cannot be empty."

    op_norm = str(operation).lower().strip()

    if op_norm in ("shell", "command", "exec", "run_shell", "bash"):
        return False, f"UNSUPPORTED_OPERATION: Self-development does not support shell execution (requested: '{operation}')."

    if op_norm in UNSUPPORTED_OPERATIONS or "delete" in op_norm or "remove" in op_norm:
        return False, f"UNSUPPORTED_OPERATION: Self-development does not support deletion (requested: '{operation}')."

    if op_norm not in ("create_file", "modify_file"):
        return False, f"UNSUPPORTED_OPERATION: Operation '{operation}' is not supported. Only create_file and modify_file are permitted."

    return True, ""


def sanitize_experience_record(
    task: str,
    action_summary: str = "",
    result: str = "",
    lesson: str = "",
) -> dict[str, str]:
    """Sanitizes text stored in ExperienceRecord by removing secrets, credentials, and raw CoT."""
    # Redact known credential patterns
    safe_task = redact_secrets(task or "")
    safe_action = redact_secrets(action_summary or "")
    safe_result = redact_secrets(result or "")
    safe_lesson = redact_secrets(lesson or "")

    # Strip chain-of-thought indicators
    for cot_pat in [r"(?i)<thought>[\s\S]*?</thought>", r"(?i)```thought[\s\S]*?```"]:
        safe_action = re.sub(cot_pat, "[REDACTED_COT]", safe_action)
        safe_result = re.sub(cot_pat, "[REDACTED_COT]", safe_result)
        safe_lesson = re.sub(cot_pat, "[REDACTED_COT]", safe_lesson)

    # Enforce concise length limits so memory describes what happened without source dumps
    return {
        "task": safe_task[:500],
        "action_summary": safe_action[:1000],
        "result": safe_result[:1000],
        "lesson": safe_lesson[:500],
    }
