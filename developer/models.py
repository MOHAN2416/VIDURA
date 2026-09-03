from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any


class ProposalOperation(str, Enum):
    """Supported operations for code change proposals."""
    CREATE_FILE = "create_file"
    MODIFY_FILE = "modify_file"


class ProposalStatus(str, Enum):
    """Status codes for code change proposals."""
    PROPOSED = "proposed"
    INVALID_PATH = "invalid_path"
    TARGET_NOT_FOUND = "target_not_found"
    PERMISSION_DENIED = "permission_denied"
    UNSUPPORTED_OPERATION = "unsupported_operation"


class ApplicationStatus(str, Enum):
    """Status codes for post-change application and verification results."""
    PERMISSION_DENIED = "permission_denied"
    APPLICATION_FAILED = "application_failed"
    STALE_PROPOSAL = "stale_proposal"
    VERIFICATION_FAILED = "verification_failed"
    APPLIED_AND_VERIFIED = "applied_and_verified"


@dataclass
class CodeChangeProposal:
    """Structured representation of a proposed code change.
    
    NOTE: This is strictly a proposal representation.
    Generating or serializing a proposal does NOT modify any file on disk.
    """
    operation: str  # "create_file" or "modify_file"
    target_file: str
    target_symbol: str | None = None
    description: str = ""
    proposed_content: str = ""
    original_content: str = ""
    rationale: str = ""
    status: str = ProposalStatus.PROPOSED.value
    is_valid: bool = True
    validation_error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Returns a JSON-serializable dictionary representation of the proposal."""
        return {
            "operation": self.operation,
            "target_file": self.target_file,
            "target_symbol": self.target_symbol,
            "description": self.description,
            "proposed_content": self.proposed_content,
            "original_content": self.original_content,
            "rationale": self.rationale,
            "status": self.status,
            "is_valid": self.is_valid,
            "validation_error": self.validation_error,
        }


@dataclass
class VerificationResult:
    """Structured representation of post-change verification results."""
    success: bool
    target_file: str
    operation: str
    verified: bool
    expected_state: str
    actual_state: str
    reason: str
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Returns a JSON-serializable dictionary representation of verification result."""
        return {
            "success": self.success,
            "target_file": self.target_file,
            "operation": self.operation,
            "verified": self.verified,
            "expected_state": self.expected_state,
            "actual_state": self.actual_state,
            "reason": self.reason,
            "error": self.error,
        }


@dataclass
class CodeChangeResult:
    """Structured representation of the outcome of applying a CodeChangeProposal on disk."""
    success: bool
    target_file: str
    operation: str
    permission_granted: bool
    reason: str
    verification_success: bool
    status_code: str = ApplicationStatus.APPLIED_AND_VERIFIED.value
    error: str | None = None
    verification_details: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Returns a JSON-serializable dictionary representation of the change result."""
        return {
            "success": self.success,
            "target_file": self.target_file,
            "operation": self.operation,
            "permission_granted": self.permission_granted,
            "reason": self.reason,
            "verification_success": self.verification_success,
            "status_code": self.status_code,
            "error": self.error,
            "verification_details": self.verification_details,
        }
