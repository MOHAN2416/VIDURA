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
    GENERATION_FAILED = "generation_failed"


class ApplicationStatus(str, Enum):
    """Status codes for post-change application and verification results."""
    PERMISSION_DENIED = "permission_denied"
    APPLICATION_FAILED = "application_failed"
    STALE_PROPOSAL = "stale_proposal"
    VERIFICATION_FAILED = "verification_failed"
    APPLIED_AND_VERIFIED = "applied_and_verified"


class ExecutionStage(str, Enum):
    """Lifecycle stages for developer execution pipeline."""
    GENERATED = "generated"
    PENDING_PERMISSION = "pending_permission"
    APPROVED = "approved"
    APPLYING = "applying"
    APPLIED = "applied"
    VERIFYING = "verifying"
    VERIFIED = "verified"
    SUCCESS = "success"
    # Failure stages
    DENIED = "denied"
    PROPOSAL_INVALID = "proposal_invalid"
    APPLICATION_FAILED = "application_failed"
    STALE = "stale"
    VERIFICATION_FAILED = "verification_failed"
    FAILED = "failed"


class ExecutionStatus(str, Enum):
    """Status codes for developer execution results."""
    SUCCESS = "success"
    PENDING_PERMISSION = "pending_permission"
    PERMISSION_DENIED = "permission_denied"
    PROPOSAL_INVALID = "proposal_invalid"
    APPLICATION_FAILED = "application_failed"
    VERIFICATION_FAILED = "verification_failed"
    STALE_PROPOSAL = "stale_proposal"
    UNEXPECTED_ERROR = "unexpected_error"


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
    proposal_id: str | None = None
    provider: str | None = None
    model: str | None = None

    def __post_init__(self) -> None:
        if not self.proposal_id:
            import uuid
            self.proposal_id = f"prop_{uuid.uuid4().hex[:8]}"

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
            "proposal_id": self.proposal_id,
            "provider": self.provider,
            "model": self.model,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CodeChangeProposal":
        """Safely deserializes a dictionary into a CodeChangeProposal."""
        if not isinstance(data, dict):
            return cls(operation="", target_file="", is_valid=False)
        return cls(
            operation=str(data.get("operation", "")),
            target_file=str(data.get("target_file", "")),
            target_symbol=data.get("target_symbol"),
            description=str(data.get("description", "")),
            proposed_content=str(data.get("proposed_content", "")),
            original_content=str(data.get("original_content", "")),
            rationale=str(data.get("rationale", "")),
            status=str(data.get("status", ProposalStatus.PROPOSED.value)),
            is_valid=bool(data.get("is_valid", True)),
            validation_error=data.get("validation_error"),
            proposal_id=data.get("proposal_id"),
            provider=data.get("provider"),
            model=data.get("model"),
        )


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


@dataclass
class DeveloperGenerationResult:
    """Structured result of developer code generation in Phase 8.3."""
    target_file: str
    operation: str
    original_content: str = ""
    proposed_content: str = ""
    generated_code: str = ""
    explanation: str = ""
    affected_symbols: list[str] = field(default_factory=list)
    assumptions: list[str] = field(default_factory=list)
    validation_status: str = "valid"
    is_valid: bool = True
    requires_plan_update: bool = False
    proposal: CodeChangeProposal | None = None
    errors: list[str] = field(default_factory=list)
    provider: str | None = None
    model: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Returns a JSON-serializable dictionary representation of generation result."""
        return {
            "target_file": self.target_file,
            "operation": self.operation,
            "original_content": self.original_content,
            "proposed_content": self.proposed_content,
            "generated_code": self.generated_code,
            "explanation": self.explanation,
            "affected_symbols": list(self.affected_symbols),
            "assumptions": list(self.assumptions),
            "validation_status": self.validation_status,
            "is_valid": self.is_valid,
            "requires_plan_update": self.requires_plan_update,
            "proposal": self.proposal.to_dict() if self.proposal else None,
            "errors": list(self.errors),
            "provider": self.provider,
            "model": self.model,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DeveloperGenerationResult":
        """Safely parses and validates dictionary into DeveloperGenerationResult."""
        if not isinstance(data, dict):
            return cls(
                target_file="",
                operation="",
                is_valid=False,
                validation_status="invalid",
                errors=["Malformed data input."],
            )

        raw_proposal = data.get("proposal")
        proposal_obj: CodeChangeProposal | None = None
        if isinstance(raw_proposal, CodeChangeProposal):
            proposal_obj = raw_proposal
        elif isinstance(raw_proposal, dict):
            proposal_obj = CodeChangeProposal(
                operation=str(raw_proposal.get("operation", "")),
                target_file=str(raw_proposal.get("target_file", "")),
                target_symbol=raw_proposal.get("target_symbol"),
                description=str(raw_proposal.get("description", "")),
                proposed_content=str(raw_proposal.get("proposed_content", "")),
                original_content=str(raw_proposal.get("original_content", "")),
                rationale=str(raw_proposal.get("rationale", "")),
                status=str(raw_proposal.get("status", ProposalStatus.PROPOSED.value)),
                is_valid=bool(raw_proposal.get("is_valid", True)),
                validation_error=raw_proposal.get("validation_error"),
                proposal_id=raw_proposal.get("proposal_id"),
                provider=raw_proposal.get("provider"),
                model=raw_proposal.get("model"),
            )

        prov = data.get("provider") or (proposal_obj.provider if proposal_obj else None)
        mod = data.get("model") or (proposal_obj.model if proposal_obj else None)

        return cls(
            target_file=str(data.get("target_file", "")),
            operation=str(data.get("operation", "")),
            original_content=str(data.get("original_content", "")),
            proposed_content=str(data.get("proposed_content", "")),
            generated_code=str(data.get("generated_code", "")),
            explanation=str(data.get("explanation", "")),
            affected_symbols=[str(s) for s in data.get("affected_symbols", []) if s],
            assumptions=[str(a) for a in data.get("assumptions", []) if a],
            validation_status=str(data.get("validation_status", "valid")),
            is_valid=bool(data.get("is_valid", True)),
            requires_plan_update=bool(data.get("requires_plan_update", False)),
            proposal=proposal_obj,
            errors=[str(e) for e in data.get("errors", []) if e],
            provider=prov,
            model=mod,
        )


@dataclass
class DeveloperExecutionResult:
    """Structured result of developer execution in Phase 8.4 and Phase 8.6."""
    success: bool
    status: str  # from ExecutionStatus
    stage: str   # from ExecutionStage
    target_file: str
    operation: str
    proposal_id: str | None = None
    proposal: CodeChangeProposal | None = None
    application_result: CodeChangeResult | None = None
    verification_result: VerificationResult | None = None
    permission_granted: bool = False
    test_plan: Any = None
    test_result: Any = None
    overall_outcome: str | None = None
    requires_plan_update: bool = False
    summary: str = ""
    error: str | None = None
    provider: str | None = None
    model: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Returns JSON-serializable dictionary representation."""
        return {
            "success": self.success,
            "status": self.status,
            "stage": self.stage,
            "target_file": self.target_file,
            "operation": self.operation,
            "proposal_id": self.proposal_id,
            "proposal": self.proposal.to_dict() if self.proposal else None,
            "application_result": self.application_result.to_dict() if self.application_result else None,
            "verification_result": self.verification_result.to_dict() if self.verification_result else None,
            "permission_granted": self.permission_granted,
            "test_plan": self.test_plan.to_dict() if self.test_plan and hasattr(self.test_plan, "to_dict") else None,
            "test_result": self.test_result.to_dict() if self.test_result and hasattr(self.test_result, "to_dict") else None,
            "overall_outcome": self.overall_outcome,
            "requires_plan_update": self.requires_plan_update,
            "summary": self.summary,
            "error": self.error,
            "provider": self.provider,
            "model": self.model,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DeveloperExecutionResult":
        """Safely parses and validates dictionary into DeveloperExecutionResult."""
        if not isinstance(data, dict):
            return cls(
                success=False,
                status=ExecutionStatus.UNEXPECTED_ERROR.value,
                stage=ExecutionStage.FAILED.value,
                target_file="",
                operation="",
                summary="Malformed execution result data.",
                error="Malformed data input.",
            )

        raw_prop = data.get("proposal")
        prop_obj: CodeChangeProposal | None = None
        if isinstance(raw_prop, CodeChangeProposal):
            prop_obj = raw_prop
        elif isinstance(raw_prop, dict):
            prop_obj = CodeChangeProposal(
                operation=str(raw_prop.get("operation", "")),
                target_file=str(raw_prop.get("target_file", "")),
                target_symbol=raw_prop.get("target_symbol"),
                description=str(raw_prop.get("description", "")),
                proposed_content=str(raw_prop.get("proposed_content", "")),
                original_content=str(raw_prop.get("original_content", "")),
                rationale=str(raw_prop.get("rationale", "")),
                status=str(raw_prop.get("status", ProposalStatus.PROPOSED.value)),
                is_valid=bool(raw_prop.get("is_valid", True)),
                validation_error=raw_prop.get("validation_error"),
                proposal_id=raw_prop.get("proposal_id"),
            )

        raw_app = data.get("application_result")
        app_obj: CodeChangeResult | None = None
        if isinstance(raw_app, CodeChangeResult):
            app_obj = raw_app
        elif isinstance(raw_app, dict):
            app_obj = CodeChangeResult(
                success=bool(raw_app.get("success", False)),
                target_file=str(raw_app.get("target_file", "")),
                operation=str(raw_app.get("operation", "")),
                permission_granted=bool(raw_app.get("permission_granted", False)),
                reason=str(raw_app.get("reason", "")),
                verification_success=bool(raw_app.get("verification_success", False)),
                status_code=str(raw_app.get("status_code", ApplicationStatus.APPLICATION_FAILED.value)),
                error=raw_app.get("error"),
                verification_details=raw_app.get("verification_details"),
            )

        raw_ver = data.get("verification_result")
        ver_obj: VerificationResult | None = None
        if isinstance(raw_ver, VerificationResult):
            ver_obj = raw_ver
        elif isinstance(raw_ver, dict):
            ver_obj = VerificationResult(
                success=bool(raw_ver.get("success", False)),
                target_file=str(raw_ver.get("target_file", "")),
                operation=str(raw_ver.get("operation", "")),
                verified=bool(raw_ver.get("verified", False)),
                expected_state=str(raw_ver.get("expected_state", "")),
                actual_state=str(raw_ver.get("actual_state", "")),
                reason=str(raw_ver.get("reason", "")),
                error=raw_ver.get("error"),
            )

        raw_plan = data.get("test_plan")
        plan_obj = None
        if raw_plan:
            if isinstance(raw_plan, dict):
                from developer.test_models import TestPlan
                plan_obj = TestPlan.from_dict(raw_plan)
            else:
                plan_obj = raw_plan

        raw_res = data.get("test_result")
        res_obj = None
        if raw_res:
            if isinstance(raw_res, dict):
                from developer.test_models import TestResult
                res_obj = TestResult.from_dict(raw_res)
            else:
                res_obj = raw_res

        return cls(
            success=bool(data.get("success", False)),
            status=str(data.get("status", ExecutionStatus.UNEXPECTED_ERROR.value)),
            stage=str(data.get("stage", ExecutionStage.FAILED.value)),
            target_file=str(data.get("target_file", "")),
            operation=str(data.get("operation", "")),
            proposal_id=data.get("proposal_id"),
            proposal=prop_obj,
            application_result=app_obj,
            verification_result=ver_obj,
            permission_granted=bool(data.get("permission_granted", False)),
            test_plan=plan_obj,
            test_result=res_obj,
            overall_outcome=data.get("overall_outcome"),
            requires_plan_update=bool(data.get("requires_plan_update", False)),
            summary=str(data.get("summary", "")),
            error=data.get("error"),
            provider=data.get("provider", prop_obj.provider if prop_obj else None),
            model=data.get("model", prop_obj.model if prop_obj else None),
        )


