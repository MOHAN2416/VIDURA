import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from developer.models import CodeChangeProposal, DeveloperExecutionResult


@dataclass
class SelfDevelopmentGoal:
    """Strongly typed model representing an explicit self-development goal."""

    description: str
    goal_id: str = field(default_factory=lambda: f"sd_goal_{uuid.uuid4().hex[:8]}")
    motivation: str = ""
    scope: list[str] = field(default_factory=list)
    constraints: list[str] = field(default_factory=list)
    requested_by: str = "user"
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    status: str = "created"

    def __post_init__(self) -> None:
        if not self.description or not self.description.strip():
            raise ValueError("SelfDevelopmentGoal description cannot be empty.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "goal_id": self.goal_id,
            "description": self.description,
            "motivation": self.motivation,
            "scope": list(self.scope),
            "constraints": list(self.constraints),
            "requested_by": self.requested_by,
            "created_at": self.created_at,
            "status": self.status,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SelfDevelopmentGoal":
        if not isinstance(data, dict):
            raise ValueError("Expected dictionary for SelfDevelopmentGoal")
        return cls(
            goal_id=str(data.get("goal_id") or f"sd_goal_{uuid.uuid4().hex[:8]}"),
            description=str(data.get("description", "")),
            motivation=str(data.get("motivation", "")),
            scope=[str(s) for s in data.get("scope", []) if s],
            constraints=[str(c) for c in data.get("constraints", []) if c],
            requested_by=str(data.get("requested_by", "user")),
            created_at=str(data.get("created_at") or datetime.now(timezone.utc).isoformat()),
            status=str(data.get("status", "created")),
        )


@dataclass
class SelfDevelopmentPlan:
    """Strongly typed model representing a validated codebase-grounded self-development plan."""

    goal: SelfDevelopmentGoal
    plan_id: str = field(default_factory=lambda: f"sd_plan_{uuid.uuid4().hex[:8]}")
    current_behavior: str = ""
    observed_problem: str = ""
    relevant_files: list[str] = field(default_factory=list)
    relevant_symbols: list[str] = field(default_factory=list)
    dependencies: list[str] = field(default_factory=list)
    proposed_improvement: str = ""
    expected_benefit: str = ""
    risks: list[str] = field(default_factory=list)
    constraints: list[str] = field(default_factory=list)
    validation_strategy: list[str] = field(default_factory=list)
    rollback_strategy: str = ""
    confidence: float = 1.0
    requires_more_information: bool = False
    missing_information_reason: str | None = None
    requires_elevated_authorization: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "plan_id": self.plan_id,
            "goal": self.goal.to_dict() if isinstance(self.goal, SelfDevelopmentGoal) else self.goal,
            "current_behavior": self.current_behavior,
            "observed_problem": self.observed_problem,
            "relevant_files": list(self.relevant_files),
            "relevant_symbols": list(self.relevant_symbols),
            "dependencies": list(self.dependencies),
            "proposed_improvement": self.proposed_improvement,
            "expected_benefit": self.expected_benefit,
            "risks": list(self.risks),
            "constraints": list(self.constraints),
            "validation_strategy": list(self.validation_strategy),
            "rollback_strategy": self.rollback_strategy,
            "confidence": self.confidence,
            "requires_more_information": self.requires_more_information,
            "missing_information_reason": self.missing_information_reason,
            "requires_elevated_authorization": self.requires_elevated_authorization,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SelfDevelopmentPlan":
        if not isinstance(data, dict):
            raise ValueError("Expected dictionary for SelfDevelopmentPlan")
        goal_raw = data.get("goal")
        goal = SelfDevelopmentGoal.from_dict(goal_raw) if isinstance(goal_raw, dict) else goal_raw
        return cls(
            plan_id=str(data.get("plan_id") or f"sd_plan_{uuid.uuid4().hex[:8]}"),
            goal=goal,
            current_behavior=str(data.get("current_behavior", "")),
            observed_problem=str(data.get("observed_problem", "")),
            relevant_files=[str(f) for f in data.get("relevant_files", []) if f],
            relevant_symbols=[str(s) for s in data.get("relevant_symbols", []) if s],
            dependencies=[str(d) for d in data.get("dependencies", []) if d],
            proposed_improvement=str(data.get("proposed_improvement", "")),
            expected_benefit=str(data.get("expected_benefit", "")),
            risks=[str(r) for r in data.get("risks", []) if r],
            constraints=[str(c) for c in data.get("constraints", []) if c],
            validation_strategy=[str(v) for v in data.get("validation_strategy", []) if v],
            rollback_strategy=str(data.get("rollback_strategy", "")),
            confidence=float(data.get("confidence", 1.0)),
            requires_more_information=bool(data.get("requires_more_information", False)),
            missing_information_reason=data.get("missing_information_reason"),
            requires_elevated_authorization=bool(data.get("requires_elevated_authorization", False)),
        )


@dataclass
class SelfDevelopmentEvaluation:
    """Strongly typed model representing the rigorous post-execution evaluation."""

    goal_id: str
    self_development_id: str
    evaluation_id: str = field(default_factory=lambda: f"sd_eval_{uuid.uuid4().hex[:8]}")
    change_applied: bool = False
    verification_status: bool = False
    test_status: str = "not_run"
    tests_passed: int = 0
    tests_failed: int = 0
    improvement_success: bool = False
    overall_outcome: str = "UNKNOWN"
    observed_result: str = ""
    regression_status: str = "not_run"
    lesson: str = ""
    recommendation: str = ""
    provider_used: str | None = None
    model_used: str | None = None
    fallback_used: bool = False
    cloud_request_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "evaluation_id": self.evaluation_id,
            "goal_id": self.goal_id,
            "self_development_id": self.self_development_id,
            "change_applied": self.change_applied,
            "verification_status": self.verification_status,
            "test_status": self.test_status,
            "tests_passed": self.tests_passed,
            "tests_failed": self.tests_failed,
            "improvement_success": self.improvement_success,
            "overall_outcome": self.overall_outcome,
            "observed_result": self.observed_result,
            "regression_status": self.regression_status,
            "lesson": self.lesson,
            "recommendation": self.recommendation,
            "provider_used": self.provider_used,
            "model_used": self.model_used,
            "fallback_used": self.fallback_used,
            "cloud_request_id": self.cloud_request_id,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SelfDevelopmentEvaluation":
        if not isinstance(data, dict):
            raise ValueError("Expected dictionary for SelfDevelopmentEvaluation")
        return cls(
            evaluation_id=str(data.get("evaluation_id") or f"sd_eval_{uuid.uuid4().hex[:8]}"),
            goal_id=str(data.get("goal_id", "")),
            self_development_id=str(data.get("self_development_id", "")),
            change_applied=bool(data.get("change_applied", False)),
            verification_status=bool(data.get("verification_status", False)),
            test_status=str(data.get("test_status", "not_run")),
            tests_passed=int(data.get("tests_passed", 0)),
            tests_failed=int(data.get("tests_failed", 0)),
            improvement_success=bool(data.get("improvement_success", False)),
            overall_outcome=str(data.get("overall_outcome", "UNKNOWN")),
            observed_result=str(data.get("observed_result", "")),
            regression_status=str(data.get("regression_status", "not_run")),
            lesson=str(data.get("lesson", "")),
            recommendation=str(data.get("recommendation", "")),
            provider_used=data.get("provider_used"),
            model_used=data.get("model_used"),
            fallback_used=bool(data.get("fallback_used", False)),
            cloud_request_id=data.get("cloud_request_id"),
        )


@dataclass
class SelfDevelopmentCycleResult:
    """Represents the complete state and audit record of a self-development cycle."""

    self_development_id: str
    goal: SelfDevelopmentGoal
    plan: SelfDevelopmentPlan | None = None
    proposal: CodeChangeProposal | None = None
    execution_result: DeveloperExecutionResult | None = None
    evaluation: SelfDevelopmentEvaluation | None = None
    status: str = "created"
    requires_elevated_authorization: bool = False
    summary: str = ""
    error: str | None = None
    provider: str | None = None
    model: str | None = None
    cloud_request_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "self_development_id": self.self_development_id,
            "goal": self.goal.to_dict() if isinstance(self.goal, SelfDevelopmentGoal) else self.goal,
            "plan": self.plan.to_dict() if self.plan else None,
            "proposal": self.proposal.to_dict() if self.proposal else None,
            "execution_result": self.execution_result.to_dict() if self.execution_result else None,
            "evaluation": self.evaluation.to_dict() if self.evaluation else None,
            "status": self.status,
            "requires_elevated_authorization": self.requires_elevated_authorization,
            "summary": self.summary,
            "error": self.error,
            "provider": self.provider,
            "model": self.model,
            "cloud_request_id": self.cloud_request_id,
        }
