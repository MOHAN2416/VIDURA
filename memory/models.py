import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


class MemoryCategory:
    """Supported Memory Categories in VIDURA."""
    CONVERSATION = "conversation"
    USER = "user"
    PROJECT = "project"
    EXPERIENCE = "experience"

    ALL_CATEGORIES = {CONVERSATION, USER, PROJECT, EXPERIENCE}

    @classmethod
    def is_valid(cls, category: str) -> bool:
        return category in cls.ALL_CATEGORIES


@dataclass
class Memory:
    """Typed representation of a persistent Memory record."""
    type: str
    content: str
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    metadata: dict[str, Any] = field(default_factory=dict)
    importance: float = 0.5
    source_session: str | None = None

    def __post_init__(self) -> None:
        if not self.content or not self.content.strip():
            raise ValueError("Memory content cannot be empty.")
        if not MemoryCategory.is_valid(self.type):
            raise ValueError(
                f"Invalid memory type '{self.type}'. Must be one of {MemoryCategory.ALL_CATEGORIES}"
            )
        if not (0.0 <= self.importance <= 1.0):
            raise ValueError("Memory importance must be between 0.0 and 1.0.")


@dataclass
class ExperienceRecord:
    """Typed representation of an Experience memory record."""
    task: str
    attempt: int = 1
    action_summary: str = ""
    result: str = ""
    success: bool = True
    lesson: str = ""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    task_type: str = "developer"
    goal: str = ""
    plan_summary: str = ""
    affected_files: list[str] = field(default_factory=list)
    affected_components: list[str] = field(default_factory=list)
    relevant_symbols: list[str] = field(default_factory=list)
    provider: str = ""
    model: str = ""
    is_cloud: bool = False
    verification_status: str = ""
    test_status: str = ""
    tests_passed: int = 0
    tests_failed: int = 0
    failure_reason: str | None = None
    failure_type: str | None = None
    recommendation: str = ""
    confidence: float = 1.0
    fallback_used: bool = False
    duplicate_count: int = 1
    evidence_count: int = 1
    related_experience_ids: list[str] = field(default_factory=list)
    cloud_request_id: str | None = None

    def __post_init__(self) -> None:
        if not self.task or not self.task.strip():
            raise ValueError("ExperienceRecord task description cannot be empty.")
        if not self.goal:
            self.goal = self.task

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "task": self.task,
            "attempt": self.attempt,
            "action_summary": self.action_summary,
            "result": self.result,
            "success": self.success,
            "lesson": self.lesson,
            "created_at": self.created_at,
            "task_type": self.task_type,
            "goal": self.goal,
            "plan_summary": self.plan_summary,
            "affected_files": list(self.affected_files),
            "affected_components": list(self.affected_components),
            "relevant_symbols": list(self.relevant_symbols),
            "provider": self.provider,
            "model": self.model,
            "is_cloud": self.is_cloud,
            "verification_status": self.verification_status,
            "test_status": self.test_status,
            "tests_passed": self.tests_passed,
            "tests_failed": self.tests_failed,
            "failure_reason": self.failure_reason,
            "failure_type": self.failure_type,
            "recommendation": self.recommendation,
            "confidence": self.confidence,
            "fallback_used": self.fallback_used,
            "duplicate_count": self.duplicate_count,
            "evidence_count": self.evidence_count,
            "related_experience_ids": list(self.related_experience_ids),
            "cloud_request_id": self.cloud_request_id,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ExperienceRecord":
        return cls(
            id=str(data.get("id") or uuid.uuid4()),
            task=str(data.get("task", "")),
            attempt=int(data.get("attempt", 1)),
            action_summary=str(data.get("action_summary", "")),
            result=str(data.get("result", "")),
            success=bool(data.get("success", True)),
            lesson=str(data.get("lesson", "")),
            created_at=str(data.get("created_at") or datetime.now(timezone.utc).isoformat()),
            task_type=str(data.get("task_type", "developer")),
            goal=str(data.get("goal", "")),
            plan_summary=str(data.get("plan_summary", "")),
            affected_files=list(data.get("affected_files", [])),
            affected_components=list(data.get("affected_components", [])),
            relevant_symbols=list(data.get("relevant_symbols", [])),
            provider=str(data.get("provider", "")),
            model=str(data.get("model", "")),
            is_cloud=bool(data.get("is_cloud", False)),
            verification_status=str(data.get("verification_status", "")),
            test_status=str(data.get("test_status", "")),
            tests_passed=int(data.get("tests_passed", 0)),
            tests_failed=int(data.get("tests_failed", 0)),
            failure_reason=data.get("failure_reason"),
            failure_type=data.get("failure_type"),
            recommendation=str(data.get("recommendation", "")),
            confidence=float(data.get("confidence", 1.0)),
            fallback_used=bool(data.get("fallback_used", False)),
            duplicate_count=int(data.get("duplicate_count", 1)),
            evidence_count=int(data.get("evidence_count", 1)),
            related_experience_ids=list(data.get("related_experience_ids", [])),
            cloud_request_id=data.get("cloud_request_id"),
        )
