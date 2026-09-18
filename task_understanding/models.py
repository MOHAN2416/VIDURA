from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


class TaskType(str, Enum):
    GENERAL = "general"
    CODE_CHANGE = "code_change"
    CODE_DEBUG = "code_debug"
    CODE_EXPLANATION = "code_explanation"
    CODE_REVIEW = "code_review"
    TEST_REQUEST = "test_request"
    UNABLE_TO_CLASSIFY = "unable_to_classify"


@dataclass
class DeveloperTask:
    """Strongly typed model representing an analyzed developer task or query."""

    task_type: TaskType = TaskType.GENERAL
    is_development_task: bool = False
    goal: str = ""
    requested_change: str | None = None
    target_files: list[str] = field(default_factory=list)
    target_symbols: list[str] = field(default_factory=list)
    constraints: list[str] = field(default_factory=list)
    expected_behavior: str | None = None
    requires_codebase_analysis: bool = False
    confidence: float = 1.0
    local_only: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Converts DeveloperTask to dictionary structure."""
        return {
            "task_type": self.task_type.value if isinstance(self.task_type, TaskType) else str(self.task_type),
            "is_development_task": self.is_development_task,
            "goal": self.goal,
            "requested_change": self.requested_change,
            "target_files": list(self.target_files),
            "target_symbols": list(self.target_symbols),
            "constraints": list(self.constraints),
            "expected_behavior": self.expected_behavior,
            "requires_codebase_analysis": self.requires_codebase_analysis,
            "confidence": self.confidence,
            "local_only": self.local_only,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DeveloperTask":
        """Safely parses and validates a dictionary into a strongly typed DeveloperTask instance."""
        if not isinstance(data, dict):
            return cls(task_type=TaskType.UNABLE_TO_CLASSIFY, is_development_task=False, goal="Malformed input")

        raw_type = str(data.get("task_type", "general")).lower().strip()
        try:
            task_type = TaskType(raw_type)
        except ValueError:
            task_type = TaskType.UNABLE_TO_CLASSIFY

        is_dev = task_type not in (TaskType.GENERAL, TaskType.UNABLE_TO_CLASSIFY)
        if "is_development_task" in data:
            is_dev = bool(data.get("is_development_task"))

        raw_files = data.get("target_files", [])
        if not isinstance(raw_files, list):
            raw_files = [str(raw_files)] if raw_files else []
        target_files = [str(f).strip() for f in raw_files if f and isinstance(f, (str, Path))]

        raw_symbols = data.get("target_symbols", [])
        if not isinstance(raw_symbols, list):
            raw_symbols = [str(raw_symbols)] if raw_symbols else []
        target_symbols = [str(s).strip() for s in raw_symbols if s and isinstance(s, str)]

        raw_constraints = data.get("constraints", [])
        if not isinstance(raw_constraints, list):
            raw_constraints = [str(raw_constraints)] if raw_constraints else []
        constraints = [str(c).strip() for c in raw_constraints if c and isinstance(c, str)]

        try:
            confidence = float(data.get("confidence", 1.0))
        except (ValueError, TypeError):
            confidence = 1.0

        req_change = data.get("requested_change")
        req_change_str = str(req_change).strip() if req_change is not None else None

        exp_behavior = data.get("expected_behavior")
        exp_behavior_str = str(exp_behavior).strip() if exp_behavior is not None else None

        return cls(
            task_type=task_type,
            is_development_task=is_dev,
            goal=str(data.get("goal", "")).strip(),
            requested_change=req_change_str,
            target_files=target_files,
            target_symbols=target_symbols,
            constraints=constraints,
            expected_behavior=exp_behavior_str,
            requires_codebase_analysis=bool(data.get("requires_codebase_analysis", False)),
            confidence=confidence,
            local_only=bool(data.get("local_only", False)),
        )
