from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from task_understanding.models import DeveloperTask, TaskType


@dataclass
class PlannedComponent:
    """Represents an inspected code component within the codebase."""
    module_name: str
    file_path: str
    classes: list[str] = field(default_factory=list)
    functions: list[str] = field(default_factory=list)
    methods: list[str] = field(default_factory=list)
    docstring: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "module_name": self.module_name,
            "file_path": self.file_path,
            "classes": list(self.classes),
            "functions": list(self.functions),
            "methods": list(self.methods),
            "docstring": self.docstring,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PlannedComponent":
        if not isinstance(data, dict):
            return cls(module_name="", file_path="")
        return cls(
            module_name=str(data.get("module_name", "")),
            file_path=str(data.get("file_path", "")),
            classes=[str(c) for c in data.get("classes", []) if c],
            functions=[str(f) for f in data.get("functions", []) if f],
            methods=[str(m) for m in data.get("methods", []) if m],
            docstring=data.get("docstring"),
        )


@dataclass
class DeveloperPlan:
    """Strongly typed model representing a validated, codebase-aware development plan."""

    original_task: DeveloperTask
    relevant_files: list[str] = field(default_factory=list)
    relevant_symbols: list[str] = field(default_factory=list)
    dependencies: list[str] = field(default_factory=list)
    affected_components: list[str] = field(default_factory=list)
    planned_changes: list[str] = field(default_factory=list)
    risks: list[str] = field(default_factory=list)
    constraints: list[str] = field(default_factory=list)
    requires_more_information: bool = False
    missing_information_reason: str | None = None
    requires_plan_update: bool = False
    confidence: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        """Converts DeveloperPlan into dictionary structure."""
        return {
            "original_task": self.original_task.to_dict() if isinstance(self.original_task, DeveloperTask) else {},
            "relevant_files": list(self.relevant_files),
            "relevant_symbols": list(self.relevant_symbols),
            "dependencies": list(self.dependencies),
            "affected_components": list(self.affected_components),
            "planned_changes": list(self.planned_changes),
            "risks": list(self.risks),
            "constraints": list(self.constraints),
            "requires_more_information": self.requires_more_information,
            "missing_information_reason": self.missing_information_reason,
            "requires_plan_update": self.requires_plan_update,
            "confidence": self.confidence,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DeveloperPlan":
        """Safely parses and validates a dictionary into a DeveloperPlan instance."""
        if not isinstance(data, dict):
            fallback_task = DeveloperTask(task_type=TaskType.UNABLE_TO_CLASSIFY, goal="Malformed plan input")
            return cls(original_task=fallback_task, requires_more_information=True, missing_information_reason="Malformed input")

        raw_task = data.get("original_task")
        if isinstance(raw_task, DeveloperTask):
            task = raw_task
        elif isinstance(raw_task, dict):
            task = DeveloperTask.from_dict(raw_task)
        else:
            task = DeveloperTask(task_type=TaskType.GENERAL, goal="Default plan task")

        def _clean_str_list(raw_val: Any) -> list[str]:
            if not isinstance(raw_val, list):
                raw_val = [raw_val] if raw_val is not None else []
            cleaned = []
            for item in raw_val:
                if item and isinstance(item, (str, Path)):
                    s = str(item).strip()
                    if s and s not in cleaned:
                        cleaned.append(s)
            return cleaned

        relevant_files = _clean_str_list(data.get("relevant_files", []))
        relevant_symbols = _clean_str_list(data.get("relevant_symbols", []))
        dependencies = _clean_str_list(data.get("dependencies", []))
        affected_components = _clean_str_list(data.get("affected_components", []))
        planned_changes = _clean_str_list(data.get("planned_changes", []))
        risks = _clean_str_list(data.get("risks", []))
        constraints = _clean_str_list(data.get("constraints", []))

        try:
            confidence = float(data.get("confidence", 1.0))
            confidence = max(0.0, min(1.0, confidence))
        except (ValueError, TypeError):
            confidence = 1.0

        missing_reason = data.get("missing_information_reason")
        missing_reason_str = str(missing_reason).strip() if missing_reason is not None else None

        return cls(
            original_task=task,
            relevant_files=relevant_files,
            relevant_symbols=relevant_symbols,
            dependencies=dependencies,
            affected_components=affected_components,
            planned_changes=planned_changes,
            risks=risks,
            constraints=constraints,
            requires_more_information=bool(data.get("requires_more_information", False)),
            missing_information_reason=missing_reason_str,
            requires_plan_update=bool(data.get("requires_plan_update", False)),
            confidence=confidence,
        )

    def validate(self, workspace_root: Path | None = None) -> tuple[bool, str | None]:
        """Validates the developer plan invariants according to Phase 8.2 specifications."""
        if not isinstance(self.original_task, DeveloperTask):
            return False, "original_task must be an instance of DeveloperTask."

        if not (0.0 <= self.confidence <= 1.0):
            return False, f"Confidence {self.confidence} out of range [0.0, 1.0]."

        executable_patterns = ["apply_code_change", "os.system", "subprocess", "rm -rf", "shutil.rmtree", "git commit"]
        for change in self.planned_changes:
            for pattern in executable_patterns:
                if pattern in change.lower():
                    return False, f"Planned changes must not contain executable operations ('{pattern}')."

        if workspace_root:
            root = Path(workspace_root).resolve()
            for f in self.relevant_files:
                p = (root / f).resolve()
                if not p.is_relative_to(root):
                    return False, f"Relevant file '{f}' resolves outside workspace root '{root}'."

        return True, None

