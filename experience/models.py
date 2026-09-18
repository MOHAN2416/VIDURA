"""VIDURA Experience Models (Phase 11).

Defines typed representations of experiences, derived lessons, retrieval queries,
and scored experience results.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from memory.models import ExperienceRecord


@dataclass
class LessonRecord:
    """Represents a generalized or reusable lesson derived from empirical experience."""
    topic: str
    lesson: str
    lesson_id: str = field(default_factory=lambda: f"lsn_{uuid.uuid4().hex[:8]}")
    task_type: str = "developer"
    component: str | None = None
    source_experience_ids: list[str] = field(default_factory=list)
    confidence: float = 1.0
    success_count: int = 0
    failure_count: int = 0
    recommendation: str = ""
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict[str, Any]:
        return {
            "lesson_id": self.lesson_id,
            "topic": self.topic,
            "lesson": self.lesson,
            "task_type": self.task_type,
            "component": self.component,
            "source_experience_ids": list(self.source_experience_ids),
            "confidence": self.confidence,
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "recommendation": self.recommendation,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LessonRecord":
        return cls(
            lesson_id=str(data.get("lesson_id") or f"lsn_{uuid.uuid4().hex[:8]}"),
            topic=str(data.get("topic", "")),
            lesson=str(data.get("lesson", "")),
            task_type=str(data.get("task_type", "developer")),
            component=data.get("component"),
            source_experience_ids=list(data.get("source_experience_ids", [])),
            confidence=float(data.get("confidence", 1.0)),
            success_count=int(data.get("success_count", 0)),
            failure_count=int(data.get("failure_count", 0)),
            recommendation=str(data.get("recommendation", "")),
            created_at=str(data.get("created_at") or datetime.now(timezone.utc).isoformat()),
            updated_at=str(data.get("updated_at") or datetime.now(timezone.utc).isoformat()),
        )


@dataclass
class ExperienceQuery:
    """Query filters and parameters for retrieving relevant past experiences."""
    query: str = ""
    task_type: str | None = None
    file_path: str | None = None
    module: str | None = None
    symbol: str | None = None
    component: str | None = None
    failure_type: str | None = None
    provider: str | None = None
    success: bool | None = None
    limit: int = 5
    min_score: float = 0.2

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "task_type": self.task_type,
            "file_path": self.file_path,
            "module": self.module,
            "symbol": self.symbol,
            "component": self.component,
            "failure_type": self.failure_type,
            "provider": self.provider,
            "success": self.success,
            "limit": self.limit,
            "min_score": self.min_score,
        }


@dataclass
class ScoredExperience:
    """An ExperienceRecord paired with its deterministic relevance score and breakdown."""
    experience: ExperienceRecord
    score: float
    score_breakdown: dict[str, float] = field(default_factory=dict)
    is_warning: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "experience": self.experience.to_dict(),
            "score": round(self.score, 4),
            "score_breakdown": {k: round(v, 4) for k, v in self.score_breakdown.items()},
            "is_warning": self.is_warning,
        }
