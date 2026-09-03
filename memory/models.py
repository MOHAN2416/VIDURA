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

    def __post_init__(self) -> None:
        if not self.task or not self.task.strip():
            raise ValueError("ExperienceRecord task description cannot be empty.")
