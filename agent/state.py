from dataclasses import dataclass, field
from typing import Any


@dataclass
class AgentState:
    """Represents the complete execution state of a VIDURA Agent instance."""
    user_request: str
    messages: list[dict[str, str]] = field(default_factory=list)
    step: int = 0
    max_steps: int = 10
    current_action: str | None = None
    action_content: Any = None
    tool_name: str | None = None
    tool_arguments: dict[str, Any] | None = None
    observation: Any = None
    task_understanding: Any = None
    developer_plan: Any = None
    generation_result: Any = None
    execution_result: Any = None
    final_response: str | None = None
    completed: bool = False
    requested_provider: str | None = None
    requested_model: str | None = None
    actual_provider: str | None = None
    actual_model: str | None = None
    fallback_used: bool = False
    fallback_reason: str | None = None
    cloud_error: str | None = None
    cloud_request_id: str | None = None
    usage_metadata: dict[str, Any] | None = None

    def is_finished(self) -> bool:
        """Returns True if the agent loop should terminate."""
        return self.completed or self.step >= self.max_steps
