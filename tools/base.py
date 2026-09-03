from abc import ABC, abstractmethod
from typing import Any


def success_result(tool_name: str, data: Any) -> dict[str, Any]:
    """Utility helper to build a successful tool result payload."""
    return {
        "success": True,
        "tool": tool_name,
        "data": data,
        "error": None,
    }


def error_result(tool_name: str, error_message: str) -> dict[str, Any]:
    """Utility helper to build an error tool result payload."""
    return {
        "success": False,
        "tool": tool_name,
        "data": None,
        "error": error_message,
    }


class BaseTool(ABC):
    """Abstract Base Class for all VIDURA Tools.
    
    Provides a standardized interface for tool definitions, parameter schemas,
    and structured output formatting.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Returns the unique tool identifier string."""
        ...

    @property
    @abstractmethod
    def description(self) -> str:
        """Returns a concise description of the tool functionality."""
        ...

    @property
    @abstractmethod
    def parameters(self) -> dict[str, Any]:
        """Returns a JSON Schema dictionary describing tool input arguments."""
        ...

    @abstractmethod
    def execute(self, **kwargs: Any) -> dict[str, Any]:
        """Executes the tool with the provided kwargs.

        Returns:
            Structured result dictionary with 'success', 'tool', 'data', and 'error' keys.
        """
        ...

    def success_result(self, data: Any) -> dict[str, Any]:
        """Utility helper to build a successful tool result payload."""
        return success_result(self.name, data)

    def error_result(self, error_message: str) -> dict[str, Any]:
        """Utility helper to build an error tool result payload."""
        return error_result(self.name, error_message)
