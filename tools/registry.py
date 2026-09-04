import logging
from typing import Any
from tools.base import BaseTool

logger = logging.getLogger("VIDURA.tools.registry")


class ToolRegistry:
    """Central registry managing VIDURA tool registration, schemas, and execution."""

    def __init__(self) -> None:
        self._tools: dict[str, BaseTool] = {}

    def register(self, tool: BaseTool) -> None:
        """Registers a tool instance.

        Args:
            tool: An instance of BaseTool.
        """
        if not isinstance(tool, BaseTool):
            raise TypeError(f"Expected instance of BaseTool, got {type(tool)}")
        
        self._tools[tool.name] = tool
        logger.info(f"Registered tool: '{tool.name}'")

    def get(self, name: str) -> BaseTool | None:
        """Retrieves a registered tool by name."""
        return self._tools.get(name)

    def list_tools(self) -> list[BaseTool]:
        """Returns a list of all registered tool instances."""
        return list(self._tools.values())

    def get_tool_definitions(self) -> list[dict[str, Any]]:
        """Generates tool schema definitions for language model prompts or tool calling API."""
        definitions = []
        for tool in self._tools.values():
            definitions.append({
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters,
            })
        return definitions

    def execute(self, name: str, arguments: dict[str, Any] | None = None) -> dict[str, Any]:
        """Executes a registered tool by name.

        Args:
            name: Name of the tool to execute.
            arguments: Dictionary of keyword arguments to pass to the tool.

        Returns:
            Structured tool result dictionary.
        """
        arguments = arguments or {}
        tool = self.get(name)

        if not tool:
            error_msg = f"Tool '{name}' is not registered."
            logger.warning(error_msg)
            return {
                "success": False,
                "tool": name,
                "data": None,
                "error": error_msg,
            }

        try:
            logger.info(f"Executing tool '{name}' with arguments: {arguments}")
            result = tool.execute(**arguments)
            return result
        except Exception as err:
            logger.error(f"Error executing tool '{name}': {err}", exc_info=True)
            return {
                "success": False,
                "tool": name,
                "data": None,
                "error": f"Execution error in '{name}': {err}",
            }
