import json
from typing import Any
import pytest
from config import load_config
from models.base import BaseLLMProvider
from tools import ToolRegistry, ListDirectoryTool, ReadFileTool, SearchFilesTool
from agent import Agent


class FakeToolModelProvider(BaseLLMProvider):
    """Fake model provider for testing tool interactions deterministically."""

    def __init__(self, responses: list[str]) -> None:
        self.responses = responses
        self.call_count = 0

    @property
    def model_name(self) -> str:
        return "fake-tool-model"

    @property
    def provider_name(self) -> str:
        return "Fake Tool Provider"

    def generate(self, messages: list[dict[str, str]], **kwargs: Any) -> str:
        response = self.responses[min(self.call_count, len(self.responses) - 1)]
        self.call_count += 1
        return response


# --- ToolRegistry Tests (1 - 4) ---

def test_registry_register_and_get() -> None:
    """Test 1: Register a tool and retrieve it by name."""
    registry = ToolRegistry()
    tool = ListDirectoryTool()
    registry.register(tool)
    assert registry.get("list_directory") is tool


def test_registry_list_tools() -> None:
    """Test 2: List registered tools."""
    registry = ToolRegistry()
    registry.register(ListDirectoryTool())
    registry.register(ReadFileTool())
    tools = registry.list_tools()
    assert len(tools) == 2
    assert {t.name for t in tools} == {"list_directory", "read_file"}


def test_registry_unknown_tool_rejection() -> None:
    """Test 3: Unknown tool execution is rejected safely."""
    registry = ToolRegistry()
    result = registry.execute("unknown_tool", {"arg": "val"})
    assert result["success"] is False
    assert "Tool 'unknown_tool' is not registered" in result["error"]


def test_registry_execute() -> None:
    """Test 4: Registered tool execution via registry succeeds."""
    registry = ToolRegistry()
    registry.register(ListDirectoryTool())
    result = registry.execute("list_directory", {"path": "."})
    assert result["success"] is True
    assert result["tool"] == "list_directory"


# --- list_directory Tests (5 - 8) ---

def test_list_directory_workspace() -> None:
    """Test 5: List the VIDURA workspace directory."""
    tool = ListDirectoryTool()
    result = tool.execute(path=".")
    assert result["success"] is True
    names = [e["name"] for e in result["data"]["entries"]]
    assert "config.py" in names
    assert "main.py" in names


def test_list_directory_nonexistent() -> None:
    """Test 6: Nonexistent directory is handled cleanly."""
    tool = ListDirectoryTool()
    result = tool.execute(path="nonexistent_folder_xyz")
    assert result["success"] is False
    assert "does not exist" in result["error"]


def test_list_directory_outside_workspace() -> None:
    """Test 7: Path outside workspace is rejected."""
    tool = ListDirectoryTool()
    result = tool.execute(path="/etc")
    assert result["success"] is False
    assert "outside the allowed workspace" in result["error"]


def test_list_directory_path_traversal() -> None:
    """Test 8: Path traversal attempts are rejected."""
    tool = ListDirectoryTool()
    result = tool.execute(path="../../")
    assert result["success"] is False
    assert "outside the allowed workspace" in result["error"]


# --- read_file Tests (9 - 12) ---

def test_read_file_valid() -> None:
    """Test 9: Read a valid text file."""
    tool = ReadFileTool()
    result = tool.execute(path="pyproject.toml")
    assert result["success"] is True
    assert "[project]" in result["data"]["content"]


def test_read_file_nonexistent() -> None:
    """Test 10: Nonexistent file is handled cleanly."""
    tool = ReadFileTool()
    result = tool.execute(path="nonexistent_file.txt")
    assert result["success"] is False
    assert "does not exist" in result["error"]


def test_read_file_directory_as_file() -> None:
    """Test 11: Directory passed as file is handled cleanly."""
    tool = ReadFileTool()
    result = tool.execute(path="models")
    assert result["success"] is False
    assert "points to a directory" in result["error"]


def test_read_file_outside_workspace() -> None:
    """Test 12: Path outside workspace is rejected (security check)."""
    tool = ReadFileTool()
    result = tool.execute(path="/etc/passwd")
    assert result["success"] is False
    assert "outside the allowed workspace" in result["error"]


# --- search_files Tests (13 - 15) ---

def test_search_files_match() -> None:
    """Test 13: Find a known string in project files."""
    tool = SearchFilesTool()
    result = tool.execute(query="OLLAMA_MODEL", path=".")
    assert result["success"] is True
    matches = result["data"]["matches"]
    assert len(matches) > 0
    assert any(m["file"] == "config.py" for m in matches)


def test_search_files_no_match() -> None:
    """Test 14: Search query with no matches returns empty list."""
    tool = SearchFilesTool()
    search_term = "NONEXISTENT_" + "QUERY_XYZ_99999"
    result = tool.execute(query=search_term, path=".")
    assert result["success"] is True
    assert len(result["data"]["matches"]) == 0


def test_search_files_outside_workspace() -> None:
    """Test 15: Outside workspace search directory is rejected."""
    tool = SearchFilesTool()
    result = tool.execute(query="root", path="/var/log")
    assert result["success"] is False
    assert "outside the allowed workspace" in result["error"]


# --- Agent-Tool Integration Tests (16 - 21) ---

def test_agent_tool_list_directory() -> None:
    """Test 16: Fake model requests list_directory tool execution."""
    registry = ToolRegistry()
    registry.register(ListDirectoryTool())

    step1 = json.dumps({"action": "tool_call", "tool_name": "list_directory", "arguments": {"path": "."}})
    step2 = json.dumps({"action": "respond", "content": "I see main.py and config.py in the workspace."})
    provider = FakeToolModelProvider([step1, step2])

    agent = Agent(model=provider, tool_registry=registry, max_steps=5)
    result = agent.run("List workspace contents")

    assert "main.py and config.py" in result
    assert provider.call_count == 2


def test_agent_tool_read_file() -> None:
    """Test 17: Fake model requests read_file tool execution."""
    registry = ToolRegistry()
    registry.register(ReadFileTool())

    step1 = json.dumps({"action": "tool_call", "tool_name": "read_file", "arguments": {"path": "pyproject.toml"}})
    step2 = json.dumps({"action": "respond", "content": "Project name is vidura."})
    provider = FakeToolModelProvider([step1, step2])

    agent = Agent(model=provider, tool_registry=registry, max_steps=5)
    result = agent.run("Read pyproject.toml")

    assert "Project name is vidura" in result
    assert provider.call_count == 2


def test_agent_tool_unknown_tool() -> None:
    """Test 18: Fake model requests unknown tool and recovers."""
    registry = ToolRegistry()
    registry.register(ReadFileTool())

    step1 = json.dumps({"action": "tool_call", "tool_name": "unregistered_tool", "arguments": {}})
    step2 = json.dumps({"action": "respond", "content": "Tool was not available."})
    provider = FakeToolModelProvider([step1, step2])

    agent = Agent(model=provider, tool_registry=registry, max_steps=5)
    result = agent.run("Call bad tool")

    assert "Tool was not available" in result
    assert provider.call_count == 2


def test_agent_direct_respond() -> None:
    """Test 19: Fake model directly responds without calling tools."""
    registry = ToolRegistry()
    registry.register(ListDirectoryTool())

    step1 = json.dumps({"action": "respond", "content": "Direct response without tools."})
    provider = FakeToolModelProvider([step1])

    agent = Agent(model=provider, tool_registry=registry, max_steps=5)
    result = agent.run("Hi")

    assert result == "Direct response without tools."
    assert provider.call_count == 1


def test_agent_tool_failure_observation() -> None:
    """Test 20: Tool execution failure becomes an observation and agent continues."""
    registry = ToolRegistry()
    registry.register(ReadFileTool())

    step1 = json.dumps({"action": "tool_call", "tool_name": "read_file", "arguments": {"path": "missing.txt"}})
    step2 = json.dumps({"action": "respond", "content": "The requested file does not exist."})
    provider = FakeToolModelProvider([step1, step2])

    agent = Agent(model=provider, tool_registry=registry, max_steps=5)
    result = agent.run("Read missing file")

    assert "does not exist" in result
    assert provider.call_count == 2


def test_agent_tool_max_steps() -> None:
    """Test 21: Repeated tool calls stop cleanly at max_steps."""
    registry = ToolRegistry()
    registry.register(ListDirectoryTool())

    loop_tool = json.dumps({"action": "tool_call", "tool_name": "list_directory", "arguments": {"path": "."}})
    provider = FakeToolModelProvider([loop_tool])

    agent = Agent(model=provider, tool_registry=registry, max_steps=3)
    result = agent.run("Loop tools forever")

    assert "Reached maximum step limit" in result
    assert provider.call_count == 3


# --- Security Path Traversal Tests ---

def test_security_path_traversal_variations() -> None:
    """Security Test: Path traversal attempts with ../, ../../, absolute paths are rejected."""
    config = load_config()
    
    bad_paths = [
        "../secret.txt",
        "../../secret.txt",
        "/etc/passwd",
        "/var/log/syslog",
        "tests/../../outside.txt",
    ]
    
    for bad_path in bad_paths:
        with pytest.raises(ValueError) as exc_info:
            config.validate_workspace_path(bad_path)
        assert "outside the allowed workspace" in str(exc_info.value)
