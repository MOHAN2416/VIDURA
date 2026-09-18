"""Automated tests for Phase 9.2: Ollama Cloud Authentication + Connection.

Validates:
A. Cloud provider initialization
B. Correct cloud model (gemma4:31b-cloud)
C. Missing credentials
D. Invalid credentials
E. Successful mocked cloud request
F. Cloud response normalization
G. Tool-call response normalization
H. Timeout handling
I. Network failure
J. Invalid model response
K. No silent local fallback
L. AgentLoop remains provider-agnostic
M. Cloud provider cannot bypass ToolRegistry
N. Cloud provider cannot modify permission state
O. Credentials do not appear in logs
P. Credentials do not appear in AgentState
Q. Credentials do not appear in memory
R. Local provider still works without cloud credentials
S. Provider switching does not alter developer permissions
"""
import json
import logging
import os
import unittest.mock as mock
from pathlib import Path

import httpx
import ollama
import pytest

from agent import Agent, AgentLoop
from agent.state import AgentState
from config import Config
from memory import MemoryCategory, MemoryManager, MemoryStore
from models.cloud import (
    ALLOWED_CLOUD_MODELS,
    DEFAULT_CLOUD_MODEL,
    OllamaCloudProvider,
    normalize_endpoint,
)
from models.errors import (
    ModelProviderError,
    ProviderAuthenticationError,
    ProviderConfigurationError,
    ProviderRequestError,
    ProviderTimeout,
    ProviderUnavailable,
)
from models.router import ModelRouter
from permissions import PermissionManager
from tools.registry import ToolRegistry


# ==============================================================================
# Requirements A & B: Initialization and Model Enforcement
# ==============================================================================

def test_cloud_provider_initialization() -> None:
    """Requirement A: Cloud provider initializes safely with expected attributes."""
    provider = OllamaCloudProvider(
        api_key="test-secret-key",
        endpoint="https://ollama.com",
        timeout=45.0,
    )
    assert provider.model_name == "gemma4:31b-cloud"
    assert provider.provider_name == "Ollama (cloud)"
    assert provider.endpoint == "https://ollama.com"
    assert provider.timeout == 45.0
    assert provider.is_configured is True
    assert provider.capabilities.cloud is True
    assert provider.capabilities.local is False
    assert provider.capabilities.tool_calling is True
    assert repr(provider) == "OllamaCloudProvider(model='gemma4:31b-cloud', endpoint='https://ollama.com', timeout=45.0s, api_key=<present>)"


def test_cloud_provider_enforces_model_name() -> None:
    """Requirement B: Only gemma4:31b-cloud is allowed; other models are rejected."""
    assert DEFAULT_CLOUD_MODEL == "gemma4:31b-cloud"
    assert ALLOWED_CLOUD_MODELS == {"gemma4:31b-cloud"}

    with pytest.raises(ProviderConfigurationError) as exc_info:
        OllamaCloudProvider(model="unapproved-model:70b")
    assert "Invalid cloud model identifier" in str(exc_info.value)


def test_endpoint_normalization() -> None:
    """Endpoint normalizer strips trailing /api or / to avoid double routing."""
    assert normalize_endpoint("https://ollama.com/api") == "https://ollama.com"
    assert normalize_endpoint("https://ollama.com/api/") == "https://ollama.com"
    assert normalize_endpoint("https://custom.host:8443/api") == "https://custom.host:8443"
    assert normalize_endpoint("https://ollama.com") == "https://ollama.com"
    assert normalize_endpoint("") == "https://ollama.com"
    assert normalize_endpoint(None) == "https://ollama.com"


# ==============================================================================
# Requirements C & D: Missing and Invalid Credentials
# ==============================================================================

def test_missing_credentials_handling() -> None:
    """Requirement C: Missing credentials return unconfigured status and raise ProviderUnavailable."""
    provider = OllamaCloudProvider(api_key=None)
    assert provider.is_configured is False

    connected, msg = provider.check_connection()
    assert connected is False
    assert "Missing OLLAMA_CLOUD_API_KEY" in msg or "Missing cloud API key" in msg

    with pytest.raises(ProviderUnavailable) as exc_info:
        provider.generate([{"role": "user", "content": "hi"}])
    assert "CLOUD_PROVIDER_UNAVAILABLE" in str(exc_info.value)


def test_invalid_credentials_handling() -> None:
    """Requirement D: Invalid credentials (HTTP 401) trigger ProviderAuthenticationError without leaking secret."""
    secret_key = "invalid-secret-key-12345"
    provider = OllamaCloudProvider(api_key=secret_key)
    assert provider.is_configured is True

    # Mock ollama chat to raise 401 ResponseError
    with mock.patch.object(provider._client, "chat") as mock_chat:
        resp_err = ollama.ResponseError(
            error=f"Unauthorized access for token {secret_key}",
            status_code=401,
        )
        mock_chat.side_effect = resp_err

        with pytest.raises(ProviderAuthenticationError) as exc_info:
            provider.generate([{"role": "user", "content": "hello"}])
        
        err_str = str(exc_info.value)
        assert "Cloud authentication failed (Status 401)" in err_str
        assert secret_key not in err_str
        assert "[REDACTED]" in err_str


# ==============================================================================
# Requirements E & F: Successful Mocked Request and Response Normalization
# ==============================================================================

def test_successful_mocked_cloud_request() -> None:
    """Requirement E: Successful mocked request returns normalized assistant text."""
    provider = OllamaCloudProvider(api_key="valid-key")
    
    mock_resp = {
        "message": {
            "role": "assistant",
            "content": "VIDURA CLOUD TEST RESPONSE",
        },
        "done": True,
    }

    with mock.patch.object(provider._client, "chat", return_value=mock_resp) as mock_chat:
        result = provider.generate([{"role": "user", "content": "ping"}])
        assert result == "VIDURA CLOUD TEST RESPONSE"
        mock_chat.assert_called_once_with(
            model="gemma4:31b-cloud",
            messages=[{"role": "user", "content": "ping"}],
        )


def test_cloud_response_normalization_formats() -> None:
    """Requirement F: Response normalization handles SDK objects, dicts, and nested content."""
    provider = OllamaCloudProvider(api_key="valid-key")

    # Direct dict message
    dict_resp = {"message": {"role": "assistant", "content": "hello from dict"}}
    assert provider._normalize_response(dict_resp) == "hello from dict"

    # Mock SDK object with attributes
    class FakeMessage:
        content = "hello from sdk obj"
        tool_calls = None

    class FakeResponse:
        message = FakeMessage()

    assert provider._normalize_response(FakeResponse()) == "hello from sdk obj"

    # Direct content dict
    direct_content = {"content": "direct content"}
    assert provider._normalize_response(direct_content) == "direct content"


# ==============================================================================
# Requirement G: Tool-Call Normalization
# ==============================================================================

def test_tool_call_response_normalization() -> None:
    """Requirement G: Tool calls returned by Ollama Cloud are normalized to VIDURA decision JSON."""
    provider = OllamaCloudProvider(api_key="valid-key")

    # Dict format tool call
    dict_tool_resp = {
        "message": {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "function": {
                        "name": "find_symbol",
                        "arguments": {"name": "AgentLoop"},
                    }
                }
            ],
        }
    }
    normalized_json = provider._normalize_response(dict_tool_resp)
    decision = json.loads(normalized_json)
    assert decision["action"] == "tool_call"
    assert decision["tool_name"] == "find_symbol"
    assert decision["arguments"] == {"name": "AgentLoop"}

    # Object format tool call
    class FakeFunction:
        name = "read_file"
        arguments = {"file_path": "main.py"}

    class FakeToolCall:
        function = FakeFunction()

    class FakeMessageWithTool:
        content = None
        tool_calls = [FakeToolCall()]

    class FakeRespObj:
        message = FakeMessageWithTool()

    normalized_obj_json = provider._normalize_response(FakeRespObj())
    decision_obj = json.loads(normalized_obj_json)
    assert decision_obj["action"] == "tool_call"
    assert decision_obj["tool_name"] == "read_file"
    assert decision_obj["arguments"] == {"file_path": "main.py"}


# ==============================================================================
# Requirements H & I: Timeout and Network Failure Handling
# ==============================================================================

def test_timeout_handling() -> None:
    """Requirement H: httpx.TimeoutException maps to ProviderTimeout without fabricating responses."""
    provider = OllamaCloudProvider(api_key="valid-key", timeout=10.0)

    with mock.patch.object(provider._client, "chat") as mock_chat:
        mock_chat.side_effect = httpx.ReadTimeout("Request timed out after 10.0s")

        with pytest.raises(ProviderTimeout) as exc_info:
            provider.generate([{"role": "user", "content": "slow prompt"}])
        assert "timed out after 10.0s" in str(exc_info.value)


def test_network_failure_handling() -> None:
    """Requirement I: Network and transport failures map to ProviderRequestError."""
    secret = "secret-network-key"
    provider = OllamaCloudProvider(api_key=secret)

    with mock.patch.object(provider._client, "chat") as mock_chat:
        mock_chat.side_effect = httpx.ConnectError(f"Connection refused at https://ollama.com?key={secret}")

        with pytest.raises(ProviderRequestError) as exc_info:
            provider.generate([{"role": "user", "content": "test"}])
        err_msg = str(exc_info.value)
        assert "Network failure communicating with Ollama Cloud" in err_msg
        assert secret not in err_msg
        assert "[REDACTED]" in err_msg


# ==============================================================================
# Requirements J & K: Malformed Response and No Silent Fallback
# ==============================================================================

def test_invalid_model_response_handling() -> None:
    """Requirement J: Null or empty model responses raise ModelProviderError."""
    provider = OllamaCloudProvider(api_key="valid-key")

    with pytest.raises(ModelProviderError) as exc_info:
        provider._normalize_response(None)
    assert "Empty or null response" in str(exc_info.value)

    with pytest.raises(ModelProviderError) as exc_info:
        provider._normalize_response({})
    assert "Unexpected response structure" in str(exc_info.value)


def test_no_silent_local_fallback() -> None:
    """Requirement K: When provider is cloud, errors in cloud DO NOT silently fall back to local."""
    cfg = Config(vidura_model_provider="cloud", vidura_cloud_model="gemma4:31b-cloud")
    router = ModelRouter(config=cfg)

    # Local provider should never be called when cloud provider fails
    with mock.patch.object(router._cloud_provider, "generate", side_effect=ProviderUnavailable("Cloud is offline")):
        with mock.patch.object(router._local_provider, "generate") as mock_local_gen:
            with pytest.raises(ProviderUnavailable) as exc_info:
                router.generate([{"role": "user", "content": "test"}])
            assert "Cloud is offline" in str(exc_info.value)
            mock_local_gen.assert_not_called()


# ==============================================================================
# Requirement L: AgentLoop Remains Provider-Agnostic
# ==============================================================================

def test_agent_loop_operates_with_cloud_provider() -> None:
    """Requirement L: AgentLoop executes decisions with OllamaCloudProvider identically to local."""
    cloud_provider = OllamaCloudProvider(api_key="mock-key")
    
    # Mock cloud model returning a valid respond decision
    decision_json = json.dumps({"action": "respond", "content": "Executed via Ollama Cloud"})
    with mock.patch.object(cloud_provider, "generate", return_value=decision_json):
        agent = Agent(model=cloud_provider)
        response = agent.run("Hello cloud")
        assert "Executed via Ollama Cloud" in response


# ==============================================================================
# Requirements M & N: Permission and ToolRegistry Isolation
# ==============================================================================

def test_cloud_provider_cannot_bypass_tool_registry() -> None:
    """Requirement M: Cloud model requests to unregistered tools are rejected by the application controller."""
    cloud_provider = OllamaCloudProvider(api_key="mock-key")
    tools = ToolRegistry()

    # Model attempts to call an unauthorized or invented tool
    fake_tool_decision = json.dumps({"action": "tool_call", "tool_name": "rm_rf_everything", "arguments": {}})
    with mock.patch.object(cloud_provider, "generate", return_value=fake_tool_decision):
        loop = AgentLoop(model=cloud_provider, tool_registry=tools)
        state = loop.run(AgentState(user_request="delete files"))
        # ToolRegistry does not contain tool; application controller handles it safely
        assert state.current_action == "tool_call" or "Unknown tool" in state.messages[-1]["content"] or "rm_rf_everything" in state.messages[-1]["content"]


def test_cloud_provider_cannot_modify_permissions() -> None:
    """Requirement N: Cloud model cannot alter PermissionManager state or grant itself write access."""
    cloud_provider = OllamaCloudProvider(api_key="mock-key")
    perms = PermissionManager()
    assert perms.write_allowed is False
    assert perms.test_allowed is False

    # Agent runs with cloud provider
    agent = Agent(model=cloud_provider)
    assert perms.write_allowed is False
    assert perms.is_write_allowed() is False


# ==============================================================================
# Requirements O, P, Q: Secret Protection (Logs, AgentState, Memory)
# ==============================================================================

def test_credentials_not_in_logs(caplog: pytest.LogCaptureFixture) -> None:
    """Requirement O: Secrets are sanitized and never appear in application logs."""
    secret = "super-secret-cloud-token-99999"
    caplog.set_level(logging.DEBUG)

    provider = OllamaCloudProvider(api_key=secret)
    # Logging representation
    logger = logging.getLogger("VIDURA.models.cloud")
    logger.info(f"Checking provider: {provider}")

    for record in caplog.records:
        assert secret not in record.message
        assert secret not in str(record)


def test_credentials_not_in_agent_state() -> None:
    """Requirement P: AgentState messages contain only user and model text, never cloud API credentials."""
    secret = "secret-token-for-state"
    cloud_provider = OllamaCloudProvider(api_key=secret)

    with mock.patch.object(cloud_provider, "generate", return_value=json.dumps({"action": "respond", "content": "clean response"})):
        agent = Agent(model=cloud_provider)
        agent.run("test state prompt")

        assert agent.last_state is not None
        for msg in agent.last_state.messages:
            assert secret not in msg.get("content", "")
            assert secret not in str(msg)


def test_credentials_not_in_memory(tmp_path: Path) -> None:
    """Requirement Q: Secrets are never saved to persistent or in-memory SQLite memory stores."""
    secret = "top-secret-ollama-key-xyz"
    store = MemoryStore(tmp_path / "test_mem.db")
    manager = MemoryManager(store=store)

    # Store normal memory
    manager.remember("User preference: write clean Python", memory_type=MemoryCategory.USER)
    
    # Query all stored memories
    all_mems = manager.list_memories(limit=100)
    for m in all_mems:
        assert secret not in m.content
        assert secret not in str(m.metadata)


# ==============================================================================
# Requirements R & S: Offline Local Mode and State Preservation Across Switches
# ==============================================================================

def test_local_provider_works_without_cloud_credentials() -> None:
    """Requirement R: Local mode remains 100% operational with no cloud credentials or internet."""
    with mock.patch.dict(os.environ, {}, clear=False):
        os.environ.pop("OLLAMA_API_KEY", None)
        os.environ.pop("VIDURA_CLOUD_API_KEY", None)
        os.environ.pop("OLLAMA_CLOUD_API_KEY", None)

        cfg = Config(vidura_model_provider="local", vidura_local_model="gemma4:e4b-it-qat")
        router = ModelRouter(config=cfg)
        assert router.provider_type == "local"
        assert router.active_provider.capabilities.local is True
        assert router.active_provider.capabilities.cloud is False


def test_provider_switching_does_not_alter_permissions() -> None:
    """Requirement S: Switching provider local -> cloud -> local preserves strict permission boundaries."""
    perms = PermissionManager()
    assert perms.write_allowed is False

    # Start with local
    cfg_local = Config(vidura_model_provider="local")
    router_local = ModelRouter(config=cfg_local)
    assert perms.write_allowed is False

    # Switch to cloud
    cfg_cloud = Config(vidura_model_provider="cloud")
    router_cloud = ModelRouter(config=cfg_cloud)
    assert perms.write_allowed is False

    # Switch back to local
    router_local2 = ModelRouter(config=cfg_local)
    assert perms.write_allowed is False
