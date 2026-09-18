import json
from typing import Any
import pytest

from config import Config
from models.base import BaseLLMProvider, ModelProvider, ProviderCapabilities
from models.errors import (
    ProviderUnavailable,
    ProviderConfigurationError,
)
from models.local import LocalProvider
from models.cloud import OllamaCloudProvider
from models.router import ModelRouter
from agent import Agent, AgentLoop, AgentState
from permissions.manager import PermissionManager
from tools.registry import ToolRegistry
from tools.filesystem import ReadFileTool, ListDirectoryTool


# ==============================================================================
# Requirement A: ModelProvider Interface
# ==============================================================================

def test_model_provider_interface() -> None:
    """Requirement A: ModelProvider exposes common interface, capabilities, and provider-agnostic methods."""
    class DummyProvider(ModelProvider):
        @property
        def model_name(self) -> str:
            return "dummy-model"

        @property
        def provider_name(self) -> str:
            return "Dummy Provider"

        def generate(self, messages: list[dict[str, str]], **kwargs: Any) -> str:
            return "dummy response"

    provider = DummyProvider()
    assert isinstance(provider, BaseLLMProvider)
    assert provider.model_name == "dummy-model"
    assert provider.provider_name == "Dummy Provider"
    assert isinstance(provider.capabilities, ProviderCapabilities)
    assert provider.chat([{"role": "user", "content": "hi"}]) == "dummy response"


# ==============================================================================
# Requirements B & C: Local Provider Creation & Default Model
# ==============================================================================

def test_local_provider_creation_and_defaults() -> None:
    """Requirements B & C: Local provider creation and default to gemma4:e4b-it-qat."""
    provider = LocalProvider()
    assert isinstance(provider, ModelProvider)
    assert provider.model_name == "gemma4:e4b-it-qat"
    assert provider.provider_name == "Ollama (local)"
    assert provider.host == "http://localhost:11434"
    assert provider.capabilities.local is True
    assert provider.capabilities.cloud is False
    assert provider.capabilities.tool_calling is True
    assert provider.capabilities.coding is True


def test_local_provider_preserves_ollama_host() -> None:
    """Requirement B: Local provider preserves custom OLLAMA_HOST."""
    provider = LocalProvider(host="http://custom-host:11434", model="gemma4:e4b-it-qat")
    assert provider.host == "http://custom-host:11434"


# ==============================================================================
# Requirements D & E: Cloud Provider Creation & Default Model
# ==============================================================================

def test_cloud_provider_creation_and_defaults() -> None:
    """Requirements D & E: Cloud provider creation and default to gemma4:31b-cloud."""
    cloud = OllamaCloudProvider()
    assert isinstance(cloud, ModelProvider)
    assert cloud.model_name == "gemma4:31b-cloud"
    assert cloud.provider_name == "Ollama (cloud)"
    assert cloud.capabilities.cloud is True
    assert cloud.capabilities.local is False
    assert cloud.capabilities.reasoning is True
    assert cloud.capabilities.tool_calling is True
    assert cloud.is_configured is False


# ==============================================================================
# Requirements F & G: Router Provider Selection
# ==============================================================================

def test_router_selects_local_provider() -> None:
    """Requirement F: Router selects LocalProvider when provider is 'local'."""
    cfg = Config()
    router = ModelRouter(config=cfg, provider_type="local")
    assert router.provider_type == "local"
    assert isinstance(router.active_provider, LocalProvider)
    assert router.model_name == "gemma4:e4b-it-qat"
    assert router.capabilities.local is True
    assert router.capabilities.cloud is False


def test_router_selects_cloud_provider() -> None:
    """Requirement G: Router selects OllamaCloudProvider when provider is 'cloud'."""
    cfg = Config()
    router = ModelRouter(config=cfg, provider_type="cloud")
    assert router.provider_type == "cloud"
    assert isinstance(router.active_provider, OllamaCloudProvider)
    assert router.model_name == "gemma4:31b-cloud"
    assert router.capabilities.cloud is True
    assert router.capabilities.local is False


# ==============================================================================
# Requirements H & I: Strict Rejection of Invalid Providers and Models
# ==============================================================================

def test_router_rejects_invalid_provider() -> None:
    """Requirement H: Invalid provider identifier raises ProviderConfigurationError."""
    cfg = Config()
    with pytest.raises(ProviderConfigurationError) as exc_info:
        ModelRouter(config=cfg, provider_type="banana")
    assert "Unsupported model provider: 'banana'" in str(exc_info.value)


def test_router_rejects_invalid_local_model() -> None:
    """Requirement I: Unauthorized local model identifier raises ProviderConfigurationError."""
    cfg = Config()
    with pytest.raises(ProviderConfigurationError) as exc_info:
        ModelRouter(config=cfg, provider_type="local", local_model="unauthorized:70b")
    assert "Invalid local model identifier" in str(exc_info.value)


def test_router_rejects_invalid_cloud_model() -> None:
    """Requirement I: Unauthorized cloud model identifier raises ProviderConfigurationError."""
    cfg = Config()
    with pytest.raises(ProviderConfigurationError) as exc_info:
        ModelRouter(config=cfg, provider_type="cloud", cloud_model="gpt-5-unapproved")
    assert "Invalid cloud model identifier" in str(exc_info.value)


# ==============================================================================
# Requirements J & K: Local Mode Offline & Cloud Structured Error
# ==============================================================================

def test_local_mode_works_without_cloud_credentials() -> None:
    """Requirement J: Local mode is fully operable without cloud credentials or internet."""
    cfg = Config(ollama_cloud_api_key=None)
    router = ModelRouter(config=cfg, provider_type="local")
    assert router.provider_type == "local"
    assert router.active_provider.capabilities.local is True


def test_cloud_mode_without_credentials_raises_error() -> None:
    """Requirement K: Cloud mode without credentials returns structured error without silent local fallback."""
    cfg = Config(ollama_cloud_api_key=None)
    router = ModelRouter(config=cfg, provider_type="cloud")
    
    # check_connection reports unconfigured
    connected, msg = router.check_connection()
    assert connected is False
    assert "Missing OLLAMA_CLOUD_API_KEY" in msg

    # generate raises ProviderUnavailable with CLOUD_PROVIDER_UNAVAILABLE
    with pytest.raises(ProviderUnavailable) as exc_info:
        router.generate([{"role": "user", "content": "test"}])
    assert "CLOUD_PROVIDER_UNAVAILABLE" in str(exc_info.value)


# ==============================================================================
# Requirement L: AgentLoop Depends on ModelProvider
# ==============================================================================

def test_agent_loop_depends_on_model_provider() -> None:
    """Requirement L: AgentLoop functions with ModelProvider/ModelRouter without knowing concrete provider."""
    class MockProvider(ModelProvider):
        @property
        def model_name(self) -> str:
            return "gemma4:e4b-it-qat"

        @property
        def provider_name(self) -> str:
            return "Mock Provider"

        def generate(self, messages: list[dict[str, str]], **kwargs: Any) -> str:
            return json.dumps({"action": "respond", "content": "Provider abstraction working"})

    mock_prov = MockProvider()
    agent = Agent(model=mock_prov)
    response = agent.run("Hello")
    assert response == "Provider abstraction working"


# ==============================================================================
# Requirements M, N, O: Provider Switching Preserves State & Permissions
# ==============================================================================

def test_provider_switching_preserves_agent_state() -> None:
    """Requirement M: Switching between providers does not reset or corrupt AgentState."""
    class CustomMockProvider(ModelProvider):
        def __init__(self, name: str) -> None:
            self._name = name

        @property
        def model_name(self) -> str:
            return "gemma4:e4b-it-qat"

        @property
        def provider_name(self) -> str:
            return self._name

        def generate(self, messages: list[dict[str, str]], **kwargs: Any) -> str:
            return json.dumps({"action": "respond", "content": f"Response from {self._name}"})

    prov1 = CustomMockProvider("Provider1")
    prov2 = CustomMockProvider("Provider2")

    loop = AgentLoop(model=prov1)
    state = AgentState(user_request="Test state preservation")
    state.messages.append({"role": "user", "content": "Step 1"})

    # Run step with Provider 1
    state1 = loop.run(state)
    assert state1.completed is True
    assert "Provider1" in state1.final_response

    # Swap model on loop to Provider 2 without altering AgentState
    loop.model = prov2
    state2 = AgentState(user_request="Followup step", messages=list(state1.messages))
    state2_res = loop.run(state2)
    assert state2_res.completed is True
    assert "Provider2" in state2_res.final_response
    assert any("Step 1" in m.get("content", "") for m in state2_res.messages)


def test_provider_switching_does_not_alter_permissions() -> None:
    """Requirement N: Permission system state is independent of model provider selection."""
    perm_mgr = PermissionManager(default_write_allowed=False)
    assert perm_mgr.is_write_allowed("test.py", "modify_file") is False

    # Simulate router initialized with local
    cfg_local = Config()
    router_local = ModelRouter(config=cfg_local, provider_type="local")
    assert perm_mgr.is_write_allowed("test.py", "modify_file") is False

    # Simulate router initialized with cloud
    cfg_cloud = Config()
    router_cloud = ModelRouter(config=cfg_cloud, provider_type="cloud")
    # Permissions remain identical: NO write permission granted simply because cloud was chosen
    assert perm_mgr.is_write_allowed("test.py", "modify_file") is False


def test_provider_switching_does_not_alter_tools() -> None:
    """Requirement O: Available tools in ToolRegistry are independent of model provider."""
    registry = ToolRegistry()
    registry.register(ReadFileTool())
    registry.register(ListDirectoryTool())

    tools_initial = [t.name for t in registry.list_tools()]

    # Router setup local
    router_local = ModelRouter(provider_type="local")
    tools_local = [t.name for t in registry.list_tools()]
    assert tools_local == tools_initial

    # Router setup cloud
    router_cloud = ModelRouter(provider_type="cloud")
    tools_cloud = [t.name for t in registry.list_tools()]
    assert tools_cloud == tools_initial


# ==============================================================================
# Requirement P: Model Cannot Dynamically Change Provider
# ==============================================================================

def test_model_cannot_dynamically_change_provider() -> None:
    """Requirement P: Model text or tool outputs cannot alter router configuration."""
    router = ModelRouter(provider_type="local")
    assert router.provider_type == "local"

    # Attempting to assign or modify private attributes or pass runtime directives fails
    with pytest.raises(AttributeError):
        router.provider_type = "cloud"

    assert router.provider_type == "local"


# ==============================================================================
# Requirement Q: Secrets Sanitization
# ==============================================================================

def test_cloud_configuration_does_not_leak_secrets() -> None:
    """Requirement Q: Cloud provider never exposes API keys in repr or error messages."""
    secret_key = "super_secret_ollama_key_998877"
    cloud_prov = OllamaCloudProvider(api_key=secret_key)
    
    repr_str = repr(cloud_prov)
    assert secret_key not in repr_str
    assert "api_key=<present>" in repr_str

    # String representation of router also does not leak credentials
    router = ModelRouter(provider_type="cloud", cloud_provider=cloud_prov)
    assert secret_key not in repr(router)
