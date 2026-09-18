"""VIDURA Cloud Failure + Local Fallback Tests (Phase 9.5).

Verifies safe, deterministic, application-level fallback from Ollama Cloud
(gemma4:31b-cloud) to Local Ollama (gemma4:e4b-it-qat) when transient failures occur.
"""
from __future__ import annotations

import os
from unittest.mock import MagicMock, patch
import pytest

from config import Config
from models.base import ModelProvider, ProviderCapabilities
from models.errors import (
    ModelProviderError,
    ProviderUnavailable,
    ProviderConfigurationError,
    ProviderAuthenticationError,
    ProviderTimeout,
    ProviderRequestError,
    CombinedProviderError,
)
from models.routing import (
    FailureCategory,
    RoutingDecision,
    RoutingMode,
    TaskComplexity,
    ReasonCode,
    classify_cloud_failure,
    is_fallback_eligible,
)
from models.router import ModelRouter
from developer.models import (
    CodeChangeProposal,
    DeveloperGenerationResult,
    DeveloperExecutionResult,
)
from developer.generator import DeveloperCodeGenerator as DeveloperGenerator
from agent.state import AgentState
from agent.loop import AgentLoop


class MockProvider(ModelProvider):
    """Configurable mock provider for fallback testing."""

    def __init__(self, name: str, model_id: str, is_cloud: bool = False) -> None:
        self._name = name
        self._model_id = model_id
        self._is_cloud = is_cloud
        self.generate_mock = MagicMock(return_value="mock response")

    @property
    def provider_name(self) -> str:
        return self._name

    @property
    def model_name(self) -> str:
        return self._model_id

    @property
    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            streaming=True,
            tool_calling=True,
            system_prompts=True,
            code_specialized=True,
            cloud=self._is_cloud,
        )

    def generate(self, messages, **kwargs) -> str:
        return self.generate_mock(messages, **kwargs)

    def chat(self, messages, **kwargs) -> str:
        return self.generate(messages, **kwargs)


@pytest.fixture
def local_mock():
    return MockProvider("LocalProvider", "gemma4:e4b-it-qat", is_cloud=False)


@pytest.fixture
def cloud_mock():
    return MockProvider("OllamaCloudProvider", "gemma4:31b-cloud", is_cloud=True)


# ==============================================================================
# 1. Error Classification & Fallback Eligibility Tests
# ==============================================================================

class TestFailureClassification:
    """Tests for classify_cloud_failure and is_fallback_eligible."""

    def test_timeout_is_transient_and_eligible(self):
        err = ProviderTimeout("Request timed out after 60s")
        assert classify_cloud_failure(err) == FailureCategory.TRANSIENT
        assert is_fallback_eligible(err) is True

    def test_request_error_is_transient_and_eligible(self):
        err = ProviderRequestError("Connection reset by peer")
        assert classify_cloud_failure(err) == FailureCategory.TRANSIENT
        assert is_fallback_eligible(err) is True

    def test_unavailable_server_error_is_transient_and_eligible(self):
        err = ProviderUnavailable("503 Service Temporarily Unavailable")
        assert classify_cloud_failure(err) == FailureCategory.TRANSIENT
        assert is_fallback_eligible(err) is True

    def test_unavailable_missing_config_is_configuration_and_ineligible(self):
        err = ProviderUnavailable("Cloud API key is missing or not configured.")
        assert classify_cloud_failure(err) == FailureCategory.CONFIGURATION
        assert is_fallback_eligible(err) is False

    def test_authentication_error_is_ineligible(self):
        err = ProviderAuthenticationError("Invalid API key (HTTP 401)")
        assert classify_cloud_failure(err) == FailureCategory.AUTHENTICATION
        assert is_fallback_eligible(err) is False

    def test_configuration_error_is_ineligible(self):
        err = ProviderConfigurationError("Unauthorized model identifier")
        assert classify_cloud_failure(err) == FailureCategory.CONFIGURATION
        assert is_fallback_eligible(err) is False

    def test_value_error_is_invalid_request_and_ineligible(self):
        err = ValueError("Malformed JSON payload")
        assert classify_cloud_failure(err) == FailureCategory.INVALID_REQUEST
        assert is_fallback_eligible(err) is False

    def test_permission_error_is_security_and_ineligible(self):
        err = PermissionError("Access denied to target resource")
        assert classify_cloud_failure(err) == FailureCategory.SECURITY
        assert is_fallback_eligible(err) is False

    def test_unknown_runtime_error_is_ineligible(self):
        err = RuntimeError("Unexpected internal error")
        assert classify_cloud_failure(err) == FailureCategory.UNKNOWN
        assert is_fallback_eligible(err) is False


# ==============================================================================
# 2. Configuration & Default Behavior Tests
# ==============================================================================

class TestFallbackConfiguration:
    """Tests for fallback configuration flags and defaults."""

    def test_fallback_disabled_by_default(self, local_mock, cloud_mock):
        config = Config()
        assert config.vidura_cloud_fallback_enabled is False

        router = ModelRouter(
            config=config,
            routing_mode="cloud",
            local_provider=local_mock,
            cloud_provider=cloud_mock,
        )
        assert router.fallback_enabled is False

    def test_fallback_enabled_via_env_var(self, monkeypatch, local_mock, cloud_mock):
        monkeypatch.setenv("VIDURA_CLOUD_FALLBACK_ENABLED", "true")
        config = Config()
        assert config.vidura_cloud_fallback_enabled is True

        router = ModelRouter(
            config=config,
            routing_mode="cloud",
            local_provider=local_mock,
            cloud_provider=cloud_mock,
        )
        assert router.fallback_enabled is True

    def test_fallback_enabled_via_router_argument(self, local_mock, cloud_mock):
        config = Config()
        router = ModelRouter(
            config=config,
            routing_mode="cloud",
            local_provider=local_mock,
            cloud_provider=cloud_mock,
            fallback_enabled=True,
        )
        assert router.fallback_enabled is True


# ==============================================================================
# 3. Router Fallback Execution Tests
# ==============================================================================

class TestRouterFallbackExecution:
    """Tests for ModelRouter.generate and fallback routing behaviors."""

    def test_cloud_success_no_fallback_needed(self, local_mock, cloud_mock):
        cloud_mock.generate_mock.return_value = "Cloud output"
        router = ModelRouter(
            routing_mode="cloud",
            local_provider=local_mock,
            cloud_provider=cloud_mock,
            fallback_enabled=True,
        )

        res = router.generate("Hello world")
        assert res == "Cloud output"
        assert cloud_mock.generate_mock.call_count == 1
        assert local_mock.generate_mock.call_count == 0

        decision = router.last_decision
        assert decision.actual_provider == "cloud"
        assert decision.actual_model == "gemma4:31b-cloud"
        assert decision.fallback_used is False
        assert decision.fallback_reason is None

    def test_fallback_disabled_raises_cloud_timeout(self, local_mock, cloud_mock):
        cloud_mock.generate_mock.side_effect = ProviderTimeout("Cloud gateway timeout")
        router = ModelRouter(
            routing_mode="cloud",
            local_provider=local_mock,
            cloud_provider=cloud_mock,
            fallback_enabled=False,
        )

        with pytest.raises(ProviderTimeout) as exc_info:
            router.generate("Hello world")

        assert "Cloud gateway timeout" in str(exc_info.value)
        assert cloud_mock.generate_mock.call_count == 1
        assert local_mock.generate_mock.call_count == 0

    def test_fallback_enabled_on_cloud_timeout_succeeds_locally(self, local_mock, cloud_mock):
        cloud_mock.generate_mock.side_effect = ProviderTimeout("Cloud timed out after 60s")
        local_mock.generate_mock.return_value = "Local fallback output"

        router = ModelRouter(
            routing_mode="cloud",
            local_provider=local_mock,
            cloud_provider=cloud_mock,
            fallback_enabled=True,
        )

        res = router.generate("Hello world")
        assert res == "Local fallback output"
        assert cloud_mock.generate_mock.call_count == 1
        assert local_mock.generate_mock.call_count == 1

        decision = router.last_decision
        assert decision.requested_provider == "cloud"
        assert decision.requested_model == "gemma4:31b-cloud"
        assert decision.actual_provider == "local"
        assert decision.actual_model == "gemma4:e4b-it-qat"
        assert decision.fallback_used is True
        assert "timed out" in decision.fallback_reason
        assert decision.cloud_error == "Cloud timed out after 60s"

    def test_fallback_enabled_on_cloud_network_drop(self, local_mock, cloud_mock):
        cloud_mock.generate_mock.side_effect = ProviderRequestError("Connection dropped by remote server")
        local_mock.generate_mock.return_value = "Local recovered response"

        router = ModelRouter(
            routing_mode="cloud",
            local_provider=local_mock,
            cloud_provider=cloud_mock,
            fallback_enabled=True,
        )

        res = router.generate("Design request")
        assert res == "Local recovered response"
        assert router.last_decision.fallback_used is True
        assert router.last_decision.actual_provider == "local"

    def test_authentication_error_never_falls_back_even_when_enabled(self, local_mock, cloud_mock):
        cloud_mock.generate_mock.side_effect = ProviderAuthenticationError("Invalid API key 401")
        router = ModelRouter(
            routing_mode="cloud",
            local_provider=local_mock,
            cloud_provider=cloud_mock,
            fallback_enabled=True,
        )

        with pytest.raises(ProviderAuthenticationError):
            router.generate("Sensitive request")

        assert cloud_mock.generate_mock.call_count == 1
        assert local_mock.generate_mock.call_count == 0

    def test_configuration_error_never_falls_back_even_when_enabled(self, local_mock, cloud_mock):
        cloud_mock.generate_mock.side_effect = ProviderConfigurationError("Unauthorized model name")
        router = ModelRouter(
            routing_mode="cloud",
            local_provider=local_mock,
            cloud_provider=cloud_mock,
            fallback_enabled=True,
        )

        with pytest.raises(ProviderConfigurationError):
            router.generate("Config test")

        assert cloud_mock.generate_mock.call_count == 1
        assert local_mock.generate_mock.call_count == 0

    def test_combined_provider_error_when_both_cloud_and_local_fail(self, local_mock, cloud_mock):
        cloud_mock.generate_mock.side_effect = ProviderTimeout("Cloud timeout 504")
        local_mock.generate_mock.side_effect = ProviderUnavailable("Local Ollama daemon unreachable")

        router = ModelRouter(
            routing_mode="cloud",
            local_provider=local_mock,
            cloud_provider=cloud_mock,
            fallback_enabled=True,
        )

        with pytest.raises(CombinedProviderError) as exc_info:
            router.generate("Both fail request")

        combined_err = exc_info.value
        assert isinstance(combined_err.cloud_error, ProviderTimeout)
        assert isinstance(combined_err.local_error, ProviderUnavailable)
        assert "Cloud provider failed" in str(combined_err)
        assert "Fallback local provider also failed" in str(combined_err)

        # Ensure exactly 1 cloud attempt and 1 local attempt occurred
        assert cloud_mock.generate_mock.call_count == 1
        assert local_mock.generate_mock.call_count == 1

    def test_local_route_does_not_trigger_fallback(self, local_mock, cloud_mock):
        local_mock.generate_mock.return_value = "Normal local response"
        router = ModelRouter(
            routing_mode="local",
            local_provider=local_mock,
            cloud_provider=cloud_mock,
            fallback_enabled=True,
        )

        res = router.generate("Simple request")
        assert res == "Normal local response"
        assert local_mock.generate_mock.call_count == 1
        assert cloud_mock.generate_mock.call_count == 0
        assert router.last_decision.fallback_used is False


# ==============================================================================
# 4. Developer Models & Generator Metadata Tests
# ==============================================================================

class TestDeveloperFallbackPropagation:
    """Tests metadata propagation to proposals, results, and execution results."""

    def test_code_change_proposal_fallback_metadata_serialization(self):
        prop = CodeChangeProposal(
            operation="modify_file",
            target_file="src/calculator.py",
            proposed_content="# updated",
            provider="local",
            model="gemma4:e4b-it-qat",
            fallback_used=True,
            fallback_reason="Cloud timed out after 60s",
        )
        d = prop.to_dict()
        assert d["fallback_used"] is True
        assert d["fallback_reason"] == "Cloud timed out after 60s"
        assert d["provider"] == "local"

        restored = CodeChangeProposal.from_dict(d)
        assert restored.fallback_used is True
        assert restored.fallback_reason == "Cloud timed out after 60s"
        assert restored.provider == "local"

    def test_developer_generation_result_fallback_serialization(self):
        prop = CodeChangeProposal(
            operation="modify_file",
            target_file="src/utils.py",
            proposed_content="# mod",
            fallback_used=True,
            fallback_reason="Transient network drop",
        )
        res = DeveloperGenerationResult(
            target_file="src/utils.py",
            operation="modify_file",
            proposal=prop,
            fallback_used=True,
            fallback_reason="Transient network drop",
            provider="local",
            model="gemma4:e4b-it-qat",
        )
        d = res.to_dict()
        assert d["fallback_used"] is True
        assert d["fallback_reason"] == "Transient network drop"

        restored = DeveloperGenerationResult.from_dict(d)
        assert restored.fallback_used is True
        assert restored.fallback_reason == "Transient network drop"

    def test_developer_execution_result_post_init_inherits_fallback(self):
        prop = CodeChangeProposal(
            operation="modify_file",
            target_file="src/api.py",
            provider="local",
            model="gemma4:e4b-it-qat",
            fallback_used=True,
            fallback_reason="Cloud timeout",
        )
        exec_res = DeveloperExecutionResult(
            success=True,
            status="applied_and_verified",
            stage="completed",
            target_file="src/api.py",
            operation="modify_file",
            proposal=prop,
        )
        # Should inherit from proposal via __post_init__
        assert exec_res.fallback_used is True
        assert exec_res.fallback_reason == "Cloud timeout"
        assert exec_res.provider == "local"
        assert exec_res.model == "gemma4:e4b-it-qat"

        d = exec_res.to_dict()
        assert d["fallback_used"] is True
        assert d["fallback_reason"] == "Cloud timeout"

        restored = DeveloperExecutionResult.from_dict(d)
        assert restored.fallback_used is True
        assert restored.fallback_reason == "Cloud timeout"

    def test_developer_generator_sets_fallback_metadata_on_proposal(self, tmp_path, local_mock, cloud_mock):
        cloud_mock.generate_mock.side_effect = ProviderTimeout("Cloud gateway timeout")
        # Local mock returns valid json proposal response
        local_mock.generate_mock.return_value = '{"proposed_content": "def test(): pass", "rationale": "Fallback test"}'

        router = ModelRouter(
            routing_mode="cloud",
            local_provider=local_mock,
            cloud_provider=cloud_mock,
            fallback_enabled=True,
        )

        test_file = tmp_path / "mod.py"
        test_file.write_text("def test(): pass\n")

        gen = DeveloperGenerator(workspace_root=tmp_path, model=router)
        proposal = gen.generate_proposal(
            request="Add new logic",
            target_file="mod.py",
            operation="modify_file",
        )

        assert proposal.is_valid is True
        assert proposal.provider == "local"
        assert proposal.model == "gemma4:e4b-it-qat"
        assert proposal.fallback_used is True
        assert "gateway timeout" in proposal.fallback_reason


# ==============================================================================
# 5. Agent Loop & State Integration Tests
# ==============================================================================

class TestAgentStateFallbackTracking:
    """Tests that AgentState records fallback metadata during agent loop execution."""

    def test_agent_loop_records_fallback_decision(self, local_mock, cloud_mock):
        cloud_mock.generate_mock.side_effect = ProviderTimeout("504 Gateway Timeout")
        local_mock.generate_mock.return_value = '{"action": "respond", "content": "Fallback answer"}'

        router = ModelRouter(
            routing_mode="cloud",
            local_provider=local_mock,
            cloud_provider=cloud_mock,
            fallback_enabled=True,
        )

        loop = AgentLoop(model=router)
        state = loop.run(AgentState(user_request="Explain the architecture"))

        assert state.requested_provider == "cloud"
        assert state.actual_provider == "local"
        assert state.fallback_used is True
        assert "504 Gateway Timeout" in state.fallback_reason
        assert "Fallback answer" in (state.final_response or "")
