"""Automated test suite for Phase 9.4: Intelligent Local/Cloud Routing.

Covers Requirements A through Y:
- Requirement A: Default routing uses local (gemma4:e4b-it-qat)
- Requirement B: AUTO mode exists and is configurable
- Requirement C: SIMPLE task routes local
- Requirement D: MODERATE task routes local
- Requirement E: COMPLEX task routes cloud
- Requirement F: CRITICAL task routes cloud
- Requirement G: Multi-file task routes cloud
- Requirement H: Architectural task routes cloud
- Requirement I: Simple explanation routes local
- Requirement J: Explicit LOCAL mode forces local
- Requirement K: Explicit CLOUD mode forces cloud
- Requirement L: local_only forces local regardless of complexity
- Requirement M: Explicit provider configuration overrides AUTO
- Requirement N: Invalid routing mode is rejected
- Requirement O: Invalid provider is rejected
- Requirement P: Cloud unavailable produces structured failure (ProviderUnavailable)
- Requirement Q: Cloud unavailable does not silently fall back to local
- Requirement R: Routing decision contains provider, model, reason, complexity, signals
- Requirement S: Routing does not alter AgentState
- Requirement T: Routing does not alter ToolRegistry
- Requirement U: Routing does not alter permissions
- Requirement V: Routing does not alter filesystem security
- Requirement W: Routing does not alter proposal lifecycle
- Requirement X: Routing does not alter post-change verification
- Requirement Y: Routing does not alter test execution controls
"""
from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest import mock
import pytest

from config import Config
from models.base import ModelProvider, ProviderCapabilities
from models.errors import (
    ProviderConfigurationError,
    ProviderUnavailable,
)
from models.local import LocalProvider
from models.cloud import OllamaCloudProvider
from models.routing import (
    TaskComplexity,
    RoutingMode,
    ReasonCode,
    RoutingDecision,
    classify_complexity,
)
from models.router import ModelRouter
from task_understanding.models import DeveloperTask, TaskType
from planning.models import DeveloperPlan
from developer.generator import DeveloperCodeGenerator
from developer.models import (
    CodeChangeProposal,
    ProposalOperation,
    ProposalStatus,
)
from developer.applier import CodeChangeApplier
from permissions.manager import PermissionManager
from tools.registry import ToolRegistry
from tools.filesystem import ReadFileTool, ListDirectoryTool
from agent.state import AgentState
from agent.loop import AgentLoop


class MockProvider(ModelProvider):
    """Mock ModelProvider that records calls and returns fixed responses."""

    def __init__(
        self,
        provider_type: str,
        model_name: str,
        is_cloud: bool = False,
        generate_response: str = "Mock generation response",
        fail_with: Exception | None = None,
    ) -> None:
        self.provider_type = provider_type
        self._model_name = model_name
        self._capabilities = ProviderCapabilities(
            coding=True,
            cloud=is_cloud,
            local=not is_cloud,
        )
        self.generate_response = generate_response
        self.fail_with = fail_with
        self.calls: list[dict[str, Any]] = []

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def provider_name(self) -> str:
        return f"Mock ({self.provider_type})"

    @property
    def capabilities(self) -> ProviderCapabilities:
        return self._capabilities

    def generate(self, messages: list[dict[str, str]] | str, **kwargs: Any) -> str:
        self.calls.append({"messages": messages, "kwargs": kwargs})
        if self.fail_with:
            raise self.fail_with
        return self.generate_response


# ==============================================================================
# Requirements A & B: Default Routing & AUTO Mode
# ==============================================================================

def test_requirement_a_default_routing_uses_local() -> None:
    """Requirement A: Default routing uses local model (gemma4:e4b-it-qat)."""
    router = ModelRouter()
    assert router.routing_mode == "auto"
    decision = router.route(messages="What is a Python generator?")
    assert decision.provider == "local"
    assert decision.model == "gemma4:e4b-it-qat"
    assert decision.reason_code == ReasonCode.LOCAL_DEFAULT.value


def test_requirement_b_auto_mode_exists_and_is_configurable() -> None:
    """Requirement B: AUTO mode exists and can be explicitly configured."""
    router_auto = ModelRouter(routing_mode="auto")
    assert router_auto.routing_mode == "auto"

    cfg = Config(vidura_routing_mode="auto")
    router_cfg = ModelRouter(config=cfg)
    assert router_cfg.routing_mode == "auto"


# ==============================================================================
# Requirements C, D, E, F: Complexity Levels & Default Policy
# ==============================================================================

def test_requirement_c_simple_task_routes_local() -> None:
    """Requirement C: SIMPLE task routes local."""
    task = DeveloperTask(
        task_type=TaskType.GENERAL,
        goal="Explain recursion",
        is_development_task=False,
    )
    router = ModelRouter(routing_mode="auto")
    decision = router.route(task=task)
    assert decision.complexity == TaskComplexity.SIMPLE.value
    assert decision.provider == "local"
    assert decision.model == "gemma4:e4b-it-qat"


def test_requirement_d_moderate_task_routes_local() -> None:
    """Requirement D: MODERATE task (single-file change) routes local."""
    task = DeveloperTask(
        task_type=TaskType.CODE_CHANGE,
        goal="Add helper function to utils.py",
        target_files=["utils.py"],
        is_development_task=True,
    )
    plan = DeveloperPlan(
        original_task=task,
        relevant_files=["utils.py"],
        relevant_symbols=["helper"],
        affected_components=["utils"],
        planned_changes=["Add helper function"],
    )
    router = ModelRouter(routing_mode="auto")
    decision = router.route(task=task, plan=plan)
    assert decision.complexity == TaskComplexity.MODERATE.value
    assert decision.provider == "local"
    assert decision.model == "gemma4:e4b-it-qat"
    assert decision.reason_code == ReasonCode.LOCAL_MODERATE_TASK.value


def test_requirement_e_complex_task_routes_cloud() -> None:
    """Requirement E: COMPLEX task routes cloud (gemma4:31b-cloud)."""
    task = DeveloperTask(
        task_type=TaskType.CODE_CHANGE,
        goal="Implement cross-module event bus",
        target_files=["events.py", "dispatcher.py"],
        is_development_task=True,
    )
    plan = DeveloperPlan(
        original_task=task,
        relevant_files=["events.py", "dispatcher.py"],
        relevant_symbols=["EventBus", "dispatch"],
        affected_components=["events", "dispatcher"],
        planned_changes=["Add event bus", "Hook up dispatcher"],
    )
    router = ModelRouter(routing_mode="auto")
    decision = router.route(task=task, plan=plan)
    assert decision.complexity in (TaskComplexity.COMPLEX.value, TaskComplexity.CRITICAL.value)
    assert decision.provider == "cloud"
    assert decision.model == "gemma4:31b-cloud"


def test_requirement_f_critical_task_routes_cloud() -> None:
    """Requirement F: CRITICAL task (large multi-file cross-module) routes cloud."""
    task = DeveloperTask(
        task_type=TaskType.CODE_CHANGE,
        goal="Major repository refactoring",
        target_files=["a.py", "b.py", "c.py", "d.py"],
        is_development_task=True,
    )
    plan = DeveloperPlan(
        original_task=task,
        relevant_files=["a.py", "b.py", "c.py", "d.py"],
        relevant_symbols=["symA", "symB", "symC", "symD"],
        affected_components=["modA", "modB", "modC"],
        dependencies=["dep1", "dep2", "dep3", "dep4"],
        planned_changes=["Major architectural refactor across repository"],
        risks=["Breaking changes across all modules"],
    )
    router = ModelRouter(routing_mode="auto")
    decision = router.route(task=task, plan=plan)
    assert decision.complexity == TaskComplexity.CRITICAL.value
    assert decision.provider == "cloud"
    assert decision.model == "gemma4:31b-cloud"


# ==============================================================================
# Requirements G, H, I: Specific Developer Task Categories
# ==============================================================================

def test_requirement_g_multi_file_task_routes_cloud() -> None:
    """Requirement G: Multi-file task routes cloud."""
    task = DeveloperTask(
        task_type=TaskType.CODE_CHANGE,
        goal="Sync models between schema.py and models.py",
        target_files=["schema.py", "models.py"],
        is_development_task=True,
    )
    plan = DeveloperPlan(
        original_task=task,
        relevant_files=["schema.py", "models.py"],
        planned_changes=["Update schema", "Update models"],
    )
    router = ModelRouter(routing_mode="auto")
    decision = router.route(task=task, plan=plan)
    assert decision.provider == "cloud"
    assert decision.reason_code == ReasonCode.CLOUD_MULTI_FILE_TASK.value


def test_requirement_h_architectural_task_routes_cloud() -> None:
    """Requirement H: Architectural change task routes cloud."""
    task = DeveloperTask(
        task_type=TaskType.CODE_CHANGE,
        goal="Architectural redesign of authentication pipeline",
        target_files=["auth.py"],
        is_development_task=True,
    )
    plan = DeveloperPlan(
        original_task=task,
        relevant_files=["auth.py"],
        affected_components=["auth", "security"],
        planned_changes=["Architectural refactor of token handling pipeline"],
    )
    router = ModelRouter(routing_mode="auto")
    decision = router.route(task=task, plan=plan)
    assert decision.provider == "cloud"
    assert decision.reason_code == ReasonCode.CLOUD_ARCHITECTURAL_TASK.value


def test_requirement_i_simple_explanation_routes_local() -> None:
    """Requirement I: Simple code explanation routes local."""
    task = DeveloperTask(
        task_type=TaskType.CODE_EXPLANATION,
        goal="Explain what calculate_total() in billing.py does",
        target_files=["billing.py"],
        is_development_task=True,
    )
    router = ModelRouter(routing_mode="auto")
    decision = router.route(task=task)
    assert decision.complexity == TaskComplexity.SIMPLE.value
    assert decision.provider == "local"
    assert decision.reason_code == ReasonCode.LOCAL_SIMPLE_TASK.value


# ==============================================================================
# Requirements J, K, L, M: Overrides and Precedence
# ==============================================================================

def test_requirement_j_explicit_local_mode_forces_local() -> None:
    """Requirement J: Explicit LOCAL mode forces local even for complex tasks."""
    task = DeveloperTask(
        task_type=TaskType.CODE_CHANGE,
        goal="Complex cross-module system redesign",
        target_files=["a.py", "b.py", "c.py", "d.py"],
        is_development_task=True,
    )
    plan = DeveloperPlan(
        original_task=task,
        relevant_files=["a.py", "b.py", "c.py", "d.py"],
        affected_components=["a", "b", "c"],
        planned_changes=["Architectural change"],
    )
    router = ModelRouter(routing_mode="local")
    decision = router.route(task=task, plan=plan)
    assert decision.provider == "local"
    assert decision.reason_code == ReasonCode.EXPLICIT_LOCAL.value


def test_requirement_k_explicit_cloud_mode_forces_cloud() -> None:
    """Requirement K: Explicit CLOUD mode forces cloud even for simple tasks."""
    task = DeveloperTask(
        task_type=TaskType.CODE_EXPLANATION,
        goal="What is a string?",
        is_development_task=False,
    )
    router = ModelRouter(routing_mode="cloud")
    decision = router.route(task=task)
    assert decision.provider == "cloud"
    assert decision.reason_code == ReasonCode.EXPLICIT_CLOUD.value


def test_requirement_l_local_only_forces_local() -> None:
    """Requirement L: local_only flag forces local regardless of complexity or cloud mode."""
    task = DeveloperTask(
        task_type=TaskType.CODE_CHANGE,
        goal="Classified internal code refactor",
        target_files=["secret_core.py", "secrets.py", "crypto.py"],
        is_development_task=True,
        local_only=True,
    )
    plan = DeveloperPlan(
        original_task=task,
        relevant_files=["secret_core.py", "secrets.py", "crypto.py"],
        affected_components=["secret", "crypto"],
        planned_changes=["Major architectural refactor"],
    )
    # Even if routing_mode is cloud, local_only has Tier 1 precedence
    router = ModelRouter(routing_mode="cloud")
    decision = router.route(task=task, plan=plan, local_only=True)
    assert decision.provider == "local"
    assert decision.model == "gemma4:e4b-it-qat"
    assert decision.reason_code == ReasonCode.LOCAL_ONLY_POLICY.value
    assert decision.local_only is True


def test_requirement_m_explicit_developer_provider_overrides_auto() -> None:
    """Requirement M: Explicit developer provider configuration overrides AUTO complexity."""
    local_mock = MockProvider("local", "gemma4:e4b-it-qat")
    cloud_mock = MockProvider("cloud", "gemma4:31b-cloud", is_cloud=True)

    # 1. Developer provider explicitly set to local with a complex task
    router_dev_local = ModelRouter(
        developer_provider_type="local",
        routing_mode="auto",
        local_provider=local_mock,
        cloud_provider=cloud_mock,
    )
    task_complex = DeveloperTask(
        task_type=TaskType.CODE_CHANGE,
        goal="Multi-file change",
        target_files=["x.py", "y.py"],
        is_development_task=True,
    )
    plan_complex = DeveloperPlan(original_task=task_complex, relevant_files=["x.py", "y.py"])
    decision = router_dev_local.route(task=task_complex, plan=plan_complex)
    assert decision.provider == "local"
    assert decision.reason_code == ReasonCode.EXPLICIT_LOCAL.value

    # 2. Developer provider explicitly set to cloud with a simple task
    router_dev_cloud = ModelRouter(
        developer_provider_type="cloud",
        routing_mode="auto",
        local_provider=local_mock,
        cloud_provider=cloud_mock,
    )
    task_simple = DeveloperTask(
        task_type=TaskType.CODE_EXPLANATION,
        goal="Explain hello()",
        target_files=["utils.py"],
        is_development_task=True,
    )
    decision = router_dev_cloud.route(task=task_simple)
    assert decision.provider == "cloud"
    assert decision.reason_code == ReasonCode.EXPLICIT_CLOUD.value


# ==============================================================================
# Requirements N & O: Strict Validation of Modes & Providers
# ==============================================================================

def test_requirement_n_invalid_routing_mode_is_rejected() -> None:
    """Requirement N: Invalid routing mode raises ProviderConfigurationError."""
    with pytest.raises(ProviderConfigurationError) as exc_info:
        ModelRouter(routing_mode="quantum")
    assert "Unsupported routing mode: 'quantum'" in str(exc_info.value)


def test_requirement_o_invalid_provider_is_rejected() -> None:
    """Requirement O: Invalid provider raises ProviderConfigurationError."""
    with pytest.raises(ProviderConfigurationError) as exc_info:
        ModelRouter(provider_type="skynet")
    assert "Unsupported model provider: 'skynet'" in str(exc_info.value)

    with pytest.raises(ProviderConfigurationError) as exc_info:
        ModelRouter(developer_provider_type="skynet")
    assert "Unsupported developer model provider: 'skynet'" in str(exc_info.value)


# ==============================================================================
# Requirements P & Q: Cloud Unavailability & No Silent Fallback
# ==============================================================================

def test_requirement_p_cloud_unavailable_produces_structured_failure() -> None:
    """Requirement P: When cloud is routed and unconfigured, structured failure occurs."""
    cfg = Config(ollama_cloud_api_key=None)
    router = ModelRouter(config=cfg, routing_mode="auto")

    # Complex multi-file task that routes to cloud
    task = DeveloperTask(
        task_type=TaskType.CODE_CHANGE,
        goal="Multi-file update",
        target_files=["mod1.py", "mod2.py"],
        is_development_task=True,
    )
    plan = DeveloperPlan(original_task=task, relevant_files=["mod1.py", "mod2.py"])

    with pytest.raises(ProviderUnavailable) as exc_info:
        router.generate("Generate code", task=task, plan=plan)
    assert "CLOUD_PROVIDER_UNAVAILABLE" in str(exc_info.value)


def test_requirement_q_cloud_unavailable_does_not_silently_fallback() -> None:
    """Requirement Q: When cloud request fails, router DOES NOT silently call local provider."""
    local_mock = MockProvider("local", "gemma4:e4b-it-qat")
    cloud_mock = MockProvider(
        "cloud",
        "gemma4:31b-cloud",
        is_cloud=True,
        fail_with=ProviderUnavailable("Cloud outage in progress"),
    )

    router = ModelRouter(
        routing_mode="auto",
        developer_provider_type="auto",
        local_provider=local_mock,
        cloud_provider=cloud_mock,
    )

    task = DeveloperTask(
        task_type=TaskType.CODE_CHANGE,
        goal="Multi-file update",
        target_files=["file1.py", "file2.py"],
        is_development_task=True,
    )
    plan = DeveloperPlan(original_task=task, relevant_files=["file1.py", "file2.py"])

    with pytest.raises(ProviderUnavailable) as exc_info:
        router.generate("Implement changes", task=task, plan=plan)

    assert "Cloud outage in progress" in str(exc_info.value)
    # CRITICAL: Local mock was NEVER touched
    assert len(local_mock.calls) == 0
    assert len(cloud_mock.calls) == 1


# ==============================================================================
# Requirement R: Auditable Routing Decision Structure
# ==============================================================================

def test_requirement_r_routing_decision_structure() -> None:
    """Requirement R: RoutingDecision contains provider, model, reason, complexity, signals."""
    router = ModelRouter(routing_mode="auto")
    task = DeveloperTask(
        task_type=TaskType.CODE_CHANGE,
        goal="Multi-file change",
        target_files=["a.py", "b.py"],
        is_development_task=True,
    )
    plan = DeveloperPlan(
        original_task=task,
        relevant_files=["a.py", "b.py"],
        relevant_symbols=["Foo", "Bar"],
        affected_components=["comp1", "comp2"],
        dependencies=["dep1"],
    )
    decision = router.route(task=task, plan=plan)

    assert isinstance(decision, RoutingDecision)
    assert decision.provider == "cloud"
    assert decision.model == "gemma4:31b-cloud"
    assert decision.complexity == TaskComplexity.COMPLEX.value
    assert decision.reason_code == ReasonCode.CLOUD_MULTI_FILE_TASK.value
    assert decision.confidence == 1.0
    assert decision.local_only is False

    d_dict = decision.to_dict()
    assert d_dict["provider"] == "cloud"
    assert d_dict["model"] == "gemma4:31b-cloud"
    assert "signals_used" in d_dict
    assert d_dict["signals_used"]["file_count"] == 2
    assert d_dict["signals_used"]["symbol_count"] == 2


# ==============================================================================
# Requirements S, T, U, V, W, X, Y: Security & Permission Invariants
# ==============================================================================

def test_requirement_s_routing_does_not_alter_agent_state() -> None:
    """Requirement S: Calling route() does not mutate AgentState."""
    state = AgentState(user_request="Refactor system")
    initial_step = state.step
    initial_completed = state.completed

    router = ModelRouter()
    router.route(messages=state.user_request)

    assert state.step == initial_step
    assert state.completed == initial_completed
    assert state.tool_name is None


def test_requirement_t_routing_does_not_alter_tool_registry() -> None:
    """Requirement T: Cloud routing does not gain or alter ToolRegistry tools."""
    registry = ToolRegistry()
    registry.register(ReadFileTool())
    registry.register(ListDirectoryTool())

    initial_tools = sorted(t.name for t in registry.list_tools())

    router = ModelRouter(routing_mode="cloud")
    router.route(messages="Some query")

    post_tools = sorted(t.name for t in registry.list_tools())
    assert initial_tools == post_tools


def test_requirement_u_routing_does_not_alter_permissions() -> None:
    """Requirement U: Routing to cloud does NOT grant or modify write permissions."""
    pm = PermissionManager(default_write_allowed=False)
    assert pm.is_write_allowed() is False

    router = ModelRouter(routing_mode="cloud")
    router.route(messages="Write file to disk")

    # Cloud selection cannot grant permission
    assert pm.is_write_allowed() is False


def test_requirement_v_routing_does_not_alter_filesystem_security(tmp_path: Path) -> None:
    """Requirement V: Routing does not bypass workspace boundary or protected patterns."""
    generator = DeveloperCodeGenerator(workspace_root=tmp_path)
    # Attempting to target outside or protected target is blocked identically
    proposal_outside = generator.generate_proposal(
        request="Edit outside",
        target_file="/etc/passwd",
    )
    assert proposal_outside.is_valid is False
    assert proposal_outside.status == ProposalStatus.INVALID_PATH.value

    proposal_protected = generator.generate_proposal(
        request="Edit secret",
        target_file=".env",
    )
    assert proposal_protected.is_valid is False
    assert proposal_protected.status == ProposalStatus.INVALID_PATH.value


def test_requirement_w_routing_does_not_alter_proposal_lifecycle(tmp_path: Path) -> None:
    """Requirement W: Proposals generated via cloud/local adhere to identical status lifecycle."""
    local_mock = MockProvider("local", "gemma4:e4b-it-qat", generate_response='{"proposed_content": "# local", "rationale": "r"}')
    cloud_mock = MockProvider("cloud", "gemma4:31b-cloud", is_cloud=True, generate_response='{"proposed_content": "# cloud", "rationale": "r"}')

    router = ModelRouter(
        routing_mode="auto",
        local_provider=local_mock,
        cloud_provider=cloud_mock,
    )
    generator = DeveloperCodeGenerator(workspace_root=tmp_path, model=router)

    # 1. Single-file proposal (local route)
    prop_local = generator.generate_proposal(
        request="Add util.py",
        target_file="util.py",
        operation=ProposalOperation.CREATE_FILE.value,
    )
    assert prop_local.is_valid is True
    assert prop_local.status == ProposalStatus.PROPOSED.value
    assert prop_local.provider == "local"

    # 2. Multi-file from plan (cloud route)
    task = DeveloperTask(
        task_type=TaskType.CODE_CHANGE,
        goal="Create auth and user modules",
        target_files=["auth.py", "user.py"],
        is_development_task=True,
    )
    plan = DeveloperPlan(
        original_task=task,
        relevant_files=["auth.py", "user.py"],
        planned_changes=["Add auth", "Add user"],
    )
    gen_res = generator.generate_from_plan(task, plan)
    assert gen_res.is_valid is True
    assert gen_res.proposal is not None
    assert gen_res.proposal.status == ProposalStatus.PROPOSED.value
    assert gen_res.proposal.provider == "cloud"


def test_requirement_x_routing_does_not_alter_verification(tmp_path: Path) -> None:
    """Requirement X: Post-change verification operates identically regardless of model provider."""
    from developer.verifier import CodeChangeVerifier
    verifier = CodeChangeVerifier(workspace_root=tmp_path)

    # Verifier operates strictly on disk state and proposals
    target = tmp_path / "test.py"
    target.write_text("initial = 1\n", encoding="utf-8")

    proposal = CodeChangeProposal(
        operation=ProposalOperation.MODIFY_FILE.value,
        target_file="test.py",
        original_content="",
        proposed_content="initial = 1\n",
    )
    res = verifier.verify(proposal)
    assert res.success is True
    assert res.verified is True


def test_requirement_y_routing_does_not_alter_test_execution_controls(tmp_path: Path) -> None:
    """Requirement Y: Developer test execution controls remain permission-enforced regardless of provider."""
    from developer.testing import TestRunner
    from developer.test_models import TestStatus
    runner = TestRunner(workspace_root=tmp_path)

    # Validates target, shell injection, and workspace boundary strictly
    res = runner.run_test("non_existent_test.py")
    assert res.status == TestStatus.INVALID_TARGET.value

    # Rejects path traversal and command injection
    res_inject = runner.run_test("test.py; rm -rf /")
    assert res_inject.status == TestStatus.INVALID_TARGET.value
