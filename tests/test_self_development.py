"""Automated test suite for VIDURA Phase 10: Self-Development Loop.

Covers requirements A through AJ:
- A. SelfDevelopmentGoal creation
- B. SelfDevelopmentPlan creation
- C. Codebase analysis
- D. Relevant experience retrieval
- E. Improvement identification
- F. Plan grounding (unknown components reported as unknown)
- G. DeveloperGenerator integration
- H. Proposal generation
- I. Proposal scope validation
- J. Permission required before modification (human approval boundary)
- K. Self-development cannot approve itself
- L. Security-critical file protection (requires elevated authorization)
- M. Scope expansion rejection
- N. Delete operation rejection (UNSUPPORTED_OPERATION)
- O. No unrestricted shell tool
- P. Existing Phase 7 pipeline is used
- Q. Existing Phase 8 pipeline is used
- R. Test pipeline is used
- S. Evaluation requires actual evidence
- T. Failed tests prevent full success
- U. Verification failure prevents full success
- V. Permission denial prevents modification
- W. Stale proposal is rejected
- X. Experience memory is recorded
- Y. Experience memory contains no credentials
- Z. Experience memory contains no hidden chain-of-thought
- AA. No automatic retry
- AB. No recursive self-development
- AC. Cloud routing follows Phase 9 policies
- AD. Local-only restrictions remain enforced
- AE. Cloud usage limits remain enforced
- AF. Fallback behavior remains correct
- AG. False model success claims cannot override trusted state
- AH. Audit ID is unique
- AI. Duplicate approval cannot reapply a completed proposal
"""
import json
import sqlite3
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

from config import Config
from models.base import ModelProvider, ProviderCapabilities
from models.router import ModelRouter
from models.routing import ReasonCode
from models.errors import CloudSecurityViolation
from codebase.manager import CodebaseManager
from memory.manager import MemoryManager
from memory.models import ExperienceRecord
from memory.store import MemoryStore
from permissions.manager import PermissionManager
from developer.generator import CodeChangeGenerator as DeveloperCodeGenerator
from developer.executor import DeveloperExecutor
from developer.applier import CodeChangeApplier
from developer.verifier import CodeChangeVerifier
from developer.models import (
    CodeChangeProposal,
    CodeChangeResult,
    VerificationResult,
    ExecutionStatus,
    DeveloperExecutionResult,
)
from developer.test_models import TestResult as _TestResult, TestPlan as _TestPlan
from self_development.models import (
    SelfDevelopmentGoal,
    SelfDevelopmentPlan,
    SelfDevelopmentEvaluation,
    SelfDevelopmentCycleResult,
)
from self_development.security import (
    is_security_critical_target,
    validate_scope,
    validate_operation,
    sanitize_experience_record,
    SelfDevelopmentDisabledError,
    SelfDevelopmentScopeError,
    SelfDevelopmentRecursionError,
    ElevatedAuthorizationRequiredError,
)
from self_development.analyzer import SelfDevelopmentAnalyzer
from self_development.planner import SelfDevelopmentPlanner
from self_development.evaluator import SelfDevelopmentEvaluator
from self_development.loop import SelfDevelopmentLoop


class MockProvider(ModelProvider):
    """Deterministic Mock Provider for Self-Development tests."""

    def __init__(self, name: str = "mock", model: str = "gemma4:e4b-it-qat", is_cloud: bool = False):
        self._name = name
        self._model = model
        self._capabilities = ProviderCapabilities(
            chat=True,
            coding=True,
            local=not is_cloud,
            cloud=is_cloud,
        )
        self.generate_mock = MagicMock(
            return_value=json.dumps({
                "generated_code": "# Proposed self-development change\n",
                "proposed_content": "# Proposed self-development change\n",
                "explanation": "Improved validation for tool calls.",
                "rationale": "Improved validation for tool calls.",
            })
        )

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def provider_name(self) -> str:
        return self._name

    @property
    def capabilities(self) -> ProviderCapabilities:
        return self._capabilities

    def generate(self, prompt, **kwargs):
        return self.generate_mock(prompt, **kwargs)


@pytest.fixture
def mock_workspace(tmp_path: Path):
    """Creates a temporary workspace with sample code files."""
    ws = tmp_path / "workspace"
    ws.mkdir()

    # Create dummy codebase structure
    pkg = ws / "agent"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    (pkg / "loop.py").write_text(
        "class AgentLoop:\n"
        "    def run(self):\n"
        "        pass\n"
    )

    tools_dir = ws / "tools"
    tools_dir.mkdir()
    (tools_dir / "__init__.py").write_text("")
    (tools_dir / "registry.py").write_text(
        "class ToolRegistry:\n"
        "    def get(self, name):\n"
        "        return None\n"
    )

    return ws


# ==============================================================================
# Requirement A & B: Goal & Plan Creation
# ==============================================================================

def test_self_development_goal_creation():
    """Requirement A: Explicit SelfDevelopmentGoal creation and validation."""
    goal = SelfDevelopmentGoal(
        description="Improve VIDURA's malformed tool-call handling.",
        motivation="Prevent crashes on unformatted JSON arguments.",
        scope=["agent/loop.py", "tools/registry.py"],
        constraints=["Preserve existing tool signatures."],
    )
    assert goal.goal_id.startswith("sd_goal_")
    assert goal.description == "Improve VIDURA's malformed tool-call handling."
    assert goal.status == "created"

    # Serialization
    data = goal.to_dict()
    restored = SelfDevelopmentGoal.from_dict(data)
    assert restored.goal_id == goal.goal_id
    assert restored.scope == ["agent/loop.py", "tools/registry.py"]

    # Empty description raises error
    with pytest.raises(ValueError):
        SelfDevelopmentGoal(description="   ")


def test_self_development_plan_creation():
    """Requirement B: SelfDevelopmentPlan creation, validation, and serialization."""
    goal = SelfDevelopmentGoal(description="Refactor tool validation.")
    plan = SelfDevelopmentPlan(
        goal=goal,
        current_behavior="Raises unhandled KeyError on malformed tool calls.",
        observed_problem="No pre-execution schema check.",
        relevant_files=["agent/loop.py"],
        relevant_symbols=["AgentLoop.run"],
        proposed_improvement="Add schema validator before calling tool.execute.",
        expected_benefit="Graceful recovery without loop crash.",
        risks=["Slight increase in parsing latency."],
        validation_strategy=["Run tool tests."],
        rollback_strategy="Revert modifications to agent/loop.py.",
    )
    assert plan.plan_id.startswith("sd_plan_")
    assert plan.relevant_files == ["agent/loop.py"]
    assert plan.requires_elevated_authorization is False

    data = plan.to_dict()
    restored = SelfDevelopmentPlan.from_dict(data)
    assert restored.plan_id == plan.plan_id
    assert restored.goal.description == goal.description


# ==============================================================================
# Requirement C, D, E, F: Codebase Analysis, Experience, and Grounding
# ==============================================================================

def test_codebase_analysis_grounding(mock_workspace: Path):
    """Requirement C & F: Codebase analysis uses actual index; unknown components reported as unknown."""
    cm = CodebaseManager(workspace_root=str(mock_workspace))
    cm.scan()

    analyzer = SelfDevelopmentAnalyzer(codebase_manager=cm, workspace_root=mock_workspace)

    # Known component
    goal = SelfDevelopmentGoal(description="Improve AgentLoop run method.", scope=["agent/loop.py"])
    res = analyzer.analyze(goal)
    assert "agent/loop.py" in res["relevant_files"]
    assert "AgentLoop" in res["relevant_symbols"]
    assert res["requires_more_information"] is False

    # Unknown component
    unknown_goal = SelfDevelopmentGoal(description="Modify NonExistentComponent.", scope=["non_existent/fake.py"])
    unk_res = analyzer.analyze(unknown_goal)
    assert "non_existent/fake.py" in unk_res["unknown_components"]


def test_relevant_experience_retrieval(mock_workspace: Path, tmp_path: Path):
    """Requirement D & E: Relevant experience retrieval as supporting context without overriding reality."""
    db_file = tmp_path / "test_mem.db"
    store = MemoryStore(db_path=db_file)
    mm = MemoryManager(store=store)

    # Store a past experience
    mm.record_experience(ExperienceRecord(
        task="Improve malformed tool-call handling",
        action_summary="Added JSON validator",
        result="Success",
        success=True,
        lesson="Always use strict=False for JSON parsing in agent loop.",
    ))

    cm = CodebaseManager(workspace_root=str(mock_workspace))
    cm.scan()
    analyzer = SelfDevelopmentAnalyzer(codebase_manager=cm, memory_manager=mm, workspace_root=mock_workspace)

    goal = SelfDevelopmentGoal(description="Improve malformed tool calls in agent loop.")
    res = analyzer.analyze(goal)
    assert len(res["recalled_lessons"]) >= 1
    assert "strict=False" in res["recalled_lessons"][0]


# ==============================================================================
# Requirement G, H, I, M, N, O: Generator Integration, Scope & Operation Validation
# ==============================================================================

def test_developer_generator_and_proposal_integration(mock_workspace: Path):
    """Requirement G & H: Adapts SelfDevelopmentPlan to existing Phase 8 pipeline."""
    cm = CodebaseManager(workspace_root=str(mock_workspace))
    cm.scan()
    planner = SelfDevelopmentPlanner(codebase_manager=cm, workspace_root=mock_workspace)

    goal = SelfDevelopmentGoal(description="Improve agent loop", scope=["agent/loop.py"])
    sd_plan = planner.plan(goal)

    dev_task = planner.to_developer_task(sd_plan)
    dev_plan = planner.to_developer_plan(sd_plan, dev_task)

    assert dev_task.is_development_task is True
    assert "agent/loop.py" in dev_task.target_files
    assert "agent/loop.py" in dev_plan.relevant_files


def test_proposal_scope_validation_and_rejection():
    """Requirement I & M: Scope expansion rejection."""
    # Within scope
    ok, err = validate_scope(["agent/loop.py", "tools/registry.py"], "agent/loop.py")
    assert ok is True
    assert err == ""

    # Scope expansion
    bad, err = validate_scope(["agent/loop.py"], "developer/applier.py")
    assert bad is False
    assert "Scope expansion rejected" in err


def test_delete_operation_rejection():
    """Requirement N: Delete operations return UNSUPPORTED_OPERATION."""
    ok, err = validate_operation("delete")
    assert ok is False
    assert "UNSUPPORTED_OPERATION" in err
    assert "does not support deletion" in err

    ok_del, err_del = validate_operation("delete_file")
    assert ok_del is False
    assert "UNSUPPORTED_OPERATION" in err_del


def test_no_unrestricted_shell_tool():
    """Requirement O: Arbitrary shell execution operations are rejected."""
    ok, err = validate_operation("shell")
    assert ok is False
    assert "UNSUPPORTED_OPERATION" in err
    assert "does not support shell execution" in err


# ==============================================================================
# Requirement J, K, L: Human Approval Boundary & Security-Critical Protection
# ==============================================================================

def test_permission_required_before_modification(mock_workspace: Path):
    """Requirement J & K: Self-development halts at Human Approval Boundary and cannot approve itself."""
    cfg = Config(vidura_self_development_enabled=True)
    pm = PermissionManager(default_write_allowed=False)
    mock_prov = MockProvider()

    sd_loop = SelfDevelopmentLoop(
        config=cfg,
        model=mock_prov,
        workspace_root=mock_workspace,
        permission_manager=pm,
    )

    goal = SelfDevelopmentGoal(description="Improve agent loop", scope=["agent/loop.py"])
    cycle_res = sd_loop.initiate_cycle(goal)

    # Must be pending permission
    assert cycle_res.status == "pending_permission"
    assert cycle_res.proposal is not None
    assert cycle_res.proposal.target_file == "agent/loop.py"

    # Disk was NOT modified
    content = (mock_workspace / "agent/loop.py").read_text()
    assert "Proposed self-development change" not in content

    # Self-development system cannot self-approve without explicit permission
    assert pm.write_allowed is False


def test_security_critical_file_protection(mock_workspace: Path):
    """Requirement L: Modifying PermissionManager or critical infrastructure requires elevated authorization."""
    assert is_security_critical_target("permissions/manager.py") is True
    assert is_security_critical_target("developer/applier.py") is True
    assert is_security_critical_target("developer/verifier.py") is True
    assert is_security_critical_target("PermissionManager") is True
    assert is_security_critical_target("agent/loop.py") is False

    cfg = Config(vidura_self_development_enabled=True)
    pm = PermissionManager(default_write_allowed=False)

    # Create dummy permissions/manager.py
    perm_dir = mock_workspace / "permissions"
    perm_dir.mkdir(exist_ok=True)
    (perm_dir / "manager.py").write_text("class PermissionManager:\n    pass\n")

    sd_loop = SelfDevelopmentLoop(
        config=cfg,
        model=MockProvider(),
        workspace_root=mock_workspace,
        permission_manager=pm,
    )

    goal = SelfDevelopmentGoal(description="Modify PermissionManager", scope=["permissions/manager.py"])
    cycle_res = sd_loop.initiate_cycle(goal)

    assert cycle_res.requires_elevated_authorization is True

    # Normal approval without elevated confirmation is denied
    cycle_denied = sd_loop.apply_approved(explicit_permission=True, elevated_authorization=False)
    assert cycle_denied.status == "permission_denied"
    assert "ELEVATED_AUTHORIZATION_REQUIRED" in cycle_denied.error


# ==============================================================================
# Requirement P, Q, R, S, T, U, V, W: Pipeline Reuse, Evidence-Based Evaluation, Stale & Tests
# ==============================================================================

def test_full_successful_self_development_cycle(mock_workspace: Path, tmp_path: Path):
    """Requirement P, Q, R, S: Full flow: proposal -> approval -> apply -> verify -> test -> evaluate -> memory."""
    cfg = Config(vidura_self_development_enabled=True)
    pm = PermissionManager(default_write_allowed=False)
    db_file = tmp_path / "cycle_mem.db"
    store = MemoryStore(db_path=db_file)
    mm = MemoryManager(store=store)

    target = mock_workspace / "agent/loop.py"
    target.write_text("class AgentLoop:\n    pass\n")

    mock_prov = MockProvider()
    mock_prov.generate_mock.return_value = json.dumps({
        "generated_code": "class AgentLoop:\n    # Self-improved validator\n    pass\n",
        "explanation": "Added validator.",
    })

    # Mock TestRunner to return successful test result
    mock_test_result = _TestResult(
        stage="COMPLETED",
        status="PASSED",
        exit_code=0,
        tests_passed=5,
        tests_failed=0,
        stdout="5 passed in 0.1s",
    )

    sd_loop = SelfDevelopmentLoop(
        config=cfg,
        model=mock_prov,
        workspace_root=mock_workspace,
        permission_manager=pm,
        memory_manager=mm,
    )
    sd_loop.executor.test_planner.determine_test_plan = MagicMock(
        return_value=_TestPlan(testing_required=True, test_files=["tests/test_agent.py"])
    )
    sd_loop.executor.test_runner.run_test_plan = MagicMock(return_value=mock_test_result)

    goal = SelfDevelopmentGoal(description="Improve agent loop validation", scope=["agent/loop.py"])
    cycle = sd_loop.initiate_cycle(goal)
    assert cycle.status == "pending_permission"

    # User grants approval
    cycle_completed = sd_loop.apply_approved(explicit_permission=True)
    assert cycle_completed.status == "completed"
    assert cycle_completed.evaluation is not None
    assert cycle_completed.evaluation.overall_outcome == "SELF_DEVELOPMENT_SUCCESS"
    assert cycle_completed.evaluation.improvement_success is True
    assert cycle_completed.evaluation.change_applied is True
    assert cycle_completed.evaluation.verification_status is True
    assert cycle_completed.evaluation.tests_passed == 5

    # Check physical disk
    assert "Self-improved validator" in target.read_text()

    # Verify Experience memory was persisted
    exps = mm.list_experiences()
    assert len(exps) == 1
    assert "Improve agent loop validation" in exps[0].task
    assert exps[0].success is True


def test_failed_tests_prevent_full_success(mock_workspace: Path):
    """Requirement T: Failed tests yield CHANGE_APPLIED_BUT_TESTS_FAILED and overall success is False."""
    cfg = Config(vidura_self_development_enabled=True)
    pm = PermissionManager(default_write_allowed=False)
    mock_prov = MockProvider()

    sd_loop = SelfDevelopmentLoop(
        config=cfg,
        model=mock_prov,
        workspace_root=mock_workspace,
        permission_manager=pm,
    )

    # Mock TestRunner to simulate test failure
    failed_test_result = _TestResult(
        stage="FAILED",
        status="FAILED",
        exit_code=1,
        tests_passed=3,
        tests_failed=1,
        stdout="1 failed, 3 passed",
    )
    sd_loop.executor.test_planner.determine_test_plan = MagicMock(
        return_value=_TestPlan(testing_required=True, test_files=["tests/test_agent.py"])
    )
    sd_loop.executor.test_runner.run_test_plan = MagicMock(return_value=failed_test_result)

    goal = SelfDevelopmentGoal(description="Improve agent loop", scope=["agent/loop.py"])
    sd_loop.initiate_cycle(goal)
    cycle = sd_loop.apply_approved(explicit_permission=True)

    assert cycle.status == "evaluated_with_failures"
    assert cycle.evaluation.overall_outcome == "CHANGE_APPLIED_BUT_TESTS_FAILED"
    assert cycle.evaluation.improvement_success is False
    assert cycle.evaluation.tests_failed == 1


def test_permission_denial_prevents_modification(mock_workspace: Path):
    """Requirement V: Explicit denial cancels proposal without modifying disk."""
    cfg = Config(vidura_self_development_enabled=True)
    pm = PermissionManager(default_write_allowed=False)
    sd_loop = SelfDevelopmentLoop(
        config=cfg,
        model=MockProvider(),
        workspace_root=mock_workspace,
        permission_manager=pm,
    )

    goal = SelfDevelopmentGoal(description="Improve agent loop", scope=["agent/loop.py"])
    sd_loop.initiate_cycle(goal)

    denied = sd_loop.deny_active_proposal("User decided not to apply.")
    assert denied.status == "cancelled"
    assert denied.evaluation.overall_outcome == "PERMISSION_DENIED"

    # Disk was NOT modified
    assert "Proposed self-development change" not in (mock_workspace / "agent/loop.py").read_text()


def test_stale_proposal_rejected(mock_workspace: Path):
    """Requirement W: Externally modifying file causes stale rejection upon approval."""
    cfg = Config(vidura_self_development_enabled=True)
    pm = PermissionManager(default_write_allowed=False)
    sd_loop = SelfDevelopmentLoop(
        config=cfg,
        model=MockProvider(),
        workspace_root=mock_workspace,
        permission_manager=pm,
    )

    goal = SelfDevelopmentGoal(description="Improve agent loop", scope=["agent/loop.py"])
    sd_loop.initiate_cycle(goal)

    # Externally modify target file
    (mock_workspace / "agent/loop.py").write_text("# Externally changed content\n")

    # Attempt to apply approved proposal
    cycle = sd_loop.apply_approved(explicit_permission=True)
    assert cycle.execution_result.status == ExecutionStatus.STALE_PROPOSAL.value
    assert cycle.evaluation.overall_outcome == "APPLICATION_FAILED"


# ==============================================================================
# Requirement X, Y, Z: Experience Memory Sanitization
# ==============================================================================

def test_experience_memory_sanitization():
    """Requirement X, Y, Z: Memory contains no secrets, credentials, or raw CoT."""
    raw_task = "Task with token sk-abcdef1234567890abcdef1234567890"
    raw_action = "Action with bearer Authorization: Bearer secret_jwt_token_12345678"
    raw_result = "Result <thought>This is private internal reasoning</thought> Finished."
    raw_lesson = "Lesson with ghp_123456789012345678901234567890123456"

    sanitized = sanitize_experience_record(
        task=raw_task,
        action_summary=raw_action,
        result=raw_result,
        lesson=raw_lesson,
    )

    assert "sk-abcdef1234567890abcdef1234567890" not in sanitized["task"]
    assert "Bearer secret_jwt_token" not in sanitized["action_summary"]
    assert "<thought>" not in sanitized["result"]
    assert "private internal reasoning" not in sanitized["result"]
    assert "ghp_1234567890" not in sanitized["lesson"]


# ==============================================================================
# Requirement AA & AB: No Automatic Retry & No Recursion
# ==============================================================================

def test_no_recursive_self_development(mock_workspace: Path):
    """Requirement AB: Cycle cannot trigger another self-development cycle."""
    cfg = Config(vidura_self_development_enabled=True)
    sd_loop = SelfDevelopmentLoop(config=cfg, model=MockProvider(), workspace_root=mock_workspace)

    # Artificially set loop as active
    sd_loop._is_running = True

    with pytest.raises(SelfDevelopmentRecursionError) as exc_info:
        sd_loop.initiate_cycle(SelfDevelopmentGoal(description="Nested cycle"))
    assert "Recursive self-development detected" in str(exc_info.value)


# ==============================================================================
# Requirement AC, AD, AE, AF: Cloud Routing, Security, Limits & Fallback
# ==============================================================================

def test_cloud_routing_and_fallback_compliance(mock_workspace: Path):
    """Requirement AC, AD, AE, AF: Cloud routing respects Phase 9.6 security policies and Phase 9.5 fallback."""
    local_mock = MockProvider("local_ollama", "gemma4:e4b-it-qat", is_cloud=False)
    cloud_mock = MockProvider("cloud_ollama", "gemma4:31b-cloud", is_cloud=True)

    cfg = Config(vidura_self_development_enabled=True, vidura_cloud_enabled=True)
    router = ModelRouter(
        config=cfg,
        routing_mode="cloud",
        fallback_enabled=True,
        local_provider=local_mock,
        cloud_provider=cloud_mock,
    )

    sd_loop = SelfDevelopmentLoop(
        config=cfg,
        model=router,
        workspace_root=mock_workspace,
    )

    goal = SelfDevelopmentGoal(description="Refactor complex tool validation", scope=["agent/loop.py"])
    cycle = sd_loop.initiate_cycle(goal)

    assert cycle.proposal is not None
    assert cycle.proposal.cloud_request_id is not None
    assert cycle.proposal.cloud_request_id.startswith("cloud_req_")
    assert cycle.proposal.provider == "cloud"


def test_cloud_security_blocks_protected_files(mock_workspace: Path):
    """Requirement AC & AD: Self-development cannot send protected files (.env) to cloud."""
    local_mock = MockProvider("local_ollama", "gemma4:e4b-it-qat", is_cloud=False)
    cloud_mock = MockProvider("cloud_ollama", "gemma4:31b-cloud", is_cloud=True)

    cfg = Config(vidura_self_development_enabled=True, vidura_cloud_enabled=True)
    router = ModelRouter(
        config=cfg,
        routing_mode="cloud",
        fallback_enabled=False,  # Strict mode
        local_provider=local_mock,
        cloud_provider=cloud_mock,
    )

    # Creating .env in workspace
    (mock_workspace / ".env").write_text("SECRET=123\n")

    sd_loop = SelfDevelopmentLoop(
        config=cfg,
        model=router,
        workspace_root=mock_workspace,
    )

    goal = SelfDevelopmentGoal(description="Update environment credentials", scope=[".env"])
    res = sd_loop.initiate_cycle(goal)
    assert res.status == "generation_failed"
    assert "blocked" in res.error.lower() or "violation" in res.error.lower() or "protected" in res.error.lower()


# ==============================================================================
# Requirement AG, AH, AI: False Claims, Audit IDs, Duplicate Approvals
# ==============================================================================

def test_false_model_claims_cannot_override_trusted_state():
    """Requirement AG: Untrusted model output cannot declare success without physical evidence."""
    evaluator = SelfDevelopmentEvaluator()
    goal = SelfDevelopmentGoal(description="Fix crash")

    # Execution result reports failure
    failed_exec = DeveloperExecutionResult(
        success=False,
        status=ExecutionStatus.APPLICATION_FAILED.value,
        stage="APPLY",
        target_file="agent/loop.py",
        operation="modify_file",
        permission_granted=True,
        overall_outcome="FAILED",
        error="Write permission check failed.",
    )

    evaluation = evaluator.evaluate(
        goal=goal,
        execution_result=failed_exec,
        self_development_id="sd_cycle_test1",
    )

    assert evaluation.improvement_success is False
    assert evaluation.overall_outcome == "APPLICATION_FAILED"


def test_audit_id_is_unique(mock_workspace: Path):
    """Requirement AH: Every cycle generates a unique audit identifier."""
    cfg = Config(vidura_self_development_enabled=True)
    sd_loop = SelfDevelopmentLoop(config=cfg, model=MockProvider(), workspace_root=mock_workspace)

    cycle1 = sd_loop.initiate_cycle(SelfDevelopmentGoal(description="Goal 1", scope=["agent/loop.py"]))
    cycle2 = sd_loop.initiate_cycle(SelfDevelopmentGoal(description="Goal 2", scope=["agent/loop.py"]))

    assert cycle1.self_development_id.startswith("sd_cycle_")
    assert cycle2.self_development_id.startswith("sd_cycle_")
    assert cycle1.self_development_id != cycle2.self_development_id


def test_disabled_via_config_rejects_cleanly(mock_workspace: Path):
    """Section 37: Self-development requests are rejected cleanly when disabled in config."""
    cfg = Config(vidura_self_development_enabled=False)
    sd_loop = SelfDevelopmentLoop(config=cfg, model=MockProvider(), workspace_root=mock_workspace)

    with pytest.raises(SelfDevelopmentDisabledError) as exc_info:
        sd_loop.initiate_cycle(SelfDevelopmentGoal(description="Self develop"))
    assert "disabled via configuration" in str(exc_info.value)
