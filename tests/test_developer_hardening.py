import json
from pathlib import Path
from typing import Any
import pytest

from permissions.manager import PermissionManager
from task_understanding.models import DeveloperTask, TaskType
from planning.models import DeveloperPlan
from planning.planner import DeveloperPlanner
from developer.models import (
    CodeChangeProposal,
    ExecutionStatus,
    ExecutionStage,
    DeveloperExecutionResult,
)
from developer.generator import CodeChangeGenerator, DeveloperCodeGenerator
from developer.applier import CodeChangeApplier
from developer.verifier import CodeChangeVerifier
from developer.executor import DeveloperExecutor
from developer.workflow import (
    DeveloperWorkflowState,
    DeveloperWorkflowStateMachine,
    MultiStageOutcome,
    WorkflowOutcome,
    InvalidStateTransitionError,
    TERMINAL_FAILURE_STATES,
)
from developer.test_models import (
    TestPlan,
    TestResult,
    TestSummary,
    TestCoverageStatus,
    TestStage,
    TestStatus,
)
from developer.testing import TestPlanner, TestRunner
from tools.developer import ProposeCodeChangeTool, ApplyCodeChangeTool, RunTestsTool
from tools.registry import ToolRegistry
from agent import AgentLoop, AgentState
from models.base import BaseLLMProvider

# Disable pytest test collection on imported classes starting with Test
TestPlan.__test__ = False
TestResult.__test__ = False
TestSummary.__test__ = False
TestPlanner.__test__ = False
TestRunner.__test__ = False
TestCoverageStatus.__test__ = False
TestStage.__test__ = False
TestStatus.__test__ = False


class FakeModelProvider(BaseLLMProvider):
    """Fake model provider for deterministic testing of AgentLoop."""

    def __init__(self, responses: list[str]) -> None:
        self.responses = responses
        self.call_count = 0

    @property
    def model_name(self) -> str:
        return "fake-model"

    @property
    def provider_name(self) -> str:
        return "Fake Provider"

    def generate(self, messages: list[dict[str, str]], **kwargs: Any) -> str:
        if not self.responses:
            return json.dumps({"action": "respond", "content": "Default response"})
        res = self.responses[min(self.call_count, len(self.responses) - 1)]
        self.call_count += 1
        return res


# ==============================================================================
# 1. AMBIGUOUS REQUEST & NEEDS_INFORMATION HARDENING
# ==============================================================================

def test_ambiguous_request_triggers_needs_information(tmp_path: Path) -> None:
    """Requirement A: Ambiguous requests result in requires_more_information=True and halt generation."""
    planner = DeveloperPlanner(workspace_root=tmp_path)
    task = DeveloperTask(
        task_type=TaskType.CODE_CHANGE,
        is_development_task=True,
        goal="Fix the bug",
        target_files=[],
        target_symbols=[],
    )

    plan = planner.plan(task)
    assert plan.requires_more_information is True
    assert plan.missing_information_reason != ""

    generator = DeveloperCodeGenerator(workspace_root=tmp_path)
    gen_res = generator.generate_from_plan(task, plan)
    assert gen_res.is_valid is False
    assert gen_res.validation_status == "requires_information"
    assert gen_res.proposal is None
    assert gen_res.generated_code == ""


def test_unknown_target_reference_triggers_needs_information(tmp_path: Path) -> None:
    """Requirement A/B: Unknown file references do not fabricate paths and require more information."""
    planner = DeveloperPlanner(workspace_root=tmp_path)
    task = DeveloperTask(
        task_type=TaskType.CODE_CHANGE,
        is_development_task=True,
        goal="Refactor the authentication logic",
        target_files=["nonexistent_auth_module.py"],
        target_symbols=["NonexistentClass"],
    )

    plan = planner.plan(task)
    assert plan.requires_more_information is True
    assert "nonexistent_auth_module.py" in plan.missing_information_reason or "NonexistentClass" in plan.missing_information_reason


# ==============================================================================
# 2. SCOPE CONTROL HARDENING
# ==============================================================================

def test_scope_control_rejects_out_of_scope_target(tmp_path: Path) -> None:
    """Requirement J: Modifications to files not listed in plan.relevant_files are rejected with requires_plan_update=True."""
    target_file = "service.py"
    (tmp_path / target_file).write_text("def run(): pass\n", encoding="utf-8")

    task = DeveloperTask(
        task_type=TaskType.CODE_CHANGE,
        is_development_task=True,
        goal="Add feature to service.py",
        target_files=[target_file],
    )
    # Plan restricts scope to only 'service.py'
    plan = DeveloperPlan(
        original_task=task,
        relevant_files=[target_file],
        requires_more_information=False,
    )

    perm_mgr = PermissionManager(default_write_allowed=False)
    applier = CodeChangeApplier(permission_manager=perm_mgr, workspace_root=tmp_path)
    executor = DeveloperExecutor(applier=applier, workspace_root=tmp_path)

    # Attempt to apply a change targeting an unapproved file 'unrelated.py'
    unrelated_proposal = CodeChangeProposal(
        target_file="unrelated.py",
        operation="create_file",
        proposed_content="print('injected')\n",
        description="Out of scope change",
    )
    executor.prepare_proposal(unrelated_proposal, plan=plan)
    perm_mgr.grant_write_permission()

    exec_result = executor.execute_proposal(proposal=unrelated_proposal, explicit_permission=True, plan=plan)
    assert exec_result.success is False
    assert exec_result.requires_plan_update is True
    assert "scope violation" in exec_result.summary.lower() or "scope" in exec_result.error.lower()
    assert not (tmp_path / "unrelated.py").exists()


# ==============================================================================
# 3. DUPLICATE APPROVAL PROTECTION
# ==============================================================================

def test_duplicate_approval_returns_cached_result(tmp_path: Path) -> None:
    """Requirement G: Approving an already applied proposal returns cached result without re-executing disk write."""
    target_file = "counter.py"
    (tmp_path / target_file).write_text("count = 0\n", encoding="utf-8")

    perm_mgr = PermissionManager(default_write_allowed=False)
    applier = CodeChangeApplier(permission_manager=perm_mgr, workspace_root=tmp_path)
    executor = DeveloperExecutor(applier=applier, workspace_root=tmp_path)

    proposal = CodeChangeProposal(
        target_file=target_file,
        operation="modify_file",
        proposed_content="count = 1\n",
        original_content="count = 0\n",
        description="Increment counter",
    )

    executor.prepare_proposal(proposal)
    perm_mgr.grant_write_permission()

    # First execution: applies to disk
    res1 = executor.execute_proposal(proposal=proposal, explicit_permission=True, run_tests=False)
    assert res1.success is True
    assert (tmp_path / target_file).read_text(encoding="utf-8") == "count = 1\n"

    # Second execution: duplicate approval
    res2 = executor.execute_proposal(proposal=proposal, explicit_permission=True, run_tests=False)
    assert res2.success is True
    assert "Change already applied" in res2.summary
    assert res2.proposal_id == res1.proposal_id


# ==============================================================================
# 4. MULTI-STAGE PRESERVED OUTCOMES
# ==============================================================================

def test_multi_stage_outcome_applied_but_tests_failed(tmp_path: Path) -> None:
    """Requirement H: When change is applied and verified on disk but tests fail, record CHANGE_APPLIED_BUT_TESTS_FAILED."""
    target_file = "calculator.py"
    (tmp_path / target_file).write_text("def add(a, b): return a - b\n", encoding="utf-8")

    perm_mgr = PermissionManager(default_write_allowed=False, default_test_allowed=False)
    applier = CodeChangeApplier(permission_manager=perm_mgr, workspace_root=tmp_path)

    class FailingTestRunner:
        def run_test_plan(self, plan: Any) -> TestResult:
            return TestResult(
                status=TestStatus.FAILED.value,
                stage=TestStage.TEST_FAILED.value,
                target="test_calculator.py",
                tests_attempted=1,
                tests_passed=0,
                tests_failed=1,
                error="AssertionError: add(1, 2) != 3",
            )

    class DummyTestPlanner:
        def determine_test_plan(self, proposal: Any, plan: Any = None) -> TestPlan:
            return TestPlan(
                test_files=["test_calculator.py"],
                testing_required=True,
                coverage_status=TestCoverageStatus.COVERED.value,
            )

    executor = DeveloperExecutor(
        applier=applier,
        workspace_root=tmp_path,
        test_planner=DummyTestPlanner(),
        test_runner=FailingTestRunner(),
    )

    proposal = CodeChangeProposal(
        target_file=target_file,
        operation="modify_file",
        proposed_content="def add(a, b): return a + b\n",
        original_content="def add(a, b): return a - b\n",
        description="Fix addition",
    )

    executor.prepare_proposal(proposal)
    perm_mgr.grant_write_permission()
    perm_mgr.grant_test_permission()

    exec_result = executor.execute_proposal(proposal=proposal, explicit_permission=True, run_tests=True)

    # The filesystem change was successfully applied and verified on disk
    assert (tmp_path / target_file).read_text(encoding="utf-8") == "def add(a, b): return a + b\n"
    assert exec_result.application_result.success is True
    assert exec_result.verification_result.success is True

    # Multi-stage outcome records CHANGE_APPLIED_BUT_TESTS_FAILED
    assert exec_result.overall_outcome == WorkflowOutcome.CHANGE_APPLIED_BUT_TESTS_FAILED.value
    assert "verified" in exec_result.summary.lower()
    assert "Tests:" in exec_result.summary
    assert exec_result.test_result.tests_failed == 1


# ==============================================================================
# 5. PERMISSION ISOLATION (WRITE vs TEST)
# ==============================================================================

def test_permission_isolation_write_allowed_test_denied(tmp_path: Path) -> None:
    """Requirement F: Write permission does NOT imply test execution permission."""
    target_file = "utils.py"
    (tmp_path / target_file).write_text("val = 10\n", encoding="utf-8")

    perm_mgr = PermissionManager(default_write_allowed=False, default_test_allowed=False)
    applier = CodeChangeApplier(permission_manager=perm_mgr, workspace_root=tmp_path)
    test_runner = TestRunner(workspace_root=tmp_path, permission_manager=perm_mgr)

    proposal = CodeChangeProposal(
        target_file=target_file,
        operation="modify_file",
        proposed_content="val = 20\n",
        original_content="val = 10\n",
        description="Update val",
    )

    plan = TestPlan(
        test_files=["test_utils.py"],
        testing_required=True,
        coverage_status=TestCoverageStatus.COVERED.value,
    )

    # Applier succeeds because write permission is granted
    applier.set_pending_proposal(proposal)
    perm_mgr.grant_write_permission()
    app_result = applier.apply_proposal(proposal)
    assert app_result.success is True

    # Test runner is rejected because test permission is denied
    test_result = test_runner.run_test_plan(plan)
    assert test_result.status == TestStatus.PERMISSION_DENIED.value
    assert test_result.tests_passed == 0


def test_permission_isolation_write_denied_test_allowed(tmp_path: Path) -> None:
    """Requirement F: Test permission does NOT imply write permission."""
    target_file = "utils.py"
    (tmp_path / target_file).write_text("val = 10\n", encoding="utf-8")

    perm_mgr = PermissionManager(default_write_allowed=False, default_test_allowed=False)
    applier = CodeChangeApplier(permission_manager=perm_mgr, workspace_root=tmp_path)

    proposal = CodeChangeProposal(
        target_file=target_file,
        operation="modify_file",
        proposed_content="val = 20\n",
        original_content="val = 10\n",
        description="Update val",
    )

    applier.set_pending_proposal(proposal)
    perm_mgr.grant_test_permission()  # only test permission granted, write remains denied
    app_result = applier.apply_proposal(proposal)
    assert app_result.success is False
    assert app_result.permission_granted is False
    assert "permission denied" in app_result.reason.lower()


# ==============================================================================
# 6. CONFLICTING PROPOSALS & IMMUTABILITY PROTECTION
# ==============================================================================

def test_conflicting_proposals_protection(tmp_path: Path) -> None:
    """Requirement E: Registering a new proposal supersedes older proposal; applying old proposal is rejected."""
    target_file = "data.py"
    (tmp_path / target_file).write_text("a = 1\n", encoding="utf-8")

    perm_mgr = PermissionManager(default_write_allowed=False)
    applier = CodeChangeApplier(permission_manager=perm_mgr, workspace_root=tmp_path)
    executor = DeveloperExecutor(applier=applier, workspace_root=tmp_path)

    prop1 = CodeChangeProposal(
        target_file=target_file,
        operation="modify_file",
        proposed_content="a = 2\n",
        original_content="a = 1\n",
        description="First proposal",
    )
    executor.prepare_proposal(prop1)

    prop2 = CodeChangeProposal(
        target_file=target_file,
        operation="modify_file",
        proposed_content="a = 3\n",
        original_content="a = 1\n",
        description="Second proposal",
    )
    executor.prepare_proposal(prop2)
    perm_mgr.grant_write_permission()

    # Attempt to apply outdated prop1
    res = executor.execute_proposal(proposal=prop1, explicit_permission=True)
    assert res.success is False
    assert "mismatch" in res.summary.lower() or "mismatch" in res.error.lower()
    # File content remains unchanged
    assert (tmp_path / target_file).read_text(encoding="utf-8") == "a = 1\n"


def test_proposal_immutability_protection(tmp_path: Path) -> None:
    """Requirement D: Mutating proposal content after generation is detected and rejected."""
    target_file = "secret.py"
    (tmp_path / target_file).write_text("key = 'initial'\n", encoding="utf-8")

    perm_mgr = PermissionManager(default_write_allowed=False)
    applier = CodeChangeApplier(permission_manager=perm_mgr, workspace_root=tmp_path)
    executor = DeveloperExecutor(applier=applier, workspace_root=tmp_path)

    prop = CodeChangeProposal(
        target_file=target_file,
        operation="modify_file",
        proposed_content="key = 'safe'\n",
        original_content="key = 'initial'\n",
        description="Safe change",
    )
    executor.prepare_proposal(prop)
    perm_mgr.grant_write_permission()

    # Mutate the caller-side proposal before execution
    mutated_prop = CodeChangeProposal(
        target_file=target_file,
        operation="modify_file",
        proposed_content="key = 'malicious_overwrite'\n",
        original_content="key = 'initial'\n",
        description="Tampered change",
        proposal_id=prop.proposal_id,
    )

    res = executor.execute_proposal(proposal=mutated_prop, explicit_permission=True)
    assert res.success is False
    assert "immutability" in res.summary.lower() or "mutated" in res.error.lower()
    assert (tmp_path / target_file).read_text(encoding="utf-8") == "key = 'initial'\n"


# ==============================================================================
# 7. TOOL CALL HARDENING
# ==============================================================================

def test_tool_call_hardening_unexpected_arguments(tmp_path: Path) -> None:
    """Requirement I: Developer tools strictly reject unknown or extra arguments."""
    generator = CodeChangeGenerator(workspace_root=tmp_path)
    prop_tool = ProposeCodeChangeTool(generator=generator)
    res = prop_tool.execute(
        request="Add something",
        target_file="module.py",
        unexpected_extra_arg="exploit",
    )
    assert res["success"] is False
    assert "Unexpected argument(s)" in res["error"]

    applier = CodeChangeApplier(workspace_root=tmp_path)
    apply_tool = ApplyCodeChangeTool(applier=applier)
    res_apply = apply_tool.execute(confirm_permission=True, bypass_permission=True)
    assert res_apply["success"] is False
    assert "Unexpected argument(s)" in res_apply["error"]

    runner = TestRunner(workspace_root=tmp_path)
    test_tool = RunTestsTool(test_runner=runner)
    res_test = test_tool.execute(test_target="test_module.py", execute_arbitrary_code=True)
    assert res_test["success"] is False
    assert "Unexpected argument(s)" in res_test["error"]


# ==============================================================================
# 8. FALSE-SUCCESS SUPPRESSION
# ==============================================================================

def test_false_success_suppression_file_modification(tmp_path: Path) -> None:
    """Requirement J: Natural language claim of file modification without verified execution is suppressed."""
    # Model claims file was successfully modified, but never called apply_code_change
    hallucinating_response = json.dumps({
        "action": "respond",
        "content": "I have successfully modified the file main.py with the new implementation."
    })
    provider = FakeModelProvider([hallucinating_response])
    agent = AgentLoop(model=provider)
    state = AgentState(user_request="Update main.py")

    final_state = agent.run(state)
    assert "🛑 No code change was applied" in final_state.final_response
    assert "successfully modified" not in final_state.final_response.lower()


def test_false_success_suppression_test_execution(tmp_path: Path) -> None:
    """Requirement J: Natural language claim of passing tests without runner execution is suppressed."""
    # Model claims tests passed, but no tests were executed
    hallucinating_response = json.dumps({
        "action": "respond",
        "content": "All tests passed! 15 passed in 0.2s."
    })
    provider = FakeModelProvider([hallucinating_response])
    agent = AgentLoop(model=provider)
    state = AgentState(user_request="Run the tests")

    final_state = agent.run(state)
    assert "🛑 Tests did not pass" in final_state.final_response


# ==============================================================================
# 9. STATE MACHINE LIFECYCLE & TERMINAL FAILURE FREEZING
# ==============================================================================

def test_state_machine_valid_forward_lifecycle() -> None:
    """Requirement L: State machine advances cleanly through valid lifecycle stages."""
    sm = DeveloperWorkflowStateMachine()
    assert sm.current_state == DeveloperWorkflowState.IDLE

    sm.transition_to(DeveloperWorkflowState.TASK_ANALYZED)
    sm.transition_to(DeveloperWorkflowState.PLAN_READY)
    sm.transition_to(DeveloperWorkflowState.GENERATION_READY)
    sm.transition_to(DeveloperWorkflowState.PROPOSAL_PENDING)
    sm.transition_to(DeveloperWorkflowState.PERMISSION_PENDING)
    sm.transition_to(DeveloperWorkflowState.APPROVED)
    sm.transition_to(DeveloperWorkflowState.APPLYING)
    sm.transition_to(DeveloperWorkflowState.APPLIED)
    sm.transition_to(DeveloperWorkflowState.VERIFYING)
    sm.transition_to(DeveloperWorkflowState.VERIFIED)
    sm.transition_to(DeveloperWorkflowState.TEST_PLANNED)
    sm.transition_to(DeveloperWorkflowState.TESTING)
    sm.transition_to(DeveloperWorkflowState.TEST_PASSED)
    sm.transition_to(DeveloperWorkflowState.COMPLETED)

    assert sm.current_state == DeveloperWorkflowState.COMPLETED
    assert sm.is_terminal is True


def test_state_machine_illegal_transition_rejected() -> None:
    """Requirement L: Illegal state transition raises InvalidStateTransitionError."""
    sm = DeveloperWorkflowStateMachine()
    assert sm.current_state == DeveloperWorkflowState.IDLE

    # Skipping directly from IDLE to APPLYING is illegal
    with pytest.raises(InvalidStateTransitionError):
        sm.transition_to(DeveloperWorkflowState.APPLYING)


def test_state_machine_terminal_failure_freezing() -> None:
    """Requirement B: Terminal failure state freezes workflow; cannot transition without explicit reset."""
    sm = DeveloperWorkflowStateMachine()
    sm.transition_to(DeveloperWorkflowState.FAILED, reason="Fatal execution error")
    assert sm.current_state in TERMINAL_FAILURE_STATES

    # Attempting any transition from FAILED raises InvalidStateTransitionError
    with pytest.raises(InvalidStateTransitionError):
        sm.transition_to(DeveloperWorkflowState.IDLE)

    with pytest.raises(InvalidStateTransitionError):
        sm.transition_to(DeveloperWorkflowState.PLAN_READY)

    # Explicit reset restores state machine to IDLE
    sm.reset()
    assert sm.current_state == DeveloperWorkflowState.IDLE


# ==============================================================================
# 10. EXCEPTION SAFETY & SANITIZATION
# ==============================================================================

def test_exception_safety_sanitizes_errors(tmp_path: Path) -> None:
    """Requirement K: Unhandled exception in execution pipeline is caught, producing structured UNEXPECTED_ERROR."""
    class CrashingApplier(CodeChangeApplier):
        def apply_proposal(self, *args: Any, **kwargs: Any) -> Any:
            raise RuntimeError("Database connection crashed: secret_db_password_12345")

    perm_mgr = PermissionManager(default_write_allowed=False)
    applier = CrashingApplier(permission_manager=perm_mgr, workspace_root=tmp_path)
    executor = DeveloperExecutor(applier=applier, workspace_root=tmp_path)

    proposal = CodeChangeProposal(
        target_file="crash.py",
        operation="create_file",
        proposed_content="print('crash')\n",
        description="Crash test",
    )

    executor.prepare_proposal(proposal)
    perm_mgr.grant_write_permission()

    exec_result = executor.execute_proposal(proposal=proposal, explicit_permission=True)

    assert exec_result.success is False
    assert exec_result.status == ExecutionStatus.UNEXPECTED_ERROR.value
    assert exec_result.overall_outcome == WorkflowOutcome.FAILED.value
    assert "secret_db_password_12345" not in exec_result.summary
    assert "secret_db_password_12345" not in (exec_result.error or "")


# ==============================================================================
# 11. SECURITY SAFEGUARDS (PATH TRAVERSAL & PROTECTED FILES)
# ==============================================================================

def test_security_protected_files_and_traversal_rejected(tmp_path: Path) -> None:
    """Security Safeguard: Targets with .git, .env, or path traversal are strictly rejected."""
    perm_mgr = PermissionManager(default_write_allowed=False)
    applier = CodeChangeApplier(permission_manager=perm_mgr, workspace_root=tmp_path)
    executor = DeveloperExecutor(applier=applier, workspace_root=tmp_path)

    for dangerous_target in [".env", ".git/config", "keys/private.key", "../../etc/passwd"]:
        prop = CodeChangeProposal(
            target_file=dangerous_target,
            operation="create_file",
            proposed_content="MALICIOUS=true\n",
            description="Exploit attempt",
        )
        res = executor.prepare_proposal(prop)
        assert res.success is False
        assert res.status == ExecutionStatus.PROPOSAL_INVALID.value
