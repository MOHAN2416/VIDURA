import os
import sys
import time
import subprocess
from pathlib import Path
from typing import Any
import pytest

from permissions.manager import PermissionManager
from developer.test_models import (
    TestStage,
    TestStatus,
    TestCoverageStatus,
    TestPlan,
    TestResult,
    TestSummary,
)
from developer.testing import (
    PytestOutputParser,
    TestPlanner,
    TestRunner,
    ALLOWED_PYTEST_FLAGS,
    FORBIDDEN_SHELL_CHARS_RE,
)

# Prevent pytest from attempting to collect helper and model classes as test suites
TestStage.__test__ = False
TestStatus.__test__ = False
TestCoverageStatus.__test__ = False
TestPlan.__test__ = False
TestResult.__test__ = False
TestSummary.__test__ = False
TestPlanner.__test__ = False
TestRunner.__test__ = False
from developer.models import (
    CodeChangeProposal,
    ProposalOperation,
    ExecutionStage,
    ExecutionStatus,
    DeveloperExecutionResult,
)
from developer.applier import CodeChangeApplier
from developer.executor import DeveloperExecutor
from tools.developer import RunTestsTool
from agent.loop import claims_test_success, AgentLoop
from agent.state import AgentState
from models.base import BaseLLMProvider


class MockLLM(BaseLLMProvider):
    """Deterministic mock LLM for testing."""

    def __init__(self, response_text: str = "") -> None:
        self.response_text = response_text
        self.captured_messages: list[dict[str, str]] = []

    @property
    def model_name(self) -> str:
        return "mock-llm"

    @property
    def provider_name(self) -> str:
        return "Mock Provider"

    def generate(self, messages: list[dict[str, str]], **kwargs: Any) -> str:
        self.captured_messages = list(messages)
        return self.response_text


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    """Fixture providing an isolated workspace."""
    ws = tmp_path / "test_ws"
    ws.mkdir(parents=True, exist_ok=True)
    return ws


# =========================================================================
# A. TestPlan Creation
# =========================================================================

def test_spec_a_test_plan_creation(workspace: Path) -> None:
    """A. TestPlan creation: Verified proposal produces valid TestPlan; absent target produces no-run plan."""
    src_file = workspace / "calculator.py"
    src_file.write_text("def add(a, b):\n    return a + b\n", encoding="utf-8")
    tests_dir = workspace / "tests"
    tests_dir.mkdir(parents=True, exist_ok=True)
    test_file = tests_dir / "test_calculator.py"
    test_file.write_text("from calculator import add\ndef test_add():\n    assert add(1, 2) == 3\n", encoding="utf-8")

    planner = TestPlanner(workspace_root=workspace)
    proposal = CodeChangeProposal(
        target_file="calculator.py",
        operation=ProposalOperation.MODIFY_FILE.value,
        proposed_content="def add(a, b):\n    return a + b\n",
    )

    plan = planner.determine_test_plan(proposal=proposal)
    assert plan.testing_required is True
    assert "tests/test_calculator.py" in plan.test_files
    assert plan.coverage_status == TestCoverageStatus.COVERED.value
    assert sys.executable in plan.test_command[0]
    assert "-m" in plan.test_command
    assert "pytest" in plan.test_command

    # Unsupported / missing target produces no-run plan
    empty_plan = planner.determine_test_plan(proposal=None)
    assert empty_plan.testing_required is False
    assert empty_plan.coverage_status == TestCoverageStatus.NO_RELEVANT_TEST_FOUND.value


# =========================================================================
# B. Relevant Test Discovery
# =========================================================================

def test_spec_b_relevant_test_discovery(workspace: Path) -> None:
    """B. Relevant test discovery: Matches target files, discovers existing tests, handles absent gracefully."""
    tests_dir = workspace / "tests"
    tests_dir.mkdir(parents=True, exist_ok=True)
    (tests_dir / "test_user_service.py").write_text("def test_user(): pass\n", encoding="utf-8")
    (tests_dir / "order_service_test.py").write_text("def test_order(): pass\n", encoding="utf-8")

    planner = TestPlanner(workspace_root=workspace)

    # test_prefix pattern
    prop1 = CodeChangeProposal(target_file="user_service.py", operation=ProposalOperation.MODIFY_FILE.value)
    plan1 = planner.determine_test_plan(proposal=prop1)
    assert plan1.testing_required is True
    assert "tests/test_user_service.py" in plan1.test_files

    # _test suffix pattern
    prop2 = CodeChangeProposal(target_file="order_service.py", operation=ProposalOperation.MODIFY_FILE.value)
    plan2 = planner.determine_test_plan(proposal=prop2)
    assert plan2.testing_required is True
    assert "tests/order_service_test.py" in plan2.test_files

    # Direct test file modified
    prop3 = CodeChangeProposal(target_file="tests/test_user_service.py", operation=ProposalOperation.MODIFY_FILE.value)
    plan3 = planner.determine_test_plan(proposal=prop3)
    assert plan3.testing_required is True
    assert "tests/test_user_service.py" in plan3.test_files

    # Absent tests handled gracefully
    prop4 = CodeChangeProposal(target_file="untested_module.py", operation=ProposalOperation.MODIFY_FILE.value)
    plan4 = planner.determine_test_plan(proposal=prop4)
    assert plan4.testing_required is False
    assert plan4.coverage_status == TestCoverageStatus.NO_RELEVANT_TEST_FOUND.value
    assert len(plan4.test_files) == 0


# =========================================================================
# C. Unknown Test Target Rejection
# =========================================================================

def test_spec_c_unknown_test_target_rejection(workspace: Path) -> None:
    """C. Unknown test target rejection: Target outside test discovery / non-existent rejected."""
    runner = TestRunner(workspace_root=workspace)
    res = runner.run_test(target="tests/test_does_not_exist.py")

    assert res.status == TestStatus.INVALID_TARGET.value
    assert res.stage == TestStage.TEST_ERROR.value
    assert "does not exist" in (res.error or "").lower()


# =========================================================================
# D. Workspace Traversal Rejection
# =========================================================================

def test_spec_d_workspace_traversal_rejection(workspace: Path) -> None:
    """D. Workspace traversal rejection: Targets with ../ or escaping workspace rejected."""
    runner = TestRunner(workspace_root=workspace)

    res = runner.run_test(target="../../etc/passwd")
    assert res.status == TestStatus.INVALID_TARGET.value
    assert "outside workspace" in (res.error or "").lower() or "forbidden" in (res.error or "").lower()

    res2 = runner.run_test(target="tests/../../secret.py")
    assert res2.status == TestStatus.INVALID_TARGET.value


# =========================================================================
# E. Absolute Outside-Workspace Rejection
# =========================================================================

def test_spec_e_absolute_outside_workspace_rejection(workspace: Path, tmp_path: Path) -> None:
    """E. Absolute outside-workspace rejection: Absolute path outside workspace rejected."""
    outside_file = tmp_path / "outside_test.py"
    outside_file.write_text("def test_out(): pass\n", encoding="utf-8")

    runner = TestRunner(workspace_root=workspace)
    res = runner.run_test(target=str(outside_file.resolve()))

    assert res.status == TestStatus.INVALID_TARGET.value
    assert "outside workspace boundary" in (res.error or "").lower()


# =========================================================================
# F. Command Injection Rejection
# =========================================================================

def test_spec_f_command_injection_rejection(workspace: Path) -> None:
    """F. Command injection rejection: Metacharacters rejected immediately without execution."""
    runner = TestRunner(workspace_root=workspace)

    injection_targets = [
        "tests/test_math.py; rm -rf /",
        "tests/test_math.py && echo evil",
        "tests/test_math.py | cat",
        "tests/test_math.py `whoami`",
        "tests/test_math.py $(id)",
        "tests/test_math.py > /tmp/out",
        "tests/test_math.py\nls",
        "tests/test_math.py&",
        "tests/test_math.py(1)",
    ]

    for inj in injection_targets:
        res = runner.run_test(target=inj)
        assert res.status == TestStatus.INVALID_TARGET.value
        assert "forbidden characters" in (res.error or "").lower()


# =========================================================================
# G. Allowed Test Command Construction
# =========================================================================

def test_spec_g_allowed_test_command_construction(workspace: Path) -> None:
    """G. Allowed test command construction: Exact executable + -m pytest + target; args strictly whitelisted."""
    tests_dir = workspace / "tests"
    tests_dir.mkdir(parents=True, exist_ok=True)
    test_file = tests_dir / "test_sample.py"
    test_file.write_text("def test_ok(): pass\n", encoding="utf-8")

    runner = TestRunner(workspace_root=workspace)

    # Valid arguments whitelisted
    validated = runner._validate_arguments(["-v", "-q", "--tb=short"])
    assert validated == ["-v", "-q", "--tb=short"]

    # Disallowed argument rejected
    with pytest.raises(ValueError, match="Disallowed test runner argument"):
        runner._validate_arguments(["--arbitrary-option"])

    with pytest.raises(ValueError, match="Forbidden characters"):
        runner._validate_arguments(["-v; rm -rf /"])


# =========================================================================
# H. No shell=True
# =========================================================================

def test_spec_h_no_shell_true(workspace: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """H. No shell=True: Verify strictly shell=False and argument array used."""
    tests_dir = workspace / "tests"
    tests_dir.mkdir(parents=True, exist_ok=True)
    test_file = tests_dir / "test_sample.py"
    test_file.write_text("def test_ok(): pass\n", encoding="utf-8")

    captured_kwargs: dict[str, Any] = {}

    real_run = subprocess.run

    def mock_subprocess_run(*args: Any, **kwargs: Any) -> subprocess.CompletedProcess:
        captured_kwargs.update(kwargs)
        return real_run(*args, **kwargs)

    monkeypatch.setattr(subprocess, "run", mock_subprocess_run)

    runner = TestRunner(workspace_root=workspace)
    runner.run_test(target="tests/test_sample.py")

    assert captured_kwargs.get("shell") is False
    assert captured_kwargs.get("cwd") == str(workspace.resolve())


# =========================================================================
# I. Test Timeout
# =========================================================================

def test_spec_i_test_timeout(workspace: Path) -> None:
    """I. Test timeout: Tests exceeding timeout marked timed_out."""
    tests_dir = workspace / "tests"
    tests_dir.mkdir(parents=True, exist_ok=True)
    sleep_test = tests_dir / "test_sleep.py"
    sleep_test.write_text(
        "import time\n"
        "def test_sleep():\n"
        "    time.sleep(3)\n",
        encoding="utf-8",
    )

    runner = TestRunner(workspace_root=workspace, timeout_seconds=1)
    res = runner.run_test(target="tests/test_sleep.py")

    assert res.status == TestStatus.TIMEOUT.value
    assert res.stage == TestStage.TEST_TIMEOUT.value
    assert "timed out after 1 seconds" in (res.error or "")


# =========================================================================
# J. Output Truncation
# =========================================================================

def test_spec_j_output_truncation(workspace: Path) -> None:
    """J. Output truncation: Excessive stdout/stderr truncated deterministically."""
    tests_dir = workspace / "tests"
    tests_dir.mkdir(parents=True, exist_ok=True)
    noisy_test = tests_dir / "test_noisy.py"
    noisy_test.write_text(
        "def test_loud():\n"
        "    for _ in range(500):\n"
        "        print('Excessive test logging line that repeats ' * 5)\n",
        encoding="utf-8",
    )

    runner = TestRunner(workspace_root=workspace, output_limit=500)
    res = runner.run_test(target="tests/test_noisy.py", extra_args=["-s"])

    assert res.output_truncated is True
    assert "[... Output truncated by VIDURA Test Runner ...]" in res.stdout
    assert len(res.stdout) < 1000


# =========================================================================
# K. Passing Test Result
# =========================================================================

def test_spec_k_passing_test_result(workspace: Path) -> None:
    """K. Passing test result: exit 0 -> status="passed", test count correctly parsed."""
    tests_dir = workspace / "tests"
    tests_dir.mkdir(parents=True, exist_ok=True)
    pass_test = tests_dir / "test_pass.py"
    pass_test.write_text(
        "def test_one():\n"
        "    assert 1 == 1\n"
        "def test_two():\n"
        "    assert 2 == 2\n",
        encoding="utf-8",
    )

    runner = TestRunner(workspace_root=workspace)
    res = runner.run_test(target="tests/test_pass.py")

    assert res.status == TestStatus.PASSED.value
    assert res.stage == TestStage.TEST_PASSED.value
    assert res.exit_code == 0
    assert res.tests_passed == 2
    assert res.tests_failed == 0
    assert res.passed is True


# =========================================================================
# L. Failing Test Result
# =========================================================================

def test_spec_l_failing_test_result(workspace: Path) -> None:
    """L. Failing test result: exit 1 -> status="failed", failure count and error captured."""
    tests_dir = workspace / "tests"
    tests_dir.mkdir(parents=True, exist_ok=True)
    fail_test = tests_dir / "test_fail.py"
    fail_test.write_text(
        "def test_ok():\n"
        "    assert True\n"
        "def test_bad():\n"
        "    assert 1 == 2\n",
        encoding="utf-8",
    )

    runner = TestRunner(workspace_root=workspace)
    res = runner.run_test(target="tests/test_fail.py")

    assert res.status == TestStatus.FAILED.value
    assert res.stage == TestStage.TEST_FAILED.value
    assert res.exit_code == 1
    assert res.tests_passed == 1
    assert res.tests_failed == 1
    assert res.passed is False


# =========================================================================
# M. Test Execution Error
# =========================================================================

def test_spec_m_test_execution_error(workspace: Path) -> None:
    """M. Test execution error: Syntax error or execution crash -> status="execution_error"."""
    tests_dir = workspace / "tests"
    tests_dir.mkdir(parents=True, exist_ok=True)
    syntax_error_test = tests_dir / "test_syntax.py"
    syntax_error_test.write_text("def test_broken(: pass\n", encoding="utf-8")

    runner = TestRunner(workspace_root=workspace)
    res = runner.run_test(target="tests/test_syntax.py")

    # Pytest exit code 4 or 2 is an error, not standard test failure exit code 1
    assert res.status == TestStatus.EXECUTION_ERROR.value
    assert res.stage == TestStage.TEST_ERROR.value
    assert res.exit_code not in (0, 1)


# =========================================================================
# N. Permission Denied
# =========================================================================

def test_spec_n_permission_denied(workspace: Path) -> None:
    """N. Permission denied: Test execution without permission blocked; granted execution succeeds."""
    tests_dir = workspace / "tests"
    tests_dir.mkdir(parents=True, exist_ok=True)
    test_file = tests_dir / "test_perm.py"
    test_file.write_text("def test_perm(): pass\n", encoding="utf-8")

    perm_mgr = PermissionManager(default_write_allowed=False, default_test_allowed=False)
    runner = TestRunner(workspace_root=workspace, permission_manager=perm_mgr)

    # Denied by default
    res = runner.run_test(target="tests/test_perm.py")
    assert res.status == TestStatus.PERMISSION_DENIED.value
    assert res.stage == TestStage.TEST_PERMISSION_PENDING.value
    assert "permission denied" in (res.error or "").lower()

    # Grant permission -> succeeds
    perm_mgr.grant_test_permission()
    res2 = runner.run_test(target="tests/test_perm.py")
    assert res2.status == TestStatus.PASSED.value

    # Revoke permission -> denied again
    perm_mgr.revoke_test_permission()
    res3 = runner.run_test(target="tests/test_perm.py")
    assert res3.status == TestStatus.PERMISSION_DENIED.value


# =========================================================================
# O. Model False-Success Claim Cannot Override Runner Result
# =========================================================================

def test_spec_o_model_false_success_claim_cannot_override_runner_result(workspace: Path) -> None:
    """O. Model false-success claim cannot override runner result: Runner failure authoritative."""
    assert claims_test_success("All tests passed successfully!") is True
    assert claims_test_success("I ran pytest and all 3 passed.") is True
    assert claims_test_success("Tests failed with exit code 1.") is False

    perm_mgr = PermissionManager(default_write_allowed=True, default_test_allowed=True)
    applier = CodeChangeApplier(permission_manager=perm_mgr, workspace_root=workspace)
    executor = DeveloperExecutor(applier=applier, permission_manager=perm_mgr, workspace_root=workspace)

    # Failing test file
    tests_dir = workspace / "tests"
    tests_dir.mkdir(parents=True, exist_ok=True)
    (tests_dir / "test_service.py").write_text("def test_boom(): assert False\n", encoding="utf-8")
    (workspace / "service.py").write_text("def svc(): pass\n", encoding="utf-8")

    proposal = CodeChangeProposal(
        target_file="service.py",
        operation=ProposalOperation.MODIFY_FILE.value,
        original_content="def svc(): pass\n",
        proposed_content="def svc(): return 1\n",
    )
    executor.prepare_proposal(proposal)
    perm_mgr.grant_write_permission()
    perm_mgr.grant_test_permission()

    res = executor.execute_proposal(proposal=proposal, explicit_permission=True)

    assert res.test_result is not None
    assert res.test_result.status == TestStatus.FAILED.value
    assert "Result: failed" in res.summary


# =========================================================================
# P. Successful Phase 8.4 Application Triggers Test Planning
# =========================================================================

def test_spec_p_successful_application_triggers_test_planning(workspace: Path) -> None:
    """P. Successful Phase 8.4 application triggers test planning and execution."""
    src = workspace / "math_lib.py"
    src.write_text("def multiply(a, b):\n    return a + b\n", encoding="utf-8")
    tests_dir = workspace / "tests"
    tests_dir.mkdir(parents=True, exist_ok=True)
    test_file = tests_dir / "test_math_lib.py"
    test_file.write_text("from math_lib import multiply\ndef test_multiply():\n    assert multiply(3, 4) == 12\n", encoding="utf-8")

    perm_mgr = PermissionManager(default_write_allowed=True, default_test_allowed=True)
    applier = CodeChangeApplier(permission_manager=perm_mgr, workspace_root=workspace)
    executor = DeveloperExecutor(applier=applier, permission_manager=perm_mgr, workspace_root=workspace)

    proposal = CodeChangeProposal(
        target_file="math_lib.py",
        operation=ProposalOperation.MODIFY_FILE.value,
        original_content="def multiply(a, b):\n    return a + b\n",
        proposed_content="def multiply(a, b):\n    return a * b\n",
        description="Fix multiplication bug",
    )
    executor.prepare_proposal(proposal)
    perm_mgr.grant_write_permission()
    perm_mgr.grant_test_permission()

    result = executor.execute_proposal(proposal=proposal, explicit_permission=True)

    assert result.success is True
    assert result.test_plan is not None
    assert result.test_plan.testing_required is True
    assert result.test_result is not None
    assert result.test_result.status == TestStatus.PASSED.value
    assert result.test_result.tests_passed == 1
    assert "Change applied:" in result.summary
    assert "Tests:" in result.summary
    assert "Passed: 1" in result.summary


# =========================================================================
# Q. Failed Application Prevents Test Execution
# =========================================================================

def test_spec_q_failed_application_prevents_test_execution(workspace: Path) -> None:
    """Q. Failed application prevents test execution: Failed apply never runs tests."""
    src = workspace / "server.py"
    src.write_text("PORT = 8080\n", encoding="utf-8")
    tests_dir = workspace / "tests"
    tests_dir.mkdir(parents=True, exist_ok=True)
    (tests_dir / "test_server.py").write_text("def test_port(): pass\n", encoding="utf-8")

    perm_mgr = PermissionManager(default_write_allowed=False, default_test_allowed=True)
    applier = CodeChangeApplier(permission_manager=perm_mgr, workspace_root=workspace)
    executor = DeveloperExecutor(applier=applier, permission_manager=perm_mgr, workspace_root=workspace)

    proposal = CodeChangeProposal(
        target_file="server.py",
        operation=ProposalOperation.MODIFY_FILE.value,
        original_content="PORT = 8080\n",
        proposed_content="PORT = 9000\n",
    )
    executor.prepare_proposal(proposal)

    # Permission denied for application
    result = executor.execute_proposal(proposal=proposal, explicit_permission=False)

    assert result.success is False
    assert result.test_plan is None
    assert result.test_result is None


# =========================================================================
# R. Failed Filesystem Verification Prevents Successful Development Result
# =========================================================================

def test_spec_r_failed_filesystem_verification_prevents_successful_result(workspace: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """R. Failed filesystem verification prevents successful development result: Never runs tests."""
    src = workspace / "handler.py"
    src.write_text("def handle(): pass\n", encoding="utf-8")
    tests_dir = workspace / "tests"
    tests_dir.mkdir(parents=True, exist_ok=True)
    (tests_dir / "test_handler.py").write_text("def test_h(): pass\n", encoding="utf-8")

    perm_mgr = PermissionManager(default_write_allowed=True, default_test_allowed=True)
    applier = CodeChangeApplier(permission_manager=perm_mgr, workspace_root=workspace)
    executor = DeveloperExecutor(applier=applier, permission_manager=perm_mgr, workspace_root=workspace)

    proposal = CodeChangeProposal(
        target_file="handler.py",
        operation=ProposalOperation.MODIFY_FILE.value,
        original_content="def handle(): pass\n",
        proposed_content="def handle(): return True\n",
    )
    executor.prepare_proposal(proposal)
    perm_mgr.grant_write_permission()
    perm_mgr.grant_test_permission()

    from developer.verifier import VerificationResult
    monkeypatch.setattr(
        applier.verifier,
        "verify",
        lambda p: VerificationResult(
            success=False,
            target_file="handler.py",
            operation="modify_file",
            verified=False,
            expected_state="def handle(): return True\n",
            actual_state="TAMPERED",
            reason="Disk mismatch",
        )
    )

    result = executor.execute_proposal(proposal=proposal, explicit_permission=True)

    assert result.success is False
    assert result.stage == ExecutionStage.VERIFICATION_FAILED.value
    assert result.test_plan is None
    assert result.test_result is None


# =========================================================================
# S. Serialization Roundtrip
# =========================================================================

def test_spec_s_serialization_roundtrip() -> None:
    """S. Serialization roundtrip for TestPlan, TestResult, and TestSummary."""
    plan = TestPlan(
        testing_required=True,
        test_files=["tests/test_foo.py"],
        test_names=["test_one"],
        test_command=["python", "-m", "pytest", "tests/test_foo.py"],
        reason="Target covered",
        scope="targeted",
        risks=["Breaking change"],
        requires_regression_testing=True,
        coverage_status=TestCoverageStatus.COVERED.value,
    )
    plan_dict = plan.to_dict()
    restored_plan = TestPlan.from_dict(plan_dict)
    assert restored_plan.testing_required is True
    assert restored_plan.test_files == ["tests/test_foo.py"]
    assert restored_plan.coverage_status == TestCoverageStatus.COVERED.value

    res = TestResult(
        status=TestStatus.PASSED.value,
        stage=TestStage.TEST_PASSED.value,
        tests_attempted=5,
        tests_passed=5,
        tests_failed=0,
        tests_skipped=0,
        exit_code=0,
        stdout="5 passed in 0.05s",
        stderr="",
        duration_seconds=0.05,
        command=["python", "-m", "pytest"],
        target="tests/test_foo.py",
        error=None,
        output_truncated=False,
    )
    res_dict = res.to_dict()
    restored_res = TestResult.from_dict(res_dict)
    assert restored_res.status == TestStatus.PASSED.value
    assert restored_res.tests_passed == 5
    assert restored_res.passed is True

    summary = TestSummary(
        target_file="foo.py",
        target_symbol="bar",
        applied=True,
        verified=True,
        test_target="tests/test_foo.py",
        test_result=TestStatus.PASSED.value,
        tests_passed=5,
        tests_failed=0,
        tests_skipped=0,
    )
    display = summary.format_display()
    assert "Change applied:" in display
    assert "- File: foo.py" in display
    assert "- Symbol: bar" in display
    assert "Tests:" in display
    assert "- Target: tests/test_foo.py" in display
    assert "- Result: passed" in display
    assert "- Passed: 5" in display


# =========================================================================
# T. RunTestsTool Verification
# =========================================================================

def test_spec_t_run_tests_tool(workspace: Path) -> None:
    """T. RunTestsTool verification: Tool execution, validation, and error handling."""
    tests_dir = workspace / "tests"
    tests_dir.mkdir(parents=True, exist_ok=True)
    test_file = tests_dir / "test_tool.py"
    test_file.write_text("def test_tool(): pass\n", encoding="utf-8")

    perm_mgr = PermissionManager(default_write_allowed=True, default_test_allowed=True)
    tool = RunTestsTool(permission_manager=perm_mgr, workspace_root=workspace)

    # Valid execution
    res = tool.execute(target="tests/test_tool.py")
    assert res["success"] is True
    assert res["data"]["status"] == TestStatus.PASSED.value

    # Traversal attack
    res_traversal = tool.execute(target="../../etc/passwd")
    assert res_traversal["success"] is False

    # Command injection attack
    res_injection = tool.execute(target="tests/test_tool.py; ls")
    assert res_injection["success"] is False

    # Empty target
    res_empty = tool.execute(target="")
    assert res_empty["success"] is False
