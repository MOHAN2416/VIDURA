import os
import re
import sys
import time
import logging
import subprocess
from pathlib import Path
from typing import Any

from config import load_config
from permissions.manager import PermissionManager
from task_understanding.models import DeveloperTask, TaskType
from planning.models import DeveloperPlan
from developer.models import CodeChangeProposal
from developer.test_models import (
    TestStage,
    TestStatus,
    TestCoverageStatus,
    TestPlan,
    TestResult,
)

logger = logging.getLogger("VIDURA.developer.testing")

# Whitelist of strictly allowed pytest command-line arguments
ALLOWED_PYTEST_FLAGS = {
    "-v", "-q", "-s",
    "--tb=short", "--tb=line", "--tb=no", "--tb=auto",
    "-x", "--maxfail=1", "--maxfail=2", "--maxfail=3", "--maxfail=5",
}

# Dangerous shell metacharacters forbidden in targets and arguments
FORBIDDEN_SHELL_CHARS_RE = re.compile(r"[;`$&|><\n\(\)\{\}\[\]!*?]")


class PytestOutputParser:
    """Deterministic parser extracting test counts and timing from pytest output."""

    @staticmethod
    def parse(stdout: str, stderr: str = "", exit_code: int = 0) -> dict[str, Any]:
        combined = f"{stdout}\n{stderr}"

        passed = 0
        failed = 0
        skipped = 0
        errors = 0
        duration = None

        m_passed = re.search(r"(\d+)\s+passed", combined)
        if m_passed:
            passed = int(m_passed.group(1))

        m_failed = re.search(r"(\d+)\s+failed", combined)
        if m_failed:
            failed = int(m_failed.group(1))

        m_skipped = re.search(r"(\d+)\s+skipped", combined)
        if m_skipped:
            skipped = int(m_skipped.group(1))

        m_errors = re.search(r"(\d+)\s+error", combined)
        if m_errors:
            errors = int(m_errors.group(1))

        m_duration = re.search(r"in\s+([\d\.]+)s", combined)
        if m_duration:
            try:
                duration = float(m_duration.group(1))
            except ValueError:
                pass

        tests_attempted = passed + failed + errors

        return {
            "tests_attempted": tests_attempted,
            "tests_passed": passed,
            "tests_failed": failed + errors,
            "tests_skipped": skipped,
            "duration_seconds": duration,
        }


class TestPlanner:
    """Discovers repository-grounded tests and determines safe testing strategies."""

    def __init__(
        self,
        workspace_root: str | Path | None = None,
        codebase_manager: Any = None,
    ) -> None:
        cfg = load_config()
        self.workspace_root = Path(workspace_root).resolve() if workspace_root else cfg.workspace_root.resolve()
        self.codebase_manager = codebase_manager

    def determine_test_plan(
        self,
        proposal: CodeChangeProposal | None = None,
        task: DeveloperTask | None = None,
        plan: DeveloperPlan | None = None,
    ) -> TestPlan:
        """Determines an appropriate, repository-grounded test plan for proposed/applied changes."""
        target_file = proposal.target_file if proposal else ""
        if not target_file and task and task.target_files:
            target_file = task.target_files[0]
        if not target_file and plan and plan.relevant_files:
            target_file = plan.relevant_files[0]

        if not target_file:
            logger.info("No target file identified; testing not required.")
            return TestPlan(
                testing_required=False,
                scope="none",
                coverage_status=TestCoverageStatus.NO_RELEVANT_TEST_FOUND.value,
                reason="No target file identified.",
            )

        # 1. Direct match: If changed file is already a test file
        target_path = Path(target_file)
        target_name = target_path.name
        is_direct_test = (
            target_name.startswith("test_")
            or target_name.endswith("_test.py")
            or "tests/" in target_file
        )
        if is_direct_test:
            full_path = self.workspace_root / target_path
            if full_path.exists():
                return TestPlan(
                    testing_required=True,
                    test_files=[str(target_file)],
                    test_command=[sys.executable, "-m", "pytest", str(target_file), "-v"],
                    scope="targeted",
                    reason=f"Modified file is itself a test file: '{target_file}'",
                    coverage_status=TestCoverageStatus.COVERED.value,
                )

        # 2. Grounded test discovery: Search actual test files in workspace
        stem = target_path.stem
        candidate_names = [f"test_{stem}.py", f"{stem}_test.py"]
        found_test_files: list[str] = []

        tests_dir = self.workspace_root / "tests"
        if tests_dir.is_dir():
            for cand in candidate_names:
                cand_path = tests_dir / cand
                if cand_path.exists():
                    found_test_files.append(f"tests/{cand}")

        # 3. Importers graph discovery via CodebaseManager
        if not found_test_files and self.codebase_manager and hasattr(self.codebase_manager, "index"):
            idx = self.codebase_manager.index
            if hasattr(idx, "importers_graph"):
                mod_name = stem
                importers = idx.importers_graph.get(mod_name, set())
                for imp in importers:
                    if imp.startswith("test_") or "test" in imp:
                        cand_file = tests_dir / f"{imp}.py"
                        if cand_file.exists():
                            found_test_files.append(f"tests/{imp}.py")

        # 4. Fallback: Search any test file in tests/ matching stem pattern
        if not found_test_files and tests_dir.is_dir():
            for item in tests_dir.glob(f"test_*{stem}*.py"):
                if item.is_file():
                    found_test_files.append(f"tests/{item.name}")

        if not found_test_files:
            logger.info(f"No relevant tests discovered for target '{target_file}'.")
            return TestPlan(
                testing_required=False,
                test_files=[],
                scope="none",
                coverage_status=TestCoverageStatus.NO_RELEVANT_TEST_FOUND.value,
                reason=f"No relevant test file found for '{target_file}' in workspace.",
            )

        # Check if broader regression testing is advised
        requires_regression = False
        risks = []
        if plan:
            risks = list(plan.risks)
            if any("risk" in r.lower() or "breaking" in r.lower() for r in risks):
                requires_regression = True
        if task and task.task_type == TaskType.TEST:
            requires_regression = True

        primary_test = found_test_files[0]
        cmd = [sys.executable, "-m", "pytest", primary_test, "-v"]

        return TestPlan(
            testing_required=True,
            test_files=found_test_files,
            test_command=cmd,
            scope="targeted" if not requires_regression else "regression",
            risks=risks,
            requires_regression_testing=requires_regression,
            reason=f"Found {len(found_test_files)} relevant test file(s) for '{target_file}'.",
            coverage_status=TestCoverageStatus.COVERED.value,
        )


class TestRunner:
    """Controlled, permission-bound test execution layer with strict security boundaries.
    
    CRITICAL SECURITY MANDATES:
    1. Predefined Runner: Strictly executes sys.executable -m pytest.
    2. Zero Shell Access: Executes strictly with shell=False. Never invokes a shell.
    3. Workspace Sandboxing: Restricts working directory to workspace_root.
    4. Path Traversal Rejection: Validates test target resolves strictly within workspace_root.
    5. Injection Prevention: Rejects shell metacharacters and argument tampering.
    6. Timeouts: Terminates child processes safely on timeout.
    7. Output Bounding: Deterministically limits stdout/stderr length.
    8. Authoritative Results: Process exit code and runner status are authoritative.
    """

    def __init__(
        self,
        workspace_root: str | Path | None = None,
        permission_manager: PermissionManager | None = None,
        timeout_seconds: int | None = None,
        output_limit: int | None = None,
    ) -> None:
        cfg = load_config()
        self.workspace_root = Path(workspace_root).resolve() if workspace_root else cfg.workspace_root.resolve()
        self.permission_manager = permission_manager
        self.timeout_seconds = timeout_seconds or getattr(cfg, "test_timeout_seconds", 30)
        self.output_limit = output_limit or getattr(cfg, "test_output_limit", 50000)

    def _validate_target(self, test_target: str) -> Path:
        """Validates that test_target resolves inside workspace_root and contains no injection."""
        if not test_target or not isinstance(test_target, str):
            raise ValueError("Test target must be a non-empty string.")

        # Check for forbidden shell metacharacters
        if FORBIDDEN_SHELL_CHARS_RE.search(test_target):
            raise ValueError(f"Forbidden characters detected in test target: '{test_target}'")

        # Strip pytest node identifier (e.g. test_file.py::test_func)
        file_part = test_target.split("::")[0].strip()

        root = self.workspace_root.resolve()
        root_real = Path(os.path.realpath(root))
        path_obj = Path(file_part)

        if path_obj.is_absolute():
            candidate = path_obj
        else:
            candidate = root / path_obj

        try:
            resolved = candidate.resolve()
            real_target = Path(os.path.realpath(resolved))
        except Exception as err:
            raise ValueError(f"Access denied: Failed to resolve test target '{test_target}': {err}")

        # Boundary checks
        if not resolved.is_relative_to(root) or not real_target.is_relative_to(root_real):
            raise ValueError(
                f"Access denied: Test target '{test_target}' resolves outside workspace boundary '{root}'."
            )

        if not resolved.exists():
            raise ValueError(f"Test target does not exist: '{test_target}'")

        return resolved

    def _validate_arguments(self, extra_args: list[str]) -> list[str]:
        """Validates extra pytest arguments against allowed whitelist."""
        validated: list[str] = []
        for arg in extra_args:
            if not arg or not isinstance(arg, str):
                continue
            if FORBIDDEN_SHELL_CHARS_RE.search(arg):
                raise ValueError(f"Forbidden characters in test argument: '{arg}'")
            if arg in ALLOWED_PYTEST_FLAGS or arg.startswith("-k=") or arg.startswith("--maxfail="):
                validated.append(arg)
            else:
                raise ValueError(f"Disallowed test runner argument: '{arg}'")
        return validated

    def run_test_plan(self, plan: TestPlan) -> TestResult:
        """Executes tests according to a validated TestPlan."""
        if not plan.testing_required or not plan.test_files:
            return TestResult(
                status=TestStatus.NOT_RUN.value,
                stage=TestStage.TEST_NOT_REQUIRED.value,
                command=[],
                target="",
                error="Testing not required by test plan.",
            )

        target = plan.test_files[0]
        return self.run_test(target=target)

    def run_test(
        self,
        target: str,
        extra_args: list[str] | None = None,
        explicit_permission: bool = False,
    ) -> TestResult:
        """Executes a single test target strictly through controlled subprocess execution."""
        # 1. Permission check
        if self.permission_manager:
            if not self.permission_manager.is_test_allowed(target):
                logger.warning(f"Test execution permission DENIED for target '{target}'.")
                return TestResult(
                    status=TestStatus.PERMISSION_DENIED.value,
                    stage=TestStage.TEST_PERMISSION_PENDING.value,
                    target=target,
                    error="Test execution permission denied: Explicit authorization is required.",
                )
        elif not explicit_permission and self.permission_manager is not None:
            logger.warning(f"Test execution permission denied for target '{target}'.")
            return TestResult(
                status=TestStatus.PERMISSION_DENIED.value,
                stage=TestStage.TEST_PERMISSION_PENDING.value,
                target=target,
                error="Test execution permission denied.",
            )

        # 2. Target validation
        try:
            self._validate_target(target)
            validated_args = self._validate_arguments(extra_args or [])
        except ValueError as err:
            logger.warning(f"Validation failed for test target '{target}': {err}")
            return TestResult(
                status=TestStatus.INVALID_TARGET.value,
                stage=TestStage.TEST_ERROR.value,
                target=target,
                error=str(err),
            )

        # 3. Structural command construction
        cmd = [sys.executable, "-m", "pytest", target] + validated_args
        logger.info(f"Executing controlled test command: {cmd} (CWD: '{self.workspace_root}')")

        start_time = time.perf_counter()
        try:
            # STRICT INVARIANT: shell=False, cwd=workspace_root
            process = subprocess.run(
                cmd,
                cwd=str(self.workspace_root),
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
                shell=False,
            )
            duration = time.perf_counter() - start_time
            stdout = process.stdout
            stderr = process.stderr
            exit_code = process.returncode

        except subprocess.TimeoutExpired as err:
            duration = time.perf_counter() - start_time
            logger.warning(f"Test execution timed out after {self.timeout_seconds} seconds.")
            return TestResult(
                status=TestStatus.TIMEOUT.value,
                stage=TestStage.TEST_TIMEOUT.value,
                exit_code=None,
                duration_seconds=duration,
                command=cmd,
                target=target,
                stdout=err.stdout if isinstance(err.stdout, str) else "",
                stderr=err.stderr if isinstance(err.stderr, str) else "",
                error=f"Test execution timed out after {self.timeout_seconds} seconds.",
            )

        except Exception as err:
            duration = time.perf_counter() - start_time
            logger.error(f"Test execution error: {err}")
            return TestResult(
                status=TestStatus.EXECUTION_ERROR.value,
                stage=TestStage.TEST_ERROR.value,
                exit_code=-1,
                duration_seconds=duration,
                command=cmd,
                target=target,
                error=str(err),
            )

        # 4. Bounded output limiting
        output_truncated = False
        if len(stdout) > self.output_limit:
            stdout = stdout[:self.output_limit] + "\n\n[... Output truncated by VIDURA Test Runner ...]\n"
            output_truncated = True
        if len(stderr) > self.output_limit:
            stderr = stderr[:self.output_limit] + "\n\n[... Stderr truncated by VIDURA Test Runner ...]\n"
            output_truncated = True

        # 5. Parse test results
        parsed = PytestOutputParser.parse(stdout, stderr, exit_code)

        if exit_code == 0:
            status = TestStatus.PASSED.value
            stage = TestStage.TEST_PASSED.value
        elif exit_code == 1:
            status = TestStatus.FAILED.value
            stage = TestStage.TEST_FAILED.value
        else:
            status = TestStatus.EXECUTION_ERROR.value
            stage = TestStage.TEST_ERROR.value

        return TestResult(
            status=status,
            stage=stage,
            tests_attempted=parsed["tests_attempted"],
            tests_passed=parsed["tests_passed"],
            tests_failed=parsed["tests_failed"],
            tests_skipped=parsed["tests_skipped"],
            exit_code=exit_code,
            stdout=stdout,
            stderr=stderr,
            duration_seconds=round(duration, 3),
            command=cmd,
            target=target,
            error=None if exit_code == 0 else f"Test run exited with code {exit_code}",
            output_truncated=output_truncated,
        )
