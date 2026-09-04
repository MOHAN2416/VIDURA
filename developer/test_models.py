from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class TestStage(str, Enum):
    """Lifecycle stages for developer test planning and execution."""
    TEST_NOT_REQUIRED = "test_not_required"
    TEST_PLANNED = "test_planned"
    TEST_PERMISSION_PENDING = "test_permission_pending"
    TEST_APPROVED = "test_approved"
    TEST_RUNNING = "test_running"
    TEST_PASSED = "test_passed"
    TEST_FAILED = "test_failed"
    TEST_TIMEOUT = "test_timeout"
    TEST_ERROR = "test_error"


class TestStatus(str, Enum):
    """Authoritative outcome status codes for test execution."""
    NOT_RUN = "not_run"
    PASSED = "passed"
    FAILED = "failed"
    TIMEOUT = "timeout"
    EXECUTION_ERROR = "execution_error"
    PERMISSION_DENIED = "permission_denied"
    INVALID_TARGET = "invalid_target"
    NO_TESTS_FOUND = "no_tests_found"


class TestCoverageStatus(str, Enum):
    """Test coverage status for codebase intelligence mapping."""
    COVERED = "covered"
    NO_RELEVANT_TEST_FOUND = "no_relevant_test_found"
    UNTESTED = "untested"


@dataclass
class TestPlan:
    """Strongly typed model representing a validated testing strategy."""
    testing_required: bool = True
    test_files: list[str] = field(default_factory=list)
    test_names: list[str] = field(default_factory=list)
    test_command: list[str] = field(default_factory=list)
    test_type: str = "pytest"
    reason: str = ""
    scope: str = "targeted"  # "targeted", "regression", "none"
    risks: list[str] = field(default_factory=list)
    requires_regression_testing: bool = False
    coverage_status: str = TestCoverageStatus.COVERED.value

    def to_dict(self) -> dict[str, Any]:
        """Returns JSON-serializable dictionary representation of TestPlan."""
        return {
            "testing_required": self.testing_required,
            "test_files": list(self.test_files),
            "test_names": list(self.test_names),
            "test_command": list(self.test_command),
            "test_type": self.test_type,
            "reason": self.reason,
            "scope": self.scope,
            "risks": list(self.risks),
            "requires_regression_testing": self.requires_regression_testing,
            "coverage_status": self.coverage_status,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TestPlan":
        """Safely parses a dictionary into a TestPlan instance."""
        if not isinstance(data, dict):
            return cls(testing_required=False, coverage_status=TestCoverageStatus.NO_RELEVANT_TEST_FOUND.value)
        return cls(
            testing_required=bool(data.get("testing_required", True)),
            test_files=[str(f) for f in data.get("test_files", []) if f],
            test_names=[str(n) for n in data.get("test_names", []) if n],
            test_command=[str(c) for c in data.get("test_command", []) if c],
            test_type=str(data.get("test_type", "pytest")),
            reason=str(data.get("reason", "")),
            scope=str(data.get("scope", "targeted")),
            risks=[str(r) for r in data.get("risks", []) if r],
            requires_regression_testing=bool(data.get("requires_regression_testing", False)),
            coverage_status=str(data.get("coverage_status", TestCoverageStatus.COVERED.value)),
        )


@dataclass
class TestResult:
    """Strongly typed model representing authoritative test execution outcomes."""
    status: str = TestStatus.NOT_RUN.value
    stage: str = TestStage.TEST_NOT_REQUIRED.value
    tests_attempted: int = 0
    tests_passed: int = 0
    tests_failed: int = 0
    tests_skipped: int = 0
    exit_code: int | None = None
    stdout: str = ""
    stderr: str = ""
    duration_seconds: float | None = None
    command: list[str] = field(default_factory=list)
    target: str = ""
    error: str | None = None
    output_truncated: bool = False

    @property
    def passed(self) -> bool:
        """Returns True if test status is strictly PASSED."""
        return self.status == TestStatus.PASSED.value

    def to_dict(self) -> dict[str, Any]:
        """Returns JSON-serializable dictionary representation of TestResult."""
        return {
            "status": self.status,
            "stage": self.stage,
            "tests_attempted": self.tests_attempted,
            "tests_passed": self.tests_passed,
            "tests_failed": self.tests_failed,
            "tests_skipped": self.tests_skipped,
            "exit_code": self.exit_code,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "duration_seconds": self.duration_seconds,
            "command": list(self.command),
            "target": self.target,
            "error": self.error,
            "output_truncated": self.output_truncated,
            "passed": self.passed,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TestResult":
        """Safely parses a dictionary into a TestResult instance."""
        if not isinstance(data, dict):
            return cls(status=TestStatus.NOT_RUN.value)
        return cls(
            status=str(data.get("status", TestStatus.NOT_RUN.value)),
            stage=str(data.get("stage", TestStage.TEST_NOT_REQUIRED.value)),
            tests_attempted=int(data.get("tests_attempted", 0)),
            tests_passed=int(data.get("tests_passed", 0)),
            tests_failed=int(data.get("tests_failed", 0)),
            tests_skipped=int(data.get("tests_skipped", 0)),
            exit_code=int(data["exit_code"]) if data.get("exit_code") is not None else None,
            stdout=str(data.get("stdout", "")),
            stderr=str(data.get("stderr", "")),
            duration_seconds=float(data["duration_seconds"]) if data.get("duration_seconds") is not None else None,
            command=[str(c) for c in data.get("command", []) if c],
            target=str(data.get("target", "")),
            error=str(data["error"]) if data.get("error") is not None else None,
            output_truncated=bool(data.get("output_truncated", False)),
        )


@dataclass
class TestSummary:
    """Concise representation of change application, verification, and testing outcome."""
    target_file: str
    target_symbol: str | None = None
    applied: bool = False
    verified: bool = False
    test_target: str = ""
    test_result: str = TestStatus.NOT_RUN.value
    tests_passed: int = 0
    tests_failed: int = 0
    tests_skipped: int = 0
    details: str = ""

    def format_display(self) -> str:
        """Formats the authoritative Section 18 concise user-facing display."""
        lines = [
            "Change applied:",
            f"- File: {self.target_file}",
        ]
        if self.target_symbol:
            lines.append(f"- Symbol: {self.target_symbol}")
        lines.append(f"- Verification: {'passed' if self.verified else 'failed'}")
        lines.append("")
        lines.append("Tests:")
        if self.test_target:
            lines.append(f"- Target: {self.test_target}")
        lines.append(f"- Result: {self.test_result}")
        if self.test_result in (TestStatus.PASSED.value, TestStatus.FAILED.value):
            lines.append(f"- Passed: {self.tests_passed}")
            lines.append(f"- Failed: {self.tests_failed}")
            if self.tests_skipped > 0:
                lines.append(f"- Skipped: {self.tests_skipped}")
        elif self.details:
            lines.append(f"- Details: {self.details}")
        return "\n".join(lines)
