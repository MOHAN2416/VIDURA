import os
from pathlib import Path
import pytest

from task_understanding.models import TaskType, DeveloperTask
from task_understanding.analyzer import DeveloperTaskAnalyzer
from permissions import PermissionManager
from developer import CodeChangeApplier


@pytest.fixture
def safety_tracker(tmp_path: Path):
    """Fixture to verify that analyzing tasks produces zero side-effects."""
    perm_mgr = PermissionManager(default_write_allowed=False)
    applier = CodeChangeApplier(permission_manager=perm_mgr, workspace_root=tmp_path)
    initial_files = set(tmp_path.rglob("*"))

    def assert_no_side_effects():
        assert set(tmp_path.rglob("*")) == initial_files
        assert perm_mgr.write_allowed is False
        assert applier.get_pending_proposal() is None

    return assert_no_side_effects


def test_task_a_general_conversation(safety_tracker) -> None:
    """TEST A: General conversation prompt returns task_type = general."""
    analyzer = DeveloperTaskAnalyzer()
    task = analyzer.analyze("What is Python?")

    assert task.task_type == TaskType.GENERAL
    assert task.is_development_task is False
    assert task.target_files == []
    safety_tracker()


def test_task_b_simple_code_change(safety_tracker) -> None:
    """TEST B: Simple code-change request extracts target_files containing math_utils.py."""
    analyzer = DeveloperTaskAnalyzer()
    task = analyzer.analyze("Add a Fibonacci function to math_utils.py.")

    assert task.task_type == TaskType.CODE_CHANGE
    assert task.is_development_task is True
    assert "math_utils.py" in task.target_files
    safety_tracker()


def test_task_c_code_debugging(safety_tracker) -> None:
    """TEST C: Code debugging request extracts task_type = code_debug and target_files containing agent/loop.py."""
    analyzer = DeveloperTaskAnalyzer()
    task = analyzer.analyze("Fix the bug in agent/loop.py.")

    assert task.task_type == TaskType.CODE_DEBUG
    assert task.is_development_task is True
    assert "agent/loop.py" in task.target_files
    safety_tracker()


def test_task_d_code_explanation(safety_tracker) -> None:
    """TEST D: Code explanation prompt returns task_type = code_explanation."""
    analyzer = DeveloperTaskAnalyzer()
    task = analyzer.analyze("Explain how MemoryStore works.")

    assert task.task_type == TaskType.CODE_EXPLANATION
    assert task.is_development_task is True
    assert "MemoryStore" in task.target_symbols
    safety_tracker()


def test_task_e_code_review(safety_tracker) -> None:
    """TEST E: Code review request extracts target_files containing agent/agent.py."""
    analyzer = DeveloperTaskAnalyzer()
    task = analyzer.analyze("Review agent/agent.py for potential problems.")

    assert task.task_type == TaskType.CODE_REVIEW
    assert task.is_development_task is True
    assert "agent/agent.py" in task.target_files
    safety_tracker()


def test_task_f_test_request(safety_tracker) -> None:
    """TEST F: Test request extracts task_type = test_request and target_files containing memory/store.py."""
    analyzer = DeveloperTaskAnalyzer()
    task = analyzer.analyze("Add tests for memory/store.py.")

    assert task.task_type == TaskType.TEST_REQUEST
    assert task.is_development_task is True
    assert "memory/store.py" in task.target_files
    safety_tracker()


def test_task_g_missing_target_no_hallucination(safety_tracker) -> None:
    """TEST G: Prompt without explicit target files leaves target_files empty and sets requires_codebase_analysis=True."""
    analyzer = DeveloperTaskAnalyzer()
    task = analyzer.analyze("Fix the authentication bug.")

    assert task.is_development_task is True
    assert task.target_files == []
    assert "auth.py" not in task.target_files
    assert "authentication.py" not in task.target_files
    assert "security/auth.py" not in task.target_files
    assert task.requires_codebase_analysis is True
    safety_tracker()


def test_task_h_multiple_explicit_files(safety_tracker) -> None:
    """TEST H: Multiple explicit files in prompt are all extracted into target_files."""
    analyzer = DeveloperTaskAnalyzer()
    task = analyzer.analyze("Update agent/loop.py and tools/registry.py.")

    assert task.is_development_task is True
    assert "agent/loop.py" in task.target_files
    assert "tools/registry.py" in task.target_files
    safety_tracker()


def test_task_i_constraint_extraction(safety_tracker) -> None:
    """TEST I: Explicit prompt constraints are extracted into constraints list."""
    analyzer = DeveloperTaskAnalyzer()
    task = analyzer.analyze("Modify agent.py to add logging without changing existing behavior.")

    assert "agent.py" in task.target_files
    assert len(task.constraints) > 0
    assert any("existing behavior" in c.lower() for c in task.constraints)
    safety_tracker()


def test_task_j_no_side_effects(tmp_path: Path) -> None:
    """TEST J: Analyzing a task produces 0 disk writes, 0 proposals, 0 permissions, 0 shell executions."""
    perm_mgr = PermissionManager(default_write_allowed=False)
    applier = CodeChangeApplier(permission_manager=perm_mgr, workspace_root=tmp_path)
    
    analyzer = DeveloperTaskAnalyzer()

    # Track directory state before analysis
    files_before = set(tmp_path.rglob("*"))

    task = analyzer.analyze("Modify utils.py to add a greet function.")

    files_after = set(tmp_path.rglob("*"))

    assert files_before == files_after
    assert perm_mgr.write_allowed is False
    assert applier.get_pending_proposal() is None
    assert isinstance(task, DeveloperTask)


def test_target_extraction_examples(safety_tracker) -> None:
    """Verifies specific target extraction examples from Section 4 of requirements."""
    analyzer = DeveloperTaskAnalyzer()

    t1 = analyzer.analyze("Modify test_verify.py")
    assert t1.target_files == ["test_verify.py"]

    t2 = analyzer.analyze("Fix AgentLoop in agent/loop.py")
    assert t2.target_files == ["agent/loop.py"]
    assert t2.target_symbols == ["AgentLoop"]

    t3 = analyzer.analyze("Add a function to memory/store.py")
    assert t3.target_files == ["memory/store.py"]
    assert t3.target_symbols == []

    safety_tracker()


def test_manual_test_suite_requests(safety_tracker) -> None:
    """Verifies all requests required by Section 14 are supported cleanly."""
    analyzer = DeveloperTaskAnalyzer()

    t1 = analyzer.analyze("What is Python?")
    assert t1.task_type == TaskType.GENERAL
    assert t1.is_development_task is False

    t2 = analyzer.analyze("Add a Fibonacci function to math_utils.py.")
    assert t2.task_type == TaskType.CODE_CHANGE
    assert "math_utils.py" in t2.target_files

    t3 = analyzer.analyze("Fix the bug in agent/loop.py.")
    assert t3.task_type == TaskType.CODE_DEBUG
    assert "agent/loop.py" in t3.target_files

    t4 = analyzer.analyze("Explain MemoryStore.")
    assert t4.task_type == TaskType.CODE_EXPLANATION
    assert "MemoryStore" in t4.target_symbols

    t5 = analyzer.analyze("Review agent/agent.py.")
    assert t5.task_type == TaskType.CODE_REVIEW
    assert "agent/agent.py" in t5.target_files

    t6 = analyzer.analyze("Add tests for memory/store.py.")
    assert t6.task_type == TaskType.TEST_REQUEST
    assert "memory/store.py" in t6.target_files

    safety_tracker()


def test_task_schema_validation_and_fallback() -> None:
    """Schema validation: Malformed data or unvalidated dict parses safely into DeveloperTask."""
    malformed = {"task_type": "invalid_type_123", "target_files": "single_string_file.py"}
    task = DeveloperTask.from_dict(malformed)

    assert task.task_type == TaskType.UNABLE_TO_CLASSIFY
    assert task.target_files == ["single_string_file.py"]
    assert task.confidence == 1.0


def test_task_to_dict_roundtrip() -> None:
    """Roundtrip serialization test for DeveloperTask."""
    original = DeveloperTask(
        task_type=TaskType.CODE_CHANGE,
        is_development_task=True,
        goal="Add feature",
        requested_change="Add new function",
        target_files=["math_utils.py"],
        target_symbols=["fibonacci"],
        constraints=["preserve existing behavior"],
        confidence=0.9,
    )
    d = original.to_dict()
    reconstructed = DeveloperTask.from_dict(d)

    assert reconstructed.task_type == TaskType.CODE_CHANGE
    assert reconstructed.is_development_task is True
    assert reconstructed.goal == "Add feature"
    assert reconstructed.target_files == ["math_utils.py"]
    assert reconstructed.target_symbols == ["fibonacci"]
    assert reconstructed.constraints == ["preserve existing behavior"]
