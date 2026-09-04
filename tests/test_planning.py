import os
from pathlib import Path
import pytest

from task_understanding.models import TaskType, DeveloperTask
from task_understanding.analyzer import DeveloperTaskAnalyzer
from planning.models import DeveloperPlan
from planning.planner import DeveloperPlanner
from permissions import PermissionManager
from developer import CodeChangeApplier


@pytest.fixture
def safety_checker(tmp_path: Path):
    """Fixture to verify that planning produces zero side-effects."""
    perm_mgr = PermissionManager(default_write_allowed=False)
    applier = CodeChangeApplier(permission_manager=perm_mgr, workspace_root=tmp_path)
    initial_files = set(tmp_path.rglob("*"))

    def assert_no_side_effects():
        assert set(tmp_path.rglob("*")) == initial_files
        assert perm_mgr.write_allowed is False
        assert applier.get_pending_proposal() is None

    return assert_no_side_effects


def test_plan_a_known_file(safety_checker) -> None:
    """TEST A: Known file 'agent/loop.py' is inspected, AgentLoop is identified, dependencies extracted."""
    analyzer = DeveloperTaskAnalyzer()
    planner = DeveloperPlanner()

    task = analyzer.analyze("Modify agent/loop.py to improve tool routing.")
    plan = planner.plan(task)

    assert "agent/loop.py" in plan.relevant_files
    assert "AgentLoop" in plan.relevant_symbols
    assert len(plan.dependencies) > 0
    assert any("tools.registry" in dep for dep in plan.dependencies)
    assert plan.requires_more_information is False
    safety_checker()


def test_plan_b_known_symbol(safety_checker) -> None:
    """TEST B: Known symbol 'AgentLoop' is resolved to its actual module and file."""
    analyzer = DeveloperTaskAnalyzer()
    planner = DeveloperPlanner()

    task = analyzer.analyze("Improve AgentLoop.")
    plan = planner.plan(task)

    assert "AgentLoop" in plan.relevant_symbols
    assert "agent/loop.py" in plan.relevant_files
    assert plan.requires_more_information is False
    safety_checker()


def test_plan_c_dependency_analysis(safety_checker) -> None:
    """TEST C: Dependency analysis on tools.registry retrieves authoritative relationships from codebase index."""
    analyzer = DeveloperTaskAnalyzer()
    planner = DeveloperPlanner()

    task = analyzer.analyze("Update tools/registry.py to support new tool validation.")
    plan = planner.plan(task)

    assert "tools/registry.py" in plan.relevant_files
    assert "ToolRegistry" in plan.relevant_symbols
    assert "tools.base" in plan.dependencies
    assert any("agent.loop" in imp for imp in plan.affected_components)
    safety_checker()


def test_plan_d_unknown_target(safety_checker) -> None:
    """TEST D: Unknown target 'Fix the authentication bug.' does NOT fabricate files and sets requires_more_information=True."""
    analyzer = DeveloperTaskAnalyzer()
    planner = DeveloperPlanner()

    task = analyzer.analyze("Fix the authentication bug.")
    plan = planner.plan(task)

    assert "auth.py" not in plan.relevant_files
    assert "authentication.py" not in plan.relevant_files
    assert "security/auth.py" not in plan.relevant_files
    assert len(plan.relevant_files) == 0
    assert plan.requires_more_information is True
    assert plan.missing_information_reason is not None
    safety_checker()


def test_plan_e_multiple_files(safety_checker) -> None:
    """TEST E: Multiple files 'agent/loop.py' and 'tools/registry.py' are both represented in plan with relationships."""
    analyzer = DeveloperTaskAnalyzer()
    planner = DeveloperPlanner()

    task = analyzer.analyze("Update agent/loop.py and tools/registry.py.")
    plan = planner.plan(task)

    assert "agent/loop.py" in plan.relevant_files
    assert "tools/registry.py" in plan.relevant_files
    assert "AgentLoop" in plan.relevant_symbols
    assert "ToolRegistry" in plan.relevant_symbols
    safety_checker()


def test_plan_f_constraint_preservation(safety_checker) -> None:
    """TEST F: Explicit constraint 'without changing existing behavior' is preserved in the plan."""
    analyzer = DeveloperTaskAnalyzer()
    planner = DeveloperPlanner()

    task = analyzer.analyze("Modify agent.py without changing existing behavior.")
    plan = planner.plan(task)

    assert len(plan.constraints) > 0
    assert any("existing behavior" in c.lower() for c in plan.constraints)
    safety_checker()


def test_plan_g_no_side_effects(tmp_path: Path) -> None:
    """TEST G: Planning produces 0 file changes, 0 proposals, 0 permissions, 0 write tools."""
    perm_mgr = PermissionManager(default_write_allowed=False)
    applier = CodeChangeApplier(permission_manager=perm_mgr, workspace_root=tmp_path)

    analyzer = DeveloperTaskAnalyzer()
    planner = DeveloperPlanner(workspace_root=tmp_path)

    files_before = set(tmp_path.rglob("*"))

    task = analyzer.analyze("Modify agent/loop.py to add new telemetry.")
    plan = planner.plan(task)

    files_after = set(tmp_path.rglob("*"))

    assert files_before == files_after
    assert perm_mgr.write_allowed is False
    assert applier.get_pending_proposal() is None
    assert isinstance(plan, DeveloperPlan)


def test_plan_h_hallucinated_symbol(safety_checker) -> None:
    """TEST H: Hallucinated/nonexistent symbol is not treated as real and reported as unresolved."""
    planner = DeveloperPlanner()

    task = DeveloperTask(
        task_type=TaskType.CODE_DEBUG,
        goal="Fix NonExistentSymbolXYZ",
        target_symbols=["NonExistentSymbolXYZ"],
        is_development_task=True,
    )
    plan = planner.plan(task)

    assert "NonExistentSymbolXYZ" not in plan.relevant_symbols
    assert plan.requires_more_information is True
    assert "NonExistentSymbolXYZ" in (plan.missing_information_reason or "")
    safety_checker()


def test_plan_i_hallucinated_file(safety_checker) -> None:
    """TEST I: Hallucinated/nonexistent file is not treated as existing and reported as missing."""
    planner = DeveloperPlanner()

    task = DeveloperTask(
        task_type=TaskType.CODE_CHANGE,
        goal="Modify nonexistent/ghost.py",
        target_files=["nonexistent/ghost.py"],
        is_development_task=True,
    )
    plan = planner.plan(task)

    assert "nonexistent/ghost.py" not in plan.relevant_files
    assert plan.requires_more_information is True
    assert "nonexistent/ghost.py" in (plan.missing_information_reason or "")
    safety_checker()


def test_manual_test_suite_planning(safety_checker) -> None:
    """Verifies all 5 manual test scenarios required by Section 16."""
    analyzer = DeveloperTaskAnalyzer()
    planner = DeveloperPlanner()

    # 1. "Explain how MemoryStore works."
    t1 = analyzer.analyze("Explain how MemoryStore works.")
    p1 = planner.plan(t1)
    assert "MemoryStore" in p1.relevant_symbols
    assert "memory/store.py" in p1.relevant_files

    # 2. "Improve AgentLoop tool routing."
    t2 = analyzer.analyze("Improve AgentLoop tool routing.")
    p2 = planner.plan(t2)
    assert "AgentLoop" in p2.relevant_symbols
    assert "agent/loop.py" in p2.relevant_files

    # 3. "Fix the authentication bug."
    t3 = analyzer.analyze("Fix the authentication bug.")
    p3 = planner.plan(t3)
    assert p3.requires_more_information is True
    assert "auth.py" not in p3.relevant_files

    # 4. "Modify agent/loop.py and tools/registry.py to improve tool handling."
    t4 = analyzer.analyze("Modify agent/loop.py and tools/registry.py to improve tool handling.")
    p4 = planner.plan(t4)
    assert "agent/loop.py" in p4.relevant_files
    assert "tools/registry.py" in p4.relevant_files

    # 5. "Modify agent.py without changing existing behavior."
    t5 = analyzer.analyze("Modify agent.py without changing existing behavior.")
    p5 = planner.plan(t5)
    assert any("existing behavior" in c.lower() for c in p5.constraints)

    safety_checker()


def test_plan_validation_rules() -> None:
    """Verifies DeveloperPlan.validate() catches invalid plans and executable changes."""
    task = DeveloperTask(task_type=TaskType.CODE_CHANGE, goal="Test task")
    valid_plan = DeveloperPlan(
        original_task=task,
        relevant_files=["agent/loop.py"],
        planned_changes=["Inspect agent/loop.py."],
        confidence=0.9,
    )
    is_valid, err = valid_plan.validate()
    assert is_valid is True
    assert err is None

    # Invalid confidence
    invalid_conf = DeveloperPlan(original_task=task, confidence=1.5)
    is_valid, err = invalid_conf.validate()
    assert is_valid is False
    assert "Confidence" in (err or "")

    # Executable operation in planned changes
    invalid_exec = DeveloperPlan(
        original_task=task,
        planned_changes=["Run apply_code_change to modify files."],
    )
    is_valid, err = invalid_exec.validate()
    assert is_valid is False
    assert "executable operations" in (err or "")


def test_plan_serialization_roundtrip() -> None:
    """Verifies to_dict and from_dict roundtrip serialization for DeveloperPlan."""
    task = DeveloperTask(
        task_type=TaskType.CODE_CHANGE,
        goal="Add logging",
        target_files=["agent.py"],
        constraints=["preserve existing behavior"],
        is_development_task=True,
    )
    original = DeveloperPlan(
        original_task=task,
        relevant_files=["agent.py"],
        relevant_symbols=["Agent"],
        dependencies=["agent.loop"],
        affected_components=["main"],
        planned_changes=["Inspect agent.py logging."],
        risks=["Modifying agent.py affects main."],
        constraints=["preserve existing behavior"],
        requires_more_information=False,
        confidence=0.95,
    )

    d = original.to_dict()
    reconstructed = DeveloperPlan.from_dict(d)

    assert reconstructed.original_task.goal == "Add logging"
    assert reconstructed.relevant_files == ["agent.py"]
    assert reconstructed.relevant_symbols == ["Agent"]
    assert reconstructed.dependencies == ["agent.loop"]
    assert reconstructed.affected_components == ["main"]
    assert reconstructed.risks == ["Modifying agent.py affects main."]
    assert reconstructed.constraints == ["preserve existing behavior"]
    assert reconstructed.requires_more_information is False
    assert reconstructed.confidence == 0.95
