import json
from pathlib import Path
from typing import Any
import pytest

from models.base import BaseLLMProvider
from permissions import PermissionManager
from task_understanding.models import DeveloperTask, TaskType
from planning.models import DeveloperPlan
from developer.models import (
    CodeChangeProposal,
    ProposalOperation,
    ProposalStatus,
    DeveloperGenerationResult,
)
from developer.generator import CodeChangeGenerator, DeveloperCodeGenerator
from developer.applier import CodeChangeApplier
from agent.state import AgentState
from agent.loop import AgentLoop


class MockLLM(BaseLLMProvider):
    """Mock LLM provider for deterministic developer code generation testing."""

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
        if self.response_text:
            return self.response_text
        return json.dumps({
            "generated_code": "def mock_function():\n    return 'mock'\n",
            "explanation": "Implemented mock function.",
            "affected_symbols": ["mock_function"],
            "assumptions": ["Assumed standard library only"],
        })


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    """Fixture providing a temporary workspace directory."""
    ws = tmp_path / "test_ws"
    ws.mkdir(parents=True, exist_ok=True)
    return ws


def test_modify_existing_file_generation(workspace: Path) -> None:
    """Test 1: Modify existing file generates modify_file proposal and preserves disk content."""
    calc_file = workspace / "calculator.py"
    original_code = "def add(a: int, b: int) -> int:\n    return a + b\n"
    calc_file.write_text(original_code, encoding="utf-8")

    task = DeveloperTask(
        task_type=TaskType.CODE_CHANGE,
        is_development_task=True,
        goal="Add subtract function to calculator.py",
        requested_change="Add subtract function",
        target_files=["calculator.py"],
        target_symbols=["subtract"],
    )
    plan = DeveloperPlan(
        original_task=task,
        relevant_files=["calculator.py"],
        relevant_symbols=["subtract"],
        planned_changes=["Add subtract function to calculator.py"],
        constraints=["Preserve existing add function"],
    )

    generator = DeveloperCodeGenerator(workspace_root=workspace)
    result = generator.generate_from_plan(task, plan)

    assert result.is_valid is True
    assert result.validation_status == "valid"
    assert result.operation == ProposalOperation.MODIFY_FILE.value
    assert result.target_file == "calculator.py"
    assert result.original_content == original_code
    assert "def subtract" in result.proposed_content
    assert "def add" in result.proposed_content
    assert result.proposal is not None
    assert result.proposal.status == ProposalStatus.PROPOSED.value

    # ZERO SIDE-EFFECT MANDATE: Disk file remains completely untouched!
    assert calc_file.read_text(encoding="utf-8") == original_code


def test_create_new_file_authorized(workspace: Path) -> None:
    """Test 2: Authorized creation of a new file generates create_file proposal without disk write."""
    new_file = workspace / "logger.py"
    assert not new_file.exists()

    task = DeveloperTask(
        task_type=TaskType.CODE_CHANGE,
        is_development_task=True,
        goal="Create new logging utility module logger.py",
        requested_change="Create file logger.py",
        target_files=["logger.py"],
        target_symbols=["setup_logger"],
    )
    plan = DeveloperPlan(
        original_task=task,
        relevant_files=["logger.py"],
        relevant_symbols=["setup_logger"],
        planned_changes=["Create file logger.py with basic logger setup"],
    )

    generator = DeveloperCodeGenerator(workspace_root=workspace)
    result = generator.generate_from_plan(task, plan)

    assert result.is_valid is True
    assert result.validation_status == "valid"
    assert result.operation == ProposalOperation.CREATE_FILE.value
    assert result.target_file == "logger.py"
    assert result.original_content == ""
    assert "def setup_logger" in result.proposed_content
    assert result.proposal is not None
    assert result.proposal.operation == "create_file"

    # ZERO SIDE-EFFECT MANDATE: New file does NOT exist on disk!
    assert not new_file.exists()


def test_requires_more_information_halts_generation(workspace: Path) -> None:
    """Test 3: Plan requiring more information halts generation immediately with zero speculative code."""
    task = DeveloperTask(
        task_type=TaskType.CODE_CHANGE,
        is_development_task=True,
        goal="Refactor helper functions",
        target_files=["helpers.py"],
    )
    plan = DeveloperPlan(
        original_task=task,
        relevant_files=["helpers.py"],
        requires_more_information=True,
        missing_information_reason="Target helper module is ambiguous and requirements are underspecified.",
    )

    generator = DeveloperCodeGenerator(workspace_root=workspace)
    result = generator.generate_from_plan(task, plan)

    assert result.is_valid is False
    assert result.validation_status == "requires_information"
    assert result.proposal is None
    assert result.generated_code == ""
    assert len(result.errors) > 0
    assert "ambiguous" in result.errors[0].lower()


def test_unauthorized_file_creation_rejected(workspace: Path) -> None:
    """Test 4: Nonexistent file on modification task without creation authorization is rejected."""
    task = DeveloperTask(
        task_type=TaskType.CODE_DEBUG,
        is_development_task=True,
        goal="Fix syntax bug in non_existent.py",
        target_files=["non_existent.py"],
    )
    plan = DeveloperPlan(
        original_task=task,
        relevant_files=["non_existent.py"],
        planned_changes=["Fix syntax bug in non_existent.py"],
    )

    generator = DeveloperCodeGenerator(workspace_root=workspace)
    result = generator.generate_from_plan(task, plan)

    assert result.is_valid is False
    assert result.validation_status == "target_not_found"
    assert result.proposal is None
    assert "does not exist" in result.errors[0].lower()


def test_clean_code_sanitization_strips_markdown_fences(workspace: Path) -> None:
    """Test 5: Strips markdown fences, backticks, and partial placeholders from LLM outputs."""
    target_file = workspace / "service.py"
    target_file.write_text("# Existing service\n", encoding="utf-8")

    # LLM returns code wrapped in ```python ... ```
    raw_llm_output = json.dumps({
        "generated_code": "```python\n# Clean Python Code\ndef serve():\n    return True\n```",
        "explanation": "Added serve function with clean Python.",
        "affected_symbols": ["serve"],
        "assumptions": ["Python 3.10+"],
    })
    mock_model = MockLLM(response_text=raw_llm_output)

    task = DeveloperTask(
        task_type=TaskType.CODE_CHANGE,
        is_development_task=True,
        goal="Add serve function",
        target_files=["service.py"],
    )
    plan = DeveloperPlan(
        original_task=task,
        relevant_files=["service.py"],
        planned_changes=["Add serve function"],
    )

    generator = DeveloperCodeGenerator(workspace_root=workspace, model=mock_model)
    result = generator.generate_from_plan(task, plan)

    assert result.is_valid is True
    assert "```" not in result.proposed_content
    assert result.proposed_content.startswith("# Clean Python Code")
    assert "def serve():" in result.proposed_content


def test_constraint_preservation_in_generation(workspace: Path) -> None:
    """Test 6: DeveloperTask and DeveloperPlan constraints are integrated into generation context."""
    target_file = workspace / "math_ops.py"
    target_file.write_text("def multiply(a, b): return a * b\n", encoding="utf-8")

    task = DeveloperTask(
        task_type=TaskType.CODE_CHANGE,
        is_development_task=True,
        goal="Add divide function",
        target_files=["math_ops.py"],
        constraints=["Handle division by zero gracefully"],
    )
    plan = DeveloperPlan(
        original_task=task,
        relevant_files=["math_ops.py"],
        planned_changes=["Add divide function"],
        constraints=["Preserve multiply function signature", "Do not import external math libraries"],
    )

    mock_model = MockLLM()
    generator = DeveloperCodeGenerator(workspace_root=workspace, model=mock_model)
    result = generator.generate_from_plan(task, plan)

    assert result.is_valid is True
    # Verify constraints were passed in prompt to LLM
    user_prompt = mock_model.captured_messages[-1]["content"]
    assert "Handle division by zero gracefully" in user_prompt
    assert "Preserve multiply function signature" in user_prompt
    assert "Do not import external math libraries" in user_prompt


def test_zero_side_effects_on_filesystem(workspace: Path) -> None:
    """Test 7: Verification that generation leaves workspace completely unchanged."""
    f1 = workspace / "a.py"
    f1.write_text("print('a')\n", encoding="utf-8")
    f2 = workspace / "b.py"
    f2.write_text("print('b')\n", encoding="utf-8")

    initial_files = {p: p.read_bytes() for p in workspace.rglob("*") if p.is_file()}

    task = DeveloperTask(
        task_type=TaskType.CODE_CHANGE,
        is_development_task=True,
        goal="Update a.py and create c.py",
        requested_change="Create file c.py",
        target_files=["c.py"],
    )
    plan = DeveloperPlan(
        original_task=task,
        relevant_files=["c.py"],
        planned_changes=["Create file c.py"],
    )

    generator = DeveloperCodeGenerator(workspace_root=workspace)
    result = generator.generate_from_plan(task, plan)

    assert result.is_valid is True
    # Verify filesystem is 100% identical to initial snapshot
    current_files = {p: p.read_bytes() for p in workspace.rglob("*") if p.is_file()}
    assert current_files == initial_files
    assert not (workspace / "c.py").exists()


def test_security_rejections(workspace: Path) -> None:
    """Test 8: Rejects protected files and paths outside workspace."""
    generator = DeveloperCodeGenerator(workspace_root=workspace)

    # 1. Protected file (.env)
    task_env = DeveloperTask(goal="Modify .env", target_files=[".env"])
    plan_env = DeveloperPlan(original_task=task_env, relevant_files=[".env"])
    res_env = generator.generate_from_plan(task_env, plan_env)
    assert res_env.is_valid is False
    assert "protected" in res_env.errors[0].lower()

    # 2. Path traversal outside workspace
    task_trav = DeveloperTask(goal="Modify external", target_files=["../../etc/passwd"])
    plan_trav = DeveloperPlan(original_task=task_trav, relevant_files=["../../etc/passwd"])
    res_trav = generator.generate_from_plan(task_trav, plan_trav)
    assert res_trav.is_valid is False
    assert "outside the allowed workspace" in res_trav.errors[0].lower()


def test_phase_7_permission_pipeline_integration(workspace: Path) -> None:
    """Test 9: Generated proposal integrates cleanly with Phase 7 permission-controlled application."""
    test_file = workspace / "workflow.py"
    initial_content = "def step_one(): pass\n"
    test_file.write_text(initial_content, encoding="utf-8")

    perm_mgr = PermissionManager(default_write_allowed=False)
    applier = CodeChangeApplier(permission_manager=perm_mgr, workspace_root=workspace)
    generator = DeveloperCodeGenerator(workspace_root=workspace, applier=applier)

    task = DeveloperTask(
        task_type=TaskType.CODE_CHANGE,
        is_development_task=True,
        goal="Add step_two to workflow.py",
        target_files=["workflow.py"],
        target_symbols=["step_two"],
    )
    plan = DeveloperPlan(
        original_task=task,
        relevant_files=["workflow.py"],
        relevant_symbols=["step_two"],
        planned_changes=["Add step_two to workflow.py"],
    )

    # 1. Generate code from plan
    gen_result = generator.generate_from_plan(task, plan)
    assert gen_result.is_valid is True
    assert applier.get_pending_proposal() is not None

    # Disk remains UNCHANGED
    assert test_file.read_text(encoding="utf-8") == initial_content

    # 2. Attempt applying proposal WITHOUT explicit permission -> MUST FAIL
    proposal = applier.get_pending_proposal()
    apply_fail = applier.apply_proposal(proposal=proposal, explicit_permission=False)
    assert apply_fail.success is False
    assert apply_fail.permission_granted is False
    assert test_file.read_text(encoding="utf-8") == initial_content

    # 3. Apply proposal WITH explicit permission -> MUST SUCCEED AND VERIFY
    perm_mgr.grant_write_permission()
    apply_success = applier.apply_proposal(proposal=proposal, explicit_permission=True)
    assert apply_success.success is True
    assert apply_success.permission_granted is True
    assert apply_success.verification_success is True

    # Disk now contains the modified content
    final_disk_content = test_file.read_text(encoding="utf-8")
    assert "def step_two" in final_disk_content
    assert "def step_one" in final_disk_content


def test_serialization_and_deserialization() -> None:
    """Test 10: Serialization and deserialization of DeveloperGenerationResult."""
    proposal = CodeChangeProposal(
        operation="modify_file",
        target_file="test.py",
        proposed_content="print(1)",
        original_content="",
    )
    res = DeveloperGenerationResult(
        target_file="test.py",
        operation="modify_file",
        original_content="",
        proposed_content="print(1)",
        generated_code="print(1)",
        explanation="Test explanation",
        affected_symbols=["test_sym"],
        assumptions=["Test assumption"],
        validation_status="valid",
        is_valid=True,
        proposal=proposal,
        errors=[],
    )

    data = res.to_dict()
    assert data["target_file"] == "test.py"
    assert data["operation"] == "modify_file"
    assert data["proposal"]["target_file"] == "test.py"

    recovered = DeveloperGenerationResult.from_dict(data)
    assert recovered.target_file == "test.py"
    assert recovered.is_valid is True
    assert recovered.proposal is not None
    assert recovered.proposal.target_file == "test.py"

    # Malformed deserialization check
    fallback = DeveloperGenerationResult.from_dict("invalid")
    assert fallback.is_valid is False
    assert fallback.validation_status == "invalid"


def test_agent_state_integration() -> None:
    """Test 11: AgentState holds generation_result and AgentLoop has developer_generator."""
    state = AgentState(user_request="Test developer request")
    assert hasattr(state, "generation_result")
    assert state.generation_result is None

    loop = AgentLoop(model=MockLLM())
    assert hasattr(loop, "developer_generator")
    assert isinstance(loop.developer_generator, DeveloperCodeGenerator)
    assert isinstance(loop.developer_generator, CodeChangeGenerator)
