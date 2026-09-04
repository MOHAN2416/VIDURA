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
    ExecutionStage,
    ExecutionStatus,
    DeveloperExecutionResult,
)
from developer.generator import CodeChangeGenerator, DeveloperCodeGenerator
from developer.applier import CodeChangeApplier
from developer.verifier import CodeChangeVerifier
from developer.executor import DeveloperExecutor
from tools.developer import ApplyCodeChangeTool
from tools.registry import ToolRegistry
from agent.state import AgentState
from agent.loop import AgentLoop, claims_file_modification


class MockLLM(BaseLLMProvider):
    """Mock LLM provider for deterministic developer pipeline execution testing."""

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


# =====================================================================
# Specification Tests A through Q for Phase 8.4
# =====================================================================

def test_spec_a_proposal_enters_pending_permission_state(workspace: Path) -> None:
    """A. Proposal enters pending permission state: prepare_proposal() returns stage=PENDING_PERMISSION."""
    applier = CodeChangeApplier(workspace_root=workspace)
    executor = DeveloperExecutor(applier=applier, workspace_root=workspace)

    proposal = CodeChangeProposal(
        target_file="module.py",
        operation=ProposalOperation.CREATE_FILE.value,
        proposed_content="print('hello')\n",
        description="Create module.py",
    )

    result = executor.prepare_proposal(proposal)

    assert result.success is False
    assert result.stage == ExecutionStage.PENDING_PERMISSION.value
    assert result.status == ExecutionStatus.PENDING_PERMISSION.value
    assert result.permission_granted is False
    assert result.target_file == "module.py"
    assert result.operation == ProposalOperation.CREATE_FILE.value
    assert "Permission required before applying" in result.summary
    assert executor.get_pending_proposal() is proposal
    assert proposal.status == "pending"


def test_spec_b_permission_denied_prevents_application(workspace: Path) -> None:
    """B. Permission denied prevents application: execute_proposal(..., explicit_permission=False) halts at stage=DENIED."""
    target_file = workspace / "script.py"
    initial_content = "x = 1\n"
    target_file.write_text(initial_content, encoding="utf-8")

    applier = CodeChangeApplier(workspace_root=workspace)
    executor = DeveloperExecutor(applier=applier, workspace_root=workspace)

    proposal = CodeChangeProposal(
        target_file="script.py",
        operation=ProposalOperation.MODIFY_FILE.value,
        original_content=initial_content,
        proposed_content="x = 2\n",
        description="Update x to 2",
    )
    executor.prepare_proposal(proposal)

    result = executor.execute_proposal(proposal=proposal, explicit_permission=False)

    assert result.success is False
    assert result.stage == ExecutionStage.DENIED.value
    assert result.status == ExecutionStatus.PERMISSION_DENIED.value
    assert result.permission_granted is False
    assert "permission was not granted" in result.summary.lower()
    assert target_file.read_text(encoding="utf-8") == initial_content


def test_spec_c_permission_granted_triggers_application(workspace: Path) -> None:
    """C. Permission granted triggers existing application layer: execute_proposal(..., explicit_permission=True) applies."""
    target_file = workspace / "calculator.py"
    initial_content = "def add(a, b): return a + b\n"
    target_file.write_text(initial_content, encoding="utf-8")

    applier = CodeChangeApplier(workspace_root=workspace)
    executor = DeveloperExecutor(applier=applier, workspace_root=workspace)

    new_content = "def add(a: int, b: int) -> int:\n    return a + b\n"
    proposal = CodeChangeProposal(
        target_file="calculator.py",
        operation=ProposalOperation.MODIFY_FILE.value,
        original_content=initial_content,
        proposed_content=new_content,
        description="Add type hints",
    )
    executor.prepare_proposal(proposal)

    result = executor.execute_proposal(proposal=proposal, explicit_permission=True)

    assert result.success is True
    assert result.stage == ExecutionStage.VERIFIED.value
    assert result.status == ExecutionStatus.SUCCESS.value
    assert result.permission_granted is True
    assert target_file.read_text(encoding="utf-8") == new_content


def test_spec_d_successful_application_triggers_verification(workspace: Path) -> None:
    """D. Successful application triggers verification: Verifier confirms disk match."""
    target_file = workspace / "utils.py"
    target_file.write_text("def helper(): pass\n", encoding="utf-8")

    applier = CodeChangeApplier(workspace_root=workspace)
    executor = DeveloperExecutor(applier=applier, workspace_root=workspace)

    proposal = CodeChangeProposal(
        target_file="utils.py",
        operation=ProposalOperation.MODIFY_FILE.value,
        original_content="def helper(): pass\n",
        proposed_content="def helper() -> None:\n    pass\n",
        description="Annotate helper",
    )
    executor.prepare_proposal(proposal)
    result = executor.execute_proposal(proposal=proposal, explicit_permission=True)

    assert result.verification_result is not None
    assert result.verification_result.verified is True
    assert result.verification_result.target_file == "utils.py"
    assert result.stage == ExecutionStage.VERIFIED.value


def test_spec_e_full_pipeline_permission_application_verification(workspace: Path) -> None:
    """E. Permission + application + verification produces success across complete developer pipeline."""
    target_file = workspace / "main.py"
    target_file.write_text("def start(): pass\n", encoding="utf-8")

    # 1. Developer Task (Phase 8.1)
    task = DeveloperTask(
        task_type=TaskType.CODE_CHANGE,
        is_development_task=True,
        goal="Add logging to start",
        target_files=["main.py"],
    )

    # 2. Developer Plan (Phase 8.2)
    plan = DeveloperPlan(
        original_task=task,
        relevant_files=["main.py"],
        relevant_symbols=["start"],
        planned_changes=["Update start function with logging statement."],
    )

    # 3. Developer Code Generator (Phase 8.3)
    mock_llm = MockLLM(response_text=json.dumps({
        "generated_code": "import logging\n\ndef start():\n    logging.info('Started')\n",
        "explanation": "Added logging",
        "affected_symbols": ["start"],
        "assumptions": [],
    }))
    applier = CodeChangeApplier(workspace_root=workspace)
    generator = DeveloperCodeGenerator(workspace_root=workspace, model=mock_llm, applier=applier)
    gen_result = generator.generate_from_plan(task=task, plan=plan)

    assert gen_result.is_valid is True
    assert gen_result.proposal is not None

    # 4. Developer Execution Pipeline (Phase 8.4)
    executor = DeveloperExecutor(applier=applier, workspace_root=workspace)
    prep_result = executor.prepare_proposal(gen_result.proposal, plan=plan)
    assert prep_result.stage == ExecutionStage.PENDING_PERMISSION.value

    exec_result = executor.execute_proposal(gen_result.proposal, explicit_permission=True)
    assert exec_result.success is True
    assert exec_result.stage == ExecutionStage.VERIFIED.value
    assert "import logging" in target_file.read_text(encoding="utf-8")


def test_spec_f_application_failure_produces_failure(workspace: Path) -> None:
    """F. Application failure produces failure: Read-only disk error or invalid op halts at stage=FAILED."""
    applier = CodeChangeApplier(workspace_root=workspace)
    executor = DeveloperExecutor(applier=applier, workspace_root=workspace)

    # Unsupported operation
    proposal = CodeChangeProposal(
        target_file="test.py",
        operation="delete_file",  # Deletion is strictly unsupported
        original_content="",
        proposed_content="",
        is_valid=False,
        validation_error="Unsupported operation 'delete_file'.",
    )

    result = executor.prepare_proposal(proposal)
    assert result.success is False
    assert result.stage == ExecutionStage.PROPOSAL_INVALID.value
    assert result.status == ExecutionStatus.PROPOSAL_INVALID.value


def test_spec_g_verification_failure_produces_failure(workspace: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """G. Verification failure produces failure: Disk verification mismatch halts at stage=VERIFICATION_FAILED."""
    target_file = workspace / "config.py"
    target_file.write_text("DEBUG = False\n", encoding="utf-8")

    applier = CodeChangeApplier(workspace_root=workspace)
    executor = DeveloperExecutor(applier=applier, workspace_root=workspace)

    proposal = CodeChangeProposal(
        target_file="config.py",
        operation=ProposalOperation.MODIFY_FILE.value,
        original_content="DEBUG = False\n",
        proposed_content="DEBUG = True\n",
        description="Enable debug",
    )
    executor.prepare_proposal(proposal)

    # Simulate post-write verification mismatch by tampering with file after write in verifier
    from developer.verifier import VerificationResult
    monkeypatch.setattr(
        applier.verifier,
        "verify",
        lambda p: VerificationResult(
            success=False,
            target_file="config.py",
            operation="modify_file",
            verified=False,
            expected_state="DEBUG = True\n",
            actual_state="CORRUPTED",
            reason="Content mismatch",
        )
    )

    result = executor.execute_proposal(proposal=proposal, explicit_permission=True)

    assert result.success is False
    assert result.stage == ExecutionStage.VERIFICATION_FAILED.value
    assert result.status == ExecutionStatus.VERIFICATION_FAILED.value


def test_spec_h_stale_proposal_is_rejected(workspace: Path) -> None:
    """H. Stale proposal is rejected: External disk modification between proposal and apply halts at stage=STALE."""
    target_file = workspace / "service.py"
    target_file.write_text("version = 1.0\n", encoding="utf-8")

    applier = CodeChangeApplier(workspace_root=workspace)
    executor = DeveloperExecutor(applier=applier, workspace_root=workspace)

    proposal = CodeChangeProposal(
        target_file="service.py",
        operation=ProposalOperation.MODIFY_FILE.value,
        original_content="version = 1.0\n",
        proposed_content="version = 2.0\n",
        description="Bump version",
    )
    executor.prepare_proposal(proposal)

    # Simulate external file modification on disk
    target_file.write_text("version = 1.1 # concurrent change\n", encoding="utf-8")

    result = executor.execute_proposal(proposal=proposal, explicit_permission=True)

    assert result.success is False
    assert result.stage == ExecutionStage.STALE.value
    assert result.status == ExecutionStatus.STALE_PROPOSAL.value
    assert "stale" in result.summary.lower()
    # Disk remains concurrent content, not proposed content
    assert target_file.read_text(encoding="utf-8") == "version = 1.1 # concurrent change\n"


def test_spec_i_no_filesystem_change_when_permission_denied(workspace: Path) -> None:
    """I. No filesystem change occurs when permission is denied: Byte-for-byte unchanged."""
    target_file = workspace / "secret.py"
    initial_bytes = b"SECRET_KEY = 'unchanged_key_12345'\n"
    target_file.write_bytes(initial_bytes)

    perm_mgr = PermissionManager(default_write_allowed=False)
    applier = CodeChangeApplier(permission_manager=perm_mgr, workspace_root=workspace)
    executor = DeveloperExecutor(applier=applier, permission_manager=perm_mgr, workspace_root=workspace)

    proposal = CodeChangeProposal(
        target_file="secret.py",
        operation=ProposalOperation.MODIFY_FILE.value,
        original_content=initial_bytes.decode("utf-8"),
        proposed_content="SECRET_KEY = 'compromised'\n",
        description="Change key",
    )
    executor.prepare_proposal(proposal)

    # Deny proposal
    deny_result = executor.deny_proposal()
    assert deny_result.stage == ExecutionStage.DENIED.value
    assert target_file.read_bytes() == initial_bytes

    # Attempt execute without permission
    exec_result = executor.execute_proposal(proposal=proposal, explicit_permission=False)
    assert exec_result.success is False
    assert target_file.read_bytes() == initial_bytes


def test_spec_j_model_success_claims_cannot_override_failed_tool(workspace: Path) -> None:
    """J. Model-generated success claims cannot override failed tool results."""
    perm_mgr = PermissionManager(default_write_allowed=False)
    applier = CodeChangeApplier(permission_manager=perm_mgr, workspace_root=workspace)
    registry = ToolRegistry()
    registry.register(ApplyCodeChangeTool(applier=applier))

    # Model attempts to hallucinate success after apply_code_change fails due to permission denial
    class HallucinatingModel(BaseLLMProvider):
        @property
        def model_name(self) -> str: return "hallucinating-model"
        @property
        def provider_name(self) -> str: return "Mock"
        def generate(self, messages: list[dict[str, str]], **kwargs: Any) -> str:
            return '{"action": "respond", "content": "I have successfully applied the change to app.py"}'

    agent = AgentLoop(model=HallucinatingModel(), tool_registry=registry)
    state = AgentState(user_request="Apply proposal")

    final_state = agent.run(state)
    assert "successfully applied and verified" not in final_state.final_response.lower()
    assert "🛑" in final_state.final_response


def test_spec_k_model_modification_claims_cannot_override_absence_of_tool_call() -> None:
    """K. Model-generated modification claims cannot override absence of application tool call."""
    class FabricatingModel(BaseLLMProvider):
        @property
        def model_name(self) -> str: return "fabricating-model"
        @property
        def provider_name(self) -> str: return "Mock"
        def generate(self, messages: list[dict[str, str]], **kwargs: Any) -> str:
            return '{"action": "respond", "content": "I updated the file calculator.py with new logic."}'

    agent = AgentLoop(model=FabricatingModel(), tool_registry=ToolRegistry())
    state = AgentState(user_request="Add subtract function")

    final_state = agent.run(state)
    assert "🛑 No code change was applied" in final_state.final_response


def test_spec_l_proposal_immutability_after_approval(workspace: Path) -> None:
    """L. Proposal content cannot change after approval (immutability)."""
    target_file = workspace / "math_utils.py"
    target_file.write_text("x = 10\n", encoding="utf-8")

    applier = CodeChangeApplier(workspace_root=workspace)
    executor = DeveloperExecutor(applier=applier, workspace_root=workspace)

    proposal = CodeChangeProposal(
        target_file="math_utils.py",
        operation=ProposalOperation.MODIFY_FILE.value,
        original_content="x = 10\n",
        proposed_content="x = 20\n",
        description="Change x to 20",
    )
    executor.prepare_proposal(proposal)

    # Malicious tampering: change proposal content before execution
    tampered_proposal = CodeChangeProposal(
        target_file="math_utils.py",
        operation=ProposalOperation.MODIFY_FILE.value,
        original_content="x = 10\n",
        proposed_content="x = 999 # TAMPERED\n",
        proposal_id=proposal.proposal_id,
        description="Tampered",
    )

    result = executor.execute_proposal(proposal=tampered_proposal, explicit_permission=True)

    assert result.success is False
    assert result.stage == ExecutionStage.FAILED.value
    assert "immutability" in result.error.lower()
    assert target_file.read_text(encoding="utf-8") == "x = 10\n"


def test_spec_m_multiple_proposals_no_cross_contamination(workspace: Path) -> None:
    """M. Multiple proposals cannot cross-contaminate permission state."""
    file_a = workspace / "a.py"
    file_b = workspace / "b.py"
    file_a.write_text("a = 1\n", encoding="utf-8")
    file_b.write_text("b = 1\n", encoding="utf-8")

    applier = CodeChangeApplier(workspace_root=workspace)
    executor = DeveloperExecutor(applier=applier, workspace_root=workspace)

    proposal_a = CodeChangeProposal(
        target_file="a.py",
        operation=ProposalOperation.MODIFY_FILE.value,
        original_content="a = 1\n",
        proposed_content="a = 2\n",
        description="Change a",
    )
    executor.prepare_proposal(proposal_a)
    assert executor.get_pending_proposal() is proposal_a

    proposal_b = CodeChangeProposal(
        target_file="b.py",
        operation=ProposalOperation.MODIFY_FILE.value,
        original_content="b = 1\n",
        proposed_content="b = 2\n",
        description="Change b",
    )
    executor.prepare_proposal(proposal_b)
    # Registering B replaced pending proposal
    assert executor.get_pending_proposal() is proposal_b

    # Attempting to execute proposal A must be rejected due to proposal mismatch
    result_a = executor.execute_proposal(proposal=proposal_a, explicit_permission=True)
    assert result_a.success is False
    assert "mismatch" in result_a.error.lower()
    assert file_a.read_text(encoding="utf-8") == "a = 1\n"


def test_spec_n_invalid_proposal_cannot_execute(workspace: Path) -> None:
    """N. Invalid proposal cannot execute: is_valid=False rejected before application."""
    applier = CodeChangeApplier(workspace_root=workspace)
    executor = DeveloperExecutor(applier=applier, workspace_root=workspace)

    invalid_proposal = CodeChangeProposal(
        target_file="",  # Missing target file
        operation=ProposalOperation.MODIFY_FILE.value,
        proposed_content="test",
        is_valid=False,
        validation_error="Missing target_file.",
    )

    result = executor.execute_proposal(proposal=invalid_proposal, explicit_permission=True)
    assert result.success is False
    assert result.stage == ExecutionStage.FAILED.value
    assert result.status == ExecutionStatus.PROPOSAL_INVALID.value


def test_spec_o_unknown_target_cannot_execute(workspace: Path) -> None:
    """O. Unknown target cannot execute: Path traversal target rejected."""
    applier = CodeChangeApplier(workspace_root=workspace)
    executor = DeveloperExecutor(applier=applier, workspace_root=workspace)

    traversal_proposal = CodeChangeProposal(
        target_file="../../../etc/passwd",
        operation=ProposalOperation.MODIFY_FILE.value,
        proposed_content="malicious",
        description="Path traversal",
    )

    result = executor.prepare_proposal(traversal_proposal)
    assert result.success is False
    assert result.stage == ExecutionStage.PROPOSAL_INVALID.value


def test_spec_p_existing_phase7_behavior_unchanged(workspace: Path) -> None:
    """P. Existing Phase 7 behavior remains unchanged: CodeChangeApplier and CodeChangeVerifier direct tests pass."""
    target_file = workspace / "p7_test.py"
    target_file.write_text("original", encoding="utf-8")

    perm_mgr = PermissionManager(default_write_allowed=False)
    applier = CodeChangeApplier(permission_manager=perm_mgr, workspace_root=workspace)
    verifier = CodeChangeVerifier(workspace_root=workspace)

    proposal = CodeChangeProposal(
        target_file="p7_test.py",
        operation=ProposalOperation.MODIFY_FILE.value,
        original_content="original",
        proposed_content="updated",
    )
    applier.set_pending_proposal(proposal)

    # Denied by default
    res_denied = applier.apply_proposal(proposal)
    assert res_denied.success is False
    assert res_denied.permission_granted is False

    # Granted
    perm_mgr.grant_write_permission()
    res_approved = applier.apply_proposal(proposal)
    assert res_approved.success is True
    assert res_approved.verification_success is True

    # Direct verifier check
    ver_res = verifier.verify(proposal)
    assert ver_res.verified is True


def test_spec_q_deny_proposal_clears_state(workspace: Path) -> None:
    """Q. Denying proposal clears active pending proposal and revokes write permission."""
    perm_mgr = PermissionManager(default_write_allowed=False)
    applier = CodeChangeApplier(permission_manager=perm_mgr, workspace_root=workspace)
    executor = DeveloperExecutor(applier=applier, permission_manager=perm_mgr, workspace_root=workspace)

    proposal = CodeChangeProposal(
        target_file="deny_me.py",
        operation=ProposalOperation.CREATE_FILE.value,
        proposed_content="content",
    )
    executor.prepare_proposal(proposal)
    assert executor.get_pending_proposal() is proposal

    perm_mgr.grant_write_permission()
    assert perm_mgr.is_write_allowed("deny_me.py", "create_file") is True

    deny_res = executor.deny_proposal()
    assert deny_res.stage == ExecutionStage.DENIED.value
    assert deny_res.permission_granted is False
    assert executor.get_pending_proposal() is None
    assert perm_mgr.is_write_allowed("deny_me.py", "create_file") is False


def test_execution_result_serialization_roundtrip() -> None:
    """Verify DeveloperExecutionResult to_dict and from_dict roundtrip preservation."""
    proposal = CodeChangeProposal(
        target_file="serialize.py",
        operation=ProposalOperation.CREATE_FILE.value,
        proposed_content="print('ok')\n",
        proposal_id="prop-roundtrip-1",
    )
    res = DeveloperExecutionResult(
        success=True,
        status=ExecutionStatus.SUCCESS.value,
        stage=ExecutionStage.VERIFIED.value,
        target_file="serialize.py",
        operation="create_file",
        proposal_id="prop-roundtrip-1",
        proposal=proposal,
        permission_granted=True,
        summary="Applied and verified.",
        error=None,
    )

    data = res.to_dict()
    restored = DeveloperExecutionResult.from_dict(data)

    assert restored.success is True
    assert restored.status == ExecutionStatus.SUCCESS.value
    assert restored.stage == ExecutionStage.VERIFIED.value
    assert restored.target_file == "serialize.py"
    assert restored.operation == "create_file"
    assert restored.proposal_id == "prop-roundtrip-1"
    assert restored.proposal is not None
    assert restored.proposal.proposal_id == "prop-roundtrip-1"
    assert restored.permission_granted is True
    assert restored.summary == "Applied and verified."
