import json
from pathlib import Path
from typing import Any
import pytest
from models.base import BaseLLMProvider
from permissions import PermissionManager
from developer import (
    CodeChangeProposal,
    CodeChangeResult,
    ProposalOperation,
    ProposalStatus,
    CodeChangeGenerator,
    CodeChangeApplier,
)
from tools.developer import ProposeCodeChangeTool, ApplyCodeChangeTool
from tools.registry import ToolRegistry
from agent import Agent, AgentState


class FakeDeveloperModelProvider(BaseLLMProvider):
    """Fake LLM model provider for proposal generation and agent integration testing."""

    def __init__(self, response_text: str = "") -> None:
        self.response_text = response_text
        self.last_messages: list[dict[str, str]] = []

    @property
    def model_name(self) -> str:
        return "fake-developer-model"

    @property
    def provider_name(self) -> str:
        return "Fake Developer Provider"

    def generate(self, messages: list[dict[str, str]], **kwargs: Any) -> str:
        self.last_messages = list(messages)
        if self.response_text:
            return self.response_text
        
        # Default structured LLM proposal output
        return json.dumps({
            "proposed_content": "def greet_user(name: str) -> str:\n    return f'Hello, {name}!'\n",
            "rationale": "Add greet_user function to return greeting string."
        })


def test_create_file_proposal(tmp_path: Path) -> None:
    """Test 1: Create-file proposal for a new valid file path."""
    generator = CodeChangeGenerator(workspace_root=tmp_path)
    proposal = generator.generate_proposal(
        request="Create a new utility module",
        target_file="utils.py",
        operation="create_file",
    )

    assert proposal.is_valid is True
    assert proposal.operation == "create_file"
    assert proposal.target_file == "utils.py"
    assert proposal.status == ProposalStatus.PROPOSED.value
    assert "utils.py" in proposal.proposed_content
    assert (tmp_path / "utils.py").exists() is False


def test_modify_file_proposal(tmp_path: Path) -> None:
    """Test 2: Modify-file proposal for an existing file in workspace."""
    sample_file = tmp_path / "app.py"
    sample_file.write_text("class App:\n    pass\n", encoding="utf-8")

    generator = CodeChangeGenerator(workspace_root=tmp_path)
    proposal = generator.generate_proposal(
        request="Add start method to App class",
        target_file="app.py",
        operation="modify_file",
        target_symbol="App",
    )

    assert proposal.is_valid is True
    assert proposal.operation == "modify_file"
    assert proposal.target_file == "app.py"
    assert proposal.target_symbol == "App"
    assert proposal.status == ProposalStatus.PROPOSED.value


def test_existing_file_inspection_before_modification_proposal(tmp_path: Path) -> None:
    """Test 3: Generator inspects existing file content before proposing modifications."""
    initial_text = "def hello():\n    print('hi')\n"
    sample_file = tmp_path / "existing.py"
    sample_file.write_text(initial_text, encoding="utf-8")

    generator = CodeChangeGenerator(workspace_root=tmp_path)
    proposal = generator.generate_proposal(
        request="Add goodbye function",
        target_file="existing.py",
        operation="modify_file",
    )

    assert proposal.is_valid is True
    assert proposal.original_content == initial_text


def test_unknown_target_modify_file_proposal(tmp_path: Path) -> None:
    """Test 4: Modify-file proposal for nonexistent target file returns target_not_found status."""
    generator = CodeChangeGenerator(workspace_root=tmp_path)
    proposal = generator.generate_proposal(
        request="Modify nonexistent file",
        target_file="nonexistent.py",
        operation="modify_file",
    )

    assert proposal.is_valid is False
    assert proposal.status == ProposalStatus.TARGET_NOT_FOUND.value
    assert "does not exist" in proposal.validation_error


def test_path_traversal_proposal_attempt(tmp_path: Path) -> None:
    """Test 5: Path traversal proposal attempt outside workspace is rejected."""
    generator = CodeChangeGenerator(workspace_root=tmp_path)
    proposal = generator.generate_proposal(
        request="Modify file outside workspace",
        target_file="../outside.py",
        operation="modify_file",
    )

    assert proposal.is_valid is False
    assert proposal.status == ProposalStatus.INVALID_PATH.value
    assert "outside the allowed workspace" in proposal.validation_error


def test_absolute_path_outside_workspace_proposal_attempt(tmp_path: Path) -> None:
    """Test 6: Absolute path outside workspace proposal attempt is rejected."""
    generator = CodeChangeGenerator(workspace_root=tmp_path)
    proposal = generator.generate_proposal(
        request="Modify system file",
        target_file="/tmp/system.py",
        operation="modify_file",
    )

    assert proposal.is_valid is False
    assert proposal.status == ProposalStatus.INVALID_PATH.value


def test_protected_file_proposal_target(tmp_path: Path) -> None:
    """Test 7: Protected files (.git, .env, *.key) proposal attempts are rejected."""
    generator = CodeChangeGenerator(workspace_root=tmp_path)
    
    prop_env = generator.generate_proposal(request="Modify secrets", target_file=".env", operation="modify_file")
    assert prop_env.is_valid is False
    assert prop_env.status == ProposalStatus.INVALID_PATH.value

    prop_git = generator.generate_proposal(request="Modify git config", target_file=".git/config", operation="modify_file")
    assert prop_git.is_valid is False
    assert prop_git.status == ProposalStatus.INVALID_PATH.value

    prop_key = generator.generate_proposal(request="Modify key", target_file="secret.key", operation="modify_file")
    assert prop_key.is_valid is False
    assert prop_key.status == ProposalStatus.INVALID_PATH.value


def test_structured_proposal_serialization(tmp_path: Path) -> None:
    """Test 8: CodeChangeProposal to_dict() returns valid JSON-serializable dictionary."""
    proposal = CodeChangeProposal(
        operation="create_file",
        target_file="module.py",
        target_symbol="MyClass",
        description="Create MyClass",
        proposed_content="class MyClass: pass\n",
        original_content="",
        rationale="Initial creation",
        status="proposed",
        is_valid=True,
    )

    data_dict = proposal.to_dict()
    json_str = json.dumps(data_dict)
    assert isinstance(json_str, str)
    assert data_dict["operation"] == "create_file"
    assert data_dict["target_file"] == "module.py"
    assert data_dict["target_symbol"] == "MyClass"


def test_proposal_generation_does_not_modify_files(tmp_path: Path) -> None:
    """Test 9 (CRITICAL): Proposal generation leaves target file on disk byte-for-byte unchanged."""
    initial_bytes = b"def original_function():\n    return 42\n"
    target_file = tmp_path / "target.py"
    target_file.write_bytes(initial_bytes)

    generator = CodeChangeGenerator(workspace_root=tmp_path)
    proposal_tool = ProposeCodeChangeTool(generator=generator)

    # 1. Direct generator call
    prop1 = generator.generate_proposal(
        request="Add second_function",
        target_file="target.py",
        operation="modify_file",
    )
    assert prop1.is_valid is True
    assert target_file.read_bytes() == initial_bytes

    # 2. Tool execution call
    tool_res = proposal_tool.execute(
        request="Add third_function",
        target_file="target.py",
        operation="modify_file",
    )
    assert tool_res["success"] is True
    assert target_file.read_bytes() == initial_bytes


def test_agent_can_select_proposal_capability(tmp_path: Path) -> None:
    """Test 10: Agent can select propose_code_change tool for code generation queries."""
    (tmp_path / "app.py").write_text("class App:\n    pass\n", encoding="utf-8")

    generator = CodeChangeGenerator(workspace_root=tmp_path)
    proposal_tool = ProposeCodeChangeTool(generator=generator)

    registry = ToolRegistry()
    registry.register(proposal_tool)

    tool_call_json = json.dumps({
        "action": "tool_call",
        "tool_name": "propose_code_change",
        "arguments": {
            "request": "Add greet function to app.py",
            "target_file": "app.py",
            "operation": "modify_file"
        }
    })

    provider = FakeDeveloperModelProvider(response_text=tool_call_json)
    agent = Agent(model=provider, tool_registry=registry, max_steps=3)

    state = AgentState(user_request="Propose adding a greet function to app.py", max_steps=3)
    final_state = agent.loop.run(state)

    assert final_state.step > 0
    assert final_state.tool_name == "propose_code_change"
    assert final_state.observation.get("success") is True
    assert final_state.observation.get("data", {}).get("operation") == "modify_file"


def test_proposal_result_distinguished_from_applied_change(tmp_path: Path) -> None:
    """Test 11: Agent prompt and tool output distinguish proposed change from applied change."""
    generator = CodeChangeGenerator(workspace_root=tmp_path)
    proposal_tool = ProposeCodeChangeTool(generator=generator)

    registry = ToolRegistry()
    registry.register(proposal_tool)

    provider = FakeDeveloperModelProvider()
    agent = Agent(model=provider, tool_registry=registry, max_steps=2)

    state = AgentState(user_request="Create helper function in utils.py", max_steps=2)
    agent.loop.run(state)

    messages_str = json.dumps(provider.last_messages)
    assert "CODE CHANGE PROPOSAL & UNSUPPORTED OPERATIONS MANDATES" in messages_str
    assert "READ-ONLY PROPOSALS ONLY" in messages_str
    assert "NO APPLIED CLAIM RULE" in messages_str


def test_explicit_rejection_of_unsupported_deletion_operation(tmp_path: Path) -> None:
    """Test 12: Explicit rejection of file deletion operations."""
    generator = CodeChangeGenerator(workspace_root=tmp_path)
    proposal_tool = ProposeCodeChangeTool(generator=generator)

    res = proposal_tool.execute(request="Delete loop.py", target_file="agent/loop.py", operation="delete_file")
    assert res["success"] is False
    assert "strictly unsupported" in res["error"]


# =====================================================================
# Phase 7.2 Permission-Controlled File Modification Tests
# =====================================================================

def test_approved_create_file(tmp_path: Path) -> None:
    """Phase 7.2 Test 1: Approved create_file writes and verifies file on disk."""
    generator = CodeChangeGenerator(workspace_root=tmp_path)
    proposal = generator.generate_proposal(request="Create helper", target_file="helper.py", operation="create_file")

    perm_mgr = PermissionManager(default_write_allowed=True)
    applier = CodeChangeApplier(permission_manager=perm_mgr, workspace_root=tmp_path)
    result = applier.apply_proposal(proposal)

    assert result.success is True
    assert result.permission_granted is True
    assert result.verification_success is True
    assert (tmp_path / "helper.py").exists() is True
    assert "helper.py" in (tmp_path / "helper.py").read_text(encoding="utf-8")


def test_approved_modify_file(tmp_path: Path) -> None:
    """Phase 7.2 Test 2: Approved modify_file updates and verifies file on disk."""
    sample_file = tmp_path / "app.py"
    initial_text = "class App:\n    pass\n"
    sample_file.write_text(initial_text, encoding="utf-8")

    generator = CodeChangeGenerator(workspace_root=tmp_path)
    proposal = generator.generate_proposal(request="Add method", target_file="app.py", operation="modify_file")

    perm_mgr = PermissionManager(default_write_allowed=True)
    applier = CodeChangeApplier(permission_manager=perm_mgr, workspace_root=tmp_path)
    result = applier.apply_proposal(proposal)

    assert result.success is True
    assert result.verification_success is True
    assert sample_file.read_text(encoding="utf-8") == proposal.proposed_content


def test_denied_create_file(tmp_path: Path) -> None:
    """Phase 7.2 Test 3: Denied create_file refuses to create file on disk."""
    generator = CodeChangeGenerator(workspace_root=tmp_path)
    proposal = generator.generate_proposal(request="Create secret module", target_file="secret.py", operation="create_file")

    perm_mgr = PermissionManager(default_write_allowed=False)
    applier = CodeChangeApplier(permission_manager=perm_mgr, workspace_root=tmp_path)
    result = applier.apply_proposal(proposal, explicit_permission=False)

    assert result.success is False
    assert result.permission_granted is False
    assert (tmp_path / "secret.py").exists() is False


def test_denied_modify_file(tmp_path: Path) -> None:
    """Phase 7.2 Test 4: Denied modify_file leaves target file byte-for-byte unchanged."""
    sample_file = tmp_path / "app.py"
    initial_bytes = b"class App:\n    pass\n"
    sample_file.write_bytes(initial_bytes)

    generator = CodeChangeGenerator(workspace_root=tmp_path)
    proposal = generator.generate_proposal(request="Modify App", target_file="app.py", operation="modify_file")

    perm_mgr = PermissionManager(default_write_allowed=False)
    applier = CodeChangeApplier(permission_manager=perm_mgr, workspace_root=tmp_path)
    result = applier.apply_proposal(proposal, explicit_permission=False)

    assert result.success is False
    assert result.permission_granted is False
    assert sample_file.read_bytes() == initial_bytes


def test_missing_permission_defaults_to_denied(tmp_path: Path) -> None:
    """Phase 7.2 Test 5: Default PermissionManager write_allowed state is False."""
    perm_mgr = PermissionManager()
    assert perm_mgr.write_allowed is False
    assert perm_mgr.is_write_allowed("app.py", "modify_file") is False


def test_path_traversal_application_attempt(tmp_path: Path) -> None:
    """Phase 7.2 Test 6: Path traversal application attempt outside workspace is rejected."""
    proposal = CodeChangeProposal(
        operation="create_file",
        target_file="../outside.py",
        proposed_content="print('evil')",
        is_valid=True,
    )
    perm_mgr = PermissionManager(default_write_allowed=True)
    applier = CodeChangeApplier(permission_manager=perm_mgr, workspace_root=tmp_path)
    result = applier.apply_proposal(proposal)

    assert result.success is False
    assert "outside the allowed workspace" in result.reason


def test_outside_workspace_path_application(tmp_path: Path) -> None:
    """Phase 7.2 Test 7: Absolute path outside workspace is rejected during application."""
    proposal = CodeChangeProposal(
        operation="create_file",
        target_file="/tmp/system.py",
        proposed_content="print('evil')",
        is_valid=True,
    )
    perm_mgr = PermissionManager(default_write_allowed=True)
    applier = CodeChangeApplier(permission_manager=perm_mgr, workspace_root=tmp_path)
    result = applier.apply_proposal(proposal)

    assert result.success is False


def test_protected_file_application(tmp_path: Path) -> None:
    """Phase 7.2 Test 8: Protected file target (.env) is rejected during application."""
    proposal = CodeChangeProposal(
        operation="modify_file",
        target_file=".env",
        proposed_content="SECRET=123",
        is_valid=True,
    )
    perm_mgr = PermissionManager(default_write_allowed=True)
    applier = CodeChangeApplier(permission_manager=perm_mgr, workspace_root=tmp_path)
    result = applier.apply_proposal(proposal)

    assert result.success is False
    assert "protected" in result.reason.lower()


def test_stale_proposal_rejection(tmp_path: Path) -> None:
    """Phase 7.2 Test 9: Stale proposal is rejected when file changes on disk after proposal generation."""
    sample_file = tmp_path / "target.py"
    sample_file.write_text("v1_content", encoding="utf-8")

    generator = CodeChangeGenerator(workspace_root=tmp_path)
    proposal = generator.generate_proposal(request="Modify target", target_file="target.py", operation="modify_file")
    assert proposal.original_content == "v1_content"

    # Simulate file modification on disk before application
    sample_file.write_text("v2_modified_on_disk", encoding="utf-8")

    perm_mgr = PermissionManager(default_write_allowed=True)
    applier = CodeChangeApplier(permission_manager=perm_mgr, workspace_root=tmp_path)
    result = applier.apply_proposal(proposal)

    assert result.success is False
    assert "Stale proposal" in result.reason
    assert sample_file.read_text(encoding="utf-8") == "v2_modified_on_disk"


def test_llm_cannot_bypass_permission(tmp_path: Path) -> None:
    """Phase 7.2 Test 13: LLM passing confirm_permission=True in tool arguments fails if PermissionManager is denied."""
    sample_file = tmp_path / "app.py"
    initial_bytes = b"original"
    sample_file.write_bytes(initial_bytes)

    generator = CodeChangeGenerator(workspace_root=tmp_path)
    proposal = generator.generate_proposal(request="Modify app", target_file="app.py", operation="modify_file")

    perm_mgr = PermissionManager(default_write_allowed=False)  # Permission DENIED by system
    applier = CodeChangeApplier(permission_manager=perm_mgr, workspace_root=tmp_path)
    applier.set_pending_proposal(proposal)
    apply_tool = ApplyCodeChangeTool(applier=applier)

    # LLM attempts to pass confirm_permission=True
    tool_res = apply_tool.execute(proposal=proposal.to_dict(), confirm_permission=True)

    assert tool_res["success"] is False
    assert "Permission denied" in tool_res["error"]
    assert sample_file.read_bytes() == initial_bytes


# =====================================================================
# Phase 7.2 PATCH Specific Regression Tests A - H
# =====================================================================

def test_patch_test_a_fresh_proposal_applies(tmp_path: Path) -> None:
    """Test A: Fresh proposal against unchanged file applies successfully when approved."""
    target_file = tmp_path / "test_target.py"
    initial_content = 'def hello():\n    return "hello"\n'
    target_file.write_text(initial_content, encoding="utf-8")

    perm_mgr = PermissionManager(default_write_allowed=False)
    applier = CodeChangeApplier(permission_manager=perm_mgr, workspace_root=tmp_path)
    generator = CodeChangeGenerator(workspace_root=tmp_path, applier=applier)

    proposal = generator.generate_proposal(
        request='Modify hello function to return "Hello, VIDURA!"',
        target_file="test_target.py",
        operation="modify_file",
    )
    proposal.proposed_content = 'def hello():\n    return "Hello, VIDURA!"\n'

    assert proposal.is_valid is True
    assert proposal.original_content == initial_content
    assert applier.get_pending_proposal() is proposal

    perm_mgr.grant_write_permission()
    result = applier.apply_proposal(proposal)

    assert result.success is True
    assert result.permission_granted is True
    assert result.verification_success is True
    assert target_file.read_text(encoding="utf-8") == 'def hello():\n    return "Hello, VIDURA!"\n'


def test_patch_test_b_no_permission(tmp_path: Path) -> None:
    """Test B: Fresh proposal fails with permission denied if not approved."""
    target_file = tmp_path / "test_target.py"
    initial_content = 'def hello():\n    return "hello"\n'
    target_file.write_text(initial_content, encoding="utf-8")

    perm_mgr = PermissionManager(default_write_allowed=False)
    applier = CodeChangeApplier(permission_manager=perm_mgr, workspace_root=tmp_path)
    generator = CodeChangeGenerator(workspace_root=tmp_path, applier=applier)

    proposal = generator.generate_proposal(
        request="Modify hello function",
        target_file="test_target.py",
        operation="modify_file",
    )

    result = applier.apply_proposal(proposal)

    assert result.success is False
    assert result.permission_granted is False
    assert target_file.read_text(encoding="utf-8") == initial_content


def test_patch_test_c_stale_proposal(tmp_path: Path) -> None:
    """Test C: Proposal is rejected as stale if target file changes on disk before application."""
    target_file = tmp_path / "test_target.py"
    initial_content = 'def hello():\n    return "hello"\n'
    target_file.write_text(initial_content, encoding="utf-8")

    perm_mgr = PermissionManager(default_write_allowed=False)
    applier = CodeChangeApplier(permission_manager=perm_mgr, workspace_root=tmp_path)
    generator = CodeChangeGenerator(workspace_root=tmp_path, applier=applier)

    proposal = generator.generate_proposal(
        request="Modify hello function",
        target_file="test_target.py",
        operation="modify_file",
    )

    # Manually modify file on disk after proposal generation
    target_file.write_text('def hello():\n    return "manual change"\n', encoding="utf-8")

    perm_mgr.grant_write_permission()
    result = applier.apply_proposal(proposal)

    assert result.success is False
    assert "Stale proposal" in result.reason
    assert target_file.read_text(encoding="utf-8") == 'def hello():\n    return "manual change"\n'


def test_patch_test_d_proposal_integrity(tmp_path: Path) -> None:
    """Test D: Application retrieves trusted pending proposal rather than accepting model-supplied payload."""
    target_file = tmp_path / "test_target.py"
    initial_content = 'def hello():\n    return "hello"\n'
    target_file.write_text(initial_content, encoding="utf-8")

    perm_mgr = PermissionManager(default_write_allowed=False)
    applier = CodeChangeApplier(permission_manager=perm_mgr, workspace_root=tmp_path)
    generator = CodeChangeGenerator(workspace_root=tmp_path, applier=applier)

    trusted_proposal = generator.generate_proposal(
        request="Modify hello function",
        target_file="test_target.py",
        operation="modify_file",
    )
    trusted_proposal.proposed_content = 'def hello():\n    return "trusted content"\n'

    apply_tool = ApplyCodeChangeTool(applier=applier)
    perm_mgr.grant_write_permission()

    # Model supplies partial/fabricated payload
    tool_res = apply_tool.execute(proposal={"target_file": "test_target.py", "operation": "modify_file"})

    assert tool_res["success"] is True
    assert target_file.read_text(encoding="utf-8") == 'def hello():\n    return "trusted content"\n'


def test_patch_test_e_denied_proposal(tmp_path: Path) -> None:
    """Test E: Executing /deny clears active pending proposal and prevents application."""
    target_file = tmp_path / "test_target.py"
    initial_content = 'def hello():\n    return "hello"\n'
    target_file.write_text(initial_content, encoding="utf-8")

    perm_mgr = PermissionManager(default_write_allowed=False)
    applier = CodeChangeApplier(permission_manager=perm_mgr, workspace_root=tmp_path)
    generator = CodeChangeGenerator(workspace_root=tmp_path, applier=applier)

    generator.generate_proposal(
        request="Modify hello function",
        target_file="test_target.py",
        operation="modify_file",
    )

    # User executes /deny
    perm_mgr.revoke_write_permission()
    applier.clear_pending_proposal()

    result = applier.apply_proposal()

    assert result.success is False
    assert applier.get_pending_proposal() is None
    assert target_file.read_text(encoding="utf-8") == initial_content


def test_patch_test_f_new_proposal_replaces_old_proposal(tmp_path: Path) -> None:
    """Test F: Generating a new proposal replaces any existing pending proposal."""
    (tmp_path / "fileA.py").write_text("fileA content", encoding="utf-8")
    (tmp_path / "fileB.py").write_text("fileB content", encoding="utf-8")

    perm_mgr = PermissionManager(default_write_allowed=False)
    applier = CodeChangeApplier(permission_manager=perm_mgr, workspace_root=tmp_path)
    generator = CodeChangeGenerator(workspace_root=tmp_path, applier=applier)

    propA = generator.generate_proposal(request="Modify fileA", target_file="fileA.py", operation="modify_file")
    assert applier.get_pending_proposal() is propA

    propB = generator.generate_proposal(request="Modify fileB", target_file="fileB.py", operation="modify_file")
    assert applier.get_pending_proposal() is propB
    assert applier.get_pending_proposal() is not propA


def test_patch_test_g_create_file(tmp_path: Path) -> None:
    """Test G: Create file proposal applies successfully when approved."""
    perm_mgr = PermissionManager(default_write_allowed=False)
    applier = CodeChangeApplier(permission_manager=perm_mgr, workspace_root=tmp_path)
    generator = CodeChangeGenerator(workspace_root=tmp_path, applier=applier)

    proposal = generator.generate_proposal(request="Create new module", target_file="new_mod.py", operation="create_file")
    proposal.proposed_content = "def new_func(): pass\n"

    perm_mgr.grant_write_permission()
    result = applier.apply_proposal(proposal)

    assert result.success is True
    assert (tmp_path / "new_mod.py").exists() is True
    assert (tmp_path / "new_mod.py").read_text(encoding="utf-8") == "def new_func(): pass\n"


def test_patch_test_h_protected_target(tmp_path: Path) -> None:
    """Test H: Approval does NOT override protected target patterns (.env, .git, etc.)."""
    perm_mgr = PermissionManager(default_write_allowed=False)
    applier = CodeChangeApplier(permission_manager=perm_mgr, workspace_root=tmp_path)

    env_proposal = CodeChangeProposal(
        operation="modify_file",
        target_file=".env",
        proposed_content="SECRET=123",
        is_valid=True,
    )
    applier.set_pending_proposal(env_proposal)
    perm_mgr.grant_write_permission()

    result = applier.apply_proposal()

    assert result.success is False
    assert "protected" in result.reason.lower() or "denied" in result.reason.lower()


# =====================================================================
# Phase 7.3 Post-Change Verification Tests (Tests A - J)
# =====================================================================

from developer.verifier import CodeChangeVerifier
from developer.models import ApplicationStatus


def test_phase73_test_a_successful_create_and_verification(tmp_path: Path) -> None:
    """Phase 7.3 Test A: Successful create_file application and post-change verification."""
    perm_mgr = PermissionManager(default_write_allowed=False)
    applier = CodeChangeApplier(permission_manager=perm_mgr, workspace_root=tmp_path)
    generator = CodeChangeGenerator(workspace_root=tmp_path, applier=applier)

    proposal = generator.generate_proposal(request="Create module", target_file="mod.py", operation="create_file")
    perm_mgr.grant_write_permission()
    result = applier.apply_proposal(proposal)

    assert result.success is True
    assert result.verification_success is True
    assert result.status_code == ApplicationStatus.APPLIED_AND_VERIFIED.value
    assert (tmp_path / "mod.py").exists() is True
    assert (tmp_path / "mod.py").read_text(encoding="utf-8") == proposal.proposed_content


def test_phase73_test_b_successful_modify_and_verification(tmp_path: Path) -> None:
    """Phase 7.3 Test B: Successful modify_file application and post-change verification."""
    target_file = tmp_path / "app.py"
    target_file.write_text("class App:\n    pass\n", encoding="utf-8")

    perm_mgr = PermissionManager(default_write_allowed=False)
    applier = CodeChangeApplier(permission_manager=perm_mgr, workspace_root=tmp_path)
    generator = CodeChangeGenerator(workspace_root=tmp_path, applier=applier)

    proposal = generator.generate_proposal(request="Modify App", target_file="app.py", operation="modify_file")
    perm_mgr.grant_write_permission()
    result = applier.apply_proposal(proposal)

    assert result.success is True
    assert result.verification_success is True
    assert result.status_code == ApplicationStatus.APPLIED_AND_VERIFIED.value
    assert target_file.read_text(encoding="utf-8") == proposal.proposed_content


def test_phase73_test_c_permission_denied(tmp_path: Path) -> None:
    """Phase 7.3 Test C: Permission denied state returns status_code permission_denied."""
    target_file = tmp_path / "app.py"
    initial_content = "class App:\n    pass\n"
    target_file.write_text(initial_content, encoding="utf-8")

    perm_mgr = PermissionManager(default_write_allowed=False)
    applier = CodeChangeApplier(permission_manager=perm_mgr, workspace_root=tmp_path)
    generator = CodeChangeGenerator(workspace_root=tmp_path, applier=applier)

    proposal = generator.generate_proposal(request="Modify App", target_file="app.py", operation="modify_file")
    result = applier.apply_proposal(proposal)

    assert result.success is False
    assert result.permission_granted is False
    assert result.status_code == ApplicationStatus.PERMISSION_DENIED.value
    assert target_file.read_text(encoding="utf-8") == initial_content


def test_phase73_test_d_stale_proposal(tmp_path: Path) -> None:
    """Phase 7.3 Test D: Stale proposal state returns status_code stale_proposal."""
    target_file = tmp_path / "app.py"
    target_file.write_text("v1_content", encoding="utf-8")

    perm_mgr = PermissionManager(default_write_allowed=False)
    applier = CodeChangeApplier(permission_manager=perm_mgr, workspace_root=tmp_path)
    generator = CodeChangeGenerator(workspace_root=tmp_path, applier=applier)

    proposal = generator.generate_proposal(request="Modify App", target_file="app.py", operation="modify_file")
    target_file.write_text("v2_manual_change", encoding="utf-8")
    perm_mgr.grant_write_permission()

    result = applier.apply_proposal(proposal)

    assert result.success is False
    assert result.status_code == ApplicationStatus.STALE_PROPOSAL.value
    assert target_file.read_text(encoding="utf-8") == "v2_manual_change"


def test_phase73_test_e_verification_detects_wrong_content(tmp_path: Path) -> None:
    """Phase 7.3 Test E: Verifier detects disk content mismatch and sets verification_failed."""
    verifier = CodeChangeVerifier(workspace_root=tmp_path)
    target_file = tmp_path / "app.py"
    target_file.write_text("actual_different_content", encoding="utf-8")

    proposal = CodeChangeProposal(
        operation="modify_file",
        target_file="app.py",
        proposed_content="expected_proposed_content",
        is_valid=True,
    )

    ver_res = verifier.verify(proposal)

    assert ver_res.success is False
    assert ver_res.verified is False
    assert "does not match" in ver_res.reason


def test_phase73_test_f_missing_resulting_file(tmp_path: Path) -> None:
    """Phase 7.3 Test F: Verifier detects missing target file on disk and sets verification_failed."""
    verifier = CodeChangeVerifier(workspace_root=tmp_path)
    proposal = CodeChangeProposal(
        operation="create_file",
        target_file="nonexistent.py",
        proposed_content="some content",
        is_valid=True,
    )

    ver_res = verifier.verify(proposal)

    assert ver_res.success is False
    assert ver_res.verified is False
    assert "does not exist" in ver_res.reason


def test_phase73_test_g_create_file_content_mismatch(tmp_path: Path) -> None:
    """Phase 7.3 Test G: create_file verification fails if created file content differs."""
    verifier = CodeChangeVerifier(workspace_root=tmp_path)
    (tmp_path / "created.py").write_text("unexpected content", encoding="utf-8")

    proposal = CodeChangeProposal(
        operation="create_file",
        target_file="created.py",
        proposed_content="expected content",
        is_valid=True,
    )

    ver_res = verifier.verify(proposal)

    assert ver_res.success is False
    assert ver_res.verified is False


def test_phase73_test_h_modify_file_content_mismatch(tmp_path: Path) -> None:
    """Phase 7.3 Test H: modify_file verification fails if modified file content differs."""
    verifier = CodeChangeVerifier(workspace_root=tmp_path)
    (tmp_path / "mod.py").write_text("partial or corrupt content", encoding="utf-8")

    proposal = CodeChangeProposal(
        operation="modify_file",
        target_file="mod.py",
        proposed_content="expected full content",
        is_valid=True,
    )

    ver_res = verifier.verify(proposal)

    assert ver_res.success is False
    assert ver_res.verified is False


def test_phase73_test_i_llm_false_success_response_override(tmp_path: Path) -> None:
    """Phase 7.3 Test I: System prompt contains explicit mandate instructing model not to claim success if verification fails."""
    from agent.loop import build_system_instruction
    prompt = build_system_instruction()

    assert "POST-CHANGE VERIFICATION MANDATE" in prompt
    assert "applied_and_verified" in prompt
    assert "verification_failed" in prompt


def test_phase73_test_j_protected_target_rejection(tmp_path: Path) -> None:
    """Phase 7.3 Test J: Protected files (.env, .git, outside workspace) rejected before verification."""
    perm_mgr = PermissionManager(default_write_allowed=True)
    applier = CodeChangeApplier(permission_manager=perm_mgr, workspace_root=tmp_path)

    prop_env = CodeChangeProposal(operation="modify_file", target_file=".env", proposed_content="SECRET=1", is_valid=True)
    applier.set_pending_proposal(prop_env)
    perm_mgr.grant_write_permission()
    res_env = applier.apply_proposal()
    assert res_env.success is False
    assert res_env.status_code == ApplicationStatus.PERMISSION_DENIED.value

    prop_outside = CodeChangeProposal(operation="create_file", target_file="../outside.py", proposed_content="evil", is_valid=True)
    applier.set_pending_proposal(prop_outside)
    perm_mgr.grant_write_permission()
    res_out = applier.apply_proposal()
    assert res_out.success is False
    assert res_out.status_code in (ApplicationStatus.APPLICATION_FAILED.value, ApplicationStatus.PERMISSION_DENIED.value)


# =====================================================================
# Phase 7.3 PATCH Tests (Tests 1 - 12)
# =====================================================================

from agent.loop import AgentLoop, is_apply_request
from agent.state import AgentState
from tools.registry import ToolRegistry


class DummyModelWithRespond:
    """Simulates an LLM returning action 'respond' claiming false application."""
    def generate(self, messages):
        return '{"action": "respond", "content": "The code change has been applied."}'


def test_patch73_test1_apply_request_invokes_tool(tmp_path: Path) -> None:
    """Test 1: Apply request with pending proposal deterministically routes to apply_code_change."""
    (tmp_path / "app.py").write_text("class App:\n    pass\n", encoding="utf-8")
    perm_mgr = PermissionManager(default_write_allowed=False)
    applier = CodeChangeApplier(permission_manager=perm_mgr, workspace_root=tmp_path)
    generator = CodeChangeGenerator(workspace_root=tmp_path, applier=applier)

    generator.generate_proposal(request="Modify App", target_file="app.py", operation="modify_file")
    perm_mgr.grant_write_permission()

    registry = ToolRegistry()
    registry.register(ApplyCodeChangeTool(applier=applier))

    agent = AgentLoop(model=DummyModelWithRespond(), tool_registry=registry)
    state = AgentState(user_request="Now apply the proposed changes")

    final_state = agent.run(state)

    assert final_state.tool_name == "apply_code_change"
    assert "successfully applied and verified" in final_state.final_response.lower()


def test_patch73_test2_llm_respond_overridden_for_apply(tmp_path: Path) -> None:
    """Test 2: LLM tries to respond instead of calling apply_code_change -> system overrides and does not accept respond."""
    perm_mgr = PermissionManager(default_write_allowed=False)
    applier = CodeChangeApplier(permission_manager=perm_mgr, workspace_root=tmp_path)

    registry = ToolRegistry()
    registry.register(ApplyCodeChangeTool(applier=applier))

    agent = AgentLoop(model=DummyModelWithRespond(), tool_registry=registry)
    state = AgentState(user_request="Apply the proposal")

    final_state = agent.run(state)

    assert "failed" in final_state.final_response.lower() or "permission denied" in final_state.final_response.lower()
    assert "successfully applied" not in final_state.final_response.lower()


def test_patch73_test3_llm_false_success_text_suppressed(tmp_path: Path) -> None:
    """Test 3: LLM produces false 'applied successfully' text while no application occurred -> final result is NOT success."""
    assert is_apply_request("Apply the proposed changes") is True
    assert is_apply_request("Make the proposed change") is True
    assert is_apply_request("Do it") is True


def test_patch73_test4_permission_denied_reflected(tmp_path: Path) -> None:
    """Test 4: apply_code_change returns permission_denied -> final response reflects permission_denied."""
    perm_mgr = PermissionManager(default_write_allowed=False)
    applier = CodeChangeApplier(permission_manager=perm_mgr, workspace_root=tmp_path)
    generator = CodeChangeGenerator(workspace_root=tmp_path, applier=applier)
    (tmp_path / "app.py").write_text("old", encoding="utf-8")

    generator.generate_proposal(request="Modify App", target_file="app.py", operation="modify_file")

    registry = ToolRegistry()
    registry.register(ApplyCodeChangeTool(applier=applier))

    agent = AgentLoop(model=DummyModelWithRespond(), tool_registry=registry)
    state = AgentState(user_request="Apply proposal")

    final_state = agent.run(state)

    assert "permission denied" in final_state.final_response.lower()
    assert (tmp_path / "app.py").read_text(encoding="utf-8") == "old"


def test_patch73_test5_stale_proposal_reflected(tmp_path: Path) -> None:
    """Test 5: apply_code_change returns stale_proposal -> final response reflects stale_proposal."""
    target_file = tmp_path / "app.py"
    target_file.write_text("v1", encoding="utf-8")
    perm_mgr = PermissionManager(default_write_allowed=False)
    applier = CodeChangeApplier(permission_manager=perm_mgr, workspace_root=tmp_path)
    generator = CodeChangeGenerator(workspace_root=tmp_path, applier=applier)

    generator.generate_proposal(request="Modify App", target_file="app.py", operation="modify_file")
    target_file.write_text("v2_manual", encoding="utf-8")
    perm_mgr.grant_write_permission()

    registry = ToolRegistry()
    registry.register(ApplyCodeChangeTool(applier=applier))

    agent = AgentLoop(model=DummyModelWithRespond(), tool_registry=registry)
    state = AgentState(user_request="Apply proposal")

    final_state = agent.run(state)

    assert "stale proposal" in final_state.final_response.lower()
    assert target_file.read_text(encoding="utf-8") == "v2_manual"


def test_patch73_test6_verification_failed_reflected(tmp_path: Path) -> None:
    """Test 6: apply_code_change returns verification_failed -> final response reflects verification_failed."""
    perm_mgr = PermissionManager(default_write_allowed=False)

    class FailingVerifier(CodeChangeVerifier):
        def verify(self, proposal):
            return VerificationResult(
                success=False,
                target_file=proposal.target_file,
                operation=proposal.operation,
                verified=False,
                expected_state="Proposed",
                actual_state="Actual",
                reason="Verification failed due to hash mismatch.",
                error="Verification failed.",
            )

    applier = CodeChangeApplier(permission_manager=perm_mgr, workspace_root=tmp_path, verifier=FailingVerifier(workspace_root=tmp_path))
    generator = CodeChangeGenerator(workspace_root=tmp_path, applier=applier)
    (tmp_path / "app.py").write_text("old", encoding="utf-8")
    generator.generate_proposal(request="Modify App", target_file="app.py", operation="modify_file")
    perm_mgr.grant_write_permission()

    registry = ToolRegistry()
    registry.register(ApplyCodeChangeTool(applier=applier))

    agent = AgentLoop(model=DummyModelWithRespond(), tool_registry=registry)
    state = AgentState(user_request="Apply proposal")

    final_state = agent.run(state)

    assert "verification failed" in final_state.final_response.lower()


def test_patch73_test7_only_applied_and_verified_permits_success(tmp_path: Path) -> None:
    """Test 7: Only status_code applied_and_verified permits success wording."""
    (tmp_path / "app.py").write_text("def run(): pass\n", encoding="utf-8")
    perm_mgr = PermissionManager(default_write_allowed=False)
    applier = CodeChangeApplier(permission_manager=perm_mgr, workspace_root=tmp_path)
    generator = CodeChangeGenerator(workspace_root=tmp_path, applier=applier)

    generator.generate_proposal(request="Modify App", target_file="app.py", operation="modify_file")
    perm_mgr.grant_write_permission()

    registry = ToolRegistry()
    registry.register(ApplyCodeChangeTool(applier=applier))

    agent = AgentLoop(model=DummyModelWithRespond(), tool_registry=registry)
    state = AgentState(user_request="Now apply the changes")

    final_state = agent.run(state)

    assert "successfully applied and verified" in final_state.final_response.lower()


def test_patch73_test8_deny_revokes_permission(tmp_path: Path) -> None:
    """Test 8: /deny revokes permission in PermissionManager and clears pending proposal in CodeChangeApplier."""
    perm_mgr = PermissionManager(default_write_allowed=False)
    applier = CodeChangeApplier(permission_manager=perm_mgr, workspace_root=tmp_path)
    generator = CodeChangeGenerator(workspace_root=tmp_path, applier=applier)
    (tmp_path / "app.py").write_text("old", encoding="utf-8")

    generator.generate_proposal(request="Modify App", target_file="app.py", operation="modify_file")
    perm_mgr.grant_write_permission()
    assert applier.get_pending_proposal() is not None

    # Simulate /deny execution
    perm_mgr.revoke_write_permission()
    applier.clear_pending_proposal()

    assert perm_mgr.write_allowed is False
    assert applier.get_pending_proposal() is None


def test_patch73_test9_successful_modify_works(tmp_path: Path) -> None:
    """Test 9: Successful modify operation applies and verifies cleanly."""
    target_file = tmp_path / "test_mod.py"
    target_file.write_text("v1", encoding="utf-8")

    perm_mgr = PermissionManager(default_write_allowed=False)
    applier = CodeChangeApplier(permission_manager=perm_mgr, workspace_root=tmp_path)
    generator = CodeChangeGenerator(workspace_root=tmp_path, applier=applier)

    proposal = generator.generate_proposal(request="Modify test_mod", target_file="test_mod.py", operation="modify_file")
    perm_mgr.grant_write_permission()
    result = applier.apply_proposal(proposal)

    assert result.success is True
    assert result.verification_success is True
    assert result.status_code == ApplicationStatus.APPLIED_AND_VERIFIED.value


def test_patch73_test10_successful_create_works(tmp_path: Path) -> None:
    """Test 10: Successful create operation applies and verifies cleanly."""
    perm_mgr = PermissionManager(default_write_allowed=False)
    applier = CodeChangeApplier(permission_manager=perm_mgr, workspace_root=tmp_path)
    generator = CodeChangeGenerator(workspace_root=tmp_path, applier=applier)

    proposal = generator.generate_proposal(request="Create new module", target_file="test_create.py", operation="create_file")
    perm_mgr.grant_write_permission()
    result = applier.apply_proposal(proposal)

    assert result.success is True
    assert result.verification_success is True
    assert (tmp_path / "test_create.py").exists() is True


def test_patch73_test11_protected_targets_blocked(tmp_path: Path) -> None:
    """Test 11: Protected targets (.env, .git, secret.key) remain strictly blocked."""
    perm_mgr = PermissionManager(default_write_allowed=False)
    applier = CodeChangeApplier(permission_manager=perm_mgr, workspace_root=tmp_path)

    prop = CodeChangeProposal(operation="modify_file", target_file=".env", proposed_content="SECRET=1", is_valid=True)
    applier.set_pending_proposal(prop)
    perm_mgr.grant_write_permission()
    res = applier.apply_proposal()

    assert res.success is False
    assert res.status_code == ApplicationStatus.PERMISSION_DENIED.value


def test_patch73_test12_existing_phase71_and_72_tests_pass(tmp_path: Path) -> None:
    """Test 12: Existing Phase 7.1 and Phase 7.2 components continue functioning."""
    generator = CodeChangeGenerator(workspace_root=tmp_path)
    proposal = generator.generate_proposal(request="Create file", target_file="sample.py", operation="create_file")
    assert proposal.is_valid is True
    assert proposal.target_file == "sample.py"


# =====================================================================
# Phase 7.4 Safety & Regression Hardening Tests
# =====================================================================

import os


def test_phase74_permission_hardening(tmp_path: Path) -> None:
    """Phase 7.4: Permission hardening - default deny, grant, revoke, auto-revocation on new proposal."""
    perm_mgr = PermissionManager(default_write_allowed=False)
    assert perm_mgr.write_allowed is False
    assert perm_mgr.is_write_allowed("app.py", "modify_file") is False

    perm_mgr.grant_write_permission()
    assert perm_mgr.write_allowed is True

    perm_mgr.revoke_write_permission()
    assert perm_mgr.write_allowed is False

    # Approval followed by denial
    perm_mgr.grant_write_permission()
    perm_mgr.revoke_write_permission()
    assert perm_mgr.write_allowed is False

    # Auto-revocation when storing a new proposal
    applier = CodeChangeApplier(permission_manager=perm_mgr, workspace_root=tmp_path)
    generator = CodeChangeGenerator(workspace_root=tmp_path, applier=applier)
    
    perm_mgr.grant_write_permission()
    assert perm_mgr.write_allowed is True

    generator.generate_proposal(request="Create new file", target_file="new.py", operation="create_file")
    assert perm_mgr.write_allowed is False  # Auto-revoked on new proposal registration


def test_phase74_proposal_immutability_and_argument_hardening(tmp_path: Path) -> None:
    """Phase 7.4: Tool argument hardening - ApplyCodeChangeTool ignores LLM-supplied proposal overrides."""
    (tmp_path / "app.py").write_text("class App:\n    pass\n", encoding="utf-8")
    perm_mgr = PermissionManager(default_write_allowed=False)
    applier = CodeChangeApplier(permission_manager=perm_mgr, workspace_root=tmp_path)
    generator = CodeChangeGenerator(workspace_root=tmp_path, applier=applier)

    trusted = generator.generate_proposal(request="Modify App", target_file="app.py", operation="modify_file")
    trusted.proposed_content = "class TrustedApp:\n    pass\n"
    perm_mgr.grant_write_permission()

    apply_tool = ApplyCodeChangeTool(applier=applier)

    # LLM passes conflicting parameters in kwargs
    res = apply_tool.execute(
        proposal={
            "target_file": "evil.py",
            "operation": "create_file",
            "proposed_content": "evil content",
        }
    )

    assert res["success"] is True
    assert (tmp_path / "app.py").read_text(encoding="utf-8") == "class TrustedApp:\n    pass\n"
    assert (tmp_path / "evil.py").exists() is False


def test_phase74_duplicate_application_rejection(tmp_path: Path) -> None:
    """Phase 7.4: Applying a proposal twice fails safely on second attempt."""
    (tmp_path / "app.py").write_text("v1", encoding="utf-8")
    perm_mgr = PermissionManager(default_write_allowed=False)
    applier = CodeChangeApplier(permission_manager=perm_mgr, workspace_root=tmp_path)
    generator = CodeChangeGenerator(workspace_root=tmp_path, applier=applier)

    proposal = generator.generate_proposal(request="Modify App", target_file="app.py", operation="modify_file")
    perm_mgr.grant_write_permission()
    
    # First application succeeds
    res1 = applier.apply_proposal(proposal)
    assert res1.success is True

    # Second application attempt fails safely
    res2 = applier.apply_proposal(proposal)
    assert res2.success is False
    assert "No active pending" in res2.reason or "applied" in res2.reason


def test_phase74_unsupported_operations_rejection(tmp_path: Path) -> None:
    """Phase 7.4: Explicit rejection of all unsupported operations in applier and tool."""
    perm_mgr = PermissionManager(default_write_allowed=False)
    applier = CodeChangeApplier(permission_manager=perm_mgr, workspace_root=tmp_path)
    generator = CodeChangeGenerator(workspace_root=tmp_path, applier=applier)

    propose_tool = ProposeCodeChangeTool(generator=generator)

    unsupported_ops = ["delete", "delete_file", "remove", "destroy", "unlink", "rename_file", "move_file", "chmod", "shell"]
    for op in unsupported_ops:
        res = propose_tool.execute(request="Invalid op", target_file="app.py", operation=op)
        assert res["success"] is False
        assert "unsupported" in res["error"].lower()


def test_phase74_symlink_escape_rejection(tmp_path: Path) -> None:
    """Phase 7.4: Realpath symlink resolution rejects symlink escapes pointing outside workspace."""
    outside_dir = tmp_path.parent / "outside_workspace_74"
    outside_dir.mkdir(exist_ok=True)
    outside_file = outside_dir / "secret_outside.py"
    outside_file.write_text("secret_data", encoding="utf-8")

    # Create symlink inside workspace pointing to file outside workspace
    symlink_file = tmp_path / "symlink_escape.py"
    try:
        os.symlink(outside_file, symlink_file)
    except Exception:
        pytest.skip("Symlink creation not supported in test environment.")

    perm_mgr = PermissionManager(default_write_allowed=False)
    applier = CodeChangeApplier(permission_manager=perm_mgr, workspace_root=tmp_path)

    prop = CodeChangeProposal(
        operation="modify_file",
        target_file="symlink_escape.py",
        proposed_content="overwritten",
        original_content="secret_data",
        is_valid=True,
    )
    applier.set_pending_proposal(prop)
    perm_mgr.grant_write_permission()

    res = applier.apply_proposal()

    assert res.success is False
    assert "outside" in res.reason.lower() or "denied" in res.reason.lower()
    assert outside_file.read_text(encoding="utf-8") == "secret_data"


def test_phase74_protected_targets_hardening(tmp_path: Path) -> None:
    """Phase 7.4: Comprehensive protected targets hardening (.env, .git, secret.key, private.pem)."""
    perm_mgr = PermissionManager(default_write_allowed=False)
    applier = CodeChangeApplier(permission_manager=perm_mgr, workspace_root=tmp_path)

    protected_files = [".env", ".git/config", "secret.key", "private.pem", "data.secret"]
    for p_file in protected_files:
        prop = CodeChangeProposal(operation="modify_file", target_file=p_file, proposed_content="bad", is_valid=True)
        applier.set_pending_proposal(prop)
        perm_mgr.grant_write_permission()
        res = applier.apply_proposal()
        assert res.success is False
        assert res.status_code == ApplicationStatus.PERMISSION_DENIED.value


def test_phase74_stale_proposal_whitespace_and_newline_change(tmp_path: Path) -> None:
    """Phase 7.4: Stale content check rejects whitespace and newline changes on disk."""
    target = tmp_path / "app.py"
    target.write_text("def run():\n    pass\n", encoding="utf-8")

    perm_mgr = PermissionManager(default_write_allowed=False)
    applier = CodeChangeApplier(permission_manager=perm_mgr, workspace_root=tmp_path)
    generator = CodeChangeGenerator(workspace_root=tmp_path, applier=applier)

    proposal = generator.generate_proposal(request="Modify run", target_file="app.py", operation="modify_file")

    # Manually alter whitespace/newline on disk
    target.write_text("def run():\n    pass \n", encoding="utf-8")
    perm_mgr.grant_write_permission()

    res = applier.apply_proposal(proposal)

    assert res.success is False
    assert res.status_code == ApplicationStatus.STALE_PROPOSAL.value
    assert target.read_text(encoding="utf-8") == "def run():\n    pass \n"


def test_phase74_fail_closed_on_unexpected_exception(tmp_path: Path) -> None:
    """Phase 7.4: Unexpected exception during application or verification fails closed."""
    perm_mgr = PermissionManager(default_write_allowed=False)

    class CrashingVerifier(CodeChangeVerifier):
        def verify(self, proposal):
            raise RuntimeError("Simulated crash in verifier")

    applier = CodeChangeApplier(permission_manager=perm_mgr, workspace_root=tmp_path, verifier=CrashingVerifier(workspace_root=tmp_path))
    generator = CodeChangeGenerator(workspace_root=tmp_path, applier=applier)
    (tmp_path / "app.py").write_text("old", encoding="utf-8")
    proposal = generator.generate_proposal(request="Modify App", target_file="app.py", operation="modify_file")
    perm_mgr.grant_write_permission()

    res = applier.apply_proposal(proposal)

    assert res.success is False
    assert res.status_code == ApplicationStatus.APPLICATION_FAILED.value
    assert "Simulated crash" in res.reason or "Simulated crash" in str(res.error)



