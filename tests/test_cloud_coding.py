"""Automated tests for Phase 9.3: Cloud Coding Model Integration.

Tests cover Requirements A through S:
- Requirement A: Developer workflow uses cloud model when configured (gemma4:31b-cloud)
- Requirement B: General conversation uses local model by default (gemma4:e4b-it-qat)
- Requirement C: Dual-model routing works (general -> local, developer -> cloud)
- Requirement D: Developer task understanding remains stable
- Requirement E: Developer planning works with cloud model
- Requirement F: Developer code generation works with cloud model
- Requirement G: CodeChangeProposal contains provider="cloud", model="gemma4:31b-cloud"
- Requirement H: Cloud model cannot bypass write permissions
- Requirement I: Cloud model cannot modify filesystem directly
- Requirement J: Cloud model cannot bypass post-change verification
- Requirement K: Cloud model cannot bypass test execution
- Requirement L: Cloud model cannot falsely claim success without disk confirmation
- Requirement M: Cloud failure does not produce invalid files (filesystem untouched)
- Requirement N: Cloud failure surfaces clearly (ProviderUnavailable, etc.)
- Requirement O: Cloud failure does NOT silently fall back to local model
- Requirement P: Cloud model cannot access files outside workspace
- Requirement Q: Cloud model cannot access protected files (.git, .env, keys)
- Requirement R: Cloud provider credentials never leaked in error messages or repr
- Requirement S: Developer execution summary clearly identifies provider and model
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
import pytest

from config import Config
from models.base import (
    ProviderCapabilities,
    ModelProvider,
)
from models.errors import (
    ModelProviderError,
    ProviderAuthenticationError,
    ProviderUnavailable,
)
from models.cloud import OllamaCloudProvider
from models.local import LocalProvider
from models.router import ModelRouter
from developer.models import (
    CodeChangeProposal,
    ProposalOperation,
    ProposalStatus,
    ExecutionStatus,
    ExecutionStage,
    DeveloperGenerationResult,
    DeveloperExecutionResult,
)
from developer.generator import DeveloperCodeGenerator
from developer.executor import DeveloperExecutor
from developer.applier import CodeChangeApplier
from permissions.manager import PermissionManager
from task_understanding.models import DeveloperTask
from planning.models import DeveloperPlan


class MockProvider(ModelProvider):
    """Configurable mock provider tracking calls and returning canned responses."""

    def __init__(
        self,
        provider_type: str,
        model_name: str,
        is_cloud: bool = False,
        generate_response: str = "Mock response",
        fail_with: Exception | None = None,
    ) -> None:
        self.provider_type = provider_type
        self._model_name = model_name
        self._capabilities = ProviderCapabilities(
            coding=True,
            cloud=is_cloud,
            local=not is_cloud,
        )
        self.generate_response = generate_response
        self.fail_with = fail_with
        self.calls: list[dict[str, Any]] = []

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def provider_name(self) -> str:
        return f"Mock ({self.provider_type})"

    @property
    def capabilities(self) -> ProviderCapabilities:
        return self._capabilities

    def generate(self, messages: list[dict[str, str]], **kwargs: Any) -> str:
        self.calls.append({"messages": messages, "kwargs": kwargs})
        if self.fail_with:
            raise self.fail_with
        return self.generate_response

    def chat(self, messages: list[dict[str, str]], **kwargs: Any) -> str:
        return self.generate(messages, **kwargs)


# ============================================================================
# Requirement A, B, C: Routing & Dual-Model Configuration
# ============================================================================

def test_requirement_a_developer_workflow_uses_cloud_when_configured() -> None:
    """Requirement A: Developer workflow uses cloud model when configured."""
    local_mock = MockProvider("local", "gemma4:e4b-it-qat", is_cloud=False)
    cloud_mock = MockProvider("cloud", "gemma4:31b-cloud", is_cloud=True)

    router = ModelRouter(
        provider_type="local",
        developer_provider_type="cloud",
        local_provider=local_mock,
        cloud_provider=cloud_mock,
    )

    assert router.developer_provider_type == "cloud"
    assert router.developer_model_name == "gemma4:31b-cloud"

    # Route developer task to cloud
    res = router.generate("Generate a new class", task_type="developer")
    assert len(cloud_mock.calls) == 1
    assert len(local_mock.calls) == 0


def test_requirement_b_general_conversation_uses_local_by_default() -> None:
    """Requirement B: General conversation uses local model by default."""
    local_mock = MockProvider("local", "gemma4:e4b-it-qat", is_cloud=False)
    cloud_mock = MockProvider("cloud", "gemma4:31b-cloud", is_cloud=True)

    router = ModelRouter(
        provider_type="local",
        developer_provider_type="cloud",
        local_provider=local_mock,
        cloud_provider=cloud_mock,
    )

    # General conversation
    res = router.generate("Hello, how are you?", task_type="general")
    assert len(local_mock.calls) == 1
    assert len(cloud_mock.calls) == 0


def test_requirement_c_dual_model_routing_works() -> None:
    """Requirement C: Dual-model routing routes general to local and developer to cloud."""
    local_mock = MockProvider("local", "gemma4:e4b-it-qat", is_cloud=False, generate_response="Local general answer")
    cloud_mock = MockProvider("cloud", "gemma4:31b-cloud", is_cloud=True, generate_response="Cloud code answer")

    router = ModelRouter(
        provider_type="local",
        developer_provider_type="cloud",
        local_provider=local_mock,
        cloud_provider=cloud_mock,
    )

    # 1. General chat -> Local
    gen_res = router.chat([{"role": "user", "content": "Explain binary search"}])
    assert gen_res == "Local general answer"
    assert len(local_mock.calls) == 1
    assert len(cloud_mock.calls) == 0

    # 2. Developer task -> Cloud
    dev_res = router.chat([{"role": "user", "content": "Write binary search in Python"}], is_developer_task=True)
    assert dev_res == "Cloud code answer"
    assert len(local_mock.calls) == 1
    assert len(cloud_mock.calls) == 1


# ============================================================================
# Requirement D, E, F, G: Task Understanding, Planning & Generation Metadata
# ============================================================================

def test_requirement_d_developer_task_understanding_stable() -> None:
    """Requirement D: DeveloperTask produced correctly regardless of provider."""
    task = DeveloperTask(
        goal="Add logger helper function",
        target_files=["utils/logger.py"],
        target_symbols=["get_logger"],
        constraints=["Preserve existing loggers"],
    )
    assert task.target_files == ["utils/logger.py"]
    assert task.target_symbols == ["get_logger"]


def test_requirement_e_and_f_code_generation_with_cloud_model(tmp_path: Path) -> None:
    """Requirements E & F: Developer planning and code generation work with cloud model."""
    target_file = tmp_path / "math_utils.py"
    target_file.write_text("def add(a, b):\n    return a + b\n")

    cloud_json = json.dumps({
        "generated_code": "def add(a, b):\n    return a + b\n\ndef multiply(a, b):\n    return a * b\n",
        "explanation": "Added multiply function.",
        "affected_symbols": ["multiply"],
        "assumptions": ["Numbers only"],
    })
    cloud_mock = MockProvider("cloud", "gemma4:31b-cloud", is_cloud=True, generate_response=cloud_json)

    generator = DeveloperCodeGenerator(workspace_root=tmp_path, model=cloud_mock)

    task = DeveloperTask(
        goal="Add multiply function",
        target_files=["math_utils.py"],
        target_symbols=["multiply"],
    )
    plan = DeveloperPlan(
        original_task=task,
        relevant_files=["math_utils.py"],
        relevant_symbols=["multiply"],
        planned_changes=["Add multiply function"],
    )

    result = generator.generate_from_plan(task=task, plan=plan)

    assert result.is_valid is True
    assert result.validation_status == "valid"
    assert "def multiply(a, b):" in result.generated_code
    assert result.provider == "cloud"
    assert result.model == "gemma4:31b-cloud"


def test_requirement_g_code_change_proposal_contains_provider_and_model(tmp_path: Path) -> None:
    """Requirement G: CodeChangeProposal contains provider='cloud' and model='gemma4:31b-cloud'."""
    cloud_json = json.dumps({
        "proposed_content": "# New service file\nclass Service:\n    pass\n",
        "rationale": "Created Service class",
    })
    cloud_mock = MockProvider("cloud", "gemma4:31b-cloud", is_cloud=True, generate_response=cloud_json)

    generator = DeveloperCodeGenerator(workspace_root=tmp_path, model=cloud_mock)
    proposal = generator.generate_proposal(
        request="Create service.py",
        target_file="service.py",
        operation=ProposalOperation.CREATE_FILE.value,
    )

    assert proposal.is_valid is True
    assert proposal.provider == "cloud"
    assert proposal.model == "gemma4:31b-cloud"

    # Verify serialization preservation
    prop_dict = proposal.to_dict()
    assert prop_dict["provider"] == "cloud"
    assert prop_dict["model"] == "gemma4:31b-cloud"

    restored = CodeChangeProposal.from_dict(prop_dict)
    assert restored.provider == "cloud"
    assert restored.model == "gemma4:31b-cloud"


# ============================================================================
# Requirement H, I, J, K, L: Architectural Boundary & Security Invariants
# ============================================================================

def test_requirement_h_cloud_model_cannot_bypass_write_permissions(tmp_path: Path) -> None:
    """Requirement H: Cloud model cannot bypass write permissions."""
    test_file = tmp_path / "protected_code.py"
    test_file.write_text("initial = 1\n")

    cloud_mock = MockProvider("cloud", "gemma4:31b-cloud", is_cloud=True)
    perm_mgr = PermissionManager()
    applier = CodeChangeApplier(workspace_root=tmp_path, permission_manager=perm_mgr)
    executor = DeveloperExecutor(applier=applier, permission_manager=perm_mgr, workspace_root=tmp_path)

    proposal = CodeChangeProposal(
        operation=ProposalOperation.MODIFY_FILE.value,
        target_file="protected_code.py",
        proposed_content="initial = 2\n",
        original_content="initial = 1\n",
        provider="cloud",
        model="gemma4:31b-cloud",
        status=ProposalStatus.PROPOSED.value,
        is_valid=True,
    )

    executor.prepare_proposal(proposal)

    # Attempt execution with permission denied
    result = executor.execute_proposal(proposal=proposal, explicit_permission=False)

    assert result.success is False
    assert result.status == ExecutionStatus.PERMISSION_DENIED.value
    assert result.stage == ExecutionStage.DENIED.value
    assert test_file.read_text() == "initial = 1\n"  # Untouched on disk


def test_requirement_i_cloud_model_cannot_modify_filesystem_directly(tmp_path: Path) -> None:
    """Requirement I: Proposal generation alone does not touch the filesystem."""
    cloud_json = json.dumps({
        "generated_code": "print('hello')",
        "explanation": "created script",
        "affected_symbols": [],
        "assumptions": [],
    })
    cloud_mock = MockProvider("cloud", "gemma4:31b-cloud", is_cloud=True, generate_response=cloud_json)
    generator = DeveloperCodeGenerator(workspace_root=tmp_path, model=cloud_mock)

    task = DeveloperTask(
        goal="create file script.py",
        target_files=["script.py"],
    )
    plan = DeveloperPlan(
        original_task=task,
        relevant_files=["script.py"],
        planned_changes=["create file script.py"],
    )

    gen_res = generator.generate_from_plan(task=task, plan=plan)
    assert gen_res.is_valid is True

    # Target file must NOT exist yet
    target_path = tmp_path / "script.py"
    assert not target_path.exists()


def test_requirement_j_cloud_model_cannot_bypass_post_change_verification(tmp_path: Path) -> None:
    """Requirement J: Verification is strictly enforced post-application."""
    test_file = tmp_path / "verify_me.py"
    test_file.write_text("x = 1\n")

    perm_mgr = PermissionManager()
    applier = CodeChangeApplier(workspace_root=tmp_path, permission_manager=perm_mgr)
    executor = DeveloperExecutor(applier=applier, permission_manager=perm_mgr, workspace_root=tmp_path)

    proposal = CodeChangeProposal(
        operation=ProposalOperation.MODIFY_FILE.value,
        target_file="verify_me.py",
        proposed_content="x = 2\n",
        original_content="x = 1\n",
        provider="cloud",
        model="gemma4:31b-cloud",
        status=ProposalStatus.PROPOSED.value,
        is_valid=True,
    )

    executor.prepare_proposal(proposal)
    perm_mgr.grant_write_permission()
    exec_result = executor.execute_proposal(proposal=proposal, explicit_permission=True, run_tests=False)

    assert exec_result.success is True
    assert exec_result.stage == ExecutionStage.VERIFIED.value
    assert exec_result.verification_result is not None
    assert exec_result.verification_result.verified is True
    assert test_file.read_text() == "x = 2\n"


def test_requirement_k_cloud_model_cannot_bypass_test_execution(tmp_path: Path) -> None:
    """Requirement K: Test execution is strictly enforced for cloud proposals when tests are present."""
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir(parents=True, exist_ok=True)
    test_file = tests_dir / "test_math.py"
    test_file.write_text("def test_dummy(): assert True\n")

    math_file = tmp_path / "math.py"
    math_file.write_text("x = 1\n")

    perm_mgr = PermissionManager()
    applier = CodeChangeApplier(workspace_root=tmp_path, permission_manager=perm_mgr)
    executor = DeveloperExecutor(applier=applier, permission_manager=perm_mgr, workspace_root=tmp_path)

    proposal = CodeChangeProposal(
        operation=ProposalOperation.MODIFY_FILE.value,
        target_file="math.py",
        proposed_content="x = 2\n",
        original_content="x = 1\n",
        provider="cloud",
        model="gemma4:31b-cloud",
        status=ProposalStatus.PROPOSED.value,
        is_valid=True,
    )

    executor.prepare_proposal(proposal)
    perm_mgr.grant_write_permission()
    result = executor.execute_proposal(proposal=proposal, explicit_permission=True, run_tests=True)

    assert result.success is True
    assert result.test_plan is not None


def test_requirement_l_cloud_model_cannot_falsely_claim_success(tmp_path: Path) -> None:
    """Requirement L: Anti-false success invariant: verification must confirm actual disk change."""
    test_file = tmp_path / "honest.py"
    test_file.write_text("val = 100\n")

    perm_mgr = PermissionManager()
    applier = CodeChangeApplier(workspace_root=tmp_path, permission_manager=perm_mgr)
    executor = DeveloperExecutor(applier=applier, permission_manager=perm_mgr, workspace_root=tmp_path)

    # Intentionally corrupt the applier to not write
    def fake_apply_proposal(proposal: CodeChangeProposal, explicit_permission: bool = True):
        from developer.models import CodeChangeResult
        return CodeChangeResult(
            success=False,
            status_code="application_failed",
            target_file=proposal.target_file,
            operation=proposal.operation,
            applied=False,
            verification_success=False,
            error="Simulated disk write failure",
        )

    applier.apply_proposal = fake_apply_proposal  # type: ignore

    proposal = CodeChangeProposal(
        operation=ProposalOperation.MODIFY_FILE.value,
        target_file="honest.py",
        proposed_content="val = 200\n",
        original_content="val = 100\n",
        provider="cloud",
        model="gemma4:31b-cloud",
        status=ProposalStatus.PROPOSED.value,
        is_valid=True,
    )

    executor.prepare_proposal(proposal)
    perm_mgr.grant_write_permission()
    result = executor.execute_proposal(proposal=proposal, explicit_permission=True, run_tests=False)

    assert result.success is False
    assert result.stage != ExecutionStage.VERIFIED.value
    assert test_file.read_text() == "val = 100\n"


# ============================================================================
# Requirement M, N, O: Failure Handling & Anti-Silent Fallback
# ============================================================================

def test_requirement_m_and_n_cloud_failure_surfaces_clearly_and_leaves_filesystem_untouched(tmp_path: Path) -> None:
    """Requirements M & N: Cloud failure surfaces structured error and leaves disk untouched."""
    source_file = tmp_path / "core.py"
    source_file.write_text("ORIGINAL CONTENT\n")

    failing_cloud = MockProvider(
        "cloud",
        "gemma4:31b-cloud",
        is_cloud=True,
        fail_with=ProviderUnavailable("Ollama cloud connection failed (503 Service Unavailable)"),
    )

    generator = DeveloperCodeGenerator(workspace_root=tmp_path, model=failing_cloud)

    task = DeveloperTask(
        goal="update core.py",
        target_files=["core.py"],
    )
    plan = DeveloperPlan(
        original_task=task,
        relevant_files=["core.py"],
        planned_changes=["update core.py"],
    )

    gen_result = generator.generate_from_plan(task=task, plan=plan)

    assert gen_result.is_valid is False
    assert gen_result.validation_status == "provider_failure"
    assert "ProviderUnavailable" in gen_result.explanation or "503" in gen_result.explanation
    assert gen_result.proposal is None
    assert source_file.read_text() == "ORIGINAL CONTENT\n"


def test_requirement_o_cloud_failure_does_not_silently_fallback_to_local(tmp_path: Path) -> None:
    """Requirement O: Cloud failure does NOT silently fall back to local model."""
    local_mock = MockProvider("local", "gemma4:e4b-it-qat", is_cloud=False, generate_response="Local generated code")
    cloud_failing = MockProvider("cloud", "gemma4:31b-cloud", is_cloud=True, fail_with=ProviderAuthenticationError("Invalid API key"))

    router = ModelRouter(
        provider_type="local",
        developer_provider_type="cloud",
        local_provider=local_mock,
        cloud_provider=cloud_failing,
    )

    generator = DeveloperCodeGenerator(workspace_root=tmp_path, model=router)

    task = DeveloperTask(
        goal="Modify file.py",
        target_files=["file.py"],
    )
    (tmp_path / "file.py").write_text("x = 1\n")
    plan = DeveloperPlan(
        original_task=task,
        relevant_files=["file.py"],
        planned_changes=["Modify file.py"],
    )

    gen_res = generator.generate_from_plan(task=task, plan=plan)

    assert gen_res.is_valid is False
    assert gen_res.validation_status == "provider_failure"
    assert len(local_mock.calls) == 0  # CRITICAL: Local mock was NEVER called
    assert len(cloud_failing.calls) == 1


# ============================================================================
# Requirement P, Q, R: Security Boundaries & Secret Safety
# ============================================================================

def test_requirement_p_cloud_model_cannot_access_outside_workspace(tmp_path: Path) -> None:
    """Requirement P: Path traversal outside workspace root is blocked."""
    cloud_mock = MockProvider("cloud", "gemma4:31b-cloud", is_cloud=True)
    generator = DeveloperCodeGenerator(workspace_root=tmp_path, model=cloud_mock)

    task = DeveloperTask(
        goal="Edit external file",
        target_files=["../../../../etc/passwd"],
    )
    plan = DeveloperPlan(
        original_task=task,
        relevant_files=["../../../../etc/passwd"],
        planned_changes=["Edit file"],
    )

    result = generator.generate_from_plan(task=task, plan=plan)
    assert result.is_valid is False
    assert "outside the allowed workspace boundary" in result.explanation or "Access denied" in result.explanation


def test_requirement_q_cloud_model_cannot_access_protected_files(tmp_path: Path) -> None:
    """Requirement Q: Access to protected files (.git, .env, *.key) is rejected."""
    cloud_mock = MockProvider("cloud", "gemma4:31b-cloud", is_cloud=True)
    generator = DeveloperCodeGenerator(workspace_root=tmp_path, model=cloud_mock)

    protected_targets = [".env", ".git/config", "secret.key", "credentials.pem"]
    for target in protected_targets:
        task = DeveloperTask(
            goal=f"Modify {target}",
            target_files=[target],
        )
        plan = DeveloperPlan(
            original_task=task,
            relevant_files=[target],
            planned_changes=[f"Modify {target}"],
        )
        result = generator.generate_from_plan(task=task, plan=plan)
        assert result.is_valid is False
        assert "Access denied" in result.explanation or "Path validation failed" in result.explanation


def test_requirement_r_cloud_credentials_never_leaked() -> None:
    """Requirement R: API keys are never exposed in repr or error messages."""
    secret = "secret_cloud_token_123456789"
    cloud_prov = OllamaCloudProvider(api_key=secret)

    assert secret not in repr(cloud_prov)
    assert "api_key=<present>" in repr(cloud_prov)

    # Check router repr
    router = ModelRouter(provider_type="cloud", cloud_provider=cloud_prov)
    assert secret not in repr(router)


# ============================================================================
# Requirement S: Provider and Model Transparency
# ============================================================================

def test_requirement_s_developer_execution_summary_identifies_provider_and_model(tmp_path: Path) -> None:
    """Requirement S: Developer execution summary clearly identifies provider and model."""
    test_file = tmp_path / "transparent.py"
    test_file.write_text("a = 1\n")

    perm_mgr = PermissionManager()
    applier = CodeChangeApplier(workspace_root=tmp_path, permission_manager=perm_mgr)
    executor = DeveloperExecutor(applier=applier, permission_manager=perm_mgr, workspace_root=tmp_path)

    proposal = CodeChangeProposal(
        operation=ProposalOperation.MODIFY_FILE.value,
        target_file="transparent.py",
        proposed_content="a = 2\n",
        original_content="a = 1\n",
        provider="cloud",
        model="gemma4:31b-cloud",
        status=ProposalStatus.PROPOSED.value,
        is_valid=True,
    )

    # Check format_proposal_display
    display = executor.format_proposal_display(proposal)
    assert "Provider: cloud (gemma4:31b-cloud)" in display

    # Execute
    prep_res = executor.prepare_proposal(proposal)
    assert prep_res.provider == "cloud"
    assert prep_res.model == "gemma4:31b-cloud"

    perm_mgr.grant_write_permission()
    exec_res = executor.execute_proposal(proposal=proposal, explicit_permission=True, run_tests=False)
    assert exec_res.success is True
    assert exec_res.provider == "cloud"
    assert exec_res.model == "gemma4:31b-cloud"
    assert "Provider: cloud" in exec_res.summary
    assert "Model: gemma4:31b-cloud" in exec_res.summary
