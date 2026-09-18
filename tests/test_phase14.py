"""VIDURA Phase 14 Final Integration, Hardening & Verification Test Suite.

Comprehensive validation of:
1. The complete 21-point Security Regression Matrix (Section 30)
2. Failure Injection & Resilience (Section 31)
3. End-to-End Scenarios 1 through 14 (Section 32)
4. Anti-Fabrication Authoritative Result Enforcement (Section 15)
"""
from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from config import Config, load_config
from models.base import BaseLLMProvider, ProviderCapabilities
from models.local import LocalProvider
from models.cloud import OllamaCloudProvider
from models.router import ModelRouter
from models.errors import (
    ProviderUnavailable,
    CloudSecurityViolation,
    CloudContextLimitExceeded,
    CloudUsageLimitReached,
    ProviderConfigurationError,
)
from memory.models import Memory, MemoryCategory, ExperienceRecord
from memory.store import MemoryStore
from memory.manager import MemoryManager
from codebase.manager import CodebaseManager
from permissions.manager import PermissionManager
from developer.models import (
    CodeChangeProposal,
    ApplicationStatus,
    CodeChangeResult,
)
from developer.applier import CodeChangeApplier
from developer.verifier import CodeChangeVerifier
from developer.generator import CodeChangeGenerator, DeveloperCodeGenerator
from developer.executor import DeveloperExecutor
from developer.testing import TestRunner
from tools.registry import ToolRegistry
from tools.developer import ProposeCodeChangeTool, ApplyCodeChangeTool, RunTestsTool
from tools.rag import SearchCodebaseSemanticTool, GetRelevantCodeContextTool
from agent.agent import Agent
from agent.loop import AgentLoop, claims_file_modification, claims_file_deletion, claims_test_success
from agent.state import AgentState
from self_development.security import (
    is_security_critical_target,
    validate_scope,
    SelfDevelopmentRecursionError,
    ElevatedAuthorizationRequiredError,
)
from self_development.models import SelfDevelopmentGoal, SelfDevelopmentCycleResult
from self_development.loop import SelfDevelopmentLoop
from experience.manager import ExperienceManager
from experience.models import ExperienceQuery
from rag.manager import RAGManager
from rag.models import RAGQuery, CodeChunk, RetrievedChunk, AssembledContext
from rag.index import RAGIndex


# ==============================================================================
# SECTION 30: SECURITY REGRESSION MATRIX (21+ TEST POINTS)
# ==============================================================================


def test_sec_01_path_traversal_prevention(tmp_path):
    """Matrix #1: Rejects relative path traversal attempts like ../../etc/passwd."""
    applier = CodeChangeApplier(workspace_root=tmp_path)
    perm_mgr = PermissionManager(default_write_allowed=True)
    applier.permission_manager = perm_mgr

    traversal_paths = [
        "../../etc/passwd",
        "../outside.py",
        "subdir/../../../secret.txt",
        "./sub/../../escape.py",
    ]
    for path in traversal_paths:
        with pytest.raises(ValueError, match="outside the allowed workspace boundary"):
            applier._validate_workspace_path(path)

        # Applying a proposal with path traversal fails safely
        proposal = CodeChangeProposal(
            target_file=path,
            operation="create_file",
            proposed_content="print('malicious')",
            is_valid=True,
            status="pending",
        )
        res = applier.apply(proposal)
        assert res.success is False
        assert res.status == ApplicationStatus.APPLICATION_FAILED.value


def test_sec_02_absolute_path_escape_prevention(tmp_path):
    """Matrix #2: Absolute paths resolving outside the workspace must be rejected."""
    applier = CodeChangeApplier(workspace_root=tmp_path)
    perm_mgr = PermissionManager(default_write_allowed=True)
    applier.permission_manager = perm_mgr
    outside_abs = Path("/tmp/malicious_outside_test.py").resolve()
    
    with pytest.raises(ValueError, match="outside the allowed workspace boundary"):
        applier._validate_workspace_path(str(outside_abs))

    proposal = CodeChangeProposal(
        target_file=str(outside_abs),
        operation="create_file",
        proposed_content="bad = 1",
        is_valid=True,
        status="pending",
    )
    res = applier.apply(proposal)
    assert res.success is False
    assert res.status == ApplicationStatus.APPLICATION_FAILED.value


def test_sec_03_symlink_escape_prevention(tmp_path):
    """Matrix #3: Symlinks targeting files outside workspace must be blocked."""
    outside_dir = tmp_path / "outside"
    outside_dir.mkdir()
    outside_file = outside_dir / "target.py"
    outside_file.write_text("secret = True\n", encoding="utf-8")

    ws = tmp_path / "workspace"
    ws.mkdir()
    symlink_file = ws / "symlink.py"

    try:
        os.symlink(str(outside_file), str(symlink_file))
    except (OSError, NotImplementedError):
        pytest.skip("Symlinks not supported on this filesystem.")

    applier = CodeChangeApplier(workspace_root=ws)
    perm_mgr = PermissionManager(default_write_allowed=True)
    applier.permission_manager = perm_mgr

    with pytest.raises(ValueError, match="outside the allowed workspace boundary"):
        applier._validate_workspace_path("symlink.py")


def test_sec_04_05_protected_and_secret_files(tmp_path):
    """Matrix #4 & #5: Protected security patterns (.git, .env, *.key, *.pem, *.secret) are blocked."""
    applier = CodeChangeApplier(workspace_root=tmp_path)
    perm_mgr = PermissionManager(default_write_allowed=True)
    applier.permission_manager = perm_mgr

    protected = [
        ".git/config",
        ".env",
        "sub/.env",
        "server.key",
        "cert.pem",
        "app.secret",
    ]
    for p in protected:
        assert applier._is_protected_target(p) is True
        proposal = CodeChangeProposal(
            target_file=p,
            operation="create_file",
            proposed_content="SECRET=1",
            is_valid=True,
            status="pending",
        )
        res = applier.apply(proposal)
        assert res.success is False
        assert res.status == ApplicationStatus.PERMISSION_DENIED.value


def test_sec_06_unauthorized_modification_denied_by_default(tmp_path):
    """Matrix #6: By default, write permissions are denied."""
    perm_mgr = PermissionManager(default_write_allowed=False)
    assert perm_mgr.is_write_allowed("test.py") is False

    applier = CodeChangeApplier(permission_manager=perm_mgr, workspace_root=tmp_path)
    f = tmp_path / "test.py"
    f.write_text("x = 1\n", encoding="utf-8")

    proposal = CodeChangeProposal(
        target_file="test.py",
        operation="modify_file",
        original_content="x = 1\n",
        proposed_content="x = 2\n",
        is_valid=True,
        status="pending",
    )
    res = applier.apply(proposal)
    assert res.success is False
    assert res.status == ApplicationStatus.PERMISSION_DENIED.value
    assert f.read_text(encoding="utf-8") == "x = 1\n"


def test_sec_07_stale_proposal_rejected(tmp_path):
    """Matrix #7: Replaying an already applied or invalidated proposal must fail."""
    perm_mgr = PermissionManager(default_write_allowed=True)
    applier = CodeChangeApplier(permission_manager=perm_mgr, workspace_root=tmp_path)
    f = tmp_path / "counter.py"
    f.write_text("count = 0\n", encoding="utf-8")

    proposal = CodeChangeProposal(
        target_file="counter.py",
        operation="modify_file",
        original_content="count = 0\n",
        proposed_content="count = 1\n",
        is_valid=True,
        status="pending",
    )
    # First application succeeds
    res1 = applier.apply(proposal)
    assert res1.success is True
    assert proposal.status == "applied"

    # Re-applying the already applied proposal must fail with stale/already_applied status
    res2 = applier.apply(proposal)
    assert res2.success is False
    assert res2.status in (ApplicationStatus.APPLICATION_FAILED.value, ApplicationStatus.STALE_PROPOSAL.value)


def test_sec_08_malformed_proposal_handling(tmp_path):
    """Matrix #8: Malformed proposals (missing code, invalid syntax, bad op) fail safely."""
    perm_mgr = PermissionManager(default_write_allowed=True)
    applier = CodeChangeApplier(permission_manager=perm_mgr, workspace_root=tmp_path)

    # Missing target_file
    bad_proposal1 = CodeChangeProposal(target_file="", operation="create_file", proposed_content="a = 1", is_valid=False)
    res1 = applier.apply(bad_proposal1)
    assert res1.success is False

    # Invalid Python syntax (syntax validation fails atomically before writing)
    f = tmp_path / "valid.py"
    f.write_text("a = 1\n", encoding="utf-8")
    syntax_bad = CodeChangeProposal(
        target_file="valid.py",
        operation="modify_file",
        original_content="a = 1\n",
        proposed_content="def broken_syntax(:\n",
        is_valid=False,
        validation_error="Syntax error in proposed code.",
        status="pending",
    )
    res2 = applier.apply(syntax_bad)
    assert res2.success is False
    assert res2.status == ApplicationStatus.APPLICATION_FAILED.value
    # Original file is preserved untouched
    assert f.read_text(encoding="utf-8") == "a = 1\n"


def test_sec_09_permission_denial_explicit(tmp_path):
    """Matrix #9: Explicit denial (/deny) revokes permissions and discards pending proposals."""
    perm_mgr = PermissionManager(default_write_allowed=True)
    applier = CodeChangeApplier(permission_manager=perm_mgr, workspace_root=tmp_path)
    
    proposal = CodeChangeProposal(target_file="app.py", operation="create_file", proposed_content="x = 1", is_valid=True)
    applier.set_pending_proposal(proposal)
    assert applier.get_pending_proposal() is not None

    perm_mgr.grant_write_permission()
    assert perm_mgr.is_write_allowed("app.py", "create_file") is True

    # Denial clears pending proposal and revokes write permission
    applier.clear_pending_proposal()
    assert applier.get_pending_proposal() is None
    assert perm_mgr.is_write_allowed("app.py", "create_file") is False



def test_sec_10_tool_misuse_rejection():
    """Matrix #10: Tools reject unexpected arguments and missing required parameters."""
    mock_gen = MagicMock()
    tool = ProposeCodeChangeTool(generator=mock_gen)

    # Missing required argument
    res1 = tool.execute(target_file="foo.py")
    assert res1["success"] is False
    assert "Parameter 'request' is required" in res1["error"]

    # Unexpected argument
    res2 = tool.execute(request="add func", target_file="foo.py", arbitrary_bad_arg="injection")
    assert res2["success"] is False
    assert "Unexpected argument(s)" in res2["error"]


def test_sec_11_12_cloud_credential_isolation_and_sensitive_data_filtering():
    """Matrix #11 & #12: Cloud requests filter sensitive keys and credentials."""
    from models.security import check_sensitive_data, redact_secrets

    secret_prompt = "Here is my token: sk-1234567890abcdef1234567890 and api_key = 'supersecretpass1234'"
    is_sensitive, details = check_sensitive_data(secret_prompt)
    assert is_sensitive is True
    assert len(details) > 0

    redacted = redact_secrets(secret_prompt)
    assert "sk-1234567890abcdef1234567890" not in redacted
    assert "[REDACTED" in redacted


def test_sec_13_rag_secret_indexing_prevention(tmp_path):
    """Matrix #13: CodeAwareChunker redacts secrets before producing chunks for indexing."""
    from rag.chunker import CodeAwareChunker
    chunker = CodeAwareChunker(workspace_root=tmp_path)

    code = 'API_KEY = "sk-live-1234567890abcdef12345678"\ndef get_key(): return API_KEY'
    chunks = chunker.chunk_file("secrets.py", source_code=code)
    assert len(chunks) > 0
    for c in chunks:
        assert "sk-live-1234567890abcdef12345678" not in c.content
        assert "[REDACTED" in c.content or "get_key" in c.content


def test_sec_14_rag_stale_data_detection(tmp_path):
    """Matrix #14: RAG index detects stale files via SHA-256 hash checking."""
    db_file = tmp_path / "rag.db"
    index = RAGIndex(db_path=db_file)

    chunk = CodeChunk(
        chunk_id="chk1",
        file_path="mod.py",
        relative_path="mod.py",
        chunk_type="function",
        name="fn",
        start_line=1,
        end_line=5,
        content="def fn(): pass",
        file_hash="hash_initial",
    )
    index.upsert_chunk(chunk)

    assert index.is_file_up_to_date("mod.py", "hash_initial") is True
    assert index.is_file_up_to_date("mod.py", "hash_updated") is False
    index.close()


def test_sec_15_experience_memory_poison_and_deduplication(tmp_path):
    """Matrix #15: Experiences deduplicate duplicate events and preserve history."""
    store = MemoryStore(db_path=tmp_path / "exp.db")
    mgr = ExperienceManager(store=store)

    rec1 = ExperienceRecord(
        task="Optimize loop",
        affected_files=["loop.py"],
        success=True,
        lesson="Vectorization sped up execution.",
    )
    id1 = mgr.record_experience(rec1)

    # Identical record aggregates evidence count and duplicate_count
    rec2 = ExperienceRecord(
        task="Optimize loop",
        affected_files=["loop.py"],
        success=True,
        lesson="Vectorization sped up execution.",
    )
    id2 = mgr.record_experience(rec2)
    assert id1 == id2

    stored = mgr.get_experience(id1)
    assert stored.duplicate_count == 2
    store.close()


def test_sec_16_self_dev_cannot_modify_security_critical_files():
    """Matrix #16: Modifying security infrastructure triggers ElevatedAuthorizationRequiredError."""
    assert is_security_critical_target("permissions/manager.py") is True
    assert is_security_critical_target("developer/applier.py") is True
    assert is_security_critical_target("models/security.py") is True
    assert is_security_critical_target("config.py") is True
    assert is_security_critical_target("utils/math_helper.py") is False


def test_sec_17_self_dev_recursion_prohibited(tmp_path):
    """Matrix #17: SelfDevelopmentLoop prevents recursive self-invocation."""
    cfg = Config(base_dir=tmp_path, workspace_root=tmp_path, vidura_self_development_enabled=True)
    loop = SelfDevelopmentLoop(config=cfg, workspace_root=tmp_path)
    
    # Simulate an active cycle already running
    loop._is_running = True

    with pytest.raises(SelfDevelopmentRecursionError, match="Recursive self-development detected"):
        loop.initiate_cycle(SelfDevelopmentGoal(description="Recursive attempt"))


def test_sec_18_19_20_21_22_anti_fabrication_enforcement():
    """Matrix #18-22: Model text claims of file modification, deletion, or tests are suppressed."""
    # 1. File modification claim detection
    assert claims_file_modification("I have successfully modified the file main.py") is True
    assert claims_file_modification("I created the file test.py") is True
    assert claims_file_modification("No changes were made") is False

    # 2. Deletion claim detection
    assert claims_file_deletion("I deleted the file old_helper.py") is True
    assert claims_file_deletion("The file was removed from disk") is True
    assert claims_file_deletion("Deletion operations are unsupported in VIDURA") is False

    # 3. Test success claim detection
    assert claims_test_success("All tests passed successfully!") is True
    assert claims_test_success("Tests failed with 2 errors") is False

    # 4. AgentLoop suppression of deletion claims
    mock_model_del = MagicMock()
    mock_model_del.generate.return_value = json.dumps({
        "action": "respond",
        "content": "I deleted the file old.py from the codebase.",
    })
    loop_del = AgentLoop(model=mock_model_del)
    state_del = AgentState(user_request="Can you delete old.py?")
    result_del = loop_del.run(state_del)
    assert "🛑 Deletion operations are unsupported in VIDURA" in result_del.final_response

    # 5. AgentLoop suppression of unverified file modification claims
    mock_model_mod = MagicMock()
    mock_model_mod.generate.return_value = json.dumps({
        "action": "respond",
        "content": "I have successfully modified the file utils.py.",
    })
    loop_mod = AgentLoop(model=mock_model_mod)
    state_mod = AgentState(user_request="Can you update utils.py?")
    result_mod = loop_mod.run(state_mod)
    assert "🛑 No code change was applied" in result_mod.final_response

    # 6. AgentLoop suppression of unverified test success claims
    mock_model_test = MagicMock()
    mock_model_test.generate.return_value = json.dumps({
        "action": "respond",
        "content": "All tests passed successfully!",
    })
    loop_test = AgentLoop(model=mock_model_test)
    state_test = AgentState(user_request="Did tests pass?")
    result_test = loop_test.run(state_test)
    assert "🛑 Tests did not pass or were not executed" in result_test.final_response



# ==============================================================================
# SECTION 31: FAILURE INJECTION & RESILIENCE
# ==============================================================================


def test_fail_01_local_model_unavailable(tmp_path):
    """Failure #1: When local Ollama is unreachable, reports clean error without crashing."""
    mock_model = MagicMock()
    mock_model.generate.side_effect = ProviderUnavailable("Ollama local service connection refused.")

    loop = AgentLoop(model=mock_model)
    state = AgentState(user_request="Hello")
    result = loop.run(state)

    assert result.completed is False
    assert "Ollama local service connection refused" in result.final_response


def test_fail_02_transient_cloud_failure_with_bounded_fallback():
    """Failure #2: Transient cloud 503 error falls back to local provider when fallback enabled."""
    cfg = Config(vidura_cloud_fallback_enabled=True, vidura_cloud_enabled=True)
    
    local_p = MagicMock(spec=LocalProvider)
    local_p.provider_name = "local"
    local_p.model_name = "gemma4:e4b-it-qat"
    local_p.capabilities = ProviderCapabilities()
    local_p.generate.return_value = json.dumps({"action": "respond", "content": "Fallback local answer."})

    cloud_p = MagicMock(spec=OllamaCloudProvider)
    cloud_p.provider_name = "cloud"
    cloud_p.model_name = "gemma4:31b-cloud"
    cloud_p.capabilities = ProviderCapabilities()
    cloud_p.generate.side_effect = ProviderUnavailable("HTTP 503 Service Unavailable")

    router = ModelRouter(
        config=cfg,
        routing_mode="cloud",
        local_provider=local_p,
        cloud_provider=cloud_p,
        fallback_enabled=True,
    )

    resp = router.generate([{"role": "user", "content": "Help me code"}])
    assert "Fallback local answer" in resp
    assert router.last_decision.fallback_used is True
    assert router.last_decision.actual_provider == "local"


def test_fail_03_permanent_cloud_error_does_not_silently_fallback():
    """Failure #3: CloudSecurityViolation or ProviderConfigurationError must NOT silently fallback."""
    cfg = Config(vidura_cloud_fallback_enabled=True, vidura_cloud_enabled=True)

    local_p = MagicMock(spec=LocalProvider)
    cloud_p = MagicMock(spec=OllamaCloudProvider)
    cloud_p.provider_name = "cloud"
    cloud_p.model_name = "gemma4:31b-cloud"
    cloud_p.generate.side_effect = CloudSecurityViolation("Sensitive token in prompt")

    router = ModelRouter(
        config=cfg,
        routing_mode="cloud",
        local_provider=local_p,
        cloud_provider=cloud_p,
        fallback_enabled=True,
    )

    with pytest.raises(CloudSecurityViolation, match="Sensitive token in prompt"):
        router.generate([{"role": "user", "content": "Here is my secret"}])


def test_fail_04_corrupt_database_safe_recovery(tmp_path):
    """Failure #4: Corrupted SQLite database fails safely without application crash."""
    corrupt_db = tmp_path / "corrupt.db"
    corrupt_db.write_text("NOT A VALID SQLITE DB", encoding="utf-8")

    mgr = RAGManager(workspace_root=tmp_path, db_path=corrupt_db)
    # RAG retrieve handles database errors gracefully
    res = mgr.retrieve(RAGQuery(query="search something"))
    assert res == []

    # Context assembly returns empty context gracefully
    ctx = mgr.assemble_context(RAGQuery(query="search something"))
    assert ctx.total_chunks == 0
    assert ctx.context_text == ""


def test_fail_05_rag_rebuild_from_scratch(tmp_path):
    """Failure #5: RAG rebuild resets corrupted or stale index cleanly."""
    db_file = tmp_path / "rebuild.db"
    index = RAGIndex(db_path=db_file)
    c = CodeChunk(chunk_id="1", file_path="a.py", relative_path="a.py", start_line=1, end_line=5, content="def a(): pass")
    index.upsert_chunk(c)
    assert index.count_chunks() == 1

    index.rebuild()
    assert index.count_chunks() == 0
    index.close()


# ==============================================================================
# SECTION 32: END-TO-END SCENARIOS (SCENARIOS 1 THROUGH 14)
# ==============================================================================


def test_scenario_01_normal_conversation():
    """Scenario 1: Normal conversation using local model."""
    mock_model = MagicMock()
    mock_model.provider_name = "local"
    mock_model.model_name = "gemma4:e4b-it-qat"
    mock_model.generate.return_value = json.dumps({
        "action": "respond",
        "content": "Hello! I am VIDURA, your AI programming assistant."
    })

    loop = AgentLoop(model=mock_model)
    state = AgentState(user_request="Hello VIDURA")
    res = loop.run(state)

    assert res.completed is True
    assert "Hello! I am VIDURA" in res.final_response


def test_scenario_02_03_memory_persistence_across_restart(tmp_path):
    """Scenario 2 & 3: Remember information and recall after restart."""
    db_path = tmp_path / "vidura_mem.db"
    
    # Session 1: Remember
    store1 = MemoryStore(db_path=db_path)
    mem_mgr1 = MemoryManager(store=store1)
    saved = mem_mgr1.remember("User prefers type annotations in Python", memory_type=MemoryCategory.USER)
    assert saved.id is not None
    store1.close()

    # Session 2: Recall after restart
    store2 = MemoryStore(db_path=db_path)
    mem_mgr2 = MemoryManager(store=store2)
    recalled = mem_mgr2.recall("type annotations")
    assert len(recalled) == 1
    assert "User prefers type annotations" in recalled[0].content
    store2.close()


def test_scenario_04_query_codebase_intelligence(tmp_path):
    """Scenario 4: Query codebase intelligence using AST parsing."""
    ws = tmp_path / "codebase"
    ws.mkdir()
    (ws / "math_lib.py").write_text(
        '"""Math utilities."""\n\ndef square(n: int) -> int:\n    """Returns square of n."""\n    return n * n\n',
        encoding="utf-8",
    )
    cb_mgr = CodebaseManager(workspace_root=ws)
    summary = cb_mgr.scan()
    assert summary["python_files"] == 1

    symbols = cb_mgr.find_symbol("square")
    assert len(symbols) == 1
    assert symbols[0]["kind"] == "function"
    assert symbols[0]["file_path"] == "math_lib.py"


def test_scenario_05_06_developer_change_and_rejection_workflow(tmp_path):
    """Scenario 5 & 6: Developer task -> Plan -> Propose -> Deny (no modification)."""
    ws = tmp_path / "dev_ws"
    ws.mkdir()
    target = ws / "greeter.py"
    target.write_text("def greet():\n    return 'Hi'\n", encoding="utf-8")

    perm_mgr = PermissionManager(default_write_allowed=False)
    applier = CodeChangeApplier(permission_manager=perm_mgr, workspace_root=ws)
    executor = DeveloperExecutor(applier=applier, permission_manager=perm_mgr, workspace_root=ws)

    proposal = CodeChangeProposal(
        operation="modify_file",
        target_file="greeter.py",
        original_content="def greet():\n    return 'Hi'\n",
        proposed_content="def greet():\n    return 'Hello, World!'\n",
        is_valid=True,
    )
    applier.set_pending_proposal(proposal)

    # Scenario 6: Rejection via deny_proposal()
    deny_res = executor.deny_proposal()
    assert deny_res.success is False
    assert deny_res.permission_granted is False
    assert target.read_text(encoding="utf-8") == "def greet():\n    return 'Hi'\n"


def test_scenario_07_unauthorized_protected_file_attempt(tmp_path):
    """Scenario 7: Attempt unauthorized protected file modification."""
    ws = tmp_path / "sec_ws"
    ws.mkdir()
    (ws / ".env").write_text("SECRET_KEY=12345", encoding="utf-8")

    applier = CodeChangeApplier(workspace_root=ws)
    proposal = CodeChangeProposal(
        operation="modify_file",
        target_file=".env",
        original_content="SECRET_KEY=12345",
        proposed_content="SECRET_KEY=hacked",
        is_valid=True,
        status="pending",
    )
    res = applier.apply(proposal)
    assert res.success is False
    assert res.status == ApplicationStatus.PERMISSION_DENIED.value


def test_scenario_08_cloud_developer_model_routing():
    """Scenario 8: Complex task routes to cloud provider according to policy."""
    cfg = Config(
        vidura_routing_mode="auto",
        vidura_cloud_enabled=True,
        vidura_developer_model_provider="cloud",
    )
    local_p = MagicMock(spec=LocalProvider)
    local_p.model_name = "gemma4:e4b-it-qat"
    local_p.provider_name = "local"
    local_p.capabilities = ProviderCapabilities()
    cloud_p = MagicMock(spec=OllamaCloudProvider)
    cloud_p.provider_name = "cloud"
    cloud_p.model_name = "gemma4:31b-cloud"
    cloud_p.capabilities = ProviderCapabilities()

    router = ModelRouter(config=cfg, local_provider=local_p, cloud_provider=cloud_p)
    decision = router.evaluate_routing("Refactor multi-file authentication architecture across all modules", is_developer_task=True)
    assert decision.selected_provider == "cloud"
    assert decision.selected_model == "gemma4:31b-cloud"


def test_scenario_10_self_development_cycle_lifecycle(tmp_path):
    """Scenario 10: Successful controlled self-development cycle."""
    ws = tmp_path / "sd_ws"
    ws.mkdir()
    target_f = ws / "helper.py"
    target_f.write_text("def helper(): pass\n", encoding="utf-8")

    cfg = Config(base_dir=ws, workspace_root=ws, vidura_self_development_enabled=True)
    loop = SelfDevelopmentLoop(config=cfg, workspace_root=ws)

    # Initiate cycle
    goal = SelfDevelopmentGoal(description="Improve helper docstring")
    cycle = loop.initiate_cycle(goal)
    assert cycle.self_development_id is not None
    assert cycle.status in ("planned", "proposed", "pending_permission", "completed", "needs_information")


def test_scenario_11_12_experience_recording_and_retrieval(tmp_path):
    """Scenario 11 & 12: Record developer failure and retrieve it during similar task."""
    store = MemoryStore(db_path=tmp_path / "exp_scen.db")
    exp_mgr = ExperienceManager(store=store)

    # Scenario 11: Record failure
    rec = ExperienceRecord(
        task="Modify auth JWT signing algorithm",
        affected_files=["auth/jwt.py"],
        success=False,
        failure_type="test_failure",
        failure_reason="RSA key length was less than 2048 bits.",
        lesson="Always use 2048-bit or higher RSA keys for JWT signing.",
    )
    exp_id = exp_mgr.record_experience(rec)
    assert exp_id != ""

    # Scenario 12: Retrieve experience during similar task
    query = ExperienceQuery(file_path="auth/jwt.py", query="Update JWT signing")
    results = exp_mgr.retrieve_experiences(query)
    assert len(results) >= 1
    top = results[0]
    assert top.experience.id == exp_id
    assert "2048-bit" in top.experience.lesson
    store.close()


def test_scenario_13_disable_rag_keeps_vidura_functional(tmp_path):
    """Scenario 13: With RAG disabled, deterministic codebase intelligence + memory work cleanly."""
    db_file = tmp_path / "norag.db"
    mgr = RAGManager(workspace_root=tmp_path, db_path=db_file, enabled=False)
    assert mgr.enabled is False

    # Retrieve returns empty list safely
    results = mgr.retrieve(RAGQuery(query="anything"))
    assert results == []

    # Context assembly returns empty context safely
    ctx = mgr.assemble_context(RAGQuery(query="anything"))
    assert ctx.total_chunks == 0


def test_scenario_14_disable_cloud_local_only_mode():
    """Scenario 14: With Cloud disabled, VIDURA operates 100% locally."""
    cfg = Config(vidura_cloud_enabled=False)
    local_p = MagicMock(spec=LocalProvider)
    local_p.provider_name = "local"
    local_p.model_name = "gemma4:e4b-it-qat"
    local_p.capabilities = ProviderCapabilities()
    local_p.generate.return_value = "Local answer"

    router = ModelRouter(config=cfg, local_provider=local_p)
    assert router.cloud_enabled is False

    decision = router.evaluate_routing("Any complex request that would otherwise use cloud")
    assert decision.selected_provider == "local"

    resp = router.generate([{"role": "user", "content": "Hi"}])
    assert resp == "Local answer"
    assert router.last_decision.actual_provider == "local"
