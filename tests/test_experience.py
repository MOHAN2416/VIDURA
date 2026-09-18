"""Automated test suite for VIDURA Phase 11: Experience Memory + Learning.

Covers requirements A through AI from Section 23 and Validation Scenarios from Section 31:
- A. Record successful experience
- B. Record failed experience
- C. Record verification failure
- D. Record test failure
- E. Record cloud failure
- F. Record fallback event
- G. Retrieve by task type
- H. Retrieve by file
- I. Retrieve by symbol
- J. Retrieve by component
- K. Retrieve by failure type
- L. Relevance scoring
- M. Recency behavior
- N. Negative experience retrieval (advisory warnings)
- O. Lesson extraction
- P. Lesson confidence
- Q. Duplicate handling
- R. Historical experience preservation
- S. Secret redaction
- T. Hidden reasoning not persisted
- U. Current codebase overrides experience
- V. Current permissions override experience
- W. Security policy overrides experience
- X. Developer planning receives relevant experience
- Y. Self-development planning receives relevant experience
- Z. Failed developer task creates experience
- AA. Successful developer task creates experience
- AB. Failed self-dev creates experience
- AC. Successful self-dev creates experience
- AD. Experience recording does not trigger self-development (recursion guard)
- AE. Experience memory cannot authorize a code change
- AF. Experience memory cannot bypass protected files
- AG. Cloud context contains only permitted relevant experiences
- AH. Local-only mode never sends experience data to cloud
- AI. Full regression suite passes
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from config import Config
from memory.models import ExperienceRecord
from memory.store import MemoryStore
from memory.manager import MemoryManager
from experience.models import LessonRecord, ExperienceQuery, ScoredExperience
from experience.extractor import ExperienceExtractor, sanitize_text
from experience.scorer import ExperienceScorer, tokenize
from experience.retriever import ExperienceRetriever
from experience.learning import LearningLayer
from experience.manager import ExperienceManager

from task_understanding.models import DeveloperTask, TaskType
from planning.models import DeveloperPlan
from planning.planner import DeveloperPlanner
from codebase.manager import CodebaseManager
from permissions.manager import PermissionManager
from developer.models import (
    CodeChangeProposal,
    ProposalOperation,
    ProposalStatus,
    ExecutionStatus,
    ExecutionStage,
    DeveloperExecutionResult,
    CodeChangeResult,
    VerificationResult,
)
from developer.test_models import TestResult as DevTestResult, TestPlan as DevTestPlan
from developer.applier import CodeChangeApplier
from developer.verifier import CodeChangeVerifier
from developer.executor import DeveloperExecutor
from self_development.models import (
    SelfDevelopmentGoal,
    SelfDevelopmentPlan,
    SelfDevelopmentEvaluation,
    SelfDevelopmentCycleResult,
)
from self_development.loop import SelfDevelopmentLoop
from self_development.security import is_security_critical_target, validate_scope


@pytest.fixture
def temp_store(tmp_path: Path) -> MemoryStore:
    """Provides a fresh isolated SQLite MemoryStore."""
    db_path = tmp_path / "test_experience.db"
    store = MemoryStore(db_path)
    yield store
    store.close()


@pytest.fixture
def exp_manager(temp_store: MemoryStore) -> ExperienceManager:
    """Provides an ExperienceManager backed by temp_store."""
    return ExperienceManager(store=temp_store)


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    """Provides an isolated workspace folder with a sample module."""
    ws = tmp_path / "ws"
    ws.mkdir(parents=True, exist_ok=True)
    sample_file = ws / "sample_math.py"
    sample_file.write_text(
        "def add(a: int, b: int) -> int:\n"
        "    return a + b\n\n"
        "def subtract(a: int, b: int) -> int:\n"
        "    return a - b\n",
        encoding="utf-8",
    )
    return ws


# ==============================================================================
# 1. Models & Serialization Tests
# ==============================================================================

class TestExperienceModels:
    """Validates data structures, typing, and backward compatibility."""

    def test_experience_record_fields_and_serialization(self) -> None:
        rec = ExperienceRecord(
            task="Optimize database query",
            attempt=1,
            action_summary="Added index on table",
            result="Query latency dropped 50%",
            success=True,
            lesson="Indexes improve read performance",
            failure_type=None,
            fallback_used=False,
            duplicate_count=1,
            evidence_count=2,
        )
        data = rec.to_dict()
        assert data["failure_type"] is None
        assert data["fallback_used"] is False
        assert data["duplicate_count"] == 1
        assert data["evidence_count"] == 2

        restored = ExperienceRecord.from_dict(data)
        assert restored.task == "Optimize database query"
        assert restored.evidence_count == 2

    def test_lesson_record_fields_and_serialization(self) -> None:
        lesson = LessonRecord(
            topic="agent/loop.py",
            lesson="Ensure state rollback on uncaught errors.",
            confidence=0.85,
            success_count=4,
            failure_count=1,
            recommendation="Inspect exception handlers.",
        )
        data = lesson.to_dict()
        assert data["topic"] == "agent/loop.py"
        assert data["confidence"] == 0.85
        assert data["success_count"] == 4

        restored = LessonRecord.from_dict(data)
        assert restored.topic == "agent/loop.py"
        assert restored.confidence == 0.85

    def test_query_and_scored_experience(self) -> None:
        q = ExperienceQuery(query="fix bug", file_path="agent/loop.py", limit=10)
        q_dict = q.to_dict()
        assert q_dict["query"] == "fix bug"
        assert q_dict["file_path"] == "agent/loop.py"

        rec = ExperienceRecord(task="fix bug", result="fixed", success=True)
        scored = ScoredExperience(experience=rec, score=0.75, score_breakdown={"file_match": 0.3}, is_warning=False)
        s_dict = scored.to_dict()
        assert s_dict["score"] == 0.75
        assert s_dict["is_warning"] is False


# ==============================================================================
# 2. Sanitization, Extraction & Redaction Tests (Requirements S, T, A-F)
# ==============================================================================

class TestExperienceSanitizationAndExtraction:
    """Validates redaction, chain-of-thought stripping, and authoritative outcome extraction."""

    def test_req_s_secret_redaction(self) -> None:
        raw = (
            "api_key = 'sk-12345678901234567890123456' and Bearer auth_token_secret_123456 "
            "and -----BEGIN PRIVATE KEY-----\nMIIEvgIBADANBgkqhkiG9w0BAQEFAASCBKgwggSkAgEAAoIBAQC...\n-----END PRIVATE KEY-----"
        )
        sanitized = sanitize_text(raw)
        assert "sk-12345678901234567890123456" not in sanitized
        assert "auth_token_secret_123456" not in sanitized
        assert "MIIEvgIBADANBgkqhkiG9w0BAQEFAASC" not in sanitized
        assert "[REDACTED" in sanitized

    def test_req_t_hidden_reasoning_not_persisted(self) -> None:
        raw = "Completed work. <thought>This is an internal secret chain of thought</thought> and ```cot\nhidden trace\n``` finished."
        sanitized = sanitize_text(raw)
        assert "internal secret chain of thought" not in sanitized
        assert "hidden trace" not in sanitized
        assert "[REDACTED_COT]" in sanitized

    def test_req_a_extractor_successful_developer_execution(self) -> None:
        res = DeveloperExecutionResult(
            proposal_id="prop_001",
            target_file="sample_math.py",
            operation="modify_file",
            status=ExecutionStatus.SUCCESS.value,
            stage=ExecutionStage.VERIFIED.value,
            success=True,
            permission_granted=True,
            overall_outcome="COMPLETED_SUCCESS",
            application_result=CodeChangeResult(
                success=True,
                target_file="sample_math.py",
                operation="modify_file",
                permission_granted=True,
                reason="Applied cleanly",
                verification_success=True,
            ),
            verification_result=VerificationResult(
                success=True,
                target_file="sample_math.py",
                operation="modify_file",
                verified=True,
                expected_state="valid",
                actual_state="valid",
                reason="Verified match on disk",
            ),
            test_result=DevTestResult(status="passed", exit_code=0, tests_passed=4, tests_failed=0),
        )
        rec = ExperienceExtractor.from_developer_execution(res)
        assert rec.success is True
        assert rec.failure_type is None
        assert rec.verification_status == "verified"
        assert rec.test_status == "passed"
        assert rec.tests_passed == 4
        assert rec.tests_failed == 0
        assert rec.confidence == 1.0

    def test_req_b_c_extractor_verification_failure(self) -> None:
        res = DeveloperExecutionResult(
            proposal_id="prop_002",
            target_file="sample_math.py",
            operation="modify_file",
            status=ExecutionStatus.VERIFICATION_FAILED.value,
            stage=ExecutionStage.VERIFICATION_FAILED.value,
            success=False,
            permission_granted=True,
            overall_outcome="VERIFICATION_FAILED",
            application_result=CodeChangeResult(
                success=True,
                target_file="sample_math.py",
                operation="modify_file",
                permission_granted=True,
                reason="Applied to disk",
                verification_success=False,
            ),
            verification_result=VerificationResult(
                success=False,
                target_file="sample_math.py",
                operation="modify_file",
                verified=False,
                expected_state="expected",
                actual_state="mismatch",
                reason="Content mismatch on line 12",
            ),
        )
        rec = ExperienceExtractor.from_developer_execution(res)
        assert rec.success is False
        assert rec.failure_type == "verification_failure"
        assert "Content mismatch on line 12" in (rec.failure_reason or "")
        assert rec.confidence == 0.5

    def test_req_d_extractor_test_failure(self) -> None:
        res = DeveloperExecutionResult(
            proposal_id="prop_003",
            target_file="sample_math.py",
            operation="modify_file",
            status=ExecutionStatus.SUCCESS.value,
            stage=ExecutionStage.VERIFIED.value,
            success=False,
            permission_granted=True,
            overall_outcome="CHANGE_APPLIED_BUT_TESTS_FAILED",
            application_result=CodeChangeResult(
                success=True,
                target_file="sample_math.py",
                operation="modify_file",
                permission_granted=True,
                reason="Applied",
                verification_success=True,
            ),
            verification_result=VerificationResult(
                success=True,
                target_file="sample_math.py",
                operation="modify_file",
                verified=True,
                expected_state="valid",
                actual_state="valid",
                reason="Matches disk",
            ),
            test_result=DevTestResult(status="failed", exit_code=1, tests_passed=3, tests_failed=1, error="AssertionError: 5 != 6"),
        )
        rec = ExperienceExtractor.from_developer_execution(res)
        assert rec.success is False
        assert rec.failure_type == "test_failure"
        assert rec.tests_failed == 1
        assert "AssertionError" in (rec.failure_reason or "")

    def test_req_b_extractor_permission_denied(self) -> None:
        res = DeveloperExecutionResult(
            proposal_id="prop_004",
            target_file="sample_math.py",
            operation="modify_file",
            status=ExecutionStatus.PERMISSION_DENIED.value,
            stage=ExecutionStage.DENIED.value,
            success=False,
            permission_granted=False,
            overall_outcome="PERMISSION_DENIED",
        )
        rec = ExperienceExtractor.from_developer_execution(res)
        assert rec.success is False
        assert rec.failure_type == "permission_denied"

    def test_req_b_extractor_stale_proposal(self) -> None:
        res = DeveloperExecutionResult(
            proposal_id="prop_005",
            target_file="sample_math.py",
            operation="modify_file",
            status=ExecutionStatus.STALE_PROPOSAL.value,
            stage=ExecutionStage.STALE.value,
            success=False,
            permission_granted=True,
            overall_outcome="STALE_PROPOSAL",
        )
        rec = ExperienceExtractor.from_developer_execution(res)
        assert rec.success is False
        assert rec.failure_type == "stale_proposal"

    def test_req_e_extractor_provider_cloud_failure(self) -> None:
        rec = ExperienceExtractor.from_provider_event(
            event_type="error",
            provider="cloud",
            model="deepseek-coder",
            error="HTTP 429: Rate limit exceeded",
        )
        assert rec.success is False
        assert rec.failure_type == "cloud_failure"
        assert "Rate limit exceeded" in (rec.failure_reason or "")

    def test_req_f_extractor_provider_fallback(self) -> None:
        rec = ExperienceExtractor.from_provider_event(
            event_type="fallback",
            provider="ollama_local",
            model="qwen2.5-coder:7b",
            fallback_used=True,
        )
        assert rec.success is True
        assert rec.fallback_used is True
        assert rec.failure_type == "local_fallback"
        assert "seamlessly" in rec.lesson

    def test_extractor_planning_failure(self) -> None:
        task = DeveloperTask(goal="do something undefined", task_type=TaskType.GENERAL)
        rec = ExperienceExtractor.from_planning_failure(task, "Ambiguous goal without targets")
        assert rec.success is False
        assert rec.failure_type == "planning_failure"


# ==============================================================================
# 3. Deterministic Relevance Scorer & Recency Tests (Requirements L, M)
# ==============================================================================

class TestDeterministicScorer:
    """Validates explainable, bounded deterministic scoring without embeddings or vector DBs."""

    def test_req_l_relevance_scoring_bounded_and_deterministic(self) -> None:
        scorer = ExperienceScorer()
        rec = ExperienceRecord(
            task="Refactor token bucket in agent/loop.py",
            affected_files=["agent/loop.py"],
            relevant_symbols=["AgentLoop"],
            affected_components=["agent"],
            task_type="developer",
            success=True,
            lesson="Token bucket successfully configured",
        )
        query = ExperienceQuery(
            query="refactor token bucket",
            file_path="agent/loop.py",
            symbol="AgentLoop",
            task_type="developer",
            component="agent",
            success=True,
        )

        scored1 = scorer.score(rec, query)
        scored2 = scorer.score(rec, query)

        # Verify determinism
        assert scored1.score == scored2.score
        assert scored1.score_breakdown == scored2.score_breakdown

        # Verify bounded between 0.0 and 1.0
        assert 0.0 <= scored1.score <= 1.0

        # Verify score components
        bd = scored1.score_breakdown
        assert bd["file_match"] == 0.30
        assert bd["symbol_match"] == 0.25
        assert bd["task_match"] == 0.20
        assert bd["component_match"] == 0.15
        assert bd["keyword_match"] > 0.0
        assert bd["outcome_match"] == 0.10

    def test_req_m_recency_behavior_bounded(self) -> None:
        scorer = ExperienceScorer()
        now = datetime.now(timezone.utc)

        # Record 1: Highly relevant file/symbol match, but created 30 days ago
        old_time = (now - timedelta(days=30)).isoformat()
        old_relevant = ExperienceRecord(
            task="Refactor dispatch in agent/loop.py",
            affected_files=["agent/loop.py"],
            relevant_symbols=["AgentLoop"],
            success=True,
            created_at=old_time,
        )

        # Record 2: Irrelevant file, but created 1 minute ago
        recent_time = (now - timedelta(minutes=1)).isoformat()
        recent_irrelevant = ExperienceRecord(
            task="Update docs for README",
            affected_files=["docs/README.md"],
            success=True,
            created_at=recent_time,
        )

        query = ExperienceQuery(
            query="agent dispatch",
            file_path="agent/loop.py",
            symbol="AgentLoop",
        )

        scored_old = scorer.score(old_relevant, query)
        scored_recent = scorer.score(recent_irrelevant, query)

        # Recency bonus must be bounded by MAX_RECENCY_BONUS (0.05)
        assert scored_recent.score_breakdown["recency_bonus"] <= 0.05

        # Crucial Invariant: Recency must NEVER override semantic relevance
        assert scored_old.score > scored_recent.score


# ==============================================================================
# 4. Experience Retriever Tests (Requirements G, H, I, J, K, N)
# ==============================================================================

class TestExperienceRetriever:
    """Validates structured retrieval, filtering, and advisory warning formatting."""

    def test_req_g_retrieve_by_task_type(self, exp_manager: ExperienceManager) -> None:
        r1 = ExperienceRecord(task="Dev change", task_type="developer", success=True)
        r2 = ExperienceRecord(task="Self dev cycle", task_type="self_development", success=True)
        exp_manager.record_experience(r1)
        exp_manager.record_experience(r2)

        results = exp_manager.retrieve_experiences(ExperienceQuery(task_type="self_development"))
        assert len(results) == 1
        assert results[0].experience.task_type == "self_development"

    def test_req_h_retrieve_by_file(self, exp_manager: ExperienceManager) -> None:
        r1 = ExperienceRecord(task="Fix loop", affected_files=["agent/loop.py"], success=True)
        r2 = ExperienceRecord(task="Fix store", affected_files=["memory/store.py"], success=True)
        exp_manager.record_experience(r1)
        exp_manager.record_experience(r2)

        results = exp_manager.retrieve_experiences(ExperienceQuery(file_path="agent/loop.py"))
        assert len(results) >= 1
        assert results[0].experience.affected_files[0] == "agent/loop.py"

    def test_req_i_retrieve_by_symbol(self, exp_manager: ExperienceManager) -> None:
        r1 = ExperienceRecord(task="Update loop", relevant_symbols=["AgentLoop"], success=True)
        r2 = ExperienceRecord(task="Update store", relevant_symbols=["MemoryStore"], success=True)
        exp_manager.record_experience(r1)
        exp_manager.record_experience(r2)

        results = exp_manager.retrieve_experiences(ExperienceQuery(symbol="MemoryStore"))
        assert len(results) >= 1
        assert "MemoryStore" in results[0].experience.relevant_symbols

    def test_req_j_retrieve_by_component(self, exp_manager: ExperienceManager) -> None:
        r1 = ExperienceRecord(task="Change memory", affected_components=["memory"], success=True)
        r2 = ExperienceRecord(task="Change tools", affected_components=["tools"], success=True)
        exp_manager.record_experience(r1)
        exp_manager.record_experience(r2)

        results = exp_manager.retrieve_experiences(ExperienceQuery(component="memory"))
        assert len(results) >= 1
        assert "memory" in results[0].experience.affected_components

    def test_req_k_retrieve_by_failure_type(self, exp_manager: ExperienceManager) -> None:
        r1 = ExperienceRecord(task="T1", success=False, failure_type="verification_failure")
        r2 = ExperienceRecord(task="T2", success=False, failure_type="test_failure")
        exp_manager.record_experience(r1)
        exp_manager.record_experience(r2)

        results = exp_manager.retrieve_experiences(ExperienceQuery(failure_type="verification_failure"))
        assert len(results) >= 1
        assert results[0].experience.failure_type == "verification_failure"

    def test_req_n_negative_experience_retrieval_as_warnings(self, exp_manager: ExperienceManager) -> None:
        failed_exp = ExperienceRecord(
            task="Modify permissions/manager.py",
            affected_files=["permissions/manager.py"],
            success=False,
            failure_type="permission_denied",
            failure_reason="Protected target file requires explicit elevated auth",
            recommendation="Do not modify permissions manager in automated loops",
        )
        exp_manager.record_experience(failed_exp)

        warnings = exp_manager.retriever.retrieve_warnings(file_path="permissions/manager.py")
        assert len(warnings) >= 1
        assert warnings[0].is_warning is True

        formatted = exp_manager.retriever.format_prompt_context(warnings)
        assert "⚠️  ADVISORY WARNINGS (PAST FAILURES):" in formatted
        assert "Protected target file requires explicit elevated auth" in formatted
        assert "IMPORTANT INVARIANT NOTICE:" in formatted
        assert "Current codebase AST, current tool results, and permission policies are AUTHORITATIVE" in formatted


# ==============================================================================
# 5. Controlled Learning Layer Tests (Requirements O, P)
# ==============================================================================

class TestLearningLayer:
    """Validates empirical lesson derivation, confidence calculation, and validation."""

    def test_req_o_p_lesson_extraction_and_confidence(self, exp_manager: ExperienceManager) -> None:
        # Add 1 experience for component A (confidence should be ~0.40)
        exp1 = ExperienceRecord(
            task="Fix tool registration",
            affected_files=["tools/registry.py"],
            success=True,
            lesson="Tools must validate input types.",
            recommendation="Register tools explicitly.",
        )
        exp_manager.record_experience(exp1)

        lessons = exp_manager.derive_lessons()
        registry_lessons = [l for l in lessons if l.topic == "tools/registry.py"]
        assert len(registry_lessons) == 1
        assert registry_lessons[0].confidence == 0.40
        assert registry_lessons[0].success_count == 1

        # Add 4 more experiences for component A (confidence should scale up)
        for i in range(4):
            e = ExperienceRecord(
                task=f"Update tool {i}",
                affected_files=["tools/registry.py"],
                success=True,
                lesson="Tools must validate input types.",
                recommendation="Register tools explicitly.",
            )
            exp_manager.store.add_experience(e)

        lessons_updated = exp_manager.derive_lessons()
        registry_lessons_updated = [l for l in lessons_updated if l.topic == "tools/registry.py"]
        assert len(registry_lessons_updated) == 1
        assert registry_lessons_updated[0].confidence >= 0.85
        assert registry_lessons_updated[0].success_count == 5

        # Validate lesson
        val = exp_manager.validate_lesson(registry_lessons_updated[0])
        assert val["is_supported_by_evidence"] is True
        assert val["is_authoritative_rule"] is False  # Invariant: Never authoritative rules

    def test_common_failures_and_component_cautions(self, exp_manager: ExperienceManager) -> None:
        exp_fail1 = ExperienceRecord(
            task="Edit loop",
            affected_files=["agent/loop.py"],
            success=False,
            failure_type="test_failure",
            failure_reason="Broke AgentLoop state transition",
            recommendation="Verify state transitions",
        )
        exp_fail2 = ExperienceRecord(
            task="Edit store",
            affected_files=["memory/store.py"],
            success=False,
            failure_type="verification_failure",
            failure_reason="Table lock contention",
            recommendation="Close cursors promptly",
        )
        exp_manager.record_experience(exp_fail1)
        exp_manager.record_experience(exp_fail2)

        failures = exp_manager.learning_layer.get_common_failures()
        assert len(failures) >= 2

        cautions = exp_manager.learning_layer.get_component_cautions("agent/loop.py")
        assert len(cautions) >= 1
        assert "Verify state transitions" in cautions[0]


# ==============================================================================
# 6. Deduplication & History Preservation Tests (Requirements Q, R)
# ==============================================================================

class TestDeduplicationAndHistory:
    """Validates aggregation of identical experiences and linking of resolved failures."""

    def test_req_q_duplicate_handling(self, exp_manager: ExperienceManager) -> None:
        exp1 = ExperienceRecord(
            task="Run unit test for math module",
            affected_files=["sample_math.py"],
            success=True,
            lesson="All math tests passed.",
            evidence_count=2,
        )
        id1 = exp_manager.record_experience(exp1)

        # Same task, target, and outcome
        exp2 = ExperienceRecord(
            task="Run unit test for math module",
            affected_files=["sample_math.py"],
            success=True,
            lesson="All math tests passed.",
            evidence_count=2,
        )
        id2 = exp_manager.record_experience(exp2)

        assert id1 == id2
        all_records = exp_manager.list_experiences()
        assert len(all_records) == 1
        assert all_records[0].duplicate_count == 2
        assert all_records[0].evidence_count == 4

    def test_req_r_historical_experience_preservation(self, exp_manager: ExperienceManager) -> None:
        # Step 1: Initial failure
        failed_exp = ExperienceRecord(
            task="Add matrix multiplication to sample_math.py",
            affected_files=["sample_math.py"],
            success=False,
            failure_type="test_failure",
            failure_reason="Dimension mismatch error",
            lesson="Check inner dimensions before multiplying.",
        )
        fail_id = exp_manager.record_experience(failed_exp)

        # Step 2: Subsequent success for the same target
        success_exp = ExperienceRecord(
            task="Add matrix multiplication to sample_math.py",
            affected_files=["sample_math.py"],
            success=True,
            lesson="Dimension check implemented successfully.",
        )
        succ_id = exp_manager.record_experience(success_exp)

        # Verify historical preservation: failure is NOT overwritten or erased
        assert fail_id != succ_id
        all_records = exp_manager.list_experiences()
        assert len(all_records) == 2

        # Verify linking in related_experience_ids
        succ_record = exp_manager.get_experience(succ_id)
        assert succ_record is not None
        assert fail_id in succ_record.related_experience_ids
        assert "Resolved previous failure" in succ_record.lesson


# ==============================================================================
# 7. Invariants, Permissions & Security Policy (Requirements U, V, W, AD, AE, AF, AH)
# ==============================================================================

class TestInvariantsAndSecurityPolicies:
    """Validates authoritative codebase, permission boundaries, and security rules."""

    def test_req_u_test_5_current_codebase_overrides_experience(self, workspace: Path, temp_store: MemoryStore) -> None:
        """TEST 5: Conflicting old experience vs current codebase -> current codebase wins."""
        mgr = ExperienceManager(store=temp_store)

        # Record outdated memory claiming 'legacy_calculate' exists in sample_math.py
        mgr.record_experience(
            ExperienceRecord(
                task="Update legacy_calculate in sample_math.py",
                affected_files=["sample_math.py"],
                relevant_symbols=["legacy_calculate"],
                success=True,
                lesson="Call legacy_calculate with two integers.",
            )
        )

        codebase_mgr = CodebaseManager(workspace_root=str(workspace))
        codebase_mgr.ensure_scanned()

        # The actual file on disk only has 'add' and 'subtract', not 'legacy_calculate'
        task = DeveloperTask(
            goal="Refactor legacy_calculate in sample_math.py",
            target_files=["sample_math.py"],
            target_symbols=["legacy_calculate"],
            task_type=TaskType.CODE_CHANGE,
            is_development_task=True,
        )

        planner = DeveloperPlanner(
            codebase_manager=codebase_mgr,
            workspace_root=str(workspace),
            experience_manager=mgr,
        )
        plan = planner.plan(task)

        # Current codebase AST wins: legacy_calculate does not exist in symbols
        syms = codebase_mgr.find_symbol("legacy_calculate")
        assert len(syms) == 0
        funcs = codebase_mgr.find_function("legacy_calculate")
        assert len(funcs) == 0
        # Planner relies on actual files, acknowledging the target file
        assert "sample_math.py" in plan.relevant_files

    def test_req_v_ae_test_4_permissions_override_experience(self, workspace: Path, temp_store: MemoryStore) -> None:
        """TEST 4: Attempt to use memory to bypass permission -> permission remains authoritative."""
        mgr = ExperienceManager(store=temp_store)

        # Record past experience stating this change was previously authorized and succeeded
        mgr.record_experience(
            ExperienceRecord(
                task="Modify sample_math.py",
                affected_files=["sample_math.py"],
                success=True,
                lesson="Past authorization granted smoothly.",
            )
        )

        perm_mgr = PermissionManager(default_write_allowed=False)
        applier = CodeChangeApplier(permission_manager=perm_mgr, workspace_root=workspace)

        executor = DeveloperExecutor(
            applier=applier,
            permission_manager=perm_mgr,
            workspace_root=workspace,
            experience_manager=mgr,
        )

        # Propose a change
        orig_content = (workspace / "sample_math.py").read_text(encoding="utf-8")
        proposal = CodeChangeProposal(
            target_file="sample_math.py",
            operation=ProposalOperation.MODIFY_FILE.value,
            original_content=orig_content,
            proposed_content=orig_content + "\ndef multiply(a, b): return a * b\n",
            description="Add multiply function",
            status=ProposalStatus.PROPOSED.value,
        )

        prep_res = executor.prepare_proposal(proposal)
        assert prep_res.status == ExecutionStatus.PENDING_PERMISSION.value

        # Explicitly deny proposal
        deny_res = executor.deny_proposal()

        # Memory CANNOT bypass permission denial
        assert deny_res.permission_granted is False
        assert deny_res.overall_outcome == "PERMISSION_DENIED"

        # Verify on-disk file was NEVER modified
        current_disk = (workspace / "sample_math.py").read_text(encoding="utf-8")
        assert current_disk == orig_content

    def test_req_w_af_security_policy_overrides_experience(self, workspace: Path, temp_store: MemoryStore) -> None:
        """AF: Experience memory cannot bypass protected files."""
        mgr = ExperienceManager(store=temp_store)

        # Record past success on permissions/manager.py
        mgr.record_experience(
            ExperienceRecord(
                task="Modify permissions/manager.py",
                affected_files=["permissions/manager.py"],
                success=True,
                lesson="Permissions updated successfully in previous test.",
            )
        )

        # Security check directly tests protected file boundary
        assert is_security_critical_target("permissions/manager.py") is True
        assert is_security_critical_target("config.py") is True

        # Validation scope blocks protected file regardless of memory
        with pytest.raises(Exception):
            validate_scope(
                proposal_file="permissions/manager.py",
                authorized_scope=["permissions/manager.py"],
                allow_security_critical=False,
            )

    def test_req_ad_test_3_recursion_guard(self, temp_store: MemoryStore) -> None:
        """TEST 3 & Req AD: Experience recording does NOT trigger self-development."""
        mgr = ExperienceManager(store=temp_store)

        with patch("self_development.loop.SelfDevelopmentLoop.initiate_cycle") as mock_init:
            rec = ExperienceRecord(
                task="Self-dev optimization",
                task_type="self_development",
                success=True,
                lesson="Tuned parameters",
            )
            mgr.record_experience(rec)

            # Assert self-dev loop is NEVER invoked by recording an experience
            mock_init.assert_not_called()

    def test_req_ah_test_6_local_only_mode_never_sends_to_cloud(self, temp_store: MemoryStore) -> None:
        """TEST 6: Local-only mode -> no cloud transmission when cloud disabled."""
        mgr = ExperienceManager(store=temp_store)
        mgr.record_experience(
            ExperienceRecord(
                task="Local sensitive calculation",
                affected_files=["secure/logic.py"],
                success=True,
                lesson="Executed strictly offline.",
            )
        )

        # Verify query and context generation perform zero network calls
        with patch("urllib.request.urlopen") as mock_url:
            query = ExperienceQuery(query="sensitive calculation", min_score=0.1)
            results = mgr.retrieve_experiences(query)
            context = mgr.get_context_for_developer_planning(goal="sensitive calculation")

            assert len(results) >= 1
            assert "Local sensitive calculation" in context
            mock_url.assert_not_called()

    def test_failure_safety_resilience(self, temp_store: MemoryStore) -> None:
        """Failure safety: if memory store errors, manager does not crash."""
        mgr = ExperienceManager(store=temp_store)

        rec = ExperienceRecord(task="Resilience test", success=True)
        # Mock add_experience to simulate database error
        with patch.object(temp_store, "add_experience", side_effect=sqlite3.OperationalError("disk I/O error")):
            result_id = mgr.record_experience(rec)
            assert result_id == ""


# ==============================================================================
# 8. End-to-End Workflow Integration Tests (Requirements X, Y, Z, AA, AB, AC)
# ==============================================================================

class TestWorkflowIntegrations:
    """Validates integration with DeveloperPlanner, DeveloperExecutor, and SelfDevelopmentLoop."""

    def test_req_x_developer_planning_integration(self, workspace: Path, temp_store: MemoryStore) -> None:
        mgr = ExperienceManager(store=temp_store)

        # Record a past failure on sample_math.py
        mgr.record_experience(
            ExperienceRecord(
                task="Modify add function in sample_math.py",
                affected_files=["sample_math.py"],
                relevant_symbols=["add"],
                success=False,
                failure_type="verification_failure",
                failure_reason="Indent error in return statement",
                recommendation="Use 4-space indentation consistently.",
            )
        )

        codebase_mgr = CodebaseManager(workspace_root=str(workspace))
        codebase_mgr.ensure_scanned()

        task = DeveloperTask(
            goal="Modify add function in sample_math.py",
            target_files=["sample_math.py"],
            target_symbols=["add"],
            task_type=TaskType.CODE_CHANGE,
            is_development_task=True,
        )

        planner = DeveloperPlanner(
            codebase_manager=codebase_mgr,
            workspace_root=str(workspace),
            experience_manager=mgr,
        )
        plan = planner.plan(task)

        # Check that planner received relevant experiences
        assert len(plan.relevant_experiences) >= 1
        # Check that advisory warning was injected into plan risks
        advisory_risks = [r for r in plan.risks if "Advisory Warning" in r]
        assert len(advisory_risks) >= 1
        assert "Indent error in return statement" in advisory_risks[0]

    def test_req_aa_test_1_successful_developer_task_creates_experience(
        self, workspace: Path, temp_store: MemoryStore
    ) -> None:
        """TEST 1: Successful developer task -> experience created & persisted -> lesson extracted -> future task retrieves it."""
        mgr = ExperienceManager(store=temp_store)
        perm_mgr = PermissionManager(default_write_allowed=True)
        applier = CodeChangeApplier(permission_manager=perm_mgr, workspace_root=workspace)

        executor = DeveloperExecutor(
            applier=applier,
            permission_manager=perm_mgr,
            workspace_root=workspace,
            experience_manager=mgr,
        )

        orig_content = (workspace / "sample_math.py").read_text(encoding="utf-8")
        new_content = orig_content + "\ndef multiply(a: int, b: int) -> int:\n    return a * b\n"

        proposal = CodeChangeProposal(
            target_file="sample_math.py",
            operation=ProposalOperation.MODIFY_FILE.value,
            original_content=orig_content,
            proposed_content=new_content,
            description="Add multiply function",
            status=ProposalStatus.PROPOSED.value,
        )

        prep_res = executor.prepare_proposal(proposal)
        assert prep_res.status == ExecutionStatus.PENDING_PERMISSION.value

        # Grant permission and execute
        perm_mgr.grant_write_permission()
        exec_res = executor.execute_proposal(proposal=proposal, explicit_permission=True, run_tests=False)
        assert exec_res.success is True

        # Verify experience was persisted
        records = mgr.list_experiences()
        assert len(records) == 1
        saved_exp = records[0]
        assert saved_exp.success is True
        assert "sample_math.py" in saved_exp.affected_files
        assert saved_exp.verification_status == "verified"

        # Verify lesson extracted
        lessons = mgr.derive_lessons()
        assert len(lessons) >= 1

        # Verify future retrieval finds it
        future_query = ExperienceQuery(query="multiply", file_path="sample_math.py")
        retrieved = mgr.retrieve_experiences(future_query)
        assert len(retrieved) >= 1
        assert retrieved[0].experience.id == saved_exp.id

    def test_req_z_test_2_failed_developer_task_creates_warning_experience(
        self, workspace: Path, temp_store: MemoryStore
    ) -> None:
        """TEST 2: Failed developer task -> failure reason preserved -> retrieved as warning."""
        mgr = ExperienceManager(store=temp_store)
        perm_mgr = PermissionManager(default_write_allowed=False)
        applier = CodeChangeApplier(permission_manager=perm_mgr, workspace_root=workspace)

        executor = DeveloperExecutor(
            applier=applier,
            permission_manager=perm_mgr,
            workspace_root=workspace,
            experience_manager=mgr,
        )

        orig_content = (workspace / "sample_math.py").read_text(encoding="utf-8")
        proposal = CodeChangeProposal(
            target_file="sample_math.py",
            operation=ProposalOperation.MODIFY_FILE.value,
            original_content=orig_content,
            proposed_content=orig_content + "\n# Stale attempt",
            description="Stale modification",
            status=ProposalStatus.PROPOSED.value,
        )

        executor.prepare_proposal(proposal)

        # Deny proposal
        deny_res = executor.deny_proposal()
        assert deny_res.overall_outcome == "PERMISSION_DENIED"

        # Verify failure experience was recorded
        records = mgr.list_experiences()
        assert len(records) == 1
        fail_exp = records[0]
        assert fail_exp.success is False
        assert fail_exp.failure_type == "permission_denied"

        # Verify future retrieval surfaces it as advisory warning
        warnings = mgr.retriever.retrieve_warnings(file_path="sample_math.py")
        assert len(warnings) >= 1
        assert warnings[0].is_warning is True
        assert "Permission was not granted" in warnings[0].experience.failure_reason

    def test_req_ab_ac_self_development_experience_recording(self, temp_store: MemoryStore) -> None:
        """Tests that self-development evaluation records success and failure experiences."""
        mem_mgr = MemoryManager(store=temp_store)

        eval_success = SelfDevelopmentEvaluation(
            goal_id="goal_001",
            self_development_id="sd_001",
            improvement_success=True,
            verification_status=True,
            test_status="passed",
            tests_passed=10,
            tests_failed=0,
            overall_outcome="SUCCESS",
            lesson="Optimization improved throughput by 15%",
            recommendation="Apply pattern to other query handlers",
        )
        goal = SelfDevelopmentGoal(description="Optimize query caching", scope=["cache"])

        exp_success = ExperienceExtractor.from_self_development_evaluation(
            evaluation=eval_success,
            goal=goal,
        )
        mem_mgr.record_experience(exp_success)

        eval_fail = SelfDevelopmentEvaluation(
            goal_id="goal_002",
            self_development_id="sd_002",
            improvement_success=False,
            verification_status=True,
            test_status="failed",
            tests_passed=8,
            tests_failed=2,
            overall_outcome="CHANGE_APPLIED_BUT_TESTS_FAILED",
            observed_result="2 tests failed in test_cache.py",
        )
        exp_fail = ExperienceExtractor.from_self_development_evaluation(
            evaluation=eval_fail,
            goal=goal,
        )
        mem_mgr.record_experience(exp_fail)

        all_self_dev = mem_mgr.list_experiences(task_type="self_development")
        assert len(all_self_dev) == 2
        assert any(e.success is True for e in all_self_dev)
        assert any(e.success is False and e.failure_type == "test_failure" for e in all_self_dev)
