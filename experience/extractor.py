"""VIDURA Experience Extractor (Phase 11).

Converts authoritative execution, verification, testing, planning, and routing
results into sanitized, structured ExperienceRecord instances.

CRITICAL INVARIANTS:
1. NEVER infer success from an LLM's natural-language response.
2. Rely strictly on physical verification, applier status, permission checks,
   and test runner exit codes.
3. Redact all credentials and secrets; strip chain-of-thought traces.
"""
from __future__ import annotations

import logging
import re
from typing import Any

from memory.models import ExperienceRecord
from models.security import redact_secrets

logger = logging.getLogger("VIDURA.experience.extractor")


def sanitize_text(text: str | None, max_length: int = 1000) -> str:
    """Sanitizes text by redacting secrets and stripping chain-of-thought blocks."""
    if not text:
        return ""
    cleaned = str(text)

    # Redact known credentials, tokens, keys
    cleaned = redact_secrets(cleaned)

    # Strip chain-of-thought patterns
    cot_patterns = [
        r"(?i)<thought>[\s\S]*?</thought>",
        r"(?i)```thought[\s\S]*?```",
        r"(?i)```cot[\s\S]*?```",
    ]
    for pat in cot_patterns:
        cleaned = re.sub(pat, "[REDACTED_COT]", cleaned)

    cleaned = cleaned.strip()
    if len(cleaned) > max_length:
        cleaned = cleaned[:max_length] + "..."
    return cleaned


class ExperienceExtractor:
    """Extracts authoritative, evidence-backed ExperienceRecord objects from workflow results."""

    @staticmethod
    def from_developer_execution(
        execution_result: Any,
        task: Any = None,
        plan: Any = None,
        provider: str = "",
        model: str = "",
        fallback_used: bool = False,
        cloud_request_id: str | None = None,
    ) -> ExperienceRecord:
        """Extracts an ExperienceRecord from a completed developer execution."""
        task_desc = ""
        goal_desc = ""
        task_type_str = "developer"
        affected_files: list[str] = []
        affected_components: list[str] = []
        relevant_symbols: list[str] = []
        plan_summary = ""

        # Extract from task if provided
        if task:
            task_desc = getattr(task, "requested_change", None) or getattr(task, "goal", "")
            goal_desc = getattr(task, "goal", "")
            t_type = getattr(task, "task_type", None)
            if t_type and hasattr(t_type, "value"):
                task_type_str = f"developer_{t_type.value}"
            elif t_type:
                task_type_str = f"developer_{t_type}"
            affected_files.extend(getattr(task, "target_files", []))
            relevant_symbols.extend(getattr(task, "target_symbols", []))

        # Extract from plan if provided
        if plan:
            if not goal_desc and getattr(plan, "original_task", None):
                goal_desc = getattr(plan.original_task, "goal", "")
            if getattr(plan, "relevant_files", None):
                for f in plan.relevant_files:
                    if f not in affected_files:
                        affected_files.append(f)
            if getattr(plan, "relevant_symbols", None):
                for s in plan.relevant_symbols:
                    if s not in relevant_symbols:
                        relevant_symbols.append(s)
            if getattr(plan, "affected_components", None):
                affected_components.extend(getattr(plan, "affected_components", []))
            if getattr(plan, "planned_changes", None):
                plan_summary = "; ".join(plan.planned_changes)

        # Extract from execution result
        target_file = getattr(execution_result, "target_file", "")
        if target_file and target_file not in affected_files:
            affected_files.insert(0, target_file)

        operation = getattr(execution_result, "operation", "modify_file")
        if not task_desc:
            task_desc = f"{operation} on {target_file}" if target_file else "Developer task"
        if not goal_desc:
            goal_desc = task_desc

        proposal = getattr(execution_result, "proposal", None)
        if proposal:
            sym = getattr(proposal, "target_symbol", None)
            if sym and sym not in relevant_symbols:
                relevant_symbols.append(sym)
            if not provider and getattr(proposal, "provider", None):
                provider = proposal.provider
            if not model and getattr(proposal, "model", None):
                model = proposal.model
            if not fallback_used and getattr(proposal, "fallback_used", False):
                fallback_used = True
            if not cloud_request_id and getattr(proposal, "cloud_request_id", None):
                cloud_request_id = proposal.cloud_request_id

        if not provider and getattr(execution_result, "provider", None):
            provider = execution_result.provider
        if not model and getattr(execution_result, "model", None):
            model = execution_result.model
        if not fallback_used and getattr(execution_result, "fallback_used", False):
            fallback_used = True
        if not cloud_request_id and getattr(execution_result, "cloud_request_id", None):
            cloud_request_id = execution_result.cloud_request_id

        is_cloud = (provider or "").lower() == "cloud"

        # Authoritative outcome determination
        app_res = getattr(execution_result, "application_result", None)
        ver_res = getattr(execution_result, "verification_result", None)
        test_res = getattr(execution_result, "test_result", None)

        change_applied = bool(app_res and getattr(app_res, "success", False))
        change_verified = bool(ver_res and getattr(ver_res, "success", False))

        tests_passed = 0
        tests_failed = 0
        test_status = "not_run"
        if test_res:
            tests_passed = getattr(test_res, "tests_passed", 0) or getattr(test_res, "passed_tests", 0) or 0
            tests_failed = getattr(test_res, "tests_failed", 0) or getattr(test_res, "failed_tests", 0) or 0
            exit_code = getattr(test_res, "exit_code", -1)
            if exit_code == 0 and tests_failed == 0:
                test_status = "passed"
            else:
                test_status = "failed"

        permission_granted = getattr(execution_result, "permission_granted", False)
        status = str(getattr(execution_result, "status", ""))
        overall_outcome = str(getattr(execution_result, "overall_outcome", ""))
        exec_error = getattr(execution_result, "error", None)

        # Classify success vs failure type strictly from evidence
        success = False
        failure_type: str | None = None
        failure_reason: str | None = None
        lesson = ""
        recommendation = ""
        action_summary = f"{operation} on '{target_file}'" if target_file else "Code change proposal"

        if status == "permission_denied" or not permission_granted or overall_outcome == "PERMISSION_DENIED" or overall_outcome == "CANCELLED":
            success = False
            failure_type = "permission_denied"
            failure_reason = "Permission was not granted by user or application policy."
            lesson = f"Modification to '{target_file}' requires explicit user authorization before disk write."
            recommendation = "Align with user and obtain explicit permission before applying changes."
            result_str = "Execution halted: Permission denied."

        elif status == "stale_proposal" or overall_outcome == "STALE_PROPOSAL":
            success = False
            failure_type = "stale_proposal"
            failure_reason = "Target file modified on disk after proposal generation."
            lesson = f"Proposal for '{target_file}' became stale because file contents changed concurrently."
            recommendation = "Regenerate code change proposals against the latest on-disk state."
            result_str = "Application failed: Proposal was stale."

        elif status == "verification_failed" or (change_applied and not change_verified):
            success = False
            failure_type = "verification_failure"
            failure_reason = ver_res.reason if ver_res and getattr(ver_res, "reason", None) else (exec_error or "On-disk verification failed.")
            lesson = f"Modification to '{target_file}' did not match expected on-disk verification criteria."
            recommendation = "Check exact file formatting, indentation, and ensure atomic modifications."
            result_str = f"Verification failed: {failure_reason}"

        elif not change_applied and (status == "application_failed" or overall_outcome == "APPLICATION_FAILED"):
            success = False
            failure_type = "application_failed"
            failure_reason = exec_error or "Disk modification could not be completed."
            lesson = f"Failed to apply proposed changes to '{target_file}'."
            recommendation = "Inspect file write permissions and proposal target path."
            result_str = f"Application failed: {failure_reason}"

        elif status == "proposal_invalid":
            success = False
            failure_type = "invalid_proposal"
            failure_reason = exec_error or "Proposal failed validation."
            lesson = f"Proposal for '{target_file}' violated validation constraints."
            recommendation = "Ensure proposed changes provide non-empty valid content within workspace."
            result_str = f"Invalid proposal: {failure_reason}"

        elif test_status == "failed" or tests_failed > 0:
            success = False
            failure_type = "test_failure"
            failure_reason = getattr(test_res, "error", None) or f"{tests_failed} test(s) failed."
            test_file = getattr(test_res, "target", "") or "test suite"
            lesson = f"Modification to '{target_file}' passed on-disk verification but broke tests in '{test_file}'."
            recommendation = f"Check test assertions in '{test_file}' and ensure callers remain compatible."
            result_str = f"Applied and verified on disk, but {tests_failed} tests failed."

        elif change_applied and change_verified:
            # Verified success!
            success = True
            failure_type = None
            failure_reason = None
            if test_status == "passed":
                lesson = f"Successfully modified '{target_file}' with {tests_passed} passing tests."
                recommendation = f"Maintain similar modular design and regression test coverage for {affected_components or target_file}."
                result_str = f"Applied, verified on disk, and passed {tests_passed} automated tests."
            else:
                lesson = f"Successfully applied and verified {operation} on '{target_file}'."
                recommendation = "Add automated unit tests to verify behavior under regression testing."
                result_str = f"Applied and verified {operation} on disk."

        else:
            # Fallback outcome
            success = bool(getattr(execution_result, "success", False))
            if not success:
                failure_type = "execution_failure"
                failure_reason = exec_error or "Execution ended with failure status."
                lesson = f"Execution on '{target_file}' was unsuccessful: {failure_reason}."
                recommendation = "Investigate failure reasons before retrying."
                result_str = f"Execution failed: {failure_reason}"
            else:
                lesson = f"Task completed successfully on '{target_file}'."
                recommendation = "Preserve verified changes."
                result_str = "Execution completed successfully."

        # Compute confidence based on physical evidence
        confidence = 1.0 if (success and test_status == "passed") else (0.8 if success else 0.5)

        return ExperienceRecord(
            task=sanitize_text(task_desc, max_length=300),
            attempt=1,
            action_summary=sanitize_text(action_summary, max_length=500),
            result=sanitize_text(result_str, max_length=500),
            success=success,
            lesson=sanitize_text(lesson, max_length=500),
            task_type=task_type_str,
            goal=sanitize_text(goal_desc, max_length=300),
            plan_summary=sanitize_text(plan_summary, max_length=500),
            affected_files=list(dict.fromkeys(affected_files)),
            affected_components=list(dict.fromkeys(affected_components)),
            relevant_symbols=list(dict.fromkeys(relevant_symbols)),
            provider=provider,
            model=model,
            is_cloud=is_cloud,
            verification_status="verified" if change_verified else ("failed" if change_applied else "not_run"),
            test_status=test_status,
            tests_passed=tests_passed,
            tests_failed=tests_failed,
            failure_reason=sanitize_text(failure_reason, max_length=500) if failure_reason else None,
            failure_type=failure_type,
            recommendation=sanitize_text(recommendation, max_length=500),
            confidence=confidence,
            fallback_used=fallback_used,
            duplicate_count=1,
            evidence_count=tests_passed + (1 if change_verified else 0),
            cloud_request_id=cloud_request_id,
        )

    @staticmethod
    def from_self_development_evaluation(
        evaluation: Any,
        goal: Any,
        plan: Any = None,
        proposal: Any = None,
        execution_result: Any = None,
    ) -> ExperienceRecord:
        """Extracts an ExperienceRecord from a completed self-development cycle evaluation."""
        goal_desc = getattr(goal, "description", "Self-development goal")
        target_file = getattr(proposal, "target_file", "") if proposal else ""
        operation = getattr(proposal, "operation", "modify_file") if proposal else "self_development"
        prov = getattr(evaluation, "provider_used", None) or (getattr(proposal, "provider", "") if proposal else "")
        mod = getattr(evaluation, "model_used", None) or (getattr(proposal, "model", "") if proposal else "")
        fb_used = bool(getattr(evaluation, "fallback_used", False) or (getattr(proposal, "fallback_used", False) if proposal else False))
        cloud_req_id = getattr(evaluation, "cloud_request_id", None) or (getattr(proposal, "cloud_request_id", None) if proposal else None)
        is_cloud = (prov or "").lower() == "cloud"

        affected_files: list[str] = []
        if target_file:
            affected_files.append(target_file)
        if plan and getattr(plan, "relevant_files", None):
            for f in plan.relevant_files:
                if f not in affected_files:
                    affected_files.append(f)

        relevant_symbols: list[str] = []
        if plan and getattr(plan, "relevant_symbols", None):
            relevant_symbols.extend(plan.relevant_symbols)

        affected_components: list[str] = list(affected_files)

        success = bool(getattr(evaluation, "improvement_success", False))
        overall_outcome = str(getattr(evaluation, "overall_outcome", "UNKNOWN"))
        tests_passed = int(getattr(evaluation, "tests_passed", 0) or 0)
        tests_failed = int(getattr(evaluation, "tests_failed", 0) or 0)
        test_status = str(getattr(evaluation, "test_status", "not_run"))
        change_verified = bool(getattr(evaluation, "verification_status", False))

        failure_type: str | None = None
        failure_reason: str | None = None

        if not success:
            if overall_outcome == "PERMISSION_DENIED":
                failure_type = "permission_denied"
                failure_reason = "User did not grant permission for self-development modification."
            elif overall_outcome == "VERIFICATION_FAILED":
                failure_type = "verification_failure"
                failure_reason = "Modified file failed on-disk verification."
            elif overall_outcome == "CHANGE_APPLIED_BUT_TESTS_FAILED":
                failure_type = "test_failure"
                failure_reason = f"Automated tests failed ({tests_failed} failures) after applying modification."
            elif overall_outcome == "APPLICATION_FAILED":
                failure_type = "application_failed"
                failure_reason = "Application of self-development change failed on disk."
            elif overall_outcome == "SCOPE_EXPANSION_REJECTED":
                failure_type = "scope_violation"
                failure_reason = "Proposal attempted to modify files outside authorized scope."
            elif overall_outcome == "UNSUPPORTED_OPERATION":
                failure_type = "unsupported_operation"
                failure_reason = "Operation requested is not supported in self-development."
            else:
                failure_type = "self_development_failure"
                failure_reason = getattr(evaluation, "observed_result", "Self-development cycle failed.")

        lesson = getattr(evaluation, "lesson", "") or (
            f"Self-development goal '{goal_desc}' succeeded." if success else f"Self-development goal '{goal_desc}' failed: {failure_reason}."
        )
        recommendation = getattr(evaluation, "recommendation", "")

        return ExperienceRecord(
            task=sanitize_text(f"Self-Development: {goal_desc}", max_length=300),
            attempt=1,
            action_summary=sanitize_text(f"{operation} on '{target_file}'" if target_file else f"Self-dev: {goal_desc}", max_length=500),
            result=sanitize_text(getattr(evaluation, "observed_result", overall_outcome), max_length=500),
            success=success,
            lesson=sanitize_text(lesson, max_length=500),
            task_type="self_development",
            goal=sanitize_text(goal_desc, max_length=300),
            plan_summary=sanitize_text(getattr(plan, "proposed_improvement", "") if plan else "", max_length=500),
            affected_files=affected_files,
            affected_components=affected_components,
            relevant_symbols=relevant_symbols,
            provider=prov,
            model=mod,
            is_cloud=is_cloud,
            verification_status="verified" if change_verified else "failed",
            test_status=test_status,
            tests_passed=tests_passed,
            tests_failed=tests_failed,
            failure_reason=sanitize_text(failure_reason, max_length=500) if failure_reason else None,
            failure_type=failure_type,
            recommendation=sanitize_text(recommendation, max_length=500),
            confidence=1.0 if (success and tests_passed > 0) else (0.8 if success else 0.5),
            fallback_used=fb_used,
            duplicate_count=1,
            evidence_count=tests_passed + (1 if change_verified else 0),
            cloud_request_id=cloud_req_id,
        )

    @staticmethod
    def from_provider_event(
        event_type: str,
        provider: str,
        model: str,
        error: str | None = None,
        fallback_used: bool = False,
        request_id: str | None = None,
        context_task: str = "",
    ) -> ExperienceRecord:
        """Extracts an ExperienceRecord from a cloud failure, timeout, or fallback event."""
        task_str = context_task or f"Provider interaction: {provider} ({model})"
        success = not bool(error)
        failure_type = "cloud_failure" if not success else ("local_fallback" if fallback_used else None)

        if fallback_used and not error:
            lesson = f"Provider request fell back from cloud to local model '{model}' seamlessly."
            recommendation = "Local fallback ensures offline resilience when cloud is unavailable."
            result_str = f"Fallback to {model} completed successfully."
        elif error:
            lesson = f"Provider '{provider}' with model '{model}' encountered error: {error}."
            recommendation = "Route requests locally or check cloud connection and credentials."
            result_str = f"Provider error: {error}"
        else:
            lesson = f"Provider '{provider}' with model '{model}' executed successfully."
            recommendation = "Provider configuration is functioning normally."
            result_str = "Provider request succeeded."

        return ExperienceRecord(
            task=sanitize_text(task_str, max_length=300),
            attempt=1,
            action_summary=sanitize_text(f"Invoked {provider}:{model} (event: {event_type})", max_length=500),
            result=sanitize_text(result_str, max_length=500),
            success=success,
            lesson=sanitize_text(lesson, max_length=500),
            task_type="provider_routing",
            goal=sanitize_text(task_str, max_length=300),
            provider=provider,
            model=model,
            is_cloud=(provider.lower() == "cloud"),
            failure_reason=sanitize_text(error, max_length=500) if error else None,
            failure_type=failure_type,
            recommendation=sanitize_text(recommendation, max_length=500),
            confidence=0.9 if success else 0.7,
            fallback_used=fallback_used,
            duplicate_count=1,
            cloud_request_id=request_id,
        )

    @staticmethod
    def from_planning_failure(task: Any, reason: str) -> ExperienceRecord:
        """Extracts an ExperienceRecord from an ambiguous or ungrounded planning failure."""
        goal_desc = getattr(task, "goal", "Ambiguous request")
        return ExperienceRecord(
            task=sanitize_text(goal_desc, max_length=300),
            attempt=1,
            action_summary="Plan generation attempted",
            result=sanitize_text(f"Planning halted: {reason}", max_length=500),
            success=False,
            lesson=sanitize_text(f"Request '{goal_desc}' could not be planned: {reason}.", max_length=500),
            task_type="developer_planning",
            goal=sanitize_text(goal_desc, max_length=300),
            failure_reason=sanitize_text(reason, max_length=500),
            failure_type="planning_failure",
            recommendation="Specify target files, symbols, or concrete requirements.",
            confidence=0.7,
            duplicate_count=1,
        )
