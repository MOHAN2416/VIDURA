import difflib
import logging
from pathlib import Path
from typing import Any

from developer.models import (
    CodeChangeProposal,
    CodeChangeResult,
    VerificationResult,
    ExecutionStage,
    ExecutionStatus,
    DeveloperExecutionResult,
    ApplicationStatus,
)
from developer.applier import CodeChangeApplier
from permissions.manager import PermissionManager
from planning.models import DeveloperPlan
from developer.workflow import (
    DeveloperWorkflowStateMachine,
    DeveloperWorkflowState,
    MultiStageOutcome,
)

logger = logging.getLogger("VIDURA.developer.executor")


class DeveloperExecutor:
    """Orchestrates validation, permission checking, application, and verification of code change proposals.

    CRITICAL ARCHITECTURAL BOUNDARY:
    The executor is strictly an orchestrator. It does NOT implement its own filesystem-writing logic.
    All disk modifications and post-change verifications are strictly delegated to Phase 7 components
    (CodeChangeApplier and CodeChangeVerifier).
    """

    def __init__(
        self,
        applier: CodeChangeApplier,
        permission_manager: PermissionManager | None = None,
        workspace_root: str | Path | None = None,
        test_planner: Any = None,
        test_runner: Any = None,
        codebase_manager: Any = None,
        experience_manager: Any = None,
    ) -> None:
        self.applier = applier
        self.permission_manager = permission_manager or getattr(applier, "permission_manager", None)
        self.workspace_root = Path(workspace_root).resolve() if workspace_root else getattr(applier, "workspace_root", None)
        self.codebase_manager = codebase_manager
        self.experience_manager = experience_manager
        from developer.testing import TestPlanner, TestRunner
        self.test_planner = test_planner or TestPlanner(workspace_root=self.workspace_root, codebase_manager=self.codebase_manager)
        self.test_runner = test_runner or TestRunner(workspace_root=self.workspace_root, permission_manager=self.permission_manager)
        self.workflow_sm = DeveloperWorkflowStateMachine()
        self._last_execution_result: DeveloperExecutionResult | None = None

    def _record_execution_experience(
        self,
        result: DeveloperExecutionResult,
        plan: DeveloperPlan | None = None,
    ) -> None:
        """Records a completed workflow outcome as an experience record with failure safety."""
        if not self.experience_manager:
            return
        try:
            from experience.extractor import ExperienceExtractor
            exp = ExperienceExtractor.from_developer_execution(
                execution_result=result,
                task=getattr(plan, "original_task", None) if plan else None,
                plan=plan,
                provider=result.provider or "",
                model=result.model or "",
                fallback_used=result.fallback_used,
                cloud_request_id=result.cloud_request_id,
            )
            self.experience_manager.record_experience(exp)
            logger.info(f"Recorded execution experience for proposal '{result.proposal_id}' (success={exp.success})")
        except Exception as err:
            logger.warning(f"Failed to record execution experience (failure safety engaged): {err}")

    def get_pending_proposal(self) -> CodeChangeProposal | None:
        """Returns the currently active pending proposal from the applier."""
        return self.applier.get_pending_proposal()

    def get_last_execution_result(self) -> DeveloperExecutionResult | None:
        """Returns the most recent execution result, if any."""
        return self._last_execution_result

    def prepare_proposal(
        self,
        proposal: CodeChangeProposal,
        plan: DeveloperPlan | None = None,
    ) -> DeveloperExecutionResult:
        """Validates and registers a proposal for user review and approval."""
        try:
            logger.info(f"Preparing proposal for execution review: target='{getattr(proposal, 'target_file', '')}', op='{getattr(proposal, 'operation', '')}'")

            prov = getattr(proposal, "provider", None) if proposal else None
            mod = getattr(proposal, "model", None) if proposal else None

            if not proposal or not isinstance(proposal, CodeChangeProposal) or not proposal.is_valid:
                err = proposal.validation_error if proposal else "Invalid proposal object."
                self.workflow_sm.reset()
                try:
                    self.workflow_sm.transition_to(DeveloperWorkflowState.FAILED, reason=err)
                except Exception:
                    pass
                result = DeveloperExecutionResult(
                    success=False,
                    status=ExecutionStatus.PROPOSAL_INVALID.value,
                    stage=ExecutionStage.PROPOSAL_INVALID.value,
                    target_file=proposal.target_file if proposal else "",
                    operation=proposal.operation if proposal else "",
                    proposal_id=getattr(proposal, "proposal_id", None),
                    proposal=proposal,
                    overall_outcome=MultiStageOutcome.FAILED.value,
                    summary=f"Proposal is invalid: {err}",
                    error=err,
                    provider=prov,
                    model=mod,
                )
                self._last_execution_result = result
                return result

            # Security check: validate workspace boundary and protected target
            try:
                if hasattr(self.applier, "_validate_workspace_path"):
                    self.applier._validate_workspace_path(proposal.target_file)
                if hasattr(self.applier, "_is_protected_target") and self.applier._is_protected_target(proposal.target_file):
                    raise ValueError(f"Protected target '{proposal.target_file}' cannot be modified.")
            except Exception as err:
                logger.warning(f"Target validation failed for proposal '{proposal.target_file}': {err}")
                self.workflow_sm.reset()
                try:
                    self.workflow_sm.transition_to(DeveloperWorkflowState.FAILED, reason=str(err))
                except Exception:
                    pass
                result = DeveloperExecutionResult(
                    success=False,
                    status=ExecutionStatus.PROPOSAL_INVALID.value,
                    stage=ExecutionStage.PROPOSAL_INVALID.value,
                    target_file=proposal.target_file,
                    operation=proposal.operation,
                    proposal_id=getattr(proposal, "proposal_id", None),
                    proposal=proposal,
                    overall_outcome=MultiStageOutcome.FAILED.value,
                    summary=f"Target validation failed: {err}",
                    error=str(err),
                    provider=prov,
                    model=mod,
                )
                self._last_execution_result = result
                return result

            # Register proposal in applier (auto-revokes write permission)
            self.applier.set_pending_proposal(proposal)

            self.workflow_sm.reset()
            self.workflow_sm.transition_to(DeveloperWorkflowState.PROPOSAL_PENDING)
            self.workflow_sm.transition_to(DeveloperWorkflowState.PERMISSION_PENDING)

            display = self.format_proposal_display(proposal, plan)
            result = DeveloperExecutionResult(
                success=False,
                status=ExecutionStatus.PENDING_PERMISSION.value,
                stage=ExecutionStage.PENDING_PERMISSION.value,
                target_file=proposal.target_file,
                operation=proposal.operation,
                proposal_id=proposal.proposal_id,
                proposal=proposal,
                permission_granted=False,
                overall_outcome="PENDING_APPROVAL",
                summary=display,
                error=None,
                provider=prov,
                model=mod,
            )
            self._last_execution_result = result
            return result
        except Exception as exc:
            logger.exception(f"Unexpected error in prepare_proposal: {exc}")
            safe_err = f"Unexpected error preparing proposal: {type(exc).__name__}"
            self.workflow_sm.reset()
            try:
                self.workflow_sm.transition_to(DeveloperWorkflowState.FAILED, reason=safe_err)
            except Exception:
                pass
            result = DeveloperExecutionResult(
                success=False,
                status=ExecutionStatus.UNEXPECTED_ERROR.value,
                stage=ExecutionStage.FAILED.value,
                target_file=proposal.target_file if proposal else "",
                operation=proposal.operation if proposal else "",
                proposal_id=getattr(proposal, "proposal_id", None),
                proposal=proposal,
                overall_outcome=MultiStageOutcome.FAILED.value,
                summary=f"An unexpected error occurred: {safe_err}",
                error=safe_err,
                provider=getattr(proposal, "provider", None) if proposal else None,
                model=getattr(proposal, "model", None) if proposal else None,
            )
            self._last_execution_result = result
            return result

    def execute_proposal(
        self,
        proposal: CodeChangeProposal | None = None,
        explicit_permission: bool = False,
        run_tests: bool = True,
        plan: DeveloperPlan | None = None,
    ) -> DeveloperExecutionResult:
        """Applies an approved proposal strictly through the Phase 7 application and verification system."""
        try:
            trusted_pending = self.applier.get_pending_proposal()
            prov = getattr(trusted_pending, "provider", None) or (getattr(proposal, "provider", None) if proposal else None)
            mod = getattr(trusted_pending, "model", None) or (getattr(proposal, "model", None) if proposal else None)

            # Duplicate approval protection:
            # If the change was already successfully applied and caller re-submits or approves again:
            cached = self._last_execution_result
            if cached and cached.success and cached.proposal_id:
                is_duplicate = False
                if proposal and getattr(proposal, "proposal_id", None) == cached.proposal_id:
                    is_duplicate = True
                elif not proposal and not trusted_pending:
                    is_duplicate = True
                elif trusted_pending and getattr(trusted_pending, "status", "") == "applied" and trusted_pending.proposal_id == cached.proposal_id:
                    is_duplicate = True

                if is_duplicate:
                    logger.info(f"Duplicate execution requested for already applied proposal '{cached.proposal_id}'. Returning cached result.")
                    dup_summary = f"Change already applied. Returning previous execution result without re-executing.\n\n{cached.summary}"
                    return DeveloperExecutionResult(
                        success=True,
                        status=cached.status,
                        stage=cached.stage,
                        target_file=cached.target_file,
                        operation=cached.operation,
                        proposal_id=cached.proposal_id,
                        proposal=cached.proposal,
                        application_result=cached.application_result,
                        verification_result=cached.verification_result,
                        permission_granted=cached.permission_granted,
                        test_plan=cached.test_plan,
                        test_result=cached.test_result,
                        requires_plan_update=False,
                        overall_outcome=cached.overall_outcome or MultiStageOutcome.COMPLETED.value,
                        summary=dup_summary,
                        error=None,
                        provider=cached.provider,
                        model=cached.model,
                    )

            if not trusted_pending:
                logger.warning("Execution rejected: No active pending proposal found.")
                try:
                    if self.workflow_sm.can_transition_to(DeveloperWorkflowState.FAILED):
                        self.workflow_sm.transition_to(DeveloperWorkflowState.FAILED, reason="No pending proposal")
                except Exception:
                    pass
                result = DeveloperExecutionResult(
                    success=False,
                    status=ExecutionStatus.PROPOSAL_INVALID.value,
                    stage=ExecutionStage.FAILED.value,
                    target_file=proposal.target_file if proposal else "",
                    operation=proposal.operation if proposal else "",
                    proposal_id=getattr(proposal, "proposal_id", None),
                    proposal=proposal,
                    overall_outcome=MultiStageOutcome.FAILED.value,
                    summary="No active pending code change proposal found. Generate a proposal first.",
                    error="No pending proposal.",
                    provider=prov,
                    model=mod,
                )
                self._last_execution_result = result
                return result

            # Multi-proposal & immutability protection: verify caller-supplied proposal matches active pending proposal
            if proposal is not None:
                if getattr(proposal, "proposal_id", None) != trusted_pending.proposal_id:
                    err_msg = (
                        f"Proposal mismatch: Attempted to apply proposal '{getattr(proposal, 'proposal_id', None)}' "
                        f"but active pending proposal is '{trusted_pending.proposal_id}'."
                    )
                    logger.warning(err_msg)
                    try:
                        if self.workflow_sm.can_transition_to(DeveloperWorkflowState.FAILED):
                            self.workflow_sm.transition_to(DeveloperWorkflowState.FAILED, reason=err_msg)
                    except Exception:
                        pass
                    result = DeveloperExecutionResult(
                        success=False,
                        status=ExecutionStatus.PROPOSAL_INVALID.value,
                        stage=ExecutionStage.FAILED.value,
                        target_file=trusted_pending.target_file,
                        operation=trusted_pending.operation,
                        proposal_id=trusted_pending.proposal_id,
                        proposal=trusted_pending,
                        overall_outcome=MultiStageOutcome.FAILED.value,
                        summary=err_msg,
                        error=err_msg,
                        provider=prov,
                        model=mod,
                    )
                    self._last_execution_result = result
                    return result

                if (
                    proposal.target_file != trusted_pending.target_file
                    or proposal.operation != trusted_pending.operation
                    or proposal.proposed_content != trusted_pending.proposed_content
                    or proposal.original_content != trusted_pending.original_content
                ):
                    err_msg = "Proposal immutability violation: Proposal content was mutated after generation. Please regenerate proposal."
                    logger.warning(err_msg)
                    try:
                        if self.workflow_sm.can_transition_to(DeveloperWorkflowState.FAILED):
                            self.workflow_sm.transition_to(DeveloperWorkflowState.FAILED, reason=err_msg)
                    except Exception:
                        pass
                    result = DeveloperExecutionResult(
                        success=False,
                        status=ExecutionStatus.PROPOSAL_INVALID.value,
                        stage=ExecutionStage.FAILED.value,
                        target_file=trusted_pending.target_file,
                        operation=trusted_pending.operation,
                        proposal_id=trusted_pending.proposal_id,
                        proposal=trusted_pending,
                        overall_outcome=MultiStageOutcome.FAILED.value,
                        summary=err_msg,
                        error=err_msg,
                        provider=prov,
                        model=mod,
                    )
                    self._last_execution_result = result
                    return result

            # Scope control check: ensure proposal target is in plan.relevant_files
            if plan and plan.relevant_files:
                def _norm(p: str) -> str:
                    return Path(p).as_posix().lstrip("./")

                target_norm = _norm(trusted_pending.target_file)
                rel_files_norm = [_norm(f) for f in plan.relevant_files]

                in_scope = False
                for rf in rel_files_norm:
                    if target_norm == rf or target_norm.endswith(f"/{rf}") or rf.endswith(f"/{target_norm}"):
                        in_scope = True
                        break
                    try:
                        if Path(target_norm).name == Path(rf).name:
                            in_scope = True
                            break
                    except Exception:
                        pass

                if not in_scope:
                    err_msg = f"Scope violation: Proposal target '{trusted_pending.target_file}' is not in plan's relevant files ({plan.relevant_files}). Execution halted."
                    logger.warning(err_msg)
                    try:
                        if self.workflow_sm.can_transition_to(DeveloperWorkflowState.FAILED):
                            self.workflow_sm.transition_to(DeveloperWorkflowState.FAILED, reason=err_msg)
                    except Exception:
                        pass
                    result = DeveloperExecutionResult(
                        success=False,
                        status=ExecutionStatus.APPLICATION_FAILED.value,
                        stage=ExecutionStage.FAILED.value,
                        target_file=trusted_pending.target_file,
                        operation=trusted_pending.operation,
                        proposal_id=trusted_pending.proposal_id,
                        proposal=trusted_pending,
                        requires_plan_update=True,
                        overall_outcome=MultiStageOutcome.FAILED.value,
                        summary=err_msg,
                        error=err_msg,
                        provider=prov,
                        model=mod,
                    )
                    self._last_execution_result = result
                    return result

            # Ensure workflow state machine is in PERMISSION_PENDING
            if self.workflow_sm.current_state == DeveloperWorkflowState.IDLE:
                self.workflow_sm.transition_to(DeveloperWorkflowState.PERMISSION_PENDING)

            # Step 1: Permission check (Default: DENY)
            permission_authorized = False
            if self.permission_manager:
                permission_authorized = self.permission_manager.is_write_allowed(trusted_pending.target_file, trusted_pending.operation)
            else:
                permission_authorized = explicit_permission

            if not permission_authorized:
                logger.warning(f"Permission denied for proposal '{trusted_pending.proposal_id}'. Halting execution.")
                try:
                    if self.workflow_sm.can_transition_to(DeveloperWorkflowState.CANCELLED):
                        self.workflow_sm.transition_to(DeveloperWorkflowState.CANCELLED, reason="Permission denied")
                except Exception:
                    pass
                result = DeveloperExecutionResult(
                    success=False,
                    status=ExecutionStatus.PERMISSION_DENIED.value,
                    stage=ExecutionStage.DENIED.value,
                    target_file=trusted_pending.target_file,
                    operation=trusted_pending.operation,
                    proposal_id=trusted_pending.proposal_id,
                    proposal=trusted_pending,
                    permission_granted=False,
                    overall_outcome=MultiStageOutcome.CANCELLED.value,
                    summary="Change was not applied because permission was not granted.",
                    error="Permission denied.",
                    provider=prov,
                    model=mod,
                )
                self._last_execution_result = result
                self._record_execution_experience(result, plan=plan)
                return result

            # Step 2: Transition to APPROVED -> APPLYING
            logger.info(f"Permission granted for '{trusted_pending.target_file}'. Applying change via Phase 7 applier.")
            try:
                if self.workflow_sm.can_transition_to(DeveloperWorkflowState.APPROVED):
                    self.workflow_sm.transition_to(DeveloperWorkflowState.APPROVED)
                if self.workflow_sm.can_transition_to(DeveloperWorkflowState.APPLYING):
                    self.workflow_sm.transition_to(DeveloperWorkflowState.APPLYING)
            except Exception:
                pass

            # Step 3: Apply through Phase 7 applier
            app_result = self.applier.apply_proposal(proposal=trusted_pending, explicit_permission=True)

            test_plan = None
            test_result = None
            overall_outcome = MultiStageOutcome.COMPLETED.value

            # Step 4: Map outcome to execution stage, status, and workflow state
            if app_result.status_code == ApplicationStatus.STALE_PROPOSAL.value:
                stage = ExecutionStage.STALE.value
                status = ExecutionStatus.STALE_PROPOSAL.value
                overall_outcome = "STALE_PROPOSAL"
                summary = "The proposal is stale because the file changed after the proposal was generated."
                try:
                    if self.workflow_sm.can_transition_to(DeveloperWorkflowState.STALE):
                        self.workflow_sm.transition_to(DeveloperWorkflowState.STALE, reason="File changed on disk")
                except Exception:
                    pass
            elif app_result.status_code == ApplicationStatus.VERIFICATION_FAILED.value:
                stage = ExecutionStage.VERIFICATION_FAILED.value
                status = ExecutionStatus.VERIFICATION_FAILED.value
                overall_outcome = "VERIFICATION_FAILED"
                summary = "The change could not be confirmed on disk."
                try:
                    if self.workflow_sm.can_transition_to(DeveloperWorkflowState.FAILED):
                        self.workflow_sm.transition_to(DeveloperWorkflowState.FAILED, reason="Verification failed")
                except Exception:
                    pass
            elif app_result.status_code == ApplicationStatus.PERMISSION_DENIED.value:
                stage = ExecutionStage.DENIED.value
                status = ExecutionStatus.PERMISSION_DENIED.value
                overall_outcome = MultiStageOutcome.CANCELLED.value
                summary = "Change was not applied because permission was not granted."
                try:
                    if self.workflow_sm.can_transition_to(DeveloperWorkflowState.CANCELLED):
                        self.workflow_sm.transition_to(DeveloperWorkflowState.CANCELLED, reason="Permission denied")
                except Exception:
                    pass
            elif not app_result.success:
                stage = ExecutionStage.APPLICATION_FAILED.value
                status = ExecutionStatus.APPLICATION_FAILED.value
                overall_outcome = "APPLICATION_FAILED"
                summary = "Change was not successfully applied."
                try:
                    if self.workflow_sm.can_transition_to(DeveloperWorkflowState.FAILED):
                        self.workflow_sm.transition_to(DeveloperWorkflowState.FAILED, reason="Application failed")
                except Exception:
                    pass
            else:
                # Succeeded and verified on disk
                stage = ExecutionStage.VERIFIED.value
                status = ExecutionStatus.SUCCESS.value
                summary = f"✅ Successfully applied and verified {trusted_pending.operation} for '{trusted_pending.target_file}' on disk."

                try:
                    if self.workflow_sm.can_transition_to(DeveloperWorkflowState.APPLIED):
                        self.workflow_sm.transition_to(DeveloperWorkflowState.APPLIED)
                    if self.workflow_sm.can_transition_to(DeveloperWorkflowState.VERIFYING):
                        self.workflow_sm.transition_to(DeveloperWorkflowState.VERIFYING)
                    if self.workflow_sm.can_transition_to(DeveloperWorkflowState.VERIFIED):
                        self.workflow_sm.transition_to(DeveloperWorkflowState.VERIFIED)
                except Exception:
                    pass

                if run_tests and self.test_planner:
                    try:
                        if self.workflow_sm.can_transition_to(DeveloperWorkflowState.TEST_PLANNED):
                            self.workflow_sm.transition_to(DeveloperWorkflowState.TEST_PLANNED)

                        test_plan = self.test_planner.determine_test_plan(proposal=trusted_pending, plan=plan)
                        if test_plan.testing_required and self.test_runner:
                            if self.workflow_sm.can_transition_to(DeveloperWorkflowState.TESTING):
                                self.workflow_sm.transition_to(DeveloperWorkflowState.TESTING)

                            test_result = self.test_runner.run_test_plan(test_plan)

                            if test_result.status == "failed" or test_result.tests_failed > 0 or test_result.error:
                                overall_outcome = "CHANGE_APPLIED_BUT_TESTS_FAILED"
                                if self.workflow_sm.can_transition_to(DeveloperWorkflowState.TEST_FAILED):
                                    self.workflow_sm.transition_to(DeveloperWorkflowState.TEST_FAILED)
                                if self.workflow_sm.can_transition_to(DeveloperWorkflowState.COMPLETED):
                                    self.workflow_sm.transition_to(DeveloperWorkflowState.COMPLETED)
                            else:
                                overall_outcome = MultiStageOutcome.COMPLETED.value
                                if self.workflow_sm.can_transition_to(DeveloperWorkflowState.TEST_PASSED):
                                    self.workflow_sm.transition_to(DeveloperWorkflowState.TEST_PASSED)
                                if self.workflow_sm.can_transition_to(DeveloperWorkflowState.COMPLETED):
                                    self.workflow_sm.transition_to(DeveloperWorkflowState.COMPLETED)
                        else:
                            overall_outcome = MultiStageOutcome.COMPLETED.value
                            if self.workflow_sm.can_transition_to(DeveloperWorkflowState.COMPLETED):
                                self.workflow_sm.transition_to(DeveloperWorkflowState.COMPLETED)
                    except Exception as test_err:
                        logger.warning(f"Error during test planning or execution: {test_err}")
                        from developer.test_models import TestResult, TestStatus, TestStage
                        test_result = TestResult(
                            status=TestStatus.EXECUTION_ERROR.value,
                            stage=TestStage.TEST_ERROR.value,
                            error=str(test_err),
                        )
                        overall_outcome = "CHANGE_APPLIED_BUT_TESTS_FAILED"
                        try:
                            if self.workflow_sm.can_transition_to(DeveloperWorkflowState.TEST_FAILED):
                                self.workflow_sm.transition_to(DeveloperWorkflowState.TEST_FAILED)
                            if self.workflow_sm.can_transition_to(DeveloperWorkflowState.COMPLETED):
                                self.workflow_sm.transition_to(DeveloperWorkflowState.COMPLETED)
                        except Exception:
                            pass
                else:
                    overall_outcome = MultiStageOutcome.COMPLETED.value
                    try:
                        if self.workflow_sm.can_transition_to(DeveloperWorkflowState.COMPLETED):
                            self.workflow_sm.transition_to(DeveloperWorkflowState.COMPLETED)
                    except Exception:
                        pass

                if test_plan or test_result:
                    from developer.test_models import TestSummary
                    summary_obj = TestSummary(
                        target_file=trusted_pending.target_file,
                        target_symbol=trusted_pending.target_symbol,
                        applied=True,
                        verified=True,
                        test_target=test_result.target if test_result and test_result.target else (test_plan.test_files[0] if test_plan and test_plan.test_files else ""),
                        test_result=test_result.status if test_result else (test_plan.coverage_status if test_plan else "not_run"),
                        tests_passed=test_result.tests_passed if test_result else 0,
                        tests_failed=test_result.tests_failed if test_result else 0,
                        tests_skipped=test_result.tests_skipped if test_result else 0,
                        details=test_result.error if test_result and test_result.error else (test_plan.reason if test_plan else ""),
                    )
                    display_output = summary_obj.format_display()
                    summary = f"{summary}\n\n{display_output}"

            ver_result = None
            if app_result.verification_details:
                ver_result = VerificationResult(
                    success=bool(app_result.verification_details.get("success", False)),
                    target_file=str(app_result.verification_details.get("target_file", trusted_pending.target_file)),
                    operation=str(app_result.verification_details.get("operation", trusted_pending.operation)),
                    verified=bool(app_result.verification_details.get("verified", False)),
                    expected_state=str(app_result.verification_details.get("expected_state", "")),
                    actual_state=str(app_result.verification_details.get("actual_state", "")),
                    reason=str(app_result.verification_details.get("reason", "")),
                    error=app_result.verification_details.get("error"),
                )

            if (prov or mod) and "Provider:" not in summary:
                summary = f"{summary} (Provider: {prov or 'unknown'}, Model: {mod or 'unknown'})"

            result = DeveloperExecutionResult(
                success=app_result.success and app_result.verification_success,
                status=status,
                stage=stage,
                target_file=trusted_pending.target_file,
                operation=trusted_pending.operation,
                proposal_id=trusted_pending.proposal_id,
                proposal=trusted_pending,
                application_result=app_result,
                verification_result=ver_result,
                permission_granted=permission_authorized,
                test_plan=test_plan,
                test_result=test_result,
                requires_plan_update=False,
                overall_outcome=overall_outcome,
                summary=summary,
                error=app_result.error,
                provider=prov,
                model=mod,
            )
            self._last_execution_result = result
            self._record_execution_experience(result, plan=plan)
            return result

        except Exception as exc:
            logger.exception(f"Unexpected error in execute_proposal: {exc}")
            safe_err = f"Unexpected execution error: {type(exc).__name__}"
            try:
                self.workflow_sm.reset()
                self.workflow_sm.transition_to(DeveloperWorkflowState.FAILED, reason=safe_err)
            except Exception:
                pass
            target_str = ""
            op_str = ""
            p_id = None
            try:
                target_str = trusted_pending.target_file if trusted_pending else (proposal.target_file if proposal else "")
                op_str = trusted_pending.operation if trusted_pending else (proposal.operation if proposal else "")
                p_id = trusted_pending.proposal_id if trusted_pending else getattr(proposal, "proposal_id", None)
            except Exception:
                pass
            result = DeveloperExecutionResult(
                success=False,
                status=ExecutionStatus.UNEXPECTED_ERROR.value,
                stage=ExecutionStage.FAILED.value,
                target_file=target_str,
                operation=op_str,
                proposal_id=p_id,
                proposal=trusted_pending or proposal,
                requires_plan_update=False,
                overall_outcome=MultiStageOutcome.FAILED.value,
                summary=f"An unexpected error occurred during execution: {safe_err}",
                error=safe_err,
                provider=prov if 'prov' in locals() else None,
                model=mod if 'mod' in locals() else None,
            )
            self._last_execution_result = result
            return result

    def deny_proposal(self) -> DeveloperExecutionResult:
        """Explicitly denies the active pending proposal and clears pending state."""
        try:
            trusted_pending = self.applier.get_pending_proposal()
            deny_prov = getattr(trusted_pending, "provider", None) if trusted_pending else None
            deny_mod = getattr(trusted_pending, "model", None) if trusted_pending else None
            self.applier.clear_pending_proposal()
            if self.permission_manager:
                self.permission_manager.revoke_write_permission()

            try:
                if self.workflow_sm.can_transition_to(DeveloperWorkflowState.CANCELLED):
                    self.workflow_sm.transition_to(DeveloperWorkflowState.CANCELLED, reason="User denied proposal")
            except Exception:
                pass

            result = DeveloperExecutionResult(
                success=False,
                status=ExecutionStatus.PERMISSION_DENIED.value,
                stage=ExecutionStage.DENIED.value,
                target_file=trusted_pending.target_file if trusted_pending else "",
                operation=trusted_pending.operation if trusted_pending else "",
                proposal_id=trusted_pending.proposal_id if trusted_pending else None,
                proposal=trusted_pending,
                permission_granted=False,
                overall_outcome="PERMISSION_DENIED",
                summary="Change was not applied because permission was not granted.",
                error="Permission denied.",
                provider=deny_prov,
                model=deny_mod,
            )
            self._last_execution_result = result
            self._record_execution_experience(result)
            return result
        except Exception as exc:
            logger.exception(f"Unexpected error in deny_proposal: {exc}")
            safe_err = f"Unexpected error denying proposal: {type(exc).__name__}"
            result = DeveloperExecutionResult(
                success=False,
                status=ExecutionStatus.UNEXPECTED_ERROR.value,
                stage=ExecutionStage.FAILED.value,
                overall_outcome="FAILED",
                summary=f"An unexpected error occurred: {safe_err}",
                error=safe_err,
                provider=None,
                model=None,
            )
            self._last_execution_result = result
            return result

    @staticmethod
    def format_proposal_display(
        proposal: CodeChangeProposal,
        plan: DeveloperPlan | None = None,
    ) -> str:
        """Formats a concise user-facing display with unified diff and permission prompt."""
        lines = [
            "Proposed change:",
            f"- Operation: {proposal.operation}",
            f"- File: {proposal.target_file}",
            f"- Symbol: {proposal.target_symbol or 'N/A'}",
            f"- Change: {proposal.description or proposal.rationale or 'Code change'}",
        ]
        if getattr(proposal, "provider", None) or getattr(proposal, "model", None):
            prov_str = proposal.provider or "unknown"
            model_str = proposal.model or "unknown"
            lines.append(f"- Provider: {prov_str} ({model_str})")
        if plan and plan.risks:
            lines.append(f"- Risks: {', '.join(plan.risks)}")

        # Add diff if applicable
        if proposal.original_content or proposal.proposed_content:
            orig_lines = proposal.original_content.splitlines(keepends=True)
            prop_lines = proposal.proposed_content.splitlines(keepends=True)
            diff = "".join(difflib.unified_diff(
                orig_lines,
                prop_lines,
                fromfile=f"a/{proposal.target_file}",
                tofile=f"b/{proposal.target_file}",
                n=3,
            ))
            if diff:
                lines.append("")
                lines.append("```diff")
                lines.append(diff.strip())
                lines.append("```")

        lines.append("")
        lines.append("Permission required before applying. Reply 'apply' or '/approve' to proceed, or 'deny' or '/deny' to cancel.")
        return "\n".join(lines)
