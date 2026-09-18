import logging
import uuid
from pathlib import Path

from config import Config, load_config
from codebase.manager import CodebaseManager
from memory.manager import MemoryManager
from models.base import BaseLLMProvider
from developer.generator import CodeChangeGenerator as DeveloperCodeGenerator
from developer.executor import DeveloperExecutor
from developer.applier import CodeChangeApplier
from developer.models import (
    CodeChangeProposal,
    DeveloperExecutionResult,
    ExecutionStatus,
)
from permissions.manager import PermissionManager
from self_development.models import (
    SelfDevelopmentGoal,
    SelfDevelopmentPlan,
    SelfDevelopmentCycleResult,
)
from self_development.analyzer import SelfDevelopmentAnalyzer
from self_development.planner import SelfDevelopmentPlanner
from self_development.evaluator import SelfDevelopmentEvaluator
from self_development.security import (
    validate_scope,
    validate_operation,
    SelfDevelopmentDisabledError,
    SelfDevelopmentRecursionError,
)

logger = logging.getLogger("VIDURA.self_development.loop")


class SelfDevelopmentLoop:
    """Orchestrates controlled, permission-bounded self-development cycles for VIDURA.

    CRITICAL ARCHITECTURAL BOUNDARIES:
    1. Human Approval Mandatory: The loop stops before applying any changes.
    2. Single Controlled Cycle: One goal = one cycle. No recursion, no auto-retries.
    3. Reuses Existing Developer Pipeline: Employs DeveloperCodeGenerator, CodeChangeApplier,
       CodeChangeVerifier, and TestRunner without duplicating logic.
    4. Evaluates Physical Evidence: Authoritative results come strictly from disk state and test runner.
    """

    def __init__(
        self,
        config: Config | None = None,
        model: BaseLLMProvider | None = None,
        codebase_manager: CodebaseManager | None = None,
        memory_manager: MemoryManager | None = None,
        permission_manager: PermissionManager | None = None,
        generator: DeveloperCodeGenerator | None = None,
        executor: DeveloperExecutor | None = None,
        analyzer: SelfDevelopmentAnalyzer | None = None,
        planner: SelfDevelopmentPlanner | None = None,
        evaluator: SelfDevelopmentEvaluator | None = None,
        workspace_root: str | Path | None = None,
    ) -> None:
        self.config = config or load_config()
        self.model = model
        self.workspace_root = Path(workspace_root).resolve() if workspace_root else self.config.workspace_root.resolve()
        self.codebase_manager = codebase_manager or CodebaseManager(workspace_root=str(self.workspace_root))
        self.memory_manager = memory_manager
        self.permission_manager = permission_manager or PermissionManager(default_write_allowed=False)

        # Existing Developer pipeline components
        applier = CodeChangeApplier(permission_manager=self.permission_manager, workspace_root=self.workspace_root)
        self.generator = generator or DeveloperCodeGenerator(
            workspace_root=self.workspace_root,
            model=self.model,
            codebase_manager=self.codebase_manager,
            applier=applier,
        )
        self.executor = executor or DeveloperExecutor(
            applier=applier,
            permission_manager=self.permission_manager,
            workspace_root=self.workspace_root,
            codebase_manager=self.codebase_manager,
        )

        # Dedicated Self-Development layer
        self.analyzer = analyzer or SelfDevelopmentAnalyzer(
            codebase_manager=self.codebase_manager,
            memory_manager=self.memory_manager,
            workspace_root=self.workspace_root,
        )
        self.planner = planner or SelfDevelopmentPlanner(
            analyzer=self.analyzer,
            codebase_manager=self.codebase_manager,
            workspace_root=self.workspace_root,
        )
        self.evaluator = evaluator or SelfDevelopmentEvaluator()

        # State tracking and recursion guard
        self._is_running = False
        self._active_cycle: SelfDevelopmentCycleResult | None = None

    @property
    def active_cycle(self) -> SelfDevelopmentCycleResult | None:
        """Returns the currently active cycle result, if any."""
        return self._active_cycle

    def initiate_cycle(self, goal: SelfDevelopmentGoal) -> SelfDevelopmentCycleResult:
        """Initiates a controlled self-development cycle and stops at the permission boundary."""
        # 1. Check Configuration Gate
        if not getattr(self.config, "vidura_self_development_enabled", False):
            msg = "Self-development is disabled via configuration (VIDURA_SELF_DEVELOPMENT_ENABLED=false)."
            logger.warning(msg)
            raise SelfDevelopmentDisabledError(msg)

        # 2. Check Recursion Guard
        if self._is_running:
            msg = "Recursive self-development detected: A self-development cycle cannot trigger another cycle."
            logger.error(msg)
            raise SelfDevelopmentRecursionError(msg)

        self._is_running = True
        cycle_id = f"sd_cycle_{uuid.uuid4().hex[:8]}"
        logger.info(f"Initiating self-development cycle '{cycle_id}' for goal: '{goal.description}'")

        try:
            # 3. Codebase Analysis and Experience Retrieval
            analysis = self.analyzer.analyze(goal)

            if analysis.get("requires_more_information"):
                reason = analysis.get("missing_information_reason") or "Insufficient information to ground goal."
                res = SelfDevelopmentCycleResult(
                    self_development_id=cycle_id,
                    goal=goal,
                    status="needs_information",
                    summary=f"Analysis requires more information: {reason}",
                    error=reason,
                )
                self._active_cycle = res
                return res

            # 4. Improvement Planning
            plan = self.planner.plan(goal)
            if plan.requires_more_information:
                res = SelfDevelopmentCycleResult(
                    self_development_id=cycle_id,
                    goal=goal,
                    plan=plan,
                    status="needs_information",
                    summary=f"Plan requires more information: {plan.missing_information_reason}",
                    error=plan.missing_information_reason,
                )
                self._active_cycle = res
                return res

            # Check for security-critical infrastructure
            requires_elevated = plan.requires_elevated_authorization

            # 5. Adapt to Existing Developer Pipeline
            dev_task = self.planner.to_developer_task(plan)
            dev_plan = self.planner.to_developer_plan(plan, dev_task)

            # 6. Generate Implementation Proposal via DeveloperCodeGenerator
            gen_result = self.generator.generate_from_plan(dev_task, dev_plan)
            proposal = gen_result.proposal

            if not gen_result.is_valid or not proposal or not proposal.is_valid:
                err_msg = gen_result.errors[0] if gen_result.errors else (proposal.validation_error if proposal else "Proposal generation failed.")
                logger.warning(f"Proposal generation failed in cycle '{cycle_id}': {err_msg}")
                res = SelfDevelopmentCycleResult(
                    self_development_id=cycle_id,
                    goal=goal,
                    plan=plan,
                    proposal=proposal,
                    status="generation_failed",
                    requires_elevated_authorization=requires_elevated,
                    summary=f"Proposal generation failed: {err_msg}",
                    error=err_msg,
                )
                self._active_cycle = res
                return res

            # 7. Scope Enforcement: Target file must be strictly within plan's relevant_files
            scope_ok, scope_err = validate_scope(plan.relevant_files, proposal.target_file)
            if not scope_ok:
                logger.error(f"Cycle '{cycle_id}' violated scope boundary: {scope_err}")
                res = SelfDevelopmentCycleResult(
                    self_development_id=cycle_id,
                    goal=goal,
                    plan=plan,
                    proposal=proposal,
                    status="scope_expansion_rejected",
                    requires_elevated_authorization=requires_elevated,
                    summary=scope_err,
                    error=scope_err,
                )
                self._active_cycle = res
                return res

            # 8. Operation Enforcement: Reject deletion or shell operations
            op_ok, op_err = validate_operation(proposal.operation)
            if not op_ok:
                logger.error(f"Cycle '{cycle_id}' requested unsupported operation: {op_err}")
                res = SelfDevelopmentCycleResult(
                    self_development_id=cycle_id,
                    goal=goal,
                    plan=plan,
                    proposal=proposal,
                    status="unsupported_operation",
                    requires_elevated_authorization=requires_elevated,
                    summary=op_err,
                    error=op_err,
                )
                self._active_cycle = res
                return res

            # 9. Register Proposal with DeveloperExecutor (stops at PERMISSION_PENDING)
            prep_res = self.executor.prepare_proposal(proposal, dev_plan)

            # Format user-facing proposal display
            display = self.format_self_development_proposal(cycle_id, goal, plan, proposal)

            res = SelfDevelopmentCycleResult(
                self_development_id=cycle_id,
                goal=goal,
                plan=plan,
                proposal=proposal,
                execution_result=prep_res,
                status="pending_permission",
                requires_elevated_authorization=requires_elevated,
                summary=display,
                provider=proposal.provider,
                model=proposal.model,
                cloud_request_id=proposal.cloud_request_id,
            )
            self._active_cycle = res
            logger.info(f"Cycle '{cycle_id}' successfully paused at Human Approval Boundary.")
            return res

        except Exception as exc:
            logger.exception(f"Unexpected error in self-development cycle '{cycle_id}': {exc}")
            res = SelfDevelopmentCycleResult(
                self_development_id=cycle_id,
                goal=goal,
                status="failed",
                summary=f"Cycle failed unexpectedly: {type(exc).__name__}: {exc}",
                error=str(exc),
            )
            self._active_cycle = res
            return res
        finally:
            self._is_running = False

    def apply_approved(
        self,
        explicit_permission: bool = True,
        elevated_authorization: bool = False,
    ) -> SelfDevelopmentCycleResult:
        """Applies an approved proposal, executes tests, evaluates results, and records experience."""
        if not self._active_cycle or not self._active_cycle.proposal:
            raise ValueError("No active self-development proposal awaiting approval.")

        cycle = self._active_cycle
        proposal = cycle.proposal
        plan = cycle.plan
        goal = cycle.goal

        # Recursion Guard
        if self._is_running:
            raise SelfDevelopmentRecursionError("Recursive self-development invocation detected.")

        self._is_running = True
        try:
            # Check elevated authorization for security-critical infrastructure
            if cycle.requires_elevated_authorization and not elevated_authorization:
                msg = (
                    "ELEVATED_AUTHORIZATION_REQUIRED: Proposal modifies security-critical infrastructure "
                    f"('{proposal.target_file}'). Explicit elevated authorization must be confirmed."
                )
                logger.warning(msg)
                exec_denied = DeveloperExecutionResult(
                    success=False,
                    status=ExecutionStatus.PERMISSION_DENIED.value,
                    stage="PERMISSION",
                    target_file=proposal.target_file,
                    operation=proposal.operation,
                    proposal_id=proposal.proposal_id,
                    proposal=proposal,
                    permission_granted=False,
                    overall_outcome="PERMISSION_DENIED",
                    summary=msg,
                    error=msg,
                )
                eval_res = self.evaluator.evaluate(
                    goal=goal,
                    plan=plan,
                    proposal=proposal,
                    execution_result=exec_denied,
                    self_development_id=cycle.self_development_id,
                )
                cycle.execution_result = exec_denied
                cycle.evaluation = eval_res
                cycle.status = "permission_denied"
                cycle.error = msg
                return cycle

            if not explicit_permission:
                # Permission denied
                exec_denied = DeveloperExecutionResult(
                    success=False,
                    status=ExecutionStatus.PERMISSION_DENIED.value,
                    stage="PERMISSION",
                    target_file=proposal.target_file,
                    operation=proposal.operation,
                    proposal_id=proposal.proposal_id,
                    proposal=proposal,
                    permission_granted=False,
                    overall_outcome="PERMISSION_DENIED",
                    summary="Self-development proposal denied by user.",
                )
                eval_res = self.evaluator.evaluate(
                    goal=goal,
                    plan=plan,
                    proposal=proposal,
                    execution_result=exec_denied,
                    self_development_id=cycle.self_development_id,
                )
                cycle.execution_result = exec_denied
                cycle.evaluation = eval_res
                cycle.status = "permission_denied"
                return cycle

            # Grant write permission for this approved proposal target
            if self.permission_manager:
                self.permission_manager.grant_write_permission(proposal.target_file)

            # Execute proposal via existing DeveloperExecutor
            dev_plan = self.planner.to_developer_plan(plan) if plan else None
            exec_res = self.executor.execute_proposal(
                proposal=proposal,
                explicit_permission=True,
                run_tests=True,
                plan=dev_plan,
            )

            # Evaluate outcome strictly using physical evidence
            evaluation = self.evaluator.evaluate(
                goal=goal,
                plan=plan,
                proposal=proposal,
                execution_result=exec_res,
                self_development_id=cycle.self_development_id,
            )

            # Record sanitized experience record in memory
            if self.memory_manager:
                try:
                    from experience.extractor import ExperienceExtractor
                    exp_rec = ExperienceExtractor.from_self_development_evaluation(
                        evaluation=evaluation,
                        goal=goal,
                        plan=plan,
                        proposal=proposal,
                        execution_result=exec_res,
                    )
                    self.memory_manager.record_experience(exp_rec)
                    logger.info(f"Persisted ExperienceRecord for cycle '{cycle.self_development_id}'.")
                except Exception as mem_err:
                    logger.warning(f"Failed to record experience memory: {mem_err}")

            cycle.execution_result = exec_res
            cycle.evaluation = evaluation
            cycle.status = "completed" if evaluation.improvement_success else "evaluated_with_failures"
            cycle.summary = (
                f"Self-Development Outcome: {evaluation.overall_outcome}\n"
                f"Applied: {evaluation.change_applied} | Verified: {evaluation.verification_status} | "
                f"Tests: {evaluation.test_status} ({evaluation.tests_passed} passed, {evaluation.tests_failed} failed)\n"
                f"{evaluation.observed_result}"
            )
            return cycle

        finally:
            self._is_running = False

    def deny_active_proposal(self, reason: str = "User denied proposal.") -> SelfDevelopmentCycleResult:
        """Explicitly denies the active self-development proposal."""
        if not self._active_cycle:
            raise ValueError("No active self-development cycle.")

        cycle = self._active_cycle
        proposal = cycle.proposal

        if self.executor and hasattr(self.executor, "applier") and self.executor.applier:
            self.executor.applier.clear_pending_proposal()

        exec_denied = DeveloperExecutionResult(
            success=False,
            status=ExecutionStatus.PERMISSION_DENIED.value,
            stage="PERMISSION",
            target_file=proposal.target_file if proposal else "",
            operation=proposal.operation if proposal else "",
            proposal_id=proposal.proposal_id if proposal else None,
            proposal=proposal,
            permission_granted=False,
            overall_outcome="PERMISSION_DENIED",
            summary=f"Proposal was denied: {reason}",
        )
        evaluation = self.evaluator.evaluate(
            goal=cycle.goal,
            plan=cycle.plan,
            proposal=proposal,
            execution_result=exec_denied,
            self_development_id=cycle.self_development_id,
        )

        cycle.execution_result = exec_denied
        cycle.evaluation = evaluation
        cycle.status = "cancelled"
        cycle.summary = f"Self-Development Proposal Cancelled: {reason}"

        if self.memory_manager:
            try:
                from experience.extractor import ExperienceExtractor
                exp_rec = ExperienceExtractor.from_self_development_evaluation(
                    evaluation=evaluation,
                    goal=cycle.goal,
                    plan=cycle.plan,
                    proposal=proposal,
                    execution_result=exec_denied,
                )
                self.memory_manager.record_experience(exp_rec)
                logger.info(f"Persisted denied ExperienceRecord for cycle '{cycle.self_development_id}'.")
            except Exception as mem_err:
                logger.warning(f"Failed to record denied self-development experience: {mem_err}")

        return cycle

    @staticmethod
    def format_self_development_proposal(
        cycle_id: str,
        goal: SelfDevelopmentGoal,
        plan: SelfDevelopmentPlan,
        proposal: CodeChangeProposal,
    ) -> str:
        """Formats the clean, user-facing self-development proposal display."""
        lines = [
            "============================================================",
            "SELF-DEVELOPMENT PROPOSAL",
            "============================================================",
            f"Cycle ID:    {cycle_id}",
            f"Goal:        {goal.description}",
            f"Scope:       {', '.join(goal.scope) if goal.scope else 'Inferred from codebase analysis'}",
            f"Target File: {proposal.target_file}",
            f"Operation:   {proposal.operation}",
            f"Symbols:     {', '.join(plan.relevant_symbols) if plan.relevant_symbols else 'N/A'}",
            f"Model:       {proposal.model or 'local'} ({proposal.provider or 'local'})",
        ]
        if proposal.cloud_request_id:
            lines.append(f"Cloud Req ID: {proposal.cloud_request_id}")
        if plan.requires_elevated_authorization:
            lines.append("⚠️  SECURITY NOTICE: Modifies security-critical components. Elevated authorization required.")
        lines.extend([
            "",
            "Change Summary:",
            f"  {plan.proposed_improvement}",
            "",
            "Risks:",
        ])
        for r in plan.risks:
            lines.append(f"  • {r}")
        lines.extend([
            "",
            "Validation Strategy:",
        ])
        for v in plan.validation_strategy:
            lines.append(f"  • {v}")
        lines.extend([
            "",
            "Permission Required. Run '/approve' to apply and test, or '/deny' to reject.",
            "============================================================",
        ])
        return "\n".join(lines)
