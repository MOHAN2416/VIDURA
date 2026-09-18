import logging
from pathlib import Path
from typing import Any

from codebase.manager import CodebaseManager
from task_understanding.models import DeveloperTask, TaskType
from planning.models import DeveloperPlan
from self_development.models import SelfDevelopmentGoal, SelfDevelopmentPlan
from self_development.analyzer import SelfDevelopmentAnalyzer
from self_development.security import is_security_critical_target

logger = logging.getLogger("VIDURA.self_development.planner")


class SelfDevelopmentPlanner:
    """Plans controlled, codebase-grounded self-development changes.

    CRITICAL ARCHITECTURAL BOUNDARY:
    Converts high-level SelfDevelopmentGoals into typed SelfDevelopmentPlans and
    adapts them to existing Phase 8 DeveloperTask / DeveloperPlan structures so
    the existing DeveloperCodeGenerator and DeveloperExecutor can be reused without duplication.
    """

    def __init__(
        self,
        analyzer: SelfDevelopmentAnalyzer | None = None,
        codebase_manager: CodebaseManager | None = None,
        workspace_root: str | Path | None = None,
    ) -> None:
        self.codebase_manager = codebase_manager or CodebaseManager(workspace_root=str(workspace_root) if workspace_root else None)
        self.analyzer = analyzer or SelfDevelopmentAnalyzer(codebase_manager=self.codebase_manager, workspace_root=workspace_root)

    def plan(self, goal: SelfDevelopmentGoal) -> SelfDevelopmentPlan:
        """Constructs a codebase-grounded SelfDevelopmentPlan from a SelfDevelopmentGoal."""
        logger.info(f"Generating SelfDevelopmentPlan for goal: '{goal.description}'")

        analysis = self.analyzer.analyze(goal)

        # Enforce security-critical infrastructure protection
        requires_elevated = bool(analysis.get("requires_elevated_authorization", False))
        for f in analysis.get("relevant_files", []):
            if is_security_critical_target(f):
                requires_elevated = True
                break

        plan = SelfDevelopmentPlan(
            goal=goal,
            current_behavior=str(analysis.get("current_behavior", "")),
            observed_problem=str(analysis.get("observed_problem", "")),
            relevant_files=list(analysis.get("relevant_files", [])),
            relevant_symbols=list(analysis.get("relevant_symbols", [])),
            dependencies=list(analysis.get("dependencies", [])),
            proposed_improvement=str(analysis.get("proposed_improvement", "")),
            expected_benefit=str(analysis.get("expected_benefit", "")),
            risks=list(analysis.get("risks", [])),
            constraints=list(analysis.get("constraints", [])),
            validation_strategy=list(analysis.get("validation_strategy", [])),
            rollback_strategy=str(analysis.get("rollback_strategy", "")),
            confidence=float(analysis.get("confidence", 1.0)),
            requires_more_information=bool(analysis.get("requires_more_information", False)),
            missing_information_reason=analysis.get("missing_information_reason"),
            requires_elevated_authorization=requires_elevated,
        )

        logger.info(
            f"Created SelfDevelopmentPlan: files={plan.relevant_files}, "
            f"elevated_auth={plan.requires_elevated_authorization}, "
            f"requires_info={plan.requires_more_information}"
        )
        return plan

    def to_developer_task(self, plan: SelfDevelopmentPlan) -> DeveloperTask:
        """Adapts a SelfDevelopmentPlan into an existing DeveloperTask."""
        return DeveloperTask(
            task_type=TaskType.CODE_CHANGE,
            is_development_task=True,
            goal=plan.goal.description,
            requested_change=plan.proposed_improvement,
            target_files=list(plan.relevant_files),
            target_symbols=list(plan.relevant_symbols),
            constraints=list(plan.constraints),
            expected_behavior=plan.expected_benefit,
            requires_codebase_analysis=True,
            confidence=plan.confidence,
            local_only=False,
        )

    def to_developer_plan(self, plan: SelfDevelopmentPlan, task: DeveloperTask | None = None) -> DeveloperPlan:
        """Adapts a SelfDevelopmentPlan into an existing DeveloperPlan."""
        dev_task = task or self.to_developer_task(plan)
        return DeveloperPlan(
            original_task=dev_task,
            relevant_files=list(plan.relevant_files),
            relevant_symbols=list(plan.relevant_symbols),
            dependencies=list(plan.dependencies),
            affected_components=list(plan.relevant_files),
            planned_changes=[plan.proposed_improvement] if plan.proposed_improvement else [],
            risks=list(plan.risks),
            constraints=list(plan.constraints),
            requires_more_information=plan.requires_more_information,
            missing_information_reason=plan.missing_information_reason,
            requires_plan_update=False,
            confidence=plan.confidence,
        )
