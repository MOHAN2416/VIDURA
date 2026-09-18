import logging
from typing import Any

from developer.models import (
    CodeChangeProposal,
    DeveloperExecutionResult,
    ApplicationStatus,
    ExecutionStatus,
)
from self_development.models import (
    SelfDevelopmentGoal,
    SelfDevelopmentPlan,
    SelfDevelopmentEvaluation,
)

logger = logging.getLogger("VIDURA.self_development.evaluator")


class SelfDevelopmentEvaluator:
    """Evaluates self-development cycles strictly using physical tool, verifier, and test runner evidence.

    CRITICAL SAFETY RULES:
    1. Untrusted Model Claims: LLM claims of success are NEVER authoritative.
    2. Physical Evidence Authority: Only successful application, physical verification, and 0-exit test runs qualify.
    3. Structured Partial Outcomes: Accurately reflects partial states (e.g. CHANGE_APPLIED_BUT_TESTS_FAILED).
    """

    def evaluate(
        self,
        goal: SelfDevelopmentGoal,
        plan: SelfDevelopmentPlan | None = None,
        proposal: CodeChangeProposal | None = None,
        execution_result: DeveloperExecutionResult | None = None,
        self_development_id: str = "",
    ) -> SelfDevelopmentEvaluation:
        """Evaluates the outcome of a self-development cycle based strictly on evidence."""
        logger.info(f"Evaluating self-development cycle: id='{self_development_id}', goal='{goal.description}'")

        change_applied = False
        verification_status = False
        test_status = "not_run"
        tests_passed = 0
        tests_failed = 0
        regression_status = "not_run"
        observed_result = ""
        overall_outcome = "FAILED"
        improvement_success = False
        lesson = ""
        recommendation = ""

        # Extract provider metadata
        prov = None
        mod = None
        fallback_used = False
        cloud_req_id = None

        if execution_result:
            prov = execution_result.provider
            mod = execution_result.model
            fallback_used = execution_result.fallback_used
            cloud_req_id = execution_result.cloud_request_id
        elif proposal:
            prov = proposal.provider
            mod = proposal.model
            fallback_used = proposal.fallback_used
            cloud_req_id = proposal.cloud_request_id

        # 1. Check if execution result exists
        if not execution_result:
            if proposal and proposal.is_valid:
                overall_outcome = "PROPOSAL_CREATED_NOT_APPLIED"
                observed_result = "Proposal was generated but was not submitted for application."
                lesson = "Self-development cycle stopped after proposal generation."
                recommendation = "Review proposal and execute with permission if desired."
            else:
                overall_outcome = "FAILED"
                observed_result = "Self-development failed before proposal execution."
                lesson = "Cycle could not complete planning or proposal generation."
                recommendation = "Refine goal description and ensure codebase components exist."

            return SelfDevelopmentEvaluation(
                goal_id=goal.goal_id,
                self_development_id=self_development_id,
                change_applied=False,
                verification_status=False,
                test_status=test_status,
                tests_passed=0,
                tests_failed=0,
                improvement_success=False,
                overall_outcome=overall_outcome,
                observed_result=observed_result,
                regression_status=regression_status,
                lesson=lesson,
                recommendation=recommendation,
                provider_used=prov,
                model_used=mod,
                fallback_used=fallback_used,
                cloud_request_id=cloud_req_id,
            )

        # 2. Check permission state
        if not execution_result.permission_granted:
            if execution_result.status == ExecutionStatus.PERMISSION_DENIED.value:
                overall_outcome = "PERMISSION_DENIED"
                observed_result = "Permission was explicitly denied by user or application."
                lesson = "Self-development proposals require human approval before any disk modifications."
                recommendation = "Re-align self-development goal with user requirements."
            elif execution_result.status == ExecutionStatus.PENDING_PERMISSION.value:
                overall_outcome = "PROPOSAL_CREATED_NOT_APPLIED"
                observed_result = "Proposal registered awaiting explicit permission."
                lesson = "Proposal halted at permission boundary as required."
                recommendation = "Authorize change with /approve to continue."
            else:
                overall_outcome = "PROPOSAL_CREATED_NOT_APPLIED"
                observed_result = f"Permission was not granted ({execution_result.status})."
                lesson = "Modifications cannot proceed without explicit authorization."
                recommendation = "Ensure permission is granted before executing proposal."

            return SelfDevelopmentEvaluation(
                goal_id=goal.goal_id,
                self_development_id=self_development_id,
                change_applied=False,
                verification_status=False,
                test_status="not_run",
                tests_passed=0,
                tests_failed=0,
                improvement_success=False,
                overall_outcome=overall_outcome,
                observed_result=observed_result,
                regression_status=regression_status,
                lesson=lesson,
                recommendation=recommendation,
                provider_used=prov,
                model_used=mod,
                fallback_used=fallback_used,
                cloud_request_id=cloud_req_id,
            )

        # 3. Check physical application result
        app_res = execution_result.application_result
        if app_res and app_res.success:
            change_applied = True
        else:
            overall_outcome = "APPLICATION_FAILED"
            observed_result = f"Physical file modification failed: {execution_result.error or 'application failure'}"
            lesson = "Application failed; filesystem was not modified."
            recommendation = "Inspect file write permissions and proposal target validity."
            return SelfDevelopmentEvaluation(
                goal_id=goal.goal_id,
                self_development_id=self_development_id,
                change_applied=False,
                verification_status=False,
                test_status="not_run",
                tests_passed=0,
                tests_failed=0,
                improvement_success=False,
                overall_outcome=overall_outcome,
                observed_result=observed_result,
                regression_status=regression_status,
                lesson=lesson,
                recommendation=recommendation,
                provider_used=prov,
                model_used=mod,
                fallback_used=fallback_used,
                cloud_request_id=cloud_req_id,
            )

        # 4. Check physical verification result
        ver_res = execution_result.verification_result
        if ver_res and ver_res.success:
            verification_status = True
        else:
            overall_outcome = "VERIFICATION_FAILED"
            observed_result = "Filesystem verification failed: disk contents do not match proposal."
            lesson = "Physical disk state did not match the expected proposal specification."
            recommendation = "Check for concurrent file modifications or stale proposal contents."
            return SelfDevelopmentEvaluation(
                goal_id=goal.goal_id,
                self_development_id=self_development_id,
                change_applied=True,
                verification_status=False,
                test_status="not_run",
                tests_passed=0,
                tests_failed=0,
                improvement_success=False,
                overall_outcome=overall_outcome,
                observed_result=observed_result,
                regression_status=regression_status,
                lesson=lesson,
                recommendation=recommendation,
                provider_used=prov,
                model_used=mod,
                fallback_used=fallback_used,
                cloud_request_id=cloud_req_id,
            )

        # 5. Check test results
        test_res = execution_result.test_result
        if test_res:
            tests_passed = getattr(test_res, "tests_passed", None)
            if tests_passed is None:
                tests_passed = getattr(test_res, "passed_tests", 0)
            tests_failed = getattr(test_res, "tests_failed", None)
            if tests_failed is None:
                tests_failed = getattr(test_res, "failed_tests", 0)
            exit_code = getattr(test_res, "exit_code", -1)

            if exit_code == 0 and tests_failed == 0:
                test_status = "passed"
                regression_status = "passed"
            else:
                test_status = "failed"
                regression_status = "failed"
        else:
            test_status = "no_tests"
            regression_status = "not_run"

        # 6. Synthesize final outcome strictly based on evidence
        if change_applied and verification_status and test_status == "passed":
            overall_outcome = "SELF_DEVELOPMENT_SUCCESS"
            improvement_success = True
            observed_result = f"Change successfully applied, physically verified, and verified with {tests_passed} passing tests."
            lesson = f"Successfully achieved self-development goal: '{goal.description}'."
            recommendation = "Preserve modification and monitor subsequent operations."
        elif change_applied and verification_status and test_status == "failed":
            overall_outcome = "CHANGE_APPLIED_BUT_TESTS_FAILED"
            improvement_success = False
            observed_result = f"Change applied and verified on disk, but test suite failed ({tests_failed} failed tests)."
            lesson = "Modification introduced regressions or failed test expectations."
            recommendation = "Roll back modification or generate a separate targeted fix with tests."
        elif change_applied and verification_status and test_status == "no_tests":
            overall_outcome = "CHANGE_APPLIED_NO_TESTS"
            improvement_success = True
            observed_result = "Change applied and verified on disk; no automated tests were configured for this target."
            lesson = "Modification applied without test verification."
            recommendation = "Add automated unit tests for modified components."
        else:
            overall_outcome = "FAILED"
            improvement_success = False
            observed_result = f"Self-development execution failed: {execution_result.error or 'unknown failure'}"
            lesson = "Cycle encountered failures during application or verification."
            recommendation = "Investigate failure reasons before attempting new cycle."

        return SelfDevelopmentEvaluation(
            goal_id=goal.goal_id,
            self_development_id=self_development_id,
            change_applied=change_applied,
            verification_status=verification_status,
            test_status=test_status,
            tests_passed=tests_passed,
            tests_failed=tests_failed,
            improvement_success=improvement_success,
            overall_outcome=overall_outcome,
            observed_result=observed_result,
            regression_status=regression_status,
            lesson=lesson,
            recommendation=recommendation,
            provider_used=prov,
            model_used=mod,
            fallback_used=fallback_used,
            cloud_request_id=cloud_req_id,
        )
