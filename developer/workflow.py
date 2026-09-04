import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger("VIDURA.developer.workflow")


class DeveloperWorkflowState(str, Enum):
    """Centralized lifecycle states for the VIDURA developer-agent workflow."""
    IDLE = "IDLE"
    TASK_ANALYZED = "TASK_ANALYZED"
    NEEDS_INFORMATION = "NEEDS_INFORMATION"
    PLAN_READY = "PLAN_READY"
    GENERATION_READY = "GENERATION_READY"
    PROPOSAL_PENDING = "PROPOSAL_PENDING"
    PERMISSION_PENDING = "PERMISSION_PENDING"
    APPROVED = "APPROVED"
    APPLYING = "APPLYING"
    APPLIED = "APPLIED"
    VERIFYING = "VERIFYING"
    VERIFIED = "VERIFIED"
    TEST_PLANNED = "TEST_PLANNED"
    TEST_PERMISSION_PENDING = "TEST_PERMISSION_PENDING"
    TESTING = "TESTING"
    TEST_PASSED = "TEST_PASSED"
    TEST_FAILED = "TEST_FAILED"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"
    STALE = "STALE"


TERMINAL_FAILURE_STATES = {
    DeveloperWorkflowState.FAILED,
    DeveloperWorkflowState.STALE,
    DeveloperWorkflowState.CANCELLED,
    DeveloperWorkflowState.NEEDS_INFORMATION,
}


class InvalidStateTransitionError(ValueError):
    """Raised when an illegal workflow state transition is attempted."""
    pass


# Strict transition mapping: CurrentState -> Set of valid next states
VALID_TRANSITIONS: dict[DeveloperWorkflowState, set[DeveloperWorkflowState]] = {
    DeveloperWorkflowState.IDLE: {
        DeveloperWorkflowState.TASK_ANALYZED,
        DeveloperWorkflowState.NEEDS_INFORMATION,
        DeveloperWorkflowState.PLAN_READY,
        DeveloperWorkflowState.GENERATION_READY,
        DeveloperWorkflowState.PROPOSAL_PENDING,
        DeveloperWorkflowState.PERMISSION_PENDING,
        DeveloperWorkflowState.FAILED,
    },
    DeveloperWorkflowState.TASK_ANALYZED: {
        DeveloperWorkflowState.PLAN_READY,
        DeveloperWorkflowState.NEEDS_INFORMATION,
        DeveloperWorkflowState.FAILED,
        DeveloperWorkflowState.CANCELLED,
    },
    DeveloperWorkflowState.PLAN_READY: {
        DeveloperWorkflowState.GENERATION_READY,
        DeveloperWorkflowState.NEEDS_INFORMATION,
        DeveloperWorkflowState.FAILED,
        DeveloperWorkflowState.CANCELLED,
    },
    DeveloperWorkflowState.GENERATION_READY: {
        DeveloperWorkflowState.PROPOSAL_PENDING,
        DeveloperWorkflowState.NEEDS_INFORMATION,
        DeveloperWorkflowState.FAILED,
        DeveloperWorkflowState.CANCELLED,
    },
    DeveloperWorkflowState.PROPOSAL_PENDING: {
        DeveloperWorkflowState.PERMISSION_PENDING,
        DeveloperWorkflowState.FAILED,
        DeveloperWorkflowState.CANCELLED,
    },
    DeveloperWorkflowState.PERMISSION_PENDING: {
        DeveloperWorkflowState.APPROVED,
        DeveloperWorkflowState.CANCELLED,
        DeveloperWorkflowState.FAILED,
    },
    DeveloperWorkflowState.APPROVED: {
        DeveloperWorkflowState.APPLYING,
        DeveloperWorkflowState.CANCELLED,
        DeveloperWorkflowState.STALE,
        DeveloperWorkflowState.FAILED,
    },
    DeveloperWorkflowState.APPLYING: {
        DeveloperWorkflowState.APPLIED,
        DeveloperWorkflowState.STALE,
        DeveloperWorkflowState.FAILED,
    },
    DeveloperWorkflowState.APPLIED: {
        DeveloperWorkflowState.VERIFYING,
        DeveloperWorkflowState.FAILED,
    },
    DeveloperWorkflowState.VERIFYING: {
        DeveloperWorkflowState.VERIFIED,
        DeveloperWorkflowState.FAILED,
    },
    DeveloperWorkflowState.VERIFIED: {
        DeveloperWorkflowState.TEST_PLANNED,
        DeveloperWorkflowState.COMPLETED,
        DeveloperWorkflowState.FAILED,
    },
    DeveloperWorkflowState.TEST_PLANNED: {
        DeveloperWorkflowState.TEST_PERMISSION_PENDING,
        DeveloperWorkflowState.TESTING,
        DeveloperWorkflowState.COMPLETED,
        DeveloperWorkflowState.FAILED,
    },
    DeveloperWorkflowState.TEST_PERMISSION_PENDING: {
        DeveloperWorkflowState.TESTING,
        DeveloperWorkflowState.COMPLETED,
        DeveloperWorkflowState.FAILED,
    },
    DeveloperWorkflowState.TESTING: {
        DeveloperWorkflowState.TEST_PASSED,
        DeveloperWorkflowState.TEST_FAILED,
        DeveloperWorkflowState.FAILED,
    },
    DeveloperWorkflowState.TEST_PASSED: {
        DeveloperWorkflowState.COMPLETED,
    },
    DeveloperWorkflowState.TEST_FAILED: {
        DeveloperWorkflowState.COMPLETED,
    },
    DeveloperWorkflowState.COMPLETED: {
        DeveloperWorkflowState.IDLE,
    },
    # Terminal failure states have NO outgoing transitions without explicit reset()
    DeveloperWorkflowState.FAILED: set(),
    DeveloperWorkflowState.STALE: set(),
    DeveloperWorkflowState.CANCELLED: set(),
    DeveloperWorkflowState.NEEDS_INFORMATION: set(),
}


class WorkflowOutcome(str, Enum):
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"
    NEEDS_INFORMATION = "NEEDS_INFORMATION"
    CHANGE_APPLIED_BUT_TESTS_FAILED = "CHANGE_APPLIED_BUT_TESTS_FAILED"
    PERMISSION_DENIED = "PERMISSION_DENIED"
    APPLICATION_FAILED = "APPLICATION_FAILED"
    VERIFICATION_FAILED = "VERIFICATION_FAILED"
    STALE_PROPOSAL = "STALE_PROPOSAL"


@dataclass
class MultiStageOutcome:
    """Detailed stage-by-stage outcome tracking ensuring no partial success is collapsed into a single boolean."""
    COMPLETED = WorkflowOutcome.COMPLETED
    FAILED = WorkflowOutcome.FAILED
    CANCELLED = WorkflowOutcome.CANCELLED
    NEEDS_INFORMATION = WorkflowOutcome.NEEDS_INFORMATION
    CHANGE_APPLIED_BUT_TESTS_FAILED = WorkflowOutcome.CHANGE_APPLIED_BUT_TESTS_FAILED
    PERMISSION_DENIED = WorkflowOutcome.PERMISSION_DENIED
    APPLICATION_FAILED = WorkflowOutcome.APPLICATION_FAILED
    VERIFICATION_FAILED = WorkflowOutcome.VERIFICATION_FAILED
    STALE_PROPOSAL = WorkflowOutcome.STALE_PROPOSAL

    task_analysis: str = "not_started"
    plan: str = "not_started"
    generation: str = "not_started"
    permission: str = "not_started"
    application: str = "not_started"
    verification: str = "not_started"
    testing: str = "not_started"
    overall: str = "not_started"
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_analysis": self.task_analysis,
            "plan": self.plan,
            "generation": self.generation,
            "permission": self.permission,
            "application": self.application,
            "verification": self.verification,
            "testing": self.testing,
            "overall": self.overall,
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MultiStageOutcome":
        if not isinstance(data, dict):
            return cls()
        return cls(
            task_analysis=str(data.get("task_analysis", "not_started")),
            plan=str(data.get("plan", "not_started")),
            generation=str(data.get("generation", "not_started")),
            permission=str(data.get("permission", "not_started")),
            application=str(data.get("application", "not_started")),
            verification=str(data.get("verification", "not_started")),
            testing=str(data.get("testing", "not_started")),
            overall=str(data.get("overall", "not_started")),
            error=data.get("error"),
        )


class DeveloperWorkflowStateMachine:
    """Central state machine managing and validating developer lifecycle transitions."""

    def __init__(self, initial_state: DeveloperWorkflowState = DeveloperWorkflowState.IDLE) -> None:
        self._current_state: DeveloperWorkflowState = initial_state
        self._history: list[DeveloperWorkflowState] = [initial_state]
        self._outcome = MultiStageOutcome()
        self._metadata: dict[str, Any] = {}

    @property
    def current_state(self) -> DeveloperWorkflowState:
        return self._current_state

    @property
    def history(self) -> list[DeveloperWorkflowState]:
        return list(self._history)

    @property
    def outcome(self) -> MultiStageOutcome:
        return self._outcome

    @property
    def is_terminal(self) -> bool:
        return self._current_state in TERMINAL_FAILURE_STATES or self._current_state == DeveloperWorkflowState.COMPLETED

    def can_transition_to(self, target_state: DeveloperWorkflowState) -> bool:
        """Checks whether a transition to target_state is legal from the current state."""
        allowed = VALID_TRANSITIONS.get(self._current_state, set())
        return target_state in allowed

    def transition_to(
        self,
        target_state: DeveloperWorkflowState | str,
        reason: str = "",
        error: str | None = None,
    ) -> DeveloperWorkflowState:
        """Transitions to target_state if valid, or raises InvalidStateTransitionError."""
        if isinstance(target_state, str):
            try:
                target_state = DeveloperWorkflowState(target_state)
            except ValueError:
                raise InvalidStateTransitionError(f"Unknown workflow state: '{target_state}'")

        if self._current_state in TERMINAL_FAILURE_STATES:
            msg = (
                f"Cannot transition from terminal state '{self._current_state.value}' to '{target_state.value}'. "
                f"Workflow requires explicit reset() or new user action."
            )
            logger.warning(msg)
            raise InvalidStateTransitionError(msg)

        if not self.can_transition_to(target_state):
            msg = f"Illegal state transition from '{self._current_state.value}' to '{target_state.value}'. Reason: {reason or 'N/A'}"
            logger.warning(msg)
            raise InvalidStateTransitionError(msg)

        logger.info(f"Workflow state transition: {self._current_state.value} -> {target_state.value} ({reason or 'no reason'})")
        self._current_state = target_state
        self._history.append(target_state)

        # Update stage outcome mapping
        self._update_outcome_for_state(target_state, error)

        return self._current_state

    def _update_outcome_for_state(self, state: DeveloperWorkflowState, error: str | None = None) -> None:
        """Maps lifecycle state changes into granular MultiStageOutcome fields."""
        if error:
            self._outcome.error = error

        if state == DeveloperWorkflowState.TASK_ANALYZED:
            self._outcome.task_analysis = "success"
        elif state == DeveloperWorkflowState.NEEDS_INFORMATION:
            self._outcome.task_analysis = "needs_information"
            self._outcome.plan = "needs_information"
            self._outcome.overall = "NEEDS_INFORMATION"
        elif state == DeveloperWorkflowState.PLAN_READY:
            self._outcome.plan = "ready"
        elif state == DeveloperWorkflowState.GENERATION_READY:
            self._outcome.generation = "ready"
        elif state == DeveloperWorkflowState.PROPOSAL_PENDING:
            self._outcome.generation = "success"
        elif state == DeveloperWorkflowState.PERMISSION_PENDING:
            self._outcome.permission = "pending"
        elif state == DeveloperWorkflowState.APPROVED:
            self._outcome.permission = "granted"
        elif state == DeveloperWorkflowState.CANCELLED:
            self._outcome.permission = "denied"
            self._outcome.overall = "CANCELLED"
        elif state == DeveloperWorkflowState.APPLYING:
            self._outcome.application = "applying"
        elif state == DeveloperWorkflowState.APPLIED:
            self._outcome.application = "success"
        elif state == DeveloperWorkflowState.VERIFYING:
            self._outcome.verification = "verifying"
        elif state == DeveloperWorkflowState.VERIFIED:
            self._outcome.verification = "passed"
        elif state == DeveloperWorkflowState.TEST_PLANNED:
            self._outcome.testing = "planned"
        elif state == DeveloperWorkflowState.TEST_PERMISSION_PENDING:
            self._outcome.testing = "permission_pending"
        elif state == DeveloperWorkflowState.TESTING:
            self._outcome.testing = "running"
        elif state == DeveloperWorkflowState.TEST_PASSED:
            self._outcome.testing = "passed"
        elif state == DeveloperWorkflowState.TEST_FAILED:
            self._outcome.testing = "failed"
        elif state == DeveloperWorkflowState.STALE:
            self._outcome.application = "stale"
            self._outcome.overall = "STALE_PROPOSAL"
        elif state == DeveloperWorkflowState.FAILED:
            self._outcome.overall = "FAILED"
        elif state == DeveloperWorkflowState.COMPLETED:
            if self._outcome.testing == "failed":
                self._outcome.overall = "CHANGE_APPLIED_BUT_TESTS_FAILED"
            elif self._outcome.verification == "passed":
                self._outcome.overall = "COMPLETED"
            else:
                self._outcome.overall = "COMPLETED"

    def reset(self) -> None:
        """Explicitly resets the workflow state machine back to IDLE."""
        logger.info(f"Resetting workflow state machine from '{self._current_state.value}' to IDLE.")
        self._current_state = DeveloperWorkflowState.IDLE
        self._history = [DeveloperWorkflowState.IDLE]
        self._outcome = MultiStageOutcome()
        self._metadata = {}
