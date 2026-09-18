"""VIDURA Routing Models and Policy Structures.

Defines deterministic complexity classification, routing modes, reason codes,
and structured routing decision models for Phase 9.4.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class TaskComplexity(str, Enum):
    """Task complexity levels for deterministic routing."""
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"
    CRITICAL = "critical"


class RoutingMode(str, Enum):
    """Application-level routing preference modes."""
    AUTO = "auto"
    LOCAL = "local"
    CLOUD = "cloud"


class ReasonCode(str, Enum):
    """Concise deterministic reason codes for routing decisions."""
    LOCAL_DEFAULT = "local_default"
    LOCAL_SIMPLE_TASK = "local_simple_task"
    LOCAL_MODERATE_TASK = "local_moderate_task"
    CLOUD_COMPLEX_TASK = "cloud_complex_task"
    CLOUD_MULTI_FILE_TASK = "cloud_multi_file_task"
    CLOUD_ARCHITECTURAL_TASK = "cloud_architectural_task"
    CLOUD_DEBUGGING_TASK = "cloud_debugging_task"
    EXPLICIT_LOCAL = "explicit_local"
    EXPLICIT_CLOUD = "explicit_cloud"
    LOCAL_ONLY_POLICY = "local_only_policy"
    CLOUD_UNAVAILABLE = "cloud_unavailable"
    CLOUD_DISABLED = "cloud_disabled"
    CLOUD_LIMIT_REACHED = "cloud_limit_reached"
    CLOUD_SECURITY_RESTRICTION = "cloud_security_restriction"
    CLOUD_CONTEXT_LIMIT_EXCEEDED = "cloud_context_limit_exceeded"


# Keywords indicating architectural or cross-module changes
ARCHITECTURAL_KEYWORDS = (
    "architecture",
    "architectural",
    "refactor",
    "redesign",
    "cross-module",
    "dependency injection",
    "pipeline restructure",
    "breaking change",
    "multi-component",
)

# Keywords indicating debugging
DEBUG_KEYWORDS = (
    "debug",
    "debugging",
    "diagnose",
    "investigate failure",
    "fix bug",
    "stacktrace",
    "traceback",
)


class FailureCategory(str, Enum):
    """Classification of cloud provider errors for fallback eligibility."""
    TRANSIENT = "transient"
    CONFIGURATION = "configuration"
    AUTHENTICATION = "authentication"
    SECURITY = "security"
    INVALID_REQUEST = "invalid_request"
    USAGE_LIMIT = "usage_limit"
    UNKNOWN = "unknown"


def classify_cloud_failure(error: Exception) -> FailureCategory:
    """Classifies an error encountered during cloud inference.

    Determines whether the error is a transient failure (eligible for fallback)
    or an ineligible configuration/auth/security/request error.
    """
    from models.errors import (
        ProviderTimeout,
        ProviderAuthenticationError,
        ProviderConfigurationError,
        ProviderUnavailable,
        ProviderRequestError,
        CloudSecurityViolation,
        CloudUsageLimitReached,
        CloudContextLimitExceeded,
        CloudDisabledError,
    )
    if isinstance(error, ProviderTimeout):
        return FailureCategory.TRANSIENT
    elif isinstance(error, ProviderAuthenticationError):
        return FailureCategory.AUTHENTICATION
    elif isinstance(error, (ProviderConfigurationError, CloudDisabledError)):
        return FailureCategory.CONFIGURATION
    elif isinstance(error, CloudUsageLimitReached):
        return FailureCategory.USAGE_LIMIT
    elif isinstance(error, (CloudSecurityViolation, PermissionError)):
        return FailureCategory.SECURITY
    elif isinstance(error, ProviderUnavailable):
        err_msg = str(error).lower()
        if "missing" in err_msg or "not configured" in err_msg:
            return FailureCategory.CONFIGURATION
        return FailureCategory.TRANSIENT
    elif isinstance(error, ProviderRequestError):
        return FailureCategory.TRANSIENT
    elif isinstance(error, (ValueError, TypeError, CloudContextLimitExceeded)):
        return FailureCategory.INVALID_REQUEST
    return FailureCategory.UNKNOWN


def is_fallback_eligible(error: Exception) -> bool:
    """Determines whether an error qualifies for automatic local fallback.

    Only TRANSIENT failures (timeouts, network drops, temporary service outages)
    and configured USAGE_LIMIT blocks qualify for fallback.
    Authentication, configuration, and security errors are strictly ineligible.
    """
    return classify_cloud_failure(error) in (FailureCategory.TRANSIENT, FailureCategory.USAGE_LIMIT)


@dataclass(frozen=True)
class RoutingDecision:
    """Structured representation of a deterministic model routing decision."""

    provider: str  # "local" or "cloud"
    model: str     # "gemma4:e4b-it-qat" or "gemma4:31b-cloud"
    complexity: str
    reason_code: str
    task_type: str | None = None
    routing_mode: str = "auto"
    confidence: float = 1.0
    signals_used: dict[str, Any] = field(default_factory=dict)
    local_only: bool = False
    requested_provider: str | None = None
    requested_model: str | None = None
    actual_provider: str | None = None
    actual_model: str | None = None
    fallback_used: bool = False
    fallback_reason: str | None = None
    cloud_error: str | None = None
    cloud_request_id: str | None = None
    usage_metadata: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.requested_provider is None:
            object.__setattr__(self, "requested_provider", self.provider)
        if self.requested_model is None:
            object.__setattr__(self, "requested_model", self.model)
        if self.actual_provider is None:
            object.__setattr__(self, "actual_provider", self.provider)
        if self.actual_model is None:
            object.__setattr__(self, "actual_model", self.model)

    @property
    def selected_provider(self) -> str:
        """Returns the determined provider."""
        return self.actual_provider or self.provider

    @property
    def selected_model(self) -> str:
        """Returns the determined model identifier."""
        return self.actual_model or self.model

    def to_dict(self) -> dict[str, Any]:
        """Converts routing decision to dictionary structure."""
        return {
            "provider": self.provider,
            "model": self.model,
            "complexity": self.complexity,
            "reason_code": self.reason_code,
            "task_type": self.task_type,
            "routing_mode": self.routing_mode,
            "confidence": self.confidence,
            "signals_used": dict(self.signals_used),
            "local_only": self.local_only,
            "requested_provider": self.requested_provider,
            "requested_model": self.requested_model,
            "actual_provider": self.actual_provider,
            "actual_model": self.actual_model,
            "fallback_used": self.fallback_used,
            "fallback_reason": self.fallback_reason,
            "cloud_error": self.cloud_error,
            "cloud_request_id": self.cloud_request_id,
            "usage_metadata": dict(self.usage_metadata) if self.usage_metadata else None,
        }


def classify_complexity(
    task: Any = None,
    plan: Any = None,
    messages: list[dict[str, str]] | None = None,
    **kwargs: Any,
) -> tuple[TaskComplexity, ReasonCode, dict[str, Any]]:
    """Deterministically classifies the complexity of a request based on application signals.

    Inspects DeveloperPlan, DeveloperTask, and context parameters.
    Does NOT call any LLM or perform expensive scans.

    Args:
        task: Optional DeveloperTask instance.
        plan: Optional DeveloperPlan instance.
        messages: Optional conversation message history.
        **kwargs: Additional contextual hints (target_files, task_type, etc.).

    Returns:
        tuple of (TaskComplexity, ReasonCode, signals_dict).
    """
    signals: dict[str, Any] = {
        "file_count": 0,
        "symbol_count": 0,
        "component_count": 0,
        "dependency_count": 0,
        "task_type": None,
        "is_debugging": False,
        "is_architectural": False,
    }

    # 1. Extract signals from DeveloperPlan if available
    if plan is not None:
        rel_files = getattr(plan, "relevant_files", []) or []
        rel_symbols = getattr(plan, "relevant_symbols", []) or []
        aff_comps = getattr(plan, "affected_components", []) or []
        deps = getattr(plan, "dependencies", []) or []
        planned_changes = getattr(plan, "planned_changes", []) or []
        risks = getattr(plan, "risks", []) or []

        signals["file_count"] = len(rel_files)
        signals["symbol_count"] = len(rel_symbols)
        signals["component_count"] = len(aff_comps)
        signals["dependency_count"] = len(deps)

        combined_plan_text = " ".join(str(c) for c in (planned_changes + risks)).lower()
        signals["is_architectural"] = any(kw in combined_plan_text for kw in ARCHITECTURAL_KEYWORDS)

        orig_task = getattr(plan, "original_task", None)
        if orig_task:
            task_type_val = getattr(orig_task, "task_type", None)
            signals["task_type"] = task_type_val.value if hasattr(task_type_val, "value") else str(task_type_val or "")
            goal_text = str(getattr(orig_task, "goal", "")).lower()
            signals["is_debugging"] = (
                signals["task_type"] in ("code_debug", "debug") or
                any(kw in goal_text for kw in DEBUG_KEYWORDS)
            )

    # 2. Extract signals from DeveloperTask if plan not provided or incomplete
    if task is not None:
        task_type_val = getattr(task, "task_type", None)
        tt_str = task_type_val.value if hasattr(task_type_val, "value") else str(task_type_val or "")
        if not signals["task_type"]:
            signals["task_type"] = tt_str

        task_files = getattr(task, "target_files", []) or []
        task_symbols = getattr(task, "target_symbols", []) or []

        signals["file_count"] = max(signals["file_count"], len(task_files))
        signals["symbol_count"] = max(signals["symbol_count"], len(task_symbols))

        goal_text = (str(getattr(task, "goal", "")) + " " + str(getattr(task, "requested_change", ""))).lower()
        if not signals["is_debugging"]:
            signals["is_debugging"] = (
                tt_str in ("code_debug", "debug") or
                any(kw in goal_text for kw in DEBUG_KEYWORDS)
            )
        if not signals["is_architectural"]:
            signals["is_architectural"] = any(kw in goal_text for kw in ARCHITECTURAL_KEYWORDS)

    # 3. Fallback extraction from kwargs
    if "target_files" in kwargs and isinstance(kwargs["target_files"], list):
        signals["file_count"] = max(signals["file_count"], len(kwargs["target_files"]))
    elif "target_file" in kwargs and kwargs["target_file"]:
        signals["file_count"] = max(signals["file_count"], 1)

    if "task_type" in kwargs and not signals["task_type"]:
        signals["task_type"] = str(kwargs["task_type"])

    # 4. Determine complexity and reason code based on extracted signals
    task_type_str = (signals["task_type"] or "").lower()

    # Rule A: CRITICAL
    # Very large multi-file changes or highly interconnected components
    if signals["file_count"] >= 4 or (signals["file_count"] >= 2 and signals["component_count"] >= 3) or signals["dependency_count"] >= 4:
        reason = ReasonCode.CLOUD_ARCHITECTURAL_TASK if signals["is_architectural"] else ReasonCode.CLOUD_COMPLEX_TASK
        return TaskComplexity.CRITICAL, reason, signals

    # Rule B: COMPLEX
    # B1: Multi-file changes (> 1 file)
    if signals["file_count"] > 1:
        return TaskComplexity.COMPLEX, ReasonCode.CLOUD_MULTI_FILE_TASK, signals

    # B2: Architectural changes across multiple components or explicit architectural refactor
    if signals["is_architectural"] or signals["component_count"] > 1 or signals["dependency_count"] >= 2:
        return TaskComplexity.COMPLEX, ReasonCode.CLOUD_ARCHITECTURAL_TASK, signals

    # B3: Difficult debugging involving multiple files or dependencies
    if signals["is_debugging"] and (signals["file_count"] > 1 or signals["component_count"] > 1 or signals["dependency_count"] > 1):
        return TaskComplexity.COMPLEX, ReasonCode.CLOUD_DEBUGGING_TASK, signals

    # Rule C: MODERATE
    # C1: Single file code change or implementation
    if signals["file_count"] == 1 and task_type_str in ("code_change", "developer", "coding"):
        return TaskComplexity.MODERATE, ReasonCode.LOCAL_MODERATE_TASK, signals

    # C2: Straightforward debugging in single file
    if signals["is_debugging"]:
        return TaskComplexity.MODERATE, ReasonCode.LOCAL_MODERATE_TASK, signals

    # C3: Focused test request
    if task_type_str in ("test_request", "testing"):
        return TaskComplexity.MODERATE, ReasonCode.LOCAL_MODERATE_TASK, signals

    # Rule D: SIMPLE
    # D1: Explanations and reviews
    if task_type_str in ("code_explanation", "code_review", "explanation"):
        return TaskComplexity.SIMPLE, ReasonCode.LOCAL_SIMPLE_TASK, signals

    # D2: General query or no developer task
    if task_type_str in ("general", "unable_to_classify", ""):
        # Check messages if present
        if messages:
            last_msg = messages[-1].get("content", "") if messages else ""
            if any(kw in last_msg.lower() for kw in ARCHITECTURAL_KEYWORDS):
                return TaskComplexity.COMPLEX, ReasonCode.CLOUD_ARCHITECTURAL_TASK, signals

        if signals["file_count"] == 0:
            return TaskComplexity.SIMPLE, ReasonCode.LOCAL_DEFAULT, signals

    # Default fallback: single file without special complexity is moderate
    if signals["file_count"] == 1:
        return TaskComplexity.MODERATE, ReasonCode.LOCAL_MODERATE_TASK, signals

    return TaskComplexity.SIMPLE, ReasonCode.LOCAL_DEFAULT, signals
