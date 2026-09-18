"""VIDURA Model Router.

Selects and routes requests to the configured ModelProvider (Local or Cloud).
Enforces strict allowlists for providers, routing modes, and model identifiers.
Implements Phase 9.4 intelligent, application-controlled, deterministic routing.
Prevents dynamic provider changes by the model and ensures permission isolation.
"""
from __future__ import annotations

import logging
import uuid
from typing import Any

from config import (
    Config,
    load_config,
    ALLOWED_PROVIDERS,
    ALLOWED_ROUTING_MODES,
    ALLOWED_LOCAL_MODELS,
    ALLOWED_CLOUD_MODELS,
)
from models.base import ModelProvider, ProviderCapabilities
from models.local import LocalProvider, DEFAULT_LOCAL_MODEL, DEFAULT_OLLAMA_HOST
from models.cloud import OllamaCloudProvider, DEFAULT_CLOUD_MODEL
from models.errors import (
    ProviderConfigurationError,
    ProviderUnavailable,
    CombinedProviderError,
    CloudSecurityViolation,
    CloudUsageLimitReached,
    CloudContextLimitExceeded,
    CloudDisabledError,
)
from models.routing import (
    TaskComplexity,
    RoutingMode,
    ReasonCode,
    RoutingDecision,
    classify_complexity,
    is_fallback_eligible,
)
from models.security import (
    is_protected_file_target,
    contains_sensitive_data,
    redact_secrets,
)
from models.usage import (
    CloudUsageTracker,
    estimate_tokens,
)

logger = logging.getLogger("VIDURA.models.router")


class ModelRouter(ModelProvider):
    """Deterministic application-level Model Router for VIDURA.

    Intelligently routes model interactions to either LocalProvider or OllamaCloudProvider
    based on application-controlled signals, task complexity, codebase scope, and explicit
    configuration policies.
    """

    def __init__(
        self,
        config: Config | None = None,
        provider_type: str | None = None,
        developer_provider_type: str | None = None,
        routing_mode: str | None = None,
        local_provider: ModelProvider | None = None,
        cloud_provider: ModelProvider | None = None,
        local_model: str | None = None,
        cloud_model: str | None = None,
        fallback_enabled: bool | None = None,
        cloud_enabled: bool | None = None,
        usage_tracker: CloudUsageTracker | None = None,
    ) -> None:
        self._config = config or load_config()

        # Determine application routing mode ("auto", "local", "cloud")
        raw_routing_mode = (
            routing_mode
            or getattr(self._config, "vidura_routing_mode", None)
            or "auto"
        )
        self._routing_mode = str(raw_routing_mode).lower().strip()
        if self._routing_mode not in ALLOWED_ROUTING_MODES:
            raise ProviderConfigurationError(
                f"Unsupported routing mode: '{self._routing_mode}'. "
                f"Supported modes are: {sorted(ALLOWED_ROUTING_MODES)}."
            )

        # Determine general provider type
        raw_provider = (
            provider_type
            or getattr(self._config, "vidura_model_provider", None)
            or getattr(self._config, "model_provider", "local")
        )
        self._provider_type = str(raw_provider).lower().strip()
        if self._provider_type not in ALLOWED_PROVIDERS:
            raise ProviderConfigurationError(
                f"Unsupported model provider: '{self._provider_type}'. "
                f"Supported providers are: {sorted(ALLOWED_PROVIDERS)}."
            )

        # Determine model identifiers
        self._local_model = (
            local_model
            or getattr(self._config, "vidura_local_model", None)
            or getattr(self._config, "ollama_model", DEFAULT_LOCAL_MODEL)
        )
        self._cloud_model = (
            cloud_model
            or getattr(self._config, "vidura_cloud_model", None)
            or getattr(self._config, "ollama_cloud_model", None)
            or DEFAULT_CLOUD_MODEL
        )

        # Strict allowlist verification
        if self._local_model not in ALLOWED_LOCAL_MODELS:
            raise ProviderConfigurationError(
                f"Invalid local model identifier: '{self._local_model}'. "
                f"Allowed local model is: {sorted(ALLOWED_LOCAL_MODELS)}."
            )

        if self._cloud_model not in ALLOWED_CLOUD_MODELS:
            raise ProviderConfigurationError(
                f"Invalid cloud model identifier: '{self._cloud_model}'. "
                f"Allowed cloud model is: {sorted(ALLOWED_CLOUD_MODELS)}."
            )

        # Determine target developer provider type
        raw_dev_provider = (
            developer_provider_type
            or getattr(self._config, "vidura_developer_model_provider", None)
            or "auto"
        )
        self._developer_provider_type = str(raw_dev_provider).lower().strip()
        if self._developer_provider_type not in ALLOWED_PROVIDERS:
            raise ProviderConfigurationError(
                f"Unsupported developer model provider: '{self._developer_provider_type}'. "
                f"Supported providers are: {sorted(ALLOWED_PROVIDERS)}."
            )

        # Initialize or assign provider instances
        host = getattr(self._config, "ollama_host", DEFAULT_OLLAMA_HOST)
        self._local_provider = local_provider or LocalProvider(host=host, model=self._local_model)

        api_key = getattr(self._config, "ollama_cloud_api_key", None)
        endpoint = getattr(self._config, "ollama_cloud_endpoint", None)
        timeout = getattr(self._config, "vidura_cloud_timeout_seconds", 60.0)
        self._cloud_provider = cloud_provider or OllamaCloudProvider(
            model=self._cloud_model,
            api_key=api_key,
            endpoint=endpoint,
            timeout=timeout,
        )

        if fallback_enabled is not None:
            self._fallback_enabled = bool(fallback_enabled)
        else:
            self._fallback_enabled = getattr(self._config, "vidura_cloud_fallback_enabled", False)

        if cloud_enabled is not None:
            self._cloud_enabled = bool(cloud_enabled)
        else:
            self._cloud_enabled = getattr(self._config, "vidura_cloud_enabled", True)

        self._max_context_tokens = getattr(self._config, "vidura_cloud_max_context_tokens", 8192)
        self._usage_tracker = usage_tracker or CloudUsageTracker(config=self._config)

        self._last_decision: RoutingDecision | None = None

        logger.info(
            f"Initialized ModelRouter: routing_mode='{self._routing_mode}', "
            f"active_provider='{self._provider_type}', "
            f"developer_provider='{self._developer_provider_type}', "
            f"local_model='{self._local_model}', cloud_model='{self._cloud_model}', "
            f"fallback_enabled={self._fallback_enabled}, "
            f"cloud_enabled={self._cloud_enabled}, "
            f"max_context_tokens={self._max_context_tokens}"
        )

    @property
    def fallback_enabled(self) -> bool:
        """Returns True if automatic cloud fallback to local model is enabled."""
        return self._fallback_enabled

    @property
    def cloud_enabled(self) -> bool:
        """Returns True if cloud model usage is globally enabled."""
        return self._cloud_enabled

    @property
    def usage_tracker(self) -> CloudUsageTracker:
        """Returns the CloudUsageTracker instance."""
        return self._usage_tracker

    @property
    def max_context_tokens(self) -> int:
        """Returns the configured max context tokens for cloud requests."""
        return self._max_context_tokens

    @property
    def routing_mode(self) -> str:
        """Returns the configured routing mode ('auto', 'local', 'cloud')."""
        return self._routing_mode


    @property
    def last_decision(self) -> RoutingDecision | None:
        """Returns the most recent RoutingDecision made by the router."""
        return self._last_decision

    @property
    def provider_type(self) -> str:
        """Returns the active provider type ('local' or 'cloud')."""
        return self._provider_type

    @property
    def developer_provider_type(self) -> str:
        """Returns the configured developer provider type ('auto', 'local', or 'cloud')."""
        return self._developer_provider_type

    @property
    def active_provider(self) -> ModelProvider:
        """Returns the default ModelProvider instance for general interaction."""
        if self._provider_type == "cloud" or self._routing_mode == RoutingMode.CLOUD.value:
            return self._cloud_provider
        return self._local_provider

    @property
    def developer_provider(self) -> ModelProvider:
        """Returns the ModelProvider instance dedicated to developer tasks."""
        if self._developer_provider_type == "cloud":
            return self._cloud_provider
        elif self._developer_provider_type == "local":
            return self._local_provider
        # When developer provider is auto, inspect last decision or default to local
        if self._last_decision and self._last_decision.provider == "cloud":
            return self._cloud_provider
        return self._local_provider

    def get_provider(self) -> ModelProvider:
        """Explicit getter returning the active general ModelProvider."""
        return self.active_provider

    def get_developer_provider(self) -> ModelProvider:
        """Explicit getter returning the developer task provider."""
        return self.developer_provider

    def get_provider_for_task(
        self,
        task: Any = None,
        plan: Any = None,
        **kwargs: Any,
    ) -> ModelProvider:
        """Routes the task/plan and returns the matching ModelProvider."""
        decision = self.route(task=task, plan=plan, is_developer_task=True, **kwargs)
        if decision.provider == "cloud":
            return self._cloud_provider
        return self._local_provider

    @property
    def model_name(self) -> str:
        """Returns the model identifier of the active provider."""
        return self.active_provider.model_name

    @property
    def developer_model_name(self) -> str:
        """Returns the model identifier of the developer provider."""
        if self._developer_provider_type == "cloud":
            return self._cloud_model
        elif self._developer_provider_type == "local":
            return self._local_model
        return self._local_model

    @property
    def provider_name(self) -> str:
        """Returns the formatted name of the active provider."""
        return f"ModelRouter({self._routing_mode.upper()}) -> {self.active_provider.provider_name}"

    @property
    def capabilities(self) -> ProviderCapabilities:
        """Exposes the capabilities of the active provider."""
        return self.active_provider.capabilities

    def check_connection(self) -> tuple[bool, str]:
        """Delegates readiness/connection check to the active provider."""
        if hasattr(self.active_provider, "check_connection"):
            return self.active_provider.check_connection()
        return True, "Provider connected."

    def evaluate_routing(self, messages_or_task: Any = None, **kwargs: Any) -> RoutingDecision:
        """Alias for route(). Evaluates routing decision given task, text, or messages."""
        if isinstance(messages_or_task, str):
            return self.route(messages=[{"role": "user", "content": messages_or_task}], **kwargs)
        return self.route(task=messages_or_task, **kwargs)

    def route(
        self,
        task: Any = None,
        plan: Any = None,
        messages: list[dict[str, str]] | str | None = None,
        local_only: bool = False,
        **kwargs: Any,
    ) -> RoutingDecision:
        """Executes the deterministic 5-tier routing policy and produces an auditable RoutingDecision.

        Precedence Hierarchy:
        1. Security restriction & Explicit LOCAL_ONLY (forces LOCAL)
        2. Cloud globally disabled check (VIDURA_CLOUD_ENABLED=false)
        3. Explicit application mode LOCAL/CLOUD (forces LOCAL or CLOUD with limit checks)
        4. Configured provider type (developer_provider_type or provider_type)
        5. Automatic complexity routing (AUTO: SIMPLE/MODERATE -> LOCAL, COMPLEX/CRITICAL -> CLOUD)
        6. Global default (LOCAL)

        Args:
            task: Optional DeveloperTask.
            plan: Optional DeveloperPlan.
            messages: Optional message history or query string.
            local_only: Explicit flag forcing offline local execution.
            **kwargs: Extra contextual arguments (is_developer_task, target_files, etc.).

        Returns:
            Structured, immutable RoutingDecision.
        """
        # Extract candidate files to inspect for protected paths
        candidate_files: list[Any] = []
        if "target_files" in kwargs and isinstance(kwargs["target_files"], list):
            candidate_files.extend(kwargs["target_files"])
        elif "target_file" in kwargs and kwargs["target_file"]:
            candidate_files.append(kwargs["target_file"])
        if task is not None and getattr(task, "target_files", None):
            candidate_files.extend(getattr(task, "target_files"))
        if plan is not None and getattr(plan, "relevant_files", None):
            candidate_files.extend(getattr(plan, "relevant_files"))

        has_sensitive_files = is_protected_file_target(candidate_files)

        # Extract content to inspect for sensitive data and credentials
        content_items: list[Any] = []
        if messages:
            content_items.append(messages)
        if task is not None:
            if getattr(task, "goal", None):
                content_items.append(task.goal)
            if getattr(task, "requested_change", None):
                content_items.append(task.requested_change)
            if getattr(task, "context", None):
                content_items.append(task.context)
        if plan is not None and getattr(plan, "planned_changes", None):
            content_items.append(plan.planned_changes)
        for k in ("code", "diff", "prompt", "patch"):
            if k in kwargs and kwargs[k]:
                content_items.append(kwargs[k])

        has_sensitive_content = contains_sensitive_data(content_items)
        is_security_restricted = has_sensitive_files or has_sensitive_content

        # Determine if task is marked local_only
        is_local_only = (
            bool(local_only)
            or bool(getattr(task, "local_only", False))
            or bool(kwargs.get("local_only", False))
        )

        msg_list: list[dict[str, str]] | None = None
        if isinstance(messages, str):
            msg_list = [{"role": "user", "content": messages}]
        elif isinstance(messages, list):
            msg_list = messages

        complexity, default_reason, signals = classify_complexity(
            task=task,
            plan=plan,
            messages=msg_list,
            **kwargs,
        )

        # ------------------------------------------------------------------
        # Tier 1: Explicit LOCAL_ONLY & Security restrictions
        # ------------------------------------------------------------------
        if is_local_only:
            decision = RoutingDecision(
                provider="local",
                model=self._local_model,
                complexity=complexity.value,
                reason_code=ReasonCode.LOCAL_ONLY_POLICY.value,
                task_type=signals.get("task_type"),
                routing_mode=self._routing_mode,
                confidence=1.0,
                signals_used=signals,
                local_only=True,
            )
            self._last_decision = decision
            logger.info(f"Routing Decision: Tier 1 (LOCAL_ONLY) -> {decision.provider} ({decision.model}) [{decision.reason_code}]")
            return decision

        if is_security_restricted:
            if self._routing_mode == RoutingMode.CLOUD.value and not self._fallback_enabled:
                logger.error("Sensitive data or protected file target detected with cloud routing and fallback disabled. Aborting.")
                raise CloudSecurityViolation("Sensitive data or credentials detected in task/messages; cloud transmission is blocked.")

            decision = RoutingDecision(
                provider="local",
                model=self._local_model,
                complexity=complexity.value,
                reason_code=ReasonCode.CLOUD_SECURITY_RESTRICTION.value,
                task_type=signals.get("task_type"),
                routing_mode=self._routing_mode,
                confidence=1.0,
                signals_used=signals,
                local_only=True,
                requested_provider="cloud" if self._routing_mode == RoutingMode.CLOUD.value else "local",
                requested_model=self._cloud_model if self._routing_mode == RoutingMode.CLOUD.value else self._local_model,
                actual_provider="local",
                actual_model=self._local_model,
                fallback_used=bool(self._routing_mode == RoutingMode.CLOUD.value),
                fallback_reason="Sensitive data or protected file target detected" if self._routing_mode == RoutingMode.CLOUD.value else None,
            )
            self._last_decision = decision
            logger.warning(f"Routing Decision: Tier 1 (SECURITY_RESTRICTION) -> {decision.provider} ({decision.model}) [{decision.reason_code}]")
            return decision

            logger.info(f"Routing Decision: Tier 1 (LOCAL_ONLY) -> {decision.provider} ({decision.model}) [{decision.reason_code}]")
            return decision

        # ------------------------------------------------------------------
        # Tier 2: Cloud globally disabled check (VIDURA_CLOUD_ENABLED=false)
        # ------------------------------------------------------------------
        if not self._cloud_enabled:
            if self._routing_mode == RoutingMode.CLOUD.value:
                if self._fallback_enabled:
                    decision = RoutingDecision(
                        provider="local",
                        model=self._local_model,
                        complexity=complexity.value,
                        reason_code=ReasonCode.CLOUD_DISABLED.value,
                        task_type=signals.get("task_type"),
                        routing_mode=self._routing_mode,
                        confidence=1.0,
                        signals_used=signals,
                        local_only=False,
                        requested_provider="cloud",
                        requested_model=self._cloud_model,
                        actual_provider="local",
                        actual_model=self._local_model,
                        fallback_used=True,
                        fallback_reason="Cloud provider is globally disabled via VIDURA_CLOUD_ENABLED=false",
                    )
                    self._last_decision = decision
                    logger.warning("Routing Decision: Tier 2 (Cloud disabled with fallback) -> local")
                    return decision
                else:
                    logger.error("Cloud provider is disabled via configuration and fallback is disabled.")
                    raise CloudDisabledError("Cloud model access is disabled via configuration (VIDURA_CLOUD_ENABLED=false).")
            elif self._routing_mode == RoutingMode.LOCAL.value:
                decision = RoutingDecision(
                    provider="local",
                    model=self._local_model,
                    complexity=complexity.value,
                    reason_code=ReasonCode.EXPLICIT_LOCAL.value,
                    task_type=signals.get("task_type"),
                    routing_mode=self._routing_mode,
                    confidence=1.0,
                    signals_used=signals,
                    local_only=False,
                )
                self._last_decision = decision
                return decision
            else:
                # AUTO mode with cloud disabled -> routes all tasks to LOCAL
                decision = RoutingDecision(
                    provider="local",
                    model=self._local_model,
                    complexity=complexity.value,
                    reason_code=ReasonCode.CLOUD_DISABLED.value,
                    task_type=signals.get("task_type"),
                    routing_mode=self._routing_mode,
                    confidence=1.0,
                    signals_used=signals,
                    local_only=False,
                )
                self._last_decision = decision
                logger.info("Routing Decision: Tier 2 (Cloud disabled in AUTO mode) -> local")
                return decision

        # Determine if developer task
        is_dev = (
            bool(kwargs.get("is_developer_task", False))
            or (kwargs.get("task_type") in ("developer", "coding", "planning"))
            or (task is not None and getattr(task, "is_development_task", True))
            or (plan is not None)
        )

        # Evaluate pre-request usage limits
        can_req, limit_reason = self._usage_tracker.check_can_request(is_developer=is_dev)

        # ------------------------------------------------------------------
        # Tier 3: Explicit application routing mode (LOCAL / CLOUD)
        # ------------------------------------------------------------------
        if self._routing_mode == RoutingMode.LOCAL.value:
            decision = RoutingDecision(
                provider="local",
                model=self._local_model,
                complexity=complexity.value,
                reason_code=ReasonCode.EXPLICIT_LOCAL.value,
                task_type=signals.get("task_type"),
                routing_mode=self._routing_mode,
                confidence=1.0,
                signals_used=signals,
                local_only=False,
            )
            self._last_decision = decision
            logger.info(f"Routing Decision: Tier 3 (EXPLICIT_LOCAL mode) -> {decision.provider} ({decision.model})")
            return decision

        if self._routing_mode == RoutingMode.CLOUD.value:
            if not can_req:
                if self._fallback_enabled:
                    decision = RoutingDecision(
                        provider="local",
                        model=self._local_model,
                        complexity=complexity.value,
                        reason_code=ReasonCode.CLOUD_LIMIT_REACHED.value,
                        task_type=signals.get("task_type"),
                        routing_mode=self._routing_mode,
                        confidence=1.0,
                        signals_used=signals,
                        local_only=False,
                        requested_provider="cloud",
                        requested_model=self._cloud_model,
                        actual_provider="local",
                        actual_model=self._local_model,
                        fallback_used=True,
                        fallback_reason=limit_reason,
                    )
                    self._last_decision = decision
                    logger.warning(f"Routing Decision: Tier 3 (Cloud limit reached with fallback) -> local ({limit_reason})")
                    return decision
                else:
                    raise CloudUsageLimitReached(f"Cloud request limit reached: {limit_reason}")

            decision = RoutingDecision(
                provider="cloud",
                model=self._cloud_model,
                complexity=complexity.value,
                reason_code=ReasonCode.EXPLICIT_CLOUD.value,
                task_type=signals.get("task_type"),
                routing_mode=self._routing_mode,
                confidence=1.0,
                signals_used=signals,
                local_only=False,
            )
            self._last_decision = decision
            logger.info(f"Routing Decision: Tier 3 (EXPLICIT_CLOUD mode) -> {decision.provider} ({decision.model})")
            return decision

        # ------------------------------------------------------------------
        # Tier 4: Configured provider overrides
        # ------------------------------------------------------------------
        if is_dev:
            if self._developer_provider_type == "local":
                decision = RoutingDecision(
                    provider="local",
                    model=self._local_model,
                    complexity=complexity.value,
                    reason_code=ReasonCode.EXPLICIT_LOCAL.value,
                    task_type=signals.get("task_type"),
                    routing_mode=self._routing_mode,
                    confidence=1.0,
                    signals_used=signals,
                    local_only=False,
                )
                self._last_decision = decision
                logger.info(f"Routing Decision: Tier 4 (Developer LOCAL override) -> {decision.provider}")
                return decision
            elif self._developer_provider_type == "cloud":
                if not can_req:
                    if self._fallback_enabled:
                        decision = RoutingDecision(
                            provider="local",
                            model=self._local_model,
                            complexity=complexity.value,
                            reason_code=ReasonCode.CLOUD_LIMIT_REACHED.value,
                            task_type=signals.get("task_type"),
                            routing_mode=self._routing_mode,
                            confidence=1.0,
                            signals_used=signals,
                            local_only=False,
                            requested_provider="cloud",
                            requested_model=self._cloud_model,
                            actual_provider="local",
                            actual_model=self._local_model,
                            fallback_used=True,
                            fallback_reason=limit_reason,
                        )
                        self._last_decision = decision
                        logger.warning(f"Routing Decision: Tier 4 (Developer CLOUD limit reached with fallback) -> local ({limit_reason})")
                        return decision
                    else:
                        raise CloudUsageLimitReached(f"Cloud request limit reached: {limit_reason}")

                decision = RoutingDecision(
                    provider="cloud",
                    model=self._cloud_model,
                    complexity=complexity.value,
                    reason_code=ReasonCode.EXPLICIT_CLOUD.value,
                    task_type=signals.get("task_type"),
                    routing_mode=self._routing_mode,
                    confidence=1.0,
                    signals_used=signals,
                    local_only=False,
                )
                self._last_decision = decision
                logger.info(f"Routing Decision: Tier 4 (Developer CLOUD override) -> {decision.provider}")
                return decision
        else:
            # General / non-developer task
            if self._provider_type == "cloud":
                if not can_req:
                    if self._fallback_enabled:
                        decision = RoutingDecision(
                            provider="local",
                            model=self._local_model,
                            complexity=complexity.value,
                            reason_code=ReasonCode.CLOUD_LIMIT_REACHED.value,
                            task_type=signals.get("task_type"),
                            routing_mode=self._routing_mode,
                            confidence=1.0,
                            signals_used=signals,
                            local_only=False,
                            requested_provider="cloud",
                            requested_model=self._cloud_model,
                            actual_provider="local",
                            actual_model=self._local_model,
                            fallback_used=True,
                            fallback_reason=limit_reason,
                        )
                        self._last_decision = decision
                        logger.warning(f"Routing Decision: Tier 4 (General CLOUD limit reached with fallback) -> local ({limit_reason})")
                        return decision
                    else:
                        raise CloudUsageLimitReached(f"Cloud request limit reached: {limit_reason}")

                decision = RoutingDecision(
                    provider="cloud",
                    model=self._cloud_model,
                    complexity=complexity.value,
                    reason_code=ReasonCode.EXPLICIT_CLOUD.value,
                    task_type=signals.get("task_type"),
                    routing_mode=self._routing_mode,
                    confidence=1.0,
                    signals_used=signals,
                    local_only=False,
                )
                self._last_decision = decision
                logger.info(f"Routing Decision: Tier 4 (General CLOUD override) -> {decision.provider}")
                return decision
            elif self._provider_type == "local":
                decision = RoutingDecision(
                    provider="local",
                    model=self._local_model,
                    complexity=complexity.value,
                    reason_code=ReasonCode.LOCAL_DEFAULT.value,
                    task_type=signals.get("task_type"),
                    routing_mode=self._routing_mode,
                    confidence=1.0,
                    signals_used=signals,
                    local_only=False,
                )
                self._last_decision = decision
                return decision

        # ------------------------------------------------------------------
        # Tier 5: Automatic complexity routing
        # ------------------------------------------------------------------
        if complexity in (TaskComplexity.COMPLEX, TaskComplexity.CRITICAL):
            if not can_req:
                target_provider = "local"
                target_model = self._local_model
                reason = ReasonCode.CLOUD_LIMIT_REACHED.value
            else:
                target_provider = "cloud"
                target_model = self._cloud_model
                reason = default_reason.value
        else:
            target_provider = "local"
            target_model = self._local_model
            reason = default_reason.value

        decision = RoutingDecision(
            provider=target_provider,
            model=target_model,
            complexity=complexity.value,
            reason_code=reason,
            task_type=signals.get("task_type"),
            routing_mode=self._routing_mode,
            confidence=1.0,
            signals_used=signals,
            local_only=False,
        )
        self._last_decision = decision
        logger.info(
            f"Routing Decision: Tier 5 (AUTO Complexity: {complexity.value}) -> "
            f"{decision.provider} ({decision.model}) [Reason: {decision.reason_code}]"
        )
        return decision

    def generate(self, messages: list[dict[str, str]] | str, **kwargs: Any) -> str:
        """Evaluates routing decision, enforces pre-request security & usage checks, and forwards text generation."""
        task = kwargs.pop("task", None)
        plan = kwargs.pop("plan", None)
        local_only = kwargs.pop("local_only", False)
        is_dev = kwargs.pop("is_developer_task", False) or (kwargs.get("task_type") in ("developer", "coding", "planning"))
        task_type = kwargs.pop("task_type", None)

        decision = self.route(
            task=task,
            plan=plan,
            messages=messages,
            local_only=local_only,
            is_developer_task=is_dev,
            task_type=task_type,
            **kwargs,
        )

        if decision.provider == "cloud":
            # --- Pre-Request Safety & Usage Checks before any network transmission ---
            # 1. Cloud globally enabled check
            if not self._cloud_enabled:
                if self._fallback_enabled:
                    logger.warning("Cloud is disabled. Falling back to local model.")
                    self._last_decision = RoutingDecision(
                        provider="cloud",
                        model=self._cloud_model,
                        complexity=decision.complexity,
                        reason_code=ReasonCode.CLOUD_DISABLED.value,
                        task_type=decision.task_type,
                        routing_mode=decision.routing_mode,
                        confidence=decision.confidence,
                        signals_used=decision.signals_used,
                        local_only=decision.local_only,
                        requested_provider="cloud",
                        requested_model=self._cloud_model,
                        actual_provider="local",
                        actual_model=self._local_model,
                        fallback_used=True,
                        fallback_reason="Cloud provider is globally disabled via configuration",
                    )
                    return self._local_provider.generate(messages, **kwargs)
                else:
                    raise CloudDisabledError("Cloud model access is disabled via configuration.")

            # 2. Sensitive content / credentials check
            target_files = kwargs.get("target_files")
            if is_protected_file_target(target_files) or contains_sensitive_data(messages):
                if self._fallback_enabled:
                    logger.warning("Sensitive data or credentials detected. Falling back to local model.")
                    self._last_decision = RoutingDecision(
                        provider="cloud",
                        model=self._cloud_model,
                        complexity=decision.complexity,
                        reason_code=ReasonCode.CLOUD_SECURITY_RESTRICTION.value,
                        task_type=decision.task_type,
                        routing_mode=decision.routing_mode,
                        confidence=decision.confidence,
                        signals_used=decision.signals_used,
                        local_only=True,
                        requested_provider="cloud",
                        requested_model=self._cloud_model,
                        actual_provider="local",
                        actual_model=self._local_model,
                        fallback_used=True,
                        fallback_reason="Sensitive data or credentials detected in request",
                    )
                    return self._local_provider.generate(messages, **kwargs)
                else:
                    raise CloudSecurityViolation("Sensitive data or credentials detected in prompt/files; cloud transmission blocked.")

            # 3. Context limit check
            est_tokens = estimate_tokens(messages)
            if self._max_context_tokens > 0 and est_tokens > self._max_context_tokens:
                if self._fallback_enabled:
                    logger.warning(
                        f"Context tokens ({est_tokens}) exceed cloud limit ({self._max_context_tokens}). "
                        f"Falling back to local model."
                    )
                    self._last_decision = RoutingDecision(
                        provider="cloud",
                        model=self._cloud_model,
                        complexity=decision.complexity,
                        reason_code=ReasonCode.CLOUD_CONTEXT_LIMIT_EXCEEDED.value,
                        task_type=decision.task_type,
                        routing_mode=decision.routing_mode,
                        confidence=decision.confidence,
                        signals_used=decision.signals_used,
                        local_only=decision.local_only,
                        requested_provider="cloud",
                        requested_model=self._cloud_model,
                        actual_provider="local",
                        actual_model=self._local_model,
                        fallback_used=True,
                        fallback_reason=f"Context tokens ({est_tokens}) exceed limit ({self._max_context_tokens})",
                    )
                    return self._local_provider.generate(messages, **kwargs)
                else:
                    raise CloudContextLimitExceeded(
                        f"Request context ({est_tokens} estimated tokens) exceeds cloud context limit ({self._max_context_tokens} tokens)."
                    )

            # 4. Usage limit check
            can_req, limit_reason = self._usage_tracker.check_can_request(is_developer=is_dev)
            if not can_req:
                if self._fallback_enabled:
                    logger.warning(f"Cloud request limit reached: {limit_reason}. Falling back to local model.")
                    self._last_decision = RoutingDecision(
                        provider="cloud",
                        model=self._cloud_model,
                        complexity=decision.complexity,
                        reason_code=ReasonCode.CLOUD_LIMIT_REACHED.value,
                        task_type=decision.task_type,
                        routing_mode=decision.routing_mode,
                        confidence=decision.confidence,
                        signals_used=decision.signals_used,
                        local_only=decision.local_only,
                        requested_provider="cloud",
                        requested_model=self._cloud_model,
                        actual_provider="local",
                        actual_model=self._local_model,
                        fallback_used=True,
                        fallback_reason=limit_reason,
                    )
                    return self._local_provider.generate(messages, **kwargs)
                else:
                    raise CloudUsageLimitReached(f"Cloud request limit reached: {limit_reason}")

            # Pre-request checks passed. Generate unique cloud request id and track attempt
            cloud_req_id = f"cloud_req_{uuid.uuid4().hex[:8]}"
            self._usage_tracker.record_cloud_attempt(is_developer=is_dev)

            try:
                logger.debug(
                    f"Executing generate() on cloud provider '{self._cloud_provider.provider_name}' ({self._cloud_provider.model_name}) "
                    f"for route: provider={decision.provider}, reason={decision.reason_code}, req_id={cloud_req_id}"
                )
                res = self._cloud_provider.generate(messages, **kwargs)
                in_tokens = estimate_tokens(messages)
                out_tokens = estimate_tokens(res)
                record = self._usage_tracker.record_cloud_success(
                    request_id=cloud_req_id,
                    model=self._cloud_model,
                    input_tokens=in_tokens,
                    output_tokens=out_tokens,
                    is_developer=is_dev,
                    task_type=task_type or decision.task_type,
                    complexity=decision.complexity,
                    fallback_used=False,
                )
                self._last_decision = RoutingDecision(
                    provider=decision.provider,
                    model=decision.model,
                    complexity=decision.complexity,
                    reason_code=decision.reason_code,
                    task_type=decision.task_type,
                    routing_mode=decision.routing_mode,
                    confidence=decision.confidence,
                    signals_used=decision.signals_used,
                    local_only=decision.local_only,
                    requested_provider=decision.requested_provider,
                    requested_model=decision.requested_model,
                    actual_provider=decision.actual_provider,
                    actual_model=decision.actual_model,
                    fallback_used=False,
                    cloud_request_id=cloud_req_id,
                    usage_metadata=record.to_dict(),
                )
                return res
            except Exception as cloud_err:
                record = self._usage_tracker.record_cloud_failure(
                    request_id=cloud_req_id,
                    model=self._cloud_model,
                    error_message=str(cloud_err),
                    is_developer=is_dev,
                    task_type=task_type or decision.task_type,
                    complexity=decision.complexity,
                )
                # Check fallback eligibility and configuration
                if not (self._fallback_enabled and is_fallback_eligible(cloud_err)):
                    logger.warning(
                        f"Cloud provider inference failed and fallback is not eligible/enabled "
                        f"(fallback_enabled={self._fallback_enabled}, error={cloud_err}): {cloud_err}"
                    )
                    raise cloud_err

                logger.warning(
                    f"Cloud inference failed with error: {cloud_err}. "
                    f"Triggering automatic local fallback to '{self._local_model}'."
                )
                self._usage_tracker.record_fallback(request_id=cloud_req_id)
                fallback_decision = RoutingDecision(
                    provider=decision.provider,
                    model=decision.model,
                    complexity=decision.complexity,
                    reason_code=decision.reason_code,
                    task_type=decision.task_type,
                    routing_mode=decision.routing_mode,
                    confidence=decision.confidence,
                    signals_used=decision.signals_used,
                    local_only=decision.local_only,
                    requested_provider="cloud",
                    requested_model=self._cloud_model,
                    actual_provider="local",
                    actual_model=self._local_model,
                    fallback_used=True,
                    fallback_reason=str(cloud_err),
                    cloud_error=str(cloud_err),
                    cloud_request_id=cloud_req_id,
                    usage_metadata=record.to_dict(),
                )
                self._last_decision = fallback_decision

                # Max 1 attempt on local provider, zero loops, no recursion
                try:
                    logger.info(
                        f"Executing fallback generate() on local provider '{self._local_provider.provider_name}' ({self._local_model})"
                    )
                    return self._local_provider.generate(messages, **kwargs)
                except Exception as local_err:
                    logger.error(
                        f"Fallback to local provider failed: {local_err}. "
                        f"Both cloud and local providers failed."
                    )
                    raise CombinedProviderError(
                        cloud_error=cloud_err,
                        local_error=local_err,
                    ) from local_err
        else:
            logger.debug(
                f"Executing generate() on local provider '{self._local_provider.provider_name}' ({self._local_provider.model_name}) "
                f"for route: provider={decision.provider}, reason={decision.reason_code}"
            )
            return self._local_provider.generate(messages, **kwargs)


    def chat(self, messages: list[dict[str, str]] | str, **kwargs: Any) -> str:
        """Standard chat interface delegating to generate."""
        return self.generate(messages, **kwargs)

    def __repr__(self) -> str:
        return (
            f"ModelRouter(routing_mode='{self._routing_mode}', "
            f"active_provider='{self._provider_type}', "
            f"developer_provider='{self._developer_provider_type}', "
            f"local_model='{self._local_model}', cloud_model='{self._cloud_model}')"
        )
