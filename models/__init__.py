"""VIDURA Models Package.

Defines model interface abstractions, local and cloud providers, capabilities,
exceptions, intelligent routing policies, and the deterministic ModelRouter.
"""

from models.base import BaseLLMProvider, ModelProvider, ProviderCapabilities
from models.errors import (
    ModelProviderError,
    ProviderUnavailable,
    ProviderConfigurationError,
    ProviderAuthenticationError,
    ProviderTimeout,
    ProviderRequestError,
    CombinedProviderError,
    UnsupportedCapability,
    CloudSecurityViolation,
    CloudUsageLimitReached,
    CloudContextLimitExceeded,
    CloudDisabledError,
)
from models.local import OllamaProvider, LocalProvider, OllamaProviderError
from models.cloud import OllamaCloudProvider
from models.security import is_protected_file_target, contains_sensitive_data, check_sensitive_data, redact_secrets
from models.usage import CloudUsageRecord, CloudUsageTracker, estimate_tokens
from models.routing import (
    TaskComplexity,
    RoutingMode,
    ReasonCode,
    RoutingDecision,
    classify_complexity,
    FailureCategory,
    classify_cloud_failure,
    is_fallback_eligible,
)
from models.router import (
    ModelRouter,
    ALLOWED_PROVIDERS,
    ALLOWED_ROUTING_MODES,
    ALLOWED_LOCAL_MODELS,
    ALLOWED_CLOUD_MODELS,
)

__all__ = [
    "BaseLLMProvider",
    "ModelProvider",
    "ProviderCapabilities",
    "OllamaProvider",
    "LocalProvider",
    "OllamaCloudProvider",
    "ModelRouter",
    "ALLOWED_PROVIDERS",
    "ALLOWED_ROUTING_MODES",
    "ALLOWED_LOCAL_MODELS",
    "ALLOWED_CLOUD_MODELS",
    "TaskComplexity",
    "RoutingMode",
    "ReasonCode",
    "RoutingDecision",
    "classify_complexity",
    "FailureCategory",
    "classify_cloud_failure",
    "is_fallback_eligible",
    "ModelProviderError",
    "ProviderUnavailable",
    "ProviderConfigurationError",
    "ProviderAuthenticationError",
    "ProviderTimeout",
    "ProviderRequestError",
    "CombinedProviderError",
    "UnsupportedCapability",
    "CloudSecurityViolation",
    "CloudUsageLimitReached",
    "CloudContextLimitExceeded",
    "CloudDisabledError",
    "is_protected_file_target",
    "contains_sensitive_data",
    "check_sensitive_data",
    "redact_secrets",
    "CloudUsageRecord",
    "CloudUsageTracker",
    "estimate_tokens",
    "OllamaProviderError",
]

