"""VIDURA Model Provider Exception Hierarchy.

Defines structured exceptions for model provider errors, configuration issues,
authentication failures, timeouts, combined fallback failures, and unsupported capabilities.
"""


class ModelProviderError(Exception):
    """Base exception for all model provider errors in VIDURA."""
    pass


class ProviderUnavailable(ModelProviderError):
    """Raised when the requested model provider or model endpoint cannot be reached or is not configured."""
    pass


class ProviderConfigurationError(ModelProviderError):
    """Raised when provider configuration is invalid, missing required values, or specifies unauthorized models/providers."""
    pass


class ProviderAuthenticationError(ModelProviderError):
    """Raised when authentication with the provider fails (e.g. invalid or expired API credentials)."""
    pass


class ProviderTimeout(ModelProviderError):
    """Raised when a request to the model provider exceeds the configured timeout threshold."""
    pass


class ProviderRequestError(ModelProviderError):
    """Raised when a request to the provider fails due to network, transport, or HTTP errors."""
    pass


class CombinedProviderError(ModelProviderError):
    """Raised when the primary cloud provider fails and the fallback local provider also fails."""

    def __init__(
        self,
        cloud_error: Exception,
        local_error: Exception,
        message: str | None = None,
    ) -> None:
        self.cloud_error = cloud_error
        self.local_error = local_error
        msg = (
            message
            or f"Cloud provider failed: {cloud_error}; Fallback local provider also failed: {local_error}"
        )
        super().__init__(msg)


class UnsupportedCapability(ModelProviderError):
    """Raised when a requested capability (e.g. vision, tool_calling) is not supported by the provider."""
    pass


class CloudSecurityViolation(ModelProviderError):
    """Raised when a cloud request violates security policy (e.g. sensitive data or protected file detected)."""
    pass


class CloudUsageLimitReached(ModelProviderError):
    """Raised when configured cloud request limits, developer request limits, or budget are exceeded."""
    pass


class CloudContextLimitExceeded(ModelProviderError):
    """Raised when cloud request context size exceeds configured token or character boundaries."""
    pass


class CloudDisabledError(ModelProviderError):
    """Raised when cloud inference is explicitly requested but cloud is globally disabled (VIDURA_CLOUD_ENABLED=false)."""
    pass

