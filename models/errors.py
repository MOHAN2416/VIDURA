"""VIDURA Model Provider Exception Hierarchy.

Defines structured exceptions for model provider errors, configuration issues,
authentication failures, timeouts, and unsupported capabilities.
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


class UnsupportedCapability(ModelProviderError):
    """Raised when a requested capability (e.g. vision, tool_calling) is not supported by the provider."""
    pass
