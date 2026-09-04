"""VIDURA Models Package.

Defines model interface abstractions, local and cloud providers, capabilities,
exceptions, and the deterministic ModelRouter.
"""

from models.base import BaseLLMProvider, ModelProvider, ProviderCapabilities
from models.errors import (
    ModelProviderError,
    ProviderUnavailable,
    ProviderConfigurationError,
    ProviderAuthenticationError,
    ProviderTimeout,
    ProviderRequestError,
    UnsupportedCapability,
)
from models.local import OllamaProvider, LocalProvider, OllamaProviderError
from models.cloud import OllamaCloudProvider
from models.router import (
    ModelRouter,
    ALLOWED_PROVIDERS,
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
    "ALLOWED_LOCAL_MODELS",
    "ALLOWED_CLOUD_MODELS",
    "ModelProviderError",
    "ProviderUnavailable",
    "ProviderConfigurationError",
    "ProviderAuthenticationError",
    "ProviderTimeout",
    "ProviderRequestError",
    "UnsupportedCapability",
    "OllamaProviderError",
]
