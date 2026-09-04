"""VIDURA Model Router.

Selects and routes requests to the configured ModelProvider (Local or Cloud).
Enforces strict allowlists for providers and model identifiers.
Prevents dynamic provider changes by the model and ensures deterministic execution.
"""
import logging
from typing import Any

from config import Config, load_config
from models.base import ModelProvider, ProviderCapabilities
from models.local import LocalProvider, DEFAULT_LOCAL_MODEL, DEFAULT_OLLAMA_HOST
from models.cloud import OllamaCloudProvider, DEFAULT_CLOUD_MODEL
from models.errors import ProviderConfigurationError

logger = logging.getLogger("VIDURA.models.router")

ALLOWED_PROVIDERS = {"local", "cloud"}
ALLOWED_LOCAL_MODELS = {"gemma4:e4b-it-qat"}
ALLOWED_CLOUD_MODELS = {"gemma4:31b-cloud"}


class ModelRouter(ModelProvider):
    """Deterministic application-level Model Router for VIDURA.

    Routes model interactions to either LocalProvider or OllamaCloudProvider.
    The active provider is determined strictly by application configuration,
    never by model output or prompt-level instructions.
    """

    def __init__(
        self,
        config: Config | None = None,
        provider_type: str | None = None,
        developer_provider_type: str | None = None,
        local_provider: ModelProvider | None = None,
        cloud_provider: ModelProvider | None = None,
        local_model: str | None = None,
        cloud_model: str | None = None,
    ) -> None:
        self._config = config or load_config()

        # Determine target provider type
        raw_provider = provider_type or getattr(self._config, "vidura_model_provider", None) or getattr(self._config, "model_provider", "local")
        self._provider_type = str(raw_provider).lower().strip()

        if self._provider_type not in ALLOWED_PROVIDERS:
            raise ProviderConfigurationError(
                f"Unsupported model provider: '{self._provider_type}'. "
                f"Supported providers are: {sorted(ALLOWED_PROVIDERS)}."
            )

        # Determine model identifiers
        self._local_model = local_model or getattr(self._config, "vidura_local_model", None) or getattr(self._config, "ollama_model", DEFAULT_LOCAL_MODEL)
        self._cloud_model = cloud_model or getattr(self._config, "vidura_cloud_model", None) or getattr(self._config, "ollama_cloud_model", None) or DEFAULT_CLOUD_MODEL

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
        raw_dev_provider = developer_provider_type or getattr(self._config, "vidura_developer_model_provider", None) or self._provider_type
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

        logger.info(
            f"Initialized ModelRouter: active_provider='{self._provider_type}', "
            f"developer_provider='{self._developer_provider_type}', "
            f"local_model='{self._local_model}', cloud_model='{self._cloud_model}'"
        )

    @property
    def provider_type(self) -> str:
        """Returns the active provider type ('local' or 'cloud')."""
        return self._provider_type

    @property
    def developer_provider_type(self) -> str:
        """Returns the configured developer provider type ('local' or 'cloud')."""
        return self._developer_provider_type

    @property
    def active_provider(self) -> ModelProvider:
        """Returns the currently active ModelProvider instance for general interaction."""
        if self._provider_type == "cloud":
            return self._cloud_provider
        return self._local_provider

    @property
    def developer_provider(self) -> ModelProvider:
        """Returns the ModelProvider instance dedicated to developer tasks."""
        if self._developer_provider_type == "cloud":
            return self._cloud_provider
        return self._local_provider

    def get_provider(self) -> ModelProvider:
        """Explicit getter returning the active ModelProvider."""
        return self.active_provider

    def get_developer_provider(self) -> ModelProvider:
        """Explicit getter returning the developer task provider."""
        return self.developer_provider

    @property
    def model_name(self) -> str:
        """Returns the model identifier of the active provider."""
        return self.active_provider.model_name

    @property
    def developer_model_name(self) -> str:
        """Returns the model identifier of the developer provider."""
        return self.developer_provider.model_name

    @property
    def provider_name(self) -> str:
        """Returns the formatted name of the active provider."""
        return f"ModelRouter({self._provider_type.upper()}) -> {self.active_provider.provider_name}"

    @property
    def capabilities(self) -> ProviderCapabilities:
        """Exposes the capabilities of the active provider."""
        return self.active_provider.capabilities

    def check_connection(self) -> tuple[bool, str]:
        """Delegates readiness/connection check to the active provider."""
        if hasattr(self.active_provider, "check_connection"):
            return self.active_provider.check_connection()
        return True, "Provider connected."

    def generate(self, messages: list[dict[str, str]], **kwargs: Any) -> str:
        """Routes text generation request to the appropriate provider (developer vs general)."""
        is_dev = kwargs.pop("is_developer_task", False) or (kwargs.pop("task_type", None) in ("developer", "coding", "planning"))
        target_provider = self.developer_provider if is_dev else self.active_provider
        logger.debug(f"Routing generate() to provider: '{target_provider.provider_name}' ({target_provider.model_name})")
        return target_provider.generate(messages, **kwargs)

    def chat(self, messages: list[dict[str, str]], **kwargs: Any) -> str:
        """Routes chat request to the appropriate provider (developer vs general)."""
        is_dev = kwargs.pop("is_developer_task", False) or (kwargs.pop("task_type", None) in ("developer", "coding", "planning"))
        target_provider = self.developer_provider if is_dev else self.active_provider
        return target_provider.chat(messages, **kwargs)

    def __repr__(self) -> str:
        return (
            f"ModelRouter(active_provider='{self._provider_type}' ({self.model_name}), "
            f"developer_provider='{self._developer_provider_type}' ({self.developer_model_name}))"
        )
