from abc import ABC, abstractmethod
from typing import Any


class BaseLLMProvider(ABC):
    """Abstract Base Class for LLM Providers in VIDURA.
    
    All specific model integrations (Local Ollama, Cloud, etc.) must implement
    this interface to keep model interaction decoupled from application logic.
    """

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Returns the configured model identifier."""
        ...

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Returns the provider name (e.g. 'Ollama (local)')."""
        ...

    @abstractmethod
    def generate(self, messages: list[dict[str, str]], **kwargs: Any) -> str:
        """Generates a text completion response given a conversation message history.

        Args:
            messages: A list of dicts with 'role' and 'content' keys.
            **kwargs: Additional provider-specific parameters.

        Returns:
            The generated response string from the model.
        """
        ...
