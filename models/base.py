from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ProviderCapabilities:
    """Structured capabilities exposed by a ModelProvider."""
    chat: bool = True
    tool_calling: bool = False
    vision: bool = False
    reasoning: bool = False
    coding: bool = True
    cloud: bool = False
    local: bool = True

    def to_dict(self) -> dict[str, bool]:
        """Returns dictionary representation of provider capabilities."""
        return {
            "chat": self.chat,
            "tool_calling": self.tool_calling,
            "vision": self.vision,
            "reasoning": self.reasoning,
            "coding": self.coding,
            "cloud": self.cloud,
            "local": self.local,
        }


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

    @property
    def capabilities(self) -> ProviderCapabilities:
        """Returns the supported capabilities of the provider."""
        return ProviderCapabilities()

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

    def chat(self, messages: list[dict[str, str]], **kwargs: Any) -> str:
        """Standard chat interface, delegating to generate.

        Args:
            messages: Conversation message list.
            **kwargs: Extra parameters passed to the model.

        Returns:
            Generated response string.
        """
        return self.generate(messages, **kwargs)


class ModelProvider(BaseLLMProvider):
    """Common provider abstraction for local and cloud models in VIDURA.

    Subclasses implement generate() and declare capabilities for their respective execution environment.
    """
    pass
