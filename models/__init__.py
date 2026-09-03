"""VIDURA Models Package.

Defines model interface abstractions for local offline models and cloud models.
"""

from models.base import BaseLLMProvider
from models.local import OllamaProvider, OllamaProviderError

__all__ = ["BaseLLMProvider", "OllamaProvider", "OllamaProviderError"]
