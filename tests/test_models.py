from unittest.mock import MagicMock, patch
import pytest
import ollama
from models import OllamaProvider, OllamaProviderError


def test_ollama_provider_initialization() -> None:
    """Verifies that OllamaProvider initializes with configured host and model."""
    provider = OllamaProvider(host="http://localhost:11434", model="gemma4:e4b-it-qat")
    assert provider.host == "http://localhost:11434"
    assert provider.model_name == "gemma4:e4b-it-qat"
    assert provider.provider_name == "Ollama (local)"


@patch("ollama.Client")
def test_ollama_provider_generate_success(mock_client_cls: MagicMock) -> None:
    """Verifies generate() returns model response content when client succeeds."""
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.message.content = "Hello! I am VIDURA, your local assistant."
    mock_client.chat.return_value = mock_response
    mock_client_cls.return_value = mock_client

    provider = OllamaProvider(host="http://localhost:11434", model="gemma4:e4b-it-qat")
    messages = [{"role": "user", "content": "Hello"}]
    
    result = provider.generate(messages)
    
    assert result == "Hello! I am VIDURA, your local assistant."
    mock_client.chat.assert_called_once_with(
        model="gemma4:e4b-it-qat",
        messages=messages
    )


@patch("ollama.Client")
def test_ollama_provider_connection_failure(mock_client_cls: MagicMock) -> None:
    """Verifies connection failure throws OllamaProviderError cleanly."""
    mock_client = MagicMock()
    mock_client.chat.side_effect = Exception("Connection refused")
    mock_client_cls.return_value = mock_client

    provider = OllamaProvider(host="http://localhost:11434", model="gemma4:e4b-it-qat")
    messages = [{"role": "user", "content": "Hello"}]
    
    with pytest.raises(OllamaProviderError) as exc_info:
        provider.generate(messages)
    
    assert "Failed to communicate with local Ollama server" in str(exc_info.value)


def test_ollama_provider_empty_messages() -> None:
    """Verifies generating with empty message list raises OllamaProviderError."""
    provider = OllamaProvider()
    with pytest.raises(OllamaProviderError) as exc_info:
        provider.generate([])
    assert "Message history cannot be empty" in str(exc_info.value)
