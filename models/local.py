import logging
from typing import Any
import ollama
from models.base import ModelProvider, ProviderCapabilities
from models.errors import ModelProviderError

logger = logging.getLogger("VIDURA.models.local")

DEFAULT_LOCAL_MODEL = "gemma4:e4b-it-qat"
DEFAULT_OLLAMA_HOST = "http://localhost:11434"


class OllamaProviderError(ModelProviderError):
    """Exception raised for errors encountered in the Ollama model provider."""
    pass


class OllamaProvider(ModelProvider):
    """Local LLM Provider utilizing Ollama."""

    def __init__(self, host: str = DEFAULT_OLLAMA_HOST, model: str = DEFAULT_LOCAL_MODEL) -> None:
        self._host = host.rstrip("/")
        self._model = model
        self._client = ollama.Client(host=self._host)
        self._capabilities = ProviderCapabilities(
            chat=True,
            tool_calling=True,
            vision=False,
            reasoning=False,
            coding=True,
            cloud=False,
            local=True,
        )

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def provider_name(self) -> str:
        return "Ollama (local)"

    @property
    def host(self) -> str:
        return self._host

    @property
    def capabilities(self) -> ProviderCapabilities:
        """Returns the capabilities of the local Ollama provider."""
        return self._capabilities

    def check_connection(self) -> tuple[bool, str]:
        """Verifies connection to Ollama server and availability of the target model.

        Returns:
            Tuple of (success: bool, status_message: str)
        """
        try:
            models_response = self._client.list()
            # ollama-python list() returns a ListResponse with a models attribute
            installed_models = []
            if hasattr(models_response, "models"):
                for m in models_response.models:
                    if hasattr(m, "model"):
                        installed_models.append(m.model)
                    elif isinstance(m, dict) and "name" in m:
                        installed_models.append(m["name"])
            elif isinstance(models_response, dict) and "models" in models_response:
                installed_models = [m.get("model") or m.get("name") for m in models_response["models"]]

            # Model match check (supporting tags like gemma4:e4b-it-qat or gemma4:e4b-it-qat:latest)
            matched = any(
                m == self._model or m.startswith(f"{self._model}:") or self._model.startswith(f"{m}:")
                for m in installed_models if m
            )

            if not matched:
                msg = (
                    f"Connected to Ollama at {self._host}, but configured model '{self._model}' "
                    f"was not found. Installed models: {installed_models or 'None'}"
                )
                logger.warning(msg)
                return False, msg

            return True, f"Connected to Ollama at {self._host} using model '{self._model}'."

        except Exception as err:
            msg = f"Cannot reach Ollama service at {self._host}. Ensure Ollama is running locally. Error: {err}"
            logger.error(msg)
            return False, msg

    def generate(self, messages: list[dict[str, str]], **kwargs: Any) -> str:
        """Sends a conversation history to local Ollama model and returns the response.

        Args:
            messages: List of dicts with 'role' and 'content'.
            **kwargs: Extra parameters passed to client.chat.

        Returns:
            Model response text.
        """
        if not messages:
            raise OllamaProviderError("Message history cannot be empty.")

        try:
            logger.debug(f"Sending chat request to {self._model} with {len(messages)} messages.")
            chat_kwargs = {k: v for k, v in kwargs.items() if k not in ("task_type", "is_developer_task")}
            response = self._client.chat(
                model=self._model,
                messages=messages,
                **chat_kwargs,
            )
            
            # Extract content from response
            if hasattr(response, "message") and hasattr(response.message, "content"):
                return response.message.content
            elif isinstance(response, dict) and "message" in response and "content" in response["message"]:
                return response["message"]["content"]
            else:
                raise OllamaProviderError(f"Unexpected response structure from Ollama: {response}")

        except ollama.ResponseError as err:
            error_msg = f"Ollama model error: {err.error} (Status code: {err.status_code})"
            logger.error(error_msg)
            raise OllamaProviderError(error_msg) from err
        except Exception as err:
            error_msg = f"Failed to communicate with local Ollama server at {self._host}: {err}"
            logger.error(error_msg)
            raise OllamaProviderError(error_msg) from err


# Clean alias for provider abstraction
LocalProvider = OllamaProvider
