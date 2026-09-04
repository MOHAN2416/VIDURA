"""VIDURA Ollama Cloud Provider.

Provides the ModelProvider interface for Ollama Cloud models, targeting gemma4:31b-cloud.
Connects via official Ollama SDK and HTTP client with authentication, configurable timeouts,
structured error mapping, response normalization, and strict secret protection.
"""
import json
import logging
import re
from typing import Any

import httpx
import ollama

from models.base import ModelProvider, ProviderCapabilities
from models.errors import (
    ModelProviderError,
    ProviderAuthenticationError,
    ProviderConfigurationError,
    ProviderRequestError,
    ProviderTimeout,
    ProviderUnavailable,
)

logger = logging.getLogger("VIDURA.models.cloud")

DEFAULT_CLOUD_MODEL = "gemma4:31b-cloud"
ALLOWED_CLOUD_MODELS = {DEFAULT_CLOUD_MODEL}
DEFAULT_CLOUD_ENDPOINT = "https://ollama.com"


def sanitize_error(error_msg: str, api_key: str | None = None) -> str:
    """Sanitizes error messages to ensure API keys and tokens are never leaked."""
    if not error_msg:
        return ""
    sanitized = str(error_msg)
    if api_key and api_key.strip():
        sanitized = sanitized.replace(api_key, "[REDACTED]")
    # Redact common Bearer token patterns
    sanitized = re.sub(r"(Bearer\s+)[A-Za-z0-9_\-\.]{8,}", r"\1[REDACTED]", sanitized, flags=re.IGNORECASE)
    sanitized = re.sub(r"(key[=:]\s*)[A-Za-z0-9_\-\.]{8,}", r"\1[REDACTED]", sanitized, flags=re.IGNORECASE)
    return sanitized


def normalize_endpoint(endpoint: str | None) -> str:
    """Normalizes the Ollama Cloud endpoint URL to avoid double /api/ paths."""
    if not endpoint or not endpoint.strip():
        return DEFAULT_CLOUD_ENDPOINT
    ep = endpoint.strip().rstrip("/")
    if ep.endswith("/api"):
        ep = ep[:-4].rstrip("/")
    return ep or DEFAULT_CLOUD_ENDPOINT


class OllamaCloudProvider(ModelProvider):
    """Cloud LLM Provider utilizing Ollama Cloud for gemma4:31b-cloud."""

    def __init__(
        self,
        model: str = DEFAULT_CLOUD_MODEL,
        api_key: str | None = None,
        endpoint: str | None = None,
        timeout: float = 60.0,
    ) -> None:
        if model not in ALLOWED_CLOUD_MODELS:
            raise ProviderConfigurationError(
                f"Invalid cloud model identifier: '{model}'. "
                f"Allowed cloud model identifier is: {sorted(ALLOWED_CLOUD_MODELS)}."
            )
        self._model = model
        self._api_key = api_key.strip() if api_key and api_key.strip() else None
        self._endpoint = normalize_endpoint(endpoint)
        self._timeout = float(timeout)
        self._capabilities = ProviderCapabilities(
            chat=True,
            tool_calling=True,
            vision=False,
            reasoning=True,
            coding=True,
            cloud=True,
            local=False,
        )

        headers: dict[str, str] = {}
        if self._api_key:
            headers["authorization"] = f"Bearer {self._api_key}"

        self._client = ollama.Client(
            host=self._endpoint,
            headers=headers,
            timeout=self._timeout,
        )

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def provider_name(self) -> str:
        return "Ollama (cloud)"

    @property
    def endpoint(self) -> str:
        return self._endpoint

    @property
    def timeout(self) -> float:
        return self._timeout

    @property
    def is_configured(self) -> bool:
        """Returns True if minimum required cloud credentials are present."""
        return bool(self._api_key)

    @property
    def capabilities(self) -> ProviderCapabilities:
        """Returns capabilities exposed by Ollama Cloud provider."""
        return self._capabilities

    def check_connection(self) -> tuple[bool, str]:
        """Checks connectivity and credential readiness of the cloud provider.

        Returns:
            Tuple of (success: bool, status_message: str)
        """
        if not self.is_configured:
            msg = "Cloud provider not configured: Missing OLLAMA_CLOUD_API_KEY or OLLAMA_API_KEY. Cloud execution requires explicit credentials."
            logger.warning(msg)
            return False, msg

        try:
            # Lightweight verification against the configured cloud endpoint
            self._client.list()
            return True, f"Connected to Ollama Cloud at {self._endpoint} ({self._model})."
        except ollama.ResponseError as err:
            sanitized = sanitize_error(err.error, self._api_key)
            if err.status_code == 401:
                return False, f"Cloud authentication failed (Status 401): {sanitized}"
            elif err.status_code == 404:
                return False, f"Cloud endpoint or model not found (Status 404): {sanitized}"
            return False, f"Cloud service error (Status {err.status_code}): {sanitized}"
        except httpx.TimeoutException:
            return False, f"Connection to Ollama Cloud timed out after {self._timeout}s."
        except Exception as err:
            sanitized = sanitize_error(str(err), self._api_key)
            return False, f"Cloud connection failed: {sanitized}"

    def generate(self, messages: list[dict[str, str]], **kwargs: Any) -> str:
        """Sends chat request to Ollama Cloud and normalizes the response.

        Args:
            messages: List of message dictionaries with 'role' and 'content'.
            **kwargs: Additional parameters passed to client.chat.

        Returns:
            Normalized model response string (text or tool call JSON).
        """
        if not messages:
            raise ProviderConfigurationError("Message history cannot be empty.")

        if not self.is_configured:
            msg = "CLOUD_PROVIDER_UNAVAILABLE: Cloud provider is not configured (missing OLLAMA_CLOUD_API_KEY or OLLAMA_API_KEY)."
            logger.error(msg)
            raise ProviderUnavailable(msg)

        try:
            logger.debug(f"Sending cloud chat request to {self._model} with {len(messages)} messages.")
            chat_kwargs = {k: v for k, v in kwargs.items() if k not in ("task_type", "is_developer_task")}
            response = self._client.chat(
                model=self._model,
                messages=messages,
                **chat_kwargs,
            )
            return self._normalize_response(response)

        except httpx.TimeoutException as err:
            msg = f"Cloud model request timed out after {self._timeout}s."
            logger.error(msg)
            raise ProviderTimeout(msg) from err

        except ollama.ResponseError as err:
            sanitized_msg = sanitize_error(err.error, self._api_key)
            if err.status_code == 401:
                error_msg = f"Cloud authentication failed (Status 401): {sanitized_msg}"
                logger.error(error_msg)
                raise ProviderAuthenticationError(error_msg) from err
            elif err.status_code in (404, 502, 503, 504):
                error_msg = f"CLOUD_PROVIDER_UNAVAILABLE: Cloud service or model unavailable (Status {err.status_code}): {sanitized_msg}"
                logger.error(error_msg)
                raise ProviderUnavailable(error_msg) from err
            else:
                error_msg = f"Cloud request error (Status {err.status_code}): {sanitized_msg}"
                logger.error(error_msg)
                raise ProviderRequestError(error_msg) from err

        except (httpx.NetworkError, httpx.ConnectError) as err:
            sanitized_msg = sanitize_error(str(err), self._api_key)
            error_msg = f"Network failure communicating with Ollama Cloud at {self._endpoint}: {sanitized_msg}"
            logger.error(error_msg)
            raise ProviderRequestError(error_msg) from err

        except (ProviderTimeout, ProviderAuthenticationError, ProviderUnavailable, ProviderRequestError, ProviderConfigurationError):
            raise

        except Exception as err:
            sanitized_msg = sanitize_error(str(err), self._api_key)
            error_msg = f"Unexpected cloud provider error: {sanitized_msg}"
            logger.error(error_msg)
            raise ModelProviderError(error_msg) from err

    def _normalize_response(self, response: Any) -> str:
        """Normalizes Ollama SDK response objects or dictionaries into a string response."""
        if response is None:
            raise ModelProviderError("Empty or null response received from Ollama Cloud.")

        msg_obj = getattr(response, "message", None)
        if msg_obj is None and isinstance(response, dict):
            msg_obj = response.get("message")

        if msg_obj is not None:
            # Check for structured tool calls
            tool_calls = getattr(msg_obj, "tool_calls", None)
            if tool_calls is None and isinstance(msg_obj, dict):
                tool_calls = msg_obj.get("tool_calls")

            if tool_calls:
                tc = tool_calls[0]
                fn_name = ""
                fn_args: Any = {}

                if hasattr(tc, "function"):
                    fn = tc.function
                    fn_name = getattr(fn, "name", "")
                    fn_args = getattr(fn, "arguments", {})
                elif isinstance(tc, dict) and "function" in tc:
                    fn = tc["function"]
                    fn_name = fn.get("name", "")
                    fn_args = fn.get("arguments", {})

                if fn_name:
                    if isinstance(fn_args, str):
                        try:
                            fn_args = json.loads(fn_args)
                        except Exception:
                            pass
                    return json.dumps({
                        "action": "tool_call",
                        "tool_name": fn_name,
                        "arguments": fn_args if isinstance(fn_args, dict) else {},
                    })

            # Check for message text content
            content = getattr(msg_obj, "content", None)
            if content is None and isinstance(msg_obj, dict):
                content = msg_obj.get("content")
            if content is not None:
                return str(content)

        # Fallback for direct dict with 'content'
        if isinstance(response, dict) and "content" in response:
            return str(response["content"])

        raise ModelProviderError(f"Unexpected response structure from Ollama Cloud: {response}")

    def __repr__(self) -> str:
        """Sanitized string representation ensuring no credential leakage."""
        key_status = "present" if self.is_configured else "missing"
        return f"OllamaCloudProvider(model='{self._model}', endpoint='{self._endpoint}', timeout={self._timeout}s, api_key=<{key_status}>)"
