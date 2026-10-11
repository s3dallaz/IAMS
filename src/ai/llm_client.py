"""
OmniRoute LLM client for the Phase 6 AI Assistant.

Talks to the project's existing OmniRoute provider, which exposes an
OpenAI-compatible `/chat/completions` endpoint (see `opencode.json`).

Design rules for this module:
- It performs ONLY outbound HTTP model calls. It never touches the database.
- The API key is read from the environment at call time and is never logged.
- Every failure mode is converted into a typed exception so the service layer
  can produce a friendly message instead of crashing Streamlit.
"""

import json
from typing import Any, Dict, List, Optional

import config
from src.utils.logger import get_logger

try:  # pragma: no cover - import guard
    import requests
except ImportError:  # pragma: no cover
    requests = None

logger = get_logger(__name__)


# ============================================================================
# Exceptions
# ============================================================================

class LLMError(Exception):
    """Base exception for LLM client failures."""
    pass


class LLMNotConfiguredError(LLMError):
    """Raised when no usable API key is available."""
    pass


class LLMUnavailableError(LLMError):
    """Raised when the provider cannot be reached (connection/timeout/DNS)."""
    pass


class LLMRequestFailedError(LLMError):
    """Raised when the provider returns an error status (auth/rate/5xx)."""
    pass


class LLMResponseFormatError(LLMError):
    """Raised when the provider responds successfully but not as expected."""
    pass


# ============================================================================
# Client
# ============================================================================

class OmniRouteClient:
    """
    Minimal, dependency-light client for the OpenAI-compatible OmniRoute API.

    Uses the `requests` library directly rather than an SDK so the project
    does not need to pin a large provider-specific package.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout: Optional[float] = None,
    ):
        """
        Initialize the client.

        Args:
            api_key: Optional explicit key. Defaults to the environment value.
            base_url: Optional base URL. Defaults to `config.OMNIROUTE_BASE_URL`.
            model: Optional model id. Defaults to `config.OMNIROUTE_MODEL`.
            timeout: Optional request timeout in seconds.
        """
        self.base_url = (base_url or config.OMNIROUTE_BASE_URL).rstrip("/")
        self.model = model or config.OMNIROUTE_MODEL
        self.timeout = timeout if timeout is not None else config.AI_REQUEST_TIMEOUT
        self._explicit_api_key = api_key

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    @property
    def api_key(self) -> str:
        """Resolve the API key live from the environment when not injected."""
        if self._explicit_api_key is not None:
            return self._explicit_api_key.strip()
        return config.get_ai_api_key()

    def is_configured(self) -> bool:
        """True when a real (non-placeholder) API key is available."""
        return config.is_ai_assistant_configured(self.api_key)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        json_mode: bool = False,
    ) -> str:
        """
        Send a chat completion request and return the assistant text.

        Args:
            messages: OpenAI-style message list (system/user/assistant).
            temperature: Optional override for sampling temperature.
            max_tokens: Optional override for max output tokens.
            json_mode: When True, ask the provider for strict JSON output.

        Returns:
            The assistant message content as a string.

        Raises:
            LLMNotConfiguredError: No usable API key.
            LLMUnavailableError: Provider unreachable.
            LLMRequestFailedError: Provider returned an error status.
            LLMResponseFormatError: Response shape was not usable.
        """
        if not self.is_configured():
            raise LLMNotConfiguredError(
                "AI Assistant is not configured. Please set OMNIROUTE_API_KEY "
                f"in your .env file ({config.ENV_FILE})."
            )

        if requests is None:  # pragma: no cover - dependency guard
            raise LLMUnavailableError(
                "The 'requests' package is required for AI features. "
                "Install it with: pip install requests"
            )

        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": config.AI_TEMPERATURE if temperature is None else temperature,
            "max_tokens": max_tokens or config.AI_MAX_TOKENS,
        }
        if json_mode:
            # OpenAI-compatible JSON mode. Providers that do not support it
            # still answer in prose, which we handle during parsing.
            payload["response_format"] = {"type": "json_object"}

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        url = f"{self.base_url}/chat/completions"

        try:
            response = requests.post(
                url,
                headers=headers,
                data=json.dumps(payload),
                timeout=self.timeout,
            )
        except requests.exceptions.Timeout as exc:
            logger.warning(f"AI provider request timed out after {self.timeout}s")
            raise LLMUnavailableError(
                f"The AI provider did not respond within {self.timeout:g} seconds."
            ) from exc
        except requests.exceptions.RequestException as exc:
            # Never log the exception payload verbatim: it can contain headers.
            logger.warning(f"AI provider connection failed: {type(exc).__name__}")
            raise LLMUnavailableError(
                f"Could not reach the AI provider at {self.base_url}."
            ) from exc

        if response.status_code in (401, 403):
            logger.warning(f"AI provider rejected credentials (HTTP {response.status_code})")
            raise LLMRequestFailedError(
                "The AI provider rejected the OMNIROUTE_API_KEY. Please check the key in your .env file."
            )
        if response.status_code == 429:
            logger.warning("AI provider rate limit reached")
            raise LLMRequestFailedError(
                "The AI provider is rate limiting requests. Please wait a moment and try again."
            )
        if response.status_code >= 400:
            logger.warning(f"AI provider returned HTTP {response.status_code}")
            raise LLMRequestFailedError(
                f"The AI provider returned an error (HTTP {response.status_code})."
            )

        return self._extract_content(response)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _extract_content(self, response: Any) -> str:
        """
        Pull the assistant text out of an OpenAI-compatible response.

        Raises:
            LLMResponseFormatError: If the payload is not shaped as expected.
        """
        try:
            data = response.json()
        except Exception as exc:
            raise LLMResponseFormatError(
                "The AI provider returned a response that could not be parsed as JSON."
            ) from exc

        if not isinstance(data, dict):
            raise LLMResponseFormatError("The AI provider returned an unexpected response shape.")

        # Some gateways return an error object with a 200 status.
        if "error" in data and "choices" not in data:
            raise LLMRequestFailedError("The AI provider reported an error.")

        choices = data.get("choices")
        if not isinstance(choices, list) or not choices:
            raise LLMResponseFormatError("The AI provider returned no completion choices.")

        first = choices[0]
        if not isinstance(first, dict):
            raise LLMResponseFormatError("The AI provider returned a malformed choice.")

        message = first.get("message")
        content = None
        if isinstance(message, dict):
            content = message.get("content")
        # Fall back to legacy `text` completions shape.
        if content is None:
            content = first.get("text")

        if not isinstance(content, str) or not content.strip():
            raise LLMResponseFormatError("The AI provider returned an empty response.")

        return content.strip()