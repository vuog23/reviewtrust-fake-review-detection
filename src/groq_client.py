"""Shared Groq Cloud client for the reasoning and image-transcription stages.

Both stages talk to the same multimodal model (``qwen/qwen3.6-27b`` by default),
so they share one client: one place that knows how the key is resolved, how
failures are classified, and how the model's thinking output is suppressed.

The client is built lazily. The app must start, serve ``/api/health``, and load
the local classifier and NER models even when no API key is configured -- only
the calls that actually need Groq should fail, and they should say why.
"""

from __future__ import annotations

import logging
import os
import re
from typing import Any


LOGGER = logging.getLogger(__name__)

# Qwen3.6 emits its chain of thought inline in `content` unless it is told not
# to. The system prompt forbids exposing that to end users, and `reasoning_format`
# already suppresses it, but this is a preview model whose defaults can change --
# so strip it here too rather than trust the server.
_THINK_BLOCK = re.compile(r"<think\b[^>]*>.*?</think>", re.DOTALL | re.IGNORECASE)
_UNCLOSED_THINK = re.compile(r"<think\b[^>]*>.*\Z", re.DOTALL | re.IGNORECASE)


class GroqUnavailableError(RuntimeError):
    """Raised when Groq is unreachable, unauthenticated, or not configured."""


class GroqClient:
    """Thin wrapper over Groq's chat-completions endpoint."""

    def __init__(
        self,
        *,
        model: str,
        api_key: str | None = None,
        api_key_env: str = "GROQ_API_KEY",
        base_url: str | None = None,
        timeout: float = 120.0,
        max_retries: int = 3,
        reasoning_effort: str | None = "none",
        reasoning_format: str | None = "hidden",
        client: Any = None,
    ) -> None:
        self.model = model
        self.api_key_env = api_key_env
        self.base_url = base_url
        self.timeout = timeout
        self.max_retries = max_retries
        self.reasoning_effort = reasoning_effort
        self.reasoning_format = reasoning_format
        self._api_key = api_key
        self._client = client

    @property
    def api_key(self) -> str | None:
        return self._api_key or os.environ.get(self.api_key_env) or None

    @property
    def is_configured(self) -> bool:
        return bool(self.api_key) or self._client is not None

    def client(self) -> Any:
        """Build the SDK client on first use so a missing key never blocks startup."""
        if self._client is not None:
            return self._client

        api_key = self.api_key
        if not api_key:
            raise GroqUnavailableError(
                f"{self.api_key_env} is not set, so the Groq model cannot be reached. "
                f"Set {self.api_key_env} in the environment and restart the server."
            )
        try:
            from groq import Groq
        except ImportError as exc:  # pragma: no cover - dependency is declared
            raise GroqUnavailableError(
                "The 'groq' package is not installed. Run: pip install -r requirements.txt"
            ) from exc

        arguments: dict[str, Any] = {
            "api_key": api_key,
            "timeout": self.timeout,
            "max_retries": self.max_retries,
        }
        if self.base_url:
            arguments["base_url"] = self.base_url
        self._client = Groq(**arguments)
        return self._client

    def complete(
        self,
        messages: list[dict[str, Any]],
        *,
        temperature: float | None = None,
        top_p: float | None = None,
        max_completion_tokens: int | None = None,
    ) -> str:
        """Return the assistant reply with any thinking output removed."""
        request: dict[str, Any] = {"model": self.model, "messages": messages}
        if temperature is not None:
            request["temperature"] = temperature
        if top_p is not None:
            request["top_p"] = top_p
        if max_completion_tokens is not None:
            request["max_completion_tokens"] = max_completion_tokens
        if self.reasoning_effort:
            request["reasoning_effort"] = self.reasoning_effort
        if self.reasoning_format:
            request["reasoning_format"] = self.reasoning_format

        client = self.client()
        try:
            completion = client.chat.completions.create(**request)
        except GroqUnavailableError:
            raise
        except Exception as exc:
            # The provider's own explanation is the only thing that makes an
            # unexpected status debuggable, and it must not travel to the browser:
            # the route returns str(exc) as the 503 detail.
            LOGGER.warning("Groq request to %s failed: %s: %s", self.model, type(exc).__name__, exc)
            raise _as_unavailable(exc, self.model) from exc

        choice = completion.choices[0]
        # Groq bills and throttles per token per minute, and the SDK absorbs a 429 by
        # retrying, which shows up as a slow request rather than an error. Logging the
        # spend makes that visible instead of mysterious.
        usage = getattr(completion, "usage", None)
        if usage is not None:
            LOGGER.info(
                "Groq %s used %s prompt + %s completion tokens",
                self.model,
                getattr(usage, "prompt_tokens", "?"),
                getattr(usage, "completion_tokens", "?"),
            )
        if getattr(choice, "finish_reason", None) == "length":
            LOGGER.warning(
                "Groq model %s hit max_completion_tokens=%s and its reply was cut short",
                self.model,
                max_completion_tokens,
            )
        return strip_reasoning(choice.message.content or "")


def strip_reasoning(text: str) -> str:
    """Remove ``<think>`` blocks, including one left unclosed by a truncated reply."""
    text = _THINK_BLOCK.sub("", text)
    text = _UNCLOSED_THINK.sub("", text)
    return text.strip()


def _as_unavailable(exc: Exception, model: str) -> GroqUnavailableError:
    """Turn an SDK exception into one error type with a message worth showing."""
    name = type(exc).__name__
    if name == "AuthenticationError":
        return GroqUnavailableError("Groq rejected the API key. Check that it is current and has access.")
    if name == "PermissionDeniedError":
        return GroqUnavailableError(f"This Groq key is not allowed to use {model}.")
    if name == "NotFoundError":
        return GroqUnavailableError(
            f"Groq does not serve a model named {model}. Preview models are retired without notice; "
            "check https://console.groq.com/docs/models for the current id."
        )
    if name == "RateLimitError":
        return GroqUnavailableError("Groq rate limit reached. Wait a moment and try again.")
    if name in {"APIConnectionError", "APITimeoutError"}:
        return GroqUnavailableError("Groq could not be reached. Check the network connection.")
    status = getattr(exc, "status_code", None)
    if status is not None and status >= 500:
        # Preview models get capacity-limited: Groq answers 503 "currently over
        # capacity" after the SDK has already backed off and retried.
        return GroqUnavailableError(
            f"{model} is temporarily unavailable on Groq (HTTP {status}). "
            "This is a preview model and can be capacity limited; try again in a moment."
        )
    if status is not None:
        return GroqUnavailableError(f"Groq returned HTTP {status} for {model}.")
    return GroqUnavailableError(f"The Groq request failed: {exc}")
