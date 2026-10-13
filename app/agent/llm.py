from abc import ABC, abstractmethod
import logging
import os
import re
import time
from typing import Any, Dict, List, Optional
import requests

from app.llm import (
    OLLAMA_MODEL,
    OLLAMA_TIMEOUT,
    OllamaConnectionError,
    OllamaError,
    OllamaModelNotFoundError,
    get_base_url,
)

logger = logging.getLogger(__name__)

# Defaults for Nebius Token Factory
DEFAULT_NEBIUS_BASE_URL = "https://api.tokenfactory.nebius.com/v1/"
DEFAULT_NEBIUS_MODEL = "nvidia/nemotron-3-super-120b-a12b"
DEFAULT_LLM_TIMEOUT = 30
DEFAULT_MAX_RETRIES = 3


def redact_secrets(text: str, secret: Optional[str]) -> str:
    """Safely redact secret tokens from strings, logs, or error messages."""
    if not secret or len(secret.strip()) == 0:
        return text
    clean_secret = secret.strip()
    return text.replace(clean_secret, "[REDACTED]")


def strip_reasoning(text: str) -> str:
    """
    Strips internal reasoning / chain-of-thought tokens (<think>...</think>)
    emitted by reasoning models (such as NVIDIA Nemotron) before returning
    to the user or passing to JSON parsers.
    """
    if not text:
        return ""
    # Strip closed <think>...</think> blocks
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    # Strip any trailing unclosed <think>... blocks
    cleaned = re.sub(r"<think>.*$", "", cleaned, flags=re.DOTALL | re.IGNORECASE)
    return cleaned.strip()


# ==============================================================================
# Exceptions
# ==============================================================================

class NebiusError(Exception):
    """Base exception for Nebius API interactions with sanitized messages."""
    pass


class NebiusAuthError(NebiusError):
    """Raised on authentication failures or missing API keys."""
    pass


class NebiusRateLimitError(NebiusError):
    """Raised when rate limits (HTTP 429) persist after retries."""
    pass


class NebiusTimeoutError(NebiusError):
    """Raised when an API request times out."""
    pass


# ==============================================================================
# Abstract Interface
# ==============================================================================

class LLMClient(ABC):
    """Abstract interface for LLM providers used across the assistant."""

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Name of the provider ('ollama', 'nebius', etc.)."""
        pass

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Active model identifier."""
        pass

    @abstractmethod
    def generate_text(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Generate a natural-language text response."""
        pass

    @abstractmethod
    def generate_json(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Generate a structured JSON response."""
        pass


# ==============================================================================
# Ollama Implementation
# ==============================================================================

class OllamaClient(LLMClient):
    """Ollama implementation of LLMClient reusing existing local configuration."""

    def __init__(
        self,
        model: Optional[str] = None,
        timeout: Optional[int] = None,
    ):
        self._model = model or os.getenv("OLLAMA_MODEL", OLLAMA_MODEL)
        self.timeout = timeout or int(os.getenv("LLM_TIMEOUT_SECONDS", str(OLLAMA_TIMEOUT)))

    @property
    def provider_name(self) -> str:
        return "ollama"

    @property
    def model_name(self) -> str:
        return self._model

    def _call(
        self,
        prompt: str,
        system: Optional[str] = None,
        json_format: bool = False,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
    ) -> str:
        base_url = get_base_url()
        url = f"{base_url}/api/generate"

        options_dict: Dict[str, Any] = {
            "temperature": temperature,
        }
        if max_tokens is not None:
            options_dict["num_predict"] = max_tokens

        payload: Dict[str, Any] = {
            "model": self._model,
            "prompt": prompt,
            "stream": False,
            "keep_alive": os.getenv("OLLAMA_KEEP_ALIVE", "30m"),
            "options": options_dict,
        }

        if system:
            payload["system"] = system

        if json_format:
            payload["format"] = "json"

        try:
            response = requests.post(url, json=payload, timeout=self.timeout)
        except requests.exceptions.ConnectionError as e:
            raise OllamaConnectionError(
                f"Could not connect to Ollama at {base_url}. Ensure Ollama is running."
            ) from e
        except requests.exceptions.Timeout as e:
            raise OllamaConnectionError(
                f"Ollama request timed out after {self.timeout} seconds."
            ) from e
        except requests.exceptions.RequestException as e:
            raise OllamaConnectionError(f"Ollama HTTP communication error: {str(e)}") from e

        if response.status_code == 404:
            raise OllamaModelNotFoundError(
                f"Model '{self._model}' not found in Ollama at {base_url}."
            )
        if response.status_code != 200:
            raise OllamaError(f"Ollama returned error status {response.status_code}: {response.text}")

        data = response.json()
        raw_output = data.get("response", "").strip()
        return strip_reasoning(raw_output)

    def generate_text(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
    ) -> str:
        return self._call(prompt=prompt, system=system, json_format=False, temperature=temperature, max_tokens=max_tokens)

    def generate_json(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
    ) -> str:
        return self._call(prompt=prompt, system=system, json_format=True, temperature=temperature, max_tokens=max_tokens)


# ==============================================================================
# Nebius Token Factory / NVIDIA Nemotron Implementation
# ==============================================================================

class NebiusClient(LLMClient):
    """
    Nebius Token Factory implementation of LLMClient.
    Connects to Nebius's OpenAI-compatible endpoint using standard requests.
    Enforces secret redaction, automatic retries with backoff, and reasoning stripping.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout: Optional[int] = None,
        max_retries: Optional[int] = None,
    ):
        raw_key = api_key if api_key is not None else os.getenv("NEBIUS_API_KEY", "")
        self._api_key = raw_key.strip()
        if not self._api_key:
            raise NebiusAuthError(
                "Nebius API key is missing. Please set the NEBIUS_API_KEY environment variable."
            )

        raw_url = base_url or os.getenv("NEBIUS_BASE_URL", DEFAULT_NEBIUS_BASE_URL)
        clean_url = raw_url.strip().rstrip("/")
        if not clean_url.endswith("/v1"):
            clean_url = f"{clean_url}/v1"
        self._base_url = clean_url

        self._model = model or os.getenv("NEBIUS_MODEL", DEFAULT_NEBIUS_MODEL)
        self.timeout = timeout or int(os.getenv("LLM_TIMEOUT_SECONDS", str(DEFAULT_LLM_TIMEOUT)))
        self.max_retries = max_retries or int(os.getenv("LLM_MAX_RETRIES", str(DEFAULT_MAX_RETRIES)))

    @property
    def provider_name(self) -> str:
        return "nebius"

    @property
    def model_name(self) -> str:
        return self._model

    def _sanitize(self, msg: str) -> str:
        """Sanitize any error message or string so API key is never exposed."""
        return redact_secrets(msg, self._api_key)

    def _call_chat_completions(
        self,
        prompt: str,
        system: Optional[str] = None,
        json_format: bool = False,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
    ) -> str:
        url = f"{self._base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

        messages: List[Dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        payload: Dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "temperature": temperature,
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens

        if json_format:
            payload["response_format"] = {"type": "json_object"}

        backoff = 1.0
        last_error: Optional[Exception] = None

        for attempt in range(1, self.max_retries + 1):
            try:
                response = requests.post(url, json=payload, headers=headers, timeout=self.timeout)
            except requests.exceptions.Timeout as e:
                sanitized_err = self._sanitize(str(e))
                last_error = NebiusTimeoutError(f"Nebius request timed out after {self.timeout}s: {sanitized_err}")
                if attempt < self.max_retries:
                    time.sleep(backoff)
                    backoff *= 2.0
                    continue
                raise last_error from None
            except requests.exceptions.RequestException as e:
                sanitized_err = self._sanitize(str(e))
                last_error = NebiusError(f"Nebius HTTP communication error: {sanitized_err}")
                if attempt < self.max_retries:
                    time.sleep(backoff)
                    backoff *= 2.0
                    continue
                raise last_error from None

            # Handle HTTP Status Codes
            if response.status_code == 200:
                try:
                    data = response.json()
                    choices = data.get("choices", [])
                    if not choices:
                        raise NebiusError("Nebius response returned empty choices array.")
                    content = choices[0].get("message", {}).get("content", "")
                    # Strip any reasoning (<think> tags) before returning
                    return strip_reasoning(content)
                except Exception as e:
                    sanitized_err = self._sanitize(str(e))
                    raise NebiusError(f"Failed to parse Nebius response content: {sanitized_err}") from None

            elif response.status_code in (401, 403):
                # Authentication error - do not retry
                err_text = self._sanitize(response.text)
                raise NebiusAuthError(
                    f"Nebius authentication failed (HTTP {response.status_code}): {err_text}"
                )

            elif response.status_code == 429:
                # Rate limit error - retry with backoff
                err_text = self._sanitize(response.text)
                last_error = NebiusRateLimitError(
                    f"Nebius rate limit exceeded (HTTP 429): {err_text}"
                )
                if attempt < self.max_retries:
                    time.sleep(backoff)
                    backoff *= 2.0
                    continue
                raise last_error from None

            elif 500 <= response.status_code < 600:
                # Server error - retry with backoff
                err_text = self._sanitize(response.text)
                last_error = NebiusError(
                    f"Nebius server error (HTTP {response.status_code}): {err_text}"
                )
                if attempt < self.max_retries:
                    time.sleep(backoff)
                    backoff *= 2.0
                    continue
                raise last_error from None

            else:
                err_text = self._sanitize(response.text)
                raise NebiusError(
                    f"Nebius returned unexpected HTTP status {response.status_code}: {err_text}"
                )

        if last_error:
            raise last_error
        raise NebiusError("Max retries exceeded communicating with Nebius.")

    def generate_text(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
    ) -> str:
        return self._call_chat_completions(
            prompt=prompt,
            system=system,
            json_format=False,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    def generate_json(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
    ) -> str:
        return self._call_chat_completions(
            prompt=prompt,
            system=system,
            json_format=True,
            temperature=temperature,
            max_tokens=max_tokens,
        )


class CountingLLMClient(LLMClient):
    """
    Counting wrapper around any LLMClient that records invocation counts
    and per-call latency into a RunContext.
    """

    def __init__(self, inner: LLMClient, run_context: Optional[Any] = None):
        self._inner = inner
        self._run_context = run_context

    @property
    def provider_name(self) -> str:
        return getattr(self._inner, "provider_name", "unknown")

    @property
    def model_name(self) -> str:
        return getattr(self._inner, "model_name", "unknown")

    def generate_text(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
    ) -> str:
        t0 = time.perf_counter()
        try:
            res = self._inner.generate_text(prompt=prompt, system=system, temperature=temperature, max_tokens=max_tokens)
        except TypeError:
            res = self._inner.generate_text(prompt=prompt, system=system, temperature=temperature)
        latency = round((time.perf_counter() - t0) * 1000, 2)
        if self._run_context is not None:
            self._run_context.record_llm_call(latency)
        return res

    def generate_json(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
    ) -> str:
        t0 = time.perf_counter()
        try:
            res = self._inner.generate_json(prompt=prompt, system=system, temperature=temperature, max_tokens=max_tokens)
        except TypeError:
            res = self._inner.generate_json(prompt=prompt, system=system, temperature=temperature)
        latency = round((time.perf_counter() - t0) * 1000, 2)
        if self._run_context is not None:
            self._run_context.record_llm_call(latency)
        return res

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


# ==============================================================================
# Provider Factory & Health
# ==============================================================================

def get_llm_client() -> LLMClient:
    """
    Factory function returning the configured LLMClient.
    If an active client is set in the current execution context, returns it.
    Otherwise selects provider via LLM_PROVIDER env variable ('ollama' or 'nebius').
    """
    try:
        from app.agent.executor import current_active_client
        active = current_active_client.get()
        if active is not None:
            return active
    except Exception:
        pass

    provider = os.getenv("LLM_PROVIDER", "ollama").strip().lower()

    if provider == "ollama":
        return OllamaClient()
    elif provider == "nebius":
        return NebiusClient()

    raise ValueError(
        f"Unsupported LLM provider '{provider}'. Supported providers: 'ollama', 'nebius'."
    )


def check_active_llm_health() -> Dict[str, Any]:
    """
    Additive health check for the active LLM provider.
    Never exposes API keys or credentials.
    """
    provider = os.getenv("LLM_PROVIDER", "ollama").strip().lower()
    if provider == "ollama":
        from app.llm import check_ollama_health
        ollama_status = check_ollama_health()
        return {
            "provider": "ollama",
            "model": ollama_status.get("model", OLLAMA_MODEL),
            "available": ollama_status.get("available", False),
            "details": ollama_status,
        }
    elif provider == "nebius":
        api_key = os.getenv("NEBIUS_API_KEY", "").strip()
        model = os.getenv("NEBIUS_MODEL", DEFAULT_NEBIUS_MODEL)
        base_url = os.getenv("NEBIUS_BASE_URL", DEFAULT_NEBIUS_BASE_URL).rstrip("/")
        if not api_key:
            return {
                "provider": "nebius",
                "model": model,
                "available": False,
                "error": "NEBIUS_API_KEY is not configured.",
            }
        try:
            # Check models endpoint using configured key
            headers = {"Authorization": f"Bearer {api_key}"}
            resp = requests.get(f"{base_url}/models", headers=headers, timeout=5)
            available = (resp.status_code == 200)
            return {
                "provider": "nebius",
                "model": model,
                "available": available,
                "status_code": resp.status_code,
            }
        except Exception as e:
            return {
                "provider": "nebius",
                "model": model,
                "available": False,
                "error": redact_secrets(str(e), api_key),
            }
    else:
        return {
            "provider": provider,
            "model": "unknown",
            "available": False,
            "error": f"Unknown provider '{provider}'",
        }
