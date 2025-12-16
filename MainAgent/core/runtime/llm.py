"""LLM wrapper for Groq API interactions with timeout, rate limiting, and proper error handling."""

from __future__ import annotations

import os
import time
import threading
from typing import Iterable, List, Optional, Callable
from functools import wraps
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError

from groq import Groq
from dotenv import load_dotenv

load_dotenv()

# Import custom exceptions
try:
    from ..exceptions import (
        LLMError,
        LLMTimeoutError,
        LLMRateLimitError,
        LLMResponseError,
        ConfigurationError,
    )
except ImportError:
    # Fallback for direct execution
    class LLMError(Exception):
        pass
    class LLMTimeoutError(LLMError):
        pass
    class LLMRateLimitError(LLMError):
        pass
    class LLMResponseError(LLMError):
        pass
    class ConfigurationError(Exception):
        pass

try:
    from ...config import CONFIG
except Exception:
    try:
        from MainAgent.config import CONFIG
    except Exception:
        CONFIG = {}

# Valid Groq models (as of 2024)
VALID_GROQ_MODELS = [
    "llama-3.3-70b-versatile",
    "llama-3.1-70b-versatile",
    "llama-3.1-8b-instant",
    "llama3-70b-8192",
    "llama3-8b-8192",
    "mixtral-8x7b-32768",
    "gemma2-9b-it",
    "gemma-7b-it",
]

DEFAULT_MODEL = "llama-3.3-70b-versatile"
DEFAULT_TIMEOUT = 120  # seconds
DEFAULT_RATE_LIMIT_RPM = 30  # requests per minute


class RateLimiter:
    """Token bucket rate limiter for API calls."""

    def __init__(self, requests_per_minute: int = DEFAULT_RATE_LIMIT_RPM):
        self.requests_per_minute = requests_per_minute
        self.tokens = requests_per_minute
        self.last_update = time.time()
        self._lock = threading.Lock()

    def acquire(self, timeout: float = 60.0) -> bool:
        """Acquire a token, blocking if necessary.

        Returns True if token acquired, False if timeout exceeded.
        """
        deadline = time.time() + timeout

        while time.time() < deadline:
            with self._lock:
                self._refill()
                if self.tokens >= 1:
                    self.tokens -= 1
                    return True

            # Wait before retrying
            time.sleep(0.1)

        return False

    def _refill(self):
        """Refill tokens based on elapsed time."""
        now = time.time()
        elapsed = now - self.last_update

        # Add tokens based on time elapsed
        tokens_to_add = elapsed * (self.requests_per_minute / 60.0)
        self.tokens = min(self.requests_per_minute, self.tokens + tokens_to_add)
        self.last_update = now

    def wait_time(self) -> float:
        """Get estimated wait time until next token available."""
        with self._lock:
            self._refill()
            if self.tokens >= 1:
                return 0.0
            # Calculate time until we have 1 token
            tokens_needed = 1 - self.tokens
            return tokens_needed * (60.0 / self.requests_per_minute)


# Global rate limiter instance
_global_rate_limiter: Optional[RateLimiter] = None
_rate_limiter_lock = threading.Lock()


def get_rate_limiter(rpm: int = DEFAULT_RATE_LIMIT_RPM) -> RateLimiter:
    """Get or create global rate limiter."""
    global _global_rate_limiter
    with _rate_limiter_lock:
        if _global_rate_limiter is None:
            _global_rate_limiter = RateLimiter(rpm)
        return _global_rate_limiter


class GroqService:
    """Thin wrapper around the Groq chat completion client with streaming, timeout, and rate limiting."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        timeout: float = DEFAULT_TIMEOUT,
        rate_limit_rpm: int = DEFAULT_RATE_LIMIT_RPM,
    ) -> None:
        # Try in order: explicit parameter > environment variable > config file
        self.api_key = (
            api_key
            or os.getenv("GROQ_API_KEY")
            or CONFIG.get("groq_api_key")
        )
        if not self.api_key:
            raise ConfigurationError(
                "groq_api_key",
                "Groq API key not configured. Set GROQ_API_KEY env var or pass api_key parameter."
            )

        # Get model from config or use default
        configured_model = (
            model
            or os.getenv("GROQ_MODEL")
            or CONFIG.get("groq_model")
        )

        # Validate and set model
        if configured_model and configured_model in VALID_GROQ_MODELS:
            self.model = configured_model
        else:
            if configured_model:
                print(f"[WARN] Model '{configured_model}' not in known models, using default: {DEFAULT_MODEL}")
            self.model = DEFAULT_MODEL

        self.timeout = timeout
        self.rate_limiter = get_rate_limiter(rate_limit_rpm)
        self._client = Groq(api_key=self.api_key)
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="llm_")

    def _check_rate_limit(self):
        """Check and wait for rate limit."""
        if not self.rate_limiter.acquire(timeout=30.0):
            wait_time = self.rate_limiter.wait_time()
            raise LLMRateLimitError(retry_after=wait_time)

    def stream_completion(
        self,
        messages: List[dict],
        *,
        max_tokens: int = 2048,
        temperature: float = 0.7,
        top_p: float = 1.0,
        stop: Optional[Iterable[str]] = None,
        timeout: Optional[float] = None,
    ) -> str:
        """Create a chat completion request and stream the result to stdout.

        Returns the accumulated response string.
        Raises LLMTimeoutError if timeout exceeded.
        Raises LLMRateLimitError if rate limit exceeded.
        """
        effective_timeout = timeout or self.timeout

        # Check rate limit first
        self._check_rate_limit()

        def _do_completion():
            completion = self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                stream=True,
                stop=stop,
            )

            full_response = ""
            for chunk in completion:
                if chunk.choices and chunk.choices[0].delta.content:
                    text = chunk.choices[0].delta.content
                    print(text, end="", flush=True)
                    full_response += text

            print("\n")
            return full_response

        try:
            future = self._executor.submit(_do_completion)
            return future.result(timeout=effective_timeout)
        except FuturesTimeoutError:
            raise LLMTimeoutError(effective_timeout, f"LLM call timed out after {effective_timeout}s")
        except Exception as exc:
            error_msg = str(exc).lower()
            if "rate" in error_msg or "429" in str(exc):
                raise LLMRateLimitError()
            print(f"\n[ERROR] Stream completion failed: {exc}\n")
            raise LLMError(f"Stream completion failed: {exc}")

    def completion(
        self,
        messages: List[dict],
        *,
        max_tokens: int = 2048,
        temperature: float = 0.7,
        top_p: float = 1.0,
        stop: Optional[Iterable[str]] = None,
        timeout: Optional[float] = None,
    ) -> str:
        """Create a non-streaming chat completion request.

        Returns the response string.
        Raises LLMTimeoutError if timeout exceeded.
        """
        effective_timeout = timeout or self.timeout

        # Check rate limit first
        self._check_rate_limit()

        def _do_completion():
            completion = self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                stream=False,
                stop=stop,
            )

            if completion.choices and completion.choices[0].message.content:
                return completion.choices[0].message.content
            return ""

        try:
            future = self._executor.submit(_do_completion)
            return future.result(timeout=effective_timeout)
        except FuturesTimeoutError:
            raise LLMTimeoutError(effective_timeout, f"LLM call timed out after {effective_timeout}s")
        except Exception as exc:
            error_msg = str(exc).lower()
            if "rate" in error_msg or "429" in str(exc):
                raise LLMRateLimitError()
            print(f"\n[ERROR] Completion failed: {exc}\n")
            raise LLMError(f"Completion failed: {exc}")

    def close(self):
        """Clean up resources."""
        self._executor.shutdown(wait=False)


def llm_call(
    prompt: str,
    *,
    service: Optional[GroqService] = None,
    max_tokens: int = 2048,
    temperature: float = 0.7,
    retries: int = 3,
    retry_delay: float = 1.0,
    timeout: float = DEFAULT_TIMEOUT,
) -> str:
    """Convenience helper that sends a single-turn prompt to the Groq API.

    Args:
        prompt: The user prompt to send
        service: Optional pre-configured GroqService
        max_tokens: Maximum tokens in response
        temperature: Sampling temperature (0.0-1.0)
        retries: Number of retry attempts on failure
        retry_delay: Base delay between retries (exponential backoff)
        timeout: Timeout in seconds for the LLM call

    Returns:
        The LLM response string, or empty string on failure

    Raises:
        LLMTimeoutError: If all retries timeout
        LLMRateLimitError: If rate limit exceeded after retries
    """
    groq_service = service or GroqService(timeout=timeout)
    last_exception = None

    for attempt in range(retries):
        try:
            return groq_service.stream_completion(
                [{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature,
                timeout=timeout,
            )
        except LLMRateLimitError as exc:
            last_exception = exc
            wait_time = exc.retry_after or (retry_delay * (2 ** attempt))
            if attempt < retries - 1:
                print(f"\n[WARN] Rate limit hit (attempt {attempt + 1}/{retries})")
                print(f"       Waiting {wait_time:.1f}s before retry...\n")
                time.sleep(wait_time)
            else:
                print(f"\n[ERROR] Rate limit exceeded after {retries} attempts\n")
                raise
        except LLMTimeoutError as exc:
            last_exception = exc
            if attempt < retries - 1:
                wait_time = retry_delay * (2 ** attempt)
                print(f"\n[WARN] Timeout (attempt {attempt + 1}/{retries}): {exc}")
                print(f"       Retrying in {wait_time}s...\n")
                time.sleep(wait_time)
            else:
                print(f"\n[ERROR] Timeout after {retries} attempts\n")
                raise
        except Exception as exc:
            last_exception = exc
            if attempt < retries - 1:
                wait_time = retry_delay * (2 ** attempt)
                print(f"\n[WARN] API call failed (attempt {attempt + 1}/{retries}): {exc}")
                print(f"       Retrying in {wait_time}s...\n")
                time.sleep(wait_time)
            else:
                print(f"\n[ERROR] API Error after {retries} attempts: {exc}\n")
                return ""

    return ""


def llm_call_with_system(
    system_prompt: str,
    user_prompt: str,
    *,
    service: Optional[GroqService] = None,
    max_tokens: int = 2048,
    temperature: float = 0.7,
    timeout: float = DEFAULT_TIMEOUT,
) -> str:
    """Send a prompt with a system message.

    Args:
        system_prompt: The system/instruction prompt
        user_prompt: The user's request
        service: Optional pre-configured GroqService
        max_tokens: Maximum tokens in response
        temperature: Sampling temperature
        timeout: Timeout in seconds

    Returns:
        The LLM response string

    Raises:
        LLMTimeoutError: If timeout exceeded
        LLMError: If completion fails
    """
    groq_service = service or GroqService(timeout=timeout)

    return groq_service.stream_completion(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        max_tokens=max_tokens,
        temperature=temperature,
        timeout=timeout,
    )


def llm_call_with_timeout(
    prompt: str,
    timeout_seconds: float,
    **kwargs
) -> str:
    """Convenience wrapper with explicit timeout parameter.

    Args:
        prompt: The prompt to send
        timeout_seconds: Maximum time to wait
        **kwargs: Additional arguments passed to llm_call

    Returns:
        LLM response string

    Raises:
        LLMTimeoutError: If timeout exceeded
    """
    return llm_call(prompt, timeout=timeout_seconds, **kwargs)
