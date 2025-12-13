"""
Resilience patterns for LLM calls and external services.

Provides:
- Circuit breaker pattern for fault tolerance
- Exponential backoff retry with jitter
- Rate limiting with token bucket
- Request timeout handling
"""

from __future__ import annotations

import time
import random
import logging
import threading
from enum import Enum
from typing import Callable, TypeVar, Generic, Optional, Any, Dict
from dataclasses import dataclass, field
from functools import wraps
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

T = TypeVar('T')


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, rejecting requests
    HALF_OPEN = "half_open"  # Testing if service recovered


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker."""
    failure_threshold: int = 5      # Failures before opening
    success_threshold: int = 2      # Successes in half-open to close
    timeout: float = 30.0           # Seconds before trying half-open
    half_open_max_calls: int = 3    # Max calls in half-open state


@dataclass
class CircuitBreakerStats:
    """Statistics for circuit breaker."""
    state: CircuitState = CircuitState.CLOSED
    failure_count: int = 0
    success_count: int = 0
    last_failure_time: Optional[float] = None
    last_success_time: Optional[float] = None
    total_failures: int = 0
    total_successes: int = 0
    total_rejections: int = 0
    state_changes: int = 0


class CircuitBreakerError(Exception):
    """Raised when circuit breaker is open."""
    def __init__(self, message: str, retry_after: Optional[float] = None):
        super().__init__(message)
        self.retry_after = retry_after


class CircuitBreaker:
    """
    Circuit breaker pattern implementation.

    Prevents cascading failures by stopping requests to failing services.

    Usage:
        breaker = CircuitBreaker("llm_api")

        @breaker
        def call_llm(prompt):
            return api.generate(prompt)

        # Or use directly:
        result = breaker.call(lambda: api.generate(prompt))
    """

    def __init__(
        self,
        name: str,
        config: Optional[CircuitBreakerConfig] = None,
    ):
        self.name = name
        self.config = config or CircuitBreakerConfig()
        self.stats = CircuitBreakerStats()
        self._lock = threading.RLock()
        self._half_open_calls = 0

    def __call__(self, func: Callable[..., T]) -> Callable[..., T]:
        """Decorator to wrap a function with circuit breaker."""
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            return self.call(lambda: func(*args, **kwargs))
        return wrapper

    def call(self, func: Callable[[], T]) -> T:
        """Execute function with circuit breaker protection."""
        with self._lock:
            self._check_state_transition()

            if self.stats.state == CircuitState.OPEN:
                self.stats.total_rejections += 1
                retry_after = self._time_until_half_open()
                raise CircuitBreakerError(
                    f"Circuit breaker '{self.name}' is OPEN. "
                    f"Retry after {retry_after:.1f}s",
                    retry_after=retry_after
                )

            if self.stats.state == CircuitState.HALF_OPEN:
                if self._half_open_calls >= self.config.half_open_max_calls:
                    raise CircuitBreakerError(
                        f"Circuit breaker '{self.name}' is HALF_OPEN with max calls reached"
                    )
                self._half_open_calls += 1

        try:
            result = func()
            self._record_success()
            return result
        except Exception as e:
            self._record_failure(e)
            raise

    def _check_state_transition(self) -> None:
        """Check if state should transition based on timeout."""
        if self.stats.state == CircuitState.OPEN:
            if self.stats.last_failure_time:
                elapsed = time.time() - self.stats.last_failure_time
                if elapsed >= self.config.timeout:
                    self._transition_to(CircuitState.HALF_OPEN)

    def _time_until_half_open(self) -> float:
        """Calculate time until circuit can try half-open."""
        if self.stats.last_failure_time is None:
            return 0.0
        elapsed = time.time() - self.stats.last_failure_time
        return max(0.0, self.config.timeout - elapsed)

    def _record_success(self) -> None:
        """Record a successful call."""
        with self._lock:
            self.stats.success_count += 1
            self.stats.total_successes += 1
            self.stats.last_success_time = time.time()

            if self.stats.state == CircuitState.HALF_OPEN:
                if self.stats.success_count >= self.config.success_threshold:
                    self._transition_to(CircuitState.CLOSED)
            elif self.stats.state == CircuitState.CLOSED:
                # Reset failure count on success
                self.stats.failure_count = 0

    def _record_failure(self, error: Exception) -> None:
        """Record a failed call."""
        with self._lock:
            self.stats.failure_count += 1
            self.stats.total_failures += 1
            self.stats.last_failure_time = time.time()

            logger.warning(
                f"Circuit breaker '{self.name}' recorded failure: {error}"
            )

            if self.stats.state == CircuitState.HALF_OPEN:
                # Any failure in half-open goes back to open
                self._transition_to(CircuitState.OPEN)
            elif self.stats.state == CircuitState.CLOSED:
                if self.stats.failure_count >= self.config.failure_threshold:
                    self._transition_to(CircuitState.OPEN)

    def _transition_to(self, new_state: CircuitState) -> None:
        """Transition to a new state."""
        old_state = self.stats.state
        self.stats.state = new_state
        self.stats.state_changes += 1

        # Reset counters on state change
        self.stats.failure_count = 0
        self.stats.success_count = 0
        self._half_open_calls = 0

        logger.info(
            f"Circuit breaker '{self.name}' transitioned: "
            f"{old_state.value} -> {new_state.value}"
        )

    def reset(self) -> None:
        """Manually reset the circuit breaker to closed state."""
        with self._lock:
            self._transition_to(CircuitState.CLOSED)
            self.stats.failure_count = 0
            self.stats.success_count = 0

    def get_stats(self) -> Dict[str, Any]:
        """Get current statistics."""
        with self._lock:
            return {
                "name": self.name,
                "state": self.stats.state.value,
                "failure_count": self.stats.failure_count,
                "success_count": self.stats.success_count,
                "total_failures": self.stats.total_failures,
                "total_successes": self.stats.total_successes,
                "total_rejections": self.stats.total_rejections,
                "state_changes": self.stats.state_changes,
            }


@dataclass
class RetryConfig:
    """Configuration for retry policy."""
    max_retries: int = 3
    base_delay: float = 1.0         # Base delay in seconds
    max_delay: float = 60.0         # Maximum delay cap
    exponential_base: float = 2.0   # Exponential backoff base
    jitter: float = 0.1             # Random jitter factor (0-1)
    retryable_exceptions: tuple = (Exception,)  # Exceptions to retry


def calculate_backoff(
    attempt: int,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    jitter: float = 0.1
) -> float:
    """
    Calculate exponential backoff with jitter.

    Args:
        attempt: Current attempt number (0-indexed)
        base_delay: Base delay in seconds
        max_delay: Maximum delay cap
        exponential_base: Exponential backoff base
        jitter: Random jitter factor (0-1)

    Returns:
        Delay in seconds
    """
    delay = base_delay * (exponential_base ** attempt)
    delay = min(delay, max_delay)

    # Add jitter
    if jitter > 0:
        jitter_range = delay * jitter
        delay += random.uniform(-jitter_range, jitter_range)

    return max(0.0, delay)


def retry_with_backoff(
    config: Optional[RetryConfig] = None,
) -> Callable:
    """
    Decorator for retry with exponential backoff.

    Usage:
        @retry_with_backoff()
        def call_api():
            return api.request()

        @retry_with_backoff(RetryConfig(max_retries=5))
        def call_llm(prompt):
            return llm.generate(prompt)
    """
    cfg = config or RetryConfig()

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            last_exception = None

            for attempt in range(cfg.max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except cfg.retryable_exceptions as e:
                    last_exception = e

                    if attempt < cfg.max_retries:
                        delay = calculate_backoff(
                            attempt,
                            cfg.base_delay,
                            cfg.max_delay,
                            cfg.exponential_base,
                            cfg.jitter
                        )
                        logger.warning(
                            f"Retry {attempt + 1}/{cfg.max_retries} for {func.__name__} "
                            f"after {delay:.2f}s due to: {e}"
                        )
                        time.sleep(delay)
                    else:
                        logger.error(
                            f"All {cfg.max_retries} retries exhausted for {func.__name__}"
                        )

            raise last_exception

        return wrapper
    return decorator


class TokenBucket:
    """
    Token bucket rate limiter.

    Usage:
        limiter = TokenBucket(rate=10, capacity=20)  # 10 req/sec, burst of 20

        if limiter.acquire():
            make_request()
        else:
            wait_or_reject()
    """

    def __init__(self, rate: float, capacity: float):
        """
        Initialize token bucket.

        Args:
            rate: Tokens added per second
            capacity: Maximum tokens in bucket
        """
        self.rate = rate
        self.capacity = capacity
        self._tokens = capacity
        self._last_update = time.time()
        self._lock = threading.Lock()

    def acquire(self, tokens: float = 1.0, blocking: bool = False) -> bool:
        """
        Try to acquire tokens.

        Args:
            tokens: Number of tokens to acquire
            blocking: If True, wait until tokens available

        Returns:
            True if tokens acquired, False otherwise
        """
        with self._lock:
            self._refill()

            if self._tokens >= tokens:
                self._tokens -= tokens
                return True

            if blocking:
                # Calculate wait time
                needed = tokens - self._tokens
                wait_time = needed / self.rate
                time.sleep(wait_time)
                self._refill()
                self._tokens -= tokens
                return True

            return False

    def _refill(self) -> None:
        """Refill tokens based on elapsed time."""
        now = time.time()
        elapsed = now - self._last_update
        self._tokens = min(
            self.capacity,
            self._tokens + elapsed * self.rate
        )
        self._last_update = now

    @property
    def available_tokens(self) -> float:
        """Get current available tokens."""
        with self._lock:
            self._refill()
            return self._tokens


class ResilientLLMClient:
    """
    Wrapper for LLM clients with built-in resilience patterns.

    Combines circuit breaker, retry with backoff, and rate limiting.

    Usage:
        client = ResilientLLMClient(
            name="groq",
            call_func=groq_client.generate,
            rate_limit=10,  # 10 requests per second
        )

        response = client.call(prompt="Hello")
    """

    def __init__(
        self,
        name: str,
        call_func: Callable,
        circuit_config: Optional[CircuitBreakerConfig] = None,
        retry_config: Optional[RetryConfig] = None,
        rate_limit: Optional[float] = None,
        rate_capacity: Optional[float] = None,
    ):
        self.name = name
        self._call_func = call_func

        # Initialize components
        self._circuit_breaker = CircuitBreaker(
            name=f"{name}_circuit",
            config=circuit_config or CircuitBreakerConfig()
        )

        self._retry_config = retry_config or RetryConfig()

        self._rate_limiter: Optional[TokenBucket] = None
        if rate_limit:
            self._rate_limiter = TokenBucket(
                rate=rate_limit,
                capacity=rate_capacity or rate_limit * 2
            )

    def call(self, *args, **kwargs) -> Any:
        """
        Make a resilient call to the LLM.

        Applies: rate limiting -> circuit breaker -> retry
        """
        # Rate limiting
        if self._rate_limiter:
            if not self._rate_limiter.acquire(blocking=True):
                raise RuntimeError("Rate limit exceeded")

        # Circuit breaker + retry
        @self._circuit_breaker
        @retry_with_backoff(self._retry_config)
        def _inner_call():
            return self._call_func(*args, **kwargs)

        return _inner_call()

    def get_stats(self) -> Dict[str, Any]:
        """Get resilience statistics."""
        stats = self._circuit_breaker.get_stats()
        if self._rate_limiter:
            stats["rate_limiter_tokens"] = self._rate_limiter.available_tokens
        return stats

    def reset(self) -> None:
        """Reset circuit breaker."""
        self._circuit_breaker.reset()


# Global circuit breakers for common services
_circuit_breakers: Dict[str, CircuitBreaker] = {}
_cb_lock = threading.Lock()


def get_circuit_breaker(
    name: str,
    config: Optional[CircuitBreakerConfig] = None
) -> CircuitBreaker:
    """Get or create a named circuit breaker."""
    global _circuit_breakers

    with _cb_lock:
        if name not in _circuit_breakers:
            _circuit_breakers[name] = CircuitBreaker(name, config)
        return _circuit_breakers[name]


def reset_all_circuit_breakers() -> None:
    """Reset all circuit breakers."""
    with _cb_lock:
        for cb in _circuit_breakers.values():
            cb.reset()
