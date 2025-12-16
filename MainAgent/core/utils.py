# ============================================================================
# FILE: utils.py
# Utility modules for progress reporting, logging, metrics, and caching
# ============================================================================

from __future__ import annotations

import sys
import time
import hashlib
import json
import logging
import threading
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from functools import wraps
from collections import OrderedDict
import atexit

logger = logging.getLogger(__name__)


# ============================================================================
# PROGRESS REPORTING
# ============================================================================

class ProgressReporter:
    """Reports progress of long-running operations.

    Features:
    - Stage-based progress tracking
    - ETA calculation
    - Callback support for UI updates
    - Thread-safe operations
    """

    def __init__(
        self,
        total_stages: int = 1,
        callback: Optional[Callable[[dict], None]] = None,
        show_output: bool = True
    ):
        """Initialize progress reporter.

        Args:
            total_stages: Total number of stages in the operation
            callback: Optional callback for progress updates
            show_output: Whether to print progress to stdout
        """
        self.total_stages = total_stages
        self.callback = callback
        self.show_output = show_output

        self._current_stage = 0
        self._current_stage_name = ""
        self._stage_progress = 0.0
        self._start_time = time.time()
        self._stage_start_time = time.time()
        self._stage_times: List[float] = []
        self._lock = threading.Lock()
        self._cancelled = False
        self._messages: List[str] = []

    def start_stage(self, stage_name: str, stage_number: Optional[int] = None):
        """Start a new stage.

        Args:
            stage_name: Name of the stage
            stage_number: Optional explicit stage number
        """
        with self._lock:
            if self._cancelled:
                return

            # Record previous stage time
            if self._current_stage > 0:
                self._stage_times.append(time.time() - self._stage_start_time)

            self._current_stage = stage_number if stage_number else self._current_stage + 1
            self._current_stage_name = stage_name
            self._stage_progress = 0.0
            self._stage_start_time = time.time()

            self._report()

    def update_progress(self, progress: float, message: str = ""):
        """Update progress within current stage.

        Args:
            progress: Progress value between 0.0 and 1.0
            message: Optional progress message
        """
        with self._lock:
            if self._cancelled:
                return

            self._stage_progress = max(0.0, min(1.0, progress))
            if message:
                self._messages.append(message)
            self._report()

    def complete_stage(self, message: str = ""):
        """Mark current stage as complete."""
        with self._lock:
            if self._cancelled:
                return

            self._stage_progress = 1.0
            self._stage_times.append(time.time() - self._stage_start_time)
            if message:
                self._messages.append(message)
            self._report()

    def cancel(self):
        """Cancel the operation."""
        with self._lock:
            self._cancelled = True
            if self.show_output:
                print("\n[CANCELLED] Operation cancelled by user")

    @property
    def is_cancelled(self) -> bool:
        """Check if operation was cancelled."""
        return self._cancelled

    def _report(self):
        """Report current progress."""
        overall_progress = (self._current_stage - 1 + self._stage_progress) / self.total_stages
        eta = self._calculate_eta()

        progress_data = {
            "overall_progress": overall_progress,
            "current_stage": self._current_stage,
            "total_stages": self.total_stages,
            "stage_name": self._current_stage_name,
            "stage_progress": self._stage_progress,
            "elapsed_seconds": time.time() - self._start_time,
            "eta_seconds": eta,
            "messages": self._messages[-5:],  # Last 5 messages
            "cancelled": self._cancelled
        }

        if self.callback:
            try:
                self.callback(progress_data)
            except Exception as e:
                logger.warning(f"Progress callback error: {e}")

        if self.show_output:
            self._print_progress(progress_data)

    def _print_progress(self, data: dict):
        """Print progress to stdout."""
        pct = data["overall_progress"] * 100
        stage = f"[{data['current_stage']}/{data['total_stages']}]"
        bar_width = 30
        filled = int(bar_width * data["overall_progress"])
        bar = "█" * filled + "░" * (bar_width - filled)

        eta_str = ""
        if data["eta_seconds"] and data["eta_seconds"] > 0:
            eta_min = int(data["eta_seconds"] / 60)
            eta_sec = int(data["eta_seconds"] % 60)
            eta_str = f" ETA: {eta_min}m{eta_sec}s"

        print(f"\r{stage} {bar} {pct:.1f}%{eta_str} - {data['stage_name']}", end="", flush=True)

        if data["stage_progress"] >= 1.0:
            print()  # New line when stage completes

    def _calculate_eta(self) -> Optional[float]:
        """Calculate estimated time remaining."""
        if not self._stage_times or self._current_stage == 0:
            return None

        avg_stage_time = sum(self._stage_times) / len(self._stage_times)
        remaining_stages = self.total_stages - self._current_stage
        current_stage_remaining = (1.0 - self._stage_progress) * avg_stage_time

        return remaining_stages * avg_stage_time + current_stage_remaining

    def get_summary(self) -> dict:
        """Get execution summary."""
        return {
            "total_time": time.time() - self._start_time,
            "stages_completed": self._current_stage,
            "stage_times": self._stage_times,
            "cancelled": self._cancelled
        }


# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================

def configure_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    format_string: Optional[str] = None,
    include_timestamp: bool = True,
    include_module: bool = True
):
    """Configure logging for the application.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional file path for log output
        format_string: Custom format string
        include_timestamp: Include timestamp in log messages
        include_module: Include module name in log messages
    """
    # Build format string
    if format_string is None:
        parts = []
        if include_timestamp:
            parts.append("%(asctime)s")
        parts.append("%(levelname)s")
        if include_module:
            parts.append("%(name)s")
        parts.append("%(message)s")
        format_string = " | ".join(parts)

    # Get numeric level
    numeric_level = getattr(logging, level.upper(), logging.INFO)

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)

    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Create formatter
    formatter = logging.Formatter(format_string, datefmt="%Y-%m-%d %H:%M:%S")

    # Add console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # Add file handler if specified
    if log_file:
        try:
            Path(log_file).parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(log_file, encoding='utf-8')
            file_handler.setLevel(numeric_level)
            file_handler.setFormatter(formatter)
            root_logger.addHandler(file_handler)
        except Exception as e:
            logger.warning(f"Failed to configure file logging: {e}")

    # Reduce noise from third-party libraries
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)


# ============================================================================
# METRICS / TELEMETRY
# ============================================================================

@dataclass
class MetricPoint:
    """A single metric data point."""
    name: str
    value: float
    timestamp: datetime = field(default_factory=datetime.now)
    tags: Dict[str, str] = field(default_factory=dict)


class MetricsCollector:
    """Collects and reports metrics.

    Features:
    - Counter, gauge, and histogram metrics
    - Tag-based filtering
    - Periodic flush to storage
    - Thread-safe operations
    """

    def __init__(self, flush_interval: float = 60.0, storage_path: Optional[str] = None):
        """Initialize metrics collector.

        Args:
            flush_interval: Seconds between automatic flushes
            storage_path: Optional path to persist metrics
        """
        self._counters: Dict[str, float] = {}
        self._gauges: Dict[str, float] = {}
        self._histograms: Dict[str, List[float]] = {}
        self._timings: Dict[str, List[float]] = {}
        self._lock = threading.Lock()
        self._points: List[MetricPoint] = []

        self.storage_path = Path(storage_path) if storage_path else None
        self.flush_interval = flush_interval
        self._last_flush = time.time()

        atexit.register(self.flush)

    def increment(self, name: str, value: float = 1.0, tags: Optional[Dict[str, str]] = None):
        """Increment a counter metric."""
        with self._lock:
            key = self._make_key(name, tags)
            self._counters[key] = self._counters.get(key, 0) + value
            self._points.append(MetricPoint(name, self._counters[key], tags=tags or {}))
            self._maybe_flush()

    def gauge(self, name: str, value: float, tags: Optional[Dict[str, str]] = None):
        """Set a gauge metric."""
        with self._lock:
            key = self._make_key(name, tags)
            self._gauges[key] = value
            self._points.append(MetricPoint(name, value, tags=tags or {}))
            self._maybe_flush()

    def histogram(self, name: str, value: float, tags: Optional[Dict[str, str]] = None):
        """Record a histogram value."""
        with self._lock:
            key = self._make_key(name, tags)
            if key not in self._histograms:
                self._histograms[key] = []
            self._histograms[key].append(value)
            self._points.append(MetricPoint(name, value, tags=tags or {}))
            self._maybe_flush()

    def timing(self, name: str, duration_ms: float, tags: Optional[Dict[str, str]] = None):
        """Record a timing metric."""
        with self._lock:
            key = self._make_key(name, tags)
            if key not in self._timings:
                self._timings[key] = []
            self._timings[key].append(duration_ms)
            self._points.append(MetricPoint(name, duration_ms, tags=tags or {}))
            self._maybe_flush()

    def time_function(self, name: str, tags: Optional[Dict[str, str]] = None):
        """Decorator to time function execution."""
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                start = time.time()
                try:
                    return func(*args, **kwargs)
                finally:
                    duration_ms = (time.time() - start) * 1000
                    self.timing(name, duration_ms, tags)
            return wrapper
        return decorator

    def _make_key(self, name: str, tags: Optional[Dict[str, str]] = None) -> str:
        """Create a unique key from name and tags."""
        if not tags:
            return name
        tag_str = ",".join(f"{k}={v}" for k, v in sorted(tags.items()))
        return f"{name}[{tag_str}]"

    def _maybe_flush(self):
        """Flush if enough time has passed."""
        if time.time() - self._last_flush >= self.flush_interval:
            self.flush()

    def flush(self):
        """Flush metrics to storage."""
        with self._lock:
            if not self._points or not self.storage_path:
                return

            try:
                self.storage_path.parent.mkdir(parents=True, exist_ok=True)

                # Append to metrics file
                with open(self.storage_path, 'a', encoding='utf-8') as f:
                    for point in self._points:
                        f.write(json.dumps({
                            "name": point.name,
                            "value": point.value,
                            "timestamp": point.timestamp.isoformat(),
                            "tags": point.tags
                        }) + "\n")

                self._points.clear()
                self._last_flush = time.time()

            except Exception as e:
                logger.warning(f"Failed to flush metrics: {e}")

    def get_stats(self) -> Dict[str, Any]:
        """Get current metric statistics."""
        with self._lock:
            stats = {
                "counters": dict(self._counters),
                "gauges": dict(self._gauges),
                "histograms": {},
                "timings": {}
            }

            for key, values in self._histograms.items():
                if values:
                    stats["histograms"][key] = {
                        "count": len(values),
                        "min": min(values),
                        "max": max(values),
                        "avg": sum(values) / len(values)
                    }

            for key, values in self._timings.items():
                if values:
                    sorted_values = sorted(values)
                    stats["timings"][key] = {
                        "count": len(values),
                        "min_ms": min(values),
                        "max_ms": max(values),
                        "avg_ms": sum(values) / len(values),
                        "p95_ms": sorted_values[int(len(sorted_values) * 0.95)] if len(sorted_values) > 1 else values[0]
                    }

            return stats


# Global metrics instance
_global_metrics: Optional[MetricsCollector] = None
_metrics_lock = threading.Lock()


def get_metrics(storage_path: Optional[str] = None) -> MetricsCollector:
    """Get global metrics collector."""
    global _global_metrics
    with _metrics_lock:
        if _global_metrics is None:
            _global_metrics = MetricsCollector(storage_path=storage_path)
        return _global_metrics


# ============================================================================
# CACHING
# ============================================================================

class LRUCache:
    """Least Recently Used cache with TTL support.

    Features:
    - Size-limited cache with LRU eviction
    - Time-to-live for entries
    - Thread-safe operations
    - Persistence support
    """

    def __init__(
        self,
        max_size: int = 100,
        ttl_seconds: float = 3600,
        persistence_path: Optional[str] = None
    ):
        """Initialize cache.

        Args:
            max_size: Maximum number of entries
            ttl_seconds: Time-to-live for entries
            persistence_path: Optional path to persist cache
        """
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self._cache: OrderedDict = OrderedDict()
        self._timestamps: Dict[str, float] = {}
        self._lock = threading.Lock()

        self.persistence_path = Path(persistence_path) if persistence_path else None
        if self.persistence_path and self.persistence_path.exists():
            self._load_from_disk()

    def get(self, key: str) -> Optional[Any]:
        """Get value from cache.

        Args:
            key: Cache key

        Returns:
            Cached value or None if not found/expired
        """
        with self._lock:
            if key not in self._cache:
                return None

            # Check TTL
            if time.time() - self._timestamps[key] > self.ttl_seconds:
                del self._cache[key]
                del self._timestamps[key]
                return None

            # Move to end (most recently used)
            self._cache.move_to_end(key)
            return self._cache[key]

    def set(self, key: str, value: Any):
        """Set value in cache.

        Args:
            key: Cache key
            value: Value to cache
        """
        with self._lock:
            # Remove if exists (to update position)
            if key in self._cache:
                del self._cache[key]

            # Evict if at capacity
            while len(self._cache) >= self.max_size:
                oldest_key = next(iter(self._cache))
                del self._cache[oldest_key]
                del self._timestamps[oldest_key]

            self._cache[key] = value
            self._timestamps[key] = time.time()

    def delete(self, key: str):
        """Delete entry from cache."""
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                del self._timestamps[key]

    def clear(self):
        """Clear all entries."""
        with self._lock:
            self._cache.clear()
            self._timestamps.clear()

    def __contains__(self, key: str) -> bool:
        return self.get(key) is not None

    def __len__(self) -> int:
        with self._lock:
            return len(self._cache)

    def save_to_disk(self):
        """Save cache to disk."""
        if not self.persistence_path:
            return

        with self._lock:
            try:
                self.persistence_path.parent.mkdir(parents=True, exist_ok=True)
                data = {
                    "entries": list(self._cache.items()),
                    "timestamps": self._timestamps,
                    "saved_at": datetime.now().isoformat()
                }
                with open(self.persistence_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, default=str)
            except Exception as e:
                logger.warning(f"Failed to save cache: {e}")

    def _load_from_disk(self):
        """Load cache from disk."""
        try:
            with open(self.persistence_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            for key, value in data.get("entries", []):
                self._cache[key] = value

            self._timestamps = data.get("timestamps", {})

            # Convert string timestamps back to float
            self._timestamps = {k: float(v) if isinstance(v, str) else v
                               for k, v in self._timestamps.items()}

        except Exception as e:
            logger.warning(f"Failed to load cache: {e}")


class PromptCache(LRUCache):
    """Specialized cache for LLM prompts and responses."""

    def __init__(
        self,
        max_size: int = 50,
        ttl_seconds: float = 3600,
        persistence_path: Optional[str] = None
    ):
        super().__init__(max_size, ttl_seconds, persistence_path)
        self._hits = 0
        self._misses = 0

    def _make_key(self, prompt: str, params: Optional[Dict] = None) -> str:
        """Create cache key from prompt and parameters."""
        key_data = prompt
        if params:
            key_data += json.dumps(params, sort_keys=True)
        return hashlib.sha256(key_data.encode()).hexdigest()[:16]

    def get_response(self, prompt: str, params: Optional[Dict] = None) -> Optional[str]:
        """Get cached response for prompt."""
        key = self._make_key(prompt, params)
        result = self.get(key)
        if result is not None:
            self._hits += 1
        else:
            self._misses += 1
        return result

    def cache_response(self, prompt: str, response: str, params: Optional[Dict] = None):
        """Cache response for prompt."""
        key = self._make_key(prompt, params)
        self.set(key, response)

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total = self._hits + self._misses
        return {
            "size": len(self),
            "max_size": self.max_size,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": self._hits / total if total > 0 else 0.0
        }


# Global prompt cache
_global_prompt_cache: Optional[PromptCache] = None
_cache_lock = threading.Lock()


def get_prompt_cache(persistence_path: Optional[str] = None) -> PromptCache:
    """Get global prompt cache."""
    global _global_prompt_cache
    with _cache_lock:
        if _global_prompt_cache is None:
            _global_prompt_cache = PromptCache(persistence_path=persistence_path)
        return _global_prompt_cache
