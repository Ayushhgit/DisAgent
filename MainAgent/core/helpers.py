# ============================================================================
# FILE: helpers.py
# Helper functions for edge case handling
# ============================================================================

from __future__ import annotations

import os
import re
import time
import unicodedata
import logging
from pathlib import Path
from typing import List, Optional, Tuple, Iterator, Any, Callable
from functools import wraps

logger = logging.getLogger(__name__)


# ============================================================================
# EMPTY PROJECT HANDLING
# ============================================================================

def is_project_empty(project_path: str) -> bool:
    """Check if a project directory is empty or contains no meaningful files.

    Args:
        project_path: Path to project directory

    Returns:
        True if project is empty or has no source files
    """
    path = Path(project_path)

    if not path.exists():
        return True

    # Directories to ignore
    ignore_dirs = {'.git', '__pycache__', 'node_modules', '.venv', 'venv', '.idea', '.vscode'}

    # Files to ignore
    ignore_files = {'.gitignore', '.gitattributes', '.DS_Store', 'Thumbs.db'}

    # Source file extensions
    source_extensions = {
        '.py', '.js', '.ts', '.jsx', '.tsx', '.java', '.go', '.rs',
        '.c', '.cpp', '.h', '.hpp', '.cs', '.rb', '.php', '.swift',
        '.html', '.css', '.scss', '.sass', '.vue', '.svelte'
    }

    for item in path.rglob('*'):
        if item.is_file():
            # Skip ignored directories
            if any(ignored in item.parts for ignored in ignore_dirs):
                continue

            # Skip ignored files
            if item.name in ignore_files:
                continue

            # Check for source files
            if item.suffix.lower() in source_extensions:
                return False

            # Check for common project files
            if item.name.lower() in {'readme.md', 'package.json', 'requirements.txt',
                                      'cargo.toml', 'go.mod', 'setup.py', 'pyproject.toml'}:
                return False

    return True


def get_project_stats(project_path: str) -> dict:
    """Get statistics about a project.

    Args:
        project_path: Path to project directory

    Returns:
        Dict with project statistics
    """
    path = Path(project_path)
    stats = {
        "total_files": 0,
        "total_dirs": 0,
        "source_files": 0,
        "total_size_bytes": 0,
        "file_types": {},
        "is_empty": True
    }

    if not path.exists():
        return stats

    ignore_dirs = {'.git', '__pycache__', 'node_modules', '.venv', 'venv'}

    for item in path.rglob('*'):
        if any(ignored in item.parts for ignored in ignore_dirs):
            continue

        if item.is_dir():
            stats["total_dirs"] += 1
        elif item.is_file():
            stats["total_files"] += 1
            stats["total_size_bytes"] += item.stat().st_size

            ext = item.suffix.lower()
            stats["file_types"][ext] = stats["file_types"].get(ext, 0) + 1

            if ext in {'.py', '.js', '.ts', '.jsx', '.tsx', '.java', '.go', '.rs'}:
                stats["source_files"] += 1

    stats["is_empty"] = stats["source_files"] == 0
    return stats


# ============================================================================
# UNICODE FILENAME HANDLING
# ============================================================================

def normalize_filename(filename: str) -> str:
    """Normalize a filename for safe filesystem operations.

    Args:
        filename: Original filename

    Returns:
        Normalized filename safe for all filesystems
    """
    # Normalize unicode characters
    filename = unicodedata.normalize('NFC', filename)

    # Replace problematic characters
    # Windows forbidden: < > : " / \ | ? *
    # Additional: control characters, leading/trailing dots and spaces
    replacements = {
        '<': '_lt_',
        '>': '_gt_',
        ':': '_',
        '"': '_',
        '/': '_',
        '\\': '_',
        '|': '_',
        '?': '_',
        '*': '_',
    }

    for char, replacement in replacements.items():
        filename = filename.replace(char, replacement)

    # Remove control characters
    filename = ''.join(c for c in filename if unicodedata.category(c) != 'Cc')

    # Remove leading/trailing dots and spaces
    filename = filename.strip('. ')

    # Ensure not empty
    if not filename:
        filename = 'unnamed'

    # Limit length (accounting for extension)
    max_length = 200
    if len(filename) > max_length:
        base, ext = os.path.splitext(filename)
        filename = base[:max_length - len(ext)] + ext

    return filename


def is_safe_filename(filename: str) -> Tuple[bool, str]:
    """Check if a filename is safe for filesystem operations.

    Args:
        filename: Filename to check

    Returns:
        Tuple of (is_safe, reason if not safe)
    """
    if not filename:
        return False, "Empty filename"

    # Check for null bytes
    if '\x00' in filename:
        return False, "Contains null byte"

    # Check for path traversal
    if '..' in filename:
        return False, "Contains path traversal"

    # Check for absolute path markers
    if filename.startswith('/') or (len(filename) > 1 and filename[1] == ':'):
        return False, "Contains absolute path"

    # Check for reserved Windows names
    reserved = {'CON', 'PRN', 'AUX', 'NUL', 'COM1', 'COM2', 'COM3', 'COM4',
                'COM5', 'COM6', 'COM7', 'COM8', 'COM9', 'LPT1', 'LPT2',
                'LPT3', 'LPT4', 'LPT5', 'LPT6', 'LPT7', 'LPT8', 'LPT9'}
    base_name = os.path.splitext(filename)[0].upper()
    if base_name in reserved:
        return False, f"Reserved filename: {base_name}"

    # Check for problematic characters
    problematic = set('<>:"|?*')
    found = [c for c in filename if c in problematic]
    if found:
        return False, f"Contains invalid characters: {found}"

    return True, ""


def safe_path_join(base: str, *parts: str) -> str:
    """Safely join path components, preventing path traversal.

    Args:
        base: Base directory path
        *parts: Path components to join

    Returns:
        Safe joined path

    Raises:
        ValueError: If path traversal is detected
    """
    base_path = Path(base).resolve()

    # Normalize and join parts
    normalized_parts = []
    for part in parts:
        # Remove leading slashes and normalize
        part = part.lstrip('/\\')
        part = normalize_filename(part)
        normalized_parts.append(part)

    result = base_path.joinpath(*normalized_parts).resolve()

    # Ensure result is under base
    try:
        result.relative_to(base_path)
    except ValueError:
        raise ValueError(f"Path traversal detected: result {result} is outside base {base_path}")

    return str(result)


# ============================================================================
# LARGE FILE HANDLING
# ============================================================================

def chunk_text(text: str, chunk_size: int = 8000, overlap: int = 500) -> Iterator[Tuple[int, str]]:
    """Split text into overlapping chunks for processing.

    Args:
        text: Text to chunk
        chunk_size: Maximum size of each chunk
        overlap: Overlap between chunks for context

    Yields:
        Tuples of (chunk_index, chunk_text)
    """
    if len(text) <= chunk_size:
        yield (0, text)
        return

    start = 0
    chunk_idx = 0

    while start < len(text):
        end = start + chunk_size

        # Try to break at a natural boundary
        if end < len(text):
            # Look for newline near the end
            newline_pos = text.rfind('\n', start + chunk_size - 200, end)
            if newline_pos > start:
                end = newline_pos + 1

        chunk = text[start:end]
        yield (chunk_idx, chunk)

        chunk_idx += 1
        start = end - overlap

        # Prevent infinite loop
        if start >= len(text):
            break


def read_file_chunked(
    file_path: str,
    chunk_size: int = 8000,
    max_chunks: int = 10
) -> Iterator[Tuple[int, str, bool]]:
    """Read a file in chunks.

    Args:
        file_path: Path to file
        chunk_size: Size of each chunk
        max_chunks: Maximum number of chunks to read

    Yields:
        Tuples of (chunk_index, chunk_content, is_last)
    """
    path = Path(file_path)

    if not path.exists():
        return

    file_size = path.stat().st_size
    total_chunks = min(max_chunks, (file_size // chunk_size) + 1)

    try:
        with open(path, 'r', encoding='utf-8', errors='replace') as f:
            for chunk_idx in range(max_chunks):
                content = f.read(chunk_size)
                if not content:
                    break

                is_last = chunk_idx == total_chunks - 1 or len(content) < chunk_size
                yield (chunk_idx, content, is_last)

                if is_last:
                    break

    except Exception as e:
        logger.warning(f"Error reading file {file_path}: {e}")


def summarize_large_file(file_path: str, max_lines: int = 100) -> str:
    """Create a summary of a large file.

    Args:
        file_path: Path to file
        max_lines: Maximum lines to include in summary

    Returns:
        File summary string
    """
    path = Path(file_path)

    if not path.exists():
        return f"File not found: {file_path}"

    try:
        file_size = path.stat().st_size

        with open(path, 'r', encoding='utf-8', errors='replace') as f:
            lines = []
            for i, line in enumerate(f):
                if i >= max_lines:
                    break
                lines.append(line)

        total_lines = sum(1 for _ in open(path, 'r', encoding='utf-8', errors='replace'))

        summary = f"=== File Summary: {path.name} ===\n"
        summary += f"Size: {file_size:,} bytes | Lines: {total_lines:,}\n"

        if total_lines > max_lines:
            summary += f"Showing first {max_lines} lines:\n\n"
        else:
            summary += "\n"

        summary += ''.join(lines)

        if total_lines > max_lines:
            summary += f"\n... ({total_lines - max_lines} more lines) ..."

        return summary

    except Exception as e:
        return f"Error reading file: {e}"


# ============================================================================
# NETWORK RESILIENCE
# ============================================================================

def retry_with_backoff(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    exceptions: Tuple = (Exception,),
    on_retry: Optional[Callable[[int, Exception], None]] = None
):
    """Decorator for retrying functions with exponential backoff.

    Args:
        max_retries: Maximum number of retry attempts
        base_delay: Initial delay between retries
        max_delay: Maximum delay between retries
        exponential_base: Base for exponential backoff
        exceptions: Tuple of exceptions to catch
        on_retry: Optional callback called on each retry
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None

            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e

                    if attempt < max_retries:
                        delay = min(base_delay * (exponential_base ** attempt), max_delay)

                        if on_retry:
                            on_retry(attempt + 1, e)

                        logger.warning(
                            f"Attempt {attempt + 1}/{max_retries + 1} failed: {e}. "
                            f"Retrying in {delay:.1f}s..."
                        )
                        time.sleep(delay)

            # All retries exhausted
            raise last_exception

        return wrapper
    return decorator


class CircuitBreaker:
    """Circuit breaker pattern for network operations.

    States:
    - CLOSED: Normal operation, requests go through
    - OPEN: Failing, requests are rejected immediately
    - HALF_OPEN: Testing, limited requests allowed
    """

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"

    def __init__(
        self,
        failure_threshold: int = 5,
        reset_timeout: float = 60.0,
        half_open_max_calls: int = 1
    ):
        """Initialize circuit breaker.

        Args:
            failure_threshold: Failures before opening circuit
            reset_timeout: Seconds before attempting reset
            half_open_max_calls: Max calls in half-open state
        """
        self.failure_threshold = failure_threshold
        self.reset_timeout = reset_timeout
        self.half_open_max_calls = half_open_max_calls

        self._state = self.CLOSED
        self._failure_count = 0
        self._last_failure_time: Optional[float] = None
        self._half_open_calls = 0

    @property
    def state(self) -> str:
        """Get current state, checking for timeout-based transitions."""
        if self._state == self.OPEN:
            if self._last_failure_time and \
               time.time() - self._last_failure_time >= self.reset_timeout:
                self._state = self.HALF_OPEN
                self._half_open_calls = 0

        return self._state

    def can_execute(self) -> bool:
        """Check if execution is allowed."""
        state = self.state

        if state == self.CLOSED:
            return True
        elif state == self.HALF_OPEN:
            return self._half_open_calls < self.half_open_max_calls
        else:  # OPEN
            return False

    def record_success(self):
        """Record successful execution."""
        if self._state == self.HALF_OPEN:
            self._state = self.CLOSED
            self._failure_count = 0
            logger.info("Circuit breaker: CLOSED (recovered)")
        elif self._state == self.CLOSED:
            self._failure_count = 0

    def record_failure(self):
        """Record failed execution."""
        self._failure_count += 1
        self._last_failure_time = time.time()

        if self._state == self.HALF_OPEN:
            self._state = self.OPEN
            logger.warning("Circuit breaker: OPEN (half-open test failed)")
        elif self._failure_count >= self.failure_threshold:
            self._state = self.OPEN
            logger.warning(f"Circuit breaker: OPEN (threshold {self.failure_threshold} reached)")

    def __call__(self, func):
        """Use as decorator."""
        @wraps(func)
        def wrapper(*args, **kwargs):
            if not self.can_execute():
                raise RuntimeError(f"Circuit breaker is {self.state}")

            if self.state == self.HALF_OPEN:
                self._half_open_calls += 1

            try:
                result = func(*args, **kwargs)
                self.record_success()
                return result
            except Exception as e:
                self.record_failure()
                raise

        return wrapper


def with_timeout(timeout_seconds: float, default: Any = None):
    """Decorator to add timeout to a function.

    Note: This uses threading, so the function will continue running
    in the background even after timeout. Use with caution.

    Args:
        timeout_seconds: Maximum execution time
        default: Default value to return on timeout
    """
    import threading
    from concurrent.futures import ThreadPoolExecutor, TimeoutError

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(func, *args, **kwargs)
                try:
                    return future.result(timeout=timeout_seconds)
                except TimeoutError:
                    logger.warning(f"Function {func.__name__} timed out after {timeout_seconds}s")
                    return default

        return wrapper
    return decorator
