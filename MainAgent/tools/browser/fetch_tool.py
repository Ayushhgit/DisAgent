"""
Web Fetch Tool - Fetches content from URLs with proper error handling.

Features:
- HTTP/HTTPS support
- Timeout handling
- User-agent spoofing
- Response caching
- Rate limiting
"""

from __future__ import annotations

import time
import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass
from urllib.parse import urlparse
import hashlib

logger = logging.getLogger(__name__)

# Try to import requests
try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    logger.warning("requests library not available - web fetch disabled")


@dataclass
class FetchResult:
    """Result from a web fetch operation."""
    success: bool
    url: str
    content: str
    status_code: int
    content_type: str
    headers: Dict[str, str]
    error: Optional[str] = None
    cached: bool = False
    fetch_time: float = 0.0


# Simple in-memory cache
_cache: Dict[str, tuple[FetchResult, float]] = {}
CACHE_TTL = 300  # 5 minutes


# Rate limiting
_last_fetch_time: Dict[str, float] = {}
MIN_FETCH_INTERVAL = 1.0  # Minimum seconds between fetches to same domain


def _get_domain(url: str) -> str:
    """Extract domain from URL."""
    try:
        parsed = urlparse(url)
        return parsed.netloc
    except Exception:
        return url


def _check_rate_limit(domain: str) -> bool:
    """Check if we can fetch from this domain (rate limiting)."""
    now = time.time()
    last_time = _last_fetch_time.get(domain, 0)

    if now - last_time < MIN_FETCH_INTERVAL:
        return False

    _last_fetch_time[domain] = now
    return True


def _get_cache_key(url: str) -> str:
    """Generate cache key for URL."""
    return hashlib.md5(url.encode()).hexdigest()


def _get_cached(url: str) -> Optional[FetchResult]:
    """Get cached result if available and not expired."""
    key = _get_cache_key(url)
    if key in _cache:
        result, timestamp = _cache[key]
        if time.time() - timestamp < CACHE_TTL:
            result.cached = True
            return result
        else:
            del _cache[key]
    return None


def _set_cache(url: str, result: FetchResult) -> None:
    """Cache a fetch result."""
    key = _get_cache_key(url)
    _cache[key] = (result, time.time())


def fetch(
    url: str,
    timeout: int = 30,
    headers: Optional[Dict[str, str]] = None,
    use_cache: bool = True,
    follow_redirects: bool = True,
) -> FetchResult:
    """Fetch content from a URL.

    Args:
        url: URL to fetch
        timeout: Request timeout in seconds
        headers: Optional custom headers
        use_cache: Whether to use caching
        follow_redirects: Whether to follow HTTP redirects

    Returns:
        FetchResult with content and metadata
    """
    if not REQUESTS_AVAILABLE:
        return FetchResult(
            success=False,
            url=url,
            content="",
            status_code=0,
            content_type="",
            headers={},
            error="requests library not available. Install with: pip install requests",
        )

    # Validate URL
    if not url or not url.startswith(('http://', 'https://')):
        return FetchResult(
            success=False,
            url=url,
            content="",
            status_code=0,
            content_type="",
            headers={},
            error=f"Invalid URL: {url}. Must start with http:// or https://",
        )

    # Check cache first
    if use_cache:
        cached = _get_cached(url)
        if cached:
            logger.debug(f"Cache hit for {url}")
            return cached

    # Rate limiting
    domain = _get_domain(url)
    if not _check_rate_limit(domain):
        return FetchResult(
            success=False,
            url=url,
            content="",
            status_code=429,
            content_type="",
            headers={},
            error=f"Rate limited: Too many requests to {domain}",
        )

    # Default headers
    default_headers = {
        "User-Agent": "Mozilla/5.0 (compatible; DisAgent/1.0; +https://github.com/DisAgent)",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
    }

    if headers:
        default_headers.update(headers)

    start_time = time.time()

    try:
        response = requests.get(
            url,
            headers=default_headers,
            timeout=timeout,
            allow_redirects=follow_redirects,
        )

        fetch_time = time.time() - start_time

        # Try to decode content
        try:
            content = response.text
        except Exception:
            content = response.content.decode('utf-8', errors='replace')

        result = FetchResult(
            success=response.ok,
            url=url,
            content=content,
            status_code=response.status_code,
            content_type=response.headers.get('content-type', ''),
            headers=dict(response.headers),
            fetch_time=fetch_time,
            error=None if response.ok else f"HTTP {response.status_code}",
        )

        # Cache successful results
        if use_cache and result.success:
            _set_cache(url, result)

        return result

    except requests.exceptions.Timeout:
        return FetchResult(
            success=False,
            url=url,
            content="",
            status_code=0,
            content_type="",
            headers={},
            error=f"Request timed out after {timeout} seconds",
            fetch_time=time.time() - start_time,
        )

    except requests.exceptions.ConnectionError as exc:
        return FetchResult(
            success=False,
            url=url,
            content="",
            status_code=0,
            content_type="",
            headers={},
            error=f"Connection error: {exc}",
            fetch_time=time.time() - start_time,
        )

    except requests.exceptions.RequestException as exc:
        return FetchResult(
            success=False,
            url=url,
            content="",
            status_code=0,
            content_type="",
            headers={},
            error=f"Request failed: {exc}",
            fetch_time=time.time() - start_time,
        )

    except Exception as exc:
        logger.exception(f"Unexpected error fetching {url}: {exc}")
        return FetchResult(
            success=False,
            url=url,
            content="",
            status_code=0,
            content_type="",
            headers={},
            error=f"Unexpected error: {exc}",
            fetch_time=time.time() - start_time,
        )


def fetch_json(url: str, timeout: int = 30, headers: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    """Fetch JSON from a URL.

    Args:
        url: URL to fetch
        timeout: Request timeout
        headers: Optional custom headers

    Returns:
        Parsed JSON as dictionary, or error dict
    """
    import json

    result = fetch(url, timeout=timeout, headers=headers)

    if not result.success:
        return {"error": result.error, "success": False}

    try:
        data = json.loads(result.content)
        return {"data": data, "success": True}
    except json.JSONDecodeError as exc:
        return {"error": f"JSON parse error: {exc}", "success": False}


def clear_cache() -> int:
    """Clear the fetch cache.

    Returns:
        Number of cache entries cleared
    """
    global _cache
    count = len(_cache)
    _cache.clear()
    return count


def get_cache_stats() -> Dict[str, Any]:
    """Get cache statistics."""
    return {
        "entries": len(_cache),
        "ttl_seconds": CACHE_TTL,
    }
