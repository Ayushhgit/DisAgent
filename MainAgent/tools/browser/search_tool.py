"""
Web Search Tool - Search the web for information.

Features:
- DuckDuckGo search (no API key required)
- Google Custom Search (requires API key)
- Result caching
- Safe search filtering
"""

from __future__ import annotations

import re
import time
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from urllib.parse import quote_plus, urljoin

logger = logging.getLogger(__name__)

# Import fetch tool
try:
    from .fetch_tool import fetch, FetchResult
except ImportError:
    try:
        from tools.browser.fetch_tool import fetch, FetchResult
    except ImportError:
        # Fallback - define minimal fetch
        def fetch(url, **kwargs):
            return type('FetchResult', (), {'success': False, 'content': '', 'error': 'fetch not available'})()


@dataclass
class SearchResult:
    """A single search result."""
    title: str
    url: str
    snippet: str
    source: str = ""


@dataclass
class SearchResults:
    """Results from a web search."""
    success: bool
    query: str
    results: List[SearchResult] = field(default_factory=list)
    total_results: int = 0
    search_time: float = 0.0
    error: Optional[str] = None


# Search cache
_search_cache: Dict[str, tuple[SearchResults, float]] = {}
SEARCH_CACHE_TTL = 3600  # 1 hour


def _get_cached_search(query: str) -> Optional[SearchResults]:
    """Get cached search results."""
    import hashlib
    key = hashlib.md5(query.encode()).hexdigest()
    if key in _search_cache:
        results, timestamp = _search_cache[key]
        if time.time() - timestamp < SEARCH_CACHE_TTL:
            return results
        else:
            del _search_cache[key]
    return None


def _cache_search(query: str, results: SearchResults) -> None:
    """Cache search results."""
    import hashlib
    key = hashlib.md5(query.encode()).hexdigest()
    _search_cache[key] = (results, time.time())


def search_duckduckgo(
    query: str,
    num_results: int = 10,
    safe_search: bool = True,
) -> SearchResults:
    """Search using DuckDuckGo HTML interface.

    Args:
        query: Search query
        num_results: Maximum number of results
        safe_search: Enable safe search

    Returns:
        SearchResults with findings
    """
    # Check cache
    cached = _get_cached_search(f"ddg:{query}")
    if cached:
        return cached

    start_time = time.time()

    # DuckDuckGo HTML search URL
    safe_param = "1" if safe_search else "-1"
    search_url = f"https://html.duckduckgo.com/html/?q={quote_plus(query)}&kp={safe_param}"

    try:
        result = fetch(search_url, timeout=15, use_cache=False)

        if not result.success:
            return SearchResults(
                success=False,
                query=query,
                error=result.error,
                search_time=time.time() - start_time,
            )

        # Parse results from HTML
        results = _parse_duckduckgo_results(result.content, num_results)

        search_results = SearchResults(
            success=True,
            query=query,
            results=results,
            total_results=len(results),
            search_time=time.time() - start_time,
        )

        # Cache results
        _cache_search(f"ddg:{query}", search_results)

        return search_results

    except Exception as exc:
        logger.exception(f"DuckDuckGo search failed: {exc}")
        return SearchResults(
            success=False,
            query=query,
            error=str(exc),
            search_time=time.time() - start_time,
        )


def _parse_duckduckgo_results(html: str, max_results: int) -> List[SearchResult]:
    """Parse DuckDuckGo HTML results page."""
    results = []

    # Try with BeautifulSoup first
    try:
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html, 'html.parser')

        # Find result divs
        for result_div in soup.select('.result'):
            if len(results) >= max_results:
                break

            # Get title and URL
            title_link = result_div.select_one('.result__a')
            if not title_link:
                continue

            title = title_link.get_text().strip()
            href = title_link.get('href', '')

            # DuckDuckGo uses redirect URLs, extract actual URL
            if 'uddg=' in href:
                try:
                    from urllib.parse import unquote, parse_qs, urlparse
                    parsed = urlparse(href)
                    params = parse_qs(parsed.query)
                    if 'uddg' in params:
                        href = unquote(params['uddg'][0])
                except Exception:
                    pass

            # Get snippet
            snippet_div = result_div.select_one('.result__snippet')
            snippet = snippet_div.get_text().strip() if snippet_div else ""

            if title and href:
                results.append(SearchResult(
                    title=title,
                    url=href,
                    snippet=snippet,
                    source="duckduckgo",
                ))

        return results

    except ImportError:
        # Fallback to regex parsing
        return _parse_duckduckgo_regex(html, max_results)


def _parse_duckduckgo_regex(html: str, max_results: int) -> List[SearchResult]:
    """Parse DuckDuckGo results using regex (fallback)."""
    results = []

    # Pattern to match result blocks
    result_pattern = r'class="result[^"]*"[^>]*>.*?<a[^>]*class="result__a"[^>]*href="([^"]*)"[^>]*>([^<]*)</a>.*?class="result__snippet"[^>]*>([^<]*)<'

    for match in re.finditer(result_pattern, html, re.DOTALL | re.IGNORECASE):
        if len(results) >= max_results:
            break

        href, title, snippet = match.groups()
        title = title.strip()
        snippet = snippet.strip()

        # Clean up URL
        if 'uddg=' in href:
            try:
                from urllib.parse import unquote, parse_qs, urlparse
                parsed = urlparse(href)
                params = parse_qs(parsed.query)
                if 'uddg' in params:
                    href = unquote(params['uddg'][0])
            except Exception:
                pass

        if title and href:
            results.append(SearchResult(
                title=title,
                url=href,
                snippet=snippet,
                source="duckduckgo",
            ))

    return results


def search(
    query: str,
    num_results: int = 10,
    engine: str = "duckduckgo",
    safe_search: bool = True,
) -> SearchResults:
    """Search the web.

    Args:
        query: Search query
        num_results: Maximum number of results
        engine: Search engine to use ('duckduckgo')
        safe_search: Enable safe search

    Returns:
        SearchResults with findings
    """
    if not query or not query.strip():
        return SearchResults(
            success=False,
            query=query,
            error="Empty search query",
        )

    if engine == "duckduckgo":
        return search_duckduckgo(query, num_results, safe_search)
    else:
        return SearchResults(
            success=False,
            query=query,
            error=f"Unknown search engine: {engine}",
        )


def search_code(
    query: str,
    language: Optional[str] = None,
    num_results: int = 10,
) -> SearchResults:
    """Search for code examples.

    Args:
        query: Search query
        language: Programming language filter
        num_results: Maximum number of results

    Returns:
        SearchResults focused on code examples
    """
    # Add code-focused terms to query
    code_query = f"{query} code example"
    if language:
        code_query = f"{language} {code_query}"

    # Add site filters for code-related sites
    code_query += " site:github.com OR site:stackoverflow.com OR site:dev.to"

    return search_duckduckgo(code_query, num_results)


def search_documentation(
    query: str,
    technology: Optional[str] = None,
    num_results: int = 10,
) -> SearchResults:
    """Search for documentation.

    Args:
        query: Search query
        technology: Technology/framework to search for
        num_results: Maximum number of results

    Returns:
        SearchResults focused on documentation
    """
    doc_query = f"{query} documentation"
    if technology:
        doc_query = f"{technology} {doc_query}"

    # Focus on documentation sites
    doc_query += " site:docs.* OR site:developer.* OR site:readthedocs.io"

    return search_duckduckgo(doc_query, num_results)


def clear_search_cache() -> int:
    """Clear the search cache.

    Returns:
        Number of cache entries cleared
    """
    global _search_cache
    count = len(_search_cache)
    _search_cache.clear()
    return count
