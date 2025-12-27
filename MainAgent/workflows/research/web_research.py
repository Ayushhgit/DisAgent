"""
Web Research Workflow - Conduct web research on topics.

This workflow integrates browser tools to:
- Search the web for information
- Fetch and parse web pages
- Extract relevant content
- Compile research findings
"""

from __future__ import annotations

import re
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


@dataclass
class ResearchSource:
    """A source of information."""
    url: str
    title: str
    content: str
    snippet: str = ""
    relevance_score: float = 0.0
    fetched_at: Optional[datetime] = None


@dataclass
class ResearchResult:
    """Result of a research operation."""
    query: str
    summary: str
    sources: List[ResearchSource] = field(default_factory=list)
    key_findings: List[str] = field(default_factory=list)
    related_topics: List[str] = field(default_factory=list)
    error: Optional[str] = None

    @property
    def success(self) -> bool:
        """Check if research was successful."""
        return self.error is None and len(self.sources) > 0


def web_research(query: str) -> Dict[str, Any]:
    """Conduct web research (legacy interface).

    Args:
        query: Research query

    Returns:
        Research result as dictionary
    """
    result = research_topic(query)
    return {
        "summary": result.summary,
        "sources": [
            {
                "url": s.url,
                "title": s.title,
                "snippet": s.snippet,
            }
            for s in result.sources
        ],
        "key_findings": result.key_findings,
        "related_topics": result.related_topics,
    }


def research_topic(
    query: str,
    max_sources: int = 5,
    include_content: bool = True,
) -> ResearchResult:
    """Research a topic using web search and content extraction.

    Args:
        query: The research query
        max_sources: Maximum number of sources to include
        include_content: Whether to fetch full page content

    Returns:
        ResearchResult with findings
    """
    if not query or not query.strip():
        return ResearchResult(
            query=query,
            summary="",
            error="Empty query provided",
        )

    sources = []
    key_findings = []
    related_topics = []

    try:
        # Import browser tools
        from ..tools.browser.search_tool import search_duckduckgo
        from ..tools.browser.fetch_tool import fetch
        from ..tools.browser.parse_tool import parse_html, extract_text

        # Search for the topic
        search_results = search_duckduckgo(query, num_results=max_sources * 2)

        if not search_results.results:
            return ResearchResult(
                query=query,
                summary=f"No search results found for: {query}",
                error="No search results",
            )

        # Process search results
        for result in search_results.results[:max_sources]:
            source = ResearchSource(
                url=result.url,
                title=result.title,
                snippet=result.snippet,
                content="",
            )

            # Fetch and parse full content if requested
            if include_content:
                try:
                    fetch_result = fetch(result.url, timeout=10)
                    if fetch_result.success:
                        parse_result = parse_html(
                            fetch_result.content,
                            base_url=result.url,
                            extract_text=True,
                        )
                        source.content = parse_result.text[:5000]  # Limit content
                        source.fetched_at = datetime.now()

                        # Extract key sentences
                        findings = _extract_key_sentences(
                            parse_result.text,
                            query,
                            max_sentences=3,
                        )
                        key_findings.extend(findings)

                except Exception as exc:
                    logger.warning(f"Failed to fetch {result.url}: {exc}")

            sources.append(source)

        # Calculate relevance scores
        for source in sources:
            source.relevance_score = _calculate_relevance(source, query)

        # Sort by relevance
        sources.sort(key=lambda s: s.relevance_score, reverse=True)

        # Extract related topics
        related_topics = _extract_related_topics(sources, query)

        # Generate summary
        summary = _generate_summary(query, sources, key_findings)

        return ResearchResult(
            query=query,
            summary=summary,
            sources=sources[:max_sources],
            key_findings=list(set(key_findings))[:10],
            related_topics=related_topics[:5],
        )

    except ImportError as exc:
        # Browser tools not available, use fallback
        logger.warning(f"Browser tools not available: {exc}")
        return _fallback_research(query)

    except Exception as exc:
        logger.exception(f"Research failed: {exc}")
        return ResearchResult(
            query=query,
            summary="",
            error=str(exc),
        )


def _fallback_research(query: str) -> ResearchResult:
    """Fallback when browser tools are not available."""
    return ResearchResult(
        query=query,
        summary=f"Research topic: {query}\n\nNote: Browser tools are not available. "
                "Install requests and beautifulsoup4 for web research capabilities.",
        key_findings=[
            "Browser tools required for web research",
            f"Query: {query}",
        ],
    )


def _extract_key_sentences(
    text: str,
    query: str,
    max_sentences: int = 5,
) -> List[str]:
    """Extract key sentences relevant to the query."""
    if not text:
        return []

    # Split into sentences
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]

    # Score sentences by relevance to query
    query_words = set(query.lower().split())
    scored = []

    for sentence in sentences:
        if len(sentence) < 20 or len(sentence) > 500:
            continue

        # Calculate word overlap
        sentence_words = set(sentence.lower().split())
        overlap = len(query_words & sentence_words)

        # Boost for exact phrase match
        if query.lower() in sentence.lower():
            overlap += 5

        if overlap > 0:
            scored.append((sentence, overlap))

    # Sort by score and return top sentences
    scored.sort(key=lambda x: x[1], reverse=True)
    return [s[0] for s in scored[:max_sentences]]


def _calculate_relevance(source: ResearchSource, query: str) -> float:
    """Calculate relevance score for a source."""
    score = 0.0
    query_lower = query.lower()
    query_words = set(query_lower.split())

    # Check title relevance
    title_lower = source.title.lower()
    for word in query_words:
        if word in title_lower:
            score += 2.0
    if query_lower in title_lower:
        score += 5.0

    # Check snippet relevance
    snippet_lower = source.snippet.lower()
    for word in query_words:
        if word in snippet_lower:
            score += 1.0
    if query_lower in snippet_lower:
        score += 3.0

    # Check content relevance
    if source.content:
        content_lower = source.content.lower()
        word_count = sum(1 for word in query_words if word in content_lower)
        score += word_count * 0.5

        # Check for exact phrase
        if query_lower in content_lower:
            score += 2.0

    # Boost for authoritative domains
    domain = urlparse(source.url).netloc.lower()
    authoritative = ['wikipedia.org', 'github.com', 'docs.', 'developer.',
                     'stackoverflow.com', '.gov', '.edu', 'medium.com']
    for auth in authoritative:
        if auth in domain:
            score += 2.0
            break

    return score


def _extract_related_topics(sources: List[ResearchSource], query: str) -> List[str]:
    """Extract related topics from sources."""
    related = set()
    query_words = set(query.lower().split())

    for source in sources:
        # Extract potential topics from titles
        title_words = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', source.title)
        for phrase in title_words:
            if phrase.lower() not in query_words and len(phrase) > 3:
                related.add(phrase)

        # Extract from content
        if source.content:
            # Look for capitalized phrases
            phrases = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b', source.content[:2000])
            for phrase in phrases:
                if phrase.lower() not in query_words and len(phrase) > 5:
                    related.add(phrase)

    # Filter and sort
    related = [t for t in related if len(t) < 50]
    return list(related)[:10]


def _generate_summary(
    query: str,
    sources: List[ResearchSource],
    key_findings: List[str],
) -> str:
    """Generate a research summary."""
    lines = [f"Research Summary: {query}", ""]

    if sources:
        lines.append(f"Found {len(sources)} relevant sources.")
        lines.append("")

        # Add key findings
        if key_findings:
            lines.append("Key Findings:")
            for i, finding in enumerate(key_findings[:5], 1):
                lines.append(f"  {i}. {finding[:200]}...")
            lines.append("")

        # Add source overview
        lines.append("Top Sources:")
        for source in sources[:3]:
            lines.append(f"  - {source.title}")
            lines.append(f"    {source.url}")
            if source.snippet:
                lines.append(f"    {source.snippet[:150]}...")
            lines.append("")

    return '\n'.join(lines)


def search_and_summarize(
    query: str,
    focus_areas: Optional[List[str]] = None,
) -> ResearchResult:
    """Search and summarize with focus on specific areas.

    Args:
        query: Main search query
        focus_areas: Specific aspects to focus on

    Returns:
        ResearchResult
    """
    # Conduct main research
    result = research_topic(query, max_sources=5, include_content=True)

    if not result.success:
        return result

    # Research focus areas if specified
    if focus_areas:
        for area in focus_areas[:3]:  # Limit to 3 focus areas
            focused_query = f"{query} {area}"
            focused_result = research_topic(focused_query, max_sources=2)

            if focused_result.success:
                # Merge results
                result.sources.extend(focused_result.sources)
                result.key_findings.extend(focused_result.key_findings)

        # Deduplicate sources
        seen_urls = set()
        unique_sources = []
        for source in result.sources:
            if source.url not in seen_urls:
                seen_urls.add(source.url)
                unique_sources.append(source)
        result.sources = unique_sources

        # Deduplicate findings
        result.key_findings = list(set(result.key_findings))

    return result


def compare_sources(
    query: str,
    perspectives: List[str],
) -> Dict[str, ResearchResult]:
    """Compare different perspectives on a topic.

    Args:
        query: Main topic
        perspectives: Different perspectives to research

    Returns:
        Dictionary mapping perspective to research result
    """
    results = {}

    for perspective in perspectives:
        focused_query = f"{query} {perspective}"
        results[perspective] = research_topic(focused_query, max_sources=3)

    return results


def research_code_topic(
    technology: str,
    topic: str,
    include_examples: bool = True,
) -> ResearchResult:
    """Research a coding/technology topic.

    Args:
        technology: Technology (e.g., "Python", "React")
        topic: Specific topic (e.g., "async await", "hooks")
        include_examples: Include code examples

    Returns:
        ResearchResult
    """
    query = f"{technology} {topic}"
    if include_examples:
        query += " example code"

    result = research_topic(query, max_sources=5, include_content=True)

    if result.success:
        # Try to extract code examples from content
        code_examples = []
        for source in result.sources:
            if source.content:
                # Look for code blocks
                code_blocks = re.findall(
                    r'```[\w]*\n(.*?)```',
                    source.content,
                    re.DOTALL,
                )
                code_examples.extend(code_blocks[:2])

        if code_examples:
            result.key_findings.append(
                f"Found {len(code_examples)} code examples"
            )

    return result
