"""
HTML Parse Tool - Parse and extract information from HTML content.

Features:
- Text extraction
- Link extraction
- Code block extraction
- Metadata extraction
- CSS selector support (when BeautifulSoup available)
"""

from __future__ import annotations

import re
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from html.parser import HTMLParser
from urllib.parse import urljoin

logger = logging.getLogger(__name__)

# Try to import BeautifulSoup
try:
    from bs4 import BeautifulSoup
    BS4_AVAILABLE = True
except ImportError:
    BS4_AVAILABLE = False
    logger.warning("BeautifulSoup not available - using basic HTML parsing")


@dataclass
class ParseResult:
    """Result from parsing HTML content."""
    success: bool
    title: str = ""
    text: str = ""
    links: List[Dict[str, str]] = field(default_factory=list)
    images: List[Dict[str, str]] = field(default_factory=list)
    code_blocks: List[str] = field(default_factory=list)
    headings: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, str] = field(default_factory=dict)
    error: Optional[str] = None


class BasicHTMLParser(HTMLParser):
    """Basic HTML parser for when BeautifulSoup is not available."""

    def __init__(self, base_url: str = ""):
        super().__init__()
        self.base_url = base_url
        self.title = ""
        self.text_parts = []
        self.links = []
        self.images = []
        self.code_blocks = []
        self.headings = []
        self.metadata = {}

        self._in_title = False
        self._in_code = False
        self._in_pre = False
        self._current_code = ""
        self._current_heading = None
        self._heading_level = 0
        self._skip_tags = {'script', 'style', 'noscript', 'svg'}
        self._skip_depth = 0

    def handle_starttag(self, tag, attrs):
        attrs_dict = dict(attrs)

        if tag in self._skip_tags:
            self._skip_depth += 1
            return

        if tag == 'title':
            self._in_title = True
        elif tag == 'a':
            href = attrs_dict.get('href', '')
            if href:
                if self.base_url:
                    href = urljoin(self.base_url, href)
                self.links.append({
                    'href': href,
                    'text': '',  # Will be filled in handle_data
                })
        elif tag == 'img':
            src = attrs_dict.get('src', '')
            if src:
                if self.base_url:
                    src = urljoin(self.base_url, src)
                self.images.append({
                    'src': src,
                    'alt': attrs_dict.get('alt', ''),
                })
        elif tag in ('code', 'pre'):
            if tag == 'code':
                self._in_code = True
            else:
                self._in_pre = True
            self._current_code = ""
        elif tag in ('h1', 'h2', 'h3', 'h4', 'h5', 'h6'):
            self._heading_level = int(tag[1])
            self._current_heading = ""
        elif tag == 'meta':
            name = attrs_dict.get('name', attrs_dict.get('property', ''))
            content = attrs_dict.get('content', '')
            if name and content:
                self.metadata[name] = content

    def handle_endtag(self, tag):
        if tag in self._skip_tags:
            self._skip_depth = max(0, self._skip_depth - 1)
            return

        if tag == 'title':
            self._in_title = False
        elif tag == 'code':
            self._in_code = False
            if self._current_code.strip():
                self.code_blocks.append(self._current_code.strip())
        elif tag == 'pre':
            self._in_pre = False
            if self._current_code.strip():
                self.code_blocks.append(self._current_code.strip())
        elif tag in ('h1', 'h2', 'h3', 'h4', 'h5', 'h6'):
            if self._current_heading:
                self.headings.append({
                    'level': self._heading_level,
                    'text': self._current_heading.strip(),
                })
            self._current_heading = None
            self._heading_level = 0

    def handle_data(self, data):
        if self._skip_depth > 0:
            return

        if self._in_title:
            self.title += data
        elif self._in_code or self._in_pre:
            self._current_code += data
        elif self._current_heading is not None:
            self._current_heading += data
        else:
            # Add to text content
            stripped = data.strip()
            if stripped:
                self.text_parts.append(stripped)
                # Update last link text if we just added a link
                if self.links and not self.links[-1]['text']:
                    self.links[-1]['text'] = stripped

    def get_text(self) -> str:
        """Get extracted text."""
        return ' '.join(self.text_parts)


def parse(html: str, base_url: str = "") -> Dict[str, Any]:
    """Parse HTML content and extract information (legacy interface).

    Args:
        html: HTML content to parse
        base_url: Base URL for resolving relative links

    Returns:
        Dictionary with extracted information
    """
    result = parse_html(html, base_url)
    return {
        "title": result.title,
        "text": result.text,
        "links": result.links,
        "images": result.images,
        "code_blocks": result.code_blocks,
        "headings": result.headings,
        "metadata": result.metadata,
        "success": result.success,
        "error": result.error,
    }


def parse_html(
    html: str,
    base_url: str = "",
    extract_text: bool = True,
    extract_links: bool = True,
    extract_images: bool = True,
    extract_code: bool = True,
) -> ParseResult:
    """Parse HTML content and extract information.

    Args:
        html: HTML content to parse
        base_url: Base URL for resolving relative links
        extract_text: Whether to extract text content
        extract_links: Whether to extract links
        extract_images: Whether to extract images
        extract_code: Whether to extract code blocks

    Returns:
        ParseResult with extracted information
    """
    if not html:
        return ParseResult(success=False, error="Empty HTML content")

    try:
        if BS4_AVAILABLE:
            return _parse_with_beautifulsoup(
                html, base_url, extract_text, extract_links, extract_images, extract_code
            )
        else:
            return _parse_with_basic_parser(
                html, base_url, extract_text, extract_links, extract_images, extract_code
            )
    except Exception as exc:
        logger.exception(f"Failed to parse HTML: {exc}")
        return ParseResult(success=False, error=f"Parse error: {exc}")


def _parse_with_beautifulsoup(
    html: str,
    base_url: str,
    extract_text: bool,
    extract_links: bool,
    extract_images: bool,
    extract_code: bool,
) -> ParseResult:
    """Parse HTML using BeautifulSoup."""
    soup = BeautifulSoup(html, 'html.parser')

    result = ParseResult(success=True)

    # Extract title
    title_tag = soup.find('title')
    if title_tag:
        result.title = title_tag.get_text().strip()

    # Extract metadata
    for meta in soup.find_all('meta'):
        name = meta.get('name') or meta.get('property')
        content = meta.get('content')
        if name and content:
            result.metadata[name] = content

    # Extract text
    if extract_text:
        # Remove script and style elements
        for tag in soup(['script', 'style', 'noscript']):
            tag.decompose()
        result.text = soup.get_text(separator=' ', strip=True)

    # Extract links
    if extract_links:
        for a in soup.find_all('a', href=True):
            href = a['href']
            if base_url:
                href = urljoin(base_url, href)
            result.links.append({
                'href': href,
                'text': a.get_text().strip(),
            })

    # Extract images
    if extract_images:
        for img in soup.find_all('img', src=True):
            src = img['src']
            if base_url:
                src = urljoin(base_url, src)
            result.images.append({
                'src': src,
                'alt': img.get('alt', ''),
            })

    # Extract code blocks
    if extract_code:
        for code in soup.find_all(['code', 'pre']):
            code_text = code.get_text().strip()
            if code_text:
                result.code_blocks.append(code_text)

    # Extract headings
    for level in range(1, 7):
        for heading in soup.find_all(f'h{level}'):
            result.headings.append({
                'level': level,
                'text': heading.get_text().strip(),
            })

    return result


def _parse_with_basic_parser(
    html: str,
    base_url: str,
    extract_text: bool,
    extract_links: bool,
    extract_images: bool,
    extract_code: bool,
) -> ParseResult:
    """Parse HTML using basic HTMLParser."""
    parser = BasicHTMLParser(base_url)

    try:
        parser.feed(html)
    except Exception as exc:
        return ParseResult(success=False, error=f"Parse error: {exc}")

    return ParseResult(
        success=True,
        title=parser.title.strip(),
        text=parser.get_text() if extract_text else "",
        links=parser.links if extract_links else [],
        images=parser.images if extract_images else [],
        code_blocks=parser.code_blocks if extract_code else [],
        headings=parser.headings,
        metadata=parser.metadata,
    )


def extract_text(html: str) -> str:
    """Extract plain text from HTML.

    Args:
        html: HTML content

    Returns:
        Extracted text
    """
    result = parse_html(html, extract_text=True, extract_links=False, extract_images=False, extract_code=False)
    return result.text if result.success else ""


def extract_links(html: str, base_url: str = "") -> List[Dict[str, str]]:
    """Extract links from HTML.

    Args:
        html: HTML content
        base_url: Base URL for resolving relative links

    Returns:
        List of link dicts with 'href' and 'text'
    """
    result = parse_html(html, base_url, extract_text=False, extract_links=True, extract_images=False, extract_code=False)
    return result.links if result.success else []


def extract_code_blocks(html: str) -> List[str]:
    """Extract code blocks from HTML.

    Args:
        html: HTML content

    Returns:
        List of code block strings
    """
    result = parse_html(html, extract_text=False, extract_links=False, extract_images=False, extract_code=True)
    return result.code_blocks if result.success else []


def select(html: str, selector: str) -> List[str]:
    """Select elements using CSS selector (requires BeautifulSoup).

    Args:
        html: HTML content
        selector: CSS selector

    Returns:
        List of matching element texts
    """
    if not BS4_AVAILABLE:
        logger.warning("CSS selectors require BeautifulSoup")
        return []

    try:
        soup = BeautifulSoup(html, 'html.parser')
        elements = soup.select(selector)
        return [el.get_text().strip() for el in elements]
    except Exception as exc:
        logger.error(f"Selector error: {exc}")
        return []
