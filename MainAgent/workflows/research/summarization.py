"""
Summarization Workflow - Summarize text and content.

This workflow provides various summarization strategies:
- Extractive summarization (selecting key sentences)
- Keyword extraction
- Key points extraction
- Content compression
"""

from __future__ import annotations

import re
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from collections import Counter
from enum import Enum

logger = logging.getLogger(__name__)


class SummaryType(Enum):
    """Types of summaries."""
    BRIEF = "brief"  # 1-2 sentences
    STANDARD = "standard"  # Paragraph
    DETAILED = "detailed"  # Multiple paragraphs
    BULLET_POINTS = "bullet_points"  # Key points as bullets


@dataclass
class SummaryResult:
    """Result of summarization."""
    summary: str
    key_points: List[str] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)
    word_count_original: int = 0
    word_count_summary: int = 0

    @property
    def compression_ratio(self) -> float:
        """Calculate compression ratio."""
        if self.word_count_original == 0:
            return 0.0
        return 1 - (self.word_count_summary / self.word_count_original)


def summarize(text: str) -> str:
    """Summarize text (legacy interface).

    Args:
        text: Text to summarize

    Returns:
        Summary text
    """
    result = summarize_text(text)
    return result.summary


def summarize_text(
    text: str,
    summary_type: SummaryType = SummaryType.STANDARD,
    max_sentences: Optional[int] = None,
    preserve_structure: bool = False,
) -> SummaryResult:
    """Summarize text using extractive summarization.

    Args:
        text: Text to summarize
        summary_type: Type of summary to generate
        max_sentences: Maximum sentences in summary
        preserve_structure: Keep paragraph structure

    Returns:
        SummaryResult with summary and metadata
    """
    if not text or not text.strip():
        return SummaryResult(
            summary="",
            word_count_original=0,
            word_count_summary=0,
        )

    # Clean and prepare text
    cleaned_text = _clean_text(text)
    words = cleaned_text.split()
    word_count_original = len(words)

    # Determine target length based on summary type
    if max_sentences is None:
        max_sentences = {
            SummaryType.BRIEF: 2,
            SummaryType.STANDARD: 5,
            SummaryType.DETAILED: 10,
            SummaryType.BULLET_POINTS: 7,
        }.get(summary_type, 5)

    # Extract sentences
    sentences = _split_into_sentences(cleaned_text)

    if len(sentences) <= max_sentences:
        # Text is already short enough
        return SummaryResult(
            summary=cleaned_text,
            key_points=sentences,
            keywords=_extract_keywords(cleaned_text),
            word_count_original=word_count_original,
            word_count_summary=len(cleaned_text.split()),
        )

    # Score sentences
    scored_sentences = _score_sentences(sentences, cleaned_text)

    # Select top sentences
    selected = sorted(scored_sentences, key=lambda x: x[1], reverse=True)[:max_sentences]

    # Reorder by original position for coherence
    selected.sort(key=lambda x: sentences.index(x[0]))

    # Format summary based on type
    if summary_type == SummaryType.BULLET_POINTS:
        summary = '\n'.join(f"• {s[0]}" for s in selected)
    elif preserve_structure:
        summary = _preserve_paragraph_structure(text, [s[0] for s in selected])
    else:
        summary = ' '.join(s[0] for s in selected)

    # Extract additional metadata
    key_points = [s[0] for s in selected]
    keywords = _extract_keywords(cleaned_text)

    return SummaryResult(
        summary=summary,
        key_points=key_points,
        keywords=keywords,
        word_count_original=word_count_original,
        word_count_summary=len(summary.split()),
    )


def _clean_text(text: str) -> str:
    """Clean text for processing."""
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove URLs
    text = re.sub(r'https?://\S+', '', text)
    # Remove email addresses
    text = re.sub(r'\S+@\S+\.\S+', '', text)
    return text.strip()


def _split_into_sentences(text: str) -> List[str]:
    """Split text into sentences."""
    # Simple sentence splitting
    sentences = re.split(r'(?<=[.!?])\s+', text)

    # Filter and clean
    result = []
    for s in sentences:
        s = s.strip()
        if len(s) > 10:  # Minimum sentence length
            result.append(s)

    return result


def _score_sentences(sentences: List[str], full_text: str) -> List[Tuple[str, float]]:
    """Score sentences by importance."""
    # Calculate word frequencies
    words = re.findall(r'\b\w+\b', full_text.lower())
    word_freq = Counter(words)

    # Remove common stop words
    stop_words = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
        'of', 'with', 'by', 'from', 'is', 'are', 'was', 'were', 'be', 'been',
        'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
        'could', 'should', 'may', 'might', 'must', 'it', 'its', 'this', 'that',
        'these', 'those', 'i', 'you', 'he', 'she', 'we', 'they', 'what', 'which',
        'who', 'whom', 'how', 'when', 'where', 'why', 'as', 'if', 'not', 'no',
        'can', 'so', 'than', 'too', 'very', 'just', 'also', 'now', 'here',
    }

    for word in stop_words:
        word_freq.pop(word, None)

    # Score each sentence
    scored = []
    for i, sentence in enumerate(sentences):
        sentence_words = re.findall(r'\b\w+\b', sentence.lower())

        # Word frequency score
        freq_score = sum(word_freq.get(w, 0) for w in sentence_words)
        if sentence_words:
            freq_score /= len(sentence_words)

        # Position score (first and last sentences are often important)
        position_score = 0
        if i == 0:
            position_score = 2.0
        elif i == len(sentences) - 1:
            position_score = 1.0
        elif i < 3:
            position_score = 0.5

        # Length score (prefer medium-length sentences)
        length = len(sentence_words)
        if 10 <= length <= 30:
            length_score = 1.0
        elif length < 10:
            length_score = 0.5
        else:
            length_score = 0.7

        # Keyword indicators
        indicator_score = 0
        indicators = ['important', 'significant', 'key', 'main', 'essential',
                      'critical', 'fundamental', 'primarily', 'notably',
                      'conclusion', 'result', 'finding', 'summary']
        for indicator in indicators:
            if indicator in sentence.lower():
                indicator_score += 0.5

        # Combined score
        total_score = (freq_score * 2 + position_score + length_score + indicator_score)
        scored.append((sentence, total_score))

    return scored


def _extract_keywords(text: str, max_keywords: int = 10) -> List[str]:
    """Extract keywords from text."""
    # Tokenize
    words = re.findall(r'\b\w+\b', text.lower())

    # Filter stop words and short words
    stop_words = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
        'of', 'with', 'by', 'from', 'is', 'are', 'was', 'were', 'be', 'been',
        'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
        'could', 'should', 'may', 'might', 'must', 'it', 'its', 'this', 'that',
    }

    filtered = [w for w in words if w not in stop_words and len(w) > 3]

    # Count frequencies
    freq = Counter(filtered)

    # Return top keywords
    return [word for word, count in freq.most_common(max_keywords)]


def _preserve_paragraph_structure(
    original: str,
    selected_sentences: List[str],
) -> str:
    """Preserve paragraph structure in summary."""
    paragraphs = original.split('\n\n')
    result_paragraphs = []

    for para in paragraphs:
        para_sentences = []
        for sentence in selected_sentences:
            if sentence in para:
                para_sentences.append(sentence)

        if para_sentences:
            result_paragraphs.append(' '.join(para_sentences))

    return '\n\n'.join(result_paragraphs)


def extract_key_points(
    text: str,
    max_points: int = 5,
) -> List[str]:
    """Extract key points from text.

    Args:
        text: Text to analyze
        max_points: Maximum number of points

    Returns:
        List of key points
    """
    result = summarize_text(
        text,
        summary_type=SummaryType.BULLET_POINTS,
        max_sentences=max_points,
    )
    return result.key_points


def compress_text(
    text: str,
    target_ratio: float = 0.5,
) -> str:
    """Compress text to target ratio.

    Args:
        text: Text to compress
        target_ratio: Target compression ratio (0-1)

    Returns:
        Compressed text
    """
    words = text.split()
    target_words = int(len(words) * (1 - target_ratio))

    # Estimate sentences needed
    avg_sentence_length = 15
    target_sentences = max(1, target_words // avg_sentence_length)

    result = summarize_text(
        text,
        max_sentences=target_sentences,
    )

    return result.summary


def summarize_document(
    sections: Dict[str, str],
    section_weights: Optional[Dict[str, float]] = None,
) -> Dict[str, str]:
    """Summarize a document with multiple sections.

    Args:
        sections: Dictionary of section name to content
        section_weights: Optional weights for section importance

    Returns:
        Dictionary of section summaries
    """
    weights = section_weights or {}
    summaries = {}

    for section_name, content in sections.items():
        # Determine summary length based on weight
        weight = weights.get(section_name, 1.0)

        if weight < 0.5:
            summary_type = SummaryType.BRIEF
        elif weight < 1.0:
            summary_type = SummaryType.STANDARD
        else:
            summary_type = SummaryType.DETAILED

        result = summarize_text(content, summary_type=summary_type)
        summaries[section_name] = result.summary

    return summaries


def generate_abstract(
    title: str,
    content: str,
    max_words: int = 150,
) -> str:
    """Generate an abstract for a document.

    Args:
        title: Document title
        content: Document content
        max_words: Maximum words in abstract

    Returns:
        Abstract text
    """
    # Get key sentences
    result = summarize_text(
        content,
        summary_type=SummaryType.STANDARD,
    )

    # Build abstract
    abstract_parts = []

    # Add context from title
    if title:
        abstract_parts.append(f"This document discusses {title.lower()}.")

    # Add key points
    for point in result.key_points[:3]:
        if len(' '.join(abstract_parts + [point]).split()) <= max_words:
            abstract_parts.append(point)

    abstract = ' '.join(abstract_parts)

    # Trim to max words if needed
    words = abstract.split()
    if len(words) > max_words:
        abstract = ' '.join(words[:max_words]) + '...'

    return abstract


def create_executive_summary(
    content: str,
    include_recommendations: bool = True,
) -> Dict[str, Any]:
    """Create an executive summary.

    Args:
        content: Full content to summarize
        include_recommendations: Include recommendations section

    Returns:
        Dictionary with summary components
    """
    result = summarize_text(
        content,
        summary_type=SummaryType.DETAILED,
    )

    executive_summary = {
        "overview": result.summary,
        "key_findings": result.key_points[:5],
        "keywords": result.keywords[:7],
        "statistics": {
            "original_words": result.word_count_original,
            "summary_words": result.word_count_summary,
            "compression": f"{result.compression_ratio:.1%}",
        },
    }

    if include_recommendations:
        # Extract action items (sentences with action verbs)
        action_verbs = ['should', 'must', 'need', 'recommend', 'suggest',
                        'consider', 'ensure', 'require', 'implement']
        recommendations = []

        sentences = _split_into_sentences(content)
        for sentence in sentences:
            lower_sentence = sentence.lower()
            if any(verb in lower_sentence for verb in action_verbs):
                recommendations.append(sentence)

        executive_summary["recommendations"] = recommendations[:5]

    return executive_summary


def summarize_code_documentation(
    docstrings: List[str],
    include_params: bool = True,
) -> str:
    """Summarize code documentation.

    Args:
        docstrings: List of docstrings to summarize
        include_params: Include parameter information

    Returns:
        Summary of documentation
    """
    if not docstrings:
        return "No documentation available."

    all_docs = '\n\n'.join(docstrings)

    # Extract first lines (usually the summary)
    summaries = []
    param_info = []

    for doc in docstrings:
        lines = doc.strip().split('\n')
        if lines:
            # First non-empty line is usually the summary
            for line in lines:
                stripped = line.strip()
                if stripped and not stripped.startswith(':') and not stripped.startswith('Args'):
                    summaries.append(stripped)
                    break

            # Extract param info if requested
            if include_params:
                in_params = False
                for line in lines:
                    stripped = line.strip()
                    if stripped.startswith('Args:') or stripped.startswith('Parameters:'):
                        in_params = True
                    elif stripped.startswith('Returns:') or stripped.startswith('Raises:'):
                        in_params = False
                    elif in_params and stripped:
                        param_info.append(stripped)

    # Build summary
    result_parts = []

    if summaries:
        unique_summaries = list(dict.fromkeys(summaries))  # Preserve order, remove dupes
        result_parts.append("Functions/Classes:")
        for summary in unique_summaries[:10]:
            result_parts.append(f"  • {summary}")

    if include_params and param_info:
        result_parts.append("\nKey Parameters:")
        unique_params = list(dict.fromkeys(param_info))[:10]
        for param in unique_params:
            result_parts.append(f"  • {param}")

    return '\n'.join(result_parts) if result_parts else "Documentation summary not available."


def compare_texts(
    text1: str,
    text2: str,
    return_common: bool = True,
) -> Dict[str, Any]:
    """Compare two texts and find similarities/differences.

    Args:
        text1: First text
        text2: Second text
        return_common: Include common elements

    Returns:
        Comparison results
    """
    # Extract keywords from both
    keywords1 = set(_extract_keywords(text1, max_keywords=20))
    keywords2 = set(_extract_keywords(text2, max_keywords=20))

    # Get key points from both
    points1 = extract_key_points(text1, max_points=5)
    points2 = extract_key_points(text2, max_points=5)

    result = {
        "text1_keywords": list(keywords1),
        "text2_keywords": list(keywords2),
        "text1_key_points": points1,
        "text2_key_points": points2,
        "unique_to_text1": list(keywords1 - keywords2),
        "unique_to_text2": list(keywords2 - keywords1),
    }

    if return_common:
        result["common_keywords"] = list(keywords1 & keywords2)
        result["similarity_score"] = len(keywords1 & keywords2) / max(len(keywords1 | keywords2), 1)

    return result
