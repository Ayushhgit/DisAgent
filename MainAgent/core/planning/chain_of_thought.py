# core/planning/chain_of_thought.py
"""
Chain-of-Thought Processor

Responsibilities:
- Convert raw LLM/user reasoning text into a structured plan (key_points, ordered steps).
- Provide utilities to sanitize/shorten reasoning for logs and UIs.
- This module intentionally avoids exposing private chain-of-thought verbatim; it
  returns compact, actionable artifacts suitable for planning.
"""

from typing import List, Dict, Tuple
import re
import logging

logger = logging.getLogger(__name__)


class ChainOfThoughtProcessor:
    """
    Convert verbose reasoning into a structured, safe-to-store representation.

    Typical usage:
        processor = ChainOfThoughtProcessor()
        plan = processor.process_raw_reasoning(raw_llm_text)
    """

    # heuristics thresholds
    _SENTENCE_SPLIT_RE = re.compile(r'(?<=[.!?])\s+')

    def __init__(self, max_key_points: int = 6, max_steps: int = 12):
        self.max_key_points = max_key_points
        self.max_steps = max_steps

    def process_raw_reasoning(self, raw_text: str) -> Dict:
        """
        Main entrypoint. Returns:
          {
            "summary": "<one-line summary>",
            "key_points": ["...", ...],
            "steps": [
              {"id": "task-1", "title": "...", "detail": "..."},
              ...
            ]
          }
        """
        if not raw_text:
            return {"summary": "", "key_points": [], "steps": []}

        try:
            summary = self._one_line_summary(raw_text)
            points = self._extract_key_points(raw_text)
            steps = self._extract_ordered_steps(raw_text)
            return {
                "summary": summary,
                "key_points": points[: self.max_key_points],
                "steps": steps[: self.max_steps],
            }
        except Exception as e:
            logger.exception("Failed to process raw reasoning: %s", e)
            return {"summary": "", "key_points": [], "steps": []}

    def _one_line_summary(self, text: str) -> str:
        """
        Lightweight summarization: pick the first strong sentence, fallback to
        first 8 words.
        """
        sentences = self._SENTENCE_SPLIT_RE.split(text.strip())
        for s in sentences:
            s = s.strip()
            if not s:
                continue
            # prefer sentences with action verbs or key phrases
            if any(kw in s.lower() for kw in ("build", "refactor", "implement", "create", "test", "deploy", "analyze")):
                return self._truncate(s, 120)
        # fallback
        first = sentences[0] if sentences else ""
        return self._truncate(first, 120)

    def _extract_key_points(self, text: str) -> List[str]:
        """
        Heuristic extraction of key points: look for bullet-like lines, numbered lists,
        and highly informative sentences.
        """
        lines = [l.strip("-•* \t") for l in text.splitlines() if l.strip()]
        candidates: List[str] = []

        # prefer explicit bullets and numbered items
        for l in lines:
            if re.match(r'^\d+[\).\s]', l) or l.startswith(("-", "*", "•")):
                candidates.append(self._truncate(l, 240))

        # supplement with top sentences by length/keywords
        if len(candidates) < self.max_key_points:
            sentences = self._SENTENCE_SPLIT_RE.split(text)
            scored: List[Tuple[int, str]] = []
            for s in sentences:
                score = len(s)
                # bump if has verbs / keywords
                if any(k in s.lower() for k in ("api", "database", "schema", "test", "ci", "docker", "llm", "model")):
                    score += 40
                scored.append((score, s.strip()))
            scored.sort(reverse=True)
            for _, s in scored:
                if s and s not in candidates:
                    candidates.append(self._truncate(s, 240))
                if len(candidates) >= self.max_key_points:
                    break

        return [c for c in candidates if c]

    def _extract_ordered_steps(self, text: str) -> List[Dict]:
        """
        Extract an ordered list of steps. Look for numbered lists first, otherwise
        fall back to sentence segmentation and heuristics to form steps.
        """
        # try to find explicit numbered lists
        numbered = re.findall(r'(?:\d+\.\s+)([^\n]+)', text)
        steps: List[Dict] = []
        if numbered:
            for i, s in enumerate(numbered, start=1):
                steps.append({"id": f"step-{i}", "title": self._short_title(s), "detail": self._truncate(s.strip(), 600)})
            return steps

        # otherwise segment sentences and group into steps (heuristic grouping)
        sentences = [s.strip() for s in self._SENTENCE_SPLIT_RE.split(text) if s.strip()]
        # promote sentences containing verbs/actions into steps
        for i, s in enumerate(sentences):
            if any(v in s.lower() for v in ("implement", "create", "write", "add", "remove", "refactor", "test", "deploy", "configure")):
                steps.append({"id": f"step-{len(steps)+1}", "title": self._short_title(s), "detail": self._truncate(s, 600)})
        # fallback: use top sentences by length if still empty
        if not steps:
            for i, s in enumerate(sentences[: self.max_steps]):
                steps.append({"id": f"step-{i+1}", "title": self._short_title(s), "detail": self._truncate(s, 600)})
        return steps

    @staticmethod
    def _truncate(s: str, max_chars: int) -> str:
        if len(s) <= max_chars:
            return s
        return s[: max_chars - 1].rstrip() + "…"

    @staticmethod
    def _short_title(text: str, max_words: int = 8) -> str:
        words = text.strip().split()
        return " ".join(words[:max_words]).rstrip(".,;:") + ("…" if len(words) > max_words else "")
