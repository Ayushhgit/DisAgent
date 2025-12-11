"""LLM wrapper for Groq API interactions."""

from __future__ import annotations

import os
from typing import Iterable, List, Optional

from groq import Groq
from dotenv import load_dotenv

load_dotenv()

try:
    from ...config import CONFIG
except Exception:
    # Fallback if config not available
    try:
        from MainAgent.config import CONFIG  # last resort for older import styles
    except Exception:
        CONFIG = {}


class GroqService:
    """Thin wrapper around the Groq chat completion client with streaming support."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
    ) -> None:
        # Try in order: explicit parameter > environment variable > config file
        self.api_key = (
            api_key
            or os.getenv("GROQ_API_KEY")
            or CONFIG.get("groq_api_key")
        )
        if not self.api_key:
            raise ValueError(
                "Groq API key not configured.\n"
                "Set it using one of these methods:\n"
                "  1. Environment variable: set GROQ_API_KEY=your_key_here\n"
                "  2. In config.py: CONFIG['groq_api_key'] = 'your_key_here'\n"
                "  3. Pass explicitly: GroqService(api_key='your_key_here')"
            )

        self.model = (
            model
            or os.getenv("GROQ_MODEL")
            or CONFIG.get("groq_model", "openai/gpt-oss-120b")
        )
        self._client = Groq(api_key=self.api_key)

    def stream_completion(
        self,
        messages: List[dict],
        *,
        max_tokens: int = 2048,
        temperature: float = 0.7,
        top_p: float = 1.0,
        reasoning_effort: Optional[str] = None,
        stop: Optional[Iterable[str]] = None,
    ) -> str:
        """Create a chat completion request and stream the result to stdout.

        Returns the accumulated response string.
        """

        completion = self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_completion_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            reasoning_effort=reasoning_effort,
            stream=True,
            stop=stop,
        )

        full_response = ""
        for chunk in completion:
            text = chunk.choices[0].delta.content or ""
            print(text, end="", flush=True)
            full_response += text

        print("\n")
        return full_response


def llm_call(
    prompt: str,
    *,
    service: Optional[GroqService] = None,
    max_tokens: int = 2048,
    temperature: float = 0.7,
    reasoning_effort: Optional[str] = None,
) -> str:
    """Convenience helper that sends a single-turn prompt to the Groq API."""

    groq_service = service or GroqService()
    try:
        return groq_service.stream_completion(
            [{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
            reasoning_effort=reasoning_effort,
        )
    except Exception as exc:  # pragma: no cover - defensive logging for runtime issues
        print(f"\n❌ API Error: {exc}\n")
        return ""

