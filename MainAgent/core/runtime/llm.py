"""LLM wrapper for Groq API interactions."""

from __future__ import annotations

import os
import time
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

# Valid Groq models (as of 2024)
VALID_GROQ_MODELS = [
    "llama-3.3-70b-versatile",
    "llama-3.1-70b-versatile",
    "llama-3.1-8b-instant",
    "llama3-70b-8192",
    "llama3-8b-8192",
    "mixtral-8x7b-32768",
    "gemma2-9b-it",
    "gemma-7b-it",
]

DEFAULT_MODEL = "llama-3.3-70b-versatile"


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

        # Get model from config or use default
        configured_model = (
            model
            or os.getenv("GROQ_MODEL")
            or CONFIG.get("groq_model")
        )

        # Validate and set model
        if configured_model and configured_model in VALID_GROQ_MODELS:
            self.model = configured_model
        else:
            if configured_model:
                print(f"[WARN] Model '{configured_model}' not in known models, using default: {DEFAULT_MODEL}")
            self.model = DEFAULT_MODEL

        self._client = Groq(api_key=self.api_key)

    def stream_completion(
        self,
        messages: List[dict],
        *,
        max_tokens: int = 2048,
        temperature: float = 0.7,
        top_p: float = 1.0,
        stop: Optional[Iterable[str]] = None,
    ) -> str:
        """Create a chat completion request and stream the result to stdout.

        Returns the accumulated response string.
        """
        try:
            completion = self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                stream=True,
                stop=stop,
            )

            full_response = ""
            for chunk in completion:
                if chunk.choices and chunk.choices[0].delta.content:
                    text = chunk.choices[0].delta.content
                    print(text, end="", flush=True)
                    full_response += text

            print("\n")
            return full_response

        except Exception as exc:
            print(f"\n[ERROR] Stream completion failed: {exc}\n")
            raise

    def completion(
        self,
        messages: List[dict],
        *,
        max_tokens: int = 2048,
        temperature: float = 0.7,
        top_p: float = 1.0,
        stop: Optional[Iterable[str]] = None,
    ) -> str:
        """Create a non-streaming chat completion request.

        Returns the response string.
        """
        try:
            completion = self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                stream=False,
                stop=stop,
            )

            if completion.choices and completion.choices[0].message.content:
                return completion.choices[0].message.content
            return ""

        except Exception as exc:
            print(f"\n[ERROR] Completion failed: {exc}\n")
            raise


def llm_call(
    prompt: str,
    *,
    service: Optional[GroqService] = None,
    max_tokens: int = 2048,
    temperature: float = 0.7,
    retries: int = 3,
    retry_delay: float = 1.0,
) -> str:
    """Convenience helper that sends a single-turn prompt to the Groq API.

    Args:
        prompt: The user prompt to send
        service: Optional pre-configured GroqService
        max_tokens: Maximum tokens in response
        temperature: Sampling temperature (0.0-1.0)
        retries: Number of retry attempts on failure
        retry_delay: Base delay between retries (exponential backoff)

    Returns:
        The LLM response string, or empty string on failure
    """
    groq_service = service or GroqService()

    for attempt in range(retries):
        try:
            return groq_service.stream_completion(
                [{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature,
            )
        except Exception as exc:
            if attempt < retries - 1:
                wait_time = retry_delay * (2 ** attempt)  # Exponential backoff
                print(f"\n[WARN] API call failed (attempt {attempt + 1}/{retries}): {exc}")
                print(f"       Retrying in {wait_time}s...\n")
                time.sleep(wait_time)
            else:
                print(f"\n[ERROR] API Error after {retries} attempts: {exc}\n")
                return ""

    return ""


def llm_call_with_system(
    system_prompt: str,
    user_prompt: str,
    *,
    service: Optional[GroqService] = None,
    max_tokens: int = 2048,
    temperature: float = 0.7,
) -> str:
    """Send a prompt with a system message.

    Args:
        system_prompt: The system/instruction prompt
        user_prompt: The user's request
        service: Optional pre-configured GroqService
        max_tokens: Maximum tokens in response
        temperature: Sampling temperature

    Returns:
        The LLM response string
    """
    groq_service = service or GroqService()

    try:
        return groq_service.stream_completion(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=max_tokens,
            temperature=temperature,
        )
    except Exception as exc:
        print(f"\n[ERROR] API Error: {exc}\n")
        return ""
