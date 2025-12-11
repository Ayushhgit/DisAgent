"""Configuration for MainAgent system.

This module provides centralized configuration management for the DisAgent system.
Configuration can be set via:
1. Environment variables (highest priority)
2. This config file (defaults)
"""

import os
from typing import Optional, Dict, Any
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Valid Groq models
VALID_MODELS = [
    "llama-3.3-70b-versatile",
    "llama-3.1-70b-versatile",
    "llama-3.1-8b-instant",
    "llama3-70b-8192",
    "llama3-8b-8192",
    "mixtral-8x7b-32768",
    "gemma2-9b-it",
    "gemma-7b-it",
]

# Default model
DEFAULT_MODEL = "llama-3.3-70b-versatile"

# Base directory
BASE_DIR = Path(__file__).parent.resolve()

# Main configuration
CONFIG: Dict[str, Any] = {
    # Project settings
    "project_name": os.getenv("PROJECT_NAME", "dis-agent-project"),

    # LLM settings
    "groq_api_key": os.getenv("GROQ_API_KEY"),
    "groq_model": os.getenv("GROQ_MODEL", DEFAULT_MODEL),

    # Agent settings
    "max_tokens": int(os.getenv("MAX_TOKENS", "8192")),
    "temperature": float(os.getenv("TEMPERATURE", "0.7")),
    "max_retries": int(os.getenv("MAX_RETRIES", "3")),

    # Memory settings
    "memory_path": os.getenv("MEMORY_PATH", str(BASE_DIR / ".agent_memory")),
    "enable_vector_store": os.getenv("ENABLE_VECTOR_STORE", "false").lower() == "true",

    # Execution settings
    "max_workers": int(os.getenv("MAX_WORKERS", "4")),
    "poll_interval": float(os.getenv("POLL_INTERVAL", "0.5")),

    # Output settings
    "output_dir": os.getenv("OUTPUT_DIR", str(BASE_DIR / "output" / "projects")),

    # Logging settings
    "log_level": os.getenv("LOG_LEVEL", "INFO"),
    "log_file": os.getenv("LOG_FILE"),

    # Feature flags
    "enable_streaming": os.getenv("ENABLE_STREAMING", "true").lower() == "true",
    "enable_memory_persistence": os.getenv("ENABLE_MEMORY_PERSISTENCE", "true").lower() == "true",
}


def get_config(key: str, default: Any = None) -> Any:
    """Get a configuration value.

    Args:
        key: Configuration key
        default: Default value if key not found

    Returns:
        Configuration value
    """
    return CONFIG.get(key, default)


def set_config(key: str, value: Any) -> None:
    """Set a configuration value.

    Args:
        key: Configuration key
        value: Value to set
    """
    CONFIG[key] = value


def validate_config() -> tuple[bool, list[str]]:
    """Validate the configuration.

    Returns:
        Tuple of (is_valid, list of error messages)
    """
    errors = []

    # Check required settings
    if not CONFIG.get("groq_api_key"):
        errors.append(
            "GROQ_API_KEY is not set. "
            "Set it via environment variable or in .env file."
        )

    # Validate model
    model = CONFIG.get("groq_model")
    if model and model not in VALID_MODELS:
        errors.append(
            f"Invalid model: {model}. "
            f"Valid models: {', '.join(VALID_MODELS)}"
        )

    # Check paths
    memory_path = Path(CONFIG.get("memory_path", ""))
    if memory_path and not memory_path.parent.exists():
        try:
            memory_path.parent.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            errors.append(f"Cannot create memory path: {e}")

    return len(errors) == 0, errors


def print_config() -> None:
    """Print current configuration (hiding sensitive values)."""
    print("\n=== DisAgent Configuration ===")
    for key, value in CONFIG.items():
        # Hide sensitive values
        if "key" in key.lower() or "secret" in key.lower():
            display_value = "***" if value else "(not set)"
        else:
            display_value = value
        print(f"  {key}: {display_value}")
    print()
