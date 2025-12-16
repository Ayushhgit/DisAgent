# ============================================================================
# FILE: security.py
# Security utilities for input validation, prompt injection protection,
# and output sanitization
# ============================================================================

import re
import html
from typing import Optional, List, Tuple, Set
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

# Import custom exceptions
try:
    from .exceptions import (
        PromptInjectionError,
        InvalidInputError,
        EmptyInputError,
        UnsafeOperationError,
    )
except ImportError:
    class PromptInjectionError(Exception):
        pass
    class InvalidInputError(Exception):
        pass
    class EmptyInputError(Exception):
        pass
    class UnsafeOperationError(Exception):
        pass


# === Prompt Injection Detection ===

# Patterns that indicate prompt injection attempts
INJECTION_PATTERNS = [
    # Direct instruction override attempts
    r'ignore\s+(all\s+)?(previous|prior|above)\s+(instructions?|prompts?|rules?)',
    r'disregard\s+(all\s+)?(previous|prior|above)',
    r'forget\s+(everything|all|what)\s+(you|i)\s+(said|told|mentioned)',
    r'new\s+instructions?\s*:',
    r'system\s*:\s*you\s+are\s+now',
    r'override\s+(instructions?|rules?|constraints?)',

    # Role manipulation
    r'you\s+are\s+now\s+(a|an|the)\s+',
    r'pretend\s+(to\s+be|you\s+are)',
    r'act\s+as\s+(if|though)\s+you',
    r'roleplay\s+as',
    r'switch\s+(to|into)\s+.+\s+mode',

    # Jailbreak attempts
    r'dan\s+mode',
    r'developer\s+mode\s+(enabled|activated)',
    r'jailbreak',
    r'bypass\s+(safety|restrictions?|filters?)',
    r'unlock\s+(hidden|secret)\s+capabilities',

    # System prompt extraction
    r'(show|reveal|display|print|output)\s+(your|the)\s+(system\s+)?(prompt|instructions?)',
    r'what\s+(is|are)\s+your\s+(system\s+)?(prompt|instructions?)',
    r'repeat\s+(your|the)\s+(system\s+)?prompt',

    # Delimiter injection
    r'```system',
    r'\[SYSTEM\]',
    r'\[INST\]',
    r'<\|im_start\|>',
    r'<\|im_end\|>',
    r'###\s*(system|instruction)',

    # Code execution attempts
    r'exec\s*\(',
    r'eval\s*\(',
    r'__import__',
    r'os\.(system|popen|exec)',
    r'subprocess\.(run|call|Popen)',
]

# Compiled patterns for efficiency
_COMPILED_PATTERNS = [re.compile(p, re.IGNORECASE) for p in INJECTION_PATTERNS]


@dataclass
class SecurityCheckResult:
    """Result of a security check."""
    is_safe: bool
    risk_level: str  # "none", "low", "medium", "high", "critical"
    issues: List[str]
    sanitized_input: Optional[str] = None

    def __bool__(self):
        return self.is_safe


def detect_prompt_injection(text: str, strict: bool = False) -> SecurityCheckResult:
    """Detect potential prompt injection attempts.

    Args:
        text: The input text to check
        strict: If True, be more aggressive in detection

    Returns:
        SecurityCheckResult with findings
    """
    if not text:
        return SecurityCheckResult(
            is_safe=True,
            risk_level="none",
            issues=[],
            sanitized_input=""
        )

    issues = []
    text_lower = text.lower()

    # Check against injection patterns
    for i, pattern in enumerate(_COMPILED_PATTERNS):
        if pattern.search(text):
            issues.append(f"Suspicious pattern detected: {INJECTION_PATTERNS[i][:50]}...")

    # Check for excessive special characters that might indicate delimiter injection
    special_char_ratio = len(re.findall(r'[<>\[\]{}|`#]', text)) / max(len(text), 1)
    if special_char_ratio > 0.15:
        issues.append("High ratio of special characters detected")

    # Check for unusual Unicode that might be used to bypass filters
    if re.search(r'[\u200b-\u200f\u2028-\u202f\ufeff]', text):
        issues.append("Invisible Unicode characters detected")

    # In strict mode, also check for suspicious formatting
    if strict:
        if text.count('\n') > 20:
            issues.append("Excessive line breaks")
        if re.search(r'={10,}|#{10,}|-{10,}', text):
            issues.append("Suspicious delimiter sequences")

    # Determine risk level
    if not issues:
        risk_level = "none"
    elif len(issues) == 1 and "special characters" in issues[0]:
        risk_level = "low"
    elif len(issues) <= 2:
        risk_level = "medium"
    else:
        risk_level = "high"

    return SecurityCheckResult(
        is_safe=len(issues) == 0,
        risk_level=risk_level,
        issues=issues,
        sanitized_input=sanitize_prompt_input(text) if issues else text
    )


def sanitize_prompt_input(text: str) -> str:
    """Sanitize user input for safe inclusion in prompts.

    Args:
        text: Raw user input

    Returns:
        Sanitized text safe for prompt inclusion
    """
    if not text:
        return ""

    # Remove null bytes
    text = text.replace('\x00', '')

    # Remove invisible Unicode characters
    text = re.sub(r'[\u200b-\u200f\u2028-\u202f\ufeff]', '', text)

    # Escape common delimiter patterns
    text = text.replace('```', '` ` `')
    text = text.replace('###', '# # #')
    text = text.replace('---', '- - -')

    # Normalize whitespace (but preserve intentional formatting)
    text = re.sub(r'\n{4,}', '\n\n\n', text)  # Max 3 consecutive newlines
    text = re.sub(r' {4,}', '   ', text)  # Max 3 consecutive spaces

    # Limit length to prevent context stuffing
    max_length = 50000  # Reasonable limit for user input
    if len(text) > max_length:
        text = text[:max_length] + "\n[Input truncated]"

    return text.strip()


def wrap_user_input(text: str, label: str = "USER_INPUT") -> str:
    """Wrap user input with clear delimiters for LLM context.

    This helps the LLM distinguish user input from system instructions.

    Args:
        text: User input to wrap
        label: Label to use for the wrapper

    Returns:
        Wrapped text with clear boundaries
    """
    sanitized = sanitize_prompt_input(text)
    return f"<{label}>\n{sanitized}\n</{label}>"


# === Input Validation ===

def validate_user_prompt(
    prompt: str,
    min_length: int = 1,
    max_length: int = 50000,
    allow_empty: bool = False,
    check_injection: bool = True
) -> Tuple[str, SecurityCheckResult]:
    """Validate and sanitize user prompt input.

    Args:
        prompt: Raw user prompt
        min_length: Minimum allowed length
        max_length: Maximum allowed length
        allow_empty: Whether to allow empty prompts
        check_injection: Whether to check for injection attempts

    Returns:
        Tuple of (sanitized_prompt, security_result)

    Raises:
        EmptyInputError: If prompt is empty and not allowed
        InvalidInputError: If prompt fails validation
        PromptInjectionError: If injection detected (only in strict mode)
    """
    # Handle None
    if prompt is None:
        prompt = ""

    # Strip whitespace
    prompt = prompt.strip()

    # Check empty
    if not prompt:
        if not allow_empty:
            raise EmptyInputError("prompt")
        return "", SecurityCheckResult(is_safe=True, risk_level="none", issues=[])

    # Check length
    if len(prompt) < min_length:
        raise InvalidInputError("prompt", f"Too short (min {min_length} chars)", prompt[:50])

    if len(prompt) > max_length:
        raise InvalidInputError("prompt", f"Too long (max {max_length} chars)")

    # Check for injection
    security_result = SecurityCheckResult(is_safe=True, risk_level="none", issues=[])
    if check_injection:
        security_result = detect_prompt_injection(prompt)
        if security_result.risk_level in ("high", "critical"):
            logger.warning(f"Prompt injection risk detected: {security_result.issues}")
            # Use sanitized version instead of raising
            prompt = security_result.sanitized_input or prompt

    # Basic sanitization
    sanitized = sanitize_prompt_input(prompt)

    return sanitized, security_result


def validate_file_path(
    path: str,
    base_path: Optional[str] = None,
    allow_absolute: bool = False
) -> str:
    """Validate a file path for safety.

    Args:
        path: Path to validate
        base_path: If provided, ensure path is within this directory
        allow_absolute: Whether to allow absolute paths

    Returns:
        Validated path

    Raises:
        InvalidInputError: If path is invalid or unsafe
    """
    if not path:
        raise EmptyInputError("file_path")

    # Normalize path
    path = path.strip()

    # Check for path traversal
    if '..' in path:
        raise InvalidInputError("file_path", "Path traversal not allowed", path)

    # Check for null bytes
    if '\x00' in path:
        raise InvalidInputError("file_path", "Null bytes not allowed", path)

    # Check for absolute paths
    if not allow_absolute and (path.startswith('/') or (len(path) > 1 and path[1] == ':')):
        raise InvalidInputError("file_path", "Absolute paths not allowed", path)

    # Check for dangerous filenames
    dangerous_names = {'.env', '.git', '.ssh', 'id_rsa', 'id_ed25519', 'credentials', 'secrets'}
    basename = path.split('/')[-1].split('\\')[-1].lower()
    if basename in dangerous_names:
        raise InvalidInputError("file_path", f"Access to '{basename}' is restricted", path)

    return path


# === Output Sanitization ===

def sanitize_llm_output_for_file(content: str, file_type: str = "text") -> str:
    """Sanitize LLM output before writing to a file.

    Args:
        content: LLM-generated content
        file_type: Type of file (text, code, html, etc.)

    Returns:
        Sanitized content safe for file writing
    """
    if not content:
        return ""

    # Remove null bytes (always unsafe)
    content = content.replace('\x00', '')

    # Remove ANSI escape codes
    content = re.sub(r'\x1b\[[0-9;]*m', '', content)

    # For HTML files, escape potentially dangerous content
    if file_type == "html":
        # Don't fully escape - just remove script tags and event handlers
        content = re.sub(r'<script\b[^>]*>[\s\S]*?</script>', '', content, flags=re.IGNORECASE)
        content = re.sub(r'\s+on\w+\s*=\s*["\'][^"\']*["\']', '', content, flags=re.IGNORECASE)

    return content


def sanitize_error_message(error: str, hide_paths: bool = True) -> str:
    """Sanitize error messages before displaying to user.

    Removes potentially sensitive information like full paths.

    Args:
        error: Raw error message
        hide_paths: Whether to redact full filesystem paths

    Returns:
        Sanitized error message
    """
    if not error:
        return ""

    # Remove ANSI codes
    error = re.sub(r'\x1b\[[0-9;]*m', '', error)

    if hide_paths:
        # Redact absolute paths (Unix)
        error = re.sub(r'/(?:home|Users|root)/[^\s:]+', '[PATH_REDACTED]', error)
        # Redact absolute paths (Windows)
        error = re.sub(r'[A-Z]:\\[^\s:]+', '[PATH_REDACTED]', error)

    # Limit length
    if len(error) > 1000:
        error = error[:1000] + "... [truncated]"

    return error


# === API Key Protection ===

def mask_api_key(key: str) -> str:
    """Mask an API key for safe logging.

    Args:
        key: The API key to mask

    Returns:
        Masked version showing only first/last few chars
    """
    if not key or len(key) < 8:
        return "***"

    return f"{key[:4]}...{key[-4:]}"


def detect_secrets_in_text(text: str) -> List[str]:
    """Detect potential secrets or API keys in text.

    Args:
        text: Text to scan

    Returns:
        List of potential secret patterns found
    """
    findings = []

    secret_patterns = [
        (r'(?:api[_-]?key|apikey)\s*[=:]\s*["\']?([a-zA-Z0-9_-]{20,})["\']?', "API key"),
        (r'(?:password|passwd|pwd)\s*[=:]\s*["\']([^"\']{8,})["\']', "Password"),
        (r'(?:secret|token)\s*[=:]\s*["\']?([a-zA-Z0-9_-]{20,})["\']?', "Secret/Token"),
        (r'(?:aws_access_key_id)\s*[=:]\s*["\']?([A-Z0-9]{20})["\']?', "AWS Access Key"),
        (r'(?:aws_secret_access_key)\s*[=:]\s*["\']?([a-zA-Z0-9/+=]{40})["\']?', "AWS Secret Key"),
        (r'sk-[a-zA-Z0-9]{32,}', "OpenAI API Key"),
        (r'ghp_[a-zA-Z0-9]{36}', "GitHub Personal Access Token"),
        (r'gsk_[a-zA-Z0-9]{50,}', "Groq API Key"),
    ]

    text_lower = text.lower()
    for pattern, name in secret_patterns:
        if re.search(pattern, text, re.IGNORECASE):
            findings.append(name)

    return findings


# === Convenience Functions ===

def safe_prompt(user_input: str, context: str = "") -> str:
    """Create a safe prompt from user input with proper wrapping.

    Args:
        user_input: Raw user input
        context: Optional context to include

    Returns:
        Safe prompt string
    """
    sanitized, _ = validate_user_prompt(user_input)

    prompt_parts = []
    if context:
        prompt_parts.append(context)

    prompt_parts.append(wrap_user_input(sanitized))

    return "\n\n".join(prompt_parts)


def is_safe_to_write(content: str, file_path: str) -> Tuple[bool, List[str]]:
    """Check if content is safe to write to a file.

    Args:
        content: Content to write
        file_path: Target file path

    Returns:
        Tuple of (is_safe, list_of_issues)
    """
    issues = []

    # Check for secrets
    secrets = detect_secrets_in_text(content)
    if secrets:
        issues.extend([f"Potential {s} detected in content" for s in secrets])

    # Check file path
    try:
        validate_file_path(file_path)
    except (InvalidInputError, EmptyInputError) as e:
        issues.append(str(e))

    # Check for suspicious patterns in code files
    code_extensions = {'.py', '.js', '.ts', '.sh', '.bash', '.ps1'}
    if any(file_path.lower().endswith(ext) for ext in code_extensions):
        if 'os.system' in content or 'subprocess.call' in content:
            if 'shell=True' in content:
                issues.append("Potentially unsafe shell execution in code")

    return len(issues) == 0, issues
