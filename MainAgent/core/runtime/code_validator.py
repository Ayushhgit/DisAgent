"""
Code Validator - Syntax and basic validation for generated code.

Provides:
- Syntax validation for Python, JavaScript, HTML, CSS, JSON
- Error message extraction for LLM correction
- File-type detection
"""

from __future__ import annotations

import ast
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class FileType(Enum):
    """Supported file types for validation."""
    PYTHON = "python"
    JAVASCRIPT = "javascript"
    TYPESCRIPT = "typescript"
    HTML = "html"
    CSS = "css"
    JSON = "json"
    UNKNOWN = "unknown"


@dataclass
class ValidationResult:
    """Result of code validation."""
    valid: bool
    file_path: str
    file_type: FileType
    errors: List[str]
    warnings: List[str]
    line_number: Optional[int] = None

    @property
    def error_summary(self) -> str:
        """Get a summary of errors for LLM correction."""
        if not self.errors:
            return ""
        return f"File: {self.file_path}\nErrors:\n" + "\n".join(f"  - {e}" for e in self.errors)


def detect_file_type(file_path: str) -> FileType:
    """Detect file type from extension."""
    ext = Path(file_path).suffix.lower()

    type_map = {
        ".py": FileType.PYTHON,
        ".js": FileType.JAVASCRIPT,
        ".jsx": FileType.JAVASCRIPT,
        ".ts": FileType.TYPESCRIPT,
        ".tsx": FileType.TYPESCRIPT,
        ".html": FileType.HTML,
        ".htm": FileType.HTML,
        ".css": FileType.CSS,
        ".json": FileType.JSON,
    }

    return type_map.get(ext, FileType.UNKNOWN)


def validate_python(content: str, file_path: str) -> ValidationResult:
    """Validate Python syntax using ast.parse."""
    errors = []
    warnings = []
    line_number = None

    try:
        ast.parse(content)
    except SyntaxError as e:
        line_number = e.lineno
        error_msg = f"Line {e.lineno}: {e.msg}"
        if e.text:
            error_msg += f"\n    {e.text.strip()}"
            if e.offset:
                error_msg += f"\n    {' ' * (e.offset - 1)}^"
        errors.append(error_msg)
    except Exception as e:
        errors.append(f"Parse error: {str(e)}")

    # Check for common issues
    if "import " in content:
        # Check for obviously wrong imports
        if re.search(r"import\s+\.\.", content):
            warnings.append("Suspicious relative import pattern")

    return ValidationResult(
        valid=len(errors) == 0,
        file_path=file_path,
        file_type=FileType.PYTHON,
        errors=errors,
        warnings=warnings,
        line_number=line_number
    )


def validate_json(content: str, file_path: str) -> ValidationResult:
    """Validate JSON syntax."""
    errors = []
    line_number = None

    try:
        json.loads(content)
    except json.JSONDecodeError as e:
        line_number = e.lineno
        errors.append(f"Line {e.lineno}, Column {e.colno}: {e.msg}")
    except Exception as e:
        errors.append(f"Parse error: {str(e)}")

    return ValidationResult(
        valid=len(errors) == 0,
        file_path=file_path,
        file_type=FileType.JSON,
        errors=errors,
        warnings=[],
        line_number=line_number
    )


def validate_html(content: str, file_path: str) -> ValidationResult:
    """Basic HTML validation - check for matching tags."""
    errors = []
    warnings = []

    # Check for basic structure
    if not re.search(r"<html", content, re.IGNORECASE):
        warnings.append("Missing <html> tag")

    if not re.search(r"<head", content, re.IGNORECASE):
        warnings.append("Missing <head> tag")

    if not re.search(r"<body", content, re.IGNORECASE):
        warnings.append("Missing <body> tag")

    # Check for unclosed tags (simple check)
    void_tags = {'br', 'hr', 'img', 'input', 'meta', 'link', 'area', 'base', 'col', 'embed', 'source', 'track', 'wbr'}

    # Find all opening tags
    opening = re.findall(r"<(\w+)[^>]*>", content, re.IGNORECASE)
    closing = re.findall(r"</(\w+)>", content, re.IGNORECASE)

    opening_counts = {}
    closing_counts = {}

    for tag in opening:
        tag_lower = tag.lower()
        if tag_lower not in void_tags:
            opening_counts[tag_lower] = opening_counts.get(tag_lower, 0) + 1

    for tag in closing:
        tag_lower = tag.lower()
        closing_counts[tag_lower] = closing_counts.get(tag_lower, 0) + 1

    for tag, count in opening_counts.items():
        close_count = closing_counts.get(tag, 0)
        if count > close_count:
            errors.append(f"Unclosed <{tag}> tag(s): {count - close_count} missing")
        elif count < close_count:
            errors.append(f"Extra </{tag}> closing tag(s): {close_count - count} extra")

    return ValidationResult(
        valid=len(errors) == 0,
        file_path=file_path,
        file_type=FileType.HTML,
        errors=errors,
        warnings=warnings
    )


def validate_css(content: str, file_path: str) -> ValidationResult:
    """Basic CSS validation - check for balanced braces."""
    errors = []
    warnings = []

    # Check for balanced braces
    open_braces = content.count('{')
    close_braces = content.count('}')

    if open_braces != close_braces:
        errors.append(f"Unbalanced braces: {open_braces} opening, {close_braces} closing")

    # Check for common syntax errors
    if re.search(r";\s*;", content):
        warnings.append("Double semicolons detected")

    if re.search(r"{\s*}", content):
        warnings.append("Empty rule blocks detected")

    return ValidationResult(
        valid=len(errors) == 0,
        file_path=file_path,
        file_type=FileType.CSS,
        errors=errors,
        warnings=warnings
    )


def validate_javascript(content: str, file_path: str) -> ValidationResult:
    """Basic JavaScript validation - check for common syntax issues."""
    errors = []
    warnings = []

    # Check for balanced braces/brackets
    if content.count('{') != content.count('}'):
        errors.append(f"Unbalanced curly braces: {content.count('{')} opening, {content.count('}')} closing")

    if content.count('[') != content.count(']'):
        errors.append(f"Unbalanced square brackets: {content.count('[')} opening, {content.count(']')} closing")

    if content.count('(') != content.count(')'):
        errors.append(f"Unbalanced parentheses: {content.count('(')} opening, {content.count(')')} closing")

    # Check for unterminated strings (simple check)
    lines = content.split('\n')
    for i, line in enumerate(lines, 1):
        # Skip comments
        line_stripped = line.strip()
        if line_stripped.startswith('//'):
            continue

        # Count quotes (very basic - doesn't handle escaped quotes perfectly)
        single_quotes = line.count("'") - line.count("\\'")
        double_quotes = line.count('"') - line.count('\\"')

        if single_quotes % 2 != 0:
            warnings.append(f"Line {i}: Possible unterminated single-quoted string")
        if double_quotes % 2 != 0:
            warnings.append(f"Line {i}: Possible unterminated double-quoted string")

    return ValidationResult(
        valid=len(errors) == 0,
        file_path=file_path,
        file_type=FileType.JAVASCRIPT,
        errors=errors,
        warnings=warnings
    )


def validate_file(file_path: str, content: str) -> ValidationResult:
    """Validate a file based on its type."""
    file_type = detect_file_type(file_path)

    validators = {
        FileType.PYTHON: validate_python,
        FileType.JSON: validate_json,
        FileType.HTML: validate_html,
        FileType.CSS: validate_css,
        FileType.JAVASCRIPT: validate_javascript,
        FileType.TYPESCRIPT: validate_javascript,  # Use JS validator for TS basics
    }

    validator = validators.get(file_type)
    if validator:
        return validator(content, file_path)

    # Unknown type - assume valid
    return ValidationResult(
        valid=True,
        file_path=file_path,
        file_type=FileType.UNKNOWN,
        errors=[],
        warnings=["Unknown file type - validation skipped"]
    )


def validate_files(files: dict) -> Tuple[bool, List[ValidationResult]]:
    """Validate multiple files.

    Args:
        files: Dict mapping file_path -> content

    Returns:
        Tuple of (all_valid, list of ValidationResults)
    """
    results = []
    all_valid = True

    for file_path, content in files.items():
        result = validate_file(file_path, content)
        results.append(result)
        if not result.valid:
            all_valid = False

    return all_valid, results


def format_validation_errors(results: List[ValidationResult]) -> str:
    """Format validation errors for LLM correction prompt."""
    error_parts = []

    for result in results:
        if not result.valid:
            error_parts.append(result.error_summary)

    if not error_parts:
        return ""

    return "VALIDATION ERRORS:\n" + "\n\n".join(error_parts)
