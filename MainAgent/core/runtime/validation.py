"""
Output validation layer for LLM responses.

Provides:
- Syntax validation for generated code
- Security scanning (secrets, credentials)
- Schema validation for structured outputs
- Idempotency checking
"""

from __future__ import annotations

import re
import ast
import json
import logging
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

logger = logging.getLogger(__name__)


class ValidationSeverity(Enum):
    """Severity levels for validation issues."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class ValidationIssue:
    """Represents a validation issue."""
    severity: ValidationSeverity
    message: str
    line: Optional[int] = None
    column: Optional[int] = None
    code: Optional[str] = None
    suggestion: Optional[str] = None


@dataclass
class ValidationResult:
    """Result of validation."""
    is_valid: bool
    issues: List[ValidationIssue] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    info: Dict[str, Any] = field(default_factory=dict)

    def add_issue(self, issue: ValidationIssue) -> None:
        self.issues.append(issue)
        if issue.severity in (ValidationSeverity.ERROR, ValidationSeverity.CRITICAL):
            self.is_valid = False

    def has_errors(self) -> bool:
        return any(
            i.severity in (ValidationSeverity.ERROR, ValidationSeverity.CRITICAL)
            for i in self.issues
        )


class SyntaxValidator:
    """Validates syntax of generated code."""

    def __init__(self):
        # File extension to validator mapping
        self._validators = {
            '.py': self._validate_python,
            '.json': self._validate_json,
            '.yaml': self._validate_yaml,
            '.yml': self._validate_yaml,
        }

    def validate(self, content: str, filename: str) -> ValidationResult:
        """Validate content syntax based on file extension."""
        result = ValidationResult(is_valid=True)

        ext = Path(filename).suffix.lower()
        validator = self._validators.get(ext)

        if validator:
            validator(content, result)
        else:
            result.info["skipped"] = f"No validator for extension: {ext}"

        return result

    def _validate_python(self, content: str, result: ValidationResult) -> None:
        """Validate Python syntax."""
        try:
            ast.parse(content)
            result.info["python_valid"] = True
        except SyntaxError as e:
            result.add_issue(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                message=f"Python syntax error: {e.msg}",
                line=e.lineno,
                column=e.offset,
                code="PYTHON_SYNTAX_ERROR"
            ))

    def _validate_json(self, content: str, result: ValidationResult) -> None:
        """Validate JSON syntax."""
        try:
            json.loads(content)
            result.info["json_valid"] = True
        except json.JSONDecodeError as e:
            result.add_issue(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                message=f"JSON syntax error: {e.msg}",
                line=e.lineno,
                column=e.colno,
                code="JSON_SYNTAX_ERROR"
            ))

    def _validate_yaml(self, content: str, result: ValidationResult) -> None:
        """Validate YAML syntax."""
        try:
            import yaml
            yaml.safe_load(content)
            result.info["yaml_valid"] = True
        except ImportError:
            result.info["yaml_skipped"] = "PyYAML not installed"
        except yaml.YAMLError as e:
            result.add_issue(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                message=f"YAML syntax error: {e}",
                code="YAML_SYNTAX_ERROR"
            ))


class SecurityScanner:
    """Scans for security issues in generated code."""

    # Common secret patterns
    SECRET_PATTERNS = [
        # API Keys
        (r'(?i)(api[_-]?key|apikey)\s*[=:]\s*["\']?([a-zA-Z0-9_\-]{20,})["\']?', "API key"),
        (r'(?i)(secret[_-]?key|secretkey)\s*[=:]\s*["\']?([a-zA-Z0-9_\-]{20,})["\']?', "Secret key"),

        # AWS
        (r'AKIA[0-9A-Z]{16}', "AWS Access Key ID"),
        (r'(?i)aws[_-]?secret[_-]?access[_-]?key\s*[=:]\s*["\']?([a-zA-Z0-9/+=]{40})["\']?', "AWS Secret Key"),

        # GitHub
        (r'ghp_[a-zA-Z0-9]{36}', "GitHub Personal Access Token"),
        (r'github_pat_[a-zA-Z0-9]{22}_[a-zA-Z0-9]{59}', "GitHub Fine-grained PAT"),

        # Google
        (r'AIza[0-9A-Za-z\-_]{35}', "Google API Key"),

        # Slack
        (r'xox[baprs]-[0-9]{10,13}-[0-9]{10,13}[a-zA-Z0-9-]*', "Slack Token"),

        # Generic
        (r'(?i)(password|passwd|pwd)\s*[=:]\s*["\']([^"\']{8,})["\']', "Hardcoded password"),
        (r'(?i)(bearer|token)\s+[a-zA-Z0-9_\-\.]{20,}', "Bearer token"),

        # Private keys
        (r'-----BEGIN (?:RSA |EC |DSA |OPENSSH )?PRIVATE KEY-----', "Private key"),
        (r'-----BEGIN PGP PRIVATE KEY BLOCK-----', "PGP Private key"),
    ]

    # Dangerous patterns
    DANGEROUS_PATTERNS = [
        # Command injection
        (r'subprocess\.(call|run|Popen)\s*\([^)]*shell\s*=\s*True', "Shell injection risk"),
        (r'os\.system\s*\(', "Command injection risk"),
        (r'eval\s*\(', "Code injection risk (eval)"),
        (r'exec\s*\(', "Code injection risk (exec)"),

        # SQL injection
        (r'execute\s*\(\s*["\'].*%s', "Potential SQL injection"),
        (r'execute\s*\(\s*f["\']', "SQL injection via f-string"),

        # Path traversal
        (r'open\s*\([^)]*\+[^)]*\)', "Path traversal risk"),

        # Pickle (deserialization)
        (r'pickle\.loads?\s*\(', "Unsafe deserialization (pickle)"),
    ]

    def scan(self, content: str, filename: str = "") -> ValidationResult:
        """Scan content for security issues."""
        result = ValidationResult(is_valid=True)

        # Check for secrets
        self._scan_secrets(content, result)

        # Check for dangerous patterns (only for code files)
        ext = Path(filename).suffix.lower() if filename else ""
        if ext in ('.py', '.js', '.ts', '.rb', '.php', '.java'):
            self._scan_dangerous_patterns(content, result)

        return result

    def _scan_secrets(self, content: str, result: ValidationResult) -> None:
        """Scan for hardcoded secrets."""
        lines = content.split('\n')

        for pattern, description in self.SECRET_PATTERNS:
            for i, line in enumerate(lines, 1):
                # Skip comments
                stripped = line.strip()
                if stripped.startswith('#') or stripped.startswith('//'):
                    continue

                matches = re.finditer(pattern, line)
                for match in matches:
                    # Check if it's likely a placeholder
                    matched_text = match.group(0)
                    if self._is_placeholder(matched_text):
                        continue

                    result.add_issue(ValidationIssue(
                        severity=ValidationSeverity.CRITICAL,
                        message=f"Potential {description} detected",
                        line=i,
                        code="SECRET_DETECTED",
                        suggestion="Use environment variables or a secrets manager"
                    ))

    def _scan_dangerous_patterns(self, content: str, result: ValidationResult) -> None:
        """Scan for dangerous code patterns."""
        lines = content.split('\n')

        for pattern, description in self.DANGEROUS_PATTERNS:
            for i, line in enumerate(lines, 1):
                if re.search(pattern, line):
                    result.add_issue(ValidationIssue(
                        severity=ValidationSeverity.WARNING,
                        message=f"{description}",
                        line=i,
                        code="DANGEROUS_PATTERN",
                        suggestion="Review this code for security implications"
                    ))

    def _is_placeholder(self, text: str) -> bool:
        """Check if text looks like a placeholder."""
        placeholder_patterns = [
            r'your[_-]?api[_-]?key',
            r'<.*>',
            r'\$\{.*\}',
            r'xxx+',
            r'\*{3,}',
            r'example',
            r'placeholder',
            r'changeme',
            r'insert[_-]?here',
        ]

        text_lower = text.lower()
        for pattern in placeholder_patterns:
            if re.search(pattern, text_lower):
                return True
        return False


class SchemaValidator:
    """Validates structured outputs against schemas."""

    def validate_json_schema(
        self,
        data: Dict[str, Any],
        schema: Dict[str, Any]
    ) -> ValidationResult:
        """Validate JSON data against a JSON schema."""
        result = ValidationResult(is_valid=True)

        try:
            import jsonschema
            jsonschema.validate(data, schema)
            result.info["schema_valid"] = True
        except ImportError:
            result.info["schema_skipped"] = "jsonschema not installed"
        except jsonschema.ValidationError as e:
            result.add_issue(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                message=f"Schema validation failed: {e.message}",
                code="SCHEMA_VALIDATION_ERROR"
            ))

        return result

    def validate_required_fields(
        self,
        data: Dict[str, Any],
        required_fields: List[str]
    ) -> ValidationResult:
        """Validate that required fields are present."""
        result = ValidationResult(is_valid=True)

        missing = [f for f in required_fields if f not in data]
        if missing:
            result.add_issue(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                message=f"Missing required fields: {', '.join(missing)}",
                code="MISSING_REQUIRED_FIELDS"
            ))

        return result


class LLMOutputValidator:
    """
    Comprehensive validator for LLM outputs.

    Combines syntax, security, and schema validation.
    """

    def __init__(self):
        self.syntax_validator = SyntaxValidator()
        self.security_scanner = SecurityScanner()
        self.schema_validator = SchemaValidator()

    def validate_code(
        self,
        content: str,
        filename: str,
        check_security: bool = True
    ) -> ValidationResult:
        """
        Validate generated code.

        Args:
            content: Code content to validate
            filename: Filename (for determining language)
            check_security: Whether to run security scan

        Returns:
            Combined validation result
        """
        result = ValidationResult(is_valid=True)

        # Syntax validation
        syntax_result = self.syntax_validator.validate(content, filename)
        result.issues.extend(syntax_result.issues)
        result.info.update(syntax_result.info)
        if syntax_result.has_errors():
            result.is_valid = False

        # Security scan
        if check_security:
            security_result = self.security_scanner.scan(content, filename)
            result.issues.extend(security_result.issues)
            if security_result.has_errors():
                result.is_valid = False

        return result

    def validate_structured_output(
        self,
        output: str,
        expected_format: str = "json",
        schema: Optional[Dict[str, Any]] = None,
        required_fields: Optional[List[str]] = None
    ) -> ValidationResult:
        """
        Validate structured LLM output.

        Args:
            output: Raw LLM output string
            expected_format: Expected format (json, yaml)
            schema: Optional JSON schema to validate against
            required_fields: Optional list of required fields

        Returns:
            Validation result
        """
        result = ValidationResult(is_valid=True)

        # Parse the output
        parsed_data = None
        if expected_format == "json":
            try:
                # Try to extract JSON from markdown code blocks
                json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', output)
                if json_match:
                    parsed_data = json.loads(json_match.group(1))
                else:
                    parsed_data = json.loads(output)
            except json.JSONDecodeError as e:
                result.add_issue(ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    message=f"Invalid JSON: {e.msg}",
                    code="INVALID_JSON"
                ))
                return result

        elif expected_format == "yaml":
            try:
                import yaml
                parsed_data = yaml.safe_load(output)
            except ImportError:
                result.info["yaml_skipped"] = "PyYAML not installed"
            except yaml.YAMLError as e:
                result.add_issue(ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    message=f"Invalid YAML: {e}",
                    code="INVALID_YAML"
                ))
                return result

        if parsed_data is None:
            return result

        # Validate against schema
        if schema:
            schema_result = self.schema_validator.validate_json_schema(parsed_data, schema)
            result.issues.extend(schema_result.issues)
            if schema_result.has_errors():
                result.is_valid = False

        # Validate required fields
        if required_fields and isinstance(parsed_data, dict):
            field_result = self.schema_validator.validate_required_fields(
                parsed_data, required_fields
            )
            result.issues.extend(field_result.issues)
            if field_result.has_errors():
                result.is_valid = False

        result.info["parsed_data"] = parsed_data
        return result

    def validate_edit_block(
        self,
        edit_block: str,
        existing_content: Optional[str] = None
    ) -> ValidationResult:
        """
        Validate an edit block format.

        Checks:
        - Valid edit format (===EDIT=== markers or OLD/NEW)
        - Code sections are present
        - If existing_content provided, checks if old code exists

        Args:
            edit_block: Edit block to validate
            existing_content: Optional content to check against

        Returns:
            Validation result
        """
        result = ValidationResult(is_valid=True)

        # Check for edit markers
        has_edit_markers = '===EDIT===' in edit_block
        has_old_new = bool(re.search(r'(?i)(OLD|FIND|REPLACE):', edit_block))

        if not has_edit_markers and not has_old_new:
            result.add_issue(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                message="Edit block doesn't contain standard markers",
                code="MISSING_EDIT_MARKERS"
            ))

        # Extract old/new sections
        if has_old_new:
            patterns = [
                r'(?i)OLD:\s*\n?([\s\S]*?)\nNEW:',
                r'(?i)FIND:\s*\n?([\s\S]*?)\nREPLACE:',
            ]

            old_code = None
            for pattern in patterns:
                match = re.search(pattern, edit_block)
                if match:
                    old_code = match.group(1).strip()
                    break

            if old_code and existing_content:
                if old_code not in existing_content:
                    # Try normalized comparison
                    normalized_old = ' '.join(old_code.split())
                    normalized_content = ' '.join(existing_content.split())

                    if normalized_old not in normalized_content:
                        result.add_issue(ValidationIssue(
                            severity=ValidationSeverity.WARNING,
                            message="Old code block not found in existing content",
                            code="OLD_CODE_NOT_FOUND",
                            suggestion="The code may have changed or the match is inexact"
                        ))

        return result


# Convenience function for quick validation
def validate_llm_output(
    content: str,
    filename: str = "",
    check_security: bool = True
) -> Tuple[bool, List[str]]:
    """
    Quick validation of LLM output.

    Args:
        content: Content to validate
        filename: Optional filename for context
        check_security: Whether to check for security issues

    Returns:
        Tuple of (is_valid, list_of_error_messages)
    """
    validator = LLMOutputValidator()
    result = validator.validate_code(content, filename, check_security)

    errors = [
        f"[{i.severity.value.upper()}] {i.message}"
        + (f" (line {i.line})" if i.line else "")
        for i in result.issues
    ]

    return result.is_valid, errors
