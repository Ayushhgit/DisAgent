"""
Code Review Flow - Automated code review workflow.

This workflow performs comprehensive code review including:
- Code quality analysis
- Security vulnerability scanning
- Best practices checking
- Documentation review
"""

from __future__ import annotations

import re
import ast
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class ReviewCategory(Enum):
    """Categories of review findings."""
    QUALITY = "quality"
    SECURITY = "security"
    PERFORMANCE = "performance"
    STYLE = "style"
    DOCUMENTATION = "documentation"
    TESTING = "testing"
    MAINTAINABILITY = "maintainability"


class Severity(Enum):
    """Severity of review findings."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


@dataclass
class ReviewFinding:
    """A single review finding."""
    category: ReviewCategory
    severity: Severity
    message: str
    line: Optional[int] = None
    suggestion: str = ""
    code_snippet: str = ""


@dataclass
class ReviewResult:
    """Result of a code review."""
    score: int  # 0-100
    findings: List[ReviewFinding] = field(default_factory=list)
    summary: str = ""
    recommendations: List[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        """Check if review passed (no critical/high findings)."""
        return not any(f.severity in (Severity.CRITICAL, Severity.HIGH) for f in self.findings)


def review(code: str, language: str = "python") -> Dict[str, Any]:
    """Review code (legacy interface).

    Args:
        code: Code to review
        language: Programming language

    Returns:
        Review result as dictionary
    """
    result = review_code(code, language)
    return {
        "score": result.score,
        "passed": result.passed,
        "issues": [
            {
                "category": f.category.value,
                "severity": f.severity.value,
                "message": f.message,
                "line": f.line,
                "suggestion": f.suggestion,
            }
            for f in result.findings
        ],
        "summary": result.summary,
        "recommendations": result.recommendations,
    }


def review_code(
    code: str,
    language: str = "python",
    include_security: bool = True,
    include_performance: bool = True,
) -> ReviewResult:
    """Perform comprehensive code review.

    Args:
        code: Code to review
        language: Programming language
        include_security: Include security checks
        include_performance: Include performance checks

    Returns:
        ReviewResult with findings
    """
    if not code or not code.strip():
        return ReviewResult(
            score=0,
            findings=[ReviewFinding(
                category=ReviewCategory.QUALITY,
                severity=Severity.CRITICAL,
                message="Empty code provided",
            )],
            summary="Cannot review empty code",
        )

    findings = []

    if language.lower() == "python":
        # Syntax check
        syntax_findings = _check_python_syntax(code)
        findings.extend(syntax_findings)

        if not any(f.severity == Severity.CRITICAL for f in syntax_findings):
            # Quality checks
            findings.extend(_review_python_quality(code))

            # Security checks
            if include_security:
                findings.extend(_review_python_security(code))

            # Performance checks
            if include_performance:
                findings.extend(_review_python_performance(code))

            # Documentation check
            findings.extend(_review_python_documentation(code))

            # Style checks
            findings.extend(_review_python_style(code))

    else:
        findings.extend(_review_generic(code))

    # Calculate score
    score = _calculate_review_score(findings)

    # Generate summary and recommendations
    summary = _generate_summary(findings, score)
    recommendations = _generate_recommendations(findings)

    return ReviewResult(
        score=score,
        findings=findings,
        summary=summary,
        recommendations=recommendations,
    )


def _check_python_syntax(code: str) -> List[ReviewFinding]:
    """Check Python syntax."""
    findings = []

    try:
        ast.parse(code)
    except SyntaxError as exc:
        findings.append(ReviewFinding(
            category=ReviewCategory.QUALITY,
            severity=Severity.CRITICAL,
            message=f"Syntax error: {exc.msg}",
            line=exc.lineno,
            code_snippet=exc.text or "",
        ))

    return findings


def _review_python_quality(code: str) -> List[ReviewFinding]:
    """Review Python code quality."""
    findings = []
    tree = ast.parse(code)
    lines = code.split('\n')

    class QualityChecker(ast.NodeVisitor):
        def visit_FunctionDef(self, node):
            # Function length
            if len(node.body) > 50:
                findings.append(ReviewFinding(
                    category=ReviewCategory.QUALITY,
                    severity=Severity.MEDIUM,
                    message=f"Function '{node.name}' is too long ({len(node.body)} statements)",
                    line=node.lineno,
                    suggestion="Break into smaller functions",
                ))

            # Too many parameters
            if len(node.args.args) > 5:
                findings.append(ReviewFinding(
                    category=ReviewCategory.QUALITY,
                    severity=Severity.MEDIUM,
                    message=f"Function '{node.name}' has too many parameters ({len(node.args.args)})",
                    line=node.lineno,
                    suggestion="Consider using a dataclass or configuration object",
                ))

            # Check for return consistency
            returns = [n for n in ast.walk(node) if isinstance(n, ast.Return)]
            has_value = any(r.value for r in returns)
            has_none = any(not r.value for r in returns)
            if has_value and has_none:
                findings.append(ReviewFinding(
                    category=ReviewCategory.QUALITY,
                    severity=Severity.LOW,
                    message=f"Function '{node.name}' has inconsistent return statements",
                    line=node.lineno,
                    suggestion="Ensure all return paths return a value or None explicitly",
                ))

            self.generic_visit(node)

        def visit_ClassDef(self, node):
            # Class with too many methods
            methods = [n for n in node.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
            if len(methods) > 20:
                findings.append(ReviewFinding(
                    category=ReviewCategory.MAINTAINABILITY,
                    severity=Severity.MEDIUM,
                    message=f"Class '{node.name}' has too many methods ({len(methods)})",
                    line=node.lineno,
                    suggestion="Consider splitting into smaller classes",
                ))

            self.generic_visit(node)

        def visit_Try(self, node):
            # Bare except clause
            for handler in node.handlers:
                if handler.type is None:
                    findings.append(ReviewFinding(
                        category=ReviewCategory.QUALITY,
                        severity=Severity.HIGH,
                        message="Bare 'except:' clause catches all exceptions including KeyboardInterrupt",
                        line=handler.lineno,
                        suggestion="Specify exception type, e.g., 'except Exception:'",
                    ))

            self.generic_visit(node)

    checker = QualityChecker()
    checker.visit(tree)

    return findings


def _review_python_security(code: str) -> List[ReviewFinding]:
    """Review Python code for security issues."""
    findings = []
    lines = code.split('\n')

    security_patterns = [
        (r'\beval\s*\(', Severity.CRITICAL, "Use of eval() - potential code injection"),
        (r'\bexec\s*\(', Severity.CRITICAL, "Use of exec() - potential code injection"),
        (r'\b__import__\s*\(', Severity.HIGH, "Dynamic import - review for injection risk"),
        (r'shell\s*=\s*True', Severity.HIGH, "subprocess with shell=True - potential command injection"),
        (r'pickle\.load', Severity.HIGH, "pickle.load - can execute arbitrary code"),
        (r'yaml\.load\s*\([^)]*\)', Severity.MEDIUM, "yaml.load without Loader - use safe_load"),
        (r'password\s*=\s*["\'][^"\']{3,}["\']', Severity.CRITICAL, "Hardcoded password detected"),
        (r'(api_key|secret|token)\s*=\s*["\'][^"\']{3,}["\']', Severity.CRITICAL, "Hardcoded secret detected"),
        (r'md5\s*\(', Severity.MEDIUM, "MD5 is cryptographically weak - use SHA-256 or better"),
        (r'random\.(random|randint|choice)', Severity.LOW, "Use secrets module for cryptographic operations"),
    ]

    for i, line in enumerate(lines, 1):
        for pattern, severity, message in security_patterns:
            if re.search(pattern, line, re.IGNORECASE):
                findings.append(ReviewFinding(
                    category=ReviewCategory.SECURITY,
                    severity=severity,
                    message=message,
                    line=i,
                    code_snippet=line.strip(),
                ))

    return findings


def _review_python_performance(code: str) -> List[ReviewFinding]:
    """Review Python code for performance issues."""
    findings = []
    lines = code.split('\n')

    try:
        tree = ast.parse(code)
    except SyntaxError:
        return findings

    class PerformanceChecker(ast.NodeVisitor):
        def visit_For(self, node):
            # Nested loops
            for child in ast.walk(node):
                if isinstance(child, ast.For) and child is not node:
                    findings.append(ReviewFinding(
                        category=ReviewCategory.PERFORMANCE,
                        severity=Severity.INFO,
                        message="Nested loop detected - consider optimization for large datasets",
                        line=node.lineno,
                    ))
                    break

            self.generic_visit(node)

        def visit_ListComp(self, node):
            # Very complex list comprehension
            if len(list(ast.walk(node))) > 20:
                findings.append(ReviewFinding(
                    category=ReviewCategory.PERFORMANCE,
                    severity=Severity.LOW,
                    message="Complex list comprehension - consider using a generator or regular loop",
                    line=node.lineno,
                ))

            self.generic_visit(node)

    checker = PerformanceChecker()
    checker.visit(tree)

    # Pattern-based checks
    for i, line in enumerate(lines, 1):
        # String concatenation in loop
        if '+=' in line and 'str' in line.lower():
            findings.append(ReviewFinding(
                category=ReviewCategory.PERFORMANCE,
                severity=Severity.LOW,
                message="String concatenation may be inefficient - consider using join()",
                line=i,
            ))

        # Reading entire file
        if '.read()' in line and 'with' not in lines[max(0, i-3):i]:
            findings.append(ReviewFinding(
                category=ReviewCategory.PERFORMANCE,
                severity=Severity.INFO,
                message="Reading entire file into memory - consider streaming for large files",
                line=i,
            ))

    return findings


def _review_python_documentation(code: str) -> List[ReviewFinding]:
    """Review Python documentation."""
    findings = []

    try:
        tree = ast.parse(code)
    except SyntaxError:
        return findings

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            if not ast.get_docstring(node):
                # Only flag public functions
                if not node.name.startswith('_'):
                    findings.append(ReviewFinding(
                        category=ReviewCategory.DOCUMENTATION,
                        severity=Severity.LOW,
                        message=f"Public function '{node.name}' is missing a docstring",
                        line=node.lineno,
                        suggestion="Add a docstring describing the function's purpose, parameters, and return value",
                    ))

        elif isinstance(node, ast.ClassDef):
            if not ast.get_docstring(node):
                findings.append(ReviewFinding(
                    category=ReviewCategory.DOCUMENTATION,
                    severity=Severity.LOW,
                    message=f"Class '{node.name}' is missing a docstring",
                    line=node.lineno,
                    suggestion="Add a docstring describing the class's purpose and usage",
                ))

    return findings


def _review_python_style(code: str) -> List[ReviewFinding]:
    """Review Python style."""
    findings = []
    lines = code.split('\n')

    for i, line in enumerate(lines, 1):
        # Line length
        if len(line) > 120:
            findings.append(ReviewFinding(
                category=ReviewCategory.STYLE,
                severity=Severity.INFO,
                message=f"Line exceeds 120 characters ({len(line)})",
                line=i,
            ))

        # Trailing whitespace
        if line.rstrip() != line:
            findings.append(ReviewFinding(
                category=ReviewCategory.STYLE,
                severity=Severity.INFO,
                message="Trailing whitespace",
                line=i,
            ))

    return findings


def _review_generic(code: str) -> List[ReviewFinding]:
    """Generic code review."""
    findings = []
    lines = code.split('\n')

    for i, line in enumerate(lines, 1):
        if len(line) > 120:
            findings.append(ReviewFinding(
                category=ReviewCategory.STYLE,
                severity=Severity.INFO,
                message=f"Line too long ({len(line)} characters)",
                line=i,
            ))

    return findings


def _calculate_review_score(findings: List[ReviewFinding]) -> int:
    """Calculate review score from findings."""
    base_score = 100

    deductions = {
        Severity.CRITICAL: 25,
        Severity.HIGH: 15,
        Severity.MEDIUM: 5,
        Severity.LOW: 2,
        Severity.INFO: 0,
    }

    for finding in findings:
        base_score -= deductions.get(finding.severity, 0)

    return max(0, min(100, base_score))


def _generate_summary(findings: List[ReviewFinding], score: int) -> str:
    """Generate review summary."""
    counts = {}
    for f in findings:
        key = f.severity.value
        counts[key] = counts.get(key, 0) + 1

    parts = [f"Score: {score}/100"]

    for severity in Severity:
        count = counts.get(severity.value, 0)
        if count > 0:
            parts.append(f"{severity.value.title()}: {count}")

    return " | ".join(parts)


def _generate_recommendations(findings: List[ReviewFinding]) -> List[str]:
    """Generate recommendations based on findings."""
    recommendations = []

    categories_found = {f.category for f in findings}

    if ReviewCategory.SECURITY in categories_found:
        recommendations.append("Address security vulnerabilities before deployment")

    if ReviewCategory.QUALITY in categories_found:
        recommendations.append("Improve code quality by refactoring complex functions")

    if ReviewCategory.DOCUMENTATION in categories_found:
        recommendations.append("Add documentation to improve maintainability")

    if ReviewCategory.PERFORMANCE in categories_found:
        recommendations.append("Consider performance optimizations for production use")

    return recommendations
