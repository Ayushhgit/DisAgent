"""
Code Quality Checker - Analyze code for quality issues and best practices.

Features:
- Syntax validation
- Complexity analysis
- Code smell detection
- Style checking
- Security pattern detection
"""

from __future__ import annotations

import re
import ast
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class IssueSeverity(Enum):
    """Severity levels for code issues."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class IssueCategory(Enum):
    """Categories of code issues."""
    SYNTAX = "syntax"
    COMPLEXITY = "complexity"
    STYLE = "style"
    SECURITY = "security"
    PERFORMANCE = "performance"
    MAINTAINABILITY = "maintainability"


@dataclass
class CodeIssue:
    """A single code quality issue."""
    message: str
    line: Optional[int] = None
    column: Optional[int] = None
    severity: IssueSeverity = IssueSeverity.WARNING
    category: IssueCategory = IssueCategory.STYLE
    rule: str = ""
    suggestion: str = ""


@dataclass
class QualityReport:
    """Code quality analysis report."""
    score: int  # 0-100
    issues: List[CodeIssue] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    summary: str = ""

    @property
    def has_errors(self) -> bool:
        return any(i.severity in (IssueSeverity.ERROR, IssueSeverity.CRITICAL) for i in self.issues)

    @property
    def issue_count(self) -> Dict[str, int]:
        counts = {s.value: 0 for s in IssueSeverity}
        for issue in self.issues:
            counts[issue.severity.value] += 1
        return counts


def check_quality(code: str, language: str = "python") -> Dict[str, Any]:
    """Check code quality (legacy interface).

    Args:
        code: Code to analyze
        language: Programming language

    Returns:
        Quality report as dictionary
    """
    report = analyze_code(code, language)
    return {
        "score": report.score,
        "issues": [
            {
                "message": i.message,
                "line": i.line,
                "severity": i.severity.value,
                "category": i.category.value,
            }
            for i in report.issues
        ],
        "metrics": report.metrics,
        "summary": report.summary,
    }


def analyze_code(code: str, language: str = "python") -> QualityReport:
    """Analyze code for quality issues.

    Args:
        code: Code to analyze
        language: Programming language

    Returns:
        QualityReport with findings
    """
    if not code or not code.strip():
        return QualityReport(
            score=0,
            issues=[CodeIssue(
                message="Empty code provided",
                severity=IssueSeverity.ERROR,
                category=IssueCategory.SYNTAX,
            )],
            summary="Cannot analyze empty code"
        )

    if language.lower() == "python":
        return _analyze_python(code)
    elif language.lower() in ("javascript", "js", "typescript", "ts"):
        return _analyze_javascript(code)
    else:
        return _analyze_generic(code)


def _analyze_python(code: str) -> QualityReport:
    """Analyze Python code."""
    issues = []
    metrics = {}

    # Count lines
    lines = code.split('\n')
    metrics['lines'] = len(lines)
    metrics['non_empty_lines'] = len([l for l in lines if l.strip()])

    # Check syntax
    syntax_valid = True
    try:
        tree = ast.parse(code)
        metrics['syntax_valid'] = True
    except SyntaxError as exc:
        syntax_valid = False
        metrics['syntax_valid'] = False
        issues.append(CodeIssue(
            message=f"Syntax error: {exc.msg}",
            line=exc.lineno,
            column=exc.offset,
            severity=IssueSeverity.ERROR,
            category=IssueCategory.SYNTAX,
        ))

    if syntax_valid:
        # Analyze AST
        issues.extend(_analyze_python_ast(tree, lines))
        metrics.update(_calculate_python_metrics(tree))

    # Style checks
    issues.extend(_check_python_style(lines))

    # Security checks
    issues.extend(_check_python_security(code, lines))

    # Calculate score
    score = _calculate_score(issues, metrics)

    # Generate summary
    summary = _generate_summary(issues, metrics, score)

    return QualityReport(
        score=score,
        issues=issues,
        metrics=metrics,
        summary=summary,
    )


def _analyze_python_ast(tree: ast.AST, lines: List[str]) -> List[CodeIssue]:
    """Analyze Python AST for issues."""
    issues = []

    class Visitor(ast.NodeVisitor):
        def __init__(self):
            self.function_count = 0
            self.class_count = 0
            self.current_function_depth = 0

        def visit_FunctionDef(self, node):
            self.function_count += 1

            # Check function complexity
            if len(node.body) > 50:
                issues.append(CodeIssue(
                    message=f"Function '{node.name}' is too long ({len(node.body)} statements)",
                    line=node.lineno,
                    severity=IssueSeverity.WARNING,
                    category=IssueCategory.COMPLEXITY,
                    suggestion="Consider breaking this function into smaller functions",
                ))

            # Check parameter count
            params = len(node.args.args)
            if params > 5:
                issues.append(CodeIssue(
                    message=f"Function '{node.name}' has too many parameters ({params})",
                    line=node.lineno,
                    severity=IssueSeverity.WARNING,
                    category=IssueCategory.COMPLEXITY,
                    suggestion="Consider using a configuration object or dataclass",
                ))

            # Check for missing docstring
            if not ast.get_docstring(node):
                issues.append(CodeIssue(
                    message=f"Function '{node.name}' is missing a docstring",
                    line=node.lineno,
                    severity=IssueSeverity.INFO,
                    category=IssueCategory.MAINTAINABILITY,
                ))

            self.generic_visit(node)

        def visit_ClassDef(self, node):
            self.class_count += 1

            # Check for missing docstring
            if not ast.get_docstring(node):
                issues.append(CodeIssue(
                    message=f"Class '{node.name}' is missing a docstring",
                    line=node.lineno,
                    severity=IssueSeverity.INFO,
                    category=IssueCategory.MAINTAINABILITY,
                ))

            # Check method count
            methods = [n for n in node.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
            if len(methods) > 20:
                issues.append(CodeIssue(
                    message=f"Class '{node.name}' has too many methods ({len(methods)})",
                    line=node.lineno,
                    severity=IssueSeverity.WARNING,
                    category=IssueCategory.COMPLEXITY,
                    suggestion="Consider splitting into multiple classes",
                ))

            self.generic_visit(node)

        def visit_If(self, node):
            # Track nesting depth
            self.current_function_depth += 1
            if self.current_function_depth > 4:
                issues.append(CodeIssue(
                    message="Deep nesting detected (>4 levels)",
                    line=node.lineno,
                    severity=IssueSeverity.WARNING,
                    category=IssueCategory.COMPLEXITY,
                    suggestion="Consider using early returns or extracting logic",
                ))
            self.generic_visit(node)
            self.current_function_depth -= 1

        def visit_For(self, node):
            self.current_function_depth += 1
            self.generic_visit(node)
            self.current_function_depth -= 1

        def visit_While(self, node):
            self.current_function_depth += 1
            self.generic_visit(node)
            self.current_function_depth -= 1

    visitor = Visitor()
    visitor.visit(tree)

    return issues


def _calculate_python_metrics(tree: ast.AST) -> Dict[str, Any]:
    """Calculate Python code metrics."""
    metrics = {
        'functions': 0,
        'classes': 0,
        'imports': 0,
    }

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            metrics['functions'] += 1
        elif isinstance(node, ast.ClassDef):
            metrics['classes'] += 1
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            metrics['imports'] += 1

    return metrics


def _check_python_style(lines: List[str]) -> List[CodeIssue]:
    """Check Python style issues."""
    issues = []

    for i, line in enumerate(lines, 1):
        # Line length
        if len(line) > 120:
            issues.append(CodeIssue(
                message=f"Line too long ({len(line)} > 120 characters)",
                line=i,
                severity=IssueSeverity.INFO,
                category=IssueCategory.STYLE,
            ))

        # Trailing whitespace
        if line.rstrip() != line:
            issues.append(CodeIssue(
                message="Trailing whitespace",
                line=i,
                severity=IssueSeverity.INFO,
                category=IssueCategory.STYLE,
            ))

        # Mixed tabs and spaces
        if '\t' in line and '    ' in line:
            issues.append(CodeIssue(
                message="Mixed tabs and spaces",
                line=i,
                severity=IssueSeverity.WARNING,
                category=IssueCategory.STYLE,
            ))

    return issues


def _check_python_security(code: str, lines: List[str]) -> List[CodeIssue]:
    """Check for security issues in Python code."""
    issues = []

    security_patterns = [
        (r'\beval\s*\(', "Use of eval() is dangerous", IssueSeverity.CRITICAL),
        (r'\bexec\s*\(', "Use of exec() is dangerous", IssueSeverity.CRITICAL),
        (r'\b__import__\s*\(', "Dynamic import can be dangerous", IssueSeverity.WARNING),
        (r'shell\s*=\s*True', "shell=True in subprocess can be dangerous", IssueSeverity.WARNING),
        (r'password\s*=\s*["\'][^"\']+["\']', "Hardcoded password detected", IssueSeverity.CRITICAL),
        (r'api_key\s*=\s*["\'][^"\']+["\']', "Hardcoded API key detected", IssueSeverity.CRITICAL),
        (r'\.format\s*\([^)]*\)', "String formatting may be vulnerable to injection", IssueSeverity.INFO),
    ]

    for i, line in enumerate(lines, 1):
        for pattern, message, severity in security_patterns:
            if re.search(pattern, line, re.IGNORECASE):
                issues.append(CodeIssue(
                    message=message,
                    line=i,
                    severity=severity,
                    category=IssueCategory.SECURITY,
                ))

    return issues


def _analyze_javascript(code: str) -> QualityReport:
    """Analyze JavaScript/TypeScript code."""
    issues = []
    metrics = {}

    lines = code.split('\n')
    metrics['lines'] = len(lines)
    metrics['non_empty_lines'] = len([l for l in lines if l.strip()])

    # Basic JS checks
    for i, line in enumerate(lines, 1):
        # var usage (prefer let/const)
        if re.search(r'\bvar\s+\w+', line):
            issues.append(CodeIssue(
                message="Use 'let' or 'const' instead of 'var'",
                line=i,
                severity=IssueSeverity.WARNING,
                category=IssueCategory.STYLE,
            ))

        # console.log in production
        if 'console.log' in line:
            issues.append(CodeIssue(
                message="console.log should be removed in production",
                line=i,
                severity=IssueSeverity.INFO,
                category=IssueCategory.STYLE,
            ))

        # == vs ===
        if re.search(r'[^=!]==[^=]', line):
            issues.append(CodeIssue(
                message="Use '===' instead of '==' for strict equality",
                line=i,
                severity=IssueSeverity.WARNING,
                category=IssueCategory.STYLE,
            ))

    score = _calculate_score(issues, metrics)
    summary = _generate_summary(issues, metrics, score)

    return QualityReport(
        score=score,
        issues=issues,
        metrics=metrics,
        summary=summary,
    )


def _analyze_generic(code: str) -> QualityReport:
    """Generic code analysis."""
    issues = []
    lines = code.split('\n')

    metrics = {
        'lines': len(lines),
        'non_empty_lines': len([l for l in lines if l.strip()]),
    }

    # Basic checks
    for i, line in enumerate(lines, 1):
        if len(line) > 120:
            issues.append(CodeIssue(
                message=f"Line too long ({len(line)} characters)",
                line=i,
                severity=IssueSeverity.INFO,
                category=IssueCategory.STYLE,
            ))

    score = _calculate_score(issues, metrics)
    summary = _generate_summary(issues, metrics, score)

    return QualityReport(
        score=score,
        issues=issues,
        metrics=metrics,
        summary=summary,
    )


def _calculate_score(issues: List[CodeIssue], metrics: Dict[str, Any]) -> int:
    """Calculate quality score from issues."""
    base_score = 100

    # Deduct points based on issue severity
    deductions = {
        IssueSeverity.CRITICAL: 20,
        IssueSeverity.ERROR: 10,
        IssueSeverity.WARNING: 3,
        IssueSeverity.INFO: 1,
    }

    for issue in issues:
        base_score -= deductions.get(issue.severity, 0)

    # Ensure score is in valid range
    return max(0, min(100, base_score))


def _generate_summary(issues: List[CodeIssue], metrics: Dict[str, Any], score: int) -> str:
    """Generate human-readable summary."""
    counts = {s.value: 0 for s in IssueSeverity}
    for issue in issues:
        counts[issue.severity.value] += 1

    parts = [f"Quality Score: {score}/100"]

    if counts['critical'] > 0:
        parts.append(f"Critical issues: {counts['critical']}")
    if counts['error'] > 0:
        parts.append(f"Errors: {counts['error']}")
    if counts['warning'] > 0:
        parts.append(f"Warnings: {counts['warning']}")
    if counts['info'] > 0:
        parts.append(f"Info: {counts['info']}")

    if 'lines' in metrics:
        parts.append(f"Lines: {metrics['lines']}")

    return " | ".join(parts)
