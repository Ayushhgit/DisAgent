"""
Critic Agent - Reviews agent outputs before acceptance.

Provides structured evaluation of code changes:
- Correctness: Does it solve the task?
- Quality: Is it well-structured and maintainable?
- Safety: Are there security or stability risks?
- Completeness: Are edge cases handled?

This addresses a key gap: existing systems generate code blindly
without structured review before integration.
"""

from __future__ import annotations

import logging
import re
import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional
import threading

logger = logging.getLogger(__name__)


class Severity(Enum):
    """Severity levels for critique issues."""
    CRITICAL = "critical"  # Must fix before acceptance
    MAJOR = "major"        # Should fix, significant impact
    MINOR = "minor"        # Nice to fix, low impact
    SUGGESTION = "suggestion"  # Optional improvement


class ApprovalStatus(Enum):
    """Approval status after critique."""
    APPROVED = "approved"
    CONDITIONAL = "conditional"  # Approved with minor fixes needed
    REJECTED = "rejected"
    NEEDS_REVISION = "needs_revision"


@dataclass
class Issue:
    """A single issue found during critique."""
    category: str  # correctness, quality, safety, completeness
    severity: Severity
    description: str
    location: str = ""  # File:line or function name
    suggestion: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "category": self.category,
            "severity": self.severity.value,
            "description": self.description,
            "location": self.location,
            "suggestion": self.suggestion
        }


@dataclass
class CritiqueResult:
    """Result of a critique evaluation."""
    status: ApprovalStatus
    issues: List[Issue] = field(default_factory=list)
    summary: str = ""
    score: float = 0.0  # 0.0 to 1.0
    recommendations: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def has_critical_issues(self) -> bool:
        return any(i.severity == Severity.CRITICAL for i in self.issues)

    @property
    def has_major_issues(self) -> bool:
        return any(i.severity == Severity.MAJOR for i in self.issues)

    @property
    def issue_count_by_severity(self) -> Dict[str, int]:
        counts = {s.value: 0 for s in Severity}
        for issue in self.issues:
            counts[issue.severity.value] += 1
        return counts

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status.value,
            "issues": [i.to_dict() for i in self.issues],
            "summary": self.summary,
            "score": self.score,
            "recommendations": self.recommendations,
            "issue_counts": self.issue_count_by_severity
        }

    def get_feedback_for_agent(self) -> str:
        """Generate feedback string for the agent to use in revision."""
        if self.status == ApprovalStatus.APPROVED:
            return "Output approved. No changes needed."

        lines = [f"Critique Status: {self.status.value}", ""]

        if self.issues:
            lines.append("Issues Found:")
            for i, issue in enumerate(self.issues, 1):
                lines.append(f"\n{i}. [{issue.severity.value.upper()}] {issue.category}")
                lines.append(f"   {issue.description}")
                if issue.location:
                    lines.append(f"   Location: {issue.location}")
                if issue.suggestion:
                    lines.append(f"   Suggestion: {issue.suggestion}")

        if self.recommendations:
            lines.append("\nRecommendations:")
            for rec in self.recommendations:
                lines.append(f"- {rec}")

        return "\n".join(lines)


class CriticAgent:
    """Reviews agent outputs before acceptance.

    Uses LLM to evaluate code changes against multiple criteria
    and provides structured feedback for revision.
    """

    def __init__(
        self,
        llm_call: Callable[[str, Optional[int]], str],
        strictness: str = "normal"  # "lenient", "normal", "strict"
    ):
        """Initialize critic agent.

        Args:
            llm_call: Function to call LLM (prompt, max_tokens) -> response
            strictness: How strict the review should be
        """
        self.llm_call = llm_call
        self.strictness = strictness
        self._lock = threading.Lock()
        self._review_history: List[CritiqueResult] = []

        # Thresholds based on strictness
        self._thresholds = {
            "lenient": {"min_score": 0.5, "allow_major": True},
            "normal": {"min_score": 0.7, "allow_major": False},
            "strict": {"min_score": 0.85, "allow_major": False}
        }

    def critique(
        self,
        agent_output: str,
        task_description: str,
        context: Optional[Dict[str, Any]] = None,
        file_path: Optional[str] = None
    ) -> CritiqueResult:
        """Evaluate agent output quality.

        Args:
            agent_output: The code/output to review
            task_description: What the agent was supposed to do
            context: Additional context (existing code, requirements, etc.)
            file_path: Path to the file being modified

        Returns:
            CritiqueResult with structured evaluation
        """
        prompt = self._build_critique_prompt(
            agent_output,
            task_description,
            context,
            file_path
        )

        try:
            response = self.llm_call(prompt, 2000)
            result = self._parse_critique_response(response)
        except Exception as e:
            logger.exception(f"Critique failed: {e}")
            # Return permissive result on failure
            result = CritiqueResult(
                status=ApprovalStatus.CONDITIONAL,
                summary=f"Critique failed: {e}. Proceeding with caution.",
                score=0.5
            )

        with self._lock:
            self._review_history.append(result)

        return result

    def _build_critique_prompt(
        self,
        agent_output: str,
        task_description: str,
        context: Optional[Dict[str, Any]],
        file_path: Optional[str]
    ) -> str:
        """Build the LLM prompt for critique."""
        context_str = ""
        if context:
            if context.get("existing_code"):
                context_str += f"\nEXISTING CODE:\n```\n{context['existing_code'][:2000]}\n```\n"
            if context.get("requirements"):
                context_str += f"\nREQUIREMENTS: {context['requirements']}\n"
            if context.get("constraints"):
                context_str += f"\nCONSTRAINTS: {context['constraints']}\n"

        strictness_instruction = {
            "lenient": "Be lenient - only flag clear errors and security issues.",
            "normal": "Be balanced - flag errors, quality issues, and improvements.",
            "strict": "Be strict - flag all issues including style and best practices."
        }.get(self.strictness, "")

        return f"""You are a code reviewer evaluating an agent's output.

TASK DESCRIPTION:
{task_description}

{f"FILE: {file_path}" if file_path else ""}
{context_str}

AGENT OUTPUT:
```
{agent_output}
```

REVIEW INSTRUCTIONS:
{strictness_instruction}

Evaluate the output on these criteria:
1. CORRECTNESS: Does it solve the task? Are there bugs or logic errors?
2. QUALITY: Is it well-structured, readable, and maintainable?
3. SAFETY: Are there security vulnerabilities or stability risks?
4. COMPLETENESS: Are edge cases handled? Is error handling adequate?

Return a JSON response:
{{
    "status": "approved|conditional|rejected|needs_revision",
    "score": 0.0-1.0,
    "summary": "Brief overall assessment",
    "issues": [
        {{
            "category": "correctness|quality|safety|completeness",
            "severity": "critical|major|minor|suggestion",
            "description": "What's wrong",
            "location": "Where in the code (optional)",
            "suggestion": "How to fix it"
        }}
    ],
    "recommendations": ["List of improvement suggestions"]
}}

Be specific and actionable. Return ONLY valid JSON."""

    def _parse_critique_response(self, response: str) -> CritiqueResult:
        """Parse LLM response into CritiqueResult."""
        # Extract JSON from response
        response = response.strip()
        json_match = re.search(r'\{[\s\S]*\}', response)
        if json_match:
            response = json_match.group()

        try:
            data = json.loads(response)
        except json.JSONDecodeError:
            logger.warning("Failed to parse critique JSON, using defaults")
            return CritiqueResult(
                status=ApprovalStatus.CONDITIONAL,
                summary="Could not parse critique response",
                score=0.6
            )

        # Parse status
        status_map = {
            "approved": ApprovalStatus.APPROVED,
            "conditional": ApprovalStatus.CONDITIONAL,
            "rejected": ApprovalStatus.REJECTED,
            "needs_revision": ApprovalStatus.NEEDS_REVISION
        }
        status = status_map.get(
            data.get("status", "conditional").lower(),
            ApprovalStatus.CONDITIONAL
        )

        # Parse issues
        issues = []
        for issue_data in data.get("issues", []):
            severity_map = {
                "critical": Severity.CRITICAL,
                "major": Severity.MAJOR,
                "minor": Severity.MINOR,
                "suggestion": Severity.SUGGESTION
            }
            issues.append(Issue(
                category=issue_data.get("category", "quality"),
                severity=severity_map.get(
                    issue_data.get("severity", "minor").lower(),
                    Severity.MINOR
                ),
                description=issue_data.get("description", ""),
                location=issue_data.get("location", ""),
                suggestion=issue_data.get("suggestion", "")
            ))

        # Apply strictness thresholds
        score = float(data.get("score", 0.7))
        thresholds = self._thresholds[self.strictness]

        if status == ApprovalStatus.APPROVED:
            # Downgrade if score is below threshold
            if score < thresholds["min_score"]:
                status = ApprovalStatus.CONDITIONAL
            # Downgrade if major issues and not allowed
            if not thresholds["allow_major"] and any(
                i.severity == Severity.MAJOR for i in issues
            ):
                status = ApprovalStatus.NEEDS_REVISION

        # Always reject if critical issues
        if any(i.severity == Severity.CRITICAL for i in issues):
            status = ApprovalStatus.REJECTED

        return CritiqueResult(
            status=status,
            issues=issues,
            summary=data.get("summary", ""),
            score=score,
            recommendations=data.get("recommendations", [])
        )

    def quick_check(
        self,
        code: str,
        language: str = "python"
    ) -> CritiqueResult:
        """Quick syntax and safety check without full LLM critique.

        Args:
            code: Code to check
            language: Programming language

        Returns:
            CritiqueResult with basic checks
        """
        issues = []

        if language == "python":
            issues.extend(self._check_python_issues(code))

        # Check for common security issues
        issues.extend(self._check_security_issues(code))

        # Determine status based on issues
        if any(i.severity == Severity.CRITICAL for i in issues):
            status = ApprovalStatus.REJECTED
            score = 0.3
        elif any(i.severity == Severity.MAJOR for i in issues):
            status = ApprovalStatus.NEEDS_REVISION
            score = 0.5
        elif issues:
            status = ApprovalStatus.CONDITIONAL
            score = 0.7
        else:
            status = ApprovalStatus.APPROVED
            score = 0.9

        return CritiqueResult(
            status=status,
            issues=issues,
            summary=f"Quick check found {len(issues)} issues",
            score=score
        )

    def _check_python_issues(self, code: str) -> List[Issue]:
        """Check Python code for common issues."""
        issues = []

        # Check syntax
        try:
            compile(code, "<string>", "exec")
        except SyntaxError as e:
            issues.append(Issue(
                category="correctness",
                severity=Severity.CRITICAL,
                description=f"Syntax error: {e.msg}",
                location=f"Line {e.lineno}" if e.lineno else "",
                suggestion="Fix the syntax error before proceeding"
            ))

        # Check for undefined names (basic)
        import_pattern = r'(?:from\s+\S+\s+)?import\s+(\w+)'
        imports = set(re.findall(import_pattern, code))

        # Check for print debugging left in
        if re.search(r'\bprint\s*\([^)]*debug', code, re.IGNORECASE):
            issues.append(Issue(
                category="quality",
                severity=Severity.MINOR,
                description="Debug print statement found",
                suggestion="Remove debug print statements before committing"
            ))

        # Check for TODO/FIXME
        todos = re.findall(r'#\s*(TODO|FIXME|XXX|HACK).*', code, re.IGNORECASE)
        for todo in todos:
            issues.append(Issue(
                category="completeness",
                severity=Severity.SUGGESTION,
                description=f"Unresolved {todo[:4].upper()} comment",
                suggestion="Address the TODO or create a tracked issue"
            ))

        return issues

    def _check_security_issues(self, code: str) -> List[Issue]:
        """Check for common security issues."""
        issues = []

        # Dangerous patterns
        dangerous_patterns = [
            (r'\beval\s*\(', "Use of eval() is dangerous"),
            (r'\bexec\s*\(', "Use of exec() is dangerous"),
            (r'subprocess.*shell\s*=\s*True', "Shell injection risk with shell=True"),
            (r'__import__\s*\(', "Dynamic import can be dangerous"),
            (r'pickle\.loads?\s*\(', "Pickle deserialization is unsafe"),
            (r'yaml\.load\s*\([^)]*\)', "Use yaml.safe_load instead of yaml.load"),
        ]

        for pattern, message in dangerous_patterns:
            if re.search(pattern, code):
                issues.append(Issue(
                    category="safety",
                    severity=Severity.MAJOR,
                    description=message,
                    suggestion="Use a safer alternative or add input validation"
                ))

        # Check for hardcoded secrets
        secret_patterns = [
            (r'(?:password|passwd|pwd)\s*=\s*["\'][^"\']+["\']', "Hardcoded password"),
            (r'(?:api_key|apikey|secret_key)\s*=\s*["\'][^"\']+["\']', "Hardcoded API key"),
            (r'(?:token)\s*=\s*["\'][a-zA-Z0-9]{20,}["\']', "Possible hardcoded token"),
        ]

        for pattern, message in secret_patterns:
            if re.search(pattern, code, re.IGNORECASE):
                issues.append(Issue(
                    category="safety",
                    severity=Severity.CRITICAL,
                    description=message,
                    suggestion="Use environment variables or secure secret management"
                ))

        return issues

    def get_review_history(self, limit: int = 10) -> List[CritiqueResult]:
        """Get recent review history."""
        with self._lock:
            return self._review_history[-limit:]

    def get_stats(self) -> Dict[str, Any]:
        """Get critic statistics."""
        with self._lock:
            if not self._review_history:
                return {"total_reviews": 0}

            return {
                "total_reviews": len(self._review_history),
                "approved": sum(1 for r in self._review_history
                               if r.status == ApprovalStatus.APPROVED),
                "rejected": sum(1 for r in self._review_history
                               if r.status == ApprovalStatus.REJECTED),
                "average_score": sum(r.score for r in self._review_history) / len(self._review_history),
                "total_issues": sum(len(r.issues) for r in self._review_history)
            }


def create_critic(
    llm_call: Callable[[str, Optional[int]], str],
    strictness: str = "normal"
) -> CriticAgent:
    """Factory function to create a critic agent.

    Args:
        llm_call: LLM function (prompt, max_tokens) -> response
        strictness: Review strictness level

    Returns:
        Configured CriticAgent
    """
    return CriticAgent(llm_call, strictness)
