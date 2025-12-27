"""
Bug Fix Flow - Automated bug detection and fixing workflow.

This workflow analyzes code for bugs, identifies the root cause,
and attempts to generate fixes using various strategies.
"""

from __future__ import annotations

import re
import ast
import logging
import traceback
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class BugType(Enum):
    """Types of bugs."""
    SYNTAX_ERROR = "syntax_error"
    RUNTIME_ERROR = "runtime_error"
    LOGIC_ERROR = "logic_error"
    TYPE_ERROR = "type_error"
    IMPORT_ERROR = "import_error"
    NAME_ERROR = "name_error"
    INDEX_ERROR = "index_error"
    KEY_ERROR = "key_error"
    ATTRIBUTE_ERROR = "attribute_error"
    VALUE_ERROR = "value_error"
    UNKNOWN = "unknown"


@dataclass
class BugAnalysis:
    """Analysis of a bug."""
    bug_type: BugType
    description: str
    location: Optional[Tuple[int, int]] = None  # (line, column)
    context: str = ""
    suggestion: str = ""


@dataclass
class FixResult:
    """Result of a bug fix attempt."""
    success: bool
    original_code: str
    fixed_code: str
    bugs_found: List[BugAnalysis] = field(default_factory=list)
    fixes_applied: List[str] = field(default_factory=list)
    error: Optional[str] = None


def fix_bug(code: str, error: str = "") -> str:
    """Fix bug in code (legacy interface).

    Args:
        code: Code with bug
        error: Error message if available

    Returns:
        Fixed code or original if fix failed
    """
    result = analyze_and_fix(code, error)
    return result.fixed_code if result.success else code


def analyze_and_fix(
    code: str,
    error_message: str = "",
    max_attempts: int = 3,
) -> FixResult:
    """Analyze code for bugs and attempt to fix them.

    Args:
        code: Code to analyze and fix
        error_message: Optional error message for context
        max_attempts: Maximum fix attempts

    Returns:
        FixResult with analysis and fixed code
    """
    if not code or not code.strip():
        return FixResult(
            success=False,
            original_code=code,
            fixed_code=code,
            error="Empty code provided",
        )

    bugs_found = []
    fixes_applied = []
    current_code = code

    # Analyze for bugs
    bugs = _analyze_bugs(current_code, error_message)
    bugs_found.extend(bugs)

    # Attempt fixes
    for attempt in range(max_attempts):
        if not bugs:
            break

        for bug in bugs:
            fixed, fix_description = _attempt_fix(current_code, bug)
            if fixed != current_code:
                current_code = fixed
                fixes_applied.append(fix_description)

        # Re-analyze to check if fixed
        remaining_bugs = _analyze_bugs(current_code, "")
        if not remaining_bugs:
            break
        bugs = remaining_bugs

    # Final validation
    success = _validate_code(current_code)

    return FixResult(
        success=success and current_code != code,
        original_code=code,
        fixed_code=current_code,
        bugs_found=bugs_found,
        fixes_applied=fixes_applied,
    )


def _analyze_bugs(code: str, error_message: str) -> List[BugAnalysis]:
    """Analyze code for bugs."""
    bugs = []

    # Check syntax
    try:
        ast.parse(code)
    except SyntaxError as exc:
        bugs.append(BugAnalysis(
            bug_type=BugType.SYNTAX_ERROR,
            description=str(exc.msg),
            location=(exc.lineno, exc.offset) if exc.lineno else None,
            context=exc.text or "",
            suggestion=_get_syntax_fix_suggestion(exc),
        ))
        return bugs  # Can't analyze further with syntax errors

    # Analyze error message if provided
    if error_message:
        bug = _analyze_error_message(error_message)
        if bug:
            bugs.append(bug)

    # Static analysis
    bugs.extend(_static_analysis(code))

    return bugs


def _analyze_error_message(error_message: str) -> Optional[BugAnalysis]:
    """Analyze an error message to determine bug type."""
    error_lower = error_message.lower()

    # Common error patterns
    patterns = [
        (r"nameerror.*name '(\w+)' is not defined", BugType.NAME_ERROR, "Undefined variable"),
        (r"typeerror.*", BugType.TYPE_ERROR, "Type mismatch"),
        (r"attributeerror.*'(\w+)' object has no attribute '(\w+)'", BugType.ATTRIBUTE_ERROR, "Missing attribute"),
        (r"indexerror.*list index out of range", BugType.INDEX_ERROR, "Index out of range"),
        (r"keyerror.*'(\w+)'", BugType.KEY_ERROR, "Missing dictionary key"),
        (r"valueerror.*", BugType.VALUE_ERROR, "Invalid value"),
        (r"importerror.*no module named '(\w+)'", BugType.IMPORT_ERROR, "Missing module"),
        (r"zerodivisionerror", BugType.RUNTIME_ERROR, "Division by zero"),
    ]

    for pattern, bug_type, description in patterns:
        if re.search(pattern, error_lower, re.IGNORECASE):
            # Try to extract line number
            line_match = re.search(r'line (\d+)', error_message, re.IGNORECASE)
            line = int(line_match.group(1)) if line_match else None

            return BugAnalysis(
                bug_type=bug_type,
                description=description,
                location=(line, 0) if line else None,
                context=error_message,
            )

    return None


def _static_analysis(code: str) -> List[BugAnalysis]:
    """Perform static analysis on code."""
    bugs = []
    lines = code.split('\n')

    try:
        tree = ast.parse(code)
    except SyntaxError:
        return bugs

    # Check for common issues
    for node in ast.walk(tree):
        # Unused variables (basic check)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            # Check if variable is used later
            pass  # Complex to implement properly

        # Division by zero potential
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
            if isinstance(node.right, ast.Constant) and node.right.value == 0:
                bugs.append(BugAnalysis(
                    bug_type=BugType.RUNTIME_ERROR,
                    description="Division by zero",
                    location=(node.lineno, node.col_offset),
                    suggestion="Check divisor is not zero before division",
                ))

        # Comparison to None using ==
        if isinstance(node, ast.Compare):
            for op, comp in zip(node.ops, node.comparators):
                if isinstance(op, ast.Eq) and isinstance(comp, ast.Constant) and comp.value is None:
                    bugs.append(BugAnalysis(
                        bug_type=BugType.LOGIC_ERROR,
                        description="Use 'is None' instead of '== None'",
                        location=(node.lineno, node.col_offset),
                        suggestion="Replace '== None' with 'is None'",
                    ))

    # Check for common string issues
    for i, line in enumerate(lines, 1):
        # Unclosed strings (basic check)
        quote_count = line.count('"') - line.count('\\"')
        if quote_count % 2 != 0:
            # Could be a multi-line string, check context
            if not (line.strip().startswith('"""') or line.strip().endswith('"""')):
                bugs.append(BugAnalysis(
                    bug_type=BugType.SYNTAX_ERROR,
                    description="Possibly unclosed string",
                    location=(i, 0),
                    context=line,
                ))

    return bugs


def _get_syntax_fix_suggestion(exc: SyntaxError) -> str:
    """Get suggestion for fixing a syntax error."""
    msg = str(exc.msg).lower() if exc.msg else ""

    suggestions = {
        "unexpected eof": "Check for unclosed parentheses, brackets, or quotes",
        "invalid syntax": "Check for missing colons, commas, or operators",
        "expected ':'": "Add a colon after the statement",
        "expected an indented block": "Add proper indentation to the block",
        "unindent does not match": "Fix indentation to match previous levels",
        "unterminated string": "Close the string with matching quote",
    }

    for pattern, suggestion in suggestions.items():
        if pattern in msg:
            return suggestion

    return "Review the syntax around the error location"


def _attempt_fix(code: str, bug: BugAnalysis) -> Tuple[str, str]:
    """Attempt to fix a specific bug.

    Returns:
        Tuple of (fixed_code, fix_description)
    """
    lines = code.split('\n')

    if bug.bug_type == BugType.SYNTAX_ERROR:
        return _fix_syntax_error(code, bug)

    elif bug.bug_type == BugType.LOGIC_ERROR:
        if "is None" in bug.suggestion:
            # Fix == None to is None
            fixed = re.sub(r'==\s*None', 'is None', code)
            if fixed != code:
                return fixed, "Changed '== None' to 'is None'"

    elif bug.bug_type == BugType.NAME_ERROR:
        # Try to add missing import or variable definition
        pass

    elif bug.bug_type == BugType.INDEX_ERROR:
        # Add bounds checking suggestion
        pass

    return code, ""


def _fix_syntax_error(code: str, bug: BugAnalysis) -> Tuple[str, str]:
    """Attempt to fix a syntax error."""
    lines = code.split('\n')

    if bug.location and bug.location[0]:
        line_num = bug.location[0] - 1
        if 0 <= line_num < len(lines):
            line = lines[line_num]

            # Missing colon
            if "expected ':'" in bug.description.lower():
                # Add colon at end if missing
                stripped = line.rstrip()
                if stripped and not stripped.endswith(':'):
                    lines[line_num] = stripped + ':'
                    return '\n'.join(lines), f"Added missing colon at line {line_num + 1}"

            # Unclosed parenthesis
            if "(" in line and ")" not in line:
                # Simple case: add closing paren
                lines[line_num] = line.rstrip() + ')'
                return '\n'.join(lines), f"Added closing parenthesis at line {line_num + 1}"

            # Unclosed bracket
            if "[" in line and "]" not in line:
                lines[line_num] = line.rstrip() + ']'
                return '\n'.join(lines), f"Added closing bracket at line {line_num + 1}"

    return code, ""


def _validate_code(code: str) -> bool:
    """Validate that code is syntactically correct."""
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False


def analyze_error_traceback(traceback_str: str) -> Dict[str, Any]:
    """Analyze a Python traceback to extract useful information.

    Args:
        traceback_str: Full traceback string

    Returns:
        Dictionary with analysis
    """
    result = {
        "error_type": "",
        "error_message": "",
        "file": "",
        "line": 0,
        "function": "",
        "code": "",
        "stack_depth": 0,
    }

    lines = traceback_str.strip().split('\n')

    # Parse error type and message (usually last line)
    if lines:
        last_line = lines[-1]
        if ':' in last_line:
            parts = last_line.split(':', 1)
            result["error_type"] = parts[0].strip()
            result["error_message"] = parts[1].strip() if len(parts) > 1 else ""

    # Parse stack frames
    file_pattern = r'File "([^"]+)", line (\d+)(?:, in (\w+))?'
    stack_depth = 0

    for i, line in enumerate(lines):
        match = re.search(file_pattern, line)
        if match:
            stack_depth += 1
            result["file"] = match.group(1)
            result["line"] = int(match.group(2))
            result["function"] = match.group(3) or ""

            # Get the code line (usually next line)
            if i + 1 < len(lines) and not lines[i + 1].startswith('File'):
                result["code"] = lines[i + 1].strip()

    result["stack_depth"] = stack_depth

    return result
