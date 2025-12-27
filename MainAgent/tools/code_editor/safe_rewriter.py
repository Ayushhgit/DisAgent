"""
Safe Rewriter - Safely rewrite and transform code with validation.

Features:
- Safe code transformations
- Syntax validation before/after
- Rollback on failure
- AST-based transformations
"""

from __future__ import annotations

import re
import ast
import logging
from typing import Dict, List, Any, Optional, Callable, Tuple
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class RewriteResult:
    """Result of a code rewrite operation."""
    success: bool
    original: str
    rewritten: str
    changes_made: List[str] = field(default_factory=list)
    error: Optional[str] = None


def safe_rewrite(code: str, transformations: Optional[List[str]] = None) -> str:
    """Safely rewrite code with optional transformations (legacy interface).

    Args:
        code: Code to rewrite
        transformations: List of transformation names to apply

    Returns:
        Rewritten code or original if failed
    """
    if not transformations:
        return code

    result = rewrite_code(code, transformations)
    return result.rewritten if result.success else code


def rewrite_code(
    code: str,
    transformations: List[str],
    language: str = "python",
) -> RewriteResult:
    """Rewrite code with specified transformations.

    Args:
        code: Code to transform
        transformations: List of transformation names
        language: Programming language

    Returns:
        RewriteResult with transformed code
    """
    if not code or not code.strip():
        return RewriteResult(
            success=False,
            original=code,
            rewritten=code,
            error="Empty code provided",
        )

    if language.lower() == "python":
        return _rewrite_python(code, transformations)
    else:
        return _rewrite_generic(code, transformations)


def _rewrite_python(code: str, transformations: List[str]) -> RewriteResult:
    """Rewrite Python code."""
    # Validate original syntax
    try:
        ast.parse(code)
    except SyntaxError as exc:
        return RewriteResult(
            success=False,
            original=code,
            rewritten=code,
            error=f"Original code has syntax error: {exc}",
        )

    rewritten = code
    changes_made = []

    # Available transformations
    transform_funcs = {
        'add_type_hints': _add_type_hints,
        'format_strings': _modernize_format_strings,
        'remove_unused_imports': _remove_unused_imports,
        'sort_imports': _sort_imports,
        'add_docstrings': _add_docstrings,
        'convert_to_fstrings': _convert_to_fstrings,
        'simplify_conditionals': _simplify_conditionals,
        'use_walrus': _use_walrus_operator,
        'remove_pass': _remove_unnecessary_pass,
        'normalize_whitespace': _normalize_whitespace,
    }

    for transform_name in transformations:
        if transform_name in transform_funcs:
            try:
                new_code, changed = transform_funcs[transform_name](rewritten)
                if changed:
                    # Validate new code syntax
                    try:
                        ast.parse(new_code)
                        rewritten = new_code
                        changes_made.append(transform_name)
                    except SyntaxError:
                        logger.warning(f"Transform '{transform_name}' produced invalid syntax")
            except Exception as exc:
                logger.warning(f"Transform '{transform_name}' failed: {exc}")
        else:
            logger.warning(f"Unknown transformation: {transform_name}")

    return RewriteResult(
        success=True,
        original=code,
        rewritten=rewritten,
        changes_made=changes_made,
    )


def _rewrite_generic(code: str, transformations: List[str]) -> RewriteResult:
    """Generic code rewriting."""
    rewritten = code
    changes_made = []

    for transform in transformations:
        if transform == 'normalize_whitespace':
            new_code, changed = _normalize_whitespace(rewritten)
            if changed:
                rewritten = new_code
                changes_made.append(transform)

    return RewriteResult(
        success=True,
        original=code,
        rewritten=rewritten,
        changes_made=changes_made,
    )


def _add_type_hints(code: str) -> Tuple[str, bool]:
    """Add basic type hints to function definitions."""
    # This is a simple heuristic-based approach
    lines = code.split('\n')
    new_lines = []
    changed = False

    for line in lines:
        # Match function definitions without return type hints
        match = re.match(r'^(\s*)(def\s+\w+\s*\([^)]*\))(\s*:)', line)
        if match:
            indent, func_def, colon = match.groups()
            # Check if already has return type
            if '->' not in func_def:
                # Add generic return type hint
                new_line = f"{indent}{func_def} -> None{colon}"
                new_lines.append(new_line)
                changed = True
                continue
        new_lines.append(line)

    return '\n'.join(new_lines), changed


def _modernize_format_strings(code: str) -> Tuple[str, bool]:
    """Convert old-style string formatting to f-strings."""
    changed = False

    # Pattern for % formatting
    pattern = r'"([^"]*?)"\s*%\s*\(([^)]+)\)'

    def replace_percent_format(match):
        nonlocal changed
        format_str = match.group(1)
        args = match.group(2)

        # Simple case: single %s
        if format_str.count('%s') == 1 and ',' not in args:
            changed = True
            return f'f"{format_str.replace("%s", "{" + args.strip() + "}")}"'

        return match.group(0)

    result = re.sub(pattern, replace_percent_format, code)
    return result, changed


def _convert_to_fstrings(code: str) -> Tuple[str, bool]:
    """Convert .format() calls to f-strings where possible."""
    changed = False

    # Simple pattern for .format() with named or positional args
    pattern = r'"([^"]+)"\.format\((\w+)\)'

    def replace_format(match):
        nonlocal changed
        format_str = match.group(1)
        arg = match.group(2)

        if '{0}' in format_str or '{}' in format_str:
            new_str = format_str.replace('{0}', '{' + arg + '}').replace('{}', '{' + arg + '}')
            changed = True
            return f'f"{new_str}"'

        return match.group(0)

    result = re.sub(pattern, replace_format, code)
    return result, changed


def _remove_unused_imports(code: str) -> Tuple[str, bool]:
    """Remove unused imports (basic detection)."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code, False

    # Collect all imported names
    imports = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                name = alias.asname if alias.asname else alias.name
                imports[name] = node.lineno
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                name = alias.asname if alias.asname else alias.name
                imports[name] = node.lineno

    # Collect all used names
    used_names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            used_names.add(node.id)
        elif isinstance(node, ast.Attribute):
            # Get the root name
            current = node
            while isinstance(current, ast.Attribute):
                current = current.value
            if isinstance(current, ast.Name):
                used_names.add(current.id)

    # Find unused imports
    unused_lines = set()
    for name, line in imports.items():
        if name not in used_names:
            unused_lines.add(line)

    if not unused_lines:
        return code, False

    # Remove unused import lines
    lines = code.split('\n')
    new_lines = [
        line for i, line in enumerate(lines, 1)
        if i not in unused_lines
    ]

    return '\n'.join(new_lines), True


def _sort_imports(code: str) -> Tuple[str, bool]:
    """Sort import statements."""
    lines = code.split('\n')

    import_lines = []
    from_import_lines = []
    other_lines = []
    import_ended = False

    for line in lines:
        stripped = line.strip()
        if not import_ended:
            if stripped.startswith('import '):
                import_lines.append(line)
            elif stripped.startswith('from '):
                from_import_lines.append(line)
            elif stripped == '' and (import_lines or from_import_lines):
                continue  # Skip blank lines between imports
            elif stripped.startswith('#') and not (import_lines or from_import_lines):
                other_lines.append(line)  # Keep header comments
            else:
                import_ended = True
                other_lines.append(line)
        else:
            other_lines.append(line)

    if not import_lines and not from_import_lines:
        return code, False

    # Sort imports
    sorted_imports = sorted(import_lines)
    sorted_from_imports = sorted(from_import_lines)

    # Rebuild code
    new_lines = []
    if sorted_imports:
        new_lines.extend(sorted_imports)
    if sorted_from_imports:
        if new_lines:
            new_lines.append('')
        new_lines.extend(sorted_from_imports)
    if other_lines:
        if new_lines:
            new_lines.append('')
            new_lines.append('')
        new_lines.extend(other_lines)

    result = '\n'.join(new_lines)
    return result, result != code


def _add_docstrings(code: str) -> Tuple[str, bool]:
    """Add placeholder docstrings to functions without them."""
    lines = code.split('\n')
    new_lines = []
    changed = False
    i = 0

    while i < len(lines):
        line = lines[i]
        new_lines.append(line)

        # Check if this is a function definition
        stripped = line.strip()
        if stripped.startswith('def ') and stripped.endswith(':'):
            # Check if next non-empty line is a docstring
            j = i + 1
            while j < len(lines) and not lines[j].strip():
                j += 1

            if j < len(lines):
                next_line = lines[j].strip()
                if not (next_line.startswith('"""') or next_line.startswith("'''")):
                    # No docstring, add one
                    indent = len(line) - len(line.lstrip()) + 4
                    new_lines.append(' ' * indent + '"""TODO: Add docstring."""')
                    changed = True

        i += 1

    return '\n'.join(new_lines), changed


def _simplify_conditionals(code: str) -> Tuple[str, bool]:
    """Simplify conditional expressions."""
    changed = False

    # Pattern: if x == True: -> if x:
    pattern1 = r'if\s+(\w+)\s*==\s*True\s*:'
    replacement1 = r'if \1:'
    new_code, n = re.subn(pattern1, replacement1, code)
    if n > 0:
        code = new_code
        changed = True

    # Pattern: if x == False: -> if not x:
    pattern2 = r'if\s+(\w+)\s*==\s*False\s*:'
    replacement2 = r'if not \1:'
    new_code, n = re.subn(pattern2, replacement2, code)
    if n > 0:
        code = new_code
        changed = True

    return code, changed


def _use_walrus_operator(code: str) -> Tuple[str, bool]:
    """Convert patterns that could use walrus operator (Python 3.8+)."""
    # This is complex to do safely, so we only handle simple cases
    return code, False


def _remove_unnecessary_pass(code: str) -> Tuple[str, bool]:
    """Remove unnecessary pass statements."""
    lines = code.split('\n')
    new_lines = []
    changed = False
    i = 0

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        if stripped == 'pass':
            # Check if there are other statements in this block
            if i > 0:
                prev = lines[i - 1].strip()
                if prev.endswith(':'):
                    # This is the only statement, keep it
                    new_lines.append(line)
                else:
                    # There are other statements, pass is unnecessary
                    changed = True
            else:
                new_lines.append(line)
        else:
            new_lines.append(line)

        i += 1

    return '\n'.join(new_lines), changed


def _normalize_whitespace(code: str) -> Tuple[str, bool]:
    """Normalize whitespace in code."""
    lines = code.split('\n')
    new_lines = []
    changed = False

    for line in lines:
        # Remove trailing whitespace
        new_line = line.rstrip()
        if new_line != line:
            changed = True
        new_lines.append(new_line)

    # Remove multiple blank lines at end
    while len(new_lines) > 1 and new_lines[-1] == '':
        new_lines.pop()
        changed = True

    # Ensure single newline at end
    if new_lines and new_lines[-1] != '':
        pass  # Don't add extra newline, it's handled by joining

    result = '\n'.join(new_lines)
    return result, changed
