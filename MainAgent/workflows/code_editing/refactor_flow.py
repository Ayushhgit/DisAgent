"""
Refactor Flow - Automated code refactoring workflow.

This workflow provides various refactoring operations:
- Extract function/method
- Rename identifiers
- Simplify expressions
- Remove dead code
- Optimize imports
"""

from __future__ import annotations

import re
import ast
import logging
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class RefactorType(Enum):
    """Types of refactoring operations."""
    EXTRACT_FUNCTION = "extract_function"
    EXTRACT_METHOD = "extract_method"
    RENAME = "rename"
    INLINE = "inline"
    SIMPLIFY = "simplify"
    REMOVE_DEAD_CODE = "remove_dead_code"
    OPTIMIZE_IMPORTS = "optimize_imports"
    EXTRACT_VARIABLE = "extract_variable"
    CONVERT_TO_PROPERTY = "convert_to_property"


@dataclass
class RefactorResult:
    """Result of a refactoring operation."""
    success: bool
    original_code: str
    refactored_code: str
    changes_made: List[str] = field(default_factory=list)
    error: Optional[str] = None


def refactor(code: str, operations: Optional[List[str]] = None) -> str:
    """Refactor code (legacy interface).

    Args:
        code: Code to refactor
        operations: List of refactoring operations to apply

    Returns:
        Refactored code
    """
    if not operations:
        # Apply safe automatic refactorings
        operations = ['optimize_imports', 'remove_dead_code', 'simplify']

    result = refactor_code(code, operations)
    return result.refactored_code if result.success else code


def refactor_code(
    code: str,
    operations: List[str],
    language: str = "python",
) -> RefactorResult:
    """Apply refactoring operations to code.

    Args:
        code: Code to refactor
        operations: List of operation names
        language: Programming language

    Returns:
        RefactorResult with refactored code
    """
    if not code or not code.strip():
        return RefactorResult(
            success=False,
            original_code=code,
            refactored_code=code,
            error="Empty code provided",
        )

    current_code = code
    changes_made = []

    for op in operations:
        try:
            new_code, description = _apply_refactoring(current_code, op, language)
            if new_code != current_code:
                current_code = new_code
                changes_made.append(description)
        except Exception as exc:
            logger.warning(f"Refactoring '{op}' failed: {exc}")

    # Validate result
    if language.lower() == "python":
        try:
            ast.parse(current_code)
        except SyntaxError as exc:
            return RefactorResult(
                success=False,
                original_code=code,
                refactored_code=code,
                error=f"Refactoring produced invalid syntax: {exc}",
            )

    return RefactorResult(
        success=True,
        original_code=code,
        refactored_code=current_code,
        changes_made=changes_made,
    )


def _apply_refactoring(code: str, operation: str, language: str) -> Tuple[str, str]:
    """Apply a single refactoring operation."""
    if language.lower() != "python":
        return code, ""

    if operation == "optimize_imports":
        return _optimize_imports(code)
    elif operation == "remove_dead_code":
        return _remove_dead_code(code)
    elif operation == "simplify":
        return _simplify_code(code)
    elif operation == "remove_unused_variables":
        return _remove_unused_variables(code)
    elif operation == "normalize_strings":
        return _normalize_strings(code)
    else:
        return code, ""


def _optimize_imports(code: str) -> Tuple[str, str]:
    """Optimize and sort imports."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code, ""

    lines = code.split('\n')

    # Find all imports
    import_lines = []
    from_import_lines = []
    import_line_numbers = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            import_line_numbers.add(node.lineno - 1)
            for alias in node.names:
                name = alias.asname if alias.asname else alias.name
                import_lines.append((name, lines[node.lineno - 1].strip()))
        elif isinstance(node, ast.ImportFrom):
            import_line_numbers.add(node.lineno - 1)
            module = node.module or ""
            for alias in node.names:
                name = alias.asname if alias.asname else alias.name
                from_import_lines.append((module, name, lines[node.lineno - 1].strip()))

    if not import_lines and not from_import_lines:
        return code, ""

    # Find unused imports
    used_names = _get_used_names(tree)
    unused = set()

    for name, line in import_lines:
        if name not in used_names and not name.startswith('_'):
            unused.add(line)

    for module, name, line in from_import_lines:
        if name not in used_names and name != '*':
            unused.add(line)

    # Remove unused import lines
    if unused:
        new_lines = []
        for i, line in enumerate(lines):
            if i not in import_line_numbers or line.strip() not in unused:
                new_lines.append(line)
        code = '\n'.join(new_lines)

    # Sort remaining imports
    code = _sort_imports(code)

    return code, f"Optimized imports (removed {len(unused)} unused)"


def _get_used_names(tree: ast.AST) -> Set[str]:
    """Get all names used in the AST."""
    names = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, ast.Attribute):
            # Get root name
            current = node
            while isinstance(current, ast.Attribute):
                current = current.value
            if isinstance(current, ast.Name):
                names.add(current.id)
        elif isinstance(node, ast.FunctionDef):
            # Decorators
            for decorator in node.decorator_list:
                if isinstance(decorator, ast.Name):
                    names.add(decorator.id)
                elif isinstance(decorator, ast.Call) and isinstance(decorator.func, ast.Name):
                    names.add(decorator.func.id)

    return names


def _sort_imports(code: str) -> str:
    """Sort import statements."""
    lines = code.split('\n')

    stdlib_imports = []
    third_party_imports = []
    local_imports = []
    other_lines = []
    in_imports = True

    for line in lines:
        stripped = line.strip()
        if in_imports:
            if stripped.startswith('import ') or stripped.startswith('from '):
                # Categorize import
                if stripped.startswith('from .') or stripped.startswith('from ..'):
                    local_imports.append(line)
                else:
                    # Simple heuristic: single-word modules are likely stdlib
                    parts = stripped.split()
                    if len(parts) >= 2:
                        module = parts[1].split('.')[0]
                        if module in _STDLIB_MODULES:
                            stdlib_imports.append(line)
                        else:
                            third_party_imports.append(line)
            elif stripped == '' and (stdlib_imports or third_party_imports or local_imports):
                continue  # Skip blank lines in import section
            elif stripped.startswith('#') and not other_lines:
                other_lines.append(line)  # Keep leading comments
            else:
                in_imports = False
                other_lines.append(line)
        else:
            other_lines.append(line)

    # Rebuild with sorted imports
    result = []

    if stdlib_imports:
        result.extend(sorted(set(stdlib_imports)))
        result.append('')

    if third_party_imports:
        result.extend(sorted(set(third_party_imports)))
        result.append('')

    if local_imports:
        result.extend(sorted(set(local_imports)))
        result.append('')

    # Add blank line before code
    if result and other_lines and other_lines[0].strip():
        result.append('')

    result.extend(other_lines)

    return '\n'.join(result)


def _remove_dead_code(code: str) -> Tuple[str, str]:
    """Remove dead/unreachable code."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code, ""

    lines = code.split('\n')
    lines_to_remove = set()

    class DeadCodeFinder(ast.NodeVisitor):
        def visit_FunctionDef(self, node):
            # Find code after return
            in_dead_zone = False
            for child in node.body:
                if in_dead_zone:
                    if hasattr(child, 'lineno'):
                        lines_to_remove.add(child.lineno - 1)
                elif isinstance(child, ast.Return):
                    in_dead_zone = True

            self.generic_visit(node)

        def visit_If(self, node):
            # Check for always-true or always-false conditions
            if isinstance(node.test, ast.Constant):
                if node.test.value is True:
                    # Else branch is dead
                    for n in node.orelse:
                        if hasattr(n, 'lineno'):
                            lines_to_remove.add(n.lineno - 1)
                elif node.test.value is False:
                    # If branch is dead
                    for n in node.body:
                        if hasattr(n, 'lineno'):
                            lines_to_remove.add(n.lineno - 1)

            self.generic_visit(node)

    finder = DeadCodeFinder()
    finder.visit(tree)

    if not lines_to_remove:
        return code, ""

    new_lines = [line for i, line in enumerate(lines) if i not in lines_to_remove]
    return '\n'.join(new_lines), f"Removed {len(lines_to_remove)} lines of dead code"


def _simplify_code(code: str) -> Tuple[str, str]:
    """Simplify code expressions."""
    changes = 0

    # Simplify boolean comparisons
    # if x == True -> if x
    code, n = re.subn(r'if\s+(\w+)\s*==\s*True\s*:', r'if \1:', code)
    changes += n

    # if x == False -> if not x
    code, n = re.subn(r'if\s+(\w+)\s*==\s*False\s*:', r'if not \1:', code)
    changes += n

    # if x != True -> if not x
    code, n = re.subn(r'if\s+(\w+)\s*!=\s*True\s*:', r'if not \1:', code)
    changes += n

    # if x != False -> if x
    code, n = re.subn(r'if\s+(\w+)\s*!=\s*False\s*:', r'if \1:', code)
    changes += n

    # x == None -> x is None
    code, n = re.subn(r'(\w+)\s*==\s*None', r'\1 is None', code)
    changes += n

    # x != None -> x is not None
    code, n = re.subn(r'(\w+)\s*!=\s*None', r'\1 is not None', code)
    changes += n

    # len(x) == 0 -> not x
    code, n = re.subn(r'len\((\w+)\)\s*==\s*0', r'not \1', code)
    changes += n

    # len(x) > 0 -> x
    code, n = re.subn(r'len\((\w+)\)\s*>\s*0', r'\1', code)
    changes += n

    if changes:
        return code, f"Simplified {changes} expressions"
    return code, ""


def _remove_unused_variables(code: str) -> Tuple[str, str]:
    """Remove unused variable assignments."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code, ""

    # Find all assigned names
    assignments = {}  # name -> line number
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    assignments[target.id] = node.lineno

    # Find all used names
    used_names = _get_used_names(tree)

    # Find unused assignments
    unused = {name: line for name, line in assignments.items()
              if name not in used_names and not name.startswith('_')}

    if not unused:
        return code, ""

    # Remove unused assignment lines (careful - this is a simple approach)
    lines = code.split('\n')
    lines_to_remove = set(unused.values())

    # Only remove simple assignments, not multi-line or part of tuple unpacking
    safe_to_remove = set()
    for line_num in lines_to_remove:
        line = lines[line_num - 1]
        # Simple assignment: name = value
        if re.match(r'^\s*\w+\s*=\s*.+$', line):
            safe_to_remove.add(line_num - 1)

    if not safe_to_remove:
        return code, ""

    new_lines = [line for i, line in enumerate(lines) if i not in safe_to_remove]
    return '\n'.join(new_lines), f"Removed {len(safe_to_remove)} unused variables"


def _normalize_strings(code: str) -> Tuple[str, str]:
    """Normalize string quotes to use consistent style."""
    # This is a simplified version - proper implementation would use tokenize
    changes = 0

    # Convert single quotes to double quotes (simple cases only)
    # This is tricky because we need to handle nested quotes

    return code, "" if changes == 0 else f"Normalized {changes} strings"


def extract_function(
    code: str,
    start_line: int,
    end_line: int,
    function_name: str,
) -> RefactorResult:
    """Extract lines into a new function.

    Args:
        code: Original code
        start_line: Start line (1-indexed)
        end_line: End line (1-indexed)
        function_name: Name for new function

    Returns:
        RefactorResult with extracted function
    """
    lines = code.split('\n')

    if start_line < 1 or end_line > len(lines) or start_line > end_line:
        return RefactorResult(
            success=False,
            original_code=code,
            refactored_code=code,
            error="Invalid line range",
        )

    # Extract the lines
    extracted_lines = lines[start_line - 1:end_line]

    # Get indentation of first line
    first_line = extracted_lines[0]
    indent = len(first_line) - len(first_line.lstrip())

    # Remove the original indentation and add function body indentation
    dedented_lines = []
    for line in extracted_lines:
        if line.strip():
            dedented = line[indent:] if len(line) > indent else line.lstrip()
            dedented_lines.append('    ' + dedented)
        else:
            dedented_lines.append('')

    # Create function definition
    function_def = f"def {function_name}():\n" + '\n'.join(dedented_lines)

    # Replace extracted lines with function call
    call_line = ' ' * indent + f"{function_name}()"

    new_lines = (
        lines[:start_line - 1] +
        [call_line] +
        lines[end_line:] +
        ['', function_def]
    )

    new_code = '\n'.join(new_lines)

    return RefactorResult(
        success=True,
        original_code=code,
        refactored_code=new_code,
        changes_made=[f"Extracted lines {start_line}-{end_line} into function '{function_name}'"],
    )


def rename_identifier(
    code: str,
    old_name: str,
    new_name: str,
) -> RefactorResult:
    """Rename an identifier throughout the code.

    Args:
        code: Original code
        old_name: Current name
        new_name: New name

    Returns:
        RefactorResult with renamed identifier
    """
    # Use word boundary to avoid partial matches
    pattern = r'\b' + re.escape(old_name) + r'\b'
    new_code, count = re.subn(pattern, new_name, code)

    if count == 0:
        return RefactorResult(
            success=False,
            original_code=code,
            refactored_code=code,
            error=f"Identifier '{old_name}' not found",
        )

    return RefactorResult(
        success=True,
        original_code=code,
        refactored_code=new_code,
        changes_made=[f"Renamed '{old_name}' to '{new_name}' ({count} occurrences)"],
    )


# Common Python standard library modules for import categorization
_STDLIB_MODULES = {
    'abc', 'argparse', 'ast', 'asyncio', 'base64', 'collections', 'contextlib',
    'copy', 'csv', 'dataclasses', 'datetime', 'decimal', 'enum', 'functools',
    'hashlib', 'heapq', 'http', 'io', 'itertools', 'json', 'logging', 'math',
    'os', 'pathlib', 'pickle', 'queue', 're', 'shutil', 'socket', 'sqlite3',
    'string', 'subprocess', 'sys', 'tempfile', 'threading', 'time', 'typing',
    'unittest', 'urllib', 'uuid', 'warnings', 'weakref', 'xml', 'zipfile',
}
