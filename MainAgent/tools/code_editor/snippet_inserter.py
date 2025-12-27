"""
Snippet Inserter - Insert code snippets at specific locations.

Features:
- Position-based insertion
- Line-based insertion
- Function/class-level insertion
- Import statement insertion
- Smart indentation handling
"""

from __future__ import annotations

import re
import ast
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class InsertPosition(Enum):
    """Position for snippet insertion."""
    START = "start"
    END = "end"
    BEFORE_LINE = "before_line"
    AFTER_LINE = "after_line"
    BEFORE_FUNCTION = "before_function"
    AFTER_FUNCTION = "after_function"
    INSIDE_FUNCTION = "inside_function"
    BEFORE_CLASS = "before_class"
    AFTER_CLASS = "after_class"
    INSIDE_CLASS = "inside_class"
    AFTER_IMPORTS = "after_imports"


@dataclass
class InsertResult:
    """Result of a snippet insertion."""
    success: bool
    content: str
    line_inserted: Optional[int] = None
    error: Optional[str] = None


def insert_snippet(
    content: str,
    snippet: str,
    pos: int = 0,
) -> str:
    """Insert snippet at character position (legacy interface).

    Args:
        content: Original content
        snippet: Snippet to insert
        pos: Character position

    Returns:
        Content with snippet inserted
    """
    return content[:pos] + snippet + content[pos:]


def insert_at_position(
    content: str,
    snippet: str,
    position: InsertPosition,
    target: Optional[str] = None,
    line_number: Optional[int] = None,
) -> InsertResult:
    """Insert snippet at specified position.

    Args:
        content: Original content
        snippet: Code snippet to insert
        position: Where to insert
        target: Target function/class name (for function/class positions)
        line_number: Line number (for line-based positions)

    Returns:
        InsertResult with new content
    """
    if not content:
        return InsertResult(
            success=True,
            content=snippet,
            line_inserted=1,
        )

    lines = content.split('\n')

    if position == InsertPosition.START:
        return InsertResult(
            success=True,
            content=snippet + '\n' + content,
            line_inserted=1,
        )

    elif position == InsertPosition.END:
        # Ensure proper spacing
        if not content.endswith('\n'):
            content += '\n'
        return InsertResult(
            success=True,
            content=content + snippet,
            line_inserted=len(lines) + 1,
        )

    elif position == InsertPosition.BEFORE_LINE:
        if line_number is None or line_number < 1 or line_number > len(lines) + 1:
            return InsertResult(
                success=False,
                content=content,
                error=f"Invalid line number: {line_number}",
            )
        # Get indentation of target line
        target_line = lines[line_number - 1] if line_number <= len(lines) else ""
        indent = _get_indentation(target_line)
        indented_snippet = _indent_snippet(snippet, indent)
        lines.insert(line_number - 1, indented_snippet)
        return InsertResult(
            success=True,
            content='\n'.join(lines),
            line_inserted=line_number,
        )

    elif position == InsertPosition.AFTER_LINE:
        if line_number is None or line_number < 1 or line_number > len(lines):
            return InsertResult(
                success=False,
                content=content,
                error=f"Invalid line number: {line_number}",
            )
        target_line = lines[line_number - 1]
        indent = _get_indentation(target_line)
        indented_snippet = _indent_snippet(snippet, indent)
        lines.insert(line_number, indented_snippet)
        return InsertResult(
            success=True,
            content='\n'.join(lines),
            line_inserted=line_number + 1,
        )

    elif position == InsertPosition.AFTER_IMPORTS:
        return _insert_after_imports(content, snippet)

    elif position in (InsertPosition.BEFORE_FUNCTION, InsertPosition.AFTER_FUNCTION,
                      InsertPosition.INSIDE_FUNCTION):
        if not target:
            return InsertResult(
                success=False,
                content=content,
                error="Target function name required",
            )
        return _insert_at_function(content, snippet, position, target)

    elif position in (InsertPosition.BEFORE_CLASS, InsertPosition.AFTER_CLASS,
                      InsertPosition.INSIDE_CLASS):
        if not target:
            return InsertResult(
                success=False,
                content=content,
                error="Target class name required",
            )
        return _insert_at_class(content, snippet, position, target)

    return InsertResult(
        success=False,
        content=content,
        error=f"Unknown position: {position}",
    )


def insert_at_line(content: str, snippet: str, line: int, after: bool = False) -> InsertResult:
    """Insert snippet at specific line.

    Args:
        content: Original content
        snippet: Snippet to insert
        line: Line number (1-indexed)
        after: If True, insert after line; if False, insert before

    Returns:
        InsertResult
    """
    position = InsertPosition.AFTER_LINE if after else InsertPosition.BEFORE_LINE
    return insert_at_position(content, snippet, position, line_number=line)


def insert_import(content: str, import_statement: str) -> InsertResult:
    """Insert an import statement in the correct location.

    Args:
        content: Original content
        import_statement: Import statement to add

    Returns:
        InsertResult
    """
    lines = content.split('\n')

    # Find the last import line
    last_import_line = 0
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith('import ') or stripped.startswith('from '):
            last_import_line = i

    # Check if import already exists
    import_stripped = import_statement.strip()
    for line in lines:
        if line.strip() == import_stripped:
            return InsertResult(
                success=True,
                content=content,
                error="Import already exists",
            )

    # Insert after last import or at beginning
    insert_line = last_import_line + 1 if last_import_line > 0 else 0

    # Skip docstrings and comments at the beginning
    if last_import_line == 0:
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith('"""') or stripped.startswith("'''"):
                # Skip docstring
                if stripped.count('"""') == 2 or stripped.count("'''") == 2:
                    insert_line = i + 1
                    break
                # Multi-line docstring
                for j in range(i + 1, len(lines)):
                    if '"""' in lines[j] or "'''" in lines[j]:
                        insert_line = j + 1
                        break
                break
            elif stripped.startswith('#'):
                insert_line = i + 1
            elif stripped:
                break

    lines.insert(insert_line, import_statement)

    return InsertResult(
        success=True,
        content='\n'.join(lines),
        line_inserted=insert_line + 1,
    )


def _insert_after_imports(content: str, snippet: str) -> InsertResult:
    """Insert snippet after all import statements."""
    lines = content.split('\n')

    # Find the last import line
    last_import_line = -1
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith('import ') or stripped.startswith('from '):
            last_import_line = i

    if last_import_line == -1:
        # No imports, insert at beginning after docstring
        insert_line = 0
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith('"""') or stripped.startswith("'''") or stripped.startswith('#'):
                insert_line = i + 1
            elif stripped:
                break
    else:
        insert_line = last_import_line + 1

    # Add blank lines for separation
    snippet_with_spacing = '\n\n' + snippet if last_import_line >= 0 else snippet

    lines.insert(insert_line, snippet_with_spacing)

    return InsertResult(
        success=True,
        content='\n'.join(lines),
        line_inserted=insert_line + 1,
    )


def _insert_at_function(
    content: str,
    snippet: str,
    position: InsertPosition,
    function_name: str,
) -> InsertResult:
    """Insert snippet relative to a function."""
    try:
        tree = ast.parse(content)
    except SyntaxError as exc:
        return InsertResult(
            success=False,
            content=content,
            error=f"Syntax error: {exc}",
        )

    lines = content.split('\n')

    # Find the function
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name == function_name:
                if position == InsertPosition.BEFORE_FUNCTION:
                    # Insert before function definition
                    insert_line = node.lineno - 1
                    indent = _get_indentation(lines[node.lineno - 1])
                    indented_snippet = _indent_snippet(snippet, indent)
                    lines.insert(insert_line, indented_snippet + '\n')
                    return InsertResult(
                        success=True,
                        content='\n'.join(lines),
                        line_inserted=insert_line + 1,
                    )

                elif position == InsertPosition.AFTER_FUNCTION:
                    # Find end of function
                    end_line = node.end_lineno if hasattr(node, 'end_lineno') else _find_function_end(lines, node.lineno)
                    indent = _get_indentation(lines[node.lineno - 1])
                    indented_snippet = '\n' + _indent_snippet(snippet, indent)
                    lines.insert(end_line, indented_snippet)
                    return InsertResult(
                        success=True,
                        content='\n'.join(lines),
                        line_inserted=end_line + 1,
                    )

                elif position == InsertPosition.INSIDE_FUNCTION:
                    # Insert as first line of function body
                    insert_line = node.lineno  # After def line
                    func_indent = _get_indentation(lines[node.lineno - 1])
                    body_indent = func_indent + '    '
                    indented_snippet = _indent_snippet(snippet, body_indent)
                    lines.insert(insert_line, indented_snippet)
                    return InsertResult(
                        success=True,
                        content='\n'.join(lines),
                        line_inserted=insert_line + 1,
                    )

    return InsertResult(
        success=False,
        content=content,
        error=f"Function '{function_name}' not found",
    )


def _insert_at_class(
    content: str,
    snippet: str,
    position: InsertPosition,
    class_name: str,
) -> InsertResult:
    """Insert snippet relative to a class."""
    try:
        tree = ast.parse(content)
    except SyntaxError as exc:
        return InsertResult(
            success=False,
            content=content,
            error=f"Syntax error: {exc}",
        )

    lines = content.split('\n')

    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            if node.name == class_name:
                if position == InsertPosition.BEFORE_CLASS:
                    insert_line = node.lineno - 1
                    indent = _get_indentation(lines[node.lineno - 1])
                    indented_snippet = _indent_snippet(snippet, indent)
                    lines.insert(insert_line, indented_snippet + '\n')
                    return InsertResult(
                        success=True,
                        content='\n'.join(lines),
                        line_inserted=insert_line + 1,
                    )

                elif position == InsertPosition.AFTER_CLASS:
                    end_line = node.end_lineno if hasattr(node, 'end_lineno') else _find_class_end(lines, node.lineno)
                    indent = _get_indentation(lines[node.lineno - 1])
                    indented_snippet = '\n' + _indent_snippet(snippet, indent)
                    lines.insert(end_line, indented_snippet)
                    return InsertResult(
                        success=True,
                        content='\n'.join(lines),
                        line_inserted=end_line + 1,
                    )

                elif position == InsertPosition.INSIDE_CLASS:
                    insert_line = node.lineno
                    class_indent = _get_indentation(lines[node.lineno - 1])
                    body_indent = class_indent + '    '
                    indented_snippet = _indent_snippet(snippet, body_indent)
                    lines.insert(insert_line, indented_snippet)
                    return InsertResult(
                        success=True,
                        content='\n'.join(lines),
                        line_inserted=insert_line + 1,
                    )

    return InsertResult(
        success=False,
        content=content,
        error=f"Class '{class_name}' not found",
    )


def _get_indentation(line: str) -> str:
    """Get the indentation of a line."""
    stripped = line.lstrip()
    if stripped:
        return line[:len(line) - len(stripped)]
    return ""


def _indent_snippet(snippet: str, indent: str) -> str:
    """Indent a snippet to match the given indentation."""
    lines = snippet.split('\n')
    return '\n'.join(indent + line if line.strip() else line for line in lines)


def _find_function_end(lines: List[str], start_line: int) -> int:
    """Find the end line of a function (heuristic)."""
    base_indent = len(lines[start_line - 1]) - len(lines[start_line - 1].lstrip())

    for i in range(start_line, len(lines)):
        line = lines[i]
        if line.strip() and not line.strip().startswith('#'):
            current_indent = len(line) - len(line.lstrip())
            if current_indent <= base_indent and i > start_line:
                return i

    return len(lines)


def _find_class_end(lines: List[str], start_line: int) -> int:
    """Find the end line of a class (heuristic)."""
    return _find_function_end(lines, start_line)
