"""
Diff Tool - Compare and generate differences between code versions.

Features:
- Unified diff generation
- Side-by-side comparison
- Semantic diff (function/class level)
- Patch application
"""

from __future__ import annotations

import difflib
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class DiffType(Enum):
    """Types of differences."""
    ADDED = "added"
    REMOVED = "removed"
    MODIFIED = "modified"
    UNCHANGED = "unchanged"


@dataclass
class DiffLine:
    """A single line in a diff."""
    line_number_old: Optional[int]
    line_number_new: Optional[int]
    content: str
    diff_type: DiffType


@dataclass
class DiffHunk:
    """A hunk of changes in a diff."""
    old_start: int
    old_count: int
    new_start: int
    new_count: int
    lines: List[DiffLine] = field(default_factory=list)


@dataclass
class DiffResult:
    """Result of a diff operation."""
    old_name: str
    new_name: str
    hunks: List[DiffHunk] = field(default_factory=list)
    stats: Dict[str, int] = field(default_factory=dict)
    unified_diff: str = ""
    has_changes: bool = False


def diff(a: str, b: str, context: int = 3) -> Dict[str, Any]:
    """Generate diff between two strings (legacy interface).

    Args:
        a: Original content
        b: Modified content
        context: Number of context lines

    Returns:
        Diff result as dictionary
    """
    result = generate_diff(a, b, context=context)
    return {
        "a_len": len(a),
        "b_len": len(b),
        "has_changes": result.has_changes,
        "unified_diff": result.unified_diff,
        "stats": result.stats,
        "hunks": len(result.hunks),
    }


def generate_diff(
    old_content: str,
    new_content: str,
    old_name: str = "original",
    new_name: str = "modified",
    context: int = 3,
) -> DiffResult:
    """Generate a diff between two content strings.

    Args:
        old_content: Original content
        new_content: New/modified content
        old_name: Name for old version
        new_name: Name for new version
        context: Number of context lines

    Returns:
        DiffResult with changes
    """
    old_lines = old_content.splitlines(keepends=True)
    new_lines = new_content.splitlines(keepends=True)

    # Generate unified diff
    unified = list(difflib.unified_diff(
        old_lines,
        new_lines,
        fromfile=old_name,
        tofile=new_name,
        n=context,
    ))

    unified_str = ''.join(unified)

    # Calculate stats
    stats = _calculate_diff_stats(old_lines, new_lines)

    # Parse hunks
    hunks = _parse_unified_diff(unified)

    return DiffResult(
        old_name=old_name,
        new_name=new_name,
        hunks=hunks,
        stats=stats,
        unified_diff=unified_str,
        has_changes=len(unified) > 0 and stats.get('changes', 0) > 0,
    )


def _calculate_diff_stats(old_lines: List[str], new_lines: List[str]) -> Dict[str, int]:
    """Calculate diff statistics."""
    matcher = difflib.SequenceMatcher(None, old_lines, new_lines)

    added = 0
    removed = 0
    unchanged = 0

    for op, i1, i2, j1, j2 in matcher.get_opcodes():
        if op == 'equal':
            unchanged += i2 - i1
        elif op == 'insert':
            added += j2 - j1
        elif op == 'delete':
            removed += i2 - i1
        elif op == 'replace':
            removed += i2 - i1
            added += j2 - j1

    return {
        'added': added,
        'removed': removed,
        'unchanged': unchanged,
        'changes': added + removed,
        'total_old': len(old_lines),
        'total_new': len(new_lines),
    }


def _parse_unified_diff(diff_lines: List[str]) -> List[DiffHunk]:
    """Parse unified diff into hunks."""
    hunks = []
    current_hunk = None

    for line in diff_lines:
        if line.startswith('@@'):
            # Parse hunk header
            # Format: @@ -old_start,old_count +new_start,new_count @@
            import re
            match = re.match(r'@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@', line)
            if match:
                if current_hunk:
                    hunks.append(current_hunk)

                current_hunk = DiffHunk(
                    old_start=int(match.group(1)),
                    old_count=int(match.group(2) or 1),
                    new_start=int(match.group(3)),
                    new_count=int(match.group(4) or 1),
                )
        elif current_hunk is not None:
            if line.startswith('-') and not line.startswith('---'):
                current_hunk.lines.append(DiffLine(
                    line_number_old=len([l for l in current_hunk.lines if l.diff_type != DiffType.ADDED]) + current_hunk.old_start,
                    line_number_new=None,
                    content=line[1:],
                    diff_type=DiffType.REMOVED,
                ))
            elif line.startswith('+') and not line.startswith('+++'):
                current_hunk.lines.append(DiffLine(
                    line_number_old=None,
                    line_number_new=len([l for l in current_hunk.lines if l.diff_type != DiffType.REMOVED]) + current_hunk.new_start,
                    content=line[1:],
                    diff_type=DiffType.ADDED,
                ))
            elif line.startswith(' '):
                current_hunk.lines.append(DiffLine(
                    line_number_old=len([l for l in current_hunk.lines if l.diff_type != DiffType.ADDED]) + current_hunk.old_start,
                    line_number_new=len([l for l in current_hunk.lines if l.diff_type != DiffType.REMOVED]) + current_hunk.new_start,
                    content=line[1:],
                    diff_type=DiffType.UNCHANGED,
                ))

    if current_hunk:
        hunks.append(current_hunk)

    return hunks


def generate_patch(old_content: str, new_content: str, filename: str = "file") -> str:
    """Generate a patch file content.

    Args:
        old_content: Original content
        new_content: New content
        filename: Filename for the patch

    Returns:
        Patch file content
    """
    old_lines = old_content.splitlines(keepends=True)
    new_lines = new_content.splitlines(keepends=True)

    # Ensure files end with newline
    if old_lines and not old_lines[-1].endswith('\n'):
        old_lines[-1] += '\n'
    if new_lines and not new_lines[-1].endswith('\n'):
        new_lines[-1] += '\n'

    diff_lines = list(difflib.unified_diff(
        old_lines,
        new_lines,
        fromfile=f"a/{filename}",
        tofile=f"b/{filename}",
    ))

    return ''.join(diff_lines)


def apply_patch(content: str, patch: str) -> Tuple[str, bool]:
    """Apply a unified diff patch to content.

    Args:
        content: Original content
        patch: Unified diff patch

    Returns:
        Tuple of (new_content, success)
    """
    lines = content.splitlines()
    patch_lines = patch.splitlines()

    result_lines = list(lines)
    offset = 0
    success = True

    current_line = 0
    for patch_line in patch_lines:
        if patch_line.startswith('@@'):
            import re
            match = re.match(r'@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@', patch_line)
            if match:
                current_line = int(match.group(1)) - 1 + offset
        elif patch_line.startswith('-') and not patch_line.startswith('---'):
            # Remove line
            if current_line < len(result_lines):
                expected = patch_line[1:]
                if result_lines[current_line].rstrip() == expected.rstrip():
                    result_lines.pop(current_line)
                    offset -= 1
                else:
                    success = False
        elif patch_line.startswith('+') and not patch_line.startswith('+++'):
            # Add line
            result_lines.insert(current_line, patch_line[1:])
            current_line += 1
            offset += 1
        elif patch_line.startswith(' '):
            # Context line - verify it matches
            if current_line < len(result_lines):
                expected = patch_line[1:]
                if result_lines[current_line].rstrip() != expected.rstrip():
                    success = False
            current_line += 1

    return '\n'.join(result_lines), success


def compare_lines(old_lines: List[str], new_lines: List[str]) -> List[Dict[str, Any]]:
    """Compare two lists of lines and return detailed comparison.

    Args:
        old_lines: Original lines
        new_lines: New lines

    Returns:
        List of comparison dictionaries
    """
    matcher = difflib.SequenceMatcher(None, old_lines, new_lines)
    result = []

    for op, i1, i2, j1, j2 in matcher.get_opcodes():
        if op == 'equal':
            for idx, line in enumerate(old_lines[i1:i2]):
                result.append({
                    'type': 'equal',
                    'old_line': i1 + idx + 1,
                    'new_line': j1 + idx + 1,
                    'content': line,
                })
        elif op == 'insert':
            for idx, line in enumerate(new_lines[j1:j2]):
                result.append({
                    'type': 'insert',
                    'old_line': None,
                    'new_line': j1 + idx + 1,
                    'content': line,
                })
        elif op == 'delete':
            for idx, line in enumerate(old_lines[i1:i2]):
                result.append({
                    'type': 'delete',
                    'old_line': i1 + idx + 1,
                    'new_line': None,
                    'content': line,
                })
        elif op == 'replace':
            for idx, line in enumerate(old_lines[i1:i2]):
                result.append({
                    'type': 'delete',
                    'old_line': i1 + idx + 1,
                    'new_line': None,
                    'content': line,
                })
            for idx, line in enumerate(new_lines[j1:j2]):
                result.append({
                    'type': 'insert',
                    'old_line': None,
                    'new_line': j1 + idx + 1,
                    'content': line,
                })

    return result


def get_side_by_side(old_content: str, new_content: str, width: int = 80) -> str:
    """Generate side-by-side diff display.

    Args:
        old_content: Original content
        new_content: New content
        width: Width for each side

    Returns:
        Side-by-side formatted diff
    """
    old_lines = old_content.splitlines()
    new_lines = new_content.splitlines()

    differ = difflib.Differ()
    diff = list(differ.compare(old_lines, new_lines))

    half_width = width // 2 - 2
    result = []
    result.append(f"{'Old':<{half_width}} | {'New':<{half_width}}")
    result.append('-' * width)

    left = []
    right = []

    for line in diff:
        if line.startswith('  '):
            # Unchanged
            left.append(line[2:])
            right.append(line[2:])
        elif line.startswith('- '):
            # Removed
            left.append(line[2:])
        elif line.startswith('+ '):
            # Added
            right.append(line[2:])
        elif line.startswith('? '):
            # Diff marker - skip
            continue

    # Align and format
    max_lines = max(len(left), len(right))
    for i in range(max_lines):
        l = left[i][:half_width] if i < len(left) else ''
        r = right[i][:half_width] if i < len(right) else ''
        result.append(f"{l:<{half_width}} | {r:<{half_width}}")

    return '\n'.join(result)
