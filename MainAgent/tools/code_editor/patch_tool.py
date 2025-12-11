"""
Patch Tool - Apply unified diff patches to content.

Supports:
- Unified diff format
- Context-aware patch application
- Fuzzy matching for approximate patches
"""

from __future__ import annotations

import re
from typing import Optional, Tuple, List
from dataclasses import dataclass


@dataclass
class PatchHunk:
    """Represents a single hunk in a patch."""
    old_start: int
    old_count: int
    new_start: int
    new_count: int
    lines: List[str]


def parse_unified_diff(patch: str) -> List[PatchHunk]:
    """Parse a unified diff format patch into hunks.

    Args:
        patch: The unified diff text

    Returns:
        List of PatchHunk objects
    """
    hunks = []
    current_hunk = None

    lines = patch.split('\n')

    for line in lines:
        # Match hunk header: @@ -start,count +start,count @@
        hunk_match = re.match(r'^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@', line)

        if hunk_match:
            if current_hunk:
                hunks.append(current_hunk)

            old_start = int(hunk_match.group(1))
            old_count = int(hunk_match.group(2)) if hunk_match.group(2) else 1
            new_start = int(hunk_match.group(3))
            new_count = int(hunk_match.group(4)) if hunk_match.group(4) else 1

            current_hunk = PatchHunk(
                old_start=old_start,
                old_count=old_count,
                new_start=new_start,
                new_count=new_count,
                lines=[]
            )
        elif current_hunk is not None:
            # Skip --- and +++ headers
            if line.startswith('---') or line.startswith('+++'):
                continue
            # Add content lines (context, additions, deletions)
            if line.startswith(' ') or line.startswith('+') or line.startswith('-') or line == '':
                current_hunk.lines.append(line)

    if current_hunk:
        hunks.append(current_hunk)

    return hunks


def apply_hunk(content_lines: List[str], hunk: PatchHunk) -> Tuple[bool, List[str]]:
    """Apply a single hunk to content lines.

    Args:
        content_lines: List of lines from the original content
        hunk: The PatchHunk to apply

    Returns:
        Tuple of (success, modified_lines)
    """
    # Convert to 0-indexed
    start_line = hunk.old_start - 1

    # Verify context lines match (if we can)
    expected_context = []
    for line in hunk.lines:
        if line.startswith(' ') or line.startswith('-'):
            expected_context.append(line[1:] if len(line) > 1 else '')

    # Check if we can find a match at the expected location
    actual_content = content_lines[start_line:start_line + len(expected_context)]

    # Try exact match first
    match_found = False
    match_offset = 0

    if len(actual_content) == len(expected_context):
        if all(a.rstrip() == e.rstrip() for a, e in zip(actual_content, expected_context)):
            match_found = True

    # If no exact match, try fuzzy search nearby
    if not match_found:
        for offset in range(-5, 6):  # Search +-5 lines
            test_start = start_line + offset
            if test_start < 0 or test_start + len(expected_context) > len(content_lines):
                continue

            test_content = content_lines[test_start:test_start + len(expected_context)]
            if len(test_content) == len(expected_context):
                if all(a.rstrip() == e.rstrip() for a, e in zip(test_content, expected_context)):
                    match_found = True
                    match_offset = offset
                    break

    if not match_found:
        return False, content_lines

    # Apply the hunk
    adjusted_start = start_line + match_offset
    result_lines = content_lines[:adjusted_start]

    for line in hunk.lines:
        if line.startswith(' '):
            # Context line - keep it
            result_lines.append(line[1:] if len(line) > 1 else '')
        elif line.startswith('+'):
            # Addition - add it
            result_lines.append(line[1:] if len(line) > 1 else '')
        elif line.startswith('-'):
            # Deletion - skip it (it's being removed)
            pass

    # Add remaining lines after the hunk
    result_lines.extend(content_lines[adjusted_start + len(expected_context):])

    return True, result_lines


def apply_patch(content: str, patch: str) -> str:
    """Apply a unified diff patch to content.

    Args:
        content: The original content string
        patch: The unified diff patch string

    Returns:
        The patched content string
    """
    if not patch or not patch.strip():
        return content

    # Try to detect if this is a unified diff
    if '@@ ' not in patch:
        # Not a unified diff, try simple old/new replacement
        return apply_simple_patch(content, patch)

    hunks = parse_unified_diff(patch)

    if not hunks:
        return content

    content_lines = content.split('\n')

    # Apply hunks in reverse order to maintain line numbers
    for hunk in reversed(hunks):
        success, content_lines = apply_hunk(content_lines, hunk)
        if not success:
            print(f"[WARN] Failed to apply hunk at line {hunk.old_start}")

    return '\n'.join(content_lines)


def apply_simple_patch(content: str, patch: str) -> str:
    """Apply a simple old/new style patch.

    Supports formats like:
    - OLD: ... NEW: ...
    - REPLACE: ... WITH: ...
    - FIND: ... REPLACE: ...
    """
    # Try various simple patch formats
    patterns = [
        # OLD/NEW format
        (r'OLD:\s*\n?([\s\S]*?)\nNEW:\s*\n?([\s\S]*?)(?:\n(?:OLD:|$)|$)', 'OLD'),
        # FIND/REPLACE format
        (r'FIND:\s*\n?([\s\S]*?)\nREPLACE:\s*\n?([\s\S]*?)(?:\n(?:FIND:|$)|$)', 'FIND'),
        # REPLACE/WITH format
        (r'REPLACE:\s*\n?([\s\S]*?)\nWITH:\s*\n?([\s\S]*?)(?:\n(?:REPLACE:|$)|$)', 'REPLACE'),
    ]

    modified = content

    for pattern, _ in patterns:
        matches = re.findall(pattern, patch, re.IGNORECASE | re.MULTILINE)
        for old_code, new_code in matches:
            old_code = old_code.strip()
            new_code = new_code.strip()

            if old_code and old_code in modified:
                modified = modified.replace(old_code, new_code, 1)

    return modified


def create_unified_diff(original: str, modified: str, filename: str = "file") -> str:
    """Create a unified diff between two strings.

    Args:
        original: The original content
        modified: The modified content
        filename: The filename to use in the diff header

    Returns:
        A unified diff string
    """
    import difflib

    original_lines = original.splitlines(keepends=True)
    modified_lines = modified.splitlines(keepends=True)

    diff = difflib.unified_diff(
        original_lines,
        modified_lines,
        fromfile=f"a/{filename}",
        tofile=f"b/{filename}",
        lineterm=''
    )

    return ''.join(diff)


def validate_patch(patch: str) -> Tuple[bool, str]:
    """Validate that a patch string is well-formed.

    Args:
        patch: The patch string to validate

    Returns:
        Tuple of (is_valid, error_message)
    """
    if not patch or not patch.strip():
        return False, "Empty patch"

    # Check for unified diff format
    if '@@ ' in patch:
        hunks = parse_unified_diff(patch)
        if not hunks:
            return False, "No valid hunks found in unified diff"

        for i, hunk in enumerate(hunks):
            if not hunk.lines:
                return False, f"Hunk {i+1} has no content lines"

        return True, ""

    # Check for simple patch format
    simple_markers = ['OLD:', 'NEW:', 'FIND:', 'REPLACE:', 'WITH:']
    has_markers = any(marker in patch.upper() for marker in simple_markers)

    if has_markers:
        return True, ""

    return False, "Unrecognized patch format"
