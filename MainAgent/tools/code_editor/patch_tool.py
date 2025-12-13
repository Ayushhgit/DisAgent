"""
Patch Tool - Three-stage patching pipeline for safe code editing.

Pipeline:
1. Stage 1: Exact match replacement
2. Stage 2: Fuzzy/semantic context-based matching
3. Stage 3: Fallback to proposed diff if automated application fails

Supports:
- Unified diff format
- Simple OLD/NEW replacement
- AST-aware matching for Python files
- Similarity-based fuzzy matching
"""

from __future__ import annotations

import re
import ast
import difflib
import logging
from typing import Optional, Tuple, List, Dict, Any
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class PatchStage(Enum):
    """Stages of the patching pipeline."""
    EXACT_MATCH = "exact_match"
    FUZZY_MATCH = "fuzzy_match"
    SEMANTIC_MATCH = "semantic_match"
    FALLBACK_DIFF = "fallback_diff"
    FAILED = "failed"


@dataclass
class PatchResult:
    """Result of a patch operation."""
    success: bool
    stage: PatchStage
    content: str
    original_content: str
    message: str = ""
    diff: str = ""
    similarity_score: float = 1.0
    matched_at_line: int = -1


@dataclass
class PatchHunk:
    """Represents a single hunk in a patch."""
    old_start: int
    old_count: int
    new_start: int
    new_count: int
    lines: List[str]


@dataclass
class SemanticMatch:
    """Result of semantic/AST-based matching."""
    found: bool
    start_line: int = 0
    end_line: int = 0
    matched_code: str = ""
    similarity: float = 0.0
    node_type: str = ""


def parse_unified_diff(patch: str) -> List[PatchHunk]:
    """Parse a unified diff format patch into hunks."""
    hunks = []
    current_hunk = None

    lines = patch.split('\n')

    for line in lines:
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
            if line.startswith('---') or line.startswith('+++'):
                continue
            if line.startswith(' ') or line.startswith('+') or line.startswith('-') or line == '':
                current_hunk.lines.append(line)

    if current_hunk:
        hunks.append(current_hunk)

    return hunks


def normalize_whitespace(s: str) -> str:
    """Normalize whitespace for comparison."""
    return ' '.join(s.split())


def calculate_similarity(s1: str, s2: str) -> float:
    """Calculate similarity ratio between two strings."""
    if not s1 and not s2:
        return 1.0
    if not s1 or not s2:
        return 0.0
    return difflib.SequenceMatcher(None, s1, s2).ratio()


def find_best_match_location(
    content_lines: List[str],
    target_lines: List[str],
    start_hint: int = 0,
    search_range: int = 50,
    min_similarity: float = 0.8
) -> Tuple[int, float]:
    """
    Find the best matching location for target lines in content.

    Returns:
        Tuple of (best_line_index, similarity_score)
    """
    if not target_lines:
        return -1, 0.0

    target_text = '\n'.join(line.rstrip() for line in target_lines)
    target_normalized = normalize_whitespace(target_text)
    best_idx = -1
    best_similarity = 0.0

    # Search around the hint location
    search_start = max(0, start_hint - search_range)
    search_end = min(len(content_lines), start_hint + search_range + len(target_lines))

    for i in range(search_start, search_end - len(target_lines) + 1):
        candidate_lines = content_lines[i:i + len(target_lines)]
        candidate_text = '\n'.join(line.rstrip() for line in candidate_lines)

        # Try exact match first
        if candidate_text == target_text:
            return i, 1.0

        # Try normalized match
        candidate_normalized = normalize_whitespace(candidate_text)
        if candidate_normalized == target_normalized:
            return i, 0.99

        # Calculate similarity
        similarity = calculate_similarity(candidate_text, target_text)
        if similarity > best_similarity:
            best_similarity = similarity
            best_idx = i

    if best_similarity >= min_similarity:
        return best_idx, best_similarity

    return -1, best_similarity


def try_ast_match_python(
    content: str,
    target_code: str,
    target_type: Optional[str] = None
) -> SemanticMatch:
    """
    Try to find target code using Python AST matching.

    Args:
        content: Full file content
        target_code: Code snippet to find
        target_type: Optional AST node type hint ('function', 'class', 'method')

    Returns:
        SemanticMatch with location if found
    """
    try:
        content_tree = ast.parse(content)
    except SyntaxError:
        return SemanticMatch(found=False)

    # Try to parse target to get its structure
    try:
        target_tree = ast.parse(target_code)
        if not target_tree.body:
            return SemanticMatch(found=False)
        target_node = target_tree.body[0]
    except SyntaxError:
        # Target might be a partial snippet, try to identify key elements
        return _fuzzy_ast_search(content, content_tree, target_code)

    content_lines = content.split('\n')

    # Search for matching node in content tree
    for node in ast.walk(content_tree):
        if type(node) == type(target_node):
            # Check if this node matches
            match_score = _compare_ast_nodes(node, target_node)
            if match_score > 0.8:
                start_line = getattr(node, 'lineno', 1) - 1
                end_line = getattr(node, 'end_lineno', start_line + 1)

                matched_code = '\n'.join(content_lines[start_line:end_line])

                return SemanticMatch(
                    found=True,
                    start_line=start_line,
                    end_line=end_line,
                    matched_code=matched_code,
                    similarity=match_score,
                    node_type=type(node).__name__
                )

    return SemanticMatch(found=False)


def _fuzzy_ast_search(content: str, tree: ast.AST, target_code: str) -> SemanticMatch:
    """Fuzzy search using AST structure hints."""
    content_lines = content.split('\n')

    # Extract identifiers from target
    identifiers = set(re.findall(r'\b([a-zA-Z_][a-zA-Z0-9_]*)\b', target_code))

    # Look for functions/classes with matching names
    for node in ast.walk(tree):
        node_name = getattr(node, 'name', None)
        if node_name and node_name in identifiers:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                start_line = node.lineno - 1
                end_line = getattr(node, 'end_lineno', start_line + 10)
                matched_code = '\n'.join(content_lines[start_line:end_line])

                # Check similarity
                similarity = calculate_similarity(
                    normalize_whitespace(target_code),
                    normalize_whitespace(matched_code)
                )

                if similarity > 0.5:
                    return SemanticMatch(
                        found=True,
                        start_line=start_line,
                        end_line=end_line,
                        matched_code=matched_code,
                        similarity=similarity,
                        node_type=type(node).__name__
                    )

    return SemanticMatch(found=False)


def _compare_ast_nodes(node1: ast.AST, node2: ast.AST) -> float:
    """Compare two AST nodes for similarity."""
    if type(node1) != type(node2):
        return 0.0

    score = 0.5  # Base score for same type

    # Compare names for named nodes
    name1 = getattr(node1, 'name', None)
    name2 = getattr(node2, 'name', None)
    if name1 and name2:
        if name1 == name2:
            score += 0.3
        else:
            score += 0.1 * calculate_similarity(name1, name2)

    # Compare function arguments
    if isinstance(node1, (ast.FunctionDef, ast.AsyncFunctionDef)):
        args1 = [arg.arg for arg in node1.args.args]
        args2 = [arg.arg for arg in node2.args.args]
        if args1 == args2:
            score += 0.2
        elif len(args1) == len(args2):
            score += 0.1

    return min(score, 1.0)


def apply_hunk_exact(
    content_lines: List[str],
    hunk: PatchHunk
) -> Tuple[bool, List[str], int]:
    """
    Stage 1: Apply hunk with exact matching.

    Returns:
        Tuple of (success, modified_lines, matched_at_line)
    """
    start_line = hunk.old_start - 1

    expected_context = []
    for line in hunk.lines:
        if line.startswith(' ') or line.startswith('-'):
            expected_context.append(line[1:] if len(line) > 1 else '')

    if start_line < 0 or start_line + len(expected_context) > len(content_lines):
        return False, content_lines, -1

    actual_content = content_lines[start_line:start_line + len(expected_context)]

    # Exact match check
    if len(actual_content) == len(expected_context):
        if all(a.rstrip() == e.rstrip() for a, e in zip(actual_content, expected_context)):
            result_lines = _apply_hunk_at(content_lines, hunk, start_line)
            return True, result_lines, start_line

    return False, content_lines, -1


def apply_hunk_fuzzy(
    content_lines: List[str],
    hunk: PatchHunk,
    search_range: int = 20,
    min_similarity: float = 0.85
) -> Tuple[bool, List[str], int, float]:
    """
    Stage 2: Apply hunk with fuzzy matching.

    Returns:
        Tuple of (success, modified_lines, matched_at_line, similarity)
    """
    expected_context = []
    for line in hunk.lines:
        if line.startswith(' ') or line.startswith('-'):
            expected_context.append(line[1:] if len(line) > 1 else '')

    if not expected_context:
        return False, content_lines, -1, 0.0

    start_hint = hunk.old_start - 1
    best_idx, similarity = find_best_match_location(
        content_lines,
        expected_context,
        start_hint=start_hint,
        search_range=search_range,
        min_similarity=min_similarity
    )

    if best_idx >= 0:
        result_lines = _apply_hunk_at(content_lines, hunk, best_idx)
        return True, result_lines, best_idx, similarity

    return False, content_lines, -1, similarity


def _apply_hunk_at(
    content_lines: List[str],
    hunk: PatchHunk,
    start_idx: int
) -> List[str]:
    """Apply a hunk at a specific location."""
    expected_context = []
    for line in hunk.lines:
        if line.startswith(' ') or line.startswith('-'):
            expected_context.append(line[1:] if len(line) > 1 else '')

    result_lines = content_lines[:start_idx]

    for line in hunk.lines:
        if line.startswith(' '):
            result_lines.append(line[1:] if len(line) > 1 else '')
        elif line.startswith('+'):
            result_lines.append(line[1:] if len(line) > 1 else '')
        # '-' lines are skipped (deleted)

    result_lines.extend(content_lines[start_idx + len(expected_context):])
    return result_lines


def apply_patch_pipeline(
    content: str,
    patch: str,
    filename: str = "",
    enable_ast: bool = True
) -> PatchResult:
    """
    Apply patch using the three-stage pipeline.

    Stage 1: Exact match
    Stage 2: Fuzzy/semantic match
    Stage 3: Fallback to proposed diff

    Args:
        content: Original file content
        patch: Patch to apply
        filename: Filename for context (enables AST for .py files)
        enable_ast: Whether to try AST-based matching

    Returns:
        PatchResult with success status and patched content
    """
    original_content = content

    if not patch or not patch.strip():
        return PatchResult(
            success=True,
            stage=PatchStage.EXACT_MATCH,
            content=content,
            original_content=original_content,
            message="Empty patch, no changes needed"
        )

    # Detect patch format
    if '@@ ' not in patch:
        return _apply_simple_patch_pipeline(content, patch, original_content)

    hunks = parse_unified_diff(patch)
    if not hunks:
        return PatchResult(
            success=False,
            stage=PatchStage.FAILED,
            content=content,
            original_content=original_content,
            message="No valid hunks found in patch"
        )

    content_lines = content.split('\n')
    all_succeeded = True
    any_succeeded = False
    stages_used = []
    failed_hunks = []

    # Apply hunks in reverse order to maintain line numbers
    for i, hunk in enumerate(reversed(hunks)):
        hunk_idx = len(hunks) - 1 - i

        # Stage 1: Exact match
        success, result_lines, match_line = apply_hunk_exact(content_lines, hunk)
        if success:
            content_lines = result_lines
            any_succeeded = True
            stages_used.append((hunk_idx, PatchStage.EXACT_MATCH, 1.0))
            continue

        # Stage 2: Fuzzy match
        success, result_lines, match_line, similarity = apply_hunk_fuzzy(
            content_lines, hunk, search_range=30, min_similarity=0.80
        )
        if success:
            content_lines = result_lines
            any_succeeded = True
            stages_used.append((hunk_idx, PatchStage.FUZZY_MATCH, similarity))
            logger.info(f"Hunk {hunk_idx} applied via fuzzy match (similarity: {similarity:.2f})")
            continue

        # Stage 2b: Try AST-based matching for Python files
        if enable_ast and filename.endswith('.py'):
            expected_lines = []
            for line in hunk.lines:
                if line.startswith(' ') or line.startswith('-'):
                    expected_lines.append(line[1:] if len(line) > 1 else '')

            target_code = '\n'.join(expected_lines)
            ast_match = try_ast_match_python('\n'.join(content_lines), target_code)

            if ast_match.found and ast_match.similarity > 0.7:
                # Apply at AST-matched location
                success, result_lines, _, sim = apply_hunk_fuzzy(
                    content_lines, hunk,
                    search_range=5,
                    min_similarity=0.6
                )
                if success:
                    content_lines = result_lines
                    any_succeeded = True
                    stages_used.append((hunk_idx, PatchStage.SEMANTIC_MATCH, ast_match.similarity))
                    logger.info(f"Hunk {hunk_idx} applied via AST match ({ast_match.node_type})")
                    continue

        # Stage 3: Failed - record for diff proposal
        all_succeeded = False
        failed_hunks.append(hunk)
        stages_used.append((hunk_idx, PatchStage.FAILED, 0.0))
        logger.warning(f"Failed to apply hunk {hunk_idx} at line {hunk.old_start}")

    result_content = '\n'.join(content_lines)

    if all_succeeded:
        stage = max((s for _, s, _ in stages_used), key=lambda x: x.value)
        return PatchResult(
            success=True,
            stage=stage,
            content=result_content,
            original_content=original_content,
            message=f"All {len(hunks)} hunks applied successfully"
        )
    elif any_succeeded:
        # Partial success - generate diff for failed hunks
        diff = create_unified_diff(original_content, result_content, filename or "file")
        return PatchResult(
            success=True,
            stage=PatchStage.FALLBACK_DIFF,
            content=result_content,
            original_content=original_content,
            message=f"Partial success: {len(hunks) - len(failed_hunks)}/{len(hunks)} hunks applied",
            diff=diff
        )
    else:
        # Complete failure - return proposed diff
        diff = patch  # The original patch becomes the proposal
        return PatchResult(
            success=False,
            stage=PatchStage.FALLBACK_DIFF,
            content=original_content,
            original_content=original_content,
            message="Could not apply patch. Proposed diff attached.",
            diff=diff
        )


def _apply_simple_patch_pipeline(
    content: str,
    patch: str,
    original_content: str
) -> PatchResult:
    """Apply simple OLD/NEW style patch with pipeline."""
    patterns = [
        (r'OLD:\s*\n?([\s\S]*?)\nNEW:\s*\n?([\s\S]*?)(?:\n(?:OLD:|$)|$)', 'OLD'),
        (r'FIND:\s*\n?([\s\S]*?)\nREPLACE:\s*\n?([\s\S]*?)(?:\n(?:FIND:|$)|$)', 'FIND'),
        (r'REPLACE:\s*\n?([\s\S]*?)\nWITH:\s*\n?([\s\S]*?)(?:\n(?:REPLACE:|$)|$)', 'REPLACE'),
    ]

    modified = content
    changes_made = 0
    failed_changes = []

    for pattern, fmt in patterns:
        matches = re.findall(pattern, patch, re.IGNORECASE | re.MULTILINE)
        for old_code, new_code in matches:
            old_code = old_code.strip()
            new_code = new_code.strip()

            if not old_code:
                continue

            # Stage 1: Exact match
            if old_code in modified:
                modified = modified.replace(old_code, new_code, 1)
                changes_made += 1
                continue

            # Stage 2: Fuzzy match (normalize whitespace)
            normalized_old = normalize_whitespace(old_code)
            lines = modified.split('\n')
            found = False

            for i in range(len(lines)):
                for j in range(i + 1, min(i + 20, len(lines) + 1)):
                    candidate = '\n'.join(lines[i:j])
                    if normalize_whitespace(candidate) == normalized_old:
                        modified = modified.replace(candidate, new_code, 1)
                        changes_made += 1
                        found = True
                        break
                if found:
                    break

            if not found:
                # Stage 2b: Similarity-based match
                best_match = None
                best_similarity = 0.0

                for i in range(len(lines)):
                    for j in range(i + 1, min(i + 20, len(lines) + 1)):
                        candidate = '\n'.join(lines[i:j])
                        similarity = calculate_similarity(old_code, candidate)
                        if similarity > best_similarity and similarity > 0.85:
                            best_similarity = similarity
                            best_match = candidate

                if best_match:
                    modified = modified.replace(best_match, new_code, 1)
                    changes_made += 1
                    logger.info(f"Applied fuzzy match with similarity {best_similarity:.2f}")
                else:
                    failed_changes.append(old_code[:50] + "..." if len(old_code) > 50 else old_code)

    if changes_made > 0 and not failed_changes:
        return PatchResult(
            success=True,
            stage=PatchStage.EXACT_MATCH if changes_made == len(re.findall(r'(OLD:|FIND:|REPLACE:)', patch, re.I)) else PatchStage.FUZZY_MATCH,
            content=modified,
            original_content=original_content,
            message=f"Applied {changes_made} changes"
        )
    elif changes_made > 0:
        return PatchResult(
            success=True,
            stage=PatchStage.FALLBACK_DIFF,
            content=modified,
            original_content=original_content,
            message=f"Partial: {changes_made} applied, {len(failed_changes)} failed",
            diff=f"Failed to match:\n" + "\n".join(failed_changes)
        )
    else:
        return PatchResult(
            success=False,
            stage=PatchStage.FAILED,
            content=original_content,
            original_content=original_content,
            message="No matches found for patch"
        )


# Legacy API compatibility
def apply_patch(content: str, patch: str) -> str:
    """Apply a patch (legacy API, returns string only)."""
    result = apply_patch_pipeline(content, patch)
    return result.content


def apply_simple_patch(content: str, patch: str) -> str:
    """Apply a simple patch (legacy API)."""
    result = _apply_simple_patch_pipeline(content, patch, content)
    return result.content


def apply_hunk(content_lines: List[str], hunk: PatchHunk) -> Tuple[bool, List[str]]:
    """Apply a hunk (legacy API)."""
    success, lines, _ = apply_hunk_exact(content_lines, hunk)
    if success:
        return True, lines

    success, lines, _, _ = apply_hunk_fuzzy(content_lines, hunk)
    return success, lines


def create_unified_diff(original: str, modified: str, filename: str = "file") -> str:
    """Create a unified diff between two strings."""
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
    """Validate that a patch string is well-formed."""
    if not patch or not patch.strip():
        return False, "Empty patch"

    if '@@ ' in patch:
        hunks = parse_unified_diff(patch)
        if not hunks:
            return False, "No valid hunks found in unified diff"

        for i, hunk in enumerate(hunks):
            if not hunk.lines:
                return False, f"Hunk {i+1} has no content lines"

        return True, ""

    simple_markers = ['OLD:', 'NEW:', 'FIND:', 'REPLACE:', 'WITH:']
    has_markers = any(marker in patch.upper() for marker in simple_markers)

    if has_markers:
        return True, ""

    return False, "Unrecognized patch format"
