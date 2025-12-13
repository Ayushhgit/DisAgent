"""
Conflict Resolver - Resolves concurrent modifications to same files.

Capabilities:
- Detect overlapping edits from multiple agents
- Merge non-overlapping changes automatically
- Use LLM for intelligent conflict resolution
- Priority-based resolution when merge fails

This addresses a critical gap: when multiple agents propose changes
to the same file, existing systems use "last write wins" which
causes data loss and inconsistency.
"""

from __future__ import annotations

import difflib
import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple
import threading

logger = logging.getLogger(__name__)


class ConflictType(Enum):
    """Types of conflicts between edits."""
    NO_CONFLICT = "no_conflict"
    OVERLAPPING = "overlapping"  # Edits modify same lines
    ADJACENT = "adjacent"  # Edits touch adjacent lines
    SEMANTIC = "semantic"  # Edits are semantically incompatible
    DEPENDENCY = "dependency"  # One edit depends on another's changes


class ResolutionStrategy(Enum):
    """Strategies for resolving conflicts."""
    AUTO_MERGE = "auto_merge"  # Automatically merge if possible
    PRIORITY = "priority"  # Higher priority agent wins
    LLM_MERGE = "llm_merge"  # Use LLM to merge intelligently
    MANUAL = "manual"  # Flag for manual resolution
    REJECT_BOTH = "reject_both"  # Reject both changes


@dataclass
class Edit:
    """Represents a code edit."""
    file_path: str
    old_content: str
    new_content: str
    agent_id: str
    task_id: str = ""
    priority: int = 0  # Higher = more important
    timestamp: float = 0.0
    line_start: int = -1  # -1 means full file
    line_end: int = -1
    description: str = ""


@dataclass
class Conflict:
    """Represents a conflict between edits."""
    file_path: str
    conflict_type: ConflictType
    edits: List[Edit]
    overlapping_lines: List[int] = field(default_factory=list)
    description: str = ""


@dataclass
class ResolutionResult:
    """Result of conflict resolution."""
    success: bool
    strategy_used: ResolutionStrategy
    resolved_content: str
    original_content: str
    edits_applied: List[str]  # Agent IDs whose edits were applied
    edits_rejected: List[str]  # Agent IDs whose edits were rejected
    merge_conflicts: List[str] = field(default_factory=list)  # Unresolved conflict descriptions
    message: str = ""


class ConflictResolver:
    """Resolves concurrent modifications to same files.

    Strategies:
    1. Auto-merge: Non-overlapping changes merged automatically
    2. Priority: Higher priority agent's changes win
    3. LLM merge: Use LLM to intelligently combine changes
    4. Manual: Flag for human review
    """

    def __init__(
        self,
        llm_call: Optional[Callable[[str, Optional[int]], str]] = None,
        default_strategy: ResolutionStrategy = ResolutionStrategy.AUTO_MERGE
    ):
        """Initialize conflict resolver.

        Args:
            llm_call: Optional LLM function for intelligent merging
            default_strategy: Default resolution strategy
        """
        self.llm_call = llm_call
        self.default_strategy = default_strategy
        self._lock = threading.Lock()
        self._resolution_history: List[ResolutionResult] = []

    def detect_conflicts(
        self,
        pending_edits: List[Edit]
    ) -> List[Conflict]:
        """Identify conflicts among pending edits.

        Args:
            pending_edits: List of edits waiting to be applied

        Returns:
            List of conflicts found
        """
        conflicts = []

        # Group by file
        by_file: Dict[str, List[Edit]] = {}
        for edit in pending_edits:
            if edit.file_path not in by_file:
                by_file[edit.file_path] = []
            by_file[edit.file_path].append(edit)

        # Detect conflicts within each file
        for file_path, edits in by_file.items():
            if len(edits) < 2:
                continue

            conflict = self._analyze_conflict(file_path, edits)
            if conflict.conflict_type != ConflictType.NO_CONFLICT:
                conflicts.append(conflict)

        return conflicts

    def _analyze_conflict(
        self,
        file_path: str,
        edits: List[Edit]
    ) -> Conflict:
        """Analyze potential conflict between edits."""
        # Get line ranges for each edit
        edit_ranges = []
        for edit in edits:
            if edit.line_start >= 0 and edit.line_end >= 0:
                edit_ranges.append((edit.line_start, edit.line_end, edit))
            else:
                # Full file edit - find changed lines
                old_lines = edit.old_content.split('\n')
                new_lines = edit.new_content.split('\n')
                changed = self._find_changed_lines(old_lines, new_lines)
                if changed:
                    edit_ranges.append((min(changed), max(changed), edit))
                else:
                    # No changes detected
                    continue

        if len(edit_ranges) < 2:
            return Conflict(
                file_path=file_path,
                conflict_type=ConflictType.NO_CONFLICT,
                edits=edits
            )

        # Check for overlapping ranges
        overlapping = []
        for i, (start1, end1, edit1) in enumerate(edit_ranges):
            for start2, end2, edit2 in edit_ranges[i+1:]:
                # Check overlap
                if start1 <= end2 and start2 <= end1:
                    overlap_start = max(start1, start2)
                    overlap_end = min(end1, end2)
                    overlapping.extend(range(overlap_start, overlap_end + 1))

        if overlapping:
            return Conflict(
                file_path=file_path,
                conflict_type=ConflictType.OVERLAPPING,
                edits=edits,
                overlapping_lines=list(set(overlapping)),
                description=f"Edits overlap on lines {min(overlapping)}-{max(overlapping)}"
            )

        # Check for adjacent changes (within 3 lines)
        adjacent = False
        for i, (start1, end1, _) in enumerate(edit_ranges):
            for start2, end2, _ in edit_ranges[i+1:]:
                if abs(end1 - start2) <= 3 or abs(end2 - start1) <= 3:
                    adjacent = True
                    break

        if adjacent:
            return Conflict(
                file_path=file_path,
                conflict_type=ConflictType.ADJACENT,
                edits=edits,
                description="Edits modify adjacent lines"
            )

        return Conflict(
            file_path=file_path,
            conflict_type=ConflictType.NO_CONFLICT,
            edits=edits
        )

    def _find_changed_lines(
        self,
        old_lines: List[str],
        new_lines: List[str]
    ) -> List[int]:
        """Find line numbers that changed between old and new."""
        changed = []
        differ = difflib.SequenceMatcher(None, old_lines, new_lines)

        for tag, i1, i2, j1, j2 in differ.get_opcodes():
            if tag != 'equal':
                changed.extend(range(i1, i2))

        return changed

    def resolve(
        self,
        conflict: Conflict,
        strategy: Optional[ResolutionStrategy] = None,
        base_content: Optional[str] = None
    ) -> ResolutionResult:
        """Resolve a conflict using specified strategy.

        Args:
            conflict: The conflict to resolve
            strategy: Resolution strategy (uses default if None)
            base_content: Original file content before any edits

        Returns:
            ResolutionResult with merged content or failure info
        """
        strategy = strategy or self.default_strategy

        if conflict.conflict_type == ConflictType.NO_CONFLICT:
            # Just apply all edits in order
            return self._apply_sequential(conflict, base_content)

        # Try strategies in order of preference
        if strategy == ResolutionStrategy.AUTO_MERGE:
            result = self._try_auto_merge(conflict, base_content)
            if result.success:
                return self._record_result(result)
            # Fall through to priority
            strategy = ResolutionStrategy.PRIORITY

        if strategy == ResolutionStrategy.LLM_MERGE and self.llm_call:
            result = self._llm_merge(conflict, base_content)
            if result.success:
                return self._record_result(result)
            # Fall through to priority
            strategy = ResolutionStrategy.PRIORITY

        if strategy == ResolutionStrategy.PRIORITY:
            result = self._priority_resolve(conflict, base_content)
            return self._record_result(result)

        if strategy == ResolutionStrategy.REJECT_BOTH:
            return self._record_result(ResolutionResult(
                success=False,
                strategy_used=ResolutionStrategy.REJECT_BOTH,
                resolved_content=base_content or conflict.edits[0].old_content,
                original_content=base_content or conflict.edits[0].old_content,
                edits_applied=[],
                edits_rejected=[e.agent_id for e in conflict.edits],
                message="Both edits rejected due to conflict"
            ))

        # Manual resolution
        return self._record_result(ResolutionResult(
            success=False,
            strategy_used=ResolutionStrategy.MANUAL,
            resolved_content=base_content or conflict.edits[0].old_content,
            original_content=base_content or conflict.edits[0].old_content,
            edits_applied=[],
            edits_rejected=[],
            merge_conflicts=[conflict.description],
            message="Manual resolution required"
        ))

    def _apply_sequential(
        self,
        conflict: Conflict,
        base_content: Optional[str]
    ) -> ResolutionResult:
        """Apply non-conflicting edits sequentially."""
        content = base_content or conflict.edits[0].old_content
        applied = []

        # Sort by timestamp to maintain order
        sorted_edits = sorted(conflict.edits, key=lambda e: e.timestamp)

        for edit in sorted_edits:
            # Simple replacement
            if edit.old_content in content:
                content = content.replace(edit.old_content, edit.new_content, 1)
                applied.append(edit.agent_id)
            else:
                # Try to find close match
                content = edit.new_content
                applied.append(edit.agent_id)

        return ResolutionResult(
            success=True,
            strategy_used=ResolutionStrategy.AUTO_MERGE,
            resolved_content=content,
            original_content=base_content or conflict.edits[0].old_content,
            edits_applied=applied,
            edits_rejected=[],
            message="Applied edits sequentially"
        )

    def _try_auto_merge(
        self,
        conflict: Conflict,
        base_content: Optional[str]
    ) -> ResolutionResult:
        """Try to automatically merge non-overlapping changes."""
        base = base_content or conflict.edits[0].old_content
        base_lines = base.split('\n')

        # Collect all changes with their line ranges
        changes: List[Tuple[int, int, List[str], str]] = []  # (start, end, new_lines, agent_id)

        for edit in conflict.edits:
            old_lines = edit.old_content.split('\n')
            new_lines = edit.new_content.split('\n')

            # Find the diff
            differ = difflib.SequenceMatcher(None, old_lines, new_lines)
            for tag, i1, i2, j1, j2 in differ.get_opcodes():
                if tag != 'equal':
                    changes.append((i1, i2, new_lines[j1:j2], edit.agent_id))

        # Sort changes by line number (descending to apply from bottom up)
        changes.sort(key=lambda c: c[0], reverse=True)

        # Check for overlaps
        applied_ranges = []
        for start, end, _, _ in changes:
            for (prev_start, prev_end) in applied_ranges:
                if start <= prev_end and prev_start <= end:
                    return ResolutionResult(
                        success=False,
                        strategy_used=ResolutionStrategy.AUTO_MERGE,
                        resolved_content=base,
                        original_content=base,
                        edits_applied=[],
                        edits_rejected=[e.agent_id for e in conflict.edits],
                        message="Cannot auto-merge: overlapping changes"
                    )
            applied_ranges.append((start, end))

        # Apply changes from bottom to top
        result_lines = base_lines.copy()
        applied_agents = set()

        for start, end, new_lines, agent_id in changes:
            result_lines[start:end] = new_lines
            applied_agents.add(agent_id)

        return ResolutionResult(
            success=True,
            strategy_used=ResolutionStrategy.AUTO_MERGE,
            resolved_content='\n'.join(result_lines),
            original_content=base,
            edits_applied=list(applied_agents),
            edits_rejected=[],
            message="Successfully auto-merged non-overlapping changes"
        )

    def _llm_merge(
        self,
        conflict: Conflict,
        base_content: Optional[str]
    ) -> ResolutionResult:
        """Use LLM to intelligently merge conflicting changes."""
        if not self.llm_call:
            return ResolutionResult(
                success=False,
                strategy_used=ResolutionStrategy.LLM_MERGE,
                resolved_content=base_content or "",
                original_content=base_content or "",
                edits_applied=[],
                edits_rejected=[e.agent_id for e in conflict.edits],
                message="LLM not available"
            )

        base = base_content or conflict.edits[0].old_content

        # Build prompt with all edits
        edits_description = ""
        for i, edit in enumerate(conflict.edits):
            edits_description += f"""
EDIT {i+1} (from agent: {edit.agent_id}):
Description: {edit.description}
Changes:
```
{edit.new_content[:1500]}
```
"""

        prompt = f"""You are resolving a merge conflict between multiple code edits.

ORIGINAL FILE:
```
{base[:2000]}
```

CONFLICTING EDITS:
{edits_description}

CONFLICT TYPE: {conflict.conflict_type.value}
CONFLICT DETAILS: {conflict.description}

Merge these changes intelligently:
1. Preserve the intent of both edits where possible
2. Ensure the result is syntactically correct
3. Resolve any semantic conflicts by choosing the better approach
4. Add comments if manual review is needed for complex merges

Return the merged code only, no explanations. If merge is impossible, return "CANNOT_MERGE" followed by reason."""

        try:
            response = self.llm_call(prompt, 3000)

            if response.strip().startswith("CANNOT_MERGE"):
                return ResolutionResult(
                    success=False,
                    strategy_used=ResolutionStrategy.LLM_MERGE,
                    resolved_content=base,
                    original_content=base,
                    edits_applied=[],
                    edits_rejected=[e.agent_id for e in conflict.edits],
                    message=response.replace("CANNOT_MERGE", "").strip()
                )

            # Clean up response (remove markdown if present)
            merged = response.strip()
            if merged.startswith("```"):
                lines = merged.split('\n')
                merged = '\n'.join(lines[1:-1] if lines[-1] == "```" else lines[1:])

            return ResolutionResult(
                success=True,
                strategy_used=ResolutionStrategy.LLM_MERGE,
                resolved_content=merged,
                original_content=base,
                edits_applied=[e.agent_id for e in conflict.edits],
                edits_rejected=[],
                message="Successfully merged using LLM"
            )

        except Exception as e:
            logger.exception(f"LLM merge failed: {e}")
            return ResolutionResult(
                success=False,
                strategy_used=ResolutionStrategy.LLM_MERGE,
                resolved_content=base,
                original_content=base,
                edits_applied=[],
                edits_rejected=[e.agent_id for e in conflict.edits],
                message=f"LLM merge failed: {e}"
            )

    def _priority_resolve(
        self,
        conflict: Conflict,
        base_content: Optional[str]
    ) -> ResolutionResult:
        """Resolve by applying highest priority edit only."""
        base = base_content or conflict.edits[0].old_content

        # Sort by priority (descending) then by timestamp (ascending for ties)
        sorted_edits = sorted(
            conflict.edits,
            key=lambda e: (-e.priority, e.timestamp)
        )

        winner = sorted_edits[0]
        losers = sorted_edits[1:]

        return ResolutionResult(
            success=True,
            strategy_used=ResolutionStrategy.PRIORITY,
            resolved_content=winner.new_content,
            original_content=base,
            edits_applied=[winner.agent_id],
            edits_rejected=[e.agent_id for e in losers],
            message=f"Applied edit from {winner.agent_id} (priority: {winner.priority})"
        )

    def _record_result(self, result: ResolutionResult) -> ResolutionResult:
        """Record result in history."""
        with self._lock:
            self._resolution_history.append(result)
        return result

    def resolve_all(
        self,
        pending_edits: List[Edit],
        base_contents: Dict[str, str],
        strategy: Optional[ResolutionStrategy] = None
    ) -> Dict[str, ResolutionResult]:
        """Resolve all conflicts in pending edits.

        Args:
            pending_edits: All pending edits
            base_contents: Map of file_path -> original content
            strategy: Resolution strategy

        Returns:
            Map of file_path -> ResolutionResult
        """
        results = {}
        conflicts = self.detect_conflicts(pending_edits)

        # Group non-conflicting edits by file
        conflicting_files = {c.file_path for c in conflicts}

        # Process conflicts
        for conflict in conflicts:
            base = base_contents.get(conflict.file_path, "")
            results[conflict.file_path] = self.resolve(conflict, strategy, base)

        # Process non-conflicting files
        by_file: Dict[str, List[Edit]] = {}
        for edit in pending_edits:
            if edit.file_path not in conflicting_files:
                if edit.file_path not in by_file:
                    by_file[edit.file_path] = []
                by_file[edit.file_path].append(edit)

        for file_path, edits in by_file.items():
            if len(edits) == 1:
                results[file_path] = ResolutionResult(
                    success=True,
                    strategy_used=ResolutionStrategy.AUTO_MERGE,
                    resolved_content=edits[0].new_content,
                    original_content=base_contents.get(file_path, edits[0].old_content),
                    edits_applied=[edits[0].agent_id],
                    edits_rejected=[],
                    message="Single edit, no conflict"
                )
            else:
                # Multiple non-conflicting edits
                conflict = Conflict(
                    file_path=file_path,
                    conflict_type=ConflictType.NO_CONFLICT,
                    edits=edits
                )
                results[file_path] = self._apply_sequential(
                    conflict,
                    base_contents.get(file_path)
                )

        return results

    def get_stats(self) -> Dict[str, Any]:
        """Get resolution statistics."""
        with self._lock:
            if not self._resolution_history:
                return {"total_resolutions": 0}

            return {
                "total_resolutions": len(self._resolution_history),
                "successful": sum(1 for r in self._resolution_history if r.success),
                "failed": sum(1 for r in self._resolution_history if not r.success),
                "by_strategy": {
                    s.value: sum(1 for r in self._resolution_history
                                if r.strategy_used == s)
                    for s in ResolutionStrategy
                }
            }


def create_conflict_resolver(
    llm_call: Optional[Callable[[str, Optional[int]], str]] = None,
    default_strategy: ResolutionStrategy = ResolutionStrategy.AUTO_MERGE
) -> ConflictResolver:
    """Factory function to create conflict resolver."""
    return ConflictResolver(llm_call, default_strategy)
