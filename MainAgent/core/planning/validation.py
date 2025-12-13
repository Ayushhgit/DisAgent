# core/planning/validation.py
"""
Validation utilities for planning.

Capabilities:
- validate_tasks: ensure each task has id and description
- validate_dependencies: detect cycles and ensure dependencies reference known tasks
- detect_cycles: find and return all cycles in dependency graph
- topological_sort: return an ordering if DAG, else raise
"""

from typing import Dict, List, Set, Optional, Tuple
from dataclasses import dataclass
from collections import deque
import logging

logger = logging.getLogger(__name__)


class ValidationError(Exception):
    """Base validation error."""
    pass


class CycleDetectedError(ValidationError):
    """Raised when a dependency cycle is detected."""

    def __init__(self, cycles: List[List[str]], message: str = "Dependency cycle detected"):
        self.cycles = cycles
        self.message = message
        super().__init__(self._format_message())

    def _format_message(self) -> str:
        cycle_strs = [" -> ".join(c + [c[0]]) for c in self.cycles]
        return f"{self.message}: {'; '.join(cycle_strs)}"


@dataclass
class ValidationResult:
    """Result of validation with detailed information."""
    is_valid: bool
    errors: List[str]
    cycles: List[List[str]]
    warnings: List[str]


class PlanValidator:
    """Validates task plans and dependency graphs."""

    def validate_tasks(self, tasks: List[Dict]) -> bool:
        """
        Ensure tasks have 'id' and 'description' keys and ids are unique.
        """
        if not isinstance(tasks, list):
            logger.debug("Tasks is not a list")
            return False
        seen: Set[str] = set()
        for t in tasks:
            if not isinstance(t, dict):
                logger.debug("Task is not a dict: %s", t)
                return False
            if "id" not in t or not t["id"]:
                logger.debug("Task missing id: %s", t)
                return False
            if "description" not in t or not t["description"]:
                logger.debug("Task missing description: %s", t)
                return False
            if t["id"] in seen:
                logger.debug("Duplicate task id: %s", t["id"])
                return False
            seen.add(t["id"])
        return True

    def validate_dependencies(self, dep_map: Dict[str, List[str]]) -> bool:
        """
        Check dependencies reference existing nodes and detect cycles.
        dep_map: { task_id: [dep_id, ...], ... }
        """
        result = self.validate_dependencies_detailed(dep_map)
        return result.is_valid

    def validate_dependencies_detailed(self, dep_map: Dict[str, List[str]]) -> ValidationResult:
        """
        Validate dependencies with detailed error reporting.

        Args:
            dep_map: Mapping of task_id to list of dependency task_ids

        Returns:
            ValidationResult with is_valid, errors, cycles, and warnings
        """
        errors: List[str] = []
        warnings: List[str] = []
        cycles: List[List[str]] = []

        # Check that all dependencies reference existing nodes
        nodes = set(dep_map.keys())
        for task_id, deps in dep_map.items():
            for dep in deps:
                if dep not in nodes:
                    errors.append(
                        f"Task '{task_id}' depends on unknown task '{dep}'"
                    )

        # Detect cycles
        detected_cycles = self.detect_cycles(dep_map)
        if detected_cycles:
            cycles = detected_cycles
            for cycle in cycles:
                cycle_str = " -> ".join(cycle + [cycle[0]])
                errors.append(f"Dependency cycle: {cycle_str}")

        # Check for self-dependencies
        for task_id, deps in dep_map.items():
            if task_id in deps:
                errors.append(f"Task '{task_id}' depends on itself")

        is_valid = len(errors) == 0

        return ValidationResult(
            is_valid=is_valid,
            errors=errors,
            cycles=cycles,
            warnings=warnings,
        )

    def detect_cycles(self, dep_map: Dict[str, List[str]]) -> List[List[str]]:
        """
        Detect all cycles in the dependency graph using DFS.

        Args:
            dep_map: Mapping of task_id to list of dependency task_ids

        Returns:
            List of cycles, where each cycle is a list of task_ids
        """
        cycles: List[List[str]] = []
        visited: Set[str] = set()
        rec_stack: Set[str] = set()
        path: List[str] = []

        def dfs(node: str) -> None:
            """DFS to find cycles."""
            if node in rec_stack:
                # Found a cycle - extract it from path
                cycle_start = path.index(node)
                cycle = path[cycle_start:]
                # Avoid duplicate cycles
                normalized = self._normalize_cycle(cycle)
                if normalized not in [self._normalize_cycle(c) for c in cycles]:
                    cycles.append(cycle)
                return

            if node in visited:
                return

            visited.add(node)
            rec_stack.add(node)
            path.append(node)

            for dep in dep_map.get(node, []):
                dfs(dep)

            path.pop()
            rec_stack.remove(node)

        for node in dep_map:
            if node not in visited:
                dfs(node)

        return cycles

    def _normalize_cycle(self, cycle: List[str]) -> Tuple[str, ...]:
        """
        Normalize cycle for comparison (start from smallest element).
        """
        if not cycle:
            return tuple()
        min_idx = cycle.index(min(cycle))
        normalized = cycle[min_idx:] + cycle[:min_idx]
        return tuple(normalized)

    def topological_sort(self, dep_map: Dict[str, List[str]]) -> List[str]:
        """
        Return ordering of tasks if DAG, else raise CycleDetectedError.
        Uses Kahn's algorithm.
        """
        # First check for cycles
        cycles = self.detect_cycles(dep_map)
        if cycles:
            raise CycleDetectedError(cycles)

        # Build inbound counts
        nodes = set(dep_map.keys())
        inbound: Dict[str, int] = {n: 0 for n in nodes}
        for src, deps in dep_map.items():
            for d in deps:
                if d in nodes:  # Only count valid dependencies
                    inbound[src] += 1

        # Queue nodes with zero inbound (use deque for O(1) popleft)
        queue = deque([n for n, c in inbound.items() if c == 0])
        order: List[str] = []

        # Adjacency is reverse: edges from dep->src
        rev_adj: Dict[str, List[str]] = {n: [] for n in nodes}
        for src, deps in dep_map.items():
            for d in deps:
                if d in nodes:
                    rev_adj[d].append(src)

        while queue:
            n = queue.popleft()
            order.append(n)
            for neighbor in rev_adj.get(n, []):
                inbound[neighbor] -= 1
                if inbound[neighbor] == 0:
                    queue.append(neighbor)

        if len(order) != len(nodes):
            # This shouldn't happen if detect_cycles worked correctly
            remaining = nodes - set(order)
            logger.error(
                "Topological sort incomplete. Remaining nodes: %s", remaining
            )
            raise ValidationError(
                f"Dependency graph issue: {len(nodes) - len(order)} nodes unreachable"
            )

        return order

    def get_dependency_order(
        self, dep_map: Dict[str, List[str]]
    ) -> Tuple[bool, List[str], List[str]]:
        """
        Get execution order with error handling.

        Args:
            dep_map: Mapping of task_id to list of dependency task_ids

        Returns:
            Tuple of (success, order_or_empty, error_messages)
        """
        try:
            order = self.topological_sort(dep_map)
            return True, order, []
        except CycleDetectedError as e:
            return False, [], [str(e)]
        except ValidationError as e:
            return False, [], [str(e)]
