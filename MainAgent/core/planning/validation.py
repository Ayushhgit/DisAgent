# core/planning/validation.py
"""
Validation utilities for planning.

Capabilities:
- validate_tasks: ensure each task has id and description
- validate_dependencies: detect cycles and ensure dependencies reference known tasks
- topological_sort: return an ordering if DAG, else raise
"""

from typing import Dict, List, Set, Optional
import logging

logger = logging.getLogger(__name__)


class ValidationError(Exception):
    pass


class PlanValidator:
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
        Check dependencies reference existing nodes and detect cycles using DFS.
        dep_map: { task_id: [dep_id, ...], ... }
        """
        # ensure keys exist
        nodes = set(dep_map.keys())
        for k, deps in dep_map.items():
            for d in deps:
                if d not in nodes:
                    logger.debug("Dependency %s referenced by %s is not in node set", d, k)
                    return False

        try:
            _ = self.topological_sort(dep_map)
            return True
        except ValidationError:
            return False

    def topological_sort(self, dep_map: Dict[str, List[str]]) -> List[str]:
        """
        Return ordering of tasks if DAG, else raise ValidationError.
        Uses Kahn's algorithm.
        """
        # build inbound counts
        nodes = set(dep_map.keys())
        inbound: Dict[str, int] = {n: 0 for n in nodes}
        for src, deps in dep_map.items():
            for d in deps:
                inbound[src] += 1

        # queue nodes with zero inbound
        queue = [n for n, c in inbound.items() if c == 0]
        order: List[str] = []
        # adjacency is reverse: edges from dep->src
        rev_adj: Dict[str, List[str]] = {n: [] for n in nodes}
        for src, deps in dep_map.items():
            for d in deps:
                rev_adj[d].append(src)

        while queue:
            n = queue.pop(0)
            order.append(n)
            for neighbor in rev_adj.get(n, []):
                inbound[neighbor] -= 1
                if inbound[neighbor] == 0:
                    queue.append(neighbor)

        if len(order) != len(nodes):
            # cycle detected
            logger.debug("Cycle detected while topologically sorting. nodes=%s order=%s", nodes, order)
            raise ValidationError("Dependency graph contains a cycle")
        return order
