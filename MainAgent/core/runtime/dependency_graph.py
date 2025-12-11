from __future__ import annotations

import ast
import os
from typing import List, Dict, Set


def _extract_imported_modules(py_path: str) -> List[str]:
    """Return top-level module names imported by the given Python file."""
    try:
        with open(py_path, "r", encoding="utf-8", errors="ignore") as fh:
            source = fh.read()
    except Exception:
        return []

    try:
        tree = ast.parse(source)
    except Exception:
        return []

    mods: List[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for n in node.names:
                mods.append(n.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                mods.append(node.module)
    # normalize to top-level module (take first segment of dotted name)
    normalized = [m.split(".")[0] for m in mods if m]
    return list(dict.fromkeys(normalized))


def build_import_graph(root: str, file_paths: List[str]) -> Dict[str, List[str]]:
    """Build a simple import graph mapping each file -> list of other repo files it depends on.

    Args:
        root: project root directory to resolve relative file paths
        file_paths: list of file paths (relative or absolute) to consider

    Returns:
        graph: dict where keys are file paths (as given) and values are list of file paths they import
    """
    # Build map of potential module -> file path (stem -> path)
    module_map: Dict[str, str] = {}
    abs_paths: List[str] = []
    for p in file_paths:
        abs_p = p if os.path.isabs(p) else os.path.join(root, p)
        abs_paths.append(abs_p)
        name = os.path.splitext(os.path.basename(p))[0]
        if name:
            module_map[name] = abs_p

    graph: Dict[str, List[str]] = {}
    for rel_p, abs_p in zip(file_paths, abs_paths):
        try:
            imported = _extract_imported_modules(abs_p)
        except Exception:
            imported = []
        deps: List[str] = []
        for mod in imported:
            target = module_map.get(mod)
            if target:
                # store relative path key if possible
                deps.append(target)
        graph[rel_p] = deps

    return graph


def reverse_graph(graph: Dict[str, List[str]]) -> Dict[str, Set[str]]:
    rev: Dict[str, Set[str]] = {k: set() for k in graph}
    for src, targets in graph.items():
        for t in targets:
            if t not in rev:
                rev[t] = set()
            rev[t].add(src)
    return rev


def dependency_closure(graph: Dict[str, List[str]], seeds: List[str], depth: int = 1) -> Set[str]:
    """Return set of files that are in the dependency closure of `seeds` up to `depth` hops.

    This follows reverse dependencies: files that (directly or indirectly) depend on seeds.
    """
    rev = reverse_graph(graph)
    result: Set[str] = set()
    frontier = set(seeds)
    for _ in range(depth):
        next_frontier: Set[str] = set()
        for f in frontier:
            deps = rev.get(f, set())
            for d in deps:
                if d not in result:
                    next_frontier.add(d)
        result.update(next_frontier)
        frontier = next_frontier
        if not frontier:
            break
    return result
