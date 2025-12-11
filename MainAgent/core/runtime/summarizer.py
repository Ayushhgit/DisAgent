from __future__ import annotations

import ast
from typing import Optional


def summarize_python_file(path: str, max_sentences: int = 5) -> str:
    """Create a short extractive summary for a Python file.

    Strategy:
    - Use module docstring if present
    - Else collect top-level class/function docstrings
    - Else return first non-empty lines up to limit
    """
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as fh:
            src = fh.read()
    except Exception:
        return ""

    try:
        tree = ast.parse(src)
    except Exception:
        # fallback: return first lines
        lines = [l.strip() for l in src.splitlines() if l.strip()]
        return " ".join(lines[:max_sentences])

    parts = []
    doc = ast.get_docstring(tree)
    if doc:
        parts.append(doc.strip())

    # collect docstrings from top-level defs
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            d = ast.get_docstring(node)
            if d:
                parts.append(d.strip())
        if len(parts) >= max_sentences:
            break

    if parts:
        # return first N sentences/parts
        return " \n".join(parts[:max_sentences])

    # fallback: first non-empty source lines
    lines = [l.strip() for l in src.splitlines() if l.strip()]
    return " ".join(lines[:max_sentences])


def semantic_score(a: str, b: str) -> float:
    """Return a simple semantic similarity score between two texts (0..1).

    Uses SequenceMatcher ratio as a lightweight approximation.
    """
    try:
        from difflib import SequenceMatcher

        if not a or not b:
            return 0.0
        return SequenceMatcher(None, a, b).ratio()
    except Exception:
        return 0.0
