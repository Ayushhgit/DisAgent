from __future__ import annotations

import os
from typing import List, Dict

from core.runtime.summarizer import summarize_python_file, semantic_score
from core.runtime.dependency_graph import build_import_graph, dependency_closure


def score_file(path: str, keywords: List[str], domains: List[str]) -> int:
    """Score a file by presence of keywords and domain tokens in filename/path.

    Content-based scoring and semantic similarity are applied in `select_relevant_files`.
    """
    score = 0
    low_path = path.lower()

    # filename/domain match
    for d in domains:
        if d and d.lower() in low_path:
            score += 5

    for kw in keywords:
        if kw and kw.lower() in low_path:
            score += 3

    return score


def select_relevant_files(all_file_paths: List[str], task_description: str, domains: List[str], max_files: int = 10) -> List[str]:
    """Select up to `max_files` from `all_file_paths` relevant to task_description/domains.

    Heuristics (combined):
    1. filename/domain token matching (fast)
    2. semantic similarity between task_description and file summary
    3. dependency closure: include files that depend on high-scoring files

    Returns selected file paths (same form as `all_file_paths`).
    """
    # build small keyword set from task description
    words = [w.strip().lower() for w in task_description.replace("/", " ").split() if len(w) > 3]
    keywords = []
    for w in words:
        if w not in keywords:
            keywords.append(w)

    # preliminary scoring by filename/path
    base_scores: Dict[str, float] = {}
    for p in all_file_paths:
        try:
            s = score_file(p, keywords[:6], domains[:3])
        except Exception:
            s = 0
        base_scores[p] = float(s)

    # compute summaries and semantic similarity
    sem_scores: Dict[str, float] = {}
    for p in all_file_paths:
        try:
            summary = summarize_python_file(p) if p.endswith(".py") else ""
            # combine file summary and filename for comparison
            text_for_comp = (summary + " " + os.path.basename(p)) if summary else os.path.basename(p)
            sem = semantic_score(task_description, text_for_comp)
        except Exception:
            sem = 0.0
        sem_scores[p] = sem

    # final combined score
    combined: Dict[str, float] = {}
    for p in all_file_paths:
        combined[p] = base_scores.get(p, 0.0) + (sem_scores.get(p, 0.0) * 10.0)

    # sort by combined score
    sorted_files = sorted(combined.items(), key=lambda kv: (-kv[1], kv[0]))
    candidate_order = [kv[0] for kv in sorted_files]

    # Use dependency graph to include upstream dependents of top candidates
    try:
        graph = build_import_graph(".", all_file_paths)
        # convert to relative keys used in graph (graph keys are as passed in)
        top_seed = [candidate_order[0]] if candidate_order else []
        deps = dependency_closure(graph, top_seed, depth=1) if top_seed else set()
    except Exception:
        deps = set()

    # assemble final list: include top candidates plus any dependency-closure files
    final_list: List[str] = []
    for p in candidate_order:
        if len(final_list) >= max_files:
            break
        final_list.append(p)

    # ensure we include dependency closure files (prepend if high priority)
    for d in deps:
        if d not in final_list and len(final_list) < max_files:
            final_list.append(d)

    # fallback
    if not final_list:
        return list(all_file_paths)[:max_files]
    return final_list[:max_files]

