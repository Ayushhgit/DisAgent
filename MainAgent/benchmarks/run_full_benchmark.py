"""
DisAgent Full Benchmark Suite (Publication-Grade)
==================================================
Runs TWO configurations back-to-back on the SAME synthetic tasks:
  1. DisAgent (Full Multi-Agent Orchestrator)
  2. Single-Agent Baseline (Raw LLM call, no orchestrator)

Each task is run for N_TRIALS to compute mean ± std for every metric.

Metrics captured:
  - Pass@1 success rate
  - File corruption rate (AST validation)
  - Average time per task
  - Recovery success rate (verification retries)
  - Event trace completeness
"""

import sys
import os
import shutil
import time
import json
import ast
import math
import traceback
from pathlib import Path
from typing import Dict, List, Any

# Fix python path
current_dir = Path(__file__).resolve().parent
parent_dir = current_dir.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

from orchestrator.unified_orchestrator import UnifiedOrchestrator
from core.runtime.llm import GroqService

# ─────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────
N_TRIALS = 3  # Trials per task per configuration

SYNTHETIC_TASKS = [
    {
        "id": "T1_fibonacci_factorial",
        "prompt": "Create a Python file named 'math_utils.py' that contains two functions: 'calculate_fibonacci(n)' which returns the n-th Fibonacci number using iteration, and 'calculate_factorial(n)' which returns the factorial of n. Both functions should handle negative inputs by raising a ValueError.",
        "expected_files": ["math_utils.py"],
        "complexity": "Low",
        "category": "Algorithmic"
    },
    {
        "id": "T2_calculator_class",
        "prompt": "Create a standard Class 'Calculator' in 'calculator.py'. It MUST implement methods: add(a, b), subtract(a, b), multiply(a, b), and divide(a, b). For divide, it must raise a ZeroDivisionError if b is 0.",
        "expected_files": ["calculator.py"],
        "complexity": "Low",
        "category": "OOP"
    },
    {
        "id": "T3_linked_list",
        "prompt": "Implement a standard Python Linked List in 'linked_list.py'. It should have a Node class and a LinkedList class with methods: append(value), prepend(value), delete_value(value), and to_list() which returns the linked list as a standard python list.",
        "expected_files": ["linked_list.py"],
        "complexity": "Medium",
        "category": "Data Structure"
    },
    {
        "id": "T4_csv_parser",
        "prompt": "Create a file 'parser.py' with a function 'parse_csv_string(csv_str)' that takes a simple CSV string (header row followed by data rows separated by newlines) and returns a list of dictionaries mapping the headers to values.",
        "expected_files": ["parser.py"],
        "complexity": "Medium",
        "category": "String Processing"
    },
    {
        "id": "T5_stack_implementation",
        "prompt": "Create a file 'stack.py' containing a class 'Stack' with methods push(item), pop(), peek(), is_empty(), and size(). Pop and peek should raise an IndexError when the stack is empty.",
        "expected_files": ["stack.py"],
        "complexity": "Low",
        "category": "Data Structure"
    },
]

# ─────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────

def validate_python_files(directory: str) -> dict:
    """AST-parse every .py file in directory. Returns stats."""
    project_dir = Path(directory)
    total = 0
    corrupted = 0
    files_info = []

    if not project_dir.exists():
        return {"total": 0, "corrupted": 0, "files": [], "has_files": False}

    for p in project_dir.rglob("*.py"):
        total += 1
        try:
            content = p.read_text(encoding="utf-8")
            ast.parse(content)
            files_info.append({"file": p.name, "valid": True, "lines": len(content.splitlines())})
        except SyntaxError as e:
            corrupted += 1
            files_info.append({"file": p.name, "valid": False, "error": str(e)})
        except Exception as e:
            corrupted += 1
            files_info.append({"file": p.name, "valid": False, "error": str(e)})

    return {"total": total, "corrupted": corrupted, "files": files_info, "has_files": total > 0}


def extract_code_from_llm_response(response: str) -> str:
    """Extract Python code from LLM response (handles markdown fences)."""
    if "```python" in response:
        parts = response.split("```python")
        if len(parts) > 1:
            code = parts[1].split("```")[0]
            return code.strip()
    if "```" in response:
        parts = response.split("```")
        if len(parts) > 1:
            code = parts[1]
            # Remove language identifier if present
            if code.startswith("py\n") or code.startswith("python\n"):
                code = code.split("\n", 1)[1]
            return code.strip()
    return response.strip()


def mean_std(values: List[float]) -> tuple:
    """Calculate mean and standard deviation."""
    if not values:
        return 0.0, 0.0
    n = len(values)
    m = sum(values) / n
    if n < 2:
        return m, 0.0
    variance = sum((x - m) ** 2 for x in values) / (n - 1)
    return m, math.sqrt(variance)


# ─────────────────────────────────────────────────────
# Benchmark Runners
# ─────────────────────────────────────────────────────

def run_disagent_trial(task: dict, output_folder: str) -> dict:
    """Run a single task through the full DisAgent orchestrator."""
    orchestrator = UnifiedOrchestrator(
        output_folder=output_folder,
        enable_verification=True,
        enable_semantic_analysis=False,
        enable_critic=True,
        enable_tracing=True,
        enable_learning=False,
        run_tests=True,
        max_verification_retries=2
    )

    start = time.time()
    error = None
    try:
        orchestrator.run(task["prompt"])
    except Exception as e:
        error = str(e)
        traceback.print_exc()
    elapsed = time.time() - start

    # Gather verification stats
    v_stats = orchestrator.verification_loop.get_stats() if orchestrator.verification_loop else {}
    total_v = v_stats.get("total_verifications", 0)
    successful_v = v_stats.get("successful", 0)
    retries = v_stats.get("retries", 0)

    # File validation
    file_check = validate_python_files(output_folder)
    is_corrupted = file_check["corrupted"] > 0

    # Pass@1 logic
    pass_at_1 = False
    recovery = False
    if not error and file_check["has_files"] and not is_corrupted:
        if total_v > 0:
            if retries == 0 and successful_v > 0:
                pass_at_1 = True
            elif retries > 0 and successful_v > 0:
                recovery = True
        else:
            pass_at_1 = True

    # Trace completeness
    trace_ok = False
    if orchestrator.tracer and len(orchestrator.tracer._spans) > 0:
        trace_ok = True

    return {
        "pass_at_1": pass_at_1,
        "recovery": recovery,
        "corrupted": is_corrupted,
        "trace_ok": trace_ok,
        "time": round(elapsed, 2),
        "error": error,
        "files_generated": file_check["total"],
        "v_stats": v_stats
    }


def run_baseline_trial(task: dict, output_folder: str) -> dict:
    """Run a single task through a raw single-agent LLM call (no orchestrator)."""
    out_dir = Path(output_folder)
    out_dir.mkdir(parents=True, exist_ok=True)

    system_prompt = (
        "You are a Python code generator. Output ONLY valid Python code. "
        "Do not include any explanations, markdown fences, or comments outside the code. "
        "The code must be syntactically valid and complete."
    )
    user_prompt = task["prompt"]

    start = time.time()
    error = None
    raw_response = ""
    try:
        service = GroqService()
        raw_response = service.completion(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=4096,
            temperature=0.7
        )
    except Exception as e:
        error = str(e)
        traceback.print_exc()
    elapsed = time.time() - start

    # Write to file
    pass_at_1 = False
    is_corrupted = False
    if raw_response and not error:
        code = extract_code_from_llm_response(raw_response)
        expected = task["expected_files"][0] if task["expected_files"] else "output.py"
        file_path = out_dir / expected
        file_path.write_text(code, encoding="utf-8")

        # Validate
        try:
            ast.parse(code)
            pass_at_1 = True
        except SyntaxError:
            is_corrupted = True

    return {
        "pass_at_1": pass_at_1,
        "recovery": False,  # No recovery mechanism in baseline
        "corrupted": is_corrupted,
        "trace_ok": False,  # No tracing in baseline
        "time": round(elapsed, 2),
        "error": error,
        "files_generated": 1 if raw_response and not error else 0,
        "v_stats": {}
    }


# ─────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────

def run_full_benchmark():
    print("=" * 70)
    print("  DISAGENT FULL BENCHMARK SUITE (Publication-Grade)")
    print(f"  Tasks: {len(SYNTHETIC_TASKS)} | Trials per task: {N_TRIALS}")
    print(f"  Configurations: DisAgent (Full) vs Single-Agent Baseline")
    print("=" * 70)

    benchmark_base = Path(__file__).parent.parent / "output" / "full_benchmark"
    if benchmark_base.exists():
        shutil.rmtree(benchmark_base)

    all_results = {}

    for config_name, runner_fn in [("DisAgent", run_disagent_trial), ("Baseline", run_baseline_trial)]:
        print(f"\n{'─' * 70}")
        print(f"  CONFIGURATION: {config_name}")
        print(f"{'─' * 70}")

        config_results = {
            "runs": [],
            "pass_at_1_per_trial": [],
            "corruption_per_trial": [],
            "time_per_trial": [],
        }

        for task in SYNTHETIC_TASKS:
            for trial in range(N_TRIALS):
                run_id = f"{task['id']}_trial_{trial}"
                print(f"\n  [{config_name}] Running: {run_id}")

                output_folder = str(benchmark_base / config_name / run_id)

                result = runner_fn(task, output_folder)
                result["run_id"] = run_id
                result["task_id"] = task["id"]
                result["category"] = task["category"]
                result["complexity"] = task["complexity"]
                result["trial"] = trial
                config_results["runs"].append(result)

                config_results["pass_at_1_per_trial"].append(1 if result["pass_at_1"] else 0)
                config_results["corruption_per_trial"].append(1 if result["corrupted"] else 0)
                config_results["time_per_trial"].append(result["time"])

                status = "✅ PASS" if result["pass_at_1"] else ("⚠️ RECOVERED" if result["recovery"] else "❌ FAIL")
                print(f"    -> {status} | Time: {result['time']}s | Corrupted: {result['corrupted']}")

        # Compute aggregate metrics
        total = len(config_results["runs"])
        pass_vals = config_results["pass_at_1_per_trial"]
        corr_vals = config_results["corruption_per_trial"]
        time_vals = config_results["time_per_trial"]

        pass_mean, pass_std = mean_std([float(v) for v in pass_vals])
        corr_mean, corr_std = mean_std([float(v) for v in corr_vals])
        time_mean, time_std = mean_std(time_vals)

        recovery_count = sum(1 for r in config_results["runs"] if r["recovery"])
        trace_count = sum(1 for r in config_results["runs"] if r["trace_ok"])

        config_results["metrics"] = {
            "total_runs": total,
            "pass_at_1_rate": round(pass_mean, 3),
            "pass_at_1_std": round(pass_std, 3),
            "file_corruption_rate": round(corr_mean, 3),
            "file_corruption_std": round(corr_std, 3),
            "avg_time": round(time_mean, 2),
            "time_std": round(time_std, 2),
            "recovery_rate": round(recovery_count / total, 3) if total > 0 else 0,
            "trace_completeness": round(trace_count / total, 3) if total > 0 else 0,
        }

        all_results[config_name] = config_results

    # ── Save Full JSON Report ──
    report_file = Path(__file__).parent / "full_benchmark_results.json"
    with open(report_file, "w") as f:
        json.dump(all_results, f, indent=4)

    # ── Print Comparative Table ──
    print("\n" + "=" * 70)
    print("  📊 COMPARATIVE RESULTS TABLE (Paper-Ready)")
    print("=" * 70)
    print(f"{'Metric':<30} {'DisAgent':<20} {'Baseline':<20}")
    print("-" * 70)
    for metric_key, label in [
        ("pass_at_1_rate", "Pass@1 Rate"),
        ("file_corruption_rate", "File Corruption Rate"),
        ("avg_time", "Avg Time/Task (s)"),
        ("recovery_rate", "Recovery Rate"),
        ("trace_completeness", "Trace Completeness"),
    ]:
        da = all_results["DisAgent"]["metrics"]
        bl = all_results["Baseline"]["metrics"]

        if metric_key in ("pass_at_1_rate", "file_corruption_rate"):
            da_str = f"{da[metric_key]*100:.1f}% ± {da.get(metric_key.replace('rate','std'), da.get(metric_key.split('_rate')[0]+'_std', 0))*100:.1f}%"
            bl_str = f"{bl[metric_key]*100:.1f}% ± {bl.get(metric_key.replace('rate','std'), bl.get(metric_key.split('_rate')[0]+'_std', 0))*100:.1f}%"
        elif metric_key == "avg_time":
            da_str = f"{da['avg_time']:.2f} ± {da['time_std']:.2f}"
            bl_str = f"{bl['avg_time']:.2f} ± {bl['time_std']:.2f}"
        else:
            da_str = f"{da[metric_key]*100:.1f}%"
            bl_str = f"{bl[metric_key]*100:.1f}%"

        print(f"{label:<30} {da_str:<20} {bl_str:<20}")

    print("=" * 70)
    print(f"\nFull report: {report_file.absolute()}")


if __name__ == "__main__":
    run_full_benchmark()
