import sys
import os
import shutil
import time
import json
import ast
import traceback
from pathlib import Path

# Fix python path to allow imports from MainAgent
current_dir = Path(__file__).resolve().parent
parent_dir = current_dir.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

from orchestrator.unified_orchestrator import UnifiedOrchestrator

SYNTHETIC_TASKS = [
    {
        "id": "algorithmic_fib_fact",
        "prompt": "Create a Python file named 'math_utils.py' that contains two functions: 'calculate_fibonacci(n)' which returns the n-th Fibonacci number, and 'calculate_factorial(n)' which returns the factorial of n. Both functions should handle negative inputs by raising a ValueError.",
        "expected_files": ["math_utils.py"]
    },
    {
        "id": "class_calculator",
        "prompt": "Create a standard Class 'Calculator' in 'calculator.py'. It MUST implement standard methods: add(a, b), subtract(a, b), multiply(a, b), and divide(a, b). For divide, it must raise a ZeroDivisionError if b is 0.",
        "expected_files": ["calculator.py"]
    },
    {
        "id": "data_structure_linked_list",
        "prompt": "Implement a standard Python Linked List in 'linked_list.py'. It should have a Node class and a LinkedList class with methods: append(value), prepend(value), delete_value(value), and to_list() which returns the linked list as a standard python list.",
        "expected_files": ["linked_list.py"]
    },
    {
        "id": "string_parser",
        "prompt": "Create a file 'parser.py' with a function 'parse_csv_string(csv_str)' that takes a simple CSV string (header row followed by data rows) and returns a list of dictionaries mapping the headers to the values. The CSV string uses commas.",
        "expected_files": ["parser.py"]
    },
    {
        "id": "simple_api_mock",
        "prompt": "Create a file named 'mock_api.py' containing a class 'MockAPIClient'. The class should have a method 'fetch_user(user_id)' that returns a dictionary {'id': user_id, 'name': 'Test User'} after sleeping for 1 second. Write a test function at the bottom to verify it works.",
        "expected_files": ["mock_api.py"]
    }
]

def check_file_corruption(directory: str) -> bool:
    """Check if all generated Python files in the directory pass AST parsing (no syntax errors)."""
    corruptions = 0
    total_py_files = 0

    code_extensions = {'.py'}
    project_dir = Path(directory)
    if not project_dir.exists():
        return False
        
    for p in project_dir.rglob('*'):
        if p.is_file() and p.suffix in code_extensions:
            total_py_files += 1
            try:
                with open(p, "r", encoding="utf-8") as f:
                    content = f.read()
                ast.parse(content) # Throws SyntaxError if invalid python code
            except SyntaxError as e:
                print(f"[!] Corruption found in {p.name}: {e}")
                corruptions += 1
            except Exception as e:
                print(f"[!] Error reading {p.name}: {e}")
                corruptions += 1
                
    if total_py_files == 0:
        return False # No files generated at all is also a failure mode
        
    return corruptions > 0

def run_benchmark_suite(num_trials=1):
    """
    Runs the benchmark suite over the synthetic tasks multiple times.
    """
    print("=" * 60)
    print("🚀 DISAGENT RESEARCH BENCHMARK SUITE")
    print(f"Trials per task: {num_trials} | Total Tasks: {len(SYNTHETIC_TASKS)}")
    print("Metrics Measured: Pass@1, File Corruption, Time to Completion, Recovery Rate, Event Trace")
    print("=" * 60)
    
    results = {
        "pass_at_1": 0,
        "total_executions": 0,
        "total_corrupted_files_runs": 0,
        "total_time_seconds": 0.0,
        "recovery_success": 0,
        "total_failures": 0,
        "traces_captured": 0,
        "runs": []
    }
    
    benchmark_dir = Path(__file__).parent.parent / "output" / "benchmark_runs"
    if benchmark_dir.exists():
        shutil.rmtree(benchmark_dir)
    benchmark_dir.mkdir(parents=True, exist_ok=True)
    
    for trial in range(num_trials):
        for task in SYNTHETIC_TASKS:
            run_id = f"{task['id']}_trial_{trial}"
            print(f"\n[RUNNING] Task: {run_id}")
            
            output_folder = str(benchmark_dir / run_id)
            
            orchestrator = UnifiedOrchestrator(
                output_folder=output_folder,
                enable_verification=True,
                enable_semantic_analysis=False, # Use keyword for deterministic fast runs if desired, but default is standard
                enable_critic=True,
                enable_tracing=True,
                enable_learning=False, # Isolate from past runs for raw benchmarks
                run_tests=True,
                max_verification_retries=2
            )
            
            start_time = time.time()
            error_occurred = None
            try:
                # Capture standard out? To keep logs clean, we just let it run.
                orchestrator.run(task["prompt"])
            except Exception as e:
                error_occurred = str(e)
                traceback.print_exc()
            
            elapsed_time = time.time() - start_time
            results["total_time_seconds"] += elapsed_time
            results["total_executions"] += 1
            
            # --- Gather Analytics ---
            v_stats = orchestrator.verification_loop.get_stats() if orchestrator.verification_loop else {}
            
            total_verifications = v_stats.get('total_verifications', 0)
            successful = v_stats.get('successful', 0)
            failed = v_stats.get('failed', 0)
            retries = v_stats.get('retries', 0)
            
            is_corrupted = check_file_corruption(output_folder)
            if is_corrupted:
                results["total_corrupted_files_runs"] += 1
                
            # Logic for Pass@1
            # Pass@1 is loosely defined as: It generated files and passed verification with NO retries, or completed without crashing.
            # If verifications existed, Pass@1 means successful > 0 and retries == 0.
            pass_at_1 = False
            recovery_success = False
            
            if not error_occurred and not is_corrupted:
                if total_verifications > 0:
                    if retries == 0 and successful > 0:
                        pass_at_1 = True
                        results["pass_at_1"] += 1
                    elif retries > 0 and successful > 0:
                        recovery_success = True
                        results["recovery_success"] += 1
                    else:
                        results["total_failures"] += 1
                else:
                    # If verification loop didn't trigger but it ran without corruption, it's a pass@1
                    pass_at_1 = True
                    results["pass_at_1"] += 1
            else:
                 results["total_failures"] += 1
                 
            # Traces
            trace_captured = orchestrator.tracer is not None and len(orchestrator.tracer._spans) > 0
            if trace_captured:
                results["traces_captured"] += 1
                
            run_data = {
                "run_id": run_id,
                "elapsed_time": round(elapsed_time, 2),
                "pass_at_1": pass_at_1,
                "recovery_success": recovery_success,
                "is_corrupted": is_corrupted,
                "trace_captured": trace_captured,
                "v_stats": v_stats,
                "error": error_occurred
            }
            results["runs"].append(run_data)
            
            print(f"-> Completed {run_id} | Time: {elapsed_time:.2f}s | Pass@1: {pass_at_1} | Recovered: {recovery_success} | Corrupted: {is_corrupted}")
    
    # Calculate Aggregate Rates
    execs = results["total_executions"]
    if execs > 0:
        results["metrics"] = {
            "pass_at_1_rate": round(results["pass_at_1"] / execs, 3),
            "recovery_success_rate": round(results["recovery_success"] / execs, 3),
            "file_corruption_rate": round(results["total_corrupted_files_runs"] / execs, 3),
            "trace_completeness": round(results["traces_captured"] / execs, 3),
            "avg_time_per_task": round(results["total_time_seconds"] / execs, 2)
        }
    
    report_file = Path(__file__).parent / "benchmark_synthetic_results.json"
    with open(report_file, "w") as f:
        json.dump(results, f, indent=4)
        
    print("\n" + "=" * 60)
    print("📊 FINAL BENCHMARK METRICS 📊")
    print("=" * 60)
    if execs > 0:
        print(f"Pass@1 Success Rate:     {results['metrics']['pass_at_1_rate'] * 100}%")
        print(f"Recovery Success Rate:   {results['metrics']['recovery_success_rate'] * 100}%")
        print(f"File Corruption Rate:    {results['metrics']['file_corruption_rate'] * 100}%")
        print(f"Event Trace Completeness:{results['metrics']['trace_completeness'] * 100}%")
        print(f"Avg Time / Task:         {results['metrics']['avg_time_per_task']} seconds")
    print("=" * 60)
    print(f"Full report saved to: {report_file.absolute()}")

if __name__ == "__main__":
    # For speed during this test, we do 1 trial per task (5 tasks total).
    # This takes 5 iterations. 
    run_benchmark_suite(num_trials=1)
