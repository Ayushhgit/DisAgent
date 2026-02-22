"""
Pydantic Migration Benchmark (Mini-Drift)
-----------------------------------------
Simulates a generic "Drift" scenario:
1. Agent learns Pydantic v1 patterns (e.g., `class Config`, `@validator`).
2. Environment shifts to Pydantic v2 (requires `model_config`, `@field_validator`).
3. We measure how many times the agent suggests stale v1 code.

Metrics:
- Stale Retrieval Rate: % of time v1 code is suggested in v2 env.
- Adaptation Speed: Trials needed to switch to v2.
"""

import sys
import os
import shutil
import random
from pathlib import Path
from typing import Tuple

import sys
import os
import shutil
import random
from pathlib import Path
from typing import Tuple

# Add parent dir to path correctly
current_dir = Path(__file__).resolve().parent
parent_dir = current_dir.parent
sys.path.append(str(parent_dir))

from core.memory.experience_learner import ExperienceLearner, ExecutionExperience

# Constants
ENV_V1 = "env_pydantic_v1"
ENV_V2 = "env_pydantic_v2"

TASKS = [
    {"id": "t1", "desc": "Define a model with config", "type": "model_def"},
    {"id": "t2", "desc": "Add validation to field", "type": "validation"},
    {"id": "t3", "desc": "Parse env vars", "type": "settings"},
]

# Mock "Ground Truth" execution function
def mock_execution(strategy: str, env_hash: str) -> Tuple[bool, str]:
    """
    Simulates running the code.
    Returns (success, error_type).
    """
    if env_hash == ENV_V1:
        if "v1" in strategy: return True, None
        return False, "SyntaxError" # v2 code fails in v1
        
    if env_hash == ENV_V2:
        if "v2" in strategy: return True, None
        if "v1" in strategy: return False, "DeprecationError" # v1 code fails in v2
        return False, "UnknownError"
    
    return False, "EnvError"

def run_benchmark():
    # Setup
    mem_path = Path("benchmark_mem")
    if mem_path.exists(): shutil.rmtree(mem_path)
    
    # Initialize SOMA Memory
    learner = ExperienceLearner(storage_path=mem_path)
    
    print("\n=== Phase 1: Learning Pydantic v1 (Stationary) ===")
    print("Agent is solving tasks in Pydantic v1 environment...")
    
    # Train heavily on v1 patterns
    for i in range(20):
        task = random.choice(TASKS)
        # In v1, the specific strategy "use_v1_config" works
        strat = "use_v1_pattern"
        
        # Record success
        obs = ExecutionExperience(
            experience_id=f"exp_v1_{i}",
            task_id=task["id"],
            agent_id="agent_1",
            task_type=task["type"],
            environment_hash=ENV_V1,
            error_type="None", # Successful run
            patch_strategy=strat,
            success=True
        )
        learner.record_episode(obs)
        if i % 5 == 0:
            print(f"  [Train] Processed {i} v1 episodes...")

    # Consolidate (Sleep Phase)
    print("  [Sleep] Consolidating memories into Semantic Rules...")
    num_rules = learner.consolidate()
    print(f"  Result: Created {num_rules} semantic rules for v1.")

    print("\n=== Phase 2: The Shift (Drift Event) ===")
    print(f"Environment silently upgrades: {ENV_V1} -> {ENV_V2}")
    
    print("\n=== Phase 3: Adaptation Evaluation ===")
    
    stale_retrievals = 0
    failures = 0
    active_grounding_triggers = 0
    total_trials = 10
    
    for i in range(total_trials):
        task = TASKS[0] # Test on "Define model" task
        
        # 1. Ask Memory for Strategy
        print(f"\n[Trial {i+1}] Task: {task['desc']}")
        
        # NOTE: suggest_strategy now returns (strategy, rule_id, confidence)
        strategy, rule_id, conf = learner.suggest_strategy(
            task_type=task["type"],
            error_type="DeprecationError", # Simulate we hit an error and ask for help
            current_env_hash=ENV_V2
        )
        
        print(f"  Memory Suggested: {strategy} (Conf: {conf:.2f})")
        
        # 2. Check for Stale Retrieval
        if strategy == "use_v1_pattern":
            print("  ⚠️ ALERT: Stale v1 strategy retrieved!")
            stale_retrievals += 1
            
            # Active Grounding Logic (Simulated interaction with Orchestrator)
            # If SOMA is working, it might flag this rule for checking if conf is low
            stale_rules = learner.get_stale_rules(ENV_V2)
            if any(r.rule_id == rule_id for r in stale_rules):
                print("  🔍 SOMA detected drift! Triggering Active Grounding Probe...")
                active_grounding_triggers += 1
                
                # Probe: Try running v1 code in v2 env -> Fails
                success, err = mock_execution(strategy, ENV_V2)
                
                # Update Memory with Probe Result
                learner.ground_stale_memory(rule_id, success, ENV_V2)
                print(f"  📝 Probe Result: {success} (Error: {err}) -> Memory Updated")
            else:
                # If confidence was high, we just try to execute and fail naturally
                success, err = mock_execution(strategy, ENV_V2)
                # Record the feedback loop
                learner.record_retrieval_outcome(rule_id, success, ENV_V2)
                print(f"  ❌ Execution Failed (Error: {err}) -> Feedback Loop Recorded")
                
        elif strategy == "use_v2_pattern":
            print("  ✅ SUCCESS: Adapted to v2 strategy!")
        else:
            print("  ℹ️ No strategy found (Cold Start in v2)")
            # Simulate agent figuring it out via LLM (not memory)
            # and saving the new v2 experience
            print("  🤖 Agent finding new solution via reasoning...")
            new_obs = ExecutionExperience(
                experience_id=f"exp_v2_{i}",
                task_id=task["id"],
                agent_id="agent_1",
                task_type=task["type"],
                environment_hash=ENV_V2,
                error_type="DeprecationError",
                patch_strategy="use_v2_pattern",
                success=True
            )
            learner.record_episode(new_obs)
            # Consolidate immediately for demo speed (usually happens periodically)
            learner.consolidate() 

    results = {
        "total_trials": total_trials,
        "stale_retrievals": stale_retrievals,
        "active_grounding_triggers": active_grounding_triggers
    }
    
    print("\n=== Results Summary ===")
    print(f"Total Trials: {total_trials}")
    print(f"Stale Retrievals: {stale_retrievals}")
    print(f"Active Grounding Triggers: {active_grounding_triggers}")
    
    # Save to file to avoid losing data on crash
    import json
    with open("benchmark_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    try:
        cleanup_mem(mem_path)
    except Exception as e:
        print(f"Cleanup warning: {e}")

def cleanup_mem(path):
    if path.exists():
        try:
            shutil.rmtree(path)
        except:
            pass

if __name__ == "__main__":
    run_benchmark()
