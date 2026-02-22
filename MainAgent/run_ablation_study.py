import sys
import os
import shutil
import random
from pathlib import Path
from collections import defaultdict
import time

sys.path.append(str(Path(".").resolve()))

from core.memory.experience_learner import ExperienceLearner, ExecutionExperience

# ---------------------------------------------------------
# Simulation Configuration
# ---------------------------------------------------------
NUM_TRIALS = 50
ENV_CHANGES = [0, 20] # Environment changes at trial 20 (v1 -> v2)

# Ground Truth: Valid strategies per environment
# v1 requires 'strat_A', v2 requires 'strat_B'
GROUND_TRUTH = {
    "env_v1": "strat_A",
    "env_v2": "strat_B"
}
NOISY_STRATEGIES = ["strat_C", "strat_D", "strat_E"]

def simulate_agent_execution(learner, task_id, env_hash, mode="full"):
    """
    Simulates an agent trying to solve a task.
    It asks the learner for a strategy first.
    """
    required_strategy = GROUND_TRUTH[env_hash]
    
    # 1. Ask Memory for help
    # Naive RAG simulates by ignoring env_hash (always passing a dummy current_env)
    query_env = env_hash if mode != "naive" else "unknown_env"
    
    suggested = learner.suggest_strategy(
        task_type="setup", 
        error_type="ImportError",
        current_env_hash=query_env
    )
    
    # 2. Determine Outcome
    success = False
    used_strategy = suggested
    
    if suggested == required_strategy:
        success = True
    else:
        # If memory failed (or was empty), agent tries random exploration
        # Simulating agent trying different things...
        # Bias towards the correct one eventually (to populate memory)
        if random.random() < 0.7: 
            used_strategy = required_strategy 
            success = True
        else:
            used_strategy = random.choice(NOISY_STRATEGIES)
            success = False
            
    # 3. Record Experience
    # In Naive mode, we still record, but maybe without valid hash? 
    # For fair comparison, let's say Naive RAG stores it but retrieves poorly.
    record_env = env_hash if mode != "naive" else "unknown_env"
    
    exp = ExecutionExperience(
        experience_id=f"exp_{task_id}",
        task_id=f"t_{task_id}",
        agent_id="sim_agent",
        task_type="setup",
        environment_hash=record_env,
        error_type="ImportError",
        patch_strategy=used_strategy,
        success=success
    )
    learner.record_episode(exp)
    
    return success, suggested == required_strategy

def run_experiment(mode_name):
    print(f"\n--- Running Experiment: {mode_name} ---")
    mem_path = f"./ablation_mem_{mode_name}"
    if os.path.exists(mem_path):
        shutil.rmtree(mem_path)
        
    learner = ExperienceLearner(storage_path=mem_path)
    
    history = []
    
    current_env = "env_v1"
    
    for i in range(NUM_TRIALS):
        # Change Environment
        if i in ENV_CHANGES and i > 0:
            current_env = "env_v2"
            print(f"[Trial {i}] Environment Shift! v1 -> v2")
            
        success, memory_was_useful = simulate_agent_execution(learner, i, current_env, mode=mode_name)
        history.append(memory_was_useful)
        
        # Periodic Sleep (Consolidation)
        if mode_name == "full" and i % 10 == 0:
            learner.consolidate()
            
    # Calculate "Adaptation Score"
    # How many times did memory give the CORRECT suggestion immediately?
    score = sum(history)
    print(f"Total Correct Suggestions: {score}/{NUM_TRIALS}")
    return score

if __name__ == "__main__":
    scores = {}
    
    # 1. Full System (DisAgent)
    scores["DisAgent (Full)"] = run_experiment("full")
    
    # 2. No Consolidation (Raw Logs Only)
    # We simulate this by just never calling consolidate() 
    # But wait, run_experiment calls it if mode == full.
    scores["No Consolidation"] = run_experiment("no_consolidation")
    
    # 3. Naive RAG (Blind to versions)
    scores["Naive RAG"] = run_experiment("naive")

    print("\n================ RESULTS ================")
    for k, v in scores.items():
        print(f"{k}: {v}/{NUM_TRIALS}")
    
    print("\nInterpretation:")
    print(" - Naive RAG should fail retrieval after Trial 20 (still suggesting v1 fixes).")
    print(" - DisAgent should drop briefly at Trial 20, then adapt fast.")
    print(" - Consolidation might help slightly or be equal to raw logs in this simple sim.")
