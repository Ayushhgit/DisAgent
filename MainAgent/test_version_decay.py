
import sys
import os
from pathlib import Path

sys.path.append(str(Path(".").resolve()))

from core.memory.experience_learner import ExperienceLearner, ExecutionExperience, ResearchMetrics

def test_version_decay():
    print("Testing Version-Aware Memory System...")
    learner = ExperienceLearner(storage_path="./test_memory_v2")
    
    # Scene: Old Environment (v1.0)
    # Strategy 'pip install old-lib' worked here
    print("\n[Recorded] Episode 1 (Env: v1.0): Error fixed by 'pip install old-lib'")
    exp1 = ExecutionExperience(
        experience_id="exp_v1",
        task_id="t1",
        agent_id="a1",
        task_type="setup",
        environment_hash="env_v1.0",  # OLD VERSION
        error_type="ImportError",
        patch_strategy="pip install old-lib",
        success=True
    )
    learner.record_episode(exp1)
    
    # Scene: New Environment (v2.0)
    # Strategy 'pip install new-lib' worked here
    print("[Recorded] Episode 2 (Env: v2.0): Error fixed by 'pip install new-lib'")
    exp2 = ExecutionExperience(
        experience_id="exp_v2",
        task_id="t2",
        agent_id="a1",
        task_type="setup",
        environment_hash="env_v2.0",  # NEW VERSION
        error_type="ImportError",
        patch_strategy="pip install new-lib",
        success=True
    )
    learner.record_episode(exp2)
    
    # TEST 1: Retrieve in v1.0 Context
    # Should prefer 'old-lib' because env matches
    print("\n--- TEST 1: Retrieve context for Env v1.0 ---")
    strategy_v1 = learner.suggest_strategy("setup", "ImportError", current_env_hash="env_v1.0")
    print(f"Suggested Strategy (Expect 'old-lib'): {strategy_v1}")
    
    # TEST 2: Retrieve in v2.0 Context
    # Should prefer 'new-lib' because env matches better than the stale v1.0 memory
    print("\n--- TEST 2: Retrieve context for Env v2.0 ---")
    strategy_v2 = learner.suggest_strategy("setup", "ImportError", current_env_hash="env_v2.0")
    print(f"Suggested Strategy (Expect 'new-lib'): {strategy_v2}")
    
    # TEST 3: Retrieve in v3.0 Context (Unknown Env)
    # Should might default to either, but both get a drift penalty.
    # However, if we add more weight to recent? 
    # For now, let's just see what it picks (likely the one with higher base score or random if equal).
    # Ideally, it picks the 'most robust' one or neither if confidence is low.
    
    # Cleanup
    import shutil
    try:
        shutil.rmtree("./test_memory_v2")
    except:
        pass

if __name__ == "__main__":
    test_version_decay()
