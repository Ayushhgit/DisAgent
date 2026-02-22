
import sys
import os
from pathlib import Path
import shutil

sys.path.append(str(Path(".").resolve()))

from core.memory.experience_learner import ExperienceLearner, ExecutionExperience

def test_consolidation():
    print("Testing Tier-2 'Sleep Phase' Consolidation...")
    mem_path = "./test_consolidation_mem"
    if os.path.exists(mem_path):
        shutil.rmtree(mem_path)
    
    learner = ExperienceLearner(storage_path=mem_path)
    
    # 1. Record 3 raw episodes (Training Phase)
    # Same problem (AttributeError), Same solution (refactor_class), Same Context (Env A)
    print("\n[Wake Phase] experiencing 3 repeated failures & fixes...")
    for i in range(3):
        exp = ExecutionExperience(
            experience_id=f"raw_{i}",
            task_id=f"t_{i}",
            agent_id="a1",
            task_type="coding",
            environment_hash="env_A",
            error_type="AttributeError",
            patch_strategy="refactor_class",
            success=True
        )
        learner.record_episode(exp)
        print(f" - Recorded episode {i}")

    # 2. Trigger 'Sleep' (Consolidation)
    print("\n[Sleep Phase] Consolidating memories...")
    rules_created = learner.consolidate()
    print(f" -> Rules Created: {rules_created}")
    
    assert rules_created == 1, "Should have created exactly 1 semantic rule from the cluster."
    
    # 3. Verify Rule Creation
    # The learner should now have a rule prioritizing 'refactor_class' for AttributeError
    # And it should have high confidence.
    
    print("\n[Wake Phase] New task arrives (AttributeError)...")
    strategy = learner.suggest_strategy("coding", "AttributeError", current_env_hash="env_A")
    print(f" -> Suggested Strategy: {strategy}")
    
    assert strategy == "refactor_class", "Should suggest the consolidated rule strategy."
    
    # 4. Cleanup
    try:
        shutil.rmtree(mem_path)
    except:
        pass
    print("\n✅ Tier-2 Consolidation Test Passed.")

if __name__ == "__main__":
    test_consolidation()
