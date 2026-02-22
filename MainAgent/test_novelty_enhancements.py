"""
Test script for Novelty Enhancements:
1. Interference-Aware Retrieval
2. Active Memory Grounding
"""

import sys
import os
import shutil
from pathlib import Path

sys.path.append(str(Path(".").resolve()))

from core.memory.experience_learner import ExperienceLearner, ExecutionExperience, SemanticRule

def test_interference_aware_retrieval():
    """Test that unreliable rules get penalized."""
    print("\n=== Test: Interference-Aware Retrieval ===")
    
    mem_path = "./test_interference_mem"
    if os.path.exists(mem_path):
        shutil.rmtree(mem_path)
    
    learner = ExperienceLearner(storage_path=mem_path)
    
    # Record experiences to create a rule
    for i in range(3):
        exp = ExecutionExperience(
            experience_id=f"exp_int_{i}",
            task_id=f"task_{i}",
            agent_id="test_agent",
            task_type="setup",
            environment_hash="env_v1",
            error_type="ImportError",
            patch_strategy="add_import",
            success=True
        )
        learner.record_episode(exp)
    
    # Consolidate to create a semantic rule
    new_rules = learner.consolidate()
    print(f"Created {new_rules} Semantic Rules")
    
    # Get strategy - should work
    strategy, rule_id, confidence = learner.suggest_strategy(
        task_type="setup",
        error_type="ImportError",
        current_env_hash="env_v1"
    )
    print(f"First retrieval: strategy={strategy}, rule_id={rule_id}, confidence={confidence:.2f}")
    
    # Simulate rule being used but FAILING multiple times
    if rule_id:
        for _ in range(6):
            learner.record_retrieval_outcome(rule_id, success=False, current_env_hash="env_v1")
    
    # Get strategy again - should have lower confidence due to interference
    strategy2, rule_id2, confidence2 = learner.suggest_strategy(
        task_type="setup",
        error_type="ImportError",
        current_env_hash="env_v1"
    )
    print(f"After 6 failures: strategy={strategy2}, rule_id={rule_id2}, confidence={confidence2:.2f}")
    
    assert confidence2 < confidence, "Interference penalty should reduce confidence!"
    print("✅ Interference penalty is working!")
    
    # Cleanup
    shutil.rmtree(mem_path)

def test_active_memory_grounding():
    """Test the Active Memory Grounding (probe verification) mechanism."""
    print("\n=== Test: Active Memory Grounding ===")
    
    mem_path = "./test_grounding_mem"
    if os.path.exists(mem_path):
        shutil.rmtree(mem_path)
    
    learner = ExperienceLearner(storage_path=mem_path)
    
    # Create experiences in OLD environment
    for i in range(3):
        exp = ExecutionExperience(
            experience_id=f"exp_gnd_{i}",
            task_id=f"task_{i}",
            agent_id="test_agent",
            task_type="api_call",
            environment_hash="env_old",
            error_type="TypeError",
            patch_strategy="use_new_api",
            success=True
        )
        learner.record_episode(exp)
    
    learner.consolidate()
    print("Created rule in env_old")
    
    # Check if rule is stale in NEW environment
    stale_rules = learner.get_stale_rules(current_env_hash="env_new", threshold=0.0)
    print(f"Stale rules needing grounding: {len(stale_rules)}")
    
    assert len(stale_rules) > 0, "Should have stale rules in new env!"
    
    # Simulate grounding probe - rule is STILL VALID
    rule = stale_rules[0]
    print(f"Grounding rule {rule.rule_id}...")
    learner.ground_stale_memory(
        rule_id=rule.rule_id,
        probe_result=True,
        current_env_hash="env_new"
    )
    
    # Now check strategy in new env
    strategy, rule_id, confidence = learner.suggest_strategy(
        task_type="api_call",
        error_type="TypeError",
        current_env_hash="env_new"
    )
    print(f"After grounding: strategy={strategy}, confidence={confidence:.2f}")
    
    assert strategy == "use_new_api", "Rule should be valid after grounding!"
    print("✅ Active Memory Grounding is working!")
    
    # Cleanup
    shutil.rmtree(mem_path)

if __name__ == "__main__":
    test_interference_aware_retrieval()
    test_active_memory_grounding()
    print("\n🎉 All Novelty Enhancement tests passed!")
