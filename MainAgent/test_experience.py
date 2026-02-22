
import sys
import os
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(".").resolve()))

from core.memory.experience_learner import ExperienceLearner, ExecutionExperience, ResearchMetrics

def test_episodic_memory():
    print("Initializing ExperienceLearner...")
    learner = ExperienceLearner(storage_path="./test_memory")
    
    # 1. Create a "failure" episode that was eventually fixed
    print("\n[Recorded] Episode 1: ImportError in db.py -> Fixed by adding dependency")
    exp1 = ExecutionExperience(
        experience_id="exp_001",
        task_id="task_123",
        agent_id="agent_coder",
        task_type="backend_feature",
        task_description="Implement user login",
        file_path="db.py",
        error_type="ImportError",
        error_context="ImportError: No module named 'sqlalchemy'",
        patch_strategy="install_dependency",
        success=True,
        metrics=ResearchMetrics(pass_at_1=False, retry_count=1, tokens_used=1500)
    )
    learner.record_episode(exp1)
    
    # 2. Create another similar episode
    print("[Recorded] Episode 2: ImportError in auth.py -> Fixed by adding dependency")
    exp2 = ExecutionExperience(
        experience_id="exp_002",
        task_id="task_124",
        agent_id="agent_coder",
        task_type="backend_feature",
        task_description="Setup auth middleware",
        file_path="auth.py",
        error_type="ImportError",
        error_context="ImportError: No module named 'jose'",
        patch_strategy="install_dependency",
        success=True,
        metrics=ResearchMetrics(pass_at_1=True, retry_count=0, tokens_used=800)
    )
    learner.record_episode(exp2)
    
    # 3. Test Retrieval
    print("\n[Retrieval] Searching for fix for new ImportError...")
    related = learner.retrieve_similar_experiences(
        task_type="backend_feature", 
        error_type="ImportError"
    )
    print(f"Found {len(related)} related episodes.")
    for r in related:
        print(f" - ID: {r.experience_id}, Strategy: {r.patch_strategy}, Score: N/A")

    # 4. Test Strategy Suggestion
    print("\n[Suggestion] Asking for best strategy for ImportError...")
    strategy = learner.suggest_strategy("backend_feature", "ImportError")
    print(f"Suggested Strategy: {strategy}")
    
    if strategy == "install_dependency":
        print("✅ SUCCESS: Correct strategy suggested.")
    else:
        print(f"❌ FAILURE: Unexpected strategy '{strategy}'")

    # 5. Test Metrics
    print("\n[Metrics] Generating research report...")
    metrics = learner.get_research_metrics()
    print(metrics)
    
    # Cleanup
    import shutil
    try:
        shutil.rmtree("./test_memory")
    except:
        pass

if __name__ == "__main__":
    test_episodic_memory()
