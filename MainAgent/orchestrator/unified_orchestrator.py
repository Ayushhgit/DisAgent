"""Unified orchestrator combining planning and execution.

This orchestrator integrates:
  - planning layer: ScopeAnalyzer → ChainOfThoughtProcessor → TaskPlanner → StateTracker
  - execution layer: dynamic agent spawning and file editing
  - memory system: unified context management
  
Architecture:
  User Request → Scope Analysis → Task Planning → Agent Execution → Integration
"""

from __future__ import annotations

import time
import logging
from typing import Dict, List, Optional
from pathlib import Path
from datetime import datetime

from .context import AgentContext, create_context
from .dynamic_agents import create_dynamic_agent
from .file_manager import FileManager
from .scope_prompting import prompt_engineer_agent

from core.planning.task_planner import TaskPlanner
from core.planning.scope_analyzer import ScopeAnalyzer
from core.planning.chain_of_thought import ChainOfThoughtProcessor
from core.planning.validation import PlanValidator
from core.context.state_tracker import StateTracker
from core.memory.memory_types import TaskInfo, TaskType, AgentState, MemoryPriority
from core.runtime.llm import llm_call
from core.runtime.agent import AgentResult, Agent
from core.runtime.scheduler import TaskScheduler

logger = logging.getLogger(__name__)


def analyze_scope(user_request: str) -> Dict:
    """Wrapper function to analyze scope from user request.

    Returns a validated scope dict with all required fields.
    Raises RuntimeError if scope analysis fails or returns invalid data.
    """
    analyzer = ScopeAnalyzer()
    scope = analyzer.analyze(user_request)

    # Validate that we got a proper dict with required fields
    if not scope or not isinstance(scope, dict):
        raise RuntimeError("ScopeAnalyzer returned empty or invalid scope result")

    # Ensure required fields exist (with safe defaults if somehow missing)
    required_fields = {
        "title": user_request[:50],
        "description": user_request,
        "tasks": [],
        "agents_needed": ["generalist"],
        "complexity": "medium",
        "scope_level": "feature",
        "domains": ["general"],
    }

    for field, default in required_fields.items():
        if field not in scope or scope[field] is None:
            logger.warning("ScopeAnalyzer missing field '%s', using default", field)
            scope[field] = default

    return scope


class UnifiedOrchestrator:
    """Orchestrates planning, execution, and integration in a unified workflow."""

    def __init__(self, output_folder: str = "./generated_project"):
        """Initialize the orchestrator with file manager and state tracking.
        
        Args:
            output_folder: Base output directory for generated files
        """
        self.output_folder = output_folder
        self.file_manager = FileManager(output_folder)
        self.state_tracker = StateTracker(main_agent_id="main_orchestrator")
        self.task_planner = TaskPlanner(self.state_tracker, PlanValidator())
        self.cot_processor = ChainOfThoughtProcessor()
        
        self.project_name = Path(output_folder).name
        self.context = create_context(self.project_name)
        
        self.execution_stats: Dict[str, float] = {}
        self.overall_start_time = time.time()

    def run(self, user_request: str) -> None:
        """Execute the complete orchestration workflow.
        
        Args:
            user_request: The user's high-level request
        """
        try:
            print("\n" + "=" * 80)
            print("UNIFIED ORCHESTRATOR - Planning + Execution")
            print("=" * 80)
            print(f"\n[REQUEST] {user_request}\n")

            # Load existing memory if available
            if self.context.load_memory():
                print(f"[OK] Loaded existing memory for project: {self.project_name}\n")
            else:
                print(f"[OK] Starting fresh memory for project: {self.project_name}\n")

            # ===== STAGE 1: SCOPE ANALYSIS =====
            print("\n" + "─" * 80)
            print("[STAGE 1: SCOPE ANALYSIS]")
            print("─" * 80)
            scope_start = time.time()
            
            scope_info = analyze_scope(user_request)
            self.execution_stats["scope_analysis"] = round(time.time() - scope_start, 2)
            self.context.set_scope(scope_info)
            
            print(f"Scope Level: {scope_info.get('scope_level', 'unknown')}")
            print(f"Complexity: {scope_info.get('complexity', 'unknown')}")
            print(f"Domains: {', '.join(scope_info.get('domains', []))}")
            print(f"Time: {self.execution_stats['scope_analysis']}s\n")

            # ===== STAGE 2: CHAIN OF THOUGHT REASONING =====
            print("\n" + "─" * 80)
            print("[STAGE 2: REASONING & PLANNING]")
            print("─" * 80)
            cot_start = time.time()

            # Build reasoning text from scope info
            reasoning_text = f"Task: {scope_info.get('title', user_request)}\n"
            reasoning_text += f"Description: {scope_info.get('description', '')}\n"
            for task in scope_info.get('tasks', []):
                reasoning_text += f"- {task.get('description', '')}\n"

            reasoning_result = self.cot_processor.process_raw_reasoning(reasoning_text)
            reasoning_steps = reasoning_result.get("steps", [])
            self.execution_stats["chain_of_thought"] = round(time.time() - cot_start, 2)

            # CRITICAL: Store reasoning in context for agents to use
            self.context.set_reasoning(reasoning_result)

            # Also store in short-term memory for persistence
            if self.context.unified_memory:
                self.context.unified_memory.short_term.add_decision(
                    agent_id="orchestrator",
                    decision=f"Chain of thought analysis: {reasoning_result.get('summary', '')}",
                    context=str(reasoning_steps),
                    priority=MemoryPriority.HIGH
                )

            print(f"Generated {len(reasoning_steps)} reasoning steps")
            if reasoning_result.get('summary'):
                print(f"Summary: {reasoning_result['summary'][:100]}...")
            print(f"Time: {self.execution_stats['chain_of_thought']}s\n")

            # ===== STAGE 3: PROMPT ENGINEERING =====
            print("\n" + "─" * 80)
            print("[STAGE 3: PROMPT ENGINEERING]")
            print("─" * 80)
            prompt_start = time.time()
            
            # Use new scope analyzer output for agent selection
            agent_names = scope_info.get("agents_needed", [])
            custom_prompts = prompt_engineer_agent(user_request, agent_names, self.context)
            self.execution_stats["prompt_engineering"] = round(time.time() - prompt_start, 2)
            
            print(f"Engineered prompts for {len(agent_names)} agents: {', '.join(agent_names)}")
            print(f"Time: {self.execution_stats['prompt_engineering']}s\n")

            # ===== STAGE 4: REGISTER AGENTS & CREATE TASKS IN StateTracker =====
            print("\n" + "─" * 80)
            print("[STAGE 4: TASK REGISTRATION & AGENT SETUP]")
            print("─" * 80)
            
            # Register all agents in StateTracker
            for agent_name in agent_names:
                self.state_tracker.register_agent(agent_name)
                self.context.register_agent(agent_name)
                print(f"[OK] Registered agent: {agent_name}")

            # Create tasks using TaskPlanner (preserves dependencies, sets assigned_agent=None)
            task_ids = self.task_planner.plan_and_register(scope_info)
            for tid in task_ids:
                print(f"[OK] Registered task: {tid}")

            print()

            # ===== STAGE 5: DYNAMIC AGENT EXECUTION =====
            print("\n" + "─" * 80)
            print("[STAGE 5: AGENT EXECUTION]")
            print("─" * 80)

            # ===== STAGE 5: DYNAMIC AGENT EXECUTION via Scheduler =====
            # Create Agent wrappers and run them via TaskScheduler so execution is concurrent
            print("\n[STAGE 5: AGENT EXECUTION (SCHEDULED)]")

            # Thin Agent wrapper around the existing create_dynamic_agent function
            # DynamicAgent now uses relevance selector to choose files
            from core.runtime.relevance import select_relevant_files

            class DynamicAgent(Agent):
                def __init__(self, name: str):
                    super().__init__(name)

                def run(self, task_id: str, prompt: str, context, file_manager):
                    # gather all project files (FileManager returns relative paths)
                    all_files = file_manager.list_files("*")
                    # derive domains from scope_info if available
                    domains = scope_info.get("domains", []) if scope_info else []
                    # compute relevant files using heuristics
                    relevant = select_relevant_files(all_files, prompt, domains, max_files=12)
                    output = create_dynamic_agent(self.name, prompt, user_request, context, file_manager, allowed_files=relevant)
                    success = bool(output and len(output) > 20)
                    return AgentResult(success=success, output=output or "", metadata={"files": relevant})

            # Build agents map
            agents_map = {name: DynamicAgent(name) for name in agent_names}

            # Instantiate scheduler and run
            scheduler = TaskScheduler(self.state_tracker, self.task_planner, max_workers=min(4, max(1, len(agent_names))))
            scheduler_stats = scheduler.run(agents_map, custom_prompts, self.context, self.file_manager)
            # Merge scheduling stats
            self.execution_stats.update(scheduler_stats)

            print()

            # ===== STAGE 6: FINAL INTEGRATION =====
            print("\n" + "─" * 80)
            print("[STAGE 6: FINAL INTEGRATION & SUMMARY]")
            print("─" * 80)

            integration_start = time.time()

            final_summary = self.file_manager.get_project_summary()
            memory_context = ""
            
            if self.context.unified_memory:
                learnings = self.context.unified_memory.long_term.get_learnings(limit=5)
                if learnings:
                    memory_context = "\nRECENT LEARNINGS:\n" + "\n".join(learnings[-3:]) + "\n"

            integration_prompt = f"""You are the Integration Specialist. All agents have completed their work.

{final_summary}

FULL CONTEXT:
{self.context.get_context()}
{memory_context}

USER REQUEST: {user_request}

Create a comprehensive FINAL SUMMARY that:
1. Integrates all agent outputs
2. Highlights main deliverables and files created
3. Explains how components work together
4. Provides clear next steps
5. Notes dependencies or requirements

Keep it under 1000 words."""

            print("[GENERATING FINAL INTEGRATION SUMMARY]...")
            summary = llm_call(integration_prompt, max_tokens=2000, temperature=0.7)
            self.execution_stats["integration"] = round(time.time() - integration_start, 2)

            # Record learning
            self.context.add_learning(
                f"Completed project '{user_request[:50]}...' with {len(agent_names)} agents",
                "orchestrator"
            )

            overall_elapsed = round(time.time() - self.overall_start_time, 2)

            # Save memory
            print("[SAVING MEMORY TO DISK]...")
            self.context.save_memory()
            self.state_tracker.save_to_disk()
            print("[OK] Memory saved\n")

            # ===== PRINT FINAL REPORT =====
            self._print_execution_report(
                user_request,
                agent_names,
                final_summary,
                summary,
                overall_elapsed
            )

        except Exception as exc:
            logger.exception("Orchestration failed: %s", exc)
            print(f"\n[ERROR] FATAL ERROR: {exc}\n")
            raise

    def _print_execution_report(
        self,
        user_request: str,
        agents: List[str],
        project_summary: str,
        final_summary: str,
        total_time: float
    ) -> None:
        """Print comprehensive execution report."""
        
        print("=" * 80)
        print("EXECUTION REPORT")
        print("=" * 80)

        print(f"\n[REQUEST] {user_request}")

        print(f"\n[AGENTS] Agents Executed ({len(agents)} total):")
        for agent in agents:
            duration = self.execution_stats.get(agent, 0)
            status = "[OK]" if duration > 0 else "[X]"
            print(f"   {status} {agent}: {duration}s")

        print(f"\n[TIMING] Timing Breakdown:")
        print(f"   - Scope Analysis: {self.execution_stats.get('scope_analysis', 0)}s")
        print(f"   - Chain of Thought: {self.execution_stats.get('chain_of_thought', 0)}s")
        print(f"   - Prompt Engineering: {self.execution_stats.get('prompt_engineering', 0)}s")
        print(f"   - Integration: {self.execution_stats.get('integration', 0)}s")
        print(f"   - Total: {total_time}s")

        print(f"\n[FILES] Project Files:")
        print(f"   - Total Files: {len(self.file_manager.created_files)}")
        print(f"   - Edit History Entries: {len(self.file_manager.edit_history)}")

        if self.file_manager.created_files:
            print("\n   File List:")
            for file_path in sorted(self.file_manager.created_files):
                rel_path = file_path.replace(str(self.file_manager.base_path) + "\\", "")
                print(f"      - {rel_path}")

        print("\n[MEMORY] Memory Statistics:")
        if self.context.unified_memory:
            print(f"   - Short-term entries: {len(self.context.unified_memory.short_term)}")
            print(f"   - Long-term memories: {len(self.context.unified_memory.long_term.memories)}")
            # Vector store is optional - may be None if lancedb not installed
            vs = self.context.unified_memory.vector_store
            vs_count = len(vs) if vs is not None else 0
            print(f"   - Vector store entries: {vs_count}")
            print(f"   - Tasks tracked: {len(self.state_tracker.tasks)}")

        print("\n[STATE] System State:")
        print(self.state_tracker.get_system_summary())

        print("\n" + "=" * 80)
        print("FINAL SUMMARY")
        print("=" * 80)
        print(f"\n{final_summary}\n")

        print("=" * 80)
        print("ORCHESTRATION COMPLETE")
        print("=" * 80)
        print(f"\nAll files created in: {self.file_manager.base_path.absolute()}")
        print(f"Memory stored in: ./.agent_memory/{self.project_name}_memory.json")
        print(f"State stored in: ./.agent_memory/state/state.json\n")


# Convenient entry point
def orchestrator(user_request: str, output_folder: str = "./generated_project") -> None:
    """Run the unified orchestrator with planning + execution.
    
    Args:
        user_request: High-level user request
        output_folder: Output directory for generated files
    """
    unified_orchestrator = UnifiedOrchestrator(output_folder)
    unified_orchestrator.run(user_request)
