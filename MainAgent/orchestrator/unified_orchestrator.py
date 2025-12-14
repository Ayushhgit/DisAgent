"""Unified orchestrator combining planning and execution.

This orchestrator integrates:
  - planning layer: ScopeAnalyzer → ChainOfThoughtProcessor → TaskPlanner → StateTracker
  - execution layer: dynamic agent spawning and file editing
  - memory system: unified context management
  - verification loop: closed-loop validation with test runner
  - agent communication: point-to-point and channel-based messaging
  - critic agent: output review before acceptance
  - adaptive planner: re-planning on failure
  - conflict resolver: concurrent edit merging
  - experience learner: incremental learning from history
  - execution tracer: structured observability
  - intent detection: distinguish questions from code generation

Architecture:
  User Request → Intent Detection → Scope Analysis → Task Planning → Agent Execution → Critic Review → Verification → Integration
"""

from __future__ import annotations

import os
import time
import logging
from typing import Dict, List, Optional
from pathlib import Path
from datetime import datetime

from .context import AgentContext, create_context
from .dynamic_agents import create_dynamic_agent
from .file_manager import FileManager
from .scope_prompting import prompt_engineer_agent
from .execution_tracer import ExecutionTracer, SpanKind, create_tracer

from core.planning.task_planner import TaskPlanner
from core.planning.scope_analyzer import ScopeAnalyzer
from core.planning.chain_of_thought import ChainOfThoughtProcessor
from core.planning.validation import PlanValidator
from core.planning.adaptive_planner import AdaptivePlanner, create_adaptive_planner
from core.context.state_tracker import StateTracker
from core.memory.memory_types import TaskInfo, TaskType, AgentState, MemoryPriority
from core.memory.experience_learner import ExperienceLearner, create_experience_learner
from core.runtime.llm import llm_call
from core.runtime.agent import AgentResult, Agent
from core.runtime.scheduler import TaskScheduler
from core.runtime.critic import CriticAgent, create_critic, ApprovalStatus
from core.runtime.conflict_resolver import ConflictResolver, create_conflict_resolver

# New research-driven components
from core.runtime.verification_loop import (
    VerificationLoop,
    VerificationConfig,
    create_verification_loop
)
from core.runtime.agent_communication import (
    get_communication_bus,
    AgentCommunicator
)
from core.planning.semantic_scope_analyzer import (
    HybridScopeAnalyzer,
    SemanticScopeAnalyzer
)

logger = logging.getLogger(__name__)


# ===== INTENT DETECTION =====
class UserIntent:
    """Represents the detected intent of a user request."""
    QUESTION = "question"  # User asking about existing code/how to run
    GENERATION = "generation"  # User wants new code generated
    MODIFICATION = "modification"  # User wants existing code modified
    HELP = "help"  # User asking for help/documentation


def detect_intent(user_request: str, project_path: Optional[str] = None) -> tuple:
    """Detect whether the user is asking a question or requesting code generation.

    Args:
        user_request: The user's request string
        project_path: Optional path to check for existing project

    Returns:
        Tuple of (intent_type, is_about_existing_project)
    """
    request_lower = user_request.lower().strip()

    # Generation/action patterns - these OVERRIDE question detection
    # If user wants to build/create something, it's generation even with "?"
    generation_patterns = [
        "build ", "create ", "make ", "implement ", "develop ", "write ",
        "add a ", "add the ", "generate ", "setup ", "set up ",
        "install ", "configure ", "deploy ", "fix ", "update ", "modify ",
        "refactor ", "optimize ", "improve ", "enhance ", "extend ",
    ]

    # Check if this is clearly a generation/action request
    is_generation_request = any(request_lower.startswith(p) or f" {p}" in request_lower
                                 for p in generation_patterns)

    # Question patterns - these should NOT trigger code regeneration
    # Only match if NOT a generation request
    question_patterns = [
        "how do i run", "how to run", "how can i run", "how do you run",
        "how to start", "how do i start", "how to execute", "how to use",
        "how does it work", "how does this work", "how does the",
        "what is the", "what are the", "what does the", "what does it",
        "where is the", "where are the", "where can i find",
        "can you explain", "explain how", "explain the", "tell me about",
        "describe the", "describe how",
        "show me how to run", "help me understand",
        "what's the command", "whats the command",
        "is there a way to run", "are there any instructions",
        "does this have", "do i need to",
        "why is the", "why does the", "why are the",
        "when should i", "which file contains", "which folder has",
        "which command", "instructions for running", "guide for running",
    ]

    # Help patterns - more specific to avoid matching "help me build"
    help_patterns = [
        "show me the readme", "read the readme", "documentation for",
        "getting started guide", "quickstart guide", "setup guide",
        "how to run this", "how do i run this", "how to start this",
        "run instructions", "running instructions", "execution instructions",
    ]

    # Check for question patterns (but not if it's clearly a generation request)
    is_question = False
    is_help = False

    if not is_generation_request:
        is_question = any(pattern in request_lower for pattern in question_patterns)
        is_help = any(pattern in request_lower for pattern in help_patterns)

    # Check if referring to existing project/app
    existing_project_patterns = [
        "this app", "this project", "the app i", "the project i",
        "you made", "you created", "you built", "you generated",
        "we made", "we created", "the existing", "the current",
        "my app", "my project", "our app", "our project",
        "app you made", "project you made", "app you created",
    ]
    is_about_existing = any(pattern in request_lower for pattern in existing_project_patterns)

    # Also check if project directory has existing files
    has_existing_files = False
    if project_path:
        try:
            project_dir = Path(project_path)
            if project_dir.exists():
                # Count actual code files (not just any files)
                code_extensions = {'.py', '.js', '.ts', '.jsx', '.tsx', '.html', '.css', '.json', '.yaml', '.yml'}
                code_files = [f for f in project_dir.rglob('*') if f.is_file() and f.suffix in code_extensions]
                has_existing_files = len(code_files) > 0
        except Exception:
            pass

    # Determine intent with priority:
    # 1. Explicit generation patterns always win
    # 2. Help/question only if about existing project AND has question patterns
    # 3. Modification if about existing project but no question patterns
    # 4. Generation as default

    if is_generation_request:
        # User explicitly wants to build/create something
        if is_about_existing or has_existing_files:
            return UserIntent.MODIFICATION, True
        return UserIntent.GENERATION, False

    if is_help and (is_about_existing or has_existing_files):
        return UserIntent.HELP, True
    elif is_question and (is_about_existing or has_existing_files):
        return UserIntent.QUESTION, True
    elif is_about_existing or has_existing_files:
        return UserIntent.MODIFICATION, True
    else:
        return UserIntent.GENERATION, False


def handle_question_intent(
    user_request: str,
    project_path: str,
    llm_call_func
) -> str:
    """Handle question/help intent by providing information instead of generating code.

    Args:
        user_request: The user's question
        project_path: Path to the project
        llm_call_func: LLM function to call

    Returns:
        Answer to the user's question
    """
    from .file_manager import FileManager

    file_manager = FileManager(project_path)
    project_structure = file_manager.get_project_structure_tree()
    available_files = file_manager.list_files("*")

    # Build context about the project
    files_context = "PROJECT STRUCTURE:\n" + project_structure + "\n\n"

    # Read key files for context (README, main files, package files)
    key_files = []
    for pattern in ['README*', 'readme*', 'package.json', 'requirements.txt', 'main.py', 'app.py', 'index.*']:
        matches = [f for f in available_files if Path(f).match(pattern)]
        key_files.extend(matches[:2])  # Limit per pattern

    if key_files:
        files_context += "KEY FILES CONTENT:\n"
        for key_file in key_files[:5]:  # Limit total
            content = file_manager.read_file(key_file)
            if content:
                files_context += f"\n--- {key_file} ---\n{content[:1500]}\n"

    prompt = f"""You are a helpful assistant answering questions about an existing project.

USER QUESTION: {user_request}

{files_context}

INSTRUCTIONS:
1. Answer the user's question based on the project structure and files shown above
2. If asking "how to run", look for:
   - README files for instructions
   - package.json for npm scripts
   - requirements.txt for Python dependencies
   - Main entry points (main.py, app.py, index.js, etc.)
3. Provide clear, step-by-step instructions when applicable
4. Do NOT generate new code - just answer the question
5. If the information isn't available, say so and suggest what they might need

Provide a helpful, concise answer:"""

    return llm_call_func(prompt, max_tokens=1500, temperature=0.3)


def analyze_scope(user_request: str, use_semantic: bool = True, project_path: Optional[str] = None) -> Dict:
    """Wrapper function to analyze scope from user request.

    Args:
        user_request: The user's request string
        use_semantic: Whether to use LLM-powered semantic analysis
        project_path: Optional path to project for file scanning

    Returns a validated scope dict with all required fields.
    Raises RuntimeError if scope analysis fails or returns invalid data.
    """
    # Try semantic analyzer if enabled
    if use_semantic:
        try:
            # Create LLM wrapper that matches expected signature
            def llm_wrapper(prompt: str, max_tokens: Optional[int] = None) -> str:
                return llm_call(prompt, max_tokens=max_tokens or 2000)

            analyzer = HybridScopeAnalyzer(
                llm_call=llm_wrapper,
                project_path=project_path,
                use_semantic=True
            )
            scope = analyzer.analyze(user_request, repo_path=project_path)
            logger.info("Used semantic scope analysis")
        except Exception as e:
            logger.warning(f"Semantic analysis failed, falling back to keyword: {e}")
            analyzer = ScopeAnalyzer()
            scope = analyzer.analyze(user_request)
    else:
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

    def __init__(
        self,
        output_folder: str = "./generated_project",
        enable_verification: bool = True,
        enable_semantic_analysis: bool = True,
        enable_critic: bool = True,
        enable_tracing: bool = True,
        enable_learning: bool = True,
        run_tests: bool = True,
        max_verification_retries: int = 3,
        critic_strictness: str = "normal"
    ):
        """Initialize the orchestrator with file manager and state tracking.

        Args:
            output_folder: Base output directory for generated files
            enable_verification: Whether to run verification loop on edits
            enable_semantic_analysis: Whether to use LLM-powered scope analysis
            enable_critic: Whether to enable critic agent for output review
            enable_tracing: Whether to enable execution tracing
            enable_learning: Whether to enable experience learning
            run_tests: Whether to run tests during verification
            max_verification_retries: Max retries for verification failures
            critic_strictness: Strictness level for critic ("lenient", "normal", "strict")
        """
        self.output_folder = output_folder
        self.file_manager = FileManager(output_folder)
        self.state_tracker = StateTracker(main_agent_id="main_orchestrator")
        self.task_planner = TaskPlanner(self.state_tracker, PlanValidator())
        self.cot_processor = ChainOfThoughtProcessor()

        self.project_name = Path(output_folder).name
        self.context = create_context(self.project_name)

        # Configuration
        self.enable_verification = enable_verification
        self.enable_semantic_analysis = enable_semantic_analysis
        self.enable_critic = enable_critic
        self.enable_tracing = enable_tracing
        self.enable_learning = enable_learning

        # Initialize execution tracer
        if enable_tracing:
            self.tracer = create_tracer(
                storage_path=f".agent_memory/traces/{self.project_name}",
                enable_console=True
            )
        else:
            self.tracer = None

        # Initialize verification loop
        if enable_verification:
            verification_config = VerificationConfig(
                max_retries=max_verification_retries,
                run_tests=run_tests,
                auto_correct=True,
                rollback_on_failure=True
            )
            self.verification_loop = create_verification_loop(
                output_folder,
                config=verification_config,
                llm_call=llm_call
            )
        else:
            self.verification_loop = None

        # Initialize critic agent
        if enable_critic:
            def llm_wrapper(prompt: str, max_tokens: Optional[int] = None) -> str:
                return llm_call(prompt, max_tokens=max_tokens or 2000)
            self.critic = create_critic(llm_wrapper, strictness=critic_strictness)
        else:
            self.critic = None

        # Initialize adaptive planner
        def llm_wrapper_adaptive(prompt: str, max_tokens: Optional[int] = None) -> str:
            return llm_call(prompt, max_tokens=max_tokens or 1500)
        self.adaptive_planner = create_adaptive_planner(llm_wrapper_adaptive)

        # Initialize conflict resolver
        self.conflict_resolver = create_conflict_resolver(llm_wrapper_adaptive)

        # Initialize experience learner
        if enable_learning:
            self.experience_learner = create_experience_learner(
                storage_path=f".agent_memory/experience/{self.project_name}"
            )
        else:
            self.experience_learner = None

        # Initialize agent communication bus
        self.communication_bus = get_communication_bus()

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

            # ===== STAGE 0: INTENT DETECTION =====
            print("\n" + "─" * 80)
            print("[STAGE 0: INTENT DETECTION]")
            print("─" * 80)

            intent_type, is_about_existing = detect_intent(user_request, self.output_folder)
            print(f"Detected Intent: {intent_type}")
            print(f"About Existing Project: {is_about_existing}")

            # Handle question/help intents without code generation
            if intent_type in (UserIntent.QUESTION, UserIntent.HELP) and is_about_existing:
                print("\n[INFO] Detected question about existing project - providing answer without code generation\n")
                answer = handle_question_intent(user_request, self.output_folder, llm_call)
                print("\n" + "=" * 80)
                print("ANSWER")
                print("=" * 80)
                print(f"\n{answer}\n")
                print("=" * 80)
                return  # Exit early - no code generation needed

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

            scope_info = analyze_scope(
                user_request,
                use_semantic=self.enable_semantic_analysis,
                project_path=self.output_folder
            )
            self.execution_stats["scope_analysis"] = round(time.time() - scope_start, 2)
            self.context.set_scope(scope_info)

            analysis_mode = "semantic" if self.enable_semantic_analysis else "keyword"
            print(f"Analysis Mode: {analysis_mode}")
            print(f"Scope Level: {scope_info.get('scope_level', 'unknown')}")
            print(f"Complexity: {scope_info.get('complexity', 'unknown')}")
            print(f"Domains: {', '.join(scope_info.get('domains', []))}")
            if scope_info.get('risk_level'):
                print(f"Risk Level: {scope_info.get('risk_level')}")
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

            # Validate agent_names - must have at least one agent
            if not agent_names:
                logger.warning("No agents specified by scope analyzer, using default 'generalist'")
                agent_names = ["generalist"]
                scope_info["agents_needed"] = agent_names

            custom_prompts = prompt_engineer_agent(user_request, agent_names, self.context)
            self.execution_stats["prompt_engineering"] = round(time.time() - prompt_start, 2)
            
            print(f"Engineered prompts for {len(agent_names)} agents: {', '.join(agent_names)}")
            print(f"Time: {self.execution_stats['prompt_engineering']}s\n")

            # ===== STAGE 4: REGISTER AGENTS & CREATE TASKS IN StateTracker =====
            print("\n" + "─" * 80)
            print("[STAGE 4: TASK REGISTRATION & AGENT SETUP]")
            print("─" * 80)
            
            # Register all agents in StateTracker and Communication Bus
            for agent_name in agent_names:
                self.state_tracker.register_agent(agent_name)
                self.context.register_agent(agent_name)
                # Register agent in communication bus
                self.communication_bus.register_agent(agent_name)
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

            # Reset execution stats for this run
            from .dynamic_agents import get_execution_stats, reset_execution_stats
            reset_execution_stats()

            # Thin Agent wrapper around the existing create_dynamic_agent function
            # DynamicAgent now uses relevance selector to choose files
            from core.runtime.relevance import select_relevant_files

            # Capture critic reference for closure
            critic_ref = self.critic
            verification_ref = self.verification_loop

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
                    output = create_dynamic_agent(
                        self.name, prompt, user_request, context, file_manager,
                        allowed_files=relevant,
                        critic_agent=critic_ref,
                        verification_loop=verification_ref
                    )
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

        # Agent execution statistics
        from .dynamic_agents import get_execution_stats
        agent_stats = get_execution_stats()
        print(f"\n[EXECUTION] Agent Execution Statistics:")
        print(f"   - Total Agent Calls: {agent_stats.get('total_agent_calls', 0)}")
        print(f"   - Successful Calls: {agent_stats.get('successful_calls', 0)}")
        print(f"   - Failed Calls: {agent_stats.get('failed_calls', 0)}")
        print(f"   - Files Created: {agent_stats.get('files_created', 0)}")
        print(f"   - Files Edited: {agent_stats.get('files_edited', 0)}")
        print(f"   - Edits Failed: {agent_stats.get('edits_failed', 0)}")

        if agent_stats.get('agent_durations'):
            print(f"   - Agent Durations:")
            for agent, duration in agent_stats['agent_durations'].items():
                print(f"      - {agent}: {duration:.2f}s")

        print(f"\n[FILES] Project Files:")
        print(f"   - Total Files: {len(self.file_manager.created_files)}")
        print(f"   - Edit History Entries: {len(self.file_manager.edit_history)}")

        if self.file_manager.created_files:
            print("\n   File List:")
            for file_path in sorted(self.file_manager.created_files):
                # Handle both Windows and Unix path separators
                base_str = str(self.file_manager.base_path)
                rel_path = file_path.replace(base_str + os.sep, "").replace(base_str + "/", "").replace(base_str + "\\", "")
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

        # Verification statistics
        if self.verification_loop:
            print("\n[VERIFICATION] Verification Statistics:")
            v_stats = self.verification_loop.get_stats()
            print(f"   - Total verifications: {v_stats.get('total_verifications', 0)}")
            print(f"   - Successful: {v_stats.get('successful', 0)}")
            print(f"   - Failed: {v_stats.get('failed', 0)}")
            print(f"   - Retries: {v_stats.get('retries', 0)}")
            print(f"   - Rollbacks: {v_stats.get('rollbacks', 0)}")

        # Communication statistics
        print("\n[COMMUNICATION] Agent Communication Statistics:")
        comm_stats = self.communication_bus.get_stats()
        print(f"   - Registered agents: {comm_stats.get('registered_agents', 0)}")
        print(f"   - Active channels: {comm_stats.get('channels', 0)}")
        print(f"   - Total deliveries: {comm_stats.get('total_deliveries', 0)}")
        print(f"   - Successful: {comm_stats.get('successful_deliveries', 0)}")
        print(f"   - Failed: {comm_stats.get('failed_deliveries', 0)}")

        # Critic statistics
        if self.critic:
            print("\n[CRITIC] Code Review Statistics:")
            critic_stats = self.critic.get_stats()
            print(f"   - Total reviews: {critic_stats.get('total_reviews', 0)}")
            print(f"   - Approved: {critic_stats.get('approved', 0)}")
            print(f"   - Rejected: {critic_stats.get('rejected', 0)}")
            print(f"   - Average score: {critic_stats.get('average_score', 0):.2f}")

        # Adaptive planner statistics
        print("\n[PLANNING] Adaptive Planner Statistics:")
        planner_stats = self.adaptive_planner.get_stats()
        print(f"   - Total executions: {planner_stats.get('total_executions', 0)}")
        print(f"   - Successful: {planner_stats.get('successful', 0)}")
        print(f"   - Failed: {planner_stats.get('failed', 0)}")
        print(f"   - Total retries: {planner_stats.get('total_retries', 0)}")

        # Conflict resolver statistics
        print("\n[CONFLICTS] Conflict Resolution Statistics:")
        conflict_stats = self.conflict_resolver.get_stats()
        print(f"   - Total resolutions: {conflict_stats.get('total_resolutions', 0)}")
        print(f"   - Successful: {conflict_stats.get('successful', 0)}")
        print(f"   - Failed: {conflict_stats.get('failed', 0)}")

        # Experience learner statistics
        if self.experience_learner:
            print("\n[LEARNING] Experience Learning Statistics:")
            learning_summary = self.experience_learner.get_learning_summary()
            print(f"   - Total experiences: {learning_summary.get('total_experiences', 0)}")
            print(f"   - Agents tracked: {learning_summary.get('agents_tracked', 0)}")
            print(f"   - Failure patterns: {learning_summary.get('failure_patterns_identified', 0)}")
            print(f"   - Task types learned: {learning_summary.get('task_types_learned', 0)}")

        print("\n[STATE] System State:")
        print(self.state_tracker.get_system_summary())

        # Print trace summary if tracing enabled
        if self.tracer:
            self.tracer.print_summary()
            trace_path = self.tracer.save_trace()
            print(f"Trace saved to: {trace_path}")

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
def orchestrator(
    user_request: str,
    output_folder: str = "./generated_project",
    enable_verification: bool = True,
    enable_semantic_analysis: bool = True,
    enable_critic: bool = True,
    enable_tracing: bool = True,
    enable_learning: bool = True,
    run_tests: bool = True,
    critic_strictness: str = "normal"
) -> None:
    """Run the unified orchestrator with planning + execution.

    Args:
        user_request: High-level user request
        output_folder: Output directory for generated files
        enable_verification: Whether to run verification loop on edits
        enable_semantic_analysis: Whether to use LLM-powered scope analysis
        enable_critic: Whether to enable critic agent for output review
        enable_tracing: Whether to enable execution tracing
        enable_learning: Whether to enable experience learning
        run_tests: Whether to run tests during verification
        critic_strictness: Strictness level for critic ("lenient", "normal", "strict")
    """
    unified_orchestrator = UnifiedOrchestrator(
        output_folder,
        enable_verification=enable_verification,
        enable_semantic_analysis=enable_semantic_analysis,
        enable_critic=enable_critic,
        enable_tracing=enable_tracing,
        enable_learning=enable_learning,
        run_tests=run_tests,
        critic_strictness=critic_strictness
    )
    unified_orchestrator.run(user_request)
