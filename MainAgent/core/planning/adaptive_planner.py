"""
Adaptive Planner - Dynamic re-planning based on execution feedback.

Capabilities:
- Update plans based on task success/failure
- Generate alternative approaches when tasks fail
- Re-prioritize remaining tasks based on new information
- Learn from execution patterns to improve future planning

This addresses a key gap: existing systems create static plans
without adapting to runtime feedback.
"""

from __future__ import annotations

import logging
import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set
import threading
import uuid

logger = logging.getLogger(__name__)


class ExecutionStatus(Enum):
    """Status of task execution."""
    SUCCESS = "success"
    FAILURE = "failure"
    PARTIAL = "partial"
    TIMEOUT = "timeout"
    SKIPPED = "skipped"


class AdaptationStrategy(Enum):
    """Strategies for adapting the plan."""
    RETRY = "retry"  # Retry same approach
    ALTERNATIVE = "alternative"  # Try different approach
    DECOMPOSE = "decompose"  # Break into smaller tasks
    ESCALATE = "escalate"  # Need human intervention
    SKIP = "skip"  # Skip and continue
    ABORT = "abort"  # Abort entire plan


@dataclass
class ExecutionResult:
    """Result of executing a task."""
    task_id: str
    status: ExecutionStatus
    output: str = ""
    error: str = ""
    duration_seconds: float = 0.0
    attempts: int = 1
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Task:
    """A task in the plan."""
    id: str
    description: str
    task_type: str
    dependencies: List[str] = field(default_factory=list)
    assigned_agent: str = ""
    priority: int = 0
    estimate: str = "medium"
    metadata: Dict[str, Any] = field(default_factory=dict)
    retry_count: int = 0
    max_retries: int = 3
    alternatives_tried: List[str] = field(default_factory=list)


@dataclass
class AdaptationResult:
    """Result of plan adaptation."""
    strategy: AdaptationStrategy
    updated_tasks: List[Task]
    removed_task_ids: List[str]
    added_tasks: List[Task]
    reason: str
    confidence: float = 0.0


@dataclass
class PlanState:
    """Current state of the plan."""
    tasks: List[Task]
    completed_tasks: Set[str] = field(default_factory=set)
    failed_tasks: Set[str] = field(default_factory=set)
    current_task_id: Optional[str] = None
    adaptation_history: List[AdaptationResult] = field(default_factory=list)


class AdaptivePlanner:
    """Plans and adapts task execution based on feedback.

    Key features:
    - Analyzes failure patterns to suggest alternatives
    - Decomposes failed tasks into smaller subtasks
    - Re-orders remaining tasks based on new dependencies
    - Learns from execution history
    """

    def __init__(
        self,
        llm_call: Optional[Callable[[str, Optional[int]], str]] = None,
        max_adaptations: int = 5
    ):
        """Initialize adaptive planner.

        Args:
            llm_call: Optional LLM function for intelligent adaptation
            max_adaptations: Maximum adaptations per task
        """
        self.llm_call = llm_call
        self.max_adaptations = max_adaptations
        self._lock = threading.Lock()
        self._execution_history: List[ExecutionResult] = []
        self._learned_patterns: Dict[str, List[str]] = {}  # failure_type -> solutions

    def create_plan(
        self,
        scope_info: Dict[str, Any]
    ) -> PlanState:
        """Create initial plan from scope information.

        Args:
            scope_info: Scope analysis result

        Returns:
            Initial PlanState
        """
        tasks = []
        raw_tasks = scope_info.get("tasks", [])

        for i, task_data in enumerate(raw_tasks):
            task = Task(
                id=task_data.get("id", f"task-{i+1}"),
                description=task_data.get("description", ""),
                task_type=task_data.get("type", "implementation"),
                dependencies=task_data.get("dependencies", []),
                priority=i,
                estimate=task_data.get("estimate", "medium"),
                metadata=task_data.get("metadata", {})
            )
            tasks.append(task)

        return PlanState(tasks=tasks)

    def adapt_plan(
        self,
        plan_state: PlanState,
        execution_result: ExecutionResult,
        context: Optional[Dict[str, Any]] = None
    ) -> AdaptationResult:
        """Adapt plan based on execution result.

        Args:
            plan_state: Current plan state
            execution_result: Result of task execution
            context: Additional context

        Returns:
            AdaptationResult describing changes
        """
        with self._lock:
            self._execution_history.append(execution_result)

        # Find the task
        task = self._find_task(plan_state, execution_result.task_id)
        if not task:
            return AdaptationResult(
                strategy=AdaptationStrategy.SKIP,
                updated_tasks=[],
                removed_task_ids=[],
                added_tasks=[],
                reason="Task not found in plan"
            )

        # Handle success
        if execution_result.status == ExecutionStatus.SUCCESS:
            plan_state.completed_tasks.add(task.id)
            return AdaptationResult(
                strategy=AdaptationStrategy.SKIP,
                updated_tasks=[],
                removed_task_ids=[],
                added_tasks=[],
                reason="Task completed successfully",
                confidence=1.0
            )

        # Handle failure - determine adaptation strategy
        strategy = self._determine_strategy(task, execution_result, plan_state)

        if strategy == AdaptationStrategy.RETRY:
            return self._handle_retry(task, execution_result, plan_state)

        elif strategy == AdaptationStrategy.ALTERNATIVE:
            return self._handle_alternative(task, execution_result, plan_state, context)

        elif strategy == AdaptationStrategy.DECOMPOSE:
            return self._handle_decompose(task, execution_result, plan_state, context)

        elif strategy == AdaptationStrategy.ESCALATE:
            return self._handle_escalate(task, execution_result, plan_state)

        elif strategy == AdaptationStrategy.ABORT:
            return self._handle_abort(task, execution_result, plan_state)

        else:  # SKIP
            return self._handle_skip(task, execution_result, plan_state)

    def _determine_strategy(
        self,
        task: Task,
        result: ExecutionResult,
        plan_state: PlanState
    ) -> AdaptationStrategy:
        """Determine the best adaptation strategy."""
        # Check retry eligibility
        if task.retry_count < task.max_retries:
            # Analyze if retry makes sense
            if result.status == ExecutionStatus.TIMEOUT:
                return AdaptationStrategy.RETRY
            if "temporary" in result.error.lower() or "network" in result.error.lower():
                return AdaptationStrategy.RETRY

        # Check if alternatives available
        if task.retry_count >= task.max_retries:
            if len(task.alternatives_tried) < 3:
                return AdaptationStrategy.ALTERNATIVE

        # Check if task is decomposable
        if self._is_decomposable(task, result):
            return AdaptationStrategy.DECOMPOSE

        # Check for blocking failures
        if self._is_blocking_failure(task, result, plan_state):
            return AdaptationStrategy.ESCALATE

        # Check adaptation limit
        if len(plan_state.adaptation_history) >= self.max_adaptations:
            return AdaptationStrategy.ABORT

        return AdaptationStrategy.SKIP

    def _is_decomposable(self, task: Task, result: ExecutionResult) -> bool:
        """Check if task can be decomposed."""
        # Large or complex tasks are decomposable
        if task.estimate in ("large", "complex"):
            return True

        # Tasks with multiple failure points
        if "multiple" in result.error.lower() or "and" in task.description.lower():
            return True

        return False

    def _is_blocking_failure(
        self,
        task: Task,
        result: ExecutionResult,
        plan_state: PlanState
    ) -> bool:
        """Check if failure blocks other tasks."""
        # Check if other tasks depend on this one
        for t in plan_state.tasks:
            if task.id in t.dependencies and t.id not in plan_state.completed_tasks:
                return True
        return False

    def _handle_retry(
        self,
        task: Task,
        result: ExecutionResult,
        plan_state: PlanState
    ) -> AdaptationResult:
        """Handle retry strategy."""
        task.retry_count += 1

        return AdaptationResult(
            strategy=AdaptationStrategy.RETRY,
            updated_tasks=[task],
            removed_task_ids=[],
            added_tasks=[],
            reason=f"Retrying task (attempt {task.retry_count + 1}/{task.max_retries})",
            confidence=0.7 - (task.retry_count * 0.1)
        )

    def _handle_alternative(
        self,
        task: Task,
        result: ExecutionResult,
        plan_state: PlanState,
        context: Optional[Dict[str, Any]]
    ) -> AdaptationResult:
        """Handle alternative approach strategy."""
        # Generate alternative approach
        alternative_task = self._generate_alternative(task, result, context)

        if alternative_task:
            task.alternatives_tried.append(task.description)
            alternative_task.id = f"{task.id}-alt-{len(task.alternatives_tried)}"

            return AdaptationResult(
                strategy=AdaptationStrategy.ALTERNATIVE,
                updated_tasks=[],
                removed_task_ids=[task.id],
                added_tasks=[alternative_task],
                reason=f"Trying alternative approach: {alternative_task.description[:50]}...",
                confidence=0.6
            )

        # Fallback to skip
        return self._handle_skip(task, result, plan_state)

    def _generate_alternative(
        self,
        task: Task,
        result: ExecutionResult,
        context: Optional[Dict[str, Any]]
    ) -> Optional[Task]:
        """Generate alternative task approach."""
        if not self.llm_call:
            return self._rule_based_alternative(task, result)

        prompt = f"""A task failed and needs an alternative approach.

ORIGINAL TASK: {task.description}
TASK TYPE: {task.task_type}
ERROR: {result.error}

Previous attempts that failed:
{chr(10).join(task.alternatives_tried) if task.alternatives_tried else "None"}

Generate an alternative approach that:
1. Achieves the same goal differently
2. Avoids the error that occurred
3. Is specific and actionable

Return JSON:
{{
    "description": "New task description",
    "approach": "How this differs from original",
    "confidence": 0.0-1.0
}}"""

        try:
            response = self.llm_call(prompt, 500)
            data = json.loads(re.search(r'\{[\s\S]*\}', response).group())

            return Task(
                id=f"{task.id}-alt",
                description=data.get("description", task.description),
                task_type=task.task_type,
                dependencies=task.dependencies,
                priority=task.priority,
                metadata={"alternative_of": task.id, "approach": data.get("approach", "")}
            )
        except Exception as e:
            logger.warning(f"Failed to generate alternative: {e}")
            return self._rule_based_alternative(task, result)

    def _rule_based_alternative(
        self,
        task: Task,
        result: ExecutionResult
    ) -> Optional[Task]:
        """Generate alternative using rules when LLM unavailable."""
        # Common alternative patterns
        alternatives = {
            "implementation": [
                "Use a simpler implementation approach",
                "Break down into smaller functions",
                "Use existing library/framework"
            ],
            "testing": [
                "Use integration test instead of unit test",
                "Mock external dependencies",
                "Test with smaller input set"
            ],
            "refactoring": [
                "Apply incremental refactoring",
                "Start with extracting methods",
                "Focus on single responsibility"
            ]
        }

        options = alternatives.get(task.task_type, ["Try different approach"])

        # Filter out already tried
        available = [o for o in options if o not in task.alternatives_tried]
        if not available:
            return None

        return Task(
            id=f"{task.id}-alt",
            description=f"{task.description} - {available[0]}",
            task_type=task.task_type,
            dependencies=task.dependencies,
            priority=task.priority,
            metadata={"alternative_of": task.id}
        )

    def _handle_decompose(
        self,
        task: Task,
        result: ExecutionResult,
        plan_state: PlanState,
        context: Optional[Dict[str, Any]]
    ) -> AdaptationResult:
        """Handle decomposition strategy."""
        subtasks = self._decompose_task(task, result, context)

        if subtasks:
            return AdaptationResult(
                strategy=AdaptationStrategy.DECOMPOSE,
                updated_tasks=[],
                removed_task_ids=[task.id],
                added_tasks=subtasks,
                reason=f"Decomposed into {len(subtasks)} subtasks",
                confidence=0.7
            )

        return self._handle_skip(task, result, plan_state)

    def _decompose_task(
        self,
        task: Task,
        result: ExecutionResult,
        context: Optional[Dict[str, Any]]
    ) -> List[Task]:
        """Decompose task into subtasks."""
        if not self.llm_call:
            return self._rule_based_decompose(task)

        prompt = f"""A complex task needs to be broken into smaller subtasks.

TASK: {task.description}
TYPE: {task.task_type}
ERROR WHEN ATTEMPTED: {result.error}

Break this into 2-4 smaller, more specific subtasks that:
1. Together accomplish the original goal
2. Each can be completed independently (with proper dependencies)
3. Are less likely to fail

Return JSON:
{{
    "subtasks": [
        {{
            "description": "Specific subtask description",
            "type": "implementation|testing|documentation",
            "estimate": "small|medium",
            "depends_on": []  // indices of other subtasks (0-based)
        }}
    ]
}}"""

        try:
            response = self.llm_call(prompt, 1000)
            data = json.loads(re.search(r'\{[\s\S]*\}', response).group())

            subtasks = []
            for i, st in enumerate(data.get("subtasks", [])):
                # Convert depends_on indices to actual IDs
                deps = task.dependencies.copy()
                for dep_idx in st.get("depends_on", []):
                    if dep_idx < i:
                        deps.append(f"{task.id}-sub-{dep_idx+1}")

                subtasks.append(Task(
                    id=f"{task.id}-sub-{i+1}",
                    description=st.get("description", ""),
                    task_type=st.get("type", task.task_type),
                    dependencies=deps,
                    priority=task.priority,
                    estimate=st.get("estimate", "small"),
                    metadata={"parent_task": task.id}
                ))

            return subtasks

        except Exception as e:
            logger.warning(f"Failed to decompose task: {e}")
            return self._rule_based_decompose(task)

    def _rule_based_decompose(self, task: Task) -> List[Task]:
        """Decompose task using rules when LLM unavailable."""
        # Simple 2-part decomposition
        return [
            Task(
                id=f"{task.id}-sub-1",
                description=f"Setup/prepare for: {task.description}",
                task_type=task.task_type,
                dependencies=task.dependencies,
                priority=task.priority,
                estimate="small",
                metadata={"parent_task": task.id}
            ),
            Task(
                id=f"{task.id}-sub-2",
                description=f"Implement/complete: {task.description}",
                task_type=task.task_type,
                dependencies=[f"{task.id}-sub-1"],
                priority=task.priority,
                estimate="small",
                metadata={"parent_task": task.id}
            )
        ]

    def _handle_escalate(
        self,
        task: Task,
        result: ExecutionResult,
        plan_state: PlanState
    ) -> AdaptationResult:
        """Handle escalation strategy."""
        plan_state.failed_tasks.add(task.id)

        return AdaptationResult(
            strategy=AdaptationStrategy.ESCALATE,
            updated_tasks=[],
            removed_task_ids=[],
            added_tasks=[],
            reason=f"Task requires human intervention: {result.error[:100]}",
            confidence=0.0
        )

    def _handle_abort(
        self,
        task: Task,
        result: ExecutionResult,
        plan_state: PlanState
    ) -> AdaptationResult:
        """Handle abort strategy."""
        return AdaptationResult(
            strategy=AdaptationStrategy.ABORT,
            updated_tasks=[],
            removed_task_ids=[t.id for t in plan_state.tasks
                            if t.id not in plan_state.completed_tasks],
            added_tasks=[],
            reason="Maximum adaptations exceeded, aborting plan",
            confidence=0.0
        )

    def _handle_skip(
        self,
        task: Task,
        result: ExecutionResult,
        plan_state: PlanState
    ) -> AdaptationResult:
        """Handle skip strategy."""
        plan_state.failed_tasks.add(task.id)

        # Remove dependent tasks
        to_remove = self._find_dependent_tasks(task.id, plan_state)

        return AdaptationResult(
            strategy=AdaptationStrategy.SKIP,
            updated_tasks=[],
            removed_task_ids=to_remove,
            added_tasks=[],
            reason=f"Skipping task and {len(to_remove)-1} dependent tasks" if len(to_remove) > 1 else "Skipping task",
            confidence=0.5
        )

    def _find_task(self, plan_state: PlanState, task_id: str) -> Optional[Task]:
        """Find task by ID."""
        for task in plan_state.tasks:
            if task.id == task_id:
                return task
        return None

    def _find_dependent_tasks(
        self,
        task_id: str,
        plan_state: PlanState
    ) -> List[str]:
        """Find all tasks that depend on the given task."""
        dependent = {task_id}
        changed = True

        while changed:
            changed = False
            for task in plan_state.tasks:
                if task.id not in dependent:
                    if any(dep in dependent for dep in task.dependencies):
                        dependent.add(task.id)
                        changed = True

        return list(dependent)

    def get_next_task(self, plan_state: PlanState) -> Optional[Task]:
        """Get the next task to execute."""
        for task in sorted(plan_state.tasks, key=lambda t: t.priority):
            if task.id in plan_state.completed_tasks:
                continue
            if task.id in plan_state.failed_tasks:
                continue

            # Check dependencies
            deps_met = all(
                dep in plan_state.completed_tasks
                for dep in task.dependencies
            )
            if deps_met:
                return task

        return None

    def get_stats(self) -> Dict[str, Any]:
        """Get planner statistics."""
        with self._lock:
            if not self._execution_history:
                return {"total_executions": 0}

            return {
                "total_executions": len(self._execution_history),
                "successful": sum(1 for r in self._execution_history
                                 if r.status == ExecutionStatus.SUCCESS),
                "failed": sum(1 for r in self._execution_history
                             if r.status == ExecutionStatus.FAILURE),
                "total_retries": sum(r.attempts - 1 for r in self._execution_history)
            }


def create_adaptive_planner(
    llm_call: Optional[Callable[[str, Optional[int]], str]] = None,
    max_adaptations: int = 5
) -> AdaptivePlanner:
    """Factory function to create adaptive planner."""
    return AdaptivePlanner(llm_call, max_adaptations)
