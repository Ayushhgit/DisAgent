"""Task Scheduler with queue-based execution.

This module provides an efficient, non-polling task scheduler that uses
condition variables for synchronization instead of busy-wait loops.
"""

from __future__ import annotations

import time
import logging
import threading
from queue import Queue, Empty
from concurrent.futures import ThreadPoolExecutor, Future
from typing import Dict, Optional, Set, Callable
from dataclasses import dataclass

from core.context.state_tracker import StateTracker
from core.planning.task_planner import TaskPlanner
from core.runtime.agent import Agent, AgentResult
from core.runtime.event_bus import EventType, subscribe_to_events, Event

logger = logging.getLogger(__name__)


@dataclass
class TaskExecution:
    """Represents a task being executed."""
    task_id: str
    agent_name: str
    start_time: float
    future: Future


class TaskScheduler:
    """Queue-based scheduler for concurrent agent execution.

    This scheduler uses:
    - A work queue for pending tasks
    - Condition variable for efficient waiting
    - Event-driven task completion handling
    - Back-pressure via bounded queue
    """

    def __init__(
        self,
        state: StateTracker,
        planner: TaskPlanner,
        max_workers: int = 4,
        max_queue_size: int = 100,
    ):
        """Initialize the scheduler.

        Args:
            state: StateTracker for task/agent state
            planner: TaskPlanner for task assignment
            max_workers: Maximum concurrent agent executions
            max_queue_size: Maximum pending tasks (back-pressure)
        """
        self.state = state
        self.planner = planner
        self.max_workers = max_workers

        # Work queue with optional size limit for back-pressure
        self._work_queue: Queue = Queue(maxsize=max_queue_size)

        # Condition variable for signaling new work / completions
        self._condition = threading.Condition()

        # Track running executions
        self._running: Dict[str, TaskExecution] = {}
        self._running_lock = threading.Lock()

        # Shutdown flag
        self._shutdown = False

        # Subscribe to task completion events for dependency unblocking
        subscribe_to_events(EventType.TASK_COMPLETED, self._on_task_completed)
        subscribe_to_events(EventType.TASK_FAILED, self._on_task_failed)

    def _on_task_completed(self, event: Event) -> None:
        """Handle task completion - check for newly unblocked tasks."""
        with self._condition:
            self._condition.notify_all()

    def _on_task_failed(self, event: Event) -> None:
        """Handle task failure - may unblock dependent tasks to fail."""
        with self._condition:
            self._condition.notify_all()

    def run(
        self,
        agents: Dict[str, Agent],
        prompts: Dict[str, str],
        context,
        file_manager,
        timeout: Optional[float] = None,
    ) -> Dict[str, float]:
        """Run scheduling until all tasks complete.

        Args:
            agents: Mapping agent_name -> Agent instance
            prompts: Mapping agent_name -> prompt string
            context: Shared context object
            file_manager: FileManager for file operations
            timeout: Optional timeout in seconds for entire run

        Returns:
            execution_stats: Mapping agent_name -> elapsed_seconds
        """
        execution_stats: Dict[str, float] = {}
        worker_agent_names = list(agents.keys())
        logger.debug("Worker agents available: %s", worker_agent_names)

        start_time = time.time()
        self._shutdown = False

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            while not self._shutdown:
                # Check timeout
                if timeout and (time.time() - start_time) > timeout:
                    logger.warning("Scheduler timeout reached after %.2fs", timeout)
                    break

                # Get ready tasks
                ready_tasks = self.planner.get_ready_tasks()
                logger.debug("Ready tasks: %s, Running: %d", ready_tasks, len(self._running))

                # Check termination: no ready tasks and nothing running
                with self._running_lock:
                    if not ready_tasks and not self._running:
                        logger.info("All tasks completed, scheduler exiting")
                        break

                # Submit ready tasks
                tasks_submitted = 0
                for task_id in ready_tasks:
                    # Skip if already running
                    with self._running_lock:
                        if task_id in self._running:
                            continue

                    # Try to assign to best agent
                    assigned = self.planner.assign_task_to_best_agent(
                        task_id, preferred_agents=worker_agent_names
                    )

                    if not assigned:
                        continue

                    # Get assignment details
                    task_info = self.state.tasks.get(task_id)
                    if not task_info:
                        logger.warning("Task %s not found after assignment", task_id)
                        continue

                    agent_name = task_info.assigned_agent
                    agent = agents.get(agent_name)
                    if not agent:
                        logger.warning(
                            "Agent %s not in agents map (available: %s)",
                            agent_name, worker_agent_names
                        )
                        continue

                    prompt = prompts.get(
                        agent_name,
                        f"You are {agent_name}. Complete task {task_id}"
                    )

                    # Submit to executor
                    logger.debug("Submitting task %s to agent %s", task_id, agent_name)
                    future = executor.submit(
                        self._run_agent, agent, task_id, prompt, context, file_manager
                    )

                    # Track execution
                    with self._running_lock:
                        self._running[task_id] = TaskExecution(
                            task_id=task_id,
                            agent_name=agent_name,
                            start_time=time.time(),
                            future=future,
                        )

                    # Add completion callback
                    future.add_done_callback(
                        lambda f, tid=task_id: self._handle_completion(
                            tid, f, execution_stats
                        )
                    )
                    tasks_submitted += 1

                # Wait for new work or completion
                # Use condition variable instead of sleep
                if not tasks_submitted:
                    with self._condition:
                        # Wait with timeout to periodically check for new ready tasks
                        self._condition.wait(timeout=0.1)

        return execution_stats

    def _handle_completion(
        self,
        task_id: str,
        future: Future,
        execution_stats: Dict[str, float],
    ) -> None:
        """Handle agent task completion."""
        with self._running_lock:
            execution = self._running.pop(task_id, None)

        if not execution:
            logger.warning("Completion for unknown task: %s", task_id)
            return

        elapsed = round(time.time() - execution.start_time, 2)
        execution_stats[execution.agent_name] = elapsed

        try:
            result: AgentResult = future.result()
            if result.success:
                self.state.complete_task(task_id, f"Completed in {elapsed}s")
                logger.info("Task %s completed by %s in %.2fs",
                           task_id, execution.agent_name, elapsed)
            else:
                self.state.fail_task(task_id, result.output)
                logger.warning("Task %s failed: %s", task_id, result.output[:200])
        except Exception as exc:
            logger.exception("Agent %s crashed on task %s: %s",
                           execution.agent_name, task_id, exc)
            self.state.fail_task(task_id, str(exc))

        # Signal that work state changed
        with self._condition:
            self._condition.notify_all()

    def shutdown(self) -> None:
        """Signal scheduler to stop processing."""
        self._shutdown = True
        with self._condition:
            self._condition.notify_all()

    def get_running_tasks(self) -> Set[str]:
        """Get set of currently running task IDs."""
        with self._running_lock:
            return set(self._running.keys())

    @staticmethod
    def _run_agent(
        agent: Agent,
        task_id: str,
        prompt: str,
        context,
        file_manager,
    ) -> AgentResult:
        """Execute agent and return result."""
        try:
            return agent.run(task_id, prompt, context, file_manager)
        except Exception as exc:
            logger.exception("Agent execution error: %s", exc)
            return AgentResult(success=False, output=str(exc), metadata={})
