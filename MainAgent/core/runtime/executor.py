# ============================================================================
# FILE: executor.py
# Parallel execution with dependency resolution and graceful shutdown
# ============================================================================

from __future__ import annotations

import time
import signal
import threading
import logging
import atexit
from typing import Dict, List, Optional, Callable, Any, Set
from dataclasses import dataclass, field
from enum import Enum
from concurrent.futures import ThreadPoolExecutor, Future, as_completed, wait, FIRST_COMPLETED
from collections import defaultdict
import queue

logger = logging.getLogger(__name__)


class TaskState(Enum):
    """State of a task in the executor."""
    PENDING = "pending"
    READY = "ready"  # Dependencies satisfied
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    SKIPPED = "skipped"


@dataclass
class ExecutorTask:
    """A task to be executed with dependencies."""
    id: str
    func: Callable
    args: tuple = field(default_factory=tuple)
    kwargs: dict = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)
    priority: int = 0  # Lower = higher priority
    timeout: Optional[float] = None
    state: TaskState = TaskState.PENDING
    result: Any = None
    error: Optional[Exception] = None
    start_time: Optional[float] = None
    end_time: Optional[float] = None

    @property
    def duration(self) -> Optional[float]:
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return None


class DependencyResolver:
    """Resolves task dependencies and determines execution order."""

    def __init__(self):
        self._tasks: Dict[str, ExecutorTask] = {}
        self._dependents: Dict[str, Set[str]] = defaultdict(set)  # task -> tasks that depend on it
        self._lock = threading.Lock()

    def add_task(self, task: ExecutorTask):
        """Add a task with its dependencies."""
        with self._lock:
            self._tasks[task.id] = task

            # Track reverse dependencies
            for dep_id in task.dependencies:
                self._dependents[dep_id].add(task.id)

    def get_ready_tasks(self) -> List[ExecutorTask]:
        """Get tasks that are ready to execute (dependencies satisfied)."""
        with self._lock:
            ready = []
            for task_id, task in self._tasks.items():
                if task.state != TaskState.PENDING:
                    continue

                # Check if all dependencies are completed
                deps_satisfied = all(
                    self._tasks.get(dep_id) and
                    self._tasks[dep_id].state == TaskState.COMPLETED
                    for dep_id in task.dependencies
                )

                if deps_satisfied:
                    task.state = TaskState.READY
                    ready.append(task)

            # Sort by priority
            ready.sort(key=lambda t: t.priority)
            return ready

    def mark_completed(self, task_id: str, result: Any = None):
        """Mark a task as completed."""
        with self._lock:
            if task_id in self._tasks:
                task = self._tasks[task_id]
                task.state = TaskState.COMPLETED
                task.result = result
                task.end_time = time.time()

    def mark_failed(self, task_id: str, error: Exception):
        """Mark a task as failed."""
        with self._lock:
            if task_id in self._tasks:
                task = self._tasks[task_id]
                task.state = TaskState.FAILED
                task.error = error
                task.end_time = time.time()

                # Mark dependent tasks as skipped
                self._skip_dependents(task_id)

    def _skip_dependents(self, task_id: str):
        """Recursively skip tasks that depend on a failed task."""
        for dep_id in self._dependents.get(task_id, []):
            if dep_id in self._tasks and self._tasks[dep_id].state == TaskState.PENDING:
                self._tasks[dep_id].state = TaskState.SKIPPED
                self._skip_dependents(dep_id)

    def has_pending_tasks(self) -> bool:
        """Check if there are pending tasks."""
        with self._lock:
            return any(
                t.state in (TaskState.PENDING, TaskState.READY, TaskState.RUNNING)
                for t in self._tasks.values()
            )

    def get_stats(self) -> Dict[str, int]:
        """Get task statistics by state."""
        with self._lock:
            stats = defaultdict(int)
            for task in self._tasks.values():
                stats[task.state.value] += 1
            return dict(stats)


class ParallelExecutor:
    """Executes tasks in parallel with dependency resolution and graceful shutdown.

    Features:
    - Parallel execution with configurable worker count
    - Dependency-aware scheduling
    - Priority-based ordering
    - Graceful shutdown on signals
    - Timeout support for individual tasks
    - Progress tracking
    """

    def __init__(
        self,
        max_workers: int = 4,
        graceful_shutdown_timeout: float = 30.0,
        on_task_complete: Optional[Callable[[ExecutorTask], None]] = None,
        on_task_error: Optional[Callable[[ExecutorTask, Exception], None]] = None
    ):
        """Initialize executor.

        Args:
            max_workers: Maximum parallel workers
            graceful_shutdown_timeout: Timeout for graceful shutdown
            on_task_complete: Callback when task completes
            on_task_error: Callback when task fails
        """
        self.max_workers = max_workers
        self.graceful_shutdown_timeout = graceful_shutdown_timeout
        self.on_task_complete = on_task_complete
        self.on_task_error = on_task_error

        self._resolver = DependencyResolver()
        self._executor: Optional[ThreadPoolExecutor] = None
        self._futures: Dict[Future, ExecutorTask] = {}
        self._lock = threading.Lock()
        self._shutdown_event = threading.Event()
        self._running = False

        # Register signal handlers
        self._original_sigint = signal.getsignal(signal.SIGINT)
        self._original_sigterm = signal.getsignal(signal.SIGTERM)

        # Register cleanup
        atexit.register(self.shutdown)

    def add_task(
        self,
        task_id: str,
        func: Callable,
        args: tuple = (),
        kwargs: dict = None,
        dependencies: List[str] = None,
        priority: int = 0,
        timeout: Optional[float] = None
    ) -> ExecutorTask:
        """Add a task to be executed.

        Args:
            task_id: Unique task identifier
            func: Function to execute
            args: Positional arguments
            kwargs: Keyword arguments
            dependencies: Task IDs that must complete first
            priority: Execution priority (lower = higher)
            timeout: Optional task timeout

        Returns:
            Created task object
        """
        task = ExecutorTask(
            id=task_id,
            func=func,
            args=args,
            kwargs=kwargs or {},
            dependencies=dependencies or [],
            priority=priority,
            timeout=timeout
        )
        self._resolver.add_task(task)
        return task

    def execute_all(self) -> Dict[str, ExecutorTask]:
        """Execute all added tasks respecting dependencies.

        Returns:
            Dict of task_id -> ExecutorTask with results
        """
        self._setup_signal_handlers()
        self._running = True
        self._executor = ThreadPoolExecutor(
            max_workers=self.max_workers,
            thread_name_prefix="executor_"
        )

        try:
            while not self._shutdown_event.is_set() and self._resolver.has_pending_tasks():
                # Get ready tasks
                ready_tasks = self._resolver.get_ready_tasks()

                # Submit ready tasks
                for task in ready_tasks:
                    if self._shutdown_event.is_set():
                        break

                    task.state = TaskState.RUNNING
                    task.start_time = time.time()

                    future = self._executor.submit(self._execute_task, task)
                    self._futures[future] = task

                # Wait for at least one task to complete
                if self._futures:
                    done, _ = wait(
                        self._futures.keys(),
                        timeout=1.0,
                        return_when=FIRST_COMPLETED
                    )

                    # Process completed tasks
                    for future in done:
                        task = self._futures.pop(future)
                        try:
                            result = future.result(timeout=0)
                            self._resolver.mark_completed(task.id, result)

                            if self.on_task_complete:
                                self.on_task_complete(task)

                            logger.debug(f"Task {task.id} completed")

                        except Exception as e:
                            self._resolver.mark_failed(task.id, e)

                            if self.on_task_error:
                                self.on_task_error(task, e)

                            logger.error(f"Task {task.id} failed: {e}")
                else:
                    # No futures, wait a bit before checking again
                    time.sleep(0.1)

        finally:
            self._restore_signal_handlers()
            self._running = False
            self._cleanup()

        return self._resolver._tasks

    def _execute_task(self, task: ExecutorTask) -> Any:
        """Execute a single task with timeout handling."""
        if task.timeout:
            # Use threading for timeout
            result = [None]
            error = [None]
            completed = threading.Event()

            def target():
                try:
                    result[0] = task.func(*task.args, **task.kwargs)
                except Exception as e:
                    error[0] = e
                finally:
                    completed.set()

            thread = threading.Thread(target=target)
            thread.start()
            thread.join(timeout=task.timeout)

            if not completed.is_set():
                raise TimeoutError(f"Task {task.id} timed out after {task.timeout}s")

            if error[0]:
                raise error[0]

            return result[0]
        else:
            return task.func(*task.args, **task.kwargs)

    def shutdown(self, wait: bool = True):
        """Shutdown the executor gracefully."""
        self._shutdown_event.set()

        if self._executor:
            logger.info("Shutting down executor...")

            # Cancel pending futures
            for future in list(self._futures.keys()):
                future.cancel()

            self._executor.shutdown(wait=wait, cancel_futures=True)
            self._executor = None

    def _setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown."""
        def signal_handler(signum, frame):
            logger.info(f"Received signal {signum}, initiating graceful shutdown...")
            self._shutdown_event.set()

        try:
            signal.signal(signal.SIGINT, signal_handler)
            signal.signal(signal.SIGTERM, signal_handler)
        except (ValueError, OSError):
            # Can't set signal handlers in non-main thread
            pass

    def _restore_signal_handlers(self):
        """Restore original signal handlers."""
        try:
            signal.signal(signal.SIGINT, self._original_sigint)
            signal.signal(signal.SIGTERM, self._original_sigterm)
        except (ValueError, OSError):
            pass

    def _cleanup(self):
        """Clean up resources."""
        if self._executor:
            self._executor.shutdown(wait=False, cancel_futures=True)
            self._executor = None

    def get_stats(self) -> Dict[str, Any]:
        """Get execution statistics."""
        task_stats = self._resolver.get_stats()
        completed_tasks = [
            t for t in self._resolver._tasks.values()
            if t.state == TaskState.COMPLETED
        ]

        total_duration = sum(t.duration or 0 for t in completed_tasks)
        avg_duration = total_duration / len(completed_tasks) if completed_tasks else 0

        return {
            "tasks": task_stats,
            "total_tasks": sum(task_stats.values()),
            "completed_duration": total_duration,
            "average_duration": avg_duration,
            "is_running": self._running,
            "shutdown_requested": self._shutdown_event.is_set()
        }


class AgentExecutor(ParallelExecutor):
    """Specialized executor for agent tasks with context sharing."""

    def __init__(
        self,
        max_workers: int = 4,
        shared_context: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        super().__init__(max_workers=max_workers, **kwargs)
        self.shared_context = shared_context or {}
        self._context_lock = threading.Lock()
        self._agent_results: Dict[str, Any] = {}

    def add_agent_task(
        self,
        agent_name: str,
        execute_func: Callable,
        dependencies: List[str] = None,
        priority: int = 0,
        timeout: float = 120.0
    ):
        """Add an agent task.

        Args:
            agent_name: Name of the agent
            execute_func: Function that executes the agent
            dependencies: Agent names that must complete first
            priority: Execution priority
            timeout: Task timeout
        """
        # Wrap to inject context
        def wrapped_execute():
            with self._context_lock:
                # Get results from dependencies
                dep_results = {
                    dep: self._agent_results.get(dep)
                    for dep in (dependencies or [])
                }

            # Execute agent with context
            result = execute_func(self.shared_context, dep_results)

            # Store result for dependents
            with self._context_lock:
                self._agent_results[agent_name] = result

            return result

        self.add_task(
            task_id=agent_name,
            func=wrapped_execute,
            dependencies=dependencies,
            priority=priority,
            timeout=timeout
        )

    def get_agent_result(self, agent_name: str) -> Any:
        """Get result from a completed agent."""
        with self._context_lock:
            return self._agent_results.get(agent_name)

    def get_all_results(self) -> Dict[str, Any]:
        """Get all agent results."""
        with self._context_lock:
            return dict(self._agent_results)


# Convenience function
def execute_in_parallel(
    tasks: List[Dict[str, Any]],
    max_workers: int = 4,
    on_complete: Optional[Callable] = None
) -> Dict[str, ExecutorTask]:
    """Execute tasks in parallel with dependency resolution.

    Args:
        tasks: List of task dicts with keys:
            - id: Task identifier
            - func: Function to execute
            - args: Optional positional arguments
            - kwargs: Optional keyword arguments
            - dependencies: Optional list of task IDs
            - priority: Optional priority (default 0)
            - timeout: Optional timeout
        max_workers: Maximum parallel workers
        on_complete: Optional callback for task completion

    Returns:
        Dict of task_id -> ExecutorTask
    """
    executor = ParallelExecutor(
        max_workers=max_workers,
        on_task_complete=on_complete
    )

    for task_def in tasks:
        executor.add_task(
            task_id=task_def["id"],
            func=task_def["func"],
            args=task_def.get("args", ()),
            kwargs=task_def.get("kwargs", {}),
            dependencies=task_def.get("dependencies", []),
            priority=task_def.get("priority", 0),
            timeout=task_def.get("timeout")
        )

    return executor.execute_all()
