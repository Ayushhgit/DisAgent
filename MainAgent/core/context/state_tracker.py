# ============================================================================
# FILE: state_tracker.py
# Tracks agent states, tasks, and execution flow
# ============================================================================

from typing import Dict, List, Optional, Set, Any
from datetime import datetime
import json
from pathlib import Path
import threading
from collections import defaultdict

from core.memory.memory_types import (
    AgentState,
    TaskInfo,
    TaskType,
    AgentStatus,
    StateSnapshot
)
from core.runtime.event_bus import EventType, publish_event


class CycleDetectedError(Exception):
    """Raised when a dependency cycle is detected."""
    def __init__(self, cycle: List[str]):
        self.cycle = cycle
        cycle_str = " -> ".join(cycle + [cycle[0]])
        super().__init__(f"Dependency cycle detected: {cycle_str}")


class StateTracker:
    """
    Tracks:
    - Agent states and availability
    - Task assignment and completion
    - Dependencies and blockers
    - System-wide execution flow

    Thread-safe with RLock around all state mutations.
    Includes cycle detection and reverse dependency index for O(1) lookups.
    """

    def __init__(self, main_agent_id: str = "main_agent", storage_path: Optional[str] = None):
        self.main_agent_id = main_agent_id
        self.storage_path = Path(storage_path) if storage_path else Path("./.agent_memory/state")
        self.agents: Dict[str, AgentStatus] = {}
        self.tasks: Dict[str, TaskInfo] = {}
        self.task_queue: List[str] = []
        self.task_dependencies: Dict[str, Set[str]] = {}  # task_id -> dependencies
        # Reverse index: dependency -> tasks that depend on it (for O(1) unblock checks)
        self._reverse_dependencies: Dict[str, Set[str]] = defaultdict(set)
        self.state_history: List[StateSnapshot] = []
        self.global_blockers: List[str] = []

        # Thread safety - single RLock for all state operations
        self._lock = threading.RLock()

        # Initialize main agent
        self.register_agent(main_agent_id)

    def _would_create_cycle(self, task_id: str, new_deps: Set[str]) -> Optional[List[str]]:
        """
        Check if adding dependencies would create a cycle.
        Returns the cycle path if found, None otherwise.
        Must be called with lock held.
        """
        if not new_deps:
            return None

        # DFS to check for cycles
        visited: Set[str] = set()
        rec_stack: Set[str] = set()
        path: List[str] = []

        def dfs(node: str) -> Optional[List[str]]:
            if node in rec_stack:
                # Found cycle - extract path
                cycle_start = path.index(node) if node in path else 0
                return path[cycle_start:]

            if node in visited:
                return None

            visited.add(node)
            rec_stack.add(node)
            path.append(node)

            # Get dependencies for this node
            if node == task_id:
                deps = new_deps
            else:
                deps = self.task_dependencies.get(node, set())

            for dep in deps:
                cycle = dfs(dep)
                if cycle:
                    return cycle

            path.pop()
            rec_stack.remove(node)
            return None

        return dfs(task_id)

    def _update_reverse_index(self, task_id: str, old_deps: Set[str], new_deps: Set[str]) -> None:
        """Update the reverse dependency index. Must be called with lock held."""
        # Remove from old dependencies
        for dep in old_deps - new_deps:
            if dep in self._reverse_dependencies:
                self._reverse_dependencies[dep].discard(task_id)
                if not self._reverse_dependencies[dep]:
                    del self._reverse_dependencies[dep]

        # Add to new dependencies
        for dep in new_deps - old_deps:
            self._reverse_dependencies[dep].add(task_id)

    def register_agent(self, agent_id: str, metadata: Optional[Dict[str, Any]] = None):
        """Register a new agent"""
        with self._lock:
            if agent_id not in self.agents:
                self.agents[agent_id] = AgentStatus(
                    agent_id=agent_id,
                    state=AgentState.IDLE,
                    metadata=metadata or {}
                )
                
                # Publish event
                publish_event(
                    EventType.AGENT_SPAWNED,
                    agent_id=agent_id,
                    data={"metadata": metadata or {}}
                )

    def create_task(self, task: TaskInfo, validate_cycles: bool = True) -> str:
        """Create and queue a new task.

        Args:
            task: TaskInfo object to create
            validate_cycles: If True, check for dependency cycles before adding

        Returns:
            The task_id of the created task

        Raises:
            CycleDetectedError: If adding this task would create a dependency cycle
        """
        with self._lock:
            new_deps = set(task.dependencies) if task.dependencies else set()

            # Check for cycles before adding
            if validate_cycles and new_deps:
                cycle = self._would_create_cycle(task.task_id, new_deps)
                if cycle:
                    raise CycleDetectedError(cycle)

            self.tasks[task.task_id] = task
            self.task_queue.append(task.task_id)

            # Track dependencies and update reverse index
            if new_deps:
                old_deps = self.task_dependencies.get(task.task_id, set())
                self.task_dependencies[task.task_id] = new_deps
                self._update_reverse_index(task.task_id, old_deps, new_deps)

            # Publish event
            publish_event(
                EventType.TASK_CREATED,
                task_id=task.task_id,
                data={
                    "description": task.description,
                    "task_type": task.task_type.value,
                    "dependencies": task.dependencies
                }
            )

        return task.task_id

    def assign_task(self, task_id: str, agent_id: str) -> bool:
        """Assign task to agent"""
        with self._lock:
            if task_id not in self.tasks or agent_id not in self.agents:
                return False

            # Check if dependencies are met
            if not self._dependencies_met(task_id):
                return False

            task = self.tasks[task_id]
            agent = self.agents[agent_id]

            task.assigned_agent = agent_id
            task.status = AgentState.EXECUTING

            agent.current_task = task_id
            agent.assigned_tasks.append(task_id)
            agent.state = AgentState.EXECUTING
            agent.last_activity = datetime.now()

            if task_id in self.task_queue:
                self.task_queue.remove(task_id)

            # Publish event
            publish_event(
                EventType.TASK_ASSIGNED,
                agent_id=agent_id,
                task_id=task_id,
                data={"task_description": task.description}
            )

        return True

    def complete_task(self, task_id: str, result: str):
        """Mark task as completed"""
        with self._lock:
            if task_id not in self.tasks:
                return

            task = self.tasks[task_id]
            agent = self.agents.get(task.assigned_agent)

            task.status = AgentState.COMPLETED
            task.result = result
            task.completed_at = datetime.now()

            if agent:
                agent.completed_tasks.append(task_id)
                agent.current_task = None
                agent.state = AgentState.IDLE
                agent.last_activity = datetime.now()

            # Publish event
            publish_event(
                EventType.TASK_COMPLETED,
                agent_id=task.assigned_agent,
                task_id=task_id,
                data={"result": result[:200] if result else ""}
            )

            # Check if this unblocks other tasks
            self._check_unblocked_tasks(task_id)

    def fail_task(self, task_id: str, error: str):
        """Mark task as failed"""
        with self._lock:
            if task_id not in self.tasks:
                return

            task = self.tasks[task_id]
            agent = self.agents.get(task.assigned_agent)

            task.status = AgentState.FAILED
            task.error = error

            if agent:
                agent.current_task = None
                agent.state = AgentState.IDLE
                agent.last_activity = datetime.now()

            # Publish event
            publish_event(
                EventType.TASK_FAILED,
                agent_id=task.assigned_agent,
                task_id=task_id,
                data={"error": error}
            )

    def block_agent(self, agent_id: str, blocker: str):
        """Block an agent"""
        with self._lock:
            if agent_id not in self.agents:
                return

            agent = self.agents[agent_id]
            agent.state = AgentState.BLOCKED
            agent.blockers.append(blocker)
            agent.last_activity = datetime.now()

            # If task is blocked, add to task blockers too
            if agent.current_task:
                task = self.tasks.get(agent.current_task)
                if task:
                    task.status = AgentState.BLOCKED

            # Publish event
            publish_event(
                EventType.AGENT_BLOCKED,
                agent_id=agent_id,
                data={"blocker": blocker}
            )

    def unblock_agent(self, agent_id: str, blocker: str):
        """Remove blocker from agent"""
        with self._lock:
            if agent_id not in self.agents:
                return

            agent = self.agents[agent_id]
            if blocker in agent.blockers:
                agent.blockers.remove(blocker)

            if not agent.blockers:
                agent.state = AgentState.IDLE if not agent.current_task else AgentState.EXECUTING
                agent.last_activity = datetime.now()

                # Unblock task too
                if agent.current_task:
                    task = self.tasks.get(agent.current_task)
                    if task:
                        task.status = AgentState.EXECUTING

                # Publish event
                publish_event(
                    EventType.AGENT_UNBLOCKED,
                    agent_id=agent_id,
                    data={"removed_blocker": blocker}
                )

    def get_available_agents(self) -> List[str]:
        """Get list of idle agents"""
        with self._lock:
            return [
                agent_id for agent_id, agent in self.agents.items()
                if agent.state == AgentState.IDLE
            ]

    def get_blocked_agents(self) -> List[AgentStatus]:
        """Get all blocked agents"""
        with self._lock:
            return [
                agent for agent in self.agents.values()
                if agent.state == AgentState.BLOCKED
            ]

    def get_active_agents(self) -> List[AgentStatus]:
        """Get all active (executing) agents"""
        with self._lock:
            return [
                agent for agent in self.agents.values()
                if agent.state == AgentState.EXECUTING
            ]

    def get_ready_tasks(self) -> List[str]:
        """Get tasks ready to be assigned (dependencies met)"""
        with self._lock:
            ready = []
            for task_id in self.task_queue:
                if self._dependencies_met(task_id):
                    ready.append(task_id)
            return ready

    def get_agent_status(self, agent_id: str) -> Optional[AgentStatus]:
        """Get current status of agent"""
        with self._lock:
            return self.agents.get(agent_id)

    def get_task_status(self, task_id: str) -> Optional[TaskInfo]:
        """Get current status of task"""
        with self._lock:
            return self.tasks.get(task_id)

    def update_agent_state(self, agent_id: str, state: AgentState):
        """Update agent state"""
        with self._lock:
            if agent_id in self.agents:
                self.agents[agent_id].state = state
                self.agents[agent_id].last_activity = datetime.now()

    def take_snapshot(self) -> StateSnapshot:
        """Record current system state (thread-safe)."""
        with self._lock:
            snapshot = StateSnapshot(
                timestamp=datetime.now(),
                active_agents=[
                    aid for aid, a in self.agents.items()
                    if a.state == AgentState.EXECUTING
                ],
                task_queue_size=len(self.task_queue),
                completed_tasks=len([t for t in self.tasks.values() if t.status == AgentState.COMPLETED]),
                blocked_agents=[
                    aid for aid, a in self.agents.items()
                    if a.state == AgentState.BLOCKED
                ],
                system_state=self._get_system_state_summary()
            )
            self.state_history.append(snapshot)
            return snapshot

    def get_system_summary(self) -> str:
        """Generate comprehensive system status summary (thread-safe)."""
        with self._lock:
            total_agents = len(self.agents)
            idle_agents = len([a for a in self.agents.values() if a.state == AgentState.IDLE])
            blocked_agents = [a for a in self.agents.values() if a.state == AgentState.BLOCKED]
            blocked = len(blocked_agents)

            total_tasks = len(self.tasks)
            completed = len([t for t in self.tasks.values() if t.status == AgentState.COMPLETED])
            failed = len([t for t in self.tasks.values() if t.status == AgentState.FAILED])
            queued = len(self.task_queue)

            summary = f"""
=== SYSTEM STATE SUMMARY ===
Timestamp: {datetime.now()}

AGENTS ({total_agents} total):
  - Idle: {idle_agents}
  - Active: {total_agents - idle_agents - blocked}
  - Blocked: {blocked}

TASKS ({total_tasks} total):
  - Completed: {completed}
  - Failed: {failed}
  - Queued: {queued}
  - In Progress: {total_tasks - completed - failed - queued}

BLOCKERS:
"""
            for agent in blocked_agents:
                summary += f"  - {agent.agent_id}: {', '.join(agent.blockers)}\n"

            if self.global_blockers:
                summary += f"\nGLOBAL BLOCKERS: {', '.join(self.global_blockers)}\n"

            return summary

    def get_agent_history(self, agent_id: str) -> List[TaskInfo]:
        """Get task history for an agent (thread-safe)."""
        with self._lock:
            return [
                task for task in self.tasks.values()
                if task.assigned_agent == agent_id
            ]

    def add_global_blocker(self, blocker: str) -> None:
        """Add a global blocker (thread-safe)."""
        with self._lock:
            if blocker not in self.global_blockers:
                self.global_blockers.append(blocker)

    def remove_global_blocker(self, blocker: str) -> None:
        """Remove a global blocker (thread-safe)."""
        with self._lock:
            if blocker in self.global_blockers:
                self.global_blockers.remove(blocker)

    def _dependencies_met(self, task_id: str) -> bool:
        """Check if all task dependencies are completed"""
        if task_id not in self.task_dependencies:
            return True

        dependencies = self.task_dependencies[task_id]
        for dep_id in dependencies:
            dep_task = self.tasks.get(dep_id)
            if not dep_task or dep_task.status != AgentState.COMPLETED:
                return False

        return True

    def _check_unblocked_tasks(self, completed_task_id: str) -> List[str]:
        """Check which tasks are now unblocked using reverse index.

        Returns list of newly ready task IDs.
        Must be called with lock held.
        """
        newly_ready = []
        # Use reverse index for O(1) lookup instead of O(n) scan
        dependent_tasks = self._reverse_dependencies.get(completed_task_id, set())
        for task_id in dependent_tasks:
            if self._dependencies_met(task_id):
                newly_ready.append(task_id)
        return newly_ready

    def _get_system_state_summary(self) -> str:
        """Get brief system state"""
        active = len([a for a in self.agents.values() if a.state == AgentState.EXECUTING])
        return f"{active} agents active, {len(self.task_queue)} tasks queued"

    def save_to_disk(self, filename: str = "state.json"):
        """Persist state to disk"""
        self.storage_path.mkdir(parents=True, exist_ok=True)
        filepath = self.storage_path / filename

        data = {
            "main_agent_id": self.main_agent_id,
            "agents": {aid: agent.to_dict() for aid, agent in self.agents.items()},
            "tasks": {tid: task.to_dict() for tid, task in self.tasks.items()},
            "task_queue": self.task_queue,
            "task_dependencies": {k: list(v) for k, v in self.task_dependencies.items()},
            "global_blockers": self.global_blockers,
            "state_history": [s.to_dict() for s in self.state_history[-100:]],  # Keep last 100
            "saved_at": datetime.now().isoformat()
        }

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)

    def load_from_disk(self, filename: str = "state.json") -> bool:
        """Load state from disk"""
        filepath = self.storage_path / filename

        if not filepath.exists():
            return False

        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)

            self.main_agent_id = data.get("main_agent_id", "main_agent")

            # Restore agents
            for aid, agent_data in data.get("agents", {}).items():
                self.agents[aid] = AgentStatus(
                    agent_id=agent_data["agent_id"],
                    state=AgentState(agent_data["state"]),
                    current_task=agent_data.get("current_task"),
                    assigned_tasks=agent_data.get("assigned_tasks", []),
                    completed_tasks=agent_data.get("completed_tasks", []),
                    blockers=agent_data.get("blockers", []),
                    last_activity=datetime.fromisoformat(agent_data["last_activity"]),
                    metadata=agent_data.get("metadata", {})
                )

            # Restore tasks
            for tid, task_data in data.get("tasks", {}).items():
                self.tasks[tid] = TaskInfo.from_dict(task_data)

            self.task_queue = data.get("task_queue", [])
            self.task_dependencies = {k: set(v) for k, v in data.get("task_dependencies", {}).items()}
            self.global_blockers = data.get("global_blockers", [])

            return True
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            print(f"Warning: Could not load state: {e}")
            return False

    def clear(self):
        """Clear all state (except main agent)"""
        self.agents = {self.main_agent_id: AgentStatus(
            agent_id=self.main_agent_id,
            state=AgentState.IDLE
        )}
        self.tasks.clear()
        self.task_queue.clear()
        self.task_dependencies.clear()
        self.state_history.clear()
        self.global_blockers.clear()

    def __len__(self) -> int:
        return len(self.tasks)
