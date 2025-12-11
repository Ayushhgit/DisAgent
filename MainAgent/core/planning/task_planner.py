# core/planning/task_planner.py
"""
Task Planner

Responsibilities:
- Convert planning scope into TaskInfo objects (typed)
- Register tasks into StateTracker
- Build, validate, and register dependency graph
- Provide simple assignment / scheduling heuristics
"""

from typing import List, Dict, Optional
import logging
from core.context.state_tracker import StateTracker
from core.memory.memory_types import TaskInfo, TaskType, AgentState
from core.planning.validation import PlanValidator

logger = logging.getLogger(__name__)


class TaskPlannerError(Exception):
    pass


class TaskPlanner:
    def __init__(self, state_tracker: StateTracker, validator: Optional[PlanValidator] = None):
        self.state = state_tracker
        self.validator = validator or PlanValidator()

    def create_tasks_from_scope(self, scope: Dict) -> List[TaskInfo]:
        """
        Convert scope['tasks'] into memory TaskInfo instances.
        Expect scope tasks like:
            {"id": "...", "description": "...", "type": "implementation", "estimate": "small", "dependencies": []}
        """
        raw_tasks = scope.get("tasks", [])
        if not self.validator.validate_tasks(raw_tasks):
            raise TaskPlannerError("Invalid tasks in scope")

        tasks: List[TaskInfo] = []
        for t in raw_tasks:
            # Map generic type string to TaskType if possible
            ttype = self._map_task_type(t.get("type", "implementation"))
            # Create TaskInfo. Adjust constructor if your TaskInfo fields differ.
            task = TaskInfo(
                task_id=t.get("id"),
                description=t.get("description", ""),
                task_type=ttype,
                dependencies=t.get("dependencies", []),
                assigned_agent="",
            )
            tasks.append(task)
        return tasks

    def register_tasks(self, tasks: List[TaskInfo]) -> List[str]:
        """
        Store tasks in StateTracker and return inserted ids.
        """
        inserted = []
        for t in tasks:
            tid = self.state.create_task(t)
            inserted.append(tid)
            logger.debug("Registered task %s", tid)
        # after registering, ensure dependencies map in state is consistent
        self._sync_task_dependencies(tasks)
        return inserted

    def _sync_task_dependencies(self, tasks: List[TaskInfo]):
        """
        Ensure StateTracker.task_dependencies contains the tasks we just registered.
        """
        for t in tasks:
            if t.dependencies:
                # state.create_task already registers dependencies in many implementations;
                # if not, ensure task_dependencies entry exists:
                self.state.task_dependencies[t.task_id] = set(t.dependencies)

    def plan_and_register(self, scope: Dict) -> List[str]:
        """
        Convenience: analyze scope -> convert to tasks -> register.
        Returns list of registered task ids.
        """
        tasks = self.create_tasks_from_scope(scope)
        # validate dependency graph
        dep_map = {t.task_id: list(t.dependencies or []) for t in tasks}
        if not self.validator.validate_dependencies(dep_map):
            raise TaskPlannerError("Dependency graph is invalid (cycles?)")
        return self.register_tasks(tasks)

    # -----------------------------
    # Scheduler / assignment helpers
    # -----------------------------
    def get_ready_tasks(self) -> List[str]:
        """
        Delegates to StateTracker.get_ready_tasks for the ready-to-assign tasks.
        """
        return self.state.get_ready_tasks()

    def assign_task_to_best_agent(self, task_id: str, preferred_agents: Optional[List[str]] = None) -> bool:
        """
        Lightweight assignment heuristic:
            1) If preferred_agents supplied, ONLY consider those agents (filter to idle ones)
            2) Otherwise pick any idle agent
            3) If no idle agents, leave task queued
        """
        avail = self.state.get_available_agents()
        logger.debug("Available agents: %s, Preferred: %s", avail, preferred_agents)

        # If preferred_agents specified, restrict candidates to ONLY those agents
        if preferred_agents:
            # Filter available agents to only those in preferred list
            candidates = [a for a in preferred_agents if a in avail]
            logger.debug("Filtered candidates (preferred & available): %s", candidates)
        else:
            candidates = avail

        if not candidates:
            logger.debug("No available agents to assign task %s", task_id)
            return False

        # prefer agents with shortest assigned_tasks list (if AgentStatus exposes it)
        best = min(candidates, key=lambda aid: len(self.state.agents[aid].assigned_tasks) if self.state.agents.get(aid) else 0)
        logger.debug("Assigning task %s to agent %s", task_id, best)
        return self.state.assign_task(task_id, best)

    @staticmethod
    def _map_task_type(ttype: str) -> TaskType:
        # Map string to TaskType enum if possible; otherwise return generic
        try:
            return TaskType(ttype)
        except Exception:
            try:
                # maybe TaskType is an enum with .IMPLEMENTATION etc
                return TaskType("generic")
            except Exception:
                # fallback: create a TaskType-like object if TaskType is not enum
                return ttype  # type: ignore
