"""Shared context object for coordinating multi-agent workflows."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from datetime import datetime

# Import the canonical ContextManager from core module
from core.context.context_manager import ContextManager as CoreContextManager


class ContextManager:
    """Simplified context manager that orchestrates memory layers.

    This is a wrapper/adapter around the core ContextManager that provides
    a simplified interface for the orchestrator.
    """

    def __init__(self, project_name: str, main_agent_id: str = "orchestrator"):
        # Use the core ContextManager internally
        self._core = CoreContextManager(project_name=project_name)
        self.project_name = project_name
        self.main_agent_id = main_agent_id
        self._agents: Dict[str, Any] = {}

    @property
    def short_term(self):
        """Access short-term memory."""
        return self._core.stm

    @property
    def long_term(self):
        """Access long-term memory."""
        return self._core.ltm

    @property
    def vector_store(self):
        """Access vector store."""
        return self._core.vector_store

    def record_agent_output(self, agent_id: str, output: str, task_description: str):
        """Record agent output to short-term memory."""
        self._core.record_agent_output(agent_id, output, task_description)

    def spawn_agent(self, agent_name: str):
        """Register a new agent."""
        self._agents[agent_name] = {"status": "active"}

    def assign_task_to_agent(self, task, agent_name: str):
        """Assign a task to an agent."""
        pass  # Task tracking is handled by StateTracker

    def complete_agent_task(self, task_id: str, result: str, modified_files=None):
        """Mark a task as complete."""
        pass  # Task tracking is handled by StateTracker

    def search_relevant_context(self, query: str, agent_name: str) -> List[str]:
        """Search for relevant context."""
        return self._core.retrieve_relevant_memory(query)

    def get_system_status(self) -> str:
        """Get system status."""
        return f"Project: {self.project_name}, Agents: {len(self._agents)}"

    def add_learning(self, learning: str, agent_name: str):
        """Record a learning."""
        self._core.ltm.add_learning(learning, agent_name)

    def get_agent_context(self, agent_name: str) -> str:
        """Get context for an agent."""
        return self._core.stm.summarize_context(agent_name)

    def save_all(self):
        """Save all memory to disk."""
        self._core.save()

    def load_all(self) -> bool:
        """Load all memory from disk."""
        return self._core.load()


# Import memory types
from core.memory import (
    ProjectContext,
    TaskInfo,
    TaskType,
    MemoryPriority,
    MemoryEntry,
    AgentState,
)


@dataclass
class AgentContext:
    """
    Holds shared knowledge between agents run by an orchestrator.
    Now integrated with the unified memory system.
    """

    memory: Dict[str, str] = field(default_factory=dict)
    decisions: Dict[str, str] = field(default_factory=dict)
    scope_info: Dict[str, str] = field(default_factory=dict)
    architecture: str = ""
    file_structure: Dict[str, str] = field(default_factory=dict)

    # Unified memory system (initialized lazily)
    _unified_memory: Optional[ContextManager] = field(default=None, repr=False)
    _project_name: str = field(default="default_project")

    def __post_init__(self):
        """Initialize unified memory system"""
        if self._unified_memory is None:
            self._unified_memory = ContextManager(
                project_name=self._project_name,
                main_agent_id="orchestrator"
            )

    @property
    def unified_memory(self) -> ContextManager:
        """Get the unified memory system"""
        if self._unified_memory is None:
            self._unified_memory = ContextManager(
                project_name=self._project_name,
                main_agent_id="orchestrator"
            )
        return self._unified_memory

    def add_result(self, agent_name: str, result: str) -> None:
        """Add agent result to both legacy and new memory systems"""
        self.memory[agent_name] = result

        # Also add to unified memory
        self.unified_memory.record_agent_output(
            agent_id=agent_name,
            output=result,
            task_description=f"Agent {agent_name} execution"
        )

    def get_context(self) -> str:
        """Get formatted context string for agents"""
        context_parts: List[str] = ["=== SHARED AGENT CONTEXT ===\n"]

        if self.scope_info:
            context_parts.append("SCOPE INFORMATION:\n")
            for key, value in self.scope_info.items():
                context_parts.append(f"  {key}: {value}\n")
            context_parts.append("\n")

        if self.architecture:
            context_parts.append("SYSTEM ARCHITECTURE:\n")
            context_parts.append(f"{self.architecture}\n\n")

        # Include relevant context from unified memory
        if self.unified_memory:
            recent_context = self.unified_memory.short_term.summarize_context()
            if recent_context and "No recent context" not in recent_context:
                context_parts.append("RECENT MEMORY CONTEXT:\n")
                context_parts.append(f"{recent_context[:500]}\n\n")

        for agent, result in self.memory.items():
            truncated = result[:2000] if len(result) > 2000 else result
            context_parts.append(f"[{agent}]:\n{truncated}\n\n---\n\n")

        return "".join(context_parts)

    def set_scope(self, scope_dict: Dict[str, str]) -> None:
        """Set scope information"""
        self.scope_info = scope_dict

        # Update unified memory with scope info
        if self.unified_memory:
            self.unified_memory.short_term.add_decision(
                agent_id="orchestrator",
                decision="Scope analysis completed",
                context=str(scope_dict),
                priority=MemoryPriority.HIGH
            )

    def set_architecture(self, architecture: str) -> None:
        """Set system architecture"""
        self.architecture = architecture

    def add_file(self, filename: str, content: str) -> None:
        """Add file to context and memory"""
        self.file_structure[filename] = content

        # Also add to unified memory
        if self.unified_memory:
            self.unified_memory.short_term.add_code_context(
                agent_id="orchestrator",
                filename=filename,
                content=content,
                purpose="Project file"
            )
            self.unified_memory.long_term.update_file(filename, content)

    def register_agent(self, agent_name: str) -> None:
        """Register a new agent in the memory system"""
        if self.unified_memory:
            self.unified_memory.spawn_agent(agent_name)

    def create_task(self, task_id: str, description: str, agent_name: str,
                    task_type: str = "implementation",
                    affected_files: Optional[List[str]] = None) -> None:
        """Create and assign a task"""
        if self.unified_memory:
            # Map task type string to enum
            type_map = {
                "implementation": TaskType.IMPLEMENTATION,
                "bug_fix": TaskType.BUG_FIX,
                "refactor": TaskType.REFACTOR,
                "testing": TaskType.TESTING,
                "documentation": TaskType.DOCUMENTATION,
                "architecture": TaskType.ARCHITECTURE,
                "research": TaskType.RESEARCH,
                "code_review": TaskType.CODE_REVIEW
            }

            task = TaskInfo(
                task_id=task_id,
                task_type=type_map.get(task_type, TaskType.IMPLEMENTATION),
                description=description,
                assigned_agent=agent_name,
                affected_files=affected_files or []
            )

            self.unified_memory.assign_task_to_agent(task, agent_name)

    def complete_task(self, task_id: str, result: str,
                      modified_files: Optional[Dict[str, str]] = None) -> None:
        """Mark a task as completed"""
        if self.unified_memory:
            self.unified_memory.complete_agent_task(task_id, result, modified_files)

    def add_decision(self, agent_name: str, decision: str, context: str) -> None:
        """Record a decision made during execution"""
        self.decisions[f"{agent_name}_{datetime.now().isoformat()}"] = decision

        if self.unified_memory:
            self.unified_memory.short_term.add_decision(
                agent_id=agent_name,
                decision=decision,
                context=context
            )

    def add_error(self, agent_name: str, error: str, context: str) -> None:
        """Record an error during execution"""
        if self.unified_memory:
            self.unified_memory.short_term.add_error(
                agent_id=agent_name,
                error=error,
                context=context
            )

    def search_relevant(self, query: str, agent_name: str = "orchestrator") -> List[str]:
        """Search for relevant context"""
        if self.unified_memory:
            return self.unified_memory.search_relevant_context(query, agent_name)
        return []

    def get_system_status(self) -> str:
        """Get system status including memory stats"""
        if self.unified_memory:
            return self.unified_memory.get_system_status()
        return "Memory system not initialized"

    def save_memory(self) -> None:
        """Save all memory to disk"""
        if self.unified_memory:
            self.unified_memory.save_all()

    def load_memory(self) -> bool:
        """Load memory from disk"""
        if self.unified_memory:
            return self.unified_memory.load_all()
        return False

    def add_learning(self, learning: str, agent_name: str = "orchestrator") -> None:
        """Record a learning or insight"""
        if self.unified_memory:
            self.unified_memory.add_learning(learning, agent_name)

    def get_agent_context(self, agent_name: str) -> str:
        """Get context specific to an agent"""
        if self.unified_memory:
            return self.unified_memory.get_agent_context(agent_name)
        return ""


def create_context(project_name: str = "default_project") -> AgentContext:
    """Factory function to create a properly initialized AgentContext"""
    context = AgentContext()
    context._project_name = project_name
    context._unified_memory = ContextManager(
        project_name=project_name,
        main_agent_id="orchestrator"
    )
    return context
