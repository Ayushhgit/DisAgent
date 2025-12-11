"""
Base Agent - Abstract base class for all agents.

All specialized agents should inherit from BaseAgent and implement:
- run(): Execute the agent's main logic
- act(): Backward-compatible method for simple actions
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Dict, Any, List
import logging

logger = logging.getLogger(__name__)


@dataclass
class AgentResult:
    """Standard result object returned by agents."""
    success: bool
    output: str
    metadata: Dict[str, Any]
    files_created: List[str] = None
    files_modified: List[str] = None
    errors: List[str] = None

    def __post_init__(self):
        if self.files_created is None:
            self.files_created = []
        if self.files_modified is None:
            self.files_modified = []
        if self.errors is None:
            self.errors = []


class BaseAgent(ABC):
    """Abstract base class for all agents in the system.

    Agents are responsible for:
    - Processing prompts/tasks
    - Interacting with files via FileManager
    - Producing structured outputs (code, documentation, etc.)

    Subclasses must implement:
    - run(task_id, prompt, context, file_manager) -> AgentResult
    """

    def __init__(self, name: str, description: str = ""):
        """Initialize the base agent.

        Args:
            name: Unique identifier for this agent
            description: Human-readable description of what this agent does
        """
        self.name = name
        self.description = description or f"Agent: {name}"
        self._execution_count = 0
        self._total_tokens = 0

    @abstractmethod
    def run(
        self,
        task_id: str,
        prompt: str,
        context: Any,
        file_manager: Any,
    ) -> AgentResult:
        """Execute the agent's main logic.

        Args:
            task_id: Unique identifier for this task
            prompt: The prompt/instructions for this execution
            context: Shared context object with scope, reasoning, memory
            file_manager: FileManager instance for file operations

        Returns:
            AgentResult with success status, output, and metadata
        """
        pass

    def act(self, *args, **kwargs) -> Any:
        """Backward-compatible simple action method.

        Override this for simple, stateless agent actions.
        """
        raise NotImplementedError(
            f"Agent {self.name} does not implement act(). "
            "Use run() for full agent execution."
        )

    def get_stats(self) -> Dict[str, Any]:
        """Get agent execution statistics."""
        return {
            "name": self.name,
            "executions": self._execution_count,
            "total_tokens": self._total_tokens,
        }

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}(name='{self.name}')>"


class SimpleAgent(BaseAgent):
    """A simple agent that wraps a callable function.

    Useful for creating agents from functions without full class inheritance.
    """

    def __init__(
        self,
        name: str,
        handler: callable,
        description: str = "",
    ):
        """Initialize a simple agent.

        Args:
            name: Agent name
            handler: Callable that takes (task_id, prompt, context, file_manager)
                     and returns AgentResult or (success, output) tuple
            description: Agent description
        """
        super().__init__(name, description)
        self._handler = handler

    def run(
        self,
        task_id: str,
        prompt: str,
        context: Any,
        file_manager: Any,
    ) -> AgentResult:
        """Execute the handler function."""
        self._execution_count += 1

        try:
            result = self._handler(task_id, prompt, context, file_manager)

            if isinstance(result, AgentResult):
                return result
            elif isinstance(result, tuple) and len(result) >= 2:
                return AgentResult(
                    success=bool(result[0]),
                    output=str(result[1]),
                    metadata=result[2] if len(result) > 2 else {},
                )
            else:
                return AgentResult(
                    success=True,
                    output=str(result),
                    metadata={},
                )

        except Exception as exc:
            logger.exception(f"Agent {self.name} failed: {exc}")
            return AgentResult(
                success=False,
                output="",
                metadata={},
                errors=[str(exc)],
            )

    def act(self, *args, **kwargs) -> Any:
        """Simple action - calls handler with args/kwargs."""
        return self._handler(*args, **kwargs)
