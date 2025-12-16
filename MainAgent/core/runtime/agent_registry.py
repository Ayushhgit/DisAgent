# ============================================================================
# FILE: agent_registry.py
# Dynamic agent registration, prioritization, and health monitoring
# ============================================================================

from __future__ import annotations

import time
import threading
import logging
from typing import Dict, List, Optional, Callable, Any, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum

logger = logging.getLogger(__name__)


class AgentPriority(Enum):
    """Priority levels for agent execution."""
    CRITICAL = 1    # Must run first (e.g., security checks)
    HIGH = 2        # Important agents (e.g., backend for backend tasks)
    NORMAL = 3      # Default priority
    LOW = 4         # Background/optional agents
    BACKGROUND = 5  # Run only if time permits


class AgentHealth(Enum):
    """Health status of an agent."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class AgentHealthMetrics:
    """Health metrics for an agent."""
    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    total_duration: float = 0.0
    last_call_time: Optional[datetime] = None
    last_error: Optional[str] = None
    consecutive_failures: int = 0
    circuit_breaker_open: bool = False
    circuit_breaker_opens_at: Optional[datetime] = None

    @property
    def success_rate(self) -> float:
        if self.total_calls == 0:
            return 1.0
        return self.successful_calls / self.total_calls

    @property
    def average_duration(self) -> float:
        if self.successful_calls == 0:
            return 0.0
        return self.total_duration / self.successful_calls

    def record_success(self, duration: float):
        """Record a successful call."""
        self.total_calls += 1
        self.successful_calls += 1
        self.total_duration += duration
        self.last_call_time = datetime.now()
        self.consecutive_failures = 0
        self.last_error = None

        # Close circuit breaker on success
        if self.circuit_breaker_open:
            self.circuit_breaker_open = False
            self.circuit_breaker_opens_at = None

    def record_failure(self, error: str, duration: float = 0.0):
        """Record a failed call."""
        self.total_calls += 1
        self.failed_calls += 1
        self.total_duration += duration
        self.last_call_time = datetime.now()
        self.last_error = error
        self.consecutive_failures += 1

    def get_health_status(self) -> AgentHealth:
        """Determine current health status."""
        if self.circuit_breaker_open:
            return AgentHealth.UNHEALTHY

        if self.total_calls == 0:
            return AgentHealth.UNKNOWN

        if self.success_rate >= 0.95:
            return AgentHealth.HEALTHY
        elif self.success_rate >= 0.7:
            return AgentHealth.DEGRADED
        else:
            return AgentHealth.UNHEALTHY


@dataclass
class AgentDefinition:
    """Definition of a registered agent type."""
    name: str
    description: str
    domains: List[str]  # e.g., ["backend", "database"]
    priority: AgentPriority = AgentPriority.NORMAL
    prompt_template: str = ""
    capabilities: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)  # Other agents this depends on
    max_concurrent: int = 1  # Max concurrent instances
    timeout: float = 120.0  # Default timeout in seconds
    enabled: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "domains": self.domains,
            "priority": self.priority.name,
            "capabilities": self.capabilities,
            "dependencies": self.dependencies,
            "max_concurrent": self.max_concurrent,
            "timeout": self.timeout,
            "enabled": self.enabled
        }


class AgentRegistry:
    """Central registry for agent types with health monitoring.

    Features:
    - Dynamic agent registration
    - Priority-based execution ordering
    - Health monitoring and circuit breakers
    - Agent capability matching
    """

    # Default agent definitions
    DEFAULT_AGENTS = [
        AgentDefinition(
            name="generalist",
            description="General-purpose agent for varied tasks",
            domains=["general"],
            priority=AgentPriority.NORMAL,
            capabilities=["code_generation", "analysis", "documentation"]
        ),
        AgentDefinition(
            name="backend",
            description="Backend development specialist",
            domains=["backend", "api", "database"],
            priority=AgentPriority.HIGH,
            capabilities=["api_design", "database_schema", "server_logic"]
        ),
        AgentDefinition(
            name="frontend",
            description="Frontend/UI development specialist",
            domains=["frontend", "ui", "ux"],
            priority=AgentPriority.HIGH,
            capabilities=["component_design", "styling", "accessibility"]
        ),
        AgentDefinition(
            name="tester",
            description="Testing and QA specialist",
            domains=["testing", "qa"],
            priority=AgentPriority.NORMAL,
            capabilities=["unit_tests", "integration_tests", "test_coverage"]
        ),
        AgentDefinition(
            name="devops",
            description="DevOps and infrastructure specialist",
            domains=["devops", "infrastructure", "deployment"],
            priority=AgentPriority.NORMAL,
            capabilities=["ci_cd", "containerization", "cloud_config"]
        ),
        AgentDefinition(
            name="security",
            description="Security specialist",
            domains=["security"],
            priority=AgentPriority.CRITICAL,
            capabilities=["vulnerability_scan", "auth_design", "encryption"]
        ),
        AgentDefinition(
            name="reviewer",
            description="Code review specialist",
            domains=["review", "quality"],
            priority=AgentPriority.NORMAL,
            capabilities=["code_review", "best_practices", "refactoring"]
        ),
        AgentDefinition(
            name="db_engineer",
            description="Database specialist",
            domains=["database", "data"],
            priority=AgentPriority.HIGH,
            capabilities=["schema_design", "queries", "optimization"]
        ),
        AgentDefinition(
            name="architect",
            description="Software architect",
            domains=["architecture", "design"],
            priority=AgentPriority.HIGH,
            capabilities=["system_design", "patterns", "scalability"]
        ),
    ]

    def __init__(self, circuit_breaker_threshold: int = 3):
        """Initialize the agent registry.

        Args:
            circuit_breaker_threshold: Number of consecutive failures to open circuit breaker
        """
        self._agents: Dict[str, AgentDefinition] = {}
        self._health: Dict[str, AgentHealthMetrics] = {}
        self._lock = threading.RLock()
        self._circuit_breaker_threshold = circuit_breaker_threshold
        self._circuit_breaker_reset_time = timedelta(minutes=5)

        # Register default agents
        for agent_def in self.DEFAULT_AGENTS:
            self.register(agent_def)

    def register(self, agent: AgentDefinition) -> bool:
        """Register a new agent type.

        Args:
            agent: Agent definition to register

        Returns:
            True if registered successfully
        """
        with self._lock:
            if agent.name in self._agents:
                logger.warning(f"Agent '{agent.name}' already registered, updating")

            self._agents[agent.name] = agent
            if agent.name not in self._health:
                self._health[agent.name] = AgentHealthMetrics()

            logger.info(f"Registered agent: {agent.name}")
            return True

    def unregister(self, agent_name: str) -> bool:
        """Unregister an agent type.

        Args:
            agent_name: Name of agent to unregister

        Returns:
            True if unregistered successfully
        """
        with self._lock:
            if agent_name in self._agents:
                del self._agents[agent_name]
                logger.info(f"Unregistered agent: {agent_name}")
                return True
            return False

    def get(self, agent_name: str) -> Optional[AgentDefinition]:
        """Get an agent definition by name."""
        with self._lock:
            return self._agents.get(agent_name)

    def list_agents(self, enabled_only: bool = True) -> List[AgentDefinition]:
        """List all registered agents."""
        with self._lock:
            agents = list(self._agents.values())
            if enabled_only:
                agents = [a for a in agents if a.enabled]
            return sorted(agents, key=lambda a: a.priority.value)

    def get_agents_for_domain(self, domain: str) -> List[AgentDefinition]:
        """Get agents that handle a specific domain."""
        with self._lock:
            return [
                agent for agent in self._agents.values()
                if domain in agent.domains and agent.enabled
            ]

    def get_agents_with_capability(self, capability: str) -> List[AgentDefinition]:
        """Get agents with a specific capability."""
        with self._lock:
            return [
                agent for agent in self._agents.values()
                if capability in agent.capabilities and agent.enabled
            ]

    def prioritize_agents(self, agent_names: List[str]) -> List[str]:
        """Sort agent names by priority.

        Args:
            agent_names: List of agent names to sort

        Returns:
            Sorted list with higher priority agents first
        """
        with self._lock:
            def get_priority(name: str) -> int:
                agent = self._agents.get(name)
                if agent:
                    return agent.priority.value
                return AgentPriority.NORMAL.value

            return sorted(agent_names, key=get_priority)

    # === Health Monitoring ===

    def record_success(self, agent_name: str, duration: float):
        """Record a successful agent call."""
        with self._lock:
            if agent_name not in self._health:
                self._health[agent_name] = AgentHealthMetrics()
            self._health[agent_name].record_success(duration)

    def record_failure(self, agent_name: str, error: str, duration: float = 0.0):
        """Record a failed agent call."""
        with self._lock:
            if agent_name not in self._health:
                self._health[agent_name] = AgentHealthMetrics()

            metrics = self._health[agent_name]
            metrics.record_failure(error, duration)

            # Check circuit breaker
            if metrics.consecutive_failures >= self._circuit_breaker_threshold:
                metrics.circuit_breaker_open = True
                metrics.circuit_breaker_opens_at = datetime.now()
                logger.warning(f"Circuit breaker opened for agent: {agent_name}")

    def get_health(self, agent_name: str) -> AgentHealthMetrics:
        """Get health metrics for an agent."""
        with self._lock:
            if agent_name not in self._health:
                self._health[agent_name] = AgentHealthMetrics()
            return self._health[agent_name]

    def get_health_status(self, agent_name: str) -> AgentHealth:
        """Get current health status of an agent."""
        with self._lock:
            metrics = self.get_health(agent_name)

            # Check if circuit breaker should reset
            if metrics.circuit_breaker_open and metrics.circuit_breaker_opens_at:
                if datetime.now() - metrics.circuit_breaker_opens_at > self._circuit_breaker_reset_time:
                    metrics.circuit_breaker_open = False
                    metrics.circuit_breaker_opens_at = None
                    logger.info(f"Circuit breaker reset for agent: {agent_name}")

            return metrics.get_health_status()

    def is_agent_available(self, agent_name: str) -> bool:
        """Check if an agent is available for use.

        Returns False if:
        - Agent is not registered
        - Agent is disabled
        - Circuit breaker is open
        """
        with self._lock:
            agent = self._agents.get(agent_name)
            if not agent or not agent.enabled:
                return False

            # Check circuit breaker
            status = self.get_health_status(agent_name)
            return status != AgentHealth.UNHEALTHY

    def get_all_health(self) -> Dict[str, Dict[str, Any]]:
        """Get health status for all agents."""
        with self._lock:
            result = {}
            for agent_name in self._agents:
                metrics = self.get_health(agent_name)
                status = self.get_health_status(agent_name)
                result[agent_name] = {
                    "status": status.value,
                    "success_rate": metrics.success_rate,
                    "total_calls": metrics.total_calls,
                    "average_duration": metrics.average_duration,
                    "last_error": metrics.last_error,
                    "circuit_breaker_open": metrics.circuit_breaker_open
                }
            return result

    def reset_health(self, agent_name: Optional[str] = None):
        """Reset health metrics.

        Args:
            agent_name: If provided, reset only this agent. Otherwise reset all.
        """
        with self._lock:
            if agent_name:
                self._health[agent_name] = AgentHealthMetrics()
            else:
                self._health = {name: AgentHealthMetrics() for name in self._agents}

    # === Agent Selection ===

    def select_best_agent(
        self,
        domains: List[str],
        capabilities: Optional[List[str]] = None,
        exclude: Optional[Set[str]] = None
    ) -> Optional[str]:
        """Select the best available agent for given requirements.

        Args:
            domains: Required domains
            capabilities: Required capabilities (optional)
            exclude: Agent names to exclude

        Returns:
            Name of best agent, or None if no suitable agent found
        """
        exclude = exclude or set()

        with self._lock:
            candidates = []

            for agent in self._agents.values():
                if not agent.enabled or agent.name in exclude:
                    continue

                if not self.is_agent_available(agent.name):
                    continue

                # Check domain match
                domain_match = sum(1 for d in domains if d in agent.domains)
                if domain_match == 0:
                    continue

                # Check capability match
                cap_match = 0
                if capabilities:
                    cap_match = sum(1 for c in capabilities if c in agent.capabilities)

                # Score: lower priority value is better, more matches is better
                score = (agent.priority.value, -domain_match, -cap_match)
                candidates.append((score, agent.name))

            if not candidates:
                # Fall back to generalist
                if "generalist" in self._agents and self.is_agent_available("generalist"):
                    return "generalist"
                return None

            # Return agent with best score
            candidates.sort()
            return candidates[0][1]


# Global registry instance
_global_registry: Optional[AgentRegistry] = None
_registry_lock = threading.Lock()


def get_registry() -> AgentRegistry:
    """Get the global agent registry instance."""
    global _global_registry
    with _registry_lock:
        if _global_registry is None:
            _global_registry = AgentRegistry()
        return _global_registry


def register_agent(agent: AgentDefinition) -> bool:
    """Register an agent in the global registry."""
    return get_registry().register(agent)


def get_agent(name: str) -> Optional[AgentDefinition]:
    """Get an agent from the global registry."""
    return get_registry().get(name)
