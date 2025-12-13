"""
Experience Learner - Incremental learning from execution history.

Capabilities:
- Learn from successful and failed task executions
- Improve agent selection based on past performance
- Refine scope estimation accuracy
- Build patterns for common failure modes
- Suggest optimizations based on historical data

This addresses a key gap: existing systems don't learn from
their own execution history to improve future performance.
"""

from __future__ import annotations

import json
import logging
import os
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import threading
import hashlib

logger = logging.getLogger(__name__)


@dataclass
class ExecutionExperience:
    """Record of a task execution experience."""
    task_id: str
    task_type: str
    task_description: str
    agent_id: str
    success: bool
    duration_seconds: float
    retries: int = 0
    error_message: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    context_hash: str = ""  # Hash of relevant context
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentPerformance:
    """Performance metrics for an agent."""
    agent_id: str
    total_tasks: int = 0
    successful_tasks: int = 0
    failed_tasks: int = 0
    total_duration: float = 0.0
    avg_retries: float = 0.0
    task_type_performance: Dict[str, Dict[str, int]] = field(default_factory=dict)
    # task_type -> {"success": n, "failure": m}

    @property
    def success_rate(self) -> float:
        if self.total_tasks == 0:
            return 0.0
        return self.successful_tasks / self.total_tasks

    @property
    def avg_duration(self) -> float:
        if self.total_tasks == 0:
            return 0.0
        return self.total_duration / self.total_tasks


@dataclass
class FailurePattern:
    """Pattern learned from failures."""
    pattern_id: str
    error_signature: str  # Simplified error message
    task_types: List[str]
    occurrence_count: int = 0
    successful_recoveries: int = 0
    recovery_strategies: List[str] = field(default_factory=list)
    context_factors: List[str] = field(default_factory=list)


@dataclass
class ScopeEstimateAccuracy:
    """Track accuracy of scope estimates."""
    estimate_type: str  # "complexity", "duration", "agent_count"
    predicted: str
    actual: str
    task_description_hash: str
    accurate: bool
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


class ExperienceLearner:
    """Learns from execution history to improve future performance.

    Key features:
    - Tracks agent performance by task type
    - Identifies failure patterns and recovery strategies
    - Improves scope estimation accuracy
    - Suggests optimal agent selection
    """

    def __init__(
        self,
        storage_path: Optional[str] = None,
        min_samples_for_learning: int = 3
    ):
        """Initialize experience learner.

        Args:
            storage_path: Path to store learned data
            min_samples_for_learning: Minimum samples before making recommendations
        """
        self.storage_path = Path(storage_path) if storage_path else Path(".agent_memory/experience")
        self.min_samples = min_samples_for_learning
        self._lock = threading.RLock()

        # In-memory state
        self._experiences: List[ExecutionExperience] = []
        self._agent_performance: Dict[str, AgentPerformance] = {}
        self._failure_patterns: Dict[str, FailurePattern] = {}
        self._scope_accuracy: List[ScopeEstimateAccuracy] = []
        self._task_type_agent_map: Dict[str, Dict[str, float]] = defaultdict(dict)
        # task_type -> agent_id -> success_rate

        # Load existing data
        self._load()

    def record_experience(
        self,
        task_id: str,
        task_type: str,
        task_description: str,
        agent_id: str,
        success: bool,
        duration_seconds: float,
        retries: int = 0,
        error_message: str = "",
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Record an execution experience.

        Args:
            task_id: Unique task identifier
            task_type: Type of task
            task_description: Task description
            agent_id: Agent that executed the task
            success: Whether execution succeeded
            duration_seconds: How long it took
            retries: Number of retries needed
            error_message: Error if failed
            metadata: Additional metadata
        """
        experience = ExecutionExperience(
            task_id=task_id,
            task_type=task_type,
            task_description=task_description,
            agent_id=agent_id,
            success=success,
            duration_seconds=duration_seconds,
            retries=retries,
            error_message=error_message,
            context_hash=self._hash_context(task_description, task_type),
            metadata=metadata or {}
        )

        with self._lock:
            self._experiences.append(experience)
            self._update_agent_performance(experience)
            self._update_task_type_mapping(experience)

            if not success and error_message:
                self._update_failure_patterns(experience)

            self._save()

    def _update_agent_performance(self, exp: ExecutionExperience) -> None:
        """Update agent performance metrics."""
        if exp.agent_id not in self._agent_performance:
            self._agent_performance[exp.agent_id] = AgentPerformance(
                agent_id=exp.agent_id
            )

        perf = self._agent_performance[exp.agent_id]
        perf.total_tasks += 1
        perf.total_duration += exp.duration_seconds

        if exp.success:
            perf.successful_tasks += 1
        else:
            perf.failed_tasks += 1

        # Update average retries
        total_retries = perf.avg_retries * (perf.total_tasks - 1) + exp.retries
        perf.avg_retries = total_retries / perf.total_tasks

        # Update task type performance
        if exp.task_type not in perf.task_type_performance:
            perf.task_type_performance[exp.task_type] = {"success": 0, "failure": 0}

        if exp.success:
            perf.task_type_performance[exp.task_type]["success"] += 1
        else:
            perf.task_type_performance[exp.task_type]["failure"] += 1

    def _update_task_type_mapping(self, exp: ExecutionExperience) -> None:
        """Update task type to agent mapping."""
        task_type = exp.task_type
        agent_id = exp.agent_id

        if agent_id not in self._task_type_agent_map[task_type]:
            self._task_type_agent_map[task_type][agent_id] = 0.0

        # Calculate running success rate
        perf = self._agent_performance.get(agent_id)
        if perf and task_type in perf.task_type_performance:
            type_perf = perf.task_type_performance[task_type]
            total = type_perf["success"] + type_perf["failure"]
            if total > 0:
                self._task_type_agent_map[task_type][agent_id] = type_perf["success"] / total

    def _update_failure_patterns(self, exp: ExecutionExperience) -> None:
        """Learn from failure to identify patterns."""
        signature = self._extract_error_signature(exp.error_message)
        pattern_id = self._hash_context(signature, exp.task_type)

        if pattern_id not in self._failure_patterns:
            self._failure_patterns[pattern_id] = FailurePattern(
                pattern_id=pattern_id,
                error_signature=signature,
                task_types=[exp.task_type]
            )

        pattern = self._failure_patterns[pattern_id]
        pattern.occurrence_count += 1

        if exp.task_type not in pattern.task_types:
            pattern.task_types.append(exp.task_type)

    def _extract_error_signature(self, error: str) -> str:
        """Extract simplified error signature."""
        # Remove specific values but keep structure
        import re

        # Remove file paths
        signature = re.sub(r'["\']?/[\w/.-]+["\']?', '<PATH>', error)
        # Remove line numbers
        signature = re.sub(r'line \d+', 'line N', signature)
        # Remove specific values
        signature = re.sub(r'\b\d+\b', 'N', signature)
        # Truncate
        return signature[:200]

    def _hash_context(self, *args) -> str:
        """Create hash of context elements."""
        content = "|".join(str(a) for a in args)
        return hashlib.md5(content.encode()).hexdigest()[:12]

    def record_scope_accuracy(
        self,
        estimate_type: str,
        predicted: str,
        actual: str,
        task_description: str
    ) -> None:
        """Record accuracy of a scope estimate.

        Args:
            estimate_type: Type of estimate (complexity, duration, etc.)
            predicted: What was predicted
            actual: What actually happened
            task_description: Task description for context
        """
        accuracy = ScopeEstimateAccuracy(
            estimate_type=estimate_type,
            predicted=predicted,
            actual=actual,
            task_description_hash=self._hash_context(task_description),
            accurate=(predicted == actual)
        )

        with self._lock:
            self._scope_accuracy.append(accuracy)
            self._save()

    def record_recovery_success(
        self,
        error_signature: str,
        task_type: str,
        recovery_strategy: str
    ) -> None:
        """Record successful recovery from a failure.

        Args:
            error_signature: Simplified error that was recovered from
            task_type: Type of task
            recovery_strategy: What strategy worked
        """
        pattern_id = self._hash_context(
            self._extract_error_signature(error_signature),
            task_type
        )

        with self._lock:
            if pattern_id in self._failure_patterns:
                pattern = self._failure_patterns[pattern_id]
                pattern.successful_recoveries += 1
                if recovery_strategy not in pattern.recovery_strategies:
                    pattern.recovery_strategies.append(recovery_strategy)
                self._save()

    def suggest_agent(
        self,
        task_type: str,
        available_agents: List[str]
    ) -> Tuple[str, float]:
        """Suggest best agent for a task type.

        Args:
            task_type: Type of task
            available_agents: List of available agent IDs

        Returns:
            Tuple of (suggested_agent_id, confidence)
        """
        with self._lock:
            if task_type not in self._task_type_agent_map:
                return (available_agents[0] if available_agents else "", 0.0)

            agent_scores = self._task_type_agent_map[task_type]

            # Filter to available agents
            candidates = [
                (agent, score) for agent, score in agent_scores.items()
                if agent in available_agents
            ]

            if not candidates:
                return (available_agents[0] if available_agents else "", 0.0)

            # Check if we have enough data
            best_agent, best_score = max(candidates, key=lambda x: x[1])
            perf = self._agent_performance.get(best_agent)

            if perf and task_type in perf.task_type_performance:
                total = (perf.task_type_performance[task_type]["success"] +
                        perf.task_type_performance[task_type]["failure"])
                if total < self.min_samples:
                    return (best_agent, 0.3)  # Low confidence

            return (best_agent, min(best_score + 0.2, 1.0))  # Add confidence boost

    def suggest_recovery_strategy(
        self,
        error_message: str,
        task_type: str
    ) -> Optional[str]:
        """Suggest recovery strategy based on past successes.

        Args:
            error_message: The error encountered
            task_type: Type of task

        Returns:
            Suggested recovery strategy or None
        """
        signature = self._extract_error_signature(error_message)
        pattern_id = self._hash_context(signature, task_type)

        with self._lock:
            pattern = self._failure_patterns.get(pattern_id)
            if pattern and pattern.recovery_strategies:
                # Return most successful strategy (first one that worked)
                return pattern.recovery_strategies[0]

            # Try to find similar patterns
            for pid, p in self._failure_patterns.items():
                if (task_type in p.task_types and
                    p.recovery_strategies and
                    self._similar_signature(signature, p.error_signature)):
                    return p.recovery_strategies[0]

        return None

    def _similar_signature(self, sig1: str, sig2: str) -> bool:
        """Check if two error signatures are similar."""
        # Simple word overlap check
        words1 = set(sig1.lower().split())
        words2 = set(sig2.lower().split())
        if not words1 or not words2:
            return False
        overlap = len(words1 & words2) / max(len(words1), len(words2))
        return overlap > 0.5

    def get_agent_performance(
        self,
        agent_id: str
    ) -> Optional[AgentPerformance]:
        """Get performance metrics for an agent."""
        with self._lock:
            return self._agent_performance.get(agent_id)

    def get_scope_accuracy_stats(self) -> Dict[str, Dict[str, float]]:
        """Get accuracy statistics for scope estimates."""
        with self._lock:
            stats: Dict[str, Dict[str, Any]] = defaultdict(lambda: {"correct": 0, "total": 0})

            for acc in self._scope_accuracy:
                stats[acc.estimate_type]["total"] += 1
                if acc.accurate:
                    stats[acc.estimate_type]["correct"] += 1

            return {
                k: {
                    "accuracy": v["correct"] / v["total"] if v["total"] > 0 else 0.0,
                    "samples": v["total"]
                }
                for k, v in stats.items()
            }

    def get_common_failure_patterns(
        self,
        task_type: Optional[str] = None,
        limit: int = 10
    ) -> List[FailurePattern]:
        """Get most common failure patterns.

        Args:
            task_type: Filter by task type (optional)
            limit: Maximum patterns to return

        Returns:
            List of failure patterns sorted by occurrence
        """
        with self._lock:
            patterns = list(self._failure_patterns.values())

            if task_type:
                patterns = [p for p in patterns if task_type in p.task_types]

            patterns.sort(key=lambda p: p.occurrence_count, reverse=True)
            return patterns[:limit]

    def get_learning_summary(self) -> Dict[str, Any]:
        """Get summary of learned knowledge."""
        with self._lock:
            return {
                "total_experiences": len(self._experiences),
                "agents_tracked": len(self._agent_performance),
                "failure_patterns_identified": len(self._failure_patterns),
                "scope_accuracy_samples": len(self._scope_accuracy),
                "task_types_learned": len(self._task_type_agent_map),
                "agent_performance": {
                    aid: {
                        "success_rate": p.success_rate,
                        "avg_duration": p.avg_duration,
                        "total_tasks": p.total_tasks
                    }
                    for aid, p in self._agent_performance.items()
                },
                "scope_accuracy": self.get_scope_accuracy_stats()
            }

    def _save(self) -> None:
        """Save learned data to disk."""
        try:
            self.storage_path.mkdir(parents=True, exist_ok=True)

            # Save experiences (last 1000)
            experiences_path = self.storage_path / "experiences.json"
            with open(experiences_path, "w") as f:
                json.dump(
                    [vars(e) for e in self._experiences[-1000:]],
                    f,
                    indent=2
                )

            # Save agent performance
            perf_path = self.storage_path / "agent_performance.json"
            with open(perf_path, "w") as f:
                json.dump(
                    {k: vars(v) for k, v in self._agent_performance.items()},
                    f,
                    indent=2
                )

            # Save failure patterns
            patterns_path = self.storage_path / "failure_patterns.json"
            with open(patterns_path, "w") as f:
                json.dump(
                    {k: vars(v) for k, v in self._failure_patterns.items()},
                    f,
                    indent=2
                )

            # Save task type mapping
            mapping_path = self.storage_path / "task_agent_mapping.json"
            with open(mapping_path, "w") as f:
                json.dump(dict(self._task_type_agent_map), f, indent=2)

        except Exception as e:
            logger.warning(f"Failed to save experience data: {e}")

    def _load(self) -> None:
        """Load learned data from disk."""
        try:
            # Load experiences
            experiences_path = self.storage_path / "experiences.json"
            if experiences_path.exists():
                with open(experiences_path) as f:
                    data = json.load(f)
                    self._experiences = [ExecutionExperience(**e) for e in data]

            # Load agent performance
            perf_path = self.storage_path / "agent_performance.json"
            if perf_path.exists():
                with open(perf_path) as f:
                    data = json.load(f)
                    self._agent_performance = {
                        k: AgentPerformance(**v) for k, v in data.items()
                    }

            # Load failure patterns
            patterns_path = self.storage_path / "failure_patterns.json"
            if patterns_path.exists():
                with open(patterns_path) as f:
                    data = json.load(f)
                    self._failure_patterns = {
                        k: FailurePattern(**v) for k, v in data.items()
                    }

            # Load task type mapping
            mapping_path = self.storage_path / "task_agent_mapping.json"
            if mapping_path.exists():
                with open(mapping_path) as f:
                    self._task_type_agent_map = defaultdict(dict, json.load(f))

            logger.info(f"Loaded {len(self._experiences)} experiences from disk")

        except Exception as e:
            logger.warning(f"Failed to load experience data: {e}")

    def clear(self) -> None:
        """Clear all learned data."""
        with self._lock:
            self._experiences.clear()
            self._agent_performance.clear()
            self._failure_patterns.clear()
            self._scope_accuracy.clear()
            self._task_type_agent_map.clear()

            # Remove files
            for file in self.storage_path.glob("*.json"):
                file.unlink()


def create_experience_learner(
    storage_path: Optional[str] = None,
    min_samples: int = 3
) -> ExperienceLearner:
    """Factory function to create experience learner."""
    return ExperienceLearner(storage_path, min_samples)
