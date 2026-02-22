"""
Experience Learner - Cognitive Memory System for Code Agents.

Implements a Tier-2 Cognitive Architecture with:
1. Version-Aware Episodic Memory ("Hippocampal Traces")
2. Semantic Rule Consolidation ("Neocortical Schemas" / Sleep Phase)
3. Interference-Aware Retrieval (Active Forgetting)
4. Active Memory Grounding (Curiosity-Driven Verification)

Bio-Inspired Memory Model:
- Awake: Record raw episodes (Task -> Action -> Outcome)
- Sleep: Consolidate repeated successes into abstract Semantic Rules
- Retrieval: Gated by Environment Hash (Version Awareness)
- Curiosity: Probe stale memories for validity when drift is high
"""

from __future__ import annotations

import json
import logging
import os
import threading
import hashlib
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Set

logger = logging.getLogger(__name__)


@dataclass
class ResearchMetrics:
    """metrics for academic evaluation."""
    pass_at_1: bool = False
    retry_count: int = 0
    tokens_used: int = 0
    execution_time_ms: float = 0.0
    cost_estimate_usd: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "pass_at_1": self.pass_at_1,
            "retry_count": self.retry_count,
            "tokens_used": self.tokens_used,
            "execution_time_ms": self.execution_time_ms,
            "cost_estimate_usd": self.cost_estimate_usd
        }


@dataclass
class SemanticRule:
    """
    Tier-2 Memory: Abstracted knowledge derived from multiple episodes.
    Represents the 'Neocortical Schema' (Semantization of experience).
    """
    rule_id: str
    trigger_error_type: str
    trigger_task_type: str
    suggested_strategy: str 
    confidence: float = 1.0
    source_episode_ids: List[str] = field(default_factory=list)
    valid_environments: Set[str] = field(default_factory=set)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    
    # Interference-Aware Fields (Novelty)
    retrieval_count: int = 0  # How often this rule is retrieved
    success_after_retrieval: int = 0  # How often it led to success
    interference_score: float = 0.0  # Higher = more likely to cause interference (noisy)
    last_grounded_at: Optional[str] = None  # When was this rule last verified?

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rule_id": self.rule_id,
            "trigger_error_type": self.trigger_error_type,
            "trigger_task_type": self.trigger_task_type,
            "suggested_strategy": self.suggested_strategy,
            "confidence": self.confidence,
            "source_episode_ids": self.source_episode_ids,
            "valid_environments": list(self.valid_environments),
            "created_at": self.created_at,
            "retrieval_count": self.retrieval_count,
            "success_after_retrieval": self.success_after_retrieval,
            "interference_score": self.interference_score,
            "last_grounded_at": self.last_grounded_at
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SemanticRule":
        data = data.copy()
        data["valid_environments"] = set(data.get("valid_environments", []))
        # Handle new fields with defaults for backward compat
        data.setdefault("retrieval_count", 0)
        data.setdefault("success_after_retrieval", 0)
        data.setdefault("interference_score", 0.0)
        data.setdefault("last_grounded_at", None)
        return cls(**data)


@dataclass
class ExecutionExperience:
    """
    Structured record of a single execution episode with Version-Awareness.
    """
    experience_id: str
    task_id: str
    agent_id: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    task_type: str = "general"
    task_description: str = ""
    file_path: Optional[str] = None
    
    # Environment (The Novelty Wedge)
    environment_hash: str = "default_env" 
    
    error_type: Optional[str] = None
    error_context: Optional[str] = None
    
    patch_strategy: str = "default"
    diff_summary: str = ""
    
    success: bool = False
    
    metrics: ResearchMetrics = field(default_factory=ResearchMetrics)
    context_hash: str = ""

    def __post_init__(self):
        if not self.experience_id:
            content = f"{self.task_description}{self.file_path}{self.error_type}{self.environment_hash}"
            self.experience_id = hashlib.md5(content.encode()).hexdigest()[:12]
        
        if not self.context_hash and self.task_type:
            self.context_hash = hashlib.md5(
                f"{self.task_type}:{self.error_type or 'none'}".encode()
            ).hexdigest()[:8]

    def to_dict(self) -> Dict[str, Any]:
        data = {k: v for k, v in vars(self).items() if k != "metrics"}
        data["metrics"] = self.metrics.to_dict()
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExecutionExperience":
        metrics_data = data.pop("metrics", {})
        metrics = ResearchMetrics(**metrics_data)
        return cls(metrics=metrics, **data)


@dataclass
class FailurePattern:
    """
    Aggregated pattern with Version-Aware validity logic.
    """
    pattern_id: str
    error_type: str
    affected_task_types: Set[str] = field(default_factory=set)
    occurrence_count: int = 0
    successful_recoveries: int = 0
    best_strategies: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    
    valid_environments: Set[str] = field(default_factory=set)
    last_verified_at: str = field(default_factory=lambda: datetime.now().isoformat())
    
    associated_rules: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "pattern_id": self.pattern_id,
            "error_type": self.error_type,
            "affected_task_types": list(self.affected_task_types),
            "occurrence_count": self.occurrence_count,
            "successful_recoveries": self.successful_recoveries,
            "best_strategies": dict(self.best_strategies),
            "valid_environments": list(self.valid_environments),
            "last_verified_at": self.last_verified_at,
            "associated_rules": self.associated_rules
        }


class ExperienceLearner:
    """
    Episodic Memory System using Version-Aware Associative Memory.
    """

    def __init__(
        self,
        storage_path: Optional[str] = None,
        min_samples_for_learning: int = 3
    ):
        self.storage_path = Path(storage_path) if storage_path else Path(".agent_memory/experience")
        self.min_samples = min_samples_for_learning
        self._lock = threading.RLock()
        
        self._episodes: List[ExecutionExperience] = []
        self._patterns: Dict[str, FailurePattern] = {}
        self._semantic_rules: Dict[str, SemanticRule] = {}
        
        self._load()

    def record_episode(self, experience: ExecutionExperience) -> None:
        """Store a new episode."""
        with self._lock:
            self._episodes.append(experience)
            if experience.error_type:
                self._update_patterns(experience)
            self._save()

    def consolidate(self) -> int:
        """
        The 'Sleep Phase': Abstract raw episodes into Semantic Rules.
        Returns number of new rules created.
        """
        with self._lock:
            new_rules_count = 0
            
            # Group success episodes by error_type + strategy
            candidates = defaultdict(list)
            for exp in self._episodes:
                if exp.success and exp.patch_strategy and exp.error_type:
                    key = (exp.error_type, exp.task_type, exp.patch_strategy)
                    candidates[key].append(exp)
            
            # Check for critical mass (e.g., > 2 similar successes)
            for (err, task, strat), group in candidates.items():
                if len(group) >= 2:
                    # Create Rule
                    rule_id = hashlib.md5(f"{err}{task}{strat}".encode()).hexdigest()[:12]
                    
                    if rule_id not in self._semantic_rules:
                        rule = SemanticRule(
                            rule_id=rule_id,
                            trigger_error_type=err,
                            trigger_task_type=task,
                            suggested_strategy=strat,
                            source_episode_ids=[e.experience_id for e in group],
                            valid_environments={e.environment_hash for e in group}
                        )
                        self._semantic_rules[rule_id] = rule
                        new_rules_count += 1
                        
                        # Link to pattern
                        pat_id = self._get_pattern_id(err)
                        if pat_id in self._patterns:
                            if rule_id not in self._patterns[pat_id].associated_rules:
                                self._patterns[pat_id].associated_rules.append(rule_id)
                    else:
                        # Reinforce existing rule
                        rule = self._semantic_rules[rule_id]
                        rule.confidence = min(1.5, rule.confidence + 0.1)
                        rule.valid_environments.update(e.environment_hash for e in group)

            self._save()
            return new_rules_count

    def retrieve_similar_experiences(
        self,
        task_type: str,
        error_type: Optional[str] = None,
        file_path: Optional[str] = None,
        current_env_hash: str = "default_env",
        limit: int = 3
    ) -> List[ExecutionExperience]:
        """
        Retrieve relevant past episodes using Drift-Aware Scoring.
        """
        with self._lock:
            candidates = []
            
            for exp in self._episodes:
                score = 0
                if not exp.success: continue

                # 1. Base Relevance
                if error_type and exp.error_type == error_type: score += 10
                if file_path and exp.file_path == file_path: score += 5
                if task_type and exp.task_type == task_type: score += 2
                
                # 2. Version/Environment Drift Penalty
                if exp.environment_hash == current_env_hash:
                    score += 5
                else:
                    score -= 3 # Penalty for staleness
                
                if score > 0:
                    candidates.append((score, exp))
            
            candidates.sort(key=lambda x: (x[0], x[1].timestamp), reverse=True)
            return [c[1] for c in candidates[:limit]]

    def suggest_strategy(
        self,
        task_type: str,
        error_type: str,
        current_env_hash: str = "default_env"
    ) -> Tuple[Optional[str], Optional[str], float]:
        """
        Suggest strategy with Interference-Aware Retrieval.
        
        Returns:
            Tuple of (strategy, rule_id, confidence_score).
            rule_id is None if suggestion comes from episodic memory.
        """
        with self._lock:
            # 1. Check Semantic Rules (Tier-2 Abstraction) with Interference Penalty
            candidates_rules = []
            for rule in self._semantic_rules.values():
                if rule.trigger_error_type == error_type:
                    base_score = rule.confidence
                    
                    # Environment Validity Bonus/Penalty
                    if current_env_hash in rule.valid_environments:
                        base_score *= 2.0
                    else:
                        base_score *= 0.5
                    
                    # Interference Penalty (Novel Mechanism)
                    interference_penalty = self._calculate_interference_penalty(rule)
                    final_score = base_score - interference_penalty
                    
                    # Track retrieval
                    rule.retrieval_count += 1
                    
                    candidates_rules.append((final_score, rule.suggested_strategy, rule.rule_id))
            
            if candidates_rules:
                best = max(candidates_rules, key=lambda x: x[0])
                return (best[1], best[2], best[0])

            # 2. Fallback to Episodic Retrieval (Slow path)
            similar = self.retrieve_similar_experiences(
                task_type, 
                error_type, 
                current_env_hash=current_env_hash,
                limit=5
            )
            
            if similar:
                strategy_scores = defaultdict(float)
                for exp in similar:
                    if not exp.patch_strategy: continue
                    weight = 1.0
                    if exp.environment_hash == current_env_hash:
                        weight = 2.0
                    strategy_scores[exp.patch_strategy] += weight
                
                if strategy_scores:
                    best_strat = max(strategy_scores.items(), key=lambda x: x[1])
                    return (best_strat[0], None, best_strat[1])
            
            return (None, None, 0.0)

    def _calculate_interference_penalty(self, rule: SemanticRule) -> float:
        """
        Calculate interference penalty for a rule.
        
        Interference sources:
        1. High retrieval count with low success rate (unreliable)
        2. Rule is too generic (matches many different contexts)
        3. Rule hasn't been grounded/verified recently
        """
        penalty = 0.0
        
        # 1. Reliability Penalty
        if rule.retrieval_count > 5:
            success_rate = rule.success_after_retrieval / rule.retrieval_count
            if success_rate < 0.5:
                penalty += (1.0 - success_rate) * 0.5  # Max 0.5 penalty
        
        # 2. Genericity Penalty (too many source episodes = too generic)
        if len(rule.source_episode_ids) > 10:
            penalty += 0.2
        
        # 3. Staleness Penalty (not grounded recently)
        if rule.last_grounded_at:
            try:
                grounded_time = datetime.fromisoformat(rule.last_grounded_at)
                days_since_grounding = (datetime.now() - grounded_time).days
                if days_since_grounding > 7:
                    penalty += min(0.3, days_since_grounding * 0.02)
            except:
                pass
        else:
            penalty += 0.1  # Never grounded
        
        return penalty

    def record_retrieval_outcome(
        self, 
        rule_id: str, 
        success: bool,
        current_env_hash: str = "default_env"
    ) -> None:
        """
        Record the outcome of using a suggested strategy.
        This enables the Interference-Aware learning loop.
        """
        with self._lock:
            if rule_id in self._semantic_rules:
                rule = self._semantic_rules[rule_id]
                if success:
                    rule.success_after_retrieval += 1
                    rule.valid_environments.add(current_env_hash)
                    rule.last_grounded_at = datetime.now().isoformat()
                    # Reduce interference score on success
                    rule.interference_score = max(0, rule.interference_score - 0.1)
                else:
                    # Increase interference score on failure
                    rule.interference_score += 0.2
                    # Consider removing environment if it keeps failing
                    if rule.interference_score > 1.0:
                        rule.valid_environments.discard(current_env_hash)
                
                self._save()

    def ground_stale_memory(
        self,
        rule_id: str,
        probe_result: bool,
        current_env_hash: str
    ) -> None:
        """
        Active Memory Grounding: Verify if a stale rule still applies.
        
        When drift is high (current_env not in valid_environments),
        the system can spawn a "probe" to test if the rule still works.
        This method records the outcome of that probe.
        """
        with self._lock:
            if rule_id in self._semantic_rules:
                rule = self._semantic_rules[rule_id]
                rule.last_grounded_at = datetime.now().isoformat()
                
                if probe_result:
                    # Rule is still valid in new environment
                    rule.valid_environments.add(current_env_hash)
                    rule.confidence = min(1.5, rule.confidence + 0.1)
                    rule.interference_score = max(0, rule.interference_score - 0.2)
                    logger.info(f"Grounded rule {rule_id}: VALID in {current_env_hash}")
                else:
                    # Rule does not apply in new environment
                    rule.confidence = max(0.1, rule.confidence - 0.2)
                    rule.interference_score += 0.3
                    logger.info(f"Grounded rule {rule_id}: INVALID in {current_env_hash}")
                
                self._save()

    def get_stale_rules(self, current_env_hash: str, threshold: float = 0.3) -> List[SemanticRule]:
        """
        Get rules that need Active Grounding (high drift + low confidence).
        These are candidates for probe verification.
        """
        with self._lock:
            stale = []
            for rule in self._semantic_rules.values():
                if current_env_hash not in rule.valid_environments:
                    # Calculate "staleness score"
                    staleness = rule.interference_score + (1.0 - rule.confidence)
                    if staleness > threshold:
                        stale.append(rule)
            return stale

    def get_research_metrics(self) -> Dict[str, Any]:
        with self._lock:
            total = len(self._episodes)
            if total == 0: return {}
            
            passed_at_1 = sum(1 for e in self._episodes if e.metrics.pass_at_1)
            total_tokens = sum(e.metrics.tokens_used for e in self._episodes)
            avg_retry = sum(e.metrics.retry_count for e in self._episodes) / total
            
            return {
                "total_episodes": total,
                "pass_at_1_rate": passed_at_1 / total,
                "avg_retry_count": avg_retry,
                "total_tokens_consumed": total_tokens,
                "learned_patterns": len(self._patterns),
                "semantic_rules": len(self._semantic_rules)
            }

    def _update_patterns(self, exp: ExecutionExperience) -> None:
        if not exp.error_type: return
        pid = self._get_pattern_id(exp.error_type)
        if pid not in self._patterns:
            self._patterns[pid] = FailurePattern(pattern_id=pid, error_type=exp.error_type)
        pat = self._patterns[pid]
        pat.affected_task_types.add(exp.task_type)
        pat.occurrence_count += 1
        if exp.success:
            pat.successful_recoveries += 1
            pat.valid_environments.add(exp.environment_hash)
            pat.last_verified_at = datetime.now().isoformat()
            if exp.patch_strategy:
                pat.best_strategies[exp.patch_strategy] += 1

    def _get_pattern_id(self, error_type: str) -> str:
        clean_type = str(error_type).strip().lower()
        return hashlib.md5(clean_type.encode()).hexdigest()[:12]

    def _save(self) -> None:
        try:
            self.storage_path.mkdir(parents=True, exist_ok=True)
            ep_data = [e.to_dict() for e in self._episodes[-1000:]]
            with open(self.storage_path / "detailed_episodes.json", "w") as f:
                json.dump(ep_data, f, indent=2)
            pat_data = {k: v.to_dict() for k, v in self._patterns.items()}
            with open(self.storage_path / "failure_patterns_v2.json", "w") as f:
                json.dump(pat_data, f, indent=2)
            rule_data = {k: v.to_dict() for k, v in self._semantic_rules.items()}
            with open(self.storage_path / "semantic_rules.json", "w") as f:
                json.dump(rule_data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save experience memory: {e}")

    def _load(self) -> None:
        try:
            ep_path = self.storage_path / "detailed_episodes.json"
            if ep_path.exists():
                with open(ep_path) as f:
                    data = json.load(f)
                    self._episodes = [ExecutionExperience.from_dict(d) for d in data]
            pat_path = self.storage_path / "failure_patterns_v2.json"
            if pat_path.exists():
                with open(pat_path) as f:
                    data = json.load(f)
                    for k, v in data.items():
                        pat = FailurePattern(
                            pattern_id=v["pattern_id"],
                            error_type=v["error_type"],
                            occurrence_count=v["occurrence_count"],
                            successful_recoveries=v["successful_recoveries"]
                        )
                        pat.affected_task_types = set(v["affected_task_types"])
                        pat.best_strategies = defaultdict(int, v["best_strategies"])
                        pat.valid_environments = set(v.get("valid_environments", []))
                        pat.last_verified_at = v.get("last_verified_at", datetime.now().isoformat())
                        pat.associated_rules = v.get("associated_rules", [])
                        self._patterns[k] = pat
            rule_path = self.storage_path / "semantic_rules.json"
            if rule_path.exists():
                with open(rule_path) as f:
                    data = json.load(f)
                    for k, v in data.items():
                        self._semantic_rules[k] = SemanticRule.from_dict(v)
        except Exception as e:
            logger.warning(f"Failed to load experience memory: {e}")
            self._episodes = []
            self._patterns = {}
            self._semantic_rules = {}

def create_experience_learner(
    storage_path: Optional[str] = None,
    min_samples: int = 3
) -> ExperienceLearner:
    return ExperienceLearner(storage_path, min_samples)
