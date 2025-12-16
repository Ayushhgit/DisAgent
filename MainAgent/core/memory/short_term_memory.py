# ============================================================================
# FILE: short_term_memory.py
# Working memory for active tasks and recent context
# With relevance-weighted cleanup and crash persistence
# ============================================================================

from typing import List, Optional, Dict, Any, Callable
from collections import deque
from datetime import datetime, timedelta
from pathlib import Path
import json
import threading
import atexit
import logging

from .memory_types import MemoryEntry, MemoryPriority

logger = logging.getLogger(__name__)


class ShortTermMemory:
    """
    Working memory for agents with intelligent cleanup.

    Stores:
    - Recent task assignments
    - Current file context
    - Recent decisions and outputs
    - Active debugging context

    Features:
    - Relevance-weighted eviction (low priority items removed first)
    - Periodic auto-save for crash recovery
    - Age-based decay for priority scoring
    """

    def __init__(
        self,
        max_size: int = 50,
        context_window: int = 10,
        persistence_path: Optional[str] = None,
        auto_save_interval: int = 60  # seconds
    ):
        self.max_size = max_size
        self.context_window = context_window
        self._memories: List[MemoryEntry] = []
        self.active_context: Dict[str, Any] = {}
        self._lock = threading.RLock()

        # Persistence
        self.persistence_path = Path(persistence_path) if persistence_path else None
        self._auto_save_interval = auto_save_interval
        self._last_save_time = datetime.now()
        self._dirty = False  # Track if changes need saving

        # Register cleanup on exit
        atexit.register(self._save_on_exit)

    def _calculate_relevance_score(self, entry: MemoryEntry) -> float:
        """Calculate relevance score for eviction decisions.

        Higher score = more relevant = keep longer.
        """
        score = 0.0

        # Priority weight (1-4 based on priority level)
        priority_weights = {
            MemoryPriority.LOW: 1,
            MemoryPriority.MEDIUM: 2,
            MemoryPriority.HIGH: 3,
            MemoryPriority.CRITICAL: 4
        }
        score += priority_weights.get(entry.priority, 2) * 10

        # Access count bonus
        score += min(entry.access_count * 2, 20)

        # Recency bonus (decay over time)
        age_hours = (datetime.now() - entry.timestamp).total_seconds() / 3600
        recency_score = max(0, 20 - age_hours * 2)  # Decays over ~10 hours
        score += recency_score

        # Tag bonuses
        important_tags = {"error", "decision", "critical", "output"}
        if any(tag in important_tags for tag in entry.tags):
            score += 15

        # File association bonus
        if entry.related_files:
            score += 5

        return score

    def _evict_if_needed(self):
        """Evict lowest relevance items if at capacity."""
        with self._lock:
            while len(self._memories) >= self.max_size:
                # Find item with lowest relevance
                if not self._memories:
                    break

                min_score = float('inf')
                min_idx = 0

                for i, entry in enumerate(self._memories):
                    score = self._calculate_relevance_score(entry)
                    if score < min_score:
                        min_score = score
                        min_idx = i

                # Remove lowest relevance item
                evicted = self._memories.pop(min_idx)
                logger.debug(f"Evicted memory: {evicted.content[:50]}... (score: {min_score:.1f})")

    def add(self, entry: MemoryEntry):
        """Add entry to working memory with intelligent eviction."""
        with self._lock:
            self._evict_if_needed()
            self._memories.append(entry)
            self._dirty = True
            self._maybe_auto_save()

    def add_decision(self, agent_id: str, decision: str, context: str,
                     priority: MemoryPriority = MemoryPriority.MEDIUM):
        """Record an agent decision"""
        entry = MemoryEntry(
            content=f"DECISION: {decision}\nCONTEXT: {context}",
            timestamp=datetime.now(),
            priority=priority,
            agent_id=agent_id,
            tags=["decision"]
        )
        self.add(entry)

    def add_code_context(self, agent_id: str, filename: str, content: str, purpose: str):
        """Add code file to working context"""
        with self._lock:
            # Truncate content for memory entry but keep full in active_context
            truncated = content[:500] + "..." if len(content) > 500 else content
            entry = MemoryEntry(
                content=f"FILE: {filename}\nPURPOSE: {purpose}\n---\n{truncated}",
                timestamp=datetime.now(),
                priority=MemoryPriority.HIGH,
                agent_id=agent_id,
                tags=["code", "context"],
                related_files=[filename]
            )
            self.add(entry)
            self.active_context[filename] = content

    def add_agent_output(self, agent_id: str, output: str, task_description: str):
        """Record agent output for context sharing"""
        entry = MemoryEntry(
            content=f"TASK: {task_description}\nOUTPUT: {output[:1000]}",
            timestamp=datetime.now(),
            priority=MemoryPriority.HIGH,
            agent_id=agent_id,
            tags=["output", "agent_result"]
        )
        self.add(entry)

    def add_error(self, agent_id: str, error: str, context: str):
        """Record an error for debugging"""
        entry = MemoryEntry(
            content=f"ERROR: {error}\nCONTEXT: {context}",
            timestamp=datetime.now(),
            priority=MemoryPriority.CRITICAL,
            agent_id=agent_id,
            tags=["error", "debug"]
        )
        self.add(entry)

    def get_recent(self, n: Optional[int] = None, agent_id: Optional[str] = None) -> List[MemoryEntry]:
        """Get n most recent memories, optionally filtered by agent"""
        with self._lock:
            n = n or self.context_window
            memories = list(self._memories)

            if agent_id:
                memories = [m for m in memories if m.agent_id == agent_id]

            # Sort by timestamp and increment access count
            result = sorted(memories, key=lambda x: x.timestamp, reverse=True)[:n]
            for entry in result:
                entry.access_count += 1

            return result

    def get_by_tags(self, tags: List[str]) -> List[MemoryEntry]:
        """Get memories matching any of the tags"""
        with self._lock:
            return [m for m in self._memories if any(tag in m.tags for tag in tags)]

    def get_by_priority(self, min_priority: MemoryPriority) -> List[MemoryEntry]:
        """Get memories at or above priority level"""
        with self._lock:
            return [m for m in self._memories if m.priority.value >= min_priority.value]

    def get_active_files(self) -> Dict[str, str]:
        """Get currently active file contexts"""
        with self._lock:
            return self.active_context.copy()

    def get_file_content(self, filename: str) -> Optional[str]:
        """Get content of a specific file from active context"""
        with self._lock:
            return self.active_context.get(filename)

    def clear_agent_context(self, agent_id: str):
        """Clear working memory for specific agent"""
        with self._lock:
            self._memories = [m for m in self._memories if m.agent_id != agent_id]
            self._dirty = True

            # Clear active context related to this agent
            to_remove = []
            for filename in self.active_context:
                related = [m for m in self._memories
                          if filename in m.related_files and m.agent_id == agent_id]
                if not related:
                    to_remove.append(filename)
            for filename in to_remove:
                del self.active_context[filename]

    def cleanup_old_entries(self, max_age_hours: float = 24.0):
        """Remove entries older than max_age_hours with low priority."""
        with self._lock:
            cutoff = datetime.now() - timedelta(hours=max_age_hours)
            before_count = len(self._memories)

            self._memories = [
                m for m in self._memories
                if m.timestamp > cutoff or m.priority in (MemoryPriority.HIGH, MemoryPriority.CRITICAL)
            ]

            removed = before_count - len(self._memories)
            if removed > 0:
                logger.info(f"Cleaned up {removed} old memory entries")
                self._dirty = True

    def summarize_context(self, agent_id: Optional[str] = None) -> str:
        """Generate summary of recent context"""
        recent = self.get_recent(self.context_window, agent_id)
        if not recent:
            return "No recent context available"

        summary = f"=== RECENT CONTEXT ({len(recent)} entries) ===\n\n"
        for entry in recent:
            summary += f"[{entry.agent_id}] {entry.timestamp.strftime('%H:%M:%S')} "
            summary += f"[{entry.priority.name}]\n"
            # Truncate long content
            content = entry.content[:200] + "..." if len(entry.content) > 200 else entry.content
            summary += f"{content}\n\n"

        return summary

    def get_decisions(self, agent_id: Optional[str] = None) -> List[MemoryEntry]:
        """Get all recorded decisions"""
        decisions = self.get_by_tags(["decision"])
        if agent_id:
            decisions = [d for d in decisions if d.agent_id == agent_id]
        return decisions

    def get_errors(self) -> List[MemoryEntry]:
        """Get all recorded errors"""
        return self.get_by_tags(["error"])

    def clear(self):
        """Clear all working memory"""
        with self._lock:
            self._memories.clear()
            self.active_context.clear()
            self._dirty = True

    def __len__(self) -> int:
        with self._lock:
            return len(self._memories)

    # === Persistence Methods ===

    def _maybe_auto_save(self):
        """Auto-save if enough time has passed since last save."""
        if not self.persistence_path or not self._dirty:
            return

        elapsed = (datetime.now() - self._last_save_time).total_seconds()
        if elapsed >= self._auto_save_interval:
            self.save_to_disk()

    def save_to_disk(self) -> bool:
        """Save current state to disk for crash recovery."""
        if not self.persistence_path:
            return False

        with self._lock:
            try:
                self.persistence_path.parent.mkdir(parents=True, exist_ok=True)
                data = self.to_dict()

                # Write atomically using temp file
                temp_path = self.persistence_path.with_suffix('.tmp')
                with open(temp_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2, default=str)

                # Atomic rename
                temp_path.replace(self.persistence_path)

                self._last_save_time = datetime.now()
                self._dirty = False
                logger.debug(f"STM saved to {self.persistence_path}")
                return True
            except Exception as e:
                logger.warning(f"Failed to save STM: {e}")
                return False

    def load_from_disk(self) -> bool:
        """Load state from disk."""
        if not self.persistence_path or not self.persistence_path.exists():
            return False

        with self._lock:
            try:
                with open(self.persistence_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                # Restore memories
                self._memories.clear()
                for mem_data in data.get("memories", []):
                    self._memories.append(MemoryEntry.from_dict(mem_data))

                self.active_context = data.get("active_context", {})
                self.max_size = data.get("max_size", self.max_size)
                self.context_window = data.get("context_window", self.context_window)

                self._dirty = False
                logger.info(f"STM loaded from {self.persistence_path}: {len(self._memories)} entries")
                return True
            except Exception as e:
                logger.warning(f"Failed to load STM: {e}")
                return False

    def _save_on_exit(self):
        """Called at exit to save state."""
        if self._dirty and self.persistence_path:
            self.save_to_disk()

    def to_dict(self) -> Dict:
        """Export for persistence"""
        with self._lock:
            return {
                "memories": [m.to_dict() for m in self._memories],
                "active_context": self.active_context,
                "max_size": self.max_size,
                "context_window": self.context_window,
                "saved_at": datetime.now().isoformat()
            }

    @classmethod
    def from_dict(cls, data: Dict) -> "ShortTermMemory":
        """Restore from persisted data"""
        instance = cls(
            max_size=data.get("max_size", 50),
            context_window=data.get("context_window", 10)
        )
        for mem_data in data.get("memories", []):
            instance._memories.append(MemoryEntry.from_dict(mem_data))
        instance.active_context = data.get("active_context", {})
        return instance

    def get_stats(self) -> Dict[str, Any]:
        """Get memory statistics."""
        with self._lock:
            priority_counts = {}
            for m in self._memories:
                p = m.priority.name
                priority_counts[p] = priority_counts.get(p, 0) + 1

            return {
                "total_entries": len(self._memories),
                "max_size": self.max_size,
                "active_files": len(self.active_context),
                "by_priority": priority_counts,
                "dirty": self._dirty
            }
