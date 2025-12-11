# ============================================================================
# FILE: short_term_memory.py
# Working memory for active tasks and recent context
# ============================================================================

from typing import List, Optional, Dict, Any
from collections import deque
from datetime import datetime

from .memory_types import MemoryEntry, MemoryPriority


class ShortTermMemory:
    """
    Working memory for agents. Stores:
    - Recent task assignments
    - Current file context
    - Recent decisions and outputs
    - Active debugging context
    """

    def __init__(self, max_size: int = 50, context_window: int = 10):
        self.max_size = max_size
        self.context_window = context_window
        self.memories: deque = deque(maxlen=max_size)
        self.active_context: Dict[str, Any] = {}

    def add(self, entry: MemoryEntry):
        """Add entry to working memory"""
        self.memories.append(entry)

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
        n = n or self.context_window
        memories = list(self.memories)

        if agent_id:
            memories = [m for m in memories if m.agent_id == agent_id]

        return sorted(memories, key=lambda x: x.timestamp, reverse=True)[:n]

    def get_by_tags(self, tags: List[str]) -> List[MemoryEntry]:
        """Get memories matching any of the tags"""
        return [m for m in self.memories if any(tag in m.tags for tag in tags)]

    def get_by_priority(self, min_priority: MemoryPriority) -> List[MemoryEntry]:
        """Get memories at or above priority level"""
        return [m for m in self.memories if m.priority.value >= min_priority.value]

    def get_active_files(self) -> Dict[str, str]:
        """Get currently active file contexts"""
        return self.active_context.copy()

    def get_file_content(self, filename: str) -> Optional[str]:
        """Get content of a specific file from active context"""
        return self.active_context.get(filename)

    def clear_agent_context(self, agent_id: str):
        """Clear working memory for specific agent"""
        self.memories = deque(
            [m for m in self.memories if m.agent_id != agent_id],
            maxlen=self.max_size
        )
        # Clear active context related to this agent
        to_remove = []
        for filename in self.active_context:
            related = [m for m in self.memories if filename in m.related_files and m.agent_id == agent_id]
            if not related:
                to_remove.append(filename)
        for filename in to_remove:
            del self.active_context[filename]

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
        self.memories.clear()
        self.active_context.clear()

    def __len__(self) -> int:
        return len(self.memories)

    def to_dict(self) -> Dict:
        """Export for persistence"""
        return {
            "memories": [m.to_dict() for m in self.memories],
            "active_context": self.active_context,
            "max_size": self.max_size,
            "context_window": self.context_window
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "ShortTermMemory":
        """Restore from persisted data"""
        instance = cls(
            max_size=data.get("max_size", 50),
            context_window=data.get("context_window", 10)
        )
        for mem_data in data.get("memories", []):
            instance.add(MemoryEntry.from_dict(mem_data))
        instance.active_context = data.get("active_context", {})
        return instance
