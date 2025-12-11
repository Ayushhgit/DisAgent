# memory subpackage
# Unified memory system for multi-agent orchestration

from .memory_types import (
    MemoryPriority,
    AgentState,
    TaskType,
    MemoryEntry,
    ProjectContext,
    TaskInfo,
    AgentStatus,
    StateSnapshot
)

from .short_term_memory import ShortTermMemory
from .long_term_memory import LongTermMemory

# VectorStore is optional - requires lancedb
try:
    from .vector_store import VectorStore
except ImportError:
    VectorStore = None

__all__ = [
    # Types
    'MemoryPriority',
    'AgentState',
    'TaskType',
    'MemoryEntry',
    'ProjectContext',
    'TaskInfo',
    'AgentStatus',
    'StateSnapshot',
    # Memory Systems
    'ShortTermMemory',
    'LongTermMemory',
    'VectorStore',
]
