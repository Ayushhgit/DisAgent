# ============================================================================
# FILE: memory_types.py
# Shared data structures and types for the memory system
# ============================================================================

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import List, Optional, Any, Dict


class MemoryPriority(Enum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


class AgentState(Enum):
    IDLE = "idle"
    PLANNING = "planning"
    EXECUTING = "executing"
    REVIEWING = "reviewing"
    BLOCKED = "blocked"
    COMPLETED = "completed"
    FAILED = "failed"


class TaskType(Enum):
    ARCHITECTURE = "architecture"
    IMPLEMENTATION = "implementation"
    BUG_FIX = "bug_fix"
    TESTING = "testing"
    REFACTOR = "refactor"
    DOCUMENTATION = "documentation"
    RESEARCH = "research"
    CODE_REVIEW = "code_review"


@dataclass
class MemoryEntry:
    """Single memory entry with metadata"""
    content: str
    timestamp: datetime
    priority: MemoryPriority
    agent_id: str
    tags: List[str] = field(default_factory=list)
    embedding: Optional[List[float]] = None
    access_count: int = 0
    related_files: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "priority": self.priority.name,
            "agent_id": self.agent_id,
            "tags": self.tags,
            "access_count": self.access_count,
            "related_files": self.related_files
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "MemoryEntry":
        return cls(
            content=data["content"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            priority=MemoryPriority[data["priority"]],
            agent_id=data["agent_id"],
            tags=data.get("tags", []),
            embedding=data.get("embedding"),
            access_count=data.get("access_count", 0),
            related_files=data.get("related_files", [])
        )


@dataclass
class ProjectContext:
    """Project-level information"""
    project_name: str
    description: str
    architecture: str
    tech_stack: List[str]
    file_structure: Dict[str, str]  # filename -> content
    dependencies: Dict[str, str]  # package -> version
    conventions: Dict[str, str]  # coding standards, naming conventions
    created_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict:
        return {
            "project_name": self.project_name,
            "description": self.description,
            "architecture": self.architecture,
            "tech_stack": self.tech_stack,
            "file_structure": self.file_structure,
            "dependencies": self.dependencies,
            "conventions": self.conventions,
            "created_at": self.created_at.isoformat()
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "ProjectContext":
        return cls(
            project_name=data["project_name"],
            description=data["description"],
            architecture=data["architecture"],
            tech_stack=data["tech_stack"],
            file_structure=data.get("file_structure", {}),
            dependencies=data.get("dependencies", {}),
            conventions=data.get("conventions", {}),
            created_at=datetime.fromisoformat(data["created_at"]) if "created_at" in data else datetime.now()
        )


@dataclass
class TaskInfo:
    """Information about a task"""
    task_id: str
    task_type: TaskType
    description: str
    assigned_agent: str
    parent_task: Optional[str] = None
    dependencies: List[str] = field(default_factory=list)
    affected_files: List[str] = field(default_factory=list)
    status: AgentState = AgentState.IDLE
    result: Optional[str] = None
    error: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None

    def to_dict(self) -> Dict:
        return {
            "task_id": self.task_id,
            "task_type": self.task_type.value,
            "description": self.description,
            "assigned_agent": self.assigned_agent,
            "parent_task": self.parent_task,
            "dependencies": self.dependencies,
            "affected_files": self.affected_files,
            "status": self.status.value,
            "result": self.result,
            "error": self.error,
            "created_at": self.created_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "TaskInfo":
        return cls(
            task_id=data["task_id"],
            task_type=TaskType(data["task_type"]),
            description=data["description"],
            assigned_agent=data["assigned_agent"],
            parent_task=data.get("parent_task"),
            dependencies=data.get("dependencies", []),
            affected_files=data.get("affected_files", []),
            status=AgentState(data.get("status", "idle")),
            result=data.get("result"),
            error=data.get("error"),
            created_at=datetime.fromisoformat(data["created_at"]) if "created_at" in data else datetime.now(),
            completed_at=datetime.fromisoformat(data["completed_at"]) if data.get("completed_at") else None
        )


@dataclass
class AgentStatus:
    """Current status of an agent"""
    agent_id: str
    state: AgentState
    current_task: Optional[str] = None
    assigned_tasks: List[str] = field(default_factory=list)
    completed_tasks: List[str] = field(default_factory=list)
    blockers: List[str] = field(default_factory=list)
    last_activity: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "agent_id": self.agent_id,
            "state": self.state.value,
            "current_task": self.current_task,
            "assigned_tasks": self.assigned_tasks,
            "completed_tasks": self.completed_tasks,
            "blockers": self.blockers,
            "last_activity": self.last_activity.isoformat(),
            "metadata": self.metadata
        }


@dataclass
class StateSnapshot:
    """System state at a point in time"""
    timestamp: datetime
    active_agents: List[str]
    task_queue_size: int
    completed_tasks: int
    blocked_agents: List[str]
    system_state: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "timestamp": self.timestamp.isoformat(),
            "active_agents": self.active_agents,
            "task_queue_size": self.task_queue_size,
            "completed_tasks": self.completed_tasks,
            "blocked_agents": self.blocked_agents,
            "system_state": self.system_state,
            "metadata": self.metadata
        }
