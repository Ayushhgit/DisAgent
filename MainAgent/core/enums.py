# ============================================================================
# FILE: enums.py
# Centralized enums to replace magic strings throughout the system
# ============================================================================

from enum import Enum, auto
from typing import List


class UserIntent(str, Enum):
    """Types of user intents detected from prompts."""
    QUESTION = "question"          # User asking a question
    GENERATION = "generation"      # User wants to generate new code/project
    MODIFICATION = "modification"  # User wants to modify existing code
    HELP = "help"                  # User asking for help/documentation
    DEBUG = "debug"                # User wants to debug an issue
    REVIEW = "review"              # User wants code review
    EXPLAIN = "explain"            # User wants code explanation
    REFACTOR = "refactor"          # User wants to refactor code
    TEST = "test"                  # User wants to run/write tests
    UNKNOWN = "unknown"            # Could not determine intent

    @classmethod
    def from_string(cls, value: str) -> "UserIntent":
        """Convert string to UserIntent, defaulting to UNKNOWN."""
        try:
            return cls(value.lower())
        except ValueError:
            return cls.UNKNOWN


class AgentType(str, Enum):
    """Types of specialized agents available."""
    GENERALIST = "generalist"
    BACKEND = "backend"
    FRONTEND = "frontend"
    DATABASE = "db_engineer"
    DEVOPS = "devops"
    TESTER = "tester"
    REVIEWER = "reviewer"
    ARCHITECT = "architect"
    SECURITY = "security"
    DOCUMENTATION = "documentation"
    API = "api"
    UI = "ui"
    MOBILE = "mobile"

    @classmethod
    def from_string(cls, value: str) -> "AgentType":
        """Convert string to AgentType, defaulting to GENERALIST."""
        value_lower = value.lower().replace("-", "_").replace(" ", "_")
        for member in cls:
            if member.value == value_lower:
                return member
        return cls.GENERALIST

    @classmethod
    def all_types(cls) -> List[str]:
        """Get all agent type values."""
        return [member.value for member in cls]


class AgentState(str, Enum):
    """States an agent can be in during execution."""
    IDLE = "idle"
    PLANNING = "planning"
    EXECUTING = "executing"
    REVIEWING = "reviewing"
    WAITING = "waiting"
    BLOCKED = "blocked"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

    def is_terminal(self) -> bool:
        """Check if this is a terminal state."""
        return self in (AgentState.COMPLETED, AgentState.FAILED, AgentState.CANCELLED)

    def is_active(self) -> bool:
        """Check if this is an active (working) state."""
        return self in (AgentState.PLANNING, AgentState.EXECUTING, AgentState.REVIEWING)


class TaskType(str, Enum):
    """Types of tasks that can be performed."""
    ARCHITECTURE = "architecture"
    IMPLEMENTATION = "implementation"
    BUG_FIX = "bug_fix"
    TESTING = "testing"
    REFACTOR = "refactor"
    DOCUMENTATION = "documentation"
    RESEARCH = "research"
    CODE_REVIEW = "code_review"
    DEPLOYMENT = "deployment"
    CONFIGURATION = "configuration"
    OPTIMIZATION = "optimization"

    @classmethod
    def from_string(cls, value: str) -> "TaskType":
        """Convert string to TaskType, defaulting to IMPLEMENTATION."""
        value_lower = value.lower().replace("-", "_").replace(" ", "_")
        for member in cls:
            if member.value == value_lower:
                return member
        return cls.IMPLEMENTATION


class TaskStatus(str, Enum):
    """Status of a task in the queue."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    BLOCKED = "blocked"
    CANCELLED = "cancelled"
    SKIPPED = "skipped"

    def is_terminal(self) -> bool:
        """Check if this is a terminal status."""
        return self in (TaskStatus.COMPLETED, TaskStatus.FAILED,
                        TaskStatus.CANCELLED, TaskStatus.SKIPPED)


class Complexity(str, Enum):
    """Complexity levels for scope analysis."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

    @classmethod
    def from_string(cls, value: str) -> "Complexity":
        """Convert string to Complexity, defaulting to MEDIUM."""
        try:
            return cls(value.lower())
        except ValueError:
            return cls.MEDIUM

    def to_int(self) -> int:
        """Convert to integer for comparison."""
        mapping = {cls.LOW: 1, cls.MEDIUM: 2, cls.HIGH: 3, cls.CRITICAL: 4}
        return mapping.get(self, 2)


class ScopeLevel(str, Enum):
    """Scope levels for task analysis."""
    TICKET = "ticket"      # Small, focused change
    FEATURE = "feature"    # Medium-sized feature
    PROJECT = "project"    # Large, project-wide change
    SYSTEM = "system"      # System-level change

    @classmethod
    def from_string(cls, value: str) -> "ScopeLevel":
        """Convert string to ScopeLevel, defaulting to FEATURE."""
        try:
            return cls(value.lower())
        except ValueError:
            return cls.FEATURE


class RiskLevel(str, Enum):
    """Risk levels for change assessment."""
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

    @classmethod
    def from_string(cls, value: str) -> "RiskLevel":
        """Convert string to RiskLevel, defaulting to LOW."""
        try:
            return cls(value.lower())
        except ValueError:
            return cls.LOW

    def requires_review(self) -> bool:
        """Check if this risk level requires manual review."""
        return self in (RiskLevel.HIGH, RiskLevel.CRITICAL)


class FileOperation(str, Enum):
    """Types of file operations."""
    CREATE = "create"
    MODIFY = "modify"
    DELETE = "delete"
    READ = "read"
    RENAME = "rename"
    MOVE = "move"


class EditStage(str, Enum):
    """Stages of the edit matching process."""
    EXACT = "exact"
    NORMALIZED = "normalized"
    NORMALIZED_EXACT = "normalized_exact"
    DEDENTED = "dedented"
    STRIPPED = "stripped"
    FUZZY = "fuzzy"
    FALLBACK = "fallback"
    FAILED = "failed"


class VerificationResult(str, Enum):
    """Results of verification loop."""
    SUCCESS = "success"
    TEST_FAILURE = "test_failure"
    PATCH_FAILURE = "patch_failure"
    TIMEOUT = "timeout"
    ERROR = "error"
    SKIPPED = "skipped"


class ApprovalStatus(str, Enum):
    """Approval status from critic agent."""
    APPROVED = "approved"
    CONDITIONAL = "conditional"
    NEEDS_REVISION = "needs_revision"
    REJECTED = "rejected"

    def is_acceptable(self) -> bool:
        """Check if this status allows proceeding."""
        return self in (ApprovalStatus.APPROVED, ApprovalStatus.CONDITIONAL)


class Severity(str, Enum):
    """Severity levels for issues."""
    SUGGESTION = "suggestion"
    MINOR = "minor"
    MAJOR = "major"
    CRITICAL = "critical"

    def to_int(self) -> int:
        """Convert to integer for comparison."""
        mapping = {cls.SUGGESTION: 1, cls.MINOR: 2, cls.MAJOR: 3, cls.CRITICAL: 4}
        return mapping.get(self, 2)


class MemoryPriority(str, Enum):
    """Priority levels for memory entries."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

    def to_int(self) -> int:
        """Convert to integer for comparison."""
        mapping = {cls.LOW: 1, cls.MEDIUM: 2, cls.HIGH: 3, cls.CRITICAL: 4}
        return mapping.get(self, 2)


class Domain(str, Enum):
    """Technical domains for task categorization."""
    BACKEND = "backend"
    FRONTEND = "frontend"
    DATABASE = "database"
    DEVOPS = "devops"
    TESTING = "testing"
    SECURITY = "security"
    DOCUMENTATION = "documentation"
    GENERAL = "general"
    API = "api"
    UI = "ui"
    MOBILE = "mobile"
    INFRASTRUCTURE = "infrastructure"

    @classmethod
    def from_string(cls, value: str) -> "Domain":
        """Convert string to Domain, defaulting to GENERAL."""
        value_lower = value.lower().replace("-", "_").replace(" ", "_")
        for member in cls:
            if member.value == value_lower:
                return member
        return cls.GENERAL


class OutputFormat(str, Enum):
    """Output formats for results."""
    TEXT = "text"
    JSON = "json"
    MARKDOWN = "markdown"
    HTML = "html"
    CODE = "code"


# === Helper functions ===

def get_agent_for_domain(domain: Domain) -> AgentType:
    """Get the appropriate agent type for a domain."""
    mapping = {
        Domain.BACKEND: AgentType.BACKEND,
        Domain.FRONTEND: AgentType.FRONTEND,
        Domain.DATABASE: AgentType.DATABASE,
        Domain.DEVOPS: AgentType.DEVOPS,
        Domain.TESTING: AgentType.TESTER,
        Domain.SECURITY: AgentType.SECURITY,
        Domain.DOCUMENTATION: AgentType.DOCUMENTATION,
        Domain.API: AgentType.API,
        Domain.UI: AgentType.UI,
        Domain.MOBILE: AgentType.MOBILE,
    }
    return mapping.get(domain, AgentType.GENERALIST)


def get_domain_for_agent(agent_type: AgentType) -> Domain:
    """Get the primary domain for an agent type."""
    mapping = {
        AgentType.BACKEND: Domain.BACKEND,
        AgentType.FRONTEND: Domain.FRONTEND,
        AgentType.DATABASE: Domain.DATABASE,
        AgentType.DEVOPS: Domain.DEVOPS,
        AgentType.TESTER: Domain.TESTING,
        AgentType.SECURITY: Domain.SECURITY,
        AgentType.DOCUMENTATION: Domain.DOCUMENTATION,
        AgentType.API: Domain.API,
        AgentType.UI: Domain.UI,
        AgentType.MOBILE: Domain.MOBILE,
    }
    return mapping.get(agent_type, Domain.GENERAL)
