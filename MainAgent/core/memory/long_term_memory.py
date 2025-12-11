# ============================================================================
# FILE: long_term_memory.py
# Persistent storage for project knowledge and learnings
# ============================================================================

from typing import Dict, List, Optional, Any
import json
from datetime import datetime
from pathlib import Path

from .memory_types import MemoryEntry, ProjectContext, TaskInfo, MemoryPriority


class LongTermMemory:
    """
    Persistent storage for:
    - Project architecture and design decisions
    - Code patterns and conventions
    - Bug fixes and solutions
    - Agent learnings and best practices
    """

    def __init__(self, project_name: str, storage_path: Optional[str] = None):
        self.project_name = project_name
        self.storage_path = Path(storage_path) if storage_path else Path("./.agent_memory")
        self.project_context: Optional[ProjectContext] = None
        self.memories: Dict[str, MemoryEntry] = {}
        self.task_history: Dict[str, TaskInfo] = {}
        self.bug_database: Dict[str, Dict[str, Any]] = {}  # bug_id -> bug info
        self.patterns: Dict[str, str] = {}  # pattern_name -> implementation JSON
        self.learnings: List[str] = []

    def initialize_project(self, context: ProjectContext):
        """Initialize or update project context"""
        self.project_context = context

    def store_memory(self, key: str, entry: MemoryEntry):
        """Store important memory"""
        self.memories[key] = entry
        entry.access_count += 1

    def retrieve_memory(self, key: str) -> Optional[MemoryEntry]:
        """Retrieve a stored memory"""
        entry = self.memories.get(key)
        if entry:
            entry.access_count += 1
        return entry

    def store_task(self, task: TaskInfo):
        """Record task completion"""
        self.task_history[task.task_id] = task

    def store_bug_fix(self, bug_id: str, description: str, solution: str,
                      affected_files: List[str], root_cause: str, fixed_by: Optional[str] = None):
        """Record bug and its solution"""
        self.bug_database[bug_id] = {
            "description": description,
            "solution": solution,
            "affected_files": affected_files,
            "root_cause": root_cause,
            "timestamp": datetime.now().isoformat(),
            "fixed_by": fixed_by
        }

    def store_pattern(self, pattern_name: str, implementation: str,
                      use_case: str, files: List[str]):
        """Store reusable code pattern"""
        self.patterns[pattern_name] = json.dumps({
            "implementation": implementation,
            "use_case": use_case,
            "files": files,
            "created_at": datetime.now().isoformat()
        })

    def add_learning(self, learning: str, agent_id: Optional[str] = None):
        """Record agent learning/insight"""
        timestamp = datetime.now().isoformat()
        agent_info = f" by {agent_id}" if agent_id else ""
        self.learnings.append(f"[{timestamp}]{agent_info} {learning}")

    def get_architecture(self) -> str:
        """Get system architecture"""
        return self.project_context.architecture if self.project_context else ""

    def get_file_structure(self) -> Dict[str, str]:
        """Get all project files"""
        return self.project_context.file_structure if self.project_context else {}

    def update_file(self, filename: str, content: str):
        """Update file in project"""
        if self.project_context:
            self.project_context.file_structure[filename] = content

    def get_tech_stack(self) -> List[str]:
        """Get project tech stack"""
        return self.project_context.tech_stack if self.project_context else []

    def get_conventions(self) -> Dict[str, str]:
        """Get project coding conventions"""
        return self.project_context.conventions if self.project_context else {}

    def search_similar_bugs(self, description: str, limit: int = 5) -> List[Dict]:
        """Find similar bugs that were fixed"""
        keywords = set(description.lower().split())
        matches = []

        for bug_id, bug in self.bug_database.items():
            bug_keywords = set(bug["description"].lower().split())
            overlap = len(keywords & bug_keywords)
            if overlap > 0:
                matches.append({
                    "bug_id": bug_id,
                    "relevance": overlap,
                    **bug
                })

        return sorted(matches, key=lambda x: x["relevance"], reverse=True)[:limit]

    def get_relevant_patterns(self, task_type: str, limit: int = 5) -> List[Dict]:
        """Get code patterns relevant to task"""
        relevant = []
        for name, pattern_json in self.patterns.items():
            pattern = json.loads(pattern_json)
            if task_type.lower() in pattern["use_case"].lower():
                relevant.append({"name": name, **pattern})
        return relevant[:limit]

    def get_task_history(self, agent_id: Optional[str] = None,
                         task_type: Optional[str] = None,
                         limit: int = 20) -> List[TaskInfo]:
        """Get historical tasks"""
        tasks = list(self.task_history.values())

        if agent_id:
            tasks = [t for t in tasks if t.assigned_agent == agent_id]
        if task_type:
            tasks = [t for t in tasks if t.task_type.value == task_type]

        return sorted(tasks, key=lambda x: x.created_at, reverse=True)[:limit]

    def get_learnings(self, limit: int = 20) -> List[str]:
        """Get recent learnings"""
        return self.learnings[-limit:]

    def search_memories(self, query: str, limit: int = 10) -> List[MemoryEntry]:
        """Search memories by content"""
        query_lower = query.lower()
        results = []

        for key, entry in self.memories.items():
            if query_lower in entry.content.lower():
                results.append(entry)

        return sorted(results, key=lambda x: x.access_count, reverse=True)[:limit]

    def get_memories_by_tags(self, tags: List[str]) -> List[MemoryEntry]:
        """Get memories matching tags"""
        return [
            entry for entry in self.memories.values()
            if any(tag in entry.tags for tag in tags)
        ]

    def export_knowledge_base(self) -> Dict[str, Any]:
        """Export all knowledge for persistence"""
        return {
            "project_name": self.project_name,
            "project_context": self.project_context.to_dict() if self.project_context else None,
            "memories": {k: v.to_dict() for k, v in self.memories.items()},
            "task_history": {k: v.to_dict() for k, v in self.task_history.items()},
            "bug_database": self.bug_database,
            "patterns": self.patterns,
            "learnings": self.learnings,
            "exported_at": datetime.now().isoformat()
        }

    def import_knowledge_base(self, data: Dict[str, Any]):
        """Import knowledge from exported data"""
        self.project_name = data.get("project_name", self.project_name)

        if data.get("project_context"):
            self.project_context = ProjectContext.from_dict(data["project_context"])

        for key, mem_data in data.get("memories", {}).items():
            self.memories[key] = MemoryEntry.from_dict(mem_data)

        for key, task_data in data.get("task_history", {}).items():
            self.task_history[key] = TaskInfo.from_dict(task_data)

        self.bug_database = data.get("bug_database", {})
        self.patterns = data.get("patterns", {})
        self.learnings = data.get("learnings", [])

    def save_to_disk(self):
        """Persist memory to disk"""
        self.storage_path.mkdir(parents=True, exist_ok=True)
        filepath = self.storage_path / f"{self.project_name}_memory.json"

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.export_knowledge_base(), f, indent=2, default=str)

    def load_from_disk(self) -> bool:
        """Load memory from disk"""
        filepath = self.storage_path / f"{self.project_name}_memory.json"

        if not filepath.exists():
            return False

        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.import_knowledge_base(data)
            return True
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Warning: Could not load memory: {e}")
            return False

    def get_summary(self) -> str:
        """Get summary of long-term memory contents"""
        summary = f"=== LONG-TERM MEMORY: {self.project_name} ===\n\n"

        if self.project_context:
            summary += f"Project: {self.project_context.project_name}\n"
            summary += f"Tech Stack: {', '.join(self.project_context.tech_stack)}\n"
            summary += f"Files: {len(self.project_context.file_structure)}\n\n"

        summary += f"Stored Memories: {len(self.memories)}\n"
        summary += f"Task History: {len(self.task_history)}\n"
        summary += f"Bug Database: {len(self.bug_database)}\n"
        summary += f"Code Patterns: {len(self.patterns)}\n"
        summary += f"Learnings: {len(self.learnings)}\n"

        return summary

    def clear(self):
        """Clear all long-term memory (use with caution)"""
        self.project_context = None
        self.memories.clear()
        self.task_history.clear()
        self.bug_database.clear()
        self.patterns.clear()
        self.learnings.clear()
