# ============================================================================
# FILE: context_manager.py
# Centralized manager for all memory layers (STM + LTM + Vector Memory)
# ============================================================================

from typing import Optional, Dict, List
from datetime import datetime

from ..memory.short_term_memory import ShortTermMemory
from ..memory.long_term_memory import LongTermMemory
from ..memory.memory_types import MemoryEntry, MemoryPriority

# VectorStore is optional - requires lancedb
try:
    from ..memory.vector_store import VectorStore
except ImportError:
    VectorStore = None



class ContextManager:
    """
    Orchestrates:
    - Short-term memory (working memory)
    - Long-term memory (project knowledge + tasks + bugs + patterns)
    - Vector memory (semantic retrieval)
    - File + architecture context
    """

    def __init__(
        self,
        project_name: str,
        vector_store: Optional[VectorStore] = None,
        stm_size: int = 60,
        stm_window: int = 12
    ):
        # Memory systems
        self.stm = ShortTermMemory(max_size=stm_size, context_window=stm_window)
        self.ltm = LongTermMemory(project_name=project_name)
        self.vector_store = vector_store

        # Context metadata
        self.project_name = project_name
        self.current_task: Optional[str] = None
        self.active_agent: Optional[str] = None

    # ----------------------------------------------------------------------
    # Recording and Processing Inputs
    # ----------------------------------------------------------------------

    def record_agent_output(
        self,
        agent_id: str,
        output: str,
        task_description: str,
        priority: MemoryPriority = MemoryPriority.HIGH
    ):
        """Record output into STM + append to vector memory"""
        # Store in STM
        self.stm.add_agent_output(agent_id, output, task_description)

        # Store into vector memory for semantic retrieval
        if self.vector_store:
            self.vector_store.add_entry(
                MemoryEntry(
                    content=output,
                    timestamp=datetime.now(),
                    agent_id=agent_id,
                    tags=["agent_output", "task_result"],
                    priority=priority,
                    related_files=[]
                )
            )

    def record_file_context(
        self,
        agent_id: str,
        filename: str,
        content: str,
        purpose: str = "analysis"
    ):
        """Store file into STM + forward to LTM project structure"""
        self.stm.add_code_context(agent_id, filename, content, purpose)
        # Also update LTM project structure
        self.ltm.update_file(filename, content)

    def record_error(self, agent_id: str, error: str, context: str):
        """Store agent errors"""
        self.stm.add_error(agent_id, error, context)

    def record_decision(self, agent_id: str, decision: str, context: str):
        """Store an agent decision"""
        self.stm.add_decision(agent_id, decision, context)

    # ----------------------------------------------------------------------
    # Semantic Retrieval Layer
    # ----------------------------------------------------------------------

    def retrieve_relevant_memory(
        self,
        query: str,
        vector_k: int = 5,
        stm_k: int = 5
    ) -> List[str]:
        """
        Retrieve top relevant memories from:
        - STM (recent context)
        - Vector store (semantic)
        """
        results = []

        # === Semantic Memory Search ===
        if self.vector_store:
            vector_hits = self.vector_store.semantic_search(query=query, k=vector_k)
            for entry, score in vector_hits:
                results.append(
                    f"[SEMANTIC MEMORY | {entry.agent_id}] score={round(score, 3)}\n{entry.content[:500]}"
                )

        # === Short-term Relevant Context ===
        stm_recent = self.stm.get_recent(stm_k)
        for m in stm_recent:
            if query.lower() in m.content.lower():
                results.append(
                    f"[RECENT CONTEXT | {m.agent_id}]\n{m.content[:500]}"
                )

        return results

    # ----------------------------------------------------------------------
    # Context Building for Agents
    # ----------------------------------------------------------------------

    def build_context(
        self,
        agent_id: Optional[str],
        query: Optional[str] = None,
        include_architecture: bool = True,
        include_files: bool = True
    ) -> str:
        """
        Construct a full context bundle for an agent.
        Includes architecture, recent memory, semantic memory, and active files.
        """
        ctx = f"=== CONTEXT PACKAGE for {agent_id or 'AGENT'} ===\n\n"

        # -----------------------------
        # Architecture + conventions
        # -----------------------------
        if include_architecture and self.ltm.project_context:
            ctx += "### PROJECT ARCHITECTURE ###\n"
            ctx += self.ltm.get_architecture() + "\n\n"

            ctx += "### TECH STACK ###\n"
            ctx += ", ".join(self.ltm.get_tech_stack()) + "\n\n"

        # -----------------------------
        # Relevant semantic memory
        # -----------------------------
        if query:
            ctx += f"### RELEVANT MEMORY (query='{query}') ###\n"
            memories = self.retrieve_relevant_memory(query)
            if memories:
                for m in memories:
                    ctx += m + "\n\n"
            else:
                ctx += "No relevant memories.\n\n"

        # -----------------------------
        # Short-term recent output
        # -----------------------------
        ctx += "### RECENT AGENT ACTIVITY ###\n"
        recent = self.stm.summarize_context(agent_id)
        ctx += recent + "\n\n"

        # -----------------------------
        # Active working files
        # -----------------------------
        if include_files:
            active_files = self.stm.get_active_files()
            if active_files:
                ctx += "### ACTIVE FILE CONTEXT ###\n"
                for fname, text in active_files.items():
                    truncated = text[:800] + "..." if len(text) > 800 else text
                    ctx += f"\n--- {fname} ---\n{truncated}\n"
            else:
                ctx += "No active file context.\n"

        return ctx

    # ----------------------------------------------------------------------
    # Persistence
    # ----------------------------------------------------------------------

    def save(self):
        """Save long-term memory to disk."""
        self.ltm.save_to_disk()

    def load(self):
        """Load long-term project knowledge."""
        return self.ltm.load_from_disk()

    # ----------------------------------------------------------------------
    # Clearing
    # ----------------------------------------------------------------------

    def clear_all(self):
        """Hard reset all memory."""
        self.stm.clear()
        self.ltm.clear()

    # ----------------------------------------------------------------------
    # Stats
    # ----------------------------------------------------------------------

    def get_stats(self) -> Dict:
        """Return combined memory statistics."""
        return {
            "stm_entries": len(self.stm),
            "ltm_memories": len(self.ltm.memories),
            "ltm_tasks": len(self.ltm.task_history),
            "bugs_recorded": len(self.ltm.bug_database),
            "patterns": len(self.ltm.patterns)
        }

