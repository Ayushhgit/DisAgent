"""Shared context object for coordinating multi-agent workflows."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class AgentContext:
    """Holds shared knowledge between agents run by an orchestrator."""

    memory: Dict[str, str] = field(default_factory=dict)
    decisions: Dict[str, str] = field(default_factory=dict)
    scope_info: Dict[str, str] = field(default_factory=dict)
    architecture: str = ""
    file_structure: Dict[str, str] = field(default_factory=dict)

    def add_result(self, agent_name: str, result: str) -> None:
        self.memory[agent_name] = result

    def get_context(self) -> str:
        context_parts: List[str] = ["=== SHARED AGENT CONTEXT ===\n"]

        if self.scope_info:
            context_parts.append("SCOPE INFORMATION:\n")
            for key, value in self.scope_info.items():
                context_parts.append(f"  {key}: {value}\n")
            context_parts.append("\n")

        if self.architecture:
            context_parts.append("SYSTEM ARCHITECTURE:\n")
            context_parts.append(f"{self.architecture}\n\n")

        for agent, result in self.memory.items():
            truncated = result[:2000] if len(result) > 2000 else result
            context_parts.append(f"[{agent}]:\n{truncated}\n\n---\n\n")

        return "".join(context_parts)

    def set_scope(self, scope_dict: Dict[str, str]) -> None:
        self.scope_info = scope_dict

    def set_architecture(self, architecture: str) -> None:
        self.architecture = architecture

    def add_file(self, filename: str, content: str) -> None:
        self.file_structure[filename] = content


