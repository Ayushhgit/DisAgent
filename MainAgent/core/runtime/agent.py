from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any


@dataclass
class AgentResult:
    success: bool
    output: str
    metadata: Dict[str, Any]


class Agent:
    """Abstract Agent interface.

    Subclasses should implement `run(task_id, prompt, context, file_manager)` and
    return an `AgentResult`.
    """

    name: str

    def __init__(self, name: str):
        self.name = name

    def run(self, task_id: str, prompt: str, context, file_manager) -> AgentResult:
        raise NotImplementedError()

