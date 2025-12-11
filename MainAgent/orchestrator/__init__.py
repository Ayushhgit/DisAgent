"""High-level orchestrators and shared utilities for the MainAgent project.

This package exposes a stable API for running orchestrators. Historically there
were separate `dynamic_orchestrator` and `planning_orchestrator` modules. The
codebase now uses a single `unified_orchestrator`. To preserve compatibility
for imports like `from orchestrator import dynamic_orchestrator`, we alias the
`unified_orchestrator` module as both `dynamic_orchestrator` and
`planning_orchestrator` here.

Avoid importing modules that don't exist here to prevent circular import
errors when the package is executed as a script (e.g., `python -m main`).
"""

from . import unified_orchestrator as dynamic_orchestrator

# Expose planning_orchestrator as an alias for compatibility
planning_orchestrator = dynamic_orchestrator

__all__ = ["dynamic_orchestrator", "planning_orchestrator", "unified_orchestrator"]


