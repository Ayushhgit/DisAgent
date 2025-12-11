"""
Code Agent - LLM-powered agent for code generation and modification.

This agent:
- Generates code based on prompts
- Modifies existing code files
- Follows the ===EDIT=== format for precise changes
- Integrates with the shared memory system
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
import logging

from .base_agent import BaseAgent, AgentResult
from core.runtime.llm import llm_call
from orchestrator.extractors import extract_and_write_files, extract_and_apply_edits

logger = logging.getLogger(__name__)


class CodeAgent(BaseAgent):
    """LLM-powered agent for code generation and modification.

    This agent uses an LLM to:
    - Generate new code files
    - Modify existing files using the ===EDIT=== format
    - Review and understand code context
    """

    def __init__(
        self,
        name: str,
        description: str = "Code generation and modification agent",
        max_tokens: int = 8192,
        temperature: float = 0.7,
    ):
        """Initialize the code agent.

        Args:
            name: Agent name
            description: Agent description
            max_tokens: Maximum tokens for LLM response
            temperature: LLM sampling temperature
        """
        super().__init__(name, description)
        self.max_tokens = max_tokens
        self.temperature = temperature

    def run(
        self,
        task_id: str,
        prompt: str,
        context: Any,
        file_manager: Any,
    ) -> AgentResult:
        """Execute the code agent.

        Args:
            task_id: Unique task identifier
            prompt: The coding task prompt
            context: Shared context with scope, reasoning, memory
            file_manager: FileManager for file operations

        Returns:
            AgentResult with generated code and file changes
        """
        self._execution_count += 1

        try:
            # Build the full prompt with context
            full_prompt = self._build_prompt(prompt, context, file_manager)

            # Call the LLM
            logger.info(f"[{self.name}] Executing task {task_id}")
            output = llm_call(
                full_prompt,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
            )

            if not output or len(output.strip()) < 50:
                logger.warning(f"[{self.name}] Produced minimal output")
                return AgentResult(
                    success=False,
                    output=output or "",
                    metadata={"task_id": task_id},
                    errors=["Agent produced minimal or no output"],
                )

            # Process output - extract and write files
            files_created = {}
            edits_applied = {}

            if file_manager:
                try:
                    files_created = extract_and_write_files(output, file_manager, self.name)
                    edits_applied = extract_and_apply_edits(output, file_manager, self.name)
                except Exception as exc:
                    logger.exception(f"[{self.name}] File processing error: {exc}")

            # Determine success
            success = True
            errors = []

            if edits_applied:
                failed_edits = [f for f, s in edits_applied.items() if not s]
                if failed_edits:
                    errors.append(f"Failed edits: {', '.join(failed_edits)}")

            return AgentResult(
                success=success,
                output=output,
                metadata={
                    "task_id": task_id,
                    "files_created_count": len(files_created),
                    "edits_applied_count": sum(1 for s in edits_applied.values() if s),
                },
                files_created=list(files_created.keys()),
                files_modified=[f for f, s in edits_applied.items() if s],
                errors=errors if errors else None,
            )

        except Exception as exc:
            logger.exception(f"[{self.name}] Task {task_id} failed: {exc}")
            return AgentResult(
                success=False,
                output="",
                metadata={"task_id": task_id},
                errors=[str(exc)],
            )

    def _build_prompt(
        self,
        base_prompt: str,
        context: Any,
        file_manager: Any,
    ) -> str:
        """Build the full prompt with context.

        Args:
            base_prompt: The base task prompt
            context: Shared context object
            file_manager: FileManager for project info

        Returns:
            Complete prompt string
        """
        parts = [base_prompt]

        # Add project structure if available
        if file_manager:
            try:
                structure = file_manager.get_project_structure_tree()
                if structure:
                    parts.append(f"\n=== PROJECT STRUCTURE ===\n{structure}")
            except Exception:
                pass

        # Add context if available
        if context and hasattr(context, 'get_context'):
            try:
                ctx_str = context.get_context()
                if ctx_str:
                    parts.append(f"\n{ctx_str}")
            except Exception:
                pass

        # Add instructions for file modifications
        parts.append("""
=== FILE MODIFICATION INSTRUCTIONS ===
To create new files, use code blocks with the filename:
```filename: path/to/file.py
# file content here
```

To edit existing files, use the ===EDIT=== format:
===EDIT===
file: path/to/file.py
old:
[exact code to replace]
new:
[new code]
===END===

IMPORTANT:
- Always review existing files before making changes
- Match whitespace and indentation exactly in the 'old' section
- Provide complete, working code
""")

        return "\n".join(parts)

    def act(self, code: str) -> str:
        """Simple action - analyze or process code.

        Args:
            code: Code string to process

        Returns:
            Analysis or processed result
        """
        if not code:
            return f"CodeAgent {self.name}: No code provided"

        return f"CodeAgent {self.name} received code of length {len(code)}"


class ReviewAgent(CodeAgent):
    """Agent specialized for code review tasks."""

    def __init__(self, name: str = "reviewer"):
        super().__init__(
            name=name,
            description="Code review and quality analysis agent",
            max_tokens=4096,
            temperature=0.5,
        )

    def _build_prompt(
        self,
        base_prompt: str,
        context: Any,
        file_manager: Any,
    ) -> str:
        """Build review-specific prompt."""
        review_instructions = """
You are a Code Review Agent. Your task is to:
1. Analyze the code for bugs, security issues, and best practices
2. Suggest improvements with specific code examples
3. Rate the code quality (1-10)
4. Identify any potential performance issues

For suggestions, use the ===EDIT=== format to show exact changes.
"""
        return review_instructions + "\n" + super()._build_prompt(base_prompt, context, file_manager)


class TestAgent(CodeAgent):
    """Agent specialized for test generation."""

    def __init__(self, name: str = "tester"):
        super().__init__(
            name=name,
            description="Test generation agent",
            max_tokens=6144,
            temperature=0.6,
        )

    def _build_prompt(
        self,
        base_prompt: str,
        context: Any,
        file_manager: Any,
    ) -> str:
        """Build test-specific prompt."""
        test_instructions = """
You are a Test Generation Agent. Your task is to:
1. Generate comprehensive unit tests for the code
2. Include edge cases and error conditions
3. Use appropriate testing frameworks (pytest, jest, etc.)
4. Add clear test descriptions and assertions

Create test files using the filename format: test_<module>.py
"""
        return test_instructions + "\n" + super()._build_prompt(base_prompt, context, file_manager)
