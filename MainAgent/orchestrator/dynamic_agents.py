"""Utilities for spawning specialized agents that manipulate project files."""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional, Dict, Any

from .context import AgentContext
from .extractors import extract_and_apply_edits, extract_and_write_files
from .file_manager import FileManager
from core.runtime.llm import llm_call
from core.memory.memory_types import MemoryPriority
from core.runtime.code_validator import validate_file, format_validation_errors, ValidationResult

# Global statistics tracking (shared across agent executions)
_execution_stats: Dict[str, Any] = {
    "total_agent_calls": 0,
    "successful_calls": 0,
    "failed_calls": 0,
    "files_created": 0,
    "files_edited": 0,
    "edits_failed": 0,
    "total_tokens_estimated": 0,
    "agent_durations": {}
}

def get_execution_stats() -> Dict[str, Any]:
    """Get current execution statistics."""
    return dict(_execution_stats)

def reset_execution_stats() -> None:
    """Reset execution statistics."""
    global _execution_stats
    _execution_stats = {
        "total_agent_calls": 0,
        "successful_calls": 0,
        "failed_calls": 0,
        "files_created": 0,
        "files_edited": 0,
        "edits_failed": 0,
        "total_tokens_estimated": 0,
        "agent_durations": {}
    }

def _update_stats(
    agent_name: str,
    success: bool,
    files_created: int = 0,
    files_edited: int = 0,
    edits_failed: int = 0,
    duration: float = 0.0
) -> None:
    """Update execution statistics."""
    global _execution_stats
    _execution_stats["total_agent_calls"] += 1
    if success:
        _execution_stats["successful_calls"] += 1
    else:
        _execution_stats["failed_calls"] += 1
    _execution_stats["files_created"] += files_created
    _execution_stats["files_edited"] += files_edited
    _execution_stats["edits_failed"] += edits_failed
    _execution_stats["agent_durations"][agent_name] = duration


def _validate_generated_files(
    file_manager: FileManager,
    files_written: Dict[str, int],
    edits_applied: Dict[str, bool]
) -> tuple[bool, List[ValidationResult]]:
    """Validate all generated/edited files for syntax errors.

    Returns:
        Tuple of (all_valid, list of validation results with errors)
    """
    results = []
    all_valid = True

    # Validate newly created files
    for filepath in files_written.keys():
        content = file_manager.read_file(filepath)
        if content:
            result = validate_file(filepath, content)
            if not result.valid:
                results.append(result)
                all_valid = False

    # Validate edited files
    for filepath, success in edits_applied.items():
        if success and not filepath.endswith(('_append', '_insert')):
            content = file_manager.read_file(filepath)
            if content:
                result = validate_file(filepath, content)
                if not result.valid:
                    results.append(result)
                    all_valid = False

    return all_valid, results


def _build_correction_prompt(
    original_prompt: str,
    original_output: str,
    validation_errors: List[ValidationResult],
    file_manager: FileManager
) -> str:
    """Build a prompt asking the LLM to fix validation errors."""
    error_summary = format_validation_errors(validation_errors)

    # Get current content of files with errors
    file_contents = ""
    for result in validation_errors:
        content = file_manager.read_file(result.file_path)
        if content:
            file_contents += f"\n=== CURRENT CONTENT OF {result.file_path} ===\n"
            file_contents += content[:2000]  # Limit size
            if len(content) > 2000:
                file_contents += "\n... (truncated)"
            file_contents += "\n"

    return f"""Your previous code generation had SYNTAX ERRORS that need to be fixed.

{error_summary}

{file_contents}

Please provide CORRECTED versions of the files with errors.
Use the same format as before:
- For new files: ```filename: path/to/file.ext
- For edits: ===EDIT=== blocks

IMPORTANT:
1. Fix ALL the syntax errors listed above
2. Make sure brackets, braces, and parentheses are balanced
3. Ensure proper indentation (especially for Python)
4. Do not change working code - only fix the errors

Provide the corrected code now:"""


def create_dynamic_agent(
    agent_name: str,
    custom_prompt: str,
    user_request: str,
    context: AgentContext,
    file_manager: FileManager,
    allowed_files: Optional[List[str]] = None,
    critic_agent=None,  # Optional CriticAgent for output review
    verification_loop=None,  # Optional VerificationLoop for validation
    max_retries: int = 2,  # Maximum correction attempts
) -> str:
    """Execute a dynamic agent prompt and materialize produced files.

    This function:
    1. Reads from shared memory to get relevant context
    2. Builds a comprehensive prompt with project context
    3. Executes the agent via LLM
    4. Optionally reviews output with critic agent
    5. Writes outputs back to shared memory
    6. Applies file changes (creates/edits)
    7. Tracks execution statistics
    """
    import time
    start_time = time.time()

    # STEP 1: Read from shared memory to get relevant context
    memory_context = ""
    if context.unified_memory:
        # Get recent decisions and outputs from other agents
        recent_memories = context.unified_memory.short_term.get_recent(n=5)
        if recent_memories:
            memory_context = "\n=== RECENT AGENT ACTIVITY (from shared memory) ===\n"
            for mem in recent_memories:
                if mem.agent_id != agent_name:  # Don't include our own past outputs
                    memory_context += f"[{mem.agent_id}]: {mem.content[:300]}...\n\n"

        # Get any relevant learnings from long-term memory
        if context.unified_memory.long_term:
            learnings = context.unified_memory.long_term.get_learnings(limit=3)
            if learnings:
                memory_context += "\n=== LEARNINGS FROM MEMORY ===\n"
                for learning in learnings:
                    memory_context += f"- {learning}\n"

    # Get project structure and file summaries
    project_structure = file_manager.get_project_structure_tree()
    available_files = file_manager.list_files("*")
    if allowed_files is not None:
        # Filter available files to only those allowed (simple path match)
        allowed_set = set(allowed_files)
        available_files = [f for f in available_files if f in allowed_set]

    # Build comprehensive file context for agents
    files_context = "=" * 70 + "\n"
    files_context += "PROJECT STRUCTURE & FILE SUMMARIES\n"
    files_context += "=" * 70 + "\n\n"
    
    if available_files:
        files_context += project_structure
        files_context += f"\nTotal Files Found: {len(available_files)}\n"

        # Show directory breakdown
        dirs = set()
        for file_path in available_files:
            parent = str(Path(file_path).parent)
            if parent != ".":
                dirs.add(parent)

        if dirs:
            files_context += f"Subdirectories: {len(dirs)}\n"
            files_context += f"   {', '.join(sorted(dirs)[:10])}"
            if len(dirs) > 10:
                files_context += f" ... and {len(dirs) - 10} more"
            files_context += "\n"

        files_context += "\n"

        files_context += "=" * 70 + "\n"
        files_context += "FILE SUMMARIES (Quick Overview)\n"
        files_context += "=" * 70 + "\n\n"

        # Generate summaries for the selected files
        for file_name in sorted(available_files):
            summary = file_manager.get_file_summary(file_name)
            files_context += f"{summary}\n\n"

        files_context += "=" * 70 + "\n"
        files_context += "FULL FILE CONTENTS (For Editing)\n"
        files_context += "=" * 70 + "\n\n"

        # Include full file contents (with size limits for very large files)
        # Increased limit for directory-level operations
        # Limit the number of full files included to avoid huge prompts
        file_limit = min(20, len(available_files))  # Show up to 20 files by default
        for file_name in sorted(available_files)[:file_limit]:
            content = file_manager.read_file(file_name)
            if content:
                files_context += f"\n{'='*70}\n"
                files_context += f"FILE: {file_name}\n"
                files_context += f"{'='*70}\n"

                # For larger files, show first and last portions with summary
                if len(content) > 3000:
                    files_context += f"[FILE SIZE: {len(content)} chars, {len(content.splitlines())} lines]\n"
                    files_context += f"\n--- FIRST 1500 CHARACTERS ---\n"
                    files_context += content[:1500]
                    files_context += f"\n\n... [MIDDLE {len(content) - 2000} CHARACTERS HIDDEN] ...\n\n"
                    files_context += f"--- LAST 500 CHARACTERS ---\n"
                    files_context += content[-500:]
                elif len(content) > 2000:
                    preview = content[:1000] + "\n\n... [FILE TRUNCATED - MIDDLE SECTION HIDDEN] ...\n\n" + content[-500:]
                    files_context += preview
                else:
                    files_context += content
                files_context += f"\n{'='*70}\n\n"
            else:
                files_context += f"\n{'='*70}\n"
                files_context += f"FILE: {file_name}\n"
                files_context += f"[Could not read file or file is empty]\n"
                files_context += f"{'='*70}\n\n"
    else:
        files_context += "(Empty project - no files exist yet)\n"
        files_context += "You can create new files using the format below.\n\n"

    full_prompt = f"""{custom_prompt}

{files_context}

=== SHARED CONTEXT FROM PREVIOUS AGENTS ===
{context.get_context()}
{memory_context}

=== ORIGINAL USER REQUEST ===
{user_request}

=== YOUR CAPABILITIES ===
You have FULL access to read and edit existing files in this project!

BEFORE MAKING CHANGES:
1. Review the PROJECT STRUCTURE above to understand the project layout
2. Check FILE SUMMARIES to understand what each file does
3. Read the FULL FILE CONTENTS to see the exact code you need to modify
4. Understand dependencies and relationships between files

WHEN UPDATING/EDITING EXISTING FILES:
- Always use the ===EDIT=== format for precise changes (recommended)
- Match the "old:" section EXACTLY (including whitespace, indentation, quotes)
- Preserve the file's structure, imports, and style
- Only change what's necessary to fulfill the request

TO CREATE NEW FILES:
```filename: path/to/new_file.ext
[complete code here]
```

TO EDIT EXISTING FILES (RECOMMENDED FOR UPDATES):
===EDIT===
file: path/to/existing_file.ext
old:
[exact code to replace - must match EXACTLY including whitespace]
new:
[new code to replace it with]
===END===

TO APPEND TO FILES:
APPEND_TO: path/to/existing_file.ext
[code to append]
END_APPEND

TO REPLACE ENTIRE FILE:
```filename: path/to/existing_file.ext
[complete new file content - use only if major rewrite needed]
```

CRITICAL EDITING RULES:
1. When updating existing code, use ===EDIT=== format (most precise)
2. The "old:" code MUST match the file content EXACTLY (copy-paste from file contents above)
3. Preserve code style, formatting, and structure
4. Maintain all imports and dependencies
5. Test that your changes don't break existing functionality
6. Always provide COMPLETE, working code (no TODOs or placeholders)

FILE ACCESS:
- All files are shown above with full contents
- File summaries help you understand each file's purpose
- Project structure shows how files are organized
- You can edit ANY file shown in the structure
"""

    print(f"\n[{agent_name.upper()}] Executing...\n")
    try:
        output = llm_call(full_prompt, max_tokens=8192, temperature=0.7)
    except Exception as exc:
        print(f"   [ERROR] Error calling LLM: {exc}\n")
        output = ""
        # Record error to memory
        if context.unified_memory:
            context.unified_memory.short_term.add_error(
                agent_id=agent_name,
                error=str(exc),
                context="LLM call failed"
            )

    if not output or len(output) < 50:
        print(f"   [WARN] Warning: {agent_name} produced minimal output")

    # STEP 4: Write output to shared memory
    context.add_result(agent_name, output)

    # Also record a decision summary if we have unified memory
    if context.unified_memory and output:
        # Extract a brief summary of what the agent did
        summary = output[:500] if len(output) > 500 else output
        context.unified_memory.short_term.add_decision(
            agent_id=agent_name,
            decision=f"Agent {agent_name} completed execution",
            context=summary,
            priority=MemoryPriority.HIGH
        )

    # STEP 5: Apply file changes
    print(f"\n   [PROCESSING] Processing {agent_name} output - creating/editing files...\n")
    files_written = {}
    edits_applied = {}

    try:
        files_written = extract_and_write_files(output, file_manager, agent_name)
        edits_applied = extract_and_apply_edits(output, file_manager, agent_name)
    except Exception as exc:
        print(f"   [ERROR] Error processing agent output: {exc}\n")
        import traceback
        traceback.print_exc()
        # Record error to memory
        if context.unified_memory:
            context.unified_memory.short_term.add_error(
                agent_id=agent_name,
                error=str(exc),
                context="File processing failed"
            )

    files_created_count = 0
    files_edited_count = 0
    edits_failed_count = 0

    if files_written:
        files_created_count = len(files_written)
        print(f"\n   [OK] Created {files_created_count} new file(s)")
        # Record files created to memory
        for filename in files_written.keys():
            if context.unified_memory:
                content = file_manager.read_file(filename) or ""
                context.unified_memory.short_term.add_code_context(
                    agent_id=agent_name,
                    filename=filename,
                    content=content[:500],
                    purpose="Created by agent"
                )

    if edits_applied:
        successful = sum(1 for value in edits_applied.values() if value)
        failed = len(edits_applied) - successful
        files_edited_count = successful
        edits_failed_count = failed
        print(f"   [OK] Applied {successful} edit(s) to existing files")
        if failed > 0:
            print(f"   [WARN] {failed} edit(s) failed to apply")
            # Record failed edits to memory for debugging
            if context.unified_memory:
                for filepath, success in edits_applied.items():
                    if not success:
                        context.unified_memory.short_term.add_error(
                            agent_id=agent_name,
                            error=f"Edit failed for {filepath}",
                            context="Patch application failed - check .proposed_change file"
                        )

    if not files_written and not edits_applied:
        print(f"\n   [WARN] WARNING: No file changes made by {agent_name}")

    # === VALIDATION AND RETRY LOOP ===
    # Validate generated files for syntax errors and retry if needed
    retry_count = 0
    while retry_count < max_retries:
        if not files_written and not edits_applied:
            break  # Nothing to validate

        print(f"\n   [VALIDATE] Checking generated files for syntax errors...")
        is_valid, validation_errors = _validate_generated_files(
            file_manager, files_written, edits_applied
        )

        if is_valid:
            print(f"   [OK] All files passed validation!")
            break

        # Validation failed - show errors
        retry_count += 1
        print(f"\n   [ERROR] Validation failed! Found {len(validation_errors)} file(s) with errors:")
        for result in validation_errors:
            print(f"      - {result.file_path}: {', '.join(result.errors[:2])}")

        if retry_count >= max_retries:
            print(f"   [WARN] Max retries ({max_retries}) reached. Some files may have syntax errors.")
            # Record to memory
            if context.unified_memory:
                for result in validation_errors:
                    context.unified_memory.short_term.add_error(
                        agent_id=agent_name,
                        error=f"Syntax error in {result.file_path}: {result.errors[0] if result.errors else 'unknown'}",
                        context="Validation failed after retries"
                    )
            break

        # Ask LLM to fix the errors
        print(f"\n   [RETRY] Attempt {retry_count}/{max_retries} - Asking LLM to fix errors...")

        correction_prompt = _build_correction_prompt(
            full_prompt, output, validation_errors, file_manager
        )

        try:
            correction_output = llm_call(correction_prompt, max_tokens=4096, temperature=0.3)
            if correction_output and len(correction_output) > 50:
                # Apply corrections
                print(f"   [FIX] Applying corrections...")
                new_files = extract_and_write_files(correction_output, file_manager, f"{agent_name}_fix")
                new_edits = extract_and_apply_edits(correction_output, file_manager, f"{agent_name}_fix")

                # Update counts
                files_written.update(new_files)
                edits_applied.update(new_edits)

                if new_files or new_edits:
                    print(f"   [OK] Applied {len(new_files)} file writes and {len(new_edits)} edits")
                else:
                    print(f"   [WARN] No corrections could be applied")
                    break
            else:
                print(f"   [WARN] LLM returned empty correction")
                break
        except Exception as e:
            print(f"   [ERROR] Correction failed: {e}")
            break

    # Track execution statistics
    duration = time.time() - start_time
    success = bool(output and (files_written or edits_applied))
    _update_stats(
        agent_name=agent_name,
        success=success,
        files_created=files_created_count,
        files_edited=files_edited_count,
        edits_failed=edits_failed_count,
        duration=duration
    )

    # Optional: Run critic review if provided
    if critic_agent and output:
        try:
            from core.runtime.critic import ApprovalStatus
            print(f"\n   [CRITIC] Reviewing {agent_name} output...")
            critique_result = critic_agent.critique(
                agent_output=output,
                task_description=user_request,
                context={"agent": agent_name}
            )
            print(f"   [CRITIC] Status: {critique_result.status.value}, Score: {critique_result.score:.2f}")
            if critique_result.issues:
                print(f"   [CRITIC] Issues found: {len(critique_result.issues)}")
        except Exception as e:
            print(f"   [CRITIC] Review failed: {e}")

    print(f"\n   [STATS] Agent {agent_name}: {duration:.2f}s, Files: +{files_created_count}, Edits: {files_edited_count}")

    return output


