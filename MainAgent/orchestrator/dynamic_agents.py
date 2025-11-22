"""Utilities for spawning specialized agents that manipulate project files."""

from __future__ import annotations

import os
from pathlib import Path
from typing import List

from .context import AgentContext
from .extractors import extract_and_apply_edits, extract_and_write_files
from .file_manager import FileManager
from MainAgent.core.runtime.llm import llm_call


def create_dynamic_agent(
    agent_name: str,
    custom_prompt: str,
    user_request: str,
    context: AgentContext,
    file_manager: FileManager,
) -> str:
    """Execute a dynamic agent prompt and materialize produced files."""

    # Get project structure and file summaries
    project_structure = file_manager.get_project_structure_tree()
    available_files = file_manager.list_files("*")

    # Build comprehensive file context for agents
    files_context = "=" * 70 + "\n"
    files_context += "📁 PROJECT STRUCTURE & FILE SUMMARIES\n"
    files_context += "=" * 70 + "\n\n"
    
    if available_files:
        files_context += project_structure
        files_context += f"\n📊 Total Files Found: {len(available_files)}\n"
        
        # Show directory breakdown
        dirs = set()
        for file_path in available_files:
            parent = str(Path(file_path).parent)
            if parent != ".":
                dirs.add(parent)
        
        if dirs:
            files_context += f"📂 Subdirectories: {len(dirs)}\n"
            files_context += f"   {', '.join(sorted(dirs)[:10])}"
            if len(dirs) > 10:
                files_context += f" ... and {len(dirs) - 10} more"
            files_context += "\n"
        
        files_context += "\n"
        
        files_context += "=" * 70 + "\n"
        files_context += "📋 FILE SUMMARIES (Quick Overview)\n"
        files_context += "=" * 70 + "\n\n"
        
        # Generate summaries for all files
        for file_name in sorted(available_files):
            summary = file_manager.get_file_summary(file_name)
            files_context += f"{summary}\n\n"
        
        files_context += "=" * 70 + "\n"
        files_context += "📄 FULL FILE CONTENTS (For Editing)\n"
        files_context += "=" * 70 + "\n\n"
        
        # Include full file contents (with size limits for very large files)
        # Increased limit for directory-level operations
        file_limit = min(50, len(available_files))  # Show up to 50 files
        for file_name in sorted(available_files)[:file_limit]:
            content = file_manager.read_file(file_name)
            if content:
                files_context += f"\n{'='*70}\n"
                files_context += f"📄 FILE: {file_name}\n"
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
                files_context += f"📄 FILE: {file_name}\n"
                files_context += f"[Could not read file or file is empty]\n"
                files_context += f"{'='*70}\n\n"
    else:
        files_context += "📁 (Empty project - no files exist yet)\n"
        files_context += "You can create new files using the format below.\n\n"

    full_prompt = f"""{custom_prompt}

{files_context}

=== SHARED CONTEXT FROM PREVIOUS AGENTS ===
{context.get_context()}

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
        print(f"   ❌ Error calling LLM: {exc}\n")
        output = ""

    if not output or len(output) < 50:
        print(f"   ⚠️  Warning: {agent_name} produced minimal output")

    context.add_result(agent_name, output)

    print(f"\n   🔄 Processing {agent_name} output - creating/editing files...\n")
    try:
        files_written = extract_and_write_files(output, file_manager, agent_name)
        edits_applied = extract_and_apply_edits(output, file_manager, agent_name)
    except Exception as exc:
        print(f"   ❌ Error processing agent output: {exc}\n")
        import traceback
        traceback.print_exc()
        files_written = {}
        edits_applied = {}

    if files_written:
        print(f"\n   ✅ Created {len(files_written)} new file(s)")
    if edits_applied:
        successful = sum(1 for value in edits_applied.values() if value)
        print(f"   ✅ Applied {successful} edit(s) to existing files")
    if not files_written and not edits_applied:
        print(f"\n   ⚠️  WARNING: No file changes made by {agent_name}")

    return output


