"""Parse agent outputs for code blocks and file edit instructions."""

from __future__ import annotations

import re
from typing import Dict

from .file_manager import FileManager


def extract_and_write_files(
    agent_output: str,
    file_manager: FileManager,
    agent_name: str,
) -> Dict[str, int]:
    """Extract code fences from agent output and write them to the project."""

    print(f"\n   🔍 DEBUG: Searching for code blocks in {agent_name} output...")
    print(f"   📝 Output length: {len(agent_output)} characters\n")

    files_written: Dict[str, int] = {}
    processed_ranges: list[tuple[int, int]] = []  # Track processed regions to avoid duplicates

    def is_already_processed(start: int, end: int) -> bool:
        """Check if this range overlaps with already processed regions."""
        for p_start, p_end in processed_ranges:
            if not (end < p_start or start > p_end):
                return True
        return False

    def clean_filename(filename: str) -> str:
        """Clean filename by removing common prefixes and invalid characters."""
        # Remove "filename: " prefix if present
        filename = re.sub(r"^filename\s*:\s*", "", filename, flags=re.IGNORECASE)
        # Remove backticks and whitespace
        filename = filename.strip("`").strip()
        # Remove any Windows-invalid characters
        invalid_chars = r'[<>:"|?*]'
        filename = re.sub(invalid_chars, "_", filename)
        return filename

    # Patterns in order of specificity (most specific first)
    patterns = [
        (r"```\s*filename\s*:\s*([^\n]+)\s*\n([\s\S]*?)```", "filename: format"),
        (
            r"FILE:\s*([^\n]+\.(?:py|js|jsx|ts|tsx|json|yaml|yml|html|css|txt|md))\s*\n```[^\n]*\n([\s\S]*?)```",
            "FILE: format",
        ),
        (
            r"```(?:python|javascript|js|jsx|typescript|ts|tsx|json|yaml|yml|html|css|bash|sh)\s+([^\n]+\.[a-zA-Z0-9]+)\s*\n([\s\S]*?)```",
            "language + filename",
        ),
        (
            r"```\s*([^\n]+\.(?:py|js|jsx|ts|tsx|json|yaml|yml|html|css|txt|md|sh|bash))\s*\n([\s\S]*?)```",
            "simple path",
        ),
        (r"```(python|javascript|js|html|css)\s*\n([\s\S]*?)```", "language only"),
    ]

    for pattern, name in patterns:
        matches = list(re.finditer(pattern, agent_output, re.IGNORECASE | re.MULTILINE))
        if matches:
            print(f"   ✓ Found {len(matches)} match(es) with pattern: {name}")

        for idx, match in enumerate(matches):
            # Skip if this region was already processed
            if is_already_processed(match.start(), match.end()):
                continue

            filename = match.group(1).strip()
            content = match.group(2).strip()

            if name == "language only":
                lang = filename
                preceding = agent_output[: match.start()]
                filename_match = re.search(
                    r"([a-zA-Z0-9_\-./]+\.(?:py|js|jsx|html|css|txt|md))(?!.*\1)",
                    preceding,
                )
                if filename_match:
                    filename = filename_match.group(1)
                else:
                    ext_map = {
                        "python": "py",
                        "javascript": "js",
                        "js": "js",
                        "html": "html",
                        "css": "css",
                    }
                    filename = f"{agent_name.lower()}_{idx + 1}.{ext_map.get(lang, 'txt')}"

            # Clean the filename
            filename = clean_filename(filename)

            # Validate filename
            if not filename or filename.startswith("filename:") or len(filename) < 3:
                continue

            # Skip if content is too short
            if not content or len(content) < 10:
                continue

            # Check if file already exists (this is an edit/replacement)
            file_exists = file_manager.read_file(filename) is not None
            
            # Skip if we already wrote this file in this run
            if filename in files_written:
                continue

            if file_exists:
                print(f"   🔄 File exists, replacing: {filename}")
            else:
                print(f"   📝 Attempting to write: {filename}")
            
            try:
                file_manager.write_file(filename, content)
                files_written[filename] = len(content)
                # Mark this region as processed
                processed_ranges.append((match.start(), match.end()))
            except Exception as exc:  # pragma: no cover
                print(f"   ❌ Error writing {filename}: {exc}")

    if not files_written:
        print(f"   ⚠️  No files extracted from {agent_name} output")
        debug_file = f"debug_{agent_name.lower()}_output.txt"
        try:
            file_manager.write_file(debug_file, agent_output)
            print(f"   📄 Saved raw output to {debug_file} for inspection")
        except Exception:
            pass

    return files_written


def extract_and_apply_edits(
    agent_output: str,
    file_manager: FileManager,
    agent_name: str,
) -> Dict[str, bool]:
    """Extract edit instructions and apply them to the project files."""

    edits_applied: Dict[str, bool] = {}
    
    # Pattern 1: ===EDIT=== format (most explicit)
    edit_pattern = (
        r"===EDIT===\s*\nfile:\s*([^\n]+)\s*\nold:\s*\n([\s\S]*?)\nnew:\s*\n([\s\S]*?)\n===END==="
    )

    for match in re.finditer(edit_pattern, agent_output, re.IGNORECASE | re.MULTILINE):
        filepath = match.group(1).strip()
        old_code = match.group(2).strip()
        new_code = match.group(3).strip()
        
        if not filepath or not old_code or not new_code:
            continue
            
        print(f"   ✏️  Attempting to edit: {filepath}")
        success = file_manager.edit_file(filepath, old_code, new_code)
        edits_applied[filepath] = success
        if not success:
            print(f"   ⚠️  Edit failed - check if old code matches exactly")

    # Pattern 2: APPEND_TO format
    append_pattern = r"APPEND_TO:\s*([^\n]+)\s*\n([\s\S]*?)\nEND_APPEND"
    for match in re.finditer(append_pattern, agent_output, re.IGNORECASE | re.MULTILINE):
        filepath = match.group(1).strip()
        content = match.group(2).strip()
        
        if not filepath or not content:
            continue
            
        print(f"   ➕ Attempting to append to: {filepath}")
        success = file_manager.append_to_file(filepath, content)
        edits_applied[f"{filepath}_append"] = success

    return edits_applied


