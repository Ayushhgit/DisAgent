"""Parse agent outputs for code blocks and file edit instructions."""

from __future__ import annotations

import re
import sys
from typing import Dict

from .file_manager import FileManager


def _safe_print(text: str) -> None:
    """Print text safely, handling encoding issues on Windows."""
    try:
        print(text)
    except UnicodeEncodeError:
        # Fallback: remove or replace emojis for Windows console
        safe_text = text.encode('ascii', 'replace').decode('ascii')
        print(safe_text)


def extract_and_write_files(
    agent_output: str,
    file_manager: FileManager,
    agent_name: str,
) -> Dict[str, int]:
    """Extract code fences from agent output and write them to the project."""

    _safe_print(f"\n   [DEBUG] Searching for code blocks in {agent_name} output...")
    _safe_print(f"   [INFO] Output length: {len(agent_output)} characters\n")

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
            _safe_print(f"   [OK] Found {len(matches)} match(es) with pattern: {name}")

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
                _safe_print(f"   [REPLACE] File exists, replacing: {filename}")
            else:
                _safe_print(f"   [WRITE] Attempting to write: {filename}")

            try:
                file_manager.write_file(filename, content)
                files_written[filename] = len(content)
                # Mark this region as processed
                processed_ranges.append((match.start(), match.end()))
            except Exception as exc:  # pragma: no cover
                _safe_print(f"   [ERROR] Error writing {filename}: {exc}")

    if not files_written:
        _safe_print(f"   [WARN] No files extracted from {agent_name} output")
        debug_file = f"debug_{agent_name.lower()}_output.txt"
        try:
            file_manager.write_file(debug_file, agent_output)
            _safe_print(f"   [DEBUG] Saved raw output to {debug_file} for inspection")
        except Exception:
            pass

    return files_written


def _normalize_line_endings(text: str) -> str:
    """Normalize line endings to Unix style and handle common whitespace issues."""
    # Convert Windows line endings to Unix
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    return text


def _strip_outer_only(text: str) -> str:
    """Strip only the outer whitespace while preserving internal indentation.

    This removes leading/trailing blank lines but keeps indentation on content lines.
    IMPORTANT: Preserves internal structure exactly as-is.
    """
    if not text:
        return ""

    lines = text.split('\n')

    # Find first non-empty line
    start = 0
    while start < len(lines) and not lines[start].strip():
        start += 1

    # Find last non-empty line
    end = len(lines) - 1
    while end >= 0 and not lines[end].strip():
        end -= 1

    if start > end:
        return ""

    # Return lines with preserved indentation, but strip trailing whitespace from each line
    result_lines = [line.rstrip() for line in lines[start:end + 1]]
    return '\n'.join(result_lines)


def _extract_code_block(text: str, start_marker: str) -> str:
    """Extract code from a block, handling both inline and multi-line formats.

    Handles formats like:
    - old:\ncode here
    - old: code here
    - old:\n```\ncode\n```
    """
    if not text:
        return ""

    text = text.strip()

    # If wrapped in code fences, extract content
    if text.startswith('```'):
        # Find the end of the code fence
        lines = text.split('\n')
        if len(lines) > 1:
            # Skip first line (```language or just ```)
            content_lines = []
            for line in lines[1:]:
                if line.strip() == '```':
                    break
                content_lines.append(line)
            return '\n'.join(content_lines)

    return text


def extract_and_apply_edits(
    agent_output: str,
    file_manager: FileManager,
    agent_name: str,
) -> Dict[str, bool]:
    """Extract edit instructions and apply them to the project files.

    Supports multiple edit formats:
    1. ===EDIT=== / ===END=== blocks
    2. APPEND_TO: / END_APPEND blocks
    3. Various whitespace and formatting variations
    """

    edits_applied: Dict[str, bool] = {}

    # Normalize line endings in the entire output first
    agent_output = _normalize_line_endings(agent_output)

    # Multiple patterns to handle different EDIT block formats
    edit_patterns = [
        # Pattern 1: Standard format with clear separators
        r"""===\s*EDIT\s*===\s*\n
            \s*file\s*:\s*([^\n]+?)\s*\n
            \s*old\s*:\s*\n
            ([\s\S]*?)
            \n\s*new\s*:\s*\n
            ([\s\S]*?)
            \n\s*===\s*END\s*===""",

        # Pattern 2: Compact format (content on same line as old:/new:)
        r"""===\s*EDIT\s*===\s*\n
            \s*file\s*:\s*([^\n]+?)\s*\n
            \s*old\s*:\s*
            ([\s\S]*?)
            \nnew\s*:\s*
            ([\s\S]*?)
            \n===\s*END\s*===""",

        # Pattern 3: With code fences inside
        r"""===\s*EDIT\s*===\s*\n
            \s*file\s*:\s*([^\n]+?)\s*\n
            \s*old\s*:\s*\n?
            ```[^\n]*\n([\s\S]*?)```\s*\n
            \s*new\s*:\s*\n?
            ```[^\n]*\n([\s\S]*?)```\s*\n
            \s*===\s*END\s*===""",

        # Pattern 4: Very lenient pattern as fallback
        r"""===\s*EDIT\s*===.*?\n
            .*?file\s*:\s*([^\n]+?)\s*\n
            .*?old\s*:[ \t]*\n?([\s\S]*?)
            \n.*?new\s*:[ \t]*\n?([\s\S]*?)
            \n.*?===\s*END\s*===""",
    ]

    processed_edits = set()  # Track to avoid duplicate edits
    processed_positions = set()  # Track by match position

    for pattern_idx, edit_pattern in enumerate(edit_patterns):
        for match in re.finditer(edit_pattern, agent_output, re.IGNORECASE | re.MULTILINE | re.VERBOSE):
            # Skip if we've already processed this position with any pattern
            if match.start() in processed_positions:
                continue

            filepath = match.group(1).strip()

            # Extract and clean old/new code
            old_code_raw = match.group(2)
            new_code_raw = match.group(3)

            # Handle code fences if present
            old_code = _extract_code_block(old_code_raw, "old") if '```' in old_code_raw else old_code_raw
            new_code = _extract_code_block(new_code_raw, "new") if '```' in new_code_raw else new_code_raw

            # Clean up the code blocks
            old_code = _strip_outer_only(old_code)
            new_code = _strip_outer_only(new_code)

            # Create unique key for this edit
            edit_key = f"{filepath}:{hash(old_code)}"
            if edit_key in processed_edits:
                continue
            processed_edits.add(edit_key)
            processed_positions.add(match.start())

            if not filepath:
                _safe_print(f"   [WARN] Skipping edit: no filepath specified")
                continue

            if not old_code:
                _safe_print(f"   [WARN] Skipping edit for {filepath}: no old code specified")
                continue

            # new_code can be empty (deletion case)
            _safe_print(f"   [EDIT] Attempting to edit: {filepath}")
            _safe_print(f"      Pattern: {pattern_idx + 1}, Old: {len(old_code)} chars, New: {len(new_code)} chars")

            # Debug: Show first few chars of old code being searched
            old_preview = old_code[:80].replace('\n', '\\n') if old_code else "(empty)"
            _safe_print(f"      Looking for: {old_preview}...")

            success = file_manager.edit_file(filepath, old_code, new_code)
            edits_applied[filepath] = success

            if not success:
                _safe_print(f"   [WARN] Edit failed for {filepath}")
                _safe_print(f"      Tip: Check .proposed_change file for manual review")

    # Pattern for APPEND_TO format
    append_pattern = r"APPEND_TO:\s*([^\n]+)\s*\n([\s\S]*?)\nEND_APPEND"
    for match in re.finditer(append_pattern, agent_output, re.IGNORECASE | re.MULTILINE):
        filepath = match.group(1).strip()
        content = _strip_outer_only(match.group(2))

        if not filepath or not content:
            continue

        _safe_print(f"   [APPEND] Attempting to append to: {filepath}")
        success = file_manager.append_to_file(filepath, content)
        edits_applied[f"{filepath}_append"] = success

    # Also look for INSERT_AT patterns
    insert_pattern = r"INSERT_AT:\s*([^\n]+)\s*line\s*:\s*(\d+)\s*\n([\s\S]*?)\nEND_INSERT"
    for match in re.finditer(insert_pattern, agent_output, re.IGNORECASE | re.MULTILINE):
        filepath = match.group(1).strip()
        line_num = int(match.group(2))
        content = _strip_outer_only(match.group(3))

        if not filepath or not content:
            continue

        _safe_print(f"   [INSERT] Attempting to insert at line {line_num} in: {filepath}")
        success = file_manager.insert_at_line(filepath, line_num, content)
        edits_applied[f"{filepath}_insert_{line_num}"] = success

    return edits_applied


