"""Utility helpers for creating and editing files during orchestration."""

from __future__ import annotations

import os
import sys
import time
import threading
from pathlib import Path
from typing import Dict, List, Optional


def _safe_print(text: str) -> None:
    """Print text safely, handling encoding issues on Windows."""
    try:
        print(text)
    except UnicodeEncodeError:
        # Fallback: remove or replace emojis for Windows console
        safe_text = text.encode('ascii', 'replace').decode('ascii')
        print(safe_text)


class FileManager:
    """Handles file system interactions for generated projects.
    
    Thread-safe file operations with locking around writes and edits.
    """

    def __init__(self, base_path: str):
        self.base_path = Path(base_path)
        self.created_files: List[str] = []
        self.created_dirs: List[str] = []
        self.edit_history: List[dict] = []
        self._lock = threading.RLock()

        self.base_path.mkdir(parents=True, exist_ok=True)
        _safe_print(f"[DIR] Project folder: {self.base_path.absolute()}\n")

    def create_directory(self, rel_path: str) -> Path:
        """Create a directory (thread-safe)."""
        with self._lock:
            full_path = self.base_path / rel_path
            full_path.mkdir(parents=True, exist_ok=True)
            if str(full_path) not in self.created_dirs:
                self.created_dirs.append(str(full_path))
                _safe_print(f"   [+] Created dir: {rel_path}/")
            return full_path

    def write_file(self, rel_path: str, content: str) -> Path:
        """Write file content (thread-safe)."""
        with self._lock:
            full_path = self.base_path / rel_path
            full_path.parent.mkdir(parents=True, exist_ok=True)
            with open(full_path, "w", encoding="utf-8") as handle:
                handle.write(content)

            if str(full_path) not in self.created_files:
                self.created_files.append(str(full_path))
            _safe_print(f"   [+] Written: {rel_path} ({len(content)} bytes)")
            return full_path

    def read_file(self, rel_path: str) -> Optional[str]:
        """Read file content (thread-safe)."""
        with self._lock:
            full_path = self.base_path / rel_path
            if not full_path.exists():
                return None
            with open(full_path, "r", encoding="utf-8") as handle:
                return handle.read()

    def _normalize_for_comparison(self, text: str) -> str:
        """Normalize text for comparison by handling line endings and trailing whitespace."""
        # Normalize line endings to Unix style
        text = text.replace('\r\n', '\n').replace('\r', '\n')
        # Strip trailing whitespace from each line but preserve indentation
        lines = [line.rstrip() for line in text.split('\n')]
        return '\n'.join(lines)

    def _find_best_match_position(self, content: str, old_code: str) -> tuple:
        """Find the best matching position for old_code in content.

        Returns (start_idx, end_idx, match_quality) where:
        - start_idx, end_idx are line indices (or -1 if no match)
        - match_quality is 'exact', 'normalized', 'fuzzy', or None
        """
        # Normalize both for comparison
        norm_content = self._normalize_for_comparison(content)
        norm_old = self._normalize_for_comparison(old_code)

        # Stage 1: Try exact match on normalized content
        if norm_old in norm_content:
            # Find line-based position for replacement
            content_lines = content.split('\n')
            old_lines = old_code.split('\n')

            # Find where the normalized match occurs
            for start_idx in range(len(content_lines) - len(old_lines) + 1):
                chunk = '\n'.join(content_lines[start_idx:start_idx + len(old_lines)])
                if self._normalize_for_comparison(chunk) == norm_old:
                    return (start_idx, start_idx + len(old_lines), 'normalized')

        # Stage 2: Try matching with whitespace-only differences
        content_lines = content.split('\n')
        old_lines = old_code.strip().split('\n')

        if len(old_lines) == 0:
            return (-1, -1, None)

        for start_idx in range(len(content_lines) - len(old_lines) + 1):
            potential_lines = content_lines[start_idx:start_idx + len(old_lines)]

            # Compare stripped versions line by line
            if all(
                ol.strip() == pl.strip()
                for ol, pl in zip(old_lines, potential_lines)
            ):
                return (start_idx, start_idx + len(old_lines), 'normalized')

        # Stage 3: Fuzzy match - find region with highest similarity
        best_ratio = 0.0
        best_start = -1

        # Normalize old_code for fuzzy comparison (remove all whitespace)
        old_normalized = ''.join(old_lines).replace(' ', '').replace('\t', '').lower()

        if len(old_normalized) < 5:  # Too short for meaningful fuzzy match
            return (-1, -1, None)

        for start_idx in range(len(content_lines) - len(old_lines) + 1):
            potential_lines = content_lines[start_idx:start_idx + len(old_lines)]
            pot_normalized = ''.join(potential_lines).replace(' ', '').replace('\t', '').lower()

            if not pot_normalized:
                continue

            # Calculate similarity ratio properly
            # Use the longer string as the denominator to avoid inflated ratios
            max_len = max(len(old_normalized), len(pot_normalized))
            if max_len == 0:
                continue

            # Count matching characters
            matches = sum(1 for a, b in zip(old_normalized, pot_normalized) if a == b)
            # Penalize length differences
            length_penalty = abs(len(old_normalized) - len(pot_normalized)) / max_len
            ratio = (matches / max_len) - (length_penalty * 0.5)

            if ratio > best_ratio:
                best_ratio = ratio
                best_start = start_idx

        # Require at least 80% match for fuzzy
        if best_ratio >= 0.80 and best_start >= 0:
            return (best_start, best_start + len(old_lines), 'fuzzy')

        return (-1, -1, None)

    def edit_file(self, rel_path: str, old_code: str, new_code: str) -> bool:
        """Edit file by replacing code section (thread-safe, 3-stage strategy).

        Stages:
        1. Exact match (after line ending normalization)
        2. Normalized match (whitespace-insensitive line comparison)
        3. Fuzzy match (80%+ character similarity)
        4. Fallback: Create .proposed_change file for manual review
        """
        with self._lock:
            content = self.read_file(rel_path)
            if content is None:
                _safe_print(f"   [!] File not found: {rel_path}")
                return False

            # Normalize line endings in both content and old_code
            norm_content = self._normalize_for_comparison(content)
            norm_old = self._normalize_for_comparison(old_code)

            # Stage 1: Direct exact match (fastest path)
            if old_code in content:
                updated = content.replace(old_code, new_code, 1)
                full_path = self.base_path / rel_path
                with open(full_path, "w", encoding="utf-8") as handle:
                    handle.write(updated)

                self.edit_history.append(
                    {"file": rel_path, "timestamp": time.time(), "action": "replace", "stage": 1}
                )
                _safe_print(f"   [~] Edited (exact): {rel_path}")
                return True

            # Stage 1b: Exact match with normalized line endings
            if norm_old in norm_content:
                updated = norm_content.replace(norm_old, self._normalize_for_comparison(new_code), 1)
                full_path = self.base_path / rel_path
                with open(full_path, "w", encoding="utf-8") as handle:
                    handle.write(updated)

                self.edit_history.append(
                    {"file": rel_path, "timestamp": time.time(), "action": "replace", "stage": "1b"}
                )
                _safe_print(f"   [~] Edited (normalized): {rel_path}")
                return True

            # Stage 2 & 3: Use position finder for normalized/fuzzy matches
            start_idx, end_idx, match_quality = self._find_best_match_position(content, old_code)

            if match_quality in ('normalized', 'fuzzy') and start_idx >= 0:
                content_lines = content.split('\n')
                new_lines = new_code.split('\n')
                updated_lines = content_lines[:start_idx] + new_lines + content_lines[end_idx:]
                updated = '\n'.join(updated_lines)

                full_path = self.base_path / rel_path
                with open(full_path, "w", encoding="utf-8") as handle:
                    handle.write(updated)

                stage = 2 if match_quality == 'normalized' else 3
                self.edit_history.append(
                    {"file": rel_path, "timestamp": time.time(), "action": "replace", "stage": stage}
                )
                _safe_print(f"   [~] Edited ({match_quality}): {rel_path}")
                return True

            # Stage 4: Proposed change file (PR-style fallback)
            proposed_file = f"{rel_path}.proposed_change"
            proposed_path = self.base_path / proposed_file

            # Show debug info to help diagnose why match failed
            old_preview = old_code[:200] + "..." if len(old_code) > 200 else old_code

            proposal = f"""=== PROPOSED CHANGE ===
File: {rel_path}
Auto-applied: False (patch failed - manual review needed)

--- OLD CODE (expected to find) ---
{old_code}

--- NEW CODE (to replace with) ---
{new_code}

--- DEBUG INFO ---
File size: {len(content)} bytes
Old code size: {len(old_code)} bytes
Stages attempted: exact, normalized, fuzzy (80% threshold)
All stages failed - content may have changed or old_code is incorrect.

Tip: Copy the exact text from the file (including whitespace) into the 'old:' section.
"""

            with open(proposed_path, "w", encoding="utf-8") as handle:
                handle.write(proposal)

            self.edit_history.append(
                {"file": rel_path, "timestamp": time.time(), "action": "propose", "stage": 4}
            )
            _safe_print(f"   [?] Created proposal: {proposed_file} (manual review needed)")
            return False

    def append_to_file(self, rel_path: str, content: str) -> bool:
        """Append content to file (thread-safe)."""
        with self._lock:
            full_path = self.base_path / rel_path
            if not full_path.exists():
                self.write_file(rel_path, content)
                return True

            with open(full_path, "a", encoding="utf-8") as handle:
                handle.write("\n" + content)

            self.edit_history.append(
                {"file": rel_path, "timestamp": time.time(), "action": "append"}
            )
            _safe_print(f"   [+] Appended to: {rel_path}")
            return True

    def insert_at_line(self, rel_path: str, line_num: int, content: str) -> bool:
        """Insert content at line number (thread-safe)."""
        with self._lock:
            file_content = self.read_file(rel_path)
            if file_content is None:
                return False

            lines = file_content.split("\n")
            if line_num < 0 or line_num > len(lines):
                _safe_print(f"   [!] Line number {line_num} out of range")
                return False

            lines.insert(line_num, content)
            updated = "\n".join(lines)
            full_path = self.base_path / rel_path
            with open(full_path, "w", encoding="utf-8") as handle:
                handle.write(updated)

            self.edit_history.append(
                {
                    "file": rel_path,
                    "line": line_num,
                    "timestamp": time.time(),
                    "action": "insert",
                }
            )
            _safe_print(f"   [+] Inserted at line {line_num}: {rel_path}")
            return True

    def list_files(self, pattern: str = "*") -> List[str]:
        """List all files recursively in the project directory, including all subdirectories."""
        try:
            files = list(self.base_path.rglob(pattern))
            # Convert to relative paths and filter only files
            relative_files = [
                str(Path(file).relative_to(self.base_path))
                for file in files
                if file.is_file()
            ]
            return relative_files
        except Exception as e:
            # If rglob fails, try simple glob as fallback
            try:
                files = list(self.base_path.glob(pattern))
                return [
                    str(Path(file).relative_to(self.base_path))
                    for file in files
                    if file.is_file()
                ]
            except:
                return []

    def get_project_structure_tree(self) -> str:
        """Generate a tree-like project structure visualization with nested directories."""
        files = self.list_files("*")
        if not files:
            return "📁 (empty project)\n"
        
        # Build a nested directory tree structure
        def build_tree(file_paths: List[str]) -> dict:
            """Build a nested tree structure from file paths, storing full relative paths."""
            tree = {}
            for file_path in file_paths:
                parts = Path(file_path).parts
                current = tree
                # Navigate/create directory structure
                for part in parts[:-1]:  # All parts except filename
                    if part not in current:
                        current[part] = {}
                    current = current[part]
                # Add file to current directory (store both filename and full path)
                if "_files" not in current:
                    current["_files"] = []
                current["_files"].append((parts[-1], file_path))  # (filename, full_relative_path)
            return tree
        
        def render_tree(node: dict, prefix: str = "", is_last: bool = True) -> str:
            """Recursively render the tree structure."""
            result = ""
            items = [(k, v) for k, v in node.items() if k != "_files"]
            files = node.get("_files", [])
            
            # Render files in current directory
            # Files are stored as (filename, full_relative_path) tuples
            for i, file_info in enumerate(sorted(files, key=lambda x: x[0] if isinstance(x, tuple) else x)):
                if isinstance(file_info, tuple):
                    file, file_path = file_info
                else:
                    file = file_info
                    file_path = file_info
                
                is_file_last = (i == len(files) - 1) and len(items) == 0
                connector = "└── " if is_file_last else "├── "
                try:
                    full_path = self.base_path / file_path
                    if full_path.exists() and full_path.is_file():
                        size = os.path.getsize(full_path)
                        file_ext = Path(file).suffix if Path(file).suffix else "(no ext)"
                        result += f"{prefix}{connector}📄 {file} ({size} bytes) {file_ext}\n"
                    else:
                        result += f"{prefix}{connector}📄 {file}\n"
                except Exception:
                    result += f"{prefix}{connector}📄 {file}\n"
            
            # Render subdirectories
            for i, (dir_name, subnode) in enumerate(sorted(items)):
                is_dir_last = i == len(items) - 1
                connector = "└── " if is_dir_last else "├── "
                result += f"{prefix}{connector}📂 {dir_name}/\n"
                extension = "    " if is_dir_last else "│   "
                result += render_tree(subnode, prefix + extension, is_dir_last)
            
            return result
        
        tree = build_tree(files)
        structure = "📁 PROJECT STRUCTURE (All Files & Subdirectories):\n"
        structure += f"Base: {self.base_path.absolute()}\n\n"
        structure += render_tree(tree)
        
        return structure

    def get_project_summary(self) -> str:
        """Get basic project summary (legacy method)."""
        summary = "PROJECT STRUCTURE:\n"
        summary += f"Base Path: {self.base_path.absolute()}\n\n"

        files = self.list_files("*")
        for file_path in sorted(files):
            try:
                size = os.path.getsize(self.base_path / file_path)
                summary += f"  ✓ {file_path} ({size} bytes)\n"
            except OSError:
                continue

        return summary

    def get_file_summary(self, rel_path: str, max_length: int = 500) -> str:
        """Generate a brief summary of a file's purpose and content."""
        content = self.read_file(rel_path)
        if not content:
            return "File not found or empty"
        
        # Simple heuristic-based summary
        ext = Path(rel_path).suffix.lower()
        lines = content.split('\n')
        line_count = len(lines)
        char_count = len(content)
        
        # Try to extract key info based on file type
        summary_parts = [
            f"📄 {rel_path}",
            f"   Size: {char_count} chars, {line_count} lines",
        ]
        
        if ext == '.py':
            # Python file - look for imports, classes, functions
            imports = [l for l in lines[:20] if l.strip().startswith('import') or l.strip().startswith('from')]
            classes = [l.strip() for l in lines if l.strip().startswith('class ')]
            functions = [l.strip() for l in lines if l.strip().startswith('def ') and not l.strip().startswith('def _')]
            
            if imports:
                summary_parts.append(f"   Imports: {', '.join([i.split()[1].split('.')[0] for i in imports[:3]])}")
            if classes:
                summary_parts.append(f"   Classes: {', '.join([c.split('(')[0].replace('class ', '') for c in classes[:3]])}")
            if functions:
                summary_parts.append(f"   Functions: {', '.join([f.split('(')[0].replace('def ', '') for f in functions[:5]])}")
        
        elif ext in ['.html', '.htm']:
            # HTML file - look for title, main elements
            title_match = [l for l in lines if '<title>' in l.lower() or '<h1>' in l.lower()]
            if title_match:
                summary_parts.append(f"   Contains: {title_match[0][:50]}...")
        
        elif ext in ['.js', '.jsx']:
            # JavaScript - look for exports, functions
            exports = [l.strip() for l in lines if 'export' in l or 'function' in l or 'const' in l]
            if exports:
                summary_parts.append(f"   Exports/Functions: {len([e for e in exports if 'function' in e or 'export' in e])} found")
        
        elif ext == '.css':
            summary_parts.append(f"   Stylesheet with {len([l for l in lines if '{' in l])} rule blocks")
        
        # Add first few non-empty lines as preview
        preview_lines = [l.strip() for l in lines[:10] if l.strip() and not l.strip().startswith('#') and not l.strip().startswith('//')]
        if preview_lines:
            preview = ' '.join(preview_lines[:2])[:100]
            summary_parts.append(f"   Preview: {preview}...")
        
        return '\n'.join(summary_parts)


