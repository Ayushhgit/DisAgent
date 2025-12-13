"""Utility helpers for creating and editing files during orchestration.

Features:
- Thread-safe file operations with RLock
- Transaction support with rollback capability
- Edit history tracking
- Automatic backup before modifications
"""

from __future__ import annotations

import os
import sys
import time
import shutil
import threading
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
from contextlib import contextmanager
import logging

logger = logging.getLogger(__name__)


def _safe_print(text: str) -> None:
    """Print text safely, handling encoding issues on Windows."""
    try:
        print(text)
    except UnicodeEncodeError:
        # Fallback: remove or replace emojis for Windows console
        safe_text = text.encode('ascii', 'replace').decode('ascii')
        print(safe_text)


@dataclass
class FileSnapshot:
    """Snapshot of a file for rollback purposes."""
    rel_path: str
    content: Optional[str]  # None means file didn't exist
    timestamp: float = field(default_factory=time.time)
    existed: bool = True


@dataclass
class Transaction:
    """Represents a file operation transaction."""
    id: str
    snapshots: Dict[str, FileSnapshot] = field(default_factory=dict)
    operations: List[Dict[str, Any]] = field(default_factory=list)
    started_at: float = field(default_factory=time.time)
    committed: bool = False
    rolled_back: bool = False


class FileManager:
    """Handles file system interactions for generated projects.

    Thread-safe file operations with locking around writes and edits.
    Supports transactions with rollback capability.
    """

    def __init__(self, base_path: str, enable_backups: bool = True, max_backups: int = 10):
        self.base_path = Path(base_path)
        self.created_files: List[str] = []
        self.created_dirs: List[str] = []
        self.edit_history: List[dict] = []
        self._lock = threading.RLock()

        # Transaction support
        self._current_transaction: Optional[Transaction] = None
        self._transaction_lock = threading.RLock()

        # Backup configuration
        self.enable_backups = enable_backups
        self.max_backups = max_backups
        self._backup_dir = self.base_path / ".backups"

        self.base_path.mkdir(parents=True, exist_ok=True)
        _safe_print(f"[DIR] Project folder: {self.base_path.absolute()}\n")

    @contextmanager
    def transaction(self, transaction_id: Optional[str] = None):
        """Context manager for file operations transaction.

        Usage:
            with file_manager.transaction("my_edit") as txn:
                file_manager.write_file("test.py", "content")
                file_manager.edit_file("other.py", old, new)
                # If exception occurs, all changes are rolled back

        Args:
            transaction_id: Optional identifier for the transaction
        """
        txn_id = transaction_id or f"txn_{int(time.time() * 1000)}"

        with self._transaction_lock:
            if self._current_transaction is not None:
                raise RuntimeError("Nested transactions are not supported")

            self._current_transaction = Transaction(id=txn_id)

        try:
            yield self._current_transaction
            # Commit on successful completion
            with self._transaction_lock:
                if self._current_transaction:
                    self._current_transaction.committed = True
                    _safe_print(f"   [✓] Transaction {txn_id} committed")
        except Exception as e:
            # Rollback on exception
            self.rollback()
            raise
        finally:
            with self._transaction_lock:
                self._current_transaction = None

    def _snapshot_file(self, rel_path: str) -> FileSnapshot:
        """Create a snapshot of a file before modification."""
        content = self.read_file(rel_path)
        return FileSnapshot(
            rel_path=rel_path,
            content=content,
            existed=content is not None
        )

    def _record_operation(self, rel_path: str, operation: str, backup: bool = True):
        """Record an operation for potential rollback."""
        with self._transaction_lock:
            if self._current_transaction is not None:
                # Take snapshot if not already taken
                if rel_path not in self._current_transaction.snapshots:
                    if backup:
                        self._current_transaction.snapshots[rel_path] = self._snapshot_file(rel_path)
                self._current_transaction.operations.append({
                    "rel_path": rel_path,
                    "operation": operation,
                    "timestamp": time.time()
                })

    def rollback(self) -> bool:
        """Rollback all changes in the current transaction.

        Returns:
            True if rollback was successful, False otherwise
        """
        with self._transaction_lock:
            if self._current_transaction is None:
                logger.warning("No active transaction to rollback")
                return False

            if self._current_transaction.rolled_back:
                return True

            txn = self._current_transaction
            _safe_print(f"   [!] Rolling back transaction {txn.id}...")

            success = True
            for rel_path, snapshot in txn.snapshots.items():
                try:
                    full_path = self.base_path / rel_path

                    if snapshot.existed and snapshot.content is not None:
                        # Restore original content
                        full_path.parent.mkdir(parents=True, exist_ok=True)
                        with open(full_path, "w", encoding="utf-8") as f:
                            f.write(snapshot.content)
                        _safe_print(f"   [←] Restored: {rel_path}")
                    elif not snapshot.existed:
                        # File didn't exist before, delete it
                        if full_path.exists():
                            full_path.unlink()
                            _safe_print(f"   [×] Removed: {rel_path}")
                except Exception as e:
                    logger.exception(f"Failed to rollback {rel_path}: {e}")
                    success = False

            txn.rolled_back = True
            return success

    def create_backup(self, rel_path: str) -> Optional[str]:
        """Create a backup of a file before modification.

        Returns:
            Path to backup file, or None if backup disabled/failed
        """
        if not self.enable_backups:
            return None

        with self._lock:
            full_path = self.base_path / rel_path
            if not full_path.exists():
                return None

            try:
                self._backup_dir.mkdir(parents=True, exist_ok=True)

                # Create timestamped backup filename
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                backup_name = f"{rel_path.replace('/', '_').replace(os.sep, '_')}_{timestamp}.bak"
                backup_path = self._backup_dir / backup_name

                shutil.copy2(full_path, backup_path)

                # Cleanup old backups
                self._cleanup_old_backups(rel_path)

                return str(backup_path)
            except Exception as e:
                logger.warning(f"Failed to create backup for {rel_path}: {e}")
                return None

    def _cleanup_old_backups(self, rel_path: str):
        """Remove old backups beyond max_backups limit."""
        try:
            prefix = rel_path.replace('/', '_').replace(os.sep, '_')
            backups = sorted(
                [f for f in self._backup_dir.glob(f"{prefix}_*.bak")],
                key=lambda p: p.stat().st_mtime,
                reverse=True
            )

            for old_backup in backups[self.max_backups:]:
                old_backup.unlink()
        except Exception as e:
            logger.warning(f"Failed to cleanup old backups: {e}")

    def restore_from_backup(self, rel_path: str, backup_index: int = 0) -> bool:
        """Restore a file from backup.

        Args:
            rel_path: Relative path to file
            backup_index: 0 = most recent, 1 = second most recent, etc.

        Returns:
            True if restoration successful
        """
        with self._lock:
            try:
                prefix = rel_path.replace('/', '_').replace(os.sep, '_')
                backups = sorted(
                    [f for f in self._backup_dir.glob(f"{prefix}_*.bak")],
                    key=lambda p: p.stat().st_mtime,
                    reverse=True
                )

                if backup_index >= len(backups):
                    _safe_print(f"   [!] No backup at index {backup_index} for {rel_path}")
                    return False

                backup_file = backups[backup_index]
                full_path = self.base_path / rel_path

                shutil.copy2(backup_file, full_path)
                _safe_print(f"   [←] Restored {rel_path} from backup")
                return True
            except Exception as e:
                logger.exception(f"Failed to restore {rel_path}: {e}")
                return False

    def list_backups(self, rel_path: str) -> List[Dict[str, Any]]:
        """List available backups for a file."""
        prefix = rel_path.replace('/', '_').replace(os.sep, '_')
        backups = sorted(
            [f for f in self._backup_dir.glob(f"{prefix}_*.bak") if f.exists()],
            key=lambda p: p.stat().st_mtime,
            reverse=True
        )

        return [
            {
                "path": str(b),
                "timestamp": datetime.fromtimestamp(b.stat().st_mtime).isoformat(),
                "size": b.stat().st_size
            }
            for b in backups
        ]

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
        """Write file content (thread-safe, with transaction support)."""
        # Record operation for potential rollback
        self._record_operation(rel_path, "write")

        with self._lock:
            full_path = self.base_path / rel_path

            # Create backup if file exists
            if full_path.exists():
                self.create_backup(rel_path)

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
            try:
                with open(full_path, "r", encoding="utf-8") as handle:
                    return handle.read()
            except UnicodeDecodeError:
                # Try with different encoding
                try:
                    with open(full_path, "r", encoding="latin-1") as handle:
                        return handle.read()
                except Exception:
                    return None

    def _normalize_for_comparison(self, text: str) -> str:
        """Normalize text for comparison by handling line endings and trailing whitespace."""
        # Normalize line endings to Unix style
        text = text.replace('\r\n', '\n').replace('\r', '\n')
        # Strip trailing whitespace from each line but preserve indentation
        lines = [line.rstrip() for line in text.split('\n')]
        return '\n'.join(lines)

    def _dedent_code(self, code: str) -> str:
        """Remove common leading indentation from all lines."""
        lines = code.split('\n')
        non_empty_lines = [l for l in lines if l.strip()]
        if not non_empty_lines:
            return code

        # Find minimum indentation
        min_indent = float('inf')
        for line in non_empty_lines:
            stripped = line.lstrip()
            if stripped:
                indent = len(line) - len(stripped)
                min_indent = min(min_indent, indent)

        if min_indent == float('inf') or min_indent == 0:
            return code

        # Remove common indentation
        result_lines = []
        for line in lines:
            if line.strip():
                result_lines.append(line[min_indent:] if len(line) >= min_indent else line)
            else:
                result_lines.append('')

        return '\n'.join(result_lines)

    def _find_best_match_position(self, content: str, old_code: str) -> tuple:
        """Find the best matching position for old_code in content.

        Uses a multi-stage matching strategy:
        1. Exact match (fastest)
        2. Normalized match (whitespace-insensitive)
        3. Dedented match (handles indent differences)
        4. Line-by-line stripped match
        5. Fuzzy match with sliding window

        Returns (start_idx, end_idx, match_quality) where:
        - start_idx, end_idx are line indices (or -1 if no match)
        - match_quality is 'exact', 'normalized', 'dedented', 'stripped', 'fuzzy', or None
        """
        content_lines = content.split('\n')
        old_lines = old_code.split('\n')

        # Clean old_lines - remove empty lines at start/end
        while old_lines and not old_lines[0].strip():
            old_lines = old_lines[1:]
        while old_lines and not old_lines[-1].strip():
            old_lines = old_lines[:-1]

        if not old_lines:
            return (-1, -1, None)

        num_old_lines = len(old_lines)

        # Stage 1: Normalized exact match
        norm_content = self._normalize_for_comparison(content)
        norm_old = self._normalize_for_comparison('\n'.join(old_lines))

        if norm_old in norm_content:
            # Find line-based position
            for start_idx in range(len(content_lines) - num_old_lines + 1):
                chunk = '\n'.join(content_lines[start_idx:start_idx + num_old_lines])
                if self._normalize_for_comparison(chunk) == norm_old:
                    return (start_idx, start_idx + num_old_lines, 'normalized')

        # Stage 2: Dedented match - normalize both sides by removing common indent
        dedented_old = self._dedent_code('\n'.join(old_lines))
        dedented_old_lines = dedented_old.split('\n')

        for start_idx in range(len(content_lines) - num_old_lines + 1):
            chunk = '\n'.join(content_lines[start_idx:start_idx + num_old_lines])
            dedented_chunk = self._dedent_code(chunk)
            if self._normalize_for_comparison(dedented_chunk) == self._normalize_for_comparison(dedented_old):
                return (start_idx, start_idx + num_old_lines, 'dedented')

        # Stage 3: Line-by-line stripped match (ignoring all leading/trailing whitespace)
        stripped_old = [l.strip() for l in old_lines]

        for start_idx in range(len(content_lines) - num_old_lines + 1):
            potential_lines = content_lines[start_idx:start_idx + num_old_lines]
            stripped_potential = [l.strip() for l in potential_lines]

            if stripped_old == stripped_potential:
                return (start_idx, start_idx + num_old_lines, 'stripped')

        # Stage 4: Fuzzy match with sliding window - lower threshold and better algorithm
        best_ratio = 0.0
        best_start = -1
        best_window_size = num_old_lines

        # Try different window sizes (in case old_code has extra/missing lines)
        for window_delta in [0, -1, 1, -2, 2]:
            window_size = num_old_lines + window_delta
            if window_size < 1 or window_size > len(content_lines):
                continue

            # Normalize old_code for fuzzy comparison
            old_joined = ''.join(stripped_old)
            old_normalized = old_joined.replace(' ', '').replace('\t', '').lower()

            if len(old_normalized) < 3:
                continue

            for start_idx in range(len(content_lines) - window_size + 1):
                potential_lines = content_lines[start_idx:start_idx + window_size]
                pot_joined = ''.join(l.strip() for l in potential_lines)
                pot_normalized = pot_joined.replace(' ', '').replace('\t', '').lower()

                if not pot_normalized:
                    continue

                # Calculate Levenshtein-like similarity
                ratio = self._sequence_similarity(old_normalized, pot_normalized)

                if ratio > best_ratio:
                    best_ratio = ratio
                    best_start = start_idx
                    best_window_size = window_size

        # Lower threshold to 65% for fuzzy matching to catch more cases
        if best_ratio >= 0.65 and best_start >= 0:
            return (best_start, best_start + best_window_size, 'fuzzy')

        return (-1, -1, None)

    def _sequence_similarity(self, s1: str, s2: str) -> float:
        """Calculate similarity between two strings using longest common subsequence."""
        if not s1 or not s2:
            return 0.0

        # Use a simplified LCS-based similarity
        len1, len2 = len(s1), len(s2)
        max_len = max(len1, len2)

        if max_len == 0:
            return 1.0

        # Quick check: if lengths are very different, low similarity
        if abs(len1 - len2) / max_len > 0.5:
            return 0.0

        # Count matching character pairs (order-sensitive)
        matches = 0
        j = 0
        for i in range(len1):
            # Try to find s1[i] in remaining s2
            while j < len2:
                if s1[i] == s2[j]:
                    matches += 1
                    j += 1
                    break
                j += 1

        # Calculate ratio based on matches vs max possible
        return (2.0 * matches) / (len1 + len2)

    def edit_file(self, rel_path: str, old_code: str, new_code: str) -> bool:
        """Edit file by replacing code section (thread-safe, multi-stage strategy).

        Stages:
        1. Exact match (after line ending normalization)
        2. Normalized match (whitespace-insensitive)
        3. Dedented match (handles indentation differences)
        4. Stripped match (line-by-line comparison)
        5. Fuzzy match (65%+ similarity)
        6. Fallback: Create .proposed_change file for manual review
        """
        # Record operation for potential rollback
        self._record_operation(rel_path, "edit")

        with self._lock:
            content = self.read_file(rel_path)
            if content is None:
                _safe_print(f"   [!] File not found: {rel_path}")
                return False

            # Clean up the old_code - remove leading/trailing empty lines
            old_code_clean = old_code.strip()
            if not old_code_clean:
                _safe_print(f"   [!] Empty old code provided for: {rel_path}")
                return False

            # Normalize line endings in both content and old_code
            norm_content = self._normalize_for_comparison(content)
            norm_old = self._normalize_for_comparison(old_code_clean)

            # Stage 1: Direct exact match (fastest path)
            if old_code_clean in content:
                updated = content.replace(old_code_clean, new_code, 1)
                full_path = self.base_path / rel_path
                with open(full_path, "w", encoding="utf-8") as handle:
                    handle.write(updated)

                self.edit_history.append(
                    {"file": rel_path, "timestamp": time.time(), "action": "replace", "stage": "exact"}
                )
                _safe_print(f"   [~] Edited (exact match): {rel_path}")
                return True

            # Stage 1b: Exact match with normalized line endings
            if norm_old in norm_content:
                updated = norm_content.replace(norm_old, self._normalize_for_comparison(new_code), 1)
                full_path = self.base_path / rel_path
                with open(full_path, "w", encoding="utf-8") as handle:
                    handle.write(updated)

                self.edit_history.append(
                    {"file": rel_path, "timestamp": time.time(), "action": "replace", "stage": "normalized_exact"}
                )
                _safe_print(f"   [~] Edited (normalized): {rel_path}")
                return True

            # Stage 2-5: Use improved position finder
            start_idx, end_idx, match_quality = self._find_best_match_position(content, old_code_clean)

            if match_quality and start_idx >= 0:
                content_lines = content.split('\n')

                # Clean up new_code - preserve internal structure but remove outer empty lines
                new_code_clean = new_code.strip('\n') if new_code else ""
                new_lines = new_code_clean.split('\n') if new_code_clean else []

                # For dedented/stripped matches, try to preserve the original indentation
                if match_quality in ('dedented', 'stripped', 'fuzzy'):
                    # Get the indentation of the first line being replaced
                    original_line = content_lines[start_idx] if start_idx < len(content_lines) else ""
                    original_indent = len(original_line) - len(original_line.lstrip())

                    # Get the indentation of new code's first line
                    first_new_line = new_lines[0] if new_lines else ""
                    new_indent = len(first_new_line) - len(first_new_line.lstrip())

                    # Adjust indentation if there's a difference
                    indent_diff = original_indent - new_indent
                    if indent_diff > 0:
                        # Add indentation to new lines
                        indent_str = ' ' * indent_diff
                        new_lines = [indent_str + line if line.strip() else line for line in new_lines]
                    elif indent_diff < 0:
                        # Remove indentation from new lines (carefully)
                        remove_indent = abs(indent_diff)
                        adjusted_lines = []
                        for line in new_lines:
                            if line.startswith(' ' * remove_indent):
                                adjusted_lines.append(line[remove_indent:])
                            else:
                                adjusted_lines.append(line)
                        new_lines = adjusted_lines

                updated_lines = content_lines[:start_idx] + new_lines + content_lines[end_idx:]
                updated = '\n'.join(updated_lines)

                full_path = self.base_path / rel_path
                with open(full_path, "w", encoding="utf-8") as handle:
                    handle.write(updated)

                self.edit_history.append(
                    {"file": rel_path, "timestamp": time.time(), "action": "replace", "stage": match_quality}
                )
                _safe_print(f"   [~] Edited ({match_quality}): {rel_path}")
                return True

            # Stage 6: Proposed change file (PR-style fallback)
            proposed_file = f"{rel_path}.proposed_change"
            proposed_path = self.base_path / proposed_file

            # Show debug info to help diagnose why match failed
            old_preview = old_code_clean[:300].replace('\n', '\\n')
            content_preview = content[:500].replace('\n', '\\n')

            proposal = f"""=== PROPOSED CHANGE ===
File: {rel_path}
Auto-applied: False (patch failed - manual review needed)

--- OLD CODE (expected to find) ---
{old_code_clean}

--- NEW CODE (to replace with) ---
{new_code}

--- DEBUG INFO ---
File size: {len(content)} bytes, {len(content.split(chr(10)))} lines
Old code size: {len(old_code_clean)} bytes, {len(old_code_clean.split(chr(10)))} lines
Stages attempted: exact, normalized, dedented, stripped, fuzzy (65% threshold)
All stages failed - content may have changed or old_code is incorrect.

--- FIRST 500 CHARS OF FILE ---
{content[:500]}

--- Tip ---
1. Copy the exact text from the file (including whitespace) into the 'old:' section
2. Make sure line endings match (use Unix-style \\n)
3. Check if the code has been modified since you last read it
"""

            with open(proposed_path, "w", encoding="utf-8") as handle:
                handle.write(proposal)

            self.edit_history.append(
                {"file": rel_path, "timestamp": time.time(), "action": "propose", "stage": "fallback"}
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


