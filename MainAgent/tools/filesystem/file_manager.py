"""
Filesystem Tool - Unified file system operations for agents.

This module provides a clean interface to the FileManager from orchestrator,
with additional utility functions for common file operations.

Features:
- File read/write/edit operations
- Directory traversal
- File search and pattern matching
- Safe file operations with validation
"""

from __future__ import annotations

import os
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Union

logger = logging.getLogger(__name__)

# Import the main FileManager from orchestrator
try:
    from orchestrator.file_manager import FileManager
except ImportError:
    try:
        from ...orchestrator.file_manager import FileManager
    except ImportError:
        FileManager = None
        logger.warning("FileManager not available from orchestrator")


class FileSystemTool:
    """Unified file system tool for agent operations.

    This wraps the orchestrator's FileManager and provides additional
    utility methods for common file operations.
    """

    def __init__(self, base_path: str = ".", file_manager: Optional[Any] = None):
        """Initialize the filesystem tool.

        Args:
            base_path: Base path for file operations
            file_manager: Optional existing FileManager instance
        """
        self.base_path = Path(base_path).resolve()

        if file_manager:
            self._fm = file_manager
        elif FileManager:
            self._fm = FileManager(str(self.base_path))
        else:
            self._fm = None
            logger.warning("Operating without FileManager - limited functionality")

    def read_file(self, path: str) -> Optional[str]:
        """Read file contents.

        Args:
            path: Relative or absolute path to file

        Returns:
            File contents or None if not found
        """
        try:
            if self._fm:
                # Use FileManager if available
                rel_path = self._to_relative(path)
                return self._fm.read_file(rel_path)
            else:
                # Direct read fallback
                full_path = self._to_absolute(path)
                if full_path.exists():
                    return full_path.read_text(encoding='utf-8')
                return None
        except Exception as exc:
            logger.error(f"Error reading file {path}: {exc}")
            return None

    def write_file(self, path: str, content: str) -> bool:
        """Write content to file.

        Args:
            path: Relative or absolute path to file
            content: Content to write

        Returns:
            True if successful
        """
        try:
            if self._fm:
                rel_path = self._to_relative(path)
                self._fm.write_file(rel_path, content)
                return True
            else:
                full_path = self._to_absolute(path)
                full_path.parent.mkdir(parents=True, exist_ok=True)
                full_path.write_text(content, encoding='utf-8')
                return True
        except Exception as exc:
            logger.error(f"Error writing file {path}: {exc}")
            return False

    def edit_file(self, path: str, old_content: str, new_content: str) -> bool:
        """Edit file by replacing content.

        Args:
            path: Path to file
            old_content: Content to find and replace
            new_content: Replacement content

        Returns:
            True if successful
        """
        try:
            if self._fm:
                rel_path = self._to_relative(path)
                return self._fm.edit_file(rel_path, old_content, new_content)
            else:
                # Fallback implementation
                full_path = self._to_absolute(path)
                if not full_path.exists():
                    return False

                content = full_path.read_text(encoding='utf-8')
                if old_content not in content:
                    return False

                new_file_content = content.replace(old_content, new_content, 1)
                full_path.write_text(new_file_content, encoding='utf-8')
                return True
        except Exception as exc:
            logger.error(f"Error editing file {path}: {exc}")
            return False

    def delete_file(self, path: str) -> bool:
        """Delete a file.

        Args:
            path: Path to file

        Returns:
            True if successful
        """
        try:
            full_path = self._to_absolute(path)
            if full_path.exists() and full_path.is_file():
                full_path.unlink()
                return True
            return False
        except Exception as exc:
            logger.error(f"Error deleting file {path}: {exc}")
            return False

    def create_directory(self, path: str) -> bool:
        """Create a directory.

        Args:
            path: Path to directory

        Returns:
            True if successful
        """
        try:
            if self._fm:
                rel_path = self._to_relative(path)
                self._fm.create_directory(rel_path)
                return True
            else:
                full_path = self._to_absolute(path)
                full_path.mkdir(parents=True, exist_ok=True)
                return True
        except Exception as exc:
            logger.error(f"Error creating directory {path}: {exc}")
            return False

    def list_files(self, pattern: str = "*", recursive: bool = True) -> List[str]:
        """List files matching pattern.

        Args:
            pattern: Glob pattern
            recursive: Whether to search recursively

        Returns:
            List of matching file paths
        """
        try:
            if self._fm:
                return self._fm.list_files(pattern)
            else:
                if recursive:
                    matches = list(self.base_path.rglob(pattern))
                else:
                    matches = list(self.base_path.glob(pattern))

                return [
                    str(m.relative_to(self.base_path))
                    for m in matches
                    if m.is_file()
                ]
        except Exception as exc:
            logger.error(f"Error listing files: {exc}")
            return []

    def file_exists(self, path: str) -> bool:
        """Check if file exists.

        Args:
            path: Path to file

        Returns:
            True if file exists
        """
        full_path = self._to_absolute(path)
        return full_path.exists() and full_path.is_file()

    def directory_exists(self, path: str) -> bool:
        """Check if directory exists.

        Args:
            path: Path to directory

        Returns:
            True if directory exists
        """
        full_path = self._to_absolute(path)
        return full_path.exists() and full_path.is_dir()

    def get_file_info(self, path: str) -> Optional[Dict[str, Any]]:
        """Get file information.

        Args:
            path: Path to file

        Returns:
            Dict with file info or None
        """
        try:
            full_path = self._to_absolute(path)
            if not full_path.exists():
                return None

            stat = full_path.stat()
            return {
                "path": str(full_path),
                "relative_path": self._to_relative(path),
                "name": full_path.name,
                "extension": full_path.suffix,
                "size": stat.st_size,
                "is_file": full_path.is_file(),
                "is_dir": full_path.is_dir(),
                "modified": stat.st_mtime,
            }
        except Exception as exc:
            logger.error(f"Error getting file info: {exc}")
            return None

    def get_project_structure(self) -> str:
        """Get project structure as tree.

        Returns:
            Tree representation of project structure
        """
        if self._fm:
            return self._fm.get_project_structure_tree()
        else:
            return self._build_tree_structure()

    def search_in_files(self, pattern: str, file_pattern: str = "*.py") -> List[Dict[str, Any]]:
        """Search for pattern in files.

        Args:
            pattern: Text pattern to search for
            file_pattern: Glob pattern for files to search

        Returns:
            List of matches with file, line, and content
        """
        import re

        matches = []
        files = self.list_files(file_pattern)

        for file_path in files:
            content = self.read_file(file_path)
            if content and pattern in content:
                lines = content.split('\n')
                for i, line in enumerate(lines, 1):
                    if pattern in line:
                        matches.append({
                            "file": file_path,
                            "line": i,
                            "content": line.strip(),
                        })

        return matches

    def copy_file(self, source: str, destination: str) -> bool:
        """Copy a file.

        Args:
            source: Source path
            destination: Destination path

        Returns:
            True if successful
        """
        try:
            import shutil
            src_path = self._to_absolute(source)
            dst_path = self._to_absolute(destination)

            if not src_path.exists():
                return False

            dst_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_path, dst_path)
            return True
        except Exception as exc:
            logger.error(f"Error copying file: {exc}")
            return False

    def move_file(self, source: str, destination: str) -> bool:
        """Move a file.

        Args:
            source: Source path
            destination: Destination path

        Returns:
            True if successful
        """
        try:
            import shutil
            src_path = self._to_absolute(source)
            dst_path = self._to_absolute(destination)

            if not src_path.exists():
                return False

            dst_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src_path), str(dst_path))
            return True
        except Exception as exc:
            logger.error(f"Error moving file: {exc}")
            return False

    def _to_absolute(self, path: str) -> Path:
        """Convert path to absolute."""
        p = Path(path)
        if p.is_absolute():
            return p
        return self.base_path / p

    def _to_relative(self, path: str) -> str:
        """Convert path to relative."""
        p = Path(path)
        if p.is_absolute():
            try:
                return str(p.relative_to(self.base_path))
            except ValueError:
                return str(p)
        return str(p)

    def _build_tree_structure(self) -> str:
        """Build tree structure manually."""
        lines = [f"Project: {self.base_path.name}/"]

        def _add_entries(directory: Path, prefix: str = ""):
            try:
                entries = sorted(directory.iterdir())
                dirs = [e for e in entries if e.is_dir() and not e.name.startswith('.')]
                files = [e for e in entries if e.is_file() and not e.name.startswith('.')]

                for i, f in enumerate(files):
                    connector = "--- " if i == len(files) - 1 and not dirs else "|-- "
                    lines.append(f"{prefix}{connector}{f.name}")

                for i, d in enumerate(dirs):
                    connector = "+-- " if i == len(dirs) - 1 else "|-- "
                    lines.append(f"{prefix}{connector}{d.name}/")
                    _add_entries(d, prefix + ("|   " if i < len(dirs) - 1 else "    "))
            except PermissionError:
                pass

        _add_entries(self.base_path)
        return "\n".join(lines)


# Convenience functions for simple operations

def read_file(path: str) -> Optional[str]:
    """Read file contents."""
    try:
        return Path(path).read_text(encoding='utf-8')
    except Exception:
        return None


def write_file(path: str, content: str) -> bool:
    """Write content to file."""
    try:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding='utf-8')
        return True
    except Exception:
        return False


def delete_file(path: str) -> bool:
    """Delete a file."""
    try:
        p = Path(path)
        if p.exists() and p.is_file():
            p.unlink()
            return True
        return False
    except Exception:
        return False


def list_files(directory: str, pattern: str = "*") -> List[str]:
    """List files in directory."""
    try:
        p = Path(directory)
        return [str(f) for f in p.rglob(pattern) if f.is_file()]
    except Exception:
        return []


def file_exists(path: str) -> bool:
    """Check if file exists."""
    return Path(path).is_file()


def create_directory(path: str) -> bool:
    """Create directory."""
    try:
        Path(path).mkdir(parents=True, exist_ok=True)
        return True
    except Exception:
        return False
