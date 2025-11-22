"""Utility helpers for creating and editing files during orchestration."""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Dict, List, Optional


class FileManager:
    """Handles file system interactions for generated projects."""

    def __init__(self, base_path: str):
        self.base_path = Path(base_path)
        self.created_files: List[str] = []
        self.created_dirs: List[str] = []
        self.edit_history: List[dict] = []

        self.base_path.mkdir(parents=True, exist_ok=True)
        print(f"📁 Project folder: {self.base_path.absolute()}\n")

    def create_directory(self, rel_path: str) -> Path:
        full_path = self.base_path / rel_path
        full_path.mkdir(parents=True, exist_ok=True)
        if str(full_path) not in self.created_dirs:
            self.created_dirs.append(str(full_path))
            print(f"   📂 Created: {rel_path}/")
        return full_path

    def write_file(self, rel_path: str, content: str) -> Path:
        full_path = self.base_path / rel_path
        full_path.parent.mkdir(parents=True, exist_ok=True)
        with open(full_path, "w", encoding="utf-8") as handle:
            handle.write(content)

        if str(full_path) not in self.created_files:
            self.created_files.append(str(full_path))
        print(f"   📄 Written: {rel_path} ({len(content)} bytes)")
        return full_path

    def read_file(self, rel_path: str) -> Optional[str]:
        full_path = self.base_path / rel_path
        if not full_path.exists():
            return None
        with open(full_path, "r", encoding="utf-8") as handle:
            return handle.read()

    def edit_file(self, rel_path: str, old_code: str, new_code: str) -> bool:
        content = self.read_file(rel_path)
        if content is None:
            print(f"   ⚠️  File not found: {rel_path}")
            return False
        if old_code not in content:
            print(f"   ⚠️  Code section not found in {rel_path}")
            return False

        updated = content.replace(old_code, new_code, 1)
        full_path = self.base_path / rel_path
        with open(full_path, "w", encoding="utf-8") as handle:
            handle.write(updated)

        self.edit_history.append(
            {"file": rel_path, "timestamp": time.time(), "action": "replace"}
        )
        print(f"   ✏️  Edited: {rel_path}")
        return True

    def append_to_file(self, rel_path: str, content: str) -> bool:
        full_path = self.base_path / rel_path
        if not full_path.exists():
            self.write_file(rel_path, content)
            return True

        with open(full_path, "a", encoding="utf-8") as handle:
            handle.write("\n" + content)

        self.edit_history.append(
            {"file": rel_path, "timestamp": time.time(), "action": "append"}
        )
        print(f"   ➕ Appended to: {rel_path}")
        return True

    def insert_at_line(self, rel_path: str, line_num: int, content: str) -> bool:
        file_content = self.read_file(rel_path)
        if file_content is None:
            return False

        lines = file_content.split("\n")
        if line_num < 0 or line_num > len(lines):
            print(f"   ⚠️  Line number {line_num} out of range")
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
        print(f"   ➕ Inserted at line {line_num}: {rel_path}")
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


