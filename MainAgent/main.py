"""Entry point for running the MainAgent orchestrators."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

# Add the workspace root to Python path so imports work
workspace_root = Path(__file__).parent.parent
if str(workspace_root) not in sys.path:
    sys.path.insert(0, str(workspace_root))

from MainAgent.orchestrator import dynamic_orchestrator, planning_orchestrator


def _generate_project_name(prompt: str) -> str:
    """Generate a project folder name from the prompt."""
    # Extract key words from prompt
    words = re.findall(r'\b[a-zA-Z]{3,}\b', prompt.lower())
    # Take first 2-3 meaningful words
    if words:
        # Filter out common words
        common_words = {'the', 'and', 'for', 'with', 'using', 'build', 'create', 'make', 'simple'}
        meaningful = [w for w in words[:5] if w not in common_words]
        if meaningful:
            name = '_'.join(meaningful[:3])
            # Sanitize for filesystem
            name = re.sub(r'[^\w\-_]', '', name)
            return name[:30]  # Limit length
    return "project"


def _list_existing_projects() -> list[Path]:
    """List all existing projects in the output/projects directory."""
    mainagent_dir = Path(__file__).parent
    projects_dir = mainagent_dir / "output" / "projects"
    
    if not projects_dir.exists():
        return []
    
    # Return all directories in projects folder
    return [p for p in projects_dir.iterdir() if p.is_dir()]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run MainAgent orchestrators.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create a new project
  python -m main --prompt "build a todo app"
  
  # Work on an existing project
  python -m main --project output/projects/my_project --prompt "add a new feature"
  
  # Specify custom output path
  python -m main --output ./my_custom_path --prompt "build something"
        """,
    )
    parser.add_argument(
        "--mode",
        choices=["dynamic", "planning"],
        default="dynamic",
        help="Which orchestrator workflow to run.",
    )
    parser.add_argument(
        "--prompt",
        required=False,
        default="Build a personal portfolio website using React, Node.js, and MongoDB.",
        help="User request or problem statement for the orchestrator.",
    )
    parser.add_argument(
        "--project",
        default=None,
        help="Path to an EXISTING project folder to update/edit (relative to MainAgent dir or absolute). Use --list-projects to see available projects.",
    )
    parser.add_argument(
        "--list-projects",
        action="store_true",
        help="List all existing projects and exit.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output directory for NEW projects (ignored if --project is specified). Defaults to output/projects/<project_name>",
    )
    args = parser.parse_args()

    # Handle list projects request
    if args.list_projects:
        projects = _list_existing_projects()
        if projects:
            print("\n📁 Existing Projects:\n")
            for project in sorted(projects):
                rel_path = project.relative_to(Path(__file__).parent)
                print(f"  • {rel_path}")
            print(f"\n💡 Use --project <path> to work on an existing project")
            print(f"   Example: --project output/projects/{projects[0].name}\n")
        else:
            print("\n📁 No existing projects found in output/projects/\n")
        return

    # Determine output path
    if args.mode == "dynamic":
        if args.project:
            # Use existing project path
            project_path = Path(args.project)
            if not project_path.is_absolute():
                # Make relative to MainAgent directory
                mainagent_dir = Path(__file__).parent
                project_path = mainagent_dir / project_path
            
            resolved_path = project_path.resolve()
            
            # Allow directory-level paths - will scan all subfolders and files
            if resolved_path.is_dir():
                args.output = str(resolved_path)
                # Count files in directory and subdirectories
                try:
                    all_files = list(resolved_path.rglob("*"))
                    file_count = sum(1 for f in all_files if f.is_file())
                    dir_count = sum(1 for f in all_files if f.is_dir())
                    print(f"📂 Working on directory: {args.output}")
                    print(f"   Found {file_count} file(s) in {dir_count} subdirectory(ies)\n")
                except Exception:
                    print(f"📂 Working on directory: {args.output}\n")
            elif not resolved_path.exists():
                print(f"\n⚠️  Warning: Path does not exist: {resolved_path}")
                print(f"   Creating new project at this location...\n")
                args.output = str(resolved_path)
            else:
                args.output = str(resolved_path)
                print(f"📂 Working on project: {args.output}\n")
        elif args.output is None:
            # Create new project with auto-generated name
            mainagent_dir = Path(__file__).parent
            project_name = _generate_project_name(args.prompt)
            args.output = str(mainagent_dir / "output" / "projects" / project_name)
            print(f"📂 Creating new project: {args.output}\n")

    if args.mode == "dynamic":
        dynamic_orchestrator.orchestrator(args.prompt, output_folder=args.output)
    else:
        planning_orchestrator.orchestrator(args.prompt)


if __name__ == "__main__":
    main()
