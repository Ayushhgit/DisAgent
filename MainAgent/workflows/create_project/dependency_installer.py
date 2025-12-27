"""
Dependency Installer - Install project dependencies.

Supports multiple package managers:
- pip (Python)
- poetry (Python)
- pipenv (Python)
- npm (Node.js)
- yarn (Node.js)
- pnpm (Node.js)
"""

from __future__ import annotations

import os
import json
import subprocess
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class PackageManager(Enum):
    """Supported package managers."""
    PIP = "pip"
    POETRY = "poetry"
    PIPENV = "pipenv"
    CONDA = "conda"
    NPM = "npm"
    YARN = "yarn"
    PNPM = "pnpm"


@dataclass
class InstallResult:
    """Result of dependency installation."""
    success: bool
    package_manager: PackageManager
    packages_installed: List[str] = field(default_factory=list)
    packages_failed: List[str] = field(default_factory=list)
    output: str = ""
    error: Optional[str] = None


def install_deps(reqs_path: str) -> bool:
    """Install dependencies (legacy interface).

    Args:
        reqs_path: Path to requirements file

    Returns:
        True if successful
    """
    result = install_dependencies(reqs_path)
    return result.success


def install_dependencies(
    project_path: str,
    package_manager: Optional[PackageManager] = None,
    dev: bool = False,
) -> InstallResult:
    """Install project dependencies.

    Args:
        project_path: Path to project directory
        package_manager: Specific package manager (auto-detected if None)
        dev: Install dev dependencies too

    Returns:
        InstallResult with installation status
    """
    path = Path(project_path).resolve()

    if not path.exists():
        return InstallResult(
            success=False,
            package_manager=PackageManager.PIP,
            error=f"Project path does not exist: {path}",
        )

    # Detect package manager if not specified
    if package_manager is None:
        package_manager = detect_package_manager(path)

    # Install using appropriate manager
    if package_manager in (PackageManager.PIP, PackageManager.PIPENV,
                           PackageManager.POETRY, PackageManager.CONDA):
        return _install_python_deps(path, package_manager, dev)
    else:
        return _install_node_deps(path, package_manager, dev)


def detect_package_manager(path: Path) -> PackageManager:
    """Detect the package manager for a project.

    Args:
        path: Project directory path

    Returns:
        Detected PackageManager
    """
    # Check for Python package managers
    if (path / "pyproject.toml").exists():
        pyproject = (path / "pyproject.toml").read_text()
        if "[tool.poetry]" in pyproject:
            return PackageManager.POETRY
        # Check for pipenv in pyproject
        if "[tool.pipenv]" in pyproject:
            return PackageManager.PIPENV

    if (path / "Pipfile").exists():
        return PackageManager.PIPENV

    if (path / "environment.yml").exists() or (path / "environment.yaml").exists():
        return PackageManager.CONDA

    if (path / "requirements.txt").exists():
        return PackageManager.PIP

    # Check for Node.js package managers
    if (path / "pnpm-lock.yaml").exists():
        return PackageManager.PNPM

    if (path / "yarn.lock").exists():
        return PackageManager.YARN

    if (path / "package-lock.json").exists() or (path / "package.json").exists():
        return PackageManager.NPM

    # Default to pip for Python projects
    return PackageManager.PIP


def _install_python_deps(
    path: Path,
    manager: PackageManager,
    dev: bool,
) -> InstallResult:
    """Install Python dependencies."""
    packages_installed = []
    output_lines = []

    try:
        if manager == PackageManager.PIP:
            return _install_pip(path, dev)
        elif manager == PackageManager.POETRY:
            return _install_poetry(path, dev)
        elif manager == PackageManager.PIPENV:
            return _install_pipenv(path, dev)
        elif manager == PackageManager.CONDA:
            return _install_conda(path, dev)
        else:
            return InstallResult(
                success=False,
                package_manager=manager,
                error=f"Unsupported Python package manager: {manager}",
            )
    except Exception as exc:
        logger.exception(f"Failed to install dependencies: {exc}")
        return InstallResult(
            success=False,
            package_manager=manager,
            error=str(exc),
        )


def _install_pip(path: Path, dev: bool) -> InstallResult:
    """Install dependencies using pip."""
    req_files = []

    # Find requirements files
    for name in ["requirements.txt", "requirements-dev.txt", "requirements_dev.txt"]:
        req_file = path / name
        if req_file.exists():
            if "dev" in name:
                if dev:
                    req_files.append(req_file)
            else:
                req_files.append(req_file)

    # Check pyproject.toml for dependencies
    pyproject = path / "pyproject.toml"
    if pyproject.exists() and not req_files:
        # Install from pyproject.toml
        cmd = ["pip", "install", "-e", "."]
        if dev:
            cmd = ["pip", "install", "-e", ".[dev]"]

        result = subprocess.run(
            cmd,
            cwd=str(path),
            capture_output=True,
            text=True,
        )

        return InstallResult(
            success=result.returncode == 0,
            package_manager=PackageManager.PIP,
            output=result.stdout,
            error=result.stderr if result.returncode != 0 else None,
        )

    if not req_files:
        return InstallResult(
            success=True,
            package_manager=PackageManager.PIP,
            output="No requirements files found",
        )

    # Install from requirements files
    all_output = []
    packages = []
    failed = []

    for req_file in req_files:
        # Parse packages from requirements file
        with open(req_file) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and not line.startswith('-'):
                    pkg_name = line.split('==')[0].split('>=')[0].split('<=')[0].strip()
                    packages.append(pkg_name)

        cmd = ["pip", "install", "-r", str(req_file)]
        result = subprocess.run(
            cmd,
            cwd=str(path),
            capture_output=True,
            text=True,
        )

        all_output.append(result.stdout)
        if result.returncode != 0:
            all_output.append(result.stderr)
            # Try to identify failed packages
            for pkg in packages:
                if pkg.lower() in result.stderr.lower():
                    failed.append(pkg)

    return InstallResult(
        success=len(failed) == 0,
        package_manager=PackageManager.PIP,
        packages_installed=[p for p in packages if p not in failed],
        packages_failed=failed,
        output='\n'.join(all_output),
    )


def _install_poetry(path: Path, dev: bool) -> InstallResult:
    """Install dependencies using Poetry."""
    cmd = ["poetry", "install"]
    if not dev:
        cmd.append("--no-dev")

    try:
        result = subprocess.run(
            cmd,
            cwd=str(path),
            capture_output=True,
            text=True,
        )

        return InstallResult(
            success=result.returncode == 0,
            package_manager=PackageManager.POETRY,
            output=result.stdout,
            error=result.stderr if result.returncode != 0 else None,
        )
    except FileNotFoundError:
        return InstallResult(
            success=False,
            package_manager=PackageManager.POETRY,
            error="Poetry is not installed",
        )


def _install_pipenv(path: Path, dev: bool) -> InstallResult:
    """Install dependencies using Pipenv."""
    cmd = ["pipenv", "install"]
    if dev:
        cmd.append("--dev")

    try:
        result = subprocess.run(
            cmd,
            cwd=str(path),
            capture_output=True,
            text=True,
        )

        return InstallResult(
            success=result.returncode == 0,
            package_manager=PackageManager.PIPENV,
            output=result.stdout,
            error=result.stderr if result.returncode != 0 else None,
        )
    except FileNotFoundError:
        return InstallResult(
            success=False,
            package_manager=PackageManager.PIPENV,
            error="Pipenv is not installed",
        )


def _install_conda(path: Path, dev: bool) -> InstallResult:
    """Install dependencies using Conda."""
    env_file = path / "environment.yml"
    if not env_file.exists():
        env_file = path / "environment.yaml"

    if not env_file.exists():
        return InstallResult(
            success=False,
            package_manager=PackageManager.CONDA,
            error="No environment.yml found",
        )

    cmd = ["conda", "env", "update", "-f", str(env_file)]

    try:
        result = subprocess.run(
            cmd,
            cwd=str(path),
            capture_output=True,
            text=True,
        )

        return InstallResult(
            success=result.returncode == 0,
            package_manager=PackageManager.CONDA,
            output=result.stdout,
            error=result.stderr if result.returncode != 0 else None,
        )
    except FileNotFoundError:
        return InstallResult(
            success=False,
            package_manager=PackageManager.CONDA,
            error="Conda is not installed",
        )


def _install_node_deps(
    path: Path,
    manager: PackageManager,
    dev: bool,
) -> InstallResult:
    """Install Node.js dependencies."""
    try:
        if manager == PackageManager.NPM:
            return _install_npm(path, dev)
        elif manager == PackageManager.YARN:
            return _install_yarn(path, dev)
        elif manager == PackageManager.PNPM:
            return _install_pnpm(path, dev)
        else:
            return InstallResult(
                success=False,
                package_manager=manager,
                error=f"Unsupported Node package manager: {manager}",
            )
    except Exception as exc:
        logger.exception(f"Failed to install dependencies: {exc}")
        return InstallResult(
            success=False,
            package_manager=manager,
            error=str(exc),
        )


def _install_npm(path: Path, dev: bool) -> InstallResult:
    """Install dependencies using npm."""
    cmd = ["npm", "install"]
    if not dev:
        cmd.append("--production")

    try:
        result = subprocess.run(
            cmd,
            cwd=str(path),
            capture_output=True,
            text=True,
        )

        # Parse installed packages from package.json
        packages = []
        pkg_json = path / "package.json"
        if pkg_json.exists():
            try:
                with open(pkg_json) as f:
                    pkg = json.load(f)
                    packages.extend(pkg.get("dependencies", {}).keys())
                    if dev:
                        packages.extend(pkg.get("devDependencies", {}).keys())
            except json.JSONDecodeError:
                pass

        return InstallResult(
            success=result.returncode == 0,
            package_manager=PackageManager.NPM,
            packages_installed=packages if result.returncode == 0 else [],
            output=result.stdout,
            error=result.stderr if result.returncode != 0 else None,
        )
    except FileNotFoundError:
        return InstallResult(
            success=False,
            package_manager=PackageManager.NPM,
            error="npm is not installed",
        )


def _install_yarn(path: Path, dev: bool) -> InstallResult:
    """Install dependencies using Yarn."""
    cmd = ["yarn", "install"]
    if not dev:
        cmd.append("--production")

    try:
        result = subprocess.run(
            cmd,
            cwd=str(path),
            capture_output=True,
            text=True,
        )

        return InstallResult(
            success=result.returncode == 0,
            package_manager=PackageManager.YARN,
            output=result.stdout,
            error=result.stderr if result.returncode != 0 else None,
        )
    except FileNotFoundError:
        return InstallResult(
            success=False,
            package_manager=PackageManager.YARN,
            error="Yarn is not installed",
        )


def _install_pnpm(path: Path, dev: bool) -> InstallResult:
    """Install dependencies using pnpm."""
    cmd = ["pnpm", "install"]
    if not dev:
        cmd.append("--prod")

    try:
        result = subprocess.run(
            cmd,
            cwd=str(path),
            capture_output=True,
            text=True,
        )

        return InstallResult(
            success=result.returncode == 0,
            package_manager=PackageManager.PNPM,
            output=result.stdout,
            error=result.stderr if result.returncode != 0 else None,
        )
    except FileNotFoundError:
        return InstallResult(
            success=False,
            package_manager=PackageManager.PNPM,
            error="pnpm is not installed",
        )


def install_package(
    package: str,
    path: str = ".",
    manager: Optional[PackageManager] = None,
    dev: bool = False,
    version: Optional[str] = None,
) -> InstallResult:
    """Install a single package.

    Args:
        package: Package name
        path: Project path
        manager: Package manager (auto-detect if None)
        dev: Install as dev dependency
        version: Specific version to install

    Returns:
        InstallResult
    """
    project_path = Path(path).resolve()

    if manager is None:
        manager = detect_package_manager(project_path)

    # Build package specification
    pkg_spec = package
    if version:
        if manager in (PackageManager.PIP, PackageManager.POETRY, PackageManager.PIPENV):
            pkg_spec = f"{package}=={version}"
        else:
            pkg_spec = f"{package}@{version}"

    # Build command based on package manager
    cmd = _build_install_command(manager, pkg_spec, dev)

    try:
        result = subprocess.run(
            cmd,
            cwd=str(project_path),
            capture_output=True,
            text=True,
        )

        return InstallResult(
            success=result.returncode == 0,
            package_manager=manager,
            packages_installed=[package] if result.returncode == 0 else [],
            packages_failed=[package] if result.returncode != 0 else [],
            output=result.stdout,
            error=result.stderr if result.returncode != 0 else None,
        )
    except FileNotFoundError:
        return InstallResult(
            success=False,
            package_manager=manager,
            packages_failed=[package],
            error=f"{manager.value} is not installed",
        )


def _build_install_command(
    manager: PackageManager,
    package: str,
    dev: bool,
) -> List[str]:
    """Build the install command for a package manager."""
    if manager == PackageManager.PIP:
        return ["pip", "install", package]

    elif manager == PackageManager.POETRY:
        cmd = ["poetry", "add", package]
        if dev:
            cmd.append("--dev")
        return cmd

    elif manager == PackageManager.PIPENV:
        cmd = ["pipenv", "install", package]
        if dev:
            cmd.append("--dev")
        return cmd

    elif manager == PackageManager.NPM:
        cmd = ["npm", "install", package]
        if dev:
            cmd.append("--save-dev")
        return cmd

    elif manager == PackageManager.YARN:
        cmd = ["yarn", "add", package]
        if dev:
            cmd.append("--dev")
        return cmd

    elif manager == PackageManager.PNPM:
        cmd = ["pnpm", "add", package]
        if dev:
            cmd.append("--save-dev")
        return cmd

    return []


def uninstall_package(
    package: str,
    path: str = ".",
    manager: Optional[PackageManager] = None,
) -> InstallResult:
    """Uninstall a package.

    Args:
        package: Package name
        path: Project path
        manager: Package manager (auto-detect if None)

    Returns:
        InstallResult
    """
    project_path = Path(path).resolve()

    if manager is None:
        manager = detect_package_manager(project_path)

    # Build command
    cmd_map = {
        PackageManager.PIP: ["pip", "uninstall", "-y", package],
        PackageManager.POETRY: ["poetry", "remove", package],
        PackageManager.PIPENV: ["pipenv", "uninstall", package],
        PackageManager.NPM: ["npm", "uninstall", package],
        PackageManager.YARN: ["yarn", "remove", package],
        PackageManager.PNPM: ["pnpm", "remove", package],
    }

    cmd = cmd_map.get(manager, [])
    if not cmd:
        return InstallResult(
            success=False,
            package_manager=manager,
            error=f"Unsupported package manager: {manager}",
        )

    try:
        result = subprocess.run(
            cmd,
            cwd=str(project_path),
            capture_output=True,
            text=True,
        )

        return InstallResult(
            success=result.returncode == 0,
            package_manager=manager,
            output=result.stdout,
            error=result.stderr if result.returncode != 0 else None,
        )
    except FileNotFoundError:
        return InstallResult(
            success=False,
            package_manager=manager,
            error=f"{manager.value} is not installed",
        )


def list_installed_packages(
    path: str = ".",
    manager: Optional[PackageManager] = None,
) -> Dict[str, str]:
    """List installed packages and their versions.

    Args:
        path: Project path
        manager: Package manager (auto-detect if None)

    Returns:
        Dictionary of package names to versions
    """
    project_path = Path(path).resolve()

    if manager is None:
        manager = detect_package_manager(project_path)

    packages = {}

    try:
        if manager == PackageManager.PIP:
            result = subprocess.run(
                ["pip", "list", "--format=json"],
                cwd=str(project_path),
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                for pkg in json.loads(result.stdout):
                    packages[pkg["name"]] = pkg["version"]

        elif manager == PackageManager.NPM:
            result = subprocess.run(
                ["npm", "list", "--json", "--depth=0"],
                cwd=str(project_path),
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                data = json.loads(result.stdout)
                for name, info in data.get("dependencies", {}).items():
                    packages[name] = info.get("version", "")

        elif manager == PackageManager.YARN:
            result = subprocess.run(
                ["yarn", "list", "--json", "--depth=0"],
                cwd=str(project_path),
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                for line in result.stdout.strip().split('\n'):
                    try:
                        data = json.loads(line)
                        if data.get("type") == "tree":
                            for item in data.get("data", {}).get("trees", []):
                                name_ver = item.get("name", "")
                                if "@" in name_ver:
                                    parts = name_ver.rsplit("@", 1)
                                    packages[parts[0]] = parts[1] if len(parts) > 1 else ""
                    except json.JSONDecodeError:
                        continue

    except (FileNotFoundError, json.JSONDecodeError):
        pass

    return packages


def check_outdated_packages(
    path: str = ".",
    manager: Optional[PackageManager] = None,
) -> List[Dict[str, str]]:
    """Check for outdated packages.

    Args:
        path: Project path
        manager: Package manager (auto-detect if None)

    Returns:
        List of outdated package info dicts
    """
    project_path = Path(path).resolve()

    if manager is None:
        manager = detect_package_manager(project_path)

    outdated = []

    try:
        if manager == PackageManager.PIP:
            result = subprocess.run(
                ["pip", "list", "--outdated", "--format=json"],
                cwd=str(project_path),
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                for pkg in json.loads(result.stdout):
                    outdated.append({
                        "name": pkg["name"],
                        "current": pkg["version"],
                        "latest": pkg["latest_version"],
                    })

        elif manager == PackageManager.NPM:
            result = subprocess.run(
                ["npm", "outdated", "--json"],
                cwd=str(project_path),
                capture_output=True,
                text=True,
            )
            # npm outdated returns non-zero if there are outdated packages
            try:
                data = json.loads(result.stdout) if result.stdout else {}
                for name, info in data.items():
                    outdated.append({
                        "name": name,
                        "current": info.get("current", ""),
                        "latest": info.get("latest", ""),
                    })
            except json.JSONDecodeError:
                pass

    except FileNotFoundError:
        pass

    return outdated
