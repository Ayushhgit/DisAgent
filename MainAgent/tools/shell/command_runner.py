# ============================================================================
# FILE: command_runner.py
# Safe command execution with shell injection protection
# ============================================================================

import subprocess
import shlex
import re
import os
from typing import Tuple, List, Optional, Set
from pathlib import Path

# Import custom exceptions
try:
    from ...core.exceptions import ShellInjectionError, UnsafeOperationError
except ImportError:
    class ShellInjectionError(Exception):
        pass
    class UnsafeOperationError(Exception):
        pass


# Dangerous patterns that indicate shell injection attempts
DANGEROUS_PATTERNS = [
    r';\s*rm\s',           # rm after semicolon
    r';\s*dd\s',           # dd after semicolon
    r';\s*mkfs',           # filesystem formatting
    r';\s*chmod\s+777',    # dangerous permissions
    r'&&\s*rm\s',          # rm after &&
    r'\|\s*rm\s',          # rm after pipe
    r'`.*`',               # backtick command substitution
    r'\$\(.*\)',           # $() command substitution
    r'>\s*/dev/sd',        # writing to block devices
    r'>\s*/etc/',          # writing to /etc
    r'rm\s+-rf\s+/',       # rm -rf /
    r'rm\s+-rf\s+\*',      # rm -rf *
    r':\(\)\s*\{',         # fork bomb pattern
    r'/dev/null\s*>',      # redirecting /dev/null
    r'>\s*/dev/null\s*2>&1\s*&',  # background with null redirect (hiding)
]

# Allowed commands (whitelist approach for safety)
ALLOWED_COMMANDS: Set[str] = {
    # Build tools
    'npm', 'npx', 'yarn', 'pnpm',
    'pip', 'pip3', 'python', 'python3',
    'node', 'deno', 'bun',
    'cargo', 'rustc',
    'go', 'make', 'cmake',
    'mvn', 'gradle',

    # Version control
    'git',

    # File operations (safe subset)
    'ls', 'cat', 'head', 'tail', 'wc',
    'find', 'grep', 'awk', 'sed',
    'mkdir', 'touch', 'cp', 'mv',

    # Testing
    'pytest', 'jest', 'mocha', 'vitest',
    'cargo test', 'go test',

    # Linting/formatting
    'eslint', 'prettier', 'black', 'flake8', 'mypy',
    'rustfmt', 'clippy',

    # Other safe tools
    'echo', 'pwd', 'which', 'env',
    'curl', 'wget',  # network tools
    'tar', 'unzip', 'gzip',  # archive tools
}

# Dangerous commands that should never be allowed
BLOCKED_COMMANDS: Set[str] = {
    'rm', 'rmdir',  # deletion (use safe_delete instead)
    'dd',           # disk operations
    'mkfs',         # filesystem formatting
    'fdisk',        # disk partitioning
    'mount', 'umount',
    'sudo', 'su',   # privilege escalation
    'chmod', 'chown',  # permission changes
    'kill', 'killall', 'pkill',
    'shutdown', 'reboot', 'halt',
    'systemctl', 'service',
    'iptables', 'ufw',
    'passwd', 'useradd', 'userdel',
    'crontab',
    'nc', 'netcat',  # network tools that can be abused
    'nmap',
    'eval', 'exec',
}


def _is_dangerous_pattern(command: str) -> Optional[str]:
    """Check if command contains dangerous patterns.

    Returns the matched pattern if dangerous, None otherwise.
    """
    command_lower = command.lower()
    for pattern in DANGEROUS_PATTERNS:
        if re.search(pattern, command_lower):
            return pattern
    return None


def _extract_base_command(command: str) -> str:
    """Extract the base command from a command string."""
    # Handle command with path
    parts = shlex.split(command)
    if not parts:
        return ""

    base = parts[0]
    # Get just the command name without path
    return os.path.basename(base)


def _is_command_allowed(command: str) -> Tuple[bool, str]:
    """Check if a command is allowed.

    Returns (is_allowed, reason).
    """
    base_cmd = _extract_base_command(command)

    if not base_cmd:
        return False, "Empty command"

    # Check blocked list first
    if base_cmd in BLOCKED_COMMANDS:
        return False, f"Command '{base_cmd}' is blocked for security reasons"

    # Check if in allowed list
    if base_cmd not in ALLOWED_COMMANDS:
        return False, f"Command '{base_cmd}' is not in the allowed list"

    return True, ""


def sanitize_command(command: str) -> str:
    """Sanitize a command string by removing dangerous characters.

    Note: This is a fallback. Prefer using run_command_safe with args list.
    """
    # Remove null bytes
    command = command.replace('\x00', '')

    # Remove or escape dangerous shell metacharacters
    # Keep basic ones needed for pipes and redirects
    dangerous_chars = ['`', '$', '\\']
    for char in dangerous_chars:
        command = command.replace(char, '')

    return command.strip()


def run_command(
    cmd: str,
    timeout: int = 60,
    check_safety: bool = True,
    allow_shell: bool = False,
    working_dir: Optional[str] = None
) -> Tuple[int, str, str]:
    """Run a shell command with safety checks.

    Args:
        cmd: Command string to execute
        timeout: Maximum execution time in seconds
        check_safety: Whether to perform safety checks
        allow_shell: Allow shell=True (dangerous, use with caution)
        working_dir: Working directory for command execution

    Returns:
        Tuple of (return_code, stdout, stderr)

    Raises:
        ShellInjectionError: If dangerous patterns detected
        UnsafeOperationError: If command is not allowed
    """
    if not cmd or not cmd.strip():
        return 1, "", "Empty command"

    if check_safety:
        # Check for dangerous patterns
        dangerous = _is_dangerous_pattern(cmd)
        if dangerous:
            raise ShellInjectionError(f"Dangerous pattern detected: {dangerous}")

        # Check if command is allowed
        allowed, reason = _is_command_allowed(cmd)
        if not allowed:
            raise UnsafeOperationError("command_execution", reason)

    try:
        if allow_shell:
            # Use shell=True but with sanitization
            sanitized_cmd = sanitize_command(cmd)
            proc = subprocess.run(
                sanitized_cmd,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=working_dir
            )
        else:
            # Safer: split command and run without shell
            args = shlex.split(cmd)
            proc = subprocess.run(
                args,
                shell=False,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=working_dir
            )

        return proc.returncode, proc.stdout, proc.stderr

    except subprocess.TimeoutExpired:
        return 124, "", f"Command timed out after {timeout} seconds"
    except FileNotFoundError as e:
        return 127, "", f"Command not found: {e}"
    except Exception as e:
        return 1, "", f"Command execution failed: {e}"


def run_command_safe(
    args: List[str],
    timeout: int = 60,
    working_dir: Optional[str] = None,
    env: Optional[dict] = None
) -> Tuple[int, str, str]:
    """Run a command with arguments list (safest method).

    Args:
        args: List of command and arguments ['npm', 'install']
        timeout: Maximum execution time in seconds
        working_dir: Working directory for command execution
        env: Environment variables (merged with current env)

    Returns:
        Tuple of (return_code, stdout, stderr)
    """
    if not args:
        return 1, "", "Empty command"

    base_cmd = os.path.basename(args[0])

    # Check if command is blocked
    if base_cmd in BLOCKED_COMMANDS:
        raise UnsafeOperationError("command_execution", f"Command '{base_cmd}' is blocked")

    # Prepare environment
    cmd_env = os.environ.copy()
    if env:
        cmd_env.update(env)

    try:
        proc = subprocess.run(
            args,
            shell=False,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=working_dir,
            env=cmd_env
        )
        return proc.returncode, proc.stdout, proc.stderr

    except subprocess.TimeoutExpired:
        return 124, "", f"Command timed out after {timeout} seconds"
    except FileNotFoundError as e:
        return 127, "", f"Command not found: {e}"
    except Exception as e:
        return 1, "", f"Command execution failed: {e}"


def run_npm_command(
    args: List[str],
    working_dir: str,
    timeout: int = 300
) -> Tuple[int, str, str]:
    """Convenience function for npm commands.

    Args:
        args: npm arguments ['install', 'react']
        working_dir: Project directory
        timeout: Timeout in seconds (default 5 minutes for npm)

    Returns:
        Tuple of (return_code, stdout, stderr)
    """
    return run_command_safe(['npm'] + args, timeout=timeout, working_dir=working_dir)


def run_python_command(
    args: List[str],
    working_dir: Optional[str] = None,
    timeout: int = 120
) -> Tuple[int, str, str]:
    """Convenience function for python commands.

    Args:
        args: Python arguments ['-m', 'pytest']
        working_dir: Working directory
        timeout: Timeout in seconds

    Returns:
        Tuple of (return_code, stdout, stderr)
    """
    return run_command_safe(['python3'] + args, timeout=timeout, working_dir=working_dir)


def run_git_command(
    args: List[str],
    working_dir: str,
    timeout: int = 60
) -> Tuple[int, str, str]:
    """Convenience function for git commands.

    Args:
        args: Git arguments ['status', '-s']
        working_dir: Repository directory
        timeout: Timeout in seconds

    Returns:
        Tuple of (return_code, stdout, stderr)
    """
    return run_command_safe(['git'] + args, timeout=timeout, working_dir=working_dir)
