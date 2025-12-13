"""
Test Runner Abstraction Layer

Provides a unified interface for running tests across different frameworks:
- Python: pytest, unittest
- JavaScript/TypeScript: npm test, jest, mocha
- Generic: custom commands

Key Features:
- Auto-detection of test framework from project structure
- Streaming test output
- Structured test results with pass/fail/skip counts
- Integration with verification loop for closed-loop validation
"""

from __future__ import annotations

import os
import re
import subprocess
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import threading

logger = logging.getLogger(__name__)


class TestFramework(Enum):
    """Supported test frameworks."""
    PYTEST = "pytest"
    UNITTEST = "unittest"
    NPM_TEST = "npm_test"
    JEST = "jest"
    MOCHA = "mocha"
    GO_TEST = "go_test"
    CARGO_TEST = "cargo_test"
    CUSTOM = "custom"
    UNKNOWN = "unknown"


class TestStatus(Enum):
    """Status of a single test."""
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    ERROR = "error"


@dataclass
class TestCase:
    """Represents a single test case result."""
    name: str
    status: TestStatus
    duration_ms: float = 0.0
    message: str = ""
    file_path: str = ""
    line_number: int = 0
    traceback: str = ""


@dataclass
class TestResult:
    """Aggregate result of a test run."""
    success: bool
    framework: TestFramework
    total: int = 0
    passed: int = 0
    failed: int = 0
    skipped: int = 0
    errors: int = 0
    duration_seconds: float = 0.0
    test_cases: List[TestCase] = field(default_factory=list)
    stdout: str = ""
    stderr: str = ""
    return_code: int = 0
    error_summary: str = ""

    @property
    def failure_rate(self) -> float:
        """Calculate failure rate."""
        if self.total == 0:
            return 0.0
        return (self.failed + self.errors) / self.total

    def get_failed_tests(self) -> List[TestCase]:
        """Get list of failed test cases."""
        return [t for t in self.test_cases if t.status in (TestStatus.FAILED, TestStatus.ERROR)]

    def get_failure_summary(self) -> str:
        """Get a summary of all failures for error correction."""
        failed = self.get_failed_tests()
        if not failed:
            return ""

        lines = [f"Failed tests ({len(failed)}/{self.total}):"]
        for test in failed[:10]:  # Limit to first 10
            lines.append(f"\n- {test.name}")
            if test.message:
                lines.append(f"  Error: {test.message}")
            if test.file_path and test.line_number:
                lines.append(f"  Location: {test.file_path}:{test.line_number}")
            if test.traceback:
                # Truncate long tracebacks
                tb = test.traceback[:500] + "..." if len(test.traceback) > 500 else test.traceback
                lines.append(f"  Traceback:\n{tb}")

        if len(failed) > 10:
            lines.append(f"\n... and {len(failed) - 10} more failures")

        return "\n".join(lines)


class BaseTestRunner(ABC):
    """Abstract base class for test runners."""

    def __init__(self, project_path: str, timeout: int = 300):
        """Initialize test runner.

        Args:
            project_path: Root directory of the project
            timeout: Maximum time for test run in seconds
        """
        self.project_path = Path(project_path).resolve()
        self.timeout = timeout
        self._lock = threading.Lock()

    @abstractmethod
    def detect(self) -> bool:
        """Check if this test runner is applicable to the project."""
        pass

    @abstractmethod
    def run(self,
            test_files: Optional[List[str]] = None,
            test_pattern: Optional[str] = None,
            verbose: bool = False) -> TestResult:
        """Run tests and return structured results.

        Args:
            test_files: Specific test files to run (None for all)
            test_pattern: Pattern to filter tests by name
            verbose: Enable verbose output

        Returns:
            TestResult with structured test results
        """
        pass

    def _run_command(self,
                     cmd: List[str],
                     cwd: Optional[Path] = None,
                     env: Optional[Dict[str, str]] = None) -> Tuple[int, str, str]:
        """Run a command and capture output.

        Returns:
            Tuple of (return_code, stdout, stderr)
        """
        full_env = os.environ.copy()
        if env:
            full_env.update(env)

        try:
            proc = subprocess.run(
                cmd,
                cwd=cwd or self.project_path,
                capture_output=True,
                text=True,
                timeout=self.timeout,
                env=full_env
            )
            return proc.returncode, proc.stdout, proc.stderr
        except subprocess.TimeoutExpired:
            logger.error(f"Test run timed out after {self.timeout}s")
            return -1, "", f"Test run timed out after {self.timeout} seconds"
        except Exception as e:
            logger.exception(f"Error running tests: {e}")
            return -1, "", str(e)


class PytestRunner(BaseTestRunner):
    """Test runner for Python projects using pytest."""

    framework = TestFramework.PYTEST

    def detect(self) -> bool:
        """Check for pytest in project."""
        # Check for pytest in requirements or pyproject.toml
        indicators = [
            self.project_path / "pytest.ini",
            self.project_path / "pyproject.toml",
            self.project_path / "setup.cfg",
            self.project_path / "conftest.py",
        ]

        for indicator in indicators:
            if indicator.exists():
                try:
                    content = indicator.read_text()
                    if "pytest" in content.lower():
                        return True
                except Exception:
                    pass

        # Check for test files
        test_patterns = ["test_*.py", "*_test.py", "tests/*.py"]
        for pattern in test_patterns:
            if list(self.project_path.glob(f"**/{pattern}")):
                return True

        return False

    def run(self,
            test_files: Optional[List[str]] = None,
            test_pattern: Optional[str] = None,
            verbose: bool = False) -> TestResult:
        """Run pytest and parse results."""
        start_time = time.time()

        cmd = ["python", "-m", "pytest"]

        # Add verbose flag
        if verbose:
            cmd.append("-v")

        # Add test selection
        if test_files:
            cmd.extend(test_files)

        if test_pattern:
            cmd.extend(["-k", test_pattern])

        # Add output format for parsing
        cmd.extend(["--tb=short", "-q"])

        return_code, stdout, stderr = self._run_command(cmd)
        duration = time.time() - start_time

        # Parse pytest output
        result = self._parse_output(stdout, stderr, return_code)
        result.duration_seconds = duration
        result.return_code = return_code
        result.stdout = stdout
        result.stderr = stderr

        return result

    def _parse_output(self, stdout: str, stderr: str, return_code: int) -> TestResult:
        """Parse pytest output into structured result."""
        test_cases = []
        passed = 0
        failed = 0
        skipped = 0
        errors = 0

        # Parse summary line: "5 passed, 2 failed, 1 skipped"
        summary_match = re.search(
            r'(\d+)\s+passed(?:.*?(\d+)\s+failed)?(?:.*?(\d+)\s+skipped)?(?:.*?(\d+)\s+error)?',
            stdout,
            re.IGNORECASE
        )

        if summary_match:
            passed = int(summary_match.group(1) or 0)
            failed = int(summary_match.group(2) or 0)
            skipped = int(summary_match.group(3) or 0)
            errors = int(summary_match.group(4) or 0)

        # Parse individual test results
        # Look for lines like: "PASSED tests/test_foo.py::test_bar"
        for line in stdout.split('\n'):
            test_match = re.match(r'^(PASSED|FAILED|ERROR|SKIPPED)\s+(.+?)(?:\s+-\s+(.+))?$', line)
            if test_match:
                status_str, test_name, message = test_match.groups()
                status_map = {
                    'PASSED': TestStatus.PASSED,
                    'FAILED': TestStatus.FAILED,
                    'ERROR': TestStatus.ERROR,
                    'SKIPPED': TestStatus.SKIPPED
                }
                test_cases.append(TestCase(
                    name=test_name,
                    status=status_map.get(status_str, TestStatus.ERROR),
                    message=message or ""
                ))

        # Parse failure details
        failure_section = re.search(r'=+ FAILURES =+(.+?)(?:=+ \w+ =+|$)', stdout, re.DOTALL)
        if failure_section:
            failure_text = failure_section.group(1)
            # Extract individual failure blocks
            failure_blocks = re.findall(
                r'_+ (\S+) _+\s*\n(.+?)(?=_+ \S+ _+|\Z)',
                failure_text,
                re.DOTALL
            )
            for test_name, traceback in failure_blocks:
                # Update existing test case with traceback
                for tc in test_cases:
                    if test_name in tc.name:
                        tc.traceback = traceback.strip()
                        break

        total = passed + failed + skipped + errors
        success = return_code == 0 and failed == 0 and errors == 0

        error_summary = ""
        if not success:
            error_summary = f"Tests failed: {failed} failures, {errors} errors out of {total} tests"

        return TestResult(
            success=success,
            framework=TestFramework.PYTEST,
            total=total,
            passed=passed,
            failed=failed,
            skipped=skipped,
            errors=errors,
            test_cases=test_cases,
            error_summary=error_summary
        )


class UnittestRunner(BaseTestRunner):
    """Test runner for Python projects using unittest."""

    framework = TestFramework.UNITTEST

    def detect(self) -> bool:
        """Check for unittest usage."""
        # Check for test files with unittest imports
        for test_file in self.project_path.glob("**/test*.py"):
            try:
                content = test_file.read_text()
                if "import unittest" in content or "from unittest" in content:
                    return True
            except Exception:
                pass
        return False

    def run(self,
            test_files: Optional[List[str]] = None,
            test_pattern: Optional[str] = None,
            verbose: bool = False) -> TestResult:
        """Run unittest and parse results."""
        start_time = time.time()

        cmd = ["python", "-m", "unittest"]

        if verbose:
            cmd.append("-v")

        if test_files:
            # Convert file paths to module names
            modules = []
            for f in test_files:
                module = f.replace("/", ".").replace("\\", ".").replace(".py", "")
                modules.append(module)
            cmd.extend(modules)
        else:
            cmd.append("discover")

        return_code, stdout, stderr = self._run_command(cmd)
        duration = time.time() - start_time

        result = self._parse_output(stdout, stderr, return_code)
        result.duration_seconds = duration
        result.return_code = return_code
        result.stdout = stdout
        result.stderr = stderr

        return result

    def _parse_output(self, stdout: str, stderr: str, return_code: int) -> TestResult:
        """Parse unittest output."""
        # Unittest outputs to stderr
        output = stderr + stdout

        # Parse "Ran X tests in Y.YYYs"
        ran_match = re.search(r'Ran (\d+) tests? in ([\d.]+)s', output)
        total = int(ran_match.group(1)) if ran_match else 0

        # Parse OK/FAILED status
        if "OK" in output:
            passed = total
            failed = 0
        else:
            fail_match = re.search(r'FAILED \((?:failures=(\d+))?(?:,? ?errors=(\d+))?\)', output)
            if fail_match:
                failed = int(fail_match.group(1) or 0)
                errors = int(fail_match.group(2) or 0)
                passed = total - failed - errors
            else:
                passed = 0
                failed = total

        success = return_code == 0

        return TestResult(
            success=success,
            framework=TestFramework.UNITTEST,
            total=total,
            passed=passed,
            failed=failed,
            error_summary="" if success else f"{failed} tests failed"
        )


class NpmTestRunner(BaseTestRunner):
    """Test runner for Node.js projects using npm test."""

    framework = TestFramework.NPM_TEST

    def detect(self) -> bool:
        """Check for package.json with test script."""
        package_json = self.project_path / "package.json"
        if package_json.exists():
            try:
                import json
                data = json.loads(package_json.read_text())
                return "test" in data.get("scripts", {})
            except Exception:
                pass
        return False

    def run(self,
            test_files: Optional[List[str]] = None,
            test_pattern: Optional[str] = None,
            verbose: bool = False) -> TestResult:
        """Run npm test."""
        start_time = time.time()

        cmd = ["npm", "test"]

        if test_pattern:
            cmd.extend(["--", f"--grep={test_pattern}"])

        return_code, stdout, stderr = self._run_command(cmd)
        duration = time.time() - start_time

        result = self._parse_output(stdout, stderr, return_code)
        result.duration_seconds = duration
        result.return_code = return_code
        result.stdout = stdout
        result.stderr = stderr

        return result

    def _parse_output(self, stdout: str, stderr: str, return_code: int) -> TestResult:
        """Parse npm test output (basic parsing)."""
        output = stdout + stderr

        # Try to detect Jest output
        jest_match = re.search(r'Tests:\s*(\d+)\s+passed(?:,\s*(\d+)\s+failed)?', output)
        if jest_match:
            passed = int(jest_match.group(1))
            failed = int(jest_match.group(2) or 0)
            return TestResult(
                success=return_code == 0,
                framework=TestFramework.JEST,
                total=passed + failed,
                passed=passed,
                failed=failed
            )

        # Try to detect Mocha output
        mocha_match = re.search(r'(\d+)\s+passing.*?(\d+)\s+failing', output, re.DOTALL)
        if mocha_match:
            passed = int(mocha_match.group(1))
            failed = int(mocha_match.group(2))
            return TestResult(
                success=return_code == 0,
                framework=TestFramework.MOCHA,
                total=passed + failed,
                passed=passed,
                failed=failed
            )

        # Generic result
        return TestResult(
            success=return_code == 0,
            framework=TestFramework.NPM_TEST,
            error_summary="" if return_code == 0 else "Tests failed"
        )


class CustomTestRunner(BaseTestRunner):
    """Custom test runner using a user-specified command."""

    framework = TestFramework.CUSTOM

    def __init__(self, project_path: str, command: str, timeout: int = 300):
        super().__init__(project_path, timeout)
        self.command = command

    def detect(self) -> bool:
        """Always returns True for custom runner."""
        return True

    def run(self,
            test_files: Optional[List[str]] = None,
            test_pattern: Optional[str] = None,
            verbose: bool = False) -> TestResult:
        """Run custom test command."""
        start_time = time.time()

        # Split command into parts
        import shlex
        cmd = shlex.split(self.command)

        return_code, stdout, stderr = self._run_command(cmd)
        duration = time.time() - start_time

        return TestResult(
            success=return_code == 0,
            framework=TestFramework.CUSTOM,
            duration_seconds=duration,
            stdout=stdout,
            stderr=stderr,
            return_code=return_code,
            error_summary="" if return_code == 0 else f"Command failed with exit code {return_code}"
        )


class TestRunnerFactory:
    """Factory for creating appropriate test runners."""

    _runners = [
        PytestRunner,
        UnittestRunner,
        NpmTestRunner,
    ]

    @classmethod
    def create(cls,
               project_path: str,
               framework: Optional[TestFramework] = None,
               custom_command: Optional[str] = None,
               timeout: int = 300) -> BaseTestRunner:
        """Create appropriate test runner for project.

        Args:
            project_path: Root directory of the project
            framework: Specific framework to use (auto-detect if None)
            custom_command: Custom test command (creates CustomTestRunner)
            timeout: Maximum time for test run

        Returns:
            Appropriate test runner instance
        """
        if custom_command:
            return CustomTestRunner(project_path, custom_command, timeout)

        if framework:
            runner_map = {
                TestFramework.PYTEST: PytestRunner,
                TestFramework.UNITTEST: UnittestRunner,
                TestFramework.NPM_TEST: NpmTestRunner,
            }
            runner_class = runner_map.get(framework)
            if runner_class:
                return runner_class(project_path, timeout)

        # Auto-detect
        for runner_class in cls._runners:
            runner = runner_class(project_path, timeout)
            if runner.detect():
                logger.info(f"Auto-detected test framework: {runner.framework.value}")
                return runner

        # Fallback to pytest (most common)
        logger.warning("Could not detect test framework, defaulting to pytest")
        return PytestRunner(project_path, timeout)

    @classmethod
    def detect_framework(cls, project_path: str) -> TestFramework:
        """Detect the test framework used in a project."""
        for runner_class in cls._runners:
            runner = runner_class(project_path)
            if runner.detect():
                return runner.framework
        return TestFramework.UNKNOWN


def run_tests(
    project_path: str,
    test_files: Optional[List[str]] = None,
    test_pattern: Optional[str] = None,
    framework: Optional[TestFramework] = None,
    custom_command: Optional[str] = None,
    timeout: int = 300,
    verbose: bool = False
) -> TestResult:
    """Convenience function to run tests.

    Args:
        project_path: Root directory of the project
        test_files: Specific test files to run
        test_pattern: Pattern to filter tests
        framework: Specific framework to use
        custom_command: Custom test command
        timeout: Maximum time for test run
        verbose: Enable verbose output

    Returns:
        TestResult with structured results
    """
    runner = TestRunnerFactory.create(
        project_path,
        framework=framework,
        custom_command=custom_command,
        timeout=timeout
    )
    return runner.run(
        test_files=test_files,
        test_pattern=test_pattern,
        verbose=verbose
    )
