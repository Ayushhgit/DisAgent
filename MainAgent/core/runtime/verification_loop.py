"""
Verification Loop - Closed-loop validation for code changes.

Integrates with:
- FileManager: Transaction support for rollback on test failure
- TestRunner: Automated test execution after changes
- LLM: Error correction when tests fail

Key Features:
- Apply code changes within a transaction
- Automatically run tests after changes
- Rollback on test failure
- LLM-assisted error correction with retry
- Configurable retry limits and strategies
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Callable, Any
import threading

from .test_runner import TestResult, TestRunnerFactory, TestFramework, run_tests
from .event_bus import publish_event, EventType

logger = logging.getLogger(__name__)


class VerificationStatus(Enum):
    """Status of a verification attempt."""
    SUCCESS = "success"
    TEST_FAILURE = "test_failure"
    PATCH_FAILURE = "patch_failure"
    ROLLBACK = "rollback"
    RETRY = "retry"
    MAX_RETRIES_EXCEEDED = "max_retries_exceeded"
    SKIPPED = "skipped"


@dataclass
class CodeEdit:
    """Represents a code edit to be verified."""
    file_path: str
    old_content: str
    new_content: str
    description: str = ""
    agent_id: str = ""
    task_id: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class VerificationResult:
    """Result of a verification attempt."""
    status: VerificationStatus
    edit: CodeEdit
    test_result: Optional[TestResult] = None
    attempts: int = 0
    duration_seconds: float = 0.0
    error_message: str = ""
    corrections_applied: List[str] = field(default_factory=list)
    final_content: str = ""

    @property
    def success(self) -> bool:
        return self.status == VerificationStatus.SUCCESS


@dataclass
class VerificationConfig:
    """Configuration for verification loop."""
    max_retries: int = 3
    run_tests: bool = True
    test_timeout: int = 300
    test_framework: Optional[TestFramework] = None
    custom_test_command: Optional[str] = None
    test_files: Optional[List[str]] = None
    test_pattern: Optional[str] = None
    require_all_tests_pass: bool = True
    allowed_failure_rate: float = 0.0  # 0 = no failures allowed
    auto_correct: bool = True  # Use LLM to correct failures
    rollback_on_failure: bool = True


class VerificationLoop:
    """Closed-loop verification system for code changes.

    Implements:
    1. Apply edit within transaction
    2. Run tests
    3. If tests fail: rollback and optionally retry with corrections
    4. Track verification history and metrics
    """

    def __init__(
        self,
        project_path: str,
        config: Optional[VerificationConfig] = None,
        llm_call: Optional[Callable[[str], str]] = None
    ):
        """Initialize verification loop.

        Args:
            project_path: Root directory of the project
            config: Verification configuration
            llm_call: Optional LLM function for error correction
        """
        self.project_path = project_path
        self.config = config or VerificationConfig()
        self.llm_call = llm_call

        self._lock = threading.Lock()
        self._verification_history: List[VerificationResult] = []
        self._stats = {
            "total_verifications": 0,
            "successful": 0,
            "failed": 0,
            "retries": 0,
            "rollbacks": 0
        }

    def verify_edit(
        self,
        edit: CodeEdit,
        file_manager: Any,  # FileManager from orchestrator
        context: Optional[Any] = None
    ) -> VerificationResult:
        """Apply an edit with verification.

        Args:
            edit: The code edit to apply and verify
            file_manager: FileManager instance for file operations
            context: Optional context for LLM corrections

        Returns:
            VerificationResult with status and details
        """
        start_time = time.time()
        attempts = 0
        corrections = []
        current_new_content = edit.new_content

        with self._lock:
            self._stats["total_verifications"] += 1

        # Skip verification if tests are disabled
        if not self.config.run_tests:
            result = self._apply_edit_only(edit, file_manager)
            result.status = VerificationStatus.SKIPPED
            return result

        while attempts < self.config.max_retries:
            attempts += 1
            logger.info(f"Verification attempt {attempts}/{self.config.max_retries} for {edit.file_path}")

            # Create a modified edit with current content
            current_edit = CodeEdit(
                file_path=edit.file_path,
                old_content=edit.old_content,
                new_content=current_new_content,
                description=edit.description,
                agent_id=edit.agent_id,
                task_id=edit.task_id,
                metadata=edit.metadata
            )

            # Attempt to apply and verify within transaction
            result = self._attempt_verification(current_edit, file_manager, context)
            result.attempts = attempts
            result.corrections_applied = corrections.copy()
            result.duration_seconds = time.time() - start_time

            if result.success:
                with self._lock:
                    self._stats["successful"] += 1
                    self._verification_history.append(result)

                publish_event(
                    EventType.TASK_COMPLETED,
                    agent_id=edit.agent_id,
                    task_id=edit.task_id,
                    data={"verification": "success", "attempts": attempts}
                )
                return result

            # Handle failure
            if result.status == VerificationStatus.PATCH_FAILURE:
                # Patch couldn't be applied, no point retrying
                logger.error(f"Patch application failed for {edit.file_path}")
                break

            if result.status == VerificationStatus.TEST_FAILURE:
                with self._lock:
                    self._stats["retries"] += 1

                # Try to get LLM correction if enabled
                if self.config.auto_correct and self.llm_call and attempts < self.config.max_retries:
                    correction = self._get_llm_correction(
                        current_edit,
                        result.test_result,
                        context
                    )
                    if correction:
                        current_new_content = correction
                        corrections.append(f"Attempt {attempts}: Applied LLM correction")
                        logger.info(f"Applied LLM correction for {edit.file_path}")
                        continue

                # No correction available, fail
                break

        # Max retries exceeded or unrecoverable failure
        with self._lock:
            self._stats["failed"] += 1
            self._verification_history.append(result)

        result.status = VerificationStatus.MAX_RETRIES_EXCEEDED
        result.attempts = attempts
        result.corrections_applied = corrections
        result.duration_seconds = time.time() - start_time

        publish_event(
            EventType.TASK_FAILED,
            agent_id=edit.agent_id,
            task_id=edit.task_id,
            data={
                "verification": "failed",
                "attempts": attempts,
                "error": result.error_message
            }
        )

        return result

    def _attempt_verification(
        self,
        edit: CodeEdit,
        file_manager: Any,
        context: Optional[Any]
    ) -> VerificationResult:
        """Single verification attempt with transaction."""
        transaction_id = f"verify_{edit.task_id}_{int(time.time() * 1000)}"

        try:
            # Start transaction
            with file_manager.transaction(transaction_id):
                # Apply the edit
                success = file_manager.edit_file(
                    edit.file_path,
                    edit.old_content,
                    edit.new_content
                )

                if not success:
                    return VerificationResult(
                        status=VerificationStatus.PATCH_FAILURE,
                        edit=edit,
                        error_message=f"Failed to apply patch to {edit.file_path}"
                    )

                # Run tests
                test_result = run_tests(
                    self.project_path,
                    test_files=self.config.test_files,
                    test_pattern=self.config.test_pattern,
                    framework=self.config.test_framework,
                    custom_command=self.config.custom_test_command,
                    timeout=self.config.test_timeout
                )

                # Check test results
                if self._tests_acceptable(test_result):
                    # Get final content
                    final_content = file_manager.read_file(edit.file_path) or edit.new_content

                    return VerificationResult(
                        status=VerificationStatus.SUCCESS,
                        edit=edit,
                        test_result=test_result,
                        final_content=final_content
                    )
                else:
                    # Tests failed - transaction will be rolled back
                    with self._lock:
                        self._stats["rollbacks"] += 1

                    raise TestVerificationError(
                        f"Tests failed: {test_result.error_summary}",
                        test_result
                    )

        except TestVerificationError as e:
            # Expected failure - tests didn't pass
            return VerificationResult(
                status=VerificationStatus.TEST_FAILURE,
                edit=edit,
                test_result=e.test_result,
                error_message=str(e)
            )
        except Exception as e:
            # Unexpected error
            logger.exception(f"Verification error: {e}")
            return VerificationResult(
                status=VerificationStatus.ROLLBACK,
                edit=edit,
                error_message=str(e)
            )

    def _tests_acceptable(self, test_result: TestResult) -> bool:
        """Check if test results are acceptable."""
        if self.config.require_all_tests_pass:
            return test_result.success

        if test_result.total == 0:
            # No tests found - consider acceptable if not requiring all pass
            return not self.config.require_all_tests_pass

        return test_result.failure_rate <= self.config.allowed_failure_rate

    def _get_llm_correction(
        self,
        edit: CodeEdit,
        test_result: Optional[TestResult],
        context: Optional[Any]
    ) -> Optional[str]:
        """Get LLM correction for failed tests."""
        if not self.llm_call or not test_result:
            return None

        failure_summary = test_result.get_failure_summary()
        if not failure_summary:
            return None

        prompt = f"""You are fixing a code change that caused test failures.

ORIGINAL FILE: {edit.file_path}

ORIGINAL CODE (that was being replaced):
```
{edit.old_content}
```

NEW CODE (that caused failures):
```
{edit.new_content}
```

TEST FAILURES:
{failure_summary}

EDIT DESCRIPTION: {edit.description}

Please provide a CORRECTED version of the new code that will fix the test failures.
Return ONLY the corrected code, no explanations.
The code should replace the same section as the original edit.
"""

        try:
            corrected = self.llm_call(prompt)
            if corrected and corrected.strip():
                # Clean up any markdown code blocks
                corrected = corrected.strip()
                if corrected.startswith("```"):
                    lines = corrected.split('\n')
                    # Remove first and last lines (```python and ```)
                    if len(lines) > 2:
                        corrected = '\n'.join(lines[1:-1])

                return corrected
        except Exception as e:
            logger.warning(f"LLM correction failed: {e}")

        return None

    def _apply_edit_only(
        self,
        edit: CodeEdit,
        file_manager: Any
    ) -> VerificationResult:
        """Apply edit without verification (when tests disabled)."""
        try:
            with file_manager.transaction(f"edit_{edit.task_id}"):
                success = file_manager.edit_file(
                    edit.file_path,
                    edit.old_content,
                    edit.new_content
                )

                if success:
                    final_content = file_manager.read_file(edit.file_path) or edit.new_content
                    return VerificationResult(
                        status=VerificationStatus.SUCCESS,
                        edit=edit,
                        final_content=final_content
                    )
                else:
                    return VerificationResult(
                        status=VerificationStatus.PATCH_FAILURE,
                        edit=edit,
                        error_message="Failed to apply patch"
                    )
        except Exception as e:
            return VerificationResult(
                status=VerificationStatus.ROLLBACK,
                edit=edit,
                error_message=str(e)
            )

    def get_stats(self) -> Dict[str, Any]:
        """Get verification statistics."""
        with self._lock:
            return dict(self._stats)

    def get_history(self, limit: int = 10) -> List[VerificationResult]:
        """Get recent verification history."""
        with self._lock:
            return self._verification_history[-limit:]

    def reset_stats(self) -> None:
        """Reset statistics."""
        with self._lock:
            self._stats = {
                "total_verifications": 0,
                "successful": 0,
                "failed": 0,
                "retries": 0,
                "rollbacks": 0
            }
            self._verification_history.clear()


class TestVerificationError(Exception):
    """Exception raised when test verification fails."""

    def __init__(self, message: str, test_result: TestResult):
        super().__init__(message)
        self.test_result = test_result


class BatchVerificationLoop:
    """Verify multiple edits as a batch with proper ordering."""

    def __init__(
        self,
        project_path: str,
        config: Optional[VerificationConfig] = None,
        llm_call: Optional[Callable[[str], str]] = None
    ):
        self.project_path = project_path
        self.config = config or VerificationConfig()
        self.llm_call = llm_call
        self._verification_loop = VerificationLoop(project_path, config, llm_call)

    def verify_batch(
        self,
        edits: List[CodeEdit],
        file_manager: Any,
        context: Optional[Any] = None,
        stop_on_failure: bool = True
    ) -> List[VerificationResult]:
        """Verify a batch of edits.

        Args:
            edits: List of edits to apply and verify
            file_manager: FileManager for file operations
            context: Optional context for LLM corrections
            stop_on_failure: Stop processing on first failure

        Returns:
            List of verification results
        """
        results = []

        for edit in edits:
            result = self._verification_loop.verify_edit(edit, file_manager, context)
            results.append(result)

            if not result.success and stop_on_failure:
                logger.warning(f"Stopping batch verification due to failure in {edit.file_path}")
                # Mark remaining edits as skipped
                for remaining_edit in edits[len(results):]:
                    results.append(VerificationResult(
                        status=VerificationStatus.SKIPPED,
                        edit=remaining_edit,
                        error_message="Skipped due to previous failure"
                    ))
                break

        return results

    def verify_batch_atomic(
        self,
        edits: List[CodeEdit],
        file_manager: Any,
        context: Optional[Any] = None
    ) -> List[VerificationResult]:
        """Verify a batch atomically - all succeed or all rollback.

        This applies all edits first, then runs tests once.
        If tests fail, all edits are rolled back.
        """
        if not edits:
            return []

        results = []
        transaction_id = f"batch_{int(time.time() * 1000)}"

        try:
            with file_manager.transaction(transaction_id):
                # Apply all edits
                for edit in edits:
                    success = file_manager.edit_file(
                        edit.file_path,
                        edit.old_content,
                        edit.new_content
                    )
                    if not success:
                        raise PatchApplicationError(f"Failed to apply patch to {edit.file_path}")

                # Run tests once for all edits
                if self.config.run_tests:
                    test_result = run_tests(
                        self.project_path,
                        test_files=self.config.test_files,
                        test_pattern=self.config.test_pattern,
                        framework=self.config.test_framework,
                        custom_command=self.config.custom_test_command,
                        timeout=self.config.test_timeout
                    )

                    if not test_result.success:
                        raise TestVerificationError(
                            f"Tests failed: {test_result.error_summary}",
                            test_result
                        )

                # All succeeded
                for edit in edits:
                    final_content = file_manager.read_file(edit.file_path) or edit.new_content
                    results.append(VerificationResult(
                        status=VerificationStatus.SUCCESS,
                        edit=edit,
                        test_result=test_result if self.config.run_tests else None,
                        final_content=final_content
                    ))

        except PatchApplicationError as e:
            for edit in edits:
                results.append(VerificationResult(
                    status=VerificationStatus.PATCH_FAILURE,
                    edit=edit,
                    error_message=str(e)
                ))

        except TestVerificationError as e:
            for edit in edits:
                results.append(VerificationResult(
                    status=VerificationStatus.TEST_FAILURE,
                    edit=edit,
                    test_result=e.test_result,
                    error_message=str(e)
                ))

        except Exception as e:
            for edit in edits:
                results.append(VerificationResult(
                    status=VerificationStatus.ROLLBACK,
                    edit=edit,
                    error_message=str(e)
                ))

        return results


class PatchApplicationError(Exception):
    """Exception raised when patch application fails."""
    pass


def create_verification_loop(
    project_path: str,
    max_retries: int = 3,
    run_tests: bool = True,
    auto_correct: bool = True,
    llm_call: Optional[Callable[[str], str]] = None,
    **kwargs
) -> VerificationLoop:
    """Factory function to create a verification loop.

    Args:
        project_path: Root directory of the project
        max_retries: Maximum retry attempts
        run_tests: Whether to run tests after edits
        auto_correct: Whether to use LLM for error correction
        llm_call: LLM function for corrections
        **kwargs: Additional config options

    Returns:
        Configured VerificationLoop instance
    """
    config = VerificationConfig(
        max_retries=max_retries,
        run_tests=run_tests,
        auto_correct=auto_correct,
        **kwargs
    )
    return VerificationLoop(project_path, config, llm_call)
