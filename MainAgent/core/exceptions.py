# ============================================================================
# FILE: exceptions.py
# Custom exceptions for consistent error handling across the system
# ============================================================================

from typing import Optional, Dict, Any


class DisAgentError(Exception):
    """Base exception for all DisAgent errors."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        self.message = message
        self.details = details or {}
        super().__init__(self.message)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "error_type": self.__class__.__name__,
            "message": self.message,
            "details": self.details
        }


# === LLM Related Exceptions ===

class LLMError(DisAgentError):
    """Base exception for LLM-related errors."""
    pass


class LLMTimeoutError(LLMError):
    """Raised when LLM call times out."""

    def __init__(self, timeout_seconds: float, message: str = "LLM call timed out"):
        super().__init__(message, {"timeout_seconds": timeout_seconds})
        self.timeout_seconds = timeout_seconds


class LLMRateLimitError(LLMError):
    """Raised when LLM rate limit is exceeded."""

    def __init__(self, retry_after: Optional[float] = None):
        message = "LLM rate limit exceeded"
        if retry_after:
            message += f", retry after {retry_after}s"
        super().__init__(message, {"retry_after": retry_after})
        self.retry_after = retry_after


class LLMResponseError(LLMError):
    """Raised when LLM returns invalid/unparseable response."""

    def __init__(self, message: str = "Invalid LLM response", raw_response: str = ""):
        super().__init__(message, {"raw_response": raw_response[:500]})
        self.raw_response = raw_response


# === Agent Related Exceptions ===

class AgentError(DisAgentError):
    """Base exception for agent-related errors."""
    pass


class AgentTimeoutError(AgentError):
    """Raised when agent execution times out."""

    def __init__(self, agent_name: str, timeout_seconds: float):
        message = f"Agent '{agent_name}' timed out after {timeout_seconds}s"
        super().__init__(message, {"agent_name": agent_name, "timeout_seconds": timeout_seconds})
        self.agent_name = agent_name
        self.timeout_seconds = timeout_seconds


class AgentExecutionError(AgentError):
    """Raised when agent execution fails."""

    def __init__(self, agent_name: str, reason: str, recoverable: bool = True):
        message = f"Agent '{agent_name}' failed: {reason}"
        super().__init__(message, {"agent_name": agent_name, "reason": reason, "recoverable": recoverable})
        self.agent_name = agent_name
        self.reason = reason
        self.recoverable = recoverable


class AgentNotFoundError(AgentError):
    """Raised when requested agent type doesn't exist."""

    def __init__(self, agent_name: str):
        message = f"Agent type '{agent_name}' not found"
        super().__init__(message, {"agent_name": agent_name})
        self.agent_name = agent_name


# === File Operation Exceptions ===

class FileOperationError(DisAgentError):
    """Base exception for file operation errors."""
    pass


class FileNotFoundError(FileOperationError):
    """Raised when file is not found (shadows builtin intentionally for consistency)."""

    def __init__(self, file_path: str):
        message = f"File not found: {file_path}"
        super().__init__(message, {"file_path": file_path})
        self.file_path = file_path


class FileWriteError(FileOperationError):
    """Raised when file write fails."""

    def __init__(self, file_path: str, reason: str):
        message = f"Failed to write file '{file_path}': {reason}"
        super().__init__(message, {"file_path": file_path, "reason": reason})
        self.file_path = file_path
        self.reason = reason


class EditMatchError(FileOperationError):
    """Raised when edit pattern doesn't match file content."""

    def __init__(self, file_path: str, pattern_preview: str = ""):
        message = f"Edit pattern not found in '{file_path}'"
        super().__init__(message, {"file_path": file_path, "pattern_preview": pattern_preview[:200]})
        self.file_path = file_path


class TransactionError(FileOperationError):
    """Raised when transaction operation fails."""

    def __init__(self, transaction_id: str, reason: str):
        message = f"Transaction '{transaction_id}' failed: {reason}"
        super().__init__(message, {"transaction_id": transaction_id, "reason": reason})
        self.transaction_id = transaction_id
        self.reason = reason


class RollbackError(FileOperationError):
    """Raised when rollback fails."""

    def __init__(self, transaction_id: str, failed_files: list):
        message = f"Rollback failed for transaction '{transaction_id}'"
        super().__init__(message, {"transaction_id": transaction_id, "failed_files": failed_files})
        self.transaction_id = transaction_id
        self.failed_files = failed_files


# === Memory Exceptions ===

class MemoryError(DisAgentError):
    """Base exception for memory-related errors."""
    pass


class VectorStoreError(MemoryError):
    """Raised when vector store operation fails."""

    def __init__(self, operation: str, reason: str):
        message = f"Vector store {operation} failed: {reason}"
        super().__init__(message, {"operation": operation, "reason": reason})


class MemoryPersistenceError(MemoryError):
    """Raised when memory save/load fails."""

    def __init__(self, operation: str, file_path: str, reason: str):
        message = f"Memory {operation} failed for '{file_path}': {reason}"
        super().__init__(message, {"operation": operation, "file_path": file_path, "reason": reason})


# === Planning/Scope Exceptions ===

class PlanningError(DisAgentError):
    """Base exception for planning-related errors."""
    pass


class ScopeAnalysisError(PlanningError):
    """Raised when scope analysis fails."""

    def __init__(self, reason: str, user_input: str = ""):
        message = f"Scope analysis failed: {reason}"
        super().__init__(message, {"reason": reason, "user_input": user_input[:200]})


class TaskValidationError(PlanningError):
    """Raised when task validation fails."""

    def __init__(self, task_id: str, reason: str):
        message = f"Task '{task_id}' validation failed: {reason}"
        super().__init__(message, {"task_id": task_id, "reason": reason})


# === Verification Exceptions ===

class VerificationError(DisAgentError):
    """Base exception for verification-related errors."""
    pass


class TestExecutionError(VerificationError):
    """Raised when test execution fails."""

    def __init__(self, test_name: str, reason: str, output: str = ""):
        message = f"Test '{test_name}' execution failed: {reason}"
        super().__init__(message, {"test_name": test_name, "reason": reason, "output": output[:500]})


class VerificationTimeoutError(VerificationError):
    """Raised when verification times out."""

    def __init__(self, timeout_seconds: float):
        message = f"Verification timed out after {timeout_seconds}s"
        super().__init__(message, {"timeout_seconds": timeout_seconds})


# === Security Exceptions ===

class SecurityError(DisAgentError):
    """Base exception for security-related errors."""
    pass


class PromptInjectionError(SecurityError):
    """Raised when prompt injection is detected."""

    def __init__(self, detected_pattern: str):
        message = "Potential prompt injection detected"
        super().__init__(message, {"detected_pattern": detected_pattern[:100]})


class ShellInjectionError(SecurityError):
    """Raised when shell injection is detected."""

    def __init__(self, command: str):
        message = "Potential shell injection detected"
        super().__init__(message, {"command_preview": command[:100]})


class UnsafeOperationError(SecurityError):
    """Raised when an unsafe operation is attempted."""

    def __init__(self, operation: str, reason: str):
        message = f"Unsafe operation blocked: {operation}"
        super().__init__(message, {"operation": operation, "reason": reason})


# === Configuration Exceptions ===

class ConfigurationError(DisAgentError):
    """Raised when configuration is invalid or missing."""

    def __init__(self, config_key: str, reason: str):
        message = f"Configuration error for '{config_key}': {reason}"
        super().__init__(message, {"config_key": config_key, "reason": reason})


# === Input Validation Exceptions ===

class ValidationError(DisAgentError):
    """Base exception for input validation errors."""
    pass


class InvalidInputError(ValidationError):
    """Raised when user input is invalid."""

    def __init__(self, field: str, reason: str, value: Any = None):
        message = f"Invalid input for '{field}': {reason}"
        details = {"field": field, "reason": reason}
        if value is not None:
            details["value_preview"] = str(value)[:100]
        super().__init__(message, details)


class EmptyInputError(ValidationError):
    """Raised when required input is empty."""

    def __init__(self, field: str):
        message = f"Required field '{field}' is empty"
        super().__init__(message, {"field": field})
