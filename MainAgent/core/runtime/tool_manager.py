"""
Tool Manager - Comprehensive tool registration and execution system.

Features:
- Tool registration with schema validation
- Tool discovery and listing
- Safe tool execution with error handling
- Tool result formatting
- Support for async tools
- Tool categories and grouping
"""

from __future__ import annotations

import inspect
import logging
import time
import traceback
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Type, Union
from functools import wraps

logger = logging.getLogger(__name__)


class ToolCategory(Enum):
    """Categories for organizing tools."""
    FILESYSTEM = "filesystem"
    BROWSER = "browser"
    CODE_EDITOR = "code_editor"
    SHELL = "shell"
    RESEARCH = "research"
    UTILITY = "utility"


@dataclass
class ToolParameter:
    """Definition of a tool parameter."""
    name: str
    param_type: Type
    description: str
    required: bool = True
    default: Any = None

    def validate(self, value: Any) -> tuple[bool, str]:
        """Validate a parameter value."""
        if value is None:
            if self.required and self.default is None:
                return False, f"Parameter '{self.name}' is required"
            return True, ""

        # Type checking
        if self.param_type == str and not isinstance(value, str):
            return False, f"Parameter '{self.name}' must be a string"
        elif self.param_type == int and not isinstance(value, int):
            return False, f"Parameter '{self.name}' must be an integer"
        elif self.param_type == bool and not isinstance(value, bool):
            return False, f"Parameter '{self.name}' must be a boolean"
        elif self.param_type == list and not isinstance(value, list):
            return False, f"Parameter '{self.name}' must be a list"
        elif self.param_type == dict and not isinstance(value, dict):
            return False, f"Parameter '{self.name}' must be a dictionary"

        return True, ""


@dataclass
class ToolResult:
    """Result from tool execution."""
    success: bool
    output: Any
    error: Optional[str] = None
    execution_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "success": self.success,
            "output": self.output,
            "error": self.error,
            "execution_time": self.execution_time,
            "metadata": self.metadata,
        }


@dataclass
class Tool:
    """Definition of a tool that can be called by agents."""
    name: str
    description: str
    handler: Callable
    category: ToolCategory = ToolCategory.UTILITY
    parameters: List[ToolParameter] = field(default_factory=list)
    requires_confirmation: bool = False
    enabled: bool = True

    def get_schema(self) -> Dict[str, Any]:
        """Get tool schema for LLM function calling."""
        properties = {}
        required = []

        for param in self.parameters:
            prop = {
                "type": self._type_to_json_type(param.param_type),
                "description": param.description,
            }
            if param.default is not None:
                prop["default"] = param.default
            properties[param.name] = prop

            if param.required:
                required.append(param.name)

        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        }

    def _type_to_json_type(self, t: Type) -> str:
        """Convert Python type to JSON schema type."""
        type_map = {
            str: "string",
            int: "integer",
            float: "number",
            bool: "boolean",
            list: "array",
            dict: "object",
        }
        return type_map.get(t, "string")

    def validate_args(self, **kwargs) -> tuple[bool, List[str]]:
        """Validate arguments against parameter definitions."""
        errors = []

        for param in self.parameters:
            value = kwargs.get(param.name, param.default)
            valid, error = param.validate(value)
            if not valid:
                errors.append(error)

        return len(errors) == 0, errors

    def execute(self, **kwargs) -> ToolResult:
        """Execute the tool with given arguments."""
        start_time = time.time()

        if not self.enabled:
            return ToolResult(
                success=False,
                output=None,
                error=f"Tool '{self.name}' is disabled",
            )

        # Validate arguments
        valid, errors = self.validate_args(**kwargs)
        if not valid:
            return ToolResult(
                success=False,
                output=None,
                error=f"Validation errors: {', '.join(errors)}",
            )

        try:
            # Apply defaults
            for param in self.parameters:
                if param.name not in kwargs and param.default is not None:
                    kwargs[param.name] = param.default

            # Execute handler
            result = self.handler(**kwargs)
            execution_time = time.time() - start_time

            return ToolResult(
                success=True,
                output=result,
                execution_time=execution_time,
            )

        except Exception as exc:
            logger.exception(f"Tool '{self.name}' execution failed: {exc}")
            return ToolResult(
                success=False,
                output=None,
                error=str(exc),
                execution_time=time.time() - start_time,
            )


class ToolManager:
    """Manages tool registration, discovery, and execution.

    Usage:
        manager = ToolManager()

        # Register a tool
        manager.register(
            name="read_file",
            handler=read_file_func,
            description="Read contents of a file",
            category=ToolCategory.FILESYSTEM,
            parameters=[
                ToolParameter("path", str, "Path to the file"),
            ]
        )

        # Execute a tool
        result = manager.execute("read_file", path="/path/to/file.txt")
    """

    def __init__(self):
        self._tools: Dict[str, Tool] = {}
        self._categories: Dict[ToolCategory, List[str]] = {cat: [] for cat in ToolCategory}
        self._execution_history: List[Dict[str, Any]] = []

    def register(
        self,
        name: str,
        handler: Callable,
        description: str,
        category: ToolCategory = ToolCategory.UTILITY,
        parameters: Optional[List[ToolParameter]] = None,
        requires_confirmation: bool = False,
    ) -> Tool:
        """Register a new tool.

        Args:
            name: Unique tool name
            handler: Function to execute
            description: Tool description
            category: Tool category
            parameters: List of parameter definitions
            requires_confirmation: Whether tool requires user confirmation

        Returns:
            The registered Tool object
        """
        if name in self._tools:
            logger.warning(f"Tool '{name}' already registered, overwriting")

        tool = Tool(
            name=name,
            description=description,
            handler=handler,
            category=category,
            parameters=parameters or [],
            requires_confirmation=requires_confirmation,
        )

        self._tools[name] = tool

        if name not in self._categories[category]:
            self._categories[category].append(name)

        logger.debug(f"Registered tool: {name} ({category.value})")
        return tool

    def register_tool(self, tool: Tool) -> None:
        """Register an existing Tool object."""
        self._tools[tool.name] = tool
        if tool.name not in self._categories[tool.category]:
            self._categories[tool.category].append(tool.name)

    def unregister(self, name: str) -> bool:
        """Unregister a tool by name."""
        if name in self._tools:
            tool = self._tools.pop(name)
            if name in self._categories[tool.category]:
                self._categories[tool.category].remove(name)
            return True
        return False

    def get(self, name: str) -> Optional[Tool]:
        """Get a tool by name."""
        return self._tools.get(name)

    def get_tools(self) -> Dict[str, Tool]:
        """Get all registered tools."""
        return self._tools.copy()

    def get_tools_by_category(self, category: ToolCategory) -> List[Tool]:
        """Get all tools in a category."""
        tool_names = self._categories.get(category, [])
        return [self._tools[name] for name in tool_names if name in self._tools]

    def list_tools(self) -> List[Dict[str, Any]]:
        """List all tools with their schemas."""
        return [tool.get_schema() for tool in self._tools.values() if tool.enabled]

    def list_tools_by_category(self) -> Dict[str, List[Dict[str, Any]]]:
        """List tools grouped by category."""
        result = {}
        for category in ToolCategory:
            tools = self.get_tools_by_category(category)
            if tools:
                result[category.value] = [t.get_schema() for t in tools if t.enabled]
        return result

    def execute(self, name: str, **kwargs) -> ToolResult:
        """Execute a tool by name.

        Args:
            name: Tool name
            **kwargs: Tool arguments

        Returns:
            ToolResult with execution outcome
        """
        tool = self._tools.get(name)
        if not tool:
            return ToolResult(
                success=False,
                output=None,
                error=f"Tool '{name}' not found",
            )

        result = tool.execute(**kwargs)

        # Record in history
        self._execution_history.append({
            "tool": name,
            "args": kwargs,
            "success": result.success,
            "time": time.time(),
            "execution_time": result.execution_time,
        })

        return result

    def execute_batch(self, calls: List[Dict[str, Any]]) -> List[ToolResult]:
        """Execute multiple tool calls.

        Args:
            calls: List of {"tool": name, "args": {...}} dicts

        Returns:
            List of ToolResults
        """
        results = []
        for call in calls:
            tool_name = call.get("tool")
            args = call.get("args", {})
            results.append(self.execute(tool_name, **args))
        return results

    def enable_tool(self, name: str) -> bool:
        """Enable a tool."""
        if name in self._tools:
            self._tools[name].enabled = True
            return True
        return False

    def disable_tool(self, name: str) -> bool:
        """Disable a tool."""
        if name in self._tools:
            self._tools[name].enabled = False
            return True
        return False

    def get_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get execution history."""
        return self._execution_history[-limit:]

    def clear_history(self) -> None:
        """Clear execution history."""
        self._execution_history.clear()

    def get_stats(self) -> Dict[str, Any]:
        """Get tool usage statistics."""
        stats = {
            "total_tools": len(self._tools),
            "enabled_tools": sum(1 for t in self._tools.values() if t.enabled),
            "total_executions": len(self._execution_history),
            "by_category": {},
        }

        for category in ToolCategory:
            tools = self.get_tools_by_category(category)
            stats["by_category"][category.value] = len(tools)

        return stats

    def get_tool_schema(self, name: str) -> Optional[Dict[str, Any]]:
        """Get schema for a specific tool.

        Args:
            name: Tool name

        Returns:
            Tool schema dict or None if not found
        """
        tool = self._tools.get(name)
        if tool:
            return tool.get_schema()
        return None


def tool(
    manager_or_name: Union[ToolManager, str, None] = None,
    description: str = None,
    category: ToolCategory = ToolCategory.UTILITY,
    requires_confirmation: bool = False,
    name: str = None,
):
    """Decorator to register a function as a tool.

    Usage:
        # Without auto-registration (stores metadata only)
        @tool(name="my_tool", description="Does something")
        def my_tool(arg1: str, arg2: int = 10) -> str:
            return f"Result: {arg1}, {arg2}"

        # With auto-registration to a manager
        @tool(manager, description="Does something")
        def my_tool(arg1: str, arg2: int = 10) -> str:
            return f"Result: {arg1}, {arg2}"
    """
    # Handle case where manager is passed as first arg
    manager = None
    tool_name = name
    if isinstance(manager_or_name, ToolManager):
        manager = manager_or_name
    elif isinstance(manager_or_name, str):
        tool_name = manager_or_name

    def decorator(func: Callable) -> Callable:
        # Extract parameters from function signature
        sig = inspect.signature(func)
        parameters = []

        for param_name, param in sig.parameters.items():
            if param_name in ('self', 'cls'):
                continue

            param_type = param.annotation if param.annotation != inspect.Parameter.empty else str
            required = param.default == inspect.Parameter.empty
            default = None if required else param.default

            parameters.append(ToolParameter(
                name=param_name,
                param_type=param_type,
                description=f"Parameter: {param_name}",
                required=required,
                default=default,
            ))

        final_name = tool_name or func.__name__
        final_description = description or func.__doc__ or f"Tool: {func.__name__}"

        # Store tool metadata on function
        func._tool_metadata = {
            "name": final_name,
            "description": final_description,
            "category": category,
            "parameters": parameters,
            "requires_confirmation": requires_confirmation,
        }

        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        wrapper._tool_metadata = func._tool_metadata

        # Auto-register if manager provided
        if manager is not None:
            manager.register(
                name=final_name,
                handler=wrapper,
                description=final_description,
                category=category,
                parameters=parameters,
                requires_confirmation=requires_confirmation,
            )

        return wrapper

    return decorator


def register_decorated_tools(manager: ToolManager, module) -> int:
    """Register all @tool decorated functions from a module.

    Args:
        manager: ToolManager instance
        module: Module to scan for decorated functions

    Returns:
        Number of tools registered
    """
    count = 0

    for name in dir(module):
        obj = getattr(module, name)
        if callable(obj) and hasattr(obj, '_tool_metadata'):
            meta = obj._tool_metadata
            manager.register(
                name=meta["name"],
                handler=obj,
                description=meta["description"],
                category=meta["category"],
                parameters=meta["parameters"],
                requires_confirmation=meta["requires_confirmation"],
            )
            count += 1

    return count


# Global tool manager instance
_global_manager: Optional[ToolManager] = None


def get_tool_manager() -> ToolManager:
    """Get or create the global tool manager."""
    global _global_manager
    if _global_manager is None:
        _global_manager = ToolManager()
    return _global_manager


def reset_tool_manager() -> None:
    """Reset the global tool manager."""
    global _global_manager
    _global_manager = None
