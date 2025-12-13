"""Internal event bus for pub/sub messaging across the orchestrator."""

from __future__ import annotations

import asyncio
import logging
from typing import Callable, Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from abc import ABC, abstractmethod
import threading
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import inspect

logger = logging.getLogger(__name__)


class EventType(Enum):
    """Types of events that can be published."""
    TASK_CREATED = "task_created"
    TASK_ASSIGNED = "task_assigned"
    TASK_STARTED = "task_started"
    TASK_COMPLETED = "task_completed"
    TASK_FAILED = "task_failed"
    TASK_RETRY = "task_retry"
    AGENT_SPAWNED = "agent_spawned"
    AGENT_BLOCKED = "agent_blocked"
    AGENT_UNBLOCKED = "agent_unblocked"
    FILE_CREATED = "file_created"
    FILE_MODIFIED = "file_modified"
    FILE_DELETED = "file_deleted"
    LOG_MESSAGE = "log_message"
    ERROR_OCCURRED = "error_occurred"
    MEMORY_UPDATED = "memory_updated"


@dataclass
class Event:
    """Base event object with metadata."""
    event_type: EventType
    timestamp: datetime = field(default_factory=datetime.now)
    agent_id: Optional[str] = None
    task_id: Optional[str] = None
    data: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize event to dict."""
        return {
            "event_type": self.event_type.value,
            "timestamp": self.timestamp.isoformat(),
            "agent_id": self.agent_id,
            "task_id": self.task_id,
            "data": self.data
        }


class EventHandler(ABC):
    """Base class for event handlers."""
    
    @abstractmethod
    async def handle(self, event: Event) -> None:
        """Handle an event."""
        pass


class EventBus:
    """Internal pub/sub event bus for orchestrator communication.

    Features:
      - Subscribe/publish pattern
      - Synchronous and asynchronous handlers (properly awaited)
      - Thread-safe operations with fine-grained locking
      - Efficient ring buffer event history
      - Filtering and priority handling
    """

    def __init__(self, max_history: int = 1000, async_executor_workers: int = 4):
        """Initialize the event bus.

        Args:
            max_history: Maximum events to keep in history
            async_executor_workers: Number of workers for async handler execution
        """
        self.max_history = max_history
        self._subscribers: Dict[EventType, List[Callable]] = {}
        # Use deque for O(1) append and automatic size limiting
        self._event_history: deque = deque(maxlen=max_history)
        self._lock = threading.RLock()
        self._subscriber_lock = threading.RLock()
        # Executor for running async handlers
        self._async_executor = ThreadPoolExecutor(max_workers=async_executor_workers)
        # Track if we have an event loop available
        self._event_loop: Optional[asyncio.AbstractEventLoop] = None

    def subscribe(self, event_type: EventType, handler: Callable) -> None:
        """Subscribe to events of a specific type.

        Args:
            event_type: Type of event to subscribe to
            handler: Callable(event) to handle the event (sync or async)
        """
        with self._subscriber_lock:
            if event_type not in self._subscribers:
                self._subscribers[event_type] = []
            if handler not in self._subscribers[event_type]:
                self._subscribers[event_type].append(handler)
                logger.debug(f"Handler subscribed to {event_type.value}")

    def unsubscribe(self, event_type: EventType, handler: Callable) -> None:
        """Unsubscribe a handler.

        Args:
            event_type: Type of event
            handler: Handler to remove
        """
        with self._subscriber_lock:
            if event_type in self._subscribers:
                try:
                    self._subscribers[event_type].remove(handler)
                    logger.debug(f"Handler unsubscribed from {event_type.value}")
                except ValueError:
                    pass

    def _run_async_handler(self, handler: Callable, event: Event) -> None:
        """Run an async handler in a new event loop.

        Args:
            handler: Async handler function
            event: Event to pass to handler
        """
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(handler(event))
            finally:
                loop.close()
        except Exception as exc:
            logger.exception(f"Error in async event handler: {exc}")

    def publish(self, event: Event) -> None:
        """Publish an event to all subscribers.

        Args:
            event: Event to publish
        """
        # Store in history (deque handles max size automatically)
        with self._lock:
            self._event_history.append(event)

        # Get a copy of handlers to avoid holding lock during execution
        with self._subscriber_lock:
            handlers = list(self._subscribers.get(event.event_type, []))

        # Call handlers outside lock to avoid deadlock
        for handler in handlers:
            try:
                if inspect.iscoroutinefunction(handler):
                    # Run async handlers in thread pool with their own event loop
                    self._async_executor.submit(self._run_async_handler, handler, event)
                    logger.debug(f"Async handler submitted for {event.event_type.value}")
                else:
                    handler(event)
            except Exception as exc:
                logger.exception(f"Error in event handler: {exc}")

    def get_event_history(
        self,
        event_type: Optional[EventType] = None,
        agent_id: Optional[str] = None,
        task_id: Optional[str] = None,
        limit: Optional[int] = None
    ) -> List[Event]:
        """Get event history with optional filtering.

        Args:
            event_type: Filter by event type
            agent_id: Filter by agent ID
            task_id: Filter by task ID
            limit: Maximum number of events to return

        Returns:
            List of matching events
        """
        with self._lock:
            # Convert deque to list for filtering
            events = list(self._event_history)

        if event_type:
            events = [e for e in events if e.event_type == event_type]
        if agent_id:
            events = [e for e in events if e.agent_id == agent_id]
        if task_id:
            events = [e for e in events if e.task_id == task_id]

        if limit:
            events = events[-limit:]

        return events

    def clear_history(self) -> None:
        """Clear event history."""
        with self._lock:
            self._event_history.clear()

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the event bus.

        Returns:
            Dict with stats
        """
        with self._lock:
            events_snapshot = list(self._event_history)

        with self._subscriber_lock:
            subscriber_counts = {
                et.value: len(handlers)
                for et, handlers in self._subscribers.items()
                if handlers
            }

        event_counts = {}
        for event in events_snapshot:
            event_type = event.event_type.value
            event_counts[event_type] = event_counts.get(event_type, 0) + 1

        return {
            "total_events": len(events_snapshot),
            "event_counts": event_counts,
            "subscribers": subscriber_counts
        }

    def shutdown(self) -> None:
        """Shutdown the event bus and cleanup resources."""
        self._async_executor.shutdown(wait=True)
        logger.info("EventBus shutdown complete")


# Singleton instance with thread-safe initialization
_event_bus: Optional[EventBus] = None
_event_bus_lock = threading.Lock()


def get_event_bus() -> EventBus:
    """Get the singleton event bus instance (thread-safe double-checked locking)."""
    global _event_bus
    if _event_bus is None:
        with _event_bus_lock:
            # Double-check after acquiring lock
            if _event_bus is None:
                _event_bus = EventBus()
    return _event_bus


def reset_event_bus() -> None:
    """Reset the singleton event bus (useful for testing)."""
    global _event_bus
    with _event_bus_lock:
        if _event_bus is not None:
            _event_bus.shutdown()
            _event_bus = None


def publish_event(
    event_type: EventType,
    agent_id: Optional[str] = None,
    task_id: Optional[str] = None,
    data: Optional[Dict[str, Any]] = None
) -> None:
    """Publish an event to the bus.
    
    Args:
        event_type: Type of event
        agent_id: Associated agent ID
        task_id: Associated task ID
        data: Additional event data
    """
    event = Event(
        event_type=event_type,
        agent_id=agent_id,
        task_id=task_id,
        data=data or {}
    )
    get_event_bus().publish(event)


def subscribe_to_events(event_type: EventType, handler: Callable) -> None:
    """Subscribe to events.
    
    Args:
        event_type: Type of event to subscribe to
        handler: Handler function
    """
    get_event_bus().subscribe(event_type, handler)
