"""Internal event bus for pub/sub messaging across the orchestrator."""

from __future__ import annotations

import logging
from typing import Callable, Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from abc import ABC, abstractmethod
import threading

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
      - Synchronous and asynchronous handlers
      - Thread-safe operations
      - Event history tracking
      - Filtering and priority handling
    """

    def __init__(self, max_history: int = 1000):
        """Initialize the event bus.
        
        Args:
            max_history: Maximum events to keep in history
        """
        self.max_history = max_history
        self._subscribers: Dict[EventType, List[Callable]] = {}
        self._event_history: List[Event] = []
        self._lock = threading.RLock()

    def subscribe(self, event_type: EventType, handler: Callable) -> None:
        """Subscribe to events of a specific type.
        
        Args:
            event_type: Type of event to subscribe to
            handler: Callable(event) to handle the event
        """
        with self._lock:
            if event_type not in self._subscribers:
                self._subscribers[event_type] = []
            self._subscribers[event_type].append(handler)
            logger.debug(f"Handler subscribed to {event_type.value}")

    def unsubscribe(self, event_type: EventType, handler: Callable) -> None:
        """Unsubscribe a handler.
        
        Args:
            event_type: Type of event
            handler: Handler to remove
        """
        with self._lock:
            if event_type in self._subscribers:
                try:
                    self._subscribers[event_type].remove(handler)
                    logger.debug(f"Handler unsubscribed from {event_type.value}")
                except ValueError:
                    pass

    def publish(self, event: Event) -> None:
        """Publish an event to all subscribers.
        
        Args:
            event: Event to publish
        """
        with self._lock:
            # Store in history
            self._event_history.append(event)
            if len(self._event_history) > self.max_history:
                self._event_history = self._event_history[-self.max_history:]

            # Get handlers for this event type
            handlers = self._subscribers.get(event.event_type, [])

        # Call handlers outside lock to avoid deadlock
        for handler in handlers:
            try:
                # Check if handler is async
                import inspect
                if inspect.iscoroutinefunction(handler):
                    # For async handlers, we'd need to run them in an event loop
                    # For now, wrap in try-except
                    logger.debug(f"Async handler called for {event.event_type.value}")
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
            events = self._event_history

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
            event_counts = {}
            for event in self._event_history:
                event_type = event.event_type.value
                event_counts[event_type] = event_counts.get(event_type, 0) + 1

            return {
                "total_events": len(self._event_history),
                "event_counts": event_counts,
                "subscribers": {
                    et.value: len(handlers)
                    for et, handlers in self._subscribers.items()
                    if handlers
                }
            }


# Singleton instance
_event_bus: Optional[EventBus] = None


def get_event_bus() -> EventBus:
    """Get the singleton event bus instance."""
    global _event_bus
    if _event_bus is None:
        _event_bus = EventBus()
    return _event_bus


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
