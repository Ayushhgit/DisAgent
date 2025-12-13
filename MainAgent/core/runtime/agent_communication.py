"""
Agent Communication Protocol - Point-to-point and group messaging for agents.

Provides:
- Direct agent-to-agent messaging
- Channel-based group communication
- Request-response patterns
- Async message handling
- Message persistence and replay

This addresses a key research gap: agents could only communicate via
global events, lacking the ability to directly request information
or coordinate with specific agents.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
import uuid
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Set,
    Union,
)
from queue import Queue, Empty
import json

logger = logging.getLogger(__name__)


class MessageType(Enum):
    """Types of inter-agent messages."""
    # Point-to-point
    DIRECT = "direct"
    REQUEST = "request"
    RESPONSE = "response"

    # Group/broadcast
    BROADCAST = "broadcast"
    CHANNEL = "channel"

    # System
    ACK = "ack"
    ERROR = "error"
    HEARTBEAT = "heartbeat"


class MessagePriority(Enum):
    """Message priority levels."""
    LOW = 0
    NORMAL = 1
    HIGH = 2
    URGENT = 3


@dataclass
class Message:
    """Inter-agent message."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    type: MessageType = MessageType.DIRECT
    sender_id: str = ""
    recipient_id: str = ""  # Empty for broadcast
    channel: str = ""  # For channel-based messages
    subject: str = ""
    body: Any = None
    priority: MessagePriority = MessagePriority.NORMAL
    timestamp: float = field(default_factory=time.time)
    correlation_id: str = ""  # For request-response correlation
    reply_to: str = ""  # Where to send responses
    ttl_seconds: int = 300  # Time to live
    metadata: Dict[str, Any] = field(default_factory=dict)

    def is_expired(self) -> bool:
        """Check if message has expired."""
        return time.time() - self.timestamp > self.ttl_seconds

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "id": self.id,
            "type": self.type.value,
            "sender_id": self.sender_id,
            "recipient_id": self.recipient_id,
            "channel": self.channel,
            "subject": self.subject,
            "body": self.body,
            "priority": self.priority.value,
            "timestamp": self.timestamp,
            "correlation_id": self.correlation_id,
            "reply_to": self.reply_to,
            "ttl_seconds": self.ttl_seconds,
            "metadata": self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Message":
        """Deserialize from dictionary."""
        return cls(
            id=data.get("id", str(uuid.uuid4())),
            type=MessageType(data.get("type", "direct")),
            sender_id=data.get("sender_id", ""),
            recipient_id=data.get("recipient_id", ""),
            channel=data.get("channel", ""),
            subject=data.get("subject", ""),
            body=data.get("body"),
            priority=MessagePriority(data.get("priority", 1)),
            timestamp=data.get("timestamp", time.time()),
            correlation_id=data.get("correlation_id", ""),
            reply_to=data.get("reply_to", ""),
            ttl_seconds=data.get("ttl_seconds", 300),
            metadata=data.get("metadata", {})
        )


@dataclass
class MessageDeliveryResult:
    """Result of message delivery attempt."""
    success: bool
    message_id: str
    recipient_id: str
    error: str = ""
    delivered_at: float = 0.0


class MessageHandler(ABC):
    """Abstract base class for message handlers."""

    @abstractmethod
    def handle(self, message: Message) -> Optional[Any]:
        """Handle a received message.

        Args:
            message: The message to handle

        Returns:
            Optional response for request-type messages
        """
        pass


class AgentMailbox:
    """Message queue for a single agent."""

    def __init__(self, agent_id: str, max_size: int = 1000):
        self.agent_id = agent_id
        self.max_size = max_size
        self._queue: Queue = Queue(maxsize=max_size)
        self._handlers: Dict[str, MessageHandler] = {}
        self._default_handler: Optional[Callable[[Message], Any]] = None
        self._lock = threading.Lock()
        self._unread_count = 0

    def receive(self, message: Message) -> bool:
        """Receive a message into the mailbox."""
        try:
            if message.is_expired():
                logger.debug(f"Dropping expired message {message.id}")
                return False

            self._queue.put_nowait(message)
            with self._lock:
                self._unread_count += 1
            return True
        except Exception:
            logger.warning(f"Mailbox full for agent {self.agent_id}")
            return False

    def get(self, timeout: float = 0.0) -> Optional[Message]:
        """Get next message from mailbox."""
        try:
            message = self._queue.get(timeout=timeout if timeout > 0 else None)
            with self._lock:
                self._unread_count = max(0, self._unread_count - 1)
            return message
        except Empty:
            return None

    def get_all(self) -> List[Message]:
        """Get all pending messages."""
        messages = []
        while True:
            msg = self.get(timeout=0.01)
            if msg is None:
                break
            messages.append(msg)
        return messages

    def register_handler(self, subject: str, handler: MessageHandler) -> None:
        """Register handler for specific subject."""
        self._handlers[subject] = handler

    def set_default_handler(self, handler: Callable[[Message], Any]) -> None:
        """Set default handler for unmatched messages."""
        self._default_handler = handler

    def process_messages(self) -> List[Any]:
        """Process all pending messages with registered handlers."""
        results = []
        messages = self.get_all()

        for message in messages:
            handler = self._handlers.get(message.subject)
            if handler:
                result = handler.handle(message)
            elif self._default_handler:
                result = self._default_handler(message)
            else:
                result = None
            results.append(result)

        return results

    @property
    def unread_count(self) -> int:
        with self._lock:
            return self._unread_count

    @property
    def is_empty(self) -> bool:
        return self._queue.empty()


class Channel:
    """Pub/sub channel for group communication."""

    def __init__(self, name: str, persistent: bool = False):
        self.name = name
        self.persistent = persistent
        self._subscribers: Set[str] = set()
        self._message_history: List[Message] = []
        self._max_history = 100
        self._lock = threading.Lock()

    def subscribe(self, agent_id: str) -> None:
        """Subscribe an agent to the channel."""
        with self._lock:
            self._subscribers.add(agent_id)
            logger.debug(f"Agent {agent_id} subscribed to channel {self.name}")

    def unsubscribe(self, agent_id: str) -> None:
        """Unsubscribe an agent from the channel."""
        with self._lock:
            self._subscribers.discard(agent_id)
            logger.debug(f"Agent {agent_id} unsubscribed from channel {self.name}")

    def get_subscribers(self) -> Set[str]:
        """Get set of subscriber IDs."""
        with self._lock:
            return self._subscribers.copy()

    def add_message(self, message: Message) -> None:
        """Add message to channel history."""
        if self.persistent:
            with self._lock:
                self._message_history.append(message)
                if len(self._message_history) > self._max_history:
                    self._message_history = self._message_history[-self._max_history:]

    def get_history(self, limit: int = 50) -> List[Message]:
        """Get recent message history."""
        with self._lock:
            return self._message_history[-limit:]


class AgentCommunicationBus:
    """Central communication bus for agent messaging.

    Features:
    - Point-to-point messaging
    - Request-response pattern
    - Channel-based pub/sub
    - Message persistence
    - Delivery tracking
    """

    def __init__(self):
        self._mailboxes: Dict[str, AgentMailbox] = {}
        self._channels: Dict[str, Channel] = {}
        self._pending_requests: Dict[str, Dict[str, Any]] = {}  # correlation_id -> request info
        self._lock = threading.Lock()
        self._delivery_log: List[MessageDeliveryResult] = []

        # Create default channels
        self._create_default_channels()

    def _create_default_channels(self):
        """Create standard channels."""
        default_channels = [
            ("system", True),  # System messages
            ("coordination", True),  # Task coordination
            ("status", False),  # Status updates
            ("errors", True),  # Error reports
        ]
        for name, persistent in default_channels:
            self._channels[name] = Channel(name, persistent)

    def register_agent(self, agent_id: str) -> AgentMailbox:
        """Register an agent and create its mailbox."""
        with self._lock:
            if agent_id not in self._mailboxes:
                self._mailboxes[agent_id] = AgentMailbox(agent_id)
                logger.info(f"Registered agent: {agent_id}")
            return self._mailboxes[agent_id]

    def unregister_agent(self, agent_id: str) -> None:
        """Unregister an agent."""
        with self._lock:
            if agent_id in self._mailboxes:
                del self._mailboxes[agent_id]
                # Remove from all channels
                for channel in self._channels.values():
                    channel.unsubscribe(agent_id)
                logger.info(f"Unregistered agent: {agent_id}")

    def get_mailbox(self, agent_id: str) -> Optional[AgentMailbox]:
        """Get an agent's mailbox."""
        return self._mailboxes.get(agent_id)

    def send(
        self,
        sender_id: str,
        recipient_id: str,
        subject: str,
        body: Any,
        priority: MessagePriority = MessagePriority.NORMAL,
        ttl_seconds: int = 300
    ) -> MessageDeliveryResult:
        """Send a direct message to another agent."""
        message = Message(
            type=MessageType.DIRECT,
            sender_id=sender_id,
            recipient_id=recipient_id,
            subject=subject,
            body=body,
            priority=priority,
            ttl_seconds=ttl_seconds
        )
        return self._deliver(message)

    def request(
        self,
        sender_id: str,
        recipient_id: str,
        subject: str,
        body: Any,
        timeout: float = 30.0,
        priority: MessagePriority = MessagePriority.HIGH
    ) -> Optional[Message]:
        """Send a request and wait for response.

        Args:
            sender_id: Sender agent ID
            recipient_id: Recipient agent ID
            subject: Request subject
            body: Request body
            timeout: Seconds to wait for response
            priority: Message priority

        Returns:
            Response message or None if timeout
        """
        correlation_id = str(uuid.uuid4())

        message = Message(
            type=MessageType.REQUEST,
            sender_id=sender_id,
            recipient_id=recipient_id,
            subject=subject,
            body=body,
            priority=priority,
            correlation_id=correlation_id,
            reply_to=sender_id,
            ttl_seconds=int(timeout)
        )

        # Set up response tracking
        response_event = threading.Event()
        response_holder: Dict[str, Optional[Message]] = {"response": None}

        with self._lock:
            self._pending_requests[correlation_id] = {
                "event": response_event,
                "response": response_holder,
                "timestamp": time.time(),
                "timeout": timeout
            }

        # Deliver request
        result = self._deliver(message)
        if not result.success:
            with self._lock:
                del self._pending_requests[correlation_id]
            return None

        # Wait for response
        if response_event.wait(timeout=timeout):
            return response_holder.get("response")

        # Timeout
        with self._lock:
            if correlation_id in self._pending_requests:
                del self._pending_requests[correlation_id]

        logger.warning(f"Request {correlation_id} timed out")
        return None

    def respond(
        self,
        original_message: Message,
        body: Any
    ) -> MessageDeliveryResult:
        """Send a response to a request."""
        response = Message(
            type=MessageType.RESPONSE,
            sender_id=original_message.recipient_id,
            recipient_id=original_message.reply_to or original_message.sender_id,
            subject=f"RE: {original_message.subject}",
            body=body,
            correlation_id=original_message.correlation_id
        )
        return self._deliver(response)

    def broadcast(
        self,
        sender_id: str,
        subject: str,
        body: Any,
        exclude: Optional[Set[str]] = None
    ) -> List[MessageDeliveryResult]:
        """Broadcast message to all agents."""
        results = []
        exclude = exclude or set()
        exclude.add(sender_id)  # Don't send to self

        with self._lock:
            recipients = [
                aid for aid in self._mailboxes.keys()
                if aid not in exclude
            ]

        for recipient_id in recipients:
            message = Message(
                type=MessageType.BROADCAST,
                sender_id=sender_id,
                recipient_id=recipient_id,
                subject=subject,
                body=body
            )
            results.append(self._deliver(message))

        return results

    def publish(
        self,
        sender_id: str,
        channel_name: str,
        subject: str,
        body: Any
    ) -> List[MessageDeliveryResult]:
        """Publish message to a channel."""
        channel = self._channels.get(channel_name)
        if not channel:
            logger.warning(f"Channel {channel_name} does not exist")
            return []

        results = []
        message = Message(
            type=MessageType.CHANNEL,
            sender_id=sender_id,
            channel=channel_name,
            subject=subject,
            body=body
        )

        # Store in channel history
        channel.add_message(message)

        # Deliver to all subscribers
        for subscriber_id in channel.get_subscribers():
            if subscriber_id != sender_id:
                msg_copy = Message(
                    type=MessageType.CHANNEL,
                    sender_id=sender_id,
                    recipient_id=subscriber_id,
                    channel=channel_name,
                    subject=subject,
                    body=body
                )
                results.append(self._deliver(msg_copy))

        return results

    def subscribe(self, agent_id: str, channel_name: str) -> bool:
        """Subscribe agent to a channel."""
        if channel_name not in self._channels:
            self._channels[channel_name] = Channel(channel_name)

        self._channels[channel_name].subscribe(agent_id)
        return True

    def unsubscribe(self, agent_id: str, channel_name: str) -> bool:
        """Unsubscribe agent from a channel."""
        if channel_name in self._channels:
            self._channels[channel_name].unsubscribe(agent_id)
            return True
        return False

    def create_channel(self, name: str, persistent: bool = False) -> Channel:
        """Create a new channel."""
        if name not in self._channels:
            self._channels[name] = Channel(name, persistent)
        return self._channels[name]

    def get_channel_history(self, channel_name: str, limit: int = 50) -> List[Message]:
        """Get message history for a channel."""
        channel = self._channels.get(channel_name)
        if channel:
            return channel.get_history(limit)
        return []

    def _deliver(self, message: Message) -> MessageDeliveryResult:
        """Internal message delivery."""
        recipient_id = message.recipient_id
        mailbox = self._mailboxes.get(recipient_id)

        if not mailbox:
            result = MessageDeliveryResult(
                success=False,
                message_id=message.id,
                recipient_id=recipient_id,
                error=f"Agent {recipient_id} not registered"
            )
            self._log_delivery(result)
            return result

        success = mailbox.receive(message)

        result = MessageDeliveryResult(
            success=success,
            message_id=message.id,
            recipient_id=recipient_id,
            error="" if success else "Mailbox full or message expired",
            delivered_at=time.time() if success else 0.0
        )

        self._log_delivery(result)

        # Handle response messages
        if message.type == MessageType.RESPONSE and message.correlation_id:
            self._handle_response(message)

        return result

    def _handle_response(self, message: Message) -> None:
        """Handle incoming response message."""
        with self._lock:
            request_info = self._pending_requests.get(message.correlation_id)
            if request_info:
                request_info["response"]["response"] = message
                request_info["event"].set()
                del self._pending_requests[message.correlation_id]

    def _log_delivery(self, result: MessageDeliveryResult) -> None:
        """Log delivery result."""
        with self._lock:
            self._delivery_log.append(result)
            # Keep only recent entries
            if len(self._delivery_log) > 1000:
                self._delivery_log = self._delivery_log[-500:]

    def get_stats(self) -> Dict[str, Any]:
        """Get communication bus statistics."""
        with self._lock:
            return {
                "registered_agents": len(self._mailboxes),
                "channels": len(self._channels),
                "pending_requests": len(self._pending_requests),
                "total_deliveries": len(self._delivery_log),
                "successful_deliveries": sum(
                    1 for r in self._delivery_log if r.success
                ),
                "failed_deliveries": sum(
                    1 for r in self._delivery_log if not r.success
                )
            }

    def cleanup_expired(self) -> int:
        """Clean up expired requests and messages."""
        cleaned = 0
        current_time = time.time()

        with self._lock:
            expired_requests = [
                cid for cid, info in self._pending_requests.items()
                if current_time - info["timestamp"] > info["timeout"]
            ]
            for cid in expired_requests:
                del self._pending_requests[cid]
                cleaned += 1

        return cleaned


# Singleton instance
_communication_bus: Optional[AgentCommunicationBus] = None
_bus_lock = threading.Lock()


def get_communication_bus() -> AgentCommunicationBus:
    """Get singleton communication bus instance."""
    global _communication_bus
    if _communication_bus is None:
        with _bus_lock:
            if _communication_bus is None:
                _communication_bus = AgentCommunicationBus()
    return _communication_bus


def reset_communication_bus() -> None:
    """Reset the communication bus (for testing)."""
    global _communication_bus
    with _bus_lock:
        _communication_bus = None


# Convenience functions for agents
class AgentCommunicator:
    """Helper class for agent communication.

    Wraps the communication bus with agent-specific methods.
    """

    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self._bus = get_communication_bus()
        self._mailbox = self._bus.register_agent(agent_id)

    def send(
        self,
        recipient_id: str,
        subject: str,
        body: Any,
        priority: MessagePriority = MessagePriority.NORMAL
    ) -> bool:
        """Send a direct message."""
        result = self._bus.send(
            self.agent_id,
            recipient_id,
            subject,
            body,
            priority
        )
        return result.success

    def request(
        self,
        recipient_id: str,
        subject: str,
        body: Any,
        timeout: float = 30.0
    ) -> Optional[Any]:
        """Send request and get response body."""
        response = self._bus.request(
            self.agent_id,
            recipient_id,
            subject,
            body,
            timeout
        )
        return response.body if response else None

    def broadcast(self, subject: str, body: Any) -> int:
        """Broadcast to all agents."""
        results = self._bus.broadcast(self.agent_id, subject, body)
        return sum(1 for r in results if r.success)

    def subscribe(self, channel: str) -> None:
        """Subscribe to a channel."""
        self._bus.subscribe(self.agent_id, channel)

    def publish(self, channel: str, subject: str, body: Any) -> int:
        """Publish to a channel."""
        results = self._bus.publish(self.agent_id, channel, subject, body)
        return sum(1 for r in results if r.success)

    def receive(self, timeout: float = 0.0) -> Optional[Message]:
        """Receive next message."""
        return self._mailbox.get(timeout)

    def receive_all(self) -> List[Message]:
        """Receive all pending messages."""
        return self._mailbox.get_all()

    def has_messages(self) -> bool:
        """Check if there are pending messages."""
        return not self._mailbox.is_empty

    def unread_count(self) -> int:
        """Get count of unread messages."""
        return self._mailbox.unread_count

    def register_handler(self, subject: str, handler: Callable[[Message], Any]) -> None:
        """Register message handler for subject."""
        class CallableHandler(MessageHandler):
            def __init__(self, fn):
                self.fn = fn

            def handle(self, message: Message) -> Any:
                return self.fn(message)

        self._mailbox.register_handler(subject, CallableHandler(handler))

    def process_messages(self) -> List[Any]:
        """Process all pending messages with handlers."""
        return self._mailbox.process_messages()

    def cleanup(self) -> None:
        """Cleanup and unregister agent."""
        self._bus.unregister_agent(self.agent_id)
