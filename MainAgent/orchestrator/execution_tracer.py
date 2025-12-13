"""
Execution Tracer - Structured observability for orchestration.

Provides:
- Structured execution traces with parent-child relationships
- Decision audit trail
- Cost tracking (tokens, time, API calls)
- Performance metrics and bottleneck identification
- Export to various formats (JSON, OpenTelemetry-compatible)

This addresses a key gap: existing systems log events but lack
structured traces for debugging and optimization.
"""

from __future__ import annotations

import json
import logging
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Union
import sys

logger = logging.getLogger(__name__)


class SpanKind(Enum):
    """Type of trace span."""
    ORCHESTRATION = "orchestration"
    PLANNING = "planning"
    AGENT_EXECUTION = "agent_execution"
    TOOL_CALL = "tool_call"
    LLM_CALL = "llm_call"
    FILE_OPERATION = "file_operation"
    VERIFICATION = "verification"
    COMMUNICATION = "communication"


class SpanStatus(Enum):
    """Status of a span."""
    UNSET = "unset"
    OK = "ok"
    ERROR = "error"


@dataclass
class SpanEvent:
    """Event within a span."""
    name: str
    timestamp: float
    attributes: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SpanContext:
    """Context for span correlation."""
    trace_id: str
    span_id: str
    parent_span_id: Optional[str] = None


@dataclass
class Span:
    """A single execution span in the trace."""
    name: str
    kind: SpanKind
    context: SpanContext
    start_time: float
    end_time: float = 0.0
    status: SpanStatus = SpanStatus.UNSET
    attributes: Dict[str, Any] = field(default_factory=dict)
    events: List[SpanEvent] = field(default_factory=list)
    error_message: str = ""

    @property
    def duration_ms(self) -> float:
        if self.end_time == 0:
            return (time.time() - self.start_time) * 1000
        return (self.end_time - self.start_time) * 1000

    @property
    def is_error(self) -> bool:
        return self.status == SpanStatus.ERROR

    def add_event(self, name: str, attributes: Optional[Dict[str, Any]] = None) -> None:
        """Add event to span."""
        self.events.append(SpanEvent(
            name=name,
            timestamp=time.time(),
            attributes=attributes or {}
        ))

    def set_attribute(self, key: str, value: Any) -> None:
        """Set span attribute."""
        self.attributes[key] = value

    def set_status(self, status: SpanStatus, message: str = "") -> None:
        """Set span status."""
        self.status = status
        if status == SpanStatus.ERROR:
            self.error_message = message

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "kind": self.kind.value,
            "trace_id": self.context.trace_id,
            "span_id": self.context.span_id,
            "parent_span_id": self.context.parent_span_id,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_ms": self.duration_ms,
            "status": self.status.value,
            "error_message": self.error_message,
            "attributes": self.attributes,
            "events": [
                {"name": e.name, "timestamp": e.timestamp, "attributes": e.attributes}
                for e in self.events
            ]
        }


@dataclass
class CostMetrics:
    """Cost tracking metrics."""
    llm_calls: int = 0
    llm_input_tokens: int = 0
    llm_output_tokens: int = 0
    api_calls: int = 0
    file_reads: int = 0
    file_writes: int = 0
    test_runs: int = 0

    @property
    def estimated_cost_usd(self) -> float:
        """Estimate cost based on token usage (rough Claude pricing)."""
        # Rough estimates: $3/M input, $15/M output
        input_cost = (self.llm_input_tokens / 1_000_000) * 3
        output_cost = (self.llm_output_tokens / 1_000_000) * 15
        return round(input_cost + output_cost, 4)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "llm_calls": self.llm_calls,
            "llm_input_tokens": self.llm_input_tokens,
            "llm_output_tokens": self.llm_output_tokens,
            "api_calls": self.api_calls,
            "file_reads": self.file_reads,
            "file_writes": self.file_writes,
            "test_runs": self.test_runs,
            "estimated_cost_usd": self.estimated_cost_usd
        }


@dataclass
class Decision:
    """Record of an autonomous decision."""
    decision_id: str
    timestamp: float
    decision_type: str  # agent_selection, task_ordering, retry, etc.
    context: str
    options_considered: List[str]
    chosen_option: str
    reasoning: str
    confidence: float = 0.0
    outcome: str = ""  # Set after execution


class ExecutionTracer:
    """Traces orchestration execution with structured spans.

    Features:
    - Hierarchical span tracking
    - Cost and performance metrics
    - Decision audit trail
    - Export to JSON or other formats
    """

    def __init__(
        self,
        trace_id: Optional[str] = None,
        storage_path: Optional[str] = None,
        enable_console_output: bool = True
    ):
        """Initialize tracer.

        Args:
            trace_id: Optional trace ID (generated if not provided)
            storage_path: Path to store traces
            enable_console_output: Whether to print span info to console
        """
        self.trace_id = trace_id or str(uuid.uuid4())
        self.storage_path = Path(storage_path) if storage_path else Path(".agent_memory/traces")
        self.enable_console = enable_console_output

        self._lock = threading.RLock()
        self._spans: List[Span] = []
        self._active_spans: Dict[str, Span] = {}  # span_id -> Span
        self._span_stack: List[str] = []  # Stack of active span IDs
        self._decisions: List[Decision] = []
        self._costs = CostMetrics()

        self._start_time = time.time()

    def _generate_span_id(self) -> str:
        """Generate unique span ID."""
        return str(uuid.uuid4())[:16]

    @contextmanager
    def span(
        self,
        name: str,
        kind: SpanKind = SpanKind.ORCHESTRATION,
        attributes: Optional[Dict[str, Any]] = None
    ) -> Generator[Span, None, None]:
        """Context manager for creating a span.

        Usage:
            with tracer.span("process_task", SpanKind.AGENT_EXECUTION) as span:
                span.set_attribute("task_id", task.id)
                # ... do work ...
                span.add_event("checkpoint", {"progress": 50})
        """
        span = self.start_span(name, kind, attributes)
        try:
            yield span
            if span.status == SpanStatus.UNSET:
                span.set_status(SpanStatus.OK)
        except Exception as e:
            span.set_status(SpanStatus.ERROR, str(e))
            raise
        finally:
            self.end_span(span.context.span_id)

    def start_span(
        self,
        name: str,
        kind: SpanKind = SpanKind.ORCHESTRATION,
        attributes: Optional[Dict[str, Any]] = None
    ) -> Span:
        """Start a new span."""
        span_id = self._generate_span_id()

        with self._lock:
            parent_span_id = self._span_stack[-1] if self._span_stack else None

            context = SpanContext(
                trace_id=self.trace_id,
                span_id=span_id,
                parent_span_id=parent_span_id
            )

            span = Span(
                name=name,
                kind=kind,
                context=context,
                start_time=time.time(),
                attributes=attributes or {}
            )

            self._active_spans[span_id] = span
            self._span_stack.append(span_id)

            if self.enable_console:
                indent = "  " * (len(self._span_stack) - 1)
                print(f"{indent}[TRACE] → {name} ({kind.value})")

        return span

    def end_span(self, span_id: str) -> None:
        """End a span."""
        with self._lock:
            if span_id not in self._active_spans:
                logger.warning(f"Span {span_id} not found")
                return

            span = self._active_spans[span_id]
            span.end_time = time.time()

            if span.status == SpanStatus.UNSET:
                span.set_status(SpanStatus.OK)

            self._spans.append(span)
            del self._active_spans[span_id]

            if self._span_stack and self._span_stack[-1] == span_id:
                self._span_stack.pop()

            if self.enable_console:
                indent = "  " * len(self._span_stack)
                status_icon = "✓" if span.status == SpanStatus.OK else "✗"
                print(f"{indent}[TRACE] ← {span.name} [{status_icon}] ({span.duration_ms:.1f}ms)")

    def record_decision(
        self,
        decision_type: str,
        context: str,
        options: List[str],
        chosen: str,
        reasoning: str,
        confidence: float = 0.0
    ) -> str:
        """Record an autonomous decision.

        Args:
            decision_type: Type of decision
            context: Context for the decision
            options: Options that were considered
            chosen: The chosen option
            reasoning: Why this option was chosen
            confidence: Confidence in the decision

        Returns:
            Decision ID for later outcome recording
        """
        decision_id = str(uuid.uuid4())[:12]

        decision = Decision(
            decision_id=decision_id,
            timestamp=time.time(),
            decision_type=decision_type,
            context=context,
            options_considered=options,
            chosen_option=chosen,
            reasoning=reasoning,
            confidence=confidence
        )

        with self._lock:
            self._decisions.append(decision)

        logger.debug(f"Decision recorded: {decision_type} -> {chosen}")
        return decision_id

    def record_decision_outcome(
        self,
        decision_id: str,
        outcome: str
    ) -> None:
        """Record the outcome of a previous decision."""
        with self._lock:
            for decision in self._decisions:
                if decision.decision_id == decision_id:
                    decision.outcome = outcome
                    break

    def record_llm_call(
        self,
        input_tokens: int,
        output_tokens: int,
        model: str = ""
    ) -> None:
        """Record LLM call metrics."""
        with self._lock:
            self._costs.llm_calls += 1
            self._costs.llm_input_tokens += input_tokens
            self._costs.llm_output_tokens += output_tokens

        # Add event to current span if exists
        if self._span_stack:
            span = self._active_spans.get(self._span_stack[-1])
            if span:
                span.add_event("llm_call", {
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "model": model
                })

    def record_file_operation(self, operation: str) -> None:
        """Record file operation."""
        with self._lock:
            if operation in ("read", "get"):
                self._costs.file_reads += 1
            elif operation in ("write", "edit", "create"):
                self._costs.file_writes += 1

    def record_test_run(self) -> None:
        """Record test execution."""
        with self._lock:
            self._costs.test_runs += 1

    def get_current_span(self) -> Optional[Span]:
        """Get the currently active span."""
        with self._lock:
            if self._span_stack:
                return self._active_spans.get(self._span_stack[-1])
        return None

    def get_trace_summary(self) -> Dict[str, Any]:
        """Get summary of the trace."""
        with self._lock:
            all_spans = self._spans + list(self._active_spans.values())

            total_duration = time.time() - self._start_time

            # Calculate time by kind
            time_by_kind: Dict[str, float] = {}
            for span in all_spans:
                kind = span.kind.value
                if kind not in time_by_kind:
                    time_by_kind[kind] = 0
                time_by_kind[kind] += span.duration_ms

            # Find slowest spans
            sorted_spans = sorted(all_spans, key=lambda s: s.duration_ms, reverse=True)
            slowest = [
                {"name": s.name, "duration_ms": s.duration_ms, "kind": s.kind.value}
                for s in sorted_spans[:5]
            ]

            # Count errors
            errors = [s for s in all_spans if s.is_error]

            return {
                "trace_id": self.trace_id,
                "total_duration_seconds": round(total_duration, 2),
                "span_count": len(all_spans),
                "active_spans": len(self._active_spans),
                "error_count": len(errors),
                "decision_count": len(self._decisions),
                "time_by_kind_ms": time_by_kind,
                "slowest_spans": slowest,
                "costs": self._costs.to_dict(),
                "errors": [
                    {"name": s.name, "message": s.error_message}
                    for s in errors[:10]
                ]
            }

    def get_decision_audit(self) -> List[Dict[str, Any]]:
        """Get audit trail of all decisions."""
        with self._lock:
            return [
                {
                    "decision_id": d.decision_id,
                    "timestamp": d.timestamp,
                    "type": d.decision_type,
                    "context": d.context,
                    "options": d.options_considered,
                    "chosen": d.chosen_option,
                    "reasoning": d.reasoning,
                    "confidence": d.confidence,
                    "outcome": d.outcome
                }
                for d in self._decisions
            ]

    def export_trace(self, format: str = "json") -> str:
        """Export trace to specified format.

        Args:
            format: Export format ("json", "summary")

        Returns:
            Exported trace as string
        """
        with self._lock:
            all_spans = self._spans + list(self._active_spans.values())

            if format == "summary":
                return json.dumps(self.get_trace_summary(), indent=2)

            # Full JSON export
            data = {
                "trace_id": self.trace_id,
                "start_time": self._start_time,
                "spans": [s.to_dict() for s in all_spans],
                "decisions": self.get_decision_audit(),
                "costs": self._costs.to_dict(),
                "summary": self.get_trace_summary()
            }

            return json.dumps(data, indent=2)

    def save_trace(self) -> str:
        """Save trace to disk.

        Returns:
            Path to saved file
        """
        self.storage_path.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"trace_{self.trace_id[:8]}_{timestamp}.json"
        filepath = self.storage_path / filename

        with open(filepath, "w") as f:
            f.write(self.export_trace())

        logger.info(f"Trace saved to {filepath}")
        return str(filepath)

    def print_summary(self) -> None:
        """Print trace summary to console."""
        summary = self.get_trace_summary()

        print("\n" + "=" * 60)
        print("EXECUTION TRACE SUMMARY")
        print("=" * 60)
        print(f"Trace ID: {summary['trace_id']}")
        print(f"Duration: {summary['total_duration_seconds']}s")
        print(f"Spans: {summary['span_count']} ({summary['error_count']} errors)")
        print(f"Decisions: {summary['decision_count']}")

        print("\nTime Breakdown:")
        for kind, ms in sorted(summary['time_by_kind_ms'].items(), key=lambda x: -x[1]):
            print(f"  {kind}: {ms:.1f}ms")

        if summary['slowest_spans']:
            print("\nSlowest Operations:")
            for span in summary['slowest_spans']:
                print(f"  {span['name']}: {span['duration_ms']:.1f}ms")

        costs = summary['costs']
        print(f"\nCost Metrics:")
        print(f"  LLM Calls: {costs['llm_calls']}")
        print(f"  Tokens: {costs['llm_input_tokens']} in / {costs['llm_output_tokens']} out")
        print(f"  Est. Cost: ${costs['estimated_cost_usd']}")
        print(f"  File Ops: {costs['file_reads']} reads / {costs['file_writes']} writes")
        print(f"  Test Runs: {costs['test_runs']}")

        if summary['errors']:
            print(f"\nErrors ({len(summary['errors'])}):")
            for err in summary['errors']:
                print(f"  - {err['name']}: {err['message'][:50]}")

        print("=" * 60)


# Global tracer instance
_tracer: Optional[ExecutionTracer] = None
_tracer_lock = threading.Lock()


def get_tracer() -> ExecutionTracer:
    """Get global tracer instance."""
    global _tracer
    if _tracer is None:
        with _tracer_lock:
            if _tracer is None:
                _tracer = ExecutionTracer()
    return _tracer


def set_tracer(tracer: ExecutionTracer) -> None:
    """Set global tracer instance."""
    global _tracer
    with _tracer_lock:
        _tracer = tracer


def reset_tracer() -> None:
    """Reset global tracer."""
    global _tracer
    with _tracer_lock:
        _tracer = None


def create_tracer(
    trace_id: Optional[str] = None,
    storage_path: Optional[str] = None,
    enable_console: bool = True
) -> ExecutionTracer:
    """Factory function to create tracer."""
    return ExecutionTracer(trace_id, storage_path, enable_console)
