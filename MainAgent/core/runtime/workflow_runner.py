"""
Workflow Runner - Execute multi-step workflows with agent coordination.

Workflows are sequences of steps that can include:
- Agent execution
- Conditional branching
- Parallel execution
- Checkpoints and rollback
"""

from __future__ import annotations

import time
import logging
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger(__name__)


class StepType(Enum):
    """Types of workflow steps."""
    AGENT = "agent"
    PARALLEL = "parallel"
    CONDITIONAL = "conditional"
    CHECKPOINT = "checkpoint"
    WAIT = "wait"
    CUSTOM = "custom"


class StepStatus(Enum):
    """Status of a workflow step."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class WorkflowStep:
    """Represents a single step in a workflow."""
    name: str
    step_type: StepType
    config: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)
    status: StepStatus = StepStatus.PENDING
    result: Optional[Any] = None
    error: Optional[str] = None
    start_time: Optional[float] = None
    end_time: Optional[float] = None

    @property
    def duration(self) -> Optional[float]:
        if self.start_time and self.end_time:
            return round(self.end_time - self.start_time, 2)
        return None


@dataclass
class Workflow:
    """A workflow is a collection of steps to execute."""
    name: str
    steps: List[WorkflowStep] = field(default_factory=list)
    context: Dict[str, Any] = field(default_factory=dict)
    checkpoints: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def add_step(self, step: WorkflowStep) -> None:
        """Add a step to the workflow."""
        self.steps.append(step)

    def get_step(self, name: str) -> Optional[WorkflowStep]:
        """Get a step by name."""
        for step in self.steps:
            if step.name == name:
                return step
        return None


@dataclass
class WorkflowResult:
    """Result of a workflow execution."""
    success: bool
    workflow_name: str
    completed_steps: List[str]
    failed_steps: List[str]
    skipped_steps: List[str]
    total_duration: float
    context: Dict[str, Any]
    errors: List[str]


class WorkflowRunner:
    """Execute workflows with support for various step types and patterns."""

    def __init__(self, max_workers: int = 4):
        """Initialize the workflow runner.

        Args:
            max_workers: Maximum parallel workers for parallel steps
        """
        self.max_workers = max_workers
        self.step_handlers: Dict[StepType, Callable] = {
            StepType.AGENT: self._run_agent_step,
            StepType.PARALLEL: self._run_parallel_step,
            StepType.CONDITIONAL: self._run_conditional_step,
            StepType.CHECKPOINT: self._run_checkpoint_step,
            StepType.WAIT: self._run_wait_step,
            StepType.CUSTOM: self._run_custom_step,
        }

    def run(
        self,
        workflow: Workflow,
        context: Optional[Dict[str, Any]] = None,
        file_manager=None,
        agents: Optional[Dict[str, Any]] = None,
    ) -> WorkflowResult:
        """Execute a workflow.

        Args:
            workflow: The workflow to execute
            context: Optional initial context
            file_manager: File manager for agent steps
            agents: Dict of agent name -> agent instance

        Returns:
            WorkflowResult with execution details
        """
        start_time = time.time()
        errors = []

        # Initialize context
        workflow.context = context or {}
        workflow.context["_file_manager"] = file_manager
        workflow.context["_agents"] = agents or {}

        completed_steps = []
        failed_steps = []
        skipped_steps = []

        logger.info(f"Starting workflow: {workflow.name}")

        # Execute steps in order, respecting dependencies
        for step in workflow.steps:
            # Check dependencies
            if not self._dependencies_met(step, workflow):
                logger.info(f"Skipping step {step.name} - dependencies not met")
                step.status = StepStatus.SKIPPED
                skipped_steps.append(step.name)
                continue

            # Execute the step
            try:
                step.status = StepStatus.RUNNING
                step.start_time = time.time()

                handler = self.step_handlers.get(step.step_type)
                if handler:
                    step.result = handler(step, workflow)
                    step.status = StepStatus.COMPLETED
                    completed_steps.append(step.name)
                else:
                    raise ValueError(f"Unknown step type: {step.step_type}")

            except Exception as exc:
                logger.exception(f"Step {step.name} failed: {exc}")
                step.status = StepStatus.FAILED
                step.error = str(exc)
                failed_steps.append(step.name)
                errors.append(f"{step.name}: {exc}")

                # Check if we should abort on failure
                if step.config.get("abort_on_failure", True):
                    logger.error(f"Aborting workflow due to failure in {step.name}")
                    break

            finally:
                step.end_time = time.time()

        total_duration = round(time.time() - start_time, 2)
        success = len(failed_steps) == 0

        logger.info(f"Workflow {workflow.name} {'completed' if success else 'failed'} in {total_duration}s")

        return WorkflowResult(
            success=success,
            workflow_name=workflow.name,
            completed_steps=completed_steps,
            failed_steps=failed_steps,
            skipped_steps=skipped_steps,
            total_duration=total_duration,
            context=workflow.context,
            errors=errors,
        )

    def _dependencies_met(self, step: WorkflowStep, workflow: Workflow) -> bool:
        """Check if all dependencies for a step are completed."""
        for dep_name in step.dependencies:
            dep_step = workflow.get_step(dep_name)
            if not dep_step or dep_step.status != StepStatus.COMPLETED:
                return False
        return True

    def _run_agent_step(self, step: WorkflowStep, workflow: Workflow) -> Any:
        """Execute an agent step."""
        agent_name = step.config.get("agent_name")
        prompt = step.config.get("prompt", "")
        agents = workflow.context.get("_agents", {})
        file_manager = workflow.context.get("_file_manager")

        agent = agents.get(agent_name)
        if not agent:
            raise ValueError(f"Agent not found: {agent_name}")

        # Format prompt with context
        if workflow.context:
            try:
                prompt = prompt.format(**workflow.context)
            except KeyError:
                pass  # Keep original prompt if formatting fails

        logger.info(f"Running agent: {agent_name}")
        result = agent.run(step.name, prompt, workflow.context, file_manager)

        # Store result in context for later steps
        workflow.context[f"{step.name}_result"] = result

        return result

    def _run_parallel_step(self, step: WorkflowStep, workflow: Workflow) -> Dict[str, Any]:
        """Execute multiple sub-steps in parallel."""
        sub_steps = step.config.get("sub_steps", [])
        results = {}

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {}
            for sub_step_config in sub_steps:
                sub_step = WorkflowStep(
                    name=sub_step_config["name"],
                    step_type=StepType(sub_step_config.get("type", "agent")),
                    config=sub_step_config.get("config", {}),
                )
                handler = self.step_handlers.get(sub_step.step_type)
                if handler:
                    future = executor.submit(handler, sub_step, workflow)
                    futures[future] = sub_step.name

            for future in as_completed(futures):
                step_name = futures[future]
                try:
                    results[step_name] = future.result()
                except Exception as exc:
                    results[step_name] = {"error": str(exc)}

        return results

    def _run_conditional_step(self, step: WorkflowStep, workflow: Workflow) -> Any:
        """Execute a conditional step based on a condition."""
        condition = step.config.get("condition", "")
        then_step = step.config.get("then")
        else_step = step.config.get("else")

        # Evaluate condition using workflow context
        try:
            result = eval(condition, {"__builtins__": {}}, workflow.context)
        except Exception:
            result = False

        if result and then_step:
            sub_step = WorkflowStep(
                name=f"{step.name}_then",
                step_type=StepType(then_step.get("type", "agent")),
                config=then_step.get("config", {}),
            )
            handler = self.step_handlers.get(sub_step.step_type)
            if handler:
                return handler(sub_step, workflow)

        elif not result and else_step:
            sub_step = WorkflowStep(
                name=f"{step.name}_else",
                step_type=StepType(else_step.get("type", "agent")),
                config=else_step.get("config", {}),
            )
            handler = self.step_handlers.get(sub_step.step_type)
            if handler:
                return handler(sub_step, workflow)

        return None

    def _run_checkpoint_step(self, step: WorkflowStep, workflow: Workflow) -> Dict[str, Any]:
        """Create a checkpoint of the current workflow state."""
        checkpoint_name = step.config.get("name", step.name)

        # Store current context as checkpoint
        workflow.checkpoints[checkpoint_name] = {
            "context": workflow.context.copy(),
            "timestamp": time.time(),
        }

        logger.info(f"Checkpoint created: {checkpoint_name}")
        return {"checkpoint": checkpoint_name}

    def _run_wait_step(self, step: WorkflowStep, workflow: Workflow) -> Dict[str, Any]:
        """Wait for a specified duration."""
        duration = step.config.get("duration", 1.0)
        logger.info(f"Waiting for {duration}s")
        time.sleep(duration)
        return {"waited": duration}

    def _run_custom_step(self, step: WorkflowStep, workflow: Workflow) -> Any:
        """Execute a custom step using a provided function."""
        func = step.config.get("function")
        if callable(func):
            return func(step, workflow)
        raise ValueError("Custom step requires a callable 'function' in config")

    def rollback_to_checkpoint(self, workflow: Workflow, checkpoint_name: str) -> bool:
        """Rollback workflow to a previous checkpoint.

        Args:
            workflow: The workflow to rollback
            checkpoint_name: Name of the checkpoint to rollback to

        Returns:
            True if rollback successful, False otherwise
        """
        checkpoint = workflow.checkpoints.get(checkpoint_name)
        if not checkpoint:
            logger.error(f"Checkpoint not found: {checkpoint_name}")
            return False

        workflow.context = checkpoint["context"].copy()
        logger.info(f"Rolled back to checkpoint: {checkpoint_name}")
        return True


def create_simple_workflow(
    name: str,
    steps: List[Dict[str, Any]],
) -> Workflow:
    """Create a simple workflow from a list of step configurations.

    Args:
        name: Workflow name
        steps: List of step configs, each with:
            - name: Step name
            - type: Step type (agent, parallel, etc.)
            - config: Step configuration

    Returns:
        A configured Workflow object
    """
    workflow = Workflow(name=name)

    for step_config in steps:
        step = WorkflowStep(
            name=step_config["name"],
            step_type=StepType(step_config.get("type", "agent")),
            config=step_config.get("config", {}),
            dependencies=step_config.get("dependencies", []),
        )
        workflow.add_step(step)

    return workflow
