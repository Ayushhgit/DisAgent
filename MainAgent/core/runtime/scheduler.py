from __future__ import annotations

import time
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Optional

from core.context.state_tracker import StateTracker
from core.planning.task_planner import TaskPlanner
from core.runtime.agent import Agent, AgentResult

logger = logging.getLogger(__name__)


class TaskScheduler:
    """Simple scheduler to assign ready tasks and execute agents concurrently.

    This is intentionally lightweight: it polls StateTracker.get_ready_tasks(),
    attempts to assign tasks using TaskPlanner.assign_task_to_best_agent(),
    and then runs the provided Agent instances in a ThreadPoolExecutor.
    """

    def __init__(self, state: StateTracker, planner: TaskPlanner, max_workers: int = 4):
        self.state = state
        self.planner = planner
        self.max_workers = max_workers

    def run(self, agents: Dict[str, Agent], prompts: Dict[str, str], context, file_manager, poll_interval: float = 0.5) -> Dict[str, float]:
        """Run scheduling loop until no ready tasks remain.

        Args:
            agents: mapping agent_name -> Agent instance
            prompts: mapping agent_name -> prompt string
            poll_interval: seconds to wait between polling loops

        Returns:
            execution_stats: mapping agent_name -> elapsed_seconds
        """
        execution_stats: Dict[str, float] = {}

        # Only consider agents that are in the agents map (worker agents)
        worker_agent_names = list(agents.keys())
        logger.debug("Worker agents available: %s", worker_agent_names)

        # Use ThreadPoolExecutor for agent execution
        with ThreadPoolExecutor(max_workers=self.max_workers) as ex:
            futures = {}

            while True:
                ready = self.planner.get_ready_tasks()
                logger.debug("Ready tasks: %s, Running futures: %d", ready, len(futures))

                if not ready and not futures:
                    # nothing queued and no running futures -> done
                    break

                # try to assign all ready tasks
                for task_id in ready:
                    # IMPORTANT: Only assign to worker agents (not main_orchestrator)
                    # Pass the list of worker agents as preferred
                    assigned = self.planner.assign_task_to_best_agent(task_id, preferred_agents=worker_agent_names)
                    logger.debug("Task %s assignment result: %s", task_id, assigned)

                    if assigned:
                        # look up which agent has this current task
                        task_info = self.state.tasks.get(task_id)
                        if not task_info:
                            logger.warning("Task %s not found in state after assignment", task_id)
                            continue
                        agent_name = task_info.assigned_agent
                        agent = agents.get(agent_name)
                        if not agent:
                            logger.warning("Assigned agent %s not found in agents map (available: %s)", agent_name, worker_agent_names)
                            continue

                        prompt = prompts.get(agent_name, f"You are {agent_name}. Complete task {task_id}")
                        logger.debug("Submitting task %s to agent %s", task_id, agent_name)
                        # submit execution
                        future = ex.submit(self._run_agent, agent, task_id, prompt, context, file_manager)
                        futures[future] = (agent_name, task_id, time.time())

                # collect completed futures
                done_now = [f for f in futures if f.done()]
                for f in done_now:
                    agent_name, task_id, start_ts = futures.pop(f)
                    try:
                        result: AgentResult = f.result()
                        elapsed = round(time.time() - start_ts, 2)
                        execution_stats[agent_name] = elapsed
                        if result.success:
                            # Mark task complete
                            self.state.complete_task(task_id, f"Completed in {elapsed}s")
                        else:
                            self.state.fail_task(task_id, result.output)
                    except Exception as exc:
                        logger.exception("Agent %s crashed: %s", agent_name, exc)
                        self.state.fail_task(task_id, str(exc))

                time.sleep(poll_interval)

        return execution_stats

    @staticmethod
    def _run_agent(agent: Agent, task_id: str, prompt: str, context, file_manager) -> AgentResult:
        try:
            # agent.run is responsible for returning an AgentResult
            return agent.run(task_id, prompt, context, file_manager)
        except Exception as exc:
            return AgentResult(success=False, output=str(exc), metadata={})

