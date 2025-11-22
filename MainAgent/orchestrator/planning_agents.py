"""Planning-focused agent definitions for the Groq multi-agent workflow."""

from __future__ import annotations

import json
import re
from typing import Callable, Dict, List, Tuple

from .context import AgentContext
from MainAgent.core.runtime.llm import llm_call


def planning_agent(problem: str) -> Tuple[List[str], str, Dict]:
    """Decompose the user problem into ordered subtasks and select agents."""

    prompt = f"""You are the Master Planning Agent for a complex software development task.

TASK: {problem}

Analyze this task thoroughly and create a detailed breakdown:
1. Understand exactly what needs to be built
2. Break it into 4-6 logical, sequential subtasks
3. Determine the correct order of execution
4. Identify dependencies

Respond ONLY in valid JSON (no markdown):
{{
  "task_summary": "brief overview",
  "subtasks": [
    {{"agent": "Architect", "description": "...", "dependencies": []}},
    {{"agent": "DatabaseDesigner", "description": "...", "dependencies": ["Architect"]}}
  ],
  "architecture_notes": "overall design decisions",
  "key_constraints": "requirements",
  "file_structure": ["list", "of", "files"]
}}"""

    print("[PLANNING AGENT] Analyzing task structure...\n")
    response = llm_call(prompt, max_tokens=1500, temperature=0.3)

    try:
        json_match = re.search(r"\{[\s\S]*\}", response)
        if not json_match:
            raise ValueError("No JSON found")

        parsed = json.loads(json_match.group())
        agents = [task["agent"] for task in parsed["subtasks"]]
        architecture = parsed.get("architecture_notes", "")
        return agents, architecture, parsed
    except Exception as exc:  # pragma: no cover
        print(f"⚠️  Parsing failed, using default agents: {exc}\n")
        return (
            ["Architect", "DatabaseDesigner", "APIImplementer", "ServiceImplementer", "Testing"],
            "",
            {},
        )


def _simple_agent_executor(
    title: str,
    problem: str,
    context: AgentContext,
    body: str,
    max_tokens: int = 8192,
) -> str:
    print(f"\n[{title.upper()}] Executing...\n")
    return llm_call(body, max_tokens=max_tokens, temperature=1)


def architecture_agent(problem: str, context: AgentContext) -> str:
    prompt = f"""You are the System Architect Agent.

TASK: {problem}

Design the complete system architecture with details on components, tech stack,
data flow, API structure, database models, external dependencies, and scalability.
Use clear sections with ## headers."""
    output = _simple_agent_executor("Architect Agent", problem, context, prompt)
    context.add_result("Architect", output)
    context.set_architecture(output)
    return output


def database_agent(problem: str, context: AgentContext) -> str:
    prompt = f"""You are the Database Design Agent.

MAIN TASK: {problem}

PREVIOUS WORK:
{context.get_context()}

Design the complete database schema and provide SQLAlchemy ORM models with
fields, relationships, constraints, timestamps, and configuration boilerplate."""
    output = _simple_agent_executor("Database Agent", problem, context, prompt)
    context.add_result("DatabaseDesigner", output)
    return output


def api_agent(problem: str, context: AgentContext) -> str:
    prompt = f"""You are the FastAPI Implementation Agent.

MAIN TASK: {problem}

PREVIOUS WORK:
{context.get_context()}

Implement full FastAPI app code with Pydantic schemas, CRUD endpoints, middleware,
database integration, and error handling."""
    output = _simple_agent_executor("API Agent", problem, context, prompt)
    context.add_result("APIImplementer", output)
    return output


def service_agent(problem: str, context: AgentContext) -> str:
    prompt = f"""You are the Service/Business Logic Agent.

MAIN TASK: {problem}

PREVIOUS WORK:
{context.get_context()}

Implement service classes, helpers, and exception handling with complete logic."""
    output = _simple_agent_executor("Service Agent", problem, context, prompt)
    context.add_result("ServiceImplementer", output)
    return output


def testing_agent(problem: str, context: AgentContext) -> str:
    prompt = f"""You are the Testing Agent.

MAIN TASK: {problem}

PREVIOUS WORK:
{context.get_context()}

Create comprehensive pytest suites covering units, integrations, and edge cases."""
    output = _simple_agent_executor("Testing Agent", problem, context, prompt)
    context.add_result("Testing", output)
    return output


def deployment_agent(problem: str, context: AgentContext) -> str:
    prompt = f"""You are the Deployment & DevOps Agent.

MAIN TASK: {problem}

PREVIOUS WORK:
{context.get_context()}

Create deployment files: requirements.txt, .env.example, Dockerfile, docker-compose,
README, and the application entrypoint."""
    output = _simple_agent_executor("Deployment Agent", problem, context, prompt)
    context.add_result("Deployment", output)
    return output


AGENT_REGISTRY: Dict[str, Callable[[str, AgentContext], str]] = {
    "Architect": architecture_agent,
    "DatabaseDesigner": database_agent,
    "APIImplementer": api_agent,
    "ServiceImplementer": service_agent,
    "Testing": testing_agent,
    "Deployment": deployment_agent,
}


