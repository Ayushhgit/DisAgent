"\"\"\"Groq-powered multi-agent orchestrator with planning and reporting.\"\"\""

from __future__ import annotations

import time
from typing import Dict, List

from .context import AgentContext
from MainAgent.core.runtime.llm import llm_call
from .planning_agents import AGENT_REGISTRY, planning_agent


def orchestrator(user_problem: str) -> None:
    """Main orchestrator managing planning, execution, and summarisation."""

    print("\n" + "=" * 70)
    print("🚀 MULTI-AGENT SYSTEM ORCHESTRATOR - Groq Powered")
    print("=" * 70)
    print(f"\n📋 Input Problem: {user_problem}\n")

    context = AgentContext()
    overall_start = time.time()
    stats: Dict[str, float] = {}

    print("\n[STAGE 1: PLANNING]\n")
    print("-" * 70)
    plan_start = time.time()
    agents, architecture, plan_detail = planning_agent(user_problem)
    stats["Planning"] = round(time.time() - plan_start, 2)

    print(f"\n✅ Planned Agents: {' → '.join(agents)}")
    print(f"⏱️  Planning Time: {stats['Planning']}s\n")
    if architecture:
        context.set_architecture(architecture)

    print("\n[STAGE 2: EXECUTION]\n")
    print("-" * 70)
    for index, agent_name in enumerate(agents, 1):
        agent_func = AGENT_REGISTRY.get(agent_name)
        if not agent_func:
            print(f"\n[{index}/{len(agents)}] Skipping unknown agent: {agent_name}")
            continue

        print(f"\n[{index}/{len(agents)}] Running {agent_name}...")
        print("-" * 70)

        start = time.time()
        agent_func(user_problem, context)
        elapsed = round(time.time() - start, 2)
        stats[agent_name] = elapsed

        print(f"✅ {agent_name} completed in {elapsed}s\n")

    print("\n[STAGE 3: FINAL SUMMARY]\n")
    print("-" * 70)
    summary_prompt = f"""You are the Final Integration Specialist.

All agents have completed their work. Here's what was generated:

{context.get_context()}

Create a 500-700 word executive summary covering:
1. What was built
2. Key components and relationships
3. File structure and organisation
4. How to run the application
5. Next steps and improvements
"""

    print("\n[GENERATING FINAL SUMMARY]...\n")
    summary = llm_call(summary_prompt, max_tokens=8192, temperature=1)

    overall_end = time.time()

    print("\n" + "=" * 70)
    print("📊 EXECUTION REPORT")
    print("=" * 70)
    print(f"\n✅ Task: {user_problem}")
    print(f"\n📦 Agents Executed:")
    for agent in agents:
        print(f"   • {agent}: {stats.get(agent, 0)}s")
    print(f"\n⏱️  Total Execution Time: {round(overall_end - overall_start, 2)}s")

    print("\n" + "=" * 70)
    print("📝 COMPONENT SUMMARY")
    print("=" * 70)
    print(f"\n{summary}")

    print("\n" + "=" * 70)
    print("✨ ORCHESTRATION COMPLETE")
    print("=" * 70 + "\n")


