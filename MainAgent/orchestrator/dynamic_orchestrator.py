"""Dynamic multi-agent orchestrator with scope analysis and prompt engineering."""

from __future__ import annotations

import time
from typing import Dict, List

from .context import AgentContext
from .dynamic_agents import create_dynamic_agent
from .file_manager import FileManager
from MainAgent.core.runtime.llm import llm_call
from .scope_prompting import prompt_engineer_agent, scope_analyzer


def orchestrator(user_request: str, output_folder: str = "./generated_project") -> None:
    """Run the dynamic agent workflow end-to-end."""

    try:
        file_manager = FileManager(output_folder)
        
        # Check if this is an existing project with files
        existing_files = file_manager.list_files("*")
        if existing_files:
            print(f"📋 Found {len(existing_files)} existing file(s) in project")
            print(f"   Agents will be able to read and edit these files\n")
        else:
            print(f"📋 New/empty project - agents will create new files\n")
            
    except Exception as exc:
        print(f"❌ Failed to initialize file manager: {exc}")
        return

    print("\n" + "=" * 70)
    print("🚀 DYNAMIC MULTI-AGENT ORCHESTRATOR WITH CODE EDITING")
    print("=" * 70)
    print(f"\n📋 User Request: {user_request}\n")

    context = AgentContext()
    overall_start = time.time()
    stats: Dict[str, float] = {}

    print("\n[STAGE 1: SCOPE ANALYSIS]\n")
    print("-" * 70)
    scope_start = time.time()
    scope_info, required_agents = scope_analyzer(user_request)
    stats["ScopeAnalysis"] = round(time.time() - scope_start, 2)
    context.set_scope(scope_info)

    print(f"\n✅ Scope Level: {scope_info.get('scope_level', 'unknown')}")
    print(f"✅ Complexity: {scope_info.get('complexity', 'unknown')}")
    print(f"✅ Domains: {', '.join(scope_info.get('domains', []))}")
    print(f"✅ Required Agents: {' → '.join(required_agents)}")
    print(f"⏱️  Analysis Time: {stats['ScopeAnalysis']}s\n")

    print("\n[STAGE 2: PROMPT ENGINEERING]\n")
    print("-" * 70)
    prompt_start = time.time()
    custom_prompts = prompt_engineer_agent(user_request, required_agents, context)
    stats["PromptEngineering"] = round(time.time() - prompt_start, 2)
    print(f"\n✅ Generated prompts for {len(custom_prompts)} agents")
    print(f"⏱️  Prompt Engineering Time: {stats['PromptEngineering']}s\n")

    print("\n[STAGE 3: DYNAMIC AGENT EXECUTION]\n")
    print("-" * 70)
    for index, agent_name in enumerate(required_agents, 1):
        agent_start = time.time()
        custom_prompt = custom_prompts.get(
            agent_name, f"You are {agent_name}. Complete: {user_request}"
        )

        print(f"\n[{index}/{len(required_agents)}] Spawning {agent_name}...")
        try:
            create_dynamic_agent(agent_name, custom_prompt, user_request, context, file_manager)
            elapsed = round(time.time() - agent_start, 2)
            stats[agent_name] = elapsed
            print(f"\n✅ {agent_name} completed in {elapsed}s")
        except Exception as exc:
            print(f"\n❌ {agent_name} failed: {exc}")
            stats[agent_name] = 0.0

    print("\n[STAGE 4: FINAL INTEGRATION]\n")
    print("-" * 70)
    final_summary = file_manager.get_project_summary()
    integration_prompt = f"""You are the Integration Specialist. All agents have completed their work.

{final_summary}

FULL CONTEXT:
{context.get_context()}

USER REQUEST: {user_request}

Create a comprehensive FINAL SUMMARY that:
1. Integrates all agent outputs
2. Highlights main deliverables and files created
3. Explains how components work together
4. Provides clear next steps
5. Notes dependencies or requirements
"""

    print("\n[GENERATING FINAL INTEGRATION SUMMARY]...\n")
    summary = llm_call(integration_prompt, max_tokens=4000, temperature=0.7)

    overall_end = time.time()

    print("\n" + "=" * 70)
    print("📊 EXECUTION REPORT")
    print("=" * 70)
    print(f"\n✅ Project: {user_request}")
    print(f"\n📦 Agents Executed ({len(required_agents)} total):")
    for agent in required_agents:
        print(f"   • {agent}: {stats.get(agent, 0)}s")

    print(f"\n⏱️  Timing Breakdown:")
    print(f"   • Scope Analysis: {stats.get('ScopeAnalysis', 0)}s")
    print(f"   • Prompt Engineering: {stats.get('PromptEngineering', 0)}s")
    print(f"   • Total Execution: {round(overall_end - overall_start, 2)}s")

    print(f"\n📁 Project Files:")
    print(f"   • Total Files: {len(file_manager.created_files)}")
    print(f"   • Edit History Entries: {len(file_manager.edit_history)}")
    if file_manager.created_files:
        print("\n📄 File List:")
        for file_path in sorted(file_manager.created_files):
            rel_path = file_path.replace(str(file_manager.base_path) + "\\", "")
            print(f"   ✓ {rel_path}")
    else:
        print("\n   ⚠️  WARNING: No files were created!")

    print("\n" + "=" * 70)
    print("📝 FINAL SUMMARY")
    print("=" * 70)
    print(f"\n{summary}")
    print("\n" + "=" * 70)
    print("✨ ORCHESTRATION COMPLETE")
    print("=" * 70)
    print(f"\n📂 All files created in: {file_manager.base_path.absolute()}\n")


