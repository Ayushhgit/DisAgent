"""Scope analysis and prompt engineering utilities."""

from __future__ import annotations

import json
import re
from typing import Dict, List, Tuple

from .context import AgentContext
from core.runtime.llm import llm_call


def scope_analyzer(user_request: str) -> Tuple[Dict, List[str]]:
    """Analyze the user request to determine scope and needed agents."""

    prompt = f"""You are a Project Scope Analyzer.

USER REQUEST: {user_request}

Analyze this request and determine:
1. Scope level: prototype, mvp, or production
2. Complexity: simple, moderate, or complex
3. Domains involved
4. Required agents for completion

Respond ONLY in valid JSON:
{{
    "scope_level": "prototype|mvp|production",
    "complexity": "simple|moderate|complex",
    "domains": ["domain1", "domain2"],
    "target_agents": ["AgentName1", "AgentName2"],
    "rationale": "brief explanation",
    "key_requirements": "important constraints"
}}"""

    print("[SCOPE ANALYZER] Analyzing project requirements...\n")
    response = llm_call(prompt, max_tokens=1000, temperature=0.3)

    try:
        json_match = re.search(r"\{[\s\S]*\}", response)
        if not json_match:
            raise ValueError("No JSON found in response")

        parsed = json.loads(json_match.group())
        scope_dict = {
            "scope_level": parsed.get("scope_level", "unknown"),
            "complexity": parsed.get("complexity", "unknown"),
            "domains": parsed.get("domains", []),
            "key_requirements": parsed.get("key_requirements", ""),
        }
        agents = parsed.get("target_agents", []) or ["CoreImplementer", "Integrator"]
        return scope_dict, agents
    except Exception as exc:  # pragma: no cover
        print(f"⚠️  Parsing failed: {exc}\n")
        return (
            {
                "scope_level": "prototype",
                "complexity": "simple",
                "domains": ["general"],
                "key_requirements": "",
            },
            ["CoreImplementer", "Integrator"],
        )


def _extract_json_safely(text: str, agent_names: List[str]) -> dict:
    """Extract and parse JSON from text, handling common LLM formatting issues."""
    
    if not text or len(text.strip()) < 10:
        raise ValueError("Response text is too short or empty")
    
    # Try multiple patterns to find JSON
    patterns = [
        r"\{[\s\S]*\}",  # Standard JSON object
        r"```json\s*(\{[\s\S]*?\})\s*```",  # JSON in code block
        r"```\s*(\{[\s\S]*?\})\s*```",  # JSON in generic code block
    ]
    
    json_str = None
    for pattern in patterns:
        json_match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if json_match:
            json_str = json_match.group(1) if json_match.lastindex else json_match.group()
            break
    
    if not json_str:
        raise ValueError("No JSON found in response")
    
    # Try direct parsing first
    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        # Log the error for debugging but continue with manual extraction
        pass
    
    # If direct parsing fails, manually extract agent prompts
    # This handles cases where JSON has unescaped newlines/quotes
    prompts = {}
    
    for agent in agent_names:
        # Look for agent name followed by colon and opening quote
        # Then capture everything until the closing quote (handling escaped quotes)
        # Pattern: "AgentName": "content..." or AgentName: "content..."
        patterns = [
            # Pattern with quotes around key
            rf'"{re.escape(agent)}"\s*:\s*"',
            # Pattern without quotes around key  
            rf'{re.escape(agent)}\s*:\s*"',
        ]
        
        for pattern in patterns:
            start_match = re.search(pattern, json_str, re.IGNORECASE)
            if start_match:
                start_pos = start_match.end()
                # Find the matching closing quote, handling escaped quotes
                content_parts = []
                i = start_pos
                while i < len(json_str):
                    if json_str[i] == '"' and (i == start_pos or json_str[i-1] != '\\'):
                        # Found unescaped closing quote
                        break
                    elif json_str[i] == '\\' and i + 1 < len(json_str):
                        # Escaped character
                        content_parts.append(json_str[i:i+2])
                        i += 2
                    else:
                        content_parts.append(json_str[i])
                        i += 1
                
                prompt_text = ''.join(content_parts)
                # Unescape common sequences
                prompt_text = (
                    prompt_text
                    .replace('\\n', '\n')
                    .replace('\\r', '\r')
                    .replace('\\t', '\t')
                    .replace('\\"', '"')
                    .replace('\\\\', '\\')
                )
                prompts[agent] = prompt_text
                break
    
    if prompts and len(prompts) > 0:
        return prompts
    
    # Last resort: try to reconstruct JSON by properly escaping
    raise ValueError("Could not extract prompts from JSON response")


def prompt_engineer_agent(
    user_request: str,
    agent_names: List[str],
    context: AgentContext,
) -> Dict[str, str]:
    """Generate specialized prompts for each downstream agent."""

    scope_context = ""
    if context.scope_info:
        scope_context = "\nSCOPE INFORMATION:\n"
        for key, val in context.scope_info.items():
            scope_context += f"  {key}: {val}\n"

    prompt = f"""You are the Prompt Engineering Agent. Your job is to create customized prompts for {len(agent_names)} specialist agents.

USER REQUEST: {user_request}
{scope_context}

AGENTS THAT NEED PROMPTS: {', '.join(agent_names)}

For EACH agent listed above, create a detailed, actionable prompt that:
1. Clearly defines their specific role and what they need to accomplish
2. References the user request and scope information
3. Instructs them on what files to read, edit, or create
4. Provides clear guidance on using the ===EDIT=== format for modifications
5. Emphasizes reviewing existing project files before making changes

CRITICAL REQUIREMENTS:
- You MUST respond with valid JSON only
- The JSON must have exactly {len(agent_names)} keys, one for each agent name
- Escape newlines as \\n and quotes as \\" in string values
- Each prompt should be at least 100 characters long
- Be specific and actionable

Example JSON format:
{{"AgentName1": "You are AgentName1. Your task: [description]. Review existing files, then [specific actions].", "AgentName2": "You are AgentName2..."}}

IMPORTANT: Respond with ONLY the JSON object, no other text before or after. Start with {{ and end with }}.

Now create the JSON with prompts for: {', '.join(agent_names)}
"""

    print("[PROMPT ENGINEER] Creating customized prompts...\n")
    
    # Try up to 2 times to get a valid response
    max_retries = 2
    response = None
    
    for attempt in range(max_retries):
        response = llm_call(prompt, max_tokens=4000, temperature=0.5)
        
        # Debug: Show response info
        response_len = len(response) if response else 0
        response_preview = response[:200] if response and len(response) > 0 else "(empty)"
        
        if attempt == 0:
            print(f"   📊 Response length: {response_len} characters")
            if response_len > 0 and response_len < 200:
                print(f"   📝 Full response: {response}\n")
            elif response_len > 0:
                print(f"   📝 Response preview: {response_preview}...\n")
        
        # Check if response is valid
        if response and len(response.strip()) >= 50:
            # Try to parse it
            try:
                _extract_json_safely(response, agent_names)
                # If we get here, the response looks valid
                break
            except Exception:
                if attempt < max_retries - 1:
                    print(f"   ⚠️  Attempt {attempt + 1} failed, retrying...\n")
                    continue
        else:
            if attempt < max_retries - 1:
                print(f"   ⚠️  Empty response on attempt {attempt + 1}, retrying...\n")
                continue
    
    # Final check
    if not response or len(response.strip()) < 50:
        print(f"⚠️  Prompt engineer returned empty/minimal response after {max_retries} attempts (length: {len(response) if response else 0})\n")
        if response and len(response) > 0:
            print(f"   Full response was: {response}\n")
        return {agent: f"You are a {agent}. Complete: {user_request}" for agent in agent_names}

    try:
        prompts = _extract_json_safely(response, agent_names)
        
        # Ensure all agents have prompts
        for agent in agent_names:
            if not prompts.get(agent) or len(prompts.get(agent, "")) < 20:
                prompts[agent] = f"You are a {agent}. Complete the task: {user_request}"
        
        return prompts
    except Exception as exc:  # pragma: no cover
        print(f"⚠️  Failed to parse prompts: {exc}\n")
        if len(response) > 0:
            preview = response[:500] if len(response) > 500 else response
            print(f"   Response preview: {preview}...\n")
        else:
            print(f"   Response was empty\n")
        # Fallback: create simple but effective prompts
        fallback_prompts = {}
        for agent in agent_names:
            fallback_prompts[agent] = f"""You are a {agent}. Your task is to: {user_request}

IMPORTANT:
- Review all existing files in the project
- Understand the current codebase structure
- Make improvements based on the user request
- Use the ===EDIT=== format for precise changes
- Provide complete, working code

Complete the task now."""
        return fallback_prompts


