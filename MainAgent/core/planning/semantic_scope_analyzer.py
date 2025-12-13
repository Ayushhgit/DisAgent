"""
Semantic Scope Analyzer - LLM-powered scope analysis.

Replaces keyword-based scope analysis with semantic understanding using LLM.

Key Features:
- Deep understanding of user intent
- Intelligent file/component identification
- Accurate complexity estimation
- Dependency chain analysis
- Impact assessment across codebase

This addresses a key weakness identified in research evaluation:
the original keyword-based analysis couldn't understand nuanced
requests or identify subtle impacts across the codebase.
"""

from __future__ import annotations

import os
import re
import json
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable
from pathlib import Path
import threading

logger = logging.getLogger(__name__)


@dataclass
class FileImpact:
    """Represents the impact on a single file."""
    path: str
    impact_type: str  # "modify", "create", "delete", "read"
    confidence: float  # 0.0 to 1.0
    reason: str
    components: List[str] = field(default_factory=list)  # Classes, functions affected
    estimated_changes: int = 0  # Lines to change


@dataclass
class DependencyChain:
    """Represents a chain of dependencies."""
    source: str
    targets: List[str]
    chain_type: str  # "import", "call", "inherit", "data"
    impact_level: str  # "direct", "indirect", "transitive"


@dataclass
class SemanticScope:
    """Complete semantic analysis of a scope."""
    title: str
    description: str
    intent: str  # Parsed user intent
    complexity: str  # "low", "medium", "high"
    scope_level: str  # "ticket", "feature", "project"
    estimated_effort: str  # "minutes", "hours", "days"

    # Detailed analysis
    tasks: List[Dict[str, Any]] = field(default_factory=list)
    agents_needed: List[str] = field(default_factory=list)
    domains: List[str] = field(default_factory=list)
    constraints: List[str] = field(default_factory=list)

    # File impact analysis
    files_to_modify: List[FileImpact] = field(default_factory=list)
    files_to_create: List[FileImpact] = field(default_factory=list)
    files_to_read: List[FileImpact] = field(default_factory=list)

    # Dependency analysis
    dependency_chains: List[DependencyChain] = field(default_factory=list)
    breaking_changes: List[str] = field(default_factory=list)

    # Risk assessment
    risk_level: str = "low"  # "low", "medium", "high", "critical"
    risk_factors: List[str] = field(default_factory=list)

    # Additional metadata
    suggested_tests: List[str] = field(default_factory=list)
    rollback_plan: str = ""
    confidence_score: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for compatibility with existing code."""
        return {
            "title": self.title,
            "description": self.description,
            "tasks": self.tasks,
            "agents_needed": self.agents_needed,
            "complexity": self.complexity,
            "scope_level": self.scope_level,
            "domains": self.domains,
            "constraints": self.constraints,
            "intent": self.intent,
            "estimated_effort": self.estimated_effort,
            "files_to_modify": [f.__dict__ for f in self.files_to_modify],
            "files_to_create": [f.__dict__ for f in self.files_to_create],
            "files_to_read": [f.__dict__ for f in self.files_to_read],
            "risk_level": self.risk_level,
            "risk_factors": self.risk_factors,
            "suggested_tests": self.suggested_tests,
            "confidence_score": self.confidence_score
        }


class SemanticScopeAnalyzer:
    """LLM-powered semantic scope analyzer.

    Uses LLM to deeply understand:
    1. What the user wants to accomplish (intent)
    2. What files/components need to change (impact)
    3. What the dependencies are (ripple effects)
    4. How complex/risky the change is (assessment)
    """

    def __init__(
        self,
        llm_call: Callable[[str, Optional[int]], str],
        project_path: Optional[str] = None
    ):
        """Initialize semantic analyzer.

        Args:
            llm_call: Function to call LLM with (prompt, max_tokens) -> response
            project_path: Optional path to project for file scanning
        """
        self.llm_call = llm_call
        self.project_path = Path(project_path) if project_path else None
        self._lock = threading.Lock()
        self._cache: Dict[str, SemanticScope] = {}

    def analyze(
        self,
        user_input: str,
        repo_path: Optional[str] = None,
        existing_files: Optional[List[str]] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Analyze scope using LLM.

        Args:
            user_input: User's request
            repo_path: Path to repository
            existing_files: List of existing files in project
            context: Additional context (memory, previous tasks, etc.)

        Returns:
            Dictionary compatible with existing ScopeAnalyzer output
        """
        if not user_input or not user_input.strip():
            raise ValueError("Empty user input")

        # Use project path if provided
        if repo_path:
            self.project_path = Path(repo_path)

        # Scan project files if path available
        project_structure = ""
        if self.project_path and self.project_path.exists():
            project_structure = self._scan_project_structure()
        elif existing_files:
            project_structure = "Existing files:\n" + "\n".join(existing_files[:50])

        # Build analysis prompt
        prompt = self._build_analysis_prompt(user_input, project_structure, context)

        try:
            # Call LLM for analysis
            response = self.llm_call(prompt, 2000)
            scope = self._parse_llm_response(response, user_input)

            return scope.to_dict()

        except Exception as e:
            logger.exception(f"LLM analysis failed: {e}")
            # Fallback to basic analysis
            return self._fallback_analysis(user_input)

    def _build_analysis_prompt(
        self,
        user_input: str,
        project_structure: str,
        context: Optional[Dict[str, Any]]
    ) -> str:
        """Build the LLM prompt for scope analysis."""
        context_str = ""
        if context:
            if context.get("recent_changes"):
                context_str += f"\nRecent changes: {context['recent_changes']}"
            if context.get("learnings"):
                context_str += f"\nRelevant learnings: {context['learnings']}"

        prompt = f"""You are a software architect analyzing a development task.

USER REQUEST:
{user_input}

{f"PROJECT STRUCTURE:{chr(10)}{project_structure}" if project_structure else "No project structure available - new project."}
{context_str}

Analyze this request and provide a JSON response with the following structure:
{{
    "intent": "Clear statement of what the user wants to accomplish",
    "title": "Short title (max 10 words)",
    "description": "Detailed description of the scope",
    "complexity": "low|medium|high",
    "scope_level": "ticket|feature|project",
    "estimated_effort": "minutes|hours|days",
    "tasks": [
        {{
            "id": "task-1",
            "description": "Task description",
            "type": "implementation|testing|documentation|refactoring|bugfix",
            "estimate": "small|medium|large",
            "dependencies": [],
            "files_affected": ["path/to/file.py"]
        }}
    ],
    "agents_needed": ["backend", "frontend", "tester", "devops", "db_engineer"],
    "domains": ["backend", "frontend", "database", "devops", "testing"],
    "files_to_modify": [
        {{
            "path": "path/to/file.py",
            "reason": "Why this file needs modification",
            "components": ["ClassName", "function_name"],
            "impact_type": "modify"
        }}
    ],
    "files_to_create": [
        {{
            "path": "path/to/new_file.py",
            "reason": "Purpose of new file"
        }}
    ],
    "files_to_read": [
        {{
            "path": "path/to/file.py",
            "reason": "Why this file needs to be read for context"
        }}
    ],
    "dependency_chains": [
        {{
            "source": "file_a.py",
            "targets": ["file_b.py", "file_c.py"],
            "chain_type": "import|call|inherit",
            "impact_level": "direct|indirect"
        }}
    ],
    "risk_level": "low|medium|high|critical",
    "risk_factors": ["List of potential risks"],
    "suggested_tests": ["Test descriptions to verify the change"],
    "constraints": ["Any constraints or requirements"]
}}

Important guidelines:
1. Be specific about which files need to change and why
2. Identify all dependencies that might be affected
3. Consider both direct and indirect impacts
4. Provide realistic complexity and effort estimates
5. Suggest comprehensive tests for verification
6. Identify potential risks early

Return ONLY the JSON, no additional text."""

        return prompt

    def _parse_llm_response(self, response: str, user_input: str) -> SemanticScope:
        """Parse LLM response into SemanticScope."""
        # Clean up response
        response = response.strip()

        # Try to extract JSON from response
        json_match = re.search(r'\{[\s\S]*\}', response)
        if json_match:
            response = json_match.group()

        try:
            data = json.loads(response)
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse LLM JSON: {e}")
            return self._create_basic_scope(user_input)

        # Build SemanticScope from parsed data
        scope = SemanticScope(
            title=data.get("title", user_input[:50]),
            description=data.get("description", user_input),
            intent=data.get("intent", user_input),
            complexity=data.get("complexity", "medium"),
            scope_level=data.get("scope_level", "feature"),
            estimated_effort=data.get("estimated_effort", "hours"),
            tasks=data.get("tasks", []),
            agents_needed=data.get("agents_needed", ["generalist"]),
            domains=data.get("domains", ["general"]),
            constraints=data.get("constraints", []),
            risk_level=data.get("risk_level", "low"),
            risk_factors=data.get("risk_factors", []),
            suggested_tests=data.get("suggested_tests", []),
            confidence_score=0.9
        )

        # Parse file impacts
        for f in data.get("files_to_modify", []):
            scope.files_to_modify.append(FileImpact(
                path=f.get("path", ""),
                impact_type="modify",
                confidence=0.8,
                reason=f.get("reason", ""),
                components=f.get("components", [])
            ))

        for f in data.get("files_to_create", []):
            scope.files_to_create.append(FileImpact(
                path=f.get("path", ""),
                impact_type="create",
                confidence=0.9,
                reason=f.get("reason", "")
            ))

        for f in data.get("files_to_read", []):
            scope.files_to_read.append(FileImpact(
                path=f.get("path", ""),
                impact_type="read",
                confidence=0.7,
                reason=f.get("reason", "")
            ))

        # Parse dependency chains
        for chain in data.get("dependency_chains", []):
            scope.dependency_chains.append(DependencyChain(
                source=chain.get("source", ""),
                targets=chain.get("targets", []),
                chain_type=chain.get("chain_type", "import"),
                impact_level=chain.get("impact_level", "direct")
            ))

        # Ensure tasks have required structure
        scope.tasks = self._normalize_tasks(scope.tasks, user_input)

        return scope

    def _normalize_tasks(
        self,
        tasks: List[Dict[str, Any]],
        user_input: str
    ) -> List[Dict[str, Any]]:
        """Ensure tasks have all required fields."""
        if not tasks:
            return [{
                "id": "task-1",
                "description": user_input[:200],
                "type": "implementation",
                "estimate": "medium",
                "dependencies": [],
                "metadata": {}
            }]

        normalized = []
        for i, task in enumerate(tasks):
            normalized.append({
                "id": task.get("id", f"task-{i+1}"),
                "description": task.get("description", ""),
                "type": task.get("type", "implementation"),
                "estimate": task.get("estimate", "medium"),
                "dependencies": task.get("dependencies", []),
                "metadata": task.get("metadata", {})
            })

        return normalized

    def _create_basic_scope(self, user_input: str) -> SemanticScope:
        """Create basic scope when LLM parsing fails."""
        return SemanticScope(
            title=user_input[:50],
            description=user_input,
            intent=user_input,
            complexity="medium",
            scope_level="feature",
            estimated_effort="hours",
            tasks=[{
                "id": "task-1",
                "description": user_input,
                "type": "implementation",
                "estimate": "medium",
                "dependencies": [],
                "metadata": {}
            }],
            agents_needed=["generalist"],
            domains=["general"],
            confidence_score=0.3
        )

    def _fallback_analysis(self, user_input: str) -> Dict[str, Any]:
        """Fallback to basic keyword analysis when LLM fails."""
        scope = self._create_basic_scope(user_input)

        # Basic keyword matching for agents
        text = user_input.lower()
        if "test" in text:
            scope.agents_needed.append("tester")
        if "api" in text or "backend" in text:
            scope.agents_needed.append("backend")
        if "ui" in text or "frontend" in text:
            scope.agents_needed.append("frontend")
        if "database" in text or "sql" in text:
            scope.agents_needed.append("db_engineer")
        if "deploy" in text or "docker" in text:
            scope.agents_needed.append("devops")

        scope.agents_needed = list(set(scope.agents_needed))
        return scope.to_dict()

    def _scan_project_structure(self, max_files: int = 100) -> str:
        """Scan project structure for context."""
        if not self.project_path or not self.project_path.exists():
            return ""

        files = []
        ignore_patterns = {
            ".git", "__pycache__", "node_modules", ".venv",
            "venv", ".idea", ".vscode", "dist", "build", ".egg-info"
        }

        try:
            for root, dirs, filenames in os.walk(self.project_path):
                # Skip ignored directories
                dirs[:] = [d for d in dirs if d not in ignore_patterns]

                rel_root = os.path.relpath(root, self.project_path)
                for filename in filenames:
                    if len(files) >= max_files:
                        break

                    # Skip hidden and common non-code files
                    if filename.startswith(".") or filename.endswith((".pyc", ".lock")):
                        continue

                    rel_path = os.path.join(rel_root, filename) if rel_root != "." else filename
                    files.append(rel_path)

                if len(files) >= max_files:
                    break

        except Exception as e:
            logger.warning(f"Error scanning project: {e}")

        return "\n".join(files) if files else "Empty project"

    def analyze_impact(
        self,
        change_description: str,
        affected_files: List[str]
    ) -> Dict[str, Any]:
        """Analyze the impact of a specific change.

        Used for understanding ripple effects of a code change.
        """
        prompt = f"""Analyze the impact of this code change:

CHANGE: {change_description}

FILES BEING MODIFIED:
{chr(10).join(affected_files)}

Identify:
1. What other files might be affected by this change?
2. What tests should be run to verify the change?
3. What are the potential risks?
4. Is there a breaking change in the public API?

Return JSON:
{{
    "affected_files": ["list of files that might need updates"],
    "tests_to_run": ["test patterns or files"],
    "risks": ["potential issues"],
    "breaking_changes": ["API changes that need migration"],
    "migration_steps": ["steps for consumers if breaking"]
}}"""

        try:
            response = self.llm_call(prompt, 1000)
            return json.loads(response)
        except Exception as e:
            logger.warning(f"Impact analysis failed: {e}")
            return {
                "affected_files": [],
                "tests_to_run": [],
                "risks": ["Analysis failed - manual review recommended"],
                "breaking_changes": [],
                "migration_steps": []
            }


def create_semantic_analyzer(
    llm_call: Callable[[str, Optional[int]], str],
    project_path: Optional[str] = None
) -> SemanticScopeAnalyzer:
    """Factory function to create semantic analyzer.

    Args:
        llm_call: LLM function (prompt, max_tokens) -> response
        project_path: Optional project path for file scanning

    Returns:
        Configured SemanticScopeAnalyzer
    """
    return SemanticScopeAnalyzer(llm_call, project_path)


# Wrapper for backward compatibility with existing ScopeAnalyzer interface
class HybridScopeAnalyzer:
    """Combines semantic and keyword-based analysis.

    Falls back to keyword analysis when LLM is unavailable or fails.
    """

    def __init__(
        self,
        llm_call: Optional[Callable[[str, Optional[int]], str]] = None,
        project_path: Optional[str] = None,
        use_semantic: bool = True
    ):
        self.llm_call = llm_call
        self.project_path = project_path
        self.use_semantic = use_semantic and llm_call is not None

        if self.use_semantic:
            self._semantic_analyzer = SemanticScopeAnalyzer(llm_call, project_path)
        else:
            self._semantic_analyzer = None

        # Import existing analyzer for fallback
        from .scope_analyzer import ScopeAnalyzer
        self._keyword_analyzer = ScopeAnalyzer()

    def analyze(
        self,
        user_input: str,
        repo_path: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Analyze scope using hybrid approach.

        Tries semantic analysis first, falls back to keyword matching.
        """
        if self.use_semantic and self._semantic_analyzer:
            try:
                result = self._semantic_analyzer.analyze(
                    user_input,
                    repo_path=repo_path or self.project_path,
                    context=context
                )

                # Validate result has required fields
                if self._validate_result(result):
                    return result

            except Exception as e:
                logger.warning(f"Semantic analysis failed, using fallback: {e}")

        # Fallback to keyword analysis
        return self._keyword_analyzer.analyze(user_input, repo_path or self.project_path)

    def _validate_result(self, result: Dict[str, Any]) -> bool:
        """Validate analysis result has required fields."""
        required = ["title", "description", "tasks", "agents_needed", "complexity"]
        return all(field in result for field in required)
