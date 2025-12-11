# core/planning/scope_analyzer.py
"""
Scope Analyzer

Responsibilities:
- Turn a user prompt, repo snapshot, or high-level request into a machine-friendly scope:
    {
      "title": "...",
      "description": "...",
      "tasks": [
         {"id": "task-1", "description": "...", "type": "implementation", "estimate": "small", "metadata": {...}},
         ...
      ],
      "agents_needed": ["code_writer", "tester"],
      "complexity": "low|medium|high",
      "scope_level": "ticket|feature|project",
      "domains": ["backend", "frontend", ...],
      "constraints": []
    }
- Optionally inspect a code folder (basic heuristics) to refine the plan.

IMPORTANT: The returned dict MUST include all fields expected by UnifiedOrchestrator:
- title, description, tasks, agents_needed, complexity (original)
- scope_level, domains, constraints (required by orchestrator)
"""

from typing import Dict, List, Optional, Tuple
import logging
import uuid
import os
import re

logger = logging.getLogger(__name__)


def _make_id(prefix: str = "task") -> str:
    return f"{prefix}-{uuid.uuid4().hex[:8]}"


class ScopeAnalyzer:
    def __init__(self):
        # map of keywords -> agent
        self.default_agent_map = {
            # --- Basic mappings ---
            "test": "tester",
            "unit test": "tester",
            "ci": "ci",
            "docker": "devops",
            "deploy": "devops",
            "database": "db_engineer",
            "schema": "db_engineer",
            "ui": "frontend",
            "react": "frontend",
            "node": "backend",
            "python": "backend",
            "style": "frontend",
            "design": "frontend",
            "error": "backend",
            # --- Frontend ---
            "css": "frontend",
            "html": "frontend",
            "angular": "frontend",
            "vue": "frontend",
            "tailwind": "frontend",
            "bootstrap": "frontend",
            "responsive": "frontend",
            "accessibility": "frontend",
            "widget": "frontend",
            "modal": "frontend",
            "ux": "design",
            "wireframe": "design",
            # --- Backend ---
            "api": "backend",
            "endpoint": "backend",
            "graphql": "backend",
            "rest": "backend",
            "java": "backend",
            "go": "backend",
            "ruby": "backend",
            "c#": "backend",
            "controller": "backend",
            "middleware": "backend",
            "microservice": "backend",
            "cache": "backend",
            "auth": "backend",
            # --- Database (db_engineer) ---
            "sql": "db_engineer",
            "postgres": "db_engineer",
            "mysql": "db_engineer",
            "mongo": "db_engineer",
            "redis": "db_engineer",
            "migration": "db_engineer",
            "query": "db_engineer",
            "index": "db_engineer",
            "procedure": "db_engineer",
            "orm": "db_engineer",
            # --- DevOps / Infrastructure ---
            "aws": "devops",
            "azure": "devops",
            "gcp": "devops",
            "kubernetes": "devops",
            "k8s": "devops",
            "terraform": "devops",
            "ansible": "devops",
            "jenkins": "devops",
            "pipeline": "devops",
            "nginx": "devops",
            "ssl": "devops",
            "monitoring": "devops",
            "server": "devops",
            "linux": "devops",
            # --- QA / Testing ---
            "selenium": "tester",
            "cypress": "tester",
            "jest": "tester",
            "integration": "tester",
            "e2e": "tester",
            "regression": "tester",
            "bug": "tester",
            "defect": "tester",
            "qa": "tester",
            "reproduce": "tester",
            # --- Security ---
            "vulnerability": "security",
            "penetration": "security",
            "firewall": "security",
            "encryption": "security",
            "jwt": "security",
            "oauth": "security",
            "xss": "security",
            # --- Mobile ---
            "ios": "mobile",
            "android": "mobile",
            "swift": "mobile",
            "kotlin": "mobile",
            "flutter": "mobile",
            "react native": "mobile",
            "apk": "mobile",
            "xcode": "mobile",
            # --- Frontend (Visuals & Interaction) ---
            "misaligned": "frontend",
            "color": "frontend",
            "font": "frontend",
            "typo": "frontend",
            "button": "frontend",
            "layout": "frontend",
            "mobile view": "frontend",
            "overlap": "frontend",
            "spelling": "frontend",
            "image": "frontend",
            "icon": "frontend",
            "broken link": "frontend",
            "resize": "frontend",
            "browser": "frontend",
            "dark mode": "frontend",
            "animation": "frontend",
            "scroll": "frontend",
            "click": "frontend",
            # --- Backend (Functionality & Logic) ---
            "crash": "backend",
            "timeout": "backend",
            "500 error": "backend",
            "not saving": "backend",
            "logic": "backend",
            "calculation": "backend",
            "upload failed": "backend",
            "processing": "backend",
            "export": "backend",
            "email not sent": "backend",
            "search results": "backend",
            "login failed": "backend",
            "password": "backend",
            "loading forever": "backend",
            # --- Database (Data Integrity) ---
            "missing data": "db_engineer",
            "duplicate": "db_engineer",
            "wrong value": "db_engineer",
            "corrupt": "db_engineer",
            "restore": "db_engineer",
            "history": "db_engineer",
            "report incorrect": "db_engineer",
            # --- DevOps (Availability & Access) ---
            "site down": "devops",
            "offline": "devops",
            "slow": "devops",
            "certificate": "devops",
            "domain": "devops",
            "access denied": "devops",
            "disk full": "devops",
            "cannot connect": "devops",
            "https": "devops",
            "environment": "devops",
            "version": "devops",
            # --- Tester / QA (Process) ---
            "steps to reproduce": "tester",
            "scenario": "tester",
            "verify": "tester",
            "acceptance": "tester",
            "staging": "tester",
            "expected result": "tester",
            "actual result": "tester",
        }

    def analyze(self, user_input: str, repo_path: Optional[str] = None) -> Dict:
        """
        Primary method.
        - user_input: the raw request or LLM summary
        - repo_path: optional path to repo to refine estimates (lightweight)

        ALWAYS returns a dict with at least:
        title, description, tasks, agents_needed, complexity, scope_level, domains, constraints.
        """
        # Validate input
        if not isinstance(user_input, str) or not user_input.strip():
            raise ValueError("ScopeAnalyzer.analyze expected non-empty string user_input")

        title = self._short_title(user_input)
        tasks = self._extract_tasks_from_text(user_input)
        agents = self._suggest_agents(user_input)
        complexity = self._estimate_complexity(user_input, repo_path)
        domains = self._infer_domains(user_input)
        scope_level = self._infer_scope_level(complexity, tasks)

        # If repo_path provided, do a light scan to update tasks/agents
        if repo_path and os.path.exists(repo_path):
            try:
                repo_info = self._quick_repo_scan(repo_path)
                # merge repo insights
                tasks = self._merge_repo_tasks(tasks, repo_info)
                agents = list(set(agents + repo_info.get("agents", [])))
                # bump complexity if repo is large
                if repo_info.get("lines", 0) > 20000:
                    complexity = "high"
                    scope_level = "project"
            except Exception as e:
                logger.warning("Repo scan failed: %s", e)

        # Build complete scope dict with ALL required fields
        scope: Dict = {
            "title": title,
            "description": self._short_description(user_input),
            "tasks": tasks or [self._task_from_description(user_input.strip())],
            "agents_needed": agents or ["generalist"],
            "complexity": complexity,
            # REQUIRED BY ORCHESTRATOR - these were previously missing!
            "scope_level": scope_level,
            "domains": domains or ["general"],
            "constraints": [],
        }

        return scope

    def _extract_tasks_from_text(self, text: str) -> List[Dict]:
        """
        Heuristic extraction:
        - Look for numbered lists.
        - Look for imperative sentences starting with verbs.
        - Fall back: single high-level implementation task.
        """
        tasks: List[Dict] = []
        # numbered items
        numbered = re.findall(r'\d+\.\s+([^\n]+)', text)
        for s in numbered:
            tasks.append(self._task_from_description(s))

        # imperative sentences (start with verb)
        sentences = re.split(r'(?<=[.!?])\s+', text)
        for s in sentences:
            if re.match(r'^(implement|create|add|remove|refactor|test|deploy|build|design)\b', s.strip(), flags=re.I):
                tasks.append(self._task_from_description(s.strip()))

        # fallback
        if not tasks:
            tasks.append(self._task_from_description(text.strip()))

        return tasks

    def _task_from_description(self, desc: str) -> Dict:
        t = {
            "id": _make_id("task"),
            "description": desc,
            "type": self._classify_task(desc),
            "estimate": self._estimate_size(desc),
            "dependencies": [],
            "metadata": {}
        }
        return t

    def _classify_task(self, desc: str) -> str:
        desc_l = desc.lower()
        for k, agent in self.default_agent_map.items():
            if k in desc_l:
                return "implementation" if agent in ("backend", "frontend", "db_engineer") else "ops"
        return "implementation"

    def _estimate_size(self, desc: str) -> str:
        # simple heuristics: presence of "small", "quick", "trivial"
        d = desc.lower()
        if any(w in d for w in ("quick", "small", "tiny", "simple", "minor")):
            return "small"
        if any(w in d for w in ("complex", "large", "major", "extensive", "complete")):
            return "large"
        return "medium"

    def _suggest_agents(self, text: str) -> List[str]:
        s = text.lower()
        agents = []
        for k, agent in self.default_agent_map.items():
            if k in s and agent not in agents:
                agents.append(agent)
        # always include a coordinator agent if we plan > 1 task
        if len(self._extract_tasks_from_text(text)) > 1 and "coordinator" not in agents:
            agents.insert(0, "coordinator")
        return agents or ["generalist"]

    def _estimate_complexity(self, text: str, repo_path: Optional[str] = None) -> str:
        score = 0
        text_l = text.lower()
        if any(k in text_l for k in ("rewrite", "migrate", "scale", "optimize", "refactor")):
            score += 1
        if repo_path and os.path.exists(repo_path):
            # bump estimate if many files
            for root, dirs, files in os.walk(repo_path):
                score += len(files) / 200.0
                if score > 2:
                    break
        if score <= 0:
            return "low"
        if score < 2:
            return "medium"
        return "high"

    def _quick_repo_scan(self, repo_path: str) -> Dict:
        """
        Lightweight scan: count lines and detect tech keywords from file extensions.
        Return: {lines: int, files: int, agents: [...]}.
        """
        lines = 0
        files = 0
        agents = set()
        ext_agents = {
            ".py": "backend",
            ".js": "frontend",
            ".ts": "frontend",
            ".java": "backend",
            ".sql": "db_engineer",
            "Dockerfile": "devops",
            ".yml": "ci",
            ".yaml": "ci",
        }
        for root, dirs, filenames in os.walk(repo_path):
            for f in filenames:
                files += 1
                p = os.path.join(root, f)
                _, ext = os.path.splitext(f)
                if f in ext_agents:
                    agents.add(ext_agents[f])
                elif ext in ext_agents:
                    agents.add(ext_agents[ext])
                try:
                    with open(p, "r", encoding="utf-8", errors="ignore") as fh:
                        for _ in fh:
                            lines += 1
                except Exception:
                    # skip unreadable files
                    continue
        return {"lines": lines, "files": files, "agents": list(agents)}

    @staticmethod
    def _merge_repo_tasks(tasks: List[Dict], repo_info: Dict) -> List[Dict]:
        # simple heuristic: if repo has tests missing but requested in text, add tasks
        new = list(tasks)
        if repo_info.get("lines", 0) > 500 and not any("test" in t["description"].lower() for t in tasks):
            new.append({
                "id": _make_id("task"),
                "description": "Add or improve automated tests for the repository",
                "type": "testing",
                "estimate": "medium",
                "dependencies": [],
                "metadata": {}
            })
        return new

    @staticmethod
    def _short_title(text: str, max_words: int = 6) -> str:
        return " ".join(text.strip().split()[:max_words])

    @staticmethod
    def _short_description(text: str, max_chars: int = 240) -> str:
        t = text.strip()
        return (t[: max_chars - 1] + "…") if len(t) > max_chars else t

    @staticmethod
    def _infer_domains(text: str) -> List[str]:
        """Infer relevant domains based on keywords in the text.

        Returns a list of domain strings that help the relevance selector
        and agents understand what areas of the codebase are involved.
        """
        t = text.lower()
        domains: List[str] = []

        # Backend/API indicators
        if any(w in t for w in ("api", "backend", "server", "endpoint", "microservice",
                                 "rest", "graphql", "fastapi", "flask", "django", "express")):
            domains.append("backend")

        # Frontend/UI indicators
        if any(w in t for w in ("ui", "frontend", "react", "vue", "next.js", "tailwind",
                                 "css", "html", "component", "button", "form", "page")):
            domains.append("frontend")

        # Database indicators
        if any(w in t for w in ("database", "schema", "sql", "migration", "postgres",
                                 "mysql", "mongodb", "redis", "orm", "model", "table")):
            domains.append("database")

        # DevOps/Infrastructure indicators
        if any(w in t for w in ("deploy", "docker", "kubernetes", "ci", "cd", "pipeline",
                                 "infra", "aws", "azure", "gcp", "terraform", "nginx")):
            domains.append("devops")

        # Testing indicators
        if any(w in t for w in ("test", "unit test", "coverage", "qa", "integration test",
                                 "pytest", "jest", "mock", "fixture")):
            domains.append("testing")

        # Documentation indicators
        if any(w in t for w in ("document", "readme", "docs", "api doc", "swagger", "openapi")):
            domains.append("documentation")

        return domains

    @staticmethod
    def _infer_scope_level(complexity: str, tasks: List[Dict]) -> str:
        """Infer scope level based on complexity and number of tasks.

        Returns one of:
        - "ticket": small, 1-2 tasks, low complexity (quick fix, single feature)
        - "feature": mixed complexity, moderate number of tasks
        - "project": large scope, many tasks, or high complexity

        This helps the orchestrator make decisions about parallelism,
        resource allocation, and user expectations.
        """
        n = len(tasks) if tasks else 0

        # Small, focused work
        if complexity == "low" and n <= 2:
            return "ticket"

        # Large or complex work
        if complexity == "high" or n > 6:
            return "project"

        # Everything else is a feature
        return "feature"
