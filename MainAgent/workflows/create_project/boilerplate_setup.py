"""
Boilerplate Setup - Set up project boilerplate and configuration.

This workflow handles setting up common boilerplate files
and configurations for projects.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class BoilerplateConfig:
    """Configuration for boilerplate setup."""
    include_ci: bool = True
    include_docker: bool = False
    include_linting: bool = True
    include_formatting: bool = True
    include_pre_commit: bool = False
    license_type: str = "MIT"


@dataclass
class SetupResult:
    """Result of boilerplate setup."""
    success: bool
    files_created: List[str] = field(default_factory=list)
    error: Optional[str] = None


def setup_boilerplate(
    template: str = "python",
    path: str = ".",
    config: Optional[BoilerplateConfig] = None,
) -> bool:
    """Set up project boilerplate (legacy interface).

    Args:
        template: Template type (python, node, etc.)
        path: Project path
        config: Optional configuration

    Returns:
        True if successful
    """
    if config is None:
        config = BoilerplateConfig()

    result = setup_project_boilerplate(path, template, config)
    return result.success


def setup_project_boilerplate(
    path: str,
    template: str,
    config: BoilerplateConfig,
) -> SetupResult:
    """Set up project boilerplate files.

    Args:
        path: Project path
        template: Template type
        config: Boilerplate configuration

    Returns:
        SetupResult with created files
    """
    project_path = Path(path).resolve()
    files_created = []

    if not project_path.exists():
        return SetupResult(
            success=False,
            error=f"Path does not exist: {project_path}",
        )

    try:
        # Get boilerplate files based on template
        boilerplate = _get_boilerplate(template, config)

        for file_path, content in boilerplate.items():
            full_path = project_path / file_path
            full_path.parent.mkdir(parents=True, exist_ok=True)

            # Don't overwrite existing files
            if not full_path.exists():
                full_path.write_text(content, encoding='utf-8')
                files_created.append(str(full_path))
            else:
                logger.info(f"Skipping existing file: {file_path}")

        return SetupResult(
            success=True,
            files_created=files_created,
        )

    except Exception as exc:
        logger.exception(f"Failed to set up boilerplate: {exc}")
        return SetupResult(
            success=False,
            error=str(exc),
        )


def _get_boilerplate(template: str, config: BoilerplateConfig) -> Dict[str, str]:
    """Get boilerplate files for template."""
    files = {}

    # Common files
    if config.include_linting:
        files.update(_get_linting_config(template))

    if config.include_formatting:
        files.update(_get_formatting_config(template))

    if config.include_ci:
        files.update(_get_ci_config(template))

    if config.include_docker:
        files.update(_get_docker_config(template))

    if config.include_pre_commit:
        files.update(_get_pre_commit_config(template))

    if config.license_type:
        files["LICENSE"] = _get_license(config.license_type)

    return files


def _get_linting_config(template: str) -> Dict[str, str]:
    """Get linting configuration files."""
    if template == "python":
        return {
            ".flake8": """[flake8]
max-line-length = 120
extend-ignore = E203
exclude =
    .git,
    __pycache__,
    build,
    dist,
    venv,
    .venv
""",
            "mypy.ini": """[mypy]
python_version = 3.10
warn_return_any = True
warn_unused_configs = True
ignore_missing_imports = True
""",
        }
    elif template in ("node", "javascript", "typescript"):
        return {
            ".eslintrc.json": """{
  "env": {
    "node": true,
    "es2021": true
  },
  "extends": ["eslint:recommended"],
  "parserOptions": {
    "ecmaVersion": "latest",
    "sourceType": "module"
  },
  "rules": {
    "no-unused-vars": "warn",
    "no-console": "off"
  }
}
""",
        }
    return {}


def _get_formatting_config(template: str) -> Dict[str, str]:
    """Get formatting configuration files."""
    if template == "python":
        return {
            "pyproject.toml": """[tool.black]
line-length = 120
target-version = ['py310']
include = '\\.pyi?$'
exclude = '''
/(
    \\.git
    | __pycache__
    | \\.venv
    | build
    | dist
)/
'''

[tool.isort]
profile = "black"
line_length = 120
""",
        }
    elif template in ("node", "javascript", "typescript"):
        return {
            ".prettierrc": """{
  "semi": true,
  "singleQuote": false,
  "tabWidth": 2,
  "trailingComma": "es5",
  "printWidth": 100
}
""",
            ".prettierignore": """node_modules
dist
build
coverage
""",
        }
    return {}


def _get_ci_config(template: str) -> Dict[str, str]:
    """Get CI/CD configuration files."""
    if template == "python":
        return {
            ".github/workflows/ci.yml": """name: CI

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.10", "3.11", "3.12"]

    steps:
    - uses: actions/checkout@v4

    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v5
      with:
        python-version: ${{ matrix.python-version }}

    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        pip install pytest pytest-cov

    - name: Run tests
      run: pytest --cov

    - name: Lint
      run: |
        pip install flake8
        flake8 .
""",
        }
    elif template in ("node", "javascript", "typescript"):
        return {
            ".github/workflows/ci.yml": """name: CI

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest

    steps:
    - uses: actions/checkout@v4

    - name: Setup Node.js
      uses: actions/setup-node@v4
      with:
        node-version: '20'
        cache: 'npm'

    - name: Install dependencies
      run: npm ci

    - name: Lint
      run: npm run lint --if-present

    - name: Test
      run: npm test --if-present
""",
        }
    return {}


def _get_docker_config(template: str) -> Dict[str, str]:
    """Get Docker configuration files."""
    if template == "python":
        return {
            "Dockerfile": """FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "-m", "src.main"]
""",
            ".dockerignore": """__pycache__
*.pyc
*.pyo
.git
.gitignore
.venv
venv
.env
*.md
tests
.pytest_cache
""",
        }
    elif template in ("node", "javascript"):
        return {
            "Dockerfile": """FROM node:20-alpine

WORKDIR /app

COPY package*.json ./
RUN npm ci --only=production

COPY . .

EXPOSE 3000

CMD ["node", "src/index.js"]
""",
            ".dockerignore": """node_modules
npm-debug.log
.git
.gitignore
.env
*.md
tests
coverage
""",
        }
    return {}


def _get_pre_commit_config(template: str) -> Dict[str, str]:
    """Get pre-commit configuration."""
    if template == "python":
        return {
            ".pre-commit-config.yaml": """repos:
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.5.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-added-large-files

  - repo: https://github.com/psf/black
    rev: 24.1.0
    hooks:
      - id: black

  - repo: https://github.com/pycqa/isort
    rev: 5.13.2
    hooks:
      - id: isort

  - repo: https://github.com/pycqa/flake8
    rev: 7.0.0
    hooks:
      - id: flake8
""",
        }
    elif template in ("node", "javascript", "typescript"):
        return {
            ".pre-commit-config.yaml": """repos:
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.5.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-json
      - id: check-added-large-files

  - repo: https://github.com/pre-commit/mirrors-prettier
    rev: v3.1.0
    hooks:
      - id: prettier
""",
        }
    return {}


def _get_license(license_type: str) -> str:
    """Get license text."""
    year = "2024"

    licenses = {
        "MIT": f"""MIT License

Copyright (c) {year}

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
""",
        "Apache-2.0": f"""Apache License
Version 2.0, January 2004
http://www.apache.org/licenses/

Copyright {year}

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
""",
    }

    return licenses.get(license_type, licenses["MIT"])
