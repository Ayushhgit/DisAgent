"""
Project Initialization - Create project directory structure.

This workflow creates the initial directory structure and files
for new projects based on templates.
"""

from __future__ import annotations

import os
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class ProjectType(Enum):
    """Types of projects."""
    PYTHON_BASIC = "python_basic"
    PYTHON_PACKAGE = "python_package"
    PYTHON_WEB = "python_web"
    NODE_BASIC = "node_basic"
    NODE_EXPRESS = "node_express"
    REACT = "react"
    GENERIC = "generic"


@dataclass
class ProjectConfig:
    """Project configuration."""
    name: str
    project_type: ProjectType = ProjectType.PYTHON_BASIC
    description: str = ""
    author: str = ""
    version: str = "0.1.0"
    use_git: bool = True
    include_tests: bool = True
    include_docs: bool = False


@dataclass
class InitResult:
    """Result of project initialization."""
    success: bool
    path: str
    files_created: List[str] = field(default_factory=list)
    directories_created: List[str] = field(default_factory=list)
    error: Optional[str] = None


def init_structure(destination: str, config: Optional[ProjectConfig] = None) -> bool:
    """Initialize project structure (legacy interface).

    Args:
        destination: Destination directory
        config: Optional project configuration

    Returns:
        True if successful
    """
    if config is None:
        # Create default config from directory name
        name = Path(destination).name or "project"
        config = ProjectConfig(name=name)

    result = create_project(destination, config)
    return result.success


def create_project(
    destination: str,
    config: ProjectConfig,
) -> InitResult:
    """Create a new project with the specified structure.

    Args:
        destination: Destination directory
        config: Project configuration

    Returns:
        InitResult with created files and directories
    """
    dest_path = Path(destination).resolve()
    files_created = []
    dirs_created = []

    try:
        # Create root directory
        dest_path.mkdir(parents=True, exist_ok=True)
        dirs_created.append(str(dest_path))

        # Get structure template
        structure = _get_project_structure(config)

        # Create directories
        for dir_path in structure.get("directories", []):
            full_path = dest_path / dir_path
            full_path.mkdir(parents=True, exist_ok=True)
            dirs_created.append(str(full_path))

        # Create files
        for file_path, content in structure.get("files", {}).items():
            full_path = dest_path / file_path
            full_path.parent.mkdir(parents=True, exist_ok=True)

            # Format content with config values
            formatted_content = content.format(
                name=config.name,
                description=config.description,
                author=config.author,
                version=config.version,
            )

            full_path.write_text(formatted_content, encoding='utf-8')
            files_created.append(str(full_path))

        # Initialize git if requested
        if config.use_git:
            _init_git(dest_path)

        return InitResult(
            success=True,
            path=str(dest_path),
            files_created=files_created,
            directories_created=dirs_created,
        )

    except Exception as exc:
        logger.exception(f"Failed to create project: {exc}")
        return InitResult(
            success=False,
            path=str(dest_path),
            error=str(exc),
        )


def _get_project_structure(config: ProjectConfig) -> Dict[str, Any]:
    """Get project structure template based on type."""
    templates = {
        ProjectType.PYTHON_BASIC: _python_basic_structure,
        ProjectType.PYTHON_PACKAGE: _python_package_structure,
        ProjectType.PYTHON_WEB: _python_web_structure,
        ProjectType.NODE_BASIC: _node_basic_structure,
        ProjectType.NODE_EXPRESS: _node_express_structure,
        ProjectType.REACT: _react_structure,
        ProjectType.GENERIC: _generic_structure,
    }

    template_func = templates.get(config.project_type, _generic_structure)
    return template_func(config)


def _python_basic_structure(config: ProjectConfig) -> Dict[str, Any]:
    """Basic Python project structure."""
    dirs = ["src", "tests"]
    if config.include_docs:
        dirs.append("docs")

    files = {
        "README.md": """# {name}

{description}

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```python
from src import main
main.run()
```

## Author

{author}
""",
        "requirements.txt": """# Project dependencies
""",
        ".gitignore": """# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
dist/
*.egg-info/
.eggs/

# Virtual environment
venv/
.venv/
ENV/

# IDE
.idea/
.vscode/
*.swp
*.swo

# Testing
.pytest_cache/
.coverage
htmlcov/

# Misc
.DS_Store
*.log
""",
        "src/__init__.py": '''"""
{name} - {description}
"""

__version__ = "{version}"
''',
        "src/main.py": '''"""
Main module for {name}.
"""


def run():
    """Run the application."""
    print("Hello from {name}!")


if __name__ == "__main__":
    run()
''',
    }

    if config.include_tests:
        files["tests/__init__.py"] = ""
        files["tests/test_main.py"] = '''"""
Tests for the main module.
"""

import pytest
from src import main


def test_run(capsys):
    """Test run function."""
    main.run()
    captured = capsys.readouterr()
    assert "{name}" in captured.out
'''

    return {"directories": dirs, "files": files}


def _python_package_structure(config: ProjectConfig) -> Dict[str, Any]:
    """Python package structure."""
    package_name = config.name.replace("-", "_").lower()
    dirs = [package_name, "tests"]
    if config.include_docs:
        dirs.append("docs")

    files = {
        "README.md": """# {name}

{description}

## Installation

```bash
pip install {name}
```

## Usage

```python
import {name}
```

## Development

```bash
pip install -e ".[dev]"
pytest
```
""".replace("{name}", config.name).format(name=config.name, description=config.description),
        "pyproject.toml": f'''[build-system]
requires = ["setuptools>=61.0"]
build-backend = "setuptools.build_meta"

[project]
name = "{config.name}"
version = "{config.version}"
description = "{config.description}"
authors = [{{name = "{config.author}"}}]
readme = "README.md"
requires-python = ">=3.8"
classifiers = [
    "Programming Language :: Python :: 3",
    "License :: OSI Approved :: MIT License",
    "Operating System :: OS Independent",
]

[project.optional-dependencies]
dev = ["pytest", "pytest-cov", "black", "mypy"]

[tool.pytest.ini_options]
testpaths = ["tests"]
''',
        ".gitignore": """# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
dist/
*.egg-info/
.eggs/

# Virtual environment
venv/
.venv/

# IDE
.idea/
.vscode/

# Testing
.pytest_cache/
.coverage
htmlcov/
""",
        f"{package_name}/__init__.py": f'''"""
{config.name} - {config.description}
"""

__version__ = "{config.version}"
''',
        f"{package_name}/core.py": f'''"""
Core functionality for {config.name}.
"""


class {config.name.replace("-", "").title()}:
    """Main class."""

    def __init__(self):
        """Initialize."""
        pass

    def run(self):
        """Run the main functionality."""
        return "Hello from {config.name}!"
''',
    }

    if config.include_tests:
        files["tests/__init__.py"] = ""
        files["tests/test_core.py"] = f'''"""
Tests for core module.
"""

import pytest
from {package_name}.core import {config.name.replace("-", "").title()}


def test_run():
    """Test run method."""
    obj = {config.name.replace("-", "").title()}()
    result = obj.run()
    assert "{config.name}" in result
'''

    return {"directories": dirs, "files": files}


def _python_web_structure(config: ProjectConfig) -> Dict[str, Any]:
    """Python web application structure (Flask-like)."""
    dirs = ["app", "app/templates", "app/static", "tests"]

    files = {
        "README.md": """# {name}

{description}

## Setup

```bash
pip install -r requirements.txt
python run.py
```

## Development

Visit http://localhost:5000
""",
        "requirements.txt": """flask>=2.0
python-dotenv
gunicorn
""",
        ".gitignore": """__pycache__/
*.py[cod]
.env
venv/
.venv/
instance/
*.db
""",
        "run.py": """from app import create_app

app = create_app()

if __name__ == "__main__":
    app.run(debug=True)
""",
        "app/__init__.py": '''"""
Application factory.
"""

from flask import Flask


def create_app():
    """Create and configure the app."""
    app = Flask(__name__)
    app.config["SECRET_KEY"] = "dev"

    from . import routes
    app.register_blueprint(routes.bp)

    return app
''',
        "app/routes.py": '''"""
Application routes.
"""

from flask import Blueprint, render_template

bp = Blueprint("main", __name__)


@bp.route("/")
def index():
    """Home page."""
    return render_template("index.html")
''',
        "app/templates/index.html": """<!DOCTYPE html>
<html>
<head>
    <title>{name}</title>
</head>
<body>
    <h1>Welcome to {name}</h1>
    <p>{description}</p>
</body>
</html>
""",
    }

    return {"directories": dirs, "files": files}


def _node_basic_structure(config: ProjectConfig) -> Dict[str, Any]:
    """Basic Node.js project structure."""
    dirs = ["src", "tests"]

    files = {
        "README.md": """# {name}

{description}

## Installation

```bash
npm install
```

## Usage

```bash
npm start
```
""",
        "package.json": f'''{{
  "name": "{config.name}",
  "version": "{config.version}",
  "description": "{config.description}",
  "main": "src/index.js",
  "scripts": {{
    "start": "node src/index.js",
    "test": "jest"
  }},
  "author": "{config.author}",
  "license": "MIT",
  "devDependencies": {{
    "jest": "^29.0.0"
  }}
}}
''',
        ".gitignore": """node_modules/
dist/
.env
*.log
.DS_Store
""",
        "src/index.js": f'''/**
 * {config.name} - {config.description}
 */

console.log("Hello from {config.name}!");
''',
    }

    if config.include_tests:
        files["tests/index.test.js"] = f'''/**
 * Tests for {config.name}
 */

describe("{config.name}", () => {{
  test("should work", () => {{
    expect(true).toBe(true);
  }});
}});
'''

    return {"directories": dirs, "files": files}


def _node_express_structure(config: ProjectConfig) -> Dict[str, Any]:
    """Express.js application structure."""
    dirs = ["src", "src/routes", "src/middleware", "tests"]

    files = {
        "README.md": """# {name}

{description}

## Setup

```bash
npm install
npm start
```

Visit http://localhost:3000
""",
        "package.json": f'''{{
  "name": "{config.name}",
  "version": "{config.version}",
  "description": "{config.description}",
  "main": "src/app.js",
  "scripts": {{
    "start": "node src/app.js",
    "dev": "nodemon src/app.js",
    "test": "jest"
  }},
  "author": "{config.author}",
  "license": "MIT",
  "dependencies": {{
    "express": "^4.18.0"
  }},
  "devDependencies": {{
    "jest": "^29.0.0",
    "nodemon": "^3.0.0"
  }}
}}
''',
        ".gitignore": """node_modules/
.env
*.log
""",
        "src/app.js": f'''const express = require("express");
const routes = require("./routes");

const app = express();
const PORT = process.env.PORT || 3000;

app.use(express.json());
app.use("/", routes);

app.listen(PORT, () => {{
  console.log(`{config.name} running on port ${{PORT}}`);
}});

module.exports = app;
''',
        "src/routes/index.js": '''const express = require("express");
const router = express.Router();

router.get("/", (req, res) => {
  res.json({ message: "Welcome to the API" });
});

module.exports = router;
''',
    }

    return {"directories": dirs, "files": files}


def _react_structure(config: ProjectConfig) -> Dict[str, Any]:
    """React application structure."""
    dirs = ["src", "src/components", "public"]

    files = {
        "README.md": """# {name}

{description}

## Setup

```bash
npm install
npm start
```
""",
        "package.json": f'''{{
  "name": "{config.name}",
  "version": "{config.version}",
  "description": "{config.description}",
  "scripts": {{
    "start": "vite",
    "build": "vite build",
    "preview": "vite preview"
  }},
  "dependencies": {{
    "react": "^18.2.0",
    "react-dom": "^18.2.0"
  }},
  "devDependencies": {{
    "@vitejs/plugin-react": "^4.0.0",
    "vite": "^5.0.0"
  }}
}}
''',
        ".gitignore": """node_modules/
dist/
.env
""",
        "vite.config.js": '''import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
});
''',
        "index.html": """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>{name}</title>
</head>
<body>
  <div id="root"></div>
  <script type="module" src="/src/main.jsx"></script>
</body>
</html>
""",
        "src/main.jsx": '''import React from "react";
import ReactDOM from "react-dom/client";
import App from "./App";

ReactDOM.createRoot(document.getElementById("root")).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);
''',
        "src/App.jsx": '''function App() {
  return (
    <div>
      <h1>{name}</h1>
      <p>{description}</p>
    </div>
  );
}

export default App;
''',
    }

    return {"directories": dirs, "files": files}


def _generic_structure(config: ProjectConfig) -> Dict[str, Any]:
    """Generic project structure."""
    dirs = ["src", "docs"]

    files = {
        "README.md": """# {name}

{description}

## Getting Started

TODO: Add instructions

## Author

{author}
""",
        ".gitignore": """# Common
.DS_Store
*.log
*.tmp
.env
""",
    }

    return {"directories": dirs, "files": files}


def _init_git(path: Path) -> bool:
    """Initialize a git repository."""
    import subprocess

    try:
        subprocess.run(
            ["git", "init"],
            cwd=str(path),
            capture_output=True,
            check=True,
        )
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        logger.warning("Failed to initialize git repository")
        return False
