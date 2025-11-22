# MainAgent - Multi-Agent Orchestration System

A dynamic multi-agent system for building full-stack applications using LLM-powered agents.

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure API Key

You need to set your Groq API key. Choose one of these methods:

**Option A: Environment Variable (Recommended)**
```powershell
# Windows PowerShell
$env:GROQ_API_KEY="your_api_key_here"

# Or set permanently:
[System.Environment]::SetEnvironmentVariable('GROQ_API_KEY', 'your_api_key_here', 'User')
```

**Option B: Update config.py**
Edit `config.py` and set:
```python
CONFIG = {
    "groq_api_key": "your_api_key_here",
    ...
}
```

### 3. Run

```bash
# Dynamic orchestrator (with file generation)
python MainAgent/main.py --mode dynamic --prompt "Your project description"

# Planning orchestrator (architecture planning)
python MainAgent/main.py --mode planning --prompt "Your project description"
```

## Project Structure

- `orchestrator/` - Orchestration workflows and agent management
- `core/` - Core utilities (context, memory, planning, runtime)
- `agents/` - Specialized agent implementations
- `tools/` - Agent tools (browser, code editor, filesystem)
- `workflows/` - Reusable workflow patterns
