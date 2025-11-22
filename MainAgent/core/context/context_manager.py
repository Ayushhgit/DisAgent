class AgentContext:
    """
    Shared memory for agents. Combines memories, architecture output,
    scope analysis results, and generated files.
    """
    def __init__(self):
        self.memory = {}
        self.architecture = ""
        self.file_structure = {}
        self.scope_info = {}

    def add_result(self, agent_name: str, result: str):
        self.memory[agent_name] = result

    def get_context(self) -> str:
        ctx = "=== SHARED AGENT CONTEXT ===\n\n"

        if self.architecture:
            ctx += f"SYSTEM ARCHITECTURE:\n{self.architecture}\n\n"

        if self.scope_info:
            ctx += "SCOPE INFORMATION:\n"
            for k, v in self.scope_info.items():
                ctx += f"  {k}: {v}\n"
            ctx += "\n"

        for name, out in self.memory.items():
            truncated = out[:1500] if len(out) > 1500 else out
            ctx += f"[{name}]:\n{truncated}\n\n---\n\n"

        return ctx

    def set_architecture(self, arch: str):
        self.architecture = arch

    def set_scope(self, scope: dict):
        self.scope_info = scope

    def add_file(self, filename: str, content: str):
        self.file_structure[filename] = content
