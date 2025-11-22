from .base_agent import BaseAgent

class FolderAgent(BaseAgent):
    def act(self, path):
        return f"FolderAgent {self.name} working on {path}"
