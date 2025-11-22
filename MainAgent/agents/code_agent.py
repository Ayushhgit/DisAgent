from .base_agent import BaseAgent

class CodeAgent(BaseAgent):
    def act(self, code):
        return f"CodeAgent {self.name} received code of length {len(code)}"
