from .base_agent import BaseAgent

class ReviewAgent(BaseAgent):
    def act(self, artifact):
        return f"ReviewAgent {self.name} reviewing {artifact}"
