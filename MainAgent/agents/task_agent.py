from .base_agent import BaseAgent

class TaskAgent(BaseAgent):
    def act(self, task):
        return f"TaskAgent {self.name} handling {task}"
