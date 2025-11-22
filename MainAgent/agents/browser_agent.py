from .base_agent import BaseAgent

class BrowserAgent(BaseAgent):
    def act(self, query):
        return f"BrowserAgent {self.name} searching for {query}"
