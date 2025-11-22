class BaseAgent:
    def __init__(self, name: str):
        self.name = name

    def act(self, *args, **kwargs):
        raise NotImplementedError()
