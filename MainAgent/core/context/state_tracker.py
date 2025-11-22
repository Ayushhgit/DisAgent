class StateTracker:
    def __init__(self):
        self.log = []
    def track(self, item):
        self.log.append(item)
    def history(self):
        return list(self.log)
