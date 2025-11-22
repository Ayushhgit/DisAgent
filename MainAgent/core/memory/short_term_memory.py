class ShortTermMemory:
    def __init__(self):
        self._store = []
    def add(self, item):
        self._store.append(item)
    def get(self):
        return list(self._store)
