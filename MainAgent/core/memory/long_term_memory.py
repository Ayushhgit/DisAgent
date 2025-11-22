class LongTermMemory:
    def __init__(self):
        self._store = {}
    def save(self, key, value):
        self._store[key] = value
    def load(self, key):
        return self._store.get(key)
