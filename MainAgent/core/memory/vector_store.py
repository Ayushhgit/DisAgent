class VectorStore:
    def __init__(self):
        self.vectors = []
    def add(self, v):
        self.vectors.append(v)
    def query(self, q):
        return []
