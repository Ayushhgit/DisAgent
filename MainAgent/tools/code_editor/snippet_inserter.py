def insert_snippet(content: str, snippet: str, pos: int=0):
    return content[:pos] + snippet + content[pos:]
