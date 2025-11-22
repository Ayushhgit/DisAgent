import os

def search_files(root: str, pattern: str):
    matches = []
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if pattern in fn:
                matches.append(os.path.join(dirpath, fn))
    return matches
