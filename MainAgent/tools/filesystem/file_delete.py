import os

def delete_file(path: str):
    os.remove(path)
    return True
