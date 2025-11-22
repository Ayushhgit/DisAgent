def write_file(path: str, content: str):
    with open(path, 'w', encoding='utf-8') as f:
        f.write(content)
    return True
