import subprocess

def run_command(cmd: str):
    proc = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return proc.returncode, proc.stdout, proc.stderr
