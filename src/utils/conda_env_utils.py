import os
from pathlib import Path

def get_shared_env_name():
    return os.environ.get("AGENTOMICS_TOOL_ENV")

def conda_run_prefix(env_path: Path, capture_output: bool = True) -> str:
    shared_env = get_shared_env_name()
    if shared_env:
        prefix = f"conda run -n {shared_env}"
    else:
        prefix = f"conda run -p {env_path}"
    if capture_output:
        prefix += " --no-capture-output"
    return prefix

def conda_env_export_command(env_path: Path, output_path: Path):
    shared_env = get_shared_env_name()
    if shared_env:
        return ["conda", "env", "export", "-n", shared_env, "-f", str(output_path)]
    return ["conda", "env", "export", "-p", str(env_path), "-f", str(output_path)]
