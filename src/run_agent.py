import subprocess
import os
from utils.create_user import create_new_user_and_rundir

agent_id = create_new_user_and_rundir()

conda_env_path = "/opt/conda"       
agent_run_path = f"/workspace/runs/{agent_id}"

bwrap_command = [
    "bwrap",
    "--ro-bind", "/usr", "/usr",
    "--ro-bind", "/lib", "/lib",
    "--ro-bind", "/lib64", "/lib64",
    "--ro-bind", "/bin", "/bin",
    "--ro-bind", "/etc", "/etc",
    "--ro-bind", "/usr/local/lib", "/usr/local/lib",
    "--ro-bind", "/usr/local/bin", "/usr/local/bin",
    "--ro-bind", "/opt/conda", "/opt/conda",
    "--proc", "/proc",
    "--dev", "/dev",
    "--tmpfs", "/tmp",
    
    "--bind", agent_run_path, agent_run_path,
    
    # "--ro-bind", "/workspace/runs/datasets", "/workspace/runs/datasets",
    "--ro-bind", "/workspace/datasets", "/workspace/runs/datasets",
    "--ro-bind", "/workspace/datasets", "/workspace/datasets",

    "--ro-bind", "/repository", "/repository",    
    "--setenv", "HOME", agent_run_path,
    "--setenv", "AGENT_ID", agent_id, # Set env variable to pass the agent_id to playground.py, probably it's better to have it as a argument ?
    # "--setenv", "PYTHONPATH", "/repository/src",
    
    "--setenv", "PATH", "/opt/conda/condabin:/opt/conda/envs/agentomics-env/bin:/usr/bin:/bin",
    
    "--chdir", agent_run_path,
    "/opt/conda/envs/agentomics-env/bin/python", "/repository/src/playground.py"
    #"/bin/bash"
]

subprocess.run(bwrap_command)
