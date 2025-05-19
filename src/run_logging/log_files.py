import wandb
from pathlib import Path


def log_files(agent_id, iteration=None):
    print(f"Logging files for agent {agent_id} with iteration {iteration}")
    dir_path = f"/home/jovyan/Vlasta/workspace/runs/{agent_id}" if iteration is not None else f"/home/jovyan/Vlasta/snapshots/{agent_id}" 
    files = get_python_files(dir_path)

    artifact_path = f"{agent_id}_iteration_{iteration}" if iteration is not None else agent_id
    artifact = wandb.Artifact(name=artifact_path,type='code')
    for file in files:
        artifact.add_file(file)
    wandb.log_artifact(artifact)

def get_python_files(path):
    run_dir = Path(path)

    py_files = []
    for element in run_dir.iterdir():
        if element.name.endswith('.py'):
            py_files.append(element)
    print(f"Found {len(py_files)} python files in {path}")
    return py_files