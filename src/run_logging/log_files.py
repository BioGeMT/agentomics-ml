import wandb

def log_files(files, agent_id):
    artifact = wandb.Artifact(agent_id,type='code')
    for file in files:
        artifact.add_file(file)
    wandb.log_artifact(artifact)