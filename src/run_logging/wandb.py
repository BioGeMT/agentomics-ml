import wandb
from dataclasses import asdict

def setup_logging(config, api_key, dir="/tmp/wandb"):
    wandb.login(key=api_key)
    wandb.init(
        dir=config.workspace_dir / dir,
        entity="ceitec-ai",
        project="Agentomics-ML",
        tags=config.tags,
        config=asdict(config),
        name=config.agent_id,
    )