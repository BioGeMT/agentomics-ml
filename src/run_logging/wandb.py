import wandb

def setup_logging(config, api_key, dir="/home/jovyan/Vlasta/tmp/wandb"):
    wandb.login(key=api_key)
    wandb.init(
        dir=dir,
        entity="ceitec-ai",
        project="Agentomics-ML",
        tags=config["tags"],
        config=config,
        name=config["agent_id"],
    )