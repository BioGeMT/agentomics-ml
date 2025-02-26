import wandb

def setup_logging(config, api_key):
    wandb.login(key=api_key)
    wandb.init(
        entity="ceitec-ai",
        project="BioAgents",
        tags=config["tags"],
        config=config,
    )