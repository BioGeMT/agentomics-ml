import wandb

def setup_logging(config, api_key):
    wandb.login(key=api_key)
    wandb.init(
        dir="/tmp/wandb",
        entity="ceitec-ai",
        project="Agentomics-ML",
        tags=config["tags"],
        config=config,
        name=config["agent_id"],
    )