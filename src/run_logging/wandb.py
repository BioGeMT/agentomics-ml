import wandb
from dataclasses import asdict

def setup_logging(config, api_key, dir="/tmp/wandb"):
    if not api_key:
        print("⚠️  No WANDB_API_KEY provided - WandB logging disabled")
        return None
    
    try:
        wandb.login(key=api_key, anonymous="allow", timeout=5)
        return wandb.init(
            dir=config.workspace_dir / dir,
            entity="ceitec-ai",
            project="Agentomics-ML",
            tags=config.tags,
            config=asdict(config),
            name=config.agent_id,
        )
    except Exception as e:
        print(f"⚠️  Failed to initialize WandB: {e}")
        return None