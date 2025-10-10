import wandb
from dataclasses import asdict
import dotenv
import os

from wandb.errors import CommError
from run_logging.logging_helpers import login_to_wandb
import weave

def setup_logging(config, dir="/tmp/wandb"):
    dotenv.load_dotenv()
    api_key = os.getenv("WANDB_API_KEY")
    wandb_project_name = os.getenv("WANDB_PROJECT_NAME")
    wandb_entity = os.getenv("WANDB_ENTITY")

    success = login_to_wandb(api_key)
    if not success:
        print("W&B login failed - skipping experiment logging")
        return False
    try:
        wandb.init(
            dir=config.runs_dir / dir,
            entity=wandb_entity,
            project=wandb_project_name,
            tags=config.tags,
            config=asdict(config),
            name=config.agent_id,
        )
        if wandb_entity and wandb_project_name:
            weave.init(f"{wandb_entity}/{wandb_project_name}")
        return True
    except CommError:
        print("W&B initialization failed - skipping experiment logging")
        return False
