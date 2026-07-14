import wandb
import os

from wandb.errors import CommError
from agentomics.run_logging.logging_helpers import login_to_wandb
from agentomics.utils.config import Config
import weave


def setup_logging(config: Config, dir=None):
    api_key = os.getenv("WANDB_API_KEY")
    wandb_project_name = os.getenv("WANDB_PROJECT_NAME")
    wandb_entity = os.getenv("WANDB_ENTITY")

    os.environ.setdefault("WANDB_SILENT", "true")
    os.environ.setdefault("WEAVE_LOG_LEVEL", "ERROR")

    success = login_to_wandb(api_key)
    if not success:
        print("W&B login failed - skipping experiment logging")
        return None
    try:
        wandb.init(
            dir=config.extras_dir / 'run_logs' if dir is None else dir,
            entity=wandb_entity,
            project=wandb_project_name,
            tags=config.tags,
            config=vars(config),
            name=config.agent_id,
        )
        if wandb_entity and wandb_project_name:
            weave.init(f"{wandb_entity}/{wandb_project_name}")
        return wandb.run.id
    except CommError:
        print("W&B initialization failed - skipping experiment logging")
        return None

def resume_wandb_run(config: Config, dir=None):
    api_key = os.getenv("WANDB_API_KEY")
    wandb_project_name = os.getenv("WANDB_PROJECT_NAME")
    wandb_entity = os.getenv("WANDB_ENTITY")

    if not (api_key and wandb_project_name and wandb_entity):
        return None
    wandb_run_id = config.wandb_run_id
    if not wandb_run_id:
        return None
    success = login_to_wandb(api_key)
    if not success:
        print("W&B login failed - cannot resume logging run")
        return None        
    try:
        run = wandb.init(
            dir=config.extras_dir / 'test_logs' if dir is None else dir,
            id=wandb_run_id,
            project=wandb_project_name,
            entity=wandb_entity,
            resume="allow"
        )
        return run
    except CommError:
        print("W&B login failed - cannot resume logging run")
        return None
