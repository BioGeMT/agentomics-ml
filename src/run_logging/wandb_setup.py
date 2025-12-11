import wandb
from dataclasses import asdict
import dotenv
import os
from pathlib import Path

from wandb.errors import CommError
from run_logging.logging_helpers import login_to_wandb
import weave

def wandb_is_configured() -> bool:
    """Return True if W&B looks configured; False otherwise."""
    # Explicit "off" switches
    if os.environ.get("WANDB_MODE", "").lower() == "disabled":
        return False
    if os.environ.get("WANDB_DISABLED", "").lower() in ("true", "1", "yes"):
        return False

    # API key present → configured
    if os.environ.get("WANDB_API_KEY"):
        return True

    # netrc present → usually means wandb login was done
    netrc_path = Path.home() / ".netrc"
    if netrc_path.exists():
        return True

    return False


def setup_logging(config, dir=None):
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
            dir=config.extras_dir / 'run_logs' if dir is None else dir,
            entity=wandb_entity,
            project=wandb_project_name,
            tags=config.tags,
            config=asdict(config),
            name=config.agent_id,
        )
        config.wandb_run_id = wandb.run.id
        if wandb_entity and wandb_project_name:
            weave.init(f"{wandb_entity}/{wandb_project_name}")
        return True
    except CommError:
        print("W&B initialization failed - skipping experiment logging")
        return False

def resume_wandb_run(config, dir=None):

    if not wandb_is_configured():
        os.environ.setdefault("WANDB_MODE", "disabled")
        print("W&B not configured; skipping wandb resume in test evaluation.")
        return None

    api_key = os.getenv("WANDB_API_KEY")
    wandb_project_name = os.getenv("WANDB_PROJECT_NAME")
    wandb_entity = os.getenv("WANDB_ENTITY")

    success = login_to_wandb(api_key)
    if not success:
        print("W&B login failed - cannot resume logging run")
        return False
    else:
        wandb.init(
            dir=config.extras_dir / 'test_logs' if dir is None else dir,
            id = config.wandb_run_id,
            project=wandb_project_name,
            entity=wandb_entity,
            resume="must"
        )
        return True