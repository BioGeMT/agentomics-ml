import wandb
from dataclasses import asdict

from wandb.errors import CommError
from run_logging.logging_helpers import login_to_wandb
import weave

def setup_logging(config, api_key, dir="/tmp/wandb"):
    #TODO handle if ppl want to use their own entity/project name!
    success = login_to_wandb(api_key)
    if not success:
        print("W&B login failed - skipping experiment logging")
        return False
    try:
        project_name = "Agentomics-ML"
        entity = "ceitec-ai"
        wandb.init(
            dir=config.workspace_dir / dir,
            entity=entity,
            project=project_name,
            tags=config.tags,
            config=asdict(config),
            name=config.agent_id,
        )
        weave.init(f"{entity}/{project_name}")
        return True
    except CommError:
        print("W&B initialization failed - skipping experiment logging")
        return False
