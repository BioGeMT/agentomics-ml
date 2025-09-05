import wandb
from wandb.errors import AuthenticationError
from utils.metrics import get_task_to_metrics_names

def login_to_wandb(api_key):
    try:
        wandb.login(key=api_key, anonymous="allow", timeout=5)
        return True
    except AuthenticationError:
        return False

def is_wandb_active():
    try:
        return wandb.run is not None
    except:
        return False

def log_inference_stage_and_metrics(stage, task_type, metrics=None):
    if not is_wandb_active():
        return
    
    wandb.log({"inference_stage": stage})
    if stage == 0 or stage == 1:
       metrics_names = get_task_to_metrics_names()[task_type]
       wandb.log({m: -1 for m in metrics_names})
    else:
        wandb.log(metrics)

def log_serial_metrics(prefix, task_type, metrics=None, iteration=None):
    if not is_wandb_active():
        return
    
    if(not metrics):
        metrics_names = get_task_to_metrics_names()[task_type]
        wandb.log({f"{prefix}/{m}": -1 for m in metrics_names}, step=iteration)

    else:
        metrics = {f"{prefix}/{k}": v for k,v in metrics.items()}
        wandb.log(metrics, step=iteration)
   