import wandb
import math
from wandb.errors import AuthenticationError, UsageError
from utils.metrics import get_task_to_metrics_names

def login_to_wandb(api_key):
    try:
        wandb.login(key=api_key, anonymous="allow", timeout=5)
        return True
    except (AuthenticationError, UsageError):
        return False

def is_wandb_active():
    try:
        return wandb.run is not None
    except:
        return False

def log_inference_stage_and_metrics(stage, task_type, metrics=None):
    if not is_wandb_active():
        return
    
    # Log to console
    stage_names = {0: "DRY_RUN", 1: "FAILED", 2: "TEST_EVALUATION"}
    stage_name = stage_names.get(stage, f"STAGE_{stage}")
    print(f"Wandb Logging - Stage: {stage_name}")
    
    wandb.log({"inference_stage": stage})
    if stage == 0 or stage == 1:
       metrics_names = get_task_to_metrics_names()[task_type]
       print(f"   Logging placeholder metrics: {metrics_names}")
       wandb.log({m: math.nan for m in metrics_names})
    else:
        if metrics:
            print(f"   Logging test metrics: {metrics}")
        wandb.log(metrics)

def log_serial_metrics(prefix, task_type, metrics=None, iteration=None):
    if not is_wandb_active():
        return
    
    if not metrics:
        metrics_names = get_task_to_metrics_names()[task_type]
        wandb.log({f"{prefix}/{m}": math.nan for m in metrics_names}, step=iteration)

    else:
        metrics = {f"{prefix}/{k}": v for k,v in metrics.items()}
        wandb.log(metrics, step=iteration)

def log_feedback_failure(message, iteration):
    if not is_wandb_active():
        return
    
    else:
        wandb.log({'feedback_exception_msg':message}, step=iteration)

def log_iteration_duration(iteration, duration):
    if not is_wandb_active():
        return
    wandb.log({'duration':duration}, step=iteration)

def log_new_best(iteration):
    if not is_wandb_active():
        return
    wandb.log({'validation/new_best':True}, step=iteration)

def log_test_inference_duration(duration):
    if not is_wandb_active():
        return
    wandb.log({'test_inference_duration':duration})
    