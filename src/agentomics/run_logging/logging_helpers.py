import math
from pathlib import Path

import wandb
from wandb.errors import AuthenticationError, UsageError

from agentomics.utils.metrics import get_higher_is_better_map, get_task_to_metrics_names
from agentomics.utils.config import Config


def login_to_wandb(api_key):
    try:
        wandb.login(key=api_key, timeout=5)
        return True
    except (AuthenticationError, UsageError):
        return False

def is_wandb_active():
    try:
        return wandb.run is not None
    except:
        return False

def define_serial_metrics(prefix, task_type, step_metric):
    if not is_wandb_active():
        return
    for metric_name in get_task_to_metrics_names()[task_type]:
        wandb.define_metric(f"{prefix}/{metric_name}", step_metric=step_metric)

def log_serial_metrics(prefix, task_type, metrics=None, iteration=None, step_metric=None):
    if not is_wandb_active():
        return
    metrics_names = get_task_to_metrics_names()[task_type]
    payload = {f"{prefix}/{metric_name}": math.nan for metric_name in metrics_names}
    if metrics:
        payload.update({f"{prefix}/{key}": value for key, value in metrics.items()})
    if step_metric is None:
        wandb.log(payload, step=iteration)
    else:
        payload[step_metric] = iteration
        wandb.log(payload)

def log_iteration_metrics(metrics, iteration, task_type):
    for prefix in ("validation", "train"):
        split_metrics = {
            key.split("/", maxsplit=1)[1]: value
            for key, value in metrics.items()
            if key.startswith(f"{prefix}/")
        }
        log_serial_metrics(
            prefix=prefix,
            task_type=task_type,
            metrics=split_metrics or None,
            iteration=iteration,
        )

def log_run_metrics(iteration, iteration_state, token_usage):
    values = {
        'duration': float(iteration_state['ended_at']) - float(iteration_state['started_at']),
        'tokens/input': sum(usage['input_tokens'] for usage in token_usage.values()),
        'tokens/output': sum(usage['output_tokens'] for usage in token_usage.values()),
        'iteration/failed': int(iteration_state['status'] != 'success'),
    }
    if not is_wandb_active():
        return values
    summary = wandb.run.summary
    values['duration_total'] = summary.get('duration_total', 0) + values['duration']
    values['tokens/input_total'] = summary.get('tokens/input_total', 0) + values['tokens/input']
    values['tokens/output_total'] = summary.get('tokens/output_total', 0) + values['tokens/output']
    values['iterations/failed'] = summary.get('iterations/failed', 0) + values['iteration/failed']
    wandb.log(values, step=iteration)
    return values

def log_main_metric(iteration, metric_name, metrics):
    value = metrics.get(f'validation/{metric_name}', math.nan)
    if not is_wandb_active():
        return value
    summary = 'max' if get_higher_is_better_map()[metric_name] else 'min'
    wandb.define_metric('main/val_metric', summary=summary)
    wandb.log({'main/val_metric': value}, step=iteration)
    return value

def log_split_is_allowed(iteration, is_allowed):
    if not is_wandb_active():
        return
    wandb.log({'validation/splitting_allowed':int(is_allowed)}, step=iteration)

def log_new_best(iteration, is_new_best: bool):
    if not is_wandb_active():
        return
    wandb.log({'validation/new_best': is_new_best}, step=iteration)

def log_test_inference_duration(duration):
    if not is_wandb_active():
        return
    wandb.log({'test_inference_duration':duration})

def log_files(config: Config, files=None, iteration=None):
    if not is_wandb_active():
        return

    if iteration is not None:
        directory_path = config.current_iteration_dir
    else:
        directory_path = config.best_iteration_snapshot_dir

    files_to_log = (
        [f for f in Path(directory_path).iterdir() if f.is_file() and f.suffix == ".py"]
        if files is None
        else files
    )
    artifact_name = f"{config.agent_id}_iteration_{iteration}" if iteration is not None else config.agent_id
    artifact = wandb.Artifact(name=artifact_name, type="code")
    for file_path in files_to_log:
        artifact.add_file(file_path)
    wandb.log_artifact(artifact)
