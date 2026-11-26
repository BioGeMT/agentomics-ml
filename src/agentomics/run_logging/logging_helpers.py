import math
from pathlib import Path

import wandb
from wandb.errors import AuthenticationError, UsageError

from agentomics.runtime.read_write_utils import get_archived_iterations, load_iteration_state
from agentomics.runtime.step_outputs import load_step_output
from agentomics.utils.ml_tasks import get_task_to_metrics_names
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

def log_iteration_duration(iteration, duration):
    if not is_wandb_active():
        return
    wandb.log({'duration':duration}, step=iteration)

def log_iteration_failure(iteration, step_id, exception_trace):
    if not is_wandb_active():
        return
    wandb.log({'failure/step': step_id, 'failure/exception_trace': exception_trace}, step=iteration)

def log_architecture_table(config: Config, iteration: int) -> None:
    if not is_wandb_active():
        return
    rows = []
    for archived_iteration in get_archived_iterations(config):
        iteration_dir = config.iteration_dir(archived_iteration)
        planned = _step_payload(config, "model_architecture", iteration_dir)
        trained = _step_payload(config, "model_training", iteration_dir).get("trained_categories", {})
        state = load_iteration_state(iteration_dir)
        rows.append([
            archived_iteration,
            ", ".join(planned.get("categories", {}).get("families", [])),
            ", ".join(trained.get("families", [])),
            trained.get("is_ensemble"),
            planned.get("architecture"),
            planned.get("hyperparameters"),
            f"failed at {state['failed_step']}" if "failed_step" in state else state["status"],
        ])
    columns = ["iteration", "planned", "trained", "ensemble", "architecture", "hyperparameters", "outcome"]
    wandb.log({"architectures": wandb.Table(columns=columns, data=rows)}, step=iteration)

def _step_payload(config: Config, step_id: str, iteration_dir: Path) -> dict:
    output = load_step_output(config, step_id, iteration_dir)
    return output.model_dump() if hasattr(output, "model_dump") else output or {}

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
