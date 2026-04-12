from __future__ import annotations

from pathlib import Path

import wandb

from agents.steps.data_split import DataSplitStep
from agents.steps.validation_evaluation import ValidationEvaluationStep
from run_logging.logging_helpers import is_wandb_active
from runtime.conda_utils import (
    export_environment_descriptor_to_path,
    get_shared_conda_root,
    get_shared_environment_path,
)
from runtime.filesystem import (
    copy_workspace_tree,
    remove_path,
)
from runtime.step_outputs import load_step_output

def update_best_run_snapshot(config, iteration: int) -> None:
    iteration_dir = config.iteration_dir(iteration)

    data_split = load_step_output(
        config,
        DataSplitStep.step_id,
        iteration_dir=iteration_dir,
    )
    validation_evaluation = load_step_output(
        config,
        ValidationEvaluationStep.step_id,
        iteration_dir=iteration_dir,
    )
    if data_split is None or validation_evaluation is None:
        return

    split_changed = bool(data_split.split_changed)
    if is_wandb_active():
        wandb.log({"validation/snapshot_reset": 1 if split_changed else 0}, step=iteration)

    if validation_evaluation.is_new_best:
        _publish_best_run_snapshot(
            config,
            source_dir=iteration_dir,
        )
        return

    if split_changed:
        remove_path(config.snapshot_dir)

def _publish_best_run_snapshot(config, source_dir: Path) -> None:
    snapshot_dir = config.snapshot_dir
    conda_source = get_shared_conda_root(config)
    conda_env = get_shared_environment_path(config)
    if not conda_env.exists():
        raise FileNotFoundError(
            f"Cannot publish best run snapshot because the shared Conda environment is missing at {conda_env}."
        )

    remove_path(snapshot_dir)
    copy_workspace_tree(source_dir, snapshot_dir)
    copy_workspace_tree(conda_source, snapshot_dir / ".conda")
    export_environment_descriptor_to_path(
        env_path=conda_env,
        descriptor_path=snapshot_dir / "environment.yml",
    )
