from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import wandb

from agentomics.agents.steps.data_split import DataSplitStep
from agentomics.agents.steps.validation_evaluation import ValidationEvaluationStep
from agentomics.run_logging.logging_helpers import is_wandb_active
from agentomics.runtime.conda_utils import (
    export_environment_archive,
    export_environment_descriptor_to_path,
    get_iteration_environment_archive_path,
    get_iteration_environment_descriptor_path,
)
from agentomics.runtime.filesystem import remove_path
from agentomics.runtime.step_outputs import load_step_output
from agentomics.utils.config import Config

def update_best_iteration_snapshot(config: Config, iteration: int) -> None:
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
        wandb.log({"validation/best_iteration_snapshot_reset": 1 if split_changed else 0}, step=iteration)

    if validation_evaluation.is_new_best:
        _publish_best_iteration_snapshot(
            config,
            source_dir=iteration_dir,
        )
        return

    if split_changed:
        remove_path(config.best_iteration_snapshot_dir)

def _publish_best_iteration_snapshot(config: Config, source_dir: Path) -> None:
    best_iteration_snapshot_dir = config.best_iteration_snapshot_dir
    conda_env = config.shared_environment_path
    if not conda_env.exists():
        raise FileNotFoundError(
            f"Cannot publish the best iteration snapshot because the shared Conda environment is missing at {conda_env}."
        )

    junk_names = {".conda", ".cache", "__pycache__"}
    remove_path(best_iteration_snapshot_dir)
    shutil.copytree(
        source_dir,
        best_iteration_snapshot_dir,
        symlinks=False,
        ignore=lambda _dir, names: {n for n in names if n in junk_names},
    )
    remove_path(best_iteration_snapshot_dir / Config.ENVIRONMENT_DESCRIPTOR_FILENAME)
    export_environment_descriptor_to_path(
        env_path=conda_env,
        descriptor_path=get_iteration_environment_descriptor_path(best_iteration_snapshot_dir),
    )
    if config.conda_export_mode == "full":
        archive_path = get_iteration_environment_archive_path(
            best_iteration_snapshot_dir
        )
        try:
            export_environment_archive(conda_env, archive_path)
        except (OSError, subprocess.CalledProcessError) as error:
            remove_path(archive_path)
            print(
                f"Warning: could not pack the best-iteration environment: {error}. "
                "The environment will be rebuilt from environment.yml when needed."
            )
