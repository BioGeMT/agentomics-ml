import os
from pathlib import Path

from agentomics.datasets.data_contract import PREPARED_DATASETS_DIR_NAME
from agentomics.run_agent import initialize_run
from agentomics.runtime.git_checkpoints import commit_step_checkpoint
from agentomics.runtime.read_write_utils import (
    load_config_from_run_dir,
    load_dataset_metadata,
)
from agentomics.runtime.setup_fork import fork_run
from agentomics.utils.config import Config


def test_fork_creates_isolated_workspace_from_checkpoint(
    tmp_path: Path,
    config_factory,
):
    source_workspace = tmp_path / "source-workspace"
    target_workspace = tmp_path / "target-workspace"
    source_config = config_factory(
        agent_id="source",
        workspace_dir=str(source_workspace),
    )
    dataset_metadata = {
        "task_type": "classification",
        "input_structure": ["input"],
        "label_to_scalar": {"negative": 0, "positive": 1},
    }
    initialize_run(source_config, dataset_metadata)

    descriptor = source_config.shared_dir / Config.ENVIRONMENT_DESCRIPTOR_FILENAME
    descriptor.write_text(
        "channels:\n"
        "  - conda-forge\n"
        "dependencies:\n"
        "  - python=3.12\n",
        encoding="utf-8",
    )
    script = source_config.shared_dir / "train.py"
    checkpoint_script = f'data = "{source_workspace}/run/shared/data.csv"\n'
    script.write_text(checkpoint_script, encoding="utf-8")

    dataset_file = tmp_path / "datasets" / "toy" / "train" / "input" / "image.png"
    dataset_file.parent.mkdir(parents=True)
    dataset_file.write_text("pixel data", encoding="utf-8")
    split_input = source_config.splits_dir / "split_0" / "train" / "input"
    split_input.mkdir(parents=True)
    split_link = split_input / dataset_file.name
    split_link.symlink_to(dataset_file)

    commit_step_checkpoint(source_config, iteration=0, step_id="data_split")

    script.write_text("uncommitted source change\n", encoding="utf-8")
    transient_dataset = (
        source_workspace
        / PREPARED_DATASETS_DIR_NAME
        / "converted"
        / "marker.txt"
    )
    transient_dataset.parent.mkdir(parents=True)
    transient_dataset.write_text("transient", encoding="utf-8")
    report = source_workspace / "reports" / "markdown" / "report.md"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("report", encoding="utf-8")
    log = source_workspace / "logs" / "run_logs" / "latest.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text("log", encoding="utf-8")

    fork_run(
        source_workspace_dir=source_workspace,
        target_agent_id="target",
        target_workspace_dir=target_workspace,
        fork_from_step="data_split",
        fork_from_iteration=0,
    )

    target_config = load_config_from_run_dir(target_workspace / Config.RUN_DIRNAME)
    assert target_config.agent_id == "target"
    assert target_config.workspace_dir == str(target_workspace)
    assert load_dataset_metadata(target_config) == dataset_metadata

    target_script = target_config.shared_dir / script.name
    assert target_script.read_text(encoding="utf-8") == (
        f'data = "{target_workspace}/run/shared/data.csv"\n'
    )
    assert script.read_text(encoding="utf-8") == "uncommitted source change\n"

    target_split_link = (
        target_config.splits_dir / "split_0" / "train" / "input" / dataset_file.name
    )
    assert target_split_link.is_symlink()
    assert os.readlink(target_split_link) == str(dataset_file)
    assert target_split_link.read_text(encoding="utf-8") == "pixel data"

    assert not (target_workspace / PREPARED_DATASETS_DIR_NAME).exists()
    assert not (target_workspace / "reports").exists()
    assert not (target_workspace / "logs").exists()
