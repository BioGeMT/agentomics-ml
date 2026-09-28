from pathlib import Path

from agentomics.datasets.data_contract import PREPARED_DATASETS_DIR_NAME
from agentomics.runtime.git_checkpoints import commit_step_checkpoint
from agentomics.utils.config import Config
from tests.helpers import run_git_cli_command


def test_checkpoint_commit_excludes_prepared_dataset_artifacts(
    initialized_run_config: Config,
):
    workspace_dir = Path(initialized_run_config.workspace_dir)
    transient_file = workspace_dir / PREPARED_DATASETS_DIR_NAME / "artifact.txt"
    transient_file.parent.mkdir()
    transient_file.write_text("transient", encoding="utf-8")

    commit_step_checkpoint(initialized_run_config, iteration=0, step_id="data_split")

    tracked_files = set(
        run_git_cli_command(workspace_dir, "ls-files").stdout.splitlines()
    )
    tracked_config = initialized_run_config.config_path.relative_to(workspace_dir)
    assert tracked_config.as_posix() in tracked_files
    assert transient_file.relative_to(workspace_dir).as_posix() not in tracked_files
