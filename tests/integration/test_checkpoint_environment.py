import subprocess
from pathlib import Path

import pytest

from agentomics.run_agent import initialize_run
from agentomics.runtime.conda_utils import (
    create_environment_from_descriptor,
    export_environment_archive,
    export_environment_descriptor_to_path,
    get_iteration_environment_archive_path,
    initialize_shared_environment,
    restore_iteration_environment,
)
from agentomics.runtime.git_checkpoints import commit_step_checkpoint
from agentomics.runtime.read_write_utils import load_config_from_run_dir
from agentomics.runtime.setup_fork import fork_run
from agentomics.utils.config import Config
from tests.helpers import run_git_cli_command


@pytest.fixture(scope="module")
def source_checkpoint_environment(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("checkpoint-environment")
    descriptor = root / "environment.yml"
    descriptor.write_text(
        "channels:\n"
        "  - conda-forge\n"
        "dependencies:\n"
        "  - python=3.12\n"
        "  - pyyaml\n",
        encoding="utf-8",
    )
    environment = root / "source"
    create_environment_from_descriptor(descriptor, environment)
    return environment


def test_environment_archive_restores_checkpoint_dependency(
    tmp_path: Path,
    source_checkpoint_environment: Path,
):
    iteration = tmp_path / "iteration"
    archive = get_iteration_environment_archive_path(iteration)
    restored_environment = tmp_path / "restored"

    export_environment_archive(source_checkpoint_environment, archive)
    restored = restore_iteration_environment(iteration, restored_environment)

    assert restored == restored_environment
    subprocess.run(
        [str(restored / "bin" / "python"), "-c", "import yaml"],
        check=True,
    )


def test_fork_recreates_selected_checkpoint_environment_without_changing_source(
    tmp_path: Path,
    config_factory,
    source_checkpoint_environment: Path,
):
    source_config = config_factory(agent_id="source")
    initialize_run(
        source_config,
        {
            "task_type": "classification",
            "input_structure": ["data.csv"],
            "label_to_scalar": {"negative": 0, "positive": 1},
        },
    )
    descriptor = source_config.shared_dir / Config.ENVIRONMENT_DESCRIPTOR_FILENAME
    export_environment_descriptor_to_path(source_checkpoint_environment, descriptor)
    checkpoint_descriptor = descriptor.read_bytes()
    commit_step_checkpoint(source_config, iteration=0, step_id="data_split")
    descriptor.write_text(
        "dependencies: [package-that-does-not-exist-for-agentomics-test]\n",
        encoding="utf-8",
    )
    commit_step_checkpoint(source_config, iteration=0, step_id="model_training")
    source_workspace = Path(source_config.workspace_dir)
    source_files = {
        path.relative_to(source_workspace): path.read_bytes()
        for path in source_workspace.rglob("*")
        if path.is_file() and ".git" not in path.relative_to(source_workspace).parts
    }
    source_head = run_git_cli_command(source_workspace, "rev-parse", "HEAD").stdout
    target_workspace = tmp_path / "fork"

    fork_run(
        source_workspace,
        target_agent_id="fork",
        target_workspace_dir=target_workspace,
        fork_from_step="data_split",
        fork_from_iteration=0,
    )

    target_config = load_config_from_run_dir(target_workspace / Config.RUN_DIRNAME)
    target_descriptor = (
        target_config.shared_dir / Config.ENVIRONMENT_DESCRIPTOR_FILENAME
    )
    assert target_descriptor.read_bytes() == checkpoint_descriptor
    restored_descriptor = tmp_path / "restored-environment.yml"
    export_environment_descriptor_to_path(
        target_config.shared_environment_path, restored_descriptor,
    )
    assert restored_descriptor.read_bytes() == checkpoint_descriptor

    initialized_environment = initialize_shared_environment(target_config)

    assert initialized_environment == target_config.shared_environment_path
    assert (initialized_environment / "conda-meta" / "history").owner() == Config.AGENT_USER
    export_environment_descriptor_to_path(
        initialized_environment, restored_descriptor,
    )
    assert restored_descriptor.read_bytes() == checkpoint_descriptor
    subprocess.run(
        [
            str(target_config.shared_environment_path / "bin" / "python"),
            "-c",
            "import yaml",
        ],
        check=True,
    )
    assert {
        path.relative_to(source_workspace): path.read_bytes()
        for path in source_workspace.rglob("*")
        if path.is_file() and ".git" not in path.relative_to(source_workspace).parts
    } == source_files
    assert run_git_cli_command(source_workspace, "rev-parse", "HEAD").stdout == source_head
