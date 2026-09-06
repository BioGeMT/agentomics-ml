from __future__ import annotations

import shutil
import subprocess
import tempfile
from collections.abc import Callable, Iterator
from pathlib import Path

import pydantic_ai.models
import pytest

from agentomics.datasets.dataset_preparation import prepare_dataset
from agentomics.run_agent import initialize_run
from agentomics.runtime.conda_utils import (
    create_environment_from_descriptor,
    initialize_shared_environment,
)
from agentomics.runtime.read_write_utils import get_next_iteration_index
from agentomics.runtime.run_lifecycle import prepare_iteration_workspace
from agentomics.tools.tool_registry import create_tool, create_tools
from agentomics.utils.config import Config
from tests.dataset_helpers import create_classification_dataset


@pytest.fixture(autouse=True)
def disable_model_requests(monkeypatch: pytest.MonkeyPatch) -> None:
    """Prevent real model requests in the container test process."""
    monkeypatch.setattr(pydantic_ai.models, "ALLOW_MODEL_REQUESTS", False)


@pytest.fixture
def default_prepared_dataset_dir(
    default_dataset_with_validation: Path,
    tmp_path: Path,
) -> Path:
    """Prepare the standard dataset through the production workflow."""
    prepared_dir = tmp_path / "prepared" / default_dataset_with_validation.name
    prepare_dataset(
        source_dir=default_dataset_with_validation,
        destination_dir=prepared_dir,
    )
    return prepared_dir


def _give_agent_read_access(path: Path) -> None:
    """Let the restricted agent read and traverse a tree without writing to it."""
    subprocess.run(
        ["chmod", "-R", "u=rwX,go=rX", str(path)],
        check=True,
    )


def _give_agent_write_access(path: Path) -> None:
    """Transfer a writable directory to the restricted agent."""
    subprocess.run(["chown", Config.AGENT_USER, str(path)], check=True)


@pytest.fixture
def config_factory(tmp_path: Path) -> Callable[..., Config]:
    """Create isolated Config instances with focused per-test overrides."""

    def create_config(**overrides) -> Config:
        defaults = {
            "agent_id": "test-agent",
            "model_name": "test-model",
            "iteration_plan_model_name": "test-model",
            "dataset": "toy",
            "tags": [],
            "val_metric": "ACC",
            "task_type": "classification",
            "input_structure": ["data.csv"],
            "workspace_dir": str(tmp_path / "workspace"),
            "datasets_dir": str(tmp_path / "datasets"),
            "environments_dir": str(tmp_path / "environments"),
        }
        return Config(**(defaults | overrides))

    return create_config


@pytest.fixture
def default_agent_config(config_factory: Callable[..., Config]) -> Iterator[Config]:
    """Create standard configuration without initializing persistent run state."""
    with tempfile.TemporaryDirectory(prefix="agentomics-test-") as directory:
        test_root = Path(directory)
        dataset_dir = create_classification_dataset(test_root)
        yield config_factory(
            agent_id=test_root.name,
            dataset=dataset_dir.name,
            workspace_dir=str(test_root / "workspace"),
            datasets_dir=str(dataset_dir.parent),
            environments_dir=str(test_root / "environments"),
        )


@pytest.fixture
def initialized_run_config(default_agent_config: Config) -> Config:
    """Initialize the standard run through the production run lifecycle."""
    initialize_run(
        default_agent_config,
        {
            "task_type": "classification",
            "input_structure": ["data.csv"],
            "label_to_scalar": {"negative": 0, "positive": 1},
        },
    )
    return default_agent_config


@pytest.fixture
def prepared_iteration(initialized_run_config: Config) -> None:
    """Prepare state required before invoking iteration-scoped step behavior."""
    prepare_iteration_workspace(
        initialized_run_config,
        iteration=get_next_iteration_index(initialized_run_config),
    )


@pytest.fixture(scope="session")
def lightweight_conda_environment(
    tmp_path_factory: pytest.TempPathFactory,
) -> Path:
    """Create one small real Python environment for snapshot lifecycle tests."""
    root = tmp_path_factory.mktemp("snapshot-environment")
    descriptor = root / "environment.yml"
    descriptor.write_text(
        "channels:\n"
        "  - conda-forge\n"
        "dependencies:\n"
        "  - python=3.12\n"
        "  - pyyaml\n",
        encoding="utf-8",
    )
    environment_path = root / "environment"
    create_environment_from_descriptor(descriptor, environment_path)
    return environment_path


@pytest.fixture
def initialized_run_with_lightweight_environment(
    initialized_run_config: Config,
    lightweight_conda_environment: Path,
) -> Config:
    """Provide initialized run state backed by a reusable real environment."""
    environment_path = initialized_run_config.shared_environment_path
    environment_path.parent.mkdir(parents=True, exist_ok=True)
    environment_path.symlink_to(lightweight_conda_environment, target_is_directory=True)
    return initialized_run_config


@pytest.fixture
def restricted_agent_workspace(
    initialized_run_config: Config,
    prepared_iteration,
) -> Path:
    """Prepare a workspace accessible to the restricted agent OS user."""
    _give_agent_read_access(Path(initialized_run_config.workspace_dir).parent)
    initialized_run_config.current_step_dir.mkdir(exist_ok=True)
    _give_agent_write_access(initialized_run_config.current_step_dir)
    return initialized_run_config.current_step_dir


@pytest.fixture
def replace_tool(default_agent_config: Config):
    """Create the Replace tool without process-environment initialization."""
    return create_tool(default_agent_config, "replace")


@pytest.fixture
def default_agent_environment(initialized_run_config: Config) -> Iterator[Path]:
    """Initialize and clean up an isolated agent Conda environment."""
    environment_path = initialized_run_config.shared_environment_path
    try:
        initialize_shared_environment(initialized_run_config)
        yield environment_path
    finally:
        if environment_path.exists():
            shutil.rmtree(environment_path)


@pytest.fixture
def default_agent_tools(
    initialized_run_config: Config,
    default_agent_environment: Path,
    restricted_agent_workspace: Path,
):
    """Create process tools after initializing the Conda environment they use."""
    return {
        tool.name: tool
        for tool in create_tools(initialized_run_config, ["bash", "run_python"])
    }


@pytest.fixture
def bash_tool(default_agent_tools):
    """Return the Bash tool configured for the isolated agent workspace."""
    return default_agent_tools["bash"]


@pytest.fixture
def run_python_tool(default_agent_tools):
    """Return the Python tool configured for the isolated agent workspace."""
    return default_agent_tools["run_python"]


@pytest.fixture
def default_dataset_dir(tmp_path: Path) -> Path:
    return create_classification_dataset(tmp_path)


@pytest.fixture
def default_dataset_with_validation(tmp_path: Path) -> Path:
    return create_classification_dataset(tmp_path, include_validation_split=True)


@pytest.fixture
def default_dataset_with_validation_and_test(tmp_path: Path) -> Path:
    return create_classification_dataset(
        tmp_path, include_validation_split=True, include_test_split=True,
    )
