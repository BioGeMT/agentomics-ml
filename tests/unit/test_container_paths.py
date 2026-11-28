import argparse
import importlib
import pathlib

import pytest

from agentomics.cli import inference, run


@pytest.fixture
def run_on_windows(monkeypatch: pytest.MonkeyPatch):
    """The CLI module as imported on Windows, where pathlib.Path follows Windows path rules."""
    monkeypatch.setattr(pathlib, "Path", pathlib.PureWindowsPath)
    yield importlib.reload(run)
    monkeypatch.undo()
    importlib.reload(run)


def test_container_paths_keep_forward_slashes_on_windows(run_on_windows):
    assert str(run_on_windows.CONTAINER_WORKSPACE_DIRECTORY) == "/workspace"
    assert str(run_on_windows.CONTAINER_FORK_SOURCE_DIRECTORY) == "/fork-source"
    assert str(run_on_windows.CONTAINER_CODEX_DIRECTORY) == "/mnt/codex-host"
    # A dataset entry's mount point, as build_dataset_mounts makes it from a Windows dataset folder.
    dataset = pathlib.PureWindowsPath(r"C:\Users\me\datasets\toy")
    assert str(run_on_windows.CONTAINER_DATASETS_DIRECTORY / dataset.name / "train.csv") == "/datasets/toy/train.csv"


def test_inference_passes_the_iteration_folder_with_forward_slashes(monkeypatch: pytest.MonkeyPatch):
    calls: list[dict] = []
    monkeypatch.setattr(inference, "run_python_in_docker", lambda **kwargs: calls.append(kwargs) or 0)
    arguments = argparse.Namespace(
        image="image",
        agent_dir=pathlib.Path("agent"),
        input_path=pathlib.Path("input.csv"),
        output=pathlib.Path("out/predictions.csv"),
        iteration_dir=pathlib.PureWindowsPath(r"run\iteration_1"),
        artifacts_dir=None,
        label_col=None,
        all_iterations=False,
        cpu_only=True,
        wandb_prefix=None,
    )

    inference._run_inference_with_resolved_paths(arguments)

    python_arguments = calls[0]["python_arguments"]
    assert python_arguments[python_arguments.index("--iteration-dir") + 1] == "run/iteration_1"


def test_codex_credentials_accept_the_clis_container_path(tmp_path):
    # The container's workflows pass CONTAINER_CODEX_DIRECTORY, a PurePosixPath, straight to this.
    # codex_auth needs pydantic-ai, which the host-only Windows job doesn't install.
    pytest.importorskip("pydantic_ai")
    from agentomics.utils.providers.codex_auth import prepare_codex_credentials

    prepare_codex_credentials(pathlib.PurePosixPath(tmp_path / "no-codex-login"))
