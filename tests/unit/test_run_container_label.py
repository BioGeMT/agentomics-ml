from pathlib import Path

import pytest

from agentomics.cli import run


@pytest.fixture
def docker_calls(monkeypatch: pytest.MonkeyPatch) -> list[dict]:
    calls: list[dict] = []
    monkeypatch.setattr(run, "run_python_in_docker", lambda **kwargs: calls.append(kwargs) or 0)
    return calls


def _label_of(call: dict) -> str:
    docker_arguments = call["docker_arguments"]
    return docker_arguments[docker_arguments.index("--label") + 1]


def test_agent_container_is_labelled_with_its_workspace_folder(tmp_path: Path, docker_calls: list[dict]):
    workspace = tmp_path / "toy-20261103-091800"
    workspace.mkdir()
    arguments = run.build_parser().parse_args(["--dataset", "toy", "--model", "gpt", "--cpu-only"])

    run._run_agent_in_docker(arguments, workspace, "agent-1", dataset_mounts=[])

    assert _label_of(docker_calls[0]) == "org.biogemt.agentomics.workspace=toy-20261103-091800"


def test_reporting_container_is_labelled_with_its_workspace_folder(tmp_path: Path, docker_calls: list[dict]):
    workspace = tmp_path / "toy-20261103-091800"
    workspace.mkdir()

    run._run_reporting_in_docker("image", workspace, "agent-1")

    assert _label_of(docker_calls[0]) == "org.biogemt.agentomics.workspace=toy-20261103-091800"
