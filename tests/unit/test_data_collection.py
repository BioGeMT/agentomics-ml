from pathlib import Path

import pytest
from pydantic import ValidationError

from agentomics.agents.data_collection import DataRequest
from agentomics.cli import collect_data as collect_data_cli


@pytest.mark.parametrize(
    ("task_type", "classes"),
    [("classification", ["benign"]), ("regression", ["low", "high"])],
)
def test_request_rejects_classes_that_do_not_fit_the_task_type(task_type, classes):
    with pytest.raises(ValidationError):
        DataRequest(task_type=task_type, description="d", inputs=["i"], label="l", classes=classes)


def test_command_runs_the_worker_in_docker_with_the_request_and_collected_data_mounted(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(collect_data_cli, "run_python_in_docker", lambda **arguments: calls.append(arguments) or 3)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    request_path = tmp_path / "request.json"
    request_path.write_text("{}")
    collected_data_directory = tmp_path / "collected"

    exit_code = collect_data_cli.main([
        "--request", str(request_path), "--collected-data-dir", str(collected_data_directory),
        "--provider", "codex", "--model", "gpt-6.1-sol", "--image", "agentomics:test",
    ])

    assert exit_code == 3
    assert calls[0]["image"] == "agentomics:test"
    assert calls[0]["cpu_only"] is True
    assert f"type=bind,src={request_path},dst=/request.json,readonly" in calls[0]["mounts"]
    assert f"type=bind,src={collected_data_directory},dst=/collected-data" in calls[0]["mounts"]
    assert calls[0]["python_arguments"] == [
        "-m", "agentomics.agents.data_collection",
        "--request", "/request.json", "--collected-data-dir", "/collected-data",
        "--provider", "codex", "--model", "gpt-6.1-sol",
    ]
