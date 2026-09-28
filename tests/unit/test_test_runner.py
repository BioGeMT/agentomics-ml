import subprocess

import pytest

from scripts.run_tests import main


@pytest.mark.parametrize("mode", [[], ["--cpu-only"]])
def test_gpu_is_required_unless_cpu_only_is_selected(mode, monkeypatch):
    def run(command, **kwargs):
        return subprocess.CompletedProcess(
            command,
            125 if "--gpus" in command else 0,
            stdout="",
            stderr="could not select device driver with capabilities: [[gpu]]",
        )

    monkeypatch.setattr("shutil.which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(subprocess, "run", run)

    if mode:
        assert main(["--image", "already-built", *mode]) == 0
    else:
        with pytest.raises(RuntimeError, match="--cpu-only"):
            main(["--image", "already-built", *mode])


@pytest.mark.parametrize("docker_installed", [False, True])
def test_unavailable_docker_fails_with_a_diagnostic(docker_installed, monkeypatch):
    def run(command, **kwargs):
        if not docker_installed:
            raise FileNotFoundError("No such file or directory: docker")
        return subprocess.CompletedProcess(
            command, 125, stdout="", stderr="Cannot connect to the Docker daemon"
        )

    monkeypatch.setattr(
        "shutil.which", lambda name: f"/usr/bin/{name}" if docker_installed else None
    )
    monkeypatch.setattr(subprocess, "run", run)

    error_type = RuntimeError if docker_installed else FileNotFoundError
    with pytest.raises(error_type, match="(?i)docker"):
        main(["--image", "already-built", "--cpu-only"])


@pytest.mark.parametrize(
    "container_exit,host_exit,expected_exit",
    [(0, 0, 0), (1, 0, 1), (0, 1, 1), (2, 0, 1)],
)
def test_both_phases_run_and_either_test_failure_fails_the_suite(
    container_exit, host_exit, expected_exit, monkeypatch
):
    completed_phases = []

    def run(command, **kwargs):
        exit_code = 0
        if "pytest" in command:
            phase = "container" if command[0] == "docker" else "host"
            completed_phases.append(phase)
            exit_code = container_exit if phase == "container" else host_exit
        return subprocess.CompletedProcess(command, exit_code, stdout="", stderr="")

    monkeypatch.setattr("shutil.which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(subprocess, "run", run)

    exit_code = main(["--image", "already-built", "--cpu-only"])

    assert completed_phases == ["container", "host"]
    assert exit_code == expected_exit


def test_interrupted_tests_do_not_leave_a_container_running(monkeypatch):
    running_containers = set()

    def run(command, **kwargs):
        if "pytest" in command:
            name = command[command.index("--name") + 1] if "--name" in command else "anonymous"
            running_containers.add(name)
            raise KeyboardInterrupt
        if command[1:3] == ["rm", "--force"]:
            running_containers.discard(command[-1])
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr("shutil.which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(subprocess, "run", run)

    with pytest.raises(KeyboardInterrupt):
        main(["--image", "already-built", "--cpu-only"])

    assert not running_containers
