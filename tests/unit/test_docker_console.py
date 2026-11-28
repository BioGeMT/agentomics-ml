import subprocess
import sys

import pytest

from agentomics.cli import docker_utils

pytestmark = pytest.mark.skipif(sys.platform != "win32", reason="console windows are a Windows matter")


@pytest.fixture
def run_calls(monkeypatch: pytest.MonkeyPatch) -> list[dict]:
    calls: list[dict] = []
    monkeypatch.setattr(docker_utils.shutil, "which", lambda _: "docker")
    monkeypatch.setattr(docker_utils.subprocess, "run", lambda command, **kwargs: calls.append(kwargs) or subprocess.CompletedProcess(command, 0))
    return calls


def _console(monkeypatch: pytest.MonkeyPatch, window: int) -> None:
    monkeypatch.setattr(docker_utils.ctypes.windll.kernel32, "GetConsoleWindow", lambda: window)


def test_docker_gets_no_window_and_prints_to_our_output_without_a_console(monkeypatch: pytest.MonkeyPatch, run_calls: list[dict]):
    _console(monkeypatch, 0)

    docker_utils.run_python_in_docker("image", cpu_only=True, mounts=[], python_arguments=[])

    assert run_calls[0]["creationflags"] == subprocess.CREATE_NO_WINDOW
    assert run_calls[0]["stdout"] is sys.stdout and run_calls[0]["stderr"] is sys.stderr


def test_docker_shares_the_terminal_it_was_run_from(monkeypatch: pytest.MonkeyPatch, run_calls: list[dict]):
    _console(monkeypatch, 1)

    docker_utils.run_python_in_docker("image", cpu_only=True, mounts=[], python_arguments=[])

    assert "creationflags" not in run_calls[0]
