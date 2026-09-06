from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import tempfile
from pathlib import Path
from uuid import uuid4


def remove_owned_containers(root: Path, environment: dict[str, str]) -> None:
    """Remove only containers with a bind mount inside this fixture's root."""
    result = subprocess.run(
        ["docker", "ps", "--all", "--quiet"], env=environment,
        capture_output=True, text=True, check=True, timeout=30,
    )
    for container_id in result.stdout.split():
        result = subprocess.run(
            ["docker", "inspect", container_id], env=environment,
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode:
            # A --rm container may have exited between ps and inspect. Do not
            # mistake a daemon failure for successful cleanup.
            if "No such object" in result.stderr:
                continue
            raise RuntimeError(f"Cannot inspect container for cleanup: {result.stderr}")
        mounts = json.loads(result.stdout)[0]["Mounts"]
        if any(
            mount["Type"] == "bind" and Path(mount["Source"]).is_relative_to(root)
            for mount in mounts
        ):
            subprocess.run(
                ["docker", "rm", "--force", container_id], env=environment,
                capture_output=True, text=True, check=True, timeout=30,
            )


def restore_workspace_ownership(root: Path, image: str, environment: dict[str, str]) -> None:
    """Restore host-user ownership so temporary files left owned by container users after a forced kill can be deleted."""
    container_name = f"agentomics-cleanup-{uuid4().hex}"
    try:
        subprocess.run(
            [
                "docker", "run", "--rm", "--pull", "never", "--name", container_name,
                "--user", "0", "--mount", f"type=bind,src={root},dst=/cleanup",
                "--entrypoint", "chown", image, "-R", "--no-dereference",
                f"{os.getuid()}:{os.getgid()}", "/cleanup",
            ],
            env=environment, capture_output=True, text=True, check=True, timeout=30,
        )
    finally:
        result = subprocess.run(
            ["docker", "rm", "--force", container_name], env=environment,
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode and "No such container" not in result.stderr:
            raise RuntimeError(f"Cannot remove cleanup container: {result.stderr}")


def scripted_run_arguments(dataset: Path, workspace: Path, model: str) -> list[str]:
    """Build the common real-CLI arguments for a scripted workflow."""
    return [
        "--dataset", dataset.name,
        "--datasets-dir", str(dataset.parent),
        "--workspace-dir", str(workspace),
        "--provider", "scripted",
        "--model", model,
        "--iteration-plan-model", "scripted-plan",
        "--iterations", "1",
        "--val-metric", "ACC",
        "--run-python-timeout", "60",
        "--disable-training-reporting",
    ]


def run_cli(
    arguments: list[str], root: Path, environment: dict[str, str], *,
    image: str, module: str = "agentomics.cli.run", timeout: float = 600,
    expect_failure: bool = False,
) -> str:
    """Bound the real CLI and clean its containers, including after interruption."""
    with tempfile.TemporaryFile(mode="w+") as output:
        process = subprocess.Popen(
            [sys.executable, "-m", module, *arguments],
            cwd=root, env=environment, stdin=subprocess.DEVNULL,
            stdout=output, stderr=subprocess.STDOUT, start_new_session=True,
        )
        return_code = None
        try:
            return_code = process.wait(timeout=timeout)
            output.seek(0)
            cli_output = output.read()
            if expect_failure:
                assert return_code != 0, (
                    f"Host CLI unexpectedly succeeded:\n{cli_output}"
                )
            else:
                assert return_code == 0, (
                    f"Host CLI exited {return_code}:\n{cli_output}"
                )
            return cli_output
        finally:
            # Kill the process group, not only Python: a Docker client or other
            # child must not start another container while cleanup is running.
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait(timeout=30)
            output.seek(0)
            print(output.read())
            remove_owned_containers(root, environment)
            if return_code is None:
                # Forced termination skips the application's ownership cleanup.
                # Normal exits must rely on that cleanup, not this fallback.
                restore_workspace_ownership(root, image, environment)


