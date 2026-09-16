from __future__ import annotations

import os
import shutil
import tempfile
from collections.abc import Iterator
from pathlib import Path

import pytest

from tests.dataset_helpers import create_classification_dataset
from tests.support.cli import run_cli, scripted_run_arguments


@pytest.fixture(scope="session")
def cli():
    """Invoke host CLIs with shared image, resource mode, and isolated credentials."""
    image = os.environ.get("AGENTOMICS_TEST_IMAGE")
    cpu_only_flag = os.environ.get("AGENTOMICS_TEST_CPU_ONLY")
    if not image or cpu_only_flag not in ("0", "1"):
        pytest.fail(
            "Select an image and CPU/GPU mode with scripts/run_tests.py, or set "
            "AGENTOMICS_TEST_IMAGE and AGENTOMICS_TEST_CPU_ONLY (0 or 1)."
        )
    with tempfile.TemporaryDirectory(prefix="agentomics-e2e-cli-") as directory:
        root = Path(directory).resolve()
        home = root / "home"
        home.mkdir()
        # Allowlist transport essentials; do not inherit provider credentials,
        # proxies, Python startup hooks, or optional remote logging settings.
        environment = {
            key: value for key, value in os.environ.items()
            if key in {
                "PATH", "DOCKER_HOST", "DOCKER_CONTEXT", "DOCKER_CONFIG",
                "DOCKER_TLS_VERIFY", "DOCKER_CERT_PATH", "XDG_RUNTIME_DIR",
            }
        }
        docker_config = Path.home() / ".docker"
        if docker_config.is_dir():
            environment.setdefault("DOCKER_CONFIG", str(docker_config))
        environment.update({
            "HOME": str(home),
            "PYTHONUNBUFFERED": "1",
        })

        def invoke(
            arguments: list[str], root: Path, *, module: str = "agentomics.cli.run",
            expect_failure: bool = False,
        ) -> str:
            arguments = ["--image", image, *arguments]
            if cpu_only_flag == "1":
                arguments.append("--cpu-only")
            return run_cli(
                arguments,
                root,
                environment,
                image=image,
                module=module,
                expect_failure=expect_failure,
            )

        yield invoke


@pytest.fixture(scope="session")
def _completed_run(request, cli) -> Iterator[Path]:
    """Run the selected scripted workflow once per session."""
    model = getattr(request, "param", "scripted-default")
    with tempfile.TemporaryDirectory(prefix="agentomics-e2e-") as directory:
        root = Path(directory).resolve()
        dataset = create_classification_dataset(
            root, include_validation_split=True, include_test_split=True,
        )
        workspace = root / "workspace"
        arguments = scripted_run_arguments(dataset, workspace, model)
        cli(arguments, root)
        yield workspace


@pytest.fixture
def completed_run(_completed_run: Path) -> Iterator[Path]:
    """Give each test a private workspace copy so it cannot modify the cached run."""
    with tempfile.TemporaryDirectory(prefix="agentomics-e2e-copy-") as directory:
        workspace = Path(directory) / "workspace"
        shutil.copytree(_completed_run, workspace, symlinks=True)
        yield workspace
