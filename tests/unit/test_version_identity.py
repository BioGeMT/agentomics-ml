from importlib.metadata import version as distribution_version

import pytest

import agentomics
from agentomics.cli.docker_utils import DEFAULT_IMAGE
from agentomics.utils.versioning import check_run_compatible


def test_runtime_version_matches_distribution_metadata():
    installed_version = distribution_version("agentomics")

    assert installed_version == agentomics.__version__


def test_default_worker_image_matches_distribution_version():
    installed_version = distribution_version("agentomics")

    assert f"biogemt/agentomics:{installed_version}" == DEFAULT_IMAGE


def test_recorded_runs_use_major_version_compatibility():
    current_version = distribution_version("agentomics")
    current_major = int(current_version.split(".", 1)[0])
    incompatible_version = f"{current_major + 1}.0.0"

    assert check_run_compatible(f"{current_major}.99.0") is None
    with pytest.raises(
        RuntimeError,
        match=rf"Agentomics {incompatible_version}.*running version {current_version}",
    ) as raised:
        check_run_compatible(incompatible_version)

    assert f"pip install 'agentomics=={current_major + 1}.*'" in str(raised.value)
