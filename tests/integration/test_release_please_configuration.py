import json
import tomllib

import pytest

from tests.helpers import REPO_ROOT, run_pr_title_validator

CONFIG_PATH = REPO_ROOT / "release-please-config.json"
MANIFEST_PATH = REPO_ROOT / ".release-please-manifest.json"
PYPROJECT_PATH = REPO_ROOT / "pyproject.toml"


@pytest.fixture
def release_config() -> dict:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def test_manifest_version_matches_package_version():
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    with PYPROJECT_PATH.open("rb") as pyproject_file:
        pyproject = tomllib.load(pyproject_file)

    assert manifest["."] == pyproject["project"]["version"]


@pytest.mark.parametrize(
    "pattern_path",
    [
        pytest.param(
            ("group-pull-request-title-pattern",),
            id="group-proposal",
        ),
        pytest.param(
            ("packages", ".", "pull-request-title-pattern"),
            id="package-proposal",
        ),
    ],
)
def test_release_proposal_title_passes_title_policy(
    release_config: dict,
    pattern_path: tuple[str, ...],
):
    pattern = release_config
    for key in pattern_path:
        pattern = pattern[key]
    title = pattern.replace("${version}", "1.2.3")

    result = run_pr_title_validator(title)

    assert result.returncode == 0, result.stderr
