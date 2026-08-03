import sys
import unittest
from importlib.metadata import version as distribution_version
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

import agentomics  # noqa: E402
from agentomics.cli.docker_utils import DEFAULT_IMAGE  # noqa: E402
from agentomics.utils.versioning import check_run_compatible, get_version  # noqa: E402


class VersionIdentityTest(unittest.TestCase):
    def test_runtime_version_matches_distribution_metadata(self):
        installed_version = distribution_version("agentomics")

        self.assertEqual(installed_version, agentomics.__version__)
        self.assertEqual(installed_version, get_version())

    def test_default_worker_image_matches_distribution_version(self):
        installed_version = distribution_version("agentomics")

        self.assertEqual(
            f"biogemt/agentomics:{installed_version}",
            DEFAULT_IMAGE,
        )

    def test_recorded_runs_use_major_version_compatibility(self):
        current_version = distribution_version("agentomics")
        current_major = int(current_version.split(".", 1)[0])
        incompatible_version = f"{current_major + 1}.0.0"

        self.assertIsNone(check_run_compatible(f"{current_major}.99.0"))
        with self.assertRaisesRegex(
            RuntimeError,
            rf"Agentomics {incompatible_version}.*running version {current_version}",
        ) as raised:
            check_run_compatible(incompatible_version)

        self.assertIn(
            f"pip install 'agentomics=={current_major + 1}.*'",
            str(raised.exception),
        )


if __name__ == "__main__":
    unittest.main()
