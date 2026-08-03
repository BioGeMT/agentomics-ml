import json
import subprocess
import sys
import tomllib
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "release-please-config.json"
MANIFEST_PATH = REPO_ROOT / ".release-please-manifest.json"
VALIDATOR = REPO_ROOT / "scripts" / "validate_pr_title.py"


class ReleasePleaseConfigurationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = json.loads(CONFIG_PATH.read_text())
        cls.manifest = json.loads(MANIFEST_PATH.read_text())
        with (REPO_ROOT / "pyproject.toml").open("rb") as pyproject_file:
            cls.pyproject = tomllib.load(pyproject_file)

    def test_manifest_starts_at_the_standardized_package_version(self):
        self.assertEqual(
            self.pyproject["project"]["version"],
            self.manifest["."],
        )

    def test_generated_proposal_title_passes_the_title_policy(self):
        patterns = (
            self.config["group-pull-request-title-pattern"],
            self.config["packages"]["."]["pull-request-title-pattern"],
        )

        for pattern in patterns:
            with self.subTest(pattern=pattern):
                title = pattern.replace("${version}", "1.2.3")
                result = subprocess.run(
                    [sys.executable, str(VALIDATOR), title],
                    capture_output=True,
                    cwd=REPO_ROOT,
                    text=True,
                )
                self.assertEqual(0, result.returncode, result.stderr)
                self.assertIn("Release Impact: none", result.stdout)


if __name__ == "__main__":
    unittest.main()
