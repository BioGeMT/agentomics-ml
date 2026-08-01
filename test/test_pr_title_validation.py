import subprocess
import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
VALIDATOR = REPO_ROOT / "scripts" / "validate_pr_title.py"


class PullRequestTitleValidationTest(unittest.TestCase):
    def validate(self, title: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, str(VALIDATOR), title],
            capture_output=True,
            cwd=REPO_ROOT,
            text=True,
        )

    def test_fix_declares_patch_release_impact(self):
        result = self.validate("fix: preserve uploaded filenames")

        self.assertEqual(0, result.returncode, result.stderr)
        self.assertIn("Release Impact: patch", result.stdout)

    def test_releasing_types_declare_the_expected_impact(self):
        examples = {
            "perf: reduce memory use": "patch",
            "feat: add CSV exports": "minor",
            "feat!: remove the legacy workspace format": "major",
            "fix!: reject the removed argument": "major",
        }

        for title, expected_impact in examples.items():
            with self.subTest(title=title):
                result = self.validate(title)
                self.assertEqual(0, result.returncode, result.stderr)
                self.assertIn(
                    f"Release Impact: {expected_impact}",
                    result.stdout,
                )

    def test_documented_non_releasing_types_declare_no_release_impact(self):
        titles = (
            "docs: explain workspace layout",
            "test: cover image selection",
            "ci: cache the container build",
            "chore: update maintainers",
            "refactor: share argument parsing",
        )

        for title in titles:
            with self.subTest(title=title):
                result = self.validate(title)
                self.assertEqual(0, result.returncode, result.stderr)
                self.assertIn("Release Impact: none", result.stdout)

    def test_release_proposal_title_is_non_releasing(self):
        result = self.validate("chore: release 1.2.3")

        self.assertEqual(0, result.returncode, result.stderr)
        self.assertIn("Release Impact: none", result.stdout)

    def test_invalid_or_ambiguous_titles_fail_with_correction_guidance(self):
        invalid_titles = (
            "feature: add CSV exports",
            "docs!: remove obsolete instructions",
            "feat(cli): add CSV exports",
            "feat : add CSV exports",
            "feat:",
            "Add CSV exports",
        )

        for title in invalid_titles:
            with self.subTest(title=title):
                result = self.validate(title)
                self.assertNotEqual(0, result.returncode)
                self.assertIn("Invalid PR title", result.stderr)
                self.assertIn("<type>[!]: <description>", result.stderr)
                self.assertIn("CONTRIBUTING.md", result.stderr)
                self.assertNotIn("Releasing types:", result.stderr)


if __name__ == "__main__":
    unittest.main()
