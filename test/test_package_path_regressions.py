import json
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from agentomics.generate_final_reports import load_prepared_dataset_meta
from agentomics.utils.api_keys import get_repo_env_path


class PackagePathRegressionsTest(unittest.TestCase):
    def test_report_metadata_prefers_explicit_prepared_datasets_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            agent_dir = tmp_path / "agent"
            explicit_prepared_dir = tmp_path / "prepared-explicit"
            configured_prepared_dir = tmp_path / "prepared-config"

            (agent_dir / "extras").mkdir(parents=True)
            (explicit_prepared_dir / "demo").mkdir(parents=True)
            (configured_prepared_dir / "demo").mkdir(parents=True)

            (agent_dir / "extras" / "config.json").write_text(
                json.dumps(
                    {
                        "dataset": "demo",
                        "prepared_datasets_dir": str(configured_prepared_dir),
                    }
                ),
                encoding="utf-8",
            )
            (explicit_prepared_dir / "demo" / "metadata.json").write_text(
                json.dumps(
                    {
                        "task_type": "classification",
                        "numeric_label_col": "label",
                        "label_to_scalar": {"neg": 0, "pos": 1},
                    }
                ),
                encoding="utf-8",
            )
            (configured_prepared_dir / "demo" / "metadata.json").write_text(
                json.dumps(
                    {
                        "task_type": "regression",
                        "numeric_label_col": "score",
                    }
                ),
                encoding="utf-8",
            )

            dataset_meta = load_prepared_dataset_meta(agent_dir, explicit_prepared_dir)

            self.assertEqual(dataset_meta.task_type, "classification")
            self.assertEqual(dataset_meta.numeric_label_col, "label")
            self.assertEqual(dataset_meta.label_to_scalar, {"neg": 0, "pos": 1})

    def test_report_metadata_falls_back_to_configured_prepared_datasets_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            agent_dir = tmp_path / "agent"
            explicit_prepared_dir = tmp_path / "missing-explicit"
            configured_prepared_dir = tmp_path / "prepared-config"

            (agent_dir / "extras").mkdir(parents=True)
            (configured_prepared_dir / "demo").mkdir(parents=True)

            (agent_dir / "extras" / "config.json").write_text(
                json.dumps(
                    {
                        "dataset": "demo",
                        "prepared_datasets_dir": str(configured_prepared_dir),
                    }
                ),
                encoding="utf-8",
            )
            (configured_prepared_dir / "demo" / "metadata.json").write_text(
                json.dumps(
                    {
                        "task_type": "classification",
                        "numeric_label_col": "label",
                        "label_to_scalar": {"neg": 0, "pos": 1},
                    }
                ),
                encoding="utf-8",
            )

            dataset_meta = load_prepared_dataset_meta(agent_dir, explicit_prepared_dir)

            self.assertEqual(dataset_meta.task_type, "classification")
            self.assertEqual(dataset_meta.numeric_label_col, "label")

    def test_api_keys_repo_env_path_points_to_repo_root(self):
        self.assertEqual(get_repo_env_path(), REPO_ROOT / ".env")


if __name__ == "__main__":
    unittest.main()
