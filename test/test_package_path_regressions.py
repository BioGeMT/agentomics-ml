import json
import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


REPO_ROOT = Path(__file__).resolve().parents[1]

from agentomics.utils.api_keys import get_repo_env_path
from agentomics.utils.foundation_models_utils import load_models_config


class PackagePathRegressionsTest(unittest.TestCase):
    def test_packaged_foundation_model_registry_loads_without_repo_override(self):
        with patch.dict("os.environ", {"FOUNDATION_MODELS_YAML": ""}, clear=False):
            models_config = load_models_config()

        self.assertIn("ESM-2", models_config)
        self.assertIn("NucleotideTransformer", models_config)
        self.assertTrue(Path(models_config["ESM-2"]["path_to_info"]).exists())

    @unittest.skipUnless(importlib.util.find_spec("matplotlib") is not None, "report dependencies not installed")
    def test_report_metadata_prefers_explicit_prepared_datasets_path(self):
        from agentomics.generate_final_reports import load_prepared_dataset_meta

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

    @unittest.skipUnless(importlib.util.find_spec("matplotlib") is not None, "report dependencies not installed")
    def test_report_metadata_falls_back_to_configured_prepared_datasets_path(self):
        from agentomics.generate_final_reports import load_prepared_dataset_meta

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

    def test_api_keys_repo_env_path_uses_detected_repo_root(self):
        with patch("agentomics.utils.api_keys.find_repo_root", return_value=REPO_ROOT):
            self.assertEqual(get_repo_env_path(), REPO_ROOT / ".env")


if __name__ == "__main__":
    unittest.main()
