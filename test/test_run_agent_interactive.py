import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

import agentomics.run_agent_interactive as run_agent_interactive


class RunAgentInteractiveListModesTest(unittest.TestCase):
    def test_list_metrics_skips_provider_and_agent_setup(self):
        with (
            patch.object(sys, "argv", ["agentomics", "--list-metrics"]),
            patch.object(run_agent_interactive, "resolve_agentomics_paths", return_value=SimpleNamespace()),
            patch.object(
                run_agent_interactive,
                "get_provider_and_api_key",
                side_effect=AssertionError("provider resolution should not run for --list-metrics"),
            ),
            patch.object(
                run_agent_interactive,
                "create_user",
                side_effect=AssertionError("AGENT_ID generation should not run for --list-metrics"),
            ),
            patch.object(run_agent_interactive, "display_metrics_table") as display_metrics_table,
        ):
            exit_code = run_agent_interactive.main()

        self.assertEqual(exit_code, 0)
        display_metrics_table.assert_called_once_with()

    def test_list_datasets_skips_provider_and_agent_setup(self):
        paths = SimpleNamespace(
            prepared_datasets_dir=Path("/tmp/prepared_datasets"),
            prepared_test_sets_dir=Path("/tmp/prepared_test_sets"),
        )
        datasets = [{"name": "demo"}]

        with (
            patch.object(sys, "argv", ["agentomics", "--list-datasets"]),
            patch.object(run_agent_interactive, "resolve_agentomics_paths", return_value=paths),
            patch.object(
                run_agent_interactive,
                "get_provider_and_api_key",
                side_effect=AssertionError("provider resolution should not run for --list-datasets"),
            ),
            patch.object(
                run_agent_interactive,
                "create_user",
                side_effect=AssertionError("AGENT_ID generation should not run for --list-datasets"),
            ),
            patch.object(run_agent_interactive, "get_all_prepared_datasets_info", return_value=datasets) as get_datasets,
            patch.object(run_agent_interactive, "print_datasets_table") as print_datasets_table,
        ):
            exit_code = run_agent_interactive.main()

        self.assertEqual(exit_code, 0)
        get_datasets.assert_called_once_with(paths.prepared_datasets_dir, paths.prepared_test_sets_dir)
        print_datasets_table.assert_called_once_with(datasets)


if __name__ == "__main__":
    unittest.main()
