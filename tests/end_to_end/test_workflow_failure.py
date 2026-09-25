import tempfile
from pathlib import Path

from tests.dataset_helpers import create_classification_dataset
from tests.support.cli import scripted_run_arguments


def test_workflow_without_a_valid_best_result_fails_the_host_cli(cli):
    with tempfile.TemporaryDirectory(prefix="agentomics-e2e-failure-") as directory:
        root = Path(directory).resolve()
        dataset = create_classification_dataset(
            root, include_validation_split=True, include_test_split=True,
        )
        workspace = root / "workspace"
        output = cli(
            scripted_run_arguments(
                dataset,
                workspace,
                "scripted-workflow-failure",
            ),
            root,
            expect_failure=True,
        )

        assert "Intentional scripted validation failure" in output
        assert "Agent did not produce a valid best iteration snapshot" in output
        assert "Run finished. Files can be found" not in output

    assert not root.exists()
