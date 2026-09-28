import csv
import json
from pathlib import Path

from tests.dataset_helpers import create_classification_dataset
from tests.support.workspace import snapshot_workspace


def test_fork_executes_an_additional_iteration_without_changing_source(cli, completed_run: Path):
    root = completed_run.parent
    dataset = create_classification_dataset(
        root, include_validation_split=True, include_test_split=True,
    )
    workspace = root / "fork"
    assert not (completed_run / "run" / "iteration_1").exists()
    source = snapshot_workspace(completed_run)
    checkpoint_dependencies = (
        completed_run / "run" / "iteration_0" / "runtime_info" / "environment.yml"
    ).read_text(encoding="utf-8")

    cli(
        ["--fork-from-run", str(completed_run),
         "--fork-from-step", "end", "--fork-from-iteration", "0",
         "--datasets-dir", str(dataset.parent), "--workspace-dir", str(workspace),
         "--provider", "scripted",
         "--model", "scripted-default", "--iteration-plan-model", "scripted-plan",
         "--iterations", "1", "--run-python-timeout", "60",
         "--disable-training-reporting"],
        root,
    )

    assert snapshot_workspace(completed_run) == source, "Fork continuation changed the source run"
    # The inherited best result may still win a metric tie. Only the new
    # iteration can establish that the fork continued and produced usable output.
    continued = workspace / "run" / "iteration_1"
    metadata = json.loads((continued / "runtime_info" / "iteration_metadata.json").read_text())
    assert metadata["iteration"] == 1
    state = json.loads((continued / "runtime_info" / "iteration_state.json").read_text())
    assert state["status"] == "success"
    # This descriptor is exported from the continued iteration's live environment.
    assert (continued / "runtime_info" / "environment.yml").read_text(
        encoding="utf-8",
    ) == checkpoint_dependencies

    evaluation = continued / "validation_evaluation"
    with (evaluation / "eval_predictions_validation.csv").open() as stream:
        predictions = list(csv.DictReader(stream))
    assert [(row["id"], row["prediction"]) for row in predictions] == [
        ("validation-0", "1"), ("validation-1", "0"),
    ]
    output = json.loads((evaluation / "output.json").read_text())["payload"]
    assert output["status"] == "success"
    assert output["metrics"]["validation/ACC"] == 1.0
