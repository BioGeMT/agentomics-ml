import csv
from pathlib import Path

import pytest

from tests.dataset_helpers import create_classification_dataset
from tests.support.workspace import snapshot_workspace


def test_retraining_produces_usable_retrained_artifacts_without_changing_original_run(
    cli, completed_run: Path,
):
    root = completed_run.parent
    dataset = create_classification_dataset(root, include_validation_split=True)
    # Reverse the original labels: using original artifacts must now give wrong predictions.
    for split in ("train", "validation"):
        (dataset / split / "labels.csv").write_text(
            f"id,label\n{split}-0,negative\n{split}-1,positive\n",
            encoding="utf-8",
        )
    # Move the saved iteration so falling back to the default cannot pass.
    iteration = completed_run / "selected_iteration"
    (completed_run / "best_iteration_snapshot").rename(iteration)
    artifacts = root / "retrained-artifacts"
    original = snapshot_workspace(completed_run)

    cli(
        ["--agent-dir", str(completed_run), "--dataset-dir", str(dataset),
         "--iteration-dir", iteration.name, "--artifacts-dir", str(artifacts)],
        root,
        module="agentomics.cli.retrain",
    )

    assert snapshot_workspace(completed_run) == original, "Retraining changed the original run"
    assert (artifacts / "model.json").read_bytes() != (
        iteration / "model_training" / "training_artifacts" / "model.json"
    ).read_bytes(), "Retraining reused the original artifacts"

    input_split = root / "inference-input"
    input_dir = input_split / "input"
    input_dir.mkdir(parents=True)
    (input_dir / "data.csv").write_text(
        "id,sequence\nretrained-2,TTGCA\nretrained-0,AACGT\nretrained-1,ATGCA\n",
        encoding="utf-8",
    )
    output_dir = root / "inference-output"
    output_dir.mkdir()
    output = output_dir / "predictions.csv"
    retrained = snapshot_workspace(artifacts)
    cli(
        ["--agent-dir", str(completed_run), "--iteration-dir", iteration.name,
         "--artifacts-dir", artifacts.name, "--input", str(input_split),
         "--output", str(output)],
        root,
        module="agentomics.cli.inference",
    )

    assert snapshot_workspace(completed_run) == original, "Inference changed the original run"
    assert snapshot_workspace(artifacts) == retrained, "Inference changed Retrained Artifacts"

    with output.open() as stream:
        predictions = list(csv.DictReader(stream))
    assert sorted((row["id"], row["prediction"]) for row in predictions) == [
        ("retrained-0", "0"), ("retrained-1", "0"), ("retrained-2", "1"),
    ], "Inference did not use Retrained Artifacts"
    for row in predictions:
        probabilities = [float(row[f"probability_{label}"]) for label in (0, 1)]
        assert all(0 <= probability <= 1 for probability in probabilities)
        assert sum(probabilities) == pytest.approx(1)
        assert probabilities[int(row["prediction"])] == max(probabilities)
