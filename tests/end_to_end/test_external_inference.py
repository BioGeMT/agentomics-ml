import csv

import pytest


def test_external_inference_predicts_for_compatible_input_using_the_best_result(
    cli, completed_run,
):
    root = completed_run.parent
    input_split = root / "external-input"
    (input_split / "input").mkdir(parents=True)
    (input_split / "input" / "data.csv").write_text(
        "id,sequence\nexternal-2,TTGCA\nexternal-0,AACGT\nexternal-1,ATGCA\n",
        encoding="utf-8",
    )
    output = root / "predictions.csv"

    cli(
        ["--agent-dir", str(completed_run), "--input", str(input_split),
         "--output", str(output)],
        root,
        module="agentomics.cli.inference",
    )

    with output.open() as predictions_file:
        predictions = list(csv.DictReader(predictions_file))
    assert sorted((row["id"], row["prediction"]) for row in predictions) == [
        ("external-0", "1"), ("external-1", "1"), ("external-2", "0"),
    ]
    for row in predictions:
        probabilities = [float(row[f"probability_{label}"]) for label in (0, 1)]
        assert all(0 <= probability <= 1 for probability in probabilities)
        assert sum(probabilities) == pytest.approx(1)
        assert probabilities[int(row["prediction"])] == max(probabilities)
