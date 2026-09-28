import csv
import json

from pypdf import PdfReader


def test_run_produces_a_best_result_held_out_evaluation_and_final_report(completed_run):
    workspace = completed_run
    best = workspace / "best_iteration_snapshot"
    metadata = json.loads((best / "runtime_info" / "iteration_metadata.json").read_text())
    assert metadata["iteration"] == 0

    assert (best / "model_training" / "train.py").is_file()
    assert (best / "model_inference" / "inference.py").is_file()
    assert json.loads(
        (best / "model_training" / "training_artifacts" / "model.json").read_text()
    ) == {"A": 1, "T": 0}
    with (best / "eval_predictions_test.csv").open() as predictions_file:
        predictions = list(csv.DictReader(predictions_file))
    assert [(row["id"], row["prediction"]) for row in predictions] == [
        ("test-0", "1"), ("test-1", "0"),
    ]
    metrics = json.loads((best / "eval_predictions_test.metrics.json").read_text())
    assert metrics["ACC"] == 1.0

    report = (workspace / "reports" / "best_iteration.md").read_text()
    assert "promoter_sequences" in report
    assert "## Model Training" in report
    assert "## Test Metrics" in report
    pdf = PdfReader(workspace / "reports" / "best_iteration.pdf")
    assert "promoter_sequences" in "\n".join(page.extract_text() for page in pdf.pages)
