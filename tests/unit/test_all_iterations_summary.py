from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from agentomics.runtime import all_iterations_summary
from agentomics.runtime.report_data import IterationInputs, RunMeta, SplitArtifacts


def _write_evaluation_files(directory, ids):
    directory.mkdir(parents=True, exist_ok=True)
    labels_path = directory / "labels.csv"
    predictions_path = directory / "predictions.csv"
    pd.DataFrame({"id": ids, "numeric_label": [0] * len(ids)}).to_csv(
        labels_path, index=False
    )
    pd.DataFrame(
        {"id": ids, "probability_1": [0.5] * len(ids)}
    ).to_csv(predictions_path, index=False)
    return labels_path, predictions_path


def test_all_iteration_reports_summarize_comparable_scores_and_best_test_results(
    initialized_run_config, monkeypatch
):
    config = initialized_run_config
    config.val_metric = "AUROC"
    iteration_inputs = {}

    for iteration, score in enumerate([0.8, 0.85, 0.82]):
        labels, predictions = _write_evaluation_files(
            Path(config.workspace_dir) / f"iteration_{iteration}",
            [f"row-{iteration}-0", f"row-{iteration}-1"],
        )
        splits = [
            SplitArtifacts("train", labels, predictions, {"AUROC": score + 0.05}),
            SplitArtifacts(
                "validation", labels, predictions, {"AUROC": score}
            )
        ]
        if iteration == 2:
            test_labels, test_predictions = _write_evaluation_files(
                Path(config.workspace_dir) / "test", ["test-row"]
            )
            splits.append(
                SplitArtifacts("test", test_labels, test_predictions, {"ACC": 1.0})
            )
        iteration_inputs[iteration] = IterationInputs(iteration, None, splits)

    monkeypatch.setattr(
        all_iterations_summary,
        "gather_iteration_inputs",
        lambda _config, iteration: iteration_inputs[iteration],
    )
    monkeypatch.setattr(
        all_iterations_summary,
        "load_step_output",
        lambda _config, _step, iteration_dir: SimpleNamespace(
            split_version={"iteration_0": 0, "iteration_1": 0, "iteration_2": 1}[
                iteration_dir.name
            ]
        ),
    )
    monkeypatch.setattr(
        all_iterations_summary,
        "_iteration_approach",
        lambda _config, iteration: f"approach {iteration}",
    )

    all_iterations_summary.write_all_iterations_summary(
        config,
        RunMeta(
            agent_id=config.agent_id,
            model_name=config.model_name,
            dataset=config.dataset,
            task_type="classification",
            val_metric="AUROC",
            split_allowed_iterations=0,
            exploration_iterations=0,
        ),
        iterations=[0, 1, 2],
        best_iteration=2,
    )

    summary = (config.reports_dir / "all_iterations.md").read_text(encoding="utf-8")
    assert "| 0 | approach 0 | split_0 | 0.8 | 2 | — |  |" in summary
    assert "+0.05 (better)" in summary
    assert "— (split changed)" in summary
    assert (
        "| 2 | approach 2 | split_1 | 0.82 | 2 | — (split changed) | Yes |"
        in summary
    )
    assert "test (n=1)" in summary
    assert "| ACC | 1 |" in summary
    assert "Train and validation metrics across iterations" in summary
    assert "pdf/plots/all_iterations_metrics.png" in summary
    assert (config.reports_dir / "all_iterations.pdf").stat().st_size > 0
    assert (
        config.pdf_reports_dir / "plots" / "all_iterations_metrics.png"
    ).stat().st_size > 0
