import argparse
import json
import sys

import pandas as pd

from datasets.data_contract import ID_COLUMN_NAME, NUMERIC_LABEL_COLUMN_NAME, PREDICTION_COLUMN_NAME
from utils.metrics import get_classification_metrics_functions, get_regression_metrics_functions
from utils.task_types import TaskTypes


def get_metrics(
    results_file,
    test_file,
    task_type,
    numeric_label_col=NUMERIC_LABEL_COLUMN_NAME,
    pred_col=PREDICTION_COLUMN_NAME,
    prob_col_prefix='probability_',
):
    results = pd.read_csv(results_file, dtype={ID_COLUMN_NAME: str}, keep_default_na=False)
    test = pd.read_csv(test_file, dtype={ID_COLUMN_NAME: str}, keep_default_na=False)
    if pred_col == numeric_label_col:
        test = test.rename(columns={numeric_label_col: f"{numeric_label_col}_to_predict"})
        numeric_label_col = f"{numeric_label_col}_to_predict"

    merged = pd.merge(results, test, on='id', how='inner')
    if len(merged) != len(test):
        print('WARNING: PREDICTIONS LENGTH DOESNT MATCH TEST DATASET')

    if task_type == TaskTypes.CLASSIFICATION:
        return _get_classification_metrics(
            results=results,
            merged=merged,
            numeric_label_col=numeric_label_col,
            pred_col=pred_col,
            prob_col_prefix=prob_col_prefix,
        )
    if task_type == TaskTypes.REGRESSION:
        return _get_regression_metrics(
            merged=merged,
            numeric_label_col=numeric_label_col,
            pred_col=pred_col,
        )
    raise ValueError(f"Unknown task_type: {task_type}. Expected one of {TaskTypes}.")


def _get_classification_metrics(results, merged, numeric_label_col: str, pred_col: str, prob_col_prefix: str):
    merged[pred_col] = merged[pred_col].astype(int)
    merged[numeric_label_col] = merged[numeric_label_col].astype(int)

    prob_cols = [col for col in results.columns if col.startswith(prob_col_prefix)]
    if not prob_cols:
        raise ValueError(f"No probability columns found with prefix '{prob_col_prefix}'.")
    prob_cols_sorted = sorted(prob_cols, key=lambda column_name: int(column_name.split('_')[1]))

    metrics = {}
    metric_to_fn = get_classification_metrics_functions()
    for metric_name, metric_fn in metric_to_fn.items():
        if metric_fn.needs_probabilities:
            metrics[metric_name] = metric_fn(
                merged[numeric_label_col],
                merged[prob_cols_sorted].astype(float).values,
            )
        else:
            metrics[metric_name] = metric_fn(merged[numeric_label_col], merged[pred_col])
    return metrics


def _get_regression_metrics(merged, numeric_label_col: str, pred_col: str):
    merged[numeric_label_col] = merged[numeric_label_col].astype(float)
    merged[pred_col] = merged[pred_col].astype(float)

    metrics = {}
    metric_to_fn = get_regression_metrics_functions()
    for metric_name, metric_fn in metric_to_fn.items():
        metrics[metric_name] = metric_fn(merged[numeric_label_col], merged[pred_col])
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate classification metrics for model predictions.")
    parser.add_argument("-r", "--results", required=True, help="Path to the results CSV file.")
    parser.add_argument("-t", "--test", required=True, help="Path to the CSV file with labels.")
    parser.add_argument("--pred-col", default="prediction", help="Name of the prediction column in the results file.")
    parser.add_argument("--numeric-label-col", default="numeric_label", help="Name of the numeric label column in the file.")
    parser.add_argument("--task-type", choices=sorted(TaskTypes), required=True, help="Type of task: classification or regression.")

    args = parser.parse_args()

    try:
        metrics = get_metrics(
            results_file=args.results,
            test_file=args.test,
            pred_col=args.pred_col,
            numeric_label_col=args.numeric_label_col,
            task_type=args.task_type,
        )
        print(json.dumps(metrics, indent=2))
    except Exception as e:
        sys.exit(f"Error: {e}")

if __name__ == '__main__':
    main()
