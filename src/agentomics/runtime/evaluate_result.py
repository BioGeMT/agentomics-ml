import argparse
import json
import sys

import pandas as pd

from agentomics.datasets.data_contract import ID_COLUMN_NAME, NUMERIC_LABEL_COLUMN_NAME, PREDICTION_COLUMN_NAME
from agentomics.utils.ml_tasks import get_task, get_task_names


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

    return get_task(task_type).score(
        merged,
        prediction_columns=results.columns.tolist(),
        numeric_label_col=numeric_label_col,
        pred_col=pred_col,
        prob_col_prefix=prob_col_prefix,
    )


def main():
    parser = argparse.ArgumentParser(description="Evaluate classification metrics for model predictions.")
    parser.add_argument("-r", "--results", required=True, help="Path to the results CSV file.")
    parser.add_argument("-t", "--test", required=True, help="Path to the CSV file with labels.")
    parser.add_argument("--pred-col", default="prediction", help="Name of the prediction column in the results file.")
    parser.add_argument("--numeric-label-col", default="numeric_label", help="Name of the numeric label column in the file.")
    parser.add_argument("--task-type", choices=sorted(get_task_names()), required=True, help="Type of task: classification or regression.")

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
